//! SIMD-accelerated inference engine
//!
//! Provides high-performance transformer inference using trueno's SIMD primitives.
//! Designed to compete with llama.cpp on CPU performance.
//!
//! ## Architecture
//!
//! ```text
//! GGUF Model → GGUFTransformer → TruenoInferenceEngine → Tokens
//! ```
//!
//! ## Performance Targets
//!
//! - Use trueno's SIMD Vector::dot for all dot products
//! - Use trueno's Matrix::matmul for weight projections
//! - Target >100 tokens/sec on CPU for 1B models

use crate::error::{RealizarError, Result};
use crate::gguf::{GGUFConfig, GGUFTransformer};
use crate::quantize::{fused_q4k_tiled_matvec, QK_K};
use rayon::prelude::*;
use trueno::{Matrix, Vector};

/// Tile size for cache-efficient tiled matmul
const TILE_SIZE: usize = 64;

// ============================================================================
// DYNAMIC THREAD ALLOCATION (per llama.cpp)
// ============================================================================
//
// llama.cpp uses different thread counts for prefill vs decode:
// - Prefill (batch): Use all cores for maximum throughput
// - Decode (single token): Use fewer cores to reduce cache thrashing
// ============================================================================

/// Thread configuration for dynamic thread allocation
///
/// Per llama.cpp: batch processing uses more threads than single-token decode.
/// This reduces cache thrashing during decode phase.
#[derive(Debug, Clone, Copy)]
pub struct ThreadConfig {
    /// Threads for batch/prefill operations (uses all cores)
    pub n_threads_batch: usize,
    /// Threads for single-token decode (uses fewer cores)
    pub n_threads_decode: usize,
}

impl ThreadConfig {
    /// Create optimal thread config based on available cores
    ///
    /// - Batch: Uses all available cores
    /// - Decode: Uses half cores (min 1) to reduce cache contention
    pub fn auto() -> Self {
        let num_cpus = rayon::current_num_threads();
        Self {
            n_threads_batch: num_cpus,
            n_threads_decode: (num_cpus / 2).max(1),
        }
    }

    /// Create with explicit thread counts
    pub fn new(n_threads_batch: usize, n_threads_decode: usize) -> Self {
        Self {
            n_threads_batch: n_threads_batch.max(1),
            n_threads_decode: n_threads_decode.max(1),
        }
    }

    /// Get the number of threads for the current operation mode
    pub fn threads_for(&self, is_prefill: bool) -> usize {
        if is_prefill {
            self.n_threads_batch
        } else {
            self.n_threads_decode
        }
    }
}

impl Default for ThreadConfig {
    fn default() -> Self {
        Self::auto()
    }
}

/// Execution mode for controlling parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    /// Prefill/prompt processing - use maximum threads
    Prefill,
    /// Single-token decode - use fewer threads to reduce cache thrashing
    Decode,
}

/// Configure the global rayon thread pool for inference
///
/// NOTE: This should be called once at startup. Rayon's global pool cannot
/// be resized dynamically. For per-operation thread control, use scoped thread pools.
///
/// # Errors
///
/// Returns `InvalidConfiguration` if the thread pool has already been initialized.
pub fn configure_thread_pool(num_threads: usize) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| {
            RealizarError::InvalidConfiguration(format!("Failed to configure thread pool: {e}"))
        })
}

// ============================================================================
// QUANTIZED WEIGHT STORAGE (Phase 3: Memory Bandwidth Optimization)
// ============================================================================
//
// Keeps weights in quantized format for 8x memory bandwidth reduction.
// Uses fused dequant+dot operations during inference.
// ============================================================================

/// Quantized weight matrix stored in Q4_K format
///
/// Stores weights in compressed form (4.5 bits/weight) and uses
/// fused dequant+dot operations during inference for 8x bandwidth reduction.
#[derive(Clone)]
pub struct Q4KWeight {
    /// Raw Q4_K data (144 bytes per 256 values)
    pub data: Vec<u8>,
    /// Input dimension (columns)
    pub in_dim: usize,
    /// Output dimension (rows)
    pub out_dim: usize,
}

impl Q4KWeight {
    /// Create a new quantized weight from raw Q4_K data
    ///
    /// # Arguments
    ///
    /// * `data` - Raw Q4_K bytes (144 bytes per 256 values)
    /// * `in_dim` - Input dimension (must be multiple of 256)
    /// * `out_dim` - Output dimension
    ///
    /// # Errors
    ///
    /// Returns error if data size doesn't match dimensions
    pub fn new(data: Vec<u8>, in_dim: usize, out_dim: usize) -> Result<Self> {
        let super_blocks_per_row = in_dim.div_ceil(QK_K);
        let bytes_per_row = super_blocks_per_row * 144;
        let expected_size = out_dim * bytes_per_row;

        if data.len() < expected_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Q4K weight data too small: got {} bytes, expected {} for {}x{}",
                    data.len(),
                    expected_size,
                    out_dim,
                    in_dim
                ),
            });
        }

        Ok(Self {
            data,
            in_dim,
            out_dim,
        })
    }

    /// Perform matrix-vector multiplication using fused SIMD operations
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector of length `in_dim`
    ///
    /// # Returns
    ///
    /// Output vector of length `out_dim`
    ///
    /// # Errors
    ///
    /// Returns error if input length doesn't match `in_dim`
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input length {} doesn't match weight in_dim {}",
                    input.len(),
                    self.in_dim
                ),
            });
        }

        fused_q4k_tiled_matvec(&self.data, input, self.in_dim, self.out_dim, None)
    }

    /// Memory usage in bytes (compressed)
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Equivalent f32 memory usage (for comparison)
    #[must_use]
    pub fn f32_equivalent_bytes(&self) -> usize {
        self.in_dim * self.out_dim * 4
    }

    /// Compression ratio vs f32
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.f32_equivalent_bytes() as f32 / self.memory_bytes() as f32
    }
}

/// Threshold for switching to parallel matmul (number of output elements)
const PARALLEL_THRESHOLD: usize = 256;

/// SIMD-accelerated matmul using trueno with automatic parallelization
///
/// # Arguments
///
/// * `input` - Input tensor [seq_len, in_dim]
/// * `weight` - Weight matrix [out_dim, in_dim] (row-major, transposed for matmul)
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension
///
/// # Returns
///
/// Output tensor [seq_len, out_dim]
pub fn simd_matmul(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let seq_len = input.len() / in_dim;

    // For single-token inference (most common case), use parallel vector dot products
    if seq_len == 1 {
        // Use parallel for large output dimensions
        if out_dim >= PARALLEL_THRESHOLD {
            return parallel_matmul_single(input, weight, in_dim, out_dim);
        }

        // Sequential for small output dimensions (parallelism overhead not worth it)
        let input_vec = Vector::from_slice(input);
        let mut output = Vec::with_capacity(out_dim);

        for o in 0..out_dim {
            let weight_row = &weight[o * in_dim..(o + 1) * in_dim];
            let weight_vec = Vector::from_slice(weight_row);
            let dot = input_vec.dot(&weight_vec).unwrap_or(0.0);
            output.push(dot);
        }
        return output;
    }

    // For batch inference, use tiled matrix multiplication for cache efficiency
    if seq_len * out_dim >= PARALLEL_THRESHOLD * 4 {
        return tiled_matmul(input, weight, seq_len, in_dim, out_dim);
    }

    // For small batches, use trueno's matrix multiplication
    let input_matrix =
        Matrix::from_vec(seq_len, in_dim, input.to_vec()).expect("Valid input matrix");
    let weight_matrix =
        Matrix::from_vec(out_dim, in_dim, weight.to_vec()).expect("Valid weight matrix");

    let weight_t = weight_matrix.transpose();
    let result = input_matrix
        .matmul(&weight_t)
        .expect("Matrix multiplication should succeed");

    result.as_slice().to_vec()
}

/// Parallel matmul for single-token inference
///
/// Uses rayon to parallelize across output dimensions with SIMD dot products.
/// This is the hot path for autoregressive generation.
fn parallel_matmul_single(
    input: &[f32],
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    (0..out_dim)
        .into_par_iter()
        .map(|o| {
            let weight_row = &weight[o * in_dim..(o + 1) * in_dim];
            let input_vec = Vector::from_slice(input);
            let weight_vec = Vector::from_slice(weight_row);
            input_vec.dot(&weight_vec).unwrap_or(0.0)
        })
        .collect()
}

/// Tiled matmul for cache-efficient batch inference
///
/// Computes C = A × B^T using cache-blocking to maximize L1/L2 cache hits.
/// This is critical for prompt processing where seq_len > 1.
fn tiled_matmul(
    input: &[f32],  // [M, K]
    weight: &[f32], // [N, K] (stored row-major, needs transpose)
    m: usize,       // seq_len
    k: usize,       // in_dim
    n: usize,       // out_dim
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];

    // Process in tiles for better cache utilization
    // Parallelize across output row tiles
    output
        .par_chunks_mut(TILE_SIZE * n)
        .enumerate()
        .for_each(|(tile_i, out_chunk)| {
            let i_start = tile_i * TILE_SIZE;
            let i_end = (i_start + TILE_SIZE).min(m);
            let rows_in_tile = i_end - i_start;

            // Process each row in this tile
            for (local_i, out_row) in out_chunk.chunks_mut(n).take(rows_in_tile).enumerate() {
                let i = i_start + local_i;
                let input_row = &input[i * k..(i + 1) * k];
                let input_vec = Vector::from_slice(input_row);

                // Process output columns in tiles
                for j_tile in (0..n).step_by(TILE_SIZE) {
                    let j_end = (j_tile + TILE_SIZE).min(n);

                    for j in j_tile..j_end {
                        let weight_row = &weight[j * k..(j + 1) * k];
                        let weight_vec = Vector::from_slice(weight_row);
                        out_row[j] = input_vec.dot(&weight_vec).unwrap_or(0.0);
                    }
                }
            }
        });

    output
}

/// SIMD-accelerated vector dot product using trueno
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);
    va.dot(&vb).unwrap_or(0.0)
}

/// SIMD-accelerated vector add using trueno
pub fn simd_add(a: &mut [f32], b: &[f32]) {
    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);
    if let Ok(result) = va.add(&vb) {
        a.copy_from_slice(result.as_slice());
    }
}

/// SIMD-accelerated element-wise multiplication
pub fn simd_mul(a: &mut [f32], b: &[f32]) {
    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);
    if let Ok(result) = va.mul(&vb) {
        a.copy_from_slice(result.as_slice());
    }
}

/// SIMD-accelerated SiLU (Swish) activation: x * sigmoid(x)
pub fn simd_silu(data: &mut [f32]) {
    for x in data.iter_mut() {
        let x_val = *x;
        let sigmoid = 1.0 / (1.0 + (-x_val).exp());
        *x = x_val * sigmoid;
    }
}

/// SIMD-accelerated GELU activation using trueno
pub fn simd_gelu(data: &mut [f32]) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const C: f32 = 0.044_715;

    for x in data.iter_mut() {
        let x_val = *x;
        let inner = SQRT_2_OVER_PI * (x_val + C * x_val * x_val * x_val);
        *x = 0.5 * x_val * (1.0 + inner.tanh());
    }
}

/// SIMD-accelerated softmax
pub fn simd_softmax(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for x in data.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for x in data.iter_mut() {
        *x *= inv_sum;
    }
}

/// KV Cache for efficient autoregressive generation
///
/// Stores key and value projections for all layers to avoid recomputation
/// during token-by-token generation.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension (per head * num_heads)
    hidden_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Current sequence length (tokens processed so far)
    seq_len: usize,
    /// Key cache: [num_layers, max_seq_len, hidden_dim]
    k_cache: Vec<f32>,
    /// Value cache: [num_layers, max_seq_len, hidden_dim]
    v_cache: Vec<f32>,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let cache_size = num_layers * max_seq_len * hidden_dim;
        Self {
            num_layers,
            hidden_dim,
            max_seq_len,
            seq_len: 0,
            k_cache: vec![0.0; cache_size],
            v_cache: vec![0.0; cache_size],
        }
    }

    /// Store K and V for a layer at the current position
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if self.seq_len >= self.max_seq_len {
            return; // Cache full
        }

        let offset = (layer * self.max_seq_len + self.seq_len) * self.hidden_dim;
        let k_len = k.len().min(self.hidden_dim);
        let v_len = v.len().min(self.hidden_dim);

        self.k_cache[offset..offset + k_len].copy_from_slice(&k[..k_len]);
        self.v_cache[offset..offset + v_len].copy_from_slice(&v[..v_len]);
    }

    /// Advance sequence position after storing all layers
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// Get all cached K values for a layer up to current position
    pub fn get_k(&self, layer: usize) -> &[f32] {
        let start = layer * self.max_seq_len * self.hidden_dim;
        let end = start + self.seq_len * self.hidden_dim;
        &self.k_cache[start..end]
    }

    /// Get all cached V values for a layer up to current position
    pub fn get_v(&self, layer: usize) -> &[f32] {
        let start = layer * self.max_seq_len * self.hidden_dim;
        let end = start + self.seq_len * self.hidden_dim;
        &self.v_cache[start..end]
    }

    /// Current sequence length
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset cache for new generation
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

/// Compute scaled dot-product attention with KV cache
///
/// For autoregressive generation, Q comes from current token while K,V come from cache.
/// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
pub fn attention_with_cache(
    q: &[f32],       // [hidden_dim] - query for current position
    k_cache: &[f32], // [seq_len, hidden_dim] - cached keys
    v_cache: &[f32], // [seq_len, hidden_dim] - cached values
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let seq_len = k_cache.len() / (num_heads * head_dim);
    if seq_len == 0 {
        return q.to_vec(); // No cache, return Q
    }

    let hidden_dim = num_heads * head_dim;
    let mut output = vec![0.0f32; hidden_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Process each attention head in parallel
    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(h, out_head)| {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            // Compute attention scores: Q * K^T / sqrt(d_k)
            let mut scores = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let k_start = pos * hidden_dim + h * head_dim;
                let k_head = &k_cache[k_start..k_start + head_dim];
                let score = simd_dot(q_head, k_head) * scale;
                scores.push(score);
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            for s in &mut scores {
                *s *= inv_sum;
            }

            // Weighted sum of values
            for (pos, &score) in scores.iter().enumerate() {
                let v_start = pos * hidden_dim + h * head_dim;
                let v_head = &v_cache[v_start..v_start + head_dim];
                for (i, &v) in v_head.iter().enumerate() {
                    out_head[i] += score * v;
                }
            }
        });

    output
}

/// Optimized KV cache with transposed V storage (per llama.cpp)
///
/// Key insight from llama.cpp: Store V as [hidden_dim, seq_len] instead of [seq_len, hidden_dim].
/// This makes the weighted sum in attention contiguous per output dimension.
///
/// Memory access pattern improvement:
/// - Original: V[pos, dim] → strided access across positions
/// - Transposed: V_T[dim, pos] → contiguous access per output dim
#[derive(Debug, Clone)]
pub struct OptimizedKVCache {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension (per head * num_heads)
    pub hidden_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Current sequence length (tokens processed so far)
    seq_len: usize,
    /// Key cache: [num_layers, max_seq_len, hidden_dim] (row-major)
    k_cache: Vec<f32>,
    /// Value cache: [num_layers, hidden_dim, max_seq_len] (TRANSPOSED for cache efficiency)
    v_cache: Vec<f32>,
}

impl OptimizedKVCache {
    /// Create a new optimized KV cache with transposed V storage
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let cache_size = num_layers * max_seq_len * hidden_dim;
        Self {
            num_layers,
            hidden_dim,
            max_seq_len,
            seq_len: 0,
            k_cache: vec![0.0; cache_size],
            v_cache: vec![0.0; cache_size],
        }
    }

    /// Store K and V for a layer at the current position
    ///
    /// K is stored in row-major [seq_len, hidden_dim].
    /// V is stored transposed [hidden_dim, seq_len] for optimal weighted sum access.
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if self.seq_len >= self.max_seq_len {
            return; // Cache full
        }

        // K: store at [layer, seq_len, :]
        let k_offset = (layer * self.max_seq_len + self.seq_len) * self.hidden_dim;
        let k_len = k.len().min(self.hidden_dim);
        self.k_cache[k_offset..k_offset + k_len].copy_from_slice(&k[..k_len]);

        // V: store transposed at [layer, :, seq_len]
        let v_base = layer * self.hidden_dim * self.max_seq_len;
        let v_len = v.len().min(self.hidden_dim);
        for (dim, &val) in v.iter().enumerate().take(v_len) {
            let v_offset = v_base + dim * self.max_seq_len + self.seq_len;
            self.v_cache[v_offset] = val;
        }
    }

    /// Advance sequence position after storing all layers
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// Get all cached K values for a layer up to current position
    /// Layout: [seq_len, hidden_dim] (row-major)
    pub fn get_k(&self, layer: usize) -> &[f32] {
        let start = layer * self.max_seq_len * self.hidden_dim;
        let end = start + self.seq_len * self.hidden_dim;
        &self.k_cache[start..end]
    }

    /// Get all cached V values for a layer up to current position (TRANSPOSED)
    /// Layout: [hidden_dim, seq_len] - optimal for weighted sum
    ///
    /// Note: Returns a packed Vec because storage has max_seq_len stride per dim.
    /// For each dimension, positions 0..seq_len are extracted and packed contiguously.
    pub fn get_v_transposed(&self, layer: usize) -> Vec<f32> {
        let v_base = layer * self.hidden_dim * self.max_seq_len;
        let mut result = Vec::with_capacity(self.hidden_dim * self.seq_len);

        // Pack V: for each dimension, copy only the valid positions
        for dim in 0..self.hidden_dim {
            let dim_start = v_base + dim * self.max_seq_len;
            result.extend_from_slice(&self.v_cache[dim_start..dim_start + self.seq_len]);
        }

        result
    }

    /// Get raw V cache slice for a layer (includes padding) - for direct access
    /// Layout: [hidden_dim, max_seq_len] where only [:, 0..seq_len] is valid
    pub fn get_v_raw(&self, layer: usize) -> &[f32] {
        let start = layer * self.hidden_dim * self.max_seq_len;
        let end = start + self.hidden_dim * self.max_seq_len;
        &self.v_cache[start..end]
    }

    /// Get the max sequence length (stride for raw V access)
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Current sequence length
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset cache for new generation
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

/// Compute scaled dot-product attention with transposed V cache
///
/// Optimized for cache efficiency: V is stored as [hidden_dim, seq_len].
/// The weighted sum loop now has contiguous memory access per output dimension.
pub fn attention_with_transposed_v(
    q: &[f32],            // [hidden_dim] - query for current position
    k_cache: &[f32],      // [seq_len, hidden_dim] - cached keys (row-major)
    v_transposed: &[f32], // [hidden_dim, seq_len] - cached values (TRANSPOSED)
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    if seq_len == 0 {
        return q.to_vec(); // No cache, return Q
    }

    let hidden_dim = num_heads * head_dim;
    let mut output = vec![0.0f32; hidden_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Process each attention head in parallel
    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(h, out_head)| {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            // Compute attention scores: Q * K^T / sqrt(d_k)
            let mut scores = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let k_start = pos * hidden_dim + h * head_dim;
                let k_head = &k_cache[k_start..k_start + head_dim];
                let score = simd_dot(q_head, k_head) * scale;
                scores.push(score);
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            for s in &mut scores {
                *s *= inv_sum;
            }

            // Weighted sum of values - NOW CONTIGUOUS per output dim!
            // V_T layout: [hidden_dim, seq_len]
            // For output dim i, V_T[h*head_dim + i, :] contains all positions
            for (i, out_val) in out_head.iter_mut().enumerate() {
                let dim_idx = h * head_dim + i;
                let v_row_start = dim_idx * seq_len;
                let v_row = &v_transposed[v_row_start..v_row_start + seq_len];

                // SIMD-friendly: contiguous access to scores and V values
                let mut acc = 0.0f32;
                for (pos, &score) in scores.iter().enumerate() {
                    acc += score * v_row[pos];
                }
                *out_val = acc;
            }
        });

    output
}

/// SIMD-accelerated layer normalization
pub fn simd_layer_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let x = &input[start..end];

        // Compute mean using trueno
        let x_vec = Vector::from_slice(x);
        let mean = x_vec.sum().unwrap_or(0.0) / hidden_dim as f32;

        // Compute variance
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / hidden_dim as f32;

        // Normalize and apply affine transform
        let inv_std = (var + eps).sqrt().recip();
        for j in 0..hidden_dim {
            let normalized = (x[j] - mean) * inv_std;
            let scaled = normalized * weight[j];
            let out = if let Some(b) = bias {
                scaled + b[j]
            } else {
                scaled
            };
            output.push(out);
        }
    }

    output
}

/// SIMD-accelerated RMS layer normalization (for LLaMA-style models)
///
/// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
/// Unlike LayerNorm, RMSNorm does not center the input (no mean subtraction).
/// This is used by LLaMA, TinyLlama, Mistral, etc.
pub fn simd_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let x = &input[start..end];

        // Compute RMS (root mean square)
        let x_vec = Vector::from_slice(x);
        let sum_sq = x_vec.dot(&x_vec).unwrap_or(0.0);
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
        let inv_rms = rms.recip();

        // Normalize and apply affine transform (no bias in RMSNorm)
        for j in 0..hidden_dim {
            let normalized = x[j] * inv_rms;
            output.push(normalized * weight[j]);
        }
    }

    output
}

/// RoPE (Rotary Position Embedding) application
///
/// Applies rotary position embeddings to query and key vectors.
/// This is essential for proper attention in LLaMA-style models.
pub fn apply_rope(x: &mut [f32], hidden_dim: usize, num_heads: usize, position: usize, theta: f32) {
    let head_dim = hidden_dim / num_heads;
    let half_dim = head_dim / 2;

    for h in 0..num_heads {
        let head_start = h * head_dim;

        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx1 = head_start + i;
            let idx2 = head_start + i + half_dim;

            if idx2 < x.len() {
                let x1 = x[idx1];
                let x2 = x[idx2];
                x[idx1] = x1 * cos_val - x2 * sin_val;
                x[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

/// SIMD-accelerated inference engine
///
/// Wraps a GGUF transformer with trueno SIMD acceleration.
pub struct TruenoInferenceEngine {
    /// Model configuration
    pub config: GGUFConfig,
    /// Token embeddings [vocab_size, hidden_dim]
    token_embedding: Vec<f32>,
    /// Per-layer weights
    layers: Vec<TruenoTransformerLayer>,
    /// Output norm weight
    output_norm_weight: Vec<f32>,
    /// Output norm bias
    output_norm_bias: Option<Vec<f32>>,
    /// LM head weight [vocab_size, hidden_dim]
    lm_head_weight: Vec<f32>,
    /// LM head bias
    lm_head_bias: Option<Vec<f32>>,
    /// Use RMSNorm (LLaMA/Mistral) vs LayerNorm (phi-2)
    use_rms_norm: bool,
}

/// Weights for a single transformer layer (optimized for trueno)
struct TruenoTransformerLayer {
    /// Attention norm weight
    attn_norm_weight: Vec<f32>,
    /// Attention norm bias
    attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weight [3*hidden_dim, hidden_dim]
    qkv_weight: Vec<f32>,
    /// QKV bias
    qkv_bias: Option<Vec<f32>>,
    /// Attention output weight [hidden_dim, hidden_dim]
    attn_output_weight: Vec<f32>,
    /// Attention output bias
    attn_output_bias: Option<Vec<f32>>,
    /// FFN gate projection for SwiGLU [intermediate_dim, hidden_dim]
    ffn_gate_weight: Option<Vec<f32>>,
    /// FFN gate bias
    ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection [intermediate_dim, hidden_dim]
    ffn_up_weight: Vec<f32>,
    /// FFN up bias
    ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection [hidden_dim, intermediate_dim]
    ffn_down_weight: Vec<f32>,
    /// FFN down bias
    ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (for models with separate FFN normalization)
    ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias
    ffn_norm_bias: Option<Vec<f32>>,
}

impl TruenoInferenceEngine {
    /// Create inference engine from GGUF transformer
    pub fn from_gguf_transformer(transformer: GGUFTransformer) -> Self {
        // Detect if model uses RMSNorm (LLaMA/Mistral) vs LayerNorm (phi-2)
        let use_rms_norm = transformer
            .config
            .architecture
            .to_lowercase()
            .contains("llama")
            || transformer
                .config
                .architecture
                .to_lowercase()
                .contains("mistral")
            || transformer
                .config
                .architecture
                .to_lowercase()
                .contains("qwen");

        let layers = transformer
            .layers
            .into_iter()
            .map(|l| TruenoTransformerLayer {
                attn_norm_weight: l.attn_norm_weight,
                attn_norm_bias: l.attn_norm_bias,
                qkv_weight: l.qkv_weight,
                qkv_bias: l.qkv_bias,
                attn_output_weight: l.attn_output_weight,
                attn_output_bias: l.attn_output_bias,
                ffn_gate_weight: l.ffn_gate_weight,
                ffn_gate_bias: l.ffn_gate_bias,
                ffn_up_weight: l.ffn_up_weight,
                ffn_up_bias: l.ffn_up_bias,
                ffn_down_weight: l.ffn_down_weight,
                ffn_down_bias: l.ffn_down_bias,
                ffn_norm_weight: l.ffn_norm_weight,
                ffn_norm_bias: l.ffn_norm_bias,
            })
            .collect();

        Self {
            config: transformer.config,
            token_embedding: transformer.token_embedding,
            layers,
            output_norm_weight: transformer.output_norm_weight,
            output_norm_bias: transformer.output_norm_bias,
            lm_head_weight: transformer.lm_head_weight,
            lm_head_bias: transformer.lm_head_bias,
            use_rms_norm,
        }
    }

    /// Embed token IDs
    fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// Forward pass using SIMD-accelerated operations
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns error if token lookup or tensor operations fail
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let num_heads = self.config.num_heads;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);
        let seq_len = token_ids.len();

        // 2. Process through transformer layers
        for layer in &self.layers {
            // 2a. Attention layer norm (RMSNorm for LLaMA, LayerNorm for phi-2)
            let normed = if self.use_rms_norm {
                simd_rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                simd_layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };

            // 2b. QKV projection (SIMD matmul)
            // Compute qkv_dim from actual weight size to support GQA models
            // For MHA: qkv_dim = 3 * hidden_dim
            // For GQA: qkv_dim = hidden_dim + 2 * kv_dim (where kv_dim = num_kv_heads * head_dim)
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = simd_matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                simd_add(&mut qkv, bias);
            }

            // Calculate K/V dimension for GQA support
            // Q always has hidden_dim, K and V share the remaining dimensions
            let kv_dim = (qkv_dim - hidden_dim) / 2;
            let num_kv_heads = self.config.num_kv_heads;
            let head_dim = hidden_dim / num_heads;

            // 2c. Scaled dot-product attention with RoPE and causal masking
            // Build K/V cache for self-attention within this forward pass
            let mut k_cache = Vec::with_capacity(seq_len * hidden_dim);
            let mut v_cache = Vec::with_capacity(seq_len * hidden_dim);
            let mut attn_out = Vec::with_capacity(seq_len * hidden_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V for this position
                // Q: [hidden_dim], K: [kv_dim], V: [kv_dim]
                let mut q = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let k_raw = &qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim];
                let v_raw = &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + qkv_dim];

                // Apply RoPE to Q and K
                apply_rope(&mut q, hidden_dim, num_heads, s, self.config.rope_theta);
                let mut k = k_raw.to_vec();
                apply_rope(&mut k, kv_dim, num_kv_heads, s, self.config.rope_theta);

                // For GQA: expand K/V by repeating each KV head to match Q heads
                // group_size = num_heads / num_kv_heads (e.g., 32/4 = 8 for TinyLlama)
                let (k_expanded, v_expanded): (Vec<f32>, Vec<f32>) = if num_kv_heads < num_heads {
                    let group_size = num_heads / num_kv_heads;
                    let expand = |raw: &[f32]| -> Vec<f32> {
                        (0..num_heads)
                            .flat_map(|h| {
                                let kv_head = h / group_size;
                                let start = kv_head * head_dim;
                                raw[start..start + head_dim].iter().copied()
                            })
                            .collect()
                    };
                    (expand(&k), expand(v_raw))
                } else {
                    (k, v_raw.to_vec())
                };

                // Add to K/V cache for causal self-attention
                k_cache.extend_from_slice(&k_expanded);
                v_cache.extend_from_slice(&v_expanded);

                // Compute attention: softmax(Q·K^T / sqrt(d_k)) · V
                // Uses causal masking - only attend to positions 0..=s
                let attn_output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);
                attn_out.extend_from_slice(&attn_output);
            }

            // 2d. Attention output projection (SIMD matmul)
            let mut attn_output =
                simd_matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim);
            if let Some(ref bias) = layer.attn_output_bias {
                simd_add(&mut attn_output, bias);
            }

            // 2e. Residual connection (SIMD add)
            simd_add(&mut hidden, &attn_output);

            // 2f. FFN (with optional FFN norm - RMSNorm for LLaMA)
            let ffn_input = if let Some(ref norm_weight) = layer.ffn_norm_weight {
                if self.use_rms_norm {
                    simd_rms_norm(&hidden, norm_weight, self.config.eps)
                } else {
                    simd_layer_norm(
                        &hidden,
                        norm_weight,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    )
                }
            } else {
                hidden.clone()
            };

            // Check for SwiGLU (gate projection) vs standard GELU FFN
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: output = down(silu(gate(x)) * up(x))
                // gate projection
                let mut gate = simd_matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim);
                if let Some(ref bias) = layer.ffn_gate_bias {
                    simd_add(&mut gate, bias);
                }
                // SiLU activation: x * sigmoid(x)
                simd_silu(&mut gate);

                // up projection
                let mut up = simd_matmul(
                    &ffn_input,
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                );
                if let Some(ref bias) = layer.ffn_up_bias {
                    simd_add(&mut up, bias);
                }

                // Element-wise multiplication: gate * up
                simd_mul(&mut gate, &up);

                // down projection
                let mut output =
                    simd_matmul(&gate, &layer.ffn_down_weight, intermediate_dim, hidden_dim);
                if let Some(ref bias) = layer.ffn_down_bias {
                    simd_add(&mut output, bias);
                }
                output
            } else {
                // Standard FFN with GELU
                let mut ffn_hidden = simd_matmul(
                    &ffn_input,
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                );
                if let Some(ref bias) = layer.ffn_up_bias {
                    simd_add(&mut ffn_hidden, bias);
                }
                simd_gelu(&mut ffn_hidden);

                let mut output = simd_matmul(
                    &ffn_hidden,
                    &layer.ffn_down_weight,
                    intermediate_dim,
                    hidden_dim,
                );
                if let Some(ref bias) = layer.ffn_down_bias {
                    simd_add(&mut output, bias);
                }
                output
            };

            // Residual connection
            simd_add(&mut hidden, &ffn_output);
        }

        // 3. Final layer norm (RMSNorm for LLaMA, LayerNorm for phi-2)
        let normed = if self.use_rms_norm {
            simd_rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            simd_layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };

        // 4. LM head projection for last token only (SIMD)
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = simd_matmul(
            last_hidden,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            simd_add(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Get the most likely next token
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails or logits are empty
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;
        Ok(max_idx as u32)
    }

    /// Generate tokens autoregressively
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `eos_token_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// Generated token IDs (including prompt)
    ///
    /// # Errors
    ///
    /// Returns error if prompt is empty or inference fails
    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let mut tokens = prompt.to_vec();

        for _ in 0..max_tokens {
            let next_token = self.predict_next(&tokens)?;

            if let Some(eos) = eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Forward pass for a single token with KV cache (O(1) per token)
    ///
    /// This is the high-performance path for autoregressive generation.
    /// Instead of reprocessing all tokens, it only processes the new token
    /// and uses cached K/V values for attention.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token to process
    /// * `position` - Position in sequence (for RoPE)
    /// * `kv_cache` - Mutable reference to KV cache
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail (dimension mismatch, allocation failure)
    pub fn forward_one_token(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup (single token)
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        let mut hidden = if end <= self.token_embedding.len() {
            self.token_embedding[start..end].to_vec()
        } else {
            vec![0.0; hidden_dim]
        };

        // 2. Process through transformer layers with KV cache
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm (RMSNorm for LLaMA, LayerNorm for phi-2)
            let normed = if self.use_rms_norm {
                simd_rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                simd_layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };

            // 2b. QKV projection for single position
            // Compute qkv_dim from actual weight size to support GQA models
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = simd_matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                simd_add(&mut qkv, bias);
            }

            // Calculate K/V dimension for GQA support
            let kv_dim = (qkv_dim - hidden_dim) / 2;
            let num_kv_heads = self.config.num_kv_heads;

            // Extract Q, K, V with GQA-aware dimensions
            let mut q = qkv[0..hidden_dim].to_vec();
            let k_raw = &qkv[hidden_dim..hidden_dim + kv_dim];
            let v_raw = &qkv[hidden_dim + kv_dim..qkv_dim];

            // Apply RoPE to Q and K
            apply_rope(
                &mut q,
                hidden_dim,
                num_heads,
                position,
                self.config.rope_theta,
            );
            let mut k = k_raw.to_vec();
            apply_rope(
                &mut k,
                kv_dim,
                num_kv_heads,
                position,
                self.config.rope_theta,
            );

            // For GQA: expand K/V for cache storage (full size for compatibility)
            let (k_expanded, v_expanded): (Vec<f32>, Vec<f32>) = if num_kv_heads < num_heads {
                let group_size = num_heads / num_kv_heads;
                let expand = |raw: &[f32]| -> Vec<f32> {
                    (0..num_heads)
                        .flat_map(|h| {
                            let kv_head = h / group_size;
                            let start = kv_head * head_dim;
                            raw[start..start + head_dim].iter().copied()
                        })
                        .collect()
                };
                (expand(&k), expand(v_raw))
            } else {
                (k, v_raw.to_vec())
            };
            let k = k_expanded;
            let v = v_expanded;

            // Store K, V in cache for this layer
            kv_cache.store(layer_idx, &k, &v);

            // 2c. Compute attention using cached K/V
            let cached_keys = kv_cache.get_k(layer_idx);
            let cached_values = kv_cache.get_v(layer_idx);

            // Use parallel attention with cache
            let attn_out =
                attention_with_cache(&q, cached_keys, cached_values, num_heads, head_dim);

            // 2d. Attention output projection
            let mut attn_output =
                simd_matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim);
            if let Some(ref bias) = layer.attn_output_bias {
                simd_add(&mut attn_output, bias);
            }

            // 2e. Residual connection
            simd_add(&mut hidden, &attn_output);

            // 2f. FFN (with optional FFN norm - RMSNorm for LLaMA)
            let ffn_input = if let Some(ref norm_weight) = layer.ffn_norm_weight {
                if self.use_rms_norm {
                    simd_rms_norm(&hidden, norm_weight, self.config.eps)
                } else {
                    simd_layer_norm(
                        &hidden,
                        norm_weight,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    )
                }
            } else {
                hidden.clone()
            };

            // Check for SwiGLU vs standard GELU
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: output = down(silu(gate(x)) * up(x))
                let mut gate = simd_matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim);
                if let Some(ref bias) = layer.ffn_gate_bias {
                    simd_add(&mut gate, bias);
                }
                simd_silu(&mut gate);

                let mut up = simd_matmul(
                    &ffn_input,
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                );
                if let Some(ref bias) = layer.ffn_up_bias {
                    simd_add(&mut up, bias);
                }

                simd_mul(&mut gate, &up);

                let mut output =
                    simd_matmul(&gate, &layer.ffn_down_weight, intermediate_dim, hidden_dim);
                if let Some(ref bias) = layer.ffn_down_bias {
                    simd_add(&mut output, bias);
                }
                output
            } else {
                // Standard FFN with GELU
                let mut ffn_hidden = simd_matmul(
                    &ffn_input,
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                );
                if let Some(ref bias) = layer.ffn_up_bias {
                    simd_add(&mut ffn_hidden, bias);
                }
                simd_gelu(&mut ffn_hidden);

                let mut output = simd_matmul(
                    &ffn_hidden,
                    &layer.ffn_down_weight,
                    intermediate_dim,
                    hidden_dim,
                );
                if let Some(ref bias) = layer.ffn_down_bias {
                    simd_add(&mut output, bias);
                }
                output
            };

            simd_add(&mut hidden, &ffn_output);
        }

        // Advance cache position after processing all layers
        kv_cache.advance();

        // 3. Final layer norm (RMSNorm for LLaMA, LayerNorm for phi-2)
        let normed = if self.use_rms_norm {
            simd_rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            simd_layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };

        // 4. LM head projection
        let mut logits = simd_matmul(
            &normed,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            simd_add(&mut logits, bias);
        }

        Ok(logits)
    }

    /// High-performance generation with KV cache (O(n) total complexity)
    ///
    /// Uses incremental decoding to avoid recomputing attention for all tokens.
    /// This is the key optimization for competing with llama.cpp.
    ///
    /// # Performance
    ///
    /// - Prefill: O(n) for processing prompt
    /// - Decode: O(1) per generated token (vs O(n) without cache)
    /// - Total: O(n + m) where n=prompt_len, m=generated_tokens
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `eos_token_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// Generated token IDs (including prompt)
    ///
    /// # Errors
    ///
    /// Returns error if prompt is empty or inference fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_layers = self.layers.len();
        let max_seq_len = prompt.len() + max_tokens;

        // Initialize KV cache
        let mut kv_cache = KVCache::new(num_layers, hidden_dim, max_seq_len);

        // Prefill: process all prompt tokens to fill the cache
        for (pos, &token_id) in prompt.iter().enumerate() {
            let _logits = self.forward_one_token(token_id, pos, &mut kv_cache)?;
        }

        let mut tokens = prompt.to_vec();

        // Get logits from last prompt token for first generation
        let last_token = *prompt.last().expect("prompt must be non-empty");
        // Reset cache to re-process with proper attention
        kv_cache.reset();
        for (pos, &token_id) in prompt.iter().enumerate() {
            let _logits = self.forward_one_token(token_id, pos, &mut kv_cache)?;
        }

        // Decode: generate tokens one at a time using cache
        for i in 0..max_tokens {
            let position = prompt.len() + i;

            // Get last token for forward pass
            let current_token = if i == 0 {
                last_token
            } else {
                *tokens.last().expect("tokens must be non-empty")
            };

            // Skip first iteration since we already computed it in prefill
            if i > 0 {
                let logits = self.forward_one_token(current_token, position - 1, &mut kv_cache)?;

                // Greedy decoding
                let next_token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32);

                if let Some(eos) = eos_token_id {
                    if next_token == eos {
                        break;
                    }
                }

                tokens.push(next_token);
            } else {
                // First token: use logits from prefill
                let logits = self.forward_one_token(
                    *tokens.last().expect("tokens must be non-empty"),
                    position,
                    &mut kv_cache,
                )?;
                let next_token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32);

                if let Some(eos) = eos_token_id {
                    if next_token == eos {
                        break;
                    }
                }

                tokens.push(next_token);
            }
        }

        Ok(tokens)
    }

    /// Generate tokens with temperature and repetition penalty
    ///
    /// Uses proper sampling instead of greedy decoding to avoid repetitive output.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 = greedy, higher = more random)
    /// * `repetition_penalty` - Penalty for repeated tokens (1.0 = no penalty, >1.0 = penalize)
    /// * `eos_token_id` - Stop generation when this token is produced
    ///
    /// # Errors
    /// Returns error if prompt is empty or inference fails
    pub fn generate_with_sampling(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f32,
        repetition_penalty: f32,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_layers = self.layers.len();
        let max_seq_len = prompt.len() + max_tokens;

        // Initialize KV cache
        let mut kv_cache = KVCache::new(num_layers, hidden_dim, max_seq_len);

        // Prefill: process all prompt tokens to fill the cache
        for (pos, &token_id) in prompt.iter().enumerate() {
            let _logits = self.forward_one_token(token_id, pos, &mut kv_cache)?;
        }

        let mut tokens = prompt.to_vec();

        // Simple RNG based on position (deterministic but varied)
        let mut rng_state: u64 = 12345;
        let next_rng = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state as f32) / (u64::MAX as f32)
        };

        // Decode: generate tokens one at a time
        for i in 0..max_tokens {
            let position = prompt.len() + i;
            let current_token = *tokens.last().expect("tokens must be non-empty");

            let mut logits = self.forward_one_token(current_token, position, &mut kv_cache)?;

            // Apply repetition penalty
            if repetition_penalty > 1.0 {
                let window_size = 64.min(tokens.len());
                let recent_tokens = &tokens[tokens.len() - window_size..];
                for &token_id in recent_tokens {
                    let idx = token_id as usize;
                    if idx < logits.len() {
                        let logit = logits[idx];
                        logits[idx] = if logit > 0.0 {
                            logit / repetition_penalty
                        } else {
                            logit * repetition_penalty
                        };
                    }
                }
            }

            // Sample next token
            let next_token = if temperature <= 0.0 || temperature < 0.01 {
                // Greedy decoding
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Temperature-scaled sampling with top-p
                let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

                // Softmax
                let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scaled.iter().map(|&x| (x - max_logit).exp()).sum();
                let probs: Vec<f32> = scaled
                    .iter()
                    .map(|&x| (x - max_logit).exp() / exp_sum)
                    .collect();

                // Top-p (nucleus) sampling with p=0.9
                let top_p = 0.9;
                let mut indexed: Vec<(usize, f32)> =
                    probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut cumsum = 0.0;
                let mut cutoff_idx = indexed.len();
                for (i, (_, p)) in indexed.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= top_p {
                        cutoff_idx = i + 1;
                        break;
                    }
                }

                // Renormalize
                let truncated = &indexed[..cutoff_idx];
                let norm: f32 = truncated.iter().map(|(_, p)| p).sum();

                // Sample
                let r = next_rng(&mut rng_state) * norm;
                let mut acc = 0.0;
                let mut chosen = truncated[0].0;
                for &(idx, p) in truncated {
                    acc += p;
                    if acc >= r {
                        chosen = idx;
                        break;
                    }
                }
                chosen as u32
            };

            if let Some(eos) = eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

// ============================================================================
// QUANTIZED INFERENCE ENGINE (Phase 3: 8x Bandwidth Reduction)
// ============================================================================
//
// Keeps weights in Q4_K format during inference instead of dequantizing upfront.
// Uses fused dequant+dot operations for massive memory bandwidth savings.
// This is the key optimization for competing with llama.cpp on CPU.
// ============================================================================

/// Quantized transformer layer with weights in Q4_K format
///
/// Stores large weight matrices (QKV, output, FFN) in compressed form.
/// Small weights (norms, biases) remain in f32 since they're tiny.
pub struct QuantizedTransformerLayer {
    /// Attention norm weight (f32, small)
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (f32, small)
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weight [3*hidden_dim, hidden_dim] (Q4_K, large)
    pub qkv_weight: Q4KWeight,
    /// QKV bias (f32, small)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output weight [hidden_dim, hidden_dim] (Q4_K, large)
    pub attn_output_weight: Q4KWeight,
    /// Attention output bias (f32, small)
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN gate projection for SwiGLU (Q4_K, large)
    pub ffn_gate_weight: Option<Q4KWeight>,
    /// FFN gate bias (f32, small)
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection (Q4_K, large)
    pub ffn_up_weight: Q4KWeight,
    /// FFN up bias (f32, small)
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection (Q4_K, large)
    pub ffn_down_weight: Q4KWeight,
    /// FFN down bias (f32, small)
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (f32, small)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (f32, small)
    pub ffn_norm_bias: Option<Vec<f32>>,
}

/// Quantized inference engine for 8x memory bandwidth reduction
///
/// Keeps all large weight matrices in Q4_K format and uses fused
/// dequant+dot operations during inference. This dramatically reduces
/// memory bandwidth requirements, which is the main bottleneck for
/// CPU inference on modern hardware.
///
/// ## Memory Savings
///
/// For a 7B parameter model:
/// - f32 weights: 28 GB (7B × 4 bytes)
/// - Q4_K weights: 3.5 GB (7B × 0.5 bytes)
/// - 8x memory reduction!
///
/// ## Performance Impact
///
/// Memory-bound operations (most of inference):
/// - f32: Limited by 50 GB/s DDR4 bandwidth
/// - Q4_K: Effective 400 GB/s (8x bandwidth amplification)
///
/// This is why llama.cpp uses quantized inference - it's not just
/// about model size, it's about memory bandwidth.
pub struct QuantizedInferenceEngine {
    /// Model configuration
    pub config: GGUFConfig,
    /// Token embeddings [vocab_size, hidden_dim] (f32 - accessed sparsely)
    token_embedding: Vec<f32>,
    /// Per-layer quantized weights
    layers: Vec<QuantizedTransformerLayer>,
    /// Output norm weight (f32, small)
    output_norm_weight: Vec<f32>,
    /// Output norm bias (f32, small)
    output_norm_bias: Option<Vec<f32>>,
    /// LM head weight [vocab_size, hidden_dim] (Q4_K, large)
    lm_head_weight: Q4KWeight,
    /// LM head bias (f32, small)
    lm_head_bias: Option<Vec<f32>>,
}

impl QuantizedInferenceEngine {
    /// Create a quantized inference engine from raw Q4_K data
    ///
    /// This constructor takes pre-quantized weights directly without
    /// dequantization. Use this when loading from GGUF files that
    /// already contain Q4_K quantized tensors.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `token_embedding` - f32 embeddings (sparse access, not worth quantizing)
    /// * `layers` - Pre-built quantized layers
    /// * `output_norm_weight` - f32 output norm
    /// * `output_norm_bias` - Optional f32 output norm bias
    /// * `lm_head_weight` - Q4_K quantized LM head
    /// * `lm_head_bias` - Optional f32 LM head bias
    pub fn new(
        config: GGUFConfig,
        token_embedding: Vec<f32>,
        layers: Vec<QuantizedTransformerLayer>,
        output_norm_weight: Vec<f32>,
        output_norm_bias: Option<Vec<f32>>,
        lm_head_weight: Q4KWeight,
        lm_head_bias: Option<Vec<f32>>,
    ) -> Self {
        Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
        }
    }

    /// Memory usage statistics
    #[must_use]
    pub fn memory_stats(&self) -> QuantizedMemoryStats {
        let mut quantized_bytes = 0usize;
        let mut f32_equivalent_bytes = 0usize;

        // LM head
        quantized_bytes += self.lm_head_weight.memory_bytes();
        f32_equivalent_bytes += self.lm_head_weight.f32_equivalent_bytes();

        // Layers
        for layer in &self.layers {
            quantized_bytes += layer.qkv_weight.memory_bytes();
            f32_equivalent_bytes += layer.qkv_weight.f32_equivalent_bytes();

            quantized_bytes += layer.attn_output_weight.memory_bytes();
            f32_equivalent_bytes += layer.attn_output_weight.f32_equivalent_bytes();

            quantized_bytes += layer.ffn_up_weight.memory_bytes();
            f32_equivalent_bytes += layer.ffn_up_weight.f32_equivalent_bytes();

            quantized_bytes += layer.ffn_down_weight.memory_bytes();
            f32_equivalent_bytes += layer.ffn_down_weight.f32_equivalent_bytes();

            if let Some(ref gate) = layer.ffn_gate_weight {
                quantized_bytes += gate.memory_bytes();
                f32_equivalent_bytes += gate.f32_equivalent_bytes();
            }
        }

        // Token embeddings (f32)
        let embedding_bytes = self.token_embedding.len() * 4;

        QuantizedMemoryStats {
            quantized_weight_bytes: quantized_bytes,
            f32_equivalent_bytes,
            embedding_bytes,
            compression_ratio: f32_equivalent_bytes as f32 / quantized_bytes as f32,
        }
    }

    /// Embed token IDs
    fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// Forward pass using fused quantized operations
    ///
    /// Uses fused dequant+dot for all large matrix operations,
    /// providing 8x memory bandwidth reduction compared to f32.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);
        let seq_len = token_ids.len();

        // 2. Process through transformer layers
        for layer in &self.layers {
            // 2a. Attention layer norm (f32 - small)
            let normed = simd_layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection (QUANTIZED - fused dequant+dot)
            // For single token, use matvec. For batch, would need batched version.
            // Use actual weight dimensions to support GQA models
            let qkv_dim = layer.qkv_weight.out_dim;
            let mut qkv = if seq_len == 1 {
                // Single token: use fused quantized matvec
                layer.qkv_weight.matvec(&normed)?
            } else {
                // Batch: process each position separately
                let mut batch_qkv = Vec::with_capacity(seq_len * qkv_dim);
                for s in 0..seq_len {
                    let pos_input = &normed[s * hidden_dim..(s + 1) * hidden_dim];
                    let pos_qkv = layer.qkv_weight.matvec(pos_input)?;
                    batch_qkv.extend(pos_qkv);
                }
                batch_qkv
            };
            if let Some(ref bias) = layer.qkv_bias {
                // Add bias to each position
                for s in 0..seq_len {
                    let offset = s * qkv_dim;
                    for (i, b) in bias.iter().enumerate() {
                        qkv[offset + i] += b;
                    }
                }
            }

            // Calculate K/V dimension for GQA support
            let kv_dim = (qkv_dim - hidden_dim) / 2;
            let num_kv_heads = self.config.num_kv_heads;
            let head_dim = hidden_dim / num_heads;

            // 2c. Scaled dot-product attention with RoPE and causal masking
            // Build K/V cache for self-attention within this forward pass
            let mut k_cache = Vec::with_capacity(seq_len * hidden_dim);
            let mut v_cache = Vec::with_capacity(seq_len * hidden_dim);
            let mut attn_out = Vec::with_capacity(seq_len * hidden_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V with GQA-aware dimensions
                let mut q = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let k_raw = &qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim];
                let v_raw = &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + qkv_dim];

                // Apply RoPE to Q and K
                apply_rope(&mut q, hidden_dim, num_heads, s, self.config.rope_theta);
                let mut k = k_raw.to_vec();
                apply_rope(&mut k, kv_dim, num_kv_heads, s, self.config.rope_theta);

                // For GQA: expand K/V by repeating each KV head to match Q heads
                let (k_expanded, v_expanded): (Vec<f32>, Vec<f32>) = if num_kv_heads < num_heads {
                    let group_size = num_heads / num_kv_heads;
                    let expand = |raw: &[f32]| -> Vec<f32> {
                        (0..num_heads)
                            .flat_map(|h| {
                                let kv_head = h / group_size;
                                let start = kv_head * head_dim;
                                raw[start..start + head_dim].iter().copied()
                            })
                            .collect()
                    };
                    (expand(&k), expand(v_raw))
                } else {
                    (k, v_raw.to_vec())
                };

                // Add to K/V cache for causal self-attention
                k_cache.extend_from_slice(&k_expanded);
                v_cache.extend_from_slice(&v_expanded);

                // Compute attention: softmax(Q·K^T / sqrt(d_k)) · V
                // Uses causal masking - only attend to positions 0..=s
                let attn_output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);
                attn_out.extend_from_slice(&attn_output);
            }

            // 2d. Attention output projection (QUANTIZED)
            let mut attn_output = if seq_len == 1 {
                layer.attn_output_weight.matvec(&attn_out)?
            } else {
                let mut batch_out = Vec::with_capacity(seq_len * hidden_dim);
                for s in 0..seq_len {
                    let pos_input = &attn_out[s * hidden_dim..(s + 1) * hidden_dim];
                    let pos_out = layer.attn_output_weight.matvec(pos_input)?;
                    batch_out.extend(pos_out);
                }
                batch_out
            };
            if let Some(ref bias) = layer.attn_output_bias {
                for s in 0..seq_len {
                    let offset = s * hidden_dim;
                    for (i, b) in bias.iter().enumerate() {
                        attn_output[offset + i] += b;
                    }
                }
            }

            // 2e. Residual connection
            simd_add(&mut hidden, &attn_output);

            // 2f. FFN (with optional FFN norm)
            let ffn_input = if let Some(ref norm_weight) = layer.ffn_norm_weight {
                simd_layer_norm(
                    &hidden,
                    norm_weight,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            // FFN forward (QUANTIZED)
            let intermediate_dim = self.config.intermediate_dim;
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: output = down(silu(gate(x)) * up(x))
                let mut gate = if seq_len == 1 {
                    gate_weight.matvec(&ffn_input)?
                } else {
                    let mut batch = Vec::with_capacity(seq_len * intermediate_dim);
                    for s in 0..seq_len {
                        let pos = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                        batch.extend(gate_weight.matvec(pos)?);
                    }
                    batch
                };
                if let Some(ref bias) = layer.ffn_gate_bias {
                    for s in 0..seq_len {
                        let offset = s * intermediate_dim;
                        for (i, b) in bias.iter().enumerate() {
                            gate[offset + i] += b;
                        }
                    }
                }
                simd_silu(&mut gate);

                let mut up = if seq_len == 1 {
                    layer.ffn_up_weight.matvec(&ffn_input)?
                } else {
                    let mut batch = Vec::with_capacity(seq_len * intermediate_dim);
                    for s in 0..seq_len {
                        let pos = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                        batch.extend(layer.ffn_up_weight.matvec(pos)?);
                    }
                    batch
                };
                if let Some(ref bias) = layer.ffn_up_bias {
                    for s in 0..seq_len {
                        let offset = s * intermediate_dim;
                        for (i, b) in bias.iter().enumerate() {
                            up[offset + i] += b;
                        }
                    }
                }

                simd_mul(&mut gate, &up);

                let mut output = if seq_len == 1 {
                    layer.ffn_down_weight.matvec(&gate)?
                } else {
                    let mut batch = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let pos = &gate[s * intermediate_dim..(s + 1) * intermediate_dim];
                        batch.extend(layer.ffn_down_weight.matvec(pos)?);
                    }
                    batch
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    for s in 0..seq_len {
                        let offset = s * hidden_dim;
                        for (i, b) in bias.iter().enumerate() {
                            output[offset + i] += b;
                        }
                    }
                }
                output
            } else {
                // Standard FFN with GELU
                let mut ffn_hidden = if seq_len == 1 {
                    layer.ffn_up_weight.matvec(&ffn_input)?
                } else {
                    let mut batch = Vec::with_capacity(seq_len * intermediate_dim);
                    for s in 0..seq_len {
                        let pos = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                        batch.extend(layer.ffn_up_weight.matvec(pos)?);
                    }
                    batch
                };
                if let Some(ref bias) = layer.ffn_up_bias {
                    for s in 0..seq_len {
                        let offset = s * intermediate_dim;
                        for (i, b) in bias.iter().enumerate() {
                            ffn_hidden[offset + i] += b;
                        }
                    }
                }
                simd_gelu(&mut ffn_hidden);

                let mut output = if seq_len == 1 {
                    layer.ffn_down_weight.matvec(&ffn_hidden)?
                } else {
                    let mut batch = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let pos = &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                        batch.extend(layer.ffn_down_weight.matvec(pos)?);
                    }
                    batch
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    for s in 0..seq_len {
                        let offset = s * hidden_dim;
                        for (i, b) in bias.iter().enumerate() {
                            output[offset + i] += b;
                        }
                    }
                }
                output
            };

            // Residual connection
            simd_add(&mut hidden, &ffn_output);
        }

        // 3. Final layer norm (f32)
        let normed = simd_layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection for last token only (QUANTIZED)
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self.lm_head_weight.matvec(last_hidden)?;
        if let Some(ref bias) = self.lm_head_bias {
            simd_add(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Forward pass for single token with KV cache (quantized)
    ///
    /// High-performance path for autoregressive generation.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token to process
    /// * `position` - Position in sequence (for RoPE)
    /// * `kv_cache` - Mutable reference to KV cache
    ///
    /// # Returns
    ///
    /// Logits for next token prediction
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail
    pub fn forward_one_token(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        let mut hidden = if end <= self.token_embedding.len() {
            self.token_embedding[start..end].to_vec()
        } else {
            vec![0.0; hidden_dim]
        };

        // 2. Process through transformer layers with KV cache
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm
            let normed = simd_layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection (QUANTIZED fused matvec)
            let mut qkv = layer.qkv_weight.matvec(&normed)?;
            if let Some(ref bias) = layer.qkv_bias {
                simd_add(&mut qkv, bias);
            }

            // Calculate dimensions for GQA support
            let qkv_dim = layer.qkv_weight.out_dim;
            let kv_dim = (qkv_dim - hidden_dim) / 2;
            let num_kv_heads = self.config.num_kv_heads;

            // Extract Q, K, V with GQA-aware dimensions
            let mut q = qkv[0..hidden_dim].to_vec();
            let k_raw = &qkv[hidden_dim..hidden_dim + kv_dim];
            let v_raw = &qkv[hidden_dim + kv_dim..qkv_dim];

            // Apply RoPE
            apply_rope(
                &mut q,
                hidden_dim,
                num_heads,
                position,
                self.config.rope_theta,
            );
            let mut k = k_raw.to_vec();
            apply_rope(
                &mut k,
                kv_dim,
                num_kv_heads,
                position,
                self.config.rope_theta,
            );

            // For GQA: expand K/V for cache storage
            let (k_expanded, v_expanded): (Vec<f32>, Vec<f32>) = if num_kv_heads < num_heads {
                let group_size = num_heads / num_kv_heads;
                let expand = |raw: &[f32]| -> Vec<f32> {
                    (0..num_heads)
                        .flat_map(|h| {
                            let kv_head = h / group_size;
                            let start = kv_head * head_dim;
                            raw[start..start + head_dim].iter().copied()
                        })
                        .collect()
                };
                (expand(&k), expand(v_raw))
            } else {
                (k, v_raw.to_vec())
            };

            // Store K, V in cache
            kv_cache.store(layer_idx, &k_expanded, &v_expanded);

            // 2c. Attention using cached K/V
            let cached_keys = kv_cache.get_k(layer_idx);
            let cached_values = kv_cache.get_v(layer_idx);
            let attn_out =
                attention_with_cache(&q, cached_keys, cached_values, num_heads, head_dim);

            // 2d. Attention output projection (QUANTIZED)
            let mut attn_output = layer.attn_output_weight.matvec(&attn_out)?;
            if let Some(ref bias) = layer.attn_output_bias {
                simd_add(&mut attn_output, bias);
            }

            // 2e. Residual
            simd_add(&mut hidden, &attn_output);

            // 2f. FFN
            let ffn_input = if let Some(ref norm_weight) = layer.ffn_norm_weight {
                simd_layer_norm(
                    &hidden,
                    norm_weight,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU (QUANTIZED)
                let mut gate = gate_weight.matvec(&ffn_input)?;
                if let Some(ref bias) = layer.ffn_gate_bias {
                    simd_add(&mut gate, bias);
                }
                simd_silu(&mut gate);

                let mut up = layer.ffn_up_weight.matvec(&ffn_input)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    simd_add(&mut up, bias);
                }

                simd_mul(&mut gate, &up);

                let mut output = layer.ffn_down_weight.matvec(&gate)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    simd_add(&mut output, bias);
                }
                output
            } else {
                // Standard GELU (QUANTIZED)
                let mut ffn_hidden = layer.ffn_up_weight.matvec(&ffn_input)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    simd_add(&mut ffn_hidden, bias);
                }
                simd_gelu(&mut ffn_hidden);

                let mut output = layer.ffn_down_weight.matvec(&ffn_hidden)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    simd_add(&mut output, bias);
                }
                output
            };

            simd_add(&mut hidden, &ffn_output);
        }

        // Advance cache
        kv_cache.advance();

        // 3. Final layer norm
        let normed = simd_layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head (QUANTIZED)
        let mut logits = self.lm_head_weight.matvec(&normed)?;
        if let Some(ref bias) = self.lm_head_bias {
            simd_add(&mut logits, bias);
        }

        Ok(logits)
    }

    /// High-performance generation with KV cache (quantized)
    ///
    /// # Errors
    ///
    /// Returns error if prompt is empty or inference fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_layers = self.layers.len();
        let max_seq_len = prompt.len() + max_tokens;

        let mut kv_cache = KVCache::new(num_layers, hidden_dim, max_seq_len);

        // Prefill
        for (pos, &token_id) in prompt.iter().enumerate() {
            let _logits = self.forward_one_token(token_id, pos, &mut kv_cache)?;
        }

        let mut tokens = prompt.to_vec();

        // Decode
        for i in 0..max_tokens {
            let position = prompt.len() + i;
            let current_token = *tokens.last().expect("tokens must be non-empty");

            let logits = self.forward_one_token(current_token, position, &mut kv_cache)?;

            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32);

            if let Some(eos) = eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

/// Memory statistics for quantized inference engine
#[derive(Debug, Clone)]
pub struct QuantizedMemoryStats {
    /// Bytes used by quantized weights
    pub quantized_weight_bytes: usize,
    /// Equivalent bytes if weights were f32
    pub f32_equivalent_bytes: usize,
    /// Bytes used by token embeddings (f32)
    pub embedding_bytes: usize,
    /// Compression ratio (f32 / quantized)
    pub compression_ratio: f32,
}

impl std::fmt::Display for QuantizedMemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let quantized_mb = self.quantized_weight_bytes as f64 / (1024.0 * 1024.0);
        let f32_mb = self.f32_equivalent_bytes as f64 / (1024.0 * 1024.0);
        let embed_mb = self.embedding_bytes as f64 / (1024.0 * 1024.0);
        write!(
            f,
            "Quantized weights: {:.1} MB (vs {:.1} MB f32, {:.1}x compression)\nEmbeddings: {:.1} MB",
            quantized_mb, f32_mb, self.compression_ratio, embed_mb
        )
    }
}

/// Small, always-compiled attention tests (no OOM risk)
#[cfg(test)]
mod attention_tests {
    use super::*;

    #[test]
    fn test_attention_causal_flow() {
        // Test causal self-attention: each position only sees 0..=s
        let num_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let seq_len = 3;

        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        let mut outputs = Vec::new();

        for s in 0..seq_len {
            let q: Vec<f32> = (0..hidden_dim)
                .map(|i| (s * hidden_dim + i) as f32 * 0.1)
                .collect();
            let k: Vec<f32> = (0..hidden_dim)
                .map(|i| (s * hidden_dim + i) as f32 * 0.2)
                .collect();
            let v: Vec<f32> = (0..hidden_dim).map(|_| (s + 1) as f32).collect();

            k_cache.extend_from_slice(&k);
            v_cache.extend_from_slice(&v);

            let attn_output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);
            outputs.push(attn_output);
        }

        // Position 0: only sees V[0]=1.0
        assert_eq!(outputs[0].len(), hidden_dim);
        for val in &outputs[0] {
            assert!((val - 1.0).abs() < 1e-4, "Pos 0 should be 1.0, got {}", val);
        }

        // Positions 1,2: weighted averages, values should be bounded
        for s in 1..seq_len {
            for val in &outputs[s] {
                assert!(val.is_finite() && *val >= 0.9 && *val <= (seq_len + 1) as f32);
            }
        }
    }

    #[test]
    fn test_gqa_expansion() {
        // GQA: 4 Q heads, 2 KV heads, group_size=2
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let group_size = num_heads / num_kv_heads;

        let k_raw: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let k_expanded: Vec<f32> = (0..num_heads)
            .flat_map(|h| {
                let kv_head = h / group_size;
                let start = kv_head * head_dim;
                k_raw[start..start + head_dim].iter().copied()
            })
            .collect();

        // Q heads 0,1 → KV head 0; Q heads 2,3 → KV head 1
        assert_eq!(k_expanded, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_attention_single_position() {
        let output = attention_with_cache(
            &[1.0, 2.0, 3.0, 4.0], // Q
            &[1.0, 2.0, 3.0, 4.0], // K (1 position)
            &[5.0, 6.0, 7.0, 8.0], // V (1 position)
            2,                     // num_heads
            2,                     // head_dim
        );

        // Single position → softmax([1]) = [1] → output = V
        assert_eq!(output.len(), 4);
        for (i, &expected) in [5.0, 6.0, 7.0, 8.0].iter().enumerate() {
            assert!((output[i] - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_embedding_lookup_basic() {
        // Test that embedding lookup produces reasonable values
        let embedding = vec![
            1.0, 2.0, 3.0, 4.0, // Token 0
            5.0, 6.0, 7.0, 8.0, // Token 1
        ];
        let hidden_dim = 4;

        // Look up token 1
        let token_id = 1u32;
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        let result = &embedding[start..end];

        assert_eq!(result, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_lm_head_projection() {
        // Test that lm_head produces varied logits
        let hidden_dim = 4;
        let vocab_size = 3;

        // Simple hidden state
        let hidden = vec![1.0, 0.0, 0.0, 0.0];

        // LM head weight [vocab_size * hidden_dim] in row-major
        // Each row is a vocab token's projection
        let lm_head = vec![
            1.0, 0.0, 0.0, 0.0, // Token 0: dot with hidden = 1.0
            0.0, 1.0, 0.0, 0.0, // Token 1: dot with hidden = 0.0
            0.0, 0.0, 1.0, 0.0, // Token 2: dot with hidden = 0.0
        ];

        // Compute logits: lm_head @ hidden
        let logits: Vec<f32> = (0..vocab_size)
            .map(|v| {
                let row_start = v * hidden_dim;
                (0..hidden_dim)
                    .map(|i| lm_head[row_start + i] * hidden[i])
                    .sum()
            })
            .collect();

        assert_eq!(logits, vec![1.0, 0.0, 0.0]);
        // Token 0 should have highest logit
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("test"))
            .map(|(i, _)| i)
            .expect("test");
        assert_eq!(max_idx, 0);
    }

    #[test]
    fn test_repetition_penalty_reduces_repeats() {
        // Test that repetition penalty modifies logits correctly
        let logits = vec![1.0, 2.0, 3.0, 4.0]; // Token 3 has highest logit
        let recent_tokens = vec![3u32]; // Token 3 was recently used
        let penalty = 1.5;

        // Apply penalty manually (same logic as generate_with_sampling)
        let mut penalized = logits.clone();
        for &token_id in &recent_tokens {
            let idx = token_id as usize;
            let logit = penalized[idx];
            penalized[idx] = if logit > 0.0 {
                logit / penalty
            } else {
                logit * penalty
            };
        }

        // Token 3 should be penalized: 4.0 / 1.5 = 2.67
        assert!((penalized[3] - 2.667_f32).abs() < 0.01);
        // Other tokens unchanged
        assert_eq!(penalized[0], 1.0);
        assert_eq!(penalized[1], 2.0);
        assert_eq!(penalized[2], 3.0);

        // Token 2 should now have highest logit (3.0 > 2.67)
        let max_idx = penalized
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("test"))
            .map(|(i, _)| i)
            .expect("test");
        assert_eq!(max_idx, 2, "After penalty, token 2 should be chosen");
    }

    #[test]
    fn test_simd_rms_norm_correctness() {
        // Test RMSNorm: output = input * weight / sqrt(mean(input^2) + eps)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0]; // Identity scaling
        let eps = 1e-5;

        let result = simd_rms_norm(&input, &weight, eps);

        // Compute expected:
        // sum_sq = 1 + 4 + 9 + 16 = 30
        // rms = sqrt(30/4 + eps) = sqrt(7.5 + eps) ≈ 2.739
        // normalized = input / rms
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 4.0 + eps).sqrt();
        let expected: Vec<f32> = input.iter().map(|&x| x / rms).collect();

        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "RMSNorm mismatch at {}: got {} expected {}",
                i,
                r,
                e
            );
        }
    }

    #[test]
    fn test_simd_rms_norm_with_gamma() {
        // Test RMSNorm with non-identity gamma weights
        let input = vec![2.0, 2.0, 2.0, 2.0];
        let weight = vec![0.5, 1.0, 1.5, 2.0]; // Different gamma per dimension
        let eps = 1e-5;

        let result = simd_rms_norm(&input, &weight, eps);

        // All inputs same → after normalization, all become 1/sqrt(1) = 1
        // (because mean(2^2) = 4, rms = 2, so 2/2 = 1)
        // Then multiply by weight
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 4.0 + eps).sqrt(); // = 2.0
        let expected: Vec<f32> = input
            .iter()
            .zip(weight.iter())
            .map(|(&x, &w)| x / rms * w)
            .collect();

        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "RMSNorm with gamma mismatch at {}: got {} expected {}",
                i,
                r,
                e
            );
        }
    }
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;

    // ===== EXTREME TDD: SIMD matmul tests =====

    #[test]
    fn test_simd_matmul_single_token() {
        // Input: [1, 4], Weight: [2, 4] (2 outputs, 4 inputs)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![
            1.0, 0.0, 0.0, 0.0, // Output 0: dot with input = 1.0
            0.0, 1.0, 0.0, 0.0, // Output 1: dot with input = 2.0
        ];

        let output = simd_matmul(&input, &weight, 4, 2);
        assert_eq!(output.len(), 2);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_matmul_batch() {
        // Input: [2, 2] (2 tokens, 2 dims), Weight: [3, 2] (3 outputs)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![
            1.0, 0.0, // Output 0
            0.0, 1.0, // Output 1
            1.0, 1.0, // Output 2
        ];

        let output = simd_matmul(&input, &weight, 2, 3);
        assert_eq!(output.len(), 6);
        // Token 0: [1,2] → [1, 2, 3]
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = simd_dot(&a, &b);
        assert!((result - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_add() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        simd_add(&mut a, &b);
        assert!((a[0] - 11.0).abs() < 1e-5);
        assert!((a[3] - 44.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_gelu() {
        let mut data = vec![0.0, 1.0, -1.0];
        simd_gelu(&mut data);
        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert!((data[0]).abs() < 1e-5);
        assert!((data[1] - 0.841).abs() < 0.01);
        assert!((data[2] + 0.159).abs() < 0.01);
    }

    #[test]
    fn test_simd_softmax() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_softmax(&mut data);
        // Sum should be 1.0
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Highest logit should have highest prob
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_simd_softmax_empty() {
        let mut data: Vec<f32> = vec![];
        simd_softmax(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_simd_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 position, dim 4
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = Some(vec![0.0, 0.0, 0.0, 0.0]);

        let output = simd_layer_norm(&input, &weight, bias.as_deref(), 1e-5);
        assert_eq!(output.len(), 4);

        // Mean should be subtracted, so output mean ≈ 0
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // dim 4, 1 head
        apply_rope(&mut x, 4, 1, 0, 10000.0);
        // Position 0 should have minimal rotation
        assert!((x[0] - 1.0).abs() < 0.01);
        assert!((x[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_rope_position() {
        let mut x1 = vec![1.0, 0.0, 0.0, 1.0];
        let mut x2 = vec![1.0, 0.0, 0.0, 1.0];

        apply_rope(&mut x1, 4, 1, 0, 10000.0);
        apply_rope(&mut x2, 4, 1, 10, 10000.0);

        // Different positions should give different embeddings
        assert!((x1[0] - x2[0]).abs() > 0.001 || (x1[2] - x2[2]).abs() > 0.001);
    }

    // ===== EXTREME TDD: Parallel matmul tests =====

    #[test]
    fn test_parallel_matmul_large_output() {
        // Test with output dim above PARALLEL_THRESHOLD (256)
        let in_dim = 16;
        let out_dim = 512;
        let input: Vec<f32> = (0..in_dim).map(|i| i as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect();

        let output = simd_matmul(&input, &weight, in_dim, out_dim);
        assert_eq!(output.len(), out_dim);

        // Verify first and last outputs are computed correctly
        let expected_0: f32 = (0..in_dim).map(|i| input[i] * weight[i]).sum();
        assert!((output[0] - expected_0).abs() < 1e-4);
    }

    #[test]
    fn test_parallel_matmul_correctness() {
        // Verify parallel and sequential give same results
        let in_dim = 8;
        let out_dim = 300; // Above threshold
        let input: Vec<f32> = (0..in_dim).map(|i| (i + 1) as f32).collect();
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| if i % in_dim == 0 { 1.0 } else { 0.0 })
            .collect();

        let output = simd_matmul(&input, &weight, in_dim, out_dim);

        // Each output should be input[0] = 1.0 since only first column is 1
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "Output[{}] = {} expected 1.0",
                i,
                val
            );
        }
    }

    // ===== EXTREME TDD: Tiled matmul tests =====

    #[test]
    fn test_tiled_matmul_batch() {
        // Test batch inference with tiled matmul
        let seq_len = 4;
        let in_dim = 8;
        let out_dim = 256;

        // Identity-like weight for first output
        let input: Vec<f32> = (0..seq_len * in_dim)
            .map(|i| (i % in_dim + 1) as f32)
            .collect();
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| if i % in_dim == 0 { 1.0 } else { 0.0 })
            .collect();

        let output = simd_matmul(&input, &weight, in_dim, out_dim);
        assert_eq!(output.len(), seq_len * out_dim);

        // First output of each token should be 1.0
        for t in 0..seq_len {
            assert!(
                (output[t * out_dim] - 1.0).abs() < 1e-5,
                "Token {} output[0] = {} expected 1.0",
                t,
                output[t * out_dim]
            );
        }
    }

    #[test]
    fn test_tiled_matmul_large_batch() {
        // Test with large batch to trigger tiled path
        let seq_len = 128;
        let in_dim = 16;
        let out_dim = 32;

        let input: Vec<f32> = vec![1.0; seq_len * in_dim];
        let weight: Vec<f32> = vec![0.0625; out_dim * in_dim]; // 1/16 so sum = 1

        let output = simd_matmul(&input, &weight, in_dim, out_dim);
        assert_eq!(output.len(), seq_len * out_dim);

        // Each output should be 1.0 (16 * 0.0625)
        for val in &output {
            assert!((*val - 1.0).abs() < 1e-4, "Expected 1.0, got {}", val);
        }
    }

    // ===== EXTREME TDD: KVCache tests =====

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4, 64, 128);
        assert_eq!(cache.num_layers, 4);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_store_and_retrieve() {
        let mut cache = KVCache::new(2, 4, 8);

        let k0 = vec![1.0, 2.0, 3.0, 4.0];
        let v0 = vec![5.0, 6.0, 7.0, 8.0];

        cache.store(0, &k0, &v0);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let k = cache.get_k(0);
        let v = cache.get_v(0);

        assert_eq!(k.len(), 4);
        assert_eq!(v.len(), 4);
        assert!((k[0] - 1.0).abs() < 1e-5);
        assert!((v[3] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_kv_cache_multiple_layers() {
        let mut cache = KVCache::new(3, 2, 4);

        // Store for layer 0
        cache.store(0, &[1.0, 2.0], &[3.0, 4.0]);
        // Store for layer 1
        cache.store(1, &[5.0, 6.0], &[7.0, 8.0]);
        // Store for layer 2
        cache.store(2, &[9.0, 10.0], &[11.0, 12.0]);
        cache.advance();

        // Verify each layer has correct values
        assert!((cache.get_k(0)[0] - 1.0).abs() < 1e-5);
        assert!((cache.get_k(1)[0] - 5.0).abs() < 1e-5);
        assert!((cache.get_k(2)[0] - 9.0).abs() < 1e-5);

        assert!((cache.get_v(0)[1] - 4.0).abs() < 1e-5);
        assert!((cache.get_v(1)[1] - 8.0).abs() < 1e-5);
        assert!((cache.get_v(2)[1] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_kv_cache_multiple_positions() {
        let mut cache = KVCache::new(1, 2, 8);

        // Store 3 positions
        cache.store(0, &[1.0, 1.0], &[2.0, 2.0]);
        cache.advance();
        cache.store(0, &[3.0, 3.0], &[4.0, 4.0]);
        cache.advance();
        cache.store(0, &[5.0, 5.0], &[6.0, 6.0]);
        cache.advance();

        assert_eq!(cache.len(), 3);

        let k = cache.get_k(0);
        assert_eq!(k.len(), 6); // 3 positions * 2 dim
        assert!((k[0] - 1.0).abs() < 1e-5); // Position 0
        assert!((k[2] - 3.0).abs() < 1e-5); // Position 1
        assert!((k[4] - 5.0).abs() < 1e-5); // Position 2
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KVCache::new(2, 4, 16);

        cache.store(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();
        cache.store(0, &[3.0; 4], &[4.0; 4]);
        cache.advance();

        assert_eq!(cache.len(), 2);

        cache.reset();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_full() {
        let mut cache = KVCache::new(1, 2, 2);

        // Fill cache
        cache.store(0, &[1.0, 1.0], &[1.0, 1.0]);
        cache.advance();
        cache.store(0, &[2.0, 2.0], &[2.0, 2.0]);
        cache.advance();

        assert_eq!(cache.len(), 2);

        // Try to store when full - should be ignored
        cache.store(0, &[3.0, 3.0], &[3.0, 3.0]);
        cache.advance();

        // Still at max
        assert_eq!(cache.len(), 2);

        // Last stored values should still be 2.0
        let k = cache.get_k(0);
        assert!((k[2] - 2.0).abs() < 1e-5);
    }

    // ===== EXTREME TDD: Attention with cache tests =====

    #[test]
    fn test_attention_with_cache_empty() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k_cache: Vec<f32> = vec![];
        let v_cache: Vec<f32> = vec![];

        let output = attention_with_cache(&q, &k_cache, &v_cache, 2, 2);

        // Empty cache should return Q unchanged
        assert_eq!(output.len(), 4);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_with_cache_single_position() {
        let num_heads = 2;
        let head_dim = 4;
        let hidden_dim = num_heads * head_dim;

        // Q and K are identical - should get high attention to V
        let q: Vec<f32> = vec![1.0; hidden_dim];
        let k_cache: Vec<f32> = vec![1.0; hidden_dim]; // 1 position
        let v_cache: Vec<f32> = (0..hidden_dim).map(|i| i as f32).collect();

        let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

        assert_eq!(output.len(), hidden_dim);
        // With single position, output should be V since softmax of single element is 1.0
        for i in 0..hidden_dim {
            assert!(
                (output[i] - v_cache[i]).abs() < 1e-4,
                "output[{}] = {} expected {}",
                i,
                output[i],
                v_cache[i]
            );
        }
    }

    #[test]
    fn test_attention_with_cache_multiple_positions() {
        let num_heads = 1;
        let head_dim = 4;
        let hidden_dim = num_heads * head_dim;

        // Q matches second K position more than first
        let q = vec![0.0, 0.0, 1.0, 1.0];
        let k_cache = vec![
            1.0, 1.0, 0.0, 0.0, // Position 0 - dot with Q = 0
            0.0, 0.0, 1.0, 1.0, // Position 1 - dot with Q = 2
        ];
        let v_cache = vec![
            1.0, 0.0, 0.0, 0.0, // V for position 0
            0.0, 1.0, 0.0, 0.0, // V for position 1
        ];

        let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

        assert_eq!(output.len(), hidden_dim);
        // Position 1 should have higher attention weight
        // Output should be weighted average favoring V[1]
        assert!(output[1] > output[0], "Should attend more to position 1");
    }

    #[test]
    fn test_attention_with_cache_multi_head() {
        let num_heads = 4;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;

        let q: Vec<f32> = (0..hidden_dim).map(|i| (i % 3) as f32).collect();
        let k_cache: Vec<f32> = vec![1.0; hidden_dim * 3]; // 3 positions
        let v_cache: Vec<f32> = (0..hidden_dim * 3).map(|i| (i % 5) as f32 * 0.1).collect();

        let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

        // Each head should produce head_dim outputs
        assert_eq!(output.len(), hidden_dim);

        // Output should be finite and reasonable
        for val in &output {
            assert!(val.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_attention_scale_factor() {
        // Verify scaling by sqrt(d_k) works correctly
        let num_heads = 1;
        let head_dim = 64; // sqrt(64) = 8
        let hidden_dim = num_heads * head_dim;

        // Large values to test scaling prevents overflow
        let q: Vec<f32> = vec![10.0; hidden_dim];
        let k_cache: Vec<f32> = vec![10.0; hidden_dim];
        let v_cache: Vec<f32> = vec![1.0; hidden_dim];

        let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

        // Should not overflow, output should be close to V (only one position)
        for val in &output {
            assert!(val.is_finite());
            assert!((*val - 1.0).abs() < 1e-4);
        }
    }

    // ===== EXTREME TDD: simd_mul and simd_silu tests =====

    #[test]
    fn test_simd_mul() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        simd_mul(&mut a, &b);
        assert!((a[0] - 2.0).abs() < 1e-5);
        assert!((a[1] - 6.0).abs() < 1e-5);
        assert!((a[2] - 12.0).abs() < 1e-5);
        assert!((a[3] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_silu() {
        let mut data = vec![0.0, 1.0, -1.0, 2.0];
        simd_silu(&mut data);
        // SiLU(0) = 0
        assert!(data[0].abs() < 1e-5);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((data[1] - 0.731).abs() < 0.01);
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.269
        assert!((data[2] + 0.269).abs() < 0.01);
        // SiLU(2) ≈ 1.762
        assert!((data[3] - 1.762).abs() < 0.01);
    }

    // ===== EXTREME TDD: KVCache integration tests =====

    #[test]
    fn test_kv_cache_attention_flow() {
        // Simulate multi-layer attention flow with KV cache
        let num_layers = 2;
        let hidden_dim = 8;
        let max_seq = 4;
        let num_heads = 2;
        let head_dim = 4;

        let mut cache = KVCache::new(num_layers, hidden_dim, max_seq);

        // Process 3 tokens
        for pos in 0..3 {
            for layer in 0..num_layers {
                // test K, V from QKV projection
                let k: Vec<f32> = (0..hidden_dim)
                    .map(|i| (pos + layer + i) as f32 * 0.1)
                    .collect();
                let v: Vec<f32> = (0..hidden_dim)
                    .map(|i| (pos + layer + i) as f32 * 0.2)
                    .collect();
                cache.store(layer, &k, &v);
            }
            cache.advance();
        }

        assert_eq!(cache.len(), 3);

        // Verify we can retrieve K/V for attention
        for layer in 0..num_layers {
            let k = cache.get_k(layer);
            let v = cache.get_v(layer);

            // 3 positions * 8 hidden_dim
            assert_eq!(k.len(), 24);
            assert_eq!(v.len(), 24);
        }

        // Query for position 3
        let q: Vec<f32> = vec![1.0; hidden_dim];
        let output = attention_with_cache(&q, cache.get_k(0), cache.get_v(0), num_heads, head_dim);
        assert_eq!(output.len(), hidden_dim);

        // Output should be finite
        for val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kv_cache_incremental_attention() {
        // Verify incremental attention matches full attention semantically
        let num_heads = 2;
        let head_dim = 4;
        let hidden_dim = num_heads * head_dim;

        // Build up cache incrementally
        let mut cache = KVCache::new(1, hidden_dim, 10);

        // Position 0
        let k0 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let v0 = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        cache.store(0, &k0, &v0);
        cache.advance();

        // Query at position 1
        let q1 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let out1 = attention_with_cache(&q1, cache.get_k(0), cache.get_v(0), num_heads, head_dim);

        // Position 1 K, V
        let k1 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v1 = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        cache.store(0, &k1, &v1);
        cache.advance();

        // Query at position 2 - now attends to both positions
        let q2 = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let out2 = attention_with_cache(&q2, cache.get_k(0), cache.get_v(0), num_heads, head_dim);

        // Output 2 should blend V0 and V1 based on attention scores
        assert_eq!(out2.len(), hidden_dim);
        // Both outputs should be finite
        for val in out1.iter().chain(out2.iter()) {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_attention_softmax_stability() {
        // Test attention doesn't overflow with large values
        let num_heads = 1;
        let head_dim = 4;

        let q = vec![100.0, 100.0, 100.0, 100.0]; // Large Q values
        let k_cache = vec![
            100.0, 100.0, 100.0, 100.0, // Large K at pos 0
            -100.0, -100.0, -100.0, -100.0, // Large negative K at pos 1
        ];
        let v_cache = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

        // Should not have NaN or Inf
        for (i, val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Output[{}] = {} should be finite", i, val);
        }

        // Position 0 has much higher attention score, so output should be close to V0
        assert!(output[0] < 2.0, "Should be close to V0 (1.0)");
    }

    #[test]
    fn test_causal_self_attention_flow() {
        // Test the causal self-attention pattern used in forward:
        // For each position s, only attend to positions 0..=s
        let num_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let seq_len = 3;

        // Simulate building K/V cache incrementally (causal)
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        let mut outputs = Vec::new();

        for s in 0..seq_len {
            // Q for position s (distinct values per position)
            let q: Vec<f32> = (0..hidden_dim)
                .map(|i| (s * hidden_dim + i) as f32 * 0.1)
                .collect();

            // K/V for position s
            let k: Vec<f32> = (0..hidden_dim)
                .map(|i| (s * hidden_dim + i) as f32 * 0.2)
                .collect();
            let v: Vec<f32> = (0..hidden_dim).map(|_| (s + 1) as f32).collect(); // V[s] = s+1

            // Add to cache BEFORE attention (causal - includes current position)
            k_cache.extend_from_slice(&k);
            v_cache.extend_from_slice(&v);

            // Compute attention over cache (only positions 0..=s visible)
            let attn_output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);
            outputs.push(attn_output);
        }

        // Position 0: only sees itself
        assert_eq!(outputs[0].len(), hidden_dim);
        for val in &outputs[0] {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "Pos 0 should output V[0]=1.0, got {}",
                val
            );
        }

        // Position 1: sees pos 0 and 1 (weighted average of V[0]=1.0 and V[1]=2.0)
        for val in &outputs[1] {
            assert!(
                *val >= 0.9 && *val <= 2.1,
                "Pos 1 should be between V[0] and V[1], got {}",
                val
            );
        }

        // Position 2: sees pos 0, 1, 2 (weighted average)
        for val in &outputs[2] {
            assert!(
                *val >= 0.9 && *val <= 3.1,
                "Pos 2 should be in range of V values, got {}",
                val
            );
        }
    }

    #[test]
    fn test_gqa_kv_expansion() {
        // Test GQA K/V expansion: 4 Q heads with 2 KV heads
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let group_size = num_heads / num_kv_heads; // 2

        // Raw K with 2 KV heads: [kv0_h0, kv0_h1, kv1_h0, kv1_h1]
        let k_raw: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(k_raw.len(), kv_dim);

        // Expand K to match Q heads: each KV head serves 2 Q heads
        let k_expanded: Vec<f32> = (0..num_heads)
            .flat_map(|h| {
                let kv_head = h / group_size;
                let start = kv_head * head_dim;
                k_raw[start..start + head_dim].iter().copied()
            })
            .collect();

        assert_eq!(k_expanded.len(), hidden_dim);
        // Q heads 0,1 use KV head 0: [1.0, 2.0]
        // Q heads 2,3 use KV head 1: [3.0, 4.0]
        assert_eq!(k_expanded, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_kv_cache_layer_isolation() {
        // Verify different layers maintain separate caches
        let mut cache = KVCache::new(3, 4, 8);

        // Store different values in each layer
        cache.store(0, &[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0]);
        cache.store(1, &[2.0, 2.0, 2.0, 2.0], &[2.0, 2.0, 2.0, 2.0]);
        cache.store(2, &[3.0, 3.0, 3.0, 3.0], &[3.0, 3.0, 3.0, 3.0]);
        cache.advance();

        // Verify each layer has its own values
        let k0 = cache.get_k(0);
        let k1 = cache.get_k(1);
        let k2 = cache.get_k(2);

        assert!((k0[0] - 1.0).abs() < 1e-5);
        assert!((k1[0] - 2.0).abs() < 1e-5);
        assert!((k2[0] - 3.0).abs() < 1e-5);
    }

    // ===== EXTREME TDD: Quantized Inference Tests =====

    /// Create valid Q4_K test data for a single super-block (256 values = 144 bytes)
    fn create_q4k_test_data(num_super_blocks: usize) -> Vec<u8> {
        const SUPER_BLOCK_BYTES: usize = 144;
        let mut data = Vec::with_capacity(num_super_blocks * SUPER_BLOCK_BYTES);

        for _ in 0..num_super_blocks {
            // Scale values (f16 x 2 = 4 bytes)
            data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0 in f16
            data.extend_from_slice(&[0x00, 0x00]); // dmin = 0.0 in f16

            // Block scale indices (12 bytes)
            data.extend_from_slice(&[0u8; 12]);

            // Quantized values (128 bytes for 256 values at 4 bits each)
            for _ in 0..128 {
                data.push(0x77); // Each nibble = 7
            }
        }

        data
    }

    #[test]
    fn test_q4k_weight_creation() {
        // Q4K format: 256 values per super-block, 144 bytes per super-block
        // in_dim = 256, out_dim = 1 → 1 super-block per row, 1 row = 144 bytes
        let data = create_q4k_test_data(1);
        let weight = Q4KWeight::new(data, 256, 1).expect("test");

        assert_eq!(weight.in_dim, 256);
        assert_eq!(weight.out_dim, 1);
        assert_eq!(weight.memory_bytes(), 144);
        assert_eq!(weight.f32_equivalent_bytes(), 1024); // 256 * 1 * 4 bytes
        assert!(weight.compression_ratio() > 7.0); // Should be ~7.1x
    }

    #[test]
    fn test_q4k_weight_invalid_size() {
        // Too little data for dimensions
        let data = vec![0u8; 100]; // Less than 144 bytes needed
        let result = Q4KWeight::new(data, 256, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_q4k_weight_matvec_dimension_mismatch() {
        let data = create_q4k_test_data(1);
        let weight = Q4KWeight::new(data, 256, 1).expect("test");

        // Wrong input dimension
        let input = vec![1.0f32; 128]; // Should be 256
        let result = weight.matvec(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_q4k_weight_matvec_correct_output_size() {
        let data = create_q4k_test_data(2); // 2 super-blocks = 512 values, 1 row
        let weight = Q4KWeight::new(data, 512, 1).expect("test");

        let input = vec![1.0f32; 512];
        let output = weight.matvec(&input).expect("test");

        assert_eq!(output.len(), 1); // out_dim = 1
    }

    #[test]
    fn test_quantized_memory_stats_display() {
        let stats = QuantizedMemoryStats {
            quantized_weight_bytes: 1024 * 1024,   // 1 MB
            f32_equivalent_bytes: 8 * 1024 * 1024, // 8 MB
            embedding_bytes: 512 * 1024,           // 0.5 MB
            compression_ratio: 8.0,
        };

        let display = format!("{}", stats);
        assert!(display.contains("1.0 MB"));
        assert!(display.contains("8.0 MB"));
        assert!(display.contains("8.0x"));
    }

    #[test]
    fn test_quantized_inference_engine_memory_stats() {
        // Build a minimal quantized engine
        let config = GGUFConfig {
            architecture: "test".to_string(),
            vocab_size: 100,
            hidden_dim: 256,
            intermediate_dim: 512,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            context_length: 2048,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // Create minimal quantized layer
        // Q4_K: 256 values = 1 super-block (144 bytes)
        // For in_dim=256: 1 super-block/row
        // For in_dim=512: 2 super-blocks/row
        let qkv_data = create_q4k_test_data(768); // 768 output rows × 1 sb/row = 768 sb
        let attn_out_data = create_q4k_test_data(256); // 256 output rows × 1 sb/row = 256 sb
        let ffn_up_data = create_q4k_test_data(512); // 512 output rows × 1 sb/row = 512 sb
        let ffn_down_data = create_q4k_test_data(256 * 2); // 256 output rows × 2 sb/row = 512 sb
        let lm_head_data = create_q4k_test_data(100); // 100 output rows × 1 sb/row = 100 sb

        let layer = QuantizedTransformerLayer {
            attn_norm_weight: vec![1.0; 256],
            attn_norm_bias: None,
            qkv_weight: Q4KWeight::new(qkv_data, 256, 768).expect("test"),
            qkv_bias: None,
            attn_output_weight: Q4KWeight::new(attn_out_data, 256, 256).expect("test"),
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: Q4KWeight::new(ffn_up_data, 256, 512).expect("test"),
            ffn_up_bias: None,
            ffn_down_weight: Q4KWeight::new(ffn_down_data, 512, 256).expect("test"),
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        };

        let token_embedding = vec![0.0f32; 100 * 256];

        let engine = QuantizedInferenceEngine::new(
            config,
            token_embedding,
            vec![layer],
            vec![1.0; 256],
            None,
            Q4KWeight::new(lm_head_data, 256, 100).expect("test"),
            None,
        );

        let stats = engine.memory_stats();

        // Should have positive values
        assert!(stats.quantized_weight_bytes > 0);
        assert!(stats.f32_equivalent_bytes > 0);
        assert!(stats.embedding_bytes > 0);
        assert!(stats.compression_ratio > 1.0);

        // Compression ratio should be around 7-8x for Q4_K
        assert!(stats.compression_ratio > 5.0);
        assert!(stats.compression_ratio < 10.0);
    }

    #[test]
    fn test_quantized_engine_embed() {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            vocab_size: 10,
            hidden_dim: 256,
            intermediate_dim: 512,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 0, // No layers for embedding test
            context_length: 2048,
            eps: 1e-5,
            rope_theta: 10000.0,
        };

        // Create embeddings where token i has value i+1 in all positions
        let mut token_embedding = vec![0.0f32; 10 * 256];
        for token_id in 0..10 {
            for j in 0..256 {
                token_embedding[token_id * 256 + j] = (token_id + 1) as f32;
            }
        }

        let lm_head_data = create_q4k_test_data(10); // 10 vocab, 256 hidden

        let engine = QuantizedInferenceEngine::new(
            config,
            token_embedding,
            vec![],
            vec![1.0; 256],
            None,
            Q4KWeight::new(lm_head_data, 256, 10).expect("test"),
            None,
        );

        // Test embedding lookup via private method simulation
        // We can't call embed directly, but we can verify through forward pass structure
        // Instead test that the engine was created successfully
        assert_eq!(engine.config.vocab_size, 10);
    }

    #[test]
    fn test_q4k_weight_compression_ratio() {
        // For Q4_K: 144 bytes per 256 values = 0.5625 bytes/value
        // f32: 4 bytes/value
        // Compression ratio = 4 / 0.5625 ≈ 7.11x

        let data = create_q4k_test_data(4); // 4 super-blocks = 1024 values
        let weight = Q4KWeight::new(data, 1024, 1).expect("test");

        let ratio = weight.compression_ratio();
        assert!(ratio > 7.0, "Compression ratio {} should be > 7.0", ratio);
        assert!(ratio < 8.0, "Compression ratio {} should be < 8.0", ratio);
    }

    #[test]
    fn test_quantized_transformer_layer_fields() {
        // Verify all fields can be set
        // Q4_K: 256 values = 1 super-block, so for in_dim=512 need 2 sb/row
        let qkv_data = create_q4k_test_data(3); // 3 output rows × 1 sb/row
        let attn_out_data = create_q4k_test_data(1); // 1 output row × 1 sb/row
        let gate_data = create_q4k_test_data(2); // 2 output rows × 1 sb/row
        let up_data = create_q4k_test_data(2); // 2 output rows × 1 sb/row
        let down_data = create_q4k_test_data(4); // 2 output rows × 2 sb/row (in_dim=512)

        let layer = QuantizedTransformerLayer {
            attn_norm_weight: vec![1.0; 256],
            attn_norm_bias: Some(vec![0.0; 256]),
            qkv_weight: Q4KWeight::new(qkv_data, 256, 3).expect("test"),
            qkv_bias: Some(vec![0.0; 768]),
            attn_output_weight: Q4KWeight::new(attn_out_data, 256, 1).expect("test"),
            attn_output_bias: Some(vec![0.0; 256]),
            ffn_gate_weight: Some(Q4KWeight::new(gate_data, 256, 2).expect("test")),
            ffn_gate_bias: Some(vec![0.0; 512]),
            ffn_up_weight: Q4KWeight::new(up_data, 256, 2).expect("test"),
            ffn_up_bias: Some(vec![0.0; 512]),
            ffn_down_weight: Q4KWeight::new(down_data, 512, 2).expect("test"),
            ffn_down_bias: Some(vec![0.0; 256]),
            ffn_norm_weight: Some(vec![1.0; 256]),
            ffn_norm_bias: Some(vec![0.0; 256]),
        };

        // All fields should be accessible
        assert_eq!(layer.attn_norm_weight.len(), 256);
        assert!(layer.attn_norm_bias.is_some());
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    // ===== EXTREME TDD: OptimizedKVCache with V transpose (per llama.cpp) =====

    #[test]
    fn test_optimized_kv_cache_creation() {
        let cache = OptimizedKVCache::new(2, 64, 512);
        assert_eq!(cache.num_layers, 2);
        assert_eq!(cache.hidden_dim, 64);
        assert_eq!(cache.max_seq_len, 512);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_optimized_kv_cache_store_and_get() {
        let mut cache = OptimizedKVCache::new(1, 4, 8);

        // Store first position
        let k1 = vec![1.0, 2.0, 3.0, 4.0];
        let v1 = vec![0.1, 0.2, 0.3, 0.4];
        cache.store(0, &k1, &v1);
        cache.advance();

        assert_eq!(cache.len(), 1);

        // K should be in row-major [seq_len, hidden_dim]
        let k_cached = cache.get_k(0);
        assert_eq!(k_cached.len(), 4);
        assert!((k_cached[0] - 1.0).abs() < 1e-6);
        assert!((k_cached[3] - 4.0).abs() < 1e-6);

        // V should be transposed [hidden_dim, seq_len]
        let v_cached = cache.get_v_transposed(0);
        assert_eq!(v_cached.len(), 4);
        // First element of each hidden_dim row
        assert!((v_cached[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_optimized_kv_cache_v_transpose_layout() {
        // Verify V is stored transposed for optimal memory access
        let mut cache = OptimizedKVCache::new(1, 4, 8);

        // Store 3 positions
        cache.store(0, &[1.0; 4], &[1.0, 2.0, 3.0, 4.0]); // pos 0
        cache.advance();
        cache.store(0, &[1.0; 4], &[5.0, 6.0, 7.0, 8.0]); // pos 1
        cache.advance();
        cache.store(0, &[1.0; 4], &[9.0, 10.0, 11.0, 12.0]); // pos 2
        cache.advance();

        // V transposed: [hidden_dim=4, seq_len=3]
        // Row 0 (dim 0 across all positions): [1.0, 5.0, 9.0]
        // Row 1 (dim 1 across all positions): [2.0, 6.0, 10.0]
        // etc.
        let v_t = cache.get_v_transposed(0);
        assert_eq!(v_t.len(), 12); // 4 * 3

        // Check transposed layout: contiguous access per output dim
        assert!((v_t[0] - 1.0).abs() < 1e-6); // dim 0, pos 0
        assert!((v_t[1] - 5.0).abs() < 1e-6); // dim 0, pos 1
        assert!((v_t[2] - 9.0).abs() < 1e-6); // dim 0, pos 2
        assert!((v_t[3] - 2.0).abs() < 1e-6); // dim 1, pos 0
        assert!((v_t[4] - 6.0).abs() < 1e-6); // dim 1, pos 1
    }

    #[test]
    fn test_attention_with_transposed_v_correctness() {
        // Verify optimized attention produces same result as original
        let hidden_dim = 8;
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 3;

        // Create test data
        let q: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.1).collect();
        let k_cache: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| (i % 7) as f32 * 0.1)
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| (i % 5) as f32 * 0.1)
            .collect();

        // Original attention
        let original = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

        // Transpose V for optimized attention
        let mut v_transposed = vec![0.0f32; hidden_dim * seq_len];
        for pos in 0..seq_len {
            for dim in 0..hidden_dim {
                v_transposed[dim * seq_len + pos] = v_cache[pos * hidden_dim + dim];
            }
        }

        let optimized =
            attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

        // Results should match within tolerance
        assert_eq!(original.len(), optimized.len());
        for i in 0..original.len() {
            assert!(
                (original[i] - optimized[i]).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i,
                original[i],
                optimized[i]
            );
        }
    }

    #[test]
    fn test_attention_with_transposed_v_empty_cache() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k_cache: Vec<f32> = vec![];
        let v_transposed: Vec<f32> = vec![];

        let result = attention_with_transposed_v(&q, &k_cache, &v_transposed, 2, 2, 0);

        // Empty cache returns Q unchanged
        assert_eq!(result, q);
    }

    #[test]
    fn test_optimized_kv_cache_reset() {
        let mut cache = OptimizedKVCache::new(1, 4, 8);
        cache.store(0, &[1.0; 4], &[1.0; 4]);
        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_optimized_kv_cache_multiple_layers() {
        let mut cache = OptimizedKVCache::new(3, 4, 8);

        // Store in different layers
        cache.store(0, &[1.0; 4], &[1.0; 4]);
        cache.store(1, &[2.0; 4], &[2.0; 4]);
        cache.store(2, &[3.0; 4], &[3.0; 4]);
        cache.advance();

        // Verify each layer has correct data
        let k0 = cache.get_k(0);
        let k1 = cache.get_k(1);
        let k2 = cache.get_k(2);

        assert!((k0[0] - 1.0).abs() < 1e-6);
        assert!((k1[0] - 2.0).abs() < 1e-6);
        assert!((k2[0] - 3.0).abs() < 1e-6);
    }

    // ========================================================================
    // ThreadConfig and InferenceMode Tests
    // ========================================================================

    #[test]
    fn test_thread_config_auto() {
        let config = ThreadConfig::auto();
        let num_cpus = rayon::current_num_threads();

        assert_eq!(config.n_threads_batch, num_cpus);
        assert_eq!(config.n_threads_decode, (num_cpus / 2).max(1));
    }

    #[test]
    fn test_thread_config_new() {
        let config = ThreadConfig::new(8, 4);
        assert_eq!(config.n_threads_batch, 8);
        assert_eq!(config.n_threads_decode, 4);
    }

    #[test]
    fn test_thread_config_new_clamps_to_one() {
        // Zero values should be clamped to 1
        let config = ThreadConfig::new(0, 0);
        assert_eq!(config.n_threads_batch, 1);
        assert_eq!(config.n_threads_decode, 1);
    }

    #[test]
    fn test_thread_config_threads_for() {
        let config = ThreadConfig::new(8, 2);

        // Prefill uses batch threads
        assert_eq!(config.threads_for(true), 8);

        // Decode uses fewer threads
        assert_eq!(config.threads_for(false), 2);
    }

    #[test]
    fn test_thread_config_default() {
        let config = ThreadConfig::default();
        let auto = ThreadConfig::auto();

        // Default should be same as auto
        assert_eq!(config.n_threads_batch, auto.n_threads_batch);
        assert_eq!(config.n_threads_decode, auto.n_threads_decode);
    }

    #[test]
    fn test_inference_mode_equality() {
        assert_eq!(InferenceMode::Prefill, InferenceMode::Prefill);
        assert_eq!(InferenceMode::Decode, InferenceMode::Decode);
        assert_ne!(InferenceMode::Prefill, InferenceMode::Decode);
    }

    #[test]
    fn test_inference_mode_debug() {
        let prefill = InferenceMode::Prefill;
        let decode = InferenceMode::Decode;

        // Test Debug trait
        assert!(format!("{:?}", prefill).contains("Prefill"));
        assert!(format!("{:?}", decode).contains("Decode"));
    }

    #[test]
    fn test_inference_mode_clone() {
        let original = InferenceMode::Prefill;
        let cloned = original;

        assert_eq!(original, cloned);
    }

    #[test]
    fn test_thread_config_with_inference_mode() {
        let config = ThreadConfig::new(16, 4);

        // Test integration with InferenceMode
        let mode = InferenceMode::Prefill;
        let threads = config.threads_for(mode == InferenceMode::Prefill);
        assert_eq!(threads, 16);

        let mode = InferenceMode::Decode;
        let threads = config.threads_for(mode == InferenceMode::Prefill);
        assert_eq!(threads, 4);
    }

    // ===== Additional Coverage Tests =====

    #[test]
    fn test_simd_operations_edge_cases_additional() {
        // Large negative values (numerical stability)
        let mut large_neg = vec![-1000.0, -1000.0, -1000.0];
        simd_softmax(&mut large_neg);
        let sum: f32 = large_neg.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_optimized_kv_cache_v_transposed_multi() {
        let mut cache = OptimizedKVCache::new(1, 4, 8);

        // Store values: V = [1, 2, 3, 4]
        cache.store(0, &[0.0; 4], &[1.0, 2.0, 3.0, 4.0]);
        cache.advance();

        // Store values: V = [5, 6, 7, 8]
        cache.store(0, &[0.0; 4], &[5.0, 6.0, 7.0, 8.0]);
        cache.advance();

        // V transposed should be [hidden_dim, seq_len]
        let v_t = cache.get_v_transposed(0);
        assert_eq!(v_t.len(), 4 * 2);
        assert!((v_t[0] - 1.0).abs() < 1e-5); // V_T[0,0]
        assert!((v_t[1] - 5.0).abs() < 1e-5); // V_T[0,1]
    }

    #[test]
    fn test_configure_thread_pool_call() {
        // Should not panic
        let result = configure_thread_pool(4);
        // May succeed or fail depending on environment, but should not panic
        let _ = result;
    }

    #[test]
    fn test_q4k_weight_multiple_output_rows() {
        // Test with multiple output rows
        let data = create_q4k_test_data(4); // 4 rows × 1 super-block each
        let weight = Q4KWeight::new(data, 256, 4).expect("test");

        assert_eq!(weight.out_dim, 4);
        let input = vec![1.0f32; 256];
        let output = weight.matvec(&input).expect("test");
        assert_eq!(output.len(), 4);
    }
}
