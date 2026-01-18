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
//! ## Modules
//!
//! - [`thread`] - Thread configuration for dynamic thread allocation
//! - [`simd`] - SIMD-accelerated operations (matmul, dot, activations)
//! - [`kv_cache`] - Key-value cache for autoregressive generation
//! - [`norm`] - Layer and RMS normalization
//! - [`rope`] - Rotary position embeddings
//! - [`engine`] - Main inference engine implementations
//! - [`quantized`] - Quantized weight storage and inference
//!
//! ## Performance Targets
//!
//! - Use trueno's SIMD Vector::dot for all dot products
//! - Use trueno's Matrix::matmul for weight projections
//! - Target >100 tokens/sec on CPU for 1B models

// Submodules
mod thread;

// Re-exports for public API
pub use thread::{
    configure_optimal_thread_pool, configure_thread_pool, InferenceMode, ThreadConfig,
};

// TODO: These will be moved to submodules incrementally
// For now, include the original content for backwards compatibility

use crate::error::{RealizarError, Result};
use crate::gguf::{GGUFConfig, GGUFTransformer};
use crate::quantize::{fused_q4k_tiled_matvec, QK_K};
use trueno::Vector;

/// Tile size for cache-efficient tiled matmul
const TILE_SIZE: usize = 64;

// ============================================================================
// QUANTIZED WEIGHT STORAGE (Phase 3: Memory Bandwidth Optimization)
// ============================================================================
//
// Keeps weights in quantized format for 8x memory bandwidth reduction.
// Uses fused dequant+dot operations during inference.
// ============================================================================

/// Quantized weight matrix stored in Q4_K format
///
/// Uses fused dequantize+dot operations for 8x memory bandwidth reduction.
/// Each row is stored as raw Q4_K bytes, dequantized on-the-fly during matmul.
#[derive(Clone)]
pub struct Q4KWeight {
    /// Raw Q4_K quantized data
    pub data: Vec<u8>,
    /// Input dimension (number of columns when dequantized)
    pub in_dim: usize,
    /// Output dimension (number of rows)
    pub out_dim: usize,
}

impl Q4KWeight {
    /// Create a new quantized weight from raw Q4_K data
    ///
    /// # Arguments
    ///
    /// * `data` - Raw Q4_K quantized bytes
    /// * `in_dim` - Number of input features (must be multiple of 256)
    /// * `out_dim` - Number of output features
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match the data size
    pub fn new(data: Vec<u8>, in_dim: usize, out_dim: usize) -> Result<Self> {
        // Q4_K uses 256-element super-blocks, each taking 144 bytes
        let blocks_per_row = in_dim.div_ceil(QK_K);
        let bytes_per_row = blocks_per_row * 144; // Q4_K block size
        let expected_bytes = out_dim * bytes_per_row;

        if data.len() != expected_bytes {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Q4KWeight data size {} doesn't match expected {} for {}x{} matrix",
                    data.len(),
                    expected_bytes,
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

    /// Perform matrix-vector multiplication using fused Q4_K operations
    ///
    /// Uses tiled operations for cache efficiency. Each output element is
    /// computed by fused dequantize+dot against the input vector.
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
    /// Returns error if input dimension doesn't match
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

        // Use tiled Q4_K matmul for cache efficiency
        fused_q4k_tiled_matvec(&self.data, input, self.out_dim, self.in_dim, None)
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get equivalent f32 memory usage for comparison
    #[must_use]
    pub fn f32_equivalent_bytes(&self) -> usize {
        self.in_dim * self.out_dim * 4
    }

    /// Get compression ratio vs f32
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.f32_equivalent_bytes() as f32 / self.memory_bytes() as f32
    }
}

// ============================================================================
// SIMD-ACCELERATED OPERATIONS
// ============================================================================

/// SIMD-accelerated matrix-vector multiplication
///
/// Uses trueno's optimized SIMD backend for maximum performance.
/// Falls back to scalar for non-SIMD architectures.
///
/// # Arguments
///
/// * `input` - Input vector of length `in_dim`
/// * `weight` - Weight matrix stored row-major [out_dim × in_dim]
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension
///
/// # Returns
///
/// Output vector of length `out_dim`
#[must_use]
pub fn simd_matmul(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    // Convert to trueno types for SIMD acceleration
    let input_vec = Vector::from_slice(input);

    // Compute each output element using SIMD dot product
    let mut output = vec![0.0; out_dim];

    // Use tiled approach for better cache utilization
    for tile_start in (0..out_dim).step_by(TILE_SIZE) {
        let tile_end = (tile_start + TILE_SIZE).min(out_dim);

        for row in tile_start..tile_end {
            let row_start = row * in_dim;
            let row_end = row_start + in_dim;
            let row_vec = Vector::from_slice(&weight[row_start..row_end]);
            output[row] = input_vec.dot(&row_vec).expect("dot product failed");
        }
    }

    output
}

/// SIMD-accelerated dot product
///
/// Uses trueno's SIMD backend for vectorized computation.
#[inline]
#[must_use]
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    Vector::from_slice(a).dot(&Vector::from_slice(b)).expect("dot product failed")
}

/// SIMD-accelerated vector addition (a += b)
#[inline]
pub fn simd_add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

/// SIMD-accelerated element-wise multiplication (a *= b)
#[inline]
pub fn simd_mul(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x *= y;
    }
}

/// SIMD-accelerated SiLU activation (x * sigmoid(x))
#[inline]
pub fn simd_silu(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

/// SIMD-accelerated GELU activation (approximate)
#[inline]
pub fn simd_gelu(data: &mut [f32]) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
    const COEF: f32 = 0.044715;

    for x in data.iter_mut() {
        let x3 = *x * *x * *x;
        let inner = SQRT_2_OVER_PI * (*x + COEF * x3);
        *x = 0.5 * *x * (1.0 + inner.tanh());
    }
}

/// SIMD-accelerated softmax with numerical stability
///
/// Uses the max-subtraction trick to prevent overflow:
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
pub fn simd_softmax(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for x in data.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for x in data.iter_mut() {
            *x *= inv_sum;
        }
    }
}

// ============================================================================
// KV CACHE
// ============================================================================

/// Key-Value cache for autoregressive generation
///
/// Stores past key and value tensors to avoid recomputation during generation.
/// Enables O(1) per-token computation instead of O(n) for sequence of length n.
#[derive(Clone)]
pub struct KVCache {
    /// Key cache: [num_layers, max_seq_len, kv_dim]
    k_cache: Vec<Vec<f32>>,
    /// Value cache: [num_layers, max_seq_len, kv_dim]
    v_cache: Vec<Vec<f32>>,
    /// Current sequence length
    seq_len: usize,
    /// Hidden dimension per head
    hidden_dim: usize,
}

impl KVCache {
    /// Create a new KV cache
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let k_cache = vec![vec![0.0; max_seq_len * hidden_dim]; num_layers];
        let v_cache = vec![vec![0.0; max_seq_len * hidden_dim]; num_layers];
        Self {
            k_cache,
            v_cache,
            seq_len: 0,
            hidden_dim,
        }
    }

    /// Store a new KV pair for a layer
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let start = self.seq_len * self.hidden_dim;
        let end = start + self.hidden_dim;

        if end <= self.k_cache[layer].len() {
            self.k_cache[layer][start..end].copy_from_slice(k);
            self.v_cache[layer][start..end].copy_from_slice(v);
        }
    }

    /// Advance the sequence position
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Get cached keys for a layer
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        &self.k_cache[layer][..self.seq_len * self.hidden_dim]
    }

    /// Get cached values for a layer
    #[must_use]
    pub fn get_v(&self, layer: usize) -> &[f32] {
        &self.v_cache[layer][..self.seq_len * self.hidden_dim]
    }

    /// Get current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset the cache for a new sequence
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

/// Compute attention with KV cache
///
/// Computes scaled dot-product attention: softmax(QK^T / sqrt(d)) V
/// Uses cached K and V from previous positions.
///
/// # Arguments
///
/// * `q` - Query vector [hidden_dim]
/// * `k_cache` - Cached keys [seq_len × hidden_dim]
/// * `v_cache` - Cached values [seq_len × hidden_dim]
/// * `current_k` - Key for current position [hidden_dim]
/// * `current_v` - Value for current position [hidden_dim]
/// * `num_heads` - Number of attention heads
///
/// # Returns
///
/// Attention output [hidden_dim]
#[must_use]
pub fn attention_with_cache(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    current_k: &[f32],
    current_v: &[f32],
    num_heads: usize,
) -> Vec<f32> {
    let hidden_dim = q.len();
    let head_dim = hidden_dim / num_heads;
    let cache_len = k_cache.len() / hidden_dim;
    let total_len = cache_len + 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; hidden_dim];

    // Process each head independently
    for h in 0..num_heads {
        let head_offset = h * head_dim;
        let q_head = &q[head_offset..head_offset + head_dim];

        // Compute attention scores for all positions
        let mut scores = Vec::with_capacity(total_len);

        // Scores against cached K
        for pos in 0..cache_len {
            let k_start = pos * hidden_dim + head_offset;
            let k_head = &k_cache[k_start..k_start + head_dim];
            let score = simd_dot(q_head, k_head) * scale;
            scores.push(score);
        }

        // Score against current K
        let current_k_head = &current_k[head_offset..head_offset + head_dim];
        scores.push(simd_dot(q_head, current_k_head) * scale);

        // Softmax
        simd_softmax(&mut scores);

        // Weighted sum of V
        let out_head = &mut output[head_offset..head_offset + head_dim];

        for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
            let v_start = pos * hidden_dim + head_offset;
            let v_head = &v_cache[v_start..v_start + head_dim];
            for (i, &v) in v_head.iter().enumerate() {
                out_head[i] += weight * v;
            }
        }

        // Add contribution from current V
        let current_v_head = &current_v[head_offset..head_offset + head_dim];
        let current_weight = scores[cache_len];
        for (i, &v) in current_v_head.iter().enumerate() {
            out_head[i] += current_weight * v;
        }
    }

    output
}

/// Optimized KV cache with contiguous storage
///
/// Uses transposed V storage for better memory access during attention.
#[derive(Clone)]
pub struct OptimizedKVCache {
    /// Key cache: [num_layers][seq_len × hidden_dim]
    k_cache: Vec<Vec<f32>>,
    /// Value cache (transposed): [num_layers][hidden_dim × seq_len]
    v_cache: Vec<Vec<f32>>,
    /// Current sequence length
    seq_len: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl OptimizedKVCache {
    /// Create a new optimized KV cache
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let k_cache = vec![vec![0.0; max_seq_len * hidden_dim]; num_layers];
        let v_cache = vec![vec![0.0; hidden_dim * max_seq_len]; num_layers];
        Self {
            k_cache,
            v_cache,
            seq_len: 0,
            hidden_dim,
            max_seq_len,
        }
    }

    /// Store a new KV pair with transposed V storage
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if self.seq_len >= self.max_seq_len {
            return;
        }

        // Store K in normal format
        let k_start = self.seq_len * self.hidden_dim;
        let k_end = k_start + self.hidden_dim;
        self.k_cache[layer][k_start..k_end].copy_from_slice(k);

        // Store V transposed: v[i] goes to v_cache[i * max_seq_len + seq_len]
        for (i, &val) in v.iter().enumerate() {
            self.v_cache[layer][i * self.max_seq_len + self.seq_len] = val;
        }
    }

    /// Advance the sequence position
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// Get cached keys for a layer
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        &self.k_cache[layer][..self.seq_len * self.hidden_dim]
    }

    /// Get cached values (transposed) for a layer
    #[must_use]
    pub fn get_v_transposed(&self, layer: usize) -> &[f32] {
        &self.v_cache[layer]
    }

    /// Get current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset the cache
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

/// Attention with transposed V cache for better memory access
#[must_use]
pub fn attention_with_transposed_v(
    q: &[f32],
    k_cache: &[f32],
    v_cache_transposed: &[f32],
    current_k: &[f32],
    current_v: &[f32],
    num_heads: usize,
    max_seq_len: usize,
) -> Vec<f32> {
    let hidden_dim = q.len();
    let head_dim = hidden_dim / num_heads;
    let cache_len = k_cache.len() / hidden_dim;
    let total_len = cache_len + 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; hidden_dim];

    for h in 0..num_heads {
        let head_offset = h * head_dim;
        let q_head = &q[head_offset..head_offset + head_dim];

        // Compute attention scores
        let mut scores = Vec::with_capacity(total_len);

        for pos in 0..cache_len {
            let k_start = pos * hidden_dim + head_offset;
            let k_head = &k_cache[k_start..k_start + head_dim];
            scores.push(simd_dot(q_head, k_head) * scale);
        }

        let current_k_head = &current_k[head_offset..head_offset + head_dim];
        scores.push(simd_dot(q_head, current_k_head) * scale);

        simd_softmax(&mut scores);

        // Weighted sum with transposed V (better cache locality)
        let out_head = &mut output[head_offset..head_offset + head_dim];

        for i in 0..head_dim {
            let v_idx = (head_offset + i) * max_seq_len;
            let mut sum = 0.0;
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                sum += weight * v_cache_transposed[v_idx + pos];
            }
            sum += scores[cache_len] * current_v[head_offset + i];
            out_head[i] = sum;
        }
    }

    output
}

// ============================================================================
// NORMALIZATION
// ============================================================================

/// SIMD-accelerated layer normalization
///
/// LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
#[must_use]
pub fn simd_layer_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute mean
    let mean: f32 = input.iter().sum::<f32>() / n as f32;

    // Compute variance
    let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    // Normalize
    let inv_std = 1.0 / (var + eps).sqrt();
    let mut output: Vec<f32> = input.iter().map(|x| (x - mean) * inv_std).collect();

    // Apply affine transformation
    for (i, out) in output.iter_mut().enumerate() {
        *out *= weight[i];
        if let Some(b) = bias {
            *out += b[i];
        }
    }

    output
}

/// SIMD-accelerated RMS normalization
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
#[must_use]
pub fn simd_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute RMS
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Normalize and scale
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * inv_rms * w)
        .collect()
}

// ============================================================================
// ROTARY POSITION EMBEDDINGS (RoPE)
// ============================================================================

/// Apply rotary position embeddings (RoPE)
///
/// RoPE encodes position information by rotating pairs of dimensions.
/// This enables relative position encoding that generalizes to longer sequences.
pub fn apply_rope(x: &mut [f32], hidden_dim: usize, num_heads: usize, position: usize, theta: f32) {
    let head_dim = hidden_dim / num_heads;
    let half_dim = head_dim / 2;

    for h in 0..num_heads {
        let head_offset = h * head_dim;

        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx0 = head_offset + i;
            let idx1 = head_offset + i + half_dim;

            let x0 = x[idx0];
            let x1 = x[idx1];

            x[idx0] = x0 * cos_val - x1 * sin_val;
            x[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

// ============================================================================
// REMAINING CONTENT (to be moved to submodules)
// ============================================================================

// TODO: Move TruenoInferenceEngine, QuantizedTransformerLayer,
// QuantizedInferenceEngine, and tests to their own submodules.
// For now, they remain here for backwards compatibility.

/// Trueno-based inference engine using SIMD primitives
pub struct TruenoInferenceEngine {
    /// Model weights
    transformer: GGUFTransformer,
    /// Model configuration
    config: GGUFConfig,
}

impl TruenoInferenceEngine {
    /// Create a new inference engine from GGUF transformer
    #[must_use]
    pub fn new(transformer: GGUFTransformer) -> Self {
        let config = transformer.config.clone();
        Self {
            transformer,
            config,
        }
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &GGUFConfig {
        &self.config
    }

    /// Run forward pass for a single token
    pub fn forward(&self, token_id: u32, position: usize) -> Result<Vec<f32>> {
        // Token embedding
        let hidden_dim = self.config.hidden_dim;
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        let mut hidden: Vec<f32> = self.transformer.token_embedding[start..end].to_vec();

        // Process through layers
        for layer in &self.transformer.layers {
            // Pre-attention norm
            let normed = simd_rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps);

            // QKV projection
            let qkv_dim = self.config.hidden_dim * 3;
            let qkv = simd_matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);

            // Split QKV
            let mut q = qkv[..hidden_dim].to_vec();
            let mut k = qkv[hidden_dim..hidden_dim * 2].to_vec();
            let v = qkv[hidden_dim * 2..].to_vec();

            // Apply RoPE
            apply_rope(
                &mut q,
                hidden_dim,
                self.config.num_heads,
                position,
                self.config.rope_theta,
            );
            apply_rope(
                &mut k,
                hidden_dim,
                self.config.num_heads,
                position,
                self.config.rope_theta,
            );

            // Self-attention (simplified without cache for single token)
            let scale = 1.0 / ((hidden_dim / self.config.num_heads) as f32).sqrt();
            let attn_output: Vec<f32> = q
                .chunks(hidden_dim / self.config.num_heads)
                .zip(k.chunks(hidden_dim / self.config.num_heads))
                .zip(v.chunks(hidden_dim / self.config.num_heads))
                .flat_map(|((q_h, k_h), v_h)| {
                    let score = simd_dot(q_h, k_h) * scale;
                    v_h.iter().map(|&v| v * score).collect::<Vec<_>>()
                })
                .collect();

            // Output projection
            let attn_proj = simd_matmul(
                &attn_output,
                &layer.attn_output_weight,
                hidden_dim,
                hidden_dim,
            );

            // Residual
            simd_add(&mut hidden, &attn_proj);

            // FFN
            if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                let ffn_normed = simd_rms_norm(&hidden, ffn_norm, self.config.eps);

                // Up projection
                let intermediate_dim = self.config.intermediate_dim;
                let up = simd_matmul(&ffn_normed, &layer.ffn_up_weight, hidden_dim, intermediate_dim);

                // Gate projection (if SwiGLU)
                let activated = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                    let mut gate = simd_matmul(&ffn_normed, gate_weight, hidden_dim, intermediate_dim);
                    simd_silu(&mut gate);
                    gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect()
                } else {
                    let mut act = up;
                    simd_gelu(&mut act);
                    act
                };

                // Down projection
                let ffn_out =
                    simd_matmul(&activated, &layer.ffn_down_weight, intermediate_dim, hidden_dim);

                simd_add(&mut hidden, &ffn_out);
            }
        }

        // Output norm
        let normed = simd_rms_norm(&hidden, &self.transformer.output_norm_weight, self.config.eps);

        // LM head
        let logits = simd_matmul(
            &normed,
            &self.transformer.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );

        Ok(logits)
    }

    /// Generate tokens autoregressively
    pub fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_new_tokens {
            let position = tokens.len() - 1;
            let logits = self.forward(*tokens.last().unwrap(), position)?;

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

// ============================================================================
// QUANTIZED INFERENCE ENGINE
// ============================================================================

/// Quantized transformer layer weights
pub struct QuantizedTransformerLayer {
    /// Attention norm weight
    pub attn_norm_weight: Vec<f32>,
    /// QKV weight (quantized)
    pub qkv_weight: Q4KWeight,
    /// Attention output weight (quantized)
    pub attn_output_weight: Q4KWeight,
    /// FFN norm weight
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN up weight (quantized)
    pub ffn_up_weight: Q4KWeight,
    /// FFN gate weight (quantized, for SwiGLU)
    pub ffn_gate_weight: Option<Q4KWeight>,
    /// FFN down weight (quantized)
    pub ffn_down_weight: Q4KWeight,
}

/// Quantized inference engine with fused Q4_K operations
pub struct QuantizedInferenceEngine {
    /// Model configuration
    config: GGUFConfig,
    /// Token embeddings (kept in f32 for fast lookup)
    token_embedding: Vec<f32>,
    /// Quantized layers
    layers: Vec<QuantizedTransformerLayer>,
    /// Output norm weight
    output_norm_weight: Vec<f32>,
    /// LM head weight (quantized)
    lm_head_weight: Q4KWeight,
}

impl QuantizedInferenceEngine {
    /// Create engine from config and weights
    pub fn new(
        config: GGUFConfig,
        token_embedding: Vec<f32>,
        layers: Vec<QuantizedTransformerLayer>,
        output_norm_weight: Vec<f32>,
        lm_head_weight: Q4KWeight,
    ) -> Self {
        Self {
            config,
            token_embedding,
            layers,
            output_norm_weight,
            lm_head_weight,
        }
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &GGUFConfig {
        &self.config
    }

    /// Run forward pass for a single token using quantized weights
    pub fn forward(&self, token_id: u32, position: usize, cache: &mut KVCache) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // Token embedding (f32 lookup)
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        let mut hidden: Vec<f32> = self.token_embedding[start..end].to_vec();

        // Process through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-attention norm
            let normed = simd_rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps);

            // QKV projection (fused Q4_K matmul)
            let qkv = layer.qkv_weight.matvec(&normed)?;

            // Split QKV
            let q_dim = hidden_dim;
            let k_dim = hidden_dim;
            let mut q = qkv[..q_dim].to_vec();
            let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
            let v = qkv[q_dim + k_dim..].to_vec();

            // Apply RoPE
            apply_rope(
                &mut q,
                hidden_dim,
                self.config.num_heads,
                position,
                self.config.rope_theta,
            );
            apply_rope(
                &mut k,
                hidden_dim,
                self.config.num_heads,
                position,
                self.config.rope_theta,
            );

            // Store KV in cache
            cache.store(layer_idx, &k, &v);

            // Attention with cache
            let attn_output = attention_with_cache(
                &q,
                cache.get_k(layer_idx),
                cache.get_v(layer_idx),
                &k,
                &v,
                self.config.num_heads,
            );

            // Output projection (fused Q4_K matmul)
            let attn_proj = layer.attn_output_weight.matvec(&attn_output)?;

            // Residual
            simd_add(&mut hidden, &attn_proj);

            // FFN
            if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                let ffn_normed = simd_rms_norm(&hidden, ffn_norm, self.config.eps);

                // Up projection (fused Q4_K matmul)
                let up = layer.ffn_up_weight.matvec(&ffn_normed)?;

                // Gate projection (if SwiGLU)
                let activated = if let Some(ref gate) = layer.ffn_gate_weight {
                    let mut gate_out = gate.matvec(&ffn_normed)?;
                    simd_silu(&mut gate_out);
                    gate_out.iter().zip(up.iter()).map(|(g, u)| g * u).collect()
                } else {
                    let mut act = up;
                    simd_gelu(&mut act);
                    act
                };

                // Down projection (fused Q4_K matmul)
                let ffn_out = layer.ffn_down_weight.matvec(&activated)?;

                simd_add(&mut hidden, &ffn_out);
            }
        }

        // Advance cache position
        cache.advance();

        // Output norm
        let normed = simd_rms_norm(&hidden, &self.output_norm_weight, self.config.eps);

        // LM head (fused Q4_K matmul)
        let logits = self.lm_head_weight.matvec(&normed)?;

        Ok(logits)
    }

    /// Generate tokens using quantized inference
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        cache: &mut KVCache,
    ) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();

        // Process prompt tokens
        for (pos, &token) in prompt.iter().enumerate() {
            let _ = self.forward(token, pos, cache)?;
        }

        // Generate new tokens
        for _ in 0..max_new_tokens {
            let position = tokens.len() - 1;
            let logits = self.forward(*tokens.last().unwrap(), position, cache)?;

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Get memory statistics
    #[must_use]
    pub fn memory_stats(&self) -> QuantizedMemoryStats {
        let mut quantized_bytes = 0usize;
        let mut f32_equivalent = 0usize;

        for layer in &self.layers {
            quantized_bytes += layer.qkv_weight.memory_bytes();
            quantized_bytes += layer.attn_output_weight.memory_bytes();
            quantized_bytes += layer.ffn_up_weight.memory_bytes();
            quantized_bytes += layer.ffn_down_weight.memory_bytes();
            if let Some(ref gate) = layer.ffn_gate_weight {
                quantized_bytes += gate.memory_bytes();
            }

            f32_equivalent += layer.qkv_weight.f32_equivalent_bytes();
            f32_equivalent += layer.attn_output_weight.f32_equivalent_bytes();
            f32_equivalent += layer.ffn_up_weight.f32_equivalent_bytes();
            f32_equivalent += layer.ffn_down_weight.f32_equivalent_bytes();
            if let Some(ref gate) = layer.ffn_gate_weight {
                f32_equivalent += gate.f32_equivalent_bytes();
            }
        }

        quantized_bytes += self.lm_head_weight.memory_bytes();
        f32_equivalent += self.lm_head_weight.f32_equivalent_bytes();

        QuantizedMemoryStats {
            quantized_bytes,
            f32_equivalent_bytes: f32_equivalent,
        }
    }
}

/// Memory statistics for quantized model
#[derive(Debug, Clone, Copy)]
pub struct QuantizedMemoryStats {
    /// Actual memory used (quantized)
    pub quantized_bytes: usize,
    /// Equivalent f32 memory
    pub f32_equivalent_bytes: usize,
}

impl std::fmt::Display for QuantizedMemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ratio = self.f32_equivalent_bytes as f64 / self.quantized_bytes as f64;
        write!(
            f,
            "Quantized: {} MB, F32 equivalent: {} MB, Compression: {:.1}x",
            self.quantized_bytes / 1_000_000,
            self.f32_equivalent_bytes / 1_000_000,
            ratio
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // SIMD Operation Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = simd_dot(&a, &b);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_zeros() {
        let a = vec![0.0; 100];
        let b = vec![1.0; 100];
        assert!((simd_dot(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_simd_add() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        simd_add(&mut a, &b);
        assert_eq!(a, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_simd_mul() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        simd_mul(&mut a, &b);
        assert_eq!(a, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_simd_softmax_basic() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_softmax(&mut data);

        // Check sum to 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check ordering preserved
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
    fn test_simd_softmax_single() {
        let mut data = vec![5.0];
        simd_softmax(&mut data);
        assert!((data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_silu() {
        let mut data = vec![0.0, 1.0, -1.0];
        simd_silu(&mut data);
        assert!((data[0]).abs() < 1e-6); // silu(0) = 0
        assert!(data[1] > 0.5); // silu(1) > 0.5
        assert!(data[2] < 0.0); // silu(-1) < 0
    }

    #[test]
    fn test_simd_gelu() {
        let mut data = vec![0.0, 1.0, -1.0];
        simd_gelu(&mut data);
        assert!((data[0]).abs() < 1e-6); // gelu(0) ≈ 0
        assert!(data[1] > 0.8); // gelu(1) ≈ 0.84
        assert!(data[2] < 0.0); // gelu(-1) < 0
    }

    // ------------------------------------------------------------------------
    // KVCache Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4, 128, 512);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_store_and_retrieve() {
        let mut cache = KVCache::new(2, 4, 10);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.store(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_k(0), &k[..]);
        assert_eq!(cache.get_v(0), &v[..]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KVCache::new(2, 4, 10);
        let k = vec![1.0; 4];
        let v = vec![2.0; 4];

        cache.store(0, &k, &v);
        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    // ------------------------------------------------------------------------
    // Normalization Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let result = simd_layer_norm(&input, &weight, None, 1e-5);

        // Mean should be 2.5, normalized values should sum to ~0
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < 1e-5);
    }

    #[test]
    fn test_simd_rms_norm() {
        let input = vec![3.0, 4.0]; // RMS = sqrt((9+16)/2) = sqrt(12.5)
        let weight = vec![1.0, 1.0];
        let result = simd_rms_norm(&input, &weight, 1e-5);

        // Result should be normalized by RMS
        let rms = (12.5f32).sqrt();
        assert!((result[0] - 3.0 / rms).abs() < 1e-5);
        assert!((result[1] - 4.0 / rms).abs() < 1e-5);
    }

    // ------------------------------------------------------------------------
    // RoPE Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_rope_position_zero() {
        let mut x = vec![1.0, 0.0, 1.0, 0.0];
        apply_rope(&mut x, 4, 1, 0, 10000.0);
        // At position 0, cos(0)=1, sin(0)=0, so no change
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_changes_values() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = original.clone();
        apply_rope(&mut x, 4, 1, 10, 10000.0);
        // Values should change at non-zero positions
        assert!(x != original);
    }

    // ------------------------------------------------------------------------
    // Q4KWeight Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_q4k_weight_memory_stats() {
        // Create minimal valid Q4_K data
        let in_dim = 256; // Minimum for Q4_K (one super-block)
        let out_dim = 1;
        let bytes_per_row = 144; // Q4_K block size for 256 elements
        let data = vec![0u8; out_dim * bytes_per_row];

        let weight = Q4KWeight::new(data, in_dim, out_dim).unwrap();
        assert_eq!(weight.memory_bytes(), bytes_per_row);
        assert_eq!(weight.f32_equivalent_bytes(), in_dim * out_dim * 4);
        assert!(weight.compression_ratio() > 1.0);
    }

    #[test]
    fn test_q4k_weight_invalid_size() {
        let data = vec![0u8; 100]; // Wrong size
        let result = Q4KWeight::new(data, 256, 1);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------------
    // QuantizedMemoryStats Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_quantized_memory_stats_display() {
        let stats = QuantizedMemoryStats {
            quantized_bytes: 1_000_000,
            f32_equivalent_bytes: 8_000_000,
        };
        let display = format!("{}", stats);
        assert!(display.contains("8.0x") || display.contains("8x"));
    }
}
