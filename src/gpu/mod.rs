//! GPU Acceleration Module (Phase 4)
//!
//! Provides GPU-accelerated compute primitives via trueno's wgpu backend.
//!
//! ## Architecture
//!
//! ```text
//! +-----------------------+
//! |    GpuCompute API     |  <- Safe public API
//! +-----------------------+
//! |   trueno::GpuBackend  |  <- wgpu-based GPU compute
//! +-----------------------+
//! |   wgpu Device/Queue   |  <- WebGPU abstraction
//! +-----------------------+
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::gpu::{GpuCompute, ComputeBackend};
//!
//! // Auto-select best backend
//! let compute = GpuCompute::auto()?;
//!
//! // GPU matmul
//! let c = compute.matmul(&a, &b, m, k, n)?;
//! ```
//!
//! ## Performance Targets (Refs REALIZAR-PERF-SPEC-001)
//!
//! | Operation | GPU Target | CPU Baseline |
//! |-----------|------------|--------------|
//! | matmul    | 20x faster | 1x           |
//! | tok/s     | ≥100       | ≥25          |

use crate::error::{RealizarError, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::time::Duration;

// PMAT-802: Extracted modules
mod scheduler;
// Types available without cuda feature
pub use scheduler::{GpuModel, GpuModelConfig, GpuGenerateConfig, WeightType, AttentionBuffers};
#[cfg(feature = "cuda")]
pub use scheduler::CudaScheduler;

// ============================================================================
// GPU Buffer Limits (IMP-090, IMP-091)
// ============================================================================

/// Maximum GPU buffer size in bytes (wgpu limit: 256MB)
const MAX_GPU_BUFFER_BYTES: usize = 256 * 1024 * 1024;

/// Vocab size threshold above which we use CPU for embedding/lm_head
/// This is calculated as: MAX_GPU_BUFFER_BYTES / (hidden_dim * sizeof(f32))
/// For hidden_dim=1536: 256MB / (1536 * 4) = 43,690 tokens
/// We use 65536 as a round threshold that works for most models
pub const LARGE_VOCAB_THRESHOLD: usize = 65536;

/// Check if a matrix operation would exceed GPU buffer limits (IMP-090)
///
/// Returns true if the operation should use CPU fallback
#[inline]
#[must_use]
pub fn exceeds_gpu_buffer_limit(elements: usize) -> bool {
    elements * std::mem::size_of::<f32>() > MAX_GPU_BUFFER_BYTES
}

/// Matmul batch operation: (A matrix, B matrix, m rows, k cols, n cols)
pub type MatmulOp = (Vec<f32>, Vec<f32>, usize, usize, usize);

// ============================================================================
// SIMD-accelerated operations (M18 - IMP-038)
// ============================================================================

/// Scalar softmax implementation (baseline for comparison)
///
/// Computes softmax using standard scalar operations.
#[must_use]
pub fn scalar_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// SIMD-accelerated softmax implementation (M18 - IMP-038)
///
/// Uses Trueno's SIMD operations for vectorized computation.
/// Falls back to scalar for unsupported sizes.
#[must_use]
pub fn simd_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Find max using SIMD via trueno
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) - exp is not SIMD accelerated
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();

    // Sum using trueno's SIMD sum
    let exp_vec = trueno::Vector::from_slice(&exp_vals);
    let sum = exp_vec.sum().unwrap_or_else(|_| exp_vals.iter().sum());

    // Normalize
    exp_vals.iter().map(|&e| e / sum).collect()
}

// ============================================================================
// Scalar and SIMD RoPE implementations (M19 - IMP-041)
// ============================================================================

/// Scalar RoPE (Rotary Position Embedding) implementation
///
/// Standard scalar implementation of rotary position embeddings.
/// Input shape: [seq_len * hidden_dim] flattened
#[must_use]
pub fn scalar_rope(input: &[f32], seq_len: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    if input.is_empty() || seq_len == 0 || head_dim == 0 {
        return Vec::new();
    }

    let hidden_dim = input.len() / seq_len;
    let num_heads = hidden_dim / head_dim;
    let mut output = vec![0.0f32; input.len()];

    // Compute RoPE for each position
    for pos in 0..seq_len {
        for head in 0..num_heads {
            let head_start = pos * hidden_dim + head * head_dim;

            // Apply rotary embedding to pairs of elements
            for i in 0..head_dim / 2 {
                let freq = 1.0 / theta.powf((2.0 * i as f32) / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx0 = head_start + i;
                let idx1 = head_start + i + head_dim / 2;

                if idx1 < input.len() {
                    let x0 = input[idx0];
                    let x1 = input[idx1];
                    output[idx0] = x0 * cos_val - x1 * sin_val;
                    output[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    output
}

/// SIMD-accelerated RoPE implementation (M19 - IMP-041)
///
/// Uses Trueno's SIMD operations for vectorized position encoding.
#[must_use]
pub fn simd_rope(input: &[f32], seq_len: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    if input.is_empty() || seq_len == 0 || head_dim == 0 {
        return Vec::new();
    }

    let hidden_dim = input.len() / seq_len;
    let num_heads = hidden_dim / head_dim;
    let half_head = head_dim / 2;

    // Pre-compute frequency table (cache-friendly)
    let mut freqs: Vec<f32> = Vec::with_capacity(half_head);
    for i in 0..half_head {
        freqs.push(1.0 / theta.powf((2.0 * i as f32) / head_dim as f32));
    }

    let mut output = vec![0.0f32; input.len()];

    // Process each position using SIMD operations
    for pos in 0..seq_len {
        // Pre-compute angles for this position
        let angles: Vec<f32> = freqs.iter().map(|&f| pos as f32 * f).collect();
        let cos_vals: Vec<f32> = angles.iter().map(|&a| a.cos()).collect();
        let sin_vals: Vec<f32> = angles.iter().map(|&a| a.sin()).collect();

        // Use trueno vectors for batch operations
        let cos_vec = trueno::Vector::from_slice(&cos_vals);
        let sin_vec = trueno::Vector::from_slice(&sin_vals);

        for head in 0..num_heads {
            let head_start = pos * hidden_dim + head * head_dim;

            // Extract x0 and x1 halves
            let x0_slice = &input[head_start..head_start + half_head];
            let x1_slice = &input[head_start + half_head..head_start + head_dim];

            let x0_vec = trueno::Vector::from_slice(x0_slice);
            let x1_vec = trueno::Vector::from_slice(x1_slice);

            // Compute: out0 = x0 * cos - x1 * sin
            //          out1 = x0 * sin + x1 * cos
            let x0_cos = x0_vec.mul(&cos_vec).unwrap_or_else(|_| x0_vec.clone());
            let x1_sin = x1_vec.mul(&sin_vec).unwrap_or_else(|_| x1_vec.clone());
            let x0_sin = x0_vec.mul(&sin_vec).unwrap_or_else(|_| x0_vec.clone());
            let x1_cos = x1_vec.mul(&cos_vec).unwrap_or_else(|_| x1_vec.clone());

            let out0 = x0_cos
                .sub(&x1_sin)
                .unwrap_or_else(|_| trueno::Vector::from_slice(x0_slice));
            let out1 = x0_sin
                .add(&x1_cos)
                .unwrap_or_else(|_| trueno::Vector::from_slice(x1_slice));

            // Copy results to output
            output[head_start..head_start + half_head].copy_from_slice(out0.as_slice());
            output[head_start + half_head..head_start + head_dim].copy_from_slice(out1.as_slice());
        }
    }

    output
}

// ============================================================================
// Contiguous Attention Buffer (M19 - IMP-040)
// ============================================================================

/// Contiguous memory buffer for attention tensors (M19 - IMP-040)
///
/// Pre-allocates a single contiguous block for Q, K, V, O tensors
/// to reduce memory fragmentation during attention computation.
#[derive(Debug)]
pub struct ContiguousAttentionBuffer {
    /// Single contiguous allocation for all tensors
    data: Vec<f32>,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Number of attention heads (stored for future use)
    #[allow(dead_code)]
    num_heads: usize,
    /// Dimension per head (stored for future use)
    #[allow(dead_code)]
    head_dim: usize,
    /// Size of each tensor (Q, K, V, O have same size)
    tensor_size: usize,
}

impl ContiguousAttentionBuffer {
    /// Create a new contiguous attention buffer
    #[must_use]
    pub fn new(max_seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        let tensor_size = max_seq_len * num_heads * head_dim;
        // Allocate 4x for Q, K, V, O in single contiguous block
        let data = vec![0.0f32; tensor_size * 4];

        Self {
            data,
            max_seq_len,
            num_heads,
            head_dim,
            tensor_size,
        }
    }

    /// Check if buffer is contiguous (always true for this implementation)
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        // Buffer is contiguous by construction
        true
    }

    /// Get views into Q, K, V, O tensors
    #[must_use]
    pub fn get_views(&self) -> (&[f32], &[f32], &[f32], &[f32]) {
        let q_start = 0;
        let k_start = self.tensor_size;
        let v_start = self.tensor_size * 2;
        let o_start = self.tensor_size * 3;

        (
            &self.data[q_start..k_start],
            &self.data[k_start..v_start],
            &self.data[v_start..o_start],
            &self.data[o_start..],
        )
    }

    /// Get mutable views into Q, K, V, O tensors
    pub fn get_views_mut(&mut self) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        let tensor_size = self.tensor_size;

        // Split the data into 4 mutable slices
        let (q, rest) = self.data.split_at_mut(tensor_size);
        let (k, rest) = rest.split_at_mut(tensor_size);
        let (v, o) = rest.split_at_mut(tensor_size);

        (q, k, v, o)
    }

    /// Reset buffer to zeros for reuse
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

// ============================================================================
// Batch Processing & Parallel Execution (M20 - IMP-043/044/045)
// ============================================================================

/// Batch token embedding lookup (M20 - IMP-043)
///
/// Performs vectorized embedding lookup for multiple tokens at once.
/// More efficient than individual lookups due to better memory access patterns.
#[must_use]
pub fn batch_embed(embedding_table: &[f32], tokens: &[usize], hidden_dim: usize) -> Vec<f32> {
    if tokens.is_empty() || embedding_table.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(tokens.len() * hidden_dim);

    // Batch copy embeddings
    for &token in tokens {
        let start_idx = token * hidden_dim;
        let end_idx = start_idx + hidden_dim;
        if end_idx <= embedding_table.len() {
            result.extend_from_slice(&embedding_table[start_idx..end_idx]);
        } else {
            // Pad with zeros for out-of-bounds tokens
            result.extend(std::iter::repeat_n(0.0, hidden_dim));
        }
    }

    result
}

/// Sequential FFN computation (baseline for comparison)
///
/// Standard two-layer feed-forward network: up projection -> activation -> down projection
#[must_use]
pub fn sequential_ffn(
    input: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Up projection: (hidden_dim,) @ (hidden_dim, intermediate_dim) -> (intermediate_dim,)
    let mut intermediate = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            sum += input[j] * w_up[j * intermediate_dim + i];
        }
        // GELU activation
        intermediate[i] =
            sum * 0.5 * (1.0 + (sum * 0.797_884_5 * (1.0 + 0.044_715 * sum * sum)).tanh());
    }

    // Down projection: (intermediate_dim,) @ (intermediate_dim, hidden_dim) -> (hidden_dim,)
    let mut output = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        let mut sum = 0.0f32;
        for j in 0..intermediate_dim {
            sum += intermediate[j] * w_down[j * hidden_dim + i];
        }
        output[i] = sum;
    }

    output
}

/// Parallel FFN computation (M20 - IMP-044)
///
/// Uses rayon parallelism for the down projection matmul.
#[must_use]
pub fn parallel_ffn(
    input: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    if input.is_empty() {
        return Vec::new();
    }

    // Up projection with GELU (sequential - typically smaller)
    let intermediate: Vec<f32> = (0..intermediate_dim)
        .map(|i| {
            let sum: f32 = (0..hidden_dim)
                .map(|j| input[j] * w_up[j * intermediate_dim + i])
                .sum();
            // GELU activation
            sum * 0.5 * (1.0 + (sum * 0.797_884_5 * (1.0 + 0.044_715 * sum * sum)).tanh())
        })
        .collect();

    // Down projection with rayon parallelism
    let output: Vec<f32> = (0..hidden_dim)
        .into_par_iter()
        .map(|i| {
            (0..intermediate_dim)
                .map(|j| intermediate[j] * w_down[j * hidden_dim + i])
                .sum()
        })
        .collect();

    output
}

/// Standard two-pass layer normalization (baseline for comparison)
///
/// First pass computes mean, second pass computes variance.
#[must_use]
pub fn standard_layernorm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let n = input.len() as f32;

    // First pass: compute mean
    let mean: f32 = input.iter().sum::<f32>() / n;

    // Second pass: compute variance
    let variance: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

    let std_dev = (variance + eps).sqrt();

    // Normalize and apply gamma/beta
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normalized = (x - mean) / std_dev;
            normalized * gamma.get(i).copied().unwrap_or(1.0) + beta.get(i).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Fused single-pass layer normalization using Welford's algorithm (M20 - IMP-045)
///
/// Computes mean and variance in a single pass, reducing memory bandwidth.
#[must_use]
pub fn fused_layernorm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let n = input.len();

    // Welford's online algorithm for mean and variance
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;

    for (i, &x) in input.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f32;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / n as f32;
    let std_dev = (variance + eps).sqrt();
    let inv_std = 1.0 / std_dev;

    // Normalize and apply gamma/beta in single pass
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normalized = (x - mean) * inv_std;
            normalized * gamma.get(i).copied().unwrap_or(1.0) + beta.get(i).copied().unwrap_or(0.0)
        })
        .collect()
}

// ============================================================================
// Cache Efficiency & Prefetch (M21 - IMP-046/047/048)
// ============================================================================

/// Cache line size in bytes (typical x86-64)
const CACHE_LINE_SIZE: usize = 64;

/// Cache-aligned buffer for tensor storage (M21 - IMP-046)
///
/// Ensures data is aligned to cache line boundaries (64 bytes) for optimal
/// memory access patterns and avoiding false sharing.
#[derive(Debug)]
pub struct CacheAlignedBuffer {
    /// Underlying storage with extra space for alignment
    data: Vec<f32>,
    /// Offset to aligned start within data
    offset: usize,
    /// Logical length of the buffer
    len: usize,
}

impl CacheAlignedBuffer {
    /// Create a new cache-aligned buffer of the given size
    #[must_use]
    pub fn new(len: usize) -> Self {
        // Allocate extra space to ensure we can align
        // 64 bytes = 16 f32 values
        let align_elements = CACHE_LINE_SIZE / std::mem::size_of::<f32>();
        let extra = align_elements - 1;
        let data = vec![0.0f32; len + extra];

        // Find the aligned offset
        let ptr = data.as_ptr() as usize;
        let misalignment = ptr % CACHE_LINE_SIZE;
        let offset = if misalignment == 0 {
            0
        } else {
            (CACHE_LINE_SIZE - misalignment) / std::mem::size_of::<f32>()
        };

        Self { data, offset, len }
    }

    /// Check if the buffer is aligned to the given boundary
    #[must_use]
    pub fn is_aligned(&self, alignment: usize) -> bool {
        let ptr = self.as_slice().as_ptr() as usize;
        ptr.is_multiple_of(alignment)
    }

    /// Get the logical length of the buffer
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get an immutable slice of the aligned data
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data[self.offset..self.offset + self.len]
    }

    /// Get a mutable slice of the aligned data
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        let offset = self.offset;
        let len = self.len;
        &mut self.data[offset..offset + len]
    }
}

/// Software prefetch hint for read access (M21 - IMP-047)
///
/// Hints to the CPU that data at the given position will be needed soon.
/// This is a no-op on platforms without prefetch support.
#[inline]
pub fn prefetch_read(data: &[f32], position: usize, distance: usize) {
    let prefetch_pos = position + distance;
    if prefetch_pos < data.len() {
        // Use a volatile read to hint the prefetch without actual side effects
        // This is a simplified portable approach; real prefetch uses intrinsics
        // SAFETY: We've verified prefetch_pos is in bounds
        let _ = unsafe { std::ptr::read_volatile(&raw const data[prefetch_pos]) };
    }
}

/// Sequential sum without prefetch (baseline for comparison)
#[must_use]
pub fn sequential_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Sum with software prefetch hints (M21 - IMP-047)
///
/// Uses prefetch hints to reduce cache miss latency for sequential access.
#[must_use]
pub fn sum_with_prefetch(data: &[f32], prefetch_distance: usize) -> f32 {
    let mut sum = 0.0f32;
    let len = data.len();

    for i in 0..len {
        // Prefetch ahead
        if i + prefetch_distance < len {
            prefetch_read(data, i, prefetch_distance);
        }
        sum += data[i];
    }

    sum
}

/// Naive matrix multiplication (baseline for comparison)
///
/// Computes C = A @ B where A is (rows x inner) and B is (inner x cols)
#[must_use]
pub fn naive_matmul(
    mat_a: &[f32],
    mat_b: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0f32;
            for idx in 0..inner {
                sum += mat_a[row * inner + idx] * mat_b[idx * cols + col];
            }
            result[row * cols + col] = sum;
        }
    }

    result
}

/// Cache-blocked matrix multiplication (M21 - IMP-048)
///
/// Uses tiling/blocking to improve cache locality for large matrices.
/// Block size should be chosen to fit in L1/L2 cache.
#[must_use]
#[allow(clippy::many_single_char_names)] // Matrix indices are standard notation
pub fn blocked_matmul(
    mat_a: &[f32],
    mat_b: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    block_size: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; rows * cols];

    // Process in blocks for better cache utilization
    for row_blk in (0..rows).step_by(block_size) {
        let row_end = (row_blk + block_size).min(rows);

        for col_blk in (0..cols).step_by(block_size) {
            let col_end = (col_blk + block_size).min(cols);

            for inner_blk in (0..inner).step_by(block_size) {
                let inner_end = (inner_blk + block_size).min(inner);

                // Inner blocked computation
                for row in row_blk..row_end {
                    for col in col_blk..col_end {
                        let mut sum = result[row * cols + col];
                        for idx in inner_blk..inner_end {
                            sum += mat_a[row * inner + idx] * mat_b[idx * cols + col];
                        }
                        result[row * cols + col] = sum;
                    }
                }
            }
        }
    }

    result
}

// ============================================================================
// Phase 13: Memory Pooling & Arena Allocation (M22)
// ============================================================================

/// Tensor memory pool for reusing buffers during inference (M22 - IMP-049)
///
/// Maintains a pool of pre-allocated buffers organized by size class
/// to reduce allocation overhead during token generation.
#[derive(Debug)]
pub struct TensorPool {
    /// Maximum number of buffers to keep in pool
    capacity: usize,
    /// Available buffers organized by size
    buffers: Vec<Vec<f32>>,
}

impl TensorPool {
    /// Create a new tensor pool with the given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffers: Vec::with_capacity(capacity),
        }
    }

    /// Get the maximum capacity of the pool
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of available buffers in the pool
    #[must_use]
    pub fn available(&self) -> usize {
        self.buffers.len()
    }

    /// Acquire a buffer of the given size
    ///
    /// If a suitable buffer exists in the pool, it will be reused.
    /// Otherwise, a new buffer is allocated.
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        // Look for a buffer of sufficient size
        if let Some(idx) = self.buffers.iter().position(|b| b.capacity() >= size) {
            let mut buffer = self.buffers.swap_remove(idx);
            buffer.resize(size, 0.0);
            buffer
        } else {
            // Allocate new buffer
            vec![0.0f32; size]
        }
    }

    /// Release a buffer back to the pool
    ///
    /// The buffer will be kept for reuse if the pool has capacity.
    pub fn release(&mut self, buffer: Vec<f32>) {
        if self.buffers.len() < self.capacity {
            self.buffers.push(buffer);
        }
        // If at capacity, buffer is dropped
    }

    /// Clear all buffers from the pool
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}

/// Arena allocator for forward pass temporaries (M22 - IMP-050)
///
/// Uses bump allocation for fast, contiguous allocation of tensors
/// during a single forward pass. Reset between passes for reuse.
#[derive(Debug)]
pub struct ForwardArena {
    /// Backing storage
    data: Vec<f32>,
    /// Current allocation offset
    offset: usize,
}

impl ForwardArena {
    /// Create a new arena with the given capacity (in f32 elements)
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0f32; capacity],
            offset: 0,
        }
    }

    /// Get the total capacity of the arena
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Get the current used amount
    #[must_use]
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Allocate a slice of the given size from the arena
    ///
    /// Returns a mutable slice into the arena's backing storage.
    /// Panics if there is insufficient capacity.
    pub fn alloc(&mut self, size: usize) -> &mut [f32] {
        let start = self.offset;
        let end = start + size;

        assert!(
            end <= self.data.len(),
            "ForwardArena: insufficient capacity (need {}, have {})",
            end,
            self.data.len()
        );

        self.offset = end;
        &mut self.data[start..end]
    }

    /// Reset the arena for reuse
    ///
    /// This does not deallocate memory, just resets the offset.
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

/// Scratch buffer for layer-wise intermediate computations (M22 - IMP-051)
///
/// Provides pre-allocated scratch space for each transformer layer,
/// avoiding repeated allocations during inference.
#[derive(Debug)]
pub struct ScratchBuffer {
    /// Number of layers
    num_layers: usize,
    /// Size per layer (in f32 elements)
    layer_size: usize,
    /// Backing storage (contiguous for all layers)
    data: Vec<f32>,
}

impl ScratchBuffer {
    /// Create scratch buffers for the given number of layers
    #[must_use]
    pub fn new(num_layers: usize, layer_size: usize) -> Self {
        Self {
            num_layers,
            layer_size,
            data: vec![0.0f32; num_layers * layer_size],
        }
    }

    /// Get the number of layers
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the size per layer
    #[must_use]
    pub fn layer_size(&self) -> usize {
        self.layer_size
    }

    /// Get the total size of all scratch buffers
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.num_layers * self.layer_size
    }

    /// Get immutable scratch space for a specific layer
    ///
    /// # Panics
    /// Panics if layer_idx >= num_layers
    #[must_use]
    pub fn get_layer(&self, layer_idx: usize) -> &[f32] {
        assert!(
            layer_idx < self.num_layers,
            "ScratchBuffer: layer index {} out of bounds (max {})",
            layer_idx,
            self.num_layers
        );
        let start = layer_idx * self.layer_size;
        let end = start + self.layer_size;
        &self.data[start..end]
    }

    /// Get mutable scratch space for a specific layer
    ///
    /// # Panics
    /// Panics if layer_idx >= num_layers
    pub fn get_layer_mut(&mut self, layer_idx: usize) -> &mut [f32] {
        assert!(
            layer_idx < self.num_layers,
            "ScratchBuffer: layer index {} out of bounds (max {})",
            layer_idx,
            self.num_layers
        );
        let start = layer_idx * self.layer_size;
        let end = start + self.layer_size;
        &mut self.data[start..end]
    }

    /// Reset all scratch buffers to zero
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }
}

// ============================================================================
// Phase 14: Quantized Compute Kernels (M23)
// ============================================================================

/// Quantized dot product for Q4_0 blocks (M23 - IMP-052)
///
/// Computes dot product directly on Q4_0 quantized data without full dequantization.
/// Q4_0 format: 2 bytes (f16 scale) + 16 bytes (32 x 4-bit values)
#[must_use]
pub fn quantized_dot_q4(block_a: &[u8], block_b: &[u8]) -> f32 {
    if block_a.len() < 18 || block_b.len() < 18 {
        return 0.0;
    }

    // Extract scales (f16 little-endian)
    let scale_a = half::f16::from_le_bytes([block_a[0], block_a[1]]).to_f32();
    let scale_b = half::f16::from_le_bytes([block_b[0], block_b[1]]).to_f32();

    // Accumulate dot product over packed 4-bit values
    let mut acc = 0i32;

    for i in 0..16 {
        let byte_a = block_a[2 + i];
        let byte_b = block_b[2 + i];

        // Extract low and high nibbles, center at 8
        let a_lo = (byte_a & 0x0F) as i32 - 8;
        let a_hi = ((byte_a >> 4) & 0x0F) as i32 - 8;
        let b_lo = (byte_b & 0x0F) as i32 - 8;
        let b_hi = ((byte_b >> 4) & 0x0F) as i32 - 8;

        acc += a_lo * b_lo + a_hi * b_hi;
    }

    // Apply combined scale
    (acc as f32) * scale_a * scale_b
}

/// Quantized dot product for Q8_0 blocks (M23 - IMP-052)
///
/// Computes dot product directly on Q8_0 quantized data without full dequantization.
/// Q8_0 format: 2 bytes (f16 scale) + 32 bytes (32 x i8 values)
#[must_use]
pub fn quantized_dot_q8(block_a: &[u8], block_b: &[u8]) -> f32 {
    if block_a.len() < 34 || block_b.len() < 34 {
        return 0.0;
    }

    // Extract scales (f16 little-endian)
    let scale_a = half::f16::from_le_bytes([block_a[0], block_a[1]]).to_f32();
    let scale_b = half::f16::from_le_bytes([block_b[0], block_b[1]]).to_f32();

    // Accumulate dot product over i8 values
    let mut acc = 0i32;

    for i in 0..32 {
        let a_val = block_a[2 + i] as i8 as i32;
        let b_val = block_b[2 + i] as i8 as i32;
        acc += a_val * b_val;
    }

    // Apply combined scale
    (acc as f32) * scale_a * scale_b
}

/// Quantized matrix-vector multiply for Q4_0 weights (M23 - IMP-053)
///
/// Computes y = W @ x where W is Q4_0 quantized without full dequantization.
/// Each row of W consists of ceil(cols/32) Q4_0 blocks.
#[must_use]
pub fn quantized_matvec_q4(weights: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const Q4_BLOCK_SIZE: usize = 18; // 2 bytes scale + 16 bytes data
    const Q4_BLOCK_VALUES: usize = 32;

    let blocks_per_row = cols.div_ceil(Q4_BLOCK_VALUES);
    let row_bytes = blocks_per_row * Q4_BLOCK_SIZE;

    let mut output = vec![0.0f32; rows];

    for (row, out_val) in output.iter_mut().enumerate().take(rows) {
        let row_offset = row * row_bytes;
        let mut acc = 0.0f32;

        for block_idx in 0..blocks_per_row {
            let block_offset = row_offset + block_idx * Q4_BLOCK_SIZE;

            if block_offset + Q4_BLOCK_SIZE > weights.len() {
                break;
            }

            // Extract scale
            let scale =
                half::f16::from_le_bytes([weights[block_offset], weights[block_offset + 1]])
                    .to_f32();

            // Process 32 values in this block
            let input_offset = block_idx * Q4_BLOCK_VALUES;

            for i in 0..16 {
                let byte = weights[block_offset + 2 + i];
                let val_lo = (byte & 0x0F) as i32 - 8;
                let val_hi = ((byte >> 4) & 0x0F) as i32 - 8;

                let in_idx_lo = input_offset + i * 2;
                let in_idx_hi = input_offset + i * 2 + 1;

                if in_idx_lo < cols {
                    acc += (val_lo as f32) * scale * input[in_idx_lo];
                }
                if in_idx_hi < cols {
                    acc += (val_hi as f32) * scale * input[in_idx_hi];
                }
            }
        }

        *out_val = acc;
    }

    output
}

/// Quantized matrix-vector multiply for Q8_0 weights (M23 - IMP-053)
///
/// Computes y = W @ x where W is Q8_0 quantized without full dequantization.
/// Each row of W consists of ceil(cols/32) Q8_0 blocks.
#[must_use]
pub fn quantized_matvec_q8(weights: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const Q8_BLOCK_SIZE: usize = 34; // 2 bytes scale + 32 bytes data
    const Q8_BLOCK_VALUES: usize = 32;

    let blocks_per_row = cols.div_ceil(Q8_BLOCK_VALUES);
    let row_bytes = blocks_per_row * Q8_BLOCK_SIZE;

    let mut output = vec![0.0f32; rows];

    for (row, out_val) in output.iter_mut().enumerate().take(rows) {
        let row_offset = row * row_bytes;
        let mut acc = 0.0f32;

        for block_idx in 0..blocks_per_row {
            let block_offset = row_offset + block_idx * Q8_BLOCK_SIZE;

            if block_offset + Q8_BLOCK_SIZE > weights.len() {
                break;
            }

            // Extract scale
            let scale =
                half::f16::from_le_bytes([weights[block_offset], weights[block_offset + 1]])
                    .to_f32();

            // Process 32 values in this block
            let input_offset = block_idx * Q8_BLOCK_VALUES;

            for i in 0..32 {
                let val = weights[block_offset + 2 + i] as i8 as i32;
                let in_idx = input_offset + i;

                if in_idx < cols {
                    acc += (val as f32) * scale * input[in_idx];
                }
            }
        }

        *out_val = acc;
    }

    output
}

/// Mixed precision accumulator for quantized computations (M23 - IMP-054)
///
/// Accumulates values in f32 precision while processing quantized data,
/// ensuring numerical accuracy during block-wise operations.
#[derive(Debug, Clone, Default)]
pub struct QuantizedAccumulator {
    /// Running sum in f32 precision
    sum: f32,
}

impl QuantizedAccumulator {
    /// Create a new zeroed accumulator
    #[must_use]
    pub fn new() -> Self {
        Self { sum: 0.0 }
    }

    /// Get the current accumulated sum
    #[must_use]
    pub fn sum(&self) -> f32 {
        self.sum
    }

    /// Reset the accumulator to zero
    pub fn reset(&mut self) {
        self.sum = 0.0;
    }

    /// Add a scaled value to the accumulator
    #[inline]
    pub fn add_scaled(&mut self, value: f32, scale: f32) {
        self.sum += value * scale;
    }

    /// Add a block contribution (block_sum * block_scale)
    #[inline]
    pub fn add_block(&mut self, block_sum: f32, block_scale: f32) {
        self.sum += block_sum * block_scale;
    }
}

// =============================================================================
// M24: Streaming & Pipelining (Phase 15)
// =============================================================================

/// Double buffer for overlapping compute with memory operations (M24 - IMP-055)
///
/// Enables loading next layer weights while computing current layer.
/// Front buffer is read-only for compute, back buffer is writable for loading.
#[derive(Debug)]
pub struct DoubleBuffer<T> {
    front: Vec<T>,
    back: Vec<T>,
}

impl<T: Default + Clone> DoubleBuffer<T> {
    /// Create a new double buffer with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            front: vec![T::default(); capacity],
            back: vec![T::default(); capacity],
        }
    }

    /// Get the capacity of each buffer
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.front.len()
    }

    /// Get immutable reference to front buffer (for reading/compute)
    #[must_use]
    pub fn front(&self) -> &[T] {
        &self.front
    }

    /// Get mutable reference to back buffer (for writing/loading)
    pub fn back_mut(&mut self) -> &mut [T] {
        &mut self.back
    }

    /// Swap front and back buffers
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.front, &mut self.back);
    }
}

/// Chunked token processor for improved cache utilization (M24 - IMP-056)
///
/// Processes tokens in configurable chunks to improve memory locality
/// and cache efficiency during batch processing.
#[derive(Debug, Clone)]
pub struct ChunkedProcessor {
    chunk_size: usize,
}

impl ChunkedProcessor {
    /// Create a new chunked processor with given chunk size
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    /// Get the chunk size
    #[must_use]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Calculate number of chunks needed for given input length
    #[must_use]
    pub fn num_chunks(&self, total_len: usize) -> usize {
        if total_len == 0 {
            return 0;
        }
        total_len.div_ceil(self.chunk_size)
    }

    /// Get bounds (start, end) for a specific chunk index
    #[must_use]
    pub fn chunk_bounds(&self, chunk_idx: usize, total_len: usize) -> (usize, usize) {
        let start = chunk_idx * self.chunk_size;
        let end = (start + self.chunk_size).min(total_len);
        (start, end)
    }

    /// Process data in chunks, accumulating results
    pub fn process_chunks<T, F>(&self, data: &[T], mut process: F) -> f32
    where
        F: FnMut(&[T]) -> f32,
    {
        let mut total = 0.0f32;
        let num_chunks = self.num_chunks(data.len());

        for chunk_idx in 0..num_chunks {
            let (start, end) = self.chunk_bounds(chunk_idx, data.len());
            total += process(&data[start..end]);
        }

        total
    }
}

/// Inference pipeline stages (M24 - IMP-057)
///
/// Represents the different stages of transformer inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuPipelineStage {
    /// Token embedding lookup
    Embed = 0,
    /// Self-attention computation
    Attention = 1,
    /// Feed-forward network
    FFN = 2,
    /// Output projection and sampling
    Output = 3,
}

/// Inference pipeline coordinator (M24 - IMP-057)
///
/// Manages multi-stage inference pipeline with timing tracking.
#[derive(Debug)]
pub struct InferencePipeline {
    num_stages: usize,
    stage_times: std::collections::HashMap<GpuPipelineStage, f32>,
}

impl InferencePipeline {
    /// Create a new pipeline with given number of stages
    #[must_use]
    pub fn new(num_stages: usize) -> Self {
        Self {
            num_stages,
            stage_times: std::collections::HashMap::new(),
        }
    }

    /// Get number of stages in the pipeline
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Record timing for a pipeline stage (in milliseconds)
    pub fn record_stage_time(&mut self, stage: GpuPipelineStage, time_ms: f32) {
        self.stage_times.insert(stage, time_ms);
    }

    /// Get total pipeline latency (sum of all stage times)
    #[must_use]
    pub fn total_latency(&self) -> f32 {
        self.stage_times.values().sum()
    }

    /// Get breakdown of stage timings
    #[must_use]
    pub fn stage_breakdown(&self) -> &std::collections::HashMap<GpuPipelineStage, f32> {
        &self.stage_times
    }

    /// Reset pipeline for new forward pass
    pub fn reset(&mut self) {
        self.stage_times.clear();
    }
}

// =============================================================================
// M25: Token Batching & Speculative Decoding (Phase 16)
// =============================================================================

/// Token batch accumulator for batched processing (M25 - IMP-058)
///
/// Accumulates tokens until batch is full, then returns for processing.
/// Improves throughput by processing multiple tokens together.
#[derive(Debug)]
pub struct TokenBatch {
    tokens: Vec<usize>,
    capacity: usize,
}

impl TokenBatch {
    /// Create a new token batch with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Get the batch capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current number of tokens in batch
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if batch is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Check if batch is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.tokens.len() >= self.capacity
    }

    /// Push a token to the batch
    ///
    /// Returns `Some(tokens)` when batch becomes full, `None` otherwise.
    pub fn push(&mut self, token: usize) -> Option<Vec<usize>> {
        self.tokens.push(token);
        if self.is_full() {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Flush and return all tokens, clearing the batch
    pub fn flush(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.tokens)
    }
}

/// Candidate token for speculative decoding
#[derive(Debug, Clone)]
struct SpeculativeCandidate {
    token: usize,
    /// Confidence score (stored for future use in acceptance thresholds)
    #[allow(dead_code)]
    confidence: f32,
}

/// Speculative token buffer for speculative decoding (M25 - IMP-059)
///
/// Manages candidate tokens generated speculatively, allowing verification
/// against actual model outputs for acceptance or rejection.
#[derive(Debug)]
pub struct SpeculativeBuffer {
    candidates: Vec<SpeculativeCandidate>,
    capacity: usize,
}

impl SpeculativeBuffer {
    /// Create a new speculative buffer with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Get the buffer capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current number of candidates
    #[must_use]
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Add a candidate token with confidence score
    pub fn add_candidate(&mut self, token: usize, confidence: f32) {
        if self.candidates.len() < self.capacity {
            self.candidates
                .push(SpeculativeCandidate { token, confidence });
        }
    }

    /// Verify candidates against actual tokens
    ///
    /// Returns (num_accepted, rejection_index) where rejection_index is
    /// the first index where mismatch occurred, or None if all matched.
    #[must_use]
    pub fn verify(&self, actual_tokens: &[usize]) -> (usize, Option<usize>) {
        let mut accepted = 0;
        for (i, candidate) in self.candidates.iter().enumerate() {
            if i < actual_tokens.len() && candidate.token == actual_tokens[i] {
                accepted += 1;
            } else {
                return (accepted, Some(i));
            }
        }
        (accepted, None)
    }

    /// Accept first n candidates, removing them from buffer
    pub fn accept(&mut self, n: usize) {
        if n >= self.candidates.len() {
            self.candidates.clear();
        } else {
            self.candidates.drain(0..n);
        }
    }

    /// Reject all remaining candidates
    pub fn reject(&mut self) {
        self.candidates.clear();
    }
}

/// Batch ID for tracking inference batches
pub type BatchId = u64;

/// Inference batch scheduler for coordinating batched processing (M25 - IMP-060)
///
/// Manages pending and completed batches, allowing asynchronous batch
/// submission and result retrieval.
#[derive(Debug)]
pub struct InferenceBatchScheduler {
    next_id: BatchId,
    pending: std::collections::HashMap<BatchId, Vec<usize>>,
    completed: std::collections::VecDeque<(BatchId, Vec<usize>)>,
}

impl InferenceBatchScheduler {
    /// Create a new inference batch scheduler
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 0,
            pending: std::collections::HashMap::new(),
            completed: std::collections::VecDeque::new(),
        }
    }

    /// Get count of pending batches
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get count of completed batches
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Submit a batch for processing
    ///
    /// Returns a unique batch ID for tracking.
    pub fn submit(&mut self, tokens: Vec<usize>) -> BatchId {
        let id = self.next_id;
        self.next_id += 1;
        self.pending.insert(id, tokens);
        id
    }

    /// Mark a batch as complete with results
    pub fn complete(&mut self, batch_id: BatchId, results: Vec<usize>) {
        self.pending.remove(&batch_id);
        self.completed.push_back((batch_id, results));
    }

    /// Poll for a completed batch
    ///
    /// Returns `Some((batch_id, results))` if a batch is ready, `None` otherwise.
    pub fn poll(&mut self) -> Option<(BatchId, Vec<usize>)> {
        self.completed.pop_front()
    }

    /// Drain all completed batches
    pub fn drain(&mut self) -> Vec<(BatchId, Vec<usize>)> {
        self.completed.drain(..).collect()
    }
}

impl Default for InferenceBatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// M26: Async I/O & Event-Driven Processing (Phase 17)
// =============================================================================

/// Async request queue for non-blocking request handling (M26 - IMP-061)
///
/// Provides a bounded FIFO queue for inference requests with backpressure
/// support via try-based operations.
#[derive(Debug)]
pub struct AsyncRequestQueue<T> {
    items: std::collections::VecDeque<T>,
    capacity: usize,
}

impl<T> AsyncRequestQueue<T> {
    /// Create a new async request queue with specified capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            items: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Get queue capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current queue length
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Check if queue is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }

    /// Try to push an item to the queue
    ///
    /// Returns `true` if successful, `false` if queue is full (backpressure).
    pub fn try_push(&mut self, item: T) -> bool {
        if self.is_full() {
            false
        } else {
            self.items.push_back(item);
            true
        }
    }

    /// Try to pop an item from the queue
    ///
    /// Returns `Some(item)` if available, `None` if queue is empty.
    pub fn try_pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }
}

/// Type alias for inference completion handler
pub type InferenceCompletionHandler = Box<dyn Fn(u64, &[usize]) + Send + Sync>;

/// Event notifier for inference completion (M26 - IMP-062)
///
/// Allows registration of handlers that are called when inference completes.
pub struct InferenceEventNotifier {
    handlers: Vec<InferenceCompletionHandler>,
}

impl std::fmt::Debug for InferenceEventNotifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEventNotifier")
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

impl InferenceEventNotifier {
    /// Create a new event notifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Get count of registered handlers
    #[must_use]
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Register a completion handler
    ///
    /// Handler receives (request_id, output_tokens) when inference completes.
    pub fn register(&mut self, handler: InferenceCompletionHandler) {
        self.handlers.push(handler);
    }

    /// Notify all handlers of completion
    ///
    /// Calls each registered handler with the request ID and output tokens.
    pub fn notify(&self, request_id: u64, tokens: &[usize]) {
        for handler in &self.handlers {
            handler(request_id, tokens);
        }
    }

    /// Clear all registered handlers
    pub fn clear(&mut self) {
        self.handlers.clear();
    }
}

impl Default for InferenceEventNotifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Request ID type for timeout tracking
pub type RequestId = u64;

/// Timeout manager for request deadline tracking (M26 - IMP-063)
///
/// Tracks request deadlines and identifies expired requests.
#[derive(Debug)]
pub struct TimeoutManager {
    deadlines: std::collections::HashMap<RequestId, std::time::Instant>,
}

impl TimeoutManager {
    /// Create a new timeout manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            deadlines: std::collections::HashMap::new(),
        }
    }

    /// Get count of active timeout registrations
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.deadlines.len()
    }

    /// Register a timeout for a request
    ///
    /// The deadline is the absolute time at which the request should timeout.
    pub fn register(&mut self, request_id: RequestId, deadline: std::time::Instant) {
        self.deadlines.insert(request_id, deadline);
    }

    /// Remove timeout registration for a request
    ///
    /// Use when request completes before timeout.
    pub fn remove(&mut self, request_id: RequestId) {
        self.deadlines.remove(&request_id);
    }

    /// Check for expired requests and remove them
    ///
    /// Returns list of request IDs that have timed out.
    pub fn check_expired(&mut self) -> Vec<RequestId> {
        let now = std::time::Instant::now();
        let expired: Vec<RequestId> = self
            .deadlines
            .iter()
            .filter(|(_, &deadline)| now >= deadline)
            .map(|(&id, _)| id)
            .collect();

        for id in &expired {
            self.deadlines.remove(id);
        }

        expired
    }
}

impl Default for TimeoutManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// M27: Request Scheduling & Resource Management (Phase 18)
// =============================================================================

/// Priority level type (higher = more important)
pub type Priority = u32;

/// Priority request wrapper for priority queue (M27 - IMP-064)
#[derive(Debug, Clone)]
pub struct PriorityRequest<T> {
    priority: Priority,
    sequence: u64, // For FIFO ordering within same priority
    data: T,
}

impl<T> PriorityRequest<T> {
    /// Create a new priority request
    #[must_use]
    pub fn new(priority: Priority, data: T) -> Self {
        Self {
            priority,
            sequence: 0, // Will be set by queue
            data,
        }
    }

    /// Get the priority level
    #[must_use]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Get reference to request data
    #[must_use]
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Consume and return the data
    #[must_use]
    pub fn into_data(self) -> T {
        self.data
    }
}

/// Priority request queue for request scheduling (M27 - IMP-064)
///
/// Implements priority-based scheduling with FIFO ordering for same-priority requests.
#[derive(Debug)]
pub struct PriorityRequestQueue<T> {
    items: Vec<PriorityRequest<T>>,
    next_sequence: u64,
}

impl<T> PriorityRequestQueue<T> {
    /// Create a new priority request queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            next_sequence: 0,
        }
    }

    /// Get number of items in queue
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Enqueue a request with priority
    pub fn enqueue(&mut self, mut request: PriorityRequest<T>) {
        request.sequence = self.next_sequence;
        self.next_sequence += 1;
        self.items.push(request);
    }

    /// Dequeue the highest priority request
    ///
    /// Returns the request with highest priority. For equal priorities,
    /// returns the earliest enqueued (FIFO).
    pub fn dequeue_highest(&mut self) -> Option<PriorityRequest<T>> {
        if self.items.is_empty() {
            return None;
        }

        // Find index of highest priority (and earliest sequence for ties)
        let mut best_idx = 0;
        for (i, item) in self.items.iter().enumerate().skip(1) {
            let best = &self.items[best_idx];
            if item.priority > best.priority
                || (item.priority == best.priority && item.sequence < best.sequence)
            {
                best_idx = i;
            }
        }

        Some(self.items.swap_remove(best_idx))
    }
}

impl<T> Default for PriorityRequestQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Token bucket rate limiter for throughput control (M27 - IMP-065)
///
/// Implements token bucket algorithm with configurable rate and burst capacity.
#[derive(Debug)]
pub struct TokenRateLimiter {
    tokens: u32,
    capacity: u32,
    rate: f64, // tokens per second
    last_refill: std::time::Instant,
}

impl TokenRateLimiter {
    /// Create a new rate limiter
    ///
    /// # Arguments
    /// * `rate` - Tokens per second to refill
    /// * `burst_capacity` - Maximum tokens that can accumulate
    #[must_use]
    pub fn new(rate: f64, burst_capacity: u32) -> Self {
        Self {
            tokens: burst_capacity, // Start full
            capacity: burst_capacity,
            rate,
            last_refill: std::time::Instant::now(),
        }
    }

    /// Get current available tokens
    #[must_use]
    pub fn tokens_available(&self) -> u32 {
        self.tokens
    }

    /// Try to acquire tokens
    ///
    /// Returns `true` if tokens were acquired, `false` if insufficient tokens.
    pub fn try_acquire(&mut self, count: u32) -> bool {
        if self.tokens >= count {
            self.tokens -= count;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    ///
    /// Call periodically to add tokens at the configured rate.
    pub fn refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = (elapsed * self.rate) as u32;

        if new_tokens > 0 {
            self.tokens = (self.tokens + new_tokens).min(self.capacity);
            self.last_refill = now;
        }
    }
}

/// Allocation ID for resource tracking
pub type AllocationId = u64;

/// Resource allocation record
#[derive(Debug, Clone)]
struct ResourceAllocation {
    memory: u64,
    compute: u32,
}

/// Resource usage tracker for memory and compute (M27 - IMP-066)
///
/// Tracks resource allocations and provides utilization metrics.
#[derive(Debug)]
pub struct ResourceTracker {
    memory_capacity: u64,
    compute_capacity: u32,
    memory_used: u64,
    compute_used: u32,
    allocations: std::collections::HashMap<AllocationId, ResourceAllocation>,
    next_id: AllocationId,
}

impl ResourceTracker {
    /// Create a new resource tracker
    ///
    /// # Arguments
    /// * `memory_capacity` - Total memory capacity in bytes
    /// * `compute_capacity` - Total compute capacity (0-100 percentage)
    #[must_use]
    pub fn new(memory_capacity: u64, compute_capacity: u32) -> Self {
        Self {
            memory_capacity,
            compute_capacity,
            memory_used: 0,
            compute_used: 0,
            allocations: std::collections::HashMap::new(),
            next_id: 0,
        }
    }

    /// Get current memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> u64 {
        self.memory_used
    }

    /// Get current compute usage (0-100)
    #[must_use]
    pub fn compute_usage(&self) -> u32 {
        self.compute_used
    }

    /// Check if allocation is possible
    #[must_use]
    pub fn can_allocate(&self, memory: u64, compute: u32) -> bool {
        self.memory_used + memory <= self.memory_capacity
            && self.compute_used + compute <= self.compute_capacity
    }

    /// Allocate resources
    ///
    /// Returns allocation ID if successful, None if insufficient resources.
    pub fn allocate(&mut self, memory: u64, compute: u32) -> Option<AllocationId> {
        if !self.can_allocate(memory, compute) {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        self.memory_used += memory;
        self.compute_used += compute;
        self.allocations
            .insert(id, ResourceAllocation { memory, compute });

        Some(id)
    }

    /// Release allocated resources
    pub fn release(&mut self, id: AllocationId) {
        if let Some(alloc) = self.allocations.remove(&id) {
            self.memory_used = self.memory_used.saturating_sub(alloc.memory);
            self.compute_used = self.compute_used.saturating_sub(alloc.compute);
        }
    }

    /// Get usage as percentages
    ///
    /// Returns (memory_percentage, compute_percentage)
    #[must_use]
    pub fn usage_percentage(&self) -> (f64, f64) {
        let mem_pct = if self.memory_capacity > 0 {
            (self.memory_used as f64 / self.memory_capacity as f64) * 100.0
        } else {
            0.0
        };
        let compute_pct = if self.compute_capacity > 0 {
            (self.compute_used as f64 / self.compute_capacity as f64) * 100.0
        } else {
            0.0
        };
        (mem_pct, compute_pct)
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        // Default: 8GB memory, 100% compute
        Self::new(8 * 1024 * 1024 * 1024, 100)
    }
}

// =============================================================================
// M28: Metrics & Health Monitoring (Phase 19)
// =============================================================================

/// Inference metrics collector (M28 - IMP-067)
///
/// Collects and aggregates inference performance metrics including
/// latency distribution and throughput.
#[derive(Debug)]
pub struct InferenceMetrics {
    latencies: Vec<std::time::Duration>,
    total_tokens: u64,
    start_time: std::time::Instant,
}

impl InferenceMetrics {
    /// Create a new inference metrics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            latencies: Vec::new(),
            total_tokens: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Get total number of recorded inferences
    #[must_use]
    pub fn total_inferences(&self) -> usize {
        self.latencies.len()
    }

    /// Get total number of tokens processed
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Record an inference with its latency and token count
    pub fn record_inference(&mut self, latency: std::time::Duration, tokens: usize) {
        self.latencies.push(latency);
        self.total_tokens += tokens as u64;
    }

    /// Get latency at given percentile (0-100)
    ///
    /// Returns None if no inferences recorded.
    #[must_use]
    pub fn latency_percentile(&self, percentile: u8) -> Option<std::time::Duration> {
        if self.latencies.is_empty() {
            return None;
        }

        let mut sorted = self.latencies.clone();
        sorted.sort();

        let idx = ((percentile as usize) * sorted.len() / 100).min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Calculate throughput in tokens per second
    #[must_use]
    pub fn throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_tokens as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.latencies.clear();
        self.total_tokens = 0;
        self.start_time = std::time::Instant::now();
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for health check function
pub type HealthCheckFn = Box<dyn Fn() -> bool + Send + Sync>;

/// Health checker for system components (M28 - IMP-068)
///
/// Monitors health status of system components via registered check functions.
pub struct HealthChecker {
    checks: Vec<(String, HealthCheckFn)>,
    last_results: std::collections::HashMap<String, bool>,
}

impl std::fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HealthChecker")
            .field("check_count", &self.checks.len())
            .field("last_results", &self.last_results)
            .finish()
    }
}

impl HealthChecker {
    /// Create a new health checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            last_results: std::collections::HashMap::new(),
        }
    }

    /// Get number of registered checks
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Register a health check function
    pub fn register_check(&mut self, name: &str, check: HealthCheckFn) {
        self.checks.push((name.to_string(), check));
    }

    /// Run all health checks and return results
    pub fn check_all(&mut self) -> std::collections::HashMap<String, bool> {
        let mut results = std::collections::HashMap::new();
        for (name, check) in &self.checks {
            let healthy = check();
            results.insert(name.clone(), healthy);
        }
        self.last_results.clone_from(&results);
        results
    }

    /// Check if system is overall healthy (all checks pass)
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        if self.checks.is_empty() {
            return true;
        }
        self.last_results.values().all(|&v| v)
    }

    /// Clear all registered checks
    pub fn clear(&mut self) {
        self.checks.clear();
        self.last_results.clear();
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for shutdown handler function
pub type ShutdownHandlerFn = Box<dyn Fn() + Send + Sync>;

/// Graceful shutdown coordinator (M28 - IMP-069)
///
/// Coordinates shutdown sequence with request draining and handler callbacks.
pub struct ShutdownCoordinator {
    shutting_down: bool,
    pending_requests: u32,
    handlers: Vec<ShutdownHandlerFn>,
}

impl std::fmt::Debug for ShutdownCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShutdownCoordinator")
            .field("shutting_down", &self.shutting_down)
            .field("pending_requests", &self.pending_requests)
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    #[must_use]
    pub fn new() -> Self {
        Self {
            shutting_down: false,
            pending_requests: 0,
            handlers: Vec::new(),
        }
    }

    /// Check if shutdown has been initiated
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    /// Get number of pending requests
    #[must_use]
    pub fn pending_requests(&self) -> u32 {
        self.pending_requests
    }

    /// Get number of registered handlers
    #[must_use]
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Register a shutdown handler
    pub fn register_handler(&mut self, handler: ShutdownHandlerFn) {
        self.handlers.push(handler);
    }

    /// Mark that a request has started
    pub fn request_started(&mut self) {
        self.pending_requests += 1;
    }

    /// Mark that a request has completed
    pub fn request_completed(&mut self) {
        self.pending_requests = self.pending_requests.saturating_sub(1);
    }

    /// Initiate shutdown sequence
    ///
    /// Calls all registered handlers.
    pub fn initiate_shutdown(&mut self) {
        if self.shutting_down {
            return;
        }
        self.shutting_down = true;

        // Call all handlers
        for handler in &self.handlers {
            handler();
        }
    }

    /// Check if shutdown is complete (initiated + no pending requests)
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.shutting_down && self.pending_requests == 0
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    /// GPU compute via trueno's wgpu backend
    Gpu,
    /// CPU compute (fallback)
    Cpu,
    /// Auto-select best available backend
    #[default]
    Auto,
}

/// GPU compute context
///
/// Provides GPU-accelerated operations with automatic fallback to CPU
/// when GPU is not available.
pub struct GpuCompute {
    backend: ComputeBackend,
    gpu: Option<trueno::backends::gpu::GpuBackend>,
}

impl GpuCompute {
    /// Create GPU compute context with auto-detected backend
    ///
    /// Attempts to initialize GPU backend, falls back to CPU if unavailable.
    ///
    /// # Errors
    ///
    /// Returns error if both GPU and CPU initialization fail (should not happen).
    pub fn auto() -> Result<Self> {
        Self::new(ComputeBackend::Auto)
    }

    /// Create GPU compute context with specified backend
    ///
    /// # Arguments
    ///
    /// * `backend` - Backend selection (Gpu, Cpu, or Auto)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `Gpu` backend requested but GPU is not available
    /// - Backend initialization fails
    pub fn new(backend: ComputeBackend) -> Result<Self> {
        match backend {
            ComputeBackend::Gpu => {
                if trueno::backends::gpu::GpuBackend::is_available() {
                    Ok(Self {
                        backend: ComputeBackend::Gpu,
                        gpu: Some(trueno::backends::gpu::GpuBackend::new()),
                    })
                } else {
                    Err(RealizarError::GpuError {
                        reason: "GPU not available".to_string(),
                    })
                }
            },
            ComputeBackend::Cpu => Ok(Self {
                backend: ComputeBackend::Cpu,
                gpu: None,
            }),
            ComputeBackend::Auto => {
                if trueno::backends::gpu::GpuBackend::is_available() {
                    Ok(Self {
                        backend: ComputeBackend::Gpu,
                        gpu: Some(trueno::backends::gpu::GpuBackend::new()),
                    })
                } else {
                    Ok(Self {
                        backend: ComputeBackend::Cpu,
                        gpu: None,
                    })
                }
            },
        }
    }

    /// Check if GPU backend is active
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        self.backend == ComputeBackend::Gpu && self.gpu.is_some()
    }

    /// Get active backend type
    #[must_use]
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    /// GPU-accelerated matrix multiplication
    ///
    /// Computes `C = A @ B` where:
    /// - A is `[m, k]`
    /// - B is `[k, n]`
    /// - C is `[m, n]`
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix as flat f32 slice, row-major `[m, k]`
    /// * `b` - Right matrix as flat f32 slice, row-major `[k, n]`
    /// * `m` - Rows in A and C
    /// * `k` - Cols in A, rows in B
    /// * `n` - Cols in B and C
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input dimensions don't match
    /// - GPU compute fails
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Validate dimensions
        if a.len() != m * k {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Matrix A size {} doesn't match m*k={}*{}={}",
                    a.len(),
                    m,
                    k,
                    m * k
                ),
            });
        }
        if b.len() != k * n {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Matrix B size {} doesn't match k*n={}*{}={}",
                    b.len(),
                    k,
                    n,
                    k * n
                ),
            });
        }

        if let Some(gpu) = &mut self.gpu {
            // GPU path
            #[allow(clippy::implicit_clone)]
            gpu.matmul(a, b, m, k, n)
                .map_err(|e| RealizarError::GpuError {
                    reason: e.to_string(),
                })
        } else {
            // CPU fallback: naive matmul
            Ok(cpu_matmul(a, b, m, k, n))
        }
    }

    /// GPU-accelerated matrix multiplication with Tensor input/output
    ///
    /// # Arguments
    ///
    /// * `a` - Left tensor `[m, k]`
    /// * `b` - Right tensor `[k, n]`
    ///
    /// # Errors
    ///
    /// Returns error if tensors are not 2D or dimensions don't match.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_tensor(&mut self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(RealizarError::InvalidShape {
                reason: "matmul_tensor requires 2D tensors".to_string(),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let k2 = b_shape[0];
        let n = b_shape[1];

        if k != k2 {
            return Err(RealizarError::InvalidShape {
                reason: format!("Inner dimensions don't match: A[{m},{k}] @ B[{k2},{n}]"),
            });
        }

        let result = self.matmul(a.data(), b.data(), m, k, n)?;
        Tensor::from_vec(vec![m, n], result)
    }

    /// GPU-accelerated vector dot product
    ///
    /// # Errors
    ///
    /// Returns error if vectors have different lengths or GPU compute fails.
    pub fn dot(&mut self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!("Vector lengths don't match: {} vs {}", a.len(), b.len()),
            });
        }

        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.dot(a, b).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            // CPU fallback
            Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
        }
    }

    /// GPU-accelerated ReLU activation
    ///
    /// # Errors
    ///
    /// Returns error if GPU compute fails.
    pub fn relu(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.relu(input).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            Ok(input.iter().map(|&x| x.max(0.0)).collect())
        }
    }

    /// GPU-accelerated sigmoid activation
    ///
    /// # Errors
    ///
    /// Returns error if GPU compute fails.
    pub fn sigmoid(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.sigmoid(input).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            Ok(input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
        }
    }
}

/// CPU fallback matmul implementation
#[allow(clippy::many_single_char_names)]
pub(crate) fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // For m=1 (vector-matrix multiply), use optimized path
    if m == 1 {
        return cpu_vector_matmul(a, b, k, n);
    }

    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// IMP-098: Parallelized vector-matrix multiply: a[1,k] @ b[k,n] -> c[1,n]
///
/// Uses parallel output chunks for multi-core utilization.
/// Each thread accumulates its chunk of outputs independently.
#[allow(clippy::many_single_char_names)]
fn cpu_vector_matmul(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    use rayon::prelude::*;

    // For small n, use sequential (avoids rayon overhead)
    if n < 2048 {
        return cpu_vector_matmul_seq(a, b, k, n);
    }

    // Parallel over output chunks
    const CHUNK_SIZE: usize = 1024;
    let num_chunks = n.div_ceil(CHUNK_SIZE);

    let chunks: Vec<Vec<f32>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(n);
            let chunk_len = end - start;
            let mut chunk_c = vec![0.0f32; chunk_len];

            // Accumulate this chunk of outputs
            for (p, &a_val) in a.iter().enumerate() {
                let row_start = p * n + start;
                let row = &b[row_start..row_start + chunk_len];
                for (j, &b_val) in row.iter().enumerate() {
                    chunk_c[j] += a_val * b_val;
                }
            }
            chunk_c
        })
        .collect();

    // Flatten chunks into result
    chunks.into_iter().flatten().collect()
}

/// Sequential fallback for small outputs
#[allow(clippy::many_single_char_names)]
fn cpu_vector_matmul_seq(a: &[f32], b: &[f32], _k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n];

    // Row-major accumulation: for each row of B, scale by corresponding a[p]
    for (p, &a_val) in a.iter().enumerate() {
        let row = &b[p * n..(p + 1) * n];
        for (j, &b_val) in row.iter().enumerate() {
            c[j] += a_val * b_val;
        }
    }

    c
}

/// CPU matmul with B transposed: A @ B^T
/// a[m,k] @ b[n,k]^T -> c[m,n]
#[allow(clippy::many_single_char_names)]
fn cpu_matmul_transpose_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                // a[i,p] * b[j,p] (b is stored row-major as [n,k])
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Transpose a matrix from [rows, cols] to [cols, rows]
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0; data.len()];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j];
        }
    }
    result
}

/// IMP-096: Parallel SIMD vector-matrix multiply using transposed weights
///
/// Computes a[1,k] @ weight_t[n,k]^T + bias[n] -> c[n]
/// Each output c[j] = dot(a, weight_t[j,:]) + bias[j]
///
/// Uses transposed weights for row-major access pattern (contiguous dot products).
/// Parallelized with rayon. Compiler auto-vectorizes the inner dot product.
#[allow(clippy::many_single_char_names)]
fn cpu_matmul_transposed_simd(
    a: &[f32],        // Input vector: [k]
    weight_t: &[f32], // Transposed weights: [n, k] (row-major)
    bias: &[f32],     // Bias: [n]
    k: usize,
    n: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    // Process in chunks for better parallelism and cache locality
    const CHUNK_SIZE: usize = 4096;

    (0..n)
        .into_par_iter()
        .step_by(CHUNK_SIZE)
        .flat_map(|chunk_start| {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n);
            (chunk_start..chunk_end)
                .map(|j| {
                    // Row-major access: weight_t[j, :] is contiguous in memory
                    let row = &weight_t[j * k..(j + 1) * k];

                    // Compiler auto-vectorizes this dot product pattern
                    let dot: f32 = row.iter().zip(a.iter()).map(|(&w, &h)| w * h).sum();
                    dot + bias[j]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

// ============================================================================
// M6: Memory Efficiency - StreamingKVCache
// ============================================================================

/// Streaming KV cache for memory-efficient inference
///
/// Implements a bounded circular buffer for key-value cache that allows
/// efficient inference on long sequences without unbounded memory growth.
///
/// ## Memory Bound
///
/// Total memory = num_layers * max_positions * num_heads * head_dim * 2 (K+V) * sizeof(f32)
///
/// For 7B model (32 layers, 2048 positions, 32 heads, 128 head_dim):
/// = 32 * 2048 * 32 * 128 * 2 * 4 = ~2GB
///
/// ## Usage
///
/// ```rust,ignore
/// let mut cache = StreamingKVCache::new(32, 2048, 32, 128);
/// cache.append(0, &key_vec, &value_vec);
/// let (keys, values) = cache.get_range(0, 0, 100);
/// ```
pub struct StreamingKVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum cached positions (context length)
    max_positions: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Key cache per layer [num_layers][max_positions * num_heads * head_dim]
    keys: Vec<Vec<f32>>,
    /// Value cache per layer
    values: Vec<Vec<f32>>,
    /// Current write position (circular)
    position: usize,
    /// Number of valid positions cached
    valid_positions: usize,
}

impl StreamingKVCache {
    /// Create a new streaming KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `max_positions` - Maximum context length to cache
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    #[must_use]
    pub fn new(num_layers: usize, max_positions: usize, num_heads: usize, head_dim: usize) -> Self {
        let kv_size = max_positions * num_heads * head_dim;
        Self {
            num_layers,
            max_positions,
            num_heads,
            head_dim,
            keys: vec![vec![0.0f32; kv_size]; num_layers],
            values: vec![vec![0.0f32; kv_size]; num_layers],
            position: 0,
            valid_positions: 0,
        }
    }

    /// Append key-value pair for a single position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0-indexed)
    /// * `key` - Key vector [num_heads * head_dim]
    /// * `value` - Value vector [num_heads * head_dim]
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or key/value dimensions are wrong.
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert_eq!(key.len(), kv_dim, "Key dimension mismatch");
        assert_eq!(value.len(), kv_dim, "Value dimension mismatch");

        let offset = self.position * kv_dim;
        self.keys[layer][offset..offset + kv_dim].copy_from_slice(key);
        self.values[layer][offset..offset + kv_dim].copy_from_slice(value);

        // Update position only after last layer
        if layer == self.num_layers - 1 {
            self.position = (self.position + 1) % self.max_positions;
            self.valid_positions = (self.valid_positions + 1).min(self.max_positions);
        }
    }

    /// Get keys and values for a range of positions
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `start` - Start position (inclusive)
    /// * `end` - End position (exclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) slices
    #[must_use]
    pub fn get_range(&self, layer: usize, start: usize, end: usize) -> (&[f32], &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        (
            &self.keys[layer][start_offset..end_offset],
            &self.values[layer][start_offset..end_offset],
        )
    }

    /// Get all valid cached keys and values for a layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) for all valid positions
    #[must_use]
    pub fn get_valid(&self, layer: usize) -> (&[f32], &[f32]) {
        self.get_range(layer, 0, self.valid_positions)
    }

    /// Get current number of valid cached positions
    #[must_use]
    pub fn len(&self) -> usize {
        self.valid_positions
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.valid_positions == 0
    }

    /// Get maximum positions (context length)
    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Reset the cache
    pub fn clear(&mut self) {
        self.position = 0;
        self.valid_positions = 0;
        // Note: We don't zero the memory for performance
    }

    /// Calculate memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let kv_size = self.max_positions * self.num_heads * self.head_dim;
        // Keys + Values, f32 = 4 bytes
        self.num_layers * kv_size * 2 * 4
    }

    /// Calculate memory usage in megabytes
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// Streaming KV cache with FP16 storage for memory efficiency (M12)
///
/// Uses half-precision (FP16) storage to halve memory usage compared to FP32,
/// enabling support for ultra-long contexts (65536+) on consumer GPUs.
///
/// # Memory Efficiency
///
/// For 65536 context with 7B model config:
/// - FP32: 32 layers × 65536 pos × 32 heads × 128 dim × 2 × 4 bytes = 68.72 GB
/// - FP16: 32 layers × 65536 pos × 32 heads × 128 dim × 2 × 2 bytes = 34.36 GB
///
/// # Example
///
/// ```
/// use realizar::gpu::StreamingKVCacheFp16;
///
/// let mut cache = StreamingKVCacheFp16::new(32, 65536, 32, 128);
/// assert!(cache.memory_mb() < 36000.0); // < 36 GB
/// ```
pub struct StreamingKVCacheFp16 {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum cached positions (context length)
    max_positions: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Key cache per layer [num_layers][max_positions * num_heads * head_dim] stored as FP16 bits
    keys: Vec<Vec<u16>>,
    /// Value cache per layer stored as FP16 bits
    values: Vec<Vec<u16>>,
    /// Current write position (circular)
    position: usize,
    /// Number of valid positions cached
    valid_positions: usize,
}

impl StreamingKVCacheFp16 {
    /// Create a new FP16 streaming KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `max_positions` - Maximum context length to cache
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    #[must_use]
    pub fn new(num_layers: usize, max_positions: usize, num_heads: usize, head_dim: usize) -> Self {
        let kv_size = max_positions * num_heads * head_dim;
        Self {
            num_layers,
            max_positions,
            num_heads,
            head_dim,
            keys: vec![vec![0u16; kv_size]; num_layers],
            values: vec![vec![0u16; kv_size]; num_layers],
            position: 0,
            valid_positions: 0,
        }
    }

    /// Convert f32 to FP16 bits
    #[inline]
    fn f32_to_f16(value: f32) -> u16 {
        half::f16::from_f32(value).to_bits()
    }

    /// Convert FP16 bits to f32
    #[inline]
    fn f16_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    /// Append key-value pair for a single position (FP32 input, stored as FP16)
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0-indexed)
    /// * `key` - Key vector [num_heads * head_dim] as FP32
    /// * `value` - Value vector [num_heads * head_dim] as FP32
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or key/value dimensions are wrong.
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert_eq!(key.len(), kv_dim, "Key dimension mismatch");
        assert_eq!(value.len(), kv_dim, "Value dimension mismatch");

        let offset = self.position * kv_dim;

        // Convert FP32 to FP16 and store
        for (i, &k) in key.iter().enumerate() {
            self.keys[layer][offset + i] = Self::f32_to_f16(k);
        }
        for (i, &v) in value.iter().enumerate() {
            self.values[layer][offset + i] = Self::f32_to_f16(v);
        }

        // Update position only after last layer
        if layer == self.num_layers - 1 {
            self.position = (self.position + 1) % self.max_positions;
            self.valid_positions = (self.valid_positions + 1).min(self.max_positions);
        }
    }

    /// Get keys and values for a range of positions (converted back to FP32)
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `start` - Start position (inclusive)
    /// * `end` - End position (exclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) as Vec<f32>
    #[must_use]
    pub fn get_range_f32(&self, layer: usize, start: usize, end: usize) -> (Vec<f32>, Vec<f32>) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        let keys: Vec<f32> = self.keys[layer][start_offset..end_offset]
            .iter()
            .map(|&bits| Self::f16_to_f32(bits))
            .collect();

        let values: Vec<f32> = self.values[layer][start_offset..end_offset]
            .iter()
            .map(|&bits| Self::f16_to_f32(bits))
            .collect();

        (keys, values)
    }

    /// Get raw FP16 keys and values for a range of positions
    #[must_use]
    pub fn get_range_raw(&self, layer: usize, start: usize, end: usize) -> (&[u16], &[u16]) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        (
            &self.keys[layer][start_offset..end_offset],
            &self.values[layer][start_offset..end_offset],
        )
    }

    /// Get all valid cached keys and values for a layer (as FP32)
    #[must_use]
    pub fn get_valid_f32(&self, layer: usize) -> (Vec<f32>, Vec<f32>) {
        self.get_range_f32(layer, 0, self.valid_positions)
    }

    /// Get current number of valid cached positions
    #[must_use]
    pub fn len(&self) -> usize {
        self.valid_positions
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.valid_positions == 0
    }

    /// Get maximum positions (context length)
    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Reset the cache
    pub fn clear(&mut self) {
        self.position = 0;
        self.valid_positions = 0;
    }

    /// Calculate memory usage in bytes (half of FP32 version)
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let kv_size = self.max_positions * self.num_heads * self.head_dim;
        // Keys + Values, u16 (FP16) = 2 bytes
        self.num_layers * kv_size * 2 * 2
    }

    /// Calculate memory usage in megabytes
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// GPU buffer pool for memory reuse
///
/// Reduces allocation overhead by caching and reusing GPU buffers.
/// Per spec: "No host blocking" through buffer pooling.
pub struct GpuBufferPool {
    /// Available buffers indexed by size bucket
    available_buffers: std::collections::HashMap<usize, Vec<Vec<f32>>>,
    /// Size buckets for efficient pooling (powers of 2)
    bucket_sizes: Vec<usize>,
    /// Maximum cached buffers per bucket
    max_per_bucket: usize,
}

impl GpuBufferPool {
    /// Create new buffer pool with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            available_buffers: std::collections::HashMap::new(),
            bucket_sizes: (10..=24).map(|i| 1 << i).collect(), // 1KB to 16MB
            max_per_bucket: 4,
        }
    }

    /// Get bucket size for requested allocation
    fn get_bucket(&self, size: usize) -> usize {
        *self
            .bucket_sizes
            .iter()
            .find(|&&b| b >= size)
            .unwrap_or(&size)
    }

    /// Acquire buffer of at least `size` elements
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        let bucket = self.get_bucket(size);
        if let Some(buffers) = self.available_buffers.get_mut(&bucket) {
            if let Some(mut buf) = buffers.pop() {
                buf.resize(size, 0.0);
                return buf;
            }
        }
        vec![0.0; size]
    }

    /// Release buffer back to pool for reuse
    pub fn release(&mut self, mut buffer: Vec<f32>) {
        let bucket = self.get_bucket(buffer.capacity());
        let buffers = self.available_buffers.entry(bucket).or_default();
        if buffers.len() < self.max_per_bucket {
            buffer.clear();
            buffers.push(buffer);
        }
        // Otherwise just drop it
    }

    /// Clear all cached buffers
    pub fn clear(&mut self) {
        self.available_buffers.clear();
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> GpuPoolStats {
        let total_buffers: usize = self.available_buffers.values().map(Vec::len).sum();
        let total_bytes: usize = self
            .available_buffers
            .iter()
            .map(|(bucket, buffers)| bucket * buffers.len() * 4)
            .sum();
        GpuPoolStats {
            cached_buffers: total_buffers,
            cached_bytes: total_bytes,
        }
    }
}

impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU buffer pool statistics
#[derive(Debug, Clone, Copy)]
pub struct GpuPoolStats {
    /// Number of cached buffers
    pub cached_buffers: usize,
    /// Total cached bytes
    pub cached_bytes: usize,
}

/// Async GPU compute handle for non-blocking operations
///
/// Per spec: "Async transfer - No host blocking"
pub struct AsyncGpuResult {
    /// Result data when ready
    result: Option<Vec<f32>>,
    /// Whether computation is complete
    ready: bool,
}

impl AsyncGpuResult {
    /// Create result that's immediately ready (CPU fallback)
    pub fn ready(data: Vec<f32>) -> Self {
        Self {
            result: Some(data),
            ready: true,
        }
    }

    /// Create pending result (GPU async)
    pub fn pending() -> Self {
        Self {
            result: None,
            ready: false,
        }
    }

    /// Check if result is ready
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Mark as ready with result
    pub fn set_result(&mut self, data: Vec<f32>) {
        self.result = Some(data);
        self.ready = true;
    }

    /// Block until result is ready (for synchronization points)
    pub fn wait(self) -> Vec<f32> {
        self.result.expect("Result not ready")
    }

    /// Try to get result without blocking
    pub fn try_get(&self) -> Option<&Vec<f32>> {
        if self.ready {
            self.result.as_ref()
        } else {
            None
        }
    }
}

/// Hybrid CPU/GPU scheduler
///
/// Automatically selects optimal backend based on workload size.
pub struct HybridScheduler {
    gpu_compute: GpuCompute,
    /// Minimum matrix size (m*k*n) to use GPU
    gpu_threshold: usize,
    /// Buffer pool for memory reuse
    buffer_pool: GpuBufferPool,
}

impl HybridScheduler {
    /// Create hybrid scheduler with auto-detected GPU
    ///
    /// # Errors
    ///
    /// Returns error if compute initialization fails.
    pub fn new() -> Result<Self> {
        Ok(Self {
            gpu_compute: GpuCompute::auto()?,
            gpu_threshold: 64 * 64 * 64, // 262K elements
            buffer_pool: GpuBufferPool::new(),
        })
    }

    /// Create scheduler with custom threshold
    ///
    /// # Arguments
    ///
    /// * `gpu_threshold` - Minimum m*k*n to trigger GPU acceleration
    ///
    /// # Errors
    ///
    /// Returns error if compute initialization fails.
    pub fn with_threshold(gpu_threshold: usize) -> Result<Self> {
        Ok(Self {
            gpu_compute: GpuCompute::auto()?,
            gpu_threshold,
            buffer_pool: GpuBufferPool::new(),
        })
    }

    /// Check if GPU is available
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.gpu_compute.is_gpu()
    }

    /// Get GPU threshold
    #[must_use]
    pub fn gpu_threshold(&self) -> usize {
        self.gpu_threshold
    }

    /// Decide whether to use GPU for given workload
    ///
    /// IMP-097: For m=1 (single-token inference), CPU is faster due to:
    /// - No GPU data transfer overhead
    /// - No kernel launch latency
    /// - CPU SIMD is sufficient for vector-matrix multiply
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn should_use_gpu(&self, m: usize, k: usize, n: usize) -> bool {
        // IMP-097: Force CPU for single-token operations (m=1)
        // GPU kernel launch overhead exceeds compute benefit for small batch sizes
        if m <= 1 {
            return false;
        }
        self.gpu_compute.is_gpu() && (m * k * n) >= self.gpu_threshold
    }

    /// Execute matmul with automatic backend selection
    ///
    /// Uses GPU for large matrices, CPU for small ones.
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)
        } else {
            Ok(cpu_matmul(a, b, m, k, n))
        }
    }

    /// Execute matmul with pooled output buffer
    ///
    /// Reduces allocation overhead by reusing buffers.
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_pooled(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Acquire buffer from pool
        let mut output = self.buffer_pool.acquire(m * n);

        // Compute result
        let result = if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)?
        } else {
            cpu_matmul(a, b, m, k, n)
        };

        // Copy to pooled buffer
        output.copy_from_slice(&result);
        Ok(output)
    }

    /// Release buffer back to pool
    ///
    /// Call this when done with a buffer returned by `matmul_pooled`.
    pub fn release_buffer(&mut self, buffer: Vec<f32>) {
        self.buffer_pool.release(buffer);
    }

    /// Get buffer pool statistics
    #[must_use]
    pub fn pool_stats(&self) -> GpuPoolStats {
        self.buffer_pool.stats()
    }

    /// Execute matmul asynchronously (non-blocking on CPU fallback)
    ///
    /// Per spec: "Async transfer - No host blocking"
    ///
    /// # Errors
    ///
    /// Returns error if compute setup fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_async(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<AsyncGpuResult> {
        // For CPU fallback, compute immediately
        // For GPU, this would submit to command queue without blocking
        let result = if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)?
        } else {
            cpu_matmul(a, b, m, k, n)
        };

        Ok(AsyncGpuResult::ready(result))
    }

    /// Process batch of matmuls with optimal scheduling
    ///
    /// Batches small operations for CPU, pipelines large ones for GPU.
    ///
    /// # Errors
    ///
    /// Returns error if any compute fails.
    pub fn matmul_batch(&mut self, operations: &[MatmulOp]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(operations.len());

        for (a, b, m, k, n) in operations {
            let result = self.matmul(a, b, *m, *k, *n)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute matmul with B transposed: A @ B^T
    ///
    /// Computes C[m,n] = A[m,k] @ B[n,k]^T
    /// where B is stored row-major as [n, k].
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // For attention: Q[seq, head_dim] @ K[seq, head_dim]^T = scores[seq, seq]
        // B is stored as [n, k], we need B^T which is [k, n]
        if self.should_use_gpu(m, k, n) {
            // Transpose B and use GPU matmul
            let b_t = transpose(b, n, k);
            self.gpu_compute.matmul(a, &b_t, m, k, n)
        } else {
            // CPU: compute A @ B^T directly
            Ok(cpu_matmul_transpose_b(a, b, m, k, n))
        }
    }
}

// ============================================================================
// M29: Error Recovery & Graceful Degradation (Phase 20)
// ============================================================================

/// Error classification for recovery decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClassification {
    /// Transient error - may succeed on retry
    Transient,
    /// Fatal error - should not retry
    Fatal,
    /// GPU-specific error - may fallback to CPU
    GpuFailure,
}

/// Recovery action to take
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry the operation with a delay
    Retry {
        /// Delay before retry
        delay: Duration,
    },
    /// Fallback to CPU inference
    FallbackToCpu,
    /// Give up and propagate error
    Fail,
}

/// Error recovery strategy with exponential backoff
pub struct ErrorRecoveryStrategy {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
    jitter: f64,
}

impl ErrorRecoveryStrategy {
    /// Create new error recovery strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: 0.1,
        }
    }

    /// Set maximum retries
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set base delay
    #[must_use]
    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    /// Set maximum delay
    #[must_use]
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    /// Set jitter factor (0.0 - 1.0)
    #[must_use]
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Get max retries
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Classify an error
    #[must_use]
    pub fn classify_error(&self, error: &std::io::Error) -> ErrorClassification {
        match error.kind() {
            std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::Interrupted
            | std::io::ErrorKind::WouldBlock => ErrorClassification::Transient,

            std::io::ErrorKind::Other => {
                let msg = error.to_string().to_lowercase();
                if msg.contains("gpu") || msg.contains("cuda") || msg.contains("wgpu") {
                    ErrorClassification::GpuFailure
                } else {
                    ErrorClassification::Transient
                }
            },

            _ => ErrorClassification::Fatal,
        }
    }

    /// Determine recovery action
    #[must_use]
    pub fn determine_action(&self, error: &std::io::Error, attempt: u32) -> RecoveryAction {
        if attempt >= self.max_retries {
            return RecoveryAction::Fail;
        }

        match self.classify_error(error) {
            ErrorClassification::Transient => RecoveryAction::Retry {
                delay: self.calculate_delay(attempt),
            },
            ErrorClassification::GpuFailure => RecoveryAction::FallbackToCpu,
            ErrorClassification::Fatal => RecoveryAction::Fail,
        }
    }

    /// Determine action with explicit GPU fallback
    #[must_use]
    pub fn determine_action_with_fallback(
        &self,
        error: &std::io::Error,
        attempt: u32,
    ) -> RecoveryAction {
        let msg = error.to_string().to_lowercase();
        if msg.contains("gpu") || msg.contains("unavailable") {
            RecoveryAction::FallbackToCpu
        } else {
            self.determine_action(error, attempt)
        }
    }

    /// Calculate delay for retry attempt with exponential backoff
    #[must_use]
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let exp_delay = base_ms * 2.0_f64.powi(attempt as i32);
        let capped_delay = exp_delay.min(self.max_delay.as_millis() as f64);

        // Add jitter
        let jitter_range = capped_delay * self.jitter;
        let jittered = capped_delay + (jitter_range * 0.5); // Simplified jitter

        Duration::from_millis(jittered as u64)
    }
}

impl Default for ErrorRecoveryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Degradation mode for system state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationMode {
    /// Normal operation
    Normal,
    /// Running on CPU fallback
    CpuFallback,
    /// Memory pressure - reduced capacity
    MemoryPressure,
    /// Low latency priority mode
    LowLatency,
    /// High throughput priority mode
    HighThroughput,
}

/// System load metrics
#[derive(Debug, Clone, Copy)]
pub struct SystemLoad {
    /// CPU utilization percentage
    pub cpu_percent: f64,
    /// Memory utilization percentage
    pub memory_percent: f64,
    /// Current queue depth
    pub queue_depth: u32,
}

/// Graceful degradation manager
pub struct DegradationManager {
    gpu_available: bool,
    memory_pressure: f64,
    system_load: Option<SystemLoad>,
    latency_priority: bool,
    mode: DegradationMode,
}

impl DegradationManager {
    /// Create new degradation manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            gpu_available: true,
            memory_pressure: 0.0,
            system_load: None,
            latency_priority: false,
            mode: DegradationMode::Normal,
        }
    }

    /// Get current degradation mode
    #[must_use]
    pub fn current_mode(&self) -> DegradationMode {
        self.mode
    }

    /// Set GPU availability
    pub fn set_gpu_available(&mut self, available: bool) {
        self.gpu_available = available;
        self.update_mode();
    }

    /// Update memory pressure (0.0 - 1.0)
    pub fn update_memory_pressure(&mut self, pressure: f64) {
        self.memory_pressure = pressure.clamp(0.0, 1.0);
        self.update_mode();
    }

    /// Update system load
    pub fn update_system_load(&mut self, load: SystemLoad) {
        self.system_load = Some(load);
        self.update_mode();
    }

    /// Set latency priority mode
    pub fn set_latency_priority(&mut self, priority: bool) {
        self.latency_priority = priority;
        self.update_mode();
    }

    /// Get recommended batch size based on system state
    #[must_use]
    pub fn recommended_batch_size(&self, requested: usize) -> usize {
        if self.memory_pressure > 0.8 {
            // Reduce batch size under memory pressure
            (requested as f64 * (1.0 - self.memory_pressure)).max(1.0) as usize
        } else {
            requested
        }
    }

    /// Get recommended max context length based on system state
    #[must_use]
    pub fn recommended_max_context(&self, requested: usize) -> usize {
        if let Some(load) = &self.system_load {
            if load.cpu_percent > 90.0 || load.memory_percent > 80.0 || load.queue_depth > 50 {
                // Reduce context length under high load
                (requested as f64 * 0.75).max(256.0) as usize
            } else {
                requested
            }
        } else {
            requested
        }
    }

    fn update_mode(&mut self) {
        self.mode = if !self.gpu_available {
            DegradationMode::CpuFallback
        } else if self.latency_priority {
            DegradationMode::LowLatency
        } else if self.memory_pressure > 0.8 {
            DegradationMode::MemoryPressure
        } else if let Some(load) = &self.system_load {
            if load.cpu_percent > 90.0 || load.memory_percent > 80.0 {
                DegradationMode::MemoryPressure
            } else {
                DegradationMode::Normal
            }
        } else {
            DegradationMode::Normal
        };
    }
}

impl Default for DegradationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Request outcome for failure tracking
#[derive(Debug, Clone)]
pub enum RequestOutcome {
    /// Request completed successfully
    Success,
    /// Request failed with error message
    Failed(String),
}

/// Failure isolator with circuit breaker
pub struct FailureIsolator {
    active_requests: std::sync::atomic::AtomicU64,
    success_count: std::sync::atomic::AtomicU64,
    failure_count: std::sync::atomic::AtomicU64,
    consecutive_failures: std::sync::atomic::AtomicU32,
    circuit_open: std::sync::atomic::AtomicBool,
    next_request_id: std::sync::atomic::AtomicU64,
    failure_threshold: u32,
    cleanups: std::sync::Mutex<HashMap<u64, Box<dyn FnOnce() + Send>>>,
}

impl FailureIsolator {
    /// Create new failure isolator
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_requests: std::sync::atomic::AtomicU64::new(0),
            success_count: std::sync::atomic::AtomicU64::new(0),
            failure_count: std::sync::atomic::AtomicU64::new(0),
            consecutive_failures: std::sync::atomic::AtomicU32::new(0),
            circuit_open: std::sync::atomic::AtomicBool::new(false),
            next_request_id: std::sync::atomic::AtomicU64::new(0),
            failure_threshold: 5,
            cleanups: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Get number of active requests
    #[must_use]
    pub fn active_requests(&self) -> u64 {
        self.active_requests
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get success count
    #[must_use]
    pub fn success_count(&self) -> u64 {
        self.success_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get failure count
    #[must_use]
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Check if circuit is open
    #[must_use]
    pub fn is_circuit_open(&self) -> bool {
        self.circuit_open.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Start a new isolated request
    #[must_use]
    pub fn start_request(&self) -> u64 {
        self.active_requests
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.next_request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Try to start a request (fails if circuit is open)
    ///
    /// # Errors
    /// Returns error if circuit breaker is open.
    pub fn try_start_request(&self) -> std::result::Result<u64, &'static str> {
        if self.is_circuit_open() {
            Err("Circuit breaker is open")
        } else {
            Ok(self.start_request())
        }
    }

    /// Register cleanup handler for a request
    pub fn register_cleanup<F: FnOnce() + Send + 'static>(&self, request_id: u64, cleanup: F) {
        if let Ok(mut cleanups) = self.cleanups.lock() {
            cleanups.insert(request_id, Box::new(cleanup));
        }
    }

    /// Complete a request with outcome
    pub fn complete_request(&self, request_id: u64, outcome: &RequestOutcome) {
        self.active_requests
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

        match outcome {
            RequestOutcome::Success => {
                self.success_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                self.consecutive_failures
                    .store(0, std::sync::atomic::Ordering::SeqCst);
            },
            RequestOutcome::Failed(_) => {
                self.failure_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let failures = self
                    .consecutive_failures
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                    + 1;

                // Open circuit if threshold exceeded
                if failures >= self.failure_threshold {
                    self.circuit_open
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                }

                // Run cleanup handler
                if let Ok(mut cleanups) = self.cleanups.lock() {
                    if let Some(cleanup) = cleanups.remove(&request_id) {
                        cleanup();
                    }
                }
            },
        }
    }

    /// Reset circuit breaker
    pub fn reset_circuit(&self) {
        self.circuit_open
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.consecutive_failures
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Default for FailureIsolator {
    fn default() -> Self {
        Self::new()
    }
}

/// Isolated request handle (unused but kept for API completeness)
#[allow(dead_code)]
pub struct IsolatedRequest {
    id: u64,
}

// ============================================================================
// M30: Connection Pooling & Resource Limits (IMP-073, IMP-074, IMP-075)
// ============================================================================

/// Connection pool configuration (IMP-073)
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    max_connections: usize,
    min_connections: usize,
    idle_timeout: Duration,
}

impl ConnectionConfig {
    /// Create new connection config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_connections: 10,
            min_connections: 1,
            idle_timeout: Duration::from_secs(300),
        }
    }

    /// Set maximum connections
    #[must_use]
    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    /// Set minimum connections
    #[must_use]
    pub fn with_min_connections(mut self, min: usize) -> Self {
        self.min_connections = min;
        self
    }

    /// Set idle timeout
    #[must_use]
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection state for health checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is healthy
    Healthy,
    /// Connection is stale and needs recycling
    Stale,
    /// Connection is broken
    Broken,
}

/// Connection handle
#[derive(Debug)]
pub struct Connection {
    #[allow(dead_code)]
    id: u64,
    created_at: std::time::Instant,
}

/// Connection pool with bounded capacity (IMP-073)
pub struct ConnectionPool {
    config: ConnectionConfig,
    active: std::sync::atomic::AtomicUsize,
    idle: std::sync::Mutex<Vec<Connection>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl ConnectionPool {
    /// Create new connection pool
    #[must_use]
    pub fn new(config: ConnectionConfig) -> Self {
        Self {
            config,
            active: std::sync::atomic::AtomicUsize::new(0),
            idle: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get max connections
    #[must_use]
    pub fn max_connections(&self) -> usize {
        self.config.max_connections
    }

    /// Get min connections
    #[must_use]
    pub fn min_connections(&self) -> usize {
        self.config.min_connections
    }

    /// Get active connection count
    #[must_use]
    pub fn active_connections(&self) -> usize {
        self.active.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get idle connection count
    #[must_use]
    pub fn idle_connections(&self) -> usize {
        self.idle.lock().expect("mutex poisoned").len()
    }

    /// Acquire a connection (blocking)
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn acquire(&self) -> std::result::Result<Connection, &'static str> {
        // Try to get from idle pool first
        {
            let mut idle = self.idle.lock().expect("mutex poisoned");
            if let Some(conn) = idle.pop() {
                self.active
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                return Ok(conn);
            }
        }

        // Create new if under limit
        let current = self.active.load(std::sync::atomic::Ordering::SeqCst);
        if current < self.config.max_connections {
            self.active
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let id = self
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return Ok(Connection {
                id,
                created_at: std::time::Instant::now(),
            });
        }

        Err("Pool exhausted")
    }

    /// Try to acquire a connection (non-blocking)
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn try_acquire(&self) -> std::result::Result<Connection, &'static str> {
        self.acquire()
    }

    /// Release a connection back to pool
    pub fn release(&self, conn: Connection) {
        self.active
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        let mut idle = self.idle.lock().expect("mutex poisoned");
        idle.push(conn);
    }

    /// Check connection health
    #[must_use]
    pub fn check_health(&self, conn: &Connection) -> ConnectionState {
        let age = conn.created_at.elapsed();
        if age > self.config.idle_timeout {
            ConnectionState::Stale
        } else {
            ConnectionState::Healthy
        }
    }

    /// Warm the pool to min_connections
    pub fn warm(&self) {
        let current_idle = self.idle_connections();
        let need = self.config.min_connections.saturating_sub(current_idle);

        let mut idle = self.idle.lock().expect("mutex poisoned");
        for _ in 0..need {
            let id = self
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            idle.push(Connection {
                id,
                created_at: std::time::Instant::now(),
            });
        }
    }
}

/// Resource configuration (IMP-074)
#[derive(Debug, Clone)]
#[allow(clippy::struct_field_names)]
pub struct ResourceConfig {
    max_memory_per_request: u64,
    max_total_memory: u64,
    max_compute_time: Duration,
    max_queue_depth: usize,
}

impl ResourceConfig {
    /// Create new resource config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_memory_per_request: 512 * 1024 * 1024, // 512MB
            max_total_memory: 4 * 1024 * 1024 * 1024,  // 4GB
            max_compute_time: Duration::from_secs(30),
            max_queue_depth: 100,
        }
    }

    /// Set max memory per request
    #[must_use]
    pub fn with_max_memory_per_request(mut self, bytes: u64) -> Self {
        self.max_memory_per_request = bytes;
        self
    }

    /// Set max total memory
    #[must_use]
    pub fn with_max_total_memory(mut self, bytes: u64) -> Self {
        self.max_total_memory = bytes;
        self
    }

    /// Set max compute time
    #[must_use]
    pub fn with_max_compute_time(mut self, time: Duration) -> Self {
        self.max_compute_time = time;
        self
    }

    /// Set max queue depth
    #[must_use]
    pub fn with_max_queue_depth(mut self, depth: usize) -> Self {
        self.max_queue_depth = depth;
        self
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of limit check
#[derive(Debug, Clone)]
pub enum LimitResult {
    /// Request is allowed
    Allowed,
    /// Request is denied with reason
    Denied {
        /// Reason for denial
        reason: String,
    },
    /// Backpressure should be applied
    Backpressure,
}

/// Resource limiter (IMP-074)
pub struct ResourceLimiter {
    config: ResourceConfig,
    current_memory: std::sync::atomic::AtomicU64,
    queue_depth: std::sync::atomic::AtomicUsize,
}

impl ResourceLimiter {
    /// Create new resource limiter
    #[must_use]
    pub fn new(config: ResourceConfig) -> Self {
        Self {
            config,
            current_memory: std::sync::atomic::AtomicU64::new(0),
            queue_depth: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Check if memory request is within limits
    #[must_use]
    pub fn check_memory(&self, bytes: u64) -> LimitResult {
        if bytes > self.config.max_memory_per_request {
            return LimitResult::Denied {
                reason: format!(
                    "Request {} bytes exceeds per-request limit {} bytes",
                    bytes, self.config.max_memory_per_request
                ),
            };
        }

        let current = self
            .current_memory
            .load(std::sync::atomic::Ordering::SeqCst);
        if current + bytes > self.config.max_total_memory {
            return LimitResult::Denied {
                reason: format!(
                    "Total memory {} + {} would exceed limit {}",
                    current, bytes, self.config.max_total_memory
                ),
            };
        }

        LimitResult::Allowed
    }

    /// Allocate memory
    ///
    /// # Errors
    /// Returns error if memory limit exceeded.
    pub fn allocate(&self, bytes: u64) -> std::result::Result<(), &'static str> {
        if let LimitResult::Denied { .. } = self.check_memory(bytes) {
            return Err("Memory limit exceeded");
        }
        self.current_memory
            .fetch_add(bytes, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Deallocate memory
    pub fn deallocate(&self, bytes: u64) {
        self.current_memory
            .fetch_sub(bytes, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get current memory usage
    #[must_use]
    pub fn current_memory(&self) -> u64 {
        self.current_memory
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Enqueue a request
    pub fn enqueue(&self) -> LimitResult {
        let current = self
            .queue_depth
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if current >= self.config.max_queue_depth {
            self.queue_depth
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            LimitResult::Backpressure
        } else {
            LimitResult::Allowed
        }
    }

    /// Try to enqueue (returns backpressure if full)
    #[must_use]
    pub fn try_enqueue(&self) -> LimitResult {
        let current = self.queue_depth.load(std::sync::atomic::Ordering::SeqCst);
        if current >= self.config.max_queue_depth {
            LimitResult::Backpressure
        } else {
            self.enqueue()
        }
    }

    /// Dequeue a request
    pub fn dequeue(&self) {
        let current = self.queue_depth.load(std::sync::atomic::Ordering::SeqCst);
        if current > 0 {
            self.queue_depth
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Start compute timer
    #[must_use]
    pub fn start_compute(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
}

/// Resource metrics snapshot (IMP-075)
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// Current memory usage in bytes
    pub memory_bytes: u64,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization: f64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Last recorded latency in milliseconds
    pub last_latency_ms: u64,
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Minimum latency in ms
    pub min_ms: u64,
    /// Maximum latency in ms
    pub max_ms: u64,
    /// Average latency in ms
    pub avg_ms: u64,
}

/// Resource monitor snapshot
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Unix timestamp
    pub timestamp: u64,
    /// Memory in bytes
    pub memory_bytes: u64,
    /// GPU utilization
    pub gpu_utilization: f64,
    /// Queue depth
    pub queue_depth: usize,
}

/// Resource monitor (IMP-075)
pub struct ResourceMonitor {
    memory_bytes: std::sync::atomic::AtomicU64,
    gpu_utilization: std::sync::Mutex<f64>,
    queue_depth: std::sync::atomic::AtomicUsize,
    latencies: std::sync::Mutex<Vec<u64>>,
    last_latency_ms: std::sync::atomic::AtomicU64,
}

impl ResourceMonitor {
    /// Create new resource monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            memory_bytes: std::sync::atomic::AtomicU64::new(0),
            gpu_utilization: std::sync::Mutex::new(0.0),
            queue_depth: std::sync::atomic::AtomicUsize::new(0),
            latencies: std::sync::Mutex::new(Vec::new()),
            last_latency_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, bytes: u64) {
        self.memory_bytes
            .store(bytes, std::sync::atomic::Ordering::SeqCst);
    }

    /// Record GPU utilization
    pub fn record_gpu_utilization(&self, utilization: f64) {
        *self.gpu_utilization.lock().expect("mutex poisoned") = utilization;
    }

    /// Record queue depth
    pub fn record_queue_depth(&self, depth: usize) {
        self.queue_depth
            .store(depth, std::sync::atomic::Ordering::SeqCst);
    }

    /// Record latency
    pub fn record_latency(&self, duration: Duration) {
        let ms = duration.as_millis() as u64;
        self.last_latency_ms
            .store(ms, std::sync::atomic::Ordering::SeqCst);
        self.latencies.lock().expect("mutex poisoned").push(ms);
    }

    /// Get current metrics
    #[must_use]
    pub fn current_metrics(&self) -> ResourceMetrics {
        ResourceMetrics {
            memory_bytes: self.memory_bytes.load(std::sync::atomic::Ordering::SeqCst),
            gpu_utilization: *self.gpu_utilization.lock().expect("mutex poisoned"),
            queue_depth: self.queue_depth.load(std::sync::atomic::Ordering::SeqCst),
            last_latency_ms: self
                .last_latency_ms
                .load(std::sync::atomic::Ordering::SeqCst),
        }
    }

    /// Get latency statistics
    #[must_use]
    pub fn latency_stats(&self) -> LatencyStats {
        let latencies = self.latencies.lock().expect("mutex poisoned");
        if latencies.is_empty() {
            return LatencyStats {
                min_ms: 0,
                max_ms: 0,
                avg_ms: 0,
            };
        }

        let min_ms = *latencies.iter().min().unwrap_or(&0);
        let max_ms = *latencies.iter().max().unwrap_or(&0);
        let sum: u64 = latencies.iter().sum();
        let avg_ms = sum / latencies.len() as u64;

        LatencyStats {
            min_ms,
            max_ms,
            avg_ms,
        }
    }

    /// Get snapshot for reporting
    #[must_use]
    pub fn snapshot(&self) -> ResourceSnapshot {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        ResourceSnapshot {
            timestamp,
            memory_bytes: self.memory_bytes.load(std::sync::atomic::Ordering::SeqCst),
            gpu_utilization: *self.gpu_utilization.lock().expect("mutex poisoned"),
            queue_depth: self.queue_depth.load(std::sync::atomic::Ordering::SeqCst),
        }
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// M31: Retry Logic & Circuit Breakers (IMP-076, IMP-077, IMP-078)
// ============================================================================

/// Error category for retry decisions (IMP-076)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Transient error that may succeed on retry
    Transient,
    /// Permanent error that will not succeed on retry
    Permanent,
}

/// Retry decision (IMP-076)
#[derive(Debug, Clone)]
pub enum RetryDecision {
    /// Retry with specified delay
    Retry {
        /// Delay before retry
        delay: Duration,
    },
    /// Abort with reason
    Abort {
        /// Reason for abort
        reason: String,
    },
}

/// Retry configuration (IMP-076)
#[derive(Debug, Clone)]
pub struct RetryConfig {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
    jitter_factor: f64,
}

impl RetryConfig {
    /// Create new retry config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            jitter_factor: 0.1,
        }
    }

    /// Set max retries
    #[must_use]
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set base delay
    #[must_use]
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Set max delay
    #[must_use]
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set jitter factor (0.0 to 1.0)
    #[must_use]
    pub fn with_jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = factor.clamp(0.0, 1.0);
        self
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Retry policy (IMP-076)
pub struct RetryPolicy {
    config: RetryConfig,
}

impl RetryPolicy {
    /// Create new retry policy
    #[must_use]
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Get max retries
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        self.config.max_retries
    }

    /// Decide whether to retry
    #[must_use]
    pub fn should_retry(&self, attempt: u32, category: ErrorCategory) -> RetryDecision {
        if category == ErrorCategory::Permanent {
            return RetryDecision::Abort {
                reason: "Permanent error".to_string(),
            };
        }

        if attempt > self.config.max_retries {
            return RetryDecision::Abort {
                reason: format!("Max retries ({}) exceeded", self.config.max_retries),
            };
        }

        RetryDecision::Retry {
            delay: self.calculate_delay(attempt),
        }
    }

    /// Calculate delay with exponential backoff
    #[must_use]
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        // Exponential backoff: base * 2^attempt
        let exp_delay_ms = self.config.base_delay.as_millis() as u64 * (1u64 << attempt.min(20));
        let delay_ms = exp_delay_ms.min(self.config.max_delay.as_millis() as u64);
        Duration::from_millis(delay_ms)
    }
}

/// Circuit breaker state (IMP-077)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests allowed
    Closed,
    /// Circuit is open, requests rejected
    Open,
    /// Circuit is half-open, probe requests allowed
    HalfOpen,
}

/// Circuit breaker configuration (IMP-077)
#[derive(Debug, Clone)]
pub struct CircuitConfig {
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
}

impl CircuitConfig {
    /// Create new circuit config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
        }
    }

    /// Set failure threshold to open circuit
    #[must_use]
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set success threshold to close circuit from half-open
    #[must_use]
    pub fn with_success_threshold(mut self, threshold: u32) -> Self {
        self.success_threshold = threshold;
        self
    }

    /// Set timeout before transitioning to half-open
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker (IMP-077)
pub struct CircuitBreaker {
    config: CircuitConfig,
    state: std::sync::Mutex<CircuitState>,
    failure_count: std::sync::atomic::AtomicU32,
    success_count: std::sync::atomic::AtomicU32,
    last_failure: std::sync::Mutex<Option<std::time::Instant>>,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    #[must_use]
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: std::sync::Mutex::new(CircuitState::Closed),
            failure_count: std::sync::atomic::AtomicU32::new(0),
            success_count: std::sync::atomic::AtomicU32::new(0),
            last_failure: std::sync::Mutex::new(None),
        }
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> CircuitState {
        *self.state.lock().expect("mutex poisoned")
    }

    /// Check if request should be allowed
    #[must_use]
    pub fn allow_request(&self) -> bool {
        let mut state = self.state.lock().expect("mutex poisoned");
        match *state {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                let last_failure = self.last_failure.lock().expect("mutex poisoned");
                if let Some(last) = *last_failure {
                    if last.elapsed() >= self.config.timeout {
                        *state = CircuitState::HalfOpen;
                        self.success_count
                            .store(0, std::sync::atomic::Ordering::SeqCst);
                        return true;
                    }
                }
                false
            },
        }
    }

    /// Record a failure
    pub fn record_failure(&self) {
        let count = self
            .failure_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        *self.last_failure.lock().expect("mutex poisoned") = Some(std::time::Instant::now());

        let mut state = self.state.lock().expect("mutex poisoned");
        match *state {
            CircuitState::Closed => {
                if count >= self.config.failure_threshold {
                    *state = CircuitState::Open;
                }
            },
            CircuitState::HalfOpen => {
                *state = CircuitState::Open;
            },
            CircuitState::Open => {},
        }
    }

    /// Record a success
    pub fn record_success(&self) {
        self.failure_count
            .store(0, std::sync::atomic::Ordering::SeqCst);
        let count = self
            .success_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        let mut state = self.state.lock().expect("mutex poisoned");
        if *state == CircuitState::HalfOpen && count >= self.config.success_threshold {
            *state = CircuitState::Closed;
        }
    }
}

/// Request type for bulkhead (IMP-078)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequestType {
    /// Standard inference request
    Inference,
    /// Embedding generation request
    Embedding,
    /// Batch processing request
    Batch,
}

/// Bulkhead permit
#[derive(Debug)]
pub struct BulkheadPermit {
    request_type: RequestType,
}

/// Bulkhead configuration (IMP-078)
pub struct BulkheadConfig {
    pools: HashMap<String, usize>,
}

impl BulkheadConfig {
    /// Create new bulkhead config
    #[must_use]
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    /// Add a pool with specified size
    #[must_use]
    pub fn with_pool(mut self, name: &str, size: usize) -> Self {
        self.pools.insert(name.to_string(), size);
        self
    }
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Bulkhead stats
#[derive(Debug, Clone)]
pub struct BulkheadStats {
    /// Number of pools
    pub pool_count: usize,
    /// Total capacity across all pools
    pub total_capacity: usize,
}

/// Bulkhead manager (IMP-078)
pub struct BulkheadManager {
    inference_available: std::sync::atomic::AtomicUsize,
    inference_capacity: usize,
    embedding_available: std::sync::atomic::AtomicUsize,
    embedding_capacity: usize,
    batch_available: std::sync::atomic::AtomicUsize,
    batch_capacity: usize,
}

impl BulkheadManager {
    /// Create new bulkhead manager
    #[must_use]
    pub fn new(config: &BulkheadConfig) -> Self {
        let inference_cap = *config.pools.get("inference").unwrap_or(&10);
        let embedding_cap = *config.pools.get("embedding").unwrap_or(&5);
        let batch_cap = *config.pools.get("batch").unwrap_or(&2);

        Self {
            inference_available: std::sync::atomic::AtomicUsize::new(inference_cap),
            inference_capacity: inference_cap,
            embedding_available: std::sync::atomic::AtomicUsize::new(embedding_cap),
            embedding_capacity: embedding_cap,
            batch_available: std::sync::atomic::AtomicUsize::new(batch_cap),
            batch_capacity: batch_cap,
        }
    }

    /// Get available slots for request type
    #[must_use]
    pub fn available(&self, request_type: RequestType) -> usize {
        match request_type {
            RequestType::Inference => self
                .inference_available
                .load(std::sync::atomic::Ordering::SeqCst),
            RequestType::Embedding => self
                .embedding_available
                .load(std::sync::atomic::Ordering::SeqCst),
            RequestType::Batch => self
                .batch_available
                .load(std::sync::atomic::Ordering::SeqCst),
        }
    }

    /// Acquire a permit
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn acquire(
        &self,
        request_type: RequestType,
    ) -> std::result::Result<BulkheadPermit, &'static str> {
        let available = match request_type {
            RequestType::Inference => &self.inference_available,
            RequestType::Embedding => &self.embedding_available,
            RequestType::Batch => &self.batch_available,
        };

        let current = available.load(std::sync::atomic::Ordering::SeqCst);
        if current == 0 {
            return Err("Pool exhausted");
        }
        available.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        Ok(BulkheadPermit { request_type })
    }

    /// Try to acquire a permit (non-blocking)
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn try_acquire(
        &self,
        request_type: RequestType,
    ) -> std::result::Result<BulkheadPermit, &'static str> {
        self.acquire(request_type)
    }

    /// Release a permit
    pub fn release(&self, permit: &BulkheadPermit) {
        let available = match permit.request_type {
            RequestType::Inference => &self.inference_available,
            RequestType::Embedding => &self.embedding_available,
            RequestType::Batch => &self.batch_available,
        };
        available.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get bulkhead stats
    #[must_use]
    pub fn stats(&self) -> BulkheadStats {
        BulkheadStats {
            pool_count: 3,
            total_capacity: self.inference_capacity + self.embedding_capacity + self.batch_capacity,
        }
    }
}

// ============================================================================
// M32: Production Logging & Diagnostics (IMP-079, IMP-080, IMP-081)
// ============================================================================

/// Log level (IMP-079)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Trace level (most verbose)
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warn,
    /// Error level
    Error,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

/// Log entry (IMP-079)
#[derive(Debug, Clone)]
pub struct LogEntry {
    level: LogLevel,
    message: String,
    timestamp: u64,
    correlation_id: Option<String>,
    fields: HashMap<String, String>,
}

impl LogEntry {
    /// Create new log entry
    #[must_use]
    pub fn new(level: LogLevel, message: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            level,
            message: message.to_string(),
            timestamp,
            correlation_id: None,
            fields: HashMap::new(),
        }
    }

    /// Set correlation ID
    #[must_use]
    pub fn with_correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }

    /// Add custom field
    #[must_use]
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }

    /// Get correlation ID
    #[must_use]
    pub fn correlation_id(&self) -> Option<&str> {
        self.correlation_id.as_deref()
    }

    /// Get log level
    #[must_use]
    pub fn level(&self) -> LogLevel {
        self.level
    }

    /// Get timestamp
    #[must_use]
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Convert to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        use std::fmt::Write;

        let mut json = format!(
            "{{\"level\":\"{}\",\"message\":\"{}\",\"timestamp\":{}",
            self.level.as_str(),
            self.message,
            self.timestamp
        );

        if let Some(ref id) = self.correlation_id {
            let _ = write!(json, ",\"correlation_id\":\"{}\"", id);
        }

        for (key, value) in &self.fields {
            let _ = write!(json, ",\"{}\":\"{}\"", key, value);
        }

        json.push('}');
        json
    }
}

/// Log configuration (IMP-079)
#[derive(Debug, Clone)]
pub struct LogConfig {
    default_level: LogLevel,
    json_format: bool,
    module_levels: HashMap<String, LogLevel>,
}

impl LogConfig {
    /// Create new log config
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_level: LogLevel::Info,
            json_format: false,
            module_levels: HashMap::new(),
        }
    }

    /// Set default log level
    #[must_use]
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.default_level = level;
        self
    }

    /// Enable JSON format
    #[must_use]
    pub fn with_json_format(mut self, enabled: bool) -> Self {
        self.json_format = enabled;
        self
    }

    /// Set module-specific log level
    #[must_use]
    pub fn with_module_level(mut self, module: &str, level: LogLevel) -> Self {
        self.module_levels.insert(module.to_string(), level);
        self
    }
}

impl Default for LogConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Logger (IMP-079)
pub struct Logger {
    config: LogConfig,
}

impl Logger {
    /// Create new logger
    #[must_use]
    pub fn new(config: LogConfig) -> Self {
        Self { config }
    }

    /// Check if log level is enabled for module
    #[must_use]
    pub fn is_enabled(&self, level: LogLevel, module: &str) -> bool {
        let min_level = self
            .config
            .module_levels
            .get(module)
            .copied()
            .unwrap_or(self.config.default_level);
        level >= min_level
    }
}

/// Phase timer for latency breakdown (IMP-080)
pub struct PhaseTimer {
    phases: std::sync::Mutex<HashMap<String, (Option<std::time::Instant>, u64)>>,
}

impl PhaseTimer {
    /// Create new phase timer
    #[must_use]
    pub fn new() -> Self {
        Self {
            phases: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Start timing a phase
    pub fn start_phase(&self, name: &str) {
        let mut phases = self.phases.lock().expect("mutex poisoned");
        phases.insert(name.to_string(), (Some(std::time::Instant::now()), 0));
    }

    /// End timing a phase
    pub fn end_phase(&self, name: &str) {
        let mut phases = self.phases.lock().expect("mutex poisoned");
        if let Some((Some(start_time), _)) = phases.get(name) {
            let elapsed = start_time.elapsed().as_micros() as u64;
            phases.insert(name.to_string(), (None, elapsed));
        }
    }

    /// Get timing breakdown
    #[must_use]
    pub fn breakdown(&self) -> HashMap<String, u64> {
        let phases = self.phases.lock().expect("mutex poisoned");
        phases.iter().map(|(k, (_, v))| (k.clone(), *v)).collect()
    }
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory report (IMP-080)
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Peak memory usage in bytes
    pub peak_bytes: u64,
    /// Current memory usage in bytes
    pub current_bytes: u64,
    /// Total allocation count
    pub allocation_count: u64,
}

/// Memory tracker (IMP-080)
pub struct MemoryTracker {
    current: std::sync::atomic::AtomicU64,
    peak: std::sync::atomic::AtomicU64,
    allocation_count: std::sync::atomic::AtomicU64,
}

impl MemoryTracker {
    /// Create new memory tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: std::sync::atomic::AtomicU64::new(0),
            peak: std::sync::atomic::AtomicU64::new(0),
            allocation_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&self, _name: &str, bytes: u64) {
        let new_current = self
            .current
            .fetch_add(bytes, std::sync::atomic::Ordering::SeqCst)
            + bytes;
        self.allocation_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Update peak if necessary
        let mut peak = self.peak.load(std::sync::atomic::Ordering::SeqCst);
        while new_current > peak {
            match self.peak.compare_exchange_weak(
                peak,
                new_current,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current_peak) => peak = current_peak,
            }
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&self, _name: &str, bytes: u64) {
        self.current
            .fetch_sub(bytes, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get memory report
    #[must_use]
    pub fn report(&self) -> MemoryReport {
        MemoryReport {
            peak_bytes: self.peak.load(std::sync::atomic::Ordering::SeqCst),
            current_bytes: self.current.load(std::sync::atomic::Ordering::SeqCst),
            allocation_count: self
                .allocation_count
                .load(std::sync::atomic::Ordering::SeqCst),
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Diagnostics summary (IMP-080)
#[derive(Debug, Clone)]
pub struct DiagnosticsSummary {
    /// Number of requests tracked
    pub request_count: u64,
}

/// Diagnostics collector (IMP-080)
pub struct DiagnosticsCollector {
    request_count: std::sync::atomic::AtomicU64,
    #[allow(dead_code)]
    timings: std::sync::Mutex<Vec<HashMap<String, u64>>>,
    #[allow(dead_code)]
    memory_snapshots: std::sync::Mutex<Vec<MemoryReport>>,
}

impl DiagnosticsCollector {
    /// Create new diagnostics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            request_count: std::sync::atomic::AtomicU64::new(0),
            timings: std::sync::Mutex::new(Vec::new()),
            memory_snapshots: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Record request timing
    pub fn record_request_timing(&self, _request_id: &str, timing: HashMap<String, u64>) {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.timings.lock().expect("mutex poisoned").push(timing);
    }

    /// Record memory snapshot
    pub fn record_memory_snapshot(&self, report: MemoryReport) {
        self.memory_snapshots
            .lock()
            .expect("mutex poisoned")
            .push(report);
    }

    /// Get diagnostics summary
    #[must_use]
    pub fn summary(&self) -> DiagnosticsSummary {
        DiagnosticsSummary {
            request_count: self.request_count.load(std::sync::atomic::Ordering::SeqCst),
        }
    }
}

impl Default for DiagnosticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug mode controller (IMP-081)
pub struct DebugMode {
    enabled: std::sync::atomic::AtomicBool,
}

impl DebugMode {
    /// Create new debug mode
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Check if debug mode is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Enable debug mode
    pub fn enable(&self) {
        self.enabled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Disable debug mode
    #[allow(dead_code)]
    pub fn disable(&self) {
        self.enabled
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Default for DebugMode {
    fn default() -> Self {
        Self::new()
    }
}

/// Request capture for replay (IMP-081)
#[derive(Debug, Clone)]
pub struct RequestCapture {
    input: String,
    params: HashMap<String, String>,
}

impl RequestCapture {
    /// Create new request capture
    #[must_use]
    pub fn new() -> Self {
        Self {
            input: String::new(),
            params: HashMap::new(),
        }
    }

    /// Set input
    #[must_use]
    pub fn with_input(mut self, input: &str) -> Self {
        self.input = input.to_string();
        self
    }

    /// Add parameter
    #[must_use]
    pub fn with_params(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }

    /// Get input
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Get params
    #[must_use]
    pub fn params(&self) -> &HashMap<String, String> {
        &self.params
    }

    /// Serialize to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        let params_json: Vec<String> = self
            .params
            .iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
            .collect();
        format!(
            "{{\"input\":\"{}\",\"params\":{{{}}}}}",
            self.input,
            params_json.join(",")
        )
    }

    /// Deserialize from JSON (simple implementation)
    ///
    /// # Errors
    /// Returns error if JSON is malformed or missing input field.
    pub fn from_json(json: &str) -> std::result::Result<Self, &'static str> {
        // Simple extraction - production would use serde
        let input_start = json.find("\"input\":\"").ok_or("Missing input")?;
        let input_rest = &json[input_start + 9..];
        let input_end = input_rest.find('"').ok_or("Invalid input")?;
        let input = &input_rest[..input_end];

        Ok(Self {
            input: input.to_string(),
            params: HashMap::new(),
        })
    }
}

impl Default for RequestCapture {
    fn default() -> Self {
        Self::new()
    }
}

/// State dump for debugging (IMP-081)
#[derive(Debug, Clone)]
pub struct StateDump {
    error: String,
    stack_trace: String,
    state: HashMap<String, String>,
}

impl StateDump {
    /// Create new state dump
    #[must_use]
    pub fn new() -> Self {
        Self {
            error: String::new(),
            stack_trace: String::new(),
            state: HashMap::new(),
        }
    }

    /// Set error
    #[must_use]
    pub fn with_error(mut self, error: &str) -> Self {
        self.error = error.to_string();
        self
    }

    /// Set stack trace
    #[must_use]
    pub fn with_stack_trace(mut self, trace: &str) -> Self {
        self.stack_trace = trace.to_string();
        self
    }

    /// Add state
    #[must_use]
    pub fn with_state(mut self, key: &str, value: &str) -> Self {
        self.state.insert(key.to_string(), value.to_string());
        self
    }

    /// Get error
    #[must_use]
    pub fn error(&self) -> &str {
        &self.error
    }

    /// Get stack trace
    #[must_use]
    pub fn stack_trace(&self) -> &str {
        &self.stack_trace
    }

    /// Get state
    #[must_use]
    pub fn state(&self) -> &HashMap<String, String> {
        &self.state
    }

    /// Convert to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        let state_json: Vec<String> = self
            .state
            .iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
            .collect();
        format!(
            "{{\"error\":\"{}\",\"stack_trace\":\"{}\",\"state\":{{{}}}}}",
            self.error,
            self.stack_trace.replace('\n', "\\n"),
            state_json.join(",")
        )
    }
}

impl Default for StateDump {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// M33: GGUF HTTP Serving Integration (IMP-082, IMP-083)
// Per spec v2.15.0: Wire GpuModel to HTTP server
// ============================================================================

/// State for holding a loaded GGUF model in HTTP server context (IMP-082)
///
/// This struct wraps a GpuModel and provides thread-safe access for
/// the HTTP server to perform inference requests.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gpu::GgufModelState;
///
/// let state = GgufModelState::new();
/// assert!(!state.is_loaded());
///
/// // Load model
/// let state = load_gguf_to_gpu(vocab_size, hidden_dim, num_layers)?;
/// assert!(state.is_loaded());
/// ```
pub struct GgufModelState {
    /// Loaded GPU model (None if not loaded)
    model: Option<GpuModel>,
    /// Model name/path
    model_name: Option<String>,
    /// Whether model is ready for inference
    ready: bool,
}

impl std::fmt::Debug for GgufModelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufModelState")
            .field("model_name", &self.model_name)
            .field("ready", &self.ready)
            .field("is_loaded", &self.model.is_some())
            .finish()
    }
}

impl GgufModelState {
    /// Create empty state (no model loaded)
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: None,
            model_name: None,
            ready: false,
        }
    }

    /// Create state with a loaded model
    #[must_use]
    pub fn with_model(model: GpuModel, name: String) -> Self {
        Self {
            model: Some(model),
            model_name: Some(name),
            ready: true,
        }
    }

    /// Check if a model is loaded
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Check if model is ready for inference
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready && self.model.is_some()
    }

    /// Get model name
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// Get vocab size (0 if no model loaded)
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.model.as_ref().map_or(0, |m| m.config().vocab_size)
    }

    /// Get reference to the model (for inference)
    #[must_use]
    pub fn model(&self) -> Option<&GpuModel> {
        self.model.as_ref()
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> Option<&mut GpuModel> {
        self.model.as_mut()
    }
}

impl Default for GgufModelState {
    fn default() -> Self {
        Self::new()
    }
}

/// Load GGUF model to GPU (IMP-083)
///
/// Creates a minimal GPU model from configuration parameters.
/// This is the pipeline entry point for serving GGUF models via HTTP.
///
/// # Arguments
///
/// * `vocab_size` - Vocabulary size
/// * `hidden_dim` - Hidden dimension
/// * `num_layers` - Number of transformer layers
///
/// # Returns
///
/// * `Ok(GgufModelState)` - State with loaded model ready for inference
/// * `Err(RealizarError)` - If model creation fails
///
/// # Errors
///
/// Returns error if GPU initialization fails or model creation fails.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gpu::load_gguf_to_gpu;
///
/// let state = load_gguf_to_gpu(32000, 4096, 32)?;
/// assert!(state.is_ready());
/// ```
pub fn load_gguf_to_gpu(
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> Result<GgufModelState> {
    // Create GPU model config
    let num_heads = hidden_dim / 64; // Standard head dim of 64
    let config = GpuModelConfig {
        vocab_size,
        hidden_dim,
        num_heads,
        num_kv_heads: num_heads, // Standard MHA (no GQA)
        num_layers,
        intermediate_dim: hidden_dim * 4, // Standard FFN expansion
        eps: 1e-5,
    };

    // Create GPU model
    let model = GpuModel::new(config)?;

    // Wrap in state
    let model_name = format!("test_{}x{}x{}", vocab_size, hidden_dim, num_layers);
    Ok(GgufModelState::with_model(model, model_name))
}

#[cfg(test)]

#[cfg(test)]
mod tests;
