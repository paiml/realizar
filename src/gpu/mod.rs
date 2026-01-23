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

use crate::error::Result;
use std::collections::HashMap;
use std::time::Duration;

// PMAT-802: Extracted modules
pub mod scheduler;
pub mod adapters;
mod allocator;
mod diagnostics;
mod resilience;
mod simd_ops;
mod streaming_kv;
mod metrics;
mod batch_scheduling;

// Types available without cuda feature
pub use scheduler::{GpuModel, GpuModelConfig, GpuGenerateConfig, WeightType, AttentionBuffers, BlockWeights};
#[cfg(feature = "cuda")]
pub use scheduler::CudaScheduler;

// Allocator exports (M21, M22)
pub use allocator::{
    CacheAlignedBuffer, TensorPool, ForwardArena, ScratchBuffer,
    prefetch_read, sequential_sum, sum_with_prefetch, naive_matmul, blocked_matmul,
};

// Diagnostics exports (M32)
pub use diagnostics::{
    LogLevel, LogEntry, LogConfig, Logger, PhaseTimer,
    MemoryReport, MemoryTracker, DiagnosticsSummary, DiagnosticsCollector,
    DebugMode, RequestCapture, StateDump,
};

// Resilience exports (M31)
pub use resilience::{
    ErrorCategory, RetryDecision, RetryConfig, RetryPolicy,
    CircuitState, CircuitConfig, CircuitBreaker,
    RequestType, BulkheadPermit, BulkheadConfig, BulkheadStats, BulkheadManager,
};

// SIMD ops exports (M18)
pub use simd_ops::{scalar_softmax, simd_softmax, scalar_rope, simd_rope};

// Streaming KV cache exports (M6)
pub use streaming_kv::{StreamingKVCache, StreamingKVCacheFp16};

// Metrics exports (M28)
pub use metrics::{
    InferenceMetrics, HealthChecker, ShutdownCoordinator, ComputeBackend,
    GpuCompute, GpuBufferPool, GpuPoolStats, AsyncGpuResult, HybridScheduler,
};
// Internal-only matmul functions (used by scheduler module)
pub(crate) use metrics::{cpu_matmul, cpu_matmul_transposed_simd};

// Batch scheduling exports (M25, M26, M27)
pub use batch_scheduling::{
    TokenBatch, SpeculativeBuffer, InferenceBatchScheduler, BatchId,
    AsyncRequestQueue, InferenceEventNotifier, InferenceCompletionHandler,
    TimeoutManager, RequestId, PriorityRequest, PriorityRequestQueue, Priority,
    TokenRateLimiter, ResourceTracker, AllocationId,
};

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
