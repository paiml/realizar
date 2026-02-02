//! CUDA Execution Engine
//!
//! This module contains the `CudaExecutor` struct which provides:
//! - GPU context and stream management
//! - Kernel execution (GEMM, GEMV, attention, etc.)
//! - Weight loading and caching
//! - CUDA graph capture and replay

#![allow(clippy::wildcard_imports)] // Submodules use super::* for internal organization

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Process-level CUDA resource management
// ---------------------------------------------------------------------------
//
// Two CUDA driver issues affect long test suites (1000+ executor cycles):
//
// 1. **Retain/release churn**: After hundreds of cuDevicePrimaryCtxRetain /
//    cuDevicePrimaryCtxRelease cycles, the driver returns CUDA_ERROR_UNKNOWN.
//
// 2. **Stream create/destroy churn**: After hundreds of cuStreamCreate /
//    cuStreamDestroy cycles, the driver returns CUDA_ERROR_UNKNOWN.
//
// Fix:
// - **Context sentinel**: A process-level CudaContext keeps refcount ≥ 1,
//   so executor releases never destroy the primary context.
// - **Stream pool**: Streams are recycled between executor lifetimes via
//   `PoolableStream`, avoiding cuStreamCreate/cuStreamDestroy churn.
//
// PTX poisoning (cuModuleLoadData failures returning CUDA_ERROR_UNKNOWN)
// permanently corrupts the CUDA context.  The sentinel detects this via
// `synchronize()` and recreates both the sentinel and stream pool.
// ---------------------------------------------------------------------------

/// Process-level sentinel: holds one cuDevicePrimaryCtxRetain reference
/// so the primary context is never fully released during the process.
static CUDA_SENTINEL: Mutex<Option<CudaContext>> = Mutex::new(None);

/// Stream pool: reusable CUDA streams to avoid cuStreamCreate/Destroy churn.
static STREAM_POOL: Mutex<Option<(CudaStream, CudaStream, CudaStream)>> = Mutex::new(None);

/// Context pool: reusable CudaContext to avoid cuDevicePrimaryCtxRetain/Release churn.
/// After the first executor, all subsequent executors reuse the same CudaContext
/// object (wrapped in ManuallyDrop in the executor to prevent Drop from releasing).
/// The sentinel keeps the primary context alive; this pool avoids retain() calls.
static CONTEXT_POOL: Mutex<Option<CudaContext>> = Mutex::new(None);

/// Ensure the sentinel context exists and is healthy for the given device.
/// If poisoned (e.g. by a previous PTX compilation failure), drops and
/// recreates it (and discards the stream pool since those streams are
/// bound to the poisoned context).
fn ensure_sentinel(device_ordinal: i32) -> Result<(), GpuError> {
    let count = device_count()?;
    if device_ordinal < 0 || device_ordinal as usize >= count {
        return Err(GpuError::DeviceNotFound(device_ordinal, count));
    }
    let mut guard = CUDA_SENTINEL.lock().unwrap();
    if let Some(ref ctx) = *guard {
        // Health check: sync detects any async kernel crashes from previous tests.
        // With sync-on-drop in CudaExecutor, poisoning should already be caught.
        // This is a second line of defense.
        if ctx.make_current().is_ok() && ctx.synchronize().is_ok() {
            return Ok(());
        }
        // Poisoned — burn everything and start fresh.
        // Clear ALL pools so refcount reaches 0 when sentinel drops.
        *STREAM_POOL.lock().unwrap() = None;
        *CONTEXT_POOL.lock().unwrap() = None;
        // Drop sentinel: refcount→0 → primary context DESTROYED
        *guard = None;
        eprintln!(
            "[CUDA-FAILFAST] Primary context poisoned — destroyed and recreating. \
             This means a kernel crashed in a previous test."
        );
    }
    *guard = Some(CudaContext::new(device_ordinal)?);
    Ok(())
}

/// Check out a CudaContext: try the pool first, create fresh if empty.
fn checkout_context(device_ordinal: i32) -> Result<CudaContext, GpuError> {
    let mut guard = CONTEXT_POOL.lock().unwrap();
    if let Some(ctx) = guard.take() {
        // Reuse pooled context — no cuDevicePrimaryCtxRetain needed.
        ctx.make_current()?;
        // Fail-fast health check: verify the context isn't poisoned.
        ctx.synchronize().map_err(|e| {
            eprintln!("[CUDA-FAILFAST] Pooled context is poisoned: {e:?}");
            e
        })?;
        return Ok(ctx);
    }
    drop(guard);
    CudaContext::new(device_ordinal)
}

/// Return a CudaContext to the pool for reuse by the next executor.
/// If the pool already has one, the returned context is dropped normally
/// (cuDevicePrimaryCtxRelease), but the sentinel keeps refcount ≥ 1.
fn checkin_context(ctx: CudaContext) {
    let mut guard = CONTEXT_POOL.lock().unwrap();
    if guard.is_none() {
        *guard = Some(ctx);
    }
    // If pool already has a context, drop the extra one (sentinel keeps primary alive)
}

/// Check out 3 streams: try the pool first, create fresh if empty.
fn checkout_streams(ctx: &CudaContext) -> Result<(CudaStream, CudaStream, CudaStream), GpuError> {
    let mut guard = STREAM_POOL.lock().unwrap();
    if let Some(streams) = guard.take() {
        return Ok(streams);
    }
    drop(guard);
    if verbose() {
        eprintln!("[CUDA-POOL] Stream pool empty, creating fresh streams");
    }
    let s1 = CudaStream::new(ctx)?;
    let s2 = CudaStream::new(ctx)?;
    let s3 = CudaStream::new(ctx)?;
    Ok((s1, s2, s3))
}

/// Return 3 streams to the pool for reuse by the next executor.
fn checkin_streams(s1: CudaStream, s2: CudaStream, s3: CudaStream) {
    let mut guard = STREAM_POOL.lock().unwrap();
    *guard = Some((s1, s2, s3));
}

/// A CUDA stream wrapper that can be safely taken out for pooling.
///
/// Implements `Deref`/`DerefMut` to `CudaStream` so existing code works
/// transparently (`self.stream.synchronize()`, `&self.compute_stream`, etc.).
/// In the custom `Drop` impl, streams are extracted via `take()` and returned
/// to the process-level pool instead of being destroyed.
pub(crate) struct PoolableStream(Option<CudaStream>);

impl PoolableStream {
    fn new(stream: CudaStream) -> Self {
        Self(Some(stream))
    }

    /// Take the inner stream, leaving None.  Used by CudaExecutor::Drop
    /// to extract the stream for pooling.
    fn take(&mut self) -> Option<CudaStream> {
        self.0.take()
    }
}

impl std::ops::Deref for PoolableStream {
    type Target = CudaStream;
    fn deref(&self) -> &CudaStream {
        self.0.as_ref().expect("stream was already taken")
    }
}

impl std::ops::DerefMut for PoolableStream {
    fn deref_mut(&mut self) -> &mut CudaStream {
        self.0.as_mut().expect("stream was already taken")
    }
}

// Don't auto-destroy the CudaStream on Drop — the custom
// CudaExecutor::Drop returns it to the pool instead.
impl Drop for PoolableStream {
    fn drop(&mut self) {
        // If the stream hasn't been taken (e.g., CudaExecutor::Drop panicked
        // or was never called), let it drop normally.  Otherwise, it was
        // already returned to the pool.
    }
}

use trueno_gpu::driver::{
    cuda_available, device_count, CaptureMode, CudaContext, CudaGraphExec, CudaModule, CudaStream,
    GpuBuffer, LaunchConfig,
};
// All kernel types are imported for API completeness
#[allow(unused_imports)]
use trueno_gpu::kernels::{
    Activation, ArgMaxFinalKernel, ArgMaxKernel, AttentionKernel,
    BatchedIncrementalAttentionKernel, BatchedQ4KGemvKernel, BatchedQ6KGemvKernel,
    BatchedResidualAddKernel, BatchedRopeKernel, BatchedSwigluKernel,
    BatchedVectorizedRmsNormKernel, BiasActivationKernel, ChunkedTiledQ4KGemvKernel,
    CoalescedGemvKernel, CoalescedQ4KGemvKernel, CoalescedQ6KGemvKernel, Dp4aQ4KGemvKernel,
    ElementwiseMulKernel, Fp16Q4KGemvKernel, FusedGateUpKernel, FusedGateUpQ4KGemvKernel,
    FusedQKVKernel, FusedResidualRmsNormKernel, FusedRmsNormQ4KGemvKernel, FusedSwigluKernel,
    GeluKernel, GemmKernel, GemvKernel, IncrementalAttentionKernel, Kernel,
    KvCacheScatterIndirectKernel, KvCacheScatterKernel, LayerNormKernel,
    MultiWarpIncrementalAttentionKernel, PackedDp4aQ4KQ8Kernel, PreciseRmsNormKernel,
    PreciseRopeIndirectKernel, Q4KGemvKernel, Q4KQ8DotKernel, Q4_0GemvKernel, Q4_1GemvKernel,
    Q5KGemvKernel, Q5KKernel, Q5_0GemvKernel, Q6KGemvKernel, Q6KKernel, Q8QuantizeKernel,
    Q8_0GemvKernel, QuantizeKernel, ResidualAddKernel, RmsNormKernel, RopeIndirectKernel,
    RopeKernel, RopeNeoxIndirectKernel, RopeNeoxKernel, SiluKernel, SoftmaxKernel,
    TensorCoreQ4KGemmKernel, TiledQ4KGemvKernel, TrueDp4aQ4KGemvKernel, VectorizedQ4KGemvKernel,
    VectorizedRmsNormKernel,
};
use trueno_gpu::GpuError;

use crate::cuda::kernels::{CudaKernels, KernelType};
use crate::cuda::memory::{
    GpuMemoryPool, PinnedHostBuffer, PoolStats, StagingBufferPool, StagingPoolStats,
};
use crate::cuda::types::{IndexedLayerWeights, TransformerWorkspace, WeightQuantType};

/// Validate that a raw device pointer is non-null before kernel launch.
///
/// Launching a CUDA kernel with a null device pointer causes an unrecoverable
/// GPU crash (CUDA_ERROR_UNKNOWN 700) that permanently poisons the device for
/// the entire process lifetime.  No amount of context destruction/recreation
/// can recover — only a process restart fixes it.
///
/// This check prevents kernel launch and returns a clean error instead.
#[inline]
fn validate_device_ptr(ptr: u64, name: &str) -> Result<(), GpuError> {
    if ptr == 0 {
        return Err(GpuError::InvalidParameter(format!(
            "{name}: null device pointer (0x0) — refusing to launch kernel \
             to prevent unrecoverable GPU device poisoning"
        )));
    }
    Ok(())
}

// Implementation modules (split from impl_main.rs for maintainability)
mod activations;
mod attention;
mod core;
mod gemm;
mod kv_cache;
mod layer;
mod layers;
mod q4k;
mod q_basic;
mod quantized;
mod weights;
mod workspace;

// Test modules
#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;

#[cfg(test)]
mod gqa_parity_tests;

#[cfg(test)]
mod test_fixtures;

#[cfg(test)]
mod poison_trace_test;

/// Process-level set of PTX hashes that failed compilation.
/// Prevents re-attempting cuModuleLoadData with the same broken PTX,
/// which would poison the CUDA context again.
static BROKEN_PTX: std::sync::LazyLock<Mutex<std::collections::HashSet<u64>>> =
    std::sync::LazyLock::new(|| Mutex::new(std::collections::HashSet::new()));

/// Check if verbose mode is enabled (REALIZAR_VERBOSE=1)
/// Default is quiet - only errors are printed
fn verbose() -> bool {
    static VERBOSE: OnceLock<bool> = OnceLock::new();
    *VERBOSE.get_or_init(|| std::env::var("REALIZAR_VERBOSE").is_ok())
}

/// CUDA execution engine for GPU-accelerated LLM inference
///
/// Manages GPU resources, kernel execution, and memory for running transformer
/// models on NVIDIA GPUs via CUDA.
pub struct CudaExecutor {
    // Drop order: first to last (kernels has no GPU resources)
    kernels: CudaKernels,
    // Memory pool for buffer reuse (IMP-900d)
    memory_pool: GpuMemoryPool,
    // Staging buffer pool for pinned memory transfers (PARITY-042)
    staging_pool: StagingBufferPool,
    // Modules wrapped in ManuallyDrop to prevent cuModuleUnload churn.
    // Thousands of cuModuleUnload cycles exhaust the CUDA driver.
    // Leaked modules (~KB each) are cleaned up at process exit.
    modules: std::mem::ManuallyDrop<HashMap<String, CudaModule>>,
    // Persistent weight buffers on GPU (PARITY-037)
    // These are loaded once at startup and reused for all forward passes
    weight_cache: HashMap<String, GpuBuffer<f32>>,
    // PAR-005: Persistent quantized weight buffers on GPU
    // For Q4_K/Q5_K/Q6_K weights that use native GEMV kernels
    // Avoids CPU->GPU transfer on every forward pass (~50+ transfers/token)
    quantized_weight_cache: HashMap<String, GpuBuffer<u8>>,
    // PAR-058: Quantization type for each cached weight (e.g., Q4K=12, Q5_0=6, Q8_0=8)
    // Stored separately to support mixed-quantization models like Qwen 0.5B
    quantized_weight_types: HashMap<String, u32>,
    // PAR-023: Cached RMSNorm gamma weights on GPU
    // Key format: "blk.{layer_idx}.{attn|ffn}_norm.gamma"
    // Pre-cached at model load to avoid per-token uploads
    rmsnorm_cache: HashMap<String, GpuBuffer<f32>>,
    // BIAS-FIX: Cached QKV and other bias vectors on GPU (FP32)
    // Key format: "blk.{layer_idx}.attn_{q|k|v}.bias"
    // Qwen2.5 models have QKV bias that must be added after GEMV
    bias_cache: HashMap<String, GpuBuffer<f32>>,
    // PAR-043: Pre-indexed layer weights for O(1) access during decode
    // Eliminates ~10ms per-token overhead from string formatting + HashMap lookups
    indexed_layer_weights: Vec<IndexedLayerWeights>,
    // PAR-043: Output norm and LM head weights (not per-layer)
    output_norm_ptr: u64,
    output_norm_len: usize,
    lm_head_ptr: u64,
    lm_head_len: usize,
    // PAR-058: LM head quantization type (Q6_K in Qwen 1.5B, not Q4_K)
    lm_head_qtype: WeightQuantType,
    // PAR-064-FIX: LM head bias pointer and length (optional, for models with output.bias)
    lm_head_bias_ptr: u64,
    lm_head_bias_len: usize,
    // PAR-043: Pre-allocated logits buffer to avoid per-token allocation
    logits_buffer: Option<GpuBuffer<f32>>,
    logits_buffer_size: usize,
    // PAR-044: Pre-allocated workspace for transformer forward pass
    // Eliminates ~288 buffer allocations per token
    workspace: TransformerWorkspace,
    // PAR-007: Cached I/O buffers for GEMV (avoid per-call allocation)
    // input_buffer: reused for all GEMV input vectors
    // output_buffer: reused for all GEMV output vectors
    // Grows as needed but never shrinks (high-water mark allocation)
    gemv_input_buffer: Option<GpuBuffer<f32>>,
    gemv_output_buffer: Option<GpuBuffer<f32>>,
    gemv_input_size: usize,  // Current capacity in elements
    gemv_output_size: usize, // Current capacity in elements
    // PAR-018 + PAR-021: GPU-resident KV cache to avoid CPU->GPU transfer each token
    // Key format: "kv_{layer_idx}_{k|v}" -> GPU buffer [num_kv_heads, max_len, head_dim]
    // For GQA models, uses num_kv_heads (smaller than num_heads)
    // Eliminates ~66 MB transfer per token for TinyLlama (22 layers x 3 MB)
    kv_cache_gpu: HashMap<String, GpuBuffer<f32>>,
    // Track how many positions are filled in each layer's KV cache
    kv_cache_lengths: HashMap<usize, usize>,
    // Max sequence length for KV cache (pre-allocated)
    kv_cache_max_len: usize,
    // KV cache dimensions (set on first use)
    kv_num_heads: usize,    // Number of Q heads (for output dimension)
    kv_num_kv_heads: usize, // Number of KV heads (for cache dimension, PAR-021 GQA)
    kv_head_dim: usize,
    // PAR-060: RoPE theta for position embeddings
    rope_theta: f32,
    // CORRECTNESS-011: RoPE type (0=NORM adjacent pairs, 2=NEOX split halves)
    rope_type: u32,
    // Compute stream for kernel execution (PARITY-038)
    // PoolableStream: returned to pool on executor drop, not destroyed.
    compute_stream: PoolableStream,
    // Transfer stream for async H2D/D2H copies (PARITY-038)
    // Runs in parallel with compute_stream for overlapped execution
    transfer_stream: PoolableStream,
    // Legacy alias for compute_stream (kept for backward compatibility)
    stream: PoolableStream,
    // PAR-054: CUDA Graph Capture for decode loop optimization
    // Captures ~280 kernel launches into single graph replay (~10us vs ~5.6ms)
    decode_graph: Option<CudaGraphExec>,
    // PAR-054: Device-side position buffer for graph replay
    // Updated before each graph replay via async memcpy
    position_buf: Option<GpuBuffer<u32>>,
    // PAR-061: Device-side seq_len buffer for attention in graph replay
    // Updated alongside position_buf (seq_len = position + 1)
    seq_len_buf: Option<GpuBuffer<u32>>,
    // PAR-119: Batched KV caches for true multi-sequence batching
    // Each layer has M separate KV caches (one per sequence in batch)
    // Size per cache: M x num_kv_heads x max_len x head_dim
    batched_kv_k_caches: HashMap<usize, GpuBuffer<f32>>,
    batched_kv_v_caches: HashMap<usize, GpuBuffer<f32>>,
    // PAR-119: Per-sequence cache lengths (all sequences share same length for now)
    batched_kv_lengths: Vec<usize>,
    // PAR-119: GPU pointer arrays for BatchedIncrementalAttentionKernel
    batched_k_ptrs: Option<GpuBuffer<u64>>,
    batched_v_ptrs: Option<GpuBuffer<u64>>,
    batched_seq_lens_gpu: Option<GpuBuffer<u32>>,
    // PAR-119: Stride for computing per-sequence pointers
    batched_kv_stride: usize,
    // PAR-119: Currently allocated batch size (for reallocation check)
    batched_kv_allocated_batch: usize,
    // PAR-121: CUDA graphs for batched forward path (one per batch size)
    // Maps batch_size -> CudaGraphExec for graph replay
    batched_decode_graphs: HashMap<usize, CudaGraphExec>,
    // PAR-121: Stable input buffer for batched graph capture
    batched_graph_input_buf: Option<GpuBuffer<f32>>,
    // PAR-121: Stable positions buffer for batched graph capture
    batched_graph_positions_buf: Option<GpuBuffer<u32>>,
    // PAR-121: Stable seq_lens buffer for batched graph capture
    batched_graph_seq_lens_buf: Option<GpuBuffer<u32>>,
    // PAR-121: Current batch size for batched graph
    batched_graph_batch_size: usize,
    // PAR-054: Stable input buffer for graph capture (same address every decode)
    graph_input_buf: Option<GpuBuffer<f32>>,
    // PAR-054: Token counter for graph capture (first decode captures, subsequent replay)
    decode_token_count: usize,
    // PAR-068: Pre-allocated argmax buffers to avoid per-token allocation
    // Block-level max values (num_blocks f32s)
    argmax_block_vals: Option<GpuBuffer<f32>>,
    // Block-level max indices (num_blocks u32s)
    argmax_block_idxs: Option<GpuBuffer<u32>>,
    // Final result (1 u32)
    argmax_result: Option<GpuBuffer<u32>>,
    // Number of blocks (based on vocab_size)
    argmax_num_blocks: u32,
    // PAR-118: Flash Decoding buffers for split-K attention
    // Partials buffer: [M, num_heads, max_chunks, head_dim + 2]
    flash_decode_partials: Option<GpuBuffer<f32>>,
    // Maximum sequence length for Flash Decoding allocation
    flash_decode_max_seq_len: usize,
    // Whether Flash Decoding is enabled (for sequences > threshold)
    flash_decode_enabled: bool,
    // QWEN-007: Q8 quantized KV cache for 4x memory reduction
    // When enabled, K/V are stored as INT8 with per-block FP32 scales
    // Block size = 32 (matches GGML Q8_0 format)
    kv_cache_q8_enabled: bool,
    // Q8 KV cache quantized values: [num_kv_heads, max_len, head_dim] as i8
    kv_cache_q8_k: HashMap<String, GpuBuffer<i8>>,
    kv_cache_q8_v: HashMap<String, GpuBuffer<i8>>,
    // Q8 KV cache scales: [num_kv_heads, max_len, head_dim / 32] as f32
    // One scale per 32 consecutive values (matches Q8_0 block size)
    kv_cache_q8_k_scales: HashMap<String, GpuBuffer<f32>>,
    kv_cache_q8_v_scales: HashMap<String, GpuBuffer<f32>>,
    // PAR-073: BrickProfiler for real per-brick timing
    // Uses std::time::Instant + CUDA sync for accurate GPU timing
    profiler: trueno::BrickProfiler,
    // CUDA context — declared last so all GPU resources above drop first
    // (they need the context alive for cuMemFree etc.).
    //
    // Wrapped in ManuallyDrop to prevent cuDevicePrimaryCtxRelease on every
    // executor drop.  Hundreds of cuDevicePrimaryCtxRetain/Release cycles
    // cause CUDA_ERROR_UNKNOWN in the driver.  ManuallyDrop skips the
    // release; the sentinel keeps the primary context alive, and
    // cuDevicePrimaryCtxRetain (just a refcount bump) is harmless.
    context: std::mem::ManuallyDrop<CudaContext>,
}
