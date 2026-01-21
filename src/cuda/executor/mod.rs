//! CUDA Execution Engine
//!
//! This module contains the `CudaExecutor` struct which provides:
//! - GPU context and stream management
//! - Kernel execution (GEMM, GEMV, attention, etc.)
//! - Weight loading and caching
//! - CUDA graph capture and replay

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use trueno_gpu::driver::{
    cuda_available, device_count, CaptureMode, CudaContext, CudaGraphExec, CudaModule, CudaStream,
    GpuBuffer, LaunchConfig,
};
use trueno_gpu::kernels::{
    Activation,
    ArgMaxFinalKernel,
    ArgMaxKernel,
    AttentionKernel,
    BatchedIncrementalAttentionKernel,
    BatchedQ4KGemvKernel,
    BatchedQ6KGemvKernel,
    BatchedResidualAddKernel,
    BatchedRopeKernel,
    BatchedSwigluKernel,
    BatchedVectorizedRmsNormKernel,
    BiasActivationKernel,
    ChunkedTiledQ4KGemvKernel,
    CoalescedGemvKernel,
    CoalescedQ4KGemvKernel,
    CoalescedQ6KGemvKernel,
    Dp4aQ4KGemvKernel,
    Dp4aSIMDQ4KGemvKernel,
    ElementwiseMulKernel,
    Fp16Q4KGemvKernel,
    FusedGateUpKernel,
    FusedGateUpQ4KGemvKernel,
    FusedQKVKernel,
    FusedResidualRmsNormKernel,
    FusedRmsNormQ4KGemvKernel,
    FusedSwigluKernel,
    GeluKernel,
    GemmKernel,
    GemvKernel,
    IncrementalAttentionKernel,
    Kernel,
    KvCacheScatterIndirectKernel,
    KvCacheScatterKernel,
    LayerNormKernel,
    MultiWarpBatchedQ4KGemvKernel,
    MultiWarpIncrementalAttentionKernel,
    PackedDp4aQ4KQ8Kernel,
    PreciseRmsNormKernel,
    PreciseRopeIndirectKernel,
    Q4KGemvKernel,
    Q4KQ8DotKernel,
    Q4_0GemvKernel,
    Q4_1GemvKernel,
    Q5KGemvKernel,
    Q5KKernel,
    Q5_0GemvKernel,
    Q6KGemvKernel,
    Q6KKernel,
    Q8QuantizeKernel,
    Q8_0GemvKernel,
    QuantizeKernel,
    ResidualAddKernel,
    RmsNormKernel,
    RopeIndirectKernel,
    RopeKernel,
    RopeNeoxIndirectKernel,
    RopeNeoxKernel,
    SiluKernel,
    SoftmaxKernel,
    TensorCoreQ4KGemmKernel,
    TiledQ4KGemvKernel,
    TrueDp4aQ4KGemvKernel,
    VectorizedQ4KGemvKernel,
    VectorizedRmsNormKernel,
};
use trueno_gpu::GpuError;

use crate::cuda::kernels::{CudaKernels, KernelType};
use crate::cuda::memory::{GpuMemoryPool, PinnedHostBuffer, PoolStats, StagingBufferPool, StagingPoolStats};
use crate::cuda::types::{IndexedLayerWeights, TransformerWorkspace, WeightQuantType};

// Implementation modules (split from impl_main.rs for maintainability)
mod activations;
mod attention;
mod core;
mod gemm;
mod kv_cache;
mod layer;
mod quantized;
mod weights;
mod workspace;

// Test modules
#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;

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
    // Modules must be dropped before context (cuModuleUnload needs valid context)
    modules: HashMap<String, CudaModule>,
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
    compute_stream: CudaStream,
    // Transfer stream for async H2D/D2H copies (PARITY-038)
    // Runs in parallel with compute_stream for overlapped execution
    transfer_stream: CudaStream,
    // Legacy alias for compute_stream (kept for backward compatibility)
    stream: CudaStream,
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
    // PAR-073: BrickProfiler for real per-brick timing
    // Uses std::time::Instant + CUDA sync for accurate GPU timing
    profiler: trueno::BrickProfiler,
    // Context MUST be dropped LAST (cuDevicePrimaryCtxRelease invalidates all handles)
    context: CudaContext,
}
