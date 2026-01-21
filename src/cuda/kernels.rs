//! CUDA Kernel Type Definitions and PTX Generation
//!
//! This module contains the `KernelType` enum for all supported GPU kernels
//! and the `CudaKernels` struct for generating PTX assembly.

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

/// CUDA kernel types supported by realizar
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Naive GEMM (simple, for reference)
    GemmNaive {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension
        k: u32,
    },
    /// Tiled GEMM with shared memory
    GemmTiled {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension
        k: u32,
        /// Tile size
        tile_size: u32,
    },
    /// Tensor Core GEMM (fp16)
    GemmTensorCore {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension
        k: u32,
    },
    /// GEMV (General Matrix-Vector Multiply) - optimized for M=1 (single token generation)
    /// Uses warp shuffle reduction for high throughput on M=1 matmuls
    Gemv {
        /// Input/reduction dimension (K)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// Coalesced GEMV - high-bandwidth M=1 kernel with memory coalescing
    /// 256 threads/block, shared memory caching, <0.1ms target latency
    /// PARITY-118: Replaces Gemv for 44x speedup (4.41ms → 0.1ms)
    CoalescedGemv {
        /// Input/reduction dimension (K)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// Numerically stable softmax
    Softmax {
        /// Vector dimension
        dim: u32,
    },
    /// Layer normalization
    LayerNorm {
        /// Hidden dimension
        hidden_size: u32,
        /// Epsilon for numerical stability
        epsilon: f32,
        /// Whether to use affine transform (gamma/beta)
        affine: bool,
    },
    /// FlashAttention-style attention (single head)
    Attention {
        /// Sequence length
        seq_len: u32,
        /// Head dimension
        head_dim: u32,
        /// Whether to use causal masking
        causal: bool,
    },
    /// Multi-head attention with parallel head processing (PARITY-043)
    /// Launches n_heads warps in parallel for maximum GPU occupancy
    MultiHeadAttention {
        /// Sequence length
        seq_len: u32,
        /// Head dimension (typically 64 or 128)
        head_dim: u32,
        /// Number of attention heads to process in parallel
        n_heads: u32,
        /// Whether to use causal masking
        causal: bool,
    },
    /// Tensor Core FlashAttention (FP16 WMMA) - REALIZAR-PARITY-001.3
    /// ~40x faster than FP32 baseline on sm_70+ GPUs (Volta+)
    /// Target: <2ms/token vs ~79ms FP32 baseline
    AttentionTensorCore {
        /// Sequence length (should be multiple of 16 for optimal WMMA)
        seq_len: u32,
        /// Head dimension (should be multiple of 16 for WMMA tiles)
        head_dim: u32,
        /// Number of attention heads
        n_heads: u32,
        /// Whether to use causal masking
        causal: bool,
    },
    /// Q4_K quantized GEMM (fused dequantization) - simplified format
    QuantizedGemm {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension (must be divisible by 32)
        k: u32,
    },
    /// Q4_K quantized GEMM (fused dequantization) - GGML super-block format (PARITY-041)
    /// Uses real GGML Q4_K layout: 256 values per super-block, 144 bytes each
    /// Layout: 2B d (f16) + 2B dmin (f16) + 12B scales + 128B quantized values
    QuantizedGemmGgml {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension (must be divisible by 256 for super-blocks)
        k: u32,
    },
    /// Q5_K quantized GEMM (fused dequantization) - GGML super-block format (PARITY-116)
    /// Uses GGML Q5_K layout: 256 values per super-block, 176 bytes each
    /// Layout: 2B d (f16) + 2B dmin (f16) + 12B scales + 128B ql (4-bit) + 32B qh (1-bit)
    Q5KQuantizedGemm {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension (must be divisible by 256 for super-blocks)
        k: u32,
    },
    /// Q6_K quantized GEMM (fused dequantization) - GGML super-block format (PARITY-117)
    /// Uses GGML Q6_K layout: 256 values per super-block, 210 bytes each
    /// Layout: 128B ql (4-bit) + 64B qh (2-bit) + 16B scales (8-bit) + 2B d (f16)
    Q6KQuantizedGemm {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension (must be divisible by 256 for super-blocks)
        k: u32,
    },
    /// Optimized GEMM with register blocking (IMP-900a)
    GemmOptimized {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension
        k: u32,
        /// Tile size for shared memory
        tile_size: u32,
        /// Register blocking factor
        reg_block: u32,
    },
    /// Fused GEMM + bias + activation (IMP-900b)
    GemmBiasActivation {
        /// Output rows
        m: u32,
        /// Output columns
        n: u32,
        /// Inner dimension
        k: u32,
        /// Activation type (0=none, 1=relu, 2=gelu)
        activation: u32,
    },
    /// Element-wise bias + activation epilogue (IMP-1000)
    BiasActivation {
        /// Number of elements
        n: u32,
        /// Bias size (for broadcasting)
        bias_size: u32,
        /// Activation type (0=none, 1=relu, 2=gelu)
        activation: u32,
    },
    /// FP16 Tensor Core GEMM with WMMA intrinsics (IMP-1000a)
    GemmFp16TensorCore {
        /// Output rows (must be multiple of 16)
        m: u32,
        /// Output columns (must be multiple of 16)
        n: u32,
        /// Inner dimension (must be multiple of 16)
        k: u32,
    },
    /// Fused Q4_K × Q8_0 dot product kernel (PARITY-073)
    /// Uses DP4A instructions for 4-way INT8 dot products
    /// Input: Q4_K weights (144 bytes per 256 values) + Q8_0 activations (36 bytes per 32 values)
    /// Output: F32 result
    FusedQ4Q8Dot {
        /// Number of values (must be multiple of 256 for Q4_K super-blocks)
        n: u32,
    },
    /// Q4_K quantized GEMV (fused dequantization) - PAR-003
    /// Optimized for M=1 token generation: one warp per output, no shared memory
    /// 7.1x memory bandwidth reduction vs dequant+GEMV
    Q4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-041: Tiled Q4_K GEMV with shared memory input caching
    /// Uses 256 threads per block (8 warps) for better GPU occupancy
    /// Input vector cached in shared memory, shared by multiple outputs
    /// Memory reduction: N / outputs_per_block fewer global input reads
    TiledQ4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
        /// Number of outputs per block (default: 4)
        outputs_per_block: u32,
    },
    /// PAR-056: Chunked Tiled Q4_K GEMV for large K dimensions
    /// Uses 32KB shared memory chunks to handle K > 8K (where TiledQ4KGemv fails)
    /// Needed for 7B+ FFN down projection where K = intermediate_dim > 8K
    ChunkedTiledQ4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
        /// Number of outputs per block (default: 4)
        outputs_per_block: u32,
    },
    /// PAR-062: Coalesced Q4_K GEMV with bandwidth-optimized memory access
    /// Lane 0 loads scales as 3 x u32, broadcasts via shuffle
    /// Reduces 384 redundant byte loads to 3 loads + 3 broadcasts per super-block
    CoalescedQ4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-069: Vectorized Q4_K GEMV with coalesced u32 weight loads
    /// Uses ld_global_u32 for 32-thread coalesced loads (128 bytes/transaction)
    /// Target: 80%+ memory bandwidth vs 6% with byte loads
    VectorizedQ4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-063: DP4A-based Q4_K GEMV with 4x instruction reduction
    /// Uses DP4A SIMD instruction to compute 4 multiply-adds in one cycle
    Dp4aQ4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-063-V2: DP4A SIMD Q4_K GEMV with true integer accumulation
    /// Advanced version using DP4A's native u32 accumulator with post-hoc scaling
    Dp4aSIMDQ4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-063-V4: Q8 Quantization kernel for activations
    /// Converts f32 activations to Q8_1 format (int8 + scale)
    Q8Quantize {
        /// Number of elements to quantize (must be multiple of 32)
        n: u32,
    },
    /// PAR-063-V5: Q4K × Q8 dot product kernel using integer arithmetic
    /// Uses pre-quantized Q8 activations for faster dot products
    Q4KQ8Dot {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-063-V6: Packed DP4A Q4K×Q8 kernel with true dp4a.u32.s32 instruction
    /// Uses nibble packing to process 4 values per DP4A instruction (4x IPC vs scalar)
    PackedDp4aQ4KQ8 {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-063-V3: True DP4A Q4K GEMV with proper nibble expansion
    TrueDp4aQ4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-094: Tensor Core Q4K GEMM for batched speculative decode
    /// Enables M>1 batched forward pass with fused dequant+GEMM
    /// Target: 8x speedup over GEMV for M≥16 speculative tokens
    TensorCoreQ4KGemm {
        /// Batch size (M) - number of tokens to process in parallel
        m: u32,
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-108: Batched Q4_K GEMV for 2x Ollama via shared dequantization
    ///
    /// Key optimization: Sequential GEMV dequantizes weights M times.
    /// Batched GEMV dequantizes once and multiplies by M different inputs.
    /// This amortizes ALU-bound dequantization cost (32% bandwidth → higher).
    ///
    /// Layout: y[M×N] = x[M×K] × W[N×K]^T (Q4_K quantized)
    BatchedQ4KGemv {
        /// Batch size (M) - number of sequences to process in parallel
        m: u32,
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-129: Multi-warp batched Q4_K GEMV for M=16/32
    /// Uses 2-4 warps per block, each handling 8 batch elements
    /// All warps share L1-cached weights, avoiding weight re-reads
    MultiWarpBatchedQ4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
        /// Number of warps per block (2 for M=16, 4 for M=32)
        warps: u32,
    },
    /// Q5_K quantized GEMV (fused dequantization) - PAR-003
    Q5KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// Q6_K quantized GEMV (fused dequantization) - PAR-003
    Q6KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-066: Coalesced Q6_K GEMV with vectorized scale loading
    /// Five-Whys root cause: Q6KGemvKernel uses single-byte loads for scales
    /// This kernel loads 16 scales as 4 x u32 via lane 0 + warp shuffle
    CoalescedQ6KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// PAR-130: Batched Q6_K GEMV for M>1 batch processing
    /// Eliminates M-1 kernel launches per layer for Q6K weights
    BatchedQ6KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
        /// Batch size (M)
        m: u32,
    },
    /// PAR-053: FP16 Q4_K GEMV - 2x bandwidth savings vs FP32
    /// Uses FP16 for input/output, FP32 for compute accumulation
    /// Halves memory bandwidth for activations, improving throughput
    Fp16Q4KGemv {
        /// Input dimension (K, must be multiple of 256)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// Q8_0 quantized GEMV (fused dequantization) - PAR-058
    /// Q8_0 format: 34 bytes per 32 values (2-byte fp16 scale + 32 int8 values)
    /// Simpler than Q4K but lower compression ratio
    Q8_0Gemv {
        /// Input dimension (K)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// Q5_0 quantized GEMV (fused dequantization) - PAR-058
    /// Q5_0 format: 22 bytes per 32 values (2-byte fp16 scale + 4-byte high bits + 16 bytes packed nibbles)
    /// Used for attention Q/K weights in Qwen 0.5B
    Q5_0Gemv {
        /// Input dimension (K)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// Q4_0 quantized GEMV (fused dequantization) - PAR-058
    /// Q4_0 format: 18 bytes per 32 values (2-byte fp16 scale + 16 bytes packed nibbles)
    /// Used when GGUF header says Q5_0 but data is actually Q4_0 (qtype mismatch)
    Q4_0Gemv {
        /// Input dimension (K)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// Q4_1 quantized GEMV (fused dequantization) - PAR-058
    /// Q4_1 format: 20 bytes per 32 values (2-byte fp16 scale + 2-byte fp16 min + 16 bytes packed nibbles)
    /// Used when Qwen2.5-0.5B FFN down is Q4_1 despite metadata claiming Q4_K
    Q4_1Gemv {
        /// Input dimension (K)
        k: u32,
        /// Output dimension (N)
        n: u32,
    },
    /// Incremental attention for M=1 autoregressive decoding (PAR-020 + PAR-021)
    /// GPU-resident KV cache, warp-level shuffle, no shared memory
    /// Optimized for single-token generation in LLM inference
    /// Supports GQA (Grouped Query Attention) where n_kv_heads < n_heads
    IncrementalAttention {
        /// Maximum sequence length for KV cache allocation
        max_seq_len: u32,
        /// Head dimension (e.g., 64 for TinyLlama)
        head_dim: u32,
        /// Number of query attention heads (e.g., 32 for TinyLlama)
        n_heads: u32,
        /// Number of key-value heads (for GQA, e.g., 4 for TinyLlama)
        n_kv_heads: u32,
        /// PAR-061: Read seq_len from device memory (enables CUDA graph replay)
        indirect: bool,
    },
    /// PAR-070: Multi-warp incremental attention for decode phase
    /// Uses multiple warps per head to parallelize across KV cache positions
    /// Performance target: 8x speedup over single-warp (from 81µs to ~10µs)
    MultiWarpAttention {
        /// Maximum sequence length for KV cache allocation
        max_seq_len: u32,
        /// Head dimension (must be <= 128)
        head_dim: u32,
        /// Number of query attention heads
        n_heads: u32,
        /// Number of key-value heads (for GQA)
        n_kv_heads: u32,
        /// Number of warps per head (4-8 recommended)
        num_warps_per_head: u32,
        /// Read seq_len from device memory (for CUDA graph)
        indirect: bool,
    },
    /// PAR-052: KV Cache Scatter kernel
    /// Replaces 672+ D2D copies per token with two kernel launches
    /// Scatters contiguous K/V vectors to strided KV cache positions
    KvCacheScatter {
        /// Number of KV heads (e.g., 8 for Qwen2.5-Coder-1.5B)
        num_kv_heads: u32,
        /// Head dimension (e.g., 128)
        head_dim: u32,
        /// Maximum sequence length (e.g., 4096)
        max_len: u32,
    },
    /// PAR-054: KV Cache Scatter with Indirect Position (CUDA Graph Compatible)
    /// Like KvCacheScatter but reads position from device memory, enabling graph capture
    KvCacheScatterIndirect {
        /// Number of KV heads
        num_kv_heads: u32,
        /// Head dimension
        head_dim: u32,
        /// Maximum sequence length
        max_len: u32,
    },
    /// PAR-023: RMSNorm kernel (Root Mean Square Layer Normalization)
    /// Used by LLaMA, Mistral, TinyLlama for pre-attention and pre-FFN normalization
    /// RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * gamma
    RmsNorm {
        /// Hidden dimension size
        hidden_size: u32,
        /// Epsilon for numerical stability (default: 1e-5)
        epsilon: f32,
    },
    /// PAR-081: Vectorized RMSNorm kernel with 256 threads
    /// 8x faster than single-warp RmsNorm through better parallelism
    VectorizedRmsNorm {
        /// Hidden dimension size
        hidden_size: u32,
        /// Epsilon for numerical stability (default: 1e-5)
        epsilon: f32,
    },
    /// PAR-112: Batched Vectorized RMSNorm kernel
    /// Processes M sequences in parallel using Grid.y = M
    /// Achieves ~4x speedup over M sequential kernel launches
    BatchedVectorizedRmsNorm {
        /// Hidden dimension size
        hidden_size: u32,
        /// Batch size (M)
        batch_size: u32,
        /// Epsilon for numerical stability (default: 1e-5)
        epsilon: f32,
    },
    /// CORRECTNESS-013: High-precision RMSNorm kernel for CPU/GPU bit-exactness
    /// Uses Kahan compensated summation and Newton-Raphson rsqrt refinement
    /// Slower than VectorizedRmsNorm but matches CPU output exactly
    PreciseRmsNorm {
        /// Hidden dimension size
        hidden_size: u32,
        /// Epsilon for numerical stability (default: 1e-5)
        epsilon: f32,
    },
    /// PAR-114: Batched RoPE kernel
    /// Processes M sequences in parallel using Grid.y = M
    BatchedRope {
        /// Number of heads
        num_heads: u32,
        /// Head dimension
        head_dim: u32,
        /// Batch size (M)
        batch_size: u32,
        /// RoPE theta base (typically 10000.0)
        theta: f32,
    },
    /// PAR-114: Batched Residual Add kernel
    /// Processes M sequences in parallel using Grid.y = M
    BatchedResidualAdd {
        /// Elements per sequence
        n: u32,
        /// Batch size (M)
        batch_size: u32,
    },
    /// PAR-114: Batched SwiGLU kernel
    /// Processes M sequences in parallel using Grid.y = M
    BatchedSwiglu {
        /// Elements per sequence
        n: u32,
        /// Batch size (M)
        batch_size: u32,
    },
    /// PAR-023: Residual Add kernel for async pipeline
    /// Element-wise addition for residual connections: output = input1 + input2
    ResidualAdd {
        /// Number of elements
        n: u32,
    },
    /// PAR-023: Fused Residual Add + RMSNorm kernel
    /// Combines residual addition and normalization in one pass
    /// output = rmsnorm(input1 + input2, gamma, epsilon)
    FusedResidualRmsNorm {
        /// Hidden dimension size
        hidden_size: u32,
        /// Epsilon for numerical stability
        epsilon: f32,
    },
    /// PAR-076: Fused RMSNorm + Q4K GEMV kernel
    /// Eliminates separate RMSNorm pass by fusing normalization into GEMV
    /// output = matmul(weights, rmsnorm(input, gamma))
    FusedRmsNormQ4KGemv {
        /// K dimension (hidden size, input dimension)
        k: u32,
        /// N dimension (output dimension)
        n: u32,
        /// Epsilon for RMSNorm numerical stability
        epsilon: f32,
    },

    /// PAR-077: Fused gate + up Q4K GEMV kernel
    /// Computes both gate and up projections in one pass
    /// gate_out = W_gate * x, up_out = W_up * x
    /// Saves 50% input bandwidth by reading x only once
    FusedGateUpQ4KGemv {
        /// K dimension (hidden size, input dimension)
        k: u32,
        /// N dimension (intermediate size, output per projection)
        n: u32,
    },

    // =========================================================================
    // PAR-023: Activation and Element-wise Kernels for GPU-Resident Pipeline
    // =========================================================================
    /// SiLU activation: output = x * sigmoid(x)
    /// Used in LLaMA/TinyLlama FFN
    Silu {
        /// Number of elements
        n: u32,
    },

    /// GELU activation: output ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    /// Used in GPT/BERT FFN
    Gelu {
        /// Number of elements
        n: u32,
    },

    /// Element-wise multiply: output = input1 * input2
    /// Used for gated activations (SwiGLU)
    ElementwiseMul {
        /// Number of elements
        n: u32,
    },

    /// Fused SwiGLU: output = silu(gate) * up
    /// Combines SiLU activation and multiply in one pass
    FusedSwiglu {
        /// Number of elements
        n: u32,
    },

    /// PMAT-PERF-009: Fused Q/K/V projection kernel
    /// Computes Q, K, V projections in single kernel (3x launch reduction)
    FusedQKV {
        /// Hidden dimension size
        hidden_size: u32,
        /// KV dimension (for GQA, may differ from hidden_size)
        kv_dim: u32,
    },

    /// PMAT-PERF-009: Fused Gate+Up FFN kernel with SwiGLU
    /// Computes gate and up projections + SiLU in single kernel (2x launch reduction)
    FusedGateUp {
        /// Hidden dimension size
        hidden_size: u32,
        /// Intermediate FFN dimension
        intermediate_size: u32,
    },

    /// PAR-060: RoPE (Rotary Position Embedding) kernel
    /// Applies rotary position embeddings to Q or K vectors
    Rope {
        /// Number of attention heads
        num_heads: u32,
        /// Dimension per head (must be even)
        head_dim: u32,
        /// RoPE base frequency (theta)
        theta: f32,
    },
    /// PAR-054: RoPE with Indirect Position (CUDA Graph Compatible)
    /// Reads position from device memory instead of kernel parameter
    /// Required for CUDA graph capture (parameters baked at capture time)
    RopeIndirect {
        /// Number of attention heads
        num_heads: u32,
        /// Dimension per head (must be even)
        head_dim: u32,
        /// RoPE base frequency (theta)
        theta: f32,
    },
    /// CORRECTNESS-011: RoPE NEOX style (split halves)
    /// Required for Qwen2.5 models (rope_type=2)
    RopeNeox {
        /// Number of attention heads
        num_heads: u32,
        /// Dimension per head (must be even)
        head_dim: u32,
        /// RoPE base frequency (theta)
        theta: f32,
    },
    /// CORRECTNESS-011: RoPE NEOX Indirect (CUDA Graph compatible)
    RopeNeoxIndirect {
        /// Number of attention heads
        num_heads: u32,
        /// Dimension per head (must be even)
        head_dim: u32,
        /// RoPE base frequency (theta)
        theta: f32,
    },
    /// CORRECTNESS-013: Precise RoPE NEOX Indirect (no .approx trig)
    /// Uses polynomial sin/cos approximation for CPU-matching precision.
    PreciseRopeNeoxIndirect {
        /// Number of attention heads
        num_heads: u32,
        /// Dimension per head (must be even)
        head_dim: u32,
        /// RoPE base frequency (theta)
        theta: f32,
    },
    /// PAR-062: ArgMax block reduction kernel
    /// First pass: each block finds local max and index, outputs to temp arrays
    /// Input: f32* logits (vocab_size floats)
    /// Output: f32* block_max_vals, u32* block_max_idxs
    ArgMax {
        /// Length of the input vector (vocab_size)
        length: u32,
    },
    /// PAR-062: ArgMax final reduction kernel
    /// Second pass: finds global max from block results
    /// Input: f32* block_max_vals, u32* block_max_idxs from first pass
    /// Output: u32* result token ID
    ArgMaxFinal {
        /// Number of blocks from first pass
        num_blocks: u32,
    },
}

/// CUDA kernel generator
///
/// Generates PTX assembly for various GPU kernels using trueno-gpu.
pub struct CudaKernels {
    _private: (),
}

impl CudaKernels {
    /// Create a new CUDA kernel generator
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Generate PTX source for the specified kernel
    ///
    /// Returns PTX assembly that can be loaded by the CUDA driver API.
    #[must_use]
    pub fn generate_ptx(&self, kernel_type: &KernelType) -> String {
        match kernel_type {
            KernelType::GemmNaive { m, n, k } => GemmKernel::naive(*m, *n, *k).emit_ptx(),
            // IMP-900a: Both tiled and optimized GEMM use same tiled kernel
            KernelType::GemmTiled { m, n, k, tile_size }
            | KernelType::GemmOptimized {
                m, n, k, tile_size, ..
            } => GemmKernel::tiled(*m, *n, *k, *tile_size).emit_ptx(),
            KernelType::GemmTensorCore { m, n, k } => {
                GemmKernel::tensor_core(*m, *n, *k).emit_ptx()
            }
            // GEMV for M=1 matmuls (critical for token generation throughput)
            KernelType::Gemv { k, n } => GemvKernel::new(*k, *n).emit_ptx(),
            // PARITY-118: Coalesced GEMV with 256 threads + shared memory caching
            KernelType::CoalescedGemv { k, n } => CoalescedGemvKernel::new(*k, *n).emit_ptx(),
            KernelType::Softmax { dim } => SoftmaxKernel::new(*dim).emit_ptx(),
            KernelType::LayerNorm {
                hidden_size,
                epsilon,
                affine,
            } => {
                let mut kernel = LayerNormKernel::new(*hidden_size);
                if (*epsilon - 1e-5).abs() > f32::EPSILON {
                    kernel = kernel.with_epsilon(*epsilon);
                }
                if !affine {
                    kernel = kernel.without_affine();
                }
                kernel.emit_ptx()
            }
            KernelType::Attention {
                seq_len,
                head_dim,
                causal,
            } => {
                let mut kernel = AttentionKernel::new(*seq_len, *head_dim);
                if *causal {
                    kernel = kernel.with_causal();
                }
                kernel.emit_ptx()
            }
            // PARITY-043: Multi-head attention with parallel head processing
            // Uses trueno's FlashAttention kernel which handles multi-head via grid config
            KernelType::MultiHeadAttention {
                seq_len,
                head_dim,
                n_heads: _, // Handled by launch config (grid.y = n_heads)
                causal,
            } => {
                // Calculate maximum tile size that fits in 48KB shared memory
                // smem_size = (tile_q * head_dim + tile_kv * head_dim * 2) * 4 bytes
                // With tile_q = tile_kv = T: smem = T * head_dim * 3 * 4
                // Max T = 48KB / (head_dim * 12)
                let max_tile = (48 * 1024) / (head_dim * 12);
                // IMP-1010: Also constrain by thread limit (1024 / head_dim)
                // Must match flash_attention_multi_head launch config
                // Kernel assumes tile_q == tile_kv (same threads load Q and K)
                let thread_limit = 1024 / head_dim;
                let tile_size = max_tile.min(64).min(*seq_len).min(thread_limit);

                let mut kernel =
                    AttentionKernel::new(*seq_len, *head_dim).with_tiles(tile_size, tile_size);
                if *causal {
                    kernel = kernel.with_causal();
                }
                kernel.emit_ptx()
            }
            // REALIZAR-PARITY-001.3: Tensor Core FlashAttention using FP16 WMMA
            // ~40x faster than FP32 baseline (target: <2ms/token vs 79ms)
            KernelType::AttentionTensorCore {
                seq_len,
                head_dim,
                n_heads: _, // Handled by launch config (grid.y = n_heads)
                causal,
            } => {
                let mut kernel = AttentionKernel::tensor_core(*seq_len, *head_dim);
                if *causal {
                    kernel = kernel.with_causal();
                }
                kernel.emit_ptx()
            }
            KernelType::QuantizedGemm { m, n, k } => QuantizeKernel::new(*m, *n, *k).emit_ptx(),
            // PARITY-041: GGML Q4_K super-block format (256 values, 144 bytes per super-block)
            KernelType::QuantizedGemmGgml { m, n, k } => {
                QuantizeKernel::ggml(*m, *n, *k).emit_ptx()
            }
            // PARITY-116: GGML Q5_K super-block format (256 values, 176 bytes per super-block)
            KernelType::Q5KQuantizedGemm { m, n, k } => Q5KKernel::new(*m, *n, *k).emit_ptx(),
            // PARITY-117: GGML Q6_K super-block format (256 values, 210 bytes per super-block)
            KernelType::Q6KQuantizedGemm { m, n, k } => Q6KKernel::new(*m, *n, *k).emit_ptx(),
            // IMP-900b: Fused GEMM+bias+activation (uses tiled GEMM for now)
            KernelType::GemmBiasActivation { m, n, k, .. } => {
                GemmKernel::tiled(*m, *n, *k, 32).emit_ptx()
            }
            // IMP-1000: Element-wise bias + activation epilogue (trueno-gpu kernel)
            KernelType::BiasActivation {
                n,
                bias_size,
                activation,
            } => {
                let kernel =
                    BiasActivationKernel::new(*n, *bias_size).with_activation(match activation {
                        1 => Activation::ReLU,
                        2 => Activation::GELU,
                        _ => Activation::None,
                    });
                kernel.emit_ptx()
            }
            // IMP-1000a: FP16 Tensor Core GEMM with WMMA - using trueno kernel
            KernelType::GemmFp16TensorCore { m, n, k } => {
                GemmKernel::wmma_fp16(*m, *n, *k).emit_ptx()
            }
            // PARITY-073: Fused Q4_K × Q8_0 dot product - use trueno's QuantizeKernel
            // Dot product is 1×n × n×1 GEMM (m=1, n=1, k=n_values)
            KernelType::FusedQ4Q8Dot { n } => QuantizeKernel::ggml(1, 1, *n).emit_ptx(),
            // PAR-003: Q4_K GEMV - fused dequant for M=1 token generation
            KernelType::Q4KGemv { k, n } => Q4KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-041: Tiled Q4_K GEMV with shared memory input caching
            KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            } => TiledQ4KGemvKernel::new(*k, *n)
                .with_outputs_per_block(*outputs_per_block)
                .emit_ptx(),
            // PAR-056: Chunked Tiled Q4_K GEMV for large K dimensions (7B+ models)
            KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            } => ChunkedTiledQ4KGemvKernel::new(*k, *n)
                .with_outputs_per_block(*outputs_per_block)
                .emit_ptx(),
            // PAR-062: Coalesced Q4K GEMV with bandwidth optimization
            KernelType::CoalescedQ4KGemv { k, n } => CoalescedQ4KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-069: Vectorized Q4K GEMV with coalesced u32 loads
            KernelType::VectorizedQ4KGemv { k, n } => {
                VectorizedQ4KGemvKernel::new(*k, *n).emit_ptx()
            }
            // PAR-063: DP4A Q4K GEMV with 4x instruction reduction
            KernelType::Dp4aQ4KGemv { k, n } => Dp4aQ4KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-063-V2: DP4A SIMD Q4K GEMV with integer accumulation
            KernelType::Dp4aSIMDQ4KGemv { k, n } => Dp4aSIMDQ4KGemvKernel::new(*k, *n).emit_ptx(),
            KernelType::Q5KGemv { k, n } => Q5KGemvKernel::new(*k, *n).emit_ptx(),
            KernelType::Q6KGemv { k, n } => Q6KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-066: Coalesced Q6K GEMV - vectorized scale loading
            KernelType::CoalescedQ6KGemv { k, n } => CoalescedQ6KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-130: Batched Q6K GEMV - batch decode
            KernelType::BatchedQ6KGemv { k, n, m } => {
                BatchedQ6KGemvKernel::new(*k, *n, *m).emit_ptx()
            }
            // PAR-053: FP16 Q4K GEMV - 2x bandwidth savings
            KernelType::Fp16Q4KGemv { k, n } => Fp16Q4KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-058: Q8_0 GEMV - simpler quantization for FFN down in some models
            KernelType::Q8_0Gemv { k, n } => Q8_0GemvKernel::new(*k, *n).emit_ptx(),
            // PAR-058: Q5_0 GEMV - used for Q/K weights in Qwen 0.5B
            KernelType::Q5_0Gemv { k, n } => Q5_0GemvKernel::new(*k, *n).emit_ptx(),
            // PAR-058: Q4_0 GEMV - used when GGUF qtype mismatch detected
            KernelType::Q4_0Gemv { k, n } => Q4_0GemvKernel::new(*k, *n).emit_ptx(),
            // PAR-058: Q4_1 GEMV - used when Qwen2.5-0.5B FFN down is Q4_1
            KernelType::Q4_1Gemv { k, n } => Q4_1GemvKernel::new(*k, *n).emit_ptx(),
            // PAR-020 + PAR-021: Incremental attention for M=1 autoregressive decoding
            // Supports GQA via with_gqa() constructor
            // PAR-061: indirect mode reads seq_len from device memory for CUDA graph
            KernelType::IncrementalAttention {
                max_seq_len,
                head_dim,
                n_heads,
                n_kv_heads,
                indirect,
            } => {
                IncrementalAttentionKernel::with_gqa(*max_seq_len, *head_dim, *n_heads, *n_kv_heads)
                    .with_indirect_seq_len(*indirect)
                    .emit_ptx()
            }
            // PAR-070: Multi-warp attention for decode with parallel position processing
            KernelType::MultiWarpAttention {
                max_seq_len,
                head_dim,
                n_heads,
                n_kv_heads,
                num_warps_per_head,
                indirect,
            } => MultiWarpIncrementalAttentionKernel::new(
                *max_seq_len,
                *head_dim,
                *n_heads,
                *n_kv_heads,
                *num_warps_per_head,
            )
            .with_indirect_seq_len(*indirect)
            .emit_ptx(),
            // PAR-052: KV Cache Scatter kernel
            KernelType::KvCacheScatter {
                num_kv_heads,
                head_dim,
                max_len,
            } => KvCacheScatterKernel::new(*num_kv_heads, *head_dim, *max_len).emit_ptx(),
            // PAR-054: KV Cache Scatter Indirect (CUDA Graph Compatible)
            KernelType::KvCacheScatterIndirect {
                num_kv_heads,
                head_dim,
                max_len,
            } => KvCacheScatterIndirectKernel::new(*num_kv_heads, *head_dim, *max_len).emit_ptx(),
            // PAR-023: RMSNorm for async pipeline
            KernelType::RmsNorm {
                hidden_size,
                epsilon,
            } => RmsNormKernel::new(*hidden_size)
                .with_epsilon(*epsilon)
                .emit_ptx(),
            // PAR-081: Vectorized RMSNorm with 256 threads (8x faster)
            KernelType::VectorizedRmsNorm {
                hidden_size,
                epsilon,
            } => VectorizedRmsNormKernel::new(*hidden_size)
                .with_epsilon(*epsilon)
                .emit_ptx(),
            // PAR-112: Batched Vectorized RMSNorm for M sequences
            KernelType::BatchedVectorizedRmsNorm {
                hidden_size,
                batch_size,
                epsilon,
            } => BatchedVectorizedRmsNormKernel::new(*hidden_size, *batch_size)
                .with_epsilon(*epsilon)
                .emit_ptx(),
            // CORRECTNESS-013: High-precision RMSNorm for CPU/GPU bit-exactness
            KernelType::PreciseRmsNorm {
                hidden_size,
                epsilon,
            } => PreciseRmsNormKernel::new(*hidden_size)
                .with_epsilon(*epsilon)
                .emit_ptx(),
            // PAR-114: Batched RoPE for M sequences
            KernelType::BatchedRope {
                num_heads,
                head_dim,
                batch_size,
                theta,
            } => BatchedRopeKernel::new(*num_heads, *head_dim, *batch_size, *theta).emit_ptx(),
            // PAR-114: Batched Residual Add for M sequences
            KernelType::BatchedResidualAdd { n, batch_size } => {
                BatchedResidualAddKernel::new(*n, *batch_size).emit_ptx()
            }
            // PAR-114: Batched SwiGLU for M sequences
            KernelType::BatchedSwiglu { n, batch_size } => {
                BatchedSwigluKernel::new(*n, *batch_size).emit_ptx()
            }
            // PAR-023: Residual Add for async pipeline
            KernelType::ResidualAdd { n } => ResidualAddKernel::new(*n).emit_ptx(),
            // PAR-023: Fused Residual Add + RMSNorm for reduced memory bandwidth
            KernelType::FusedResidualRmsNorm {
                hidden_size,
                epsilon,
            } => FusedResidualRmsNormKernel::new(*hidden_size)
                .with_epsilon(*epsilon)
                .emit_ptx(),
            // PAR-076: Fused RMSNorm + Q4K GEMV
            KernelType::FusedRmsNormQ4KGemv { k, n, epsilon } => {
                FusedRmsNormQ4KGemvKernel::new(*k, *n)
                    .with_epsilon(*epsilon)
                    .emit_ptx()
            }
            // PAR-077: Fused gate + up Q4K GEMV
            KernelType::FusedGateUpQ4KGemv { k, n } => {
                FusedGateUpQ4KGemvKernel::new(*k, *n).emit_ptx()
            }
            // PAR-023: Activation kernels for GPU-resident pipeline
            KernelType::Silu { n } => SiluKernel::new(*n).emit_ptx(),
            KernelType::Gelu { n } => GeluKernel::new(*n).emit_ptx(),
            KernelType::ElementwiseMul { n } => ElementwiseMulKernel::new(*n).emit_ptx(),
            KernelType::FusedSwiglu { n } => FusedSwigluKernel::new(*n).emit_ptx(),
            // PMAT-PERF-009: Fused QKV projection (3 GEMV → 1 kernel)
            KernelType::FusedQKV {
                hidden_size,
                kv_dim,
            } => FusedQKVKernel::new(*hidden_size as usize, *kv_dim as usize).emit_ptx(),
            // PMAT-PERF-009: Fused Gate+Up FFN with SwiGLU (2 GEMV → 1 kernel)
            KernelType::FusedGateUp {
                hidden_size,
                intermediate_size,
            } => FusedGateUpKernel::new(*hidden_size as usize, *intermediate_size as usize)
                .emit_ptx(),
            // PAR-060: RoPE kernel for GPU-resident position embeddings
            KernelType::Rope {
                num_heads,
                head_dim,
                theta,
            } => RopeKernel::new(*num_heads, *head_dim, *theta).emit_ptx(),
            // PAR-054: RoPE Indirect for CUDA graph capture
            KernelType::RopeIndirect {
                num_heads,
                head_dim,
                theta,
            } => RopeIndirectKernel::new(*num_heads, *head_dim, *theta).emit_ptx(),
            // CORRECTNESS-011: RoPE NEOX style (split halves)
            KernelType::RopeNeox {
                num_heads,
                head_dim,
                theta,
            } => RopeNeoxKernel::new(*num_heads, *head_dim, *theta).emit_ptx(),
            // CORRECTNESS-011: RoPE NEOX Indirect (CUDA Graph compatible)
            KernelType::RopeNeoxIndirect {
                num_heads,
                head_dim,
                theta,
            } => RopeNeoxIndirectKernel::new(*num_heads, *head_dim, *theta).emit_ptx(),
            // CORRECTNESS-013: Precise RoPE NEOX Indirect (no .approx trig)
            KernelType::PreciseRopeNeoxIndirect {
                num_heads,
                head_dim,
                theta,
            } => PreciseRopeIndirectKernel::new(*num_heads, *head_dim, *theta).emit_ptx(),
            // PAR-063-V3: True DP4A Q4K GEMV with proper nibble expansion
            KernelType::TrueDp4aQ4KGemv { k, n } => TrueDp4aQ4KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-094: Tensor Core Q4K GEMM for batched speculative decode
            KernelType::TensorCoreQ4KGemm { m, k, n } => {
                TensorCoreQ4KGemmKernel::new(*m, *k, *n).emit_ptx()
            }
            // PAR-108: Batched Q4K GEMV for 2x Ollama via shared dequantization
            KernelType::BatchedQ4KGemv { m, k, n } => {
                BatchedQ4KGemvKernel::new(*k, *n, *m).emit_ptx()
            }
            // PAR-129: Multi-warp batched Q4K GEMV for M=16/32 (2-4 warps × 8 batch elements)
            KernelType::MultiWarpBatchedQ4KGemv { k, n, warps } => {
                MultiWarpBatchedQ4KGemvKernel::new(*k, *n, *warps).emit_ptx()
            }
            // PAR-063-V4: Q8 Quantization kernel for activations (f32 → Q8_1)
            KernelType::Q8Quantize { n } => Q8QuantizeKernel { n: *n }.emit_ptx(),
            // PAR-063-V5: Q4K × Q8 dot product using integer arithmetic
            KernelType::Q4KQ8Dot { k, n } => Q4KQ8DotKernel { k: *k, n: *n }.emit_ptx(),
            // PAR-063-V6: Packed DP4A Q4K × Q8 kernel with true dp4a.u32.s32
            KernelType::PackedDp4aQ4KQ8 { k, n } => PackedDp4aQ4KQ8Kernel::new(*k, *n).emit_ptx(),
            // PAR-062: ArgMax kernels for GPU-side greedy sampling
            KernelType::ArgMax { length } => ArgMaxKernel::new(*length).emit_ptx(),
            KernelType::ArgMaxFinal { num_blocks } => {
                ArgMaxFinalKernel::new(*num_blocks).emit_ptx()
            }
        }
    }

    /// Get kernel name for the specified type
    #[must_use]
    pub fn kernel_name(&self, kernel_type: &KernelType) -> &'static str {
        match kernel_type {
            KernelType::GemmNaive { .. } => "gemm_naive",
            // All tiled variants use the same kernel name
            KernelType::GemmTiled { .. }
            | KernelType::GemmOptimized { .. }
            | KernelType::GemmBiasActivation { .. } => "gemm_tiled",
            KernelType::GemmTensorCore { .. } => "gemm_tensor_core",
            KernelType::Gemv { .. } => "gemv_warp_reduce",
            KernelType::CoalescedGemv { .. } => "gemv_coalesced",
            KernelType::Softmax { .. } => "softmax_warp_shuffle",
            KernelType::LayerNorm { .. } => "layernorm",
            KernelType::Attention { causal, .. } => {
                if *causal {
                    "flash_attention_causal"
                } else {
                    "flash_attention"
                }
            }
            // PARITY-043: Multi-head attention uses trueno's FlashAttention kernel
            KernelType::MultiHeadAttention { causal, .. } => {
                if *causal {
                    "flash_attention_causal"
                } else {
                    "flash_attention"
                }
            }
            // REALIZAR-PARITY-001.3: Tensor Core attention kernel names
            KernelType::AttentionTensorCore { causal, .. } => {
                if *causal {
                    "flash_attention_tensor_core_causal"
                } else {
                    "flash_attention_tensor_core"
                }
            }
            KernelType::QuantizedGemm { .. } => "q4k_gemm_fused",
            KernelType::QuantizedGemmGgml { .. } => "q4k_gemm_ggml",
            KernelType::Q5KQuantizedGemm { .. } => "q5k_gemm_ggml",
            KernelType::Q6KQuantizedGemm { .. } => "q6k_gemm_ggml",
            KernelType::BiasActivation { .. } => "bias_activation",
            KernelType::GemmFp16TensorCore { .. } => "gemm_wmma_fp16",
            // FusedQ4Q8Dot now uses trueno's QuantizeKernel::ggml (same as QuantizedGemmGgml)
            KernelType::FusedQ4Q8Dot { .. } => "q4k_gemm_ggml",
            // PAR-003: Quantized GEMV kernel names (M=1 token generation)
            KernelType::Q4KGemv { .. } => "q4k_gemv_warp_reduce",
            // PAR-041: Tiled Q4K GEMV (256 threads, shared memory caching)
            KernelType::TiledQ4KGemv { .. } => "tiled_q4k_gemv",
            // PAR-056: Chunked Tiled Q4K GEMV (32KB chunks, handles large K)
            KernelType::ChunkedTiledQ4KGemv { .. } => "chunked_tiled_q4k_gemv",
            // PAR-062: Coalesced Q4K GEMV (bandwidth optimized)
            KernelType::CoalescedQ4KGemv { .. } => "coalesced_q4k_gemv",
            // PAR-069: Vectorized Q4K GEMV (coalesced u32 loads)
            KernelType::VectorizedQ4KGemv { .. } => "vectorized_q4k_gemv",
            // PAR-063: DP4A Q4K GEMV (instruction optimized)
            KernelType::Dp4aQ4KGemv { .. } => "dp4a_q4k_gemv",
            // PAR-063-V2: DP4A SIMD Q4K GEMV (integer accumulation)
            KernelType::Dp4aSIMDQ4KGemv { .. } => "dp4a_simd_q4k_gemv",
            KernelType::Q5KGemv { .. } => "q5k_gemv_warp_reduce",
            KernelType::Q6KGemv { .. } => "q6k_gemv_warp_reduce",
            // PAR-066: Coalesced Q6K GEMV
            KernelType::CoalescedQ6KGemv { .. } => "coalesced_q6k_gemv",
            // PAR-130: Batched Q6K GEMV
            KernelType::BatchedQ6KGemv { .. } => "batched_q6k_gemv_warp_reduce",
            // PAR-053: FP16 Q4K GEMV
            KernelType::Fp16Q4KGemv { .. } => "fp16_q4k_gemv",
            // PAR-058: Q8_0 GEMV
            KernelType::Q8_0Gemv { .. } => "q8_0_gemv_warp_reduce",
            // PAR-058: Q5_0 GEMV
            KernelType::Q5_0Gemv { .. } => "q5_0_gemv_warp_reduce",
            // PAR-058: Q4_0 GEMV
            KernelType::Q4_0Gemv { .. } => "q4_0_gemv_warp_reduce",
            // PAR-058: Q4_1 GEMV
            KernelType::Q4_1Gemv { .. } => "q4_1_gemv_warp_reduce",
            // PAR-020: Incremental attention for M=1 autoregressive decoding
            // PAR-061: indirect mode returns different kernel name
            KernelType::IncrementalAttention { indirect, .. } => {
                if *indirect {
                    "incremental_attention_indirect"
                } else {
                    "incremental_attention"
                }
            }
            // PAR-070: Multi-warp attention for decode
            KernelType::MultiWarpAttention { indirect, .. } => {
                if *indirect {
                    "multi_warp_attention_indirect"
                } else {
                    "multi_warp_attention"
                }
            }
            // PAR-052: KV Cache Scatter
            KernelType::KvCacheScatter { .. } => "kv_cache_scatter",
            // PAR-054: KV Cache Scatter Indirect (CUDA Graph Compatible)
            KernelType::KvCacheScatterIndirect { .. } => "kv_cache_scatter_indirect",
            // PAR-023: RMSNorm
            KernelType::RmsNorm { .. } => "rmsnorm",
            // PAR-081: Vectorized RMSNorm
            KernelType::VectorizedRmsNorm { .. } => "rmsnorm_vectorized",
            // PAR-112: Batched Vectorized RMSNorm
            KernelType::BatchedVectorizedRmsNorm { .. } => "batched_rmsnorm_vectorized",
            // CORRECTNESS-013: High-precision RMSNorm
            KernelType::PreciseRmsNorm { .. } => "rmsnorm_precise",
            // PAR-114: Batched RoPE
            KernelType::BatchedRope { .. } => "batched_rope",
            // PAR-114: Batched Residual Add
            KernelType::BatchedResidualAdd { .. } => "batched_residual_add",
            // PAR-114: Batched SwiGLU
            KernelType::BatchedSwiglu { .. } => "batched_swiglu",
            // PAR-023: Residual Add
            KernelType::ResidualAdd { .. } => "residual_add",
            // PAR-023: Fused Residual Add + RMSNorm
            KernelType::FusedResidualRmsNorm { .. } => "fused_residual_rmsnorm",
            // PAR-076: Fused RMSNorm + Q4K GEMV
            KernelType::FusedRmsNormQ4KGemv { .. } => "fused_rmsnorm_q4k_gemv",
            // PAR-077: Fused gate + up Q4K GEMV
            KernelType::FusedGateUpQ4KGemv { .. } => "fused_gate_up_q4k_gemv",
            // PAR-023: Activation kernels
            KernelType::Silu { .. } => "silu",
            KernelType::Gelu { .. } => "gelu",
            KernelType::ElementwiseMul { .. } => "elementwise_mul",
            KernelType::FusedSwiglu { .. } => "fused_swiglu",
            // PMAT-PERF-009: Fused QKV and Gate+Up kernels
            KernelType::FusedQKV { .. } => "fused_qkv_gemv",
            KernelType::FusedGateUp { .. } => "fused_gate_up_swiglu",
            // PAR-060: RoPE kernel
            KernelType::Rope { .. } => "rope",
            // PAR-054: RoPE Indirect for CUDA graph capture
            KernelType::RopeIndirect { .. } => "rope_indirect",
            // CORRECTNESS-011: RoPE NEOX style (split halves)
            KernelType::RopeNeox { .. } => "rope_neox",
            // CORRECTNESS-011: RoPE NEOX Indirect (CUDA Graph compatible)
            KernelType::RopeNeoxIndirect { .. } => "rope_neox_indirect",
            // CORRECTNESS-013: Precise RoPE NEOX Indirect (no .approx trig)
            KernelType::PreciseRopeNeoxIndirect { .. } => "rope_precise_indirect",
            // PAR-063-V3: True DP4A Q4K GEMV
            KernelType::TrueDp4aQ4KGemv { .. } => "true_dp4a_q4k_gemv",
            // PAR-094: Tensor Core Q4K GEMM for batched speculative decode
            KernelType::TensorCoreQ4KGemm { .. } => "tensor_core_q4k_gemm",
            // PAR-108: Batched Q4K GEMV for 2x Ollama
            KernelType::BatchedQ4KGemv { .. } => "batched_q4k_gemv_warp_reduce",
            // PAR-129: Multi-warp batched Q4K GEMV for M=16
            KernelType::MultiWarpBatchedQ4KGemv { .. } => "multi_warp_batched_q4k_gemv",
            // PAR-063-V4: Q8 Quantization kernel
            KernelType::Q8Quantize { .. } => "q8_quantize",
            // PAR-063-V5: Q4K × Q8 dot product
            KernelType::Q4KQ8Dot { .. } => "q4k_q8_dot",
            // PAR-063-V6: Packed DP4A Q4K × Q8
            KernelType::PackedDp4aQ4KQ8 { .. } => "packed_dp4a_q4k_q8",
            // PAR-062: ArgMax kernels
            KernelType::ArgMax { .. } => "argmax_block_reduce",
            KernelType::ArgMaxFinal { .. } => "argmax_final_reduce",
        }
    }

    /// Check if CUDA is likely available on this system
    ///
    /// Note: This is a heuristic check. Actual CUDA availability requires
    /// driver API initialization.
    #[must_use]
    pub fn cuda_likely_available() -> bool {
        // Check for NVIDIA GPU indicators
        std::path::Path::new("/dev/nvidia0").exists()
            || std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }
}

impl Default for CudaKernels {
    fn default() -> Self {
        Self::new()
    }
}
