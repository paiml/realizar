//! CUDA Kernel Type Definitions and PTX Generation
//!
//! This module contains the `KernelType` enum for all supported GPU kernels
//! and the `CudaKernels` struct for generating PTX assembly.

// All kernel types are imported for exhaustive KernelType enum coverage
#[allow(unused_imports)]
use trueno_gpu::kernels::{
    Activation, ArgMaxFinalKernel, ArgMaxKernel, AttentionKernel,
    BatchedIncrementalAttentionKernel, BatchedQ4KGemvKernel, BatchedQ6KGemvKernel,
    BatchedResidualAddKernel, BatchedRopeKernel, BatchedSwigluKernel,
    BatchedVectorizedRmsNormKernel, BiasActivationKernel, ChunkedTiledQ4KGemvKernel,
    CoalescedGemvKernel, CoalescedQ4KGemvKernel, CoalescedQ6KGemvKernel, Dp4aQ4KGemvKernel,
    ElementwiseMulKernel, Fp16Q4KGemvKernel, FusedGateUpKernel, FusedGateUpQ4KGemvKernel,
    FusedQKVKernel, FusedResidualRmsNormKernel, FusedRmsNormGateUpSwigluQ4KKernel,
    FusedRmsNormQ4KGemvKernel, FusedSwigluKernel, GeluKernel, GemmKernel, GemvKernel,
    IncrementalAttentionKernel, Kernel, KvCacheScatterIndirectKernel, KvCacheScatterKernel,
    LayerNormKernel, MultiWarpIncrementalAttentionKernel, PackedDp4aQ4KQ8Kernel,
    PreciseRmsNormKernel, PreciseRopeIndirectKernel, Q4KGemvKernel, Q4KQ8DotKernel,
    Q4_0GemvKernel, Q4_1GemvKernel, Q5KGemvKernel, Q5KKernel, Q5_0GemvKernel, Q6KGemvKernel,
    Q6KKernel, Q8QuantizeKernel, Q8_0GemvKernel, QuantizeKernel, ResidualAddKernel,
    RmsNormKernel, RopeIndirectKernel, RopeKernel, RopeNeoxIndirectKernel, RopeNeoxKernel,
    SiluKernel, SoftmaxKernel, TensorCoreQ4KGemmKernel, TiledQ4KGemvKernel,
    TrueDp4aQ4KGemvKernel, VectorizedQ4KGemvKernel, VectorizedRmsNormKernel,
    WideQ4KGemvKernel,
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
    /// PAR-132: Wide Q4_K GEMV with 256 threads (8 warps) per output
    /// Root cause fix for 3x Ollama gap: 32 threads = 33% SM occupancy
    /// 8 warps = 67-100% occupancy, enables memory latency hiding
    /// Cross-warp reduction via shared memory (32 bytes)
    WideQ4KGemv {
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

    /// QWEN-009: 3-way fused kernel: RMSNorm → Gate/Up Q4K GEMV → SwiGLU
    /// Combines RMSNorm, dual Q4K projections (gate & up), and SwiGLU activation
    /// in a single kernel pass for 1.2x FFN forward speedup.
    ///
    /// Flow: input → RMSNorm(input, gamma) → gate=W_gate·x, up=W_up·x → silu(gate)*up
    ///
    /// Memory savings:
    /// - Normalized input stays in shared memory (not written to global)
    /// - Gate/up projections computed in parallel using same normalized input
    /// - SwiGLU applied immediately before storing final result
    FusedRmsNormGateUpSwigluQ4K {
        /// K dimension (hidden size, input dimension)
        k: u32,
        /// N dimension (intermediate size, output dimension)
        n: u32,
        /// Epsilon for RMSNorm numerical stability (default 1e-6 for Qwen)
        epsilon: f32,
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
    /// QWEN-007: Q8 dequantization kernel for KV cache
    /// Dequantizes INT8 values to FP32 using per-block scales
    /// Input: i8* quants, f32* scales
    /// Output: f32* output
    /// Formula: output[i] = quants[i] * scales[i / 32]
    Q8Dequant {
        /// Number of elements to dequantize
        n: u32,
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
            },
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
            },
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
            },
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
            },
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
            },
            KernelType::QuantizedGemm { m, n, k } => QuantizeKernel::new(*m, *n, *k).emit_ptx(),
            // PARITY-041: GGML Q4_K super-block format (256 values, 144 bytes per super-block)
            KernelType::QuantizedGemmGgml { m, n, k } => {
                QuantizeKernel::ggml(*m, *n, *k).emit_ptx()
            },
            // PARITY-116: GGML Q5_K super-block format (256 values, 176 bytes per super-block)
            KernelType::Q5KQuantizedGemm { m, n, k } => Q5KKernel::new(*m, *n, *k).emit_ptx(),
            // PARITY-117: GGML Q6_K super-block format (256 values, 210 bytes per super-block)
            KernelType::Q6KQuantizedGemm { m, n, k } => Q6KKernel::new(*m, *n, *k).emit_ptx(),
            // IMP-900b: Fused GEMM+bias+activation (uses tiled GEMM for now)
            KernelType::GemmBiasActivation { m, n, k, .. } => {
                GemmKernel::tiled(*m, *n, *k, 32).emit_ptx()
            },
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
            },
            // IMP-1000a: FP16 Tensor Core GEMM with WMMA - using trueno kernel
            KernelType::GemmFp16TensorCore { m, n, k } => {
                GemmKernel::wmma_fp16(*m, *n, *k).emit_ptx()
            },
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
            // PAR-132: Wide Q4K GEMV with 256 threads (8 warps) for SM occupancy
            KernelType::WideQ4KGemv { k, n } => WideQ4KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-069: Vectorized Q4K GEMV with coalesced u32 loads
            KernelType::VectorizedQ4KGemv { k, n } => {
                VectorizedQ4KGemvKernel::new(*k, *n).emit_ptx()
            },
            // PAR-063: DP4A Q4K GEMV with 4x instruction reduction
            KernelType::Dp4aQ4KGemv { k, n } => Dp4aQ4KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-063-V2: DP4A SIMD Q4K GEMV - deprecated, fallback to Dp4aQ4KGemv
            KernelType::Dp4aSIMDQ4KGemv { k, n } => Dp4aQ4KGemvKernel::new(*k, *n).emit_ptx(),
            KernelType::Q5KGemv { k, n } => Q5KGemvKernel::new(*k, *n).emit_ptx(),
            KernelType::Q6KGemv { k, n } => Q6KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-066: Coalesced Q6K GEMV - vectorized scale loading
            KernelType::CoalescedQ6KGemv { k, n } => CoalescedQ6KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-130: Batched Q6K GEMV - batch decode
            KernelType::BatchedQ6KGemv { k, n, m } => {
                BatchedQ6KGemvKernel::new(*k, *n, *m).emit_ptx()
            },
            // PAR-053: FP16 Q4K GEMV - 2x bandwidth savings
            KernelType::Fp16Q4KGemv { k, n } => Fp16Q4KGemvKernel::new(*k, *n).emit_ptx(),
            // PAR-058: Q8_0 GEMV - simpler quantization for FFN down in some models
            KernelType::Q8_0Gemv { k, n } => Q8_0GemvKernel::new(*k, *n).emit_ptx(),
            // PAR-058: Q5_0 GEMV - Candle layout for GGUF compatibility
            // The trueno Q5_0GemvKernel uses interleaved nibble layout, but GGUF uses
            // candle layout where:
            //   - Low nibbles (byte & 0xF) at positions 0-15
            //   - High nibbles (byte >> 4) at positions 16-31
            // Custom PTX for candle layout compatibility (GGUF-002)
            KernelType::Q5_0Gemv { k, n } => generate_q5_0_candle_ptx(*k, *n),
            // PAR-058: Q4_0 GEMV - Candle layout for GGUF compatibility
            // The trueno Q4_0GemvKernel uses interleaved nibble layout, but GGUF uses
            // candle layout where:
            //   - Low nibbles (byte & 0xF) at positions 0-15
            //   - High nibbles (byte >> 4) at positions 16-31
            // Custom PTX for candle layout compatibility (GGUF-001)
            KernelType::Q4_0Gemv { k, n } => generate_q4_0_candle_ptx(*k, *n),
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
            },
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
            },
            // PAR-114: Batched SwiGLU for M sequences
            KernelType::BatchedSwiglu { n, batch_size } => {
                BatchedSwigluKernel::new(*n, *batch_size).emit_ptx()
            },
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
            },
            // PAR-077: Fused gate + up Q4K GEMV
            KernelType::FusedGateUpQ4KGemv { k, n } => {
                FusedGateUpQ4KGemvKernel::new(*k, *n).emit_ptx()
            },
            // QWEN-009: 3-way fused RMSNorm + Gate/Up Q4K GEMV + SwiGLU
            KernelType::FusedRmsNormGateUpSwigluQ4K { k, n, epsilon } => {
                FusedRmsNormGateUpSwigluQ4KKernel::new(*k, *n)
                    .with_epsilon(*epsilon)
                    .emit_ptx()
            },
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
            },
            // PAR-108: Batched Q4K GEMV for 2x Ollama via shared dequantization
            KernelType::BatchedQ4KGemv { m, k, n } => {
                BatchedQ4KGemvKernel::new(*k, *n, *m).emit_ptx()
            },
            // PAR-129: Multi-warp batched Q4K GEMV - uses BatchedQ4KGemv with m = warps * 8
            // Each warp handles 8 batch elements, so warps=2 means m=16, warps=4 means m=32
            KernelType::MultiWarpBatchedQ4KGemv { k, n, warps } => {
                BatchedQ4KGemvKernel::new(*k, *n, *warps * 8).emit_ptx()
            },
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
            },
            // QWEN-007: Q8 dequantization kernel for KV cache
            KernelType::Q8Dequant { n } => generate_q8_dequant_ptx(*n),
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
            KernelType::LayerNorm { .. } => "layernorm_warp_shuffle",
            KernelType::Attention { causal, .. } => {
                if *causal {
                    "flash_attention_causal"
                } else {
                    "flash_attention"
                }
            },
            // PARITY-043: Multi-head attention uses trueno's FlashAttention kernel
            KernelType::MultiHeadAttention { causal, .. } => {
                if *causal {
                    "flash_attention_causal"
                } else {
                    "flash_attention"
                }
            },
            // REALIZAR-PARITY-001.3: Tensor Core attention kernel names
            KernelType::AttentionTensorCore { causal, .. } => {
                if *causal {
                    "flash_attention_tensor_core_causal"
                } else {
                    "flash_attention_tensor_core"
                }
            },
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
            // PAR-132: Wide Q4K GEMV (8 warps, high occupancy)
            KernelType::WideQ4KGemv { .. } => "wide_q4k_gemv",
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
            },
            // PAR-070: Multi-warp attention for decode
            KernelType::MultiWarpAttention { indirect, .. } => {
                if *indirect {
                    "multi_warp_attention_indirect"
                } else {
                    "multi_warp_attention"
                }
            },
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
            // QWEN-009: 3-way fused RMSNorm + Gate/Up Q4K GEMV + SwiGLU
            KernelType::FusedRmsNormGateUpSwigluQ4K { .. } => "fused_rmsnorm_gate_up_swiglu_q4k",
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
            // Uses same trueno kernel as BatchedQ4KGemv (just different warps parameter)
            KernelType::MultiWarpBatchedQ4KGemv { .. } => "batched_q4k_gemv_warp_reduce",
            // PAR-063-V4: Q8 Quantization kernel
            KernelType::Q8Quantize { .. } => "q8_quantize",
            // PAR-063-V5: Q4K × Q8 dot product
            KernelType::Q4KQ8Dot { .. } => "q4k_q8_dot",
            // PAR-063-V6: Packed DP4A Q4K × Q8
            KernelType::PackedDp4aQ4KQ8 { .. } => "packed_dp4a_q4k_q8",
            // PAR-062: ArgMax kernels
            KernelType::ArgMax { .. } => "argmax_block_reduce",
            KernelType::ArgMaxFinal { .. } => "argmax_final_reduce",
            // QWEN-007: Q8 dequantization kernel
            KernelType::Q8Dequant { .. } => "q8_dequant",
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

/// BUG-GGUF-001 FIX: Generate Q4_0 GEMV PTX with correct candle layout
///
/// The GGUF Q4_0 format uses "candle layout" where:
/// - 16 bytes contain 32 nibbles for 32 weights
/// - Low nibbles (byte & 0x0F) map to positions 0-15
/// - High nibbles (byte >> 4) map to positions 16-31
///
/// The trueno Q4_0GemvKernel incorrectly uses interleaved layout where:
/// - Thread 0 → byte 0 low nibble (position 0)
/// - Thread 1 → byte 0 high nibble (position 1)
/// - Thread 2 → byte 1 low nibble (position 2)
/// - etc.
///
/// This function generates correct PTX for GGUF Q4_0 models.
fn generate_q4_0_candle_ptx(k: u32, n: u32) -> String {
    // k and n are used for grid size configuration in the caller, not embedded in PTX
    let _ = (k, n);

    // Note: num_blocks is computed dynamically in PTX from k_dim parameter
    // This allows the same kernel to work for any K dimension
    String::from(
        r"
.version 7.5
.target sm_80
.address_size 64

// BUG-GGUF-001 FIX: Q4_0 GEMV with candle nibble layout
// Each warp (32 threads) computes one output element
// Thread 0-15: use low nibbles from bytes 0-15
// Thread 16-31: use high nibbles from bytes 0-15
.visible .entry q4_0_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<16>;
    .reg .b16 %h<4>;
    .reg .pred %p<8>;

    // r0=tid, r1=ctaid, r2=n_dim, r3=k_dim
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    // Bounds check: if ctaid >= n_dim, exit
    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_exit;

    // f0 = accumulator
    mov.f32 %f0, 0f00000000;

    // r4 = num_blocks = ceil(k_dim / 32)
    add.u32 %r4, %r3, 31;
    shr.u32 %r4, %r4, 5;

    // rd3 = row_base = w_ptr + ctaid * num_blocks * 18
    mul.lo.u32 %r5, %r4, 18;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // r6 = blk_idx (loop counter)
    mov.u32 %r6, 0;

$L_blk_loop:
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra $L_blk_loop_end;

    // rd4 = blk_addr = row_base + blk_idx * 18
    mul.wide.u32 %rd4, %r6, 18;
    add.u64 %rd4, %rd3, %rd4;

    // f1 = scale d (fp16 at offset 0) - use b16 register for f16 conversion
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // rd5 = qs_base = blk_addr + 2
    add.u64 %rd5, %rd4, 2;

    // CANDLE LAYOUT:
    // Thread 0-15 read bytes 0-15 (low nibbles -> positions 0-15)
    // Thread 16-31 read bytes 0-15 (high nibbles -> positions 16-31)
    // r8 = byte_idx = tid < 16 ? tid : tid - 16
    setp.ge.u32 %p2, %r0, 16;
    mov.u32 %r8, %r0;
    @%p2 sub.u32 %r8, %r0, 16;

    // Load byte from qs[byte_idx]
    cvt.u64.u32 %rd6, %r8;
    add.u64 %rd6, %rd5, %rd6;
    ld.global.u8 %r9, [%rd6];

    // r10 = nibble value
    // Threads 0-15: low nibble (byte & 0xF)
    // Threads 16-31: high nibble (byte >> 4)
    mov.u32 %r10, %r9;
    @%p2 shr.u32 %r10, %r9, 4;
    and.b32 %r10, %r10, 15;

    // r11 = centered value = nibble - 8 (as signed)
    sub.u32 %r11, %r10, 8;

    // f2 = dequantized = d * centered
    cvt.rn.f32.s32 %f2, %r11;
    mul.f32 %f2, %f1, %f2;

    // r12 = x_idx = blk_idx * 32 + tid
    shl.b32 %r12, %r6, 5;
    add.u32 %r12, %r12, %r0;

    // Bounds check for last block
    setp.ge.u32 %p3, %r12, %r3;
    @%p3 bra $L_skip_mul;

    // f3 = x[x_idx]
    cvt.u64.u32 %rd7, %r12;
    shl.b64 %rd7, %rd7, 2;
    add.u64 %rd7, %rd2, %rd7;
    ld.global.f32 %f3, [%rd7];

    // f0 += f2 * f3
    fma.rn.f32 %f0, %f2, %f3, %f0;

$L_skip_mul:
    add.u32 %r6, %r6, 1;
    bra $L_blk_loop;

$L_blk_loop_end:
    // Warp reduction using shfl.sync.down
    shfl.sync.down.b32 %f4, %f0, 16, 31, 0xffffffff;
    add.f32 %f0, %f0, %f4;
    shfl.sync.down.b32 %f5, %f0, 8, 31, 0xffffffff;
    add.f32 %f0, %f0, %f5;
    shfl.sync.down.b32 %f6, %f0, 4, 31, 0xffffffff;
    add.f32 %f0, %f0, %f6;
    shfl.sync.down.b32 %f7, %f0, 2, 31, 0xffffffff;
    add.f32 %f0, %f0, %f7;
    shfl.sync.down.b32 %f8, %f0, 1, 31, 0xffffffff;
    add.f32 %f0, %f0, %f8;

    // Thread 0 writes result
    setp.ne.u32 %p4, %r0, 0;
    @%p4 bra $L_exit;

    // y[ctaid] = f0
    mul.wide.u32 %rd8, %r1, 4;
    add.u64 %rd8, %rd0, %rd8;
    st.global.f32 [%rd8], %f0;

$L_exit:
    ret;
}
",
    )
}

/// BUG-GGUF-002 FIX: Generate Q5_0 GEMV PTX with correct candle layout
///
/// The GGUF Q5_0 format uses "candle layout" where:
/// - 16 bytes contain 32 nibbles for 32 weights (low bits)
/// - 4 bytes contain 32 high bits (qh)
/// - Low nibbles (byte & 0x0F) + qh bits 0-15 map to positions 0-15
/// - High nibbles (byte >> 4) + qh bits 16-31 map to positions 16-31
///
/// The trueno Q5_0GemvKernel incorrectly uses interleaved layout where:
/// - Thread 0 → byte 0 low nibble + qh bit 0 (position 0)
/// - Thread 1 → byte 0 high nibble + qh bit 1 (position 1)
/// - Thread 2 → byte 1 low nibble + qh bit 2 (position 2)
/// - etc.
///
/// This function generates correct PTX for GGUF Q5_0 models.
fn generate_q5_0_candle_ptx(k: u32, n: u32) -> String {
    // k and n are used for grid size configuration in the caller, not embedded in PTX
    let _ = (k, n);

    // Q5_0 block: 2 bytes (d fp16) + 4 bytes (qh) + 16 bytes (qs) = 22 bytes
    // Note: num_blocks is computed dynamically in PTX from k_dim parameter
    String::from(
        r"
.version 7.5
.target sm_80
.address_size 64

// BUG-GGUF-002 FIX: Q5_0 GEMV with candle nibble layout
// Each warp (32 threads) computes one output element
// Thread 0-15: use low nibbles from bytes 0-15, qh bits 0-15
// Thread 16-31: use high nibbles from bytes 0-15, qh bits 16-31
.visible .entry q5_0_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<40>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<16>;
    .reg .b16 %h<4>;
    .reg .pred %p<8>;

    // r0=tid, r1=ctaid, r2=n_dim, r3=k_dim
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    // Bounds check: if ctaid >= n_dim, exit
    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_exit;

    // f0 = accumulator
    mov.f32 %f0, 0f00000000;

    // r4 = num_blocks = ceil(k_dim / 32)
    add.u32 %r4, %r3, 31;
    shr.u32 %r4, %r4, 5;

    // rd3 = row_base = w_ptr + ctaid * num_blocks * 22
    mul.lo.u32 %r5, %r4, 22;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // r6 = blk_idx (loop counter)
    mov.u32 %r6, 0;

$L_blk_loop:
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra $L_blk_loop_end;

    // rd4 = blk_addr = row_base + blk_idx * 22
    mul.wide.u32 %rd4, %r6, 22;
    add.u64 %rd4, %rd3, %rd4;

    // f1 = scale d (fp16 at offset 0) - use b16 register for f16 conversion
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // Load qh (4 bytes at offset 2) using byte loads for unaligned access
    add.u64 %rd5, %rd4, 2;
    ld.global.u8 %r20, [%rd5];
    add.u64 %rd6, %rd4, 3;
    ld.global.u8 %r21, [%rd6];
    add.u64 %rd7, %rd4, 4;
    ld.global.u8 %r22, [%rd7];
    add.u64 %rd8, %rd4, 5;
    ld.global.u8 %r23, [%rd8];
    // Combine: qh = r20 | (r21 << 8) | (r22 << 16) | (r23 << 24)
    shl.b32 %r24, %r21, 8;
    shl.b32 %r25, %r22, 16;
    shl.b32 %r26, %r23, 24;
    or.b32 %r27, %r20, %r24;
    or.b32 %r28, %r27, %r25;
    or.b32 %r8, %r28, %r26;  // r8 = qh

    // rd9 = qs_base = blk_addr + 6
    add.u64 %rd9, %rd4, 6;

    // CANDLE LAYOUT:
    // Thread 0-15 read bytes 0-15 (low nibbles -> positions 0-15), qh bits 0-15
    // Thread 16-31 read bytes 0-15 (high nibbles -> positions 16-31), qh bits 16-31
    // r9 = byte_idx = tid < 16 ? tid : tid - 16
    setp.ge.u32 %p2, %r0, 16;
    mov.u32 %r9, %r0;
    @%p2 sub.u32 %r9, %r0, 16;

    // Load byte from qs[byte_idx]
    cvt.u64.u32 %rd10, %r9;
    add.u64 %rd10, %rd9, %rd10;
    ld.global.u8 %r10, [%rd10];

    // r11 = nibble value
    // Threads 0-15: low nibble (byte & 0xF)
    // Threads 16-31: high nibble (byte >> 4)
    mov.u32 %r11, %r10;
    @%p2 shr.u32 %r11, %r10, 4;
    and.b32 %r11, %r11, 15;

    // Extract high bit: (qh >> tid) & 1
    // For candle layout, threads 0-15 use qh bits 0-15, threads 16-31 use qh bits 16-31
    shr.b32 %r12, %r8, %r0;
    and.b32 %r12, %r12, 1;

    // Combine: q5 = nibble | (high_bit << 4)
    shl.b32 %r13, %r12, 4;
    or.b32 %r14, %r11, %r13;

    // r15 = centered value = q5 - 16 (as signed)
    sub.u32 %r15, %r14, 16;

    // f2 = dequantized = d * centered
    cvt.rn.f32.s32 %f2, %r15;
    mul.f32 %f2, %f1, %f2;

    // r16 = x_idx = blk_idx * 32 + tid
    shl.b32 %r16, %r6, 5;
    add.u32 %r16, %r16, %r0;

    // Bounds check for last block
    setp.ge.u32 %p3, %r16, %r3;
    @%p3 bra $L_skip_mul;

    // f3 = x[x_idx]
    cvt.u64.u32 %rd11, %r16;
    shl.b64 %rd11, %rd11, 2;
    add.u64 %rd11, %rd2, %rd11;
    ld.global.f32 %f3, [%rd11];

    // f0 += f2 * f3
    fma.rn.f32 %f0, %f2, %f3, %f0;

$L_skip_mul:
    add.u32 %r6, %r6, 1;
    bra $L_blk_loop;

$L_blk_loop_end:
    // Warp reduction using shfl.sync.down
    shfl.sync.down.b32 %f4, %f0, 16, 31, 0xffffffff;
    add.f32 %f0, %f0, %f4;
    shfl.sync.down.b32 %f5, %f0, 8, 31, 0xffffffff;
    add.f32 %f0, %f0, %f5;
    shfl.sync.down.b32 %f6, %f0, 4, 31, 0xffffffff;
    add.f32 %f0, %f0, %f6;
    shfl.sync.down.b32 %f7, %f0, 2, 31, 0xffffffff;
    add.f32 %f0, %f0, %f7;
    shfl.sync.down.b32 %f8, %f0, 1, 31, 0xffffffff;
    add.f32 %f0, %f0, %f8;

    // Thread 0 writes result
    setp.ne.u32 %p4, %r0, 0;
    @%p4 bra $L_exit;

    // y[ctaid] = f0
    mul.wide.u32 %rd12, %r1, 4;
    add.u64 %rd12, %rd0, %rd12;
    st.global.f32 [%rd12], %f0;

$L_exit:
    ret;
}
",
    )
}

// ============================================================================
// QWEN-007: Q8 Dequantization Kernel
// ============================================================================

/// Generate PTX for Q8 dequantization kernel
///
/// Dequantizes INT8 values to FP32 using per-block scales (block size = 32)
/// Formula: output[i] = quants[i] * scales[i / 32]
///
/// Parameters:
/// - quants: i8* input quantized values
/// - scales: f32* per-block scale factors
/// - output: f32* dequantized output
/// - n: u32 number of elements
fn generate_q8_dequant_ptx(_n: u32) -> String {
    // Note: n is used by caller for launch config, not embedded in PTX
    // The kernel uses n_param from arguments for bounds checking
    r"
.version 8.0
.target sm_89
.address_size 64

.visible .entry q8_dequant(
    .param .u64 quants_ptr,
    .param .u64 scales_ptr,
    .param .u64 output_ptr,
    .param .u32 n_param
) {{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    .reg .b16 %h<2>;

    // Get global thread index
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;  // global_idx = blockIdx * blockDim + threadIdx

    // Load n parameter
    ld.param.u32 %r4, [n_param];

    // Bounds check
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $L_exit;

    // Load pointers
    ld.param.u64 %rd0, [quants_ptr];
    ld.param.u64 %rd1, [scales_ptr];
    ld.param.u64 %rd2, [output_ptr];

    // Calculate quants address: quants_ptr + global_idx
    cvt.u64.u32 %rd3, %r3;
    add.u64 %rd4, %rd0, %rd3;

    // Load quantized value (i8)
    ld.global.s8 %h0, [%rd4];
    cvt.rn.f32.s16 %f0, %h0;  // Convert i8 to f32

    // Calculate scale index: global_idx / 32
    shr.u32 %r5, %r3, 5;  // scale_idx = global_idx >> 5

    // Calculate scales address: scales_ptr + scale_idx * 4
    cvt.u64.u32 %rd5, %r5;
    shl.b64 %rd5, %rd5, 2;  // scale_idx * 4 (bytes)
    add.u64 %rd6, %rd1, %rd5;

    // Load scale (f32)
    ld.global.f32 %f1, [%rd6];

    // Dequantize: output = quant * scale
    mul.f32 %f2, %f0, %f1;

    // Calculate output address: output_ptr + global_idx * 4
    shl.b64 %rd3, %rd3, 2;  // global_idx * 4 (bytes)
    add.u64 %rd7, %rd2, %rd3;

    // Store result
    st.global.f32 [%rd7], %f2;

$L_exit:
    ret;
}}
".to_string()
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod kernels_tests;
