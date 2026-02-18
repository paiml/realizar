
/// CUDA kernel types supported by realizar
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum KernelType {
    /// Naive GEMM (simple, for reference)
    GemmNaive {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Tiled GEMM with shared memory
    GemmTiled {
        m: u32,
        n: u32,
        k: u32,
        tile_size: u32,
    },
    /// Tensor Core GEMM (fp16)
    GemmTensorCore {
        m: u32,
        n: u32,
        k: u32,
    },
    /// GEMV (General Matrix-Vector Multiply) - optimized for M=1 (single token generation)
    Gemv {
        k: u32,
        n: u32,
    },
    /// Coalesced GEMV - high-bandwidth M=1 kernel with memory coalescing
    CoalescedGemv {
        k: u32,
        n: u32,
    },
    /// Numerically stable softmax
    Softmax {
        dim: u32,
    },
    /// Layer normalization
    LayerNorm {
        hidden_size: u32,
        epsilon: f32,
        affine: bool,
    },
    /// FlashAttention-style attention (single head)
    Attention {
        seq_len: u32,
        head_dim: u32,
        causal: bool,
    },
    /// Multi-head attention with parallel head processing (PARITY-043)
    MultiHeadAttention {
        seq_len: u32,
        head_dim: u32,
        n_heads: u32,
        causal: bool,
    },
    /// Tensor Core FlashAttention (FP16 WMMA) - REALIZAR-PARITY-001.3
    AttentionTensorCore {
        seq_len: u32,
        head_dim: u32,
        n_heads: u32,
        causal: bool,
    },
    /// Q4_K quantized GEMM (fused dequantization) - simplified format
    QuantizedGemm {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Q4_K quantized GEMM (fused dequantization) - GGML super-block format (PARITY-041)
    QuantizedGemmGgml {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Q5_K quantized GEMM (fused dequantization) - GGML super-block format (PARITY-116)
    Q5KQuantizedGemm {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Q6_K quantized GEMM (fused dequantization) - GGML super-block format (PARITY-117)
    Q6KQuantizedGemm {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Optimized GEMM with register blocking (IMP-900a)
    GemmOptimized {
        m: u32,
        n: u32,
        k: u32,
        tile_size: u32,
        reg_block: u32,
    },
    /// Fused GEMM + bias + activation (IMP-900b)
    GemmBiasActivation {
        m: u32,
        n: u32,
        k: u32,
        activation: u32,
    },
    /// Element-wise bias + activation epilogue (IMP-1000)
    BiasActivation {
        n: u32,
        bias_size: u32,
        activation: u32,
    },
    /// FP16 Tensor Core GEMM with WMMA intrinsics (IMP-1000a)
    GemmFp16TensorCore {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Fused Q4_K × Q8_0 dot product kernel (PARITY-073)
    FusedQ4Q8Dot {
        n: u32,
    },
    /// Q4_K quantized GEMV (fused dequantization) - PAR-003
    Q4KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-041: Tiled Q4_K GEMV with shared memory input caching
    TiledQ4KGemv {
        k: u32,
        n: u32,
        outputs_per_block: u32,
    },
    /// PAR-056: Chunked Tiled Q4_K GEMV for large K dimensions
    ChunkedTiledQ4KGemv {
        k: u32,
        n: u32,
        outputs_per_block: u32,
    },
    /// PAR-062: Coalesced Q4_K GEMV with bandwidth-optimized memory access
    CoalescedQ4KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-132: Wide Q4_K GEMV with 256 threads (8 warps) per output
    WideQ4KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-069: Vectorized Q4_K GEMV with coalesced u32 weight loads
    VectorizedQ4KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-082-V2: Multi-warp Vectorized Q4_K GEMV
    MwvQ4KGemv {
        k: u32,
        n: u32,
        num_warps: u32,
    },
    /// PAR-082-V4: Multi-warp DP4A Q4_K GEMV with Q8_1-quantized activations
    MwvDp4aQ4KGemv {
        k: u32,
        n: u32,
        num_warps: u32,
    },
    /// PAR-063: DP4A-based Q4_K GEMV with 4x instruction reduction
    Dp4aQ4KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-063-V2: DP4A SIMD Q4_K GEMV with true integer accumulation
    Dp4aSIMDQ4KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-063-V4: Q8 Quantization kernel for activations
    Q8Quantize {
        n: u32,
    },
    /// PAR-063-V5: Q4K × Q8 dot product kernel using integer arithmetic
    Q4KQ8Dot {
        k: u32,
        n: u32,
    },
    /// PAR-063-V6: Packed DP4A Q4K×Q8 kernel with true dp4a.u32.s32 instruction
    PackedDp4aQ4KQ8 {
        k: u32,
        n: u32,
    },
    /// PAR-063-V3: True DP4A Q4K GEMV with proper nibble expansion
    TrueDp4aQ4KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-094: Tensor Core Q4K GEMM for batched speculative decode
    TensorCoreQ4KGemm {
        m: u32,
        k: u32,
        n: u32,
    },
    /// PAR-108: Batched Q4_K GEMV for 2x Ollama via shared dequantization
    BatchedQ4KGemv {
        m: u32,
        k: u32,
        n: u32,
    },
    /// PAR-129: Multi-warp batched Q4_K GEMV for M=16/32
    MultiWarpBatchedQ4KGemv {
        k: u32,
        n: u32,
        warps: u32,
    },
    /// Q5_K quantized GEMV (fused dequantization) - PAR-003
    Q5KGemv {
        k: u32,
        n: u32,
    },
    /// Q6_K quantized GEMV (fused dequantization) - PAR-003
    Q6KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-066: Coalesced Q6_K GEMV with vectorized scale loading
    CoalescedQ6KGemv {
        k: u32,
        n: u32,
    },
    /// PAR-130: Batched Q6_K GEMV for M>1 batch processing
    BatchedQ6KGemv {
        k: u32,
        n: u32,
        m: u32,
    },
    /// PAR-053: FP16 Q4_K GEMV - 2x bandwidth savings vs FP32
    Fp16Q4KGemv {
        k: u32,
        n: u32,
    },
    /// Q8_0 quantized GEMV (fused dequantization) - PAR-058
    Q8_0Gemv {
        k: u32,
        n: u32,
    },
    /// Q5_0 quantized GEMV (fused dequantization) - PAR-058
    Q5_0Gemv {
        k: u32,
        n: u32,
    },
    /// Q4_0 quantized GEMV (fused dequantization) - PAR-058
    Q4_0Gemv {
        k: u32,
        n: u32,
    },
    /// Q4_1 quantized GEMV (fused dequantization) - PAR-058
    Q4_1Gemv {
        k: u32,
        n: u32,
    },
    /// Incremental attention for M=1 autoregressive decoding (PAR-020 + PAR-021)
    IncrementalAttention {
        max_seq_len: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        indirect: bool,
    },
    /// PAR-070: Multi-warp incremental attention for decode phase
    MultiWarpAttention {
        max_seq_len: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        num_warps_per_head: u32,
        indirect: bool,
    },
    /// PAR-052: KV Cache Scatter kernel
    KvCacheScatter {
        num_kv_heads: u32,
        head_dim: u32,
        max_len: u32,
    },
    /// PAR-054: KV Cache Scatter with Indirect Position (CUDA Graph Compatible)
    KvCacheScatterIndirect {
        num_kv_heads: u32,
        head_dim: u32,
        max_len: u32,
    },
    /// PAR-023: RMSNorm kernel (Root Mean Square Layer Normalization)
    RmsNorm {
        hidden_size: u32,
        epsilon: f32,
    },
    /// PAR-081: Vectorized RMSNorm kernel with 256 threads
    VectorizedRmsNorm {
        hidden_size: u32,
        epsilon: f32,
    },
    /// PAR-112: Batched Vectorized RMSNorm kernel
    BatchedVectorizedRmsNorm {
        hidden_size: u32,
        batch_size: u32,
        epsilon: f32,
    },
    /// CORRECTNESS-013: High-precision RMSNorm kernel for CPU/GPU bit-exactness
    PreciseRmsNorm {
        hidden_size: u32,
        epsilon: f32,
    },
    /// GH-280: Per-head QK RMSNorm (Qwen3)
    PerHeadRmsNorm {
        head_dim: u32,
        num_heads: u32,
        epsilon: f32,
    },
    /// PAR-114: Batched RoPE kernel
    BatchedRope {
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
        theta: f32,
    },
    /// PAR-114: Batched Residual Add kernel
    BatchedResidualAdd {
        n: u32,
        batch_size: u32,
    },
    /// PAR-114: Batched SwiGLU kernel
    BatchedSwiglu {
        n: u32,
        batch_size: u32,
    },
    /// PAR-023: Residual Add kernel for async pipeline
    ResidualAdd {
        n: u32,
    },
    /// PAR-023: Fused Residual Add + RMSNorm kernel
    FusedResidualRmsNorm {
        hidden_size: u32,
        epsilon: f32,
    },
    /// PAR-076: Fused RMSNorm + Q4K GEMV kernel
    FusedRmsNormQ4KGemv {
        k: u32,
        n: u32,
        epsilon: f32,
    },

    /// PAR-077: Fused gate + up Q4K GEMV kernel
    FusedGateUpQ4KGemv {
        k: u32,
        n: u32,
    },

    /// QWEN-009: 3-way fused kernel: RMSNorm → Gate/Up Q4K GEMV → SwiGLU
    FusedRmsNormGateUpSwigluQ4K {
        k: u32,
        n: u32,
        epsilon: f32,
    },

    // =========================================================================
    // PAR-023: Activation and Element-wise Kernels for GPU-Resident Pipeline
    // =========================================================================
    /// SiLU activation: output = x * sigmoid(x)
    Silu {
        n: u32,
    },

    /// GELU activation: output ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    Gelu {
        n: u32,
    },

    /// Element-wise multiply: output = input1 * input2
    ElementwiseMul {
        n: u32,
    },

    /// Fused SwiGLU: output = silu(gate) * up
    FusedSwiglu {
        n: u32,
    },

    /// PMAT-PERF-009: Fused Q/K/V projection kernel
    FusedQKV {
        hidden_size: u32,
        kv_dim: u32,
    },

    /// PMAT-PERF-009: Fused Gate+Up FFN kernel with SwiGLU
    FusedGateUp {
        hidden_size: u32,
        intermediate_size: u32,
    },

    /// PAR-060: RoPE (Rotary Position Embedding) kernel
    Rope {
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    },
    /// PAR-054: RoPE with Indirect Position (CUDA Graph Compatible)
    RopeIndirect {
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    },
    /// CORRECTNESS-011: RoPE NEOX style (split halves)
    RopeNeox {
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    },
    /// CORRECTNESS-011: RoPE NEOX Indirect (CUDA Graph compatible)
    RopeNeoxIndirect {
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    },
    /// CORRECTNESS-013: Precise RoPE NEOX Indirect (no .approx trig)
    PreciseRopeNeoxIndirect {
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    },
    /// PAR-062: ArgMax block reduction kernel
    ArgMax {
        length: u32,
    },
    /// PAR-062: ArgMax final reduction kernel
    ArgMaxFinal {
        num_blocks: u32,
    },
    /// QWEN-007: Q8 dequantization kernel for KV cache
    Q8Dequant {
        n: u32,
    },
}
