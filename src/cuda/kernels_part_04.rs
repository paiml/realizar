
impl CudaKernels {

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
            // PAR-082-V2: Multi-warp Vectorized Q4K GEMV
            KernelType::MwvQ4KGemv { .. } => "mwv_q4k_gemv",
            // PAR-082-V4: Multi-warp DP4A Q4K GEMV
            KernelType::MwvDp4aQ4KGemv { .. } => "mwv_dp4a_q4k_gemv",
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

include!("kernels_part_04_part_02.rs");
