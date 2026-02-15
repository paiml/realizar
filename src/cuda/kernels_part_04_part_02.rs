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
            // PAR-082-V2: Multi-warp Vectorized Q4K GEMV (configurable warps + u32 loads)
            KernelType::MwvQ4KGemv { k, n, num_warps } => {
                let mut kernel = MultiWarpVectorizedQ4KGemvKernel::new(*k, *n);
                kernel.num_warps = *num_warps;
                kernel.emit_ptx()
            },
            // PAR-082-V4: Multi-warp DP4A Q4K GEMV with Q8_1 activations
            KernelType::MwvDp4aQ4KGemv { k, n, num_warps } => {
                let mut kernel = MwvDp4aQ4KGemvKernel::new(*k, *n);
                kernel.num_warps = *num_warps;
                kernel.emit_ptx()
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
}
