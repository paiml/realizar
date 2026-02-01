use super::*;
use crate::cuda::memory::{SizeClass, TransferMode};
use crate::cuda::pipeline::{
use proptest::prelude::*;
use serial_test::serial;
#[test]
fn test_tcov001f_tiled_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::TiledQ4KGemv {
        k: 4096,
        n: 4096,
        outputs_per_block: 8,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001g_chunked_tiled_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ChunkedTiledQ4KGemv {
        k: 4096,
        n: 4096,
        outputs_per_block: 8,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001h_coalesced_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::CoalescedQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001i_vectorized_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::VectorizedQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001j_dp4a_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Dp4aQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001k_dp4a_simd_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Dp4aSIMDQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001l_q5k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q5KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001m_q6k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q6KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001n_coalesced_q6k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::CoalescedQ6KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001o_batched_q6k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedQ6KGemv {
        k: 4096,
        n: 4096,
        m: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001p_fp16_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Fp16Q4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001q_q8_0_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q8_0Gemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001r_q5_0_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q5_0Gemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001s_q4_0_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q4_0Gemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001t_q4_1_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q4_1Gemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001u_incremental_attention_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::IncrementalAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 8,
        indirect: false,
    });
    assert!(ptx.contains(".version"));

    // Test with indirect=true
    let ptx_indirect = kernels.generate_ptx(&KernelType::IncrementalAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 8,
        indirect: true,
    });
    assert!(ptx_indirect.contains(".version"));
}

#[test]
fn test_tcov001v_multi_warp_attention_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::MultiWarpAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 8,
        num_warps_per_head: 4,
        indirect: false,
    });
    assert!(ptx.contains(".version"));

    // Test with indirect=true
    let ptx_indirect = kernels.generate_ptx(&KernelType::MultiWarpAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 8,
        num_warps_per_head: 4,
        indirect: true,
    });
    assert!(ptx_indirect.contains(".version"));
}

#[test]
fn test_tcov001w_kv_cache_scatter_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::KvCacheScatter {
        num_kv_heads: 8,
        head_dim: 64,
        max_len: 2048,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001x_kv_cache_scatter_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::KvCacheScatterIndirect {
        num_kv_heads: 8,
        head_dim: 64,
        max_len: 2048,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001y_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::RmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001z_vectorized_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::VectorizedRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001aa_batched_vectorized_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedVectorizedRmsNorm {
        hidden_size: 4096,
        batch_size: 4,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ab_precise_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::PreciseRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ac_batched_rope_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedRope {
        num_heads: 32,
        head_dim: 64,
        batch_size: 4,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ad_batched_residual_add_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedResidualAdd {
        n: 4096,
        batch_size: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ae_batched_swiglu_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedSwiglu {
        n: 4096,
        batch_size: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001af_residual_add_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ResidualAdd { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ag_fused_residual_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedResidualRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ah_fused_rmsnorm_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedRmsNormQ4KGemv {
        k: 4096,
        n: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ai_fused_gate_up_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedGateUpQ4KGemv { k: 4096, n: 11008 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001aj_packed_dp4a_q4k_q8_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::PackedDp4aQ4KQ8 { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ak_true_dp4a_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::TrueDp4aQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001al_q4kq8_dot_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q4KQ8Dot { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001am_q8_quantize_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q8Quantize { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001an_batched_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedQ4KGemv {
        k: 4096,
        n: 4096,
        m: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ao_multi_warp_batched_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::MultiWarpBatchedQ4KGemv {
        k: 4096,
        n: 4096,
        warps: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ap_tensor_core_q4k_gemm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::TensorCoreQ4KGemm {
        m: 1,
        n: 4096,
        k: 4096,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001aq_precise_rope_neox_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::PreciseRopeNeoxIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ar_rope_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Rope {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001as_rope_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::RopeIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001at_rope_neox_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::RopeNeox {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001au_rope_neox_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::RopeNeoxIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001av_precise_rope_neox_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::PreciseRopeNeoxIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001aw_silu_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Silu { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ax_gelu_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Gelu { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ay_elementwise_mul_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ElementwiseMul { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001az_fused_swiglu_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedSwiglu { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ba_fused_qkv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedQKV {
        hidden_size: 4096,
        kv_dim: 4096,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bb_fused_gate_up_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedGateUp {
        hidden_size: 4096,
        intermediate_size: 11008,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bc_argmax_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ArgMax { length: 32000 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bd_argmax_final_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ArgMaxFinal { num_blocks: 128 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001be_gemm_optimized_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmOptimized {
        m: 64,
        n: 64,
        k: 64,
        tile_size: 32,
        reg_block: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bf_q5k_quantized_gemm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q5KQuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bg_q6k_quantized_gemm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q6KQuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    });
    assert!(ptx.contains(".version"));
}

// =========================================================================
// T-COV-002: kernel_name() coverage - verify names are non-empty
// =========================================================================

#[test]
fn test_tcov002_kernel_names_non_empty() {
    let kernels = CudaKernels::new();

    // Test kernel types return valid non-empty names
    let test_cases: Vec<KernelType> = vec![
        KernelType::GemmNaive {
            m: 64,
            n: 64,
            k: 64,
        },
        KernelType::GemmTiled {
            m: 64,
            n: 64,
            k: 64,
            tile_size: 32,
        },
        KernelType::GemmTensorCore {
            m: 64,
            n: 64,
            k: 64,
        },
        KernelType::Gemv { k: 4096, n: 4096 },
        KernelType::Softmax { dim: 4096 },
        KernelType::LayerNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
            affine: true,
        },
        KernelType::Attention {
            seq_len: 128,
            head_dim: 64,
            causal: false,
        },
        KernelType::RmsNorm {
            hidden_size: 4096,
            epsilon: 1e-6,
        },
        KernelType::ResidualAdd { n: 4096 },
    ];

    for kernel_type in test_cases {
        let name = kernels.kernel_name(&kernel_type);
        assert!(
            !name.is_empty(),
            "KernelType {:?} should have non-empty name",
            kernel_type
        );
    }
}

// =========================================================================
// T-COV-003: WeightQuantType coverage
// =========================================================================

#[test]
fn test_tcov003a_weight_quant_type_from_ggml() {
    // Test all GGML type mappings
    assert_eq!(
        WeightQuantType::from_ggml_type(2),
        Some(WeightQuantType::Q4_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(3),
        Some(WeightQuantType::Q4_1)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(6),
        Some(WeightQuantType::Q5_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(8),
        Some(WeightQuantType::Q8_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(12),
        Some(WeightQuantType::Q4K)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(13),
        Some(WeightQuantType::Q5K)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(14),
        Some(WeightQuantType::Q6K)
    );

    // Unknown types
    assert_eq!(WeightQuantType::from_ggml_type(255), None);
    assert_eq!(WeightQuantType::from_ggml_type(0), None);
}

#[test]
fn test_tcov003b_weight_quant_type_bytes() {
    // Q4_K: 256 values, 144 bytes per superblock
    assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);

    // Q5_K: 256 values, 176 bytes per superblock
    assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);

    // Q6_K: 256 values, 210 bytes per superblock
    assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);

    // Q4_0: 32 values, 18 bytes per block (8 blocks = 144 bytes per 256)
    assert_eq!(WeightQuantType::Q4_0.bytes_per_superblock(), 18 * 8);

    // Q4_1: 32 values, 20 bytes per block
    assert_eq!(WeightQuantType::Q4_1.bytes_per_superblock(), 20 * 8);

    // Q5_0: 32 values, 22 bytes per block
    assert_eq!(WeightQuantType::Q5_0.bytes_per_superblock(), 22 * 8);

    // Q8_0: 32 values, 34 bytes per block
    assert_eq!(WeightQuantType::Q8_0.bytes_per_superblock(), 34 * 8);
}

#[test]
fn test_tcov003c_weight_quant_type_matches_size() {
    // Q4_K: n_rows × n_cols / 256 superblocks × 144 bytes
    let rows = 4096;
    let cols = 4096;
    let q4k_size = (rows * cols / 256) * 144;
    assert!(WeightQuantType::Q4K.matches_size(q4k_size, rows, cols));

    // Q6_K: n_rows × n_cols / 256 superblocks × 210 bytes
    let q6k_size = (rows * cols / 256) * 210;
    assert!(WeightQuantType::Q6K.matches_size(q6k_size, rows, cols));

    // Wrong size should not match
    assert!(!WeightQuantType::Q4K.matches_size(q6k_size, rows, cols));
}

#[test]
fn test_tcov003d_weight_quant_type_from_size() {
    let rows = 4096;
    let cols = 4096;

    // Q4_K detection
    let q4k_size = (rows * cols / 256) * 144;
    let detected = WeightQuantType::from_size(q4k_size, rows, cols);
    assert_eq!(detected, Some(WeightQuantType::Q4K));

    // Q6_K detection
    let q6k_size = (rows * cols / 256) * 210;
    let detected = WeightQuantType::from_size(q6k_size, rows, cols);
    assert_eq!(detected, Some(WeightQuantType::Q6K));

    // Q8_0 detection (small block format)
    let q8_0_size = (rows * cols / 32) * 34;
    let detected = WeightQuantType::from_size(q8_0_size, rows, cols);
    assert_eq!(detected, Some(WeightQuantType::Q8_0));
}

// =========================================================================
// T-COV-004: SizeClass coverage
// =========================================================================

#[test]
fn test_tcov004a_size_class_for_size() {
    // Various sizes
    let small = SizeClass::for_size(1024);
    assert!(small.is_some());

    let medium = SizeClass::for_size(64 * 1024);
    assert!(medium.is_some());

    let large = SizeClass::for_size(1024 * 1024);
    assert!(large.is_some());

    // Very large sizes may or may not be supported
    let very_large = SizeClass::for_size(200_000_000);
    // Just verify it doesn't panic
    let _ = very_large;
}

#[test]
fn test_tcov004b_size_class_bytes() {
    // Get a size class and verify bytes() returns a value
    if let Some(class) = SizeClass::for_size(1024) {
        let bytes = class.bytes();
        assert!(bytes >= 1024);
    }
}

// =========================================================================
// T-COV-005: GpuMemoryPool coverage
// =========================================================================

#[test]
fn test_tcov005a_gpu_memory_pool_basic() {
    let mut pool = GpuMemoryPool::new();

    // Record allocation/deallocation
    pool.record_allocation(1024);
    pool.record_deallocation(1024);
}

#[test]
fn test_tcov005b_gpu_memory_pool_with_max_size() {
    let max_size = 128 * 1024 * 1024;
    let pool = GpuMemoryPool::with_max_size(max_size);
    // Pool created with custom max size
    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 0);
}

#[test]
fn test_tcov005c_gpu_memory_pool_stats() {
    let mut pool = GpuMemoryPool::new();

    // Record several operations
    for i in 0..10 {
        pool.record_allocation((i + 1) * 1024);
    }

    let stats = pool.stats();
    assert!(stats.peak_usage > 0);
}

#[test]
fn test_tcov005d_gpu_memory_pool_try_get() {
    let mut pool = GpuMemoryPool::new();

    // Try to get a buffer - should fail (empty pool)
    let result = pool.try_get(1024);
    assert!(result.is_none()); // Pool starts empty

    // Verify miss was recorded
    let stats = pool.stats();
    assert_eq!(stats.pool_misses, 1);
}

// =========================================================================
// T-COV-006: StagingBufferPool extended coverage
// =========================================================================

#[test]
fn test_tcov006a_staging_pool_with_max_size() {
    let max_size = 64 * 1024 * 1024;
    let pool = StagingBufferPool::with_max_size(max_size);
    let stats = pool.stats();
    assert_eq!(stats.free_buffers, 0);
}

#[test]
fn test_tcov006b_staging_pool_size_classes() {
    let mut pool = StagingBufferPool::new();

    // Request different size classes
    let buf_tiny = pool.get(512);
    let buf_small = pool.get(8 * 1024);
    let buf_medium = pool.get(128 * 1024);

    // Return them
    pool.put(buf_tiny);
    pool.put(buf_small);
    pool.put(buf_medium);

    let stats = pool.stats();
    assert!(stats.free_buffers >= 3);
}

// =========================================================================
// T-COV-007: Presets module coverage
// =========================================================================

#[test]
fn test_tcov007_presets_coverage() {
    // Test all preset functions
    let llama_attn = presets::llama_attention(2048, 128);
    match llama_attn {
        KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 128);
            assert!(causal);
        },
        _ => panic!("Expected Attention kernel"),
    }

    let ffn = presets::ffn_gemm(1, 4096, 11008);
    match ffn {
        KernelType::GemmTiled { m, n, k, tile_size } => {
            assert_eq!(m, 1);
            assert_eq!(n, 11008);
            assert_eq!(k, 4096);
            assert_eq!(tile_size, 32);
        },
        _ => panic!("Expected GemmTiled kernel"),
    }

    let q4k = presets::q4k_inference(1, 4096, 4096);
    match q4k {
        KernelType::QuantizedGemm { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected QuantizedGemm kernel"),
    }

    let rmsnorm = presets::rmsnorm(4096);
    match rmsnorm {
        KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine,
        } => {
            assert_eq!(hidden_size, 4096);
            assert!(epsilon > 0.0);
            assert!(!affine); // RMSNorm preset uses affine=false
        },
        _ => panic!("Expected LayerNorm kernel (preset::rmsnorm returns LayerNorm)"),
    }

    let mha = presets::multi_head_attention(2048, 64, 32);
    match mha {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected MultiHeadAttention kernel"),
    }

    let phi2_mha = presets::phi2_multi_head_attention(2048);
    match phi2_mha {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 80);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected MultiHeadAttention kernel"),
    }

    let tc_attn = presets::tensor_core_attention(2048, 64, 32);
    match tc_attn {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected AttentionTensorCore kernel"),
    }

    let llama_tc = presets::llama_tensor_core_attention(2048);
    match llama_tc {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 128);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected AttentionTensorCore kernel"),
    }
}

// =========================================================================
// T-COV-009: CudaKernels::cuda_likely_available() coverage
// =========================================================================

#[test]
fn test_tcov009_cuda_likely_available() {
    // This function checks environment heuristics
    let likely = CudaKernels::cuda_likely_available();
    // On RTX 4090, this should return true
    // The function itself should not panic
    println!("cuda_likely_available: {}", likely);
}

// =========================================================================
// T-COV-010: Additional kernel_name branch coverage (verify non-empty)
// =========================================================================

#[test]
fn test_tcov010_more_kernel_names() {
    let kernels = CudaKernels::new();

    // Verify a variety of kernel types return non-empty names
    let kernel_types: Vec<KernelType> = vec![
        KernelType::FusedRmsNormQ4KGemv {
            k: 4096,
            n: 4096,
            epsilon: 1e-6,
        },
        KernelType::FusedGateUpQ4KGemv { k: 4096, n: 11008 },
        KernelType::PackedDp4aQ4KQ8 { k: 4096, n: 4096 },
        KernelType::TrueDp4aQ4KGemv { k: 4096, n: 4096 },
        KernelType::Q4KQ8Dot { k: 4096, n: 4096 },
        KernelType::Q8Quantize { n: 4096 },
        KernelType::BatchedQ4KGemv {
            k: 4096,
            n: 4096,
            m: 4,
        },
        KernelType::MultiWarpBatchedQ4KGemv {
            k: 4096,
            n: 4096,
            warps: 4,
        },
        KernelType::TensorCoreQ4KGemm {
            m: 1,
            n: 4096,
            k: 4096,
        },
        KernelType::Rope {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeIndirect {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeNeox {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeNeoxIndirect {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::PreciseRopeNeoxIndirect {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::Silu { n: 4096 },
        KernelType::Gelu { n: 4096 },
        KernelType::ElementwiseMul { n: 4096 },
        KernelType::FusedSwiglu { n: 4096 },
        KernelType::FusedQKV {
            hidden_size: 4096,
            kv_dim: 4096,
        },
        KernelType::FusedGateUp {
            hidden_size: 4096,
            intermediate_size: 11008,
        },
        KernelType::ArgMax { length: 32000 },
        KernelType::ArgMaxFinal { num_blocks: 128 },
    ];

    for kernel_type in kernel_types {
        let name = kernels.kernel_name(&kernel_type);
        assert!(
            !name.is_empty(),
            "KernelType {:?} should have non-empty name",
            kernel_type
        );
    }
}
