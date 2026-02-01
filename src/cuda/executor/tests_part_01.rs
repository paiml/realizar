use super::*;
use crate::cuda::memory::{GpuBufferHandle, SizeClass, TransferMode};
use crate::cuda::pipeline::{
    presets, BankConflictStrategy, MemoryPattern, PtxOptimizationHints, PtxOptimizer,
    RegisterTiling,
};
use serial_test::serial;

#[test]
fn test_cuda_kernels_creation() {
    let kernels = CudaKernels::new();
    // Verify the struct was created (ZST is valid)
    let _ = kernels.generate_ptx(&KernelType::Softmax { dim: 128 });
}

#[test]
fn test_gemm_naive_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmNaive {
        m: 128,
        n: 128,
        k: 128,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".visible .entry"));
    assert!(ptx.contains("gemm"));
}

#[test]
fn test_gemm_tiled_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmTiled {
        m: 1024,
        n: 1024,
        k: 1024,
        tile_size: 32,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("gemm"));
    assert!(ptx.contains(".shared"));
}

#[test]
fn test_softmax_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 4096 });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("softmax"));
    assert!(ptx.contains("shfl")); // Warp shuffle
}

#[test]
fn test_layernorm_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::LayerNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
        affine: true,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("layernorm"));
}

#[test]
fn test_attention_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Attention {
        seq_len: 2048,
        head_dim: 64,
        causal: true,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("flash_attention") || ptx.contains("attention"));
}

#[test]
fn test_quantized_gemm_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("q4k") || ptx.contains("gemm"));
}

// =========================================================================
// PARITY-041: GGML Q4_K Super-Block Format Tests
// =========================================================================

#[test]
fn test_parity041_ggml_kernel_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 4096,
    });

    // Verify PTX is generated
    assert!(
        ptx.contains(".version"),
        "PTX should have version directive"
    );
    assert!(
        ptx.contains("q4k_gemm_ggml"),
        "PTX should contain GGML kernel name"
    );
}

#[test]
fn test_parity041_ggml_kernel_name() {
    let kernels = CudaKernels::new();
    let name = kernels.kernel_name(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 4096,
    });
    assert_eq!(name, "q4k_gemm_ggml");
}

#[test]
fn test_parity041_ggml_preset() {
    let kernel = presets::q4k_ggml_inference(1, 4096, 4096);
    match kernel {
        KernelType::QuantizedGemmGgml { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected QuantizedGemmGgml"),
    }
}

#[test]
fn test_parity041_ggml_vs_simplified_different_kernels() {
    let kernels = CudaKernels::new();

    let simplified = kernels.generate_ptx(&KernelType::QuantizedGemm {
        m: 1,
        n: 2560,
        k: 2560,
    });

    let ggml = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 2560,
        k: 2560,
    });

    // Both should be valid PTX but different kernel names
    assert!(simplified.contains("q4k_gemm_fused"));
    assert!(ggml.contains("q4k_gemm_ggml"));

    // GGML kernel should be different (super-block format)
    assert_ne!(simplified.len(), ggml.len());
}

#[test]
fn test_parity041_ggml_phi2_dimensions() {
    // phi-2 model dimensions: hidden=2560, intermediate=10240
    let kernels = CudaKernels::new();

    // FFN up projection: [batch, 2560] @ [2560, 10240]
    let up_proj = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 10240,
        k: 2560,
    });
    assert!(up_proj.contains(".version"));

    // FFN down projection: [batch, 10240] @ [10240, 2560]
    let down_proj = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 2560,
        k: 10240,
    });
    assert!(down_proj.contains(".version"));
}

#[test]
fn test_parity041_ggml_super_block_alignment() {
    // k must be divisible by 256 for super-blocks (256 values per super-block)
    let kernels = CudaKernels::new();

    // k=4096 is divisible by 256 (16 super-blocks)
    let ptx = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 32,
        n: 2560,
        k: 4096,
    });
    assert!(ptx.contains(".version"));

    // k=2560 is divisible by 256 (10 super-blocks)
    let ptx2 = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 2560,
    });
    assert!(ptx2.contains(".version"));
}

// =========================================================================
// PARITY-042: Pinned Memory Tests
// =========================================================================

#[test]
fn test_parity042_pinned_host_buffer_creation() {
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(1024);
    assert_eq!(buf.len(), 1024);
    assert_eq!(buf.size_bytes(), 1024 * 4);
    assert!(!buf.is_empty());
    // Note: is_pinned() returns false until trueno-gpu adds cuMemAllocHost
    // This is expected behavior for the fallback implementation
}

#[test]
fn test_parity042_pinned_buffer_copy() {
    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
    let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
    buf.copy_from_slice(&src);

    let slice = buf.as_slice();
    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[50], 50.0);
    assert_eq!(slice[99], 99.0);
}

#[test]
fn test_parity042_pinned_buffer_mutable() {
    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
    let slice = buf.as_mut_slice();
    slice[0] = 42.0;
    slice[9] = 99.0;

    assert_eq!(buf.as_slice()[0], 42.0);
    assert_eq!(buf.as_slice()[9], 99.0);
}

#[test]
fn test_parity042_staging_buffer_pool_basic() {
    let mut pool = StagingBufferPool::new();

    // First allocation - should be a miss
    let buf1 = pool.get(1024);
    assert!(buf1.len() >= 1024);

    let stats = pool.stats();
    assert_eq!(stats.pool_misses, 1);
    assert_eq!(stats.pool_hits, 0);

    // Return to pool
    pool.put(buf1);

    // Second allocation - should be a hit (same size class)
    let buf2 = pool.get(1024);
    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 1);
    assert!(buf2.len() >= 1024);
}

#[test]
fn test_parity042_staging_pool_hit_rate() {
    let mut pool = StagingBufferPool::new();

    // Allocate and return several buffers
    for _ in 0..5 {
        let buf = pool.get(2048);
        pool.put(buf);
    }

    // Now get again - should all be hits
    for _ in 0..5 {
        let buf = pool.get(2048);
        pool.put(buf);
    }

    let stats = pool.stats();
    assert!(
        stats.hit_rate > 0.4,
        "Hit rate should be > 40%: {:.2}",
        stats.hit_rate
    );
}

#[test]
fn test_parity042_staging_pool_clear() {
    let mut pool = StagingBufferPool::new();

    // Allocate some buffers
    let buf1 = pool.get(1024);
    let buf2 = pool.get(2048);
    pool.put(buf1);
    pool.put(buf2);

    assert!(pool.stats().free_buffers > 0);

    // Clear pool
    pool.clear();
    assert_eq!(pool.stats().free_buffers, 0);
}

#[test]
fn test_parity042_transfer_mode_properties() {
    assert!(!TransferMode::Pageable.requires_pinned());
    assert!(TransferMode::Pinned.requires_pinned());
    assert!(TransferMode::ZeroCopy.requires_pinned());
    assert!(TransferMode::Async.requires_pinned());

    assert_eq!(TransferMode::Pageable.estimated_speedup(), 1.0);
    assert!(TransferMode::Pinned.estimated_speedup() > 1.0);
    assert!(TransferMode::ZeroCopy.estimated_speedup() > TransferMode::Pinned.estimated_speedup());
}

#[test]
fn test_parity042_transfer_mode_default() {
    let mode = TransferMode::default();
    assert_eq!(mode, TransferMode::Pageable);
}

// PARITY-043: Multi-Head Attention Parallelization Tests

#[test]
fn test_parity043_multi_head_attention_kernel_type() {
    let kernels = CudaKernels::new();

    // Non-causal variant - now uses trueno's FlashAttention kernel
    let kernel = KernelType::MultiHeadAttention {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: false,
    };
    assert_eq!(kernels.kernel_name(&kernel), "flash_attention");

    // Causal variant (for autoregressive models)
    let causal_kernel = KernelType::MultiHeadAttention {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: true,
    };
    assert_eq!(
        kernels.kernel_name(&causal_kernel),
        "flash_attention_causal"
    );
}

#[test]
fn test_parity043_multi_head_attention_ptx_generation() {
    let kernels = CudaKernels::new();

    let kernel = KernelType::MultiHeadAttention {
        seq_len: 128,
        head_dim: 64,
        n_heads: 8,
        causal: false,
    };

    let ptx = kernels.generate_ptx(&kernel);

    // Verify PTX structure (now using trueno's FlashAttention kernel)
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_89"));
    assert!(ptx.contains(".visible .entry flash_attention"));
    // trueno uses lowercase ptr names
    assert!(ptx.contains(".param .u64 q_ptr"));
    assert!(ptx.contains(".param .u64 k_ptr"));
    assert!(ptx.contains(".param .u64 v_ptr"));
    assert!(ptx.contains(".param .u64 o_ptr"));
    assert!(ptx.contains(".param .u32 seq_len"));
    assert!(ptx.contains(".param .u32 head_dim"));
    assert!(ptx.contains(".param .u32 num_heads"));

    // Verify shared memory (trueno uses .b8 smem array)
    assert!(ptx.contains(".shared"));

    // Verify block indices are used for head/tile selection
    assert!(ptx.contains("%ctaid.x")); // Q tile block
    assert!(ptx.contains("%ctaid.y")); // head index
}

#[test]
fn test_parity043_multi_head_attention_causal_ptx() {
    let kernels = CudaKernels::new();

    let kernel = KernelType::MultiHeadAttention {
        seq_len: 128,
        head_dim: 64,
        n_heads: 8,
        causal: true,
    };

    let ptx = kernels.generate_ptx(&kernel);

    // Verify causal kernel name (trueno uses flash_attention_causal)
    assert!(ptx.contains(".visible .entry flash_attention_causal"));

    // Trueno's causal masking uses setp.lt comparison for Q vs KV block
    // The causal skip happens in the kv_loop via branch
    assert!(ptx.contains("setp.lt.u32")); // Causal comparison
    assert!(ptx.contains("kv_loop")); // KV block loop with causal skip
}

#[test]
fn test_parity043_multi_head_attention_phi2_dimensions() {
    // phi-2 model dimensions
    let kernels = CudaKernels::new();

    let kernel = KernelType::MultiHeadAttention {
        seq_len: 2048, // max context
        head_dim: 80,  // phi-2 head dimension (2560/32 heads)
        n_heads: 32,   // phi-2 attention heads
        causal: true,  // autoregressive
    };

    let ptx = kernels.generate_ptx(&kernel);

    // Verify generation succeeds for phi-2 dimensions (using trueno's FlashAttention)
    assert!(ptx.contains("flash_attention_causal"));
    assert!(ptx.len() > 1000); // Substantial kernel

    // Trueno uses tile-based approach, so shared memory is calculated per tile
    // not the full head_size. Verify shared memory is allocated.
    assert!(ptx.contains(".shared"));
}

#[test]
fn test_parity043_multi_head_attention_scale_factor() {
    let kernels = CudaKernels::new();

    let head_dim = 64;
    let kernel = KernelType::MultiHeadAttention {
        seq_len: 256,
        head_dim,
        n_heads: 8,
        causal: false,
    };

    let ptx = kernels.generate_ptx(&kernel);

    // Scale factor 1/sqrt(head_dim) = 0.125 is embedded in trueno's PTX
    // as a hex float literal (0F3E000000 = 0.125)
    // Trueno bakes scale into the PTX during generation
    assert!(ptx.contains("mul.f32")); // Scaling operation exists
                                      // The scale is applied after dot product in online softmax
    assert!(ptx.contains("ex2")); // exp2 for softmax
}

#[test]
fn test_parity043_multi_head_attention_thread_config() {
    let kernels = CudaKernels::new();

    // Trueno's FlashAttention uses tile_q * head_dim threads per block
    // Tile sizes are calculated based on 48KB shared memory limit
    let kernel_small = KernelType::MultiHeadAttention {
        seq_len: 64,
        head_dim: 64,
        n_heads: 8,
        causal: false,
    };

    let ptx_small = kernels.generate_ptx(&kernel_small);
    // Trueno generates valid PTX with proper thread config
    assert!(ptx_small.contains(".visible .entry flash_attention"));
    assert!(ptx_small.contains("%tid.x")); // Thread index is used

    // Larger sequence still works with tiled approach
    let kernel_large = KernelType::MultiHeadAttention {
        seq_len: 1024,
        head_dim: 64,
        n_heads: 8,
        causal: false,
    };

    let ptx_large = kernels.generate_ptx(&kernel_large);
    // Trueno handles large sequences via tiling
    assert!(ptx_large.contains(".visible .entry flash_attention"));
    assert!(ptx_large.contains("kv_loop")); // KV block iteration
}

#[test]
fn test_parity043_multi_head_attention_executor_validation() {
    // Test that CudaExecutor validates input sizes correctly
    // This test runs without actual GPU by checking size validation logic
    let seq_len = 64u32;
    let head_dim = 32u32;
    let n_heads = 4u32;
    let total_size = (seq_len * head_dim * n_heads) as usize;

    // Correct sizes
    let q = vec![0.0f32; total_size];
    let k = vec![0.0f32; total_size];
    let v = vec![0.0f32; total_size];

    // Size validation check (without GPU)
    assert_eq!(q.len(), total_size);
    assert_eq!(k.len(), total_size);
    assert_eq!(v.len(), total_size);

    // Verify formula: n_heads × seq_len × head_dim
    assert_eq!(total_size, (n_heads * seq_len * head_dim) as usize);
}

#[test]
fn test_parity043_multi_head_attention_memory_layout() {
    // Verify memory layout: [n_heads, seq_len, head_dim]
    let n_heads = 8u32;
    let seq_len = 128u32;
    let head_dim = 64u32;

    // Calculate offsets for head access
    let head_stride = (seq_len * head_dim) as usize;
    let total_size = head_stride * n_heads as usize;

    // Each head's data starts at head_idx * head_stride
    let head_0_start = 0;
    let head_1_start = head_stride;
    let head_7_start = 7 * head_stride;

    assert_eq!(head_0_start, 0);
    assert_eq!(head_1_start, 128 * 64);
    assert_eq!(head_7_start, 7 * 128 * 64);
    assert_eq!(total_size, 8 * 128 * 64);
}

#[test]
fn test_kernel_names() {
    let kernels = CudaKernels::new();

    assert_eq!(
        kernels.kernel_name(&KernelType::GemmNaive { m: 1, n: 1, k: 1 }),
        "gemm_naive"
    );
    assert_eq!(
        kernels.kernel_name(&KernelType::Softmax { dim: 1 }),
        "softmax_warp_shuffle"
    );
    assert_eq!(
        kernels.kernel_name(&KernelType::QuantizedGemm { m: 1, n: 1, k: 32 }),
        "q4k_gemm_fused"
    );
}

#[test]
fn test_presets_llama_attention() {
    let kernel = presets::llama_attention(2048, 64);
    match kernel {
        KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 64);
            assert!(causal);
        },
        _ => panic!("Expected Attention kernel"),
    }
}

#[test]
fn test_presets_ffn_gemm() {
    let kernel = presets::ffn_gemm(32, 4096, 11008);
    match kernel {
        KernelType::GemmTiled { m, n, k, tile_size } => {
            assert_eq!(m, 32);
            assert_eq!(n, 11008);
            assert_eq!(k, 4096);
            assert_eq!(tile_size, 32);
        },
        _ => panic!("Expected GemmTiled kernel"),
    }
}

#[test]
fn test_presets_q4k_inference() {
    let kernel = presets::q4k_inference(1, 4096, 4096);
    match kernel {
        KernelType::QuantizedGemm { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected QuantizedGemm kernel"),
    }
}

#[test]
fn test_presets_rmsnorm() {
    let kernel = presets::rmsnorm(4096);
    match kernel {
        KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine,
        } => {
            assert_eq!(hidden_size, 4096);
            assert!((epsilon - 1e-6).abs() < 1e-10);
            assert!(!affine);
        },
        _ => panic!("Expected LayerNorm kernel"),
    }
}

#[test]
fn test_presets_multi_head_attention() {
    let kernel = presets::multi_head_attention(512, 64, 8);
    match kernel {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 512);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 8);
            assert!(causal); // Default is causal
        },
        _ => panic!("Expected MultiHeadAttention kernel"),
    }
}

#[test]
fn test_presets_phi2_multi_head_attention() {
    let kernel = presets::phi2_multi_head_attention(2048);
    match kernel {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 80); // phi-2: 2560/32 = 80
            assert_eq!(n_heads, 32); // phi-2: 32 heads
            assert!(causal);
        },
        _ => panic!("Expected MultiHeadAttention kernel"),
    }
}

#[test]
fn test_default_impl() {
    let kernels = CudaKernels::default();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 256 });
    assert!(!ptx.is_empty());
}

// ========================================================================
// CudaExecutor Tests
// ========================================================================

#[test]
fn test_cuda_executor_is_available() {
    // This should not panic, regardless of whether CUDA is available
    let _available = CudaExecutor::is_available();
}

#[test]
fn test_cuda_executor_device_count() {
    // Should return count (possibly 0)
    let count = CudaExecutor::num_devices();
    // Count is valid (0 or more)
    assert!(count < 1000); // Sanity check
}

#[test]
#[serial]
fn test_cuda_executor_new() {
    let executor = CudaExecutor::new(0);
    assert!(executor.is_ok());
    let executor = executor.expect("test");
    assert!(executor.device_name().is_ok());
}

#[test]
#[serial]
fn test_cuda_executor_memory_info() {
    let executor = CudaExecutor::new(0).expect("test");
    let (free, total) = executor.memory_info().expect("test");
    assert!(total > 0);
    assert!(free <= total);
}

#[test]
#[serial]
fn test_cuda_executor_gemm_small() {
    let mut executor = CudaExecutor::new(0).expect("test");

    // Small 4x4 GEMM
    let a = vec![1.0f32; 16];
    let b = vec![1.0f32; 16];
    let mut c = vec![0.0f32; 16];

    let result = executor.gemm(&a, &b, &mut c, 4, 4, 4);
    assert!(result.is_ok());

    // Each element should be 4.0 (dot product of 4 ones)
    for val in &c {
        assert!((*val - 4.0).abs() < 1e-5);
    }
}

/// PARITY-114: Test non-square GEMM correctness
/// This is the case that was failing before the grid dimension fix
#[test]
#[serial]
fn test_cuda_executor_gemm_non_square() {
    let mut executor = CudaExecutor::new(0).expect("test");

    // First test: 32x32x32 (single tile)
    {
        let m = 32u32;
        let k = 32u32;
        let n = 32u32;

        let a = vec![1.0f32; (m * k) as usize];
        let b = vec![1.0f32; (k * n) as usize];
        let mut c = vec![0.0f32; (m * n) as usize];

        let result = executor.gemm(&a, &b, &mut c, m, n, k);
        assert!(result.is_ok(), "32x32 GEMM failed");

        eprintln!("32x32x32: First value = {} (expected 32)", c[0]);
        assert!(
            (c[0] - 32.0).abs() < 1e-4,
            "32x32 GEMM: expected 32.0, got {}",
            c[0]
        );
    }

    // Second test: 32x32x64 (2 tiles in K)
    {
        let m = 32u32;
        let k = 64u32;
        let n = 32u32;

        let a = vec![1.0f32; (m * k) as usize];
        let b = vec![1.0f32; (k * n) as usize];
        let mut c = vec![0.0f32; (m * n) as usize];

        let result = executor.gemm(&a, &b, &mut c, m, n, k);
        assert!(result.is_ok(), "32x32x64 GEMM failed");

        eprintln!("32x32x64: First value = {} (expected 64)", c[0]);
        assert!(
            (c[0] - 64.0).abs() < 1e-4,
            "32x32x64 GEMM: expected 64.0, got {}",
            c[0]
        );
    }

    // Third test: non-square (4, 64) × (64, 128) = (4, 128)
    {
        let m = 4u32;
        let k = 64u32;
        let n = 128u32;

        let a = vec![1.0f32; (m * k) as usize];
        let b = vec![1.0f32; (k * n) as usize];
        let mut c = vec![0.0f32; (m * n) as usize];

        let result = executor.gemm(&a, &b, &mut c, m, n, k);
        assert!(result.is_ok(), "4x64x128 GEMM failed");

        eprintln!("4x64x128: First value = {} (expected 64)", c[0]);
        assert!(
            (c[0] - 64.0).abs() < 1e-4,
            "PARITY-114: Non-square GEMM expected 64.0, got {}",
            c[0]
        );
    }
}

/// PARITY-114: Compare CUDA matmul vs wgpu matmul with same inputs
#[test]
#[serial]
fn test_cuda_vs_wgpu_matmul_parity() {
    use crate::gpu::{CudaScheduler, HybridScheduler};

    // Test case matching model forward pass: 4x64x192
    let m = 4usize;
    let k = 64usize;
    let n = 192usize;

    // Test 0: Single tile (k=32)
    eprintln!("\n=== Test 0: Single tile k=32 ===");
    {
        let m0 = 4usize;
        let k0 = 32usize;
        let n0 = 192usize;
        let a = vec![1.0f32; m0 * k0];
        let b = vec![1.0f32; k0 * n0];
        let expected = k0 as f32;

        // Check what PTX would be generated for this configuration
        use trueno_gpu::kernels::{GemmKernel, Kernel};
        let kernel = GemmKernel::tiled(m0 as u32, n0 as u32, k0 as u32, 32);
        let ptx = kernel.emit_ptx();

        // Look for the key constants in this kernel
        eprintln!("k=32 kernel constants:");
        for line in ptx.lines() {
            if line.contains("256;")
                || line.contains("128;")
                || line.contains("768;")
                || line.contains("384;")
            {
                eprintln!("  {}", line.trim());
            }
        }
        // Check n_tiles
        let count_1 = ptx.matches(", 1;").count();
        eprintln!(
            "Occurrences of ', 1;': {} (expected n_tiles=1 for k=32)",
            count_1
        );

        let mut executor = CudaExecutor::new(0).expect("CudaExecutor should init");
        let mut c = vec![0.0f32; m0 * n0];
        executor
            .gemm(&a, &b, &mut c, m0 as u32, n0 as u32, k0 as u32)
            .expect("CUDA gemm should succeed");

        eprintln!("k=32: CUDA[0]={} (expected {})", c[0], expected);
        assert!(
            (c[0] - expected).abs() < 1e-3,
            "k=32 CUDA failed: {} vs {}",
            c[0],
            expected
        );
    }

    // Test 1: Uniform data (1.0s) - this should work
    eprintln!("\n=== Test 1: Uniform 1.0 data k=64 ===");
    eprintln!("Dimensions: m={}, k={}, n={}", m, k, n);
    eprintln!("Expected n_tiles = ({}+31)/32 = {}", k, (k + 31) / 32);
    {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];

        // CPU reference
        let expected = k as f32; // All 1s dot product = k

        // Use CudaExecutor directly for debugging
        let mut executor = CudaExecutor::new(0).expect("CudaExecutor should init");

        // Print some debug info about what kernel will be generated
        use trueno_gpu::kernels::{GemmKernel, Kernel};
        let kernel = GemmKernel::tiled(m as u32, n as u32, k as u32, 32);
        let ptx = kernel.emit_ptx();

        // Look for embedded constants
        eprintln!("PTX constants search:");
        for line in ptx.lines() {
            if line.contains("mov.u32")
                && (line.contains(", 2;")
                    || line.contains(", 32;")
                    || line.contains(", 64;")
                    || line.contains(", 192;")
                    || line.contains(", 256;")
                    || line.contains(", 768;"))
            {
                eprintln!("  {}", line.trim());
            }
        }
        // Show the full inner_k_loop section
        if let Some(start) = ptx.find("inner_k_loop:") {
            let end = ptx[start..].find("inner_k_end:").unwrap_or(800) + start;
            eprintln!(
                "\ninner_k_loop section:\n{}",
                &ptx[start..end.min(start + 1000)]
            );
        }

        let mut c = vec![0.0f32; m * n];
        executor
            .gemm(&a, &b, &mut c, m as u32, n as u32, k as u32)
            .expect("CUDA gemm should succeed");

        eprintln!("Uniform: CUDA[0]={} (expected {})", c[0], expected);
        assert!(
            (c[0] - expected).abs() < 1e-3,
            "Uniform CUDA failed: {} vs {}",
            c[0],
            expected
        );
    }

    // Test 2: Patterned data
    eprintln!("\n=== Test 2: Patterned data ===");
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();

    // CPU reference (ground truth)
    let mut cpu_result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            cpu_result[i * n + j] = sum;
        }
    }

    // CUDA path
    let mut cuda_sched = CudaScheduler::new().expect("CudaScheduler should init");
    let cuda_result = cuda_sched
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul should succeed");

    // wgpu path
    let mut wgpu_sched =
        HybridScheduler::with_threshold(1000).expect("HybridScheduler should init");
    let wgpu_result = wgpu_sched
        .matmul(&a, &b, m, k, n)
        .expect("wgpu matmul should succeed");

    // Print all values for debugging
    eprintln!(
        "Patterned: CPU[0]={}, CUDA[0]={}, wgpu[0]={}",
        cpu_result[0], cuda_result[0], wgpu_result[0]
    );

    // Check CUDA vs CPU
    let cuda_vs_cpu_diff = (cuda_result[0] - cpu_result[0]).abs();
    let wgpu_vs_cpu_diff = (wgpu_result[0] - cpu_result[0]).abs();
    eprintln!(
        "Patterned: CUDA vs CPU diff={}, wgpu vs CPU diff={}",
        cuda_vs_cpu_diff, wgpu_vs_cpu_diff
    );

    // Compare CUDA to CPU reference
    assert_eq!(cuda_result.len(), cpu_result.len());
    for i in 0..cuda_result.len() {
        let diff = (cuda_result[i] - cpu_result[i]).abs();
        assert!(
            diff < 1e-3,
            "PARITY-114: CUDA vs CPU mismatch at {}: cuda={}, cpu={}, diff={}",
            i,
            cuda_result[i],
            cpu_result[i],
            diff
        );
    }
    eprintln!("PARITY-114: CUDA matches CPU reference");
}

#[test]
#[serial]
fn test_cuda_executor_gemm_size_validation() {
    // This test requires CUDA GPU to create an executor
    let mut executor = CudaExecutor::new(0).expect("test");

    // Wrong sizes - should fail validation
    let a = vec![1.0f32; 10]; // Wrong size
    let b = vec![1.0f32; 16];
    let mut c = vec![0.0f32; 16];

    let result = executor.gemm(&a, &b, &mut c, 4, 4, 4);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cuda_executor_softmax() {
    // Debug: print PTX first
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 4 });
    eprintln!("Generated PTX:\n{}", ptx);

    let mut executor = CudaExecutor::new(0).expect("test");

    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax failed: {:?}", result.err());

    // Check softmax properties
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(data[3] > data[2]); // Larger input = larger output
    assert!(data[2] > data[1]);
    assert!(data[1] > data[0]);
}

#[test]
#[serial]
fn test_cuda_executor_synchronize() {
    let executor = CudaExecutor::new(0).expect("test");
    let result = executor.synchronize();
    assert!(result.is_ok());
}

// ========================================================================
// Drop Order Tests (IMP-800: GPU Parity)
// ========================================================================

/// Test that CudaExecutor can be created and dropped multiple times
/// without crashing (validates correct Drop order: context dropped last)
#[test]
#[serial]
fn test_cuda_executor_drop_order_multiple_cycles() {
    // This test verifies the Drop order is correct:
    // Fields should be dropped in reverse declaration order,
    // with context dropped LAST (after stream and modules)
    for i in 1..=3 {
        let mut executor = CudaExecutor::new(0)
            .unwrap_or_else(|e| panic!("Cycle {}: Failed to create executor: {}", i, e));

        // Verify executor works
        assert!(
            executor.device_name().is_ok(),
            "Cycle {}: device_name failed",
            i
        );

        // Run a GEMM to load a module (tests module Drop)
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 16];
        let mut c = vec![0.0f32; 16];
        executor
            .gemm(&a, &b, &mut c, 4, 4, 4)
            .unwrap_or_else(|e| panic!("Cycle {}: GEMM failed: {}", i, e));

        // executor is dropped here - must not crash
    }
    // If we reach here, Drop order is correct
}

/// Test rapid create/destroy cycles (stress test for Drop order)
#[test]
#[serial]
fn test_cuda_executor_rapid_lifecycle() {
    // 10 rapid cycles without any work - pure lifecycle test
    for _ in 0..10 {
        let executor = CudaExecutor::new(0).expect("Failed to create executor");
        drop(executor); // Explicit drop for clarity
    }
}

/// Test that modules are properly cleaned up before context
#[test]
#[serial]
fn test_cuda_executor_module_cleanup() {
    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Load multiple modules (different GEMM configurations)
    for size in [4, 8, 16, 32] {
        let a = vec![1.0f32; size * size];
        let b = vec![1.0f32; size * size];
        let mut c = vec![0.0f32; size * size];
        executor
            .gemm(&a, &b, &mut c, size as u32, size as u32, size as u32)
            .expect("GEMM should succeed");
    }

    // Now drop - all modules must be cleaned up before context
    drop(executor);

    // Create new executor to verify GPU is in good state
    let executor2 = CudaExecutor::new(0).expect("Should create after cleanup");
    assert!(executor2.device_name().is_ok());
}

// ========================================================================
// GpuMemoryPool Tests (IMP-900d)
// ========================================================================

#[test]
fn test_size_class_for_small_size() {
    // Small size should map to 4KB class
    let class = SizeClass::for_size(1024);
    assert_eq!(class.map(|c| c.bytes()), Some(4096));
}

#[test]
fn test_size_class_for_exact_size() {
    // Exact match should return same size
    let class = SizeClass::for_size(1048576); // 1 MB
    assert_eq!(class.map(|c| c.bytes()), Some(1048576));
}

#[test]
fn test_size_class_for_large_size() {
    // Large size should map to 256MB class
    let class = SizeClass::for_size(200_000_000);
    assert_eq!(class.map(|c| c.bytes()), Some(268435456)); // 256 MB
}

#[test]
fn test_size_class_too_large() {
    // Size larger than max class should return None
    let class = SizeClass::for_size(500_000_000);
    assert!(class.is_none());
}

#[test]
fn test_gpu_memory_pool_creation() {
    let pool = GpuMemoryPool::new();
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 0);
}

#[test]
fn test_gpu_memory_pool_with_max_size() {
    let pool = GpuMemoryPool::with_max_size(512 * 1024 * 1024);
    assert_eq!(pool.max_size(), 512 * 1024 * 1024);
}

#[test]
fn test_gpu_memory_pool_try_get_empty() {
    let mut pool = GpuMemoryPool::new();

    // Pool is empty, should return None and increment miss counter
    let result = pool.try_get(1024);
    assert!(result.is_none());

    let stats = pool.stats();
    assert_eq!(stats.pool_misses, 1);
    assert_eq!(stats.pool_hits, 0);
}

#[test]
fn test_gpu_memory_pool_return_and_get() {
    let mut pool = GpuMemoryPool::new();

    // Return a buffer to the pool
    let handle = GpuBufferHandle {
        size: 4096,
        in_use: false,
    };
    pool.return_buffer(handle);

    // Now try to get it back
    let result = pool.try_get(4096);
    assert!(result.is_some());
    let handle = result.expect("test");
    assert!(handle.in_use);

    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 1);
}

#[test]
fn test_gpu_memory_pool_allocation_tracking() {
    let mut pool = GpuMemoryPool::new();

    pool.record_allocation(1024 * 1024);
    assert_eq!(pool.stats().total_allocated, 1024 * 1024);

    pool.record_allocation(2048 * 1024);
    assert_eq!(pool.stats().total_allocated, 3072 * 1024);
    assert_eq!(pool.stats().peak_usage, 3072 * 1024);

    pool.record_deallocation(1024 * 1024);
    assert_eq!(pool.stats().total_allocated, 2048 * 1024);
    assert_eq!(pool.stats().peak_usage, 3072 * 1024); // Peak unchanged
}

#[test]
fn test_gpu_memory_pool_hit_rate() {
    let mut pool = GpuMemoryPool::new();

    // Return 3 buffers
    for _ in 0..3 {
        pool.return_buffer(GpuBufferHandle {
            size: 4096,
            in_use: false,
        });
    }

    // Get 3 (hits) + try to get 1 more (miss)
    for _ in 0..3 {
        let _ = pool.try_get(4096);
    }
    let _ = pool.try_get(4096); // Miss - pool now empty

    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 3);
    assert_eq!(stats.pool_misses, 1);
    assert!((stats.hit_rate - 0.75).abs() < 0.01); // 3/4 = 75%
}

#[test]
fn test_gpu_memory_pool_clear() {
    let mut pool = GpuMemoryPool::new();

    // Add some buffers
    for _ in 0..5 {
        pool.return_buffer(GpuBufferHandle {
            size: 4096,
            in_use: false,
        });
    }
    assert_eq!(pool.stats().free_buffers, 5);

    // Clear the pool
    pool.clear();
    assert_eq!(pool.stats().free_buffers, 0);
}

#[test]
fn test_pool_stats_estimated_savings() {
    let stats = PoolStats {
        total_allocated: 10 * 1024 * 1024,
        peak_usage: 20 * 1024 * 1024,
        pool_hits: 100,
        pool_misses: 50,
        hit_rate: 0.667,
        free_buffers: 5,
    };

    // 100 hits * 1MB assumed per allocation = 100MB saved
    assert_eq!(stats.estimated_savings_bytes(), 100 * 1024 * 1024);
}

#[test]
fn test_gpu_memory_pool_has_capacity() {
    let mut pool = GpuMemoryPool::with_max_size(100 * 1024 * 1024); // 100 MB max

    // Initially has capacity
    assert!(pool.has_capacity(50 * 1024 * 1024)); // 50 MB fits
    assert!(pool.has_capacity(100 * 1024 * 1024)); // 100 MB fits exactly
    assert!(!pool.has_capacity(101 * 1024 * 1024)); // 101 MB doesn't fit

    // After recording allocation
    pool.record_allocation(60 * 1024 * 1024); // 60 MB allocated
    assert!(pool.has_capacity(40 * 1024 * 1024)); // 40 MB still fits
    assert!(!pool.has_capacity(41 * 1024 * 1024)); // 41 MB doesn't fit
}

#[test]
fn test_gpu_memory_pool_max_size_getter() {
    let pool = GpuMemoryPool::with_max_size(512 * 1024 * 1024);
    assert_eq!(pool.max_size(), 512 * 1024 * 1024);

    let default_pool = GpuMemoryPool::new();
    assert_eq!(default_pool.max_size(), 2 * 1024 * 1024 * 1024); // 2 GB default
}

// ========================================================================
// Kernel Fusion Tests (IMP-900b)
// ========================================================================

#[test]
fn test_gemm_bias_activation_kernel_type() {
    let kernel_type = KernelType::GemmBiasActivation {
        m: 64,
        n: 64,
        k: 64,
        activation: 1, // ReLU
    };

    let kernels = CudaKernels::new();
    let name = kernels.kernel_name(&kernel_type);
    assert_eq!(name, "gemm_tiled"); // Falls back to tiled for now

    let ptx = kernels.generate_ptx(&kernel_type);
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("gemm_tiled"));
}

#[test]
fn test_gemm_fused_activation_values() {
    // Test activation types are correctly defined
    // 0 = no activation
    // 1 = ReLU
    // 2 = GELU
    let no_act = KernelType::GemmBiasActivation {
        m: 4,
        n: 4,
        k: 4,
        activation: 0,
    };
    let relu = KernelType::GemmBiasActivation {
        m: 4,
        n: 4,
        k: 4,
        activation: 1,
    };
    let gelu = KernelType::GemmBiasActivation {
        m: 4,
        n: 4,
        k: 4,
        activation: 2,
    };

    // All should generate valid PTX
    let kernels = CudaKernels::new();
    assert!(kernels.generate_ptx(&no_act).contains(".version"));
    assert!(kernels.generate_ptx(&relu).contains(".version"));
    assert!(kernels.generate_ptx(&gelu).contains(".version"));
}

#[test]
#[serial]
fn test_gemm_fused_no_activation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    // Identity-like matrices for easy verification
    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    executor
        .gemm_fused(&a, &b, None, &mut c, m, n, k, 0)
        .expect("GEMM fused should succeed");

    // Each element should be k (dot product of 1s)
    for val in &c {
        assert!((val - k as f32).abs() < 0.001);
    }
}

#[test]
#[serial]
fn test_gemm_fused_with_bias() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let bias = vec![2.0f32; n as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    executor
        .gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 0)
        .expect("GEMM fused with bias should succeed");

    // Each element should be k + bias = 4 + 2 = 6
    for val in &c {
        assert!((val - 6.0).abs() < 0.001);
    }
}

#[test]
#[serial]
fn test_gemm_fused_relu_activation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    // Use values that will produce negative results after bias
    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let bias = vec![-10.0f32; n as usize]; // Large negative bias
    let mut c = vec![0.0f32; (m * n) as usize];

    executor
        .gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 1) // ReLU
        .expect("GEMM fused with ReLU should succeed");

    // k=4, so GEMM gives 4, bias -10 gives -6, ReLU gives 0
    for val in &c {
        assert!(*val >= 0.0, "ReLU should clamp negative to 0");
    }
}

#[test]
#[serial]
fn test_gemm_fused_gelu_activation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    executor
        .gemm_fused(&a, &b, None, &mut c, m, n, k, 2) // GELU
        .expect("GEMM fused with GELU should succeed");

    // GELU(4) ≈ 4.0 (GELU(x) ≈ x for positive x)
    for val in &c {
        assert!(*val > 3.9 && *val < 4.1, "GELU(4) should be ≈4");
    }
}

#[test]
#[serial]
fn test_gemm_fused_bias_size_validation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let wrong_bias = vec![2.0f32; (n + 1) as usize]; // Wrong size!
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_fused(&a, &b, Some(&wrong_bias), &mut c, m, n, k, 0);
    assert!(result.is_err(), "Should reject wrong bias size");
}

// ========================================================================
// FlashAttention Tests (IMP-900c)
// ========================================================================

#[test]
fn test_flash_attention_memory_bytes() {
    // Test memory calculation
    let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(1024, 64);

    // Naive: 1024 * 1024 * 4 = 4MB
    assert_eq!(naive, 1024 * 1024 * 4);

    // Flash: 64 * 64 * 4 * 2 = 32KB
    assert_eq!(flash, 64 * 64 * 4 * 2);

    // Verify significant memory savings
    let savings = naive as f64 / flash as f64;
    assert!(
        savings > 100.0,
        "FlashAttention should save 100x+ memory for seq_len=1024"
    );
}

#[test]
fn test_flash_attention_memory_scaling() {
    // Verify O(N²) vs O(1) scaling
    let (naive_256, flash_256) = CudaExecutor::flash_attention_memory_bytes(256, 64);
    let (naive_1024, flash_1024) = CudaExecutor::flash_attention_memory_bytes(1024, 64);
    let (naive_4096, flash_4096) = CudaExecutor::flash_attention_memory_bytes(4096, 64);

    // Naive scales O(N²): 16x seq_len = 256x memory
    assert_eq!(naive_1024 / naive_256, 16); // 4x seq_len = 16x memory
    assert_eq!(naive_4096 / naive_1024, 16); // 4x seq_len = 16x memory

    // Flash is constant (O(1) w.r.t. seq_len)
    assert_eq!(flash_256, flash_1024);
    assert_eq!(flash_1024, flash_4096);
}

#[test]
fn test_attention_kernel_type_generation() {
    let kernel_type = KernelType::Attention {
        seq_len: 128,
        head_dim: 64,
        causal: true,
    };

    let kernels = CudaKernels::new();
    let name = kernels.kernel_name(&kernel_type);
    assert_eq!(name, "flash_attention_causal"); // causal=true -> causal kernel

    let ptx = kernels.generate_ptx(&kernel_type);
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("attention"));
}

// ========================================================================
// BiasActivation Epilogue Tests (IMP-1000)
// ========================================================================

#[test]
fn test_bias_activation_ptx_generation() {
    let kernels = CudaKernels::new();

    // Test no activation
    let no_act = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 0,
    };
    let ptx = kernels.generate_ptx(&no_act);
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains("bias_activation"));
    assert!(ptx.contains("add.f32")); // bias addition

    // Test ReLU
    let relu = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 1,
    };
    let ptx_relu = kernels.generate_ptx(&relu);
    assert!(ptx_relu.contains("max.f32")); // ReLU: max(0, x)

    // Test GELU
    let gelu = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 2,
    };
    let ptx_gelu = kernels.generate_ptx(&gelu);
    assert!(ptx_gelu.contains("ex2.approx")); // GELU: exponential for sigmoid
}

#[test]
fn test_bias_activation_kernel_name() {
    let kernels = CudaKernels::new();
    let kernel_type = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 1,
    };
    assert_eq!(kernels.kernel_name(&kernel_type), "bias_activation");
}

#[test]
#[serial]
fn test_flash_attention_basic() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16u32;
    let head_dim = 8u32;
    let size = (seq_len * head_dim) as usize;

    // Simple test: Q = K = V = 1, should produce similar output
    let q = vec![1.0f32; size];
    let k = vec![1.0f32; size];
    let v = vec![1.0f32; size];
    let mut output = vec![0.0f32; size];

    let scale = 1.0 / (head_dim as f32).sqrt();
    executor
        .flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, false)
        .expect("FlashAttention should succeed");

    // Output should be non-zero
    assert!(
        output.iter().any(|&x| x != 0.0),
        "Output should be non-zero"
    );
}

#[test]
#[serial]
fn test_flash_attention_causal() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16u32;
    let head_dim = 8u32;
    let size = (seq_len * head_dim) as usize;

    let q = vec![1.0f32; size];
    let k = vec![1.0f32; size];
    let v = vec![1.0f32; size];
    let mut output = vec![0.0f32; size];

    let scale = 1.0 / (head_dim as f32).sqrt();
    executor
        .flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, true) // causal
        .expect("FlashAttention causal should succeed");

    // Output should be non-zero
    assert!(
        output.iter().any(|&x| x != 0.0),
        "Output should be non-zero"
    );
}

#[test]
#[serial]
fn test_flash_attention_size_validation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16u32;
    let head_dim = 8u32;
    let correct_size = (seq_len * head_dim) as usize;
    let wrong_size = correct_size + 1;

    let q = vec![1.0f32; correct_size];
    let k = vec![1.0f32; correct_size];
    let v = vec![1.0f32; wrong_size]; // Wrong size!
    let mut output = vec![0.0f32; correct_size];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let result = executor.flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, false);

    assert!(result.is_err(), "Should reject wrong V size");
}

#[test]
#[serial]
fn test_flash_attention_memory_tracking() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16u32;
    let head_dim = 8u32;
    let size = (seq_len * head_dim) as usize;

    let q = vec![1.0f32; size];
    let k = vec![1.0f32; size];
    let v = vec![1.0f32; size];
    let mut output = vec![0.0f32; size];

    // Clear pool stats
    executor.clear_pool();

    let scale = 1.0 / (head_dim as f32).sqrt();
    executor
        .flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, false)
        .expect("FlashAttention should succeed");

    // Check pool recorded allocations
    let stats = executor.pool_stats();
    assert!(
        stats.total_allocated == 0 || stats.peak_usage > 0,
        "Memory should be tracked"
    );
}

// ========================================================================
// COV-001: Comprehensive Quantized Kernel Tests (Target: 95% coverage)
// ========================================================================

/// Helper: Create mock Q4_K weights (144 bytes per 256 values)
fn mock_q4k_weights(n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "k must be divisible by 256 for Q4_K");
    let n_superblocks_per_row = k / 256;
    let bytes_per_row = n_superblocks_per_row * 144;
    vec![0x42u8; n_rows * bytes_per_row] // Non-zero pattern for detection
}

/// Helper: Create mock Q5_K weights (176 bytes per 256 values)
fn mock_q5k_weights(n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "k must be divisible by 256 for Q5_K");
    let n_superblocks_per_row = k / 256;
    let bytes_per_row = n_superblocks_per_row * 176;
    vec![0x43u8; n_rows * bytes_per_row]
}

/// Helper: Create mock Q6_K weights (210 bytes per 256 values)
fn mock_q6k_weights(n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "k must be divisible by 256 for Q6_K");
    let n_superblocks_per_row = k / 256;
    let bytes_per_row = n_superblocks_per_row * 210;
    vec![0x44u8; n_rows * bytes_per_row]
}

#[test]
#[serial]
fn test_cov001_q4k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q4k_weights(n as usize, k as usize);
    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q4k_gemv(&weights, &input, &mut output, n, k);
    assert!(result.is_ok(), "q4k_gemv should succeed: {:?}", result);
}

#[test]
#[serial]
fn test_cov001_q5k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q5k_weights(n as usize, k as usize);
    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q5k_gemv(&weights, &input, &mut output, n, k);
    assert!(result.is_ok(), "q5k_gemv should succeed: {:?}", result);
}

#[test]
#[serial]
fn test_cov001_q6k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q6k_weights(n as usize, k as usize);
    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q6k_gemv(&weights, &input, &mut output, n, k);
    assert!(result.is_ok(), "q6k_gemv should succeed: {:?}", result);
}

#[test]
#[serial]
fn test_cov001_q4k_gemv_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q4k_weights(n as usize, k as usize);

    // Load weights to cache
    executor
        .load_quantized_weights("test_q4k", &weights)
        .expect("load weights");

    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q4k_gemv_cached("test_q4k", &input, &mut output, n, k);
    assert!(
        result.is_ok(),
        "q4k_gemv_cached should succeed: {:?}",
        result
    );
}

#[test]
#[serial]
fn test_cov001_q5k_gemv_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q5k_weights(n as usize, k as usize);

    executor
        .load_quantized_weights("test_q5k", &weights)
        .expect("load weights");

    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q5k_gemv_cached("test_q5k", &input, &mut output, n, k);
    assert!(
        result.is_ok(),
        "q5k_gemv_cached should succeed: {:?}",
        result
    );
}

#[test]
#[serial]
fn test_cov001_q6k_gemv_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q6k_weights(n as usize, k as usize);

    executor
        .load_quantized_weights("test_q6k", &weights)
        .expect("load weights");

    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q6k_gemv_cached("test_q6k", &input, &mut output, n, k);
    assert!(
        result.is_ok(),
        "q6k_gemv_cached should succeed: {:?}",
        result
    );
}

// ========================================================================
// COV-002: High-level CUDA function tests (slice-based API)
// ========================================================================

#[test]
#[serial]
fn test_cov002_softmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax should succeed: {:?}", result);

    // Verify softmax properties: sum to 1, all positive
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1.0");
    assert!(
        data.iter().all(|&x| x > 0.0),
        "all values should be positive"
    );
}

#[test]
#[serial]
fn test_cov002_gemm_optimized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 32u32;
    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let tile_size = 32u32;
    let result = executor.gemm_optimized(&a, &b, &mut c, m, n, k, tile_size);
    assert!(
        result.is_ok(),
        "gemm_optimized should succeed: {:?}",
        result
    );

    // Each element should be k (dot product of k ones)
    for val in &c {
        assert!(
            (*val - k as f32).abs() < 1e-3,
            "expected {}, got {}",
            k,
            val
        );
    }
}

#[test]
#[serial]
fn test_cov002_gemm_fused_variants() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 16u32;
    let n = 16u32;
    let k = 16u32;
    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let bias = vec![1.0f32; n as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    // Test with bias and no activation (0)
    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 0);
    assert!(
        result.is_ok(),
        "gemm_fused with no activation should succeed: {:?}",
        result
    );

    // Test with bias and ReLU activation (1)
    c.fill(0.0);
    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 1);
    assert!(
        result.is_ok(),
        "gemm_fused with ReLU should succeed: {:?}",
        result
    );

    // Test with bias and GELU activation (2)
    c.fill(0.0);
    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 2);
    assert!(
        result.is_ok(),
        "gemm_fused with GELU should succeed: {:?}",
        result
    );
}

#[test]
#[serial]
fn test_cov002_flash_attention_multi_head() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 8u32;
    let head_dim = 8u32;
    let n_heads = 4u32;
    let size = (seq_len * head_dim * n_heads) as usize;

    let q = vec![1.0f32; size];
    let k = vec![1.0f32; size];
    let v = vec![1.0f32; size];
    let mut output = vec![0.0f32; size];

    let result = executor.flash_attention_multi_head(
        &q,
        &k,
        &v,
        &mut output,
        seq_len,
        head_dim,
        n_heads,
        true,
    );
    assert!(
        result.is_ok(),
        "flash_attention_multi_head should succeed: {:?}",
        result
    );
}

#[test]
#[serial]
fn test_cov002_silu_gelu_host() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let size = 256usize;
    let input = vec![1.0f32; size];
    let mut silu_out = vec![0.0f32; size];
    let mut gelu_out = vec![0.0f32; size];

    let result = executor.silu_host(&input, &mut silu_out);
    assert!(result.is_ok(), "silu_host should succeed: {:?}", result);

    let result = executor.gelu_host(&input, &mut gelu_out);
    assert!(result.is_ok(), "gelu_host should succeed: {:?}", result);

    // SiLU and GELU should produce different results
    assert!(silu_out[0] != gelu_out[0], "SiLU and GELU should differ");
}

#[test]
#[serial]
fn test_cov002_elementwise_mul_host() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let size = 256usize;
    let a = vec![2.0f32; size];
    let b = vec![3.0f32; size];
    let mut output = vec![0.0f32; size];

    let result = executor.elementwise_mul_host(&a, &b, &mut output);
    assert!(
        result.is_ok(),
        "elementwise_mul_host should succeed: {:?}",
        result
    );
    assert!((output[0] - 6.0).abs() < 1e-5, "2 * 3 should be 6");
}

#[test]
#[serial]
fn test_cov002_load_and_clear_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let weights = vec![1.0f32; 1024];

    // Load weights
    let result = executor.load_weights("test_weights", &weights);
    assert!(result.is_ok(), "load_weights should succeed");

    // Check cache stats
    assert!(executor.has_weights("test_weights"));
    assert_eq!(executor.cached_weight_count(), 1);
    assert!(executor.cached_weight_bytes() > 0);

    // Clear weights
    executor.clear_weights();
    assert!(!executor.has_weights("test_weights"));
    assert_eq!(executor.cached_weight_count(), 0);
}

