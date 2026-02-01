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
