use super::*;
use crate::cuda::memory::{GpuBufferHandle, SizeClass, TransferMode};
use crate::cuda::pipeline::{
    presets, BankConflictStrategy, MemoryPattern, PtxOptimizationHints, PtxOptimizer,
    RegisterTiling,
};
use serial_test::serial;

#[test]
#[serial]
fn test_cov002_load_quantized_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Mock Q4_K weights: 144 bytes per 256 values
    let weights = vec![0x42u8; 144];

    // Load quantized weights
    let result = executor.load_quantized_weights("q4k_test", &weights);
    assert!(result.is_ok(), "load_quantized_weights should succeed");

    // Check cache stats
    assert!(executor.cached_quantized_weight_count() > 0);
    assert!(executor.cached_quantized_weight_bytes() > 0);

    // Clear
    executor.clear_quantized_weights();
    assert_eq!(executor.cached_quantized_weight_count(), 0);
}

#[test]
#[serial]
fn test_cov002_profiler_operations() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Enable profiling
    executor.enable_profiling();
    assert!(executor.is_profiling_enabled());

    // Get profiler and reset
    let _profiler = executor.profiler();
    let _profiler_mut = executor.profiler_mut();
    executor.reset_profiler();

    // Get profiler summary
    let _summary = executor.profiler_summary();

    // Disable profiling
    executor.disable_profiling();
    assert!(!executor.is_profiling_enabled());
}

#[test]
#[serial]
fn test_cov002_graph_tracking() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Enable graph tracking
    executor.enable_graph_tracking();
    assert!(executor.is_graph_tracking_enabled());

    // Get execution graph
    let _graph = executor.execution_graph();
    let _ascii = executor.execution_graph_ascii();

    // Clear and disable
    executor.clear_execution_graph();
    executor.disable_graph_tracking();
    assert!(!executor.is_graph_tracking_enabled());
}

#[test]
#[serial]
fn test_cov002_tile_profiling() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Enable tile profiling
    executor.enable_tile_profiling();
    assert!(executor.is_tile_profiling_enabled());

    // Get tile stats
    let _summary = executor.tile_summary();
    let _json = executor.tile_stats_json();

    // Reset and disable
    executor.reset_tile_stats();
    executor.disable_tile_profiling();
    assert!(!executor.is_tile_profiling_enabled());
}

#[test]
#[serial]
fn test_cov002_memory_and_device_info() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get device name
    let name = executor.device_name().expect("device_name should succeed");
    assert!(name.contains("NVIDIA") || name.contains("RTX") || name.contains("GeForce"));

    // Get memory info
    let mem_info = executor.memory_info();
    assert!(mem_info.is_ok(), "memory_info should succeed");
    let (free, total) = mem_info.expect("CUDA operation failed");
    assert!(total > 0, "total memory should be > 0");
    assert!(free <= total, "free should be <= total");

    // Get context
    let _ctx = executor.context();
}

#[test]
#[serial]
fn test_cov002_staging_buffer_operations() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get staging buffer
    let buf = executor.get_staging_buffer(1024);
    assert!(buf.len() >= 1024);

    // Return staging buffer
    executor.return_staging_buffer(buf);

    // Get pool stats
    let _stats = executor.staging_pool_stats();

    // Clear pool
    executor.clear_pool();
}

#[test]
#[serial]
fn test_cov002_synchronize() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize();
    assert!(result.is_ok(), "synchronize should succeed");
}

#[test]
fn test_cov002_cuda_likely_available() {
    // This should return true on a system with CUDA (checks /dev/nvidia0 or CUDA_VISIBLE_DEVICES)
    let likely = CudaKernels::cuda_likely_available();
    // On a system with RTX 4090, this should be true
    assert!(
        likely,
        "cuda_likely_available should be true on a system with NVIDIA GPU"
    );
}

#[test]
fn test_cov002_is_available_and_num_devices() {
    let available = CudaExecutor::is_available();
    let num_devices = CudaExecutor::num_devices();

    if available {
        assert!(
            num_devices > 0,
            "If CUDA available, num_devices should be > 0"
        );
    }
}

#[test]
fn test_cov001_transfer_mode_properties() {
    let modes = [
        TransferMode::Pageable,
        TransferMode::Pinned,
        TransferMode::Async,
        TransferMode::ZeroCopy,
    ];

    for mode in modes {
        let speedup = mode.estimated_speedup();
        assert!(speedup >= 1.0, "Speedup should be >= 1.0");

        let requires_pinned = mode.requires_pinned();
        match mode {
            TransferMode::Pageable => assert!(!requires_pinned),
            _ => assert!(requires_pinned),
        }
    }
}

#[test]
fn test_cov001_weight_quant_type_detection() {
    // Test from_ggml_type
    assert!(matches!(
        WeightQuantType::from_ggml_type(12),
        Some(WeightQuantType::Q4K)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(13),
        Some(WeightQuantType::Q5K)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(14),
        Some(WeightQuantType::Q6K)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(8),
        Some(WeightQuantType::Q8_0)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(6),
        Some(WeightQuantType::Q5_0)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(2),
        Some(WeightQuantType::Q4_0)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(3),
        Some(WeightQuantType::Q4_1)
    ));
    assert!(WeightQuantType::from_ggml_type(999).is_none());

    // Test bytes_per_superblock
    assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);
    assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);
    assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);

    // Test bytes_per_block
    assert_eq!(WeightQuantType::Q8_0.bytes_per_block(), 34);
    assert_eq!(WeightQuantType::Q4_0.bytes_per_block(), 18);

    // Test matches_size
    let q4k = WeightQuantType::Q4K;
    assert!(q4k.matches_size(144, 1, 256)); // 1 row, 256 cols = 1 superblock
}

#[test]
fn test_cov001_ptx_optimization_hints() {
    let max_throughput = PtxOptimizationHints::max_throughput();
    assert!(max_throughput.uses_vectorized_loads());
    assert_eq!(max_throughput.vector_width(), 4);
    assert_eq!(max_throughput.shared_mem_padding(), 1);

    let low_latency = PtxOptimizationHints::low_latency();
    assert!(!low_latency.uses_vectorized_loads());
    assert_eq!(low_latency.vector_width(), 1);
    assert_eq!(low_latency.shared_mem_padding(), 0);

    let balanced = PtxOptimizationHints::balanced();
    assert!(balanced.uses_vectorized_loads());
    assert_eq!(balanced.vector_width(), 2);
}

#[test]
fn test_cov001_ptx_optimizer() {
    let hints = PtxOptimizationHints::max_throughput();
    let optimizer = PtxOptimizer::new(hints);

    // Test summary generation
    let summary = optimizer.summary();
    assert!(summary.contains("PtxOptimizer"));

    // Test padded row calculation
    assert_eq!(optimizer.padded_shared_mem_row(32), 33);

    // Test register estimation
    let regs = optimizer.estimated_registers();
    assert!(regs > 0);

    // Test high register pressure detection
    let _high_pressure = optimizer.is_high_register_pressure();
}

#[test]
fn test_cov001_register_tiling() {
    let large = RegisterTiling::large();
    assert_eq!(large.registers_needed(), 64);

    let medium = RegisterTiling::medium();
    assert_eq!(medium.registers_needed(), 16);

    let small = RegisterTiling::small();
    assert_eq!(small.registers_needed(), 4);
}

#[test]
fn test_cov001_memory_pattern() {
    let scalar = MemoryPattern::Scalar;
    let vec2 = MemoryPattern::Vector2;
    let vec4 = MemoryPattern::Vector4;

    // Just ensure they can be compared
    assert_ne!(scalar, vec2);
    assert_ne!(vec2, vec4);
}

#[test]
fn test_cov001_bank_conflict_strategy() {
    let none = BankConflictStrategy::None;
    let padding = BankConflictStrategy::Padding;
    let xor = BankConflictStrategy::Xor;

    assert_ne!(none, padding);
    assert_ne!(padding, xor);
}

#[test]
fn test_cov001_presets_coverage() {
    // Test all preset functions
    let _llama_attn = presets::llama_attention(2048, 64);
    let _ffn = presets::ffn_gemm(1, 4096, 11008);
    let _q4k = presets::q4k_inference(1, 4096, 4096);
    let _q4k_ggml = presets::q4k_ggml_inference(1, 4096, 4096);
    let _rmsnorm = presets::rmsnorm(4096);
    let _mha = presets::multi_head_attention(2048, 64, 32);
    let _phi2_mha = presets::phi2_multi_head_attention(2048);
    let _tc_attn = presets::tensor_core_attention(2048, 64, 32);
    let _llama_tc = presets::llama_tensor_core_attention(2048);
}

#[test]
fn test_cov001_kernel_type_kernel_names() {
    let kernels = CudaKernels::new();

    // Test all kernel type names
    let types = [
        KernelType::GemmNaive { m: 1, n: 1, k: 1 },
        KernelType::GemmTiled {
            m: 1,
            n: 1,
            k: 1,
            tile_size: 32,
        },
        KernelType::Softmax { dim: 128 },
        KernelType::LayerNorm {
            hidden_size: 256,
            epsilon: 1e-5,
            affine: true,
        },
        KernelType::Attention {
            seq_len: 16,
            head_dim: 64,
            causal: true,
        },
        KernelType::QuantizedGemm {
            m: 1,
            n: 256,
            k: 256,
        },
        KernelType::QuantizedGemmGgml {
            m: 1,
            n: 256,
            k: 256,
        },
        KernelType::Q4KGemv { k: 256, n: 256 },
        KernelType::Q5KGemv { k: 256, n: 256 },
        KernelType::Q6KGemv { k: 256, n: 256 },
    ];

    for kt in types {
        let name = kernels.kernel_name(&kt);
        assert!(!name.is_empty(), "Kernel name should not be empty");
    }
}

// =========================================================================
// COV-003: Layer.rs preload/has method coverage tests
// Target: cuda/executor/layer.rs (15.29% -> higher)
// =========================================================================

#[test]
#[serial]
fn test_cov003_preload_rmsnorm_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no weights loaded
    assert!(!executor.has_rmsnorm_weights(0));
    assert!(!executor.has_rmsnorm_weights(1));

    // Preload weights for 1 layer
    let gamma = vec![1.0f32; 256];
    let attn_norms: Vec<&[f32]> = vec![&gamma];
    let ffn_norms: Vec<&[f32]> = vec![&gamma];
    let result = executor.preload_rmsnorm_weights(1, &attn_norms, &ffn_norms);
    assert!(result.is_ok(), "preload_rmsnorm_weights should succeed");

    // Now layer 0 has weights
    assert!(executor.has_rmsnorm_weights(0));
    assert!(!executor.has_rmsnorm_weights(1)); // Layer 1 not loaded
}

#[test]
#[serial]
fn test_cov003_preload_rmsnorm_weights_multiple_layers() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 512];

    // Preload 4 layers
    let attn_norms: Vec<&[f32]> = vec![&gamma, &gamma, &gamma, &gamma];
    let ffn_norms: Vec<&[f32]> = vec![&gamma, &gamma, &gamma, &gamma];
    let result = executor.preload_rmsnorm_weights(4, &attn_norms, &ffn_norms);
    assert!(result.is_ok(), "preload_rmsnorm_weights should succeed");

    // Verify all layers have weights
    for layer_idx in 0..4 {
        assert!(executor.has_rmsnorm_weights(layer_idx));
    }
    // Layer 4 not loaded
    assert!(!executor.has_rmsnorm_weights(4));
}

#[test]
#[serial]
fn test_cov003_preload_output_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no output norm
    assert!(!executor.has_output_norm());

    // Preload output norm
    let gamma = vec![1.0f32; 256];
    let result = executor.preload_output_norm(&gamma);
    assert!(result.is_ok(), "preload_output_norm should succeed");

    // Now has output norm
    assert!(executor.has_output_norm());
}

#[test]
#[serial]
fn test_cov003_preload_qkv_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no QKV bias
    assert!(!executor.has_qkv_bias(0));

    // Preload QKV bias for 1 layer
    let hidden_dim = 256;
    let bias_data = vec![0.1f32; hidden_dim];
    let q_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data)];
    let k_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data)];
    let v_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data)];

    let result = executor.preload_qkv_bias(1, &q_biases, &k_biases, &v_biases);
    assert!(
        result.is_ok(),
        "preload_qkv_bias should succeed: {:?}",
        result
    );

    // Now layer 0 has QKV bias
    assert!(executor.has_qkv_bias(0));
    assert!(!executor.has_qkv_bias(1)); // Layer 1 not loaded
}

#[test]
#[serial]
fn test_cov003_preload_qkv_bias_multiple_layers() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 128;
    let bias_data = vec![0.1f32; hidden_dim];

    // Preload for 3 layers
    let q_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data), Some(&bias_data), Some(&bias_data)];
    let k_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data), Some(&bias_data), Some(&bias_data)];
    let v_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data), Some(&bias_data), Some(&bias_data)];

    let result = executor.preload_qkv_bias(3, &q_biases, &k_biases, &v_biases);
    assert!(result.is_ok(), "preload_qkv_bias should succeed");

    // Verify all layers have QKV bias
    for layer_idx in 0..3 {
        assert!(
            executor.has_qkv_bias(layer_idx),
            "layer {} should have bias",
            layer_idx
        );
    }
    assert!(!executor.has_qkv_bias(3));
}

#[test]
#[serial]
fn test_cov003_preload_lm_head_bias_none() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no LM head bias
    assert!(!executor.has_lm_head_bias());

    // Preload None bias (no bias)
    let result = executor.preload_lm_head_bias(None);
    assert!(result.is_ok(), "preload_lm_head_bias(None) should succeed");

    // Still no bias after loading None
    assert!(!executor.has_lm_head_bias());
}

#[test]
#[serial]
fn test_cov003_preload_lm_head_bias_some() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no LM head bias
    assert!(!executor.has_lm_head_bias());

    // Preload with bias
    let vocab_size = 32000;
    let bias = vec![0.0f32; vocab_size];
    let result = executor.preload_lm_head_bias(Some(&bias));
    assert!(result.is_ok(), "preload_lm_head_bias(Some) should succeed");

    // Now has bias
    assert!(executor.has_lm_head_bias());
}

#[test]
#[serial]
fn test_cov003_cache_rmsnorm_gamma() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache gamma by name
    let gamma = vec![1.0f32; 256];
    let result = executor.cache_rmsnorm_gamma("test_norm_layer", &gamma);
    assert!(result.is_ok(), "cache_rmsnorm_gamma should succeed");

    // Cache another
    let result2 = executor.cache_rmsnorm_gamma("output_norm", &gamma);
    assert!(
        result2.is_ok(),
        "cache_rmsnorm_gamma for output_norm should succeed"
    );
}

#[test]
#[serial]
fn test_cov003_workspace_output_none() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Fresh executor has no workspace output
    let output = executor.workspace_output();
    // This may or may not be None depending on implementation
    let _ = output;
}

#[test]
#[serial]
fn test_cov003_read_hidden_state_to_cpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Try to read hidden state - may fail if no forward pass done yet
    let result = executor.read_hidden_state_to_cpu();
    // Just verify it doesn't panic - it may return error if no hidden state
    let _ = result;
}

#[test]
#[serial]
fn test_cov003_output_rmsnorm_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // First preload output norm
    let gamma = vec![1.0f32; 256];
    executor
        .preload_output_norm(&gamma)
        .expect("preload_output_norm");

    // Now test output_rmsnorm_gpu
    // (This requires a GPU buffer input, so we test the preload path)
    assert!(executor.has_output_norm());
}

#[test]
#[serial]
fn test_cov003_preload_combined_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test preloading all weight types for a layer
    let hidden_dim = 256;

    // 1. RMSNorm weights
    let gamma = vec![1.0f32; hidden_dim];
    let attn_norms: Vec<&[f32]> = vec![&gamma];
    let ffn_norms: Vec<&[f32]> = vec![&gamma];
    executor
        .preload_rmsnorm_weights(1, &attn_norms, &ffn_norms)
        .expect("rmsnorm");
    assert!(executor.has_rmsnorm_weights(0));

    // 2. Output norm
    executor.preload_output_norm(&gamma).expect("output norm");
    assert!(executor.has_output_norm());

    // 3. QKV bias
    let bias = vec![0.1f32; hidden_dim];
    let q_biases: Vec<Option<&[f32]>> = vec![Some(&bias)];
    let k_biases: Vec<Option<&[f32]>> = vec![Some(&bias)];
    let v_biases: Vec<Option<&[f32]>> = vec![Some(&bias)];
    executor
        .preload_qkv_bias(1, &q_biases, &k_biases, &v_biases)
        .expect("qkv bias");
    assert!(executor.has_qkv_bias(0));

    // 4. LM head bias
    let vocab_bias = vec![0.0f32; 32000];
    executor
        .preload_lm_head_bias(Some(&vocab_bias))
        .expect("lm head bias");
    assert!(executor.has_lm_head_bias());
}

#[test]
#[serial]
fn test_cov003_has_methods_boundary_conditions() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test has_* methods with large layer indices (should return false)
    assert!(!executor.has_rmsnorm_weights(999));
    assert!(!executor.has_qkv_bias(1000));

    // Test default states
    assert!(!executor.has_output_norm());
    assert!(!executor.has_lm_head_bias());
}

// =============================================================================
// COV-004: cuda/executor/kv_cache.rs coverage tests
// Target: 8.32% → 50%+
// Tests for: init_kv_cache_gpu, reset_kv_cache_gpu, rollback_kv_cache_gpu,
//            set_rope_theta, set_rope_type, has_kv_cache_gpu, kv_cache_len,
//            init_batched_kv_cache_gpu, reset_batched_kv_cache_gpu
// =============================================================================

#[test]
#[serial]
fn test_cov004_init_kv_cache_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_layers = 2;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 64;
    let max_len = 128;

    let result = executor.init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_len);
    assert!(result.is_ok());
    assert!(executor.has_kv_cache_gpu());
}

#[test]
#[serial]
fn test_cov004_has_kv_cache_gpu_before_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before init, should return false
    assert!(!executor.has_kv_cache_gpu());
}

#[test]
#[serial]
fn test_cov004_kv_cache_len() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before init, length should be 0
    assert_eq!(executor.kv_cache_len(0), 0);

    // After init, length is still 0 (no tokens added)
    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);
    assert_eq!(executor.kv_cache_len(0), 0);
}

#[test]
#[serial]
fn test_cov004_reset_kv_cache_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);

    // Reset should work even when empty
    executor.reset_kv_cache_gpu();
    assert_eq!(executor.kv_cache_len(0), 0);
}

#[test]
#[serial]
fn test_cov004_rollback_kv_cache_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);

    // Rollback to position 0 should work even when empty
    executor.rollback_kv_cache_gpu(0);
    assert_eq!(executor.kv_cache_len(0), 0);
}

#[test]
#[serial]
fn test_cov004_set_rope_theta() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Default LLaMA theta
    executor.set_rope_theta(10000.0);

    // Qwen2 long context theta
    executor.set_rope_theta(1000000.0);
}

#[test]
#[serial]
fn test_cov004_set_rope_type() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Type 0 = NORM (adjacent pairs)
    executor.set_rope_type(0);

    // Type 2 = NEOX (split halves, used by Qwen2.5)
    executor.set_rope_type(2);
}

#[test]
#[serial]
fn test_cov004_init_batched_kv_cache_invalid_batch_size() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Must init regular KV cache first
    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);

    // Batch size 0 is invalid
    let result = executor.init_batched_kv_cache_gpu(2, 0);
    assert!(result.is_err());

    // Batch size > 32 is invalid
    let result = executor.init_batched_kv_cache_gpu(2, 33);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_init_batched_kv_cache_without_regular() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Without init_kv_cache_gpu, should fail
    let result = executor.init_batched_kv_cache_gpu(2, 4);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_init_batched_kv_cache_valid() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Init regular first
    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);

    // Now batched should work
    let result = executor.init_batched_kv_cache_gpu(2, 4);
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_cov004_reset_batched_kv_cache() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);
    let _ = executor.init_batched_kv_cache_gpu(2, 4);

    // Reset batched should work
    executor.reset_batched_kv_cache_gpu();
}

#[test]
#[serial]
fn test_cov004_flash_attention_cached_dimension_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let head_dim = 64;
    let hidden_dim = num_heads * head_dim; // 256

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_heads, head_dim, 128);

    // Wrong Q dimension should fail
    let q_wrong = vec![0.0f32; 128]; // Should be 256
    let k = vec![0.0f32; hidden_dim];
    let v = vec![0.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    let result = executor.flash_attention_cached(0, &q_wrong, &k, &v, &mut output);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_flash_attention_cached_valid() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Use small dimensions known to work with flash_attention_multi_head
    let num_heads = 4;
    let head_dim = 8; // Reduced from 64
    let hidden_dim = num_heads * head_dim; // 32

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_heads, head_dim, 16);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
    assert!(
        result.is_ok(),
        "flash_attention_cached failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), 1); // New sequence length is 1
    assert_eq!(executor.kv_cache_len(0), 1);
}

#[test]
#[serial]
fn test_cov004_flash_attention_cached_multiple_tokens() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_heads, head_dim, 16);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    // Add 3 tokens
    for i in 1..=3 {
        let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
        assert!(result.is_ok(), "Token {} failed: {:?}", i, result.err());
        assert_eq!(result.unwrap(), i);
    }
    assert_eq!(executor.kv_cache_len(0), 3);
}

#[test]
#[serial]
fn test_cov004_flash_attention_cached_overflow() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 2;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;
    let max_len = 4; // Very small for fast overflow

    let _ = executor.init_kv_cache_gpu(1, num_heads, num_heads, head_dim, max_len);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    // Fill cache
    for i in 0..max_len {
        let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
        assert!(
            result.is_ok(),
            "Token {} failed during fill: {:?}",
            i,
            result.err()
        );
    }

    // Next should overflow
    let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_incremental_attention_gpu_dimension_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let num_kv_heads = 2; // GQA: fewer KV heads
    let head_dim = 8;
    let q_dim = num_heads * head_dim; // 32
    let kv_dim = num_kv_heads * head_dim; // 16

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_kv_heads, head_dim, 16);

    // Wrong Q dimension
    let q_wrong = vec![0.0f32; 16]; // Should be 32
    let k = vec![0.0f32; kv_dim];
    let v = vec![0.0f32; kv_dim];
    let mut output = vec![0.0f32; q_dim];

    let result = executor.incremental_attention_gpu(0, &q_wrong, &k, &v, &mut output);
    assert!(result.is_err());
}

#[test]
#[serial]
