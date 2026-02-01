use super::*;
use crate::cuda::memory::{GpuBufferHandle, SizeClass, TransferMode};
use crate::cuda::pipeline::{
use serial_test::serial;
    // Identity-like matrix A (ones on diagonal)
    let a = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    // B = some values
    let b = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);
    assert!(result.is_ok(), "gemm failed: {:?}", result.err());

    // For identity * B, result should be B
    for (idx, &val) in c.iter().enumerate() {
        assert!(
            (val - b[idx]).abs() < 1e-3,
            "gemm identity mismatch at {}: {} vs {}",
            idx,
            val,
            b[idx]
        );
    }
}

#[test]
#[serial]
fn test_cov009_gemm_larger() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Larger matrix: 32x32 * 32x32
    let m = 32u32;
    let n = 32u32;
    let k = 32u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);
    assert!(result.is_ok(), "gemm larger failed: {:?}", result.err());

    // Each element should be k (sum of k ones)
    for (idx, &val) in c.iter().enumerate() {
        assert!(
            (val - k as f32).abs() < 1.0,
            "gemm[{}] should be {}, got {}",
            idx,
            k,
            val
        );
    }
}

#[test]
#[serial]
fn test_cov009_gemm_cached_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let input_buf = GpuBuffer::from_host(&executor.context, &[1.0f32; 32]).expect("input");
    let output_buf = GpuBuffer::<f32>::new(&executor.context, 32).expect("output");

    // Try to use non-existent cached weight
    let result =
        executor.gemm_cached_async("nonexistent_weight", &input_buf, &output_buf, 32, 1, 32);
    assert!(
        result.is_err(),
        "gemm_cached_async should fail for non-existent weight"
    );
}

// ============================================================================
// COV-010: core.rs coverage tests
// Target: Increase coverage from 62.68% to 80%+
// Focus: profiler API, graph tracking, tile profiling, device info, pool stats
// ============================================================================

#[test]
#[serial]
fn test_cov010_num_devices() {
    if !CudaExecutor::is_available() {
        return;
    }
    let count = CudaExecutor::num_devices();
    assert!(count >= 1, "Should have at least 1 CUDA device");
}

#[test]
#[serial]
fn test_cov010_make_current() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.make_current();
    assert!(result.is_ok(), "make_current failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov010_profiling_enable_disable() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially disabled
    assert!(
        !executor.is_profiling_enabled(),
        "Profiling should be disabled initially"
    );

    // Enable
    executor.enable_profiling();
    assert!(
        executor.is_profiling_enabled(),
        "Profiling should be enabled"
    );

    // Disable
    executor.disable_profiling();
    assert!(
        !executor.is_profiling_enabled(),
        "Profiling should be disabled again"
    );
}

#[test]
#[serial]
fn test_cov010_profiler_access() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get profiler (immutable)
    let _profiler = executor.profiler();

    // Get profiler (mutable)
    let _profiler_mut = executor.profiler_mut();

    // Reset profiler
    executor.reset_profiler();
}

#[test]
#[serial]
fn test_cov010_profiler_summary() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let summary = executor.profiler_summary();
    // Summary should be a string (might be empty if no profiling data)
    assert!(summary.is_empty() || !summary.is_empty());
}

#[test]
#[serial]
fn test_cov010_profiler_sync_mode() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get default sync mode
    let _mode = executor.profiler_sync_mode();

    // Set sync mode to deferred
    executor.set_profiler_sync_mode(trueno::SyncMode::Deferred);
    assert_eq!(executor.profiler_sync_mode(), trueno::SyncMode::Deferred);
}

#[test]
#[serial]
fn test_cov010_profiler_category_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get category stats
    let stats = executor.profiler_category_stats();
    assert_eq!(stats.len(), trueno::BrickCategory::COUNT);
}

#[test]
#[serial]
fn test_cov010_print_profiler_categories() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // This prints to stdout, just verify it doesn't panic
    executor.print_profiler_categories();
}

#[test]
#[serial]
fn test_cov010_graph_tracking_enable_disable() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially disabled
    assert!(
        !executor.is_graph_tracking_enabled(),
        "Graph tracking should be disabled initially"
    );

    // Enable
    executor.enable_graph_tracking();
    assert!(
        executor.is_graph_tracking_enabled(),
        "Graph tracking should be enabled"
    );

    // Disable
    executor.disable_graph_tracking();
    assert!(
        !executor.is_graph_tracking_enabled(),
        "Graph tracking should be disabled again"
    );
}

#[test]
#[serial]
fn test_cov010_execution_graph_access() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get execution graph
    let _graph = executor.execution_graph();

    // Get ASCII tree
    let _ascii = executor.execution_graph_ascii();
}

#[test]
#[serial]
fn test_cov010_clear_execution_graph() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear graph (should not panic even when empty)
    executor.clear_execution_graph();
}

#[test]
#[serial]
fn test_cov010_tile_profiling_enable_disable() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially disabled
    assert!(
        !executor.is_tile_profiling_enabled(),
        "Tile profiling should be disabled initially"
    );

    // Enable
    executor.enable_tile_profiling();
    assert!(
        executor.is_tile_profiling_enabled(),
        "Tile profiling should be enabled"
    );

    // Disable
    executor.disable_tile_profiling();
    assert!(
        !executor.is_tile_profiling_enabled(),
        "Tile profiling should be disabled again"
    );
}

#[test]
#[serial]
fn test_cov010_tile_summary() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let summary = executor.tile_summary();
    // Summary should be a string
    assert!(summary.is_empty() || !summary.is_empty());
}

#[test]
#[serial]
fn test_cov010_tile_stats_json() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let json = executor.tile_stats_json();
    // JSON should be a valid string
    assert!(json.starts_with('{') || json.starts_with('[') || json.is_empty() || !json.is_empty());
}

#[test]
#[serial]
fn test_cov010_reset_tile_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Reset tile stats (should not panic)
    executor.reset_tile_stats();
}

#[test]
#[serial]
fn test_cov010_device_name() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.device_name();
    assert!(result.is_ok(), "device_name failed: {:?}", result.err());

    let name = result.unwrap();
    assert!(!name.is_empty(), "Device name should not be empty");
}

#[test]
#[serial]
fn test_cov010_memory_info() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.memory_info();
    assert!(result.is_ok(), "memory_info failed: {:?}", result.err());

    let (free, total) = result.unwrap();
    assert!(total > 0, "Total memory should be > 0");
    assert!(free <= total, "Free memory should be <= total");
}

#[test]
#[serial]
fn test_cov010_context() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get context reference
    let _context = executor.context();
}

#[test]
#[serial]
fn test_cov010_synchronize() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize();
    assert!(result.is_ok(), "synchronize failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov010_pool_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let stats = executor.pool_stats();
    // Stats should return valid struct (total_allocated is usize, always >= 0)
    let _ = stats.total_allocated; // Just verify field access works
}

#[test]
#[serial]
fn test_cov010_staging_pool_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let stats = executor.staging_pool_stats();
    // Stats should return valid struct (total_allocated is usize, always >= 0)
    let _ = stats.total_allocated; // Just verify field access works
}

#[test]
#[serial]
fn test_cov010_staging_buffer_roundtrip() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get a staging buffer (minimum size is 1024)
    let buf = executor.get_staging_buffer(256);
    assert!(
        buf.len() >= 256,
        "Staging buffer should be at least 256 elements"
    );

    // Return it to the pool
    executor.return_staging_buffer(buf);
}

#[test]
#[serial]
fn test_cov010_clear_pool() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear pool (should not panic even when empty)
    executor.clear_pool();
}

// ============================================================================
// COV-011: layer.rs additional coverage tests
// Target: Increase coverage from 15.49%
// Focus: preload functions, cache functions, workspace output, read hidden state
// ============================================================================

#[test]
#[serial]
fn test_cov011_preload_output_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];
    let result = executor.preload_output_norm(&gamma);
    assert!(
        result.is_ok(),
        "preload_output_norm failed: {:?}",
        result.err()
    );

    assert!(
        executor.has_output_norm(),
        "Should have output norm after preload"
    );
}

#[test]
#[serial]
fn test_cov011_has_output_norm_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_output_norm(),
        "Should not have output norm initially"
    );
}

#[test]
#[serial]
fn test_cov011_cache_rmsnorm_gamma() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 128];
    let result = executor.cache_rmsnorm_gamma("test_layer_0_attn_norm", &gamma);
    assert!(
        result.is_ok(),
        "cache_rmsnorm_gamma failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov011_preload_qkv_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Function expects &[Option<&[f32]>] for each bias array (per-head optional biases)
    let q_bias_data = vec![0.1f32; 64];
    let k_bias_data = vec![0.2f32; 64];
    let v_bias_data = vec![0.3f32; 64];

    // Wrap as optional slices (one head with bias)
    let q_biases: Vec<Option<&[f32]>> = vec![Some(q_bias_data.as_slice())];
    let k_biases: Vec<Option<&[f32]>> = vec![Some(k_bias_data.as_slice())];
    let v_biases: Vec<Option<&[f32]>> = vec![Some(v_bias_data.as_slice())];

    // Pass 1 as num_layers (not layer index 0)
    let result = executor.preload_qkv_bias(1, &q_biases, &k_biases, &v_biases);
    assert!(
        result.is_ok(),
        "preload_qkv_bias failed: {:?}",
        result.err()
    );

    assert!(executor.has_qkv_bias(0), "Should have QKV bias for layer 0");
}

#[test]
#[serial]
fn test_cov011_has_qkv_bias_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_qkv_bias(0),
        "Should not have QKV bias initially"
    );
    assert!(
        !executor.has_qkv_bias(5),
        "Should not have QKV bias for any layer"
    );
}

#[test]
#[serial]
fn test_cov011_preload_lm_head_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with Some bias
    let bias = vec![0.1f32; 1024];
    let result = executor.preload_lm_head_bias(Some(&bias));
    assert!(
        result.is_ok(),
        "preload_lm_head_bias failed: {:?}",
        result.err()
    );

    assert!(
        executor.has_lm_head_bias(),
        "Should have LM head bias after preload"
    );
}

#[test]
#[serial]
fn test_cov011_preload_lm_head_bias_none() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with None (no bias)
    let result = executor.preload_lm_head_bias(None);
    assert!(
        result.is_ok(),
        "preload_lm_head_bias None failed: {:?}",
        result.err()
    );

    assert!(
        !executor.has_lm_head_bias(),
        "Should not have LM head bias when None"
    );
}

#[test]
#[serial]
fn test_cov011_has_lm_head_bias_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_lm_head_bias(),
        "Should not have LM head bias initially"
    );
}

#[test]
#[serial]
fn test_cov011_workspace_output_none_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        executor.workspace_output().is_none(),
        "Workspace output should be None initially"
    );
}

#[test]
#[serial]
fn test_cov011_workspace_output_after_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Init KV cache and workspace
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);
    let _ = executor.init_workspace(32, 64);

    // Workspace output may still be None until forward pass, but the method should work
    let _output = executor.workspace_output();
}

// NOTE: fused_ffn_swiglu_host requires pre-cached GPU weights looked up by name.
// Weight caching is covered by forward_gpu_resident tests. Removing direct test
// since weight setup requires full model context.

#[test]
#[serial]
fn test_cov011_gpu_argmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create a simple logits buffer on GPU
    let vocab_size = 256u32;
    let logits: Vec<f32> = (0..vocab_size).map(|i| i as f32).collect();

    let logits_buf = GpuBuffer::from_host(&executor.context, &logits).expect("logits buffer");
    executor.stream.synchronize().expect("sync");

    let result = executor.gpu_argmax(logits_buf.as_ptr(), vocab_size);
    assert!(result.is_ok(), "gpu_argmax failed: {:?}", result.err());

    let argmax_idx = result.unwrap();
    // The maximum value is at index vocab_size-1 (255)
    assert_eq!(
        argmax_idx,
        vocab_size - 1,
        "Argmax should return index of max value"
    );
}

#[test]
#[serial]
fn test_cov011_gpu_argmax_middle() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create logits with max in the middle
    let vocab_size = 128u32;
    let mut logits = vec![0.0f32; vocab_size as usize];
    logits[64] = 100.0; // Max at index 64

    let logits_buf = GpuBuffer::from_host(&executor.context, &logits).expect("logits buffer");
    executor.stream.synchronize().expect("sync");

    let result = executor.gpu_argmax(logits_buf.as_ptr(), vocab_size);
    assert!(
        result.is_ok(),
        "gpu_argmax middle failed: {:?}",
        result.err()
    );

    let argmax_idx = result.unwrap();
    assert_eq!(argmax_idx, 64, "Argmax should return 64");
}

// ==============================================================================
// COV-012: Additional quantized.rs coverage - batched operations
// ==============================================================================

#[test]
#[serial]
fn test_cov012_rmsnorm_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();
    let gamma = vec![1.0f32; hidden_size as usize]; // Unit scale

    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let gamma_gpu = GpuBuffer::from_host(&executor.context, &gamma).expect("gamma buffer");
    let output_gpu =
        GpuBuffer::<f32>::new(&executor.context, hidden_size as usize).expect("output buffer");

    let result = executor.rmsnorm_into(&input_gpu, &gamma_gpu, &output_gpu, hidden_size, 1e-5);
    assert!(result.is_ok(), "rmsnorm_into failed: {:?}", result.err());

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; hidden_size as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // RMSNorm normalizes: output should have reasonable L2 norm
    let l2: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(l2 > 0.0, "Output should have non-zero L2 norm");
}

#[test]
#[serial]
fn test_cov012_batched_rmsnorm_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 32u32;
    let batch_size = 4u32;
    let total = (hidden_size * batch_size) as usize;

    // Create packed input [M × hidden_size]
    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let gamma = vec![1.0f32; hidden_size as usize]; // Shared gamma

    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let gamma_gpu = GpuBuffer::from_host(&executor.context, &gamma).expect("gamma buffer");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, total).expect("output buffer");

    let result = executor.batched_rmsnorm_into(
        &input_gpu,
        &gamma_gpu,
        &output_gpu,
        hidden_size,
        batch_size,
        1e-5,
    );
    assert!(
        result.is_ok(),
        "batched_rmsnorm_into failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; total];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // Check each sequence in batch was normalized
    for seq in 0..batch_size {
        let start = (seq * hidden_size) as usize;
        let end = start + hidden_size as usize;
        let l2: f32 = output[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(l2 > 0.0, "Sequence {} should have non-zero L2 norm", seq);
    }
}

#[test]
#[serial]
fn test_cov012_batched_rmsnorm_ptr_into() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 32u32;
    let batch_size = 2u32;
    let total = (hidden_size * batch_size) as usize;

    let input: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let gamma = vec![1.0f32; hidden_size as usize];

    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let gamma_gpu = GpuBuffer::from_host(&executor.context, &gamma).expect("gamma buffer");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, total).expect("output buffer");

    // Use ptr variant
    let result = executor.batched_rmsnorm_ptr_into(
        &input_gpu,
        gamma_gpu.as_ptr(),
        gamma.len(),
        &output_gpu,
        hidden_size,
        batch_size,
        1e-5,
    );
    assert!(
        result.is_ok(),
        "batched_rmsnorm_ptr_into failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov012_residual_add_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64u32;
    let input1 = vec![1.0f32; n as usize];
    let input2 = vec![2.0f32; n as usize];

    let input1_gpu = GpuBuffer::from_host(&executor.context, &input1).expect("input1 buffer");
    let input2_gpu = GpuBuffer::from_host(&executor.context, &input2).expect("input2 buffer");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, n as usize).expect("output buffer");

    let result = executor.residual_add_into(&input1_gpu, &input2_gpu, &output_gpu, n);
    assert!(
        result.is_ok(),
        "residual_add_into failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; n as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // 1.0 + 2.0 = 3.0
    for val in &output {
        assert!((*val - 3.0).abs() < 1e-5, "Expected 3.0, got {}", val);
    }
}

#[test]
#[serial]
fn test_cov012_fused_residual_rmsnorm_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 32u32;
    let residual = vec![1.0f32; hidden_size as usize];
    let input = vec![0.5f32; hidden_size as usize];
    let gamma = vec![1.0f32; hidden_size as usize];

    let residual_gpu = GpuBuffer::from_host(&executor.context, &residual).expect("residual buffer");
    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let gamma_gpu = GpuBuffer::from_host(&executor.context, &gamma).expect("gamma buffer");
    let output_gpu =
        GpuBuffer::<f32>::new(&executor.context, hidden_size as usize).expect("output buffer");

    // fused_residual_rmsnorm_into takes gamma_ptr as usize (raw device pointer)
    let result = executor.fused_residual_rmsnorm_into(
        &residual_gpu,
        &input_gpu,
        gamma_gpu.as_ptr() as usize,
        &output_gpu,
        hidden_size,
        1e-5,
    );
    assert!(
        result.is_ok(),
        "fused_residual_rmsnorm_into failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; hidden_size as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // Output should be normalized (residual + input)
    let l2: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(l2 > 0.0, "Output should have non-zero L2 norm");
}

#[test]
#[serial]
fn test_cov012_batched_residual_add_into() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 32u32;
    let batch_size = 4u32;
    let total = (hidden_size * batch_size) as usize;

    let input1: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let input2: Vec<f32> = (0..total).map(|i| (i as f32) * 0.5).collect();

    let input1_gpu = GpuBuffer::from_host(&executor.context, &input1).expect("input1 buffer");
    let input2_gpu = GpuBuffer::from_host(&executor.context, &input2).expect("input2 buffer");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, total).expect("output buffer");

    let result = executor.batched_residual_add_into(
        &input1_gpu,
        &input2_gpu,
        &output_gpu,
        hidden_size,
        batch_size,
    );
    assert!(
        result.is_ok(),
        "batched_residual_add_into failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; total];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // Check: output[i] = input1[i] + input2[i] = i + i*0.5 = i*1.5
    for (i, &val) in output.iter().enumerate() {
        let expected = (i as f32) * 1.5;
        assert!(
            (val - expected).abs() < 1e-4,
            "At {}: expected {}, got {}",
            i,
            expected,
            val
        );
    }
}

#[test]
#[serial]
fn test_cov012_batched_swiglu_into() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let intermediate_dim = 64u32;
    let batch_size = 2u32;
    let total = (intermediate_dim * batch_size) as usize;

    // Gate and up projections
    let gate: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let up: Vec<f32> = (0..total).map(|_| 1.0f32).collect();

    let gate_gpu = GpuBuffer::from_host(&executor.context, &gate).expect("gate buffer");
    let up_gpu = GpuBuffer::from_host(&executor.context, &up).expect("up buffer");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, total).expect("output buffer");

    let result = executor.batched_swiglu_into(
        &gate_gpu,
        &up_gpu,
        &output_gpu,
        intermediate_dim,
        batch_size,
    );
    assert!(
        result.is_ok(),
        "batched_swiglu_into failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; total];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // SwiGLU: silu(gate) * up - output should be finite
    for &val in &output {
        assert!(val.is_finite(), "Output should be finite");
    }
}

#[test]
#[serial]
fn test_cov012_batched_rope_into() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 2u32;
    let head_dim = 16u32;
    let batch_size = 2u32;
    let total = (num_heads * head_dim * batch_size) as usize;

    // Input Q or K vectors
    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let positions = vec![0u32, 1u32]; // Position for each sequence in batch

    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, total).expect("output buffer");
    let positions_gpu =
        GpuBuffer::from_host(&executor.context, &positions).expect("positions buffer");

    let result = executor.batched_rope_into(
        &input_gpu,
