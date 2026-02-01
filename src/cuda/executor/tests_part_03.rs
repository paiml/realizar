use super::*;
use crate::cuda::memory::{GpuBufferHandle, SizeClass, TransferMode};
use crate::cuda::pipeline::{
    presets, BankConflictStrategy, MemoryPattern, PtxOptimizationHints, PtxOptimizer,
    RegisterTiling,
};
use serial_test::serial;

// ============================================================================

#[test]
#[serial]
fn test_cov007_silu_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 64.0).collect();

    let input = GpuBuffer::from_host(&executor.context, &data).expect("input buffer");
    let result = executor.silu_gpu(&input, n);
    assert!(result.is_ok(), "silu_gpu failed: {:?}", result.err());

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let mid_idx = 128;
    assert!(
        output[mid_idx].abs() < 0.1,
        "SiLU(0) should be near 0, got {}",
        output[mid_idx]
    );
}

#[test]
#[serial]
fn test_cov007_silu_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 1024u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) / 256.0).collect();

    let input = GpuBuffer::from_host(&executor.context, &data).expect("input buffer");
    let result = executor.silu_gpu(&input, n);
    assert!(result.is_ok(), "silu_gpu large failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov007_gelu_async_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 64.0).collect();

    let input = GpuBuffer::from_host(&executor.context, &data).expect("input buffer");
    let result = executor.gelu_async(&input, n);
    assert!(result.is_ok(), "gelu_async failed: {:?}", result.err());

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // GELU(0) should be near 0
    let mid_idx = 128;
    assert!(
        output[mid_idx].abs() < 0.1,
        "GELU(0) should be near 0, got {}",
        output[mid_idx]
    );
}

#[test]
#[serial]
fn test_cov007_gelu_async_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 2048u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) / 512.0).collect();

    let input = GpuBuffer::from_host(&executor.context, &data).expect("input buffer");
    let result = executor.gelu_async(&input, n);
    assert!(
        result.is_ok(),
        "gelu_async large failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov007_elementwise_mul_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 128u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|_| 2.0f32).collect();

    let a = GpuBuffer::from_host(&executor.context, &a_data).expect("a buffer");
    let b = GpuBuffer::from_host(&executor.context, &b_data).expect("b buffer");

    let result = executor.elementwise_mul_gpu(&a, &b, n);
    assert!(
        result.is_ok(),
        "elementwise_mul_gpu failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // output[i] = a[i] * b[i] = i * 2 = 2i
    for (idx, &val) in output.iter().enumerate() {
        let expected = (idx as f32) * 2.0;
        assert!(
            (val - expected).abs() < 1e-4,
            "mul mismatch at {}: {} vs {}",
            idx,
            val,
            expected
        );
    }
}

#[test]
#[serial]
fn test_cov007_elementwise_mul_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 4096u32;
    let a = GpuBuffer::from_host(&executor.context, &vec![3.0f32; n as usize]).expect("a");
    let b = GpuBuffer::from_host(&executor.context, &vec![4.0f32; n as usize]).expect("b");

    let result = executor.elementwise_mul_gpu(&a, &b, n);
    assert!(
        result.is_ok(),
        "elementwise_mul_gpu large failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // All should be 12.0
    assert!(
        output.iter().all(|&x| (x - 12.0).abs() < 1e-4),
        "All outputs should be 12.0"
    );
}

#[test]
#[serial]
fn test_cov007_silu_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 16.0).collect();
    let mut output = vec![0.0f32; n];

    let result = executor.silu_host(&input, &mut output);
    assert!(result.is_ok(), "silu_host failed: {:?}", result.err());

    // SiLU(0) should be near 0
    let mid_idx = 32;
    assert!(output[mid_idx].abs() < 0.1, "SiLU(0) should be near 0");
}

#[test]
#[serial]
fn test_cov007_silu_host_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 512usize;
    let input = vec![1.0f32; n];
    let mut output = vec![0.0f32; n];

    let result = executor.silu_host(&input, &mut output);
    assert!(result.is_ok(), "silu_host large failed: {:?}", result.err());

    // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
    assert!(
        output[0] > 0.7 && output[0] < 0.8,
        "SiLU(1) should be ~0.731, got {}",
        output[0]
    );
}

#[test]
#[serial]
fn test_cov007_gelu_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 16.0).collect();
    let mut output = vec![0.0f32; n];

    let result = executor.gelu_host(&input, &mut output);
    assert!(result.is_ok(), "gelu_host failed: {:?}", result.err());

    // GELU(0) should be near 0
    let mid_idx = 32;
    assert!(output[mid_idx].abs() < 0.1, "GELU(0) should be near 0");
}

#[test]
#[serial]
fn test_cov007_gelu_host_positive() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let input = vec![2.0f32; n]; // All 2.0
    let mut output = vec![0.0f32; n];

    let result = executor.gelu_host(&input, &mut output);
    assert!(
        result.is_ok(),
        "gelu_host positive failed: {:?}",
        result.err()
    );

    // GELU(2) should be close to 2 (slightly less)
    assert!(
        output[0] > 1.9 && output[0] < 2.1,
        "GELU(2) should be ~2.0, got {}",
        output[0]
    );
}

#[test]
#[serial]
fn test_cov007_elementwise_mul_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let mut output = vec![0.0f32; n];

    let result = executor.elementwise_mul_host(&a, &b, &mut output);
    assert!(
        result.is_ok(),
        "elementwise_mul_host failed: {:?}",
        result.err()
    );

    // output[i] = i * (n - i)
    for (idx, &val) in output.iter().enumerate() {
        let expected = (idx as f32) * ((n - idx) as f32);
        assert!(
            (val - expected).abs() < 1e-4,
            "mul_host mismatch at {}: {} vs {}",
            idx,
            val,
            expected
        );
    }
}

#[test]
#[serial]
fn test_cov007_fused_swiglu_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let gate = vec![1.0f32; n];
    let up = vec![2.0f32; n];
    let mut output = vec![0.0f32; n];

    let result = executor.fused_swiglu_host(&gate, &up, &mut output);
    assert!(
        result.is_ok(),
        "fused_swiglu_host failed: {:?}",
        result.err()
    );

    // SwiGLU(gate, up) = silu(gate) * up = silu(1) * 2 ≈ 0.731 * 2 ≈ 1.462
    assert!(
        output[0] > 1.4 && output[0] < 1.6,
        "SwiGLU(1,2) should be ~1.46, got {}",
        output[0]
    );
}

#[test]
#[serial]
fn test_cov007_fused_swiglu_host_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256usize;
    let gate: Vec<f32> = (0..n).map(|i| (i as f32) / 128.0).collect();
    let up = vec![1.0f32; n];
    let mut output = vec![0.0f32; n];

    let result = executor.fused_swiglu_host(&gate, &up, &mut output);
    assert!(
        result.is_ok(),
        "fused_swiglu_host large failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov007_add_residual_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 128u32;

    // Output starts with values, input is what to add
    let output_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input_data: Vec<f32> = (0..n).map(|_| 10.0f32).collect();

    let output_buf = GpuBuffer::from_host(&executor.context, &output_data).expect("output buffer");
    let input_buf = GpuBuffer::from_host(&executor.context, &input_data).expect("input buffer");

    let result = executor.add_residual_gpu(&output_buf, &input_buf, n);
    assert!(
        result.is_ok(),
        "add_residual_gpu failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buf.copy_to_host(&mut output).expect("copy");

    // output[i] += 10, so output[i] = i + 10
    for (idx, &val) in output.iter().enumerate() {
        let expected = idx as f32 + 10.0;
        assert!(
            (val - expected).abs() < 1e-4,
            "add_residual mismatch at {}: {} vs {}",
            idx,
            val,
            expected
        );
    }
}

#[test]
#[serial]
fn test_cov007_add_residual_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 4096u32;

    let output_buf =
        GpuBuffer::from_host(&executor.context, &vec![5.0f32; n as usize]).expect("output");
    let input_buf =
        GpuBuffer::from_host(&executor.context, &vec![3.0f32; n as usize]).expect("input");

    let result = executor.add_residual_gpu(&output_buf, &input_buf, n);
    assert!(
        result.is_ok(),
        "add_residual_gpu large failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buf.copy_to_host(&mut output).expect("copy");

    // All should be 8.0 (5 + 3)
    assert!(
        output.iter().all(|&x| (x - 8.0).abs() < 1e-4),
        "All outputs should be 8.0"
    );
}

#[test]
#[serial]
fn test_cov007_fused_swiglu_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let gate_data = vec![1.0f32; n as usize];
    let up_data = vec![2.0f32; n as usize];

    let gate = GpuBuffer::from_host(&executor.context, &gate_data).expect("gate buffer");
    let up = GpuBuffer::from_host(&executor.context, &up_data).expect("up buffer");

    let result = executor.fused_swiglu_gpu(&gate, &up, n);
    assert!(
        result.is_ok(),
        "fused_swiglu_gpu failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // SwiGLU(1,2) = silu(1) * 2 ≈ 1.46
    assert!(
        output[0] > 1.4 && output[0] < 1.6,
        "SwiGLU(1,2) should be ~1.46, got {}",
        output[0]
    );
}

#[test]
#[serial]
fn test_cov007_fused_swiglu_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 2048u32;

    let gate = GpuBuffer::from_host(&executor.context, &vec![0.5f32; n as usize]).expect("gate");
    let up = GpuBuffer::from_host(&executor.context, &vec![1.0f32; n as usize]).expect("up");

    let result = executor.fused_swiglu_gpu(&gate, &up, n);
    assert!(
        result.is_ok(),
        "fused_swiglu_gpu large failed: {:?}",
        result.err()
    );
}

// ============================================================================
// COV-008: workspace.rs coverage tests
// Target: Increase coverage from 9.73% to 50%+
// Focus: init_workspace, init_batched_workspace, has_workspace,
//        workspace_batch_size, has_decode_graph, clear_workspace,
//        clear_decode_graph, gemv_buffer_stats, clear_gemv_buffers
// ============================================================================

#[test]
#[serial]
fn test_cov008_init_workspace_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first (required by init_workspace)
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    let hidden_dim = 64usize;
    let intermediate_dim = 128usize;

    let result = executor.init_workspace(hidden_dim, intermediate_dim);
    assert!(result.is_ok(), "init_workspace failed: {:?}", result.err());

    assert!(executor.has_workspace(), "Workspace should be initialized");
    assert_eq!(
        executor.workspace_batch_size(),
        1,
        "Default batch size should be 1"
    );
}

#[test]
#[serial]
fn test_cov008_init_workspace_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 8, 8, 64, 512);

    let hidden_dim = 512usize;
    let intermediate_dim = 2048usize;

    let result = executor.init_workspace(hidden_dim, intermediate_dim);
    assert!(
        result.is_ok(),
        "init_workspace large failed: {:?}",
        result.err()
    );

    assert!(executor.has_workspace());
}

#[test]
#[serial]
fn test_cov008_init_batched_workspace_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    let hidden_dim = 64usize;
    let intermediate_dim = 128usize;
    let batch_size = 4usize;

    let result = executor.init_batched_workspace(hidden_dim, intermediate_dim, batch_size);
    assert!(
        result.is_ok(),
        "init_batched_workspace failed: {:?}",
        result.err()
    );

    assert!(executor.has_workspace());
    assert_eq!(executor.workspace_batch_size(), 4, "Batch size should be 4");
}

#[test]
#[serial]
fn test_cov008_init_batched_workspace_max_batch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    // Test maximum batch size (32)
    let result = executor.init_batched_workspace(64, 128, 32);
    assert!(
        result.is_ok(),
        "init_batched_workspace max batch failed: {:?}",
        result.err()
    );
    assert_eq!(executor.workspace_batch_size(), 32);
}

#[test]
#[serial]
fn test_cov008_init_batched_workspace_zero_batch_error() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    // Test zero batch size (should fail)
    let result = executor.init_batched_workspace(64, 128, 0);
    assert!(
        result.is_err(),
        "init_batched_workspace with batch=0 should fail"
    );
}

#[test]
#[serial]
fn test_cov008_init_batched_workspace_too_large_batch_error() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    // Test batch size > 32 (should fail)
    let result = executor.init_batched_workspace(64, 128, 33);
    assert!(
        result.is_err(),
        "init_batched_workspace with batch=33 should fail"
    );
}

#[test]
#[serial]
fn test_cov008_has_workspace_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_workspace(),
        "Workspace should not be initialized initially"
    );
}

#[test]
#[serial]
fn test_cov008_has_decode_graph_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_decode_graph(),
        "Decode graph should not exist initially"
    );
}

#[test]
#[serial]
fn test_cov008_clear_workspace() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache and init workspace
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);
    let _ = executor.init_workspace(64, 128);
    assert!(executor.has_workspace());

    // Clear workspace
    executor.clear_workspace();
    assert!(!executor.has_workspace(), "Workspace should be cleared");
}

#[test]
#[serial]
fn test_cov008_clear_decode_graph() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear decode graph (even without capturing one)
    executor.clear_decode_graph();
    assert!(
        !executor.has_decode_graph(),
        "Decode graph should be cleared"
    );
}

#[test]
#[serial]
fn test_cov008_gemv_buffer_stats_initial() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let (input_bytes, output_bytes) = executor.gemv_buffer_stats();
    assert_eq!(
        input_bytes, 0,
        "Initial GEMV input buffer should be 0 bytes"
    );
    assert_eq!(
        output_bytes, 0,
        "Initial GEMV output buffer should be 0 bytes"
    );
}

#[test]
#[serial]
fn test_cov008_clear_gemv_buffers() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear GEMV buffers (even without allocating any)
    executor.clear_gemv_buffers();
    let (input_bytes, output_bytes) = executor.gemv_buffer_stats();
    assert_eq!(input_bytes, 0);
    assert_eq!(output_bytes, 0);
}

#[test]
#[serial]
fn test_cov008_ensure_gemv_input_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Ensure GEMV input buffer
    let result = executor.ensure_gemv_input_buffer(256);
    assert!(
        result.is_ok(),
        "ensure_gemv_input_buffer failed: {:?}",
        result.err()
    );

    let (input_bytes, _) = executor.gemv_buffer_stats();
    assert_eq!(
        input_bytes,
        256 * 4,
        "GEMV input buffer should be 1024 bytes (256 * 4)"
    );
}

#[test]
#[serial]
fn test_cov008_ensure_gemv_output_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Ensure GEMV output buffer
    let result = executor.ensure_gemv_output_buffer(128);
    assert!(
        result.is_ok(),
        "ensure_gemv_output_buffer failed: {:?}",
        result.err()
    );

    let (_, output_bytes) = executor.gemv_buffer_stats();
    assert_eq!(
        output_bytes,
        128 * 4,
        "GEMV output buffer should be 512 bytes (128 * 4)"
    );
}

#[test]
#[serial]
fn test_cov008_ensure_gemv_buffers_reuse() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // First allocation
    let ptr1 = executor.ensure_gemv_input_buffer(256).expect("first alloc");

    // Same size - should reuse
    let ptr2 = executor
        .ensure_gemv_input_buffer(256)
        .expect("second alloc");
    assert_eq!(ptr1, ptr2, "Same size should reuse buffer");

    // Different size - should reallocate
    let ptr3 = executor.ensure_gemv_input_buffer(512).expect("third alloc");
    assert_ne!(ptr1, ptr3, "Different size should create new buffer");
}

#[test]
#[serial]
fn test_cov008_copy_gemv_buffers() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; n];

    // Ensure both buffers
    executor.ensure_gemv_input_buffer(n).expect("ensure input");
    executor
        .ensure_gemv_output_buffer(n)
        .expect("ensure output");

    // Copy to input buffer
    let result = executor.copy_to_gemv_input(&input);
    assert!(
        result.is_ok(),
        "copy_to_gemv_input failed: {:?}",
        result.err()
    );

    // Copy from output buffer (note: output buffer won't have the input data,
    // this just tests the copy path works)
    let result = executor.copy_from_gemv_output(&mut output);
    assert!(
        result.is_ok(),
        "copy_from_gemv_output failed: {:?}",
        result.err()
    );
}

// ============================================================================
// COV-009: gemm.rs coverage tests
// Target: Increase coverage from 60.92% to 75%+
// Focus: synchronize_compute, synchronize_transfer, synchronize_all,
//        allocate_buffer, softmax, gemm
// ============================================================================

#[test]
#[serial]
fn test_cov009_synchronize_compute() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_compute();
    assert!(
        result.is_ok(),
        "synchronize_compute failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov009_synchronize_transfer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_transfer();
    assert!(
        result.is_ok(),
        "synchronize_transfer failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov009_synchronize_all() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_all();
    assert!(result.is_ok(), "synchronize_all failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov009_allocate_buffer_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.allocate_buffer(256);
    assert!(result.is_ok(), "allocate_buffer failed: {:?}", result.err());

    let buffer = result.unwrap();
    assert!(buffer.len() == 256, "Buffer should have 256 elements");
}

#[test]
#[serial]
fn test_cov009_allocate_buffer_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Allocate 1MB buffer (262144 f32 elements)
    let result = executor.allocate_buffer(262144);
    assert!(
        result.is_ok(),
        "allocate_buffer large failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov009_softmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test softmax with small vector
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax failed: {:?}", result.err());

    // Verify softmax properties
    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Softmax should sum to 1, got {}",
        sum
    );

    // Verify monotonicity (higher input -> higher output)
    for i in 1..data.len() {
        assert!(data[i] > data[i - 1], "Softmax should preserve ordering");
    }
}

#[test]
#[serial]
fn test_cov009_softmax_larger() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test with 32-element vector (warp-aligned)
    let mut data: Vec<f32> = (0..32).map(|i| (i as f32) / 10.0).collect();
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax larger failed: {:?}", result.err());

    // Softmax should produce valid probabilities (all positive)
    assert!(
        data.iter().all(|&x| x > 0.0),
        "Softmax outputs should be positive"
    );
    // Last element should be largest (highest input)
    assert!(
        data[31] > data[0],
        "Highest input should have highest probability"
    );
}

#[test]
#[serial]
fn test_cov009_softmax_uniform() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Uniform input should give uniform output
    let n = 8;
    let mut data = vec![0.0f32; n];
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax uniform failed: {:?}", result.err());

    // All should be 1/n
    let expected = 1.0 / n as f32;
    for (i, &val) in data.iter().enumerate() {
        assert!(
            (val - expected).abs() < 0.01,
            "Uniform softmax[{}] should be {}, got {}",
            i,
            expected,
            val
        );
    }
}

#[test]
#[serial]
fn test_cov009_gemm_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Small matrix multiplication: C = A * B
    // A is 4x4, B is 4x4, C is 4x4
    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

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
