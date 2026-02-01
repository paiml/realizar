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

