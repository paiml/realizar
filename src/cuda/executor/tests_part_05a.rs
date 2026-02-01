use super::*;
use crate::cuda::memory::{GpuBufferHandle, SizeClass, TransferMode};
use crate::cuda::pipeline::{
    presets, BankConflictStrategy, MemoryPattern, PtxOptimizationHints, PtxOptimizer,
    RegisterTiling,
};
use serial_test::serial;

    // Output should have non-zero values
    let has_nonzero = c.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}

// =========================================================================
// COV-020: gemm.rs coverage tests
// Target: 63.33% -> ~70%+ coverage
// =========================================================================

#[test]
#[serial]
fn test_cov020_synchronize_compute() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_compute();
    assert!(
        result.is_ok(),
        "synchronize_compute should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_synchronize_transfer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_transfer();
    assert!(
        result.is_ok(),
        "synchronize_transfer should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_synchronize_all() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_all();
    assert!(
        result.is_ok(),
        "synchronize_all should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_allocate_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.allocate_buffer(1024);
    assert!(
        result.is_ok(),
        "allocate_buffer should succeed: {:?}",
        result.err()
    );

    let buf = result.unwrap();
    assert_eq!(buf.len(), 1024, "Buffer should have correct length");
}

#[test]
#[serial]
fn test_cov020_gemm_cached_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let b = vec![0.1f32; 64 * 32];
    let mut c = vec![0.0f32; 64 * 32];

    let result = executor.gemm_cached("nonexistent_weight", &b, &mut c, 64, 32, 64);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_cached_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a weight
    let weight_data = vec![0.1f32; 64 * 64];
    executor
        .load_weights("test_weight", &weight_data)
        .expect("load weight");

    // Wrong B size
    let b = vec![0.1f32; 100]; // Wrong size
    let mut c = vec![0.0f32; 64 * 32];

    let result = executor.gemm_cached("test_weight", &b, &mut c, 64, 32, 64);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_b_cached_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let a = vec![0.1f32; 64 * 64];
    let mut c = vec![0.0f32; 64 * 32];

    let result = executor.gemm_b_cached("nonexistent_b_weight", &a, &mut c, 64, 32, 64);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_b_cached_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a weight (B matrix: k x n = 64 x 32)
    let weight_data = vec![0.1f32; 64 * 32];
    executor
        .load_weights("test_b_weight", &weight_data)
        .expect("load weight");

    // Wrong A size
    let a = vec![0.1f32; 100]; // Wrong size (should be m * k = 64 * 64)
    let mut c = vec![0.0f32; 64 * 32];

    let result = executor.gemm_b_cached("test_b_weight", &a, &mut c, 64, 32, 64);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    // A is wrong size
    let a = vec![0.1f32; 100]; // Should be m * k = 32 * 64 = 2048
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_valid_run() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);

    assert!(result.is_ok(), "gemm should succeed: {:?}", result.err());

    // Should have non-zero output
    let has_nonzero = c.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}

#[test]
#[serial]
fn test_cov020_gemm_gemv_path() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // M=1 triggers GEMV path
    let m = 1u32;
    let n = 64u32;
    let k = 128u32;

    let a = vec![0.1f32; (m * k) as usize]; // 1 x 128 row vector
    let b = vec![0.1f32; (k * n) as usize]; // 128 x 64 matrix
    let mut c = vec![0.0f32; (m * n) as usize]; // 1 x 64 output

    let result = executor.gemm(&a, &b, &mut c, m, n, k);

    assert!(
        result.is_ok(),
        "gemm (gemv path) should succeed: {:?}",
        result.err()
    );

    // Should have non-zero output
    let has_nonzero = c.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}

#[test]
#[serial]
fn test_cov020_gemv_cached_input_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a weight
    let k = 128u32;
    let n = 64u32;
    let weight_data = vec![0.1f32; (k * n) as usize];
    executor
        .load_weights("gemv_weight", &weight_data)
        .expect("load weight");

    // Wrong input size
    let x = vec![0.1f32; 50]; // Should be k = 128
    let mut y = vec![0.0f32; n as usize];

    let result = executor.gemv_cached("gemv_weight", &x, &mut y, k, n);

    assert!(result.is_err(), "Should fail with input size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemv_cached_output_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a weight
    let k = 128u32;
    let n = 64u32;
    let weight_data = vec![0.1f32; (k * n) as usize];
    executor
        .load_weights("gemv_weight2", &weight_data)
        .expect("load weight");

    let x = vec![0.1f32; k as usize]; // Correct input size
    let mut y = vec![0.0f32; 10]; // Wrong output size (should be n = 64)

    let result = executor.gemv_cached("gemv_weight2", &x, &mut y, k, n);

    assert!(result.is_err(), "Should fail with output size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemv_cached_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let k = 128u32;
    let n = 64u32;
    let x = vec![0.1f32; k as usize];
    let mut y = vec![0.0f32; n as usize];

    let result = executor.gemv_cached("nonexistent_gemv_weight", &x, &mut y, k, n);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_optimized_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 64u32;
    let n = 64u32;
    let k = 128u32;
    let tile_size = 32u32;

    // A is wrong size
    let a = vec![0.1f32; 100]; // Should be m * k
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_optimized(&a, &b, &mut c, m, n, k, tile_size);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_optimized_valid_run() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 64u32;
    let n = 64u32;
    let k = 128u32;
    let tile_size = 32u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_optimized(&a, &b, &mut c, m, n, k, tile_size);

    assert!(
        result.is_ok(),
        "gemm_optimized should succeed: {:?}",
        result.err()
    );

    let has_nonzero = c.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}

#[test]
#[serial]
fn test_cov020_gemm_fused_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    // A is wrong size
    let a = vec![0.1f32; 100];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_fused(&a, &b, None, &mut c, m, n, k, 0);

    assert!(result.is_err(), "Should fail with size mismatch");
}

#[test]
#[serial]
fn test_cov020_gemm_fused_bias_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];
    let bias = vec![0.1f32; 10]; // Wrong size, should be n = 32

    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 0);

    assert!(result.is_err(), "Should fail with bias size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Bias") || err_msg.contains("mismatch"),
        "Error should mention bias mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_fused_valid_no_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_fused(&a, &b, None, &mut c, m, n, k, 0);

    assert!(
        result.is_ok(),
        "gemm_fused (no bias) should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_gemm_fused_with_bias_and_relu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];
    let bias = vec![0.1f32; n as usize]; // Correct size

    // activation = 1 (ReLU)
    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 1);

    assert!(
        result.is_ok(),
        "gemm_fused with bias+relu should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_softmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];

    let result = executor.softmax(&mut data);

    assert!(result.is_ok(), "softmax should succeed: {:?}", result.err());

    // Check output sums to 1
    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Softmax should sum to 1, got {}",
        sum
    );

    // Check values are in (0, 1)
    for val in &data {
        assert!(
            *val > 0.0 && *val < 1.0,
            "Softmax values should be in (0, 1): {}",
            val
        );
    }
}

#[test]
#[serial]
fn test_cov020_gemm_cached_async_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let input_buf = GpuBuffer::<f32>::new(&executor.context, (k * n) as usize).expect("input");
    let output_buf = GpuBuffer::<f32>::new(&executor.context, (m * n) as usize).expect("output");

    let result =
        executor.gemm_cached_async("nonexistent_async_weight", &input_buf, &output_buf, m, n, k);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_q4k_matvec_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32; // Output dimension
    let k = 256u32; // Input dimension (must be divisible by 256 for Q4K)

    // Q4K format: 144 bytes per 256 values
    let num_superblocks = (m as usize * k as usize + 255) / 256;
    let weight_bytes = num_superblocks * 144;
    let weights = vec![0u8; weight_bytes];

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; m as usize];

    let result = executor.q4k_matvec(&weights, &input, &mut output, m, k);

    assert!(
        result.is_ok(),
        "q4k_matvec should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_q4k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // Output dimension
    let k = 256u32; // Input dimension (must be divisible by 256)

    // Q4K format: 144 bytes per 256 values
    let num_superblocks = (n as usize * k as usize + 255) / 256;
    let weight_bytes = num_superblocks * 144;
    let weights = vec![0u8; weight_bytes];

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q4k_gemv(&weights, &input, &mut output, n, k);

    assert!(
        result.is_ok(),
        "q4k_gemv should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_q5k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Q5K format: 176 bytes per 256 values
    let num_superblocks = (n as usize * k as usize + 255) / 256;
    let weight_bytes = num_superblocks * 176;
    let weights = vec![0u8; weight_bytes];

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q5k_gemv(&weights, &input, &mut output, n, k);

    assert!(
        result.is_ok(),
        "q5k_gemv should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_q6k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Q6K format: 210 bytes per 256 values
    let num_superblocks = (n as usize * k as usize + 255) / 256;
    let weight_bytes = num_superblocks * 210;
    let weights = vec![0u8; weight_bytes];

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q6k_gemv(&weights, &input, &mut output, n, k);

    assert!(
        result.is_ok(),
        "q6k_gemv should succeed: {:?}",
        result.err()
    );
}

// =============================================================================
// COV-021: activations.rs untested functions (Refs PMAT-802)
// =============================================================================

#[test]
#[serial]
fn test_cov021_q4k_gemv_gpu_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Create GPU buffers
    let input = vec![0.1f32; k as usize];
    let input_buf = GpuBuffer::from_host(executor.context(), &input).expect("input buf");
    let output_buf = GpuBuffer::new(executor.context(), n as usize).expect("output buf");

    // Weight not cached - should fail
    let result = executor.q4k_gemv_gpu("nonexistent_weight", &input_buf, &output_buf, n, k);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_tensor_core_q4k_gemm_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 16u32;
    let k = 256u32;
    let n = 32u32;

    // Create GPU buffers
    let input = vec![0.1f32; (m * k) as usize];
    let input_buf = GpuBuffer::from_host(executor.context(), &input).expect("input buf");
    let output_buf = GpuBuffer::new(executor.context(), (m * n) as usize).expect("output buf");

    // Weight not cached - should fail
    let result =
        executor.tensor_core_q4k_gemm("nonexistent_weight", &input_buf, &output_buf, m, k, n);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_tensor_core_q4k_gemm_cached_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    let input = vec![0.1f32; (m * k) as usize];
    let mut output = vec![0.0f32; (m * n) as usize];

    // Weight not cached - should fail
    let result =
        executor.tensor_core_q4k_gemm_cached("nonexistent_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_tensor_core_q4k_gemm_cached_input_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    // Wrong input size - should fail validation before checking weight
    let input = vec![0.1f32; 100]; // Should be m*k = 1024
    let mut output = vec![0.0f32; (m * n) as usize];

    let result = executor.tensor_core_q4k_gemm_cached("any_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail with input size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Input size"),
        "Error should mention input size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_tensor_core_q4k_gemm_cached_output_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    // Correct input size but wrong output size
    let input = vec![0.1f32; (m * k) as usize];
    let mut output = vec![0.0f32; 50]; // Should be m*n = 128

    let result = executor.tensor_core_q4k_gemm_cached("any_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail with output size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Output size"),
        "Error should mention output size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_batched_q4k_gemv_cached_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    let input = vec![0.1f32; (m * k) as usize];
    let mut output = vec![0.0f32; (m * n) as usize];

    // Weight not cached - should fail
    let result =
        executor.batched_q4k_gemv_cached("nonexistent_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_batched_q4k_gemv_cached_input_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    // Wrong input size - should fail validation
    let input = vec![0.1f32; 100]; // Should be m*k = 1024
    let mut output = vec![0.0f32; (m * n) as usize];

    let result = executor.batched_q4k_gemv_cached("any_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail with input size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Input size"),
        "Error should mention input size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_batched_q4k_gemv_cached_output_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    // Correct input size but wrong output size
    let input = vec![0.1f32; (m * k) as usize];
    let mut output = vec![0.0f32; 50]; // Should be m*n = 128

    let result = executor.batched_q4k_gemv_cached("any_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail with output size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Output size"),
        "Error should mention output size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_fused_ffn_q4k_up_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 256u32;
    let intermediate_dim = 512u32;

    let input = vec![0.1f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];

    // FFN up weight not cached - should fail
    let result = executor.fused_ffn_q4k(
        &input,
        &mut output,
        "nonexistent_up",
        "any_down",
        hidden_dim,
        intermediate_dim,
    );

    assert!(result.is_err(), "Should fail when up weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("FFN up weight") || err_msg.contains("not cached"),
        "Error should mention FFN up weight not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_fused_ffn_q4k_down_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 256u32;
    let intermediate_dim = 512u32;

    // Cache the up weight first (Q4K format: 144 bytes per 256 values)
    let num_superblocks = (intermediate_dim as usize * hidden_dim as usize + 255) / 256;
    let up_weight_bytes = num_superblocks * 144;
    let up_weights = vec![0u8; up_weight_bytes];
    executor
        .load_quantized_weights("test_up_weight", &up_weights)
        .expect("load up weight");

    let input = vec![0.1f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];

    // FFN down weight not cached - should fail
    let result = executor.fused_ffn_q4k(
        &input,
        &mut output,
        "test_up_weight",
        "nonexistent_down",
        hidden_dim,
        intermediate_dim,
    );

    assert!(result.is_err(), "Should fail when down weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("FFN down weight") || err_msg.contains("not cached"),
        "Error should mention FFN down weight not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_rope_neox_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4u32;
    let head_dim = 64u32;
    let total_size = (num_heads * head_dim) as usize;

    // Create GPU buffers for input and output
    let input_data = vec![0.5f32; total_size];
    let input_buf = GpuBuffer::from_host(executor.context(), &input_data).expect("input buf");
    let output_buf = GpuBuffer::new(executor.context(), total_size).expect("output buf");

    // Position starts at 0
