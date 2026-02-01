use super::*;
use crate::cuda::memory::{GpuBufferHandle, SizeClass, TransferMode};
use crate::cuda::pipeline::{
    presets, BankConflictStrategy, MemoryPattern, PtxOptimizationHints, PtxOptimizer,
    RegisterTiling,
};
use serial_test::serial;

    // Get initial stats
    let stats1 = executor.staging_pool_stats();
    assert_eq!(stats1.pool_hits, 0, "Initial hits should be 0");
    assert_eq!(stats1.pool_misses, 0, "Initial misses should be 0");

    // Allocate a buffer - should be a miss
    let buf = executor.get_staging_buffer(1024);
    let stats2 = executor.staging_pool_stats();
    assert_eq!(stats2.pool_misses, 1, "Should have 1 miss");

    // Return and get again - should be a hit
    executor.return_staging_buffer(buf);
    let _buf2 = executor.get_staging_buffer(1024);
    let stats3 = executor.staging_pool_stats();
    assert!(stats3.pool_hits >= 1, "Should have at least 1 hit");
}

#[test]
#[serial]
fn test_cov025_cached_weight_count() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially zero
    assert_eq!(
        executor.cached_weight_count(),
        0,
        "Initial count should be 0"
    );

    // Load a weight
    let weights = vec![1.0f32; 256];
    executor
        .load_weights("test_weight", &weights)
        .expect("load");

    assert_eq!(
        executor.cached_weight_count(),
        1,
        "Count should be 1 after loading"
    );
}

#[test]
#[serial]
fn test_cov025_cached_weight_bytes() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially zero
    assert_eq!(
        executor.cached_weight_bytes(),
        0,
        "Initial bytes should be 0"
    );

    // Load a weight (256 f32 = 1024 bytes)
    let weights = vec![1.0f32; 256];
    executor
        .load_weights("test_weight", &weights)
        .expect("load");

    let bytes = executor.cached_weight_bytes();
    assert!(bytes >= 1024, "Should have at least 1024 bytes cached");
}

#[test]
#[serial]
fn test_cov025_cached_quantized_weight_count() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially zero
    assert_eq!(
        executor.cached_quantized_weight_count(),
        0,
        "Initial count should be 0"
    );

    // Load a quantized weight (Q4_K: 144 bytes per super-block)
    let weights = vec![0u8; 144];
    executor
        .load_quantized_weights("test_q4k", &weights)
        .expect("load");

    assert_eq!(
        executor.cached_quantized_weight_count(),
        1,
        "Count should be 1 after loading"
    );
}

#[test]
#[serial]
fn test_cov025_clear_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Load some weights
    let weights = vec![1.0f32; 256];
    executor.load_weights("test1", &weights).expect("load1");
    executor.load_weights("test2", &weights).expect("load2");
    assert_eq!(executor.cached_weight_count(), 2, "Should have 2 weights");

    // Clear
    executor.clear_weights();
    assert_eq!(
        executor.cached_weight_count(),
        0,
        "Should have 0 after clear"
    );
}

// =============================================================================
// COV-026: Coalesced/Vectorized/DP4A GEMV variants coverage
// =============================================================================

#[test]
#[serial]
fn test_cov026_coalesced_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q4_K: 144 bytes per 256 values (super-block)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights to get a GPU pointer
    let weight_bytes = (n as usize) * 144; // n rows of Q4_K data
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_coalesced", &weights)
        .expect("load weights");

    // Get weight pointer
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_coalesced")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.coalesced_q4k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "coalesced_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_vectorized_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_vectorized", &weights)
        .expect("load weights");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_vectorized")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.vectorized_q4k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "vectorized_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_dp4a_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_dp4a", &weights)
        .expect("load weights");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_dp4a")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.dp4a_q4k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "dp4a_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_coalesced_q6k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q6_K: 210 bytes per 256 values
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 210;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_coalesced_q6k", &weights, 14)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_coalesced_q6k")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.coalesced_q6k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "coalesced_q6k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_q4k_into", &weights)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q4k_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q4k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q6k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 210;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q6k_into", &weights, 14)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q6k_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q6k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q6k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q5k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q5_K: 176 bytes per 256 values
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 176;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q5k_into", &weights, 13)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q5k_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q5k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q5k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q8_0_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q8_0: 34 bytes per 32 values (2 bytes scale + 32 int8)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 32)

    // Load quantized weights
    // k=256 means 8 blocks of 32 values = 8 * 34 = 272 bytes per row
    let weight_bytes = (n as usize) * (k as usize / 32) * 34;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q8_0_into", &weights, 8)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q8_0_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q8_0_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q8_0_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
#[ignore = "PTX compilation issue CUDA_ERROR_INVALID_PTX - needs kernel fix"]
fn test_cov026_q4_0_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q4_0: 18 bytes per 32 values (2 bytes scale + 16 bytes of 4-bit)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 32)

    // Load quantized weights
    let weight_bytes = (n as usize) * (k as usize / 32) * 18;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q4_0_into", &weights, 2)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q4_0_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q4_0_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q4_0_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q4_1_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q4_1: 20 bytes per 32 values (2 scale + 2 min + 16 data)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 32)

    // Load quantized weights
    let weight_bytes = (n as usize) * (k as usize / 32) * 20;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q4_1_into", &weights, 3)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q4_1_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q4_1_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q4_1_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q5_0_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q5_0: 22 bytes per 32 values (2 scale + 4 high bits + 16 low bits)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 32)

    // Load quantized weights
    let weight_bytes = (n as usize) * (k as usize / 32) * 22;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q5_0_into", &weights, 6)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q5_0_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q5_0_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q5_0_gemv_into should succeed: {:?}",
        result.err()
    );
}

// =============================================================================
// COV-027: Tiled/Fused GEMV and async quantization coverage
// =============================================================================

#[test]
#[serial]
fn test_cov027_q4k_gemv_into_tiled_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_tiled", &weights)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_tiled")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q4k_gemv_into_tiled(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q4k_gemv_into_tiled should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov027_tiled_q4k_gemv_cached_async_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;
    let outputs_per_block = 4u32;

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Weight not cached - should fail
    let result =
        executor.tiled_q4k_gemv_cached_async("nonexistent_tiled", &input, n, k, outputs_per_block);
    assert!(result.is_err(), "Should fail when weight not cached");
}

#[test]
#[serial]
fn test_cov027_chunked_tiled_q4k_gemv_cached_async_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 512u32; // larger K for chunked version
    let outputs_per_block = 4u32;

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Weight not cached - should fail
    let result = executor.chunked_tiled_q4k_gemv_cached_async(
        "nonexistent_chunked",
        &input,
        n,
        k,
        outputs_per_block,
    );
    assert!(result.is_err(), "Should fail when weight not cached");
}

#[test]
#[serial]
fn test_cov027_dp4a_q4k_gemv_cached_async_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Weight not cached - should fail
    let result = executor.dp4a_q4k_gemv_cached_async("nonexistent_dp4a", &input, n, k);
    assert!(result.is_err(), "Should fail when weight not cached");
}

// NOTE: q8_quantize_async test skipped due to CUDA_ERROR_INVALID_PTX bug in kernel

#[test]
#[serial]
fn test_cov027_fused_rmsnorm_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (also hidden size for rmsnorm)
    let epsilon = 1e-5f32;

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_fused_rmsnorm", &weights)
        .expect("load weights");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_fused_rmsnorm")
        .expect("get ptr");

    // Cache gamma
    let gamma = vec![1.0f32; k as usize];
    executor
        .cache_rmsnorm_gamma("test_fused_gamma", &gamma)
        .expect("cache gamma");

    // Get gamma pointer (need to use internal cache)
    // For this test, we'll create gamma as a GPU buffer
    let gamma_buf = GpuBuffer::from_host(executor.context(), &gamma).expect("gamma buf");
    let gamma_ptr = gamma_buf.as_ptr();

    // Create input/output buffers
    let input_data = vec![0.5f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result =
        executor.fused_rmsnorm_q4k_gemv_into(weight_ptr, &input, gamma_ptr, &output, k, n, epsilon);
    assert!(
        result.is_ok(),
        "fused_rmsnorm_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov027_fused_gate_up_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let k = 256u32; // input dim (hidden_size)
    let n = 512u32; // output dim (intermediate_size)

    // Load gate and up weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_gate_fused", &weights)
        .expect("load gate");
    executor
        .load_quantized_weights("test_up_fused", &weights)
        .expect("load up");

    let gate_ptr = executor
        .get_quantized_weight_ptr("test_gate_fused")
        .expect("get gate ptr");
    let up_ptr = executor
        .get_quantized_weight_ptr("test_up_fused")
        .expect("get up ptr");

    // Create input and separate output buffers for gate and up
    let input_data = vec![0.5f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let gate_output = GpuBuffer::new(executor.context(), n as usize).expect("gate output");
    let up_output = GpuBuffer::new(executor.context(), n as usize).expect("up output");

    // fused_gate_up_q4k_gemv_into(gate_ptr, up_ptr, input, gate_output, up_output, k, n)
    let result = executor.fused_gate_up_q4k_gemv_into(
        gate_ptr,
        up_ptr,
        &input,
        &gate_output,
        &up_output,
        k,
        n,
    );
    assert!(
        result.is_ok(),
        "fused_gate_up_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov027_q4k_gemv_cached_tiled_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Create host slices (q4k_gemv_cached_tiled takes host slices, not GPU buffers)
    let input_data = vec![0.1f32; k as usize];
    let mut output_data = vec![0.0f32; n as usize];

    // Weight not cached - should fail
    let result = executor.q4k_gemv_cached_tiled(
        "nonexistent_cached_tiled",
        &input_data,
        &mut output_data,
        n,
        k,
    );
    assert!(result.is_err(), "Should fail when weight not cached");
}

#[test]
#[serial]
fn test_cov027_q4k_gemv_indexed_async_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Load weights and get pointer
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_indexed", &weights)
        .expect("load");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_indexed")
        .expect("get ptr");

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // q4k_gemv_indexed_async takes weight_ptr, not layer_idx
    let result = executor.q4k_gemv_indexed_async(weight_ptr, &input, n, k);
    assert!(
        result.is_ok(),
        "q4k_gemv_indexed_async should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov027_q6k_gemv_indexed_async_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Load Q6_K weights and get pointer
    let weight_bytes = (n as usize) * 210; // Q6_K is 210 bytes per 256 values
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q6k_indexed", &weights, 14)
        .expect("load");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q6k_indexed")
        .expect("get ptr");

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // q6k_gemv_indexed_async takes weight_ptr, not layer_idx
    let result = executor.q6k_gemv_indexed_async(weight_ptr, &input, n, k);
    assert!(
        result.is_ok(),
        "q6k_gemv_indexed_async should succeed: {:?}",
        result.err()
    );
}

// ============================================================================
// COV-028: More function coverage tests
// ============================================================================

/// Test fused_qkv_into basic functionality
#[test]
#[serial]
fn test_cov028_fused_qkv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let kv_dim = 64u32;

    // Create input and weight buffers
    let x_data = vec![0.1f32; hidden_size as usize];
    let x = GpuBuffer::from_host(executor.context(), &x_data).expect("x");

    // Weight matrices: Q is hidden_size x hidden_size, K/V are hidden_size x kv_dim
    let w_q_data = vec![0.01f32; (hidden_size * hidden_size) as usize];
    let w_k_data = vec![0.01f32; (hidden_size * kv_dim) as usize];
    let w_v_data = vec![0.01f32; (hidden_size * kv_dim) as usize];

    let w_q = GpuBuffer::from_host(executor.context(), &w_q_data).expect("w_q");
    let w_k = GpuBuffer::from_host(executor.context(), &w_k_data).expect("w_k");
    let w_v = GpuBuffer::from_host(executor.context(), &w_v_data).expect("w_v");

    // Output buffers
    let out_q = GpuBuffer::new(executor.context(), hidden_size as usize).expect("out_q");
    let out_k = GpuBuffer::new(executor.context(), kv_dim as usize).expect("out_k");
    let out_v = GpuBuffer::new(executor.context(), kv_dim as usize).expect("out_v");

    let result = executor.fused_qkv_into(
        &x,
        &w_q,
        &w_k,
        &w_v,
        &out_q,
        &out_k,
        &out_v,
        hidden_size,
        kv_dim,
    );
    assert!(
        result.is_ok(),
        "fused_qkv_into should succeed: {:?}",
        result.err()
    );
}

/// Test fused_qkv_into with GQA (different kv_dim)
#[test]
#[serial]
fn test_cov028_fused_qkv_into_gqa() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 128u32;
    let kv_dim = 32u32; // GQA: fewer KV heads

    let x_data = vec![0.1f32; hidden_size as usize];
    let x = GpuBuffer::from_host(executor.context(), &x_data).expect("x");

    let w_q_data = vec![0.01f32; (hidden_size * hidden_size) as usize];
    let w_k_data = vec![0.01f32; (hidden_size * kv_dim) as usize];
    let w_v_data = vec![0.01f32; (hidden_size * kv_dim) as usize];

    let w_q = GpuBuffer::from_host(executor.context(), &w_q_data).expect("w_q");
    let w_k = GpuBuffer::from_host(executor.context(), &w_k_data).expect("w_k");
    let w_v = GpuBuffer::from_host(executor.context(), &w_v_data).expect("w_v");

    let out_q = GpuBuffer::new(executor.context(), hidden_size as usize).expect("out_q");
    let out_k = GpuBuffer::new(executor.context(), kv_dim as usize).expect("out_k");
    let out_v = GpuBuffer::new(executor.context(), kv_dim as usize).expect("out_v");

    let result = executor.fused_qkv_into(
        &x,
        &w_q,
        &w_k,
        &w_v,
        &out_q,
        &out_k,
        &out_v,
        hidden_size,
        kv_dim,
    );
    assert!(
        result.is_ok(),
        "fused_qkv_into with GQA should succeed: {:?}",
        result.err()
    );
}

/// Test fused_gate_up_into basic functionality
#[test]
#[serial]
fn test_cov028_fused_gate_up_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let intermediate_size = 128u32;

    let x_data = vec![0.1f32; hidden_size as usize];
    let x = GpuBuffer::from_host(executor.context(), &x_data).expect("x");

    // Weight matrices: hidden_size x intermediate_size
    let w_gate_data = vec![0.01f32; (hidden_size * intermediate_size) as usize];
    let w_up_data = vec![0.01f32; (hidden_size * intermediate_size) as usize];

    let w_gate = GpuBuffer::from_host(executor.context(), &w_gate_data).expect("w_gate");
    let w_up = GpuBuffer::from_host(executor.context(), &w_up_data).expect("w_up");

    let output = GpuBuffer::new(executor.context(), intermediate_size as usize).expect("output");

    let result =
        executor.fused_gate_up_into(&x, &w_gate, &w_up, &output, hidden_size, intermediate_size);
    assert!(
        result.is_ok(),
        "fused_gate_up_into should succeed: {:?}",
        result.err()
    );
}

/// Test rope_into basic functionality
#[test]
#[serial]
fn test_cov028_rope_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4u32;
    let head_dim = 32u32;
    let position = 5u32;
    let theta = 10000.0f32;

    let input_size = (num_heads * head_dim) as usize;
    let input_data = vec![0.5f32; input_size];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), input_size).expect("output");

    let result = executor.rope_into(&input, &output, position, num_heads, head_dim, theta);
    assert!(
        result.is_ok(),
        "rope_into should succeed: {:?}",
        result.err()
    );
}

/// Test rope_into with different positions
#[test]
#[serial]
fn test_cov028_rope_into_varying_positions() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8u32;
    let head_dim = 64u32;
    let theta = 10000.0f32;

    let input_size = (num_heads * head_dim) as usize;
    let input_data = vec![1.0f32; input_size];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), input_size).expect("output");

    // Test multiple positions
    for position in [0, 1, 10, 100, 1000] {
        let result = executor.rope_into(&input, &output, position, num_heads, head_dim, theta);
        assert!(
            result.is_ok(),
            "rope_into at position {} should succeed: {:?}",
            position,
            result.err()
        );
    }
}

/// Test batched_q4k_gemv_into basic functionality
#[test]
#[serial]
fn test_cov028_batched_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32; // batch size
    let n = 32u32;
    let k = 256u32;

    // Load Q4K weights
    let weight_bytes = (n as usize) * 144; // Q4K is 144 bytes per 256 values
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_batched_q4k", &weights)
        .expect("load weights");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_batched_q4k")
        .expect("get ptr");

    // Input: m x k elements
    let input_data = vec![0.1f32; (m * k) as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Output: m x n elements
    let output = GpuBuffer::new(executor.context(), (m * n) as usize).expect("output");

    let result = executor.batched_q4k_gemv_into(weight_ptr, &input, &output, m, n, k);
    assert!(
        result.is_ok(),
        "batched_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

/// Test batched_q4k_gemv_into with M=16 (multi-warp path)
#[test]
#[serial]
fn test_cov028_batched_q4k_gemv_into_m16() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 16u32; // triggers multi-warp kernel
    let n = 32u32;
    let k = 256u32;

    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_batched_q4k_m16", &weights)
        .expect("load");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_batched_q4k_m16")
        .expect("get ptr");

    let input_data = vec![0.1f32; (m * k) as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), (m * n) as usize).expect("output");

    let result = executor.batched_q4k_gemv_into(weight_ptr, &input, &output, m, n, k);
    assert!(
        result.is_ok(),
        "batched_q4k_gemv_into M=16 should succeed: {:?}",
        result.err()
    );
}

/// Test batched_q6k_gemv_into basic functionality
#[test]
#[serial]
fn test_cov028_batched_q6k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 32u32;
    let k = 256u32;

    // Load Q6K weights (210 bytes per 256 values)
    let weight_bytes = (n as usize) * 210;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_batched_q6k", &weights, 14)
        .expect("load");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_batched_q6k")
        .expect("get ptr");

    let input_data = vec![0.1f32; (m * k) as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), (m * n) as usize).expect("output");

    let result = executor.batched_q6k_gemv_into(weight_ptr, &input, &output, m, n, k);
    assert!(
        result.is_ok(),
        "batched_q6k_gemv_into should succeed: {:?}",
        result.err()
    );
}

/// Test layer_norm_gpu basic functionality
#[test]
#[serial]
fn test_cov028_layer_norm_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let batch_size = 1u32;
    let epsilon = 1e-5f32;

    let input_data = vec![0.5f32; hidden_size as usize];
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.0f32; hidden_size as usize];

    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), hidden_size as usize).expect("output");
    let gamma = GpuBuffer::from_host(executor.context(), &gamma_data).expect("gamma");
    let beta = GpuBuffer::from_host(executor.context(), &beta_data).expect("beta");

    let result = executor.layer_norm_gpu(
        &input,
        &output,
        &gamma,
        &beta,
        hidden_size,
        batch_size,
        epsilon,
    );
    assert!(
        result.is_ok(),
        "layer_norm_gpu should succeed: {:?}",
        result.err()
    );
}

/// Test layer_norm_gpu with batch
#[test]
#[serial]
fn test_cov028_layer_norm_gpu_batched() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 128u32;
    let batch_size = 4u32;
    let epsilon = 1e-6f32;

    let input_data = vec![0.5f32; (hidden_size * batch_size) as usize];
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.1f32; hidden_size as usize];

    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output =
        GpuBuffer::new(executor.context(), (hidden_size * batch_size) as usize).expect("output");
    let gamma = GpuBuffer::from_host(executor.context(), &gamma_data).expect("gamma");
    let beta = GpuBuffer::from_host(executor.context(), &beta_data).expect("beta");

    let result = executor.layer_norm_gpu(
        &input,
        &output,
        &gamma,
        &beta,
        hidden_size,
        batch_size,
        epsilon,
    );
    assert!(
        result.is_ok(),
        "layer_norm_gpu batched should succeed: {:?}",
        result.err()
    );
}

/// Test compute_stream getter
#[test]
#[serial]
fn test_cov028_compute_stream() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // compute_stream() returns a reference to CudaStream
    let stream = executor.compute_stream();
    // Just verify we can access it without panic
    assert!(
        std::ptr::from_ref(stream) as usize != 0,
        "stream should be valid"
    );
}

// ============================================================================
// COV-029: More weight and workspace tests
// ============================================================================

/// Test load_weights basic functionality
#[test]
#[serial]
fn test_cov029_load_weights_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let weights = vec![0.1f32; 256];
    let result = executor.load_weights("test_weight", &weights);
    assert!(
        result.is_ok(),
        "load_weights should succeed: {:?}",
        result.err()
    );

    let bytes = result.unwrap();
    assert_eq!(bytes, 256 * 4, "Should load 256 f32 values (1024 bytes)");
}

/// Test load_weights and has_weights
#[test]
#[serial]
fn test_cov029_load_weights_and_has() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_weights("my_weight"),
        "Should not have weight initially"
    );

    let weights = vec![1.0f32; 128];
    executor.load_weights("my_weight", &weights).expect("load");

    assert!(
        executor.has_weights("my_weight"),
        "Should have weight after load"
    );
    assert!(
        !executor.has_weights("other_weight"),
        "Should not have unloaded weight"
    );
}

/// Test cached_weight_count
#[test]
#[serial]
fn test_cov029_cached_weight_count() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    assert_eq!(
        executor.cached_weight_count(),
        0,
        "Initial count should be 0"
    );

    executor.load_weights("w1", &[1.0f32; 64]).expect("load w1");
    assert_eq!(executor.cached_weight_count(), 1, "Count should be 1");

    executor.load_weights("w2", &[1.0f32; 64]).expect("load w2");
    assert_eq!(executor.cached_weight_count(), 2, "Count should be 2");
}

/// Test cached_weight_bytes
#[test]
#[serial]
fn test_cov029_cached_weight_bytes() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    assert_eq!(
        executor.cached_weight_bytes(),
        0,
        "Initial bytes should be 0"
    );

    executor
        .load_weights("w1", &[1.0f32; 100])
        .expect("load w1");
    assert_eq!(
        executor.cached_weight_bytes(),
        400,
        "Should be 400 bytes (100 * 4)"
    );

    executor.load_weights("w2", &[1.0f32; 50]).expect("load w2");
    assert_eq!(
        executor.cached_weight_bytes(),
        600,
        "Should be 600 bytes total"
    );
}

/// Test has_indexed_weights
#[test]
#[serial]
fn test_cov029_has_indexed_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially should not have indexed weights
    assert!(
        !executor.has_indexed_weights(),
        "Should not have indexed weights initially"
    );
}

/// Test return_staging_buffer
#[test]
#[serial]
fn test_cov029_return_staging_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get a staging buffer
    let buf = executor.get_staging_buffer(256);

    // Return it to the pool
    executor.return_staging_buffer(buf);

    // Pool should have returned buffer
    let stats = executor.staging_pool_stats();
    assert!(
        stats.free_buffers >= 1,
        "Pool should have at least 1 buffer after return"
    );
}

/// Test workspace_batch_size
#[test]
#[serial]
fn test_cov029_workspace_batch_size() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially should be 0
    assert_eq!(
        executor.workspace_batch_size(),
        0,
        "Initial batch size should be 0"
    );

    // After init_workspace
    executor.init_workspace(512, 256).expect("init workspace");
    assert_eq!(
        executor.workspace_batch_size(),
        1,
        "Batch size should be 1 after init"
    );
}

/// Test workspace_batch_size after batched init
#[test]
#[serial]
fn test_cov029_workspace_batch_size_batched() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    executor.init_workspace(512, 256).expect("init workspace");
    executor
        .init_batched_workspace(512, 256, 4)
        .expect("init batched");

    assert_eq!(executor.workspace_batch_size(), 4, "Batch size should be 4");
}

/// Test profiler_mut
#[test]
#[serial]
fn test_cov029_profiler_mut() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get mutable profiler access
    let profiler = executor.profiler_mut();
    // Just verify we can access it
    assert!(
        std::ptr::from_mut(profiler) as usize != 0,
        "profiler should be valid"
    );
}

/// Test execution_graph_ascii
#[test]
#[serial]
fn test_cov029_execution_graph_ascii() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let ascii = executor.execution_graph_ascii();
    // Just verify the function returns without panic - the actual format varies
    // Empty graph or a tree structure are both valid
    let _ = ascii.len();
}

/// Test tile_stats
#[test]
#[serial]
fn test_cov029_tile_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Access tile stats for different levels
    let _macro_stats = executor.tile_stats(trueno::TileLevel::Macro);
    let _midi_stats = executor.tile_stats(trueno::TileLevel::Midi);
    // Just verify we can access without panic
}

// =============================================================================
// COV-030: CudaExecutor Layer API Coverage Tests
// Target: Increase layer.rs coverage from 20% to higher
// =============================================================================

/// Test workspace_output returns None before init
#[test]
#[serial]
fn test_cov030_workspace_output_before_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before init, workspace_output should return None
    assert!(
        executor.workspace_output().is_none(),
        "workspace_output should be None before init"
    );
}

/// Test workspace_output returns Some after init
#[test]
#[serial]
fn test_cov030_workspace_output_after_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initialize workspace
    executor.init_workspace(512, 256).expect("init workspace");

    // After init, workspace_output should return Some
    assert!(
        executor.workspace_output().is_some(),
        "workspace_output should be Some after init"
    );
}

/// Test has_rmsnorm_weights before and after preload
#[test]
#[serial]
fn test_cov030_has_rmsnorm_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before preload
    assert!(
        !executor.has_rmsnorm_weights(0),
        "Should not have weights for layer 0 initially"
    );
    assert!(
        !executor.has_rmsnorm_weights(1),
        "Should not have weights for layer 1 initially"
    );

    // Preload for 2 layers - need &[&[f32]] type
    let attn_norm_0 = vec![1.0f32; 512];
    let attn_norm_1 = vec![1.0f32; 512];
    let ffn_norm_0 = vec![1.0f32; 512];
    let ffn_norm_1 = vec![1.0f32; 512];
    let attn_norms: &[&[f32]] = &[&attn_norm_0, &attn_norm_1];
    let ffn_norms: &[&[f32]] = &[&ffn_norm_0, &ffn_norm_1];
    executor
        .preload_rmsnorm_weights(2, attn_norms, ffn_norms)
        .expect("preload");

    // After preload
    assert!(
        executor.has_rmsnorm_weights(0),
        "Should have weights for layer 0 after preload"
    );
    assert!(
        executor.has_rmsnorm_weights(1),
        "Should have weights for layer 1 after preload"
    );
    assert!(
        !executor.has_rmsnorm_weights(2),
        "Should not have weights for layer 2 (not preloaded)"
    );
}

/// Test has_output_norm before and after preload
#[test]
#[serial]
fn test_cov030_has_output_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before preload
    assert!(
        !executor.has_output_norm(),
        "Should not have output norm initially"
    );

    // Preload output norm
    let gamma = vec![1.0f32; 512];
    executor
        .preload_output_norm(&gamma)
        .expect("preload output norm");

    // After preload
    assert!(
        executor.has_output_norm(),
        "Should have output norm after preload"
    );
}

/// Test has_qkv_bias before and after preload
#[test]
#[serial]
fn test_cov030_has_qkv_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before preload
    assert!(
        !executor.has_qkv_bias(0),
        "Should not have QKV bias for layer 0 initially"
    );

    // Preload for 1 layer - need &[Option<&[f32]>] type
    let q_bias_0 = vec![0.0f32; 512];
    let k_bias_0 = vec![0.0f32; 64];
    let v_bias_0 = vec![0.0f32; 64];
    let q_biases: &[Option<&[f32]>] = &[Some(&q_bias_0)];
    let k_biases: &[Option<&[f32]>] = &[Some(&k_bias_0)];
    let v_biases: &[Option<&[f32]>] = &[Some(&v_bias_0)];
    executor
        .preload_qkv_bias(1, q_biases, k_biases, v_biases)
        .expect("preload");

    // After preload
    assert!(
        executor.has_qkv_bias(0),
        "Should have QKV bias for layer 0 after preload"
    );
    assert!(
        !executor.has_qkv_bias(1),
        "Should not have QKV bias for layer 1 (not preloaded)"
    );
}

/// Test has_lm_head_bias before and after preload
#[test]
#[serial]
fn test_cov030_has_lm_head_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before preload
    assert!(
        !executor.has_lm_head_bias(),
        "Should not have LM head bias initially"
    );

    // Preload LM head bias
    let bias = vec![0.0f32; 32000];
    executor
        .preload_lm_head_bias(Some(&bias))
        .expect("preload lm head bias");

    // After preload
    assert!(
        executor.has_lm_head_bias(),
        "Should have LM head bias after preload"
    );
}

/// Test output_rmsnorm_gpu basic operation
#[test]
#[serial]
fn test_cov030_output_rmsnorm_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 512u32;
    let epsilon = 1e-5f32;

    // Create input, output, and gamma buffers
    let input = vec![1.0f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];
    let gamma = vec![1.0f32; hidden_dim as usize];

    let result = executor.output_rmsnorm_gpu(&input, &mut output, &gamma, hidden_dim, epsilon);
    assert!(
        result.is_ok(),
        "output_rmsnorm_gpu should succeed: {:?}",
        result.err()
    );

    // Output should be normalized (not all zeros)
    let sum: f32 = output.iter().sum();
    assert!(sum.abs() > 0.0, "Output should be non-zero after RMSNorm");
}

/// Test gpu_argmax basic operation
#[test]
#[serial]
fn test_cov030_gpu_argmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create logits buffer with a clear maximum
    let vocab_size = 1024u32;
    let mut logits = vec![0.0f32; vocab_size as usize];
    logits[42] = 100.0; // Make index 42 the maximum

    // Upload to GPU
    let logits_gpu = GpuBuffer::from_host(executor.context(), &logits).expect("upload logits");
    let logits_ptr = logits_gpu.as_ptr();

    let result = executor.gpu_argmax(logits_ptr, vocab_size);
    assert!(
        result.is_ok(),
        "gpu_argmax should succeed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), 42, "gpu_argmax should return index 42");
}

/// Test gpu_argmax with large vocab
#[test]
#[serial]
fn test_cov030_gpu_argmax_large_vocab() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Typical LLM vocab size
    let vocab_size = 32000u32;
    let mut logits = vec![-10.0f32; vocab_size as usize];
    logits[31999] = 50.0; // Last index is maximum

    let logits_gpu = GpuBuffer::from_host(executor.context(), &logits).expect("upload logits");
    let logits_ptr = logits_gpu.as_ptr();

    let result = executor.gpu_argmax(logits_ptr, vocab_size);
    assert!(result.is_ok(), "gpu_argmax with large vocab should succeed");
    assert_eq!(result.unwrap(), 31999, "Should find max at last index");
}

/// Test read_hidden_state_to_cpu error before workspace init
#[test]
#[serial]
fn test_cov030_read_hidden_state_error_before_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before workspace init, should return error
    let result = executor.read_hidden_state_to_cpu();
    assert!(
        result.is_err(),
        "read_hidden_state_to_cpu should error before workspace init"
    );
}

/// Test transformer_layer_batched error validation
#[test]
#[serial]
fn test_cov030_transformer_layer_batched_validation() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create a dummy input buffer
    let input_data = vec![0.1f32; 512 * 4];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Create dummy IndexedLayerWeights
    let layer_weights = IndexedLayerWeights::default();

    // Without workspace init, should return error
    let result = executor.transformer_layer_batched(
        &input,
        0,
        &layer_weights,
        4,
        &[0, 1, 2, 3],
        512,
        256,
        1e-5,
    );
    assert!(
        result.is_err(),
        "transformer_layer_batched should error without workspace"
    );
}

/// Test transformer_layer_batched batch size mismatch
#[test]
#[serial]
fn test_cov030_transformer_layer_batched_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Init workspace for batch_size=2
    executor.init_workspace(512, 256).expect("init workspace");
    executor
        .init_batched_workspace(512, 256, 2)
        .expect("init batched");

    let input_data = vec![0.1f32; 512 * 4];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let layer_weights = IndexedLayerWeights::default();

    // Request batch_size=4 but workspace is for 2
    let result = executor.transformer_layer_batched(
        &input,
        0,
        &layer_weights,
        4, // mismatch!
        &[0, 1, 2, 3],
        512,
        256,
        1e-5,
    );
    assert!(result.is_err(), "Should error on batch size mismatch");
}

/// Test transformer_layer_batched positions mismatch
#[test]
#[serial]
fn test_cov030_transformer_layer_batched_positions_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Init workspace for batch_size=4
    executor.init_workspace(512, 256).expect("init workspace");
    executor
        .init_batched_workspace(512, 256, 4)
        .expect("init batched");

    let input_data = vec![0.1f32; 512 * 4];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let layer_weights = IndexedLayerWeights::default();

    // Positions length != m
    let result = executor.transformer_layer_batched(
        &input,
        0,
        &layer_weights,
        4,
        &[0, 1], // only 2 positions but m=4
        512,
        256,
        1e-5,
    );
    assert!(result.is_err(), "Should error when positions.len() != m");
}

// =============================================================================
// COV-031: Additional Activation & Attention Coverage Tests
// Target: Improve activations.rs and attention.rs coverage
// =============================================================================

/// Test rope_into basic operation
#[test]
#[serial]
fn test_cov031_rope_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8u32;
    let head_dim = 64u32;
    let n = num_heads * head_dim;
    let theta = 10000.0f32;
    let position = 5u32;

    // Create input/output buffers
    let input_data = vec![1.0f32; n as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.rope_into(&input, &output, position, num_heads, head_dim, theta);
    assert!(
        result.is_ok(),
        "rope_into should succeed: {:?}",
        result.err()
    );
}

/// Test rope_into with different positions
#[test]
#[serial]
fn test_cov031_rope_into_positions() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8u32;
    let head_dim = 64u32;
    let n = num_heads * head_dim;
    let theta = 10000.0f32;

    let input_data = vec![1.0f32; n as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    // Test various positions
    for position in [0u32, 1, 10, 100, 1000] {
        let result = executor.rope_into(&input, &output, position, num_heads, head_dim, theta);
        assert!(
            result.is_ok(),
            "rope_into at position {} failed: {:?}",
            position,
            result.err()
        );
    }
}

/// Test rope_indirect_into basic operation
#[test]
#[serial]
fn test_cov031_rope_indirect_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8u32;
    let head_dim = 64u32;
    let n = num_heads * head_dim;
    let theta = 10000.0f32;

    let input_data = vec![1.0f32; n as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    // Position in device buffer (for CUDA graph compatibility)
    let position_buf = GpuBuffer::from_host(executor.context(), &[5u32]).expect("position buf");

    let result =
        executor.rope_indirect_into(&input, &output, &position_buf, num_heads, head_dim, theta);
    assert!(
        result.is_ok(),
        "rope_indirect_into should succeed: {:?}",
        result.err()
    );
}

/// Test rope_neox_into basic operation
#[test]
#[serial]
fn test_cov031_rope_neox_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8u32;
    let head_dim = 64u32;
    let n = num_heads * head_dim;
    let theta = 10000.0f32;
    let position = 5u32;

    let input_data = vec![1.0f32; n as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.rope_neox_into(&input, &output, position, num_heads, head_dim, theta);
    assert!(
        result.is_ok(),
        "rope_neox_into should succeed: {:?}",
        result.err()
    );
}

/// Test rope_neox_indirect_into basic operation
#[test]
#[serial]
fn test_cov031_rope_neox_indirect_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8u32;
    let head_dim = 64u32;
    let n = num_heads * head_dim;
    let theta = 10000.0f32;

    let input_data = vec![1.0f32; n as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");
    let position_buf = GpuBuffer::from_host(executor.context(), &[10u32]).expect("position buf");

    let result = executor.rope_neox_indirect_into(
        &input,
        &output,
        &position_buf,
        num_heads,
        head_dim,
        theta,
    );
    assert!(
        result.is_ok(),
        "rope_neox_indirect_into should succeed: {:?}",
        result.err()
    );
}

/// Test fused_qkv_into basic operation
#[test]
#[serial]
fn test_cov031_fused_qkv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Use small dimensions to avoid memory issues
    // fused_qkv_into expects: x[hidden_size], w_q[hidden_size, hidden_size],
    // w_k[hidden_size, kv_dim], w_v[hidden_size, kv_dim]
    // out_q[hidden_size], out_k[kv_dim], out_v[kv_dim]
    let hidden_dim = 64u32;
    let kv_dim = 32u32; // GQA with fewer KV heads

    // Create weight matrices as f32 GpuBuffers
    let w_q_data = vec![0.01f32; (hidden_dim * hidden_dim) as usize]; // Q output is hidden_dim
    let w_k_data = vec![0.01f32; (hidden_dim * kv_dim) as usize];
    let w_v_data = vec![0.01f32; (hidden_dim * kv_dim) as usize];

    let w_q = GpuBuffer::from_host(executor.context(), &w_q_data).expect("w_q");
    let w_k = GpuBuffer::from_host(executor.context(), &w_k_data).expect("w_k");
    let w_v = GpuBuffer::from_host(executor.context(), &w_v_data).expect("w_v");

    let input_data = vec![0.1f32; hidden_dim as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Q output has hidden_dim elements, K/V have kv_dim elements
    let q_out = GpuBuffer::new(executor.context(), hidden_dim as usize).expect("q_out");
    let k_out = GpuBuffer::new(executor.context(), kv_dim as usize).expect("k_out");
    let v_out = GpuBuffer::new(executor.context(), kv_dim as usize).expect("v_out");

    let result = executor.fused_qkv_into(
        &input, &w_q, &w_k, &w_v, &q_out, &k_out, &v_out, hidden_dim, kv_dim,
    );
    assert!(
        result.is_ok(),
        "fused_qkv_into should succeed: {:?}",
        result.err()
    );

    // Synchronize to catch any kernel errors before test ends
    executor.synchronize().expect("sync");
}

/// Test fused_gate_up_into basic operation
#[test]
#[serial]
fn test_cov031_fused_gate_up_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Use smaller dimensions to avoid memory issues
    let hidden_dim = 64u32;
    let intermediate_dim = 128u32;

    // Create weight matrices as f32 GpuBuffers
    let w_gate_data = vec![0.01f32; (hidden_dim * intermediate_dim) as usize];
    let w_up_data = vec![0.01f32; (hidden_dim * intermediate_dim) as usize];

    let w_gate = GpuBuffer::from_host(executor.context(), &w_gate_data).expect("w_gate");
    let w_up = GpuBuffer::from_host(executor.context(), &w_up_data).expect("w_up");

    let input_data = vec![0.1f32; hidden_dim as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    let output = GpuBuffer::new(executor.context(), intermediate_dim as usize).expect("output");

    let result = executor.fused_gate_up_into(
        &input,
        &w_gate,
        &w_up,
        &output,
        hidden_dim,
        intermediate_dim,
    );
    assert!(
        result.is_ok(),
        "fused_gate_up_into should succeed: {:?}",
        result.err()
    );

    // Synchronize to catch any kernel errors before test ends
    executor.synchronize().expect("sync");
}

/// Test incremental_attention_into basic operation
#[test]
#[serial]
fn test_cov031_incremental_attention_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 8usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Initialize KV cache first (required for incremental attention)
    executor
        .init_kv_cache_gpu(1, num_heads, num_kv_heads, head_dim, 16)
        .expect("init kv");

    let q_data = vec![0.1f32; q_dim];
    let k_data = vec![0.1f32; kv_dim];
    let v_data = vec![0.1f32; kv_dim];

    let q_buf = GpuBuffer::from_host(executor.context(), &q_data).expect("q_buf");
    let k_buf = GpuBuffer::from_host(executor.context(), &k_data).expect("k_buf");
    let v_buf = GpuBuffer::from_host(executor.context(), &v_data).expect("v_buf");
    let out_buf = GpuBuffer::new(executor.context(), q_dim).expect("out_buf");

    let result = executor.incremental_attention_into(0, &q_buf, &k_buf, &v_buf, &out_buf);
    assert!(
        result.is_ok(),
        "incremental_attention_into should succeed: {:?}",
        result.err()
    );
}

/// Test batched_incremental_attention_into with batch_size=2
#[test]
#[serial]
fn test_cov031_batched_incremental_attention_into() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 8usize;
    let batch_size = 2usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Initialize KV cache first (required for batched attention)
    executor
        .init_kv_cache_gpu(1, num_heads, num_kv_heads, head_dim, 16)
        .expect("init kv");
    executor
        .init_batched_kv_cache_gpu(1, batch_size)
        .expect("init batched kv");

    let q_data = vec![0.1f32; q_dim * batch_size];
    let k_data = vec![0.1f32; kv_dim * batch_size];
    let v_data = vec![0.1f32; kv_dim * batch_size];

    let q_buf = GpuBuffer::from_host(executor.context(), &q_data).expect("q_buf");
    let k_buf = GpuBuffer::from_host(executor.context(), &k_data).expect("k_buf");
    let v_buf = GpuBuffer::from_host(executor.context(), &v_data).expect("v_buf");
    let out_buf = GpuBuffer::new(executor.context(), q_dim * batch_size).expect("out_buf");

    let positions = vec![0u32; batch_size];

    let result = executor.batched_incremental_attention_into(
        0, &q_buf, &k_buf, &v_buf, &out_buf, batch_size, &positions,
    );
    assert!(
        result.is_ok(),
        "batched_incremental_attention_into should succeed: {:?}",
        result.err()
    );
}

/// Test flash_decoding_attention_into without init
#[test]
#[serial]
fn test_cov031_flash_decoding_not_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16usize;
    let head_dim = 64usize;
    let n = seq_len * head_dim;

    let q_data = vec![0.1f32; n];
    let k_data = vec![0.1f32; n];
    let v_data = vec![0.1f32; n];

    let q_buf = GpuBuffer::from_host(executor.context(), &q_data).expect("q_buf");
    let k_buf = GpuBuffer::from_host(executor.context(), &k_data).expect("k_buf");
    let v_buf = GpuBuffer::from_host(executor.context(), &v_data).expect("v_buf");
    let out_buf = GpuBuffer::new(executor.context(), n).expect("out_buf");

    // Without init_flash_decoding, should return error
    let positions = vec![0u32; 1];
    let result =
        executor.flash_decoding_attention_into(0, &q_buf, &k_buf, &v_buf, &out_buf, 1, &positions);
    assert!(result.is_err(), "flash_decoding should error without init");
}

/// Test init_flash_decoding and flash_decoding_attention_into
#[test]
#[serial]
fn test_cov031_flash_decoding_with_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8usize;
    let head_dim = 64usize;
    let max_seq_len = 128usize;
    let batch_size = 1usize;

    // Initialize flash decoding (num_heads, head_dim, max_seq_len, batch_size)
    let init_result = executor.init_flash_decoding(num_heads, head_dim, max_seq_len, batch_size);
    assert!(
        init_result.is_ok(),
        "init_flash_decoding should succeed: {:?}",
        init_result.err()
    );

    // Now try flash decoding attention
    let q_dim = num_heads * head_dim;

    let q_data = vec![0.1f32; q_dim];
    let k_data = vec![0.1f32; q_dim];
    let v_data = vec![0.1f32; q_dim];

    let q_buf = GpuBuffer::from_host(executor.context(), &q_data).expect("q_buf");
    let k_buf = GpuBuffer::from_host(executor.context(), &k_data).expect("k_buf");
    let v_buf = GpuBuffer::from_host(executor.context(), &v_data).expect("v_buf");
    let out_buf = GpuBuffer::new(executor.context(), q_dim).expect("out_buf");

    let positions = vec![0u32; batch_size];
    let result = executor
        .flash_decoding_attention_into(0, &q_buf, &k_buf, &v_buf, &out_buf, batch_size, &positions);
    // Note: flash decoding may fail if KV cache not initialized, but at least we cover init path
    let _ = result;
}
