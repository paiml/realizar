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
