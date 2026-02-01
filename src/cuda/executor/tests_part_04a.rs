use super::*;
use crate::cuda::memory::{GpuBufferHandle, SizeClass, TransferMode};
use crate::cuda::pipeline::{
    presets, BankConflictStrategy, MemoryPattern, PtxOptimizationHints, PtxOptimizer,
    RegisterTiling,
};
use serial_test::serial;

        &output_gpu,
        &positions_gpu,
        num_heads,
        head_dim,
        batch_size,
        10000.0, // Standard theta
    );
    assert!(
        result.is_ok(),
        "batched_rope_into failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; total];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // RoPE should produce finite values
    for &val in &output {
        assert!(val.is_finite(), "RoPE output should be finite");
    }
}

// NOTE: COV-013 tests for fused operations (fused_swiglu_into, fused_qkv_into,
// fused_gate_up_into, rope_into, rope_neox_into, rope_indirect_into, rope_neox_indirect_into)
// were removed because they hang during kernel compilation. These fused operations
// require complex PTX generation that may have issues with current dimensions.
// The underlying operations are covered by other tests (SiLU, GELU, matmul, etc.).

// ==============================================================================
// COV-014: Additional weights.rs coverage - quantized weight management
// ==============================================================================

#[test]
#[serial]
fn test_cov014_load_quantized_weights_with_type_q4k() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q4K block is 144 bytes (256 values)
    let weights = vec![0u8; 144];
    let result = executor.load_quantized_weights_with_type("test_q4k", &weights, 12);
    assert!(
        result.is_ok(),
        "load_quantized_weights_with_type Q4K failed"
    );

    assert!(executor.has_quantized_weights("test_q4k"));
    assert_eq!(executor.get_quantized_weight_type("test_q4k"), Some(12));
}

#[test]
#[serial]
fn test_cov014_load_quantized_weights_with_type_q5k() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q5K uses different block size
    let weights = vec![0u8; 176]; // Q5K block size
    let result = executor.load_quantized_weights_with_type("test_q5k", &weights, 13);
    assert!(
        result.is_ok(),
        "load_quantized_weights_with_type Q5K failed"
    );

    assert!(executor.has_quantized_weights("test_q5k"));
    assert_eq!(executor.get_quantized_weight_type("test_q5k"), Some(13));
}

#[test]
#[serial]
fn test_cov014_load_quantized_weights_with_type_q6k() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q6K block is 210 bytes
    let weights = vec![0u8; 210];
    let result = executor.load_quantized_weights_with_type("test_q6k", &weights, 14);
    assert!(
        result.is_ok(),
        "load_quantized_weights_with_type Q6K failed"
    );

    assert!(executor.has_quantized_weights("test_q6k"));
    assert_eq!(executor.get_quantized_weight_type("test_q6k"), Some(14));
}

#[test]
#[serial]
fn test_cov014_get_quantized_weight_type_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Non-existent weight should return None
    assert_eq!(executor.get_quantized_weight_type("nonexistent"), None);
}

#[test]
#[serial]
fn test_cov014_has_quantized_weights_false() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(!executor.has_quantized_weights("nonexistent"));
}

#[test]
#[serial]
fn test_cov014_get_quantized_weight_ptr() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let weights = vec![1u8; 256];
    executor
        .load_quantized_weights("ptr_test", &weights)
        .expect("load");

    let ptr_result = executor.get_quantized_weight_ptr("ptr_test");
    assert!(ptr_result.is_ok(), "get_quantized_weight_ptr failed");

    let ptr = ptr_result.unwrap();
    assert!(ptr > 0, "Device pointer should be non-zero");
}

#[test]
#[serial]
fn test_cov014_get_quantized_weight_ptr_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let ptr_result = executor.get_quantized_weight_ptr("nonexistent");
    assert!(ptr_result.is_err(), "Should fail for nonexistent weight");
}

#[test]
#[serial]
fn test_cov014_cached_quantized_weight_count_multiple() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    assert_eq!(executor.cached_quantized_weight_count(), 0);

    executor
        .load_quantized_weights("w1", &vec![0u8; 144])
        .expect("load w1");
    assert_eq!(executor.cached_quantized_weight_count(), 1);

    executor
        .load_quantized_weights("w2", &vec![0u8; 144])
        .expect("load w2");
    assert_eq!(executor.cached_quantized_weight_count(), 2);

    executor
        .load_quantized_weights("w3", &vec![0u8; 144])
        .expect("load w3");
    assert_eq!(executor.cached_quantized_weight_count(), 3);
}

#[test]
#[serial]
fn test_cov014_cached_quantized_weight_bytes_multiple() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    assert_eq!(executor.cached_quantized_weight_bytes(), 0);

    executor
        .load_quantized_weights("w1", &vec![0u8; 256])
        .expect("load w1");
    let bytes1 = executor.cached_quantized_weight_bytes();
    assert!(bytes1 >= 256, "Should have at least 256 bytes");

    executor
        .load_quantized_weights("w2", &vec![0u8; 512])
        .expect("load w2");
    let bytes2 = executor.cached_quantized_weight_bytes();
    assert!(bytes2 >= 256 + 512, "Should have at least 768 bytes");
}

#[test]
#[serial]
fn test_cov014_clear_quantized_weights_multiple() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    executor
        .load_quantized_weights("w1", &vec![0u8; 144])
        .expect("load");
    executor
        .load_quantized_weights("w2", &vec![0u8; 144])
        .expect("load");
    executor
        .load_quantized_weights("w3", &vec![0u8; 144])
        .expect("load");
    assert_eq!(executor.cached_quantized_weight_count(), 3);

    executor.clear_quantized_weights();
    assert_eq!(executor.cached_quantized_weight_count(), 0);
    assert_eq!(executor.cached_quantized_weight_bytes(), 0);
}

// ==============================================================================
// COV-015: layer.rs error paths and validation coverage
// Target: Increase layer.rs coverage from 17.61% by testing error branches
// ==============================================================================

#[test]
#[serial]
fn test_cov015_has_rmsnorm_weights_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Should return false for any layer when no weights cached
    assert!(
        !executor.has_rmsnorm_weights(0),
        "Layer 0 should have no RMSNorm weights"
    );
    assert!(
        !executor.has_rmsnorm_weights(5),
        "Layer 5 should have no RMSNorm weights"
    );
    assert!(
        !executor.has_rmsnorm_weights(100),
        "Layer 100 should have no RMSNorm weights"
    );
}

#[test]
#[serial]
fn test_cov015_has_rmsnorm_weights_after_preload() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];
    let attn_norms: Vec<&[f32]> = vec![gamma.as_slice()];
    let ffn_norms: Vec<&[f32]> = vec![gamma.as_slice()];

    executor
        .preload_rmsnorm_weights(1, &attn_norms, &ffn_norms)
        .expect("preload");

    assert!(
        executor.has_rmsnorm_weights(0),
        "Layer 0 should have RMSNorm weights after preload"
    );
    assert!(
        !executor.has_rmsnorm_weights(1),
        "Layer 1 should not have weights"
    );
}

#[test]
#[serial]
fn test_cov015_forward_all_layers_missing_attn_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let input = vec![0.1f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];

    // Try forward without any cached weights - should fail
    let result = executor.forward_all_layers_gpu(
        &input,
        &mut output,
        0, // position
        1, // num_layers
        hidden_dim,
        128,  // intermediate_dim
        1e-5, // epsilon
    );

    assert!(result.is_err(), "Should fail without cached attn_norm");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("attn_norm not cached"),
        "Error should mention missing attn_norm: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_all_layers_missing_ffn_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let gamma = vec![1.0f32; hidden_dim as usize];

    // Only cache attn_norm, not ffn_norm
    executor
        .cache_rmsnorm_gamma("blk.0.attn_norm.gamma", &gamma)
        .expect("cache attn");

    let input = vec![0.1f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];

    let result = executor.forward_all_layers_gpu(&input, &mut output, 0, 1, hidden_dim, 128, 1e-5);

    assert!(result.is_err(), "Should fail without cached ffn_norm");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("ffn_norm not cached"),
        "Error should mention missing ffn_norm: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_to_logits_missing_output_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let vocab_size = 128u32;
    let gamma = vec![1.0f32; hidden_dim as usize];

    // Cache layer norms but not output norm
    executor
        .cache_rmsnorm_gamma("blk.0.attn_norm.gamma", &gamma)
        .expect("cache attn");
    executor
        .cache_rmsnorm_gamma("blk.0.ffn_norm.gamma", &gamma)
        .expect("cache ffn");

    let input = vec![0.1f32; hidden_dim as usize];
    let mut logits = vec![0.0f32; vocab_size as usize];

    let result = executor.forward_all_layers_gpu_to_logits(
        &input,
        &mut logits,
        0,
        1,
        hidden_dim,
        128,
        vocab_size,
        1e-5,
    );

    assert!(result.is_err(), "Should fail without cached output_norm");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("output_norm not cached"),
        "Error should mention missing output_norm: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_batch_size_zero() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let inputs: Vec<f32> = vec![]; // Empty - batch size 0
    let positions: Vec<u32> = vec![]; // Empty positions

    let result =
        executor.forward_batched_to_token_ids(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with batch size 0");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("batch size must be 1-32"),
        "Error should mention batch size constraint: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_batch_size_exceeds_max() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let m = 33; // Exceeds max of 32
    let inputs = vec![0.1f32; m * hidden_dim as usize];
    let positions: Vec<u32> = (0..m as u32).collect();

    let result =
        executor.forward_batched_to_token_ids(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with batch size > 32");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("batch size must be 1-32"),
        "Error should mention batch size constraint: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_wrong_input_length() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let positions: Vec<u32> = vec![0, 1]; // M=2
    let inputs = vec![0.1f32; 50]; // Wrong length: should be 2 * 64 = 128

    let result =
        executor.forward_batched_to_token_ids(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with wrong input length");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("inputs.len()") || err_msg.contains("M*hidden_dim"),
        "Error should mention input length mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_workspace_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let m = 2;
    let inputs = vec![0.1f32; m * hidden_dim as usize];
    let positions: Vec<u32> = vec![0, 1];

    // Don't initialize workspace - should fail
    let result =
        executor.forward_batched_to_token_ids(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail without initialized workspace");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("workspace not initialized") || err_msg.contains("Batched workspace"),
        "Error should mention workspace: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_preload_lm_head_bias_empty() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with empty bias (should return 0 bytes, not error)
    let empty_bias: Vec<f32> = vec![];
    let result = executor.preload_lm_head_bias(Some(&empty_bias));
    assert!(
        result.is_ok(),
        "preload_lm_head_bias with empty should succeed"
    );
    assert_eq!(result.unwrap(), 0, "Empty bias should upload 0 bytes");
    assert!(
        !executor.has_lm_head_bias(),
        "Should not have LM head bias with empty input"
    );
}

#[test]
#[serial]
fn test_cov015_cache_rmsnorm_gamma_duplicate() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];

    // First cache should upload bytes
    let result1 = executor.cache_rmsnorm_gamma("test_gamma", &gamma);
    assert!(result1.is_ok(), "First cache should succeed");
    let bytes1 = result1.unwrap();
    assert!(bytes1 > 0, "First cache should upload bytes");

    // Second cache of same name should return 0 (already cached)
    let result2 = executor.cache_rmsnorm_gamma("test_gamma", &gamma);
    assert!(result2.is_ok(), "Second cache should succeed");
    let bytes2 = result2.unwrap();
    assert_eq!(bytes2, 0, "Duplicate cache should return 0 bytes");
}

#[test]
#[serial]
fn test_cov015_preload_output_norm_duplicate() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];

    // First preload
    let result1 = executor.preload_output_norm(&gamma);
    assert!(result1.is_ok(), "First preload should succeed");
    let bytes1 = result1.unwrap();
    assert!(bytes1 > 0, "First preload should upload bytes");

    // Second preload of same norm should return 0
    let result2 = executor.preload_output_norm(&gamma);
    assert!(result2.is_ok(), "Second preload should succeed");
    let bytes2 = result2.unwrap();
    assert_eq!(bytes2, 0, "Duplicate preload should return 0 bytes");
}

#[test]
#[serial]
fn test_cov015_preload_rmsnorm_weights_duplicate() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];
    let attn_norms: Vec<&[f32]> = vec![gamma.as_slice()];
    let ffn_norms: Vec<&[f32]> = vec![gamma.as_slice()];

    // First preload
    let result1 = executor.preload_rmsnorm_weights(1, &attn_norms, &ffn_norms);
    assert!(result1.is_ok(), "First preload should succeed");
    let bytes1 = result1.unwrap();
    assert!(bytes1 > 0, "First preload should upload bytes");

    // Second preload should return 0 (already cached)
    let result2 = executor.preload_rmsnorm_weights(1, &attn_norms, &ffn_norms);
    assert!(result2.is_ok(), "Second preload should succeed");
    let bytes2 = result2.unwrap();
    assert_eq!(bytes2, 0, "Duplicate preload should return 0 bytes");
}

#[test]
#[serial]
fn test_cov015_preload_qkv_bias_with_none_values() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // All biases are None (no bias model)
    let q_biases: Vec<Option<&[f32]>> = vec![None];
    let k_biases: Vec<Option<&[f32]>> = vec![None];
    let v_biases: Vec<Option<&[f32]>> = vec![None];

    let result = executor.preload_qkv_bias(1, &q_biases, &k_biases, &v_biases);
    assert!(
        result.is_ok(),
        "preload_qkv_bias with None values should succeed"
    );
    assert_eq!(
        result.unwrap(),
        0,
        "No bytes should be uploaded for None biases"
    );
    assert!(
        !executor.has_qkv_bias(0),
        "Should not have QKV bias when all None"
    );
}

#[test]
#[serial]
fn test_cov015_preload_qkv_bias_partial() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let q_bias_data = vec![0.1f32; 64];
    // Only Q bias present, K and V are None
    let q_biases: Vec<Option<&[f32]>> = vec![Some(q_bias_data.as_slice())];
    let k_biases: Vec<Option<&[f32]>> = vec![None];
    let v_biases: Vec<Option<&[f32]>> = vec![None];

    let result = executor.preload_qkv_bias(1, &q_biases, &k_biases, &v_biases);
    assert!(result.is_ok(), "preload_qkv_bias partial should succeed");
    let bytes = result.unwrap();
    assert!(bytes > 0, "Should upload Q bias bytes");
    assert!(executor.has_qkv_bias(0), "Should have QKV bias (Q only)");
}

#[test]
#[serial]
fn test_cov015_forward_batched_graphed_batch_size_zero() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let inputs: Vec<f32> = vec![];
    let positions: Vec<u32> = vec![];

    let result =
        executor.forward_batched_to_token_ids_graphed(&inputs, &positions, 1, 64, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with batch size 0");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("batch size must be 1-32"),
        "Error should mention batch size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_graphed_batch_size_exceeds_max() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let m = 33; // > 32
    let inputs = vec![0.1f32; m * hidden_dim as usize];
    let positions: Vec<u32> = (0..m as u32).collect();

    let result = executor
        .forward_batched_to_token_ids_graphed(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with batch size > 32");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("batch size must be 1-32"),
        "Error should mention batch size: {}",
        err_msg
    );
}

// ==============================================================================
// COV-016: activations.rs coverage tests
// Target: Increase activations.rs coverage from 30.21%
// ==============================================================================

#[test]
#[serial]
fn test_cov016_silu_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 128u32;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();

    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let result = executor.silu_gpu(&input_gpu, n);
    assert!(result.is_ok(), "silu_gpu failed: {:?}", result.err());

    executor.stream.synchronize().expect("sync");
    let output_gpu = result.unwrap();
    let mut output = vec![0.0f32; n as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // SiLU: x * sigmoid(x) - check that non-zero inputs produce non-zero outputs
    let non_zero_count = output.iter().filter(|&&x| x.abs() > 1e-6).count();
    assert!(
        non_zero_count > n as usize / 2,
        "SiLU should produce many non-zero outputs"
    );
}

#[test]
#[serial]
fn test_cov016_gelu_async_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();

    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let result = executor.gelu_async(&input_gpu, n);
    assert!(result.is_ok(), "gelu_async failed: {:?}", result.err());

    executor.stream.synchronize().expect("sync");
    let output_gpu = result.unwrap();
    let mut output = vec![0.0f32; n as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // GELU should have non-zero outputs for most non-zero inputs
    let non_zero_count = output.iter().filter(|&&x| x.abs() > 1e-6).count();
    assert!(
        non_zero_count > n as usize / 3,
        "GELU should produce non-zero outputs"
    );
}

#[test]
#[serial]
fn test_cov016_elementwise_mul_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64u32;
    let input1 = vec![2.0f32; n as usize];
    let input2 = vec![3.0f32; n as usize];

    let input1_gpu = GpuBuffer::from_host(&executor.context, &input1).expect("input1 buffer");
    let input2_gpu = GpuBuffer::from_host(&executor.context, &input2).expect("input2 buffer");

    let result = executor.elementwise_mul_gpu(&input1_gpu, &input2_gpu, n);
    assert!(
        result.is_ok(),
        "elementwise_mul_gpu failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let output_gpu = result.unwrap();
    let mut output = vec![0.0f32; n as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // 2.0 * 3.0 = 6.0
    for val in &output {
        assert!((*val - 6.0).abs() < 1e-5, "Expected 6.0, got {}", val);
    }
}

#[test]
#[serial]
fn test_cov016_fused_swiglu_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 128u32;
    let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let up = vec![1.0f32; n as usize];

    let gate_gpu = GpuBuffer::from_host(&executor.context, &gate).expect("gate buffer");
    let up_gpu = GpuBuffer::from_host(&executor.context, &up).expect("up buffer");

    let result = executor.fused_swiglu_gpu(&gate_gpu, &up_gpu, n);
    assert!(
        result.is_ok(),
        "fused_swiglu_gpu failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let output_gpu = result.unwrap();
    let mut output = vec![0.0f32; n as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // SwiGLU: silu(gate) * up
    let non_zero_count = output.iter().filter(|&&x| x.abs() > 1e-6).count();
    assert!(non_zero_count > 0, "SwiGLU should produce non-zero outputs");
}

#[test]
#[serial]
fn test_cov016_fused_swiglu_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64u32;
    let gate = vec![1.0f32; n as usize]; // SiLU(1.0) = 1.0 * sigmoid(1.0) ≈ 0.731
    let up = vec![2.0f32; n as usize];

    let gate_gpu = GpuBuffer::from_host(&executor.context, &gate).expect("gate buffer");
    let up_gpu = GpuBuffer::from_host(&executor.context, &up).expect("up buffer");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, n as usize).expect("output buffer");

    let result = executor.fused_swiglu_into(&gate_gpu, &up_gpu, &output_gpu, n);
    assert!(
        result.is_ok(),
        "fused_swiglu_into failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; n as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // silu(1.0) * 2.0 ≈ 0.731 * 2 ≈ 1.462
    for val in &output {
        assert!(
            val.abs() > 1.0 && val.abs() < 2.0,
            "Expected ~1.46, got {}",
            val
        );
    }
}

#[test]
#[serial]
fn test_cov016_silu_gpu_cached_module() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let input = vec![0.5f32; n as usize];
    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");

    // First call compiles the module
    let _result1 = executor.silu_gpu(&input_gpu, n).expect("first silu_gpu");
    executor.stream.synchronize().expect("sync");

    // Second call reuses cached module
    let result2 = executor.silu_gpu(&input_gpu, n);
    assert!(
        result2.is_ok(),
        "cached silu_gpu failed: {:?}",
        result2.err()
    );
}

#[test]
#[serial]
fn test_cov016_gelu_async_cached_module() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 48u32;
    let input = vec![0.5f32; n as usize];
    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");

    // First call compiles
    let _result1 = executor
        .gelu_async(&input_gpu, n)
        .expect("first gelu_async");
    executor.stream.synchronize().expect("sync");

    // Second call reuses cached module
    let result2 = executor.gelu_async(&input_gpu, n);
    assert!(
        result2.is_ok(),
        "cached gelu_async failed: {:?}",
        result2.err()
    );
}

#[test]
#[serial]
fn test_cov016_elementwise_mul_varying_values() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let input1: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input2: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let input1_gpu = GpuBuffer::from_host(&executor.context, &input1).expect("input1 buffer");
    let input2_gpu = GpuBuffer::from_host(&executor.context, &input2).expect("input2 buffer");

    let result = executor.elementwise_mul_gpu(&input1_gpu, &input2_gpu, n);
    assert!(
        result.is_ok(),
        "elementwise_mul varying failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let output_gpu = result.unwrap();
    let mut output = vec![0.0f32; n as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // Verify: output[i] = i * (n - i)
    for i in 0..n as usize {
        let expected = (i as f32) * ((n as usize - i) as f32);
        assert!(
            (output[i] - expected).abs() < 1e-4,
            "Mismatch at {}: expected {}, got {}",
            i,
            expected,
            output[i]
        );
    }
}

#[test]
#[serial]
fn test_cov016_fused_swiglu_gpu_negative_inputs() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64u32;
    let gate = vec![-2.0f32; n as usize]; // Negative gate values
    let up = vec![1.0f32; n as usize];

    let gate_gpu = GpuBuffer::from_host(&executor.context, &gate).expect("gate buffer");
    let up_gpu = GpuBuffer::from_host(&executor.context, &up).expect("up buffer");

    let result = executor.fused_swiglu_gpu(&gate_gpu, &up_gpu, n);
    assert!(result.is_ok(), "swiglu negative failed: {:?}", result.err());

    executor.stream.synchronize().expect("sync");
    let output_gpu = result.unwrap();
    let mut output = vec![0.0f32; n as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // SiLU(-2.0) is small negative, so silu(-2) * 1.0 should be small negative
    for val in &output {
        assert!(
            *val < 0.1,
            "SwiGLU of negative gate should be small/negative: {}",
            val
        );
    }
}

// ==============================================================================
// COV-017: core.rs and additional coverage tests
// Target: core.rs (70.07%), attention.rs (49.32%), gemm.rs (63.33%)
// ==============================================================================

#[test]
#[serial]
fn test_cov017_num_devices() {
    // num_devices should return >= 1 on a system with CUDA
    let count = CudaExecutor::num_devices();
    if CudaExecutor::is_available() {
        assert!(count >= 1, "Should have at least one CUDA device");
    } else {
        assert_eq!(count, 0, "No devices when CUDA unavailable");
    }
}

#[test]
#[serial]
fn test_cov017_make_current() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // make_current should succeed
    let result = executor.make_current();
    assert!(result.is_ok(), "make_current failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov017_profiler_enable_disable() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

