
// ============================================================================
// transpose_matrix Edge Cases
// ============================================================================

#[test]
fn test_transpose_3x3_explicit() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = transpose_matrix(&data, 3, 3);

    // Original: [[1,2,3],[4,5,6],[7,8,9]]
    // Transposed: [[1,4,7],[2,5,8],[3,6,9]]
    assert_eq!(result, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
}

#[test]
fn test_transpose_2x4() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = transpose_matrix(&data, 2, 4);

    // Original 2x4: [[1,2,3,4],[5,6,7,8]]
    // Transposed 4x2: [[1,5],[2,6],[3,7],[4,8]]
    assert_eq!(result, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
}

#[test]
fn test_transpose_4x2() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = transpose_matrix(&data, 4, 2);

    // Original 4x2: [[1,2],[3,4],[5,6],[7,8]]
    // Transposed 2x4: [[1,3,5,7],[2,4,6,8]]
    assert_eq!(result, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_transpose_double_is_identity() {
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let transposed = transpose_matrix(&original, 2, 3);
    let double_transposed = transpose_matrix(&transposed, 3, 2);

    assert_eq!(original, double_transposed);
}

#[test]
fn test_transpose_preserves_sum() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let original_sum: f32 = data.iter().sum();

    let transposed = transpose_matrix(&data, 3, 4);
    let transposed_sum: f32 = transposed.iter().sum();

    assert!((original_sum - transposed_sum).abs() < 1e-6);
}

#[test]
fn test_transpose_symmetric_matrix() {
    // A symmetric matrix should have some invariant properties
    let data = vec![1.0, 2.0, 2.0, 3.0]; // 2x2 symmetric
    let result = transpose_matrix(&data, 2, 2);

    // For symmetric, transpose should give different result unless it's identity-like
    // [1, 2]   -> [1, 2]
    // [2, 3]      [2, 3]
    assert_eq!(result, vec![1.0, 2.0, 2.0, 3.0]);
}

// ============================================================================
// MockExecutor Integration Tests
// ============================================================================

#[test]
fn test_mock_executor_with_gpu_model_from_apr_f32() {
    let apr = create_minimal_f32_apr();
    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());

    let mut gpu_model = result.unwrap();

    // Inject mock executor
    let mock = MockExecutor::new("apr_f32_test");
    gpu_model.with_test_executor(Box::new(mock));

    assert!(gpu_model.has_test_executor());

    // Perform a forward pass with mock
    let token_ids = vec![1, 2, 3];
    let forward_result = gpu_model.forward_gpu(&token_ids);
    assert!(forward_result.is_ok());
}

#[test]
fn test_mock_executor_with_gpu_model_from_apr_q4() {
    let apr = create_minimal_q4_apr();
    let result = AprToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());

    let mut gpu_model = result.unwrap();

    // Inject mock executor
    let mock = MockExecutor::new("apr_q4_test");
    gpu_model.with_test_executor(Box::new(mock));

    assert!(gpu_model.has_test_executor());

    // Perform a forward pass with mock
    let token_ids = vec![5, 10, 15];
    let forward_result = gpu_model.forward_gpu(&token_ids);
    assert!(forward_result.is_ok());
}

#[test]
fn test_mock_executor_generate_with_adapted_model() {
    let apr = create_minimal_f32_apr();
    let mut gpu_model = AprF32ToGpuAdapter::to_gpu_model(&apr).unwrap();

    let mock = MockExecutor::new("generate_test");
    gpu_model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(3);
    let prompt = vec![1, 2];
    let result = gpu_model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(tokens.len() >= prompt.len());
}

#[test]
fn test_mock_executor_clear_and_restore() {
    let apr = create_minimal_f32_apr();
    let mut gpu_model = AprF32ToGpuAdapter::to_gpu_model(&apr).unwrap();

    // Initially no test executor
    assert!(!gpu_model.has_test_executor());

    // Add mock
    let mock = MockExecutor::new("clear_test");
    gpu_model.with_test_executor(Box::new(mock));
    assert!(gpu_model.has_test_executor());

    // Clear mock
    gpu_model.clear_test_executor();
    assert!(!gpu_model.has_test_executor());

    // Add another mock
    let mock2 = MockExecutor::new("restored_test");
    gpu_model.with_test_executor(Box::new(mock2));
    assert!(gpu_model.has_test_executor());
}

// ============================================================================
// GpuModelConfig Derived Tests
// ============================================================================

#[test]
fn test_gpu_model_config_from_apr_f32_dimensions() {
    let apr = create_minimal_f32_apr();
    let gpu_model = AprF32ToGpuAdapter::to_gpu_model(&apr).unwrap();

    // Verify derived dimensions
    let config = gpu_model.config();
    assert_eq!(config.head_dim(), 16); // 64 / 4
    assert_eq!(config.kv_dim(), 64); // 4 * 16 (MHA)
    assert_eq!(config.qkv_dim(), 192); // 64 + 2*64 (MHA)
    assert!(!config.is_gqa());
}

#[test]
fn test_gpu_model_config_from_apr_f32_gqa_dimensions() {
    let apr = create_gqa_f32_apr();
    let gpu_model = AprF32ToGpuAdapter::to_gpu_model(&apr).unwrap();

    let config = gpu_model.config();
    assert_eq!(config.head_dim(), 8); // 64 / 8
    assert_eq!(config.kv_dim(), 16); // 2 * 8 (GQA)
    assert_eq!(config.qkv_dim(), 96); // 64 + 2*16 (GQA)
    assert!(config.is_gqa());
}

#[test]
fn test_gpu_model_config_from_apr_q4_dimensions() {
    let apr = create_minimal_q4_apr();
    let gpu_model = AprToGpuAdapter::to_gpu_model(&apr).unwrap();

    let config = gpu_model.config();
    assert_eq!(config.head_dim(), 16);
    assert_eq!(config.kv_dim(), 64);
    assert_eq!(config.qkv_dim(), 192);
    assert!(!config.is_gqa());
}

// ============================================================================
// Multiple Layer Tests
// ============================================================================

#[test]
fn test_apr_f32_to_gpu_multiple_layers() {
    let config = create_minimal_apr_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;

    let apr = AprTransformer {
        config: config.clone(),
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: (0..4).map(|_| create_f32_layer(&config)).collect(), // 4 layers
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab_size * hidden_dim],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());

    let gpu_model = result.unwrap();
    // Config should still say num_layers from config, but model has 4 layers
    assert_eq!(gpu_model.config.num_layers, 2); // Config says 2
}

#[test]
fn test_apr_q4_to_gpu_single_layer() {
    let config = AprTransformerConfig {
        architecture: "single".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let apr = QuantizedAprTransformerQ4 {
        config: config.clone(),
        token_embedding: vec![0.01; 100 * 64],
        layers: vec![create_q4_layer(&config)], // Single layer
        output_norm_weight: vec![1.0; 64],
        lm_head_weight: QuantizedAprTensorQ4::zeros(64, 100),
    };

    let result = AprToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());

    let gpu_model = result.unwrap();
    assert_eq!(gpu_model.config.num_layers, 1);
}

#[test]
fn test_apr_q4_to_gpu_no_layers() {
    let config = AprTransformerConfig {
        architecture: "empty".to_string(),
        hidden_dim: 64,
        num_layers: 0,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let apr = QuantizedAprTransformerQ4 {
        config: config.clone(),
        token_embedding: vec![0.01; 100 * 64],
        layers: vec![], // No layers
        output_norm_weight: vec![1.0; 64],
        lm_head_weight: QuantizedAprTensorQ4::zeros(64, 100),
    };

    let result = AprToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());

    let gpu_model = result.unwrap();
    assert_eq!(gpu_model.config.num_layers, 0);
}

// ============================================================================
// Dequantization Edge Cases
// ============================================================================

#[test]
fn test_dequantize_tensor_zero_expected() {
    let result = AprToGpuAdapter::dequantize_tensor(&[], 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_dequantize_tensor_single_block() {
    // Q4_0 block: 2 bytes scale + 16 bytes data = 18 bytes for 32 values
    let mut data = vec![0u8; 18];
    data[0] = 0x00;
    data[1] = 0x3c; // f16 1.0

    let result = AprToGpuAdapter::dequantize_tensor(&data, 32);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_dequantize_tensor_needs_padding() {
    // Single block produces 32 values, but we request 64
    let mut data = vec![0u8; 18];
    data[0] = 0x00;
    data[1] = 0x3c;

    let result = AprToGpuAdapter::dequantize_tensor(&data, 64);
    assert!(result.is_ok());

    let values = result.unwrap();
    assert_eq!(values.len(), 64);
    // Last 32 values should be zero (padded)
    for &v in &values[32..] {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_dequantize_tensor_needs_truncation() {
    // Two blocks produce 64 values, but we only request 32
    let mut data = vec![0u8; 36];
    data[0] = 0x00;
    data[1] = 0x3c;
    data[18] = 0x00;
    data[19] = 0x3c;

    let result = AprToGpuAdapter::dequantize_tensor(&data, 32);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32);
}

// ============================================================================
// RoPE Theta Tests
// ============================================================================

#[test]
fn test_rope_theta_preservation() {
    let apr_config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 500000.0, // Non-standard theta
        eps: 1e-5,
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);
    assert_eq!(gpu_config.rope_theta, 500000.0);
}

#[test]
fn test_eps_preservation() {
    let apr_config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-6, // Different epsilon
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);
    assert_eq!(gpu_config.eps, 1e-6);
}

// ============================================================================
// FFN Weight Extraction Tests
// ============================================================================

#[test]
fn test_extract_ffn_weights_dimensions() {
    let config = create_minimal_apr_config();
    let layer = create_q4_layer(&config);

    let result =
        AprToGpuAdapter::extract_ffn_weights(&layer, config.hidden_dim, config.intermediate_dim);
    assert!(result.is_ok());

    let (fc1, fc2) = result.unwrap();
    assert_eq!(fc1.len(), config.hidden_dim * config.intermediate_dim);
    assert_eq!(fc2.len(), config.intermediate_dim * config.hidden_dim);
}

#[test]
fn test_extract_out_weights_dimensions() {
    let config = create_minimal_apr_config();
    let layer = create_q4_layer(&config);

    let result = AprToGpuAdapter::extract_out_weights(&layer, config.hidden_dim);
    assert!(result.is_ok());

    let weights = result.unwrap();
    assert_eq!(weights.len(), config.hidden_dim * config.hidden_dim);
}
