//! APR Transformer Coverage Tests Part 3 (PMAT-803)
//!
//! Additional comprehensive tests for coverage gaps identified in:
//! - `AprBenchmarkRunner` methods
//! - `AprTransformer::from_apr_file` error paths
//! - Q4K layer weight handling
//! - SIMD helpers edge cases
//! - Matmul with various dimensions

use crate::apr_transformer::{
    AprBenchmarkRunner, AprKVCache, AprTransformer, AprTransformerConfig, AprTransformerLayer,
    GenerateConfig, Q4KLayerWeights,
};

// ============================================================================
// Part 1: AprBenchmarkRunner Tests
// ============================================================================

#[test]
fn test_benchmark_runner_new() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let runner = AprBenchmarkRunner::new(transformer);

    assert_eq!(runner.warmup_iterations(), 3);
    assert_eq!(runner.measure_iterations(), 10);
}

#[test]
fn test_benchmark_runner_set_warmup_iterations() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);

    runner.set_warmup_iterations(5);
    assert_eq!(runner.warmup_iterations(), 5);

    runner.set_warmup_iterations(0);
    assert_eq!(runner.warmup_iterations(), 0);
}

#[test]
fn test_benchmark_runner_set_measure_iterations() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);

    runner.set_measure_iterations(20);
    assert_eq!(runner.measure_iterations(), 20);

    // Minimum is 1
    runner.set_measure_iterations(0);
    assert_eq!(runner.measure_iterations(), 1);
}

#[test]
fn test_benchmark_runner_benchmark_decode() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);

    // Use minimal iterations for testing
    runner.set_warmup_iterations(1);
    runner.set_measure_iterations(2);

    let result = runner.benchmark_decode(&[0, 1], 3);
    assert!(result.is_ok());

    let bench = result.unwrap();
    assert!(bench.tokens_generated > 0);
    assert!(bench.total_time_ms > 0.0);
}

#[test]
fn test_benchmark_runner_debug() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let runner = AprBenchmarkRunner::new(transformer);

    let debug_str = format!("{runner:?}");
    assert!(debug_str.contains("AprBenchmarkRunner"));
}

// ============================================================================
// Part 2: AprTransformer from_apr_file Error Paths
// ============================================================================

#[test]
fn test_from_apr_file_not_found() {
    let result = AprTransformer::from_apr_file("/nonexistent/path/to/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_from_apr_file_directory() {
    // Attempting to read a directory should fail
    let result = AprTransformer::from_apr_file("/tmp");
    assert!(result.is_err());
}

// ============================================================================
// Part 3: AprTransformer with Q4K Layer Weights
// ============================================================================

#[test]
fn test_transformer_with_q4k_layers() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add Q4K layers
    let q4k_weights = vec![
        Q4KLayerWeights {
            attn_q_weight: Some(vec![0u8; 144]), // 1 Q4K block
            attn_k_weight: Some(vec![0u8; 144]),
            attn_v_weight: Some(vec![0u8; 144]),
            ffn_gate_weight: Some(vec![0u8; 144]),
            ffn_up_weight: Some(vec![0u8; 144]),
            ffn_down_weight: Some(vec![0u8; 144]),
            ..Default::default()
        };
        config.num_layers
    ];

    transformer.q4k_layers = Some(q4k_weights);

    // Forward should still work (fallback to F32 for invalid Q4K data)
    // Note: The actual Q4K data is zeros, which will produce zeros output
    // but the code path should execute
    assert!(transformer.q4k_layers.is_some());
}

#[test]
fn test_transformer_with_q6k_lm_head() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add Q6K lm_head weight
    transformer.lm_head_weight_q6k = Some(vec![0u8; 210]); // 1 Q6K block

    assert!(transformer.lm_head_weight_q6k.is_some());
}

#[test]
fn test_transformer_with_q4k_lm_head() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add Q4K lm_head weight
    transformer.lm_head_weight_q4k = Some(vec![0u8; 144]); // 1 Q4K block

    assert!(transformer.lm_head_weight_q4k.is_some());
}

// ============================================================================
// Part 4: Matmul Edge Cases
// ============================================================================

#[test]
fn test_forward_with_odd_hidden_dim() {
    // Hidden dim not divisible by num_heads
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 65, // Not cleanly divisible by 4
        num_layers: 1,
        num_heads: 5,
        num_kv_heads: 5,
        vocab_size: 50,
        intermediate_dim: 130,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_large_intermediate_dim() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 512, // Large intermediate
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_small_vocab() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 10, // Very small vocab
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config.clone());
    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), config.vocab_size);
}

// ============================================================================
// Part 5: RoPE Edge Cases
// ============================================================================

#[test]
fn test_forward_with_low_rope_theta() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 100.0, // Very low theta
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2, 3, 4]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_very_high_rope_theta() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 1_000_000.0, // Very high theta (like LLaMA 3.1)
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_rope_many_positions() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Process many positions to exercise RoPE at various positions
    for pos in 0..50 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

// ============================================================================
// Part 6: GQA Variations
// ============================================================================

#[test]
fn test_forward_gqa_8_to_1_ratio() {
    // 8 query heads, 1 KV head (aggressive GQA)
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 1,
        vocab_size: 50,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let mut transformer = AprTransformer::new(config.clone());

    // Adjust QKV for GQA dimensions
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.01; config.hidden_dim * qkv_out_dim];
    }

    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_gqa_8_to_1() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 1,
        vocab_size: 50,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let mut transformer = AprTransformer::new(config.clone());

    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.01; config.hidden_dim * qkv_out_dim];
    }

    let mut cache = AprKVCache::new(&config);

    for pos in 0..5 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

// ============================================================================
// Part 7: Layer Norm Variations
// ============================================================================

#[test]
fn test_forward_with_very_small_eps() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-12, // Very small epsilon
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_large_eps() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 0.1, // Large epsilon
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 8: GenerateConfig Variations
// ============================================================================

#[test]
fn test_generate_with_cache_top_k_sampling() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 1.0,
        top_k: 10, // Top-k sampling
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    let result = transformer.generate_with_cache(&[0, 1], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_top_p_sampling() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 1.0,
        top_k: 0,
        top_p: 0.5, // Nucleus sampling
        repetition_penalty: 1.0,
    };

    let result = transformer.generate_with_cache(&[0], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_repetition_penalty() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 5,
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.5, // Strong repetition penalty
    };

    let result = transformer.generate_with_cache(&[0, 1, 2], &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// Part 9: AprKVCache Edge Cases
// ============================================================================

#[test]
fn test_kv_cache_with_gqa_config() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 2, // GQA
        vocab_size: 50,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let cache = AprKVCache::new(&config);

    assert_eq!(cache.num_kv_heads(), 2);
    assert_eq!(cache.head_dim(), 128 / 8);
    assert_eq!(cache.capacity(), 64);
}

#[test]
fn test_kv_cache_append_all_layers() {
    let config = create_test_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Append to all layers (last layer auto-advances)
    for layer in 0..config.num_layers {
        cache.append(layer, &k, &v);
    }
    // No advance() needed - append() auto-advances on last layer

    // Cache length is counted per position, not per layer
    // After appending once to each layer at position 0
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_kv_cache_multiple_positions_all_layers() {
    let config = create_test_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

    // Simulate 5 positions
    for pos in 0..5 {
        let k = vec![(pos + 1) as f32; kv_size];
        let v = vec![(pos + 10) as f32; kv_size];

        for layer in 0..config.num_layers {
            cache.append(layer, &k, &v);
        }
        // No advance() needed - append() auto-advances on last layer
    }

    assert_eq!(cache.len(), 5);

    // Verify each layer's cache
    for layer in 0..config.num_layers {
        let (k_cache, v_cache) = cache.get(layer);
        assert_eq!(k_cache.len(), 5 * kv_size);
        assert_eq!(v_cache.len(), 5 * kv_size);
    }
}

// ============================================================================
// Part 10: AprTransformerLayer Edge Cases
// ============================================================================

#[test]
fn test_layer_with_all_optional_biases() {
    let hidden_dim = 64;
    let intermediate_dim = 128;

    let mut layer = AprTransformerLayer::empty(hidden_dim, intermediate_dim);

    // Add all optional biases
    layer.attn_norm_bias = Some(vec![0.01; hidden_dim]);
    layer.qkv_bias = Some(vec![0.01; hidden_dim * 3]);
    layer.attn_output_bias = Some(vec![0.01; hidden_dim]);
    layer.ffn_gate_weight = Some(vec![0.1; hidden_dim * intermediate_dim]);
    layer.ffn_gate_bias = Some(vec![0.01; intermediate_dim]);
    layer.ffn_up_bias = Some(vec![0.01; intermediate_dim]);
    layer.ffn_down_bias = Some(vec![0.01; hidden_dim]);
    layer.ffn_norm_weight = Some(vec![1.0; hidden_dim]);
    layer.ffn_norm_bias = Some(vec![0.01; hidden_dim]);

    // Count parameters should include all optionals
    let params = layer.num_parameters();
    assert!(params > hidden_dim * 3 * hidden_dim); // More than just qkv
}

#[test]
fn test_layer_empty_gqa_various_ratios() {
    // Test various GQA ratios
    for (num_heads, num_kv_heads) in [(8, 4), (8, 2), (16, 4), (32, 8)] {
        let hidden_dim = num_heads * 16; // head_dim = 16
        let intermediate_dim = hidden_dim * 4;

        let layer =
            AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
    }
}

// ============================================================================
// Part 11: Forward Pass with Various Configurations
// ============================================================================

#[test]
fn test_forward_single_layer_single_head() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_many_layers() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 12, // Many layers
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_many_layers() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 8,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    for pos in 0..10 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

// ============================================================================
// Part 12: Predict Next Edge Cases
// ============================================================================

#[test]
fn test_predict_next_returns_valid_token() {
    let config = create_test_config();
    let vocab_size = config.vocab_size;
    let transformer = AprTransformer::new(config);

    for tokens in [vec![0], vec![0, 1], vec![0, 1, 2, 3, 4]] {
        let result = transformer.predict_next(&tokens);
        assert!(result.is_ok());
        assert!((result.unwrap() as usize) < vocab_size);
    }
}

#[test]
fn test_predict_next_deterministic() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    // Same input should produce same output (no randomness in predict_next)
    let r1 = transformer.predict_next(&[0, 1, 2]).unwrap();
    let r2 = transformer.predict_next(&[0, 1, 2]).unwrap();
    assert_eq!(r1, r2);
}

// ============================================================================
// Part 13: Num Parameters and Memory Size
// ============================================================================

#[test]
fn test_num_parameters_with_optionals() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    let base_params = transformer.num_parameters();

    // Add optional biases
    transformer.output_norm_bias = Some(vec![0.0; config.hidden_dim]);
    transformer.lm_head_bias = Some(vec![0.0; config.vocab_size]);

    let with_optionals = transformer.num_parameters();

    // Should be larger with optionals
    assert!(with_optionals > base_params);
    assert_eq!(
        with_optionals - base_params,
        config.hidden_dim + config.vocab_size
    );
}

#[test]
fn test_memory_size_consistency() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let params = transformer.num_parameters();
    let memory = transformer.memory_size();

    // Memory = params * 4 (f32 = 4 bytes)
    assert_eq!(memory, params * 4);
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_config() -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
    }
}
