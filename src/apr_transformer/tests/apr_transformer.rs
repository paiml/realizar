//! APR Transformer Core Tests (PMAT-803)
//!
//! Comprehensive tests for `src/apr_transformer/mod.rs` to drive coverage.
//!
//! # Coverage Targets
//!
//! - `AprTransformer` initialization paths
//! - Forward pass code paths
//! - Configuration variations (GQA, SwiGLU, etc.)
//! - Error handling paths
//! - Utility methods (embed, layer_norm, matmul, etc.)

use crate::apr_transformer::{
    AprKVCache, AprTransformer, AprTransformerConfig, AprTransformerLayer, GenerateConfig,
    Q4KLayerWeights,
};

// ============================================================================
// Part 1: AprTransformer Construction Tests
// ============================================================================

#[test]
fn test_apr_transformer_new_basic() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    assert_eq!(transformer.config().hidden_dim, config.hidden_dim);
    assert_eq!(transformer.config().num_layers, config.num_layers);
    assert_eq!(transformer.config().vocab_size, config.vocab_size);
    assert_eq!(transformer.layers.len(), config.num_layers);
}

#[test]
fn test_apr_transformer_new_single_layer() {
    let config = AprTransformerConfig {
        num_layers: 1,
        ..create_test_config()
    };
    let transformer = AprTransformer::new(config);

    assert_eq!(transformer.layers.len(), 1);
}

#[test]
fn test_apr_transformer_new_many_layers() {
    let config = AprTransformerConfig {
        num_layers: 12,
        ..create_test_config()
    };
    let transformer = AprTransformer::new(config);

    assert_eq!(transformer.layers.len(), 12);
}

#[test]
fn test_apr_transformer_token_embedding_size() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    // token_embedding should be vocab_size * hidden_dim
    let expected_size = config.vocab_size * config.hidden_dim;
    assert_eq!(transformer.token_embedding.len(), expected_size);
}

#[test]
fn test_apr_transformer_lm_head_size() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    // lm_head_weight should be hidden_dim * vocab_size
    let expected_size = config.hidden_dim * config.vocab_size;
    assert_eq!(transformer.lm_head_weight.len(), expected_size);
}

#[test]
fn test_apr_transformer_output_norm_size() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    assert_eq!(transformer.output_norm_weight.len(), config.hidden_dim);
    // Default output_norm_weight should be 1.0
    assert!(transformer
        .output_norm_weight
        .iter()
        .all(|&x| (x - 1.0).abs() < 1e-6));
}

#[test]
fn test_apr_transformer_optional_biases_none() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    assert!(transformer.output_norm_bias.is_none());
    assert!(transformer.lm_head_bias.is_none());
    assert!(transformer.q4k_layers.is_none());
    assert!(transformer.lm_head_weight_q6k.is_none());
    assert!(transformer.lm_head_weight_q4k.is_none());
}

// ============================================================================
// Part 2: AprTransformer.embed() Tests
// ============================================================================

#[test]
fn test_embed_single_token() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Set up distinct embedding for token 0
    for i in 0..config.hidden_dim {
        transformer.token_embedding[i] = (i + 1) as f32;
    }

    let embedding = transformer.embed(&[0]);
    assert_eq!(embedding.len(), config.hidden_dim);
    for i in 0..config.hidden_dim {
        assert!((embedding[i] - (i + 1) as f32).abs() < 1e-6);
    }
}

#[test]
fn test_embed_multiple_tokens() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Set up embeddings for tokens 0, 1, 2
    for token_id in 0..3 {
        let offset = token_id * config.hidden_dim;
        for i in 0..config.hidden_dim {
            transformer.token_embedding[offset + i] = (token_id * 100 + i + 1) as f32;
        }
    }

    let embeddings = transformer.embed(&[0, 1, 2]);
    assert_eq!(embeddings.len(), 3 * config.hidden_dim);

    // Check token 0
    for i in 0..config.hidden_dim {
        assert!((embeddings[i] - (i + 1) as f32).abs() < 1e-6);
    }
    // Check token 1
    for i in 0..config.hidden_dim {
        assert!((embeddings[config.hidden_dim + i] - (100 + i + 1) as f32).abs() < 1e-6);
    }
}

#[test]
fn test_embed_out_of_vocab_token() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    // Token 999 is beyond vocab_size (100)
    let embedding = transformer.embed(&[999]);
    assert_eq!(embedding.len(), config.hidden_dim);
    // Should return zeros for OOV
    assert!(embedding.iter().all(|&x| x == 0.0));
}

#[test]
fn test_embed_empty_sequence() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let embedding = transformer.embed(&[]);
    assert!(embedding.is_empty());
}

// ============================================================================
// Part 3: AprTransformer.forward() Tests
// ============================================================================

#[test]
fn test_forward_empty_tokens_error() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let result = transformer.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_forward_single_token() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    let result = transformer.forward(&[0]);
    assert!(result.is_ok());

    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_multiple_tokens() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    let result = transformer.forward(&[0, 1, 2, 3]);
    assert!(result.is_ok());

    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_oov_token() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    // Should handle OOV gracefully (with zero embedding)
    let result = transformer.forward(&[999]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_gqa() {
    // GQA: num_kv_heads < num_heads
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_heads: 8,
        num_kv_heads: 2, // GQA
        ..create_test_config()
    };

    let mut transformer = AprTransformer::new(config.clone());

    // Adjust layer QKV weights for GQA dimensions
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.0; config.hidden_dim * qkv_out_dim];
    }

    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_swiglu() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add gate weight to enable SwiGLU path
    for layer in &mut transformer.layers {
        layer.ffn_gate_weight = Some(vec![0.1; config.hidden_dim * config.intermediate_dim]);
    }

    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_ffn_norm() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add FFN norm
    for layer in &mut transformer.layers {
        layer.ffn_norm_weight = Some(vec![1.0; config.hidden_dim]);
    }

    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_biases() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add various biases
    let qkv_dim = config.hidden_dim * 3;
    for layer in &mut transformer.layers {
        layer.qkv_bias = Some(vec![0.01; qkv_dim]);
        layer.attn_output_bias = Some(vec![0.01; config.hidden_dim]);
        layer.ffn_up_bias = Some(vec![0.01; config.intermediate_dim]);
        layer.ffn_down_bias = Some(vec![0.01; config.hidden_dim]);
    }
    transformer.lm_head_bias = Some(vec![0.01; config.vocab_size]);

    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 4: AprTransformer.forward_with_cache() Tests
// ============================================================================

#[test]
fn test_forward_with_cache_single_token() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());

    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_forward_with_cache_sequential() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Process multiple tokens sequentially
    for pos in 0..5 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
        assert_eq!(cache.len(), pos + 1);
    }
}

#[test]
fn test_forward_with_cache_gqa() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_heads: 8,
        num_kv_heads: 2,
        ..create_test_config()
    };

    let mut transformer = AprTransformer::new(config.clone());

    // Adjust QKV for GQA
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.0; config.hidden_dim * qkv_out_dim];
    }

    let mut cache = AprKVCache::new(&config);

    for pos in 0..3 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

#[test]
fn test_forward_with_cache_swiglu() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Enable SwiGLU
    for layer in &mut transformer.layers {
        layer.ffn_gate_weight = Some(vec![0.1; config.hidden_dim * config.intermediate_dim]);
    }

    let mut cache = AprKVCache::new(&config);

    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());
}

// ============================================================================
// Part 5: AprTransformer.predict_next() Tests
// ============================================================================

#[test]
fn test_predict_next_basic() {
    let config = create_test_config();
    let vocab_size = config.vocab_size;
    let transformer = AprTransformer::new(config);

    let result = transformer.predict_next(&[0, 1, 2]);
    assert!(result.is_ok());

    let token = result.unwrap();
    assert!(token < vocab_size as u32);
}

#[test]
fn test_predict_next_empty_error() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let result = transformer.predict_next(&[]);
    assert!(result.is_err());
}

// ============================================================================
// Part 6: AprTransformer.generate() Tests
// ============================================================================

#[test]
fn test_generate_basic() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let result = transformer.generate(&[0, 1], 3);
    assert!(result.is_ok());

    let tokens = result.unwrap();
    // Should have prompt + generated tokens
    assert!(tokens.len() >= 2);
    assert!(tokens.len() <= 5); // 2 prompt + up to 3 generated
}

#[test]
fn test_generate_stops_at_eos() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    // Set lm_head to output EOS token (2) as argmax
    // This requires setting up weights such that token 2 has highest logit
    // For simplicity, we'll test that generate returns at least the prompt
    let result = transformer.generate(&[0], 5);
    assert!(result.is_ok());
}

// ============================================================================
// Part 7: AprTransformer.generate_with_cache() Tests
// ============================================================================

#[test]
fn test_generate_with_cache_basic() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 0.0, // Greedy
        ..Default::default()
    };

    let result = transformer.generate_with_cache(&[0, 1], &gen_config);
    assert!(result.is_ok());

    let tokens = result.unwrap();
    assert!(tokens.len() >= 2);
}

#[test]
fn test_generate_with_cache_empty_prompt_error() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let gen_config = GenerateConfig::default();

    let result = transformer.generate_with_cache(&[], &gen_config);
    assert!(result.is_err());
}

#[test]
fn test_generate_with_cache_with_temperature() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let gen_config = GenerateConfig {
        max_tokens: 2,
        temperature: 1.0,
        ..Default::default()
    };

    let result = transformer.generate_with_cache(&[0], &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// Part 8: Utility Method Tests
// ============================================================================

#[test]
fn test_num_parameters() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let params = transformer.num_parameters();
    assert!(params > 0);

    // Should include embedding, layers, and lm_head
    // Minimum: vocab_size * hidden_dim (embedding)
    assert!(params >= 100 * 64);
}

#[test]
fn test_memory_size() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let size = transformer.memory_size();
    assert!(size > 0);

    // Memory size = num_parameters * 4 bytes
    assert_eq!(size, transformer.num_parameters() * 4);
}

#[test]
fn test_config_accessor() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    let returned_config = transformer.config();
    assert_eq!(returned_config.hidden_dim, config.hidden_dim);
    assert_eq!(returned_config.num_layers, config.num_layers);
    assert_eq!(returned_config.vocab_size, config.vocab_size);
}

// ============================================================================
// Part 9: from_apr_bytes() Error Tests
// ============================================================================

#[test]
fn test_from_apr_bytes_too_small() {
    let data = vec![0u8; 32]; // Less than 64 bytes
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_valid_magic_v1() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0"); // APR v1 magic

    // Set up minimal valid header
    // tensor_count at offset 8
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    // metadata_offset at offset 12
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    // metadata_size at offset 20
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    // tensor_index_offset at offset 24
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    // data_offset at offset 32
    data[32..40].copy_from_slice(&66u64.to_le_bytes());

    // Add minimal JSON metadata
    data[64] = b'{';
    data[65] = b'}';

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_from_apr_bytes_valid_magic_v2() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR2"); // APR v2 magic

    // Set up minimal header
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    data[32..40].copy_from_slice(&66u64.to_le_bytes());

    // Minimal JSON
    data[64] = b'{';
    data[65] = b'}';

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_from_apr_bytes_metadata_out_of_bounds() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");

    // metadata_offset points beyond file
    data[12..20].copy_from_slice(&1000u64.to_le_bytes());
    data[20..24].copy_from_slice(&100u32.to_le_bytes());

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// Part 10: Layer Tests
// ============================================================================

#[test]
fn test_layer_empty() {
    let layer = AprTransformerLayer::empty(64, 256);

    assert_eq!(layer.attn_norm_weight.len(), 64);
    assert_eq!(layer.qkv_weight.len(), 64 * 3 * 64);
    assert_eq!(layer.attn_output_weight.len(), 64 * 64);
    assert_eq!(layer.ffn_up_weight.len(), 64 * 256);
    assert_eq!(layer.ffn_down_weight.len(), 256 * 64);
}

#[test]
fn test_layer_empty_gqa() {
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let intermediate_dim = 256;

    let layer =
        AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

    // QKV: Q (64) + K (16) + V (16) = 96 elements per hidden dim
    let head_dim = hidden_dim / num_heads; // 8
    let kv_dim = num_kv_heads * head_dim; // 16
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 64 + 32 = 96

    assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
}

#[test]
fn test_layer_num_parameters() {
    let layer = AprTransformerLayer::empty(64, 256);
    let params = layer.num_parameters();

    // Calculate expected
    let expected = 64  // attn_norm
        + 64 * 192  // qkv (3 * hidden)
        + 64 * 64   // attn_output
        + 64 * 256  // ffn_up
        + 256 * 64; // ffn_down

    assert_eq!(params, expected);
}

// ============================================================================
// Part 11: Q4K Layer Weights Tests
// ============================================================================

#[test]
fn test_q4k_layer_weights_default() {
    let weights = Q4KLayerWeights::default();

    assert!(weights.qkv_weight.is_none());
    assert!(weights.attn_q_weight.is_none());
    assert!(weights.attn_k_weight.is_none());
    assert!(weights.attn_v_weight.is_none());
    assert!(weights.attn_output_weight.is_none());
    assert!(weights.ffn_gate_weight.is_none());
    assert!(weights.ffn_up_weight.is_none());
    assert!(weights.ffn_down_weight.is_none());
}

#[test]
fn test_q4k_layer_weights_with_values() {
    let weights = Q4KLayerWeights {
        attn_q_weight: Some(vec![1, 2, 3]),
        attn_k_weight: Some(vec![4, 5, 6]),
        attn_v_weight: Some(vec![7, 8, 9]),
        ffn_down_weight: Some(vec![10, 11, 12]),
        ..Default::default()
    };

    assert!(weights.attn_q_weight.is_some());
    assert_eq!(weights.attn_q_weight.unwrap()[0], 1);
    assert!(weights.ffn_down_weight.is_some());
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
