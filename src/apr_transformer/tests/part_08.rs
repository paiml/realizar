//! T-COV-95 Phase 51: Deep coverage for apr_transformer/mod.rs uncovered lines
//!
//! Covers:
//! - AprTransformer::new() constructor (lines ~1141-1162)
//! - AprTransformer::config() accessor
//! - AprTransformer::embed() with in-vocab and out-of-vocab tokens (lines ~1230-1262)
//! - AprTransformer::num_parameters() and memory_size() (lines ~1209-1226)
//! - AprTransformer::forward() on pygmy model (lines ~1311-1686)
//! - AprTransformer::forward() error on empty input
//! - AprTransformer::forward_traced() on pygmy model (lines ~1704-1940)
//! - AprTransformer::forward_traced() error on empty input
//! - AprTransformer::predict_next() (lines ~1955-1968)
//! - AprTransformer::generate() with EOS stopping (lines ~1180-1205)
//! - TracedForward trait impl
//! - AprTransformer serialization roundtrip (Serialize/Deserialize)
//! - from_apr_bytes() with valid APR v2 binary (lines ~418-1138)
//! - from_apr_bytes() error cases (too small, bad magic, metadata beyond file)
//! - forward_with_cache() on pygmy model (lines ~1983-2509)
//! - generate_with_cache() delegation

use crate::apr_transformer::AprTransformer;
use crate::apr_transformer::{
    ActivationStats, AprKVCache, AprTransformerConfig, AprTransformerLayer, ForwardTrace,
    GenerateConfig, LayerActivation, Q4KLayerWeights, TracedForward,
};

// ============================================================================
// Helper: Create a pygmy (tiny) model for testing forward pass coverage
// ============================================================================

/// Create a minimal AprTransformer with non-zero weights for testing.
/// Uses hidden_dim=8, num_layers=1, num_heads=2, vocab_size=16, intermediate_dim=16
fn make_pygmy_model() -> AprTransformer {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 16;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads; // 4
    let kv_dim = num_kv_heads * head_dim; // 8
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 8 + 16 = 24

    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    // Token embedding: identity-like (each token maps to distinct vector)
    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok + d) as f32) * 0.01;
        }
    }

    // QKV weight: small non-zero values [qkv_out_dim, hidden_dim]
    let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();

    // Attention output weight: [hidden_dim, hidden_dim]
    let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| if i % (hidden_dim + 1) == 0 { 0.1 } else { 0.01 })
        .collect();

    // FFN weights: use SwiGLU path (gate + up + down)
    let ffn_gate_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();
    let ffn_up_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 3) as f32 - 1.0) * 0.01)
        .collect();
    let ffn_down_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| ((i % 4) as f32 - 1.5) * 0.01)
        .collect();

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight,
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_gate_weight: Some(ffn_gate_weight),
        ffn_gate_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
    };

    // LM head weight: [hidden_dim, vocab_size] -> we need hidden_dim * vocab_size
    let lm_head_weight: Vec<f32> = (0..hidden_dim * vocab_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
        .collect();

    AprTransformer {
        config,
        token_embedding,
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight,
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

/// Create a pygmy model WITHOUT SwiGLU (no gate weight) for standard GELU path coverage
fn make_pygmy_model_gelu() -> AprTransformer {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 16;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let config = AprTransformerConfig {
        architecture: "test-gelu".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok + d) as f32) * 0.01;
        }
    }

    let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();

    let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| if i % (hidden_dim + 1) == 0 { 0.1 } else { 0.01 })
        .collect();

    // No ffn_gate_weight -> standard GELU MLP path
    let ffn_up_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 3) as f32 - 1.0) * 0.01)
        .collect();
    let ffn_down_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| ((i % 4) as f32 - 1.5) * 0.01)
        .collect();

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight,
        qkv_bias: Some(vec![0.01; qkv_out_dim]),
        attn_output_weight,
        attn_output_bias: Some(vec![0.001; hidden_dim]),
        ffn_gate_weight: None, // No gate -> GELU path
        ffn_gate_bias: None,
        ffn_up_weight,
        ffn_up_bias: Some(vec![0.001; intermediate_dim]),
        ffn_down_weight,
        ffn_down_bias: Some(vec![0.001; hidden_dim]),
        ffn_norm_weight: None, // No FFN norm -> hidden passed directly
        ffn_norm_bias: None,
    };

    let lm_head_weight: Vec<f32> = (0..hidden_dim * vocab_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
        .collect();

    AprTransformer {
        config,
        token_embedding,
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: Some(vec![0.0; hidden_dim]),
        lm_head_weight,
        lm_head_bias: Some(vec![0.0; vocab_size]),
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

// ============================================================================
// AprTransformer::new()
// ============================================================================

#[test]
fn test_apr_transformer_new_basic() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 64,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
    };
    let model = AprTransformer::new(config.clone());

    assert_eq!(model.config.hidden_dim, 32);
    assert_eq!(model.config.num_layers, 2);
    assert_eq!(model.layers.len(), 2);
    assert_eq!(model.token_embedding.len(), 100 * 32);
    assert_eq!(model.output_norm_weight.len(), 32);
    assert_eq!(model.lm_head_weight.len(), 32 * 100);
    assert!(model.output_norm_bias.is_none());
    assert!(model.lm_head_bias.is_none());
    assert!(model.q4k_layers.is_none());
    assert!(model.lm_head_weight_q6k.is_none());
    assert!(model.lm_head_weight_q4k.is_none());
}

#[test]
fn test_apr_transformer_new_output_norm_is_ones() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        ..Default::default()
    };
    let model = AprTransformer::new(config);
    // Output norm weight should be initialized to 1.0
    assert!(model
        .output_norm_weight
        .iter()
        .all(|&w| (w - 1.0).abs() < 1e-6));
}

// ============================================================================
// AprTransformer::config()
// ============================================================================

#[test]
fn test_apr_transformer_config_accessor() {
    let config = AprTransformerConfig {
        architecture: "phi".to_string(),
        hidden_dim: 256,
        ..Default::default()
    };
    let model = AprTransformer::new(config);
    let cfg = model.config();
    assert_eq!(cfg.architecture, "phi");
    assert_eq!(cfg.hidden_dim, 256);
}

// ============================================================================
// AprTransformer::embed()
// ============================================================================

#[test]
fn test_embed_single_token() {
    let model = make_pygmy_model();
    let embeddings = model.embed(&[0]);
    assert_eq!(embeddings.len(), 8); // hidden_dim = 8
                                     // Token 0: values should be (0+d)*0.01 for d in 0..8
    for d in 0..8 {
        let expected = (d as f32) * 0.01;
        assert!(
            (embeddings[d] - expected).abs() < 1e-6,
            "embed[{d}]: expected {expected}, got {}",
            embeddings[d]
        );
    }
}

#[test]
fn test_embed_multiple_tokens() {
    let model = make_pygmy_model();
    let embeddings = model.embed(&[0, 1, 2]);
    assert_eq!(embeddings.len(), 3 * 8); // 3 tokens * hidden_dim=8
}

#[test]
fn test_embed_out_of_vocab_returns_zeros() {
    let model = make_pygmy_model();
    // vocab_size=16, so token 999 is out of vocab
    let embeddings = model.embed(&[999]);
    assert_eq!(embeddings.len(), 8);
    assert!(embeddings.iter().all(|&v| v == 0.0));
}

#[test]
fn test_embed_empty_input() {
    let model = make_pygmy_model();
    let embeddings = model.embed(&[]);
    assert!(embeddings.is_empty());
}

// ============================================================================
// AprTransformer::num_parameters() and memory_size()
// ============================================================================

#[test]
fn test_num_parameters_pygmy() {
    let model = make_pygmy_model();
    let params = model.num_parameters();
    // token_embedding: 16 * 8 = 128
    // layer.num_parameters(): attn_norm(8) + qkv(24*8=192) + attn_out(8*8=64)
    //   + ffn_gate(16*8=128) + ffn_up(16*8=128) + ffn_down(8*16=128) + ffn_norm(8)
    //   = 8+192+64+128+128+128+8 = 656
    // output_norm: 8
    // lm_head: 8*16 = 128
    // Total: 128 + 656 + 8 + 128 = 920
    assert!(params > 0, "num_parameters should be positive");
    // Verify exact count
    let expected = 128 + 656 + 8 + 128;
    assert_eq!(
        params, expected,
        "Expected {expected} parameters, got {params}"
    );
}

#[test]
fn test_memory_size_is_4x_params() {
    let model = make_pygmy_model();
    assert_eq!(model.memory_size(), model.num_parameters() * 4);
}

#[test]
fn test_num_parameters_with_bias() {
    let mut model = make_pygmy_model();
    let base_params = model.num_parameters();
    model.output_norm_bias = Some(vec![0.0; 8]);
    model.lm_head_bias = Some(vec![0.0; 16]);
    assert_eq!(model.num_parameters(), base_params + 8 + 16);
}

// ============================================================================
// AprTransformer::forward() - SwiGLU path
// ============================================================================

#[test]
fn test_forward_swiglu_produces_logits() {
    let model = make_pygmy_model();
    let logits = model.forward(&[1]).expect("forward should succeed");
    assert_eq!(logits.len(), 16); // vocab_size
                                  // Logits should be finite
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "All logits should be finite"
    );
}

#[test]
fn test_forward_swiglu_multi_token() {
    let model = make_pygmy_model();
    let logits = model.forward(&[1, 2, 3]).expect("forward should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_empty_tokens_error() {
    let model = make_pygmy_model();
    let result = model.forward(&[]);
    assert!(result.is_err(), "forward with empty tokens should error");
}

// ============================================================================
// AprTransformer::forward() - GELU path (no gate weight)
// ============================================================================

#[test]
fn test_forward_gelu_path_produces_logits() {
    let model = make_pygmy_model_gelu();
    let logits = model.forward(&[1]).expect("forward should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_gelu_path_multi_token() {
    let model = make_pygmy_model_gelu();
    let logits = model.forward(&[0, 1, 2]).expect("forward should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_gelu_with_biases() {
    // This model has qkv_bias, attn_output_bias, ffn_up_bias, ffn_down_bias,
    // output_norm_bias, and lm_head_bias - covers all bias application paths
    let model = make_pygmy_model_gelu();
    let logits = model.forward(&[5]).expect("forward should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// AprTransformer::forward_traced() - SwiGLU path
// ============================================================================

#[test]
fn test_forward_traced_swiglu_returns_trace() {
    let model = make_pygmy_model();
    let trace = model
        .forward_traced(&[1])
        .expect("forward_traced should succeed");
    assert_eq!(trace.input_tokens, vec![1]);
    assert_eq!(trace.logits.len(), 16);
    assert_eq!(trace.layer_activations.len(), 1);

    // Check embed stats
    assert_eq!(trace.embed_stats.count, 8); // hidden_dim=8, seq_len=1

    // Check layer activation stats
    let layer = &trace.layer_activations[0];
    assert_eq!(layer.layer_idx, 0);
    assert!(layer.attn_norm_stats.count > 0);
    assert!(layer.qkv_stats.count > 0);
    assert!(layer.attn_out_stats.count > 0);
    assert!(layer.ffn_norm_stats.count > 0);
    assert!(layer.ffn_out_stats.count > 0);
    assert!(layer.output_stats.count > 0);

    // Final norm and logits stats
    assert!(trace.final_norm_stats.count > 0);
    assert!(trace.logits_stats.count > 0);
}

#[test]
fn test_forward_traced_swiglu_multi_token() {
    let model = make_pygmy_model();
    let trace = model
        .forward_traced(&[0, 1])
        .expect("forward_traced should succeed");
    assert_eq!(trace.input_tokens.len(), 2);
    assert_eq!(trace.logits.len(), 16);
    // embed_stats covers seq_len*hidden_dim = 2*8 = 16 elements
    assert_eq!(trace.embed_stats.count, 16);
}

#[test]
fn test_forward_traced_empty_tokens_error() {
    let model = make_pygmy_model();
    let result = model.forward_traced(&[]);
    assert!(result.is_err());
}

// ============================================================================
// AprTransformer::forward_traced() - GELU path
// ============================================================================

#[test]
fn test_forward_traced_gelu_returns_trace() {
    let model = make_pygmy_model_gelu();
    let trace = model
        .forward_traced(&[1])
        .expect("forward_traced should succeed");
    assert_eq!(trace.logits.len(), 16);
    assert_eq!(trace.layer_activations.len(), 1);
    // All stats should be populated
    assert!(trace.embed_stats.count > 0);
    assert!(trace.final_norm_stats.count > 0);
    assert!(trace.logits_stats.count > 0);
}

// ============================================================================
// TracedForward trait impl
// ============================================================================

#[test]
fn test_traced_forward_trait_impl() {
    let mut model = make_pygmy_model();
    // Call through the trait
    let trace =
        TracedForward::forward_traced(&mut model, &[1]).expect("TracedForward should succeed");
    assert_eq!(trace.logits.len(), 16);
}

// ============================================================================
// AprTransformer::predict_next()
// ============================================================================

#[test]
fn test_predict_next_returns_token() {
    let model = make_pygmy_model();
    let next = model
        .predict_next(&[1])
        .expect("predict_next should succeed");
    assert!(next < 16, "Predicted token should be within vocab_size");
}

#[test]
fn test_predict_next_deterministic() {
    let model = make_pygmy_model();
    let next1 = model
        .predict_next(&[1])
        .expect("predict_next should succeed");
    let next2 = model
        .predict_next(&[1])
        .expect("predict_next should succeed");
    assert_eq!(
        next1, next2,
        "predict_next should be deterministic for same input"
    );
}

#[test]
fn test_predict_next_empty_error() {
    let model = make_pygmy_model();
    let result = model.predict_next(&[]);
    assert!(result.is_err());
}

// ============================================================================
// AprTransformer::generate()
// ============================================================================

#[test]
fn test_generate_produces_tokens() {
    let model = make_pygmy_model();
    let output = model.generate(&[1], 3).expect("generate should succeed");
    // Output includes prompt + generated tokens
    assert!(output.len() >= 2, "Should generate at least one token");
    assert!(
        output.len() <= 4,
        "Should generate at most 3 tokens + prompt"
    );
    assert_eq!(output[0], 1, "First token should be prompt");
}

#[test]
fn test_generate_stops_at_eos() {
    // Create model where token 2 (EOS) is always the highest logit
    let hidden_dim = 8;
    let vocab_size = 16;
    let config = AprTransformerConfig {
        architecture: "test-eos".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
    };
    let mut model = AprTransformer::new(config);

    // Bias lm_head so token 2 always wins
    // Set lm_head_bias so index 2 has a very large value
    let mut bias = vec![0.0f32; vocab_size];
    bias[2] = 100.0; // EOS token = 2
    model.lm_head_bias = Some(bias);

    let output = model.generate(&[1], 10).expect("generate should succeed");
    // Should stop early due to EOS
    assert!(
        output.len() <= 3,
        "Should stop after generating EOS token 2. Got len={}",
        output.len()
    );
    // Last generated token should be EOS (2)
    assert_eq!(*output.last().expect("output should not be empty"), 2);
}

#[test]
fn test_generate_max_tokens_limit() {
    let model = make_pygmy_model();
    let output = model.generate(&[1], 5).expect("generate should succeed");
    // Output length = prompt(1) + generated(up to 5)
    assert!(output.len() <= 6);
}

// ============================================================================
// AprTransformer::forward_with_cache()
// ============================================================================

#[test]
fn test_forward_with_cache_first_token() {
    let model = make_pygmy_model();
    let mut cache = AprKVCache::new(&model.config);
    let logits = model
        .forward_with_cache(1, &mut cache, 0)
        .expect("forward_with_cache should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_with_cache_multiple_positions() {
    let model = make_pygmy_model();
    let mut cache = AprKVCache::new(&model.config);

    // Process first token
    let logits0 = model
        .forward_with_cache(1, &mut cache, 0)
        .expect("forward_with_cache pos=0 should succeed");
    assert_eq!(logits0.len(), 16);

    // Process second token (uses cached KV from first)
    let logits1 = model
        .forward_with_cache(2, &mut cache, 1)
        .expect("forward_with_cache pos=1 should succeed");
    assert_eq!(logits1.len(), 16);

    // Cache should have 2 positions
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_forward_with_cache_gelu_path() {
    let model = make_pygmy_model_gelu();
    let mut cache = AprKVCache::new(&model.config);
    let logits = model
        .forward_with_cache(1, &mut cache, 0)
        .expect("forward_with_cache gelu should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// AprTransformer::generate_with_cache()
// ============================================================================

#[test]
fn test_generate_with_cache_delegation() {
    let model = make_pygmy_model();
    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let output = model
        .generate_with_cache(&[1], &gen_config)
        .expect("generate_with_cache should succeed");
    assert!(output.len() >= 2);
    assert_eq!(output[0], 1);
}

#[test]
fn test_generate_with_cache_empty_prompt_error() {
    let model = make_pygmy_model();
    let gen_config = GenerateConfig::default();
    let result = model.generate_with_cache(&[], &gen_config);
    assert!(result.is_err());
}

// ============================================================================
// AprTransformer serialization (Serialize/Deserialize)
// ============================================================================

#[test]
fn test_apr_transformer_serde_roundtrip() {
    let model = make_pygmy_model();
    let json = serde_json::to_vec(&model).expect("serialize should succeed");
    let deserialized: AprTransformer =
        serde_json::from_slice(&json).expect("deserialize should succeed");

    assert_eq!(deserialized.config.hidden_dim, model.config.hidden_dim);
    assert_eq!(deserialized.config.num_layers, model.config.num_layers);
    assert_eq!(deserialized.config.vocab_size, model.config.vocab_size);
    assert_eq!(
        deserialized.token_embedding.len(),
        model.token_embedding.len()
    );
    assert_eq!(deserialized.layers.len(), model.layers.len());
    assert_eq!(
        deserialized.lm_head_weight.len(),
        model.lm_head_weight.len()
    );
}

#[test]
fn test_apr_transformer_debug() {
    let model = make_pygmy_model();
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("AprTransformer"));
}

#[test]
fn test_apr_transformer_clone() {
    let model = make_pygmy_model();
    let cloned = model.clone();
    assert_eq!(cloned.config.hidden_dim, model.config.hidden_dim);
    assert_eq!(cloned.token_embedding.len(), model.token_embedding.len());
    assert_eq!(cloned.layers.len(), model.layers.len());
}

// ============================================================================
// from_apr_bytes() error cases
// ============================================================================

#[test]
fn test_from_apr_bytes_too_small() {
    let result = AprTransformer::from_apr_bytes(&[0; 32]);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("too small"),
        "Error should mention size: {err_msg}"
    );
}

#[test]
fn test_from_apr_bytes_bad_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX");
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("magic") || err_msg.contains("Invalid"),
        "Error should mention magic: {err_msg}"
    );
}

#[test]
fn test_from_apr_bytes_valid_magic_apr0() {
    // Test APR\0 magic (version 0) with minimal valid structure
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(b"APR\0");
    // tensor_count = 0
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    // metadata_offset = 64, metadata_size = 2 (valid JSON "{}")
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    // tensor_index_offset = 66
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    // data_offset = 66
    data[32..40].copy_from_slice(&66u64.to_le_bytes());
    // Write metadata "{}" at offset 64
    data[64] = b'{';
    data[65] = b'}';

    // This should parse without error (uses defaults for missing tensors)
    // But will fail because no embedding tensor found
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("embedding") || err_msg.contains("FATAL"),
        "Should fail with missing embedding: {err_msg}"
    );
}

#[test]
fn test_from_apr_bytes_metadata_beyond_file() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count=0
                                                      // metadata_offset = 64, metadata_size = 9999 (beyond file)
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&9999u32.to_le_bytes());
    data[24..32].copy_from_slice(&64u64.to_le_bytes());
    data[32..40].copy_from_slice(&64u64.to_le_bytes());

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("beyond") || err_msg.contains("Metadata"),
        "Should fail with metadata beyond file: {err_msg}"
    );
}

#[test]
fn test_from_apr_bytes_valid_magic_apr2() {
    // Test APR2 magic (version 2) -- same error expected for no embedding
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(b"APR2");
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    data[32..40].copy_from_slice(&66u64.to_le_bytes());
    data[64] = b'{';
    data[65] = b'}';

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err()); // No embedding tensor
}

// ============================================================================
// from_apr_bytes() with embedded tensors (F32 dtype)
// ============================================================================

/// Build a minimal valid APR v2 binary with embedding + lm_head + layer tensors
fn build_minimal_apr_bytes() -> Vec<u8> {
    let hidden_dim: usize = 4;
    let vocab_size: usize = 4;
    let num_layers: usize = 1;
    let num_heads: usize = 2;
    let num_kv_heads: usize = 2;
    let intermediate_dim: usize = 8;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let _qkv_out_dim = hidden_dim + 2 * kv_dim;

    // Metadata JSON
    let metadata = format!(
        r#"{{"hidden_size":{hidden_dim},"num_hidden_layers":{num_layers},"num_attention_heads":{num_heads},"num_key_value_heads":{num_kv_heads},"vocab_size":{vocab_size},"intermediate_size":{intermediate_dim},"rope_theta":10000.0,"rms_norm_eps":0.00001,"context_length":64}}"#
    );
    let metadata_bytes = metadata.as_bytes();

    // Define tensors we need
    struct TensorDef {
        name: String,
        dims: Vec<usize>,
        dtype: u8,
    }

    let tensor_defs = vec![
        TensorDef {
            name: "model.embed_tokens.weight".to_string(),
            dims: vec![vocab_size, hidden_dim],
            dtype: 0, // F32
        },
        TensorDef {
            name: "lm_head.weight".to_string(),
            dims: vec![vocab_size, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.norm.weight".to_string(),
            dims: vec![hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dims: vec![hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dims: vec![hidden_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.self_attn.k_proj.weight".to_string(),
            dims: vec![kv_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.self_attn.v_proj.weight".to_string(),
            dims: vec![kv_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.self_attn.o_proj.weight".to_string(),
            dims: vec![hidden_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.mlp.up_proj.weight".to_string(),
            dims: vec![intermediate_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.mlp.down_proj.weight".to_string(),
            dims: vec![hidden_dim, intermediate_dim],
            dtype: 0,
        },
    ];

    // Calculate tensor sizes
    let mut tensor_data_parts: Vec<Vec<u8>> = Vec::new();
    let mut running_offset: usize = 0;

    struct TensorEntry {
        name: String,
        dims: Vec<usize>,
        dtype: u8,
        offset: usize,
        size: usize,
    }

    let mut entries = Vec::new();
    for def in &tensor_defs {
        let num_elems: usize = def.dims.iter().product();
        let byte_size = num_elems * 4; // F32 = 4 bytes
        let data: Vec<u8> = (0..num_elems)
            .flat_map(|i| {
                let val = ((i % 7) as f32 - 3.0) * 0.01;
                val.to_le_bytes().to_vec()
            })
            .collect();
        entries.push(TensorEntry {
            name: def.name.clone(),
            dims: def.dims.clone(),
            dtype: def.dtype,
            offset: running_offset,
            size: byte_size,
        });
        tensor_data_parts.push(data);
        running_offset += byte_size;
    }

    // Build tensor index
    let mut tensor_index = Vec::new();
    for entry in &entries {
        let name_bytes = entry.name.as_bytes();
        tensor_index.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        tensor_index.extend_from_slice(name_bytes);
        tensor_index.push(entry.dtype);
        tensor_index.push(entry.dims.len() as u8);
        for &dim in &entry.dims {
            tensor_index.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        tensor_index.extend_from_slice(&(entry.offset as u64).to_le_bytes());
        tensor_index.extend_from_slice(&(entry.size as u64).to_le_bytes());
    }

    // Layout: Header (64) | Metadata | Tensor Index | Tensor Data
    let metadata_offset: usize = 64;
    let metadata_size = metadata_bytes.len();
    let tensor_index_offset = metadata_offset + metadata_size;
    let data_offset = tensor_index_offset + tensor_index.len();

    let mut bytes = vec![0u8; 64]; // header
    bytes[0..4].copy_from_slice(b"APR\0");
    // tensor_count
    bytes[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    // metadata_offset
    bytes[12..20].copy_from_slice(&(metadata_offset as u64).to_le_bytes());
    // metadata_size
    bytes[20..24].copy_from_slice(&(metadata_size as u32).to_le_bytes());
    // tensor_index_offset
    bytes[24..32].copy_from_slice(&(tensor_index_offset as u64).to_le_bytes());
    // data_offset
    bytes[32..40].copy_from_slice(&(data_offset as u64).to_le_bytes());

    // Metadata
    bytes.extend_from_slice(metadata_bytes);
    // Tensor index
    bytes.extend_from_slice(&tensor_index);
    // Tensor data
    for part in &tensor_data_parts {
        bytes.extend_from_slice(part);
    }

    bytes
}

#[test]
fn test_from_apr_bytes_valid_minimal() {
    let data = build_minimal_apr_bytes();
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_ok(),
        "Should parse valid APR bytes: {:?}",
        result.err()
    );

    let model = result.expect("parse should succeed");
    assert_eq!(model.config.hidden_dim, 4);
    assert_eq!(model.config.num_layers, 1);
    assert_eq!(model.config.num_heads, 2);
    assert_eq!(model.config.num_kv_heads, 2);
    assert_eq!(model.config.vocab_size, 4);
    assert_eq!(model.config.intermediate_dim, 8);
    assert_eq!(model.token_embedding.len(), 4 * 4); // vocab_size * hidden_dim
    assert_eq!(model.layers.len(), 1);
    assert_eq!(model.lm_head_weight.len(), 4 * 4); // vocab_size * hidden_dim
}

#[test]
fn test_from_apr_bytes_then_forward() {
    let data = build_minimal_apr_bytes();
    let model = AprTransformer::from_apr_bytes(&data).expect("parse should succeed");
    let logits = model.forward(&[0]).expect("forward should succeed");
    assert_eq!(logits.len(), 4); // vocab_size=4
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Multi-layer model tests (more coverage of layer iteration)
// ============================================================================

#[test]
fn test_forward_two_layer_model() {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 8;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let config = AprTransformerConfig {
        architecture: "test-2layer".to_string(),
        hidden_dim,
        num_layers: 2,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let make_layer = || AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; qkv_out_dim * hidden_dim],
        qkv_bias: None,
        attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.01; intermediate_dim * hidden_dim]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
    };

    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok * hidden_dim + d) as f32) * 0.001;
        }
    }

    let model = AprTransformer {
        config,
        token_embedding,
        layers: vec![make_layer(), make_layer()],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let logits = model.forward(&[0, 1]).expect("forward should succeed");
    assert_eq!(logits.len(), vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// forward_with_cache with GELU path and subsequent tokens
// ============================================================================

#[test]
fn test_forward_with_cache_three_tokens_sequential() {
    let model = make_pygmy_model();
    let mut cache = AprKVCache::new(&model.config);

    for pos in 0..3 {
        let logits = model
            .forward_with_cache(pos as u32, &mut cache, pos)
            .expect("forward_with_cache should succeed");
        assert_eq!(logits.len(), 16);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "Logits at pos={pos} should all be finite"
        );
    }
    assert_eq!(cache.len(), 3);
}

#[test]
fn test_forward_with_cache_gelu_multiple_positions() {
    let model = make_pygmy_model_gelu();
    let mut cache = AprKVCache::new(&model.config);

    let logits0 = model
        .forward_with_cache(0, &mut cache, 0)
        .expect("forward_with_cache pos=0 should succeed");
    let logits1 = model
        .forward_with_cache(1, &mut cache, 1)
        .expect("forward_with_cache pos=1 should succeed");

    assert_eq!(logits0.len(), 16);
    assert_eq!(logits1.len(), 16);
    // Different inputs should typically produce different outputs
    // (though not guaranteed with zero-ish weights)
}

// ============================================================================
// forward_with_cache without FFN norm (exercises else branch)
// ============================================================================

#[test]
fn test_forward_with_cache_no_ffn_norm() {
    let model = make_pygmy_model_gelu(); // This has ffn_norm_weight = None
    let mut cache = AprKVCache::new(&model.config);
    let logits = model
        .forward_with_cache(5, &mut cache, 0)
        .expect("should succeed without ffn_norm");
    assert_eq!(logits.len(), 16);
}

// ============================================================================
// Edge cases: forward parity between forward() and forward_traced()
// ============================================================================

#[test]
fn test_forward_and_forward_traced_logits_match() {
    let model = make_pygmy_model();
    let logits_forward = model.forward(&[1]).expect("forward should succeed");
    let trace = model
        .forward_traced(&[1])
        .expect("forward_traced should succeed");

    assert_eq!(logits_forward.len(), trace.logits.len());
    for (i, (a, b)) in logits_forward.iter().zip(trace.logits.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "Logit mismatch at index {i}: forward={a}, traced={b}"
        );
    }
}

#[test]
fn test_forward_and_forward_traced_gelu_logits_match() {
    let model = make_pygmy_model_gelu();
    let logits_forward = model.forward(&[0]).expect("forward should succeed");
    let trace = model
        .forward_traced(&[0])
        .expect("forward_traced should succeed");

    assert_eq!(logits_forward.len(), trace.logits.len());
    for (i, (a, b)) in logits_forward.iter().zip(trace.logits.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "Logit mismatch at index {i}: forward={a}, traced={b}"
        );
    }
}
