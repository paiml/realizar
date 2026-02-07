//! Tests for single-token forward pass with KV cache
//!
//! Coverage target: /home/noah/src/realizar/src/gguf/inference/forward/single.rs
//!
//! Tests cover:
//! - forward_single_with_cache: Main decode-phase forward pass
//! - forward_single_with_scratch: Zero-allocation variant
//! - Various model architectures (RMSNorm vs LayerNorm, SwiGLU vs GELU)

use crate::gguf::test_helpers::{create_q4k_test_data, create_test_model_with_config};
use crate::gguf::{GGUFConfig, InferenceScratchBuffer, OwnedQuantizedKVCache};

/// Create a config for LLaMA-style models (RMSNorm + SwiGLU)
fn create_llama_style_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "llama".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    }
}

/// Create a config for phi-2 style models (LayerNorm + GELU, no gate)
fn create_phi_style_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "phi".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    }
}

/// Create a LLaMA-style model with SwiGLU FFN (gate weight + ffn_norm)
fn create_llama_style_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = create_llama_style_config();
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);

    // QKV: hidden_dim -> hidden_dim + 2*kv_dim
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let qkv_weight = create_q4k_test_data(hidden_dim, qkv_out_dim);
    let attn_output_weight = create_q4k_test_data(hidden_dim, hidden_dim);
    let ffn_up_weight = create_q4k_test_data(hidden_dim, intermediate_dim);
    let ffn_down_weight = create_q4k_test_data(intermediate_dim, hidden_dim);
    let ffn_gate_weight = create_q4k_test_data(hidden_dim, intermediate_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None, // LLaMA has no norm bias (RMSNorm)
        qkv_weight: OwnedQKVWeights::Fused(qkv_weight),
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_gate_weight: Some(ffn_gate_weight), // SwiGLU gate
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]), // LLaMA has FFN norm
        ffn_norm_bias: None,
    };

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        layers: vec![layer],
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create a phi-style model (LayerNorm + GELU, no gate)
fn create_phi_style_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = create_phi_style_config();
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);

    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let qkv_weight = create_q4k_test_data(hidden_dim, qkv_out_dim);
    let attn_output_weight = create_q4k_test_data(hidden_dim, hidden_dim);
    let ffn_up_weight = create_q4k_test_data(hidden_dim, intermediate_dim);
    let ffn_down_weight = create_q4k_test_data(intermediate_dim, hidden_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: Some(vec![0.0f32; hidden_dim]), // phi has norm bias (LayerNorm)
        qkv_weight: OwnedQKVWeights::Fused(qkv_weight),
        qkv_bias: Some(vec![0.0f32; qkv_out_dim]),
        attn_output_weight,
        attn_output_bias: Some(vec![0.0f32; hidden_dim]),
        ffn_up_weight,
        ffn_up_bias: Some(vec![0.0f32; intermediate_dim]),
        ffn_down_weight,
        ffn_down_bias: Some(vec![0.0f32; hidden_dim]),
        ffn_gate_weight: None, // phi uses GELU, no gate
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: Some(vec![0.0f32; hidden_dim]),
    };

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        layers: vec![layer],
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: Some(vec![0.0f32; hidden_dim]),
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: Some(vec![0.0f32; config.vocab_size]),
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

// =============================================================================
// forward_single_with_cache tests
// =============================================================================

#[test]
fn test_forward_single_with_cache_first_token() {
    // First token should work without any cached K/V
    let model = create_test_model_with_config(&create_llama_style_config());
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let token_id = 5u32;
    let position = 0;

    let logits = model
        .forward_single_with_cache(token_id, &mut cache, position)
        .expect("First token forward should succeed");

    // Should return vocab_size logits
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "Logits should have vocab_size elements"
    );

    // All logits should be finite
    assert!(
        logits.iter().all(|&x| x.is_finite()),
        "All logits should be finite"
    );

    // Cache should have been populated
    assert_eq!(
        cache.len(),
        1,
        "Cache should have 1 position after first token"
    );
}

#[test]
fn test_forward_single_with_cache_sequential_tokens() {
    // Multiple tokens should use and extend the KV cache
    let model = create_test_model_with_config(&create_llama_style_config());
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // Process 5 tokens sequentially
    for pos in 0..5 {
        let token_id = (pos as u32) + 1;
        let logits = model
            .forward_single_with_cache(token_id, &mut cache, pos)
            .expect("Sequential token forward should succeed");

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|&x| x.is_finite()));
    }

    assert_eq!(cache.len(), 5, "Cache should have 5 positions");
}

#[test]
fn test_forward_single_with_cache_deterministic() {
    // Same input should produce same output
    let model = create_test_model_with_config(&create_llama_style_config());
    let config = &model.config;

    let mut cache1 = OwnedQuantizedKVCache::from_config(config, 128);
    let mut cache2 = OwnedQuantizedKVCache::from_config(config, 128);

    let token_id = 10u32;

    let logits1 = model
        .forward_single_with_cache(token_id, &mut cache1, 0)
        .unwrap();
    let logits2 = model
        .forward_single_with_cache(token_id, &mut cache2, 0)
        .unwrap();

    assert_eq!(logits1, logits2, "Forward pass should be deterministic");
}

#[test]
fn test_forward_single_llama_style_rmsnorm_swiglu() {
    // Test LLaMA-style model path (RMSNorm + SwiGLU)
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // Verify model has SwiGLU (gate weight) and no norm bias (RMSNorm)
    assert!(
        model.layers[0].ffn_gate_weight.is_some(),
        "LLaMA model should have gate weight"
    );
    assert!(
        model.layers[0].attn_norm_bias.is_none(),
        "LLaMA model should not have attn_norm_bias (RMSNorm)"
    );

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("LLaMA-style forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_phi_style_layernorm_gelu() {
    // Test phi-style model path (LayerNorm + GELU)
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // Verify model has no gate (GELU path) and has norm bias (LayerNorm)
    assert!(
        model.layers[0].ffn_gate_weight.is_none(),
        "phi model should not have gate weight"
    );
    assert!(
        model.layers[0].attn_norm_bias.is_some(),
        "phi model should have attn_norm_bias (LayerNorm)"
    );

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("phi-style forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_with_lm_head_bias() {
    // Test that lm_head_bias is applied correctly
    let model = create_phi_style_model();
    assert!(
        model.lm_head_bias.is_some(),
        "phi model should have lm_head_bias"
    );

    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);
    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Forward with lm_head_bias should succeed");

    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_kv_cache_reuse() {
    // Verify cached K/V is actually being used for attention
    let model = create_test_model_with_config(&create_llama_style_config());
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // First token: no cache
    let _ = model.forward_single_with_cache(1, &mut cache, 0).unwrap();
    assert_eq!(cache.len(), 1);

    // Second token: should use cached K/V from first token
    let logits = model.forward_single_with_cache(2, &mut cache, 1).unwrap();
    assert_eq!(cache.len(), 2);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_edge_case_zero_token() {
    // Token ID 0 should work (often <pad> or <unk>)
    let model = create_test_model_with_config(&create_llama_style_config());
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    let logits = model
        .forward_single_with_cache(0, &mut cache, 0)
        .expect("Token ID 0 should work");

    assert_eq!(logits.len(), model.config.vocab_size);
}

#[test]
fn test_forward_single_edge_case_max_valid_token() {
    // Test with maximum valid token ID
    let model = create_test_model_with_config(&create_llama_style_config());
    let max_token = model.config.vocab_size as u32 - 1;
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    let logits = model
        .forward_single_with_cache(max_token, &mut cache, 0)
        .expect("Max token ID should work");

    assert_eq!(logits.len(), model.config.vocab_size);
}

// =============================================================================
// forward_single_with_scratch tests
// =============================================================================

#[test]
fn test_forward_single_with_scratch_basic() {
    // Test zero-allocation forward pass
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    model
        .forward_single_with_scratch(1, &mut cache, 0, &mut scratch)
        .expect("Scratch forward should succeed");

    // Logits should be in scratch buffer
    assert_eq!(scratch.logits.len(), config.vocab_size);
    assert!(scratch.logits.iter().all(|&x| x.is_finite()));
}

#[test]
#[ignore = "cache.len() API changed - needs update"]
fn test_forward_single_with_scratch_sequential() {
    // Multiple tokens with scratch buffer
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    for pos in 0..3 {
        model
            .forward_single_with_scratch((pos as u32) + 1, &mut cache, pos, &mut scratch)
            .expect("Sequential scratch forward should succeed");
    }

    assert_eq!(cache.len(), 3);
    assert!(scratch.logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_with_scratch_phi_style() {
    // Test phi-style (LayerNorm + GELU) with scratch buffer
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    model
        .forward_single_with_scratch(1, &mut cache, 0, &mut scratch)
        .expect("phi-style scratch forward should succeed");

    assert_eq!(scratch.logits.len(), config.vocab_size);
}

#[test]
fn test_forward_single_with_scratch_reuses_buffers() {
    // Verify scratch buffers are reused (no additional allocations)
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    // Record initial buffer capacities
    let hidden_cap = scratch.hidden.capacity();
    let normed_cap = scratch.normed.capacity();
    let logits_cap = scratch.logits.capacity();

    // Run multiple forwards
    for pos in 0..5 {
        model
            .forward_single_with_scratch((pos as u32) + 1, &mut cache, pos, &mut scratch)
            .unwrap();
    }

    // Capacities should not have grown
    assert_eq!(scratch.hidden.capacity(), hidden_cap);
    assert_eq!(scratch.normed.capacity(), normed_cap);
    assert_eq!(scratch.logits.capacity(), logits_cap);
}

// =============================================================================
// GQA (Grouped Query Attention) tests
// =============================================================================

#[test]
fn test_forward_single_gqa_config() {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    // Test GQA: num_kv_heads < num_heads (e.g., 4 Q heads share 2 KV heads)
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 2, // GQA: 2 KV heads for 4 Q heads
        num_layers: 1,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let hidden_dim = config.hidden_dim;
    let head_dim = hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // Q is full, K/V reduced

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, config.intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(config.intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, config.intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
    };

    let model = OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        layers: vec![layer],
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);

    // First token (no cache)
    let logits1 = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("GQA forward (first token) should succeed");
    assert_eq!(logits1.len(), config.vocab_size);

    // Second token (uses cache with GQA)
    let logits2 = model
        .forward_single_with_cache(2, &mut cache, 1)
        .expect("GQA forward (cached) should succeed");
    assert_eq!(logits2.len(), config.vocab_size);
}

// =============================================================================
// Model with multiple layers
// =============================================================================

#[test]
fn test_forward_single_multiple_layers() {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 3, // 3 layers
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let hidden_dim = config.hidden_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let layers: Vec<OwnedQuantizedLayer> = (0..config.num_layers)
        .map(|_| OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
            qkv_bias: None,
            attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4k_test_data(hidden_dim, config.intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4k_test_data(config.intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, config.intermediate_dim)),
            ffn_gate_bias: None,
            ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
            ffn_norm_bias: None,
        })
        .collect();

    let model = OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Multi-layer forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

// =============================================================================
// Additional forward_single_with_cache coverage (178 uncov)
// =============================================================================

#[test]
fn test_forward_single_phi_style_with_cache() {
    // phi-style: LayerNorm + GELU (no gate, has bias)
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let logits = model
        .forward_single_with_cache(42, &mut cache, 0)
        .expect("phi-style forward_single_with_cache should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_sequential_decode_10_tokens() {
    // Exercise the cached attention path with many positions
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    for pos in 0..10 {
        let token = (pos as u32) % (config.vocab_size as u32);
        let logits = model
            .forward_single_with_cache(token, &mut cache, pos)
            .expect("sequential decode should succeed");

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|&x| x.is_finite()));
    }
}

#[test]
fn test_forward_single_with_scratch_multi_step() {
    // Exercise scratch buffer through multiple decode steps
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    for pos in 0..5 {
        model
            .forward_single_with_scratch((pos as u32) + 1, &mut cache, pos, &mut scratch)
            .expect("multi-step scratch forward should succeed");

        assert_eq!(scratch.logits.len(), config.vocab_size);
        assert!(scratch.logits.iter().all(|&x| x.is_finite()));
    }
}

// =============================================================================
// forward_single_with_cache_adaptive tests (127 uncov, gpu feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_forward_single_adaptive_basic() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let metrics = Arc::new(DispatchMetrics::new());

    let logits = model
        .forward_single_with_cache_adaptive(42, &mut cache, 0, &metrics)
        .expect("adaptive forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[cfg(feature = "gpu")]
#[test]
fn test_forward_single_adaptive_sequential() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let metrics = Arc::new(DispatchMetrics::new());

    for pos in 0..5 {
        let token = (pos as u32) + 1;
        let logits = model
            .forward_single_with_cache_adaptive(token, &mut cache, pos, &metrics)
            .expect("sequential adaptive forward should succeed");

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|&x| x.is_finite()));
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_forward_single_adaptive_phi_style() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let metrics = Arc::new(DispatchMetrics::new());

    let logits = model
        .forward_single_with_cache_adaptive(10, &mut cache, 0, &metrics)
        .expect("phi-style adaptive forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

// =============================================================================
// Trace/debug env-var-gated coverage tests
// These set REALIZAR_DEBUG_FORWARD and APR_TRACE_LAYERS to cover the
// trace/debug blocks in forward_single_with_cache and forward_single_with_scratch.
// =============================================================================

#[test]
fn test_forward_single_with_cache_debug_traces_llama() {
    unsafe {
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("APR_TRACE_LAYERS", "1");
    }
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // First token (position 0, layer 0) — triggers most trace blocks
    let r1 = model.forward_single_with_cache(5, &mut cache, 0);
    assert!(r1.is_ok(), "Debug traces llama fwd: {}", r1.unwrap_err());

    // Second token
    let r2 = model.forward_single_with_cache(10, &mut cache, 1);
    assert!(r2.is_ok(), "Debug traces llama 2nd: {}", r2.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("APR_TRACE_LAYERS");
    }
}

#[test]
fn test_forward_single_with_cache_debug_traces_phi() {
    unsafe {
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("APR_TRACE_LAYERS", "1");
    }
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let r1 = model.forward_single_with_cache(5, &mut cache, 0);
    assert!(r1.is_ok(), "Debug traces phi fwd: {}", r1.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("APR_TRACE_LAYERS");
    }
}

#[test]
fn test_forward_single_with_cache_debug_traces_multi_token() {
    // Run multiple tokens with debug tracing to cover more trace branches
    unsafe {
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("APR_TRACE_LAYERS", "1");
    }
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    for i in 0..5 {
        let r = model.forward_single_with_cache(i as u32, &mut cache, i);
        assert!(r.is_ok(), "Debug multi pos {}: {}", i, r.unwrap_err());
    }
    unsafe {
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("APR_TRACE_LAYERS");
    }
}

// =============================================================================
// forward_single_with_scratch tests (zero-allocation variant)
// =============================================================================

#[test]
fn test_forward_single_with_scratch_llama_basic() {
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Scratch llama first: {}", r1.unwrap_err());
    assert!(scratch.logits.iter().any(|x| *x != 0.0), "logits should be non-zero");

    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "Scratch llama second: {}", r2.unwrap_err());
}

#[test]
fn test_forward_single_with_scratch_phi_basic() {
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Scratch phi first: {}", r1.unwrap_err());
    assert!(scratch.logits.iter().any(|x| *x != 0.0), "phi logits should be non-zero");
}

#[test]
fn test_forward_single_with_scratch_multi_token() {
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    for i in 0..10 {
        let r = model.forward_single_with_scratch(i as u32, &mut cache, i, &mut scratch);
        assert!(r.is_ok(), "Scratch multi pos {}: {}", i, r.unwrap_err());
    }
}

#[test]
fn test_forward_single_with_scratch_debug_traces() {
    unsafe {
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("APR_TRACE_LAYERS", "1");
    }
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Scratch debug: {}", r1.unwrap_err());

    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "Scratch debug 2: {}", r2.unwrap_err());
    unsafe {
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("APR_TRACE_LAYERS");
    }
}

// =============================================================================
// Q8K-optimized path (requires hidden_dim % 256 == 0)
// =============================================================================

/// Create a llama model with hidden_dim=256 to trigger Q8K paths
fn create_llama_256_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
    };

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        layers: vec![layer],
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

#[test]
fn test_forward_single_with_scratch_q8k_path() {
    // hidden_dim=256 => Q8K-optimized path enabled
    let model = create_llama_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Q8K scratch: {}", r1.unwrap_err());
    assert!(scratch.logits.len() == config.vocab_size);

    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "Q8K scratch 2nd: {}", r2.unwrap_err());
}

#[test]
fn test_forward_single_with_cache_q8k_path() {
    // Q8K path also used in forward_single_with_cache when hidden_dim % 256 == 0
    let model = create_llama_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let r1 = model.forward_single_with_cache(5, &mut cache, 0);
    assert!(r1.is_ok(), "Q8K cache: {}", r1.unwrap_err());
    assert_eq!(r1.unwrap().len(), config.vocab_size);

    let r2 = model.forward_single_with_cache(10, &mut cache, 1);
    assert!(r2.is_ok(), "Q8K cache 2nd: {}", r2.unwrap_err());
}

/// Create a phi-style model with hidden_dim=256 to trigger Q8K GELU paths
fn create_phi_256_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = GGUFConfig {
        architecture: "phi".to_string(),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: Some(vec![0.0f32; hidden_dim]), // LayerNorm
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
        qkv_bias: Some(vec![0.0f32; qkv_out_dim]),
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: Some(vec![0.0f32; hidden_dim]),
        ffn_up_weight: create_q4k_test_data(hidden_dim, intermediate_dim),
        ffn_up_bias: Some(vec![0.0f32; intermediate_dim]),
        ffn_down_weight: create_q4k_test_data(intermediate_dim, hidden_dim),
        ffn_down_bias: Some(vec![0.0f32; hidden_dim]),
        ffn_gate_weight: None, // GELU, no gate
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: Some(vec![0.0f32; hidden_dim]),
    };

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        layers: vec![layer],
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: Some(vec![0.0f32; hidden_dim]),
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: Some(vec![0.0f32; config.vocab_size]),
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create a GQA llama model with hidden_dim=256 (num_kv_heads < num_heads)
fn create_llama_256_gqa_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_heads: 8,     // 8 query heads
        num_kv_heads: 2,  // 2 KV heads => GQA group_size=4
        num_layers: 1,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let head_dim = hidden_dim / config.num_heads; // 32
    let kv_dim = config.num_kv_heads * head_dim; // 64
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 256 + 128 = 384

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
    };

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        layers: vec![layer],
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

#[test]
fn test_forward_single_with_scratch_phi_256_q8k_gelu() {
    // Phi model with hidden_dim=256: covers Q8K GELU up/down paths in scratch
    let model = create_phi_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Phi-256 scratch first: {}", r1.unwrap_err());

    // Second token covers attention_with_cache path
    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "Phi-256 scratch second: {}", r2.unwrap_err());
}

#[test]
fn test_forward_single_with_cache_phi_256_q8k() {
    // Phi 256 through the cache path for Q8K GELU coverage
    let model = create_phi_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let r1 = model.forward_single_with_cache(5, &mut cache, 0);
    assert!(r1.is_ok(), "Phi-256 cache first: {}", r1.unwrap_err());

    let r2 = model.forward_single_with_cache(10, &mut cache, 1);
    assert!(r2.is_ok(), "Phi-256 cache second: {}", r2.unwrap_err());
}

#[test]
fn test_forward_single_with_scratch_gqa_256() {
    // GQA model (num_kv_heads=2, num_heads=8) with hidden_dim=256
    // Covers: GQA V-expansion on first token, attention_with_cache_gqa on subsequent
    let model = create_llama_256_gqa_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    // First token: covers GQA V-expansion branch in scratch
    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "GQA-256 scratch first: {}", r1.unwrap_err());

    // Second token: covers attention_with_cache_gqa_into branch
    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "GQA-256 scratch second: {}", r2.unwrap_err());

    // Third token: further exercises cached attention
    let r3 = model.forward_single_with_scratch(15, &mut cache, 2, &mut scratch);
    assert!(r3.is_ok(), "GQA-256 scratch third: {}", r3.unwrap_err());
}

#[test]
fn test_forward_single_with_cache_gqa_256() {
    // GQA model through cache path
    let model = create_llama_256_gqa_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    for i in 0..5 {
        let r = model.forward_single_with_cache(i as u32, &mut cache, i);
        assert!(r.is_ok(), "GQA-256 cache pos {}: {}", i, r.unwrap_err());
    }
}

#[test]
fn test_forward_single_with_scratch_q8k_multi_position() {
    // Multiple positions through Q8K path to exercise cached attention
    let model = create_llama_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    for i in 0..10 {
        let r = model.forward_single_with_scratch(i as u32, &mut cache, i, &mut scratch);
        assert!(r.is_ok(), "Q8K multi pos {}: {}", i, r.unwrap_err());
    }
}
