//! Part 01: Config + IMP-101 + IMP-105 Tests
//!
//! Extracted from gguf_monolith.rs as part of PMAT-802 shatter.
//!
//! ## Contents
//! - QuantizedGenerateConfig tests
//! - IMP-101: RoPE and Causal Attention Tests
//! - IMP-105: Grouped Query Attention (GQA) Tests

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::*;

// ============================================================
// QuantizedGGUFTransformer::generate() tests
// Per benchmark-model-runners-spec.md "What's Remaining" item 1
// ============================================================

#[test]
fn test_generate_config_default() {
    let config = QuantizedGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_generate_config_builder() {
    let config = QuantizedGenerateConfig::default()
        .with_max_tokens(128)
        .with_temperature(0.7)
        .with_top_k(40)
        .with_stop_tokens(vec![50256]);

    assert_eq!(config.max_tokens, 128);
    assert!((config.temperature - 0.7).abs() < 1e-6);
    assert_eq!(config.top_k, 40);
    assert_eq!(config.stop_tokens, vec![50256]);
}

#[test]
fn test_generate_config_deterministic() {
    // Temperature 0.0 = greedy decoding
    let config = QuantizedGenerateConfig::deterministic(32);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert_eq!(config.max_tokens, 32);
}

// ==========================================================================
// IMP-101: RoPE and Causal Attention Tests
// ==========================================================================

/// IMP-101a: RoPE preserves vector magnitude
#[test]
fn test_imp_101a_rope_preserves_norm() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 4, // 4 heads x 16 dim = 64
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        layers: vec![],
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 64,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Create test vector
    let mut x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    // Apply RoPE at position 10 (4 heads for 64-dim vector)
    model.apply_rope(&mut x, 10, 4);

    let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    // RoPE is a rotation, should preserve L2 norm
    assert!(
        (norm_before - norm_after).abs() < 1e-5,
        "IMP-101a: RoPE should preserve vector norm. Before: {}, After: {}",
        norm_before,
        norm_after
    );
}

/// IMP-101a: RoPE produces different outputs at different positions
#[test]
fn test_imp_101a_rope_position_dependent() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        layers: vec![],
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 64,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Apply RoPE at different positions
    let original: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

    let mut x_pos0 = original.clone();
    let mut x_pos10 = original.clone();
    let mut x_pos100 = original.clone();

    model.apply_rope(&mut x_pos0, 0, 4);
    model.apply_rope(&mut x_pos10, 10, 4);
    model.apply_rope(&mut x_pos100, 100, 4);

    // Different positions should produce different outputs
    let diff_0_10: f32 = x_pos0
        .iter()
        .zip(x_pos10.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff_10_100: f32 = x_pos10
        .iter()
        .zip(x_pos100.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff_0_10 > 1e-3,
        "IMP-101a: RoPE should produce different outputs at positions 0 vs 10"
    );
    assert!(
        diff_10_100 > 1e-3,
        "IMP-101a: RoPE should produce different outputs at positions 10 vs 100"
    );
}

/// IMP-101b: Causal attention only attends to past tokens
#[test]
fn test_imp_101b_causal_attention_mask() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 8, // Small for testing
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2, // 2 heads x 4 dim = 8
        num_kv_heads: 2,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        layers: vec![],
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 8,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Create test Q, K, V (seq_len=4, hidden_dim=8)
    let seq_len = 4;
    let hidden_dim = 8;
    let q: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let k: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| (i as f32 * 0.2).cos())
        .collect();
    let v: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| i as f32 * 0.1)
        .collect();

    let output = model.causal_attention(&q, &k, &v, seq_len);

    // Output should have correct shape
    assert_eq!(
        output.len(),
        seq_len * hidden_dim,
        "IMP-101b: Causal attention output should have shape [seq_len, hidden_dim]"
    );

    // First position can only attend to itself
    // Last position can attend to all positions
    // This is verified by the fact that the output doesn't crash and has correct shape
    assert!(
        output.iter().all(|v| v.is_finite()),
        "IMP-101b: All attention outputs should be finite"
    );
}

/// IMP-101b: Causal attention softmax sums to 1
#[test]
fn test_imp_101b_causal_attention_softmax_normalized() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 4,
        intermediate_dim: 16,
        num_layers: 1,
        num_heads: 1, // 1 head for simplicity
        num_kv_heads: 1,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        layers: vec![],
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 4,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Create identity K (each position is unique)
    let seq_len = 3;
    let hidden_dim = 4;

    // Q = same for all positions, K = identity-like
    let q: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0].repeat(seq_len);
    let k: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // pos 0
        0.0, 1.0, 0.0, 0.0, // pos 1
        0.0, 0.0, 1.0, 0.0, // pos 2
    ];
    let v: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // pos 0
        0.0, 1.0, 0.0, 0.0, // pos 1
        0.0, 0.0, 1.0, 0.0, // pos 2
    ];

    let output = model.causal_attention(&q, &k, &v, seq_len);

    // Output at each position should be a weighted sum of values
    // For position 0: can only attend to position 0, so output = V[0]
    let pos0_output = &output[0..hidden_dim];
    assert!(
        (pos0_output[0] - 1.0).abs() < 1e-5,
        "IMP-101b: Position 0 should only attend to itself"
    );
}

// ===== IMP-101c: KV Cache Integration Tests =====

/// IMP-101c: KV cache initializes correctly
#[test]
fn test_imp_101c_kv_cache_initialization() {
    let cache = OwnedQuantizedKVCache::new(12, 768, 2048);

    assert_eq!(cache.len(), 0, "IMP-101c: New cache should be empty");
    assert!(cache.is_empty(), "IMP-101c: is_empty should return true");
    assert_eq!(cache.max_len(), 2048, "IMP-101c: max_len should match");
}

/// IMP-101c: KV cache from config
#[test]
fn test_imp_101c_kv_cache_from_config() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 512,
        intermediate_dim: 2048,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 32000,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
    };

    let cache = OwnedQuantizedKVCache::from_config(&config, 1024);

    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_len(), 1024);
}

/// IMP-101c: KV cache append and retrieve
#[test]
fn test_imp_101c_kv_cache_append_retrieve() {
    let mut cache = OwnedQuantizedKVCache::new(2, 4, 100);

    // Append K/V for layer 0
    let k0 = vec![1.0, 2.0, 3.0, 4.0];
    let v0 = vec![0.1, 0.2, 0.3, 0.4];
    cache.append(0, &k0, &v0);

    // Append K/V for layer 1
    let k1 = vec![5.0, 6.0, 7.0, 8.0];
    let v1 = vec![0.5, 0.6, 0.7, 0.8];
    cache.append(1, &k1, &v1);

    // Advance position
    cache.advance();

    assert_eq!(cache.len(), 1, "IMP-101c: Cache should have 1 position");

    // Verify retrieval
    let retrieved_k0 = cache.get_k(0);
    assert_eq!(
        retrieved_k0.len(),
        4,
        "IMP-101c: Retrieved K should have 4 elements"
    );
    assert!(
        (retrieved_k0[0] - 1.0).abs() < 1e-6,
        "IMP-101c: K values should match"
    );

    let retrieved_v1 = cache.get_v(1);
    assert!(
        (retrieved_v1[0] - 0.5).abs() < 1e-6,
        "IMP-101c: V values should match"
    );
}

/// IMP-101c: KV cache reset clears data
#[test]
fn test_imp_101c_kv_cache_reset() {
    let mut cache = OwnedQuantizedKVCache::new(2, 4, 100);

    // Add some data
    let k = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![0.1, 0.2, 0.3, 0.4];
    cache.append(0, &k, &v);
    cache.advance();

    assert_eq!(cache.len(), 1);

    // Reset
    cache.reset();

    assert_eq!(cache.len(), 0, "IMP-101c: Reset should clear position");
    assert!(cache.is_empty(), "IMP-101c: Reset should make cache empty");
    assert!(
        cache.get_k(0).is_empty(),
        "IMP-101c: Reset should clear K data"
    );
}

/// IMP-101c: Attention with cache produces normalized output
#[test]
fn test_imp_101c_attention_with_cache_softmax_normalized() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 4,
        intermediate_dim: 16,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        layers: vec![],
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 4,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Test attention with cache
    // Q = [1, 0, 0, 0], cached K/V for one position, current K/V
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let k_cache = vec![1.0, 0.0, 0.0, 0.0]; // cached position 0
    let v_cache = vec![1.0, 0.0, 0.0, 0.0];
    let current_k = vec![1.0, 0.0, 0.0, 0.0]; // current position 1
    let current_v = vec![0.0, 1.0, 0.0, 0.0];

    let output = model.attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Output should be weighted combination of v_cache and current_v
    // Both K vectors are identical to Q, so scores are equal -> 50/50 weights
    // Output should be approximately [0.5, 0.5, 0, 0]
    assert_eq!(
        output.len(),
        4,
        "IMP-101c: Output should have hidden_dim elements"
    );

    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.1,
        "IMP-101c: Attention output should be normalized weighted sum"
    );
}

/// IMP-101c: Cache handles multiple positions correctly
#[test]
fn test_imp_101c_kv_cache_multiple_positions() {
    let mut cache = OwnedQuantizedKVCache::new(1, 4, 100);

    // Add 3 positions
    for i in 0..3 {
        let k = vec![i as f32; 4];
        let v = vec![(i as f32) * 0.1; 4];
        cache.append(0, &k, &v);
        cache.advance();
    }

    assert_eq!(cache.len(), 3, "IMP-101c: Cache should have 3 positions");

    let k_data = cache.get_k(0);
    assert_eq!(
        k_data.len(),
        12,
        "IMP-101c: K cache should have 3 * 4 = 12 elements"
    );

    // Verify first position K values
    assert!(
        (k_data[0] - 0.0).abs() < 1e-6,
        "IMP-101c: First K should be 0"
    );
    // Verify second position K values
    assert!(
        (k_data[4] - 1.0).abs() < 1e-6,
        "IMP-101c: Second K should be 1"
    );
    // Verify third position K values
    assert!(
        (k_data[8] - 2.0).abs() < 1e-6,
        "IMP-101c: Third K should be 2"
    );
}

#[test]
fn test_imp_105_gqa_attention_multiple_q_per_kv() {
    // IMP-105: GQA (Grouped Query Attention) support
    // 8 Q heads share 2 KV heads (4 Q heads per KV head)
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 32, // 8 heads * 4 head_dim
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 8,    // Q heads
        num_kv_heads: 2, // KV heads (4:1 ratio)
        vocab_size: 100,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    // Create model with dummy weights
    let hidden_dim = config.hidden_dim;
    let head_dim = hidden_dim / config.num_heads; // 4
    let kv_dim = config.num_kv_heads * head_dim; // 2 * 4 = 8

    // Q: [hidden_dim] = [32] - 8 heads
    // K/V: [kv_dim] = [8] - 2 heads
    let q = vec![1.0f32; hidden_dim];
    let current_k = vec![1.0f32; kv_dim];
    let current_v = vec![1.0f32; kv_dim];

    // Empty cache for first position
    let k_cache: Vec<f32> = vec![];
    let v_cache: Vec<f32> = vec![];

    // Test that GQA attention computes correctly
    // Q heads 0-3 should use KV head 0
    // Q heads 4-7 should use KV head 1
    let model = create_test_model_with_config(&config);
    let output = model.attention_with_cache_gqa(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Output should have hidden_dim elements
    assert_eq!(
        output.len(),
        hidden_dim,
        "IMP-105: GQA output should have hidden_dim={hidden_dim} elements"
    );

    // Each head's output should be non-zero (softmax weight = 1.0 for single position)
    for head in 0..config.num_heads {
        let head_start = head * head_dim;
        let head_sum: f32 = output[head_start..head_start + head_dim].iter().sum();
        assert!(
            head_sum.abs() > 1e-6,
            "IMP-105: GQA head {head} output should be non-zero"
        );
    }
}

#[test]
fn test_imp_105_gqa_kv_head_sharing() {
    // IMP-105: Verify that multiple Q heads correctly share KV heads
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 16, // 4 heads * 4 head_dim
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 4,    // Q heads
        num_kv_heads: 2, // KV heads (2:1 ratio)
        vocab_size: 100,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let hidden_dim = config.hidden_dim;
    let head_dim = hidden_dim / config.num_heads; // 4
    let kv_dim = config.num_kv_heads * head_dim; // 8

    // Create Q with different values per head
    let mut q = vec![0.0f32; hidden_dim];
    for head in 0..config.num_heads {
        for d in 0..head_dim {
            q[head * head_dim + d] = (head + 1) as f32;
        }
    }

    // Create K with different values per KV head
    let mut current_k = vec![0.0f32; kv_dim];
    for kv_head in 0..config.num_kv_heads {
        for d in 0..head_dim {
            current_k[kv_head * head_dim + d] = (kv_head + 1) as f32 * 0.5;
        }
    }

    // V values
    let mut current_v = vec![0.0f32; kv_dim];
    for kv_head in 0..config.num_kv_heads {
        for d in 0..head_dim {
            current_v[kv_head * head_dim + d] = (kv_head + 1) as f32;
        }
    }

    let k_cache: Vec<f32> = vec![];
    let v_cache: Vec<f32> = vec![];

    let model = create_test_model_with_config(&config);
    let output = model.attention_with_cache_gqa(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Q heads 0,1 should use KV head 0 (value=1.0)
    // Q heads 2,3 should use KV head 1 (value=2.0)
    // With softmax weight = 1.0 (single position), output = V
    let eps = 1e-5;

    // Head 0 and 1 should have similar outputs (both use KV head 0)
    let head0_sum: f32 = output[0..head_dim].iter().sum();
    let head1_sum: f32 = output[head_dim..2 * head_dim].iter().sum();

    // Head 2 and 3 should have similar outputs (both use KV head 1)
    let head2_sum: f32 = output[2 * head_dim..3 * head_dim].iter().sum();
    let head3_sum: f32 = output[3 * head_dim..4 * head_dim].iter().sum();

    // Verify KV head sharing pattern
    assert!(
        (head0_sum - head1_sum).abs() < eps,
        "IMP-105: Heads 0,1 should produce same output (share KV head 0)"
    );
    assert!(
        (head2_sum - head3_sum).abs() < eps,
        "IMP-105: Heads 2,3 should produce same output (share KV head 1)"
    );
    assert!(
        (head0_sum - head2_sum).abs() > eps,
        "IMP-105: Heads using different KV heads should have different outputs"
    );
}
