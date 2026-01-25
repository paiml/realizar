//! Q4_0 SIMD Transformer Tests (PMAT-803)
//!
//! Comprehensive tests for `src/apr_transformer/q4_simd.rs` to drive coverage.
//!
//! # Coverage Targets
//!
//! - `QuantizedAprTensorQ4` - Q4_0 tensor struct
//! - `QuantizedAprLayerQ4` - Quantized layer struct
//! - `QuantizedAprTransformerQ4` - Full transformer with SIMD matmul
//! - `AprInferenceScratch` - Zero-allocation scratch buffer
//!
//! # Test Organization
//!
//! - Part 1: QuantizedAprTensorQ4 tests
//! - Part 2: AprInferenceScratch tests
//! - Part 3: QuantizedAprLayerQ4 tests
//! - Part 4: QuantizedAprTransformerQ4 construction tests
//! - Part 5: Forward pass tests
//! - Part 6: KV cache tests
//! - Part 7: RoPE and attention tests
//! - Part 8: Memory and utility tests

use crate::apr_transformer::{
    AprInferenceScratch, AprTransformerConfig, QuantizedAprLayerQ4, QuantizedAprTensorQ4,
    QuantizedAprTransformerQ4,
};

// ============================================================================
// Part 1: QuantizedAprTensorQ4 Tests
// ============================================================================

#[test]
fn test_quantized_tensor_new() {
    let data = vec![0u8; 18]; // 1 Q4_0 block
    let tensor = QuantizedAprTensorQ4::new(data.clone(), 32, 1);

    assert_eq!(tensor.data, data);
    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 1);
}

#[test]
fn test_quantized_tensor_new_empty() {
    let tensor = QuantizedAprTensorQ4::new(vec![], 0, 0);

    assert!(tensor.data.is_empty());
    assert_eq!(tensor.in_dim, 0);
    assert_eq!(tensor.out_dim, 0);
}

#[test]
fn test_quantized_tensor_new_large() {
    // 1024 x 4096 matrix
    let num_elements: usize = 1024 * 4096;
    let num_blocks = num_elements.div_ceil(32);
    let data = vec![0u8; num_blocks * 18];
    let tensor = QuantizedAprTensorQ4::new(data, 1024, 4096);

    assert_eq!(tensor.in_dim, 1024);
    assert_eq!(tensor.out_dim, 4096);
}

#[test]
fn test_quantized_tensor_zeros_basic() {
    let tensor = QuantizedAprTensorQ4::zeros(64, 128);

    assert_eq!(tensor.in_dim, 64);
    assert_eq!(tensor.out_dim, 128);

    // 64 * 128 = 8192 elements, 256 blocks, 256 * 18 = 4608 bytes
    let expected_bytes = QuantizedAprTensorQ4::expected_bytes(64 * 128);
    assert_eq!(tensor.data.len(), expected_bytes);
    assert!(tensor.data.iter().all(|&b| b == 0));
}

#[test]
fn test_quantized_tensor_zeros_single_block() {
    // Exactly 32 elements = 1 block
    let tensor = QuantizedAprTensorQ4::zeros(32, 1);

    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 1);
    assert_eq!(tensor.data.len(), 18); // 1 block = 18 bytes
}

#[test]
fn test_quantized_tensor_zeros_partial_block() {
    // 33 elements = 2 blocks (ceil)
    let tensor = QuantizedAprTensorQ4::zeros(33, 1);

    assert_eq!(tensor.in_dim, 33);
    assert_eq!(tensor.out_dim, 1);
    assert_eq!(tensor.data.len(), 36); // 2 blocks = 36 bytes
}

#[test]
fn test_quantized_tensor_expected_bytes() {
    // Exact block boundary
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(96), 54);

    // Partial blocks (ceil division)
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(1), 18); // 1 element still needs 1 block
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(31), 18);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(33), 36);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(65), 54);
}

#[test]
fn test_quantized_tensor_expected_bytes_zero() {
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(0), 0);
}

#[test]
fn test_quantized_tensor_expected_bytes_large() {
    // 1M elements = 31250 blocks = 562500 bytes
    let expected = (1_000_000_usize.div_ceil(32)) * 18;
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(1_000_000), expected);
}

#[test]
fn test_quantized_tensor_clone() {
    let tensor = QuantizedAprTensorQ4::zeros(64, 64);
    let cloned = tensor.clone();

    assert_eq!(cloned.data.len(), tensor.data.len());
    assert_eq!(cloned.in_dim, tensor.in_dim);
    assert_eq!(cloned.out_dim, tensor.out_dim);
}

// ============================================================================
// Part 2: AprInferenceScratch Tests
// ============================================================================

#[test]
fn test_scratch_from_config_basic() {
    let config = create_test_config(64, 128, 4, 4, 100);
    let scratch = AprInferenceScratch::from_config(&config);

    assert_eq!(scratch.hidden.len(), 64);
    assert_eq!(scratch.normed.len(), 64);
    assert_eq!(scratch.qkv_out.len(), 64 * 3); // Conservative estimate
    assert_eq!(scratch.q.len(), 64);
    assert_eq!(scratch.k.len(), 64);
    assert_eq!(scratch.v.len(), 64);
    assert_eq!(scratch.attn_out.len(), 64);
    assert_eq!(scratch.ffn_input.len(), 64);
    assert_eq!(scratch.ffn_up.len(), 128);
    assert_eq!(scratch.ffn_gate.len(), 128);
    assert_eq!(scratch.ffn_out.len(), 64);
}

#[test]
fn test_scratch_from_config_large() {
    let config = create_test_config(2048, 8192, 32, 8, 32000);
    let scratch = AprInferenceScratch::from_config(&config);

    assert_eq!(scratch.hidden.len(), 2048);
    assert_eq!(scratch.ffn_up.len(), 8192);
    assert_eq!(scratch.ffn_gate.len(), 8192);
}

#[test]
fn test_scratch_clear() {
    let config = create_test_config(64, 128, 4, 4, 100);
    let mut scratch = AprInferenceScratch::from_config(&config);

    // Set some values
    scratch.hidden[0] = 1.0;
    scratch.normed[0] = 2.0;
    scratch.qkv_out[0] = 3.0;
    scratch.q[0] = 4.0;
    scratch.k[0] = 5.0;
    scratch.v[0] = 6.0;
    scratch.attn_out[0] = 7.0;
    scratch.ffn_input[0] = 8.0;
    scratch.ffn_up[0] = 9.0;
    scratch.ffn_gate[0] = 10.0;
    scratch.ffn_out[0] = 11.0;

    scratch.clear();

    // Verify all cleared
    assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    assert!(scratch.normed.iter().all(|&x| x == 0.0));
    assert!(scratch.qkv_out.iter().all(|&x| x == 0.0));
    assert!(scratch.q.iter().all(|&x| x == 0.0));
    assert!(scratch.k.iter().all(|&x| x == 0.0));
    assert!(scratch.v.iter().all(|&x| x == 0.0));
    assert!(scratch.attn_out.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_input.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_up.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_gate.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_out.iter().all(|&x| x == 0.0));
}

#[test]
fn test_scratch_multiple_clears() {
    let config = create_test_config(32, 64, 2, 2, 50);
    let mut scratch = AprInferenceScratch::from_config(&config);

    for i in 0..10 {
        scratch.hidden.fill(i as f32);
        scratch.clear();
        assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    }
}

// ============================================================================
// Part 3: QuantizedAprLayerQ4 Tests
// ============================================================================

#[test]
fn test_layer_q4_construction() {
    let hidden_dim = 64;
    let intermediate_dim = 128;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    };

    assert_eq!(layer.attn_norm_weight.len(), hidden_dim);
    assert_eq!(layer.qkv_weight.out_dim, hidden_dim * 3);
    assert!(layer.ffn_gate_weight.is_none());
    assert!(layer.ffn_norm_weight.is_some());
}

#[test]
fn test_layer_q4_with_swiglu_gate() {
    let hidden_dim = 64;
    let intermediate_dim = 128;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)),
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    };

    assert!(layer.ffn_gate_weight.is_some());
    let gate = layer.ffn_gate_weight.as_ref().unwrap();
    assert_eq!(gate.in_dim, hidden_dim);
    assert_eq!(gate.out_dim, intermediate_dim);
}

#[test]
fn test_layer_q4_clone() {
    let layer = create_test_layer_q4(64, 128, true);
    let cloned = layer.clone();

    assert_eq!(cloned.attn_norm_weight.len(), layer.attn_norm_weight.len());
    assert_eq!(cloned.qkv_weight.data.len(), layer.qkv_weight.data.len());
    assert!(cloned.ffn_gate_weight.is_some());
}

// ============================================================================
// Part 4: QuantizedAprTransformerQ4 Construction Tests
// ============================================================================

#[test]
fn test_transformer_q4_construction() {
    let transformer = create_minimal_q4_transformer(1);

    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.layers.len(), 1);
    assert_eq!(transformer.token_embedding.len(), 100 * 64);
    assert_eq!(transformer.output_norm_weight.len(), 64);
}

#[test]
fn test_transformer_q4_multi_layer() {
    let num_layers = 4;
    let transformer = create_minimal_q4_transformer(num_layers);

    assert_eq!(transformer.config.num_layers, num_layers);
    assert_eq!(transformer.layers.len(), num_layers);
}

#[test]
fn test_transformer_q4_config_accessor() {
    let transformer = create_minimal_q4_transformer(2);
    let config = transformer.config();

    assert_eq!(config.hidden_dim, 64);
    assert_eq!(config.num_heads, 4);
    assert_eq!(config.num_kv_heads, 4);
    assert_eq!(config.vocab_size, 100);
    assert_eq!(config.intermediate_dim, 128);
    assert!((config.rope_theta - 10000.0).abs() < 1e-6);
}

#[test]
fn test_transformer_q4_create_scratch() {
    let transformer = create_minimal_q4_transformer(1);
    let scratch = transformer.create_scratch();

    assert_eq!(scratch.hidden.len(), 64);
    assert_eq!(scratch.ffn_up.len(), 128);
}

#[test]
fn test_transformer_q4_create_kv_cache() {
    let transformer = create_minimal_q4_transformer(2);
    let cache = transformer.create_kv_cache();

    assert_eq!(cache.len(), 0); // Empty cache initially
}

// ============================================================================
// Part 5: Forward Pass Tests
// ============================================================================

#[test]
fn test_forward_empty_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let result = transformer.forward(&[]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err,
        crate::error::RealizarError::InvalidShape { .. }
    ));
}

#[test]
fn test_forward_single_token() {
    let transformer = create_minimal_q4_transformer(1);
    let result = transformer.forward(&[0]);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 100); // vocab_size
}

#[test]
fn test_forward_multiple_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let result = transformer.forward(&[0, 1, 2]);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 100); // Only last token logits returned
}

#[test]
fn test_forward_oov_token() {
    // Out-of-vocabulary token should be handled gracefully
    let transformer = create_minimal_q4_transformer(1);
    let result = transformer.forward(&[999]); // vocab_size = 100

    assert!(result.is_ok()); // Should use zero embedding
}

#[test]
fn test_forward_multi_layer() {
    let transformer = create_minimal_q4_transformer(4);
    let result = transformer.forward(&[42]);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_forward_with_swiglu() {
    let transformer = create_q4_transformer_with_gate(2);
    let result = transformer.forward(&[1, 2, 3]);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_forward_single_with_scratch() {
    let transformer = create_minimal_q4_transformer(1);
    let mut scratch = transformer.create_scratch();

    let result = transformer.forward_single_with_scratch(0, &mut scratch);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_forward_single_with_scratch_oov() {
    let transformer = create_minimal_q4_transformer(1);
    let mut scratch = transformer.create_scratch();

    let result = transformer.forward_single_with_scratch(999, &mut scratch);

    assert!(result.is_ok()); // Should handle OOV gracefully
}

#[test]
fn test_forward_single_with_scratch_reuse() {
    let transformer = create_minimal_q4_transformer(1);
    let mut scratch = transformer.create_scratch();

    // Multiple forward passes with same scratch
    for token_id in 0..10u32 {
        let result = transformer.forward_single_with_scratch(token_id, &mut scratch);
        assert!(result.is_ok());
    }
}

#[test]
fn test_forward_single_with_scratch_swiglu() {
    let transformer = create_q4_transformer_with_gate(1);
    let mut scratch = transformer.create_scratch();

    let result = transformer.forward_single_with_scratch(5, &mut scratch);
    assert!(result.is_ok());
}

// ============================================================================
// Part 6: KV Cache Tests
// ============================================================================

#[test]
fn test_forward_with_cache_empty_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    let result = transformer.forward_with_cache(&[], &mut cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_with_cache_single_token() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    let result = transformer.forward_with_cache(&[0], &mut cache);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 100);
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_forward_with_cache_sequential_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    // Process tokens one at a time (autoregressive)
    for (i, &token) in [1u32, 2, 3, 4, 5].iter().enumerate() {
        let result = transformer.forward_with_cache(&[token], &mut cache);
        assert!(result.is_ok());
        assert_eq!(cache.len(), i + 1);
    }
}

#[test]
fn test_forward_with_cache_batch_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    // Process multiple tokens at once (prefill)
    let result = transformer.forward_with_cache(&[1, 2, 3, 4], &mut cache);

    assert!(result.is_ok());
    assert_eq!(cache.len(), 4);
}

#[test]
fn test_forward_with_cache_multi_layer() {
    let transformer = create_minimal_q4_transformer(4);
    let mut cache = transformer.create_kv_cache();

    let result = transformer.forward_with_cache(&[10, 20, 30], &mut cache);

    assert!(result.is_ok());
    assert_eq!(cache.len(), 3);
}

#[test]
fn test_forward_with_cache_swiglu() {
    let transformer = create_q4_transformer_with_gate(2);
    let mut cache = transformer.create_kv_cache();

    // Prefill
    let _ = transformer.forward_with_cache(&[1, 2, 3], &mut cache);

    // Generate
    let result = transformer.forward_with_cache(&[4], &mut cache);

    assert!(result.is_ok());
    assert_eq!(cache.len(), 4);
}

#[test]
fn test_forward_with_cache_vs_no_cache() {
    let transformer = create_minimal_q4_transformer(1);

    // Without cache
    let no_cache_result = transformer.forward(&[1, 2, 3]);

    // With cache (prefill all at once)
    let mut cache = transformer.create_kv_cache();
    let cache_result = transformer.forward_with_cache(&[1, 2, 3], &mut cache);

    assert!(no_cache_result.is_ok());
    assert!(cache_result.is_ok());

    // Both should produce same-size output
    assert_eq!(no_cache_result.unwrap().len(), cache_result.unwrap().len());
}

// ============================================================================
// Part 7: Memory and Utility Tests
// ============================================================================

#[test]
fn test_memory_size_basic() {
    let transformer = create_minimal_q4_transformer(1);
    let size = transformer.memory_size();

    // Should include embedding, norm, lm_head, and layer weights
    assert!(size > 0);

    // Rough calculation:
    // token_embedding: 100 * 64 * 4 = 25600 bytes
    // output_norm: 64 * 4 = 256 bytes
    // lm_head: Q4_0 bytes for 64 * 100
    // Plus layer weights
    assert!(size > 25000);
}

#[test]
fn test_memory_size_scales_with_layers() {
    let transformer1 = create_minimal_q4_transformer(1);
    let transformer2 = create_minimal_q4_transformer(4);

    let size1 = transformer1.memory_size();
    let size2 = transformer2.memory_size();

    // More layers = more memory
    assert!(size2 > size1);
}

#[test]
fn test_memory_size_with_gate() {
    let transformer_no_gate = create_minimal_q4_transformer(1);
    let transformer_with_gate = create_q4_transformer_with_gate(1);

    let size_no_gate = transformer_no_gate.memory_size();
    let size_with_gate = transformer_with_gate.memory_size();

    // Gate adds extra FFN weights
    assert!(size_with_gate > size_no_gate);
}

// ============================================================================
// Part 8: Edge Cases and Stress Tests
// ============================================================================

#[test]
fn test_forward_large_sequence() {
    let transformer = create_minimal_q4_transformer(1);

    // Process a longer sequence
    let tokens: Vec<u32> = (0..20).collect();
    let result = transformer.forward(&tokens);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 100);
}

#[test]
fn test_forward_with_cache_long_generation() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    // Prefill
    let _ = transformer.forward_with_cache(&[1, 2, 3, 4, 5], &mut cache);

    // Generate 20 more tokens
    for i in 0..20u32 {
        let result = transformer.forward_with_cache(&[i + 10], &mut cache);
        assert!(result.is_ok());
    }

    assert_eq!(cache.len(), 25); // 5 prefill + 20 generation
}

#[test]
fn test_gqa_different_kv_heads() {
    // Test with different num_kv_heads (GQA configuration)
    let transformer = create_gqa_transformer(4, 2); // 4 Q heads, 2 KV heads

    let result = transformer.forward(&[0, 1]);

    assert!(result.is_ok());
}

#[test]
fn test_small_head_dim() {
    // Test with small head dimension
    let hidden_dim = 32;
    let num_heads = 4; // head_dim = 8

    let transformer = create_custom_transformer(hidden_dim, 64, num_heads, num_heads, 50, 1);

    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

#[test]
fn test_rope_at_various_positions() {
    let transformer = create_minimal_q4_transformer(1);

    // Test RoPE at different sequence positions
    for i in 0..10 {
        let tokens: Vec<u32> = (0..i + 1).map(|x| x as u32).collect();
        let mut cache = transformer.create_kv_cache(); // Fresh cache each iteration
        let result = transformer.forward_with_cache(&tokens, &mut cache);
        assert!(result.is_ok());
    }
}

#[test]
fn test_parallel_attention_threshold() {
    // Test with >=4 heads (should use parallel path)
    let transformer_parallel = create_custom_transformer(64, 128, 8, 8, 100, 1);
    let result = transformer_parallel.forward(&[0, 1]);
    assert!(result.is_ok());

    // Test with <4 heads (sequential path)
    let transformer_sequential = create_custom_transformer(64, 128, 2, 2, 100, 1);
    let result2 = transformer_sequential.forward(&[0, 1]);
    assert!(result2.is_ok());
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_config(
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
) -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
    }
}

fn create_test_layer_q4(
    hidden_dim: usize,
    intermediate_dim: usize,
    with_gate: bool,
) -> QuantizedAprLayerQ4 {
    QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: if with_gate {
            Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim))
        } else {
            None
        },
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    }
}

fn create_minimal_q4_transformer(num_layers: usize) -> QuantizedAprTransformerQ4 {
    create_custom_transformer(64, 128, 4, 4, 100, num_layers)
}

fn create_q4_transformer_with_gate(num_layers: usize) -> QuantizedAprTransformerQ4 {
    let hidden_dim = 64;
    let vocab_size = 100;
    let intermediate_dim = 128;

    let layers: Vec<QuantizedAprLayerQ4> = (0..num_layers)
        .map(|_| QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)),
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        })
        .collect();

    QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test_swiglu".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

fn create_gqa_transformer(num_heads: usize, num_kv_heads: usize) -> QuantizedAprTransformerQ4 {
    let hidden_dim = num_heads * 16; // head_dim = 16
    create_custom_transformer(hidden_dim, hidden_dim * 2, num_heads, num_kv_heads, 100, 1)
}

fn create_custom_transformer(
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    num_layers: usize,
) -> QuantizedAprTransformerQ4 {
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_dim = q_dim + 2 * kv_dim;

    let layers: Vec<QuantizedAprLayerQ4> = (0..num_layers)
        .map(|_| QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_dim),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: None,
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        })
        .collect();

    QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test_custom".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}
