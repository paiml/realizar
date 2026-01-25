//! APR Transformer Coverage Tests (PMAT-803)
//!
//! Comprehensive tests to improve coverage for:
//! - loader.rs: MmapAprTransformer, AprQuantizationType, QuantizedAprTransformer
//! - dequant.rs: f16_to_f32, extract_scale_min_apr, dequantize_q4_k_apr, dequantize_q6_k_apr
//! - config.rs: AprKVCache, GenerateConfig, AprTransformerConfig, AprTransformerLayer, Q4KLayerWeights
//! - helpers.rs: SIMD primitives

use crate::apr::MAGIC;
use crate::apr_transformer::{
    AprKVCache, AprQuantizationType, AprTransformerConfig, AprTransformerLayer, GenerateConfig,
    MmapAprTransformer, Q4KLayerWeights, QuantizedAprTransformer, APR_TRANSFORMER_HEADER_SIZE,
};

// ============================================================================
// Part 1: AprQuantizationType Tests
// ============================================================================

#[test]
fn test_quantization_type_default() {
    let qt: AprQuantizationType = Default::default();
    assert_eq!(qt, AprQuantizationType::F32);
}

#[test]
fn test_quantization_type_bits_per_weight() {
    assert_eq!(AprQuantizationType::F32.bits_per_weight(), 32.0);
    assert_eq!(AprQuantizationType::Q4_K.bits_per_weight(), 4.5);
    assert_eq!(AprQuantizationType::Q8_0.bits_per_weight(), 8.0);
}

#[test]
fn test_quantization_type_bytes_per_block() {
    assert_eq!(AprQuantizationType::F32.bytes_per_block(), 4);
    assert_eq!(AprQuantizationType::Q4_K.bytes_per_block(), 144);
    assert_eq!(AprQuantizationType::Q8_0.bytes_per_block(), 36);
}

#[test]
fn test_quantization_type_values_per_block() {
    assert_eq!(AprQuantizationType::F32.values_per_block(), 1);
    assert_eq!(AprQuantizationType::Q4_K.values_per_block(), 256);
    assert_eq!(AprQuantizationType::Q8_0.values_per_block(), 32);
}

#[test]
fn test_quantization_type_to_byte() {
    assert_eq!(AprQuantizationType::F32.to_byte(), 0);
    assert_eq!(AprQuantizationType::Q4_K.to_byte(), 1);
    assert_eq!(AprQuantizationType::Q8_0.to_byte(), 2);
}

#[test]
fn test_quantization_type_from_byte_valid() {
    assert_eq!(
        AprQuantizationType::from_byte(0),
        Some(AprQuantizationType::F32)
    );
    assert_eq!(
        AprQuantizationType::from_byte(1),
        Some(AprQuantizationType::Q4_K)
    );
    assert_eq!(
        AprQuantizationType::from_byte(2),
        Some(AprQuantizationType::Q8_0)
    );
}

#[test]
fn test_quantization_type_from_byte_invalid() {
    assert_eq!(AprQuantizationType::from_byte(3), None);
    assert_eq!(AprQuantizationType::from_byte(255), None);
}

#[test]
fn test_quantization_type_roundtrip() {
    for qt in [
        AprQuantizationType::F32,
        AprQuantizationType::Q4_K,
        AprQuantizationType::Q8_0,
    ] {
        let byte = qt.to_byte();
        let recovered = AprQuantizationType::from_byte(byte);
        assert_eq!(recovered, Some(qt));
    }
}

#[test]
fn test_quantization_type_clone() {
    let original = AprQuantizationType::Q4_K;
    let cloned = original;
    assert_eq!(original, cloned);
}

#[test]
fn test_quantization_type_debug() {
    let qt = AprQuantizationType::Q8_0;
    let debug_str = format!("{:?}", qt);
    assert!(debug_str.contains("Q8_0"));
}

// ============================================================================
// Part 2: QuantizedAprTransformer Tests
// ============================================================================

fn create_test_apr_config() -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
    }
}

#[test]
fn test_quantized_transformer_new_f32() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

    assert_eq!(transformer.quantization_type(), AprQuantizationType::F32);
    assert_eq!(transformer.bits_per_weight(), 32.0);
    assert_eq!(transformer.config().hidden_dim, 64);
    assert_eq!(transformer.config().num_layers, 2);
}

#[test]
fn test_quantized_transformer_new_q4k() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);

    assert_eq!(transformer.quantization_type(), AprQuantizationType::Q4_K);
    assert_eq!(transformer.bits_per_weight(), 4.5);
}

#[test]
fn test_quantized_transformer_new_q8_0() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);

    assert_eq!(transformer.quantization_type(), AprQuantizationType::Q8_0);
    assert_eq!(transformer.bits_per_weight(), 8.0);
}

#[test]
fn test_quantized_transformer_weight_bytes() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

    let bytes = transformer.weight_bytes();
    assert!(bytes > 0);
}

#[test]
fn test_quantized_transformer_f32_equivalent_bytes() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);

    let f32_bytes = transformer.f32_equivalent_bytes();
    let actual_bytes = transformer.weight_bytes();

    // Q4_K should be smaller than F32 equivalent
    // Note: embeddings are still F32, so compression is partial
    assert!(f32_bytes > 0);
    assert!(actual_bytes > 0);
}

#[test]
fn test_quantized_transformer_num_parameters() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

    let params = transformer.num_parameters();
    assert!(params > 0);

    // Estimate: embedding (100*64*2) + layers + norms
    // Should be at least vocab * hidden * 2
    assert!(params >= 100 * 64 * 2);
}

#[test]
fn test_quantized_transformer_forward_empty_tokens() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

    let result = transformer.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_forward_single_token() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

    let result = transformer.forward(&[0]);
    assert!(result.is_ok());

    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_quantized_transformer_forward_multiple_tokens() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

    let result = transformer.forward(&[0, 1, 2, 3]);
    assert!(result.is_ok());

    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_quantized_transformer_forward_oov_token() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

    // Token ID beyond vocab_size (100)
    let result = transformer.forward(&[999]);
    assert!(result.is_ok()); // Should handle gracefully with zeros
}

#[test]
fn test_quantized_transformer_forward_with_cache() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
    let mut cache = AprKVCache::new(&config);

    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());

    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_quantized_transformer_forward_with_cache_sequential() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
    let mut cache = AprKVCache::new(&config);

    // Process multiple tokens sequentially
    for (pos, token_id) in [0u32, 1, 2, 3].iter().enumerate() {
        let result = transformer.forward_with_cache(*token_id, &mut cache, pos);
        assert!(result.is_ok());
    }
}

#[test]
fn test_quantized_transformer_to_bytes_and_from_bytes() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);

    // Serialize
    let bytes = transformer.to_bytes();
    assert!(bytes.is_ok());
    let bytes = bytes.unwrap();

    // Verify magic
    assert_eq!(&bytes[0..4], MAGIC);

    // Deserialize
    let recovered = QuantizedAprTransformer::from_bytes(&bytes);
    assert!(recovered.is_ok());
    let recovered = recovered.unwrap();

    // Verify config matches
    assert_eq!(recovered.config().hidden_dim, config.hidden_dim);
    assert_eq!(recovered.config().num_layers, config.num_layers);
    assert_eq!(recovered.quantization_type(), AprQuantizationType::Q4_K);
}

#[test]
fn test_quantized_transformer_from_bytes_too_small() {
    let bytes = vec![0u8; 32]; // Too small for header
    let result = QuantizedAprTransformer::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_from_bytes_invalid_magic() {
    let mut bytes = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];
    bytes[0..4].copy_from_slice(b"GGUF"); // Wrong magic

    let result = QuantizedAprTransformer::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_from_bytes_invalid_quant_type() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

    let mut bytes = transformer.to_bytes().unwrap();

    // Corrupt quantization type byte (offset 48)
    bytes[48] = 255; // Invalid quant type

    let result = QuantizedAprTransformer::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_calculate_quantized_bytes() {
    // F32: 1 value per block, 4 bytes per block
    assert_eq!(
        QuantizedAprTransformer::calculate_quantized_bytes(100, AprQuantizationType::F32),
        100 * 4
    );

    // Q4_K: 256 values per block, 144 bytes per block
    // 100 values = 1 block (ceil)
    assert_eq!(
        QuantizedAprTransformer::calculate_quantized_bytes(100, AprQuantizationType::Q4_K),
        144
    );

    // Q8_0: 32 values per block, 36 bytes per block
    // 100 values = 4 blocks (ceil(100/32) = 4)
    assert_eq!(
        QuantizedAprTransformer::calculate_quantized_bytes(100, AprQuantizationType::Q8_0),
        4 * 36
    );
}

// ============================================================================
// Part 3: AprKVCache Tests
// ============================================================================

#[test]
fn test_kv_cache_new() {
    let config = create_test_apr_config();
    let cache = AprKVCache::new(&config);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), config.context_length);
    assert_eq!(cache.num_kv_heads(), config.num_kv_heads);
    assert_eq!(cache.head_dim(), config.hidden_dim / config.num_heads);
}

#[test]
fn test_kv_cache_append_and_get() {
    let config = create_test_apr_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Append to layer 0
    cache.append(0, &k, &v);

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // Get and verify
    let (cached_k, cached_v) = cache.get(0);
    assert_eq!(cached_k.len(), kv_size);
    assert_eq!(cached_v.len(), kv_size);
    assert_eq!(cached_k[0], 1.0);
    assert_eq!(cached_v[0], 2.0);
}

#[test]
fn test_kv_cache_append_multiple_positions() {
    let config = create_test_apr_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

    // Append 3 positions
    for i in 0..3 {
        let k = vec![(i + 1) as f32; kv_size];
        let v = vec![(i + 10) as f32; kv_size];

        // Append to all layers
        for layer in 0..config.num_layers {
            cache.append(layer, &k, &v);
        }
    }

    assert_eq!(cache.len(), 3);

    // Verify layer 0 values
    let (cached_k, cached_v) = cache.get(0);
    assert_eq!(cached_k.len(), 3 * kv_size);
    assert_eq!(cached_v.len(), 3 * kv_size);
}

#[test]
fn test_kv_cache_clear() {
    let config = create_test_apr_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    cache.append(0, &k, &v);
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    // Capacity should be preserved
    assert_eq!(cache.capacity(), config.context_length);
}

#[test]
#[should_panic(expected = "Layer index out of bounds")]
fn test_kv_cache_append_invalid_layer() {
    let config = create_test_apr_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Layer index beyond num_layers
    cache.append(999, &k, &v);
}

// ============================================================================
// Part 4: GenerateConfig Tests
// ============================================================================

#[test]
fn test_generate_config_default() {
    let config = GenerateConfig::default();

    assert_eq!(config.max_tokens, 32);
    assert!((config.temperature - 1.0).abs() < 1e-6);
    assert!((config.top_p - 0.9).abs() < 1e-6);
    assert_eq!(config.top_k, 0);
    assert!((config.repetition_penalty - 1.0).abs() < 1e-6);
}

#[test]
fn test_generate_config_custom() {
    let config = GenerateConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 50,
        repetition_penalty: 1.1,
    };

    assert_eq!(config.max_tokens, 100);
    assert!((config.temperature - 0.7).abs() < 1e-6);
    assert!((config.top_p - 0.95).abs() < 1e-6);
    assert_eq!(config.top_k, 50);
    assert!((config.repetition_penalty - 1.1).abs() < 1e-6);
}

#[test]
fn test_generate_config_clone() {
    let original = GenerateConfig {
        max_tokens: 50,
        temperature: 0.5,
        top_p: 0.8,
        top_k: 40,
        repetition_penalty: 1.2,
    };

    let cloned = original.clone();

    assert_eq!(cloned.max_tokens, original.max_tokens);
    assert!((cloned.temperature - original.temperature).abs() < 1e-6);
}

#[test]
fn test_generate_config_debug() {
    let config = GenerateConfig::default();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("max_tokens"));
    assert!(debug_str.contains("temperature"));
}

// ============================================================================
// Part 5: AprTransformerConfig Tests
// ============================================================================

#[test]
fn test_transformer_config_default() {
    let config = AprTransformerConfig::default();

    assert_eq!(config.architecture, "unknown");
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.num_layers, 6);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.intermediate_dim, 2048);
    assert_eq!(config.context_length, 2048);
    assert!((config.rope_theta - 10000.0).abs() < 1e-6);
    assert!((config.eps - 1e-5).abs() < 1e-9);
}

#[test]
fn test_transformer_config_custom() {
    let config = AprTransformerConfig {
        architecture: "llama".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 128000,
        intermediate_dim: 14336,
        context_length: 8192,
        rope_theta: 500000.0,
        eps: 1e-6,
    };

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_kv_heads, 8); // GQA
}

#[test]
fn test_transformer_config_serialization() {
    let config = create_test_apr_config();

    // Serialize to JSON
    let json = serde_json::to_string(&config).unwrap();

    // Deserialize
    let recovered: AprTransformerConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config, recovered);
}

#[test]
fn test_transformer_config_clone() {
    let original = create_test_apr_config();
    let cloned = original.clone();

    assert_eq!(original, cloned);
}

#[test]
fn test_transformer_config_debug() {
    let config = create_test_apr_config();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("hidden_dim"));
    assert!(debug_str.contains("num_layers"));
}

// ============================================================================
// Part 6: AprTransformerLayer Tests
// ============================================================================

#[test]
fn test_layer_empty() {
    let layer = AprTransformerLayer::empty(64, 128);

    assert_eq!(layer.attn_norm_weight.len(), 64);
    assert!(layer
        .attn_norm_weight
        .iter()
        .all(|&x| (x - 1.0).abs() < 1e-6));

    assert_eq!(layer.qkv_weight.len(), 64 * 3 * 64);
    assert!(layer.qkv_weight.iter().all(|&x| x == 0.0));

    assert_eq!(layer.attn_output_weight.len(), 64 * 64);
    assert_eq!(layer.ffn_up_weight.len(), 64 * 128);
    assert_eq!(layer.ffn_down_weight.len(), 128 * 64);

    assert!(layer.attn_norm_bias.is_none());
    assert!(layer.qkv_bias.is_none());
    assert!(layer.ffn_gate_weight.is_none());
}

#[test]
fn test_layer_empty_gqa() {
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let intermediate_dim = 128;

    let layer =
        AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

    // For GQA: QKV = Q + K + V = hidden_dim + kv_dim + kv_dim
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
    assert_eq!(layer.attn_output_weight.len(), hidden_dim * hidden_dim);
}

#[test]
fn test_layer_num_parameters() {
    let layer = AprTransformerLayer::empty(64, 128);
    let params = layer.num_parameters();

    // attn_norm (64) + qkv (64*192) + attn_out (64*64) + ffn_up (64*128) + ffn_down (128*64)
    let expected = 64 + (64 * 3 * 64) + (64 * 64) + (64 * 128) + (128 * 64);
    assert_eq!(params, expected);
}

#[test]
fn test_layer_num_parameters_with_optionals() {
    let mut layer = AprTransformerLayer::empty(64, 128);

    // Add optional fields
    layer.attn_norm_bias = Some(vec![0.0; 64]);
    layer.qkv_bias = Some(vec![0.0; 64 * 3]);
    layer.ffn_gate_weight = Some(vec![0.0; 64 * 128]);
    layer.ffn_norm_weight = Some(vec![1.0; 64]);

    let params = layer.num_parameters();

    // Base params + optionals
    let base = 64 + (64 * 3 * 64) + (64 * 64) + (64 * 128) + (128 * 64);
    let optionals = 64 + (64 * 3) + (64 * 128) + 64;
    assert_eq!(params, base + optionals);
}

#[test]
fn test_layer_serialization() {
    let layer = AprTransformerLayer::empty(32, 64);

    // Serialize to JSON
    let json = serde_json::to_string(&layer).unwrap();

    // Deserialize
    let recovered: AprTransformerLayer = serde_json::from_str(&json).unwrap();

    assert_eq!(
        layer.attn_norm_weight.len(),
        recovered.attn_norm_weight.len()
    );
    assert_eq!(layer.qkv_weight.len(), recovered.qkv_weight.len());
}

// ============================================================================
// Part 7: Q4KLayerWeights Tests
// ============================================================================

#[test]
fn test_q4k_layer_weights_default() {
    let weights: Q4KLayerWeights = Default::default();

    assert!(weights.qkv_weight.is_none());
    assert!(weights.attn_q_weight.is_none());
    assert!(weights.attn_k_weight.is_none());
    assert!(weights.attn_v_weight.is_none());
    assert!(weights.attn_v_weight_q6k.is_none());
    assert!(weights.attn_output_weight.is_none());
    assert!(weights.ffn_gate_weight.is_none());
    assert!(weights.ffn_up_weight.is_none());
    assert!(weights.ffn_down_weight.is_none());
    assert!(weights.ffn_down_weight_q6k.is_none());
    assert!(weights.ffn_up_weight_q6k.is_none());
}

#[test]
fn test_q4k_layer_weights_with_data() {
    let weights = Q4KLayerWeights {
        qkv_weight: Some(vec![0u8; 144]), // 1 Q4_K block
        attn_q_weight: Some(vec![0u8; 144]),
        attn_k_weight: Some(vec![0u8; 144]),
        attn_v_weight: Some(vec![0u8; 144]),
        attn_v_weight_q6k: Some(vec![0u8; 210]),
        attn_output_weight: Some(vec![0u8; 144]),
        ffn_gate_weight: Some(vec![0u8; 144]),
        ffn_up_weight: Some(vec![0u8; 144]),
        ffn_down_weight: Some(vec![0u8; 144]),
        ffn_down_weight_q6k: Some(vec![0u8; 210]),
        ffn_up_weight_q6k: Some(vec![0u8; 210]),
    };

    assert!(weights.qkv_weight.is_some());
    assert_eq!(weights.qkv_weight.as_ref().unwrap().len(), 144);
    assert!(weights.attn_v_weight_q6k.is_some());
    assert_eq!(weights.attn_v_weight_q6k.as_ref().unwrap().len(), 210);
}

#[test]
fn test_q4k_layer_weights_serialization() {
    let weights = Q4KLayerWeights {
        attn_q_weight: Some(vec![1u8, 2, 3]),
        ..Default::default()
    };

    let json = serde_json::to_string(&weights).unwrap();
    let recovered: Q4KLayerWeights = serde_json::from_str(&json).unwrap();

    assert_eq!(weights.attn_q_weight, recovered.attn_q_weight);
}

#[test]
fn test_q4k_layer_weights_clone() {
    let original = Q4KLayerWeights {
        ffn_up_weight: Some(vec![42u8; 100]),
        ..Default::default()
    };

    let cloned = original.clone();
    assert_eq!(original.ffn_up_weight, cloned.ffn_up_weight);
}

// ============================================================================
// Part 8: Dequantization Tests
// ============================================================================

#[test]
fn test_f16_to_f32_zero() {
    use super::super::dequant::f16_to_f32;

    // Positive zero
    assert_eq!(f16_to_f32(0x0000), 0.0);
    // Negative zero
    assert_eq!(f16_to_f32(0x8000), -0.0);
}

#[test]
fn test_f16_to_f32_one() {
    use super::super::dequant::f16_to_f32;

    // 1.0 in f16 = 0x3C00 (sign=0, exp=15, mant=0)
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_negative_one() {
    use super::super::dequant::f16_to_f32;

    // -1.0 in f16 = 0xBC00 (sign=1, exp=15, mant=0)
    let result = f16_to_f32(0xBC00);
    assert!((result - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_infinity() {
    use super::super::dequant::f16_to_f32;

    // +Inf in f16 = 0x7C00 (sign=0, exp=31, mant=0)
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite() && result.is_sign_positive());

    // -Inf in f16 = 0xFC00 (sign=1, exp=31, mant=0)
    let result_neg = f16_to_f32(0xFC00);
    assert!(result_neg.is_infinite() && result_neg.is_sign_negative());
}

#[test]
fn test_f16_to_f32_nan() {
    use super::super::dequant::f16_to_f32;

    // NaN in f16 = 0x7C01 (sign=0, exp=31, mant!=0)
    let result = f16_to_f32(0x7C01);
    assert!(result.is_nan());
}

#[test]
fn test_f16_to_f32_subnormal() {
    use super::super::dequant::f16_to_f32;

    // Smallest positive subnormal: exp=0, mant=1
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0);
    assert!(result < 1e-4); // Very small number
}

#[test]
fn test_f16_to_f32_normal_range() {
    use super::super::dequant::f16_to_f32;

    // 2.0 in f16 = 0x4000 (sign=0, exp=16, mant=0)
    let result = f16_to_f32(0x4000);
    assert!((result - 2.0).abs() < 1e-4);

    // 0.5 in f16 = 0x3800 (sign=0, exp=14, mant=0)
    let result_half = f16_to_f32(0x3800);
    assert!((result_half - 0.5).abs() < 1e-4);
}

#[test]
fn test_extract_scale_min_first_4_blocks() {
    use super::super::dequant::extract_scale_min_apr;

    // 12-byte scales array
    let scales = [
        0b00111111u8,
        0b00111110,
        0b00111101,
        0b00111100, // first 4 bytes (low 6 bits)
        0b00101010,
        0b00101011,
        0b00101100,
        0b00101101, // second 4 bytes (min values)
        0,
        0,
        0,
        0, // last 4 bytes (high bits)
    ];

    // Block 0: scale = scales[0] & 63 = 63, min = scales[4] & 63 = 42
    let (scale, min) = extract_scale_min_apr(&scales, 0);
    assert_eq!(scale, 63.0);
    assert_eq!(min, 42.0);

    // Block 1
    let (scale, min) = extract_scale_min_apr(&scales, 1);
    assert_eq!(scale, 62.0);
    assert_eq!(min, 43.0);
}

#[test]
fn test_extract_scale_min_last_4_blocks() {
    use super::super::dequant::extract_scale_min_apr;

    // Construct scales for blocks 4-7 (packed layout)
    let scales = [
        0b11_000001u8,
        0b11_000010,
        0b11_000011,
        0b11_000100, // first 4 (high bits used for 4-7)
        0b11_000101,
        0b11_000110,
        0b11_000111,
        0b11_001000, // second 4 (mins for 0-3)
        0x15,
        0x26,
        0x37,
        0x48, // last 4 (scale/min for 4-7 low bits)
    ];

    // Block 4: uses packed layout
    // d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = (0x15 & 0x0F) | (0b11 << 4) = 5 | 48 = 53
    let (scale, _min) = extract_scale_min_apr(&scales, 4);
    assert_eq!(scale, 53.0);
}

#[test]
fn test_dequantize_q4_k_empty() {
    use super::super::dequant::dequantize_q4_k_apr;

    let result = dequantize_q4_k_apr(&[], 0);
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q4_k_insufficient_data() {
    use super::super::dequant::dequantize_q4_k_apr;

    // Request 256 elements but provide only 10 bytes (need 144)
    let data = vec![0u8; 10];
    let result = dequantize_q4_k_apr(&data, 256);

    // Should return zeros
    assert_eq!(result.len(), 256);
    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn test_dequantize_q4_k_single_block() {
    use super::super::dequant::dequantize_q4_k_apr;

    // 144 bytes for 256 values
    let mut data = vec![0u8; 144];

    // Set d (f16 scale) at offset 0-1
    // 1.0 in f16 = 0x3C00
    data[0] = 0x00;
    data[1] = 0x3C;

    // Set dmin (f16 min) at offset 2-3 to 0
    data[2] = 0x00;
    data[3] = 0x00;

    // Set scales (12 bytes at offset 4-15) to have scale=1 for all blocks
    for i in 0..4 {
        data[4 + i] = 1; // scale for blocks 0-3
        data[8 + i] = 0; // min for blocks 0-3
    }

    // Set qs (128 bytes at offset 16) to 0 (produces 0*scale values)
    // Each nibble=0, so output = d * scale * 0 - dmin * m = 0 - 0 = 0

    let result = dequantize_q4_k_apr(&data, 256);
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q4_k_truncation() {
    use super::super::dequant::dequantize_q4_k_apr;

    // Provide enough data for 1 block (256 values) but request only 100
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_apr(&data, 100);

    assert_eq!(result.len(), 100);
}

#[test]
fn test_dequantize_q6_k_empty() {
    use super::super::dequant::dequantize_q6_k_apr;

    let result = dequantize_q6_k_apr(&[], 0);
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q6_k_insufficient_data() {
    use super::super::dequant::dequantize_q6_k_apr;

    // Request 256 elements but provide only 10 bytes (need 210)
    let data = vec![0u8; 10];
    let result = dequantize_q6_k_apr(&data, 256);

    // Should return zeros
    assert_eq!(result.len(), 256);
    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn test_dequantize_q6_k_single_block() {
    use super::super::dequant::dequantize_q6_k_apr;

    // 210 bytes for 256 values
    // Layout: ql (128) + qh (64) + scales (16) + d (2) = 210
    let mut data = vec![0u8; 210];

    // Set d (f16 scale) at offset 208-209
    // 1.0 in f16 = 0x3C00
    data[208] = 0x00;
    data[209] = 0x3C;

    let result = dequantize_q6_k_apr(&data, 256);
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q6_k_truncation() {
    use super::super::dequant::dequantize_q6_k_apr;

    // Provide enough data for 1 block but request only 100
    let data = vec![0u8; 210];
    let result = dequantize_q6_k_apr(&data, 100);

    assert_eq!(result.len(), 100);
}

// ============================================================================
// Part 9: SIMD Helpers Tests
// ============================================================================

#[test]
fn test_simd_dot_f32_empty() {
    use super::super::helpers::simd_dot_f32;

    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let result = simd_dot_f32(&a, &b);
    assert_eq!(result, 0.0);
}

#[test]
fn test_simd_dot_f32_small() {
    use super::super::helpers::simd_dot_f32;

    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    let result = simd_dot_f32(&a, &b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((result - 32.0).abs() < 1e-6);
}

#[test]
fn test_simd_dot_f32_exact_8() {
    use super::super::helpers::simd_dot_f32;

    // Exactly 8 elements - exercises AVX2 path without remainder
    let a = [1.0f32; 8];
    let b = [2.0f32; 8];
    let result = simd_dot_f32(&a, &b);

    // 8 * (1.0 * 2.0) = 16.0
    assert!((result - 16.0).abs() < 1e-5);
}

#[test]
fn test_simd_dot_f32_large() {
    use super::super::helpers::simd_dot_f32;

    // Large enough to use AVX2 path with remainder
    let n = 100;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let result = simd_dot_f32(&a, &b);

    // Expected: sum of i * (n - i) for i in 0..n
    let expected: f32 = (0..n).map(|i| (i as f32) * ((n - i) as f32)).sum();
    assert!((result - expected).abs() < 1e-2);
}

#[test]
fn test_simd_add_weighted_empty() {
    use super::super::helpers::simd_add_weighted;

    let mut out: [f32; 0] = [];
    let val: [f32; 0] = [];
    simd_add_weighted(&mut out, &val, 2.0);
    // Should not panic
}

#[test]
fn test_simd_add_weighted_small() {
    use super::super::helpers::simd_add_weighted;

    let mut out = [1.0f32, 2.0, 3.0];
    let val = [1.0f32, 1.0, 1.0];
    simd_add_weighted(&mut out, &val, 2.0);

    // out[i] += 2.0 * val[i]
    assert!((out[0] - 3.0).abs() < 1e-6); // 1 + 2*1 = 3
    assert!((out[1] - 4.0).abs() < 1e-6); // 2 + 2*1 = 4
    assert!((out[2] - 5.0).abs() < 1e-6); // 3 + 2*1 = 5
}

#[test]
fn test_simd_add_weighted_exact_8() {
    use super::super::helpers::simd_add_weighted;

    let mut out = [0.0f32; 8];
    let val = [1.0f32; 8];
    simd_add_weighted(&mut out, &val, 3.0);

    assert!(out.iter().all(|&x| (x - 3.0).abs() < 1e-5));
}

#[test]
fn test_simd_add_weighted_large() {
    use super::super::helpers::simd_add_weighted;

    let n = 100;
    let mut out = vec![1.0f32; n];
    let val: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let weight = 0.5;

    simd_add_weighted(&mut out, &val, weight);

    for (i, &o) in out.iter().enumerate() {
        let expected = 1.0 + 0.5 * (i as f32);
        assert!((o - expected).abs() < 1e-5, "Mismatch at index {i}");
    }
}

#[test]
fn test_simd_add_weighted_negative_weight() {
    use super::super::helpers::simd_add_weighted;

    let mut out = [10.0f32, 20.0, 30.0, 40.0];
    let val = [1.0f32, 2.0, 3.0, 4.0];
    simd_add_weighted(&mut out, &val, -1.0);

    // out[i] -= val[i]
    assert!((out[0] - 9.0).abs() < 1e-6);
    assert!((out[1] - 18.0).abs() < 1e-6);
    assert!((out[2] - 27.0).abs() < 1e-6);
    assert!((out[3] - 36.0).abs() < 1e-6);
}

// ============================================================================
// Part 10: MmapAprTransformer Tests (Error Paths)
// ============================================================================

#[test]
fn test_mmap_transformer_file_not_found() {
    let result = MmapAprTransformer::from_file("/nonexistent/path/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_get_tensor_bytes_out_of_bounds() {
    // Create a minimal valid APR file in memory and test bounds
    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];

    // Set magic
    data[0..4].copy_from_slice(&MAGIC);

    // Set version to 1
    data[4..8].copy_from_slice(&1u32.to_le_bytes());

    // Set minimal config
    data[8..12].copy_from_slice(&64u32.to_le_bytes()); // hidden_dim
    data[12..16].copy_from_slice(&1u32.to_le_bytes()); // num_layers
    data[16..20].copy_from_slice(&4u32.to_le_bytes()); // num_heads
    data[20..24].copy_from_slice(&4u32.to_le_bytes()); // num_kv_heads
    data[24..28].copy_from_slice(&100u32.to_le_bytes()); // vocab_size
    data[28..32].copy_from_slice(&128u32.to_le_bytes()); // intermediate_dim
    data[32..36].copy_from_slice(&256u32.to_le_bytes()); // context_length
    data[36..40].copy_from_slice(&10000.0f32.to_le_bytes()); // rope_theta
    data[40..44].copy_from_slice(&1e-5f32.to_le_bytes()); // eps
    data[44..48].copy_from_slice(&(APR_TRANSFORMER_HEADER_SIZE as u32).to_le_bytes()); // tensor_data_offset

    // Write to temp file
    use std::io::Write;
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();

    let transformer = MmapAprTransformer::from_file(temp_file.path()).unwrap();

    // Try to read beyond file bounds
    let result = transformer.get_tensor_bytes(0, 1000);
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_invalid_magic() {
    use std::io::Write;

    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];
    data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();

    let result = MmapAprTransformer::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_file_too_small() {
    use std::io::Write;

    let data = vec![0u8; 32]; // Too small

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();

    let result = MmapAprTransformer::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_unsupported_version() {
    use std::io::Write;

    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];
    data[0..4].copy_from_slice(&MAGIC);
    data[4..8].copy_from_slice(&99u32.to_le_bytes()); // Invalid version

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();

    let result = MmapAprTransformer::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_accessors() {
    use std::io::Write;

    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];

    data[0..4].copy_from_slice(&MAGIC);
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    data[8..12].copy_from_slice(&64u32.to_le_bytes());
    data[12..16].copy_from_slice(&2u32.to_le_bytes());
    data[16..20].copy_from_slice(&4u32.to_le_bytes());
    data[20..24].copy_from_slice(&4u32.to_le_bytes());
    data[24..28].copy_from_slice(&100u32.to_le_bytes());
    data[28..32].copy_from_slice(&128u32.to_le_bytes());
    data[32..36].copy_from_slice(&256u32.to_le_bytes());
    data[36..40].copy_from_slice(&10000.0f32.to_le_bytes());
    data[40..44].copy_from_slice(&1e-5f32.to_le_bytes());
    data[44..48].copy_from_slice(&(APR_TRANSFORMER_HEADER_SIZE as u32).to_le_bytes());

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();

    let transformer = MmapAprTransformer::from_file(temp_file.path()).unwrap();

    assert!(transformer.is_mmap());
    assert_eq!(transformer.file_size(), data.len());
    assert!(transformer.num_parameters() > 0);
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 2);
}

#[test]
fn test_mmap_transformer_get_tensor_f32() {
    use std::io::Write;

    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];

    data[0..4].copy_from_slice(&MAGIC);
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    data[8..12].copy_from_slice(&64u32.to_le_bytes());
    data[12..16].copy_from_slice(&1u32.to_le_bytes());
    data[16..20].copy_from_slice(&4u32.to_le_bytes());
    data[20..24].copy_from_slice(&4u32.to_le_bytes());
    data[24..28].copy_from_slice(&100u32.to_le_bytes());
    data[28..32].copy_from_slice(&128u32.to_le_bytes());
    data[32..36].copy_from_slice(&256u32.to_le_bytes());
    data[36..40].copy_from_slice(&10000.0f32.to_le_bytes());
    data[40..44].copy_from_slice(&1e-5f32.to_le_bytes());
    data[44..48].copy_from_slice(&(APR_TRANSFORMER_HEADER_SIZE as u32).to_le_bytes());

    // Write some f32 values after header
    let test_values = [1.0f32, 2.0, 3.0, 4.0];
    for (i, &val) in test_values.iter().enumerate() {
        let bytes = val.to_le_bytes();
        let offset = APR_TRANSFORMER_HEADER_SIZE + i * 4;
        data[offset..offset + 4].copy_from_slice(&bytes);
    }

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();

    let transformer = MmapAprTransformer::from_file(temp_file.path()).unwrap();

    let floats = transformer.get_tensor_f32(0, 4).unwrap();
    assert_eq!(floats.len(), 4);
    assert!((floats[0] - 1.0).abs() < 1e-6);
    assert!((floats[1] - 2.0).abs() < 1e-6);
    assert!((floats[2] - 3.0).abs() < 1e-6);
    assert!((floats[3] - 4.0).abs() < 1e-6);
}

// ============================================================================
// Benchmark Infrastructure Tests (apr_transformer/benchmark.rs)
// ============================================================================

use crate::apr_transformer::benchmark::{
    AprBenchmarkResult, AprLoadResult, AprParityComparison, AprPrefillResult,
    APR_CPU_DECODE_THRESHOLD_TOK_S, APR_PARITY_THRESHOLD_PCT, APR_PREFILL_THRESHOLD_TOK_S,
};

#[test]
fn test_apr_benchmark_result_default() {
    let result = AprBenchmarkResult::default();
    assert_eq!(result.tokens_generated, 0);
    assert_eq!(result.total_time_ms, 0.0);
    assert_eq!(result.tokens_per_second, 0.0);
    assert_eq!(result.throughput_p50, 0.0);
    assert_eq!(result.throughput_p99, 0.0);
    assert_eq!(result.throughput_std_dev, 0.0);
    assert_eq!(result.peak_memory_mb, 0.0);
    assert_eq!(result.model_memory_mb, 0.0);
}

#[test]
fn test_apr_benchmark_result_meets_threshold_above() {
    let result = AprBenchmarkResult {
        tokens_per_second: 60.0,
        ..Default::default()
    };
    assert!(result.meets_threshold(APR_CPU_DECODE_THRESHOLD_TOK_S));
}

#[test]
fn test_apr_benchmark_result_meets_threshold_below() {
    let result = AprBenchmarkResult {
        tokens_per_second: 40.0,
        ..Default::default()
    };
    assert!(!result.meets_threshold(APR_CPU_DECODE_THRESHOLD_TOK_S));
}

#[test]
fn test_apr_benchmark_result_meets_threshold_exact() {
    let result = AprBenchmarkResult {
        tokens_per_second: APR_CPU_DECODE_THRESHOLD_TOK_S,
        ..Default::default()
    };
    assert!(result.meets_threshold(APR_CPU_DECODE_THRESHOLD_TOK_S));
}

#[test]
fn test_apr_benchmark_result_compare_to_baseline() {
    let result = AprBenchmarkResult {
        tokens_per_second: 95.0,
        peak_memory_mb: 100.0,
        ..Default::default()
    };
    let baseline = AprBenchmarkResult {
        tokens_per_second: 100.0,
        peak_memory_mb: 80.0,
        ..Default::default()
    };
    let comparison = result.compare_to_baseline(&baseline);
    assert!((comparison.throughput_ratio - 0.95).abs() < 1e-6);
    assert!((comparison.memory_ratio - 1.25).abs() < 1e-6);
    assert_eq!(comparison.parity_threshold_pct, APR_PARITY_THRESHOLD_PCT);
}

#[test]
fn test_apr_benchmark_result_compare_to_baseline_zero_baseline() {
    let result = AprBenchmarkResult {
        tokens_per_second: 50.0,
        peak_memory_mb: 100.0,
        ..Default::default()
    };
    let baseline = AprBenchmarkResult {
        tokens_per_second: 0.0,
        peak_memory_mb: 0.0,
        ..Default::default()
    };
    let comparison = result.compare_to_baseline(&baseline);
    // Division by zero should give 1.0
    assert_eq!(comparison.throughput_ratio, 1.0);
    assert_eq!(comparison.memory_ratio, 1.0);
}

#[test]
fn test_apr_prefill_result_default() {
    let result = AprPrefillResult::default();
    assert_eq!(result.prompt_tokens, 0);
    assert_eq!(result.prefill_time_ms, 0.0);
    assert_eq!(result.prefill_tok_s, 0.0);
}

#[test]
fn test_apr_load_result_default() {
    let result = AprLoadResult::default();
    assert_eq!(result.load_time_ms, 0.0);
}

#[test]
fn test_apr_parity_comparison_is_parity_true() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.96,
        memory_ratio: 1.0,
        parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
    };
    assert!(comparison.is_parity());
}

#[test]
fn test_apr_parity_comparison_is_parity_false() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.90,
        memory_ratio: 1.0,
        parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
    };
    assert!(!comparison.is_parity());
}

#[test]
fn test_apr_parity_comparison_is_parity_exact() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.95,
        memory_ratio: 1.0,
        parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
    };
    assert!(comparison.is_parity());
}

#[test]
fn test_apr_benchmark_constants() {
    assert_eq!(APR_CPU_DECODE_THRESHOLD_TOK_S, 50.0);
    assert_eq!(APR_PREFILL_THRESHOLD_TOK_S, 100.0);
    assert_eq!(APR_PARITY_THRESHOLD_PCT, 95.0);
}

#[test]
fn test_apr_benchmark_result_clone() {
    let result = AprBenchmarkResult {
        tokens_generated: 100,
        tokens_per_second: 50.0,
        throughput_p50: 48.0,
        throughput_p99: 52.0,
        throughput_std_dev: 2.0,
        peak_memory_mb: 200.0,
        model_memory_mb: 150.0,
        total_time_ms: 2000.0,
    };
    let cloned = result.clone();
    assert_eq!(cloned.tokens_generated, 100);
    assert_eq!(cloned.tokens_per_second, 50.0);
}

#[test]
fn test_apr_benchmark_result_debug() {
    let result = AprBenchmarkResult::default();
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("AprBenchmarkResult"));
}

#[test]
fn test_apr_prefill_result_clone_debug() {
    let result = AprPrefillResult {
        prompt_tokens: 128,
        prefill_time_ms: 10.0,
        prefill_tok_s: 12800.0,
    };
    let cloned = result.clone();
    assert_eq!(cloned.prompt_tokens, 128);

    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("AprPrefillResult"));
}

#[test]
fn test_apr_load_result_clone_debug() {
    let result = AprLoadResult {
        load_time_ms: 500.0,
    };
    let cloned = result.clone();
    assert_eq!(cloned.load_time_ms, 500.0);

    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("AprLoadResult"));
}

#[test]
fn test_apr_parity_comparison_clone_debug() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.95,
        memory_ratio: 1.1,
        parity_threshold_pct: 95.0,
    };
    let cloned = comparison.clone();
    assert!((cloned.throughput_ratio - 0.95).abs() < 1e-6);

    let debug_str = format!("{comparison:?}");
    assert!(debug_str.contains("AprParityComparison"));
}
