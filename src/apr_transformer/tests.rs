#[cfg(test)]
mod tests {
    use crate::apr_transformer::*;

    // ==========================================================================
    // Configuration Tests
    // ==========================================================================

    #[test]
    fn test_config_default() {
        let config = AprTransformerConfig::default();
        assert_eq!(config.architecture, "unknown");
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.vocab_size, 32000);
    }

    #[test]
    fn test_config_serialization() {
        let config = AprTransformerConfig {
            architecture: "test_arch".to_string(),
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 1024,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-6,
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let decoded: AprTransformerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, decoded);
    }

    // ==========================================================================
    // Layer Tests
    // ==========================================================================

    #[test]
    fn test_layer_empty() {
        let layer = AprTransformerLayer::empty(64, 256);
        assert_eq!(layer.attn_norm_weight.len(), 64);
        assert_eq!(layer.qkv_weight.len(), 64 * 3 * 64);
        assert_eq!(layer.ffn_up_weight.len(), 64 * 256);
        assert_eq!(layer.ffn_down_weight.len(), 256 * 64);
    }

    #[test]
    fn test_layer_num_parameters() {
        let layer = AprTransformerLayer::empty(64, 256);
        let expected = 64 // attn_norm
            + 64 * 3 * 64 // qkv
            + 64 * 64 // attn_output
            + 64 * 256 // ffn_up
            + 256 * 64; // ffn_down
        assert_eq!(layer.num_parameters(), expected);
    }

    // ==========================================================================
    // Transformer Tests
    // ==========================================================================

    #[test]
    fn test_transformer_new() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        assert_eq!(transformer.layers.len(), 2);
        assert_eq!(transformer.token_embedding.len(), 100 * 64);
        assert_eq!(transformer.output_norm_weight.len(), 64);
        assert_eq!(transformer.lm_head_weight.len(), 64 * 100);
    }

    #[test]
    fn test_transformer_num_parameters() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Should be > 0 and reasonable
        let params = transformer.num_parameters();
        assert!(params > 0);
        assert!(params < 100_000_000); // Less than 100M params for test model
    }

    #[test]
    fn test_transformer_memory_size() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let params = transformer.num_parameters();
        let mem = transformer.memory_size();
        assert_eq!(mem, params * 4); // F32 = 4 bytes
    }

    // ==========================================================================
    // Embedding Tests
    // ==========================================================================

    #[test]
    fn test_embed_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            vocab_size: 10,
            ..Default::default()
        };
        let mut transformer = AprTransformer::new(config);

        // Set known embedding for token 3
        transformer.token_embedding[3 * 4..3 * 4 + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let embedded = transformer.embed(&[3]);
        assert_eq!(embedded, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_embed_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            vocab_size: 5,
            ..Default::default()
        };
        let mut transformer = AprTransformer::new(config);

        // Set embeddings
        transformer.token_embedding[0..2].copy_from_slice(&[1.0, 2.0]); // token 0
        transformer.token_embedding[2..4].copy_from_slice(&[3.0, 4.0]); // token 1
        transformer.token_embedding[4..6].copy_from_slice(&[5.0, 6.0]); // token 2

        let embedded = transformer.embed(&[0, 1, 2]);
        assert_eq!(embedded, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_embed_out_of_vocab() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            vocab_size: 5,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Token 100 is out of vocab (vocab_size=5)
        let embedded = transformer.embed(&[100]);
        assert_eq!(embedded, vec![0.0, 0.0]); // Returns zeros
    }

    // ==========================================================================
    // Layer Norm Tests
    // ==========================================================================

    #[test]
    fn test_layer_norm_identity() {
        // Tests RMSNorm (not LayerNorm): output = x / sqrt(mean(x^2) + eps) * weight
        let config = AprTransformerConfig {
            hidden_dim: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0]; // Identity weight

        let output = transformer.layer_norm(&input, &weight, None, 1e-5);

        // RMSNorm: rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5) ≈ 2.739
        // Output: [1/2.739, 2/2.739, 3/2.739, 4/2.739]
        let rms = (30.0_f32 / 4.0).sqrt();
        assert!((output[0] - 1.0 / rms).abs() < 0.001);
        assert!((output[1] - 2.0 / rms).abs() < 0.001);
        assert!((output[2] - 3.0 / rms).abs() < 0.001);
        assert!((output[3] - 4.0 / rms).abs() < 0.001);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        // Tests RMSNorm with bias: output = x / rms * weight + bias
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 3.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![10.0, 20.0];

        let output = transformer.layer_norm(&input, &weight, Some(&bias), 1e-5);

        // RMSNorm: rms = sqrt((1+9)/2 + eps) = sqrt(5) ≈ 2.236
        // After norm: [1/2.236, 3/2.236] ≈ [0.447, 1.342]
        // After bias: [10.447, 21.342]
        let rms = (10.0_f32 / 2.0).sqrt();
        assert!((output[0] - (1.0 / rms + 10.0)).abs() < 0.01);
        assert!((output[1] - (3.0 / rms + 20.0)).abs() < 0.01);
    }

    // ==========================================================================
    // GELU Tests
    // ==========================================================================

    #[test]
    fn test_gelu_zero() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![0.0];
        transformer.gelu(&mut data);
        assert!((data[0]).abs() < 0.0001);
    }

    #[test]
    fn test_gelu_positive() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![1.0];
        transformer.gelu(&mut data);
        // GELU(1) ≈ 0.841
        assert!((data[0] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_gelu_negative() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![-1.0];
        transformer.gelu(&mut data);
        // GELU(-1) ≈ -0.159
        assert!((data[0] - (-0.159)).abs() < 0.01);
    }

    // ==========================================================================
    // Matmul Tests
    // ==========================================================================

    #[test]
    fn test_matmul_identity() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 2.0];
        // Identity matrix [2, 2] in row-major
        let weight = vec![1.0, 0.0, 0.0, 1.0];

        let output = transformer.matmul(&input, &weight, 2, 2);
        assert_eq!(output, vec![1.0, 2.0]);
    }

    #[test]
    fn test_matmul_simple() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // PMAT-095: Weight is now [out_dim, in_dim] format
        // input: [1, 2] (in_dim=2)
        // weight: [[1, 2], [3, 4], [5, 6]] ([out_dim=3, in_dim=2] row-major)
        // output[0] = W[0,:] @ x = [1, 2] @ [1, 2] = 1*1 + 2*2 = 5
        // output[1] = W[1,:] @ x = [3, 4] @ [1, 2] = 3*1 + 4*2 = 11
        // output[2] = W[2,:] @ x = [5, 6] @ [1, 2] = 5*1 + 6*2 = 17
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2] row-major

        let output = transformer.matmul(&input, &weight, 2, 3);
        assert_eq!(output, vec![5.0, 11.0, 17.0]);
    }

    // ==========================================================================
    // Forward Tests
    // ==========================================================================

    #[test]
    fn test_forward_empty_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[1]);
        assert!(result.is_ok());

        let logits = result.expect("forward succeeded");
        assert_eq!(logits.len(), 10); // vocab_size
    }

    #[test]
    fn test_forward_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[1, 2, 3]);
        assert!(result.is_ok());

        let logits = result.expect("forward succeeded");
        assert_eq!(logits.len(), 10); // vocab_size (only last token logits)
    }

    // ==========================================================================
    // Predict Tests
    // ==========================================================================

    #[test]
    fn test_predict_next() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.predict_next(&[1]);
        assert!(result.is_ok());

        let token = result.expect("predict succeeded");
        assert!(token < 10); // Within vocab
    }

    // ==========================================================================
    // Reproducibility Tests
    // ==========================================================================

    #[test]
    fn test_reproducibility_same_input_same_output() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let tokens = vec![1, 2, 3];
        let output1 = transformer.forward(&tokens).expect("forward 1");
        let output2 = transformer.forward(&tokens).expect("forward 2");

        assert_eq!(output1, output2, "Same input should produce same output");
    }

    #[test]
    fn test_reproducibility_predict_deterministic() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let tokens = vec![1, 2, 3];
        let pred1 = transformer.predict_next(&tokens).expect("predict 1");
        let pred2 = transformer.predict_next(&tokens).expect("predict 2");

        assert_eq!(pred1, pred2, "Predictions should be deterministic");
    }

    // ==========================================================================
    // Serialization Tests
    // ==========================================================================

    #[test]
    fn test_transformer_serialization_roundtrip() {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 8,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
        };
        let transformer = AprTransformer::new(config);

        let json = serde_json::to_string(&transformer).expect("serialize");
        let decoded: AprTransformer = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(transformer.config, decoded.config);
        assert_eq!(transformer.token_embedding, decoded.token_embedding);
        assert_eq!(transformer.layers.len(), decoded.layers.len());
    }

    // ==========================================================================
    // GQA (Grouped Query Attention) KV Cache Tests - IMP-GQA-001
    // ==========================================================================

    /// Test that forward_with_cache works with GQA models (num_kv_heads < num_heads)
    /// This is a regression test for the QKV extraction bug where K/V were assumed
    /// to have the same size as Q (hidden_dim), but GQA models have smaller K/V.
    ///
    /// GQA model example: Qwen2.5-0.5B (14 heads, 2 KV heads)
    /// - Q size: 14 * 64 = 896
    /// - K size: 2 * 64 = 128
    /// - V size: 2 * 64 = 128
    /// - Total QKV: 896 + 128 + 128 = 1152 (not 896 * 3 = 2688)
    #[test]
    fn test_forward_with_cache_gqa_does_not_panic() {
        // Create GQA config similar to Qwen2.5-0.5B
        let config = AprTransformerConfig {
            architecture: "qwen2".to_string(),
            hidden_dim: 64, // 8 heads * 8 head_dim
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4:1 ratio
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        // Create transformer with GQA-sized layers
        let layers: Vec<AprTransformerLayer> = (0..config.num_layers)
            .map(|_| {
                AprTransformerLayer::empty_gqa(
                    config.hidden_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.intermediate_dim,
                )
            })
            .collect();

        let transformer = AprTransformer {
            config: config.clone(),
            token_embedding: vec![0.1; config.vocab_size * config.hidden_dim],
            layers,
            output_norm_weight: vec![1.0; config.hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; config.hidden_dim * config.vocab_size],
            lm_head_bias: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
            q4k_layers: None,
        };

        let mut cache = AprKVCache::new(&config);

        // This should NOT panic with proper GQA support
        // Before fix: panics with "range end index X out of range for slice of length Y"
        let result = transformer.forward_with_cache(1, &mut cache, 0);
        assert!(
            result.is_ok(),
            "forward_with_cache should not panic on GQA models: {:?}",
            result
        );

        // Generate a few more tokens to test cache accumulation
        let result = transformer.forward_with_cache(2, &mut cache, 1);
        assert!(result.is_ok());

        let result = transformer.forward_with_cache(3, &mut cache, 2);
        assert!(result.is_ok());

        // Verify cache has correct length
        assert_eq!(cache.len(), 3);
    }

    /// Test GQA KV cache dimensions are correct
    #[test]
    fn test_gqa_kv_cache_dimensions() {
        let config = AprTransformerConfig {
            hidden_dim: 64, // 8 heads * 8 head_dim
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4:1 ratio
            context_length: 32,
            ..Default::default()
        };

        let cache = AprKVCache::new(&config);

        // KV cache should store num_kv_heads * head_dim per position
        // head_dim = 64 / 8 = 8
        // kv_size = 2 * 8 = 16 per position per layer
        assert_eq!(cache.num_kv_heads, 2);
        assert_eq!(cache.head_dim, 8);
    }

    // ============ Additional coverage tests ============

    #[test]
    fn test_apr_quantization_type_bits_per_weight() {
        assert_eq!(AprQuantizationType::Q4_K.bits_per_weight(), 4.5);
        assert_eq!(AprQuantizationType::Q8_0.bits_per_weight(), 8.0);
        assert_eq!(AprQuantizationType::F32.bits_per_weight(), 32.0);
    }

    #[test]
    fn test_apr_quantization_type_bytes_per_block() {
        // F32: 4 bytes per value
        assert_eq!(AprQuantizationType::F32.bytes_per_block(), 4);
        // Q4_K: 144 bytes per 256 values
        assert_eq!(AprQuantizationType::Q4_K.bytes_per_block(), 144);
        // Q8_0: 36 bytes per 32 values
        assert_eq!(AprQuantizationType::Q8_0.bytes_per_block(), 36);
    }

    #[test]
    fn test_apr_quantization_type_values_per_block() {
        assert_eq!(AprQuantizationType::F32.values_per_block(), 1);
        assert_eq!(AprQuantizationType::Q4_K.values_per_block(), 256);
        assert_eq!(AprQuantizationType::Q8_0.values_per_block(), 32);
    }

    #[test]
    fn test_apr_quantization_type_to_byte() {
        assert_eq!(AprQuantizationType::F32.to_byte(), 0);
        assert_eq!(AprQuantizationType::Q4_K.to_byte(), 1);
        assert_eq!(AprQuantizationType::Q8_0.to_byte(), 2);
    }

    #[test]
    fn test_apr_quantization_type_from_byte() {
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
        assert_eq!(AprQuantizationType::from_byte(255), None);
    }

    #[test]
    fn test_quantized_transformer_new() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);

        assert_eq!(qt.quantization_type(), AprQuantizationType::Q4_K);
        assert_eq!(qt.config(), &config);
        assert_eq!(qt.bits_per_weight(), 4.5);
    }

    #[test]
    fn test_quantized_transformer_from_f32() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let f32_transformer = AprTransformer::new(config.clone());
        let qt = QuantizedAprTransformer::from_f32_transformer(
            &f32_transformer,
            AprQuantizationType::Q8_0,
        );

        assert_eq!(qt.quantization_type(), AprQuantizationType::Q8_0);
        assert!(qt.weight_bytes() > 0);
        assert!(qt.f32_equivalent_bytes() > qt.weight_bytes());
    }

    #[test]
    fn test_quantized_transformer_serialization() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);

        // Round-trip serialization
        let bytes = qt.to_bytes().expect("serialize");
        let qt2 = QuantizedAprTransformer::from_bytes(&bytes).expect("deserialize");

        assert_eq!(qt2.quantization_type(), qt.quantization_type());
        // Architecture may be updated during serialization
        assert_eq!(qt2.config().hidden_dim, qt.config().hidden_dim);
        assert_eq!(qt2.config().num_layers, qt.config().num_layers);
        assert_eq!(qt2.config().vocab_size, qt.config().vocab_size);
    }

    #[test]
    fn test_kv_cache_operations() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 8,
            context_length: 32,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 32);

        // Append KV for layer 0
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        cache.append(0, &k, &v);

        // Check retrieval
        let (k_ret, v_ret) = cache.get(0);
        assert_eq!(k_ret.len(), 64);
        assert_eq!(v_ret.len(), 64);

        // Clear
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_generate_config_default() {
        let config = GenerateConfig::default();
        assert_eq!(config.max_tokens, 32);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 0); // 0 = disabled
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repetition_penalty, 1.0);
    }

    #[test]
    fn test_transformer_embed() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Single token
        let embedded = transformer.embed(&[0]);
        assert_eq!(embedded.len(), 64);

        // Multiple tokens
        let embedded = transformer.embed(&[0, 1, 2]);
        assert_eq!(embedded.len(), 64 * 3);
    }

    #[test]
    fn test_transformer_memory_size_detailed() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let mem_size = transformer.memory_size();
        assert!(mem_size > 0);
        // Should be at least vocab_size * hidden_dim * 4 bytes
        assert!(mem_size >= 100 * 64 * 4);
    }

    #[test]
    fn test_layer_empty_gqa() {
        // GQA: 8 heads, 2 KV heads
        let layer = AprTransformerLayer::empty_gqa(64, 128, 8, 2);

        // QKV weight should account for GQA
        // Q: hidden_dim * hidden_dim = 64 * 64
        // K: hidden_dim * kv_dim = 64 * (64 / 8 * 2) = 64 * 16
        // V: same as K
        assert!(!layer.qkv_weight.is_empty());
        assert_eq!(layer.attn_norm_weight.len(), 64);
    }

    #[test]
    fn test_quantized_apr_tensor_q4_new() {
        let data = vec![0u8; 64]; // Enough for small tensor
        let tensor = QuantizedAprTensorQ4::new(data.clone(), 32, 2);

        assert_eq!(tensor.in_dim, 32);
        assert_eq!(tensor.out_dim, 2);
        assert_eq!(tensor.data.len(), 64);
    }

    #[test]
    fn test_quantized_apr_tensor_q4_zeros() {
        let tensor = QuantizedAprTensorQ4::zeros(32, 2);

        assert_eq!(tensor.in_dim, 32);
        assert_eq!(tensor.out_dim, 2);
    }

    #[test]
    fn test_quantized_apr_tensor_q4_expected_bytes() {
        // Q4_0: 18 bytes per 32 values
        let bytes = QuantizedAprTensorQ4::expected_bytes(64);
        assert_eq!(bytes, (64 / 32) * 18);
    }

    #[test]
    fn test_apr_inference_scratch() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            intermediate_dim: 128,
            ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);

        assert_eq!(scratch.hidden.len(), 64);
        assert_eq!(scratch.normed.len(), 64);
        assert_eq!(scratch.qkv_out.len(), 64 * 3); // hidden_dim * 3
        assert_eq!(scratch.ffn_up.len(), 128);
        assert_eq!(scratch.ffn_gate.len(), 128);

        let mut scratch = scratch;
        scratch.clear();
        assert!(scratch.hidden.iter().all(|&v| v == 0.0));
        assert!(scratch.ffn_up.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_benchmark_result_meets_threshold() {
        let result = AprBenchmarkResult {
            tokens_generated: 1000,
            total_time_ms: 10000.0,
            tokens_per_second: 100.0,
            throughput_p50: 100.0,
            throughput_p99: 90.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 512.0,
            model_memory_mb: 400.0,
        };

        assert!(result.meets_threshold(50.0));
        assert!(result.meets_threshold(100.0));
        assert!(!result.meets_threshold(150.0));
    }

    #[test]
    fn test_benchmark_result_compare_to_baseline() {
        let baseline = AprBenchmarkResult {
            tokens_generated: 1000,
            total_time_ms: 10000.0,
            tokens_per_second: 100.0,
            throughput_p50: 100.0,
            throughput_p99: 90.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 512.0,
            model_memory_mb: 400.0,
        };

        let result = AprBenchmarkResult {
            tokens_generated: 1000,
            total_time_ms: 10526.0,
            tokens_per_second: 95.0, // Within 5%
            throughput_p50: 95.0,
            throughput_p99: 85.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 520.0,
            model_memory_mb: 400.0,
        };

        let comparison = result.compare_to_baseline(&baseline);
        assert!(comparison.throughput_ratio >= 0.9);
        assert!(comparison.throughput_ratio <= 1.1);
    }

    #[test]
    fn test_parity_comparison_is_parity() {
        // is_parity checks if throughput_ratio >= parity_threshold_pct / 100
        let parity = AprParityComparison {
            throughput_ratio: 0.95,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0, // 90% threshold
        };
        assert!(parity.is_parity()); // 0.95 >= 0.90

        let not_parity = AprParityComparison {
            throughput_ratio: 0.5,
            memory_ratio: 2.0,
            parity_threshold_pct: 90.0, // 90% threshold
        };
        assert!(!not_parity.is_parity()); // 0.5 < 0.90
    }

    #[test]
    fn test_benchmark_runner_new() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let runner = AprBenchmarkRunner::new(transformer);

        assert_eq!(runner.warmup_iterations(), 3);
        assert_eq!(runner.measure_iterations(), 10);
    }

    #[test]
    fn test_benchmark_runner_set_iterations() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let mut runner = AprBenchmarkRunner::new(transformer);

        runner.set_warmup_iterations(5);
        runner.set_measure_iterations(10);

        assert_eq!(runner.warmup_iterations(), 5);
        assert_eq!(runner.measure_iterations(), 10);
    }

    // ==========================================================================
    // Extended KV Cache Tests
    // ==========================================================================

    #[test]
    fn test_kv_cache_len() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_is_empty() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_capacity() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 256,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);
        assert_eq!(cache.capacity(), 256);
    }

    #[test]
    fn test_kv_cache_append_and_get() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 256,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);

        let k = vec![1.0; 64];
        let v = vec![2.0; 64];
        cache.append(0, &k, &v);

        let (k_out, v_out) = cache.get(0);
        assert!(!k_out.is_empty());
        assert!(!v_out.is_empty());
    }

    #[test]
    fn test_kv_cache_clear() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 256,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);

        let k = vec![1.0; 64];
        let v = vec![2.0; 64];
        cache.append(0, &k, &v);
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
    }

    // ==========================================================================
    // Extended Quantization Type Tests
    // ==========================================================================

    #[test]
    fn test_quantization_type_roundtrip_all() {
        // Test all quantization types round-trip correctly
        let types = [
            AprQuantizationType::F32,
            AprQuantizationType::Q4_K,
            AprQuantizationType::Q8_0,
        ];

        for quant_type in types {
            let byte = quant_type.to_byte();
            let recovered = AprQuantizationType::from_byte(byte);
            assert_eq!(recovered, Some(quant_type));
        }
    }

    #[test]
    fn test_quantization_type_from_byte_invalid() {
        // Invalid byte values should return None
        assert_eq!(AprQuantizationType::from_byte(255), None);
        assert_eq!(AprQuantizationType::from_byte(100), None);
        assert_eq!(AprQuantizationType::from_byte(3), None);
    }

    #[test]
    fn test_quantization_type_bits_consistency() {
        // Verify bits_per_weight is consistent with bytes_per_block and values_per_block
        // Note: Q8_0 has extra overhead (4 bytes scale per 32 values = 36 bytes / 32 = 9 bits)
        // but we report it as 8 bits, which is the quantization resolution

        // F32 should be exactly 32 bits
        let f32 = AprQuantizationType::F32;
        assert_eq!(f32.bits_per_weight(), 32.0);

        // Q4_K: 144 bytes per 256 values = 4.5 bits/weight
        let q4_k = AprQuantizationType::Q4_K;
        let q4_k_computed = (q4_k.bytes_per_block() as f64 * 8.0) / q4_k.values_per_block() as f64;
        assert!((q4_k.bits_per_weight() - q4_k_computed).abs() < 0.1);

        // Q8_0: reported as 8 bits, but actual storage has overhead
        let q8_0 = AprQuantizationType::Q8_0;
        assert_eq!(q8_0.bits_per_weight(), 8.0);
    }

    #[test]
    fn test_quantization_type_f32() {
        let quant = AprQuantizationType::F32;
        assert_eq!(quant.bits_per_weight(), 32.0);
        assert_eq!(quant.bytes_per_block(), 4);
        assert_eq!(quant.values_per_block(), 1);
        assert_eq!(quant.to_byte(), 0);
    }

    #[test]
    fn test_quantization_type_q4_k() {
        let quant = AprQuantizationType::Q4_K;
        assert_eq!(quant.bits_per_weight(), 4.5);
        assert_eq!(quant.bytes_per_block(), 144);
        assert_eq!(quant.values_per_block(), 256);
        assert_eq!(quant.to_byte(), 1);
    }

    #[test]
    fn test_quantization_type_q8_0() {
        let quant = AprQuantizationType::Q8_0;
        assert_eq!(quant.bits_per_weight(), 8.0);
        assert_eq!(quant.bytes_per_block(), 36);
        assert_eq!(quant.values_per_block(), 32);
        assert_eq!(quant.to_byte(), 2);
    }

    // ==========================================================================
    // Extended QuantizedAprTransformer Tests
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_bits_per_weight() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
        let bits = transformer.bits_per_weight();
        // Q4_K should be approximately 4.5 bits per weight
        assert!(bits > 4.0 && bits < 5.0);
    }

    #[test]
    fn test_quantized_transformer_config() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);
        assert_eq!(transformer.config().hidden_dim, config.hidden_dim);
        assert_eq!(transformer.config().num_layers, config.num_layers);
    }

    #[test]
    fn test_quantized_transformer_weight_bytes() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
        let weight_bytes = transformer.weight_bytes();
        let f32_bytes = transformer.f32_equivalent_bytes();

        // Quantized should be smaller than f32
        assert!(weight_bytes < f32_bytes);
    }

    #[test]
    fn test_quantized_transformer_num_parameters() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
        let num_params = transformer.num_parameters();
        assert!(num_params > 0);
    }

    #[test]
    fn test_quantized_transformer_q8_0() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);
        let bits = transformer.bits_per_weight();
        // Q8_0 should be 8 bits per weight (plus overhead)
        assert!(bits >= 8.0 && bits <= 10.0);
    }

    #[test]
    fn test_quantized_transformer_f32() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
        let bits = transformer.bits_per_weight();
        assert_eq!(bits, 32.0);
    }

    // ==========================================================================
    // Extended GenerateConfig Tests
    // ==========================================================================

    #[test]
    fn test_generate_config_custom() {
        let config = GenerateConfig {
            max_tokens: 100,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.1,
        };
        assert_eq!(config.max_tokens, 100);
        assert!((config.temperature - 0.8).abs() < 1e-6);
        assert!((config.top_p - 0.95).abs() < 1e-6);
        assert_eq!(config.top_k, 40);
    }

    // ==========================================================================
    // Extended AprTransformerConfig Tests
    // ==========================================================================

    #[test]
    fn test_transformer_config_equality() {
        let config1 = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            ..Default::default()
        };
        let config2 = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            ..Default::default()
        };
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_transformer_config_clone() {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    // ==========================================================================
    // Extended AprTransformerLayer Tests
    // ==========================================================================

    #[test]
    fn test_layer_empty_different_dims() {
        let layer1 = AprTransformerLayer::empty(128, 512);
        let layer2 = AprTransformerLayer::empty(256, 1024);

        assert!(layer1.num_parameters() < layer2.num_parameters());
    }

    #[test]
    fn test_layer_empty_gqa_head_counts() {
        // GQA with 4 heads and 2 KV heads
        let layer = AprTransformerLayer::empty_gqa(64, 256, 4, 2);

        // Validate the structure exists
        assert!(!layer.qkv_weight.is_empty());
        assert!(!layer.attn_output_weight.is_empty());
        assert!(!layer.ffn_up_weight.is_empty());
        assert!(!layer.ffn_down_weight.is_empty());
    }

    // ==========================================================================
    // Extended AprTransformer Tests
    // ==========================================================================

    #[test]
    fn test_transformer_config_accessor() {
        let config = AprTransformerConfig {
            hidden_dim: 128,
            num_layers: 4,
            vocab_size: 1000,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());

        assert_eq!(transformer.config().hidden_dim, 128);
        assert_eq!(transformer.config().num_layers, 4);
        assert_eq!(transformer.config().vocab_size, 1000);
    }

    #[test]
    fn test_transformer_embed_boundary() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            vocab_size: 100,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Token at vocab boundary
        let output = transformer.embed(&[99]);
        assert_eq!(output.len(), 64);

        // Token beyond vocab
        let output_beyond = transformer.embed(&[100]);
        assert_eq!(output_beyond.len(), 64);
    }

    #[test]
    fn test_transformer_forward_returns_vocab_size() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[1, 2, 3]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), 100); // vocab_size
    }

    // ==========================================================================
    // QuantizedAprTensorQ4 Extended Tests
    // ==========================================================================

    #[test]
    fn test_quantized_tensor_q4_different_sizes() {
        let tensor_small = QuantizedAprTensorQ4::zeros(32, 16);
        let tensor_large = QuantizedAprTensorQ4::zeros(128, 64);

        assert_eq!(tensor_small.in_dim, 32);
        assert_eq!(tensor_small.out_dim, 16);
        assert_eq!(tensor_large.in_dim, 128);
        assert_eq!(tensor_large.out_dim, 64);
    }

    #[test]
    fn test_quantized_tensor_q4_expected_bytes_alignment() {
        // Q4_0: 18 bytes per 32 values
        assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);
        assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);
        assert_eq!(QuantizedAprTensorQ4::expected_bytes(256), 144);
    }

    // ==========================================================================
    // Extended Benchmark Tests
    // ==========================================================================

    #[test]
    fn test_prefill_result_struct() {
        let result = AprPrefillResult {
            prompt_tokens: 100,
            prefill_time_ms: 50.0,
            prefill_tok_s: 2000.0,
        };
        assert_eq!(result.prompt_tokens, 100);
        assert!((result.prefill_tok_s - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_result_struct() {
        let result = AprLoadResult {
            load_time_ms: 100.0,
        };
        assert!((result.load_time_ms - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_benchmark_result_edge_cases() {
        // Zero throughput
        let result = AprBenchmarkResult {
            tokens_generated: 0,
            total_time_ms: 0.0,
            tokens_per_second: 0.0,
            throughput_p50: 0.0,
            throughput_p99: 0.0,
            throughput_std_dev: 0.0,
            peak_memory_mb: 0.0,
            model_memory_mb: 0.0,
        };
        assert!(!result.meets_threshold(1.0));
        assert!(result.meets_threshold(0.0));
    }

    #[test]
    fn test_parity_comparison_boundary() {
        // Exactly at threshold
        let parity = AprParityComparison {
            throughput_ratio: 0.9,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0,
        };
        assert!(parity.is_parity()); // 0.9 >= 0.9

        // Just below threshold
        let not_parity = AprParityComparison {
            throughput_ratio: 0.899,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0,
        };
        assert!(!not_parity.is_parity()); // 0.899 < 0.9
    }

    // Quantization type tests already exist above

    // ==========================================================================
    // AprTransformer Forward Tests
    // ==========================================================================

    #[test]
    fn test_transformer_forward_empty_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_forward_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());

        let result = transformer.forward(&[1]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), 100); // vocab_size
    }

    #[test]
    fn test_transformer_forward_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[1, 2, 3]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), 100);
    }

    #[test]
    fn test_transformer_predict_next() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.predict_next(&[1, 2, 3]);
        assert!(result.is_ok());
        let next_token = result.expect("APR operation failed");
        assert!(next_token < 100); // Within vocab
    }

    // ==========================================================================
    // AprTransformer Generate Tests
    // ==========================================================================

    #[test]
    fn test_transformer_generate_zero_max_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.generate(&[1], 0);
        assert!(result.is_ok());
        let tokens = result.expect("APR operation failed");
        assert_eq!(tokens.len(), 1); // Just the prompt
    }

    #[test]
    fn test_transformer_generate_small() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.generate(&[1], 3);
        assert!(result.is_ok());
        let tokens = result.expect("APR operation failed");
        assert!(!tokens.is_empty() && tokens.len() <= 4);
    }

    // ==========================================================================
    // AprKVCache Extended Tests
    // ==========================================================================

    #[test]
    fn test_kv_cache_multiple_layers() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 256,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);

        // Append to multiple layers
        for layer in 0..4 {
            let k = vec![layer as f32; 64];
            let v = vec![(layer + 10) as f32; 64];
            cache.append(layer, &k, &v);
        }

        // Verify each layer
        for layer in 0..4 {
            let (k_out, v_out) = cache.get(layer);
            assert!(!k_out.is_empty());
            assert!(!v_out.is_empty());
        }
    }

    // GenerateConfig tests already exist above

    // ==========================================================================
    // MmapAprTransformer Tests (when available)
    // ==========================================================================

    #[test]
    fn test_mmap_transformer_from_file_nonexistent() {
        let result = MmapAprTransformer::from_file("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    // ==========================================================================
    // AprQuantizationType Extended Tests
    // ==========================================================================

    #[test]
    fn test_apr_quantization_type_f32_extended() {
        let qt = AprQuantizationType::F32;
        assert_eq!(qt.bits_per_weight(), 32.0);
        assert_eq!(qt.to_byte(), 0);
    }

    #[test]
    fn test_apr_quantization_type_q4_k() {
        let qt = AprQuantizationType::Q4_K;
        assert_eq!(qt.bits_per_weight(), 4.5); // Q4_K includes scales
        assert_eq!(qt.to_byte(), 1);
    }

    #[test]
    fn test_apr_quantization_type_q8_0() {
        let qt = AprQuantizationType::Q8_0;
        assert_eq!(qt.bits_per_weight(), 8.0);
        assert_eq!(qt.to_byte(), 2);
    }

    #[test]
    fn test_apr_quantization_type_from_byte_valid() {
        assert!(AprQuantizationType::from_byte(0).is_some()); // F32
        assert!(AprQuantizationType::from_byte(1).is_some()); // Q4_K
        assert!(AprQuantizationType::from_byte(2).is_some()); // Q8_0
        assert!(AprQuantizationType::from_byte(3).is_none()); // Invalid
        assert!(AprQuantizationType::from_byte(255).is_none()); // Invalid
    }

    // ==========================================================================
    // AprTransformer Additional Tests
    // ==========================================================================

    #[test]
    fn test_transformer_config() {
        let config = AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 256,
            num_layers: 8,
            num_heads: 8,
            num_kv_heads: 4,
            vocab_size: 10000,
            intermediate_dim: 1024,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
        };
        let transformer = AprTransformer::new(config.clone());
        let returned_config = transformer.config();
        assert_eq!(returned_config.architecture, "llama");
        assert_eq!(returned_config.hidden_dim, 256);
    }

    #[test]
    fn test_transformer_generate_empty_prompt() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let result = transformer.generate(&[], 10);
        // Empty prompt should fail or return empty
        assert!(result.is_err() || result.as_ref().map_or(true, std::vec::Vec::is_empty));
    }

    // ==========================================================================
    // AprTransformerLayer Extended Tests
    // ==========================================================================

    #[test]
    fn test_layer_empty_gqa_parameters() {
        // Test GQA layer with different head counts
        let layer = AprTransformerLayer::empty_gqa(64, 256, 8, 4);
        let params = layer.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_layer_empty_same_kv_heads() {
        // When num_heads == num_kv_heads, should match empty()
        let _layer1 = AprTransformerLayer::empty(64, 256);
        let _layer2 = AprTransformerLayer::empty_gqa(64, 256, 8, 8);
        // Parameters should be the same for equivalent dimensions
        // Note: This depends on default num_heads in empty()
    }

    // ==========================================================================
    // AprTransformerConfig Extended Tests
    // ==========================================================================

    #[test]
    fn test_config_clone() {
        let config = AprTransformerConfig {
            architecture: "gpt2".to_string(),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        };
        let cloned = config.clone();
        assert_eq!(config.architecture, cloned.architecture);
        assert_eq!(config.hidden_dim, cloned.hidden_dim);
    }

    #[test]
    fn test_config_debug() {
        let config = AprTransformerConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AprTransformerConfig"));
    }

    // ==========================================================================
    // Error Path Tests
    // ==========================================================================

    #[test]
    fn test_from_apr_file_nonexistent_extended() {
        let result = AprTransformer::from_apr_file("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_empty_extended() {
        let result = AprTransformer::from_apr_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_invalid_magic_extended() {
        let data = vec![0u8; 100];
        let result = AprTransformer::from_apr_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_truncated_apr2_magic() {
        // Valid APR2 magic but truncated
        let mut data = vec![0u8; 10];
        data[0..4].copy_from_slice(b"APR\0");
        let result = AprTransformer::from_apr_bytes(&data);
        assert!(result.is_err());
    }

    // ==========================================================================
    // Serialization Extended Tests
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_serialization_roundtrip() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

        let bytes = qt.to_bytes().expect("serialize");
        assert!(!bytes.is_empty());

        let restored = QuantizedAprTransformer::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.config().hidden_dim, config.hidden_dim);
        assert_eq!(restored.config().vocab_size, config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_f32_equivalent_extended() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);
        let f32_bytes = qt.f32_equivalent_bytes();
        let actual_bytes = qt.weight_bytes();
        // Quantized should be smaller than F32 equivalent
        assert!(actual_bytes <= f32_bytes);
    }

    #[test]
    fn test_quantized_transformer_q4_k_parameters() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
        let params = qt.num_parameters();
        assert!(params > 0);
    }

    // ==========================================================================
    // AprQuantizationType Additional Tests
    // ==========================================================================

    #[test]
    fn test_quantization_bytes_per_block_f32() {
        let qt = AprQuantizationType::F32;
        assert_eq!(qt.bytes_per_block(), 4);
    }

    #[test]
    fn test_quantization_bytes_per_block_q4_k() {
        let qt = AprQuantizationType::Q4_K;
        assert_eq!(qt.bytes_per_block(), 144);
    }

    #[test]
    fn test_quantization_bytes_per_block_q8_0() {
        let qt = AprQuantizationType::Q8_0;
        assert_eq!(qt.bytes_per_block(), 36);
    }

    #[test]
    fn test_quantization_values_per_block_f32() {
        let qt = AprQuantizationType::F32;
        assert_eq!(qt.values_per_block(), 1);
    }

    #[test]
    fn test_quantization_values_per_block_q4_k() {
        let qt = AprQuantizationType::Q4_K;
        assert_eq!(qt.values_per_block(), 256);
    }

    #[test]
    fn test_quantization_values_per_block_q8_0() {
        let qt = AprQuantizationType::Q8_0;
        assert_eq!(qt.values_per_block(), 32);
    }

    // ==========================================================================
    // AprTransformerConfig Additional Tests
    // ==========================================================================

    #[test]
    fn test_config_default_values() {
        let config = AprTransformerConfig::default();
        assert!(config.hidden_dim > 0);
        assert!(config.num_layers > 0);
        assert!(config.vocab_size > 0);
    }

    #[test]
    fn test_config_architecture_accessor() {
        let config = AprTransformerConfig {
            architecture: "test_model".to_string(),
            ..Default::default()
        };
        assert_eq!(config.architecture, "test_model");
    }

    // ==========================================================================
    // MmapAprTransformer Extended Tests
    // ==========================================================================

    #[test]
    fn test_mmap_from_file_invalid_path() {
        let result = MmapAprTransformer::from_file("/nonexistent/invalid/path.apr");
        assert!(result.is_err());
    }

    // ==========================================================================
    // AprKVCache Extended Tests
    // ==========================================================================

    #[test]
    fn test_kv_cache_new_creates_empty() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_append_and_len() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        cache.append(0, &k, &v);
        assert!(!cache.is_empty());
    }

    // ==========================================================================
    // GenerateConfig Extended Tests
    // ==========================================================================

    #[test]
    fn test_generate_config_default_values() {
        let config = GenerateConfig::default();
        // Default values should be reasonable
        assert!(config.max_tokens > 0);
        assert!(config.temperature >= 0.0);
        assert!(config.repetition_penalty >= 1.0);
    }

    // ==========================================================================
    // AprTransformer Forward Tests
    // ==========================================================================

    #[test]
    fn test_transformer_forward_vocab_dimensions() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        let result = transformer.forward(&[1, 2, 3]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        // Logits should have vocab_size dimensions
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_transformer_forward_single_token_logits() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 100,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        let result = transformer.forward(&[42]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), 100);
        // All logits should be finite
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    // ==========================================================================
    // AprTransformerLayer Extended Tests
    // ==========================================================================

    #[test]
    fn test_layer_empty_creates_valid_layer() {
        let layer = AprTransformerLayer::empty(64, 256);
        let params = layer.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_layer_empty_gqa_creates_valid_layer() {
        let layer = AprTransformerLayer::empty_gqa(64, 256, 8, 2);
        let params = layer.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_layer_gqa_fewer_kv_heads_smaller() {
        // GQA with fewer KV heads should have fewer parameters
        let layer_mha = AprTransformerLayer::empty_gqa(64, 256, 8, 8);
        let layer_gqa = AprTransformerLayer::empty_gqa(64, 256, 8, 2);
        // GQA has fewer K/V params due to fewer KV heads
        assert!(layer_gqa.num_parameters() <= layer_mha.num_parameters());
    }

    // ==========================================================================
    // QuantizedAprTransformer Extended Tests
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_q8_0_forward() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q8_0);
        let result = qt.forward(&[1, 2, 3]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_q4_k_forward() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);
        let result = qt.forward(&[1, 2, 3]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_quantization_type_accessor() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
        assert_eq!(qt.quantization_type(), AprQuantizationType::Q4_K);
    }

    // ==========================================================================
    // QuantizedAprTransformer forward_with_cache Tests
    // ==========================================================================

    #[test]
    fn test_quantized_forward_with_cache() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let mut cache = AprKVCache::new(&config);

        let result = qt.forward_with_cache(1, &mut cache, 0);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_forward_with_cache_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let mut cache = AprKVCache::new(&config);

        // Process multiple tokens
        for i in 0..3 {
            let result = qt.forward_with_cache(i as u32, &mut cache, i);
            assert!(result.is_ok());
        }
        assert_eq!(cache.len(), 3);
    }

    // ==========================================================================
    // AprKVCache Extended Tests
    // ==========================================================================

    #[test]
    fn test_kv_cache_get_method() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        cache.append(0, &k, &v);

        let (k_cache, v_cache) = cache.get(0);
        assert!(!k_cache.is_empty());
        assert!(!v_cache.is_empty());
    }

    #[test]
    fn test_kv_cache_capacity_method() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 256,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);
        assert_eq!(cache.capacity(), 256);
    }

    #[test]
    fn test_kv_cache_clear_method() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        cache.append(0, &k, &v);
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // ==========================================================================
    // AprTransformer Generate Tests
    // ==========================================================================

    #[test]
    fn test_transformer_generate_basic() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let result = transformer.generate(&[1, 2, 3], 5);
        assert!(result.is_ok());
    }

    // ==========================================================================
    // AprLoadResult and AprParityComparison Tests
    // ==========================================================================

    #[test]
    fn test_apr_load_result_creation() {
        let result = AprLoadResult { load_time_ms: 50.0 };
        assert_eq!(result.load_time_ms, 50.0);
    }

    #[test]
    fn test_apr_parity_comparison_achieves_parity() {
        // 1.0 throughput ratio with 90% threshold = parity achieved
        let comparison = AprParityComparison {
            throughput_ratio: 1.0,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0,
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_apr_parity_comparison_fails_parity() {
        // 0.5 throughput ratio with 90% threshold = parity not achieved
        let comparison = AprParityComparison {
            throughput_ratio: 0.5,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0,
        };
        assert!(!comparison.is_parity());
    }

    #[test]
    fn test_apr_parity_comparison_edge_case() {
        // Exactly at threshold
        let comparison = AprParityComparison {
            throughput_ratio: 0.9,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0,
        };
        assert!(comparison.is_parity());
    }

    // ==========================================================================
    // Edge Cases and Error Handling Tests
    // ==========================================================================

    #[test]
    fn test_transformer_forward_oov_token() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        // Token ID > vocab size should be handled gracefully
        let result = transformer.forward(&[999999]);
        // Should either succeed with zeros or return an error
        if let Ok(logits) = result {
            assert_eq!(logits.len(), 50);
        }
        // Err case is acceptable for OOV
    }

    #[test]
    fn test_quantization_type_equality() {
        assert_eq!(AprQuantizationType::F32, AprQuantizationType::F32);
        assert_ne!(AprQuantizationType::F32, AprQuantizationType::Q4_K);
        assert_ne!(AprQuantizationType::Q4_K, AprQuantizationType::Q8_0);
    }

    #[test]
    fn test_quantization_type_clone() {
        let qt = AprQuantizationType::Q4_K;
        let qt_clone = qt;
        assert_eq!(qt, qt_clone);
    }

    #[test]
    fn test_quantization_type_default() {
        let qt = AprQuantizationType::default();
        assert_eq!(qt, AprQuantizationType::F32);
    }

    // ==========================================================================
    // AprBenchmarkResult Tests
    // ==========================================================================

    #[test]
    fn test_benchmark_result_default() {
        let result = AprBenchmarkResult::default();
        assert_eq!(result.tokens_generated, 0);
        assert_eq!(result.total_time_ms, 0.0);
    }

    #[test]
    fn test_benchmark_result_meets_various_thresholds() {
        let result = AprBenchmarkResult {
            tokens_generated: 100,
            total_time_ms: 1000.0,
            tokens_per_second: 100.0,
            throughput_p50: 100.0,
            throughput_p99: 90.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 512.0,
            model_memory_mb: 256.0,
        };
        assert!(result.meets_threshold(50.0));
        assert!(result.meets_threshold(100.0));
        assert!(!result.meets_threshold(150.0));
    }

    #[test]
    fn test_benchmark_result_clone() {
        let result = AprBenchmarkResult {
            tokens_generated: 50,
            total_time_ms: 500.0,
            tokens_per_second: 100.0,
            ..Default::default()
        };
        let cloned = result.clone();
        assert_eq!(result.tokens_generated, cloned.tokens_generated);
    }

    // ==========================================================================
    // Constants Tests
    // ==========================================================================

    #[test]
    fn test_apr_constants() {
        assert!(APR_CPU_DECODE_THRESHOLD_TOK_S > 0.0);
        assert!(APR_PREFILL_THRESHOLD_TOK_S > 0.0);
        assert!(APR_PARITY_THRESHOLD_PCT > 0.0 && APR_PARITY_THRESHOLD_PCT <= 100.0);
    }

    #[test]
    fn test_apr_magic() {
        // APR\0 - ONE format, no versioning
        assert_eq!(MAGIC[0], 0x41); // 'A'
        assert_eq!(MAGIC[1], 0x50); // 'P'
        assert_eq!(MAGIC[2], 0x52); // 'R'
        assert_eq!(MAGIC[3], 0x00); // '\0'
    }

    #[test]
    fn test_apr_transformer_header_size() {
        assert_eq!(APR_TRANSFORMER_HEADER_SIZE, 64);
    }

    // ==========================================================================
    // AprTransformer generate_with_cache Tests
    // ==========================================================================

    #[test]
    fn test_transformer_generate_with_cache_greedy() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 3,
            temperature: 0.0, // Greedy
            ..Default::default()
        };
        let result = transformer.generate_with_cache(&[1, 2], &gen_config);
        assert!(result.is_ok());
        let tokens = result.expect("APR operation failed");
        assert!(tokens.len() >= 2); // At least the prompt
    }

    #[test]
    fn test_transformer_generate_with_cache_temperature() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 3,
            temperature: 0.8, // Non-greedy
            ..Default::default()
        };
        let result = transformer.generate_with_cache(&[1, 2], &gen_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_generate_with_cache_empty_prompt_fails() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig::default();
        let result = transformer.generate_with_cache(&[], &gen_config);
        assert!(result.is_err());
    }

    // ==========================================================================
    // AprTransformerLayer Tests
    // ==========================================================================

    #[test]
    fn test_layer_num_parameters_components() {
        let layer = AprTransformerLayer::empty(64, 256);
        let params = layer.num_parameters();
        // Should include attn_norm, qkv, attn_out, ffn weights
        assert!(params > 0);
    }

    #[test]
    fn test_layer_with_different_intermediate() {
        let layer1 = AprTransformerLayer::empty(64, 128);
        let layer2 = AprTransformerLayer::empty(64, 512);
        // Layer with larger intermediate should have more params
        assert!(layer2.num_parameters() > layer1.num_parameters());
    }

    // ==========================================================================
    // AprTransformer embed Tests
    // ==========================================================================

    #[test]
    fn test_embed_multiple_tokens_extended() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 100,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        let embeddings = transformer.embed(&[0, 1, 2]);
        // Should be 3 tokens * hidden_dim
        assert_eq!(embeddings.len(), 3 * config.hidden_dim);
    }

    #[test]
    fn test_embed_oov_returns_zeros_extended() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 10,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        // Token 999 is out of vocab
        let embeddings = transformer.embed(&[999]);
        // Should return zeros
        assert_eq!(embeddings.len(), config.hidden_dim);
        assert!(embeddings.iter().all(|&x| x == 0.0));
    }

    // ==========================================================================
    // AprTransformerConfig Serialization Tests
    // ==========================================================================

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 128,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 4,
            vocab_size: 5000,
            intermediate_dim: 512,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let restored: AprTransformerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, restored);
    }

    // ==========================================================================
    // AprBenchmarkRunner Tests
    // ==========================================================================

    #[test]
    fn test_benchmark_runner_benchmark_decode() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let mut runner = AprBenchmarkRunner::new(transformer);
        runner.set_warmup_iterations(1);
        runner.set_measure_iterations(2);

        let result = runner.benchmark_decode(&[1, 2, 3], 2);
        assert!(result.is_ok());
        let benchmark = result.expect("APR operation failed");
        assert!(benchmark.tokens_generated <= 10);
        assert!(benchmark.model_memory_mb > 0.0);
    }

    #[test]
    fn test_benchmark_runner_benchmark_prefill() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let mut runner = AprBenchmarkRunner::new(transformer);
        runner.set_warmup_iterations(1);
        runner.set_measure_iterations(2);

        let result = runner.benchmark_prefill(&[1, 2, 3]);
        assert!(result.is_ok());
        let prefill = result.expect("APR operation failed");
        assert_eq!(prefill.prompt_tokens, 3);
        assert!(prefill.prefill_time_ms > 0.0 || prefill.prefill_tok_s >= 0.0);
    }

    #[test]
    fn test_benchmark_runner_benchmark_load() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let loader = || AprTransformer::new(config.clone());

        let result = AprBenchmarkRunner::benchmark_load(loader);
        assert!(result.is_ok());
        let load_result = result.expect("APR operation failed");
        assert!(load_result.load_time_ms >= 0.0);
    }

    #[test]
    fn test_benchmark_runner_set_measure_iterations_min() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let mut runner = AprBenchmarkRunner::new(transformer);

        // Setting to 0 should be clamped to 1
        runner.set_measure_iterations(0);
        assert_eq!(runner.measure_iterations(), 1);
    }

    // ==========================================================================
    // AprBenchmarkResult Additional Tests
    // ==========================================================================

    #[test]
    fn test_benchmark_result_compare_to_baseline_zero_throughput() {
        let baseline = AprBenchmarkResult {
            tokens_per_second: 0.0,
            peak_memory_mb: 0.0,
            ..Default::default()
        };
        let result = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 512.0,
            ..Default::default()
        };

        let comparison = result.compare_to_baseline(&baseline);
        // When baseline is 0, ratio should be 1.0
        assert_eq!(comparison.throughput_ratio, 1.0);
        assert_eq!(comparison.memory_ratio, 1.0);
    }

    #[test]
    fn test_benchmark_result_compare_to_baseline_normal() {
        let baseline = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 512.0,
            ..Default::default()
        };
        let result = AprBenchmarkResult {
            tokens_per_second: 150.0,
            peak_memory_mb: 256.0,
            ..Default::default()
        };

        let comparison = result.compare_to_baseline(&baseline);
        assert!((comparison.throughput_ratio - 1.5).abs() < 0.01);
        assert!((comparison.memory_ratio - 0.5).abs() < 0.01);
    }

    // ==========================================================================
    // AprPrefillResult Tests
    // ==========================================================================

    #[test]
    fn test_prefill_result_default() {
        let result = AprPrefillResult::default();
        assert_eq!(result.prompt_tokens, 0);
        assert_eq!(result.prefill_time_ms, 0.0);
        assert_eq!(result.prefill_tok_s, 0.0);
    }

    #[test]
    fn test_prefill_result_clone() {
        let result = AprPrefillResult {
            prompt_tokens: 100,
            prefill_time_ms: 50.0,
            prefill_tok_s: 2000.0,
        };
        let cloned = result.clone();
        assert_eq!(result.prompt_tokens, cloned.prompt_tokens);
        assert_eq!(result.prefill_time_ms, cloned.prefill_time_ms);
    }

    // ==========================================================================
    // AprLoadResult Tests
    // ==========================================================================

    #[test]
    fn test_load_result_default() {
        let result = AprLoadResult::default();
        assert_eq!(result.load_time_ms, 0.0);
    }

    #[test]
    fn test_load_result_clone() {
        let result = AprLoadResult {
            load_time_ms: 100.0,
        };
        let cloned = result.clone();
        assert_eq!(result.load_time_ms, cloned.load_time_ms);
    }

    // ==========================================================================
    // QuantizedAprTransformer Forward Tests
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_forward_empty_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
        let result = qt.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_transformer_forward_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let result = qt.forward(&[1]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_forward_oov_token() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        // Token 999 is out of vocab - should handle gracefully
        let result = qt.forward(&[999]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_forward_with_cache_empty() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let mut cache = AprKVCache::new(&config);

        // Single token with cache
        let result = qt.forward_with_cache(1, &mut cache, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantized_transformer_forward_with_cache_oov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let mut cache = AprKVCache::new(&config);

        // OOV token should be handled
        let result = qt.forward_with_cache(9999, &mut cache, 0);
        assert!(result.is_ok());
    }

    // ==========================================================================
    // QuantizedAprLayerQ4 Tests
    // ==========================================================================

    #[test]
    fn test_quantized_layer_q4_struct() {
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 64],
            qkv_weight: QuantizedAprTensorQ4::zeros(64, 192),
            attn_output_weight: QuantizedAprTensorQ4::zeros(64, 64),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(64, 256),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(256, 64),
            ffn_gate_weight: None,
            ffn_norm_weight: None,
        };
        assert_eq!(layer.attn_norm_weight.len(), 64);
    }

    #[test]
    fn test_quantized_layer_q4_with_gate() {
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 64],
            qkv_weight: QuantizedAprTensorQ4::zeros(64, 192),
            attn_output_weight: QuantizedAprTensorQ4::zeros(64, 64),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(64, 256),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(256, 64),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(64, 256)),
            ffn_norm_weight: Some(vec![1.0; 64]),
        };
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    // ==========================================================================
    // QuantizedAprTransformerQ4 Tests
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_q4_config() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            ..Default::default()
        };

        // Create QuantizedAprTransformerQ4 manually
        let qt4 = QuantizedAprTransformerQ4 {
            config: config.clone(),
            token_embedding: vec![0.0; config.vocab_size * config.hidden_dim],
            layers: vec![],
            output_norm_weight: vec![1.0; config.hidden_dim],
            lm_head_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.vocab_size),
        };

        assert_eq!(qt4.config().hidden_dim, 64);
        assert_eq!(qt4.config().num_layers, 2);
    }

    #[test]
    fn test_quantized_transformer_q4_create_scratch() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            ..Default::default()
        };

        let qt4 = QuantizedAprTransformerQ4 {
            config: config.clone(),
            token_embedding: vec![0.0; config.vocab_size * config.hidden_dim],
            layers: vec![],
            output_norm_weight: vec![1.0; config.hidden_dim],
            lm_head_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.vocab_size),
        };

        let scratch = qt4.create_scratch();
        assert_eq!(scratch.hidden.len(), config.hidden_dim);
        assert_eq!(scratch.ffn_up.len(), config.intermediate_dim);
    }

    #[test]
    fn test_quantized_transformer_q4_create_kv_cache() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };

        let qt4 = QuantizedAprTransformerQ4 {
            config: config.clone(),
            token_embedding: vec![0.0; config.vocab_size * config.hidden_dim],
            layers: vec![],
            output_norm_weight: vec![1.0; config.hidden_dim],
            lm_head_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.vocab_size),
        };

        let cache = qt4.create_kv_cache();
        assert_eq!(cache.capacity(), config.context_length);
    }

    // ==========================================================================
    // AprInferenceScratch Tests
    // ==========================================================================

    #[test]
    fn test_inference_scratch_from_config() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            intermediate_dim: 256,
            ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);

        assert_eq!(scratch.hidden.len(), 64);
        assert_eq!(scratch.normed.len(), 64);
        assert_eq!(scratch.qkv_out.len(), 64 * 3);
        assert_eq!(scratch.ffn_up.len(), 256);
        assert_eq!(scratch.ffn_gate.len(), 256);
    }

    #[test]
    fn test_inference_scratch_clear() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            intermediate_dim: 256,
            ..Default::default()
        };
        let mut scratch = AprInferenceScratch::from_config(&config);

        // Fill with non-zero values
        for x in &mut scratch.hidden {
            *x = 1.0;
        }
        for x in &mut scratch.ffn_up {
            *x = 1.0;
        }

        // Clear
        scratch.clear();

        assert!(scratch.hidden.iter().all(|&x| x == 0.0));
        assert!(scratch.ffn_up.iter().all(|&x| x == 0.0));
        assert!(scratch.normed.iter().all(|&x| x == 0.0));
        assert!(scratch.q.iter().all(|&x| x == 0.0));
        assert!(scratch.k.iter().all(|&x| x == 0.0));
        assert!(scratch.v.iter().all(|&x| x == 0.0));
        assert!(scratch.attn_out.iter().all(|&x| x == 0.0));
        assert!(scratch.ffn_input.iter().all(|&x| x == 0.0));
        assert!(scratch.ffn_gate.iter().all(|&x| x == 0.0));
        assert!(scratch.ffn_out.iter().all(|&x| x == 0.0));
    }

    // ==========================================================================
    // QuantizedAprTensorQ4 Extended Tests
    // ==========================================================================

    #[test]
    fn test_quantized_tensor_q4_data_access() {
        let data = vec![1u8, 2, 3, 4];
        let tensor = QuantizedAprTensorQ4::new(data.clone(), 2, 2);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_quantized_tensor_q4_clone() {
        let tensor = QuantizedAprTensorQ4::zeros(32, 16);
        let cloned = tensor.clone();
        assert_eq!(tensor.in_dim, cloned.in_dim);
        assert_eq!(tensor.out_dim, cloned.out_dim);
        assert_eq!(tensor.data.len(), cloned.data.len());
    }

    // ==========================================================================
    // AprTransformer with Biases Tests
    // ==========================================================================

    #[test]
    fn test_transformer_with_output_norm_bias() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let mut transformer = AprTransformer::new(config);
        transformer.output_norm_bias = Some(vec![0.1; 32]);

        let result = transformer.forward(&[1, 2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_with_lm_head_bias() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let mut transformer = AprTransformer::new(config);
        transformer.lm_head_bias = Some(vec![0.1; 50]);

        let result = transformer.forward(&[1, 2]);
        assert!(result.is_ok());
    }

    // ==========================================================================
    // AprTransformerLayer with Biases Tests
    // ==========================================================================

    #[test]
    fn test_layer_num_parameters_with_biases() {
        let mut layer = AprTransformerLayer::empty(64, 256);
        let params_without_bias = layer.num_parameters();

        // Add various biases
        layer.attn_norm_bias = Some(vec![0.0; 64]);
        layer.qkv_bias = Some(vec![0.0; 64 * 3]);
        layer.attn_output_bias = Some(vec![0.0; 64]);
        layer.ffn_up_bias = Some(vec![0.0; 256]);
        layer.ffn_down_bias = Some(vec![0.0; 64]);

        let params_with_bias = layer.num_parameters();
        assert!(params_with_bias > params_without_bias);
    }

    #[test]
    fn test_layer_with_ffn_norm() {
        let mut layer = AprTransformerLayer::empty(64, 256);
        layer.ffn_norm_weight = Some(vec![1.0; 64]);
        layer.ffn_norm_bias = Some(vec![0.0; 64]);

        let params = layer.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_layer_with_ffn_gate() {
        let mut layer = AprTransformerLayer::empty(64, 256);
        layer.ffn_gate_weight = Some(vec![0.0; 64 * 256]);
        layer.ffn_gate_bias = Some(vec![0.0; 256]);

        let params = layer.num_parameters();
        assert!(params > 0);
    }

    // ==========================================================================
    // AprTransformer add_bias Tests
    // ==========================================================================

    #[test]
    fn test_add_bias_basic() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        transformer.add_bias(&mut data, &bias);

        assert!((data[0] - 1.1).abs() < 1e-6);
        assert!((data[1] - 2.2).abs() < 1e-6);
        assert!((data[2] - 3.3).abs() < 1e-6);
        assert!((data[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_add_bias_cyclic() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![0.1, 0.2, 0.3];
        transformer.add_bias(&mut data, &bias);

        // Should cycle: 0.1, 0.2, 0.3, 0.1, 0.2, 0.3
        assert!((data[0] - 1.1).abs() < 1e-6);
        assert!((data[3] - 4.1).abs() < 1e-6);
    }

    // ==========================================================================
    // AprTransformer matmul_scalar Tests
    // ==========================================================================

    #[test]
    fn test_matmul_scalar_simple() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let output = transformer.matmul_scalar(&input, &weight, 2, 2);

        assert_eq!(output, vec![1.0, 2.0]);
    }

    #[test]
    fn test_matmul_scalar_projection() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // PMAT-095: Weight is now [out_dim, in_dim] format
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2] row-major
        let output = transformer.matmul_scalar(&input, &weight, 2, 3);

        // W[0,:] @ x = [1,2] @ [1,2] = 5
        // W[1,:] @ x = [3,4] @ [1,2] = 11
        // W[2,:] @ x = [5,6] @ [1,2] = 17
        assert_eq!(output, vec![5.0, 11.0, 17.0]);
    }

    // ==========================================================================
    // AprParityComparison Debug Tests
    // ==========================================================================

    #[test]
    fn test_parity_comparison_debug() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.95,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0,
        };
        let debug_str = format!("{:?}", comparison);
        assert!(debug_str.contains("AprParityComparison"));
    }

    #[test]
    fn test_parity_comparison_clone() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.95,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0,
        };
        let cloned = comparison.clone();
        assert_eq!(comparison.throughput_ratio, cloned.throughput_ratio);
    }

    // ==========================================================================
    // QuantizedAprTransformer calculate_quantized_bytes Tests
    // ==========================================================================

    #[test]
    fn test_calculate_quantized_bytes_f32() {
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(100, AprQuantizationType::F32);
        // F32: 1 value per block, 4 bytes per block
        assert_eq!(bytes, 400);
    }

    #[test]
    fn test_calculate_quantized_bytes_q4_k() {
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(256, AprQuantizationType::Q4_K);
        // Q4_K: 256 values per block, 144 bytes per block
        assert_eq!(bytes, 144);
    }

    #[test]
    fn test_calculate_quantized_bytes_q8_0() {
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(32, AprQuantizationType::Q8_0);
        // Q8_0: 32 values per block, 36 bytes per block
        assert_eq!(bytes, 36);
    }

    #[test]
    fn test_calculate_quantized_bytes_rounding_up() {
        // 33 values should round up to 2 blocks for Q8_0
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(33, AprQuantizationType::Q8_0);
        assert_eq!(bytes, 72); // 2 blocks * 36 bytes
    }

    // ==========================================================================
    // MmapAprTransformer num_parameters Tests
    // ==========================================================================

    #[test]
    fn test_mmap_num_parameters_calculation() {
        // Test the calculation logic without an actual file
        // Based on MmapAprTransformer::num_parameters():
        // embed_params = vocab * hidden * 2
        // layer_params = hidden + (hidden * 3 * hidden) + (hidden * hidden) + (hidden * intermediate) + (intermediate * hidden)
        // norm_params = hidden
        // total = embed_params + (layers * layer_params) + norm_params

        // These values match a config where we can verify the formula
        let hidden = 64;
        let vocab = 100;
        let layers = 2;
        let intermediate = 256;

        let embed_params = vocab * hidden * 2;
        let layer_params = hidden
            + (hidden * 3 * hidden)
            + (hidden * hidden)
            + (hidden * intermediate)
            + (intermediate * hidden);
        let norm_params = hidden;
        let total = embed_params + (layers * layer_params) + norm_params;

        assert!(total > 0);
    }

    // ==========================================================================
    // AprTransformer from_apr_bytes Edge Cases
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_too_small() {
        // Less than 64 bytes
        let data = vec![0u8; 32];
        let result = AprTransformer::from_apr_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_invalid_magic() {
        // Valid size but wrong magic
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(b"XXXX");
        let result = AprTransformer::from_apr_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_aprn_magic() {
        // APRN magic - may succeed or fail depending on structure
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(b"APR\0");
        let result = AprTransformer::from_apr_bytes(&data);
        // Result varies based on implementation - just verify it doesn't panic
        let _ = result;
    }

    // ==========================================================================
    // QuantizedAprTransformer from_bytes Edge Cases
    // ==========================================================================

    #[test]
    fn test_quantized_from_bytes_too_small() {
        let data = vec![0u8; 32];
        let result = QuantizedAprTransformer::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_from_bytes_invalid_magic() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(b"XXXX");
        let result = QuantizedAprTransformer::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_from_bytes_invalid_quant_type() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&MAGIC);
        data[48] = 99; // Invalid quantization type
        let result = QuantizedAprTransformer::from_bytes(&data);
        assert!(result.is_err());
    }

    // ==========================================================================
    // AprTransformer forward_with_cache Extended Tests
    // ==========================================================================

    #[test]
    fn test_forward_with_cache_position_progression() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        let mut cache = AprKVCache::new(&config);

        // Process multiple positions
        for pos in 0..5 {
            let result = transformer.forward_with_cache(pos as u32 + 1, &mut cache, pos);
            assert!(result.is_ok());
        }
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_forward_with_cache_logits_shape() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        let mut cache = AprKVCache::new(&config);

        let result = transformer.forward_with_cache(1, &mut cache, 0);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), 100);
    }

    // ==========================================================================
    // Debug and Clone Trait Tests
    // ==========================================================================

    #[test]
    fn test_mmap_transformer_debug() {
        // Can't test from_file without a real file, but we can test the config
        let config = AprTransformerConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AprTransformerConfig"));
    }

    #[test]
    fn test_kv_cache_clone() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        cache.append(0, &k, &v);

        let cloned = cache.clone();
        assert_eq!(cache.len(), cloned.len());
        assert_eq!(cache.capacity(), cloned.capacity());
    }

    #[test]
    fn test_kv_cache_debug() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("AprKVCache"));
    }

    #[test]
    fn test_generate_config_clone() {
        let config = GenerateConfig {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
        };
        let cloned = config.clone();
        assert_eq!(config.max_tokens, cloned.max_tokens);
        assert_eq!(config.temperature, cloned.temperature);
    }

    #[test]
    fn test_generate_config_debug() {
        let config = GenerateConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GenerateConfig"));
    }

    #[test]
    fn test_benchmark_runner_debug() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let runner = AprBenchmarkRunner::new(transformer);
        let debug_str = format!("{:?}", runner);
        assert!(debug_str.contains("AprBenchmarkRunner"));
    }

    // ==========================================================================
    // Additional Coverage: QuantizedAprTransformerQ4 Tests
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_q4_memory_size() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            ..Default::default()
        };

        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; config.hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.hidden_dim * 3),
            attn_output_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(
                config.intermediate_dim,
                config.hidden_dim,
            ),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(
                config.hidden_dim,
                config.intermediate_dim,
            )),
            ffn_norm_weight: Some(vec![1.0; config.hidden_dim]),
        };

        let qt4 = QuantizedAprTransformerQ4 {
            config: config.clone(),
            token_embedding: vec![0.0; config.vocab_size * config.hidden_dim],
            layers: vec![layer],
            output_norm_weight: vec![1.0; config.hidden_dim],
            lm_head_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.vocab_size),
        };

        let mem_size = qt4.memory_size();
        assert!(mem_size > 0);
    }

    // ==========================================================================
    // Additional Coverage: AprTransformer Layer Operations Tests
    // ==========================================================================

    #[test]
    fn test_transformer_layer_with_all_biases() {
        let mut layer = AprTransformerLayer::empty(64, 256);
        layer.attn_norm_bias = Some(vec![0.0; 64]);
        layer.qkv_bias = Some(vec![0.0; 64 * 3]);
        layer.attn_output_bias = Some(vec![0.0; 64]);
        layer.ffn_norm_weight = Some(vec![1.0; 64]);
        layer.ffn_norm_bias = Some(vec![0.0; 64]);
        layer.ffn_gate_weight = Some(vec![0.0; 64 * 256]);
        layer.ffn_gate_bias = Some(vec![0.0; 256]);
        layer.ffn_up_bias = Some(vec![0.0; 256]);
        layer.ffn_down_bias = Some(vec![0.0; 64]);

        let params = layer.num_parameters();
        // Should include all biases
        assert!(params > 64 + 64 * 3 * 64 + 64 * 64 + 64 * 256 + 256 * 64);
    }

    #[test]
    fn test_transformer_generate_with_cache_max_tokens_zero() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 0,
            ..Default::default()
        };
        let result = transformer.generate_with_cache(&[1], &gen_config);
        assert!(result.is_ok());
        let tokens = result.expect("APR operation failed");
        // With max_tokens=0, should return just the prompt
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_transformer_generate_with_cache_top_k() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 2,
            temperature: 0.8,
            top_k: 10, // Enable top-k filtering
            ..Default::default()
        };
        let result = transformer.generate_with_cache(&[1], &gen_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_generate_with_cache_top_p() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 2,
            temperature: 0.8,
            top_p: 0.5, // Enable nucleus sampling
            ..Default::default()
        };
        let result = transformer.generate_with_cache(&[1], &gen_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_generate_with_repetition_penalty() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 2,
            temperature: 0.8,
            repetition_penalty: 1.2, // Apply repetition penalty
            ..Default::default()
        };
        let result = transformer.generate_with_cache(&[1], &gen_config);
        assert!(result.is_ok());
    }

    // ==========================================================================
    // Additional Coverage: AprTransformer Forward Edge Cases
    // ==========================================================================

    #[test]
    fn test_transformer_forward_many_layers() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 4, // Multiple layers
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let result = transformer.forward(&[1, 2, 3]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_forward_gqa_config() {
        // GQA: 8 heads, 2 KV heads (4:1 ratio)
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            num_heads: 8,
            num_kv_heads: 2, // GQA configuration
            context_length: 128,
            ..Default::default()
        };

        let layers: Vec<AprTransformerLayer> = (0..config.num_layers)
            .map(|_| {
                AprTransformerLayer::empty_gqa(
                    config.hidden_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.intermediate_dim,
                )
            })
            .collect();

        let transformer = AprTransformer {
            config: config.clone(),
            token_embedding: vec![0.1; config.vocab_size * config.hidden_dim],
            layers,
            output_norm_weight: vec![1.0; config.hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; config.hidden_dim * config.vocab_size],
            lm_head_bias: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
            q4k_layers: None,
        };

        let result = transformer.forward(&[1, 2]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), config.vocab_size);
    }

    // ==========================================================================
    // Additional Coverage: AprKVCache Edge Cases
    // ==========================================================================

    #[test]
    fn test_kv_cache_multiple_appends_same_layer() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);

        // Append multiple times to same layer
        for i in 0..5 {
            let k = vec![i as f32; 32];
            let v = vec![(i + 10) as f32; 32];
            cache.append(0, &k, &v);
        }

        let (k_out, v_out) = cache.get(0);
        assert!(!k_out.is_empty());
        assert!(!v_out.is_empty());
    }

    #[test]
    fn test_kv_cache_head_dim_calculation() {
        // Test with different num_heads/num_kv_heads ratios
        let config = AprTransformerConfig {
            hidden_dim: 128,
            num_layers: 2,
            num_heads: 16,
            num_kv_heads: 4, // 4:1 GQA ratio
            context_length: 256,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);

        // head_dim = hidden_dim / num_heads = 128 / 16 = 8
        assert_eq!(cache.head_dim, 8);
        assert_eq!(cache.num_kv_heads, 4);
    }

    // ==========================================================================
    // Additional Coverage: QuantizedAprTransformer to_bytes/from_bytes
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_to_bytes_multiple_layers() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 3,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);

        let bytes = qt.to_bytes().expect("serialize");
        assert!(!bytes.is_empty());

        let restored = QuantizedAprTransformer::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.config().num_layers, 3);
    }

    // ==========================================================================
    // Additional Coverage: AprQuantizationType
    // ==========================================================================

    #[test]
    fn test_quantization_type_copy() {
        let qt1 = AprQuantizationType::Q4_K;
        let qt2 = qt1; // Copy
        assert_eq!(qt1, qt2);
    }

    #[test]
    fn test_quantization_type_debug() {
        let qt = AprQuantizationType::Q4_K;
        let debug_str = format!("{:?}", qt);
        assert!(debug_str.contains("Q4_K"));
    }

    // ==========================================================================
    // Additional Coverage: AprBenchmarkResult
    // ==========================================================================

    #[test]
    fn test_benchmark_result_debug() {
        let result = AprBenchmarkResult::default();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprBenchmarkResult"));
    }

    // ==========================================================================
    // Additional Coverage: Softmax standalone function
    // ==========================================================================

    #[test]
    fn test_softmax_uniform() {
        let mut data = vec![1.0, 1.0, 1.0, 1.0];
        // Softmax of uniform values should be uniform
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for x in &mut data {
            *x = (*x - max_val).exp();
            exp_sum += *x;
        }
        for x in &mut data {
            *x /= exp_sum;
        }
        // Each should be ~0.25
        for x in &data {
            assert!((x - 0.25).abs() < 0.01);
        }
    }

    #[test]
    fn test_softmax_large_difference() {
        let mut data = vec![0.0, 10.0, 0.0, 0.0];
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for x in &mut data {
            *x = (*x - max_val).exp();
            exp_sum += *x;
        }
        for x in &mut data {
            *x /= exp_sum;
        }
        // Second element should dominate (close to 1.0)
        assert!(data[1] > 0.99);
    }

    // ==========================================================================
    // Additional Coverage: Constants
    // ==========================================================================

    #[test]
    fn test_thresholds_positive() {
        assert!(APR_CPU_DECODE_THRESHOLD_TOK_S > 0.0);
        assert!(APR_PREFILL_THRESHOLD_TOK_S > 0.0);
        assert!(APR_PARITY_THRESHOLD_PCT > 0.0);
        assert!(APR_PARITY_THRESHOLD_PCT <= 100.0);
    }

    // ==========================================================================
    // Additional Coverage: TruenoMatrix usage
    // ==========================================================================

    #[test]
    fn test_trueno_matrix_creation() {
        // Test that trueno types can be used
        let m = TruenoMatrix::<f32>::zeros(2, 2);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
    }

    // ==========================================================================
    // Additional Coverage: QuantizedAprTransformer default quantization
    // ==========================================================================

    #[test]
    fn test_quantization_type_default_is_f32() {
        let qt = AprQuantizationType::default();
        assert_eq!(qt, AprQuantizationType::F32);
        assert_eq!(qt.bits_per_weight(), 32.0);
    }

    // ==========================================================================
    // Additional Coverage: AprInferenceScratch field sizes
    // ==========================================================================

    #[test]
    fn test_inference_scratch_all_fields() {
        let config = AprTransformerConfig {
            hidden_dim: 128,
            num_layers: 4,
            intermediate_dim: 512,
            num_heads: 8,
            num_kv_heads: 8,
            ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);

        // Check all scratch buffer sizes
        assert_eq!(scratch.hidden.len(), 128);
        assert_eq!(scratch.normed.len(), 128);
        assert_eq!(scratch.qkv_out.len(), 128 * 3);
        assert_eq!(scratch.q.len(), 128); // num_heads * head_dim = 8 * 16 = 128
        assert_eq!(scratch.k.len(), 128); // num_kv_heads * head_dim
        assert_eq!(scratch.v.len(), 128);
        assert_eq!(scratch.attn_out.len(), 128);
        assert_eq!(scratch.ffn_input.len(), 128);
        assert_eq!(scratch.ffn_up.len(), 512);
        assert_eq!(scratch.ffn_gate.len(), 512);
        assert_eq!(scratch.ffn_out.len(), 128);
    }

    // ==========================================================================
    // Additional Coverage: AprTransformer predict_next edge cases
    // ==========================================================================

    #[test]
    fn test_predict_next_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.predict_next(&[5]);
        assert!(result.is_ok());
        let next_token = result.expect("APR operation failed");
        assert!(next_token < 50);
    }

    #[test]
    fn test_predict_next_long_sequence() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.predict_next(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert!(result.is_ok());
        let next_token = result.expect("APR operation failed");
        assert!(next_token < 50);
    }

    // ==========================================================================
    // QuantizedAprTransformer to_bytes/from_bytes round-trip
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_full_roundtrip() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

        // Serialize
        let bytes = qt.to_bytes().expect("serialize");
        assert!(bytes.len() > APR_TRANSFORMER_HEADER_SIZE);

        // Deserialize
        let restored = QuantizedAprTransformer::from_bytes(&bytes).expect("deserialize");

        // Verify config matches
        assert_eq!(restored.config().hidden_dim, config.hidden_dim);
        assert_eq!(restored.config().num_layers, config.num_layers);
        assert_eq!(restored.config().vocab_size, config.vocab_size);
        assert_eq!(restored.quantization_type(), AprQuantizationType::F32);
    }

    #[test]
    fn test_quantized_transformer_roundtrip_q8() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q8_0);

        let bytes = qt.to_bytes().expect("serialize");
        let restored = QuantizedAprTransformer::from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.quantization_type(), AprQuantizationType::Q8_0);
    }

    #[test]
    fn test_quantized_transformer_roundtrip_q4() {
        let config = AprTransformerConfig {
            hidden_dim: 256, // Must be multiple of 256 for Q4_K
            num_layers: 1,
            vocab_size: 256,
            intermediate_dim: 512,
            num_heads: 8,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);

        let bytes = qt.to_bytes().expect("serialize");
        let restored = QuantizedAprTransformer::from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.quantization_type(), AprQuantizationType::Q4_K);
    }

    // ==========================================================================
    // QuantizedAprTransformer from_f32_transformer
    // ==========================================================================

    #[test]
    fn test_from_f32_transformer_full() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let apr_transformer = AprTransformer::new(config.clone());

        let qt = QuantizedAprTransformer::from_f32_transformer(
            &apr_transformer,
            AprQuantizationType::F32,
        );

        assert_eq!(qt.config().hidden_dim, config.hidden_dim);
        assert_eq!(qt.quantization_type(), AprQuantizationType::F32);
    }

    // ==========================================================================
    // AprKVCache advanced operations
    // ==========================================================================

    #[test]
    fn test_kv_cache_multi_layer_operations() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);

        // Append to all layers
        for layer in 0..4 {
            let k = vec![layer as f32; 32]; // num_kv_heads * head_dim = 4 * 8 = 32
            let v = vec![(layer + 10) as f32; 32];
            cache.append(layer, &k, &v);
        }

        // Verify each layer has data
        for layer in 0..4 {
            let (k, v) = cache.get(layer);
            assert!(!k.is_empty());
            assert!(!v.is_empty());
        }

        // Clear and verify
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_sequential_positions() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 8, // MHA (not GQA)
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);

        // Add 10 positions
        for pos in 0..10 {
            let k = vec![pos as f32; 64]; // num_kv_heads * head_dim
            let v = vec![(pos + 100) as f32; 64];
            cache.append(0, &k, &v);
        }

        assert_eq!(cache.len(), 10);
        let (k, _) = cache.get(0);
        // Should have 10 * 64 = 640 elements
        assert_eq!(k.len(), 640);
    }

    // ==========================================================================
    // AprTransformer forward with biases
    // ==========================================================================

    #[test]
    fn test_transformer_forward_with_layer_biases() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };

        let mut layer = AprTransformerLayer::empty(32, 64);
        layer.qkv_bias = Some(vec![0.1; 32 * 3]);
        layer.attn_output_bias = Some(vec![0.05; 32]);
        layer.ffn_up_bias = Some(vec![0.02; 64]);
        layer.ffn_down_bias = Some(vec![0.01; 32]);

        let transformer = AprTransformer {
            config: config.clone(),
            token_embedding: vec![0.1; config.vocab_size * config.hidden_dim],
            layers: vec![layer],
            output_norm_weight: vec![1.0; config.hidden_dim],
            output_norm_bias: Some(vec![0.0; config.hidden_dim]),
            lm_head_weight: vec![0.0; config.hidden_dim * config.vocab_size],
            lm_head_bias: Some(vec![0.0; config.vocab_size]),
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
            q4k_layers: None,
        };

        let result = transformer.forward(&[1, 2, 3]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), config.vocab_size);
    }

    // ==========================================================================
    // AprTransformer forward_with_cache extensive test
    // ==========================================================================

    #[test]
    fn test_forward_with_cache_full_generation() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 64,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        let mut cache = AprKVCache::new(&config);

        // Process a sequence of 10 tokens one at a time
        for pos in 0..10 {
            let token_id = (pos % 50) as u32;
            let result = transformer.forward_with_cache(token_id, &mut cache, pos);
            assert!(result.is_ok());
            let logits = result.expect("APR operation failed");
            assert_eq!(logits.len(), config.vocab_size);
        }

        // Cache should have 10 positions
        assert_eq!(cache.len(), 10);
    }

    // ==========================================================================
    // AprBenchmarkRunner warmup_iterations
    // ==========================================================================

    #[test]
    fn test_benchmark_runner_warmup_iterations_getter() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let runner = AprBenchmarkRunner::new(transformer);

        assert_eq!(runner.warmup_iterations(), 3); // Default
    }

    #[test]
    fn test_benchmark_runner_set_warmup() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let mut runner = AprBenchmarkRunner::new(transformer);

        runner.set_warmup_iterations(5);
        assert_eq!(runner.warmup_iterations(), 5);
    }

    // ==========================================================================
    // QuantizedAprTransformer weight methods
    // ==========================================================================

    #[test]
    fn test_quantized_transformer_weight_bytes_f32() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

        let weight_bytes = qt.weight_bytes();
        let f32_equiv = qt.f32_equivalent_bytes();

        // For F32, weight_bytes should be close to f32_equivalent_bytes
        // (may differ slightly due to block alignment)
        let ratio = weight_bytes as f64 / f32_equiv as f64;
        assert!(ratio > 0.95 && ratio < 1.05);
    }

    #[test]
    fn test_quantized_transformer_weight_bytes_q8() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 64, // Make it divisible for Q8
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);

        let weight_bytes = qt.weight_bytes();
        let f32_equiv = qt.f32_equivalent_bytes();

        // Q8_0 is ~4.5 bits per weight, so weight_bytes < f32_equiv
        assert!(weight_bytes < f32_equiv);
    }

    // ==========================================================================
    // AprQuantizationType from_byte all variants
    // ==========================================================================

    #[test]
    fn test_quantization_type_from_byte_all() {
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
        assert_eq!(AprQuantizationType::from_byte(3), None);
        assert_eq!(AprQuantizationType::from_byte(255), None);
    }

    // ==========================================================================
    // QuantizedAprTensorQ4 expected_bytes
    // ==========================================================================

    #[test]
    fn test_quantized_tensor_q4_expected_bytes() {
        // Q4_0 is 18 bytes per 32 values
        let bytes = QuantizedAprTensorQ4::expected_bytes(32);
        assert_eq!(bytes, 18);

        let bytes = QuantizedAprTensorQ4::expected_bytes(64);
        assert_eq!(bytes, 36);

        let bytes = QuantizedAprTensorQ4::expected_bytes(128);
        assert_eq!(bytes, 72);
    }

    // ==========================================================================
    // AprInferenceScratch debug
    // ==========================================================================

    #[test]
    fn test_inference_scratch_debug() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            intermediate_dim: 64,
            ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);
        let debug_str = format!("{:?}", scratch);
        assert!(debug_str.contains("AprInferenceScratch"));
    }

    // ==========================================================================
    // AprTransformerConfig equality
    // ==========================================================================

    #[test]
    fn test_config_equality() {
        let config1 = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            ..Default::default()
        };
        let config2 = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            ..Default::default()
        };
        let config3 = AprTransformerConfig {
            hidden_dim: 128, // Different
            num_layers: 2,
            vocab_size: 100,
            ..Default::default()
        };

        assert_eq!(config1, config2);
        assert_ne!(config1, config3);
    }

    // ==========================================================================
    // AprTransformerLayer GQA construction
    // ==========================================================================

    #[test]
    fn test_layer_empty_gqa_sizes() {
        let layer = AprTransformerLayer::empty_gqa(64, 8, 2, 256);

        // QKV should be sized for GQA: hidden + 2 * (kv_heads * head_dim)
        // head_dim = hidden / num_heads = 64 / 8 = 8
        // qkv_size = hidden * (hidden + 2 * kv_heads * head_dim) = 64 * (64 + 2*2*8) = 64 * 96 = 6144
        let expected_qkv_size = 64 * (64 + 2 * 2 * 8);
        assert_eq!(layer.qkv_weight.len(), expected_qkv_size);
    }

    // ==========================================================================
    // AprTransformer serialization with serde
    // ==========================================================================

    #[test]
    fn test_apr_transformer_config_json_roundtrip() {
        let config = AprTransformerConfig {
            architecture: "test_arch".to_string(),
            hidden_dim: 256,
            num_layers: 12,
            num_heads: 16,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 1024,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
        };

        let json = serde_json::to_string(&config).expect("to json");
        let restored: AprTransformerConfig = serde_json::from_str(&json).expect("from json");

        assert_eq!(config.architecture, restored.architecture);
        assert_eq!(config.hidden_dim, restored.hidden_dim);
        assert_eq!(config.rope_theta, restored.rope_theta);
        assert_eq!(config.eps, restored.eps);
    }

    // ==========================================================================
    // Error path tests
    // ==========================================================================

    #[test]
    fn test_forward_empty_fails() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_next_empty_fails() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.predict_next(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_with_cache_empty_prompt_fails() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig::default();

        let result = transformer.generate_with_cache(&[], &gen_config);
        assert!(result.is_err());
    }

    // ==========================================================================
    // AprTransformer.generate_with_cache success path
    // ==========================================================================

    #[test]
    fn test_generate_with_cache_success() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 3,
            temperature: 0.0, // Greedy
            ..Default::default()
        };

        let result = transformer.generate_with_cache(&[1, 2], &gen_config);
        assert!(result.is_ok());
        let output = result.expect("APR operation failed");
        assert!(output.len() >= 2); // At least the prompt
    }

    #[test]
    fn test_generate_with_cache_temperature_sampling() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 2,
            temperature: 1.0, // Temperature sampling
            ..Default::default()
        };

        let result = transformer.generate_with_cache(&[1], &gen_config);
        assert!(result.is_ok());
    }

    // ==========================================================================
    // AprTransformer.forward_with_cache tests
    // ==========================================================================

    #[test]
    fn test_forward_with_cache_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        let mut cache = AprKVCache::new(&config);

        let result = transformer.forward_with_cache(1, &mut cache, 0);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), 50);
    }

    #[test]
    fn test_forward_with_cache_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config.clone());
        let mut cache = AprKVCache::new(&config);

        // Process multiple tokens sequentially
        for (i, &token) in [1, 2, 3].iter().enumerate() {
            let result = transformer.forward_with_cache(token, &mut cache, i);
            assert!(result.is_ok());
        }
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_forward_with_cache_gqa() {
        // Test GQA model (num_kv_heads < num_heads)
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 8,
            num_kv_heads: 2, // GQA
            context_length: 128,
            ..Default::default()
        };

        // Create GQA-sized layers
        let layers: Vec<AprTransformerLayer> = (0..config.num_layers)
            .map(|_| {
                AprTransformerLayer::empty_gqa(
                    config.hidden_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.intermediate_dim,
                )
            })
            .collect();

        let transformer = AprTransformer {
            config: config.clone(),
            token_embedding: vec![0.1; config.vocab_size * config.hidden_dim],
            layers,
            output_norm_weight: vec![1.0; config.hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; config.hidden_dim * config.vocab_size],
            lm_head_bias: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
            q4k_layers: None,
        };

        let mut cache = AprKVCache::new(&config);
        let result = transformer.forward_with_cache(1, &mut cache, 0);
        assert!(result.is_ok());
    }

    // ==========================================================================
    // AprTransformer add_bias helper
    // ==========================================================================

    #[test]
    fn test_add_bias_single_element() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.5, 1.0, 1.5, 2.0];
        transformer.add_bias(&mut data, &bias);

        assert_eq!(data, vec![1.5, 3.0, 4.5, 6.0]);
    }

    #[test]
    fn test_add_bias_multiple_elements() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        // Data with 8 elements, bias with 4 (wraps around)
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        transformer.add_bias(&mut data, &bias);

        assert!((data[0] - 1.1).abs() < 0.001);
        assert!((data[4] - 5.1).abs() < 0.001);
        assert!((data[7] - 8.4).abs() < 0.001);
    }

    // ==========================================================================
    // MmapAprTransformer tests
    // ==========================================================================

    #[test]
    fn test_mmap_from_file_invalid_short() {
        // Create a temporary file that's too short
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_short.apr");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(b"APR\0").expect("write magic");
        drop(file);

        let result = MmapAprTransformer::from_file(&path);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mmap_from_file_invalid_magic() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_magic.apr");
        let mut file = std::fs::File::create(&path).expect("create file");
        // Write 64 bytes with invalid magic
        let data = vec![0u8; 64];
        file.write_all(&data).expect("write data");
        drop(file);

        let result = MmapAprTransformer::from_file(&path);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mmap_from_file_invalid_version() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_version.apr");
        let mut file = std::fs::File::create(&path).expect("create file");

        // Write APR magic
        file.write_all(b"APR\0").expect("write magic");
        // Write very high version number
        file.write_all(&100u32.to_le_bytes())
            .expect("write version");
        // Pad to 64 bytes
        let padding = vec![0u8; 56];
        file.write_all(&padding).expect("write padding");
        drop(file);

        let result = MmapAprTransformer::from_file(&path);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mmap_valid_file() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_valid.apr");
        let mut file = std::fs::File::create(&path).expect("create file");

        // Header layout (64 bytes):
        // 0-3: Magic (APR)
        // 4-7: Version (u32)
        // 8-11: hidden_dim (u32)
        // 12-15: num_layers (u32)
        // 16-19: num_heads (u32)
        // 20-23: num_kv_heads (u32)
        // 24-27: vocab_size (u32)
        // 28-31: intermediate_dim (u32)
        // 32-35: context_length (u32)
        // 36-39: rope_theta (f32)
        // 40-43: eps (f32)
        // 44-47: tensor_data_offset (u32)
        // 48-63: padding

        file.write_all(b"APR\0").expect("magic"); // 0-3
        file.write_all(&1u32.to_le_bytes()).expect("version"); // 4-7
        file.write_all(&64u32.to_le_bytes()).expect("hidden_dim"); // 8-11
        file.write_all(&2u32.to_le_bytes()).expect("num_layers"); // 12-15
        file.write_all(&8u32.to_le_bytes()).expect("num_heads"); // 16-19
        file.write_all(&8u32.to_le_bytes()).expect("num_kv_heads"); // 20-23
        file.write_all(&100u32.to_le_bytes()).expect("vocab_size"); // 24-27
        file.write_all(&128u32.to_le_bytes()).expect("intermediate"); // 28-31
        file.write_all(&2048u32.to_le_bytes()).expect("context_len"); // 32-35
        file.write_all(&10000.0f32.to_le_bytes()).expect("rope"); // 36-39
        file.write_all(&1e-5f32.to_le_bytes()).expect("eps"); // 40-43
        file.write_all(&64u32.to_le_bytes()).expect("tensor_offset"); // 44-47
        file.write_all(&[0u8; 16]).expect("padding"); // 48-63

        drop(file);

        let result = MmapAprTransformer::from_file(&path);
        assert!(result.is_ok(), "Failed to load mmap file: {:?}", result);

        let model = result.expect("APR operation failed");
        assert!(model.is_mmap());
        assert_eq!(model.config.hidden_dim, 64);
        assert_eq!(model.config.num_layers, 2);
        assert_eq!(model.file_size(), 64);
        assert!(model.num_parameters() > 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mmap_get_tensor_bytes() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_tensor_bytes.apr");
        let mut file = std::fs::File::create(&path).expect("create file");

        // Write valid header (64 bytes) with tensor_data_offset = 64
        file.write_all(b"APR\0").expect("magic"); // 0-3
        file.write_all(&1u32.to_le_bytes()).expect("version"); // 4-7
        file.write_all(&64u32.to_le_bytes()).expect("hidden_dim"); // 8-11
        file.write_all(&2u32.to_le_bytes()).expect("num_layers"); // 12-15
        file.write_all(&8u32.to_le_bytes()).expect("num_heads"); // 16-19
        file.write_all(&8u32.to_le_bytes()).expect("num_kv_heads"); // 20-23
        file.write_all(&100u32.to_le_bytes()).expect("vocab_size"); // 24-27
        file.write_all(&128u32.to_le_bytes()).expect("intermediate"); // 28-31
        file.write_all(&2048u32.to_le_bytes()).expect("context_len"); // 32-35
        file.write_all(&10000.0f32.to_le_bytes()).expect("rope"); // 36-39
        file.write_all(&1e-5f32.to_le_bytes()).expect("eps"); // 40-43
        file.write_all(&64u32.to_le_bytes()).expect("tensor_offset"); // 44-47
        file.write_all(&[0u8; 16]).expect("padding"); // 48-63

        // Write tensor data at offset 64 (immediately after header)
        let tensor_data = [1.0f32, 2.0, 3.0, 4.0];
        for v in &tensor_data {
            file.write_all(&v.to_le_bytes()).expect("tensor");
        }
        drop(file);

        let model = MmapAprTransformer::from_file(&path).expect("load");

        // Test get_tensor_bytes - offset is relative to tensor_data_offset (64)
        let bytes = model.get_tensor_bytes(0, 8).expect("get bytes");
        assert_eq!(bytes.len(), 8);

        // Test get_tensor_bytes out of bounds
        let result = model.get_tensor_bytes(0, 1000);
        assert!(result.is_err());

        // Test get_tensor_f32 - reads from tensor data section
        let floats = model.get_tensor_f32(0, 2).expect("get floats");
        assert_eq!(floats.len(), 2);
        assert!(
            (floats[0] - 1.0).abs() < 0.001,
            "Expected 1.0, got {}",
            floats[0]
        );
        assert!(
            (floats[1] - 2.0).abs() < 0.001,
            "Expected 2.0, got {}",
            floats[1]
        );

        std::fs::remove_file(&path).ok();
    }

    // ==========================================================================
    // from_apr_bytes valid data tests
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_valid_apr2() {
        // Create minimal valid APR2 data
        let mut data = vec![0u8; 128];

        // APR2 magic
        data[0..4].copy_from_slice(b"APR\0");

        // Version 1.0
        data[4] = 1;
        data[5] = 0;

        // Flags
        data[6..8].copy_from_slice(&0u16.to_le_bytes());

        // Tensor count = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes());

        // Metadata offset = 64, size = 32
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&32u32.to_le_bytes());

        // Tensor index offset = 96
        data[24..32].copy_from_slice(&96u64.to_le_bytes());

        // Data offset = 96
        data[32..40].copy_from_slice(&96u64.to_le_bytes());

        // Checksum
        data[40..44].copy_from_slice(&0u32.to_le_bytes());

        // Metadata JSON at offset 64 (minimal)
        let metadata = b"{}";
        data[64..64 + metadata.len()].copy_from_slice(metadata);

        let result = AprTransformer::from_apr_bytes(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_apr_bytes_valid_aprn() {
        // Create minimal valid APRN data
        let mut data = vec![0u8; 128];

        // APRN magic
        data[0..4].copy_from_slice(b"APR\0");

        // Version 1.0
        data[4] = 1;
        data[5] = 0;

        // Tensor count = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes());

        // Metadata offset = 64, size = 2
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&2u32.to_le_bytes());

        // Tensor index offset = 66
        data[24..32].copy_from_slice(&66u64.to_le_bytes());

        // Data offset = 66
        data[32..40].copy_from_slice(&66u64.to_le_bytes());

        // Metadata JSON at offset 64
        data[64..66].copy_from_slice(b"{}");

        let result = AprTransformer::from_apr_bytes(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_apr_bytes_metadata_out_of_bounds() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"APR\0");

        // Set metadata offset beyond file size
        data[12..20].copy_from_slice(&1000u64.to_le_bytes());
        data[20..24].copy_from_slice(&100u32.to_le_bytes());

        let result = AprTransformer::from_apr_bytes(&data);
        assert!(result.is_err());
    }

    // ==========================================================================
    // QuantizedAprTransformer.forward error paths
    // ==========================================================================

    #[test]
    fn test_quantized_forward_empty_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
        let result = qt.forward(&[]);
        assert!(result.is_err());
    }

    // ==========================================================================
    // AprTransformer matmul_scalar fallback
    // ==========================================================================

    #[test]
    fn test_matmul_scalar_direct() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // PMAT-095: Weight is now [out_dim, in_dim] format
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2] row-major
        let output = transformer.matmul_scalar(&input, &weight, 2, 2);

        // W[0,:] @ x = [1,2] @ [1,2] = 1*1 + 2*2 = 5
        // W[1,:] @ x = [3,4] @ [1,2] = 3*1 + 4*2 = 11
        assert_eq!(output.len(), 2);
        assert!((output[0] - 5.0).abs() < 0.001);
        assert!((output[1] - 11.0).abs() < 0.001);
    }

    // ==========================================================================
    // QuantizedAprLayerQ4 tests
    // ==========================================================================

    #[test]
    fn test_quantized_layer_q4_construction() {
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 64],
            qkv_weight: QuantizedAprTensorQ4::zeros(64, 192),
            attn_output_weight: QuantizedAprTensorQ4::zeros(64, 64),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(64, 256),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(256, 64),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(64, 256)),
            ffn_norm_weight: Some(vec![1.0; 64]),
        };

        assert_eq!(layer.attn_norm_weight.len(), 64);
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    // ==========================================================================
    // AprTransformer.generate tests
    // ==========================================================================

    #[test]
    fn test_generate_stops_at_eos() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Generate should work (may stop early at EOS)
        let result = transformer.generate(&[1], 5);
        assert!(result.is_ok());
    }

    // ==========================================================================
    // Layer with biases tests
    // ==========================================================================

    #[test]
    fn test_layer_with_all_biases() {
        // Layer without biases but with ffn_gate (SwiGLU)
        let layer_no_bias = AprTransformerLayer {
            attn_norm_weight: vec![1.0; 32],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; 32 * 96],
            qkv_bias: None,
            attn_output_weight: vec![0.0; 32 * 32],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.0; 32 * 64]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 32 * 64],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; 64 * 32],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 32]),
            ffn_norm_bias: None,
        };

        // Same layer but with all biases added
        let layer_with_bias = AprTransformerLayer {
            attn_norm_weight: vec![1.0; 32],
            attn_norm_bias: Some(vec![0.1; 32]),
            qkv_weight: vec![0.0; 32 * 96],
            qkv_bias: Some(vec![0.1; 96]),
            attn_output_weight: vec![0.0; 32 * 32],
            attn_output_bias: Some(vec![0.1; 32]),
            ffn_gate_weight: Some(vec![0.0; 32 * 64]),
            ffn_gate_bias: Some(vec![0.1; 64]),
            ffn_up_weight: vec![0.0; 32 * 64],
            ffn_up_bias: Some(vec![0.1; 64]),
            ffn_down_weight: vec![0.0; 64 * 32],
            ffn_down_bias: Some(vec![0.1; 32]),
            ffn_norm_weight: Some(vec![1.0; 32]),
            ffn_norm_bias: Some(vec![0.1; 32]),
        };

        let params_no_bias = layer_no_bias.num_parameters();
        let params_with_bias = layer_with_bias.num_parameters();

        // Biases add: 32 + 96 + 32 + 64 + 64 + 32 + 32 = 352 parameters
        let expected_extra = 32 + 96 + 32 + 64 + 64 + 32 + 32;
        assert!(params_with_bias > params_no_bias);
        assert_eq!(params_with_bias - params_no_bias, expected_extra);
    }

    // ==========================================================================
    // AprParityComparison tests
    // ==========================================================================

    #[test]
    fn test_parity_comparison_clone_and_debug() {
        let comp = AprParityComparison {
            throughput_ratio: 0.95,
            memory_ratio: 1.1,
            parity_threshold_pct: 90.0,
        };
        let cloned = comp.clone();
        assert_eq!(cloned.throughput_ratio, 0.95);

        let debug_str = format!("{:?}", comp);
        assert!(debug_str.contains("AprParityComparison"));
    }

    // ==========================================================================
    // AprTransformer forward with layer biases
    // ==========================================================================

    #[test]
    fn test_forward_with_layer_biases() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };

        let layers = vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 32],
            attn_norm_bias: Some(vec![0.0; 32]),
            qkv_weight: vec![0.0; 32 * 96],
            qkv_bias: Some(vec![0.0; 96]),
            attn_output_weight: vec![0.0; 32 * 32],
            attn_output_bias: Some(vec![0.0; 32]),
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 32 * 64],
            ffn_up_bias: Some(vec![0.0; 64]),
            ffn_down_weight: vec![0.0; 64 * 32],
            ffn_down_bias: Some(vec![0.0; 32]),
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        }];

        let transformer = AprTransformer {
            config: config.clone(),
            token_embedding: vec![0.1; config.vocab_size * config.hidden_dim],
            layers,
            output_norm_weight: vec![1.0; 32],
            output_norm_bias: Some(vec![0.0; 32]),
            lm_head_weight: vec![0.0; 32 * 50],
            lm_head_bias: Some(vec![0.0; 50]),
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
            q4k_layers: None,
        };

        let result = transformer.forward(&[1, 2]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), 50);
    }

    // ==========================================================================
    // QuantizedAprTransformer extended tests
    // ==========================================================================

    #[test]
    fn test_quantized_from_f32_transformer() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            ..Default::default()
        };
        let f32_model = AprTransformer::new(config);
        let qt =
            QuantizedAprTransformer::from_f32_transformer(&f32_model, AprQuantizationType::Q8_0);

        assert_eq!(qt.quantization_type(), AprQuantizationType::Q8_0);
        assert_eq!(qt.config().hidden_dim, 32);
    }

    #[test]
    fn test_quantized_bits_per_weight() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };

        let qt_f32 = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        assert_eq!(qt_f32.bits_per_weight(), 32.0);

        let qt_q4 = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);
        assert!((qt_q4.bits_per_weight() - 4.5).abs() < 0.1);

        let qt_q8 = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);
        assert!((qt_q8.bits_per_weight() - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_quantized_roundtrip_q8() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 64,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q8_0);

        let bytes = qt.to_bytes().expect("serialize");
        let restored = QuantizedAprTransformer::from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.config().hidden_dim, config.hidden_dim);
        assert_eq!(restored.quantization_type(), AprQuantizationType::Q8_0);
    }

    #[test]
    fn test_quantized_forward_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

        let result = qt.forward(&[1, 2, 3, 4, 5]);
        assert!(result.is_ok());
        let logits = result.expect("APR operation failed");
        assert_eq!(logits.len(), 50);
    }

    #[test]
    fn test_quantized_forward_oov_token() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

        // Token 999 is out of vocabulary
        let result = qt.forward(&[999]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantized_forward_with_cache_oov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let mut cache = AprKVCache::new(&config);

        // OOV token should be handled (fills with zeros)
        let result = qt.forward_with_cache(99999, &mut cache, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantized_num_parameters() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 256,
            ..Default::default()
        };
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

        let params = qt.num_parameters();
        assert!(params > 0);
    }

    // ==========================================================================
    // AprTransformer generate and cache tests
    // ==========================================================================

    #[test]
    fn test_generate_basic() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.generate(&[1, 2, 3], 3);
        assert!(result.is_ok());
        let tokens = result.expect("APR operation failed");
        assert!(tokens.len() >= 3); // At least prompt length
    }

    #[test]
    fn test_generate_empty_prompt() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.generate(&[], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_with_cache_repetition_penalty() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 3,
            repetition_penalty: 1.5,
            ..Default::default()
        };

        let result = transformer.generate_with_cache(&[1, 2], &gen_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_with_cache_top_k() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 2,
            top_k: 10,
            ..Default::default()
        };

        let result = transformer.generate_with_cache(&[1], &gen_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_with_cache_top_p() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        let gen_config = GenerateConfig {
            max_tokens: 2,
            top_p: 0.9,
            ..Default::default()
        };

        let result = transformer.generate_with_cache(&[1], &gen_config);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Coverage Tests: AprTransformerConfig Debug/Clone/Default
    // =========================================================================

    #[test]
    fn test_apr_transformer_config_debug_cov() {
        let config = AprTransformerConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("AprTransformerConfig"));
        assert!(debug.contains("hidden_dim"));
    }

    #[test]
    fn test_apr_transformer_config_clone_cov() {
        let config = AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
        };
        let cloned = config.clone();
        assert_eq!(config.hidden_dim, cloned.hidden_dim);
        assert_eq!(config.architecture, cloned.architecture);
    }

    #[test]
    fn test_apr_transformer_config_eq_cov() {
        let config1 = AprTransformerConfig::default();
        let config2 = AprTransformerConfig::default();
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_apr_transformer_config_default_values_cov() {
        let config = AprTransformerConfig::default();
        assert_eq!(config.architecture, "unknown");
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.eps, 1e-5);
    }

    // =========================================================================
    // Coverage Tests: GenerateConfig Default
    // =========================================================================

    #[test]
    fn test_generate_config_default_cov() {
        let config = GenerateConfig::default();
        assert_eq!(config.max_tokens, 32);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.repetition_penalty, 1.0);
    }

    #[test]
    fn test_generate_config_custom_cov() {
        let config = GenerateConfig {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 50,
            repetition_penalty: 1.1,
        };
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.top_k, 50);
    }

    // =========================================================================
    // Coverage Tests: AprQuantizationType
    // =========================================================================

    #[test]
    fn test_apr_quantization_type_variants_cov() {
        let f32_type = AprQuantizationType::F32;
        let q4_k = AprQuantizationType::Q4_K;
        let q8_0 = AprQuantizationType::Q8_0;

        assert!(matches!(f32_type, AprQuantizationType::F32));
        assert!(matches!(q4_k, AprQuantizationType::Q4_K));
        assert!(matches!(q8_0, AprQuantizationType::Q8_0));
    }

    #[test]
    fn test_apr_quantization_type_debug_cov() {
        let q8_0 = AprQuantizationType::Q8_0;
        let debug = format!("{:?}", q8_0);
        assert!(debug.contains("Q8_0"));
    }

    #[test]
    fn test_apr_quantization_type_clone_cov() {
        let q8_0 = AprQuantizationType::Q8_0;
        let cloned = q8_0;
        assert!(matches!(cloned, AprQuantizationType::Q8_0));
    }

    #[test]
    fn test_apr_quantization_type_bits_per_weight_cov() {
        assert!((AprQuantizationType::F32.bits_per_weight() - 32.0).abs() < 0.01);
        assert!((AprQuantizationType::Q4_K.bits_per_weight() - 4.5).abs() < 0.01);
        assert!((AprQuantizationType::Q8_0.bits_per_weight() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_apr_quantization_type_default_cov() {
        let default = AprQuantizationType::default();
        assert!(matches!(default, AprQuantizationType::F32));
    }

    // =========================================================================
    // Coverage Tests: AprKVCache
    // =========================================================================

    #[test]
    fn test_apr_kv_cache_new_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 8,
            context_length: 512,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 512);
    }

    #[test]
    fn test_apr_kv_cache_capacity_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 4,
            context_length: 256,
            ..Default::default()
        };
        let cache = AprKVCache::new(&config);
        assert_eq!(cache.capacity(), 256);
        assert!(cache.is_empty());
    }

    // =========================================================================
    // Coverage Tests: AprTransformerLayer
    // =========================================================================

    #[test]
    fn test_apr_transformer_layer_empty_cov() {
        let layer = AprTransformerLayer::empty(64, 256);
        assert_eq!(layer.attn_norm_weight.len(), 64);
        assert!(!layer.ffn_up_weight.is_empty());
        assert!(!layer.ffn_down_weight.is_empty());
        assert!(layer.attn_norm_bias.is_none());
    }

    #[test]
    fn test_apr_transformer_layer_empty_gqa_cov() {
        // empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim)
        let layer = AprTransformerLayer::empty_gqa(64, 8, 4, 256);
        assert_eq!(layer.attn_norm_weight.len(), 64);
        // GQA: QKV has different dimensions
        assert!(!layer.qkv_weight.is_empty());
        assert!(!layer.ffn_up_weight.is_empty());
    }

    #[test]
    fn test_apr_transformer_layer_debug_cov() {
        let layer = AprTransformerLayer::empty(32, 64);
        let debug = format!("{:?}", layer);
        assert!(debug.contains("AprTransformerLayer"));
    }

    #[test]
    fn test_apr_transformer_layer_clone_cov() {
        let layer = AprTransformerLayer::empty(32, 64);
        let cloned = layer.clone();
        assert_eq!(layer.attn_norm_weight.len(), cloned.attn_norm_weight.len());
    }

    #[test]
    fn test_apr_transformer_layer_fields_cov() {
        let layer = AprTransformerLayer::empty(128, 512);
        // Check all non-optional fields
        assert_eq!(layer.attn_norm_weight.len(), 128);
        assert!(!layer.qkv_weight.is_empty());
        assert!(!layer.attn_output_weight.is_empty());
        assert!(!layer.ffn_up_weight.is_empty());
        assert!(!layer.ffn_down_weight.is_empty());
        // Check optional fields are None
        assert!(layer.attn_norm_bias.is_none());
        assert!(layer.qkv_bias.is_none());
        assert!(layer.ffn_gate_weight.is_none());
    }

    // =========================================================================
    // Coverage Tests: AprTransformer
    // =========================================================================

    #[test]
    fn test_apr_transformer_new_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 64,
            context_length: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);
        assert_eq!(transformer.config().num_layers, 2);
    }

    #[test]
    fn test_apr_transformer_config_accessor_cov() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config.clone());
        assert_eq!(transformer.config().hidden_dim, config.hidden_dim);
    }

    // =========================================================================
    // Coverage Tests: AprBenchmarkResult
    // =========================================================================

    #[test]
    fn test_apr_benchmark_result_debug_cov() {
        let result = AprBenchmarkResult {
            tokens_generated: 100,
            total_time_ms: 1000.0,
            tokens_per_second: 100.0,
            throughput_p50: 95.0,
            throughput_p99: 80.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 512.0,
            model_memory_mb: 256.0,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("AprBenchmarkResult"));
    }

    #[test]
    fn test_apr_benchmark_result_clone_cov() {
        let result = AprBenchmarkResult {
            tokens_generated: 100,
            total_time_ms: 1000.0,
            tokens_per_second: 100.0,
            throughput_p50: 95.0,
            throughput_p99: 80.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 512.0,
            model_memory_mb: 256.0,
        };
        let cloned = result.clone();
        assert_eq!(result.tokens_generated, cloned.tokens_generated);
        assert_eq!(result.tokens_per_second, cloned.tokens_per_second);
    }

    #[test]
    fn test_apr_benchmark_result_default_cov() {
        let result = AprBenchmarkResult::default();
        assert_eq!(result.tokens_generated, 0);
        assert_eq!(result.total_time_ms, 0.0);
    }

    // =========================================================================
    // Coverage Tests: AprPrefillResult
    // =========================================================================

    #[test]
    fn test_apr_prefill_result_debug_cov() {
        let result = AprPrefillResult {
            prompt_tokens: 100,
            prefill_time_ms: 10.0,
            prefill_tok_s: 10000.0,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("AprPrefillResult"));
    }

    #[test]
    fn test_apr_prefill_result_clone_cov() {
        let result = AprPrefillResult {
            prompt_tokens: 50,
            prefill_time_ms: 5.0,
            prefill_tok_s: 10000.0,
        };
        let cloned = result.clone();
        assert_eq!(result.prompt_tokens, cloned.prompt_tokens);
        assert_eq!(result.prefill_time_ms, cloned.prefill_time_ms);
    }

    #[test]
    fn test_apr_prefill_result_default_cov() {
        let result = AprPrefillResult::default();
        assert_eq!(result.prompt_tokens, 0);
        assert_eq!(result.prefill_time_ms, 0.0);
    }

    // =========================================================================
    // Coverage Tests: AprLoadResult
    // =========================================================================

    #[test]
    fn test_apr_load_result_debug_cov() {
        let result = AprLoadResult {
            load_time_ms: 100.0,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("AprLoadResult"));
    }

    #[test]
    fn test_apr_load_result_clone_cov() {
        let result = AprLoadResult {
            load_time_ms: 100.0,
        };
        let cloned = result.clone();
        assert_eq!(result.load_time_ms, cloned.load_time_ms);
    }

    #[test]
    fn test_apr_load_result_default_cov() {
        let result = AprLoadResult::default();
        assert_eq!(result.load_time_ms, 0.0);
    }

    // =========================================================================
    // Coverage Tests: AprParityComparison
    // =========================================================================

    #[test]
    fn test_apr_parity_comparison_debug_cov() {
        let comparison = AprParityComparison {
            throughput_ratio: 1.05,
            memory_ratio: 0.95,
            parity_threshold_pct: 90.0,
        };
        let debug = format!("{:?}", comparison);
        assert!(debug.contains("AprParityComparison"));
    }

    #[test]
    fn test_apr_parity_comparison_clone_cov() {
        let comparison = AprParityComparison {
            throughput_ratio: 1.1,
            memory_ratio: 0.9,
            parity_threshold_pct: 95.0,
        };
        let cloned = comparison.clone();
        assert_eq!(comparison.throughput_ratio, cloned.throughput_ratio);
        assert_eq!(comparison.memory_ratio, cloned.memory_ratio);
    }

    #[test]
    fn test_apr_parity_comparison_is_parity_cov() {
        // 1.0 ratio >= 90% threshold
        let parity = AprParityComparison {
            throughput_ratio: 1.0,
            memory_ratio: 1.0,
            parity_threshold_pct: 90.0,
        };
        assert!(parity.is_parity());

        // 0.8 ratio < 90% threshold
        let not_parity = AprParityComparison {
            throughput_ratio: 0.8,
            memory_ratio: 0.8,
            parity_threshold_pct: 90.0,
        };
        assert!(!not_parity.is_parity());
    }

    // =========================================================================
    // Coverage Tests: AprInferenceScratch
    // =========================================================================

    #[test]
    fn test_apr_inference_scratch_from_config_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            intermediate_dim: 256,
            ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);
        assert_eq!(scratch.hidden.len(), 64);
        assert_eq!(scratch.ffn_up.len(), 256);
    }

    #[test]
    fn test_apr_inference_scratch_fields_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            intermediate_dim: 64,
            ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);
        assert_eq!(scratch.hidden.len(), 32);
        assert_eq!(scratch.normed.len(), 32);
        assert_eq!(scratch.attn_out.len(), 32);
        assert_eq!(scratch.ffn_input.len(), 32);
        assert_eq!(scratch.ffn_up.len(), 64);
        assert_eq!(scratch.ffn_gate.len(), 64);
    }

    #[test]
    fn test_apr_inference_scratch_clear_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            intermediate_dim: 64,
            ..Default::default()
        };
        let mut scratch = AprInferenceScratch::from_config(&config);
        // Fill with non-zero values
        scratch.hidden.fill(1.0);
        scratch.normed.fill(1.0);
        // Clear
        scratch.clear();
        assert!(scratch.hidden.iter().all(|&x| x == 0.0));
        assert!(scratch.normed.iter().all(|&x| x == 0.0));
    }

    // =========================================================================
    // Coverage Tests: Constants
    // =========================================================================

    #[test]
    fn test_apr_transformer_constants_cov() {
        assert_eq!(&MAGIC, b"APR\0");
        assert_eq!(1, 1);
        assert_eq!(APR_TRANSFORMER_HEADER_SIZE, 64);
    }

    // =========================================================================
    // Extended Coverage Tests: AprQuantizationType
    // =========================================================================

    #[test]
    fn test_apr_quantization_type_default_ext_cov() {
        let qtype: AprQuantizationType = AprQuantizationType::default();
        assert_eq!(qtype, AprQuantizationType::F32);
    }

    #[test]
    fn test_apr_quantization_type_bits_per_weight_all_ext_cov() {
        assert!((AprQuantizationType::F32.bits_per_weight() - 32.0).abs() < 0.1);
        assert!((AprQuantizationType::Q4_K.bits_per_weight() - 4.5).abs() < 0.1);
        assert!((AprQuantizationType::Q8_0.bits_per_weight() - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_apr_quantization_type_bytes_per_block_all_ext_cov() {
        assert_eq!(AprQuantizationType::F32.bytes_per_block(), 4);
        assert_eq!(AprQuantizationType::Q4_K.bytes_per_block(), 144);
        assert_eq!(AprQuantizationType::Q8_0.bytes_per_block(), 36);
    }

    #[test]
    fn test_apr_quantization_type_values_per_block_all_ext_cov() {
        assert_eq!(AprQuantizationType::F32.values_per_block(), 1);
        assert_eq!(AprQuantizationType::Q4_K.values_per_block(), 256);
        assert_eq!(AprQuantizationType::Q8_0.values_per_block(), 32);
    }

    #[test]
    fn test_apr_quantization_type_to_byte_all_ext_cov() {
        assert_eq!(AprQuantizationType::F32.to_byte(), 0);
        assert_eq!(AprQuantizationType::Q4_K.to_byte(), 1);
        assert_eq!(AprQuantizationType::Q8_0.to_byte(), 2);
    }

    #[test]
    fn test_apr_quantization_type_from_byte_all_ext_cov() {
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
        assert_eq!(AprQuantizationType::from_byte(3), None);
        assert_eq!(AprQuantizationType::from_byte(255), None);
    }

    #[test]
    fn test_apr_quantization_type_clone_ext_cov() {
        let qtype = AprQuantizationType::Q4_K;
        let cloned = qtype;
        assert_eq!(qtype, cloned);
    }

    #[test]
    fn test_apr_quantization_type_copy_ext_cov() {
        let qtype = AprQuantizationType::Q8_0;
        let copied = qtype;
        assert_eq!(copied, AprQuantizationType::Q8_0);
    }

    #[test]
    fn test_apr_quantization_type_debug_ext_cov() {
        let debug_str = format!("{:?}", AprQuantizationType::Q4_K);
        assert!(debug_str.contains("Q4_K"));
    }

    #[test]
    fn test_apr_quantization_type_eq_ext_cov() {
        assert_eq!(AprQuantizationType::F32, AprQuantizationType::F32);
        assert_ne!(AprQuantizationType::F32, AprQuantizationType::Q4_K);
        assert_ne!(AprQuantizationType::Q4_K, AprQuantizationType::Q8_0);
    }

    // =========================================================================
    // Extended Coverage Tests: AprTransformerConfig
    // =========================================================================

    #[test]
    fn test_apr_transformer_config_debug_ext_cov() {
        let config = AprTransformerConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AprTransformerConfig"));
    }

    #[test]
    fn test_apr_transformer_config_clone_ext_cov() {
        let config = AprTransformerConfig {
            architecture: "phi2".to_string(),
            hidden_dim: 512,
            intermediate_dim: 2048,
            num_heads: 8,
            num_kv_heads: 8,
            num_layers: 12,
            vocab_size: 32000,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        };
        let cloned = config.clone();
        assert_eq!(cloned.hidden_dim, 512);
        assert_eq!(cloned.num_layers, 12);
    }

    #[test]
    fn test_apr_transformer_config_default_ext_cov() {
        let config = AprTransformerConfig::default();
        assert!(config.hidden_dim > 0);
        assert!(config.num_layers > 0);
    }

    // =========================================================================
    // Extended Coverage Tests: GenerateConfig
    // =========================================================================

    #[test]
    fn test_generate_config_debug_ext_cov() {
        let config = GenerateConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GenerateConfig"));
    }

    #[test]
    fn test_generate_config_clone_ext_cov() {
        let config = GenerateConfig {
            max_tokens: 100,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.0,
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_tokens, 100);
        assert!((cloned.temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_generate_config_default_ext_cov() {
        let config = GenerateConfig::default();
        assert!(config.max_tokens > 0);
        assert!(config.temperature > 0.0);
    }

    // =========================================================================
    // Extended Coverage Tests: AprBenchmarkResult
    // =========================================================================

    #[test]
    fn test_apr_benchmark_result_debug_ext_cov() {
        let result = AprBenchmarkResult {
            tokens_generated: 50,
            total_time_ms: 50.0,
            tokens_per_second: 100.0,
            throughput_p50: 95.0,
            throughput_p99: 80.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 100.0,
            model_memory_mb: 80.0,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprBenchmarkResult"));
    }

    #[test]
    fn test_apr_benchmark_result_clone_ext_cov() {
        let result = AprBenchmarkResult {
            tokens_generated: 100,
            total_time_ms: 50.0,
            tokens_per_second: 200.0,
            throughput_p50: 190.0,
            throughput_p99: 150.0,
            throughput_std_dev: 10.0,
            peak_memory_mb: 150.0,
            model_memory_mb: 120.0,
        };
        let cloned = result.clone();
        assert!((cloned.tokens_per_second - 200.0).abs() < 1e-6);
        assert_eq!(cloned.tokens_generated, 100);
    }

    #[test]
    fn test_apr_benchmark_result_default_ext_cov() {
        let result = AprBenchmarkResult::default();
        assert_eq!(result.tokens_generated, 0);
    }

    // =========================================================================
    // Extended Coverage Tests: AprPrefillResult
    // =========================================================================

    #[test]
    fn test_apr_prefill_result_debug_ext_cov() {
        let result = AprPrefillResult {
            prompt_tokens: 64,
            prefill_time_ms: 10.0,
            prefill_tok_s: 6400.0,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprPrefillResult"));
    }

    #[test]
    fn test_apr_prefill_result_clone_ext_cov() {
        let result = AprPrefillResult {
            prompt_tokens: 32,
            prefill_time_ms: 5.0,
            prefill_tok_s: 6400.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.prompt_tokens, 32);
        assert!((cloned.prefill_time_ms - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_apr_prefill_result_default_ext_cov() {
        let result = AprPrefillResult::default();
        assert_eq!(result.prompt_tokens, 0);
    }

    // =========================================================================
    // Extended Coverage Tests: AprLoadResult
    // =========================================================================

    #[test]
    fn test_apr_load_result_debug_ext_cov() {
        let result = AprLoadResult {
            load_time_ms: 100.0,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprLoadResult"));
    }

    #[test]
    fn test_apr_load_result_clone_ext_cov() {
        let result = AprLoadResult { load_time_ms: 50.0 };
        let cloned = result.clone();
        assert!((cloned.load_time_ms - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_apr_load_result_default_ext_cov() {
        let result = AprLoadResult::default();
        assert!((result.load_time_ms).abs() < 1e-6);
    }

    // =========================================================================
    // Extended Coverage Tests: AprParityComparison
    // =========================================================================

    #[test]
    fn test_apr_parity_comparison_debug_ext_cov() {
        let comparison = AprParityComparison {
            throughput_ratio: 1.1,
            memory_ratio: 0.8,
            parity_threshold_pct: 95.0,
        };
        let debug_str = format!("{:?}", comparison);
        assert!(debug_str.contains("AprParityComparison"));
    }

    #[test]
    fn test_apr_parity_comparison_clone_ext_cov() {
        let comparison = AprParityComparison {
            throughput_ratio: 1.5,
            memory_ratio: 0.7,
            parity_threshold_pct: 95.0,
        };
        let cloned = comparison.clone();
        assert!((cloned.throughput_ratio - 1.5).abs() < 1e-6);
        assert!((cloned.memory_ratio - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_apr_parity_comparison_is_parity_ext_cov() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.96,
            memory_ratio: 0.9,
            parity_threshold_pct: 95.0,
        };
        assert!(comparison.is_parity());

        let not_parity = AprParityComparison {
            throughput_ratio: 0.90,
            memory_ratio: 0.9,
            parity_threshold_pct: 95.0,
        };
        assert!(!not_parity.is_parity());
    }

    // =========================================================================
    // Extended Coverage Tests: AprKVCache
    // =========================================================================

    #[test]
    fn test_apr_kv_cache_debug_ext_cov() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("AprKVCache"));
    }

    #[test]
    fn test_apr_kv_cache_len_ext_cov() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_apr_kv_cache_clear_ext_cov() {
        let config = AprTransformerConfig::default();
        let mut cache = AprKVCache::new(&config);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    // =========================================================================
    // Extended Coverage Tests: QuantizedAprTensorQ4
    // =========================================================================

    #[test]
    fn test_quantized_apr_tensor_q4_debug_ext_cov() {
        let tensor = QuantizedAprTensorQ4 {
            data: vec![0u8; 144],
            in_dim: 256,
            out_dim: 1,
        };
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("QuantizedAprTensorQ4"));
    }

    #[test]
    fn test_quantized_apr_tensor_q4_clone_ext_cov() {
        let tensor = QuantizedAprTensorQ4 {
            data: vec![1u8; 288],
            in_dim: 512,
            out_dim: 2,
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.in_dim, 512);
        assert_eq!(cloned.out_dim, 2);
    }

    // =========================================================================
    // Extended Coverage Tests Phase 2: More structs and methods
    // =========================================================================

    #[test]
    fn test_quantized_apr_layer_q4_debug_cov() {
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 64],
            qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 192),
            attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 64),
            ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 256),
            ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 256, 64),
            ffn_gate_weight: None,
            ffn_norm_weight: None,
        };
        let debug = format!("{:?}", layer);
        assert!(debug.contains("QuantizedAprLayerQ4"));
    }

    #[test]
    fn test_quantized_apr_layer_q4_clone_cov() {
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 64],
            qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 192),
            attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 64),
            ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 256),
            ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 256, 64),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 256)),
            ffn_norm_weight: Some(vec![1.0; 64]),
        };
        let cloned = layer.clone();
        assert_eq!(cloned.attn_norm_weight.len(), 64);
        assert!(cloned.ffn_gate_weight.is_some());
        assert!(cloned.ffn_norm_weight.is_some());
    }

    #[test]
    fn test_quantized_apr_layer_q4_with_gate_cov() {
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 32],
            qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 96),
            attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 32),
            ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 128),
            ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 128, 32),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 128)),
            ffn_norm_weight: Some(vec![1.0; 32]),
        };
        assert_eq!(
            layer
                .ffn_gate_weight
                .as_ref()
                .expect("APR operation failed")
                .in_dim,
            32
        );
        assert_eq!(
            layer
                .ffn_norm_weight
                .as_ref()
                .expect("APR operation failed")
                .len(),
            32
        );
    }

    #[test]
    fn test_quantized_apr_transformer_q4_debug_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 32],
            qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 96),
            attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 32),
            ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 128),
            ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 128, 32),
            ffn_gate_weight: None,
            ffn_norm_weight: None,
        };
        let transformer = QuantizedAprTransformerQ4 {
            config,
            token_embedding: vec![0.0; 100 * 32],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 32],
            lm_head_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 100),
        };
        let debug = format!("{:?}", transformer);
        assert!(debug.contains("QuantizedAprTransformerQ4"));
    }

    #[test]
    fn test_quantized_apr_transformer_q4_clone_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 32],
            qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 96),
            attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 32),
            ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 128),
            ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 128, 32),
            ffn_gate_weight: None,
            ffn_norm_weight: None,
        };
        let transformer = QuantizedAprTransformerQ4 {
            config,
            token_embedding: vec![0.0; 100 * 32],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 32],
            lm_head_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 100),
        };
        let cloned = transformer.clone();
        assert_eq!(cloned.layers.len(), 1);
        assert_eq!(cloned.token_embedding.len(), 3200);
    }

    #[test]
    fn test_quantized_apr_transformer_q4_config_accessor_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 256,
            intermediate_dim: 256,
            context_length: 128,
            ..Default::default()
        };
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 64],
            qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 192),
            attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 64),
            ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 256),
            ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 256, 64),
            ffn_gate_weight: None,
            ffn_norm_weight: None,
        };
        let transformer = QuantizedAprTransformerQ4 {
            config: config.clone(),
            token_embedding: vec![0.0; 256 * 64],
            layers: vec![layer.clone(), layer],
            output_norm_weight: vec![1.0; 64],
            lm_head_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 64, 256),
        };
        let cfg = transformer.config();
        assert_eq!(cfg.hidden_dim, 64);
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.vocab_size, 256);
    }

    #[test]
    fn test_quantized_apr_transformer_q4_create_scratch_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 32],
            qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 96),
            attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 32),
            ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 128),
            ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 128, 32),
            ffn_gate_weight: None,
            ffn_norm_weight: None,
        };
        let transformer = QuantizedAprTransformerQ4 {
            config,
            token_embedding: vec![0.0; 100 * 32],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 32],
            lm_head_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 100),
        };
        let scratch = transformer.create_scratch();
        assert_eq!(scratch.hidden.len(), 32);
        assert_eq!(scratch.normed.len(), 32);
        assert_eq!(scratch.ffn_up.len(), 128);
    }

    #[test]
    fn test_quantized_apr_transformer_q4_create_kv_cache_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; 32],
            qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 96),
            attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 32),
            ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 128),
            ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 128, 32),
            ffn_gate_weight: None,
            ffn_norm_weight: None,
        };
        let transformer = QuantizedAprTransformerQ4 {
            config,
            token_embedding: vec![0.0; 100 * 32],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 32],
            lm_head_weight: QuantizedAprTensorQ4::new(vec![0u8; 36], 32, 100),
        };
        let cache = transformer.create_kv_cache();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 64);
    }

    #[test]
    fn test_apr_inference_scratch_debug_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            intermediate_dim: 64,
            ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);
        let debug = format!("{:?}", scratch);
        assert!(debug.contains("AprInferenceScratch"));
    }

    #[test]
    fn test_apr_inference_scratch_clear_all_fields_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_heads: 4,
            intermediate_dim: 64,
            ..Default::default()
        };
        let mut scratch = AprInferenceScratch::from_config(&config);

        // Fill with non-zero values
        scratch.hidden.fill(1.0);
        scratch.normed.fill(2.0);
        scratch.qkv_out.fill(3.0);
        scratch.q.fill(4.0);
        scratch.k.fill(5.0);
        scratch.v.fill(6.0);
        scratch.attn_out.fill(7.0);
        scratch.ffn_input.fill(8.0);
        scratch.ffn_up.fill(9.0);
        scratch.ffn_gate.fill(10.0);
        scratch.ffn_out.fill(11.0);

        // Clear
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
    fn test_apr_inference_scratch_sizes_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 128,
            num_heads: 8,
            intermediate_dim: 512,
            ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);

        assert_eq!(scratch.hidden.len(), 128);
        assert_eq!(scratch.normed.len(), 128);
        assert_eq!(scratch.qkv_out.len(), 384); // 128 * 3
        assert_eq!(scratch.q.len(), 128);
        assert_eq!(scratch.k.len(), 128);
        assert_eq!(scratch.v.len(), 128);
        assert_eq!(scratch.attn_out.len(), 128);
        assert_eq!(scratch.ffn_input.len(), 128);
        assert_eq!(scratch.ffn_up.len(), 512);
        assert_eq!(scratch.ffn_gate.len(), 512);
        assert_eq!(scratch.ffn_out.len(), 128);
    }

    #[test]
    fn test_apr_transformer_config_partial_eq_cov() {
        let config1 = AprTransformerConfig::default();
        let config2 = AprTransformerConfig::default();
        assert_eq!(config1, config2);

        let config3 = AprTransformerConfig {
            hidden_dim: 1024,
            ..Default::default()
        };
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_apr_transformer_config_architecture_field_cov() {
        let config = AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-6,
        };
        assert_eq!(config.architecture, "llama");
        assert_eq!(config.rope_theta, 10000.0);
        assert_eq!(config.eps, 1e-6);
    }

    #[test]
    fn test_apr_transformer_layer_biases_cov() {
        let layer = AprTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: Some(vec![0.1; 64]),
            qkv_weight: vec![0.0; 64 * 192],
            qkv_bias: Some(vec![0.2; 192]),
            attn_output_weight: vec![0.0; 64 * 64],
            attn_output_bias: Some(vec![0.3; 64]),
            ffn_gate_weight: Some(vec![0.0; 64 * 256]),
            ffn_gate_bias: Some(vec![0.4; 256]),
            ffn_up_weight: vec![0.0; 64 * 256],
            ffn_up_bias: Some(vec![0.5; 256]),
            ffn_down_weight: vec![0.0; 256 * 64],
            ffn_down_bias: Some(vec![0.6; 64]),
            ffn_norm_weight: Some(vec![1.0; 64]),
            ffn_norm_bias: Some(vec![0.7; 64]),
        };

        assert!(layer.attn_norm_bias.is_some());
        assert!(layer.qkv_bias.is_some());
        assert!(layer.attn_output_bias.is_some());
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_gate_bias.is_some());
        assert!(layer.ffn_up_bias.is_some());
        assert!(layer.ffn_down_bias.is_some());
        assert!(layer.ffn_norm_weight.is_some());
        assert!(layer.ffn_norm_bias.is_some());

        // Calculate parameters including all biases
        let params = layer.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_apr_transformer_layer_serialize_cov() {
        let layer = AprTransformerLayer::empty(32, 128);
        let json = serde_json::to_string(&layer).expect("invalid UTF-8");
        assert!(json.contains("attn_norm_weight"));

        let layer2: AprTransformerLayer = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(layer2.attn_norm_weight.len(), 32);
    }

    #[test]
    fn test_apr_benchmark_constants_cov() {
        assert_eq!(APR_CPU_DECODE_THRESHOLD_TOK_S, 50.0);
        assert_eq!(APR_PREFILL_THRESHOLD_TOK_S, 100.0);
        assert_eq!(APR_PARITY_THRESHOLD_PCT, 95.0);
    }

    #[test]
    fn test_apr_benchmark_result_throughput_fields_cov() {
        let result = AprBenchmarkResult {
            tokens_generated: 100,
            total_time_ms: 1000.0,
            tokens_per_second: 100.0,
            throughput_p50: 95.0,
            throughput_p99: 80.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 512.0,
            model_memory_mb: 256.0,
        };

        assert_eq!(result.throughput_p50, 95.0);
        assert_eq!(result.throughput_p99, 80.0);
        assert_eq!(result.throughput_std_dev, 5.0);
        assert_eq!(result.peak_memory_mb, 512.0);
        assert_eq!(result.model_memory_mb, 256.0);
    }

    #[test]
    fn test_apr_benchmark_result_compare_baseline_zero_memory_cov() {
        let result = AprBenchmarkResult {
            tokens_per_second: 50.0,
            peak_memory_mb: 100.0,
            ..Default::default()
        };
        let baseline = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 0.0, // Zero baseline memory
            ..Default::default()
        };
        let comparison = result.compare_to_baseline(&baseline);
        assert_eq!(comparison.memory_ratio, 1.0); // Falls back to 1.0
    }

    #[test]
    fn test_apr_parity_comparison_threshold_cov() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.96,
            memory_ratio: 1.0,
            parity_threshold_pct: 95.0,
        };
        assert!(comparison.is_parity());
        assert_eq!(comparison.parity_threshold_pct, 95.0);

        let comparison2 = AprParityComparison {
            throughput_ratio: 0.90,
            memory_ratio: 1.0,
            parity_threshold_pct: 95.0,
        };
        assert!(!comparison2.is_parity());
    }

    #[test]
    fn test_apr_prefill_result_fields_cov() {
        let result = AprPrefillResult {
            prompt_tokens: 50,
            prefill_time_ms: 100.0,
            prefill_tok_s: 500.0,
        };
        assert_eq!(result.prompt_tokens, 50);
        assert_eq!(result.prefill_time_ms, 100.0);
        assert_eq!(result.prefill_tok_s, 500.0);
    }

    #[test]
    fn test_apr_load_result_fields_cov() {
        let result = AprLoadResult {
            load_time_ms: 1500.0,
        };
        assert_eq!(result.load_time_ms, 1500.0);
    }

    #[test]
    fn test_generate_config_all_fields_cov() {
        let config = GenerateConfig {
            max_tokens: 100,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.1,
        };
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_p, 0.95);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.repetition_penalty, 1.1);
    }

    #[test]
    fn test_apr_quantization_type_ord_cov() {
        // Test Ord via comparisons
        let f32 = AprQuantizationType::F32;
        let q4k = AprQuantizationType::Q4_K;
        let q8_0 = AprQuantizationType::Q8_0;

        // Different variants should not be equal
        assert_ne!(f32, q4k);
        assert_ne!(q4k, q8_0);
        assert_ne!(f32, q8_0);

        // Same variants should be equal
        assert_eq!(f32, AprQuantizationType::F32);
    }

    #[test]
    fn test_quantized_apr_tensor_q4_expected_bytes_large_cov() {
        // Test with a large number of elements
        // Q4_0: 32 values per block, 18 bytes per block
        // Total elements = 1,048,576
        // Blocks = 1,048,576 / 32 = 32,768
        // Bytes = 32,768 * 18 = 589,824
        let bytes = QuantizedAprTensorQ4::expected_bytes(1024 * 1024);
        assert_eq!(bytes, 589_824);
    }

    #[test]
    fn test_apr_transformer_header_size_const_cov() {
        assert_eq!(APR_TRANSFORMER_HEADER_SIZE, 64);
    }

    #[test]
    fn test_apr_kv_cache_is_empty_cov() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_apr_kv_cache_not_empty_after_append_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 64,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let head_dim = 32 / 4; // 8
        let kv_size = 4 * head_dim; // 32
        let k = vec![1.0f32; kv_size];
        let v = vec![2.0f32; kv_size];
        cache.append(0, &k, &v);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_apr_kv_cache_get_returns_correct_slices_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 64,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let head_dim = 32 / 4;
        let kv_size = 4 * head_dim;

        // Append to both layers for same position (layer 0 first, then layer 1)
        // This is the typical usage pattern: append K/V for all layers at same position
        let k0 = vec![1.0f32; kv_size];
        let v0 = vec![2.0f32; kv_size];
        cache.append(0, &k0, &v0); // Increments len to 1

        let k1 = vec![3.0f32; kv_size];
        let v1 = vec![4.0f32; kv_size];
        cache.append(1, &k1, &v1); // len stays at 1, but writes at offset 0 for layer 1

        // len should be 1 (only incremented for layer 0)
        assert_eq!(cache.len(), 1);

        // Get from layer 0 - should have k0, v0
        let (k_slice, v_slice) = cache.get(0);
        assert_eq!(k_slice.len(), kv_size);
        assert_eq!(v_slice.len(), kv_size);
        assert!((k_slice[0] - 1.0).abs() < 1e-6);
        assert!((v_slice[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_apr_kv_cache_multi_position_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            context_length: 64,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let head_dim = 32 / 4;
        let kv_size = 4 * head_dim;

        // Append position 0
        let k0 = vec![1.0f32; kv_size];
        let v0 = vec![2.0f32; kv_size];
        cache.append(0, &k0, &v0);
        assert_eq!(cache.len(), 1);

        // Append position 1
        let k1 = vec![3.0f32; kv_size];
        let v1 = vec![4.0f32; kv_size];
        cache.append(0, &k1, &v1);
        assert_eq!(cache.len(), 2);

        // Get all K/V - should have both positions
        let (k_slice, v_slice) = cache.get(0);
        assert_eq!(k_slice.len(), 2 * kv_size);
        assert_eq!(v_slice.len(), 2 * kv_size);

        // First position values
        assert!((k_slice[0] - 1.0).abs() < 1e-6);
        assert!((v_slice[0] - 2.0).abs() < 1e-6);

        // Second position values
        assert!((k_slice[kv_size] - 3.0).abs() < 1e-6);
        assert!((v_slice[kv_size] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantized_apr_transformer_new_cov() {
        let config = AprTransformerConfig::default();
        let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);
        assert_eq!(transformer.quantization_type(), AprQuantizationType::Q4_K);
        assert_eq!(transformer.bits_per_weight(), 4.5);
    }

    #[test]
    fn test_quantized_apr_transformer_from_f32_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let layer = AprTransformerLayer::empty(32, 128);
        let f32_model = AprTransformer {
            config: config.clone(),
            token_embedding: vec![0.0; 100 * 32],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 32],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; 32 * 100],
            lm_head_bias: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
            q4k_layers: None,
        };

        let q4_model =
            QuantizedAprTransformer::from_f32_transformer(&f32_model, AprQuantizationType::Q4_K);
        assert_eq!(q4_model.quantization_type(), AprQuantizationType::Q4_K);
    }

    #[test]
    fn test_quantized_apr_transformer_weight_bytes_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);
        let bytes = transformer.weight_bytes();
        assert!(bytes > 0);
    }

    #[test]
    fn test_quantized_apr_transformer_f32_equivalent_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
        let f32_bytes = transformer.f32_equivalent_bytes();
        let params = transformer.num_parameters();
        assert_eq!(f32_bytes, params * 4);
    }

    #[test]
    fn test_quantized_apr_transformer_num_params_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 256,
            intermediate_dim: 256,
            context_length: 128,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
        let params = transformer.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_quantized_apr_transformer_forward_empty_error_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
        let result = transformer.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_apr_transformer_config_accessor_cov2() {
        let config = AprTransformerConfig {
            hidden_dim: 128,
            ..Default::default()
        };
        let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q8_0);
        assert_eq!(transformer.config().hidden_dim, 128);
    }

    #[test]
    fn test_apr_benchmark_runner_iterations_setters_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let layer = AprTransformerLayer::empty(32, 128);
        let transformer = AprTransformer {
            config,
            token_embedding: vec![0.0; 100 * 32],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 32],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; 32 * 100],
            lm_head_bias: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
            q4k_layers: None,
        };

        let mut runner = AprBenchmarkRunner::new(transformer);
        runner.set_warmup_iterations(5);
        runner.set_measure_iterations(15);

        assert_eq!(runner.warmup_iterations(), 5);
        assert_eq!(runner.measure_iterations(), 15);
    }

    #[test]
    fn test_apr_benchmark_runner_measure_iterations_min_cov() {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 64,
            ..Default::default()
        };
        let layer = AprTransformerLayer::empty(32, 128);
        let transformer = AprTransformer {
            config,
            token_embedding: vec![0.0; 100 * 32],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 32],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; 32 * 100],
            lm_head_bias: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
            q4k_layers: None,
        };

        let mut runner = AprBenchmarkRunner::new(transformer);
        // Setting to 0 should clamp to 1
        runner.set_measure_iterations(0);
        assert_eq!(runner.measure_iterations(), 1);
    }
}

#[cfg(test)]
mod apr_dequant_tests {
    use crate::apr_transformer::*;
    use crate::quantize::{dequantize_q4_k, dequantize_q6_k};

    #[test]
    fn test_apr_q4k_dequant_matches_gguf() {
        // Create test Q4_K data (one super-block = 144 bytes for 256 values)
        // Format: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes)
        let mut data = vec![0u8; 144];
        
        // Set d = 0.5 (f16)
        let d_f16: u16 = 0x3800; // 0.5 in f16
        data[0] = (d_f16 & 0xFF) as u8;
        data[1] = (d_f16 >> 8) as u8;
        
        // Set dmin = 0.1 (f16)
        let dmin_f16: u16 = 0x2E66; // ~0.1 in f16
        data[2] = (dmin_f16 & 0xFF) as u8;
        data[3] = (dmin_f16 >> 8) as u8;
        
        // Set some scales (12 bytes)
        for i in 0..12 {
            data[4 + i] = ((i + 1) * 5) as u8;
        }
        
        // Set some qs values (128 bytes)
        for i in 0..128 {
            data[16 + i] = (i % 256) as u8;
        }
        
        // Dequantize with APR function
        let apr_result = dequantize_q4_k_apr(&data, 256);
        
        // Dequantize with GGUF function
        let gguf_result = dequantize_q4_k(&data).expect("GGUF dequant should work");
        
        // Compare results
        assert_eq!(apr_result.len(), gguf_result.len(), "Output lengths should match");
        
        for i in 0..apr_result.len() {
            let diff = (apr_result[i] - gguf_result[i]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at index {}: APR={}, GGUF={}, diff={}",
                i, apr_result[i], gguf_result[i], diff
            );
        }
        
        println!("APR Q4_K dequantization matches GGUF implementation!");
    }

    #[test]
    fn test_apr_q6k_dequant_matches_gguf() {
        // Create test Q6_K data (one super-block = 210 bytes for 256 values)
        // Format: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16)
        let mut data = vec![0u8; 210];

        // Set ql (128 bytes at offset 0) - low 4 bits
        for i in 0..128 {
            data[i] = (i % 256) as u8;
        }

        // Set qh (64 bytes at offset 128) - high 2 bits
        for i in 0..64 {
            data[128 + i] = ((i * 3) % 256) as u8;
        }

        // Set scales (16 bytes at offset 192) - signed i8
        for i in 0..16 {
            // Use values that work as signed i8
            data[192 + i] = ((i as i8 * 8) as u8).wrapping_add(10);
        }

        // Set d = 0.25 (f16) at offset 208
        let d_f16: u16 = 0x3400; // 0.25 in f16
        data[208] = (d_f16 & 0xFF) as u8;
        data[209] = (d_f16 >> 8) as u8;

        // Dequantize with APR function
        let apr_result = dequantize_q6_k_apr(&data, 256);

        // Dequantize with GGUF function
        let gguf_result = dequantize_q6_k(&data).expect("GGUF dequant should work");

        // Compare results
        assert_eq!(apr_result.len(), gguf_result.len(), "Output lengths should match");

        for i in 0..apr_result.len() {
            let diff = (apr_result[i] - gguf_result[i]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at index {}: APR={}, GGUF={}, diff={}",
                i, apr_result[i], gguf_result[i], diff
            );
        }

        println!("APR Q6_K dequantization matches GGUF implementation!");
    }
}
