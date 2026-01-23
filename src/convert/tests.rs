#[cfg(test)]
mod tests {
    use crate::convert::*;

    // ==========================================================================
    // Converter Tests
    // ==========================================================================

    #[test]
    fn test_from_gguf_transformer_config_preserved() {
        // Create a mock GGUF transformer
        let gguf = create_mock_gguf_transformer(4, 1, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.config.architecture, gguf.config.architecture);
        assert_eq!(apr.config.hidden_dim, gguf.config.hidden_dim);
        assert_eq!(apr.config.num_layers, gguf.config.num_layers);
        assert_eq!(apr.config.vocab_size, gguf.config.vocab_size);
    }

    #[test]
    fn test_from_gguf_transformer_weights_preserved() {
        let gguf = create_mock_gguf_transformer(4, 1, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.token_embedding, gguf.token_embedding);
        assert_eq!(apr.output_norm_weight, gguf.output_norm_weight);
        assert_eq!(apr.lm_head_weight, gguf.lm_head_weight);
    }

    #[test]
    fn test_from_gguf_transformer_layers_preserved() {
        let gguf = create_mock_gguf_transformer(4, 2, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.layers.len(), gguf.layers.len());
        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.attn_norm_weight, gguf_layer.attn_norm_weight);
            assert_eq!(apr_layer.qkv_weight, gguf_layer.qkv_weight);
            assert_eq!(apr_layer.ffn_up_weight, gguf_layer.ffn_up_weight);
            assert_eq!(apr_layer.ffn_down_weight, gguf_layer.ffn_down_weight);
        }
    }

    // ==========================================================================
    // APR Serialization Tests
    // ==========================================================================

    #[test]
    fn test_to_apr_bytes_header_valid() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Check header (APR v2 format)
        assert_eq!(&bytes[0..4], &MAGIC); // APR2 magic
        assert_eq!(bytes[4], 2); // version major (v2)
        assert_eq!(bytes[5], 0); // version minor

        // Check tensor count (at bytes 8-11 in v2)
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(tensor_count, 1); // We store weights as single tensor
    }

    #[test]
    fn test_apr_bytes_roundtrip() {
        let original = create_test_apr_transformer(4, 1, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(original.config, loaded.config);
        assert_eq!(original.token_embedding, loaded.token_embedding);
        assert_eq!(original.layers.len(), loaded.layers.len());
    }

    #[test]
    fn test_from_apr_bytes_missing_weights() {
        // Create bytes with valid v2 header but no weights tensor
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&66u64.to_le_bytes()); // data offset (same = empty index)
        bytes[64..66].copy_from_slice(b"{}"); // minimal JSON metadata

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: no weights tensor
    }

    // ==========================================================================
    // Stats Tests
    // ==========================================================================

    #[test]
    fn test_stats_basic() {
        let apr = create_test_apr_transformer(64, 2, 1000, 256);
        let stats = GgufToAprConverter::stats(&apr);

        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.hidden_dim, 64);
        assert_eq!(stats.vocab_size, 1000);
        assert!(stats.total_parameters > 0);
        assert!(stats.memory_bytes_f32 > 0);
    }

    #[test]
    fn test_stats_memory_conversions() {
        let apr = create_test_apr_transformer(64, 1, 100, 128);
        let stats = GgufToAprConverter::stats(&apr);

        // Memory should be params * 4 bytes
        assert_eq!(stats.memory_bytes_f32, stats.total_parameters * 4);

        // MB should be bytes / 1M
        let expected_mb = stats.memory_bytes_f32 as f64 / (1024.0 * 1024.0);
        assert!((stats.memory_mb() - expected_mb).abs() < 0.0001);
    }

    #[test]
    fn test_stats_parameter_conversions() {
        let apr = create_test_apr_transformer(64, 1, 100, 128);
        let stats = GgufToAprConverter::stats(&apr);

        let expected_m = stats.total_parameters as f64 / 1_000_000.0;
        assert!((stats.parameters_m() - expected_m).abs() < 0.0001);
    }

    // ==========================================================================
    // Inference Equivalence Tests
    // ==========================================================================

    #[test]
    fn test_inference_produces_output() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let tokens = vec![1, 2, 3];

        let result = apr.forward(&tokens);
        assert!(result.is_ok());

        let logits = result.expect("forward");
        assert_eq!(logits.len(), apr.config.vocab_size);
    }

    #[test]
    fn test_inference_deterministic() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let tokens = vec![1, 2, 3];

        let logits1 = apr.forward(&tokens).expect("forward 1");
        let logits2 = apr.forward(&tokens).expect("forward 2");

        assert_eq!(logits1, logits2, "Inference should be deterministic");
    }

    // ==========================================================================
    // Helper Functions
    // ==========================================================================

    fn create_mock_gguf_transformer(
        hidden_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        intermediate_dim: usize,
    ) -> GGUFTransformer {
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

        let config = GGUFConfig {
            architecture: "test_arch".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0, // NORM style (adjacent pairs)
        };

        let layers: Vec<GGUFTransformerLayer> = (0..num_layers)
            .map(|_| GGUFTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
                qkv_bias: None,
                attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
            })
            .collect();

        GGUFTransformer {
            config,
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; hidden_dim * vocab_size],
            lm_head_bias: None,
        }
    }

    fn create_test_apr_transformer(
        hidden_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        intermediate_dim: usize,
    ) -> AprTransformer {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let layers: Vec<AprTransformerLayer> = (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
                qkv_bias: None,
                attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
            })
            .collect();

        AprTransformer {
            config,
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; hidden_dim * vocab_size],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    // ==========================================================================
    // ConversionStats Coverage Tests
    // ==========================================================================

    #[test]
    fn test_stats_memory_gb() {
        let stats = ConversionStats {
            total_parameters: 1_000_000_000, // 1B params
            memory_bytes_f32: 4_000_000_000, // 4GB
            num_layers: 24,
            hidden_dim: 2048,
            vocab_size: 50000,
            architecture: "test".to_string(),
        };

        let expected_gb = 4.0 / 1.073741824; // 4GB / GiB conversion
        assert!((stats.memory_gb() - expected_gb).abs() < 0.1);
    }

    #[test]
    fn test_stats_parameters_b() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000,  // 7B params
            memory_bytes_f32: 28_000_000_000, // 28GB
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };

        assert!((stats.parameters_b() - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_debug() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 32,
            vocab_size: 100,
            architecture: "mini".to_string(),
        };

        // Test Debug trait
        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("mini"));
        assert!(debug_str.contains("1000"));
    }

    #[test]
    fn test_stats_clone() {
        let stats = ConversionStats {
            total_parameters: 500,
            memory_bytes_f32: 2000,
            num_layers: 2,
            hidden_dim: 16,
            vocab_size: 50,
            architecture: "tiny".to_string(),
        };

        let cloned = stats.clone();
        assert_eq!(cloned.total_parameters, stats.total_parameters);
        assert_eq!(cloned.architecture, stats.architecture);
    }

    // ==========================================================================
    // Error Path Coverage Tests
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_truncated_tensor_index() {
        // Create bytes with valid v2 header but truncated before tensor index
        let mut bytes = vec![0u8; 80]; // Just past header
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data offset beyond end
        bytes[64..66].copy_from_slice(b"{}"); // minimal JSON metadata

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: truncated
    }

    #[test]
    fn test_from_apr_bytes_truncated_tensor_data() {
        // Create bytes with valid header and index but truncated data
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&110u64.to_le_bytes()); // data starts at 110
        bytes[64..66].copy_from_slice(b"{}");

        // Add a tensor index entry pointing to data beyond file end
        let index_json =
            r#"[{"name":"weights","dtype":"json","shape":[1000],"offset":0,"size":1000}]"#;
        let index_bytes = index_json.as_bytes();
        let index_end = 66 + index_bytes.len();
        bytes.resize(index_end + 10, 0); // Only add 10 bytes, not 1000
        bytes[66..index_end].copy_from_slice(index_bytes);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: truncated tensor data
    }

    #[test]
    fn test_from_apr_bytes_invalid_json_tensor_index() {
        // Create bytes with valid header but invalid JSON in tensor index
        let mut bytes = vec![0u8; 100];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // index at 66
        bytes[32..40].copy_from_slice(&90u64.to_le_bytes()); // data at 90
        bytes[64..66].copy_from_slice(b"{}");
        // Invalid JSON at tensor index position
        bytes[66..78].copy_from_slice(b"not valid js");

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: invalid JSON
    }

    // ==========================================================================
    // RawTensor Coverage Tests
    // ==========================================================================

    #[test]
    fn test_raw_tensor_debug() {
        let tensor = RawTensor {
            name: "test.weight".to_string(),
            data: vec![0u8; 100],
            shape: vec![10, 10],
            dtype: 0, // F32
        };

        let debug_str = format!("{tensor:?}");
        assert!(debug_str.contains("test.weight"));
        assert!(debug_str.contains("[10, 10]"));
    }

    #[test]
    fn test_raw_tensor_clone() {
        let tensor = RawTensor {
            name: "test.weight".to_string(),
            data: vec![1, 2, 3, 4],
            shape: vec![2, 2],
            dtype: 1, // F16
        };

        let cloned = tensor.clone();
        assert_eq!(cloned.name, tensor.name);
        assert_eq!(cloned.data, tensor.data);
        assert_eq!(cloned.shape, tensor.shape);
        assert_eq!(cloned.dtype, tensor.dtype);
    }

    // ==========================================================================
    // GgufToAprQ4KConverter Helper Tests
    // ==========================================================================

    #[test]
    fn test_get_string_helper() {
        use crate::gguf::GGUFValue;
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert(
            "name".to_string(),
            GGUFValue::String("test_model".to_string()),
        );
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "name");
        assert_eq!(result, Some("test_model".to_string()));

        let missing = GgufToAprQ4KConverter::get_string(&metadata, "nonexistent");
        assert_eq!(missing, None);

        // Test wrong type returns None
        let wrong_type = GgufToAprQ4KConverter::get_string(&metadata, "count");
        assert_eq!(wrong_type, None);
    }

    #[test]
    fn test_get_u32_helper() {
        use crate::gguf::GGUFValue;
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));
        metadata.insert("signed".to_string(), GGUFValue::Int32(100));
        metadata.insert("big".to_string(), GGUFValue::UInt64(200));
        metadata.insert("name".to_string(), GGUFValue::String("test".to_string()));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "count");
        assert_eq!(result, Some(42));

        let signed = GgufToAprQ4KConverter::get_u32(&metadata, "signed");
        assert_eq!(signed, Some(100));

        let big = GgufToAprQ4KConverter::get_u32(&metadata, "big");
        assert_eq!(big, Some(200));

        let missing = GgufToAprQ4KConverter::get_u32(&metadata, "nonexistent");
        assert_eq!(missing, None);

        let wrong_type = GgufToAprQ4KConverter::get_u32(&metadata, "name");
        assert_eq!(wrong_type, None);
    }

    #[test]
    fn test_get_f32_helper() {
        use crate::gguf::GGUFValue;
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert("scale".to_string(), GGUFValue::Float32(3.14));
        metadata.insert("big_scale".to_string(), GGUFValue::Float64(2.71828));
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "scale");
        assert!(result.is_some());
        assert!((result.expect("operation failed") - 3.14).abs() < 0.001);

        let big = GgufToAprQ4KConverter::get_f32(&metadata, "big_scale");
        assert!(big.is_some());
        assert!((big.expect("operation failed") - 2.71828).abs() < 0.001);

        let missing = GgufToAprQ4KConverter::get_f32(&metadata, "nonexistent");
        assert_eq!(missing, None);

        let wrong_type = GgufToAprQ4KConverter::get_f32(&metadata, "count");
        assert_eq!(wrong_type, None);
    }

    // ==========================================================================
    // Q4KConversionStats Coverage Tests
    // ==========================================================================

    #[test]
    fn test_q4k_conversion_stats_debug() {
        let stats = Q4KConversionStats {
            tensor_count: 100,
            q4k_tensor_count: 80,
            total_bytes: 1_000_000,
            architecture: "llama".to_string(),
            num_layers: 32,
            hidden_size: 4096,
        };

        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("llama"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("32"));
    }

    #[test]
    fn test_q4k_conversion_stats_clone() {
        let stats = Q4KConversionStats {
            tensor_count: 50,
            q4k_tensor_count: 40,
            total_bytes: 500_000,
            architecture: "qwen".to_string(),
            num_layers: 16,
            hidden_size: 2048,
        };

        let cloned = stats.clone();
        assert_eq!(cloned.tensor_count, stats.tensor_count);
        assert_eq!(cloned.architecture, stats.architecture);
        assert_eq!(cloned.num_layers, stats.num_layers);
    }

    // ==========================================================================
    // Additional From APR Bytes Error Tests
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_v1_format() {
        // Create APR v1 format header (should be handled or error gracefully)
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 1; // v1 (not v2)
        bytes[5] = 0;

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        // May succeed with v1 fallback or fail, but shouldn't panic
        let _ = result;
    }

    #[test]
    fn test_from_apr_bytes_wrong_magic() {
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(b"XXXX"); // Wrong magic
        bytes[4] = 2;

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_too_short() {
        // Only 4 bytes (magic only)
        let bytes = vec![0x41, 0x50, 0x52, 0x32]; // APR2

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    // ==========================================================================
    // Stats Edge Cases
    // ==========================================================================

    #[test]
    fn test_stats_zero_params() {
        let stats = ConversionStats {
            total_parameters: 0,
            memory_bytes_f32: 0,
            num_layers: 0,
            hidden_dim: 0,
            vocab_size: 0,
            architecture: "empty".to_string(),
        };

        assert_eq!(stats.memory_mb(), 0.0);
        assert_eq!(stats.memory_gb(), 0.0);
        assert_eq!(stats.parameters_m(), 0.0);
        assert_eq!(stats.parameters_b(), 0.0);
    }

    #[test]
    fn test_stats_small_model() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 16,
            vocab_size: 100,
            architecture: "tiny".to_string(),
        };

        assert!(stats.memory_mb() > 0.0);
        assert!(stats.parameters_m() > 0.0);
        assert!(stats.parameters_b() < 0.001);
    }

    // ==========================================================================
    // APR Bytes Serialization Additional Tests
    // ==========================================================================

    #[test]
    fn test_to_apr_bytes_multiple_layers() {
        let apr = create_test_apr_transformer(64, 4, 1000, 256);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Should have valid header
        assert_eq!(&bytes[0..4], &MAGIC);
        assert!(bytes.len() > HEADER_SIZE);
    }

    #[test]
    fn test_to_apr_bytes_single_layer() {
        let apr = create_test_apr_transformer(32, 1, 100, 64);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        assert_eq!(&bytes[0..4], &MAGIC);
    }

    #[test]
    fn test_apr_roundtrip_multiple_layers() {
        let original = create_test_apr_transformer(32, 3, 500, 128);
        let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(original.config.num_layers, loaded.config.num_layers);
        assert_eq!(original.layers.len(), loaded.layers.len());
    }

    // ==========================================================================
    // Coverage Tests: ConversionStats
    // ==========================================================================

    #[test]
    fn test_conversion_stats_debug() {
        let stats = ConversionStats {
            total_parameters: 1_000_000,
            memory_bytes_f32: 4_000_000,
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50000,
            architecture: "bert".to_string(),
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ConversionStats"));
        assert!(debug_str.contains("1000000"));
        assert!(debug_str.contains("bert"));
    }

    #[test]
    fn test_conversion_stats_clone() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000,
            memory_bytes_f32: 28_000_000_000,
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_parameters, stats.total_parameters);
        assert_eq!(cloned.architecture, stats.architecture);
    }

    #[test]
    fn test_conversion_stats_large_model() {
        let stats = ConversionStats {
            total_parameters: 70_000_000_000, // 70B
            memory_bytes_f32: 280_000_000_000,
            num_layers: 80,
            hidden_dim: 8192,
            vocab_size: 128000,
            architecture: "llama3".to_string(),
        };
        assert!(stats.parameters_b() > 69.0 && stats.parameters_b() < 71.0);
        assert!(stats.memory_gb() > 250.0);
    }

    // ==========================================================================
    // Coverage Tests: from_apr_bytes error paths
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_invalid_tensor_index_json() {
        // Create header pointing to invalid JSON
        let mut bytes = vec![0u8; 200];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&100u64.to_le_bytes()); // data offset
        bytes[64..66].copy_from_slice(b"{}"); // metadata
                                              // Invalid JSON for tensor index (length must match exactly)
        let invalid_json = b"not valid json{{{";
        bytes[66..66 + invalid_json.len()].copy_from_slice(invalid_json);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    // ==========================================================================
    // Coverage Tests: GgufToAprQ4KConverter helpers
    // ==========================================================================

    #[test]
    fn test_q4k_converter_get_string_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_string(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_f32_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_string_present() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("value".to_string()));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn test_q4k_converter_get_u32_from_int32() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(42));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_q4k_converter_get_u32_from_uint64() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(100));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(100));
    }

    #[test]
    fn test_q4k_converter_get_f32_from_float64() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(3.14159));

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.expect("operation failed") - 3.14159).abs() < 0.0001);
    }

    #[test]
    fn test_q4k_converter_get_string_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "key".to_string(),
            GGUFValue::String("not a number".to_string()),
        );

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_f32_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "key".to_string(),
            GGUFValue::String("not a float".to_string()),
        );

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_none());
    }

    // =========================================================================
    // Coverage Tests: GgufToAprConverter
    // =========================================================================

    #[test]
    fn test_gguf_to_apr_converter_zero_layers_cov() {
        let gguf = create_mock_gguf_transformer(4, 0, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!(apr.layers.is_empty());
    }

    #[test]
    fn test_gguf_to_apr_converter_multiple_layers_cov() {
        let gguf = create_mock_gguf_transformer(8, 4, 20, 16);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.layers.len(), 4);
    }

    #[test]
    fn test_gguf_to_apr_converter_config_all_fields_cov() {
        let gguf = create_mock_gguf_transformer(16, 2, 32, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.config.hidden_dim, gguf.config.hidden_dim);
        assert_eq!(apr.config.num_layers, gguf.config.num_layers);
        assert_eq!(apr.config.num_heads, gguf.config.num_heads);
        assert_eq!(apr.config.num_kv_heads, gguf.config.num_kv_heads);
        assert_eq!(apr.config.vocab_size, gguf.config.vocab_size);
        assert_eq!(apr.config.intermediate_dim, gguf.config.intermediate_dim);
        assert_eq!(apr.config.context_length, gguf.config.context_length);
        assert_eq!(apr.config.rope_theta, gguf.config.rope_theta);
        assert_eq!(apr.config.eps, gguf.config.eps);
    }

    // =========================================================================
    // Coverage Tests: APR bytes serialization
    // =========================================================================

    #[test]
    fn test_to_apr_bytes_alignment_cov() {
        let apr = create_test_apr_transformer(8, 2, 20, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Check header is 64 bytes (HEADER_SIZE) and data follows
        assert!(bytes.len() >= HEADER_SIZE);
        // Metadata offset starts at HEADER_SIZE (64 bytes)
        let metadata_offset =
            u64::from_le_bytes(bytes[12..20].try_into().expect("index out of bounds")) as usize;
        assert_eq!(metadata_offset, HEADER_SIZE);
    }

    #[test]
    fn test_apr_bytes_large_model_cov() {
        // Larger model to test serialization
        let apr = create_test_apr_transformer(64, 4, 100, 32);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Should successfully serialize
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], &MAGIC);
    }

    // =========================================================================
    // Coverage Tests: from_apr_bytes edge cases
    // =========================================================================

    #[test]
    fn test_from_apr_bytes_too_short_cov() {
        let bytes = vec![0u8; 10]; // Too short for header
        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_bad_magic_cov() {
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(b"BADM"); // Wrong magic
        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_wrong_version_cov() {
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 99; // Unsupported version
        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    // =========================================================================
    // Coverage Tests: Q4K converter helper functions
    // =========================================================================

    #[test]
    fn test_q4k_converter_get_string_missing_cov() {
        use std::collections::HashMap;
        let metadata: HashMap<String, crate::gguf::GGUFValue> = HashMap::new();
        let result = GgufToAprQ4KConverter::get_string(&metadata, "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_missing_cov() {
        use std::collections::HashMap;
        let metadata: HashMap<String, crate::gguf::GGUFValue> = HashMap::new();
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_f32_missing_cov() {
        use std::collections::HashMap;
        let metadata: HashMap<String, crate::gguf::GGUFValue> = HashMap::new();
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_from_uint32_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(256));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(256));
    }

    #[test]
    fn test_q4k_converter_get_u32_from_uint64_large_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(1024));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(1024));
    }

    #[test]
    fn test_q4k_converter_get_f32_from_float32_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float32(1.5));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.expect("operation failed") - 1.5).abs() < 0.0001);
    }

    // =========================================================================
    // Coverage Tests: ConversionStats
    // =========================================================================

    #[test]
    fn test_conversion_stats_fields_cov() {
        let stats = ConversionStats {
            total_parameters: 1000000,
            memory_bytes_f32: 4000000,
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        assert_eq!(stats.total_parameters, 1000000);
        assert_eq!(stats.memory_bytes_f32, 4000000);
        assert_eq!(stats.num_layers, 12);
        assert_eq!(stats.hidden_dim, 768);
        assert_eq!(stats.vocab_size, 32000);
        assert_eq!(stats.architecture, "llama");
    }

    #[test]
    fn test_conversion_stats_zero_values_cov() {
        let stats = ConversionStats {
            total_parameters: 0,
            memory_bytes_f32: 0,
            num_layers: 0,
            hidden_dim: 0,
            vocab_size: 0,
            architecture: String::new(),
        };
        assert_eq!(stats.total_parameters, 0);
        assert_eq!(stats.memory_bytes_f32, 0);
        assert_eq!(stats.num_layers, 0);
    }

    #[test]
    fn test_conversion_stats_debug_clone_cov() {
        let stats = ConversionStats {
            total_parameters: 500000,
            memory_bytes_f32: 2000000,
            num_layers: 6,
            hidden_dim: 512,
            vocab_size: 10000,
            architecture: "phi".to_string(),
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ConversionStats"));

        let cloned = stats.clone();
        assert_eq!(cloned.total_parameters, stats.total_parameters);
    }

    // =========================================================================
    // Coverage Tests: AprTransformerConfig
    // =========================================================================

    #[test]
    fn test_apr_transformer_config_partial_eq_cov() {
        let config1 = create_test_apr_transformer(4, 1, 10, 8).config;
        let config2 = create_test_apr_transformer(4, 1, 10, 8).config;
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_apr_transformer_config_not_equal_cov() {
        let config1 = create_test_apr_transformer(4, 1, 10, 8).config;
        let config2 = create_test_apr_transformer(8, 2, 20, 16).config;
        assert_ne!(config1, config2);
    }

    // =========================================================================
    // Coverage Tests: RawTensor struct
    // =========================================================================

    #[test]
    fn test_raw_tensor_debug_cov() {
        let tensor = RawTensor {
            name: "test_tensor".to_string(),
            data: vec![1, 2, 3, 4],
            shape: vec![2, 2],
            dtype: 0, // F32
        };
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("test_tensor"));
        assert!(debug_str.contains("shape"));
    }

    #[test]
    fn test_raw_tensor_clone_cov() {
        let tensor = RawTensor {
            name: "cloneable".to_string(),
            data: vec![10, 20, 30],
            shape: vec![3],
            dtype: 1, // F16
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.name, tensor.name);
        assert_eq!(cloned.data, tensor.data);
        assert_eq!(cloned.shape, tensor.shape);
        assert_eq!(cloned.dtype, tensor.dtype);
    }

    #[test]
    fn test_raw_tensor_empty_data_cov() {
        let tensor = RawTensor {
            name: "empty".to_string(),
            data: vec![],
            shape: vec![0],
            dtype: 0,
        };
        assert!(tensor.data.is_empty());
        assert_eq!(tensor.shape, vec![0]);
    }

    #[test]
    fn test_raw_tensor_large_dtype_cov() {
        let tensor = RawTensor {
            name: "q4k".to_string(),
            data: vec![0u8; 144], // Q4_K: 256 elements = 144 bytes
            shape: vec![256],
            dtype: 12, // Q4_K
        };
        assert_eq!(tensor.dtype, 12);
        assert_eq!(tensor.data.len(), 144);
    }

    #[test]
    fn test_raw_tensor_multidim_shape_cov() {
        let tensor = RawTensor {
            name: "3d_tensor".to_string(),
            data: vec![0u8; 24],
            shape: vec![2, 3, 4],
            dtype: 0,
        };
        assert_eq!(tensor.shape.len(), 3);
        let total: usize = tensor.shape.iter().product();
        assert_eq!(total, 24);
    }

    // =========================================================================
    // Coverage Tests: Q4KConversionStats struct
    // =========================================================================

    #[test]
    fn test_q4k_conversion_stats_all_fields_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 200,
            q4k_tensor_count: 150,
            total_bytes: 5_000_000,
            architecture: "mistral".to_string(),
            num_layers: 24,
            hidden_size: 3072,
        };
        assert_eq!(stats.tensor_count, 200);
        assert_eq!(stats.q4k_tensor_count, 150);
        assert_eq!(stats.total_bytes, 5_000_000);
        assert_eq!(stats.architecture, "mistral");
        assert_eq!(stats.num_layers, 24);
        assert_eq!(stats.hidden_size, 3072);
    }

    #[test]
    fn test_q4k_conversion_stats_zero_values_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 0,
            q4k_tensor_count: 0,
            total_bytes: 0,
            architecture: String::new(),
            num_layers: 0,
            hidden_size: 0,
        };
        assert_eq!(stats.tensor_count, 0);
        assert_eq!(stats.total_bytes, 0);
        assert!(stats.architecture.is_empty());
    }

    // =========================================================================
    // Coverage Tests: GgufToAprConverter::stats
    // =========================================================================

    #[test]
    fn test_stats_function_cov() {
        let apr = create_test_apr_transformer(16, 4, 1000, 64);
        let stats = GgufToAprConverter::stats(&apr);

        assert_eq!(stats.num_layers, 4);
        assert_eq!(stats.hidden_dim, 16);
        assert_eq!(stats.vocab_size, 1000);
        assert!(stats.total_parameters > 0);
        assert!(stats.memory_bytes_f32 > 0);
    }

    #[test]
    fn test_stats_memory_calculations_cov() {
        let apr = create_test_apr_transformer(32, 2, 500, 128);
        let stats = GgufToAprConverter::stats(&apr);

        // Memory should be related to parameters
        assert!(stats.memory_mb() > 0.0);
        assert!(stats.memory_gb() < stats.memory_mb());
        assert!(stats.parameters_m() > 0.0);
    }

    // =========================================================================
    // Coverage Tests: from_apr_bytes additional error paths
    // =========================================================================

    #[test]
    fn test_from_apr_bytes_truncated_data_cov() {
        // Create valid header but truncate before data section
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let mut bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Truncate to only keep header and partial metadata
        bytes.truncate(100);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_invalid_metadata_json_cov() {
        let mut bytes = vec![0u8; 256];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&50u32.to_le_bytes()); // metadata size = 50
        bytes[24..32].copy_from_slice(&120u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data offset

        // Invalid JSON metadata (just garbage bytes)
        for i in 64..114 {
            bytes[i] = b'X';
        }

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        // Should error due to invalid metadata JSON
        assert!(result.is_err());
    }

    // =========================================================================
    // Coverage Tests: to_apr_bytes edge cases
    // =========================================================================

    #[test]
    fn test_to_apr_bytes_empty_layers_cov() {
        let apr = create_test_apr_transformer(4, 0, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Should still serialize with valid header
        assert_eq!(&bytes[0..4], &MAGIC);
        assert!(bytes.len() > HEADER_SIZE);
    }

    #[test]
    fn test_to_apr_bytes_large_vocab_cov() {
        let apr = create_test_apr_transformer(8, 1, 50000, 32);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], &MAGIC);
    }

    #[test]
    fn test_to_apr_bytes_small_dims_cov() {
        // Minimum viable transformer
        let apr = create_test_apr_transformer(2, 1, 4, 2);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        assert!(!bytes.is_empty());
    }

    // =========================================================================
    // Coverage Tests: GgufToAprConverter::from_gguf_transformer edge cases
    // =========================================================================

    #[test]
    fn test_from_gguf_transformer_large_model_cov() {
        let gguf = create_mock_gguf_transformer(128, 8, 32000, 256);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.config.hidden_dim, 128);
        assert_eq!(apr.config.num_layers, 8);
        assert_eq!(apr.config.vocab_size, 32000);
        assert_eq!(apr.layers.len(), 8);
    }

    #[test]
    fn test_from_gguf_transformer_single_head_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 1);
        gguf.config.num_heads = 1;
        gguf.config.num_kv_heads = 1;

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.config.num_heads, 1);
        assert_eq!(apr.config.num_kv_heads, 1);
    }

    #[test]
    fn test_from_gguf_transformer_gqa_cov() {
        // Grouped Query Attention: num_kv_heads < num_heads
        let mut gguf = create_mock_gguf_transformer(32, 2, 100, 8);
        gguf.config.num_heads = 8;
        gguf.config.num_kv_heads = 2; // GQA ratio of 4:1

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.config.num_heads, 8);
        assert_eq!(apr.config.num_kv_heads, 2);
    }

    // =========================================================================
    // Coverage Tests: Layer weight preservation
    // =========================================================================

    #[test]
    fn test_layer_bias_preservation_cov() {
        let gguf = create_mock_gguf_transformer(8, 2, 20, 4);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.attn_norm_bias, gguf_layer.attn_norm_bias);
            assert_eq!(apr_layer.qkv_bias, gguf_layer.qkv_bias);
            assert_eq!(apr_layer.attn_output_bias, gguf_layer.attn_output_bias);
            assert_eq!(apr_layer.ffn_gate_bias, gguf_layer.ffn_gate_bias);
            assert_eq!(apr_layer.ffn_up_bias, gguf_layer.ffn_up_bias);
            assert_eq!(apr_layer.ffn_down_bias, gguf_layer.ffn_down_bias);
            assert_eq!(apr_layer.ffn_norm_bias, gguf_layer.ffn_norm_bias);
        }
    }

    #[test]
    fn test_layer_ffn_weights_cov() {
        let gguf = create_mock_gguf_transformer(16, 3, 50, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.ffn_gate_weight, gguf_layer.ffn_gate_weight);
            assert_eq!(apr_layer.ffn_up_weight, gguf_layer.ffn_up_weight);
            assert_eq!(apr_layer.ffn_down_weight, gguf_layer.ffn_down_weight);
            assert_eq!(apr_layer.ffn_norm_weight, gguf_layer.ffn_norm_weight);
        }
    }

    #[test]
    fn test_layer_attention_weights_cov() {
        let gguf = create_mock_gguf_transformer(32, 4, 100, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.attn_norm_weight, gguf_layer.attn_norm_weight);
            assert_eq!(apr_layer.qkv_weight, gguf_layer.qkv_weight);
            assert_eq!(apr_layer.attn_output_weight, gguf_layer.attn_output_weight);
        }
    }

    // =========================================================================
    // Coverage Tests: Config field preservation
    // =========================================================================

    #[test]
    fn test_config_rope_theta_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 4);
        gguf.config.rope_theta = 1000000.0; // Llama-3 style

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!((apr.config.rope_theta - 1000000.0).abs() < 0.01);
    }

    #[test]
    fn test_config_eps_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 4);
        gguf.config.eps = 1e-6; // Different epsilon

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!((apr.config.eps - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn test_config_context_length_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 4);
        gguf.config.context_length = 131072; // Long context

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.config.context_length, 131072);
    }

    #[test]
    fn test_config_intermediate_dim_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 4);
        gguf.config.intermediate_dim = 2048;

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.config.intermediate_dim, 2048);
    }

    // =========================================================================
    // Coverage Tests: get_f32 Float64 conversion
    // =========================================================================

    #[test]
    fn test_q4k_converter_get_f32_float64_large_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(1e10));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
    }

    #[test]
    fn test_q4k_converter_get_f32_float64_small_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(1e-10));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!(result.expect("operation failed") > 0.0);
    }

    // =========================================================================
    // Coverage Tests: get_u32 Int32 conversion
    // =========================================================================

    #[test]
    fn test_q4k_converter_get_u32_int32_positive_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(12345));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(12345));
    }

    #[test]
    fn test_q4k_converter_get_u32_int32_zero_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(0));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(0));
    }

    // =========================================================================
    // Extended Coverage Tests for ConversionStats
    // =========================================================================

    #[test]
    fn test_conversion_stats_memory_mb_cov() {
        let stats = ConversionStats {
            total_parameters: 1_000_000,
            memory_bytes_f32: 4 * 1024 * 1024, // 4 MB
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50257,
            architecture: "gpt2".to_string(),
        };
        assert!((stats.memory_mb() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_conversion_stats_memory_gb_cov() {
        let stats = ConversionStats {
            total_parameters: 1_000_000_000,
            memory_bytes_f32: 4 * 1024 * 1024 * 1024, // 4 GB
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        assert!((stats.memory_gb() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_conversion_stats_parameters_m_cov() {
        let stats = ConversionStats {
            total_parameters: 125_000_000, // 125M
            memory_bytes_f32: 500_000_000,
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50257,
            architecture: "gpt2".to_string(),
        };
        assert!((stats.parameters_m() - 125.0).abs() < 0.1);
    }

    #[test]
    fn test_conversion_stats_parameters_b_cov() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000, // 7B
            memory_bytes_f32: 28_000_000_000,
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        assert!((stats.parameters_b() - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_conversion_stats_clone_cov() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 64,
            vocab_size: 100,
            architecture: "test".to_string(),
        };
        let cloned = stats.clone();
        assert_eq!(stats.total_parameters, cloned.total_parameters);
        assert_eq!(stats.architecture, cloned.architecture);
    }

    #[test]
    fn test_conversion_stats_debug_cov() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 64,
            vocab_size: 100,
            architecture: "debug_test".to_string(),
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("total_parameters"));
        assert!(debug_str.contains("debug_test"));
    }

    // =========================================================================
    // Extended Coverage Tests for RawTensor
    // =========================================================================

    #[test]
    fn test_raw_tensor_new_cov() {
        let tensor = RawTensor {
            name: "test_tensor".to_string(),
            data: vec![0u8; 100],
            shape: vec![10, 10],
            dtype: 0, // F32
        };
        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.data.len(), 100);
        assert_eq!(tensor.shape, vec![10, 10]);
        assert_eq!(tensor.dtype, 0);
    }

    // =========================================================================
    // Extended Coverage Tests for Q4KConversionStats
    // =========================================================================

    #[test]
    fn test_q4k_conversion_stats_new_ext_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 10,
            q4k_tensor_count: 5,
            total_bytes: 1000,
            architecture: "llama".to_string(),
            num_layers: 12,
            hidden_size: 768,
        };
        assert_eq!(stats.tensor_count, 10);
        assert_eq!(stats.q4k_tensor_count, 5);
        assert_eq!(stats.total_bytes, 1000);
        assert_eq!(stats.architecture, "llama");
        assert_eq!(stats.num_layers, 12);
        assert_eq!(stats.hidden_size, 768);
    }

    #[test]
    fn test_q4k_conversion_stats_clone_ext_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 5,
            q4k_tensor_count: 3,
            total_bytes: 500,
            architecture: "gpt2".to_string(),
            num_layers: 6,
            hidden_size: 512,
        };
        let cloned = stats.clone();
        assert_eq!(stats.tensor_count, cloned.tensor_count);
        assert_eq!(stats.q4k_tensor_count, cloned.q4k_tensor_count);
        assert_eq!(stats.architecture, cloned.architecture);
    }

    #[test]
    fn test_q4k_conversion_stats_debug_ext_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 1,
            q4k_tensor_count: 1,
            total_bytes: 100,
            architecture: "tiny".to_string(),
            num_layers: 2,
            hidden_size: 256,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("tensor_count"));
        assert!(debug_str.contains("architecture"));
    }

    // =========================================================================
    // Extended Coverage Tests for GgufToAprQ4KConverter helpers
    // =========================================================================

    #[test]
    fn test_q4k_converter_get_string_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("value".to_string()));
        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn test_q4k_converter_get_string_wrong_type_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));
        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, None);
    }

    #[test]
    fn test_q4k_converter_get_u32_uint32_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(12345));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(12345));
    }

    #[test]
    fn test_q4k_converter_get_u32_uint64_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(999));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(999));
    }

    #[test]
    fn test_q4k_converter_get_u32_wrong_type_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "key".to_string(),
            GGUFValue::String("not a number".to_string()),
        );
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, None);
    }

    #[test]
    fn test_q4k_converter_get_f32_float32_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float32(3.14159));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.expect("operation failed") - 3.14159).abs() < 1e-5);
    }

    #[test]
    fn test_q4k_converter_get_f32_wrong_type_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert_eq!(result, None);
    }

    // ==========================================================================
    // Deep Coverage Tests (prefix: _deep_ccov_)
    // ==========================================================================

    #[test]
    fn test_deep_ccov_from_apr_bytes_invalid_transformer_json() {
        // Create valid APR header with a weights tensor that has invalid JSON
        let mut bytes = vec![0u8; 512];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[5] = 0;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data offset

        // Add minimal metadata at offset 64
        bytes[64..66].copy_from_slice(b"{}");

        // Add tensor index entry at offset 66 pointing to "weights"
        let tensor_index =
            r#"[{"name":"weights","dtype":"json","shape":[100],"offset":0,"size":100}]"#;
        let index_bytes = tensor_index.as_bytes();
        bytes[66..66 + index_bytes.len()].copy_from_slice(index_bytes);

        // Add invalid JSON for the weights payload at offset 200
        let invalid_weights = b"{ this is not valid json }}}";
        bytes[200..200 + invalid_weights.len()].copy_from_slice(invalid_weights);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(
            result.is_err(),
            "Should fail on invalid transformer JSON: {:?}",
            result
        );
    }

    #[test]
    fn test_deep_ccov_conversion_stats_memory_gb_precision() {
        // Test precision of memory_gb calculation with exact values
        let stats = ConversionStats {
            total_parameters: 1_073_741_824, // 2^30 = 1 GiB of params
            memory_bytes_f32: 1_073_741_824, // 1 GiB
            num_layers: 1,
            hidden_dim: 1,
            vocab_size: 1,
            architecture: "test".to_string(),
        };
        // 1 GiB = 1.0 GB in GiB units
        let expected_gb = 1.0;
        assert!(
            (stats.memory_gb() - expected_gb).abs() < 0.001,
            "Expected {}, got {}",
            expected_gb,
            stats.memory_gb()
        );
    }

    #[test]
    fn test_deep_ccov_conversion_stats_parameters_b_precision() {
        // Test precision of parameters_b calculation
        let stats = ConversionStats {
            total_parameters: 1_000_000_000, // Exactly 1B
            memory_bytes_f32: 4_000_000_000,
            num_layers: 1,
            hidden_dim: 1,
            vocab_size: 1,
            architecture: "test".to_string(),
        };
        assert!(
            (stats.parameters_b() - 1.0).abs() < 0.0001,
            "Expected 1.0B, got {}",
            stats.parameters_b()
        );
    }

    #[test]
    fn test_deep_ccov_raw_tensor_q8_0_dtype() {
        // Test Q8_0 dtype (GGML type 8)
        let tensor = RawTensor {
            name: "q8_0_weights".to_string(),
            data: vec![0u8; 34], // Q8_0: 32 elements = 34 bytes
            shape: vec![32],
            dtype: 8, // Q8_0
        };
        assert_eq!(tensor.dtype, 8);
        assert_eq!(tensor.data.len(), 34);
    }

    #[test]
    fn test_deep_ccov_raw_tensor_q5_k_dtype() {
        // Test Q5_K dtype (GGML type 13)
        let tensor = RawTensor {
            name: "q5_k_weights".to_string(),
            data: vec![0u8; 176], // Q5_K: 256 elements = 176 bytes
            shape: vec![256],
            dtype: 13, // Q5_K
        };
        assert_eq!(tensor.dtype, 13);
        assert_eq!(tensor.data.len(), 176);
    }

    #[test]
    fn test_deep_ccov_raw_tensor_q6_k_dtype() {
        // Test Q6_K dtype (GGML type 14)
        let tensor = RawTensor {
            name: "q6_k_weights".to_string(),
            data: vec![0u8; 210], // Q6_K: 256 elements = 210 bytes
            shape: vec![256],
            dtype: 14, // Q6_K
        };
        assert_eq!(tensor.dtype, 14);
        assert_eq!(tensor.data.len(), 210);
    }

    #[test]
    fn test_deep_ccov_raw_tensor_f16_dtype() {
        // Test F16 dtype (GGML type 1)
        let tensor = RawTensor {
            name: "f16_weights".to_string(),
            data: vec![0u8; 128], // 64 F16 elements = 128 bytes
            shape: vec![64],
            dtype: 1, // F16
        };
        assert_eq!(tensor.dtype, 1);
        assert_eq!(tensor.data.len(), 128);
    }

    #[test]
    fn test_deep_ccov_from_apr_bytes_exact_minimum_size() {
        // Test with exactly HEADER_SIZE bytes (minimum valid size for header parsing)
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[5] = 0;
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset at header end
        bytes[20..24].copy_from_slice(&0u32.to_le_bytes()); // 0 metadata size
        bytes[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor index at 64
        bytes[32..40].copy_from_slice(&64u64.to_le_bytes()); // data at 64

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        // Should fail because no weights tensor found or invalid tensor index
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_ccov_apr_bytes_roundtrip_with_biases() {
        // Test roundtrip with a transformer that has bias terms
        let config = AprTransformerConfig {
            architecture: "bias_test".to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 16,
            intermediate_dim: 16,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let layer = AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: Some(vec![0.1; 8]), // With bias
            qkv_weight: vec![0.01; 8 * 3 * 8],
            qkv_bias: Some(vec![0.02; 3 * 8]), // With bias
            attn_output_weight: vec![0.01; 8 * 8],
            attn_output_bias: Some(vec![0.03; 8]), // With bias
            ffn_gate_weight: Some(vec![0.01; 8 * 16]),
            ffn_gate_bias: Some(vec![0.04; 16]), // With bias
            ffn_up_weight: vec![0.01; 8 * 16],
            ffn_up_bias: Some(vec![0.05; 16]), // With bias
            ffn_down_weight: vec![0.01; 16 * 8],
            ffn_down_bias: Some(vec![0.06; 8]), // With bias
            ffn_norm_weight: Some(vec![1.0; 8]),
            ffn_norm_bias: Some(vec![0.07; 8]), // With bias
        };

        let original = AprTransformer {
            config,
            token_embedding: vec![0.1; 16 * 8],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 8],
            output_norm_bias: Some(vec![0.08; 8]), // With bias
            lm_head_weight: vec![0.01; 8 * 16],
            lm_head_bias: Some(vec![0.09; 16]), // With bias
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(original.config, loaded.config);
        assert_eq!(
            original.layers[0].attn_norm_bias,
            loaded.layers[0].attn_norm_bias
        );
        assert_eq!(original.lm_head_bias, loaded.lm_head_bias);
    }

    #[test]
    fn test_deep_ccov_stats_architecture_preservation() {
        // Test that stats preserves the full architecture name
        let apr = AprTransformer {
            config: AprTransformerConfig {
                architecture: "special-arch-name/v2.1".to_string(),
                hidden_dim: 4,
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                vocab_size: 10,
                intermediate_dim: 8,
                context_length: 32,
                rope_theta: 10000.0,
                eps: 1e-5,
            },
            token_embedding: vec![0.1; 40],
            layers: vec![AprTransformerLayer {
                attn_norm_weight: vec![1.0; 4],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; 48],
                qkv_bias: None,
                attn_output_weight: vec![0.01; 16],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; 32],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; 32],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
            }],
            output_norm_weight: vec![1.0; 4],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; 40],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        let stats = GgufToAprConverter::stats(&apr);
        assert_eq!(stats.architecture, "special-arch-name/v2.1");
    }

    #[test]
    fn test_deep_ccov_from_gguf_transformer_all_optional_biases() {
        // Test conversion when all optional biases are None
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

        let config = GGUFConfig {
            architecture: "no_bias".to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 16,
            intermediate_dim: 16,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 192],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 64],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 128],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        };

        let gguf = GGUFTransformer {
            config,
            token_embedding: vec![0.1; 128],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 8],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; 128],
            lm_head_bias: None,
        };

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!(apr.layers[0].attn_norm_bias.is_none());
        assert!(apr.layers[0].qkv_bias.is_none());
        assert!(apr.layers[0].ffn_gate_weight.is_none());
        assert!(apr.output_norm_bias.is_none());
        assert!(apr.lm_head_bias.is_none());
    }

    #[test]
    fn test_deep_ccov_from_gguf_transformer_with_gate_weights() {
        // Test conversion when ffn_gate_weight is present (SwiGLU FFN)
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

        let config = GGUFConfig {
            architecture: "swiglu".to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 16,
            intermediate_dim: 16,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 192],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 64],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.02; 128]), // With gate weights
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 128],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 8]),
            ffn_norm_bias: None,
        };

        let gguf = GGUFTransformer {
            config,
            token_embedding: vec![0.1; 128],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 8],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; 128],
            lm_head_bias: None,
        };

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!(apr.layers[0].ffn_gate_weight.is_some());
        assert_eq!(
            apr.layers[0]
                .ffn_gate_weight
                .as_ref()
                .expect("index out of bounds")
                .len(),
            128
        );
        assert!(apr.layers[0].ffn_norm_weight.is_some());
    }

    #[test]
    fn test_deep_ccov_to_apr_bytes_metadata_alignment() {
        // Verify metadata is padded to 64-byte boundary
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Parse header to get offsets
        let metadata_offset =
            u64::from_le_bytes(bytes[12..20].try_into().expect("index out of bounds")) as usize;
        let tensor_index_offset =
            u64::from_le_bytes(bytes[24..32].try_into().expect("index out of bounds")) as usize;

        // Metadata should start at HEADER_SIZE (64)
        assert_eq!(metadata_offset, HEADER_SIZE);

        // Tensor index offset should be aligned to 64 bytes from metadata start
        let metadata_region_size = tensor_index_offset - metadata_offset;
        assert_eq!(
            metadata_region_size % ALIGNMENT,
            0,
            "Metadata region should be aligned to {} bytes",
            ALIGNMENT
        );
    }

    #[test]
    fn test_deep_ccov_q4k_stats_debug_format() {
        let stats = Q4KConversionStats {
            tensor_count: 123,
            q4k_tensor_count: 99,
            total_bytes: 999_888,
            architecture: "test-arch".to_string(),
            num_layers: 12,
            hidden_size: 2048,
        };

        let debug_str = format!("{:?}", stats);

        // Verify all fields appear in debug output
        assert!(debug_str.contains("tensor_count: 123"));
        assert!(debug_str.contains("q4k_tensor_count: 99"));
        assert!(debug_str.contains("total_bytes: 999888"));
        assert!(debug_str.contains("test-arch"));
        assert!(debug_str.contains("num_layers: 12"));
        assert!(debug_str.contains("hidden_size: 2048"));
    }

    #[test]
    fn test_deep_ccov_conversion_stats_debug_format() {
        let stats = ConversionStats {
            total_parameters: 123_456_789,
            memory_bytes_f32: 493_827_156,
            num_layers: 24,
            hidden_dim: 4096,
            vocab_size: 128000,
            architecture: "llama-3-test".to_string(),
        };

        let debug_str = format!("{:?}", stats);

        // Verify all fields appear in debug output
        assert!(debug_str.contains("total_parameters: 123456789"));
        assert!(debug_str.contains("memory_bytes_f32: 493827156"));
        assert!(debug_str.contains("num_layers: 24"));
        assert!(debug_str.contains("hidden_dim: 4096"));
        assert!(debug_str.contains("vocab_size: 128000"));
        assert!(debug_str.contains("llama-3-test"));
    }

    #[test]
    fn test_deep_ccov_raw_tensor_unknown_dtype() {
        // Test with an unknown dtype that would use default F32 size calculation
        let tensor = RawTensor {
            name: "unknown_type".to_string(),
            data: vec![0u8; 40], // 10 F32 elements
            shape: vec![10],
            dtype: 255, // Unknown type
        };
        assert_eq!(tensor.dtype, 255);
    }

    #[test]
    fn test_deep_ccov_apr_header_offsets_consistency() {
        // Verify header offsets are consistent after serialization
        let apr = create_test_apr_transformer(8, 2, 50, 16);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        let metadata_offset =
            u64::from_le_bytes(bytes[12..20].try_into().expect("index out of bounds"));
        let metadata_size =
            u32::from_le_bytes(bytes[20..24].try_into().expect("index out of bounds"));
        let tensor_index_offset =
            u64::from_le_bytes(bytes[24..32].try_into().expect("index out of bounds"));
        let data_offset =
            u64::from_le_bytes(bytes[32..40].try_into().expect("index out of bounds"));

        // Metadata should start at header end
        assert_eq!(metadata_offset, HEADER_SIZE as u64);

        // Tensor index should come after metadata (with alignment)
        assert!(tensor_index_offset >= metadata_offset + metadata_size as u64);

        // Data should come after tensor index
        assert!(data_offset >= tensor_index_offset);

        // Total file size should be at least data_offset
        assert!(bytes.len() as u64 >= data_offset);
    }

    #[test]
    fn test_deep_ccov_stats_with_single_param() {
        // Edge case: single parameter model
        let stats = ConversionStats {
            total_parameters: 1,
            memory_bytes_f32: 4,
            num_layers: 1,
            hidden_dim: 1,
            vocab_size: 1,
            architecture: "minimal".to_string(),
        };

        assert!((stats.parameters_m() - 0.000001).abs() < 0.0000001);
        assert!((stats.parameters_b() - 0.000000001).abs() < 0.0000000001);
        assert!(stats.memory_mb() > 0.0);
        assert!(stats.memory_gb() > 0.0);
    }

    #[test]
    fn test_deep_ccov_from_apr_bytes_empty_tensor_index() {
        // Create APR with empty tensor index (no tensors)
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&68u64.to_le_bytes()); // data offset

        bytes[64..66].copy_from_slice(b"{}");
        bytes[66..68].copy_from_slice(b"[]"); // Empty array

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(
            result.is_err(),
            "Should fail with no weights tensor: {:?}",
            result
        );
    }
}
