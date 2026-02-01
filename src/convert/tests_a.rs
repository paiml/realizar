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
    // PMAT-107: infer_rope_type Tests
    // =========================================================================

    #[test]
    fn test_pmat_107_infer_rope_type_qwen2_is_neox() {
        // Qwen2 architecture should use NEOX style (type 2)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("qwen2", &metadata);
        assert_eq!(result, 2, "Qwen2 should use NEOX style (rope_type=2)");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_llama_is_norm() {
        // LLaMA architecture should use NORM style (type 0)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("llama", &metadata);
        assert_eq!(result, 0, "LLaMA should use NORM style (rope_type=0)");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_phi3_is_neox() {
        // Phi3 architecture should use NEOX style (type 2)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("phi3", &metadata);
        assert_eq!(result, 2, "Phi3 should use NEOX style (rope_type=2)");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_gemma_is_neox() {
        // Gemma architecture should use NEOX style (type 2)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("gemma", &metadata);
        assert_eq!(result, 2, "Gemma should use NEOX style (rope_type=2)");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_scaling_yarn_overrides() {
        // rope.scaling.type=yarn should override architecture inference
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("yarn".to_string()),
        );
        let result = GgufToAprQ4KConverter::infer_rope_type("llama", &metadata);
        assert_eq!(result, 2, "yarn scaling type should override to NEOX style");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_scaling_linear_is_norm() {
        // rope.scaling.type=linear should use NORM style
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "qwen2.rope.scaling.type".to_string(),
            GGUFValue::String("linear".to_string()),
        );
        let result = GgufToAprQ4KConverter::infer_rope_type("qwen2", &metadata);
        assert_eq!(result, 0, "linear scaling type should use NORM style");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_unknown_defaults_to_norm() {
        // Unknown architecture should default to NORM style (type 0)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("unknown_arch", &metadata);
        assert_eq!(result, 0, "Unknown arch should default to NORM style");
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
