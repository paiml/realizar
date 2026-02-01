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

    // ==========================================================================
    // PMAT-107: Falsification Test for GQA num_kv_heads Preservation
    // ==========================================================================
    // This test was added after discovering that APR models hang on GPU
    // because num_kv_heads was being stripped during conversion.
    //
    // Five-Whys Root Cause:
    // 1. Why did APR hang? GPU treated GQA (2 kv_heads) as MHA (12 kv_heads)
    // 2. Why wrong kv_heads? metadata.num_kv_heads was None
    // 3. Why None? APR loading returned default() on parse failure
    // 4. Why parse failure? (We need to verify this is NOT the case)
    // 5. Root cause: Silent failure via unwrap_or_default()

    /// FALSIFICATION TEST: Verify num_kv_heads survives APR round-trip
    /// This test MUST catch the bug where GQA models are converted to MHA.
    #[test]
    fn test_falsification_gqa_num_kv_heads_preserved() {
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

        // Create a GQA model: 12 Q heads, 2 KV heads (like Qwen 1.5B)
        let num_heads = 12;
        let num_kv_heads = 2; // GQA: fewer KV heads than Q heads
        let hidden_dim = 64;
        let num_layers = 2;
        let vocab_size = 100;
        let intermediate_dim = 128;

        let config = GGUFConfig {
            architecture: "qwen2".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads, // CRITICAL: This must be preserved!
            vocab_size,
            intermediate_dim,
            context_length: 512,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2, // NEOX style
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

        let gguf = crate::gguf::GGUFTransformer {
            config,
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; hidden_dim * vocab_size],
            lm_head_bias: None,
        };

        // Step 1: Convert GGUF -> APR Transformer
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        // Verify config is preserved in memory
        assert_eq!(
            apr.config.num_heads, num_heads,
            "num_heads not preserved in AprTransformer"
        );
        assert_eq!(
            apr.config.num_kv_heads, num_kv_heads,
            "FALSIFICATION FAILED: num_kv_heads not preserved in AprTransformer config"
        );

        // Step 2: Serialize to APR bytes
        let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("Failed to serialize APR");

        // Step 3: Verify num_kv_heads is in the serialized JSON metadata
        // Find the metadata section and check it contains num_kv_heads
        let metadata_json = String::from_utf8_lossy(&apr_bytes[64..512]);
        assert!(
            metadata_json.contains("\"num_kv_heads\":2"),
            "FALSIFICATION FAILED: num_kv_heads not in serialized APR metadata.\n\
             Metadata: {}",
            &metadata_json[..200.min(metadata_json.len())]
        );

        // Step 4: Deserialize back and verify
        let apr_loaded =
            GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("Failed to load APR from bytes");

        assert_eq!(
            apr_loaded.config.num_heads, num_heads,
            "num_heads not preserved after round-trip"
        );
        assert_eq!(
            apr_loaded.config.num_kv_heads, num_kv_heads,
            "FALSIFICATION FAILED: num_kv_heads corrupted after APR round-trip!\n\
             Expected: {}, Got: {}\n\
             This bug causes GPU inference to hang for GQA models.",
            num_kv_heads, apr_loaded.config.num_kv_heads
        );

        println!(
            "✅ FALSIFICATION TEST PASSED: num_kv_heads={} preserved through APR round-trip",
            num_kv_heads
        );
    }
}
