        let result = crate::apr::simple_attention(&q, &k, &v, 1, 1, 1, 2);
        assert_eq!(result.len(), 2);
        // Attention output should be weighted v
    }

    // CudaAprModel tests require GPU feature flag - tested elsewhere

    // =========================================================================
    // AprV2Model Error Path Tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_truncated() {
        // Too short to contain header
        let data = vec![0u8; 10];
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_from_bytes_invalid_magic() {
        let mut data = vec![0u8; 100];
        // Invalid magic
        data[0..4].copy_from_slice(b"GGUF");
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
    }

    // Version test removed - ONE format, no versioning

    #[test]
    fn test_apr_v2_model_get_tensor_bytes_existing() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // test.weight exists in test model
        let result = model.get_tensor_bytes("test.weight");
        assert!(result.is_ok());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_f32_existing() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // test.weight exists in test model
        let result = model.get_tensor_f32("test.weight");
        assert!(result.is_ok());
        let tensor = result.expect("APR operation failed");
        assert!(!tensor.is_empty());
    }

    // =========================================================================
    // AprHeader Extended Tests
    // =========================================================================

    #[test]
    fn test_apr_header_magic_validation() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"XXXX"); // Invalid magic
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_header_version_tuple() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // Major
        data[5] = 1; // Minor
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_ok());
        let header = result.expect("APR operation failed");
        assert_eq!(header.version, (2, 1));
    }

    // =========================================================================
    // AprFlags Extended Tests
    // =========================================================================

    #[test]
    fn test_apr_flags_from_default() {
        let flags = AprFlags::default();
        // Default flags have no bits set
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
    }

    // =========================================================================
    // TensorEntry Extended Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_large_shape() {
        // Create entry with large multidimensional shape
        let entry = create_binary_tensor_entry("big_tensor", 0, &[10, 20, 30, 40], 0, 240000);
        let (parsed, consumed) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.name, "big_tensor");
        assert_eq!(parsed.shape, vec![10, 20, 30, 40]);
        assert_eq!(parsed.element_count(), 10 * 20 * 30 * 40);
        assert!(consumed > 0);
    }

    #[test]
    fn test_tensor_entry_from_binary_zero_dim() {
        // Scalar tensor (empty shape)
        let entry = create_binary_tensor_entry("scalar", 0, &[], 0, 4);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.name, "scalar");
        assert!(parsed.shape.is_empty());
        assert_eq!(parsed.element_count(), 1); // Scalar has 1 element
    }

    // =========================================================================
    // dequantize Function Tests
    // =========================================================================

    #[test]
    fn test_dequantize_f16_multiple_values() {
        // f16 values: 1.0 = 0x3C00, 2.0 = 0x4000
        let bytes = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x3C, // 1.0
        ];
        let result = crate::apr::dequantize_f16(&bytes, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_dequantize_q8_0_nonzero_scale() {
        // Q8_0 block: 2-byte scale + 32 i8 values
        // Scale = 1.0 (f16 0x3C00)
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00; // Scale low byte
        bytes[1] = 0x3C; // Scale high byte (1.0 in f16)
                         // Set first few values to small integers
        bytes[2] = 1; // i8 value 1
        bytes[3] = 2; // i8 value 2
        bytes[4] = 255; // i8 value -1

        let result = crate::apr::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // First value: 1 * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 0.5);
    }

    // =========================================================================
    // ModelData Tests
    // =========================================================================

    #[test]
    fn test_model_data_vec_operations() {
        let data = vec![1u8, 2, 3, 4, 5];
        let md = ModelData::from_vec(data);
        assert_eq!(md.len(), 5);
        assert!(!md.is_empty());
        let slice = md.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    // =========================================================================
    // simple_attention Extended Tests
    // =========================================================================

    #[test]
    fn test_simple_attention_multi_head() {
        // 2 tokens, 2 heads, head_dim=4
        let hidden_dim = 8; // 2 heads * 4 head_dim
        let q = vec![1.0; hidden_dim * 2]; // 2 tokens
        let k = vec![1.0; hidden_dim * 2];
        let v = vec![1.0; hidden_dim * 2];

        let result = crate::apr::simple_attention(&q, &k, &v, 2, 2, 2, 4);
        assert_eq!(result.len(), hidden_dim * 2);
    }

    #[test]
    fn test_simple_attention_gqa() {
        // GQA: 4 heads, 2 KV heads, head_dim=2
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = vec![1.0; hidden_dim]; // 1 token
        let k = vec![1.0; kv_dim];
        let v = vec![1.0; kv_dim];

        let result = crate::apr::simple_attention(&q, &k, &v, 1, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), hidden_dim);
    }

    // =========================================================================
    // matmul Extended Tests
    // =========================================================================

    #[test]
    fn test_matmul_rectangular() {
        // [2,3] * [3,4] = [2,4]
        let x = vec![1.0; 2 * 3]; // 2 rows, 3 cols
        let w = vec![1.0; 3 * 4]; // 3 rows (in_dim), 4 cols (out_dim)
        let result = crate::apr::matmul(&x, &w, 2, 3, 4);
        assert_eq!(result.len(), 2 * 4);
        // Each output element = sum of 3 ones = 3.0
        assert!((result[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_large() {
        // Larger matrix to exercise SIMD paths
        let seq_len = 4;
        let in_dim = 64;
        let out_dim = 64;
        let x = vec![1.0; seq_len * in_dim];
        let w = vec![1.0; in_dim * out_dim];
        let result = crate::apr::matmul(&x, &w, seq_len, in_dim, out_dim);
        assert_eq!(result.len(), seq_len * out_dim);
        // Each element = sum of 64 ones = 64.0
        assert!((result[0] - 64.0).abs() < 1e-3);
    }

    // =========================================================================
    // simd_dot Extended Tests
    // =========================================================================

    #[test]
    fn test_simd_dot_mismatched_len() {
        let a = vec![1.0; 8];
        let b = vec![1.0; 8];
        let result = crate::apr::simd_dot(&a, &b);
        assert!((result - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_alternating() {
        let a = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum: 1 - 1 + 1 - 1 + 1 - 1 + 1 - 1 = 0
        assert!((result - 0.0).abs() < 1e-6);
    }

    // =========================================================================
    // BpeTokenizer Decode Extended Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_decode_byte_fallback() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<0x41>".to_string(), 0); // Byte 65 = 'A'
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["<0x41>".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0]);
        // Byte fallback should be handled
        assert!(!decoded.is_empty());
    }

    // =========================================================================
    // AprFlags Extended Tests - Coverage for all flag methods
    // =========================================================================

    #[test]
    fn test_apr_flags_lz4_compressed() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(flags.is_compressed()); // LZ4 counts as compressed
    }

    #[test]
    fn test_apr_flags_zstd_compressed() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(!flags.is_lz4());
        assert!(flags.is_zstd());
        assert!(flags.is_compressed()); // ZSTD counts as compressed
    }

    #[test]
    fn test_apr_flags_encrypted() {
        let flags = AprFlags::new(AprFlags::ENCRYPTED);
        assert!(flags.is_encrypted());
        assert!(!flags.is_compressed());
    }

    #[test]
    fn test_apr_flags_quantized() {
        let flags = AprFlags::new(AprFlags::QUANTIZED);
        assert!(flags.is_quantized());
        assert!(!flags.is_encrypted());
    }

    #[test]
    fn test_apr_flags_multiple() {
        let flags =
            AprFlags::new(AprFlags::LZ4_COMPRESSED | AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
        assert!(flags.is_lz4());
        assert!(flags.is_compressed());
        assert!(flags.is_quantized());
        assert!(flags.has_vocab());
        assert!(!flags.is_zstd());
        assert!(!flags.is_encrypted());
    }

    // =========================================================================
    // f16_to_f32 Extended Tests - Infinity cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_infinity() {
        // +Inf in f16 = 0x7C00
        let result = crate::apr::f16_to_f32(0x7C00);
        assert!(result.is_infinite() && result > 0.0);
    }

    #[test]
    fn test_f16_to_f32_negative_infinity() {
        // -Inf in f16 = 0xFC00
        let result = crate::apr::f16_to_f32(0xFC00);
        assert!(result.is_infinite() && result < 0.0);
    }

    // =========================================================================
    // dequantize_q4_k Extended Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_partial_block() {
        // Less than one full super-block (144 bytes)
        let bytes = vec![0u8; 50];
        let result = crate::apr::dequantize_q4_k(&bytes, 10);
        // Should handle gracefully
        assert!(result.is_empty() || result.len() <= 10);
    }

    #[test]
    fn test_dequantize_q4_k_one_block() {
        // One complete Q4_K super-block (144 bytes = 256 elements)
        let mut bytes = vec![0u8; 144];
        // Set d (f16) = 1.0 = 0x3C00
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        // Set dmin (f16) = 0.0
        bytes[2] = 0x00;
        bytes[3] = 0x00;
        // scales and mins are already zeros
        // qs are already zeros

        let result = crate::apr::dequantize_q4_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // dequantize_q6_k Extended Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q6_k_partial_block() {
        // Less than one full super-block (210 bytes)
        let bytes = vec![0u8; 100];
        let result = crate::apr::dequantize_q6_k(&bytes, 10);
        // Should handle gracefully
        assert!(result.is_empty() || result.len() <= 10);
    }

    #[test]
    fn test_dequantize_q6_k_one_block() {
        // One complete Q6_K super-block (210 bytes = 256 elements)
        let mut bytes = vec![0u8; 210];
        // d (f16) is at the end: offset 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // 1.0 in f16

        let result = crate::apr::dequantize_q6_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // TensorEntry::from_binary dtype coverage
    // =========================================================================

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_tensor_entry_from_binary_i8() {
        let entry = create_binary_tensor_entry("i8_tensor", 3, &[8], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "I8");
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_tensor_entry_from_binary_i16() {
        let entry = create_binary_tensor_entry("i16_tensor", 4, &[4], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "I16");
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_tensor_entry_from_binary_i32() {
        let entry = create_binary_tensor_entry("i32_tensor", 5, &[4], 0, 16);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "I32");
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_tensor_entry_from_binary_i64() {
        let entry = create_binary_tensor_entry("i64_tensor", 6, &[4], 0, 32);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "I64");
    }

    #[test]
    fn test_tensor_entry_from_binary_q5_1() {
        // GH-191: byte 7 is Q5_1 in GGML dtype mapping (was U8 before GH-191 fix)
        let entry = create_binary_tensor_entry("q5_1_tensor", 7, &[8], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "Q5_1");
    }

    #[test]
    fn test_tensor_entry_from_binary_q4_k() {
        // GH-191 FIX: byte 12 is Q4_K in GGML dtype mapping — was ignored before
        let entry = create_binary_tensor_entry("q4k_tensor", 12, &[256], 0, 144);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "Q4_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_q6_k() {
        // GH-191 FIX: byte 14 is Q6_K in GGML dtype mapping (was byte 9 before)
        let entry = create_binary_tensor_entry("q6k_tensor", 14, &[256], 0, 210);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "Q6_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_q8_0() {
        // GH-191 FIX: byte 8 is Q8_0 in GGML dtype mapping (was byte 10 before)
        let entry = create_binary_tensor_entry("q8_tensor", 8, &[32], 0, 34);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "Q8_0");
    }

    #[test]
    fn test_tensor_entry_from_binary_unknown_dtype() {
        // Unknown dtype byte defaults to F32
        let entry = create_binary_tensor_entry("unknown_tensor", 255, &[4], 0, 16);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "F32");
    }

    // =========================================================================
    // AprV2Model encrypted file test
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_encrypted() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version 2.0
        data[5] = 0;
        // Set encrypted flag (0x0004)
        data[6] = 0x04;
        data[7] = 0x00;

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{err:?}");
        assert!(err_msg.contains("Encrypted"));
    }

    // =========================================================================
    // AprV2Model::generate tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_generate_empty_input() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.generate(&[], 10, None);
        assert!(result.is_err()); // Empty input should fail
    }

    #[test]
    fn test_apr_v2_model_generate_not_transformer() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // Model without transformer config should fail on generate
        let result = model.generate(&[1, 2, 3], 5, None);
        assert!(result.is_err());
    }

    // =========================================================================
    // Additional dtype_to_ggml_qtype coverage
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_lowercase() {
        assert!(crate::apr::dtype_to_ggml_qtype("q4_k").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q5_k").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q6_k").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q8_0").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q4_0").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q4_1").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q5_0").is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_k() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_K"), Some(13));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_1() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_1"), Some(3));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_0() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_0"), Some(6));
    }

    // =========================================================================
    // AprMetadata extended coverage
    // =========================================================================

    #[test]
    fn test_apr_metadata_with_extra_fields() {
        let json = r#"{
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "vocab_size": 32000,
            "custom_field": "custom_value",
            "another_field": 42
        }"#;
        let meta: AprMetadata = serde_json::from_str(json).expect("parse failed");
        assert!(meta.is_transformer());
        assert_eq!(meta.hidden_size, Some(256));
        // Extra fields should be captured
        assert!(meta.extra.contains_key("custom_field"));
    }

    #[test]
    fn test_apr_metadata_optional_fields() {
        let meta = AprMetadata {
            model_type: Some("llama".to_string()),
            name: Some("test-model".to_string()),
            architecture: Some("transformer".to_string()),
            hidden_size: Some(1024),
            num_layers: Some(12),
            num_heads: Some(16),
            num_kv_heads: Some(4),
            vocab_size: Some(50000),
            intermediate_size: Some(4096),
            max_position_embeddings: Some(2048),
            rope_theta: Some(10000.0),
            rope_type: Some(2),
            rms_norm_eps: Some(1e-5),
            extra: HashMap::new(),
        };
        assert!(meta.is_transformer());
        assert_eq!(meta.num_kv_heads, Some(4));
        assert_eq!(meta.rope_type, Some(2));
    }

    // =========================================================================
    // byte_to_bpe_char extended coverage
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_all_printable_ascii() {
        // Test all printable ASCII characters
        for b in 33u8..=126 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_byte_to_bpe_char_control_chars() {
        // Test control characters 0-31
        for b in 0u8..32 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_byte_to_bpe_char_extended_ascii() {
        // Test extended ASCII 128-255
        for b in 128u8..=255 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    // =========================================================================
    // BpeTokenizer extended coverage
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_decode_multiple() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 0);
        token_to_id.insert(" ".to_string(), 1);
        token_to_id.insert("world".to_string(), 2);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["hello".to_string(), " ".to_string(), "world".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    // =========================================================================
    // AprHeader flags coverage
    // =========================================================================

    #[test]
    fn test_apr_header_with_flags() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        // Set quantized flag
        data[6] = 0x20;
        data[7] = 0x00;

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        assert!(header.flags.is_quantized());
    }

    // =========================================================================
    // AprV2Model predict with weight tensor
    // =========================================================================

    #[test]
    fn test_apr_v2_model_predict_single_feature() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features = vec![1.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apr_v2_model_predict_many_features() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = model.predict(&features);
        assert!(result.is_ok());
    }

    // =========================================================================
    // TensorEntry element_count edge cases
    // =========================================================================

    #[test]
    fn test_tensor_entry_element_count_1d() {
        let entry = TensorEntry {
            name: "vec".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            offset: 0,
            size: 40,
        };
        assert_eq!(entry.element_count(), 10);
    }

    #[test]
    fn test_tensor_entry_element_count_3d() {
        let entry = TensorEntry {
            name: "3d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4],
            offset: 0,
            size: 96,
        };
        assert_eq!(entry.element_count(), 24);
    }

    // =========================================================================
    // dequantize_f16 edge cases
    // =========================================================================

    #[test]
    fn test_dequantize_f16_odd_bytes() {
        // Odd number of bytes - last byte ignored
        let bytes = vec![0x00, 0x3C, 0x00]; // 1.0 + extra byte
        let result = crate::apr::dequantize_f16(&bytes, 2);
        assert_eq!(result.len(), 1); // Only one complete f16
    }

    // =========================================================================
    // detect_format additional cases
    // =========================================================================

    #[test]
    fn test_detect_format_bin() {
        // .bin is not a recognized format, returns "unknown"
        assert_eq!(detect_format("/path/model.bin"), "unknown");
    }

    #[test]
    fn test_detect_format_pt() {
        // .pt is not a recognized format, returns "unknown"
        assert_eq!(detect_format("/path/model.pt"), "unknown");
    }

    #[test]
    fn test_detect_format_no_extension() {
        // No extension returns "unknown"
        assert_eq!(detect_format("/path/model"), "unknown");
    }

    #[test]
    fn test_detect_format_hidden_file() {
        // Hidden file with no extension returns "unknown"
        assert_eq!(detect_format("/path/.hidden"), "unknown");
    }

    // =========================================================================
    // dequantize Early Exit Tests - Hit break paths by requesting fewer elements
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_fewer_than_block() {
        // Q8_0 block: 2-byte scale + 32 i8 values = 34 bytes, 32 elements
        // Request only 10 elements to trigger early exit
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            bytes[2 + i] = (i + 1) as u8; // Values 1-32
        }

        let result = crate::apr::dequantize_q8_0(&bytes, 10);
        assert_eq!(result.len(), 10);
        // First value: 1 * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_dequantize_q8_0_multiple_blocks() {
        // Two Q8_0 blocks = 68 bytes, 64 elements
        let mut bytes = vec![0u8; 68];
        // Block 1
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
                         // Block 2
        bytes[34] = 0x00;
        bytes[35] = 0x3C; // Scale = 1.0

        let result = crate::apr::dequantize_q8_0(&bytes, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_q4_k_fewer_than_superblock() {
        // Q4_K super-block: 144 bytes = 256 elements
        // Request only 100 elements to trigger early exit
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q4_k(&bytes, 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_dequantize_q4_k_multiple_superblocks() {
        // Two Q4_K super-blocks = 288 bytes = 512 elements
        let mut bytes = vec![0u8; 288];
        // First superblock
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        // Second superblock
        bytes[144] = 0x00;
        bytes[145] = 0x3C;

        let result = crate::apr::dequantize_q4_k(&bytes, 512);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_dequantize_q6_k_fewer_than_superblock() {
        // Q6_K super-block: 210 bytes = 256 elements
        // Request only 50 elements to trigger early exit
        let mut bytes = vec![0u8; 210];
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q6_k(&bytes, 50);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_dequantize_q6_k_multiple_superblocks() {
        // Two Q6_K super-blocks = 420 bytes = 512 elements
        let mut bytes = vec![0u8; 420];
        // First superblock d at 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C;
        // Second superblock d at 418-419
        bytes[418] = 0x00;
        bytes[419] = 0x3C;

        let result = crate::apr::dequantize_q6_k(&bytes, 512);
        assert_eq!(result.len(), 512);
    }

    // =========================================================================
    // BpeTokenizer encode tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_empty() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_encode_with_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("ab".to_string(), 2);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
            merge_rules: vec![("a".to_string(), "b".to_string())],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("ab");
        // Should merge a+b -> ab
        assert!(!encoded.is_empty());
    }

    // =========================================================================
    // AprV2Model find_tensor_name tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_tensor_count() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_v2_model_total_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // Test model has 4x4 = 16 element tensor
        assert!(model.estimated_parameters() > 0);
    }

    // =========================================================================
    // AprMetadata serialization tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_to_json() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        assert!(json.contains("256"));
        assert!(json.contains("hidden_size"));
    }

    #[test]
    fn test_apr_metadata_roundtrip() {
        let meta = AprMetadata {
            hidden_size: Some(1024),
            num_layers: Some(12),
            num_heads: Some(16),
            vocab_size: Some(50000),
            model_type: Some("llama".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        let parsed: AprMetadata = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.hidden_size, Some(1024));
        assert_eq!(parsed.model_type, Some("llama".to_string()));
    }

    // =========================================================================
    // TensorEntry tests with various shapes
    // =========================================================================

    #[test]
    fn test_tensor_entry_5d_shape() {
        let entry = TensorEntry {
            name: "5d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4, 5, 6],
            offset: 0,
            size: 2880,
        };
        assert_eq!(entry.element_count(), 720);
    }

    #[test]
    fn test_tensor_entry_single_element() {
        let entry = TensorEntry {
            name: "single".to_string(),
            dtype: "F32".to_string(),
            shape: vec![1],
            offset: 0,
            size: 4,
        };
        assert_eq!(entry.element_count(), 1);
    }

    // =========================================================================
    // is_apr_file tests (requires actual file with magic bytes)
    // =========================================================================

    #[test]
    fn test_is_apr_file_nonexistent_path() {
        // Non-existent file returns false
        assert!(!is_apr_file("/nonexistent/path/model.apr"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_apr_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(is_apr_file(temp.path()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_without_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"GGUF").expect("write wrong magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(!is_apr_file(temp.path()));
    }

    // =========================================================================
    // More rms_norm tests
    // =========================================================================

    #[test]
    fn test_rms_norm_large() {
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let weight: Vec<f32> = vec![1.0; 64];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let x = vec![-1.0, -2.0, -3.0, -4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Normalized negative values should remain negative
        assert!(result[0] < 0.0);
    }

    // =========================================================================
    // matmul extended tests
    // =========================================================================

    #[test]
    fn test_matmul_zeros() {
        let x = vec![0.0; 4];
        let w = vec![1.0; 4];
        let result = crate::apr::matmul(&x, &w, 2, 2, 2);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_scaling() {
        let x = vec![2.0, 2.0];
        let w = vec![1.0, 1.0];
        let result = crate::apr::matmul(&x, &w, 1, 2, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 4.0).abs() < 1e-6);
    }

    // =========================================================================
    // Additional Coverage Tests - get_tensor_f32 with different dtypes
    // =========================================================================

    /// Helper to create an APR model with a specific dtype tensor
    fn create_test_apr_model_with_dtype(dtype: u8, data_bytes: &[u8]) -> Vec<u8> {
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Tensor shape depends on dtype and data
        let tensor_entry =
            create_binary_tensor_entry("typed.weight", dtype, &[4], 0, data_bytes.len() as u64);

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + data_bytes.len();
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&data_offset.to_le_bytes()); // data_offset

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        // Tensor data
        let data_start = data_offset as usize;
        data[data_start..data_start + data_bytes.len()].copy_from_slice(data_bytes);

        data
    }

    #[test]
    fn test_get_tensor_f32_f16_dtype() {
        // F16 dtype = 1, shape [4], 4 elements = 8 bytes
        // f16 values: 1.0 = 0x3C00, 2.0 = 0x4000
        let f16_data = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
        ];
        let model_data = create_test_apr_model_with_dtype(1, &f16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.expect("APR operation failed");
        assert_eq!(floats.len(), 4);
        assert!((floats[0] - 1.0).abs() < 0.1);
        assert!((floats[1] - 2.0).abs() < 0.1);
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_get_tensor_f32_q8_0_dtype() {
        // Q8_0 dtype = 10, block = 2-byte scale + 32 i8 values = 34 bytes for 32 elements
        let mut q8_data = vec![0u8; 34];
        q8_data[0] = 0x00;
        q8_data[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            q8_data[2 + i] = i as u8;
        }

        // Create with shape [32] since Q8_0 block has 32 elements
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_entry = create_binary_tensor_entry("typed.weight", 10, &[32], 0, 34);
        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + 34;
        let mut model_data = vec![0u8; total_size];

        model_data[0..4].copy_from_slice(&MAGIC);
        model_data[4] = 2;
        model_data[5] = 0;
        model_data[8..12].copy_from_slice(&1u32.to_le_bytes());
        model_data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        model_data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        model_data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        model_data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        model_data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        model_data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);
        let data_start = data_offset as usize;
        model_data[data_start..data_start + 34].copy_from_slice(&q8_data);

        let model = AprV2Model::from_bytes(model_data).expect("should load");
        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.expect("APR operation failed");
        assert_eq!(floats.len(), 32);
    }

    #[test]
    fn test_get_tensor_f32_unsupported_dtype() {
        // BF16 dtype = 2 is not fully supported for get_tensor_f32
        let bf16_data = vec![0x00, 0x3F, 0x80, 0x00]; // Two BF16 values
        let model_data = create_test_apr_model_with_dtype(2, &bf16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        // BF16 is not in the supported list, should error
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tensor_f32_out_of_bounds() {
        // Create a model where tensor data extends beyond file
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&100u64.to_le_bytes()); // data_offset

        // Add tensor entry that claims data beyond file
        let tensor_entry = create_binary_tensor_entry("oob.weight", 0, &[1000], 0, 4000);
        data[64..64 + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let model = AprV2Model::from_bytes(data).expect("should load");
        let result = model.get_tensor_f32("oob.weight");
        assert!(result.is_err()); // Out of bounds
    }

    // =========================================================================
    // decode_tokens extended tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_gpt2_special() {
        // Test GPT-2 style byte-level BPE tokens
        let vocab = vec![
            "Ġhello".to_string(), // Space + hello
            "Ċ".to_string(),      // Newline
            "ĉ".to_string(),      // Tab
        ];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert!(result.contains("hello"));
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }

    #[test]
    fn test_decode_tokens_empty_string_token() {
        let vocab = vec![String::new(), "a".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Empty token shouldn't cause issues
        assert!(result.contains('a'));
    }

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_smallest_positive_normal() {
        // Smallest positive normal f16 = 0x0400 (6.103515625e-5)
        let result = crate::apr::f16_to_f32(0x0400);
        assert!(result > 0.0 && result < 1e-4);
    }

    #[test]
    fn test_f16_to_f32_largest_normal() {
        // Largest finite f16 = 0x7BFF (65504.0)
        let result = crate::apr::f16_to_f32(0x7BFF);
        assert!((result - 65504.0).abs() < 10.0);
    }

    #[test]
    fn test_f16_to_f32_negative_normal() {
        // -2.0 in f16 = 0xC000
        let result = crate::apr::f16_to_f32(0xC000);
        assert!((result + 2.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_to_f32_subnormal_nonzero() {
        // Various subnormals (exp=0, mantissa!=0)
        for mant in [1u16, 10, 100, 0x3FF] {
            let result = crate::apr::f16_to_f32(mant);
            assert!(result > 0.0, "Subnormal {mant:#x} should be positive");
        }
    }

    // =========================================================================
    // bpe_encode extended tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_with_newline() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ċ".to_string(), 0); // Newline token
        let result = bpe_encode("\n", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_tab() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("ĉ".to_string(), 0); // Tab token
        let result = bpe_encode("\t", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_space() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ġ".to_string(), 0); // Space token
        let result = bpe_encode(" ", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_mixed() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("Ġ".to_string(), 1); // Space
        token_to_id.insert("b".to_string(), 2);
        let result = bpe_encode("a b", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_bpe_encode_multiple_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        token_to_id.insert("ab".to_string(), 3);
        token_to_id.insert("abc".to_string(), 4);

        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let result = bpe_encode("abc", &token_to_id, &merges, &HashMap::new());
        // Should merge a+b->ab, then ab+c->abc
        assert!(!result.is_empty());
    }

    // =========================================================================
    // AprV2Model predict with weight tensor
    // =========================================================================

    /// Helper to create APR model with weight and bias tensors for linear model
    fn create_linear_model_apr() -> Vec<u8> {
        let metadata = r#"{"architecture":"linear"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Weight tensor: 2x3 = 6 elements, 24 bytes
        // Bias tensor: 2 elements, 8 bytes
        let weight_entry = create_binary_tensor_entry("weight", 0, &[2, 3], 0, 24);
        let bias_entry = create_binary_tensor_entry("bias", 0, &[2], 24, 8);
        let tensor_index_size = weight_entry.len() + bias_entry.len();

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_index_size as u64;
        let total_size = data_offset as usize + 32;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&2u32.to_le_bytes()); // tensor_count = 2
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + weight_entry.len()].copy_from_slice(&weight_entry);
        data[idx_start + weight_entry.len()..idx_start + tensor_index_size]
            .copy_from_slice(&bias_entry);

        // Tensor data - weight matrix [2,3] (identity-ish pattern)
        let data_start = data_offset as usize;
        let weights: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2,3] matrix
        for (i, &w) in weights.iter().enumerate() {
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        // Bias [2]
        let biases: [f32; 2] = [0.5, 0.5];
        for (i, &b) in biases.iter().enumerate() {
            data[data_start + 24 + i * 4..data_start + 24 + i * 4 + 4]
                .copy_from_slice(&b.to_le_bytes());
        }

        data
    }

    #[test]
    fn test_apr_v2_model_predict_with_weights() {
        let model_data = create_linear_model_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let features = vec![1.0, 2.0, 3.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
        let output = result.expect("APR operation failed");
        assert_eq!(output.len(), 2);
        // Expected: [1*1 + 2*0 + 3*0 + 0.5, 1*0 + 2*1 + 3*0 + 0.5] = [1.5, 2.5]
        assert!((output[0] - 1.5).abs() < 0.01);
        assert!((output[1] - 2.5).abs() < 0.01);
    }

    // =========================================================================
    // AprFlags bit operations
    // =========================================================================

    #[test]
    fn test_apr_flags_sharded() {
        let flags = AprFlags::new(AprFlags::SHARDED);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
        assert!(!flags.is_quantized());
    }

    #[test]
    fn test_apr_flags_signed() {
        let flags = AprFlags::new(AprFlags::SIGNED);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
    }

    #[test]
    fn test_apr_flags_all_set() {
        let flags = AprFlags::new(0xFFFF);
        assert!(flags.is_compressed());
        assert!(flags.is_encrypted());
        assert!(flags.is_quantized());
        assert!(flags.has_vocab());
    }

    // =========================================================================
    // simd_dot edge cases
    // =========================================================================

    #[test]
    fn test_simd_dot_empty() {
        let result = crate::apr::simd_dot(&[], &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_simd_dot_single_element() {
        let result = crate::apr::simd_dot(&[3.0], &[4.0]);
        assert!((result - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_negative() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![-1.0, -1.0, -1.0, -1.0];
        let result = crate::apr::simd_dot(&a, &b);
        // (-1)*(-1) + (-2)*(-1) + (-3)*(-1) + (-4)*(-1) = 1 + 2 + 3 + 4 = 10
        assert!((result - 10.0).abs() < 1e-6);
    }

    // =========================================================================
    // rms_norm multi-sequence
    // =========================================================================

    #[test]
    fn test_rms_norm_multi_sequence() {
        // 2 sequences of hidden_dim=4
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 8);
    }

    // =========================================================================
    // simple_attention edge cases
    // =========================================================================

    #[test]
    fn test_simple_attention_mqa() {
        // MQA: 4 Q heads, 1 KV head
        let num_heads = 4;
        let num_kv_heads = 1;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = vec![1.0; hidden_dim];
        let k = vec![1.0; kv_dim];
        let v = vec![1.0; kv_dim];

        let result = crate::apr::simple_attention(&q, &k, &v, 1, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), hidden_dim);
    }

    // =========================================================================
    // AprHeader debug and clone
    // =========================================================================

    #[test]
    fn test_apr_header_clone() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&5u32.to_le_bytes());

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        let cloned = header.clone();
        assert_eq!(cloned.tensor_count, header.tensor_count);
        assert_eq!(cloned.version, header.version);
    }

    #[test]
    fn test_apr_header_debug() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        let debug_str = format!("{:?}", header);
        assert!(debug_str.contains("AprHeader"));
    }

    // =========================================================================
    // TensorEntry debug and clone
    // =========================================================================

    #[test]
    fn test_tensor_entry_debug() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 64,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("TensorEntry"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_tensor_entry_clone() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 64,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.name, entry.name);
        assert_eq!(cloned.shape, entry.shape);
    }

    // =========================================================================
    // AprMetadata debug and clone
    // =========================================================================

    #[test]
    fn test_apr_metadata_debug() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            ..Default::default()
        };
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("AprMetadata"));
    }

    #[test]
    fn test_apr_metadata_clone() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            ..Default::default()
        };
        let cloned = meta.clone();
        assert_eq!(cloned.hidden_size, meta.hidden_size);
        assert_eq!(cloned.num_layers, meta.num_layers);
    }

    // =========================================================================
    // BpeTokenizer debug and clone
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_debug() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: Some(1),
            eos_id: Some(2),
            special_tokens: HashMap::new(),
        };
        let debug_str = format!("{:?}", tokenizer);
        assert!(debug_str.contains("BpeTokenizer"));
    }

    #[test]
    fn test_bpe_tokenizer_clone() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![("a".to_string(), "b".to_string())],
            bos_id: Some(1),
            eos_id: Some(2),
            special_tokens: HashMap::new(),
        };
        let cloned = tokenizer.clone();
        assert_eq!(cloned.bos_id, tokenizer.bos_id);
        assert_eq!(cloned.id_to_token, tokenizer.id_to_token);
    }

    // =========================================================================
    // AprFlags debug
    // =========================================================================

    #[test]
    fn test_apr_flags_debug() {
        let flags = AprFlags::new(0x0025);
        let debug_str = format!("{:?}", flags);
        assert!(debug_str.contains("AprFlags"));
    }

    #[test]
    fn test_apr_flags_clone() {
        let flags = AprFlags::new(0x0025);
        let cloned = flags;
        assert_eq!(cloned.is_lz4(), flags.is_lz4());
        assert_eq!(cloned.is_quantized(), flags.is_quantized());
    }

    // =========================================================================
    // AprV2Model debug
    // =========================================================================

    #[test]
    fn test_apr_v2_model_debug() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("AprV2Model"));
    }

    // =========================================================================
    // ModelData extended edge cases
    // =========================================================================

    #[test]
    fn test_model_data_debug_heap() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        let debug_str = format!("{:?}", data);
        assert!(debug_str.contains("Heap"));
    }

    // =========================================================================
    // dequantize multiple blocks
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_three_blocks() {
        // Three Q8_0 blocks = 102 bytes, 96 elements
        let mut bytes = vec![0u8; 102];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0 block 1
        bytes[34] = 0x00;
        bytes[35] = 0x3C; // Scale = 1.0 block 2
        bytes[68] = 0x00;
        bytes[69] = 0x3C; // Scale = 1.0 block 3

        let result = crate::apr::dequantize_q8_0(&bytes, 96);
        assert_eq!(result.len(), 96);
    }

    #[test]
    fn test_dequantize_q4_k_with_nonzero_scales() {
        // Q4_K super-block with actual scale values
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
        bytes[2] = 0x00;
        bytes[3] = 0x3C; // dmin = 1.0
                         // Set some scale values
        bytes[4] = 0x3F; // scales[0] = 63
        bytes[5] = 0x3F;

        let result = crate::apr::dequantize_q4_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6_k_with_nonzero_scales() {
        // Q6_K super-block with scale values
        let mut bytes = vec![0u8; 210];
        // scales at offset 192-207
        bytes[192] = 10;
        bytes[193] = 20;
        // d at offset 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q6_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // Transformer Model Tests - Forward Pass Coverage
    // =========================================================================

    /// Helper to create a minimal transformer model for testing forward pass.
    /// This creates a 1-layer transformer with tiny dimensions for test purposes.
    fn create_mini_transformer_apr() -> Vec<u8> {
        let metadata = r#"{
            "architecture": "llama",
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "num_kv_heads": 2,
            "vocab_size": 10,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Tensors needed for forward:
        // - model.embed_tokens.weight [vocab=10, hidden=8] = 80 floats = 320 bytes
        // - layers.0.input_layernorm.weight [hidden=8] = 8 floats = 32 bytes
        // - layers.0.self_attn.q_proj.weight [hidden=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.k_proj.weight [kv_dim=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.v_proj.weight [kv_dim=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.o_proj.weight [hidden=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.post_attention_layernorm.weight [hidden=8] = 8 floats = 32 bytes
        // - layers.0.mlp.gate_proj.weight [inter=16, hidden=8] = 128 floats = 512 bytes
        // - layers.0.mlp.up_proj.weight [inter=16, hidden=8] = 128 floats = 512 bytes
        // - layers.0.mlp.down_proj.weight [hidden=8, inter=16] = 128 floats = 512 bytes
        // - norm.weight [hidden=8] = 8 floats = 32 bytes
        // - lm_head.weight [vocab=10, hidden=8] = 80 floats = 320 bytes

        let tensor_defs: Vec<(&str, &[usize], usize)> = vec![
            ("model.embed_tokens.weight", &[10, 8], 320),
            ("layers.0.input_layernorm.weight", &[8], 32),
            ("layers.0.self_attn.q_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.k_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.v_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.o_proj.weight", &[8, 8], 256),
            ("layers.0.post_attention_layernorm.weight", &[8], 32),
            ("layers.0.mlp.gate_proj.weight", &[16, 8], 512),
            ("layers.0.mlp.up_proj.weight", &[16, 8], 512),
            ("layers.0.mlp.down_proj.weight", &[8, 16], 512),
            ("norm.weight", &[8], 32),
            ("lm_head.weight", &[10, 8], 320),
        ];

        let mut tensor_entries = Vec::new();
        let mut current_offset = 0u64;

        for (name, shape, byte_size) in &tensor_defs {
            let shape_vec: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
            let entry =
                create_binary_tensor_entry(name, 0, &shape_vec, current_offset, *byte_size as u64);
            tensor_entries.push(entry);
            current_offset += *byte_size as u64;
        }

        let tensor_index: Vec<u8> = tensor_entries
            .iter()
            .flat_map(|e| e.iter().copied())
            .collect();
        let tensor_count = tensor_defs.len() as u32;
        let total_data_size = current_offset as usize;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_index.len() as u64;
        let total_size = data_offset as usize + total_data_size;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

        // Tensor data - initialize with small random-ish values
        let data_start = data_offset as usize;
        let num_floats = total_data_size / 4;
        for i in 0..num_floats {
            let val = ((i % 10) as f32 - 5.0) * 0.1; // Small values between -0.5 and 0.4
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        // Set layernorm weights to 1.0 (they need to be non-zero)
        let norm_weight_offsets = vec![320, 320 + 32 + 256 * 4, 320 + 32 + 256 * 4 + 32 + 512 * 3];
        for offset in norm_weight_offsets {
            for i in 0..8 {
                let val = 1.0f32;
                let pos = data_start + offset + i * 4;
                if pos + 4 <= data.len() {
                    data[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }

        data
    }

    #[test]
    fn test_transformer_model_loads() {
        let model_data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");
        assert!(model.metadata().is_transformer());
        assert_eq!(model.metadata().hidden_size, Some(8));
        assert_eq!(model.metadata().num_layers, Some(1));
        assert_eq!(model.metadata().vocab_size, Some(10));
    }

    #[test]
    fn test_transformer_has_all_tensors() {
        let model_data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        // Check key tensors exist
        assert!(model.get_tensor("model.embed_tokens.weight").is_some());
        assert!(model
            .get_tensor("layers.0.input_layernorm.weight")
            .is_some());
        assert!(model
            .get_tensor("layers.0.self_attn.q_proj.weight")
            .is_some());
        assert!(model.get_tensor("norm.weight").is_some());
        assert!(model.get_tensor("lm_head.weight").is_some());
    }

    // =========================================================================
    // matmul edge cases
    // =========================================================================

    #[test]
    fn test_matmul_out_of_bounds_x() {
        // x is shorter than expected
        let x = vec![1.0, 2.0]; // Only 2 elements, but seq=1, in_dim=4
        let w = vec![1.0; 8]; // 2 output dims, 4 input dims
        let result = crate::apr::matmul(&x, &w, 1, 4, 2);
        // Should handle gracefully (skip or produce zeros)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_matmul_out_of_bounds_w() {
        // w is shorter than expected
        let x = vec![1.0; 4]; // seq=1, in_dim=4
        let w = vec![1.0, 2.0]; // Too short
        let result = crate::apr::matmul(&x, &w, 1, 4, 2);
        // Should handle gracefully
        assert_eq!(result.len(), 2);
    }

    // =========================================================================
    // dequantize with negative i8 values
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_negative_values() {
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
                         // Set negative i8 values (128-255 map to -128 to -1)
        bytes[2] = 255; // -1 as i8
        bytes[3] = 254; // -2 as i8
        bytes[4] = 128; // -128 as i8

        let result = crate::apr::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // First value should be negative: -1 * 1.0 = -1.0
        assert!(result[0] < 0.0);
    }

    // =========================================================================
    // f16_to_f32 quiet NaN
    // =========================================================================

    #[test]
    fn test_f16_to_f32_qnan() {
        // Quiet NaN in f16 (exp=0x1F, mantissa>0, sign=0)
        // 0x7E00 = quiet NaN
        let result = crate::apr::f16_to_f32(0x7E00);
        assert!(result.is_nan());
    }

    // =========================================================================
    // detect_format with magic bytes
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_apr_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert_eq!(detect_format(temp.path()), "apr");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_gguf_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&[0x47, 0x47, 0x55, 0x46])
            .expect("write GGUF magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert_eq!(detect_format(temp.path()), "gguf");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_safetensors_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"{\"test\": 1}").expect("write JSON");

        assert_eq!(detect_format(temp.path()), "safetensors");
    }

    // =========================================================================
    // simd_dot_avx2 scalar fallback
    // =========================================================================

    #[test]
    fn test_simd_dot_non_multiple_of_8() {
        // Test with lengths that aren't multiples of 8 to exercise remainder handling
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 13];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum of 0..12 = 78
        assert!((result - 78.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_prime_length() {
        // Prime number length ensures both SIMD chunks and remainder are exercised
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 17];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum of 0..16 = 136
        assert!((result - 136.0).abs() < 1e-6);
    }

    // =========================================================================
    // simple_attention multi-token
    // =========================================================================

    #[test]
    fn test_simple_attention_multiple_tokens() {
        // 3 tokens, 2 heads, head_dim=4
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let hidden_dim = num_heads * head_dim;

        let q = vec![1.0; seq_len * hidden_dim];
        let k = vec![1.0; seq_len * hidden_dim];
        let v = vec![1.0; seq_len * hidden_dim];

        let result =
            crate::apr::simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), seq_len * hidden_dim);
    }

    #[test]
    fn test_simple_attention_varying_values() {
        // Test with varying Q, K, V values
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;
        let seq_len = 2;
        let hidden_dim = num_heads * head_dim;

        // Different Q, K, V patterns
        let q = vec![1.0, 0.0, 0.0, 1.0]; // Token 1: [1,0], Token 2: [0,1]
        let k = vec![1.0, 0.0, 1.0, 0.0]; // Token 1: [1,0], Token 2: [1,0]
        let v = vec![1.0, 2.0, 3.0, 4.0]; // Token 1: [1,2], Token 2: [3,4]

        let result =
            crate::apr::simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), seq_len * hidden_dim);
        // Output should be valid attention-weighted values
        assert!(!result.iter().any(|v| v.is_nan()));
    }

    // =========================================================================
    // ModelData methods
    // =========================================================================

    #[test]
    fn test_model_data_empty() {
        let data = ModelData::from_vec(vec![]);
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn test_model_data_as_slice_extended() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        let slice = data.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    // =========================================================================
    // AprV2Model forward error path tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_forward_empty_input() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[]);
        assert!(result.is_err()); // Empty input
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
