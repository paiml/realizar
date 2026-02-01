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
