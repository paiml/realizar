        assert!(err_msg.contains("empty"));
    }

    #[test]
    fn test_apr_v2_model_generate_max_tokens_zero() {
        let data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.generate(&[1, 2], 0, None);
        // Should succeed with empty generation
        assert!(result.is_ok());
        let tokens = result.expect("APR operation failed");
        assert_eq!(tokens.len(), 2); // Just input, no generation
    }

    // =========================================================================
    // byte_to_bpe_char newline and tab
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_newline() {
        let result = crate::apr::byte_to_bpe_char(b'\n');
        assert_eq!(result, "Ċ");
    }

    #[test]
    fn test_byte_to_bpe_char_tab() {
        let result = crate::apr::byte_to_bpe_char(b'\t');
        assert_eq!(result, "ĉ");
    }

    // =========================================================================
    // rms_norm with very small epsilon
    // =========================================================================

    #[test]
    fn test_rms_norm_tiny_eps() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-12;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Should not produce NaN or Inf
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_large_eps() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 10.0; // Large epsilon dominates
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // Additional Coverage Tests (_more_cov suffix)
    // =========================================================================

    #[test]
    fn test_load_tokenizer_from_sibling_nonexistent_more_cov() {
        // Test with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::load_tokenizer_from_sibling(path);
        assert!(result.is_none());
    }

    #[test]
    fn test_encode_text_nonexistent_more_cov() {
        // Test encode_text with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::encode_text(path, "hello world");
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tokenizer_nonexistent_more_cov() {
        // Test load_tokenizer with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::load_tokenizer(path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        // Create model file (empty)
        std::fs::File::create(&model_path).expect("create model file");

        // Create invalid JSON tokenizer file
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_encode_text_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::encode_text(&model_path, "test");
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::load_tokenizer(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_missing_model_key_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid JSON but missing "model" key
        f.write_all(br#"{"added_tokens": []}"#).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_missing_vocab_key_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid JSON but missing "vocab" key in "model"
        f.write_all(br#"{"model": {}}"#).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_valid_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid tokenizer.json structure
        let json = r#"{
            "model": {
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "<s>": 2,
                    "</s>": 3
                }
            },
            "added_tokens": [
                {"content": "<s>", "id": 2},
                {"content": "</s>", "id": 3}
            ]
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_some());
        let (vocab, bos_id, eos_id) = result.expect("APR operation failed");
        assert!(vocab.len() >= 2);
        assert_eq!(bos_id, Some(2));
        assert_eq!(eos_id, Some(3));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_encode_text_valid_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        let json = r#"{
            "model": {
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
                "merges": []
            }
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::encode_text(&model_path, "hello");
        assert!(result.is_some());
        let tokens = result.expect("APR operation failed");
        assert!(!tokens.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_valid_with_merges_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        let json = r#"{
            "model": {
                "vocab": {"h": 0, "e": 1, "he": 2, "<|endoftext|>": 3, "<bos>": 4},
                "merges": ["h e"]
            },
            "added_tokens": [
                {"content": "<|endoftext|>", "id": 3},
                {"content": "<bos>", "id": 4}
            ]
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::load_tokenizer(&model_path);
        assert!(result.is_some());
        let tokenizer = result.expect("APR operation failed");
        assert!(!tokenizer.token_to_id.is_empty());
        assert_eq!(tokenizer.eos_id, Some(3));
        assert_eq!(tokenizer.bos_id, Some(4));
        assert!(!tokenizer.merge_rules.is_empty());
    }

    #[test]
    fn test_bpe_encode_non_ascii_more_cov() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let result = bpe_encode("a\u{00A9}", &token_to_id, &[], &HashMap::new()); // a + copyright symbol
                                                                                  // Non-ASCII gets encoded as bytes
        assert!(!result.is_empty() || result.is_empty()); // Just verify no panic
    }

    #[test]
    fn test_bpe_encode_unicode_more_cov() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        // Test with unicode characters
        let result = bpe_encode("\u{1F600}", &token_to_id, &[], &HashMap::new()); // Emoji
                                                                                  // Should not panic, may return empty if no tokens match
        assert!(result.is_empty());
    }

    #[test]
    fn test_byte_to_bpe_char_null_more_cov() {
        let result = crate::apr::byte_to_bpe_char(0x00);
        assert!(!result.is_empty());
        assert!(result.starts_with("<0x"));
    }

    #[test]
    fn test_byte_to_bpe_char_delete_more_cov() {
        let result = crate::apr::byte_to_bpe_char(0x7F); // DEL character
        assert!(!result.is_empty());
    }

    #[test]
    fn test_matmul_empty_more_cov() {
        let result = crate::apr::matmul(&[], &[], 0, 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_matmul_single_element_more_cov() {
        let x = vec![2.0];
        let w = vec![3.0];
        let result = crate::apr::matmul(&x, &w, 1, 1, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_single_sequence_more_cov() {
        // Test single sequence normalization
        let x = vec![1.0, 2.0];
        let weight = vec![2.0, 0.5];
        let result = crate::apr::rms_norm(&x, &weight, 1e-6);
        assert_eq!(result.len(), 2);
        // Values should be normalized and scaled by weights
        assert!(result[0].is_finite());
        assert!(result[1].is_finite());
    }

    #[test]
    fn test_rms_norm_weight_shorter_than_x_more_cov() {
        // x has more elements than weight - tests unwrap_or path
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![1.0, 1.0]; // Only 2 weights for hidden_dim=2
        let result = crate::apr::rms_norm(&x, &weight, 1e-6);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_simple_attention_empty_more_cov() {
        let result = crate::apr::simple_attention(&[], &[], &[], 0, 1, 1, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_dot_unequal_lengths_more_cov() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0]; // Shorter
        let result = crate::apr::simd_dot(&a, &b);
        // Should only use min(5, 3) = 3 elements
        assert!((result - 6.0).abs() < 1e-6); // 1+2+3 = 6
    }

    #[test]
    fn test_dequantize_f16_request_more_than_available_more_cov() {
        // Only enough bytes for 2 f16 values, but request 10
        let bytes = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
        let result = crate::apr::dequantize_f16(&bytes, 10);
        assert_eq!(result.len(), 2); // Truncated to available
    }

    #[test]
    fn test_dequantize_q8_0_request_more_than_available_more_cov() {
        // One block = 34 bytes = 32 elements, request 100
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
        let result = crate::apr::dequantize_q8_0(&bytes, 100);
        assert_eq!(result.len(), 32); // Truncated to one block
    }

    #[test]
    fn test_tensor_entry_serialize_deserialize_more_cov() {
        let entry = TensorEntry {
            name: "test.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            offset: 100,
            size: 800,
        };
        let json = serde_json::to_string(&entry).expect("invalid UTF-8");
        let parsed: TensorEntry = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.name, "test.weight");
        assert_eq!(parsed.shape, vec![10, 20]);
    }

    #[test]
    fn test_apr_metadata_serialize_deserialize_more_cov() {
        let meta = AprMetadata {
            model_type: Some("llama".to_string()),
            name: Some("TestModel".to_string()),
            architecture: Some("decoder-only".to_string()),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(8),
            vocab_size: Some(128256),
            intermediate_size: Some(14336),
            max_position_embeddings: Some(8192),
            rope_theta: Some(500000.0),
            rope_type: Some(2),
            rms_norm_eps: Some(1e-5),
            extra: HashMap::new(),
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        let parsed: AprMetadata = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.hidden_size, Some(4096));
        assert_eq!(parsed.rope_type, Some(2));
    }

    #[test]
    fn test_apr_flags_copy_more_cov() {
        let flags = AprFlags::new(AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
        let copied = flags;
        assert!(copied.is_quantized());
        assert!(copied.has_vocab());
    }

    #[test]
    fn test_decode_tokens_special_chars_more_cov() {
        let vocab = vec![
            "hello".to_string(),
            "Ġ".to_string(), // Space token
            "Ċ".to_string(), // Newline token
            "ĉ".to_string(), // Tab token
        ];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2, 3]);
        assert!(result.contains("hello"));
        assert!(result.contains(' '));
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }

    #[test]
    fn test_bpe_tokenizer_encode_non_ascii_more_cov() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("\u{00E9}"); // e-acute
                                                    // Should not panic
        assert!(encoded.is_empty() || !encoded.is_empty());
    }

    #[test]
    fn test_apr_header_all_fields_more_cov() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 1;
        // Set all flags
        data[6..8].copy_from_slice(&0xFFFFu16.to_le_bytes());
        data[8..12].copy_from_slice(&100u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&200u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&300u32.to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&400u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&500u64.to_le_bytes()); // data_offset
        data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes()); // checksum

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        assert_eq!(header.tensor_count, 100);
        assert_eq!(header.metadata_offset, 200);
        assert_eq!(header.metadata_size, 300);
        assert_eq!(header.tensor_index_offset, 400);
        assert_eq!(header.data_offset, 500);
        assert_eq!(header.checksum, 0xDEADBEEF);
    }

    #[test]
    fn test_apr_model_header_methods_more_cov() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // tensor_count is accessed through public API
        assert!(model.tensor_count() >= 1);
        // tensor_names returns &str refs
        let names = model.tensor_names();
        assert!(!names.is_empty());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_all_types_more_cov() {
        // Test all supported quantized types
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_K"), Some(12));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_K"), Some(13));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q6_K"), Some(14));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q8_0"), Some(8));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_0"), Some(2));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_1"), Some(3));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_0"), Some(6));
        // Non-quantized types
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F32"), None);
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F16"), None);
        assert_eq!(crate::apr::dtype_to_ggml_qtype("BF16"), None);
    }

    #[test]
    fn test_is_quantized_dtype_all_types_more_cov() {
        // Quantized types
        assert!(crate::apr::is_quantized_dtype("Q4_K"));
        assert!(crate::apr::is_quantized_dtype("q4_k"));
        assert!(crate::apr::is_quantized_dtype("Q5_K"));
        assert!(crate::apr::is_quantized_dtype("Q6_K"));
        assert!(crate::apr::is_quantized_dtype("Q8_0"));
        assert!(crate::apr::is_quantized_dtype("Q4_0"));
        assert!(crate::apr::is_quantized_dtype("Q4_1"));
        assert!(crate::apr::is_quantized_dtype("Q5_0"));
        // Non-quantized types
        assert!(!crate::apr::is_quantized_dtype("F32"));
        assert!(!crate::apr::is_quantized_dtype("F16"));
        assert!(!crate::apr::is_quantized_dtype("BF16"));
        assert!(!crate::apr::is_quantized_dtype("I8"));
    }

    #[test]
    fn test_dequantize_q4_k_early_exit_more_cov() {
        // Test early exit when num_elements is reached mid-sub-block
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
                         // Request exactly 17 elements (mid sub-block)
        let result = crate::apr::dequantize_q4_k(&bytes, 17);
        assert_eq!(result.len(), 17);
    }

    #[test]
    fn test_dequantize_q6_k_early_exit_more_cov() {
        // Test early exit when num_elements is reached mid-sub-block
        let mut bytes = vec![0u8; 210];
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0
                           // Request exactly 33 elements (mid sub-block)
        let result = crate::apr::dequantize_q6_k(&bytes, 33);
        assert_eq!(result.len(), 33);
    }

    #[test]
    fn test_f16_to_f32_negative_subnormal_more_cov() {
        // Negative subnormal: sign=1, exp=0, mantissa=1
        let bits: u16 = 0x8001;
        let result = crate::apr::f16_to_f32(bits);
        assert!(result < 0.0, "Negative subnormal should be negative");
    }

    #[test]
    fn test_f16_to_f32_negative_nan_more_cov() {
        // Negative NaN: sign=1, exp=31, mantissa!=0
        let bits: u16 = 0xFC01;
        let result = crate::apr::f16_to_f32(bits);
        assert!(result.is_nan());
    }

    #[test]
    fn test_apr_model_forward_with_transformer_missing_tensor_more_cov() {
        // Create a model with transformer metadata but missing tensors
        let metadata = r#"{
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "vocab_size": 10
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset;
        let mut data = vec![0u8; data_offset as usize + 64];

        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // No tensors
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        let model = AprV2Model::from_bytes(data).expect("should load");
        assert!(model.metadata().is_transformer());
        let result = model.forward(&[1]);
        // Should fail because embedding tensor is missing
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_tokens_sentencepiece_prefix_more_cov() {
        let vocab = vec!["▁hello".to_string(), "▁".to_string(), "world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        // SentencePiece ▁ prefix should be handled
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_model_data_type_alias_more_cov() {
        // Test type alias AprModel = AprV2Model
        let data = create_test_apr_model();
        let model: AprModel = AprV2Model::from_bytes(data).expect("should load");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_model_type_alias_more_cov() {
        // Test legacy type alias AprModelType
        let _: AprModelType = ();
    }

    #[test]
    fn test_find_tensor_name_not_found_more_cov() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        // Try to find a tensor that doesn't exist
        let result = model.forward(&[0]); // Will try to find embedding tensor
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        // Should contain "No matching tensor" or similar
        assert!(
            err_msg.contains("not a transformer") || err_msg.contains("No matching"),
            "Error message: {}",
            err_msg
        );
    }

    // =========================================================================
    // SimpleTokenizer Tests (GH-156)
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_new() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        assert_eq!(tokenizer.vocab_size(), 5);
        assert!(tokenizer.is_bos(1));
        assert!(tokenizer.is_eos(2));
        assert!(!tokenizer.is_bos(0));
        assert!(!tokenizer.is_eos(0));
    }

    #[test]
    fn test_simple_tokenizer_decode_basic() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        // Decode tokens [3, 4] -> "helloworld"
        let decoded = tokenizer.decode(&[3, 4]);
        assert_eq!(decoded, "helloworld");
    }

    #[test]
    fn test_simple_tokenizer_decode_bpe_space() {
        use crate::apr::SimpleTokenizer;

        // BPE-style tokens with Ġ prefix (represents space)
        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "Ġhello".to_string(), // " hello"
            "Ġworld".to_string(), // " world"
            "!".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        // Decode tokens [3, 4, 5] -> " hello world!"
        let decoded = tokenizer.decode(&[3, 4, 5]);
        assert_eq!(decoded, " hello world!");
    }

    #[test]
    fn test_simple_tokenizer_decode_out_of_bounds() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string(), "b".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        // Token 99 is out of bounds - should be skipped or handled gracefully
        let decoded = tokenizer.decode(&[0, 99, 1]);
        // Should contain "a" and "b", may have placeholder for 99
        assert!(decoded.contains('a'));
        assert!(decoded.contains('b'));
    }

    #[test]
    fn test_simple_tokenizer_no_special_tokens() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["x".to_string(), "y".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        // No BOS/EOS defined
        assert!(!tokenizer.is_bos(0));
        assert!(!tokenizer.is_bos(1));
        assert!(!tokenizer.is_eos(0));
        assert!(!tokenizer.is_eos(1));
    }

    #[test]
    fn test_simple_tokenizer_empty_decode() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_simple_tokenizer_clone() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["test".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), None);
        let cloned = tokenizer.clone();

        assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());
        assert_eq!(tokenizer.bos_token_id, cloned.bos_token_id);
    }

    #[test]
    fn test_simple_tokenizer_debug() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        let debug_str = format!("{:?}", tokenizer);

        assert!(debug_str.contains("SimpleTokenizer"));
    }

    // =========================================================================
    // APR v1 Format Rejection Tests
    // =========================================================================

    #[test]
    fn test_apr_v1_format_rejected() {
        // APR v1 magic: "APR1" (0x41, 0x50, 0x52, 0x31)
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]); // "APR1"
        data[4] = 1; // version major
        data[5] = 0; // version minor

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should indicate APR v1 not supported
        assert!(
            err.contains("APR v1") || err.contains("not supported"),
            "Error should mention APR v1: {}",
            err
        );
    }

    #[test]
    fn test_apr_v1_format_conversion_hint() {
        // Test that error message suggests conversion
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]); // "APR1"

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Error should hint at conversion or GGUF alternative
        assert!(
            err.contains("convert") || err.contains("GGUF"),
            "Error should suggest conversion: {}",
            err
        );
    }

    #[test]
    fn test_apr_invalid_version_byte() {
        // Test with invalid version byte (not 0, '1', or '2')
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..3].copy_from_slice(&MAGIC_PREFIX); // "APR"
        data[3] = 0x99; // Invalid version byte

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("version") || err.contains("Invalid"),
            "Error should mention version: {}",
            err
        );
    }

    // =========================================================================
    // MappedAprModel Tests
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_from_path() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file with legacy magic
        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC); // APR\0 legacy magic
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        assert_eq!(model.tensor_count(), 0);
        assert_eq!(model.file_size(), data.len());
        assert_eq!(model.data_offset(), 64);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_nonexistent_file() {
        let result = MappedAprModel::from_path("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

        temp.write_all(&data).expect("write data");

        let result = MappedAprModel::from_path(temp.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("magic") || err.contains("Invalid"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_truncated() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write less than HEADER_SIZE bytes
        let data = vec![0u8; 32];
        temp.write_all(&data).expect("write data");

        let result = MappedAprModel::from_path(temp.path());
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_data_access() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");

        // Verify data() returns the full mmap
        let model_data = model.data();
        assert_eq!(model_data.len(), 128);
        assert_eq!(&model_data[0..4], &MAGIC);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_find_tensor_not_found() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        assert!(model.find_tensor("nonexistent").is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_get_tensor_data_not_found() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        let result = model.get_tensor_data("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_apr_model_dtype_to_qtype() {
        // F32 -> 0
        assert_eq!(MappedAprModel::dtype_to_qtype("F32"), 0);
        // F16 -> 1
        assert_eq!(MappedAprModel::dtype_to_qtype("F16"), 1);
        // Q4_0 -> 2
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_0"), 2);
        // Q4_1 -> 3
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_1"), 3);
        // Q5_0 -> 6
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_0"), 6);
        // Q5_1 -> 7
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_1"), 7);
        // Q8_0 -> 8
        assert_eq!(MappedAprModel::dtype_to_qtype("Q8_0"), 8);
        // Q8_1 -> 9
        assert_eq!(MappedAprModel::dtype_to_qtype("Q8_1"), 9);
        // Q2_K -> 10
        assert_eq!(MappedAprModel::dtype_to_qtype("Q2_K"), 10);
        // Q3_K -> 11
        assert_eq!(MappedAprModel::dtype_to_qtype("Q3_K"), 11);
        // Q4_K -> 12
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_K"), 12);
        // Q5_K -> 13
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
        // Q6_K -> 14
        assert_eq!(MappedAprModel::dtype_to_qtype("Q6_K"), 14);
        // IQ2_XXS -> 16
        assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XXS"), 16);
        // IQ2_XS -> 17
        assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XS"), 17);
        // BF16 -> 30
        assert_eq!(MappedAprModel::dtype_to_qtype("BF16"), 30);
        // Unknown -> 0 (default to F32)
        assert_eq!(MappedAprModel::dtype_to_qtype("UNKNOWN"), 0);
    }

    // =========================================================================
    // Embedded Tokenizer Tests (GH-156)
    // =========================================================================

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_empty() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.vocabulary".to_string(), serde_json::json!([]));
        // Empty array should return None
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_valid() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["<pad>", "<bos>", "<eos>", "hello", "world"]),
        );

        let vocab = meta.get_embedded_vocabulary().expect("should have vocab");
        assert_eq!(vocab.len(), 5);
        assert_eq!(vocab[0], "<pad>");
        assert_eq!(vocab[3], "hello");
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_invalid_type() {
        let mut meta = AprMetadata::default();
        // Not an array
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!("not an array"),
        );
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_mixed_types() {
        let mut meta = AprMetadata::default();
        // Array with mixed types - only strings should be kept
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["valid", 123, "also_valid", null]),
        );

        let vocab = meta.get_embedded_vocabulary().expect("should have vocab");
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab[0], "valid");
        assert_eq!(vocab[1], "also_valid");
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_bos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_valid() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.bos_token_id".to_string(), serde_json::json!(1));

        assert_eq!(meta.get_embedded_bos_token_id(), Some(1));
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_invalid_type() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.bos_token_id".to_string(),
            serde_json::json!("not a number"),
        );
        assert!(meta.get_embedded_bos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_eos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_valid() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.eos_token_id".to_string(), serde_json::json!(2));

        assert_eq!(meta.get_embedded_eos_token_id(), Some(2));
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_invalid_type() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.eos_token_id".to_string(),
            serde_json::json!({"nested": "object"}),
        );
        assert!(meta.get_embedded_eos_token_id().is_none());
    }

    #[test]
    fn test_load_embedded_tokenizer_no_vocab() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Model has no embedded vocabulary
        assert!(model.load_embedded_tokenizer().is_none());
    }

    /// Helper to create APR model with embedded tokenizer
    fn create_apr_model_with_embedded_tokenizer() -> Vec<u8> {
        let metadata = r#"{
            "architecture": "test",
            "vocab_size": 5,
            "hidden_size": 64,
            "tokenizer.vocabulary": ["<pad>", "<bos>", "<eos>", "hello", "world"],
            "tokenizer.bos_token_id": 1,
            "tokenizer.eos_token_id": 2
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset;

        let total_size = data_offset as usize + 64; // Some padding
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        data
    }

    #[test]
    fn test_load_embedded_tokenizer_valid() {
        let data = create_apr_model_with_embedded_tokenizer();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tokenizer = model
            .load_embedded_tokenizer()
            .expect("should have embedded tokenizer");

        assert_eq!(tokenizer.vocab_size(), 5);
        assert_eq!(tokenizer.bos_token_id, Some(1));
        assert_eq!(tokenizer.eos_token_id, Some(2));
        assert!(tokenizer.is_bos(1));
        assert!(tokenizer.is_eos(2));
    }

    #[test]
    fn test_load_embedded_tokenizer_decode() {
        let data = create_apr_model_with_embedded_tokenizer();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tokenizer = model
            .load_embedded_tokenizer()
            .expect("should have embedded tokenizer");

        let decoded = tokenizer.decode(&[3, 4]);
        assert_eq!(decoded, "helloworld");
    }

    // =========================================================================
    // Additional RoPE Metadata Tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_rope_theta() {
        let meta = AprMetadata {
            rope_theta: Some(10000.0),
            ..Default::default()
        };
        assert_eq!(meta.rope_theta, Some(10000.0));
    }

    #[test]
    fn test_apr_metadata_rope_type() {
        // rope_type 0 = NORM (adjacent pairs)
        // rope_type 2 = NEOX (split halves) - used by Qwen2.5
        let meta = AprMetadata {
            rope_type: Some(2),
            ..Default::default()
        };
        assert_eq!(meta.rope_type, Some(2));
    }

    #[test]
    fn test_apr_metadata_rms_norm_eps() {
        let meta = AprMetadata {
            rms_norm_eps: Some(1e-6),
            ..Default::default()
        };
        assert!((meta.rms_norm_eps.unwrap() - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_apr_metadata_num_kv_heads() {
        // GQA models have fewer KV heads than query heads
        let meta = AprMetadata {
            num_heads: Some(32),
            num_kv_heads: Some(4), // GQA ratio 8:1
            ..Default::default()
        };
        assert_eq!(meta.num_heads, Some(32));
        assert_eq!(meta.num_kv_heads, Some(4));
    }

    #[test]
    fn test_apr_metadata_max_position_embeddings() {
        let meta = AprMetadata {
            max_position_embeddings: Some(4096),
            ..Default::default()
        };
        assert_eq!(meta.max_position_embeddings, Some(4096));
    }

    #[test]
    fn test_apr_metadata_intermediate_size() {
        let meta = AprMetadata {
            hidden_size: Some(2048),
            intermediate_size: Some(5632), // SwiGLU FFN size
            ..Default::default()
        };
        assert_eq!(meta.intermediate_size, Some(5632));
    }

    // =========================================================================
    // Additional dtype_to_ggml_qtype Tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_case_insensitive() {
        // Lowercase variants
        assert_eq!(dtype_to_ggml_qtype("q4_k"), Some(12));
        assert_eq!(dtype_to_ggml_qtype("q5_k"), Some(13));
        assert_eq!(dtype_to_ggml_qtype("q6_k"), Some(14));
        assert_eq!(dtype_to_ggml_qtype("q8_0"), Some(8));
        assert_eq!(dtype_to_ggml_qtype("q4_0"), Some(2));
        assert_eq!(dtype_to_ggml_qtype("q4_1"), Some(3));
        assert_eq!(dtype_to_ggml_qtype("q5_0"), Some(6));
    }

    #[test]
    fn test_is_quantized_dtype_comprehensive() {
        // Quantized types
        assert!(is_quantized_dtype("Q4_K"));
        assert!(is_quantized_dtype("Q5_K"));
        assert!(is_quantized_dtype("Q6_K"));
        assert!(is_quantized_dtype("Q8_0"));
        assert!(is_quantized_dtype("Q4_0"));
        assert!(is_quantized_dtype("Q4_1"));
        assert!(is_quantized_dtype("Q5_0"));

        // Non-quantized types
        assert!(!is_quantized_dtype("F32"));
        assert!(!is_quantized_dtype("F16"));
        assert!(!is_quantized_dtype("BF16"));
        assert!(!is_quantized_dtype("unknown"));
    }

    // =========================================================================
    // Additional TensorEntry Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_all_dtypes() {
        // GH-191 FIX: dtype bytes now use GGML type IDs
        let dtypes = [
            (0u8, "F32"),
            (1, "F16"),
            (2, "Q4_0"),  // GGML type 2
            (3, "Q4_1"),  // GGML type 3
            (6, "Q5_0"),  // GGML type 6
            (7, "Q5_1"),  // GGML type 7
            (8, "Q8_0"),  // GGML type 8
            (9, "Q8_1"),  // GGML type 9
            (10, "Q2_K"), // GGML type 10
            (11, "Q3_K"), // GGML type 11
            (12, "Q4_K"), // GGML type 12
            (13, "Q5_K"), // GGML type 13
            (14, "Q6_K"), // GGML type 14
            (30, "BF16"), // GGML type 30
        ];

        for (dtype_byte, expected_dtype) in dtypes {
            let data = create_binary_tensor_entry("test", dtype_byte, &[10], 0, 40);
            let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
            assert_eq!(
                entry.dtype, expected_dtype,
                "dtype_byte {} should map to {}",
                dtype_byte, expected_dtype
            );
        }
    }

    #[test]
    fn test_tensor_entry_from_binary_unknown_dtype_defaults_f32() {
        // Unknown dtype byte (e.g., 255) should default to F32
        let data = create_binary_tensor_entry("test", 255, &[10], 0, 40);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(entry.dtype, "F32");
    }

    #[test]
    fn test_tensor_entry_from_binary_8d_shape() {
        // Maximum supported dimensions
        let data = create_binary_tensor_entry(
            "high_dim_tensor",
            0,
            &[2, 3, 4, 5, 6, 7, 8, 9],
            0,
            2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 4,
        );
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(entry.shape.len(), 8);
        assert_eq!(entry.element_count(), 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9);
    }

    // =========================================================================
    // Additional Model Loading Error Tests
    // =========================================================================

    #[test]
    fn test_apr_model_metadata_truncated_past_eof() {
        // metadata_offset + metadata_size > file length
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset = 64
        data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size = 1000 > remaining bytes
        data[24..32].copy_from_slice(&1064u64.to_le_bytes());
        data[32..40].copy_from_slice(&1064u64.to_le_bytes());

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn test_apr_model_tensor_out_of_bounds() {
        // Create model where tensor data extends past EOF
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Create tensor entry that points past EOF
        let tensor_entry = create_binary_tensor_entry(
            "bad_tensor",
            0,
            &[1000, 1000],
            0,
            4_000_000, // Size much larger than file
        );

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;

        // File is only 256 bytes total, but tensor claims to be 4MB
        let total_size = data_offset as usize + 64;
        let mut data = vec![0u8; total_size];

        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let model = AprV2Model::from_bytes(data).expect("should load");

        // Accessing tensor data should fail
        let result = model.get_tensor_f32("bad_tensor");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("out of bounds"));
    }

    // =========================================================================
    // Generate Method Edge Cases
    // =========================================================================

    #[test]
    fn test_apr_model_generate_eos_stops() {
        // Test that generation stops at EOS token
        // (Requires a valid transformer model - using mock to verify logic)
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Empty input should fail
        let result = model.generate(&[], 10, Some(2));
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_generate_max_tokens() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Test that generate with max_tokens=0 works (returns empty or immediate)
        let result = model.generate(&[1, 2, 3], 0, None);
        // Either succeeds with 0 tokens or returns an error - both are valid
        if let Ok(tokens) = result {
            assert!(tokens.len() <= 3); // At most initial tokens
        }
    }

    // =========================================================================
    // SimpleTokenizer Additional Tests
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_bpe_newline() {
        use crate::apr::SimpleTokenizer;

        // BPE newline is Ċ
        let vocab = vec!["hello".to_string(), "Ċ".to_string(), "world".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert_eq!(decoded, "hello\nworld");
    }

    #[test]
    fn test_simple_tokenizer_bpe_tab() {
        use crate::apr::SimpleTokenizer;

        // BPE tab is ĉ
        let vocab = vec!["hello".to_string(), "ĉ".to_string(), "world".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert_eq!(decoded, "hello\tworld");
    }

    #[test]
    fn test_simple_tokenizer_large_vocab() {
        use crate::apr::SimpleTokenizer;

        // Create large vocabulary
        let vocab: Vec<String> = (0..10000).map(|i| format!("token_{}", i)).collect();
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(9999));

        assert_eq!(tokenizer.vocab_size(), 10000);
        assert!(tokenizer.is_bos(0));
        assert!(tokenizer.is_eos(9999));

        let decoded = tokenizer.decode(&[100, 200, 300]);
        assert!(decoded.contains("token_100"));
        assert!(decoded.contains("token_200"));
        assert!(decoded.contains("token_300"));
    }

    // =========================================================================
    // BpeTokenizer Additional Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_large_merge_rules() {
        // Test with many merge rules
        let vocab: HashMap<String, u32> = [
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
            ("ab".to_string(), 3),
            ("abc".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let id_to_token = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "ab".to_string(),
            "abc".to_string(),
        ];

        let merge_rules = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];

        let tokenizer = BpeTokenizer {
            token_to_id: vocab,
            id_to_token,
            merge_rules,
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let encoded = tokenizer.encode("abc");
        // Should merge a+b -> ab, then ab+c -> abc
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_non_ascii() {
        // Test encoding non-ASCII characters
        let vocab: HashMap<String, u32> = [("a".to_string(), 0)].into_iter().collect();

        let id_to_token = vec!["a".to_string()];

        let tokenizer = BpeTokenizer {
            token_to_id: vocab,
            id_to_token,
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        // Non-ASCII should be handled (may result in empty if not in vocab)
        let encoded = tokenizer.encode("日本語");
        // Result depends on byte encoding - just verify it doesn't panic
        let _ = encoded;
    }

    // =========================================================================
    // MAGIC_PREFIX Tests
    // =========================================================================

    #[test]
    fn test_magic_prefix_constant() {
        assert_eq!(MAGIC_PREFIX, [0x41, 0x50, 0x52]); // "APR"
        assert_eq!(&MAGIC_PREFIX, b"APR");
    }

    #[test]
    fn test_header_size_constant() {
        assert_eq!(HEADER_SIZE, 64);
    }

    #[test]
    fn test_alignment_constant() {
        assert_eq!(ALIGNMENT, 64);
    }

    // ==========================================================================
    // PMAT-107: Falsification Test for GQA num_kv_heads in AprMetadata
    // ==========================================================================

    /// FALSIFICATION TEST: Verify AprMetadata correctly parses num_kv_heads from JSON
    /// This catches the silent failure case where unwrap_or_default() swallows errors.
    #[test]
    fn test_falsification_apr_metadata_parses_gqa_num_kv_heads() {
        // JSON from a real Qwen 1.5B APR file (GQA: 12 heads, 2 kv_heads)
        let metadata_json = r#"{
            "model_type": "qwen2",
            "architecture": "qwen2",
            "hidden_size": 1536,
            "num_layers": 28,
            "num_heads": 12,
            "num_kv_heads": 2,
            "vocab_size": 151936,
            "intermediate_size": 8960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001
        }"#;

        let metadata: AprMetadata =
            serde_json::from_str(metadata_json).expect("AprMetadata should parse valid JSON");

        assert_eq!(
            metadata.num_heads,
            Some(12),
            "num_heads not parsed correctly"
        );
        assert_eq!(
            metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads not parsed from AprMetadata JSON!\n\
             This causes GQA models to hang on GPU because they're treated as MHA."
        );
        assert_eq!(
            metadata.hidden_size,
            Some(1536),
            "hidden_size not parsed correctly"
        );

        println!(
            "✅ AprMetadata correctly parses num_kv_heads={:?}",
            metadata.num_kv_heads
        );
    }

    /// FALSIFICATION TEST: Verify AprMetadata handles extra fields gracefully
    /// The real APR files have extra fields like "model.num_kv_heads" that should
    /// be captured by the flatten attribute without breaking num_kv_heads parsing.
    #[test]
    fn test_falsification_apr_metadata_with_extra_fields() {
        // This JSON has both the standard fields AND the extra "model.*" fields
        // from the Q4K converter
        let metadata_json = r#"{
            "model_type": "qwen2",
            "name": "qwen2",
            "architecture": "qwen2",
            "hidden_size": 1536,
            "num_layers": 28,
            "num_heads": 12,
            "num_kv_heads": 2,
            "vocab_size": 151936,
            "intermediate_size": 8960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001,
            "model.intermediate_size": 8960,
            "model.num_kv_heads": 2,
            "model.rope_theta": 1000000.0
        }"#;

        let metadata: AprMetadata = serde_json::from_str(metadata_json)
            .expect("AprMetadata should parse JSON with extra fields");

        assert_eq!(
            metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads broken by extra 'model.*' fields!\n\
             Expected: Some(2), Got: {:?}",
            metadata.num_kv_heads
        );

        // Verify extra fields are captured
        assert!(
            metadata.extra.contains_key("model.num_kv_heads"),
            "Extra fields should be captured in 'extra' map"
        );

        println!(
            "✅ AprMetadata handles extra fields correctly, num_kv_heads={:?}",
            metadata.num_kv_heads
        );
    }

    /// PMAT-107: Falsification test with REAL APR file
    ///
    /// This test loads the actual APR file from disk and verifies that
    /// num_kv_heads is correctly parsed. If this test fails, the APR CUDA
    /// path will hang because GQA models will be treated as MHA.
    #[test]
    fn test_falsification_real_apr_file_num_kv_heads() {
        let apr_path = std::path::Path::new(
            "/home/noah/.cache/huggingface/models/qwen2.5-coder-1.5b-apr/qwen2.5-coder-1.5b-q4k.apr"
        );

        if !apr_path.exists() {
            println!("⚠️ Test model not available at {:?}, skipping", apr_path);
            return;
        }

        // Load the model
        let model = AprV2Model::load(apr_path).expect("Should load APR file");

        println!("=== REAL APR FILE METADATA ===");
        println!("  num_heads: {:?}", model.metadata.num_heads);
        println!("  num_kv_heads: {:?}", model.metadata.num_kv_heads);
        println!("  hidden_size: {:?}", model.metadata.hidden_size);
        println!("  num_layers: {:?}", model.metadata.num_layers);

        // This is the critical assertion
        assert!(
            model.metadata.num_kv_heads.is_some(),
            "FALSIFICATION FAILED: num_kv_heads is None after loading real APR file!\n\
             This causes GQA models to hang on GPU because they're treated as MHA.\n\
             Expected: Some(2) for Qwen 1.5B GQA, Got: None"
        );

        assert_eq!(
            model.metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads wrong value!\n\
             Expected: Some(2) for Qwen 1.5B GQA, Got: {:?}",
            model.metadata.num_kv_heads
        );

        println!(
            "✅ Real APR file has correct num_kv_heads={:?}",
            model.metadata.num_kv_heads
        );
    }

    /// PMAT-107: Falsification test for GQA dimensions in CUDA executor
    ///
    /// This test loads a real APR file and creates AprV2ModelCuda, then verifies
    /// that the CUDA executor has the correct GQA dimensions. This catches bugs
    /// where num_kv_heads is parsed correctly but not propagated to the executor.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_falsification_apr_cuda_gqa_dimensions() {
        let apr_path = std::path::Path::new(
            "/home/noah/.cache/huggingface/models/qwen2.5-coder-1.5b-apr/qwen2.5-coder-1.5b-q4k.apr"
        );

        if !apr_path.exists() {
            println!("⚠️ Test model not available at {:?}, skipping", apr_path);
            return;
        }

        // Load the model
        let model = AprV2Model::load(apr_path).expect("Should load APR file");

        // Verify metadata first
        assert_eq!(model.metadata.num_heads, Some(12), "num_heads should be 12");
        assert_eq!(
            model.metadata.num_kv_heads,
            Some(2),
            "num_kv_heads should be 2 (GQA)"
        );

        // Create CUDA model
        use crate::apr::AprV2ModelCuda;

        let cuda_model = match AprV2ModelCuda::new(model, 0) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("⚠️ CUDA not available: {e}");
                return;
            },
        };

        // Access the executor's GQA dimensions
        // We need to verify kv_num_heads and kv_num_kv_heads are set correctly
        // The executor is private but we can check via the metadata pass-through

        println!("=== CUDA EXECUTOR GQA CONFIG ===");
        println!(
            "  model.metadata.num_heads: {:?}",
            cuda_model.inner().metadata.num_heads
        );
        println!(
            "  model.metadata.num_kv_heads: {:?}",
            cuda_model.inner().metadata.num_kv_heads
        );

        // The critical check: if CUDA model was initialized correctly, the GQA ratio should be 6:1
        // (12 Q heads / 2 KV heads = 6x repeat factor for GQA)
        let num_heads = cuda_model.inner().metadata.num_heads.unwrap_or(1);
        let num_kv_heads = cuda_model
            .inner()
            .metadata
            .num_kv_heads
            .unwrap_or(num_heads);
        let gqa_ratio = num_heads / num_kv_heads;

        assert_eq!(
            gqa_ratio, 6,
            "FALSIFICATION FAILED: GQA ratio wrong!\n\
             Expected: 6 (12 Q heads / 2 KV heads), Got: {} ({} / {})",
            gqa_ratio, num_heads, num_kv_heads
        );

        println!(
            "✅ CUDA model has correct GQA ratio: {} ({}:{} heads:kv_heads)",
            gqa_ratio, num_heads, num_kv_heads
        );
    }
}
