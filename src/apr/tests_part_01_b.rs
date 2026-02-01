    #[test]
    fn test_bpe_tokenizer_unknown_char() {
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
        let encoded = tokenizer.encode("xyz");
        // Unknown chars should be handled gracefully
        assert!(encoded.is_empty() || encoded.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_bpe_tokenizer_decode_empty() {
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
        let decoded = tokenizer.decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode_valid() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 0);
        token_to_id.insert("world".to_string(), 1);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["hello".to_string(), "world".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0, 1]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_bpe_tokenizer_with_merge_rules() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("he".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["h".to_string(), "e".to_string(), "he".to_string()],
            merge_rules: vec![("h".to_string(), "e".to_string())],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("he");
        // After merging h+e -> he, should have 1 token
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_with_bos_eos() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<s>".to_string(), 0);
        token_to_id.insert("</s>".to_string(), 1);
        token_to_id.insert("a".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["<s>".to_string(), "</s>".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: Some(0),
            eos_id: Some(1),
            special_tokens: HashMap::new(),
        };
        assert_eq!(tokenizer.bos_id, Some(0));
        assert_eq!(tokenizer.eos_id, Some(1));
    }

    // =========================================================================
    // GH-189: extract_special_tokens_from_vocab Tests
    // =========================================================================

    #[test]
    fn test_extract_special_tokens_from_vocab_qwen() {
        // GH-189: Test extraction of Qwen-style special tokens
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<|im_start|>".to_string(), 151644);
        vocab.insert("<|im_end|>".to_string(), 151645);
        vocab.insert("<|endoftext|>".to_string(), 151643);
        vocab.insert("hello".to_string(), 15339);
        vocab.insert("world".to_string(), 1917);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert_eq!(special.len(), 3);
        assert_eq!(special.get("<|im_start|>"), Some(&151644));
        assert_eq!(special.get("<|im_end|>"), Some(&151645));
        assert_eq!(special.get("<|endoftext|>"), Some(&151643));
        // Regular tokens should NOT be in special_tokens
        assert!(!special.contains_key("hello"));
        assert!(!special.contains_key("world"));
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_llama() {
        // GH-189: Test extraction of LLaMA-style special tokens
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<s>".to_string(), 1);
        vocab.insert("</s>".to_string(), 2);
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<pad>".to_string(), 3);
        vocab.insert("the".to_string(), 1000);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert!(special.contains_key("<s>"));
        assert!(special.contains_key("</s>"));
        assert!(special.contains_key("<unk>"));
        assert!(special.contains_key("<pad>"));
        assert!(!special.contains_key("the"));
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_empty() {
        // GH-189: Empty vocab should return empty special tokens
        let vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let special = extract_special_tokens_from_vocab(&vocab);
        assert!(special.is_empty());
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_no_special() {
        // GH-189: Vocab with no special tokens should return empty
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("the".to_string(), 2);

        let special = extract_special_tokens_from_vocab(&vocab);
        assert!(special.is_empty());
    }

    #[test]
    fn test_extract_special_tokens_pattern_matching() {
        // GH-189: Test that any <|...|> pattern is captured
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<|custom_token|>".to_string(), 999);
        vocab.insert("<|another|>".to_string(), 998);
        vocab.insert("not_special".to_string(), 100);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert!(special.contains_key("<|custom_token|>"));
        assert!(special.contains_key("<|another|>"));
        assert!(!special.contains_key("not_special"));
    }

    // =========================================================================
    // is_apr_file Tests
    // =========================================================================

    #[test]
    fn test_is_apr_file_nonexistent() {
        // Non-existent file returns false (can't read magic bytes)
        assert!(!is_apr_file("/nonexistent/path/model.apr"));
    }

    #[test]
    fn test_is_apr_file_wrong_extension() {
        // Non-existent files all return false
        assert!(!is_apr_file("/some/path/model.gguf"));
        assert!(!is_apr_file("/some/path/model.safetensors"));
        assert!(!is_apr_file("/some/path/model.bin"));
    }

    #[test]
    fn test_is_apr_file_no_extension() {
        assert!(!is_apr_file("/some/path/model"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_valid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        // Write APR magic bytes
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(is_apr_file(temp.path()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_wrong_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        // Write wrong magic bytes
        temp.write_all(b"GGUF").expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(!is_apr_file(temp.path()));
    }

    // =========================================================================
    // detect_format Tests
    // =========================================================================

    #[test]
    fn test_detect_format_apr() {
        assert_eq!(detect_format("/path/model.apr"), "apr");
    }

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_format("/path/model.gguf"), "gguf");
    }

    #[test]
    fn test_detect_format_safetensors() {
        assert_eq!(detect_format("/path/model.safetensors"), "safetensors");
    }

    // =========================================================================
    // TensorEntry Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_dtype_sizes() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            offset: 0,
            size: 800, // 10*20*4 bytes
        };
        assert_eq!(entry.element_count(), 200);
    }

    #[test]
    fn test_tensor_entry_empty_shape() {
        let entry = TensorEntry {
            name: "scalar".to_string(),
            dtype: "F32".to_string(),
            shape: vec![],
            offset: 0,
            size: 4,
        };
        assert_eq!(entry.element_count(), 1);
    }

    #[test]
    fn test_tensor_entry_4d_shape() {
        let entry = TensorEntry {
            name: "4d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4, 5],
            offset: 0,
            size: 480,
        };
        assert_eq!(entry.element_count(), 120);
    }

    #[test]
    fn test_tensor_entry_1d_shape() {
        let entry = TensorEntry {
            name: "1d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100],
            offset: 0,
            size: 400,
        };
        assert_eq!(entry.element_count(), 100);
    }

    // =========================================================================
    // ModelData Tests
    // =========================================================================

    #[test]
    fn test_model_data_len() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(data.len(), 5);
    }

    #[test]
    fn test_model_data_is_empty() {
        let empty = ModelData::from_vec(vec![]);
        assert!(empty.is_empty());

        let non_empty = ModelData::from_vec(vec![1]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_model_data_as_slice() {
        let data = ModelData::from_vec(vec![10, 20, 30]);
        let slice = data.as_slice();
        assert_eq!(slice, &[10, 20, 30]);
    }

    #[test]
    fn test_model_data_large() {
        let large_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let data = ModelData::from_vec(large_data);
        assert_eq!(data.len(), 1000);
        assert_eq!(data.as_slice()[0], 0);
        assert_eq!(data.as_slice()[255], 255);
        assert_eq!(data.as_slice()[256], 0);
    }

    // =========================================================================
    // AprHeader Tests
    // =========================================================================

    #[test]
    fn test_apr_header_from_bytes_valid() {
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC); // APR\0
        data.extend_from_slice(&[2, 0]); // version 2.0
        data.extend_from_slice(&[0, 0]); // flags
        data.extend_from_slice(&5u32.to_le_bytes()); // tensor_count = 5
        data.extend_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data.extend_from_slice(&100u32.to_le_bytes()); // metadata_size
        data.extend_from_slice(&164u64.to_le_bytes()); // tensor_index_offset
        data.extend_from_slice(&500u64.to_le_bytes()); // data_offset
        data.extend_from_slice(&0u32.to_le_bytes()); // checksum
        data.extend_from_slice(&[0u8; 20]); // reserved

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_ok());
        let header = result.expect("APR operation failed");
        assert_eq!(header.version.0, 2);
        assert_eq!(header.version.1, 0);
        assert_eq!(header.tensor_count, 5);
    }

    #[test]
    fn test_apr_header_from_bytes_wrong_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // Wrong magic
        data.extend_from_slice(&[0u8; 60]); // padding

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_header_from_bytes_too_short() {
        let data = vec![0u8; 10]; // Too short
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    // =========================================================================
    // AprMetadata Tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_is_transformer_true() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_hidden() {
        let meta = AprMetadata {
            hidden_size: None,
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_layers() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: None,
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_default() {
        let meta = AprMetadata::default();
        assert!(meta.hidden_size.is_none());
        assert!(meta.num_layers.is_none());
        assert!(!meta.is_transformer());
    }

    // =========================================================================
    // AprV2Model Tests - Basic Operations
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_minimal() {
        let data = create_test_apr_model();
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_ok());
        let model = result.expect("APR operation failed");
        // Helper creates 1 tensor
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_v2_model_tensor_names() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let names = model.tensor_names();
        // Helper creates tensor "test.weight"
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"test.weight"));
    }

    #[test]
    fn test_apr_v2_model_metadata_default() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let meta = model.metadata();
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert!(model.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_bytes_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_f32_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_estimated_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // Helper creates 1 tensor with shape [4,4] = 16 elements
        assert_eq!(model.estimated_parameters(), 16);
    }

    #[test]
    fn test_apr_v2_model_is_mmap_false() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert!(!model.is_mmap());
    }

    // =========================================================================
    // AprV2Model Tests - predict
    // =========================================================================

    #[test]
    fn test_apr_v2_model_predict_no_tensors() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features = vec![1.0, 2.0, 3.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
        // With no tensors, returns sum of features
        let output = result.expect("APR operation failed");
        assert_eq!(output.len(), 1);
        assert!((output[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_apr_v2_model_predict_empty_features() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features: Vec<f32> = vec![];
        let result = model.predict(&features);
        assert!(result.is_ok());
        let output = result.expect("APR operation failed");
        assert_eq!(output[0], 0.0);
    }

    // =========================================================================
    // AprV2Model Tests - forward
    // =========================================================================

    #[test]
    fn test_apr_v2_model_forward_empty_tokens() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[]);
        assert!(result.is_err()); // Empty tokens should fail
    }

    #[test]
    fn test_apr_v2_model_forward_not_transformer() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[1, 2, 3]);
        // Should fail because metadata doesn't indicate transformer
        assert!(result.is_err());
    }

    // =========================================================================
    // decode_tokens Tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_basic() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_decode_tokens_empty_input() {
        let vocab = vec!["hello".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_tokens_out_of_bounds() {
        let vocab = vec!["hello".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 5, 10]);
        // Should contain "hello" and [id] for invalid tokens
        assert!(result.contains("hello"));
        assert!(result.contains("[5]"));
        assert!(result.contains("[10]"));
    }

    #[test]
    fn test_decode_tokens_sentencepiece_prefix() {
        let vocab = vec!["▁hello".to_string(), "▁world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Sentencepiece prefix should be converted to space
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_decode_tokens_empty_vocab() {
        let vocab: Vec<String> = vec![];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        // All tokens out of bounds, formatted as [id]
        assert!(result.contains("[0]"));
        assert!(result.contains("[1]"));
        assert!(result.contains("[2]"));
    }

    // =========================================================================
    // bpe_encode Tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_empty_text() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("", &token_to_id, &merge_rules, &HashMap::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_single_char() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("a", &token_to_id, &merge_rules, &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_unknown_chars() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("xyz", &token_to_id, &merge_rules, &HashMap::new());
        // Unknown chars return empty or default behavior
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_with_merge() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("he".to_string(), 2);
        let merge_rules = vec![("h".to_string(), "e".to_string())];
        let result = bpe_encode("he", &token_to_id, &merge_rules, &HashMap::new());
        // Should merge h+e -> he
        assert!(!result.is_empty());
    }

    // =========================================================================
    // BpeTokenizer Extended Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_whitespace() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert(" ".to_string(), 0);
        token_to_id.insert("a".to_string(), 1);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec![" ".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode(" a ");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode_sentencepiece() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("▁hello".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["▁hello".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0]);
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_bpe_tokenizer_decode_unknown_id() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0, 100, 200]);
        // Should handle out of bounds gracefully
        assert!(decoded.contains("a") || decoded.contains("<unk>"));
    }

    // =========================================================================
    // dequantize_q4_k Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_empty() {
        let result = crate::apr::dequantize_q4_k(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6_k_empty() {
        let result = crate::apr::dequantize_q6_k(&[], 0);
        assert!(result.is_empty());
    }

    // =========================================================================
    // dtype_to_ggml_qtype Tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_f32() {
        // F32 is not a quantized type, returns None
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F32"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_f16() {
        // F16 is not a quantized type, returns None
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F16"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_0() {
        // Q4_0 is qtype 2 in GGML
        let result = crate::apr::dtype_to_ggml_qtype("Q4_0");
        assert!(result.is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q8_0() {
        // Q8_0 is qtype 8 in GGML
        let result = crate::apr::dtype_to_ggml_qtype("Q8_0");
        assert!(result.is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_unknown() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("UNKNOWN"), None);
    }

    // =========================================================================
    // is_quantized_dtype Tests
    // =========================================================================

    #[test]
    fn test_is_quantized_dtype_f32() {
        assert!(!crate::apr::is_quantized_dtype("F32"));
    }

    #[test]
    fn test_is_quantized_dtype_f16() {
        assert!(!crate::apr::is_quantized_dtype("F16"));
    }

    #[test]
    fn test_is_quantized_dtype_q4_0() {
        assert!(crate::apr::is_quantized_dtype("Q4_0"));
    }

    #[test]
    fn test_is_quantized_dtype_q8_0() {
        assert!(crate::apr::is_quantized_dtype("Q8_0"));
    }

    #[test]
    fn test_is_quantized_dtype_q4_k() {
        assert!(crate::apr::is_quantized_dtype("Q4_K"));
    }

    #[test]
    fn test_is_quantized_dtype_q6_k() {
        assert!(crate::apr::is_quantized_dtype("Q6_K"));
    }

    // =========================================================================
    // byte_to_bpe_char Tests
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_ascii() {
        // ASCII printable range
        assert_eq!(crate::apr::byte_to_bpe_char(b'a'), "a");
        assert_eq!(crate::apr::byte_to_bpe_char(b'z'), "z");
        assert_eq!(crate::apr::byte_to_bpe_char(b'0'), "0");
    }

    #[test]
    fn test_byte_to_bpe_char_space() {
        // Space is encoded as Ġ in GPT-2 byte-level BPE
        assert_eq!(crate::apr::byte_to_bpe_char(b' '), "Ġ");
    }

    #[test]
    fn test_byte_to_bpe_char_special() {
        // Control chars get mapped to special unicode
        let result = crate::apr::byte_to_bpe_char(0);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_byte_to_bpe_char_high_byte() {
        let result = crate::apr::byte_to_bpe_char(255);
        assert!(!result.is_empty());
    }

    // =========================================================================
    // rms_norm Tests
    // =========================================================================

    #[test]
    fn test_rms_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.74
        // Each element normalized by RMS
    }

    #[test]
    fn test_rms_norm_zeros() {
        let x = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // All zeros remain zeros
        for &v in &result {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Weight should scale the result
        assert!(result[0] > 1.0);
    }

    // =========================================================================
    // matmul Tests
    // =========================================================================

    #[test]
    fn test_matmul_identity() {
        // 2x2 identity matrix times [1,2,1,2] (2 seq, 2 dim)
        let x = vec![1.0, 2.0, 1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let result = crate::apr::matmul(&x, &w, 2, 2, 2);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_simple() {
        // [1,2] * [[1],[1]] = [3]
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 1.0];
        let result = crate::apr::matmul(&x, &w, 1, 2, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.0).abs() < 1e-6);
    }

    // transpose_matrix is not in public API

    // =========================================================================
    // simd_dot Tests
    // =========================================================================

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = crate::apr::simd_dot(&a, &b);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_zeros() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let result = crate::apr::simd_dot(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_large() {
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 256];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum of 0..255 = 255*256/2 = 32640
        assert!((result - 32640.0).abs() < 1e-3);
    }

    // detect_format and AprFlags tests already exist above
    // f16_to_f32 tests already exist above

    // =========================================================================
    // dequantize_f16 Tests (extended)
    // =========================================================================

    #[test]
    fn test_dequantize_f16_basic() {
        // f16 1.0 = 0x3C00, stored as little-endian [0x00, 0x3C]
        let bytes = vec![0x00, 0x3C, 0x00, 0x3C];
        let result = crate::apr::dequantize_f16(&bytes, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_f16_truncated() {
        let bytes = vec![0x00]; // Only 1 byte, needs 2
        let result = crate::apr::dequantize_f16(&bytes, 1);
        // Truncated input returns empty
        assert_eq!(result.len(), 0);
    }

    // =========================================================================
    // dequantize_q8_0 Tests (extended)
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_zero_scale() {
        // Q8_0 block: 2-byte scale + 32 bytes data
        let mut bytes = vec![0u8; 34];
        // Scale = 0.0 (f16)
        bytes[0] = 0x00;
        bytes[1] = 0x00;
        let result = crate::apr::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // All zeros with zero scale
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    // BpeTokenizer tests already exist above

    // =========================================================================
    // TensorEntry Tests (extended)
    // =========================================================================

    #[test]
    fn test_tensor_entry_byte_size_f32() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            offset: 0,
            size: 24, // 6 elements * 4 bytes
        };
        assert_eq!(entry.element_count(), 6);
    }

    #[test]
    fn test_tensor_entry_byte_size_f16() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F16".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 32, // 16 elements * 2 bytes
        };
        assert_eq!(entry.element_count(), 16);
    }

    // =========================================================================
    // AprMetadata Tests (extended)
    // =========================================================================

    #[test]
    fn test_apr_metadata_is_transformer_missing_heads() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: None,
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_vocab() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: None,
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    // =========================================================================
    // simple_attention Tests
    // =========================================================================

    #[test]
    fn test_simple_attention_single_head() {
        // Very simple case: 1 token, 1 head, head_dim=2
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![1.0, 1.0];
