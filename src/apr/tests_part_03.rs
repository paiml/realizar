//! Additional tests for APR module - Part 03
//!
//! This file provides comprehensive tests for uncovered functions in mod.rs and submodules.
//! Focus: dtype utilities, embedded tokenizer metadata, inference helpers, BPE tokenizer.

#[cfg(test)]
mod tests {
    use crate::apr::*;
    use std::collections::HashMap;

    // =========================================================================
    // Helper: Create binary tensor entry for tests
    // =========================================================================

    fn create_binary_tensor_entry(
        name: &str,
        dtype: u8,
        shape: &[u64],
        offset: u64,
        size: u64,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.push(dtype);
        data.push(shape.len() as u8);
        for &dim in shape {
            data.extend_from_slice(&dim.to_le_bytes());
        }
        data.extend_from_slice(&offset.to_le_bytes());
        data.extend_from_slice(&size.to_le_bytes());
        data
    }

    // =========================================================================
    // dtype_to_ggml_qtype tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_q4_k_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q4_k"), Some(12));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_k_uppercase() {
        assert_eq!(dtype_to_ggml_qtype("Q5_K"), Some(13));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_k_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q5_k"), Some(13));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q6_k_uppercase() {
        assert_eq!(dtype_to_ggml_qtype("Q6_K"), Some(14));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q6_k_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q6_k"), Some(14));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q8_0_uppercase() {
        assert_eq!(dtype_to_ggml_qtype("Q8_0"), Some(8));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q8_0_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q8_0"), Some(8));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_0_uppercase() {
        assert_eq!(dtype_to_ggml_qtype("Q4_0"), Some(2));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_0_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q4_0"), Some(2));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_1() {
        assert_eq!(dtype_to_ggml_qtype("Q4_1"), Some(3));
        assert_eq!(dtype_to_ggml_qtype("q4_1"), Some(3));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_0() {
        assert_eq!(dtype_to_ggml_qtype("Q5_0"), Some(6));
        assert_eq!(dtype_to_ggml_qtype("q5_0"), Some(6));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_f32_returns_none() {
        assert_eq!(dtype_to_ggml_qtype("F32"), None);
        assert_eq!(dtype_to_ggml_qtype("f32"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_f16_returns_none() {
        assert_eq!(dtype_to_ggml_qtype("F16"), None);
        assert_eq!(dtype_to_ggml_qtype("f16"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_unknown_returns_none() {
        assert_eq!(dtype_to_ggml_qtype("UNKNOWN"), None);
        assert_eq!(dtype_to_ggml_qtype("BF16"), None);
    }

    // =========================================================================
    // is_quantized_dtype tests
    // =========================================================================

    #[test]
    fn test_is_quantized_dtype_true_for_q4_k() {
        assert!(is_quantized_dtype("Q4_K"));
        assert!(is_quantized_dtype("q4_k"));
    }

    #[test]
    fn test_is_quantized_dtype_true_for_q8_0() {
        assert!(is_quantized_dtype("Q8_0"));
    }

    #[test]
    fn test_is_quantized_dtype_false_for_f32() {
        assert!(!is_quantized_dtype("F32"));
        assert!(!is_quantized_dtype("f32"));
    }

    #[test]
    fn test_is_quantized_dtype_false_for_bf16() {
        assert!(!is_quantized_dtype("BF16"));
    }

    // =========================================================================
    // TensorEntry from_binary - additional dtype coverage (using GGML dtype IDs)
    // GGML types: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1,
    //   8=Q8_0, 9=Q8_1, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 30=BF16
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_dtype_q5_0() {
        let data = create_binary_tensor_entry("test.q50", 6, &[32], 0, 22);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q5_0");
        assert_eq!(entry.dtype, "Q5_0");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q5_1() {
        let data = create_binary_tensor_entry("test.q51", 7, &[32], 0, 24);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q5_1");
        assert_eq!(entry.dtype, "Q5_1");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q8_1() {
        let data = create_binary_tensor_entry("test.q81", 9, &[32], 0, 36);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q8_1");
        assert_eq!(entry.dtype, "Q8_1");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q2_k() {
        let data = create_binary_tensor_entry("test.q2k", 10, &[256], 0, 84);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q2_k");
        assert_eq!(entry.dtype, "Q2_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q4_k() {
        let data = create_binary_tensor_entry("test.q4k", 12, &[256], 0, 144);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q4k");
        assert_eq!(entry.dtype, "Q4_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q6_k() {
        let data = create_binary_tensor_entry("test.q6k", 14, &[256], 0, 210);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q6k");
        assert_eq!(entry.dtype, "Q6_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q8_0() {
        let data = create_binary_tensor_entry("test.q80", 8, &[32], 0, 34);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q8_0");
        assert_eq!(entry.dtype, "Q8_0");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_unknown_defaults_f32() {
        let data = create_binary_tensor_entry("test.unknown", 255, &[100], 0, 400);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse unknown");
        assert_eq!(entry.dtype, "F32");
    }

    // =========================================================================
    // AprMetadata embedded tokenizer tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_present() {
        let mut extra = HashMap::new();
        extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["hello", "world", "test"]),
        );
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };

        let vocab = meta.get_embedded_vocabulary();
        assert!(vocab.is_some());
        let v = vocab.unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], "hello");
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_missing() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_empty_array() {
        let mut extra = HashMap::new();
        extra.insert("tokenizer.vocabulary".to_string(), serde_json::json!([]));
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_not_array() {
        let mut extra = HashMap::new();
        extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!("not an array"),
        );
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_present() {
        let mut extra = HashMap::new();
        extra.insert("tokenizer.bos_token_id".to_string(), serde_json::json!(1));
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };
        assert_eq!(meta.get_embedded_bos_token_id(), Some(1));
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_missing() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_bos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_present() {
        let mut extra = HashMap::new();
        extra.insert("tokenizer.eos_token_id".to_string(), serde_json::json!(2));
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };
        assert_eq!(meta.get_embedded_eos_token_id(), Some(2));
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_missing() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_eos_token_id().is_none());
    }

    // =========================================================================
    // SimpleTokenizer tests
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_new() {
        let vocab = vec!["<s>".to_string(), "hello".to_string(), "</s>".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(2));

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.bos_token_id, Some(0));
        assert_eq!(tokenizer.eos_token_id, Some(2));
    }

    #[test]
    fn test_simple_tokenizer_vocab_size() {
        let tokenizer = SimpleTokenizer::new(vec!["a".to_string(), "b".to_string()], None, None);
        assert_eq!(tokenizer.vocab_size(), 2);
    }

    #[test]
    fn test_simple_tokenizer_is_eos_true() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], None, Some(42));
        assert!(tokenizer.is_eos(42));
    }

    #[test]
    fn test_simple_tokenizer_is_eos_false() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], None, Some(42));
        assert!(!tokenizer.is_eos(99));
    }

    #[test]
    fn test_simple_tokenizer_is_eos_none() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], None, None);
        assert!(!tokenizer.is_eos(0));
    }

    #[test]
    fn test_simple_tokenizer_is_bos_true() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], Some(10), None);
        assert!(tokenizer.is_bos(10));
    }

    #[test]
    fn test_simple_tokenizer_is_bos_false() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], Some(10), None);
        assert!(!tokenizer.is_bos(99));
    }

    #[test]
    fn test_simple_tokenizer_is_bos_none() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], None, None);
        assert!(!tokenizer.is_bos(0));
    }

    // =========================================================================
    // BpeTokenizer tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_simple() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("l".to_string(), 2);
        token_to_id.insert("o".to_string(), 3);
        token_to_id.insert("he".to_string(), 4);
        token_to_id.insert("ll".to_string(), 5);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["h", "e", "l", "o", "he", "ll"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            merge_rules: vec![
                ("h".to_string(), "e".to_string()),
                ("l".to_string(), "l".to_string()),
            ],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let ids = tokenizer.encode("hello");
        // "hello" -> ['h', 'e', 'l', 'l', 'o'] -> ['he', 'l', 'l', 'o'] -> ['he', 'll', 'o']
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec!["hello".to_string(), "Ġworld".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let text = tokenizer.decode(&[0, 1]);
        assert_eq!(text, "hello world");
    }

    // =========================================================================
    // byte_to_bpe_char tests
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_space() {
        assert_eq!(byte_to_bpe_char(b' '), "Ġ");
    }

    #[test]
    fn test_byte_to_bpe_char_newline() {
        assert_eq!(byte_to_bpe_char(b'\n'), "Ċ");
    }

    #[test]
    fn test_byte_to_bpe_char_tab() {
        assert_eq!(byte_to_bpe_char(b'\t'), "ĉ");
    }

    #[test]
    fn test_byte_to_bpe_char_ascii_letter() {
        assert_eq!(byte_to_bpe_char(b'a'), "a");
        assert_eq!(byte_to_bpe_char(b'Z'), "Z");
    }

    #[test]
    fn test_byte_to_bpe_char_ascii_digit() {
        assert_eq!(byte_to_bpe_char(b'0'), "0");
        assert_eq!(byte_to_bpe_char(b'9'), "9");
    }

    #[test]
    fn test_byte_to_bpe_char_punctuation() {
        assert_eq!(byte_to_bpe_char(b'!'), "!");
        assert_eq!(byte_to_bpe_char(b'.'), ".");
    }

    #[test]
    fn test_byte_to_bpe_char_non_printable() {
        // Non-printable control character
        let result = byte_to_bpe_char(0x01);
        assert!(result.starts_with("<0x"));
        assert!(result.ends_with('>'));
    }

    #[test]
    fn test_byte_to_bpe_char_high_byte() {
        let result = byte_to_bpe_char(0xFF);
        assert_eq!(result, "<0xFF>");
    }

    // =========================================================================
    // AprV2Model::decode_tokens tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_basic() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let text = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        assert_eq!(text, "helloworld");
    }

    #[test]
    fn test_decode_tokens_with_bpe_special_chars() {
        let vocab = vec!["Ġhello".to_string(), "Ċ".to_string(), "ĉtab".to_string()];
        let text = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert_eq!(text, " hello\n\ttab");
    }

    #[test]
    fn test_decode_tokens_out_of_bounds() {
        let vocab = vec!["test".to_string()];
        let text = AprV2Model::decode_tokens(&vocab, &[0, 100]);
        assert!(text.contains("test"));
        assert!(text.contains("[100]"));
    }

    #[test]
    fn test_decode_tokens_empty() {
        let vocab = vec!["test".to_string()];
        let text = AprV2Model::decode_tokens(&vocab, &[]);
        assert!(text.is_empty());
    }

    // =========================================================================
    // AprV2Model predict tests (linear model inference)
    // =========================================================================

    #[test]
    fn test_predict_empty_model() {
        // Create minimal APR with no tensors
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        let model = AprV2Model::from_bytes(data).expect("load model");
        let output = model.predict(&[1.0, 2.0, 3.0]).expect("predict");

        // Empty model sums inputs
        assert_eq!(output.len(), 1);
        assert!((output[0] - 6.0).abs() < 0.001);
    }

    // =========================================================================
    // AprV2Model forward error cases
    // =========================================================================

    #[test]
    fn test_forward_empty_tokens() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        let model = AprV2Model::from_bytes(data).expect("load");
        let result = model.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_non_transformer() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        let model = AprV2Model::from_bytes(data).expect("load");
        let result = model.forward(&[1, 2, 3]);
        assert!(result.is_err());
    }

    // =========================================================================
    // AprV2Model generate error case
    // =========================================================================

    #[test]
    fn test_generate_empty_input() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        let model = AprV2Model::from_bytes(data).expect("load");
        let result = model.generate(&[], 10, None);
        assert!(result.is_err());
    }

    // =========================================================================
    // simd_dot tests
    // =========================================================================

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = simd_dot(&a, &b);
        assert!((result - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_empty() {
        let result = simd_dot(&[], &[]);
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let result = simd_dot(&a, &b);
        // Only first 2 elements are used
        assert!((result - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_large_vector() {
        // Test AVX2 path with >8 elements
        let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 32];
        let result = simd_dot(&a, &b);
        // sum of 0..31 = 496
        assert!((result - 496.0).abs() < 0.01);
    }

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_negative_zero() {
        // Negative zero: sign=1, exp=0, mant=0 -> 0x8000
        let result = f16_to_f32(0x8000);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_one() {
        // f16 for 1.0 = 0x3C00
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        // f16 for -1.0 = 0xBC00
        let result = f16_to_f32(0xBC00);
        assert!((result + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_positive_inf() {
        // +Inf: sign=0, exp=31, mant=0 -> 0x7C00
        let result = f16_to_f32(0x7C00);
        assert!(result.is_infinite());
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_f16_to_f32_negative_inf() {
        // -Inf: sign=1, exp=31, mant=0 -> 0xFC00
        let result = f16_to_f32(0xFC00);
        assert!(result.is_infinite());
        assert!(result.is_sign_negative());
    }
}
