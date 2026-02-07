//! T-COV-95 Coverage Bridge: apr/mod.rs Part 05
//!
//! Targets uncovered lines: AprFlags methods (is_compressed, is_lz4, is_zstd,
//! is_encrypted, is_quantized, has_vocab), AprHeader::from_bytes edge cases,
//! TensorEntry::from_binary edge cases, TensorEntry::element_count,
//! AprMetadata methods, AprV2Model::from_bytes, decode_tokens,
//! MappedAprModel::dtype_to_qtype, extract_special_tokens_from_vocab.

use std::collections::HashMap;

use crate::apr::*;

// ============================================================================
// AprFlags tests
// ============================================================================

#[test]
fn test_apr_flags_new() {
    let flags = AprFlags::new(0);
    assert!(!flags.is_compressed());
    assert!(!flags.is_lz4());
    assert!(!flags.is_zstd());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_quantized());
    assert!(!flags.has_vocab());
}

#[test]
fn test_apr_flags_lz4() {
    let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
    assert!(flags.is_lz4());
    assert!(flags.is_compressed());
    assert!(!flags.is_zstd());
    assert!(!flags.is_encrypted());
}

#[test]
fn test_apr_flags_zstd() {
    let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
    assert!(flags.is_zstd());
    assert!(flags.is_compressed());
    assert!(!flags.is_lz4());
}

#[test]
fn test_apr_flags_both_compression() {
    let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED | AprFlags::ZSTD_COMPRESSED);
    assert!(flags.is_compressed());
    assert!(flags.is_lz4());
    assert!(flags.is_zstd());
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
}

#[test]
fn test_apr_flags_has_vocab() {
    let flags = AprFlags::new(AprFlags::HAS_VOCAB);
    assert!(flags.has_vocab());
}

#[test]
fn test_apr_flags_all_set() {
    let flags = AprFlags::new(
        AprFlags::LZ4_COMPRESSED
            | AprFlags::ZSTD_COMPRESSED
            | AprFlags::ENCRYPTED
            | AprFlags::QUANTIZED
            | AprFlags::HAS_VOCAB
            | AprFlags::SIGNED
            | AprFlags::SHARDED,
    );
    assert!(flags.is_compressed());
    assert!(flags.is_lz4());
    assert!(flags.is_zstd());
    assert!(flags.is_encrypted());
    assert!(flags.is_quantized());
    assert!(flags.has_vocab());
}

#[test]
fn test_apr_flags_default() {
    let flags = AprFlags::default();
    assert!(!flags.is_compressed());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_quantized());
    assert!(!flags.has_vocab());
}

#[test]
fn test_apr_flags_debug() {
    let flags = AprFlags::new(0x0021);
    let debug = format!("{:?}", flags);
    assert!(debug.contains("AprFlags"));
}

#[test]
fn test_apr_flags_clone_copy() {
    let flags = AprFlags::new(0x0005);
    let cloned = flags;
    assert_eq!(cloned.is_encrypted(), flags.is_encrypted());
}

// ============================================================================
// AprHeader::from_bytes tests
// ============================================================================

#[test]
fn test_apr_header_too_small() {
    let data = vec![0u8; 32]; // Less than 64 bytes
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("too small") || err.contains("header"));
}

#[test]
fn test_apr_header_wrong_magic() {
    let mut data = vec![0u8; 64];
    data[0] = 0x00; // Not APR
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("magic") || err.contains("Invalid"));
}

#[test]
fn test_apr_header_v1_rejected() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]); // APR1
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("v1") || err.contains("not supported"));
}

#[test]
fn test_apr_header_v2_accepted() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x32]); // APR2
    data[4] = 2; // version major
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_apr_header_legacy_magic_accepted() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x00]); // APR\0
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_apr_header_invalid_version_byte() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x33]); // APR3 (invalid)
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_apr_header_parses_fields() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // version major
    data[5] = 1; // version minor
    data[6..8].copy_from_slice(&0x0020u16.to_le_bytes()); // flags: QUANTIZED
    data[8..12].copy_from_slice(&10u32.to_le_bytes()); // tensor_count
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&100u32.to_le_bytes()); // metadata_size
    data[24..32].copy_from_slice(&200u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&500u64.to_le_bytes()); // data_offset
    data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes()); // checksum

    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.version, (2, 1));
    assert!(header.flags.is_quantized());
    assert_eq!(header.tensor_count, 10);
    assert_eq!(header.metadata_offset, 64);
    assert_eq!(header.metadata_size, 100);
    assert_eq!(header.tensor_index_offset, 200);
    assert_eq!(header.data_offset, 500);
    assert_eq!(header.checksum, 0xDEADBEEF);
}

#[test]
fn test_apr_header_debug() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&MAGIC);
    let header = AprHeader::from_bytes(&data).expect("should parse");
    let debug = format!("{:?}", header);
    assert!(debug.contains("AprHeader"));
}

#[test]
fn test_apr_header_clone() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&MAGIC);
    let header = AprHeader::from_bytes(&data).expect("should parse");
    let cloned = header.clone();
    assert_eq!(cloned.tensor_count, header.tensor_count);
}

// ============================================================================
// TensorEntry::from_binary tests
// ============================================================================

fn make_tensor_entry_binary(
    name: &str,
    dtype: u8,
    ndim: u8,
    dims: &[u64],
    offset: u64,
    size: u64,
) -> Vec<u8> {
    let mut data = Vec::new();
    let name_bytes = name.as_bytes();
    data.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
    data.extend_from_slice(name_bytes);
    data.push(dtype);
    data.push(ndim);
    for &dim in dims {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&offset.to_le_bytes());
    data.extend_from_slice(&size.to_le_bytes());
    data
}

#[test]
fn test_tensor_entry_from_binary_f32() {
    let data = make_tensor_entry_binary("test.weight", 0, 2, &[100, 64], 0, 25600);
    let (entry, consumed) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.name, "test.weight");
    assert_eq!(entry.dtype, "F32");
    assert_eq!(entry.shape, vec![100, 64]);
    assert_eq!(entry.offset, 0);
    assert_eq!(entry.size, 25600);
    assert_eq!(consumed, data.len());
}

#[test]
fn test_tensor_entry_from_binary_f16() {
    let data = make_tensor_entry_binary("embed", 1, 2, &[32000, 512], 0, 32768000);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "F16");
}

#[test]
fn test_tensor_entry_from_binary_q4k() {
    let data = make_tensor_entry_binary("layer.0.qkv", 12, 2, &[1024, 512], 100, 50000);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q4_K");
}

#[test]
fn test_tensor_entry_from_binary_q6k() {
    let data = make_tensor_entry_binary("lm_head", 14, 2, &[32000, 512], 0, 100000);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q6_K");
}

#[test]
fn test_tensor_entry_from_binary_q8_0() {
    let data = make_tensor_entry_binary("layer.0.ffn", 8, 2, &[256, 64], 0, 5000);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q8_0");
}

#[test]
fn test_tensor_entry_from_binary_bf16() {
    let data = make_tensor_entry_binary("bf16_tensor", 30, 1, &[1024], 0, 2048);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "BF16");
}

#[test]
fn test_tensor_entry_from_binary_unknown_dtype() {
    let data = make_tensor_entry_binary("unknown", 255, 1, &[100], 0, 400);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "F32"); // Unknown defaults to F32
}

#[test]
fn test_tensor_entry_from_binary_1d() {
    let data = make_tensor_entry_binary("bias", 0, 1, &[512], 0, 2048);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.shape, vec![512]);
}

#[test]
fn test_tensor_entry_from_binary_truncated_too_short() {
    let data = vec![0u8; 3]; // Less than 4 bytes
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_from_binary_truncated_at_name() {
    let data = vec![0x10, 0x00]; // name_len=16 but no name data
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_from_binary_truncated_at_shape() {
    // name_len=1, name="a", dtype=0, ndim=2, but no dim data
    let data = vec![0x01, 0x00, b'a', 0x00, 0x02];
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_from_binary_all_qtypes() {
    let qtypes = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q5_0"),
        (7, "Q5_1"),
        (8, "Q8_0"),
        (9, "Q8_1"),
        (10, "Q2_K"),
        (11, "Q3_K"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
        (16, "IQ2_XXS"),
        (17, "IQ2_XS"),
        (30, "BF16"),
    ];

    for (dtype_byte, expected_name) in qtypes {
        let data = make_tensor_entry_binary("t", dtype_byte, 1, &[100], 0, 400);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(
            entry.dtype, expected_name,
            "dtype byte {} should map to {}",
            dtype_byte, expected_name
        );
    }
}

// ============================================================================
// TensorEntry::element_count
// ============================================================================

#[test]
fn test_tensor_entry_element_count_2d() {
    let entry = TensorEntry {
        name: "test".to_string(),
        dtype: "F32".to_string(),
        shape: vec![100, 64],
        offset: 0,
        size: 25600,
    };
    assert_eq!(entry.element_count(), 6400);
}

#[test]
fn test_tensor_entry_element_count_1d() {
    let entry = TensorEntry {
        name: "bias".to_string(),
        dtype: "F32".to_string(),
        shape: vec![512],
        offset: 0,
        size: 2048,
    };
    assert_eq!(entry.element_count(), 512);
}

#[test]
fn test_tensor_entry_element_count_empty_shape() {
    let entry = TensorEntry {
        name: "scalar".to_string(),
        dtype: "F32".to_string(),
        shape: vec![],
        offset: 0,
        size: 4,
    };
    assert_eq!(entry.element_count(), 1); // Product of empty = 1
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

// ============================================================================
// AprMetadata tests
// ============================================================================

#[test]
fn test_apr_metadata_default() {
    let m = AprMetadata::default();
    assert!(m.model_type.is_none());
    assert!(m.name.is_none());
    assert!(m.architecture.is_none());
    assert!(m.hidden_size.is_none());
    assert!(m.num_layers.is_none());
    assert!(m.num_heads.is_none());
    assert!(m.vocab_size.is_none());
}

#[test]
fn test_apr_metadata_is_transformer_true() {
    let m = AprMetadata {
        hidden_size: Some(512),
        num_layers: Some(6),
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(m.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_false_missing_hidden() {
    let m = AprMetadata {
        num_layers: Some(6),
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!m.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_false_missing_layers() {
    let m = AprMetadata {
        hidden_size: Some(512),
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!m.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_false_missing_heads() {
    let m = AprMetadata {
        hidden_size: Some(512),
        num_layers: Some(6),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!m.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_false_missing_vocab() {
    let m = AprMetadata {
        hidden_size: Some(512),
        num_layers: Some(6),
        num_heads: Some(8),
        ..Default::default()
    };
    assert!(!m.is_transformer());
}

// ============================================================================
// AprMetadata embedded tokenizer tests
// ============================================================================

#[test]
fn test_get_embedded_vocabulary_present() {
    let mut extra = HashMap::new();
    extra.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["hello", "world", "test"]),
    );
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    let vocab = m.get_embedded_vocabulary();
    assert!(vocab.is_some());
    assert_eq!(vocab.unwrap(), vec!["hello", "world", "test"]);
}

#[test]
fn test_get_embedded_vocabulary_missing() {
    let m = AprMetadata::default();
    assert!(m.get_embedded_vocabulary().is_none());
}

#[test]
fn test_get_embedded_vocabulary_empty_array() {
    let mut extra = HashMap::new();
    extra.insert("tokenizer.vocabulary".to_string(), serde_json::json!([]));
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert!(m.get_embedded_vocabulary().is_none());
}

#[test]
fn test_get_embedded_vocabulary_not_array() {
    let mut extra = HashMap::new();
    extra.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!("not_array"),
    );
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert!(m.get_embedded_vocabulary().is_none());
}

#[test]
fn test_get_embedded_bos_token_id_present() {
    let mut extra = HashMap::new();
    extra.insert("tokenizer.bos_token_id".to_string(), serde_json::json!(1));
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert_eq!(m.get_embedded_bos_token_id(), Some(1));
}

#[test]
fn test_get_embedded_bos_token_id_missing() {
    let m = AprMetadata::default();
    assert!(m.get_embedded_bos_token_id().is_none());
}

#[test]
fn test_get_embedded_eos_token_id_present() {
    let mut extra = HashMap::new();
    extra.insert("tokenizer.eos_token_id".to_string(), serde_json::json!(2));
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert_eq!(m.get_embedded_eos_token_id(), Some(2));
}

#[test]
fn test_get_embedded_eos_token_id_missing() {
    let m = AprMetadata::default();
    assert!(m.get_embedded_eos_token_id().is_none());
}

// ============================================================================
// AprMetadata::get_embedded_merges
// ============================================================================

#[test]
fn test_get_embedded_merges_present() {
    let mut extra = HashMap::new();
    extra.insert(
        "tokenizer.merges".to_string(),
        serde_json::json!(["a b", "c d", "ef gh"]),
    );
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    let merges = m.get_embedded_merges();
    assert!(merges.is_some());
    let merges = merges.unwrap();
    assert_eq!(merges.len(), 3);
    assert_eq!(merges[0], ("a".to_string(), "b".to_string()));
    assert_eq!(merges[1], ("c".to_string(), "d".to_string()));
    assert_eq!(merges[2], ("ef".to_string(), "gh".to_string()));
}

#[test]
fn test_get_embedded_merges_missing() {
    let m = AprMetadata::default();
    assert!(m.get_embedded_merges().is_none());
}

#[test]
fn test_get_embedded_merges_empty() {
    let mut extra = HashMap::new();
    extra.insert("tokenizer.merges".to_string(), serde_json::json!([]));
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert!(m.get_embedded_merges().is_none());
}

#[test]
fn test_get_embedded_merges_invalid_format() {
    let mut extra = HashMap::new();
    // Single words (no space separator) should be skipped
    extra.insert(
        "tokenizer.merges".to_string(),
        serde_json::json!(["nospace", "a b"]),
    );
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    let merges = m.get_embedded_merges();
    assert!(merges.is_some());
    // Only "a b" should be parsed (nospace has no space)
    assert_eq!(merges.unwrap().len(), 1);
}

// ============================================================================
// AprMetadata serde aliases
// ============================================================================

#[test]
fn test_apr_metadata_hidden_dim_alias() {
    let json = r#"{"hidden_dim": 1024}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.hidden_size, Some(1024));
}

#[test]
fn test_apr_metadata_num_hidden_layers_alias() {
    let json = r#"{"num_hidden_layers": 12}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_layers, Some(12));
}

#[test]
fn test_apr_metadata_num_attention_heads_alias() {
    let json = r#"{"num_attention_heads": 16}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_heads, Some(16));
}

#[test]
fn test_apr_metadata_d_model_alias() {
    let json = r#"{"d_model": 768}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.hidden_size, Some(768));
}

#[test]
fn test_apr_metadata_n_vocab_alias() {
    let json = r#"{"n_vocab": 50257}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.vocab_size, Some(50257));
}

#[test]
fn test_apr_metadata_intermediate_dim_alias() {
    let json = r#"{"intermediate_dim": 2048}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.intermediate_size, Some(2048));
}

#[test]
fn test_apr_metadata_context_length_alias() {
    let json = r#"{"context_length": 4096}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.max_position_embeddings, Some(4096));
}

#[test]
fn test_apr_metadata_norm_eps_alias() {
    let json = r#"{"norm_eps": 0.00001}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert!(m.rms_norm_eps.is_some());
}

// ============================================================================
// AprV2Model::from_bytes tests
// ============================================================================

#[test]
fn test_apr_v2_model_from_bytes_encrypted_rejected() {
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[6..8].copy_from_slice(&AprFlags::ENCRYPTED.to_le_bytes()); // flags: encrypted
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
    data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("ncrypt"));
}

#[test]
fn test_apr_v2_model_from_bytes_minimal_valid() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata_size = 2
    data[24..32].copy_from_slice(&128u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&128u64.to_le_bytes()); // data_offset
                                                         // Put valid JSON metadata at offset 64
    data[64] = b'{';
    data[65] = b'}';

    let model = AprV2Model::from_bytes(data).expect("should parse");
    assert_eq!(model.tensor_count(), 0);
    assert!(model.tensor_names().is_empty());
}

#[test]
fn test_apr_v2_model_from_bytes_with_metadata() {
    let metadata = serde_json::json!({
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "vocab_size": 32000,
        "architecture": "llama"
    });
    let meta_bytes = serde_json::to_vec(&metadata).unwrap();
    let meta_padded = ((meta_bytes.len() + 63) / 64) * 64;

    let total_size = 64 + meta_padded;
    let mut data = vec![0u8; total_size];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&(total_size as u64).to_le_bytes());
    data[32..40].copy_from_slice(&(total_size as u64).to_le_bytes());
    data[64..64 + meta_bytes.len()].copy_from_slice(&meta_bytes);

    let model = AprV2Model::from_bytes(data).expect("should parse");
    let meta = model.metadata();
    assert!(meta.is_transformer());
    assert_eq!(meta.hidden_size, Some(512));
    assert_eq!(meta.architecture, Some("llama".to_string()));
}

// ============================================================================
// decode_tokens
// ============================================================================

#[test]
fn test_decode_tokens_basic() {
    let vocab = vec!["hello".to_string(), "Ġworld".to_string(), "!".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "hello world!");
}

#[test]
fn test_decode_tokens_special_chars() {
    let vocab = vec![
        "Ċ".to_string(),      // \n
        "ĉ".to_string(),      // \t
        "Ġhello".to_string(), // space + hello
    ];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "\n\t hello");
}

#[test]
fn test_decode_tokens_out_of_vocab() {
    let vocab = vec!["hello".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 99]);
    assert_eq!(result, "hello[99]");
}

#[test]
fn test_decode_tokens_empty() {
    let vocab = vec!["hello".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[]);
    assert_eq!(result, "");
}

#[test]
fn test_decode_tokens_empty_vocab() {
    let vocab: Vec<String> = vec![];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "[0][1][2]");
}

// ============================================================================
// MappedAprModel::dtype_to_qtype
// ============================================================================

#[test]
fn test_dtype_to_qtype_all_variants() {
    assert_eq!(MappedAprModel::dtype_to_qtype("F32"), 0);
    assert_eq!(MappedAprModel::dtype_to_qtype("F16"), 1);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_0"), 2);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_1"), 3);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_0"), 6);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_1"), 7);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q8_0"), 8);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q8_1"), 9);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q2_K"), 10);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q3_K"), 11);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_K"), 12);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q6_K"), 14);
    assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XXS"), 16);
    assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XS"), 17);
    assert_eq!(MappedAprModel::dtype_to_qtype("BF16"), 30);
    assert_eq!(MappedAprModel::dtype_to_qtype("UNKNOWN"), 0); // Default
}

// ============================================================================
// extract_special_tokens_from_vocab
// ============================================================================

#[test]
fn test_extract_special_tokens_known_patterns() {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<|im_start|>".to_string(), 151643);
    vocab.insert("<|im_end|>".to_string(), 151644);
    vocab.insert("<|endoftext|>".to_string(), 151645);
    vocab.insert("<s>".to_string(), 1);
    vocab.insert("</s>".to_string(), 2);
    vocab.insert("<pad>".to_string(), 0);
    vocab.insert("<unk>".to_string(), 3);
    vocab.insert("hello".to_string(), 100); // Not special

    let specials = extract_special_tokens_from_vocab(&vocab);
    assert!(specials.contains_key("<|im_start|>"));
    assert!(specials.contains_key("<|im_end|>"));
    assert!(specials.contains_key("<|endoftext|>"));
    assert!(specials.contains_key("<s>"));
    assert!(specials.contains_key("</s>"));
    assert!(specials.contains_key("<pad>"));
    assert!(specials.contains_key("<unk>"));
    assert!(!specials.contains_key("hello"));
}

#[test]
fn test_extract_special_tokens_custom_pattern() {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<|custom_token|>".to_string(), 999);
    vocab.insert("regular_token".to_string(), 100);

    let specials = extract_special_tokens_from_vocab(&vocab);
    // <|custom_token|> matches <|...|> pattern
    assert!(specials.contains_key("<|custom_token|>"));
    assert!(!specials.contains_key("regular_token"));
}

#[test]
fn test_extract_special_tokens_empty_vocab() {
    let vocab: HashMap<String, u32> = HashMap::new();
    let specials = extract_special_tokens_from_vocab(&vocab);
    assert!(specials.is_empty());
}

#[test]
fn test_extract_special_tokens_code_model() {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<|fim_prefix|>".to_string(), 100);
    vocab.insert("<|fim_middle|>".to_string(), 101);
    vocab.insert("<|fim_suffix|>".to_string(), 102);

    let specials = extract_special_tokens_from_vocab(&vocab);
    assert!(specials.contains_key("<|fim_prefix|>"));
    assert!(specials.contains_key("<|fim_middle|>"));
    assert!(specials.contains_key("<|fim_suffix|>"));
}

// ============================================================================
// AprMetadata serialization roundtrip
// ============================================================================

#[test]
fn test_apr_metadata_serialization_roundtrip() {
    let m = AprMetadata {
        model_type: Some("transformer_lm".to_string()),
        architecture: Some("qwen2".to_string()),
        hidden_size: Some(2048),
        num_layers: Some(24),
        num_heads: Some(16),
        num_kv_heads: Some(4),
        vocab_size: Some(152064),
        intermediate_size: Some(8192),
        max_position_embeddings: Some(32768),
        rope_theta: Some(1000000.0),
        rope_type: Some(2),
        rms_norm_eps: Some(1e-6),
        ..Default::default()
    };

    let json = serde_json::to_string(&m).expect("serialize");
    let restored: AprMetadata = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.hidden_size, Some(2048));
    assert_eq!(restored.architecture, Some("qwen2".to_string()));
    assert!(restored.is_transformer());
}

#[test]
fn test_apr_metadata_debug() {
    let m = AprMetadata::default();
    let debug = format!("{:?}", m);
    assert!(debug.contains("AprMetadata"));
}

#[test]
fn test_apr_metadata_clone() {
    let m = AprMetadata {
        hidden_size: Some(512),
        ..Default::default()
    };
    let cloned = m.clone();
    assert_eq!(cloned.hidden_size, Some(512));
}
