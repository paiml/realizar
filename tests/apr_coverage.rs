//! EXTREME TDD coverage tests for apr.rs module (Refs PMAT-802)
//!
//! Focus areas:
//! - MappedAprModel (memory-mapped loading)
//! - AprV2Model edge cases
//! - Error handling paths
//! - BpeTokenizer edge cases
//! - Dequantization functions
//!
//! Target: 85%+ coverage for apr.rs

use realizar::apr::{
    detect_format, is_apr_file, AprFlags, AprHeader, AprMetadata, AprV2Model, BpeTokenizer,
    MappedAprModel, TensorEntry, ALIGNMENT, HEADER_SIZE, MAGIC,
};
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper to create binary tensor entry for APR v2 format
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

/// Helper to create minimal valid APR v2 model bytes
fn create_minimal_apr_model() -> Vec<u8> {
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let tensor_entry = create_binary_tensor_entry("test.weight", 0, &[4, 4], 0, 64);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let data_size = 64usize;
    let total_size = data_offset as usize + data_size;
    let mut data = vec![0u8; total_size];

    // Header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&0u16.to_le_bytes());
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    // Tensor data (16 floats)
    let data_start = data_offset as usize;
    for i in 0..16 {
        let val = i as f32;
        data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    data
}

/// Helper to create APR file on disk
fn create_apr_file() -> NamedTempFile {
    let mut temp = NamedTempFile::new().expect("create temp file");
    let data = create_minimal_apr_model();
    temp.write_all(&data).expect("write data");
    temp
}

// ============================================================================
// MappedAprModel Coverage Tests
// ============================================================================

#[test]
fn test_mapped_apr_model_from_path_valid() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.tensor_count(), 1);
    assert!(model.file_size() > 0);
}

#[test]
fn test_mapped_apr_model_from_path_nonexistent() {
    let result = MappedAprModel::from_path("/nonexistent/path/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_mapped_apr_model_data_accessor() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let data = model.data();
    assert!(!data.is_empty());
    assert_eq!(&data[0..4], &MAGIC);
}

#[test]
fn test_mapped_apr_model_file_size() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert!(model.file_size() >= HEADER_SIZE);
}

#[test]
fn test_mapped_apr_model_tensor_count() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.tensor_count(), 1);
}

#[test]
fn test_mapped_apr_model_data_offset() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert!(model.data_offset() > 0);
}

#[test]
fn test_mapped_apr_model_find_tensor_exists() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let tensor = model.find_tensor("test.weight");
    assert!(tensor.is_some());
    let entry = tensor.unwrap();
    assert_eq!(entry.name, "test.weight");
}

#[test]
fn test_mapped_apr_model_find_tensor_not_exists() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let tensor = model.find_tensor("nonexistent");
    assert!(tensor.is_none());
}

#[test]
fn test_mapped_apr_model_get_tensor_data_valid() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let data = model.get_tensor_data("test.weight").expect("should get data");
    assert_eq!(data.len(), 64);
}

#[test]
fn test_mapped_apr_model_get_tensor_data_not_found() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let result = model.get_tensor_data("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_f32() {
    assert_eq!(MappedAprModel::dtype_to_qtype("F32"), 0);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_f16() {
    assert_eq!(MappedAprModel::dtype_to_qtype("F16"), 1);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q4_0() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_0"), 2);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q4_1() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_1"), 3);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q5_0() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_0"), 6);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q5_1() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_1"), 7);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q8_0() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q8_0"), 8);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q8_1() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q8_1"), 9);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q2_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q2_K"), 10);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q3_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q3_K"), 11);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q4_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_K"), 12);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q5_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q6_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q6_K"), 14);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_iq2_xxs() {
    assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XXS"), 16);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_iq2_xs() {
    assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XS"), 17);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_bf16() {
    assert_eq!(MappedAprModel::dtype_to_qtype("BF16"), 30);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_unknown() {
    assert_eq!(MappedAprModel::dtype_to_qtype("UNKNOWN"), 0);
}

#[test]
fn test_mapped_apr_model_invalid_magic() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"GGUF"); // Wrong magic
    data[4] = 2;
    data[5] = 0;
    temp.write_all(&data).expect("write data");

    let result = MappedAprModel::from_path(temp.path());
    assert!(result.is_err());
}

#[test]
fn test_mapped_apr_model_header_fields() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.header.magic, MAGIC);
    assert_eq!(model.header.version.0, 2);
}

#[test]
fn test_mapped_apr_model_metadata_fields() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.metadata.vocab_size, Some(100));
    assert_eq!(model.metadata.hidden_size, Some(64));
}

#[test]
fn test_mapped_apr_model_tensors_field() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].name, "test.weight");
}

// ============================================================================
// AprV2Model Load/from_bytes Coverage
// ============================================================================

#[test]
fn test_apr_v2_model_load_from_file() {
    let temp = create_apr_file();
    let model = AprV2Model::load(temp.path()).expect("should load");
    assert_eq!(model.tensor_count(), 1);
    assert!(model.is_mmap());
}

#[test]
fn test_apr_v2_model_load_nonexistent() {
    let result = AprV2Model::load("/nonexistent/path/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_from_bytes_valid() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.tensor_count(), 1);
    assert!(!model.is_mmap());
}

#[test]
fn test_apr_v2_model_from_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX");
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_from_bytes_truncated() {
    let data = vec![0u8; 10];
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

// ============================================================================
// AprHeader Coverage
// ============================================================================

#[test]
fn test_apr_header_from_bytes_valid() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 1;
    data[8..12].copy_from_slice(&42u32.to_le_bytes());

    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.version, (2, 1));
    assert_eq!(header.tensor_count, 42);
}

#[test]
fn test_apr_header_from_bytes_too_small() {
    let data = vec![0u8; 10];
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_apr_header_from_bytes_wrong_magic() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(b"TEST");
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_apr_header_checksum_field() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.checksum, 0xDEADBEEF);
}

// ============================================================================
// AprFlags Coverage
// ============================================================================

#[test]
fn test_apr_flags_new() {
    let flags = AprFlags::new(0x0000);
    assert!(!flags.is_compressed());
    assert!(!flags.is_encrypted());
}

#[test]
fn test_apr_flags_lz4() {
    let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
    assert!(flags.is_lz4());
    assert!(flags.is_compressed());
    assert!(!flags.is_zstd());
}

#[test]
fn test_apr_flags_zstd() {
    let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
    assert!(flags.is_zstd());
    assert!(flags.is_compressed());
    assert!(!flags.is_lz4());
}

#[test]
fn test_apr_flags_encrypted() {
    let flags = AprFlags::new(AprFlags::ENCRYPTED);
    assert!(flags.is_encrypted());
    assert!(!flags.is_compressed());
}

#[test]
fn test_apr_flags_signed() {
    let flags = AprFlags::new(AprFlags::SIGNED);
    assert!(!flags.is_encrypted());
}

#[test]
fn test_apr_flags_sharded() {
    let flags = AprFlags::new(AprFlags::SHARDED);
    assert!(!flags.is_quantized());
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
fn test_apr_flags_combined() {
    let flags =
        AprFlags::new(AprFlags::LZ4_COMPRESSED | AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
    assert!(flags.is_lz4());
    assert!(flags.is_quantized());
    assert!(flags.has_vocab());
}

// ============================================================================
// TensorEntry Coverage
// ============================================================================

#[test]
fn test_tensor_entry_from_binary_valid() {
    let data = create_binary_tensor_entry("weights", 0, &[128, 256], 0, 131072);
    let (entry, consumed) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.name, "weights");
    assert_eq!(entry.dtype, "F32");
    assert_eq!(entry.shape, vec![128, 256]);
    assert!(consumed > 0);
}

#[test]
fn test_tensor_entry_from_binary_too_short() {
    let data = vec![0u8; 2];
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_from_binary_truncated_name() {
    let mut data = Vec::new();
    data.extend_from_slice(&100u16.to_le_bytes());
    data.extend_from_slice(b"short");
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_element_count_2d() {
    let entry = TensorEntry {
        name: "test".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        offset: 0,
        size: 800,
    };
    assert_eq!(entry.element_count(), 200);
}

#[test]
fn test_tensor_entry_element_count_empty() {
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
fn test_tensor_entry_dtypes() {
    let dtypes = [
        (0, "F32"),
        (1, "F16"),
        (2, "BF16"),
        (3, "I8"),
        (4, "I16"),
        (5, "I32"),
        (6, "I64"),
        (7, "U8"),
        (8, "Q4_K"),
        (9, "Q6_K"),
        (10, "Q8_0"),
    ];
    for (dtype_byte, expected) in dtypes {
        let data = create_binary_tensor_entry("test", dtype_byte, &[1], 0, 4);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(entry.dtype, expected, "dtype byte {} should be {}", dtype_byte, expected);
    }
}

// ============================================================================
// AprMetadata Coverage
// ============================================================================

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

#[test]
fn test_apr_metadata_default() {
    let meta = AprMetadata::default();
    assert!(meta.hidden_size.is_none());
    assert!(!meta.is_transformer());
}

#[test]
fn test_apr_metadata_serialization() {
    let meta = AprMetadata {
        hidden_size: Some(1024),
        num_layers: Some(12),
        ..Default::default()
    };
    let json = serde_json::to_string(&meta).unwrap();
    let parsed: AprMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.hidden_size, Some(1024));
}

// ============================================================================
// BpeTokenizer Coverage
// ============================================================================

#[test]
fn test_bpe_tokenizer_encode_empty() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec![],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
    };
    let encoded = tokenizer.encode("");
    assert!(encoded.is_empty());
}

#[test]
fn test_bpe_tokenizer_encode_single_char() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    let tokenizer = BpeTokenizer {
        token_to_id,
        id_to_token: vec!["a".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
    };
    let encoded = tokenizer.encode("a");
    assert_eq!(encoded, vec![0]);
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
    };
    let encoded = tokenizer.encode("ab");
    assert!(!encoded.is_empty());
}

#[test]
fn test_bpe_tokenizer_decode_empty() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec!["test".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
    };
    let decoded = tokenizer.decode(&[]);
    assert!(decoded.is_empty());
}

#[test]
fn test_bpe_tokenizer_decode_valid() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec!["hello".to_string(), "world".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
    };
    let decoded = tokenizer.decode(&[0, 1]);
    assert!(decoded.contains("hello"));
    assert!(decoded.contains("world"));
}

#[test]
fn test_bpe_tokenizer_decode_out_of_bounds() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec!["a".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
    };
    let decoded = tokenizer.decode(&[0, 100]);
    assert!(decoded.contains("a"));
    assert!(decoded.contains("[100]"));
}

#[test]
fn test_bpe_tokenizer_bos_eos() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec![],
        merge_rules: vec![],
        bos_id: Some(1),
        eos_id: Some(2),
    };
    assert_eq!(tokenizer.bos_id, Some(1));
    assert_eq!(tokenizer.eos_id, Some(2));
}

// ============================================================================
// is_apr_file / detect_format Coverage
// ============================================================================

#[test]
fn test_is_apr_file_valid() {
    let temp = create_apr_file();
    assert!(is_apr_file(temp.path()));
}

#[test]
fn test_is_apr_file_nonexistent() {
    assert!(!is_apr_file("/nonexistent/path/model.apr"));
}

#[test]
fn test_is_apr_file_wrong_magic() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    temp.write_all(b"GGUF").expect("write magic");
    temp.write_all(&[0u8; 60]).expect("write padding");
    assert!(!is_apr_file(temp.path()));
}

#[test]
fn test_detect_format_apr_extension() {
    assert_eq!(detect_format("/path/model.apr"), "apr");
}

#[test]
fn test_detect_format_gguf_extension() {
    assert_eq!(detect_format("/path/model.gguf"), "gguf");
}

#[test]
fn test_detect_format_safetensors_extension() {
    assert_eq!(detect_format("/path/model.safetensors"), "safetensors");
}

#[test]
fn test_detect_format_unknown_extension() {
    assert_eq!(detect_format("/path/model.bin"), "unknown");
}

#[test]
fn test_detect_format_no_extension() {
    assert_eq!(detect_format("/path/model"), "unknown");
}

// ============================================================================
// Constants Coverage
// ============================================================================

#[test]
fn test_magic_constant() {
    assert_eq!(MAGIC, [0x41, 0x50, 0x52, 0x00]);
    assert_eq!(&MAGIC, b"APR\0");
}

#[test]
fn test_header_size_constant() {
    assert_eq!(HEADER_SIZE, 64);
}

#[test]
fn test_alignment_constant() {
    assert_eq!(ALIGNMENT, 64);
}

// ============================================================================
// AprV2Model Methods Coverage
// ============================================================================

#[test]
fn test_apr_v2_model_tensor_count() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.tensor_count(), 1);
}

#[test]
fn test_apr_v2_model_tensor_names() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let names = model.tensor_names();
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "test.weight");
}

#[test]
fn test_apr_v2_model_metadata() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let meta = model.metadata();
    assert_eq!(meta.vocab_size, Some(100));
}

#[test]
fn test_apr_v2_model_get_tensor() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let tensor = model.get_tensor("test.weight");
    assert!(tensor.is_some());
}

#[test]
fn test_apr_v2_model_get_tensor_not_found() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let tensor = model.get_tensor("nonexistent");
    assert!(tensor.is_none());
}

#[test]
fn test_apr_v2_model_get_tensor_f32() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let floats = model.get_tensor_f32("test.weight").expect("should get f32");
    assert_eq!(floats.len(), 16);
}

#[test]
fn test_apr_v2_model_get_tensor_f32_not_found() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.get_tensor_f32("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_get_tensor_bytes() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let bytes = model.get_tensor_bytes("test.weight").expect("should get bytes");
    assert_eq!(bytes.len(), 64);
}

#[test]
fn test_apr_v2_model_get_tensor_bytes_not_found() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.get_tensor_bytes("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_estimated_parameters() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.estimated_parameters(), 16);
}

#[test]
fn test_apr_v2_model_predict_empty_features() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.predict(&[]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap()[0], 0.0);
}

#[test]
fn test_apr_v2_model_predict_with_features() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.predict(&[1.0, 2.0, 3.0]);
    assert!(result.is_ok());
}

#[test]
fn test_apr_v2_model_forward_empty_tokens() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_generate_empty_input() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.generate(&[], 10, None);
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_decode_tokens() {
    let vocab = vec!["hello".to_string(), " ".to_string(), "world".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "hello world");
}

#[test]
fn test_apr_v2_model_decode_tokens_out_of_bounds() {
    let vocab = vec!["a".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 99]);
    assert!(result.contains("a"));
    assert!(result.contains("[99]"));
}

// ============================================================================
// MappedAprModel Truncated File Handling
// ============================================================================

#[test]
fn test_mapped_apr_model_truncated_metadata() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    let mut data = vec![0u8; 100];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size = 1000 (larger than file)
    temp.write_all(&data).expect("write data");

    let result = MappedAprModel::from_path(temp.path());
    assert!(result.is_err());
}

#[test]
fn test_mapped_apr_model_debug_impl() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("MappedAprModel"));
}

// ============================================================================
// AprV2Model Encrypted File Handling
// ============================================================================

#[test]
fn test_apr_v2_model_encrypted_error() {
    let mut data = vec![0u8; HEADER_SIZE + 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&AprFlags::ENCRYPTED.to_le_bytes());

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

// ============================================================================
// AprV2Model Compressed File Handling (without apr-compression feature)
// ============================================================================

#[cfg(not(feature = "apr-compression"))]
#[test]
fn test_apr_v2_model_compressed_requires_feature() {
    let mut data = vec![0u8; HEADER_SIZE + 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&AprFlags::LZ4_COMPRESSED.to_le_bytes());
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&0u32.to_le_bytes());
    data[24..32].copy_from_slice(&64u64.to_le_bytes());
    data[32..40].copy_from_slice(&64u64.to_le_bytes());

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("apr-compression"));
}

// ============================================================================
// Release CPU Pages Coverage (Unix only)
// ============================================================================

#[cfg(all(unix, not(target_arch = "wasm32")))]
#[test]
fn test_apr_v2_model_release_cpu_pages() {
    let temp = create_apr_file();
    let model = AprV2Model::load(temp.path()).expect("should load");
    let result = model.release_cpu_pages();
    assert!(result.is_ok());
}

// ============================================================================
// ModelData Type Alias Coverage
// ============================================================================

#[test]
fn test_apr_model_type_alias() {
    // AprModel is a type alias for AprV2Model
    let data = create_minimal_apr_model();
    let model: realizar::apr::AprModel = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.tensor_count(), 1);
}
