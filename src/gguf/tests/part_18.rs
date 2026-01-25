//! Phase 33: GGUF Loader coverage tests
//!
//! These lib tests illuminate gguf/loader.rs:
//! - from_bytes() - GGUF parsing
//! - parse_header() - Header validation
//! - parse_metadata() - Metadata parsing
//! - parse_tensor_info() - Tensor info extraction
//! - get_tensor_f32() - Tensor extraction with dequantization
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::{GGUFModel, GGUFValue, GGUF_MAGIC, GGUF_VERSION_V3};

// =============================================================================
// GGUF Test Data Builder
// =============================================================================

/// Build valid GGUF header bytes
fn build_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    // Tensor count
    data.extend_from_slice(&tensor_count.to_le_bytes());
    // Metadata count
    data.extend_from_slice(&metadata_count.to_le_bytes());
    data
}

/// Build a GGUF string: u64 length + UTF-8 bytes
fn build_gguf_string(s: &str) -> Vec<u8> {
    let mut data = Vec::new();
    let len = s.len() as u64;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(s.as_bytes());
    data
}

/// Build a GGUF metadata key-value pair
fn build_gguf_metadata(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(build_gguf_string(key));
    data.extend_from_slice(&value_type.to_le_bytes());
    data.extend_from_slice(value_bytes);
    data
}

/// Build a GGUF tensor info entry
fn build_tensor_info(name: &str, dims: &[u64], qtype: u32, offset: u64) -> Vec<u8> {
    let mut data = Vec::new();
    // Name string
    data.extend(build_gguf_string(name));
    // n_dims
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    // Dimensions (reversed for GGML order)
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    // Quantization type
    data.extend_from_slice(&qtype.to_le_bytes());
    // Offset
    data.extend_from_slice(&offset.to_le_bytes());
    data
}

// =============================================================================
// Header Parsing Tests
// =============================================================================

#[test]
fn test_phase33_loader_valid_header() {
    // Minimal valid GGUF with no tensors, no metadata
    let data = build_gguf_header(0, 0);
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "Valid header should parse");
    let model = result.unwrap();
    assert_eq!(model.header.magic, GGUF_MAGIC);
    assert_eq!(model.header.version, GGUF_VERSION_V3);
    assert_eq!(model.header.tensor_count, 0);
    assert_eq!(model.header.metadata_count, 0);
}

#[test]
fn test_phase33_loader_invalid_magic() {
    let mut data = build_gguf_header(0, 0);
    // Corrupt magic
    data[0] = 0xFF;
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("magic") || err.contains("Magic"), "Error should mention magic: {}", err);
}

#[test]
fn test_phase33_loader_wrong_version() {
    let mut data = Vec::new();
    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version 2 (not supported)
    data.extend_from_slice(&2u32.to_le_bytes());
    // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes());
    // Metadata count
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("version") || err.contains("Version"), "Error should mention version: {}", err);
}

#[test]
fn test_phase33_loader_truncated_header() {
    // Only magic, missing rest
    let data = GGUF_MAGIC.to_le_bytes();
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated header should fail");
}

#[test]
fn test_phase33_loader_empty_data() {
    let result = GGUFModel::from_bytes(&[]);
    assert!(result.is_err(), "Empty data should fail");
}

// =============================================================================
// Metadata Parsing Tests
// =============================================================================

#[test]
fn test_phase33_loader_metadata_uint8() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_u8", 0, &[42u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_u8").expect("Key should exist");
    assert!(matches!(value, GGUFValue::UInt8(42)));
}

#[test]
fn test_phase33_loader_metadata_int8() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_i8", 1, &[(-10i8).to_le_bytes()[0]]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_i8").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Int8(-10)));
}

#[test]
fn test_phase33_loader_metadata_uint16() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_u16", 2, &1000u16.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_u16").expect("Key should exist");
    assert!(matches!(value, GGUFValue::UInt16(1000)));
}

#[test]
fn test_phase33_loader_metadata_int16() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_i16", 3, &(-500i16).to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_i16").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Int16(-500)));
}

#[test]
fn test_phase33_loader_metadata_uint32() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_u32", 4, &100000u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_u32").expect("Key should exist");
    assert!(matches!(value, GGUFValue::UInt32(100000)));
}

#[test]
fn test_phase33_loader_metadata_int32() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_i32", 5, &(-50000i32).to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_i32").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Int32(-50000)));
}

#[test]
fn test_phase33_loader_metadata_float32() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_f32", 6, &3.14f32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_f32").expect("Key should exist");
    if let GGUFValue::Float32(v) = value {
        assert!((v - 3.14).abs() < 0.001);
    } else {
        panic!("Expected Float32");
    }
}

#[test]
fn test_phase33_loader_metadata_bool_true() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_bool", 7, &[1u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_bool").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Bool(true)));
}

#[test]
fn test_phase33_loader_metadata_bool_false() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_bool", 7, &[0u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_bool").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Bool(false)));
}

#[test]
fn test_phase33_loader_metadata_string() {
    let mut data = build_gguf_header(0, 1);
    let string_value = build_gguf_string("hello world");
    data.extend(build_gguf_metadata("test_string", 8, &string_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_string").expect("Key should exist");
    if let GGUFValue::String(s) = value {
        assert_eq!(s, "hello world");
    } else {
        panic!("Expected String");
    }
}

#[test]
fn test_phase33_loader_metadata_uint64() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_u64", 10, &999999999u64.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_u64").expect("Key should exist");
    assert!(matches!(value, GGUFValue::UInt64(999999999)));
}

#[test]
fn test_phase33_loader_metadata_int64() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_i64", 11, &(-999999999i64).to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_i64").expect("Key should exist");
    assert!(matches!(value, GGUFValue::Int64(-999999999)));
}

#[test]
fn test_phase33_loader_metadata_float64() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_f64", 12, &2.71828f64.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_f64").expect("Key should exist");
    if let GGUFValue::Float64(v) = value {
        assert!((v - 2.71828).abs() < 0.00001);
    } else {
        panic!("Expected Float64");
    }
}

#[test]
fn test_phase33_loader_metadata_array_u32() {
    let mut data = build_gguf_header(0, 1);
    // Array: element_type (u32=4) + array_len (u64=3) + elements
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
    array_bytes.extend_from_slice(&3u64.to_le_bytes()); // array length: 3
    array_bytes.extend_from_slice(&10u32.to_le_bytes());
    array_bytes.extend_from_slice(&20u32.to_le_bytes());
    array_bytes.extend_from_slice(&30u32.to_le_bytes());
    data.extend(build_gguf_metadata("test_array", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let value = model.metadata.get("test_array").expect("Key should exist");
    if let GGUFValue::Array(arr) = value {
        assert_eq!(arr.len(), 3);
        assert!(matches!(arr[0], GGUFValue::UInt32(10)));
        assert!(matches!(arr[1], GGUFValue::UInt32(20)));
        assert!(matches!(arr[2], GGUFValue::UInt32(30)));
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_phase33_loader_metadata_unsupported_type() {
    let mut data = build_gguf_header(0, 1);
    // Type 99 doesn't exist
    data.extend(build_gguf_metadata("test_bad", 99, &[0u8; 8]));

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("99") || err.contains("type"), "Error should mention bad type: {}", err);
}

#[test]
fn test_phase33_loader_multiple_metadata() {
    let mut data = build_gguf_header(0, 3);
    data.extend(build_gguf_metadata("hidden_dim", 4, &256u32.to_le_bytes()));
    data.extend(build_gguf_metadata("num_layers", 4, &12u32.to_le_bytes()));
    data.extend(build_gguf_metadata("rope_theta", 6, &10000.0f32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.metadata.len(), 3);
    assert!(matches!(model.metadata.get("hidden_dim"), Some(GGUFValue::UInt32(256))));
    assert!(matches!(model.metadata.get("num_layers"), Some(GGUFValue::UInt32(12))));
}

// =============================================================================
// Tensor Info Parsing Tests
// =============================================================================

#[test]
fn test_phase33_loader_tensor_info_basic() {
    let mut data = build_gguf_header(1, 0);
    // Add tensor info: 2D tensor [256, 512], F32, offset 0
    data.extend(build_tensor_info("weights", &[256, 512], 0, 0)); // qtype 0 = F32

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].name, "weights");
    assert_eq!(model.tensors[0].n_dims, 2);
    // Dims should be reversed from input (GGML order handling)
    assert_eq!(model.tensors[0].dims, vec![256, 512]);
    assert_eq!(model.tensors[0].qtype, 0);
    assert_eq!(model.tensors[0].offset, 0);
}

#[test]
fn test_phase33_loader_tensor_info_multiple() {
    let mut data = build_gguf_header(3, 0);
    data.extend(build_tensor_info("embed", &[100, 64], 0, 0));
    data.extend(build_tensor_info("attn.q", &[64, 64], 2, 25600)); // Q4_0
    data.extend(build_tensor_info("ffn.w1", &[64, 256], 6, 27648)); // Q8_0

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors.len(), 3);
    assert_eq!(model.tensors[0].name, "embed");
    assert_eq!(model.tensors[1].name, "attn.q");
    assert_eq!(model.tensors[2].name, "ffn.w1");
}

#[test]
fn test_phase33_loader_tensor_info_1d() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("bias", &[512], 0, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].n_dims, 1);
    assert_eq!(model.tensors[0].dims, vec![512]);
}

#[test]
fn test_phase33_loader_tensor_info_4d() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("conv", &[3, 3, 64, 128], 0, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].n_dims, 4);
    assert_eq!(model.tensors[0].dims.len(), 4);
}

// =============================================================================
// get_tensor_f32 Tests (requires tensor data)
// =============================================================================

#[test]
fn test_phase33_loader_get_tensor_f32_not_found() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");

    let result = model.get_tensor_f32("nonexistent", &data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("nonexistent") || err.contains("not found"), "Error: {}", err);
}

#[test]
fn test_phase33_loader_get_tensor_f32_basic() {
    // Build GGUF with one F32 tensor
    let mut data = build_gguf_header(1, 0);
    // Tensor: 4 elements of F32
    data.extend(build_tensor_info("test", &[4], 0, 0)); // F32=0

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add tensor data: 4 f32 values
    let values = [1.0f32, 2.0, 3.0, 4.0];
    for v in &values {
        data.extend_from_slice(&v.to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");

    assert_eq!(tensor.len(), 4);
    assert!((tensor[0] - 1.0).abs() < 0.001);
    assert!((tensor[1] - 2.0).abs() < 0.001);
    assert!((tensor[2] - 3.0).abs() < 0.001);
    assert!((tensor[3] - 4.0).abs() < 0.001);
}

#[test]
fn test_phase33_loader_get_tensor_f32_out_of_bounds() {
    // Build GGUF claiming tensor data that doesn't exist
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[1000000], 0, 0)); // 1M elements
    // No actual tensor data - should fail

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_err());
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_phase33_loader_empty_string_key() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("", 4, &42u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.metadata.contains_key(""));
}

#[test]
fn test_phase33_loader_empty_tensor_name() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("", &[64], 0, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].name, "");
}

#[test]
fn test_phase33_loader_unicode_metadata_key() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("test_\u{1F600}", 4, &42u32.to_le_bytes())); // emoji

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.metadata.contains_key("test_\u{1F600}"));
}

#[test]
fn test_phase33_loader_large_tensor_count() {
    // Just header with large tensor count (won't have data)
    let mut data = build_gguf_header(100, 0);
    // Add 100 tensor infos
    for i in 0..100 {
        data.extend(build_tensor_info(&format!("t{i}"), &[32], 0, i as u64 * 128));
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors.len(), 100);
}

#[test]
fn test_phase33_loader_large_metadata_count() {
    let mut data = build_gguf_header(0, 50);
    for i in 0..50 {
        data.extend(build_gguf_metadata(&format!("key_{i}"), 4, &(i as u32).to_le_bytes()));
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.metadata.len(), 50);
}

#[test]
fn test_phase33_loader_mixed_content() {
    // Full GGUF with metadata and tensors
    let mut data = build_gguf_header(2, 3);

    // Metadata
    data.extend(build_gguf_metadata("arch", 8, &build_gguf_string("llama")));
    data.extend(build_gguf_metadata("hidden", 4, &128u32.to_le_bytes()));
    data.extend(build_gguf_metadata("layers", 4, &2u32.to_le_bytes()));

    // Tensors
    data.extend(build_tensor_info("embed", &[100, 128], 0, 0));
    data.extend(build_tensor_info("norm", &[128], 0, 51200));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.metadata.len(), 3);
    assert_eq!(model.tensors.len(), 2);

    if let Some(GGUFValue::String(arch)) = model.metadata.get("arch") {
        assert_eq!(arch, "llama");
    }
}

// =============================================================================
// Metadata Accessors Tests
// =============================================================================

#[test]
fn test_phase33_loader_architecture() {
    let mut data = build_gguf_header(0, 1);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.architecture(), Some("llama"));
}

#[test]
fn test_phase33_loader_architecture_missing() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.architecture(), None);
}

#[test]
fn test_phase33_loader_embedding_dim() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata("llama.embedding_length", 4, &256u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.embedding_dim(), Some(256));
}

#[test]
fn test_phase33_loader_num_layers() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("qwen2");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata("qwen2.block_count", 4, &24u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_layers(), Some(24));
}

#[test]
fn test_phase33_loader_num_heads() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata("llama.attention.head_count", 4, &32u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_heads(), Some(32));
}

#[test]
fn test_phase33_loader_context_length() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata("llama.context_length", 4, &4096u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.context_length(), Some(4096));
}

#[test]
fn test_phase33_loader_num_kv_heads() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata("llama.attention.head_count_kv", 4, &8u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_kv_heads(), Some(8));
}

#[test]
fn test_phase33_loader_rope_freq_base() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata("llama.rope.freq_base", 6, &10000.0f32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let rope = model.rope_freq_base().expect("Should have rope");
    assert!((rope - 10000.0).abs() < 0.1);
}

#[test]
fn test_phase33_loader_rms_epsilon() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    data.extend(build_gguf_metadata("llama.attention.layer_norm_rms_epsilon", 6, &1e-5f32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let eps = model.rms_epsilon().expect("Should have epsilon");
    assert!((eps - 1e-5).abs() < 1e-7);
}

// =============================================================================
// Quantized Tensor Extraction Tests
// =============================================================================

#[test]
fn test_phase33_loader_get_tensor_q4_0() {
    use crate::gguf::GGUF_TYPE_Q4_0;

    // Q4_0: 18 bytes per 32 elements (2 f16 scale + 16 bytes quants)
    let mut data = build_gguf_header(1, 0);
    // 32 elements, Q4_0, offset 0
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q4_0, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q4_0 block: f16 scale + 16 bytes
    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    // 16 bytes of quants (each byte has 2 4-bit values)
    data.extend([0x11u8; 16]); // All 1s = small values

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q4_0 extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_q8_0() {
    use crate::gguf::GGUF_TYPE_Q8_0;

    // Q8_0: 34 bytes per 32 elements (2 f16 scale + 32 i8 quants)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q8_0, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q8_0 block: f16 scale + 32 i8 quants
    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend([0i8 as u8; 32]); // All zeros

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q8_0 extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_f16() {
    use crate::gguf::GGUF_TYPE_F16;

    // F16: 2 bytes per element
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[4], GGUF_TYPE_F16, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add F16 data: 4 values
    let values = [1.0f32, 2.0, 3.0, 4.0];
    for v in &values {
        let f16_val = half::f16::from_f32(*v);
        data.extend_from_slice(&f16_val.to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "F16 extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 4);
    assert!((tensor[0] - 1.0).abs() < 0.01);
}

#[test]
fn test_phase33_loader_get_tensor_q4_1() {
    use crate::gguf::GGUF_TYPE_Q4_1;

    // Q4_1: 20 bytes per 32 elements (2 scale + 2 min + 16 quants)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q4_1, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q4_1 block: f16 scale + f16 min + 16 bytes quants
    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0x00u8; 16]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q4_1 extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_q5_0() {
    use crate::gguf::GGUF_TYPE_Q5_0;

    // Q5_0: 22 bytes per 32 elements (2 scale + 4 high bits + 16 quants)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_0, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q5_0 block
    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend([0u8; 4]);  // high bits
    data.extend([0x00u8; 16]); // quants

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q5_0 extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_q5_1() {
    use crate::gguf::GGUF_TYPE_Q5_1;

    // Q5_1: 24 bytes per 32 elements (2 scale + 2 min + 4 high bits + 16 quants)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_1, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q5_1 block
    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0u8; 4]);  // high bits
    data.extend([0x00u8; 16]); // quants

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q5_1 extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_phase33_loader_get_tensor_unsupported_qtype() {
    // Use a fake quantization type that doesn't exist
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], 255, 0)); // 255 = invalid qtype

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);
    data.extend([0u8; 64]); // dummy data

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("255") || err.contains("Unsupported"), "Error: {}", err);
}

#[test]
fn test_phase33_loader_get_tensor_q2_k() {
    use crate::gguf::GGUF_TYPE_Q2_K;

    // Q2_K: 84 bytes per 256 elements (super-block)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q2_K, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q2_K super-block (84 bytes)
    data.extend([0u8; 84]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q2_K extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 256);
}

#[test]
fn test_phase33_loader_get_tensor_q4_k() {
    use crate::gguf::GGUF_TYPE_Q4_K;

    // Q4_K: 144 bytes per 256 elements (super-block)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q4_K, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q4_K super-block (144 bytes)
    data.extend([0u8; 144]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q4_K extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 256);
}

#[test]
fn test_phase33_loader_get_tensor_q5_k() {
    use crate::gguf::GGUF_TYPE_Q5_K;

    // Q5_K: 176 bytes per 256 elements (super-block)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q5_K, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q5_K super-block (176 bytes)
    data.extend([0u8; 176]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q5_K extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 256);
}

#[test]
fn test_phase33_loader_get_tensor_q6_k() {
    use crate::gguf::GGUF_TYPE_Q6_K;

    // Q6_K: 210 bytes per 256 elements (super-block)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q6_K, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q6_K super-block (210 bytes)
    data.extend([0u8; 210]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_ok(), "Q6_K extraction should work: {:?}", result.err());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 256);
}

// =============================================================================
// RoPE Type Tests
// =============================================================================

#[test]
fn test_phase33_loader_rope_type_llama() {
    let mut data = build_gguf_header(0, 1);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(0)); // NORM style for LLaMA
}

#[test]
fn test_phase33_loader_rope_type_qwen2() {
    let mut data = build_gguf_header(0, 1);
    let arch_value = build_gguf_string("qwen2");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX style for Qwen2
}

#[test]
fn test_phase33_loader_rope_type_phi3() {
    let mut data = build_gguf_header(0, 1);
    let arch_value = build_gguf_string("phi3");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX style for Phi3
}

#[test]
fn test_phase33_loader_rope_type_from_scaling() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let yarn_value = build_gguf_string("yarn");
    data.extend(build_gguf_metadata("custom.rope.scaling.type", 8, &yarn_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX from yarn scaling
}

#[test]
fn test_phase33_loader_rope_type_linear_scaling() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let linear_value = build_gguf_string("linear");
    data.extend(build_gguf_metadata("custom.rope.scaling.type", 8, &linear_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(0)); // NORM from linear scaling
}

// =============================================================================
// Token ID Tests
// =============================================================================

#[test]
fn test_phase33_loader_bos_token_id() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("tokenizer.ggml.bos_token_id", 4, &1u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.bos_token_id(), Some(1));
}

#[test]
fn test_phase33_loader_bos_token_id_missing() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.bos_token_id(), None);
}

#[test]
fn test_phase33_loader_eos_token_id() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata("tokenizer.ggml.eos_token_id", 4, &2u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.eos_token_id(), Some(2));
}

// =============================================================================
// Vocabulary Tests
// =============================================================================

#[test]
fn test_phase33_loader_vocabulary() {
    let mut data = build_gguf_header(0, 1);
    // Array of strings: element_type (8=string) + array_len (3) + strings
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes()); // element type: string
    array_bytes.extend_from_slice(&3u64.to_le_bytes()); // array length: 3
    array_bytes.extend(build_gguf_string("hello"));
    array_bytes.extend(build_gguf_string("world"));
    array_bytes.extend(build_gguf_string("!"));
    data.extend(build_gguf_metadata("tokenizer.ggml.tokens", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let vocab = model.vocabulary().expect("Should have vocabulary");
    assert_eq!(vocab.len(), 3);
    assert_eq!(vocab[0], "hello");
    assert_eq!(vocab[1], "world");
    assert_eq!(vocab[2], "!");
}

#[test]
fn test_phase33_loader_vocabulary_missing() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.vocabulary().is_none());
}

// =============================================================================
// Decode Tests
// =============================================================================

#[test]
fn test_phase33_loader_decode_basic() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("hello"));
    array_bytes.extend(build_gguf_string("▁world")); // SentencePiece space
    array_bytes.extend(build_gguf_string("!"));
    data.extend(build_gguf_metadata("tokenizer.ggml.tokens", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let decoded = model.decode(&[0, 1, 2]);
    assert_eq!(decoded, "hello world!");
}

#[test]
fn test_phase33_loader_decode_no_vocab() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let decoded = model.decode(&[65, 66, 67]); // ASCII fallback
    assert_eq!(decoded, "ABC");
}

#[test]
fn test_phase33_loader_decode_byte_tokens() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("<0x48>")); // 'H'
    array_bytes.extend(build_gguf_string("<0x69>")); // 'i'
    data.extend(build_gguf_metadata("tokenizer.ggml.tokens", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let decoded = model.decode(&[0, 1]);
    assert_eq!(decoded, "Hi");
}

// =============================================================================
// Encode Tests
// =============================================================================

#[test]
fn test_phase33_loader_encode_basic() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("▁hello")); // SentencePiece prefix
    array_bytes.extend(build_gguf_string("▁world"));
    array_bytes.extend(build_gguf_string("!"));
    data.extend(build_gguf_metadata("tokenizer.ggml.tokens", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tokens = model.encode("hello world!").expect("Should encode");
    // Encoding adds ▁ prefix to " hello" -> "▁hello" matches token 0
    assert!(!tokens.is_empty());
}

#[test]
fn test_phase33_loader_encode_no_vocab() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.encode("hello").is_none());
}
