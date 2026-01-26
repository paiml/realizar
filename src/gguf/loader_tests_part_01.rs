//! Loader Tests Part 01 - Loading Functions, Header Parsing, and Error Handling
//!
//! Comprehensive tests for /src/gguf/loader.rs focusing on:
//! - Header parsing edge cases (magic, version validation)
//! - Metadata parsing with corrupted data
//! - Tensor info parsing with invalid dimensions
//! - Error propagation through read_* helper methods
//! - Alignment calculations
//! - Data range validation in get_tensor_f32
//!
//! This file targets uncovered paths identified in coverage analysis.

use crate::error::RealizarError;
use crate::gguf::{
    GGUFModel, GGUFValue, GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32,
    GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1,
    GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};

// =============================================================================
// Test Data Builders
// =============================================================================

/// Build valid GGUF header bytes
fn build_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&metadata_count.to_le_bytes());
    data
}

/// Build GGUF string (u64 length + UTF-8 bytes)
fn build_string(s: &str) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(s.len() as u64).to_le_bytes());
    data.extend_from_slice(s.as_bytes());
    data
}

/// Build metadata entry
fn build_metadata(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(build_string(key));
    data.extend_from_slice(&value_type.to_le_bytes());
    data.extend_from_slice(value_bytes);
    data
}

/// Build tensor info entry
fn build_tensor_info(name: &str, dims: &[u64], qtype: u32, offset: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(build_string(name));
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&qtype.to_le_bytes());
    data.extend_from_slice(&offset.to_le_bytes());
    data
}

/// Pad data to GGUF alignment (32 bytes)
fn align_to_boundary(data: &mut Vec<u8>) {
    let current_len = data.len();
    let aligned = current_len.div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);
}

// =============================================================================
// Header Parsing - Edge Cases
// =============================================================================

#[test]
fn test_loader_header_exactly_minimum_size() {
    // Exactly 24 bytes: magic(4) + version(4) + tensor_count(8) + metadata_count(8)
    let data = build_header(0, 0);
    assert_eq!(data.len(), 24);
    let model = GGUFModel::from_bytes(&data).expect("Minimum header should parse");
    assert_eq!(model.header.magic, GGUF_MAGIC);
    assert_eq!(model.header.version, GGUF_VERSION_V3);
}

#[test]
fn test_loader_header_magic_partial_bytes() {
    // Only 3 bytes of magic
    let data = vec![0x47, 0x47, 0x55];
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_loader_header_version_truncated() {
    // Magic complete, version truncated
    let mut data = GGUF_MAGIC.to_le_bytes().to_vec();
    data.extend_from_slice(&[0x03, 0x00]); // Only 2 bytes of version
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_loader_header_tensor_count_truncated() {
    // Magic + version complete, tensor_count truncated
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Only 4 bytes of tensor_count
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_loader_header_metadata_count_truncated() {
    // Magic + version + tensor_count complete, metadata_count truncated
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&[0x00, 0x00, 0x00]); // Only 3 bytes of metadata_count
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_loader_header_version_1_unsupported() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // Version 1
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("version") || err_msg.contains("Version"));
}

#[test]
fn test_loader_header_version_4_unsupported() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // Version 4 (future)
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_loader_header_all_zeros_magic() {
    let mut data = vec![0u8; 24];
    // All zeros - magic is 0, not GGUF_MAGIC
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string().to_lowercase();
    assert!(err.contains("magic"));
}

#[test]
fn test_loader_header_big_endian_magic() {
    // GGUF magic in big-endian (wrong byte order)
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_be_bytes()); // Wrong endianness
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_loader_header_max_tensor_count() {
    // Very large tensor count - parsing should succeed but no tensor data
    let mut data = build_header(u64::MAX, 0);
    // Don't add any tensor info - just test header parsing succeeds
    // The tensor info parsing will fail due to truncation

    let model = GGUFModel::from_bytes(&data);
    // Should fail during tensor info parsing due to truncation
    assert!(model.is_err());
}

// =============================================================================
// Metadata Parsing - Value Type Coverage
// =============================================================================

#[test]
fn test_loader_metadata_all_integer_types() {
    let mut data = build_header(0, 6);

    // u8
    data.extend(build_metadata("val_u8", 0, &[255u8]));
    // i8
    data.extend(build_metadata("val_i8", 1, &(-127i8).to_le_bytes()));
    // u16
    data.extend(build_metadata("val_u16", 2, &65535u16.to_le_bytes()));
    // i16
    data.extend(build_metadata("val_i16", 3, &(-32768i16).to_le_bytes()));
    // u32
    data.extend(build_metadata("val_u32", 4, &0xDEADBEEFu32.to_le_bytes()));
    // i32
    data.extend(build_metadata("val_i32", 5, &(-2147483648i32).to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse all integer types");

    assert!(matches!(
        model.metadata.get("val_u8"),
        Some(GGUFValue::UInt8(255))
    ));
    assert!(matches!(
        model.metadata.get("val_i8"),
        Some(GGUFValue::Int8(-127))
    ));
    assert!(matches!(
        model.metadata.get("val_u16"),
        Some(GGUFValue::UInt16(65535))
    ));
    assert!(matches!(
        model.metadata.get("val_i16"),
        Some(GGUFValue::Int16(-32768))
    ));
    assert!(matches!(
        model.metadata.get("val_u32"),
        Some(GGUFValue::UInt32(0xDEADBEEF))
    ));
    assert!(matches!(
        model.metadata.get("val_i32"),
        Some(GGUFValue::Int32(-2147483648))
    ));
}

#[test]
fn test_loader_metadata_bool_nonzero_values() {
    let mut data = build_header(0, 3);

    // bool: 0 = false
    data.extend(build_metadata("bool_zero", 7, &[0u8]));
    // bool: 1 = true
    data.extend(build_metadata("bool_one", 7, &[1u8]));
    // bool: 255 = also true (any non-zero)
    data.extend(build_metadata("bool_255", 7, &[255u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");

    assert!(matches!(
        model.metadata.get("bool_zero"),
        Some(GGUFValue::Bool(false))
    ));
    assert!(matches!(
        model.metadata.get("bool_one"),
        Some(GGUFValue::Bool(true))
    ));
    assert!(matches!(
        model.metadata.get("bool_255"),
        Some(GGUFValue::Bool(true))
    ));
}

#[test]
fn test_loader_metadata_float_special_values() {
    let mut data = build_header(0, 6);

    // f32 specials
    data.extend(build_metadata(
        "f32_nan",
        6,
        &f32::NAN.to_le_bytes(),
    ));
    data.extend(build_metadata(
        "f32_inf",
        6,
        &f32::INFINITY.to_le_bytes(),
    ));
    data.extend(build_metadata(
        "f32_neg_inf",
        6,
        &f32::NEG_INFINITY.to_le_bytes(),
    ));

    // f64 specials
    data.extend(build_metadata(
        "f64_nan",
        12,
        &f64::NAN.to_le_bytes(),
    ));
    data.extend(build_metadata(
        "f64_inf",
        12,
        &f64::INFINITY.to_le_bytes(),
    ));
    data.extend(build_metadata(
        "f64_neg_inf",
        12,
        &f64::NEG_INFINITY.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse special floats");

    if let Some(GGUFValue::Float32(v)) = model.metadata.get("f32_nan") {
        assert!(v.is_nan());
    } else {
        panic!("Expected Float32 NaN");
    }

    if let Some(GGUFValue::Float32(v)) = model.metadata.get("f32_inf") {
        assert!(v.is_infinite() && *v > 0.0);
    }

    if let Some(GGUFValue::Float64(v)) = model.metadata.get("f64_neg_inf") {
        assert!(v.is_infinite() && *v < 0.0);
    }
}

#[test]
fn test_loader_metadata_empty_string() {
    let mut data = build_header(0, 1);
    let empty_str = build_string("");
    data.extend(build_metadata("empty", 8, &empty_str));

    let model = GGUFModel::from_bytes(&data).expect("Should parse empty string");

    if let Some(GGUFValue::String(s)) = model.metadata.get("empty") {
        assert!(s.is_empty());
    } else {
        panic!("Expected empty String");
    }
}

#[test]
fn test_loader_metadata_array_empty() {
    let mut data = build_header(0, 1);

    // Empty array of u32
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
    array_bytes.extend_from_slice(&0u64.to_le_bytes()); // array length: 0

    data.extend(build_metadata("empty_arr", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse empty array");

    if let Some(GGUFValue::Array(arr)) = model.metadata.get("empty_arr") {
        assert!(arr.is_empty());
    } else {
        panic!("Expected empty Array");
    }
}

#[test]
fn test_loader_metadata_array_of_arrays_unsupported() {
    // Nested arrays (array of arrays) - type 9 inside type 9
    // This tests recursive array parsing
    let mut data = build_header(0, 1);

    let mut outer_array = Vec::new();
    outer_array.extend_from_slice(&9u32.to_le_bytes()); // element type: array
    outer_array.extend_from_slice(&1u64.to_le_bytes()); // outer array length: 1

    // Inner array: [1, 2]
    outer_array.extend_from_slice(&4u32.to_le_bytes()); // inner element type: u32
    outer_array.extend_from_slice(&2u64.to_le_bytes()); // inner array length: 2
    outer_array.extend_from_slice(&1u32.to_le_bytes());
    outer_array.extend_from_slice(&2u32.to_le_bytes());

    data.extend(build_metadata("nested", 9, &outer_array));

    let model = GGUFModel::from_bytes(&data).expect("Should parse nested arrays");

    if let Some(GGUFValue::Array(outer)) = model.metadata.get("nested") {
        assert_eq!(outer.len(), 1);
        if let GGUFValue::Array(inner) = &outer[0] {
            assert_eq!(inner.len(), 2);
        } else {
            panic!("Expected inner array");
        }
    } else {
        panic!("Expected outer Array");
    }
}

#[test]
fn test_loader_metadata_type_13_invalid() {
    let mut data = build_header(0, 1);
    // Type 13 doesn't exist in GGUF spec
    data.extend(build_metadata("bad_type", 13, &[0u8; 8]));

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_loader_metadata_type_max_u32_invalid() {
    let mut data = build_header(0, 1);
    // Type u32::MAX doesn't exist
    data.extend(build_string("bad_type"));
    data.extend_from_slice(&u32::MAX.to_le_bytes());
    data.extend_from_slice(&[0u8; 8]);

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// =============================================================================
// Tensor Info Parsing - Edge Cases
// =============================================================================

#[test]
fn test_loader_tensor_info_zero_dims() {
    let mut data = build_header(1, 0);
    // Tensor with 0 dimensions (scalar)
    data.extend(build_string("scalar"));
    data.extend_from_slice(&0u32.to_le_bytes()); // n_dims = 0
    data.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    let model = GGUFModel::from_bytes(&data).expect("Should parse scalar tensor");
    assert_eq!(model.tensors[0].n_dims, 0);
    assert!(model.tensors[0].dims.is_empty());
}

#[test]
fn test_loader_tensor_info_high_dimensional() {
    let mut data = build_header(1, 0);
    // 6-dimensional tensor
    data.extend(build_string("high_dim"));
    data.extend_from_slice(&6u32.to_le_bytes()); // n_dims = 6
    for _ in 0..6 {
        data.extend_from_slice(&2u64.to_le_bytes()); // Each dim = 2
    }
    data.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let model = GGUFModel::from_bytes(&data).expect("Should parse high-dim tensor");
    assert_eq!(model.tensors[0].n_dims, 6);
    assert_eq!(model.tensors[0].dims.len(), 6);
}

#[test]
fn test_loader_tensor_info_dimension_reversal() {
    // GGUF stores dims in GGML order (reversed)
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("weights", &[256, 512], GGUF_TYPE_F32, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Dims should match input order after reversal handling
    assert_eq!(model.tensors[0].dims, vec![256, 512]);
}

#[test]
fn test_loader_tensor_info_large_dimension() {
    let mut data = build_header(1, 0);
    // Single very large dimension
    data.extend(build_tensor_info("large", &[1_000_000_000], GGUF_TYPE_F32, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse large dim");
    assert_eq!(model.tensors[0].dims[0], 1_000_000_000);
}

#[test]
fn test_loader_tensor_info_multiple_qtypes() {
    let mut data = build_header(5, 0);

    data.extend(build_tensor_info("f32", &[32], GGUF_TYPE_F32, 0));
    data.extend(build_tensor_info("f16", &[32], GGUF_TYPE_F16, 128));
    data.extend(build_tensor_info("q4_0", &[32], GGUF_TYPE_Q4_0, 192));
    data.extend(build_tensor_info("q8_0", &[32], GGUF_TYPE_Q8_0, 210));
    data.extend(build_tensor_info("q4_k", &[256], GGUF_TYPE_Q4_K, 244));

    let model = GGUFModel::from_bytes(&data).expect("Should parse multiple qtypes");

    assert_eq!(model.tensors.len(), 5);
    assert_eq!(model.tensors[0].qtype, GGUF_TYPE_F32);
    assert_eq!(model.tensors[1].qtype, GGUF_TYPE_F16);
    assert_eq!(model.tensors[2].qtype, GGUF_TYPE_Q4_0);
    assert_eq!(model.tensors[3].qtype, GGUF_TYPE_Q8_0);
    assert_eq!(model.tensors[4].qtype, GGUF_TYPE_Q4_K);
}

// =============================================================================
// Alignment Calculation Tests
// =============================================================================

#[test]
fn test_loader_alignment_at_boundary() {
    // Header + metadata that lands exactly on 32-byte boundary
    let mut data = build_header(0, 0);
    assert_eq!(data.len(), 24);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // tensor_data_start should be aligned to 32
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
    assert_eq!(model.tensor_data_start, 32); // 24 rounded up to 32
}

#[test]
fn test_loader_alignment_with_metadata() {
    let mut data = build_header(0, 1);
    // Add metadata that changes alignment requirement
    data.extend(build_metadata("test", 4, &42u32.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
}

#[test]
fn test_loader_alignment_with_tensors() {
    let mut data = build_header(2, 0);
    data.extend(build_tensor_info("t1", &[64], GGUF_TYPE_F32, 0));
    data.extend(build_tensor_info("t2", &[64], GGUF_TYPE_F32, 256));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
}

// =============================================================================
// get_tensor_f32 - Data Range Validation
// =============================================================================

#[test]
fn test_loader_get_tensor_f32_exact_boundary() {
    // Tensor data ends exactly at file end
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[4], GGUF_TYPE_F32, 0));
    align_to_boundary(&mut data);

    let tensor_start = data.len();
    // 4 f32 values = 16 bytes
    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 4);
}

#[test]
fn test_loader_get_tensor_f32_one_byte_short() {
    // Tensor data is 1 byte short
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[4], GGUF_TYPE_F32, 0));
    align_to_boundary(&mut data);

    // 4 f32 = 16 bytes, but only provide 15
    data.extend(&[0u8; 15]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_err());
}

#[test]
fn test_loader_get_tensor_f32_2d_matrix() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("matrix", &[3, 4], GGUF_TYPE_F32, 0));
    align_to_boundary(&mut data);

    // 3 * 4 = 12 f32 values = 48 bytes
    for i in 0..12 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("matrix", &data).expect("Should extract");
    assert_eq!(tensor.len(), 12);
    assert!((tensor[5] - 5.0).abs() < f32::EPSILON);
}

#[test]
fn test_loader_get_tensor_q4_0_partial_block() {
    // Q4_0 with elements not divisible by block size (32)
    // Tests the truncation after dequantization
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[48], GGUF_TYPE_Q4_0, 0)); // 48 = 1.5 blocks
    align_to_boundary(&mut data);

    // Need 2 blocks (64 elements worth) for 48 elements
    // Each Q4_0 block: 2 bytes f16 scale + 16 bytes quants = 18 bytes
    for _ in 0..2 {
        let scale = half::f16::from_f32(1.0);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0x00u8; 16]);
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 48); // Truncated to exact size
}

#[test]
fn test_loader_get_tensor_q8_0_partial_block() {
    // Q8_0 with elements not divisible by block size (32)
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[50], GGUF_TYPE_Q8_0, 0)); // 50 = 1.56 blocks
    align_to_boundary(&mut data);

    // Need 2 blocks for 50 elements
    // Each Q8_0 block: 2 bytes f16 scale + 32 bytes quants = 34 bytes
    for _ in 0..2 {
        let scale = half::f16::from_f32(0.5);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0i8 as u8; 32]);
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 50);
}

#[test]
fn test_loader_get_tensor_f16_values() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[8], GGUF_TYPE_F16, 0));
    align_to_boundary(&mut data);

    // 8 f16 values
    let values = [0.0f32, 0.5, 1.0, 1.5, 2.0, -1.0, -0.5, 0.25];
    for v in &values {
        let f16_val = half::f16::from_f32(*v);
        data.extend_from_slice(&f16_val.to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 8);
    assert!((tensor[2] - 1.0).abs() < 0.01);
    assert!((tensor[5] - (-1.0)).abs() < 0.01);
}

#[test]
fn test_loader_get_tensor_q4_1_block() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q4_1, 0));
    align_to_boundary(&mut data);

    // Q4_1: 2 f16 scale + 2 f16 min + 16 bytes quants = 20 bytes
    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0x00u8; 16]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_loader_get_tensor_q5_0_block() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_0, 0));
    align_to_boundary(&mut data);

    // Q5_0: 2 f16 scale + 4 high bits + 16 bytes quants = 22 bytes
    let scale = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend([0u8; 4]); // high bits
    data.extend([0x00u8; 16]); // quants

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_loader_get_tensor_q5_1_block() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[32], GGUF_TYPE_Q5_1, 0));
    align_to_boundary(&mut data);

    // Q5_1: 2 f16 scale + 2 f16 min + 4 high bits + 16 bytes quants = 24 bytes
    let scale = half::f16::from_f32(1.0);
    let min = half::f16::from_f32(0.0);
    data.extend_from_slice(&scale.to_le_bytes());
    data.extend_from_slice(&min.to_le_bytes());
    data.extend([0u8; 4]); // high bits
    data.extend([0x00u8; 16]); // quants

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 32);
}

#[test]
fn test_loader_get_tensor_k_quant_partial() {
    // K-quant with partial super-block (elements not divisible by 256)
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[300], GGUF_TYPE_Q4_K, 0)); // 300 = 1.17 super-blocks
    align_to_boundary(&mut data);

    // Need 2 super-blocks (512 elements worth)
    // Q4_K super-block: 144 bytes
    data.extend([0u8; 288]); // 2 * 144

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 300); // Truncated
}

#[test]
fn test_loader_get_tensor_offset_nonzero() {
    // Tensor at non-zero offset
    let mut data = build_header(1, 0);
    // Offset 64 bytes into tensor data section
    data.extend(build_tensor_info("test", &[4], GGUF_TYPE_F32, 64));
    align_to_boundary(&mut data);

    // Padding before tensor data
    data.extend([0xFFu8; 64]);
    // Actual tensor data
    for i in 0..4 {
        data.extend_from_slice(&((i + 100) as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 4);
    assert!((tensor[0] - 100.0).abs() < f32::EPSILON);
}

// =============================================================================
// Error Message Quality Tests
// =============================================================================

#[test]
fn test_loader_error_tensor_not_found_message() {
    let data = build_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");

    let result = model.get_tensor_f32("nonexistent_tensor", &data);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("nonexistent_tensor") || err_str.contains("not found"),
        "Error should mention tensor name: {}",
        err_str
    );
}

#[test]
fn test_loader_error_unsupported_qtype_message() {
    let mut data = build_header(1, 0);
    data.extend(build_tensor_info("test", &[32], 200, 0)); // qtype 200 doesn't exist
    align_to_boundary(&mut data);
    data.extend([0u8; 128]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("200") || err_str.contains("Unsupported"),
        "Error should mention unsupported type: {}",
        err_str
    );
}

#[test]
fn test_loader_error_data_range_message() {
    let mut data = build_header(1, 0);
    // Claim huge tensor that exceeds file size
    data.extend(build_tensor_info("huge", &[1_000_000], GGUF_TYPE_F32, 0));
    // No actual tensor data

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("huge", &data);

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("exceeds") || err_str.contains("range"),
        "Error should mention data range issue: {}",
        err_str
    );
}

// =============================================================================
// Edge Cases in String Parsing
// =============================================================================

#[test]
fn test_loader_string_with_null_bytes() {
    let mut data = build_header(0, 1);
    // String containing null bytes
    let s = "hello\0world";
    let str_bytes = build_string(s);
    data.extend(build_metadata("null_str", 8, &str_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::String(val)) = model.metadata.get("null_str") {
        assert_eq!(val, "hello\0world");
    }
}

#[test]
fn test_loader_string_utf8_multibyte() {
    let mut data = build_header(0, 1);
    // Various UTF-8 multibyte sequences
    let s = "\u{00E9}\u{4E2D}\u{1F600}"; // e-acute, Chinese char, emoji
    let str_bytes = build_string(s);
    data.extend(build_metadata("utf8", 8, &str_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::String(val)) = model.metadata.get("utf8") {
        assert_eq!(val, s);
    }
}

#[test]
fn test_loader_tensor_name_with_dots() {
    let mut data = build_header(1, 0);
    // Typical GGUF tensor name with multiple dots
    data.extend(build_tensor_info(
        "blk.15.attn_q.weight",
        &[4096, 4096],
        GGUF_TYPE_Q4_K,
        0,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].name, "blk.15.attn_q.weight");
}
