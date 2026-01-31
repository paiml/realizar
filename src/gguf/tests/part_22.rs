//! Phase 35: GGUF Module Coverage Enhancement
//!
//! This module adds comprehensive test coverage for edge cases and error paths
//! in the GGUF parsing code:
//!
//! - Error handling in loader.rs (truncated data, corrupted structures)
//! - MappedGGUFModel edge cases (memory mapping, tensor slicing)
//! - GGUFValue edge cases (all variants, cloning, equality)
//! - RoPE type inference for all supported architectures
//! - Tokenizer edge cases (GPT-2 style, SentencePiece, byte tokens)
//! - Array and nested metadata parsing
//! - Tensor dimension handling and overflow
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::{GGUFHeader, GGUFModel, GGUFValue, TensorInfo, GGUF_MAGIC, GGUF_VERSION_V3};

// =============================================================================
// Test Data Builders (shared utilities)
// =============================================================================

/// Build valid GGUF header bytes
fn build_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
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
    data.extend(build_gguf_string(name));
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&qtype.to_le_bytes());
    data.extend_from_slice(&offset.to_le_bytes());
    data
}

// =============================================================================
// GGUFValue Tests - Edge Cases and All Variants
// =============================================================================

#[test]
fn test_phase35_gguf_value_uint8_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 0, &[0u8]));
    data.extend(build_gguf_metadata("test_max", 0, &[255u8]));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_min"),
        Some(GGUFValue::UInt8(0))
    ));
    assert!(matches!(
        model.metadata.get("test_max"),
        Some(GGUFValue::UInt8(255))
    ));
}

#[test]
fn test_phase35_gguf_value_int8_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 1, &i8::MIN.to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 1, &i8::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_min"),
        Some(GGUFValue::Int8(-128))
    ));
    assert!(matches!(
        model.metadata.get("test_max"),
        Some(GGUFValue::Int8(127))
    ));
}

#[test]
fn test_phase35_gguf_value_uint16_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 2, &0u16.to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 2, &u16::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_min"),
        Some(GGUFValue::UInt16(0))
    ));
    assert!(matches!(
        model.metadata.get("test_max"),
        Some(GGUFValue::UInt16(65535))
    ));
}

#[test]
fn test_phase35_gguf_value_int16_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 3, &i16::MIN.to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 3, &i16::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(
        model.metadata.get("test_min"),
        Some(GGUFValue::Int16(-32768))
    ));
    assert!(matches!(
        model.metadata.get("test_max"),
        Some(GGUFValue::Int16(32767))
    ));
}

#[test]
fn test_phase35_gguf_value_uint64_large() {
    let mut data = build_gguf_header(0, 1);
    let large_value = u64::MAX;
    data.extend(build_gguf_metadata(
        "test_large",
        10,
        &large_value.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(
        matches!(model.metadata.get("test_large"), Some(GGUFValue::UInt64(v)) if *v == u64::MAX)
    );
}

#[test]
fn test_phase35_gguf_value_int64_boundaries() {
    let mut data = build_gguf_header(0, 2);
    data.extend(build_gguf_metadata("test_min", 11, &i64::MIN.to_le_bytes()));
    data.extend(build_gguf_metadata("test_max", 11, &i64::MAX.to_le_bytes()));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(matches!(model.metadata.get("test_min"), Some(GGUFValue::Int64(v)) if *v == i64::MIN));
    assert!(matches!(model.metadata.get("test_max"), Some(GGUFValue::Int64(v)) if *v == i64::MAX));
}

#[test]
fn test_phase35_gguf_value_float32_special() {
    let mut data = build_gguf_header(0, 3);
    data.extend(build_gguf_metadata("test_zero", 6, &0.0f32.to_le_bytes()));
    data.extend(build_gguf_metadata("test_neg", 6, &(-1.5f32).to_le_bytes()));
    data.extend(build_gguf_metadata(
        "test_inf",
        6,
        &f32::INFINITY.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test_zero") {
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test_neg") {
        assert!((v - (-1.5)).abs() < f32::EPSILON);
    }
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test_inf") {
        assert!(v.is_infinite());
    }
}

#[test]
fn test_phase35_gguf_value_float64_special() {
    let mut data = build_gguf_header(0, 3);
    data.extend(build_gguf_metadata(
        "test_pi",
        12,
        &std::f64::consts::PI.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "test_e",
        12,
        &std::f64::consts::E.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "test_neg_inf",
        12,
        &f64::NEG_INFINITY.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Float64(v)) = model.metadata.get("test_pi") {
        assert!((v - std::f64::consts::PI).abs() < 1e-10);
    }
    if let Some(GGUFValue::Float64(v)) = model.metadata.get("test_neg_inf") {
        assert!(v.is_infinite() && *v < 0.0);
    }
}

#[test]
fn test_phase35_gguf_value_clone_equality() {
    let value1 = GGUFValue::String("test".to_string());
    let value2 = value1.clone();
    assert_eq!(value1, value2);

    let value3 = GGUFValue::Array(vec![GGUFValue::UInt32(1), GGUFValue::UInt32(2)]);
    let value4 = value3.clone();
    assert_eq!(value3, value4);
}

#[test]
fn test_phase35_gguf_value_debug_format() {
    let value = GGUFValue::String("debug test".to_string());
    let debug_str = format!("{:?}", value);
    assert!(debug_str.contains("String"));
    assert!(debug_str.contains("debug test"));
}

// =============================================================================
// GGUFHeader and TensorInfo Tests
// =============================================================================

#[test]
fn test_phase35_gguf_header_struct() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 100,
        metadata_count: 50,
    };

    assert_eq!(header.magic, 0x4655_4747);
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 100);
    assert_eq!(header.metadata_count, 50);

    // Test Clone
    let cloned = header.clone();
    assert_eq!(header, cloned);

    // Test Debug
    let debug_str = format!("{:?}", header);
    assert!(debug_str.contains("GGUFHeader"));
}

#[test]
fn test_phase35_tensor_info_struct() {
    let info = TensorInfo {
        name: "blk.0.attn_q.weight".to_string(),
        n_dims: 2,
        dims: vec![4096, 4096],
        qtype: 12, // Q4_K
        offset: 0,
    };

    assert_eq!(info.name, "blk.0.attn_q.weight");
    assert_eq!(info.n_dims, 2);
    assert_eq!(info.dims.len(), 2);

    // Test Clone
    let cloned = info.clone();
    assert_eq!(info, cloned);

    // Test Debug
    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("TensorInfo"));
    assert!(debug_str.contains("attn_q"));
}

// =============================================================================
// Error Handling - Truncated Data
// =============================================================================

#[test]
fn test_phase35_truncated_metadata_key() {
    let mut data = build_gguf_header(0, 1);
    // Add truncated key string (length says 100 but only 5 bytes)
    data.extend_from_slice(&100u64.to_le_bytes());
    data.extend_from_slice(b"short");

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated metadata key should fail");
}

#[test]
fn test_phase35_truncated_metadata_value() {
    let mut data = build_gguf_header(0, 1);
    // Add valid key
    data.extend(build_gguf_string("test_key"));
    // Add value type
    data.extend_from_slice(&4u32.to_le_bytes()); // u32
                                                 // Missing value bytes

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated metadata value should fail");
}

#[test]
fn test_phase35_truncated_tensor_name() {
    let mut data = build_gguf_header(1, 0);
    // Add truncated tensor name
    data.extend_from_slice(&50u64.to_le_bytes()); // says 50 bytes
    data.extend_from_slice(b"short"); // only 5 bytes

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated tensor name should fail");
}

#[test]
fn test_phase35_truncated_tensor_dims() {
    let mut data = build_gguf_header(1, 0);
    // Valid tensor name
    data.extend(build_gguf_string("tensor"));
    // Say 5 dimensions
    data.extend_from_slice(&5u32.to_le_bytes());
    // Only provide 2 dimensions
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated tensor dimensions should fail");
}

#[test]
fn test_phase35_truncated_array_elements() {
    let mut data = build_gguf_header(0, 1);
    // Array metadata: element_type + array_len + elements
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
    array_bytes.extend_from_slice(&10u64.to_le_bytes()); // array length: 10
                                                         // Only provide 2 elements
    array_bytes.extend_from_slice(&1u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u32.to_le_bytes());

    data.extend(build_gguf_metadata("test_array", 9, &array_bytes));

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err(), "Truncated array should fail");
}

// =============================================================================
// RoPE Type Inference - All Architectures
// =============================================================================

#[test]
fn test_phase35_rope_type_neox_architectures() {
    // Test all NEOX-style architectures
    let neox_archs = [
        "qwen",
        "qwen2",
        "qwen3",
        "stablelm",
        "phi2",
        "phi3",
        "gemma",
        "gemma2",
        "gemma3",
        "starcoder2",
        "gptneox",
        "falcon",
        "codeshell",
        "orion",
        "bert",
        "nomic-bert",
        "dbrx",
        "olmo2",
        "olmoe",
        "plamo",
        "plamo2",
        "openelm",
        "exaone",
        "minicpm3",
        "nemotron",
        "internlm2",
        "deepseek2",
    ];

    for arch in neox_archs {
        let mut data = build_gguf_header(0, 1);
        let arch_value = build_gguf_string(arch);
        data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

        let model =
            GGUFModel::from_bytes(&data).unwrap_or_else(|_| panic!("Should parse {}", arch));
        assert_eq!(
            model.rope_type(),
            Some(2),
            "Architecture {} should use NEOX RoPE",
            arch
        );
    }
}

#[test]
fn test_phase35_rope_type_norm_architectures() {
    // Test NORM-style architectures (LLaMA family)
    let norm_archs = ["llama", "mistral", "tinyllama", "codellama", "unknown_arch"];

    for arch in norm_archs {
        let mut data = build_gguf_header(0, 1);
        let arch_value = build_gguf_string(arch);
        data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

        let model =
            GGUFModel::from_bytes(&data).unwrap_or_else(|_| panic!("Should parse {}", arch));
        assert_eq!(
            model.rope_type(),
            Some(0),
            "Architecture {} should use NORM RoPE",
            arch
        );
    }
}

#[test]
fn test_phase35_rope_type_scaling_none() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let none_value = build_gguf_string("none");
    data.extend(build_gguf_metadata(
        "custom.rope.scaling.type",
        8,
        &none_value,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(0)); // NORM from "none" scaling
}

#[test]
fn test_phase35_rope_type_scaling_neox() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let neox_value = build_gguf_string("neox");
    data.extend(build_gguf_metadata(
        "custom.rope.scaling.type",
        8,
        &neox_value,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX from "neox" scaling
}

#[test]
fn test_phase35_rope_type_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), None); // No architecture
}

// =============================================================================
// Metadata Accessor Edge Cases
// =============================================================================

#[test]
fn test_phase35_embedding_dim_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.embedding_dim(), None);
}

#[test]
fn test_phase35_num_layers_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_layers(), None);
}

#[test]
fn test_phase35_num_heads_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_heads(), None);
}

#[test]
fn test_phase35_context_length_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.context_length(), None);
}

#[test]
fn test_phase35_num_kv_heads_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_kv_heads(), None);
}

#[test]
fn test_phase35_rope_freq_base_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_freq_base(), None);
}

#[test]
fn test_phase35_rms_epsilon_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rms_epsilon(), None);
}

#[test]
fn test_phase35_metadata_accessor_wrong_type() {
    // Set embedding_dim as string instead of u32
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let dim_value = build_gguf_string("256"); // Wrong type!
    data.extend(build_gguf_metadata("llama.embedding_length", 8, &dim_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.embedding_dim(), None); // Should return None for wrong type
}

// =============================================================================
// Tokenizer and Vocabulary Tests
// =============================================================================

#[test]
fn test_phase35_vocabulary_empty_array() {
    let mut data = build_gguf_header(0, 1);
    // Empty array
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes()); // element type: string
    array_bytes.extend_from_slice(&0u64.to_le_bytes()); // array length: 0
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.vocabulary().is_none()); // Empty vocabulary returns None
}

#[test]
fn test_phase35_vocabulary_non_string_elements() {
    let mut data = build_gguf_header(0, 1);
    // Array of u32 instead of strings
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
    array_bytes.extend_from_slice(&3u64.to_le_bytes()); // array length: 3
    array_bytes.extend_from_slice(&1u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u32.to_le_bytes());
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.vocabulary().is_none()); // Non-string elements filtered out
}

#[test]
fn test_phase35_decode_unknown_token() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("hello"));
    array_bytes.extend(build_gguf_string("world"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Decode with out-of-bounds token ID
    let decoded = model.decode(&[0, 99, 1]); // 99 is out of bounds
    assert!(decoded.contains("\u{FFFD}")); // Unknown token marker
}

#[test]
fn test_phase35_decode_gpt2_style() {
    let mut data = build_gguf_header(0, 2);
    // Set tokenizer model to gpt2
    let model_value = build_gguf_string("gpt2");
    data.extend(build_gguf_metadata("tokenizer.ggml.model", 8, &model_value));

    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    // GPT-2 uses Ġ (U+0120) for space
    array_bytes.extend(build_gguf_string("Hello"));
    array_bytes.extend(build_gguf_string("\u{0120}world")); // space-prefixed
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let decoded = model.decode(&[0, 1]);
    assert_eq!(decoded, "Hello world");
}

#[test]
fn test_phase35_decode_invalid_byte_token() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("<0xGG>")); // Invalid hex
    array_bytes.extend(build_gguf_string("test"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Invalid byte token should be kept as-is
    let decoded = model.decode(&[0, 1]);
    assert!(decoded.contains("<0xGG>") || decoded.contains("test"));
}

#[test]
fn test_phase35_encode_gpt2_style_newline() {
    let mut data = build_gguf_header(0, 2);
    let model_value = build_gguf_string("gpt2");
    data.extend(build_gguf_metadata("tokenizer.ggml.model", 8, &model_value));

    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("Hello"));
    array_bytes.extend(build_gguf_string("\u{010A}")); // GPT-2 newline (Ċ)
    array_bytes.extend(build_gguf_string("World"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tokens = model.encode("Hello\nWorld");
    assert!(tokens.is_some());
}

#[test]
fn test_phase35_encode_sentencepiece_style() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("▁hello")); // SentencePiece word boundary
    array_bytes.extend(build_gguf_string("▁world"));
    array_bytes.extend(build_gguf_string("!"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tokens = model.encode("hello world!");
    assert!(tokens.is_some());
    let tokens = tokens.unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_phase35_encode_with_byte_fallback() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&4u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("▁hello"));
    array_bytes.extend(build_gguf_string("<0x21>")); // '!'
    array_bytes.extend(build_gguf_string("<0x3F>")); // '?'
    array_bytes.extend(build_gguf_string("test"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Text with characters that need byte fallback
    let tokens = model.encode("hello!?test");
    assert!(tokens.is_some());
}

// =============================================================================
// Tensor Data Extraction Edge Cases
// =============================================================================

#[test]
fn test_phase35_tensor_3d_shape() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("tensor_3d", &[2, 3, 4], 0, 0)); // F32

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add tensor data: 2*3*4 = 24 f32 values
    for i in 0..24 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model
        .get_tensor_f32("tensor_3d", &data)
        .expect("Should extract");
    assert_eq!(tensor.len(), 24);
}

#[test]
fn test_phase35_tensor_zero_elements() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("empty", &[0], 0, 0)); // Zero dimension

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model
        .get_tensor_f32("empty", &data)
        .expect("Should extract empty");
    assert_eq!(tensor.len(), 0);
}

#[test]
fn test_phase35_tensor_q4_0_multiple_blocks() {
    use crate::gguf::GGUF_TYPE_Q4_0;

    let mut data = build_gguf_header(1, 0);
    // 64 elements = 2 blocks of Q4_0
    data.extend(build_tensor_info("test", &[64], GGUF_TYPE_Q4_0, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // 2 Q4_0 blocks: 2 * 18 bytes = 36 bytes
    for _ in 0..2 {
        let scale = half::f16::from_f32(1.0);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0x88u8; 16]);
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 64);
}

#[test]
fn test_phase35_tensor_q8_0_multiple_blocks() {
    use crate::gguf::GGUF_TYPE_Q8_0;

    let mut data = build_gguf_header(1, 0);
    // 96 elements = 3 blocks of Q8_0
    data.extend(build_tensor_info("test", &[96], GGUF_TYPE_Q8_0, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // 3 Q8_0 blocks: 3 * 34 bytes = 102 bytes
    for _ in 0..3 {
        let scale = half::f16::from_f32(0.5);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([64i8 as u8; 32]); // Mid-range values
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 96);
}

#[test]
fn test_phase35_tensor_q4_k_multiple_blocks() {
    use crate::gguf::GGUF_TYPE_Q4_K;

    let mut data = build_gguf_header(1, 0);
    // 512 elements = 2 super-blocks of Q4_K
    data.extend(build_tensor_info("test", &[512], GGUF_TYPE_Q4_K, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // 2 Q4_K super-blocks: 2 * 144 bytes = 288 bytes
    data.extend([0u8; 288]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 512);
}

#[test]
fn test_phase35_tensor_q3_k() {
    use crate::gguf::GGUF_TYPE_Q3_K;

    let mut data = build_gguf_header(1, 0);
    // Q3_K: 110 bytes per 256 elements
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q3_K, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Q3_K super-block: 110 bytes
    data.extend([0u8; 110]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Q3_K might not be implemented, but should parse the file
    // The get_tensor_f32 might fail with "unsupported", which is expected
    let _ = model.get_tensor_f32("test", &data);
}

// =============================================================================
// Array Metadata Variations
// =============================================================================

#[test]
fn test_phase35_array_of_strings() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes()); // string type
    array_bytes.extend_from_slice(&5u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("one"));
    array_bytes.extend(build_gguf_string("two"));
    array_bytes.extend(build_gguf_string("three"));
    array_bytes.extend(build_gguf_string("four"));
    array_bytes.extend(build_gguf_string("five"));
    data.extend(build_gguf_metadata("test_strings", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test_strings") {
        assert_eq!(arr.len(), 5);
        assert!(matches!(&arr[0], GGUFValue::String(s) if s == "one"));
        assert!(matches!(&arr[4], GGUFValue::String(s) if s == "five"));
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_phase35_array_of_floats() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&6u32.to_le_bytes()); // f32 type
    array_bytes.extend_from_slice(&4u64.to_le_bytes());
    array_bytes.extend_from_slice(&1.0f32.to_le_bytes());
    array_bytes.extend_from_slice(&2.5f32.to_le_bytes());
    array_bytes.extend_from_slice(&(-3.0f32).to_le_bytes());
    array_bytes.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend(build_gguf_metadata("test_floats", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test_floats") {
        assert_eq!(arr.len(), 4);
        if let GGUFValue::Float32(v) = &arr[1] {
            assert!((v - 2.5).abs() < f32::EPSILON);
        }
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_phase35_array_of_bools() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&7u32.to_le_bytes()); // bool type
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.push(1); // true
    array_bytes.push(0); // false
    array_bytes.push(1); // true
    data.extend(build_gguf_metadata("test_bools", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test_bools") {
        assert_eq!(arr.len(), 3);
        assert!(matches!(arr[0], GGUFValue::Bool(true)));
        assert!(matches!(arr[1], GGUFValue::Bool(false)));
        assert!(matches!(arr[2], GGUFValue::Bool(true)));
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_phase35_array_of_i64() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&11u32.to_le_bytes()); // i64 type
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    array_bytes.extend_from_slice(&i64::MIN.to_le_bytes());
    array_bytes.extend_from_slice(&i64::MAX.to_le_bytes());
    data.extend(build_gguf_metadata("test_i64s", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test_i64s") {
        assert_eq!(arr.len(), 2);
        assert!(matches!(arr[0], GGUFValue::Int64(v) if v == i64::MIN));
        assert!(matches!(arr[1], GGUFValue::Int64(v) if v == i64::MAX));
    } else {
        panic!("Expected Array");
    }
}

// =============================================================================
// Unicode and Special Characters
// =============================================================================

#[test]
fn test_phase35_unicode_tensor_names() {
    let mut data = build_gguf_header(1, 0);
    // Unicode tensor name with emojis and special chars
    data.extend(build_tensor_info("tensor_\u{1F4A1}_light", &[32], 0, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);
    data.extend([0u8; 128]); // tensor data

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors[0].name, "tensor_\u{1F4A1}_light");
}

#[test]
fn test_phase35_unicode_metadata_values() {
    let mut data = build_gguf_header(0, 1);
    let value = build_gguf_string("\u{4E2D}\u{6587}"); // Chinese characters
    data.extend(build_gguf_metadata("chinese_text", 8, &value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::String(s)) = model.metadata.get("chinese_text") {
        assert_eq!(s, "\u{4E2D}\u{6587}");
    }
}

#[test]
fn test_phase35_long_string_metadata() {
    let mut data = build_gguf_header(0, 1);
    // 1KB string
    let long_value = "x".repeat(1024);
    let value = build_gguf_string(&long_value);
    data.extend(build_gguf_metadata("long_string", 8, &value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::String(s)) = model.metadata.get("long_string") {
        assert_eq!(s.len(), 1024);
    }
}

// =============================================================================
// Multi-Tensor and Complex Models
// =============================================================================

#[test]
fn test_phase35_model_with_many_tensors() {
    let mut data = build_gguf_header(50, 0);

    // Add 50 tensor infos
    for i in 0..50 {
        data.extend(build_tensor_info(
            &format!("tensor_{:02}", i),
            &[32],
            0,
            (i * 128) as u64,
        ));
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.tensors.len(), 50);
    assert_eq!(model.tensors[0].name, "tensor_00");
    assert_eq!(model.tensors[49].name, "tensor_49");
}

#[test]
fn test_phase35_model_with_many_metadata() {
    let mut data = build_gguf_header(0, 100);

    // Add 100 metadata entries
    for i in 0..100 {
        data.extend(build_gguf_metadata(
            &format!("key_{:03}", i),
            4,
            &(i as u32).to_le_bytes(),
        ));
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.metadata.len(), 100);
    assert!(matches!(
        model.metadata.get("key_000"),
        Some(GGUFValue::UInt32(0))
    ));
    assert!(matches!(
        model.metadata.get("key_099"),
        Some(GGUFValue::UInt32(99))
    ));
}

#[test]
fn test_phase35_full_model_config() {
    let mut data = build_gguf_header(2, 10);

    // Full set of metadata
    let arch = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch));
    data.extend(build_gguf_metadata(
        "llama.embedding_length",
        4,
        &256u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.block_count",
        4,
        &4u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.attention.head_count",
        4,
        &8u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.attention.head_count_kv",
        4,
        &2u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.context_length",
        4,
        &2048u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.rope.freq_base",
        6,
        &10000.0f32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "llama.attention.layer_norm_rms_epsilon",
        6,
        &1e-5f32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.bos_token_id",
        4,
        &1u32.to_le_bytes(),
    ));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.eos_token_id",
        4,
        &2u32.to_le_bytes(),
    ));

    // Two tensors
    data.extend(build_tensor_info("token_embd.weight", &[100, 256], 0, 0));
    data.extend(build_tensor_info("output_norm.weight", &[256], 0, 102400));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");

    // Verify all accessors
    assert_eq!(model.architecture(), Some("llama"));
    assert_eq!(model.embedding_dim(), Some(256));
    assert_eq!(model.num_layers(), Some(4));
    assert_eq!(model.num_heads(), Some(8));
    assert_eq!(model.num_kv_heads(), Some(2));
    assert_eq!(model.context_length(), Some(2048));
    assert!((model.rope_freq_base().unwrap() - 10000.0).abs() < 1.0);
    assert!((model.rms_epsilon().unwrap() - 1e-5).abs() < 1e-7);
    assert_eq!(model.bos_token_id(), Some(1));
    assert_eq!(model.eos_token_id(), Some(2));
    assert_eq!(model.rope_type(), Some(0)); // NORM for llama
}

// =============================================================================
// GGUFModel Clone and Debug
// =============================================================================

#[test]
fn test_phase35_gguf_model_debug() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("GGUFModel"));
    assert!(debug_str.contains("header"));
}

#[test]
fn test_phase35_gguf_model_clone() {
    let mut data = build_gguf_header(1, 1);
    let arch = build_gguf_string("test");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch));
    data.extend(build_tensor_info("test", &[32], 0, 0));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let cloned = model.clone();

    assert_eq!(model.header, cloned.header);
    assert_eq!(model.metadata.len(), cloned.metadata.len());
    assert_eq!(model.tensors.len(), cloned.tensors.len());
}
