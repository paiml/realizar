//! GGUF Tests Part 28: T-COV-95 Deep Coverage Bridge - loader.rs remaining paths
//!
//! Targets:
//! - get_tensor_f32: Q2_K, F16, Q4_1, Q5_0, Q5_1 dequantization branches
//! - read_value: all metadata type branches (UInt8, Int8, UInt16, Int16, Bool, UInt64, Int64, Float64, Array)
//! - decode: vocabulary-based (SentencePiece, GPT-2, byte tokens)
//! - encode: greedy tokenization with vocabulary
//! - from_bytes: error paths (truncated, wrong version)
//! - rope_type: explicit scaling type metadata
//!
//! Refs PMAT-802: Protocol T-COV-95

use crate::gguf::test_factory::*;
use crate::gguf::GGUFModel;

// ============================================================================
// get_tensor_f32: Q2_K dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q2_k() {
    let n = 256; // one super-block
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q2_k_tensor("test.weight", &[n as u64], &create_q2_k_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(values.is_ok(), "Q2_K get_tensor_f32 failed: {:?}", values.err());
    assert_eq!(values.unwrap().len(), n);
}

#[test]
fn test_get_tensor_f32_q2_k_multi_block() {
    let n = 512; // 2 super-blocks
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q2_k_tensor("test.weight", &[n as u64], &create_q2_k_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data).unwrap();
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: F16 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_f16() {
    let n = 32;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f16_tensor("test.weight", &[n as u64], &create_f16_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(values.is_ok(), "F16 get_tensor_f32 failed: {:?}", values.err());
    let v = values.unwrap();
    assert_eq!(v.len(), n);
}

#[test]
fn test_get_tensor_f32_f16_values_correct() {
    let n = 4;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f16_tensor("test.weight", &[n as u64], &create_f16_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data).unwrap();
    // Values should be approximately 0.0, 0.01, 0.02, 0.03
    assert!((values[0] - 0.0).abs() < 0.01);
    assert!((values[1] - 0.01).abs() < 0.01);
}

// ============================================================================
// get_tensor_f32: Q4_1 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q4_1() {
    let n = 32; // one block
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q4_1_tensor("test.weight", &[n as u64], &create_q4_1_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(values.is_ok(), "Q4_1 get_tensor_f32 failed: {:?}", values.err());
    assert_eq!(values.unwrap().len(), n);
}

#[test]
fn test_get_tensor_f32_q4_1_multi_block() {
    let n = 128; // 4 blocks
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q4_1_tensor("test.weight", &[n as u64], &create_q4_1_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data).unwrap();
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: Q5_0 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q5_0() {
    let n = 32;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_0_tensor("test.weight", &[n as u64], &create_q5_0_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(values.is_ok(), "Q5_0 get_tensor_f32 failed: {:?}", values.err());
    assert_eq!(values.unwrap().len(), n);
}

#[test]
fn test_get_tensor_f32_q5_0_multi_block() {
    let n = 128;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_0_tensor("test.weight", &[n as u64], &create_q5_0_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data).unwrap();
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: Q5_1 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q5_1() {
    let n = 32;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_1_tensor("test.weight", &[n as u64], &create_q5_1_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(values.is_ok(), "Q5_1 get_tensor_f32 failed: {:?}", values.err());
    assert_eq!(values.unwrap().len(), n);
}

#[test]
fn test_get_tensor_f32_q5_1_multi_block() {
    let n = 128;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_1_tensor("test.weight", &[n as u64], &create_q5_1_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let values = model.get_tensor_f32("test.weight", &data).unwrap();
    assert_eq!(values.len(), n);
}

// ============================================================================
// read_value: metadata type branches
// ============================================================================

#[test]
fn test_metadata_u8() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_u8("test.u8_val", 42)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.u8_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_i8() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_i8("test.i8_val", -42)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.i8_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_u16() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_u16("test.u16_val", 1234)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.u16_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_i16() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_i16("test.i16_val", -1234)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.i16_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_i32() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_i32("test.i32_val", -100_000)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.i32_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_bool_true() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_bool("test.bool_val", true)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.bool_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_bool_false() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_bool("test.bool_val", false)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.bool_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_u64() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_u64("test.u64_val", 123_456_789_000)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.u64_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_i64() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_i64("test.i64_val", -123_456_789_000)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.i64_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_f64() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f64("test.f64_val", 3.14159265358979)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.f64_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_string_array() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string_array("test.array", &["hello", "world", "test"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let val = model.metadata.get("test.array");
    assert!(val.is_some());
}

#[test]
fn test_metadata_all_types_in_one_model() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_u8("t.u8", 1)
        .add_i8("t.i8", -1)
        .add_u16("t.u16", 100)
        .add_i16("t.i16", -100)
        .add_i32("t.i32", -50000)
        .add_bool("t.bool", true)
        .add_u64("t.u64", 999)
        .add_i64("t.i64", -999)
        .add_f64("t.f64", 2.718)
        .add_string_array("t.arr", &["a", "b"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    // 4 (arch+dim+layers+heads) + 10 = 14
    assert_eq!(model.metadata.len(), 14);
}

// ============================================================================
// decode: vocabulary-based (SentencePiece style)
// ============================================================================

#[test]
fn test_decode_sentencepiece_style() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["▁Hello", "▁world", "!", "▁test"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    assert!(model.vocabulary().is_some());

    let text = model.decode(&[0, 1, 2]);
    assert!(text.contains("Hello"));
    assert!(text.contains("world"));
    assert!(text.contains("!"));
}

#[test]
fn test_decode_gpt2_style() {
    let data = GGUFBuilder::new()
        .architecture("qwen2")
        .hidden_dim("qwen2", 32)
        .num_layers("qwen2", 1)
        .num_heads("qwen2", 1)
        .add_string("tokenizer.ggml.model", "gpt2")
        .add_string_array("tokenizer.ggml.tokens", &["H", "e", "l", "o"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    assert!(model.vocabulary().is_some());

    let text = model.decode(&[0, 1, 2, 2, 3]);
    // GPT-2 unicode decode maps ASCII chars directly
    assert!(!text.is_empty());
}

#[test]
fn test_decode_byte_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["<0x48>", "<0x69>", "!"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();

    let text = model.decode(&[0, 1, 2]);
    // <0x48> = 'H', <0x69> = 'i'
    assert!(text.contains('H'));
    assert!(text.contains('i'));
    assert!(text.contains('!'));
}

#[test]
fn test_decode_unknown_token_id() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["hello"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();

    // Token ID 999 is out of vocabulary bounds
    let text = model.decode(&[999]);
    assert!(text.contains('�') || text.contains('?'));
}

#[test]
fn test_decode_empty_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["hello"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();

    let text = model.decode(&[]);
    assert!(text.is_empty());
}

// ============================================================================
// encode: vocabulary-based
// ============================================================================

#[test]
fn test_encode_sentencepiece_basic() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["▁Hello", "▁world", "!"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();

    let tokens = model.encode("Hello world!");
    assert!(tokens.is_some());
}

#[test]
fn test_encode_no_vocabulary_returns_none() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    assert!(model.encode("test").is_none());
}

#[test]
fn test_encode_empty_text() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["hello"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();

    let tokens = model.encode("");
    assert!(tokens.is_some());
    // Empty text produces empty or minimal tokens
}

// ============================================================================
// from_bytes: error paths
// ============================================================================

#[test]
fn test_from_bytes_empty() {
    let result = GGUFModel::from_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_truncated_4_bytes() {
    let result = GGUFModel::from_bytes(&[0x47, 0x47, 0x55, 0x46]);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_truncated_8_bytes() {
    // Valid magic but truncated version
    let mut data = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_wrong_magic() {
    let mut data = Vec::new();
    data.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("magic") || err.contains("Invalid") || err.contains("Magic"));
}

#[test]
fn test_from_bytes_wrong_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&2u32.to_le_bytes()); // version 2 (unsupported)
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("version") || err.contains("Unsupported"));
}

#[test]
fn test_from_bytes_truncated_16_bytes() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    // missing metadata count
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// rope_type: explicit scaling type metadata
// ============================================================================

#[test]
fn test_rope_type_scaling_none() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_string("llama.rope.scaling.type", "none")
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    assert_eq!(model.rope_type(), Some(0)); // NORM
}

#[test]
fn test_rope_type_scaling_linear() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_string("llama.rope.scaling.type", "linear")
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    assert_eq!(model.rope_type(), Some(0)); // NORM
}

#[test]
fn test_rope_type_scaling_yarn() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_string("llama.rope.scaling.type", "yarn")
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_scaling_neox() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_string("llama.rope.scaling.type", "neox")
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

// ============================================================================
// vocabulary: edge cases
// ============================================================================

#[test]
fn test_vocabulary_returns_some_with_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string_array("tokenizer.ggml.tokens", &["a", "b", "c"])
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    let vocab = model.vocabulary();
    assert!(vocab.is_some());
    assert_eq!(vocab.unwrap().len(), 3);
}

#[test]
fn test_vocabulary_returns_none_without_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .build();
    let model = GGUFModel::from_bytes(&data).unwrap();
    assert!(model.vocabulary().is_none());
}

// ============================================================================
// get_tensor_f32: unsupported qtype error path
// ============================================================================

#[test]
fn test_get_tensor_f32_unsupported_qtype() {
    // Build raw GGUF with unknown qtype (99)
    let mut data = Vec::new();
    // Header
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata
    // Tensor info: name "bad.weight", 1 dim [4], qtype=99, offset=0
    let name = "bad.weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
    data.extend_from_slice(&99u32.to_le_bytes()); // qtype = 99 (unsupported)
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    // Align to 32 bytes
    let aligned = data.len().div_ceil(32) * 32;
    data.resize(aligned, 0);
    // Tensor data (some bytes)
    data.extend_from_slice(&[0u8; 64]);

    let model = GGUFModel::from_bytes(&data).unwrap();
    let result = model.get_tensor_f32("bad.weight", &data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Unsupported") || err.contains("quantization"));
}

// ============================================================================
// Mixed quant types including new ones
// ============================================================================

#[test]
fn test_model_with_all_quant_types() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f32_tensor("f32.weight", &[32], &vec![1.0f32; 32])
        .add_q4_0_tensor("q4_0.weight", &[32], &create_q4_0_data(32))
        .add_q8_0_tensor("q8_0.weight", &[32], &create_q8_0_data(32))
        .add_q4_k_tensor("q4_k.weight", &[256], &create_q4_k_data(256))
        .add_q5_k_tensor("q5_k.weight", &[256], &create_q5_k_data(256))
        .add_q6_k_tensor("q6_k.weight", &[256], &create_q6_k_data(256))
        .add_q2_k_tensor("q2_k.weight", &[256], &create_q2_k_data(256))
        .add_f16_tensor("f16.weight", &[32], &create_f16_data(32))
        .add_q4_1_tensor("q4_1.weight", &[32], &create_q4_1_data(32))
        .add_q5_0_tensor("q5_0.weight", &[32], &create_q5_0_data(32))
        .add_q5_1_tensor("q5_1.weight", &[32], &create_q5_1_data(32))
        .build();

    let model = GGUFModel::from_bytes(&data).unwrap();
    assert_eq!(model.tensors.len(), 11);

    // Verify each tensor can be dequantized
    for tensor in &model.tensors {
        let result = model.get_tensor_f32(&tensor.name, &data);
        assert!(
            result.is_ok(),
            "Failed to dequantize {}: {:?}",
            tensor.name,
            result.err()
        );
    }
}
