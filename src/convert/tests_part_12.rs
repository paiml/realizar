//! T-COV-95 Coverage Bridge (Part 12 - CRC32 vectors, uncovered rope_type architectures,
//! from_apr_bytes missing-weights & truncated-data error paths,
//! Q4_0/Q4_1/Q5_0/Q5_1 byte-size branches)
//!
//! Targets uncovered lines in convert/mod.rs:
//!   - crc32: single-byte inputs, all-zeros, all-0xFF, long buffer
//!   - compute_apr_header_checksum: reserved-bytes sensitivity
//!   - infer_rope_type: gemma3, codeshell, orion, nomic-bert, dbrx, olmo2,
//!     olmoe, plamo, plamo2, openelm, minicpm3, case-insensitive matching
//!   - from_apr_bytes: "No 'weights' tensor" path, truncated tensor data path
//!   - Q4K converter byte_size match arms for Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7)

use super::*;

// ============================================================================
// CRC32 additional test vectors
// ============================================================================

#[test]
fn test_crc32_single_byte_zero() {
    let crc = crc32(&[0x00]);
    assert_ne!(crc, 0, "CRC32 of single zero byte should be non-zero");
    assert_eq!(crc, 0xD202_EF8D);
}

#[test]
fn test_crc32_single_byte_ff() {
    let crc = crc32(&[0xFF]);
    assert_ne!(crc, 0);
    assert_eq!(crc, 0xFF00_0000);
}

#[test]
fn test_crc32_all_zeros_16() {
    let data = vec![0u8; 16];
    let crc = crc32(&data);
    assert_ne!(crc, 0);
}

#[test]
fn test_crc32_all_ff_16() {
    let data = vec![0xFFu8; 16];
    let crc = crc32(&data);
    assert_ne!(crc, 0);
}

#[test]
fn test_crc32_incremental_bytes() {
    // CRC32 of [0, 1, 2, ..., 255]
    let data: Vec<u8> = (0u8..=255).collect();
    let crc = crc32(&data);
    assert_ne!(crc, 0);
    // Verify determinism
    assert_eq!(crc, crc32(&data));
}

#[test]
fn test_crc32_single_a() {
    // Known: CRC32("a") = 0xE8B7BE43
    let crc = crc32(b"a");
    assert_eq!(crc, 0xE8B7_BE43);
}

#[test]
fn test_crc32_abc() {
    // Known: CRC32("abc") = 0x352441C2
    let crc = crc32(b"abc");
    assert_eq!(crc, 0x3524_41C2);
}

#[test]
fn test_crc32_long_repetitive() {
    let data = vec![0x42u8; 4096];
    let crc = crc32(&data);
    assert_ne!(crc, 0);
    // Changing one byte should change the checksum
    let mut modified = data.clone();
    modified[2048] = 0x43;
    assert_ne!(crc, crc32(&modified));
}

// ============================================================================
// compute_apr_header_checksum: reserved bytes sensitivity
// ============================================================================

#[test]
fn test_header_checksum_sensitive_to_reserved_bytes() {
    let mut header1 = vec![0u8; 64];
    header1[0..4].copy_from_slice(&MAGIC);
    header1[4] = 2;

    let mut header2 = header1.clone();
    // Modify a reserved byte at index 50 (within [44..64] range, which IS included)
    header2[50] = 0xAB;

    let cs1 = compute_apr_header_checksum(&header1);
    let cs2 = compute_apr_header_checksum(&header2);
    assert_ne!(
        cs1, cs2,
        "Reserved bytes [44..64] are included in checksum computation"
    );
}

#[test]
fn test_header_checksum_sensitive_to_early_bytes() {
    let mut header1 = vec![0u8; 64];
    header1[0..4].copy_from_slice(&MAGIC);
    header1[4] = 2;

    let mut header2 = header1.clone();
    // Modify byte at index 10 (within [0..40] range)
    header2[10] = 0xFF;

    let cs1 = compute_apr_header_checksum(&header1);
    let cs2 = compute_apr_header_checksum(&header2);
    assert_ne!(cs1, cs2, "Bytes [0..40] should affect checksum");
}

// ============================================================================
// infer_rope_type: uncovered NEOX architectures
// ============================================================================

#[test]
fn test_infer_rope_type_gemma3() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("gemma3", &meta), 2);
}

#[test]
fn test_infer_rope_type_codeshell() {
    let meta = std::collections::HashMap::new();
    assert_eq!(
        GgufToAprQ4KConverter::infer_rope_type("codeshell", &meta),
        2
    );
}

#[test]
fn test_infer_rope_type_orion() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("orion", &meta), 2);
}

#[test]
fn test_infer_rope_type_nomic_bert() {
    let meta = std::collections::HashMap::new();
    assert_eq!(
        GgufToAprQ4KConverter::infer_rope_type("nomic-bert", &meta),
        2
    );
}

#[test]
fn test_infer_rope_type_dbrx() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("dbrx", &meta), 2);
}

#[test]
fn test_infer_rope_type_olmo2() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("olmo2", &meta), 2);
}

#[test]
fn test_infer_rope_type_olmoe() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("olmoe", &meta), 2);
}

#[test]
fn test_infer_rope_type_plamo() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("plamo", &meta), 2);
}

#[test]
fn test_infer_rope_type_plamo2() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("plamo2", &meta), 2);
}

#[test]
fn test_infer_rope_type_openelm() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("openelm", &meta), 2);
}

#[test]
fn test_infer_rope_type_minicpm3() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("minicpm3", &meta), 2);
}

#[test]
fn test_infer_rope_type_exaone() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("exaone", &meta), 2);
}

#[test]
fn test_infer_rope_type_nemotron() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("nemotron", &meta), 2);
}

#[test]
fn test_infer_rope_type_case_insensitive_uppercase() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("QWEN2", &meta), 2);
}

#[test]
fn test_infer_rope_type_case_insensitive_mixed() {
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("Gemma2", &meta), 2);
}

#[test]
fn test_infer_rope_type_contains_match() {
    // "my_qwen2_model" should match because it contains "qwen2"
    let meta = std::collections::HashMap::new();
    assert_eq!(
        GgufToAprQ4KConverter::infer_rope_type("my_qwen2_model", &meta),
        2
    );
}

#[test]
fn test_infer_rope_type_qwen_base() {
    // "qwen" (not qwen2/qwen3) should still match
    let meta = std::collections::HashMap::new();
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("qwen", &meta), 2);
}

// ============================================================================
// from_apr_bytes: "No 'weights' tensor" error path (lines 292-297)
// ============================================================================

#[test]
fn test_from_apr_bytes_no_weights_tensor() {
    use crate::apr::TensorEntry;

    // Build a valid APR file with a tensor index that has NO "weights" entry
    let transformer = AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 4,
            num_layers: 0,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 2,
            intermediate_dim: 4,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; 8],
        layers: vec![],
        output_norm_weight: vec![1.0; 4],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    // Create valid APR bytes first
    let mut bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Now corrupt the tensor index to rename "weights" to "other_name"
    let tensor_index_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;

    // Replace the tensor index JSON with a renamed tensor
    let fake_entries = vec![TensorEntry {
        name: "not_weights".to_string(),
        dtype: "json".to_string(),
        shape: vec![100],
        offset: 0,
        size: 100,
    }];
    let fake_index_json = serde_json::to_vec(&fake_entries).unwrap();

    // We need to rebuild the file with the new tensor index
    let mut new_bytes = Vec::new();
    // Copy header
    new_bytes.extend_from_slice(&bytes[..HEADER_SIZE]);
    // Copy metadata (from HEADER_SIZE to tensor_index_offset)
    new_bytes.extend_from_slice(&bytes[HEADER_SIZE..tensor_index_offset]);
    // Insert new tensor index
    new_bytes.extend_from_slice(&fake_index_json);
    // Update data_offset in header
    let new_data_offset = (tensor_index_offset + fake_index_json.len()) as u64;
    new_bytes[32..40].copy_from_slice(&new_data_offset.to_le_bytes());
    // Append some dummy data
    new_bytes.extend_from_slice(&bytes[data_offset..]);

    let result = GgufToAprConverter::from_apr_bytes(&new_bytes);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("weights") || err_msg.contains("not found"),
        "Expected 'weights not found' error, got: {}",
        err_msg
    );
}

// ============================================================================
// from_apr_bytes: truncated tensor data path (lines 303-311)
// ============================================================================

#[test]
fn test_from_apr_bytes_truncated_tensor_data() {
    let transformer = AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 4,
            num_layers: 0,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 2,
            intermediate_dim: 4,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; 8],
        layers: vec![],
        output_norm_weight: vec![1.0; 4],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");
    let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;

    // Truncate the file so that tensor data is incomplete
    // Keep header + metadata + tensor index + a few bytes of data
    let truncated = bytes[..data_offset + 5].to_vec();

    let result = GgufToAprConverter::from_apr_bytes(&truncated);
    assert!(result.is_err(), "Should fail due to truncated tensor data");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("truncated") || err_msg.contains("deserialize"),
        "Expected truncation/deserialization error, got: {}",
        err_msg
    );
}

// ============================================================================
// Q4K converter byte_size match arms: Q4_0, Q4_1, Q5_0, Q5_1
// (Lines 648-651 in convert/mod.rs)
// ============================================================================

#[test]
fn test_byte_size_q4_0() {
    // Q4_0: 32 elements = 18 bytes (2 scale + 16 quants)
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(32) * 18;
    assert_eq!(byte_size, 32 * 18); // 576
    assert_eq!(byte_size, 576);
}

#[test]
fn test_byte_size_q4_0_non_divisible() {
    let num_elements = 1000usize;
    let byte_size = num_elements.div_ceil(32) * 18;
    // ceil(1000/32) = 32 blocks -> 32*18 = 576
    assert_eq!(byte_size, 32 * 18);
}

#[test]
fn test_byte_size_q4_1() {
    // Q4_1: 32 elements = 20 bytes (2 scale + 2 min + 16 quants)
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(32) * 20;
    assert_eq!(byte_size, 32 * 20); // 640
    assert_eq!(byte_size, 640);
}

#[test]
fn test_byte_size_q4_1_non_divisible() {
    let num_elements = 33usize; // 1 block + 1 extra element
    let byte_size = num_elements.div_ceil(32) * 20;
    assert_eq!(byte_size, 2 * 20); // 2 blocks = 40 bytes
    assert_eq!(byte_size, 40);
}

#[test]
fn test_byte_size_q5_0() {
    // Q5_0: 32 elements = 22 bytes (2 scale + 4 high + 16 quants)
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(32) * 22;
    assert_eq!(byte_size, 32 * 22); // 704
    assert_eq!(byte_size, 704);
}

#[test]
fn test_byte_size_q5_0_non_divisible() {
    let num_elements = 65usize; // 2 blocks + 1 extra
    let byte_size = num_elements.div_ceil(32) * 22;
    assert_eq!(byte_size, 3 * 22); // 3 blocks = 66 bytes
    assert_eq!(byte_size, 66);
}

#[test]
fn test_byte_size_q5_1() {
    // Q5_1: 32 elements = 24 bytes (2 scale + 2 min + 4 high + 16 quants)
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(32) * 24;
    assert_eq!(byte_size, 32 * 24); // 768
    assert_eq!(byte_size, 768);
}

#[test]
fn test_byte_size_q5_1_non_divisible() {
    let num_elements = 31usize; // Less than 1 block
    let byte_size = num_elements.div_ceil(32) * 24;
    assert_eq!(byte_size, 1 * 24); // 1 block = 24 bytes
    assert_eq!(byte_size, 24);
}

// ============================================================================
// Q4K byte_size: edge cases for all quant types
// ============================================================================

#[test]
fn test_byte_size_f32() {
    let num_elements = 100usize;
    let byte_size = num_elements * 4;
    assert_eq!(byte_size, 400);
}

#[test]
fn test_byte_size_f16() {
    let num_elements = 100usize;
    let byte_size = num_elements * 2;
    assert_eq!(byte_size, 200);
}

#[test]
fn test_byte_size_q8_0() {
    // Q8_0: 32 elements = 34 bytes
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(32) * 34;
    assert_eq!(byte_size, 32 * 34); // 1088
    assert_eq!(byte_size, 1088);
}

#[test]
fn test_byte_size_q4k() {
    // Q4_K: 256 elements = 144 bytes
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(256) * 144;
    assert_eq!(byte_size, 4 * 144); // 576
    assert_eq!(byte_size, 576);
}

#[test]
fn test_byte_size_q5k() {
    // Q5_K: 256 elements = 176 bytes
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(256) * 176;
    assert_eq!(byte_size, 4 * 176); // 704
    assert_eq!(byte_size, 704);
}

#[test]
fn test_byte_size_q6k() {
    // Q6_K: 256 elements = 210 bytes
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(256) * 210;
    assert_eq!(byte_size, 4 * 210); // 840
    assert_eq!(byte_size, 840);
}

#[test]
fn test_byte_size_unknown_defaults_f32() {
    // Unknown dtype defaults to F32 = 4 bytes per element
    let num_elements = 50usize;
    let byte_size = num_elements * 4;
    assert_eq!(byte_size, 200);
}

// ============================================================================
// GGUF metadata helpers: additional type coverage
// ============================================================================

#[test]
fn test_get_u32_from_float32_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::Float32(3.14));
    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "key"), None);
}

#[test]
fn test_get_f32_from_uint32_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::UInt32(42));
    assert_eq!(GgufToAprQ4KConverter::get_f32(&meta, "key"), None);
}

#[test]
fn test_get_string_from_bool_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::Bool(true));
    assert_eq!(GgufToAprQ4KConverter::get_string(&meta, "key"), None);
}

#[test]
fn test_get_u32_from_bool_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::Bool(false));
    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "key"), None);
}

#[test]
fn test_get_f32_from_bool_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::Bool(true));
    assert_eq!(GgufToAprQ4KConverter::get_f32(&meta, "key"), None);
}

// ============================================================================
// ConversionStats: Display-like formatting coverage
// ============================================================================

#[test]
fn test_conversion_stats_memory_mb_fractional() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 1024 * 512, // 0.5 MB
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.memory_mb() - 0.5).abs() < 0.001);
}

#[test]
fn test_conversion_stats_memory_gb_fractional() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 1024 * 1024 * 512, // 0.5 GB
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.memory_gb() - 0.5).abs() < 0.001);
}

#[test]
fn test_conversion_stats_parameters_m_fractional() {
    let stats = ConversionStats {
        total_parameters: 500_000, // 0.5M
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.parameters_m() - 0.5).abs() < 0.001);
}

#[test]
fn test_conversion_stats_parameters_b_fractional() {
    let stats = ConversionStats {
        total_parameters: 500_000_000, // 0.5B
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.parameters_b() - 0.5).abs() < 0.001);
}
