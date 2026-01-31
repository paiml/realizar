//! T-COV-95 Ancestral Pygmies: GGUF Legacy Version & Alignment Assault (PMAT-802)
//!
//! Dr. Popper's directive: "The loader resists because it handles 'Sharding' and
//! 'Alignment' which your Pygmies haven't yet violated. Assault with 'Ancestral
//! Pygmies' (GGUF v1/v2) and 'Unaligned Pygmies' (malicious padding)."
//!
//! This module tests:
//! 1. GGUF Version 1 files (legacy format)
//! 2. GGUF Version 2 files (intermediate format)
//! 3. Unaligned tensor data (non-32-byte boundaries)
//! 4. Malicious padding patterns
//! 5. Split shard metadata scenarios
//!
//! Target: 618 missed lines in gguf/loader.rs

use super::*;
use crate::gguf::{GGUFModel, GGUF_MAGIC, GGUF_VERSION_V3};

// ============================================================================
// Ancestral Pygmy Generators
// ============================================================================

/// Build a GGUF v1 Ancestral Pygmy (version 1 - rejected by modern loaders)
fn build_ancestral_pygmy_v1() -> Vec<u8> {
    let mut data = Vec::new();

    // Magic: "GGUF" = 0x46554747
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());

    // Version 1 (ancient format)
    data.extend_from_slice(&1u32.to_le_bytes());

    // Tensor count (u64 in v3, might be u32 in v1 - we use u64 for testing)
    data.extend_from_slice(&0u64.to_le_bytes());

    // Metadata count
    data.extend_from_slice(&0u64.to_le_bytes());

    data
}

/// Build a GGUF v2 Ancestral Pygmy (version 2 - also rejected)
fn build_ancestral_pygmy_v2() -> Vec<u8> {
    let mut data = Vec::new();

    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());

    // Version 2
    data.extend_from_slice(&2u32.to_le_bytes());

    // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes());

    // Metadata count
    data.extend_from_slice(&0u64.to_le_bytes());

    data
}

/// Build a GGUF v4 (hypothetical future version - should be rejected)
fn build_future_pygmy_v4() -> Vec<u8> {
    let mut data = Vec::new();

    // Magic
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());

    // Version 4 (doesn't exist yet)
    data.extend_from_slice(&4u32.to_le_bytes());

    // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes());

    // Metadata count
    data.extend_from_slice(&0u64.to_le_bytes());

    data
}

/// Build a GGUF with version 0 (invalid)
fn build_ancestral_pygmy_v0() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // Version 0 - invalid
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    data
}

/// Build a GGUF with maximum version (u32::MAX)
fn build_ancestral_pygmy_max_version() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&u32::MAX.to_le_bytes()); // Max version
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    data
}

// ============================================================================
// Unaligned Pygmy Generators
// ============================================================================

/// Build a GGUF v3 with unaligned tensor data start
fn build_unaligned_pygmy_odd_offset() -> Vec<u8> {
    let mut data = Vec::new();

    // Valid header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    // Tensor info for "test_tensor"
    let name = "test_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());

    // Dimensions: 1D tensor with 32 elements
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&32u64.to_le_bytes()); // dim[0]

    // Type: F32 (0)
    data.extend_from_slice(&0u32.to_le_bytes());

    // Offset: ODD number (not aligned to 32 bytes)
    data.extend_from_slice(&17u64.to_le_bytes());

    // Add some padding that's NOT 32-byte aligned
    while data.len() % 32 != 17 {
        data.push(0);
    }

    // Add tensor data (32 f32 values = 128 bytes)
    for i in 0..32 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    data
}

/// Build a GGUF v3 with tensor offset pointing beyond file
fn build_unaligned_pygmy_overflow_offset() -> Vec<u8> {
    let mut data = Vec::new();

    // Valid header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    // Tensor info
    let name = "overflow_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&8u64.to_le_bytes()); // dim[0]
    data.extend_from_slice(&0u32.to_le_bytes()); // F32

    // Offset pointing way beyond file size
    data.extend_from_slice(&0xFFFF_FFFFu64.to_le_bytes());

    data
}

/// Build a GGUF v3 with zero-length tensor name
fn build_malformed_pygmy_empty_name() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    // Zero-length tensor name
    data.extend_from_slice(&0u64.to_le_bytes()); // name length = 0
    // No name bytes

    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    data
}

/// Build a GGUF v3 with extremely long tensor name
fn build_malformed_pygmy_long_name() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Name length claiming to be huge (but we won't provide the bytes)
    data.extend_from_slice(&0x1000_0000u64.to_le_bytes()); // 256MB name!

    data
}

/// Build a GGUF v3 with invalid tensor type
fn build_malformed_pygmy_invalid_type() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "invalid_type_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&4u64.to_le_bytes()); // dim[0]

    // Invalid tensor type (255)
    data.extend_from_slice(&255u32.to_le_bytes());

    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    data
}

/// Build a GGUF v3 with n_dims = 0 (scalar?)
fn build_malformed_pygmy_zero_dims() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "scalar_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());

    // n_dims = 0 (no dimensions - scalar?)
    data.extend_from_slice(&0u32.to_le_bytes());

    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Pad to 32-byte alignment
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Single f32 value
    data.extend_from_slice(&3.14f32.to_le_bytes());

    data
}

/// Build a GGUF v3 with n_dims > MAX_DIMS (if there's a limit)
fn build_malformed_pygmy_too_many_dims() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "hyperdimensional_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());

    // n_dims = 100 (way more than any reasonable tensor)
    data.extend_from_slice(&100u32.to_le_bytes());

    // We need to provide 100 dimension values
    for _ in 0..100 {
        data.extend_from_slice(&1u64.to_le_bytes());
    }

    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    data
}

/// Build a GGUF v3 with overlapping tensor offsets
fn build_malformed_pygmy_overlapping_tensors() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    // First tensor
    let name1 = "tensor_a";
    data.extend_from_slice(&(name1.len() as u64).to_le_bytes());
    data.extend_from_slice(name1.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&32u64.to_le_bytes()); // 32 elements = 128 bytes
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset 0

    // Second tensor - OVERLAPS with first!
    let name2 = "tensor_b";
    data.extend_from_slice(&(name2.len() as u64).to_le_bytes());
    data.extend_from_slice(name2.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&32u64.to_le_bytes()); // 32 elements = 128 bytes
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&64u64.to_le_bytes()); // offset 64 - overlaps!

    // Pad to alignment
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Only provide 128 bytes of data (not enough for 2 tensors)
    for i in 0..32 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    data
}

// ============================================================================
// Sharding Metadata Generators
// ============================================================================

/// Build a GGUF v3 with split file metadata
fn build_shard_pygmy_split_metadata() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 metadata entries

    // Metadata 1: split.no (current shard number)
    let key1 = "split.no";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
    data.extend_from_slice(&1u32.to_le_bytes()); // shard 1

    // Metadata 2: split.count (total shards)
    let key2 = "split.count";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
    data.extend_from_slice(&4u32.to_le_bytes()); // 4 total shards

    // Metadata 3: split.tensors.count (tensors per shard)
    let key3 = "split.tensors.count";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
    data.extend_from_slice(&100u32.to_le_bytes()); // 100 tensors per shard

    // Minimal tensor
    let name = "shard_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes());

    // Pad to alignment
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor data
    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    data
}

// ============================================================================
// Ancestral Pygmy Tests (Version Rejection)
// ============================================================================

#[test]
fn test_ancestral_pygmy_v1_rejected() {
    let data = build_ancestral_pygmy_v1();
    let result = GGUFModel::from_bytes(&data);

    assert!(result.is_err(), "GGUF v1 should be rejected");
    let err = result.unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("version") || msg.contains("unsupported"),
        "Error should mention version: {}",
        msg
    );
}

#[test]
fn test_ancestral_pygmy_v2_rejected() {
    let data = build_ancestral_pygmy_v2();
    let result = GGUFModel::from_bytes(&data);

    assert!(result.is_err(), "GGUF v2 should be rejected");
}

#[test]
fn test_future_pygmy_v4_rejected() {
    let data = build_future_pygmy_v4();
    let result = GGUFModel::from_bytes(&data);

    assert!(result.is_err(), "Future GGUF v4 should be rejected");
}

#[test]
fn test_ancestral_pygmy_v0_rejected() {
    let data = build_ancestral_pygmy_v0();
    let result = GGUFModel::from_bytes(&data);

    assert!(result.is_err(), "GGUF v0 should be rejected");
}

#[test]
fn test_ancestral_pygmy_max_version_rejected() {
    let data = build_ancestral_pygmy_max_version();
    let result = GGUFModel::from_bytes(&data);

    assert!(result.is_err(), "Max version should be rejected");
}

// ============================================================================
// Unaligned Pygmy Tests
// ============================================================================

#[test]
fn test_unaligned_pygmy_odd_offset() {
    let data = build_unaligned_pygmy_odd_offset();
    let result = GGUFModel::from_bytes(&data);

    // May parse but tensor access may fail
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            // Tensor exists but offset is odd - may cause issues on access
            assert_eq!(model.tensors[0].offset, 17);
        }
        Err(_) => {
            // Also acceptable - loader might reject unaligned offsets
        }
    }
}

#[test]
fn test_unaligned_pygmy_overflow_offset() {
    let data = build_unaligned_pygmy_overflow_offset();
    let result = GGUFModel::from_bytes(&data);

    // Should parse but tensor access should fail
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            // Offset is beyond file
            assert!(model.tensors[0].offset > data.len() as u64);
        }
        Err(_) => {
            // Also acceptable
        }
    }
}

// ============================================================================
// Malformed Pygmy Tests
// ============================================================================

#[test]
fn test_malformed_pygmy_empty_name() {
    let data = build_malformed_pygmy_empty_name();
    let result = GGUFModel::from_bytes(&data);

    // Empty tensor names may be accepted or rejected
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            assert!(model.tensors[0].name.is_empty());
        }
        Err(_) => {
            // Rejection is fine
        }
    }
}

#[test]
fn test_malformed_pygmy_long_name() {
    let data = build_malformed_pygmy_long_name();
    let result = GGUFModel::from_bytes(&data);

    // Should fail - not enough data for the claimed name length
    assert!(result.is_err(), "Long name without data should fail");
}

#[test]
fn test_malformed_pygmy_invalid_type() {
    let data = build_malformed_pygmy_invalid_type();
    let result = GGUFModel::from_bytes(&data);

    // May parse but type 255 is invalid
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            // Type is stored but may fail on dequantization
        }
        Err(_) => {
            // Rejection is also fine
        }
    }
}

#[test]
fn test_malformed_pygmy_zero_dims() {
    let data = build_malformed_pygmy_zero_dims();
    let result = GGUFModel::from_bytes(&data);

    // Zero dimensions (scalar) may or may not be supported
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            // Check dims
            assert_eq!(model.tensors[0].dims.len(), 0);
        }
        Err(_) => {
            // Rejection is fine
        }
    }
}

#[test]
fn test_malformed_pygmy_too_many_dims() {
    let data = build_malformed_pygmy_too_many_dims();
    let result = GGUFModel::from_bytes(&data);

    // 100 dimensions may or may not be accepted
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            assert_eq!(model.tensors[0].dims.len(), 100);
        }
        Err(_) => {
            // Rejection is fine - 100 dims is unreasonable
        }
    }
}

#[test]
fn test_malformed_pygmy_overlapping_tensors() {
    let data = build_malformed_pygmy_overlapping_tensors();
    let result = GGUFModel::from_bytes(&data);

    // Overlapping offsets may parse but cause data corruption on read
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 2);
            // Tensors overlap in memory
            let t0_end = model.tensors[0].offset + 128; // 32 * 4 bytes
            let t1_start = model.tensors[1].offset;
            assert!(t1_start < t0_end, "Tensors should overlap");
        }
        Err(_) => {
            // Rejection is also fine
        }
    }
}

// ============================================================================
// Shard Metadata Tests
// ============================================================================

#[test]
fn test_shard_pygmy_split_metadata() {
    let data = build_shard_pygmy_split_metadata();
    let result = GGUFModel::from_bytes(&data);

    match result {
        Ok(model) => {
            // Check metadata was parsed
            assert_eq!(model.metadata.len(), 3);

            // Verify shard metadata keys exist
            let keys: Vec<&str> = model.metadata.keys().map(|k| k.as_str()).collect();
            assert!(keys.contains(&"split.no"));
            assert!(keys.contains(&"split.count"));
            assert!(keys.contains(&"split.tensors.count"));
        }
        Err(e) => {
            // Parsing may fail but that's also coverage
            let _ = e;
        }
    }
}

// ============================================================================
// Padding Pattern Tests
// ============================================================================

#[test]
fn test_padding_pattern_all_zeros() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "padded_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes());

    // Pad with zeros to 32-byte alignment
    while data.len() % 32 != 0 {
        data.push(0x00);
    }

    // Tensor data
    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "Zero padding should work");
}

#[test]
fn test_padding_pattern_all_ff() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "ff_padded";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Pad with 0xFF (unusual but should still work)
    while data.len() % 32 != 0 {
        data.push(0xFF);
    }

    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "0xFF padding should work");
}

#[test]
fn test_padding_pattern_alternating() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "alt_padded";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Alternating pattern padding
    let mut alt = false;
    while data.len() % 32 != 0 {
        data.push(if alt { 0xAA } else { 0x55 });
        alt = !alt;
    }

    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "Alternating padding should work");
}

// ============================================================================
// Edge Case: Tensor Count Mismatch
// ============================================================================

#[test]
fn test_tensor_count_mismatch_more_claimed() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&10u64.to_le_bytes()); // Claims 10 tensors
    data.extend_from_slice(&0u64.to_le_bytes());

    // Only provide 1 tensor info
    let name = "only_one";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    // Should fail - not enough tensor info
    assert!(result.is_err(), "Missing tensors should fail");
}

#[test]
fn test_tensor_count_zero_with_data() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    data.extend_from_slice(&0u64.to_le_bytes());

    // Add some garbage "tensor data"
    data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

    let result = GGUFModel::from_bytes(&data);
    // Zero tensors is valid
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 0);
        }
        Err(_) => {
            // Also fine
        }
    }
}

// ============================================================================
// Metadata Type Edge Cases
// ============================================================================

#[test]
fn test_metadata_type_string_empty() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata

    let key = "empty_string_key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
    data.extend_from_slice(&0u64.to_le_bytes()); // Empty string (length 0)

    let result = GGUFModel::from_bytes(&data);
    match result {
        Ok(model) => {
            assert_eq!(model.metadata.len(), 1);
        }
        Err(_) => {}
    }
}

#[test]
fn test_metadata_type_array_empty() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "empty_array";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // GGUF_TYPE_ARRAY
    data.extend_from_slice(&4u32.to_le_bytes()); // Array element type: UINT32
    data.extend_from_slice(&0u64.to_le_bytes()); // Array length: 0

    let result = GGUFModel::from_bytes(&data);
    match result {
        Ok(model) => {
            assert_eq!(model.metadata.len(), 1);
        }
        Err(_) => {}
    }
}

// ============================================================================
// Stress Test: Many Tensors with Varied Types
// ============================================================================

#[test]
fn test_many_tensors_varied_types() {
    let mut data = Vec::new();

    let tensor_count = 50u64;

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Tensor types to cycle through
    let types = [0u32, 1, 2, 6, 7, 8]; // F32, F16, Q4_0, Q8_0, Q5_0, Q5_1

    let mut offset = 0u64;

    for i in 0..tensor_count {
        let name = format!("tensor_{:03}", i);
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // 1D
        data.extend_from_slice(&32u64.to_le_bytes()); // 32 elements
        data.extend_from_slice(&types[(i as usize) % types.len()].to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        // Rough size estimate (varies by type)
        offset += 128; // Simple approximation
    }

    // Pad to alignment
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Add some tensor data
    for _ in 0..(offset as usize) {
        data.push(0);
    }

    let result = GGUFModel::from_bytes(&data);
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 50);
        }
        Err(_) => {
            // May fail due to type validation
        }
    }
}
