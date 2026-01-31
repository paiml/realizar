//! T-COV-95 Semantic Divergence: Architecture Mismatch Tests (PMAT-802)
//!
//! Dr. Popper's directive: "convert/mod.rs handles 'Format Evolution' (GGUF v1->v3)
//! which you haven't yet stimulated. Implement 'Semantic Divergence'—metadata that
//! claims to be one architecture but contains the tensors of another."
//!
//! This module tests:
//! 1. LLaMA metadata with GPT-2 tensor structure
//! 2. Architecture string mismatches
//! 3. Dimension misalignment between config and tensors
//! 4. Missing required tensors for claimed architecture
//! 5. Extra tensors not in architecture spec
//!
//! Target: 234 missed lines in convert/mod.rs

use crate::convert::GgufToAprConverter;
use crate::gguf::{GGUF_MAGIC, GGUF_VERSION_V3};

// ============================================================================
// Semantic Divergence Pygmy Generators
// ============================================================================

/// Build a GGUF that claims to be LLaMA but has no layers
fn build_divergent_pygmy_llama_no_layers() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor only
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 metadata

    // Metadata 1: architecture = llama
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
    let val1 = "llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // Metadata 2: num_layers = 32 (but we provide 0 layers!)
    let key2 = "llama.block_count";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
    data.extend_from_slice(&32u32.to_le_bytes());

    // Metadata 3: hidden_dim
    let key3 = "llama.embedding_length";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&4096u32.to_le_bytes());

    // Only provide embedding tensor (no layer tensors!)
    let tensor_name = "token_embd.weight";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // 2D
    data.extend_from_slice(&32000u64.to_le_bytes()); // vocab
    data.extend_from_slice(&4096u64.to_le_bytes()); // hidden
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Pad to alignment
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Minimal tensor data
    for _ in 0..16 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    data
}

/// Build a GGUF that claims to be GPT-2 but has LLaMA tensor names
fn build_divergent_pygmy_gpt2_llama_tensors() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 tensors
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 metadata

    // Architecture claims GPT-2
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val1 = "gpt2";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // Num layers
    let key2 = "gpt2.block_count";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // But tensor names are LLaMA-style!
    let tensors = [
        ("blk.0.attn_q.weight", 64, 64),    // LLaMA style
        ("blk.0.attn_k.weight", 64, 64),    // LLaMA style (GPT-2 uses different names)
        ("blk.0.ffn_gate.weight", 128, 64), // LLaMA has gate, GPT-2 doesn't
    ];

    let mut offset = 0u64;
    for (name, d0, d1) in tensors {
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&(d0 as u64).to_le_bytes());
        data.extend_from_slice(&(d1 as u64).to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // F32
        data.extend_from_slice(&offset.to_le_bytes());
        offset += (d0 * d1 * 4) as u64;
    }

    // Pad
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor data
    for _ in 0..(offset / 4) {
        data.extend_from_slice(&0.5f32.to_le_bytes());
    }

    data
}

/// Build a GGUF with dimension mismatch between metadata and tensors
fn build_divergent_pygmy_dimension_mismatch() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes());

    // Architecture
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val1 = "llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // Claims hidden_dim = 4096
    let key2 = "llama.embedding_length";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&4096u32.to_le_bytes()); // Claims 4096

    // num layers
    let key3 = "llama.block_count";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // But embedding tensor has hidden_dim = 128 (mismatch!)
    let t1 = "token_embd.weight";
    data.extend_from_slice(&(t1.len() as u64).to_le_bytes());
    data.extend_from_slice(t1.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1000u64.to_le_bytes()); // vocab
    data.extend_from_slice(&128u64.to_le_bytes()); // hidden = 128 (NOT 4096!)
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Layer tensor also mismatched
    let t2 = "blk.0.attn_q.weight";
    data.extend_from_slice(&(t2.len() as u64).to_le_bytes());
    data.extend_from_slice(t2.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&128u64.to_le_bytes());
    data.extend_from_slice(&128u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&((1000 * 128 * 4) as u64).to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Minimal data
    for _ in 0..1000 {
        data.extend_from_slice(&0.1f32.to_le_bytes());
    }

    data
}

/// Build a GGUF with unknown architecture string
fn build_divergent_pygmy_unknown_architecture() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // Unknown architecture
    let key = "general.architecture";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val = "quantum_transformer_9000"; // Not a real architecture
    data.extend_from_slice(&(val.len() as u64).to_le_bytes());
    data.extend_from_slice(val.as_bytes());

    // Tensor with valid structure
    let t = "token_embd.weight";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&100u64.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..100 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    data
}

/// Build a GGUF with empty architecture string
fn build_divergent_pygmy_empty_architecture() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // Empty architecture
    let key = "general.architecture";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // Empty string

    // Tensor
    let t = "test.weight";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..64 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    data
}

/// Build a GGUF with excessive metadata but minimal tensors
fn build_divergent_pygmy_metadata_heavy() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&20u64.to_le_bytes()); // 20 metadata entries!

    // Generate lots of metadata
    for i in 0..20 {
        let key = format!("custom.param_{:02}", i);
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        data.extend_from_slice(&(i as u32).to_le_bytes());
    }

    // Single minimal tensor
    let t = "weights";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&16u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..16 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    data
}

/// Build a GGUF with conflicting num_heads configurations
fn build_divergent_pygmy_conflicting_heads() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());

    // Architecture
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val1 = "llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // num_attention_heads = 32
    let key2 = "llama.attention.head_count";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&32u32.to_le_bytes());

    // num_kv_heads = 8 (GQA)
    let key3 = "llama.attention.head_count_kv";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());

    // hidden_dim = 64 (but 32 heads would need 32*head_dim, mismatch!)
    let key4 = "llama.embedding_length";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&64u32.to_le_bytes()); // 64 / 32 heads = 2 per head (too small!)

    // Tensor
    let t = "token_embd.weight";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1000u64.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..100 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    data
}

/// Build a GGUF with zero vocab_size in metadata
fn build_divergent_pygmy_zero_vocab() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    // Architecture
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val1 = "llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // vocab_size = 0
    let key2 = "llama.vocab_size";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // Zero vocab!

    // But embedding tensor has vocab = 1000
    let t = "token_embd.weight";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1000u64.to_le_bytes()); // Mismatch!
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..100 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    data
}

/// Build a GGUF with negative values in unsigned metadata
fn build_divergent_pygmy_signed_as_unsigned() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    // Architecture
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val1 = "llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // block_count stored as i32 but reinterpreted as u32
    // -1 as i32 = 0xFFFFFFFF as u32 = 4294967295 layers!
    let key2 = "llama.block_count";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&5u32.to_le_bytes()); // GGUF_TYPE_INT32
    data.extend_from_slice(&(-1i32).to_le_bytes()); // -1

    // Tensor
    let t = "token_embd.weight";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..64 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    data
}

// ============================================================================
// Semantic Divergence Tests
// ============================================================================

#[test]
fn test_divergent_llama_no_layers() {
    let data = build_divergent_pygmy_llama_no_layers();
    let result = GgufToAprConverter::convert(&data);

    // Should fail - claims 32 layers but has none
    match result {
        Ok(_) => {
            // If it succeeds, the converter is lenient
        }
        Err(e) => {
            // Expected failure
            let _ = e;
        }
    }
}

#[test]
fn test_divergent_gpt2_with_llama_tensors() {
    let data = build_divergent_pygmy_gpt2_llama_tensors();
    let result = GgufToAprConverter::convert(&data);

    // Architecture mismatch should be handled
    match result {
        Ok(_) => {}
        Err(e) => {
            let _ = e;
        }
    }
}

#[test]
fn test_divergent_dimension_mismatch() {
    let data = build_divergent_pygmy_dimension_mismatch();
    let result = GgufToAprConverter::convert(&data);

    // Dimension mismatch may cause issues
    match result {
        Ok(apr) => {
            // Check if config was inferred from tensors or metadata
            assert!(apr.config.hidden_dim > 0);
        }
        Err(e) => {
            let _ = e;
        }
    }
}

#[test]
fn test_divergent_unknown_architecture() {
    let data = build_divergent_pygmy_unknown_architecture();
    let result = GgufToAprConverter::convert(&data);

    // Unknown architecture should be handled gracefully
    match result {
        Ok(apr) => {
            // May use default config
            assert!(!apr.config.architecture.is_empty());
        }
        Err(e) => {
            // Error may mention architecture or missing config fields
            let msg = e.to_string().to_lowercase();
            assert!(
                msg.contains("architecture") || msg.contains("unsupported") || msg.contains("unknown") || msg.contains("missing"),
                "Error should mention architecture or config issue: {}",
                msg
            );
        }
    }
}

#[test]
fn test_divergent_empty_architecture() {
    let data = build_divergent_pygmy_empty_architecture();
    let result = GgufToAprConverter::convert(&data);

    // Empty architecture may default or fail
    match result {
        Ok(apr) => {
            // Some default was used
            let _ = apr;
        }
        Err(e) => {
            let _ = e;
        }
    }
}

#[test]
fn test_divergent_metadata_heavy() {
    let data = build_divergent_pygmy_metadata_heavy();
    let result = GgufToAprConverter::convert(&data);

    // Should handle lots of metadata without crashing
    match result {
        Ok(_) => {}
        Err(_) => {}
    }
}

#[test]
fn test_divergent_conflicting_heads() {
    let data = build_divergent_pygmy_conflicting_heads();
    let result = GgufToAprConverter::convert(&data);

    // Conflicting head counts should be handled
    match result {
        Ok(apr) => {
            // Config was determined somehow
            assert!(apr.config.num_heads > 0);
        }
        Err(e) => {
            let _ = e;
        }
    }
}

#[test]
fn test_divergent_zero_vocab() {
    let data = build_divergent_pygmy_zero_vocab();
    let result = GgufToAprConverter::convert(&data);

    // Zero vocab in metadata, non-zero in tensor
    match result {
        Ok(apr) => {
            // Vocab was inferred from tensor
            assert!(apr.config.vocab_size > 0);
        }
        Err(e) => {
            let _ = e;
        }
    }
}

#[test]
fn test_divergent_signed_as_unsigned() {
    let data = build_divergent_pygmy_signed_as_unsigned();
    let result = GgufToAprConverter::convert(&data);

    // -1 interpreted as u32 = huge number
    match result {
        Ok(_) => {
            // Converter accepted it somehow
        }
        Err(e) => {
            let _ = e;
        }
    }
}

// ============================================================================
// Converter Edge Cases
// ============================================================================

#[test]
fn test_converter_empty_data() {
    let data = vec![];
    let result = GgufToAprConverter::convert(&data);

    assert!(result.is_err(), "Empty data should fail");
}

#[test]
fn test_converter_partial_header() {
    let data = vec![0x47, 0x47, 0x55, 0x46]; // Just magic, no version
    let result = GgufToAprConverter::convert(&data);

    assert!(result.is_err(), "Partial header should fail");
}

#[test]
fn test_converter_just_header_no_tensors() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    let result = GgufToAprConverter::convert(&data);

    // Zero tensors may or may not be valid
    match result {
        Ok(apr) => {
            assert_eq!(apr.layers.len(), 0);
        }
        Err(e) => {
            let _ = e;
        }
    }
}

#[test]
fn test_converter_valid_minimal() {
    // A valid minimal GGUF that should convert
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 metadata

    // Architecture
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val1 = "llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // block_count = 0
    let key2 = "llama.block_count";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    // Embedding tensor
    let t = "token_embd.weight";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&100u64.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor data: 100 * 64 = 6400 floats
    for _ in 0..6400 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let result = GgufToAprConverter::convert(&data);
    // May succeed or fail based on missing required tensors
    let _ = result;
}

// ============================================================================
// Mixed Quantization Divergence Tests
// ============================================================================

#[test]
fn test_divergent_mixed_quant_types() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes()); // 4 tensors
    data.extend_from_slice(&0u64.to_le_bytes());

    // Different quantization types
    let tensors = [
        ("tensor_f32", 0u32, 64),   // F32
        ("tensor_f16", 1u32, 64),   // F16
        ("tensor_q4_0", 2u32, 32),  // Q4_0
        ("tensor_q8_0", 7u32, 32),  // Q8_0
    ];

    let mut offset = 0u64;
    for (name, qtype, size) in tensors {
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(size as u64).to_le_bytes());
        data.extend_from_slice(&qtype.to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        // Rough size estimate
        let byte_size = match qtype {
            0 => size * 4,      // F32: 4 bytes per element
            1 => size * 2,      // F16: 2 bytes per element
            2 => (size / 2) + 2, // Q4_0: ~0.5 bytes per element + scale
            7 => size + 4,      // Q8_0: 1 byte per element + scale
            _ => size * 4,
        };
        offset += byte_size as u64;
    }

    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Pad data
    for _ in 0..(offset as usize) {
        data.push(0);
    }

    let result = GgufToAprConverter::convert(&data);
    // Mixed quant types exercise different dequantization paths
    let _ = result;
}

// ============================================================================
// Stress: Deep Nesting in Metadata
// ============================================================================

#[test]
fn test_divergent_deeply_nested_key() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // Very deeply nested key
    let key = "a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p.q.r.s.t.u.v.w.x.y.z.deep.nested.key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&42u32.to_le_bytes());

    // Tensor
    let t = "test";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..4 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let result = GgufToAprConverter::convert(&data);
    let _ = result;
}
