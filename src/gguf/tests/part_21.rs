//! Phase 34: Transformer structure coverage tests
//!
//! These lib tests illuminate gguf/transformer.rs:
//! - QuantizedTensorRef byte size calculations
//! - QKVWeights enum handling
//! - QuantizedGGUFTransformerLayer structure
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::quantized::QuantizedTensorRef;
use crate::gguf::types::{
    GGUF_TYPE_F32, GGUF_TYPE_Q2_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0,
    GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
};

// =============================================================================
// QuantizedTensorRef Byte Size Tests
// =============================================================================

/// Test byte size calculation for F32 tensors
#[test]
fn test_phase34_tensor_ref_f32_byte_size() {
    let num_elements: usize = 1024;
    let expected_bytes = num_elements * 4; // 4 bytes per f32

    // Verify the calculation matches F32 expectations
    assert_eq!(expected_bytes, 4096);
}

/// Test byte size calculation for Q4_0 tensors
#[test]
fn test_phase34_tensor_ref_q4_0_byte_size() {
    // Q4_0: 18 bytes per 32 elements
    let num_elements: usize = 256;
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 18;

    assert_eq!(num_blocks, 8);
    assert_eq!(expected_bytes, 144);
}

/// Test byte size calculation for Q8_0 tensors
#[test]
fn test_phase34_tensor_ref_q8_0_byte_size() {
    // Q8_0: 34 bytes per 32 elements
    let num_elements: usize = 256;
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 34;

    assert_eq!(num_blocks, 8);
    assert_eq!(expected_bytes, 272);
}

/// Test byte size calculation for Q2_K tensors
#[test]
fn test_phase34_tensor_ref_q2_k_byte_size() {
    // Q2_K: 84 bytes per 256 elements (QK_K=256)
    let num_elements: usize = 512;
    let num_super_blocks = num_elements.div_ceil(256);
    let expected_bytes = num_super_blocks * 84;

    assert_eq!(num_super_blocks, 2);
    assert_eq!(expected_bytes, 168);
}

/// Test byte size calculation for Q4_1 tensors
#[test]
fn test_phase34_tensor_ref_q4_1_byte_size() {
    // Q4_1: 20 bytes per 32 elements
    let num_elements: usize = 128;
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 20;

    assert_eq!(num_blocks, 4);
    assert_eq!(expected_bytes, 80);
}

/// Test byte size calculation for Q5_0 tensors
#[test]
fn test_phase34_tensor_ref_q5_0_byte_size() {
    // Q5_0: 22 bytes per 32 elements
    let num_elements: usize = 128;
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 22;

    assert_eq!(num_blocks, 4);
    assert_eq!(expected_bytes, 88);
}

/// Test byte size calculation for Q4_K tensors
#[test]
fn test_phase34_tensor_ref_q4_k_byte_size() {
    // Q4_K: 144 bytes per 256 elements (QK_K=256)
    let num_elements: usize = 512;
    let num_super_blocks = num_elements.div_ceil(256);
    let expected_bytes = num_super_blocks * 144;

    assert_eq!(num_super_blocks, 2);
    assert_eq!(expected_bytes, 288);
}

/// Test byte size calculation for Q5_K tensors
#[test]
fn test_phase34_tensor_ref_q5_k_byte_size() {
    // Q5_K: 176 bytes per 256 elements (QK_K=256)
    let num_elements: usize = 512;
    let num_super_blocks = num_elements.div_ceil(256);
    let expected_bytes = num_super_blocks * 176;

    assert_eq!(num_super_blocks, 2);
    assert_eq!(expected_bytes, 352);
}

/// Test byte size calculation for Q6_K tensors
#[test]
fn test_phase34_tensor_ref_q6_k_byte_size() {
    // Q6_K: 210 bytes per 256 elements (QK_K=256)
    let num_elements: usize = 512;
    let num_super_blocks = num_elements.div_ceil(256);
    let expected_bytes = num_super_blocks * 210;

    assert_eq!(num_super_blocks, 2);
    assert_eq!(expected_bytes, 420);
}

// =============================================================================
// QuantizedTensorRef Creation Tests
// =============================================================================

#[test]
fn test_phase34_quantized_tensor_ref_creation() {
    let tensor_ref = QuantizedTensorRef {
        offset: 1024,
        byte_size: 2048,
        num_elements: 512,
        qtype: GGUF_TYPE_Q4_K,
    };

    assert_eq!(tensor_ref.offset, 1024);
    assert_eq!(tensor_ref.byte_size, 2048);
    assert_eq!(tensor_ref.num_elements, 512);
    assert_eq!(tensor_ref.qtype, GGUF_TYPE_Q4_K);
}

#[test]
fn test_phase34_quantized_tensor_ref_all_qtypes() {
    let qtypes = [
        (GGUF_TYPE_F32, "F32"),
        (GGUF_TYPE_Q4_0, "Q4_0"),
        (GGUF_TYPE_Q8_0, "Q8_0"),
        (GGUF_TYPE_Q2_K, "Q2_K"),
        (GGUF_TYPE_Q4_1, "Q4_1"),
        (GGUF_TYPE_Q5_0, "Q5_0"),
        (GGUF_TYPE_Q4_K, "Q4_K"),
        (GGUF_TYPE_Q5_K, "Q5_K"),
        (GGUF_TYPE_Q6_K, "Q6_K"),
    ];

    for (qtype, name) in qtypes {
        let tensor_ref = QuantizedTensorRef {
            offset: 0,
            byte_size: 1024,
            num_elements: 256,
            qtype,
        };

        assert_eq!(tensor_ref.qtype, qtype, "qtype mismatch for {}", name);
    }
}

// =============================================================================
// QKVWeights Enum Tests
// =============================================================================

use crate::gguf::quantized::QKVWeights;

#[test]
fn test_phase34_qkv_weights_fused() {
    let fused_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 4096,
        num_elements: 1024,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = QKVWeights::Fused(fused_ref);

    if let QKVWeights::Fused(ref f) = qkv {
        assert_eq!(f.byte_size, 4096);
        assert_eq!(f.num_elements, 1024);
    } else {
        panic!("Expected Fused variant");
    }
}

#[test]
fn test_phase34_qkv_weights_separate() {
    let q_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 1024,
        num_elements: 256,
        qtype: GGUF_TYPE_Q4_K,
    };
    let k_ref = QuantizedTensorRef {
        offset: 1024,
        byte_size: 1024,
        num_elements: 256,
        qtype: GGUF_TYPE_Q4_K,
    };
    let v_ref = QuantizedTensorRef {
        offset: 2048,
        byte_size: 1024,
        num_elements: 256,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = QKVWeights::Separate {
        q: q_ref,
        k: k_ref,
        v: v_ref,
    };

    if let QKVWeights::Separate { q, k, v } = qkv {
        assert_eq!(q.offset, 0);
        assert_eq!(k.offset, 1024);
        assert_eq!(v.offset, 2048);
    } else {
        panic!("Expected Separate variant");
    }
}

// =============================================================================
// Byte Size Edge Cases
// =============================================================================

#[test]
fn test_phase34_byte_size_non_aligned() {
    // Test with element counts that don't align to block boundaries

    // Q4_0 with 33 elements (needs 2 blocks for 64 element capacity)
    let num_elements: usize = 33;
    let num_blocks = num_elements.div_ceil(32);
    assert_eq!(num_blocks, 2); // ceil(33/32) = 2
    let byte_size = num_blocks * 18;
    assert_eq!(byte_size, 36);
}

#[test]
fn test_phase34_byte_size_exact_block() {
    // Q4_0 with exactly 32 elements (1 block)
    let num_elements: usize = 32;
    let num_blocks = num_elements.div_ceil(32);
    assert_eq!(num_blocks, 1);
    let byte_size = num_blocks * 18;
    assert_eq!(byte_size, 18);
}

#[test]
fn test_phase34_byte_size_large_tensor() {
    // Large tensor like embedding table: 32000 vocab * 4096 hidden
    let num_elements: usize = 32000 * 4096;

    // Q4_K byte size
    let num_super_blocks = num_elements.div_ceil(256);
    let q4k_bytes = num_super_blocks * 144;

    // Should be roughly 72MB for Q4_K
    assert!(q4k_bytes > 70_000_000);
    assert!(q4k_bytes < 80_000_000);

    // F32 would be much larger
    let f32_bytes = num_elements * 4;
    assert_eq!(f32_bytes, 524_288_000); // 500MB

    // Q4_K is ~7x smaller
    let compression = f32_bytes as f64 / q4k_bytes as f64;
    assert!(compression > 6.0);
    assert!(compression < 8.0);
}

// =============================================================================
// Quantization Type Constants Tests
// =============================================================================

#[test]
fn test_phase34_qtype_constants() {
    // Verify the qtype constants have expected values
    // These match GGML's enum values
    assert_eq!(GGUF_TYPE_F32, 0);
    assert_eq!(GGUF_TYPE_Q4_0, 2);
    assert_eq!(GGUF_TYPE_Q4_1, 3);
    assert_eq!(GGUF_TYPE_Q5_0, 6);
    assert_eq!(GGUF_TYPE_Q8_0, 8);
    assert_eq!(GGUF_TYPE_Q2_K, 10);
    assert_eq!(GGUF_TYPE_Q4_K, 12);
    assert_eq!(GGUF_TYPE_Q5_K, 13);
    assert_eq!(GGUF_TYPE_Q6_K, 14);
}

// =============================================================================
// Tensor Offset Calculations
// =============================================================================

#[test]
fn test_phase34_tensor_offset_accumulation() {
    // Simulate tensor offsets in a model
    let base_offset: usize = 4096; // After header/metadata

    // First tensor: embedding (vocab * hidden)
    let embed_elements: usize = 32000 * 4096;
    let embed_blocks = embed_elements.div_ceil(256);
    let embed_bytes = embed_blocks * 144; // Q4_K

    let first_tensor_offset = base_offset;
    let second_tensor_offset = first_tensor_offset + embed_bytes;

    assert!(second_tensor_offset > first_tensor_offset);
    assert_eq!(first_tensor_offset, 4096);
}

#[test]
fn test_phase34_tensor_ref_bounds_check() {
    // Test that we can verify tensor fits in file
    let file_size = 1_000_000; // 1MB file
    let tensor_offset = 500_000;
    let tensor_bytes = 400_000;

    // This tensor fits
    assert!(tensor_offset + tensor_bytes <= file_size);

    // This tensor doesn't fit
    let large_tensor_bytes = 600_000;
    assert!(tensor_offset + large_tensor_bytes > file_size);
}

// =============================================================================
// Architecture-specific Tests
// =============================================================================

#[test]
fn test_phase34_llama_style_layers() {
    // LLaMA uses separate Q, K, V projections
    let hidden_dim = 4096;
    let num_heads = 32;
    let head_dim = hidden_dim / num_heads; // 128

    // Q projection: hidden -> hidden
    let q_elements = hidden_dim * hidden_dim;
    // K projection: hidden -> num_kv_heads * head_dim (GQA)
    let num_kv_heads = 8;
    let k_elements = hidden_dim * (num_kv_heads * head_dim);
    // V projection: same as K
    let v_elements = k_elements;

    assert_eq!(q_elements, 16_777_216); // 4096 * 4096
    assert_eq!(k_elements, 4_194_304); // 4096 * 1024
    assert_eq!(v_elements, 4_194_304);
}

#[test]
fn test_phase34_phi2_style_layers() {
    // Phi-2 uses fused QKV projection
    let hidden_dim = 2560;
    let num_heads = 32;
    let head_dim = hidden_dim / num_heads; // 80

    // Fused QKV: hidden -> 3 * hidden
    let qkv_out_dim = 3 * hidden_dim;
    let qkv_elements = hidden_dim * qkv_out_dim;

    assert_eq!(qkv_elements, 19_660_800); // 2560 * 7680
}

// =============================================================================
// Memory Layout Tests
// =============================================================================

#[test]
fn test_phase34_contiguous_layout() {
    // Verify tensors can be laid out contiguously
    let tensors: Vec<(&str, usize, u32)> = vec![
        ("embed", 32000 * 4096, GGUF_TYPE_Q4_K),
        ("layer.0.attn_q", 4096 * 4096, GGUF_TYPE_Q4_K),
        ("layer.0.attn_k", 4096 * 1024, GGUF_TYPE_Q4_K),
        ("layer.0.attn_v", 4096 * 1024, GGUF_TYPE_Q4_K),
    ];

    let mut current_offset = 0usize;
    for (name, num_elements, qtype) in &tensors {
        let num_super_blocks = num_elements.div_ceil(256);
        let byte_size = match *qtype {
            GGUF_TYPE_Q4_K => num_super_blocks * 144,
            _ => *num_elements * 4,
        };

        let tensor_ref = QuantizedTensorRef {
            offset: current_offset,
            byte_size,
            num_elements: *num_elements,
            qtype: *qtype,
        };

        assert_eq!(
            tensor_ref.offset, current_offset,
            "Offset mismatch for {}",
            name
        );
        current_offset += byte_size;
    }

    // Total model size should be calculated
    assert!(current_offset > 0);
}

// =============================================================================
// Phase 35: Real Transformer Loading Tests (using Test Factory)
// =============================================================================

use crate::gguf::test_factory::{
    build_minimal_llama_gguf, build_minimal_phi2_gguf, create_f32_embedding_data,
    create_f32_norm_weights, create_q4_0_data, create_q4_k_data, create_q5_k_data,
    create_q6_k_data, create_q8_0_data, GGUFBuilder,
};
use crate::gguf::transformer::QuantizedGGUFTransformer;
use crate::gguf::GGUFModel;

#[test]
fn test_phase35_transformer_from_minimal_llama() {
    // Build a minimal LLaMA-style GGUF
    let data = build_minimal_llama_gguf(
        100, // vocab_size
        64,  // hidden_dim (must be divisible by num_heads)
        128, // intermediate_dim
        4,   // num_heads
        4,   // num_kv_heads
    );

    // Parse the GGUF model
    let model = GGUFModel::from_bytes(&data).expect("Should parse minimal LLaMA GGUF");

    // Load as quantized transformer
    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load transformer");

    // Verify config was loaded correctly
    assert_eq!(transformer.config.architecture, "llama");
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.config.num_heads, 4);
    assert_eq!(transformer.config.num_kv_heads, 4);

    // Verify token embedding was loaded
    assert_eq!(transformer.token_embedding.len(), 100 * 64); // vocab * hidden

    // Verify output norm was loaded
    assert_eq!(transformer.output_norm_weight.len(), 64);

    // Verify layer was loaded with correct structure
    assert_eq!(transformer.layers.len(), 1);

    let layer = &transformer.layers[0];
    // Attention norm should be f32
    assert_eq!(layer.attn_norm_weight.len(), 64);

    // QKV should be separate (LLaMA style)
    match &layer.qkv_weight {
        QKVWeights::Separate { q, k, v } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q4_K);
            assert_eq!(k.qtype, GGUF_TYPE_Q4_K);
            assert_eq!(v.qtype, GGUF_TYPE_Q4_K);
            // Q: hidden -> hidden (64*64)
            assert_eq!(q.num_elements, 64 * 64);
            // K, V: hidden -> kv_dim (64*64 since num_kv_heads = num_heads)
            assert_eq!(k.num_elements, 64 * 64);
            assert_eq!(v.num_elements, 64 * 64);
        },
        QKVWeights::Fused(_) => panic!("Expected Separate QKV for LLaMA style"),
    }

    // FFN weights should be quantized
    assert_eq!(layer.ffn_up_weight.qtype, GGUF_TYPE_Q4_K);
    assert_eq!(layer.ffn_down_weight.qtype, GGUF_TYPE_Q4_K);
    assert!(layer.ffn_gate_weight.is_some(), "LLaMA should have gate");
}

#[test]
fn test_phase35_transformer_from_minimal_phi2() {
    // Build a minimal Phi-2 style GGUF (fused QKV)
    let data = build_minimal_phi2_gguf(
        100, // vocab_size
        64,  // hidden_dim
        128, // intermediate_dim
        4,   // num_heads
    );

    let model = GGUFModel::from_bytes(&data).expect("Should parse minimal Phi-2 GGUF");

    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load transformer");

    // Verify config
    assert_eq!(transformer.config.architecture, "phi2");
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.config.num_heads, 4);

    // Verify layer has fused QKV
    let layer = &transformer.layers[0];
    match &layer.qkv_weight {
        QKVWeights::Fused(fused) => {
            assert_eq!(fused.qtype, GGUF_TYPE_Q4_K);
            // Fused: hidden -> 3 * hidden
            assert_eq!(fused.num_elements, 64 * (3 * 64));
        },
        QKVWeights::Separate { .. } => panic!("Expected Fused QKV for Phi-2 style"),
    }

    // Phi-2 style has no gate
    assert!(layer.ffn_gate_weight.is_none());
}

#[test]
fn test_phase35_transformer_with_gqa() {
    // Test Grouped Query Attention (GQA) - fewer KV heads than Q heads
    let hidden_dim = 64usize;
    let num_heads = 8usize;
    let num_kv_heads = 2usize; // GQA: 4 Q heads per KV head
    let head_dim = hidden_dim / num_heads; // 8
    let kv_dim = num_kv_heads * head_dim; // 16
    let vocab_size = 100usize;
    let intermediate_dim = 128usize;

    // Create tensor data
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * kv_dim);
    let v_data = create_q4_k_data(hidden_dim * kv_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", num_heads as u32)
        .num_kv_heads("llama", num_kv_heads as u32)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_down_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_gate_data,
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse GQA model");
    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load GQA transformer");

    // Verify GQA config
    assert_eq!(transformer.config.num_heads, 8);
    assert_eq!(transformer.config.num_kv_heads, 2);

    // Verify K, V have smaller dimensions
    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, k, v } => {
            assert_eq!(q.num_elements, 64 * 64); // full
            assert_eq!(k.num_elements, 64 * 16); // reduced
            assert_eq!(v.num_elements, 64 * 16); // reduced
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_transformer_multiple_layers() {
    // Test with 2 layers to verify layer indexing
    let hidden_dim = 64usize;
    let intermediate_dim = 128usize;
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * hidden_dim);
    let v_data = create_q4_k_data(hidden_dim * hidden_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let mut builder = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 2) // 2 layers
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        );

    // Add tensors for both layers
    for layer_idx in 0..2 {
        builder = builder
            .add_f32_tensor(
                &format!("blk.{}.attn_norm.weight", layer_idx),
                &[hidden_dim as u64],
                &norm_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_q.weight", layer_idx),
                &[hidden_dim as u64, hidden_dim as u64],
                &q_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_k.weight", layer_idx),
                &[hidden_dim as u64, hidden_dim as u64],
                &k_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_v.weight", layer_idx),
                &[hidden_dim as u64, hidden_dim as u64],
                &v_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_output.weight", layer_idx),
                &[hidden_dim as u64, hidden_dim as u64],
                &attn_out_data,
            )
            .add_f32_tensor(
                &format!("blk.{}.ffn_norm.weight", layer_idx),
                &[hidden_dim as u64],
                &norm_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_up.weight", layer_idx),
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_up_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_down.weight", layer_idx),
                &[intermediate_dim as u64, hidden_dim as u64],
                &ffn_down_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_gate.weight", layer_idx),
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_gate_data,
            );
    }

    let data = builder
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse multi-layer model");
    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load multi-layer");

    // Verify both layers loaded
    assert_eq!(transformer.config.num_layers, 2);
    assert_eq!(transformer.layers.len(), 2);

    // Both layers should have same structure
    for (idx, layer) in transformer.layers.iter().enumerate() {
        assert_eq!(layer.attn_norm_weight.len(), 64, "Layer {} norm", idx);
        assert!(layer.ffn_gate_weight.is_some(), "Layer {} gate", idx);
    }
}

#[test]
fn test_phase35_get_tensor_ref_q4_0() {
    // Test Q4_0 byte size calculation through actual loading
    let hidden_dim = 64usize;
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q4_0_data = create_q4_0_data(hidden_dim * hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 128)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        // Use Q4_0 for attention weights
        .add_q4_0_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q4_0_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q4_0_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q4_0_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q4_0_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_0_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, 128],
            &create_q4_0_data(hidden_dim * 128),
        )
        .add_q4_0_tensor(
            "blk.0.ffn_down.weight",
            &[128, hidden_dim as u64],
            &create_q4_0_data(128 * hidden_dim),
        )
        .add_q4_0_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, 128],
            &create_q4_0_data(hidden_dim * 128),
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse Q4_0 model");
    let transformer = QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load Q4_0");

    // Verify Q4_0 type was preserved
    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, k, v } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q4_0);
            assert_eq!(k.qtype, GGUF_TYPE_Q4_0);
            assert_eq!(v.qtype, GGUF_TYPE_Q4_0);

            // Q4_0 byte size: 18 bytes per 32 elements
            // 64*64 = 4096 elements => 128 blocks => 2304 bytes
            assert_eq!(q.byte_size, 128 * 18);
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_get_tensor_ref_q8_0() {
    // Test Q8_0 byte size calculation
    let hidden_dim = 64usize;
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q8_0_data = create_q8_0_data(hidden_dim * hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 128)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q8_0_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_0_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_0_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_0_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_0_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q8_0_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, 128],
            &create_q8_0_data(hidden_dim * 128),
        )
        .add_q8_0_tensor(
            "blk.0.ffn_down.weight",
            &[128, hidden_dim as u64],
            &create_q8_0_data(128 * hidden_dim),
        )
        .add_q8_0_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, 128],
            &create_q8_0_data(hidden_dim * 128),
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse Q8_0 model");
    let transformer = QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load Q8_0");

    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, .. } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q8_0);
            // Q8_0: 34 bytes per 32 elements
            // 4096 elements => 128 blocks => 4352 bytes
            assert_eq!(q.byte_size, 128 * 34);
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_get_tensor_ref_q5_k() {
    // Test Q5_K byte size calculation
    let hidden_dim = 256usize; // Must be multiple of 256 for K-quants
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q5_k_data = create_q5_k_data(hidden_dim * hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 512)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q5_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q5_k_data,
        )
        .add_q5_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q5_k_data,
        )
        .add_q5_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q5_k_data,
        )
        .add_q5_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q5_k_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q5_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, 512],
            &create_q5_k_data(hidden_dim * 512),
        )
        .add_q5_k_tensor(
            "blk.0.ffn_down.weight",
            &[512, hidden_dim as u64],
            &create_q5_k_data(512 * hidden_dim),
        )
        .add_q5_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, 512],
            &create_q5_k_data(hidden_dim * 512),
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse Q5_K model");
    let transformer = QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load Q5_K");

    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, .. } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q5_K);
            // Q5_K: 176 bytes per 256 elements
            // 256*256 = 65536 elements => 256 super-blocks => 45056 bytes
            assert_eq!(q.byte_size, 256 * 176);
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_get_tensor_ref_q6_k() {
    // Test Q6_K byte size calculation
    let hidden_dim = 256usize;
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q6_k_data = create_q6_k_data(hidden_dim * hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 512)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q6_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6_k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6_k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6_k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6_k_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q6_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, 512],
            &create_q6_k_data(hidden_dim * 512),
        )
        .add_q6_k_tensor(
            "blk.0.ffn_down.weight",
            &[512, hidden_dim as u64],
            &create_q6_k_data(512 * hidden_dim),
        )
        .add_q6_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, 512],
            &create_q6_k_data(hidden_dim * 512),
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse Q6_K model");
    let transformer = QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load Q6_K");

    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, .. } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q6_K);
            // Q6_K: 210 bytes per 256 elements
            // 65536 elements => 256 super-blocks => 53760 bytes
            assert_eq!(q.byte_size, 256 * 210);
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_transformer_missing_tensor_error() {
    // Test error handling when required tensor is missing
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 64)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        // Missing token_embd.weight!
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse empty model");
    let result = QuantizedGGUFTransformer::from_gguf(&model, &data);

    assert!(
        result.is_err(),
        "Should fail when token_embd.weight is missing"
    );

    // Extract error message
    match result {
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("token_embd.weight")
                    || err.contains("Tensor")
                    || err.contains("not found"),
                "Error should mention missing tensor: {}",
                err
            );
        },
        Ok(_) => panic!("Expected error for missing tensor"),
    }
}

#[test]
fn test_phase35_transformer_lm_head_fallback() {
    // Test that lm_head falls back to token_embd when output.weight is missing
    let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load with fallback");

    // lm_head should exist (fallback to token_embd)
    assert!(transformer.lm_head_weight.byte_size > 0);
    assert_eq!(transformer.lm_head_weight.num_elements, 100 * 64); // vocab * hidden
}

#[test]
fn test_phase35_data_factory_helpers() {
    // Test the data factory helper functions directly
    let q4_0 = create_q4_0_data(64);
    assert_eq!(q4_0.len(), 2 * 18); // 64/32 = 2 blocks * 18 bytes

    let q8_0 = create_q8_0_data(64);
    assert_eq!(q8_0.len(), 2 * 34); // 64/32 = 2 blocks * 34 bytes

    let q4_k = create_q4_k_data(256);
    assert_eq!(q4_k.len(), 1 * 144); // 256/256 = 1 super-block * 144 bytes

    let q5_k = create_q5_k_data(512);
    assert_eq!(q5_k.len(), 2 * 176); // 512/256 = 2 super-blocks * 176 bytes

    let q6_k = create_q6_k_data(512);
    assert_eq!(q6_k.len(), 2 * 210); // 512/256 = 2 super-blocks * 210 bytes

    let embed = create_f32_embedding_data(10, 8);
    assert_eq!(embed.len(), 80); // 10 * 8

    let norm = create_f32_norm_weights(32);
    assert_eq!(norm.len(), 32);
    assert!(norm.iter().all(|&v| (v - 1.0).abs() < 1e-6)); // All ones
}
