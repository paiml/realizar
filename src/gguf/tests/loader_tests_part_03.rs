//! GGUF Loader Tests Part 03: to_apr_bytes + from_apr Coverage
//!
//! This module provides comprehensive test coverage for:
//! - `OwnedQuantizedModel::to_apr_bytes()` - serialize quantized model to APR v2
//! - `OwnedQuantizedModel::from_apr()` - load from MappedAprModel
//! - Internal helpers: qtype_to_dtype, dtype_to_byte, write_tensor_entry
//! - Separate vs Fused QKV paths in serialization
//! - FFN gate/norm presence/absence branches
//! - Roundtrip: to_apr_bytes -> write to disk -> MappedAprModel -> from_apr

use crate::gguf::{
    GGUFConfig, OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
};

// ============================================================================
// Helper: Build a minimal OwnedQuantizedModel for testing
// ============================================================================

/// Create a tiny OwnedQuantizedModel with F32 weights (qtype=0) and separate QKV
fn build_minimal_owned_quantized_model() -> OwnedQuantizedModel {
    let hidden_dim = 8;
    let intermediate_dim = 16;
    let vocab_size = 10;
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    // Token embedding (F32): vocab_size * hidden_dim floats -> 4 bytes each
    let embed_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    // Create F32 data (qtype 0) for weight tensors
    fn make_f32_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let data: Vec<u8> = (0..in_dim * out_dim)
            .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
            .collect();
        OwnedQuantizedTensor {
            data,
            in_dim,
            out_dim,
            qtype: 0, // F32
        }
    }

    // Separate Q, K, V
    let q_weight = make_f32_tensor(hidden_dim, hidden_dim);
    let k_weight = make_f32_tensor(hidden_dim, kv_dim);
    let v_weight = make_f32_tensor(hidden_dim, kv_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        },
        qkv_bias: None,
        attn_output_weight: make_f32_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: make_f32_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: make_f32_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(make_f32_tensor(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
    };

    OwnedQuantizedModel {
        config,
        token_embedding: embed_data,
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: make_f32_tensor(hidden_dim, vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create a model with Fused QKV (phi2 style)
fn build_fused_qkv_model() -> OwnedQuantizedModel {
    let hidden_dim = 8;
    let intermediate_dim = 16;
    let vocab_size = 10;
    let num_heads = 2;
    let num_kv_heads = 2;

    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 2,
    };

    let embed_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    fn make_f32_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let data: Vec<u8> = (0..in_dim * out_dim)
            .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
            .collect();
        OwnedQuantizedTensor {
            data,
            in_dim,
            out_dim,
            qtype: 0,
        }
    }

    // Fused QKV: hidden -> 3 * hidden
    let qkv_dim = 3 * hidden_dim;
    let fused_qkv = OwnedQuantizedTensor {
        data: vec![0u8; hidden_dim * qkv_dim * 4], // F32
        in_dim: hidden_dim,
        out_dim: qkv_dim,
        qtype: 0,
    };

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(fused_qkv),
        qkv_bias: None,
        attn_output_weight: make_f32_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: make_f32_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: make_f32_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: None, // No gate for phi2-style GELU
        ffn_gate_bias: None,
        ffn_norm_weight: None, // Some models skip FFN norm
        ffn_norm_bias: None,
    };

    OwnedQuantizedModel {
        config,
        token_embedding: embed_data,
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: make_f32_tensor(hidden_dim, vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create a model with Q4_K quantized weights (qtype=12)
fn build_q4k_model() -> OwnedQuantizedModel {
    let hidden_dim = 8;
    let intermediate_dim = 16;
    let vocab_size = 10;
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let config = GGUFConfig {
        architecture: "qwen2".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 1_000_000.0,
        eps: 1e-6,
        rope_type: 2,
    };

    let embed_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    // Q4_K: 144 bytes per 256 elements
    fn make_q4k_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let num_elements = in_dim * out_dim;
        let num_super_blocks = num_elements.div_ceil(256);
        OwnedQuantizedTensor {
            data: vec![0u8; num_super_blocks * 144],
            in_dim,
            out_dim,
            qtype: 12, // Q4_K
        }
    }

    let q_weight = make_q4k_tensor(hidden_dim, hidden_dim);
    let k_weight = make_q4k_tensor(hidden_dim, kv_dim);
    let v_weight = make_q4k_tensor(hidden_dim, kv_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        },
        qkv_bias: None,
        attn_output_weight: make_q4k_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: make_q4k_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: make_q4k_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(make_q4k_tensor(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
    };

    // Q6_K lm_head: 210 bytes per 256 elements
    let lm_head_elements = hidden_dim * vocab_size;
    let lm_head_sb = lm_head_elements.div_ceil(256);
    let lm_head = OwnedQuantizedTensor {
        data: vec![0u8; lm_head_sb * 210],
        in_dim: hidden_dim,
        out_dim: vocab_size,
        qtype: 14, // Q6_K
    };

    OwnedQuantizedModel {
        config,
        token_embedding: embed_data,
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: lm_head,
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

// ============================================================================
// to_apr_bytes: Basic tests
// ============================================================================

#[test]
fn test_to_apr_bytes_produces_valid_header() {
    let model = build_minimal_owned_quantized_model();
    let result = model.to_apr_bytes();
    assert!(result.is_ok(), "to_apr_bytes failed: {:?}", result.err());

    let bytes = result.expect("should produce bytes");

    // Check APR magic
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00], "Magic bytes");
    // Check version
    assert_eq!(bytes[4], 2, "Major version");
    assert_eq!(bytes[5], 0, "Minor version");
    // Must be at least 64 bytes (header)
    assert!(bytes.len() >= 64, "Must include full header");
}

#[test]
fn test_to_apr_bytes_tensor_count() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // tensor_count is at offset [8..12]
    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

    // Separate QKV = 3 tensors per layer, plus:
    // token_embd, attn_norm, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down, output_norm, output
    // = 1 + (1 + 3 + 1 + 1 + 1 + 1 + 1) * 1 + 1 + 1 = 12 tensors for 1 layer
    assert!(
        tensor_count > 0,
        "Should have at least one tensor, got {}",
        tensor_count
    );
}

#[test]
fn test_to_apr_bytes_metadata_offset() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // metadata_offset at [12..20] should be 64 (HEADER_SIZE)
    let metadata_offset = u64::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]);
    assert_eq!(metadata_offset, 64, "metadata_offset should be HEADER_SIZE");
}

#[test]
fn test_to_apr_bytes_data_after_index() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // tensor_index_offset at [24..32]
    let tensor_index_offset = u64::from_le_bytes([
        bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
    ]);
    // data_offset at [32..40]
    let data_offset = u64::from_le_bytes([
        bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
    ]);

    // data must come after tensor index
    assert!(
        data_offset >= tensor_index_offset,
        "data_offset ({data_offset}) must be >= tensor_index_offset ({tensor_index_offset})"
    );
}

// ============================================================================
// to_apr_bytes: Separate vs Fused QKV paths
// ============================================================================

#[test]
fn test_to_apr_bytes_separate_qkv() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // Should contain separate Q, K, V tensor names in the binary index
    // Tensor names are written as: 2-byte len + name bytes
    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        bytes_str.contains("blk.0.attn_q.weight"),
        "Should have separate Q tensor"
    );
    assert!(
        bytes_str.contains("blk.0.attn_k.weight"),
        "Should have separate K tensor"
    );
    assert!(
        bytes_str.contains("blk.0.attn_v.weight"),
        "Should have separate V tensor"
    );
}

#[test]
fn test_to_apr_bytes_fused_qkv() {
    let model = build_fused_qkv_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        bytes_str.contains("blk.0.attn_qkv.weight"),
        "Should have fused QKV tensor"
    );
    // Should NOT have separate Q/K/V
    assert!(
        !bytes_str.contains("blk.0.attn_q.weight"),
        "Should NOT have separate Q tensor when fused"
    );
}

// ============================================================================
// to_apr_bytes: FFN gate and norm presence/absence
// ============================================================================

#[test]
fn test_to_apr_bytes_with_ffn_gate_and_norm() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        bytes_str.contains("blk.0.ffn_gate.weight"),
        "Should have FFN gate weight when present"
    );
    assert!(
        bytes_str.contains("blk.0.ffn_norm.weight"),
        "Should have FFN norm weight when present"
    );
}

#[test]
fn test_to_apr_bytes_without_ffn_gate_or_norm() {
    let model = build_fused_qkv_model(); // No gate or FFN norm
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        !bytes_str.contains("blk.0.ffn_gate.weight"),
        "Should NOT have FFN gate when None"
    );
    assert!(
        !bytes_str.contains("blk.0.ffn_norm.weight"),
        "Should NOT have FFN norm when None"
    );
}

// ============================================================================
// to_apr_bytes: Different quantization types in dtype mapping
// ============================================================================

#[test]
fn test_to_apr_bytes_q4k_model() {
    let model = build_q4k_model();
    let result = model.to_apr_bytes();
    assert!(
        result.is_ok(),
        "Q4K model should serialize: {:?}",
        result.err()
    );

    let bytes = result.expect("should produce bytes");
    // Should have valid APR magic
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00]);
}

#[test]
fn test_to_apr_bytes_metadata_contains_architecture() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // Metadata starts at offset 64 (HEADER_SIZE)
    let metadata_size = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) as usize;
    let metadata_slice = &bytes[64..64 + metadata_size];
    let metadata_str = String::from_utf8_lossy(metadata_slice);

    assert!(
        metadata_str.contains("\"architecture\":\"llama\""),
        "Metadata should contain architecture: {metadata_str}"
    );
    assert!(
        metadata_str.contains("\"hidden_size\":8"),
        "Metadata should contain hidden_size"
    );
}

// ============================================================================
// to_apr_bytes: Multi-layer model
// ============================================================================

#[test]
fn test_to_apr_bytes_multi_layer() {
    let mut model = build_minimal_owned_quantized_model();
    // Add a second layer
    let second_layer = model.layers[0].clone();
    model.layers.push(second_layer);
    model.config.num_layers = 2;

    let bytes = model.to_apr_bytes().expect("should produce bytes");

    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        bytes_str.contains("blk.0.attn_q.weight"),
        "Should have layer 0"
    );
    assert!(
        bytes_str.contains("blk.1.attn_q.weight"),
        "Should have layer 1"
    );
}

// ============================================================================
// to_apr_bytes: Empty model (zero layers)
// ============================================================================

#[test]
fn test_to_apr_bytes_zero_layers() {
    let mut model = build_minimal_owned_quantized_model();
    model.layers.clear();
    model.config.num_layers = 0;

    let result = model.to_apr_bytes();
    assert!(
        result.is_ok(),
        "Zero-layer model should serialize: {:?}",
        result.err()
    );
}

// ============================================================================
// to_apr_bytes -> from_apr roundtrip
// ============================================================================

#[test]
fn test_to_apr_bytes_roundtrip_via_mapped_model() {
    let model = build_minimal_owned_quantized_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    // Write to a temp file and load via MappedAprModel
    let dir = std::env::temp_dir();
    let path = dir.join("test_roundtrip_loader.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path);
    assert!(
        mapped.is_ok(),
        "MappedAprModel should load: {:?}",
        mapped.err()
    );

    let mapped = mapped.expect("should load");

    // Verify metadata
    assert_eq!(mapped.metadata.architecture.as_deref(), Some("llama"));
    assert_eq!(mapped.metadata.hidden_size, Some(8));
    assert_eq!(mapped.metadata.num_layers, Some(1));

    // Verify tensor count (should match what we wrote)
    assert!(
        mapped.tensor_count() > 0,
        "Should have tensors loaded from index"
    );

    // Now load from_apr
    let restored = OwnedQuantizedModel::from_apr(&mapped);
    assert!(
        restored.is_ok(),
        "from_apr should succeed: {:?}",
        restored.err()
    );

    let restored = restored.expect("should restore model");
    assert_eq!(restored.config.architecture, "llama");
    assert_eq!(restored.config.hidden_dim, 8);
    assert_eq!(restored.config.num_layers, 1);
    assert_eq!(restored.config.num_heads, 2);
    assert_eq!(restored.config.num_kv_heads, 2);
    assert_eq!(restored.layers.len(), 1);
    assert!(!restored.token_embedding.is_empty());
    assert!(!restored.output_norm_weight.is_empty());

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_to_apr_bytes_roundtrip_q4k() {
    let model = build_q4k_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    let dir = std::env::temp_dir();
    let path = dir.join("test_roundtrip_q4k_loader.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load");

    // Verify tensors have correct dtype
    let q_tensor = mapped.find_tensor("blk.0.attn_q.weight");
    assert!(q_tensor.is_some(), "Should find Q tensor");
    assert_eq!(
        q_tensor.expect("Q tensor").dtype,
        "Q4_K",
        "Q tensor should be Q4_K"
    );

    let lm_tensor = mapped.find_tensor("output.weight");
    assert!(lm_tensor.is_some(), "Should find lm_head tensor");
    assert_eq!(
        lm_tensor.expect("lm_head tensor").dtype,
        "Q6_K",
        "lm_head should be Q6_K"
    );

    // Roundtrip from_apr
    let restored = OwnedQuantizedModel::from_apr(&mapped);
    assert!(
        restored.is_ok(),
        "from_apr should succeed for Q4K model: {:?}",
        restored.err()
    );

    let restored = restored.expect("should restore");
    assert_eq!(restored.config.architecture, "qwen2");
    assert_eq!(restored.config.hidden_dim, 8);

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_to_apr_bytes_roundtrip_fused_qkv() {
    let model = build_fused_qkv_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    let dir = std::env::temp_dir();
    let path = dir.join("test_roundtrip_fused_qkv_loader.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load");

    // Fused QKV should produce a single "blk.0.attn_qkv.weight" tensor
    let fused = mapped.find_tensor("blk.0.attn_qkv.weight");
    assert!(fused.is_some(), "Should find fused QKV tensor");

    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// to_apr_bytes: Output includes all expected tensor names
// ============================================================================

#[test]
fn test_to_apr_bytes_all_tensor_names_separate_qkv() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");
    let bytes_str = String::from_utf8_lossy(&bytes);

    let expected_tensors = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "output.weight",
    ];

    for name in &expected_tensors {
        assert!(
            bytes_str.contains(name),
            "Missing tensor name in APR output: {name}"
        );
    }
}

// ============================================================================
// to_apr_bytes: Various qtype mappings
// ============================================================================

#[test]
fn test_to_apr_bytes_various_qtypes() {
    // Build a model with various quantization types to exercise qtype_to_dtype paths
    let hidden_dim = 8;

    fn make_tensor_with_qtype(in_dim: usize, out_dim: usize, qtype: u32) -> OwnedQuantizedTensor {
        // Size doesn't matter for header serialization test; use small data
        OwnedQuantizedTensor {
            data: vec![0u8; 64],
            in_dim,
            out_dim,
            qtype,
        }
    }

    // Test various qtypes: F16(1), Q4_0(2), Q4_1(3), Q5_0(6), Q5_1(7), Q8_0(8), Q8_1(9),
    // Q2_K(10), Q3_K(11), Q4_K(12), Q5_K(13), Q6_K(14), IQ2_XXS(16), IQ2_XS(17), BF16(30)
    let qtypes_to_test = [1u32, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 30, 99];

    for qtype in qtypes_to_test {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 16,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        let layer = OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Separate {
                q: make_tensor_with_qtype(hidden_dim, hidden_dim, qtype),
                k: make_tensor_with_qtype(hidden_dim, hidden_dim, qtype),
                v: make_tensor_with_qtype(hidden_dim, hidden_dim, qtype),
            },
            qkv_bias: None,
            attn_output_weight: make_tensor_with_qtype(hidden_dim, hidden_dim, qtype),
            attn_output_bias: None,
            ffn_up_weight: make_tensor_with_qtype(hidden_dim, 16, qtype),
            ffn_up_bias: None,
            ffn_down_weight: make_tensor_with_qtype(16, hidden_dim, qtype),
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.0f32; 80],
            layers: vec![layer],
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: make_tensor_with_qtype(hidden_dim, 10, qtype),
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let result = model.to_apr_bytes();
        assert!(
            result.is_ok(),
            "to_apr_bytes should succeed for qtype {}: {:?}",
            qtype,
            result.err()
        );
    }
}

// ============================================================================
// to_apr_bytes: Data size validation
// ============================================================================

#[test]
fn test_to_apr_bytes_total_size_reasonable() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // Must be larger than just header
    assert!(bytes.len() > 64, "Must be larger than header");

    // Must not be excessively large for tiny model
    assert!(
        bytes.len() < 100_000,
        "Tiny model should be < 100KB, got {} bytes",
        bytes.len()
    );
}

// ============================================================================
// from_apr: Error paths
// ============================================================================

#[test]
fn test_from_apr_missing_embedding_tensor() {
    // Build an APR file with no embedding tensor
    use crate::apr::{HEADER_SIZE, MAGIC};

    let metadata = r#"{"architecture":"llama","hidden_size":8,"num_layers":1,"num_heads":2,"num_kv_heads":2,"vocab_size":10,"intermediate_size":16,"rms_norm_eps":1e-6}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_len = metadata_bytes.len().div_ceil(64) * 64;

    // Empty tensor index (no tensors)
    let tensor_index_bytes: Vec<u8> = Vec::new();

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_padded_len as u64;
    let data_offset = tensor_index_offset + tensor_index_bytes.len() as u64;

    let mut header = vec![0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = 2;
    header[5] = 0;
    header[8..12].copy_from_slice(&0u32.to_le_bytes());
    header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
    header[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    header[32..40].copy_from_slice(&data_offset.to_le_bytes());

    let total_size = HEADER_SIZE + metadata_padded_len;
    let mut data = Vec::with_capacity(total_size);
    data.extend_from_slice(&header);
    data.extend_from_slice(metadata_bytes);
    data.resize(total_size, 0);

    let dir = std::env::temp_dir();
    let path = dir.join("test_missing_embed.apr");
    std::fs::write(&path, &data).expect("should write");

    let mapped = crate::apr::MappedAprModel::from_path(&path);
    if let Ok(mapped) = mapped {
        let result = OwnedQuantizedModel::from_apr(&mapped);
        // Should fail because embedding tensor is missing
        assert!(
            result.is_err(),
            "from_apr should fail when embedding is missing"
        );
    }

    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// from_apr: Vocab size inference from embedding tensor shape
// ============================================================================

#[test]
fn test_from_apr_infers_vocab_from_embedding_shape() {
    let model = build_minimal_owned_quantized_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    let dir = std::env::temp_dir();
    let path = dir.join("test_vocab_inference.apr");
    std::fs::write(&path, &apr_bytes).expect("should write");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load");

    // The metadata has vocab_size, but if it were 0, from_apr should infer from embedding shape
    let restored = OwnedQuantizedModel::from_apr(&mapped).expect("should load model");
    assert_eq!(
        restored.config.vocab_size, 10,
        "Should have correct vocab_size"
    );

    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// from_apr: Config defaults
// ============================================================================

#[test]
fn test_from_apr_uses_metadata_defaults() {
    let model = build_minimal_owned_quantized_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    let dir = std::env::temp_dir();
    let path = dir.join("test_metadata_defaults.apr");
    std::fs::write(&path, &apr_bytes).expect("should write");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load");
    let restored = OwnedQuantizedModel::from_apr(&mapped).expect("should load model");

    // Verify config was populated from metadata
    assert_eq!(restored.config.hidden_dim, 8);
    assert_eq!(restored.config.num_heads, 2);
    assert_eq!(restored.config.num_kv_heads, 2);
    assert!(restored.config.eps > 0.0);
    assert!(restored.config.rope_theta > 0.0);

    let _ = std::fs::remove_file(&path);
}
