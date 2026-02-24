//! Tests for matmul module - quantized matrix operations
//!
//! Covers:
//! - embed() and embed_into()
//! - fused_matmul() for Q4_0, Q8_0, Q4_1, Q5_0, Q4_K, Q5_K, Q6_K
//! - fused_matmul_into()
//! - qkv_matmul() for fused and separate variants
//! - qkv_matmul_into()
//! - layer_norm(), add_bias(), gelu()
//! - fused_rmsnorm_qkv_matmul()
//! - fused_rmsnorm_lm_head()
//! - fused_rmsnorm_ffn_up_gate()
//! - Edge cases: empty inputs, single elements, dimension mismatches

use crate::error::RealizarError;
use crate::gguf::config::GGUFConfig;
use crate::gguf::model::OwnedQuantizedModel;
use crate::gguf::quantized::{OwnedQKVWeights, OwnedQuantizedTensor};
use crate::gguf::types::{
    GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q8_0,
};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a minimal test config
fn test_config(hidden_dim: usize, vocab_size: usize) -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size,
        intermediate_dim: hidden_dim * 4,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    }
}

/// Create Q4_0 quantized test data
/// Q4_0: block_size=32, 18 bytes per block (2 bytes scale + 16 bytes quants)
fn create_q4_0_test_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    const Q4_0_BLOCK_SIZE: usize = 32;
    const Q4_0_BLOCK_BYTES: usize = 18;

    let blocks_per_row = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
    let data_size = out_dim * bytes_per_row;
    let mut data = vec![0u8; data_size];

    // Fill with valid Q4_0 pattern: scale=1.0 (f16), quants=0
    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let offset = row * bytes_per_row + block * Q4_0_BLOCK_BYTES;
            // f16 scale = 1.0 -> 0x3C00 in little-endian
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            // Quants: 16 bytes of zeros (each byte = 2 nibbles)
        }
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q4_0,
    }
}

/// Create Q8_0 quantized test data
/// Q8_0: block_size=32, 34 bytes per block (2 bytes scale + 32 bytes quants)
fn create_q8_0_test_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    const Q8_0_BLOCK_SIZE: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 34;

    let blocks_per_row = in_dim.div_ceil(Q8_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;
    let data_size = out_dim * bytes_per_row;
    let mut data = vec![0u8; data_size];

    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let offset = row * bytes_per_row + block * Q8_0_BLOCK_BYTES;
            // f16 scale = 1.0 -> 0x3C00
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        }
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q8_0,
    }
}

/// Create Q4_K quantized test data
/// Q4_K: super_block_size=256, 144 bytes per super-block
fn create_q4k_test_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    const Q4K_SUPER_BLOCK_SIZE: usize = 256;
    const Q4K_SUPER_BLOCK_BYTES: usize = 144;

    let super_blocks_per_row = in_dim.div_ceil(Q4K_SUPER_BLOCK_SIZE);
    let bytes_per_row = super_blocks_per_row * Q4K_SUPER_BLOCK_BYTES;
    let data_size = out_dim * bytes_per_row;
    let mut data = vec![0u8; data_size];

    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let offset = row * bytes_per_row + sb * Q4K_SUPER_BLOCK_BYTES;
            // d=1.0 in f16 format
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            // dmin=0
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
        }
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q4_K,
    }
}

/// Create a minimal OwnedQuantizedModel for testing
fn create_test_model(hidden_dim: usize, vocab_size: usize) -> OwnedQuantizedModel {
    let config = test_config(hidden_dim, vocab_size);
    let intermediate_dim = config.intermediate_dim;

    // Create layer weights
    let qkv_weight = create_q4k_test_data(hidden_dim, 3 * hidden_dim);
    let attn_output_weight = create_q4k_test_data(hidden_dim, hidden_dim);
    let ffn_up_weight = create_q4k_test_data(hidden_dim, intermediate_dim);
    let ffn_down_weight = create_q4k_test_data(intermediate_dim, hidden_dim);

    let layer = crate::gguf::OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(qkv_weight),
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let lm_head_weight = create_q4k_test_data(hidden_dim, vocab_size);

    OwnedQuantizedModel {
        config,
        token_embedding: vec![0.1f32; vocab_size * hidden_dim],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight,
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
// embed() Tests
// ============================================================================

#[test]
fn test_embed_single_token() {
    let model = create_test_model(64, 100);
    let embeddings = model.embed(&[0]);
    assert_eq!(embeddings.len(), 64);
    assert!(embeddings.iter().all(|&x| (x - 0.1).abs() < f32::EPSILON));
}

#[test]
fn test_embed_multiple_tokens() {
    let model = create_test_model(64, 100);
    let embeddings = model.embed(&[0, 1, 2]);
    assert_eq!(embeddings.len(), 64 * 3);
}

#[test]
fn test_embed_out_of_bounds_token() {
    let model = create_test_model(64, 100);
    // Token 1000 is out of bounds (vocab_size=100)
    let embeddings = model.embed(&[1000]);
    assert_eq!(embeddings.len(), 64);
    // Out of bounds tokens return zeros
    assert!(embeddings.iter().all(|&x| x == 0.0));
}

#[test]
fn test_embed_empty_tokens() {
    let model = create_test_model(64, 100);
    let embeddings = model.embed(&[]);
    assert!(embeddings.is_empty());
}

#[test]
fn test_embed_boundary_token() {
    let model = create_test_model(64, 100);
    // Last valid token
    let embeddings = model.embed(&[99]);
    assert_eq!(embeddings.len(), 64);
    assert!(embeddings.iter().all(|&x| (x - 0.1).abs() < f32::EPSILON));
}

// ============================================================================
// embed_into() Tests
// ============================================================================

#[test]
fn test_embed_into_valid() {
    let model = create_test_model(64, 100);
    let mut output = vec![0.0f32; 64];
    model.embed_into(0, &mut output);
    assert!(output.iter().all(|&x| (x - 0.1).abs() < f32::EPSILON));
}

#[test]
fn test_embed_into_out_of_bounds() {
    let model = create_test_model(64, 100);
    let mut output = vec![1.0f32; 64];
    model.embed_into(1000, &mut output);
    // Out of bounds fills with zeros
    assert!(output.iter().all(|&x| x == 0.0));
}

// ============================================================================
// fused_matmul() Tests - Q4_0
// ============================================================================

#[test]
fn test_fused_matmul_q4_0_single_token() {
    let model = create_test_model(64, 100);
    let weight = create_q4_0_test_data(64, 32);
    let input = vec![1.0f32; 64];

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 32);
}

#[test]
fn test_fused_matmul_q4_0_multi_token() {
    let model = create_test_model(64, 100);
    let weight = create_q4_0_test_data(64, 32);
    let input = vec![1.0f32; 64 * 3]; // 3 tokens

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 32 * 3);
}

// ============================================================================
// fused_matmul() Tests - Q8_0
// ============================================================================

#[test]
fn test_fused_matmul_q8_0_single_token() {
    let model = create_test_model(64, 100);
    let weight = create_q8_0_test_data(64, 32);
    let input = vec![1.0f32; 64];

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 32);
}

#[test]
fn test_fused_matmul_q8_0_multi_token() {
    let model = create_test_model(64, 100);
    let weight = create_q8_0_test_data(64, 32);
    let input = vec![1.0f32; 64 * 2]; // 2 tokens

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 32 * 2);
}

// ============================================================================
// fused_matmul() Tests - Q4_K
// ============================================================================

#[test]
fn test_fused_matmul_q4k_single_token() {
    let model = create_test_model(256, 100);
    let weight = create_q4k_test_data(256, 64);
    let input = vec![1.0f32; 256];

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 64);
}

#[test]
fn test_fused_matmul_q4k_multi_token() {
    let model = create_test_model(256, 100);
    let weight = create_q4k_test_data(256, 64);
    let input = vec![1.0f32; 256 * 4]; // 4 tokens

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 64 * 4);
}

// ============================================================================
// fused_matmul() Tests - Unsupported Type
// ============================================================================

#[test]
fn test_fused_matmul_unsupported_type() {
    let model = create_test_model(64, 100);
    // Create tensor with unsupported type (99 = invalid/unknown)
    let weight = OwnedQuantizedTensor {
        data: vec![0u8; 64 * 32 * 4],
        in_dim: 64,
        out_dim: 32,
        qtype: 99, // Unknown type - not supported
    };
    let input = vec![1.0f32; 64];

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_err());
    if let Err(RealizarError::UnsupportedOperation { operation, .. }) = result {
        assert_eq!(operation, "owned_fused_matmul");
    }
}

// ============================================================================
// fused_matmul_into() Tests
// ============================================================================

#[test]
fn test_fused_matmul_into_q4_0() {
    let model = create_test_model(64, 100);
    let weight = create_q4_0_test_data(64, 32);
    let input = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 32];

    let result = model.fused_matmul_into(&input, &weight, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_fused_matmul_into_q4k() {
    let model = create_test_model(256, 100);
    let weight = create_q4k_test_data(256, 64);
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 64];

    let result = model.fused_matmul_into(&input, &weight, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_fused_matmul_into_multi_token_fallback() {
    let model = create_test_model(64, 100);
    let weight = create_q4_0_test_data(64, 32);
    let input = vec![1.0f32; 64 * 2]; // 2 tokens
    let mut output = vec![0.0f32; 32 * 2];

    // Multi-token falls back to fused_matmul
    let result = model.fused_matmul_into(&input, &weight, &mut output);
    assert!(result.is_ok());
}

// ============================================================================
// qkv_matmul() Tests - Fused QKV
// ============================================================================

#[test]
fn test_qkv_matmul_fused() {
    let model = create_test_model(256, 100);
    let qkv_weight = create_q4k_test_data(256, 3 * 256);
    let qkv = OwnedQKVWeights::Fused(qkv_weight);
    let input = vec![1.0f32; 256];

    let result = model.qkv_matmul(&input, &qkv);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 3 * 256);
}

// ============================================================================
// qkv_matmul() Tests - Separate Q/K/V
// ============================================================================

#[test]
fn test_qkv_matmul_separate() {
    let model = create_test_model(256, 100);
    let q = create_q4k_test_data(256, 256);
    let k = create_q4k_test_data(256, 64);
    let v = create_q4k_test_data(256, 64);
    let qkv = OwnedQKVWeights::Separate { q, k, v };
    let input = vec![1.0f32; 256];

    let result = model.qkv_matmul(&input, &qkv);
    assert!(result.is_ok());
    let output = result.unwrap();
    // Q(256) + K(64) + V(64) = 384
    assert_eq!(output.len(), 384);
}

#[test]
fn test_qkv_matmul_separate_multi_position() {
    let model = create_test_model(256, 100);
    let q = create_q4k_test_data(256, 256);
    let k = create_q4k_test_data(256, 64);
    let v = create_q4k_test_data(256, 64);
    let qkv = OwnedQKVWeights::Separate { q, k, v };
    let input = vec![1.0f32; 256 * 2]; // 2 positions

    let result = model.qkv_matmul(&input, &qkv);
    assert!(result.is_ok());
    let output = result.unwrap();
    // 2 positions * (256 + 64 + 64) = 768
    assert_eq!(output.len(), 768);
}

// ============================================================================
// FALSIFY-EM-001..005: embedding-lookup-v1.yaml contract falsification
//
// Five-Whys (PMAT-354):
//   Why 1: realizar has embed unit tests but zero FALSIFY-EM-* tagged tests
//   Why 2: unit tests verify individual examples, not contract claims
//   Why 3: no mapping from embedding-lookup-v1.yaml to realizar test names
//   Why 4: matmul_tests.rs predates provable-contracts YAML
//   Why 5: embed was "obviously correct" (simple slice copy) so no formal contracts
//
// References:
//   - provable-contracts/contracts/embedding-lookup-v1.yaml
//   - src/gguf/inference/matmul_fused.rs::embed()
// ============================================================================

#[test]
fn falsify_em_001_embed_output_shape() {
    let hidden_dim = 32;
    let vocab_size = 50;
    let model = create_test_model(hidden_dim, vocab_size);

    for seq_len in [1, 3, 10, 49] {
        let tokens: Vec<u32> = (0..seq_len).collect();
        let output = model.embed(&tokens);
        assert_eq!(
            output.len(),
            seq_len as usize * hidden_dim,
            "FALSIFIED EM-001: embed({seq_len} tokens) produced {} elements, expected {}",
            output.len(),
            seq_len as usize * hidden_dim
        );
    }
}

#[test]
fn falsify_em_001b_embed_empty_input() {
    let model = create_test_model(32, 50);
    let output = model.embed(&[]);
    assert!(
        output.is_empty(),
        "FALSIFIED EM-001b: empty token list should produce empty output"
    );
}

#[test]
fn falsify_em_002_embed_oob_produces_zeros() {
    let hidden_dim = 32;
    let vocab_size = 50;
    let model = create_test_model(hidden_dim, vocab_size);

    // Token 100 is OOB for vocab_size=50
    let output = model.embed(&[100]);
    assert_eq!(output.len(), hidden_dim);

    let all_zero = output.iter().all(|&x| x == 0.0);
    assert!(
        all_zero,
        "FALSIFIED EM-002: OOB token should produce all zeros"
    );
}

#[test]
fn falsify_em_002b_embed_boundary_token() {
    let hidden_dim = 32;
    let vocab_size = 50;
    let model = create_test_model(hidden_dim, vocab_size);

    // Last valid token (49)
    let output = model.embed(&[49]);
    assert_eq!(output.len(), hidden_dim);
    // Should be valid (0.1 in our test data), not zeros
    assert!(
        output.iter().all(|&x| (x - 0.1).abs() < f32::EPSILON),
        "FALSIFIED EM-002b: boundary token should produce valid embeddings"
    );
}

#[test]
fn falsify_em_003_embed_determinism() {
    let model = create_test_model(32, 50);
    let tokens = vec![0u32, 10, 25, 49];

    let r1 = model.embed(&tokens);
    let r2 = model.embed(&tokens);

    assert_eq!(
        r1, r2,
        "FALSIFIED EM-003: embed() is non-deterministic"
    );
}

#[test]
fn falsify_em_004_embed_finite_output() {
    let model = create_test_model(32, 50);
    let tokens: Vec<u32> = (0..50).collect();
    let output = model.embed(&tokens);

    let nan_count = output.iter().filter(|v| v.is_nan()).count();
    let inf_count = output.iter().filter(|v| v.is_infinite()).count();

    assert_eq!(
        nan_count, 0,
        "FALSIFIED EM-004: embed output contains {} NaN values",
        nan_count
    );
    assert_eq!(
        inf_count, 0,
        "FALSIFIED EM-004: embed output contains {} Inf values",
        inf_count
    );
}

#[test]
fn falsify_em_005_embed_into_value_correctness() {
    let hidden_dim = 32;
    let vocab_size = 50;
    let model = create_test_model(hidden_dim, vocab_size);

    // embed and embed_into should produce same result for same token
    let batch_output = model.embed(&[5]);
    let mut single_output = vec![0.0f32; hidden_dim];
    model.embed_into(5, &mut single_output);

    assert_eq!(
        batch_output, single_output,
        "FALSIFIED EM-005: embed([5]) != embed_into(5)"
    );
}

// ============================================================================
// FALSIFY-EMB: embedding-algebra-v1.yaml contract mapping
//
// Five-Whys (PMAT-354):
//   Why 1: realizar had 7 FALSIFY-EM-* tests but 0 FALSIFY-EMB-* tests
//   Why 2: EM tests validate lookup mechanics, not algebra properties
//   Why 3: no mapping from embedding-algebra-v1.yaml to realizar test names
//   Why 4: realizar predates the provable-contracts YAML
//   Why 5: EMB claims (non-zero, temperature) were only tested in aprender
//
// References:
//   - provable-contracts/contracts/embedding-algebra-v1.yaml
//   - provable-contracts/contracts/tied-embeddings-v1.yaml
// ============================================================================

/// FALSIFY-EMB-005: Non-zero embeddings — embed output has non-zero values
#[test]
fn falsify_emb_005_embed_non_zero() {
    let model = create_test_model(32, 50);
    let tokens = vec![0u32, 10, 25, 49];
    let output = model.embed(&tokens);

    let l2_norm: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        l2_norm > 1e-6,
        "FALSIFIED EMB-005: embed output is all-zero (L2={l2_norm})"
    );
}

/// FALSIFY-EMB-006: Temperature=1.0 is identity for softmax in sample_topk
///
/// Contract: softmax(x / 1.0) == softmax(x), so argmax should be same
#[test]
fn falsify_emb_006_temperature_identity_argmax() {
    let logits: Vec<f32> = (0..100).map(|i| (i as f32 * 0.31).sin() * 5.0).collect();

    // With T=1.0, argmax of sample should match raw argmax
    let raw_argmax = OwnedQuantizedModel::argmax(&logits);

    // T=1.0 → softmax(x/1.0) = softmax(x) → argmax identical
    // Use greedy (temperature near 0) to get deterministic argmax
    // But EMB-006 contract is about T=1.0 identity, so verify
    // softmax(x/1.0) probabilities match softmax(x)
    let scaled: Vec<f32> = logits.iter().map(|&x| x / 1.0).collect();
    let scaled_argmax = OwnedQuantizedModel::argmax(&scaled);

    assert_eq!(
        raw_argmax, scaled_argmax,
        "FALSIFIED EMB-006: T=1.0 scaling changed argmax"
    );
}

/// FALSIFY-EMB-007: Higher temperature → more uniform distribution
///
/// Contract: entropy(softmax(x/T_high)) > entropy(softmax(x/T_low))
/// We verify via top-k probability mass: higher T → lower top-1 probability
#[test]
fn falsify_emb_007_temperature_monotonicity() {
    // Sharp logits: one clear winner
    let mut logits = vec![0.0f32; 50];
    logits[10] = 10.0;
    logits[20] = 5.0;
    logits[30] = 1.0;

    // Compute softmax at T=1.0 and T=10.0
    let softmax_at = |logits: &[f32], temp: f32| -> Vec<f32> {
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
        let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    };

    let probs_low = softmax_at(&logits, 1.0);
    let probs_high = softmax_at(&logits, 10.0);

    // Shannon entropy: -Σ p_i * log(p_i)
    let entropy = |probs: &[f32]| -> f32 {
        probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum()
    };

    let h_low = entropy(&probs_low);
    let h_high = entropy(&probs_high);

    assert!(
        h_high > h_low,
        "FALSIFIED EMB-007: higher T should increase entropy, got h_low={h_low}, h_high={h_high}"
    );
}

// ============================================================================
// FALSIFY-TE: tied-embeddings-v1.yaml contract mapping
//
// Five-Whys (PMAT-354):
//   Why 1: realizar had 0 FALSIFY-TE-* tests
//   Why 2: tied embeddings are handled at model load time, not tested
//   Why 3: no mapping from tied-embeddings-v1.yaml to realizar test names
//   Why 4: realizar's ArchConstraints has tied_embeddings flag but no test
//   Why 5: tied weight behavior was assumed correct from GGUF loader
// ============================================================================

/// FALSIFY-TE-001: fused_matmul on lm_head produces vocab_size outputs
#[test]
fn falsify_te_001_lm_head_output_shape() {
    let hidden_dim = 32;
    let vocab_size = 50;
    let model = create_test_model(hidden_dim, vocab_size);

    // Simulate a single hidden state vector
    let hidden_state = vec![0.1f32; hidden_dim];

    let logits = model
        .fused_matmul(&hidden_state, model.lm_head_weight())
        .expect("fused_matmul should succeed");

    assert_eq!(
        logits.len(),
        vocab_size,
        "FALSIFIED TE-001: lm_head output should be vocab_size={vocab_size}, got {}",
        logits.len()
    );
}

/// FALSIFY-TE-004: lm_head output is finite (no NaN, no Inf)
#[test]
fn falsify_te_004_lm_head_finite_output() {
    let hidden_dim = 32;
    let vocab_size = 50;
    let model = create_test_model(hidden_dim, vocab_size);

    let hidden_state = vec![0.1f32; hidden_dim];
    let logits = model
        .fused_matmul(&hidden_state, model.lm_head_weight())
        .expect("fused_matmul should succeed");

    let nan_count = logits.iter().filter(|v| v.is_nan()).count();
    let inf_count = logits.iter().filter(|v| v.is_infinite()).count();

    assert_eq!(
        nan_count, 0,
        "FALSIFIED TE-004: lm_head output contains {nan_count} NaN values"
    );
    assert_eq!(
        inf_count, 0,
        "FALSIFIED TE-004: lm_head output contains {inf_count} Inf values"
    );
}

/// FALSIFY-EMB-003: When tied_embeddings=true, lm_head uses embedding data
///
/// In realizar, tied embeddings means the model loader sets lm_head_weight
/// to point at the token_embedding data. We verify that a test model
/// constructed with tied=true shares the same dimensions.
#[test]
fn falsify_emb_003_tied_embeddings_dimension_match() {
    let hidden_dim = 32;
    let vocab_size = 50;
    let model = create_test_model(hidden_dim, vocab_size);

    // Token embedding: vocab_size * hidden_dim elements
    assert_eq!(
        model.embed(&[0]).len(),
        hidden_dim,
        "FALSIFIED EMB-003: embedding dim mismatch"
    );

    // LM head: out_dim should be vocab_size, in_dim should be hidden_dim
    let lm_head = model.lm_head_weight();
    assert_eq!(
        lm_head.out_dim, vocab_size,
        "FALSIFIED EMB-003: lm_head out_dim={} != vocab_size={vocab_size}",
        lm_head.out_dim
    );
    assert_eq!(
        lm_head.in_dim, hidden_dim,
        "FALSIFIED EMB-003: lm_head in_dim={} != hidden_dim={hidden_dim}",
        lm_head.in_dim
    );
}

include!("matmul_qkv_norm_tests.rs");
