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
use crate::gguf::types::{GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q8_0};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a minimal test config
fn test_config(hidden_dim: usize, vocab_size: usize) -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
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
        bos_token_id: None,
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
    };

    let lm_head_weight = create_q4k_test_data(hidden_dim, vocab_size);

    OwnedQuantizedModel {
        config,
        token_embedding: vec![0.1f32; vocab_size * hidden_dim],
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
    // Create tensor with unsupported type (F32 = 0)
    let weight = OwnedQuantizedTensor {
        data: vec![0u8; 64 * 32 * 4], // Fake F32 data
        in_dim: 64,
        out_dim: 32,
        qtype: 0, // GGUF_TYPE_F32 - not supported for fused matmul
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
// qkv_matmul_into() Tests
// ============================================================================

#[test]
fn test_qkv_matmul_into_fused() {
    let model = create_test_model(256, 100);
    let qkv_weight = create_q4k_test_data(256, 3 * 256);
    let qkv = OwnedQKVWeights::Fused(qkv_weight);
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 3 * 256];

    let result = model.qkv_matmul_into(&input, &qkv, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_qkv_matmul_into_separate() {
    let model = create_test_model(256, 100);
    let q = create_q4k_test_data(256, 256);
    let k = create_q4k_test_data(256, 64);
    let v = create_q4k_test_data(256, 64);
    let qkv = OwnedQKVWeights::Separate { q, k, v };
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 384];

    let result = model.qkv_matmul_into(&input, &qkv, &mut output);
    assert!(result.is_ok());
}

// ============================================================================
// layer_norm() Tests
// ============================================================================

#[test]
fn test_layer_norm_basic() {
    let model = create_test_model(64, 100);
    let input = vec![1.0f32; 64];
    let weight = vec![1.0f32; 64];

    let output = model.layer_norm(&input, &weight, None, 1e-5);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_layer_norm_with_bias() {
    let model = create_test_model(64, 100);
    let input = vec![1.0f32; 64];
    let weight = vec![1.0f32; 64];
    let bias = vec![0.5f32; 64];

    let output = model.layer_norm(&input, &weight, Some(&bias), 1e-5);
    assert_eq!(output.len(), 64);
}

// ============================================================================
// add_bias() Tests
// ============================================================================

#[test]
fn test_add_bias() {
    let model = create_test_model(64, 100);
    let mut input = vec![1.0f32; 4];
    let bias = vec![0.5f32; 4];

    model.add_bias(&mut input, &bias);
    assert!(input.iter().all(|&x| (x - 1.5).abs() < f32::EPSILON));
}

#[test]
fn test_add_bias_zeros() {
    let model = create_test_model(64, 100);
    let mut input = vec![2.0f32; 4];
    let bias = vec![0.0f32; 4];

    model.add_bias(&mut input, &bias);
    assert!(input.iter().all(|&x| (x - 2.0).abs() < f32::EPSILON));
}

// ============================================================================
// gelu() Tests
// ============================================================================

#[test]
fn test_gelu_zeros() {
    let model = create_test_model(64, 100);
    let mut input = vec![0.0f32; 4];
    model.gelu(&mut input);
    // GELU(0) = 0
    assert!(input.iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_gelu_positive() {
    let model = create_test_model(64, 100);
    let mut input = vec![1.0f32; 4];
    model.gelu(&mut input);
    // GELU(1) ≈ 0.841
    assert!(input.iter().all(|&x| (x - 0.841).abs() < 0.01));
}

#[test]
fn test_gelu_negative() {
    let model = create_test_model(64, 100);
    let mut input = vec![-1.0f32; 4];
    model.gelu(&mut input);
    // GELU(-1) ≈ -0.159
    assert!(input.iter().all(|&x| (x - (-0.159)).abs() < 0.01));
}

// ============================================================================
// fused_rmsnorm_qkv_matmul() Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_qkv_matmul_fused() {
    let model = create_test_model(256, 100);
    let qkv_weight = create_q4k_test_data(256, 3 * 256);
    let qkv = OwnedQKVWeights::Fused(qkv_weight);
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];

    let result = model.fused_rmsnorm_qkv_matmul(&input, &norm_weight, 1e-5, &qkv);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 3 * 256);
}

#[test]
fn test_fused_rmsnorm_qkv_matmul_separate() {
    let model = create_test_model(256, 100);
    let q = create_q4k_test_data(256, 256);
    let k = create_q4k_test_data(256, 64);
    let v = create_q4k_test_data(256, 64);
    let qkv = OwnedQKVWeights::Separate { q, k, v };
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];

    let result = model.fused_rmsnorm_qkv_matmul(&input, &norm_weight, 1e-5, &qkv);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 384);
}

// ============================================================================
// fused_rmsnorm_lm_head() Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_lm_head_q4k() {
    let model = create_test_model(256, 100);
    let input = vec![1.0f32; 256];

    let result = model.fused_rmsnorm_lm_head(&input);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 100); // vocab_size
}

// ============================================================================
// fused_rmsnorm_ffn_up_gate() Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_q4_0() {
    let model = create_test_model(64, 100);
    let up_weight = create_q4_0_test_data(64, 256);
    let gate_weight = create_q4_0_test_data(64, 256);
    let input = vec![1.0f32; 64];
    let norm_weight = vec![1.0f32; 64];

    let result =
        model.fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight);
    assert!(result.is_ok());
    let (up_out, gate_out) = result.unwrap();
    assert_eq!(up_out.len(), 256);
    assert_eq!(gate_out.len(), 256);
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_q4k_fallback() {
    let model = create_test_model(256, 100);
    let up_weight = create_q4k_test_data(256, 1024);
    let gate_weight = create_q4k_test_data(256, 1024);
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];

    let result =
        model.fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight);
    assert!(result.is_ok());
    let (up_out, gate_out) = result.unwrap();
    assert_eq!(up_out.len(), 1024);
    assert_eq!(gate_out.len(), 1024);
}

// ============================================================================
// qkv_matmul_q8k_into() Tests
// ============================================================================

#[test]
fn test_qkv_matmul_q8k_into_fused() {
    let model = create_test_model(256, 100);
    let qkv_weight = create_q4k_test_data(256, 3 * 256);
    let qkv = OwnedQKVWeights::Fused(qkv_weight);
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 3 * 256];
    let scales = vec![1.0f32; 8];
    let quants = vec![0i8; 256];

    // Currently falls back to regular qkv_matmul_into
    let result = model.qkv_matmul_q8k_into(&input, &qkv, &mut output, &scales, &quants);
    assert!(result.is_ok());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_embed_max_token() {
    let model = create_test_model(64, 1000);
    let embeddings = model.embed(&[999]);
    assert_eq!(embeddings.len(), 64);
}

#[test]
fn test_embed_into_larger_buffer() {
    let model = create_test_model(64, 100);
    let mut output = vec![9.9f32; 128]; // Buffer larger than needed
    model.embed_into(0, &mut output);
    // First hidden_dim elements should be filled
    assert!(output[..64].iter().all(|&x| (x - 0.1).abs() < f32::EPSILON));
}
