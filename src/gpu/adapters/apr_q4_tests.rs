//! Comprehensive tests for apr_q4.rs - Q4_0 GPU Adapter
//!
//! Tests cover:
//! - Activation functions (silu, gelu)
//! - RMSNorm with various weights and edge cases
//! - RoPE position encoding
//! - CPU attention implementation
//! - Model creation from APR transformers
//! - Error handling paths
//!
//! Target: Improve coverage from ~24% to >70%

use super::{AprQ4ToGpuAdapter, GpuModelQ4, LayerNorms};
use crate::apr_transformer::{
    AprTransformerConfig, QuantizedAprLayerQ4, QuantizedAprTensorQ4, QuantizedAprTransformerQ4,
};

// ============================================================================
// Activation Function Tests
// ============================================================================

/// Test silu activation at various points
#[test]
fn test_silu_values() {
    use super::super::apr_q4::tests::{gelu, silu};

    // Test at x=0: silu(0) = 0
    assert!((silu(0.0) - 0.0).abs() < 1e-6);

    // Test positive values
    let silu_1 = silu(1.0);
    assert!(silu_1 > 0.7 && silu_1 < 0.8, "silu(1) = {silu_1}");

    let silu_2 = silu(2.0);
    assert!(silu_2 > 1.7 && silu_2 < 1.8, "silu(2) = {silu_2}");

    // Test negative values
    let silu_neg1 = silu(-1.0);
    assert!(silu_neg1 > -0.3 && silu_neg1 < -0.2, "silu(-1) = {silu_neg1}");

    // Test large values (should approach x)
    let silu_10 = silu(10.0);
    assert!(silu_10 > 9.9, "silu(10) = {silu_10}");

    // Test very negative (should approach 0)
    let silu_neg10 = silu(-10.0);
    assert!(silu_neg10.abs() < 0.001, "silu(-10) = {silu_neg10}");
}

/// Test gelu activation at various points
#[test]
fn test_gelu_values() {
    use super::super::apr_q4::tests::{gelu, silu};

    // Test at x=0
    assert!((gelu(0.0) - 0.0).abs() < 1e-6);

    // Test positive values
    let gelu_1 = gelu(1.0);
    assert!(gelu_1 > 0.8 && gelu_1 < 0.9, "gelu(1) = {gelu_1}");

    let gelu_2 = gelu(2.0);
    assert!(gelu_2 > 1.9 && gelu_2 < 2.0, "gelu(2) = {gelu_2}");

    // Test negative values
    let gelu_neg1 = gelu(-1.0);
    assert!(gelu_neg1 > -0.2 && gelu_neg1 < -0.1, "gelu(-1) = {gelu_neg1}");

    // Test large values (should approach x)
    let gelu_10 = gelu(10.0);
    assert!(gelu_10 > 9.9, "gelu(10) = {gelu_10}");

    // Test very negative (should approach 0)
    let gelu_neg10 = gelu(-10.0);
    assert!(gelu_neg10.abs() < 0.001, "gelu(-10) = {gelu_neg10}");
}

// ============================================================================
// RMSNorm Tests
// ============================================================================

/// Test RMSNorm with uniform weights
#[test]
fn test_rms_norm_uniform_weights() {
    let model = create_tiny_model();
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    model.rms_norm_inplace(&mut x, &weight);

    // RMS = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + 1e-5)
    let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-5_f32).sqrt();
    let scale = 1.0 / rms;

    assert!((x[0] - 1.0 * scale).abs() < 1e-5);
    assert!((x[1] - 2.0 * scale).abs() < 1e-5);
    assert!((x[2] - 3.0 * scale).abs() < 1e-5);
    assert!((x[3] - 4.0 * scale).abs() < 1e-5);
}

/// Test RMSNorm with non-uniform weights
#[test]
fn test_rms_norm_weighted() {
    let model = create_tiny_model();
    let mut x = vec![2.0, 2.0, 2.0, 2.0];
    let weight = vec![0.5, 1.0, 1.5, 2.0];

    model.rms_norm_inplace(&mut x, &weight);

    // RMS = sqrt(16/4 + eps) = sqrt(4 + 1e-5) = 2.0
    let rms = (16.0 / 4.0 + 1e-5_f32).sqrt();
    let scale = 1.0 / rms;

    assert!((x[0] - 2.0 * scale * 0.5).abs() < 1e-5);
    assert!((x[1] - 2.0 * scale * 1.0).abs() < 1e-5);
    assert!((x[2] - 2.0 * scale * 1.5).abs() < 1e-5);
    assert!((x[3] - 2.0 * scale * 2.0).abs() < 1e-5);
}

/// Test RMSNorm with zeros (edge case)
#[test]
fn test_rms_norm_zeros() {
    let model = create_tiny_model();
    let mut x = vec![0.0, 0.0, 0.0, 0.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    model.rms_norm_inplace(&mut x, &weight);

    // All zeros should normalize to zeros (with eps preventing div by 0)
    for v in &x {
        assert!(v.abs() < 1e-5);
    }
}

/// Test RMSNorm with negative values
#[test]
fn test_rms_norm_negative() {
    let model = create_tiny_model();
    let mut x = vec![-1.0, -2.0, -3.0, -4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    model.rms_norm_inplace(&mut x, &weight);

    // Signs should be preserved
    assert!(x[0] < 0.0);
    assert!(x[1] < 0.0);
    assert!(x[2] < 0.0);
    assert!(x[3] < 0.0);
}

/// Test RMSNorm with weight shorter than input
#[test]
fn test_rms_norm_short_weight() {
    let model = create_tiny_model();
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![2.0, 2.0]; // Only 2 weights for 4 elements

    model.rms_norm_inplace(&mut x, &weight);

    // First two should use weight 2.0, rest should use default 1.0
    let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-5_f32).sqrt();
    let scale = 1.0 / rms;

    assert!((x[0] - 1.0 * scale * 2.0).abs() < 1e-5);
    assert!((x[1] - 2.0 * scale * 2.0).abs() < 1e-5);
    // Third and fourth use default weight 1.0
    assert!((x[2] - 3.0 * scale * 1.0).abs() < 1e-5);
    assert!((x[3] - 4.0 * scale * 1.0).abs() < 1e-5);
}

// ============================================================================
// RoPE Tests
// ============================================================================

/// Test RoPE at position 0 (should be identity for most freqs)
#[test]
fn test_rope_position_zero() {
    let model = create_model_for_rope(16, 2, 2); // hidden=16, 2 heads
    let head_dim = 16 / 2;
    let kv_dim = 2 * head_dim; // 16
    let qkv_dim = 16 + 2 * kv_dim; // 48

    // QKV for single token at position 0
    let mut qkv = vec![1.0_f32; qkv_dim];

    model.apply_rope_to_qkv(&mut qkv, 1, 16, 2, 2);

    // At position 0, angle = 0, cos=1, sin=0
    // x1' = x1 * 1 - x2 * 0 = x1
    // x2' = x1 * 0 + x2 * 1 = x2
    // So values should be unchanged at position 0
    for (i, &v) in qkv.iter().enumerate().take(16) {
        assert!(
            (v - 1.0).abs() < 1e-5,
            "Q[{i}] changed at pos 0: {v} != 1.0"
        );
    }
}

/// Test RoPE at non-zero position
#[test]
fn test_rope_position_one() {
    let model = create_model_for_rope(8, 2, 2); // hidden=8, 2 heads, head_dim=4
    let head_dim = 4;
    let kv_dim = 2 * head_dim; // 8
    let qkv_dim = 8 + 2 * kv_dim; // 24

    // QKV for single token at position 1
    let mut qkv = vec![1.0_f32; qkv_dim];

    model.apply_rope_to_qkv(&mut qkv, 1, 8, 2, 2);

    // At position 1 with theta=10000, the first rotation (i=0) has
    // freq = 1/10000^(0/4) = 1.0, angle = 1 * 1 = 1
    // cos(1) ~= 0.54, sin(1) ~= 0.84
    // So Q values SHOULD change
    let changed = qkv.iter().any(|&v| (v - 1.0).abs() > 0.01);
    assert!(changed, "RoPE should modify values at position 1");
}

/// Test RoPE preserves Q and K separately
#[test]
fn test_rope_qk_structure() {
    let model = create_model_for_rope(8, 2, 2);
    let kv_dim = 8; // 2 heads * 4 head_dim
    let qkv_dim = 8 + 2 * kv_dim; // 24

    // Initialize with distinct values
    let mut qkv: Vec<f32> = (0..qkv_dim).map(|i| i as f32).collect();
    let original_v = qkv[16..24].to_vec(); // V portion

    model.apply_rope_to_qkv(&mut qkv, 1, 8, 2, 2);

    // V (last kv_dim elements) should be unchanged
    for (i, (&orig, &new)) in original_v.iter().zip(qkv[16..24].iter()).enumerate() {
        assert!(
            (orig - new).abs() < 1e-6,
            "V[{i}] was modified: {orig} -> {new}"
        );
    }
}

/// Test apply_rope_inplace directly
#[test]
fn test_apply_rope_inplace_single_head() {
    let model = create_model_for_rope(4, 1, 1);

    // Single head, head_dim=4, half_dim=2
    let mut x = vec![1.0, 2.0, 3.0, 4.0];

    model.apply_rope_inplace(&mut x, 0, 1, 4, 10000.0);

    // Position 0: angle=0 for all freqs, cos=1, sin=0
    // x stays the same
    assert!((x[0] - 1.0).abs() < 1e-5);
    assert!((x[1] - 2.0).abs() < 1e-5);
}

/// Test apply_rope_inplace with partial head
#[test]
fn test_apply_rope_bounds_check() {
    let model = create_model_for_rope(4, 1, 1);

    // Smaller array than head_dim (edge case)
    let mut x = vec![1.0, 2.0]; // Only 2 elements

    // This should not panic - bounds check in apply_rope_inplace
    model.apply_rope_inplace(&mut x, 1, 1, 4, 10000.0);

    // Values should be partially modified (or unchanged due to bounds)
    assert!(x[0].is_finite());
    assert!(x[1].is_finite());
}

// ============================================================================
// Attention CPU Tests
// ============================================================================

/// Test single token attention returns V
#[test]
fn test_attention_single_token_returns_v() {
    let model = create_model_for_attention(8, 2, 2);

    // QKV layout: Q[8] | K[8] | V[8]
    let qkv = vec![
        // Q (ignored for single token)
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // K (ignored for single token)
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // V (this is returned)
        0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
    ];

    let out = model.attention_cpu(&qkv, 1, 8, 2, 2);

    // For single token, output = V
    assert_eq!(out.len(), 8);
    for &v in &out {
        assert!((v - 0.9).abs() < 1e-6);
    }
}

/// Test attention with GQA (grouped query attention)
#[test]
fn test_attention_gqa() {
    // 4 Q heads, 2 KV heads (GQA ratio = 2)
    let model = create_model_for_attention(8, 4, 2);

    let head_dim = 8 / 4; // 2
    let kv_dim = 2 * head_dim; // 4

    // QKV layout: Q[8] | K[4] | V[4]
    let mut qkv = vec![0.0; 8 + 4 + 4];

    // Q: all 1s
    qkv[0..8].fill(1.0);
    // K: ignored
    qkv[8..12].fill(0.5);
    // V: [0.1, 0.2, 0.3, 0.4]
    qkv[12] = 0.1;
    qkv[13] = 0.2;
    qkv[14] = 0.3;
    qkv[15] = 0.4;

    let out = model.attention_cpu(&qkv, 1, 8, 4, 2);

    // With GQA, KV heads are repeated
    // kv_repeat = 4 / 2 = 2
    // Head 0 and 1 use kv_h=0, head 2 and 3 use kv_h=1
    assert_eq!(out.len(), 8);

    // Heads 0,1 map to KV head 0: V[0..2] = [0.1, 0.2]
    assert!((out[0] - 0.1).abs() < 1e-6);
    assert!((out[1] - 0.2).abs() < 1e-6);
    assert!((out[2] - 0.1).abs() < 1e-6);
    assert!((out[3] - 0.2).abs() < 1e-6);

    // Heads 2,3 map to KV head 1: V[2..4] = [0.3, 0.4]
    assert!((out[4] - 0.3).abs() < 1e-6);
    assert!((out[5] - 0.4).abs() < 1e-6);
    assert!((out[6] - 0.3).abs() < 1e-6);
    assert!((out[7] - 0.4).abs() < 1e-6);
}

/// Test multi-token attention path
#[test]
fn test_attention_multi_token() {
    let model = create_model_for_attention(8, 2, 2);

    // For seq_len > 1, simplified path returns first hidden_dim values of V
    let qkv = vec![
        // Token 1
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Q
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, // K
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // V
    ];

    // seq_len=2 but we only have data for 1 token structure
    // The implementation should handle this gracefully
    let out = model.attention_cpu(&qkv, 2, 8, 2, 2);

    // For multi-token, returns V truncated to hidden_dim
    assert_eq!(out.len(), 8);
}

// ============================================================================
// Model Creation Tests
// ============================================================================

/// Test create_model extracts all fields correctly
#[test]
fn test_create_model_full() {
    let apr = create_test_apr_model(true, true);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.vocab_size, 100);
    assert_eq!(model.num_layers, 2);
    assert!(model.has_gate);
    assert_eq!(model.layer_norms.len(), 2);
    assert_eq!(model.token_embedding.len(), 64 * 100);
    assert_eq!(model.output_norm_weight.len(), 64);
}

/// Test create_model without gate (standard FFN)
#[test]
fn test_create_model_no_gate() {
    let apr = create_test_apr_model(false, true);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    assert!(!model.has_gate);
}

/// Test create_model with missing ffn_norm
#[test]
fn test_create_model_default_ffn_norm() {
    let apr = create_test_apr_model(true, false);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    // When ffn_norm is None, should default to vec![1.0; hidden_dim]
    for layer_norm in &model.layer_norms {
        assert_eq!(layer_norm.ffn_norm.len(), 64);
        for &w in &layer_norm.ffn_norm {
            assert!((w - 1.0).abs() < 1e-6);
        }
    }
}

/// Test create_model with empty layers
#[test]
fn test_create_model_no_layers() {
    let apr = QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            num_layers: 0,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; 64 * 100],
        layers: vec![],
        output_norm_weight: vec![1.0; 64],
        lm_head_weight: QuantizedAprTensorQ4::zeros(64, 100),
    };

    let model = AprQ4ToGpuAdapter::create_model(&apr);

    assert_eq!(model.num_layers, 0);
    assert!(!model.has_gate); // No layers means no gate
    assert!(model.layer_norms.is_empty());
}

// ============================================================================
// LayerNorms Tests
// ============================================================================

/// Test LayerNorms struct basic creation
#[test]
fn test_layer_norms_create() {
    let norms = LayerNorms {
        attn_norm: vec![1.0, 2.0, 3.0],
        ffn_norm: vec![4.0, 5.0, 6.0],
    };

    assert_eq!(norms.attn_norm.len(), 3);
    assert_eq!(norms.ffn_norm.len(), 6);
}

/// Test LayerNorms Clone
#[test]
fn test_layer_norms_clone_deep() {
    let norms = LayerNorms {
        attn_norm: vec![1.0, 2.0],
        ffn_norm: vec![3.0, 4.0],
    };

    let cloned = norms.clone();

    assert_eq!(cloned.attn_norm, norms.attn_norm);
    assert_eq!(cloned.ffn_norm, norms.ffn_norm);

    // Verify it's a deep clone
    let mut modified = cloned;
    modified.attn_norm[0] = 99.0;
    assert!((norms.attn_norm[0] - 1.0).abs() < 1e-6);
}

/// Test LayerNorms Debug
#[test]
fn test_layer_norms_debug_format() {
    let norms = LayerNorms {
        attn_norm: vec![1.0],
        ffn_norm: vec![2.0],
    };

    let debug = format!("{norms:?}");
    assert!(debug.contains("LayerNorms"));
    assert!(debug.contains("attn_norm"));
    assert!(debug.contains("ffn_norm"));
}

// ============================================================================
// GpuModelQ4 Tests
// ============================================================================

/// Test GpuModelQ4 Clone
#[test]
fn test_gpu_model_q4_clone_deep() {
    let model = create_tiny_model();
    let cloned = model.clone();

    assert_eq!(cloned.config.hidden_dim, model.config.hidden_dim);
    assert_eq!(cloned.num_layers, model.num_layers);
    assert_eq!(cloned.has_gate, model.has_gate);
    assert_eq!(cloned.token_embedding.len(), model.token_embedding.len());
}

/// Test GpuModelQ4 Debug
#[test]
fn test_gpu_model_q4_debug_format() {
    let model = create_tiny_model();
    let debug = format!("{model:?}");

    assert!(debug.contains("GpuModelQ4"));
    assert!(debug.contains("config"));
    assert!(debug.contains("num_layers"));
}

// ============================================================================
// QuantizedAprTensorQ4 Tests
// ============================================================================

/// Test QuantizedAprTensorQ4::new
#[test]
fn test_quantized_tensor_new() {
    let data = vec![1u8, 2, 3, 4];
    let tensor = QuantizedAprTensorQ4::new(data.clone(), 32, 64);

    assert_eq!(tensor.data, data);
    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 64);
}

/// Test QuantizedAprTensorQ4::zeros
#[test]
fn test_quantized_tensor_zeros() {
    let tensor = QuantizedAprTensorQ4::zeros(32, 64);

    // 32 * 64 = 2048 elements
    // 2048 / 32 = 64 blocks
    // 64 * 18 = 1152 bytes
    assert_eq!(tensor.data.len(), 1152);
    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 64);

    // All bytes should be 0
    for &b in &tensor.data {
        assert_eq!(b, 0);
    }
}

/// Test QuantizedAprTensorQ4::expected_bytes
#[test]
fn test_quantized_tensor_expected_bytes() {
    // 32 elements = 1 block = 18 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);

    // 64 elements = 2 blocks = 36 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);

    // 33 elements = 2 blocks (ceil) = 36 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(33), 36);

    // 0 elements = 0 blocks = 0 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(0), 0);
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_tiny_model() -> GpuModelQ4 {
    GpuModelQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            context_length: 16,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; 10 * 4],
        output_norm_weight: vec![1.0; 4],
        layer_norms: vec![LayerNorms {
            attn_norm: vec![1.0; 4],
            ffn_norm: vec![1.0; 4],
        }],
        num_layers: 1,
        has_gate: false,
    }
}

fn create_model_for_rope(hidden_dim: usize, num_heads: usize, num_kv_heads: usize) -> GpuModelQ4 {
    GpuModelQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            vocab_size: 10,
            intermediate_dim: hidden_dim * 2,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; 10 * hidden_dim],
        output_norm_weight: vec![1.0; hidden_dim],
        layer_norms: vec![LayerNorms {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
        }],
        num_layers: 1,
        has_gate: false,
    }
}

fn create_model_for_attention(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> GpuModelQ4 {
    GpuModelQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            vocab_size: 10,
            intermediate_dim: hidden_dim * 2,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; 10 * hidden_dim],
        output_norm_weight: vec![1.0; hidden_dim],
        layer_norms: vec![LayerNorms {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
        }],
        num_layers: 1,
        has_gate: false,
    }
}

fn create_test_apr_model(with_gate: bool, with_ffn_norm: bool) -> QuantizedAprTransformerQ4 {
    let hidden_dim = 64;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let intermediate_dim = 128;
    let vocab_size = 100;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: if with_gate {
            Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim))
        } else {
            None
        },
        ffn_norm_weight: if with_ffn_norm {
            Some(vec![1.0; hidden_dim])
        } else {
            None
        },
    };

    QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 2,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
        },
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers: vec![layer.clone(), layer],
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

// Make the private activation functions available for testing via super import
// The tests module in apr_q4.rs already exports silu and gelu
#[cfg(test)]
mod tests {
    // Re-export activation functions from parent module tests
    pub use super::super::apr_q4::silu;
    pub use super::super::apr_q4::gelu;
}
