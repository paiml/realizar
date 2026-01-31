//! Additional coverage tests for attention.rs
//!
//! Targets uncovered paths in:
//! - Attention: empty tensor errors, head_dim validation
//! - Flash Attention: block_size=0, shape mismatches
//! - Flash Attention v2: shape validation, SIMD paths
//! - Flash Attention parallel: shape errors, single position
//! - SlidingWindowAttention: bidirectional mode, edge cases
//! - FusedQKVAttention: weight accessors, input validation
//! - MultiHeadAttention: GQA configurations, error paths

use crate::layers::*;

// =========================================================================
// Attention: Empty and Invalid Tensor Tests
// =========================================================================

#[test]
fn test_attention_forward_empty_query_shape_error() {
    let attn = Attention::new(4).expect("test");

    // Create tensors with valid last dimension but test shape validation
    // Empty shape is caught at Tensor creation, so test with wrong head_dim
    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err(), "Should error on Q head_dim mismatch");
}

#[test]
fn test_attention_forward_empty_key_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err(), "Should error on K head_dim mismatch");
}

#[test]
fn test_attention_forward_empty_value_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err(), "Should error on V head_dim mismatch");
}

#[test]
fn test_attention_forward_single_dim_tensors() {
    let attn = Attention::new(4).expect("test");

    // 1D tensors: [4] treated as seq_len=1, head_dim=4
    let q = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // Single position self-attention should return the value
    assert_eq!(output.shape(), &[1, 4]);
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_attention_scale_computation() {
    // Test scale is computed correctly: 1 / sqrt(head_dim)
    for head_dim in [1, 4, 16, 64, 128] {
        let attn = Attention::new(head_dim).expect("test");
        let expected_scale = 1.0 / (head_dim as f32).sqrt();
        assert!(
            (attn.scale() - expected_scale).abs() < 1e-6,
            "Scale for head_dim={} should be {}",
            head_dim,
            expected_scale
        );
    }
}

// =========================================================================
// Flash Attention: Shape Validation and Edge Cases
// =========================================================================

#[test]
fn test_flash_forward_empty_q_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward should error on Q head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_empty_k_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward should error on K head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_empty_v_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim

    let result = attn.flash_forward(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward should error on V head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_kv_seq_len_mismatch() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test"); // seq_len=3
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test"); // seq_len=2

    let result = attn.flash_forward(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward should error on K/V seq_len mismatch"
    );
}

#[test]
fn test_flash_forward_single_position() {
    let attn = Attention::new(4).expect("test");

    // Single position: seq_len=1
    let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![2.0, 4.0, 6.0, 8.0]).expect("test");

    let output = attn.flash_forward(&q, &k, &v, 1).expect("test");

    // Self-attention on single position returns the value
    assert_eq!(output.shape(), &[1, 4]);
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_flash_forward_block_size_larger_than_seq() {
    let attn = Attention::new(4).expect("test");

    // seq_len=2, block_size=10 (larger than sequence)
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.flash_forward(&q, &k, &v, 10).expect("test");
    assert_eq!(output.shape(), &[2, 4]);
}

// =========================================================================
// Flash Attention v2: Shape Validation and SIMD Testing
// =========================================================================

#[test]
fn test_flash_forward_v2_empty_q_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward_v2(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_v2 should error on Q head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_v2_empty_k_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward_v2(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_v2 should error on K head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_v2_empty_v_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim

    let result = attn.flash_forward_v2(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_v2 should error on V head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_v2_kv_seq_len_mismatch() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test"); // seq_len=3
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test"); // seq_len=2

    let result = attn.flash_forward_v2(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_v2 should error on K/V seq_len mismatch"
    );
}

#[test]
fn test_flash_forward_v2_single_position() {
    let attn = Attention::new(8).expect("test");

    // Single position with head_dim=8 (exercises SIMD path with remainder)
    let q = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.flash_forward_v2(&q, &k, &v, 1).expect("test");

    assert_eq!(output.shape(), &[1, 8]);
    // Self-attention on single position returns the value
    for i in 0..8 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-5);
    }
}

#[test]
fn test_flash_forward_v2_simd_aligned_dimensions() {
    // Test with dimensions that align with SIMD (8 floats for AVX2)
    let attn = Attention::new(16).expect("test");

    let q = Tensor::from_vec(vec![4, 16], vec![0.1; 64]).expect("test");
    let k = Tensor::from_vec(vec![4, 16], vec![0.2; 64]).expect("test");
    let v = Tensor::from_vec(vec![4, 16], vec![0.3; 64]).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 2).expect("test");

    assert_eq!(standard.shape(), v2.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
            "SIMD aligned mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            v2.data()[i]
        );
    }
}

#[test]
fn test_flash_forward_v2_simd_unaligned_dimensions() {
    // Test with dimensions that don't align with SIMD (7 floats, has remainder)
    let attn = Attention::new(7).expect("test");

    let q = Tensor::from_vec(vec![3, 7], vec![0.1; 21]).expect("test");
    let k = Tensor::from_vec(vec![3, 7], vec![0.2; 21]).expect("test");
    let v = Tensor::from_vec(vec![3, 7], vec![0.3; 21]).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 2).expect("test");

    assert_eq!(standard.shape(), v2.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
            "SIMD unaligned mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            v2.data()[i]
        );
    }
}

// =========================================================================
// Flash Attention Parallel: Shape Validation and Edge Cases
// =========================================================================

#[test]
fn test_flash_forward_parallel_empty_q_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward_parallel(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_parallel should error on Q head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_parallel_empty_k_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward_parallel(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_parallel should error on K head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_parallel_empty_v_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim

    let result = attn.flash_forward_parallel(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_parallel should error on V head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_parallel_kv_seq_len_mismatch() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test"); // seq_len=3
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test"); // seq_len=2

    let result = attn.flash_forward_parallel(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_parallel should error on K/V seq_len mismatch"
    );
}

#[test]
fn test_flash_forward_parallel_single_position() {
    let attn = Attention::new(4).expect("test");

    // Single position
    let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![2.0, 4.0, 6.0, 8.0]).expect("test");

    let output = attn.flash_forward_parallel(&q, &k, &v, 1).expect("test");

    assert_eq!(output.shape(), &[1, 4]);
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
    }
}

// =========================================================================
// SlidingWindowAttention: Bidirectional Mode and Edge Cases
// =========================================================================

#[test]
fn test_sliding_window_bidirectional_shape_errors() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // Q head_dim mismatch
    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = swa.forward_with_mask(&q, &k, &v, false);
    assert!(
        result.is_err(),
        "Bidirectional should error on Q head_dim mismatch"
    );
}

#[test]
fn test_sliding_window_bidirectional_k_shape_error() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // K head_dim mismatch
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = swa.forward_with_mask(&q, &k, &v, false);
    assert!(
        result.is_err(),
        "Bidirectional should error on K head_dim mismatch"
    );
}

#[test]
fn test_sliding_window_bidirectional_v_shape_error() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // V head_dim mismatch
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");

    let result = swa.forward_with_mask(&q, &k, &v, false);
    assert!(
        result.is_err(),
        "Bidirectional should error on V head_dim mismatch"
    );
}

#[test]
fn test_sliding_window_bidirectional_kv_mismatch() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // K/V seq_len mismatch
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = swa.forward_with_mask(&q, &k, &v, false);
    assert!(
        result.is_err(),
        "Bidirectional should error on K/V seq_len mismatch"
    );
}

#[test]
fn test_sliding_window_bidirectional_single_position() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // Single position - bidirectional window centered on it
    let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![2.0, 4.0, 6.0, 8.0]).expect("test");

    let output = swa.forward_with_mask(&q, &k, &v, false).expect("test");

    assert_eq!(output.shape(), &[1, 4]);
}

#[test]
fn test_sliding_window_bidirectional_long_sequence() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // 10 positions with window_size=3, bidirectional
    let q = Tensor::from_vec(vec![10, 4], vec![1.0; 40]).expect("test");
    let k = Tensor::from_vec(vec![10, 4], vec![1.0; 40]).expect("test");
    let v = Tensor::from_vec(vec![10, 4], (0..40).map(|i| i as f32 * 0.1).collect()).expect("test");

    let output = swa.forward_with_mask(&q, &k, &v, false).expect("test");

    assert_eq!(output.shape(), &[10, 4]);

    // All outputs should be finite
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_sliding_window_memory_ratio_zero_seq() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // Zero sequence length edge case
    let ratio = swa.memory_ratio(0);
    assert!(
        (ratio - 1.0).abs() < 1e-6,
        "memory_ratio(0) should return 1.0"
    );
}

#[test]
fn test_sliding_window_memory_ratio_various_seq_lens() {
    let swa = SlidingWindowAttention::new(64, 4096).expect("test");

    // Test various sequence lengths
    let test_cases = [
        (100, 1.0),  // seq_len < window_size: min(4096, 100) / 100 = 1.0
        (4096, 1.0), // seq_len == window_size: 4096 / 4096 = 1.0
        (8192, 0.5), // seq_len > window_size: 4096 / 8192 = 0.5
    ];

    for (seq_len, expected) in test_cases {
        let ratio = swa.memory_ratio(seq_len);
        assert!(
            (ratio - expected).abs() < 0.01,
            "memory_ratio({}) should be ~{}, got {}",
            seq_len,
            expected,
            ratio
        );
    }
}

// =========================================================================
// FusedQKVAttention: Weight Accessors and Input Validation
// =========================================================================

#[test]
fn test_fused_qkv_weight_accessors() {
    let mut fused = FusedQKVAttention::new(4, 16).expect("test");

    // Test all weight accessors
    let w_q = fused.w_q_mut();
    assert_eq!(w_q.len(), 16 * 16, "w_q should have hidden_dim^2 elements");
    w_q[0] = 42.0;
    assert!((fused.w_q_mut()[0] - 42.0).abs() < 1e-6);

    let w_k = fused.w_k_mut();
    assert_eq!(w_k.len(), 16 * 16);
    w_k[0] = 43.0;
    assert!((fused.w_k_mut()[0] - 43.0).abs() < 1e-6);

    let w_v = fused.w_v_mut();
    assert_eq!(w_v.len(), 16 * 16);
    w_v[0] = 44.0;
    assert!((fused.w_v_mut()[0] - 44.0).abs() < 1e-6);

    let w_o = fused.w_o_mut();
    assert_eq!(w_o.len(), 16 * 16);
    w_o[0] = 45.0;
    assert!((fused.w_o_mut()[0] - 45.0).abs() < 1e-6);
}

#[test]
fn test_fused_qkv_forward_1d_input_error() {
    let fused = FusedQKVAttention::new(4, 16).expect("test");

    // 1D input should error (requires 2D)
    let input = Tensor::from_vec(vec![16], vec![0.1; 16]).expect("test");

    let result = fused.forward(&input);
    assert!(result.is_err(), "FusedQKV should error on 1D input");
}

#[test]
fn test_fused_qkv_forward_wrong_hidden_dim() {
    let fused = FusedQKVAttention::new(4, 16).expect("test");

    // Wrong hidden_dim
    let input = Tensor::from_vec(vec![2, 32], vec![0.1; 64]).expect("test");

    let result = fused.forward(&input);
    assert!(result.is_err(), "FusedQKV should error on wrong hidden_dim");
}

#[test]
fn test_fused_qkv_long_sequence() {
    let fused = FusedQKVAttention::new(4, 16).expect("test");

    // Long sequence
    let input = Tensor::from_vec(vec![32, 16], vec![0.1; 512]).expect("test");

    let output = fused.forward(&input).expect("test");
    assert_eq!(output.shape(), &[32, 16]);
}

#[test]
fn test_fused_qkv_hidden_dim_not_divisible_error() {
    // hidden_dim must be divisible by head_dim
    let result = FusedQKVAttention::new(7, 16);
    assert!(
        result.is_err(),
        "Should error when hidden_dim not divisible by head_dim"
    );
}

// =========================================================================
// MultiHeadAttention: GQA Configurations and Edge Cases
// =========================================================================

#[test]
fn test_mha_gqa_various_group_sizes() {
    // Test GQA with different num_heads / num_kv_heads ratios
    let test_cases = [
        (64, 8, 1), // MQA: 8 heads, 1 KV head
        (64, 8, 2), // GQA: 8 heads, 2 KV heads (4 per group)
        (64, 8, 4), // GQA: 8 heads, 4 KV heads (2 per group)
        (64, 8, 8), // MHA: 8 heads, 8 KV heads
    ];

    for (hidden_dim, num_heads, num_kv_heads) in test_cases {
        let mha =
            MultiHeadAttention::new(hidden_dim, num_heads, num_kv_heads).unwrap_or_else(|_| {
                panic!(
                    "Should create MHA with ({}, {}, {})",
                    hidden_dim, num_heads, num_kv_heads
                )
            });

        let input = Tensor::from_vec(vec![4, hidden_dim], vec![0.1; 4 * hidden_dim]).expect("test");
        let output = mha.forward(&input).expect("test");

        assert_eq!(output.shape(), &[4, hidden_dim]);
    }
}

#[test]
fn test_mha_3d_input_error() {
    let mha = MultiHeadAttention::mha(64, 8).expect("test");

    // 3D input should error (expects 2D)
    let input = Tensor::from_vec(vec![2, 4, 64], vec![0.1; 512]).expect("test");

    let result = mha.forward(&input);
    assert!(result.is_err(), "MHA should error on 3D input");
}

#[test]
fn test_mha_single_token() {
    let mha = MultiHeadAttention::mha(64, 8).expect("test");

    // Single token sequence
    let input = Tensor::from_vec(vec![1, 64], vec![0.5; 64]).expect("test");

    let output = mha.forward(&input).expect("test");
    assert_eq!(output.shape(), &[1, 64]);
}

#[test]
fn test_mha_is_mqa_is_gqa_is_mha() {
    // MQA
    let mqa = MultiHeadAttention::mqa(64, 8).expect("test");
    assert!(mqa.is_mqa());
    assert!(!mqa.is_gqa());
    assert!(!mqa.is_mha());

    // GQA
    let gqa = MultiHeadAttention::gqa(64, 8, 2).expect("test");
    assert!(!gqa.is_mqa());
    assert!(gqa.is_gqa());
    assert!(!gqa.is_mha());

    // MHA
    let mha = MultiHeadAttention::mha(64, 8).expect("test");
    assert!(!mha.is_mqa());
    assert!(!mha.is_gqa());
    assert!(mha.is_mha());
}

#[test]
fn test_mha_large_hidden_dim() {
    // Large hidden_dim
    let mha = MultiHeadAttention::mha(256, 16).expect("test");

    let input = Tensor::from_vec(vec![2, 256], vec![0.1; 512]).expect("test");

    let output = mha.forward(&input).expect("test");
    assert_eq!(output.shape(), &[2, 256]);

    // All outputs should be finite
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

// =========================================================================
// Attention: Numerical Stability Tests
// =========================================================================

#[test]
fn test_attention_large_values_stability() {
    let attn = Attention::new(4).expect("test");

    // Large values that could overflow naive softmax
    let q = Tensor::from_vec(vec![2, 4], vec![100.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![100.0; 8]).expect("test");
    let v =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    for &val in output.data() {
        assert!(val.is_finite(), "Large inputs should not cause overflow");
    }
}

#[test]
fn test_attention_small_values_stability() {
    let attn = Attention::new(4).expect("test");

    // Small values that could underflow
    let q = Tensor::from_vec(vec![2, 4], vec![1e-10; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1e-10; 8]).expect("test");
    let v =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    for &val in output.data() {
        assert!(val.is_finite(), "Small inputs should not cause underflow");
    }
}

#[test]
fn test_attention_negative_values() {
    let attn = Attention::new(4).expect("test");

    // Negative values
    let q = Tensor::from_vec(
        vec![2, 4],
        vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
    )
    .expect("test");
    let k = Tensor::from_vec(
        vec![2, 4],
        vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
    )
    .expect("test");
    let v =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    for &val in output.data() {
        assert!(val.is_finite(), "Negative inputs should work correctly");
    }
}

// =========================================================================
// All Flash Attention Variants: Consistency Tests
// =========================================================================

#[test]
fn test_all_attention_variants_consistency() {
    let attn = Attention::new(8).expect("test");

    let q = Tensor::from_vec(
        vec![4, 8],
        (0..32).map(|i| (i as f32 * 0.1).sin()).collect(),
    )
    .expect("test");
    let k = Tensor::from_vec(
        vec![4, 8],
        (0..32).map(|i| (i as f32 * 0.2).cos()).collect(),
    )
    .expect("test");
    let v = Tensor::from_vec(vec![4, 8], (0..32).map(|i| i as f32 * 0.05).collect()).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let flash = attn.flash_forward(&q, &k, &v, 2).expect("test");
    let flash_v2 = attn.flash_forward_v2(&q, &k, &v, 2).expect("test");
    let flash_parallel = attn.flash_forward_parallel(&q, &k, &v, 2).expect("test");

    // All should produce same results within tolerance
    for i in 0..standard.data().len() {
        let s = standard.data()[i];
        let f = flash.data()[i];
        let v2 = flash_v2.data()[i];
        let p = flash_parallel.data()[i];

        assert!(
            (s - f).abs() < 1e-4,
            "standard vs flash mismatch at {}: {} vs {}",
            i,
            s,
            f
        );
        assert!(
            (s - v2).abs() < 1e-4,
            "standard vs flash_v2 mismatch at {}: {} vs {}",
            i,
            s,
            v2
        );
        assert!(
            (s - p).abs() < 1e-4,
            "standard vs flash_parallel mismatch at {}: {} vs {}",
            i,
            s,
            p
        );
    }
}
