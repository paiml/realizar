//! Part 03: KV Cache, Normalization, RoPE, and Q4KWeight tests
//!
//! Additional coverage for:
//! - KVCache edge cases and boundary conditions
//! - OptimizedKVCache transposed storage
//! - Attention with cache integration
//! - Layer normalization and RMS normalization edge cases
//! - RoPE position encoding edge cases
//! - Q4KWeight quantized inference

use crate::inference::{
    apply_rope, attention_with_cache, attention_with_transposed_v, simd_layer_norm, simd_rms_norm,
    KVCache, OptimizedKVCache, Q4KWeight,
};

// ============================================================================
// KVCache Edge Cases
// ============================================================================

#[test]
fn test_kv_cache_single_layer() {
    let mut cache = KVCache::new(1, 2, 5);

    cache.store(0, &[1.0, 2.0], &[3.0, 4.0]);
    cache.advance();

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    assert_eq!(cache.get_k(0).len(), 2);
    assert_eq!(cache.get_v(0).len(), 2);
}

#[test]
fn test_kv_cache_fill_to_max() {
    let max_seq = 5;
    let mut cache = KVCache::new(1, 2, max_seq);

    // Fill to capacity
    for i in 0..max_seq {
        cache.store(0, &[i as f32; 2], &[(i + 10) as f32; 2]);
        cache.advance();
    }

    assert_eq!(cache.len(), max_seq);
    assert_eq!(cache.get_k(0).len(), max_seq * 2);
}

#[test]
fn test_kv_cache_overflow_protection() {
    let max_seq = 3;
    let mut cache = KVCache::new(1, 2, max_seq);

    // Fill to capacity
    for i in 0..max_seq {
        cache.store(0, &[i as f32; 2], &[i as f32; 2]);
        cache.advance();
    }

    // Try to store one more (should be silently ignored based on bounds check)
    cache.store(0, &[99.0; 2], &[99.0; 2]);
    cache.advance();

    // Verify the overflow store didn't corrupt data
    let k = cache.get_k(0);
    // The stored values should still be 0, 1, 2 pattern
    assert!((k[0] - 0.0).abs() < 1e-5);
    assert!((k[2] - 1.0).abs() < 1e-5);
    assert!((k[4] - 2.0).abs() < 1e-5);
}

#[test]
fn test_kv_cache_multiple_resets() {
    let mut cache = KVCache::new(2, 4, 10);

    // First sequence
    cache.store(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);

    // Reset and verify
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // Second sequence
    cache.store(0, &[3.0; 4], &[4.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);

    // Reset again
    cache.reset();
    assert!(cache.is_empty());
}

#[test]
fn test_kv_cache_layer_isolation() {
    let mut cache = KVCache::new(3, 2, 10);

    // Store different values in each layer
    cache.store(0, &[1.0, 1.0], &[1.0, 1.0]);
    cache.store(1, &[2.0, 2.0], &[2.0, 2.0]);
    cache.store(2, &[3.0, 3.0], &[3.0, 3.0]);
    cache.advance();

    // Verify each layer has its own data
    assert_eq!(cache.get_k(0), &[1.0, 1.0]);
    assert_eq!(cache.get_k(1), &[2.0, 2.0]);
    assert_eq!(cache.get_k(2), &[3.0, 3.0]);

    assert_eq!(cache.get_v(0), &[1.0, 1.0]);
    assert_eq!(cache.get_v(1), &[2.0, 2.0]);
    assert_eq!(cache.get_v(2), &[3.0, 3.0]);
}

#[test]
fn test_kv_cache_advance_without_store() {
    let mut cache = KVCache::new(1, 4, 10);

    // Advance without storing - seq_len increases but data is zeros
    cache.advance();
    cache.advance();
    cache.advance();

    assert_eq!(cache.len(), 3);
    // Data should be zeros
    let k = cache.get_k(0);
    for &v in k {
        assert!((v).abs() < 1e-5);
    }
}

// ============================================================================
// OptimizedKVCache Edge Cases
// ============================================================================

#[test]
fn test_optimized_cache_single_layer() {
    let mut cache = OptimizedKVCache::new(1, 4, 10);

    cache.store(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
    cache.advance();

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    assert_eq!(cache.max_len(), 10);
}

#[test]
fn test_optimized_cache_transposed_v_multiple_positions() {
    let mut cache = OptimizedKVCache::new(1, 2, 5);

    // Store 3 positions
    cache.store(0, &[1.0, 2.0], &[10.0, 20.0]);
    cache.advance();
    cache.store(0, &[3.0, 4.0], &[30.0, 40.0]);
    cache.advance();
    cache.store(0, &[5.0, 6.0], &[50.0, 60.0]);
    cache.advance();

    // Verify transposed layout
    let v_t = cache.get_v_transposed(0);
    // v[0] at positions 0,1,2: indices 0,1,2
    assert!((v_t[0] - 10.0).abs() < 1e-5);
    assert!((v_t[1] - 30.0).abs() < 1e-5);
    assert!((v_t[2] - 50.0).abs() < 1e-5);

    // v[1] at positions 0,1,2: indices 5,6,7 (max_seq_len stride)
    assert!((v_t[5] - 20.0).abs() < 1e-5);
    assert!((v_t[6] - 40.0).abs() < 1e-5);
    assert!((v_t[7] - 60.0).abs() < 1e-5);
}

#[test]
fn test_optimized_cache_fill_and_overflow() {
    let mut cache = OptimizedKVCache::new(1, 2, 3);

    // Fill to capacity
    for i in 0..3 {
        cache.store(0, &[i as f32; 2], &[i as f32; 2]);
        cache.advance();
    }
    assert_eq!(cache.len(), 3);

    // Try overflow - should be capped
    cache.store(0, &[99.0; 2], &[99.0; 2]);
    cache.advance();
    assert_eq!(cache.len(), 3); // Should not increase
}

#[test]
fn test_optimized_cache_multiple_layers() {
    let mut cache = OptimizedKVCache::new(4, 8, 16);

    for layer in 0..4 {
        let k = vec![(layer * 10) as f32; 8];
        let v = vec![(layer * 100) as f32; 8];
        cache.store(layer, &k, &v);
    }
    cache.advance();

    // Verify each layer
    for layer in 0..4 {
        let k = cache.get_k(layer);
        assert!((k[0] - (layer * 10) as f32).abs() < 1e-5);
    }
}

#[test]
fn test_optimized_cache_reset_clears_position() {
    let mut cache = OptimizedKVCache::new(1, 4, 10);

    cache.store(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();
    cache.store(0, &[3.0; 4], &[4.0; 4]);
    cache.advance();

    assert_eq!(cache.len(), 2);

    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // Should be able to store again from position 0
    cache.store(0, &[5.0; 4], &[6.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);

    let k = cache.get_k(0);
    assert!((k[0] - 5.0).abs() < 1e-5);
}

// ============================================================================
// Attention with Cache Edge Cases
// ============================================================================

#[test]
fn test_attention_with_cache_zero_dimension() {
    // Edge case: zero hidden dimension
    let q: Vec<f32> = vec![];
    let k_cache: Vec<f32> = vec![];
    let v_cache: Vec<f32> = vec![];
    let current_k: Vec<f32> = vec![];
    let current_v: Vec<f32> = vec![];

    let output = attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, 1);
    assert!(output.is_empty());
}

#[test]
fn test_attention_with_cache_large_cache() {
    let hidden_dim = 8;
    let num_heads = 2;
    let cache_len = 100;

    let q = vec![0.1; hidden_dim];
    let k_cache = vec![0.1; hidden_dim * cache_len];
    let v_cache = vec![0.5; hidden_dim * cache_len];
    let current_k = vec![0.1; hidden_dim];
    let current_v = vec![0.5; hidden_dim];

    let output = attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

    assert_eq!(output.len(), hidden_dim);
    // With uniform inputs, output should be close to v values
    for &v in &output {
        assert!((v - 0.5).abs() < 0.01);
    }
}

#[test]
fn test_attention_with_cache_varying_scores() {
    let _hidden_dim = 4; // Used for documentation
    let num_heads = 1;

    // Q that prefers certain K positions
    let q = vec![1.0, 0.0, 0.0, 0.0]; // Only cares about first dimension

    // Two cached positions with different first dimensions
    let k_cache = vec![
        0.0, 0.0, 0.0, 0.0, // pos 0: low score
        1.0, 0.0, 0.0, 0.0, // pos 1: high score
    ];
    let v_cache = vec![
        1.0, 1.0, 1.0, 1.0, // pos 0 value
        2.0, 2.0, 2.0, 2.0, // pos 1 value
    ];

    let current_k = vec![0.5, 0.0, 0.0, 0.0]; // medium score
    let current_v = vec![3.0, 3.0, 3.0, 3.0];

    let output = attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);

    // Output should weight pos 1 more than pos 0
    // Exact values depend on softmax, but should be between min and max V
    for &v in &output {
        assert!(v >= 1.0);
        assert!(v <= 3.0);
    }
}

// ============================================================================
// Attention with Transposed V Edge Cases
// ============================================================================

#[test]
fn test_transposed_attention_empty_cache() {
    let hidden_dim = 4;
    let num_heads = 2;
    let max_seq_len = 10;

    let q = vec![1.0; hidden_dim];
    let k_cache: Vec<f32> = vec![];
    let v_cache_transposed = vec![0.0; hidden_dim * max_seq_len];
    let current_k = vec![1.0; hidden_dim];
    let current_v = vec![5.0; hidden_dim];

    let output = attention_with_transposed_v(
        &q,
        &k_cache,
        &v_cache_transposed,
        &current_k,
        &current_v,
        num_heads,
        max_seq_len,
    );

    assert_eq!(output.len(), hidden_dim);
    // Only current_v should contribute
    for &v in &output {
        assert!((v - 5.0).abs() < 1e-5);
    }
}

#[test]
fn test_transposed_attention_multi_position() {
    let hidden_dim = 4;
    let num_heads = 2;
    let max_seq_len = 5;

    let q = vec![1.0; hidden_dim];

    // 2 cached positions
    let k_cache = vec![
        1.0, 1.0, 1.0, 1.0, // pos 0
        1.0, 1.0, 1.0, 1.0, // pos 1
    ];

    // Transposed V: v[i] at pos j is at index i * max_seq_len + j
    let mut v_cache_transposed = vec![0.0; hidden_dim * max_seq_len];
    for i in 0..hidden_dim {
        v_cache_transposed[i * max_seq_len + 0] = 1.0; // v[i] at pos 0
        v_cache_transposed[i * max_seq_len + 1] = 2.0; // v[i] at pos 1
    }

    let current_k = vec![1.0; hidden_dim];
    let current_v = vec![3.0; hidden_dim];

    let output = attention_with_transposed_v(
        &q,
        &k_cache,
        &v_cache_transposed,
        &current_k,
        &current_v,
        num_heads,
        max_seq_len,
    );

    assert_eq!(output.len(), hidden_dim);
    // Uniform attention: (1 + 2 + 3) / 3 = 2
    for &v in &output {
        assert!((v - 2.0).abs() < 1e-5);
    }
}

// ============================================================================
// Layer Normalization Edge Cases
// ============================================================================

#[test]
fn test_layer_norm_very_small_variance() {
    let input = vec![1.0, 1.0 + 1e-10, 1.0 - 1e-10, 1.0];
    let weight = vec![1.0; 4];
    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Should not produce NaN/Inf due to near-zero variance
    for &v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_layer_norm_large_range() {
    let input = vec![-1000.0, -100.0, 100.0, 1000.0];
    let weight = vec![1.0; 4];
    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Mean should be ~0
    let mean: f32 = output.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);

    // Std should be ~1
    let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
    assert!((var.sqrt() - 1.0).abs() < 0.01);
}

#[test]
fn test_layer_norm_asymmetric_scale() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 2.0, 0.5, 0.0];
    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Last element should be 0 (scaled by 0)
    assert!((output[3]).abs() < 1e-5);

    // Other elements should have non-zero values
    assert!(output[0].abs() > 0.1);
    assert!(output[1].abs() > 0.1);
    assert!(output[2].abs() > 0.1);
}

#[test]
fn test_layer_norm_scale_and_shift() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let weight = vec![1.0; 4];
    let bias = vec![10.0, 20.0, 30.0, 40.0];
    let output = simd_layer_norm(&input, &weight, Some(&bias), 1e-5);

    // With uniform input and bias, output = bias
    for (i, &v) in output.iter().enumerate() {
        let expected = (i + 1) as f32 * 10.0;
        assert!((v - expected).abs() < 1e-3);
    }
}

// ============================================================================
// RMS Normalization Edge Cases
// ============================================================================

#[test]
fn test_rms_norm_very_small_input() {
    let input = vec![1e-10, 1e-10, 1e-10];
    let weight = vec![1.0; 3];
    let output = simd_rms_norm(&input, &weight, 1e-5);

    // Should not produce NaN/Inf
    for &v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rms_norm_large_input() {
    let input = vec![1e6, 1e6, 1e6];
    let weight = vec![1.0; 3];
    let output = simd_rms_norm(&input, &weight, 1e-5);

    // RMS = sqrt(mean(x^2)) = sqrt(1e12) = 1e6
    // So output = input / rms = 1
    for &v in &output {
        assert!((v - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_rms_norm_mixed_signs() {
    let input = vec![3.0, -4.0];
    let weight = vec![1.0, 1.0];
    let output = simd_rms_norm(&input, &weight, 1e-5);

    // RMS = sqrt((9 + 16) / 2) = sqrt(12.5) ~ 3.54
    let rms = (12.5_f32).sqrt();

    assert!((output[0] - 3.0 / rms).abs() < 1e-5);
    assert!((output[1] - (-4.0 / rms)).abs() < 1e-5);
}

#[test]
fn test_rms_norm_asymmetric_weight() {
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![0.0, 1.0, 2.0];
    let output = simd_rms_norm(&input, &weight, 1e-5);

    // First element should be 0 (weight is 0)
    assert!((output[0]).abs() < 1e-5);

    // Other elements scaled appropriately
    let rms = ((1.0 + 4.0 + 9.0) / 3.0_f32).sqrt();
    assert!((output[1] - 2.0 / rms * 1.0).abs() < 1e-5);
    assert!((output[2] - 3.0 / rms * 2.0).abs() < 1e-5);
}

// ============================================================================
// RoPE Edge Cases
// ============================================================================

#[test]
fn test_rope_single_head() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    apply_rope(&mut x, 4, 1, 0, 10000.0);

    // At position 0, no rotation
    assert!((x[0] - 1.0).abs() < 1e-5);
    assert!((x[1] - 2.0).abs() < 1e-5);
    assert!((x[2] - 3.0).abs() < 1e-5);
    assert!((x[3] - 4.0).abs() < 1e-5);
}

#[test]
fn test_rope_position_one() {
    let mut x = vec![1.0, 0.0, 1.0, 0.0]; // 4 hidden, 1 head
    apply_rope(&mut x, 4, 1, 1, 10000.0);

    // At position 1, some rotation should occur
    // Magnitude should be preserved
    let mag0 = (x[0] * x[0] + x[2] * x[2]).sqrt();
    let mag1 = (x[1] * x[1] + x[3] * x[3]).sqrt();

    assert!((mag0 - 1.0).abs() < 1e-5);
    assert!((mag1 - 0.0).abs() < 1e-5);
}

#[test]
fn test_rope_very_large_position() {
    let mut x = vec![1.0; 8];
    apply_rope(&mut x, 8, 2, 10000, 10000.0);

    // Results should still be finite
    for &v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rope_small_theta() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    apply_rope(&mut x, 4, 1, 1, 100.0);

    // With smaller theta, rotations are faster
    // Magnitudes should still be preserved
    let orig_mag0 = (1.0_f32.powi(2) + 3.0_f32.powi(2)).sqrt();
    let orig_mag1 = (2.0_f32.powi(2) + 4.0_f32.powi(2)).sqrt();

    let new_mag0 = (x[0] * x[0] + x[2] * x[2]).sqrt();
    let new_mag1 = (x[1] * x[1] + x[3] * x[3]).sqrt();

    assert!((new_mag0 - orig_mag0).abs() < 1e-4);
    assert!((new_mag1 - orig_mag1).abs() < 1e-4);
}

#[test]
fn test_rope_many_heads() {
    let hidden_dim = 128;
    let num_heads = 32;
    let mut x: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.1).collect();
    let original = x.clone();

    apply_rope(&mut x, hidden_dim, num_heads, 5, 10000.0);

    // Length should be preserved
    assert_eq!(x.len(), hidden_dim);

    // Values should change (at non-zero position)
    assert!(x != original);

    // All values should be finite
    for &v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rope_position_sequence() {
    let hidden_dim = 8;
    let num_heads = 2;
    let mut results = Vec::new();

    // Apply RoPE at multiple positions
    for pos in 0..5 {
        let mut x = vec![1.0; hidden_dim];
        apply_rope(&mut x, hidden_dim, num_heads, pos, 10000.0);
        results.push(x);
    }

    // Position 0 should be unchanged
    for &v in &results[0] {
        assert!((v - 1.0).abs() < 1e-5);
    }

    // Subsequent positions should differ
    for i in 1..5 {
        assert!(results[i] != results[0]);
    }
}

// ============================================================================
// Q4KWeight Tests
// ============================================================================

#[test]
fn test_q4k_weight_compression_ratio() {
    let in_dim = 256;
    let out_dim = 4;
    let bytes_per_row = 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("valid weight");

    // F32 would need: 256 * 4 * 4 = 4096 bytes
    // Q4_K uses: 4 * 144 = 576 bytes
    let ratio = weight.compression_ratio();
    assert!(ratio > 7.0, "Expected >7x compression, got {}", ratio);
}

#[test]
fn test_q4k_weight_invalid_data_size() {
    let data = vec![0u8; 100]; // Wrong size
    let result = Q4KWeight::new(data, 256, 1);
    assert!(result.is_err());
}

#[test]
fn test_q4k_weight_memory_stats_consistency() {
    let in_dim: usize = 512;
    let out_dim: usize = 2;
    let blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = blocks_per_row * 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("valid weight");

    assert_eq!(weight.memory_bytes(), out_dim * bytes_per_row);
    assert_eq!(weight.f32_equivalent_bytes(), in_dim * out_dim * 4);
    assert!(weight.compression_ratio() > 1.0);
}

#[test]
fn test_q4k_weight_clone() {
    let in_dim = 256;
    let out_dim = 1;
    let bytes_per_row = 144;
    let data = vec![42u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("valid weight");
    let cloned = weight.clone();

    assert_eq!(weight.in_dim, cloned.in_dim);
    assert_eq!(weight.out_dim, cloned.out_dim);
    assert_eq!(weight.data, cloned.data);
}

#[test]
fn test_q4k_weight_matvec_dimension_mismatch() {
    let in_dim = 256;
    let out_dim = 1;
    let bytes_per_row = 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("valid weight");

    // Wrong input dimension
    let wrong_input = vec![1.0; in_dim + 1];
    let result = weight.matvec(&wrong_input);
    assert!(result.is_err());
}

// ============================================================================
// Integration: Full Attention Pipeline
// ============================================================================

#[test]
fn test_full_attention_pipeline() {
    let num_layers = 2;
    let hidden_dim = 8;
    let num_heads = 2;
    let max_seq_len = 10;

    let mut cache = KVCache::new(num_layers, hidden_dim, max_seq_len);

    // Simulate processing 5 tokens
    for pos in 0..5 {
        // Create Q, K, V for this position
        let q = vec![0.1 * (pos + 1) as f32; hidden_dim];
        let k = vec![0.2 * (pos + 1) as f32; hidden_dim];
        let v = vec![0.3 * (pos + 1) as f32; hidden_dim];

        // Apply RoPE to Q and K
        let mut q_rope = q.clone();
        let mut k_rope = k.clone();
        apply_rope(&mut q_rope, hidden_dim, num_heads, pos, 10000.0);
        apply_rope(&mut k_rope, hidden_dim, num_heads, pos, 10000.0);

        // Compute attention using cached KV
        for layer in 0..num_layers {
            let output = attention_with_cache(
                &q_rope,
                cache.get_k(layer),
                cache.get_v(layer),
                &k_rope,
                &v,
                num_heads,
            );

            assert_eq!(output.len(), hidden_dim);
            for &val in &output {
                assert!(val.is_finite());
            }

            // Store new KV
            cache.store(layer, &k_rope, &v);
        }
        cache.advance();
    }

    assert_eq!(cache.len(), 5);
}

#[test]
fn test_attention_with_normalization() {
    let hidden_dim = 8;
    let num_heads = 2;

    // Create input and normalize
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0; hidden_dim];

    let normalized = simd_rms_norm(&input, &weight, 1e-5);

    // Apply RoPE
    let mut q = normalized.clone();
    apply_rope(&mut q, hidden_dim, num_heads, 0, 10000.0);

    // Compute attention (no history)
    let k = normalized.clone();
    let v = normalized.clone();

    let output = attention_with_cache(&q, &[], &[], &k, &v, num_heads);

    assert_eq!(output.len(), hidden_dim);
    // Output should equal v when no history and uniform attention
    for (out, v_val) in output.iter().zip(v.iter()) {
        assert!((out - v_val).abs() < 1e-5);
    }
}
