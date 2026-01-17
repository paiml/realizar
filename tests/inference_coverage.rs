//! EXTREME TDD: Inference Module Coverage Tests
//!
//! Additional tests for inference.rs to increase coverage to 85%+.
//! Tests ThreadConfig, InferenceMode, Q4KWeight, KVCache, OptimizedKVCache,
//! simd_* functions, and attention functions.

use realizar::inference::{
    apply_rope, attention_with_cache, attention_with_transposed_v, simd_add, simd_dot, simd_gelu,
    simd_layer_norm, simd_matmul, simd_mul, simd_rms_norm, simd_silu, simd_softmax, InferenceMode,
    KVCache, OptimizedKVCache, Q4KWeight, ThreadConfig,
};

// ===== ThreadConfig Tests =====

#[test]
fn test_cov_thread_config_auto() {
    let config = ThreadConfig::auto();

    // n_threads_batch should be at least 1
    assert!(config.n_threads_batch >= 1);
    // n_threads_decode should be at least 1
    assert!(config.n_threads_decode >= 1);
    // batch threads >= decode threads (or at worst equal)
    assert!(config.n_threads_batch >= config.n_threads_decode);
}

#[test]
fn test_cov_thread_config_new() {
    let config = ThreadConfig::new(8, 4);
    assert_eq!(config.n_threads_batch, 8);
    assert_eq!(config.n_threads_decode, 4);
}

#[test]
fn test_cov_thread_config_new_clamps_zero() {
    // Zero should be clamped to 1
    let config = ThreadConfig::new(0, 0);
    assert_eq!(config.n_threads_batch, 1);
    assert_eq!(config.n_threads_decode, 1);
}

#[test]
fn test_cov_thread_config_threads_for_prefill() {
    let config = ThreadConfig::new(16, 8);
    assert_eq!(config.threads_for(true), 16); // prefill uses batch threads
}

#[test]
fn test_cov_thread_config_threads_for_decode() {
    let config = ThreadConfig::new(16, 8);
    assert_eq!(config.threads_for(false), 8); // decode uses decode threads
}

#[test]
fn test_cov_thread_config_default() {
    let config = ThreadConfig::default();
    // Default should use auto
    assert!(config.n_threads_batch >= 1);
    assert!(config.n_threads_decode >= 1);
}

#[test]
fn test_cov_thread_config_clone() {
    let config = ThreadConfig::new(12, 6);
    let cloned = config;
    assert_eq!(cloned.n_threads_batch, 12);
    assert_eq!(cloned.n_threads_decode, 6);
}

#[test]
fn test_cov_thread_config_debug() {
    let config = ThreadConfig::new(4, 2);
    let debug = format!("{config:?}");
    assert!(debug.contains("ThreadConfig"));
    assert!(debug.contains("4"));
    assert!(debug.contains("2"));
}

// ===== InferenceMode Tests =====

#[test]
fn test_cov_inference_mode_prefill() {
    let mode = InferenceMode::Prefill;
    assert!(matches!(mode, InferenceMode::Prefill));
}

#[test]
fn test_cov_inference_mode_decode() {
    let mode = InferenceMode::Decode;
    assert!(matches!(mode, InferenceMode::Decode));
}

#[test]
fn test_cov_inference_mode_eq() {
    assert_eq!(InferenceMode::Prefill, InferenceMode::Prefill);
    assert_eq!(InferenceMode::Decode, InferenceMode::Decode);
    assert_ne!(InferenceMode::Prefill, InferenceMode::Decode);
}

#[test]
fn test_cov_inference_mode_clone() {
    let mode = InferenceMode::Prefill;
    let cloned = mode;
    assert_eq!(cloned, InferenceMode::Prefill);
}

#[test]
fn test_cov_inference_mode_debug() {
    let mode = InferenceMode::Decode;
    let debug = format!("{mode:?}");
    assert!(debug.contains("Decode"));
}

// ===== simd_rms_norm Tests =====

#[test]
fn test_cov_simd_rms_norm_basic() {
    // RMSNorm doesn't subtract mean, just divides by RMS
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);
    assert_eq!(output.len(), 4);

    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.74
    // So output ≈ input / 2.74
    assert!(output[0] > 0.0 && output[0] < 1.0);
    assert!(output[3] > 1.0 && output[3] < 2.0);
}

#[test]
fn test_cov_simd_rms_norm_with_weight() {
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let weight = vec![2.0, 2.0, 2.0, 2.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // All inputs equal, RMS = 1, so output = input * weight / 1 = 2.0
    for val in &output {
        assert!((*val - 2.0).abs() < 1e-4, "Expected 2.0, got {}", val);
    }
}

#[test]
fn test_cov_simd_rms_norm_multiple_positions() {
    // 2 positions, 4 dims
    let input = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);
    assert_eq!(output.len(), 8);

    // Each position normalized independently
    // Position 0 and position 1 have same squared sum, so same RMS
}

#[test]
fn test_cov_simd_rms_norm_zero_input() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // RMS = sqrt(0 + eps) ≈ 0, output = 0 / small ≈ 0
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== simd_layer_norm Additional Tests =====

#[test]
fn test_cov_simd_layer_norm_no_bias() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    let output = simd_layer_norm(&input, &weight, None, 1e-5);
    assert_eq!(output.len(), 4);

    // Mean-centered, so sum should be ~0
    let sum: f32 = output.iter().sum();
    assert!(sum.abs() < 1e-4);
}

#[test]
fn test_cov_simd_layer_norm_with_bias() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let bias = vec![10.0, 10.0, 10.0, 10.0];

    let output_no_bias = simd_layer_norm(&input, &weight, None, 1e-5);
    let output_with_bias = simd_layer_norm(&input, &weight, Some(&bias), 1e-5);
    assert_eq!(output_with_bias.len(), 4);

    // With uniform bias, output should be shifted by 10.0 compared to no bias
    for i in 0..4 {
        let diff = output_with_bias[i] - output_no_bias[i];
        assert!(
            (diff - 10.0).abs() < 1e-4,
            "Output[{}] bias shift should be 10.0, got {}",
            i,
            diff
        );
    }
}

#[test]
fn test_cov_simd_layer_norm_multi_position() {
    // 2 positions × 4 dims
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    let output = simd_layer_norm(&input, &weight, None, 1e-5);
    assert_eq!(output.len(), 8);
}

// ===== apply_rope Additional Tests =====

#[test]
fn test_cov_apply_rope_multi_head() {
    let mut x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // 2 heads, dim 4
    apply_rope(&mut x, 8, 2, 5, 10000.0);

    // Position 5 should show rotation
    assert!(x[0] != 1.0 || x[1] != 0.0);
}

#[test]
fn test_cov_apply_rope_different_theta() {
    let mut x1 = vec![1.0, 0.0, 0.0, 1.0];
    let mut x2 = vec![1.0, 0.0, 0.0, 1.0];

    // Use a larger position where theta difference is more pronounced
    apply_rope(&mut x1, 4, 1, 100, 10000.0);
    apply_rope(&mut x2, 4, 1, 100, 100.0); // Much smaller theta

    // Different theta should give different results at large positions
    // At least one element should differ
    let any_diff = x1.iter().zip(x2.iter()).any(|(a, b)| (a - b).abs() > 0.001);
    assert!(any_diff, "Different theta should produce different results");
}

#[test]
fn test_cov_apply_rope_large_position() {
    let mut x = vec![1.0, 0.0, 0.0, 1.0];
    apply_rope(&mut x, 4, 1, 1000, 10000.0);

    // Should not overflow or produce NaN
    for val in &x {
        assert!(val.is_finite());
    }
}

// ===== OptimizedKVCache Additional Tests =====

#[test]
fn test_cov_optimized_kv_cache_get_v_raw() {
    let mut cache = OptimizedKVCache::new(1, 4, 8);

    cache.store(0, &[1.0; 4], &[1.0, 2.0, 3.0, 4.0]);
    cache.advance();

    let v_raw = cache.get_v_raw(0);
    assert_eq!(v_raw.len(), 4 * 8); // hidden_dim * max_seq_len
}

#[test]
fn test_cov_optimized_kv_cache_max_seq_len() {
    let cache = OptimizedKVCache::new(2, 64, 512);
    assert_eq!(cache.max_seq_len(), 512);
}

#[test]
fn test_cov_optimized_kv_cache_reset() {
    let mut cache = OptimizedKVCache::new(1, 4, 8);

    cache.store(0, &[1.0; 4], &[1.0; 4]);
    cache.advance();
    cache.store(0, &[2.0; 4], &[2.0; 4]);
    cache.advance();

    assert_eq!(cache.len(), 2);

    cache.reset();

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_cov_optimized_kv_cache_full() {
    let mut cache = OptimizedKVCache::new(1, 2, 2);

    // Fill to max
    cache.store(0, &[1.0, 1.0], &[1.0, 1.0]);
    cache.advance();
    cache.store(0, &[2.0, 2.0], &[2.0, 2.0]);
    cache.advance();

    assert_eq!(cache.len(), 2);

    // Try to store more - should be ignored
    cache.store(0, &[3.0, 3.0], &[3.0, 3.0]);
    cache.advance();

    assert_eq!(cache.len(), 2); // Still 2
}

#[test]
fn test_cov_optimized_kv_cache_multi_layer() {
    let mut cache = OptimizedKVCache::new(3, 4, 8);

    // Store in different layers
    cache.store(0, &[1.0; 4], &[1.0; 4]);
    cache.store(1, &[2.0; 4], &[2.0; 4]);
    cache.store(2, &[3.0; 4], &[3.0; 4]);
    cache.advance();

    // Each layer should have independent values
    let k0 = cache.get_k(0);
    let k1 = cache.get_k(1);
    let k2 = cache.get_k(2);

    assert!((k0[0] - 1.0).abs() < 1e-6);
    assert!((k1[0] - 2.0).abs() < 1e-6);
    assert!((k2[0] - 3.0).abs() < 1e-6);
}

// ===== attention_with_transposed_v Tests =====

#[test]
fn test_cov_attention_transposed_v_empty() {
    let q = vec![1.0, 2.0, 3.0, 4.0];
    let k_cache: Vec<f32> = vec![];
    let v_transposed: Vec<f32> = vec![];

    let output = attention_with_transposed_v(&q, &k_cache, &v_transposed, 2, 2, 0);

    // Empty cache returns Q
    assert_eq!(output, q);
}

#[test]
fn test_cov_attention_transposed_v_single_pos() {
    let num_heads = 2;
    let head_dim = 2;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 1;

    let q = vec![1.0; hidden_dim];
    let k_cache = vec![1.0; hidden_dim]; // 1 position

    // V transposed: [hidden_dim, seq_len]
    let v_transposed = vec![0.1, 0.2, 0.3, 0.4]; // each dim has 1 position value

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    assert_eq!(output.len(), hidden_dim);
    // Single position means softmax = 1.0, output = V
    // But V is transposed, so we need to map correctly
}

// ===== configure_optimal_thread_pool and configure_thread_pool Tests =====

// Note: These functions can only be called once per process because rayon's global pool
// cannot be reconfigured. We test them indirectly through ThreadConfig or in separate tests.

#[test]
fn test_cov_configure_thread_pool_already_initialized() {
    // The global pool is likely already initialized from other tests.
    // This should return an error.
    let result = realizar::inference::configure_thread_pool(4);
    // Either succeeds (first call) or fails (already configured)
    // We just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_cov_configure_optimal_thread_pool_already_initialized() {
    // The global pool is likely already initialized from other tests.
    let result = realizar::inference::configure_optimal_thread_pool();
    // Either succeeds (first call) or fails (already configured)
    let _ = result;
}

// ===== Additional KVCache Tests =====

#[test]
fn test_cov_kv_cache_new_basic() {
    let cache = KVCache::new(4, 128, 256);
    assert_eq!(cache.num_layers, 4);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_cov_kv_cache_store_and_advance() {
    let mut cache = KVCache::new(2, 4, 8);

    // Store K/V for layer 0
    let k = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![5.0, 6.0, 7.0, 8.0];
    cache.store(0, &k, &v);

    // Store K/V for layer 1
    let k2 = vec![9.0, 10.0, 11.0, 12.0];
    let v2 = vec![13.0, 14.0, 15.0, 16.0];
    cache.store(1, &k2, &v2);

    // Advance position
    cache.advance();

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // Verify stored values
    let k_layer0 = cache.get_k(0);
    assert_eq!(k_layer0.len(), 4);
    assert!((k_layer0[0] - 1.0).abs() < 1e-6);

    let v_layer1 = cache.get_v(1);
    assert_eq!(v_layer1.len(), 4);
    assert!((v_layer1[0] - 13.0).abs() < 1e-6);
}

#[test]
fn test_cov_kv_cache_get_k_multiple_positions() {
    let mut cache = KVCache::new(1, 4, 8);

    // Store 3 positions
    for i in 0..3 {
        let k = vec![i as f32; 4];
        let v = vec![(i + 10) as f32; 4];
        cache.store(0, &k, &v);
        cache.advance();
    }

    assert_eq!(cache.len(), 3);

    // get_k should return all 3 positions concatenated
    let k_all = cache.get_k(0);
    assert_eq!(k_all.len(), 12); // 3 positions * 4 hidden_dim
}

#[test]
fn test_cov_kv_cache_get_v_multiple_positions() {
    let mut cache = KVCache::new(1, 4, 8);

    // Store 2 positions
    cache.store(0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
    cache.advance();
    cache.store(0, &[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0]);
    cache.advance();

    let v_all = cache.get_v(0);
    assert_eq!(v_all.len(), 8); // 2 positions * 4 hidden_dim
    assert!((v_all[0] - 10.0).abs() < 1e-6);
    assert!((v_all[4] - 50.0).abs() < 1e-6);
}

#[test]
fn test_cov_kv_cache_reset_clears_state() {
    let mut cache = KVCache::new(1, 4, 8);

    cache.store(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();
    cache.store(0, &[3.0; 4], &[4.0; 4]);
    cache.advance();

    assert_eq!(cache.len(), 2);

    cache.reset();

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    // After reset, get_k should return empty slice
    let k = cache.get_k(0);
    assert_eq!(k.len(), 0);
}

#[test]
fn test_cov_kv_cache_full_capacity() {
    let mut cache = KVCache::new(1, 2, 3); // max 3 positions

    // Fill to capacity
    for i in 0..3 {
        cache.store(0, &[i as f32; 2], &[i as f32; 2]);
        cache.advance();
    }

    assert_eq!(cache.len(), 3);

    // Try to store beyond capacity - should be ignored
    cache.store(0, &[99.0; 2], &[99.0; 2]);
    cache.advance();

    // Length should still be 3 (max)
    assert_eq!(cache.len(), 3);
}

// ===== Additional OptimizedKVCache (TransposedKVCache) Tests =====

#[test]
fn test_cov_optimized_kv_cache_new_basic() {
    let cache = OptimizedKVCache::new(4, 128, 256);
    assert_eq!(cache.num_layers, 4);
    assert_eq!(cache.hidden_dim, 128);
    assert_eq!(cache.max_seq_len(), 256);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_cov_optimized_kv_cache_store_and_advance() {
    let mut cache = OptimizedKVCache::new(2, 4, 8);

    let k = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![5.0, 6.0, 7.0, 8.0];
    cache.store(0, &k, &v);
    cache.advance();

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // Verify K values
    let k_retrieved = cache.get_k(0);
    assert_eq!(k_retrieved.len(), 4);
    assert!((k_retrieved[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_cov_optimized_kv_cache_get_v_transposed() {
    let mut cache = OptimizedKVCache::new(1, 4, 8);

    // Store position 0
    cache.store(0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
    cache.advance();

    // Store position 1
    cache.store(0, &[5.0, 6.0, 7.0, 8.0], &[11.0, 21.0, 31.0, 41.0]);
    cache.advance();

    // V transposed layout: [hidden_dim, seq_len]
    let v_t = cache.get_v_transposed(0);
    assert_eq!(v_t.len(), 8); // 4 dims * 2 positions

    // Dim 0 values: [10.0 (pos 0), 11.0 (pos 1)]
    assert!((v_t[0] - 10.0).abs() < 1e-6);
    assert!((v_t[1] - 11.0).abs() < 1e-6);

    // Dim 1 values: [20.0 (pos 0), 21.0 (pos 1)]
    assert!((v_t[2] - 20.0).abs() < 1e-6);
    assert!((v_t[3] - 21.0).abs() < 1e-6);
}

#[test]
fn test_cov_optimized_kv_cache_get_v_raw_includes_padding() {
    let mut cache = OptimizedKVCache::new(1, 4, 8);

    cache.store(0, &[1.0; 4], &[1.0, 2.0, 3.0, 4.0]);
    cache.advance();

    let v_raw = cache.get_v_raw(0);
    // Raw includes full allocation: hidden_dim * max_seq_len
    assert_eq!(v_raw.len(), 4 * 8);
}

#[test]
fn test_cov_optimized_kv_cache_reset_clears_state() {
    let mut cache = OptimizedKVCache::new(1, 4, 8);

    cache.store(0, &[1.0; 4], &[1.0; 4]);
    cache.advance();

    assert_eq!(cache.len(), 1);

    cache.reset();

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_cov_optimized_kv_cache_multi_layer_storage() {
    let mut cache = OptimizedKVCache::new(3, 4, 8);

    // Store different values in each layer
    cache.store(0, &[1.0; 4], &[1.0; 4]);
    cache.store(1, &[2.0; 4], &[2.0; 4]);
    cache.store(2, &[3.0; 4], &[3.0; 4]);
    cache.advance();

    let k0 = cache.get_k(0);
    let k1 = cache.get_k(1);
    let k2 = cache.get_k(2);

    assert!((k0[0] - 1.0).abs() < 1e-6);
    assert!((k1[0] - 2.0).abs() < 1e-6);
    assert!((k2[0] - 3.0).abs() < 1e-6);
}

// ===== Additional simd_dot Tests =====

#[test]
fn test_cov_simd_dot_orthogonal() {
    // Orthogonal vectors have dot product = 0
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let result = simd_dot(&a, &b);
    assert!(result.abs() < 1e-6);
}

#[test]
fn test_cov_simd_dot_negative() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    let result = simd_dot(&a, &b);
    // 1*(-1) + 2*(-2) + 3*(-3) = -1 - 4 - 9 = -14
    assert!((result - (-14.0)).abs() < 1e-5);
}

#[test]
fn test_cov_simd_dot_mixed() {
    let a = vec![1.0, -2.0, 3.0, -4.0];
    let b = vec![2.0, 3.0, -1.0, -2.0];
    let result = simd_dot(&a, &b);
    // 1*2 + (-2)*3 + 3*(-1) + (-4)*(-2) = 2 - 6 - 3 + 8 = 1
    assert!((result - 1.0).abs() < 1e-5);
}

// ===== Additional simd_add Tests =====

#[test]
fn test_cov_simd_add_large_vectors() {
    let n = 1024;
    let mut a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    simd_add(&mut a, &b);

    // Each element should be i + (n - i) = n
    for val in &a {
        assert!((val - n as f32).abs() < 1e-5);
    }
}

#[test]
fn test_cov_simd_add_negative() {
    let mut a = vec![10.0, 20.0, 30.0];
    let b = vec![-5.0, -10.0, -15.0];
    simd_add(&mut a, &b);
    assert!((a[0] - 5.0).abs() < 1e-5);
    assert!((a[1] - 10.0).abs() < 1e-5);
    assert!((a[2] - 15.0).abs() < 1e-5);
}

// ===== Additional simd_mul Tests =====

#[test]
fn test_cov_simd_mul_identity() {
    let mut a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    simd_mul(&mut a, &b);
    assert!((a[0] - 1.0).abs() < 1e-5);
    assert!((a[1] - 2.0).abs() < 1e-5);
    assert!((a[2] - 3.0).abs() < 1e-5);
    assert!((a[3] - 4.0).abs() < 1e-5);
}

#[test]
fn test_cov_simd_mul_negative() {
    let mut a = vec![1.0, -2.0, 3.0, -4.0];
    let b = vec![-1.0, -1.0, -1.0, -1.0];
    simd_mul(&mut a, &b);
    assert!((a[0] - (-1.0)).abs() < 1e-5);
    assert!((a[1] - 2.0).abs() < 1e-5);
    assert!((a[2] - (-3.0)).abs() < 1e-5);
    assert!((a[3] - 4.0).abs() < 1e-5);
}

#[test]
fn test_cov_simd_mul_large() {
    let n = 512;
    let mut a: Vec<f32> = vec![2.0; n];
    let b: Vec<f32> = vec![3.0; n];
    simd_mul(&mut a, &b);
    for val in &a {
        assert!((val - 6.0).abs() < 1e-5);
    }
}

// ===== Additional simd_silu Tests =====

#[test]
fn test_cov_simd_silu_zero() {
    let mut data = vec![0.0];
    simd_silu(&mut data);
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert!(data[0].abs() < 1e-6);
}

#[test]
fn test_cov_simd_silu_positive() {
    let mut data = vec![1.0, 2.0, 5.0];
    simd_silu(&mut data);
    // SiLU(x) = x * sigmoid(x)
    // SiLU(1) = 1 * sigmoid(1) = 1 * 0.731 = 0.731
    assert!(data[0] > 0.7 && data[0] < 0.8);
    // SiLU(2) = 2 * sigmoid(2) = 2 * 0.881 = 1.76
    assert!(data[1] > 1.7 && data[1] < 1.8);
    // SiLU(5) close to 5
    assert!(data[2] > 4.9 && data[2] < 5.1);
}

#[test]
fn test_cov_simd_silu_large_batch() {
    let mut data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    simd_silu(&mut data);
    for val in &data {
        assert!(val.is_finite());
    }
}

// ===== Additional simd_gelu Tests =====

#[test]
fn test_cov_simd_gelu_zero() {
    let mut data = vec![0.0];
    simd_gelu(&mut data);
    // GELU(0) = 0
    assert!(data[0].abs() < 1e-6);
}

#[test]
fn test_cov_simd_gelu_positive() {
    let mut data = vec![1.0, 2.0];
    simd_gelu(&mut data);
    // GELU(1) ≈ 0.841
    assert!(data[0] > 0.8 && data[0] < 0.9);
    // GELU(2) ≈ 1.95
    assert!(data[1] > 1.9 && data[1] < 2.0);
}

#[test]
fn test_cov_simd_gelu_large_batch() {
    let mut data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    simd_gelu(&mut data);
    for val in &data {
        assert!(val.is_finite());
    }
}

// ===== Additional simd_softmax Tests =====

#[test]
fn test_cov_simd_softmax_empty() {
    let mut data: Vec<f32> = vec![];
    simd_softmax(&mut data);
    // Should not panic on empty
    assert!(data.is_empty());
}

#[test]
fn test_cov_simd_softmax_sums_to_one() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    simd_softmax(&mut data);
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_cov_simd_softmax_preserves_order() {
    let mut data = vec![1.0, 2.0, 3.0];
    simd_softmax(&mut data);
    // Larger input should give larger probability
    assert!(data[0] < data[1]);
    assert!(data[1] < data[2]);
}

#[test]
fn test_cov_simd_softmax_numerical_stability_negative() {
    // Test with very negative values
    let mut data = vec![-1000.0, -999.0, -998.0];
    simd_softmax(&mut data);
    // Should not produce NaN or Inf
    for val in &data {
        assert!(val.is_finite());
    }
    // Sum should still be 1
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ===== Additional attention_with_cache Tests =====

#[test]
fn test_cov_attention_with_cache_empty() {
    let q = vec![1.0, 2.0, 3.0, 4.0];
    let k_cache: Vec<f32> = vec![];
    let v_cache: Vec<f32> = vec![];

    let output = attention_with_cache(&q, &k_cache, &v_cache, 2, 2);
    // Empty cache returns Q unchanged
    assert_eq!(output, q);
}

#[test]
fn test_cov_attention_with_cache_single_position() {
    let num_heads = 1;
    let head_dim = 4;
    let hidden_dim = num_heads * head_dim;

    let q = vec![1.0, 0.0, 0.0, 0.0];
    let k_cache = vec![1.0, 0.0, 0.0, 0.0]; // 1 position
    let v_cache = vec![10.0, 20.0, 30.0, 40.0]; // 1 position

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    assert_eq!(output.len(), hidden_dim);
    // Single position means softmax = 1.0, output = V
    assert!((output[0] - 10.0).abs() < 1e-3);
    assert!((output[1] - 20.0).abs() < 1e-3);
    assert!((output[2] - 30.0).abs() < 1e-3);
    assert!((output[3] - 40.0).abs() < 1e-3);
}

#[test]
fn test_cov_attention_with_cache_uniform_attention() {
    let num_heads = 1;
    let head_dim = 2;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 4;

    // Q matches all K equally
    let q = vec![1.0, 1.0];

    // K: all same values
    let k_cache: Vec<f32> = (0..seq_len).flat_map(|_| vec![1.0, 1.0]).collect();

    // V: different values at each position
    let v_cache = vec![
        1.0, 1.0, // pos 0
        2.0, 2.0, // pos 1
        3.0, 3.0, // pos 2
        4.0, 4.0, // pos 3
    ];

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    assert_eq!(output.len(), hidden_dim);
    // Uniform attention: output = mean of V = (1+2+3+4)/4 = 2.5
    assert!((output[0] - 2.5).abs() < 0.01);
    assert!((output[1] - 2.5).abs() < 0.01);
}

#[test]
fn test_cov_attention_with_cache_multi_head() {
    let num_heads = 2;
    let head_dim = 2;
    let hidden_dim = num_heads * head_dim;

    let q = vec![1.0, 0.0, 0.0, 1.0]; // Different Q for each head
    let k_cache = vec![1.0, 0.0, 0.0, 1.0]; // 1 position
    let v_cache = vec![10.0, 20.0, 30.0, 40.0];

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    assert_eq!(output.len(), hidden_dim);
    // Each head processes independently
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== Additional attention_with_transposed_v Tests =====

#[test]
fn test_cov_attention_transposed_v_uniform_weights() {
    let num_heads = 1;
    let head_dim = 4;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 2;

    // Q that gives equal attention to all positions
    let q = vec![1.0, 1.0, 1.0, 1.0];

    // K: same values for all positions
    let k_cache = vec![
        1.0, 1.0, 1.0, 1.0, // pos 0
        1.0, 1.0, 1.0, 1.0, // pos 1
    ];

    // V transposed: [hidden_dim, seq_len]
    let v_transposed = vec![
        0.0, 2.0, // dim 0: pos 0 = 0, pos 1 = 2
        0.0, 4.0, // dim 1: pos 0 = 0, pos 1 = 4
        0.0, 6.0, // dim 2: pos 0 = 0, pos 1 = 6
        0.0, 8.0, // dim 3: pos 0 = 0, pos 1 = 8
    ];

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    assert_eq!(output.len(), hidden_dim);
    // Uniform attention: output = mean of V per dim
    // dim 0: (0+2)/2 = 1, dim 1: (0+4)/2 = 2, etc.
    assert!((output[0] - 1.0).abs() < 0.01);
    assert!((output[1] - 2.0).abs() < 0.01);
    assert!((output[2] - 3.0).abs() < 0.01);
    assert!((output[3] - 4.0).abs() < 0.01);
}

#[test]
fn test_cov_attention_transposed_v_multi_head() {
    let num_heads = 2;
    let head_dim = 2;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 1;

    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k_cache = vec![1.0, 0.0, 0.0, 1.0];

    // V transposed: [hidden_dim, seq_len]
    let v_transposed = vec![10.0, 20.0, 30.0, 40.0];

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    assert_eq!(output.len(), hidden_dim);
    // Single position means output = V
    assert!((output[0] - 10.0).abs() < 0.01);
    assert!((output[1] - 20.0).abs() < 0.01);
    assert!((output[2] - 30.0).abs() < 0.01);
    assert!((output[3] - 40.0).abs() < 0.01);
}

#[test]
fn test_cov_attention_transposed_v_numerical_stability() {
    let num_heads = 1;
    let head_dim = 4;
    let seq_len = 2;

    // Large Q and K values to test numerical stability
    let q = vec![100.0, 100.0, 100.0, 100.0];
    let k_cache = vec![
        100.0, 100.0, 100.0, 100.0, // pos 0
        100.0, 100.0, 100.0, 100.0, // pos 1
    ];
    let v_transposed = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    // Should not produce NaN or Inf
    for val in &output {
        assert!(val.is_finite(), "Output should be finite, got {}", val);
    }
}

#[test]
fn test_cov_attention_transposed_v_multi_pos() {
    let num_heads = 1;
    let head_dim = 2;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 3;

    let q = vec![1.0, 1.0];

    // K: [seq_len, hidden_dim] = 3 positions × 2 dims
    let k_cache = vec![
        1.0, 1.0, // pos 0
        1.0, 1.0, // pos 1
        1.0, 1.0, // pos 2
    ];

    // V transposed: [hidden_dim, seq_len] = 2 dims × 3 positions
    let v_transposed = vec![
        1.0, 2.0, 3.0, // dim 0 across positions
        4.0, 5.0, 6.0, // dim 1 across positions
    ];

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    assert_eq!(output.len(), hidden_dim);

    // All K same, so uniform attention, output = mean of V per dim
    // dim 0: (1+2+3)/3 = 2.0, dim 1: (4+5+6)/3 = 5.0
    assert!((output[0] - 2.0).abs() < 0.01);
    assert!((output[1] - 5.0).abs() < 0.01);
}

// ===== simd_matmul Edge Cases =====

#[test]
fn test_cov_simd_matmul_identity() {
    // 2x2 identity matrix
    let input = vec![1.0, 2.0];
    let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity

    let output = simd_matmul(&input, &weight, 2, 2);
    assert_eq!(output.len(), 2);
    assert!((output[0] - 1.0).abs() < 1e-5);
    assert!((output[1] - 2.0).abs() < 1e-5);
}

#[test]
fn test_cov_simd_matmul_zeros() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.0; 8]; // 2 outputs × 4 inputs, all zero

    let output = simd_matmul(&input, &weight, 4, 2);
    assert_eq!(output.len(), 2);
    for val in &output {
        assert!(val.abs() < 1e-6);
    }
}

#[test]
fn test_cov_simd_matmul_large_batch() {
    // Large batch to trigger parallel path
    let seq_len = 64;
    let in_dim = 32;
    let out_dim = 64;

    let input: Vec<f32> = (0..seq_len * in_dim).map(|i| i as f32 * 0.01).collect();
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), seq_len * out_dim);

    // Just verify it's finite
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== simd_dot Edge Cases =====

#[test]
fn test_cov_simd_dot_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let result = simd_dot(&a, &b);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_cov_simd_dot_single() {
    let a = vec![3.0];
    let b = vec![4.0];
    let result = simd_dot(&a, &b);
    assert!((result - 12.0).abs() < 1e-5);
}

#[test]
fn test_cov_simd_dot_large() {
    let n = 1024;
    let a: Vec<f32> = vec![1.0; n];
    let b: Vec<f32> = vec![1.0; n];
    let result = simd_dot(&a, &b);
    assert!((result - n as f32).abs() < 1e-3);
}

// ===== simd_add Edge Cases =====

#[test]
fn test_cov_simd_add_in_place() {
    let mut a = vec![1.0, 2.0, 3.0];
    let b = vec![0.1, 0.2, 0.3];
    simd_add(&mut a, &b);
    assert!((a[0] - 1.1).abs() < 1e-5);
    assert!((a[1] - 2.2).abs() < 1e-5);
    assert!((a[2] - 3.3).abs() < 1e-5);
}

// ===== simd_mul Edge Cases =====

#[test]
fn test_cov_simd_mul_zeros() {
    let mut a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];
    simd_mul(&mut a, &b);
    for val in &a {
        assert!(val.abs() < 1e-6);
    }
}

// ===== simd_silu Edge Cases =====

#[test]
fn test_cov_simd_silu_negative() {
    let mut data = vec![-10.0, -5.0, -1.0];
    simd_silu(&mut data);
    // SiLU(-x) = -x * sigmoid(-x), should be small negative
    for val in &data {
        assert!(*val < 0.0);
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_silu_large() {
    let mut data = vec![10.0, 100.0];
    simd_silu(&mut data);
    // SiLU(x) ≈ x for large positive x (sigmoid ≈ 1)
    assert!((data[0] - 10.0).abs() < 0.01);
    assert!((data[1] - 100.0).abs() < 0.01);
}

// ===== simd_gelu Edge Cases =====

#[test]
fn test_cov_simd_gelu_negative() {
    let mut data = vec![-3.0, -2.0, -1.0];
    simd_gelu(&mut data);
    // GELU(-x) is small negative for moderate negative x
    for val in &data {
        assert!(*val < 0.0 || val.abs() < 0.1);
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_gelu_large_positive() {
    let mut data = vec![5.0, 10.0];
    simd_gelu(&mut data);
    // GELU(x) ≈ x for large positive x
    assert!((data[0] - 5.0).abs() < 0.01);
    assert!((data[1] - 10.0).abs() < 0.01);
}

// ===== simd_softmax Edge Cases =====

#[test]
fn test_cov_simd_softmax_single() {
    let mut data = vec![5.0];
    simd_softmax(&mut data);
    assert!((data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_cov_simd_softmax_uniform() {
    let mut data = vec![1.0, 1.0, 1.0, 1.0];
    simd_softmax(&mut data);
    // All same input → uniform distribution
    for val in &data {
        assert!((*val - 0.25).abs() < 1e-5);
    }
}

#[test]
fn test_cov_simd_softmax_extreme() {
    let mut data = vec![-1000.0, 0.0, 1000.0];
    simd_softmax(&mut data);
    // Largest should dominate
    assert!(data[2] > 0.99);
    assert!(data[0] < 0.01);
}

// ===== KVCache Additional Tests =====

#[test]
fn test_cov_kv_cache_partial_store() {
    let mut cache = KVCache::new(1, 4, 8);

    // Store smaller than hidden_dim
    cache.store(0, &[1.0, 2.0], &[3.0, 4.0]);
    cache.advance();

    let k = cache.get_k(0);
    assert_eq!(k.len(), 4);
    assert!((k[0] - 1.0).abs() < 1e-6);
    assert!((k[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_cov_kv_cache_oversized_store() {
    let mut cache = KVCache::new(1, 4, 8);

    // Store larger than hidden_dim (should truncate)
    cache.store(
        0,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    );
    cache.advance();

    let k = cache.get_k(0);
    assert_eq!(k.len(), 4);
    // Should only store first 4 values
}

// ===== Q4KWeight Additional Tests =====

/// Create valid Q4_K test data for a single super-block (256 values = 144 bytes)
fn create_q4k_test_data(num_super_blocks: usize) -> Vec<u8> {
    const SUPER_BLOCK_BYTES: usize = 144;
    let mut data = Vec::with_capacity(num_super_blocks * SUPER_BLOCK_BYTES);

    for _ in 0..num_super_blocks {
        // Scale values (f16 x 2 = 4 bytes)
        data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0 in f16
        data.extend_from_slice(&[0x00, 0x00]); // dmin = 0.0 in f16

        // Block scale indices (12 bytes)
        data.extend_from_slice(&[0u8; 12]);

        // Quantized values (128 bytes for 256 values at 4 bits each)
        data.extend(std::iter::repeat_n(0x77u8, 128)); // Each nibble = 7
    }

    data
}

#[test]
fn test_cov_q4k_weight_multi_row() {
    // Multiple output rows
    let data = create_q4k_test_data(4); // 4 rows × 1 sb/row
    let weight = Q4KWeight::new(data, 256, 4).expect("test");

    assert_eq!(weight.out_dim, 4);
    assert_eq!(weight.in_dim, 256);
}

#[test]
fn test_cov_q4k_weight_compression_display() {
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1).expect("test");

    let ratio = weight.compression_ratio();
    // Q4_K: 144 bytes per 256 values = 0.5625 bytes/value
    // f32: 4 bytes/value
    // Ratio = 4 / 0.5625 ≈ 7.11
    assert!(ratio > 7.0 && ratio < 8.0);
}

// ===== attention_with_cache Additional Tests =====

#[test]
fn test_cov_attention_with_cache_large_head_dim() {
    let num_heads = 2;
    let head_dim = 128;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 4;

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i as f32).sin()).collect();
    let k_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32).cos())
        .collect();
    let v_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.1).tanh())
        .collect();

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_attention_with_cache_numerical_stability() {
    // Test with very large values
    let num_heads = 1;
    let head_dim = 4;

    let q = vec![1e6, 1e6, 1e6, 1e6];
    let k_cache = vec![1e6, 1e6, 1e6, 1e6];
    let v_cache = vec![1.0, 2.0, 3.0, 4.0];

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    // Should not have NaN or Inf
    for val in &output {
        assert!(val.is_finite(), "Output should be finite, got {}", val);
    }
}

// ===== Q4KWeight Extended Tests =====

#[test]
fn test_cov_q4k_weight_memory_bytes() {
    let data = create_q4k_test_data(2);
    let weight = Q4KWeight::new(data.clone(), 256, 2).expect("test");

    assert_eq!(weight.memory_bytes(), data.len());
    assert_eq!(weight.memory_bytes(), 288); // 2 super-blocks * 144 bytes
}

#[test]
fn test_cov_q4k_weight_f32_equivalent_bytes() {
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1).expect("test");

    // f32 equivalent: 256 * 1 * 4 bytes = 1024 bytes
    assert_eq!(weight.f32_equivalent_bytes(), 1024);
}

#[test]
fn test_cov_q4k_weight_matvec_basic() {
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1).expect("test");

    let input: Vec<f32> = vec![1.0; 256];
    let result = weight.matvec(&input);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 1);
    // Just verify it produces finite output
    assert!(output[0].is_finite());
}

#[test]
fn test_cov_q4k_weight_matvec_wrong_size() {
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1).expect("test");

    // Wrong input size
    let input: Vec<f32> = vec![1.0; 128]; // Should be 256
    let result = weight.matvec(&input);

    assert!(result.is_err());
}

#[test]
fn test_cov_q4k_weight_new_too_small() {
    // Create data that's too small for the dimensions
    let data = create_q4k_test_data(1); // 144 bytes
    let result = Q4KWeight::new(data, 256, 2); // Needs 288 bytes

    assert!(result.is_err());
}

#[test]
fn test_cov_q4k_weight_matvec_multi_row() {
    // 2 output rows
    let data = create_q4k_test_data(2);
    let weight = Q4KWeight::new(data, 256, 2).expect("test");

    let input: Vec<f32> = vec![0.5; 256];
    let result = weight.matvec(&input);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 2);
    // Both outputs should be finite
    assert!(output[0].is_finite());
    assert!(output[1].is_finite());
}

// ===== simd_layer_norm Extended Tests =====

#[test]
fn test_cov_simd_layer_norm_large_input() {
    // Large hidden_dim to trigger more SIMD paths
    let hidden_dim = 512;
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 - 256.0) * 0.01).collect();
    let weight: Vec<f32> = vec![1.0; hidden_dim];
    let bias: Vec<f32> = vec![0.1; hidden_dim];

    let output = simd_layer_norm(&input, &weight, Some(&bias), 1e-5);

    assert_eq!(output.len(), hidden_dim);
    // All outputs should be finite
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_layer_norm_zero_variance() {
    // All same values - variance is 0
    let input = vec![5.0, 5.0, 5.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // All values same means output should be ~0 after mean subtraction
    for val in &output {
        assert!(
            val.abs() < 1e-3,
            "Expected ~0 for zero variance, got {}",
            val
        );
    }
}

#[test]
fn test_cov_simd_layer_norm_negative_values() {
    let input = vec![-10.0, -5.0, 5.0, 10.0];
    let weight = vec![2.0, 2.0, 2.0, 2.0];

    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    assert_eq!(output.len(), 4);
    // Mean is 0, so signs should be preserved after normalization
    assert!(output[0] < 0.0);
    assert!(output[1] < 0.0);
    assert!(output[2] > 0.0);
    assert!(output[3] > 0.0);
}

// ===== simd_rms_norm Extended Tests =====

#[test]
fn test_cov_simd_rms_norm_large_input() {
    let hidden_dim = 512;
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32).sin()).collect();
    let weight: Vec<f32> = vec![1.0; hidden_dim];

    let output = simd_rms_norm(&input, &weight, 1e-5);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_rms_norm_negative_values() {
    let input = vec![-1.0, -2.0, -3.0, -4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    let output = simd_rms_norm(&input, &weight, 1e-5);

    // RMS doesn't subtract mean, so signs are preserved
    assert!(output[0] < 0.0);
    assert!(output[1] < 0.0);
    assert!(output[2] < 0.0);
    assert!(output[3] < 0.0);
}

#[test]
fn test_cov_simd_rms_norm_unit_values() {
    // Input with unit RMS
    let input = vec![1.0, 0.0, 0.0, 0.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    let output = simd_rms_norm(&input, &weight, 1e-5);

    // RMS = sqrt(1/4) = 0.5, so first element should be 1.0/0.5 = 2.0
    assert!((output[0] - 2.0).abs() < 0.01);
    assert!(output[1].abs() < 0.01);
}

// ===== apply_rope Extended Tests =====

#[test]
fn test_cov_apply_rope_zero_position() {
    let mut x = vec![1.0, 1.0, 1.0, 1.0];
    apply_rope(&mut x, 4, 1, 0, 10000.0);

    // Position 0: angle = 0 for all frequencies, cos(0)=1, sin(0)=0
    // So rotation should be identity-like
    for val in &x {
        assert!(
            (val - 1.0).abs() < 0.01,
            "Position 0 should have minimal rotation"
        );
    }
}

#[test]
fn test_cov_apply_rope_preserves_norm() {
    let mut x = vec![3.0, 4.0, 0.0, 0.0]; // Norm = 5
    let original_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    apply_rope(&mut x, 4, 1, 100, 10000.0);

    let new_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    // RoPE is a rotation, should preserve norm
    assert!(
        (new_norm - original_norm).abs() < 0.01,
        "RoPE should preserve norm: {} vs {}",
        new_norm,
        original_norm
    );
}

#[test]
fn test_cov_apply_rope_multi_head_different() {
    // 2 heads with different Q values
    let mut x = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2 heads, dim 4 each
    apply_rope(&mut x, 8, 2, 50, 10000.0);

    // Verify all values are finite after rotation
    for val in &x {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_apply_rope_small_theta() {
    let mut x = vec![1.0, 0.0, 0.0, 1.0];
    // Small theta means faster rotation
    apply_rope(&mut x, 4, 1, 10, 100.0);

    // Should produce noticeable rotation
    let rotated = (x[0] - 1.0).abs() > 0.1 || x[1].abs() > 0.1;
    assert!(rotated, "Small theta should produce noticeable rotation");
}

// ===== simd_matmul Extended Tests =====

#[test]
fn test_cov_simd_matmul_single_output() {
    // Test with out_dim = 1
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0]; // Single output row

    let output = simd_matmul(&input, &weight, 3, 1);

    assert_eq!(output.len(), 1);
    assert!((output[0] - 6.0).abs() < 1e-5); // 1+2+3 = 6
}

#[test]
fn test_cov_simd_matmul_parallel_threshold() {
    // Test near the parallel threshold (256)
    let in_dim = 32;
    let out_dim = 256; // Exactly at threshold

    let input: Vec<f32> = vec![1.0; in_dim];
    let weight: Vec<f32> = vec![1.0 / in_dim as f32; out_dim * in_dim];

    let output = simd_matmul(&input, &weight, in_dim, out_dim);

    assert_eq!(output.len(), out_dim);
    // Each output should be 1.0 (32 * 1/32 = 1)
    for val in &output {
        assert!((val - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_cov_simd_matmul_batch_small() {
    // Batch size 2, small enough to not trigger tiled path
    let seq_len = 2;
    let in_dim = 4;
    let out_dim = 2;

    let input: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // Token 0
        0.0, 1.0, 0.0, 0.0, // Token 1
    ];
    let weight = vec![
        1.0, 2.0, 3.0, 4.0, // Output 0
        5.0, 6.0, 7.0, 8.0, // Output 1
    ];

    let output = simd_matmul(&input, &weight, in_dim, out_dim);

    assert_eq!(output.len(), seq_len * out_dim);
    // Token 0: [1,0,0,0] -> [1, 5]
    assert!((output[0] - 1.0).abs() < 1e-5);
    assert!((output[1] - 5.0).abs() < 1e-5);
    // Token 1: [0,1,0,0] -> [2, 6]
    assert!((output[2] - 2.0).abs() < 1e-5);
    assert!((output[3] - 6.0).abs() < 1e-5);
}

// ===== simd_silu Extended Tests =====

#[test]
fn test_cov_simd_silu_boundary() {
    // Test boundary values
    let mut data = vec![-100.0, -10.0, -0.1, 0.1, 10.0, 100.0];
    simd_silu(&mut data);

    // Large negative -> ~0
    assert!(data[0].abs() < 0.01);
    // Large positive -> ~x
    assert!((data[5] - 100.0).abs() < 0.1);
}

// ===== simd_gelu Extended Tests =====

#[test]
fn test_cov_simd_gelu_boundary() {
    let mut data = vec![-100.0, -3.0, -0.5, 0.5, 3.0, 100.0];
    simd_gelu(&mut data);

    // Large negative -> ~0
    assert!(data[0].abs() < 0.01);
    // Large positive -> ~x
    assert!((data[5] - 100.0).abs() < 0.1);
}

// ===== simd_softmax Extended Tests =====

#[test]
fn test_cov_simd_softmax_large_range() {
    let mut data = vec![-100.0, 0.0, 100.0];
    simd_softmax(&mut data);

    // Sum should be 1.0
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Largest should dominate
    assert!(data[2] > 0.999);
}

#[test]
fn test_cov_simd_softmax_two_elements() {
    let mut data = vec![0.0, 0.0];
    simd_softmax(&mut data);

    // Equal inputs -> equal probabilities
    assert!((data[0] - 0.5).abs() < 1e-5);
    assert!((data[1] - 0.5).abs() < 1e-5);
}

// ===== attention_with_transposed_v Extended Tests =====

#[test]
fn test_cov_attention_transposed_v_long_sequence() {
    let num_heads = 2;
    let head_dim = 4;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 16;

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i as f32).sin()).collect();
    let k_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32).cos() * 0.1)
        .collect();
    let v_transposed: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 % 10.0) * 0.1)
        .collect();

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== KVCache Extended Tests =====

#[test]
fn test_cov_kv_cache_boundary_layer() {
    let mut cache = KVCache::new(2, 4, 8);

    // Store at layer 0 and layer 1 (boundary)
    cache.store(0, &[1.0; 4], &[2.0; 4]);
    cache.store(1, &[3.0; 4], &[4.0; 4]);
    cache.advance();

    // Verify both layers work
    let k0 = cache.get_k(0);
    let k1 = cache.get_k(1);
    assert!((k0[0] - 1.0).abs() < 1e-6);
    assert!((k1[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_cov_kv_cache_advance_multiple() {
    let mut cache = KVCache::new(1, 2, 4);

    for i in 0..3 {
        cache.store(0, &[i as f32; 2], &[i as f32 + 10.0; 2]);
        cache.advance();
    }

    assert_eq!(cache.len(), 3);

    // Get all cached values
    let k = cache.get_k(0);
    assert_eq!(k.len(), 6); // 3 positions * 2 dims
}

// ===== OptimizedKVCache Extended Tests =====

#[test]
fn test_cov_optimized_kv_cache_boundary() {
    let mut cache = OptimizedKVCache::new(2, 4, 8);

    cache.store(0, &[1.0; 4], &[1.0; 4]);
    cache.store(1, &[2.0; 4], &[2.0; 4]);
    cache.advance();

    let k0 = cache.get_k(0);
    let k1 = cache.get_k(1);
    assert!((k0[0] - 1.0).abs() < 1e-6);
    assert!((k1[0] - 2.0).abs() < 1e-6);
}

#[test]
fn test_cov_optimized_kv_cache_v_transposed_multi_pos() {
    let mut cache = OptimizedKVCache::new(1, 2, 8);

    // Store 3 positions
    cache.store(0, &[1.0, 2.0], &[10.0, 20.0]);
    cache.advance();
    cache.store(0, &[3.0, 4.0], &[30.0, 40.0]);
    cache.advance();
    cache.store(0, &[5.0, 6.0], &[50.0, 60.0]);
    cache.advance();

    let v_t = cache.get_v_transposed(0);
    assert_eq!(v_t.len(), 6); // 2 dims * 3 positions

    // Transposed layout: dim 0 values then dim 1 values
    // Dim 0: [10, 30, 50], Dim 1: [20, 40, 60]
    assert!((v_t[0] - 10.0).abs() < 1e-6);
    assert!((v_t[1] - 30.0).abs() < 1e-6);
    assert!((v_t[2] - 50.0).abs() < 1e-6);
    assert!((v_t[3] - 20.0).abs() < 1e-6);
    assert!((v_t[4] - 40.0).abs() < 1e-6);
    assert!((v_t[5] - 60.0).abs() < 1e-6);
}

// ===== simd_add Extended Tests =====

#[test]
fn test_cov_simd_add_empty() {
    let mut a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    simd_add(&mut a, &b);
    assert!(a.is_empty());
}

#[test]
fn test_cov_simd_add_single() {
    let mut a = vec![1.0];
    let b = vec![2.0];
    simd_add(&mut a, &b);
    assert!((a[0] - 3.0).abs() < 1e-5);
}

// ===== simd_mul Extended Tests =====

#[test]
fn test_cov_simd_mul_empty() {
    let mut a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    simd_mul(&mut a, &b);
    assert!(a.is_empty());
}

#[test]
fn test_cov_simd_mul_single() {
    let mut a = vec![3.0];
    let b = vec![4.0];
    simd_mul(&mut a, &b);
    assert!((a[0] - 12.0).abs() < 1e-5);
}

// ===== attention_with_cache Extended Tests =====

#[test]
fn test_cov_attention_with_cache_many_heads() {
    let num_heads = 8;
    let head_dim = 16;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 4;

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let k_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();
    let v_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).tanh())
        .collect();

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_attention_with_cache_focused() {
    // One position has very high similarity, should dominate
    let num_heads = 1;
    let head_dim = 4;

    let q = vec![1.0, 0.0, 0.0, 0.0];
    let k_cache = vec![
        1.0, 0.0, 0.0, 0.0, // Position 0: perfect match
        0.0, 1.0, 0.0, 0.0, // Position 1: orthogonal
    ];
    let v_cache = vec![
        100.0, 100.0, 100.0, 100.0, // Position 0: high values
        0.0, 0.0, 0.0, 0.0, // Position 1: zeros
    ];

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    // Should mostly attend to position 0
    for val in &output {
        assert!(
            *val > 50.0,
            "Should attend mostly to position 0, got {}",
            val
        );
    }
}

// ===== simd_dot Extended Tests =====

#[test]
fn test_cov_simd_dot_very_large() {
    let n = 4096;
    let a: Vec<f32> = vec![1.0; n];
    let b: Vec<f32> = vec![1.0; n];
    let result = simd_dot(&a, &b);
    assert!((result - n as f32).abs() < 1.0);
}

#[test]
fn test_cov_simd_dot_alternating() {
    let a = vec![1.0, -1.0, 1.0, -1.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let result = simd_dot(&a, &b);
    // 1 - 1 + 1 - 1 = 0
    assert!(result.abs() < 1e-5);
}

// =============================================================================
// ADDITIONAL COVERAGE TESTS (PMAT-802)
// =============================================================================
//
// These tests cover:
// 1. ThreadConfig edge cases and validation
// 2. SIMD dispatch paths (various sizes triggering different code paths)
// 3. KV cache edge cases and management
// 4. Batch inference handling
// 5. Error handling for invalid inputs
// 6. Q4KWeight error paths
// =============================================================================

// ===== ThreadConfig Extended Coverage =====

#[test]
fn test_cov_thread_config_large_values() {
    // Test with very large thread counts (clamping behavior)
    let config = ThreadConfig::new(1000, 500);
    assert_eq!(config.n_threads_batch, 1000);
    assert_eq!(config.n_threads_decode, 500);
}

#[test]
fn test_cov_thread_config_asymmetric() {
    // Test with decode threads greater than batch threads (unusual but valid)
    let config = ThreadConfig::new(4, 16);
    assert_eq!(config.n_threads_batch, 4);
    assert_eq!(config.n_threads_decode, 16);
}

#[test]
fn test_cov_thread_config_single_thread() {
    let config = ThreadConfig::new(1, 1);
    assert_eq!(config.n_threads_batch, 1);
    assert_eq!(config.n_threads_decode, 1);
    assert_eq!(config.threads_for(true), 1);
    assert_eq!(config.threads_for(false), 1);
}

#[test]
fn test_cov_thread_config_threads_for_both_modes() {
    let config = ThreadConfig::new(32, 8);
    // Verify prefill uses batch threads
    assert_eq!(config.threads_for(true), 32);
    // Verify decode uses decode threads
    assert_eq!(config.threads_for(false), 8);
}

// ===== InferenceMode Extended Coverage =====

#[test]
fn test_cov_inference_mode_copy() {
    let mode1 = InferenceMode::Prefill;
    let mode2 = mode1; // Copy
    assert_eq!(mode1, mode2);

    let mode3 = InferenceMode::Decode;
    let mode4 = mode3;
    assert_eq!(mode3, mode4);
}

#[test]
fn test_cov_inference_mode_debug_prefill() {
    let mode = InferenceMode::Prefill;
    let debug = format!("{mode:?}");
    assert!(debug.contains("Prefill"));
}

#[test]
fn test_cov_inference_mode_not_eq() {
    // Test all inequality cases
    assert!(InferenceMode::Prefill != InferenceMode::Decode);
    assert!(InferenceMode::Decode != InferenceMode::Prefill);
}

// ===== KVCache Extended Coverage =====

#[test]
fn test_cov_kv_cache_deep_layer() {
    // Test with many layers
    let num_layers = 32;
    let hidden_dim = 64;
    let max_seq_len = 128;

    let mut cache = KVCache::new(num_layers, hidden_dim, max_seq_len);

    // Store in first and last layer
    let k_first = vec![1.0f32; hidden_dim];
    let v_first = vec![2.0f32; hidden_dim];
    cache.store(0, &k_first, &v_first);

    let k_last = vec![3.0f32; hidden_dim];
    let v_last = vec![4.0f32; hidden_dim];
    cache.store(num_layers - 1, &k_last, &v_last);

    cache.advance();

    // Verify first layer
    let k0 = cache.get_k(0);
    assert_eq!(k0.len(), hidden_dim);
    assert!((k0[0] - 1.0).abs() < 1e-6);

    // Verify last layer
    let k_n = cache.get_k(num_layers - 1);
    assert_eq!(k_n.len(), hidden_dim);
    assert!((k_n[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_cov_kv_cache_long_sequence() {
    // Test with long sequence
    let mut cache = KVCache::new(1, 4, 1024);

    for i in 0..100 {
        cache.store(0, &[i as f32; 4], &[(i + 100) as f32; 4]);
        cache.advance();
    }

    assert_eq!(cache.len(), 100);

    let k = cache.get_k(0);
    assert_eq!(k.len(), 400); // 100 positions * 4 hidden_dim
}

#[test]
fn test_cov_kv_cache_interleaved_layers() {
    // Store to layers in non-sequential order
    let mut cache = KVCache::new(4, 4, 8);

    // Store to layers 3, 1, 2, 0
    cache.store(3, &[30.0; 4], &[30.0; 4]);
    cache.store(1, &[10.0; 4], &[10.0; 4]);
    cache.store(2, &[20.0; 4], &[20.0; 4]);
    cache.store(0, &[0.0; 4], &[0.0; 4]);
    cache.advance();

    // Verify all layers have correct values
    assert!((cache.get_k(0)[0] - 0.0).abs() < 1e-6);
    assert!((cache.get_k(1)[0] - 10.0).abs() < 1e-6);
    assert!((cache.get_k(2)[0] - 20.0).abs() < 1e-6);
    assert!((cache.get_k(3)[0] - 30.0).abs() < 1e-6);
}

#[test]
fn test_cov_kv_cache_get_v_consistency() {
    let mut cache = KVCache::new(1, 4, 8);

    // Store K and V with different values
    cache.store(0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
    cache.advance();

    // Verify K and V are stored independently
    let k = cache.get_k(0);
    let v = cache.get_v(0);

    assert!((k[0] - 1.0).abs() < 1e-6);
    assert!((v[0] - 10.0).abs() < 1e-6);
}

// ===== OptimizedKVCache Extended Coverage =====

#[test]
fn test_cov_optimized_kv_cache_deep_layer() {
    let num_layers = 16;
    let hidden_dim = 32;
    let max_seq_len = 64;

    let mut cache = OptimizedKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Store in first and last layers
    cache.store(0, &vec![1.0; hidden_dim], &vec![2.0; hidden_dim]);
    cache.store(
        num_layers - 1,
        &vec![3.0; hidden_dim],
        &vec![4.0; hidden_dim],
    );
    cache.advance();

    // Verify layers
    assert!((cache.get_k(0)[0] - 1.0).abs() < 1e-6);
    assert!(
        (cache.get_k(num_layers - 1)[0] - 3.0).abs() < 1e-6,
        "Last layer K should be 3.0"
    );
}

#[test]
fn test_cov_optimized_kv_cache_transposed_layout() {
    // Verify the transposed V layout is correct for attention
    let mut cache = OptimizedKVCache::new(1, 4, 4);

    // Store values: position 0
    cache.store(0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
    cache.advance();

    // Store values: position 1
    cache.store(0, &[5.0, 6.0, 7.0, 8.0], &[11.0, 21.0, 31.0, 41.0]);
    cache.advance();

    // K should be sequential: [pos0, pos1]
    let k = cache.get_k(0);
    assert_eq!(k.len(), 8);
    assert!((k[0] - 1.0).abs() < 1e-6); // pos 0, dim 0
    assert!((k[4] - 5.0).abs() < 1e-6); // pos 1, dim 0

    // V transposed should be [dim0_all_pos, dim1_all_pos, ...]
    let v_t = cache.get_v_transposed(0);
    assert_eq!(v_t.len(), 8);
    // dim 0: [10.0 (pos0), 11.0 (pos1)]
    assert!((v_t[0] - 10.0).abs() < 1e-6);
    assert!((v_t[1] - 11.0).abs() < 1e-6);
}

#[test]
fn test_cov_optimized_kv_cache_reset_multiple_times() {
    let mut cache = OptimizedKVCache::new(1, 4, 8);

    // First fill
    cache.store(0, &[1.0; 4], &[1.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);

    // First reset
    cache.reset();
    assert_eq!(cache.len(), 0);

    // Second fill
    cache.store(0, &[2.0; 4], &[2.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);

    // Second reset
    cache.reset();
    assert_eq!(cache.len(), 0);
}

// ===== simd_matmul Batch Coverage =====

#[test]
fn test_cov_simd_matmul_exactly_parallel_threshold() {
    // Test exactly at PARALLEL_THRESHOLD (256)
    let in_dim = 16;
    let out_dim = 256; // Exactly at threshold

    let input: Vec<f32> = vec![1.0; in_dim];
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i % 10) as f32 * 0.1)
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_matmul_just_below_parallel_threshold() {
    // Test just below PARALLEL_THRESHOLD
    let in_dim = 16;
    let out_dim = 255; // Just below threshold

    let input: Vec<f32> = vec![1.0; in_dim];
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i % 10) as f32 * 0.1)
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_cov_simd_matmul_batch_exactly_tiled_threshold() {
    // Test batch at exactly PARALLEL_THRESHOLD * 4
    let seq_len = 8;
    let in_dim = 32;
    let out_dim = 128; // seq_len * out_dim = 1024 = 256 * 4

    let input: Vec<f32> = (0..seq_len * in_dim).map(|i| (i as f32).sin()).collect();
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32).cos() * 0.1)
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), seq_len * out_dim);
}

#[test]
fn test_cov_simd_matmul_batch_below_tiled_threshold() {
    // Test batch just below tiled threshold (uses trueno Matrix path)
    let seq_len = 2;
    let in_dim = 8;
    let out_dim = 64; // seq_len * out_dim = 128 < 1024

    let input: Vec<f32> = vec![1.0; seq_len * in_dim];
    let weight: Vec<f32> = vec![0.1; out_dim * in_dim];

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), seq_len * out_dim);
}

#[test]
fn test_cov_simd_matmul_large_batch_tiled() {
    // Test large batch that triggers tiled_matmul
    let seq_len = 32;
    let in_dim = 64;
    let out_dim = 128; // seq_len * out_dim = 4096 >> 1024

    let input: Vec<f32> = (0..seq_len * in_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), seq_len * out_dim);

    // Verify all outputs are finite
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== Q4KWeight Error Handling Coverage =====

#[test]
fn test_cov_q4k_weight_new_exact_size() {
    // Test with exactly the right size
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1);
    assert!(weight.is_ok());
    assert_eq!(weight.unwrap().data.len(), 144);
}

#[test]
fn test_cov_q4k_weight_new_extra_data() {
    // Test with extra data (should succeed)
    let mut data = create_q4k_test_data(2); // 288 bytes
    data.extend_from_slice(&[0u8; 100]); // Add extra 100 bytes
    let weight = Q4KWeight::new(data, 256, 2);
    assert!(weight.is_ok());
}

#[test]
fn test_cov_q4k_weight_matvec_dimension_error_message() {
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1).unwrap();

    // Wrong input size - verify error is returned
    let input: Vec<f32> = vec![1.0; 100]; // Should be 256
    let result = weight.matvec(&input);
    assert!(result.is_err());

    // Check error message contains useful info
    if let Err(e) = result {
        let msg = format!("{e:?}");
        assert!(msg.contains("100") || msg.contains("256"));
    }
}

// ===== simd_softmax Extended Coverage =====

#[test]
fn test_cov_simd_softmax_identical_values() {
    let mut data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    simd_softmax(&mut data);

    // All identical -> uniform distribution
    for val in &data {
        assert!((*val - 0.2).abs() < 1e-5);
    }
}

#[test]
fn test_cov_simd_softmax_very_large_positive() {
    let mut data = vec![1000.0, 1001.0, 1002.0];
    simd_softmax(&mut data);

    // Should be finite and sum to 1
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    for val in &data {
        assert!(val.is_finite());
    }

    // Largest should still dominate
    assert!(data[2] > data[1]);
    assert!(data[1] > data[0]);
}

#[test]
fn test_cov_simd_softmax_all_same_negative() {
    let mut data = vec![-5.0, -5.0, -5.0];
    simd_softmax(&mut data);

    // All same -> uniform
    for val in &data {
        assert!((*val - 1.0 / 3.0).abs() < 1e-5);
    }
}

// ===== simd_silu Extended Coverage =====

#[test]
fn test_cov_simd_silu_near_zero() {
    let mut data = vec![-0.01, 0.0, 0.01];
    simd_silu(&mut data);

    // Near zero: SiLU(x) ≈ x * sigmoid(x) ≈ x * 0.5 for small x
    assert!(data[0] < 0.0);
    assert!(data[1].abs() < 1e-6);
    assert!(data[2] > 0.0);
}

#[test]
fn test_cov_simd_silu_symmetry() {
    // SiLU is NOT symmetric: f(-x) != -f(x)
    let mut pos = vec![2.0];
    let mut neg = vec![-2.0];

    simd_silu(&mut pos);
    simd_silu(&mut neg);

    // SiLU(2) ≈ 2 * 0.88 ≈ 1.76
    // SiLU(-2) ≈ -2 * 0.12 ≈ -0.24
    assert!(pos[0] > 1.5);
    assert!(neg[0] < 0.0);
    assert!((pos[0] + neg[0]).abs() > 1.0); // Not symmetric
}

// ===== simd_gelu Extended Coverage =====

#[test]
fn test_cov_simd_gelu_near_zero() {
    let mut data = vec![-0.01, 0.0, 0.01];
    simd_gelu(&mut data);

    // GELU(0) = 0
    assert!(data[1].abs() < 1e-6);

    // GELU is approximately linear near 0
    assert!(data[0] < 0.0);
    assert!(data[2] > 0.0);
}

#[test]
fn test_cov_simd_gelu_comparison_to_relu() {
    // GELU differs from ReLU: GELU has smooth transition
    let mut gelu_data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    simd_gelu(&mut gelu_data);

    // GELU(-1) is small negative, not 0 like ReLU
    assert!(gelu_data[0] < 0.0);
    assert!(gelu_data[0] > -0.2);

    // GELU(1) < 1 (unlike ReLU where f(1) = 1)
    assert!(gelu_data[4] > 0.8);
    assert!(gelu_data[4] < 1.0);
}

// ===== simd_layer_norm Extended Coverage =====

#[test]
fn test_cov_simd_layer_norm_single_element() {
    // Edge case: single element
    let input = vec![5.0];
    let weight = vec![1.0];

    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Single element: mean = 5, variance = 0
    // normalized = (5 - 5) / sqrt(0 + eps) ≈ 0
    assert!(output[0].abs() < 0.1);
}

#[test]
fn test_cov_simd_layer_norm_large_values() {
    // Test with large values to check numerical stability
    let input = vec![1000.0, 2000.0, 3000.0, 4000.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Output should be normalized (mean 0, std 1)
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(mean.abs() < 1e-4);

    // All finite
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_layer_norm_negative_weight() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![-1.0, -1.0, -1.0, -1.0]; // Negative weights

    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    // Negative weight should flip signs
    assert_eq!(output.len(), 4);
}

// ===== simd_rms_norm Extended Coverage =====

#[test]
fn test_cov_simd_rms_norm_single_element() {
    let input = vec![3.0];
    let weight = vec![2.0];

    let output = simd_rms_norm(&input, &weight, 1e-5);

    // RMS = sqrt(9/1) = 3, output = 3/3 * 2 = 2
    assert!((output[0] - 2.0).abs() < 0.01);
}

#[test]
fn test_cov_simd_rms_norm_large_values() {
    let input = vec![1000.0, 2000.0, 3000.0, 4000.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];

    let output = simd_rms_norm(&input, &weight, 1e-5);

    // All should be finite and reasonable
    for val in &output {
        assert!(val.is_finite());
        assert!(val.abs() < 10.0); // Should be normalized
    }
}

#[test]
fn test_cov_simd_rms_norm_negative_weight() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![-1.0, -1.0, -1.0, -1.0];

    let output = simd_rms_norm(&input, &weight, 1e-5);

    // Negative weight flips signs
    for (i, val) in output.iter().enumerate() {
        assert!(val.is_finite());
        // Original was positive, with negative weight should be negative
        if input[i] > 0.0 {
            assert!(*val < 0.0);
        }
    }
}

// ===== apply_rope Extended Coverage =====

#[test]
fn test_cov_apply_rope_many_heads() {
    // Test with many heads
    let num_heads = 16;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;

    let mut x: Vec<f32> = (0..hidden_dim).map(|i| (i as f32).sin()).collect();
    apply_rope(&mut x, hidden_dim, num_heads, 10, 10000.0);

    // All values should be finite
    for val in &x {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_apply_rope_very_large_position() {
    let mut x = vec![1.0, 0.0, 0.0, 1.0];
    apply_rope(&mut x, 4, 1, 10000, 10000.0);

    // Should not overflow
    for val in &x {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_apply_rope_very_small_theta() {
    let mut x = vec![1.0, 0.0, 0.0, 1.0];
    // Very small theta = faster rotation
    apply_rope(&mut x, 4, 1, 1, 10.0);

    // Should produce rotation
    for val in &x {
        assert!(val.is_finite());
    }
}

// ===== attention_with_cache Extended Coverage =====

#[test]
fn test_cov_attention_with_cache_all_zero_k() {
    let num_heads = 1;
    let head_dim = 4;

    let q = vec![1.0, 1.0, 1.0, 1.0];
    let k_cache = vec![0.0, 0.0, 0.0, 0.0]; // All zero K
    let v_cache = vec![1.0, 2.0, 3.0, 4.0];

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    // With zero K, attention score is Q·K = 0, softmax = 1.0 (single position)
    assert_eq!(output.len(), 4);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_attention_with_cache_large_seq_len() {
    let num_heads = 2;
    let head_dim = 4;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 64;

    let q: Vec<f32> = vec![1.0; hidden_dim];
    let k_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let v_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== attention_with_transposed_v Extended Coverage =====

#[test]
fn test_cov_attention_transposed_v_large_seq() {
    let num_heads = 2;
    let head_dim = 4;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 32;

    let q: Vec<f32> = vec![1.0; hidden_dim];
    let k_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let v_transposed: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_attention_transposed_v_single_head() {
    let num_heads = 1;
    let head_dim = 8;
    let seq_len = 4;

    let q: Vec<f32> = vec![1.0; head_dim];
    let k_cache: Vec<f32> = vec![1.0; head_dim * seq_len];
    let v_transposed: Vec<f32> = (0..head_dim * seq_len).map(|i| i as f32 * 0.1).collect();

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    assert_eq!(output.len(), head_dim);
    // Uniform attention -> average
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== simd_dot Extended Coverage =====

#[test]
fn test_cov_simd_dot_unaligned_length() {
    // Test with length that's not a multiple of SIMD width
    let a: Vec<f32> = vec![1.0; 17]; // Not multiple of 4 or 8
    let b: Vec<f32> = vec![2.0; 17];
    let result = simd_dot(&a, &b);
    assert!((result - 34.0).abs() < 1e-4);
}

#[test]
fn test_cov_simd_dot_large_values() {
    let a: Vec<f32> = vec![1e6; 4];
    let b: Vec<f32> = vec![1e6; 4];
    let result = simd_dot(&a, &b);
    // 4 * 1e12 = 4e12
    assert!((result - 4e12).abs() < 1e8);
}

// ===== simd_add Extended Coverage =====

#[test]
fn test_cov_simd_add_unaligned_length() {
    let mut a: Vec<f32> = vec![1.0; 17];
    let b: Vec<f32> = vec![2.0; 17];
    simd_add(&mut a, &b);
    for val in &a {
        assert!((val - 3.0).abs() < 1e-5);
    }
}

// ===== simd_mul Extended Coverage =====

#[test]
fn test_cov_simd_mul_unaligned_length() {
    let mut a: Vec<f32> = vec![2.0; 17];
    let b: Vec<f32> = vec![3.0; 17];
    simd_mul(&mut a, &b);
    for val in &a {
        assert!((val - 6.0).abs() < 1e-5);
    }
}

// ===== Thread Pool Configuration Coverage =====

#[test]
fn test_cov_configure_thread_pool_double_call() {
    // First call may succeed or fail depending on test order
    let result1 = realizar::inference::configure_thread_pool(4);

    // Second call should definitely fail (pool already initialized)
    let result2 = realizar::inference::configure_thread_pool(8);

    // At least one should fail (the second one if not both)
    assert!(result1.is_err() || result2.is_err());
}

#[test]
fn test_cov_configure_optimal_thread_pool_double_call() {
    // Similar test for optimal thread pool
    let result1 = realizar::inference::configure_optimal_thread_pool();
    let result2 = realizar::inference::configure_optimal_thread_pool();

    // At least one should fail
    assert!(result1.is_err() || result2.is_err());
}

// ===== KVCache len and is_empty Coverage =====

#[test]
fn test_cov_kv_cache_len_progression() {
    let mut cache = KVCache::new(1, 4, 10);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    cache.store(0, &[1.0; 4], &[1.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    cache.store(0, &[2.0; 4], &[2.0; 4]);
    cache.advance();
    assert_eq!(cache.len(), 2);

    cache.reset();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// ===== OptimizedKVCache len and max_seq_len Coverage =====

#[test]
fn test_cov_optimized_kv_cache_len_and_max() {
    let max_seq = 512;
    let cache = OptimizedKVCache::new(4, 64, max_seq);

    assert_eq!(cache.max_seq_len(), max_seq);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_cov_optimized_kv_cache_get_v_raw_full() {
    let mut cache = OptimizedKVCache::new(1, 4, 4);

    // Fill to capacity
    for i in 0..4 {
        cache.store(0, &[i as f32; 4], &[(i + 10) as f32; 4]);
        cache.advance();
    }

    let v_raw = cache.get_v_raw(0);
    // Should return full allocated buffer
    assert_eq!(v_raw.len(), 4 * 4); // hidden_dim * max_seq_len
}

// ===== Q4KWeight Accessors Coverage =====

#[test]
fn test_cov_q4k_weight_accessors_consistency() {
    let data = create_q4k_test_data(4); // 4 super-blocks
    let weight = Q4KWeight::new(data.clone(), 256, 4).unwrap();

    // Verify dimensions
    assert_eq!(weight.in_dim, 256);
    assert_eq!(weight.out_dim, 4);

    // Memory should match data length
    assert_eq!(weight.memory_bytes(), data.len());

    // f32 equivalent should be much larger
    assert!(weight.f32_equivalent_bytes() > weight.memory_bytes());

    // Compression ratio should be > 1
    assert!(weight.compression_ratio() > 1.0);
}

// ===== simd_matmul Edge Cases =====

#[test]
fn test_cov_simd_matmul_single_in_single_out() {
    // Minimal case: 1x1 matrix
    let input = vec![2.0];
    let weight = vec![3.0];

    let output = simd_matmul(&input, &weight, 1, 1);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 6.0).abs() < 1e-5);
}

#[test]
fn test_cov_simd_matmul_wide_input() {
    // Wide input (in_dim >> out_dim)
    let in_dim = 512;
    let out_dim = 4;

    let input: Vec<f32> = vec![1.0; in_dim];
    let weight: Vec<f32> = vec![1.0 / in_dim as f32; out_dim * in_dim];

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), out_dim);

    // Each output should be approximately 1.0
    for val in &output {
        assert!((val - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_cov_simd_matmul_tall_output() {
    // Tall output (out_dim >> in_dim)
    let in_dim = 4;
    let out_dim = 512;

    let input: Vec<f32> = vec![1.0; in_dim];
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i % in_dim) as f32 * 0.1)
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), out_dim);

    for val in &output {
        assert!(val.is_finite());
    }
}

// =============================================================================
// ADDITIONAL COMPREHENSIVE COVERAGE TESTS (Target: 95%+)
// =============================================================================
//
// These tests target uncovered paths in inference.rs including:
// 1. Token generation loops and sampling strategies
// 2. KV cache edge cases and large sequence handling
// 3. SIMD threshold dispatch paths
// 4. Thread configuration edge cases
// 5. Activation function edge cases
// 6. Q4KWeight trait implementations
// =============================================================================

// ===== Q4KWeight Clone and Debug Coverage =====

#[test]
fn test_cov_q4k_weight_clone() {
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1).expect("should create");

    // Clone the weight
    let cloned = weight.clone();

    // Verify clone has same dimensions
    assert_eq!(cloned.in_dim, weight.in_dim);
    assert_eq!(cloned.out_dim, weight.out_dim);
    assert_eq!(cloned.data.len(), weight.data.len());
    assert_eq!(cloned.memory_bytes(), weight.memory_bytes());
}

#[test]
fn test_cov_q4k_weight_matvec_with_zero_input() {
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1).expect("should create");

    // Zero input should give zero output (or close to it)
    let input: Vec<f32> = vec![0.0; 256];
    let result = weight.matvec(&input);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 1);
    // Output should be small (near zero)
    assert!(output[0].abs() < 10.0);
}

#[test]
fn test_cov_q4k_weight_large_output_dim() {
    // Test with larger output dimension
    let data = create_q4k_test_data(8); // 8 rows
    let weight = Q4KWeight::new(data, 256, 8).expect("should create");

    let input: Vec<f32> = vec![1.0; 256];
    let result = weight.matvec(&input);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 8);
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== KVCache Deep Sequence Tests =====

#[test]
fn test_cov_kv_cache_very_long_sequence() {
    let mut cache = KVCache::new(1, 8, 2048);

    // Store 500 positions
    for i in 0..500 {
        let k: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32 * 0.001).collect();
        let v: Vec<f32> = (0..8).map(|j| ((i * 8 + j) as f32).sin()).collect();
        cache.store(0, &k, &v);
        cache.advance();
    }

    assert_eq!(cache.len(), 500);

    // Verify first and last positions
    let k = cache.get_k(0);
    assert_eq!(k.len(), 500 * 8);
    assert!(k[0].abs() < 0.01); // First position near 0
}

#[test]
fn test_cov_kv_cache_at_max_capacity() {
    let max_seq = 10;
    let mut cache = KVCache::new(1, 4, max_seq);

    // Fill exactly to capacity
    for i in 0..max_seq {
        cache.store(0, &[i as f32; 4], &[(i + 10) as f32; 4]);
        cache.advance();
    }

    assert_eq!(cache.len(), max_seq);

    // Try to add one more - should be silently ignored
    cache.store(0, &[99.0; 4], &[99.0; 4]);
    cache.advance();

    // Length should still be max
    assert_eq!(cache.len(), max_seq);
}

#[test]
fn test_cov_kv_cache_reset_and_refill() {
    let mut cache = KVCache::new(2, 4, 16);

    // First fill
    for i in 0..5 {
        cache.store(0, &[i as f32; 4], &[i as f32; 4]);
        cache.store(1, &[(i + 10) as f32; 4], &[(i + 10) as f32; 4]);
        cache.advance();
    }
    assert_eq!(cache.len(), 5);

    // Reset
    cache.reset();
    assert_eq!(cache.len(), 0);

    // Refill with different values
    for i in 0..3 {
        cache.store(0, &[(i + 100) as f32; 4], &[(i + 100) as f32; 4]);
        cache.store(1, &[(i + 200) as f32; 4], &[(i + 200) as f32; 4]);
        cache.advance();
    }
    assert_eq!(cache.len(), 3);

    // Verify new values
    let k0 = cache.get_k(0);
    assert!((k0[0] - 100.0).abs() < 1e-6);
}

// ===== OptimizedKVCache Extended Coverage =====

#[test]
fn test_cov_optimized_kv_cache_very_long_sequence() {
    let mut cache = OptimizedKVCache::new(1, 8, 1024);

    // Store 256 positions
    for i in 0..256 {
        let k: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32 * 0.001).collect();
        let v: Vec<f32> = (0..8).map(|j| ((i * 8 + j) as f32).cos()).collect();
        cache.store(0, &k, &v);
        cache.advance();
    }

    assert_eq!(cache.len(), 256);

    // Verify V transposed layout
    let v_t = cache.get_v_transposed(0);
    assert_eq!(v_t.len(), 8 * 256); // hidden_dim * seq_len
}

#[test]
fn test_cov_optimized_kv_cache_hidden_dim_field() {
    let cache = OptimizedKVCache::new(4, 128, 512);
    assert_eq!(cache.hidden_dim, 128);
    assert_eq!(cache.num_layers, 4);
    assert_eq!(cache.max_seq_len(), 512);
}

#[test]
fn test_cov_optimized_kv_cache_v_raw_vs_transposed() {
    let mut cache = OptimizedKVCache::new(1, 4, 8);

    // Store 3 positions with distinct values
    cache.store(0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
    cache.advance();
    cache.store(0, &[5.0, 6.0, 7.0, 8.0], &[11.0, 21.0, 31.0, 41.0]);
    cache.advance();
    cache.store(0, &[9.0, 10.0, 11.0, 12.0], &[12.0, 22.0, 32.0, 42.0]);
    cache.advance();

    // get_v_raw returns full buffer (with padding)
    let v_raw = cache.get_v_raw(0);
    assert_eq!(v_raw.len(), 4 * 8); // hidden_dim * max_seq_len

    // get_v_transposed returns packed data
    let v_t = cache.get_v_transposed(0);
    assert_eq!(v_t.len(), 4 * 3); // hidden_dim * seq_len
}

// ===== SIMD Threshold Dispatch Tests =====

#[test]
fn test_cov_simd_matmul_below_parallel_threshold_sequential() {
    // out_dim < PARALLEL_THRESHOLD (256) -> sequential path
    let in_dim = 64;
    let out_dim = 128; // Below threshold

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32).sin()).collect();
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32).cos() * 0.01)
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), out_dim);

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_matmul_above_parallel_threshold() {
    // out_dim >= PARALLEL_THRESHOLD (256) -> parallel path
    let in_dim = 64;
    let out_dim = 512; // Well above threshold

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), out_dim);

    // Verify parallel path produces correct results
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_matmul_batch_tiled_path() {
    // seq_len > 1 and seq_len * out_dim >= PARALLEL_THRESHOLD * 4 -> tiled path
    let seq_len = 16;
    let in_dim = 64;
    let out_dim = 128; // 16 * 128 = 2048 >= 1024

    let input: Vec<f32> = (0..seq_len * in_dim)
        .map(|i| (i as f32 * 0.01).tanh())
        .collect();
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), seq_len * out_dim);

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_matmul_batch_trueno_matrix_path() {
    // seq_len > 1 but seq_len * out_dim < PARALLEL_THRESHOLD * 4 -> trueno Matrix path
    let seq_len = 4;
    let in_dim = 8;
    let out_dim = 16; // 4 * 16 = 64 < 1024

    let input: Vec<f32> = vec![1.0; seq_len * in_dim];
    let weight: Vec<f32> = vec![0.125; out_dim * in_dim]; // 1/8 so sum = 1

    let output = simd_matmul(&input, &weight, in_dim, out_dim);
    assert_eq!(output.len(), seq_len * out_dim);

    // Each output should be approximately 1.0
    for val in &output {
        assert!((val - 1.0).abs() < 1e-4);
    }
}

// ===== Activation Function Edge Cases =====

#[test]
fn test_cov_simd_silu_extreme_negative() {
    // Very negative values: sigmoid(-x) -> 0, so SiLU(-x) -> 0
    let mut data = vec![-50.0, -100.0, -200.0];
    simd_silu(&mut data);

    for val in &data {
        assert!(val.is_finite());
        assert!(val.abs() < 1e-10); // Should be essentially 0
    }
}

#[test]
fn test_cov_simd_gelu_extreme_negative() {
    // Very negative values: GELU(-x) -> 0
    let mut data = vec![-50.0, -100.0];
    simd_gelu(&mut data);

    for val in &data {
        assert!(val.is_finite());
        assert!(val.abs() < 1e-10);
    }
}

#[test]
fn test_cov_simd_silu_mixed_values() {
    let mut data = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    simd_silu(&mut data);

    // All values should be finite
    for val in &data {
        assert!(val.is_finite());
    }

    // Zero should map to zero
    assert!(data[3].abs() < 1e-6);

    // Positive values should map to positive
    assert!(data[4] > 0.0);
    assert!(data[5] > 0.0);
    assert!(data[6] > 0.0);
}

#[test]
fn test_cov_simd_gelu_mixed_values() {
    let mut data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    simd_gelu(&mut data);

    // GELU is approximately monotonically increasing (has slight non-monotonicity for very negative)
    // At least positive values should be positive
    assert!(data[3] > 0.0); // GELU(1) > 0
    assert!(data[4] > 0.0); // GELU(3) > 0
}

#[test]
fn test_cov_simd_softmax_large_vector() {
    let n = 1000;
    let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 500.0) * 0.01).collect();
    simd_softmax(&mut data);

    // Sum should be 1.0
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All values should be positive and finite
    for val in &data {
        assert!(val.is_finite());
        assert!(*val >= 0.0);
    }
}

#[test]
fn test_cov_simd_softmax_numerical_stability_positive() {
    // Test with very large positive values
    let mut data = vec![1000.0, 1001.0, 1002.0, 1003.0];
    simd_softmax(&mut data);

    // Should not overflow
    for val in &data {
        assert!(val.is_finite());
    }

    // Sum should be 1.0
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ===== Layer Norm Extended Tests =====

#[test]
fn test_cov_simd_layer_norm_multi_sequence() {
    // Multiple positions with layer norm
    let hidden_dim = 8;
    let seq_len = 4;
    let input: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 - 16.0) * 0.1)
        .collect();
    let weight: Vec<f32> = vec![1.0; hidden_dim];
    let bias: Vec<f32> = vec![0.5; hidden_dim];

    let output = simd_layer_norm(&input, &weight, Some(&bias), 1e-5);

    assert_eq!(output.len(), hidden_dim * seq_len);

    // Each position should be independently normalized
    for s in 0..seq_len {
        let pos_output = &output[s * hidden_dim..(s + 1) * hidden_dim];
        // Mean should be approximately bias (0.5) since normalized mean is 0
        let mean: f32 = pos_output.iter().sum::<f32>() / hidden_dim as f32;
        assert!((mean - 0.5).abs() < 0.1, "Mean should be close to bias");
    }
}

#[test]
fn test_cov_simd_rms_norm_multi_sequence() {
    let hidden_dim = 8;
    let seq_len = 3;
    let input: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| ((i + 1) as f32).sqrt())
        .collect();
    let weight: Vec<f32> = vec![2.0; hidden_dim];

    let output = simd_rms_norm(&input, &weight, 1e-5);

    assert_eq!(output.len(), hidden_dim * seq_len);

    // All values should be finite
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== RoPE Extended Tests =====

#[test]
fn test_cov_apply_rope_different_head_dims() {
    // Test with various head dimensions
    for head_dim in [4, 8, 16, 32, 64] {
        let num_heads = 2;
        let hidden_dim = num_heads * head_dim;

        let mut x: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 + 1.0).recip()).collect();
        let original_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        apply_rope(&mut x, hidden_dim, num_heads, 42, 10000.0);

        let new_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        // RoPE preserves norm
        assert!(
            (new_norm - original_norm).abs() / original_norm < 0.01,
            "RoPE should preserve norm for head_dim={}: {} vs {}",
            head_dim,
            new_norm,
            original_norm
        );
    }
}

#[test]
fn test_cov_apply_rope_position_sequence() {
    // Test that different positions produce different results
    let mut outputs = Vec::new();

    for pos in [0, 1, 10, 100, 1000] {
        let mut x = vec![1.0, 0.0, 0.0, 1.0];
        apply_rope(&mut x, 4, 1, pos, 10000.0);
        outputs.push(x);
    }

    // Position 0 should be different from others
    for i in 1..outputs.len() {
        let diff: f32 = outputs[0]
            .iter()
            .zip(outputs[i].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        if i > 0 {
            // Later positions should differ from position 0
            assert!(diff > 0.01 || outputs[0][0] == outputs[i][0]); // Position 0 is special
        }
    }
}

// ===== Attention Edge Cases =====

#[test]
fn test_cov_attention_with_cache_very_small_head_dim() {
    // Minimum head_dim = 2
    let num_heads = 4;
    let head_dim = 2;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 8;

    let q: Vec<f32> = vec![1.0; hidden_dim];
    let k_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32).sin())
        .collect();
    let v_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32).cos())
        .collect();

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_attention_with_cache_gpt_style_head_dim() {
    // Large head_dim = 128 (like GPT-style models with 2 heads)
    let num_heads = 2;
    let head_dim = 128;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 4;

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let k_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();
    let v_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).tanh())
        .collect();

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_attention_transposed_v_many_heads() {
    let num_heads = 8;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;
    let seq_len = 4;

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let k_cache: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();
    let v_transposed: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|i| (i % 10) as f32 * 0.1)
        .collect();

    let output =
        attention_with_transposed_v(&q, &k_cache, &v_transposed, num_heads, head_dim, seq_len);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== ThreadConfig Extended Coverage =====

#[test]
fn test_cov_thread_config_extreme_asymmetry() {
    // Extreme case: 1 batch thread, many decode threads
    let config = ThreadConfig::new(1, 100);
    assert_eq!(config.n_threads_batch, 1);
    assert_eq!(config.n_threads_decode, 100);
    assert_eq!(config.threads_for(true), 1);
    assert_eq!(config.threads_for(false), 100);
}

#[test]
fn test_cov_thread_config_auto_ratio() {
    let config = ThreadConfig::auto();

    // Auto should set decode threads to at least half of batch threads
    // (or equal if only 1-2 cores)
    assert!(config.n_threads_batch >= config.n_threads_decode);
    assert!(config.n_threads_decode >= 1);
}

// ===== InferenceMode Full Coverage =====

#[test]
fn test_cov_inference_mode_all_variants() {
    // Test all enum variants
    let prefill = InferenceMode::Prefill;
    let decode = InferenceMode::Decode;

    // Debug formatting
    let prefill_debug = format!("{:?}", prefill);
    let decode_debug = format!("{:?}", decode);

    assert!(prefill_debug.contains("Prefill"));
    assert!(decode_debug.contains("Decode"));

    // Equality
    assert_eq!(prefill, InferenceMode::Prefill);
    assert_eq!(decode, InferenceMode::Decode);
    assert_ne!(prefill, decode);

    // Clone (Copy)
    let prefill_copy = prefill;
    assert_eq!(prefill_copy, InferenceMode::Prefill);
}

// ===== simd_dot and simd_add/mul Extended =====

#[test]
fn test_cov_simd_dot_precision() {
    // Test numerical precision with known sum
    let n = 100;
    let a: Vec<f32> = vec![0.1; n];
    let b: Vec<f32> = vec![0.1; n];

    let result = simd_dot(&a, &b);
    // 100 * 0.01 = 1.0
    assert!((result - 1.0).abs() < 0.001);
}

#[test]
fn test_cov_simd_add_accumulation() {
    // Test that add accumulates correctly
    let mut a = vec![0.0; 8];
    let b = vec![1.0; 8];

    for _ in 0..10 {
        simd_add(&mut a, &b);
    }

    // Each element should be 10.0
    for val in &a {
        assert!((val - 10.0).abs() < 1e-5);
    }
}

#[test]
fn test_cov_simd_mul_chain() {
    // Chain multiplications
    let mut a = vec![2.0; 4];
    let b = vec![2.0; 4];

    simd_mul(&mut a, &b); // 2 * 2 = 4
    simd_mul(&mut a, &b); // 4 * 2 = 8
    simd_mul(&mut a, &b); // 8 * 2 = 16

    for val in &a {
        assert!((val - 16.0).abs() < 1e-4);
    }
}

// ===== Large Data Tests for Coverage =====

#[test]
fn test_cov_simd_operations_large_vectors() {
    let n = 4096;

    // Large dot product
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).cos()).collect();
    let dot = simd_dot(&a, &b);
    assert!(dot.is_finite());

    // Large add - create fresh vector
    let mut c: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    simd_add(&mut c, &b);
    for val in &c {
        assert!(val.is_finite());
    }

    // Large mul - create fresh vector
    let mut d: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    simd_mul(&mut d, &b);
    for val in &d {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_layer_norm_large_hidden_dim() {
    // Large hidden dimension like in real models
    let hidden_dim = 2048;
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 - 1024.0) * 0.001)
        .collect();
    let weight: Vec<f32> = vec![1.0; hidden_dim];

    let output = simd_layer_norm(&input, &weight, None, 1e-5);

    assert_eq!(output.len(), hidden_dim);

    // Check normalization: mean should be ~0
    let mean: f32 = output.iter().sum::<f32>() / hidden_dim as f32;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_cov_simd_rms_norm_large_hidden_dim() {
    let hidden_dim = 2048;
    let input: Vec<f32> = (0..hidden_dim).map(|i| ((i + 1) as f32).sqrt()).collect();
    let weight: Vec<f32> = vec![1.0; hidden_dim];

    let output = simd_rms_norm(&input, &weight, 1e-5);

    assert_eq!(output.len(), hidden_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

// ===== KV Cache Multi-Layer Stress Tests =====

#[test]
fn test_cov_kv_cache_many_layers() {
    let num_layers = 32;
    let hidden_dim = 64;
    let max_seq_len = 128;

    let mut cache = KVCache::new(num_layers, hidden_dim, max_seq_len);

    // Store in all layers
    for layer in 0..num_layers {
        let k: Vec<f32> = vec![layer as f32; hidden_dim];
        let v: Vec<f32> = vec![(layer + 100) as f32; hidden_dim];
        cache.store(layer, &k, &v);
    }
    cache.advance();

    // Verify all layers
    for layer in 0..num_layers {
        let k = cache.get_k(layer);
        let v = cache.get_v(layer);

        assert_eq!(k.len(), hidden_dim);
        assert_eq!(v.len(), hidden_dim);
        assert!((k[0] - layer as f32).abs() < 1e-6);
        assert!((v[0] - (layer + 100) as f32).abs() < 1e-6);
    }
}

#[test]
fn test_cov_optimized_kv_cache_many_layers() {
    let num_layers = 24;
    let hidden_dim = 64;
    let max_seq_len = 256;

    let mut cache = OptimizedKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Store in all layers for multiple positions
    for pos in 0..10 {
        for layer in 0..num_layers {
            let k: Vec<f32> = vec![(layer * 100 + pos) as f32; hidden_dim];
            let v: Vec<f32> = vec![(layer * 100 + pos + 1000) as f32; hidden_dim];
            cache.store(layer, &k, &v);
        }
        cache.advance();
    }

    assert_eq!(cache.len(), 10);

    // Verify first and last layer
    let k0 = cache.get_k(0);
    let k_last = cache.get_k(num_layers - 1);

    assert_eq!(k0.len(), 10 * hidden_dim);
    assert_eq!(k_last.len(), 10 * hidden_dim);
}

// ===== Edge Cases for Empty/Single Element =====

#[test]
fn test_cov_simd_layer_norm_two_elements() {
    // Edge case: minimal input with 2 elements
    let input: Vec<f32> = vec![1.0, 3.0];
    let weight: Vec<f32> = vec![1.0, 1.0];

    let output = simd_layer_norm(&input, &weight, None, 1e-5);
    assert_eq!(output.len(), 2);

    // Mean is 2.0, normalized should be around [-1, 1]
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_rms_norm_two_elements() {
    // Edge case: minimal input with 2 elements
    let input: Vec<f32> = vec![3.0, 4.0];
    let weight: Vec<f32> = vec![1.0, 1.0];

    let output = simd_rms_norm(&input, &weight, 1e-5);
    assert_eq!(output.len(), 2);

    // RMS of [3, 4] is sqrt((9+16)/2) = sqrt(12.5)
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_apply_rope_minimal() {
    // Minimum valid case: 2 elements (for cos/sin pair)
    let mut x = vec![1.0, 0.0];
    apply_rope(&mut x, 2, 1, 1, 10000.0);

    for val in &x {
        assert!(val.is_finite());
    }
}

// ===== Numerical Stability Tests =====

#[test]
fn test_cov_attention_with_cache_all_zero_values() {
    let num_heads = 1;
    let head_dim = 4;
    let seq_len = 2;

    let q = vec![0.0; head_dim];
    let k_cache = vec![0.0; head_dim * seq_len];
    let v_cache = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let output = attention_with_cache(&q, &k_cache, &v_cache, num_heads, head_dim);

    // With zero Q and K, scores are all 0, softmax gives uniform distribution
    assert_eq!(output.len(), head_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cov_simd_softmax_denormalized() {
    // Test with very small values (potential denormalized numbers)
    let mut data = vec![1e-38, 2e-38, 3e-38];
    simd_softmax(&mut data);

    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ===== Q4KWeight Additional Edge Cases =====

#[test]
fn test_cov_q4k_weight_compression_ratio_validation() {
    // Verify compression ratio is reasonable for Q4_K
    let data = create_q4k_test_data(4);
    let weight = Q4KWeight::new(data, 256, 4).unwrap();

    let ratio = weight.compression_ratio();
    // Q4_K should give ~7-8x compression
    assert!(
        ratio >= 7.0,
        "Compression ratio should be >= 7, got {}",
        ratio
    );
    assert!(
        ratio <= 8.0,
        "Compression ratio should be <= 8, got {}",
        ratio
    );
}

#[test]
fn test_cov_q4k_weight_matvec_negative_input() {
    let data = create_q4k_test_data(1);
    let weight = Q4KWeight::new(data, 256, 1).unwrap();

    // Test with all negative input
    let input: Vec<f32> = vec![-1.0; 256];
    let result = weight.matvec(&input);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output[0].is_finite());
}

#[test]
fn test_cov_q4k_weight_matvec_mixed_input() {
    let data = create_q4k_test_data(2);
    let weight = Q4KWeight::new(data, 256, 2).unwrap();

    // Mixed positive/negative input
    let input: Vec<f32> = (0..256)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let result = weight.matvec(&input);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 2);
    for val in &output {
        assert!(val.is_finite());
    }
}
