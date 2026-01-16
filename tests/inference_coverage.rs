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
