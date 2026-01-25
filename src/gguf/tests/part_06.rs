//! GGUF Part 06: IMP-121 (Production Serving) + IMP-122 (Forward with Cache) +
//!               IMP-123 (Metrics Tracking) + IMP-129 (Latency Histogram) +
//!               IMP-124 (Forward Single Adaptive) + IMP-125 (Generate Adaptive) +
//!               PARITY-002 (Batched Prefill)
//!
//! Extracted from gguf_monolith.rs (PMAT-802)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{
    ContiguousKVCache, DispatchMetrics, GGUFConfig, OwnedQuantizedKVCache,
    OwnedQuantizedModelCached, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
};

// ========================================================================
// IMP-121: Integrate Adaptive Attention into Production Serving
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_121a_cached_sync_has_adaptive_attention() {
    // IMP-121a: OwnedQuantizedModelCachedSync should expose adaptive attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = OwnedQuantizedModelCachedSync::new(model);

    let seq_len = 32;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    // Thread-safe cached model should expose adaptive attention
    let result = cached_sync
        .adaptive_fused_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("Adaptive attention should succeed on CachedSync");

    assert_eq!(
        result.len(),
        seq_len * head_dim,
        "IMP-121a: Output should have correct shape"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_121b_cached_sync_adaptive_multihead() {
    // IMP-121b: OwnedQuantizedModelCachedSync should expose adaptive multihead attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = OwnedQuantizedModelCachedSync::new(model);

    let seq_len = 64;
    let hidden_dim = 64;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 17) as f32 * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 13) as f32 * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 11) as f32 * 0.05)
        .collect();

    // Thread-safe cached model should expose adaptive multihead attention
    let result = cached_sync
        .adaptive_multihead_attention(&q, &k, &v, seq_len)
        .expect("Adaptive multihead attention should succeed on CachedSync");

    assert_eq!(
        result.len(),
        seq_len * hidden_dim,
        "IMP-121b: Output should have shape [seq_len, hidden_dim]"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_121c_generate_with_adaptive_attention() {
    // IMP-121c: Cached model should have generate_with_adaptive_attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: Vec::new(),
            trace: false,
    };

    // Generate with adaptive attention (should use CPU for short prompts)
    let result = cached_model
        .generate_with_adaptive_attention(&prompt, &gen_config)
        .expect("generate_with_adaptive_attention should succeed");

    assert!(
        result.len() > prompt.len(),
        "IMP-121c: Generated output should include new tokens"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_121d_thread_safe_adaptive_attention() {
    // IMP-121d: Verify thread-safe access to adaptive attention
    use std::sync::Arc;
    use std::thread;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = Arc::new(OwnedQuantizedModelCachedSync::new(model));

    let seq_len = 16;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    // Spawn multiple threads accessing adaptive attention concurrently
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let model = Arc::clone(&cached_sync);
            let q = q.clone();
            let k = k.clone();
            let v = v.clone();

            thread::spawn(move || {
                model
                    .adaptive_fused_attention(&q, &k, &v, seq_len, head_dim, scale)
                    .expect("Concurrent adaptive attention should succeed")
            })
        })
        .collect();

    // All threads should complete successfully
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.join().expect("Thread should not panic");
        assert_eq!(
            result.len(),
            seq_len * head_dim,
            "IMP-121d: Thread {} should produce correct output",
            i
        );
    }
}

// ========================================================================
// IMP-122: Integrate Adaptive Attention into Forward with Cache
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_122a_adaptive_attention_with_cache() {
    // IMP-122a: Test attention_with_cache can use adaptive backend
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let hidden_dim = 64;
    let _head_dim = 16; // Used for documentation, computed as hidden_dim / num_heads
    let cache_len = 32;

    // Simulate Q for single token
    let q: Vec<f32> = (0..hidden_dim).map(|i| (i % 17) as f32 * 0.1).collect();

    // Cached K/V from previous positions
    let k_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 13) as f32 * 0.05)
        .collect();
    let v_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 11) as f32 * 0.05)
        .collect();

    // Current K/V
    let current_k: Vec<f32> = (0..hidden_dim).map(|i| (i % 7) as f32 * 0.1).collect();
    let current_v: Vec<f32> = (0..hidden_dim).map(|i| (i % 5) as f32 * 0.1).collect();

    // Test adaptive attention with cache
    let result = model
        .adaptive_attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v)
        .expect("Adaptive attention with cache should succeed");

    assert_eq!(
        result.len(),
        hidden_dim,
        "IMP-122a: Output should have shape [hidden_dim]"
    );

    // Result should have non-zero values
    let sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "IMP-122a: Output should have non-zero values");
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_122b_adaptive_matches_standard() {
    // IMP-122b: Adaptive attention with cache should match standard implementation
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let hidden_dim = 64;
    let cache_len = 16;

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i % 19) as f32 * 0.05).collect();
    let k_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 23) as f32 * 0.05)
        .collect();
    let v_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 29) as f32 * 0.05)
        .collect();
    let current_k: Vec<f32> = (0..hidden_dim).map(|i| (i % 7) as f32 * 0.1).collect();
    let current_v: Vec<f32> = (0..hidden_dim).map(|i| (i % 11) as f32 * 0.1).collect();

    // Standard attention
    let standard_result =
        model.attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Adaptive attention
    let adaptive_result = model
        .adaptive_attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v)
        .expect("Adaptive attention should succeed");

    assert_eq!(standard_result.len(), adaptive_result.len());
    for i in 0..standard_result.len() {
        let diff = (standard_result[i] - adaptive_result[i]).abs();
        assert!(
            diff < 1e-2,
            "IMP-122b: Results differ at {}: std={}, adaptive={}, diff={}",
            i,
            standard_result[i],
            adaptive_result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_122c_long_sequence_uses_gpu() {
    // IMP-122c: Long sequence should automatically use GPU path
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let hidden_dim = 128;
    let cache_len = 128; // Long cache triggers GPU

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i % 17) as f32 * 0.05).collect();
    let k_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 13) as f32 * 0.02)
        .collect();
    let v_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 11) as f32 * 0.02)
        .collect();
    let current_k: Vec<f32> = (0..hidden_dim).map(|i| (i % 7) as f32 * 0.1).collect();
    let current_v: Vec<f32> = (0..hidden_dim).map(|i| (i % 5) as f32 * 0.1).collect();

    let result = model
        .adaptive_attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v)
        .expect("Long sequence adaptive attention should succeed");

    assert_eq!(
        result.len(),
        hidden_dim,
        "IMP-122c: Long sequence should produce correct output"
    );
}

// ========================================================================
// IMP-123: Metrics Tracking for CPU vs GPU Dispatch Decisions
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_123a_dispatch_metrics_struct() {
    // IMP-123a: DispatchMetrics struct should track CPU vs GPU decisions
    let metrics = DispatchMetrics::new();

    assert_eq!(metrics.cpu_dispatches(), 0);
    assert_eq!(metrics.gpu_dispatches(), 0);
    assert_eq!(metrics.total_dispatches(), 0);
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_123b_record_dispatch_decisions() {
    // IMP-123b: Metrics should correctly record dispatch decisions
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    assert_eq!(metrics.cpu_dispatches(), 2);
    assert_eq!(metrics.gpu_dispatches(), 1);
    assert_eq!(metrics.total_dispatches(), 3);
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_123c_dispatch_ratio() {
    // IMP-123c: Should calculate GPU dispatch ratio
    let metrics = DispatchMetrics::new();

    // 3 CPU + 1 GPU = 25% GPU ratio
    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    let ratio = metrics.gpu_ratio();
    assert!(
        (ratio - 0.25).abs() < 0.01,
        "IMP-123c: GPU ratio should be ~25%, got {}",
        ratio
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_123d_thread_safe_metrics() {
    // IMP-123d: Metrics should be thread-safe
    use std::sync::Arc;
    use std::thread;

    let metrics = Arc::new(DispatchMetrics::new());
    let num_threads = 4;
    let dispatches_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let m = Arc::clone(&metrics);
            thread::spawn(move || {
                for _ in 0..dispatches_per_thread {
                    if i % 2 == 0 {
                        m.record_cpu_dispatch();
                    } else {
                        m.record_gpu_dispatch();
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // 2 threads did CPU, 2 did GPU
    assert_eq!(
        metrics.total_dispatches(),
        num_threads * dispatches_per_thread,
        "IMP-123d: Should have all dispatches recorded"
    );
    assert_eq!(
        metrics.cpu_dispatches(),
        2 * dispatches_per_thread,
        "IMP-123d: Should have correct CPU count"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        2 * dispatches_per_thread,
        "IMP-123d: Should have correct GPU count"
    );
}

// ========================================================================
// IMP-129: Dispatch Latency Histogram
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_129a_latency_histogram_struct() {
    // IMP-129a: DispatchMetrics should track latency with histogram buckets
    let metrics = DispatchMetrics::new();

    // Should have latency tracking methods
    assert_eq!(metrics.cpu_latency_count(), 0);
    assert_eq!(metrics.gpu_latency_count(), 0);
    assert!(metrics.cpu_latency_mean_us() == 0.0 || metrics.cpu_latency_mean_us().is_nan());
    assert!(metrics.gpu_latency_mean_us() == 0.0 || metrics.gpu_latency_mean_us().is_nan());
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_129b_record_latency() {
    // IMP-129b: Should record latency for CPU and GPU dispatches
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record CPU latency
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));

    // Record GPU latency
    metrics.record_gpu_latency(Duration::from_micros(1000));

    assert_eq!(metrics.cpu_latency_count(), 2);
    assert_eq!(metrics.gpu_latency_count(), 1);

    // Mean should be calculated correctly
    let cpu_mean = metrics.cpu_latency_mean_us();
    assert!(
        (cpu_mean - 150.0).abs() < 1.0,
        "IMP-129b: CPU mean should be ~150us, got {}",
        cpu_mean
    );

    let gpu_mean = metrics.gpu_latency_mean_us();
    assert!(
        (gpu_mean - 1000.0).abs() < 1.0,
        "IMP-129b: GPU mean should be ~1000us, got {}",
        gpu_mean
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_129c_histogram_buckets() {
    // IMP-129c: Should have histogram bucket counts
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record various latencies to populate buckets
    // Buckets: 0-100us, 100-500us, 500-1000us, 1000-5000us, 5000+us
    metrics.record_cpu_latency(Duration::from_micros(50)); // bucket 0
    metrics.record_cpu_latency(Duration::from_micros(200)); // bucket 1
    metrics.record_cpu_latency(Duration::from_micros(600)); // bucket 2
    metrics.record_cpu_latency(Duration::from_micros(2000)); // bucket 3
    metrics.record_cpu_latency(Duration::from_micros(10000)); // bucket 4

    let buckets = metrics.cpu_latency_buckets();
    assert_eq!(buckets.len(), 5, "IMP-129c: Should have 5 buckets");
    assert_eq!(buckets[0], 1, "IMP-129c: Bucket 0 (0-100us) should have 1");
    assert_eq!(
        buckets[1], 1,
        "IMP-129c: Bucket 1 (100-500us) should have 1"
    );
    assert_eq!(
        buckets[2], 1,
        "IMP-129c: Bucket 2 (500-1000us) should have 1"
    );
    assert_eq!(
        buckets[3], 1,
        "IMP-129c: Bucket 3 (1000-5000us) should have 1"
    );
    assert_eq!(buckets[4], 1, "IMP-129c: Bucket 4 (5000+us) should have 1");
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_129d_thread_safe_latency() {
    // IMP-129d: Latency recording should be thread-safe
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());
    let num_threads = 4;
    let recordings_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let m = Arc::clone(&metrics);
            thread::spawn(move || {
                for j in 0..recordings_per_thread {
                    let latency = Duration::from_micros((i * 100 + j) as u64);
                    if i % 2 == 0 {
                        m.record_cpu_latency(latency);
                    } else {
                        m.record_gpu_latency(latency);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // 2 threads did CPU, 2 did GPU
    assert_eq!(
        metrics.cpu_latency_count(),
        2 * recordings_per_thread,
        "IMP-129d: Should have all CPU latencies recorded"
    );
    assert_eq!(
        metrics.gpu_latency_count(),
        2 * recordings_per_thread,
        "IMP-129d: Should have all GPU latencies recorded"
    );
}

// ============================================================
// IMP-124: Wire adaptive attention into forward_single_with_cache
// RED phase: Tests written first, implementation to follow
// ============================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_124a_forward_single_with_cache_adaptive() {
    // IMP-124a: forward_single_with_cache_adaptive should exist and produce valid output
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(
        config.num_layers,
        config.hidden_dim,
        128, // max_seq_len
    );
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process first token (position 0) - cache is empty, no dispatch recorded
    let result = model.forward_single_with_cache_adaptive(0, &mut cache, 0, &metrics);
    assert!(result.is_ok(), "IMP-124a: Should produce valid output");

    let logits = result.expect("Should have logits");
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "IMP-124a: Should output vocab_size logits"
    );

    // Process second token (position 1) - cache now has entries, dispatch will be recorded
    let result2 = model.forward_single_with_cache_adaptive(1, &mut cache, 1, &metrics);
    assert!(result2.is_ok(), "IMP-124a: Second token should work");

    // Metrics should now record dispatches (from non-empty cache attention)
    assert!(
        metrics.total_dispatches() > 0,
        "IMP-124a: Should record dispatch decisions after second token"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_124b_adaptive_matches_standard() {
    // IMP-124b: Adaptive forward should match standard forward
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let mut cache1 = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let mut cache2 = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Generate 10 tokens with both methods
    for i in 0..10 {
        let token = (i % 10) as u32;
        let standard = model
            .forward_single_with_cache(token, &mut cache1, i)
            .expect("Standard forward should work");
        let adaptive = model
            .forward_single_with_cache_adaptive(token, &mut cache2, i, &metrics)
            .expect("Adaptive forward should work");

        // Outputs should match (within floating point tolerance)
        for (j, (&s, &a)) in standard.iter().zip(adaptive.iter()).enumerate() {
            assert!(
                (s - a).abs() < 1e-4,
                "IMP-124b: Output mismatch at position {} token {}: {} vs {}",
                j,
                i,
                s,
                a
            );
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_124c_tracks_metrics_per_layer() {
    // IMP-124c: Each layer should record a dispatch decision
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process 5 tokens
    for i in 0..5 {
        let _ = model.forward_single_with_cache_adaptive(i as u32, &mut cache, i, &metrics);
    }

    // With short sequences (< 64 tokens), should use CPU path exclusively
    // First token (position 0) has empty cache, no dispatch recorded
    // Tokens at positions 1-4 should each record at least one dispatch
    // Note: actual count depends on layer count in test model
    let expected_min_dispatches = 4; // At least 1 dispatch per non-first token
    assert!(
        metrics.total_dispatches() >= expected_min_dispatches,
        "IMP-124c: Should record at least {} dispatches, got {}",
        expected_min_dispatches,
        metrics.total_dispatches()
    );

    // All dispatches should be CPU (cache_len < 64)
    assert_eq!(
        metrics.cpu_dispatches(),
        metrics.total_dispatches(),
        "IMP-124c: All dispatches should be CPU for short sequences"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "IMP-124c: No GPU dispatches for short sequences"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_124d_long_cache_uses_gpu() {
    // IMP-124d: Long cache (>= 64 tokens) should trigger GPU dispatch
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(
        config.num_layers,
        config.hidden_dim,
        256, // Enough for 65+ tokens
    );
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process 70 tokens to exceed GPU threshold (64)
    for i in 0..70 {
        let _ = model.forward_single_with_cache_adaptive(i as u32, &mut cache, i, &metrics);
    }

    // After 64 tokens, GPU should start being used
    assert!(
        metrics.gpu_dispatches() > 0,
        "IMP-124d: Should have GPU dispatches for long sequences, got cpu={} gpu={}",
        metrics.cpu_dispatches(),
        metrics.gpu_dispatches()
    );

    // GPU ratio should be positive
    assert!(
        metrics.gpu_ratio() > 0.0,
        "IMP-124d: GPU ratio should be > 0 for long sequences"
    );
}

// ============================================================
// IMP-125: Generate with cache adaptive for full generation loop
// RED phase: Tests written first, implementation to follow
// ============================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_125a_generate_with_cache_adaptive() {
    // IMP-125a: generate_with_cache_adaptive should exist and produce valid tokens
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Greedy for determinism
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let prompt = vec![1u32, 2, 3]; // 3-token prompt
    let result = model.generate_with_cache_adaptive(&prompt, &gen_config, &metrics);

    assert!(result.is_ok(), "IMP-125a: Should produce valid output");
    let tokens = result.expect("Should have tokens");

    // Should have prompt + generated tokens
    assert!(
        tokens.len() >= prompt.len(),
        "IMP-125a: Output should include at least prompt tokens"
    );
    assert!(
        tokens.len() <= prompt.len() + gen_config.max_tokens,
        "IMP-125a: Output should not exceed max length"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_125b_adaptive_matches_standard() {
    // IMP-125b: Adaptive generate should match standard generate
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0, // Greedy for determinism
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let prompt = vec![1u32, 2, 3];

    // Generate with both methods
    let standard = model
        .generate_with_cache(&prompt, &gen_config)
        .expect("Standard should work");
    let adaptive = model
        .generate_with_cache_adaptive(&prompt, &gen_config, &metrics)
        .expect("Adaptive should work");

    // Token sequences should match (same sampling with temp=0)
    assert_eq!(
        standard, adaptive,
        "IMP-125b: Adaptive should produce same tokens as standard"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_125c_tracks_metrics_during_generation() {
    // IMP-125c: Should track metrics during full generation
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let prompt = vec![1u32, 2, 3, 4, 5];
    let _ = model.generate_with_cache_adaptive(&prompt, &gen_config, &metrics);

    // Should have recorded dispatches
    assert!(
        metrics.total_dispatches() > 0,
        "IMP-125c: Should track dispatch decisions during generation"
    );

    // With short context, all should be CPU
    assert_eq!(
        metrics.cpu_dispatches(),
        metrics.total_dispatches(),
        "IMP-125c: Short generation should use CPU"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_125d_long_generation_uses_gpu() {
    // IMP-125d: Long generation should eventually use GPU
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 100, // Long enough to trigger GPU
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let prompt = vec![1u32, 2, 3, 4, 5];
    let _ = model.generate_with_cache_adaptive(&prompt, &gen_config, &metrics);

    // After 64+ tokens, GPU should be used
    assert!(
        metrics.gpu_dispatches() > 0,
        "IMP-125d: Long generation should use GPU, got cpu={} gpu={}",
        metrics.cpu_dispatches(),
        metrics.gpu_dispatches()
    );

    // GPU ratio should be positive
    assert!(
        metrics.gpu_ratio() > 0.0,
        "IMP-125d: GPU ratio should be > 0 for long generation"
    );
}

// ============================================================
// PARITY-002: Batched Prompt Prefill Tests
// RED phase: Tests written first, implementation to follow
// ============================================================

/// PARITY-002a: forward_batch_with_cache should exist and process multiple tokens
#[test]
#[cfg(feature = "gpu")]
fn test_parity002a_forward_batch_with_cache_exists() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process a batch of 4 tokens at once
    let prompt = vec![1u32, 2, 3, 4];
    let result = model.forward_batch_with_cache(&prompt, &mut cache, &metrics);

    assert!(
        result.is_ok(),
        "PARITY-002a: forward_batch_with_cache should exist and produce valid output"
    );

    let logits = result.expect("Should have logits");
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "PARITY-002a: Should output vocab_size logits for last token"
    );
}

/// PARITY-002b: Batched prefill should process all tokens and populate cache
#[test]
#[cfg(feature = "gpu")]
fn test_parity002b_batched_prefill_populates_cache() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process 8 tokens at once
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let _ = model.forward_batch_with_cache(&prompt, &mut cache, &metrics);

    // Cache should have all 8 positions filled
    assert_eq!(
        cache.len(),
        8,
        "PARITY-002b: Cache should have all 8 prompt tokens after batched prefill"
    );
}

/// PARITY-002c: Batched prefill uses CPU (GPU is intentionally disabled)
///
/// FINDING (PARITY-002): GPU matmul is 6.6x SLOWER than CPU for attention
/// because attention = per-head MATVEC, and GPU overhead dominates for small matrices.
/// See IMP-600: GPU 2.7x slower for MATVEC, 57x faster for GEMM.
///
/// This test verifies CPU path is used (correct behavior for single-request inference).
#[test]
#[cfg(feature = "gpu")]
fn test_parity002c_batched_prefill_triggers_gpu() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 512);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process batch
    let prompt: Vec<u32> = (0..64).map(|i| i as u32 % 100).collect();
    let _ = model.forward_batch_with_cache(&prompt, &mut cache, &metrics);

    // PARITY-002 FINDING: CPU path is used (GPU disabled because it's slower)
    // GPU dispatches = 0 is CORRECT behavior for attention MATVEC
    // CPU dispatches should be > 0
    assert!(
        metrics.cpu_dispatches() > 0,
        "PARITY-002c: CPU should be used for attention (dispatches: {}, expected > 0)",
        metrics.cpu_dispatches()
    );
    // GPU is intentionally disabled for attention (per-head MATVEC is slower on GPU)
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "PARITY-002c: GPU should NOT be used for attention (MATVEC is slower on GPU)"
    );
}

/// PARITY-002d: Batched prefill should produce same result as sequential
#[test]
#[cfg(feature = "gpu")]
fn test_parity002d_batched_matches_sequential() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let prompt = vec![1u32, 2, 3, 4];

    // Sequential processing (baseline)
    let mut cache_seq = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics_seq = std::sync::Arc::new(DispatchMetrics::new());
    let mut logits_seq = vec![];
    for (pos, &token) in prompt.iter().enumerate() {
        logits_seq = model
            .forward_single_with_cache_adaptive(token, &mut cache_seq, pos, &metrics_seq)
            .expect("Sequential should work");
    }

    // Batched processing
    let mut cache_batch = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics_batch = std::sync::Arc::new(DispatchMetrics::new());
    let logits_batch = model
        .forward_batch_with_cache(&prompt, &mut cache_batch, &metrics_batch)
        .expect("Batched should work");

    // Results should match (within floating point tolerance)
    assert_eq!(
        logits_seq.len(),
        logits_batch.len(),
        "PARITY-002d: Logits length should match"
    );

    let max_diff: f32 = logits_seq
        .iter()
        .zip(logits_batch.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "PARITY-002d: Batched and sequential should produce same logits (max_diff: {})",
        max_diff
    );
}

/// PARITY-002e: generate_with_batched_prefill should use batched prefill then sequential gen
#[test]
#[cfg(feature = "gpu")]
fn test_parity002e_generate_with_batched_prefill() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_model_with_config(&config);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let prompt = vec![1u32, 2, 3, 4];
    let result = model.generate_with_batched_prefill(&prompt, &gen_config, &metrics);

    assert!(
        result.is_ok(),
        "PARITY-002e: generate_with_batched_prefill should exist and work"
    );

    let tokens = result.expect("Should have tokens");

    // Should have prompt + generated tokens
    assert!(
        tokens.len() >= prompt.len(),
        "PARITY-002e: Output should include at least prompt tokens"
    );
    assert!(
        tokens.len() <= prompt.len() + gen_config.max_tokens,
        "PARITY-002e: Output should not exceed prompt + max_tokens"
    );
}
