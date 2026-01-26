//! Tests for inference_types.rs - targeting uncovered ~23%
//!
//! Coverage targets:
//! - InferenceScratchBuffer: from_config, reset
//! - OwnedInferenceScratchBuffer: from_config, reset
//! - ContiguousKVCache: edge cases (out-of-bounds layer, full sequence)
//! - DispatchMetrics: latency stats, percentiles, variance, CV, speedup

use super::inference_types::*;
use super::config::GGUFConfig;
use std::time::Duration;

// ============================================================================
// Helper: Create test config
// ============================================================================

fn test_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    }
}

// ============================================================================
// InferenceScratchBuffer Tests
// ============================================================================

#[test]
fn test_inference_scratch_buffer_from_config() {
    let config = test_config();
    let buf = InferenceScratchBuffer::from_config(&config);

    assert_eq!(buf.hidden.len(), 256);
    assert_eq!(buf.normed.len(), 256);
    assert_eq!(buf.qkv.len(), 256 * 3);
    assert_eq!(buf.q.len(), 256);
    assert_eq!(buf.k.len(), 256);
    assert_eq!(buf.v.len(), 256);
    assert_eq!(buf.attn_out.len(), 256);
    assert_eq!(buf.attn_proj.len(), 256);
    assert_eq!(buf.ffn_up.len(), 512);
    assert_eq!(buf.ffn_gate.len(), 512);
    assert_eq!(buf.ffn_down.len(), 256);
    assert_eq!(buf.logits.len(), 1000);
}

#[test]
fn test_inference_scratch_buffer_q8k_allocation() {
    let config = test_config();
    let buf = InferenceScratchBuffer::from_config(&config);

    // Q8K uses 256-element super-blocks
    // hidden_dim=256, so 256/256=1 scale, 256 quants
    assert_eq!(buf.q8k_hidden_scales.len(), 1);
    assert_eq!(buf.q8k_hidden_quants.len(), 256);

    // intermediate_dim=512, so 512/256=2 scales, 512 quants
    assert_eq!(buf.q8k_inter_scales.len(), 2);
    assert_eq!(buf.q8k_inter_quants.len(), 512);
}

#[test]
fn test_inference_scratch_buffer_q8k_padding() {
    // Test padding for non-multiple-of-256 dimensions
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 300,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        vocab_size: 500,
        intermediate_dim: 700,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let buf = InferenceScratchBuffer::from_config(&config);

    // 300 → ceil(300/256)*256 = 512, scales = 512/256 = 2
    assert_eq!(buf.q8k_hidden_scales.len(), 2);
    assert_eq!(buf.q8k_hidden_quants.len(), 512);

    // 700 → ceil(700/256)*256 = 768, scales = 768/256 = 3
    assert_eq!(buf.q8k_inter_scales.len(), 3);
    assert_eq!(buf.q8k_inter_quants.len(), 768);
}

#[test]
fn test_inference_scratch_buffer_reset() {
    let config = test_config();
    let mut buf = InferenceScratchBuffer::from_config(&config);

    // Fill with non-zero values
    buf.hidden.iter_mut().for_each(|x| *x = 1.0);
    buf.normed.iter_mut().for_each(|x| *x = 2.0);

    buf.reset();

    assert!(buf.hidden.iter().all(|&x| x == 0.0));
    assert!(buf.normed.iter().all(|&x| x == 0.0));
}

#[test]
fn test_inference_scratch_buffer_debug() {
    let config = test_config();
    let buf = InferenceScratchBuffer::from_config(&config);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("InferenceScratchBuffer"));
}

// ============================================================================
// OwnedInferenceScratchBuffer Tests
// ============================================================================

#[test]
fn test_owned_scratch_buffer_from_config() {
    let config = test_config();
    let buf = OwnedInferenceScratchBuffer::from_config(&config);

    // head_dim = 256/8 = 32
    // kv_dim = 4 * 32 = 128
    // qkv_dim = 256 + 2*128 = 512
    assert_eq!(buf.qkv.len(), 512);
    assert_eq!(buf.attn_out.len(), 256);
    // intermediate = hidden * 6 = 1536
    assert_eq!(buf.ffn_up.len(), 1536);
    assert_eq!(buf.ffn_gate.len(), 1536);
    assert_eq!(buf.ffn_down.len(), 256);
    assert_eq!(buf.expanded_v.len(), 256);
    assert_eq!(buf.logits.len(), 1000);
}

#[test]
fn test_owned_scratch_buffer_q8_allocation() {
    let config = test_config();
    let buf = OwnedInferenceScratchBuffer::from_config(&config);

    // num_blocks = ceil(256/32) = 8
    assert_eq!(buf.q8_scales.len(), 8);
    assert_eq!(buf.q8_quants.len(), 256);
}

#[test]
fn test_owned_scratch_buffer_reset() {
    let config = test_config();
    let mut buf = OwnedInferenceScratchBuffer::from_config(&config);

    // Fill buffers
    buf.qkv.extend([1.0, 2.0, 3.0]);
    buf.attn_out.extend([4.0, 5.0]);

    buf.reset();

    // reset() clears the vectors (len=0, capacity preserved)
    assert!(buf.qkv.is_empty());
    assert!(buf.attn_out.is_empty());
    assert!(buf.ffn_up.is_empty());
    assert!(buf.ffn_gate.is_empty());
    assert!(buf.ffn_down.is_empty());
    assert!(buf.expanded_v.is_empty());
    assert!(buf.logits.is_empty());
    assert!(buf.q8_scales.is_empty());
    assert!(buf.q8_quants.is_empty());
    assert!(buf.q8k_hidden_scales.is_empty());
    assert!(buf.q8k_hidden_quants.is_empty());
    assert!(buf.q8k_inter_scales.is_empty());
    assert!(buf.q8k_inter_quants.is_empty());
}

#[test]
fn test_owned_scratch_buffer_debug() {
    let config = test_config();
    let buf = OwnedInferenceScratchBuffer::from_config(&config);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("OwnedInferenceScratchBuffer"));
}

// ============================================================================
// ContiguousKVCache Tests
// ============================================================================

#[test]
fn test_kv_cache_new() {
    let cache = ContiguousKVCache::new(4, 64, 16);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_len(), 16);
    assert!(cache.is_contiguous());
    assert!(cache.is_cache_aligned());
}

#[test]
fn test_kv_cache_from_config() {
    let config = test_config();
    let cache = ContiguousKVCache::from_config(&config, 32);

    assert_eq!(cache.max_len(), 32);
    assert!(cache.is_cache_aligned());
}

#[test]
fn test_kv_cache_layer_stride() {
    let cache = ContiguousKVCache::new(2, 64, 8);
    // layer_stride should be aligned to 16 floats (64 bytes)
    let stride = cache.layer_stride();
    assert!(stride % 16 == 0, "stride {} should be multiple of 16", stride);
}

#[test]
fn test_kv_cache_append_and_advance() {
    let mut cache = ContiguousKVCache::new(2, 4, 8);
    let k = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![5.0, 6.0, 7.0, 8.0];

    cache.append(0, &k, &v);
    cache.advance();

    assert_eq!(cache.len(), 1);
    let k_cached = cache.get_k(0);
    assert_eq!(k_cached, &[1.0, 2.0, 3.0, 4.0]);
    let v_cached = cache.get_v(0);
    assert_eq!(v_cached, &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_kv_cache_out_of_bounds_layer() {
    let cache = ContiguousKVCache::new(2, 4, 8);

    // Layer 5 is out of bounds (only 0, 1 valid)
    let k = cache.get_k(5);
    let v = cache.get_v(5);
    assert!(k.is_empty());
    assert!(v.is_empty());
}

#[test]
fn test_kv_cache_out_of_bounds_layer_mut() {
    let mut cache = ContiguousKVCache::new(2, 4, 8);
    cache.append(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();

    // Layer 5 is out of bounds
    let k_mut = cache.get_k_mut(5);
    let v_mut = cache.get_v_mut(5);
    assert!(k_mut.is_empty());
    assert!(v_mut.is_empty());
}

#[test]
fn test_kv_cache_append_out_of_bounds_layer() {
    let mut cache = ContiguousKVCache::new(2, 4, 8);

    // Append to out-of-bounds layer should be no-op
    cache.append(5, &[1.0; 4], &[2.0; 4]);
    cache.advance();

    // Nothing should have been added
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_kv_cache_full_sequence() {
    let mut cache = ContiguousKVCache::new(1, 4, 2);

    // Fill to capacity
    cache.append(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();
    cache.append(0, &[3.0; 4], &[4.0; 4]);
    cache.advance();

    assert_eq!(cache.len(), 2);

    // Further appends should be no-op (seq_len >= max_seq_len)
    cache.append(0, &[5.0; 4], &[6.0; 4]);
    cache.advance(); // advance also no-op

    assert_eq!(cache.len(), 2);
}

#[test]
fn test_kv_cache_reset() {
    let mut cache = ContiguousKVCache::new(2, 4, 8);
    cache.append(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();

    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_kv_cache_reset_and_zero() {
    let mut cache = ContiguousKVCache::new(2, 4, 8);
    cache.append(0, &[1.0; 4], &[2.0; 4]);
    cache.advance();

    cache.reset_and_zero();
    assert!(cache.is_empty());
}

#[test]
fn test_kv_cache_memory_bytes() {
    let cache = ContiguousKVCache::new(2, 64, 16);
    let mem = cache.memory_bytes();
    // Should be non-trivial
    assert!(mem > 0);
    // Each k/v allocation: 2 layers * stride floats * 4 bytes
    // mem = (k_data.len() + v_data.len()) * 4
    assert!(mem >= 2 * 2 * 64 * 16 * 4);
}

#[test]
fn test_kv_cache_prefetch() {
    let cache = ContiguousKVCache::new(2, 4, 8);
    // prefetch should not panic for valid layers
    cache.prefetch_k(0);
    cache.prefetch_v(0);
    cache.prefetch_k(1);
    cache.prefetch_v(1);
    // Out of bounds - should be no-op (no panic)
    cache.prefetch_k(5);
    cache.prefetch_v(5);
}

#[test]
fn test_kv_cache_get_k_mut_and_modify() {
    let mut cache = ContiguousKVCache::new(2, 4, 8);
    cache.append(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
    cache.advance();

    // Modify K cache
    let k_mut = cache.get_k_mut(0);
    k_mut[0] = 100.0;

    let k = cache.get_k(0);
    assert_eq!(k[0], 100.0);
}

#[test]
fn test_kv_cache_debug() {
    let cache = ContiguousKVCache::new(2, 4, 8);
    let debug = format!("{:?}", cache);
    assert!(debug.contains("ContiguousKVCache"));
}

// ============================================================================
// DispatchMetrics Tests
// ============================================================================

#[test]
fn test_dispatch_metrics_new() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_dispatches(), 0);
    assert_eq!(m.gpu_dispatches(), 0);
    assert_eq!(m.total_dispatches(), 0);
}

#[test]
fn test_dispatch_metrics_default() {
    let m = DispatchMetrics::default();
    assert_eq!(m.cpu_dispatches(), 0);
}

#[test]
fn test_dispatch_metrics_record_dispatches() {
    let m = DispatchMetrics::new();
    m.record_cpu_dispatch();
    m.record_cpu_dispatch();
    m.record_gpu_dispatch();

    assert_eq!(m.cpu_dispatches(), 2);
    assert_eq!(m.gpu_dispatches(), 1);
    assert_eq!(m.total_dispatches(), 3);
}

#[test]
fn test_dispatch_metrics_gpu_ratio() {
    let m = DispatchMetrics::new();
    assert_eq!(m.gpu_ratio(), 0.0); // no dispatches

    m.record_cpu_dispatch();
    m.record_gpu_dispatch();
    assert!((m.gpu_ratio() - 0.5).abs() < 0.01);

    m.record_gpu_dispatch();
    m.record_gpu_dispatch();
    // 1 CPU, 3 GPU = 75%
    assert!((m.gpu_ratio() - 0.75).abs() < 0.01);
}

#[test]
fn test_dispatch_metrics_cpu_latency() {
    let m = DispatchMetrics::new();
    m.record_cpu_latency(Duration::from_micros(100));
    m.record_cpu_latency(Duration::from_micros(200));

    assert_eq!(m.cpu_latency_count(), 2);
    assert_eq!(m.cpu_latency_sum_us(), 300);
    assert!((m.cpu_latency_mean_us() - 150.0).abs() < 0.01);
    assert_eq!(m.cpu_latency_min_us(), 100);
    assert_eq!(m.cpu_latency_max_us(), 200);
}

#[test]
fn test_dispatch_metrics_gpu_latency() {
    let m = DispatchMetrics::new();
    m.record_gpu_latency(Duration::from_micros(50));
    m.record_gpu_latency(Duration::from_micros(150));

    assert_eq!(m.gpu_latency_count(), 2);
    assert_eq!(m.gpu_latency_sum_us(), 200);
    assert!((m.gpu_latency_mean_us() - 100.0).abs() < 0.01);
    assert_eq!(m.gpu_latency_min_us(), 50);
    assert_eq!(m.gpu_latency_max_us(), 150);
}

#[test]
fn test_dispatch_metrics_zero_latency_mean() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_latency_mean_us(), 0.0);
    assert_eq!(m.gpu_latency_mean_us(), 0.0);
}

#[test]
fn test_dispatch_metrics_zero_latency_min() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_latency_min_us(), 0);
    assert_eq!(m.gpu_latency_min_us(), 0);
}

#[test]
fn test_dispatch_metrics_variance_stddev() {
    let m = DispatchMetrics::new();
    // Need at least 2 samples for variance
    assert_eq!(m.cpu_latency_variance_us(), 0.0);
    assert_eq!(m.cpu_latency_stddev_us(), 0.0);

    m.record_cpu_latency(Duration::from_micros(100));
    assert_eq!(m.cpu_latency_variance_us(), 0.0); // only 1 sample

    m.record_cpu_latency(Duration::from_micros(200));
    // mean=150, variance = (100^2+200^2)/2 - 150^2 = 25000 - 22500 = 2500
    let var = m.cpu_latency_variance_us();
    assert!((var - 2500.0).abs() < 1.0, "variance was {}", var);
    let stddev = m.cpu_latency_stddev_us();
    assert!((stddev - 50.0).abs() < 0.1, "stddev was {}", stddev);
}

#[test]
fn test_dispatch_metrics_gpu_variance_stddev() {
    let m = DispatchMetrics::new();
    assert_eq!(m.gpu_latency_variance_us(), 0.0);
    assert_eq!(m.gpu_latency_stddev_us(), 0.0);

    m.record_gpu_latency(Duration::from_micros(50));
    m.record_gpu_latency(Duration::from_micros(150));
    // mean=100, variance = (50^2+150^2)/2 - 100^2 = 12500 - 10000 = 2500
    let var = m.gpu_latency_variance_us();
    assert!((var - 2500.0).abs() < 1.0, "variance was {}", var);
}

#[test]
fn test_dispatch_metrics_histogram_buckets() {
    let m = DispatchMetrics::new();
    // Bucket boundaries: [100, 500, 1000, 5000]
    // <100, 100-500, 500-1000, 1000-5000, >=5000

    m.record_cpu_latency(Duration::from_micros(50));   // bucket 0
    m.record_cpu_latency(Duration::from_micros(200));  // bucket 1
    m.record_cpu_latency(Duration::from_micros(700));  // bucket 2
    m.record_cpu_latency(Duration::from_micros(2000)); // bucket 3
    m.record_cpu_latency(Duration::from_micros(8000)); // bucket 4

    let buckets = m.cpu_latency_buckets();
    assert_eq!(buckets, [1, 1, 1, 1, 1]);
}

#[test]
fn test_dispatch_metrics_gpu_histogram_buckets() {
    let m = DispatchMetrics::new();
    m.record_gpu_latency(Duration::from_micros(10));
    m.record_gpu_latency(Duration::from_micros(10));

    let buckets = m.gpu_latency_buckets();
    assert_eq!(buckets[0], 2);
}

#[test]
fn test_dispatch_metrics_percentiles_empty() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_latency_p50_us(), 0.0);
    assert_eq!(m.cpu_latency_p95_us(), 0.0);
    assert_eq!(m.cpu_latency_p99_us(), 0.0);
    assert_eq!(m.gpu_latency_p50_us(), 0.0);
    assert_eq!(m.gpu_latency_p95_us(), 0.0);
    assert_eq!(m.gpu_latency_p99_us(), 0.0);
}

#[test]
fn test_dispatch_metrics_percentiles() {
    let m = DispatchMetrics::new();
    // All samples in bucket 0 (<100us)
    for _ in 0..100 {
        m.record_cpu_latency(Duration::from_micros(50));
    }

    let p50 = m.cpu_latency_p50_us();
    let p95 = m.cpu_latency_p95_us();
    let p99 = m.cpu_latency_p99_us();

    // All in first bucket [0-100], estimates should be in that range
    assert!(p50 >= 0.0 && p50 <= 100.0);
    assert!(p95 >= 0.0 && p95 <= 100.0);
    assert!(p99 >= 0.0 && p99 <= 100.0);
}

#[test]
fn test_dispatch_metrics_bucket_boundaries() {
    let m = DispatchMetrics::new();
    let boundaries = m.bucket_boundaries_us();
    assert_eq!(boundaries.len(), 5);
    assert_eq!(boundaries[0], "0-100");
    assert_eq!(boundaries[4], "5000+");
}

#[test]
fn test_dispatch_metrics_cv_zero_mean() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_latency_cv(), 0.0);
    assert_eq!(m.gpu_latency_cv(), 0.0);
}

#[test]
fn test_dispatch_metrics_cv() {
    let m = DispatchMetrics::new();
    m.record_cpu_latency(Duration::from_micros(100));
    m.record_cpu_latency(Duration::from_micros(200));
    // mean=150, stddev=50, CV = (50/150)*100 = 33.33%
    let cv = m.cpu_latency_cv();
    assert!((cv - 33.33).abs() < 1.0, "CV was {}", cv);
}

#[test]
fn test_dispatch_metrics_speedup_zero_gpu() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_gpu_speedup(), 0.0);
}

#[test]
fn test_dispatch_metrics_speedup() {
    let m = DispatchMetrics::new();
    m.record_cpu_latency(Duration::from_micros(200));
    m.record_gpu_latency(Duration::from_micros(50));

    let speedup = m.cpu_gpu_speedup();
    // speedup = cpu_mean / gpu_mean = 200/50 = 4.0
    assert!((speedup - 4.0).abs() < 0.01, "speedup was {}", speedup);
}

#[test]
fn test_dispatch_metrics_start_time() {
    let m = DispatchMetrics::new();
    let start = m.start_time_ms();
    assert!(start > 0);
}

#[test]
fn test_dispatch_metrics_elapsed_seconds() {
    let m = DispatchMetrics::new();
    let elapsed = m.elapsed_seconds();
    // Should be very small (just created)
    assert!(elapsed >= 0.0);
    assert!(elapsed < 1.0);
}

#[test]
fn test_dispatch_metrics_throughput_zero_elapsed() {
    let m = DispatchMetrics::new();
    // With essentially 0 elapsed time, throughput should be 0 to avoid div by zero
    let throughput = m.throughput_rps();
    assert!(throughput >= 0.0);
}

#[test]
fn test_dispatch_metrics_reset() {
    let m = DispatchMetrics::new();
    m.record_cpu_dispatch();
    m.record_gpu_dispatch();
    m.record_cpu_latency(Duration::from_micros(100));
    m.record_gpu_latency(Duration::from_micros(50));

    m.reset();

    assert_eq!(m.cpu_dispatches(), 0);
    assert_eq!(m.gpu_dispatches(), 0);
    assert_eq!(m.cpu_latency_count(), 0);
    assert_eq!(m.gpu_latency_count(), 0);
    assert_eq!(m.cpu_latency_sum_us(), 0);
    assert_eq!(m.gpu_latency_sum_us(), 0);
    assert_eq!(m.cpu_latency_max_us(), 0);
    assert_eq!(m.gpu_latency_max_us(), 0);
    // min is reset to MAX
    assert_eq!(m.cpu_latency_min_us(), 0); // getter returns 0 when count=0
    assert_eq!(m.gpu_latency_min_us(), 0);
}

#[test]
fn test_dispatch_metrics_debug() {
    let m = DispatchMetrics::new();
    let debug = format!("{:?}", m);
    assert!(debug.contains("DispatchMetrics"));
}

#[test]
fn test_dispatch_metrics_bucket_boundaries_constant() {
    assert_eq!(DispatchMetrics::BUCKET_BOUNDARIES, [100, 500, 1000, 5000]);
}
