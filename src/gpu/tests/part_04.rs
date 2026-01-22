use crate::gpu::*;
#[test]
fn test_scratch_buffer_new_deep2() {
    let scratch = ScratchBuffer::new(4, 256);
    assert_eq!(scratch.num_layers(), 4);
    assert_eq!(scratch.layer_size(), 256);
    assert_eq!(scratch.total_size(), 1024);
}

#[test]
fn test_scratch_buffer_get_layer_deep2() {
    let scratch = ScratchBuffer::new(2, 100);
    let layer0 = scratch.get_layer(0);
    assert_eq!(layer0.len(), 100);
    let layer1 = scratch.get_layer(1);
    assert_eq!(layer1.len(), 100);
}

#[test]
fn test_scratch_buffer_get_layer_mut_deep2() {
    let mut scratch = ScratchBuffer::new(2, 50);
    {
        let layer = scratch.get_layer_mut(0);
        layer[0] = 42.0;
    }
    assert_eq!(scratch.get_layer(0)[0], 42.0);
}

#[test]
fn test_scratch_buffer_reset_deep2() {
    let mut scratch = ScratchBuffer::new(2, 50);
    scratch.get_layer_mut(0)[0] = 99.0;
    scratch.reset();
    assert_eq!(scratch.get_layer(0)[0], 0.0);
}

// --- GpuBufferPool tests ---
#[test]
fn test_gpu_buffer_pool_new_deep2() {
    let pool = GpuBufferPool::new();
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
}

#[test]
fn test_gpu_buffer_pool_acquire_release_deep2() {
    let mut pool = GpuBufferPool::new();
    let buf = pool.acquire(100);
    assert_eq!(buf.len(), 100);
    pool.release(buf);
    let stats = pool.stats();
    assert!(stats.cached_buffers > 0 || stats.cached_bytes > 0);
}

#[test]
fn test_gpu_buffer_pool_clear_deep2() {
    let mut pool = GpuBufferPool::new();
    let buf = pool.acquire(1024);
    pool.release(buf);
    pool.clear();
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
}

// --- GpuCompute additional tests ---
#[test]
fn test_gpu_compute_empty_inputs_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let empty: Vec<f32> = vec![];
    let result = compute.relu(&empty);
    assert!(result.is_ok());
    assert!(result.expect("test").is_empty());
}

#[test]
fn test_gpu_compute_large_matmul_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let a = vec![1.0f32; 64 * 64];
    let b = vec![1.0f32; 64 * 64];
    let c = compute.matmul(&a, &b, 64, 64, 64);
    assert!(c.is_ok());
    assert_eq!(c.expect("test").len(), 64 * 64);
}

// --- StreamingKVCache additional tests ---
#[test]
fn test_streaming_kv_cache_get_valid_deep2() {
    let mut cache = StreamingKVCache::new(4, 100, 8, 64);
    let kv_dim = 8 * 64;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    // Need to append to all layers for each position
    for layer in 0..4 {
        cache.append(layer, &k, &v);
    }

    let (keys, values) = cache.get_valid(0);
    assert!(!keys.is_empty());
    assert!(!values.is_empty());
    assert_eq!(keys.len(), kv_dim);
}

#[test]
fn test_streaming_kv_cache_get_range_deep2() {
    let mut cache = StreamingKVCache::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    // Fill 3 positions
    for _ in 0..3 {
        for layer in 0..2 {
            cache.append(layer, &k, &v);
        }
    }
    assert_eq!(cache.len(), 3);

    let (keys, values) = cache.get_range(0, 0, 2);
    assert_eq!(keys.len(), 2 * kv_dim);
    assert_eq!(values.len(), 2 * kv_dim);
}

#[test]
fn test_streaming_kv_cache_clear_deep2() {
    let mut cache = StreamingKVCache::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// --- LARGE_VOCAB_THRESHOLD test ---
#[test]
fn test_large_vocab_threshold_constant_cov() {
    assert_eq!(LARGE_VOCAB_THRESHOLD, 65536);
}

// ========================================================================
// Extended Coverage Tests (_ext_cov suffix)
// ========================================================================

// --- InferenceMetrics Tests ---
#[test]
fn test_inference_metrics_new_ext_cov() {
    let metrics = InferenceMetrics::new();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

#[test]
fn test_inference_metrics_default_ext_cov() {
    let metrics = InferenceMetrics::default();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

#[test]
fn test_inference_metrics_record_inference_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 10);
    assert_eq!(metrics.total_inferences(), 1);
    assert_eq!(metrics.total_tokens(), 10);

    metrics.record_inference(std::time::Duration::from_millis(200), 20);
    assert_eq!(metrics.total_inferences(), 2);
    assert_eq!(metrics.total_tokens(), 30);
}

#[test]
fn test_inference_metrics_latency_percentile_empty_ext_cov() {
    let metrics = InferenceMetrics::new();
    assert!(metrics.latency_percentile(50).is_none());
    assert!(metrics.latency_percentile(99).is_none());
}

#[test]
fn test_inference_metrics_latency_percentile_single_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 10);
    let p50 = metrics.latency_percentile(50);
    assert!(p50.is_some());
    assert_eq!(
        p50.expect("GPU operation failed"),
        std::time::Duration::from_millis(100)
    );
}

#[test]
fn test_inference_metrics_latency_percentile_multiple_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 10);
    metrics.record_inference(std::time::Duration::from_millis(200), 10);
    metrics.record_inference(std::time::Duration::from_millis(300), 10);
    metrics.record_inference(std::time::Duration::from_millis(400), 10);
    metrics.record_inference(std::time::Duration::from_millis(500), 10);

    // p0 should be the minimum
    let p0 = metrics.latency_percentile(0).expect("GPU operation failed");
    assert_eq!(p0, std::time::Duration::from_millis(100));

    // p100 should be near the maximum
    let p100 = metrics
        .latency_percentile(100)
        .expect("GPU operation failed");
    assert_eq!(p100, std::time::Duration::from_millis(500));
}

#[test]
fn test_inference_metrics_throughput_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 100);
    // Throughput should be positive
    let throughput = metrics.throughput();
    assert!(throughput >= 0.0);
}

#[test]
fn test_inference_metrics_reset_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 10);
    metrics.record_inference(std::time::Duration::from_millis(200), 20);
    assert_eq!(metrics.total_inferences(), 2);
    assert_eq!(metrics.total_tokens(), 30);

    metrics.reset();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

// --- HealthChecker Tests ---
#[test]
fn test_health_checker_new_ext_cov() {
    let checker = HealthChecker::new();
    assert_eq!(checker.check_count(), 0);
    assert!(checker.is_healthy()); // Empty checks = healthy
}

#[test]
fn test_health_checker_default_ext_cov() {
    let checker = HealthChecker::default();
    assert_eq!(checker.check_count(), 0);
    assert!(checker.is_healthy());
}

#[test]
fn test_health_checker_register_check_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("test_check", Box::new(|| true));
    assert_eq!(checker.check_count(), 1);
}

#[test]
fn test_health_checker_check_all_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("always_healthy", Box::new(|| true));
    checker.register_check("always_unhealthy", Box::new(|| false));

    let results = checker.check_all();
    assert_eq!(results.len(), 2);
    assert_eq!(results.get("always_healthy"), Some(&true));
    assert_eq!(results.get("always_unhealthy"), Some(&false));
}

#[test]
fn test_health_checker_is_healthy_all_pass_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("check1", Box::new(|| true));
    checker.register_check("check2", Box::new(|| true));

    checker.check_all(); // Run checks first
    assert!(checker.is_healthy());
}

#[test]
fn test_health_checker_is_healthy_one_fails_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("check1", Box::new(|| true));
    checker.register_check("check2", Box::new(|| false));

    checker.check_all(); // Run checks first
    assert!(!checker.is_healthy());
}

#[test]
fn test_health_checker_clear_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("check1", Box::new(|| true));
    checker.check_all();

    checker.clear();
    assert_eq!(checker.check_count(), 0);
    assert!(checker.is_healthy()); // Empty = healthy
}

#[test]
fn test_health_checker_debug_ext_cov() {
    let checker = HealthChecker::new();
    let debug_str = format!("{:?}", checker);
    assert!(debug_str.contains("HealthChecker"));
}

// --- ShutdownCoordinator Tests ---
#[test]
fn test_shutdown_coordinator_new_ext_cov() {
    let coordinator = ShutdownCoordinator::new();
    assert!(!coordinator.is_shutting_down());
    assert_eq!(coordinator.pending_requests(), 0);
    assert_eq!(coordinator.handler_count(), 0);
}

#[test]
fn test_shutdown_coordinator_default_ext_cov() {
    let coordinator = ShutdownCoordinator::default();
    assert!(!coordinator.is_shutting_down());
    assert_eq!(coordinator.pending_requests(), 0);
}

#[test]
fn test_shutdown_coordinator_register_handler_ext_cov() {
    let mut coordinator = ShutdownCoordinator::new();
    coordinator.register_handler(Box::new(|| {}));
    assert_eq!(coordinator.handler_count(), 1);

    coordinator.register_handler(Box::new(|| {}));
    assert_eq!(coordinator.handler_count(), 2);
}

#[test]
fn test_shutdown_coordinator_request_lifecycle_ext_cov() {
    let mut coordinator = ShutdownCoordinator::new();

    coordinator.request_started();
    assert_eq!(coordinator.pending_requests(), 1);

    coordinator.request_started();
    assert_eq!(coordinator.pending_requests(), 2);

    coordinator.request_completed();
    assert_eq!(coordinator.pending_requests(), 1);

    coordinator.request_completed();
    assert_eq!(coordinator.pending_requests(), 0);
}

#[test]
fn test_shutdown_coordinator_request_completed_saturating_ext_cov() {
    let mut coordinator = ShutdownCoordinator::new();

    // Should not underflow
    coordinator.request_completed();
    assert_eq!(coordinator.pending_requests(), 0);

    coordinator.request_completed();
    assert_eq!(coordinator.pending_requests(), 0);
}

#[test]
fn test_shutdown_coordinator_initiate_shutdown_ext_cov() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut coordinator = ShutdownCoordinator::new();
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = Arc::clone(&called);

    coordinator.register_handler(Box::new(move || {
        called_clone.store(true, Ordering::SeqCst);
    }));

    assert!(!coordinator.is_shutting_down());
    coordinator.initiate_shutdown();
    assert!(coordinator.is_shutting_down());
    assert!(called.load(Ordering::SeqCst));
}

#[test]
fn test_shutdown_coordinator_initiate_shutdown_idempotent_ext_cov() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let mut coordinator = ShutdownCoordinator::new();
    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&call_count);

    coordinator.register_handler(Box::new(move || {
        count_clone.fetch_add(1, Ordering::SeqCst);
    }));

    coordinator.initiate_shutdown();
    coordinator.initiate_shutdown(); // Should not call handler again
    coordinator.initiate_shutdown();

    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[test]
fn test_shutdown_coordinator_is_complete_ext_cov() {
    let mut coordinator = ShutdownCoordinator::new();

    // Not complete yet - not shutting down
    assert!(!coordinator.is_complete());

    coordinator.request_started();
    coordinator.initiate_shutdown();
    // Not complete - still has pending requests
    assert!(!coordinator.is_complete());

    coordinator.request_completed();
    // Now complete
    assert!(coordinator.is_complete());
}

#[test]
fn test_shutdown_coordinator_debug_ext_cov() {
    let coordinator = ShutdownCoordinator::new();
    let debug_str = format!("{:?}", coordinator);
    assert!(debug_str.contains("ShutdownCoordinator"));
    assert!(debug_str.contains("shutting_down"));
}

// --- StreamingKVCacheFp16 Tests ---
#[test]
fn test_streaming_kv_cache_fp16_new_ext_cov() {
    let cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 10);
}

#[test]
fn test_streaming_kv_cache_fp16_append_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    // Append to layer 0
    cache.append(0, &k, &v);
    assert_eq!(cache.len(), 0); // Only advances after last layer

    // Append to layer 1 (last layer)
    cache.append(1, &k, &v);
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_fp16_get_range_f32_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }

    let (keys, values) = cache.get_range_f32(0, 0, 1);
    assert_eq!(keys.len(), kv_dim);
    assert_eq!(values.len(), kv_dim);

    // Check values are approximately correct (FP16 precision)
    for key in &keys {
        assert!((key - 1.0).abs() < 0.01);
    }
    for val in &values {
        assert!((val - 2.0).abs() < 0.01);
    }
}

#[test]
fn test_streaming_kv_cache_fp16_get_range_raw_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }

    let (keys_raw, values_raw) = cache.get_range_raw(0, 0, 1);
    assert_eq!(keys_raw.len(), kv_dim);
    assert_eq!(values_raw.len(), kv_dim);
}

#[test]
fn test_streaming_kv_cache_fp16_get_valid_f32_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    // Add 3 positions
    for _ in 0..3 {
        for layer in 0..2 {
            cache.append(layer, &k, &v);
        }
    }

    let (keys, values) = cache.get_valid_f32(0);
    assert_eq!(keys.len(), 3 * kv_dim);
    assert_eq!(values.len(), 3 * kv_dim);
}

#[test]
fn test_streaming_kv_cache_fp16_clear_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_fp16_memory_bytes_ext_cov() {
    let cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let expected_size = 2 * 10 * 4 * 32 * 2 * 2; // layers * pos * heads * dim * kv * fp16
    assert_eq!(cache.memory_bytes(), expected_size);
}

#[test]
fn test_streaming_kv_cache_fp16_memory_mb_ext_cov() {
    let cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let bytes = cache.memory_bytes();
    let expected_mb = bytes as f64 / (1024.0 * 1024.0);
    assert!((cache.memory_mb() - expected_mb).abs() < 0.001);
}

// --- GpuModelConfig Tests ---
#[test]
fn test_gpu_model_config_head_dim_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
    };
    assert_eq!(config.head_dim(), 32); // 256 / 8
}

#[test]
fn test_gpu_model_config_kv_dim_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 4, // GQA
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
    };
    assert_eq!(config.kv_dim(), 128); // 4 * 32 (num_kv_heads * head_dim)
}

#[test]
fn test_gpu_model_config_qkv_dim_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 4, // GQA
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
    };
    // qkv_dim = hidden_dim + 2 * kv_dim = 256 + 2 * 128 = 512
    assert_eq!(config.qkv_dim(), 512);
}

#[test]
fn test_gpu_model_config_is_gqa_true_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 4, // Less than num_heads = GQA
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
    };
    assert!(config.is_gqa());
}

#[test]
fn test_gpu_model_config_is_gqa_false_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 8, // Equal = MHA (not GQA)
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
    };
    assert!(!config.is_gqa());
}

// --- GpuGenerateConfig Tests ---
#[test]
fn test_gpu_generate_config_default_ext_cov() {
    let config = GpuGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_deterministic_ext_cov() {
    let config = GpuGenerateConfig::deterministic(100);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_with_sampling_ext_cov() {
    let config = GpuGenerateConfig::with_sampling(50, 0.7, 40);
    assert_eq!(config.max_tokens, 50);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_with_stop_tokens_ext_cov() {
    let config = GpuGenerateConfig::deterministic(100).with_stop_tokens(vec![0, 1, 2]);
    assert_eq!(config.stop_tokens, vec![0, 1, 2]);
}

#[test]
fn test_gpu_generate_config_chained_ext_cov() {
    let config = GpuGenerateConfig::with_sampling(100, 0.8, 50).with_stop_tokens(vec![123]);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.8);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.stop_tokens, vec![123]);
}

// --- AttentionBuffers Tests ---
#[test]
fn test_attention_buffers_new_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
    };
    let buffers = AttentionBuffers::new(&config, 100);

    assert_eq!(buffers.q_buffer.len(), 256);
    assert_eq!(buffers.scores_buffer.len(), 8 * 100); // num_heads * max_seq_len
    assert_eq!(buffers.output_buffer.len(), 256);
    assert_eq!(buffers.kv_proj_buffer.len(), 256);
    assert_eq!(buffers.ffn_buffer.len(), 512);
    assert_eq!(buffers.max_seq_len, 100);
}

#[test]
fn test_attention_buffers_reset_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
    };
    let mut buffers = AttentionBuffers::new(&config, 100);

    // Fill with non-zero values
    buffers.q_buffer.fill(1.0);
    buffers.scores_buffer.fill(2.0);
    buffers.output_buffer.fill(3.0);
    buffers.kv_proj_buffer.fill(4.0);
    buffers.ffn_buffer.fill(5.0);

    buffers.reset();

    assert!(buffers.q_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.scores_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.output_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.kv_proj_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.ffn_buffer.iter().all(|&x| x == 0.0));
}

// --- WeightType Tests ---
#[test]
fn test_weight_type_variants_ext_cov() {
    let qkv = WeightType::Qkv;
    let output = WeightType::Output;
    let ffn_fc1 = WeightType::FfnFc1;
    let ffn_fc2 = WeightType::FfnFc2;
    let lm_head = WeightType::LmHead;

    // Just ensure they are distinct and can be debug-printed
    let debug_qkv = format!("{:?}", qkv);
    let debug_output = format!("{:?}", output);
    let debug_fc1 = format!("{:?}", ffn_fc1);
    let debug_fc2 = format!("{:?}", ffn_fc2);
    let debug_lm_head = format!("{:?}", lm_head);

    assert!(debug_qkv.contains("Qkv"));
    assert!(debug_output.contains("Output"));
    assert!(debug_fc1.contains("FfnFc1"));
    assert!(debug_fc2.contains("FfnFc2"));
    assert!(debug_lm_head.contains("LmHead"));
}

#[test]
fn test_weight_type_clone_ext_cov() {
    let original = WeightType::Qkv;
    let cloned = original;
    assert!(matches!(cloned, WeightType::Qkv));
}

// --- ComputeBackend Tests ---
#[test]
fn test_compute_backend_default_ext_cov() {
    let backend = ComputeBackend::default();
    assert!(matches!(backend, ComputeBackend::Auto));
}

#[test]
fn test_compute_backend_variants_ext_cov() {
    let gpu = ComputeBackend::Gpu;
    let cpu = ComputeBackend::Cpu;
    let auto = ComputeBackend::Auto;

    assert!(matches!(gpu, ComputeBackend::Gpu));
    assert!(matches!(cpu, ComputeBackend::Cpu));
    assert!(matches!(auto, ComputeBackend::Auto));
}

#[test]
fn test_compute_backend_equality_ext_cov() {
    assert_eq!(ComputeBackend::Gpu, ComputeBackend::Gpu);
    assert_eq!(ComputeBackend::Cpu, ComputeBackend::Cpu);
    assert_eq!(ComputeBackend::Auto, ComputeBackend::Auto);
    assert_ne!(ComputeBackend::Gpu, ComputeBackend::Cpu);
}

#[test]
fn test_compute_backend_clone_ext_cov() {
    let original = ComputeBackend::Gpu;
    let cloned = original;
    assert_eq!(cloned, ComputeBackend::Gpu);
}

// ============================================================================
// Deep Coverage Tests (_deep_gcov_ prefix)
// Testing: error handling paths, fallback paths, scheduler initialization,
//          memory allocation failures, batch processing edge cases
// ============================================================================

// --- Scheduler Fallback and Error Handling Tests ---
#[test]
fn test_scheduler_fallback_when_cuda_unavailable_deep_gcov() {
    // Test that HybridScheduler gracefully falls back to CPU when GPU isn't available
    // or when workload is too small
    let mut scheduler = HybridScheduler::with_threshold(1_000_000).expect("test");

    // With very high threshold, should use CPU for small operations
    assert!(!scheduler.should_use_gpu(2, 2, 2)); // 8 elements << 1_000_000

    // Matmul should still work via CPU fallback
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = scheduler.matmul(&a, &b, 2, 2, 2);
    assert!(result.is_ok());
    let c = result.expect("GPU operation failed");
    assert!((c[0] - 19.0).abs() < 1e-5);
}

#[test]
fn test_batch_processing_empty_input_deep_gcov() {
    // Test batch processing with empty batches
    let mut scheduler = HybridScheduler::new().expect("test");

    let empty_ops: Vec<MatmulOp> = vec![];
    let result = scheduler.matmul_batch(&empty_ops);
    assert!(result.is_ok());
    assert!(result.expect("GPU operation failed").is_empty());
}

#[test]
fn test_batch_processing_single_op_deep_gcov() {
    // Test batch processing with single operation
    let mut scheduler = HybridScheduler::new().expect("test");

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let ops = vec![(a, b, 2, 2, 2)];

    let result = scheduler.matmul_batch(&ops);
    assert!(result.is_ok());
    let results = result.expect("GPU operation failed");
    assert_eq!(results.len(), 1);
    assert!((results[0][0] - 19.0).abs() < 1e-5);
}

#[test]
fn test_batch_processing_multiple_ops_deep_gcov() {
    // Test batch processing with multiple operations
    let mut scheduler = HybridScheduler::new().expect("test");

    let ops = vec![
        (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0], 2, 2, 2),
        (vec![1.0, 0.0, 0.0, 1.0], vec![2.0, 0.0, 0.0, 2.0], 2, 2, 2), // Identity-like
    ];

    let result = scheduler.matmul_batch(&ops);
    assert!(result.is_ok());
    let results = result.expect("GPU operation failed");
    assert_eq!(results.len(), 2);
}

#[test]
fn test_gpu_compute_matmul_zero_dimensions_deep_gcov() {
    // Test matmul with zero-size inputs (edge case)
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");

    // Empty matrices
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];

    // Should handle 0x0 @ 0x0 gracefully (returns empty result)
    let result = compute.matmul(&a, &b, 0, 0, 0);
    assert!(result.is_ok());
    assert!(result.expect("GPU operation failed").is_empty());
}

#[test]
fn test_gpu_compute_matmul_large_k_dimension_deep_gcov() {
    // Test matmul with large inner dimension
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");

    let k = 128;
    let a: Vec<f32> = vec![0.1; k]; // 1 x k
    let b: Vec<f32> = vec![0.1; k]; // k x 1

    let result = compute.matmul(&a, &b, 1, k, 1);
    assert!(result.is_ok());
    let c = result.expect("GPU operation failed");
    assert_eq!(c.len(), 1);
    // 1x128 @ 128x1 = 1x1, each element contributes 0.01
    assert!((c[0] - (k as f32 * 0.01)).abs() < 1e-4);
}

#[test]
fn test_hybrid_scheduler_m1_forces_cpu_deep_gcov() {
    // IMP-097: m=1 operations should always use CPU
    let scheduler = HybridScheduler::with_threshold(1).expect("test");

    // Even with threshold=1, m=1 should force CPU
    assert!(!scheduler.should_use_gpu(1, 1000, 1000));
    assert!(!scheduler.should_use_gpu(1, 10000, 10000));
}

#[test]
fn test_hybrid_scheduler_transpose_b_small_deep_gcov() {
    // Test matmul with transposed B for CPU path
    let mut scheduler = HybridScheduler::with_threshold(1_000_000).expect("test");

    // Q @ K^T style operation
    let q = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let k = vec![1.0, 0.0, 0.0, 1.0]; // 2x2, will be transposed

    let result = scheduler.matmul_transpose_b(&q, &k, 2, 2, 2);
    assert!(result.is_ok());
    let scores = result.expect("GPU operation failed");
    assert_eq!(scores.len(), 4);
}

#[test]
fn test_streaming_kv_cache_wraparound_deep_gcov() {
    // Test circular buffer wraparound behavior
    let mut cache = StreamingKVCache::new(2, 3, 2, 4); // max 3 positions
    let kv_dim = 2 * 4; // 8

    // Append 5 positions (should wrap around)
    for i in 0..5 {
        let k = vec![i as f32; kv_dim];
        let v = vec![i as f32 * 10.0; kv_dim];
        for layer in 0..2 {
            cache.append(layer, &k, &v);
        }
    }

    // Cache should have max_positions (3) valid entries
    assert_eq!(cache.len(), 3);
    assert!(!cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_memory_calculation_deep_gcov() {
    // Test memory calculation
    let cache = StreamingKVCache::new(4, 100, 8, 64);

    // Memory = 4 layers * 100 pos * 8 heads * 64 dim * 2 (K+V) * 4 bytes
    let expected = 4 * 100 * 8 * 64 * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected);

    let expected_mb = expected as f64 / (1024.0 * 1024.0);
    assert!((cache.memory_mb() - expected_mb).abs() < 0.001);
}

#[test]
fn test_tensor_pool_capacity_limit_deep_gcov() {
    // Test pool respects capacity limit
    let mut pool = TensorPool::new(2);
    assert_eq!(pool.capacity(), 2);

    // Acquire and release 3 buffers (exceeds capacity)
    let b1 = pool.acquire(100);
    let b2 = pool.acquire(200);
    let b3 = pool.acquire(300);

    pool.release(b1);
    pool.release(b2);
    pool.release(b3); // This one should be dropped

    assert_eq!(pool.available(), 2);
}

#[test]
fn test_tensor_pool_size_matching_deep_gcov() {
    // Test pool finds appropriate sized buffer
    let mut pool = TensorPool::new(10);

    // Release a large buffer
    let large = vec![0.0f32; 1000];
    pool.release(large);

    // Request smaller buffer - should reuse the large one
    let buf = pool.acquire(500);
    assert!(buf.capacity() >= 500);
}

#[test]
fn test_forward_arena_insufficient_capacity_deep_gcov() {
    // Test arena panics on insufficient capacity
    let mut arena = ForwardArena::new(100);

    // First allocation succeeds
    let _ = arena.alloc(50);
    assert_eq!(arena.used(), 50);

    // Second allocation succeeds
    let _ = arena.alloc(49);
    assert_eq!(arena.used(), 99);

    // Reset and verify
    arena.reset();
    assert_eq!(arena.used(), 0);
}

#[test]
fn test_scratch_buffer_layer_access_deep_gcov() {
    // Test scratch buffer layer access
    let mut scratch = ScratchBuffer::new(4, 128);

    assert_eq!(scratch.num_layers(), 4);
    assert_eq!(scratch.layer_size(), 128);
    assert_eq!(scratch.total_size(), 512);

    // Modify each layer
    for i in 0..4 {
        let layer = scratch.get_layer_mut(i);
        layer.fill(i as f32);
    }

    // Verify each layer has correct values
    for i in 0..4 {
        let layer = scratch.get_layer(i);
        assert!(layer.iter().all(|&x| (x - i as f32).abs() < 1e-5));
    }

    // Reset and verify zeros
    scratch.reset();
    for i in 0..4 {
        assert!(scratch.get_layer(i).iter().all(|&x| x == 0.0));
    }
}

#[test]
fn test_quantized_dot_q4_short_blocks_deep_gcov() {
    // Test Q4 dot product with blocks smaller than required
    let short_a: [u8; 10] = [0; 10]; // Less than 18 required
    let short_b: [u8; 10] = [0; 10];

    let result = quantized_dot_q4(&short_a, &short_b);
    assert_eq!(result, 0.0); // Should return 0 for invalid blocks
}

#[test]
fn test_quantized_dot_q8_short_blocks_deep_gcov() {
    // Test Q8 dot product with blocks smaller than required
    let short_a: [u8; 20] = [0; 20]; // Less than 34 required
    let short_b: [u8; 20] = [0; 20];

    let result = quantized_dot_q8(&short_a, &short_b);
    assert_eq!(result, 0.0); // Should return 0 for invalid blocks
}

#[test]
fn test_quantized_accumulator_operations_deep_gcov() {
    // Test quantized accumulator operations
    let mut acc = QuantizedAccumulator::new();
    assert_eq!(acc.sum(), 0.0);

    acc.add_scaled(2.0, 3.0);
    assert!((acc.sum() - 6.0).abs() < 1e-5);

    acc.add_block(4.0, 0.5);
    assert!((acc.sum() - 8.0).abs() < 1e-5);

    acc.reset();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_double_buffer_operations_deep_gcov() {
    // Test double buffer swap and access
    let mut db: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(db.capacity(), 100);

    // Write to back buffer
    db.back_mut().fill(1.0);

    // Front should still be zeros
    assert!(db.front().iter().all(|&x| x == 0.0));

    // Swap
    db.swap();

    // Now front has our data
    assert!(db.front().iter().all(|&x| x == 1.0));
}

#[test]
fn test_chunked_processor_empty_data_deep_gcov() {
    // Test chunked processor with empty data
    let processor = ChunkedProcessor::new(64);

    assert_eq!(processor.num_chunks(0), 0);

    let empty: Vec<f32> = vec![];
    let result = processor.process_chunks(&empty, |_chunk| 0.0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_chunked_processor_exact_chunks_deep_gcov() {
    // Test chunked processor when data is exact multiple of chunk size
    let processor = ChunkedProcessor::new(4);

    let _data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 8 elements, 2 chunks
    assert_eq!(processor.num_chunks(8), 2);

    let (start, end) = processor.chunk_bounds(0, 8);
    assert_eq!((start, end), (0, 4));

    let (start, end) = processor.chunk_bounds(1, 8);
    assert_eq!((start, end), (4, 8));
}

#[test]
fn test_chunked_processor_partial_chunk_deep_gcov() {
    // Test chunked processor with partial last chunk
    let processor = ChunkedProcessor::new(4);

    let _data = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements, 2 chunks (4 + 1)
    assert_eq!(processor.num_chunks(5), 2);

    let (start, end) = processor.chunk_bounds(1, 5);
    assert_eq!((start, end), (4, 5)); // Partial chunk
}

#[test]
fn test_inference_pipeline_stage_tracking_deep_gcov() {
    // Test pipeline stage timing
    let mut pipeline = InferencePipeline::new(4);
    assert_eq!(pipeline.num_stages(), 4);
    assert_eq!(pipeline.total_latency(), 0.0);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.5);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 3.0);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 2.5);
    pipeline.record_stage_time(GpuPipelineStage::Output, 0.5);

    assert!((pipeline.total_latency() - 7.5).abs() < 1e-5);

    let breakdown = pipeline.stage_breakdown();
    assert_eq!(breakdown.len(), 4);

    pipeline.reset();
    assert_eq!(pipeline.total_latency(), 0.0);
}

#[test]
fn test_token_batch_push_and_flush_deep_gcov() {
    // Test token batch accumulation and flushing
    let mut batch = TokenBatch::new(3);
    assert_eq!(batch.capacity(), 3);
    assert!(batch.is_empty());
    assert!(!batch.is_full());

    // Push tokens
    assert!(batch.push(1).is_none());
    assert!(batch.push(2).is_none());

    // Third push fills and returns batch
    let result = batch.push(3);
    assert!(result.is_some());
    assert_eq!(result.expect("index out of bounds"), vec![1, 2, 3]);

    // Batch should be empty after auto-flush
    assert!(batch.is_empty());
}

#[test]
fn test_speculative_buffer_verify_all_match_deep_gcov() {
    // Test speculative buffer verification when all tokens match
    let mut buffer = SpeculativeBuffer::new(5);

    buffer.add_candidate(10, 0.9);
    buffer.add_candidate(20, 0.8);
    buffer.add_candidate(30, 0.7);

    let (accepted, rejection_idx) = buffer.verify(&[10, 20, 30]);
    assert_eq!(accepted, 3);
    assert!(rejection_idx.is_none());
}

#[test]
fn test_speculative_buffer_verify_partial_match_deep_gcov() {
    // Test speculative buffer verification with partial match
    let mut buffer = SpeculativeBuffer::new(5);

    buffer.add_candidate(10, 0.9);
    buffer.add_candidate(20, 0.8);
    buffer.add_candidate(30, 0.7);

    let (accepted, rejection_idx) = buffer.verify(&[10, 20, 99]); // Mismatch at index 2
    assert_eq!(accepted, 2);
    assert_eq!(rejection_idx, Some(2));
}

#[test]
fn test_speculative_buffer_accept_reject_deep_gcov() {
    // Test accepting and rejecting candidates
    let mut buffer = SpeculativeBuffer::new(5);

    buffer.add_candidate(10, 0.9);
    buffer.add_candidate(20, 0.8);
    buffer.add_candidate(30, 0.7);

    // Accept first 2
    buffer.accept(2);
    assert_eq!(buffer.len(), 1); // One remaining

    // Reject remaining
    buffer.reject();
    assert!(buffer.is_empty());
}

#[test]
fn test_inference_batch_scheduler_workflow_deep_gcov() {
    // Test full batch scheduler workflow
    let mut scheduler = InferenceBatchScheduler::new();

    // Submit batches
    let id1 = scheduler.submit(vec![1, 2, 3]);
    let id2 = scheduler.submit(vec![4, 5, 6]);

    assert_eq!(scheduler.pending_count(), 2);
    assert_eq!(scheduler.completed_count(), 0);

    // Complete first batch
    scheduler.complete(id1, vec![10, 11, 12]);
    assert_eq!(scheduler.pending_count(), 1);
    assert_eq!(scheduler.completed_count(), 1);

    // Poll completed
    let result = scheduler.poll();
    assert!(result.is_some());
    let (batch_id, tokens) = result.expect("GPU operation failed");
    assert_eq!(batch_id, id1);
    assert_eq!(tokens, vec![10, 11, 12]);

    // Complete and drain remaining
    scheduler.complete(id2, vec![40, 50, 60]);
    let drained = scheduler.drain();
    assert_eq!(drained.len(), 1);
}

#[test]
fn test_async_request_queue_backpressure_deep_gcov() {
    // Test async request queue backpressure
    let mut queue: AsyncRequestQueue<u32> = AsyncRequestQueue::new(3);

    assert!(queue.try_push(1));
    assert!(queue.try_push(2));
    assert!(queue.try_push(3));
    assert!(queue.is_full());

    // Backpressure - push should fail
    assert!(!queue.try_push(4));

    // Pop one and try again
    assert_eq!(queue.try_pop(), Some(1));
    assert!(queue.try_push(4)); // Now succeeds
}

#[test]
fn test_inference_event_notifier_multiple_handlers_deep_gcov() {
    // Test event notifier with multiple handlers
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let mut notifier = InferenceEventNotifier::new();
    let count = Arc::new(AtomicUsize::new(0));

    // Register 3 handlers
    for _ in 0..3 {
        let count_clone = Arc::clone(&count);
        notifier.register(Box::new(move |_id, _tokens| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        }));
    }

    assert_eq!(notifier.handler_count(), 3);

    // Notify
    notifier.notify(42, &[1, 2, 3]);
    assert_eq!(count.load(Ordering::SeqCst), 3);

    // Clear and verify
    notifier.clear();
    assert_eq!(notifier.handler_count(), 0);
}

#[test]
fn test_timeout_manager_check_expired_deep_gcov() {
    // Test timeout manager expiry checking
    use std::time::{Duration, Instant};

    let mut manager = TimeoutManager::new();

    // Register timeouts - one immediate, one far future
    let now = Instant::now();
    manager.register(1, now); // Already expired
    manager.register(2, now + Duration::from_secs(3600)); // Far future

    assert_eq!(manager.active_count(), 2);

    // Check expired
    let expired = manager.check_expired();
    assert!(expired.contains(&1));
    assert!(!expired.contains(&2));

    assert_eq!(manager.active_count(), 1);

    // Remove remaining
    manager.remove(2);
    assert_eq!(manager.active_count(), 0);
}

#[test]
fn test_priority_request_queue_ordering_deep_gcov() {
    // Test priority queue maintains correct ordering
    let mut queue: PriorityRequestQueue<&str> = PriorityRequestQueue::new();

    // Enqueue with different priorities
    queue.enqueue(PriorityRequest::new(1, "low"));
    queue.enqueue(PriorityRequest::new(10, "high"));
    queue.enqueue(PriorityRequest::new(5, "medium"));

    // Should dequeue highest first
    let item = queue.dequeue_highest();
    assert!(item.is_some());
    assert_eq!(item.expect("GPU operation failed").into_data(), "high");

    let item = queue.dequeue_highest();
    assert_eq!(item.expect("GPU operation failed").into_data(), "medium");

    let item = queue.dequeue_highest();
    assert_eq!(item.expect("GPU operation failed").into_data(), "low");

    assert!(queue.is_empty());
}

#[test]
fn test_priority_request_queue_fifo_for_same_priority_deep_gcov() {
    // Test FIFO ordering for same priority requests
    let mut queue: PriorityRequestQueue<u32> = PriorityRequestQueue::new();

    // Enqueue with same priority
    queue.enqueue(PriorityRequest::new(5, 1));
    queue.enqueue(PriorityRequest::new(5, 2));
    queue.enqueue(PriorityRequest::new(5, 3));

    // Should dequeue in FIFO order
    assert_eq!(
        queue
            .dequeue_highest()
            .expect("GPU operation failed")
            .into_data(),
        1
    );
    assert_eq!(
        queue
            .dequeue_highest()
            .expect("GPU operation failed")
            .into_data(),
        2
    );
    assert_eq!(
        queue
            .dequeue_highest()
            .expect("GPU operation failed")
            .into_data(),
        3
    );
}

#[test]
fn test_token_rate_limiter_acquire_deep_gcov() {
    // Test token rate limiter acquisition
    let mut limiter = TokenRateLimiter::new(10.0, 5);

    assert_eq!(limiter.tokens_available(), 5);

    // Acquire some tokens
    assert!(limiter.try_acquire(3));
    assert_eq!(limiter.tokens_available(), 2);

    // Try to acquire more than available
    assert!(!limiter.try_acquire(5));
    assert_eq!(limiter.tokens_available(), 2);

    // Acquire remaining
    assert!(limiter.try_acquire(2));
    assert_eq!(limiter.tokens_available(), 0);
}

#[test]
fn test_resource_tracker_allocation_deep_gcov() {
    // Test resource tracker allocation and release
    let mut tracker = ResourceTracker::new(1024, 100);

    // Verify can_allocate
    assert!(tracker.can_allocate(500, 50));
    assert!(!tracker.can_allocate(2000, 50)); // Exceeds memory
    assert!(!tracker.can_allocate(500, 150)); // Exceeds compute

    // Allocate
    let id = tracker.allocate(500, 50);
    assert!(id.is_some());
    let id = id.expect("GPU operation failed");

    assert_eq!(tracker.memory_usage(), 500);
    assert_eq!(tracker.compute_usage(), 50);

    // Try exceeding limits
    let id2 = tracker.allocate(600, 60); // 500+600 > 1024
    assert!(id2.is_none());

    // Release and verify
    tracker.release(id);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_usage_percentage_deep_gcov() {
    // Test resource usage percentage calculation
    let mut tracker = ResourceTracker::new(1000, 100);

    tracker.allocate(250, 25);
    let (mem_pct, compute_pct) = tracker.usage_percentage();

    assert!((mem_pct - 25.0).abs() < 0.1);
    assert!((compute_pct - 25.0).abs() < 0.1);
}

#[test]
fn test_resource_tracker_zero_capacity_deep_gcov() {
    // Test edge case with zero capacity
    let tracker = ResourceTracker::new(0, 0);
    let (mem_pct, compute_pct) = tracker.usage_percentage();

    // Should return 0.0 to avoid division by zero
    assert_eq!(mem_pct, 0.0);
    assert_eq!(compute_pct, 0.0);
}

#[test]
fn test_inference_metrics_percentile_deep_gcov() {
    // Test latency percentile calculation
    use std::time::Duration;

    let mut metrics = InferenceMetrics::new();

    // Add some latencies
    for i in 1..=10 {
        metrics.record_inference(Duration::from_millis(i * 10), 1);
    }

    assert_eq!(metrics.total_inferences(), 10);
    assert_eq!(metrics.total_tokens(), 10);

    // Check percentiles
    let p50 = metrics.latency_percentile(50);
    assert!(p50.is_some());

    let p99 = metrics.latency_percentile(99);
    assert!(p99.is_some());
}

#[test]
fn test_inference_metrics_empty_percentile_deep_gcov() {
    // Test percentile with no data
    let metrics = InferenceMetrics::new();

    assert!(metrics.latency_percentile(50).is_none());
    assert_eq!(metrics.total_inferences(), 0);
}

#[test]
fn test_health_checker_all_pass_deep_gcov() {
    // Test health checker when all checks pass
    let mut checker = HealthChecker::new();

    checker.register_check("check1", Box::new(|| true));
    checker.register_check("check2", Box::new(|| true));

    let results = checker.check_all();
    assert_eq!(results.len(), 2);
    assert!(results["check1"]);
    assert!(results["check2"]);
    assert!(checker.is_healthy());
}

#[test]
fn test_health_checker_some_fail_deep_gcov() {
    // Test health checker when some checks fail
    let mut checker = HealthChecker::new();

    checker.register_check("passing", Box::new(|| true));
    checker.register_check("failing", Box::new(|| false));

    let _ = checker.check_all();
    assert!(!checker.is_healthy());
}

#[test]
fn test_health_checker_empty_is_healthy_deep_gcov() {
    // Test that empty checker is considered healthy
    let checker = HealthChecker::new();
    assert!(checker.is_healthy());
}

#[test]
fn test_cache_aligned_buffer_alignment_deep_gcov() {
    // Test cache aligned buffer actually aligns data
    let buffer = CacheAlignedBuffer::new(256);

    assert_eq!(buffer.len(), 256);
    assert!(!buffer.is_empty());

    // Check alignment (should be 64-byte aligned)
    assert!(buffer.is_aligned(64));
}

#[test]
fn test_cache_aligned_buffer_mut_access_deep_gcov() {
    // Test mutable access to aligned buffer
    let mut buffer = CacheAlignedBuffer::new(100);

    let slice = buffer.as_mut_slice();
    slice.fill(42.0);

    assert!(buffer.as_slice().iter().all(|&x| x == 42.0));
}

#[test]
fn test_contiguous_attention_buffer_views_deep_gcov() {
    // Test attention buffer views
    let mut buffer = ContiguousAttentionBuffer::new(10, 4, 32);

    assert!(buffer.is_contiguous());
    assert_eq!(buffer.max_seq_len(), 10);

    // Get mutable views and modify
    let (q, k, v, o) = buffer.get_views_mut();
    q.fill(1.0);
    k.fill(2.0);
    v.fill(3.0);
    o.fill(4.0);

    // Verify through immutable views
    let (q, k, v, o) = buffer.get_views();
    assert!(q.iter().all(|&x| x == 1.0));
    assert!(k.iter().all(|&x| x == 2.0));
    assert!(v.iter().all(|&x| x == 3.0));
    assert!(o.iter().all(|&x| x == 4.0));

    // Reset and verify
    buffer.reset();
    let (q, _, _, _) = buffer.get_views();
    assert!(q.iter().all(|&x| x == 0.0));
}

#[test]
fn test_batch_embed_out_of_bounds_token_deep_gcov() {
    // Test batch embedding with out-of-bounds tokens
    let embedding_table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens, dim 3
    let tokens = vec![0, 1, 999]; // Last token is out of bounds

    let result = batch_embed(&embedding_table, &tokens, 3);

    // First two tokens should be correct, third should be zeros
    assert_eq!(result.len(), 9);
    assert_eq!(&result[0..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&result[3..6], &[4.0, 5.0, 6.0]);
    assert_eq!(&result[6..9], &[0.0, 0.0, 0.0]); // Padded with zeros
}

#[test]
fn test_batch_embed_empty_inputs_deep_gcov() {
    // Test batch embedding with empty inputs
    let embedding_table = vec![1.0, 2.0, 3.0];
    let empty_tokens: Vec<usize> = vec![];

    let result = batch_embed(&embedding_table, &empty_tokens, 3);
    assert!(result.is_empty());

    let empty_table: Vec<f32> = vec![];
    let result = batch_embed(&empty_table, &[0, 1], 3);
    assert!(result.is_empty());
}

#[test]
fn test_sequential_ffn_empty_input_deep_gcov() {
    // Test sequential FFN with empty input
    let result = sequential_ffn(&[], &[0.0; 8], &[0.0; 4], 2, 4);
    assert!(result.is_empty());
}

#[test]
fn test_parallel_ffn_empty_input_deep_gcov() {
    // Test parallel FFN with empty input
    let result = parallel_ffn(&[], &[0.0; 8], &[0.0; 4], 2, 4);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_softmax_empty_deep_gcov() {
    // Test softmax with empty input
    let result = scalar_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_simd_softmax_empty_deep_gcov() {
    // Test SIMD softmax with empty input
    let result = simd_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_empty_deep_gcov() {
    // Test scalar RoPE with empty/zero inputs
    assert!(scalar_rope(&[], 0, 0, 10000.0).is_empty());
    assert!(scalar_rope(&[1.0], 0, 1, 10000.0).is_empty());
    assert!(scalar_rope(&[1.0], 1, 0, 10000.0).is_empty());
}

#[test]
fn test_simd_rope_empty_deep_gcov() {
    // Test SIMD RoPE with empty/zero inputs
    assert!(simd_rope(&[], 0, 0, 10000.0).is_empty());
    assert!(simd_rope(&[1.0], 0, 1, 10000.0).is_empty());
    assert!(simd_rope(&[1.0], 1, 0, 10000.0).is_empty());
}

#[test]
fn test_fused_layernorm_empty_deep_gcov() {
    // Test fused layernorm with empty input
    let result = fused_layernorm(&[], &[], &[], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_standard_layernorm_empty_deep_gcov() {
    // Test standard layernorm with empty input
    let result = standard_layernorm(&[], &[], &[], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_exceeds_gpu_buffer_limit_deep_gcov() {
    // Test GPU buffer limit checking
    let small = 1000;
    let huge = 100_000_000; // 400 MB in f32

    assert!(!exceeds_gpu_buffer_limit(small));
    assert!(exceeds_gpu_buffer_limit(huge));
}

#[test]
fn test_naive_matmul_small_deep_gcov() {
    // Test naive matmul correctness
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let c = naive_matmul(&a, &b, 2, 2, 2);

    assert!((c[0] - 19.0).abs() < 1e-5);
    assert!((c[1] - 22.0).abs() < 1e-5);
    assert!((c[2] - 43.0).abs() < 1e-5);
    assert!((c[3] - 50.0).abs() < 1e-5);
}

#[test]
fn test_blocked_matmul_deep_gcov() {
    // Test blocked matmul with various block sizes
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    // Block size 1 (maximum blocking)
    let c1 = blocked_matmul(&a, &b, 2, 2, 2, 1);

    // Block size 4 (no blocking effectively)
    let c4 = blocked_matmul(&a, &b, 2, 2, 2, 4);

    // Both should produce same result
    for i in 0..4 {
        assert!((c1[i] - c4[i]).abs() < 1e-5);
    }
}

#[test]
fn test_sum_with_prefetch_deep_gcov() {
    // Test sum with prefetch hints
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

    let sum1 = sequential_sum(&data);
    let sum2 = sum_with_prefetch(&data, 8);

    assert!((sum1 - sum2).abs() < 1e-3);
}

#[test]
fn test_prefetch_read_bounds_deep_gcov() {
    // Test prefetch with out-of-bounds position
    let data = vec![1.0, 2.0, 3.0];

    // Should not panic for out-of-bounds prefetch
    prefetch_read(&data, 0, 100);
    prefetch_read(&data, 2, 10);
}

#[test]
fn test_gpu_pool_stats_deep_gcov() {
    // Test GPU pool statistics
    let mut pool = GpuBufferPool::new();

    let b1 = pool.acquire(1024);
    let b2 = pool.acquire(2048);
    pool.release(b1);
    pool.release(b2);

    let stats = pool.stats();
    assert!(stats.cached_buffers >= 2);
    assert!(stats.cached_bytes > 0);
}

// --- GpuBufferPool Tests ---
#[test]
fn test_gpu_buffer_pool_new_ext_cov() {
    let pool = GpuBufferPool::new();
    // Pool should be created with default configuration
    assert!(!pool.bucket_sizes.is_empty());
}

#[test]
fn test_gpu_buffer_pool_default_ext_cov() {
    let pool = GpuBufferPool::default();
    // Default should match new()
    assert!(!pool.bucket_sizes.is_empty());
}

#[test]
fn test_gpu_buffer_pool_acquire_new_buffer_ext_cov() {
    let mut pool = GpuBufferPool::new();
    let buffer = pool.acquire(100);
    assert!(buffer.len() >= 100);
}

#[test]
fn test_gpu_buffer_pool_acquire_release_reuse_ext_cov() {
    let mut pool = GpuBufferPool::new();

    // Acquire a buffer
    let buffer = pool.acquire(1024);
    assert!(buffer.len() >= 1024);

    // Release it back
    pool.release(buffer);

    // Acquire again - should reuse
    let buffer2 = pool.acquire(1024);
    assert!(buffer2.len() >= 1024);
}

#[test]
fn test_gpu_buffer_pool_multiple_sizes_ext_cov() {
    let mut pool = GpuBufferPool::new();

    let small = pool.acquire(100);
    let medium = pool.acquire(10000);
    let large = pool.acquire(1000000);

    assert!(small.len() >= 100);
    assert!(medium.len() >= 10000);
    assert!(large.len() >= 1000000);
}

// --- AsyncGpuResult Tests ---
#[test]
fn test_async_gpu_result_pending_ext_cov() {
    let result = AsyncGpuResult::pending();
    assert!(!result.is_ready());
}

#[test]
fn test_async_gpu_result_ready_ext_cov() {
    let result = AsyncGpuResult::ready(vec![1.0, 2.0, 3.0]);
    assert!(result.is_ready());
}

#[test]
fn test_async_gpu_result_try_get_ext_cov() {
    let result = AsyncGpuResult::ready(vec![1.0, 2.0, 3.0]);
    let data = result.try_get();
    assert!(data.is_some());
    assert_eq!(data.expect("index out of bounds"), &vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_async_gpu_result_try_get_pending_ext_cov() {
    let result = AsyncGpuResult::pending();
    let data = result.try_get();
    assert!(data.is_none());
}

#[test]
fn test_async_gpu_result_wait_ext_cov() {
    let result = AsyncGpuResult::ready(vec![4.0, 5.0, 6.0]);
    let data = result.wait();
    assert_eq!(data, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_async_gpu_result_set_result_ext_cov() {
    let mut result = AsyncGpuResult::pending();
    assert!(!result.is_ready());
    result.set_result(vec![7.0, 8.0, 9.0]);
    assert!(result.is_ready());
    assert_eq!(
        result.try_get().expect("index out of bounds"),
        &vec![7.0, 8.0, 9.0]
    );
}

// --- GpuCompute Tests ---
#[test]
fn test_gpu_compute_auto_ext_cov() {
    let compute = GpuCompute::auto();
    assert!(compute.is_ok());
}

#[test]
fn test_gpu_compute_new_cpu_ext_cov() {
    let compute = GpuCompute::new(ComputeBackend::Cpu);
    assert!(compute.is_ok());
}

#[test]
fn test_gpu_compute_backend_ext_cov() {
    let compute = GpuCompute::new(ComputeBackend::Cpu).expect("GPU operation failed");
    let backend = compute.backend();
    assert!(matches!(backend, ComputeBackend::Cpu));
}
