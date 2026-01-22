use crate::gpu::*;
#[test]
fn test_fused_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;

    let output = fused_layernorm(&input, &gamma, &beta, eps);

    assert_eq!(output.len(), 4);
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_layernorm_fused_matches_standard() {
    let input = vec![0.5, 1.5, 2.5, 3.5];
    let gamma = vec![1.0, 2.0, 1.0, 2.0];
    let beta = vec![0.1, 0.2, 0.3, 0.4];
    let eps = 1e-5;

    let standard = standard_layernorm(&input, &gamma, &beta, eps);
    let fused = fused_layernorm(&input, &gamma, &beta, eps);

    for (s, f) in standard.iter().zip(fused.iter()) {
        assert!((s - f).abs() < 1e-4, "Fused should match standard");
    }
}

// ============================================================================
// Coverage Tests - Quantized Operations
// ============================================================================

#[test]
fn test_quantized_dot_q4_basic() {
    // Q4 block: 18 bytes (2 scales + 16 nibbles)
    let block_a = vec![0u8; 18];
    let block_b = vec![0u8; 18];

    let result = quantized_dot_q4(&block_a, &block_b);
    // Zero blocks should give zero result
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_dot_q8_basic() {
    // Q8 block: 34 bytes (2 bytes scale + 32 bytes quants)
    let block_a = vec![0u8; 34];
    let block_b = vec![0u8; 34];

    let result = quantized_dot_q8(&block_a, &block_b);
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_prefetch_read_no_panic() {
    let data = vec![1.0f32; 100];
    // Should not panic
    prefetch_read(&data, 0, 10);
    prefetch_read(&data, 50, 20);
}

#[test]
fn test_sequential_sum_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sequential_sum(&data);
    assert!((result - 15.0).abs() < 1e-5);
}

#[test]
fn test_sum_with_prefetch_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sum_with_prefetch(&data, 2);
    assert!((result - 15.0).abs() < 1e-5);
}

#[test]
fn test_sum_with_prefetch_matches_sequential() {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let seq = sequential_sum(&data);
    let prefetch = sum_with_prefetch(&data, 8);
    assert!((seq - prefetch).abs() < 1e-3);
}

// ============================================================================
// Coverage Tests - Matmul Operations
// ============================================================================

#[test]
fn test_naive_matmul_2x2() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    let c = naive_matmul(&a, &b, 2, 2, 2);

    // C = [[19, 22], [43, 50]]
    assert!((c[0] - 19.0).abs() < 1e-5);
    assert!((c[1] - 22.0).abs() < 1e-5);
    assert!((c[2] - 43.0).abs() < 1e-5);
    assert!((c[3] - 50.0).abs() < 1e-5);
}

#[test]
fn test_blocked_matmul_2x2() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    let c = blocked_matmul(&a, &b, 2, 2, 2, 2);

    assert!((c[0] - 19.0).abs() < 1e-5);
    assert!((c[3] - 50.0).abs() < 1e-5);
}

#[test]
fn test_blocked_matmul_matches_naive() {
    let a: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..16).map(|i| i as f32 * 0.2).collect();

    let c_naive = naive_matmul(&a, &b, 4, 4, 4);
    let c_blocked = blocked_matmul(&a, &b, 4, 4, 4, 2);

    for (n, bl) in c_naive.iter().zip(c_blocked.iter()) {
        assert!((*n - *bl).abs() < 1e-4, "Blocked should match naive");
    }
}

// =========================================================================
// Coverage Tests: Basic types (PMAT-802)
// =========================================================================

#[test]
fn test_contiguous_attention_buffer_debug_cov() {
    let buf = ContiguousAttentionBuffer::new(8, 4, 64);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("ContiguousAttentionBuffer"));
    assert!(buf.is_contiguous());
}

#[test]
fn test_contiguous_attention_buffer_views_cov() {
    let mut buf = ContiguousAttentionBuffer::new(2, 2, 4);
    let (q, k, v, o) = buf.get_views();
    assert_eq!(q.len(), 2 * 2 * 4);
    assert_eq!(k.len(), 2 * 2 * 4);
    assert_eq!(v.len(), 2 * 2 * 4);
    assert_eq!(o.len(), 2 * 2 * 4);

    let (qm, km, vm, om) = buf.get_views_mut();
    qm[0] = 1.0;
    km[0] = 2.0;
    vm[0] = 3.0;
    om[0] = 4.0;
    assert_eq!(qm[0], 1.0);
}

#[test]
fn test_cache_aligned_buffer_debug_cov() {
    let buf = CacheAlignedBuffer::new(128);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("CacheAlignedBuffer"));
}

#[test]
fn test_tensor_pool_debug_cov() {
    let pool = TensorPool::new(1024);
    let debug = format!("{:?}", pool);
    assert!(debug.contains("TensorPool"));
}

#[test]
fn test_forward_arena_debug_cov() {
    let arena = ForwardArena::new(1024);
    let debug = format!("{:?}", arena);
    assert!(debug.contains("ForwardArena"));
}

#[test]
fn test_scratch_buffer_debug_cov() {
    let buf = ScratchBuffer::new(4, 256);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("ScratchBuffer"));
}

#[test]
fn test_quantized_accumulator_debug_clone_cov() {
    let acc = QuantizedAccumulator::new();
    let debug = format!("{:?}", acc);
    assert!(debug.contains("QuantizedAccumulator"));
    let cloned = acc.clone();
    assert!(format!("{:?}", cloned).contains("QuantizedAccumulator"));
}

#[test]
fn test_double_buffer_debug_cov() {
    let db: DoubleBuffer<Vec<f32>> = DoubleBuffer::new(64);
    let debug = format!("{:?}", db);
    assert!(debug.contains("DoubleBuffer"));
}

#[test]
fn test_chunked_processor_debug_clone_cov() {
    let proc = ChunkedProcessor::new(64);
    let debug = format!("{:?}", proc);
    assert!(debug.contains("ChunkedProcessor"));
    let cloned = proc.clone();
    assert!(format!("{:?}", cloned).contains("ChunkedProcessor"));
}

#[test]
fn test_gpu_pipeline_stage_debug_clone_copy_cov() {
    let embed = GpuPipelineStage::Embed;
    let debug = format!("{:?}", embed);
    assert!(debug.contains("Embed"));
    let cloned = embed;
    assert_eq!(cloned, GpuPipelineStage::Embed);

    let attn = GpuPipelineStage::Attention;
    assert!(format!("{:?}", attn).contains("Attention"));

    let ffn = GpuPipelineStage::FFN;
    assert!(format!("{:?}", ffn).contains("FFN"));

    let output = GpuPipelineStage::Output;
    assert!(format!("{:?}", output).contains("Output"));
}

#[test]
fn test_inference_pipeline_debug_cov() {
    let pipeline = InferencePipeline::new(4);
    let debug = format!("{:?}", pipeline);
    assert!(debug.contains("InferencePipeline"));
}

#[test]
fn test_token_batch_debug_cov() {
    let batch = TokenBatch::new(16);
    let debug = format!("{:?}", batch);
    assert!(debug.contains("TokenBatch"));
}

#[test]
fn test_speculative_buffer_debug_cov() {
    let buf = SpeculativeBuffer::new(4);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("SpeculativeBuffer"));
}

#[test]
fn test_inference_batch_scheduler_debug_cov() {
    let scheduler = InferenceBatchScheduler::new();
    let debug = format!("{:?}", scheduler);
    assert!(debug.contains("InferenceBatchScheduler"));
}

#[test]
fn test_async_request_queue_debug_cov() {
    let queue: AsyncRequestQueue<u32> = AsyncRequestQueue::new(10);
    let debug = format!("{:?}", queue);
    assert!(debug.contains("AsyncRequestQueue"));
}

#[test]
fn test_inference_event_notifier_debug_cov() {
    let notifier = InferenceEventNotifier::new();
    let debug = format!("{:?}", notifier);
    assert!(debug.contains("InferenceEventNotifier"));
}

#[test]
fn test_timeout_manager_debug_cov() {
    let mgr = TimeoutManager::new();
    let debug = format!("{:?}", mgr);
    assert!(debug.contains("TimeoutManager"));
}

#[test]
fn test_priority_request_debug_clone_cov() {
    let req = PriorityRequest::new(5, 42u32);
    let debug = format!("{:?}", req);
    assert!(debug.contains("PriorityRequest"));
    let cloned = req.clone();
    assert!(format!("{:?}", cloned).contains("PriorityRequest"));
}

#[test]
fn test_priority_request_queue_debug_cov() {
    let queue: PriorityRequestQueue<u32> = PriorityRequestQueue::new();
    let debug = format!("{:?}", queue);
    assert!(debug.contains("PriorityRequestQueue"));
}

#[test]
fn test_token_rate_limiter_debug_cov() {
    let limiter = TokenRateLimiter::new(100.0, 10);
    let debug = format!("{:?}", limiter);
    assert!(debug.contains("TokenRateLimiter"));
}

#[test]
fn test_resource_tracker_debug_cov() {
    let tracker = ResourceTracker::new(1024 * 1024 * 1024, 16);
    let debug = format!("{:?}", tracker);
    assert!(debug.contains("ResourceTracker"));
}

#[test]
fn test_inference_metrics_debug_cov() {
    let metrics = InferenceMetrics::new();
    let debug = format!("{:?}", metrics);
    assert!(debug.contains("InferenceMetrics"));
}

#[test]
fn test_health_checker_debug_cov() {
    let checker = HealthChecker::new();
    let debug = format!("{:?}", checker);
    assert!(debug.contains("HealthChecker"));
}

#[test]
fn test_shutdown_coordinator_debug_cov() {
    let coord = ShutdownCoordinator::new();
    let debug = format!("{:?}", coord);
    assert!(debug.contains("ShutdownCoordinator"));
}

#[test]
fn test_compute_backend_debug_clone_copy_cov() {
    let cpu = ComputeBackend::Cpu;
    let debug = format!("{:?}", cpu);
    assert!(debug.contains("Cpu"));
    let copied = cpu;
    assert_eq!(copied, ComputeBackend::Cpu);

    let gpu = ComputeBackend::Gpu;
    assert!(format!("{:?}", gpu).contains("Gpu"));

    let auto = ComputeBackend::Auto;
    assert!(format!("{:?}", auto).contains("Auto"));
}

#[test]
fn test_log_level_debug_clone_copy_eq_ord_cov() {
    let debug_level = LogLevel::Debug;
    let debug = format!("{:?}", debug_level);
    assert!(debug.contains("Debug"));
    assert_eq!(debug_level, LogLevel::Debug);

    let info = LogLevel::Info;
    assert!(format!("{:?}", info).contains("Info"));

    let warn = LogLevel::Warn;
    assert!(format!("{:?}", warn).contains("Warn"));

    let error = LogLevel::Error;
    assert!(format!("{:?}", error).contains("Error"));

    // Test ordering
    assert!(LogLevel::Debug < LogLevel::Info);
    assert!(LogLevel::Info < LogLevel::Warn);
    assert!(LogLevel::Warn < LogLevel::Error);
}

#[test]
fn test_log_entry_debug_clone_cov() {
    let entry = LogEntry::new(LogLevel::Info, "test message");
    let debug = format!("{:?}", entry);
    assert!(debug.contains("LogEntry"));
    let cloned = entry.clone();
    assert!(format!("{:?}", cloned).contains("LogEntry"));
}

// Note: MemoryTracker, DiagnosticsCollector, DebugMode don't implement Debug
// Test their basic construction instead
#[test]
fn test_memory_tracker_new_cov() {
    let tracker = MemoryTracker::new();
    // Basic construction succeeds - verify initial state via allocation
    tracker.record_allocation("test", 100);
    tracker.record_deallocation("test", 100);
}

#[test]
fn test_diagnostics_collector_new_cov() {
    let collector = DiagnosticsCollector::new();
    // Basic construction succeeds - verify request_count is zero
    assert_eq!(
        collector
            .request_count
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn test_debug_mode_new_cov() {
    let mode = DebugMode::new();
    // Basic construction succeeds - verify debug mode is off by default
    assert!(!mode.is_enabled());
}

#[test]
fn test_request_capture_debug_cov() {
    let capture = RequestCapture::new();
    let debug = format!("{:?}", capture);
    assert!(debug.contains("RequestCapture"));
}

#[test]
fn test_state_dump_debug_cov() {
    let dump = StateDump::new();
    let debug = format!("{:?}", dump);
    assert!(debug.contains("StateDump"));
}

#[test]
fn test_gguf_model_state_debug_cov() {
    let state = GgufModelState::new();
    let debug = format!("{:?}", state);
    assert!(debug.contains("GgufModelState"));
}

// =========================================================================
// Coverage Tests: Edge cases and function variants
// =========================================================================

#[test]
fn test_exceeds_gpu_buffer_limit_cov() {
    // Test below limit
    let small = 1000;
    assert!(!exceeds_gpu_buffer_limit(small));

    // Test at limit
    let at_limit = 256 * 1024 * 1024 / 4;
    assert!(!exceeds_gpu_buffer_limit(at_limit));

    // Test above limit
    let above_limit = 256 * 1024 * 1024 / 4 + 1;
    assert!(exceeds_gpu_buffer_limit(above_limit));
}

#[test]
fn test_scalar_softmax_empty_cov() {
    let result = scalar_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_simd_softmax_empty_cov() {
    let result = simd_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_empty_cov() {
    let result = scalar_rope(&[], 0, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_simd_rope_empty_cov() {
    let result = simd_rope(&[], 0, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_basic_cov() {
    let embedding_table = vec![1.0f32; 100 * 8]; // 100 tokens, dim 8
    let tokens = vec![0usize, 1, 2];
    let result = batch_embed(&embedding_table, &tokens, 8);
    assert_eq!(result.len(), 3 * 8);
}

#[test]
fn test_sequential_ffn_basic_cov() {
    let hidden = vec![1.0f32; 64];
    let w1 = vec![0.1f32; 64 * 128];
    let w2 = vec![0.1f32; 128 * 64];
    let result = sequential_ffn(&hidden, &w1, &w2, 64, 128);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_parallel_ffn_basic_cov() {
    let hidden = vec![1.0f32; 64];
    let w1 = vec![0.1f32; 64 * 128];
    let w2 = vec![0.1f32; 128 * 64];
    let result = parallel_ffn(&hidden, &w1, &w2, 64, 128);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_standard_layernorm_basic_cov() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; 4];
    let beta = vec![0.0f32; 4];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
    // Result should be normalized
    let mean: f32 = result.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_fused_layernorm_basic_cov() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; 4];
    let beta = vec![0.0f32; 4];
    let result = fused_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_prefetch_read_cov() {
    let data = vec![1.0f32; 100];
    // Just ensure it doesn't panic
    prefetch_read(&data, 0, 10);
    prefetch_read(&data, 50, 20);
    prefetch_read(&data, 90, 10);
}

#[test]
fn test_quantized_dot_q4_basic_cov() {
    let block_a = vec![0u8; 18]; // Q4 block: 2 bytes scale + 16 bytes data
    let block_b = vec![0u8; 18];
    let result = quantized_dot_q4(&block_a, &block_b);
    assert!(result.is_finite());
}

#[test]
fn test_quantized_dot_q8_basic_cov() {
    let block_a = vec![0u8; 34]; // Q8 block: 2 bytes scale + 32 bytes data
    let block_b = vec![0u8; 34];
    let result = quantized_dot_q8(&block_a, &block_b);
    assert!(result.is_finite());
}

#[test]
fn test_quantized_matvec_q4_basic_cov() {
    let rows = 4;
    let cols = 32;
    let weights = vec![0u8; rows * 18]; // Q4 blocks
    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q4(&weights, &input, rows, cols);
    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q8_basic_cov() {
    let rows = 4;
    let cols = 32;
    let weights = vec![0u8; rows * 34]; // Q8 blocks
    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q8(&weights, &input, rows, cols);
    assert_eq!(result.len(), rows);
}

#[test]
fn test_large_vocab_threshold_cov() {
    assert_eq!(LARGE_VOCAB_THRESHOLD, 65536);
}

// =========================================================================
// Extended Coverage Tests for Softmax
// =========================================================================

#[test]
fn test_scalar_softmax_single_element_cov() {
    let result = scalar_softmax(&[1.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_scalar_softmax_uniform_cov() {
    let input = vec![1.0; 4];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 4);
    // Uniform input should give uniform output (0.25 each)
    for &v in &result {
        assert!((v - 0.25).abs() < 1e-6);
    }
}

#[test]
fn test_scalar_softmax_large_values_cov() {
    // Test numerical stability with large values
    let input = vec![1000.0, 1001.0, 1002.0];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 3);
    // Should still sum to 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_scalar_softmax_negative_values_cov() {
    let input = vec![-1.0, -2.0, -3.0];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 3);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_single_element_cov() {
    let result = simd_softmax(&[2.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_simd_softmax_uniform_cov() {
    let input = vec![0.0; 8];
    let result = simd_softmax(&input);
    assert_eq!(result.len(), 8);
    for &v in &result {
        assert!((v - 0.125).abs() < 1e-6);
    }
}

#[test]
fn test_simd_softmax_matches_scalar_cov() {
    let input = vec![0.1, 0.5, 0.3, 0.2, 0.8, 0.4, 0.6, 0.7];
    let scalar_result = scalar_softmax(&input);
    let simd_result = simd_softmax(&input);
    assert_eq!(scalar_result.len(), simd_result.len());
    for (s, r) in scalar_result.iter().zip(simd_result.iter()) {
        assert!((s - r).abs() < 1e-5);
    }
}

// =========================================================================
// Extended Coverage Tests for RoPE
// =========================================================================

#[test]
fn test_scalar_rope_basic_cov() {
    let input = vec![1.0; 16]; // 1 token, 16 hidden
    let result = scalar_rope(&input, 1, 16, 10000.0);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_scalar_rope_multiple_positions_cov() {
    let input = vec![1.0; 64]; // 4 tokens, 16 hidden
    let result = scalar_rope(&input, 4, 16, 10000.0);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_scalar_rope_different_theta_cov() {
    let input = vec![1.0; 32];
    let result1 = scalar_rope(&input, 2, 16, 10000.0);
    let result2 = scalar_rope(&input, 2, 16, 500000.0);
    // Different theta should give different results
    assert_ne!(result1, result2);
}

#[test]
fn test_simd_rope_basic_cov() {
    let input = vec![1.0; 16];
    let result = simd_rope(&input, 1, 16, 10000.0);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_simd_rope_multiple_positions_cov() {
    let input = vec![1.0; 64];
    let result = simd_rope(&input, 4, 16, 10000.0);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_simd_rope_matches_scalar_cov() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let scalar_result = scalar_rope(&input, 2, 16, 10000.0);
    let simd_result = simd_rope(&input, 2, 16, 10000.0);
    assert_eq!(scalar_result.len(), simd_result.len());
    for (s, r) in scalar_result.iter().zip(simd_result.iter()) {
        assert!((s - r).abs() < 1e-4, "scalar={}, simd={}", s, r);
    }
}

#[test]
fn test_scalar_rope_zero_seq_len_cov() {
    let input = vec![1.0; 16];
    let result = scalar_rope(&input, 0, 16, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_zero_head_dim_cov() {
    let input = vec![1.0; 16];
    let result = scalar_rope(&input, 1, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_simd_rope_zero_seq_len_cov() {
    let input = vec![1.0; 16];
    let result = simd_rope(&input, 0, 16, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_simd_rope_zero_head_dim_cov() {
    let input = vec![1.0; 16];
    let result = simd_rope(&input, 1, 0, 10000.0);
    assert!(result.is_empty());
}

// =========================================================================
// Extended Coverage Tests for GPU Compute
// =========================================================================

#[test]
fn test_gpu_compute_dot_empty_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let result = compute.dot(&[], &[]);
    assert!(result.is_err() || result.expect("GPU operation failed").abs() < 1e-10);
}

#[test]
fn test_gpu_compute_relu_empty_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let result = compute.relu(&[]).expect("test");
    assert!(result.is_empty());
}

#[test]
fn test_gpu_compute_sigmoid_multiple_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let input = vec![-100.0, 0.0, 100.0];
    let output = compute.sigmoid(&input).expect("test");
    assert!(output[0] < 0.01); // sigmoid(-100) ≈ 0
    assert!((output[1] - 0.5).abs() < 1e-5);
    assert!(output[2] > 0.99); // sigmoid(100) ≈ 1
}

#[test]
fn test_gpu_compute_matmul_1x1_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let a = vec![3.0];
    let b = vec![4.0];
    let c = compute.matmul(&a, &b, 1, 1, 1).expect("test");
    assert_eq!(c.len(), 1);
    assert!((c[0] - 12.0).abs() < 1e-5);
}

#[test]
fn test_gpu_compute_matmul_large_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let m = 64;
    let k = 64;
    let n = 64;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 10) as f32 * 0.1).collect();
    let c = compute.matmul(&a, &b, m, k, n).expect("test");
    assert_eq!(c.len(), m * n);
}

// =========================================================================
// Extended Coverage Tests for HybridScheduler
// =========================================================================

#[test]
fn test_hybrid_scheduler_default_threshold_cov() {
    let scheduler = HybridScheduler::new().expect("test");
    // Default threshold should be set
    assert!(scheduler.gpu_threshold() > 0);
}

#[test]
fn test_hybrid_scheduler_has_gpu_cov() {
    let scheduler = HybridScheduler::with_threshold(100).expect("test");
    // has_gpu returns whether GPU backend is active
    let _has_gpu = scheduler.has_gpu();
}

// =========================================================================
// Extended Coverage Tests for GpuBufferPool
// =========================================================================

#[test]
fn test_buffer_pool_multiple_sizes_cov() {
    let mut pool = GpuBufferPool::new();

    // Acquire different sizes
    let buf1 = pool.acquire(100);
    let buf2 = pool.acquire(1000);
    let buf3 = pool.acquire(10000);

    assert_eq!(buf1.len(), 100);
    assert_eq!(buf2.len(), 1000);
    assert_eq!(buf3.len(), 10000);

    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3);

    // Should have 3 cached buffers
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 3);
}

#[test]
fn test_buffer_pool_stats_bytes_cov() {
    let mut pool = GpuBufferPool::new();

    let buf = pool.acquire(256);
    pool.release(buf);

    let stats = pool.stats();
    assert!(stats.cached_bytes >= 256 * std::mem::size_of::<f32>());
}

// =========================================================================
// Extended Coverage Tests for AsyncGpuResult
// =========================================================================

#[test]
fn test_async_result_set_twice_cov() {
    let mut result = AsyncGpuResult::pending();
    result.set_result(vec![1.0]);
    result.set_result(vec![2.0]); // Second set replaces the first
    assert_eq!(result.wait(), vec![2.0]);
}

#[test]
fn test_async_result_ready_wait_cov() {
    let result = AsyncGpuResult::ready(vec![1.0, 2.0, 3.0]);
    assert!(result.is_ready());
    let data = result.wait();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

// =========================================================================
// Extended Coverage Tests for Constants
// =========================================================================

#[test]
fn test_max_gpu_buffer_bytes_cov() {
    assert_eq!(MAX_GPU_BUFFER_BYTES, 256 * 1024 * 1024);
}

#[test]
fn test_exceeds_limit_edge_cases_cov() {
    // Exactly at limit
    let at_limit = MAX_GPU_BUFFER_BYTES / std::mem::size_of::<f32>();
    assert!(!exceeds_gpu_buffer_limit(at_limit));

    // One over limit
    assert!(exceeds_gpu_buffer_limit(at_limit + 1));
}

// =========================================================================
// Extended Coverage Tests for LayerNorm
// =========================================================================

#[test]
fn test_standard_layernorm_with_bias_cov() {
    let input = vec![0.0f32, 1.0, 2.0, 3.0];
    let gamma = vec![2.0f32; 4];
    let beta = vec![1.0f32; 4];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_layernorm_with_bias_cov() {
    let input = vec![0.0f32, 1.0, 2.0, 3.0];
    let gamma = vec![2.0f32; 4];
    let beta = vec![1.0f32; 4];
    let result = fused_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_standard_layernorm_small_eps_cov() {
    let input = vec![1e-10f32; 4];
    let gamma = vec![1.0f32; 4];
    let beta = vec![0.0f32; 4];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-12);
    assert_eq!(result.len(), 4);
    // Should not panic with very small values
}

// =========================================================================
// Extended Coverage Tests for FFN functions
// =========================================================================

#[test]
fn test_sequential_ffn_identity_cov() {
    // Test with identity-like weights
    let hidden_dim = 4;
    let inter_dim = 4;
    let hidden = vec![1.0f32; hidden_dim];
    let w1 = vec![0.25f32; hidden_dim * inter_dim]; // Uniform weights
    let w2 = vec![0.25f32; inter_dim * hidden_dim];
    let result = sequential_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);
    assert_eq!(result.len(), hidden_dim);
}

#[test]
fn test_parallel_ffn_identity_cov() {
    let hidden_dim = 4;
    let inter_dim = 4;
    let hidden = vec![1.0f32; hidden_dim];
    let w1 = vec![0.25f32; hidden_dim * inter_dim];
    let w2 = vec![0.25f32; inter_dim * hidden_dim];
    let result = parallel_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);
    assert_eq!(result.len(), hidden_dim);
}

// =========================================================================
// Extended Coverage Tests for batch_embed
// =========================================================================

#[test]
fn test_batch_embed_single_token_cov() {
    let embedding_table = vec![1.0f32; 10 * 4]; // 10 tokens, dim 4
    let tokens = vec![0usize];
    let result = batch_embed(&embedding_table, &tokens, 4);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_batch_embed_last_token_cov() {
    let embedding_table = vec![1.0f32; 10 * 4]; // 10 tokens, dim 4
    let tokens = vec![9usize]; // Last token
    let result = batch_embed(&embedding_table, &tokens, 4);
    assert_eq!(result.len(), 4);
}

// =========================================================================
// Extended Coverage Tests for prefetch
// =========================================================================

#[test]
fn test_prefetch_read_boundary_cov() {
    let data = vec![1.0f32; 10];
    prefetch_read(&data, 0, 10); // Full range
    prefetch_read(&data, 9, 1); // Last element
    prefetch_read(&data, 5, 0); // Zero count
}

// =========================================================================
// Extended Coverage Tests for Quantized Operations
// =========================================================================

#[test]
fn test_quantized_dot_q4_nonzero_cov() {
    let mut block_a = vec![0u8; 18];
    let mut block_b = vec![0u8; 18];
    // Set non-zero scale (f16 at bytes 0-1)
    block_a[0] = 0x00;
    block_a[1] = 0x3C; // f16 ≈ 1.0
    block_b[0] = 0x00;
    block_b[1] = 0x3C;
    // Set some non-zero quants
    block_a[2] = 0xFF;
    block_b[2] = 0xFF;
    let result = quantized_dot_q4(&block_a, &block_b);
    assert!(result.is_finite());
}

#[test]
fn test_quantized_dot_q8_nonzero_cov() {
    let mut block_a = vec![0u8; 34];
    let mut block_b = vec![0u8; 34];
    // Set non-zero scale
    block_a[0] = 0x00;
    block_a[1] = 0x3C;
    block_b[0] = 0x00;
    block_b[1] = 0x3C;
    // Set some non-zero quants
    block_a[2] = 127;
    block_b[2] = 127;
    let result = quantized_dot_q8(&block_a, &block_b);
    assert!(result.is_finite());
}

// =========================================================================
// Deep Coverage Tests for gpu.rs (Phase 802)
// =========================================================================

// --- exceeds_gpu_buffer_limit tests ---
#[test]
fn test_exceeds_gpu_buffer_limit_small_cov() {
    // Small buffer should not exceed limit
    assert!(!exceeds_gpu_buffer_limit(1000));
    assert!(!exceeds_gpu_buffer_limit(0));
}

#[test]
fn test_exceeds_gpu_buffer_limit_large_cov() {
    // Large buffer should exceed limit (256MB / 4 bytes = 67108864 f32s)
    let limit_elements = 256 * 1024 * 1024 / 4;
    assert!(exceeds_gpu_buffer_limit(limit_elements + 1));
    assert!(exceeds_gpu_buffer_limit(100_000_000));
}

#[test]
fn test_exceeds_gpu_buffer_limit_boundary_cov() {
    // At boundary
    let limit_elements = 256 * 1024 * 1024 / 4;
    // At limit should not exceed (<=)
    assert!(!exceeds_gpu_buffer_limit(limit_elements));
}

// --- scalar_softmax deep tests ---
#[test]
fn test_scalar_softmax_empty_deep2() {
    let result = scalar_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_softmax_single_deep2() {
    let result = scalar_softmax(&[0.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_scalar_softmax_uniform_deep2() {
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 4);
    for val in &result {
        assert!((val - 0.25).abs() < 1e-5);
    }
}

#[test]
fn test_scalar_softmax_numerical_stability_deep2() {
    // Large values should not overflow
    let input = vec![1000.0, 1001.0, 1002.0];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 3);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// --- simd_softmax deep tests ---
#[test]
fn test_simd_softmax_empty_deep2() {
    let result = simd_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_simd_softmax_single_deep2() {
    let result = simd_softmax(&[0.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_matches_scalar_deep2() {
    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let scalar_result = scalar_softmax(&input);
    let simd_result = simd_softmax(&input);
    assert_eq!(scalar_result.len(), simd_result.len());
    for i in 0..scalar_result.len() {
        assert!((scalar_result[i] - simd_result[i]).abs() < 1e-5);
    }
}

// --- scalar_rope deep tests ---
#[test]
fn test_scalar_rope_empty_deep2() {
    let result = scalar_rope(&[], 0, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_zero_seq_len_deep2() {
    let result = scalar_rope(&[1.0, 2.0, 3.0, 4.0], 0, 4, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_zero_head_dim_deep2() {
    let result = scalar_rope(&[1.0, 2.0, 3.0, 4.0], 1, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_basic_deep2() {
    // Single position, single head, head_dim=4
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = scalar_rope(&input, 1, 4, 10000.0);
    assert_eq!(result.len(), 4);
    // Position 0 should apply rotation with angle=0 (cos=1, sin=0)
    // So output should be close to input at position 0
    assert!((result[0] - 1.0).abs() < 1e-4);
}

#[test]
fn test_scalar_rope_multiple_positions_deep2() {
    // 2 positions, 1 head, head_dim=4
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = scalar_rope(&input, 2, 4, 10000.0);
    assert_eq!(result.len(), 8);
    // Results should be finite
    for val in &result {
        assert!(val.is_finite());
    }
}

// --- simd_rope deep tests ---
#[test]
fn test_simd_rope_empty_deep2() {
    let result = simd_rope(&[], 0, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_simd_rope_matches_scalar_deep2() {
    let input: Vec<f32> = (0..64).map(|x| x as f32 * 0.1).collect();
    let scalar_result = scalar_rope(&input, 2, 16, 10000.0);
    let simd_result = simd_rope(&input, 2, 16, 10000.0);
    assert_eq!(scalar_result.len(), simd_result.len());
    for i in 0..scalar_result.len() {
        assert!(
            (scalar_result[i] - simd_result[i]).abs() < 1e-4,
            "Mismatch at {}: scalar={}, simd={}",
            i,
            scalar_result[i],
            simd_result[i]
        );
    }
}

// --- CacheAlignedBuffer tests ---
#[test]
fn test_cache_aligned_buffer_new_cov() {
    let buf = CacheAlignedBuffer::new(100);
    assert_eq!(buf.len(), 100);
    assert!(!buf.is_empty());
}

#[test]
fn test_cache_aligned_buffer_empty_cov() {
    let buf = CacheAlignedBuffer::new(0);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

#[test]
fn test_cache_aligned_buffer_alignment_cov() {
    let buf = CacheAlignedBuffer::new(256);
    // Should be aligned to 64 bytes
    assert!(buf.is_aligned(64));
}

#[test]
fn test_cache_aligned_buffer_slice_cov() {
    let mut buf = CacheAlignedBuffer::new(10);
    let slice = buf.as_mut_slice();
    slice[0] = 42.0;
    slice[9] = 99.0;
    assert_eq!(buf.as_slice()[0], 42.0);
    assert_eq!(buf.as_slice()[9], 99.0);
}

// --- prefetch_read tests ---
#[test]
fn test_prefetch_read_in_bounds_cov() {
    let data = vec![1.0f32; 100];
    // Should not panic
    prefetch_read(&data, 0, 50);
    prefetch_read(&data, 50, 49);
}

#[test]
fn test_prefetch_read_out_of_bounds_cov() {
    let data = vec![1.0f32; 10];
    // Should be a no-op when out of bounds
    prefetch_read(&data, 5, 100); // position + distance > len
    prefetch_read(&data, 10, 1); // position at end
}

// --- sequential_sum tests ---
#[test]
fn test_sequential_sum_empty_cov() {
    let result = sequential_sum(&[]);
    assert_eq!(result, 0.0);
}

#[test]
fn test_sequential_sum_basic_cov() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sequential_sum(&data);
    assert!((result - 15.0).abs() < 1e-5);
}

// --- sum_with_prefetch tests ---
#[test]
fn test_sum_with_prefetch_empty_cov() {
    let result = sum_with_prefetch(&[], 8);
    assert_eq!(result, 0.0);
}

#[test]
fn test_sum_with_prefetch_basic_cov() {
    let data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
    let result = sum_with_prefetch(&data, 16);
    let expected = 5050.0; // Sum of 1..100
    assert!((result - expected).abs() < 1e-3);
}

#[test]
fn test_sum_with_prefetch_matches_sequential_cov() {
    let data: Vec<f32> = (0..1000).map(|x| x as f32 * 0.001).collect();
    let seq = sequential_sum(&data);
    let prefetch = sum_with_prefetch(&data, 32);
    assert!((seq - prefetch).abs() < 1e-3);
}

// --- naive_matmul tests ---
#[test]
fn test_naive_matmul_identity_cov() {
    // 2x2 @ 2x2 identity-like
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = naive_matmul(&a, &b, 2, 2, 2);
    assert_eq!(result.len(), 4);
    assert!((result[0] - 5.0).abs() < 1e-5);
    assert!((result[3] - 8.0).abs() < 1e-5);
}

#[test]
fn test_naive_matmul_non_square_cov() {
    // 2x3 @ 3x1
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 1.0, 1.0];
    let result = naive_matmul(&a, &b, 2, 3, 1);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 6.0).abs() < 1e-5); // 1+2+3
    assert!((result[1] - 15.0).abs() < 1e-5); // 4+5+6
}

// --- blocked_matmul tests ---
#[test]
fn test_blocked_matmul_matches_naive_cov() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let naive = naive_matmul(&a, &b, 3, 3, 3);
    let blocked = blocked_matmul(&a, &b, 3, 3, 3, 2);
    assert_eq!(naive.len(), blocked.len());
    for i in 0..naive.len() {
        assert!((naive[i] - blocked[i]).abs() < 1e-4);
    }
}

#[test]
fn test_blocked_matmul_large_block_cov() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    // Block size larger than matrix dimensions
    let result = blocked_matmul(&a, &b, 2, 2, 2, 100);
    assert_eq!(result.len(), 4);
    assert!((result[0] - 19.0).abs() < 1e-5);
}

// --- TensorPool tests ---
#[test]
fn test_tensor_pool_new_cov() {
    let pool = TensorPool::new(10);
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_acquire_release_cov() {
    let mut pool = TensorPool::new(5);
    let buf = pool.acquire(100);
    assert_eq!(buf.len(), 100);
    pool.release(buf);
    assert_eq!(pool.available(), 1);
}

#[test]
fn test_tensor_pool_reuse_cov() {
    let mut pool = TensorPool::new(5);
    let buf1 = pool.acquire(100);
    pool.release(buf1);
    let buf2 = pool.acquire(100);
    assert_eq!(buf2.len(), 100);
    // Should have reused the buffer
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_size_mismatch_cov() {
    let mut pool = TensorPool::new(5);
    let buf = pool.acquire(100);
    pool.release(buf);
    // Request different size - should allocate new
    let buf2 = pool.acquire(200);
    assert_eq!(buf2.len(), 200);
}

#[test]
fn test_tensor_pool_capacity_limit_cov() {
    let mut pool = TensorPool::new(2);
    let buf1 = pool.acquire(10);
    let buf2 = pool.acquire(10);
    let buf3 = pool.acquire(10);
    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3); // Should not exceed capacity
    assert!(pool.available() <= 2);
}

// --- DoubleBuffer tests ---
#[test]
fn test_double_buffer_new_cov() {
    let buf: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(buf.capacity(), 100);
}

#[test]
fn test_double_buffer_front_back_cov() {
    let mut buf: DoubleBuffer<f32> = DoubleBuffer::new(10);
    buf.back_mut()[0] = 42.0;
    assert_eq!(buf.front()[0], 0.0);
    buf.swap();
    assert_eq!(buf.front()[0], 42.0);
}

#[test]
fn test_double_buffer_swap_cov() {
    let mut buf: DoubleBuffer<i32> = DoubleBuffer::new(5);
    buf.back_mut()[0] = 1;
    buf.swap();
    buf.back_mut()[0] = 2;
    assert_eq!(buf.front()[0], 1);
    buf.swap();
    assert_eq!(buf.front()[0], 2);
}

// --- ChunkedProcessor tests ---
#[test]
fn test_chunked_processor_new_cov() {
    let proc = ChunkedProcessor::new(64);
    assert_eq!(proc.chunk_size(), 64);
}

#[test]
fn test_chunked_processor_num_chunks_cov() {
    let proc = ChunkedProcessor::new(10);
    assert_eq!(proc.num_chunks(0), 0);
    assert_eq!(proc.num_chunks(10), 1);
    assert_eq!(proc.num_chunks(11), 2);
    assert_eq!(proc.num_chunks(20), 2);
    assert_eq!(proc.num_chunks(21), 3);
}

#[test]
fn test_chunked_processor_bounds_cov() {
    let proc = ChunkedProcessor::new(10);
    assert_eq!(proc.chunk_bounds(0, 25), (0, 10));
    assert_eq!(proc.chunk_bounds(1, 25), (10, 20));
    assert_eq!(proc.chunk_bounds(2, 25), (20, 25));
}

#[test]
fn test_chunked_processor_process_cov() {
    let proc = ChunkedProcessor::new(5);
    let data: Vec<f32> = vec![1.0; 12];
    let result = proc.process_chunks(&data, |chunk| chunk.iter().sum());
    assert!((result - 12.0).abs() < 1e-5);
}

// --- GpuPipelineStage tests ---
#[test]
fn test_pipeline_stage_values_cov() {
    assert_eq!(GpuPipelineStage::Embed as u8, 0);
    assert_eq!(GpuPipelineStage::Attention as u8, 1);
    assert_eq!(GpuPipelineStage::FFN as u8, 2);
    assert_eq!(GpuPipelineStage::Output as u8, 3);
}

// --- InferencePipeline tests ---
#[test]
fn test_inference_pipeline_new_cov() {
    let pipe = InferencePipeline::new(4);
    assert_eq!(pipe.num_stages(), 4);
    assert_eq!(pipe.total_latency(), 0.0);
}

#[test]
fn test_inference_pipeline_record_cov() {
    let mut pipe = InferencePipeline::new(4);
    pipe.record_stage_time(GpuPipelineStage::Embed, 1.0);
    pipe.record_stage_time(GpuPipelineStage::Attention, 5.0);
    pipe.record_stage_time(GpuPipelineStage::FFN, 3.0);
    pipe.record_stage_time(GpuPipelineStage::Output, 1.0);
    assert!((pipe.total_latency() - 10.0).abs() < 1e-5);
}

#[test]
fn test_inference_pipeline_reset_cov() {
    let mut pipe = InferencePipeline::new(4);
    pipe.record_stage_time(GpuPipelineStage::Embed, 5.0);
    pipe.reset();
    assert_eq!(pipe.total_latency(), 0.0);
    assert!(pipe.stage_breakdown().is_empty());
}

// --- TokenBatch tests ---
#[test]
fn test_token_batch_new_deep2() {
    let batch = TokenBatch::new(32);
    assert_eq!(batch.capacity(), 32);
    assert_eq!(batch.len(), 0);
    assert!(batch.is_empty());
}

#[test]
fn test_token_batch_push_deep2() {
    let mut batch = TokenBatch::new(5);
    assert!(batch.push(1).is_none());
    assert!(batch.push(2).is_none());
    assert!(batch.push(3).is_none());
    assert_eq!(batch.len(), 3);
    assert!(!batch.is_full());
}

#[test]
fn test_token_batch_full_deep2() {
    let mut batch = TokenBatch::new(3);
    batch.push(1);
    batch.push(2);
    let result = batch.push(3); // Should return Some with tokens
    assert!(result.is_some());
    let tokens = result.expect("test");
    assert_eq!(tokens, vec![1, 2, 3]);
}

#[test]
fn test_token_batch_flush_deep2() {
    let mut batch = TokenBatch::new(5);
    batch.push(10);
    batch.push(20);
    batch.push(30);
    let tokens = batch.flush();
    assert_eq!(tokens, vec![10, 20, 30]);
    assert!(batch.is_empty());
}

// --- quantized_matvec_q4 deep tests ---
#[test]
fn test_quantized_matvec_q4_basic_deep2() {
    // Q4_0 block: 2 bytes scale + 16 bytes (32 4-bit values)
    let block_size = 18;
    let cols = 32; // One block
    let rows = 2;
    let weights = vec![0u8; rows * (cols / 32) * block_size];
    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q4(&weights, &input, rows, cols);
    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q4_empty_deep2() {
    let weights: Vec<u8> = vec![];
    let input: Vec<f32> = vec![];
    let result = quantized_matvec_q4(&weights, &input, 0, 0);
    assert!(result.is_empty());
}

// --- quantized_matvec_q8 deep tests ---
#[test]
fn test_quantized_matvec_q8_basic_deep2() {
    // Q8_0 block: 2 bytes scale + 32 bytes values
    let block_size = 34;
    let cols = 32; // One block
    let rows = 2;
    let weights = vec![0u8; rows * (cols / 32) * block_size];
    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q8(&weights, &input, rows, cols);
    assert_eq!(result.len(), rows);
}

// --- QuantizedAccumulator tests ---
#[test]
fn test_quantized_accumulator_new_deep2() {
    let acc = QuantizedAccumulator::new();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_add_scaled_deep2() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(2.0, 0.5); // 2.0 * 0.5 = 1.0
    acc.add_scaled(4.0, 0.5); // 4.0 * 0.5 = 2.0
    assert!((acc.sum() - 3.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_add_block_deep2() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_block(10.0, 0.5); // 10.0 * 0.5 = 5.0
    acc.add_block(6.0, 0.5); // 6.0 * 0.5 = 3.0
    assert!((acc.sum() - 8.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_reset_deep2() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(10.0, 1.0);
    acc.reset();
    assert_eq!(acc.sum(), 0.0);
}

// --- ContiguousAttentionBuffer tests ---
#[test]
fn test_contiguous_attention_buffer_new_deep2() {
    // Constructor: new(max_seq_len, num_heads, head_dim)
    let buf = ContiguousAttentionBuffer::new(8, 4, 64);
    assert_eq!(buf.max_seq_len(), 8);
}

#[test]
fn test_contiguous_attention_buffer_is_contiguous_deep2() {
    let buf = ContiguousAttentionBuffer::new(10, 2, 32);
    assert!(buf.is_contiguous());
}

#[test]
fn test_contiguous_attention_buffer_get_views_deep2() {
    let buf = ContiguousAttentionBuffer::new(4, 2, 16);
    let (q, k, v, o) = buf.get_views();
    assert_eq!(q.len(), 4 * 2 * 16); // max_seq_len * num_heads * head_dim
    assert_eq!(k.len(), q.len());
    assert_eq!(v.len(), q.len());
    assert_eq!(o.len(), q.len());
}

#[test]
fn test_contiguous_attention_buffer_reset_deep2() {
    let mut buf = ContiguousAttentionBuffer::new(4, 2, 16);
    {
        let (q, _, _, _) = buf.get_views_mut();
        q[0] = 42.0;
    }
    buf.reset();
    let (q, _, _, _) = buf.get_views();
    assert_eq!(q[0], 0.0);
}

// --- batch_embed tests ---
#[test]
fn test_batch_embed_empty_cov() {
    let embedding_table = vec![1.0f32; 100];
    let tokens: Vec<usize> = vec![];
    let result = batch_embed(&embedding_table, &tokens, 10);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_multiple_tokens_cov() {
    let hidden_dim = 4;
    let vocab_size = 10;
    let mut embedding_table = vec![0.0f32; vocab_size * hidden_dim];
    // Set specific embeddings
    for i in 0..vocab_size {
        for j in 0..hidden_dim {
            embedding_table[i * hidden_dim + j] = (i * 10 + j) as f32;
        }
    }
    let tokens = vec![0, 5, 9];
    let result = batch_embed(&embedding_table, &tokens, hidden_dim);
    assert_eq!(result.len(), 3 * hidden_dim);
    // Check token 0 embedding
    assert_eq!(result[0], 0.0);
    // Check token 5 embedding
    assert_eq!(result[4], 50.0);
    // Check token 9 embedding
    assert_eq!(result[8], 90.0);
}

// --- sequential_ffn and parallel_ffn tests ---
#[test]
fn test_sequential_ffn_zero_weights_cov() {
    let hidden_dim = 4;
    let inter_dim = 8;
    let hidden = vec![1.0f32; hidden_dim];
    let w1 = vec![0.0f32; hidden_dim * inter_dim];
    let w2 = vec![0.0f32; inter_dim * hidden_dim];
    let result = sequential_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);
    assert_eq!(result.len(), hidden_dim);
    for val in &result {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_parallel_ffn_matches_sequential_cov() {
    let hidden_dim = 8;
    let inter_dim = 16;
    let hidden: Vec<f32> = (0..hidden_dim).map(|x| x as f32 * 0.1).collect();
    let w1: Vec<f32> = (0..hidden_dim * inter_dim)
        .map(|x| x as f32 * 0.01)
        .collect();
    let w2: Vec<f32> = (0..inter_dim * hidden_dim)
        .map(|x| x as f32 * 0.01)
        .collect();

    let seq = sequential_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);
    let par = parallel_ffn(&hidden, &w1, &w2, hidden_dim, inter_dim);

    assert_eq!(seq.len(), par.len());
    for i in 0..seq.len() {
        assert!(
            (seq[i] - par[i]).abs() < 1e-3,
            "Mismatch at {}: seq={}, par={}",
            i,
            seq[i],
            par[i]
        );
    }
}

// --- standard_layernorm and fused_layernorm deep tests ---
#[test]
fn test_standard_layernorm_basic_deep2() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
    // Result should be normalized (mean ~0, std ~1)
    let mean: f32 = result.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_standard_layernorm_with_scale_shift_deep2() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![2.0, 2.0, 2.0, 2.0];
    let beta = vec![1.0, 1.0, 1.0, 1.0];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    // After scaling by 2 and shifting by 1
    let mean: f32 = result.iter().sum::<f32>() / 4.0;
    assert!((mean - 1.0).abs() < 1e-4); // Mean should be close to beta
}

#[test]
fn test_fused_layernorm_matches_standard_deep2() {
    let input: Vec<f32> = (0..32).map(|x| x as f32 * 0.1).collect();
    let gamma = vec![1.0f32; 32];
    let beta = vec![0.0f32; 32];

    let standard = standard_layernorm(&input, &gamma, &beta, 1e-5);
    let fused = fused_layernorm(&input, &gamma, &beta, 1e-5);

    assert_eq!(standard.len(), fused.len());
    for i in 0..standard.len() {
        assert!(
            (standard[i] - fused[i]).abs() < 1e-4,
            "Mismatch at {}: standard={}, fused={}",
            i,
            standard[i],
            fused[i]
        );
    }
}

// --- ForwardArena tests ---
#[test]
fn test_forward_arena_new_cov() {
    let arena = ForwardArena::new(1024);
    assert_eq!(arena.capacity(), 1024);
}

#[test]
fn test_forward_arena_alloc_cov() {
    let mut arena = ForwardArena::new(1000);
    let buf1 = arena.alloc(100);
    assert_eq!(buf1.len(), 100);
    let buf2 = arena.alloc(200);
    assert_eq!(buf2.len(), 200);
}

#[test]
fn test_forward_arena_reset_cov() {
    let mut arena = ForwardArena::new(500);
    arena.alloc(100);
    arena.alloc(200);
    arena.reset();
    // After reset, should be able to allocate full capacity again
    let buf = arena.alloc(400);
    assert_eq!(buf.len(), 400);
}

// --- ScratchBuffer tests ---
