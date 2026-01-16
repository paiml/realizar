//! EXTREME TDD coverage tests for realizar/src/gpu.rs
//!
//! Tests target uncovered public functions, structs, and error paths
//! to increase coverage from 77% to 90%+.

#![cfg(feature = "gpu")]

use realizar::gpu::{
    batch_embed, blocked_matmul, exceeds_gpu_buffer_limit, fused_layernorm, naive_matmul,
    parallel_ffn, prefetch_read, quantized_dot_q4, quantized_dot_q8, quantized_matvec_q4,
    quantized_matvec_q8, scalar_rope, scalar_softmax, sequential_ffn, sequential_sum,
    simd_rope, simd_softmax, standard_layernorm, sum_with_prefetch,
    CacheAlignedBuffer, ChunkedProcessor, ComputeBackend, ConnectionConfig, ConnectionPool,
    ContiguousAttentionBuffer, DegradationManager, DegradationMode,
    DoubleBuffer, ErrorClassification, ErrorRecoveryStrategy, FailureIsolator, ForwardArena,
    GpuBufferPool, GpuCompute, GpuGenerateConfig, GpuModelConfig, GpuPipelineStage,
    HealthChecker, HybridScheduler, InferenceBatchScheduler, InferenceEventNotifier,
    InferenceMetrics, InferencePipeline, PriorityRequest, PriorityRequestQueue, QuantizedAccumulator,
    RecoveryAction, RequestOutcome, ResourceTracker, ScratchBuffer, SpeculativeBuffer,
    StreamingKVCache, StreamingKVCacheFp16, TensorPool, TimeoutManager, TokenBatch,
    TokenRateLimiter, WeightType, LARGE_VOCAB_THRESHOLD, AsyncRequestQueue,
};
use std::time::{Duration, Instant};

// ============================================================================
// GPU Buffer Limit Tests
// ============================================================================

#[test]
fn test_exceeds_gpu_buffer_limit_boundary() {
    // 256MB = 268435456 bytes / 4 bytes per f32 = 67108864 elements
    let max_elements = 256 * 1024 * 1024 / 4;

    assert!(!exceeds_gpu_buffer_limit(max_elements - 1));
    assert!(!exceeds_gpu_buffer_limit(max_elements));
    assert!(exceeds_gpu_buffer_limit(max_elements + 1));
}

#[test]
fn test_large_vocab_threshold_constant() {
    assert_eq!(LARGE_VOCAB_THRESHOLD, 65536);
}

// ============================================================================
// Softmax Tests
// ============================================================================

#[test]
fn test_scalar_softmax_numerical_stability() {
    // Large values should not cause overflow due to max subtraction
    let input = vec![1000.0, 1001.0, 1002.0];
    let output = scalar_softmax(&input);

    assert_eq!(output.len(), 3);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0");
}

#[test]
fn test_simd_softmax_numerical_stability() {
    let input = vec![1000.0, 1001.0, 1002.0];
    let output = simd_softmax(&input);

    assert_eq!(output.len(), 3);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0");
}

#[test]
fn test_softmax_negative_values() {
    let input = vec![-100.0, -50.0, 0.0];
    let scalar_out = scalar_softmax(&input);
    let simd_out = simd_softmax(&input);

    for i in 0..3 {
        assert!((scalar_out[i] - simd_out[i]).abs() < 1e-5);
    }
}

// ============================================================================
// RoPE Tests
// ============================================================================

#[test]
fn test_rope_with_multiple_heads() {
    let seq_len = 2;
    let head_dim = 4;
    let num_heads = 2;
    let hidden_dim = num_heads * head_dim;
    let input = vec![1.0; seq_len * hidden_dim];

    let scalar_out = scalar_rope(&input, seq_len, head_dim, 10000.0);
    let simd_out = simd_rope(&input, seq_len, head_dim, 10000.0);

    assert_eq!(scalar_out.len(), input.len());
    assert_eq!(simd_out.len(), input.len());

    for i in 0..scalar_out.len() {
        assert!((scalar_out[i] - simd_out[i]).abs() < 1e-4);
    }
}

#[test]
fn test_rope_position_encoding() {
    let seq_len = 4;
    let head_dim = 8;
    let input: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();

    let output = scalar_rope(&input, seq_len, head_dim, 10000.0);

    // Position 0 should be unchanged (cos(0)=1, sin(0)=0)
    assert!((output[0] - input[0]).abs() < 1e-5);
}

// ============================================================================
// ContiguousAttentionBuffer Tests
// ============================================================================

#[test]
fn test_contiguous_attention_buffer_creation() {
    let buffer = ContiguousAttentionBuffer::new(128, 8, 64);
    assert!(buffer.is_contiguous());
    assert_eq!(buffer.max_seq_len(), 128);
}

#[test]
fn test_contiguous_attention_buffer_views() {
    let buffer = ContiguousAttentionBuffer::new(4, 2, 4);
    let (q, k, v, o) = buffer.get_views();

    // Each tensor should have size: max_seq_len * num_heads * head_dim = 4 * 2 * 4 = 32
    assert_eq!(q.len(), 32);
    assert_eq!(k.len(), 32);
    assert_eq!(v.len(), 32);
    assert_eq!(o.len(), 32);
}

#[test]
fn test_contiguous_attention_buffer_mutable_views() {
    let mut buffer = ContiguousAttentionBuffer::new(2, 2, 2);
    let (q, k, v, o) = buffer.get_views_mut();

    q[0] = 1.0;
    k[0] = 2.0;
    v[0] = 3.0;
    o[0] = 4.0;

    let (q_read, k_read, v_read, o_read) = buffer.get_views();
    assert!((q_read[0] - 1.0).abs() < 1e-5);
    assert!((k_read[0] - 2.0).abs() < 1e-5);
    assert!((v_read[0] - 3.0).abs() < 1e-5);
    assert!((o_read[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_contiguous_attention_buffer_reset() {
    let mut buffer = ContiguousAttentionBuffer::new(2, 2, 2);

    {
        let (q, _, _, _) = buffer.get_views_mut();
        q[0] = 100.0;
    }

    buffer.reset();

    let (q, _, _, _) = buffer.get_views();
    assert!((q[0] - 0.0).abs() < 1e-5);
}

// ============================================================================
// Batch Embed Tests
// ============================================================================

#[test]
fn test_batch_embed_empty_tokens() {
    let table = vec![1.0; 100];
    let result = batch_embed(&table, &[], 10);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_empty_table() {
    let result = batch_embed(&[], &[0, 1], 10);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_out_of_bounds() {
    let table = vec![1.0; 20]; // vocab_size * hidden_dim = 2 * 10 = 20
    let tokens = vec![0, 1, 5]; // token 5 is out of bounds
    let result = batch_embed(&table, &tokens, 10);

    // First two should be valid, third should be padded with zeros
    assert_eq!(result.len(), 30);
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[20] - 0.0).abs() < 1e-5); // Padded zeros
}

// ============================================================================
// FFN Tests
// ============================================================================

#[test]
fn test_sequential_ffn_empty_input() {
    let result = sequential_ffn(&[], &[1.0], &[1.0], 4, 8);
    assert!(result.is_empty());
}

#[test]
fn test_parallel_ffn_empty_input() {
    let result = parallel_ffn(&[], &[1.0], &[1.0], 4, 8);
    assert!(result.is_empty());
}

#[test]
fn test_ffn_sequential_vs_parallel() {
    let hidden_dim = 4;
    let intermediate_dim = 8;
    let input = vec![0.5; hidden_dim];
    let w_up = vec![0.1; hidden_dim * intermediate_dim];
    let w_down = vec![0.1; intermediate_dim * hidden_dim];

    let seq = sequential_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);
    let par = parallel_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    assert_eq!(seq.len(), par.len());
    for i in 0..seq.len() {
        assert!((seq[i] - par[i]).abs() < 1e-4);
    }
}

// ============================================================================
// LayerNorm Tests
// ============================================================================

#[test]
fn test_standard_layernorm_empty() {
    let result = standard_layernorm(&[], &[], &[], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_fused_layernorm_empty() {
    let result = fused_layernorm(&[], &[], &[], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_layernorm_partial_gamma_beta() {
    // When gamma/beta are shorter than input, defaults are used
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![2.0, 2.0]; // Shorter
    let beta = vec![0.5]; // Even shorter

    let std_result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    let fused_result = fused_layernorm(&input, &gamma, &beta, 1e-5);

    assert_eq!(std_result.len(), 4);
    assert_eq!(fused_result.len(), 4);
}

// ============================================================================
// CacheAlignedBuffer Tests
// ============================================================================

#[test]
fn test_cache_aligned_buffer_creation() {
    let buffer = CacheAlignedBuffer::new(100);
    assert_eq!(buffer.len(), 100);
    assert!(!buffer.is_empty());
}

#[test]
fn test_cache_aligned_buffer_empty() {
    let buffer = CacheAlignedBuffer::new(0);
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());
}

#[test]
fn test_cache_aligned_buffer_alignment() {
    let buffer = CacheAlignedBuffer::new(256);
    // Should be aligned to 64-byte cache line
    assert!(buffer.is_aligned(64) || buffer.is_aligned(32) || buffer.is_aligned(16));
}

#[test]
fn test_cache_aligned_buffer_slice_access() {
    let mut buffer = CacheAlignedBuffer::new(10);
    let slice = buffer.as_mut_slice();
    slice[0] = 42.0;

    assert!((buffer.as_slice()[0] - 42.0).abs() < 1e-5);
}

// ============================================================================
// Prefetch and Sum Tests
// ============================================================================

#[test]
fn test_prefetch_read_within_bounds() {
    let data = vec![1.0; 100];
    // Should not panic
    prefetch_read(&data, 0, 10);
    prefetch_read(&data, 50, 20);
}

#[test]
fn test_prefetch_read_out_of_bounds() {
    let data = vec![1.0; 10];
    // Should not panic when prefetch position exceeds bounds
    prefetch_read(&data, 5, 10);
}

#[test]
fn test_sum_with_prefetch_equals_sequential() {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

    let seq_sum = sequential_sum(&data);
    let prefetch_sum = sum_with_prefetch(&data, 8);

    assert!((seq_sum - prefetch_sum).abs() < 1e-5);
}

// ============================================================================
// Matmul Tests
// ============================================================================

#[test]
fn test_naive_matmul_identity() {
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let vec = vec![3.0, 4.0];

    let result = naive_matmul(&identity, &vec, 2, 2, 1);
    assert!((result[0] - 3.0).abs() < 1e-5);
    assert!((result[1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_blocked_matmul_equals_naive() {
    let m = 16;
    let k = 16;
    let n = 16;
    let a: Vec<f32> = (0..m*k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k*n).map(|i| (i % 11) as f32 * 0.1).collect();

    let naive = naive_matmul(&a, &b, m, k, n);
    let blocked = blocked_matmul(&a, &b, m, k, n, 4);

    for i in 0..naive.len() {
        assert!((naive[i] - blocked[i]).abs() < 1e-4);
    }
}

// ============================================================================
// TensorPool Tests
// ============================================================================

#[test]
fn test_tensor_pool_capacity() {
    let pool = TensorPool::new(5);
    assert_eq!(pool.capacity(), 5);
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_acquire_release() {
    let mut pool = TensorPool::new(3);

    let buf1 = pool.acquire(100);
    assert_eq!(buf1.len(), 100);

    pool.release(buf1);
    assert_eq!(pool.available(), 1);

    // Re-acquire should reuse
    let buf2 = pool.acquire(50);
    assert_eq!(pool.available(), 0);
    assert!(buf2.capacity() >= 100); // Reused buffer has original capacity
}

#[test]
fn test_tensor_pool_clear() {
    let mut pool = TensorPool::new(5);

    // Acquire 3 distinct buffers by requesting different sizes
    let buf1 = pool.acquire(100);
    let buf2 = pool.acquire(200);
    let buf3 = pool.acquire(300);

    // Release them back
    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3);

    assert_eq!(pool.available(), 3);
    pool.clear();
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_exceeds_capacity() {
    let mut pool = TensorPool::new(2);

    let buf1 = pool.acquire(100);
    let buf2 = pool.acquire(100);
    let buf3 = pool.acquire(100);

    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3); // This should be dropped, not kept

    assert_eq!(pool.available(), 2);
}

// ============================================================================
// ForwardArena Tests
// ============================================================================

#[test]
fn test_forward_arena_creation() {
    let arena = ForwardArena::new(1000);
    assert_eq!(arena.capacity(), 1000);
    assert_eq!(arena.used(), 0);
}

#[test]
fn test_forward_arena_alloc() {
    let mut arena = ForwardArena::new(100);

    let slice1 = arena.alloc(30);
    assert_eq!(slice1.len(), 30);
    assert_eq!(arena.used(), 30);

    let slice2 = arena.alloc(40);
    assert_eq!(slice2.len(), 40);
    assert_eq!(arena.used(), 70);
}

#[test]
fn test_forward_arena_reset() {
    let mut arena = ForwardArena::new(100);
    arena.alloc(50);
    assert_eq!(arena.used(), 50);

    arena.reset();
    assert_eq!(arena.used(), 0);
}

#[test]
#[should_panic(expected = "insufficient capacity")]
fn test_forward_arena_overflow() {
    let mut arena = ForwardArena::new(10);
    arena.alloc(20); // Should panic
}

// ============================================================================
// ScratchBuffer Tests
// ============================================================================

#[test]
fn test_scratch_buffer_creation() {
    let scratch = ScratchBuffer::new(4, 100);
    assert_eq!(scratch.num_layers(), 4);
    assert_eq!(scratch.layer_size(), 100);
    assert_eq!(scratch.total_size(), 400);
}

#[test]
fn test_scratch_buffer_get_layer() {
    let scratch = ScratchBuffer::new(3, 50);

    let layer0 = scratch.get_layer(0);
    let layer2 = scratch.get_layer(2);

    assert_eq!(layer0.len(), 50);
    assert_eq!(layer2.len(), 50);
}

#[test]
fn test_scratch_buffer_get_layer_mut() {
    let mut scratch = ScratchBuffer::new(2, 10);

    {
        let layer = scratch.get_layer_mut(0);
        layer[0] = 42.0;
    }

    assert!((scratch.get_layer(0)[0] - 42.0).abs() < 1e-5);
}

#[test]
#[should_panic(expected = "layer index")]
fn test_scratch_buffer_out_of_bounds() {
    let scratch = ScratchBuffer::new(2, 10);
    let _ = scratch.get_layer(5); // Should panic
}

#[test]
fn test_scratch_buffer_reset() {
    let mut scratch = ScratchBuffer::new(2, 10);
    scratch.get_layer_mut(0)[0] = 100.0;

    scratch.reset();
    assert!((scratch.get_layer(0)[0] - 0.0).abs() < 1e-5);
}

// ============================================================================
// Quantized Dot Product Tests
// ============================================================================

#[test]
fn test_quantized_dot_q4_short_input() {
    let block_a = vec![0u8; 10]; // Too short
    let block_b = vec![0u8; 18];

    let result = quantized_dot_q4(&block_a, &block_b);
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_dot_q8_short_input() {
    let block_a = vec![0u8; 30]; // Too short
    let block_b = vec![0u8; 34];

    let result = quantized_dot_q8(&block_a, &block_b);
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_matvec_q4_basic() {
    // Q4_0 block: 2 bytes scale + 16 bytes data = 18 bytes per 32 values
    let rows = 1;
    let cols = 32;
    let block_size = 18;

    // Create a simple weight matrix with scale = 1.0
    let mut weights = vec![0u8; rows * block_size];
    // Set scale to 1.0 (in f16)
    let scale_bits = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale_bits[0];
    weights[1] = scale_bits[1];
    // Set all values to 8 (which becomes 0 after centering)
    for weight in weights.iter_mut().take(18).skip(2) {
        *weight = 0x88; // 8 in both nibbles
    }

    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q4(&weights, &input, rows, cols);

    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q8_basic() {
    // Q8_0 block: 2 bytes scale + 32 bytes data = 34 bytes per 32 values
    let rows = 1;
    let cols = 32;
    let block_size = 34;

    let mut weights = vec![0u8; rows * block_size];
    let scale_bits = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale_bits[0];
    weights[1] = scale_bits[1];

    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q8(&weights, &input, rows, cols);

    assert_eq!(result.len(), rows);
}

// ============================================================================
// QuantizedAccumulator Tests
// ============================================================================

#[test]
fn test_quantized_accumulator_new() {
    let acc = QuantizedAccumulator::new();
    assert!((acc.sum() - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_add_scaled() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(2.0, 3.0);
    assert!((acc.sum() - 6.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_add_block() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_block(10.0, 0.5);
    acc.add_block(4.0, 0.5);
    assert!((acc.sum() - 7.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_reset() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(100.0, 1.0);
    acc.reset();
    assert!((acc.sum() - 0.0).abs() < 1e-5);
}

// ============================================================================
// DoubleBuffer Tests
// ============================================================================

#[test]
fn test_double_buffer_creation() {
    let buffer: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(buffer.capacity(), 100);
}

#[test]
fn test_double_buffer_front_back() {
    let mut buffer: DoubleBuffer<f32> = DoubleBuffer::new(10);

    buffer.back_mut()[0] = 42.0;
    assert!((buffer.front()[0] - 0.0).abs() < 1e-5);

    buffer.swap();
    assert!((buffer.front()[0] - 42.0).abs() < 1e-5);
}

// ============================================================================
// ChunkedProcessor Tests
// ============================================================================

#[test]
fn test_chunked_processor_creation() {
    let processor = ChunkedProcessor::new(64);
    assert_eq!(processor.chunk_size(), 64);
}

#[test]
fn test_chunked_processor_num_chunks() {
    let processor = ChunkedProcessor::new(10);
    assert_eq!(processor.num_chunks(0), 0);
    assert_eq!(processor.num_chunks(5), 1);
    assert_eq!(processor.num_chunks(10), 1);
    assert_eq!(processor.num_chunks(11), 2);
    assert_eq!(processor.num_chunks(25), 3);
}

#[test]
fn test_chunked_processor_bounds() {
    let processor = ChunkedProcessor::new(10);

    assert_eq!(processor.chunk_bounds(0, 25), (0, 10));
    assert_eq!(processor.chunk_bounds(1, 25), (10, 20));
    assert_eq!(processor.chunk_bounds(2, 25), (20, 25));
}

#[test]
fn test_chunked_processor_process() {
    let processor = ChunkedProcessor::new(5);
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    let total = processor.process_chunks(&data, |chunk| chunk.iter().sum());
    let expected: f32 = (0..12).sum::<i32>() as f32;

    assert!((total - expected).abs() < 1e-5);
}

// ============================================================================
// InferencePipeline Tests
// ============================================================================

#[test]
fn test_inference_pipeline_creation() {
    let pipeline = InferencePipeline::new(4);
    assert_eq!(pipeline.num_stages(), 4);
}

#[test]
fn test_inference_pipeline_timing() {
    let mut pipeline = InferencePipeline::new(4);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 2.0);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 3.0);
    pipeline.record_stage_time(GpuPipelineStage::Output, 0.5);

    assert!((pipeline.total_latency() - 6.5).abs() < 1e-5);
}

#[test]
fn test_inference_pipeline_stage_breakdown() {
    let mut pipeline = InferencePipeline::new(4);
    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);

    let breakdown = pipeline.stage_breakdown();
    assert!(breakdown.contains_key(&GpuPipelineStage::Embed));
}

#[test]
fn test_inference_pipeline_reset() {
    let mut pipeline = InferencePipeline::new(4);
    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);

    pipeline.reset();
    assert!((pipeline.total_latency() - 0.0).abs() < 1e-5);
}

// ============================================================================
// TokenBatch Tests
// ============================================================================

#[test]
fn test_token_batch_creation() {
    let batch = TokenBatch::new(8);
    assert_eq!(batch.capacity(), 8);
    assert_eq!(batch.len(), 0);
    assert!(batch.is_empty());
    assert!(!batch.is_full());
}

#[test]
fn test_token_batch_push_not_full() {
    let mut batch = TokenBatch::new(4);

    assert!(batch.push(1).is_none());
    assert!(batch.push(2).is_none());
    assert_eq!(batch.len(), 2);
}

#[test]
fn test_token_batch_push_full() {
    let mut batch = TokenBatch::new(2);

    assert!(batch.push(1).is_none());
    let flushed = batch.push(2);

    assert!(flushed.is_some());
    assert_eq!(flushed.unwrap(), vec![1, 2]);
    assert!(batch.is_empty());
}

#[test]
fn test_token_batch_flush() {
    let mut batch = TokenBatch::new(10);
    batch.push(1);
    batch.push(2);

    let flushed = batch.flush();
    assert_eq!(flushed, vec![1, 2]);
    assert!(batch.is_empty());
}

// ============================================================================
// SpeculativeBuffer Tests
// ============================================================================

#[test]
fn test_speculative_buffer_creation() {
    let buffer = SpeculativeBuffer::new(8);
    assert_eq!(buffer.capacity(), 8);
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());
}

#[test]
fn test_speculative_buffer_add_verify() {
    let mut buffer = SpeculativeBuffer::new(10);

    buffer.add_candidate(1, 0.9);
    buffer.add_candidate(2, 0.8);
    buffer.add_candidate(3, 0.7);

    // All match
    let (accepted, rejected_idx) = buffer.verify(&[1, 2, 3]);
    assert_eq!(accepted, 3);
    assert!(rejected_idx.is_none());
}

#[test]
fn test_speculative_buffer_verify_mismatch() {
    let mut buffer = SpeculativeBuffer::new(10);

    buffer.add_candidate(1, 0.9);
    buffer.add_candidate(2, 0.8);
    buffer.add_candidate(3, 0.7);

    // Mismatch at index 1
    let (accepted, rejected_idx) = buffer.verify(&[1, 5, 3]);
    assert_eq!(accepted, 1);
    assert_eq!(rejected_idx, Some(1));
}

#[test]
fn test_speculative_buffer_accept_reject() {
    let mut buffer = SpeculativeBuffer::new(10);

    buffer.add_candidate(1, 0.9);
    buffer.add_candidate(2, 0.8);
    buffer.add_candidate(3, 0.7);

    buffer.accept(2);
    assert_eq!(buffer.len(), 1); // Only one left

    buffer.reject();
    assert!(buffer.is_empty());
}

// ============================================================================
// InferenceBatchScheduler Tests
// ============================================================================

#[test]
fn test_batch_scheduler_creation() {
    let scheduler = InferenceBatchScheduler::new();
    assert_eq!(scheduler.pending_count(), 0);
    assert_eq!(scheduler.completed_count(), 0);
}

#[test]
fn test_batch_scheduler_submit_complete() {
    let mut scheduler = InferenceBatchScheduler::new();

    let id1 = scheduler.submit(vec![1, 2, 3]);
    let id2 = scheduler.submit(vec![4, 5]);

    assert_eq!(scheduler.pending_count(), 2);

    scheduler.complete(id1, vec![10, 11]);
    assert_eq!(scheduler.pending_count(), 1);
    assert_eq!(scheduler.completed_count(), 1);

    let result = scheduler.poll();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), (id1, vec![10, 11]));

    scheduler.complete(id2, vec![20]);
    let drained = scheduler.drain();
    assert_eq!(drained.len(), 1);
}

// ============================================================================
// AsyncRequestQueue Tests
// ============================================================================

#[test]
fn test_async_request_queue_creation() {
    let queue: AsyncRequestQueue<i32> = AsyncRequestQueue::new(5);
    assert_eq!(queue.capacity(), 5);
    assert_eq!(queue.len(), 0);
    assert!(queue.is_empty());
    assert!(!queue.is_full());
}

#[test]
fn test_async_request_queue_push_pop() {
    let mut queue: AsyncRequestQueue<i32> = AsyncRequestQueue::new(3);

    assert!(queue.try_push(1));
    assert!(queue.try_push(2));
    assert!(queue.try_push(3));
    assert!(!queue.try_push(4)); // Full

    assert_eq!(queue.try_pop(), Some(1));
    assert!(queue.try_push(4)); // Now has space
}

// ============================================================================
// InferenceEventNotifier Tests
// ============================================================================

#[test]
fn test_event_notifier_creation() {
    let notifier = InferenceEventNotifier::new();
    assert_eq!(notifier.handler_count(), 0);
}

#[test]
fn test_event_notifier_register_notify() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    let mut notifier = InferenceEventNotifier::new();
    let counter = Arc::new(AtomicU64::new(0));
    let counter_clone = counter.clone();

    notifier.register(Box::new(move |id, _tokens| {
        counter_clone.fetch_add(id, Ordering::SeqCst);
    }));

    assert_eq!(notifier.handler_count(), 1);

    notifier.notify(42, &[1, 2, 3]);
    assert_eq!(counter.load(Ordering::SeqCst), 42);
}

#[test]
fn test_event_notifier_clear() {
    let mut notifier = InferenceEventNotifier::new();
    notifier.register(Box::new(|_, _| {}));
    assert_eq!(notifier.handler_count(), 1);

    notifier.clear();
    assert_eq!(notifier.handler_count(), 0);
}

// ============================================================================
// TimeoutManager Tests
// ============================================================================

#[test]
fn test_timeout_manager_creation() {
    let manager = TimeoutManager::new();
    assert_eq!(manager.active_count(), 0);
}

#[test]
fn test_timeout_manager_register_remove() {
    let mut manager = TimeoutManager::new();
    let deadline = Instant::now() + Duration::from_secs(60);

    manager.register(1, deadline);
    manager.register(2, deadline);
    assert_eq!(manager.active_count(), 2);

    manager.remove(1);
    assert_eq!(manager.active_count(), 1);
}

#[test]
fn test_timeout_manager_check_expired() {
    let mut manager = TimeoutManager::new();

    // Register one that's already expired
    let past = Instant::now() - Duration::from_secs(1);
    manager.register(1, past);

    // Register one that's not expired
    let future = Instant::now() + Duration::from_secs(60);
    manager.register(2, future);

    let expired = manager.check_expired();
    assert_eq!(expired, vec![1]);
    assert_eq!(manager.active_count(), 1);
}

// ============================================================================
// PriorityRequestQueue Tests
// ============================================================================

#[test]
fn test_priority_queue_creation() {
    let queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();
    assert_eq!(queue.len(), 0);
    assert!(queue.is_empty());
}

#[test]
fn test_priority_request_creation() {
    let req = PriorityRequest::new(10, "test".to_string());
    assert_eq!(req.priority(), 10);
    assert_eq!(req.data(), &"test".to_string());
    assert_eq!(req.into_data(), "test".to_string());
}

#[test]
fn test_priority_queue_ordering() {
    let mut queue: PriorityRequestQueue<&str> = PriorityRequestQueue::new();

    queue.enqueue(PriorityRequest::new(1, "low"));
    queue.enqueue(PriorityRequest::new(10, "high"));
    queue.enqueue(PriorityRequest::new(5, "medium"));

    // Should dequeue highest priority first
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), "high");
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), "medium");
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), "low");
    assert!(queue.dequeue_highest().is_none());
}

#[test]
fn test_priority_queue_fifo_same_priority() {
    let mut queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();

    queue.enqueue(PriorityRequest::new(5, 1));
    queue.enqueue(PriorityRequest::new(5, 2));
    queue.enqueue(PriorityRequest::new(5, 3));

    // FIFO for same priority
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), 1);
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), 2);
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), 3);
}

// ============================================================================
// TokenRateLimiter Tests
// ============================================================================

#[test]
fn test_rate_limiter_creation() {
    let limiter = TokenRateLimiter::new(100.0, 50);
    assert_eq!(limiter.tokens_available(), 50);
}

#[test]
fn test_rate_limiter_acquire() {
    let mut limiter = TokenRateLimiter::new(100.0, 10);

    assert!(limiter.try_acquire(5));
    assert_eq!(limiter.tokens_available(), 5);

    assert!(!limiter.try_acquire(10)); // Not enough
    assert_eq!(limiter.tokens_available(), 5);
}

#[test]
fn test_rate_limiter_refill() {
    let mut limiter = TokenRateLimiter::new(1000000.0, 100); // Very high rate
    limiter.try_acquire(100);
    assert_eq!(limiter.tokens_available(), 0);

    std::thread::sleep(Duration::from_millis(10));
    limiter.refill();

    // Should have refilled something
    assert!(limiter.tokens_available() > 0);
}

// ============================================================================
// ResourceTracker Tests
// ============================================================================

#[test]
fn test_resource_tracker_creation() {
    let tracker = ResourceTracker::new(1024, 100);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_allocation() {
    let mut tracker = ResourceTracker::new(1000, 100);

    assert!(tracker.can_allocate(500, 50));
    let id1 = tracker.allocate(500, 50);
    assert!(id1.is_some());

    assert_eq!(tracker.memory_usage(), 500);
    assert_eq!(tracker.compute_usage(), 50);

    // Can't allocate beyond capacity
    assert!(!tracker.can_allocate(600, 10));
    assert!(tracker.allocate(600, 10).is_none());
}

#[test]
fn test_resource_tracker_release() {
    let mut tracker = ResourceTracker::new(1000, 100);

    let id = tracker.allocate(500, 50).unwrap();
    tracker.release(id);

    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_usage_percentage() {
    let mut tracker = ResourceTracker::new(1000, 100);
    tracker.allocate(500, 25);

    let (mem_pct, compute_pct) = tracker.usage_percentage();
    assert!((mem_pct - 50.0).abs() < 1e-5);
    assert!((compute_pct - 25.0).abs() < 1e-5);
}

#[test]
fn test_resource_tracker_zero_capacity() {
    let tracker = ResourceTracker::new(0, 0);
    let (mem_pct, compute_pct) = tracker.usage_percentage();

    assert!((mem_pct - 0.0).abs() < 1e-5);
    assert!((compute_pct - 0.0).abs() < 1e-5);
}

// ============================================================================
// InferenceMetrics Tests
// ============================================================================

#[test]
fn test_inference_metrics_creation() {
    let metrics = InferenceMetrics::new();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

#[test]
fn test_inference_metrics_record() {
    let mut metrics = InferenceMetrics::new();

    metrics.record_inference(Duration::from_millis(100), 10);
    metrics.record_inference(Duration::from_millis(200), 20);

    assert_eq!(metrics.total_inferences(), 2);
    assert_eq!(metrics.total_tokens(), 30);
}

#[test]
fn test_inference_metrics_percentile() {
    let mut metrics = InferenceMetrics::new();

    // Empty case
    assert!(metrics.latency_percentile(50).is_none());

    // Add some latencies
    for i in 1..=100 {
        metrics.record_inference(Duration::from_millis(i), 1);
    }

    let p50 = metrics.latency_percentile(50).unwrap();
    assert!(p50.as_millis() >= 40 && p50.as_millis() <= 60);
}

#[test]
fn test_inference_metrics_reset() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(Duration::from_millis(100), 10);

    metrics.reset();

    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

// ============================================================================
// HealthChecker Tests
// ============================================================================

#[test]
fn test_health_checker_creation() {
    let checker = HealthChecker::new();
    assert_eq!(checker.check_count(), 0);
    assert!(checker.is_healthy()); // No checks = healthy
}

#[test]
fn test_health_checker_register_check() {
    let mut checker = HealthChecker::new();

    checker.register_check("test1", Box::new(|| true));
    checker.register_check("test2", Box::new(|| false));

    assert_eq!(checker.check_count(), 2);

    let results = checker.check_all();
    assert!(results["test1"]);
    assert!(!results["test2"]);

    assert!(!checker.is_healthy()); // One check failed
}

#[test]
fn test_health_checker_clear() {
    let mut checker = HealthChecker::new();
    checker.register_check("test", Box::new(|| true));

    checker.clear();
    assert_eq!(checker.check_count(), 0);
}

// ============================================================================
// ErrorRecoveryStrategy Tests
// ============================================================================

#[test]
fn test_error_recovery_strategy_creation() {
    let strategy = ErrorRecoveryStrategy::new()
        .with_max_retries(5)
        .with_base_delay(Duration::from_millis(50))
        .with_max_delay(Duration::from_secs(5))
        .with_jitter(0.2);

    assert_eq!(strategy.max_retries(), 5);
}

#[test]
fn test_error_recovery_classify_transient() {
    let strategy = ErrorRecoveryStrategy::new();

    let timeout_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
    assert_eq!(strategy.classify_error(&timeout_err), ErrorClassification::Transient);

    let interrupted_err = std::io::Error::new(std::io::ErrorKind::Interrupted, "interrupted");
    assert_eq!(strategy.classify_error(&interrupted_err), ErrorClassification::Transient);
}

#[test]
fn test_error_recovery_classify_fatal() {
    let strategy = ErrorRecoveryStrategy::new();

    let not_found_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
    assert_eq!(strategy.classify_error(&not_found_err), ErrorClassification::Fatal);
}

#[test]
fn test_error_recovery_classify_gpu() {
    let strategy = ErrorRecoveryStrategy::new();

    let gpu_err = std::io::Error::other("GPU unavailable");
    assert_eq!(strategy.classify_error(&gpu_err), ErrorClassification::GpuFailure);
}

#[test]
fn test_error_recovery_determine_action() {
    let strategy = ErrorRecoveryStrategy::new().with_max_retries(3);

    let timeout_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");

    // Should retry on transient error
    match strategy.determine_action(&timeout_err, 0) {
        RecoveryAction::Retry { delay: _ } => (),
        _ => panic!("Expected Retry action"),
    }

    // Should fail after max retries
    match strategy.determine_action(&timeout_err, 3) {
        RecoveryAction::Fail => (),
        _ => panic!("Expected Fail action"),
    }
}

#[test]
fn test_error_recovery_gpu_fallback() {
    let strategy = ErrorRecoveryStrategy::new();

    let gpu_err = std::io::Error::other("GPU error");

    match strategy.determine_action(&gpu_err, 0) {
        RecoveryAction::FallbackToCpu => (),
        _ => panic!("Expected FallbackToCpu action"),
    }
}

#[test]
fn test_error_recovery_calculate_delay() {
    let strategy = ErrorRecoveryStrategy::new()
        .with_base_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_secs(10))
        .with_jitter(0.0); // No jitter for predictable test

    let delay0 = strategy.calculate_delay(0);
    let delay1 = strategy.calculate_delay(1);
    let delay2 = strategy.calculate_delay(2);

    // Exponential backoff: 100ms, 200ms, 400ms
    assert!(delay0.as_millis() >= 100 && delay0.as_millis() <= 110);
    assert!(delay1.as_millis() >= 200 && delay1.as_millis() <= 220);
    assert!(delay2.as_millis() >= 400 && delay2.as_millis() <= 440);
}

// ============================================================================
// DegradationManager Tests
// ============================================================================

#[test]
fn test_degradation_manager_creation() {
    let manager = DegradationManager::new();
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_gpu_unavailable() {
    let mut manager = DegradationManager::new();
    manager.set_gpu_available(false);
    assert_eq!(manager.current_mode(), DegradationMode::CpuFallback);
}

#[test]
fn test_degradation_manager_memory_pressure() {
    let mut manager = DegradationManager::new();
    manager.update_memory_pressure(0.9);
    assert_eq!(manager.current_mode(), DegradationMode::MemoryPressure);
}

#[test]
fn test_degradation_manager_latency_priority() {
    let mut manager = DegradationManager::new();
    manager.set_latency_priority(true);
    assert_eq!(manager.current_mode(), DegradationMode::LowLatency);
}

#[test]
fn test_degradation_manager_recommended_batch_size() {
    let mut manager = DegradationManager::new();

    assert_eq!(manager.recommended_batch_size(16), 16);

    manager.update_memory_pressure(0.9);
    let reduced = manager.recommended_batch_size(100);
    assert!(reduced < 100);
}

#[test]
fn test_degradation_manager_recommended_context() {
    let mut manager = DegradationManager::new();

    assert_eq!(manager.recommended_max_context(4096), 4096);

    manager.update_system_load(realizar::gpu::SystemLoad {
        cpu_percent: 95.0,
        memory_percent: 85.0,
        queue_depth: 100,
    });

    let reduced = manager.recommended_max_context(4096);
    assert!(reduced < 4096);
}

// ============================================================================
// FailureIsolator Tests
// ============================================================================

#[test]
fn test_failure_isolator_creation() {
    let isolator = FailureIsolator::new();
    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 0);
    assert_eq!(isolator.failure_count(), 0);
    assert!(!isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_request_lifecycle() {
    let isolator = FailureIsolator::new();

    let id = isolator.start_request();
    assert_eq!(isolator.active_requests(), 1);

    isolator.complete_request(id, &RequestOutcome::Success);
    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 1);
}

#[test]
fn test_failure_isolator_circuit_breaker() {
    let isolator = FailureIsolator::new();

    // Fail 5 times to open circuit
    for _ in 0..5 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    assert!(isolator.is_circuit_open());
    assert!(isolator.try_start_request().is_err());

    isolator.reset_circuit();
    assert!(!isolator.is_circuit_open());
}

// ============================================================================
// ConnectionConfig and ConnectionPool Tests
// ============================================================================

#[test]
fn test_connection_config_builder() {
    let config = ConnectionConfig::new()
        .with_max_connections(20)
        .with_min_connections(5)
        .with_idle_timeout(Duration::from_secs(600));

    // Just ensure builder pattern works
    let _ = config;
}

#[test]
fn test_connection_pool_creation() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    assert_eq!(pool.active_connections(), 0);
    assert_eq!(pool.idle_connections(), 0);
}

// ============================================================================
// GpuModelConfig Tests
// ============================================================================

#[test]
fn test_gpu_model_config_derived_values() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        num_layers: 32,
        intermediate_dim: 11008,
        eps: 1e-5,
    };

    assert_eq!(config.head_dim(), 128);
    assert_eq!(config.kv_dim(), 1024); // 8 * 128
    assert!(config.is_gqa());

    // qkv_dim = hidden_dim + 2 * kv_dim = 4096 + 2 * 1024 = 6144
    assert_eq!(config.qkv_dim(), 6144);
}

#[test]
fn test_gpu_model_config_non_gqa() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 768,
        num_heads: 12,
        num_kv_heads: 12, // MHA (not GQA)
        num_layers: 12,
        intermediate_dim: 3072,
        eps: 1e-5,
    };

    assert!(!config.is_gqa());
}

// ============================================================================
// GpuGenerateConfig Tests
// ============================================================================

#[test]
fn test_gpu_generate_config_default() {
    let config = GpuGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 0.0).abs() < 1e-5);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_gpu_generate_config_deterministic() {
    let config = GpuGenerateConfig::deterministic(100);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_gpu_generate_config_with_sampling() {
    let config = GpuGenerateConfig::with_sampling(50, 0.7, 40);
    assert_eq!(config.max_tokens, 50);
    assert!((config.temperature - 0.7).abs() < 1e-5);
    assert_eq!(config.top_k, 40);
}

#[test]
fn test_gpu_generate_config_with_stop_tokens() {
    let config = GpuGenerateConfig::default()
        .with_stop_tokens(vec![0, 1, 2]);

    assert_eq!(config.stop_tokens, vec![0, 1, 2]);
}

// ============================================================================
// WeightType Tests
// ============================================================================

#[test]
fn test_weight_type_variants() {
    let types = [
        WeightType::Qkv,
        WeightType::Output,
        WeightType::FfnFc1,
        WeightType::FfnFc2,
        WeightType::LmHead,
    ];

    // Just verify all variants exist
    assert_eq!(types.len(), 5);
}

// ============================================================================
// GpuPipelineStage Tests
// ============================================================================

#[test]
fn test_gpu_pipeline_stage_values() {
    assert_eq!(GpuPipelineStage::Embed as u8, 0);
    assert_eq!(GpuPipelineStage::Attention as u8, 1);
    assert_eq!(GpuPipelineStage::FFN as u8, 2);
    assert_eq!(GpuPipelineStage::Output as u8, 3);
}

// ============================================================================
// StreamingKVCacheFp16 Tests
// ============================================================================

#[test]
fn test_fp16_cache_creation() {
    let cache = StreamingKVCacheFp16::new(4, 1024, 8, 64);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 1024);
}

#[test]
fn test_fp16_cache_append_get() {
    let mut cache = StreamingKVCacheFp16::new(1, 10, 2, 4);
    let kv_dim = 2 * 4;

    let key: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.1).collect();
    let value: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.2).collect();

    cache.append(0, &key, &value);
    assert_eq!(cache.len(), 1);

    let (keys, values) = cache.get_valid_f32(0);

    // Check precision within FP16 tolerance
    for i in 0..kv_dim {
        assert!((keys[i] - key[i]).abs() < 0.01);
        assert!((values[i] - value[i]).abs() < 0.01);
    }
}

#[test]
fn test_fp16_cache_memory_half() {
    let fp32_cache = StreamingKVCache::new(4, 1024, 8, 64);
    let fp16_cache = StreamingKVCacheFp16::new(4, 1024, 8, 64);

    assert_eq!(fp16_cache.memory_bytes(), fp32_cache.memory_bytes() / 2);
}

#[test]
fn test_fp16_cache_get_range_raw() {
    let mut cache = StreamingKVCacheFp16::new(1, 10, 2, 2);
    let kv_dim = 4;

    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];
    cache.append(0, &key, &value);

    let (raw_keys, raw_values) = cache.get_range_raw(0, 0, 1);
    assert_eq!(raw_keys.len(), kv_dim);
    assert_eq!(raw_values.len(), kv_dim);
}

// ============================================================================
// ComputeBackend Tests
// ============================================================================

#[test]
fn test_compute_backend_default() {
    let backend: ComputeBackend = Default::default();
    assert_eq!(backend, ComputeBackend::Auto);
}

#[test]
fn test_gpu_compute_backend_accessor() {
    let compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();
    assert_eq!(compute.backend(), ComputeBackend::Cpu);
}

// ============================================================================
// HybridScheduler Tests
// ============================================================================

#[test]
fn test_hybrid_scheduler_transpose_b() {
    let mut scheduler = HybridScheduler::new().unwrap();

    // A: 2x3, B: 2x3 (will be transposed to 3x2)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // Two rows, three cols

    let result = scheduler.matmul_transpose_b(&a, &b, 2, 3, 2).unwrap();

    // A @ B^T where B^T is 3x2
    assert_eq!(result.len(), 4);
}

// ============================================================================
// GpuBufferPool Stats Tests
// ============================================================================

#[test]
fn test_gpu_pool_stats() {
    let mut pool = GpuBufferPool::new();

    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
    assert_eq!(stats.cached_bytes, 0);

    let buf = pool.acquire(1000);
    pool.release(buf);

    let stats = pool.stats();
    assert!(stats.cached_buffers > 0);
    assert!(stats.cached_bytes > 0);
}
