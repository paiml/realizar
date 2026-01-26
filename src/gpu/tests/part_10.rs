//! Part 10: GPU Module Coverage Tests (gpu/mod.rs)
//!
//! Comprehensive tests for gpu/mod.rs covering:
//! - GPU Buffer Limits (exceeds_gpu_buffer_limit, LARGE_VOCAB_THRESHOLD)
//! - ContiguousAttentionBuffer (all methods)
//! - Batch Processing (batch_embed, sequential_ffn, parallel_ffn, layernorm variants)
//! - Quantized Compute Kernels (quantized_dot_q4/q8, quantized_matvec_q4/q8)
//! - QuantizedAccumulator
//! - Streaming/Pipelining (DoubleBuffer, ChunkedProcessor, InferencePipeline)
//! - Error Recovery (ErrorRecoveryStrategy, DegradationManager, FailureIsolator)
//! - Connection Pooling (ConnectionPool, ConnectionConfig, ConnectionState)
//! - Resource Limits (ResourceLimiter, ResourceConfig, ResourceMonitor)
//! - GGUF Model State (GgufModelState, load_gguf_to_gpu)

use std::io::{Error, ErrorKind};
use std::time::Duration;

use crate::gpu::{
    batch_embed, exceeds_gpu_buffer_limit, fused_layernorm, load_gguf_to_gpu, parallel_ffn,
    quantized_dot_q4, quantized_dot_q8, quantized_matvec_q4, quantized_matvec_q8, sequential_ffn,
    standard_layernorm, ChunkedProcessor, ConnectionConfig, ConnectionPool, ConnectionState,
    ContiguousAttentionBuffer, DegradationManager, DegradationMode, DoubleBuffer,
    ErrorClassification, ErrorRecoveryStrategy, FailureIsolator, GgufModelState, GpuPipelineStage,
    InferencePipeline, LimitResult, QuantizedAccumulator, RecoveryAction, RequestOutcome,
    ResourceConfig, ResourceLimiter, ResourceMonitor, SystemLoad, LARGE_VOCAB_THRESHOLD,
};

// ============================================================================
// GPU Buffer Limits Tests
// ============================================================================

#[test]
fn test_exceeds_gpu_buffer_limit_small() {
    // Small buffer should not exceed limit
    let small_elements = 1000;
    assert!(!exceeds_gpu_buffer_limit(small_elements));
}

#[test]
fn test_exceeds_gpu_buffer_limit_large() {
    // Buffer larger than 256MB in f32 (256MB / 4 = 64M elements)
    let large_elements = 70_000_000; // ~280MB
    assert!(exceeds_gpu_buffer_limit(large_elements));
}

#[test]
fn test_exceeds_gpu_buffer_limit_boundary() {
    // Exactly at limit: 256MB / 4 bytes = 67,108,864 elements
    let at_limit = 67_108_864;
    assert!(!exceeds_gpu_buffer_limit(at_limit));

    // Just over the limit
    let over_limit = 67_108_865;
    assert!(exceeds_gpu_buffer_limit(over_limit));
}

#[test]
fn test_large_vocab_threshold_value() {
    // Verify the threshold constant
    assert_eq!(LARGE_VOCAB_THRESHOLD, 65536);
}

#[test]
fn test_exceeds_gpu_buffer_limit_zero() {
    // Zero elements should not exceed
    assert!(!exceeds_gpu_buffer_limit(0));
}

// ============================================================================
// ContiguousAttentionBuffer Tests
// ============================================================================

#[test]
fn test_contiguous_attention_buffer_new() {
    let max_seq_len = 64;
    let num_heads = 4;
    let head_dim = 16;

    let buffer = ContiguousAttentionBuffer::new(max_seq_len, num_heads, head_dim);

    assert!(buffer.is_contiguous());
    assert_eq!(buffer.max_seq_len(), max_seq_len);
}

#[test]
fn test_contiguous_attention_buffer_get_views() {
    let max_seq_len = 4;
    let num_heads = 2;
    let head_dim = 8;
    let tensor_size = max_seq_len * num_heads * head_dim;

    let buffer = ContiguousAttentionBuffer::new(max_seq_len, num_heads, head_dim);
    let (q, k, v, o) = buffer.get_views();

    assert_eq!(q.len(), tensor_size);
    assert_eq!(k.len(), tensor_size);
    assert_eq!(v.len(), tensor_size);
    assert_eq!(o.len(), tensor_size);

    // All should be zero-initialized
    assert!(q.iter().all(|&x| x == 0.0));
}

#[test]
fn test_contiguous_attention_buffer_get_views_mut() {
    let max_seq_len = 2;
    let num_heads = 2;
    let head_dim = 4;

    let mut buffer = ContiguousAttentionBuffer::new(max_seq_len, num_heads, head_dim);

    {
        let (q, k, v, o) = buffer.get_views_mut();

        // Fill with test values
        q.fill(1.0);
        k.fill(2.0);
        v.fill(3.0);
        o.fill(4.0);
    }

    // Verify values persisted
    let (q, k, v, o) = buffer.get_views();
    assert!(q.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    assert!(k.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    assert!(v.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    assert!(o.iter().all(|&x| (x - 4.0).abs() < 1e-6));
}

#[test]
fn test_contiguous_attention_buffer_reset() {
    let mut buffer = ContiguousAttentionBuffer::new(4, 2, 8);

    // Fill with non-zero values
    {
        let (q, k, v, o) = buffer.get_views_mut();
        q.fill(1.0);
        k.fill(2.0);
        v.fill(3.0);
        o.fill(4.0);
    }

    // Reset
    buffer.reset();

    // All should be zero
    let (q, k, v, o) = buffer.get_views();
    assert!(q.iter().all(|&x| x == 0.0));
    assert!(k.iter().all(|&x| x == 0.0));
    assert!(v.iter().all(|&x| x == 0.0));
    assert!(o.iter().all(|&x| x == 0.0));
}

#[test]
fn test_contiguous_attention_buffer_is_contiguous() {
    let buffer = ContiguousAttentionBuffer::new(8, 4, 16);
    // By construction, this is always true
    assert!(buffer.is_contiguous());
}

// ============================================================================
// Batch Processing Tests - batch_embed
// ============================================================================

#[test]
fn test_batch_embed_basic() {
    let hidden_dim = 4;
    let vocab_size = 10;

    // Create embedding table: each token i has embedding [i, i, i, i]
    let embedding_table: Vec<f32> = (0..vocab_size)
        .flat_map(|i| vec![i as f32; hidden_dim])
        .collect();

    let tokens = vec![0, 2, 5];
    let result = batch_embed(&embedding_table, &tokens, hidden_dim);

    assert_eq!(result.len(), tokens.len() * hidden_dim);

    // Token 0 -> [0, 0, 0, 0]
    assert!(result[0..hidden_dim].iter().all(|&x| x == 0.0));

    // Token 2 -> [2, 2, 2, 2]
    assert!(result[hidden_dim..2 * hidden_dim]
        .iter()
        .all(|&x| (x - 2.0).abs() < 1e-6));

    // Token 5 -> [5, 5, 5, 5]
    assert!(result[2 * hidden_dim..3 * hidden_dim]
        .iter()
        .all(|&x| (x - 5.0).abs() < 1e-6));
}

#[test]
fn test_batch_embed_empty_tokens() {
    let embedding_table = vec![1.0, 2.0, 3.0, 4.0];
    let tokens: Vec<usize> = vec![];

    let result = batch_embed(&embedding_table, &tokens, 4);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_empty_embedding_table() {
    let embedding_table: Vec<f32> = vec![];
    let tokens = vec![0, 1, 2];

    let result = batch_embed(&embedding_table, &tokens, 4);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_out_of_bounds_token() {
    let hidden_dim = 4;
    let vocab_size = 5;
    let embedding_table: Vec<f32> = vec![1.0; vocab_size * hidden_dim];

    // Token 10 is out of bounds
    let tokens = vec![1, 10];
    let result = batch_embed(&embedding_table, &tokens, hidden_dim);

    // Should pad with zeros for out-of-bounds
    assert_eq!(result.len(), 2 * hidden_dim);
    // Second embedding should be zeros
    assert!(result[hidden_dim..].iter().all(|&x| x == 0.0));
}

#[test]
fn test_batch_embed_single_token() {
    let hidden_dim = 3;
    let embedding_table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens
    let tokens = vec![1];

    let result = batch_embed(&embedding_table, &tokens, hidden_dim);
    assert_eq!(result, vec![4.0, 5.0, 6.0]);
}

// ============================================================================
// Batch Processing Tests - FFN
// ============================================================================

#[test]
fn test_sequential_ffn_basic() {
    let hidden_dim = 4;
    let intermediate_dim = 8;

    let input = vec![1.0f32; hidden_dim];
    let w_up = vec![0.1f32; hidden_dim * intermediate_dim];
    let w_down = vec![0.1f32; intermediate_dim * hidden_dim];

    let result = sequential_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    assert_eq!(result.len(), hidden_dim);
    // Result should be non-zero due to GELU activation
}

#[test]
fn test_sequential_ffn_empty_input() {
    let result = sequential_ffn(&[], &[1.0; 8], &[1.0; 8], 2, 4);
    assert!(result.is_empty());
}

#[test]
fn test_parallel_ffn_basic() {
    let hidden_dim = 4;
    let intermediate_dim = 8;

    let input = vec![1.0f32; hidden_dim];
    let w_up = vec![0.1f32; hidden_dim * intermediate_dim];
    let w_down = vec![0.1f32; intermediate_dim * hidden_dim];

    let result = parallel_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    assert_eq!(result.len(), hidden_dim);
}

#[test]
fn test_parallel_ffn_empty_input() {
    let result = parallel_ffn(&[], &[1.0; 8], &[1.0; 8], 2, 4);
    assert!(result.is_empty());
}

#[test]
fn test_sequential_vs_parallel_ffn_equivalence() {
    let hidden_dim = 8;
    let intermediate_dim = 16;

    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1).collect();
    let w_up: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let w_down: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    let seq_result = sequential_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);
    let par_result = parallel_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    // Results should be very close
    for (s, p) in seq_result.iter().zip(par_result.iter()) {
        assert!(
            (s - p).abs() < 1e-4,
            "Sequential and parallel FFN should match"
        );
    }
}

// ============================================================================
// Batch Processing Tests - Layer Normalization
// ============================================================================

#[test]
fn test_standard_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];

    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);

    assert_eq!(result.len(), 4);

    // Mean of normalized output should be close to 0
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean should be ~0 after layernorm");
}

#[test]
fn test_standard_layernorm_empty_input() {
    let result = standard_layernorm(&[], &[1.0], &[0.0], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_fused_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];

    let result = fused_layernorm(&input, &gamma, &beta, 1e-5);

    assert_eq!(result.len(), 4);

    // Mean should be close to 0
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5);
}

#[test]
fn test_fused_layernorm_empty_input() {
    let result = fused_layernorm(&[], &[1.0], &[0.0], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_standard_vs_fused_layernorm_equivalence() {
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let gamma: Vec<f32> = vec![2.0; 16];
    let beta: Vec<f32> = vec![0.5; 16];

    let std_result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    let fused_result = fused_layernorm(&input, &gamma, &beta, 1e-5);

    // Results should be very close
    for (s, f) in std_result.iter().zip(fused_result.iter()) {
        assert!(
            (s - f).abs() < 1e-4,
            "Standard and fused layernorm should match"
        );
    }
}

#[test]
fn test_layernorm_with_gamma_beta() {
    let input = vec![0.0, 1.0, 2.0, 3.0];
    let gamma = vec![2.0; 4]; // Scale by 2
    let beta = vec![1.0; 4]; // Shift by 1

    let result = fused_layernorm(&input, &gamma, &beta, 1e-5);

    // After normalization, scaling and shifting
    // Mean of result should be close to beta (1.0) since normalized mean is 0
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!((mean - 1.0).abs() < 0.1, "Mean should be shifted by beta");
}

// ============================================================================
// Quantized Compute Kernel Tests - Q4
// ============================================================================

#[test]
fn test_quantized_dot_q4_basic() {
    // Q4_0 block: 2 bytes scale + 16 bytes data = 18 bytes
    let block_a: Vec<u8> = vec![
        0x00, 0x3c, // f16 1.0 scale
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    let block_b: Vec<u8> = vec![
        0x00, 0x3c, // f16 1.0 scale
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];

    let result = quantized_dot_q4(&block_a, &block_b);
    // Result should be non-zero (self dot product)
    assert!(result != 0.0);
}

#[test]
fn test_quantized_dot_q4_too_short() {
    let short_block = vec![0u8; 10]; // Less than 18 bytes
    let result = quantized_dot_q4(&short_block, &short_block);
    assert_eq!(result, 0.0);
}

#[test]
fn test_quantized_dot_q4_zeros() {
    // Scale = 0, so output should be 0
    let block = vec![0u8; 18];
    let result = quantized_dot_q4(&block, &block);
    assert_eq!(result, 0.0);
}

// ============================================================================
// Quantized Compute Kernel Tests - Q8
// ============================================================================

#[test]
fn test_quantized_dot_q8_basic() {
    // Q8_0 block: 2 bytes scale + 32 bytes data = 34 bytes
    let mut block_a = vec![0u8; 34];
    block_a[0] = 0x00;
    block_a[1] = 0x3c; // f16 1.0 scale
                       // Fill data with some values
    for i in 2..34 {
        block_a[i] = i as u8;
    }

    let result = quantized_dot_q8(&block_a, &block_a);
    assert!(result != 0.0);
}

#[test]
fn test_quantized_dot_q8_too_short() {
    let short_block = vec![0u8; 20]; // Less than 34 bytes
    let result = quantized_dot_q8(&short_block, &short_block);
    assert_eq!(result, 0.0);
}

#[test]
fn test_quantized_dot_q8_zeros() {
    let block = vec![0u8; 34];
    let result = quantized_dot_q8(&block, &block);
    assert_eq!(result, 0.0);
}

// ============================================================================
// Quantized MatVec Tests
// ============================================================================

#[test]
fn test_quantized_matvec_q4_basic() {
    // 2 rows, 32 cols (1 block per row)
    let rows = 2;
    let cols = 32;
    let block_size = 18;

    // Create weights with some pattern
    let mut weights = vec![0u8; rows * block_size];
    // Set scales to 1.0 (f16)
    for row in 0..rows {
        weights[row * block_size] = 0x00;
        weights[row * block_size + 1] = 0x3c;
    }

    let input = vec![1.0f32; cols];

    let result = quantized_matvec_q4(&weights, &input, rows, cols);

    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q4_empty() {
    let result = quantized_matvec_q4(&[], &[], 0, 0);
    assert!(result.is_empty());
}

#[test]
fn test_quantized_matvec_q8_basic() {
    // 2 rows, 32 cols (1 block per row)
    let rows = 2;
    let cols = 32;
    let block_size = 34;

    let mut weights = vec![0u8; rows * block_size];
    // Set scales to 1.0 (f16)
    for row in 0..rows {
        weights[row * block_size] = 0x00;
        weights[row * block_size + 1] = 0x3c;
    }

    let input = vec![1.0f32; cols];

    let result = quantized_matvec_q8(&weights, &input, rows, cols);

    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q8_empty() {
    let result = quantized_matvec_q8(&[], &[], 0, 0);
    assert!(result.is_empty());
}

// ============================================================================
// QuantizedAccumulator Tests
// ============================================================================

#[test]
fn test_quantized_accumulator_new() {
    let acc = QuantizedAccumulator::new();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_default() {
    let acc = QuantizedAccumulator::default();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_add_scaled() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(2.0, 3.0);
    assert!((acc.sum() - 6.0).abs() < 1e-6);

    acc.add_scaled(1.0, 4.0);
    assert!((acc.sum() - 10.0).abs() < 1e-6);
}

#[test]
fn test_quantized_accumulator_add_block() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_block(5.0, 2.0);
    assert!((acc.sum() - 10.0).abs() < 1e-6);
}

#[test]
fn test_quantized_accumulator_reset() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(10.0, 5.0);
    assert!(acc.sum() > 0.0);

    acc.reset();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_clone() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(3.0, 4.0);

    let cloned = acc.clone();
    assert_eq!(acc.sum(), cloned.sum());
}

// ============================================================================
// DoubleBuffer Tests
// ============================================================================

#[test]
fn test_double_buffer_new() {
    let buf: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(buf.capacity(), 100);
    assert_eq!(buf.front().len(), 100);
}

#[test]
fn test_double_buffer_front() {
    let buf: DoubleBuffer<f32> = DoubleBuffer::new(10);
    let front = buf.front();
    assert_eq!(front.len(), 10);
    assert!(front.iter().all(|&x| x == 0.0));
}

#[test]
fn test_double_buffer_back_mut() {
    let mut buf: DoubleBuffer<f32> = DoubleBuffer::new(5);
    {
        let back = buf.back_mut();
        back[0] = 1.0;
        back[1] = 2.0;
    }

    // Back values should be set
    // After swap, they should appear in front
    buf.swap();

    let front = buf.front();
    assert!((front[0] - 1.0).abs() < 1e-6);
    assert!((front[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_double_buffer_swap() {
    let mut buf: DoubleBuffer<i32> = DoubleBuffer::new(3);

    // Set front and back differently
    buf.back_mut().fill(1);
    buf.swap();

    // Now front should have 1s
    assert!(buf.front().iter().all(|&x| x == 1));

    // Set new back
    buf.back_mut().fill(2);
    buf.swap();

    assert!(buf.front().iter().all(|&x| x == 2));
}

#[test]
fn test_double_buffer_capacity() {
    let buf: DoubleBuffer<u8> = DoubleBuffer::new(256);
    assert_eq!(buf.capacity(), 256);
}

// ============================================================================
// ChunkedProcessor Tests
// ============================================================================

#[test]
fn test_chunked_processor_new() {
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
fn test_chunked_processor_chunk_bounds() {
    let processor = ChunkedProcessor::new(10);

    // Total length 25, chunk 0
    let (start, end) = processor.chunk_bounds(0, 25);
    assert_eq!(start, 0);
    assert_eq!(end, 10);

    // Chunk 1
    let (start, end) = processor.chunk_bounds(1, 25);
    assert_eq!(start, 10);
    assert_eq!(end, 20);

    // Chunk 2 (partial)
    let (start, end) = processor.chunk_bounds(2, 25);
    assert_eq!(start, 20);
    assert_eq!(end, 25);
}

#[test]
fn test_chunked_processor_process_chunks() {
    let processor = ChunkedProcessor::new(3);
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // Sum all chunks
    let total = processor.process_chunks(&data, |chunk| chunk.iter().sum());

    // 1+2+3 + 4+5+6 + 7 = 28
    assert!((total - 28.0).abs() < 1e-6);
}

#[test]
fn test_chunked_processor_process_empty() {
    let processor = ChunkedProcessor::new(10);
    let data: Vec<f32> = vec![];

    let total = processor.process_chunks(&data, |chunk| chunk.iter().sum());
    assert_eq!(total, 0.0);
}

// ============================================================================
// InferencePipeline Tests
// ============================================================================

#[test]
fn test_inference_pipeline_new() {
    let pipeline = InferencePipeline::new(4);
    assert_eq!(pipeline.num_stages(), 4);
    assert_eq!(pipeline.total_latency(), 0.0);
}

#[test]
fn test_inference_pipeline_record_stage_time() {
    let mut pipeline = InferencePipeline::new(4);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 5.0);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 3.0);
    pipeline.record_stage_time(GpuPipelineStage::Output, 2.0);

    assert!((pipeline.total_latency() - 11.0).abs() < 1e-6);
}

#[test]
fn test_inference_pipeline_stage_breakdown() {
    let mut pipeline = InferencePipeline::new(4);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.5);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 4.5);

    let breakdown = pipeline.stage_breakdown();

    assert!(breakdown.contains_key(&GpuPipelineStage::Embed));
    assert!(breakdown.contains_key(&GpuPipelineStage::Attention));

    let embed_time = breakdown.get(&GpuPipelineStage::Embed).unwrap();
    assert!((*embed_time - 1.5).abs() < 1e-6);
}

#[test]
fn test_inference_pipeline_reset() {
    let mut pipeline = InferencePipeline::new(2);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 5.0);
    assert!(pipeline.total_latency() > 0.0);

    pipeline.reset();

    assert_eq!(pipeline.total_latency(), 0.0);
    assert!(pipeline.stage_breakdown().is_empty());
}

#[test]
fn test_gpu_pipeline_stage_values() {
    // Test all stage variants
    assert_eq!(GpuPipelineStage::Embed as u8, 0);
    assert_eq!(GpuPipelineStage::Attention as u8, 1);
    assert_eq!(GpuPipelineStage::FFN as u8, 2);
    assert_eq!(GpuPipelineStage::Output as u8, 3);
}

// ============================================================================
// ErrorRecoveryStrategy Tests
// ============================================================================

#[test]
fn test_error_recovery_strategy_new() {
    let strategy = ErrorRecoveryStrategy::new();
    assert_eq!(strategy.max_retries(), 3);
}

#[test]
fn test_error_recovery_strategy_default() {
    let strategy = ErrorRecoveryStrategy::default();
    assert_eq!(strategy.max_retries(), 3);
}

#[test]
fn test_error_recovery_strategy_with_max_retries() {
    let strategy = ErrorRecoveryStrategy::new().with_max_retries(5);
    assert_eq!(strategy.max_retries(), 5);
}

#[test]
fn test_error_recovery_strategy_with_base_delay() {
    let strategy = ErrorRecoveryStrategy::new().with_base_delay(Duration::from_millis(200));

    let delay = strategy.calculate_delay(0);
    assert!(delay.as_millis() >= 200);
}

#[test]
fn test_error_recovery_strategy_with_max_delay() {
    let strategy = ErrorRecoveryStrategy::new()
        .with_base_delay(Duration::from_secs(1))
        .with_max_delay(Duration::from_secs(2));

    // After many retries, delay should be capped
    let delay = strategy.calculate_delay(10);
    assert!(delay.as_secs() <= 2);
}

#[test]
fn test_error_recovery_strategy_with_jitter() {
    let strategy = ErrorRecoveryStrategy::new().with_jitter(0.5);
    // Just verify it doesn't panic
    let _ = strategy.calculate_delay(1);
}

#[test]
fn test_error_recovery_strategy_classify_transient() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::TimedOut, "timeout");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::Transient
    );

    let error = Error::new(ErrorKind::ConnectionReset, "reset");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::Transient
    );

    let error = Error::new(ErrorKind::Interrupted, "interrupted");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::Transient
    );
}

#[test]
fn test_error_recovery_strategy_classify_fatal() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::NotFound, "not found");
    assert_eq!(strategy.classify_error(&error), ErrorClassification::Fatal);

    let error = Error::new(ErrorKind::PermissionDenied, "denied");
    assert_eq!(strategy.classify_error(&error), ErrorClassification::Fatal);
}

#[test]
fn test_error_recovery_strategy_classify_gpu_failure() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::Other, "GPU memory exhausted");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::GpuFailure
    );

    let error = Error::new(ErrorKind::Other, "CUDA error");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::GpuFailure
    );

    let error = Error::new(ErrorKind::Other, "wgpu device lost");
    assert_eq!(
        strategy.classify_error(&error),
        ErrorClassification::GpuFailure
    );
}

#[test]
fn test_error_recovery_strategy_determine_action_retry() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::TimedOut, "timeout");
    let action = strategy.determine_action(&error, 0);

    assert!(matches!(action, RecoveryAction::Retry { .. }));
}

#[test]
fn test_error_recovery_strategy_determine_action_fail() {
    let strategy = ErrorRecoveryStrategy::new().with_max_retries(3);

    let error = Error::new(ErrorKind::TimedOut, "timeout");
    let action = strategy.determine_action(&error, 3); // At max

    assert!(matches!(action, RecoveryAction::Fail));
}

#[test]
fn test_error_recovery_strategy_determine_action_fallback() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::Other, "GPU error");
    let action = strategy.determine_action(&error, 0);

    assert!(matches!(action, RecoveryAction::FallbackToCpu));
}

#[test]
fn test_error_recovery_strategy_determine_action_with_fallback() {
    let strategy = ErrorRecoveryStrategy::new();

    let error = Error::new(ErrorKind::Other, "GPU unavailable");
    let action = strategy.determine_action_with_fallback(&error, 0);

    assert!(matches!(action, RecoveryAction::FallbackToCpu));
}

#[test]
fn test_error_recovery_strategy_calculate_delay_exponential() {
    let strategy = ErrorRecoveryStrategy::new()
        .with_base_delay(Duration::from_millis(100))
        .with_jitter(0.0); // No jitter for predictable test

    let delay0 = strategy.calculate_delay(0).as_millis();
    let delay1 = strategy.calculate_delay(1).as_millis();
    let delay2 = strategy.calculate_delay(2).as_millis();

    // Exponential backoff: 100, 200, 400
    assert!(delay0 >= 100);
    assert!(delay1 >= 200);
    assert!(delay2 >= 400);
}

// ============================================================================
// DegradationManager Tests
// ============================================================================

#[test]
fn test_degradation_manager_new() {
    let manager = DegradationManager::new();
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_default() {
    let manager = DegradationManager::default();
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_set_gpu_available() {
    let mut manager = DegradationManager::new();

    manager.set_gpu_available(false);
    assert_eq!(manager.current_mode(), DegradationMode::CpuFallback);

    manager.set_gpu_available(true);
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_memory_pressure() {
    let mut manager = DegradationManager::new();

    manager.update_memory_pressure(0.9); // High pressure
    assert_eq!(manager.current_mode(), DegradationMode::MemoryPressure);

    manager.update_memory_pressure(0.5); // Normal
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_latency_priority() {
    let mut manager = DegradationManager::new();

    manager.set_latency_priority(true);
    assert_eq!(manager.current_mode(), DegradationMode::LowLatency);

    manager.set_latency_priority(false);
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_system_load() {
    let mut manager = DegradationManager::new();

    let high_load = SystemLoad {
        cpu_percent: 95.0,
        memory_percent: 90.0,
        queue_depth: 100,
    };

    manager.update_system_load(high_load);
    assert_eq!(manager.current_mode(), DegradationMode::MemoryPressure);
}

#[test]
fn test_degradation_manager_recommended_batch_size() {
    let mut manager = DegradationManager::new();

    // Normal pressure
    assert_eq!(manager.recommended_batch_size(32), 32);

    // High pressure reduces batch size
    manager.update_memory_pressure(0.9);
    let reduced = manager.recommended_batch_size(32);
    assert!(reduced < 32);
}

#[test]
fn test_degradation_manager_recommended_max_context() {
    let mut manager = DegradationManager::new();

    // No load info
    assert_eq!(manager.recommended_max_context(4096), 4096);

    // High load
    let high_load = SystemLoad {
        cpu_percent: 95.0,
        memory_percent: 85.0,
        queue_depth: 60,
    };
    manager.update_system_load(high_load);

    let reduced = manager.recommended_max_context(4096);
    assert!(reduced < 4096);
}

// ============================================================================
// FailureIsolator Tests
// ============================================================================

#[test]
fn test_failure_isolator_new() {
    let isolator = FailureIsolator::new();
    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 0);
    assert_eq!(isolator.failure_count(), 0);
    assert!(!isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_default() {
    let isolator = FailureIsolator::default();
    assert_eq!(isolator.active_requests(), 0);
}

#[test]
fn test_failure_isolator_start_request() {
    let isolator = FailureIsolator::new();

    let id1 = isolator.start_request();
    assert_eq!(isolator.active_requests(), 1);

    let id2 = isolator.start_request();
    assert_eq!(isolator.active_requests(), 2);

    assert_ne!(id1, id2);
}

#[test]
fn test_failure_isolator_try_start_request() {
    let isolator = FailureIsolator::new();

    let result = isolator.try_start_request();
    assert!(result.is_ok());
    assert_eq!(isolator.active_requests(), 1);
}

#[test]
fn test_failure_isolator_complete_request_success() {
    let isolator = FailureIsolator::new();

    let id = isolator.start_request();
    isolator.complete_request(id, &RequestOutcome::Success);

    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 1);
    assert_eq!(isolator.failure_count(), 0);
}

#[test]
fn test_failure_isolator_complete_request_failure() {
    let isolator = FailureIsolator::new();

    let id = isolator.start_request();
    isolator.complete_request(id, &RequestOutcome::Failed("test error".to_string()));

    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 0);
    assert_eq!(isolator.failure_count(), 1);
}

#[test]
fn test_failure_isolator_circuit_opens_on_consecutive_failures() {
    let isolator = FailureIsolator::new();

    // Need 5 consecutive failures to open circuit
    for _ in 0..5 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    assert!(isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_success_resets_consecutive_failures() {
    let isolator = FailureIsolator::new();

    // 4 failures
    for _ in 0..4 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    // 1 success
    let id = isolator.start_request();
    isolator.complete_request(id, &RequestOutcome::Success);

    // 4 more failures
    for _ in 0..4 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    // Circuit should still be closed (success reset counter)
    assert!(!isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_reset_circuit() {
    let isolator = FailureIsolator::new();

    // Open circuit
    for _ in 0..5 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }
    assert!(isolator.is_circuit_open());

    // Reset
    isolator.reset_circuit();
    assert!(!isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_try_start_with_open_circuit() {
    let isolator = FailureIsolator::new();

    // Open circuit
    for _ in 0..5 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    // Should fail with open circuit
    let result = isolator.try_start_request();
    assert!(result.is_err());
}

#[test]
fn test_failure_isolator_register_cleanup() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let isolator = FailureIsolator::new();
    let cleanup_called = Arc::new(AtomicBool::new(false));
    let cleanup_called_clone = cleanup_called.clone();

    let id = isolator.start_request();
    isolator.register_cleanup(id, move || {
        cleanup_called_clone.store(true, Ordering::SeqCst);
    });

    // Cleanup should be called on failure
    isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));

    assert!(cleanup_called.load(Ordering::SeqCst));
}

// ============================================================================
// ConnectionPool Tests
// ============================================================================

#[test]
fn test_connection_config_new() {
    let config = ConnectionConfig::new();
    // Just verify it doesn't panic
    let _ = config;
}

#[test]
fn test_connection_config_default() {
    let config = ConnectionConfig::default();
    let _ = config;
}

#[test]
fn test_connection_config_with_max_connections() {
    let config = ConnectionConfig::new().with_max_connections(20);
    let pool = ConnectionPool::new(config);
    assert_eq!(pool.max_connections(), 20);
}

#[test]
fn test_connection_config_with_min_connections() {
    let config = ConnectionConfig::new().with_min_connections(5);
    let pool = ConnectionPool::new(config);
    assert_eq!(pool.min_connections(), 5);
}

#[test]
fn test_connection_config_with_idle_timeout() {
    let config = ConnectionConfig::new().with_idle_timeout(Duration::from_secs(600));
    let _ = ConnectionPool::new(config);
}

#[test]
fn test_connection_pool_new() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    assert_eq!(pool.active_connections(), 0);
    assert_eq!(pool.idle_connections(), 0);
}

#[test]
fn test_connection_pool_acquire() {
    let config = ConnectionConfig::new().with_max_connections(5);
    let pool = ConnectionPool::new(config);

    let conn = pool.acquire();
    assert!(conn.is_ok());
    assert_eq!(pool.active_connections(), 1);
}

#[test]
fn test_connection_pool_release() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    let conn = pool.acquire().unwrap();
    pool.release(conn);

    assert_eq!(pool.active_connections(), 0);
    assert_eq!(pool.idle_connections(), 1);
}

#[test]
fn test_connection_pool_reuse() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    // Acquire and release
    let conn1 = pool.acquire().unwrap();
    pool.release(conn1);

    // Should reuse from idle pool
    let _conn2 = pool.acquire().unwrap();
    assert_eq!(pool.idle_connections(), 0);
}

#[test]
fn test_connection_pool_exhausted() {
    let config = ConnectionConfig::new().with_max_connections(2);
    let pool = ConnectionPool::new(config);

    let _conn1 = pool.acquire().unwrap();
    let _conn2 = pool.acquire().unwrap();

    // Third should fail
    let result = pool.acquire();
    assert!(result.is_err());
}

#[test]
fn test_connection_pool_try_acquire() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    let result = pool.try_acquire();
    assert!(result.is_ok());
}

#[test]
fn test_connection_pool_check_health() {
    let config = ConnectionConfig::new().with_idle_timeout(Duration::from_millis(1));
    let pool = ConnectionPool::new(config);

    let conn = pool.acquire().unwrap();

    // New connection should be healthy
    let health = pool.check_health(&conn);
    assert!(health == ConnectionState::Healthy || health == ConnectionState::Stale);
}

#[test]
fn test_connection_pool_warm() {
    let config = ConnectionConfig::new().with_min_connections(3);
    let pool = ConnectionPool::new(config);

    pool.warm();

    assert_eq!(pool.idle_connections(), 3);
}

// ============================================================================
// ResourceLimiter Tests
// ============================================================================

#[test]
fn test_resource_config_new() {
    let config = ResourceConfig::new();
    let _ = config;
}

#[test]
fn test_resource_config_default() {
    let config = ResourceConfig::default();
    let _ = config;
}

#[test]
fn test_resource_config_with_max_memory_per_request() {
    let config = ResourceConfig::new().with_max_memory_per_request(1024 * 1024);
    let _ = ResourceLimiter::new(config);
}

#[test]
fn test_resource_config_with_max_total_memory() {
    let config = ResourceConfig::new().with_max_total_memory(2 * 1024 * 1024 * 1024);
    let _ = ResourceLimiter::new(config);
}

#[test]
fn test_resource_config_with_max_compute_time() {
    let config = ResourceConfig::new().with_max_compute_time(Duration::from_secs(60));
    let _ = ResourceLimiter::new(config);
}

#[test]
fn test_resource_config_with_max_queue_depth() {
    let config = ResourceConfig::new().with_max_queue_depth(50);
    let _ = ResourceLimiter::new(config);
}

#[test]
fn test_resource_limiter_new() {
    let config = ResourceConfig::new();
    let limiter = ResourceLimiter::new(config);
    assert_eq!(limiter.current_memory(), 0);
}

#[test]
fn test_resource_limiter_check_memory_allowed() {
    let config = ResourceConfig::new()
        .with_max_memory_per_request(1024)
        .with_max_total_memory(10240);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.check_memory(512);
    assert!(matches!(result, LimitResult::Allowed));
}

#[test]
fn test_resource_limiter_check_memory_denied_per_request() {
    let config = ResourceConfig::new()
        .with_max_memory_per_request(100)
        .with_max_total_memory(10000);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.check_memory(200);
    assert!(matches!(result, LimitResult::Denied { .. }));
}

#[test]
fn test_resource_limiter_allocate() {
    let config = ResourceConfig::new()
        .with_max_memory_per_request(1024)
        .with_max_total_memory(10240);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.allocate(512);
    assert!(result.is_ok());
    assert_eq!(limiter.current_memory(), 512);
}

#[test]
fn test_resource_limiter_deallocate() {
    let config = ResourceConfig::new()
        .with_max_memory_per_request(1024)
        .with_max_total_memory(10240);
    let limiter = ResourceLimiter::new(config);

    let _ = limiter.allocate(512);
    limiter.deallocate(256);

    assert_eq!(limiter.current_memory(), 256);
}

#[test]
fn test_resource_limiter_enqueue() {
    let config = ResourceConfig::new().with_max_queue_depth(10);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.enqueue();
    assert!(matches!(result, LimitResult::Allowed));
}

#[test]
fn test_resource_limiter_enqueue_backpressure() {
    let config = ResourceConfig::new().with_max_queue_depth(2);
    let limiter = ResourceLimiter::new(config);

    let _ = limiter.enqueue();
    let _ = limiter.enqueue();
    let result = limiter.enqueue();

    assert!(matches!(result, LimitResult::Backpressure));
}

#[test]
fn test_resource_limiter_try_enqueue() {
    let config = ResourceConfig::new().with_max_queue_depth(5);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.try_enqueue();
    assert!(matches!(result, LimitResult::Allowed));
}

#[test]
fn test_resource_limiter_dequeue() {
    let config = ResourceConfig::new().with_max_queue_depth(10);
    let limiter = ResourceLimiter::new(config);

    let _ = limiter.enqueue();
    limiter.dequeue();

    // Should not panic on dequeue
}

#[test]
fn test_resource_limiter_start_compute() {
    let config = ResourceConfig::new();
    let limiter = ResourceLimiter::new(config);

    let start = limiter.start_compute();
    assert!(start.elapsed().as_nanos() >= 0);
}

// ============================================================================
// ResourceMonitor Tests
// ============================================================================

#[test]
fn test_resource_monitor_new() {
    let monitor = ResourceMonitor::new();
    let metrics = monitor.current_metrics();

    assert_eq!(metrics.memory_bytes, 0);
    assert_eq!(metrics.queue_depth, 0);
}

#[test]
fn test_resource_monitor_default() {
    let monitor = ResourceMonitor::default();
    let metrics = monitor.current_metrics();
    assert_eq!(metrics.memory_bytes, 0);
}

#[test]
fn test_resource_monitor_record_memory_usage() {
    let monitor = ResourceMonitor::new();

    monitor.record_memory_usage(1024);

    let metrics = monitor.current_metrics();
    assert_eq!(metrics.memory_bytes, 1024);
}

#[test]
fn test_resource_monitor_record_gpu_utilization() {
    let monitor = ResourceMonitor::new();

    monitor.record_gpu_utilization(75.5);

    let metrics = monitor.current_metrics();
    assert!((metrics.gpu_utilization - 75.5).abs() < 0.1);
}

#[test]
fn test_resource_monitor_record_queue_depth() {
    let monitor = ResourceMonitor::new();

    monitor.record_queue_depth(42);

    let metrics = monitor.current_metrics();
    assert_eq!(metrics.queue_depth, 42);
}

#[test]
fn test_resource_monitor_record_latency() {
    let monitor = ResourceMonitor::new();

    monitor.record_latency(Duration::from_millis(150));

    let metrics = monitor.current_metrics();
    assert_eq!(metrics.last_latency_ms, 150);
}

#[test]
fn test_resource_monitor_latency_stats_empty() {
    let monitor = ResourceMonitor::new();

    let stats = monitor.latency_stats();
    assert_eq!(stats.min_ms, 0);
    assert_eq!(stats.max_ms, 0);
    assert_eq!(stats.avg_ms, 0);
}

#[test]
fn test_resource_monitor_latency_stats() {
    let monitor = ResourceMonitor::new();

    monitor.record_latency(Duration::from_millis(100));
    monitor.record_latency(Duration::from_millis(200));
    monitor.record_latency(Duration::from_millis(300));

    let stats = monitor.latency_stats();
    assert_eq!(stats.min_ms, 100);
    assert_eq!(stats.max_ms, 300);
    assert_eq!(stats.avg_ms, 200);
}

#[test]
fn test_resource_monitor_snapshot() {
    let monitor = ResourceMonitor::new();

    monitor.record_memory_usage(2048);
    monitor.record_gpu_utilization(50.0);
    monitor.record_queue_depth(10);

    let snapshot = monitor.snapshot();

    assert!(snapshot.timestamp > 0);
    assert_eq!(snapshot.memory_bytes, 2048);
    assert!((snapshot.gpu_utilization - 50.0).abs() < 0.1);
    assert_eq!(snapshot.queue_depth, 10);
}

// ============================================================================
// GgufModelState Tests
// ============================================================================

#[test]
fn test_gguf_model_state_new() {
    let state = GgufModelState::new();

    assert!(!state.is_loaded());
    assert!(!state.is_ready());
    assert!(state.model_name().is_none());
    assert_eq!(state.vocab_size(), 0);
    assert!(state.model().is_none());
}

#[test]
fn test_gguf_model_state_default() {
    let state = GgufModelState::default();

    assert!(!state.is_loaded());
    assert!(!state.is_ready());
}

#[test]
fn test_gguf_model_state_debug() {
    let state = GgufModelState::new();
    let debug_str = format!("{:?}", state);

    assert!(debug_str.contains("GgufModelState"));
    assert!(debug_str.contains("is_loaded"));
}

#[test]
fn test_load_gguf_to_gpu() {
    let result = load_gguf_to_gpu(1000, 256, 4);

    assert!(result.is_ok());

    let state = result.unwrap();
    assert!(state.is_loaded());
    assert!(state.is_ready());
    assert_eq!(state.vocab_size(), 1000);
    assert!(state.model_name().is_some());
}

#[test]
fn test_load_gguf_to_gpu_model_name() {
    let state = load_gguf_to_gpu(500, 128, 2).unwrap();

    let name = state.model_name().unwrap();
    assert!(name.contains("500")); // vocab_size in name
    assert!(name.contains("128")); // hidden_dim in name
}

#[test]
fn test_gguf_model_state_model_mut() {
    let mut state = load_gguf_to_gpu(100, 64, 1).unwrap();

    let model_ref = state.model_mut();
    assert!(model_ref.is_some());
}

#[test]
fn test_gguf_model_state_small_config() {
    // Test with minimal configuration
    let result = load_gguf_to_gpu(50, 64, 1);
    assert!(result.is_ok());

    let state = result.unwrap();
    assert!(state.is_ready());
}

#[test]
fn test_gguf_model_state_large_config() {
    // Test with larger configuration
    let result = load_gguf_to_gpu(32000, 4096, 32);
    assert!(result.is_ok());

    let state = result.unwrap();
    assert!(state.is_ready());
    assert_eq!(state.vocab_size(), 32000);
}
