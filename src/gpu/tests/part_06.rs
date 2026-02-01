//! Phase 48 - Batch Scenario Tests (gpu/scheduler/batch.rs coverage)
//!
//! Tests for:
//! - `generate_gpu`: Token generation with MockExecutor
//! - `forward_single_token`: Single token forward pass
//! - `forward_single_token_greedy`: Optimized greedy sampling
//! - `forward_block_single`: Single token through transformer block
//! - `optimized_gqa_attention`: GQA attention implementation
//!
//! Strategy: Use MockExecutor to test CPU logic paths without GPU hardware.

use crate::gpu::executor::MockExecutor;
use crate::gpu::scheduler::batch::{
    argmax, forward_block_single, forward_single_token, forward_single_token_greedy, generate_gpu,
    optimized_gqa_attention, optimized_lm_head_argmax_transposed, simplified_attention,
};
use crate::gpu::scheduler::{GpuModel, GpuModelConfig};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a minimal test model configuration
fn create_test_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4, // MHA (not GQA)
        vocab_size: 100,
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}

/// Create a GQA test configuration (num_heads > num_kv_heads)
fn create_gqa_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 2, // GQA: 4 Q heads per KV head
        vocab_size: 100,
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}

/// Create a small vocab config (uses GPU path in forward_single_token)
fn create_small_vocab_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50, // Small vocab triggers GPU path
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}

/// Create a large vocab config (uses CPU path in forward_single_token)
fn create_large_vocab_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 10000, // Large vocab triggers CPU path
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}

// ============================================================================
// forward_single_token Tests
// ============================================================================

#[test]
fn test_forward_single_token_basic() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Install MockExecutor
    let mock = MockExecutor::new("test_forward_single");
    model.with_test_executor(Box::new(mock));

    let tokens = vec![1, 2, 3];
    let result = forward_single_token(&mut model, &tokens);

    assert!(result.is_ok(), "forward_single_token should succeed");
    let logits = result.unwrap();
    assert_eq!(logits.len(), model.config.vocab_size);
}

#[test]
fn test_forward_single_token_empty_tokens() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let tokens: Vec<usize> = vec![];
    let result = forward_single_token(&mut model, &tokens);

    assert!(result.is_err(), "empty tokens should fail");
}

#[test]
fn test_forward_single_token_out_of_bounds() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Token 9999 is out of bounds for vocab_size=50
    let tokens = vec![1, 2, 9999];
    let result = forward_single_token(&mut model, &tokens);

    assert!(result.is_err(), "out of bounds token should fail");
}

#[test]
fn test_forward_single_token_single_token() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_single_token");
    model.with_test_executor(Box::new(mock));

    let tokens = vec![5];
    let result = forward_single_token(&mut model, &tokens);

    assert!(result.is_ok());
}

#[test]
fn test_forward_single_token_with_mock_calls() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_mock_calls");
    model.with_test_executor(Box::new(mock));

    let tokens = vec![1, 2];
    let _ = forward_single_token(&mut model, &tokens);

    // Verify mock was used (small vocab triggers GPU matmul)
    assert!(model.has_test_executor());
}

// ============================================================================
// forward_single_token_greedy Tests
// ============================================================================

#[test]
fn test_forward_single_token_greedy_basic() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_greedy");
    model.with_test_executor(Box::new(mock));

    let tokens = vec![1, 2, 3];
    let result = forward_single_token_greedy(&mut model, &tokens);

    assert!(result.is_ok());
    let next_token = result.unwrap();
    assert!(next_token < model.config.vocab_size);
}

#[test]
fn test_forward_single_token_greedy_empty() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let tokens: Vec<usize> = vec![];
    let result = forward_single_token_greedy(&mut model, &tokens);

    assert!(result.is_err());
}

#[test]
fn test_forward_single_token_greedy_out_of_bounds() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let tokens = vec![99999]; // Out of bounds
    let result = forward_single_token_greedy(&mut model, &tokens);

    assert!(result.is_err());
}

#[test]
fn test_forward_single_token_greedy_large_vocab() {
    // Large vocab triggers CPU path
    let config = create_large_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let tokens = vec![1, 2, 3];
    let result = forward_single_token_greedy(&mut model, &tokens);

    assert!(result.is_ok());
    let next_token = result.unwrap();
    assert!(next_token < model.config.vocab_size);
}

// ============================================================================
// forward_block_single Tests
// ============================================================================

#[test]
fn test_forward_block_single_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let input = vec![0.1f32; model.config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), model.config.hidden_dim);
}

#[test]
fn test_forward_block_single_all_layers() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let mut hidden = vec![0.1f32; config.hidden_dim];

    for block_idx in 0..config.num_layers {
        let result = forward_block_single(&mut model, &hidden, block_idx);
        assert!(result.is_ok(), "block {} should succeed", block_idx);
        hidden = result.unwrap();
        assert_eq!(hidden.len(), config.hidden_dim);
    }
}

#[test]
fn test_forward_block_single_gqa() {
    // Test GQA path (num_heads > num_kv_heads)
    let config = create_gqa_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let input = vec![0.1f32; model.config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), model.config.hidden_dim);
}

#[test]
fn test_forward_block_single_zeros() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let input = vec![0.0f32; model.config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);

    assert!(result.is_ok());
}

#[test]
fn test_forward_block_single_large_values() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let input = vec![100.0f32; model.config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);

    assert!(result.is_ok());
}

// ============================================================================
// generate_gpu Tests
// ============================================================================

#[test]
fn test_generate_gpu_basic() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_generate");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![1, 2];
    let result = generate_gpu(&mut model, &prompt, 3);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    // Should have prompt + max_tokens (minus initial prediction)
    assert!(tokens.len() >= prompt.len());
}

#[test]
fn test_generate_gpu_single_token() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_generate_single");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![5];
    let result = generate_gpu(&mut model, &prompt, 1);

    assert!(result.is_ok());
}

#[test]
fn test_generate_gpu_large_vocab() {
    // Large vocab uses different code path
    let config = create_large_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let prompt = vec![1, 2];
    let result = generate_gpu(&mut model, &prompt, 3);

    assert!(result.is_ok());
}

#[test]
fn test_generate_gpu_zero_max_tokens() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_zero_tokens");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![1];
    let result = generate_gpu(&mut model, &prompt, 0);

    assert!(result.is_ok());
}

// ============================================================================
// optimized_gqa_attention Tests
// ============================================================================

#[test]
fn test_optimized_gqa_attention_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let mock = MockExecutor::new("test_gqa_attention");
    model.with_test_executor(Box::new(mock));

    let seq_len = 4;
    let hidden_dim = config.hidden_dim;
    let kv_dim = config.kv_dim();

    // QKV size for MHA: seq_len * (hidden_dim + 2*kv_dim) = seq_len * 3*hidden_dim for MHA
    let qkv = vec![0.1f32; seq_len * (hidden_dim + 2 * kv_dim)];

    let result = optimized_gqa_attention(&mut model, &qkv, seq_len);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), seq_len * hidden_dim);
}

#[test]
fn test_optimized_gqa_attention_single_token() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let mock = MockExecutor::new("test_gqa_single");
    model.with_test_executor(Box::new(mock));

    let seq_len = 1;
    let hidden_dim = config.hidden_dim;
    let kv_dim = config.kv_dim();

    let qkv = vec![0.1f32; seq_len * (hidden_dim + 2 * kv_dim)];

    let result = optimized_gqa_attention(&mut model, &qkv, seq_len);

    assert!(result.is_ok());
}

#[test]
fn test_optimized_gqa_attention_gqa_mode() {
    // Test with actual GQA (num_heads > num_kv_heads)
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let mock = MockExecutor::new("test_gqa_mode");
    model.with_test_executor(Box::new(mock));

    let seq_len = 4;
    let hidden_dim = config.hidden_dim;
    let kv_dim = config.kv_dim();

    // GQA: Q has hidden_dim, K/V have kv_dim
    let qkv = vec![0.1f32; seq_len * (hidden_dim + 2 * kv_dim)];

    let result = optimized_gqa_attention(&mut model, &qkv, seq_len);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), seq_len * hidden_dim);
}

// ============================================================================
// argmax Additional Coverage Tests
// ============================================================================

#[test]
fn test_argmax_empty() {
    let logits: Vec<f32> = vec![];
    let result = argmax(&logits);
    assert_eq!(result, 0); // Default for empty
}

#[test]
fn test_argmax_boundary_1024() {
    // Exactly at boundary where chunking kicks in
    let mut logits = vec![0.0f32; 1024];
    logits[512] = 1.0;
    assert_eq!(argmax(&logits), 512);
}

#[test]
fn test_argmax_boundary_1025() {
    // Just over boundary - triggers chunked path
    let mut logits = vec![0.0f32; 1025];
    logits[1024] = 1.0;
    assert_eq!(argmax(&logits), 1024);
}

#[test]
fn test_argmax_large_8192() {
    let mut logits = vec![0.0f32; 8192];
    logits[4000] = 1.0;
    assert_eq!(argmax(&logits), 4000);
}

// ============================================================================
// optimized_lm_head_argmax_transposed Additional Tests
// ============================================================================

#[test]
fn test_optimized_lm_head_argmax_deterministic() {
    let hidden_dim = 64;
    let vocab_size = 1000;

    let hidden = vec![0.1f32; hidden_dim];
    let weight_t = vec![0.01f32; vocab_size * hidden_dim];
    let bias = vec![0.0f32; vocab_size];

    // Run multiple times to verify determinism
    let r1 = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    let r2 = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);

    assert_eq!(r1, r2, "results should be deterministic");
}

#[test]
fn test_optimized_lm_head_argmax_with_varied_bias() {
    let hidden_dim = 32;
    let vocab_size = 100;

    let hidden = vec![0.1f32; hidden_dim];
    let weight_t = vec![0.0f32; vocab_size * hidden_dim]; // Zero weights

    // Bias determines the winner
    let mut bias = vec![0.0f32; vocab_size];
    bias[42] = 10.0;

    let result =
        optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    assert_eq!(result, 42);
}

// ============================================================================
// simplified_attention Additional Tests
// ============================================================================

#[test]
fn test_simplified_attention_multiple_heads() {
    let config = GpuModelConfig {
        hidden_dim: 64,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        intermediate_dim: 128,
        num_layers: 1,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let seq_len = 4;
    // MHA: qkv has 3 * hidden_dim per position
    let qkv = vec![0.1f32; seq_len * 3 * config.hidden_dim];

    let result = simplified_attention(&config, &qkv, seq_len);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), seq_len * config.hidden_dim);
}

#[test]
fn test_simplified_attention_longer_sequence() {
    let config = create_test_config();
    let seq_len = 16;
    let qkv = vec![0.1f32; seq_len * 3 * config.hidden_dim];

    let result = simplified_attention(&config, &qkv, seq_len);
    assert!(result.is_ok());
}

// ============================================================================
// GpuModelConfig Additional Coverage
// ============================================================================

#[test]
fn test_gpu_model_config_kv_dim_mha() {
    let config = create_test_config();
    // MHA: kv_dim = num_kv_heads * head_dim = num_kv_heads * (hidden_dim / num_heads)
    // For our config: 4 * (64/4) = 64 = hidden_dim
    assert_eq!(config.kv_dim(), config.hidden_dim);
}

#[test]
fn test_gpu_model_config_kv_dim_gqa() {
    let config = create_gqa_config();
    // GQA: 2 * (64/8) = 2 * 8 = 16
    let expected_kv_dim = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    assert_eq!(config.kv_dim(), expected_kv_dim);
}

#[test]
fn test_gpu_model_config_qkv_dim_mha() {
    let config = create_test_config();
    // MHA: qkv_dim = hidden_dim + 2*kv_dim = 3*hidden_dim
    assert_eq!(config.qkv_dim(), 3 * config.hidden_dim);
}

#[test]
fn test_gpu_model_config_qkv_dim_gqa() {
    let config = create_gqa_config();
    // GQA: qkv_dim = hidden_dim + 2*kv_dim = 64 + 2*16 = 96
    let expected = config.hidden_dim + 2 * config.kv_dim();
    assert_eq!(config.qkv_dim(), expected);
}

#[test]
fn test_gpu_model_config_head_dim() {
    let config = create_test_config();
    assert_eq!(config.head_dim(), config.hidden_dim / config.num_heads);
}

// ============================================================================
// GpuModel with MockExecutor Tests
// ============================================================================

#[test]
fn test_gpu_model_with_test_executor() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    assert!(!model.has_test_executor());

    let mock = MockExecutor::new("test");
    model.with_test_executor(Box::new(mock));

    assert!(model.has_test_executor());
}

#[test]
fn test_gpu_model_clear_test_executor() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test");
    model.with_test_executor(Box::new(mock));
    assert!(model.has_test_executor());

    model.clear_test_executor();
    assert!(!model.has_test_executor());
}

#[test]
fn test_mock_executor_failure() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Create mock that fails on matmul
    let mock = MockExecutor::new("failing").with_matmul_failure();
    model.with_test_executor(Box::new(mock));

    let tokens = vec![1, 2];
    let result = forward_single_token(&mut model, &tokens);

    // Should fail due to mock failure
    assert!(result.is_err());
}

// ============================================================================
// GpuGenerateConfig Tests
// ============================================================================

use crate::gpu::scheduler::GpuGenerateConfig;

#[test]
fn test_gpu_generate_config_default() {
    let config = GpuGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_deterministic() {
    let config = GpuGenerateConfig::deterministic(100);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_gpu_generate_config_with_sampling() {
    let config = GpuGenerateConfig::with_sampling(50, 0.7, 40);
    assert_eq!(config.max_tokens, 50);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
}

#[test]
fn test_gpu_generate_config_with_stop_tokens() {
    let config = GpuGenerateConfig::deterministic(32).with_stop_tokens(vec![1, 2, 50256]);
    assert_eq!(config.stop_tokens, vec![1, 2, 50256]);
}

#[test]
fn test_gpu_generate_config_chained() {
    let config = GpuGenerateConfig::with_sampling(128, 0.9, 50).with_stop_tokens(vec![0, 1]);

    assert_eq!(config.max_tokens, 128);
    assert_eq!(config.temperature, 0.9);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.stop_tokens, vec![0, 1]);
}

// ============================================================================
// AttentionBuffers Tests
// ============================================================================

use crate::gpu::scheduler::AttentionBuffers;

#[test]
fn test_attention_buffers_new() {
    let config = create_test_config();
    let max_seq_len = 128;

    let buffers = AttentionBuffers::new(&config, max_seq_len);

    assert_eq!(buffers.q_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.scores_buffer.len(), config.num_heads * max_seq_len);
    assert_eq!(buffers.output_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.kv_proj_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.ffn_buffer.len(), config.intermediate_dim);
    assert_eq!(buffers.max_seq_len, max_seq_len);
}

#[test]
fn test_attention_buffers_reset() {
    let config = create_test_config();
    let mut buffers = AttentionBuffers::new(&config, 64);

    // Fill with non-zero values
    buffers.q_buffer.fill(1.0);
    buffers.scores_buffer.fill(2.0);
    buffers.output_buffer.fill(3.0);
    buffers.kv_proj_buffer.fill(4.0);
    buffers.ffn_buffer.fill(5.0);

    // Reset should zero everything
    buffers.reset();

    assert!(buffers.q_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.scores_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.output_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.kv_proj_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.ffn_buffer.iter().all(|&x| x == 0.0));
}

#[test]
fn test_attention_buffers_gqa_config() {
    let config = create_gqa_config();
    let max_seq_len = 256;

    let buffers = AttentionBuffers::new(&config, max_seq_len);

    // GQA config has 8 heads
    assert_eq!(buffers.scores_buffer.len(), 8 * max_seq_len);
}

// ============================================================================
// WeightType and matmul_split Tests
// ============================================================================

use crate::gpu::scheduler::WeightType;

#[test]
fn test_weight_type_enum() {
    // Verify all variants exist and can be cloned/debugged
    let qkv = WeightType::Qkv;
    let output = WeightType::Output;
    let fc1 = WeightType::FfnFc1;
    let fc2 = WeightType::FfnFc2;
    let lm_head = WeightType::LmHead;

    // Clone test
    let _cloned = qkv;

    // Debug test
    let _ = format!("{:?}", output);
    let _ = format!("{:?}", fc1);
    let _ = format!("{:?}", fc2);
    let _ = format!("{:?}", lm_head);
}

#[test]
fn test_matmul_split_qkv() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::Qkv);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.qkv_dim());
}

#[test]
fn test_matmul_split_output() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::Output);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_matmul_split_ffn_fc1() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::FfnFc1);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.intermediate_dim);
}

#[test]
fn test_matmul_split_ffn_fc2() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.intermediate_dim];
    let result = model.matmul_split(&input, 0, WeightType::FfnFc2);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_matmul_split_lm_head() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::LmHead);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.vocab_size);
}

#[test]
fn test_matmul_split_all_layers() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];

    for layer_idx in 0..config.num_layers {
        let result = model.matmul_split(&input, layer_idx, WeightType::Qkv);
        assert!(result.is_ok(), "layer {} QKV should work", layer_idx);
    }
}

// ============================================================================
// do_matmul and do_matmul_transpose_b Tests
// ============================================================================

#[test]
fn test_do_matmul_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let a = vec![1.0f32; 64];
    let b = vec![0.1f32; 64 * 128];

    let result = model.do_matmul(&a, &b, 1, 64, 128);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 128);
}

#[test]
fn test_do_matmul_with_test_executor() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("matmul_test");
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 32];
    let b = vec![0.1f32; 32 * 64];

    let result = model.do_matmul(&a, &b, 1, 32, 64);
    assert!(result.is_ok());
}

#[test]
fn test_do_matmul_transpose_b() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let a = vec![1.0f32; 32];
    // b is transposed: [n, k] = [64, 32]
    let b = vec![0.1f32; 64 * 32];

    let result = model.do_matmul_transpose_b(&a, &b, 1, 32, 64);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 64);
}

#[test]
fn test_do_matmul_transpose_b_with_mock() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("transpose_test");
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 16];
    let b = vec![0.1f32; 32 * 16];

    let result = model.do_matmul_transpose_b(&a, &b, 1, 16, 32);
    assert!(result.is_ok());
}

// ============================================================================
// GpuModelConfig Additional Tests
// ============================================================================

#[test]
fn test_gpu_model_config_is_gqa_true() {
    let config = create_gqa_config();
    assert!(config.is_gqa());
}

#[test]
fn test_gpu_model_config_is_gqa_false() {
    let config = create_test_config();
    assert!(!config.is_gqa());
}

#[test]
fn test_gpu_model_config_edge_case_single_head() {
    let config = GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 1000,
        eps: 1e-6,
        rope_theta: 10000.0,
    };

    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.kv_dim(), 64);
    assert!(!config.is_gqa());
}

// ============================================================================
// BlockWeights Tests
// ============================================================================

use crate::gpu::scheduler::BlockWeights;

#[test]
fn test_block_weights_construction() {
    let weights = BlockWeights {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: vec![0.0; 64],
        qkv_weight: vec![0.01; 64 * 192],
        qkv_bias: vec![0.0; 192],
        out_weight: vec![0.01; 64 * 64],
        out_bias: vec![0.0; 64],
        ffn_norm_weight: vec![1.0; 64],
        ffn_norm_bias: vec![0.0; 64],
        ffn_fc1_weight: vec![0.01; 64 * 256],
        ffn_fc1_bias: vec![0.0; 256],
        ffn_fc2_weight: vec![0.01; 256 * 64],
        ffn_fc2_bias: vec![0.0; 64],
        ffn_gate_weight: None,
    };

    assert_eq!(weights.attn_norm_weight.len(), 64);
    assert!(weights.ffn_gate_weight.is_none());
}

#[test]
fn test_block_weights_with_gate() {
    let weights = BlockWeights {
        attn_norm_weight: vec![1.0; 32],
        attn_norm_bias: vec![0.0; 32],
        qkv_weight: vec![0.01; 32 * 96],
        qkv_bias: vec![0.0; 96],
        out_weight: vec![0.01; 32 * 32],
        out_bias: vec![0.0; 32],
        ffn_norm_weight: vec![1.0; 32],
        ffn_norm_bias: vec![0.0; 32],
        ffn_fc1_weight: vec![0.01; 32 * 128],
        ffn_fc1_bias: vec![0.0; 128],
        ffn_fc2_weight: vec![0.01; 128 * 32],
        ffn_fc2_bias: vec![0.0; 32],
        ffn_gate_weight: Some(vec![0.01; 32 * 128]), // SwiGLU gate
    };

    assert!(weights.ffn_gate_weight.is_some());
    assert_eq!(weights.ffn_gate_weight.as_ref().unwrap().len(), 32 * 128);
}

// ============================================================================
// KV Cache Forward Pass Tests (gpu/scheduler/kv.rs)
// ============================================================================

use crate::gpu::scheduler::{forward_gpu_incremental, forward_gpu_with_cache, generate_with_cache};
use crate::gpu::StreamingKVCache;

fn create_kv_cache_for_model(config: &GpuModelConfig, max_seq_len: usize) -> StreamingKVCache {
    let head_dim = config.hidden_dim / config.num_heads;
    StreamingKVCache::new(
        config.num_layers,
        max_seq_len,
        config.num_kv_heads,
        head_dim,
    )
}

#[test]
fn test_forward_gpu_with_cache_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    let tokens = vec![1, 2, 3];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_gpu_with_cache_single_token() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 32);

    let tokens = vec![5];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_ok());
}

#[test]
fn test_forward_gpu_with_cache_empty_tokens() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 32);

    let tokens: Vec<usize> = vec![];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_with_cache_out_of_bounds() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 32);

    // Token 9999 exceeds vocab_size (100)
    let tokens = vec![9999];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_with_cache_gqa() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    let tokens = vec![1, 2, 3, 4];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_ok());
}

// ============================================================================
// Forward GPU Incremental Tests
// ============================================================================

#[test]
fn test_forward_gpu_incremental_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    // First, populate cache with initial tokens
    let _ = forward_gpu_with_cache(&mut model, &[1, 2], &mut kv_cache);

    // Then do incremental forward for new token
    let result = forward_gpu_incremental(&mut model, 3, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_gpu_incremental_multiple() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    // Populate cache
    let _ = forward_gpu_with_cache(&mut model, &[1], &mut kv_cache);

    // Multiple incremental forwards
    for token in [2, 3, 4, 5] {
        let result = forward_gpu_incremental(&mut model, token, &mut kv_cache);
        assert!(result.is_ok(), "token {} forward should work", token);
    }
}

#[test]
fn test_forward_gpu_incremental_out_of_bounds() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 32);

    // Populate cache first
    let _ = forward_gpu_with_cache(&mut model, &[1], &mut kv_cache);

    // Token out of bounds
    let result = forward_gpu_incremental(&mut model, 9999, &mut kv_cache);
    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_incremental_gqa() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    // Populate cache
    let _ = forward_gpu_with_cache(&mut model, &[1, 2, 3], &mut kv_cache);

    // Incremental forward
    let result = forward_gpu_incremental(&mut model, 4, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// Generate with Cache Tests
// ============================================================================

#[test]
fn test_generate_with_cache_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(5);
    let prompt = vec![1, 2];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(tokens.len() >= prompt.len()); // At least prompt length
}

#[test]
fn test_generate_with_cache_empty_prompt() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(5);
    let prompt: Vec<usize> = vec![];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_err());
}

#[test]
fn test_generate_with_cache_with_stop_tokens() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Stop on token 0 (which might be generated from uniform random weights)
    let gen_config = GpuGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);
    let prompt = vec![1];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_sampling() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Use sampling with temperature
    let gen_config = GpuGenerateConfig::with_sampling(3, 0.5, 10);
    let prompt = vec![1, 2, 3];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_zero_max_tokens() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(0);
    let prompt = vec![1, 2];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
    // Should at least return prompt + 1 token (or just prompt if first token is stop)
}

#[test]
fn test_generate_with_cache_gqa() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(3);
    let prompt = vec![1, 2];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_single_token_prompt() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(2);
    let prompt = vec![5];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// StreamingKVCache Tests
// ============================================================================

#[test]
fn test_streaming_kv_cache_new() {
    let num_layers = 4;
    let max_seq_len = 128;
    let num_kv_heads = 4;
    let head_dim = 16;

    let cache = StreamingKVCache::new(num_layers, max_seq_len, num_kv_heads, head_dim);

    // Cache should be created and initially empty
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_positions(), max_seq_len);
}

#[test]
fn test_streaming_kv_cache_append_and_get() {
    let cache_config = create_test_config();
    let head_dim = cache_config.hidden_dim / cache_config.num_heads;
    let kv_dim = cache_config.num_kv_heads * head_dim;

    let mut cache = StreamingKVCache::new(
        cache_config.num_layers,
        64,
        cache_config.num_kv_heads,
        head_dim,
    );

    // Append K/V data to ALL layers (position only increments after last layer)
    let k = vec![0.1f32; kv_dim];
    let v = vec![0.2f32; kv_dim];

    for layer in 0..cache_config.num_layers {
        cache.append(layer, &k, &v);
    }

    // Now cache has 1 valid position
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);

    // Get cached data for layer 0
    let (cached_k, cached_v) = cache.get_valid(0);
    assert_eq!(cached_k.len(), kv_dim);
    assert_eq!(cached_v.len(), kv_dim);
}

#[test]
fn test_streaming_kv_cache_multiple_layers() {
    let config = create_test_config();
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;

    let mut cache = StreamingKVCache::new(config.num_layers, 32, config.num_kv_heads, head_dim);

    let k = vec![0.1f32; kv_dim];
    let v = vec![0.2f32; kv_dim];

    // Append to all layers
    for layer in 0..config.num_layers {
        cache.append(layer, &k, &v);
    }

    // Verify each layer has data
    for layer in 0..config.num_layers {
        let (cached_k, _) = cache.get_valid(layer);
        assert_eq!(cached_k.len(), kv_dim);
    }
}

#[test]
fn test_streaming_kv_cache_clear() {
    let config = create_test_config();
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;

    let mut cache = StreamingKVCache::new(config.num_layers, 32, config.num_kv_heads, head_dim);

    // Populate cache - must append to ALL layers for position to increment
    let k = vec![0.1f32; kv_dim];
    let v = vec![0.2f32; kv_dim];
    for layer in 0..config.num_layers {
        cache.append(layer, &k, &v);
    }
    assert!(!cache.is_empty());

    // Clear
    cache.clear();

    // Should be empty after clear
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}
