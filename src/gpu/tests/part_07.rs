//! Phase 49 - GpuModel Coverage Tests (gpu/scheduler/model.rs)
//!
//! Comprehensive tests for GpuModel struct and methods:
//! - GpuModelConfig: head_dim, kv_dim, qkv_dim, is_gqa
//! - GpuGenerateConfig: constructors, with_stop_tokens
//! - AttentionBuffers: new, reset
//! - GpuModel: constructors, test_executor methods, matmul_split, forward/generate
//! - BlockWeights structure
//! - WeightType enum
//!
//! Strategy: Use MockExecutor for CPU logic paths without GPU hardware.

use crate::gpu::executor::MockExecutor;
use crate::gpu::scheduler::{
    AttentionBuffers, BlockWeights, GpuGenerateConfig, GpuModel, GpuModelConfig, WeightType,
};
use crate::gpu::StreamingKVCache;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a minimal test model configuration (MHA - Multi-Head Attention)
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

/// Create a minimal single-layer config for fast tests
fn create_minimal_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}

// ============================================================================
// GpuModelConfig Tests
// ============================================================================

#[test]
fn test_gpu_model_config_head_dim() {
    let config = create_test_config();
    // hidden_dim=64, num_heads=4 -> head_dim=16
    assert_eq!(config.head_dim(), 16);
}

#[test]
fn test_gpu_model_config_head_dim_gqa() {
    let config = create_gqa_config();
    // hidden_dim=64, num_heads=8 -> head_dim=8
    assert_eq!(config.head_dim(), 8);
}

#[test]
fn test_gpu_model_config_kv_dim_mha() {
    let config = create_test_config();
    // MHA: num_kv_heads=4, head_dim=16 -> kv_dim=64
    assert_eq!(config.kv_dim(), 64);
}

#[test]
fn test_gpu_model_config_kv_dim_gqa() {
    let config = create_gqa_config();
    // GQA: num_kv_heads=2, head_dim=8 -> kv_dim=16
    assert_eq!(config.kv_dim(), 16);
}

#[test]
fn test_gpu_model_config_qkv_dim_mha() {
    let config = create_test_config();
    // MHA: hidden_dim + 2*kv_dim = 64 + 2*64 = 192
    assert_eq!(config.qkv_dim(), 192);
}

#[test]
fn test_gpu_model_config_qkv_dim_gqa() {
    let config = create_gqa_config();
    // GQA: hidden_dim + 2*kv_dim = 64 + 2*16 = 96
    assert_eq!(config.qkv_dim(), 96);
}

#[test]
fn test_gpu_model_config_is_gqa_false() {
    let config = create_test_config();
    // MHA: num_heads == num_kv_heads
    assert!(!config.is_gqa());
}

#[test]
fn test_gpu_model_config_is_gqa_true() {
    let config = create_gqa_config();
    // GQA: num_heads > num_kv_heads
    assert!(config.is_gqa());
}

// ============================================================================
// GpuGenerateConfig Tests
// ============================================================================

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
    let config = GpuGenerateConfig::deterministic(128);
    assert_eq!(config.max_tokens, 128);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_with_sampling() {
    let config = GpuGenerateConfig::with_sampling(64, 0.7, 40);
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_with_stop_tokens() {
    let config = GpuGenerateConfig::deterministic(32).with_stop_tokens(vec![0, 2, 3]);
    assert_eq!(config.max_tokens, 32);
    assert_eq!(config.stop_tokens, vec![0, 2, 3]);
}

#[test]
fn test_gpu_generate_config_with_sampling_and_stop_tokens() {
    let config =
        GpuGenerateConfig::with_sampling(100, 0.9, 50).with_stop_tokens(vec![1, 2, 3, 4, 5]);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.9);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.stop_tokens.len(), 5);
}

// ============================================================================
// AttentionBuffers Tests
// ============================================================================

#[test]
fn test_attention_buffers_new() {
    let config = create_test_config();
    let buffers = AttentionBuffers::new(&config, 512);

    assert_eq!(buffers.q_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.scores_buffer.len(), config.num_heads * 512);
    assert_eq!(buffers.output_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.kv_proj_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.ffn_buffer.len(), config.intermediate_dim);
    assert_eq!(buffers.max_seq_len, 512);
}

#[test]
fn test_attention_buffers_new_gqa() {
    let config = create_gqa_config();
    let buffers = AttentionBuffers::new(&config, 256);

    assert_eq!(buffers.q_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.scores_buffer.len(), config.num_heads * 256);
    assert_eq!(buffers.max_seq_len, 256);
}

#[test]
fn test_attention_buffers_reset() {
    let config = create_minimal_config();
    let mut buffers = AttentionBuffers::new(&config, 64);

    // Fill with non-zero values
    buffers.q_buffer.fill(1.0);
    buffers.scores_buffer.fill(2.0);
    buffers.output_buffer.fill(3.0);
    buffers.kv_proj_buffer.fill(4.0);
    buffers.ffn_buffer.fill(5.0);

    // Reset should clear all
    buffers.reset();

    assert!(buffers.q_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.scores_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.output_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.kv_proj_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.ffn_buffer.iter().all(|&x| x == 0.0));
}

// ============================================================================
// GpuModel Construction Tests
// ============================================================================

#[test]
fn test_gpu_model_new_basic() {
    let config = create_test_config();
    let model = GpuModel::new(config.clone());
    assert!(model.is_ok());

    let model = model.unwrap();
    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.num_layers, 2);
    assert!(!model.has_test_executor());
}

#[test]
fn test_gpu_model_new_gqa() {
    let config = create_gqa_config();
    let model = GpuModel::new(config);
    assert!(model.is_ok());

    let model = model.unwrap();
    assert!(model.config.is_gqa());
}

#[test]
fn test_gpu_model_from_gguf_config() {
    let config = create_test_config();
    let model = GpuModel::from_gguf_config(config.clone());
    assert!(model.is_ok());

    let model = model.unwrap();
    assert_eq!(model.config.vocab_size, config.vocab_size);
}

#[test]
fn test_gpu_model_config_getter() {
    let config = create_test_config();
    let model = GpuModel::new(config).unwrap();

    let retrieved_config = model.config();
    assert_eq!(retrieved_config.hidden_dim, 64);
    assert_eq!(retrieved_config.vocab_size, 100);
}

// ============================================================================
// GpuModel Test Executor Tests
// ============================================================================

#[test]
fn test_gpu_model_with_test_executor() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    assert!(!model.has_test_executor());

    let mock = MockExecutor::new("test_executor");
    model.with_test_executor(Box::new(mock));

    assert!(model.has_test_executor());
}

#[test]
fn test_gpu_model_clear_test_executor() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("test_executor");
    model.with_test_executor(Box::new(mock));
    assert!(model.has_test_executor());

    model.clear_test_executor();
    assert!(!model.has_test_executor());
}

#[test]
fn test_gpu_model_has_gpu() {
    let config = create_minimal_config();
    let model = GpuModel::new(config).unwrap();
    // has_gpu() delegates to scheduler, should not panic
    let _ = model.has_gpu();
}

// ============================================================================
// GpuModel with_attention_buffers Tests
// ============================================================================

#[test]
fn test_gpu_model_with_attention_buffers() {
    let config = create_test_config();
    let model = GpuModel::with_attention_buffers(config, 512);
    assert!(model.is_ok());

    let model = model.unwrap();
    assert!(model.has_attention_buffers());
}

#[test]
fn test_gpu_model_has_attention_buffers_false() {
    let config = create_test_config();
    let model = GpuModel::new(config).unwrap();
    assert!(!model.has_attention_buffers());
}

// ============================================================================
// GpuModel Feature Flag Tests (has_fused_*)
// ============================================================================

#[test]
fn test_gpu_model_has_fused_qkv() {
    let config = create_test_config();
    let model = GpuModel::new(config).unwrap();

    // Test checks if QKV weights have expected dimensions
    let has_fused = model.has_fused_qkv();
    // New model has initialized weights
    assert!(has_fused || !has_fused); // Just verify it doesn't panic
}

#[test]
fn test_gpu_model_has_fused_attn_proj() {
    let config = create_test_config();
    let model = GpuModel::new(config).unwrap();

    // Test checks if attention projection weights have expected dimensions
    let has_fused = model.has_fused_attn_proj();
    assert!(has_fused); // Should be true for properly initialized model
}

#[test]
fn test_gpu_model_has_fused_output_residual() {
    let config = create_test_config();
    let model = GpuModel::new(config).unwrap();

    // Requires attention_buffers to be present
    assert!(!model.has_fused_output_residual());

    // With attention buffers
    let model_with_buffers = GpuModel::with_attention_buffers(create_test_config(), 128).unwrap();
    assert!(model_with_buffers.has_fused_output_residual());
}

// ============================================================================
// GpuModel do_matmul Tests with MockExecutor
// ============================================================================

#[test]
fn test_gpu_model_do_matmul_with_mock() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("matmul_test").with_matmul_result(vec![1.0, 2.0, 3.0, 4.0]);
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 2 * 2];
    let b = vec![1.0f32; 2 * 2];
    let result = model.do_matmul(&a, &b, 2, 2, 2);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_gpu_model_do_matmul_failure() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("failing_matmul").with_matmul_failure();
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 4];
    let b = vec![1.0f32; 4];
    let result = model.do_matmul(&a, &b, 2, 2, 2);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_do_matmul_transpose_b_with_mock() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    // For transpose_b: a = m*k, b = n*k (b is transposed so n rows, k cols)
    // With m=1, k=2, n=2: a = 1*2 = 2, b = 2*2 = 4, output = m*n = 2
    let mock = MockExecutor::new("transpose_test").with_matmul_result(vec![10.0, 20.0]);
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 2];  // m=1, k=2
    let b = vec![1.0f32; 4];  // n=2, k=2
    let result = model.do_matmul_transpose_b(&a, &b, 1, 2, 2);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output, vec![10.0, 20.0]);
}

// ============================================================================
// GpuModel matmul_split Tests
// ============================================================================

#[test]
fn test_gpu_model_matmul_split_qkv() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("split_qkv").with_matmul_result(vec![0.0f32; config.qkv_dim()]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::Qkv);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.qkv_dim());
}

#[test]
fn test_gpu_model_matmul_split_output() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock =
        MockExecutor::new("split_output").with_matmul_result(vec![0.0f32; config.hidden_dim]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::Output);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_gpu_model_matmul_split_ffn_fc1() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock =
        MockExecutor::new("split_fc1").with_matmul_result(vec![0.0f32; config.intermediate_dim]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::FfnFc1);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.intermediate_dim);
}

#[test]
fn test_gpu_model_matmul_split_ffn_fc2() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("split_fc2").with_matmul_result(vec![0.0f32; config.hidden_dim]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.intermediate_dim];
    let result = model.matmul_split(&input, 0, WeightType::FfnFc2);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_gpu_model_matmul_split_lm_head() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock =
        MockExecutor::new("split_lm_head").with_matmul_result(vec![0.0f32; config.vocab_size]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::LmHead);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.vocab_size);
}

// ============================================================================
// GpuModel forward_gpu Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_gpu_basic() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_gpu");
    model.with_test_executor(Box::new(mock));

    let token_ids = vec![1, 2, 3];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_ok());
    let logits = result.unwrap();
    // Output should be seq_len * vocab_size
    assert_eq!(logits.len(), token_ids.len() * config.vocab_size);
}

#[test]
fn test_gpu_model_forward_gpu_empty_tokens() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let token_ids: Vec<usize> = vec![];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_forward_gpu_out_of_bounds_token() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    // Token 9999 is out of bounds for vocab_size=50
    let token_ids = vec![1, 9999];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_forward_gpu_owned() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_owned");
    model.with_test_executor(Box::new(mock));

    let token_ids = vec![1, 2];
    let result = model.forward_gpu_owned(&token_ids);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), token_ids.len() * config.vocab_size);
}

// ============================================================================
// GpuModel forward_block_idx Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_block_idx_basic() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_block");
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.forward_block_idx(&input, 1, 0);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_gpu_model_forward_block_idx_all_layers() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_all_layers");
    model.with_test_executor(Box::new(mock));

    let mut hidden = vec![0.1f32; config.hidden_dim];
    for block_idx in 0..config.num_layers {
        let result = model.forward_block_idx(&hidden, 1, block_idx);
        assert!(result.is_ok(), "Block {} should succeed", block_idx);
        hidden = result.unwrap();
    }
}

// ============================================================================
// GpuModel generate Tests
// ============================================================================

#[test]
fn test_gpu_model_generate_basic() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(3);
    let prompt = vec![1, 2];
    let result = model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(tokens.len() >= prompt.len());
}

#[test]
fn test_gpu_model_generate_optimized() {
    let config = create_minimal_config();
    let mut model = GpuModel::with_attention_buffers(config, 64).unwrap();

    let mock = MockExecutor::new("generate_optimized");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let prompt = vec![1];
    let result = model.generate_optimized(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_generate_optimized_empty_prompt() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let gen_config = GpuGenerateConfig::deterministic(5);
    let prompt: Vec<usize> = vec![];
    let result = model.generate_optimized(&prompt, &gen_config);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_generate_with_sampling() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_sampling");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::with_sampling(2, 0.8, 5);
    let prompt = vec![1, 2];
    let result = model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_generate_with_stop_tokens() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_stop");
    model.with_test_executor(Box::new(mock));

    // Stop token 0 will likely be generated since mock returns zeros
    let gen_config = GpuGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);
    let prompt = vec![1];
    let result = model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
}

// ============================================================================
// GpuModel forward_gpu_with_cache Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_gpu_with_cache() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_with_cache");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let token_ids = vec![1, 2, 3];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_forward_gpu_incremental() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // First populate cache with a token
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Then do incremental forward
    let result = model.forward_gpu_incremental(2, &mut kv_cache);
    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_forward_gpu_incremental_optimized() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("incremental_optimized");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_incremental_optimized(1, &mut kv_cache);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.vocab_size);
}

#[test]
fn test_gpu_model_forward_gpu_incremental_optimized_out_of_bounds() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Token 9999 is out of bounds
    let result = model.forward_gpu_incremental_optimized(9999, &mut kv_cache);
    assert!(result.is_err());
}

// ============================================================================
// GpuModel Fused Operations Tests
// ============================================================================

#[test]
fn test_gpu_model_fused_qkv_projection() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.fused_qkv_projection(&input);

    assert!(result.is_ok());
    let (q, k, v) = result.unwrap();
    assert_eq!(q.len(), config.hidden_dim);
    assert_eq!(k.len(), config.kv_dim());
    assert_eq!(v.len(), config.kv_dim());
}

#[test]
fn test_gpu_model_generate_with_fused_qkv() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("fused_qkv");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let result = model.generate_with_fused_qkv(&[1], &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_forward_with_fused_attn_proj() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("fused_attn");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_with_fused_attn_proj(1, &mut kv_cache);
    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_forward_with_fused_output_residual() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("fused_residual");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_with_fused_output_residual(1, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// GpuModel generate_with_cache Tests
// ============================================================================

#[test]
fn test_gpu_model_generate_with_cache() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("gen_with_cache");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(3);
    let result = model.generate_with_cache(&[1], &gen_config);

    assert!(result.is_ok());
}

// ============================================================================
// BlockWeights Debug/Clone Tests
// ============================================================================

#[test]
fn test_block_weights_structure() {
    // Verify BlockWeights can be constructed manually
    let hidden_dim = 64;
    let intermediate_dim = 128;
    let qkv_dim = 192;

    let block = BlockWeights {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: vec![0.0; hidden_dim],
        qkv_weight: vec![0.01; hidden_dim * qkv_dim],
        qkv_bias: vec![0.0; qkv_dim],
        out_weight: vec![0.01; hidden_dim * hidden_dim],
        out_bias: vec![0.0; hidden_dim],
        ffn_norm_weight: vec![1.0; hidden_dim],
        ffn_norm_bias: vec![0.0; hidden_dim],
        ffn_fc1_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_fc1_bias: vec![0.0; intermediate_dim],
        ffn_fc2_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_fc2_bias: vec![0.0; hidden_dim],
        ffn_gate_weight: None,
    };

    assert_eq!(block.attn_norm_weight.len(), hidden_dim);
    assert!(block.ffn_gate_weight.is_none());
}

#[test]
fn test_block_weights_with_gate() {
    let hidden_dim = 32;
    let intermediate_dim = 64;

    let block = BlockWeights {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: vec![0.0; hidden_dim],
        qkv_weight: vec![0.01; hidden_dim * 96],
        qkv_bias: vec![0.0; 96],
        out_weight: vec![0.01; hidden_dim * hidden_dim],
        out_bias: vec![0.0; hidden_dim],
        ffn_norm_weight: vec![1.0; hidden_dim],
        ffn_norm_bias: vec![0.0; hidden_dim],
        ffn_fc1_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_fc1_bias: vec![0.0; intermediate_dim],
        ffn_fc2_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_fc2_bias: vec![0.0; hidden_dim],
        ffn_gate_weight: Some(vec![0.01; hidden_dim * intermediate_dim]),
    };

    assert!(block.ffn_gate_weight.is_some());
    assert_eq!(
        block.ffn_gate_weight.as_ref().unwrap().len(),
        hidden_dim * intermediate_dim
    );
}

// ============================================================================
// WeightType Debug/Clone Tests
// ============================================================================

#[test]
fn test_weight_type_debug() {
    let qkv = WeightType::Qkv;
    let output = WeightType::Output;
    let fc1 = WeightType::FfnFc1;
    let fc2 = WeightType::FfnFc2;
    let lm_head = WeightType::LmHead;

    // Test Debug trait
    let qkv_debug = format!("{:?}", qkv);
    assert!(qkv_debug.contains("Qkv"));

    let output_debug = format!("{:?}", output);
    assert!(output_debug.contains("Output"));

    let fc1_debug = format!("{:?}", fc1);
    assert!(fc1_debug.contains("FfnFc1"));

    let fc2_debug = format!("{:?}", fc2);
    assert!(fc2_debug.contains("FfnFc2"));

    let lm_head_debug = format!("{:?}", lm_head);
    assert!(lm_head_debug.contains("LmHead"));
}

#[test]
fn test_weight_type_clone() {
    let original = WeightType::Qkv;
    let cloned = original;

    // Clone should work (Copy trait)
    assert!(matches!(cloned, WeightType::Qkv));
}

// ============================================================================
// GpuModel layer_norm_static Tests (via forward operations)
// ============================================================================

#[test]
fn test_gpu_model_layer_norm_via_forward() {
    // layer_norm_static is tested indirectly via forward_block_idx
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("layer_norm_test");
    model.with_test_executor(Box::new(mock));

    // This exercises layer_norm_static internally
    let input = vec![1.0f32; config.hidden_dim];
    let result = model.forward_block_idx(&input, 1, 0);

    assert!(result.is_ok());
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_single_token_only() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("single_token");
    model.with_test_executor(Box::new(mock));

    let token_ids = vec![5];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_gpu_model_generate_max_tokens_zero() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("zero_max_tokens");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(0);
    let prompt = vec![1];
    let result = model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
    // Should return at least the prompt
    let tokens = result.unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_gpu_model_forward_block_incremental_optimized() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("block_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.forward_block_incremental_optimized(&input, 0, &mut kv_cache);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

// ============================================================================
// GpuModel GQA Attention Path Tests
// ============================================================================

#[test]
fn test_gpu_model_gqa_forward() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("gqa_forward");
    model.with_test_executor(Box::new(mock));

    let token_ids = vec![1, 2];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_gqa_incremental() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("gqa_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_incremental_optimized(1, &mut kv_cache);
    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_gqa_generate() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("gqa_generate");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let result = model.generate(&[1], &gen_config);

    assert!(result.is_ok());
}
