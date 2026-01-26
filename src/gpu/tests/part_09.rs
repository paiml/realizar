//! Phase 50 - KV Cache Scheduler Coverage Tests (gpu/scheduler/kv.rs)
//!
//! Comprehensive tests for KV cache operations:
//! - forward_gpu_with_cache: Forward pass with KV cache population
//! - forward_gpu_incremental: Incremental forward using cached KV
//! - generate_with_cache: Full generation loop with KV caching
//! - apply_rope: Rotary position embedding (internal)
//! - gqa_attention_with_kv / gqa_incremental_attention: GQA attention patterns
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
fn create_kv_test_config() -> GpuModelConfig {
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
fn create_kv_gqa_config() -> GpuModelConfig {
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

/// Create a single-layer config for fast iteration tests
fn create_kv_single_layer_config() -> GpuModelConfig {
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

/// Create a deeper model for stress testing
fn create_kv_deep_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}

// ============================================================================
// StreamingKVCache Construction Tests
// ============================================================================

#[test]
fn test_kv_cache_new_basic() {
    let config = create_kv_test_config();
    let cache = StreamingKVCache::new(
        config.num_layers,
        256,
        config.num_kv_heads,
        config.head_dim(),
    );

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 256);
}

#[test]
fn test_kv_cache_new_gqa() {
    let config = create_kv_gqa_config();
    let cache = StreamingKVCache::new(
        config.num_layers,
        512,
        config.num_kv_heads,
        config.head_dim(),
    );

    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 512);
}

#[test]
fn test_kv_cache_memory_calculation() {
    let config = create_kv_test_config();
    let max_positions = 128;
    let cache = StreamingKVCache::new(
        config.num_layers,
        max_positions,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Memory = num_layers * max_positions * num_kv_heads * head_dim * 2 (K+V) * 4 bytes
    let expected = config.num_layers * max_positions * config.num_kv_heads * config.head_dim() * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected);
}

#[test]
fn test_kv_cache_memory_mb_calculation() {
    let config = create_kv_test_config();
    let cache = StreamingKVCache::new(
        config.num_layers,
        1024,
        config.num_kv_heads,
        config.head_dim(),
    );

    let bytes = cache.memory_bytes();
    let expected_mb = bytes as f64 / (1024.0 * 1024.0);
    assert!((cache.memory_mb() - expected_mb).abs() < 0.001);
}

// ============================================================================
// StreamingKVCache Append and Get Tests
// ============================================================================

#[test]
fn test_kv_cache_append_single_position() {
    let config = create_kv_test_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();
    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    // Append to all layers for one position
    for layer in 0..config.num_layers {
        cache.append(layer, &key, &value);
    }

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_append_multiple_positions() {
    let config = create_kv_test_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();

    // Append 5 positions
    for pos in 0..5 {
        let key = vec![pos as f32; kv_dim];
        let value = vec![(pos as f32) * 2.0; kv_dim];

        for layer in 0..config.num_layers {
            cache.append(layer, &key, &value);
        }
    }

    assert_eq!(cache.len(), 5);
}

#[test]
fn test_kv_cache_get_valid_single_layer() {
    let config = create_kv_single_layer_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        32,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();
    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    cache.append(0, &key, &value);

    let (keys, values) = cache.get_valid(0);
    assert_eq!(keys.len(), kv_dim);
    assert_eq!(values.len(), kv_dim);

    // Verify values
    assert!(keys.iter().all(|&k| (k - 1.0).abs() < 1e-6));
    assert!(values.iter().all(|&v| (v - 2.0).abs() < 1e-6));
}

#[test]
fn test_kv_cache_get_range() {
    let config = create_kv_test_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();

    // Append 10 positions
    for pos in 0..10 {
        let key = vec![pos as f32; kv_dim];
        let value = vec![(pos as f32) * 10.0; kv_dim];

        for layer in 0..config.num_layers {
            cache.append(layer, &key, &value);
        }
    }

    // Get range 2..5
    let (keys, values) = cache.get_range(0, 2, 5);
    assert_eq!(keys.len(), 3 * kv_dim);
    assert_eq!(values.len(), 3 * kv_dim);

    // First element should be from position 2
    assert!((keys[0] - 2.0).abs() < 1e-6);
    // Last element should be from position 4
    assert!((keys[(2 * kv_dim)] - 4.0).abs() < 1e-6);
}

#[test]
fn test_kv_cache_clear() {
    let config = create_kv_test_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();
    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    // Add some positions
    for _ in 0..5 {
        for layer in 0..config.num_layers {
            cache.append(layer, &key, &value);
        }
    }
    assert_eq!(cache.len(), 5);

    // Clear
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_kv_cache_circular_buffer_wraparound() {
    let config = create_kv_single_layer_config();
    let max_positions = 4;
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        max_positions,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();

    // Append more positions than max_positions (should wrap around)
    for pos in 0..10 {
        let key = vec![pos as f32; kv_dim];
        let value = vec![(pos as f32) * 10.0; kv_dim];

        for layer in 0..config.num_layers {
            cache.append(layer, &key, &value);
        }
    }

    // Cache should be capped at max_positions
    assert_eq!(cache.len(), max_positions);
}

// ============================================================================
// forward_gpu_with_cache Tests
// ============================================================================

#[test]
fn test_forward_gpu_with_cache_single_token() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_with_cache_single");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let token_ids = vec![1];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);

    // Cache should have 1 position after forward
    assert_eq!(kv_cache.len(), 1);
}

#[test]
fn test_forward_gpu_with_cache_multiple_tokens() {
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_with_cache_multi");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    let token_ids = vec![1, 2, 3, 4, 5];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.unwrap();
    // Output is only for final position
    assert_eq!(logits.len(), config.vocab_size);

    // Cache should have all positions
    assert_eq!(kv_cache.len(), token_ids.len());
}

#[test]
fn test_forward_gpu_with_cache_empty_tokens_error() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let token_ids: Vec<usize> = vec![];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_with_cache_out_of_bounds_token_error() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // vocab_size is 50, so token 999 is out of bounds
    let token_ids = vec![1, 999];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_with_cache_gqa_model() {
    let config = create_kv_gqa_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_gqa_cache");
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
    assert_eq!(kv_cache.len(), 3);
}

// ============================================================================
// forward_gpu_incremental Tests
// ============================================================================

#[test]
fn test_forward_gpu_incremental_basic() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // First populate cache
    let _ = model.forward_gpu_with_cache(&[1, 2], &mut kv_cache);
    assert_eq!(kv_cache.len(), 2);

    // Now do incremental forward
    let result = model.forward_gpu_incremental(3, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);

    // Cache should now have 3 positions
    assert_eq!(kv_cache.len(), 3);
}

#[test]
fn test_forward_gpu_incremental_sequential() {
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("forward_incremental_seq");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initial forward
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Multiple incremental forwards
    for token in [2, 3, 4, 5] {
        let result = model.forward_gpu_incremental(token, &mut kv_cache);
        assert!(result.is_ok());
    }

    assert_eq!(kv_cache.len(), 5);
}

#[test]
fn test_forward_gpu_incremental_out_of_bounds_error() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // vocab_size is 50, so token 9999 is out of bounds
    let result = model.forward_gpu_incremental(9999, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_incremental_gqa() {
    let config = create_kv_gqa_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("incremental_gqa");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initial forward
    let _ = model.forward_gpu_with_cache(&[1, 2], &mut kv_cache);

    // Incremental forward with GQA
    let result = model.forward_gpu_incremental(3, &mut kv_cache);
    assert!(result.is_ok());
    assert_eq!(kv_cache.len(), 3);
}

#[test]
fn test_forward_gpu_incremental_deep_model() {
    let config = create_kv_deep_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("incremental_deep");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initial forward
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Incremental forward through 4 layers
    let result = model.forward_gpu_incremental(2, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// generate_with_cache Tests
// ============================================================================

#[test]
fn test_generate_with_cache_basic() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_cache");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(3);
    let prompt = vec![1, 2];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    // Should have at least prompt tokens
    assert!(tokens.len() >= prompt.len());
}

#[test]
fn test_generate_with_cache_empty_prompt_error() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).unwrap();

    let gen_config = GpuGenerateConfig::deterministic(5);
    let prompt: Vec<usize> = vec![];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_err());
}

#[test]
fn test_generate_with_cache_with_stop_tokens() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_stop");
    model.with_test_executor(Box::new(mock));

    // Mock returns zeros, so token 0 will be generated and should stop
    let gen_config = GpuGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);
    let prompt = vec![1];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_with_sampling() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_sampling");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::with_sampling(3, 0.8, 5);
    let prompt = vec![1, 2];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_gqa() {
    let config = create_kv_gqa_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_gqa");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let prompt = vec![1];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_max_tokens_reached() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_max_tokens");
    model.with_test_executor(Box::new(mock));

    // Generate exactly max_tokens
    let max_tokens = 5;
    let gen_config = GpuGenerateConfig::deterministic(max_tokens);
    let prompt = vec![1];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    // Should have prompt + up to max_tokens (may stop earlier if stop token hit)
    assert!(tokens.len() <= prompt.len() + max_tokens);
}

#[test]
fn test_generate_with_cache_single_token_prompt() {
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_single_prompt");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let prompt = vec![5];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(!tokens.is_empty());
}

// ============================================================================
// KV Cache State Isolation Tests
// ============================================================================

#[test]
fn test_kv_cache_state_isolation_between_calls() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("state_isolation");
    model.with_test_executor(Box::new(mock));

    // First cache
    let mut cache1 = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Second cache
    let mut cache2 = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Populate first cache
    let _ = model.forward_gpu_with_cache(&[1, 2, 3], &mut cache1);
    assert_eq!(cache1.len(), 3);

    // Second cache should be independent
    let _ = model.forward_gpu_with_cache(&[4, 5], &mut cache2);
    assert_eq!(cache2.len(), 2);

    // First cache should still have 3
    assert_eq!(cache1.len(), 3);
}

#[test]
fn test_kv_cache_clear_between_generations() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("clear_between_gen");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // First generation
    let _ = model.forward_gpu_with_cache(&[1, 2, 3], &mut cache);
    assert_eq!(cache.len(), 3);

    // Clear for new generation
    cache.clear();
    assert_eq!(cache.len(), 0);

    // Second generation
    let _ = model.forward_gpu_with_cache(&[4, 5], &mut cache);
    assert_eq!(cache.len(), 2);
}

// ============================================================================
// Memory and Performance Tests
// ============================================================================

#[test]
fn test_kv_cache_large_context() {
    let config = create_kv_test_config();
    let max_positions = 2048;
    let cache = StreamingKVCache::new(
        config.num_layers,
        max_positions,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Verify memory calculation for larger context
    let expected = config.num_layers * max_positions * config.num_kv_heads * config.head_dim() * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected);

    // Should be less than 1MB for this config
    assert!(cache.memory_mb() < 1.0);
}

#[test]
fn test_kv_cache_stress_many_positions() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("stress_positions");
    model.with_test_executor(Box::new(mock));

    let max_positions = 128;
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        max_positions,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Populate with many positions via incremental forward
    let _ = model.forward_gpu_with_cache(&[1], &mut cache);

    for token in 2..100 {
        let result = model.forward_gpu_incremental(token % config.vocab_size, &mut cache);
        assert!(result.is_ok());
    }

    // Should have 99 positions (or wrapped around to max_positions)
    assert!(cache.len() <= max_positions);
}

// ============================================================================
// RoPE Integration Tests (via forward passes)
// ============================================================================

#[test]
fn test_rope_applied_in_forward_with_cache() {
    // RoPE is applied internally during forward_gpu_with_cache
    // We verify it doesn't panic and produces consistent output
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("rope_forward");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[1, 2, 3], &mut cache);
    assert!(result.is_ok());
}

#[test]
fn test_rope_applied_in_incremental_forward() {
    // RoPE uses start_pos in incremental forward
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("rope_incremental");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Populate cache first
    let _ = model.forward_gpu_with_cache(&[1, 2], &mut cache);

    // Incremental forward with correct position
    let result = model.forward_gpu_incremental(3, &mut cache);
    assert!(result.is_ok());
}

#[test]
fn test_rope_theta_variations() {
    // Test with different rope_theta values
    let mut config = create_kv_test_config();

    // Higher rope_theta (e.g., for longer context)
    config.rope_theta = 100000.0;

    let mut model = GpuModel::new(config.clone()).unwrap();
    let mock = MockExecutor::new("rope_theta");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[1, 2], &mut cache);
    assert!(result.is_ok());
}

// ============================================================================
// GQA Attention Pattern Tests
// ============================================================================

#[test]
fn test_gqa_heads_per_kv_ratio() {
    let config = create_kv_gqa_config();

    // num_heads=8, num_kv_heads=2 => 4 Q heads per KV head
    let heads_per_kv = config.num_heads / config.num_kv_heads;
    assert_eq!(heads_per_kv, 4);
}

#[test]
fn test_gqa_kv_dim_calculation() {
    let config = create_kv_gqa_config();

    // head_dim = hidden_dim / num_heads = 64 / 8 = 8
    assert_eq!(config.head_dim(), 8);

    // kv_dim = num_kv_heads * head_dim = 2 * 8 = 16
    assert_eq!(config.kv_dim(), 16);
}

#[test]
fn test_gqa_cache_size_smaller() {
    // GQA should use less cache memory
    let mha_config = create_kv_test_config();
    let gqa_config = create_kv_gqa_config();

    let max_positions = 512;

    let mha_cache = StreamingKVCache::new(
        mha_config.num_layers,
        max_positions,
        mha_config.num_kv_heads,
        mha_config.head_dim(),
    );

    let gqa_cache = StreamingKVCache::new(
        gqa_config.num_layers,
        max_positions,
        gqa_config.num_kv_heads,
        gqa_config.head_dim(),
    );

    // GQA cache should be smaller (fewer kv_heads)
    assert!(gqa_cache.memory_bytes() < mha_cache.memory_bytes());
}

#[test]
fn test_gqa_forward_with_cache_full_sequence() {
    let config = create_kv_gqa_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("gqa_full_seq");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Longer sequence to exercise GQA
    let token_ids: Vec<usize> = (0..20).map(|i| i % config.vocab_size).collect();
    let result = model.forward_gpu_with_cache(&token_ids, &mut cache);

    assert!(result.is_ok());
    assert_eq!(cache.len(), 20);
}

// ============================================================================
// Error Path Tests
// ============================================================================

#[test]
fn test_forward_with_cache_boundary_token() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("boundary_token");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Use maximum valid token (vocab_size - 1)
    let max_valid_token = config.vocab_size - 1;
    let result = model.forward_gpu_with_cache(&[max_valid_token], &mut cache);

    assert!(result.is_ok());
}

#[test]
fn test_incremental_with_empty_cache() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("empty_cache");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Incremental forward with empty cache (first token)
    let result = model.forward_gpu_incremental(1, &mut cache);

    // Should work - effectively same as first token
    assert!(result.is_ok());
    assert_eq!(cache.len(), 1);
}

// ============================================================================
// Integration Tests (combining multiple operations)
// ============================================================================

#[test]
fn test_full_inference_workflow() {
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("full_workflow");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    // 1. Prefill with prompt
    let prompt = vec![1, 2, 3, 4, 5];
    let prefill_result = model.forward_gpu_with_cache(&prompt, &mut cache);
    assert!(prefill_result.is_ok());
    assert_eq!(cache.len(), prompt.len());

    // 2. Decode multiple tokens
    for i in 0..10 {
        let token = (i + 6) % config.vocab_size;
        let decode_result = model.forward_gpu_incremental(token, &mut cache);
        assert!(decode_result.is_ok());
    }

    // Cache should have prompt + decoded tokens
    assert_eq!(cache.len(), prompt.len() + 10);
}

#[test]
fn test_repeated_generations_with_cache_clear() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("repeated_gen");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Multiple generation runs
    for run in 0..3 {
        cache.clear();
        assert!(cache.is_empty());

        let prompt = vec![run % config.vocab_size];
        let result = model.forward_gpu_with_cache(&prompt, &mut cache);
        assert!(result.is_ok(), "Run {} failed", run);

        for i in 0..5 {
            let token = (run + i + 1) % config.vocab_size;
            let result = model.forward_gpu_incremental(token, &mut cache);
            assert!(result.is_ok(), "Incremental {} in run {} failed", i, run);
        }
    }
}

#[test]
fn test_kv_cache_with_different_configs() {
    // Test that KV cache works correctly with various model configs
    let configs = vec![
        create_kv_test_config(),
        create_kv_gqa_config(),
        create_kv_single_layer_config(),
        create_kv_deep_config(),
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let mut model = GpuModel::new(config.clone()).unwrap();

        let mock = MockExecutor::new(&format!("config_{}", i));
        model.with_test_executor(Box::new(mock));

        let mut cache = StreamingKVCache::new(
            config.num_layers,
            64,
            config.num_kv_heads,
            config.head_dim(),
        );

        let result = model.forward_gpu_with_cache(&[1, 2], &mut cache);
        assert!(result.is_ok(), "Config {} failed", i);
        assert_eq!(cache.len(), 2, "Config {} cache length mismatch", i);
    }
}
