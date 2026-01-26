//! Comprehensive tests for GPU Batch Planner (Phase 47)
//!
//! These tests target the ~7% uncovered code paths in planner.rs:
//! - `GenerationConfig::from_model`
//! - `BatchPlanner::plan_next` edge cases (Initial state, None token)
//! - `BatchPlanner::config()` accessor
//! - `BlockForwardPlan::attention_output_size()`
//! - `SamplingStrategy::Default`
//! - Edge cases in `plan_sampling` (boundary values)

use super::planner::{
    plan_lm_head_path, plan_sampling, BatchPlanner, BlockForwardPlan, GenerationConfig,
    GenerationStep, LmHeadPath, SamplingStrategy,
};
use super::GpuModelConfig;

// ============================================================================
// GenerationConfig Tests
// ============================================================================

#[test]
fn test_generation_config_from_model() {
    let model_config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 2048,
        num_heads: 32,
        num_kv_heads: 4,
        num_layers: 22,
        intermediate_dim: 5632,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let gen_config = GenerationConfig::from_model(&model_config, 50);

    assert_eq!(gen_config.max_tokens, 50);
    assert_eq!(gen_config.vocab_size, 32000);
    assert_eq!(gen_config.greedy_vocab_threshold, 8192);
    assert!(gen_config.stop_token.is_none());
    assert!(gen_config.use_greedy_path()); // 32000 > 8192
}

#[test]
fn test_generation_config_from_model_small_vocab() {
    let model_config = GpuModelConfig {
        vocab_size: 1000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 6,
        intermediate_dim: 2048,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let gen_config = GenerationConfig::from_model(&model_config, 100);

    assert_eq!(gen_config.max_tokens, 100);
    assert_eq!(gen_config.vocab_size, 1000);
    assert!(!gen_config.use_greedy_path()); // 1000 < 8192
}

#[test]
fn test_generation_config_boundary_vocab_size() {
    // Exactly at threshold
    let config = GenerationConfig {
        vocab_size: 8192,
        ..Default::default()
    };
    assert!(!config.use_greedy_path()); // 8192 is NOT > 8192

    // Just above threshold
    let config = GenerationConfig {
        vocab_size: 8193,
        ..Default::default()
    };
    assert!(config.use_greedy_path()); // 8193 > 8192
}

// ============================================================================
// BatchPlanner Edge Case Tests
// ============================================================================

#[test]
fn test_planner_initial_state_without_prompt() {
    let config = GenerationConfig::default();
    let mut planner = BatchPlanner::new(config);

    // Calling plan_next without setting prompt should return Done
    let step = planner.plan_next(None);
    assert!(
        matches!(step, GenerationStep::Done { tokens } if tokens.is_empty()),
        "Expected Done with empty tokens when no prompt set"
    );
}

#[test]
fn test_planner_plan_next_with_none_token() {
    let config = GenerationConfig {
        max_tokens: 5,
        ..Default::default()
    };
    let mut planner = BatchPlanner::new(config);

    // Start with prompt
    let _ = planner.start_with_prompt(&[1, 2, 3]);

    // Call plan_next with None (simulates first call after prompt)
    let step = planner.plan_next(None);
    assert!(matches!(step, GenerationStep::GenerateToken { .. }));

    // Tokens should still be just the prompt
    assert_eq!(planner.tokens(), &[1, 2, 3]);
    assert_eq!(planner.generated_count(), 0);
}

#[test]
fn test_planner_config_accessor() {
    let config = GenerationConfig {
        max_tokens: 42,
        vocab_size: 5000,
        greedy_vocab_threshold: 4000,
        stop_token: Some(123),
    };
    let planner = BatchPlanner::new(config);

    let retrieved_config = planner.config();
    assert_eq!(retrieved_config.max_tokens, 42);
    assert_eq!(retrieved_config.vocab_size, 5000);
    assert_eq!(retrieved_config.greedy_vocab_threshold, 4000);
    assert_eq!(retrieved_config.stop_token, Some(123));
}

#[test]
fn test_planner_done_state_idempotent() {
    let config = GenerationConfig {
        max_tokens: 1,
        ..Default::default()
    };
    let mut planner = BatchPlanner::new(config);

    let _ = planner.start_with_prompt(&[1]);
    let _ = planner.plan_next(Some(100)); // This should trigger Done (max_tokens = 1)

    assert!(planner.is_done());

    // Calling plan_next again should still return Done
    let step = planner.plan_next(Some(200));
    assert!(matches!(step, GenerationStep::Done { .. }));

    // Token 200 should not have been added since we were already done
    assert_eq!(planner.tokens(), &[1, 100]);
}

#[test]
fn test_planner_empty_prompt() {
    let config = GenerationConfig {
        max_tokens: 3,
        ..Default::default()
    };
    let mut planner = BatchPlanner::new(config);

    // Start with empty prompt
    let step = planner.start_with_prompt(&[]);
    assert!(matches!(step, GenerationStep::ProcessPrompt { tokens } if tokens.is_empty()));

    // Should still be able to generate tokens
    let step = planner.plan_next(Some(1));
    assert!(matches!(step, GenerationStep::GenerateToken { .. }));
    assert_eq!(planner.tokens(), &[1]);
}

// ============================================================================
// BlockForwardPlan Tests
// ============================================================================

#[test]
fn test_block_forward_plan_attention_output_size() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 2048,
        num_heads: 32,
        num_kv_heads: 4,
        num_layers: 22,
        intermediate_dim: 5632,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let plan = BlockForwardPlan::from_config(&config, 0, true);
    assert_eq!(plan.attention_output_size(), 2048);
}

#[test]
fn test_block_forward_plan_all_fields() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 4096,
        num_heads: 64,
        num_kv_heads: 8,
        num_layers: 40,
        intermediate_dim: 14336,
        eps: 1e-6,
        rope_theta: 500000.0,
    };

    let plan = BlockForwardPlan::from_config(&config, 10, true);

    assert_eq!(plan.block_idx, 10);
    assert_eq!(plan.hidden_dim, 4096);
    assert_eq!(plan.num_heads, 64);
    assert_eq!(plan.num_kv_heads, 8);
    assert_eq!(plan.head_dim, 64); // 4096 / 64
    assert_eq!(plan.intermediate_dim, 14336);
    assert!(plan.use_swiglu);
    assert_eq!(plan.heads_per_kv, 8); // 64 / 8
    assert!(plan.is_gqa());
    assert_eq!(plan.attention_output_size(), 4096);

    // Check kv_dim and qkv_dim are set correctly
    // kv_dim = num_kv_heads * head_dim = 8 * 64 = 512
    // qkv_dim = hidden_dim + 2 * kv_dim = 4096 + 2 * 512 = 5120
    assert_eq!(plan.kv_dim, 512);
    assert_eq!(plan.qkv_dim, 5120);
}

#[test]
fn test_block_forward_plan_mqa() {
    // Multi-Query Attention: single KV head
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 2048,
        num_heads: 32,
        num_kv_heads: 1, // MQA: single KV head
        num_layers: 24,
        intermediate_dim: 5632,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let plan = BlockForwardPlan::from_config(&config, 0, false);
    assert!(plan.is_gqa()); // heads_per_kv = 32 > 1
    assert_eq!(plan.heads_per_kv, 32);
    assert!(!plan.use_swiglu);
}

// ============================================================================
// SamplingStrategy Tests
// ============================================================================

#[test]
fn test_sampling_strategy_default() {
    let strategy = SamplingStrategy::default();
    assert_eq!(strategy, SamplingStrategy::Greedy);
}

#[test]
fn test_sampling_strategy_clone_and_debug() {
    let strategies = [
        SamplingStrategy::Greedy,
        SamplingStrategy::TopK { k: 50 },
        SamplingStrategy::TopP { p: 0.9 },
        SamplingStrategy::Temperature { temp: 0.7 },
    ];

    for strategy in &strategies {
        let cloned = strategy.clone();
        assert_eq!(&cloned, strategy);

        // Test Debug implementation
        let debug_str = format!("{:?}", strategy);
        assert!(!debug_str.is_empty());
    }
}

// ============================================================================
// plan_sampling Edge Case Tests
// ============================================================================

#[test]
fn test_plan_sampling_top_k_zero() {
    // top_k = 0 should fall through to Greedy
    let strategy = plan_sampling(None, Some(0), None);
    assert_eq!(strategy, SamplingStrategy::Greedy);
}

#[test]
fn test_plan_sampling_top_k_max() {
    // top_k = usize::MAX should fall through to Greedy
    let strategy = plan_sampling(None, Some(usize::MAX), None);
    assert_eq!(strategy, SamplingStrategy::Greedy);
}

#[test]
fn test_plan_sampling_top_p_one() {
    // top_p = 1.0 should fall through to Greedy
    let strategy = plan_sampling(None, None, Some(1.0));
    assert_eq!(strategy, SamplingStrategy::Greedy);
}

#[test]
fn test_plan_sampling_top_p_zero() {
    // top_p = 0.0 should fall through to Greedy
    let strategy = plan_sampling(None, None, Some(0.0));
    assert_eq!(strategy, SamplingStrategy::Greedy);
}

#[test]
fn test_plan_sampling_temperature_negative() {
    // Negative temperature should fall through to Greedy
    let strategy = plan_sampling(Some(-0.5), None, None);
    assert_eq!(strategy, SamplingStrategy::Greedy);
}

#[test]
fn test_plan_sampling_all_invalid() {
    // All parameters invalid should return Greedy
    let strategy = plan_sampling(Some(0.0), Some(0), Some(1.0));
    assert_eq!(strategy, SamplingStrategy::Greedy);
}

#[test]
fn test_plan_sampling_boundary_values() {
    // top_p just below 1.0
    let strategy = plan_sampling(None, None, Some(0.999));
    assert_eq!(strategy, SamplingStrategy::TopP { p: 0.999 });

    // top_p just above 0.0
    let strategy = plan_sampling(None, None, Some(0.001));
    assert_eq!(strategy, SamplingStrategy::TopP { p: 0.001 });

    // top_k = 1 is valid
    let strategy = plan_sampling(None, Some(1), None);
    assert_eq!(strategy, SamplingStrategy::TopK { k: 1 });

    // top_k = usize::MAX - 1 is valid
    let strategy = plan_sampling(None, Some(usize::MAX - 1), None);
    assert_eq!(
        strategy,
        SamplingStrategy::TopK {
            k: usize::MAX - 1
        }
    );
}

// ============================================================================
// plan_lm_head_path Tests
// ============================================================================

#[test]
fn test_plan_lm_head_path_boundary_vocab() {
    // Exactly at 8192 - should use GPU
    let path = plan_lm_head_path(8192, 768, 100_000_000);
    assert_eq!(path, LmHeadPath::Gpu);

    // Just above 8192 - should use CPU
    let path = plan_lm_head_path(8193, 768, 100_000_000);
    assert_eq!(path, LmHeadPath::CpuTransposed);
}

#[test]
fn test_plan_lm_head_path_exact_buffer_limit() {
    // Elements exactly at buffer limit - should use GPU
    let buffer_limit = 5000 * 768; // 3,840,000
    let path = plan_lm_head_path(5000, 768, buffer_limit);
    assert_eq!(path, LmHeadPath::Gpu);

    // Elements just over buffer limit - should use CPU
    let path = plan_lm_head_path(5000, 768, buffer_limit - 1);
    assert_eq!(path, LmHeadPath::CpuTransposed);
}

#[test]
fn test_lm_head_path_clone_and_debug() {
    let paths = [LmHeadPath::CpuTransposed, LmHeadPath::Gpu];

    for path in &paths {
        let cloned = path.clone();
        assert_eq!(&cloned, path);

        let debug_str = format!("{:?}", path);
        assert!(!debug_str.is_empty());
    }
}

// ============================================================================
// GenerationStep Tests
// ============================================================================

#[test]
fn test_generation_step_clone_and_debug() {
    let steps = [
        GenerationStep::ProcessPrompt {
            tokens: vec![1, 2, 3],
        },
        GenerationStep::GenerateToken {
            tokens: vec![1, 2, 3, 4],
            use_greedy_optimization: true,
        },
        GenerationStep::Done {
            tokens: vec![1, 2, 3, 4, 5],
        },
    ];

    for step in &steps {
        let cloned = step.clone();
        assert_eq!(&cloned, step);

        let debug_str = format!("{:?}", step);
        assert!(!debug_str.is_empty());
    }
}

#[test]
fn test_generation_step_equality() {
    let step1 = GenerationStep::ProcessPrompt {
        tokens: vec![1, 2, 3],
    };
    let step2 = GenerationStep::ProcessPrompt {
        tokens: vec![1, 2, 3],
    };
    let step3 = GenerationStep::ProcessPrompt {
        tokens: vec![1, 2, 4],
    };

    assert_eq!(step1, step2);
    assert_ne!(step1, step3);
}

// ============================================================================
// GenerationConfig Tests
// ============================================================================

#[test]
fn test_generation_config_clone_and_debug() {
    let config = GenerationConfig {
        max_tokens: 100,
        vocab_size: 32000,
        greedy_vocab_threshold: 8192,
        stop_token: Some(2),
    };

    let cloned = config.clone();
    assert_eq!(cloned.max_tokens, config.max_tokens);
    assert_eq!(cloned.vocab_size, config.vocab_size);
    assert_eq!(cloned.greedy_vocab_threshold, config.greedy_vocab_threshold);
    assert_eq!(cloned.stop_token, config.stop_token);

    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("100"));
    assert!(debug_str.contains("32000"));
}

// ============================================================================
// BatchPlanner Tests
// ============================================================================

#[test]
fn test_batch_planner_clone_and_debug() {
    let config = GenerationConfig::default();
    let planner = BatchPlanner::new(config);

    let cloned = planner.clone();
    assert_eq!(cloned.tokens(), planner.tokens());
    assert_eq!(cloned.generated_count(), planner.generated_count());
    assert_eq!(cloned.is_done(), planner.is_done());

    let debug_str = format!("{:?}", planner);
    assert!(!debug_str.is_empty());
}

#[test]
fn test_batch_planner_multiple_generation_cycles() {
    let config = GenerationConfig {
        max_tokens: 2,
        stop_token: None,
        ..Default::default()
    };
    let mut planner = BatchPlanner::new(config);

    // Start with prompt
    let step = planner.start_with_prompt(&[100]);
    assert!(matches!(step, GenerationStep::ProcessPrompt { .. }));

    // Generate token 1
    let step = planner.plan_next(Some(200));
    assert!(matches!(step, GenerationStep::GenerateToken { .. }));
    assert_eq!(planner.generated_count(), 1);

    // Generate token 2 - should complete
    let step = planner.plan_next(Some(300));
    assert!(matches!(step, GenerationStep::Done { .. }));
    assert_eq!(planner.generated_count(), 2);

    // Verify final tokens
    assert_eq!(planner.tokens(), &[100, 200, 300]);
}

// ============================================================================
// BlockForwardPlan Tests
// ============================================================================

#[test]
fn test_block_forward_plan_clone_and_debug() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 768,
        num_heads: 12,
        num_kv_heads: 12,
        num_layers: 12,
        intermediate_dim: 3072,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let plan = BlockForwardPlan::from_config(&config, 5, true);
    let cloned = plan.clone();

    assert_eq!(cloned.block_idx, plan.block_idx);
    assert_eq!(cloned.hidden_dim, plan.hidden_dim);
    assert_eq!(cloned.num_heads, plan.num_heads);
    assert_eq!(cloned.use_swiglu, plan.use_swiglu);

    let debug_str = format!("{:?}", plan);
    assert!(debug_str.contains("BlockForwardPlan"));
}

#[test]
fn test_block_forward_plan_equality() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 768,
        num_heads: 12,
        num_kv_heads: 12,
        num_layers: 12,
        intermediate_dim: 3072,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let plan1 = BlockForwardPlan::from_config(&config, 0, true);
    let plan2 = BlockForwardPlan::from_config(&config, 0, true);
    let plan3 = BlockForwardPlan::from_config(&config, 1, true);

    assert_eq!(plan1, plan2);
    assert_ne!(plan1, plan3);
}
