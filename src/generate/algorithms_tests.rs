//! Comprehensive tests for advanced sampling algorithms
//!
//! This module tests edge cases and code paths in algorithms.rs
//! that may not be covered by the main test suite.

use crate::generate::algorithms::*;
use crate::tensor::Tensor;

// =============================================================================
// Min-P Sampling Tests
// =============================================================================

#[test]
fn test_sample_min_p_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_min_p_invalid_min_p_negative() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let result = sample_min_p(&logits, -0.01, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_sample_min_p_invalid_min_p_greater_than_one() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let result = sample_min_p(&logits, 1.01, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_sample_min_p_boundary_zero() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    // min_p = 0.0 should include all tokens
    let result = sample_min_p(&logits, 0.0, 0.5).expect("test");
    assert!(result < 4);
}

#[test]
fn test_sample_min_p_boundary_one() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 100.0]).expect("test");
    // min_p = 1.0 should only include max prob token
    let result = sample_min_p(&logits, 1.0, 0.5).expect("test");
    assert_eq!(result, 3);
}

#[test]
fn test_sample_min_p_all_equal_probs() {
    let logits = Tensor::from_vec(vec![5], vec![0.0; 5]).expect("test");
    // All equal, all should pass threshold
    let result = sample_min_p(&logits, 0.5, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_min_p_rng_selection() {
    let logits = Tensor::from_vec(vec![3], vec![10.0, 10.0, 0.0]).expect("test");
    // Two equal high tokens, rng=0.0 should pick first
    let result = sample_min_p(&logits, 0.5, 0.0).expect("test");
    assert!(result == 0 || result == 1);
}

// =============================================================================
// MirostatState Tests
// =============================================================================

#[test]
fn test_mirostat_state_default_values() {
    let state = MirostatState::default();
    assert!((state.tau - 5.0).abs() < 1e-6);
    assert!((state.eta - 0.1).abs() < 1e-6);
    assert!((state.mu - 10.0).abs() < 1e-6);
}

#[test]
fn test_mirostat_state_new_tau_sets_mu() {
    let state = MirostatState::new(3.0);
    assert!((state.tau - 3.0).abs() < 1e-6);
    assert!((state.mu - 6.0).abs() < 1e-6); // mu = 2 * tau
}

#[test]
fn test_mirostat_state_with_eta_builder() {
    let state = MirostatState::new(5.0).with_eta(0.5);
    assert!((state.eta - 0.5).abs() < 1e-6);
}

#[test]
fn test_mirostat_state_update_increases_mu() {
    let mut state = MirostatState::new(5.0).with_eta(0.1);
    let initial_mu = state.mu;
    // Observed surprise < tau, so mu should increase
    state.update(2.0);
    assert!(state.mu > initial_mu);
}

#[test]
fn test_mirostat_state_update_decreases_mu() {
    let mut state = MirostatState::new(5.0).with_eta(0.1);
    let initial_mu = state.mu;
    // Observed surprise > tau, so mu should decrease
    state.update(10.0);
    assert!(state.mu < initial_mu);
}

#[test]
fn test_mirostat_state_clone() {
    let state = MirostatState::new(3.0).with_eta(0.2);
    let cloned = state.clone();
    assert!((cloned.tau - state.tau).abs() < 1e-6);
    assert!((cloned.eta - state.eta).abs() < 1e-6);
    assert!((cloned.mu - state.mu).abs() < 1e-6);
}

// =============================================================================
// Mirostat Sampling Tests
// =============================================================================

#[test]
fn test_sample_mirostat_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_mirostat_single_token() {
    let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let mut state = MirostatState::default();
    let result = sample_mirostat(&logits, &mut state, 0.5).expect("test");
    assert_eq!(result, 0);
}

#[test]
fn test_sample_mirostat_low_mu_fallback() {
    // Very low mu should still return at least top candidate
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let mut state = MirostatState::new(0.01); // Very low tau
    state.mu = 0.001; // Extremely low mu
    let result = sample_mirostat(&logits, &mut state, 0.5).expect("test");
    assert!(result < 3);
}

#[test]
fn test_sample_mirostat_updates_state() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 10.0]).expect("test");
    let mut state = MirostatState::default();
    let initial_mu = state.mu;
    let _ = sample_mirostat(&logits, &mut state, 0.5).expect("test");
    assert!((state.mu - initial_mu).abs() > 1e-6);
}

// =============================================================================
// TFS (Tail-Free Sampling) Tests
// =============================================================================

#[test]
fn test_sample_tfs_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_tfs_two_tokens_greedy() {
    // Less than 3 tokens, uses greedy
    let logits = Tensor::from_vec(vec![2], vec![1.0, 5.0]).expect("test");
    let result = sample_tfs(&logits, 0.95, 0.5).expect("test");
    assert_eq!(result, 1);
}

#[test]
fn test_sample_tfs_z_zero_strict() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    // z=0 should be very restrictive
    let result = sample_tfs(&logits, 0.0, 0.0).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_tfs_z_one_permissive() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    // z=1 should include many tokens
    let result = sample_tfs(&logits, 1.0, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_tfs_uniform_distribution() {
    let logits = Tensor::from_vec(vec![5], vec![0.0; 5]).expect("test");
    // Uniform distribution - second derivatives all zero
    let result = sample_tfs(&logits, 0.5, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_tfs_single_dominant() {
    let logits = Tensor::from_vec(vec![5], vec![100.0, 0.0, 0.0, 0.0, 0.0]).expect("test");
    let result = sample_tfs(&logits, 0.95, 0.0).expect("test");
    assert_eq!(result, 0);
}

// =============================================================================
// Typical Sampling Tests
// =============================================================================

#[test]
fn test_sample_typical_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_typical_single_token() {
    let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let result = sample_typical(&logits, 0.95, 0.5).expect("test");
    assert_eq!(result, 0);
}

#[test]
fn test_sample_typical_p_very_small() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    // Very small p should select most typical token(s)
    let result = sample_typical(&logits, 0.01, 0.0).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_typical_p_one() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    let result = sample_typical(&logits, 1.0, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_typical_all_zero_entropy() {
    // One token has all the probability
    let logits = Tensor::from_vec(vec![3], vec![100.0, -100.0, -100.0]).expect("test");
    let result = sample_typical(&logits, 0.95, 0.5).expect("test");
    assert_eq!(result, 0);
}

// =============================================================================
// DryConfig Tests
// =============================================================================

#[test]
fn test_dry_config_default() {
    let config = DryConfig::default();
    assert!((config.multiplier - 0.8).abs() < 1e-6);
    assert!((config.base - 1.75).abs() < 1e-6);
    assert_eq!(config.allowed_length, 2);
    assert_eq!(config.penalty_last_n, 256);
    assert!(config.is_enabled());
}

#[test]
fn test_dry_config_new() {
    let config = DryConfig::new(0.5);
    assert!((config.multiplier - 0.5).abs() < 1e-6);
}

#[test]
fn test_dry_config_disabled() {
    let config = DryConfig::new(0.0);
    assert!(!config.is_enabled());
}

#[test]
fn test_dry_config_builders() {
    let config = DryConfig::new(1.0)
        .with_base(2.0)
        .with_allowed_length(3)
        .with_penalty_last_n(128);
    assert!((config.base - 2.0).abs() < 1e-6);
    assert_eq!(config.allowed_length, 3);
    assert_eq!(config.penalty_last_n, 128);
}

// =============================================================================
// DRY Penalty Tests
// =============================================================================

#[test]
fn test_apply_dry_penalty_disabled() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = DryConfig::new(0.0);
    let result = apply_dry_penalty(&logits, &[0, 1, 0, 1], &config);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_dry_penalty_short_context() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = DryConfig::new(1.0).with_allowed_length(5);
    // Context shorter than allowed_length
    let result = apply_dry_penalty(&logits, &[0, 1, 2], &config);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_dry_penalty_window_truncation() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = DryConfig::new(1.0).with_penalty_last_n(3);
    // Long context, but only last 3 tokens used
    let long_context: Vec<usize> = (0..100).collect();
    let result = apply_dry_penalty(&logits, &long_context, &config);
    // Should still work
    assert_eq!(result.data().len(), 5);
}

#[test]
fn test_apply_dry_penalty_repetition_detected() {
    let logits = Tensor::from_vec(vec![5], vec![10.0; 5]).expect("test");
    let config = DryConfig::new(1.0).with_allowed_length(2);
    // Pattern: [0,1] repeats, next token 0 would extend
    let context = vec![0, 1, 0, 1];
    let result = apply_dry_penalty(&logits, &context, &config);
    // Token 0 should be penalized
    assert!(result.data()[0] < 10.0);
}

#[test]
fn test_apply_dry_penalty_no_repetition() {
    let logits = Tensor::from_vec(vec![5], vec![10.0; 5]).expect("test");
    let config = DryConfig::new(1.0).with_allowed_length(2);
    // No repetition pattern
    let context = vec![0, 1, 2, 3];
    let result = apply_dry_penalty(&logits, &context, &config);
    // No penalty should be applied
    for val in result.data() {
        assert!((*val - 10.0).abs() < 1e-6);
    }
}

// =============================================================================
// XtcConfig Tests
// =============================================================================

#[test]
fn test_xtc_config_default() {
    let config = XtcConfig::default();
    assert!((config.probability - 0.0).abs() < 1e-6);
    assert!((config.threshold - 0.5).abs() < 1e-6);
    assert_eq!(config.min_keep, 1);
    assert!(!config.is_enabled());
}

#[test]
fn test_xtc_config_new() {
    let config = XtcConfig::new(0.5);
    assert!((config.probability - 0.5).abs() < 1e-6);
    assert!(config.is_enabled());
}

#[test]
fn test_xtc_config_builders() {
    let config = XtcConfig::new(0.8).with_threshold(0.3).with_min_keep(2);
    assert!((config.threshold - 0.3).abs() < 1e-6);
    assert_eq!(config.min_keep, 2);
}

// =============================================================================
// XTC (Exclude Top Choices) Tests
// =============================================================================

#[test]
fn test_apply_xtc_disabled() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = XtcConfig::default(); // probability = 0
    let result = apply_xtc(&logits, &config, 0.5);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_xtc_rng_above_probability() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = XtcConfig::new(0.5); // 50% chance
                                      // rng = 0.6 > 0.5, so no exclusion
    let result = apply_xtc(&logits, &config, 0.6);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_xtc_too_few_tokens() {
    let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let config = XtcConfig::new(1.0).with_min_keep(2);
    // Only 1 token, can't exclude
    let result = apply_xtc(&logits, &config, 0.0);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_xtc_excludes_top_token() {
    let logits = Tensor::from_vec(vec![3], vec![0.0, 0.0, 100.0]).expect("test");
    let config = XtcConfig::new(1.0).with_threshold(0.5).with_min_keep(1);
    // Token 2 has high probability, should be excluded
    let result = apply_xtc(&logits, &config, 0.0);
    assert_eq!(result.data()[2], f32::NEG_INFINITY);
}

#[test]
fn test_apply_xtc_respects_min_keep() {
    let logits = Tensor::from_vec(vec![3], vec![100.0, 100.0, 100.0]).expect("test");
    let config = XtcConfig::new(1.0).with_threshold(0.1).with_min_keep(2);
    let result = apply_xtc(&logits, &config, 0.0);
    // Should keep at least 2 tokens (not NEG_INFINITY)
    let finite_count = result.data().iter().filter(|&&x| x.is_finite()).count();
    assert!(finite_count >= 2);
}

// =============================================================================
// EtaConfig Tests
// =============================================================================

#[test]
fn test_eta_config_default() {
    let config = EtaConfig::default();
    assert!((config.eta - 0.3).abs() < 1e-6);
    assert!((config.min_p - 0.0001).abs() < 1e-6);
    assert!(config.is_enabled());
}

#[test]
fn test_eta_config_new() {
    let config = EtaConfig::new(0.5);
    assert!((config.eta - 0.5).abs() < 1e-6);
}

#[test]
fn test_eta_config_disabled() {
    let config = EtaConfig::new(0.0);
    assert!(!config.is_enabled());
}

#[test]
fn test_eta_config_with_min_p() {
    let config = EtaConfig::new(0.5).with_min_p(0.01);
    assert!((config.min_p - 0.01).abs() < 1e-6);
}

// =============================================================================
// Eta Sampling Tests
// =============================================================================

#[test]
fn test_sample_eta_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_eta_single_token() {
    let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let config = EtaConfig::default();
    let result = sample_eta(&logits, &config, 0.5).expect("test");
    assert_eq!(result, 0);
}

#[test]
fn test_sample_eta_high_entropy() {
    // Uniform distribution has high entropy
    let logits = Tensor::from_vec(vec![5], vec![0.0; 5]).expect("test");
    let config = EtaConfig::new(0.5);
    let result = sample_eta(&logits, &config, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_eta_low_entropy_fallback() {
    // Single dominant token
    let logits = Tensor::from_vec(vec![3], vec![100.0, -100.0, -100.0]).expect("test");
    let config = EtaConfig::new(0.5);
    let result = sample_eta(&logits, &config, 0.5).expect("test");
    assert_eq!(result, 0);
}

// =============================================================================
// TokenHealingConfig Tests
// =============================================================================

#[test]
fn test_token_healing_config_default() {
    let config = TokenHealingConfig::default();
    assert!(!config.enabled);
    assert_eq!(config.max_backup_chars, 0);
}

#[test]
fn test_token_healing_config_new() {
    let config = TokenHealingConfig::new(true);
    assert!(config.enabled);
    assert_eq!(config.max_backup_chars, 10);
}

#[test]
fn test_token_healing_config_with_max_backup() {
    let config = TokenHealingConfig::new(true).with_max_backup(20);
    assert_eq!(config.max_backup_chars, 20);
}

// =============================================================================
// Token Healing Analysis Tests
// =============================================================================

#[test]
fn test_analyze_token_healing_no_healing_needed() {
    let tokens = vec![1, 2, 3];
    let result = analyze_token_healing(&tokens, Some(" complete"));
    assert_eq!(result.tokens_removed, 0);
    assert!(result.prefix_constraint.is_none());
    assert_eq!(result.adjusted_tokens, tokens);
}

#[test]
fn test_analyze_token_healing_partial_token() {
    let tokens = vec![1, 2, 3];
    // Short alphanumeric without space = partial
    let result = analyze_token_healing(&tokens, Some("wo"));
    assert_eq!(result.tokens_removed, 1);
    assert_eq!(result.prefix_constraint, Some("wo".to_string()));
    assert_eq!(result.adjusted_tokens, vec![1, 2]);
}

#[test]
fn test_analyze_token_healing_empty_tokens() {
    let tokens: Vec<usize> = vec![];
    let result = analyze_token_healing(&tokens, Some("a"));
    assert_eq!(result.tokens_removed, 0);
    assert_eq!(result.adjusted_tokens.len(), 0);
}

#[test]
fn test_analyze_token_healing_none_text() {
    let tokens = vec![1, 2, 3];
    let result = analyze_token_healing(&tokens, None);
    assert_eq!(result.tokens_removed, 0);
    assert!(result.prefix_constraint.is_none());
}

#[test]
fn test_analyze_token_healing_long_text_no_heal() {
    let tokens = vec![1, 2, 3];
    // More than 3 chars = not partial
    let result = analyze_token_healing(&tokens, Some("word"));
    assert_eq!(result.tokens_removed, 0);
}

#[test]
fn test_analyze_token_healing_non_alphanumeric() {
    let tokens = vec![1, 2, 3];
    // Contains non-alphanumeric
    let result = analyze_token_healing(&tokens, Some("a!"));
    assert_eq!(result.tokens_removed, 0);
}

// =============================================================================
// CfgConfig Tests
// =============================================================================

#[test]
fn test_cfg_config_default() {
    let config = CfgConfig::default();
    assert!((config.scale - 1.0).abs() < 1e-6);
    assert!(config.negative_prompt_tokens.is_empty());
    assert!(!config.is_enabled());
}

#[test]
fn test_cfg_config_new() {
    let config = CfgConfig::new(2.0);
    assert!((config.scale - 2.0).abs() < 1e-6);
    assert!(config.is_enabled());
}

#[test]
fn test_cfg_config_boundary_not_enabled() {
    let config = CfgConfig::new(1.0);
    assert!(!config.is_enabled());
}

#[test]
fn test_cfg_config_with_negative_prompt() {
    let config = CfgConfig::new(1.5).with_negative_prompt(vec![1, 2, 3]);
    assert_eq!(config.negative_prompt_tokens, vec![1, 2, 3]);
}

// =============================================================================
// CFG (Classifier-Free Guidance) Tests
// =============================================================================

#[test]
fn test_apply_cfg_shape_mismatch() {
    let cond = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let uncond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = apply_cfg(&cond, &uncond, 1.5);
    assert!(result.is_err());
}

#[test]
fn test_apply_cfg_scale_one_returns_conditional() {
    let cond = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let uncond = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");
    let result = apply_cfg(&cond, &uncond, 1.0).expect("test");
    for (i, val) in result.data().iter().enumerate() {
        assert!((*val - cond.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_apply_cfg_scale_zero_returns_unconditional() {
    let cond = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let uncond = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).expect("test");
    let result = apply_cfg(&cond, &uncond, 0.0).expect("test");
    for (i, val) in result.data().iter().enumerate() {
        assert!((*val - uncond.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_apply_cfg_amplifies_difference() {
    let cond = Tensor::from_vec(vec![3], vec![2.0, 4.0, 6.0]).expect("test");
    let uncond = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    // scale=2: uncond + 2*(cond - uncond) = uncond + 2*diff
    // = [1,2,3] + 2*[1,2,3] = [3,6,9]
    let result = apply_cfg(&cond, &uncond, 2.0).expect("test");
    assert!((result.data()[0] - 3.0).abs() < 1e-6);
    assert!((result.data()[1] - 6.0).abs() < 1e-6);
    assert!((result.data()[2] - 9.0).abs() < 1e-6);
}

#[test]
fn test_apply_cfg_negative_scale() {
    // Negative scale reverses direction
    let cond = Tensor::from_vec(vec![2], vec![5.0, 10.0]).expect("test");
    let uncond = Tensor::from_vec(vec![2], vec![0.0, 0.0]).expect("test");
    let result = apply_cfg(&cond, &uncond, -1.0).expect("test");
    // uncond + (-1)*(cond - uncond) = -cond + 2*uncond = [-5, -10]
    assert!((result.data()[0] - (-5.0)).abs() < 1e-6);
    assert!((result.data()[1] - (-10.0)).abs() < 1e-6);
}

// =============================================================================
// Serialization Tests (DryConfig, XtcConfig, EtaConfig, CfgConfig)
// =============================================================================

#[test]
fn test_dry_config_debug() {
    let config = DryConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("multiplier"));
}

#[test]
fn test_xtc_config_debug() {
    let config = XtcConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("probability"));
}

#[test]
fn test_eta_config_debug() {
    let config = EtaConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("eta"));
}

#[test]
fn test_cfg_config_debug() {
    let config = CfgConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("scale"));
}

#[test]
fn test_mirostat_state_debug() {
    let state = MirostatState::default();
    let debug_str = format!("{:?}", state);
    assert!(debug_str.contains("tau"));
}

#[test]
fn test_token_healing_result_debug() {
    let result = TokenHealingResult {
        adjusted_tokens: vec![1, 2],
        prefix_constraint: Some("test".to_string()),
        tokens_removed: 1,
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("adjusted_tokens"));
}
