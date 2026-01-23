#[cfg(test)]
mod tests {
    use crate::generate::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.strategy, SamplingStrategy::Greedy);
        assert!((config.temperature - 1.0).abs() < 1e-6);
        assert!(config.eos_token_id.is_none());
    }

    #[test]
    fn test_generation_config_builders() {
        let config = GenerationConfig::greedy().with_max_tokens(50);
        assert_eq!(config.max_tokens, 50);
        assert_eq!(config.strategy, SamplingStrategy::Greedy);

        let config = GenerationConfig::top_k(10).with_temperature(0.8);
        assert_eq!(config.strategy, SamplingStrategy::TopK { k: 10 });
        assert!((config.temperature - 0.8).abs() < 1e-6);

        let config = GenerationConfig::top_p(0.9).with_eos_token_id(2);
        assert_eq!(config.strategy, SamplingStrategy::TopP { p: 0.9 });
        assert_eq!(config.eos_token_id, Some(2));
    }

    #[test]
    fn test_apply_temperature() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

        // Temperature = 1.0 should return same values
        let scaled = apply_temperature(&logits, 1.0).expect("test");
        for i in 0..4 {
            assert!((scaled.data()[i] - logits.data()[i]).abs() < 1e-6);
        }

        // Temperature = 2.0 should halve values
        let scaled = apply_temperature(&logits, 2.0).expect("test");
        assert!((scaled.data()[0] - 0.5).abs() < 1e-6);
        assert!((scaled.data()[3] - 2.0).abs() < 1e-6);

        // Temperature = 0.5 should double values
        let scaled = apply_temperature(&logits, 0.5).expect("test");
        assert!((scaled.data()[0] - 2.0).abs() < 1e-6);
        assert!((scaled.data()[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_temperature_invalid() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        assert!(apply_temperature(&logits, 0.0).is_err());
        assert!(apply_temperature(&logits, -1.0).is_err());
    }

    #[test]
    fn test_sample_greedy() {
        // Clear winner at index 2
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 2);

        // Winner at last index
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 5.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 2);

        // Winner at first index
        let logits = Tensor::from_vec(vec![3], vec![5.0, 2.0, 1.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_greedy_empty_error() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        // Single element should work
        assert_eq!(sample_greedy(&logits).expect("test"), 0);
    }

    #[test]
    fn test_sample_top_k() {
        // Strong preference for index 0
        let logits = Tensor::from_vec(vec![5], vec![100.0, 1.0, 1.0, 1.0, 1.0]).expect("test");

        // With rng_value = 0.0, should always get first (highest prob)
        let token = sample_top_k(&logits, 3, 0.0).expect("test");
        assert_eq!(token, 0);

        // With k=1, should always get highest
        let token = sample_top_k(&logits, 1, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_k_distribution() {
        // Two equally likely tokens
        let logits = Tensor::from_vec(vec![4], vec![10.0, 10.0, 0.0, 0.0]).expect("test");

        // Low rng should get index 0 or 1 (they're equal)
        let token = sample_top_k(&logits, 2, 0.1).expect("test");
        assert!(token == 0 || token == 1);

        // High rng should get index 0 or 1
        let token = sample_top_k(&logits, 2, 0.9).expect("test");
        assert!(token == 0 || token == 1);
    }

    #[test]
    fn test_sample_top_k_errors() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        assert!(sample_top_k(&logits, 0, 0.5).is_err());
    }

    #[test]
    fn test_sample_top_p() {
        // One dominant token
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");

        // With p=0.9, nucleus likely just the first token
        let token = sample_top_p(&logits, 0.9, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_p_uniform() {
        // Equal logits
        let logits = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).expect("test");

        // With p=1.0, all tokens in nucleus
        // Low rng should get early token
        let token = sample_top_p(&logits, 1.0, 0.1).expect("test");
        assert!(token < 4);

        // High rng should get later token
        let token = sample_top_p(&logits, 1.0, 0.9).expect("test");
        assert!(token < 4);
    }

    #[test]
    fn test_sample_top_p_errors() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        assert!(sample_top_p(&logits, 0.0, 0.5).is_err());
        assert!(sample_top_p(&logits, 1.1, 0.5).is_err());
        assert!(sample_top_p(&logits, -0.1, 0.5).is_err());
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let config = GenerationConfig::greedy();
        let token = sample_token(&logits, &config, 0.5).expect("test");
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_with_temperature() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let config = GenerationConfig::greedy().with_temperature(0.5);
        let token = sample_token(&logits, &config, 0.5).expect("test");
        // Higher temperature doesn't change greedy selection
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_top_k() {
        let logits = Tensor::from_vec(vec![5], vec![100.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = GenerationConfig::top_k(3);
        let token = sample_token(&logits, &config, 0.0).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_token_top_p() {
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");
        let config = GenerationConfig::top_p(0.95);
        let token = sample_token(&logits, &config, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    // =====================================================================
    // Advanced Sampling Feature Tests
    // =====================================================================

    // ----- Stop Sequence Detector Tests -----

    #[test]
    fn test_stop_sequence_detector_new() {
        let detector = StopSequenceDetector::new();
        assert!(!detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_add_token_sequence() {
        let detector = StopSequenceDetector::new()
            .with_token_sequence(vec![1, 2, 3])
            .with_token_sequence(vec![4, 5]);
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_add_string_pattern() {
        let detector = StopSequenceDetector::new()
            .with_string_pattern("<|end|>")
            .with_string_pattern("\n\n");
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_token_match() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![10, 20, 30]);

        // Add tokens one by one
        assert!(!detector.check_token(10)); // Partial match
        assert!(!detector.check_token(20)); // Still partial
        assert!(detector.check_token(30)); // Complete match!
    }

    #[test]
    fn test_stop_sequence_detector_token_no_match() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![10, 20, 30]);

        detector.check_token(10);
        detector.check_token(25); // Wrong token breaks sequence
        assert!(!detector.check_token(30)); // 30 alone doesn't match
    }

    #[test]
    fn test_stop_sequence_detector_string_match() {
        let detector = StopSequenceDetector::new().with_string_pattern("<|end|>");

        assert!(detector.check_text("Hello world").is_none());
        assert!(detector.check_text("Output: <|end|>").is_some());
        assert!(detector.check_text("<|end|> extra").is_some());
    }

    #[test]
    fn test_stop_sequence_detector_buffer_limit() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2]); // max_seq_len = 2

        // Add many tokens
        for i in 0..100 {
            detector.check_token(i);
        }

        // Detector should still work (buffer is trimmed internally)
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_reset() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2, 3]);

        detector.check_token(1);
        detector.check_token(2);
        detector.reset();

        // After reset, need to match sequence again from start
        assert!(!detector.check_token(3)); // Just 3 alone won't match
    }

    // ----- Repetition Penalty Tests -----

    #[test]
    fn test_repetition_penalty_config_default() {
        let config = RepetitionPenaltyConfig::default();
        assert_eq!(config.penalty, 1.0); // No penalty by default
        assert_eq!(config.window_size, 64);
    }

    #[test]
    fn test_repetition_penalty_config_builder() {
        let config = RepetitionPenaltyConfig::new(1.5).with_window(128);
        assert_eq!(config.penalty, 1.5);
        assert_eq!(config.window_size, 128);
    }

    #[test]
    fn test_apply_repetition_penalty_basic() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 3.0, 0.5, -1.0]).expect("test");
        let context = vec![0, 2, 4]; // Penalize tokens 0, 2, 4
        let config = RepetitionPenaltyConfig::new(2.0);

        let result = apply_repetition_penalty(&logits, &context, &config);

        // Positive logits should be divided by penalty
        assert_eq!(result.data()[0], 1.0); // 2.0 / 2.0
        assert_eq!(result.data()[1], 1.0); // Unchanged (not in context)
        assert_eq!(result.data()[2], 1.5); // 3.0 / 2.0

        // Negative logits should be multiplied by penalty
        assert_eq!(result.data()[4], -2.0); // -1.0 * 2.0
    }

    #[test]
    fn test_apply_repetition_penalty_window() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 2.0, 2.0, 2.0, 2.0]).expect("test");
        let context = vec![0, 1, 2, 3, 4]; // All tokens in context
        let config = RepetitionPenaltyConfig::new(2.0).with_window(2); // Only last 2 tokens

        let result = apply_repetition_penalty(&logits, &context, &config);

        // Only tokens 3, 4 should be penalized (last 2 in window)
        assert_eq!(result.data()[0], 2.0); // Unchanged
        assert_eq!(result.data()[1], 2.0); // Unchanged
        assert_eq!(result.data()[2], 2.0); // Unchanged
        assert_eq!(result.data()[3], 1.0); // 2.0 / 2.0
        assert_eq!(result.data()[4], 1.0); // 2.0 / 2.0
    }

    #[test]
    fn test_apply_repetition_penalty_no_penalty() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let context = vec![0, 1, 2];
        let config = RepetitionPenaltyConfig::new(1.0); // No penalty

        let result = apply_repetition_penalty(&logits, &context, &config);

        // No change when penalty is 1.0
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 3.0);
    }

    #[test]
    fn test_repetition_penalty_is_enabled() {
        let disabled = RepetitionPenaltyConfig::new(1.0);
        assert!(!disabled.is_enabled());

        let enabled = RepetitionPenaltyConfig::new(1.1);
        assert!(enabled.is_enabled());
    }

    // ----- Presence/Frequency Penalty Tests -----

    #[test]
    fn test_presence_frequency_penalty_default() {
        let config = PresenceFrequencyPenalty::default();
        assert_eq!(config.presence_penalty, 0.0);
        assert_eq!(config.frequency_penalty, 0.0);
    }

    #[test]
    fn test_presence_frequency_penalty_new() {
        let config = PresenceFrequencyPenalty::new(0.5, 0.3);
        assert_eq!(config.presence_penalty, 0.5);
        assert_eq!(config.frequency_penalty, 0.3);
    }

    #[test]
    fn test_apply_presence_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 appears twice, token 1 once
        let config = PresenceFrequencyPenalty::new(1.0, 0.0);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Tokens 0 and 1 get presence penalty (constant)
        assert_eq!(result.data()[0], 9.0); // 10.0 - 1.0
        assert_eq!(result.data()[1], 9.0); // 10.0 - 1.0
        assert_eq!(result.data()[2], 10.0); // Unchanged
    }

    #[test]
    fn test_apply_frequency_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 0, 1]; // Token 0 appears 3x, token 1 once
        let config = PresenceFrequencyPenalty::new(0.0, 1.0);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Frequency penalty is proportional to count
        assert_eq!(result.data()[0], 7.0); // 10.0 - 3*1.0
        assert_eq!(result.data()[1], 9.0); // 10.0 - 1*1.0
        assert_eq!(result.data()[2], 10.0); // Unchanged
    }

    #[test]
    fn test_apply_combined_penalties() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 appears 2x, token 1 once
        let config = PresenceFrequencyPenalty::new(0.5, 0.5);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Token 0: 10.0 - 0.5(presence) - 2*0.5(freq) = 10.0 - 1.5 = 8.5
        assert_eq!(result.data()[0], 8.5);
        // Token 1: 10.0 - 0.5(presence) - 1*0.5(freq) = 10.0 - 1.0 = 9.0
        assert_eq!(result.data()[1], 9.0);
    }

    #[test]
    fn test_presence_frequency_is_enabled() {
        let disabled = PresenceFrequencyPenalty::new(0.0, 0.0);
        assert!(!disabled.is_enabled());

        let enabled = PresenceFrequencyPenalty::new(0.1, 0.0);
        assert!(enabled.is_enabled());
    }

    // ----- Logit Bias Tests -----

    #[test]
    fn test_logit_bias_default() {
        let bias = LogitBias::default();
        assert!(bias.is_empty());
    }

    #[test]
    fn test_logit_bias_add() {
        let bias = LogitBias::new().with_bias(10, 5.0).with_bias(20, -100.0);
        assert!(!bias.is_empty());
        assert_eq!(bias.get(10), 5.0);
        assert_eq!(bias.get(20), -100.0);
        assert_eq!(bias.get(30), 0.0); // Not set
    }

    #[test]
    fn test_apply_logit_bias() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let bias = LogitBias::new()
            .with_bias(0, 10.0)
            .with_bias(2, -100.0)
            .with_bias(4, 3.0);

        let result = apply_logit_bias(&logits, &bias);

        assert_eq!(result.data()[0], 11.0); // 1.0 + 10.0
        assert_eq!(result.data()[1], 2.0); // Unchanged
        assert_eq!(result.data()[2], -97.0); // 3.0 - 100.0
        assert_eq!(result.data()[3], 4.0); // Unchanged
        assert_eq!(result.data()[4], 8.0); // 5.0 + 3.0
    }

    #[test]
    fn test_apply_logit_bias_out_of_range() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let bias = LogitBias::new().with_bias(100, 50.0); // Index out of range

        let result = apply_logit_bias(&logits, &bias);

        // Should not panic, just skip out-of-range indices
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 3.0);
    }

    // ----- Min-P Sampling Tests -----

    #[test]
    fn test_sample_min_p_basic() {
        // Token 0 has probability ~0.7, token 1 ~0.2, token 2 ~0.1
        let logits = Tensor::from_vec(vec![3], vec![1.0, -0.5, -1.0]).expect("test");

        // With min_p = 0.3 (30% of max), only token 0 should remain
        let token = sample_min_p(&logits, 0.3, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_min_p_all_pass() {
        // All tokens have similar logits
        let logits = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");

        // With min_p = 0.9, all tokens should pass (all equal)
        let token = sample_min_p(&logits, 0.9, 0.3).expect("test");
        assert!(token < 3);
    }

    #[test]
    fn test_sample_min_p_low_threshold() {
        let logits = Tensor::from_vec(vec![4], vec![10.0, 1.0, 0.5, 0.1]).expect("test");

        // With very low min_p, all tokens can be sampled
        let token = sample_min_p(&logits, 0.001, 0.99).expect("test");
        assert!(token < 4);
    }

    #[test]
    fn test_sample_min_p_edge_cases() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");

        // min_p = 0 should include all tokens
        let _ = sample_min_p(&logits, 0.0, 0.5).expect("test");

        // min_p = 1.0 should still return something (at least the max)
        let token = sample_min_p(&logits, 1.0, 0.5).expect("test");
        assert_eq!(token, 2); // Highest probability token
    }

    #[test]
    fn test_sample_min_p_rng_boundary() {
        // Test with rng_value at boundary (0.0)
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let token = sample_min_p(&logits, 0.5, 0.0).expect("test");
        assert!(token < 3);
    }

    // ----- Mirostat Sampling Tests -----

    #[test]
    fn test_mirostat_state_default() {
        let state = MirostatState::default();
        assert_eq!(state.tau, 5.0);
        assert_eq!(state.eta, 0.1);
        assert_eq!(state.mu, 10.0);
    }

    #[test]
    fn test_mirostat_state_builder() {
        let state = MirostatState::new(3.0).with_eta(0.2);
        assert_eq!(state.tau, 3.0);
        assert_eq!(state.eta, 0.2);
        assert_eq!(state.mu, 6.0); // 2 * tau
    }

    #[test]
    fn test_mirostat_state_update() {
        let mut state = MirostatState::new(5.0).with_eta(0.1);

        let initial_mu = state.mu;

        // High surprise should decrease mu (mu -= eta * (surprise - tau))
        state.update(10.0); // surprise > tau => mu decreases
        assert!(state.mu < initial_mu);

        // Reset
        state.mu = initial_mu;

        // Low surprise should increase mu
        state.update(2.0); // surprise < tau => mu increases
        assert!(state.mu > initial_mu);
    }

    #[test]
    fn test_sample_mirostat_basic() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 5.0, 1.0, 0.0, -5.0]).expect("test");
        let mut state = MirostatState::default();

        let token = sample_mirostat(&logits, &mut state, 0.5).expect("test");
        assert!(token < 5);
    }

    #[test]
    fn test_sample_mirostat_deterministic() {
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");
        let mut state = MirostatState::new(0.1); // Low target perplexity

        // With very low tau, should prefer highest probability token
        let token = sample_mirostat(&logits, &mut state, 0.0).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_mirostat_state_evolution() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 5.0, 1.0, 0.0, -5.0]).expect("test");
        let mut state = MirostatState::default();

        let initial_mu = state.mu;

        // Sample multiple times and verify mu evolves
        for _ in 0..10 {
            let _ = sample_mirostat(&logits, &mut state, 0.5).expect("test");
        }

        // Mu should have changed from initial
        assert_ne!(state.mu, initial_mu);
    }

    #[test]
    fn test_sample_mirostat_rng_boundary() {
        // Test with rng_value at boundary (1.0 - epsilon)
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let mut state = MirostatState::default();
        let token = sample_mirostat(&logits, &mut state, 0.999).expect("test");
        assert!(token < 3);
    }

    // ----- Advanced Generation Config Tests -----

    #[test]
    fn test_advanced_generation_config_default() {
        let config = AdvancedGenerationConfig::default();
        assert!(config.stop_detector.is_none());
        assert!(config.repetition_penalty.is_none());
        assert!(config.presence_frequency.is_none());
        assert!(config.logit_bias.is_none());
    }

    #[test]
    fn test_advanced_generation_config_builder() {
        let config = AdvancedGenerationConfig::new(GenerationConfig::greedy())
            .with_stop_sequences(vec!["<|end|>".to_string()])
            .with_repetition_penalty(1.5)
            .with_presence_frequency(0.5, 0.3)
            .with_logit_bias(LogitBias::new().with_bias(0, 10.0));

        assert!(config.stop_detector.is_some());
        assert!(config.repetition_penalty.is_some());
        assert!(config.presence_frequency.is_some());
        assert!(config.logit_bias.is_some());
    }

    // ----- Apply All Penalties Tests -----

    #[test]
    fn test_apply_all_penalties_empty() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let original = logits.data().to_vec();
        let context: Vec<usize> = vec![];
        let config = AdvancedGenerationConfig::default();

        let result = apply_all_penalties(&logits, &context, &config);

        // No penalties applied
        assert_eq!(result.data(), original.as_slice());
    }

    #[test]
    fn test_apply_all_penalties_combined() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 twice, token 1 once

        let config = AdvancedGenerationConfig::new(GenerationConfig::greedy())
            .with_repetition_penalty(2.0)
            .with_presence_frequency(1.0, 0.5)
            .with_logit_bias(LogitBias::new().with_bias(4, 100.0));

        let result = apply_all_penalties(&logits, &context, &config);

        // Token 4 should be highest due to bias
        let max_idx = result
            .data()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("test")
            .0;
        assert_eq!(max_idx, 4);

        // Token 0 should be penalized most (repetition + presence + frequency)
        assert!(result.data()[0] < result.data()[2]);
    }

    #[test]
    fn test_stop_sequence_with_stop_strings() {
        let detector = StopSequenceDetector::new()
            .with_stop_strings(vec!["stop".to_string(), "end".to_string()]);

        assert!(detector.check_text("this has stop in it").is_some());
        assert!(detector.check_text("the end").is_some());
        assert!(detector.check_text("nothing here").is_none());
    }

    // ===== Tail-Free Sampling (TFS) Tests =====

    #[test]
    fn test_tfs_basic_filtering() {
        // Create logits with distinct probabilities
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        // With z=0.95, should filter some low-probability tokens
        let result = sample_tfs(&logits, 0.95, 0.0);
        assert!(result.is_ok());
        // Should return one of the high-probability tokens
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_tfs_z_one_returns_greedy() {
        // z=1.0 should keep all tokens (no filtering)
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        // rng=0.0 should select the first valid token after filtering
        let result = sample_tfs(&logits, 1.0, 0.0).expect("test");
        // Should be a valid token
        assert!(result < 5);
    }

    #[test]
    fn test_tfs_z_zero_selects_top() {
        // z=0.0 should filter aggressively, keeping only top tokens
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        let result = sample_tfs(&logits, 0.01, 0.0).expect("test");
        // Should select from top tokens
        assert!(result < 3);
    }

    #[test]
    fn test_tfs_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_tfs_uniform_distribution() {
        // With uniform logits, all tokens have equal second derivative
        let logits = Tensor::from_vec(vec![5], vec![1.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5).expect("test");
        assert!(result < 5);
    }

    #[test]
    fn test_tfs_two_tokens() {
        // Test with minimum viable token count
        let logits = Tensor::from_vec(vec![2], vec![1.0, 0.5]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 2);
    }

    // ===== Locally Typical Sampling Tests =====

    #[test]
    fn test_typical_basic_sampling() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.5, 1.0, 0.5, 0.0]).expect("test");

        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_typical_p_one_keeps_all() {
        // p=1.0 should keep all tokens
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let result = sample_typical(&logits, 1.0, 0.5).expect("test");
        assert!(result < 5);
    }

    #[test]
    fn test_typical_low_p_selects_typical() {
        // Low p should select only the most typical tokens (closest to entropy)
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let result = sample_typical(&logits, 0.1, 0.0).expect("test");
        // Should select a token
        assert!(result < 5);
    }

    #[test]
    fn test_typical_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_typical_uniform_distribution() {
        // Uniform distribution - all tokens equally typical
        let logits = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5).expect("test");
        assert!(result < 4);
    }

    #[test]
    fn test_typical_two_tokens() {
        // Test with minimum viable token count
        let logits = Tensor::from_vec(vec![2], vec![1.0, 0.5]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 2);
    }

    // ===== DRY (Don't Repeat Yourself) Sampling Tests =====

    #[test]
    fn test_dry_config_default() {
        let config = DryConfig::default();
        assert_eq!(config.multiplier, 0.8);
        assert_eq!(config.base, 1.75);
        assert_eq!(config.allowed_length, 2);
        assert_eq!(config.penalty_last_n, 256);
        assert!(config.is_enabled()); // Default is enabled
    }

    #[test]
    fn test_dry_config_disabled() {
        let config = DryConfig::new(0.0);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_dry_config_enabled() {
        let config = DryConfig::new(0.5)
            .with_base(1.5)
            .with_allowed_length(3)
            .with_penalty_last_n(64);
        assert!(config.is_enabled());
        assert_eq!(config.base, 1.5);
        assert_eq!(config.allowed_length, 3);
        assert_eq!(config.penalty_last_n, 64);
    }

    #[test]
    fn test_dry_no_penalty_when_disabled() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig::new(0.0); // disabled (multiplier=0)
        let context = vec![0, 1, 0, 1, 0];

        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_dry_penalty_applied() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 64,
        };
        // Context with repeated pattern: [0, 1, 0, 1] - if next is 0, it continues [0,1] pattern
        let context = vec![0, 1, 0, 1];

        let result = apply_dry_penalty(&logits, &context, &config);
        // Token 0 should be penalized (would create [0,1,0] repetition)
        assert!(result.data()[0] < logits.data()[0]);
    }

    #[test]
    fn test_dry_short_context_no_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 3,
            penalty_last_n: 64,
        };
        // Context shorter than allowed_length
        let context = vec![0, 1];

        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_dry_respects_penalty_last_n() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 3, // Only look at last 3 tokens
        };
        // Repetition is outside the window
        let context = vec![0, 1, 2, 3, 4];

        let result = apply_dry_penalty(&logits, &context, &config);
        // Should not detect repetition from early in context
        // (penalty window is only last 3: [2, 3, 4])
        assert!(result.data().iter().sum::<f32>() > 0.0);
    }

    // ===== Beam Search Tests =====

    #[test]
    fn test_beam_hypothesis_creation() {
        let hyp = BeamHypothesis::new(vec![1, 2, 3], -1.5);
        assert_eq!(hyp.tokens.len(), 3);
        assert!(!hyp.finished);
        assert_eq!(hyp.score, -1.5);
    }

    #[test]
    fn test_beam_hypothesis_extend() {
        let hyp = BeamHypothesis::new(vec![1, 2], -1.0);
        let extended = hyp.extend(3, -0.5, false);
        assert_eq!(extended.tokens, vec![1, 2, 3]);
        assert_eq!(extended.score, -1.5);
        assert!(!extended.finished);
    }

    #[test]
    fn test_beam_hypothesis_extend_with_eos() {
        let hyp = BeamHypothesis::new(vec![1, 2], -1.0);
        let extended = hyp.extend(99, -0.5, true);
        assert_eq!(extended.tokens, vec![1, 2, 99]);
        assert!(extended.finished);
    }

    #[test]
    fn test_beam_hypothesis_normalized_score() {
        let hyp = BeamHypothesis::new(vec![1, 2, 3, 4], -4.0);
        // length_penalty = 1.0 means divide by length
        assert_eq!(hyp.normalized_score(1.0), -1.0);
        // length_penalty = 0.0 means score / 1.0 = score
        assert_eq!(hyp.normalized_score(0.0), -4.0);
    }

    #[test]
    fn test_beam_search_config_default() {
        let config = BeamSearchConfig::default();
        assert_eq!(config.num_beams, 4);
        assert_eq!(config.length_penalty, 1.0);
        assert!(config.early_stopping); // Default is true
        assert_eq!(config.num_return, 1);
    }

    #[test]
    fn test_beam_search_config_new() {
        let config = BeamSearchConfig::new(8);
        assert_eq!(config.num_beams, 8);
        assert!(config.early_stopping);
    }

    #[test]
    fn test_beam_search_config_builder() {
        let config = BeamSearchConfig::new(4)
            .with_length_penalty(0.8)
            .with_early_stopping(false)
            .with_num_return(2);
        assert_eq!(config.num_beams, 4);
        assert_eq!(config.length_penalty, 0.8);
        assert!(!config.early_stopping);
        assert_eq!(config.num_return, 2);
    }

    #[test]
    fn test_beam_search_state_creation() {
        let config = BeamSearchConfig::new(3)
            .with_length_penalty(0.8)
            .with_num_return(2);
        let state = BeamSearchState::new(config, vec![1, 2, 3]);
        assert_eq!(state.hypotheses.len(), 1); // Starts with one hypothesis
        assert!(state.finished.is_empty());
        assert_eq!(state.hypotheses[0].tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_beam_search_state_step() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Create log probabilities for 1 hypothesis, 5 tokens
        let log_probs = vec![vec![-0.1, -0.5, -1.0, -2.0, -3.0]];

        state.step(&log_probs, Some(4)); // EOS token is 4

        // Should have expanded to num_beams hypotheses
        assert!(!state.hypotheses.is_empty());
    }

    #[test]
    fn test_beam_search_state_with_finished() {
        let config = BeamSearchConfig::new(2);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Manually add some hypotheses
        state.hypotheses = vec![
            BeamHypothesis::new(vec![1, 2], -1.0),
            BeamHypothesis::new(vec![1, 3], -2.0),
        ];
        state.finished = vec![BeamHypothesis {
            tokens: vec![1, 2, 4],
            score: -1.5,
            finished: true,
        }];

        assert_eq!(state.hypotheses.len(), 2);
        assert_eq!(state.finished.len(), 1);
    }

    #[test]
    fn test_beam_search_state_should_stop_empty() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);
        state.hypotheses.clear();

        // Empty hypotheses means should stop
        assert!(state.should_stop());
    }

    #[test]
    fn test_beam_search_state_should_stop_early() {
        let config = BeamSearchConfig::new(2).with_early_stopping(true);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Not done initially
        assert!(!state.should_stop());

        // Add num_beams finished hypotheses
        state.finished.push(BeamHypothesis {
            tokens: vec![1, 2, 4],
            score: -1.0,
            finished: true,
        });
        state.finished.push(BeamHypothesis {
            tokens: vec![1, 3, 4],
            score: -1.5,
            finished: true,
        });

        // Should be done with early_stopping=true and num_beams finished
        assert!(state.should_stop());
    }

    #[test]
    fn test_beam_search_state_all_finished() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);
        state.hypotheses = vec![
            BeamHypothesis {
                tokens: vec![1],
                score: -1.0,
                finished: true,
            },
            BeamHypothesis {
                tokens: vec![2],
                score: -2.0,
                finished: true,
            },
        ];

        // All hypotheses finished
        assert!(state.should_stop());
    }

    // ===== Streaming Generation Tests =====

    #[test]
    fn test_streaming_generator_creation() {
        let generator = StreamingGenerator::new();
        assert!(generator.tokens.is_empty());
        assert!(generator.text.is_empty());
        assert!(!generator.finished);
        assert_eq!(generator.total_tokens, 0);
    }

    #[test]
    fn test_streaming_generator_default() {
        let generator = StreamingGenerator::default();
        assert!(generator.tokens.is_empty());
        assert!(!generator.finished);
    }

    #[test]
    fn test_streaming_generator_add_token() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, None);
        generator.add_token(2, Some("hello"));
        assert_eq!(generator.tokens, vec![1, 2]);
        assert_eq!(generator.text, "hello");
        assert_eq!(generator.total_tokens, 2);
    }

    #[test]
    fn test_streaming_generator_add_token_with_text() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(0, Some("Hello "));
        generator.add_token(1, Some("world"));
        generator.add_token(2, Some("!"));
        assert_eq!(generator.text, "Hello world!");
        assert_eq!(generator.token_count(), 3);
    }

    #[test]
    fn test_streaming_generator_token_count() {
        let mut generator = StreamingGenerator::new();
        assert_eq!(generator.token_count(), 0);
        generator.add_token(1, None);
        assert_eq!(generator.token_count(), 1);
        generator.add_token(2, None);
        generator.add_token(3, None);
        assert_eq!(generator.token_count(), 3);
    }

    #[test]
    fn test_streaming_generator_finish() {
        let mut generator = StreamingGenerator::new();
        assert!(!generator.finished);
        generator.add_token(1, Some("test"));
        generator.finish();
        assert!(generator.finished);
    }

    #[test]
    fn test_streaming_generator_accumulates_text() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, Some("The "));
        generator.add_token(2, Some("quick "));
        generator.add_token(3, Some("brown "));
        generator.add_token(4, Some("fox"));
        assert_eq!(generator.text, "The quick brown fox");
    }

    #[test]
    fn test_streaming_generator_none_text_no_accumulation() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, None);
        generator.add_token(2, None);
        assert!(generator.text.is_empty());
        assert_eq!(generator.tokens, vec![1, 2]);
    }

    // ===== XTC (Exclude Top Choices) Sampling Tests =====

    #[test]
    fn test_xtc_config_default() {
        let config = XtcConfig::default();
        assert_eq!(config.probability, 0.0);
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.min_keep, 1);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_xtc_config_enabled() {
        let config = XtcConfig::new(0.5).with_threshold(0.3).with_min_keep(2);
        assert!(config.is_enabled());
        assert_eq!(config.probability, 0.5);
        assert_eq!(config.threshold, 0.3);
        assert_eq!(config.min_keep, 2);
    }

    #[test]
    fn test_xtc_disabled_no_change() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::default(); // disabled
        let result = apply_xtc(&logits, &config, 0.5);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_xtc_rng_above_probability_no_change() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::new(0.5); // 50% probability
        let result = apply_xtc(&logits, &config, 0.8); // rng > probability
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_xtc_excludes_top_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::new(1.0).with_threshold(0.5); // Always exclude, high threshold
        let result = apply_xtc(&logits, &config, 0.0); // rng < probability
                                                       // Top token (index 0) should be excluded (set to NEG_INFINITY)
        assert_eq!(result.data()[0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_xtc_respects_min_keep() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 9.0, 8.0]).expect("test");
        let config = XtcConfig::new(1.0).with_threshold(0.1).with_min_keep(2);
        let result = apply_xtc(&logits, &config, 0.0);
        // Should keep at least 2 tokens (not set all to NEG_INFINITY)
        let finite_count = result.data().iter().filter(|&&x| x.is_finite()).count();
        assert!(finite_count >= 2);
    }

    // ===== Eta Sampling Tests =====

    #[test]
    fn test_eta_config_default() {
        let config = EtaConfig::default();
        assert_eq!(config.eta, 0.3);
        assert_eq!(config.min_p, 0.0001);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_eta_config_disabled() {
        let config = EtaConfig::new(0.0);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_eta_config_builder() {
        let config = EtaConfig::new(0.5).with_min_p(0.001);
        assert_eq!(config.eta, 0.5);
        assert_eq!(config.min_p, 0.001);
    }

    #[test]
    fn test_eta_sampling_basic() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_eta_sampling_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_eta_sampling_uniform() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5).expect("test");
        assert!(result < 4);
    }

    // ===== Token Healing Tests =====

    #[test]
    fn test_token_healing_config_default() {
        let config = TokenHealingConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_backup_chars, 0);
    }

    #[test]
    fn test_token_healing_config_enabled() {
        let config = TokenHealingConfig::new(true).with_max_backup(15);
        assert!(config.enabled);
        assert_eq!(config.max_backup_chars, 15);
    }

    #[test]
    fn test_token_healing_no_heal_needed() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = analyze_token_healing(&tokens, Some("hello"));
        assert_eq!(result.adjusted_tokens, tokens);
        assert!(result.prefix_constraint.is_none());
        assert_eq!(result.tokens_removed, 0);
    }

    #[test]
    fn test_token_healing_partial_word() {
        let tokens = vec![1, 2, 3, 4, 5];
        // "wo" is a short alphanumeric token without leading space - should heal
        let result = analyze_token_healing(&tokens, Some("wo"));
        assert_eq!(result.adjusted_tokens, vec![1, 2, 3, 4]);
        assert_eq!(result.prefix_constraint, Some("wo".to_string()));
        assert_eq!(result.tokens_removed, 1);
    }

    #[test]
    fn test_token_healing_empty_tokens() {
        let tokens: Vec<usize> = vec![];
        let result = analyze_token_healing(&tokens, Some("a"));
        assert!(result.adjusted_tokens.is_empty());
        assert!(result.prefix_constraint.is_none());
    }

    #[test]
    fn test_token_healing_space_prefix_no_heal() {
        let tokens = vec![1, 2, 3];
        // Token starting with space - no healing needed
        let result = analyze_token_healing(&tokens, Some(" word"));
        assert_eq!(result.adjusted_tokens, tokens);
        assert!(result.prefix_constraint.is_none());
    }

    // ===== Classifier-Free Guidance (CFG) Tests =====

    #[test]
    fn test_cfg_config_default() {
        let config = CfgConfig::default();
        assert_eq!(config.scale, 1.0);
        assert!(config.negative_prompt_tokens.is_empty());
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_cfg_config_enabled() {
        let config = CfgConfig::new(1.5).with_negative_prompt(vec![1, 2, 3]);
        assert!(config.is_enabled());
        assert_eq!(config.scale, 1.5);
        assert_eq!(config.negative_prompt_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_cfg_scale_one_no_change() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![4], vec![0.5, 1.5, 2.5, 3.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 1.0).expect("test");
        // scale=1.0: uncond + 1.0 * (cond - uncond) = cond
        assert_eq!(result.data(), cond.data());
    }

    #[test]
    fn test_cfg_scale_zero_returns_uncond() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![4], vec![0.5, 1.5, 2.5, 3.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 0.0).expect("test");
        // scale=0.0: uncond + 0.0 * (cond - uncond) = uncond
        assert_eq!(result.data(), uncond.data());
    }

    #[test]
    fn test_cfg_amplifies_difference() {
        let cond = Tensor::from_vec(vec![3], vec![2.0, 1.0, 0.0]).expect("test");
        let uncond = Tensor::from_vec(vec![3], vec![1.0, 1.0, 1.0]).expect("test");
        let result = apply_cfg(&cond, &uncond, 2.0).expect("test");
        // scale=2.0: uncond + 2.0 * (cond - uncond)
        // = [1,1,1] + 2*([2,1,0] - [1,1,1])
        // = [1,1,1] + 2*[1,0,-1]
        // = [1,1,1] + [2,0,-2]
        // = [3,1,-1]
        assert_eq!(result.data(), &[3.0, 1.0, -1.0]);
    }

    #[test]
    fn test_cfg_shape_mismatch_error() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![3], vec![0.5, 1.5, 2.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 1.5);
        assert!(result.is_err());
    }

    // ===== Prompt Cache Tests =====

    #[test]
    fn test_prompt_cache_creation() {
        let cache = PromptCache::new(50);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_prompt_cache_default() {
        let cache = PromptCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prompt_cache_add_and_find() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);
        assert_eq!(cache.len(), 1);

        // Find exact match
        let result = cache.find_prefix(&[1, 2, 3]);
        assert!(result.is_some());
        let (len, kv_hash) = result.expect("test");
        assert_eq!(len, 3);
        assert_eq!(kv_hash, 12345);
    }

    #[test]
    fn test_prompt_cache_find_prefix() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2], 111);
        cache.add(vec![1, 2, 3], 222);

        // Should find longer prefix first
        let result = cache.find_prefix(&[1, 2, 3, 4]);
        assert!(result.is_some());
        let (len, _) = result.expect("test");
        assert_eq!(len, 3);
    }

    #[test]
    fn test_prompt_cache_miss() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);

        // No matching prefix
        let result = cache.find_prefix(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn test_prompt_cache_clear() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);
        cache.add(vec![4, 5, 6], 67890);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prompt_cache_stats() {
        let mut cache = PromptCache::new(100);
        cache.add(vec![1, 2, 3], 12345);

        // Hit the cache
        cache.find_prefix(&[1, 2, 3]);
        cache.find_prefix(&[1, 2, 3]);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.total_hits, 2);
        assert_eq!(stats.max_entries, 100);
    }

    #[test]
    fn test_prompt_cache_eviction() {
        let mut cache = PromptCache::new(2);
        cache.add(vec![1], 111);
        cache.add(vec![2], 222);
        assert_eq!(cache.len(), 2);

        // Adding third entry should evict LRU
        cache.add(vec![3], 333);
        assert_eq!(cache.len(), 2);
    }

    // ========================================================================
    // Dynamic Temperature Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_dyn_temp_config_default() {
        let config = DynTempConfig::default();
        assert!((config.temp - 1.0).abs() < 1e-6);
        assert!((config.delta - 0.0).abs() < 1e-6);
        assert!((config.exponent - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_config_new() {
        let config = DynTempConfig::new(0.8, 0.2, 1.5);
        assert!((config.temp - 0.8).abs() < 1e-6);
        assert!((config.delta - 0.2).abs() < 1e-6);
        assert!((config.exponent - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_config_static() {
        let config = DynTempConfig::static_temp(0.5);
        assert!((config.temp - 0.5).abs() < 1e-6);
        assert!((config.delta - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_no_delta_uses_static() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DynTempConfig::static_temp(0.5);

        let result = apply_dynamic_temperature(&logits, &config);
        let static_result = apply_temperature(&logits, 0.5).expect("test");

        // Should be identical to static temperature
        for (a, b) in result.data().iter().zip(static_result.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dyn_temp_single_element() {
        let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // Single element should return unchanged
        assert!((result.data()[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_low_entropy_higher_temp() {
        // Low entropy (one dominant logit) should use higher temperature
        let logits = Tensor::from_vec(vec![5], vec![10.0, 0.0, 0.0, 0.0, 0.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // Result should be scaled, but logits should still be ordered
        assert!(result.data()[0] > result.data()[1]);
    }

    #[test]
    fn test_dyn_temp_high_entropy_lower_temp() {
        // High entropy (uniform logits) should use lower temperature
        let logits = Tensor::from_vec(vec![5], vec![1.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // With uniform logits and high entropy, should use max temp
        // All values should be close to 1.0 (uniform scaled)
        let sum: f32 = result.data().iter().sum();
        assert!(sum.abs() > 0.0); // Non-degenerate
    }

    #[test]
    fn test_dyn_temp_exponent_affects_scaling() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.5, 1.0, 0.5, 0.0]).expect("test");
        let config_exp1 = DynTempConfig::new(1.0, 0.5, 1.0);
        let config_exp2 = DynTempConfig::new(1.0, 0.5, 2.0);

        let result1 = apply_dynamic_temperature(&logits, &config_exp1);
        let result2 = apply_dynamic_temperature(&logits, &config_exp2);

        // Different exponents should produce different results
        let diff: f32 = result1
            .data()
            .iter()
            .zip(result2.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6);
    }

    // ========================================================================
    // Infill/FIM Sampler Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_infill_config_default() {
        let config = InfillConfig::default();
        assert!(config.eog_tokens.is_empty());
        assert!((config.eog_ratio_threshold - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_config_new() {
        let config = InfillConfig::new(vec![1, 2, 3]);
        assert_eq!(config.eog_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_infill_config_with_threshold() {
        let config = InfillConfig::new(vec![1]).with_threshold(5.0);
        assert!((config.eog_ratio_threshold - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_empty_eog_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = InfillConfig::default();

        let result = apply_infill_sampling(&logits, &config);
        assert!(!result.force_eog);
        assert!((result.p_txt - 1.0).abs() < 1e-6);
        assert!((result.p_eog - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_no_force_eog_when_text_dominant() {
        // Text tokens have much higher probability than EOG
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 0.0]).expect("test");
        let config = InfillConfig::new(vec![4]); // Token 4 is EOG

        let result = apply_infill_sampling(&logits, &config);
        assert!(!result.force_eog);
        assert!(result.p_txt > result.p_eog);
    }

    #[test]
    fn test_infill_force_eog_when_eog_dominant() {
        // EOG token has high probability relative to text
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 0.0, 10.0]).expect("test");
        let config = InfillConfig::new(vec![4]); // Token 4 is EOG

        let result = apply_infill_sampling(&logits, &config);
        assert!(result.force_eog);
        assert!(result.p_eog > 0.5);
    }

    #[test]
    fn test_infill_modified_logits_when_force_eog() {
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 0.0, 10.0]).expect("test");
        let config = InfillConfig::new(vec![4]);

        let result = apply_infill_sampling(&logits, &config);
        if result.force_eog {
            // Non-EOG tokens should be -inf
            assert!(result.logits.data()[0] == f32::NEG_INFINITY);
            assert!(result.logits.data()[1] == f32::NEG_INFINITY);
            // EOG token should remain
            assert!(result.logits.data()[4] > f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_infill_multiple_eog_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 5.0, 5.0]).expect("test");
        let config = InfillConfig::new(vec![3, 4]); // Tokens 3 and 4 are EOG

        let result = apply_infill_sampling(&logits, &config);
        // Check that both EOG tokens contribute to p_eog
        assert!(result.p_eog > 0.0);
    }

    // ========================================================================
    // Sampler Chain Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_sampler_chain_new() {
        let chain = SamplerChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_sampler_chain_default() {
        let chain = SamplerChain::default();
        assert!(chain.is_empty());
    }

    #[test]
    fn test_sampler_chain_with_sampler() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(0.8));
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.names(), vec!["temperature"]);
    }

    #[test]
    fn test_sampler_chain_multiple_samplers() {
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(50))
            .with_sampler(TopPSampler::new(0.9));

        assert_eq!(chain.len(), 3);
        assert_eq!(chain.names(), vec!["temperature", "top_k", "top_p"]);
    }

    #[test]
    fn test_sampler_chain_push() {
        let mut chain = SamplerChain::new();
        chain.push(Box::new(TemperatureSampler::new(0.5)));
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_sampler_chain_apply() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(0.5));

        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let context = SamplerContext::new();

        chain.apply(&mut logits, &context);

        // Temperature 0.5 should double the logits
        assert!((logits.data()[0] - 2.0).abs() < 1e-6);
        assert!((logits.data()[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sampler_chain_sample() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(1.0));

        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let context = SamplerContext::new();

        let result = chain.sample(&logits, &context).expect("test");
        assert_eq!(result, 4); // Greedy should pick max
    }

    #[test]
    fn test_sampler_chain_clone() {
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(10));

        let cloned = chain.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.names(), vec!["temperature", "top_k"]);
    }

    #[test]
    fn test_sampler_context_default() {
        let ctx = SamplerContext::default();
        assert!(ctx.tokens.is_empty());
        assert!((ctx.rng_value - 0.0).abs() < 1e-6);
        assert_eq!(ctx.step, 0);
    }

    #[test]
    fn test_sampler_context_builders() {
        let ctx = SamplerContext::new()
            .with_tokens(vec![1, 2, 3])
            .with_rng(0.5)
            .with_step(10);

        assert_eq!(ctx.tokens, vec![1, 2, 3]);
        assert!((ctx.rng_value - 0.5).abs() < 1e-6);
        assert_eq!(ctx.step, 10);
    }

    #[test]
    fn test_temperature_sampler() {
        let sampler = TemperatureSampler::new(0.5);
        assert_eq!(sampler.name(), "temperature");
    }

    #[test]
    fn test_dyn_temp_sampler() {
        let sampler = DynTempSampler::new(DynTempConfig::new(1.0, 0.5, 1.0));
        assert_eq!(sampler.name(), "dyn_temp");
    }

    #[test]
    fn test_top_k_sampler() {
        let sampler = TopKSampler::new(10);
        assert_eq!(sampler.name(), "top_k");
        assert_eq!(sampler.k, 10);
    }

    #[test]
    fn test_top_p_sampler() {
        let sampler = TopPSampler::new(0.9);
        assert_eq!(sampler.name(), "top_p");
        assert!((sampler.p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_sampler() {
        let sampler = RepetitionPenaltySampler::new(RepetitionPenaltyConfig::new(1.2));
        assert_eq!(sampler.name(), "repetition_penalty");
    }

    #[test]
    fn test_infill_sampler() {
        let sampler = InfillSampler::new(InfillConfig::new(vec![1, 2]));
        assert_eq!(sampler.name(), "infill");
    }

    #[test]
    fn test_top_k_sampler_apply() {
        let sampler = TopKSampler::new(2);
        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 5.0, 3.0, 2.0, 4.0]).expect("test");
        let context = SamplerContext::new();

        sampler.apply(&mut logits, &context);

        // Only top 2 (indices 1 and 4) should remain
        let data = logits.data();
        assert!(data[0] == f32::NEG_INFINITY);
        assert!(data[1] > f32::NEG_INFINITY); // 5.0 is top
        assert!(data[2] == f32::NEG_INFINITY);
        assert!(data[3] == f32::NEG_INFINITY);
        assert!(data[4] > f32::NEG_INFINITY); // 4.0 is second
    }

    #[test]
    fn test_top_p_sampler_apply() {
        let sampler = TopPSampler::new(0.5);
        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 5.0, 2.0, 0.0, 0.0]).expect("test");
        let context = SamplerContext::new();

        sampler.apply(&mut logits, &context);

        // Top token (index 1 with 5.0) should definitely remain
        let data = logits.data();
        assert!(data[1] > f32::NEG_INFINITY);
    }

    #[test]
    fn test_full_sampler_pipeline() {
        // Build a realistic pipeline: temp -> top_k -> top_p
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(50))
            .with_sampler(TopPSampler::new(0.95));

        let logits = Tensor::from_vec(
            vec![10],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("test");
        let context = SamplerContext::new();

        let result = chain.sample(&logits, &context).expect("test");
        assert_eq!(result, 9); // Should still pick max after pipeline
    }

    // =========================================================================
    // LogitProcessor Tests (RLZR-GEN-001)
    // =========================================================================

    #[test]
    fn test_logit_processor_context() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let ctx = LogitProcessorContext::new(&tokens, 3, 1000);

        assert_eq!(ctx.tokens, &[1, 2, 3, 4, 5]);
        assert_eq!(ctx.step, 3);
        assert_eq!(ctx.n_vocab, 1000);
    }

    #[test]
    fn test_token_suppressor_basic() {
        let suppressor = TokenSuppressor::new(vec![0, 5, 9]);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ctx = LogitProcessorContext::new(&[], 0, 10);

        suppressor.process(&mut logits, &ctx);

        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!(logits[5].is_infinite() && logits[5] < 0.0);
        assert!(logits[9].is_infinite() && logits[9] < 0.0);
    }

    #[test]
    fn test_token_suppressor_out_of_bounds() {
        let suppressor = TokenSuppressor::new(vec![100, 200]); // Out of bounds
        let mut logits = vec![1.0, 2.0, 3.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        // Should not panic
        suppressor.process(&mut logits, &ctx);

        // Logits unchanged
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_token_suppressor_name() {
        let suppressor = TokenSuppressor::new(vec![]);
        assert_eq!(suppressor.name(), "token_suppressor");
    }

    #[test]
    fn test_repetition_penalty_basic() {
        let penalty = RepetitionPenalty::with_penalty(2.0);
        let tokens = vec![1u32, 3, 5];
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 6);

        penalty.process(&mut logits, &ctx);

        // Token 1 (logit 2.0) should be halved: 2.0 / 2.0 = 1.0
        assert!((logits[1] - 1.0).abs() < 1e-6);
        // Token 3 (logit 4.0) should be halved: 4.0 / 2.0 = 2.0
        assert!((logits[3] - 2.0).abs() < 1e-6);
        // Token 5 (logit 6.0) should be halved: 6.0 / 2.0 = 3.0
        assert!((logits[5] - 3.0).abs() < 1e-6);
        // Token 0 unchanged
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let penalty = RepetitionPenalty::with_penalty(2.0);
        let tokens = vec![0u32];
        let mut logits = vec![-2.0, 1.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 2);

        penalty.process(&mut logits, &ctx);

        // Negative logit should be multiplied: -2.0 * 2.0 = -4.0
        assert!((logits[0] - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_with_window() {
        let penalty = RepetitionPenalty::new(2.0, 2); // Window of 2
        let tokens = vec![1u32, 2, 3, 4]; // Only last 2 (3, 4) should be penalized
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 5);

        penalty.process(&mut logits, &ctx);

        // Token 1, 2 NOT penalized (outside window)
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
        // Token 3, 4 penalized (inside window)
        assert!((logits[3] - 2.0).abs() < 1e-6); // 4.0 / 2.0
        assert!((logits[4] - 2.5).abs() < 1e-6); // 5.0 / 2.0
    }

    #[test]
    fn test_temperature_scaler_basic() {
        let scaler = TemperatureScaler::new(2.0);
        let mut logits = vec![2.0, 4.0, 6.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        scaler.process(&mut logits, &ctx);

        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_scaler_no_effect_at_1() {
        let scaler = TemperatureScaler::new(1.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        scaler.process(&mut logits, &ctx);

        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_temperature_scaler_panics_on_zero() {
        let _ = TemperatureScaler::new(0.0);
    }

    #[test]
    fn test_processor_chain_empty() {
        let chain = LogitProcessorChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_processor_chain_add() {
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(RepetitionPenalty::with_penalty(1.5));

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_processor_chain_names() {
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(RepetitionPenalty::with_penalty(1.5))
            .with_processor(TemperatureScaler::new(0.8));

        let names = chain.processor_names();
        assert_eq!(
            names,
            vec![
                "token_suppressor",
                "repetition_penalty",
                "temperature_scaler"
            ]
        );
    }

    #[test]
    fn test_processor_chain_applies_in_order() {
        // Suppress token 0, then apply temp scaling
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(TemperatureScaler::new(2.0));

        let mut logits = vec![10.0, 4.0, 2.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        chain.process(&mut logits, &ctx);

        // Token 0 suppressed (still -inf after scaling)
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        // Other logits scaled
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_processor_chain_as_logit_processor() {
        let chain = LogitProcessorChain::new().with_processor(TokenSuppressor::new(vec![0]));

        // Use as dyn LogitProcessor
        let processor: &dyn LogitProcessor = &chain;
        assert_eq!(processor.name(), "processor_chain");

        let mut logits = vec![1.0, 2.0];
        let ctx = LogitProcessorContext::new(&[], 0, 2);
        processor.process(&mut logits, &ctx);

        assert!(logits[0].is_infinite());
    }

    // =========================================================================
    // GenerationPipeline Tests
    // =========================================================================

    /// Mock model for testing GenerationPipeline
    struct MockModel {
        vocab_size: usize,
        /// Returns logits with this token as highest
        highest_token: usize,
        call_count: usize,
    }

    impl MockModel {
        fn new(vocab_size: usize, highest_token: usize) -> Self {
            Self {
                vocab_size,
                highest_token,
                call_count: 0,
            }
        }
    }

    impl GenerativeModel for MockModel {
        fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
            self.call_count += 1;
            let mut logits = vec![0.0f32; self.vocab_size];
            logits[self.highest_token] = 10.0;
            Ok(logits)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    #[test]
    fn test_generation_pipeline_basic() {
        let model = MockModel::new(100, 42);
        let mut pipeline = GenerationPipeline::new(model)
            .with_config(GenerationConfig::greedy().with_max_tokens(3));

        let result = pipeline.generate(&[1, 2]).expect("test");

        // Initial tokens + 3 generated
        assert_eq!(result.len(), 5);
        // All generated tokens should be 42 (highest)
        assert_eq!(result[2], 42);
        assert_eq!(result[3], 42);
        assert_eq!(result[4], 42);
    }

    #[test]
    fn test_generation_pipeline_with_eos() {
        // Model that returns EOS token (99) on third call
        struct EosModel {
            call_count: usize,
        }
        impl GenerativeModel for EosModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                self.call_count += 1;
                let mut logits = vec![0.0f32; 100];
                if self.call_count >= 3 {
                    logits[99] = 10.0; // EOS
                } else {
                    logits[50] = 10.0; // Regular token
                }
                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                100
            }
        }

        let model = EosModel { call_count: 0 };
        let mut pipeline = GenerationPipeline::new(model).with_config(
            GenerationConfig::greedy()
                .with_max_tokens(10)
                .with_eos_token_id(99),
        );

        let result = pipeline.generate(&[1]).expect("test");

        // Should stop at EOS: [1, 50, 50, 99]
        assert_eq!(result.len(), 4);
        assert_eq!(result[result.len() - 1], 99);
    }

    #[test]
    fn test_generation_pipeline_with_token_suppression() {
        // Model that would return token 0 if not suppressed
        struct ZeroModel;
        impl GenerativeModel for ZeroModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                let mut logits = vec![0.0f32; 10];
                logits[0] = 10.0; // Token 0 is highest
                logits[5] = 5.0; // Token 5 is second highest
                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                10
            }
        }

        let model = ZeroModel;
        let mut pipeline = GenerationPipeline::new(model)
            .add_processor(TokenSuppressor::new(vec![0])) // Suppress token 0
            .with_config(GenerationConfig::greedy().with_max_tokens(1));

        let result = pipeline.generate(&[1]).expect("test");

        // Should pick token 5 (second highest) since 0 is suppressed
        assert_eq!(result, vec![1, 5]);
    }

    #[test]
    fn test_generation_pipeline_whisper_use_case() {
        // Simulate Whisper: suppress SOT (50257) to prevent hallucination
        const SOT: u32 = 50257;
        const EOT: u32 = 50256;

        struct WhisperMockModel {
            call_count: usize,
        }
        impl GenerativeModel for WhisperMockModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                self.call_count += 1;
                let mut logits = vec![0.0f32; 51865];

                // Test scenario: SOT has highest logit (intentional for testing SOT suppression)
                logits[SOT as usize] = 11.0;

                // Text token has second highest
                logits[440] = 10.0; // "The" token

                // EOT after 3 calls
                if self.call_count >= 4 {
                    logits[EOT as usize] = 20.0;
                }

                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                51865
            }
        }

        let model = WhisperMockModel { call_count: 0 };
        let mut pipeline = GenerationPipeline::new(model)
            .add_processor(TokenSuppressor::new(vec![SOT])) // Suppress SOT
            .with_config(
                GenerationConfig::greedy()
                    .with_max_tokens(10)
                    .with_eos_token_id(EOT as usize),
            );

        let result = pipeline.generate(&[50257, 50258]).expect("test");

        // Should NOT contain SOT (50257) in generated tokens
        for &token in &result[2..] {
            // Skip initial tokens
            assert_ne!(token, SOT, "SOT should be suppressed");
        }

        // Should contain the text token and EOT
        assert!(result.contains(&440), "Should contain text token");
        assert!(result.contains(&EOT), "Should end with EOT");
    }
}
