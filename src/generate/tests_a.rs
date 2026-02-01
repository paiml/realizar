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
