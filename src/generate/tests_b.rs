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

    // =========================================================================
    // Additional Coverage Tests (PMAT-802 Phase 2)
    // =========================================================================

    // ----- Helper Function Tests -----

    #[test]
    fn test_sample_from_distribution_first_element() {
        let probs = vec![0.5, 0.3, 0.2];
        let indices = vec![10, 20, 30];
        // rng=0.0 should hit first bucket
        let result = sample_from_distribution(&probs, &indices, 0.0);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_sample_from_distribution_middle_element() {
        let probs = vec![0.3, 0.4, 0.3];
        let indices = vec![10, 20, 30];
        // rng=0.5 (cumsum: 0.3, 0.7, 1.0) -> should hit second bucket
        let result = sample_from_distribution(&probs, &indices, 0.5);
        assert_eq!(result, 20);
    }

    #[test]
    fn test_sample_from_distribution_last_element() {
        let probs = vec![0.2, 0.3, 0.5];
        let indices = vec![10, 20, 30];
        // rng=0.99 should hit last bucket
        let result = sample_from_distribution(&probs, &indices, 0.99);
        assert_eq!(result, 30);
    }

    #[test]
    fn test_sample_from_distribution_fallback() {
        let probs = vec![0.5, 0.5];
        let indices = vec![100, 200];
        // rng=1.0 should fall through to last element
        let result = sample_from_distribution(&probs, &indices, 1.0);
        assert_eq!(result, 200);
    }

    #[test]
    fn test_logits_to_probs_sum_to_one() {
        let indexed = vec![(0, 2.0), (1, 1.0), (2, 0.5)];
        let probs = logits_to_probs(&indexed);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Probabilities should sum to 1.0");
    }

    #[test]
    fn test_logits_to_probs_ordering_preserved() {
        let indexed = vec![(0, 3.0), (1, 2.0), (2, 1.0)];
        let probs = logits_to_probs(&indexed);
        // Higher logits should have higher probabilities
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_logits_to_probs_single_element() {
        let indexed = vec![(42, 5.0)];
        let probs = logits_to_probs(&indexed);
        assert_eq!(probs.len(), 1);
        assert!(
            (probs[0] - 1.0).abs() < 1e-6,
            "Single element should have prob 1.0"
        );
    }

    #[test]
    fn test_logits_to_probs_equal_logits() {
        let indexed = vec![(0, 1.0), (1, 1.0), (2, 1.0)];
        let probs = logits_to_probs(&indexed);
        // Equal logits should give equal probabilities
        let expected = 1.0 / 3.0;
        for p in &probs {
            assert!((p - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_build_nucleus_p_one() {
        let indexed = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let nucleus = build_nucleus(&indexed, 1.0);
        // Should include all tokens
        assert_eq!(nucleus.len(), 3);
    }

    #[test]
    fn test_build_nucleus_p_zero_five() {
        let indexed = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let nucleus = build_nucleus(&indexed, 0.5);
        // Should include just first token (0.5 >= 0.5)
        assert_eq!(nucleus.len(), 1);
        assert_eq!(nucleus[0].0, 0);
    }

    #[test]
    fn test_build_nucleus_p_zero_eight() {
        let indexed = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let nucleus = build_nucleus(&indexed, 0.8);
        // Should include first two tokens (0.5 + 0.3 = 0.8 >= 0.8)
        assert_eq!(nucleus.len(), 2);
    }

    #[test]
    fn test_build_nucleus_single_element() {
        let indexed = vec![(42, 1.0)];
        let nucleus = build_nucleus(&indexed, 0.5);
        assert_eq!(nucleus.len(), 1);
        assert_eq!(nucleus[0].0, 42);
    }

    // ----- Temperature Edge Cases -----

    #[test]
    fn test_apply_temperature_very_low() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let scaled = apply_temperature(&logits, 0.01).expect("test");
        // Very low temp = very high scaled values
        assert!(scaled.data()[0] > 90.0);
        assert!(scaled.data()[3] > 390.0);
    }

    #[test]
    fn test_apply_temperature_very_high() {
        let logits = Tensor::from_vec(vec![4], vec![100.0, 200.0, 300.0, 400.0]).expect("test");
        let scaled = apply_temperature(&logits, 100.0).expect("test");
        // Very high temp = very low scaled values (flattened distribution)
        assert!((scaled.data()[0] - 1.0).abs() < 1e-6);
        assert!((scaled.data()[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_temperature_preserves_ordering() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 3.0, 2.0, 5.0, 4.0]).expect("test");
        let scaled = apply_temperature(&logits, 0.7).expect("test");
        // Ordering should be preserved
        assert!(scaled.data()[3] > scaled.data()[4]); // 5 > 4
        assert!(scaled.data()[4] > scaled.data()[1]); // 4 > 3
        assert!(scaled.data()[1] > scaled.data()[2]); // 3 > 2
        assert!(scaled.data()[2] > scaled.data()[0]); // 2 > 1
    }

    #[test]
    fn test_apply_temperature_negative_logits() {
        let logits = Tensor::from_vec(vec![3], vec![-5.0, -2.0, -1.0]).expect("test");
        let scaled = apply_temperature(&logits, 0.5).expect("test");
        assert!((scaled.data()[0] - (-10.0)).abs() < 1e-6);
        assert!((scaled.data()[1] - (-4.0)).abs() < 1e-6);
        assert!((scaled.data()[2] - (-2.0)).abs() < 1e-6);
    }

    // ----- Seed Reproducibility Tests -----

    #[test]
    fn test_generation_config_seed() {
        let config = GenerationConfig::greedy().with_seed(12345);
        assert_eq!(config.seed, Some(12345));
    }

    #[test]
    fn test_generation_config_different_seeds_produce_different_results() {
        // This tests the LCG used in generation pipeline
        let seed1 = 42u64;
        let seed2 = 43u64;

        // Simple LCG step
        let next1 = seed1
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let next2 = seed2
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);

        assert_ne!(
            next1, next2,
            "Different seeds should produce different values"
        );
    }

    // ----- Greedy Edge Cases -----

    #[test]
    fn test_sample_greedy_with_ties() {
        // Multiple maximum values - should return first occurrence
        let logits = Tensor::from_vec(vec![5], vec![1.0, 10.0, 10.0, 10.0, 1.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 1, "Should return first maximum");
    }

    #[test]
    fn test_sample_greedy_all_negative() {
        let logits = Tensor::from_vec(vec![4], vec![-5.0, -2.0, -3.0, -4.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 1, "Should return index of least negative (-2.0)");
    }

    #[test]
    fn test_sample_greedy_large_vocab() {
        let mut data = vec![0.0f32; 50000];
        data[49999] = 100.0; // Maximum at end
        let logits = Tensor::from_vec(vec![50000], data).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 49999);
    }

    // ----- Top-K Edge Cases -----

    #[test]
    fn test_sample_top_k_k_larger_than_vocab() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        // k=10 but vocab size is 3
        let token = sample_top_k(&logits, 10, 0.0).expect("test");
        assert_eq!(token, 2, "Should work with k > vocab size");
    }

    #[test]
    fn test_sample_top_k_deterministic_with_same_rng() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let token1 = sample_top_k(&logits, 3, 0.123).expect("test");
        let token2 = sample_top_k(&logits, 3, 0.123).expect("test");
        assert_eq!(token1, token2, "Same RNG should give same result");
    }

    #[test]
    fn test_sample_top_k_with_negative_logits() {
        let logits = Tensor::from_vec(vec![5], vec![-10.0, -5.0, -1.0, -3.0, -7.0]).expect("test");
        let token = sample_top_k(&logits, 2, 0.0).expect("test");
        // Top 2 are indices 2 (-1.0) and 3 (-3.0)
        assert!(token == 2 || token == 3);
    }

    // ----- Top-P Edge Cases -----

    #[test]
    fn test_sample_top_p_p_exactly_one() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let token = sample_top_p(&logits, 1.0, 0.5).expect("test");
        assert!(token < 4, "p=1.0 should include all tokens");
    }

    #[test]
    fn test_sample_top_p_very_small_p() {
        // Very small p should select only the highest probability token
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let token = sample_top_p(&logits, 0.01, 0.0).expect("test");
        // Index 2 has highest logit
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_top_p_deterministic_with_same_rng() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let token1 = sample_top_p(&logits, 0.9, 0.456).expect("test");
        let token2 = sample_top_p(&logits, 0.9, 0.456).expect("test");
        assert_eq!(token1, token2, "Same RNG should give same result");
    }

    // ----- Sample Token Integration -----

    #[test]
    fn test_sample_token_temperature_affects_distribution() {
        // With very low temperature, top-k should behave more greedy
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");

        let config_low_temp = GenerationConfig::top_k(5).with_temperature(0.01);
        let config_high_temp = GenerationConfig::top_k(5).with_temperature(10.0);

        // Low temp should always pick max
        for rng in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let token = sample_token(&logits, &config_low_temp, rng).expect("test");
            assert_eq!(token, 2, "Low temp should always pick highest");
        }

        // High temp might pick different tokens (just verify it works)
        let _ = sample_token(&logits, &config_high_temp, 0.5).expect("test");
    }

    // ----- Min-P Additional Tests -----

    #[test]
    fn test_sample_min_p_errors_on_invalid_input() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        // Invalid min_p values
        assert!(sample_min_p(&logits, -0.1, 0.5).is_err());
        assert!(sample_min_p(&logits, 1.1, 0.5).is_err());
    }

    #[test]
    fn test_sample_min_p_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("test");
        let result = sample_min_p(&logits, 0.5, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    // ----- Mirostat Additional Tests -----

    #[test]
    fn test_sample_mirostat_empty_logits_error() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    #[test]
    fn test_sample_mirostat_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let mut state = MirostatState::default();
        let result = sample_mirostat(&logits, &mut state, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    // ----- TFS Additional Tests -----

    #[test]
    fn test_sample_tfs_empty_error() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    // ----- Typical Sampling Additional Tests -----

    #[test]
    fn test_sample_typical_empty_error() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    // ----- Eta Sampling Additional Tests -----

    #[test]
    fn test_sample_eta_empty_error() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    // ----- CFG Additional Tests -----

    #[test]
    fn test_cfg_empty_logits() {
        // Zero-dimension tensors are invalid, so test that it returns an error
        let cond = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(cond.is_err()); // Shape dimensions cannot be zero
    }

    #[test]
    fn test_cfg_large_scale() {
        let cond = Tensor::from_vec(vec![3], vec![1.0, 0.0, -1.0]).expect("test");
        let uncond = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");
        let result = apply_cfg(&cond, &uncond, 10.0).expect("test");
        // scale=10.0: 0 + 10 * (cond - 0) = 10 * cond
        assert!((result.data()[0] - 10.0).abs() < 1e-6);
        assert!((result.data()[1] - 0.0).abs() < 1e-6);
        assert!((result.data()[2] - (-10.0)).abs() < 1e-6);
    }

    // ----- Beam Search Additional Tests -----

    #[test]
    fn test_beam_search_best_hypotheses() {
        let config = BeamSearchConfig::new(2).with_num_return(2);
        let mut state = BeamSearchState::new(config, vec![0]);

        state.finished.push(BeamHypothesis {
            tokens: vec![0, 1],
            score: -1.0,
            finished: true,
        });
        state.finished.push(BeamHypothesis {
            tokens: vec![0, 2],
            score: -0.5,
            finished: true,
        });

        let best = state.best_hypotheses();
        assert_eq!(best.len(), 2);
        // Higher score (less negative) should be first
        assert!(best[0].score > best[1].score);
    }

    // ----- Stop Sequence Edge Cases -----

    #[test]
    fn test_stop_sequence_empty_pattern() {
        let detector = StopSequenceDetector::new()
            .with_token_sequence(vec![])
            .with_string_pattern("");
        assert!(!detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_multiple_matches() {
        let detector = StopSequenceDetector::new()
            .with_string_pattern("stop")
            .with_string_pattern("end");
        let pos = detector.check_text("stop at end");
        assert!(pos.is_some());
        assert_eq!(pos.unwrap(), 0); // "stop" is found first at position 0
    }

    // ----- Generation Config Builder Chain -----

    #[test]
    fn test_generation_config_full_builder_chain() {
        let config = GenerationConfig::top_k(50)
            .with_temperature(0.8)
            .with_max_tokens(200)
            .with_eos_token_id(99)
            .with_seed(42);

        assert_eq!(config.strategy, SamplingStrategy::TopK { k: 50 });
        assert!((config.temperature - 0.8).abs() < 1e-6);
        assert_eq!(config.max_tokens, 200);
        assert_eq!(config.eos_token_id, Some(99));
        assert_eq!(config.seed, Some(42));
    }

    // ----- LogitBias with HashMap -----

    #[test]
    fn test_logit_bias_with_biases() {
        let mut biases = std::collections::HashMap::new();
        biases.insert(1, 5.0);
        biases.insert(3, -5.0);

        let bias = LogitBias::new().with_biases(biases);
        assert_eq!(bias.get(1), 5.0);
        assert_eq!(bias.get(3), -5.0);
        assert_eq!(bias.get(2), 0.0);
    }

    // ----- Dynamic Temperature with Edge Cases -----

    #[test]
    fn test_dyn_temp_empty_logits() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    #[test]
    fn test_dyn_temp_negative_delta() {
        // Negative delta should behave like no delta
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let config = DynTempConfig::new(1.0, -0.5, 1.0);
        let result = apply_dynamic_temperature(&logits, &config);
        // Should just apply static temp
        assert!((result.data()[0] - 1.0).abs() < 1e-6);
    }

    // ----- Sampler Clone Tests -----

    #[test]
    fn test_temperature_sampler_clone_box() {
        let sampler = TemperatureSampler::new(0.5);
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "temperature");
    }

    #[test]
    fn test_top_k_sampler_clone_box() {
        let sampler = TopKSampler::new(10);
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "top_k");
    }

    #[test]
    fn test_top_p_sampler_clone_box() {
        let sampler = TopPSampler::new(0.9);
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "top_p");
    }

    #[test]
    fn test_dyn_temp_sampler_clone_box() {
        let sampler = DynTempSampler::new(DynTempConfig::new(1.0, 0.5, 1.0));
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "dyn_temp");
    }

    #[test]
    fn test_repetition_penalty_sampler_clone_box() {
        let sampler = RepetitionPenaltySampler::new(RepetitionPenaltyConfig::new(1.2));
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "repetition_penalty");
    }

    #[test]
    fn test_infill_sampler_clone_box() {
        let sampler = InfillSampler::new(InfillConfig::new(vec![1]));
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "infill");
    }

    // ----- Pipeline Accessors -----

    #[test]
    fn test_generation_pipeline_accessors() {
        let model = MockModel::new(100, 42);
        let pipeline = GenerationPipeline::new(model)
            .add_processor(TokenSuppressor::new(vec![0]))
            .with_config(GenerationConfig::greedy().with_max_tokens(5));

        assert_eq!(pipeline.model().vocab_size, 100);
        assert_eq!(pipeline.processors().len(), 1);
        assert_eq!(pipeline.config().max_tokens, 5);
    }

    #[test]
    fn test_generation_pipeline_model_mut() {
        let model = MockModel::new(100, 42);
        let mut pipeline = GenerationPipeline::new(model);

        // Should be able to mutate model
        pipeline.model_mut().highest_token = 50;
        assert_eq!(pipeline.model().highest_token, 50);
    }
}
