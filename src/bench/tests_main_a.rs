use crate::bench::*;
    use crate::bench::*;

    // ========================================================================
    // DynamicSampler Tests
    // ========================================================================

    #[test]
    fn test_dynamic_sampler_continues_until_min_samples() {
        let mut dyn_sampler = DynamicSampler::new(100, 10_000, 0.05);
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();

        assert!(dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_stops_at_max_samples() {
        let mut dyn_sampler = DynamicSampler::new(10, 100, 0.05);

        // Generate 100 data points with high variance
        let data: Vec<f64> = (0..100).map(|i| (i % 50) as f64 * 10.0).collect();

        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_stops_when_cv_stable() {
        let mut dyn_sampler = DynamicSampler::new(10, 10_000, 0.05);
        dyn_sampler.stability_count = 1; // Stop after 1 stable check

        // Generate 100 data points with very low variance (CV ~= 0)
        let data: Vec<f64> = vec![100.0; 100];

        // Should stop because CV = 0 < 0.05
        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_requires_stability_streak() {
        let mut dyn_sampler = DynamicSampler::new(10, 10_000, 0.05);
        dyn_sampler.stability_count = 3;

        // Stable data points
        let data: Vec<f64> = vec![100.0; 100];

        // First check - streak = 1
        assert!(dyn_sampler.should_continue(&data));
        // Second check - streak = 2
        assert!(dyn_sampler.should_continue(&data));
        // Third check - streak = 3, should stop
        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_reset() {
        let mut sampler = DynamicSampler::new(10, 10_000, 0.05);
        sampler.stable_streak = 5;
        sampler.reset();
        assert_eq!(sampler.stable_streak, 0);
    }

    #[test]
    fn test_compute_cv_constant_values() {
        let data = vec![100.0; 50];
        let cv = compute_cv(&data);
        assert!(cv.abs() < 1e-10, "CV of constant values should be ~0");
    }

    #[test]
    fn test_compute_cv_varied_values() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let cv = compute_cv(&data);
        // CV = std_dev / mean = 15.81 / 30 ~= 0.527
        assert!(cv > 0.5 && cv < 0.6, "CV should be ~0.527, got {cv}");
    }

    #[test]
    fn test_compute_cv_empty_data() {
        let data: Vec<f64> = vec![];
        let cv = compute_cv(&data);
        assert!(cv.is_infinite());
    }

    // ========================================================================
    // ThermalGuard Tests
    // ========================================================================

    #[test]
    fn test_thermal_guard_valid_low_variance() {
        let guard = ThermalGuard::default();
        let temps = vec![75.0, 75.5, 74.8, 75.2, 75.1];

        assert_eq!(guard.validate_run(&temps), ThermalValidity::Valid);
    }

    #[test]
    fn test_thermal_guard_invalid_high_variance() {
        let guard = ThermalGuard::default();
        // Variance std_dev > 2°C
        let temps = vec![70.0, 75.0, 80.0, 72.0, 78.0];

        match guard.validate_run(&temps) {
            ThermalValidity::Invalid(msg) => {
                assert!(msg.contains("exceeds threshold"));
            },
            ThermalValidity::Valid => panic!("Expected Invalid"),
        }
    }

    #[test]
    fn test_thermal_guard_empty_temps() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.validate_run(&[]), ThermalValidity::Valid);
    }

    #[test]
    fn test_thermal_guard_max_temp() {
        let guard = ThermalGuard::default();
        let temps = vec![70.0, 75.0, 85.0, 72.0];
        assert_eq!(guard.max_temp(&temps), 85.0);
    }

    // ========================================================================
    // KvCacheMetrics Tests
    // ========================================================================

    #[test]
    fn test_kv_cache_metrics_no_waste() {
        let metrics = KvCacheMetrics::new(1000, 1000);
        assert_eq!(metrics.fragmentation_pct, 0.0);
        assert!(metrics.is_acceptable(10.0));
    }

    #[test]
    fn test_kv_cache_metrics_with_waste() {
        let metrics = KvCacheMetrics::new(1000, 800);
        assert!((metrics.fragmentation_pct - 20.0).abs() < 0.01);
        assert!(!metrics.is_acceptable(10.0));
        assert!(metrics.is_acceptable(25.0));
    }

    #[test]
    fn test_kv_cache_metrics_zero_allocated() {
        let metrics = KvCacheMetrics::new(0, 0);
        assert_eq!(metrics.fragmentation_pct, 0.0);
    }

    #[test]
    fn test_kv_cache_metrics_mb_conversion() {
        let metrics = KvCacheMetrics::new(1024 * 1024 * 100, 1024 * 1024 * 80);
        assert!((metrics.allocated_mb() - 100.0).abs() < 0.01);
        assert!((metrics.used_mb() - 80.0).abs() < 0.01);
    }

    // ========================================================================
    // EnergyMetrics Tests
    // ========================================================================

    #[test]
    fn test_energy_metrics_joules_per_token() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 1000);
        assert!((metrics.joules_per_token() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_energy_metrics_zero_tokens() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 0);
        assert_eq!(metrics.joules_per_token(), 0.0);
    }

    #[test]
    fn test_energy_metrics_tokens_per_joule() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 1000);
        assert!((metrics.tokens_per_joule() - 10.0).abs() < 0.001);
    }

    // ========================================================================
    // ItlMetrics Tests
    // ========================================================================

    #[test]
    fn test_itl_metrics_from_measurements() {
        let itl = vec![10.0, 12.0, 11.0, 15.0, 13.0, 14.0, 11.0, 12.0, 13.0, 10.0];
        let metrics = ItlMetrics::from_measurements(&itl);

        // Median should be around 12
        assert!(metrics.median_ms > 11.0 && metrics.median_ms < 13.0);
        // Std dev should be small
        assert!(metrics.std_dev_ms < 5.0);
        // p99 should be around 15
        assert!(metrics.p99_ms >= 14.0);
    }

    #[test]
    fn test_itl_metrics_empty() {
        let metrics = ItlMetrics::from_measurements(&[]);
        assert_eq!(metrics.median_ms, 0.0);
        assert_eq!(metrics.std_dev_ms, 0.0);
    }

    #[test]
    fn test_itl_metrics_low_jitter() {
        let itl = vec![10.0; 100];
        let metrics = ItlMetrics::from_measurements(&itl);
        assert!(metrics.is_low_jitter(1.0));
    }

    #[test]
    fn test_itl_metrics_high_jitter() {
        let itl: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let metrics = ItlMetrics::from_measurements(&itl);
        assert!(!metrics.is_low_jitter(5.0));
    }

    // ========================================================================
    // KL-Divergence Tests
    // ========================================================================

    #[test]
    fn test_kl_divergence_identical_distributions() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = validate_quantization_quality(&logits, &logits, 0.01);

        match result {
            QualityResult::Pass { kl_divergence } => {
                assert!(kl_divergence < 1e-10, "KL should be ~0 for identical");
            },
            QualityResult::Fail { .. } => panic!("Expected Pass for identical"),
        }
    }

    #[test]
    fn test_kl_divergence_slightly_different() {
        let fp32 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quant = vec![1.01, 2.01, 3.01, 4.01, 5.01];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);

        match result {
            QualityResult::Pass { kl_divergence } => {
                assert!(kl_divergence < 0.001, "KL should be very small");
            },
            QualityResult::Fail { .. } => panic!("Expected Pass for small diff"),
        }
    }

    #[test]
    fn test_kl_divergence_very_different() {
        let fp32 = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        let quant = vec![0.0, 0.0, 0.0, 0.0, 10.0];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);

        match result {
            QualityResult::Fail { kl_divergence, .. } => {
                assert!(kl_divergence > 1.0, "KL should be large for opposite");
            },
            QualityResult::Pass { .. } => panic!("Expected Fail for very different"),
        }
    }

    #[test]
    fn test_kl_divergence_mismatched_lengths() {
        let fp32 = vec![1.0, 2.0, 3.0];
        let quant = vec![1.0, 2.0];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);
        assert!(matches!(result, QualityResult::Fail { .. }));
    }

    #[test]
    fn test_kl_divergence_empty() {
        let result = validate_quantization_quality(&[], &[], 0.01);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    // ========================================================================
    // BenchmarkResult Tests
    // ========================================================================

    #[test]
    fn test_benchmark_result_summary() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0, 22.0, 21.0, 25.0, 23.0, 24.0, 22.0, 21.0, 20.0, 26.0],
            itl_ms: vec![10.0, 11.0, 10.5, 11.5, 10.2, 10.8, 11.2, 10.3, 10.7, 11.0],
            generation_tok_s: vec![140.0, 142.0, 141.0, 143.0, 139.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let summary = result.summary();

        // Check percentiles are reasonable
        assert!(summary.ttft_p50 > 20.0 && summary.ttft_p50 < 25.0);
        assert!(summary.ttft_p99 >= summary.ttft_p50);
        assert!(summary.ttft_p999 >= summary.ttft_p99);

        // Check ITL
        assert!(summary.itl_median > 10.0 && summary.itl_median < 12.0);
        assert!(summary.itl_std_dev < 2.0);

        // Check throughput
        assert!(summary.throughput_median > 139.0 && summary.throughput_median < 144.0);

        // Check energy
        assert!((summary.token_joules - 0.05).abs() < 0.001);

        // Check metadata
        assert_eq!(summary.iterations, 500);
        assert!((summary.cv_final - 0.045).abs() < 0.001);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!(percentile(&data, 50.0) >= 5.0 && percentile(&data, 50.0) <= 6.0);
        assert!(percentile(&data, 90.0) >= 9.0);
        assert_eq!(percentile(&data, 100.0), 10.0);
    }

    #[test]
    fn test_bootstrap_ci() {
        let data = vec![100.0; 100];
        let (lower, upper) = bootstrap_ci(&data, 0.95, 1000);

        // For constant data, CI should be tight around 100
        assert!((lower - 100.0).abs() < 0.01);
        assert!((upper - 100.0).abs() < 0.01);
    }

    // ========================================================================
    // Softmax Tests
    // ========================================================================

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_monotonic() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);

        // Higher logits should have higher probabilities
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1]);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large logits shouldn't overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // ========================================================================
    // WorkloadType Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_workload_type_short_qa() {
        let workload = WorkloadType::ShortQa;
        assert_eq!(workload.input_tokens(), 32);
        assert_eq!(workload.output_tokens(), 64);
    }

    #[test]
    fn test_workload_type_long_context() {
        let workload = WorkloadType::LongContext;
        assert_eq!(workload.input_tokens(), 2048);
        assert_eq!(workload.output_tokens(), 512);
    }

    // ========================================================================
    // ConvoyTestConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_convoy_config_default() {
        let config = ConvoyTestConfig::default();
        assert_eq!(config.long_requests, 10);
        assert_eq!(config.short_requests, 100);
        assert!((config.max_p99_increase_pct - 50.0).abs() < 0.01);
        assert!((config.max_hol_blocking_ms - 500.0).abs() < 0.01);
        assert!((config.max_kv_fragmentation_pct - 15.0).abs() < 0.01);
    }

    // ========================================================================
    // ConvoyTestResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_convoy_test_result_pass() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0, 12.0, 11.0, 13.0, 10.5]; // p99 ~= 13
        let convoy = vec![12.0, 14.0, 13.0, 15.0, 12.5]; // p99 ~= 15 (15% increase)
        let hol = vec![50.0, 100.0, 75.0, 80.0, 60.0];
        let kv_frag = 10.0; // 10% < 15% threshold

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(result.passed, "Should pass with acceptable metrics");
        assert!(result.failure_reasons.is_empty());
        assert!(result.p99_increase_pct < 50.0);
        assert!(result.max_hol_blocking_ms < 500.0);
    }

    #[test]
    fn test_convoy_test_result_fail_p99() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![20.0; 100]; // 100% increase > 50% threshold
        let hol = vec![50.0; 100];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with 100% p99 increase");
        assert!(result.failure_reasons.iter().any(|r| r.contains("P99")));
    }

    #[test]
    fn test_convoy_test_result_fail_hol_blocking() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![11.0; 100]; // 10% increase - acceptable
        let hol = vec![600.0; 100]; // 600ms > 500ms threshold
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with HOL blocking > 500ms");
        assert!(result.failure_reasons.iter().any(|r| r.contains("HOL")));
    }

    #[test]
    fn test_convoy_test_result_fail_kv_fragmentation() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![11.0; 100];
        let hol = vec![50.0; 100];
        let kv_frag = 20.0; // 20% > 15% threshold

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with KV fragmentation > 15%");
        assert!(result.failure_reasons.iter().any(|r| r.contains("KV")));
    }

    // ========================================================================
    // SaturationTestConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_saturation_config_default() {
        let config = SaturationTestConfig::default();
        assert_eq!(config.cpu_load_pct, 50);
        assert!((config.max_throughput_degradation_pct - 30.0).abs() < 0.01);
        assert!((config.max_p99_increase_pct - 100.0).abs() < 0.01);
    }

    // ========================================================================
    // SaturationTestResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_saturation_test_result_pass() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let stressed_throughput = vec![85.0, 87.0, 83.0, 86.0, 84.0]; // ~15% degradation
        let baseline_latency = vec![10.0, 12.0, 11.0, 10.5, 11.5];
        let stressed_latency = vec![15.0, 17.0, 16.0, 15.5, 16.5]; // ~50% increase

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(result.passed, "Should pass with acceptable degradation");
        assert!(result.throughput_degradation_pct < 30.0);
        assert!(result.p99_increase_pct < 100.0);
    }

    #[test]
    fn test_saturation_test_result_fail_throughput() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0; 100];
        let stressed_throughput = vec![50.0; 100]; // 50% degradation > 30%
        let baseline_latency = vec![10.0; 100];
        let stressed_latency = vec![15.0; 100]; // 50% increase - acceptable

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(
            !result.passed,
            "Should fail with 50% throughput degradation"
        );
        assert!(result
            .failure_reasons
            .iter()
            .any(|r| r.contains("Throughput")));
    }

    #[test]
    fn test_saturation_test_result_fail_p99() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0; 100];
        let stressed_throughput = vec![90.0; 100]; // 10% degradation - acceptable
        let baseline_latency = vec![10.0; 100];
        let stressed_latency = vec![25.0; 100]; // 150% increase > 100%

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(!result.passed, "Should fail with 150% p99 increase");
        assert!(result.failure_reasons.iter().any(|r| r.contains("P99")));
    }

    // ========================================================================
    // HardwareSpec Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_hardware_spec_default() {
        let spec = HardwareSpec::default();
        assert_eq!(spec.cpu, "Unknown");
        assert!(spec.gpu.is_none());
        assert_eq!(spec.memory_gb, 0);
        assert_eq!(spec.storage, "Unknown");
    }

    // ========================================================================
    // SamplingConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.method, "dynamic_cv");
        assert!((config.cv_threshold - 0.05).abs() < 0.001);
        assert_eq!(config.warmup_iterations, 100);
    }

    // ========================================================================
    // ThermalInfo Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_thermal_info_default() {
        let info = ThermalInfo::default();
        assert!(info.valid);
        assert!((info.temp_variance_c - 0.0).abs() < 0.001);
        assert!((info.max_temp_c - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // FullBenchmarkResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_full_benchmark_result_from_benchmark_result() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0, 22.0, 21.0, 25.0, 23.0],
            itl_ms: vec![10.0, 11.0, 10.5, 11.5, 10.2],
            generation_tok_s: vec![140.0, 142.0, 141.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let hardware = HardwareSpec {
            cpu: "Apple M3 Max".to_string(),
            gpu: Some("Apple M3 Max (40 cores)".to_string()),
            memory_gb: 128,
            storage: "NVMe".to_string(),
        };

        let temps = vec![72.0, 73.0, 72.5, 73.5, 72.0];
        let kl_div = 0.031;

        let full_result =
            FullBenchmarkResult::from_benchmark_result(&result, hardware, &temps, kl_div);

        assert_eq!(full_result.version, "1.1");
        assert!(full_result.timestamp.contains("1970")); // Simple timestamp format
        assert_eq!(full_result.config.model, "test");
        assert_eq!(full_result.hardware.cpu, "Apple M3 Max");
        assert_eq!(full_result.sampling.actual_iterations, 500);
        assert!(full_result.thermal.valid);
        assert!((full_result.quality.kl_divergence_vs_fp32 - 0.031).abs() < 0.001);
    }

    #[test]
    fn test_full_benchmark_result_json_roundtrip() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0, 22.0, 21.0],
            itl_ms: vec![10.0, 11.0, 10.5],
            generation_tok_s: vec![140.0, 142.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let full_result =
            FullBenchmarkResult::from_benchmark_result(&result, HardwareSpec::default(), &[], 0.0);

        let json = full_result.to_json().expect("Should serialize");
        let parsed: FullBenchmarkResult =
            FullBenchmarkResult::from_json(&json).expect("Should parse");

        assert_eq!(parsed.version, "1.1");
        assert_eq!(parsed.config.model, "test");
        assert_eq!(parsed.sampling.actual_iterations, 500);
    }

    // ========================================================================
    // BenchmarkComparison Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_benchmark_comparison_realizar_wins() {
        let baseline = create_test_full_result("llama.cpp", 40.0, 100.0, 1500, 0.06);
        let current = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);

        assert_eq!(comparison.winner, "realizar");
        assert!(comparison.ttft_p99_change_pct < 0.0); // Improvement
        assert!(comparison.throughput_change_pct > 0.0); // Improvement
        assert!(comparison.memory_change_pct < 0.0); // Improvement
        assert!(comparison.energy_change_pct < 0.0); // Improvement
    }

    #[test]
    fn test_benchmark_comparison_tie() {
        let baseline = create_test_full_result("runtime_a", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("runtime_b", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);

        assert_eq!(comparison.winner, "tie");
    }

    // ========================================================================
    // RegressionResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_regression_result_no_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 29.0, 145.0, 1150, 0.038);

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(!regression.regression_detected);
        assert!(regression.regressed_metrics.is_empty());
    }

    #[test]
    fn test_regression_result_ttft_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 35.0, 140.0, 1200, 0.04); // 16.7% worse TTFT

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("ttft")));
    }

    #[test]
    fn test_regression_result_throughput_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 30.0, 120.0, 1200, 0.04); // 14.3% worse throughput

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("throughput")));
    }

    #[test]
    fn test_regression_result_memory_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 30.0, 140.0, 1400, 0.04); // 16.7% worse memory

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("memory")));
    }

    /// Helper function to create test FullBenchmarkResult
    fn create_test_full_result(
        runtime: &str,
        ttft_p99: f64,
        throughput: f64,
        memory_mb: u64,
        token_joules: f64,
    ) -> FullBenchmarkResult {
        FullBenchmarkResult {
            version: "1.1".to_string(),
            timestamp: "2025-12-09T12:00:00Z".to_string(),
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: runtime.to_string(),
                runtime_version: "1.0.0".to_string(),
            },
            hardware: HardwareSpec::default(),
            sampling: SamplingConfig::default(),
            thermal: ThermalInfo::default(),
            results: BenchmarkResults {
                ttft_ms: TtftResults {
                    p50: ttft_p99 * 0.7,
                    p95: ttft_p99 * 0.9,
                    p99: ttft_p99,
                    p999: ttft_p99 * 1.2,
                },
                itl_ms: ItlResults {
                    median: 10.0,
                    std_dev: 2.0,
                    p99: 15.0,
                },
                throughput_tok_s: ThroughputResults {
                    median: throughput,
                    ci_95: (throughput * 0.95, throughput * 1.05),
                },
                memory_mb: MemoryResults {
                    model_mb: memory_mb / 2,
                    peak_rss_mb: memory_mb,
                    kv_waste_pct: 3.0,
                },
                energy: EnergyResults {
                    total_joules: 50.0,
                    token_joules,
                    idle_watts: 8.0,
                },
                cold_start_ms: ColdStartResults {
                    median: 100.0,
                    p99: 150.0,
                },
            },
            quality: QualityValidation {
                kl_divergence_vs_fp32: 0.03,
                perplexity_wikitext2: Some(5.89),
            },
        }
    }

    // ========================================================================
    // Additional Coverage Tests (Phase 3 - 95% Target)
    // ========================================================================

    #[test]
    fn test_dynamic_sampler_current_cv_empty() {
        let sampler = DynamicSampler::default();
        let cv = sampler.current_cv(&[]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_dynamic_sampler_current_cv_single_value() {
        let sampler = DynamicSampler::default();
        let cv = sampler.current_cv(&[100.0]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_dynamic_sampler_current_cv_constant_values() {
        let sampler = DynamicSampler::default();
        let data: Vec<f64> = vec![50.0; 100];
        let cv = sampler.current_cv(&data);
        assert!(cv.abs() < 1e-10, "CV of constant should be ~0");
    }

    #[test]
    fn test_dynamic_sampler_current_cv_varied_window() {
        let sampler = DynamicSampler {
            cv_window: 10,
            ..Default::default()
        };
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 % 10.0)).collect();
        let cv = sampler.current_cv(&data);
        assert!(cv > 0.0 && cv < 1.0);
    }

    #[test]
    fn test_dynamic_sampler_current_cv_small_window() {
        let sampler = DynamicSampler {
            cv_window: 5,
            ..Default::default()
        };
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let cv = sampler.current_cv(&data);
        assert!(cv > 0.4 && cv < 0.6);
    }

    #[test]
    fn test_dynamic_sampler_default_values() {
        let sampler = DynamicSampler::default();
        assert_eq!(sampler.min_samples, 100);
        assert_eq!(sampler.max_samples, 10_000);
        assert!((sampler.cv_threshold - 0.05).abs() < 0.001);
        assert_eq!(sampler.cv_window, 50);
        assert_eq!(sampler.stability_count, 3);
        // stable_streak is private, tested via should_continue
    }

    #[test]
    fn test_thermal_guard_temp_variance_empty() {
        let guard = ThermalGuard::default();
        let variance = guard.temp_variance(&[]);
        assert!((variance - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_thermal_guard_temp_variance_single() {
        let guard = ThermalGuard::default();
        let variance = guard.temp_variance(&[75.0]);
        assert!((variance - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_thermal_guard_temp_variance_constant() {
        let guard = ThermalGuard::default();
        let temps = vec![72.0; 100];
        let variance = guard.temp_variance(&temps);
        assert!(variance < 0.001);
    }

    #[test]
    fn test_thermal_guard_temp_variance_varied() {
        let guard = ThermalGuard::default();
        let temps = vec![70.0, 72.0, 74.0, 76.0, 78.0];
        let variance = guard.temp_variance(&temps);
        assert!(variance > 2.0 && variance < 4.0);
    }

    #[test]
    fn test_thermal_guard_max_temp_empty() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp(&[]), 0.0);
    }

    #[test]
    fn test_thermal_guard_max_temp_single() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp(&[82.5]), 82.5);
    }

    #[test]
    fn test_thermal_guard_cooldown_not_needed() {
        let guard = ThermalGuard::default();
        // Should not sleep when temp is below max
        guard.cooldown_if_needed(70.0);
        // Test passes if no timeout
    }

    #[test]
    fn test_chrono_timestamp_format() {
        let ts = chrono_timestamp();
        assert!(ts.contains("1970"));
        assert!(ts.contains("T"));
        assert!(ts.contains("Z"));
        assert!(ts.contains("+"));
        assert!(ts.contains("s"));
    }

    #[test]
    fn test_bootstrap_ci_empty() {
        let (lower, upper) = bootstrap_ci(&[], 0.95, 1000);
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 0.0);
    }

    #[test]
    fn test_bootstrap_ci_single_value() {
        let (lower, upper) = bootstrap_ci(&[42.0], 0.95, 1000);
        assert!((lower - 42.0).abs() < 0.01);
        assert!((upper - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_bootstrap_ci_varied_data() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let (lower, upper) = bootstrap_ci(&data, 0.95, 1000);
        // Mean is 50.5, CI should contain mean
        assert!(lower < 55.0);
        assert!(upper > 45.0);
        assert!(lower < upper);
    }

    #[test]
    fn test_bootstrap_ci_narrow_confidence() {
        let data = vec![100.0; 50];
        let (lower, upper) = bootstrap_ci(&data, 0.50, 100);
        // Even narrow CI should be close to 100 for constant data
        assert!((lower - 100.0).abs() < 0.1);
        assert!((upper - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
        assert_eq!(percentile(&[42.0], 99.0), 42.0);
    }

    #[test]
    fn test_compute_std_dev_constant() {
        let data = vec![100.0; 50];
        let std_dev = compute_std_dev(&data);
        assert!(std_dev < 0.001);
    }

    #[test]
    fn test_compute_std_dev_empty() {
        let std_dev = compute_std_dev(&[]);
        assert_eq!(std_dev, 0.0);
    }

    #[test]
    fn test_compute_variance_empty() {
        assert_eq!(compute_variance(&[]), 0.0);
    }

    #[test]
    fn test_compute_variance_single() {
        assert_eq!(compute_variance(&[100.0]), 0.0);
    }

    #[test]
    fn test_compute_cv_single_value() {
        let cv = compute_cv(&[100.0]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_compute_cv_zero_mean() {
        // Mix of positive and negative that averages to near zero
        let data = vec![-1.0, 1.0, -1.0, 1.0];
        let cv = compute_cv(&data);
        // Mean is 0, CV should be infinite
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_energy_metrics_tokens_per_joule_zero_joules() {
        let metrics = EnergyMetrics::new(0.0, 10.0, 50.0, 1000);
        assert_eq!(metrics.tokens_per_joule(), 0.0);
    }

    #[test]
    fn test_energy_metrics_very_small_joules() {
        let metrics = EnergyMetrics::new(1e-15, 10.0, 50.0, 1000);
        assert_eq!(metrics.tokens_per_joule(), 0.0);
    }

    #[test]
    fn test_itl_metrics_single_value() {
        let metrics = ItlMetrics::from_measurements(&[15.0]);
        assert_eq!(metrics.median_ms, 15.0);
        assert_eq!(metrics.p99_ms, 15.0);
        assert_eq!(metrics.p999_ms, 15.0);
        assert_eq!(metrics.std_dev_ms, 0.0);
    }

    #[test]
    fn test_itl_metrics_two_values() {
        let metrics = ItlMetrics::from_measurements(&[10.0, 20.0]);
        assert_eq!(metrics.median_ms, 15.0);
        assert!(metrics.std_dev_ms > 0.0);
    }

    #[test]
    fn test_convoy_test_result_empty_hol() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 10];
        let convoy = vec![11.0; 10];
        let hol: Vec<f64> = vec![];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);
        assert_eq!(result.avg_hol_blocking_ms, 0.0);
        assert_eq!(result.max_hol_blocking_ms, 0.0);
    }

    #[test]
    fn test_convoy_test_result_zero_baseline() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![0.0; 10];
        let convoy = vec![10.0; 10];
        let hol = vec![50.0; 10];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);
        assert_eq!(result.p99_increase_pct, 0.0);
    }

    #[test]
    fn test_saturation_test_result_empty_data() {
        let config = SaturationTestConfig::default();
        let result = SaturationTestResult::new(&config, &[], &[], &[], &[]);

        assert_eq!(result.baseline_throughput, 0.0);
        assert_eq!(result.stressed_throughput, 0.0);
        assert_eq!(result.throughput_degradation_pct, 0.0);
    }

    #[test]
    fn test_saturation_test_result_zero_baseline() {
        let config = SaturationTestConfig::default();
        let result =
            SaturationTestResult::new(&config, &[0.0; 10], &[50.0; 10], &[0.0; 10], &[10.0; 10]);

        assert_eq!(result.throughput_degradation_pct, 0.0);
        assert_eq!(result.p99_increase_pct, 0.0);
    }

    #[test]
    fn test_benchmark_comparison_zero_baselines() {
        let baseline = create_test_full_result("baseline", 0.0, 0.0, 0, 0.0);
        let current = create_test_full_result("current", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);
        assert_eq!(comparison.ttft_p99_change_pct, 0.0);
        assert_eq!(comparison.throughput_change_pct, 0.0);
        assert_eq!(comparison.memory_change_pct, 0.0);
        assert_eq!(comparison.energy_change_pct, 0.0);
    }

    #[test]
    fn test_regression_result_zero_baselines() {
        let baseline = create_test_full_result("test", 0.0, 0.0, 0, 0.0);
        let current = create_test_full_result("test", 30.0, 140.0, 1200, 0.04);

        let regression = RegressionResult::check(&baseline, &current, 5.0);
        // No regression detected because baseline is zero
        assert!(!regression.regression_detected);
    }

    #[test]
    fn test_benchmark_result_zero_tokens() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0],
            itl_ms: vec![10.0],
            generation_tok_s: vec![140.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 0,
            actual_iterations: 100,
            cv_at_stop: 0.04,
            timestamp: 12345,
        };

        let summary = result.summary();
        assert_eq!(summary.token_joules, 0.0);
    }

    #[test]
    fn test_kv_cache_used_more_than_allocated() {
        // Edge case: used > allocated (shouldn't happen but test boundary)
        let metrics = KvCacheMetrics::new(1000, 1500);
        // saturating_sub gives 0 waste
        assert_eq!(metrics.fragmentation_pct, 0.0);
    }

    #[test]
    fn test_softmax_single_value() {
        let probs = softmax(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_negative_values() {
        let logits = vec![-5.0, -3.0, -1.0, 0.0, 1.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Should still be monotonic
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1]);
        }
    }

    #[test]
    fn test_full_benchmark_result_invalid_thermal() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0],
            itl_ms: vec![10.0],
            generation_tok_s: vec![140.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 100,
            cv_at_stop: 0.04,
            timestamp: 12345,
        };

        // High variance temps that should be invalid
        let temps = vec![60.0, 70.0, 80.0, 65.0, 85.0];
        let full_result = FullBenchmarkResult::from_benchmark_result(
            &result,
            HardwareSpec::default(),
            &temps,
            0.03,
        );

        assert!(!full_result.thermal.valid);
        assert!(full_result.thermal.temp_variance_c > 2.0);
    }

    #[test]
    fn test_benchmark_comparison_baseline_wins() {
        let baseline = create_test_full_result("baseline", 25.0, 160.0, 1000, 0.03);
        let current = create_test_full_result("current", 40.0, 100.0, 1500, 0.06);

        let comparison = BenchmarkComparison::compare(&baseline, &current);
        assert_eq!(comparison.winner, "baseline");
    }

    #[test]
    fn test_thermal_validity_debug() {
        let valid = ThermalValidity::Valid;
        let invalid = ThermalValidity::Invalid("test".to_string());
        // Test Debug derive
        assert!(format!("{valid:?}").contains("Valid"));
        assert!(format!("{invalid:?}").contains("Invalid"));
    }

    #[test]
    fn test_quality_result_debug() {
        let pass = QualityResult::Pass {
            kl_divergence: 0.01,
        };
        let fail = QualityResult::Fail {
            kl_divergence: 0.5,
            threshold: 0.1,
            message: "test",
        };
        // Test Debug derive
        assert!(format!("{pass:?}").contains("Pass"));
        assert!(format!("{fail:?}").contains("Fail"));
    }

    #[test]
    fn test_workload_type_equality() {
        assert_eq!(WorkloadType::ShortQa, WorkloadType::ShortQa);
        assert_eq!(WorkloadType::LongContext, WorkloadType::LongContext);
        assert_ne!(WorkloadType::ShortQa, WorkloadType::LongContext);
    }

    #[test]
    fn test_workload_type_copy() {
        let wt = WorkloadType::ShortQa;
        let wt_copy = wt;
        assert_eq!(wt, wt_copy);
    }

    // ========================================================================
    // BENCH-002: RuntimeBackend trait tests (TDD RED -> GREEN)
    // ========================================================================

    #[test]
    fn test_runtime_type_display() {
        assert_eq!(RuntimeType::Realizar.as_str(), "realizar");
        assert_eq!(RuntimeType::LlamaCpp.as_str(), "llama-cpp");
        assert_eq!(RuntimeType::Vllm.as_str(), "vllm");
        assert_eq!(RuntimeType::Ollama.as_str(), "ollama");
    }

    #[test]
    fn test_runtime_type_from_str() {
        assert_eq!(RuntimeType::parse("realizar"), Some(RuntimeType::Realizar));
        assert_eq!(RuntimeType::parse("llama-cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("llama.cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("vllm"), Some(RuntimeType::Vllm));
        assert_eq!(RuntimeType::parse("ollama"), Some(RuntimeType::Ollama));
        assert_eq!(RuntimeType::parse("unknown"), None);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_inference_request_default() {
        let req = InferenceRequest::default();
        assert_eq!(req.prompt, "");
        assert_eq!(req.max_tokens, 100);
        assert!(req.temperature > 0.0);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_inference_request_builder() {
        let req = InferenceRequest::new("Hello, world!")
            .with_max_tokens(50)
            .with_temperature(0.5);
        assert_eq!(req.prompt, "Hello, world!");
        assert_eq!(req.max_tokens, 50);
        assert!((req.temperature - 0.5).abs() < 0.001);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_inference_response_tokens_per_second() {
        let response = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 50.0,
            total_time_ms: 1000.0,
            itl_ms: vec![10.0, 10.0, 10.0],
        };
        assert!((response.tokens_per_second() - 100.0).abs() < 0.1);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_inference_response_tokens_per_second_zero_time() {
        let response = InferenceResponse {
            text: String::new(),
            tokens_generated: 100,
            ttft_ms: 0.0,
            total_time_ms: 0.0,
            itl_ms: vec![],
        };
        assert_eq!(response.tokens_per_second(), 0.0);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_mock_backend_inference() {
        let backend = MockBackend::new(42.0, 150.0);
        let req = InferenceRequest::new("test prompt");
        let response = backend.inference(&req);

        assert!(response.is_ok());
        let resp = response.expect("test");
        assert!((resp.ttft_ms - 42.0).abs() < 0.001);
        assert!(resp.tokens_generated > 0);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_mock_backend_info() {
        let backend = MockBackend::new(30.0, 140.0);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Realizar);
        assert!(!info.version.is_empty());
        assert!(info.supports_streaming);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_backend_registry_default() {
        let registry = BackendRegistry::new();
        assert!(registry.get(RuntimeType::Realizar).is_none());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_backend_registry_register_and_get() {
        let mut registry = BackendRegistry::new();
        let backend = Box::new(MockBackend::new(30.0, 140.0));
        registry.register(RuntimeType::Realizar, backend);

        assert!(registry.get(RuntimeType::Realizar).is_some());
        assert!(registry.get(RuntimeType::LlamaCpp).is_none());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_backend_registry_list() {
        let mut registry = BackendRegistry::new();
        registry.register(
            RuntimeType::Realizar,
            Box::new(MockBackend::new(30.0, 140.0)),
        );
        registry.register(
            RuntimeType::LlamaCpp,
            Box::new(MockBackend::new(35.0, 130.0)),
        );

        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&RuntimeType::Realizar));
        assert!(list.contains(&RuntimeType::LlamaCpp));
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_config_default() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.binary_path, "llama-cli");
        assert_eq!(config.n_gpu_layers, 0);
        assert_eq!(config.ctx_size, 2048);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_config_builder() {
        let config = LlamaCppConfig::new("/usr/bin/llama-cli")
            .with_model("/models/test.gguf")
            .with_gpu_layers(32)
            .with_ctx_size(4096);

        assert_eq!(config.binary_path, "/usr/bin/llama-cli");
        assert_eq!(config.model_path, Some("/models/test.gguf".to_string()));
        assert_eq!(config.n_gpu_layers, 32);
        assert_eq!(config.ctx_size, 4096);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_config_default() {
        let config = VllmConfig::default();
        assert_eq!(config.base_url, "http://localhost:8000");
        assert_eq!(config.api_version, "v1");
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_config_builder() {
        let config = VllmConfig::new("http://gpu-server:8080")
            .with_model("meta-llama/Llama-2-7b")
            .with_api_key("test-key");

        assert_eq!(config.base_url, "http://gpu-server:8080");
        assert_eq!(config.model, Some("meta-llama/Llama-2-7b".to_string()));
        assert_eq!(config.api_key, Some("test-key".to_string()));
    }

    // =========================================================================
    // LlamaCppBackend Tests (BENCH-002: Runtime Backend Integration)
    // =========================================================================

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_creation() {
        let config = LlamaCppConfig::new("llama-cli");
        let backend = LlamaCppBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
        assert!(!info.version.is_empty());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_info() {
        let config = LlamaCppConfig::new("llama-cli").with_model("test.gguf");
        let backend = LlamaCppBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
        assert!(!info.supports_streaming); // CLI doesn't support streaming
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_missing_binary() {
        let config = LlamaCppConfig::new("/nonexistent/llama-cli");
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        // Should return error for missing binary
        assert!(result.is_err());
    }

    // =========================================================================
    // VllmBackend Tests (BENCH-003: HTTP Client Integration)
    // Requires bench-http feature for HTTP client
    // =========================================================================

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_creation() {
        let config = VllmConfig::new("http://localhost:8000");
        let backend = VllmBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Vllm);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_info() {
        let config = VllmConfig::new("http://localhost:8000").with_model("meta-llama/Llama-2-7b");
        let backend = VllmBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Vllm);
        assert!(info.supports_streaming); // vLLM supports streaming
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_connection_error() {
        let config = VllmConfig::new("http://localhost:99999"); // Invalid port
        let backend = VllmBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        // Should return error for connection failure
        assert!(result.is_err());
    }
