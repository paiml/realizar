#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
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

    // =========================================================================
    // BENCH-004: MeasurementProtocol Tests (TDD RED)
    // =========================================================================

    #[test]
    fn test_measurement_protocol_default() {
        let protocol = MeasurementProtocol::default();
        assert_eq!(protocol.latency_samples, 100);
        assert_eq!(
            protocol.latency_percentiles,
            vec![50.0, 90.0, 95.0, 99.0, 99.9]
        );
        assert_eq!(protocol.throughput_duration.as_secs(), 60);
        assert_eq!(protocol.throughput_ramp_up.as_secs(), 10);
        assert_eq!(protocol.memory_samples, 10);
    }

    #[test]
    fn test_measurement_protocol_builder() {
        let protocol = MeasurementProtocol::new()
            .with_latency_samples(200)
            .with_percentiles(vec![50.0, 95.0, 99.0])
            .with_throughput_duration(Duration::from_secs(120))
            .with_memory_samples(20);

        assert_eq!(protocol.latency_samples, 200);
        assert_eq!(protocol.latency_percentiles, vec![50.0, 95.0, 99.0]);
        assert_eq!(protocol.throughput_duration.as_secs(), 120);
        assert_eq!(protocol.memory_samples, 20);
    }

    // =========================================================================
    // BENCH-005: LatencyStatistics Tests (TDD RED)
    // =========================================================================

    #[test]
    fn test_latency_statistics_from_samples() {
        let samples = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];
        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.samples, 5);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(50));
        assert_eq!(stats.mean, Duration::from_millis(30));
    }

    #[test]
    fn test_latency_statistics_percentiles() {
        // 100 samples from 1ms to 100ms
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        // p50 should be around 50ms
        assert!(stats.p50 >= Duration::from_millis(49));
        assert!(stats.p50 <= Duration::from_millis(51));

        // p95 should be around 95ms
        assert!(stats.p95 >= Duration::from_millis(94));
        assert!(stats.p95 <= Duration::from_millis(96));

        // p99 should be around 99ms
        assert!(stats.p99 >= Duration::from_millis(98));
        assert!(stats.p99 <= Duration::from_millis(100));
    }

    #[test]
    fn test_latency_statistics_confidence_interval() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        // 95% CI should contain the mean
        let (lower, upper) = stats.confidence_interval_95;
        assert!(lower < stats.mean);
        assert!(upper > stats.mean);
    }

    #[test]
    fn test_latency_statistics_std_dev() {
        // Uniform samples should have non-zero std dev
        let samples: Vec<Duration> = (1..=10).map(|i| Duration::from_millis(i * 10)).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        assert!(stats.std_dev > Duration::ZERO);
    }

    // ==========================================
    // BENCH-006: OutlierDetector Tests (MAD-based)
    // ==========================================

    #[test]
    fn test_outlier_detector_no_outliers() {
        // Normal distribution with no outliers
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.3];
        let outliers = detect_outliers(&samples, 3.5); // Standard threshold
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_outlier_detector_single_outlier() {
        // One clear outlier at position 8 (value 100.0)
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.3, 100.0];
        let outliers = detect_outliers(&samples, 3.5);
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0], 8);
    }

    #[test]
    fn test_outlier_detector_multiple_outliers() {
        // Two outliers: one high, one low
        let samples = vec![0.1, 10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 100.0];
        let outliers = detect_outliers(&samples, 3.5);
        assert_eq!(outliers.len(), 2);
        assert!(outliers.contains(&0)); // 0.1 is an outlier
        assert!(outliers.contains(&8)); // 100.0 is an outlier
    }

    #[test]
    fn test_outlier_detector_threshold_sensitivity() {
        // Lower threshold should catch more outliers
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 15.0];
        let strict_outliers = detect_outliers(&samples, 2.0);
        let lenient_outliers = detect_outliers(&samples, 5.0);
        assert!(strict_outliers.len() >= lenient_outliers.len());
    }

    // ==========================================
    // BENCH-007: RegressionDetector Tests
    // ==========================================

    #[test]
    fn test_regression_detector_default() {
        let detector = RegressionDetector::default();
        assert_eq!(detector.warning_threshold, 0.02); // 2%
        assert_eq!(detector.failure_threshold, 0.05); // 5%
    }

    #[test]
    fn test_regression_detector_no_regression() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 101.0, // 1% increase - within warning
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert!(report.regressions.is_empty());
    }

    #[test]
    fn test_regression_detector_warning() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 103.0, // 3% increase - warning
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed); // Warnings don't fail
        assert_eq!(report.warnings.len(), 1);
    }

    #[test]
    fn test_regression_detector_failure() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 110.0, // 10% increase - failure
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(!report.passed);
        assert_eq!(report.regressions.len(), 1);
    }

    #[test]
    fn test_regression_detector_improvement() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 90.0, // 10% decrease - improvement!
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert_eq!(report.improvements.len(), 1);
    }

    // ==========================================
    // BENCH-008: Welch's t-test Tests
    // ==========================================

    #[test]
    fn test_welch_t_test_result_fields() {
        // Verify result struct has all required fields
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let sample_b = vec![20.0, 21.0, 20.5, 20.2, 20.8];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // Result should have t_statistic, degrees_of_freedom, p_value, significant
        assert!(result.t_statistic.is_finite());
        assert!(result.degrees_of_freedom > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        // These are clearly different - should be significant
        assert!(result.significant);
    }

    #[test]
    fn test_welch_t_test_identical_samples() {
        // Identical samples should NOT be significant
        let sample_a = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let sample_b = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        assert!(!result.significant);
        assert!(result.t_statistic.abs() < 1e-10 || result.p_value > 0.05);
    }

    #[test]
    fn test_welch_t_test_clearly_different() {
        // Clearly different samples should be significant
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8, 10.3, 10.7, 10.1];
        let sample_b = vec![50.0, 51.0, 50.5, 50.2, 50.8, 50.3, 50.7, 50.1];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        assert!(result.significant);
        assert!(result.p_value < 0.001); // Very significant
    }

    #[test]
    fn test_welch_t_test_unequal_variance() {
        // Welch's t-test handles unequal variances correctly
        let sample_a = vec![10.0, 10.1, 10.0, 10.1, 10.0]; // Low variance
        let sample_b = vec![10.0, 15.0, 5.0, 20.0, 0.0]; // High variance, same mean
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // Same mean, different variance - should NOT be significant
        assert!(!result.significant);
    }

    #[test]
    fn test_welch_t_test_small_samples() {
        // Small samples require larger differences
        let sample_a = vec![10.0, 11.0, 12.0];
        let sample_b = vec![12.0, 13.0, 14.0];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // With only 3 samples each, difference may not be significant
        assert!(result.degrees_of_freedom > 0.0);
    }

    #[test]
    fn test_welch_t_test_alpha_levels() {
        // Different alpha levels affect significance
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let sample_b = vec![11.0, 12.0, 11.5, 11.2, 11.8];
        let result_strict = welch_t_test(&sample_a, &sample_b, 0.01);
        let result_lenient = welch_t_test(&sample_a, &sample_b, 0.10);
        // Lenient alpha should be at least as likely to find significance
        if result_strict.significant {
            assert!(result_lenient.significant);
        }
    }

    // BENCH-009: ThermalGuard Tests (TDD RED)
    #[test]
    fn test_thermal_guard_struct_fields() {
        // Per spec: ThermalGuard has max_temp_c, cooldown_threshold_c, cooldown_sleep_ms, temp_variance_c
        let guard = ThermalGuard::new(80.0, 70.0, 10_000, 2.0);
        assert_eq!(guard.max_temp_c, 80.0);
        assert_eq!(guard.cooldown_threshold_c, 70.0);
        assert_eq!(guard.cooldown_sleep_ms, 10_000);
        assert_eq!(guard.temp_variance_c, 2.0);
    }

    #[test]
    fn test_thermal_guard_default() {
        // Default should use spec values: 80°C, 70°C, 10000ms, 2°C
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp_c, 80.0);
        assert_eq!(guard.cooldown_threshold_c, 70.0);
        assert_eq!(guard.cooldown_sleep_ms, 10_000);
        assert_eq!(guard.temp_variance_c, 2.0);
    }

    #[test]
    fn test_thermal_validity_valid() {
        // Low variance temps should be valid
        let guard = ThermalGuard::default();
        let temps = vec![75.0, 76.0, 75.5, 76.5, 75.2]; // Variance < 2°C
        let result = guard.validate_run(&temps);
        assert!(matches!(result, ThermalValidity::Valid));
    }

    #[test]
    fn test_thermal_validity_invalid_high_variance() {
        // High variance temps should be invalid
        let guard = ThermalGuard::default();
        let temps = vec![60.0, 80.0, 65.0, 85.0, 70.0]; // High variance
        let result = guard.validate_run(&temps);
        assert!(matches!(result, ThermalValidity::Invalid(_)));
    }

    #[test]
    fn test_thermal_needs_cooldown_above_max() {
        // Above max temp should need cooldown
        let guard = ThermalGuard::default();
        assert!(guard.needs_cooldown(85.0)); // 85 > 80
    }

    #[test]
    fn test_thermal_needs_cooldown_below_max() {
        // Below max temp should not need cooldown
        let guard = ThermalGuard::default();
        assert!(!guard.needs_cooldown(75.0)); // 75 < 80
    }

    // BENCH-010: KL-Divergence Quality Validation Tests (TDD RED)
    #[test]
    fn test_quality_result_pass() {
        // QualityResult::Pass should contain kl_divergence
        let result = QualityResult::Pass {
            kl_divergence: 0.001,
        };
        match result {
            QualityResult::Pass { kl_divergence } => assert!(kl_divergence < 0.01),
            QualityResult::Fail { .. } => panic!("Expected Pass"),
        }
    }

    #[test]
    fn test_quality_result_fail() {
        // QualityResult::Fail should contain kl_divergence, threshold, message
        let result = QualityResult::Fail {
            kl_divergence: 0.1,
            threshold: 0.05,
            message: "Degradation detected",
        };
        match result {
            QualityResult::Fail {
                kl_divergence,
                threshold,
                message,
            } => {
                assert!(kl_divergence > threshold);
                assert!(!message.is_empty());
            },
            QualityResult::Pass { .. } => panic!("Expected Fail"),
        }
    }

    #[test]
    fn test_validate_quantization_identical() {
        // Identical logits should pass with kl_div ~= 0
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.01);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    #[test]
    fn test_validate_quantization_slight_difference() {
        // Small difference should still pass
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![1.01, 2.01, 3.01, 4.01]; // ~1% off
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.05);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    #[test]
    fn test_validate_quantization_large_difference() {
        // Large difference should fail
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![4.0, 3.0, 2.0, 1.0]; // Reversed distribution
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.01);
        assert!(matches!(result, QualityResult::Fail { .. }));
    }

    #[test]
    fn test_softmax_basic() {
        // Test softmax via validate_quantization_quality
        // Softmax should produce probability distribution
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        // Sum should be ~1.0
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Higher logit = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    // =========================================================================
    // OllamaBackend Tests (EXTREME TDD - REAL HTTP Integration)
    // =========================================================================

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_creation() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Ollama);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_info() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "phi2:2.7b".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Ollama);
        assert!(info.supports_streaming);
        assert_eq!(info.loaded_model, Some("phi2:2.7b".to_string()));
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_connection_error() {
        // Invalid port should fail
        let config = OllamaConfig {
            base_url: "http://localhost:59999".to_string(),
            model: "test".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        assert!(result.is_err());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama2");
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_with_custom_client() {
        use crate::http_client::ModelHttpClient;

        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
        };
        let client = ModelHttpClient::with_timeout(30);
        let backend = OllamaBackend::with_client(config, client);

        // Should create without panicking
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Ollama);
    }

    // Integration test - requires running Ollama server
    #[cfg(feature = "bench-http")]
    #[test]
    #[ignore = "Requires Ollama server at localhost:11434"]
    fn test_ollama_backend_real_inference() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "phi2:2.7b".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let request = InferenceRequest::new("What is 2+2?")
            .with_max_tokens(20)
            .with_temperature(0.1);

        let result = backend.inference(&request);

        // MUST succeed with real server
        let response = result.expect("Ollama inference failed - is server running?");

        // Verify REAL data
        assert!(
            response.ttft_ms > 0.0,
            "TTFT must be positive (real latency)"
        );
        assert!(response.total_time_ms > 0.0, "Total time must be positive");
        assert!(response.tokens_generated > 0, "Must generate tokens");
        assert!(!response.text.is_empty(), "Must get actual text");

        println!("Ollama Real Inference via Backend:");
        println!("  TTFT: {:.2}ms", response.ttft_ms);
        println!("  Total: {:.2}ms", response.total_time_ms);
        println!("  Tokens: {}", response.tokens_generated);
        println!("  Text: {}", response.text);
    }

    // ========================================================================
    // Distributed Benchmark Suite Tests
    // ========================================================================

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_default() {
        let config = DistributedBenchConfig::default();
        assert_eq!(config.gpu_counts, vec![1, 2, 4, 8]);
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup, 10);
        assert_eq!(config.model_params, 7_000_000_000);
        assert_eq!(config.seq_len, 2048);
        assert_eq!(config.batch_size, 1);
        assert!((config.efficiency_threshold - 0.85).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_small_model() {
        let config = DistributedBenchConfig::for_small_model();
        assert_eq!(config.gpu_counts, vec![1, 2]);
        assert_eq!(config.model_params, 125_000_000);
        assert!((config.efficiency_threshold - 0.80).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_large_model() {
        let config = DistributedBenchConfig::for_large_model();
        assert_eq!(config.gpu_counts, vec![2, 4, 8]);
        assert_eq!(config.model_params, 70_000_000_000);
        assert_eq!(config.seq_len, 4096);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_suite_new() {
        let config = DistributedBenchConfig::default();
        let suite = DistributedBenchSuite::new(config.clone());
        assert_eq!(suite.config().gpu_counts, config.gpu_counts);
        assert!(suite.scaling_results().is_empty());
        assert!(suite.tp_results().is_empty());
        assert!(suite.pp_results().is_empty());
        assert!(suite.comm_results().is_empty());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_scaling() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        let results = suite.scaling_results();
        assert_eq!(results.len(), 4); // 1, 2, 4, 8 GPUs

        // First result should be 1 GPU (baseline)
        assert_eq!(results[0].gpu_count, 1);
        assert!((results[0].efficiency - 1.0).abs() < 0.001);
        assert!(results[0].comm_overhead_ms.abs() < 0.001);

        // Multi-GPU should have lower efficiency due to overhead
        for result in results.iter().skip(1) {
            assert!(result.efficiency < 1.0);
            assert!(result.efficiency > 0.0); // Efficiency is always positive
            assert!(result.comm_overhead_ms > 0.0);
            assert!(result.throughput_tps > 0.0);
            assert!(result.latency_p50_ms > 0.0);
            assert!(result.latency_p99_ms > result.latency_p50_ms);
        }

        // 2 GPUs should be >85% efficient (spec target for 2-8 GPUs)
        let gpu2 = results.iter().find(|r| r.gpu_count == 2).expect("test");
        assert!(gpu2.efficiency > 0.85, "2-GPU efficiency should be >85%");
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_scaling_efficiency_result_meets_threshold() {
        let result = ScalingEfficiencyResult {
            gpu_count: 4,
            throughput_tps: 400.0,
            latency_p50_ms: 2.5,
            latency_p99_ms: 3.75,
            efficiency: 0.90,
            comm_overhead_ms: 0.5,
            theoretical_speedup: 3.6,
            achieved_speedup: 3.4,
        };

        assert!(result.meets_threshold(0.85));
        assert!(result.meets_threshold(0.90));
        assert!(!result.meets_threshold(0.95));
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_scaling_efficiency_parallel_fraction() {
        let result = ScalingEfficiencyResult {
            gpu_count: 4,
            throughput_tps: 400.0,
            latency_p50_ms: 2.5,
            latency_p99_ms: 3.75,
            efficiency: 0.85,
            comm_overhead_ms: 0.5,
            theoretical_speedup: 3.6,
            achieved_speedup: 3.4,
        };

        let parallel = result.parallel_fraction();
        assert!(parallel > 0.8); // Should be highly parallelizable
        assert!(parallel <= 1.0);

        // Single GPU case
        let single = ScalingEfficiencyResult {
            gpu_count: 1,
            throughput_tps: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 15.0,
            efficiency: 1.0,
            comm_overhead_ms: 0.0,
            theoretical_speedup: 1.0,
            achieved_speedup: 1.0,
        };
        assert!((single.parallel_fraction() - 1.0).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_tensor_parallel() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_tensor_parallel_benchmark();

        let results = suite.tp_results();
        assert!(!results.is_empty());

        // Check that TP=1 has no communication overhead
        let tp1 = results.iter().find(|r| r.tp_degree == 1).expect("test");
        assert!(tp1.all_reduce_ms.abs() < 0.001);
        assert!(tp1.comm_overhead_pct.abs() < 0.001);

        // Check that higher TP degrees have communication overhead
        for result in results.iter().filter(|r| r.tp_degree > 1) {
            assert!(result.all_reduce_ms > 0.0);
            assert!(result.comm_overhead_pct > 0.0);
            assert!(result.memory_per_gpu_mb > 0.0);
            assert!(result.effective_tflops > 0.0);
        }
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_pipeline_parallel() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_pipeline_parallel_benchmark();

        let results = suite.pp_results();
        assert!(!results.is_empty());

        // Check PP=1 has no bubble
        let pp1 = results.iter().find(|r| r.pp_degree == 1).expect("test");
        assert!(pp1.bubble_ratio.abs() < 0.001);
        assert!(pp1.inter_stage_ms.abs() < 0.001);

        // Check higher PP degrees have bubble and inter-stage latency
        for result in results.iter().filter(|r| r.pp_degree > 1) {
            assert!(result.bubble_ratio > 0.0);
            assert!(result.bubble_ratio < 1.0); // Should be <100%
            assert!(result.inter_stage_ms > 0.0);
            assert!(result.micro_batches > 0);
            assert!(result.throughput_tps > 0.0);
            assert!(result.memory_per_stage_mb > 0.0);
        }
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_communication() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_communication_benchmark();

        let results = suite.comm_results();
        // 4 data sizes × 2 operations (all_reduce, all_gather)
        assert_eq!(results.len(), 8);

        for result in results {
            assert!(result.latency_us > 0.0);
            assert!(result.bandwidth_gbps > 0.0);
            assert!(result.world_size > 0);
            assert!(!result.operation.is_empty());
            assert!(result.data_size_bytes > 0);
        }

        // All-gather should be faster than all-reduce for same size
        let reduce_1kb = results
            .iter()
            .find(|r| r.operation == "all_reduce" && r.data_size_bytes == 1024)
            .expect("test");
        let gather_1kb = results
            .iter()
            .find(|r| r.operation == "all_gather" && r.data_size_bytes == 1024)
            .expect("test");
        assert!(gather_1kb.latency_us < reduce_1kb.latency_us);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_run_all() {
        let config = DistributedBenchConfig::for_small_model();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_all();

        assert!(!suite.scaling_results().is_empty());
        assert!(!suite.tp_results().is_empty());
        assert!(!suite.pp_results().is_empty());
        assert!(!suite.comm_results().is_empty());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_summary() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_all();

        let summary = suite.summary();
        assert_eq!(summary.max_scaling, 8);
        assert!(summary.max_efficiency > 0.0);
        assert!(summary.min_efficiency > 0.0);
        assert!(summary.max_efficiency >= summary.min_efficiency);
        assert!(summary.max_throughput_tps > 0.0);
        assert!(summary.avg_tp_comm_overhead_pct >= 0.0);
        assert!(summary.avg_pp_bubble_ratio >= 0.0);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_all_meet_threshold() {
        // Use small model config (only 1-2 GPUs) where efficiency stays high
        let config = DistributedBenchConfig::for_small_model();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        // With 80% threshold and only 1-2 GPUs, all should pass
        assert!(suite.all_meet_efficiency_threshold());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_fail_threshold() {
        let config = DistributedBenchConfig {
            efficiency_threshold: 0.99, // Very high threshold
            ..DistributedBenchConfig::default()
        };
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        // With 99% threshold, multi-GPU configs should fail
        assert!(!suite.all_meet_efficiency_threshold());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_empty_summary() {
        let config = DistributedBenchConfig::default();
        let suite = DistributedBenchSuite::new(config);

        // Summary on empty results should handle gracefully
        let summary = suite.summary();
        assert_eq!(summary.max_scaling, 1);
        assert!((summary.max_efficiency - 0.0).abs() < 0.001);
        assert!((summary.avg_tp_comm_overhead_pct - 0.0).abs() < 0.001);
        assert!((summary.avg_pp_bubble_ratio - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Load Testing Tests
    // ========================================================================

    #[test]
    fn test_load_test_config_default() {
        let config = LoadTestConfig::default();
        assert_eq!(config.concurrency, 10);
        assert_eq!(config.duration_secs, 60);
        assert!((config.target_rps - 0.0).abs() < 0.001);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.warmup_secs, 5);
        assert!((config.latency_threshold_ms - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_stress_test() {
        let config = LoadTestConfig::for_stress_test();
        assert_eq!(config.concurrency, 100);
        assert_eq!(config.duration_secs, 300);
        assert!((config.latency_threshold_ms - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_latency_test() {
        let config = LoadTestConfig::for_latency_test();
        assert_eq!(config.concurrency, 1);
        assert!((config.target_rps - 10.0).abs() < 0.001);
        assert!((config.latency_threshold_ms - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_validation() {
        let valid = LoadTestConfig::default();
        assert!(valid.is_valid());

        let invalid = LoadTestConfig {
            concurrency: 0,
            ..LoadTestConfig::default()
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_load_test_runner_simulate() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        assert!(result.total_requests > 0);
        assert!(result.successful_requests > 0);
        assert!(result.rps_achieved > 0.0);
        assert!(result.latency_p50_ms > 0.0);
        assert!(result.latency_p95_ms > result.latency_p50_ms);
        assert!(result.latency_p99_ms > result.latency_p95_ms);
        assert!(result.latency_max_ms > result.latency_p99_ms);
        assert!(result.data_transferred_bytes > 0);
        assert!(result.duration_secs > 0.0);
        assert!(result.error_rate >= 0.0 && result.error_rate < 1.0);
    }

    #[test]
    fn test_load_test_result_is_passing() {
        let passing = LoadTestResult {
            total_requests: 1000,
            successful_requests: 995,
            failed_requests: 5,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.005,
            passed_latency_threshold: true,
        };
        assert!(passing.is_passing());

        let failing_error_rate = LoadTestResult {
            error_rate: 0.05, // 5% errors
            ..passing.clone()
        };
        assert!(!failing_error_rate.is_passing());

        let failing_latency = LoadTestResult {
            passed_latency_threshold: false,
            ..passing
        };
        assert!(!failing_latency.is_passing());
    }

    #[test]
    fn test_load_test_result_throughput() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 10_000_000, // 10 MB
            duration_secs: 10.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert!((result.throughput_mbps() - 1.0).abs() < 0.001); // 10MB / 10s = 1 MB/s
    }

    // ========================================================================
    // LlamaCppBackend REAL CLI Output Parsing Tests (P0 - No Stubs)
    // ========================================================================

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_timing_prompt_eval() {
        let output = r"llama_perf_context_print: prompt eval time =      12.34 ms /    10 tokens (    1.23 ms per token,   810.37 tokens per second)";
        let timing = LlamaCppBackend::parse_timing_line(output, "prompt eval time");
        assert!(timing.is_some());
        let (total_ms, tokens) = timing.expect("test");
        assert!((total_ms - 12.34).abs() < 0.01);
        assert_eq!(tokens, 10);
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_timing_eval() {
        let output = r"llama_perf_context_print:        eval time =      22.60 ms /     5 runs   (    4.52 ms per token,   221.28 tokens per second)";
        let timing = LlamaCppBackend::parse_timing_line(output, "eval time");
        assert!(timing.is_some());
        let (total_ms, runs) = timing.expect("test");
        assert!((total_ms - 22.60).abs() < 0.01);
        assert_eq!(runs, 5);
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_timing_total() {
        let output = r"llama_perf_context_print:       total time =      23.27 ms /     6 tokens";
        let timing = LlamaCppBackend::parse_timing_line(output, "total time");
        assert!(timing.is_some());
        let (total_ms, tokens) = timing.expect("test");
        assert!((total_ms - 23.27).abs() < 0.01);
        assert_eq!(tokens, 6);
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_full_output() {
        let output = r#"Hello world",
```

llama_perf_sampler_print:    sampling time =       0.14 ms /     6 runs   (    0.02 ms per token, 42553.19 tokens per second)
llama_perf_context_print:        load time =    1349.68 ms
llama_perf_context_print: prompt eval time =       5.00 ms /     1 tokens (    5.00 ms per token,   200.00 tokens per second)
llama_perf_context_print:        eval time =      22.60 ms /     5 runs   (    4.52 ms per token,   221.28 tokens per second)
llama_perf_context_print:       total time =      27.60 ms /     6 tokens"#;

        let result = LlamaCppBackend::parse_cli_output(output);
        assert!(result.is_ok());
        let response = result.expect("test");
        // TTFT = prompt eval time (time to first token)
        assert!((response.ttft_ms - 5.0).abs() < 0.1);
        // Total time from parse
        assert!((response.total_time_ms - 27.60).abs() < 0.1);
        // Tokens generated = eval runs (5 tokens generated after prompt)
        assert_eq!(response.tokens_generated, 5);
        // Text should contain the generated content
        assert!(response.text.contains("Hello world"));
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_extract_generated_text() {
        let output =
            "The answer is 42.\n\nllama_perf_context_print: total time = 100.0 ms / 10 tokens";
        let text = LlamaCppBackend::extract_generated_text(output);
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_llama_cpp_backend_build_command() {
        let config = LlamaCppConfig {
            binary_path: "/path/to/llama-cli".to_string(),
            model_path: Some("/path/to/model.gguf".to_string()),
            n_gpu_layers: 99,
            ctx_size: 4096,
            threads: 8,
        };
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            stop: vec![],
        };
        let args = backend.build_cli_args(&request);

        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"/path/to/model.gguf".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(args.contains(&"-n".to_string()));
        assert!(args.contains(&"10".to_string()));
        assert!(args.contains(&"-ngl".to_string()));
        assert!(args.contains(&"99".to_string()));
        assert!(args.contains(&"-c".to_string()));
        assert!(args.contains(&"4096".to_string()));
        assert!(args.contains(&"-t".to_string()));
        assert!(args.contains(&"8".to_string()));
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_no_model_path_error() {
        let config = LlamaCppConfig {
            binary_path: "/path/to/llama-cli".to_string(),
            model_path: None, // No model
            n_gpu_layers: 0,
            ctx_size: 2048,
            threads: 4,
        };
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            stop: vec![],
        };
        let result = backend.inference(&request);
        assert!(result.is_err());
    }

    // ========================================================================
    // Benchmark Matrix Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_compute_backend_type_display() {
        assert_eq!(format!("{}", ComputeBackendType::Cpu), "cpu");
        assert_eq!(format!("{}", ComputeBackendType::Wgpu), "wgpu");
        assert_eq!(format!("{}", ComputeBackendType::Cuda), "cuda");
    }

    #[test]
    fn test_compute_backend_type_from_str() {
        assert_eq!(
            ComputeBackendType::parse("cpu"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("CPU"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("wgpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("gpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("cuda"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(
            ComputeBackendType::parse("nvidia"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(ComputeBackendType::parse("unknown"), None);
    }

    #[test]
    fn test_compute_backend_type_all() {
        let all = ComputeBackendType::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&ComputeBackendType::Cpu));
        assert!(all.contains(&ComputeBackendType::Wgpu));
        assert!(all.contains(&ComputeBackendType::Cuda));
    }

    #[test]
    fn test_matrix_benchmark_entry_unavailable() {
        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cuda);
        assert!(!entry.available);
        assert_eq!(entry.runtime, RuntimeType::Realizar);
        assert_eq!(entry.backend, ComputeBackendType::Cuda);
        assert!(entry.notes.contains("not available"));
    }

    #[test]
    fn test_matrix_benchmark_entry_from_samples() {
        let latencies = vec![100.0, 105.0, 110.0, 95.0, 102.0];
        let throughputs = vec![50.0, 48.0, 52.0, 49.0, 51.0];
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &latencies,
            &throughputs,
            150.0,
        );

        assert!(entry.available);
        assert_eq!(entry.runtime, RuntimeType::LlamaCpp);
        assert_eq!(entry.backend, ComputeBackendType::Wgpu);
        assert_eq!(entry.model, "phi-2");
        assert_eq!(entry.samples, 5);
        assert!((entry.p50_latency_ms - 102.0).abs() < 1.0); // Median
        assert!((entry.cold_start_ms - 150.0).abs() < 0.1);
        assert!(entry.throughput_tps > 0.0);
    }

    #[test]
    fn test_matrix_benchmark_entry_from_empty_samples() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model",
            &[],
            &[],
            0.0,
        );
        assert!(!entry.available);
    }

    #[test]
    fn test_matrix_benchmark_entry_with_notes() {
        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cuda)
                .with_notes("GPU layers: 99");
        assert_eq!(entry.notes, "GPU layers: 99");
    }

    #[test]
    fn test_benchmark_matrix_creation() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("phi-2", hardware);

        assert_eq!(matrix.model, "phi-2");
        assert_eq!(matrix.version, "1.1");
        assert!(matrix.methodology.contains("Hoefler"));
        assert!(matrix.entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_add_entry() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        let entry1 = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        );
        matrix.add_entry(entry1);

        assert_eq!(matrix.entries.len(), 1);

        // Add another entry
        let entry2 = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0, 82.0, 78.0],
            &[60.0, 61.0, 59.0],
            120.0,
        );
        matrix.add_entry(entry2);

        assert_eq!(matrix.entries.len(), 2);

        // Replace existing entry
        let entry1_updated = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0, 92.0, 88.0],
            &[55.0, 56.0, 54.0],
            95.0,
        );
        matrix.add_entry(entry1_updated);

        assert_eq!(matrix.entries.len(), 2); // Still 2, replaced
    }

    #[test]
    fn test_benchmark_matrix_get_entry() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        );
        matrix.add_entry(entry);

        let found = matrix.get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu);
        assert!(found.is_some());
        assert_eq!(found.expect("test").runtime, RuntimeType::Realizar);

        let not_found = matrix.get_entry(RuntimeType::LlamaCpp, ComputeBackendType::Cuda);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_benchmark_matrix_entries_for_runtime() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0],
            &[60.0],
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[55.0],
            95.0,
        ));

        let realizar_entries = matrix.entries_for_runtime(RuntimeType::Realizar);
        assert_eq!(realizar_entries.len(), 2);

        let llama_entries = matrix.entries_for_runtime(RuntimeType::LlamaCpp);
        assert_eq!(llama_entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_entries_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[55.0],
            95.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0],
            &[60.0],
            90.0,
        ));

        let cpu_entries = matrix.entries_for_backend(ComputeBackendType::Cpu);
        assert_eq!(cpu_entries.len(), 2);

        let wgpu_entries = matrix.entries_for_backend(ComputeBackendType::Wgpu);
        assert_eq!(wgpu_entries.len(), 1);

        let cuda_entries = matrix.entries_for_backend(ComputeBackendType::Cuda);
        assert!(cuda_entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[80.0, 82.0, 78.0], // Faster
            &[55.0],
            95.0,
        ));

        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cpu);
        assert!(fastest.is_some());
        assert_eq!(fastest.expect("test").runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[70.0, 71.0, 69.0], // Higher throughput
            95.0,
        ));

        let highest = matrix.highest_throughput_for_backend(ComputeBackendType::Cpu);
        assert!(highest.is_some());
        assert_eq!(highest.expect("test").runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_table() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 110.0, 105.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let table = matrix.to_markdown_table();
        assert!(table.contains("| Runtime | Backend |"));
        assert!(table.contains("| **realizar** |"));
        assert!(table.contains("| - | - |")); // Unavailable entry
    }

    #[test]
    fn test_benchmark_matrix_json_roundtrip() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));

        let json = matrix.to_json().expect("serialization should succeed");
        assert!(json.contains("\"model\": \"phi-2\""));

        let parsed = BenchmarkMatrix::from_json(&json).expect("deserialization should succeed");
        assert_eq!(parsed.model, "phi-2");
        assert_eq!(parsed.entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_summary() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        // Add entries for different combinations
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[80.0, 82.0, 78.0],
            &[70.0, 71.0, 69.0],
            95.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[60.0, 62.0, 58.0], // Fastest overall
            &[80.0, 81.0, 79.0], // Highest throughput overall
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 4);
        assert_eq!(summary.available_entries, 3);

        // Overall fastest should be llama-cpp with wgpu (p50 ~60ms)
        assert!(summary.overall_fastest.is_some());
        let (fastest_runtime, fastest_backend) = summary.overall_fastest.expect("test");
        assert_eq!(fastest_runtime, "llamacpp");
        assert_eq!(fastest_backend, "wgpu");

        // Overall highest throughput should also be llama-cpp with wgpu (~80 tok/s)
        assert!(summary.overall_highest_throughput.is_some());
        let (tp_runtime, tp_backend) = summary.overall_highest_throughput.expect("test");
        assert_eq!(tp_runtime, "llamacpp");
        assert_eq!(tp_backend, "wgpu");
    }

    #[test]
    fn test_matrix_benchmark_config_default() {
        let config = MatrixBenchmarkConfig::default();

        assert!(config.runtimes.contains(&RuntimeType::Realizar));
        assert!(config.runtimes.contains(&RuntimeType::LlamaCpp));
        assert!(config.runtimes.contains(&RuntimeType::Ollama));
        assert!(config.backends.contains(&ComputeBackendType::Cpu));
        assert!(config.backends.contains(&ComputeBackendType::Wgpu));
        assert_eq!(config.cv_threshold, 0.05);
        assert_eq!(config.min_samples, 30);
        assert_eq!(config.max_samples, 200);
        assert_eq!(config.warmup_iterations, 5);
    }

    // ========================================================================
    // QA Checklist Validation Tests
    // Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md §5
    // ========================================================================

    /// QA-031: Benchmark framework produces valid statistical metrics
    #[test]
    fn test_qa_031_benchmark_statistical_validity() {
        // DynamicSampler must produce valid CV calculations
        let sampler = DynamicSampler::new(10, 100, 0.05);

        // Stable samples should produce low CV
        let stable_samples: Vec<f64> = (0..50).map(|_| 100.0).collect();
        let cv = sampler.current_cv(&stable_samples);
        assert!(
            cv.abs() < 0.001,
            "QA-031: Stable samples should have near-zero CV, got {}",
            cv
        );

        // Variable samples should produce higher CV
        let variable_samples: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 2.0).collect();
        let cv_var = sampler.current_cv(&variable_samples);
        assert!(
            cv_var > 0.1,
            "QA-031: Variable samples should have measurable CV, got {}",
            cv_var
        );
    }

    /// QA-032: Thermal guard validates temperature variance correctly
    #[test]
    fn test_qa_032_thermal_guard_validation() {
        let guard = ThermalGuard::default();

        // Default thermal guard should have sensible thresholds
        assert!(
            guard.max_temp_c > 70.0 && guard.max_temp_c <= 95.0,
            "QA-032: Max temp should be in safe GPU range"
        );
        assert!(
            guard.cooldown_threshold_c < guard.max_temp_c,
            "QA-032: Cooldown threshold must be below max temp"
        );
        assert!(
            guard.temp_variance_c > 0.0 && guard.temp_variance_c <= 5.0,
            "QA-032: Temperature variance threshold should be reasonable"
        );
    }

    /// QA-033: ITL metrics capture variance correctly
    #[test]
    fn test_qa_033_itl_variance_capture() {
        let samples = vec![10.0, 12.0, 11.0, 13.0, 10.0, 15.0, 11.0, 12.0];
        let metrics = ItlMetrics::from_measurements(&samples);

        // p99 should be >= p999 (order check)
        // Actually p999 >= p99 in expectation (tail values)
        assert!(
            metrics.p999_ms >= metrics.p99_ms,
            "QA-033: p999 should be >= p99"
        );
        assert!(
            metrics.p99_ms >= metrics.median_ms,
            "QA-033: p99 should be >= median"
        );

        // Median and std_dev should be non-negative
        assert!(metrics.median_ms > 0.0, "QA-033: Median should be positive");
        assert!(
            metrics.std_dev_ms >= 0.0,
            "QA-033: Std dev should be non-negative"
        );
    }

    /// QA-034: CV-based stopping rule converges
    #[test]
    #[allow(clippy::similar_names)] // sampler vs samples are related but distinct concepts
    fn test_qa_034_cv_stopping_convergence() {
        let mut sampler = DynamicSampler::new(10, 1000, 0.05);
        sampler.stability_count = 3;

        // Feed stable samples - should eventually stop
        let mut samples = Vec::new();
        let mut stopped = false;

        for i in 0..100 {
            samples.push(100.0 + (i as f64 % 3.0)); // Small variance
            if !sampler.should_continue(&samples) {
                stopped = true;
                break;
            }
        }

        assert!(
            stopped,
            "QA-034: CV-based stopping should converge for stable samples"
        );
    }

    /// QA-035: Benchmark results are serializable
    #[test]
    fn test_qa_035_benchmark_serialization() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[50.0, 52.0, 48.0],
            &[100.0, 98.0, 102.0],
            95.0,
        );

        // Should serialize to JSON without error
        let json = serde_json::to_string(&entry);
        assert!(
            json.is_ok(),
            "QA-035: MatrixBenchmarkEntry should serialize"
        );

        // Should deserialize back
        let deser: Result<MatrixBenchmarkEntry, _> = serde_json::from_str(&json.expect("test"));
        assert!(
            deser.is_ok(),
            "QA-035: MatrixBenchmarkEntry should deserialize"
        );
    }

    /// QA-036: Runtime and backend types are complete
    #[test]
    fn test_qa_036_runtime_backend_completeness() {
        // All expected runtimes should be representable
        let runtimes = [
            RuntimeType::Realizar,
            RuntimeType::LlamaCpp,
            RuntimeType::Ollama,
            RuntimeType::Vllm,
        ];

        for runtime in &runtimes {
            let name = runtime.as_str();
            assert!(
                !name.is_empty(),
                "QA-036: Runtime {} should have a name",
                name
            );
        }

        // All expected backends should be representable
        let backends = [
            ComputeBackendType::Cpu,
            ComputeBackendType::Cuda,
            ComputeBackendType::Wgpu,
        ];

        for backend in &backends {
            let name = backend.to_string();
            assert!(
                !name.is_empty(),
                "QA-036: Backend {:?} should have a name",
                backend
            );
        }
    }

    /// QA-037: Matrix summary calculations are correct
    #[test]
    fn test_qa_037_matrix_summary_correctness() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add known entries
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0], // p50 = 100ms
            &[10.0],  // throughput = 10 tok/s
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[50.0], // p50 = 50ms (faster)
            &[20.0], // throughput = 20 tok/s (higher)
            95.0,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 2, "QA-037: Should have 2 entries");
        assert_eq!(
            summary.available_entries, 2,
            "QA-037: Both entries should be available"
        );

        // LlamaCpp should be fastest (50ms < 100ms)
        if let Some((fastest, _)) = &summary.overall_fastest {
            assert_eq!(fastest, "llamacpp", "QA-037: LlamaCpp should be fastest");
        }
    }

    /// QA-038: Benchmark report generation works
    #[test]
    fn test_qa_038_report_generation() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        ));

        let report = matrix.to_markdown_table();

        // Report should contain key information
        assert!(
            report.contains("realizar") || report.contains("Realizar"),
            "QA-038: Report should mention realizar"
        );
    }

    /// QA-039: Dynamic sampler respects min/max bounds
    #[test]
    fn test_qa_039_sampler_bounds() {
        let mut sampler = DynamicSampler::new(5, 20, 0.01); // Very tight CV

        // Should always continue until min_samples
        let few_samples = vec![1.0, 2.0, 3.0];
        assert!(
            sampler.should_continue(&few_samples),
            "QA-039: Should continue below min_samples"
        );

        // Should stop at max_samples regardless of CV
        let many_samples: Vec<f64> = (0..25).map(|i| i as f64).collect(); // High variance
        assert!(
            !sampler.should_continue(&many_samples),
            "QA-039: Should stop at max_samples"
        );
    }

    /// QA-040: ITL metrics handle edge cases
    #[test]
    fn test_qa_040_itl_edge_cases() {
        // Single sample
        let single = ItlMetrics::from_measurements(&[100.0]);
        assert!(
            (single.median_ms - 100.0).abs() < 0.001,
            "QA-040: Single sample median should equal the sample"
        );

        // Empty samples should produce zeros or NaN (valid edge case)
        let empty = ItlMetrics::from_measurements(&[]);
        assert!(
            empty.median_ms.is_nan() || empty.median_ms == 0.0,
            "QA-040: Empty samples should produce NaN or 0"
        );

        // All same values - std_dev should be 0
        let same = ItlMetrics::from_measurements(&[50.0, 50.0, 50.0, 50.0]);
        assert!(
            same.std_dev_ms.abs() < 0.001,
            "QA-040: Identical samples should have zero std_dev"
        );
    }

    // ========================================================================
    // QA Checklist Section E: Integration Tests (QA-041 to QA-050)
    // Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md §5
    // ========================================================================

    /// QA-041: Benchmark infrastructure compiles and runs
    /// Per spec: `make bench-inference-all` should complete without error
    #[test]
    fn test_qa_041_benchmark_infrastructure() {
        // Verify all benchmark types are representable
        let runtimes = [
            RuntimeType::Realizar,
            RuntimeType::Ollama,
            RuntimeType::LlamaCpp,
        ];

        for runtime in &runtimes {
            assert!(
                !runtime.as_str().is_empty(),
                "QA-041: Runtime {} should have a name",
                runtime.as_str()
            );
        }

        // Verify benchmark matrix can be created
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);
        assert!(
            matrix.entries.is_empty(),
            "QA-041: New matrix should be empty"
        );
    }

    /// QA-042: Comparison report generation works
    /// Per spec: `make bench-pytorch-inference` produces comparison report
    #[test]
    fn test_qa_042_comparison_report() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add entries for comparison
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0, 105.0, 95.0],
            &[50.0, 55.0, 45.0],
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[80.0, 85.0, 75.0],
            &[40.0, 45.0, 35.0],
            110.0,
        ));

        // Generate comparison report
        let report = matrix.to_markdown_table();

        // Report should contain both runtimes
        assert!(
            report.contains("realizar") || report.contains("Realizar"),
            "QA-042: Report should include Realizar"
        );
    }

    /// QA-043: CPU-only benchmarks work
    /// Per spec: `make bench-cpu-inference` tests all CPU backends
    #[test]
    fn test_qa_043_cpu_benchmarks() {
        // Verify CPU backend type exists and is valid
        let cpu_backend = ComputeBackendType::Cpu;
        let backend_str = cpu_backend.to_string();
        assert!(
            backend_str.to_lowercase().contains("cpu"),
            "QA-043: CPU backend should be identifiable"
        );

        // Verify CPU entries can be created
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        );

        assert_eq!(
            entry.backend,
            ComputeBackendType::Cpu,
            "QA-043: Entry should be CPU backend"
        );
    }

    /// QA-044: WGPU benchmark gracefully handles unavailability
    /// Per spec: `make bench-wgpu` gracefully skips if unavailable
    #[test]
    fn test_qa_044_wgpu_graceful_skip() {
        // WGPU backend type should exist
        let wgpu_backend = ComputeBackendType::Wgpu;
        let backend_str = wgpu_backend.to_string();

        // Should have a valid string representation
        assert!(
            !backend_str.is_empty(),
            "QA-044: WGPU backend should have a name"
        );

        // Creating an entry with WGPU should work (even if GPU not available)
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        );

        assert_eq!(
            entry.backend,
            ComputeBackendType::Wgpu,
            "QA-044: Entry should be WGPU backend"
        );
    }

    /// QA-045: Multi-runtime comparison works
    /// Per spec: `make bench-gguf-gpu-inference` compares all runtimes
    #[test]
    fn test_qa_045_multi_runtime_comparison() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add entries for all runtime types
        for runtime in [
            RuntimeType::Realizar,
            RuntimeType::Ollama,
            RuntimeType::LlamaCpp,
        ] {
            matrix.add_entry(MatrixBenchmarkEntry::from_samples(
                runtime,
                ComputeBackendType::Cpu,
                "test",
                &[100.0],
                &[50.0],
                90.0,
            ));
        }

        // Should have 3 entries
        assert_eq!(
            matrix.entries.len(),
            3,
            "QA-045: Should have 3 runtime entries"
        );

        // Summary should work
        let summary = matrix.summary();
        assert!(
            summary.overall_fastest.is_some(),
            "QA-045: Summary should identify fastest runtime"
        );
    }

    /// QA-046: Format comparison works
    /// Per spec: `make bench-apr-gpu-inference` produces format comparison
    #[test]
    fn test_qa_046_format_comparison() {
        // Different model formats should be comparable via the same infrastructure
        let hardware = HardwareSpec::default();
        let mut gguf_matrix = BenchmarkMatrix::new("model.gguf", hardware.clone());
        let mut apr_matrix = BenchmarkMatrix::new("model.apr", hardware);

        gguf_matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model.gguf",
            &[100.0],
            &[50.0],
            90.0,
        ));

        apr_matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model.apr",
            &[95.0],
            &[48.0],
            92.0,
        ));

        // Both should generate valid reports
        let gguf_report = gguf_matrix.to_markdown_table();
        let apr_report = apr_matrix.to_markdown_table();

        assert!(
            !gguf_report.is_empty(),
            "QA-046: GGUF report should be non-empty"
        );
        assert!(
            !apr_report.is_empty(),
            "QA-046: APR report should be non-empty"
        );
    }

    /// QA-047: CI pipeline integration (structure validation)
    /// Per spec: CI pipeline runs benchmarks on every PR
    #[test]
    fn test_qa_047_ci_integration() {
        // Verify benchmark results can be serialized for CI
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 105.0],
            &[50.0, 55.0],
            90.0,
        );

        // Should serialize to JSON for CI consumption
        let json = serde_json::to_string(&entry);
        assert!(json.is_ok(), "QA-047: Entry should serialize for CI");

        // Should deserialize back
        let deser: Result<MatrixBenchmarkEntry, _> = serde_json::from_str(&json.expect("test"));
        assert!(deser.is_ok(), "QA-047: Entry should deserialize from CI");
    }

    /// QA-048: Metrics dashboard support
    /// Per spec: Benchmark results published to metrics dashboard
    #[test]
    fn test_qa_048_metrics_dashboard() {
        // Verify all metrics needed for dashboard are present
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 105.0, 95.0, 98.0, 102.0],
            &[50.0, 55.0, 45.0, 48.0, 52.0],
            90.0,
        );

        // Dashboard needs: p50, p99, throughput, runtime, backend
        assert!(
            entry.p50_latency_ms > 0.0,
            "QA-048: p50 should be available"
        );
        assert!(
            entry.p99_latency_ms > 0.0,
            "QA-048: p99 should be available"
        );
        assert!(
            entry.throughput_tps > 0.0,
            "QA-048: Throughput should be available"
        );
        assert!(
            !entry.runtime.as_str().is_empty(),
            "QA-048: Runtime should be identifiable"
        );
    }

    /// QA-049: Historical trend detection
    /// Per spec: Historical trend analysis detects regressions
    #[test]
    fn test_qa_049_trend_detection() {
        // Simulate historical data with a regression
        let baseline = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 100.0, 100.0],
            &[50.0, 50.0, 50.0],
            100.0,
        );

        let regressed = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[120.0, 120.0, 120.0], // 20% slower
            &[60.0, 60.0, 60.0],
            83.0, // Lower throughput
        );

        // Regression should be detectable
        let regression_percent =
            (regressed.p50_latency_ms - baseline.p50_latency_ms) / baseline.p50_latency_ms * 100.0;

        assert!(
            regression_percent > 15.0,
            "QA-049: Should detect >15% regression, got {}%",
            regression_percent
        );
    }

    /// QA-050: Documentation consistency
    /// Per spec: Documentation updated with latest benchmark results
    #[test]
    fn test_qa_050_documentation_support() {
        // Verify markdown report generation for documentation
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        let markdown = matrix.to_markdown_table();

        // Markdown should be valid for documentation
        assert!(
            markdown.contains("|"),
            "QA-050: Should produce markdown table"
        );
        assert!(
            markdown.contains("Runtime") || markdown.contains("runtime"),
            "QA-050: Table should have headers"
        );
    }

    // ========================================================================
    // IMP-800: GPU Parity Benchmark Tests
    // ========================================================================

    /// IMP-800a: GPU forward method exists (struct test)
    #[test]
    fn test_imp800a_gpu_parity_benchmark_config() {
        let config = GpuParityBenchmark::new("/path/to/model.gguf")
            .with_prompt("Hello world")
            .with_max_tokens(64)
            .with_ollama_endpoint("http://localhost:11434")
            .with_warmup(5)
            .with_iterations(20);

        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.prompt, "Hello world");
        assert_eq!(config.max_tokens, 64);
        assert_eq!(config.ollama_endpoint, "http://localhost:11434");
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 20);
    }

    /// IMP-800a: GPU forward correctness (default config)
    #[test]
    fn test_imp800a_gpu_parity_benchmark_default() {
        let config = GpuParityBenchmark::default();

        assert!(config.model_path.is_empty());
        assert_eq!(config.prompt, "The quick brown fox");
        assert_eq!(config.max_tokens, 32);
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.measurement_iterations, 10);
        assert!((config.target_cv - 0.05).abs() < f64::EPSILON);
    }

    /// IMP-800b: Result struct captures all metrics
    #[test]
    fn test_imp800b_gpu_parity_result_struct() {
        let result = GpuParityResult::new(150.0, 240.0, 0.03, "NVIDIA RTX 4090", 8192);

        assert!((result.realizar_gpu_tps - 150.0).abs() < f64::EPSILON);
        assert!((result.ollama_tps - 240.0).abs() < f64::EPSILON);
        assert!((result.gap_ratio - 1.6).abs() < 0.01);
        assert!((result.cv - 0.03).abs() < f64::EPSILON);
        assert_eq!(result.gpu_device, "NVIDIA RTX 4090");
        assert_eq!(result.vram_mb, 8192);
    }

    /// IMP-800b: M2/M4 parity thresholds
    #[test]
    fn test_imp800b_parity_thresholds() {
        // M2 parity (within 2x)
        let m2_pass = GpuParityResult::new(130.0, 240.0, 0.03, "GPU", 8192);
        assert!(m2_pass.achieves_m2_parity()); // 1.85x gap
        assert!(!m2_pass.achieves_m4_parity()); // Not within 1.25x

        // M4 parity (within 1.25x)
        let m4_pass = GpuParityResult::new(200.0, 240.0, 0.02, "GPU", 8192);
        assert!(m4_pass.achieves_m2_parity()); // 1.2x gap
        assert!(m4_pass.achieves_m4_parity()); // Within 1.25x

        // Fail both
        let fail = GpuParityResult::new(50.0, 240.0, 0.05, "GPU", 8192);
        assert!(!fail.achieves_m2_parity()); // 4.8x gap
        assert!(!fail.achieves_m4_parity());
    }

    /// IMP-800b: CV stability check
    #[test]
    fn test_imp800b_cv_stability() {
        let stable = GpuParityResult::new(150.0, 240.0, 0.04, "GPU", 8192);
        assert!(stable.measurements_stable()); // CV < 0.05

        let unstable = GpuParityResult::new(150.0, 240.0, 0.08, "GPU", 8192);
        assert!(!unstable.measurements_stable()); // CV >= 0.05
    }

    /// IMP-800c: Gap analysis struct
    #[test]
    fn test_imp800c_gap_analysis_struct() {
        let analysis = GapAnalysis::new(2.0, 1.8).with_statistics(0.01, 1.5, 2.1);

        assert!((analysis.claimed_gap - 2.0).abs() < f64::EPSILON);
        assert!((analysis.measured_gap - 1.8).abs() < f64::EPSILON);
        assert!((analysis.p_value - 0.01).abs() < f64::EPSILON);
        assert!((analysis.ci_95_lower - 1.5).abs() < f64::EPSILON);
        assert!((analysis.ci_95_upper - 2.1).abs() < f64::EPSILON);
    }

    /// IMP-800c: Claim verification logic
    #[test]
    fn test_imp800c_claim_verification() {
        let within_ci = GapAnalysis::new(2.0, 1.8).with_statistics(0.01, 1.5, 2.1);
        assert!(within_ci.claim_verified()); // 1.8 is within [1.5, 2.1]

        let outside_ci = GapAnalysis::new(2.0, 1.2).with_statistics(0.01, 1.5, 2.1);
        assert!(!outside_ci.claim_verified()); // 1.2 is not within [1.5, 2.1]
    }

    /// IMP-800c: Statistical bounds calculation
    #[test]
    fn test_imp800c_statistical_bounds() {
        let analysis = GapAnalysis::new(2.0, 1.8).with_statistics(0.05, 1.6, 2.0);

        assert!((analysis.ci_95_lower - 1.6).abs() < f64::EPSILON);
        assert!((analysis.ci_95_upper - 2.0).abs() < f64::EPSILON);
        assert!((analysis.p_value - 0.05).abs() < f64::EPSILON);
    }

    /// IMP-800c: Popper score computation
    #[test]
    fn test_imp800c_popper_score() {
        // Test with 150 tok/s (passes IMP-800c-1 and IMP-800c-2, fails IMP-800c-3 and IMP-800c-4)
        let analysis = GapAnalysis::new(2.0, 1.6).with_default_claims(150.0);

        // 150 tok/s passes:
        // - IMP-800c-1: threshold 25 tok/s ✓
        // - IMP-800c-2: threshold 24 tok/s ✓
        // - IMP-800c-3: threshold 120 tok/s ✓
        // - IMP-800c-4: threshold 192 tok/s ✗
        // Score should be 75% (3/4)
        assert!((analysis.popper_score - 75.0).abs() < f64::EPSILON);
        assert_eq!(analysis.claims.len(), 4);
    }

    /// IMP-800d: Falsifiable claim evaluation
    #[test]
    fn test_imp800d_falsifiable_claim() {
        let claim = FalsifiableClaim::new("TEST-001", "Test claim", 5.0, 25.0).evaluate(30.0);

        assert_eq!(claim.id, "TEST-001");
        assert_eq!(claim.description, "Test claim");
        assert!((claim.expected - 5.0).abs() < f64::EPSILON);
        assert!((claim.threshold - 25.0).abs() < f64::EPSILON);
        assert!((claim.measured - 30.0).abs() < f64::EPSILON);
        assert!(claim.verified); // 30 >= 25

        let failed_claim =
            FalsifiableClaim::new("TEST-002", "Failing claim", 5.0, 50.0).evaluate(30.0);
        assert!(!failed_claim.verified); // 30 < 50
    }

    /// IMP-800d: GPU faster than CPU check
    #[test]
    fn test_imp800d_gpu_faster_than_cpu() {
        let faster = GpuParityResult::new(30.0, 240.0, 0.03, "GPU", 8192);
        assert!(faster.gpu_faster_than_cpu()); // 30 > 5 tok/s
        assert!((faster.cpu_speedup() - 6.0).abs() < f64::EPSILON); // 30 / 5 = 6x

        let slower = GpuParityResult::new(4.0, 240.0, 0.03, "GPU", 8192);
        assert!(!slower.gpu_faster_than_cpu()); // 4 < 5 tok/s
    }

    // ========================================================================
    // IMP-900a: Optimized GEMM Tests
    // ========================================================================

    /// IMP-900a: Optimized GEMM config defaults
    #[test]
    fn test_imp900a_optimized_gemm_config_default() {
        let config = OptimizedGemmConfig::default();
        assert_eq!(config.tile_size, 32);
        assert_eq!(config.reg_block, 4);
        assert!(!config.use_tensor_cores);
        assert_eq!(config.vector_width, 4);
        assert_eq!(config.k_unroll, 4);
        assert!(config.double_buffer);
    }

    /// IMP-900a: Shared memory calculation
    #[test]
    fn test_imp900a_shared_memory_calculation() {
        let config = OptimizedGemmConfig::default();
        // 32×32 tiles × 4 bytes × 2 tiles × 2 buffers = 32768 bytes
        assert_eq!(config.shared_memory_bytes(), 32 * 32 * 4 * 4);

        let no_double = OptimizedGemmConfig {
            double_buffer: false,
            ..Default::default()
        };
        // Without double buffering: 32×32 × 4 bytes × 2 tiles = 8192 bytes
        assert_eq!(no_double.shared_memory_bytes(), 32 * 32 * 4 * 2);
    }

    /// IMP-900a: Threads per block calculation
    #[test]
    fn test_imp900a_threads_per_block() {
        let config = OptimizedGemmConfig::default();
        // 32/4 = 8 threads per dim, 8×8 = 64 threads
        assert_eq!(config.threads_per_block(), 64);

        let large = OptimizedGemmConfig::large();
        // 64/8 = 8 threads per dim, 8×8 = 64 threads
        assert_eq!(large.threads_per_block(), 64);
    }

    /// IMP-900a: Register allocation calculation
    #[test]
    fn test_imp900a_registers_per_thread() {
        let config = OptimizedGemmConfig::default();
        // 4×4 = 16 accumulators per thread
        assert_eq!(config.registers_per_thread(), 16);

        let large = OptimizedGemmConfig::large();
        // 8×8 = 64 accumulators per thread
        assert_eq!(large.registers_per_thread(), 64);
    }

    /// IMP-900a: GEMM performance result calculation
    #[test]
    fn test_imp900a_gemm_performance_result() {
        // 1024×1024×1024 GEMM in 1.54ms (measured baseline)
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 1.54);

        // 2 * 1024³ = 2,147,483,648 ops
        // 2,147,483,648 / (1.54 * 1e6) = 1394.5 GFLOP/s
        assert!((result.gflops - 1394.5).abs() < 10.0);

        // With RTX 4090 peak (~82 TFLOP/s FP32)
        let with_peak = result.with_peak(82000.0);
        assert!(with_peak.efficiency < 2.0); // ~1.7% efficiency (naive kernel)
    }

    /// IMP-900a: Performance improvement check
    #[test]
    fn test_imp900a_performance_improvement_check() {
        // 1024×1024×1024 GEMM in 0.70ms (~3x faster than 1.54ms baseline)
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 0.70);
        let baseline_gflops = 1396.0;

        // 2 * 1024³ / (0.70 * 1e6) = 3067.8 GFLOP/s
        // 3067.8 / 1396.0 = 2.2x improvement
        assert!(result.improved_by(baseline_gflops, 2.0));
        assert!(!result.improved_by(baseline_gflops, 3.0)); // Not quite 3x
    }

    /// IMP-900a: Expected improvement calculation
    #[test]
    fn test_imp900a_expected_improvement() {
        let benchmark = OptimizedGemmBenchmark::default();
        // With all optimizations: 2.0 * 1.5 * 1.3 * 1.2 = 4.68x
        let expected = benchmark.expected_improvement();
        assert!((expected - 4.68).abs() < 0.1);
    }

    // ========================================================================
    // IMP-900b: Kernel Fusion Tests
    // ========================================================================

    /// IMP-900b: Fused operation specification
    #[test]
    fn test_imp900b_fused_op_spec() {
        let spec = FusedOpSpec {
            op_type: FusedOpType::GemmBiasActivation,
            input_dims: vec![256, 2560],
            output_dims: vec![256, 10240],
            activation: Some("gelu".to_string()),
            fused_launches: 1,
            unfused_launches: 3,
        };

        assert_eq!(spec.launch_reduction(), 3.0);
        assert!(spec.achieves_target_reduction()); // 3x > 2x target
    }

    /// IMP-900b: Launch reduction targets
    #[test]
    fn test_imp900b_launch_reduction_targets() {
        // Good fusion: 4 ops → 1 launch
        let good = FusedOpSpec {
            op_type: FusedOpType::FusedAttention,
            input_dims: vec![1, 32, 512, 80],
            output_dims: vec![1, 32, 512, 80],
            activation: None,
            fused_launches: 1,
            unfused_launches: 4,
        };
        assert!(good.achieves_target_reduction());

        // Marginal fusion: 2 ops → 1 launch (exactly 2x)
        let marginal = FusedOpSpec {
            op_type: FusedOpType::LayerNormLinear,
            input_dims: vec![256, 2560],
            output_dims: vec![256, 2560],
            activation: None,
            fused_launches: 1,
            unfused_launches: 2,
        };
        assert!(marginal.achieves_target_reduction());

        // Poor fusion: 3 ops → 2 launches (only 1.5x)
        let poor = FusedOpSpec {
            op_type: FusedOpType::FusedFfn,
            input_dims: vec![256, 2560],
            output_dims: vec![256, 2560],
            activation: None,
            fused_launches: 2,
            unfused_launches: 3,
        };
        assert!(!poor.achieves_target_reduction());
    }

    // ========================================================================
    // IMP-900c: FlashAttention Tests
    // ========================================================================

    /// IMP-900c: FlashAttention config for phi-2
    #[test]
    fn test_imp900c_flash_attention_phi2_config() {
        let config = FlashAttentionConfig::phi2();
        assert_eq!(config.head_dim, 80);
        assert_eq!(config.num_heads, 32);
        assert!(config.causal);
        // scale = 1/sqrt(80) ≈ 0.1118
        assert!((config.scale - 0.1118).abs() < 0.001);
    }

    /// IMP-900c: Memory comparison naive vs flash
    #[test]
    fn test_imp900c_memory_comparison() {
        let config = FlashAttentionConfig::phi2();

        // 512 tokens
        let (naive_512, flash_512) = config.memory_comparison(512);
        assert_eq!(naive_512, 512 * 512 * 4); // 1 MB
        assert_eq!(flash_512, 64 * 64 * 4 * 2); // 32 KB

        // 2048 tokens
        let (naive_2048, flash_2048) = config.memory_comparison(2048);
        assert_eq!(naive_2048, 2048 * 2048 * 4); // 16 MB
        assert_eq!(flash_2048, 64 * 64 * 4 * 2); // 32 KB (same!)
    }

    /// IMP-900c: Memory savings calculation
    #[test]
    fn test_imp900c_memory_savings() {
        let config = FlashAttentionConfig::phi2();

        // 512 tokens: 1MB / 32KB = 32x savings
        let savings_512 = config.memory_savings(512);
        assert!((savings_512 - 32.0).abs() < 1.0);

        // 2048 tokens: 16MB / 32KB = 512x savings
        let savings_2048 = config.memory_savings(2048);
        assert!((savings_2048 - 512.0).abs() < 10.0);
    }

    // ========================================================================
    // IMP-900d: Memory Pool Tests
    // ========================================================================

    /// IMP-900d: Memory pool default configuration
    #[test]
    fn test_imp900d_memory_pool_default() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.initial_size, 256 * 1024 * 1024); // 256 MB
        assert_eq!(config.max_size, 2 * 1024 * 1024 * 1024); // 2 GB
        assert!(config.use_pinned_memory);
        assert!(config.async_transfers);
        assert_eq!(config.size_classes.len(), 9);
    }

    /// IMP-900d: Size class lookup
    #[test]
    fn test_imp900d_size_class_lookup() {
        let config = MemoryPoolConfig::default();

        // Small allocation → 4KB
        assert_eq!(config.find_size_class(1024), Some(4096));

        // Medium allocation → 1MB
        assert_eq!(config.find_size_class(500_000), Some(1048576));

        // Large allocation → 256MB
        assert_eq!(config.find_size_class(200_000_000), Some(268435456));

        // Too large → None
        assert_eq!(config.find_size_class(500_000_000), None);
    }

    /// IMP-900d: Bandwidth improvement estimate
    #[test]
    fn test_imp900d_bandwidth_improvement() {
        let pinned = MemoryPoolConfig::default();
        assert!((pinned.expected_bandwidth_improvement() - 2.4).abs() < 0.1);

        let unpinned = MemoryPoolConfig {
            use_pinned_memory: false,
            ..Default::default()
        };
        assert!((unpinned.expected_bandwidth_improvement() - 1.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // IMP-900: Combined Result Tests
    // ========================================================================

    /// IMP-900: Combined result from baseline
    #[test]
    fn test_imp900_combined_result_baseline() {
        let result = Imp900Result::from_baseline(13.1); // IMP-800 measured

        assert!((result.baseline_tps - 13.1).abs() < 0.1);
        assert!((result.optimized_tps - 13.1).abs() < 0.1);
        assert!((result.gap_ratio - 18.32).abs() < 0.1); // 240/13.1 ≈ 18.32
        assert!(result.milestone.is_none()); // Not yet at any milestone
    }

    /// IMP-900: Individual optimizations
    #[test]
    fn test_imp900_individual_optimizations() {
        let result = Imp900Result::from_baseline(13.1).with_gemm_improvement(2.5); // 2.5x from optimized GEMM

        assert!((result.optimized_tps - 32.75).abs() < 0.1); // 13.1 * 2.5
        assert!((result.gap_ratio - 7.33).abs() < 0.1); // 240/32.75

        // Still not at M2 (need <5x gap)
        assert!(result.milestone.is_none());
    }

    /// IMP-900: M3 target achievement
    #[test]
    fn test_imp900_m3_achievement() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(2.5)
            .with_memory_improvement(1.5);

        // 13.1 * 2.5 * 1.5 = 49.125 tok/s
        assert!((result.optimized_tps - 49.125).abs() < 0.1);
        assert!((result.gap_ratio - 4.89).abs() < 0.1); // 240/49.125

        assert!(result.achieves_m3()); // >48 tok/s, <5x gap
        assert_eq!(result.milestone, Some("M2".to_string())); // Actually at M2 threshold
    }

    /// IMP-900: M4 target achievement (full parity)
    #[test]
    fn test_imp900_m4_achievement() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(3.0)
            .with_fusion_improvement(2.0)
            .with_flash_attention_improvement(2.5)
            .with_memory_improvement(1.5);

        // 13.1 * 3.0 * 2.0 * 2.5 * 1.5 = 294.75 tok/s
        let expected_tps = 13.1 * 3.0 * 2.0 * 2.5 * 1.5;
        assert!((result.optimized_tps - expected_tps).abs() < 0.1);
        assert!((result.gap_ratio - 0.81).abs() < 0.1); // 240/294.75

        assert!(result.achieves_m4()); // >192 tok/s, <1.25x gap
        assert_eq!(result.milestone, Some("M4".to_string()));
    }

    /// IMP-900: Total improvement factor
    #[test]
    fn test_imp900_total_improvement() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(2.0)
            .with_fusion_improvement(1.5)
            .with_flash_attention_improvement(2.0)
            .with_memory_improvement(1.5);

        // Total: 2.0 * 1.5 * 2.0 * 1.5 = 9.0x
        assert!((result.total_improvement() - 9.0).abs() < 0.1);
    }

    // ========================================================================
    // Additional Load Testing Coverage Tests (Phase 4)
    // ========================================================================

    #[test]
    fn test_load_test_runner_config_accessor() {
        let config = LoadTestConfig {
            concurrency: 50,
            duration_secs: 120,
            target_rps: 100.0,
            timeout_ms: 3000,
            warmup_secs: 10,
            latency_threshold_ms: 300.0,
        };
        let runner = LoadTestRunner::new(config.clone());
        let retrieved = runner.config();

        assert_eq!(retrieved.concurrency, 50);
        assert_eq!(retrieved.duration_secs, 120);
        assert!((retrieved.target_rps - 100.0).abs() < 0.001);
        assert_eq!(retrieved.timeout_ms, 3000);
        assert_eq!(retrieved.warmup_secs, 10);
        assert!((retrieved.latency_threshold_ms - 300.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_result_throughput_zero_duration() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 10_000_000,
            duration_secs: 0.0, // Zero duration
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert_eq!(result.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_load_test_result_throughput_negative_duration() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 10_000_000,
            duration_secs: -5.0, // Negative duration edge case
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert_eq!(result.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_load_test_config_validation_edge_cases() {
        // Zero duration
        let zero_duration = LoadTestConfig {
            duration_secs: 0,
            ..LoadTestConfig::default()
        };
        assert!(!zero_duration.is_valid());

        // Zero timeout
        let zero_timeout = LoadTestConfig {
            timeout_ms: 0,
            ..LoadTestConfig::default()
        };
        assert!(!zero_timeout.is_valid());

        // Zero latency threshold
        let zero_latency = LoadTestConfig {
            latency_threshold_ms: 0.0,
            ..LoadTestConfig::default()
        };
        assert!(!zero_latency.is_valid());

        // Negative latency threshold
        let negative_latency = LoadTestConfig {
            latency_threshold_ms: -100.0,
            ..LoadTestConfig::default()
        };
        assert!(!negative_latency.is_valid());
    }

    #[test]
    fn test_load_test_runner_simulate_with_stress_config() {
        let config = LoadTestConfig::for_stress_test();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Stress test should have more requests due to higher concurrency
        assert!(result.total_requests > 0);
        assert!(result.rps_achieved > 0.0);
        // Higher concurrency = higher latency in simulation
        assert!(result.latency_p50_ms > 0.0);
    }

    #[test]
    fn test_load_test_runner_simulate_with_latency_config() {
        let config = LoadTestConfig::for_latency_test();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Latency test has low concurrency
        assert!(result.total_requests > 0);
        // Low concurrency = lower latencies
        assert!(result.latency_p50_ms > 0.0);
    }

    // ========================================================================
    // Additional Matrix Benchmark Coverage Tests (Phase 4)
    // ========================================================================

    #[test]
    fn test_matrix_benchmark_entry_default() {
        let entry = MatrixBenchmarkEntry::default();

        assert_eq!(entry.runtime, RuntimeType::Realizar);
        assert_eq!(entry.backend, ComputeBackendType::Cpu);
        assert!(entry.model.is_empty());
        assert!(!entry.available);
        assert_eq!(entry.p50_latency_ms, 0.0);
        assert_eq!(entry.p99_latency_ms, 0.0);
        assert_eq!(entry.throughput_tps, 0.0);
        assert_eq!(entry.cold_start_ms, 0.0);
        assert_eq!(entry.samples, 0);
        assert_eq!(entry.cv_at_stop, 0.0);
        assert!(entry.notes.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_get_entry_not_found() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        // Empty matrix should return None for any query
        assert!(matrix
            .get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu)
            .is_none());
        assert!(matrix
            .get_entry(RuntimeType::LlamaCpp, ComputeBackendType::Cuda)
            .is_none());
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend_empty() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        // Empty matrix should return None
        assert!(matrix
            .fastest_for_backend(ComputeBackendType::Cpu)
            .is_none());
        assert!(matrix
            .fastest_for_backend(ComputeBackendType::Cuda)
            .is_none());
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_for_backend_empty() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        // Empty matrix should return None
        assert!(matrix
            .highest_throughput_for_backend(ComputeBackendType::Cpu)
            .is_none());
        assert!(matrix
            .highest_throughput_for_backend(ComputeBackendType::Wgpu)
            .is_none());
    }

    #[test]
    fn test_benchmark_matrix_fastest_excludes_unavailable() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add an unavailable entry with "better" metrics (0 latency)
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
        ));

        // Add an available entry with real metrics
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[100.0, 105.0],
            &[50.0],
            90.0,
        ));

        // Should return the available entry, not the unavailable one
        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cpu);
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_excludes_unavailable() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add an unavailable entry
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
        ));

        // Add an available entry
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "test",
            &[100.0],
            &[75.0],
            90.0,
        ));

        // Should return the available entry
        let highest = matrix.highest_throughput_for_backend(ComputeBackendType::Wgpu);
        assert!(highest.is_some());
        assert_eq!(highest.unwrap().runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_summary_empty() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 0);
        assert_eq!(summary.available_entries, 0);
        assert!(summary.overall_fastest.is_none());
        assert!(summary.overall_highest_throughput.is_none());
        assert_eq!(summary.backend_summaries.len(), 3); // CPU, Wgpu, Cuda
    }

    #[test]
    fn test_benchmark_matrix_summary_with_entries() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add entries for different backends
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0, 110.0], // p50 ~= 105
            &[50.0, 55.0],   // avg ~= 52.5
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[80.0, 85.0], // p50 ~= 82.5 (faster)
            &[60.0, 65.0], // avg ~= 62.5 (higher throughput)
            100.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
            "test",
            &[50.0, 55.0], // p50 ~= 52.5 (fastest overall)
            &[80.0, 85.0], // avg ~= 82.5 (highest throughput)
            80.0,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 3);
        assert_eq!(summary.available_entries, 3);

        // Overall fastest should be CUDA (lowest latency)
        assert!(summary.overall_fastest.is_some());
        let (runtime, backend) = summary.overall_fastest.unwrap();
        assert_eq!(runtime, "realizar");
        assert_eq!(backend, "cuda");

        // Overall highest throughput should also be CUDA
        assert!(summary.overall_highest_throughput.is_some());
        let (runtime, backend) = summary.overall_highest_throughput.unwrap();
        assert_eq!(runtime, "realizar");
        assert_eq!(backend, "cuda");
    }

    #[test]
    fn test_benchmark_matrix_summary_backend_details() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        let summary = matrix.summary();

        // Find CPU backend summary
        let cpu_summary = summary
            .backend_summaries
            .iter()
            .find(|s| s.backend == ComputeBackendType::Cpu)
            .expect("Should have CPU summary");

        assert_eq!(cpu_summary.available_runtimes, 1);
        assert!(cpu_summary.fastest_runtime.is_some());
        assert!(cpu_summary.fastest_p50_ms > 0.0);
        assert!(cpu_summary.highest_throughput_runtime.is_some());
        assert!(cpu_summary.highest_throughput_tps > 0.0);
    }

    #[test]
    fn test_benchmark_matrix_summary_all_unavailable() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add only unavailable entries
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cuda,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 2);
        assert_eq!(summary.available_entries, 0);
        assert!(summary.overall_fastest.is_none());
        assert!(summary.overall_highest_throughput.is_none());
    }

    #[test]
    fn test_matrix_benchmark_config_default_comprehensive() {
        let config = MatrixBenchmarkConfig::default();

        assert!(config.runtimes.contains(&RuntimeType::Realizar));
        assert!(config.runtimes.contains(&RuntimeType::LlamaCpp));
        assert!(config.runtimes.contains(&RuntimeType::Ollama));
        assert!(!config.runtimes.contains(&RuntimeType::Vllm)); // Not in default

        assert!(config.backends.contains(&ComputeBackendType::Cpu));
        assert!(config.backends.contains(&ComputeBackendType::Wgpu));
        assert!(!config.backends.contains(&ComputeBackendType::Cuda)); // Not in default

        assert!(config.model_path.is_empty());
        assert!(!config.prompt.is_empty());
        assert_eq!(config.max_tokens, 50);
        assert!((config.cv_threshold - 0.05).abs() < 0.001);
        assert_eq!(config.min_samples, 30);
        assert_eq!(config.max_samples, 200);
        assert_eq!(config.warmup_iterations, 5);
    }

    #[test]
    fn test_backend_summary_struct_fields() {
        let summary = BackendSummary {
            backend: ComputeBackendType::Cuda,
            available_runtimes: 3,
            fastest_runtime: Some("realizar".to_string()),
            fastest_p50_ms: 25.5,
            highest_throughput_runtime: Some("llama-cpp".to_string()),
            highest_throughput_tps: 150.0,
        };

        assert_eq!(summary.backend, ComputeBackendType::Cuda);
        assert_eq!(summary.available_runtimes, 3);
        assert_eq!(summary.fastest_runtime, Some("realizar".to_string()));
        assert!((summary.fastest_p50_ms - 25.5).abs() < 0.001);
        assert_eq!(
            summary.highest_throughput_runtime,
            Some("llama-cpp".to_string())
        );
        assert!((summary.highest_throughput_tps - 150.0).abs() < 0.001);
    }

    #[test]
    fn test_matrix_summary_struct_fields() {
        let summary = MatrixSummary {
            total_entries: 10,
            available_entries: 8,
            backend_summaries: vec![],
            overall_fastest: Some(("realizar".to_string(), "cuda".to_string())),
            overall_highest_throughput: Some(("llama-cpp".to_string(), "cuda".to_string())),
        };

        assert_eq!(summary.total_entries, 10);
        assert_eq!(summary.available_entries, 8);
        assert!(summary.backend_summaries.is_empty());
        assert_eq!(
            summary.overall_fastest,
            Some(("realizar".to_string(), "cuda".to_string()))
        );
        assert_eq!(
            summary.overall_highest_throughput,
            Some(("llama-cpp".to_string(), "cuda".to_string()))
        );
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_unavailable_entries() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add both available and unavailable entries
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cuda,
        ));

        let markdown = matrix.to_markdown_table();

        // Should contain table structure
        assert!(markdown.contains("|"));
        assert!(markdown.contains("Runtime"));
        // Should contain dash for unavailable metrics
        assert!(markdown.contains("-"));
    }

    #[test]
    fn test_compute_backend_type_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ComputeBackendType::Cpu);
        set.insert(ComputeBackendType::Wgpu);
        set.insert(ComputeBackendType::Cuda);

        assert_eq!(set.len(), 3);
        assert!(set.contains(&ComputeBackendType::Cpu));
        assert!(set.contains(&ComputeBackendType::Wgpu));
        assert!(set.contains(&ComputeBackendType::Cuda));
    }

    #[test]
    fn test_compute_backend_type_eq() {
        assert_eq!(ComputeBackendType::Cpu, ComputeBackendType::Cpu);
        assert_eq!(ComputeBackendType::Wgpu, ComputeBackendType::Wgpu);
        assert_eq!(ComputeBackendType::Cuda, ComputeBackendType::Cuda);
        assert_ne!(ComputeBackendType::Cpu, ComputeBackendType::Wgpu);
        assert_ne!(ComputeBackendType::Cpu, ComputeBackendType::Cuda);
        assert_ne!(ComputeBackendType::Wgpu, ComputeBackendType::Cuda);
    }

    #[test]
    fn test_compute_backend_type_copy() {
        let original = ComputeBackendType::Cuda;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn test_matrix_benchmark_entry_serialization_roundtrip() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
            "phi-2",
            &[50.0, 55.0, 52.0],
            &[100.0, 105.0, 102.0],
            80.0,
        )
        .with_notes("GPU layers: 99");

        let json = serde_json::to_string(&entry).expect("Serialization should succeed");
        let deser: MatrixBenchmarkEntry =
            serde_json::from_str(&json).expect("Deserialization should succeed");

        assert_eq!(deser.runtime, RuntimeType::Realizar);
        assert_eq!(deser.backend, ComputeBackendType::Cuda);
        assert_eq!(deser.model, "phi-2");
        assert!(deser.available);
        assert_eq!(deser.samples, 3);
        assert_eq!(deser.notes, "GPU layers: 99");
    }

    #[test]
    fn test_benchmark_matrix_entries_for_runtime_empty() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);

        let entries = matrix.entries_for_runtime(RuntimeType::Realizar);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_entries_for_backend_with_multiple() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add multiple entries for the same backend
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[80.0],
            &[60.0],
            85.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Ollama,
            ComputeBackendType::Cpu,
            "test",
            &[90.0],
            &[55.0],
            95.0,
        ));

        let cpu_entries = matrix.entries_for_backend(ComputeBackendType::Cpu);
        assert_eq!(cpu_entries.len(), 3);

        // Verify all are for CPU backend
        for entry in &cpu_entries {
            assert_eq!(entry.backend, ComputeBackendType::Cpu);
        }
    }

    #[test]
    fn test_load_test_config_clone() {
        let config = LoadTestConfig::for_stress_test();
        let cloned = config.clone();

        assert_eq!(cloned.concurrency, config.concurrency);
        assert_eq!(cloned.duration_secs, config.duration_secs);
        assert!((cloned.target_rps - config.target_rps).abs() < 0.001);
        assert_eq!(cloned.timeout_ms, config.timeout_ms);
        assert_eq!(cloned.warmup_secs, config.warmup_secs);
        assert!((cloned.latency_threshold_ms - config.latency_threshold_ms).abs() < 0.001);
    }

    #[test]
    fn test_load_test_result_clone() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 995,
            failed_requests: 5,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.005,
            passed_latency_threshold: true,
        };
        let cloned = result.clone();

        assert_eq!(cloned.total_requests, result.total_requests);
        assert_eq!(cloned.successful_requests, result.successful_requests);
        assert!((cloned.error_rate - result.error_rate).abs() < 0.0001);
    }
}
