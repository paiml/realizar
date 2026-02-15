//! Part 4 of bench module tests - Statistical Tools, Regression Detection, HTTP Backends
//!
//! Focus areas:
//! - BENCH-006: OutlierDetector (MAD-based)
//! - BENCH-007: RegressionDetector
//! - BENCH-008: Welch's t-test
//! - BENCH-009: ThermalGuard (TDD RED)
//! - BENCH-010: KL-Divergence Quality Validation
//! - OllamaBackend HTTP Integration
//! - Distributed Benchmark Suite
//! - Load Testing
//! - LlamaCppBackend CLI Output Parsing
//! - Benchmark Matrix (EXTREME TDD)
//! - QA Checklist Validation Tests (QA-031 to QA-040)

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    use crate::bench::*;

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
}
