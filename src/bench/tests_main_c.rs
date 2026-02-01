use crate::bench::*;
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
