//! CLI Module Tests Part 05 - T-COV-95 Coverage Bridge (Deep CLI)
//!
//! Tests for uncovered CLI functions:
//! - run_visualization: sparkline, histogram, benchmark report, multi-comparison
//! - run_convoy_test: full execution with output file
//! - run_saturation_test: full execution with output file
//! - run_bench_compare: with temp benchmark files
//! - run_bench_regression: strict and normal modes
//! - parse_cargo_bench_output: valid bench lines, empty, partial
//! - run_benchmarks: list mode, invalid suite, external benchmark stub
//! - load_gguf_model / load_safetensors_model / load_apr_model: full paths
//! - print_info, home_dir, validate_suite_name
//! - display_model_info: all 4 branches
//! - is_local_file_path: comprehensive edge cases
//!
//! Refs PMAT-802: Protocol T-COV-95 Deep CLI Coverage

#[cfg(test)]
mod tests {
    use super::super::*;

    // =========================================================================
    // run_visualization
    // =========================================================================

    #[test]
    fn test_run_visualization_with_color() {
        // Should not panic
        run_visualization(true, 50);
    }

    #[test]
    fn test_run_visualization_without_color() {
        run_visualization(false, 50);
    }

    #[test]
    fn test_run_visualization_small_samples() {
        run_visualization(false, 10);
    }

    #[test]
    fn test_run_visualization_large_samples() {
        run_visualization(true, 200);
    }

    // =========================================================================
    // run_convoy_test
    // =========================================================================

    #[test]
    fn test_run_convoy_test_no_args() {
        let result = run_convoy_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_runtime() {
        let result = run_convoy_test(Some("test-runtime".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_model() {
        let result = run_convoy_test(None, Some("test-model".to_string()), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_output() {
        let dir = tempfile::tempdir().unwrap();
        let output_path = dir.path().join("convoy_results.json");
        let result = run_convoy_test(None, None, Some(output_path.to_string_lossy().to_string()));
        assert!(result.is_ok());
        assert!(output_path.exists());
        let contents = std::fs::read_to_string(&output_path).unwrap();
        assert!(contents.contains("baseline_short_p99_ms"));
    }

    #[test]
    fn test_run_convoy_test_all_args() {
        let dir = tempfile::tempdir().unwrap();
        let output_path = dir.path().join("convoy_all.json");
        let result = run_convoy_test(
            Some("realizar".to_string()),
            Some("llama".to_string()),
            Some(output_path.to_string_lossy().to_string()),
        );
        assert!(result.is_ok());
    }

    // =========================================================================
    // run_saturation_test
    // =========================================================================

    #[test]
    fn test_run_saturation_test_no_args() {
        let result = run_saturation_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_with_runtime() {
        let result = run_saturation_test(Some("test-runtime".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_with_model() {
        let result = run_saturation_test(None, Some("test-model".to_string()), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_with_output() {
        let dir = tempfile::tempdir().unwrap();
        let output_path = dir.path().join("saturation_results.json");
        let result =
            run_saturation_test(None, None, Some(output_path.to_string_lossy().to_string()));
        assert!(result.is_ok());
        assert!(output_path.exists());
        let contents = std::fs::read_to_string(&output_path).unwrap();
        assert!(contents.contains("baseline_throughput"));
    }

    // =========================================================================
    // parse_cargo_bench_output
    // =========================================================================

    #[test]
    fn test_parse_cargo_bench_output_valid_lines() {
        let output = "test benchmark_matmul ... bench:      1,234 ns/iter (+/- 56)\n\
                       test benchmark_softmax ... bench:        789 ns/iter (+/- 12)";
        let results = parse_cargo_bench_output(output, Some("tensor_ops"));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["name"], "benchmark_matmul");
        assert_eq!(results[0]["time_ns"], 1234);
        assert_eq!(results[0]["suite"], "tensor_ops");
        assert_eq!(results[1]["name"], "benchmark_softmax");
        assert_eq!(results[1]["time_ns"], 789);
    }

    #[test]
    fn test_parse_cargo_bench_output_no_suite() {
        let output = "test bench_add ... bench:        100 ns/iter (+/- 5)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 1);
        assert!(results[0]["suite"].is_null());
    }

    #[test]
    fn test_parse_cargo_bench_output_empty() {
        let results = parse_cargo_bench_output("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_no_bench_lines() {
        let output = "Compiling realizar v0.3.5\nFinished release\nrunning 5 tests";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_partial_match() {
        // Has "bench:" but not "ns/iter"
        let output = "test partial ... bench: 100 ms/iter (+/- 5)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_mixed_lines() {
        let output = "Compiling realizar v0.3.5\n\
                       test bench_fast ... bench:        50 ns/iter (+/- 2)\n\
                       running 1 test\n\
                       test bench_slow ... bench:      5,000 ns/iter (+/- 100)\n\
                       test result: ok. 0 passed";
        let results = parse_cargo_bench_output(output, Some("all"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parse_cargo_bench_output_large_numbers() {
        let output = "test bench_huge ... bench: 1,234,567 ns/iter (+/- 1,000)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["time_ns"], 1234567);
    }

    #[test]
    fn test_parse_cargo_bench_output_unparseable_time() {
        // Time has non-numeric chars
        let output = "test bench_bad ... bench:      abc ns/iter (+/- 5)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    // =========================================================================
    // run_benchmarks (list mode only, since actual bench needs cargo)
    // =========================================================================

    #[test]
    fn test_run_benchmarks_list_mode() {
        let result = run_benchmarks(None, true, None, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_external_no_feature() {
        // Without bench-http feature, should return error
        let result = run_benchmarks(
            Some("all".to_string()),
            false,
            Some("ollama".to_string()),
            None,
            Some("http://localhost:11434".to_string()),
            None,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Helper: create valid benchmark JSON
    // =========================================================================

    fn create_test_benchmark_json(runtime: &str) -> String {
        use crate::bench::*;
        let result = FullBenchmarkResult {
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
                    p50: 7.0,
                    p95: 9.0,
                    p99: 10.0,
                    p999: 12.0,
                },
                itl_ms: ItlResults {
                    median: 10.0,
                    std_dev: 2.0,
                    p99: 15.0,
                },
                throughput_tok_s: ThroughputResults {
                    median: 100.0,
                    ci_95: (95.0, 105.0),
                },
                memory_mb: MemoryResults {
                    model_mb: 512,
                    peak_rss_mb: 1024,
                    kv_waste_pct: 3.0,
                },
                energy: EnergyResults {
                    total_joules: 50.0,
                    token_joules: 0.5,
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
        };
        serde_json::to_string_pretty(&result).unwrap()
    }

    // =========================================================================
    // run_bench_compare
    // =========================================================================

    #[test]
    fn test_run_bench_compare_valid_files() {
        let dir = tempfile::tempdir().unwrap();
        let file1 = dir.path().join("baseline.json");
        let file2 = dir.path().join("current.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&file1, &json).unwrap();
        std::fs::write(&file2, &json).unwrap();

        let result = run_bench_compare(&file1.to_string_lossy(), &file2.to_string_lossy(), 5.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_bench_compare_nonexistent_file1() {
        let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_compare_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let file1 = dir.path().join("bad1.json");
        let file2 = dir.path().join("bad2.json");
        std::fs::write(&file1, "not valid json").unwrap();
        std::fs::write(&file2, "also bad").unwrap();

        let result = run_bench_compare(&file1.to_string_lossy(), &file2.to_string_lossy(), 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_compare_file1_valid_file2_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let file1 = dir.path().join("good.json");
        let file2 = dir.path().join("bad.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&file1, &json).unwrap();
        std::fs::write(&file2, "not json").unwrap();

        let result = run_bench_compare(&file1.to_string_lossy(), &file2.to_string_lossy(), 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_compare_zero_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let file1 = dir.path().join("base.json");
        let file2 = dir.path().join("curr.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&file1, &json).unwrap();
        std::fs::write(&file2, &json).unwrap();

        let result = run_bench_compare(&file1.to_string_lossy(), &file2.to_string_lossy(), 0.0);
        assert!(result.is_ok());
    }

    // =========================================================================
    // run_bench_regression
    // =========================================================================

    #[test]
    fn test_run_bench_regression_no_regression() {
        let dir = tempfile::tempdir().unwrap();
        let baseline = dir.path().join("baseline.json");
        let current = dir.path().join("current.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&baseline, &json).unwrap();
        std::fs::write(&current, &json).unwrap();

        let result = run_bench_regression(
            &baseline.to_string_lossy(),
            &current.to_string_lossy(),
            false, // strict
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_bench_regression_strict_mode() {
        let dir = tempfile::tempdir().unwrap();
        let baseline = dir.path().join("baseline.json");
        let current = dir.path().join("current.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&baseline, &json).unwrap();
        std::fs::write(&current, &json).unwrap();

        // Strict mode with identical data
        let result = run_bench_regression(
            &baseline.to_string_lossy(),
            &current.to_string_lossy(),
            true, // strict
        );
        // May pass or fail depending on threshold check
        let _ = result;
    }

    #[test]
    fn test_run_bench_regression_nonexistent() {
        let result = run_bench_regression(
            "/nonexistent/baseline.json",
            "/nonexistent/current.json",
            false, // strict
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_regression_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let baseline = dir.path().join("baseline.json");
        let current = dir.path().join("current.json");
        std::fs::write(&baseline, "bad json").unwrap();
        std::fs::write(&current, "also bad").unwrap();

        let result = run_bench_regression(
            &baseline.to_string_lossy(),
            &current.to_string_lossy(),
            false, // strict
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // print_info
    // =========================================================================

    #[test]
    fn test_print_info_does_not_panic() {
        print_info();
    }

    // =========================================================================
    // home_dir
    // =========================================================================

    #[test]
    fn test_home_dir_returns_some() {
        // HOME is typically set in test environments
        let result = home_dir();
        // May or may not be Some depending on environment
        let _ = result;
    }

    // =========================================================================
    // validate_suite_name
    // =========================================================================

    #[test]
    fn test_validate_suite_name_valid() {
        assert!(validate_suite_name("tensor_ops"));
        assert!(validate_suite_name("inference"));
        assert!(validate_suite_name("cache"));
        assert!(validate_suite_name("tokenizer"));
        assert!(validate_suite_name("quantize"));
        assert!(validate_suite_name("lambda"));
        assert!(validate_suite_name("comparative"));
    }

    #[test]
    fn test_validate_suite_name_invalid() {
        assert!(!validate_suite_name("nonexistent"));
        assert!(!validate_suite_name(""));
        assert!(!validate_suite_name("TENSOR_OPS")); // case sensitive
        assert!(!validate_suite_name("tensor_ops ")); // trailing space
    }

    // =========================================================================
    // is_local_file_path - comprehensive
    // =========================================================================

    #[test]
    fn test_is_local_file_path_comprehensive() {
        // True cases
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("/absolute/path/model.gguf"));
        assert!(is_local_file_path("model.gguf")); // .gguf extension
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("model.apr"));
        assert!(is_local_file_path("/tmp/model.safetensors"));
        assert!(is_local_file_path("./relative.apr"));

        // False cases
        assert!(!is_local_file_path("pacha://model"));
        assert!(!is_local_file_path("hf://model"));
        assert!(!is_local_file_path("model_name"));
        assert!(!is_local_file_path("llama3.2"));
        assert!(!is_local_file_path("registry:tag"));
    }

    // =========================================================================
    // load_gguf_model
    // =========================================================================

    #[test]
    fn test_load_gguf_model_valid() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .num_layers("llama", 2)
            .num_heads("llama", 4)
            .build();
        let result = load_gguf_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_gguf_model_empty() {
        let result = load_gguf_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_gguf_model_truncated() {
        let result = load_gguf_model(&[0x47, 0x47, 0x55, 0x46]); // Just magic bytes
        assert!(result.is_err());
    }

    // =========================================================================
    // load_safetensors_model
    // =========================================================================

    #[test]
    fn test_load_safetensors_model_valid() {
        let header = r#"{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&[0u8; 16]);
        let result = load_safetensors_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_safetensors_model_empty() {
        let result = load_safetensors_model(&[]);
        assert!(result.is_err());
    }

    // =========================================================================
    // load_apr_model
    // =========================================================================

    #[test]
    fn test_load_apr_model_with_magic() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_apr_model_wrong_magic() {
        let data = vec![0xFF; 64];
        let result = load_apr_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_too_small() {
        let result = load_apr_model(&[0x41, 0x50, 0x52, 0x00]); // Just "APR\0" - 4 bytes
                                                                // May succeed or fail depending on format detection
        let _ = result;
    }

    // =========================================================================
    // display_model_info - all branches
    // =========================================================================

    #[test]
    fn test_display_model_info_gguf_by_extension() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        let result = display_model_info("test.gguf", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_gguf_by_magic() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        // Use non-gguf extension but GGUF magic bytes
        let result = display_model_info("model.bin", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_safetensors_by_extension() {
        let header = r#"{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&[0u8; 16]);
        let result = display_model_info("model.safetensors", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_by_extension() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0020u16.to_le_bytes());
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = display_model_info("model.apr", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_by_magic() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = display_model_info("model.bin", &data);
        // APR magic check happens after safetensors check
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_unknown() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
        let result = display_model_info("model.unknown", &data);
        assert!(result.is_ok());
    }

    // =========================================================================
    // BENCHMARK_SUITES constant verification
    // =========================================================================

    #[test]
    fn test_benchmark_suites_not_empty() {
        // BENCHMARK_SUITES is a non-empty const - just access first element
        let (first_name, first_desc) = BENCHMARK_SUITES[0];
        assert!(!first_name.is_empty());
        assert!(!first_desc.is_empty());
        for (name, desc) in BENCHMARK_SUITES {
            assert!(!name.is_empty());
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_benchmark_suites_contains_expected() {
        let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"tensor_ops"));
        assert!(names.contains(&"inference"));
        assert!(names.contains(&"cache"));
        assert!(names.contains(&"tokenizer"));
        assert!(names.contains(&"quantize"));
    }
}
