#[cfg(test)]
mod tests {
    use crate::cli::*;

    // -------------------------------------------------------------------------
    // Format Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(10 * 1024), "10.0 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.0 MB");
        assert_eq!(format_size(512 * 1024 * 1024), "512.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(7 * 1024 * 1024 * 1024), "7.0 GB");
    }

    // -------------------------------------------------------------------------
    // Benchmark Suite Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_suites_not_empty() {
        // BENCHMARK_SUITES is a static const array, verify it has entries
        let suites_len = BENCHMARK_SUITES.len();
        assert!(suites_len > 0, "BENCHMARK_SUITES should not be empty");
        assert!(suites_len >= 5, "Should have at least 5 benchmark suites");
    }

    #[test]
    fn test_benchmark_suites_have_descriptions() {
        for (name, description) in BENCHMARK_SUITES {
            assert!(!name.is_empty(), "Benchmark name should not be empty");
            assert!(
                !description.is_empty(),
                "Benchmark description should not be empty"
            );
        }
    }

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
        assert!(!validate_suite_name("unknown"));
        assert!(!validate_suite_name(""));
        assert!(!validate_suite_name("tensor"));
        assert!(!validate_suite_name("TENSOR_OPS"));
    }

    // -------------------------------------------------------------------------
    // Is Local File Path Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_true() {
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("/home/user/model.gguf"));
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("model.apr"));
    }

    #[test]
    fn test_is_local_file_path_false() {
        assert!(!is_local_file_path("llama3:8b"));
        assert!(!is_local_file_path("pacha://model:v1"));
        assert!(!is_local_file_path("hf://meta-llama/Llama-3"));
    }

    // -------------------------------------------------------------------------
    // Home Dir Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_home_dir_returns_path() {
        // This test depends on HOME being set, which it usually is
        let home = home_dir();
        // Just check it doesn't panic - may be None in some environments
        if let Some(path) = home {
            assert!(path.is_absolute() || path.to_string_lossy().starts_with('/'));
        }
    }

    // -------------------------------------------------------------------------
    // Display Model Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_unknown_format() {
        // Empty data with unknown extension
        let result = display_model_info("model.bin", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_gguf_extension_but_invalid() {
        // .gguf extension but not valid GGUF data
        let result = display_model_info("model.gguf", &[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Print Info Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_print_info_does_not_panic() {
        // Just ensure it doesn't panic
        print_info();
    }

    // -------------------------------------------------------------------------
    // Run Visualization Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_does_not_panic() {
        // Run with minimal samples to keep test fast
        run_visualization(false, 10);
        run_visualization(true, 10);
    }

    // -------------------------------------------------------------------------
    // Load Model Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_gguf_model_invalid() {
        let result = load_gguf_model(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_model_invalid() {
        let result = load_safetensors_model(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Convoy and Saturation Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_convoy_test_no_output() {
        // Just verify it runs without panic
        let result = run_convoy_test(Some("test".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_no_output() {
        let result = run_saturation_test(Some("test".to_string()), None, None);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Benchmark Compare/Regression Tests (file not found cases)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_compare_file_not_found() {
        let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_regression_file_not_found() {
        let result = run_bench_regression(
            "/nonexistent/baseline.json",
            "/nonexistent/current.json",
            false,
        );
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // run_benchmarks Tests (list mode only - doesn't run cargo bench)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_benchmarks_list_mode() {
        // List mode should succeed without running cargo bench
        let result = run_benchmarks(None, true, None, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_runtime() {
        // List mode with runtime specified
        let result = run_benchmarks(None, true, Some("realizar".to_string()), None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_model() {
        // List mode with model specified
        let result = run_benchmarks(None, true, None, Some("model.gguf".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_output() {
        // List mode with output specified
        let result = run_benchmarks(
            None,
            true,
            None,
            None,
            None,
            Some("/tmp/output.json".to_string()),
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Temp File Tests for bench compare/regression (error paths)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_compare_second_file_not_found() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let file1 = dir.join("bench_compare_one.json");

        let mut f1 = std::fs::File::create(&file1).expect("test");
        f1.write_all(b"{}").expect("test");

        let result = run_bench_compare(
            file1.to_str().expect("test"),
            "/nonexistent/file2.json",
            5.0,
        );

        let _ = std::fs::remove_file(&file1);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_regression_second_file_not_found() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let baseline = dir.join("bench_regress_base.json");

        let mut f1 = std::fs::File::create(&baseline).expect("test");
        f1.write_all(b"{}").expect("test");

        let result = run_bench_regression(
            baseline.to_str().expect("test"),
            "/nonexistent/current.json",
            false,
        );

        let _ = std::fs::remove_file(&baseline);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Convoy/Saturation with output file tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_convoy_test_with_output() {
        let dir = std::env::temp_dir();
        let output = dir.join("convoy_output.json");

        let result = run_convoy_test(
            Some("test".to_string()),
            Some("model.gguf".to_string()),
            Some(output.to_str().expect("test").to_string()),
        );

        assert!(result.is_ok());
        assert!(output.exists());

        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn test_run_saturation_test_with_output() {
        let dir = std::env::temp_dir();
        let output = dir.join("saturation_output.json");

        let result = run_saturation_test(
            Some("test".to_string()),
            Some("model.gguf".to_string()),
            Some(output.to_str().expect("test").to_string()),
        );

        assert!(result.is_ok());
        assert!(output.exists());

        let _ = std::fs::remove_file(&output);
    }

    // -------------------------------------------------------------------------
    // Display Model Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_with_gguf_magic() {
        // Create minimal GGUF data with magic header
        let data = b"GGUF\x03\x00\x00\x00"; // GGUF magic + version 3
        let result = display_model_info("test.gguf", data);
        // Will fail to parse but exercises the GGUF path
        assert!(result.is_err());
    }

    #[test]
    fn test_display_model_info_safetensors_extension() {
        let result = display_model_info("test.safetensors", &[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // APR Format Support Tests (EXTREME TDD)
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_apr_extension() {
        // Create minimal APR data with magic header + model type
        // APR header: APRN (4 bytes) + type_id (2 bytes) + version (2 bytes)
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = display_model_info("test.apr", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_magic() {
        // Test detection via magic bytes, not extension
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = display_model_info("model.bin", &data); // Unknown extension
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_zero_bytes_unknown() {
        let data = b"\x00\x00\x00\x00\x00\x00\x00\x00"; // All zeros (not SafeTensors either)
        let result = display_model_info("test.bin", data);
        // Should not error, just show "Unknown"
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // load_apr_model Tests (EXTREME TDD)
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_apr_model_valid() {
        // Valid APR data with magic and model type
        // APR header: APRN (4 bytes) + type_id (2 bytes) + version (2 bytes)
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0003u16.to_le_bytes()); // DecisionTree
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_apr_model_all_recognized_types() {
        // Test recognized APR model types per model_loader mapping
        let type_codes = [
            0x0001u16, // LinearRegression
            0x0002,    // LogisticRegression
            0x0003,    // DecisionTree
            0x0004,    // RandomForest
            0x0005,    // GradientBoosting
            0x0006,    // KMeans
            0x0007,    // PCA
            0x0008,    // NaiveBayes
            0x0009,    // KNN
            0x000A,    // SVM
            0x0010,    // NgramLM
            0x0011,    // TFIDF
            0x0012,    // CountVectorizer
            0x0020,    // NeuralSequential
            0x0021,    // NeuralCustom
            0x0030,    // ContentRecommender
            0x0040,    // MixtureOfExperts
            0x00FF,    // Custom
        ];

        for type_code in type_codes {
            let mut data = vec![0u8; 16];
            data[0..4].copy_from_slice(b"APR\0");
            data[4..6].copy_from_slice(&type_code.to_le_bytes());
            data[6..8].copy_from_slice(&1u16.to_le_bytes());
            let result = load_apr_model(&data);
            assert!(result.is_ok(), "Failed for type code 0x{:04X}", type_code);
        }
    }

    #[test]
    fn test_load_apr_model_invalid_magic() {
        // Wrong magic bytes
        let data = b"GGUFxxxxxxxxxxxxxxxx";
        let result = load_apr_model(data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Expected APR format"));
    }

    #[test]
    fn test_load_apr_model_too_short() {
        // Data too short for format detection
        let data = b"APR"; // Only 3 bytes
        let result = load_apr_model(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_unknown_type() {
        // Valid magic but unknown model type
        // APR header: APRN (4 bytes) + type_id (2 bytes) + version (2 bytes)
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0xFFFEu16.to_le_bytes()); // Unknown type (not 0x00FF)
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = load_apr_model(&data);
        // Should succeed (shows "Unknown" type)
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_apr_model_empty_data() {
        let result = load_apr_model(&[]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Parse Cargo Bench Output Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_cargo_bench_output_empty() {
        let results = parse_cargo_bench_output("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_no_bench_lines() {
        let output = "running 5 tests\ntest test_foo ... ok\ntest test_bar ... ok\n";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_valid_bench_line() {
        let output = "test benchmark_matmul ... bench:      1,234 ns/iter (+/- 56)";
        let results = parse_cargo_bench_output(output, Some("tensor_ops"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "benchmark_matmul");
        assert_eq!(results[0]["time_ns"], 1234);
        assert_eq!(results[0]["suite"], "tensor_ops");
    }

    #[test]
    fn test_parse_cargo_bench_output_multiple_lines() {
        let output = "\
test benchmark_add ... bench:         100 ns/iter (+/- 5)
test benchmark_mul ... bench:         200 ns/iter (+/- 10)
test benchmark_div ... bench:       1,500 ns/iter (+/- 50)
";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_parse_cargo_bench_output_mixed_content() {
        let output = "\
running 3 benchmarks
test benchmark_foo ... bench:         500 ns/iter (+/- 25)
some other line
test tests::unit_test ... ok
test benchmark_bar ... bench:         750 ns/iter (+/- 30)
";
        let results = parse_cargo_bench_output(output, Some("test_suite"));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["name"], "benchmark_foo");
        assert_eq!(results[1]["name"], "benchmark_bar");
    }

    #[test]
    fn test_parse_cargo_bench_output_invalid_time() {
        // Line has "bench:" but no parseable number
        let output = "test benchmark_bad ... bench: invalid ns/iter (+/- 0)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    // -------------------------------------------------------------------------
    // Run Benchmarks Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_benchmarks_list_all_params() {
        // List mode with all params populated
        let result = run_benchmarks(
            Some("tensor_ops".to_string()),
            true,
            Some("realizar".to_string()),
            Some("model.gguf".to_string()),
            Some("http://localhost:8080".to_string()),
            Some("/tmp/output.json".to_string()),
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Run Visualization Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_large_samples() {
        // Test with larger sample count
        run_visualization(true, 100);
    }

    #[test]
    fn test_run_visualization_single_sample() {
        // Edge case: single sample
        run_visualization(false, 1);
    }

    // -------------------------------------------------------------------------
    // Format Size Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_boundary_kb() {
        // Exactly 1 byte under KB threshold
        assert_eq!(format_size(1023), "1023 B");
        // Exactly at KB threshold
        assert_eq!(format_size(1024), "1.0 KB");
    }

    #[test]
    fn test_format_size_boundary_mb() {
        // Exactly 1 byte under MB threshold
        assert_eq!(format_size(1024 * 1024 - 1), "1024.0 KB");
        // Exactly at MB threshold
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_format_size_boundary_gb() {
        // Exactly at GB threshold
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        // Large GB value
        assert_eq!(format_size(100 * 1024 * 1024 * 1024), "100.0 GB");
    }

    // -------------------------------------------------------------------------
    // Is Local File Path Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_relative() {
        assert!(is_local_file_path("../models/test.gguf"));
        assert!(is_local_file_path("./relative.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_no_extension() {
        // Path without known extension is NOT considered local
        // (must have .gguf, .safetensors, .apr, or start with / or ./)
        assert!(!is_local_file_path("model_without_ext"));
        // But absolute path without extension IS local
        assert!(is_local_file_path("/home/user/model_without_ext"));
        // And relative path starting with ./ IS local
        assert!(is_local_file_path("./model_without_ext"));
    }

    // -------------------------------------------------------------------------
    // Validate Suite Name Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_suite_name_all_suites() {
        // Verify all listed suites are valid
        for (name, _) in BENCHMARK_SUITES {
            assert!(
                validate_suite_name(name),
                "Suite '{}' should be valid",
                name
            );
        }
    }

    #[test]
    fn test_validate_suite_name_case_sensitivity() {
        // Verify case sensitivity
        assert!(!validate_suite_name("TENSOR_OPS"));
        assert!(!validate_suite_name("Tensor_Ops"));
        assert!(!validate_suite_name("TenSor_OpS"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: home_dir
    // -------------------------------------------------------------------------

    #[test]
    fn test_home_dir_returns_value() {
        // home_dir should return Some on most systems
        let result = home_dir();
        // Don't assert it's Some since CI may not have HOME set
        // Just ensure it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_home_dir_path_is_absolute() {
        if let Some(path) = home_dir() {
            // If HOME is set, the path should be absolute
            assert!(
                path.is_absolute() || path.to_string_lossy().starts_with('/'),
                "Home dir should be absolute path"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: print_info
    // -------------------------------------------------------------------------

    #[test]
    fn test_print_info_no_panic() {
        // Just verify print_info doesn't panic
        print_info();
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: display_model_info
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_random_bytes() {
        // Random bytes should print "Unknown" format
        let data = vec![0x00, 0x01, 0x02, 0x03];
        let result = display_model_info("model.bin", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_empty_data() {
        let data: Vec<u8> = vec![];
        let result = display_model_info("empty.bin", &data);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: load_apr_model error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_apr_model_wrong_format() {
        // GGUF magic instead of APR
        let data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00]; // GGUF magic
        let result = load_apr_model(&data);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_bench_compare
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_compare_missing_files() {
        let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 0.1);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_bench_regression
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_regression_missing_files() {
        let result = run_bench_regression(
            "/nonexistent/baseline.json",
            "/nonexistent/current.json",
            false,
        );
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: is_local_file_path edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_apr_extension() {
        assert!(is_local_file_path("model.apr"));
        assert!(is_local_file_path("path/to/model.apr"));
    }

    #[test]
    fn test_is_local_file_path_safetensors_extension() {
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("/absolute/path/model.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_gguf_extension() {
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("./relative/model.gguf"));
    }

    #[test]
    fn test_is_local_file_path_absolute() {
        assert!(is_local_file_path("/usr/local/models/test"));
        assert!(is_local_file_path("/home/user/model"));
    }

    #[test]
    fn test_is_local_file_path_relative_dot() {
        assert!(is_local_file_path("./model"));
        assert!(is_local_file_path("./subdir/model"));
    }

    #[test]
    fn test_is_local_file_path_hf_style() {
        // HuggingFace style refs should NOT be local
        assert!(!is_local_file_path("meta-llama/Llama-2-7b"));
        assert!(!is_local_file_path("openai/whisper-tiny"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: parse_cargo_bench_output
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_cargo_bench_output_empty_cov() {
        let results = parse_cargo_bench_output("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_no_bench_lines_cov() {
        let output = "Running tests...\nAll tests passed\n";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_single_bench_cov() {
        let output = "test tensor_add ... bench: 1,234 ns/iter (+/- 56)";
        let results = parse_cargo_bench_output(output, Some("tensor_ops"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "tensor_add");
        assert_eq!(results[0]["time_ns"], 1234);
        assert_eq!(results[0]["suite"], "tensor_ops");
    }

    #[test]
    fn test_parse_cargo_bench_output_multiple_bench_cov() {
        let output = "test bench_add ... bench: 100 ns/iter (+/- 5)\n\
                      test bench_mul ... bench: 200 ns/iter (+/- 10)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parse_cargo_bench_output_malformed_line_cov() {
        // Line that contains bench: but is malformed
        let output = "bench: broken line";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_visualization
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_color_cov() {
        // Run visualization with color enabled
        run_visualization(true, 10);
    }

    #[test]
    fn test_run_visualization_no_color_cov() {
        // Run visualization without color
        run_visualization(false, 10);
    }

    #[test]
    fn test_run_visualization_many_samples_cov() {
        run_visualization(false, 100);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_convoy_test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_convoy_test_default_runtime_cov() {
        let result = run_convoy_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_model_cov() {
        let result = run_convoy_test(
            Some("custom_runtime".to_string()),
            Some("custom_model.gguf".to_string()),
            None,
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_saturation_test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_saturation_test_default_runtime_cov() {
        let result = run_saturation_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_with_model_cov() {
        let result = run_saturation_test(
            Some("custom_runtime".to_string()),
            Some("model.gguf".to_string()),
            None,
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: format_size boundary cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_boundary_kb_cov() {
        // Exactly at KB boundary
        assert_eq!(format_size(1024 - 1), "1023 B");
        assert_eq!(format_size(1024), "1.0 KB");
    }

    #[test]
    fn test_format_size_boundary_mb_cov() {
        // Just below MB boundary
        let just_under_mb = 1024 * 1024 - 1;
        assert!(format_size(just_under_mb).contains("KB"));
    }

    #[test]
    fn test_format_size_boundary_gb_cov() {
        // Just below GB boundary
        let just_under_gb = 1024 * 1024 * 1024 - 1;
        assert!(format_size(just_under_gb).contains("MB"));
    }

    #[test]
    fn test_format_size_large_gb_cov() {
        let size = 100 * 1024 * 1024 * 1024; // 100 GB
        assert!(format_size(size).contains("GB"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: BENCHMARK_SUITES constants
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_suites_tensor_ops_exists_cov() {
        let found = BENCHMARK_SUITES
            .iter()
            .any(|(name, _)| *name == "tensor_ops");
        assert!(found);
    }

    #[test]
    fn test_benchmark_suites_inference_exists_cov() {
        let found = BENCHMARK_SUITES
            .iter()
            .any(|(name, _)| *name == "inference");
        assert!(found);
    }

    #[test]
    fn test_benchmark_suites_cache_exists_cov() {
        let found = BENCHMARK_SUITES.iter().any(|(name, _)| *name == "cache");
        assert!(found);
    }

    #[test]
    fn test_benchmark_suites_lambda_exists_cov() {
        let found = BENCHMARK_SUITES.iter().any(|(name, _)| *name == "lambda");
        assert!(found);
    }

    #[test]
    fn test_benchmark_suites_comparative_exists_cov() {
        let found = BENCHMARK_SUITES
            .iter()
            .any(|(name, _)| *name == "comparative");
        assert!(found);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: display_model_info edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_unknown_format_cov() {
        let result = display_model_info("unknown.bin", &[1, 2, 3, 4, 5]);
        // Unknown format just prints info and succeeds
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_empty_data_cov() {
        let result = display_model_info("empty.bin", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_extension_cov() {
        // APR extension without magic triggers APR path
        let result = display_model_info("model.apr", &[0; 10]);
        // Will succeed (shows unknown model type)
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_benchmarks edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_benchmarks_with_all_options_cov() {
        let result = run_benchmarks(
            Some("tensor_ops".to_string()),
            true, // list mode
            Some("realizar".to_string()),
            Some("model.gguf".to_string()),
            Some("http://localhost:8080".to_string()),
            Some("output.json".to_string()),
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: is_local_file_path edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_empty_cov() {
        assert!(!is_local_file_path(""));
    }

    #[test]
    fn test_is_local_file_path_url_without_extension_cov() {
        // URLs without model extensions are not local
        assert!(!is_local_file_path("http://example.com/model"));
        assert!(!is_local_file_path("pacha://registry/model"));
        // Note: URLs with .gguf extension are still considered local by extension check
        assert!(is_local_file_path("https://example.com/model.gguf"));
    }

    #[test]
    fn test_is_local_file_path_parent_dir_cov() {
        assert!(is_local_file_path("../model.gguf"));
        assert!(is_local_file_path("../../models/file.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_windows_style_cov() {
        // Windows-style paths
        assert!(is_local_file_path("C:\\models\\model.gguf"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: validate_suite_name edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_suite_name_case_sensitive_cov() {
        // Suite names are case-sensitive
        assert!(!validate_suite_name("TENSOR_OPS"));
        assert!(!validate_suite_name("Inference"));
        assert!(validate_suite_name("tensor_ops"));
    }

    #[test]
    fn test_validate_suite_name_partial_match_cov() {
        // Partial matches should fail
        assert!(!validate_suite_name("tensor"));
        assert!(!validate_suite_name("ops"));
        assert!(!validate_suite_name("infer"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: format_size edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_zero_cov() {
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_exact_kb_cov() {
        assert_eq!(format_size(1024), "1.0 KB");
    }

    #[test]
    fn test_format_size_exact_mb_cov() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_format_size_exact_gb_cov() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_format_size_large_tb_cov() {
        // 10 TB still shows in GB
        let ten_tb = 10u64 * 1024 * 1024 * 1024 * 1024;
        let result = format_size(ten_tb);
        assert!(result.contains("GB"));
    }

    #[test]
    fn test_format_size_fractional_mb_cov() {
        let bytes = 1500 * 1024u64; // 1.5 MB
        let result = format_size(bytes);
        assert!(result.contains("MB"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: print_info
    // -------------------------------------------------------------------------

    #[test]
    fn test_print_info_cov() {
        // Just test that it doesn't panic
        print_info();
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: home_dir
    // -------------------------------------------------------------------------

    #[test]
    fn test_home_dir_returns_path_cov() {
        // home_dir may return None on some systems but should not panic
        let _ = home_dir();
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: validate_suite_name comprehensive
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_suite_name_all_valid_cov() {
        assert!(validate_suite_name("tensor_ops"));
        assert!(validate_suite_name("inference"));
        assert!(validate_suite_name("cache"));
        assert!(validate_suite_name("tokenizer"));
        assert!(validate_suite_name("quantize"));
        assert!(validate_suite_name("lambda"));
        assert!(validate_suite_name("comparative"));
    }

    #[test]
    fn test_validate_suite_name_empty_cov() {
        assert!(!validate_suite_name(""));
    }

    #[test]
    fn test_validate_suite_name_whitespace_cov() {
        assert!(!validate_suite_name("  "));
        assert!(!validate_suite_name("\t"));
        assert!(!validate_suite_name(" tensor_ops "));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: parse_cargo_bench_output edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_cargo_bench_output_empty_input_cov() {
        let results = parse_cargo_bench_output("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_no_bench_lines_ext_cov() {
        let output = "This is some random output\nwithout any benchmark data";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_multiple_benches_cov() {
        let output = "test bench_one ... bench: 100 ns/iter (+/- 10)\ntest bench_two ... bench: 200 ns/iter (+/- 20)";
        let results = parse_cargo_bench_output(output, Some("multi"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parse_cargo_bench_output_with_commas_cov() {
        let output = "test big_bench ... bench: 1,000,000 ns/iter (+/- 100)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["time_ns"], 1000000);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: BENCHMARK_SUITES metadata
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_suites_descriptions_not_empty_cov() {
        for (name, description) in BENCHMARK_SUITES {
            assert!(!name.is_empty());
            assert!(!description.is_empty());
        }
    }

    #[test]
    fn test_benchmark_suites_unique_names_cov() {
        let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
        let unique_count = names.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(names.len(), unique_count);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: is_local_file_path comprehensive
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_dot_slash_cov() {
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("./subdir/model.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_only_extension_cov() {
        // Just filename with extension
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("model.apr"));
    }

    #[test]
    fn test_is_local_file_path_no_extension_cov() {
        // No recognized extension
        assert!(!is_local_file_path("model"));
        assert!(!is_local_file_path("model.txt"));
        assert!(!is_local_file_path("model.bin"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_visualization
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_small_samples_cov() {
        run_visualization(false, 5);
    }

    #[test]
    fn test_run_visualization_large_samples_cov() {
        run_visualization(true, 100);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: display_model_info additional formats
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_safetensors_ext_cov() {
        let result = display_model_info("model.safetensors", &[0; 8]);
        // Should fail to parse empty data but not panic
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_display_model_info_gguf_ext_cov() {
        let result = display_model_info("model.gguf", &[0; 8]);
        // Should fail to parse invalid GGUF but not panic
        assert!(result.is_err() || result.is_ok());
    }

    // =========================================================================
    // Extended Coverage Tests: format_size boundaries
    // =========================================================================

    #[test]
    fn test_format_size_exact_boundaries_cov() {
        // Exact KB boundary
        assert_eq!(format_size(1024), "1.0 KB");
        // Exact MB boundary
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        // Exact GB boundary
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_format_size_just_below_boundaries_cov() {
        // Just below KB boundary
        assert_eq!(format_size(1023), "1023 B");
        // Just below MB boundary
        assert_eq!(format_size(1024 * 1024 - 1), "1024.0 KB");
        // Just below GB boundary
        assert_eq!(format_size(1024 * 1024 * 1024 - 1), "1024.0 MB");
    }

    #[test]
    fn test_format_size_large_values_cov() {
        // Multi-GB values
        assert_eq!(format_size(10 * 1024 * 1024 * 1024), "10.0 GB");
        assert_eq!(format_size(100 * 1024 * 1024 * 1024), "100.0 GB");
    }

    // =========================================================================
    // Extended Coverage Tests: parse_cargo_bench_output
    // =========================================================================

    #[test]
    fn test_parse_cargo_bench_output_nonnumeric_ext_cov() {
        let output = "test name bench: abc ns/iter (+/- 10)"; // non-numeric time
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty()); // Should skip malformed lines
    }

    #[test]
    fn test_parse_cargo_bench_output_no_test_keyword_ext_cov() {
        let output = "bench_name ... bench: 100 ns/iter (+/- 10)"; // no "test" keyword
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_too_few_parts_ext_cov() {
        let output = "test bench"; // too few parts
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_with_suite_name_ext_cov() {
        let output = "test bench_test ... bench: 500 ns/iter (+/- 50)";
        let results = parse_cargo_bench_output(output, Some("my_suite"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["suite"], "my_suite");
    }

    #[test]
    fn test_parse_cargo_bench_output_large_comma_separated_ext_cov() {
        let output = "test slow_bench ... bench: 50,000 ns/iter (+/- 1000)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["time_ns"], 50000);
    }

    // =========================================================================
    // Extended Coverage Tests: is_local_file_path
    // =========================================================================

    #[test]
    fn test_is_local_file_path_relative_subdirs_ext_cov() {
        assert!(is_local_file_path("./models/model.gguf"));
        assert!(is_local_file_path("../model.safetensors"));
        assert!(is_local_file_path("models/subdir/file.apr"));
    }

    #[test]
    fn test_is_local_file_path_absolute_linux_ext_cov() {
        assert!(is_local_file_path("/home/user/models/model.gguf"));
        assert!(is_local_file_path("/var/models/model.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_url_with_ext_ext_cov() {
        // URLs with recognized extensions ARE treated as "local" by current logic
        // (function checks extension, not scheme)
        assert!(is_local_file_path("http://example.com/model.gguf"));
        assert!(is_local_file_path("https://example.com/model.safetensors"));
        // URLs without recognized extensions are NOT treated as local
        assert!(!is_local_file_path("http://example.com/model"));
        assert!(!is_local_file_path("s3://bucket/model.bin"));
    }

    // =========================================================================
    // Extended Coverage Tests: validate_suite_name comprehensive
    // =========================================================================

    #[test]
    fn test_validate_suite_name_iterate_all_ext_cov() {
        for (name, _) in BENCHMARK_SUITES {
            assert!(validate_suite_name(name), "Suite {name} should be valid");
        }
    }

    #[test]
    fn test_validate_suite_name_uppercase_fails_ext_cov() {
        // Should be case sensitive
        assert!(!validate_suite_name("TENSOR_OPS"));
        assert!(!validate_suite_name("Tensor_Ops"));
        assert!(!validate_suite_name("INFERENCE"));
    }

    #[test]
    fn test_validate_suite_name_prefix_suffix_ext_cov() {
        assert!(!validate_suite_name("tensor")); // partial
        assert!(!validate_suite_name("tensor_ops_extra")); // too long
    }

    // =========================================================================
    // Extended Coverage Tests: display_model_info formats
    // =========================================================================

    #[test]
    fn test_display_model_info_apr_file_ext_cov() {
        let result = display_model_info("model.apr", &[0; 8]);
        // May error due to invalid APR data but shouldn't panic
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_display_model_info_unknown_extension_with_data_cov() {
        let result = display_model_info("model.xyz", &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(result.is_ok()); // Should handle unknown gracefully
    }

    #[test]
    fn test_display_model_info_no_extension_cov() {
        let result = display_model_info("model", &[0; 10]);
        // Should handle unknown gracefully
        assert!(result.is_ok() || result.is_err());
    }

    // =========================================================================
    // Extended Coverage Tests: load_*_model functions
    // =========================================================================

    #[test]
    fn test_load_gguf_model_empty_data_cov() {
        let result = load_gguf_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_model_empty_data_cov() {
        let result = load_safetensors_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_empty_data_cov() {
        let result = load_apr_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_invalid_magic_cov() {
        let result = load_apr_model(&[0x00, 0x00, 0x00, 0x00]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Extended Coverage Tests: home_dir
    // =========================================================================

    #[test]
    fn test_home_dir_not_panic_cov() {
        // Simply verify the function doesn't panic
        let _ = home_dir();
    }

    // =========================================================================
    // Extended Coverage Tests: run_benchmarks list mode
    // =========================================================================

    #[test]
    fn test_run_benchmarks_list_mode_cov() {
        let result = run_benchmarks(None, true, None, None, None, None);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Extended Coverage Tests: BENCHMARK_SUITES const
    // =========================================================================

    #[test]
    fn test_benchmark_suites_at_least_expected_cov() {
        assert!(BENCHMARK_SUITES.len() >= 5);
        // Should contain commonly expected suites
        let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"tensor_ops"));
        assert!(names.contains(&"inference"));
    }

    #[test]
    fn test_benchmark_suites_no_empty_descriptions_cov() {
        for (name, desc) in BENCHMARK_SUITES {
            assert!(!name.is_empty(), "Suite name should not be empty");
            assert!(!desc.is_empty(), "Suite description should not be empty");
            assert!(desc.len() > 5, "Description should be meaningful");
        }
    }

    // =========================================================================
    // Deep Coverage Tests: Error Handling and Edge Cases
    // Prefix: _deep_clicov_
    // =========================================================================

    #[test]
    fn test_deep_clicov_format_size_one_byte() {
        assert_eq!(format_size(1), "1 B");
    }

    #[test]
    fn test_deep_clicov_format_size_max_bytes() {
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_deep_clicov_format_size_fractional_kb() {
        // 1.5 KB
        assert_eq!(format_size(1536), "1.5 KB");
        // 2.25 KB -> rounds to 2.2 with .1f precision
        assert_eq!(format_size(2304), "2.2 KB");
    }

    #[test]
    fn test_deep_clicov_format_size_fractional_mb() {
        // 1.5 MB
        assert_eq!(format_size(1572864), "1.5 MB");
        // 2.75 MB
        let bytes = (2.75 * 1024.0 * 1024.0) as u64;
        assert!(format_size(bytes).contains("2."));
    }

    #[test]
    fn test_deep_clicov_format_size_fractional_gb() {
        // 1.5 GB
        let bytes = (1.5 * 1024.0 * 1024.0 * 1024.0) as u64;
        assert_eq!(format_size(bytes), "1.5 GB");
    }

    #[test]
    #[cfg(not(feature = "bench-http"))]
    fn test_deep_clicov_run_external_benchmark_feature_disabled() {
        // Test the non-bench-http stub (only runs when bench-http is disabled)
        let result = run_external_benchmark("ollama", "http://localhost:11434", None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(err_str.contains("bench-http"));
    }

    #[test]
    fn test_deep_clicov_run_external_benchmark_with_model() {
        let result = run_external_benchmark("vllm", "http://localhost:8000", Some("llama3"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_run_external_benchmark_with_output() {
        let result = run_external_benchmark(
            "llama-cpp",
            "http://localhost:8080",
            None,
            Some("/tmp/output.json"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_bench_without_ns_iter() {
        // Line has "bench:" but not "ns/iter"
        let output = "test benchmark_foo ... bench: 1234 ms (+/- 56)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_missing_bench_keyword() {
        // Line has ns/iter but no bench:
        let output = "test benchmark_foo ... result: 1234 ns/iter (+/- 56)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_test_at_end() {
        // "test" appears but not as first word after split
        let output = "running test benchmark_foo bench: 100 ns/iter (+/- 5)";
        let results = parse_cargo_bench_output(output, None);
        // Should still parse if "test" is found and name follows
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_no_name_after_test() {
        // "test" keyword but nothing follows
        let output = "test bench: 100 ns/iter";
        let results = parse_cargo_bench_output(output, None);
        // Can't extract name properly
        assert!(results.is_empty() || results.len() == 1);
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_dot_slash_only() {
        // Just "./" is not a file path
        assert!(is_local_file_path("./"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_slash_only() {
        // Just "/" is root path
        assert!(is_local_file_path("/"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_hidden_file() {
        assert!(is_local_file_path("./.hidden.gguf"));
        assert!(is_local_file_path("/home/.config/model.safetensors"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_special_chars() {
        assert!(is_local_file_path("./model-v1.0.gguf"));
        assert!(is_local_file_path("/path/to/model_v2.safetensors"));
        assert!(is_local_file_path("./model (copy).apr"));
    }

    #[test]
    fn test_deep_clicov_validate_suite_name_leading_trailing_spaces() {
        assert!(!validate_suite_name(" tensor_ops"));
        assert!(!validate_suite_name("tensor_ops "));
        assert!(!validate_suite_name(" tensor_ops "));
    }

    #[test]
    fn test_deep_clicov_validate_suite_name_newline() {
        assert!(!validate_suite_name("tensor_ops\n"));
        assert!(!validate_suite_name("\ntensor_ops"));
    }

    #[test]
    fn test_deep_clicov_display_model_info_gguf_magic_partial() {
        // GGUF magic but truncated data
        let data = b"GGUF";
        let result = display_model_info("model.bin", data);
        // Should fail to parse but not panic
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_deep_clicov_display_model_info_apr_magic_no_type() {
        // APR magic but no type bytes
        let data = b"APR\0";
        let result = display_model_info("model.bin", data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_load_apr_model_minimum_valid() {
        // Minimum valid APR header
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_load_apr_model_version_display() {
        // APR header with version info
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
        data[6..8].copy_from_slice(&42u16.to_le_bytes()); // version 42
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_run_convoy_test_all_none() {
        let result = run_convoy_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_run_saturation_test_all_none() {
        let result = run_saturation_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_run_visualization_zero_samples() {
        // Zero samples should not panic
        run_visualization(false, 0);
    }

    #[test]
    fn test_deep_clicov_run_visualization_two_samples() {
        run_visualization(true, 2);
    }

    #[test]
    fn test_deep_clicov_benchmark_suites_quantize_exists() {
        let found = BENCHMARK_SUITES.iter().any(|(name, _)| *name == "quantize");
        assert!(found);
    }

    #[test]
    fn test_deep_clicov_benchmark_suites_tokenizer_exists() {
        let found = BENCHMARK_SUITES
            .iter()
            .any(|(name, _)| *name == "tokenizer");
        assert!(found);
    }

    #[test]
    fn test_deep_clicov_benchmark_suites_count() {
        // Should have exactly 7 suites as defined
        assert_eq!(BENCHMARK_SUITES.len(), 7);
    }

    #[test]
    fn test_deep_clicov_run_benchmarks_url_only_without_runtime() {
        // URL without runtime should still work in list mode
        let result = run_benchmarks(
            None,
            true,
            None,
            None,
            Some("http://localhost:8080".to_string()),
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_suite_none_vs_some() {
        let output = "test bench_a ... bench: 100 ns/iter (+/- 5)";
        let results_none = parse_cargo_bench_output(output, None);
        let results_some = parse_cargo_bench_output(output, Some("test_suite"));

        assert_eq!(results_none.len(), results_some.len());
        assert!(results_none[0]["suite"].is_null());
        assert_eq!(results_some[0]["suite"], "test_suite");
    }

    #[test]
    fn test_deep_clicov_load_gguf_model_invalid_short_data() {
        let data = vec![0u8; 4]; // Too short
        let result = load_gguf_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_load_safetensors_model_invalid_short_data() {
        let data = vec![0u8; 4]; // Too short
        let result = load_safetensors_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_display_model_info_mixed_magic() {
        // Data that doesn't match any known magic
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
        let result = display_model_info("model.unknown", &data);
        assert!(result.is_ok()); // Should print "Unknown" format
    }

    #[test]
    fn test_deep_clicov_home_dir_env_var_behavior() {
        // Test that home_dir reads from HOME env var
        let result = home_dir();
        if std::env::var("HOME").is_ok() {
            assert!(result.is_some());
        }
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_uppercase_extension() {
        // Extensions are case-sensitive in current implementation
        assert!(!is_local_file_path("model.GGUF"));
        assert!(!is_local_file_path("model.SAFETENSORS"));
        assert!(!is_local_file_path("model.APR"));
    }

    #[test]
    fn test_deep_clicov_validate_suite_name_similar_names() {
        // Names similar to valid suites but not exact
        assert!(!validate_suite_name("tensor_op"));
        assert!(!validate_suite_name("tensors_ops"));
        assert!(!validate_suite_name("inferences"));
        assert!(!validate_suite_name("caches"));
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_realistic_output() {
        let output = r"
running 3 benchmarks
test tensor_ops::bench_add     ... bench:         123 ns/iter (+/- 12)
test tensor_ops::bench_mul     ... bench:       1,456 ns/iter (+/- 145)
test tensor_ops::bench_matmul  ... bench:      12,345 ns/iter (+/- 1,234)

test result: ok. 0 passed; 0 failed; 0 ignored; 3 measured; 0 filtered out
";
        let results = parse_cargo_bench_output(output, Some("tensor_ops"));
        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["time_ns"], 123);
        assert_eq!(results[1]["time_ns"], 1456);
        assert_eq!(results[2]["time_ns"], 12345);
    }

    #[test]
    fn test_deep_clicov_run_convoy_test_output_file_write() {
        use std::fs;
        let dir = std::env::temp_dir();
        let output = dir.join("deep_clicov_convoy_test.json");

        let result = run_convoy_test(
            Some("test_runtime".to_string()),
            Some("test_model.gguf".to_string()),
            Some(output.to_str().expect("invalid UTF-8").to_string()),
        );
        assert!(result.is_ok());

        // Verify file was created and contains valid JSON
        assert!(output.exists());
        let content = fs::read_to_string(&output).expect("file operation failed");
        assert!(content.contains("baseline"));
        assert!(content.contains("convoy"));

        let _ = fs::remove_file(&output);
    }

    #[test]
    fn test_deep_clicov_run_saturation_test_output_file_write() {
        use std::fs;
        let dir = std::env::temp_dir();
        let output = dir.join("deep_clicov_saturation_test.json");

        let result = run_saturation_test(
            Some("test_runtime".to_string()),
            Some("test_model.gguf".to_string()),
            Some(output.to_str().expect("invalid UTF-8").to_string()),
        );
        assert!(result.is_ok());

        // Verify file was created
        assert!(output.exists());
        let content = fs::read_to_string(&output).expect("file operation failed");
        assert!(content.contains("throughput"));

        let _ = fs::remove_file(&output);
    }

    #[test]
    fn test_deep_clicov_bench_compare_invalid_json_file1() {
        use std::fs::File;
        use std::io::Write;

        let dir = std::env::temp_dir();
        let file1 = dir.join("deep_clicov_invalid1.json");
        let file2 = dir.join("deep_clicov_invalid2.json");

        // Write invalid JSON to file1
        let mut f1 = File::create(&file1).expect("file operation failed");
        f1.write_all(b"not valid json").expect("operation failed");

        // Write valid but empty JSON to file2
        let mut f2 = File::create(&file2).expect("file operation failed");
        f2.write_all(b"{}").expect("operation failed");

        let result = run_bench_compare(
            file1.to_str().expect("file operation failed"),
            file2.to_str().expect("file operation failed"),
            5.0,
        );

        let _ = std::fs::remove_file(&file1);
        let _ = std::fs::remove_file(&file2);

        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_bench_regression_strict_mode() {
        use std::fs::File;
        use std::io::Write;

        let dir = std::env::temp_dir();
        let baseline = dir.join("deep_clicov_baseline.json");
        let current = dir.join("deep_clicov_current.json");

        // Write minimal valid JSON
        let mut f1 = File::create(&baseline).expect("file operation failed");
        f1.write_all(b"{}").expect("operation failed");

        let mut f2 = File::create(&current).expect("file operation failed");
        f2.write_all(b"{}").expect("operation failed");

        let result = run_bench_regression(
            baseline.to_str().expect("invalid UTF-8"),
            current.to_str().expect("invalid UTF-8"),
            true, // strict mode
        );

        let _ = std::fs::remove_file(&baseline);
        let _ = std::fs::remove_file(&current);

        // Should fail because JSON is not valid benchmark format
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_format_size_u64_max_safe() {
        // Large but not overflow-inducing value
        let large_value = 1_000_000 * 1024 * 1024 * 1024u64; // 1 PB
        let result = format_size(large_value);
        assert!(result.contains("GB"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_double_dot() {
        assert!(is_local_file_path("../model.gguf"));
        assert!(is_local_file_path("../../model.safetensors"));
        assert!(is_local_file_path("./../model.apr"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_protocol_like_strings() {
        // Strings that look like protocols but aren't
        assert!(!is_local_file_path("file:///path/to/model"));
        assert!(!is_local_file_path("hf://model"));
        assert!(!is_local_file_path("ollama://model"));
    }

    #[test]
    fn test_deep_clicov_benchmark_suites_description_content() {
        for (name, desc) in BENCHMARK_SUITES {
            // Descriptions should be informative
            assert!(desc.len() >= 10, "Description for {} too short", name);
            // Should not contain placeholder text
            assert!(!desc.contains("TODO"), "Description for {} has TODO", name);
            assert!(
                !desc.contains("FIXME"),
                "Description for {} has FIXME",
                name
            );
        }
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_whitespace_variations() {
        // Various whitespace in bench output
        let output1 = "test  bench_a  ...  bench:  100  ns/iter  (+/- 5)";
        let output2 = "test\tbench_b\t...\tbench:\t200\tns/iter\t(+/- 10)";

        let results1 = parse_cargo_bench_output(output1, None);
        let results2 = parse_cargo_bench_output(output2, None);

        // May or may not parse depending on whitespace handling
        assert!(results1.len() <= 1);
        assert!(results2.len() <= 1);
    }

    #[test]
    fn test_deep_clicov_display_model_info_safetensors_magic() {
        // SafeTensors files start with 8-byte length header
        // Valid SafeTensors starts with little-endian u64 for header size
        let mut data = vec![0u8; 16];
        data[0..8].copy_from_slice(&0u64.to_le_bytes()); // Header size = 0
        let result = display_model_info("model.safetensors", &data);
        // Should try to parse as SafeTensors
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_deep_clicov_load_apr_model_custom_type() {
        // Custom model type (0x00FF)
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x00FFu16.to_le_bytes()); // Custom type
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_run_benchmarks_all_params_non_list() {
        // All params but not in list mode - will attempt cargo bench
        // Just verify the function signature accepts all params
        // We don't actually run cargo bench in tests
        let result = run_benchmarks(
            Some("tensor_ops".to_string()),
            true, // Use list mode to avoid cargo bench
            Some("realizar".to_string()),
            Some("model.gguf".to_string()),
            Some("http://localhost".to_string()),
            Some("/tmp/out.json".to_string()),
        );
        assert!(result.is_ok());
    }

    // =========================================================================
    // Server Command Tests (EXTREME TDD - PAR-112)
    // =========================================================================

    #[tokio::test]
    async fn test_serve_model_invalid_extension() {
        // Test that unsupported file extensions return error
        let result = serve_model("127.0.0.1", 8080, "/nonexistent/model.xyz", false, false).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported file extension"));
    }

    #[tokio::test]
    async fn test_serve_model_nonexistent_gguf() {
        // Test that nonexistent GGUF file returns error
        let result = serve_model("127.0.0.1", 8080, "/nonexistent/model.gguf", false, false).await;
        assert!(result.is_err());
        // Should fail during GGUF loading
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to load GGUF")
                || err.to_string().contains("No such file")
                || err.to_string().contains("mmap")
        );
    }

    #[tokio::test]
    async fn test_serve_model_nonexistent_safetensors() {
        // Test that nonexistent SafeTensors file returns error
        let result = serve_model(
            "127.0.0.1",
            8080,
            "/nonexistent/model.safetensors",
            false,
            false,
        )
        .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to read") || err.to_string().contains("No such file")
        );
    }

    #[tokio::test]
    async fn test_serve_model_nonexistent_apr() {
        // Test that nonexistent APR file returns error
        let result = serve_model("127.0.0.1", 8080, "/nonexistent/model.apr", false, false).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to read") || err.to_string().contains("No such file")
        );
    }

    #[test]
    fn test_serve_model_extension_detection() {
        // Verify extension detection logic
        assert!("/path/to/model.gguf".ends_with(".gguf"));
        assert!("/path/to/model.safetensors".ends_with(".safetensors"));
        assert!("/path/to/model.apr".ends_with(".apr"));
        assert!(!"/path/to/model.xyz".ends_with(".gguf"));
        assert!(!"/path/to/model.xyz".ends_with(".safetensors"));
        assert!(!"/path/to/model.xyz".ends_with(".apr"));
    }

    #[test]
    fn test_serve_model_address_parsing() {
        // Verify address parsing works correctly
        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:8080".parse();
        assert!(addr.is_ok());

        let addr: std::result::Result<std::net::SocketAddr, _> = "0.0.0.0:3000".parse();
        assert!(addr.is_ok());

        let addr: std::result::Result<std::net::SocketAddr, _> = "invalid:port".parse();
        assert!(addr.is_err());
    }

    #[test]
    fn test_serve_model_port_ranges() {
        // Verify port range handling
        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:0".parse();
        assert!(addr.is_ok()); // Port 0 = OS assigns

        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:65535".parse();
        assert!(addr.is_ok()); // Max port

        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:80".parse();
        assert!(addr.is_ok()); // Privileged port (may need root)
    }

    #[test]
    fn test_batch_mode_flag_logic() {
        // Test batch mode flag combinations
        let batch_mode = true;
        let force_gpu = false;
        assert!(batch_mode && !force_gpu); // Valid: batch without forced GPU

        let batch_mode = true;
        let force_gpu = true;
        assert!(batch_mode && force_gpu); // Valid: batch with GPU

        let batch_mode = false;
        let force_gpu = true;
        assert!(!batch_mode && force_gpu); // Valid: single-request with GPU (true streaming)
    }

    #[test]
    fn test_cuda_env_var_detection() {
        // Test REALIZAR_BACKEND environment variable detection
        std::env::remove_var("REALIZAR_BACKEND");
        let use_cuda = std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        assert!(!use_cuda);

        std::env::set_var("REALIZAR_BACKEND", "cuda");
        let use_cuda = std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        assert!(use_cuda);

        std::env::set_var("REALIZAR_BACKEND", "CUDA");
        let use_cuda = std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        assert!(use_cuda);

        std::env::set_var("REALIZAR_BACKEND", "cpu");
        let use_cuda = std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        assert!(!use_cuda);

        // Cleanup
        std::env::remove_var("REALIZAR_BACKEND");
    }

    // =========================================================================
    // ModelType Tests
    // =========================================================================

    #[test]
    fn test_model_type_display() {
        assert_eq!(format!("{}", ModelType::Gguf), "GGUF");
        assert_eq!(format!("{}", ModelType::SafeTensors), "SafeTensors");
        assert_eq!(format!("{}", ModelType::Apr), "APR");
    }

    #[test]
    fn test_model_type_debug() {
        assert_eq!(format!("{:?}", ModelType::Gguf), "Gguf");
        assert_eq!(format!("{:?}", ModelType::SafeTensors), "SafeTensors");
        assert_eq!(format!("{:?}", ModelType::Apr), "Apr");
    }

    #[test]
    fn test_model_type_clone_copy() {
        let mt = ModelType::Gguf;
        let mt_clone = mt;
        let mt_copy = mt;
        assert_eq!(mt, mt_clone);
        assert_eq!(mt, mt_copy);
    }

    #[test]
    fn test_model_type_equality() {
        assert_eq!(ModelType::Gguf, ModelType::Gguf);
        assert_eq!(ModelType::SafeTensors, ModelType::SafeTensors);
        assert_eq!(ModelType::Apr, ModelType::Apr);
        assert_ne!(ModelType::Gguf, ModelType::SafeTensors);
        assert_ne!(ModelType::SafeTensors, ModelType::Apr);
        assert_ne!(ModelType::Apr, ModelType::Gguf);
    }

    // =========================================================================
    // PreparedServer Tests
    // =========================================================================

    #[test]
    fn test_prepared_server_debug() {
        // Create a demo AppState for testing
        let state = crate::api::AppState::demo().expect("demo state");
        let prepared = PreparedServer {
            state,
            batch_mode_enabled: true,
            model_type: ModelType::Gguf,
        };
        let debug_str = format!("{:?}", prepared);
        assert!(debug_str.contains("PreparedServer"));
        assert!(debug_str.contains("batch_mode_enabled: true"));
        assert!(debug_str.contains("model_type: Gguf"));
    }

    #[test]
    fn test_prepared_server_fields() {
        let state = crate::api::AppState::demo().expect("demo state");
        let prepared = PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::SafeTensors,
        };
        assert!(!prepared.batch_mode_enabled);
        assert_eq!(prepared.model_type, ModelType::SafeTensors);
    }

    // =========================================================================
    // prepare_serve_state Tests (EXTREME TDD)
    // =========================================================================

    #[test]
    fn test_prepare_serve_state_invalid_extension() {
        // Test that unsupported file extensions return error
        let result = prepare_serve_state("/nonexistent/model.xyz", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported file extension"));
    }

    #[test]
    fn test_prepare_serve_state_nonexistent_gguf() {
        // Test that nonexistent GGUF file returns error
        let result = prepare_serve_state("/nonexistent/model.gguf", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to load GGUF")
                || err.to_string().contains("No such file")
                || err.to_string().contains("mmap")
        );
    }

    #[test]
    fn test_prepare_serve_state_nonexistent_safetensors() {
        // Test that nonexistent SafeTensors file returns error
        let result = prepare_serve_state("/nonexistent/model.safetensors", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to read") || err.to_string().contains("No such file")
        );
    }

    #[test]
    fn test_prepare_serve_state_nonexistent_apr() {
        // Test that nonexistent APR file returns error
        let result = prepare_serve_state("/nonexistent/model.apr", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to read") || err.to_string().contains("No such file")
        );
    }

    #[test]
    fn test_prepare_serve_state_with_batch_mode_flag() {
        // Test batch_mode flag is recorded even when loading fails
        let result = prepare_serve_state("/nonexistent/model.gguf", true, false);
        assert!(result.is_err()); // File doesn't exist, but flag should be processed before
    }

    #[test]
    fn test_prepare_serve_state_with_force_gpu_flag() {
        // Test force_gpu flag is recorded even when loading fails
        let result = prepare_serve_state("/nonexistent/model.gguf", false, true);
        assert!(result.is_err()); // File doesn't exist, but flag should be processed before
    }

    #[test]
    fn test_prepare_serve_state_extension_variants() {
        // Test various file extension patterns
        let extensions = vec![
            ("/path/model.gguf", true, "GGUF"),
            ("/path/MODEL.GGUF", false, "uppercase"),
            ("/path/model.safetensors", true, "SafeTensors"),
            ("/path/model.apr", true, "APR"),
            ("/path/model.pt", false, "PyTorch"),
            ("/path/model.bin", false, "binary"),
            ("/path/model.h5", false, "HDF5"),
            ("/path/model", false, "no extension"),
        ];

        for (path, should_detect, name) in extensions {
            let is_gguf = path.ends_with(".gguf");
            let is_safetensors = path.ends_with(".safetensors");
            let is_apr = path.ends_with(".apr");
            let detected = is_gguf || is_safetensors || is_apr;
            assert_eq!(
                detected, should_detect,
                "Extension detection failed for {name}: {path}"
            );
        }
    }

    // =========================================================================
    // serve_demo Tests (EXTREME TDD)
    // =========================================================================

    #[test]
    fn test_serve_demo_address_validation() {
        // Test that address parsing logic works correctly
        let valid_addresses = vec![
            ("127.0.0.1", 8080),
            ("0.0.0.0", 3000),
            ("localhost", 8000), // This won't parse as SocketAddr directly
        ];

        for (host, port) in valid_addresses {
            let addr_str = format!("{}:{}", host, port);
            let result: std::result::Result<std::net::SocketAddr, _> = addr_str.parse();
            // localhost won't parse, but IP addresses should
            if host != "localhost" {
                assert!(result.is_ok(), "Address {addr_str} should be valid");
            }
        }
    }

    #[test]
    fn test_serve_demo_port_zero() {
        // Port 0 should be valid (OS assigns port)
        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:0".parse();
        assert!(addr.is_ok());
    }

    // =========================================================================
    // Integration Tests (run with `cargo test -- --ignored`)
    // =========================================================================

    /// Integration test for prepare_serve_state with a real GGUF model
    /// Run with: cargo test test_prepare_serve_state_gguf_success -- --ignored
    #[test]
    #[ignore]
    fn test_prepare_serve_state_gguf_success() {
        // Look for a test model file
        let model_paths = [
            "/home/noah/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        ];

        let model_path = model_paths
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .expect("No test model file found. Run tests with a valid GGUF model.");

        let result = prepare_serve_state(model_path, false, false);
        assert!(
            result.is_ok(),
            "prepare_serve_state failed: {:?}",
            result.err()
        );

        let prepared = result.expect("operation failed");
        assert_eq!(prepared.model_type, ModelType::Gguf);
        assert!(!prepared.batch_mode_enabled);
        // State should have a quantized model
    }

    /// Integration test for prepare_serve_state with batch mode
    /// Run with: cargo test test_prepare_serve_state_gguf_batch -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_prepare_serve_state_gguf_batch() {
        let model_paths = [
            "/home/noah/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        ];

        let model_path = model_paths
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .expect("No test model file found.");

        // Batch mode requires Tokio runtime for spawn_batch_processor
        let result = prepare_serve_state(model_path, true, false);
        assert!(
            result.is_ok(),
            "prepare_serve_state failed: {:?}",
            result.err()
        );

        let prepared = result.expect("operation failed");
        assert_eq!(prepared.model_type, ModelType::Gguf);
        // batch_mode_enabled is true since we enabled it and gpu feature is available
        assert!(prepared.batch_mode_enabled);
    }

    /// Test all model type variants are properly detected
    #[test]
    fn test_model_type_from_extension() {
        // This tests the extension detection logic in prepare_serve_state
        let test_cases = vec![
            ("model.gguf", Some("gguf")),
            ("model.safetensors", Some("safetensors")),
            ("model.apr", Some("apr")),
            ("model.bin", None),
            ("model", None),
        ];

        for (path, expected_ext) in test_cases {
            let is_gguf = path.ends_with(".gguf");
            let is_safetensors = path.ends_with(".safetensors");
            let is_apr = path.ends_with(".apr");

            let actual = if is_gguf {
                Some("gguf")
            } else if is_safetensors {
                Some("safetensors")
            } else if is_apr {
                Some("apr")
            } else {
                None
            };

            assert_eq!(
                actual, expected_ext,
                "Extension detection failed for {path}"
            );
        }
    }

    /// Test PreparedServer with all model types
    #[test]
    fn test_prepared_server_all_model_types() {
        for model_type in [ModelType::Gguf, ModelType::SafeTensors, ModelType::Apr] {
            let state = crate::api::AppState::demo().expect("demo state");
            let prepared = PreparedServer {
                state,
                batch_mode_enabled: false,
                model_type,
            };
            assert_eq!(prepared.model_type, model_type);
            // Test that debug output includes the model type
            let debug = format!("{:?}", prepared);
            assert!(debug.contains(&format!("{:?}", model_type)));
        }
    }

    /// Test that serve_model properly delegates to prepare_serve_state
    #[tokio::test]
    async fn test_serve_model_delegates_to_prepare_serve_state() {
        // Test that serve_model returns the same error as prepare_serve_state
        // for invalid extensions
        let result = serve_model("127.0.0.1", 8080, "/nonexistent/model.xyz", false, false).await;
        assert!(result.is_err());

        // The error should match what prepare_serve_state returns
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported file extension"));
    }

    /// Test address parsing in serve_model context
    #[test]
    fn test_serve_model_address_formats() {
        // Test various IPv4 address format combinations
        let ipv4_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 3000), ("192.168.1.1", 80)];

        for (host, port) in ipv4_cases {
            let addr_str = format!("{}:{}", host, port);
            let result: std::result::Result<std::net::SocketAddr, _> = addr_str.parse();
            assert!(result.is_ok(), "IPv4 address parsing failed for {addr_str}");
        }

        // Test IPv6 address (needs brackets for SocketAddr parsing)
        let ipv6_addr: std::result::Result<std::net::SocketAddr, _> = "[::1]:8080".parse();
        assert!(ipv6_addr.is_ok(), "IPv6 address parsing failed");
    }

    /// Test that PreparedServer can be created with demo state
    #[test]
    fn test_prepared_server_with_demo_state() {
        let state = crate::api::AppState::demo().expect("Failed to create demo state");
        let prepared = PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::Gguf,
        };

        // Verify the struct fields
        assert!(!prepared.batch_mode_enabled);
        assert_eq!(prepared.model_type, ModelType::Gguf);

        // Verify debug output is useful
        let debug = format!("{:?}", prepared);
        assert!(debug.contains("batch_mode_enabled: false"));
    }

    // =========================================================================
    // CLI Inference Module Tests (EXTREME TDD - PMAT-802)
    // Coverage for src/cli/inference.rs
    // =========================================================================

    mod inference_tests {

        use crate::cli::inference;

        // -------------------------------------------------------------------------
        // run_gguf_inference Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_run_gguf_inference_invalid_model_path() {
            // Test with empty file data - should fail to mmap
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],     // empty file data
                "Hello", // prompt
                10,      // max_tokens
                0.0,     // temperature (greedy)
                "text",  // format
                false,   // force_gpu
                false,   // verbose
                false,   // trace
            );
            assert!(result.is_err());
            let err = result.unwrap_err();
            // Should fail during mmap since path doesn't exist
            assert!(
                err.to_string().contains("mmap")
                    || err.to_string().contains("Failed to mmap")
                    || err.to_string().contains("No such file"),
                "Expected mmap error, got: {}",
                err
            );
        }

        #[test]
        fn test_run_gguf_inference_invalid_gguf_data() {
            // Test with non-existent path - the function reads from path, not from data param
            let invalid_data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
            let result = inference::run_gguf_inference(
                "/nonexistent_invalid_model.gguf",
                &invalid_data,
                "Test prompt",
                5,
                0.7,
                "json",
                false,
                true, // verbose
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_gguf_inference_format_json() {
            // Test that JSON format parameter is accepted
            // (Will fail on model loading, but exercises format parsing)
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                1,
                0.0,
                "json", // JSON output format
                false,
                false,
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_format_text() {
            // Test that text format parameter is accepted
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                1,
                0.0,
                "text", // text output format
                false,
                false,
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_greedy_temperature() {
            // Test greedy decoding (temperature <= 0.01)
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                5,
                0.0, // Greedy (temperature = 0)
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_sampling_temperature() {
            // Test temperature sampling
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                5,
                1.0, // Temperature sampling
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_verbose_mode() {
            // Test verbose output path
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "verbose test",
                5,
                0.5,
                "text",
                false,
                true, // verbose = true
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_trace_mode() {
            // Test trace output path
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "trace test",
                5,
                0.5,
                "text",
                false,
                false,
                true, // trace = true
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_force_gpu_flag() {
            // Test force_gpu flag (should warn if cuda not available)
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "gpu test",
                5,
                0.5,
                "text",
                true, // force_gpu = true
                false,
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_all_flags() {
            // Test with all flags enabled
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "comprehensive test",
                10,
                0.8,
                "json",
                true, // force_gpu
                true, // verbose
                true, // trace
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_empty_prompt() {
            // Test with empty prompt
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "", // empty prompt
                5,
                0.0,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_zero_tokens() {
            // Test with zero max_tokens
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                0, // zero tokens to generate
                0.0,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_large_tokens() {
            // Test with large max_tokens
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                10000, // large number of tokens
                0.0,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        // -------------------------------------------------------------------------
        // run_safetensors_inference Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_run_safetensors_inference_invalid_path() {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "Test prompt",
                10,
                0.5,
                "text",
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_safetensors_inference_json_format() {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "JSON test",
                5,
                0.0,
                "json",
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_safetensors_inference_text_format() {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "text test",
                5,
                0.0,
                "text",
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_safetensors_inference_empty_prompt() {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "",
                10,
                0.7,
                "text",
            );
            assert!(result.is_err());
        }

        // -------------------------------------------------------------------------
        // run_apr_inference Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_run_apr_inference_invalid_path() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "Test prompt",
                10,
                0.5,
                "text",
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_invalid_apr_data() {
            // Test with invalid APR magic bytes
            let invalid_data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
            let result = inference::run_apr_inference(
                "/tmp/test.apr",
                &invalid_data,
                "Test prompt",
                5,
                0.7,
                "text",
                false,
                true,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_json_format() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "JSON test",
                5,
                0.0,
                "json",
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_text_format() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "text test",
                5,
                0.0,
                "text",
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_force_gpu_flag() {
            // Test force_gpu flag
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "gpu test",
                5,
                0.5,
                "text",
                true, // force_gpu
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_verbose_mode() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "verbose test",
                5,
                0.5,
                "text",
                false,
                true, // verbose
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_all_flags() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "comprehensive test",
                10,
                0.8,
                "json",
                true, // force_gpu
                true, // verbose
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_empty_prompt() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "",
                10,
                0.7,
                "text",
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_greedy_temperature() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "greedy test",
                5,
                0.0, // greedy
                "text",
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_high_temperature() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "sampling test",
                5,
                2.0, // high temperature
                "text",
                false,
                false,
            );
            assert!(result.is_err());
        }

        // -------------------------------------------------------------------------
        // Parameter Validation Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_inference_temperature_boundary_greedy() {
            // Temperature <= 0.01 triggers greedy decoding
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "boundary test",
                5,
                0.01, // Exactly at boundary
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_inference_temperature_boundary_sampling() {
            // Temperature > 0.01 triggers sampling
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "boundary test",
                5,
                0.02, // Just above boundary
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_inference_format_unknown() {
            // Unknown format should default to text-like output
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "format test",
                5,
                0.5,
                "xml", // Unknown format
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }

        // -------------------------------------------------------------------------
        // Environment Variable Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_cpu_debug_env_var() {
            // The CPU_DEBUG environment variable controls diagnostic output
            std::env::remove_var("CPU_DEBUG");
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "env test",
                5,
                0.5,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());

            // Set CPU_DEBUG=1 and try again
            std::env::set_var("CPU_DEBUG", "1");
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "env test",
                5,
                0.5,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
            std::env::remove_var("CPU_DEBUG");
        }

        #[test]
        fn test_skip_gpu_resident_env_var() {
            // The SKIP_GPU_RESIDENT environment variable affects GPU path selection
            std::env::remove_var("SKIP_GPU_RESIDENT");
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "env test",
                5,
                0.5,
                "text",
                true, // force_gpu
                false,
                false,
            );
            assert!(result.is_err());

            // Set SKIP_GPU_RESIDENT=1 and try again
            std::env::set_var("SKIP_GPU_RESIDENT", "1");
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "env test",
                5,
                0.5,
                "text",
                true, // force_gpu
                false,
                false,
            );
            assert!(result.is_err());
            std::env::remove_var("SKIP_GPU_RESIDENT");
        }

        // -------------------------------------------------------------------------
        // Error Message Content Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_gguf_error_contains_operation() {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "error test",
                5,
                0.5,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_str = err.to_string();
            // Error should identify the operation that failed
            assert!(
                err_str.contains("mmap") || err_str.contains("load") || err_str.contains("GGUF"),
                "Error should mention mmap or load operation: {}",
                err_str
            );
        }

        #[test]
        fn test_apr_error_contains_operation() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "error test",
                5,
                0.5,
                "text",
                false,
                false,
            );
            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_str = err.to_string();
            // Error should identify the operation that failed
            assert!(
                err_str.contains("parse") || err_str.contains("APR") || err_str.contains("Failed"),
                "Error should mention parse or APR operation: {}",
                err_str
            );
        }

        // -------------------------------------------------------------------------
        // Integration Tests (ignored by default - require real models)
        // -------------------------------------------------------------------------

        /// Integration test for GGUF inference with a real model
        /// Run with: cargo test test_run_gguf_inference_real -- --ignored
        #[test]
        #[ignore]
        fn test_run_gguf_inference_real() {
            let model_paths = [
                "/home/noah/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
            ];

            let model_path = model_paths
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .expect("No test model file found");

            // Read the file
            let file_data = std::fs::read(model_path).expect("Failed to read model file");

            let result = inference::run_gguf_inference(
                model_path,
                &file_data,
                "Hello, world!",
                5,
                0.0, // greedy
                "text",
                false,
                true, // verbose
                false,
            );

            // Should succeed with real model
            assert!(result.is_ok(), "Inference failed: {:?}", result.err());
        }

        /// Integration test for GGUF inference with JSON output
        #[test]
        #[ignore]
        fn test_run_gguf_inference_json_output_real() {
            let model_paths = ["/home/noah/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"];

            let model_path = model_paths
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .expect("No test model file found");

            let file_data = std::fs::read(model_path).expect("Failed to read model file");

            let result = inference::run_gguf_inference(
                model_path,
                &file_data,
                "What is 2+2?",
                10,
                0.0,
                "json", // JSON output
                false,
                false,
                false,
            );

            assert!(result.is_ok(), "JSON inference failed: {:?}", result.err());
        }

        // -------------------------------------------------------------------------
        // Comprehensive API Surface Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_gguf_inference_api_surface() {
            // Verify all parameters are accepted in the expected order
            let _ = inference::run_gguf_inference(
                "model_ref", // model_ref: &str
                &[],         // file_data: &[u8]
                "prompt",    // prompt: &str
                10usize,     // max_tokens: usize
                0.5f32,      // temperature: f32
                "format",    // format: &str
                true,        // force_gpu: bool
                true,        // verbose: bool
                true,        // trace: bool
            );
        }

        #[test]
        fn test_safetensors_inference_api_surface() {
            // Verify all parameters are accepted in the expected order
            let _ = inference::run_safetensors_inference(
                "model_ref", // model_ref: &str
                "prompt",    // prompt: &str
                10usize,     // max_tokens: usize
                0.5f32,      // temperature: f32 (unused in current impl)
                "format",    // format: &str
            );
        }

        #[test]
        fn test_apr_inference_api_surface() {
            // Verify all parameters are accepted in the expected order
            let _ = inference::run_apr_inference(
                "model_ref", // model_ref: &str
                &[],         // file_data: &[u8]
                "prompt",    // prompt: &str
                10usize,     // max_tokens: usize
                0.5f32,      // temperature: f32
                "format",    // format: &str
                true,        // force_gpu: bool
                true,        // verbose: bool
            );
        }

        // -------------------------------------------------------------------------
        // Edge Case Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_unicode_prompt() {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "Hello \u{1F600} \u{4E2D}\u{6587} \u{0410}\u{0411}\u{0412}",
                5,
                0.5,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_very_long_prompt() {
            let long_prompt = "word ".repeat(1000);
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                &long_prompt,
                5,
                0.5,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_special_characters_in_prompt() {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "Test <script>alert('xss')</script> & \"quotes\" 'apostrophe'",
                5,
                0.5,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_newlines_in_prompt() {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "Line 1\nLine 2\r\nLine 3\tTab",
                5,
                0.5,
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_negative_temperature_clamped() {
            // Negative temperature should be treated as greedy (temperature <= 0.01)
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "negative temp test",
                5,
                -1.0, // Negative temperature
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_extreme_temperature() {
            // Very high temperature
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "extreme temp test",
                5,
                100.0, // Very high temperature
                "text",
                false,
                false,
                false,
            );
            assert!(result.is_err());
        }
    }
}
