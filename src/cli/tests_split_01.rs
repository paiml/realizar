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
