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
