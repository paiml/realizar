//! EXTREME TDD: CLI Module Coverage Tests
//!
//! Additional tests for cli.rs to increase coverage to 85%+.
//! Tests format_size, display_model_info, benchmark functions, and utility functions.

use realizar::cli::{
    display_model_info, format_size, home_dir, is_local_file_path, print_info, run_bench_compare,
    run_bench_regression, run_benchmarks, run_convoy_test, run_saturation_test, run_visualization,
    validate_suite_name, BENCHMARK_SUITES,
};

// ===== format_size Tests =====

#[test]
fn test_cov_format_size_exact_boundaries() {
    // Exact KB boundary
    assert_eq!(format_size(1024), "1.0 KB");

    // Exact MB boundary
    assert_eq!(format_size(1024 * 1024), "1.0 MB");

    // Exact GB boundary
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
}

#[test]
fn test_cov_format_size_fractional() {
    // 1.5 KB
    assert_eq!(format_size(1536), "1.5 KB");

    // 2.5 MB
    assert_eq!(format_size(2621440), "2.5 MB");

    // 3.5 GB
    assert_eq!(format_size(3758096384), "3.5 GB");
}

#[test]
fn test_cov_format_size_large_values() {
    // 100 GB
    assert_eq!(format_size(107374182400), "100.0 GB");

    // 1 TB (displayed as GB)
    assert_eq!(format_size(1099511627776), "1024.0 GB");
}

#[test]
fn test_cov_format_size_small_values() {
    assert_eq!(format_size(0), "0 B");
    assert_eq!(format_size(1), "1 B");
    assert_eq!(format_size(100), "100 B");
    assert_eq!(format_size(1023), "1023 B");
}

// ===== BENCHMARK_SUITES Tests =====

#[test]
fn test_cov_benchmark_suites_contains_expected() {
    let suite_names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();

    assert!(suite_names.contains(&"tensor_ops"));
    assert!(suite_names.contains(&"inference"));
    assert!(suite_names.contains(&"cache"));
    assert!(suite_names.contains(&"tokenizer"));
    assert!(suite_names.contains(&"quantize"));
}

#[test]
fn test_cov_benchmark_suites_unique_names() {
    let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
    let mut unique_names = names.clone();
    unique_names.sort();
    unique_names.dedup();
    assert_eq!(
        names.len(),
        unique_names.len(),
        "Benchmark names should be unique"
    );
}

#[test]
fn test_cov_benchmark_suites_descriptions_not_empty() {
    for (name, description) in BENCHMARK_SUITES {
        assert!(!name.is_empty(), "Suite name should not be empty");
        assert!(
            !description.is_empty(),
            "Suite '{}' description should not be empty",
            name
        );
        assert!(
            description.len() > 5,
            "Suite '{}' description should be meaningful",
            name
        );
    }
}

// ===== validate_suite_name Tests =====

#[test]
fn test_cov_validate_suite_name_all_valid() {
    for (name, _) in BENCHMARK_SUITES {
        assert!(
            validate_suite_name(name),
            "Suite '{}' should be valid",
            name
        );
    }
}

#[test]
fn test_cov_validate_suite_name_invalid_cases() {
    assert!(!validate_suite_name(""));
    assert!(!validate_suite_name(" "));
    assert!(!validate_suite_name("TENSOR_OPS")); // case sensitive
    assert!(!validate_suite_name("tensor-ops")); // hyphen vs underscore
    assert!(!validate_suite_name("nonexistent"));
    assert!(!validate_suite_name("inference ")); // trailing space
    assert!(!validate_suite_name(" inference")); // leading space
}

// ===== is_local_file_path Tests =====

#[test]
fn test_cov_is_local_file_path_gguf() {
    assert!(is_local_file_path("model.gguf"));
    assert!(is_local_file_path("./models/model.gguf"));
    assert!(is_local_file_path("/absolute/path/model.gguf"));
    assert!(is_local_file_path("../relative/model.gguf"));
}

#[test]
fn test_cov_is_local_file_path_safetensors() {
    assert!(is_local_file_path("model.safetensors"));
    assert!(is_local_file_path("./model.safetensors"));
    assert!(is_local_file_path("/path/to/model.safetensors"));
}

#[test]
fn test_cov_is_local_file_path_apr() {
    assert!(is_local_file_path("model.apr"));
    assert!(is_local_file_path("./model.apr"));
    assert!(is_local_file_path("/path/to/model.apr"));
}

#[test]
fn test_cov_is_local_file_path_relative() {
    assert!(is_local_file_path("./model.bin"));
    // ../model.bin doesn't match the function's criteria (doesn't start with "./" or "/")
    // models/subdir/file.gguf returns true because it ends with .gguf
    assert!(is_local_file_path("models/subdir/file.gguf"));
}

#[test]
fn test_cov_is_local_file_path_absolute() {
    assert!(is_local_file_path("/home/user/model.gguf"));
    assert!(is_local_file_path("/tmp/model.safetensors"));
}

#[test]
fn test_cov_is_local_file_path_remote_schemes() {
    // Model references that don't look like local files (no extension, no leading ./ or /)
    assert!(!is_local_file_path("llama3:8b"));
    assert!(!is_local_file_path("pacha://model:v1"));
    assert!(!is_local_file_path("hf://meta-llama/Llama-3"));
    // Note: URLs ending in .gguf/.safetensors/.apr will return true due to extension check
    // This is the actual behavior of the function
    assert!(is_local_file_path("http://example.com/model.gguf")); // ends with .gguf
}

// ===== home_dir Tests =====

#[test]
fn test_cov_home_dir_returns_option() {
    let home = home_dir();
    // May be None in some environments, but should not panic
    if let Some(path) = home {
        // If present, should be a valid path
        let path_str = path.to_string_lossy();
        assert!(!path_str.is_empty());
    }
}

// ===== display_model_info Tests =====

#[test]
fn test_cov_display_model_info_empty_data() {
    let result = display_model_info("model.bin", &[]);
    assert!(result.is_ok());
}

#[test]
fn test_cov_display_model_info_random_data() {
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let result = display_model_info("model.bin", &data);
    assert!(result.is_ok());
}

#[test]
fn test_cov_display_model_info_apr_extension() {
    // APR magic is "APR\0"
    let data = b"APR\0invalid".to_vec();
    let result = display_model_info("model.apr", &data);
    // May fail parsing, but shouldn't panic
    let _ = result;
}

#[test]
fn test_cov_display_model_info_safetensors_extension_invalid() {
    let result = display_model_info("model.safetensors", &[0, 1, 2, 3]);
    assert!(result.is_err()); // Invalid safetensors format
}

// ===== print_info Tests =====

#[test]
fn test_cov_print_info_no_panic() {
    // Just verify it doesn't panic
    print_info();
}

// ===== run_visualization Tests =====

#[test]
fn test_cov_run_visualization_minimal() {
    run_visualization(false, 5);
}

#[test]
fn test_cov_run_visualization_with_color() {
    run_visualization(true, 5);
}

#[test]
fn test_cov_run_visualization_larger() {
    run_visualization(false, 100);
}

// ===== run_benchmarks Tests =====

#[test]
fn test_cov_run_benchmarks_list_only() {
    let result = run_benchmarks(None, true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_benchmarks_list_with_suite() {
    let result = run_benchmarks(Some("tensor_ops".to_string()), true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_benchmarks_list_with_all_options() {
    let result = run_benchmarks(
        Some("inference".to_string()),
        true,
        Some("realizar".to_string()),
        Some("model.gguf".to_string()),
        Some("http://localhost:8080".to_string()),
        Some("/tmp/output.json".to_string()),
    );
    assert!(result.is_ok());
}

// ===== run_convoy_test Tests =====

#[test]
fn test_cov_run_convoy_test_minimal() {
    let result = run_convoy_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_convoy_test_with_url() {
    let result = run_convoy_test(Some("http://localhost:8080".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_convoy_test_with_model() {
    let result = run_convoy_test(None, Some("model.gguf".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_convoy_test_with_output() {
    let output_path = std::env::temp_dir().join("convoy_test_output.json");
    let result = run_convoy_test(
        Some("test".to_string()),
        Some("model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output_path);
}

// ===== run_saturation_test Tests =====

#[test]
fn test_cov_run_saturation_test_minimal() {
    let result = run_saturation_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_saturation_test_with_url() {
    let result = run_saturation_test(Some("http://localhost:8080".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_saturation_test_with_model() {
    let result = run_saturation_test(None, Some("model.gguf".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_saturation_test_with_output() {
    let output_path = std::env::temp_dir().join("saturation_test_output.json");
    let result = run_saturation_test(
        Some("test".to_string()),
        Some("model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output_path);
}

// ===== run_bench_compare Tests =====

#[test]
fn test_cov_run_bench_compare_both_missing() {
    let result = run_bench_compare("/nonexistent/a.json", "/nonexistent/b.json", 5.0);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_compare_first_exists_second_missing() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_compare_first.json");

    let mut f = std::fs::File::create(&file1).expect("test");
    f.write_all(b"{}").expect("test");

    let result = run_bench_compare(
        file1.to_str().expect("test"),
        "/nonexistent/second.json",
        5.0,
    );

    let _ = std::fs::remove_file(&file1);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_compare_invalid_json() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_cmp_inv1.json");
    let file2 = dir.join("bench_cmp_inv2.json");

    let mut f1 = std::fs::File::create(&file1).expect("test");
    f1.write_all(b"{}").expect("test");
    let mut f2 = std::fs::File::create(&file2).expect("test");
    f2.write_all(b"{}").expect("test");

    // Empty JSON is invalid for FullBenchmarkResult - should error
    let result = run_bench_compare(
        file1.to_str().expect("test"),
        file2.to_str().expect("test"),
        10.0,
    );

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);
    // Invalid JSON format should return error
    assert!(result.is_err());
}

// ===== run_bench_regression Tests =====

#[test]
fn test_cov_run_bench_regression_both_missing() {
    let result = run_bench_regression(
        "/nonexistent/baseline.json",
        "/nonexistent/current.json",
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_invalid_json_strict() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_regress_inv_strict.json");
    let current = dir.join("bench_regress_inv_curr.json");

    let mut f1 = std::fs::File::create(&baseline).expect("test");
    f1.write_all(b"{}").expect("test");
    let mut f2 = std::fs::File::create(&current).expect("test");
    f2.write_all(b"{}").expect("test");

    // Empty JSON is invalid - should error
    let result = run_bench_regression(
        baseline.to_str().expect("test"),
        current.to_str().expect("test"),
        true, // strict mode
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_invalid_json_non_strict() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_regress_inv_non.json");
    let current = dir.join("bench_regress_inv_curr_non.json");

    let mut f1 = std::fs::File::create(&baseline).expect("test");
    f1.write_all(b"{}").expect("test");
    let mut f2 = std::fs::File::create(&current).expect("test");
    f2.write_all(b"{}").expect("test");

    // Empty JSON is invalid - should error
    let result = run_bench_regression(
        baseline.to_str().expect("test"),
        current.to_str().expect("test"),
        false, // non-strict mode
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);
    assert!(result.is_err());
}

// ===== Edge Cases =====

#[test]
fn test_cov_format_size_just_under_kb() {
    assert_eq!(format_size(1023), "1023 B");
}

#[test]
fn test_cov_format_size_just_under_mb() {
    // 1024 * 1024 - 1 = 1048575
    let result = format_size(1048575);
    assert!(result.contains("KB"));
}

#[test]
fn test_cov_format_size_just_under_gb() {
    // 1024 * 1024 * 1024 - 1
    let result = format_size(1073741823);
    assert!(result.contains("MB"));
}

// ===== load_gguf_model Tests =====

#[test]
fn test_cov_load_gguf_model_invalid_magic() {
    // Invalid magic bytes
    let data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let result = realizar::cli::load_gguf_model(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_load_gguf_model_truncated_header() {
    // GGUF magic but truncated after magic
    // GGUF magic is "GGUF" (0x47, 0x47, 0x55, 0x46)
    let data = vec![0x47, 0x47, 0x55, 0x46]; // Just magic, no version
    let result = realizar::cli::load_gguf_model(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_load_gguf_model_empty() {
    let result = realizar::cli::load_gguf_model(&[]);
    assert!(result.is_err());
}

// ===== load_safetensors_model Tests =====

#[test]
fn test_cov_load_safetensors_model_invalid() {
    // Random bytes - not valid safetensors
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
    let result = realizar::cli::load_safetensors_model(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_load_safetensors_model_empty() {
    let result = realizar::cli::load_safetensors_model(&[]);
    assert!(result.is_err());
}

#[test]
fn test_cov_load_safetensors_model_small_header_invalid_json() {
    // SafeTensors header size (small) followed by invalid JSON
    let mut data = vec![0u8; 32];
    // Header size = 16 bytes (reasonable)
    data[0..8].copy_from_slice(&16u64.to_le_bytes());
    // Invalid JSON content after header size
    data[8..24].copy_from_slice(b"not valid json!!");
    let result = realizar::cli::load_safetensors_model(&data);
    assert!(result.is_err());
}

// ===== load_apr_model Tests =====

#[test]
fn test_cov_load_apr_model_valid_linear_regression() {
    // APR magic is "APR\0"
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version
    let result = realizar::cli::load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_cov_load_apr_model_valid_decision_tree() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0003u16.to_le_bytes()); // DecisionTree
    data[6..8].copy_from_slice(&2u16.to_le_bytes()); // version 2
    let result = realizar::cli::load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_cov_load_apr_model_wrong_magic() {
    let data = b"GGUF\x00\x00\x00\x00\x00\x00\x00\x00"; // GGUF magic instead
    let result = realizar::cli::load_apr_model(data);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Expected APR format"));
}

// ===== run_external_benchmark (stub) Tests =====

// Note: Without the bench-http feature, this should return an error
// This test validates the stub behavior
#[test]
#[cfg(not(feature = "bench-http"))]
fn test_cov_run_benchmarks_external_without_feature() {
    // When calling with runtime and url but without bench-http feature,
    // it should error with UnsupportedOperation
    let result = run_benchmarks(
        None,
        false, // not list mode
        Some("ollama".to_string()),
        None,
        Some("http://localhost:11434".to_string()),
        None,
    );
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("bench-http"));
}

// ===== display_model_info Additional Tests =====

#[test]
fn test_cov_display_model_info_apr_valid() {
    // Valid APR with magic
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = display_model_info("model.apr", &data);
    assert!(result.is_ok());
}

#[test]
fn test_cov_display_model_info_gguf_extension_invalid_data() {
    // .gguf extension but random data (not GGUF magic)
    let data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let result = display_model_info("model.gguf", &data);
    assert!(result.is_err());
}

#[test]
fn test_cov_display_model_info_safetensors_extension_invalid_data() {
    // .safetensors extension with small header size but invalid JSON after
    let mut data = vec![0u8; 32];
    // Small header size (16 bytes)
    data[0..8].copy_from_slice(&16u64.to_le_bytes());
    // Invalid JSON
    data[8..24].copy_from_slice(b"not valid json!!");
    let result = display_model_info("model.safetensors", &data);
    assert!(result.is_err());
}

#[test]
fn test_cov_display_model_info_unknown_extension_unknown_data() {
    // Unknown extension and unknown data - should show "Unknown" and succeed
    let data = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let result = display_model_info("model.xyz", &data);
    assert!(result.is_ok());
}

// ===== run_visualization Additional Edge Cases =====

#[test]
fn test_cov_run_visualization_zero_samples() {
    // Zero samples edge case
    run_visualization(false, 0);
}

#[test]
fn test_cov_run_visualization_single_sample() {
    run_visualization(true, 1);
}

#[test]
fn test_cov_run_visualization_many_samples() {
    run_visualization(false, 500);
}

// ===== run_benchmarks Additional Tests =====

#[test]
fn test_cov_run_benchmarks_list_with_url_only() {
    // List mode with url but no runtime
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

// ===== Convoy Test Additional Edge Cases =====

#[test]
fn test_cov_run_convoy_test_all_none() {
    let result = run_convoy_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_convoy_test_with_all_params() {
    let output_path = std::env::temp_dir().join("convoy_all_params.json");
    let result = run_convoy_test(
        Some("realizar".to_string()),
        Some("test-model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    // Verify output file was created
    assert!(output_path.exists());
    let _ = std::fs::remove_file(&output_path);
}

// ===== Saturation Test Additional Edge Cases =====

#[test]
fn test_cov_run_saturation_test_all_none() {
    let result = run_saturation_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_saturation_test_with_all_params() {
    let output_path = std::env::temp_dir().join("saturation_all_params.json");
    let result = run_saturation_test(
        Some("realizar".to_string()),
        Some("test-model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    assert!(output_path.exists());
    let _ = std::fs::remove_file(&output_path);
}

// ===== Bench Compare and Regression Additional Error Path Tests =====
// Note: Valid JSON schema tests are covered in src/bench.rs unit tests
// Here we test the error paths and file handling

#[test]
fn test_cov_run_bench_compare_with_zero_threshold() {
    // Test threshold edge case - both files missing
    let result = run_bench_compare("/nonexistent/a.json", "/nonexistent/b.json", 0.0);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_compare_with_large_threshold() {
    // Test threshold edge case - both files missing
    let result = run_bench_compare("/nonexistent/a.json", "/nonexistent/b.json", 100.0);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_compare_files_exist_but_invalid_schema() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_cmp_invalid_schema1.json");
    let file2 = dir.join("bench_cmp_invalid_schema2.json");

    // Valid JSON but wrong schema
    let json = r#"{"foo": "bar"}"#;

    let mut f1 = std::fs::File::create(&file1).expect("create file1");
    f1.write_all(json.as_bytes()).expect("write file1");

    let mut f2 = std::fs::File::create(&file2).expect("create file2");
    f2.write_all(json.as_bytes()).expect("write file2");

    let result = run_bench_compare(
        file1.to_str().expect("path1"),
        file2.to_str().expect("path2"),
        5.0,
    );

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);

    // Should error because JSON doesn't match schema
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_files_exist_but_invalid_schema() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_regress_inv_schema_base.json");
    let current = dir.join("bench_regress_inv_schema_curr.json");

    // Valid JSON but wrong schema
    let json = r#"{"version": "1.0", "data": []}"#;

    let mut f1 = std::fs::File::create(&baseline).expect("create baseline");
    f1.write_all(json.as_bytes()).expect("write baseline");

    let mut f2 = std::fs::File::create(&current).expect("create current");
    f2.write_all(json.as_bytes()).expect("write current");

    let result = run_bench_regression(
        baseline.to_str().expect("baseline path"),
        current.to_str().expect("current path"),
        false,
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);

    // Should error because JSON doesn't match schema
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_strict_vs_non_strict_threshold() {
    // Test both strict and non-strict modes error on missing files
    let result_strict =
        run_bench_regression("/nonexistent/base.json", "/nonexistent/curr.json", true);
    assert!(result_strict.is_err());

    let result_non_strict =
        run_bench_regression("/nonexistent/base.json", "/nonexistent/curr.json", false);
    assert!(result_non_strict.is_err());
}

#[test]
fn test_cov_run_bench_compare_first_file_parse_error() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_cmp_first_parse_err.json");
    let file2 = dir.join("bench_cmp_second_ok.json");

    // First file has invalid JSON
    let json1 = r#"not valid json"#;
    let json2 = r#"{"test": true}"#;

    let mut f1 = std::fs::File::create(&file1).expect("create file1");
    f1.write_all(json1.as_bytes()).expect("write file1");

    let mut f2 = std::fs::File::create(&file2).expect("create file2");
    f2.write_all(json2.as_bytes()).expect("write file2");

    let result = run_bench_compare(
        file1.to_str().expect("path1"),
        file2.to_str().expect("path2"),
        5.0,
    );

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);

    // Should error because first file has invalid JSON
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_first_file_parse_error() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_regress_base_parse_err.json");
    let current = dir.join("bench_regress_curr_ok.json");

    // First file has invalid JSON
    let json1 = r#"{invalid"#;
    let json2 = r#"{"test": true}"#;

    let mut f1 = std::fs::File::create(&baseline).expect("create baseline");
    f1.write_all(json1.as_bytes()).expect("write baseline");

    let mut f2 = std::fs::File::create(&current).expect("create current");
    f2.write_all(json2.as_bytes()).expect("write current");

    let result = run_bench_regression(
        baseline.to_str().expect("baseline path"),
        current.to_str().expect("current path"),
        false,
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);

    // Should error because baseline file has invalid JSON
    assert!(result.is_err());
}

// ===== is_local_file_path Additional Edge Cases =====

#[test]
fn test_cov_is_local_file_path_empty_string() {
    assert!(!is_local_file_path(""));
}

#[test]
fn test_cov_is_local_file_path_just_dot() {
    // Just "." - doesn't match criteria
    assert!(!is_local_file_path("."));
}

#[test]
fn test_cov_is_local_file_path_just_slash() {
    // Just "/" - starts with /
    assert!(is_local_file_path("/"));
}

#[test]
fn test_cov_is_local_file_path_windows_style() {
    // Windows-style path with .gguf extension returns true because of .gguf ending
    // This is the actual function behavior
    assert!(is_local_file_path("C:\\Users\\model.gguf"));
    // Without .gguf extension - not local
    assert!(!is_local_file_path("C:\\Users\\model.bin"));
}

// ===== validate_suite_name Edge Cases =====

#[test]
fn test_cov_validate_suite_name_with_whitespace() {
    assert!(!validate_suite_name(" tensor_ops"));
    assert!(!validate_suite_name("tensor_ops "));
    assert!(!validate_suite_name(" tensor_ops "));
    assert!(!validate_suite_name("\ttensor_ops"));
    assert!(!validate_suite_name("tensor_ops\n"));
}

#[test]
fn test_cov_validate_suite_name_partial_match() {
    // Partial names should not match
    assert!(!validate_suite_name("tensor"));
    assert!(!validate_suite_name("ops"));
    assert!(!validate_suite_name("infer"));
    assert!(!validate_suite_name("cach"));
}

// ===== home_dir Edge Cases =====

#[test]
fn test_cov_home_dir_path_format() {
    if let Some(path) = home_dir() {
        // Home directory should not be empty
        assert!(!path.as_os_str().is_empty());
        // Should be a valid path string
        let path_str = path.to_string_lossy();
        assert!(!path_str.is_empty());
    }
}

// ===== BENCHMARK_SUITES Comprehensive Tests =====

#[test]
fn test_cov_benchmark_suites_count() {
    // Verify we have the expected number of suites
    assert!(
        BENCHMARK_SUITES.len() >= 7,
        "Should have at least 7 benchmark suites"
    );
}

#[test]
fn test_cov_benchmark_suites_specific_entries() {
    // Verify specific suites exist
    let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();

    assert!(names.contains(&"lambda"), "Should have lambda suite");
    assert!(
        names.contains(&"comparative"),
        "Should have comparative suite"
    );
}

// ===== print_info Tests =====

#[test]
fn test_cov_print_info_multiple_calls() {
    // Verify multiple calls don't cause issues
    print_info();
    print_info();
    print_info();
}
