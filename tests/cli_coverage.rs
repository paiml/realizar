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
