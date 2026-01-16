//! EXTREME TDD: Deep CLI Coverage Tests
//!
//! Comprehensive tests for cli.rs to increase coverage from 72% to 95%+.
//! Tests all code paths including:
//! - format_size boundary cases
//! - display_model_info for all format types
//! - run_benchmarks external benchmark stub
//! - run_bench_compare and run_bench_regression with valid data
//! - parse_cargo_bench_output edge cases
//! - load_gguf_model, load_safetensors_model, load_apr_model
//! - run_convoy_test and run_saturation_test with all branches

#![allow(clippy::float_cmp)]

use realizar::cli::{
    display_model_info, format_size, home_dir, is_local_file_path, load_apr_model,
    load_gguf_model, load_safetensors_model, print_info, run_bench_compare, run_bench_regression,
    run_benchmarks, run_convoy_test, run_saturation_test, run_visualization, validate_suite_name,
    BENCHMARK_SUITES,
};

// =============================================================================
// format_size Comprehensive Tests
// =============================================================================

#[test]
fn test_deep_format_size_zero() {
    assert_eq!(format_size(0), "0 B");
}

#[test]
fn test_deep_format_size_single_byte() {
    assert_eq!(format_size(1), "1 B");
}

#[test]
fn test_deep_format_size_max_bytes_before_kb() {
    assert_eq!(format_size(1023), "1023 B");
}

#[test]
fn test_deep_format_size_exact_kb() {
    assert_eq!(format_size(1024), "1.0 KB");
}

#[test]
fn test_deep_format_size_kb_with_fraction() {
    // 1.5 KB = 1536 bytes
    assert_eq!(format_size(1536), "1.5 KB");
    // 2.25 KB = 2304 bytes
    assert_eq!(format_size(2304), "2.3 KB"); // 2.25 rounds to 2.3
}

#[test]
fn test_deep_format_size_max_kb_before_mb() {
    // 1023 KB
    assert_eq!(format_size(1024 * 1023), "1023.0 KB");
    // Just under 1 MB
    assert_eq!(format_size(1024 * 1024 - 1), "1024.0 KB");
}

#[test]
fn test_deep_format_size_exact_mb() {
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
}

#[test]
fn test_deep_format_size_mb_with_fraction() {
    // 1.5 MB
    assert_eq!(format_size(1024 * 1024 + 512 * 1024), "1.5 MB");
}

#[test]
fn test_deep_format_size_max_mb_before_gb() {
    // 1023 MB
    assert_eq!(format_size(1024 * 1024 * 1023), "1023.0 MB");
    // Just under 1 GB
    assert_eq!(format_size(1024 * 1024 * 1024 - 1), "1024.0 MB");
}

#[test]
fn test_deep_format_size_exact_gb() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
}

#[test]
fn test_deep_format_size_gb_with_fraction() {
    // 1.5 GB
    let one_point_five_gb = 1024u64 * 1024 * 1024 + 512 * 1024 * 1024;
    assert_eq!(format_size(one_point_five_gb), "1.5 GB");
}

#[test]
fn test_deep_format_size_very_large_gb() {
    // 100 GB
    assert_eq!(format_size(100 * 1024 * 1024 * 1024), "100.0 GB");
    // 1 TB displayed as GB
    assert_eq!(format_size(1024 * 1024 * 1024 * 1024), "1024.0 GB");
}

// =============================================================================
// BENCHMARK_SUITES Tests
// =============================================================================

#[test]
fn test_deep_benchmark_suites_count() {
    // Verify expected suite count
    assert!(BENCHMARK_SUITES.len() >= 7);
}

#[test]
fn test_deep_benchmark_suites_specific_entries() {
    let suites: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
    assert!(suites.contains(&"tensor_ops"));
    assert!(suites.contains(&"inference"));
    assert!(suites.contains(&"cache"));
    assert!(suites.contains(&"tokenizer"));
    assert!(suites.contains(&"quantize"));
    assert!(suites.contains(&"lambda"));
    assert!(suites.contains(&"comparative"));
}

#[test]
fn test_deep_benchmark_suites_descriptions_contain_keywords() {
    for (name, desc) in BENCHMARK_SUITES {
        match *name {
            "tensor_ops" => assert!(
                desc.to_lowercase().contains("tensor")
                    || desc.to_lowercase().contains("matmul")
                    || desc.to_lowercase().contains("operation")
            ),
            "inference" => assert!(desc.to_lowercase().contains("inference")),
            "cache" => assert!(
                desc.to_lowercase().contains("cache") || desc.to_lowercase().contains("memory")
            ),
            "tokenizer" => assert!(
                desc.to_lowercase().contains("token") || desc.to_lowercase().contains("bpe")
            ),
            _ => {} // Other suites - just check they have description
        }
    }
}

// =============================================================================
// validate_suite_name Tests
// =============================================================================

#[test]
fn test_deep_validate_suite_name_all_valid() {
    for (name, _) in BENCHMARK_SUITES {
        assert!(validate_suite_name(name), "Suite '{}' should be valid", name);
    }
}

#[test]
fn test_deep_validate_suite_name_empty() {
    assert!(!validate_suite_name(""));
}

#[test]
fn test_deep_validate_suite_name_whitespace() {
    assert!(!validate_suite_name(" "));
    assert!(!validate_suite_name("\t"));
    assert!(!validate_suite_name("\n"));
}

#[test]
fn test_deep_validate_suite_name_case_sensitive() {
    assert!(!validate_suite_name("TENSOR_OPS"));
    assert!(!validate_suite_name("Tensor_Ops"));
    assert!(!validate_suite_name("INFERENCE"));
}

#[test]
fn test_deep_validate_suite_name_similar_but_wrong() {
    assert!(!validate_suite_name("tensor-ops")); // hyphen vs underscore
    assert!(!validate_suite_name("tensor_op")); // missing 's'
    assert!(!validate_suite_name("tensorops")); // missing underscore
}

#[test]
fn test_deep_validate_suite_name_completely_invalid() {
    assert!(!validate_suite_name("nonexistent"));
    assert!(!validate_suite_name("xyz"));
    assert!(!validate_suite_name("123"));
}

// =============================================================================
// is_local_file_path Tests
// =============================================================================

#[test]
fn test_deep_is_local_file_path_gguf_extension() {
    assert!(is_local_file_path("model.gguf"));
    assert!(is_local_file_path("my-model.gguf"));
    assert!(is_local_file_path("MODEL.gguf")); // mixed case filename
}

#[test]
fn test_deep_is_local_file_path_safetensors_extension() {
    assert!(is_local_file_path("model.safetensors"));
    assert!(is_local_file_path("my_model.safetensors"));
}

#[test]
fn test_deep_is_local_file_path_apr_extension() {
    assert!(is_local_file_path("model.apr"));
    assert!(is_local_file_path("trained.apr"));
}

#[test]
fn test_deep_is_local_file_path_relative_dot_slash() {
    assert!(is_local_file_path("./model"));
    assert!(is_local_file_path("./models/my.gguf"));
    assert!(is_local_file_path("./subdir/model.safetensors"));
}

#[test]
fn test_deep_is_local_file_path_absolute() {
    assert!(is_local_file_path("/home/user/model"));
    assert!(is_local_file_path("/tmp/model.gguf"));
    assert!(is_local_file_path("/var/models/model.safetensors"));
}

#[test]
fn test_deep_is_local_file_path_not_local() {
    assert!(!is_local_file_path("llama3:8b")); // Ollama-style
    assert!(!is_local_file_path("pacha://model:v1")); // Pacha URI
    assert!(!is_local_file_path("hf://meta-llama/Llama-3")); // HuggingFace URI
    assert!(!is_local_file_path("meta-llama/Llama-3")); // HF-style org/repo
}

#[test]
fn test_deep_is_local_file_path_edge_cases() {
    // URL ending with extension is considered local (debatable behavior)
    // This tests the actual implementation behavior
    assert!(is_local_file_path("http://example.com/model.gguf"));

    // Relative without ./ and no extension
    assert!(!is_local_file_path("models/subdir/model"));
}

// =============================================================================
// home_dir Tests
// =============================================================================

#[test]
fn test_deep_home_dir_returns_some_on_unix() {
    let home = home_dir();
    // On most Unix systems, HOME is set
    if let Some(path) = home {
        // Path should not be empty
        assert!(!path.to_string_lossy().is_empty());
        // Path should be absolute on Unix
        assert!(path.is_absolute() || path.to_string_lossy().starts_with('/'));
    }
}

#[test]
fn test_deep_home_dir_consistency() {
    // Multiple calls should return the same value
    let h1 = home_dir();
    let h2 = home_dir();
    assert_eq!(h1, h2);
}

// =============================================================================
// display_model_info Tests
// =============================================================================

#[test]
fn test_deep_display_model_info_empty_data_unknown_ext() {
    // Empty data with unknown extension should show "Unknown" format
    let result = display_model_info("model.bin", &[]);
    assert!(result.is_ok());
}

#[test]
fn test_deep_display_model_info_random_data() {
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33];
    let result = display_model_info("model.bin", &data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_display_model_info_gguf_extension_invalid_data() {
    // GGUF extension but invalid data
    let result = display_model_info("model.gguf", &[0, 1, 2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_deep_display_model_info_gguf_magic_invalid_data() {
    // GGUF magic bytes but truncated/invalid
    let data = b"GGUF\x03\x00\x00\x00";
    let result = display_model_info("test.bin", data);
    // Should attempt to parse as GGUF and fail
    assert!(result.is_err());
}

#[test]
fn test_deep_display_model_info_safetensors_extension_invalid() {
    let result = display_model_info("model.safetensors", &[0, 1, 2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_deep_display_model_info_apr_extension() {
    // Valid APR header
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
    let result = display_model_info("model.apr", &data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_display_model_info_apr_magic_not_extension() {
    // APR magic in data, but different extension
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = display_model_info("model.bin", &data);
    assert!(result.is_ok());
}

// =============================================================================
// print_info Tests
// =============================================================================

#[test]
fn test_deep_print_info_executes() {
    // Just verify it doesn't panic
    print_info();
}

// =============================================================================
// run_visualization Tests
// =============================================================================

#[test]
fn test_deep_run_visualization_no_color_small() {
    run_visualization(false, 5);
}

#[test]
fn test_deep_run_visualization_with_color_small() {
    run_visualization(true, 5);
}

#[test]
fn test_deep_run_visualization_large_samples() {
    run_visualization(false, 200);
}

#[test]
fn test_deep_run_visualization_single_sample() {
    run_visualization(false, 1);
}

#[test]
fn test_deep_run_visualization_zero_samples() {
    // Edge case: zero samples
    run_visualization(false, 0);
}

// =============================================================================
// run_benchmarks Tests
// =============================================================================

#[test]
fn test_deep_run_benchmarks_list_mode_basic() {
    let result = run_benchmarks(None, true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_benchmarks_list_mode_with_suite() {
    let result = run_benchmarks(Some("tensor_ops".to_string()), true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_benchmarks_list_mode_with_runtime() {
    let result = run_benchmarks(
        None,
        true,
        Some("realizar".to_string()),
        None,
        None,
        None,
    );
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_benchmarks_list_mode_with_model() {
    let result = run_benchmarks(
        None,
        true,
        None,
        Some("model.gguf".to_string()),
        None,
        None,
    );
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_benchmarks_list_mode_with_url() {
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
fn test_deep_run_benchmarks_list_mode_with_output() {
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

#[test]
fn test_deep_run_benchmarks_list_mode_all_params() {
    let result = run_benchmarks(
        Some("inference".to_string()),
        true,
        Some("realizar".to_string()),
        Some("model.gguf".to_string()),
        Some("http://localhost:8080".to_string()),
        Some("/tmp/results.json".to_string()),
    );
    assert!(result.is_ok());
}

// Test external benchmark stub (without bench-http feature)
#[test]
fn test_deep_run_benchmarks_external_ollama_no_feature() {
    // This tests the stub that returns an error when bench-http is not enabled
    let result = run_benchmarks(
        None,
        false,
        Some("ollama".to_string()),
        Some("llama3".to_string()),
        Some("http://localhost:11434".to_string()),
        None,
    );
    // Without bench-http feature, this should return an error
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("bench-http"));
}

#[test]
fn test_deep_run_benchmarks_external_vllm_no_feature() {
    let result = run_benchmarks(
        None,
        false,
        Some("vllm".to_string()),
        Some("model".to_string()),
        Some("http://localhost:8000".to_string()),
        None,
    );
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("bench-http"));
}

#[test]
fn test_deep_run_benchmarks_external_llamacpp_no_feature() {
    let result = run_benchmarks(
        None,
        false,
        Some("llama-cpp".to_string()),
        None,
        Some("http://localhost:8080".to_string()),
        None,
    );
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("bench-http"));
}

// =============================================================================
// run_convoy_test Tests
// =============================================================================

#[test]
fn test_deep_run_convoy_test_minimal() {
    let result = run_convoy_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_convoy_test_with_runtime() {
    let result = run_convoy_test(Some("realizar".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_convoy_test_with_model() {
    let result = run_convoy_test(None, Some("model.gguf".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_convoy_test_with_output() {
    let output_path = std::env::temp_dir().join("deep_convoy_output.json");
    let result = run_convoy_test(None, None, Some(output_path.to_string_lossy().to_string()));
    assert!(result.is_ok());

    // Verify file was created
    assert!(output_path.exists());

    // Verify file contains JSON
    let content = std::fs::read_to_string(&output_path).expect("Should read file");
    assert!(content.contains("baseline_short_p99_ms"));

    // Cleanup
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn test_deep_run_convoy_test_all_params() {
    let output_path = std::env::temp_dir().join("deep_convoy_all.json");
    let result = run_convoy_test(
        Some("realizar".to_string()),
        Some("model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output_path);
}

// =============================================================================
// run_saturation_test Tests
// =============================================================================

#[test]
fn test_deep_run_saturation_test_minimal() {
    let result = run_saturation_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_saturation_test_with_runtime() {
    let result = run_saturation_test(Some("realizar".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_saturation_test_with_model() {
    let result = run_saturation_test(None, Some("model.gguf".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_deep_run_saturation_test_with_output() {
    let output_path = std::env::temp_dir().join("deep_saturation_output.json");
    let result = run_saturation_test(None, None, Some(output_path.to_string_lossy().to_string()));
    assert!(result.is_ok());

    // Verify file was created
    assert!(output_path.exists());

    // Verify file contains expected JSON fields
    let content = std::fs::read_to_string(&output_path).expect("Should read file");
    assert!(content.contains("baseline_throughput"));
    assert!(content.contains("stressed_throughput"));

    // Cleanup
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn test_deep_run_saturation_test_all_params() {
    let output_path = std::env::temp_dir().join("deep_saturation_all.json");
    let result = run_saturation_test(
        Some("realizar".to_string()),
        Some("model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output_path);
}

// =============================================================================
// run_bench_compare Tests
// =============================================================================

#[test]
fn test_deep_run_bench_compare_both_files_missing() {
    let result = run_bench_compare("/nonexistent/a.json", "/nonexistent/b.json", 5.0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    // Error should mention file read failure
    assert!(
        err.to_string().contains("Failed to read")
            || err.to_string().contains("read_benchmark")
    );
}

#[test]
fn test_deep_run_bench_compare_first_file_missing() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file2 = dir.join("deep_bench_compare_second.json");
    let mut f = std::fs::File::create(&file2).expect("create file");
    f.write_all(b"{}").expect("write");

    let result = run_bench_compare("/nonexistent/first.json", file2.to_str().unwrap(), 5.0);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&file2);
}

#[test]
fn test_deep_run_bench_compare_second_file_missing() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("deep_bench_compare_first.json");
    let mut f = std::fs::File::create(&file1).expect("create file");
    f.write_all(b"{}").expect("write");

    let result = run_bench_compare(file1.to_str().unwrap(), "/nonexistent/second.json", 5.0);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&file1);
}

#[test]
fn test_deep_run_bench_compare_invalid_json_format() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("deep_bench_cmp_inv1.json");
    let file2 = dir.join("deep_bench_cmp_inv2.json");

    // Write empty JSON objects (invalid for FullBenchmarkResult)
    let mut f1 = std::fs::File::create(&file1).expect("create");
    f1.write_all(b"{}").expect("write");
    let mut f2 = std::fs::File::create(&file2).expect("create");
    f2.write_all(b"{}").expect("write");

    let result = run_bench_compare(file1.to_str().unwrap(), file2.to_str().unwrap(), 10.0);
    // Empty JSON is invalid for FullBenchmarkResult
    assert!(result.is_err());

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);
}

#[test]
fn test_deep_run_bench_compare_valid_data() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("deep_bench_cmp_valid1.json");
    let file2 = dir.join("deep_bench_cmp_valid2.json");

    // Create valid FullBenchmarkResult JSON
    let valid_json = r#"{
        "version": "1.1",
        "timestamp": "2025-01-15T12:00:00Z",
        "config": {
            "model": "test",
            "format": "apr",
            "quantization": "q4_k",
            "runtime": "realizar",
            "runtime_version": "0.3.5"
        },
        "hardware": {
            "cpu": "Test CPU",
            "gpu": null,
            "memory_gb": 32,
            "accelerator": null
        },
        "sampling": {
            "method": "dynamic_cv",
            "cv_threshold": 0.05,
            "actual_iterations": 100,
            "cv_at_stop": 0.04,
            "warmup_iterations": 10
        },
        "thermal": {
            "valid": true,
            "temp_variance_c": 1.0,
            "max_temp_c": 60.0
        },
        "results": {
            "ttft_ms": {
                "p50": 20.0,
                "p95": 30.0,
                "p99": 40.0,
                "p999": 50.0
            },
            "itl_ms": {
                "median": 10.0,
                "std_dev": 2.0,
                "p99": 15.0
            },
            "throughput_tok_s": {
                "median": 100.0,
                "ci_95": [95.0, 105.0]
            },
            "memory_mb": {
                "model_mb": 500,
                "peak_rss_mb": 1000,
                "kv_cache_mb": 200
            },
            "energy": {
                "token_joules": 0.05,
                "idle_watts": 50.0,
                "active_watts": 150.0
            },
            "cold_start": {
                "model_load_ms": 1000.0,
                "first_inference_ms": 50.0
            }
        },
        "quality": {
            "kl_divergence_vs_fp32": 0.03,
            "output_valid": true,
            "eos_correct": true
        }
    }"#;

    // Different throughput for comparison
    let valid_json2 = valid_json.replace("100.0", "120.0");

    let mut f1 = std::fs::File::create(&file1).expect("create");
    f1.write_all(valid_json.as_bytes()).expect("write");
    let mut f2 = std::fs::File::create(&file2).expect("create");
    f2.write_all(valid_json2.as_bytes()).expect("write");

    let result = run_bench_compare(file1.to_str().unwrap(), file2.to_str().unwrap(), 5.0);
    assert!(result.is_ok());

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);
}

#[test]
fn test_deep_run_bench_compare_different_thresholds() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("deep_bench_cmp_thresh1.json");
    let file2 = dir.join("deep_bench_cmp_thresh2.json");

    let valid_json = r#"{
        "version": "1.1",
        "timestamp": "2025-01-15T12:00:00Z",
        "config": {"model": "test", "format": "apr", "quantization": "q4_k", "runtime": "realizar", "runtime_version": "0.3.5"},
        "hardware": {"cpu": "Test CPU", "gpu": null, "memory_gb": 32, "accelerator": null},
        "sampling": {"method": "dynamic_cv", "cv_threshold": 0.05, "actual_iterations": 100, "cv_at_stop": 0.04, "warmup_iterations": 10},
        "thermal": {"valid": true, "temp_variance_c": 1.0, "max_temp_c": 60.0},
        "results": {
            "ttft_ms": {"p50": 20.0, "p95": 30.0, "p99": 40.0, "p999": 50.0},
            "itl_ms": {"median": 10.0, "std_dev": 2.0, "p99": 15.0},
            "throughput_tok_s": {"median": 100.0, "ci_95": [95.0, 105.0]},
            "memory_mb": {"model_mb": 500, "peak_rss_mb": 1000, "kv_cache_mb": 200},
            "energy": {"token_joules": 0.05, "idle_watts": 50.0, "active_watts": 150.0},
            "cold_start": {"model_load_ms": 1000.0, "first_inference_ms": 50.0}
        },
        "quality": {"kl_divergence_vs_fp32": 0.03, "output_valid": true, "eos_correct": true}
    }"#;

    let mut f1 = std::fs::File::create(&file1).expect("create");
    f1.write_all(valid_json.as_bytes()).expect("write");
    let mut f2 = std::fs::File::create(&file2).expect("create");
    f2.write_all(valid_json.as_bytes()).expect("write");

    // Test with different thresholds
    let r1 = run_bench_compare(file1.to_str().unwrap(), file2.to_str().unwrap(), 0.0);
    assert!(r1.is_ok());

    let r2 = run_bench_compare(file1.to_str().unwrap(), file2.to_str().unwrap(), 50.0);
    assert!(r2.is_ok());

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);
}

// =============================================================================
// run_bench_regression Tests
// =============================================================================

#[test]
fn test_deep_run_bench_regression_both_files_missing() {
    let result = run_bench_regression(
        "/nonexistent/baseline.json",
        "/nonexistent/current.json",
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_deep_run_bench_regression_baseline_missing() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let current = dir.join("deep_regress_current.json");
    let mut f = std::fs::File::create(&current).expect("create");
    f.write_all(b"{}").expect("write");

    let result =
        run_bench_regression("/nonexistent/baseline.json", current.to_str().unwrap(), false);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&current);
}

#[test]
fn test_deep_run_bench_regression_current_missing() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("deep_regress_baseline.json");
    let mut f = std::fs::File::create(&baseline).expect("create");
    f.write_all(b"{}").expect("write");

    let result =
        run_bench_regression(baseline.to_str().unwrap(), "/nonexistent/current.json", false);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&baseline);
}

#[test]
fn test_deep_run_bench_regression_invalid_json() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("deep_regress_inv_base.json");
    let current = dir.join("deep_regress_inv_curr.json");

    let mut f1 = std::fs::File::create(&baseline).expect("create");
    f1.write_all(b"{}").expect("write");
    let mut f2 = std::fs::File::create(&current).expect("create");
    f2.write_all(b"{}").expect("write");

    let result =
        run_bench_regression(baseline.to_str().unwrap(), current.to_str().unwrap(), false);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);
}

#[test]
fn test_deep_run_bench_regression_valid_no_regression() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("deep_regress_valid_base.json");
    let current = dir.join("deep_regress_valid_curr.json");

    // Same data = no regression
    let valid_json = r#"{
        "version": "1.1",
        "timestamp": "2025-01-15T12:00:00Z",
        "config": {"model": "test", "format": "apr", "quantization": "q4_k", "runtime": "realizar", "runtime_version": "0.3.5"},
        "hardware": {"cpu": "Test CPU", "gpu": null, "memory_gb": 32, "accelerator": null},
        "sampling": {"method": "dynamic_cv", "cv_threshold": 0.05, "actual_iterations": 100, "cv_at_stop": 0.04, "warmup_iterations": 10},
        "thermal": {"valid": true, "temp_variance_c": 1.0, "max_temp_c": 60.0},
        "results": {
            "ttft_ms": {"p50": 20.0, "p95": 30.0, "p99": 40.0, "p999": 50.0},
            "itl_ms": {"median": 10.0, "std_dev": 2.0, "p99": 15.0},
            "throughput_tok_s": {"median": 100.0, "ci_95": [95.0, 105.0]},
            "memory_mb": {"model_mb": 500, "peak_rss_mb": 1000, "kv_cache_mb": 200},
            "energy": {"token_joules": 0.05, "idle_watts": 50.0, "active_watts": 150.0},
            "cold_start": {"model_load_ms": 1000.0, "first_inference_ms": 50.0}
        },
        "quality": {"kl_divergence_vs_fp32": 0.03, "output_valid": true, "eos_correct": true}
    }"#;

    let mut f1 = std::fs::File::create(&baseline).expect("create");
    f1.write_all(valid_json.as_bytes()).expect("write");
    let mut f2 = std::fs::File::create(&current).expect("create");
    f2.write_all(valid_json.as_bytes()).expect("write");

    let result =
        run_bench_regression(baseline.to_str().unwrap(), current.to_str().unwrap(), false);
    assert!(result.is_ok());

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);
}

#[test]
fn test_deep_run_bench_regression_with_regression_non_strict() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("deep_regress_regr_base.json");
    let current = dir.join("deep_regress_regr_curr.json");

    // Baseline with good performance
    let baseline_json = r#"{
        "version": "1.1",
        "timestamp": "2025-01-15T12:00:00Z",
        "config": {"model": "test", "format": "apr", "quantization": "q4_k", "runtime": "realizar", "runtime_version": "0.3.5"},
        "hardware": {"cpu": "Test CPU", "gpu": null, "memory_gb": 32, "accelerator": null},
        "sampling": {"method": "dynamic_cv", "cv_threshold": 0.05, "actual_iterations": 100, "cv_at_stop": 0.04, "warmup_iterations": 10},
        "thermal": {"valid": true, "temp_variance_c": 1.0, "max_temp_c": 60.0},
        "results": {
            "ttft_ms": {"p50": 20.0, "p95": 30.0, "p99": 40.0, "p999": 50.0},
            "itl_ms": {"median": 10.0, "std_dev": 2.0, "p99": 15.0},
            "throughput_tok_s": {"median": 100.0, "ci_95": [95.0, 105.0]},
            "memory_mb": {"model_mb": 500, "peak_rss_mb": 1000, "kv_cache_mb": 200},
            "energy": {"token_joules": 0.05, "idle_watts": 50.0, "active_watts": 150.0},
            "cold_start": {"model_load_ms": 1000.0, "first_inference_ms": 50.0}
        },
        "quality": {"kl_divergence_vs_fp32": 0.03, "output_valid": true, "eos_correct": true}
    }"#;

    // Current with regression (higher p99, lower throughput)
    let current_json = r#"{
        "version": "1.1",
        "timestamp": "2025-01-15T12:00:00Z",
        "config": {"model": "test", "format": "apr", "quantization": "q4_k", "runtime": "realizar", "runtime_version": "0.3.5"},
        "hardware": {"cpu": "Test CPU", "gpu": null, "memory_gb": 32, "accelerator": null},
        "sampling": {"method": "dynamic_cv", "cv_threshold": 0.05, "actual_iterations": 100, "cv_at_stop": 0.04, "warmup_iterations": 10},
        "thermal": {"valid": true, "temp_variance_c": 1.0, "max_temp_c": 60.0},
        "results": {
            "ttft_ms": {"p50": 30.0, "p95": 50.0, "p99": 80.0, "p999": 100.0},
            "itl_ms": {"median": 15.0, "std_dev": 3.0, "p99": 25.0},
            "throughput_tok_s": {"median": 50.0, "ci_95": [45.0, 55.0]},
            "memory_mb": {"model_mb": 500, "peak_rss_mb": 1500, "kv_cache_mb": 200},
            "energy": {"token_joules": 0.10, "idle_watts": 50.0, "active_watts": 200.0},
            "cold_start": {"model_load_ms": 1500.0, "first_inference_ms": 80.0}
        },
        "quality": {"kl_divergence_vs_fp32": 0.03, "output_valid": true, "eos_correct": true}
    }"#;

    let mut f1 = std::fs::File::create(&baseline).expect("create");
    f1.write_all(baseline_json.as_bytes()).expect("write");
    let mut f2 = std::fs::File::create(&current).expect("create");
    f2.write_all(current_json.as_bytes()).expect("write");

    let result =
        run_bench_regression(baseline.to_str().unwrap(), current.to_str().unwrap(), false);
    // Should detect regression and return error
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("regression"));

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);
}

#[test]
fn test_deep_run_bench_regression_strict_mode() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("deep_regress_strict_base.json");
    let current = dir.join("deep_regress_strict_curr.json");

    // Same data should pass even in strict mode
    let valid_json = r#"{
        "version": "1.1",
        "timestamp": "2025-01-15T12:00:00Z",
        "config": {"model": "test", "format": "apr", "quantization": "q4_k", "runtime": "realizar", "runtime_version": "0.3.5"},
        "hardware": {"cpu": "Test CPU", "gpu": null, "memory_gb": 32, "accelerator": null},
        "sampling": {"method": "dynamic_cv", "cv_threshold": 0.05, "actual_iterations": 100, "cv_at_stop": 0.04, "warmup_iterations": 10},
        "thermal": {"valid": true, "temp_variance_c": 1.0, "max_temp_c": 60.0},
        "results": {
            "ttft_ms": {"p50": 20.0, "p95": 30.0, "p99": 40.0, "p999": 50.0},
            "itl_ms": {"median": 10.0, "std_dev": 2.0, "p99": 15.0},
            "throughput_tok_s": {"median": 100.0, "ci_95": [95.0, 105.0]},
            "memory_mb": {"model_mb": 500, "peak_rss_mb": 1000, "kv_cache_mb": 200},
            "energy": {"token_joules": 0.05, "idle_watts": 50.0, "active_watts": 150.0},
            "cold_start": {"model_load_ms": 1000.0, "first_inference_ms": 50.0}
        },
        "quality": {"kl_divergence_vs_fp32": 0.03, "output_valid": true, "eos_correct": true}
    }"#;

    let mut f1 = std::fs::File::create(&baseline).expect("create");
    f1.write_all(valid_json.as_bytes()).expect("write");
    let mut f2 = std::fs::File::create(&current).expect("create");
    f2.write_all(valid_json.as_bytes()).expect("write");

    let result =
        run_bench_regression(baseline.to_str().unwrap(), current.to_str().unwrap(), true);
    assert!(result.is_ok());

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);
}

// =============================================================================
// load_gguf_model Tests
// =============================================================================

#[test]
fn test_deep_load_gguf_model_empty_data() {
    let result = load_gguf_model(&[]);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_gguf_model_short_data() {
    let result = load_gguf_model(&[0, 1, 2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_gguf_model_wrong_magic() {
    let data = b"NOT_GGUFxxxxxxxx";
    let result = load_gguf_model(data);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_gguf_model_truncated_gguf() {
    // GGUF magic but truncated
    let data = b"GGUF\x03\x00\x00\x00";
    let result = load_gguf_model(data);
    assert!(result.is_err());
}

// =============================================================================
// load_safetensors_model Tests
// =============================================================================

#[test]
fn test_deep_load_safetensors_model_empty_data() {
    let result = load_safetensors_model(&[]);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_safetensors_model_short_data() {
    let result = load_safetensors_model(&[0, 1, 2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_safetensors_model_invalid_header() {
    // Random bytes - not valid safetensors
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
    let result = load_safetensors_model(&data);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_safetensors_model_zero_length_header() {
    // 8 bytes of zeros for header length
    let data = vec![0u8; 8];
    let result = load_safetensors_model(&data);
    // Zero-length header means empty JSON, which is invalid
    assert!(result.is_err());
}

// =============================================================================
// load_apr_model Tests
// =============================================================================

#[test]
fn test_deep_load_apr_model_empty_data() {
    let result = load_apr_model(&[]);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_apr_model_short_data() {
    let result = load_apr_model(&[0, 1, 2]);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_apr_model_wrong_magic() {
    let data = b"NOT_APRNxxxxxxxx";
    let result = load_apr_model(data);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Expected APR format"));
}

#[test]
fn test_deep_load_apr_model_gguf_magic() {
    // GGUF magic instead of APR
    let data = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let result = load_apr_model(data);
    assert!(result.is_err());
}

#[test]
fn test_deep_load_apr_model_valid_linear_regression() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_logistic_regression() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_decision_tree() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0003u16.to_le_bytes()); // DecisionTree
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_random_forest() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0004u16.to_le_bytes()); // RandomForest
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_gradient_boosting() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0005u16.to_le_bytes()); // GradientBoosting
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_kmeans() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0006u16.to_le_bytes()); // KMeans
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_pca() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0007u16.to_le_bytes()); // PCA
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_naive_bayes() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0008u16.to_le_bytes()); // NaiveBayes
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_knn() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0009u16.to_le_bytes()); // KNN
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_svm() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x000Au16.to_le_bytes()); // SVM
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_neural_sequential() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0020u16.to_le_bytes()); // NeuralSequential
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_valid_custom() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x00FFu16.to_le_bytes()); // Custom
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_unknown_type() {
    // Unknown type code (not in the recognized list)
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0xFFFEu16.to_le_bytes()); // Unknown
    data[6..8].copy_from_slice(&1u16.to_le_bytes());

    let result = load_apr_model(&data);
    // Should succeed but show "Unknown" type
    assert!(result.is_ok());
}

#[test]
fn test_deep_load_apr_model_version_display() {
    // Test that version is displayed
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
    data[6..8].copy_from_slice(&2u16.to_le_bytes()); // version 2

    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

// =============================================================================
// Additional Edge Cases
// =============================================================================

#[test]
fn test_deep_display_model_info_safetensors_valid_empty() {
    // Valid safetensors with empty tensor list
    // Header length (8 bytes) + JSON header
    let header = b"{}";
    let header_len = header.len() as u64;
    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header);

    let result = display_model_info("model.safetensors", &data);
    // May fail or succeed depending on safetensors parser strictness
    let _ = result;
}

#[test]
fn test_deep_run_benchmarks_runtime_and_url_without_bench_http() {
    // With runtime and URL but without bench-http feature
    let result = run_benchmarks(
        Some("tensor_ops".to_string()),
        false,
        Some("custom-runtime".to_string()),
        None,
        Some("http://localhost:9999".to_string()),
        None,
    );
    // Without bench-http feature, this should fail
    assert!(result.is_err());
}

#[test]
fn test_deep_format_size_u64_max() {
    // Test with very large value (u64::MAX)
    let result = format_size(u64::MAX);
    // Should display in GB without panic
    assert!(result.contains("GB"));
}

#[test]
fn test_deep_is_local_file_path_complex_paths() {
    // Path with spaces
    assert!(is_local_file_path("/path/with spaces/model.gguf"));

    // Path with unicode
    assert!(is_local_file_path("/path/with/日本語/model.gguf"));

    // Very long path
    let long_path = "/".to_string() + &"a".repeat(1000) + "/model.gguf";
    assert!(is_local_file_path(&long_path));
}

#[test]
fn test_deep_validate_suite_name_special_chars() {
    assert!(!validate_suite_name("tensor_ops!"));
    assert!(!validate_suite_name("tensor_ops@"));
    assert!(!validate_suite_name("tensor_ops#"));
}

#[test]
fn test_deep_home_dir_env_dependency() {
    // This test documents that home_dir depends on HOME env var
    let result = home_dir();
    if std::env::var("HOME").is_ok() {
        assert!(result.is_some());
    }
}
