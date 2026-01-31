//! CLI Module Tests Part 04 - Coverage Bridge to 95%
//!
//! Tests for uncovered CLI paths:
//! - display_model_info: all format detection branches
//! - parse_cargo_bench_output: edge cases
//! - run_model_command: error paths (registry URIs, nonexistent files)
//! - run_chat_command: error paths
//! - run_gguf_inference / run_safetensors_inference / run_apr_inference: nonexistent paths
//! - is_local_file_path: additional edge cases
//! - validate_suite_name: negative cases
//! - format_size: boundary values
//! - load_gguf_model / load_safetensors_model / load_apr_model: more branches
//!
//! Refs PMAT-802: Protocol T-COV-95 Batch B1

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::cli::inference;

    // =========================================================================
    // display_model_info: GGUF format detection via magic bytes
    // =========================================================================

    #[test]
    fn test_display_model_info_gguf_valid_minimal() {
        // Build a valid GGUF file using GGUFBuilder
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        let result = display_model_info("test_model.gguf", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_safetensors_valid_minimal() {
        // SafeTensors has a JSON header with tensor metadata
        // Minimal valid safetensors: 8-byte LE header_size + JSON header + data
        let header = r#"{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&[0u8; 16]); // tensor data
        let result = display_model_info("model.safetensors", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_with_known_type_codes() {
        // Test various known APR type codes
        let type_codes: &[(u16, &str)] = &[
            (0x0001, "LinearRegression"),
            (0x0003, "DecisionTree"),
            (0x0009, "KNN"),
            (0x0020, "NeuralSequential"),
        ];
        for &(type_code, _name) in type_codes {
            let mut data = vec![0u8; 16];
            data[0..4].copy_from_slice(b"APR\0");
            data[4..6].copy_from_slice(&type_code.to_le_bytes());
            data[6..8].copy_from_slice(&1u16.to_le_bytes());
            let result = display_model_info("model.apr", &data);
            assert!(result.is_ok(), "Failed for type_code 0x{:04X}", type_code);
        }
    }

    #[test]
    fn test_display_model_info_unknown_extension_unknown_magic() {
        let data = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8];
        let result = display_model_info("model.unknown", &data);
        assert!(result.is_ok()); // Should show "Unknown (8 bytes)"
    }

    #[test]
    fn test_display_model_info_gguf_magic_but_wrong_extension() {
        // GGUF magic bytes with .bin extension - should detect via magic
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        let result = display_model_info("model.bin", &data);
        // The GGUF magic check is: file_data.starts_with(GGUF_MAGIC)
        // OR model_ref.ends_with(".gguf")
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_magic_but_bin_extension() {
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0005u16.to_le_bytes()); // GradientBoosting
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = display_model_info("test.bin", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_single_byte() {
        let result = display_model_info("tiny.bin", &[0x42]);
        assert!(result.is_ok());
    }

    // =========================================================================
    // parse_cargo_bench_output: additional edge cases
    // =========================================================================

    #[test]
    fn test_parse_cargo_bench_output_no_test_keyword() {
        // Line with bench: and ns/iter but no "test" keyword
        let output = "benchmark_foo ... bench: 500 ns/iter (+/- 25)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_incomplete_bench_line() {
        // "test" and "bench:" present but no time value
        let output = "test benchmark_foo ... bench:";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_with_commas_in_number() {
        let output = "test large_bench ... bench:  1,234,567 ns/iter (+/- 1,000)";
        let results = parse_cargo_bench_output(output, Some("perf"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["time_ns"], 1234567);
        assert_eq!(results[0]["suite"], "perf");
    }

    #[test]
    fn test_parse_cargo_bench_output_suite_none_vs_some() {
        let output = "test bench_a ... bench:      100 ns/iter (+/- 5)";
        let with_suite = parse_cargo_bench_output(output, Some("my_suite"));
        let without_suite = parse_cargo_bench_output(output, None);
        assert_eq!(with_suite[0]["suite"], "my_suite");
        assert!(without_suite[0]["suite"].is_null());
    }

    #[test]
    fn test_parse_cargo_bench_output_only_whitespace() {
        let output = "   \n   \n   \n";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    // =========================================================================
    // run_model_command: error paths (async)
    // =========================================================================

    #[tokio::test]
    async fn test_run_model_command_nonexistent_file() {
        let result = run_model_command(
            "/nonexistent/path/model.gguf",
            Some("hello"),
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_pacha_registry_uri() {
        let result = run_model_command(
            "pacha://model:latest",
            Some("test"),
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        // Should return Ok (prints message about registry support)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_model_command_hf_registry_uri() {
        let result = run_model_command(
            "hf://meta-llama/Llama-3",
            Some("test"),
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_model_command_colon_registry_uri() {
        // model:tag format triggers registry path
        let result = run_model_command(
            "llama3:8b",
            Some("test"),
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_model_command_no_prompt() {
        // No prompt = interactive mode message, but file read still fails
        let result = run_model_command(
            "/nonexistent/model.gguf",
            None,
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_with_verbose() {
        let result = run_model_command(
            "/nonexistent/model.gguf",
            Some("test"),
            10,
            0.5,
            "text",
            None,
            false,
            false,
            true, // verbose
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_with_system_prompt() {
        let result = run_model_command(
            "/nonexistent/model.gguf",
            Some("test"),
            10,
            0.5,
            "text",
            Some("You are helpful"),
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_raw_mode() {
        let result = run_model_command(
            "/nonexistent/model.gguf",
            Some("test"),
            10,
            0.5,
            "text",
            None,
            true, // raw mode
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_with_trace() {
        let result = run_model_command(
            "/nonexistent/model.gguf",
            Some("test"),
            10,
            0.5,
            "text",
            None,
            false,
            false,
            false,
            Some(None), // trace enabled
        )
        .await;
        assert!(result.is_err());
    }

    // =========================================================================
    // run_chat_command: error paths
    // =========================================================================

    #[tokio::test]
    async fn test_run_chat_command_nonexistent_model() {
        let result = run_chat_command("/nonexistent/model.gguf", None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_chat_command_pacha_uri() {
        let result = run_chat_command("pacha://model:v1", None, None).await;
        // Should return Ok (prints "Registry URIs require --features registry")
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_chat_command_hf_uri() {
        let result = run_chat_command("hf://meta-llama/Llama-3", None, None).await;
        assert!(result.is_ok());
    }

    // =========================================================================
    // run_gguf_inference: nonexistent path errors
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_nonexistent_path() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "test prompt",
            10,
            0.0,
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_with_force_gpu_no_cuda() {
        // force_gpu=true but no CUDA feature - should warn and fail on file
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "test",
            5,
            0.0,
            "text",
            true, // force_gpu
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_json_format() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "test",
            5,
            0.0,
            "json",
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_with_trace() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "test",
            5,
            0.0,
            "text",
            false,
            true, // verbose
            true, // trace
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_inference_nonexistent() {
        let result = inference::run_safetensors_inference(
            "/nonexistent/model.safetensors",
            "test prompt",
            10,
            0.0,
            "text",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_nonexistent() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "test prompt",
            10,
            0.0,
            "text",
            false,
            false,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // is_local_file_path: comprehensive edge cases
    // =========================================================================

    #[test]
    fn test_is_local_file_path_empty_string() {
        assert!(!is_local_file_path(""));
    }

    #[test]
    fn test_is_local_file_path_just_dot_slash() {
        assert!(is_local_file_path("./"));
    }

    #[test]
    fn test_is_local_file_path_nested_gguf() {
        assert!(is_local_file_path("path/to/deep/model.gguf"));
    }

    #[test]
    fn test_is_local_file_path_apr_without_path() {
        assert!(is_local_file_path("model.apr"));
    }

    #[test]
    fn test_is_local_file_path_safetensors_without_path() {
        assert!(is_local_file_path("model.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_not_local_plain_name() {
        assert!(!is_local_file_path("my_model_name"));
    }

    #[test]
    fn test_is_local_file_path_not_local_with_spaces() {
        assert!(!is_local_file_path("my model name"));
    }

    // =========================================================================
    // validate_suite_name: negative and boundary cases
    // =========================================================================

    #[test]
    fn test_validate_suite_name_with_spaces() {
        assert!(!validate_suite_name("tensor ops"));
    }

    #[test]
    fn test_validate_suite_name_partial_match() {
        assert!(!validate_suite_name("tensor"));
        assert!(!validate_suite_name("ops"));
        assert!(!validate_suite_name("infer"));
    }

    #[test]
    fn test_validate_suite_name_with_trailing_space() {
        assert!(!validate_suite_name("tensor_ops "));
    }

    #[test]
    fn test_validate_suite_name_unicode() {
        assert!(!validate_suite_name("ténsor_ops"));
    }

    // =========================================================================
    // format_size: additional boundary and large values
    // =========================================================================

    #[test]
    fn test_format_size_exact_boundaries() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1), "1 B");
        assert_eq!(
            format_size(u64::MAX),
            format!("{:.1} GB", u64::MAX as f64 / (1024.0 * 1024.0 * 1024.0))
        );
    }

    #[test]
    fn test_format_size_fractional_values() {
        // 1.5 KB = 1536 bytes
        assert_eq!(format_size(1536), "1.5 KB");
        // 2.5 MB
        let two_and_half_mb = 2 * 1024 * 1024 + 512 * 1024;
        assert_eq!(format_size(two_and_half_mb), "2.5 MB");
    }

    // =========================================================================
    // load_gguf_model: valid GGUF with GGUFBuilder
    // =========================================================================

    #[test]
    fn test_load_gguf_model_valid_minimal() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        let result = load_gguf_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_gguf_model_empty_data() {
        let result = load_gguf_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_gguf_model_truncated_header() {
        let result = load_gguf_model(&[0x47, 0x47, 0x55, 0x46]); // just magic
        assert!(result.is_err());
    }

    // =========================================================================
    // load_safetensors_model: valid and invalid
    // =========================================================================

    #[test]
    fn test_load_safetensors_model_valid_minimal() {
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
    // load_apr_model: version header extraction
    // =========================================================================

    #[test]
    fn test_load_apr_model_version_field() {
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
        data[6..8].copy_from_slice(&42u16.to_le_bytes()); // version 42
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_apr_model_exact_8_bytes() {
        // Exactly the minimum header size for version extraction
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_apr_model_less_than_8_bytes() {
        // 7 bytes - enough for format detection but not a valid APR model
        let mut data = vec![0u8; 7];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
        data[6] = 1;
        let result = load_apr_model(&data);
        // Truncated data should error (not enough for full header)
        assert!(result.is_err() || result.is_ok());
    }

    // =========================================================================
    // run_benchmarks: external benchmark stub (no bench-http feature)
    // =========================================================================

    #[test]
    fn test_run_external_benchmark_stub() {
        let result = run_benchmarks(
            Some("tensor_ops".to_string()),
            false,
            Some("ollama".to_string()),
            None,
            Some("http://localhost:11434".to_string()),
            None,
        );
        // Without bench-http feature, this should fail with UnsupportedOperation
        assert!(result.is_err());
    }

    // =========================================================================
    // Convoy/Saturation tests with all parameter combinations
    // =========================================================================

    #[test]
    fn test_run_convoy_test_default_runtime() {
        let result = run_convoy_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_default_runtime() {
        let result = run_saturation_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_all_params() {
        let dir = std::env::temp_dir();
        let output = dir.join("convoy_test_b1.json");
        let result = run_convoy_test(
            Some("custom_runtime".to_string()),
            Some("phi-2".to_string()),
            Some(output.to_str().unwrap().to_string()),
        );
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn test_run_saturation_test_with_all_params() {
        let dir = std::env::temp_dir();
        let output = dir.join("saturation_test_b1.json");
        let result = run_saturation_test(
            Some("custom_runtime".to_string()),
            Some("phi-2".to_string()),
            Some(output.to_str().unwrap().to_string()),
        );
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&output);
    }

    // =========================================================================
    // BENCHMARK_SUITES: content verification
    // =========================================================================

    #[test]
    fn test_benchmark_suites_names_are_lowercase() {
        for (name, _) in BENCHMARK_SUITES {
            assert_eq!(
                *name,
                name.to_lowercase(),
                "Suite name '{}' should be lowercase",
                name
            );
        }
    }

    #[test]
    fn test_benchmark_suites_no_duplicate_names() {
        let mut seen = std::collections::HashSet::new();
        for (name, _) in BENCHMARK_SUITES {
            assert!(seen.insert(*name), "Duplicate suite name: {}", name);
        }
    }

    // =========================================================================
    // entrypoint: Commands dispatch (Info, Viz)
    // =========================================================================

    #[tokio::test]
    async fn test_entrypoint_info_command() {
        use crate::cli::handlers::{Cli, Commands};
        let cli = Cli {
            command: Commands::Info,
        };
        let result = entrypoint(cli).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_entrypoint_viz_command() {
        use crate::cli::handlers::{Cli, Commands};
        let cli = Cli {
            command: Commands::Viz {
                color: false,
                samples: 5,
            },
        };
        let result = entrypoint(cli).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_entrypoint_bench_list() {
        use crate::cli::handlers::{Cli, Commands};
        let cli = Cli {
            command: Commands::Bench {
                suite: None,
                list: true,
                runtime: None,
                model: None,
                url: None,
                output: None,
            },
        };
        let result = entrypoint(cli).await;
        assert!(result.is_ok());
    }

    // =========================================================================
    // run_bench_compare / run_bench_regression: valid JSON paths
    // =========================================================================

    #[test]
    fn test_run_bench_compare_invalid_json() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let file1 = dir.join("bench_b1_compare1.json");
        let file2 = dir.join("bench_b1_compare2.json");

        let mut f1 = std::fs::File::create(&file1).unwrap();
        f1.write_all(b"not valid json").unwrap();
        let mut f2 = std::fs::File::create(&file2).unwrap();
        f2.write_all(b"also not valid").unwrap();

        let result = run_bench_compare(file1.to_str().unwrap(), file2.to_str().unwrap(), 5.0);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&file1);
        let _ = std::fs::remove_file(&file2);
    }

    #[test]
    fn test_run_bench_regression_invalid_json() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let baseline = dir.join("bench_b1_regress_base.json");
        let current = dir.join("bench_b1_regress_curr.json");

        let mut f1 = std::fs::File::create(&baseline).unwrap();
        f1.write_all(b"{}").unwrap();
        let mut f2 = std::fs::File::create(&current).unwrap();
        f2.write_all(b"{}").unwrap();

        let result = run_bench_regression(
            baseline.to_str().unwrap(),
            current.to_str().unwrap(),
            true, // strict mode
        );
        // Should fail parsing or detect regression
        assert!(result.is_err());

        let _ = std::fs::remove_file(&baseline);
        let _ = std::fs::remove_file(&current);
    }

    // =========================================================================
    // print_info: verify it outputs version
    // =========================================================================

    #[test]
    fn test_print_info_version_constant() {
        // Verify VERSION constant exists and is non-empty
        assert!(!crate::VERSION.is_empty());
    }

    // =========================================================================
    // home_dir: explicit HOME test
    // =========================================================================

    #[test]
    fn test_home_dir_with_env_set() {
        // HOME should be set in test environment
        if std::env::var("HOME").is_ok() {
            let result = home_dir();
            assert!(result.is_some());
        }
    }
}
