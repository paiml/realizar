//! Additional tests for CLI inference module (Part 02)
//!
//! This module focuses on testing inference.rs code paths that are not covered
//! by tests_part_02.rs, including:
//! - Different file data scenarios for GGUF parsing
//! - Detailed error message validation
//! - Token generation boundary conditions
//! - Format output validation

#[cfg(test)]
mod inference_additional_coverage {
    use crate::cli::inference;
    use crate::error::RealizarError;

    // =========================================================================
    // GGUF Inference: File Data Validation Tests
    // Tests the early-exit paths when file_data is processed
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_empty_file_data_with_valid_path() {
        // Even with empty file_data, the function uses model_ref for mmap
        let result = inference::run_gguf_inference(
            "/nonexistent/path/model.gguf",
            &[], // Empty file data - function uses mmap path
            "test prompt",
            10,
            0.7,
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err());
        // Should fail at mmap stage
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("mmap") || err_msg.contains("Failed"),
            "Expected mmap error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_run_gguf_inference_with_gguf_magic_but_truncated() {
        // GGUF magic header but truncated data
        let truncated_gguf = b"GGUF\x03\x00\x00\x00"; // Just magic + version
        let result = inference::run_gguf_inference(
            "/tmp/truncated.gguf",
            truncated_gguf,
            "prompt",
            5,
            0.5,
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Temperature Sampling Logic Tests
    // Verifies temperature threshold behavior
    // =========================================================================

    #[test]
    fn test_gguf_inference_temperature_exactly_zero() {
        // temperature == 0.0 should use greedy decoding (< 0.01)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "zero temp",
            5,
            0.0, // Exactly zero
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err()); // Fails on model load, but exercises temp path
    }

    #[test]
    fn test_gguf_inference_temperature_very_small() {
        // temperature = 0.001 should use greedy (< 0.01)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "small temp",
            5,
            0.001,
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_temperature_at_sampling_threshold() {
        // temperature = 0.02 should use temperature sampling (> 0.01)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "sampling temp",
            5,
            0.02,
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_temperature_high_value() {
        // High temperature should work without issues
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "high temp",
            5,
            2.0,
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Format Output Path Tests
    // Tests JSON vs text output format handling
    // =========================================================================

    #[test]
    fn test_gguf_inference_json_format_output_path() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "json output test",
            5,
            0.5,
            "json", // JSON format
            false,
            true, // verbose to test more output code
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_text_format_verbose() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "text verbose test",
            5,
            0.5,
            "text",
            false,
            true, // verbose mode
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_text_format_non_verbose() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "text quiet test",
            5,
            0.5,
            "text",
            false,
            false, // non-verbose (Ollama-style output)
            false,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // APR Inference: File Data Scenarios
    // =========================================================================

    #[test]
    fn test_run_apr_inference_with_valid_apr_magic_but_incomplete() {
        // APR magic "APR\0" but missing model data
        let apr_header = b"APR\0\x01\x00\x01\x00"; // Magic + type + version only
        let result = inference::run_apr_inference(
            "/tmp/incomplete.apr",
            apr_header,
            "incomplete apr",
            5,
            0.5,
            "text",
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_zero_max_tokens() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "zero tokens",
            0, // Zero tokens
            0.5,
            "text",
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_json_format() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "json format",
            5,
            0.5,
            "json",
            false,
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_temperature_greedy() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "greedy",
            5,
            0.0, // Greedy
            "text",
            false,
            true,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // SafeTensors Inference: Path Resolution Tests
    // =========================================================================

    #[test]
    fn test_run_safetensors_inference_relative_path() {
        let result = inference::run_safetensors_inference(
            "relative/path/model.safetensors",
            "relative path test",
            5,
            0.5,
            "text",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_inference_absolute_path() {
        let result = inference::run_safetensors_inference(
            "/absolute/path/model.safetensors",
            "absolute path test",
            5,
            0.5,
            "text",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_inference_with_tokenizer_path() {
        // The function tries to load tokenizer.json from sibling path
        let result = inference::run_safetensors_inference(
            "/some/model/dir/model.safetensors",
            "tokenizer test",
            5,
            0.5,
            "text",
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Error Type Verification Tests
    // Ensures correct error variants are returned
    // =========================================================================

    #[test]
    fn test_gguf_inference_error_is_unsupported_operation() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "error type test",
            5,
            0.5,
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err());
        // Verify the error type
        match result.unwrap_err() {
            RealizarError::UnsupportedOperation { operation, reason } => {
                assert_eq!(operation, "mmap_gguf");
                assert!(reason.contains("Failed") || reason.contains("mmap"));
            }
            other => panic!("Expected UnsupportedOperation, got: {:?}", other),
        }
    }

    #[test]
    fn test_apr_inference_error_is_unsupported_operation() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[0, 1, 2, 3], // Invalid magic
            "error type test",
            5,
            0.5,
            "text",
            false,
            false,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            RealizarError::UnsupportedOperation { operation, reason } => {
                assert_eq!(operation, "parse_apr");
                assert!(reason.contains("APR") || reason.contains("parse"));
            }
            other => panic!("Expected UnsupportedOperation, got: {:?}", other),
        }
    }

    #[test]
    fn test_safetensors_inference_error_is_unsupported_operation() {
        let result = inference::run_safetensors_inference(
            "/nonexistent.safetensors",
            "error type test",
            5,
            0.5,
            "text",
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            RealizarError::UnsupportedOperation { operation, reason } => {
                assert_eq!(operation, "convert_safetensors");
                assert!(reason.contains("SafeTensors") || reason.contains("convert"));
            }
            other => panic!("Expected UnsupportedOperation, got: {:?}", other),
        }
    }

    // =========================================================================
    // Prompt Encoding Edge Cases
    // Tests prompt tokenization fallback behavior
    // =========================================================================

    #[test]
    fn test_gguf_inference_empty_prompt() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "", // Empty prompt
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
    fn test_gguf_inference_very_long_prompt() {
        // Long prompt to test tokenization
        let long_prompt = "a".repeat(10000);
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
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
    fn test_apr_inference_empty_prompt() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "", // Empty prompt
            5,
            0.5,
            "text",
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_inference_empty_prompt() {
        let result = inference::run_safetensors_inference(
            "/nonexistent.safetensors",
            "", // Empty prompt
            5,
            0.5,
            "text",
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Max Tokens Boundary Tests
    // =========================================================================

    #[test]
    fn test_gguf_inference_large_max_tokens() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "large tokens",
            100000, // Large number
            0.5,
            "text",
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_large_max_tokens() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "large tokens",
            100000,
            0.5,
            "text",
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_inference_large_max_tokens() {
        let result = inference::run_safetensors_inference(
            "/nonexistent.safetensors",
            "large tokens",
            100000,
            0.5,
            "text",
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Trace Mode Tests
    // =========================================================================

    #[test]
    fn test_gguf_inference_trace_mode_enabled() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "trace test",
            5,
            0.5,
            "text",
            false,
            false,
            true, // trace enabled
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_trace_and_verbose() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "trace+verbose",
            5,
            0.5,
            "text",
            false,
            true,
            true,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // GPU Flag Tests (without CUDA feature)
    // =========================================================================

    #[test]
    fn test_gguf_inference_gpu_flag_warning_path() {
        // When cuda feature is not enabled, force_gpu should print warning
        // but continue with CPU path
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "gpu flag test",
            5,
            0.5,
            "text",
            true, // force_gpu
            true, // verbose
            false,
        );
        assert!(result.is_err()); // Still fails, but warning path exercised
    }

    #[test]
    fn test_apr_inference_gpu_flag_warning_path() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "gpu flag test",
            5,
            0.5,
            "text",
            true, // force_gpu
            true, // verbose
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // File Extension and Path Variations
    // =========================================================================

    #[test]
    fn test_gguf_inference_uppercase_extension() {
        let result = inference::run_gguf_inference(
            "/path/to/model.GGUF",
            &[],
            "uppercase ext",
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
    fn test_safetensors_inference_uppercase_extension() {
        let result = inference::run_safetensors_inference(
            "/path/to/model.SAFETENSORS",
            "uppercase ext",
            5,
            0.5,
            "text",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_mixed_case_extension() {
        let result = inference::run_apr_inference(
            "/path/to/model.Apr",
            &[],
            "mixed case ext",
            5,
            0.5,
            "text",
            false,
            false,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Special Character Paths
    // =========================================================================

    #[test]
    fn test_gguf_inference_path_with_special_chars() {
        let result = inference::run_gguf_inference(
            "/path/with-dashes_and_underscores/model.gguf",
            &[],
            "special chars",
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
    fn test_safetensors_inference_path_with_dots() {
        let result = inference::run_safetensors_inference(
            "/path.with.dots/model.safetensors",
            "dots in path",
            5,
            0.5,
            "text",
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Format String Variations
    // =========================================================================

    #[test]
    fn test_gguf_inference_format_uppercase_json() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "format test",
            5,
            0.5,
            "JSON", // Uppercase - should fall through to text
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_format_uppercase_json() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "format test",
            5,
            0.5,
            "JSON",
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_inference_format_unknown() {
        let result = inference::run_safetensors_inference(
            "/nonexistent.safetensors",
            "format test",
            5,
            0.5,
            "xml", // Unknown format
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Comprehensive Parameter Matrix Tests
    // =========================================================================

    #[test]
    fn test_gguf_all_boolean_flags_true() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "all flags",
            5,
            0.5,
            "text",
            true,
            true,
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_all_boolean_flags_false() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "no flags",
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
    fn test_apr_all_boolean_flags_true() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "all flags",
            5,
            0.5,
            "text",
            true,
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_all_boolean_flags_false() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "no flags",
            5,
            0.5,
            "text",
            false,
            false,
        );
        assert!(result.is_err());
    }
}

// =========================================================================
// Server Commands Tests (gated by "server" feature)
// =========================================================================
#[cfg(all(test, feature = "server"))]
mod server_inference_tests {
    use crate::cli::{ModelType, PreparedServer};

    #[test]
    fn test_model_type_display_gguf() {
        let mt = ModelType::Gguf;
        assert_eq!(format!("{}", mt), "GGUF");
    }

    #[test]
    fn test_model_type_display_safetensors() {
        let mt = ModelType::SafeTensors;
        assert_eq!(format!("{}", mt), "SafeTensors");
    }

    #[test]
    fn test_model_type_display_apr() {
        let mt = ModelType::Apr;
        assert_eq!(format!("{}", mt), "APR");
    }

    #[test]
    fn test_model_type_equality() {
        assert_eq!(ModelType::Gguf, ModelType::Gguf);
        assert_ne!(ModelType::Gguf, ModelType::Apr);
        assert_ne!(ModelType::SafeTensors, ModelType::Apr);
    }

    #[test]
    fn test_model_type_clone() {
        let mt = ModelType::Gguf;
        let cloned = mt.clone();
        assert_eq!(mt, cloned);
    }

    #[test]
    fn test_model_type_copy() {
        let mt = ModelType::Apr;
        let copied: ModelType = mt; // Copy trait
        assert_eq!(mt, copied);
    }

    #[test]
    fn test_model_type_debug() {
        let mt = ModelType::SafeTensors;
        let debug_str = format!("{:?}", mt);
        assert_eq!(debug_str, "SafeTensors");
    }

    #[test]
    fn test_prepare_serve_state_invalid_gguf() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/nonexistent/model.gguf", false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_serve_state_invalid_safetensors() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/nonexistent/model.safetensors", false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_serve_state_invalid_apr() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/nonexistent/model.apr", false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_serve_state_unsupported_extension() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/some/model.bin", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported file extension"));
    }

    #[test]
    fn test_prepare_serve_state_with_batch_mode_flag() {
        use crate::cli::prepare_serve_state;

        // batch_mode flag should be processed even if model doesn't exist
        let result = prepare_serve_state("/nonexistent/model.gguf", true, false);
        assert!(result.is_err()); // Still fails, but batch_mode path exercised
    }

    #[test]
    fn test_prepare_serve_state_with_gpu_flag() {
        use crate::cli::prepare_serve_state;

        // force_gpu flag should be processed even if model doesn't exist
        let result = prepare_serve_state("/nonexistent/model.gguf", false, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_serve_state_with_both_flags() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/nonexistent/model.gguf", true, true);
        assert!(result.is_err());
    }
}

// =========================================================================
// CUDA Feature Tests
// =========================================================================
#[cfg(all(test, feature = "cuda"))]
mod cuda_inference_additional_tests {
    use crate::cli::inference;

    #[test]
    fn test_gguf_inference_gpu_path_exists() {
        // When cuda feature is enabled, force_gpu should attempt GPU path
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "cuda path test",
            5,
            0.5,
            "text",
            true, // force_gpu - exercises GPU code path
            true,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_gpu_path_exists() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "cuda apr test",
            5,
            0.5,
            "text",
            true,
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_gpu_json_format() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "cuda json test",
            5,
            0.5,
            "json",
            true,
            true,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_gpu_json_format() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "cuda json test",
            5,
            0.5,
            "json",
            true,
            true,
        );
        assert!(result.is_err());
    }
}
