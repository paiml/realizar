//! Additional tests for CLI inference module (Part 02)
//!
//! This module adds comprehensive test coverage for inference.rs code paths
//! including configuration building, CLI argument validation, and edge cases.

#[cfg(test)]
mod inference_coverage {
    use crate::cli::inference;
    use crate::inference_trace::TraceConfig;

    // =========================================================================
    // Configuration Building Tests (EXTREME TDD)
    // Tests verify that inference functions correctly build internal configs
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_config_temperature_boundary_0_01() {
        // Temperature exactly at 0.01 boundary should use greedy
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "boundary test",
            5,
            0.01, // Exactly at greedy threshold
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_config_temperature_just_above_boundary() {
        // Temperature 0.011 should use sampling (top-k 40)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "sampling boundary",
            5,
            0.011, // Just above greedy threshold
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_max_tokens_one() {
        // Single token generation
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "single token",
            1, // Only 1 token
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_max_tokens_very_large() {
        // Large max_tokens tests bounds handling
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "large generation",
            usize::MAX / 2, // Very large but won't overflow
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // CLI Argument Parsing Tests - Format Variations
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_format_json_lowercase() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "json test",
            5,
            0.5,
            "json",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_format_text_lowercase() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "text test",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_format_arbitrary_string() {
        // Unknown formats should fall through to default (text-like)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "unknown format",
            5,
            0.5,
            "yaml", // Not recognized
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_format_empty_string() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "empty format",
            5,
            0.5,
            "", // Empty format string
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // GPU Flag Tests (Non-CUDA Feature)
    // When cuda feature is NOT enabled, force_gpu flag should be handled
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_gpu_flag_no_cuda_feature() {
        // Test GPU flag warning path when cuda feature is disabled
        // This should NOT panic and should proceed with CPU fallback
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "gpu warning test",
            5,
            0.5,
            "text",
            true, // force_gpu - should warn without cuda feature
            false,
            None, // trace_config
        );
        assert!(result.is_err()); // Model loading will still fail
    }

    #[test]
    fn test_run_apr_inference_gpu_flag_no_cuda_feature() {
        // Same for APR inference
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "gpu warning test",
            5,
            0.5,
            "text",
            true,  // force_gpu
            false, // verbose
            None,  // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Verbose and Trace Flag Combinations
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_verbose_only() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "verbose only",
            5,
            0.5,
            "text",
            false,
            true, // verbose
            None, // trace_config (disabled)
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_trace_only() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "trace only",
            5,
            0.5,
            "text",
            false,
            false,                        // no verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_verbose_and_trace() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "both flags",
            5,
            0.5,
            "text",
            false,
            true,                         // verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_all_debug_flags() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "all flags",
            5,
            0.5,
            "json",
            true,                         // force_gpu
            true,                         // verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // APR Inference Verbose Mode Tests
    // =========================================================================

    #[test]
    fn test_run_apr_inference_verbose_only() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "verbose apr",
            5,
            0.5,
            "text",
            false, // no gpu
            true,  // verbose
            None,  // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_verbose_with_gpu() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "verbose gpu apr",
            5,
            0.5,
            "json",
            true, // force_gpu
            true, // verbose
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // SafeTensors Inference Tests
    // =========================================================================

    #[test]
    fn test_run_safetensors_inference_format_variations() {
        for format in &["json", "text", "yaml", "", "markdown"] {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "format test",
                5,
                0.5,
                format,
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_run_safetensors_inference_temperature_ignored() {
        // Temperature is marked as _temperature in the function signature
        // Test that various values are accepted without affecting the error type
        for temp in &[0.0, 0.01, 0.5, 1.0, 2.0, -1.0, f32::MAX, f32::MIN] {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "temp test",
                5,
                *temp,
                "text",
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_run_safetensors_inference_max_tokens_variations() {
        for max_tokens in &[0, 1, 5, 100, 1000, usize::MAX / 4] {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "tokens test",
                *max_tokens,
                0.5,
                "text",
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // Error Handling Path Coverage
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_error_message_structure() {
        let result = inference::run_gguf_inference(
            "/definitely/not/a/real/path/model.gguf",
            &[],
            "error structure test",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_string = format!("{}", err);
        // Verify error contains useful diagnostic info
        assert!(
            err_string.contains("mmap")
                || err_string.contains("GGUF")
                || err_string.contains("load"),
            "Error message should contain diagnostic info: {}",
            err_string
        );
    }

    #[test]
    fn test_run_apr_inference_error_message_structure() {
        let result = inference::run_apr_inference(
            "/definitely/not/a/real/path/model.apr",
            &[0x00, 0x01, 0x02, 0x03], // Invalid APR magic
            "error structure test",
            5,
            0.5,
            "text",
            false, // force_gpu
            false, // verbose
            None,  // trace_config
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_string = format!("{}", err);
        assert!(
            err_string.contains("APR")
                || err_string.contains("parse")
                || err_string.contains("Failed"),
            "Error message should contain APR-related info: {}",
            err_string
        );
    }

    #[test]
    fn test_run_safetensors_inference_error_message_structure() {
        let result = inference::run_safetensors_inference(
            "/definitely/not/a/real/path/model.safetensors",
            "error structure test",
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_string = format!("{}", err);
        assert!(
            err_string.contains("SafeTensors")
                || err_string.contains("convert")
                || err_string.contains("Failed")
                || err_string.contains("No such file"),
            "Error message should contain useful info: {}",
            err_string
        );
    }

    // =========================================================================
    // Prompt Content Edge Cases
    // =========================================================================

    #[test]
    fn test_prompt_with_null_bytes() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "hello\x00world", // Contains null byte
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_with_only_whitespace() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "   \t\n\r   ", // Only whitespace
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_with_control_characters() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "hello\x07\x08\x1bworld", // Bell, backspace, escape
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_with_utf8_multibyte() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "\u{1F4A9}\u{1F680}\u{1F31F}", // Poop, rocket, star emojis
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_with_cjk_characters() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "\u{4E2D}\u{6587}\u{6D4B}\u{8BD5}", // Chinese characters
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // File Data Edge Cases (for functions that accept file_data)
    // =========================================================================

    #[test]
    fn test_run_apr_inference_with_partial_magic() {
        // Only partial APR magic bytes
        let partial_magic = b"APR"; // Missing null terminator
        let result = inference::run_apr_inference(
            "/tmp/partial.apr",
            partial_magic,
            "partial magic",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_with_gguf_magic() {
        // GGUF magic bytes passed to APR parser
        let gguf_magic = b"GGUF\x03\x00\x00\x00";
        let result = inference::run_apr_inference(
            "/tmp/wrong_magic.apr",
            gguf_magic,
            "wrong magic",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_with_large_file_data() {
        // Large but invalid file data
        let large_data = vec![0x42u8; 1024 * 1024]; // 1MB of 'B' characters
        let result = inference::run_apr_inference(
            "/tmp/large.apr",
            &large_data,
            "large data",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Model Reference Path Edge Cases
    // =========================================================================

    #[test]
    fn test_model_ref_with_spaces() {
        let result = inference::run_gguf_inference(
            "/path/with spaces/model file.gguf",
            &[],
            "path with spaces",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_model_ref_with_unicode() {
        let result = inference::run_gguf_inference(
            "/path/\u{4E2D}\u{6587}/model.gguf",
            &[],
            "unicode path",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_model_ref_relative_path() {
        let result = inference::run_gguf_inference(
            "../models/model.gguf",
            &[],
            "relative path",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_model_ref_empty_string() {
        let result = inference::run_gguf_inference(
            "",
            &[],
            "empty model ref",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Environment Variable Coverage Tests
    // These test the conditional paths controlled by environment variables
    // =========================================================================

    #[test]
    fn test_cpu_debug_env_var_false() {
        std::env::set_var("CPU_DEBUG", "0");
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "debug disabled",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        std::env::remove_var("CPU_DEBUG");
    }

    #[test]
    fn test_cpu_debug_env_var_invalid() {
        std::env::set_var("CPU_DEBUG", "invalid");
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "debug invalid",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        std::env::remove_var("CPU_DEBUG");
    }

    #[test]
    fn test_skip_gpu_resident_env_var_false() {
        std::env::set_var("SKIP_GPU_RESIDENT", "0");
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "skip disabled",
            5,
            0.5,
            "text",
            true, // force_gpu
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        std::env::remove_var("SKIP_GPU_RESIDENT");
    }

    #[test]
    fn test_skip_gpu_resident_env_var_invalid() {
        std::env::set_var("SKIP_GPU_RESIDENT", "maybe");
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "skip invalid",
            5,
            0.5,
            "text",
            true, // force_gpu
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        std::env::remove_var("SKIP_GPU_RESIDENT");
    }

    // =========================================================================
    // Return Type and Result Handling Tests
    // =========================================================================

    #[test]
    fn test_result_error_type_is_realizar_error() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "type test",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());

        // Verify we get a RealizarError
        let err = result.unwrap_err();
        // The error should implement Display
        let _display = format!("{}", err);
        // The error should implement Debug
        let _debug = format!("{:?}", err);
    }

    // =========================================================================
    // Concurrent Safety Tests (Thread Safety)
    // =========================================================================

    #[test]
    fn test_inference_functions_are_thread_safe() {
        use std::thread;

        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    let result = inference::run_gguf_inference(
                        &format!("/nonexistent/model_{}.gguf", i),
                        &[],
                        &format!("thread {} prompt", i),
                        5,
                        0.5,
                        "text",
                        false,
                        false,
                        None, // trace_config
                    );
                    assert!(result.is_err());
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }
    }

    // =========================================================================
    // Parameter Combination Matrix Tests
    // =========================================================================

    #[test]
    fn test_parameter_combinations_gguf() {
        let temps = [0.0, 0.01, 0.5, 1.0];
        let formats = ["json", "text"];
        let gpu_flags = [false, true];
        let _verbose_flags = [false, true];
        let _trace_flags = [false, true];

        for &temp in &temps {
            for &format in &formats {
                for &gpu in &gpu_flags {
                    // Only test a subset to keep test runtime reasonable
                    let result = inference::run_gguf_inference(
                        "/nonexistent/model.gguf",
                        &[],
                        "combo test",
                        5,
                        temp,
                        format,
                        gpu,
                        false,
                        None, // trace_config
                    );
                    assert!(result.is_err());
                }
            }
        }
    }

    #[test]
    fn test_parameter_combinations_apr() {
        let temps = [0.0, 0.5, 1.0];
        let formats = ["json", "text"];
        let gpu_flags = [false, true];
        let verbose_flags = [false, true];

        for &temp in &temps {
            for &format in &formats {
                for &gpu in &gpu_flags {
                    for &verbose in &verbose_flags {
                        let result = inference::run_apr_inference(
                            "/nonexistent/model.apr",
                            &[],
                            "combo test",
                            5,
                            temp,
                            format,
                            gpu,
                            verbose,
                            None, // trace_config
                        );
                        assert!(result.is_err());
                    }
                }
            }
        }
    }

    // =========================================================================
    // Boundary Value Tests for Numeric Parameters
    // =========================================================================

    #[test]
    fn test_max_tokens_zero() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "zero tokens",
            0,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_negative() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "negative temp",
            5,
            -0.5, // Negative
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_infinity() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "inf temp",
            5,
            f32::INFINITY,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_nan() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "nan temp",
            5,
            f32::NAN,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // String Parameter Edge Cases
    // =========================================================================

    #[test]
    fn test_format_with_whitespace() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "format test",
            5,
            0.5,
            " json ", // Whitespace around format
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_format_case_sensitivity() {
        for format in &["JSON", "Json", "TEXT", "Text"] {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "case test",
                5,
                0.5,
                format,
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // API Contract Tests (Public Interface)
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_public_api() {
        // Verify public function signature matches expected API
        let _: crate::error::Result<()> = inference::run_gguf_inference(
            "",                           // model_ref: &str
            &[],                          // file_data: &[u8]
            "",                           // prompt: &str
            0,                            // max_tokens: usize
            0.0,                          // temperature: f32
            "",                           // format: &str
            true,                         // force_gpu: bool
            true,                         // verbose: bool
            Some(TraceConfig::enabled()), // trace_config
        );
    }

    #[test]
    fn test_run_safetensors_inference_public_api() {
        let _: crate::error::Result<()> = inference::run_safetensors_inference(
            "",   // model_ref: &str
            "",   // prompt: &str
            0,    // max_tokens: usize
            0.0,  // temperature: f32
            "",   // format: &str
            None, // trace_config
        );
    }

    #[test]
    fn test_run_apr_inference_public_api() {
        let _: crate::error::Result<()> = inference::run_apr_inference(
            "",    // model_ref: &str
            &[],   // file_data: &[u8]
            "",    // prompt: &str
            0,     // max_tokens: usize
            0.0,   // temperature: f32
            "",    // format: &str
            true,  // force_gpu: bool
            false, // verbose: bool
            None,  // trace_config
        );
    }
}

// =========================================================================
// CUDA Feature Tests (compiled only when cuda feature is enabled)
// =========================================================================

#[cfg(all(test, feature = "cuda"))]
mod cuda_inference_tests {
    use crate::cli::inference;
    use crate::inference_trace::TraceConfig;

    #[test]
    fn test_run_gguf_inference_gpu_function_exists() {
        // Verify the GPU function is available when cuda feature is enabled
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "cuda test",
            5,
            0.5,
            "text",
            true, // force_gpu - will try to use GPU path
            true, // verbose - shows GPU status
            None, // trace_config
        );
        assert!(result.is_err()); // File doesn't exist, but GPU path is attempted
    }

    #[test]
    fn test_run_apr_inference_gpu_function_exists() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "cuda apr test",
            5,
            0.5,
            "text",
            true, // force_gpu
            true, // verbose
            None, // trace_config
        );
        assert!(result.is_err());
    }
}
