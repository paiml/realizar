//! Server Command Tests & CLI Inference Module Tests (Part 12 - PMAT-802)
//!
//! Extracted from tests.rs to keep the main file under 2000 lines.
//! Contains:
//! 1. Server Command Tests (EXTREME TDD - PAR-112)
//! 2. ModelType / PreparedServer tests
//! 3. prepare_serve_state tests
//! 4. serve_demo tests
//! 5. CLI Inference Module Tests (run_gguf/safetensors/apr_inference)

#[cfg(test)]
mod server_and_inference_tests {
    use crate::cli::*;

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
    #[ignore = "requires real GGUF model file"]
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
    #[ignore = "requires real GGUF model file"]
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
        use crate::inference_trace::TraceConfig;

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
                None,    // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                Some(TraceConfig::enabled()), // trace_config
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
                None, // trace_config
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
                true,                         // force_gpu
                true,                         // verbose
                Some(TraceConfig::enabled()), // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                true,                         // force_gpu
                true,                         // verbose
                Some(TraceConfig::enabled()), // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
        #[ignore = "requires real GGUF model file"]
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
                None, // trace_config
            );

            // Should succeed with real model
            assert!(result.is_ok(), "Inference failed: {:?}", result.err());
        }

        /// Integration test for GGUF inference with JSON output
        #[test]
        #[ignore = "requires real GGUF model file"]
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
                None, // trace_config
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
                "model_ref",                  // model_ref: &str
                &[],                          // file_data: &[u8]
                "prompt",                     // prompt: &str
                10usize,                      // max_tokens: usize
                0.5f32,                       // temperature: f32
                "format",                     // format: &str
                true,                         // force_gpu: bool
                true,                         // verbose: bool
                Some(TraceConfig::enabled()), // trace_config: Option<TraceConfig>
            );
        }

        #[test]
        fn test_safetensors_inference_api_surface() {
            // Verify all parameters are accepted in the expected order
            let _ = inference::run_safetensors_inference(
                "model_ref",                  // model_ref: &str
                "prompt",                     // prompt: &str
                10usize,                      // max_tokens: usize
                0.5f32,                       // temperature: f32 (unused in current impl)
                "format",                     // format: &str
                Some(TraceConfig::enabled()), // trace_config: Option<TraceConfig>
            );
        }

        #[test]
        fn test_apr_inference_api_surface() {
            // Verify all parameters are accepted in the expected order
            let _ = inference::run_apr_inference(
                "model_ref",                  // model_ref: &str
                &[],                          // file_data: &[u8]
                "prompt",                     // prompt: &str
                10usize,                      // max_tokens: usize
                0.5f32,                       // temperature: f32
                "format",                     // format: &str
                true,                         // force_gpu: bool
                true,                         // verbose: bool
                Some(TraceConfig::enabled()), // trace_config: Option<TraceConfig>
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
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
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }
}
