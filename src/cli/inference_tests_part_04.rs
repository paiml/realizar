//! T-COV-95 Artifact Falsification: CLI Inference via Temp Files
//!
//! Dr. Popper's directive: "Do not mock the file; *become* the file."
//!
//! These tests write 1KB Active Pygmies to disk as actual .gguf and .apr files,
//! then invoke the real CLI inference functions to exercise:
//! - File discovery and mmap loading
//! - Argument parsing and validation
//! - Error handling for filesystem operations
//! - Full inference pipeline from CLI entry point

#[cfg(test)]
mod artifact_falsification {
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    use crate::apr::test_factory::build_executable_pygmy_apr;
    use crate::cli::inference;
    use crate::gguf::test_factory::build_executable_pygmy_gguf;

    // =========================================================================
    // GGUF Artifact Falsification - Real File, Real CLI
    // =========================================================================

    fn create_gguf_artifact() -> NamedTempFile {
        let data = build_executable_pygmy_gguf();
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(&data).expect("write gguf data");
        file.flush().expect("flush");
        file
    }

    #[test]
    fn test_gguf_artifact_cli_inference_text_format() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[],  // Empty - forces mmap path
            "Hello",
            5,    // max_tokens
            0.0,  // temperature (greedy)
            "text",
            false, // verbose
            false, // show_probs
            false, // debug
        );

        assert!(result.is_ok(), "GGUF CLI inference failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_artifact_cli_inference_json_format() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[],
            "Test prompt",
            3,
            0.0,
            "json",
            false,
            false,
            false,
        );

        assert!(result.is_ok(), "GGUF JSON format failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_artifact_cli_inference_verbose() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[],
            "Verbose test",
            2,
            0.0,
            "text",
            true,  // verbose
            false,
            false,
        );

        assert!(result.is_ok(), "GGUF verbose mode failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_artifact_cli_inference_with_temperature() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[],
            "Temperature test",
            3,
            0.7,  // non-zero temperature
            "text",
            false,
            false,
            false,
        );

        assert!(result.is_ok(), "GGUF with temperature failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_artifact_nonexistent_path_error() {
        let result = inference::run_gguf_inference(
            "/nonexistent/path/model.gguf",
            &[],
            "Test",
            5,
            0.0,
            "text",
            false,
            false,
            false,
        );

        assert!(result.is_err(), "Should fail for nonexistent file");
    }

    // =========================================================================
    // APR Artifact Falsification - Real File, Real CLI
    // =========================================================================

    fn create_apr_artifact() -> (NamedTempFile, Vec<u8>) {
        let data = build_executable_pygmy_apr();
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(&data).expect("write apr data");
        file.flush().expect("flush");
        (file, data)
    }

    #[test]
    fn test_apr_artifact_cli_inference_text_format() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path,
            &data,
            "Hello",
            5,
            0.0,
            "text",
            false,
            false,
        );

        assert!(result.is_ok(), "APR CLI inference failed: {:?}", result.err());
    }

    #[test]
    fn test_apr_artifact_cli_inference_json_format() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path,
            &data,
            "JSON test",
            3,
            0.0,
            "json",
            false,
            false,
        );

        assert!(result.is_ok(), "APR JSON format failed: {:?}", result.err());
    }

    #[test]
    fn test_apr_artifact_cli_inference_verbose() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path,
            &data,
            "Verbose",
            2,
            0.0,
            "text",
            true,  // verbose
            false,
        );

        assert!(result.is_ok(), "APR verbose mode failed: {:?}", result.err());
    }

    #[test]
    fn test_apr_artifact_cli_inference_with_temperature() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path,
            &data,
            "Temp test",
            3,
            0.8,
            "text",
            false,
            false,
        );

        assert!(result.is_ok(), "APR with temperature failed: {:?}", result.err());
    }

    #[test]
    fn test_apr_artifact_nonexistent_path_error() {
        // With empty data, should fail to parse
        let result = inference::run_apr_inference(
            "/nonexistent/path/model.apr",
            &[],
            "Test",
            5,
            0.0,
            "text",
            false,
            false,
        );

        assert!(result.is_err(), "Should fail for empty data");
    }

    // =========================================================================
    // CLI mod.rs Artifact Falsification - display_model_info, load_*_model
    // =========================================================================

    #[test]
    fn test_display_model_info_gguf_artifact() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();
        let data = std::fs::read(file.path()).unwrap();

        let result = crate::cli::display_model_info(path, &data);
        assert!(result.is_ok(), "display_model_info GGUF failed: {:?}", result.err());
    }

    #[test]
    fn test_display_model_info_apr_artifact() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = crate::cli::display_model_info(path, &data);
        assert!(result.is_ok(), "display_model_info APR failed: {:?}", result.err());
    }

    #[test]
    fn test_load_gguf_model_artifact() {
        let file = create_gguf_artifact();
        let data = std::fs::read(file.path()).unwrap();

        let result = crate::cli::load_gguf_model(&data);
        assert!(result.is_ok(), "load_gguf_model failed: {:?}", result.err());
    }

    #[test]
    fn test_load_apr_model_artifact() {
        let (_file, data) = create_apr_artifact();

        let result = crate::cli::load_apr_model(&data);
        assert!(result.is_ok(), "load_apr_model failed: {:?}", result.err());
    }

    // =========================================================================
    // Edge Cases and Error Paths
    // =========================================================================

    #[test]
    fn test_gguf_artifact_zero_max_tokens() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path, &[], "Test", 0, 0.0, "text", false, false, false,
        );
        // Should handle gracefully (either Ok with empty output or specific error)
        let _ = result; // Don't assert - just ensure no panic
    }

    #[test]
    fn test_apr_artifact_zero_max_tokens() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path, &data, "Test", 0, 0.0, "text", false, false,
        );
        let _ = result;
    }

    #[test]
    fn test_gguf_artifact_empty_prompt() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path, &[], "", 5, 0.0, "text", false, false, false,
        );
        // Empty prompt should be handled
        let _ = result;
    }

    #[test]
    fn test_apr_artifact_empty_prompt() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path, &data, "", 5, 0.0, "text", false, false,
        );
        let _ = result;
    }

    #[test]
    fn test_gguf_artifact_high_temperature() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path, &[], "High temp", 3, 2.0, "text", false, false, false,
        );
        assert!(result.is_ok(), "High temperature should work");
    }

    #[test]
    fn test_apr_artifact_debug_mode() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path, &data, "Debug", 2, 0.0, "text", false, true, // verbose (debug)
        );
        assert!(result.is_ok(), "Debug mode failed: {:?}", result.err());
    }

    // =========================================================================
    // Directory and Path Edge Cases
    // =========================================================================

    #[test]
    fn test_gguf_artifact_in_subdirectory() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("models");
        std::fs::create_dir(&subdir).unwrap();

        let file_path = subdir.join("pygmy.gguf");
        let data = build_executable_pygmy_gguf();
        std::fs::write(&file_path, &data).unwrap();

        let result = inference::run_gguf_inference(
            file_path.to_str().unwrap(),
            &[],
            "Subdir test",
            3,
            0.0,
            "text",
            false,
            false,
            false,
        );

        assert!(result.is_ok(), "Subdirectory inference failed: {:?}", result.err());
    }

    #[test]
    fn test_apr_artifact_in_subdirectory() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("models");
        std::fs::create_dir(&subdir).unwrap();

        let file_path = subdir.join("pygmy.apr");
        let data = build_executable_pygmy_apr();
        std::fs::write(&file_path, &data).unwrap();

        let result = inference::run_apr_inference(
            file_path.to_str().unwrap(),
            &data,
            "Subdir test",
            3,
            0.0,
            "text",
            false,
            false,
        );

        assert!(result.is_ok(), "APR subdirectory inference failed: {:?}", result.err());
    }

    // =========================================================================
    // Additional CLI mod.rs coverage - format detection and loading
    // =========================================================================

    #[test]
    fn test_display_model_info_unknown_format() {
        // Unknown format prints "Unknown" but succeeds
        let unknown_data = b"UNKNOWN_FORMAT_DATA_HERE";
        let result = crate::cli::display_model_info("/tmp/unknown.bin", unknown_data);
        // display_model_info handles unknown formats gracefully (prints format info)
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_gguf_model_invalid_data() {
        // Invalid GGUF data should fail
        let invalid = b"NOT_GGUF_DATA";
        let result = crate::cli::load_gguf_model(invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_invalid_data() {
        // Invalid APR data should fail
        let invalid = b"NOT_APR_DATA";
        let result = crate::cli::load_apr_model(invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_gguf_model_truncated() {
        // Truncated GGUF (just magic)
        let truncated = b"GGUF\x03\x00\x00\x00";
        let result = crate::cli::load_gguf_model(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_truncated() {
        // APR magic but truncated
        let truncated = b"APR\x00\x02\x00";
        let result = crate::cli::load_apr_model(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_local_file_path_absolute() {
        assert!(crate::cli::is_local_file_path("/home/user/model.gguf"));
        assert!(crate::cli::is_local_file_path("/tmp/model.apr"));
    }

    #[test]
    fn test_is_local_file_path_relative() {
        assert!(crate::cli::is_local_file_path("./model.gguf"));
        assert!(crate::cli::is_local_file_path("../models/model.apr"));
    }

    #[test]
    fn test_is_local_file_path_registry_uri() {
        assert!(!crate::cli::is_local_file_path("hf://model/name"));
        assert!(!crate::cli::is_local_file_path("ollama://llama2"));
    }

    #[test]
    fn test_validate_suite_name_valid() {
        // These are the actual benchmark suite names
        assert!(crate::cli::validate_suite_name("tensor_ops"));
        assert!(crate::cli::validate_suite_name("inference"));
        assert!(crate::cli::validate_suite_name("cache"));
        assert!(crate::cli::validate_suite_name("tokenizer"));
        assert!(crate::cli::validate_suite_name("quantize"));
    }

    #[test]
    fn test_validate_suite_name_invalid() {
        assert!(!crate::cli::validate_suite_name(""));
        assert!(!crate::cli::validate_suite_name("nonexistent_suite"));
        assert!(!crate::cli::validate_suite_name("random"));
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(crate::cli::format_size(0), "0 B");
        assert_eq!(crate::cli::format_size(512), "512 B");
        assert_eq!(crate::cli::format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(crate::cli::format_size(1024), "1.0 KB");
        assert_eq!(crate::cli::format_size(2048), "2.0 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(crate::cli::format_size(1024 * 1024), "1.0 MB");
        assert_eq!(crate::cli::format_size(10 * 1024 * 1024), "10.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(crate::cli::format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    // =========================================================================
    // GGUF inference edge cases - more format/option combinations
    // =========================================================================

    #[test]
    fn test_gguf_artifact_show_probs() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path, &[], "Test", 3, 0.0, "text",
            false, true, false, // show_probs=true
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_gguf_artifact_debug_mode() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path, &[], "Debug test", 2, 0.0, "text",
            false, false, true, // debug=true
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_gguf_artifact_all_options() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path, &[], "All options", 3, 0.5, "json",
            true, true, true, // verbose, show_probs, debug all true
        );
        assert!(result.is_ok());
    }
}
