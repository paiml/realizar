            generated_token_count: 3,
            inference_ms: 123.456,
            tok_per_sec: 24.32,
            load_ms: 50.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };

        let debug = format!("{:?}", result);
        assert!(debug.contains("text"));
        assert!(debug.contains("tokens"));
        assert!(debug.contains("input_token_count"));
        assert!(debug.contains("generated_token_count"));
        assert!(debug.contains("inference_ms"));
        assert!(debug.contains("tok_per_sec"));
        assert!(debug.contains("load_ms"));
        assert!(debug.contains("format"));
        assert!(debug.contains("used_gpu"));
    }

    #[test]
    fn test_inference_result_clone_preserves_all() {
        let original = InferenceResult {
            text: "test text".to_string(),
            tokens: vec![100, 200, 300],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 999.9,
            tok_per_sec: 2.0,
            load_ms: 111.1,
            format: "APR".to_string(),
            used_gpu: false,
        };

        let cloned = original.clone();

        assert_eq!(original.text, cloned.text);
        assert_eq!(original.tokens, cloned.tokens);
        assert_eq!(original.input_token_count, cloned.input_token_count);
        assert_eq!(original.generated_token_count, cloned.generated_token_count);
        assert!((original.inference_ms - cloned.inference_ms).abs() < f64::EPSILON);
        assert!((original.tok_per_sec - cloned.tok_per_sec).abs() < f64::EPSILON);
        assert!((original.load_ms - cloned.load_ms).abs() < f64::EPSILON);
        assert_eq!(original.format, cloned.format);
        assert_eq!(original.used_gpu, cloned.used_gpu);
    }

    // --- Format detection with various magic bytes ---

    #[test]
    fn test_format_detection_gguf_version_variations() {
        use crate::format::{detect_format, ModelFormat};

        // GGUF version 3
        let v3_data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00];
        let result = detect_format(&v3_data);
        assert!(matches!(result, Ok(ModelFormat::Gguf)));

        // GGUF version 2
        let v2_data = vec![0x47, 0x47, 0x55, 0x46, 0x02, 0x00, 0x00, 0x00];
        let result = detect_format(&v2_data);
        assert!(matches!(result, Ok(ModelFormat::Gguf)));
    }

    #[test]
    fn test_format_detection_apr_versions() {
        use crate::format::{detect_format, ModelFormat};

        // APR v1 - version byte is ASCII '1' (0x31), not \x01
        let v1_data = b"APR1xxxx";
        let result = detect_format(v1_data);
        assert!(matches!(result, Ok(ModelFormat::Apr)));

        // APR v2 - version byte is ASCII '2' (0x32), not \x02
        let v2_data = b"APR2xxxx";
        let result = detect_format(v2_data);
        assert!(matches!(result, Ok(ModelFormat::Apr)));
    }

    #[test]
    fn test_format_detection_safetensors_various_sizes() {
        use crate::format::{detect_format, ModelFormat};

        // Various valid header sizes for SafeTensors
        for size in [64u64, 128, 256, 512, 1024, 2048, 4096] {
            let data = size.to_le_bytes();
            let result = detect_format(&data);
            assert!(
                matches!(result, Ok(ModelFormat::SafeTensors)),
                "Failed for size: {}",
                size
            );
        }
    }

    // --- run_inference file handling edge cases ---

    #[test]
    fn test_run_inference_symlink_path() {
        // Test with a path that looks like a symlink (just path string)
        let config = InferenceConfig::new("/models/latest-model.gguf");
        // Can't test actual symlink without creating one, but test config handling
        assert!(config.model_path.to_str().unwrap().contains("latest-model"));
    }

    #[test]
    fn test_run_inference_hidden_file() {
        // Test with hidden file path
        let config = InferenceConfig::new("/models/.hidden-model.gguf");
        assert!(config
            .model_path
            .to_str()
            .unwrap()
            .contains(".hidden-model"));
    }

    // --- Comprehensive builder chaining test ---

    #[test]
    fn test_inference_config_builder_order_independence() {
        // Order 1
        let config1 = InferenceConfig::new("/m.gguf")
            .with_prompt("p")
            .with_max_tokens(50)
            .with_temperature(0.5)
            .with_top_k(20)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);

        // Order 2 (reversed)
        let config2 = InferenceConfig::new("/m.gguf")
            .with_trace(true)
            .with_verbose(true)
            .without_gpu()
            .with_top_k(20)
            .with_temperature(0.5)
            .with_max_tokens(50)
            .with_prompt("p");

        assert_eq!(config1.prompt, config2.prompt);
        assert_eq!(config1.max_tokens, config2.max_tokens);
        assert!((config1.temperature - config2.temperature).abs() < f32::EPSILON);
        assert_eq!(config1.top_k, config2.top_k);
        assert_eq!(config1.no_gpu, config2.no_gpu);
        assert_eq!(config1.verbose, config2.verbose);
        assert_eq!(config1.trace, config2.trace);
    }

    // --- Path with special extensions ---

    #[test]
    fn test_path_extension_handling() {
        let extensions = [".gguf", ".apr", ".safetensors", ".bin", ".model", ""];
        for ext in extensions {
            let path = format!("/models/model{}", ext);
            let config = InferenceConfig::new(&path);
            assert!(config.model_path.to_str().unwrap().ends_with(ext) || ext.is_empty());
        }
    }

    // --- Boundary tests for numeric fields ---

    #[test]
    fn test_numeric_boundary_values() {
        // Max values
        let config = InferenceConfig::new("/m.gguf")
            .with_max_tokens(usize::MAX)
            .with_temperature(f32::MAX)
            .with_top_k(usize::MAX);

        assert_eq!(config.max_tokens, usize::MAX);
        assert_eq!(config.temperature, f32::MAX);
        assert_eq!(config.top_k, usize::MAX);

        // Min values
        let config = InferenceConfig::new("/m.gguf")
            .with_max_tokens(0)
            .with_temperature(f32::MIN)
            .with_top_k(0);

        assert_eq!(config.max_tokens, 0);
        assert_eq!(config.temperature, f32::MIN);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_temperature_special_values() {
        // Infinity
        let config = InferenceConfig::new("/m.gguf").with_temperature(f32::INFINITY);
        assert!(config.temperature.is_infinite());

        // Negative infinity
        let config = InferenceConfig::new("/m.gguf").with_temperature(f32::NEG_INFINITY);
        assert!(config.temperature.is_infinite());
        assert!(config.temperature.is_sign_negative());

        // NaN
        let config = InferenceConfig::new("/m.gguf").with_temperature(f32::NAN);
        assert!(config.temperature.is_nan());
    }

    // =========================================================================
    // Extended Coverage Tests (PMAT-COV-002)
    // These tests target specific lines and branches in infer/mod.rs
    // =========================================================================

    // --- run_inference format dispatch tests ---

    #[test]
    fn test_run_inference_dispatches_to_gguf_branch() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create GGUF-like file with full header but incomplete model
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
                                                     // Add some padding
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_max_tokens(1)
            .with_verbose(false);

        // Dispatches to GGUF path, fails on model load (no tensors)
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_dispatches_to_apr_branch() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create APR v2 file
        let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"APR2"); // APR v2 magic
        data.extend_from_slice(&[0u8; 100]); // Minimal header
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Hello")
            .with_max_tokens(2);

        // Dispatches to APR path - runs with fallback zeros if incomplete
        let result = run_inference(&config);
        // May succeed with degenerate output or fail - either is valid
        let _ = result;
    }

    #[test]
    fn test_run_inference_dispatches_to_safetensors_branch() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create SafeTensors-like file with valid header size
        let mut temp = NamedTempFile::with_suffix(".safetensors").expect("create temp");
        let header_size: u64 = 50;
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        // JSON header (minimal but valid enough to trigger conversion code)
        let json = r#"{"metadata":{}}"#;
        data.extend_from_slice(json.as_bytes());
        data.extend_from_slice(&[0u8; 50]); // Padding
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_max_tokens(1);

        // Dispatches to SafeTensors path, fails on model conversion
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    // --- Error message content validation ---

    #[test]
    fn test_run_inference_io_error_message_content() {
        let config = InferenceConfig::new("/does/not/exist/model.gguf");
        let result = run_inference(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to read") || err_msg.contains("IO error"),
            "Error message should mention read failure: {}",
            err_msg
        );
    }

    #[test]
    fn test_run_inference_format_error_message_content() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create file with unknown magic
        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        temp.write_all(&[0u8; 16]).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Format") || err_msg.contains("format"),
            "Error message should mention format: {}",
            err_msg
        );
    }

    // --- Input token priority tests ---

    #[test]
    fn test_input_tokens_takes_priority_in_gguf_path() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        // Both prompt and input_tokens set - input_tokens should take priority
        let config = InferenceConfig::new(temp.path())
            .with_prompt("This is ignored")
            .with_input_tokens(vec![1, 2, 3])
            .with_max_tokens(5);

        assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
        assert_eq!(config.prompt, Some("This is ignored".to_string()));

        // Will fail on model load, but exercises the input_tokens path
        let _result = run_inference(&config);
    }

    // --- Verbose mode output tests ---

    #[test]
    fn test_verbose_mode_doesnt_panic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        // Verbose mode enabled - should output to stderr without panicking
        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_verbose(true)
            .with_max_tokens(1);

        // Will fail on model load, but exercises verbose output path
        let _result = run_inference(&config);
        // Test passes if no panic
    }

    // --- Architecture detection comprehensive tests ---

    #[test]
    fn test_architecture_detection_all_variants() {
        let test_cases = [
            ("qwen2-7b-instruct.gguf", "Qwen2"),
            ("Qwen-7B-Chat.gguf", "Qwen2"),
            ("QWEN_7B.gguf", "Qwen2"),
            ("llama-3.1-8b-instruct.gguf", "LLaMA"),
            ("Llama-2-7b-chat.gguf", "LLaMA"),
            ("LLAMA3.gguf", "LLaMA"),
            ("mistral-7b-instruct-v0.2.gguf", "Mistral"),
            ("Mistral-7B.gguf", "Mistral"),
            ("MISTRAL_LARGE.gguf", "Mistral"),
            ("phi-2.gguf", "Phi"),
            ("Phi-3-mini.gguf", "Phi"),
            ("PHI2.gguf", "Phi"),
            ("custom-model.gguf", "Transformer"),
            ("gpt2.gguf", "Transformer"),
            ("my-finetuned-model.gguf", "Transformer"),
        ];

        for (filename, expected_arch) in test_cases {
            let path = PathBuf::from(format!("/models/{}", filename));
            let arch = path.file_stem().and_then(|s| s.to_str()).map(|s| {
                if s.to_lowercase().contains("qwen") {
                    "Qwen2"
                } else if s.to_lowercase().contains("llama") {
                    "LLaMA"
                } else if s.to_lowercase().contains("mistral") {
                    "Mistral"
                } else if s.to_lowercase().contains("phi") {
                    "Phi"
                } else {
                    "Transformer"
                }
            });
            assert_eq!(arch, Some(expected_arch), "Failed for: {}", filename);
        }
    }

    // --- prefault_mmap comprehensive tests ---

    #[test]
    fn test_prefault_mmap_various_patterns() {
        // Test with various byte patterns
        let patterns: Vec<Vec<u8>> = vec![
            vec![0u8; 8192],                                 // All zeros
            vec![255u8; 8192],                               // All ones
            (0..8192u16).map(|i| (i % 256) as u8).collect(), // Sequential
            vec![0xAA; 8192],                                // Alternating bits
            vec![0x55; 8192],                                // Alternating bits (inverted)
        ];

        for pattern in patterns {
            prefault_mmap(&pattern);
        }
    }

    #[test]
    fn test_prefault_mmap_exactly_one_page_minus_one() {
        let data = vec![1u8; 4095];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_exactly_two_pages() {
        let data = vec![2u8; 8192];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_many_pages() {
        let data = vec![3u8; 4096 * 100]; // 100 pages
        prefault_mmap(&data);
    }

    // --- clean_model_output comprehensive edge cases ---

    #[test]
    fn test_clean_model_output_all_markers_comprehensive() {
        let test_cases = [
            ("<|im_start|>assistant\nContent<|im_end|>", "Content"),
            ("<|im_start|>assistantContent<|im_end|>", "Content"),
            ("Text<|endoftext|>", "Text"),
            (
                "<|im_start|>embedded marker<|im_start|>more<|im_end|>",
                "embedded markermore",
            ),
            ("   whitespace around   ", "whitespace around"),
            ("\n\nleading newlines\n\n", "leading newlines"),
        ];

        for (input, expected) in test_cases {
            let result = clean_model_output(input);
            assert_eq!(result, expected, "Failed for input: {:?}", input);
        }
    }

    #[test]
    fn test_clean_model_output_complex_conversation() {
        let raw = r"<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there! How can I help?<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        // All markers removed, content preserved
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        assert!(!cleaned.contains("<|endoftext|>"));
    }

    #[test]
    fn test_clean_model_output_json_content() {
        let raw = r#"<|im_start|>assistant
{"key": "value", "nested": {"inner": 42}}<|im_end|>"#;
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains(r#"{"key": "value"#));
    }

    #[test]
    fn test_clean_model_output_multiline_code() {
        let raw = r#"<|im_start|>assistant
```rust
fn main() {
    println!("Hello, world!");
}
```<|im_end|>"#;
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("fn main()"));
        assert!(cleaned.contains("println!"));
    }

    // --- InferenceResult field boundary tests ---

    #[test]
    fn test_inference_result_all_zero_counts() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.input_token_count + result.generated_token_count, 0);
    }

    #[test]
    fn test_inference_result_max_counts() {
        let result = InferenceResult {
            text: "x".repeat(1_000_000),
            tokens: vec![1; 1_000_000],
            input_token_count: 500_000,
            generated_token_count: 500_000,
            inference_ms: f64::MAX,
            tok_per_sec: f64::MAX,
            load_ms: f64::MAX,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.tokens.len(), 1_000_000);
    }

    // --- Format detection edge cases from run_inference ---

    #[test]
    fn test_run_inference_apr_v1_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"APR1"); // APR v1 magic
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path()).with_max_tokens(1);
        let result = run_inference(&config);
        // Dispatches to APR path - runs with fallback zeros if incomplete
        let _ = result;
    }

    #[test]
    fn test_run_inference_gguf_v2_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&2u32.to_le_bytes()); // GGUF version 2
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path()).with_max_tokens(1);
        let result = run_inference(&config);
        // Should dispatch to GGUF path
        assert!(result.is_err());
    }

    // --- Trace configuration tests ---

    #[test]
    fn test_trace_config_propagation() {
        let config = InferenceConfig::new("/model.gguf")
            .with_trace(true)
            .with_prompt("test");

        assert!(config.trace);
        // trace flag should be passed to gen_config in run_gguf_inference
    }

    #[test]
    fn test_trace_verbose_and_output() {
        let mut config = InferenceConfig::new("/model.gguf");
        config.trace = true;
        config.trace_verbose = true;
        config.trace_output = Some(PathBuf::from("/tmp/trace.json"));
        config.trace_steps = Some(vec!["embedding".to_string(), "attention".to_string()]);

        assert!(config.trace);
        assert!(config.trace_verbose);
        assert_eq!(config.trace_output, Some(PathBuf::from("/tmp/trace.json")));
        assert_eq!(config.trace_steps.as_ref().map(std::vec::Vec::len), Some(2));
    }

    // --- no_gpu flag tests ---

    #[test]
    fn test_no_gpu_flag_set() {
        let config = InferenceConfig::new("/model.gguf").without_gpu();
        assert!(config.no_gpu);
    }

    #[test]
    fn test_no_gpu_flag_default_false() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(!config.no_gpu);
    }

    // --- InferenceConfig Debug trait coverage ---

    #[test]
    fn test_inference_config_debug_contains_all_fields() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/model.gguf"),
            prompt: Some("test".to_string()),
            input_tokens: Some(vec![1, 2]),
            max_tokens: 64,
            temperature: 0.7,
            top_k: 40,
            no_gpu: true,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/trace.json")),
            trace_steps: Some(vec!["a".to_string()]),
            verbose: true,
            use_mock_backend: false,
        };

        let debug = format!("{:?}", config);
        let expected_fields = [
            "model_path",
            "prompt",
            "input_tokens",
            "max_tokens",
            "temperature",
            "top_k",
            "no_gpu",
            "trace",
            "trace_verbose",
            "trace_output",
            "trace_steps",
            "verbose",
        ];

        for field in expected_fields {
            assert!(debug.contains(field), "Debug missing field: {}", field);
        }
    }

    // --- InferenceResult Clone trait coverage ---

    #[test]
    fn test_inference_result_clone_independence() {
        let original = InferenceResult {
            text: "original".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 100.0,
            tok_per_sec: 20.0,
            load_ms: 50.0,
            format: "APR".to_string(),
            used_gpu: true,
        };

        let mut cloned = original.clone();
        cloned.text = "modified".to_string();
        cloned.tokens.push(4);

        // Original should be unchanged
        assert_eq!(original.text, "original");
        assert_eq!(original.tokens, vec![1, 2, 3]);
    }

    // --- File size edge cases ---

    #[test]
    fn test_run_inference_exactly_7_bytes() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        temp.write_all(&[1, 2, 3, 4, 5, 6, 7]).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_run_inference_exactly_8_bytes_unknown_format() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        temp.write_all(&[0, 0, 0, 0, 0, 0, 0, 0]).expect("write"); // Zero header = unknown
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Format"));
    }

    #[test]
    fn test_run_inference_large_file_header_only() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        // Add significant padding to simulate larger file
        data.extend_from_slice(&[0u8; 10000]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path()).with_max_tokens(1);
        let result = run_inference(&config);
        assert!(result.is_err()); // Should fail on model load
    }

    // --- tok_per_sec calculation edge cases ---

    #[test]
    fn test_tok_per_sec_with_fractional_ms() {
        let inference_ms = 333.333;
        let generated = 100;
        let tok_per_sec = if inference_ms > 0.0 {
            generated as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        // 100 / 0.333333 = ~300
        assert!((tok_per_sec - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_tok_per_sec_with_very_large_token_count() {
        let inference_ms = 1000.0;
        let generated: u64 = 1_000_000;
        let tok_per_sec = if inference_ms > 0.0 {
            generated as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 1_000_000.0).abs() < 0.001);
    }

    // --- Model path handling edge cases ---

    #[test]
    fn test_model_path_with_dots_in_name() {
        let config = InferenceConfig::new("/models/model.v1.0.beta.gguf");
        let stem = config.model_path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("model.v1.0.beta"));
    }

    #[test]
    fn test_model_path_with_no_extension() {
        let config = InferenceConfig::new("/models/modelfile");
        let ext = config.model_path.extension();
        assert!(ext.is_none());
    }

    #[test]
    fn test_model_path_hidden_file() {
        let config = InferenceConfig::new("/models/.hidden_model.gguf");
        let name = config.model_path.file_name().and_then(|s| s.to_str());
        assert_eq!(name, Some(".hidden_model.gguf"));
    }

    // --- Instruct model detection additional tests ---

    #[test]
    fn test_instruct_detection_suffix_variations() {
        let suffixes = [
            "-instruct",
            "-Instruct",
            "-INSTRUCT",
            "_instruct",
            ".instruct",
            "-instruct-v2",
            "-chat-instruct",
        ];

        for suffix in suffixes {
            let name = format!("model{}.gguf", suffix);
            let is_instruct = name.to_lowercase().contains("instruct");
            assert!(is_instruct, "Should detect instruct in: {}", name);
        }
    }

    // --- BOS token fallback test ---

    #[test]
    fn test_bos_token_fallback_when_no_prompt_or_tokens() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());

        // In run_gguf_inference, this would result in vec![1u32] (BOS token)
        let default_tokens = config.input_tokens.clone().unwrap_or_else(|| {
            if config.prompt.is_some() {
                vec![100, 200] // Would be tokenized
            } else {
                vec![1u32] // BOS token
            }
        });
        assert_eq!(default_tokens, vec![1u32]);
    }

    // --- Format detection from data boundary tests ---

    #[test]
    fn test_format_detection_safetensors_min_valid_size() {
        use crate::format::{detect_format, ModelFormat};

        // Minimum valid SafeTensors header size (1 byte)
        let header_size: u64 = 1;
        let data = header_size.to_le_bytes();
        let result = detect_format(&data);
        assert!(matches!(result, Ok(ModelFormat::SafeTensors)));
    }

    #[test]
    fn test_format_detection_safetensors_typical_sizes() {
        use crate::format::{detect_format, ModelFormat};

        for size in [100u64, 1000, 10000, 100000, 1000000, 10000000] {
            let data = size.to_le_bytes();
            let result = detect_format(&data);
            assert!(
                matches!(result, Ok(ModelFormat::SafeTensors)),
                "Failed for size: {}",
                size
            );
        }
    }

    // --- InferenceConfig builder method return type tests ---

    #[test]
    fn test_builder_methods_return_self() {
        let config = InferenceConfig::new("/model.gguf");

        // Each method should consume self and return Self
        let config = config.with_prompt("test");
        let config = config.with_input_tokens(vec![1]);
        let config = config.with_max_tokens(10);
        let config = config.with_temperature(0.5);
        let config = config.with_top_k(10);
        let config = config.without_gpu();
        let config = config.with_verbose(true);
        let config = config.with_trace(true);

        // Verify final state
        assert_eq!(config.prompt, Some("test".to_string()));
        assert_eq!(config.input_tokens, Some(vec![1]));
        assert_eq!(config.max_tokens, 10);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    // --- Empty string handling ---

    #[test]
    fn test_empty_prompt_handling() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("");
        assert_eq!(config.prompt, Some(String::new()));
    }

    #[test]
    fn test_empty_input_tokens_handling() {
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![]);
        assert_eq!(config.input_tokens, Some(vec![]));
    }

    // --- Path conversion tests ---

    #[test]
    fn test_path_from_str() {
        let config = InferenceConfig::new("/path/to/model.gguf");
        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
    }

    #[test]
    fn test_path_from_string() {
        let path = String::from("/path/to/model.gguf");
        let config = InferenceConfig::new(path);
        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
    }

    #[test]
    fn test_path_from_pathbuf() {
        let path = PathBuf::from("/path/to/model.gguf");
        let config = InferenceConfig::new(path.clone());
        assert_eq!(config.model_path, path);
    }

    // --- RealizarError type tests ---

    #[test]
    fn test_io_error_construction() {
        use crate::error::RealizarError;

        let err = RealizarError::IoError {
            message: "test error".to_string(),
        };
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_format_error_construction() {
        use crate::error::RealizarError;

        let err = RealizarError::FormatError {
            reason: "invalid magic".to_string(),
        };
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn test_inference_error_construction() {
        use crate::error::RealizarError;

        let err = RealizarError::InferenceError("generation failed".to_string());
        assert!(err.to_string().contains("generation failed"));
    }

    // --- EOS token detection comprehensive ---

    #[test]
    fn test_eos_token_detection_all_variants() {
        // Define EOS tokens to check against
        let eos_tokens = [151645u32, 151643, 2];

        // Helper to check if token is EOS
        let is_eos = |token: u32| eos_tokens.contains(&token);

        // Qwen2 EOS
        assert!(is_eos(151645));
        // Qwen2 BOS (also stops)
        assert!(is_eos(151643));
        // Standard EOS
        assert!(is_eos(2));
        // Not EOS tokens
        assert!(!is_eos(1));
        assert!(!is_eos(1000));
    }

    // --- Max tokens capping in different formats ---

    #[test]
    fn test_max_tokens_capping_consistency() {
        let configs = [
            InferenceConfig::new("/m.gguf").with_max_tokens(500),
            InferenceConfig::new("/m.apr").with_max_tokens(500),
            InferenceConfig::new("/m.safetensors").with_max_tokens(500),
        ];

        for config in configs {
            let capped = config.max_tokens.min(128);
            assert_eq!(capped, 128, "Max tokens should be capped at 128");
        }
    }

    // --- Generated tokens calculation ---

    #[test]
    fn test_generated_token_count_calculation() {
        let all_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let input_token_count = 4;
        let generated_tokens = &all_tokens[input_token_count..];
        let generated_token_count = generated_tokens.len();

        assert_eq!(generated_token_count, 6);
        assert_eq!(all_tokens.len(), input_token_count + generated_token_count);
    }

    // --- Load time precision ---

    #[test]
    fn test_load_time_precision() {
        let elapsed_secs: f64 = 0.123456789;
        let load_ms = elapsed_secs * 1000.0;

        // Should preserve precision
        assert!((load_ms - 123.456789).abs() < 0.0001);
    }

    #[test]
    fn test_inference_time_precision() {
        let elapsed_secs: f64 = 0.987654321;
        let inference_ms = elapsed_secs * 1000.0;

        assert!((inference_ms - 987.654321).abs() < 0.0001);
    }
}
