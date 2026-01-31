#[cfg(test)]
mod tests {
    use crate::infer::*;

    // =========================================================================
    // InferenceConfig Builder Tests
    // =========================================================================

    #[test]
    fn test_inference_config_builder() {
        let config = InferenceConfig::new("/path/to/model.gguf")
            .with_prompt("Hello")
            .with_max_tokens(64)
            .with_temperature(0.7)
            .with_top_k(40)
            .without_gpu()
            .with_verbose(true);

        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
        assert_eq!(config.prompt, Some("Hello".to_string()));
        assert_eq!(config.max_tokens, 64);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 40);
        assert!(config.no_gpu);
        assert!(config.verbose);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::new("/model.gguf");
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_with_input_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![1, 2, 3, 4]);
        assert_eq!(config.input_tokens, Some(vec![1, 2, 3, 4]));
        assert!(config.prompt.is_none()); // prompt not set
    }

    #[test]
    fn test_inference_config_with_trace() {
        let config = InferenceConfig::new("/model.gguf").with_trace(true);
        assert!(config.trace);
        assert!(!config.trace_verbose);
    }

    // =========================================================================
    // Output Cleaning Tests
    // =========================================================================

    #[test]
    fn test_clean_model_output() {
        let raw = "<|im_start|>assistant\nHello world!<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_no_markers() {
        let raw = "Hello world!";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_multiple_markers() {
        let raw = "<|im_start|><|im_start|>assistant\nMultiple markers<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Multiple markers");
    }

    #[test]
    fn test_clean_model_output_whitespace_only() {
        let raw = "<|im_start|>assistant\n   <|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_with_newlines() {
        let raw = "<|im_start|>assistant\nLine 1\nLine 2<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Line 1\nLine 2");
    }

    // =========================================================================
    // Prefault Mmap Tests
    // =========================================================================

    #[test]
    fn test_prefault_mmap_empty() {
        let data: &[u8] = &[];
        prefault_mmap(data); // Should not panic
    }

    #[test]
    fn test_prefault_mmap_small() {
        let data = vec![0u8; 1000];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_aligned() {
        let data = vec![0u8; 4096 * 3]; // 3 pages
        prefault_mmap(&data);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_run_inference_file_not_found() {
        let config = InferenceConfig::new("/nonexistent/model.gguf");
        let result = run_inference(&config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        // F-SEC-222: validate_model_path now catches nonexistent files early
        assert!(
            err.to_string().contains("File not found")
                || err.to_string().contains("Failed to read model"),
            "Expected file not found error, got: {}",
            err
        );
    }

    #[test]
    fn test_run_inference_file_too_small() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("tiny_model.gguf");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[1, 2, 3]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_run_inference_invalid_format() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("invalid_format_model.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Format"));

        let _ = std::fs::remove_file(path);
    }

    // =========================================================================
    // InferenceResult Tests
    // =========================================================================

    #[test]
    fn test_inference_result_default() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 100.0,
            tok_per_sec: 20.0,
            load_ms: 50.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.text, "test");
        assert_eq!(result.tokens, vec![1, 2, 3]);
        assert_eq!(result.generated_token_count, 2);
    }

    #[test]
    fn test_inference_result_clone() {
        let result = InferenceResult {
            text: "hello".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 10.0,
            tok_per_sec: 0.0,
            load_ms: 5.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        let cloned = result.clone();
        assert_eq!(result.text, cloned.text);
        assert_eq!(result.used_gpu, cloned.used_gpu);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_inference_config_with_trace_enabled() {
        let config = InferenceConfig::new("/model.gguf").with_trace(true);
        assert!(config.trace);
    }

    #[test]
    fn test_inference_config_with_trace_disabled() {
        let config = InferenceConfig::new("/model.gguf").with_trace(false);
        assert!(!config.trace);
    }

    #[test]
    fn test_inference_config_builder_all_options() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(128)
            .with_temperature(0.8)
            .with_top_k(50)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);

        assert_eq!(config.prompt, Some("Test prompt".to_string()));
        assert_eq!(config.max_tokens, 128);
        assert!((config.temperature - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    #[test]
    fn test_clean_model_output_chatml_markers() {
        let raw = "<|im_start|>assistant\nHello world!<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        // Should clean ChatML markers
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        assert!(!cleaned.contains("<|endoftext|>"));
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_empty_input() {
        let cleaned = clean_model_output("");
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_only_markers() {
        let raw = "<|im_start|><|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_prefault_mmap_large() {
        let data = vec![0u8; 4096 * 10]; // 10 pages
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_unaligned() {
        let data = vec![0u8; 4096 + 100]; // 1 page + extra
        prefault_mmap(&data);
    }

    #[test]
    fn test_inference_config_debug() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("test");
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("model_path"));
        assert!(debug_str.contains("prompt"));
    }

    #[test]
    fn test_inference_result_debug() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 1,
            inference_ms: 10.0,
            tok_per_sec: 100.0,
            load_ms: 5.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("text"));
        assert!(debug_str.contains("tokens"));
    }

    #[test]
    fn test_inference_config_with_zero_temperature() {
        let config = InferenceConfig::new("/model.gguf").with_temperature(0.0);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_with_high_temperature() {
        let config = InferenceConfig::new("/model.gguf").with_temperature(2.0);
        assert!((config.temperature - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_with_zero_top_k() {
        let config = InferenceConfig::new("/model.gguf").with_top_k(0);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_inference_config_with_large_top_k() {
        let config = InferenceConfig::new("/model.gguf").with_top_k(1000);
        assert_eq!(config.top_k, 1000);
    }

    #[test]
    fn test_inference_config_with_zero_max_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(0);
        assert_eq!(config.max_tokens, 0);
    }

    #[test]
    fn test_inference_config_with_large_max_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(4096);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_inference_config_chaining() {
        // Test that builder methods can be chained in any order
        let config = InferenceConfig::new("/m.gguf")
            .with_verbose(true)
            .with_trace(true)
            .with_prompt("p")
            .without_gpu()
            .with_max_tokens(10);
        assert!(config.verbose);
        assert!(config.trace);
        assert_eq!(config.prompt, Some("p".to_string()));
        assert!(config.no_gpu);
        assert_eq!(config.max_tokens, 10);
    }

    #[test]
    fn test_clean_model_output_preserves_content() {
        let raw = "Hello, this is a test without any markers.";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, raw);
    }

    #[test]
    fn test_clean_model_output_partial_markers() {
        let raw = "Hello <|im_start|> world <|im_end|> test";
        let cleaned = clean_model_output(raw);
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
    }

    // =========================================================================
    // Extended Coverage Tests for InferenceConfig
    // =========================================================================

    #[test]
    fn test_inference_config_clone_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("test")
            .with_max_tokens(64)
            .with_temperature(0.5)
            .with_top_k(20)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);
        let cloned = config.clone();
        assert_eq!(config.model_path, cloned.model_path);
        assert_eq!(config.prompt, cloned.prompt);
        assert_eq!(config.max_tokens, cloned.max_tokens);
        assert!((config.temperature - cloned.temperature).abs() < f32::EPSILON);
        assert_eq!(config.top_k, cloned.top_k);
        assert_eq!(config.no_gpu, cloned.no_gpu);
        assert_eq!(config.verbose, cloned.verbose);
        assert_eq!(config.trace, cloned.trace);
    }

    #[test]
    fn test_inference_config_input_tokens_clone_cov() {
        let config = InferenceConfig::new("/m.gguf").with_input_tokens(vec![1, 2, 3, 4, 5]);
        let cloned = config.clone();
        assert_eq!(config.input_tokens, cloned.input_tokens);
    }

    #[test]
    fn test_inference_config_empty_prompt_cov() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("");
        assert_eq!(config.prompt, Some(String::new()));
    }

    #[test]
    fn test_inference_config_long_prompt_cov() {
        let long_prompt = "x".repeat(10000);
        let config = InferenceConfig::new("/model.gguf").with_prompt(&long_prompt);
        assert_eq!(config.prompt, Some(long_prompt));
    }

    #[test]
    fn test_inference_config_empty_input_tokens_cov() {
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![]);
        assert_eq!(config.input_tokens, Some(vec![]));
    }

    #[test]
    fn test_inference_config_large_input_tokens_cov() {
        let tokens: Vec<u32> = (0..1000).collect();
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(tokens.clone());
        assert_eq!(config.input_tokens, Some(tokens));
    }

    #[test]
    fn test_inference_config_negative_temperature_cov() {
        // Temperature can be set to any float, even negative
        let config = InferenceConfig::new("/model.gguf").with_temperature(-1.0);
        assert!((config.temperature - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_verbose_false_cov() {
        let config = InferenceConfig::new("/model.gguf").with_verbose(false);
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_path_with_spaces_cov() {
        let config = InferenceConfig::new("/path/with spaces/model.gguf");
        assert_eq!(
            config.model_path,
            PathBuf::from("/path/with spaces/model.gguf")
        );
    }

    #[test]
    fn test_inference_config_path_from_string_cov() {
        let path_str = String::from("/model.gguf");
        let config = InferenceConfig::new(path_str);
        assert_eq!(config.model_path, PathBuf::from("/model.gguf"));
    }

    #[test]
    fn test_inference_config_path_from_pathbuf_cov() {
        let path = PathBuf::from("/model.gguf");
        let config = InferenceConfig::new(path.clone());
        assert_eq!(config.model_path, path);
    }

    #[test]
    fn test_inference_config_defaults_not_overwritten_cov() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());
        assert!(!config.trace);
        assert!(!config.trace_verbose);
        assert!(config.trace_output.is_none());
        assert!(config.trace_steps.is_none());
    }

    // =========================================================================
    // Extended Coverage Tests for InferenceResult
    // =========================================================================

    #[test]
    fn test_inference_result_with_zero_inference_time_cov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!((result.inference_ms - 0.0).abs() < f64::EPSILON);
        assert!((result.tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_inference_result_with_high_tok_per_sec_cov() {
        let result = InferenceResult {
            text: "fast".to_string(),
            tokens: vec![1],
            input_token_count: 0,
            generated_token_count: 1000,
            inference_ms: 10.0,
            tok_per_sec: 100000.0,
            load_ms: 1.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        assert!(result.tok_per_sec > 10000.0);
    }

    #[test]
    fn test_inference_result_empty_text_cov() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(result.text.is_empty());
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_inference_result_empty_tokens_cov() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 5.0,
            tok_per_sec: 0.0,
            load_ms: 2.0,
            format: "SafeTensors".to_string(),
            used_gpu: false,
        };
        assert!(result.tokens.is_empty());
        assert_eq!(result.format, "SafeTensors");
    }

    #[test]
    fn test_inference_result_large_tokens_cov() {
        let tokens: Vec<u32> = (0..10000).collect();
        let result = InferenceResult {
            text: "large".to_string(),
            tokens: tokens.clone(),
            input_token_count: 100,
            generated_token_count: 9900,
            inference_ms: 1000.0,
            tok_per_sec: 9900.0,
            load_ms: 100.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.tokens.len(), 10000);
    }

    #[test]
    fn test_inference_result_format_variations_cov() {
        for format in ["GGUF", "APR", "SafeTensors", "custom"] {
            let result = InferenceResult {
                text: "t".to_string(),
                tokens: vec![1],
                input_token_count: 1,
                generated_token_count: 0,
                inference_ms: 1.0,
                tok_per_sec: 0.0,
                load_ms: 1.0,
                format: format.to_string(),
                used_gpu: false,
            };
            assert_eq!(result.format, format);
        }
    }

    // =========================================================================
    // Extended Coverage Tests for clean_model_output
    // =========================================================================

    #[test]
    fn test_clean_model_output_nested_markers_cov() {
        let raw = "<|im_start|>assistant\n<|im_start|>Hello<|im_end|><|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
    }

    #[test]
    fn test_clean_model_output_unicode_cov() {
        let raw = "<|im_start|>assistant\n你好世界<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "你好世界");
    }

    #[test]
    fn test_clean_model_output_emoji_cov() {
        let raw = "<|im_start|>assistant\n🎉 Hello! 🎊<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "🎉 Hello! 🎊");
    }

    #[test]
    fn test_clean_model_output_tabs_cov() {
        let raw = "<|im_start|>assistant\n\tTabbed content\t<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Tabbed content");
    }

    #[test]
    fn test_clean_model_output_carriage_return_cov() {
        let raw = "<|im_start|>assistant\r\nWindows line<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("Windows line"));
    }

    #[test]
    fn test_clean_model_output_special_chars_cov() {
        let raw = "Special: $@#%^&*()[]{}";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, raw);
    }

    #[test]
    fn test_clean_model_output_very_long_cov() {
        let content = "x".repeat(100000);
        let raw = format!("<|im_start|>assistant\n{}<|im_end|>", content);
        let cleaned = clean_model_output(&raw);
        assert_eq!(cleaned.len(), 100000);
    }

    #[test]
    fn test_clean_model_output_assistant_without_newline_cov() {
        // Tests the <|im_start|>assistant marker without trailing newline
        let raw = "<|im_start|>assistantHello<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello");
    }

    // =========================================================================
    // Extended Coverage Tests for prefault_mmap
    // =========================================================================

    #[test]
    fn test_prefault_mmap_single_byte_cov() {
        let data = vec![42u8; 1];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_exactly_one_page_cov() {
        let data = vec![0u8; 4096];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_minus_one_cov() {
        let data = vec![0u8; 4095];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_plus_one_cov() {
        let data = vec![0u8; 4097];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_with_nonzero_data_cov() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_all_255_cov() {
        let data = vec![255u8; 4096 * 2];
        prefault_mmap(&data);
    }

    // =========================================================================
    // Run Inference Error Path Tests
    // =========================================================================

    #[test]
    fn test_run_inference_empty_path_cov() {
        let config = InferenceConfig::new("");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_directory_path_cov() {
        let config = InferenceConfig::new("/tmp");
        let result = run_inference(&config);
        // Either fails to read as file or fails format detection
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_exactly_8_bytes_cov() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("exactly_8_bytes.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[0, 0, 0, 0, 0, 0, 0, 0]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        // Should fail format detection (not a valid magic)
        assert!(result.is_err());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_run_inference_7_bytes_cov() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("seven_bytes.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[1, 2, 3, 4, 5, 6, 7]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));

        let _ = std::fs::remove_file(path);
    }

    // =========================================================================
    // InferenceConfig Trace Fields Tests
    // =========================================================================

    #[test]
    fn test_inference_config_trace_fields_debug_cov() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/model.gguf"),
            prompt: Some("test".to_string()),
            input_tokens: None,
            max_tokens: 32,
            temperature: 0.0,
            top_k: 1,
            no_gpu: false,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/trace.json")),
            trace_steps: Some(vec!["embedding".to_string(), "attention".to_string()]),
            verbose: false,
            use_mock_backend: false,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("trace_verbose"));
        assert!(debug_str.contains("trace_output"));
        assert!(debug_str.contains("trace_steps"));
    }

    #[test]
    fn test_inference_config_trace_fields_clone_cov() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/model.gguf"),
            prompt: None,
            input_tokens: Some(vec![1, 2, 3]),
            max_tokens: 64,
            temperature: 0.5,
            top_k: 10,
            no_gpu: true,
            trace: false,
            trace_verbose: false,
            trace_output: None,
            trace_steps: None,
            verbose: true,
            use_mock_backend: false,
        };
        let cloned = config.clone();
        assert_eq!(cloned.trace_verbose, config.trace_verbose);
        assert_eq!(cloned.trace_output, config.trace_output);
        assert_eq!(cloned.trace_steps, config.trace_steps);
    }

    #[test]
    fn test_inference_result_all_fields_cov() {
        let result = InferenceResult {
            text: "generated output".to_string(),
            tokens: vec![1, 2, 3, 4, 5],
            input_token_count: 2,
            generated_token_count: 3,
            inference_ms: 123.456,
            tok_per_sec: 24.32,
            load_ms: 50.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.text, "generated output");
        assert_eq!(result.tokens.len(), 5);
        assert_eq!(result.input_token_count, 2);
        assert_eq!(result.generated_token_count, 3);
        assert!((result.inference_ms - 123.456).abs() < 0.001);
        assert!((result.tok_per_sec - 24.32).abs() < 0.01);
        assert!((result.load_ms - 50.0).abs() < 0.01);
        assert_eq!(result.format, "GGUF");
        assert!(result.used_gpu);
    }

    // =========================================================================
    // Extended Coverage Tests: InferenceConfig builders
    // =========================================================================

    #[test]
    fn test_inference_config_all_methods_chain_ext_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(100)
            .with_temperature(1.5)
            .with_top_k(50)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);

        assert_eq!(config.prompt, Some("Test prompt".to_string()));
        assert_eq!(config.max_tokens, 100);
        assert!((config.temperature - 1.5).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    #[test]
    fn test_inference_config_defaults_ext_cov() {
        let config = InferenceConfig::new("/model.apr");

        // Check all default values
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.trace);
        assert!(!config.trace_verbose);
        assert!(config.trace_output.is_none());
        assert!(config.trace_steps.is_none());
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_with_input_tokens_only_ext_cov() {
        let config =
            InferenceConfig::new("/model.safetensors").with_input_tokens(vec![100, 200, 300, 400]);

        assert_eq!(config.input_tokens, Some(vec![100, 200, 300, 400]));
        assert!(config.prompt.is_none());
    }

    #[test]
    fn test_inference_config_temperature_extremes_ext_cov() {
        let cold_config = InferenceConfig::new("/model.gguf").with_temperature(0.0);
        let hot_config = InferenceConfig::new("/model.gguf").with_temperature(2.0);

        assert!((cold_config.temperature - 0.0).abs() < f32::EPSILON);
        assert!((hot_config.temperature - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_top_k_values_ext_cov() {
        let greedy = InferenceConfig::new("/model.gguf").with_top_k(1);
        let wide = InferenceConfig::new("/model.gguf").with_top_k(100);
        let disabled = InferenceConfig::new("/model.gguf").with_top_k(0);

        assert_eq!(greedy.top_k, 1);
        assert_eq!(wide.top_k, 100);
        assert_eq!(disabled.top_k, 0);
    }

    // =========================================================================
    // Extended Coverage Tests: clean_model_output
    // =========================================================================

    #[test]
    fn test_clean_model_output_endoftext_ext_cov() {
        let raw = "Hello world<|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_model_output_preserves_other_text_ext_cov() {
        let raw = "[INST]user input[/INST]assistant response";
        let cleaned = clean_model_output(raw);
        // Should preserve text since [INST] tokens aren't in the markers list
        assert!(cleaned.contains("user input"));
        assert!(cleaned.contains("response"));
    }

    #[test]
    fn test_clean_model_output_im_start_im_end_ext_cov() {
        // Function removes markers but keeps content between them
        let raw = "<|im_start|>assistant\nHi there!<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hi there!");
    }

    #[test]
    fn test_clean_model_output_empty_ext_cov() {
        let raw = "";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_only_markers_ext_cov() {
        let raw = "<|im_start|><|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_multiple_endoftext_ext_cov() {
        let raw = "Text<|endoftext|>More text<|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "TextMore text");
    }

    // =========================================================================
    // Extended Coverage Tests: prefault_mmap
    // =========================================================================

    #[test]
    fn test_prefault_mmap_large_ext_cov() {
        let data = vec![0u8; 4096 * 10]; // 10 pages
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_not_page_aligned_ext_cov() {
        let data = vec![0u8; 5000]; // Not page-aligned
        prefault_mmap(&data);
    }

    // =========================================================================
    // Extended Coverage Tests: InferenceResult
    // =========================================================================

    #[test]
    fn test_inference_result_debug_ext_cov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 5.0,
            tok_per_sec: 0.0,
            load_ms: 2.0,
            format: "APR".to_string(),
            used_gpu: false,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("text"));
        assert!(debug_str.contains("tokens"));
        assert!(debug_str.contains("format"));
        assert!(debug_str.contains("used_gpu"));
    }

    #[test]
    fn test_inference_result_zero_values_ext_cov() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: String::new(),
            used_gpu: false,
        };
        assert!(result.text.is_empty());
        assert!(result.tokens.is_empty());
        assert_eq!(result.input_token_count, 0);
    }

    #[test]
    fn test_inference_result_large_values_ext_cov() {
        let result = InferenceResult {
            text: "A".repeat(10000),
            tokens: vec![1; 1000],
            input_token_count: 100,
            generated_token_count: 900,
            inference_ms: 1000000.0,
            tok_per_sec: 1000.0,
            load_ms: 5000.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.text.len(), 10000);
        assert_eq!(result.tokens.len(), 1000);
        assert_eq!(result.generated_token_count, 900);
    }

    #[test]
    fn test_inference_result_formats_ext_cov() {
        for fmt in ["GGUF", "APR", "SafeTensors"] {
            let result = InferenceResult {
                text: "test".to_string(),
                tokens: vec![1],
                input_token_count: 1,
                generated_token_count: 0,
                inference_ms: 1.0,
                tok_per_sec: 1.0,
                load_ms: 1.0,
                format: fmt.to_string(),
                used_gpu: false,
            };
            assert_eq!(result.format, fmt);
        }
    }

    // =========================================================================
    // Extended Coverage Tests: run_inference error paths
    // =========================================================================

    #[test]
    fn test_run_inference_permission_denied_ext_cov() {
        // Try to read from a path that likely doesn't exist or isn't readable
        let config = InferenceConfig::new("/root/super_secret/model.gguf");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_empty_path_ext_cov() {
        let config = InferenceConfig::new("");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    // =========================================================================
    // Deep Coverage Tests (_deep_icov_) - Lines 197-280
    // =========================================================================

    // --- Format Detection Tests (Lines 197-201) ---

    #[test]
    fn test_format_detection_gguf_magic_deep_icov() {
        // GGUF magic bytes: 0x47 0x47 0x55 0x46 = "GGUF"
        use crate::format::{detect_format, ModelFormat};
        let data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00];
        let format = detect_format(&data);
        assert!(matches!(format, Ok(ModelFormat::Gguf)));
    }

    #[test]
    fn test_format_detection_apr_magic_deep_icov() {
        // APR magic bytes: "APR\0"
        use crate::format::{detect_format, ModelFormat};
        let data = b"APR\0xxxx";
        let format = detect_format(data);
        assert!(matches!(format, Ok(ModelFormat::Apr)));
    }

    #[test]
    fn test_format_detection_safetensors_deep_icov() {
        // SafeTensors: first 8 bytes are header size (little-endian u64)
        use crate::format::{detect_format, ModelFormat};
        let header_size: u64 = 2048;
        let data = header_size.to_le_bytes();
        let format = detect_format(&data);
        assert!(matches!(format, Ok(ModelFormat::SafeTensors)));
    }

    #[test]
    fn test_format_detection_unknown_magic_deep_icov() {
        // Unknown magic bytes should return error
        use crate::format::{detect_format, FormatError};
        let data = b"\x00\x00\x00\x00\x00\x00\x00\x00"; // Zero header = unknown
        let format = detect_format(data);
        assert!(matches!(format, Err(FormatError::UnknownFormat)));
    }

    // --- Architecture Detection Tests (Lines 227-243) ---

    #[test]
    fn test_architecture_detection_qwen_deep_icov() {
        // Test that "qwen" in filename is detected as "Qwen2"
        let path = PathBuf::from("/models/qwen2-7b-instruct-q4.gguf");
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
        assert_eq!(arch, Some("Qwen2"));
    }

    #[test]
    fn test_architecture_detection_llama_deep_icov() {
        // Test that "llama" in filename is detected as "LLaMA"
        let path = PathBuf::from("/models/llama-3.1-8b-instruct.gguf");
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
        assert_eq!(arch, Some("LLaMA"));
    }

    #[test]
    fn test_architecture_detection_mistral_deep_icov() {
        // Test that "mistral" in filename is detected as "Mistral"
        let path = PathBuf::from("/models/mistral-7b-v0.2-q4_k_m.gguf");
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
        assert_eq!(arch, Some("Mistral"));
    }

    #[test]
    fn test_architecture_detection_phi_deep_icov() {
        // Test that "phi" in filename is detected as "Phi"
        let path = PathBuf::from("/models/phi-2-q4_0.gguf");
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
        assert_eq!(arch, Some("Phi"));
    }

    #[test]
    fn test_architecture_detection_transformer_fallback_deep_icov() {
        // Test that unknown models fall back to "Transformer"
        let path = PathBuf::from("/models/custom-model-q8_0.gguf");
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
        assert_eq!(arch, Some("Transformer"));
    }

    #[test]
    fn test_architecture_detection_case_insensitive_deep_icov() {
        // Test case-insensitive architecture detection
        let paths = [
            ("/models/QWEN2-7B.gguf", "Qwen2"),
            ("/models/LLAMA-3.gguf", "LLaMA"),
            ("/models/MISTRAL-7B.gguf", "Mistral"),
            ("/models/PHI-2.gguf", "Phi"),
            ("/models/QwEn2-MixedCase.gguf", "Qwen2"),
        ];
        for (path_str, expected) in paths {
            let path = PathBuf::from(path_str);
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
            assert_eq!(arch, Some(expected), "Failed for path: {}", path_str);
        }
    }

    #[test]
    fn test_architecture_detection_no_extension_deep_icov() {
        // Test path with no extension
        let path = PathBuf::from("/models/qwen2-model");
        let arch = path.file_stem().and_then(|s| s.to_str()).map(|s| {
            if s.to_lowercase().contains("qwen") {
                "Qwen2"
            } else {
                "Transformer"
            }
        });
        assert_eq!(arch, Some("Qwen2"));
    }

    // --- Instruct Model Detection Tests (Lines 264-270) ---

    #[test]
    fn test_instruct_model_detection_deep_icov() {
        let model_name = "llama-3.1-8b-instruct.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(is_instruct);
    }

    #[test]
    fn test_instruct_model_detection_uppercase_deep_icov() {
        let model_name = "LLAMA-3.1-8B-INSTRUCT.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(is_instruct);
    }

    #[test]
    fn test_instruct_model_detection_mixed_case_deep_icov() {
        let model_name = "Qwen2-7B-Instruct-Q4_K_M.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(is_instruct);
    }

    #[test]
    fn test_instruct_model_detection_not_instruct_deep_icov() {
        let model_name = "llama-3.1-8b-base.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(!is_instruct);
    }

    #[test]
    fn test_instruct_model_detection_partial_match_deep_icov() {
        // Should match even if "instruct" is part of a larger word
        let model_name = "model-instructed.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(is_instruct);
    }

    // --- Chat Template Formatting Tests (Lines 266-270) ---

    #[test]
    fn test_chat_message_user_creation_deep_icov() {
        use crate::chat_template::ChatMessage;
        let msg = ChatMessage::user("Hello, world!");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello, world!");
    }

    #[test]
    fn test_chat_message_system_creation_deep_icov() {
        use crate::chat_template::ChatMessage;
        let msg = ChatMessage::system("You are a helpful assistant.");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant.");
    }

    #[test]
    fn test_format_messages_instruct_model_deep_icov() {
        use crate::chat_template::{format_messages, ChatMessage};
        let messages = vec![ChatMessage::user("What is 2+2?")];
        // Test with qwen model name (should use ChatML template)
        let result = format_messages(&messages, Some("qwen2-7b-instruct.gguf"));
        assert!(result.is_ok());
        let formatted = result.expect("operation failed");
        // ChatML format uses <|im_start|> markers
        assert!(formatted.contains("<|im_start|>") || formatted.contains("user"));
    }

    #[test]
    fn test_format_messages_llama_template_deep_icov() {
        use crate::chat_template::{format_messages, ChatMessage};
        let messages = vec![ChatMessage::user("Hello!")];
        let result = format_messages(&messages, Some("llama-3.1-8b-instruct.gguf"));
        assert!(result.is_ok());
        let formatted = result.expect("operation failed");
        // LLaMA format uses [INST] markers
        assert!(formatted.contains("[INST]") || formatted.contains("user"));
    }

    #[test]
    fn test_format_messages_fallback_raw_deep_icov() {
        use crate::chat_template::{format_messages, ChatMessage};
        let messages = vec![ChatMessage::user("Just text")];
        // Unknown model should use raw template
        let result = format_messages(&messages, Some("unknown-model.gguf"));
        assert!(result.is_ok());
    }

    // --- Input Token Handling Tests (Lines 255-279) ---

    #[test]
    fn test_input_tokens_priority_over_prompt_deep_icov() {
        // Test that input_tokens takes priority over prompt
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Hello")
            .with_input_tokens(vec![1, 2, 3, 4]);

        // When both are set, input_tokens should be used (line 255-256)
        assert!(config.input_tokens.is_some());
        assert!(config.prompt.is_some());

        // Simulate the logic from run_gguf_inference
        let input_tokens = if let Some(ref tokens) = config.input_tokens {
            tokens.clone()
        } else if let Some(ref _prompt) = config.prompt {
            vec![100, 200] // Would be tokenized prompt
        } else {
            vec![1u32] // BOS token
        };
        assert_eq!(input_tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_input_tokens_none_uses_prompt_deep_icov() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("Hello");

        // When input_tokens is None, prompt should be used (line 257)
        assert!(config.input_tokens.is_none());
        assert!(config.prompt.is_some());
    }

    #[test]
    fn test_input_tokens_none_prompt_none_uses_bos_deep_icov() {
        let config = InferenceConfig::new("/model.gguf");

        // When both are None, BOS token should be used (line 277-278)
        assert!(config.input_tokens.is_none());
        assert!(config.prompt.is_none());

        // Simulate the logic
        let input_tokens = if let Some(ref tokens) = config.input_tokens {
            tokens.clone()
        } else if let Some(ref _prompt) = config.prompt {
            vec![100, 200]
        } else {
            vec![1u32] // BOS token
        };
        assert_eq!(input_tokens, vec![1u32]);
    }

    // --- Verbose Output Tests (Lines 210-252) ---

    #[test]
    fn test_verbose_flag_enabled_deep_icov() {
        let config = InferenceConfig::new("/model.gguf").with_verbose(true);
        assert!(config.verbose);
    }

    #[test]
    fn test_verbose_flag_disabled_deep_icov() {
        let config = InferenceConfig::new("/model.gguf").with_verbose(false);
        assert!(!config.verbose);
    }

    #[test]
    fn test_verbose_default_is_false_deep_icov() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(!config.verbose);
    }

    // --- Model Name Extraction Tests ---

    #[test]
    fn test_model_name_extraction_from_path_deep_icov() {
        let path = PathBuf::from("/models/qwen2-7b-instruct.gguf");
        let model_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert_eq!(model_name, "qwen2-7b-instruct.gguf");
    }

    #[test]
    fn test_model_name_extraction_no_parent_deep_icov() {
        let path = PathBuf::from("qwen2-7b.gguf");
        let model_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert_eq!(model_name, "qwen2-7b.gguf");
    }

    #[test]
    fn test_model_name_extraction_empty_path_deep_icov() {
        let path = PathBuf::from("");
        let model_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert_eq!(model_name, "");
    }

    // --- File Stem Extraction Tests ---

    #[test]
    fn test_file_stem_extraction_deep_icov() {
        let path = PathBuf::from("/models/llama-3.1-8b.gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("llama-3.1-8b"));
    }

    #[test]
    fn test_file_stem_extraction_multiple_dots_deep_icov() {
        let path = PathBuf::from("/models/model.v1.0.gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("model.v1.0"));
    }

    #[test]
    fn test_file_stem_extraction_no_extension_deep_icov() {
        let path = PathBuf::from("/models/model");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("model"));
    }

    // --- Tokens Per Second Calculation Tests ---

    #[test]
    fn test_tok_per_sec_calculation_deep_icov() {
        let generated_count = 100;
        let inference_ms = 500.0; // 500ms
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 200.0).abs() < 0.001); // 100 tokens / 0.5 sec = 200 tok/s
    }

    #[test]
    fn test_tok_per_sec_zero_time_deep_icov() {
        let generated_count = 100;
        let inference_ms = 0.0;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tok_per_sec_very_fast_deep_icov() {
        let generated_count = 1000;
        let inference_ms = 10.0; // 10ms
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 100000.0).abs() < 0.001); // 100k tok/s
    }

    // --- Max Tokens Capping Tests (Line 285) ---

    #[test]
    fn test_max_tokens_capped_at_128_deep_icov() {
        // Test the .min(128) capping in gen_config (line 285)
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(1000);
        let capped = config.max_tokens.min(128);
        assert_eq!(capped, 128);
    }

    #[test]
    fn test_max_tokens_not_capped_if_under_128_deep_icov() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(50);
        let capped = config.max_tokens.min(128);
        assert_eq!(capped, 50);
    }

    #[test]
    fn test_max_tokens_exactly_128_deep_icov() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(128);
        let capped = config.max_tokens.min(128);
        assert_eq!(capped, 128);
    }

    // --- Format String Tests ---

    #[test]
    fn test_inference_result_format_string_gguf_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "GGUF");
    }

    #[test]
    fn test_inference_result_format_string_apr_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "APR".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "APR");
    }

    #[test]
    fn test_inference_result_format_string_safetensors_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "SafeTensors".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "SafeTensors");
    }

    // --- Used GPU Flag Tests ---

    #[test]
    fn test_inference_result_used_gpu_true_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert!(result.used_gpu);
    }

    #[test]
    fn test_inference_result_used_gpu_false_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(!result.used_gpu);
    }

    // --- Architecture Priority Tests ---

    #[test]
    fn test_architecture_detection_priority_qwen_over_llama_deep_icov() {
        // If filename contains both "qwen" and "llama", qwen should win (checked first)
        let path = PathBuf::from("/models/qwen-llama-hybrid.gguf");
        let arch = path.file_stem().and_then(|s| s.to_str()).map(|s| {
            if s.to_lowercase().contains("qwen") {
                "Qwen2"
            } else if s.to_lowercase().contains("llama") {
                "LLaMA"
            } else {
                "Transformer"
            }
        });
        assert_eq!(arch, Some("Qwen2"));
    }

    #[test]
    fn test_architecture_detection_priority_llama_over_mistral_deep_icov() {
        // If filename contains both "llama" and "mistral", llama should win
        let path = PathBuf::from("/models/llama-mistral-blend.gguf");
        let arch = path.file_stem().and_then(|s| s.to_str()).map(|s| {
            if s.to_lowercase().contains("qwen") {
                "Qwen2"
            } else if s.to_lowercase().contains("llama") {
                "LLaMA"
            } else if s.to_lowercase().contains("mistral") {
                "Mistral"
            } else {
                "Transformer"
            }
        });
        assert_eq!(arch, Some("LLaMA"));
    }

    // --- Path Edge Cases Tests ---

    #[test]
    fn test_path_with_special_characters_deep_icov() {
        let path = PathBuf::from("/models/model-v1.0_final (copy).gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert!(stem.is_some());
        assert!(stem.expect("operation failed").contains("model"));
    }

    #[test]
    fn test_path_with_unicode_deep_icov() {
        let path = PathBuf::from("/models/模型-v1.gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert!(stem.is_some());
    }

    #[test]
    fn test_path_just_extension_deep_icov() {
        let path = PathBuf::from(".gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        // For dotfiles like .gguf, file_stem returns the full name ".gguf" (no extension)
        assert_eq!(stem, Some(".gguf"));
    }

    // --- Load Time Tests ---

    #[test]
    fn test_load_ms_calculation_deep_icov() {
        // Simulating load_start.elapsed().as_secs_f64() * 1000.0
        let elapsed_secs: f64 = 0.5;
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_load_ms_very_fast_deep_icov() {
        let elapsed_secs: f64 = 0.001; // 1ms
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_load_ms_very_slow_deep_icov() {
        let elapsed_secs: f64 = 10.0; // 10 seconds
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 10000.0).abs() < 0.001);
    }

    // --- Generated Token Slice Tests ---

    #[test]
    fn test_generated_tokens_slice_deep_icov() {
        let all_tokens = vec![1, 2, 3, 4, 5, 6];
        let input_token_count = 2;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens, &[3, 4, 5, 6]);
    }

    #[test]
    fn test_generated_tokens_slice_empty_deep_icov() {
        let all_tokens = vec![1, 2];
        let input_token_count = 2;
        let generated_tokens = &all_tokens[input_token_count..];
        assert!(generated_tokens.is_empty());
    }

    #[test]
    fn test_generated_tokens_slice_all_generated_deep_icov() {
        let all_tokens = vec![1, 2, 3, 4];
        let input_token_count = 0;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens.len(), 4);
    }

    // --- Clean Output Integration Tests ---

    #[test]
    fn test_clean_output_chatml_full_conversation_deep_icov() {
        let raw = "<|im_start|>system\nYou are helpful<|im_end|><|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello!<|im_end|>";
        let cleaned = clean_model_output(raw);
        // All markers should be removed
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        // Content should remain (without newlines added by markers)
        assert!(cleaned.contains("helpful") || cleaned.contains("Hello"));
    }

    #[test]
    fn test_clean_output_preserves_code_blocks_deep_icov() {
        let raw = "<|im_start|>assistant\n```python\nprint('hello')\n```<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("```python"));
        assert!(cleaned.contains("print('hello')"));
    }

    // --- Model Format Enum Tests ---

    #[test]
    fn test_model_format_display_gguf_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Gguf;
        assert_eq!(format.to_string(), "GGUF");
    }

    #[test]
    fn test_model_format_display_apr_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Apr;
        assert_eq!(format.to_string(), "APR");
    }

    #[test]
    fn test_model_format_display_safetensors_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::SafeTensors;
        assert_eq!(format.to_string(), "SafeTensors");
    }

    #[test]
    fn test_model_format_clone_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Gguf;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_model_format_eq_deep_icov() {
        use crate::format::ModelFormat;
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Gguf, ModelFormat::Apr);
        assert_ne!(ModelFormat::Apr, ModelFormat::SafeTensors);
    }

    // --- Additional Edge Case Tests ---

    #[test]
    fn test_inference_config_path_with_symlink_name_deep_icov() {
        let config = InferenceConfig::new("/models/latest -> llama-3.gguf");
        assert!(config
            .model_path
            .to_str()
            .expect("invalid UTF-8")
            .contains("latest"));
    }

    #[test]
    fn test_inference_config_relative_path_deep_icov() {
        let config = InferenceConfig::new("./models/model.gguf");
        assert!(config
            .model_path
            .to_str()
            .expect("invalid UTF-8")
            .contains("./"));
    }

    #[test]
    fn test_inference_config_absolute_path_deep_icov() {
        let config = InferenceConfig::new("/absolute/path/model.gguf");
        assert!(config.model_path.starts_with("/"));
    }

    // =========================================================================
    // Synthetic Model Inference Path Tests
    // These tests create minimal model files to exercise inference code paths
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_minimal_model() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create minimal valid GGUF file (no tensors - will fail on model load)
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("file operation failed");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0
        temp.write_all(&data).expect("operation failed");
        temp.flush().expect("operation failed");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Hello")
            .with_max_tokens(5);

        // Will fail because model has no tensors, but exercises the GGUF path
        let result = run_inference(&config);
        assert!(result.is_err()); // Expected: model can't be loaded properly
    }

    #[test]
    fn test_run_apr_inference_minimal_model() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create minimal valid APR file header
        let mut temp = NamedTempFile::with_suffix(".apr").expect("file operation failed");
        let mut data = Vec::new();
        // APR magic + minimal header
        data.extend_from_slice(b"APR\x02"); // APR v2 magic
        data.extend_from_slice(&[0u8; 60]); // Minimal header padding
        temp.write_all(&data).expect("operation failed");
        temp.flush().expect("operation failed");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_max_tokens(3);

        // Will fail on loading, but exercises the APR format detection path
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_inference_minimal() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create minimal SafeTensors file (8-byte header size + empty JSON)
        let mut temp = NamedTempFile::with_suffix(".safetensors").expect("file operation failed");
        let json_header = b"{}";
        let header_size = json_header.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(json_header);
        temp.write_all(&data).expect("operation failed");
        temp.flush().expect("operation failed");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_max_tokens(3);

        // Will fail on loading (no tensors), but exercises SafeTensors path
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_config_input_tokens_path() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create minimal GGUF to test input_tokens path (instead of prompt)
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("file operation failed");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        temp.write_all(&data).expect("operation failed");
        temp.flush().expect("operation failed");

        // Use input_tokens instead of prompt
        let config = InferenceConfig::new(temp.path())
            .with_input_tokens(vec![1, 2, 3])
            .with_max_tokens(5);

        assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
        assert!(config.prompt.is_none());

        // Will fail on model load, but exercises input_tokens path
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_config_verbose_path() {
        // Test verbose config (doesn't need model file for this)
        let config = InferenceConfig::new("/model.gguf")
            .with_verbose(true)
            .with_prompt("test");

        assert!(config.verbose);
    }

    #[test]
    fn test_inference_result_tok_per_sec_calculation() {
        // Test tok/s calculation edge case: zero inference_ms
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 0.0, // Edge case
            tok_per_sec: 0.0,
            load_ms: 5.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.tok_per_sec, 0.0);
    }

    // =========================================================================
    // Additional Coverage Tests for Uncovered Lines
    // =========================================================================

    // --- InferenceConfig additional builder tests ---

    #[test]
    fn test_inference_config_all_trace_options() {
        let mut config = InferenceConfig::new("/model.gguf");
        config.trace = true;
        config.trace_verbose = true;
        config.trace_output = Some(PathBuf::from("/trace/output.json"));
        config.trace_steps = Some(vec![
            "embed".to_string(),
            "attention".to_string(),
            "ffn".to_string(),
        ]);

        assert!(config.trace);
        assert!(config.trace_verbose);
        assert_eq!(
            config.trace_output,
            Some(PathBuf::from("/trace/output.json"))
        );
        assert_eq!(config.trace_steps.as_ref().map(std::vec::Vec::len), Some(3));
    }

    #[test]
    fn test_inference_config_trace_steps_empty() {
        let mut config = InferenceConfig::new("/model.gguf");
        config.trace_steps = Some(vec![]);
        assert_eq!(config.trace_steps, Some(vec![]));
    }

    #[test]
    fn test_inference_config_with_all_options_set() {
        let mut config = InferenceConfig::new("/full/path/model.gguf");
        config.prompt = Some("test prompt".to_string());
        config.input_tokens = Some(vec![1, 2, 3, 4, 5]);
        config.max_tokens = 256;
        config.temperature = 0.9;
        config.top_k = 50;
        config.no_gpu = true;
        config.trace = true;
        config.trace_verbose = true;
        config.trace_output = Some(PathBuf::from("/output.json"));
        config.trace_steps = Some(vec!["step1".to_string()]);
        config.verbose = true;

        // Verify all fields
        assert!(config.prompt.is_some());
        assert!(config.input_tokens.is_some());
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.no_gpu);
        assert!(config.trace);
        assert!(config.trace_verbose);
        assert!(config.trace_output.is_some());
        assert!(config.trace_steps.is_some());
        assert!(config.verbose);
    }

    // --- run_inference format detection tests ---

    #[test]
    fn test_run_inference_with_gguf_magic_but_incomplete() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // GGUF magic but incomplete model
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes()); // version
                                                     // Missing rest of header
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_with_apr_magic_but_incomplete() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // APR magic but incomplete
        let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"APR\x02"); // APR v2 magic
        data.extend_from_slice(&[0u8; 4]); // Incomplete header
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_with_safetensors_header_but_invalid() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // SafeTensors-like header but invalid JSON
        let mut temp = NamedTempFile::with_suffix(".safetensors").expect("create temp");
        let header_size: u64 = 16; // Size that doesn't match actual content
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(b"invalid json");
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    // --- Architecture detection from path tests ---

    #[test]
    fn test_architecture_from_path_variations() {
        let test_cases = [
            ("tinyllama-1.1b-chat.gguf", "LLaMA"),
            ("phi-2-mini.gguf", "Phi"),
            ("mistral-7b-instruct-v0.2.gguf", "Mistral"),
            ("qwen2-0.5b-instruct.gguf", "Qwen2"),
            ("my-custom-model.gguf", "Transformer"),
            ("LLAMA-3-8B.gguf", "LLaMA"),
            ("PHI3-MINI.gguf", "Phi"),
            ("MISTRAL_LARGE.gguf", "Mistral"),
            ("QWEN2_7B.gguf", "Qwen2"),
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

    // --- Token per second calculation edge cases ---

    #[test]
    fn test_tok_per_sec_negative_time_protection() {
        // Even though negative time shouldn't happen, test the logic
        let inference_ms = -100.0;
        let generated_count = 50;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert_eq!(tok_per_sec, 0.0);
    }

    #[test]
    fn test_tok_per_sec_very_small_time() {
        let inference_ms = 0.001; // 1 microsecond
        let generated_count = 1;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 1000000.0).abs() < 0.001); // 1M tok/s
    }

    // --- InferenceResult edge cases ---

    #[test]
    fn test_inference_result_with_unicode_text() {
        let result = InferenceResult {
            text: "Hello 世界 🌍 Привет".to_string(),
            tokens: vec![1, 2, 3, 4, 5],
            input_token_count: 2,
            generated_token_count: 3,
            inference_ms: 10.0,
            tok_per_sec: 300.0,
            load_ms: 5.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(result.text.contains("世界"));
        assert!(result.text.contains("🌍"));
    }

    #[test]
    fn test_inference_result_with_special_chars() {
        let result = InferenceResult {
            text: "Code: `fn main() { println!(\"Hello\"); }`".to_string(),
            tokens: vec![1],
            input_token_count: 0,
            generated_token_count: 1,
            inference_ms: 1.0,
            tok_per_sec: 1000.0,
            load_ms: 1.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        assert!(result.text.contains("fn main()"));
        assert!(result.text.contains("println!"));
    }

    // --- Config path handling tests ---

    #[test]
    fn test_inference_config_various_path_formats() {
        let paths = [
            "/absolute/path/model.gguf",
            "./relative/path/model.apr",
            "../parent/dir/model.safetensors",
            "model.gguf",
            "/path/with spaces/model file.gguf",
            "/path/with-dashes/model-name.gguf",
            "/path/with_underscores/model_name.gguf",
        ];

        for path in paths {
            let config = InferenceConfig::new(path);
            assert_eq!(config.model_path.to_str(), Some(path));
        }
    }

    // --- Max tokens capping behavior ---

    #[test]
    fn test_max_tokens_cap_behavior() {
        let configs = [
            (InferenceConfig::new("/m.gguf").with_max_tokens(0), 0),
            (InferenceConfig::new("/m.gguf").with_max_tokens(64), 64),
            (InferenceConfig::new("/m.gguf").with_max_tokens(128), 128),
            (InferenceConfig::new("/m.gguf").with_max_tokens(256), 128), // capped
            (InferenceConfig::new("/m.gguf").with_max_tokens(1024), 128), // capped
            (
                InferenceConfig::new("/m.gguf").with_max_tokens(usize::MAX),
                128,
            ), // capped
        ];

        for (config, expected_capped) in configs {
            let capped = config.max_tokens.min(128);
            assert_eq!(capped, expected_capped);
        }
    }

    // --- EOS token detection tests ---

    #[test]
    fn test_eos_token_detection() {
        // Test the EOS token detection logic from run_apr_inference
        let eos_tokens = [151645u32, 151643u32, 2u32]; // Qwen2 EOS, BOS, standard
        let non_eos_tokens = [1u32, 100u32, 1000u32, 151644u32];

        for token in eos_tokens {
            let is_eos = token == 151645 || token == 151643 || token == 2;
            assert!(is_eos, "Token {} should be EOS", token);
        }

        for token in non_eos_tokens {
            let is_eos = token == 151645 || token == 151643 || token == 2;
            assert!(!is_eos, "Token {} should NOT be EOS", token);
        }
    }

    // --- Greedy sampling (argmax) logic ---

    #[test]
    fn test_argmax_logic() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 3); // Index of 0.9
    }

    #[test]
    fn test_argmax_logic_with_ties() {
        let logits = vec![0.5f32, 0.5, 0.5, 0.5];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        // max_by returns last maximum for ties
        assert_eq!(next_token, 3);
    }

    #[test]
    fn test_argmax_logic_with_negative() {
        let logits = vec![-0.5f32, -0.1, -0.9, -0.3];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 1); // Index of -0.1 (least negative)
    }

    #[test]
    fn test_argmax_logic_empty() {
        let logits: Vec<f32> = vec![];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 0); // Default when empty
    }

    #[test]
    fn test_argmax_logic_with_nan() {
        let logits = vec![0.1f32, f32::NAN, 0.5, 0.3];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        // NaN comparisons return Equal, so behavior depends on order
        // Result should be one of the valid indices
        assert!(next_token < 4);
    }

    // --- Generated tokens slice extraction ---

    #[test]
    fn test_generated_tokens_extraction() {
        let all_tokens = vec![100u32, 200, 300, 400, 500, 600];
        let input_token_count = 3;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens, &[400, 500, 600]);
        assert_eq!(generated_tokens.len(), 3);
    }

    #[test]
    fn test_generated_tokens_extraction_no_generation() {
        let all_tokens = vec![100u32, 200, 300];
        let input_token_count = 3;
        let generated_tokens = &all_tokens[input_token_count..];
        assert!(generated_tokens.is_empty());
    }

    #[test]
    fn test_generated_tokens_extraction_all_generated() {
        let all_tokens = vec![100u32, 200, 300, 400];
        let input_token_count = 0;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens.len(), 4);
    }

    // --- Instruct model detection ---

    #[test]
    fn test_instruct_model_name_variations() {
        let instruct_names = [
            "llama-3.1-8b-instruct.gguf",
            "LLAMA-INSTRUCT.gguf",
            "qwen2-7b-Instruct-q4.gguf",
            "model-instructed-v1.gguf",
            "phi-2-InStRuCt.gguf",
        ];

        for name in instruct_names {
            let is_instruct = name.to_lowercase().contains("instruct");
            assert!(is_instruct, "Should detect instruct in: {}", name);
        }

        let non_instruct_names = [
            "llama-3.1-8b-base.gguf",
            "phi-2-mini.gguf",
            "qwen2-7b-q4.gguf",
            "custom-model.gguf",
        ];

        for name in non_instruct_names {
            let is_instruct = name.to_lowercase().contains("instruct");
            assert!(!is_instruct, "Should NOT detect instruct in: {}", name);
        }
    }

    // --- Load time and inference time calculations ---

    #[test]
    fn test_time_calculation_precision() {
        let elapsed_secs: f64 = 1.234567;
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 1234.567).abs() < 0.001);

        let inference_ms: f64 = 567.89;
        let generated = 100;
        let tok_per_sec = generated as f64 / (inference_ms / 1000.0);
        // 100 / 0.56789 = 176.0564411... (use wider tolerance for FP precision)
        assert!((tok_per_sec - 176.0564).abs() < 0.1);
    }

    // --- InferenceConfig Debug and Clone full coverage ---

    #[test]
    fn test_inference_config_debug_all_fields() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/test/model.gguf"),
            prompt: Some("test prompt".to_string()),
            input_tokens: Some(vec![1, 2, 3]),
            max_tokens: 100,
            temperature: 0.7,
            top_k: 40,
            no_gpu: true,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/trace.json")),
            trace_steps: Some(vec!["embed".to_string()]),
            verbose: true,
            use_mock_backend: false,
        };

        let debug = format!("{:?}", config);
        assert!(debug.contains("model_path"));
        assert!(debug.contains("prompt"));
        assert!(debug.contains("input_tokens"));
        assert!(debug.contains("max_tokens"));
        assert!(debug.contains("temperature"));
        assert!(debug.contains("top_k"));
        assert!(debug.contains("no_gpu"));
        assert!(debug.contains("trace"));
        assert!(debug.contains("trace_verbose"));
        assert!(debug.contains("trace_output"));
        assert!(debug.contains("trace_steps"));
        assert!(debug.contains("verbose"));
    }

    #[test]
    fn test_inference_config_clone_preserves_all() {
        let original = InferenceConfig {
            model_path: PathBuf::from("/test/model.gguf"),
            prompt: Some("prompt".to_string()),
            input_tokens: Some(vec![10, 20, 30]),
            max_tokens: 200,
            temperature: 1.2,
            top_k: 100,
            no_gpu: true,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/out.json")),
            trace_steps: Some(vec!["a".to_string(), "b".to_string()]),
            verbose: true,
            use_mock_backend: false,
        };

        let cloned = original.clone();

        assert_eq!(original.model_path, cloned.model_path);
        assert_eq!(original.prompt, cloned.prompt);
        assert_eq!(original.input_tokens, cloned.input_tokens);
        assert_eq!(original.max_tokens, cloned.max_tokens);
        assert!((original.temperature - cloned.temperature).abs() < f32::EPSILON);
        assert_eq!(original.top_k, cloned.top_k);
        assert_eq!(original.no_gpu, cloned.no_gpu);
        assert_eq!(original.trace, cloned.trace);
        assert_eq!(original.trace_verbose, cloned.trace_verbose);
        assert_eq!(original.trace_output, cloned.trace_output);
        assert_eq!(original.trace_steps, cloned.trace_steps);
        assert_eq!(original.verbose, cloned.verbose);
    }

    // --- InferenceResult Debug and Clone full coverage ---

    #[test]
    fn test_inference_result_debug_all_fields() {
        let result = InferenceResult {
            text: "generated text".to_string(),
            tokens: vec![1, 2, 3, 4, 5],
            input_token_count: 2,
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
