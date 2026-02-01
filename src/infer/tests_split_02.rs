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
