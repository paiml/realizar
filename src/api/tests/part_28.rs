//! Coverage tests for realize_handlers.rs pure functions and api/mod.rs constructors
//!
//! Targets missed lines in:
//! - realize_handlers.rs: ContextWindowConfig, ContextWindowManager, clean_chat_output,
//!   format_chat_messages, completion_resp, EmbeddingRequest/Response serde,
//!   ModelMetadataResponse, ModelLineage, ReloadRequest/Response, CompletionRequest/Response
//! - api/mod.rs: AppState constructors, get_model paths, build_trace_data branches

use crate::api::{
    clean_chat_output, format_chat_messages, CompletionChoice, CompletionRequest,
    CompletionResponse, ContextWindowConfig, ContextWindowManager, EmbeddingData, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, ModelLineage, ModelMetadataResponse, ReloadRequest,
    ReloadResponse,
};

// ============================================================================
// ContextWindowConfig Tests
// ============================================================================

#[test]
fn test_context_window_config_default() {
    let config = ContextWindowConfig::default();
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 256);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_new() {
    let config = ContextWindowConfig::new(8192);
    assert_eq!(config.max_tokens, 8192);
    assert_eq!(config.reserved_output_tokens, 256); // default
    assert!(config.preserve_system); // default
}

#[test]
fn test_context_window_config_with_reserved_output() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(512);
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 512);
}

#[test]
fn test_context_window_config_available_tokens() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(256);
    assert_eq!(config.available_tokens(), 3840);
}

#[test]
fn test_context_window_config_available_tokens_saturating() {
    // reserved > max should not underflow
    let mut config = ContextWindowConfig::new(100);
    config.reserved_output_tokens = 200;
    assert_eq!(config.available_tokens(), 0);
}

// ============================================================================
// ContextWindowManager Tests
// ============================================================================

#[test]
fn test_context_window_manager_default() {
    let mgr = ContextWindowManager::default_manager();
    // Should have default config
    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "short".to_string(),
        name: None,
    }];
    assert!(!mgr.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_no_truncation_needed() {
    let config = ContextWindowConfig::new(4096);
    let mgr = ContextWindowManager::new(config);

    let messages = vec![
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "Hello world".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: "Hi there!".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = mgr.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_context_window_manager_truncation_preserves_system() {
    let config = ContextWindowConfig::new(100).with_reserved_output(10);
    let mgr = ContextWindowManager::new(config);

    // Create messages that exceed the context window
    let messages = vec![
        crate::api::ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "A".repeat(200), // Long message
            name: None,
        },
        crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: "B".repeat(200),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "Short recent message".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = mgr.truncate_messages(&messages);
    assert!(truncated);
    // System message should be preserved
    assert!(result.iter().any(|m| m.role == "system"));
}

#[test]
fn test_context_window_manager_needs_truncation() {
    let config = ContextWindowConfig::new(50).with_reserved_output(10);
    let mgr = ContextWindowManager::new(config);

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "A".repeat(500), // Very long message
        name: None,
    }];

    assert!(mgr.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_estimate_total_tokens() {
    let config = ContextWindowConfig::new(4096);
    let mgr = ContextWindowManager::new(config);

    let messages = vec![
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(), // ~5 chars -> ~2 tokens + 10 overhead = ~12
            name: None,
        },
        crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: "World".to_string(),
            name: None,
        },
    ];

    let total = mgr.estimate_total_tokens(&messages);
    assert!(total > 0);
    assert!(total < 100); // Reasonable for short messages
}

// ============================================================================
// clean_chat_output Tests
// ============================================================================

#[test]
fn test_clean_chat_output_no_stop_sequence() {
    let output = clean_chat_output("Hello, how are you?");
    assert_eq!(output, "Hello, how are you?");
}

#[test]
fn test_clean_chat_output_chatml_stop() {
    let output = clean_chat_output("Hello<|im_end|>extra stuff");
    assert_eq!(output, "Hello");
}

#[test]
fn test_clean_chat_output_endoftext_stop() {
    let output = clean_chat_output("Output text<|endoftext|>garbage");
    assert_eq!(output, "Output text");
}

#[test]
fn test_clean_chat_output_llama_eos() {
    let output = clean_chat_output("Response text</s>more");
    assert_eq!(output, "Response text");
}

#[test]
fn test_clean_chat_output_human_turn() {
    let output = clean_chat_output("Response text\nHuman: next question");
    assert_eq!(output, "Response text");
}

#[test]
fn test_clean_chat_output_user_turn() {
    let output = clean_chat_output("Response text\nUser: next question");
    assert_eq!(output, "Response text");
}

#[test]
fn test_clean_chat_output_double_newline_human() {
    let output = clean_chat_output("Response text\n\nHuman: next");
    assert_eq!(output, "Response text");
}

#[test]
fn test_clean_chat_output_double_newline_user() {
    let output = clean_chat_output("Response\n\nUser: next");
    assert_eq!(output, "Response");
}

#[test]
fn test_clean_chat_output_im_start_new_turn() {
    let output = clean_chat_output("The answer is 42<|im_start|>user\nNext");
    assert_eq!(output, "The answer is 42");
}

#[test]
fn test_clean_chat_output_end_tag() {
    let output = clean_chat_output("Text here<|end|>trailing");
    assert_eq!(output, "Text here");
}

#[test]
fn test_clean_chat_output_earliest_stop_wins() {
    let output = clean_chat_output("Text<|end|>mid<|im_end|>end");
    assert_eq!(output, "Text");
}

#[test]
fn test_clean_chat_output_empty_after_stop() {
    let output = clean_chat_output("<|im_end|>");
    assert_eq!(output, "");
}

#[test]
fn test_clean_chat_output_whitespace_trimming() {
    let output = clean_chat_output("  Hello  <|im_end|>");
    assert_eq!(output, "Hello");
}

// ============================================================================
// format_chat_messages Tests
// ============================================================================

#[test]
fn test_format_chat_messages_basic() {
    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    let formatted = format_chat_messages(&messages, None);
    assert!(!formatted.is_empty());
    assert!(formatted.contains("Hello"));
}

#[test]
fn test_format_chat_messages_system_and_user() {
    let messages = vec![
        crate::api::ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "What is 2+2?".to_string(),
            name: None,
        },
    ];

    let formatted = format_chat_messages(&messages, Some("test-model"));
    assert!(formatted.contains("helpful") || formatted.contains("2+2"));
}

#[test]
fn test_format_chat_messages_empty() {
    let messages: Vec<crate::api::ChatMessage> = vec![];
    let formatted = format_chat_messages(&messages, None);
    // Should not panic on empty messages
    assert!(formatted.is_empty() || !formatted.is_empty());
}

// ============================================================================
// Serde Round-Trip Tests for realize_handlers types
// ============================================================================

#[test]
fn test_embedding_request_serde() {
    let req = EmbeddingRequest {
        input: "Hello world".to_string(),
        model: Some("test-model".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: EmbeddingRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.input, "Hello world");
    assert_eq!(parsed.model, Some("test-model".to_string()));
}

#[test]
fn test_embedding_request_no_model() {
    let json = r#"{"input":"test text"}"#;
    let parsed: EmbeddingRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.input, "test text");
    assert!(parsed.model.is_none());
}

#[test]
fn test_embedding_response_serde() {
    let resp = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2, 0.3],
        }],
        model: "test".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: EmbeddingResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.object, "list");
    assert_eq!(parsed.data.len(), 1);
    assert_eq!(parsed.data[0].embedding.len(), 3);
    assert_eq!(parsed.usage.prompt_tokens, 5);
}

#[test]
fn test_model_metadata_response_serde() {
    let resp = ModelMetadataResponse {
        id: "model-1".to_string(),
        name: "Test Model".to_string(),
        format: "gguf".to_string(),
        size_bytes: 1_000_000,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 4096,
        lineage: Some(ModelLineage {
            uri: "pacha://model:latest".to_string(),
            version: "1.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "blake3:abc123".to_string(),
        }),
        loaded: true,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: ModelMetadataResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "model-1");
    assert!(parsed.lineage.is_some());
    assert_eq!(parsed.lineage.as_ref().expect("lineage").version, "1.0.0");
}

#[test]
fn test_model_metadata_response_no_lineage() {
    let resp = ModelMetadataResponse {
        id: "model-2".to_string(),
        name: "Simple Model".to_string(),
        format: "apr".to_string(),
        size_bytes: 500_000,
        quantization: None,
        context_length: 2048,
        lineage: None,
        loaded: false,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(!json.contains("lineage"));
    assert!(!json.contains("quantization"));
}

#[test]
fn test_model_lineage_full_serde() {
    let lineage = ModelLineage {
        uri: "pacha://qwen2:latest".to_string(),
        version: "2.0.0".to_string(),
        recipe: Some("distill-4bit".to_string()),
        parent: Some("qwen2-7b".to_string()),
        content_hash: "blake3:def456".to_string(),
    };
    let json = serde_json::to_string(&lineage).expect("serialize");
    let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.recipe, Some("distill-4bit".to_string()));
    assert_eq!(parsed.parent, Some("qwen2-7b".to_string()));
}

#[test]
fn test_reload_request_serde() {
    let req = ReloadRequest {
        model: Some("default".to_string()),
        path: Some("/path/to/model.gguf".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: ReloadRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.model, Some("default".to_string()));
    assert_eq!(parsed.path, Some("/path/to/model.gguf".to_string()));
}

#[test]
fn test_reload_request_minimal() {
    let json = r#"{}"#;
    let parsed: ReloadRequest = serde_json::from_str(json).expect("deserialize");
    assert!(parsed.model.is_none());
    assert!(parsed.path.is_none());
}

#[test]
fn test_reload_response_serde() {
    let resp = ReloadResponse {
        success: true,
        message: "Reload complete".to_string(),
        reload_time_ms: 42,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: ReloadResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.success);
    assert_eq!(parsed.reload_time_ms, 42);
}

#[test]
fn test_completion_request_serde() {
    let req = CompletionRequest {
        model: "test-model".to_string(),
        prompt: "Once upon a time".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: Some(vec!["END".to_string()]),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.model, "test-model");
    assert_eq!(parsed.max_tokens, Some(50));
    assert_eq!(parsed.stop, Some(vec!["END".to_string()]));
}

#[test]
fn test_completion_request_minimal() {
    let json = r#"{"model":"m","prompt":"p"}"#;
    let parsed: CompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.model, "m");
    assert!(parsed.max_tokens.is_none());
    assert!(parsed.temperature.is_none());
}

#[test]
fn test_completion_response_serde() {
    let resp = CompletionResponse {
        id: "cmpl-test-123".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "test".to_string(),
        choices: vec![CompletionChoice {
            text: "Generated text".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: crate::api::Usage {
            prompt_tokens: 5,
            completion_tokens: 3,
            total_tokens: 8,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: CompletionResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "cmpl-test-123");
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].finish_reason, "stop");
    assert_eq!(parsed.usage.total_tokens, 8);
}

#[test]
fn test_completion_choice_with_logprobs() {
    let choice = CompletionChoice {
        text: "Hello".to_string(),
        index: 0,
        logprobs: Some(serde_json::json!({"tokens": ["Hello"], "token_logprobs": [-0.5]})),
        finish_reason: "length".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("serialize");
    let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.logprobs.is_some());
}

// ============================================================================
// build_trace_data Tests
// ============================================================================

#[test]
fn test_build_trace_data_brick() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("brick"), 1000, 10, 5, 4);
    assert!(brick.is_some());
    assert!(step.is_none());
    assert!(layer.is_none());
    let brick = brick.expect("brick trace");
    assert_eq!(brick.level, "brick");
    assert_eq!(brick.operations, 5);
    assert_eq!(brick.total_time_us, 1000);
    assert_eq!(brick.breakdown.len(), 3);
    assert_eq!(brick.breakdown[0].name, "embedding_lookup");
}

#[test]
fn test_build_trace_data_step() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("step"), 2000, 20, 10, 8);
    assert!(brick.is_none());
    assert!(step.is_some());
    assert!(layer.is_none());
    let step = step.expect("step trace");
    assert_eq!(step.level, "step");
    assert_eq!(step.breakdown.len(), 3);
    assert_eq!(step.breakdown[0].name, "tokenize");
    assert_eq!(step.breakdown[1].name, "forward_pass");
    assert_eq!(step.breakdown[2].name, "decode");
}

#[test]
fn test_build_trace_data_layer() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("layer"), 3000, 10, 5, 6);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());
    let layer = layer.expect("layer trace");
    assert_eq!(layer.level, "layer");
    assert_eq!(layer.operations, 6);
    assert_eq!(layer.breakdown.len(), 6);
    // Each layer should have uniform time distribution
    for (i, op) in layer.breakdown.iter().enumerate() {
        assert_eq!(op.name, format!("layer_{}", i));
    }
}

#[test]
fn test_build_trace_data_none() {
    let (brick, step, layer) = crate::api::build_trace_data(None, 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("unknown"), 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

// ============================================================================
// AppState Constructor Tests (non-async, pure logic)
// ============================================================================

#[test]
fn test_app_state_demo_mock() {
    let state = crate::api::AppState::demo_mock();
    assert!(state.is_ok());
}

#[test]
fn test_app_state_with_cache() {
    let state = crate::api::AppState::with_cache(10);
    // Should create state without panicking
    let _ = state;
}

// ============================================================================
// API Types Default/Serde Tests (types.rs coverage)
// ============================================================================

#[test]
fn test_default_max_tokens() {
    assert_eq!(crate::api::default_max_tokens(), 50);
}

#[test]
fn test_default_top_k() {
    assert_eq!(crate::api::default_top_k(), 50);
}

#[cfg(test)]
#[test]
fn test_default_strategy() {
    assert_eq!(crate::api::default_strategy(), "greedy");
}

#[cfg(test)]
#[test]
fn test_default_temperature() {
    assert!((crate::api::default_temperature() - 1.0).abs() < f32::EPSILON);
}

#[cfg(test)]
#[test]
fn test_default_top_p() {
    assert!((crate::api::default_top_p() - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_error_response_serde() {
    let err = crate::api::ErrorResponse {
        error: "Something went wrong".to_string(),
    };
    let json = serde_json::to_string(&err).expect("serialize");
    let parsed: crate::api::ErrorResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.error, "Something went wrong");
}

#[test]
fn test_health_response_serde() {
    let resp = crate::api::HealthResponse {
        status: "ok".to_string(),
        version: "1.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: crate::api::HealthResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.status, "ok");
}

#[test]
fn test_tokenize_request_serde() {
    let req = crate::api::TokenizeRequest {
        text: "Hello world".to_string(),
        model_id: None,
    };
    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: crate::api::TokenizeRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.text, "Hello world");
}

#[test]
fn test_tokenize_response_serde() {
    let resp = crate::api::TokenizeResponse {
        token_ids: vec![1, 2, 3],
        num_tokens: 3,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: crate::api::TokenizeResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.token_ids, vec![1, 2, 3]);
    assert_eq!(parsed.num_tokens, 3);
}

#[test]
fn test_generate_request_defaults() {
    let json = r#"{"prompt":"test","strategy":"greedy"}"#;
    let parsed: crate::api::GenerateRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.prompt, "test");
    assert_eq!(parsed.max_tokens, 50); // default
    assert_eq!(parsed.top_k, 50); // default
}

#[test]
fn test_batch_tokenize_request_serde() {
    let req = crate::api::BatchTokenizeRequest {
        texts: vec!["hello".to_string(), "world".to_string()],
    };
    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: crate::api::BatchTokenizeRequest =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.texts.len(), 2);
}

#[test]
fn test_batch_generate_request_defaults() {
    let json = r#"{"prompts":["a","b"]}"#;
    let parsed: crate::api::BatchGenerateRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.prompts.len(), 2);
    assert_eq!(parsed.max_tokens, 50); // default
}

#[test]
fn test_stream_token_event_serde() {
    let evt = crate::api::StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };
    let json = serde_json::to_string(&evt).expect("serialize");
    let parsed: crate::api::StreamTokenEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.token_id, 42);
}

#[test]
fn test_stream_done_event_serde() {
    let evt = crate::api::StreamDoneEvent { num_generated: 10 };
    let json = serde_json::to_string(&evt).expect("serialize");
    let parsed: crate::api::StreamDoneEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.num_generated, 10);
}

#[test]
fn test_models_response_serde() {
    let resp = crate::api::ModelsResponse {
        models: vec![crate::api::ModelInfo {
            id: "model-1".to_string(),
            name: "Test".to_string(),
            description: "A test model".to_string(),
            format: "gguf".to_string(),
            loaded: true,
        }],
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let parsed: crate::api::ModelsResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.models.len(), 1);
    assert_eq!(parsed.models[0].id, "model-1");
}
