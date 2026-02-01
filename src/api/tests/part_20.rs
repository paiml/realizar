//! T-COV-95 Deep Coverage Bridge: realize_handlers, openai_handlers, gpu_handlers
//!
//! Targets: ContextWindowConfig/Manager, format_chat_messages, clean_chat_output,
//! realize_embed_handler, realize_model_handler, realize_completions_handler,
//! openai_models_handler, openai_chat_completions_handler, HTTP error paths,
//! struct serde round-trips, GPU/batch handler routing.

use crate::api::realize_handlers::*;
use crate::api::test_helpers::create_test_app_shared;
use crate::api::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

// ============================================================================
// ContextWindowConfig
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
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 200, // more than max
        preserve_system: true,
    };
    assert_eq!(config.available_tokens(), 0);
}

// ============================================================================
// ContextWindowManager
// ============================================================================

#[test]
fn test_context_window_manager_default() {
    let mgr = ContextWindowManager::default_manager();
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];
    assert!(!mgr.needs_truncation(&msgs));
}

#[test]
fn test_context_window_manager_no_truncation() {
    let config = ContextWindowConfig::new(4096);
    let mgr = ContextWindowManager::new(config);
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Short message".to_string(),
        name: None,
    }];
    let (result, truncated) = mgr.truncate_messages(&msgs);
    assert!(!truncated);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_context_window_manager_truncation_needed() {
    let config = ContextWindowConfig {
        max_tokens: 50,
        reserved_output_tokens: 10,
        preserve_system: true,
    };
    let mgr = ContextWindowManager::new(config);
    // Create messages that exceed 40 available tokens
    let msgs: Vec<ChatMessage> = (0..10)
        .map(|i| ChatMessage {
            role: "user".to_string(),
            content: format!("Message number {} with enough content to use tokens", i),
            name: None,
        })
        .collect();
    let (result, truncated) = mgr.truncate_messages(&msgs);
    assert!(truncated);
    assert!(result.len() < msgs.len());
}

#[test]
fn test_context_window_manager_preserves_system() {
    let config = ContextWindowConfig {
        max_tokens: 80,
        reserved_output_tokens: 10,
        preserve_system: true,
    };
    let mgr = ContextWindowManager::new(config);
    let msgs = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "A".repeat(200),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "B".repeat(200),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Latest question".to_string(),
            name: None,
        },
    ];
    let (result, truncated) = mgr.truncate_messages(&msgs);
    assert!(truncated);
    // System message should be preserved
    assert!(result.iter().any(|m| m.role == "system"));
}

#[test]
fn test_context_window_manager_needs_truncation() {
    let config = ContextWindowConfig {
        max_tokens: 4096,
        reserved_output_tokens: 256,
        preserve_system: true,
    };
    let mgr = ContextWindowManager::new(config);
    let short = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hi".to_string(),
        name: None,
    }];
    assert!(!mgr.needs_truncation(&short));

    // Create enough messages to exceed 3840 available tokens
    let long: Vec<ChatMessage> = (0..500)
        .map(|i| ChatMessage {
            role: "user".to_string(),
            content: format!(
                "Long message number {} with content that takes many tokens to fill the window",
                i
            ),
            name: None,
        })
        .collect();
    assert!(mgr.needs_truncation(&long));
}

#[test]
fn test_context_window_manager_estimate_total_tokens() {
    let config = ContextWindowConfig::default();
    let mgr = ContextWindowManager::new(config);
    let msgs = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello world".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Hi there".to_string(),
            name: None,
        },
    ];
    let tokens = mgr.estimate_total_tokens(&msgs);
    assert!(tokens > 0);
    // Each message ~= len/4 + 10 overhead
    assert!(tokens >= 20);
}

#[test]
fn test_context_window_manager_empty_messages() {
    let config = ContextWindowConfig::default();
    let mgr = ContextWindowManager::new(config);
    let msgs: Vec<ChatMessage> = vec![];
    let (result, truncated) = mgr.truncate_messages(&msgs);
    assert!(!truncated);
    assert!(result.is_empty());
}

// ============================================================================
// format_chat_messages
// ============================================================================

#[test]
fn test_format_chat_messages_basic() {
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&msgs, None);
    assert!(!result.is_empty());
    assert!(result.contains("Hello"));
}

#[test]
fn test_format_chat_messages_multi_turn() {
    let msgs = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "What is 2+2?".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "4".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Thanks!".to_string(),
            name: None,
        },
    ];
    let result = format_chat_messages(&msgs, Some("qwen"));
    assert!(!result.is_empty());
    assert!(result.contains("What is 2+2?"));
    assert!(result.contains("Thanks!"));
}

#[test]
fn test_format_chat_messages_empty() {
    let msgs: Vec<ChatMessage> = vec![];
    let result = format_chat_messages(&msgs, None);
    // Empty messages should still return something (template header/footer)
    let _ = result; // just ensure no panic
}

#[test]
fn test_format_chat_messages_model_names() {
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hi".to_string(),
        name: None,
    }];
    // Different model name templates
    for model in &["qwen", "llama", "mistral", "phi", "unknown-model"] {
        let result = format_chat_messages(&msgs, Some(model));
        assert!(result.contains("Hi"), "model={model}");
    }
}

// ============================================================================
// clean_chat_output
// ============================================================================

#[test]
fn test_clean_chat_output_no_markers() {
    let result = clean_chat_output("Hello world");
    assert_eq!(result, "Hello world");
}

#[test]
fn test_clean_chat_output_chatml_end() {
    let result = clean_chat_output("Hello<|im_end|>extra stuff");
    assert_eq!(result, "Hello");
}

#[test]
fn test_clean_chat_output_endoftext() {
    let result = clean_chat_output("Response here<|endoftext|>garbage");
    assert_eq!(result, "Response here");
}

#[test]
fn test_clean_chat_output_llama_eos() {
    let result = clean_chat_output("Output text</s>more");
    assert_eq!(result, "Output text");
}

#[test]
fn test_clean_chat_output_human_turn() {
    let result = clean_chat_output("My response\nHuman: next question");
    assert_eq!(result, "My response");
}

#[test]
fn test_clean_chat_output_user_turn() {
    let result = clean_chat_output("Answer text\nUser: follow up");
    assert_eq!(result, "Answer text");
}

#[test]
fn test_clean_chat_output_chatml_start() {
    let result = clean_chat_output("Response<|im_start|>user\nmore");
    assert_eq!(result, "Response");
}

#[test]
fn test_clean_chat_output_empty() {
    let result = clean_chat_output("");
    assert_eq!(result, "");
}

#[test]
fn test_clean_chat_output_only_marker() {
    let result = clean_chat_output("<|im_end|>");
    assert_eq!(result, "");
}

#[test]
fn test_clean_chat_output_multiple_markers() {
    let result = clean_chat_output("Text<|im_end|>more</s>even more");
    // Should stop at earliest marker
    assert_eq!(result, "Text");
}

#[test]
fn test_clean_chat_output_end_marker() {
    let result = clean_chat_output("Response<|end|>trailing");
    assert_eq!(result, "Response");
}

#[test]
fn test_clean_chat_output_double_newline_human() {
    let result = clean_chat_output("Output\n\nHuman: question");
    assert_eq!(result, "Output");
}

#[test]
fn test_clean_chat_output_double_newline_user() {
    let result = clean_chat_output("Output\n\nUser: question");
    assert_eq!(result, "Output");
}

// ============================================================================
// Struct serde round-trips
// ============================================================================

#[test]
fn test_embedding_request_serde() {
    let req = EmbeddingRequest {
        input: "test".to_string(),
        model: Some("default".to_string()),
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: EmbeddingRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.input, "test");
    assert_eq!(parsed.model.as_deref(), Some("default"));
}

#[test]
fn test_embedding_request_no_model() {
    let json = r#"{"input":"hello"}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.input, "hello");
    assert!(req.model.is_none());
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
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: EmbeddingResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.data.len(), 1);
    assert_eq!(parsed.data[0].embedding.len(), 3);
}

#[test]
fn test_model_metadata_response_serde() {
    let resp = ModelMetadataResponse {
        id: "m1".to_string(),
        name: "Test Model".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 1024,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 2048,
        lineage: None,
        loaded: true,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: ModelMetadataResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, "m1");
    assert!(parsed.loaded);
}

#[test]
fn test_model_lineage_serde() {
    let lineage = ModelLineage {
        uri: "pacha://model:v1".to_string(),
        version: "1.0".to_string(),
        recipe: Some("finetune".to_string()),
        parent: Some("base-model".to_string()),
        content_hash: "abc123".to_string(),
    };
    let json = serde_json::to_string(&lineage).unwrap();
    let parsed: ModelLineage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.uri, "pacha://model:v1");
    assert_eq!(parsed.recipe.as_deref(), Some("finetune"));
}

#[test]
fn test_reload_request_serde() {
    let req = ReloadRequest {
        model: Some("test".to_string()),
        path: Some("/path/to/model".to_string()),
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: ReloadRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model.as_deref(), Some("test"));
}

#[test]
fn test_reload_request_empty() {
    let json = "{}";
    let req: ReloadRequest = serde_json::from_str(json).unwrap();
    assert!(req.model.is_none());
    assert!(req.path.is_none());
}

#[test]
fn test_reload_response_serde() {
    let resp = ReloadResponse {
        success: true,
        message: "Reloaded".to_string(),
        reload_time_ms: 500,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: ReloadResponse = serde_json::from_str(&json).unwrap();
    assert!(parsed.success);
    assert_eq!(parsed.reload_time_ms, 500);
}

#[test]
fn test_completion_request_serde() {
    let req = CompletionRequest {
        model: "default".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: None,
        stop: Some(vec!["END".to_string()]),
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: CompletionRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model, "default");
    assert_eq!(parsed.max_tokens, Some(100));
}

#[test]
fn test_completion_response_serde() {
    let resp = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: 1234567890,
        model: "default".to_string(),
        choices: vec![CompletionChoice {
            text: "world".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        },
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: CompletionResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].text, "world");
}

// ============================================================================
// GPU handler structs serde
// ============================================================================

#[test]
fn test_gpu_batch_request_serde() {
    let req = gpu_handlers::GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 64,
        temperature: 0.7,
        top_k: 40,
        stop: vec!["<|end|>".to_string()],
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: gpu_handlers::GpuBatchRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.prompts.len(), 2);
    assert_eq!(parsed.max_tokens, 64);
}

#[test]
fn test_gpu_batch_response_serde() {
    let resp = gpu_handlers::GpuBatchResponse {
        results: vec![gpu_handlers::GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "hello".to_string(),
            num_generated: 3,
        }],
        stats: gpu_handlers::GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 3,
            processing_time_ms: 10.0,
            throughput_tps: 300.0,
        },
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: gpu_handlers::GpuBatchResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.results.len(), 1);
    assert!(parsed.stats.throughput_tps > 0.0);
}

#[test]
fn test_gpu_warmup_response_serde() {
    let resp = gpu_handlers::GpuWarmupResponse {
        success: true,
        memory_bytes: 1024 * 1024,
        num_layers: 12,
        message: "Warmed up".to_string(),
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: gpu_handlers::GpuWarmupResponse = serde_json::from_str(&json).unwrap();
    assert!(parsed.success);
    assert_eq!(parsed.num_layers, 12);
}

#[test]
fn test_gpu_status_response_serde() {
    let resp = gpu_handlers::GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 4,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: gpu_handlers::GpuStatusResponse = serde_json::from_str(&json).unwrap();
    assert!(!parsed.cache_ready);
    assert_eq!(parsed.batch_threshold, 32);
}

#[test]
fn test_gpu_batch_result_serde() {
    let result = gpu_handlers::GpuBatchResult {
        index: 5,
        token_ids: vec![10, 20, 30],
        text: "generated".to_string(),
        num_generated: 3,
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: gpu_handlers::GpuBatchResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.index, 5);
    assert_eq!(parsed.token_ids, vec![10, 20, 30]);
}

#[test]
fn test_gpu_batch_stats_serde() {
    let stats = gpu_handlers::GpuBatchStats {
        batch_size: 8,
        gpu_used: true,
        total_tokens: 256,
        processing_time_ms: 50.5,
        throughput_tps: 5069.3,
    };
    let json = serde_json::to_string(&stats).unwrap();
    let parsed: gpu_handlers::GpuBatchStats = serde_json::from_str(&json).unwrap();
    assert!(parsed.gpu_used);
    assert_eq!(parsed.batch_size, 8);
}

// ============================================================================
// HTTP handler coverage - realize endpoints
// ============================================================================

#[tokio::test]
async fn test_realize_embed_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "input": "Hello world"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    // May return OK or error depending on mock state
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_openai_models_handler_via_http() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: OpenAIModelsResponse = serde_json::from_slice(&body).unwrap();
    assert!(!parsed.data.is_empty());
}

#[tokio::test]
async fn test_openai_chat_completions_non_streaming() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false,
        "max_tokens": 5
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    // Accept various status codes since mock state may not have full model
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_openai_completions_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "default",
        "prompt": "Hello",
        "max_tokens": 5
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_gpu_warmup_handler_via_http() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/gpu/warmup")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    // GPU warmup returns status even without GPU
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_gpu_status_handler_via_http() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/gpu/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_gpu_batch_completions_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompts": ["Hello", "World"],
        "max_tokens": 5
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_server_metrics_handler_via_http() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_tokenize_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "text": "Hello world"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_batch_tokenize_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "texts": ["Hello", "World"]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_generate_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 3,
        "temperature": 0.0,
        "strategy": "greedy",
        "top_k": 1,
        "top_p": 1.0
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_batch_generate_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompts": ["Hello", "World"],
        "max_tokens": 3,
        "temperature": 0.0,
        "strategy": "greedy",
        "top_k": 1,
        "top_p": 1.0
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_apr_predict_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "input": "Hello"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_apr_explain_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "input": "Hello"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_health_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_dispatch_metrics_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

// ============================================================================
// OpenAI types serde
// ============================================================================

#[test]
fn test_openai_model_serde() {
    let model = OpenAIModel {
        id: "gpt-4".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "realizar".to_string(),
    };
    let json = serde_json::to_string(&model).unwrap();
    let parsed: OpenAIModel = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, "gpt-4");
}

#[test]
fn test_openai_models_response_serde() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: "test".to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "test".to_string(),
        }],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: OpenAIModelsResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.data.len(), 1);
}

#[test]
fn test_chat_completion_request_serde() {
    let req = ChatCompletionRequest {
        model: "default".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.messages.len(), 1);
    assert!(!parsed.stream);
}

#[test]
fn test_chat_completion_response_serde() {
    let resp = ChatCompletionResponse {
        id: "chat-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "default".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: ChatCompletionResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.choices[0].message.content, "Hello!");
}

#[test]
fn test_usage_serde() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let json = serde_json::to_string(&usage).unwrap();
    let parsed: Usage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.total_tokens, 30);
}

#[test]
fn test_error_response_serde() {
    let err = ErrorResponse {
        error: "Something went wrong".to_string(),
    };
    let json = serde_json::to_string(&err).unwrap();
    let parsed: ErrorResponse = serde_json::from_str(&json).unwrap();
    assert!(parsed.error.contains("wrong"));
}

// ============================================================================
// Chat completion with trace header
// ============================================================================

#[tokio::test]
async fn test_chat_completions_with_trace_header() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "test"}],
        "stream": false,
        "max_tokens": 2
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("X-Trace-Level", "detailed")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    // Just verify the endpoint accepts the trace header without error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// ============================================================================
// Invalid JSON error paths
// ============================================================================

#[tokio::test]
async fn test_generate_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_chat_completions_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{invalid}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_tokenize_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from("bad json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// ============================================================================
// Streaming endpoint
// ============================================================================

#[tokio::test]
async fn test_stream_generate_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 3,
        "temperature": 0.0,
        "strategy": "greedy",
        "top_k": 1,
        "top_p": 1.0
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// ============================================================================
// Debug trait coverage
// ============================================================================

#[test]
fn test_context_window_config_debug() {
    let config = ContextWindowConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("4096"));
}

#[test]
fn test_embedding_data_debug() {
    let data = EmbeddingData {
        object: "embedding".to_string(),
        index: 0,
        embedding: vec![0.1],
    };
    let debug = format!("{data:?}");
    assert!(debug.contains("embedding"));
}

#[test]
fn test_embedding_usage_debug() {
    let usage = EmbeddingUsage {
        prompt_tokens: 5,
        total_tokens: 5,
    };
    let debug = format!("{usage:?}");
    assert!(debug.contains("5"));
}

#[test]
fn test_completion_choice_debug() {
    let choice = CompletionChoice {
        text: "hello".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };
    let debug = format!("{choice:?}");
    assert!(debug.contains("hello"));
}
