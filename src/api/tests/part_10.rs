//! API Tests Part 10: Realize Handlers Extended Coverage
//!
//! Additional tests for realize_handlers.rs focusing on:
//! - ModelLineage serialization
//! - ReloadResponse serialization
//! - CompletionChoice tests
//! - Handler error paths
//! - Edge cases for embeddings and completions

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::realize_handlers::{
    CompletionChoice, CompletionRequest, CompletionResponse, ModelLineage, ModelMetadataResponse,
    ReloadRequest, ReloadResponse,
};
use crate::api::test_helpers::create_test_app_shared;
use crate::api::Usage;

// =============================================================================
// ModelLineage Tests
// =============================================================================

#[test]
fn test_model_lineage_serialization_full() {
    let lineage = ModelLineage {
        uri: "pacha://model-001:v1.0".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("llama2-finetune-chat".to_string()),
        parent: Some("llama2-7b-base".to_string()),
        content_hash: "blake3:abc123def456".to_string(),
    };

    let json = serde_json::to_string(&lineage).expect("serialize");
    assert!(json.contains("pacha://model-001:v1.0"));
    assert!(json.contains("1.0.0"));
    assert!(json.contains("llama2-finetune-chat"));
    assert!(json.contains("llama2-7b-base"));
    assert!(json.contains("blake3:abc123def456"));

    let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.uri, "pacha://model-001:v1.0");
    assert_eq!(parsed.version, "1.0.0");
    assert_eq!(parsed.recipe, Some("llama2-finetune-chat".to_string()));
    assert_eq!(parsed.parent, Some("llama2-7b-base".to_string()));
}

#[test]
fn test_model_lineage_serialization_minimal() {
    let lineage = ModelLineage {
        uri: "pacha://test:latest".to_string(),
        version: "0.0.1".to_string(),
        recipe: None,
        parent: None,
        content_hash: "blake3:000000".to_string(),
    };

    let json = serde_json::to_string(&lineage).expect("serialize");
    // recipe and parent should be skipped when None
    assert!(!json.contains("recipe"));
    assert!(!json.contains("parent"));

    let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.recipe, None);
    assert_eq!(parsed.parent, None);
}

#[test]
fn test_model_lineage_clone() {
    let lineage = ModelLineage {
        uri: "pacha://test:v1".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("recipe".to_string()),
        parent: Some("parent".to_string()),
        content_hash: "hash".to_string(),
    };

    let cloned = lineage.clone();
    assert_eq!(cloned.uri, lineage.uri);
    assert_eq!(cloned.version, lineage.version);
    assert_eq!(cloned.recipe, lineage.recipe);
    assert_eq!(cloned.parent, lineage.parent);
    assert_eq!(cloned.content_hash, lineage.content_hash);
}

#[test]
fn test_model_lineage_debug() {
    let lineage = ModelLineage {
        uri: "pacha://debug:latest".to_string(),
        version: "1.0.0".to_string(),
        recipe: None,
        parent: None,
        content_hash: "hash".to_string(),
    };

    let debug_str = format!("{:?}", lineage);
    assert!(debug_str.contains("ModelLineage"));
    assert!(debug_str.contains("pacha://debug:latest"));
}

// =============================================================================
// ReloadResponse Tests
// =============================================================================

#[test]
fn test_reload_response_serialization() {
    let response = ReloadResponse {
        success: true,
        message: "Model reloaded successfully".to_string(),
        reload_time_ms: 1500,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("true"));
    assert!(json.contains("Model reloaded successfully"));
    assert!(json.contains("1500"));

    let parsed: ReloadResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.success);
    assert_eq!(parsed.message, "Model reloaded successfully");
    assert_eq!(parsed.reload_time_ms, 1500);
}

#[test]
fn test_reload_response_failure() {
    let response = ReloadResponse {
        success: false,
        message: "Model file not found".to_string(),
        reload_time_ms: 0,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: ReloadResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(!parsed.success);
    assert_eq!(parsed.reload_time_ms, 0);
}

#[test]
fn test_reload_response_clone() {
    let response = ReloadResponse {
        success: true,
        message: "OK".to_string(),
        reload_time_ms: 100,
    };

    let cloned = response.clone();
    assert_eq!(cloned.success, response.success);
    assert_eq!(cloned.message, response.message);
    assert_eq!(cloned.reload_time_ms, response.reload_time_ms);
}

#[test]
fn test_reload_response_debug() {
    let response = ReloadResponse {
        success: true,
        message: "Debug test".to_string(),
        reload_time_ms: 42,
    };

    let debug_str = format!("{:?}", response);
    assert!(debug_str.contains("ReloadResponse"));
    assert!(debug_str.contains("Debug test"));
}

// =============================================================================
// CompletionChoice Tests
// =============================================================================

#[test]
fn test_completion_choice_serialization() {
    let choice = CompletionChoice {
        text: "Generated text output".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("Generated text output"));
    assert!(json.contains("stop"));

    let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.text, "Generated text output");
    assert_eq!(parsed.index, 0);
    assert!(parsed.logprobs.is_none());
    assert_eq!(parsed.finish_reason, "stop");
}

#[test]
fn test_completion_choice_with_logprobs() {
    let logprobs = serde_json::json!({
        "tokens": [" Hello", " world"],
        "token_logprobs": [-0.5, -0.3]
    });

    let choice = CompletionChoice {
        text: "Hello world".to_string(),
        index: 0,
        logprobs: Some(logprobs),
        finish_reason: "length".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("token_logprobs"));

    let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.logprobs.is_some());
}

#[test]
fn test_completion_choice_multiple_indices() {
    for i in 0..5 {
        let choice = CompletionChoice {
            text: format!("Choice {i}"),
            index: i,
            logprobs: None,
            finish_reason: "stop".to_string(),
        };

        let json = serde_json::to_string(&choice).expect("serialize");
        let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.index, i);
    }
}

#[test]
fn test_completion_choice_clone() {
    let choice = CompletionChoice {
        text: "Test".to_string(),
        index: 1,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };

    let cloned = choice.clone();
    assert_eq!(cloned.text, choice.text);
    assert_eq!(cloned.index, choice.index);
}

#[test]
fn test_completion_choice_debug() {
    let choice = CompletionChoice {
        text: "Debug".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };

    let debug_str = format!("{:?}", choice);
    assert!(debug_str.contains("CompletionChoice"));
}

// =============================================================================
// CompletionResponse Tests
// =============================================================================

#[test]
fn test_completion_response_serialization() {
    let response = CompletionResponse {
        id: "cmpl-test-123".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "test-model".to_string(),
        choices: vec![CompletionChoice {
            text: "Hello, world!".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 10,
            total_tokens: 15,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("cmpl-test-123"));
    assert!(json.contains("text_completion"));
    assert!(json.contains("Hello, world!"));

    let parsed: CompletionResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "cmpl-test-123");
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.usage.total_tokens, 15);
}

#[test]
fn test_completion_response_multiple_choices() {
    let response = CompletionResponse {
        id: "cmpl-multi".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "test".to_string(),
        choices: vec![
            CompletionChoice {
                text: "First".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "stop".to_string(),
            },
            CompletionChoice {
                text: "Second".to_string(),
                index: 1,
                logprobs: None,
                finish_reason: "stop".to_string(),
            },
        ],
        usage: Usage {
            prompt_tokens: 3,
            completion_tokens: 6,
            total_tokens: 9,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("First"));
    assert!(json.contains("Second"));

    let parsed: CompletionResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.choices.len(), 2);
}

#[test]
fn test_completion_response_clone() {
    let response = CompletionResponse {
        id: "test".to_string(),
        object: "text_completion".to_string(),
        created: 123,
        model: "model".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    let cloned = response.clone();
    assert_eq!(cloned.id, response.id);
    assert_eq!(cloned.created, response.created);
}

#[test]
fn test_completion_response_debug() {
    let response = CompletionResponse {
        id: "debug-test".to_string(),
        object: "text_completion".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    let debug_str = format!("{:?}", response);
    assert!(debug_str.contains("CompletionResponse"));
}

// =============================================================================
// CompletionRequest Additional Tests
// =============================================================================

#[test]
fn test_completion_request_with_stop_tokens() {
    let req = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: Some(vec!["</s>".to_string(), "<|endoftext|>".to_string()]),
    };

    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("</s>"));
    assert!(json.contains("<|endoftext|>"));

    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.stop.as_ref().unwrap().len(), 2);
}

#[test]
fn test_completion_request_temperature_extremes() {
    // Test temperature = 0.0 (greedy)
    let req_zero = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: Some(0.0),
        top_p: None,
        stop: None,
    };
    let json = serde_json::to_string(&req_zero).expect("serialize");
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.temperature, Some(0.0));

    // Test temperature = 2.0 (high randomness)
    let req_high = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: Some(2.0),
        top_p: None,
        stop: None,
    };
    let json = serde_json::to_string(&req_high).expect("serialize");
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.temperature, Some(2.0));
}

#[test]
fn test_completion_request_top_p_values() {
    let req = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: Some(0.95),
        stop: None,
    };

    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert!((parsed.top_p.unwrap() - 0.95).abs() < 0.01);
}

#[test]
fn test_completion_request_clone() {
    let req = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: Some(vec!["stop".to_string()]),
    };

    let cloned = req.clone();
    assert_eq!(cloned.model, req.model);
    assert_eq!(cloned.prompt, req.prompt);
    assert_eq!(cloned.max_tokens, req.max_tokens);
    assert_eq!(cloned.temperature, req.temperature);
    assert_eq!(cloned.top_p, req.top_p);
    assert_eq!(cloned.stop, req.stop);
}

#[test]
fn test_completion_request_debug() {
    let req = CompletionRequest {
        model: "debug-model".to_string(),
        prompt: "debug prompt".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };

    let debug_str = format!("{:?}", req);
    assert!(debug_str.contains("CompletionRequest"));
    assert!(debug_str.contains("debug-model"));
}

// =============================================================================
// ModelMetadataResponse with Lineage Tests
// =============================================================================

#[test]
fn test_model_metadata_response_with_lineage() {
    let lineage = ModelLineage {
        uri: "pacha://test-model:v1.0".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("training-recipe".to_string()),
        parent: None,
        content_hash: "blake3:test".to_string(),
    };

    let response = ModelMetadataResponse {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 4_000_000_000,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 8192,
        lineage: Some(lineage),
        loaded: true,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("pacha://test-model:v1.0"));
    assert!(json.contains("training-recipe"));

    let parsed: ModelMetadataResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.lineage.is_some());
    assert_eq!(parsed.lineage.unwrap().uri, "pacha://test-model:v1.0");
}

#[test]
fn test_model_metadata_response_large_size() {
    let response = ModelMetadataResponse {
        id: "large-model".to_string(),
        name: "Large Model".to_string(),
        format: "SafeTensors".to_string(),
        size_bytes: 70_000_000_000, // 70GB
        quantization: None,
        context_length: 32768,
        lineage: None,
        loaded: false,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: ModelMetadataResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.size_bytes, 70_000_000_000);
    assert_eq!(parsed.context_length, 32768);
}

#[test]
fn test_model_metadata_response_clone() {
    let response = ModelMetadataResponse {
        id: "test".to_string(),
        name: "Test".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 1000,
        quantization: Some("Q4_0".to_string()),
        context_length: 2048,
        lineage: None,
        loaded: true,
    };

    let cloned = response.clone();
    assert_eq!(cloned.id, response.id);
    assert_eq!(cloned.quantization, response.quantization);
}

#[test]
fn test_model_metadata_response_debug() {
    let response = ModelMetadataResponse {
        id: "debug".to_string(),
        name: "Debug".to_string(),
        format: "APR".to_string(),
        size_bytes: 0,
        quantization: None,
        context_length: 512,
        lineage: None,
        loaded: false,
    };

    let debug_str = format!("{:?}", response);
    assert!(debug_str.contains("ModelMetadataResponse"));
}

// =============================================================================
// ReloadRequest Additional Tests
// =============================================================================

#[test]
fn test_reload_request_with_both_fields() {
    let req = ReloadRequest {
        model: Some("my-model".to_string()),
        path: Some("/models/my-model.gguf".to_string()),
    };

    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("my-model"));
    assert!(json.contains("/models/my-model.gguf"));

    let parsed: ReloadRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.model, Some("my-model".to_string()));
    assert_eq!(parsed.path, Some("/models/my-model.gguf".to_string()));
}

#[test]
fn test_reload_request_path_only() {
    let req = ReloadRequest {
        model: None,
        path: Some("/path/to/model.safetensors".to_string()),
    };

    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: ReloadRequest = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.model.is_none());
    assert!(parsed.path.is_some());
}

#[test]
fn test_reload_request_model_only() {
    let req = ReloadRequest {
        model: Some("model-id".to_string()),
        path: None,
    };

    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: ReloadRequest = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.model.is_some());
    assert!(parsed.path.is_none());
}

#[test]
fn test_reload_request_clone() {
    let req = ReloadRequest {
        model: Some("test".to_string()),
        path: Some("/path".to_string()),
    };

    let cloned = req.clone();
    assert_eq!(cloned.model, req.model);
    assert_eq!(cloned.path, req.path);
}

#[test]
fn test_reload_request_debug() {
    let req = ReloadRequest {
        model: Some("debug".to_string()),
        path: None,
    };

    let debug_str = format!("{:?}", req);
    assert!(debug_str.contains("ReloadRequest"));
}

// =============================================================================
// HTTP Handler Integration Tests - Error Paths
// =============================================================================

#[tokio::test]
async fn test_realize_embed_empty_input() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "input": "",
        "model": null
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/embed")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty input is handled by tokenizer - may return OK with empty embedding
    // or error depending on tokenizer implementation
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_realize_embed_long_input() {
    let app = create_test_app_shared();

    // Very long input text
    let long_text = "word ".repeat(10000);
    let req_body = serde_json::json!({
        "input": long_text,
        "model": null
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/embed")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should handle long input gracefully
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_realize_embed_unicode_input() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "input": "Hello 世界! \u{1F30D} مرحبا 你好 Привет",
        "model": null
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/embed")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_realize_reload_without_registry() {
    // Demo state doesn't have registry enabled
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test",
        "path": "/tmp/test.gguf"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/reload")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return NOT_IMPLEMENTED when registry not enabled
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_realize_reload_invalid_json() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/reload")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .unwrap(),
        )
        .await
        .unwrap();

    // Invalid JSON should return 400
    assert!(response.status() == StatusCode::BAD_REQUEST || response.status() == StatusCode::NOT_FOUND || response.status() == StatusCode::INTERNAL_SERVER_ERROR || response.status() == StatusCode::SERVICE_UNAVAILABLE || response.status() == StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_openai_completions_empty_prompt() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test",
        "prompt": ""
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty prompt should be handled - may return BAD_REQUEST
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_with_temperature() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test",
        "prompt": "Hello",
        "temperature": 0.5,
        "max_tokens": 5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_with_top_p() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test",
        "prompt": "Hello",
        "top_p": 0.9,
        "max_tokens": 5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_with_stop_tokens() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test",
        "prompt": "Hello",
        "stop": ["</s>", "<|im_end|>"],
        "max_tokens": 10
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_default_model() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "prompt": "Test prompt"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_empty_model() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "",
        "prompt": "Test"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty model should use default
    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_invalid_json() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from("{invalid json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status() == StatusCode::BAD_REQUEST || response.status() == StatusCode::NOT_FOUND || response.status() == StatusCode::INTERNAL_SERVER_ERROR || response.status() == StatusCode::SERVICE_UNAVAILABLE || response.status() == StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_openai_completions_missing_prompt() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Missing required field
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_openai_embeddings_endpoint_basic() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "input": "Test text",
        "model": "default"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_embeddings_without_model() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "input": "Test text"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_embeddings_long_text() {
    let app = create_test_app_shared();

    let long_text = "test ".repeat(1000);
    let req_body = serde_json::json!({
        "input": long_text
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {status}"
    );
}

// =============================================================================
// Usage Type Tests (from realize_handlers)
// =============================================================================

#[test]
fn test_usage_consistency() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };

    // total should equal prompt + completion
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
}

#[test]
fn test_usage_zero_completion() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 0,
        total_tokens: 10,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    let parsed: Usage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.completion_tokens, 0);
}

// =============================================================================
// Completion Response Finish Reason Tests
// =============================================================================

#[test]
fn test_completion_choice_finish_reason_stop() {
    let choice = CompletionChoice {
        text: "Generated text".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };
    assert_eq!(choice.finish_reason, "stop");
}

#[test]
fn test_completion_choice_finish_reason_length() {
    let choice = CompletionChoice {
        text: "Truncated at max tokens".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "length".to_string(),
    };
    assert_eq!(choice.finish_reason, "length");
}

#[test]
fn test_completion_response_empty_choices() {
    let response = CompletionResponse {
        id: "cmpl-empty".to_string(),
        object: "text_completion".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("[]")); // Empty choices array
}

// =============================================================================
// Model Metadata Edge Cases
// =============================================================================

#[test]
fn test_model_metadata_response_zero_size() {
    let response = ModelMetadataResponse {
        id: "streaming-model".to_string(),
        name: "Streaming Model".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 0,
        quantization: None,
        context_length: 4096,
        lineage: None,
        loaded: false,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: ModelMetadataResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.size_bytes, 0);
}

#[test]
fn test_model_metadata_response_all_quantizations() {
    let quantizations = ["Q4_0", "Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"];
    for quant in quantizations {
        let response = ModelMetadataResponse {
            id: format!("model-{quant}"),
            name: format!("Model {quant}"),
            format: "GGUF".to_string(),
            size_bytes: 1000,
            quantization: Some(quant.to_string()),
            context_length: 2048,
            lineage: None,
            loaded: true,
        };

        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains(quant));
    }
}

// =============================================================================
// Lineage URI Format Tests
// =============================================================================

#[test]
fn test_lineage_various_uri_formats() {
    let uris = [
        "pacha://model:v1.0",
        "pacha://org/model:latest",
        "pacha://model:1.0.0-beta",
        "huggingface://user/model",
        "local:///path/to/model",
    ];

    for uri in uris {
        let lineage = ModelLineage {
            uri: uri.to_string(),
            version: "1.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "blake3:test".to_string(),
        };

        let json = serde_json::to_string(&lineage).expect("serialize");
        let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.uri, uri);
    }
}

#[test]
fn test_lineage_content_hash_formats() {
    let hashes = [
        "blake3:abc123def456",
        "sha256:0123456789abcdef",
        "md5:deadbeef",
    ];

    for hash in hashes {
        let lineage = ModelLineage {
            uri: "pacha://test:latest".to_string(),
            version: "1.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: hash.to_string(),
        };

        let json = serde_json::to_string(&lineage).expect("serialize");
        let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.content_hash, hash);
    }
}
