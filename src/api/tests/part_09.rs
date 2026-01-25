//! API Tests Part 09: OpenAI Handlers Extended Coverage
//!
//! Additional tests for openai_handlers.rs to improve coverage beyond 27%.
//! Focus: Streaming endpoints, error paths, edge cases, empty prompts.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app;
use crate::api::{
    create_router, AppState, ChatChoice, ChatChunkChoice, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, ChatDelta, ChatMessage, ErrorResponse,
    OpenAIModel, OpenAIModelsResponse, TraceData, TraceOperation, Usage,
};

// =============================================================================
// Streaming Handler Tests (openai_chat_completions_stream_handler)
// =============================================================================

#[tokio::test]
async fn test_streaming_handler_basic() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Streaming returns 200 OK with SSE content type
    assert_eq!(response.status(), StatusCode::OK);

    // Check content type for SSE
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "Expected SSE content type, got: {}",
        content_type
    );
}

#[tokio::test]
async fn test_streaming_handler_with_system_message() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_streaming_handler_with_temperature() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Test"}],
        "stream": true,
        "temperature": 0.5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_streaming_handler_with_max_tokens() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Count to 10"}],
        "stream": true,
        "max_tokens": 3
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_streaming_handler_with_top_p() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "top_p": 0.95
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_streaming_handler_empty_model() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty model uses default
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_streaming_handler_multi_turn() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"}
        ],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// =============================================================================
// Empty/Bad Prompt Tests (400 errors)
// =============================================================================

#[tokio::test]
async fn test_chat_completions_empty_messages_array() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": []
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty messages should return 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let error: ErrorResponse = serde_json::from_slice(&body).unwrap();
    assert!(error.error.contains("empty") || error.error.contains("Messages"));
}

#[tokio::test]
async fn test_streaming_empty_messages_array() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty messages should return 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_missing_model_field() {
    let app = create_test_app();

    // Missing model field (required)
    let req_body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Missing required field returns 422
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_chat_completions_invalid_message_role() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "invalid_role", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Invalid role still parses, may or may not return OK depending on implementation
    // The key is that it doesn't crash
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_malformed_json() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{not valid json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_empty_json_body() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    // Missing required fields returns 422
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_chat_completions_null_messages() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": null
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // null messages should be rejected
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// =============================================================================
// Trace Header Variations
// =============================================================================

#[tokio::test]
async fn test_chat_completions_trace_header_case_insensitive() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    // Use uppercase in header value - should be normalized to lowercase
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("X-Trace-Level", "BRICK")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_chat_completions_trace_header_empty() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("X-Trace-Level", "")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty trace level should not cause error
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_chat_completions_trace_header_mixed_case() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Test"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("X-Trace-Level", "StEp")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// =============================================================================
// Model Not Found Scenarios
// =============================================================================

#[tokio::test]
async fn test_chat_completions_nonexistent_model() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "nonexistent-model-xyz-12345",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Demo mode falls back to default model, so this still works
    assert_eq!(response.status(), StatusCode::OK);
}

// =============================================================================
// OpenAI Models Endpoint Tests
// =============================================================================

#[tokio::test]
async fn test_models_endpoint_response_structure() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
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
    let result: OpenAIModelsResponse = serde_json::from_slice(&body).unwrap();

    // Verify OpenAI-compatible structure
    assert_eq!(result.object, "list");
    assert!(!result.data.is_empty());

    for model in &result.data {
        assert_eq!(model.object, "model");
        assert!(!model.id.is_empty());
        assert!(!model.owned_by.is_empty());
        assert!(model.created > 0);
    }
}

#[tokio::test]
async fn test_models_endpoint_post_not_allowed() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// =============================================================================
// Chat Completion Chunk Struct Tests
// =============================================================================

#[test]
fn test_chunk_initial_has_role() {
    let chunk = ChatCompletionChunk::initial("test-id", "test-model");
    assert!(chunk.choices[0].delta.role.is_some());
    assert_eq!(chunk.choices[0].delta.role.as_ref().unwrap(), "assistant");
}

#[test]
fn test_chunk_content_no_role() {
    let chunk = ChatCompletionChunk::content("test-id", "test-model", "hello");
    assert!(chunk.choices[0].delta.role.is_none());
    assert_eq!(chunk.choices[0].delta.content.as_ref().unwrap(), "hello");
}

#[test]
fn test_chunk_done_has_finish_reason() {
    let chunk = ChatCompletionChunk::done("test-id", "test-model");
    assert_eq!(chunk.choices[0].finish_reason.as_ref().unwrap(), "stop");
}

#[test]
fn test_chunk_serialization_preserves_structure() {
    let chunk = ChatCompletionChunk::content("test-id", "model", "Hi");
    let json = serde_json::to_string(&chunk).expect("serialize");
    let parsed: ChatCompletionChunk = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.id, "test-id");
    assert_eq!(parsed.model, "model");
    assert_eq!(parsed.choices[0].delta.content.as_ref().unwrap(), "Hi");
}

// =============================================================================
// Response Validation Tests
// =============================================================================

#[tokio::test]
async fn test_response_has_valid_id_format() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();

    // ID should start with chatcmpl-
    assert!(result.id.starts_with("chatcmpl-"));
}

#[tokio::test]
async fn test_response_object_type_correct() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(result.object, "chat.completion");
}

#[tokio::test]
async fn test_response_model_matches_request() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(result.model, "default");
}

#[tokio::test]
async fn test_response_has_exactly_one_choice() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(result.choices.len(), 1);
    assert_eq!(result.choices[0].index, 0);
}

#[tokio::test]
async fn test_response_choice_has_assistant_role() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(result.choices[0].message.role, "assistant");
}

// =============================================================================
// Usage Token Counting Tests
// =============================================================================

#[tokio::test]
async fn test_usage_prompt_tokens_positive() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello world"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();

    // Prompt should have at least 1 token
    assert!(result.usage.prompt_tokens > 0);
}

#[tokio::test]
async fn test_usage_total_equals_sum() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let result: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();

    // Total should equal prompt + completion
    assert_eq!(
        result.usage.total_tokens,
        result.usage.prompt_tokens + result.usage.completion_tokens
    );
}

// =============================================================================
// Content-Type Validation
// =============================================================================

#[tokio::test]
async fn test_chat_completions_wrong_content_type() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "text/plain")
                .body(Body::from("model=default&messages=hi"))
                .unwrap(),
        )
        .await
        .unwrap();

    // Wrong content type should be rejected
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}

#[tokio::test]
async fn test_chat_completions_no_content_type() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                // Note: no content-type header
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // axum may accept this or reject it depending on configuration
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}

// =============================================================================
// Long Content Tests
// =============================================================================

#[tokio::test]
async fn test_chat_completions_long_message() {
    let app = create_test_app();

    // Create a message with 1000 characters
    let long_content = "x".repeat(1000);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": long_content}],
        "max_tokens": 5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_chat_completions_many_messages() {
    let app = create_test_app();

    // Create 20 messages
    let mut messages = vec![];
    for i in 0..20 {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        messages.push(serde_json::json!({
            "role": role,
            "content": format!("Message {}", i)
        }));
    }

    let req_body = serde_json::json!({
        "model": "default",
        "messages": messages,
        "max_tokens": 5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// =============================================================================
// TraceData and TraceOperation Tests
// =============================================================================

#[test]
fn test_trace_data_empty_breakdown() {
    let trace = TraceData {
        level: "brick".to_string(),
        operations: 0,
        total_time_us: 0,
        breakdown: vec![],
    };

    let json = serde_json::to_string(&trace).expect("serialize");
    assert!(json.contains("[]")); // Empty breakdown array
}

#[test]
fn test_trace_operation_with_details() {
    let op = TraceOperation {
        name: "matmul".to_string(),
        time_us: 1000,
        details: Some("512x512".to_string()),
    };

    let json = serde_json::to_string(&op).expect("serialize");
    assert!(json.contains("matmul"));
    assert!(json.contains("512x512"));
}

#[test]
fn test_trace_operation_without_details() {
    let op = TraceOperation {
        name: "softmax".to_string(),
        time_us: 50,
        details: None,
    };

    let json = serde_json::to_string(&op).expect("serialize");
    assert!(json.contains("softmax"));
    // details field should be skipped
    assert!(!json.contains("details"));
}

// =============================================================================
// ErrorResponse Tests
// =============================================================================

#[test]
fn test_error_response_serialization() {
    let err = ErrorResponse {
        error: "Test error".to_string(),
    };

    let json = serde_json::to_string(&err).expect("serialize");
    assert!(json.contains("Test error"));
}

#[test]
fn test_error_response_deserialization() {
    let json = r#"{"error":"Something went wrong"}"#;
    let err: ErrorResponse = serde_json::from_str(json).expect("deserialize");
    assert_eq!(err.error, "Something went wrong");
}

// =============================================================================
// ChatMessage Edge Cases
// =============================================================================

#[test]
fn test_chat_message_very_long_content() {
    let long_content = "x".repeat(10000);
    let msg = ChatMessage {
        role: "user".to_string(),
        content: long_content.clone(),
        name: None,
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.content.len(), 10000);
}

#[test]
fn test_chat_message_with_newlines() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "line1\nline2\nline3".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.content.contains('\n'));
}

#[test]
fn test_chat_message_with_tabs() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "col1\tcol2\tcol3".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.content.contains('\t'));
}

// =============================================================================
// Usage Struct Edge Cases
// =============================================================================

#[test]
fn test_usage_max_values() {
    let usage = Usage {
        prompt_tokens: usize::MAX,
        completion_tokens: 0,
        total_tokens: usize::MAX,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    let parsed: Usage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.prompt_tokens, usize::MAX);
}

// =============================================================================
// OpenAIModel Tests
// =============================================================================

#[test]
fn test_openai_model_with_numeric_id() {
    let model = OpenAIModel {
        id: "12345".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "test".to_string(),
    };

    let json = serde_json::to_string(&model).expect("serialize");
    assert!(json.contains("12345"));
}

#[test]
fn test_openai_model_with_special_chars_in_id() {
    let model = OpenAIModel {
        id: "model-v1.0-beta".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "test-org".to_string(),
    };

    let json = serde_json::to_string(&model).expect("serialize");
    let parsed: OpenAIModel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "model-v1.0-beta");
}

// =============================================================================
// ChatChoice Edge Cases
// =============================================================================

#[test]
fn test_chat_choice_with_empty_content() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("\"content\":\"\""));
}

#[test]
fn test_chat_choice_with_large_index() {
    let choice = ChatChoice {
        index: 999,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "test".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("999"));
}

// =============================================================================
// ChatDelta Edge Cases
// =============================================================================

#[test]
fn test_chat_delta_empty() {
    let delta = ChatDelta {
        role: None,
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    // Should be empty object
    assert_eq!(json, "{}");
}

#[test]
fn test_chat_delta_both_present() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("Hello".to_string()),
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("assistant"));
    assert!(json.contains("Hello"));
}

// =============================================================================
// ChatChunkChoice Edge Cases
// =============================================================================

#[test]
fn test_chat_chunk_choice_with_length_finish_reason() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: None,
        },
        finish_reason: Some("length".to_string()),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("length"));
}

#[test]
fn test_chat_chunk_choice_streaming_content() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: Some("partial".to_string()),
        },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("partial"));
    // finish_reason should not be present
    assert!(!json.contains("finish_reason"));
}
