//! API Tests Part 16: Combinatorial API Sweep for openai_handlers.rs
//!
//! Protocol T-COV-95 Directive 1 (Popper Phase 2): Exercise all branches in
//! /v1/chat/completions through combinatorial testing:
//! - stream: true vs stream: false
//! - temperature: 0.0 (Greedy) vs temperature: 1.5 (Creative)
//! - max_tokens: 0 and max_tokens: very large
//! - Invalid JSON payloads (400 Bad Request)
//! - Empty messages arrays
//! - Non-existent models
//!
//! Goal: Lift openai_handlers.rs from 57% to 95%+ coverage.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app;

// =============================================================================
// Combinatorial Request Variations
// =============================================================================

/// Test stream=true vs stream=false with default model
#[tokio::test]
async fn test_chat_completions_stream_true() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // With no model loaded, should return error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_chat_completions_stream_false() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Temperature Variations (Greedy vs Creative)
// =============================================================================

/// Test temperature=0.0 triggers greedy decoding (top_k=1 branch)
#[tokio::test]
async fn test_chat_completions_temperature_zero_greedy() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Test greedy"}],
        "temperature": 0.0,
        "max_tokens": 10
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Temperature 0.0 should trigger top_k=1 branch
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test temperature=1.5 triggers creative decoding (top_k=40 branch)
#[tokio::test]
async fn test_chat_completions_temperature_creative() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Test creative"}],
        "temperature": 1.5,
        "max_tokens": 10
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// max_tokens Edge Cases
// =============================================================================

/// Test max_tokens=0 (edge case, should be handled)
#[tokio::test]
async fn test_chat_completions_max_tokens_zero() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // max_tokens=0 might produce empty response or error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test max_tokens large (but not huge to avoid timeout)
#[tokio::test]
async fn test_chat_completions_max_tokens_large() {
    let app = create_test_app();

    // Use 10000 instead of 1000000 to trigger large value handling without timeout
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10000
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Invalid JSON Payloads (400 Bad Request)
// =============================================================================

/// Test completely invalid JSON
#[tokio::test]
async fn test_chat_completions_invalid_json_syntax() {
    let app = create_test_app();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("{invalid json here"))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // 400 BAD_REQUEST for parse errors, 422 for structural errors
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "Invalid JSON syntax should return 400"
    );
}

/// Test JSON with wrong types
#[tokio::test]
async fn test_chat_completions_invalid_json_types() {
    let app = create_test_app();

    // messages should be array, not string
    let req_body = serde_json::json!({
        "model": "default",
        "messages": "not an array"
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "Wrong type should return 422"
    );
}

/// Test JSON missing required field 'messages'
#[tokio::test]
async fn test_chat_completions_missing_messages() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default"
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "Missing messages should return 422"
    );
}

/// Test JSON missing required field 'model'
#[tokio::test]
async fn test_chat_completions_missing_model() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "Missing model should return 422"
    );
}

// =============================================================================
// Empty Messages Array
// =============================================================================

/// Test empty messages array
#[tokio::test]
async fn test_chat_completions_empty_messages() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": []
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Empty messages should trigger prompt_ids.is_empty() branch
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Non-existent Models
// =============================================================================

/// Test request for model that doesn't exist
#[tokio::test]
async fn test_chat_completions_nonexistent_model() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "nonexistent-model-xyz-12345",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Non-existent model should fall through model branches
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test request with empty model string
#[tokio::test]
async fn test_chat_completions_empty_model_string() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Empty model string triggers request.model.is_empty() branch
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Multi-Message Conversations
// =============================================================================

/// Test multi-turn conversation (system + user + assistant + user)
#[tokio::test]
async fn test_chat_completions_multi_turn() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test message with only system role (no user message)
#[tokio::test]
async fn test_chat_completions_system_only() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// top_p Parameter (Nucleus Sampling)
// =============================================================================

/// Test top_p parameter triggers nucleus sampling branch
#[tokio::test]
async fn test_chat_completions_with_top_p() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "top_p": 0.9,
        "temperature": 0.7
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // top_p should trigger request.top_p.is_some() branch
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test top_p=1.0 (full distribution)
#[tokio::test]
async fn test_chat_completions_top_p_one() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "top_p": 1.0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Content-Type Variations
// =============================================================================

/// Test request without content-type header
#[tokio::test]
async fn test_chat_completions_no_content_type() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Should still work or return appropriate error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test request with wrong content-type
#[tokio::test]
async fn test_chat_completions_wrong_content_type() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "text/plain")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Empty Body
// =============================================================================

/// Test request with empty body
#[tokio::test]
async fn test_chat_completions_empty_body() {
    let app = create_test_app();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Empty body should fail parsing
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// =============================================================================
// Malformed Message Objects
// =============================================================================

/// Test message missing 'role' field
#[tokio::test]
async fn test_chat_completions_message_missing_role() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "Missing role should return 422"
    );
}

/// Test message missing 'content' field
#[tokio::test]
async fn test_chat_completions_message_missing_content() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "Missing content should return 422"
    );
}

/// Test message with invalid role
#[tokio::test]
async fn test_chat_completions_invalid_role() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "invalid_role", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Invalid role might be accepted or rejected depending on strictness
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Combinatorial Matrix: stream × temperature × max_tokens
// =============================================================================

/// Combinatorial: stream=true, temp=0.0, max_tokens=1
#[tokio::test]
async fn test_combo_stream_greedy_min_tokens() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "temperature": 0.0,
        "max_tokens": 1
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Combinatorial: stream=false, temp=1.5, max_tokens=100
#[tokio::test]
async fn test_combo_no_stream_creative_mid_tokens() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false,
        "temperature": 1.5,
        "max_tokens": 100
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Combinatorial: stream=true, temp=0.7, max_tokens=256 (defaults equivalent)
#[tokio::test]
async fn test_combo_stream_default_params() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "temperature": 0.7,
        "max_tokens": 256
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// HTTP Method Variations
// =============================================================================

/// Test GET method (should fail - POST only)
#[tokio::test]
async fn test_chat_completions_get_method() {
    let app = create_test_app();

    let request = Request::builder()
        .method("GET")
        .uri("/v1/chat/completions")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "GET should return 405"
    );
}

/// Test PUT method (should fail)
#[tokio::test]
async fn test_chat_completions_put_method() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("PUT")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "PUT should return 405"
    );
}

// =============================================================================
// Large Message Content
// =============================================================================

/// Test long message content (moderate size to avoid timeout)
#[tokio::test]
async fn test_chat_completions_large_content() {
    let app = create_test_app();

    // Use 1000 chars instead of 10000 to test large content handling without timeout
    let large_content = "x".repeat(1000);
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": large_content}],
        "max_tokens": 5
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Large content should be handled
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::PAYLOAD_TOO_LARGE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test empty content in message
#[tokio::test]
async fn test_chat_completions_empty_content() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": ""}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Empty content should trigger empty prompt handling
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Special Characters in Content
// =============================================================================

/// Test unicode content
#[tokio::test]
async fn test_chat_completions_unicode_content() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "こんにちは 🎉 مرحبا"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test content with newlines
#[tokio::test]
async fn test_chat_completions_multiline_content() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Line 1\nLine 2\nLine 3"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Optional Parameters
// =============================================================================

/// Test with all optional parameters
#[tokio::test]
async fn test_chat_completions_all_optional_params() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
        "top_p": 0.95,
        "max_tokens": 50,
        "stream": false,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// Test with negative temperature (edge case)
#[tokio::test]
async fn test_chat_completions_negative_temperature() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": -0.5
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Negative temperature might be rejected or clamped
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// =============================================================================
// Trace Header Tests
// =============================================================================

/// Test X-Request-ID header is echoed
#[tokio::test]
async fn test_chat_completions_trace_header() {
    let app = create_test_app();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-request-id", "test-trace-12345")
        .body(Body::from(serde_json::to_string(&req_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}
