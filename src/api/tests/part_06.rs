//! API Tests Part 06: Error Response Coverage (PMAT-803)
//!
//! Tests for error paths in openai_handlers.rs to drive coverage from 27% to 70%+.
//! Focus: 400, 401, 404, 422, 500 error responses.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app;
use crate::api::{
    build_trace_data, create_router, format_chat_messages, AppState, ChatChoice, ChatChunkChoice,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatDelta, ChatMessage,
    ErrorResponse, GenerateRequest, GenerateResponse, HealthResponse, OpenAIModel,
    OpenAIModelsResponse, StreamDoneEvent, StreamTokenEvent, TokenizeResponse, TraceData,
    TraceOperation, Usage,
};

// ============================================================================
// ErrorResponse Coverage Tests
// ============================================================================

#[test]
fn test_error_response_json_structure() {
    let err = ErrorResponse {
        error: "Test error message".to_string(),
    };
    let json = serde_json::to_string(&err).expect("serialize");
    assert!(json.contains("error"));
    assert!(json.contains("Test error message"));

    // Verify deserialization
    let parsed: ErrorResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.error, "Test error message");
}

#[test]
fn test_error_response_with_special_chars() {
    let err = ErrorResponse {
        error: "Error: \"quoted\" and 'apostrophe' <tag>".to_string(),
    };
    let json = serde_json::to_string(&err).expect("serialize");
    // Verify JSON escaping
    assert!(json.contains("Error:"));

    // Round-trip
    let parsed: ErrorResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.error.contains("quoted"));
}

#[test]
fn test_error_response_unicode() {
    let err = ErrorResponse {
        error: "エラー: 无法处理请求 🚫".to_string(),
    };
    let json = serde_json::to_string(&err).expect("serialize");
    let parsed: ErrorResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.error.contains("エラー"));
}

// ============================================================================
// OpenAI Models Handler Tests
// ============================================================================

#[tokio::test]
async fn test_openai_models_handler_demo_mode() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let models: OpenAIModelsResponse = serde_json::from_slice(&body).expect("test");

    assert_eq!(models.object, "list");
    assert!(!models.data.is_empty());
    assert_eq!(models.data[0].object, "model");
    assert_eq!(models.data[0].owned_by, "realizar");
}

#[tokio::test]
async fn test_openai_models_response_format() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: "test-model".to_string(),
            object: "model".to_string(),
            created: 1234567890,
            owned_by: "test-owner".to_string(),
        }],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("test-model"));
    assert!(json.contains("1234567890"));
    assert!(json.contains("test-owner"));
}

// ============================================================================
// Chat Completion Request Validation Tests
// ============================================================================

#[test]
fn test_chat_completion_request_minimal() {
    let request = ChatCompletionRequest {
        model: "test".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };

    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("test"));
    assert!(json.contains("user"));
    assert!(json.contains("Hello"));
}

#[test]
fn test_chat_completion_request_full() {
    let request = ChatCompletionRequest {
        model: "qwen2.5-coder".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
                name: Some("Alice".to_string()),
            },
        ],
        max_tokens: Some(256),
        temperature: Some(0.7),
        top_p: Some(0.9),
        n: 1,
        stream: true,
        stop: Some(vec!["stop".to_string()]),
        user: Some("test-user".to_string()),
    };

    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("qwen2.5-coder"));
    assert!(json.contains("system"));
    assert!(json.contains("256"));
    assert!(json.contains("0.7"));
}

#[test]
fn test_chat_completion_request_deserialization() {
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}]
    }"#;

    let request: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(request.model, "test");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.max_tokens, None);
    assert!(!request.stream);
}

// ============================================================================
// Chat Message Tests
// ============================================================================

#[test]
fn test_chat_message_roles() {
    for role in &["user", "assistant", "system", "function"] {
        let msg = ChatMessage {
            role: role.to_string(),
            content: "test".to_string(),
            name: None,
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains(role));
    }
}

#[test]
fn test_chat_message_with_name() {
    let msg = ChatMessage {
        role: "function".to_string(),
        content: "result".to_string(),
        name: Some("get_weather".to_string()),
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    assert!(json.contains("get_weather"));
}

#[test]
fn test_chat_message_empty_content() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.content.is_empty());
}

// ============================================================================
// Chat Completion Response Tests
// ============================================================================

#[test]
fn test_chat_completion_response_structure() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "test-model".to_string(),
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
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("chatcmpl-123"));
    assert!(json.contains("chat.completion"));
    assert!(json.contains("Hello!"));
    assert!(json.contains("15")); // total_tokens
}

#[test]
fn test_chat_completion_response_with_traces() {
    let trace = TraceData {
        level: "brick".to_string(),
        operations: 5,
        total_time_us: 1000,
        breakdown: vec![TraceOperation {
            name: "embedding".to_string(),
            time_us: 100,
            details: Some("5 tokens".to_string()),
        }],
    };

    let response = ChatCompletionResponse {
        id: "test".to_string(),
        object: "chat.completion".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
        brick_trace: Some(trace.clone()),
        step_trace: Some(trace.clone()),
        layer_trace: Some(trace),
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("brick"));
    assert!(json.contains("embedding"));
    assert!(json.contains("1000"));
}

// ============================================================================
// Chat Completion Chunk Tests (Streaming)
// ============================================================================

#[test]
fn test_chat_completion_chunk_initial() {
    let chunk = ChatCompletionChunk::initial("chatcmpl-123", "test-model");

    assert_eq!(chunk.id, "chatcmpl-123");
    assert_eq!(chunk.object, "chat.completion.chunk");
    assert_eq!(chunk.model, "test-model");
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.role, Some("assistant".to_string()));
    assert_eq!(chunk.choices[0].delta.content, None);
}

#[test]
fn test_chat_completion_chunk_content() {
    let chunk = ChatCompletionChunk::content("chatcmpl-123", "test-model", "Hello");

    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.role, None);
    assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    assert_eq!(chunk.choices[0].finish_reason, None);
}

#[test]
fn test_chat_completion_chunk_done() {
    let chunk = ChatCompletionChunk::done("chatcmpl-123", "test-model");

    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.role, None);
    assert_eq!(chunk.choices[0].delta.content, None);
    assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
}

#[test]
fn test_chat_completion_chunk_serialization() {
    let chunk = ChatCompletionChunk::content("id", "model", "test");
    let json = serde_json::to_string(&chunk).expect("serialize");

    // Verify OpenAI-compatible format
    assert!(json.contains("chat.completion.chunk"));
    assert!(json.contains("delta"));
    assert!(json.contains("test"));
}

// ============================================================================
// Usage Tests
// ============================================================================

#[test]
fn test_usage_structure() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    assert!(json.contains("100"));
    assert!(json.contains("50"));
    assert!(json.contains("150"));
}

#[test]
fn test_usage_zero_tokens() {
    let usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    let parsed: Usage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.total_tokens, 0);
}

// ============================================================================
// ChatChoice Tests
// ============================================================================

#[test]
fn test_chat_choice_structure() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Test response".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("index"));
    assert!(json.contains("message"));
    assert!(json.contains("finish_reason"));
}

#[test]
fn test_chat_choice_length_finish_reason() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "...".to_string(),
            name: None,
        },
        finish_reason: "length".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("length"));
}

// ============================================================================
// format_chat_messages Tests
// ============================================================================

#[test]
fn test_format_chat_messages_chatml() {
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hi!".to_string(),
            name: None,
        },
    ];

    let formatted = format_chat_messages(&messages, Some("qwen"));
    assert!(formatted.contains("<|im_start|>"));
    assert!(formatted.contains("system"));
    assert!(formatted.contains("You are helpful."));
    assert!(formatted.contains("user"));
    assert!(formatted.contains("Hi!"));
}

#[test]
fn test_format_chat_messages_alpaca() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    let formatted = format_chat_messages(&messages, Some("alpaca"));
    assert!(formatted.contains("### Instruction:"));
}

#[test]
fn test_format_chat_messages_empty() {
    let messages: Vec<ChatMessage> = vec![];
    let formatted = format_chat_messages(&messages, None);
    assert!(formatted.is_empty() || formatted.contains("assistant"));
}

// ============================================================================
// build_trace_data Tests
// ============================================================================

#[test]
fn test_build_trace_data_none() {
    let (brick, step, layer) = build_trace_data(None, 1000, 10, 5, 12);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_brick() {
    let (brick, step, layer) = build_trace_data(Some("brick"), 5000, 20, 10, 28);
    assert!(brick.is_some());
    let trace = brick.unwrap();
    assert_eq!(trace.level, "brick");
    assert!(trace.total_time_us > 0);
    assert!(!trace.breakdown.is_empty());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_step() {
    let (brick, step, layer) = build_trace_data(Some("step"), 5000, 20, 10, 28);
    // "step" level only returns step trace
    assert!(brick.is_none());
    assert!(step.is_some());
    let step_trace = step.unwrap();
    assert_eq!(step_trace.level, "step");
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_layer() {
    let (brick, step, layer) = build_trace_data(Some("layer"), 5000, 20, 10, 28);
    // "layer" level only returns layer trace
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());
    let layer_trace = layer.unwrap();
    assert_eq!(layer_trace.level, "layer");
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = build_trace_data(Some("unknown"), 5000, 20, 10, 28);
    // Unknown levels should return None for all
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

// ============================================================================
// OpenAIModel Tests
// ============================================================================

#[test]
fn test_openai_model_structure() {
    let model = OpenAIModel {
        id: "qwen2.5-coder-1.5b".to_string(),
        object: "model".to_string(),
        created: 1700000000,
        owned_by: "realizar".to_string(),
    };

    let json = serde_json::to_string(&model).expect("serialize");
    assert!(json.contains("qwen2.5-coder-1.5b"));
    assert!(json.contains("1700000000"));
}

// ============================================================================
// Invalid Request Tests (Bad Request 400)
// ============================================================================

#[tokio::test]
async fn test_generate_invalid_json() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .expect("test"),
        )
        .await
        .expect("test");

    // Should get a 400 or 422 for invalid JSON
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_generate_missing_required_field() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"temperature": 0.5}"#))
                .expect("test"),
        )
        .await
        .expect("test");

    // Missing required 'prompt' field
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// ============================================================================
// Method Not Allowed Tests (405)
// ============================================================================

#[tokio::test]
async fn test_generate_get_method_not_allowed() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/generate")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    // POST-only endpoint should reject GET
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_chat_completions_get_method_not_allowed() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/chat/completions")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ============================================================================
// Not Found Tests (404)
// ============================================================================

#[tokio::test]
async fn test_unknown_endpoint_404() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/unknown/endpoint")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_v1_unknown_404() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/unknown")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ============================================================================
// Content-Type Tests
// ============================================================================

#[tokio::test]
async fn test_generate_wrong_content_type() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "text/plain")
                .body(Body::from("prompt=test"))
                .expect("test"),
        )
        .await
        .expect("test");

    // Should get error for wrong content type
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// ============================================================================
// Deep Coverage: ChatDelta and ChatChunkChoice
// ============================================================================

#[test]
fn test_chunk_delta_role_only() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("assistant"));
}

#[test]
fn test_chunk_delta_content_only() {
    let delta = ChatDelta {
        role: None,
        content: Some("test content".to_string()),
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("test content"));
}

#[test]
fn test_chunk_choice_structure() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: Some("hi".to_string()),
        },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("delta"));
    assert!(json.contains("hi"));
}
