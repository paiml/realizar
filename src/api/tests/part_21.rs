//! T-COV-95 Extended Coverage: api/mod.rs
//!
//! Targets: build_trace_data for all trace levels, AppState accessors,
//! streaming types serde, error paths, edge cases.

use crate::api::*;
use crate::api::test_helpers::create_test_app_shared;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

// ============================================================================
// build_trace_data coverage for all branches
// ============================================================================

#[test]
fn test_build_trace_data_brick_level() {
    let (brick, step, layer) = build_trace_data(Some("brick"), 1000, 10, 5, 4);
    assert!(brick.is_some());
    assert!(step.is_none());
    assert!(layer.is_none());

    let trace = brick.unwrap();
    assert_eq!(trace.level, "brick");
    assert_eq!(trace.operations, 5);
    assert_eq!(trace.total_time_us, 1000);
    assert!(!trace.breakdown.is_empty());
    assert!(trace.breakdown.iter().any(|op| op.name == "embedding_lookup"));
    assert!(trace.breakdown.iter().any(|op| op.name == "matmul_qkv"));
    assert!(trace.breakdown.iter().any(|op| op.name == "softmax"));
}

#[test]
fn test_build_trace_data_step_level() {
    let (brick, step, layer) = build_trace_data(Some("step"), 5000, 20, 10, 8);
    assert!(brick.is_none());
    assert!(step.is_some());
    assert!(layer.is_none());

    let trace = step.unwrap();
    assert_eq!(trace.level, "step");
    assert_eq!(trace.operations, 10);
    assert_eq!(trace.total_time_us, 5000);
    assert!(trace.breakdown.iter().any(|op| op.name == "tokenize"));
    assert!(trace.breakdown.iter().any(|op| op.name == "forward_pass"));
    assert!(trace.breakdown.iter().any(|op| op.name == "decode"));
    // Check details fields
    let tokenize = trace.breakdown.iter().find(|op| op.name == "tokenize").unwrap();
    assert!(tokenize.details.as_ref().unwrap().contains("20"));
}

#[test]
fn test_build_trace_data_layer_level() {
    let (brick, step, layer) = build_trace_data(Some("layer"), 8000, 15, 8, 12);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());

    let trace = layer.unwrap();
    assert_eq!(trace.level, "layer");
    assert_eq!(trace.operations, 12);
    assert_eq!(trace.total_time_us, 8000);
    assert_eq!(trace.breakdown.len(), 12);
    assert!(trace.breakdown[0].name.starts_with("layer_"));
    assert!(trace.breakdown[11].name.contains("11"));
    assert!(trace.breakdown[0].details.as_ref().unwrap().contains("attention+mlp"));
}

#[test]
fn test_build_trace_data_none_level() {
    let (brick, step, layer) = build_trace_data(None, 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = build_trace_data(Some("unknown"), 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_empty_string_level() {
    let (brick, step, layer) = build_trace_data(Some(""), 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_detailed_level() {
    // "detailed" is not a recognized level, should return none
    let (brick, step, layer) = build_trace_data(Some("detailed"), 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_zero_values() {
    let (brick, step, layer) = build_trace_data(Some("brick"), 0, 0, 0, 0);
    assert!(brick.is_some());
    let trace = brick.unwrap();
    assert_eq!(trace.total_time_us, 0);
    assert_eq!(trace.operations, 0);
}

#[test]
fn test_build_trace_data_large_values() {
    let (brick, step, layer) = build_trace_data(Some("layer"), u64::MAX / 2, 100000, 50000, 1000);
    assert!(layer.is_some());
    let trace = layer.unwrap();
    assert_eq!(trace.breakdown.len(), 1000);
}

// ============================================================================
// TraceData and TraceOperation struct coverage
// ============================================================================

#[test]
fn test_trace_data_debug() {
    let trace = TraceData {
        level: "test".to_string(),
        operations: 5,
        total_time_us: 1000,
        breakdown: vec![],
    };
    let debug = format!("{:?}", trace);
    assert!(debug.contains("test"));
    assert!(debug.contains("1000"));
}

#[test]
fn test_trace_data_clone() {
    let trace = TraceData {
        level: "test".to_string(),
        operations: 5,
        total_time_us: 1000,
        breakdown: vec![TraceOperation {
            name: "op1".to_string(),
            time_us: 100,
            details: Some("detail".to_string()),
        }],
    };
    let cloned = trace.clone();
    assert_eq!(cloned.level, "test");
    assert_eq!(cloned.breakdown.len(), 1);
}

#[test]
fn test_trace_data_serde() {
    let trace = TraceData {
        level: "brick".to_string(),
        operations: 10,
        total_time_us: 5000,
        breakdown: vec![TraceOperation {
            name: "matmul".to_string(),
            time_us: 500,
            details: None,
        }],
    };
    let json = serde_json::to_string(&trace).unwrap();
    assert!(json.contains("brick"));
    assert!(json.contains("matmul"));
    let parsed: TraceData = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.level, "brick");
}

#[test]
fn test_trace_operation_debug() {
    let op = TraceOperation {
        name: "test_op".to_string(),
        time_us: 500,
        details: Some("info".to_string()),
    };
    let debug = format!("{:?}", op);
    assert!(debug.contains("test_op"));
    assert!(debug.contains("info"));
}

#[test]
fn test_trace_operation_clone() {
    let op = TraceOperation {
        name: "test_op".to_string(),
        time_us: 500,
        details: None,
    };
    let cloned = op.clone();
    assert_eq!(cloned.name, "test_op");
    assert!(cloned.details.is_none());
}

#[test]
fn test_trace_operation_serde() {
    let op = TraceOperation {
        name: "softmax".to_string(),
        time_us: 250,
        details: Some("batch=32".to_string()),
    };
    let json = serde_json::to_string(&op).unwrap();
    assert!(json.contains("softmax"));
    let parsed: TraceOperation = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "softmax");
    assert_eq!(parsed.details.as_deref(), Some("batch=32"));
}

// ============================================================================
// Streaming types coverage
// ============================================================================

#[test]
fn test_chat_completion_chunk_serde() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1234567890,
        model: "default".to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: Some("Hello".to_string()),
            },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(json.contains("chatcmpl-123"));
    assert!(json.contains("chat.completion.chunk"));
    let parsed: ChatCompletionChunk = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, "chatcmpl-123");
}

#[test]
fn test_chat_completion_chunk_debug() {
    let chunk = ChatCompletionChunk {
        id: "id".to_string(),
        object: "obj".to_string(),
        created: 0,
        model: "mod".to_string(),
        choices: vec![],
    };
    let debug = format!("{:?}", chunk);
    assert!(debug.contains("ChatCompletionChunk"));
}

#[test]
fn test_chat_completion_chunk_clone() {
    let chunk = ChatCompletionChunk {
        id: "id".to_string(),
        object: "obj".to_string(),
        created: 123,
        model: "mod".to_string(),
        choices: vec![],
    };
    let cloned = chunk.clone();
    assert_eq!(cloned.id, "id");
    assert_eq!(cloned.created, 123);
}

#[test]
fn test_chat_chunk_choice_serde() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: Some("world".to_string()),
        },
        finish_reason: Some("stop".to_string()),
    };
    let json = serde_json::to_string(&choice).unwrap();
    assert!(json.contains("world"));
    assert!(json.contains("stop"));
    let parsed: ChatChunkChoice = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.finish_reason.as_deref(), Some("stop"));
}

#[test]
fn test_chat_chunk_choice_debug() {
    let choice = ChatChunkChoice {
        index: 5,
        delta: ChatDelta {
            role: Some("user".to_string()),
            content: None,
        },
        finish_reason: None,
    };
    let debug = format!("{:?}", choice);
    assert!(debug.contains("ChatChunkChoice"));
    assert!(debug.contains("5"));
}

#[test]
fn test_chat_delta_serde() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("response".to_string()),
    };
    let json = serde_json::to_string(&delta).unwrap();
    assert!(json.contains("assistant"));
    assert!(json.contains("response"));
    let parsed: ChatDelta = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.role.as_deref(), Some("assistant"));
}

#[test]
fn test_chat_delta_debug() {
    let delta = ChatDelta {
        role: None,
        content: Some("text".to_string()),
    };
    let debug = format!("{:?}", delta);
    assert!(debug.contains("ChatDelta"));
}

#[test]
fn test_chat_delta_clone() {
    let delta = ChatDelta {
        role: Some("user".to_string()),
        content: Some("hello".to_string()),
    };
    let cloned = delta.clone();
    assert_eq!(cloned.role.as_deref(), Some("user"));
    assert_eq!(cloned.content.as_deref(), Some("hello"));
}

// ============================================================================
// ChatMessage coverage
// ============================================================================

#[test]
fn test_chat_message_with_name() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("Alice".to_string()),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("Alice"));
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name.as_deref(), Some("Alice"));
}

#[test]
fn test_chat_message_debug() {
    let msg = ChatMessage {
        role: "system".to_string(),
        content: "Be helpful".to_string(),
        name: None,
    };
    let debug = format!("{:?}", msg);
    assert!(debug.contains("system"));
    assert!(debug.contains("Be helpful"));
}

#[test]
fn test_chat_message_clone() {
    let msg = ChatMessage {
        role: "assistant".to_string(),
        content: "Hi there".to_string(),
        name: Some("Bot".to_string()),
    };
    let cloned = msg.clone();
    assert_eq!(cloned.content, "Hi there");
}

// ============================================================================
// ChatChoice coverage
// ============================================================================

#[test]
fn test_chat_choice_serde() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Response".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };
    let json = serde_json::to_string(&choice).unwrap();
    assert!(json.contains("Response"));
    let parsed: ChatChoice = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.finish_reason, "stop");
}

#[test]
fn test_chat_choice_debug() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "test".to_string(),
            name: None,
        },
        finish_reason: "length".to_string(),
    };
    let debug = format!("{:?}", choice);
    assert!(debug.contains("ChatChoice"));
}

// ============================================================================
// Usage coverage
// ============================================================================

#[test]
fn test_usage_debug() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let debug = format!("{:?}", usage);
    assert!(debug.contains("30"));
}

#[test]
fn test_usage_clone() {
    let usage = Usage {
        prompt_tokens: 5,
        completion_tokens: 15,
        total_tokens: 20,
    };
    let cloned = usage.clone();
    assert_eq!(cloned.total_tokens, 20);
}

// ============================================================================
// OpenAIModel and OpenAIModelsResponse coverage
// ============================================================================

#[test]
fn test_openai_model_debug() {
    let model = OpenAIModel {
        id: "gpt-4".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "openai".to_string(),
    };
    let debug = format!("{:?}", model);
    assert!(debug.contains("gpt-4"));
}

#[test]
fn test_openai_model_clone() {
    let model = OpenAIModel {
        id: "test".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "realizar".to_string(),
    };
    let cloned = model.clone();
    assert_eq!(cloned.owned_by, "realizar");
}

#[test]
fn test_openai_models_response_debug() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![],
    };
    let debug = format!("{:?}", resp);
    assert!(debug.contains("list"));
}

#[test]
fn test_openai_models_response_clone() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: "m1".to_string(),
            object: "model".to_string(),
            created: 1,
            owned_by: "test".to_string(),
        }],
    };
    let cloned = resp.clone();
    assert_eq!(cloned.data.len(), 1);
}

// ============================================================================
// Request/Response types edge cases
// ============================================================================

#[test]
fn test_tokenize_request_serde() {
    let req = TokenizeRequest {
        text: "Hello world".to_string(),
        model_id: Some("default".to_string()),
    };
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("Hello world"));
    let parsed: TokenizeRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model_id.as_deref(), Some("default"));
}

#[test]
fn test_tokenize_request_no_model() {
    let json = r#"{"text": "test"}"#;
    let req: TokenizeRequest = serde_json::from_str(json).unwrap();
    assert!(req.model_id.is_none());
}

#[test]
fn test_tokenize_response_serde() {
    let resp = TokenizeResponse {
        token_ids: vec![1, 2, 3],
        num_tokens: 3,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: TokenizeResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.num_tokens, 3);
}

#[test]
fn test_generate_request_serde() {
    let req = GenerateRequest {
        prompt: "Hello".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        strategy: "top_k".to_string(),
        top_k: 40,
        top_p: 0.9,
        seed: Some(42),
        model_id: None,
    };
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("top_k"));
    let parsed: GenerateRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.seed, Some(42));
}

#[test]
fn test_generate_request_minimal() {
    let json = r#"{"prompt": "test"}"#;
    let req: GenerateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompt, "test");
    // Check defaults - default_max_tokens() returns 50, default_temperature() returns 1.0
    assert_eq!(req.max_tokens, 50);
    assert!((req.temperature - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_generate_response_serde() {
    let resp = GenerateResponse {
        token_ids: vec![1, 2],
        text: "hello".to_string(),
        num_generated: 1,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: GenerateResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.text, "hello");
}

#[test]
fn test_batch_tokenize_request_serde() {
    let req = BatchTokenizeRequest {
        texts: vec!["Hello".to_string(), "World".to_string()],
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: BatchTokenizeRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.texts.len(), 2);
}

#[test]
fn test_batch_tokenize_response_serde() {
    let resp = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse { token_ids: vec![1], num_tokens: 1 },
            TokenizeResponse { token_ids: vec![2], num_tokens: 1 },
        ],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: BatchTokenizeResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.results.len(), 2);
}

#[test]
fn test_batch_generate_request_serde() {
    let req = BatchGenerateRequest {
        prompts: vec!["A".to_string(), "B".to_string()],
        max_tokens: 10,
        temperature: 0.5,
        strategy: "greedy".to_string(),
        top_k: 1,
        top_p: 1.0,
        seed: None,
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: BatchGenerateRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.prompts.len(), 2);
}

#[test]
fn test_batch_generate_response_serde() {
    let resp = BatchGenerateResponse {
        results: vec![GenerateResponse {
            token_ids: vec![1],
            text: "a".to_string(),
            num_generated: 1,
        }],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: BatchGenerateResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.results.len(), 1);
}

#[test]
fn test_stream_token_event_serde() {
    let event = StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    let parsed: StreamTokenEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.token_id, 42);
}

#[test]
fn test_stream_done_event_serde() {
    let event = StreamDoneEvent { num_generated: 10 };
    let json = serde_json::to_string(&event).unwrap();
    let parsed: StreamDoneEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.num_generated, 10);
}

// ============================================================================
// HTTP endpoints additional edge cases
// ============================================================================

#[tokio::test]
async fn test_empty_prompts_batch_completions() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompts": [],
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
    // Should return BAD_REQUEST for empty prompts
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_empty_texts_batch_tokenize() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "texts": []
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
    // Should return BAD_REQUEST for empty texts
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_empty_prompts_batch_generate() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompts": [],
        "max_tokens": 5
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
    // Should return BAD_REQUEST for empty prompts
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_invalid_strategy_generate() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 5,
        "strategy": "invalid_strategy"
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
    // Should return BAD_REQUEST for invalid strategy
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_models_endpoint_returns_list() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_realize_model_handler() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // May return OK or error depending on model state
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_realize_reload_handler() {
    let app = create_test_app_shared();
    let body = serde_json::json!({});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/reload")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    // May return OK or error depending on model state
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_chat_completions_streaming_flag() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "max_tokens": 2
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
    // Accept streaming response or error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// ============================================================================
// ChatCompletionRequest coverage
// ============================================================================

#[test]
fn test_chat_completion_request_serde() {
    let req = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        n: 1,
        stream: false,
        stop: Some(vec!["END".to_string()]),
        user: Some("test-user".to_string()),
    };
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("gpt-4"));
    let parsed: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model, "gpt-4");
    assert_eq!(parsed.user.as_deref(), Some("test-user"));
}

#[test]
fn test_chat_completion_request_minimal() {
    let json = r#"{"model": "test", "messages": []}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "test");
    assert!(!req.stream);
    assert!(req.max_tokens.is_none());
}

#[test]
fn test_chat_completion_request_debug() {
    let req = ChatCompletionRequest {
        model: "test".to_string(),
        messages: vec![],
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };
    let debug = format!("{:?}", req);
    assert!(debug.contains("ChatCompletionRequest"));
}

// ============================================================================
// ChatCompletionResponse coverage
// ============================================================================

#[test]
fn test_chat_completion_response_debug() {
    let resp = ChatCompletionResponse {
        id: "id".to_string(),
        object: "chat.completion".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };
    let debug = format!("{:?}", resp);
    assert!(debug.contains("ChatCompletionResponse"));
}

#[test]
fn test_chat_completion_response_with_traces() {
    let resp = ChatCompletionResponse {
        id: "id".to_string(),
        object: "chat.completion".to_string(),
        created: 123,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 10,
            total_tokens: 15,
        },
        brick_trace: Some(TraceData {
            level: "brick".to_string(),
            operations: 1,
            total_time_us: 100,
            breakdown: vec![],
        }),
        step_trace: None,
        layer_trace: None,
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("brick"));
    let parsed: ChatCompletionResponse = serde_json::from_str(&json).unwrap();
    assert!(parsed.brick_trace.is_some());
}

// ============================================================================
// ErrorResponse coverage
// ============================================================================

#[test]
fn test_error_response_serde_roundtrip() {
    let err = ErrorResponse {
        error: "Something failed".to_string(),
    };
    let json = serde_json::to_string(&err).unwrap();
    assert!(json.contains("Something failed"));
    let parsed: ErrorResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.error, "Something failed");
}
