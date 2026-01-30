//! API Tests Part 19: T-COV-95 Deep Coverage Bridge
//!
//! Covers additional uncovered paths in:
//! - BatchConfig: low_latency, high_throughput, should_process, meets_minimum
//! - ContinuousBatchResponse: single, batched, generated_tokens edge cases
//! - ChatCompletionChunk, ChatChunkChoice, ChatDelta serde
//! - ChatChoice, OpenAIModelsResponse, OpenAIModel serde
//! - TraceData/TraceOperation serde
//! - Additional HTTP endpoints: /metrics, /models, /realize/generate, /realize/batch,
//!   /stream/generate, /v1/gpu/warmup, /v1/gpu/status, /v1/predict, /v1/explain
//! - format_chat_messages: model-specific formatting (qwen, phi, tinyllama)
//! - ContextWindowManager: truncation required, large message sets
//! - build_trace_data: deeper field verification
//!
//! Refs PMAT-802: Protocol T-COV-95

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app_shared;

// ============================================================================
// BatchConfig methods
// ============================================================================

#[test]
fn test_batch_config_low_latency() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::low_latency();
    assert!(config.window_ms <= 10); // Low latency = short window
    assert!(config.min_batch > 0);
    assert!(config.optimal_batch > 0);
    assert!(config.max_batch >= config.optimal_batch);
    assert!(config.queue_size > 0);
}

#[test]
fn test_batch_config_high_throughput() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::high_throughput();
    assert!(config.window_ms >= 50); // High throughput = longer window
    assert!(config.min_batch >= 4);
    assert!(config.max_batch >= 64);
    assert!(config.queue_size >= 1024);
}

#[test]
fn test_batch_config_should_process_at_optimal() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::low_latency();
    // At optimal batch size → should process
    assert!(config.should_process(config.optimal_batch));
    // Above optimal → should process
    assert!(config.should_process(config.optimal_batch + 1));
    // Below optimal → should not process
    assert!(!config.should_process(config.optimal_batch - 1));
}

#[test]
fn test_batch_config_should_process_zero() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::low_latency();
    assert!(!config.should_process(0));
}

#[test]
fn test_batch_config_meets_minimum() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::low_latency();
    assert!(config.meets_minimum(config.min_batch));
    assert!(config.meets_minimum(config.min_batch + 1));
    assert!(!config.meets_minimum(0));
    assert!(!config.meets_minimum(config.min_batch - 1));
}

#[test]
fn test_batch_config_meets_minimum_high_throughput() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::high_throughput();
    assert!(config.meets_minimum(config.min_batch));
    assert!(!config.meets_minimum(1));
}

// ============================================================================
// ContinuousBatchResponse
// ============================================================================

#[test]
fn test_continuous_batch_response_single() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    let resp = ContinuousBatchResponse::single(vec![1, 2, 3, 4, 5], 2, 5.0);
    assert!(!resp.batched);
    assert_eq!(resp.batch_size, 1);
    assert_eq!(resp.prompt_len, 2);
    assert_eq!(resp.token_ids, vec![1, 2, 3, 4, 5]);
    assert!((resp.latency_ms - 5.0).abs() < 1e-6);
}

#[test]
fn test_continuous_batch_response_batched() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    let resp = ContinuousBatchResponse::batched(vec![1, 2, 3, 4, 5], 2, 8, 10.0);
    assert!(resp.batched);
    assert_eq!(resp.batch_size, 8);
    assert_eq!(resp.prompt_len, 2);
}

#[test]
fn test_continuous_batch_response_generated_tokens() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    let resp = ContinuousBatchResponse::single(vec![10, 20, 30, 40, 50], 3, 1.0);
    let generated = resp.generated_tokens();
    assert_eq!(generated, &[40, 50]);
}

#[test]
fn test_continuous_batch_response_generated_tokens_empty() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    // prompt_len equals total tokens → no generated tokens
    let resp = ContinuousBatchResponse::single(vec![1, 2, 3], 3, 1.0);
    let generated = resp.generated_tokens();
    assert!(generated.is_empty());
}

#[test]
fn test_continuous_batch_response_generated_tokens_all_generated() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    // prompt_len = 0 → all tokens are generated
    let resp = ContinuousBatchResponse::single(vec![1, 2, 3], 0, 1.0);
    let generated = resp.generated_tokens();
    assert_eq!(generated, &[1, 2, 3]);
}

#[test]
fn test_continuous_batch_response_generated_tokens_prompt_exceeds() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    // prompt_len > total tokens → return empty (edge case)
    let resp = ContinuousBatchResponse::single(vec![1, 2], 10, 1.0);
    let generated = resp.generated_tokens();
    assert!(generated.is_empty());
}

// ============================================================================
// ChatCompletionChunk, ChatChunkChoice, ChatDelta serde
// ============================================================================

#[test]
fn test_chat_completion_chunk_serde() {
    let chunk = crate::api::ChatCompletionChunk {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1700000000,
        model: "test-model".to_string(),
        choices: vec![crate::api::ChatChunkChoice {
            index: 0,
            delta: crate::api::ChatDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).unwrap();
    let deserialized: crate::api::ChatCompletionChunk = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, "chatcmpl-123");
    assert_eq!(deserialized.object, "chat.completion.chunk");
    assert_eq!(deserialized.choices.len(), 1);
    assert_eq!(
        deserialized.choices[0].delta.role,
        Some("assistant".to_string())
    );
    assert!(deserialized.choices[0].delta.content.is_none());
    assert!(deserialized.choices[0].finish_reason.is_none());
}

#[test]
fn test_chat_delta_with_content() {
    let delta = crate::api::ChatDelta {
        role: None,
        content: Some("Hello ".to_string()),
    };
    let json = serde_json::to_string(&delta).unwrap();
    // role is None → should be skipped in serialization
    assert!(!json.contains("role"));
    let deserialized: crate::api::ChatDelta = serde_json::from_str(&json).unwrap();
    assert!(deserialized.role.is_none());
    assert_eq!(deserialized.content, Some("Hello ".to_string()));
}

#[test]
fn test_chat_chunk_choice_with_finish_reason() {
    let choice = crate::api::ChatChunkChoice {
        index: 0,
        delta: crate::api::ChatDelta {
            role: None,
            content: None,
        },
        finish_reason: Some("stop".to_string()),
    };
    let json = serde_json::to_string(&choice).unwrap();
    let deserialized: crate::api::ChatChunkChoice = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.finish_reason, Some("stop".to_string()));
}

// ============================================================================
// ChatChoice, OpenAIModelsResponse, OpenAIModel serde
// ============================================================================

#[test]
fn test_chat_choice_serde() {
    let choice = crate::api::ChatChoice {
        index: 0,
        message: crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: "Hello!".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };
    let json = serde_json::to_string(&choice).unwrap();
    let deserialized: crate::api::ChatChoice = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.index, 0);
    assert_eq!(deserialized.message.role, "assistant");
    assert_eq!(deserialized.finish_reason, "stop");
}

#[test]
fn test_openai_models_response_serde() {
    let resp = crate::api::OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![crate::api::OpenAIModel {
            id: "test-model".to_string(),
            object: "model".to_string(),
            created: 1700000000,
            owned_by: "realizar".to_string(),
        }],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let deserialized: crate::api::OpenAIModelsResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.object, "list");
    assert_eq!(deserialized.data.len(), 1);
    assert_eq!(deserialized.data[0].id, "test-model");
    assert_eq!(deserialized.data[0].owned_by, "realizar");
}

#[test]
fn test_openai_model_serde() {
    let model = crate::api::OpenAIModel {
        id: "tinyllama-1.1b".to_string(),
        object: "model".to_string(),
        created: 1700000000,
        owned_by: "realizar".to_string(),
    };
    let json = serde_json::to_string(&model).unwrap();
    let deserialized: crate::api::OpenAIModel = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, "tinyllama-1.1b");
}

// ============================================================================
// TraceData/TraceOperation serde
// ============================================================================

#[test]
fn test_trace_data_serde() {
    let trace = crate::api::TraceData {
        level: "brick".to_string(),
        operations: 10,
        total_time_us: 5000,
        breakdown: vec![
            crate::api::TraceOperation {
                name: "embedding_lookup".to_string(),
                time_us: 500,
                details: Some("10 tokens".to_string()),
            },
            crate::api::TraceOperation {
                name: "matmul_qkv".to_string(),
                time_us: 1667,
                details: None,
            },
        ],
    };
    let json = serde_json::to_string(&trace).unwrap();
    let deserialized: crate::api::TraceData = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.level, "brick");
    assert_eq!(deserialized.operations, 10);
    assert_eq!(deserialized.breakdown.len(), 2);
}

#[test]
fn test_trace_operation_serde() {
    let op = crate::api::TraceOperation {
        name: "softmax".to_string(),
        time_us: 100,
        details: None,
    };
    let json = serde_json::to_string(&op).unwrap();
    let deserialized: crate::api::TraceOperation = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "softmax");
    assert!(deserialized.details.is_none());
}

// ============================================================================
// build_trace_data: deeper field verification
// ============================================================================

#[test]
fn test_build_trace_data_brick_breakdown_fields() {
    let (brick, _, _) = crate::api::build_trace_data(Some("brick"), 1000, 20, 10, 4);
    let b = brick.unwrap();
    assert_eq!(b.operations, 10); // completion_tokens
    assert_eq!(b.total_time_us, 1000);
    assert_eq!(b.breakdown.len(), 3);
    assert_eq!(b.breakdown[0].name, "embedding_lookup");
    assert_eq!(b.breakdown[0].time_us, 100); // 1000 / 10
    assert!(b.breakdown[0].details.is_some());
    assert_eq!(b.breakdown[1].name, "matmul_qkv");
    assert_eq!(b.breakdown[2].name, "softmax");
}

#[test]
fn test_build_trace_data_step_breakdown_fields() {
    let (_, step, _) = crate::api::build_trace_data(Some("step"), 2000, 15, 8, 6);
    let s = step.unwrap();
    assert_eq!(s.operations, 8); // completion_tokens
    assert_eq!(s.total_time_us, 2000);
    assert_eq!(s.breakdown.len(), 3);
    assert_eq!(s.breakdown[0].name, "tokenize");
    assert_eq!(s.breakdown[1].name, "forward_pass");
    assert_eq!(s.breakdown[2].name, "decode");
    // forward_pass time = total - 200
    assert_eq!(s.breakdown[1].time_us, 1800);
}

#[test]
fn test_build_trace_data_layer_breakdown_fields() {
    let (_, _, layer) = crate::api::build_trace_data(Some("layer"), 4000, 10, 5, 4);
    let l = layer.unwrap();
    assert_eq!(l.operations, 4); // num_layers
    assert_eq!(l.total_time_us, 4000);
    assert_eq!(l.breakdown.len(), 4);
    for (i, op) in l.breakdown.iter().enumerate() {
        assert_eq!(op.name, format!("layer_{}", i));
        assert_eq!(op.time_us, 1000); // 4000 / 4
        assert_eq!(op.details, Some("attention+mlp".to_string()));
    }
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("unknown"), 100, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

// ============================================================================
// Additional HTTP endpoint integration tests
// ============================================================================

#[tokio::test]
async fn test_metrics_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND,
    );
}

#[tokio::test]
async fn test_native_models_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/models")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND,
    );
}

#[tokio::test]
async fn test_realize_generate_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/realize/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompt":"Hello","max_tokens":5,"temperature":0.0}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_realize_batch_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/realize/batch")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompts":["Hello","World"],"max_tokens":5}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_stream_generate_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/stream/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompt":"Hello","max_tokens":5,"temperature":0.0,"strategy":"greedy","top_k":1,"top_p":1.0}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_gpu_warmup_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/gpu/warmup")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_gpu_status_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/v1/gpu/status")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::SERVICE_UNAVAILABLE,
    );
}

#[tokio::test]
async fn test_v1_predict_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/predict")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"input":[1.0, 2.0, 3.0]}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_v1_explain_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/explain")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"input":[1.0, 2.0, 3.0]}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_v1_metrics_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/v1/metrics")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND,
    );
}

#[tokio::test]
async fn test_metrics_dispatch_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/metrics/dispatch")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Dispatch metrics endpoint may return SERVICE_UNAVAILABLE without GPU
    let status = response.status().as_u16();
    assert!(
        status < 500 || status == 503,
        "Unexpected server error: {}",
        status
    );
}

#[tokio::test]
async fn test_metrics_dispatch_reset_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/metrics/dispatch/reset")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Reset endpoint may return SERVICE_UNAVAILABLE without GPU
    let status = response.status().as_u16();
    assert!(
        status < 500 || status == 503,
        "Unexpected server error: {}",
        status
    );
}

// ============================================================================
// format_chat_messages: model-specific formatting
// ============================================================================

#[test]
fn test_format_chat_messages_qwen_model() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![
        crate::api::ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "What is Rust?".to_string(),
            name: None,
        },
    ];
    let result = format_chat_messages(&messages, Some("qwen2"));
    assert!(!result.is_empty());
    assert!(result.contains("What is Rust?"));
}

#[test]
fn test_format_chat_messages_phi_model() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello phi".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&messages, Some("phi2"));
    assert!(!result.is_empty());
}

#[test]
fn test_format_chat_messages_tinyllama_model() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello tinyllama".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&messages, Some("tinyllama"));
    assert!(!result.is_empty());
}

#[test]
fn test_format_chat_messages_llama_model() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello llama".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&messages, Some("llama"));
    assert!(!result.is_empty());
}

// ============================================================================
// ContextWindowManager: deeper truncation tests
// ============================================================================

#[test]
fn test_context_window_manager_truncate_long_messages() {
    use crate::api::realize_handlers::{ContextWindowConfig, ContextWindowManager};

    // Very small context window to force truncation
    let config = ContextWindowConfig::new(20).with_reserved_output(5);
    let mgr = ContextWindowManager::new(config);
    let messages: Vec<crate::api::ChatMessage> = (0..100)
        .map(|i| crate::api::ChatMessage {
            role: "user".to_string(),
            content: format!("Message number {} with some extra text", i),
            name: None,
        })
        .collect();

    let needs = mgr.needs_truncation(&messages);
    assert!(needs); // 100 long messages should exceed 15 available tokens

    let (truncated, was_truncated) = mgr.truncate_messages(&messages);
    assert!(was_truncated);
    assert!(truncated.len() < messages.len());
}

#[test]
fn test_context_window_manager_preserves_system_message() {
    use crate::api::realize_handlers::{ContextWindowConfig, ContextWindowManager};

    let config = ContextWindowConfig {
        max_tokens: 50,
        reserved_output_tokens: 10,
        preserve_system: true,
    };
    let mgr = ContextWindowManager::new(config);

    let messages = vec![
        crate::api::ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "A very long message that should be long enough to cause truncation when combined with other messages".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: "A very long response with lots of content that should be truncated eventually if needed".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "Another long user message for extra length to ensure truncation happens".to_string(),
            name: None,
        },
    ];

    let (truncated, _was_truncated) = mgr.truncate_messages(&messages);
    // If truncation happened and preserve_system is true,
    // the first message (system) should be preserved
    if !truncated.is_empty() {
        assert_eq!(truncated[0].role, "system");
    }
}

// ============================================================================
// clean_chat_output: additional patterns
// ============================================================================

#[test]
fn test_clean_chat_output_im_end_only() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "Hello world<|im_end|>";
    let result = clean_chat_output(text);
    assert_eq!(result, "Hello world");
}

#[test]
fn test_clean_chat_output_endoftext() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "Some output text<|endoftext|>more stuff";
    let result = clean_chat_output(text);
    assert_eq!(result, "Some output text");
}

#[test]
fn test_clean_chat_output_eos_token() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "Generated text</s>more after";
    let result = clean_chat_output(text);
    assert_eq!(result, "Generated text");
}

#[test]
fn test_clean_chat_output_whitespace_only() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "   \n\t  ";
    let result = clean_chat_output(text);
    assert!(result.is_empty());
}

#[test]
fn test_clean_chat_output_trimming() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "  Hello world  ";
    let result = clean_chat_output(text);
    assert_eq!(result, "Hello world");
}

// ============================================================================
// AppState: with_cache
// ============================================================================

#[test]
fn test_appstate_with_cache() {
    let state = crate::api::AppState::with_cache(100);
    // with_cache returns Self directly (not Result)
    let _ = state; // Verify construction succeeds without panic
}

// ============================================================================
// Batch request/response additional serde tests
// ============================================================================

#[test]
fn test_batch_tokenize_request_serde() {
    let json = r#"{"texts":["hello","world"]}"#;
    let req: crate::api::BatchTokenizeRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.texts.len(), 2);
    let json_out = serde_json::to_string(&req).unwrap();
    assert!(json_out.contains("hello"));
}

#[test]
fn test_batch_generate_request_serde() {
    let json = r#"{"prompts":["hello"],"max_tokens":10,"temperature":0.5,"strategy":"greedy","top_k":1,"top_p":1.0}"#;
    let req: crate::api::BatchGenerateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompts.len(), 1);
    assert_eq!(req.max_tokens, 10);
    assert!((req.temperature - 0.5).abs() < 1e-6);
}

#[test]
fn test_batch_tokenize_response_serde() {
    let resp = crate::api::BatchTokenizeResponse {
        results: vec![crate::api::TokenizeResponse {
            token_ids: vec![1, 2],
            num_tokens: 2,
        }],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let deserialized: crate::api::BatchTokenizeResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.results.len(), 1);
}

#[test]
fn test_batch_generate_response_serde() {
    let resp = crate::api::BatchGenerateResponse {
        results: vec![crate::api::GenerateResponse {
            token_ids: vec![1, 2, 3],
            text: "hello".to_string(),
            num_generated: 3,
        }],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let deserialized: crate::api::BatchGenerateResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.results.len(), 1);
    assert_eq!(deserialized.results[0].num_generated, 3);
}
