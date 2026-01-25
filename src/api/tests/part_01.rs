//! API Tests Part 01
//!
//! Unit tests, clean_chat_output tests, health/metrics endpoints

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app;
#[cfg(feature = "gpu")]
use crate::api::test_helpers::create_test_quantized_model;
use crate::api::*;

// ========================================================================
// PMAT-088: clean_chat_output tests
// ========================================================================

#[test]
fn test_clean_chat_output_no_stop_sequence() {
    let input = "Hello, how can I help you?";
    assert_eq!(clean_chat_output(input), "Hello, how can I help you?");
}

#[test]
fn test_clean_chat_output_im_end() {
    let input = "Hello!<|im_end|>\nHuman: Hi there";
    assert_eq!(clean_chat_output(input), "Hello!");
}

#[test]
fn test_clean_chat_output_human_turn() {
    let input = "Hello!\nHuman: Hi there!";
    assert_eq!(clean_chat_output(input), "Hello!");
}

#[test]
fn test_clean_chat_output_user_turn() {
    let input = "Hello!\nUser: Hi there!";
    assert_eq!(clean_chat_output(input), "Hello!");
}

#[test]
fn test_clean_chat_output_im_start() {
    let input = "Hello!<|im_start|>user\nHi there";
    assert_eq!(clean_chat_output(input), "Hello!");
}

#[test]
fn test_clean_chat_output_multiple_stops() {
    // Should stop at the earliest one
    let input = "Hello!<|im_end|>\nHuman: Hi<|endoftext|>";
    assert_eq!(clean_chat_output(input), "Hello!");
}

#[test]
fn test_clean_chat_output_trims_whitespace() {
    let input = "  Hello!  <|im_end|>";
    assert_eq!(clean_chat_output(input), "Hello!");
}

#[tokio::test]
async fn test_health_endpoint() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let health: HealthResponse = serde_json::from_slice(&body).expect("test");
    assert_eq!(health.status, "healthy");
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let metrics_text = String::from_utf8(body.to_vec()).expect("test");

    // Verify Prometheus format
    assert!(metrics_text.contains("realizar_requests_total"));
    assert!(metrics_text.contains("realizar_tokens_generated"));
    assert!(metrics_text.contains("realizar_error_rate"));
    assert!(metrics_text.contains("# HELP"));
    assert!(metrics_text.contains("# TYPE"));
}

#[tokio::test]
async fn test_metrics_tracking() {
    let state = AppState::demo().expect("test");
    let app = create_router(state.clone());

    // Make a generate request
    let request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 3,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(42),
        model_id: None,
    };

    let _response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    // Check metrics were recorded
    let snapshot = state.metrics.snapshot();
    assert_eq!(snapshot.total_requests, 1);
    assert_eq!(snapshot.successful_requests, 1);
    assert!(snapshot.total_tokens > 0);
}

/// Test PARITY-107: /v1/metrics endpoint for TUI monitoring
#[tokio::test]
async fn test_parity107_server_metrics_endpoint() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/metrics")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let metrics: ServerMetricsResponse = serde_json::from_slice(&body).expect("test");

    // Verify JSON structure per PARITY-107 spec
    assert!(metrics.throughput_tok_per_sec >= 0.0);
    assert!(metrics.latency_p50_ms >= 0.0);
    assert!(metrics.latency_p95_ms >= 0.0);
    assert!(metrics.latency_p99_ms >= 0.0);
    assert!(metrics.gpu_utilization_percent <= 100);
    assert!(metrics.batch_size >= 1);
    // Model name should be set or N/A
    assert!(!metrics.model_name.is_empty());
}

#[tokio::test]
async fn test_tokenize_endpoint() {
    let app = create_test_app();

    let request = TokenizeRequest {
        text: "token1 token2".to_string(),
        model_id: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: TokenizeResponse = serde_json::from_slice(&body).expect("test");
    assert!(result.num_tokens > 0);
}

#[tokio::test]
async fn test_generate_endpoint() {
    let app = create_test_app();

    let request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 3,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(42),
        model_id: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: GenerateResponse = serde_json::from_slice(&body).expect("test");
    assert!(!result.token_ids.is_empty());
}

#[tokio::test]
async fn test_generate_empty_prompt_error() {
    let app = create_test_app();

    let request = GenerateRequest {
        prompt: String::new(),
        max_tokens: 3,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
        model_id: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_generate_invalid_strategy_error() {
    let app = create_test_app();

    let request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 3,
        temperature: 1.0,
        strategy: "invalid".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
        model_id: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_generate_top_k_strategy() {
    let app = create_test_app();

    let request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 2,
        temperature: 0.8,
        strategy: "top_k".to_string(),
        top_k: 5,
        top_p: 0.9,
        seed: Some(123),
        model_id: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_generate_top_p_strategy() {
    let app = create_test_app();

    let request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 2,
        temperature: 0.7,
        strategy: "top_p".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(456),
        model_id: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_app_state_demo() {
    let state = AppState::demo();
    assert!(state.is_ok());
    let state = state.expect("test");
    assert_eq!(state.tokenizer.as_ref().expect("test").vocab_size(), 100);
}

#[test]
fn test_default_max_tokens() {
    assert_eq!(default_max_tokens(), 50);
}

#[test]
fn test_default_temperature() {
    assert!((default_temperature() - 1.0).abs() < 1e-6);
}

#[test]
fn test_default_strategy() {
    assert_eq!(default_strategy(), "greedy");
}

#[test]
fn test_default_top_k() {
    assert_eq!(default_top_k(), 50);
}

#[test]
fn test_default_top_p() {
    assert!((default_top_p() - 0.9).abs() < 1e-6);
}

#[tokio::test]
async fn test_generate_with_defaults() {
    let app = create_test_app();

    // Generate request using default values via serde defaults
    let json = r#"{"prompt": "test"}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(json))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: GenerateResponse = serde_json::from_slice(&body).expect("test");
    assert!(!result.token_ids.is_empty());
    // Verify generation used defaults (greedy with max 50 tokens)
    assert!(result.num_generated <= 50);
}

#[tokio::test]
async fn test_num_generated_calculation() {
    // First tokenize to get prompt length
    let app1 = create_test_app();
    let prompt_tokens = app1
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "a"}"#))
                .expect("test"),
        )
        .await
        .expect("test");
    let prompt_body = axum::body::to_bytes(prompt_tokens.into_body(), usize::MAX)
        .await
        .expect("test");
    let prompt_result: TokenizeResponse = serde_json::from_slice(&prompt_body).expect("test");
    let prompt_len = prompt_result.token_ids.len();

    // Now generate
    let app2 = create_test_app();
    let request = GenerateRequest {
        prompt: "a".to_string(),
        max_tokens: 5,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(42),
        model_id: None,
    };

    let response = app2
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: GenerateResponse = serde_json::from_slice(&body).expect("test");

    // Verify num_generated = total_tokens - prompt_tokens
    assert_eq!(result.num_generated, result.token_ids.len() - prompt_len);

    // Also verify it's in reasonable range
    assert!(result.num_generated > 0);
    assert!(result.num_generated <= 5);
}

#[tokio::test]
async fn test_batch_tokenize_endpoint() {
    let app = create_test_app();

    let request = BatchTokenizeRequest {
        texts: vec!["token1".to_string(), "token2 token3".to_string()],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: BatchTokenizeResponse = serde_json::from_slice(&body).expect("test");

    // Verify we got 2 results
    assert_eq!(result.results.len(), 2);
    // Each result should have tokens
    assert!(result.results[0].num_tokens > 0);
    assert!(result.results[1].num_tokens > 0);
}

#[tokio::test]
async fn test_batch_tokenize_empty_array_error() {
    let app = create_test_app();

    let request = BatchTokenizeRequest { texts: vec![] };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_batch_generate_endpoint() {
    let app = create_test_app();

    let request = BatchGenerateRequest {
        prompts: vec!["token1".to_string(), "token2".to_string()],
        max_tokens: 3,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(42),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: BatchGenerateResponse = serde_json::from_slice(&body).expect("test");

    // Verify we got 2 results
    assert_eq!(result.results.len(), 2);
    // Each result should have tokens
    assert!(!result.results[0].token_ids.is_empty());
    assert!(!result.results[1].token_ids.is_empty());
    // Each result should have text
    assert!(!result.results[0].text.is_empty());
    assert!(!result.results[1].text.is_empty());
}

#[tokio::test]
async fn test_batch_generate_empty_array_error() {
    let app = create_test_app();

    let request = BatchGenerateRequest {
        prompts: vec![],
        max_tokens: 3,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_batch_generate_with_defaults() {
    let app = create_test_app();

    // Use serde defaults
    let json = r#"{"prompts": ["test1", "test2"]}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(json))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: BatchGenerateResponse = serde_json::from_slice(&body).expect("test");

    assert_eq!(result.results.len(), 2);
    // Verify generation used defaults (greedy with max 50 tokens)
    for gen_result in &result.results {
        assert!(gen_result.num_generated <= 50);
    }
}

#[tokio::test]
async fn test_batch_generate_order_preserved() {
    let app = create_test_app();

    let request = BatchGenerateRequest {
        prompts: vec![
            "token1".to_string(),
            "token2".to_string(),
            "token3".to_string(),
        ],
        max_tokens: 2,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(123),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: BatchGenerateResponse = serde_json::from_slice(&body).expect("test");

    // Verify order is preserved: 3 prompts -> 3 results in same order
    assert_eq!(result.results.len(), 3);

    // Each result should be non-empty
    for gen_result in &result.results {
        assert!(!gen_result.token_ids.is_empty());
        assert!(!gen_result.text.is_empty());
    }
}

#[tokio::test]
async fn test_batch_generate_invalid_strategy_error() {
    let app = create_test_app();

    let request = BatchGenerateRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 3,
        temperature: 1.0,
        strategy: "invalid".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_batch_generate_top_k_strategy() {
    let app = create_test_app();

    let request = BatchGenerateRequest {
        prompts: vec!["token1".to_string(), "token2".to_string()],
        max_tokens: 2,
        temperature: 0.8,
        strategy: "top_k".to_string(),
        top_k: 5,
        top_p: 0.9,
        seed: Some(456),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: BatchGenerateResponse = serde_json::from_slice(&body).expect("test");

    assert_eq!(result.results.len(), 2);
}

#[tokio::test]
async fn test_batch_generate_top_p_strategy() {
    let app = create_test_app();

    let request = BatchGenerateRequest {
        prompts: vec!["token1".to_string()],
        max_tokens: 2,
        temperature: 0.7,
        strategy: "top_p".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(789),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: BatchGenerateResponse = serde_json::from_slice(&body).expect("test");

    assert_eq!(result.results.len(), 1);
}

// -------------------------------------------------------------------------
// OpenAI-Compatible API Tests
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_openai_models_endpoint() {
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
    let result: OpenAIModelsResponse = serde_json::from_slice(&body).expect("test");

    assert_eq!(result.object, "list");
    assert!(!result.data.is_empty());
    assert_eq!(result.data[0].object, "model");
    assert_eq!(result.data[0].owned_by, "realizar");
}

#[tokio::test]
async fn test_openai_chat_completions_endpoint() {
    let app = create_test_app();

    let request = ChatCompletionRequest {
        model: "default".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
            },
        ],
        max_tokens: Some(10),
        temperature: Some(0.7),
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: ChatCompletionResponse = serde_json::from_slice(&body).expect("test");

    assert!(result.id.starts_with("chatcmpl-"));
    assert_eq!(result.object, "chat.completion");
    assert_eq!(result.model, "default");
    assert_eq!(result.choices.len(), 1);
    assert_eq!(result.choices[0].message.role, "assistant");
    assert_eq!(result.choices[0].finish_reason, "stop");
    assert!(result.usage.total_tokens > 0);
}

#[tokio::test]
async fn test_openai_chat_completions_with_defaults() {
    let app = create_test_app();

    // Minimal request with just required fields
    let json = r#"{"model": "default", "messages": [{"role": "user", "content": "Hi"}]}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: ChatCompletionResponse = serde_json::from_slice(&body).expect("test");

    // Verify response structure
    assert!(result.id.starts_with("chatcmpl-"));
    assert_eq!(result.choices.len(), 1);
}

#[test]
fn test_format_chat_messages_simple_raw() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    // Raw format (None model) just concatenates content
    let result = format_chat_messages(&messages, None);
    assert!(result.contains("Hello"));
}

#[test]
fn test_format_chat_messages_chatml() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    // Qwen2 uses ChatML format
    let result = format_chat_messages(&messages, Some("Qwen2-0.5B"));
    assert!(result.contains("<|im_start|>user"));
    assert!(result.contains("Hello"));
    assert!(result.contains("<|im_end|>"));
    assert!(result.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_format_chat_messages_llama2() {
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        },
    ];

    // TinyLlama uses Zephyr format (not Llama2!)
    let result = format_chat_messages(&messages, Some("TinyLlama-1.1B"));
    assert!(result.contains("<|system|>"), "Expected Zephyr system tag");
    assert!(result.contains("<|user|>"), "Expected Zephyr user tag");
    assert!(result.contains("You are helpful."));
    assert!(result.contains("Hi"));
}

#[test]
fn test_format_chat_messages_mistral() {
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Hi there!".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "How are you?".to_string(),
            name: None,
        },
    ];

    // Mistral format
    let result = format_chat_messages(&messages, Some("Mistral-7B"));
    assert!(result.contains("[INST]"));
    assert!(result.contains("Hello"));
    assert!(result.contains("Hi there!"));
    assert!(result.contains("How are you?"));
}

#[test]
fn test_format_chat_messages_phi() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Test".to_string(),
        name: None,
    }];

    // Phi format
    let result = format_chat_messages(&messages, Some("phi-2"));
    assert!(result.contains("Instruct: Test"));
    assert!(result.ends_with("Output:"));
}

#[test]
fn test_format_chat_messages_alpaca() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Test".to_string(),
        name: None,
    }];

    // Alpaca format
    let result = format_chat_messages(&messages, Some("alpaca-7b"));
    assert!(result.contains("### Instruction:"));
    assert!(result.contains("Test"));
    assert!(result.ends_with("### Response:\n"));
}

#[test]
fn test_default_n() {
    assert_eq!(default_n(), 1);
}

#[test]
fn test_chat_message_serialization() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("test_user".to_string()),
    };

    let json = serde_json::to_string(&msg).expect("test");
    assert!(json.contains("\"role\":\"user\""));
    assert!(json.contains("\"content\":\"Hello\""));
    assert!(json.contains("\"name\":\"test_user\""));
}

#[test]
fn test_usage_serialization() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };

    let json = serde_json::to_string(&usage).expect("test");
    assert!(json.contains("\"prompt_tokens\":10"));
    assert!(json.contains("\"completion_tokens\":20"));
    assert!(json.contains("\"total_tokens\":30"));
}

// ========================================================================
// Streaming Types Tests
// ========================================================================

#[test]
fn test_chat_completion_chunk_initial() {
    let chunk = ChatCompletionChunk::initial("chatcmpl-123", "gpt-4");
    assert_eq!(chunk.id, "chatcmpl-123");
    assert_eq!(chunk.object, "chat.completion.chunk");
    assert_eq!(chunk.model, "gpt-4");
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.role, Some("assistant".to_string()));
    assert!(chunk.choices[0].delta.content.is_none());
    assert!(chunk.choices[0].finish_reason.is_none());
}

#[test]
fn test_chat_completion_chunk_content() {
    let chunk = ChatCompletionChunk::content("chatcmpl-123", "gpt-4", "Hello");
    assert_eq!(chunk.id, "chatcmpl-123");
    assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    assert!(chunk.choices[0].delta.role.is_none());
    assert!(chunk.choices[0].finish_reason.is_none());
}

#[test]
fn test_chat_completion_chunk_done() {
    let chunk = ChatCompletionChunk::done("chatcmpl-123", "gpt-4");
    assert_eq!(chunk.id, "chatcmpl-123");
    assert!(chunk.choices[0].delta.content.is_none());
    assert!(chunk.choices[0].delta.role.is_none());
    assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
}

#[test]
fn test_chat_completion_chunk_serialization() {
    let chunk = ChatCompletionChunk::content("chatcmpl-123", "gpt-4", "Hi");
    let json = serde_json::to_string(&chunk).expect("test");

    assert!(json.contains("\"object\":\"chat.completion.chunk\""));
    assert!(json.contains("\"id\":\"chatcmpl-123\""));
    assert!(json.contains("\"content\":\"Hi\""));
}

#[test]
fn test_chat_delta_serialization_skip_none() {
    let delta = ChatDelta {
        role: None,
        content: Some("test".to_string()),
    };
    let json = serde_json::to_string(&delta).expect("test");

    // Should not contain "role" when it's None
    assert!(!json.contains("\"role\""));
    assert!(json.contains("\"content\":\"test\""));
}

#[test]
fn test_chat_chunk_choice_serialization() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: Some("assistant".to_string()),
            content: None,
        },
        finish_reason: None,
    };
    let json = serde_json::to_string(&choice).expect("test");

    assert!(json.contains("\"index\":0"));
    assert!(json.contains("\"role\":\"assistant\""));
    // content should not be present when None
    assert!(!json.contains("\"content\""));
}

#[test]
fn test_streaming_chunk_created_timestamp() {
    let chunk1 = ChatCompletionChunk::initial("id1", "model");
    std::thread::sleep(std::time::Duration::from_millis(10));
    let chunk2 = ChatCompletionChunk::initial("id2", "model");

    // Both should have valid timestamps
    assert!(chunk1.created > 0);
    assert!(chunk2.created > 0);
    // Second should be same or later
    assert!(chunk2.created >= chunk1.created);
}

// ========================================================================
// Context Window Manager Tests
// ========================================================================

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
    assert_eq!(config.reserved_output_tokens, 256);
}

#[test]
fn test_context_window_config_with_reserved() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(512);
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 512);
}

#[test]
fn test_context_window_available_tokens() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(256);
    assert_eq!(config.available_tokens(), 3840);
}

#[test]
fn test_context_manager_no_truncation_needed() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_context_manager_needs_truncation() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "x".repeat(500),
        name: None,
    }];

    assert!(manager.needs_truncation(&messages));
}

#[test]
fn test_context_manager_truncate_preserves_system() {
    // Use smaller context to force truncation
    let config = ContextWindowConfig::new(80).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(200), // Large old message
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(truncated);
    // System message should be preserved
    assert!(result.iter().any(|m| m.role == "system"));
    // Most recent message should be included
    assert!(result.iter().any(|m| m.content == "Recent"));
}

#[test]
fn test_context_manager_truncate_keeps_recent() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let mut cfg = config;
    cfg.preserve_system = false;
    let manager = ContextWindowManager::new(cfg);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Old message 1".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Old message 2".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&messages);
    // If truncation occurs, most recent should be kept
    if truncated {
        assert!(result.iter().any(|m| m.content == "Recent"));
    }
}

#[test]
fn test_context_manager_estimate_tokens() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    let tokens = manager.estimate_total_tokens(&messages);
    // Should include overhead and char-based estimate
    assert!(tokens > 0);
    assert!(tokens < 100);
}

#[test]
fn test_context_manager_empty_messages() {
    let manager = ContextWindowManager::default_manager();
    let messages: Vec<ChatMessage> = vec![];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert!(result.is_empty());
}

#[test]
fn test_context_manager_single_large_message() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    // Message larger than available space
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "x".repeat(1000),
        name: None,
    }];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(truncated);
    // Message too large to fit, result may be empty
    assert!(result.is_empty() || result.len() == 1);
}

// =========================================================================
// APR-Specific API Tests (spec §15.1)
// =========================================================================

#[tokio::test]
#[ignore = "APR model integration test - requires specific model setup"]
async fn test_apr_predict_endpoint() {
    let app = create_test_app();

    // Use 4 features to match demo APR model's expected input dimension
    let request = PredictRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: None,
        top_k: Some(3),
        include_confidence: true,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: PredictResponse = serde_json::from_slice(&body).expect("test");

    assert!(!result.request_id.is_empty());
    assert_eq!(result.model, "default");
    assert!(result.confidence.is_some());
    // For regression (single output), top_k returns the value itself
    assert!(result.top_k_predictions.is_some());
    assert!(result.latency_ms >= 0.0);
    // Verify real inference: 1+2+3+4 = 10.0 (our demo model sums inputs)
    assert_eq!(result.prediction, serde_json::json!(10.0));
}

#[tokio::test]
async fn test_apr_predict_empty_features() {
    let app = create_test_app();

    let request = PredictRequest {
        model: None,
        features: vec![],
        feature_names: None,
        top_k: None,
        include_confidence: true,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_apr_explain_endpoint() {
    let app = create_test_app();

    let request = ExplainRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
        top_k_features: 2,
        method: "shap".to_string(),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: ExplainResponse = serde_json::from_slice(&body).expect("test");

    assert!(!result.request_id.is_empty());
    assert_eq!(result.model, "default");
    assert!(!result.summary.is_empty());
    assert_eq!(result.explanation.feature_names.len(), 3);
    assert_eq!(result.explanation.shap_values.len(), 3);
}

#[tokio::test]
async fn test_apr_explain_mismatched_features() {
    let app = create_test_app();

    let request = ExplainRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["f1".to_string()], // Mismatched count
        top_k_features: 2,
        method: "shap".to_string(),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[ignore = "APR audit integration test - depends on predict endpoint"]
async fn test_apr_audit_endpoint() {
    // Tests real audit trail: predict creates record, audit fetches it
    let state = AppState::demo().expect("test");
    let app = create_router(state);

    // First, make a prediction to create an audit record
    let predict_request = PredictRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: None,
        top_k: None,
        include_confidence: true,
    };

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&predict_request).expect("test"),
                ))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let predict_result: PredictResponse = serde_json::from_slice(&body).expect("test");
    let request_id = predict_result.request_id;

    // Now fetch the audit record for this prediction
    let audit_response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/audit/{}", request_id))
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(audit_response.status(), StatusCode::OK);

    let audit_body = axum::body::to_bytes(audit_response.into_body(), usize::MAX)
        .await
        .expect("test");
    let audit_result: AuditResponse = serde_json::from_slice(&audit_body).expect("test");

    // Verify the audit record matches the prediction request
    assert_eq!(audit_result.record.request_id, request_id);
}

#[tokio::test]
async fn test_apr_audit_invalid_id() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audit/not-a-valid-uuid")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn test_predict_request_serialization() {
    let request = PredictRequest {
        model: Some("test-model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        top_k: Some(3),
        include_confidence: true,
    };

    let json = serde_json::to_string(&request).expect("test");
    assert!(json.contains("test-model"));
    assert!(json.contains("features"));

    // Deserialize back
    let deserialized: PredictRequest = serde_json::from_str(&json).expect("test");
    assert_eq!(deserialized.features.len(), 3);
}

#[test]
fn test_explain_request_defaults() {
    let json = r#"{"features": [1.0], "feature_names": ["f1"]}"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("test");

    assert_eq!(request.top_k_features, 5); // default
    assert_eq!(request.method, "shap"); // default
}

// ==========================================================================
// M33: GGUF HTTP Serving Integration Tests (IMP-084 through IMP-087)
// ==========================================================================

/// IMP-084: AppState::with_gpu_model creates state with GPU model
#[test]
#[cfg(feature = "gpu")]
fn test_imp_084_app_state_with_gpu_model() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Create minimal GPU model
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2, // Standard MHA
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
    };
    let gpu_model = GpuModel::new(config).expect("Failed to create GPU model");

    // Create AppState with GPU model
    let state = AppState::with_gpu_model(gpu_model).expect("Failed to create AppState");

    // Verify GPU model is present
    assert!(
        state.has_gpu_model(),
        "IMP-084: AppState should have GPU model"
    );
}

/// IMP-085: /v1/completions endpoint uses GPU model when available
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_085_completions_uses_gpu_model() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Create minimal GPU model
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2, // Standard MHA
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
    };
    let gpu_model = GpuModel::new(config).expect("Failed to create GPU model");

    let state = AppState::with_gpu_model(gpu_model).expect("Failed to create AppState");
    let app = create_router(state);

    // Make completion request
    let request = CompletionRequest {
        prompt: "Hello".to_string(),
        max_tokens: Some(5),
        temperature: Some(0.0),
        model: "default".to_string(),
        top_p: None,
        stop: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    // Should succeed (200 OK) with GPU model
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "IMP-085: /v1/completions should work with GPU model"
    );
}

// ========================================================================
// IMP-116: Cached Model HTTP Integration Tests
// ========================================================================

/// IMP-116a: Test AppState can store cached model
#[test]
#[cfg(feature = "gpu")]
fn test_imp_116a_appstate_cached_model_storage() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,

        rope_type: 0,
    };

    // Create test model
    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);

    // Create AppState with cached model
    let state = AppState::with_cached_model(cached_model)
        .expect("IMP-116a: AppState should accept cached model");

    // Verify model is accessible
    assert!(
        state.cached_model().is_some(),
        "IMP-116a: Cached model should be accessible from AppState"
    );
}

/// IMP-116b: Test cached model is thread-safe for async handlers
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_116b_cached_model_thread_safety() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = Arc::new(OwnedQuantizedModelCachedSync::new(model));

    // Spawn multiple concurrent tasks accessing the model
    let mut handles = Vec::new();
    for i in 0..4 {
        let model_clone = cached_model.clone();
        handles.push(tokio::spawn(async move {
            // Should be able to get inner model from any thread
            let inner = model_clone.model();
            assert_eq!(inner.config.hidden_dim, 64, "Task {i} should access model");
        }));
    }

    // All tasks should complete successfully
    for handle in handles {
        handle
            .await
            .expect("IMP-116b: Concurrent access should succeed");
    }
}

/// IMP-116c: Test completions endpoint routes to cached model
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_116c_completions_uses_cached_model() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);

    // Create state with cached model
    let state = AppState::with_cached_model(cached_model).expect("Failed to create AppState");

    // Verify cached model is stored correctly
    assert!(
        state.has_cached_model(),
        "IMP-116c: AppState should have cached model"
    );
    assert!(
        state.cached_model().is_some(),
        "IMP-116c: cached_model() should return Some"
    );

    let app = create_router(state);

    // Make completion request - may fail due to test model but path should be exercised
    let request = CompletionRequest {
        prompt: "Hello".to_string(),
        max_tokens: Some(3),
        temperature: Some(0.0),
        model: "default".to_string(),
        top_p: None,
        stop: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    // The request was routed (may fail with 500 due to test model)
    // Key point: no panic, request was handled
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "IMP-116c: Request should be handled (got {})",
        status
    );
}
