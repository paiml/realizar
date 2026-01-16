//! EXTREME TDD coverage tests for realizar/src/api.rs
//!
//! This module provides comprehensive coverage for API request/response structures,
//! handler functions, validation logic, error responses, and streaming code.
//!
//! Coverage targets:
//! - Request/Response struct serialization/deserialization
//! - Default value functions
//! - Error response generation
//! - Context window management
//! - OpenAI-compatible API types
//! - GPU batch inference types
//! - APR prediction types

use realizar::api::{
    BatchGenerateRequest, BatchGenerateResponse, BatchTokenizeRequest, BatchTokenizeResponse,
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatDelta, ChatMessage,
    CompletionChoice, CompletionRequest, CompletionResponse, ContextWindowConfig,
    ContextWindowManager, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    ErrorResponse, ExplainRequest, ExplainResponse, GenerateRequest, GenerateResponse,
    GpuBatchRequest, GpuBatchResponse, GpuBatchResult, GpuBatchStats, GpuStatusResponse,
    GpuWarmupResponse, HealthResponse, ModelLineage, ModelMetadataResponse, ModelsResponse,
    OpenAIModel, OpenAIModelsResponse, PredictRequest, PredictResponse, PredictionWithScore,
    ReloadRequest, ReloadResponse, ServerMetricsResponse, StreamDoneEvent, StreamTokenEvent,
    TokenizeRequest, TokenizeResponse, Usage,
};
use realizar::explain::ShapExplanation;
use realizar::registry::ModelInfo;

// ============================================================================
// Test 1-5: Request/Response Struct Serialization
// ============================================================================

#[test]
fn test_health_response_serialization() {
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""status":"healthy""#));
    assert!(json.contains(r#""version":"1.0.0""#));

    let deserialized: HealthResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.status, "healthy");
    assert_eq!(deserialized.version, "1.0.0");
}

#[test]
fn test_tokenize_request_with_model_id() {
    let request = TokenizeRequest {
        text: "hello world".to_string(),
        model_id: Some("test-model".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""text":"hello world""#));
    assert!(json.contains(r#""model_id":"test-model""#));

    let deserialized: TokenizeRequest = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.text, "hello world");
    assert_eq!(deserialized.model_id, Some("test-model".to_string()));
}

#[test]
fn test_tokenize_request_without_model_id() {
    let json = r#"{"text": "test input"}"#;
    let request: TokenizeRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.text, "test input");
    assert!(request.model_id.is_none());
}

#[test]
fn test_tokenize_response_serialization() {
    let response = TokenizeResponse {
        token_ids: vec![1, 2, 3, 4, 5],
        num_tokens: 5,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""token_ids":[1,2,3,4,5]"#));
    assert!(json.contains(r#""num_tokens":5"#));
}

#[test]
fn test_generate_response_serialization() {
    let response = GenerateResponse {
        token_ids: vec![10, 20, 30],
        text: "generated text".to_string(),
        num_generated: 3,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""token_ids":[10,20,30]"#));
    assert!(json.contains(r#""text":"generated text""#));
    assert!(json.contains(r#""num_generated":3"#));
}

// ============================================================================
// Test 6-10: Generate Request Default Values
// ============================================================================

#[test]
fn test_generate_request_all_defaults() {
    let json = r#"{"prompt": "hello"}"#;
    let request: GenerateRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.prompt, "hello");
    assert_eq!(request.max_tokens, 50);
    assert!((request.temperature - 1.0).abs() < 1e-6);
    assert_eq!(request.strategy, "greedy");
    assert_eq!(request.top_k, 50);
    assert!((request.top_p - 0.9).abs() < 1e-6);
    assert!(request.seed.is_none());
    assert!(request.model_id.is_none());
}

#[test]
fn test_generate_request_custom_values() {
    let json = r#"{
        "prompt": "test prompt",
        "max_tokens": 100,
        "temperature": 0.7,
        "strategy": "top_k",
        "top_k": 10,
        "top_p": 0.95,
        "seed": 42,
        "model_id": "custom-model"
    }"#;

    let request: GenerateRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.prompt, "test prompt");
    assert_eq!(request.max_tokens, 100);
    assert!((request.temperature - 0.7).abs() < 1e-6);
    assert_eq!(request.strategy, "top_k");
    assert_eq!(request.top_k, 10);
    assert!((request.top_p - 0.95).abs() < 1e-6);
    assert_eq!(request.seed, Some(42));
    assert_eq!(request.model_id, Some("custom-model".to_string()));
}

#[test]
fn test_error_response_serialization() {
    let response = ErrorResponse {
        error: "Something went wrong".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""error":"Something went wrong""#));
}

#[test]
fn test_batch_tokenize_request_serialization() {
    let request = BatchTokenizeRequest {
        texts: vec!["text1".to_string(), "text2".to_string()],
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""texts":["text1","text2"]"#));

    let deserialized: BatchTokenizeRequest =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.texts.len(), 2);
}

#[test]
fn test_batch_tokenize_response_serialization() {
    let response = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1, 2],
                num_tokens: 2,
            },
            TokenizeResponse {
                token_ids: vec![3, 4, 5],
                num_tokens: 3,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: BatchTokenizeResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.results.len(), 2);
    assert_eq!(deserialized.results[0].num_tokens, 2);
    assert_eq!(deserialized.results[1].num_tokens, 3);
}

// ============================================================================
// Test 11-15: Batch Generate Request/Response
// ============================================================================

#[test]
fn test_batch_generate_request_defaults() {
    let json = r#"{"prompts": ["prompt1", "prompt2"]}"#;
    let request: BatchGenerateRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.prompts.len(), 2);
    assert_eq!(request.max_tokens, 50);
    assert!((request.temperature - 1.0).abs() < 1e-6);
    assert_eq!(request.strategy, "greedy");
    assert_eq!(request.top_k, 50);
    assert!((request.top_p - 0.9).abs() < 1e-6);
    assert!(request.seed.is_none());
}

#[test]
fn test_batch_generate_response_serialization() {
    let response = BatchGenerateResponse {
        results: vec![
            GenerateResponse {
                token_ids: vec![1, 2, 3],
                text: "output1".to_string(),
                num_generated: 3,
            },
            GenerateResponse {
                token_ids: vec![4, 5],
                text: "output2".to_string(),
                num_generated: 2,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: BatchGenerateResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.results.len(), 2);
}

#[test]
fn test_stream_token_event_serialization() {
    let event = StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };

    let json = serde_json::to_string(&event).expect("should serialize");
    assert!(json.contains(r#""token_id":42"#));
    assert!(json.contains(r#""text":"hello""#));

    let deserialized: StreamTokenEvent = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.token_id, 42);
    assert_eq!(deserialized.text, "hello");
}

#[test]
fn test_stream_done_event_serialization() {
    let event = StreamDoneEvent { num_generated: 100 };

    let json = serde_json::to_string(&event).expect("should serialize");
    assert!(json.contains(r#""num_generated":100"#));

    let deserialized: StreamDoneEvent = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.num_generated, 100);
}

#[test]
fn test_models_response_serialization() {
    let response = ModelsResponse {
        models: vec![
            ModelInfo {
                id: "model-1".to_string(),
                name: "Test Model 1".to_string(),
                description: "First test model".to_string(),
                format: "gguf".to_string(),
                loaded: true,
            },
            ModelInfo {
                id: "model-2".to_string(),
                name: "Test Model 2".to_string(),
                description: "Second test model".to_string(),
                format: "apr".to_string(),
                loaded: false,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ModelsResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.models.len(), 2);
    assert_eq!(deserialized.models[0].id, "model-1");
    assert!(deserialized.models[0].loaded);
    assert!(!deserialized.models[1].loaded);
}

// ============================================================================
// Test 16-20: OpenAI-Compatible API Types
// ============================================================================

#[test]
fn test_chat_completion_request_minimal() {
    let json = r#"{
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }"#;

    let request: ChatCompletionRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.model, "gpt-4");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.n, 1); // default
    assert!(!request.stream); // default
}

#[test]
fn test_chat_completion_request_full() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
                name: Some("user123".to_string()),
            },
        ],
        max_tokens: Some(256),
        temperature: Some(0.7),
        top_p: Some(0.95),
        n: 2,
        stream: true,
        stop: Some(vec!["STOP".to_string()]),
        user: Some("user-id".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: ChatCompletionRequest =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.model, "gpt-4");
    assert_eq!(deserialized.messages.len(), 2);
    assert_eq!(deserialized.max_tokens, Some(256));
    assert_eq!(deserialized.n, 2);
    assert!(deserialized.stream);
}

#[test]
fn test_chat_completion_response_serialization() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "gpt-4".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello! How can I help?".to_string(),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""id":"chatcmpl-123""#));
    assert!(json.contains(r#""object":"chat.completion""#));
    assert!(json.contains(r#""finish_reason":"stop""#));
}

#[test]
fn test_openai_models_response_serialization() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![
            OpenAIModel {
                id: "model-1".to_string(),
                object: "model".to_string(),
                created: 1000000000,
                owned_by: "realizar".to_string(),
            },
            OpenAIModel {
                id: "model-2".to_string(),
                object: "model".to_string(),
                created: 1000000001,
                owned_by: "realizar".to_string(),
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: OpenAIModelsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.object, "list");
    assert_eq!(deserialized.data.len(), 2);
    assert_eq!(deserialized.data[0].owned_by, "realizar");
}

#[test]
fn test_chat_delta_skip_none_serialization() {
    // With only content
    let delta1 = ChatDelta {
        role: None,
        content: Some("hello".to_string()),
    };
    let json1 = serde_json::to_string(&delta1).expect("should serialize");
    assert!(!json1.contains("role"));
    assert!(json1.contains(r#""content":"hello""#));

    // With only role
    let delta2 = ChatDelta {
        role: Some("assistant".to_string()),
        content: None,
    };
    let json2 = serde_json::to_string(&delta2).expect("should serialize");
    assert!(json2.contains(r#""role":"assistant""#));
    assert!(!json2.contains("content"));
}

// ============================================================================
// Test 21-25: Context Window Management
// ============================================================================

#[test]
fn test_context_window_config_builder() {
    let config = ContextWindowConfig::new(8192).with_reserved_output(512);

    assert_eq!(config.max_tokens, 8192);
    assert_eq!(config.reserved_output_tokens, 512);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_manager_no_truncation() {
    let manager = ContextWindowManager::default_manager();

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        },
    ];

    assert!(!manager.needs_truncation(&messages));

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_context_window_manager_forces_truncation() {
    let config = ContextWindowConfig::new(50).with_reserved_output(10);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(200), // Very long message
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "short".to_string(),
            name: None,
        },
    ];

    assert!(manager.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_token_estimation() {
    let manager = ContextWindowManager::default_manager();

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello world! This is a test.".to_string(),
        name: None,
    }];

    let tokens = manager.estimate_total_tokens(&messages);
    // Estimate: ~4 chars per token + 10 overhead = ~17 tokens
    assert!(tokens > 0);
    assert!(tokens < 50);
}

#[test]
fn test_context_window_preserves_recent_messages() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Old message that should be dropped".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(100),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&messages);

    if truncated {
        // Most recent message should be preserved
        let has_recent = result.iter().any(|m| m.content == "Recent");
        assert!(
            has_recent || result.is_empty(),
            "Recent message should be preserved when possible"
        );
    }
}

// ============================================================================
// Test 26-30: GPU Batch and APR Types
// ============================================================================

#[test]
fn test_gpu_batch_request_defaults() {
    let json = r#"{"prompts": ["hello", "world"]}"#;
    let request: GpuBatchRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.prompts.len(), 2);
    assert_eq!(request.max_tokens, 50);
    assert!((request.temperature - 0.0).abs() < 1e-6);
    assert_eq!(request.top_k, 50);
    assert!(request.stop.is_empty());
}

#[test]
fn test_gpu_batch_response_serialization() {
    let response = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "output".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: true,
            total_tokens: 3,
            processing_time_ms: 10.5,
            throughput_tps: 285.7,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""gpu_used":true"#));
    assert!(json.contains(r#""throughput_tps"#));

    let deserialized: GpuBatchResponse = serde_json::from_str(&json).expect("should deserialize");
    assert!(deserialized.stats.gpu_used);
    assert_eq!(deserialized.results.len(), 1);
}

#[test]
fn test_gpu_warmup_response_serialization() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 1024 * 1024 * 512,
        num_layers: 32,
        message: "Warmup complete".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuWarmupResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.success);
    assert_eq!(deserialized.num_layers, 32);
}

#[test]
fn test_gpu_status_response_serialization() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1024 * 1024 * 100,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuStatusResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.cache_ready);
    assert_eq!(deserialized.batch_threshold, 32);
}

#[test]
fn test_predict_request_full_serialization() {
    let request = PredictRequest {
        model: Some("my-model".to_string()),
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: Some(vec![
            "f1".to_string(),
            "f2".to_string(),
            "f3".to_string(),
            "f4".to_string(),
        ]),
        top_k: Some(3),
        include_confidence: true,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: PredictRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.features.len(), 4);
    assert_eq!(deserialized.top_k, Some(3));
    assert!(deserialized.include_confidence);
}

// ============================================================================
// Test 31-35: APR Explain and Audit Types
// ============================================================================

#[test]
fn test_explain_request_defaults() {
    let json = r#"{"features": [1.0, 2.0], "feature_names": ["a", "b"]}"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.top_k_features, 5); // default
    assert_eq!(request.method, "shap"); // default
}

#[test]
fn test_explain_response_serialization() {
    let response = ExplainResponse {
        request_id: "req-123".to_string(),
        model: "test-model".to_string(),
        prediction: serde_json::json!(0.85),
        confidence: Some(0.92),
        explanation: ShapExplanation {
            base_value: 0.5,
            shap_values: vec![0.1, -0.05, 0.3],
            feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
            prediction: 0.85,
        },
        summary: "Top features: f3 (+), f1 (+)".to_string(),
        latency_ms: 15.5,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""request_id":"req-123""#));
    assert!(json.contains(r#""summary"#));

    let deserialized: ExplainResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.explanation.shap_values.len(), 3);
}

#[test]
fn test_prediction_with_score_serialization() {
    let prediction = PredictionWithScore {
        label: "class_0".to_string(),
        score: 0.95,
    };

    let json = serde_json::to_string(&prediction).expect("should serialize");
    assert!(json.contains(r#""label":"class_0""#));
    assert!(json.contains(r#""score":0.95"#));
}

#[test]
fn test_predict_response_regression() {
    let response = PredictResponse {
        request_id: "uuid-123".to_string(),
        model: "regressor".to_string(),
        prediction: serde_json::json!(42.5),
        confidence: Some(1.0),
        top_k_predictions: None,
        latency_ms: 5.2,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: PredictResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.prediction, serde_json::json!(42.5));
    assert!(deserialized.top_k_predictions.is_none());
}

#[test]
fn test_predict_response_classification() {
    let response = PredictResponse {
        request_id: "uuid-456".to_string(),
        model: "classifier".to_string(),
        prediction: serde_json::json!("class_2"),
        confidence: Some(0.85),
        top_k_predictions: Some(vec![
            PredictionWithScore {
                label: "class_2".to_string(),
                score: 0.85,
            },
            PredictionWithScore {
                label: "class_1".to_string(),
                score: 0.10,
            },
            PredictionWithScore {
                label: "class_0".to_string(),
                score: 0.05,
            },
        ]),
        latency_ms: 8.3,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: PredictResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.top_k_predictions.is_some());
    let top_k = deserialized.top_k_predictions.unwrap();
    assert_eq!(top_k.len(), 3);
    assert_eq!(top_k[0].label, "class_2");
}

// ============================================================================
// Test 36-40: Embedding and Completion Types
// ============================================================================

#[test]
fn test_embedding_request_serialization() {
    let request = EmbeddingRequest {
        input: "test input text".to_string(),
        model: Some("text-embedding-ada".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: EmbeddingRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.input, "test input text");
    assert_eq!(deserialized.model, Some("text-embedding-ada".to_string()));
}

#[test]
fn test_embedding_response_serialization() {
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        }],
        model: "text-embedding".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: EmbeddingResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.data[0].embedding.len(), 4);
    assert_eq!(deserialized.usage.prompt_tokens, 5);
}

#[test]
fn test_completion_request_serialization() {
    let request = CompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        prompt: "Once upon a time".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: Some(vec!["END".to_string()]),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: CompletionRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.prompt, "Once upon a time");
    assert_eq!(deserialized.max_tokens, Some(50));
}

#[test]
fn test_completion_response_serialization() {
    let response = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: 1234567890,
        model: "gpt-3.5".to_string(),
        choices: vec![CompletionChoice {
            text: "there was a princess".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 4,
            completion_tokens: 5,
            total_tokens: 9,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: CompletionResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.choices[0].text, "there was a princess");
    assert_eq!(deserialized.usage.total_tokens, 9);
}

#[test]
fn test_completion_choice_with_logprobs() {
    let choice = CompletionChoice {
        text: "output".to_string(),
        index: 0,
        logprobs: Some(serde_json::json!({
            "tokens": ["out", "put"],
            "token_logprobs": [-0.5, -0.3]
        })),
        finish_reason: "length".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains("logprobs"));
    assert!(json.contains("token_logprobs"));
}

// ============================================================================
// Test 41-45: Model Metadata and Reload Types
// ============================================================================

#[test]
fn test_model_metadata_response_full() {
    let response = ModelMetadataResponse {
        id: "phi-2".to_string(),
        name: "Phi-2".to_string(),
        format: "gguf".to_string(),
        size_bytes: 2_700_000_000,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 2048,
        lineage: Some(ModelLineage {
            uri: "pacha://phi-2:latest".to_string(),
            version: "1.0.0".to_string(),
            recipe: Some("instruct-tuned".to_string()),
            parent: Some("phi-1.5".to_string()),
            content_hash: "blake3:abc123".to_string(),
        }),
        loaded: true,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ModelMetadataResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.format, "gguf");
    assert!(deserialized.lineage.is_some());
    let lineage = deserialized.lineage.unwrap();
    assert_eq!(lineage.parent, Some("phi-1.5".to_string()));
}

#[test]
fn test_model_metadata_response_minimal() {
    let response = ModelMetadataResponse {
        id: "simple".to_string(),
        name: "Simple Model".to_string(),
        format: "apr".to_string(),
        size_bytes: 100_000,
        quantization: None,
        context_length: 512,
        lineage: None,
        loaded: false,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(!json.contains("quantization"));
    assert!(!json.contains("lineage"));
}

#[test]
fn test_reload_request_serialization() {
    let request = ReloadRequest {
        model: Some("model-1".to_string()),
        path: Some("/path/to/model.gguf".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: ReloadRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.model, Some("model-1".to_string()));
    assert_eq!(deserialized.path, Some("/path/to/model.gguf".to_string()));
}

#[test]
fn test_reload_response_serialization() {
    let response = ReloadResponse {
        success: true,
        message: "Model reloaded successfully".to_string(),
        reload_time_ms: 1500,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ReloadResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.success);
    assert_eq!(deserialized.reload_time_ms, 1500);
}

#[test]
fn test_server_metrics_response_serialization() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: 150.5,
        latency_p50_ms: 10.2,
        latency_p95_ms: 25.8,
        latency_p99_ms: 50.1,
        gpu_memory_used_bytes: 8_000_000_000,
        gpu_memory_total_bytes: 24_000_000_000,
        gpu_utilization_percent: 75,
        cuda_path_active: true,
        batch_size: 32,
        queue_depth: 5,
        total_tokens: 100_000,
        total_requests: 500,
        uptime_secs: 3600,
        model_name: "phi-2-q4k".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ServerMetricsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert!((deserialized.throughput_tok_per_sec - 150.5).abs() < 0.01);
    assert!(deserialized.cuda_path_active);
    assert_eq!(deserialized.model_name, "phi-2-q4k");
}

// ============================================================================
// Test 46-50: Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn test_empty_messages_array() {
    let json = r#"{"model": "test", "messages": []}"#;
    let request: ChatCompletionRequest = serde_json::from_str(json).expect("should deserialize");
    assert!(request.messages.is_empty());
}

#[test]
fn test_large_token_ids() {
    let response = TokenizeResponse {
        token_ids: vec![u32::MAX, u32::MAX - 1, 0],
        num_tokens: 3,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: TokenizeResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.token_ids[0], u32::MAX);
    assert_eq!(deserialized.token_ids[1], u32::MAX - 1);
}

#[test]
fn test_zero_latency() {
    let response = PredictResponse {
        request_id: "test".to_string(),
        model: "test".to_string(),
        prediction: serde_json::json!(1.0),
        confidence: Some(1.0),
        top_k_predictions: None,
        latency_ms: 0.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""latency_ms":0.0"#));
}

#[test]
fn test_unicode_in_messages() {
    let message = ChatMessage {
        role: "user".to_string(),
        content: "Hello World".to_string(),
        name: Some("User".to_string()),
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    let deserialized: ChatMessage = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.content.contains("Hello"));
    assert!(deserialized.content.contains("World"));
}

#[test]
fn test_special_characters_in_text() {
    let request = TokenizeRequest {
        text: r#"Test with "quotes" and \backslashes\ and newlines"#.to_string(),
        model_id: None,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: TokenizeRequest = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.text.contains("quotes"));
    assert!(deserialized.text.contains("backslashes"));
}
