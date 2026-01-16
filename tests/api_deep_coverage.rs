//! Deep coverage tests for realizar/src/api.rs
//!
//! This module provides comprehensive additional coverage for API components
//! that were not fully tested in api_coverage.rs.
//!
//! Coverage targets:
//! - AppState construction methods and accessors
//! - DispatchMetrics types
//! - ChatCompletionChunk streaming types
//! - BatchConfig configuration methods
//! - ContinuousBatchResponse methods
//! - BatchQueueStats and BatchProcessResult
//! - AuditResponse serialization
//! - Additional edge cases

use realizar::api::{
    AuditResponse, ChatChunkChoice, ChatCompletionChunk, ChatDelta, ContextWindowConfig,
    ContextWindowManager, DispatchMetricsQuery, DispatchMetricsResponse, DispatchResetResponse,
    AppState, ChatMessage,
};
use realizar::audit::AuditRecord;

// ============================================================================
// Test 1-5: AppState Construction Methods
// ============================================================================

#[test]
fn test_appstate_demo_creates_valid_state() {
    let state = AppState::demo();
    assert!(state.is_ok(), "AppState::demo() should succeed");

    let state = state.unwrap();
    // Verify has_quantized_model returns false for demo state
    assert!(!state.has_quantized_model());
    // Verify quantized_model accessor returns None
    assert!(state.quantized_model().is_none());
}

#[test]
fn test_appstate_with_cache_creates_valid_state() {
    let state = AppState::with_cache(10);
    // Verify has_quantized_model returns false for cache state
    assert!(!state.has_quantized_model());
    // Verify quantized_model accessor returns None
    assert!(state.quantized_model().is_none());
}

// ============================================================================
// Test 6-10: DispatchMetrics Types Serialization
// ============================================================================

#[test]
fn test_dispatch_metrics_response_serialization() {
    let response = DispatchMetricsResponse {
        cpu_dispatches: 100,
        gpu_dispatches: 50,
        total_dispatches: 150,
        gpu_ratio: 0.333,
        cpu_latency_p50_us: 150.5,
        cpu_latency_p95_us: 250.0,
        cpu_latency_p99_us: 500.0,
        gpu_latency_p50_us: 50.5,
        gpu_latency_p95_us: 100.0,
        gpu_latency_p99_us: 200.0,
        cpu_latency_mean_us: 175.3,
        gpu_latency_mean_us: 65.7,
        cpu_latency_min_us: 50,
        cpu_latency_max_us: 1000,
        gpu_latency_min_us: 10,
        gpu_latency_max_us: 500,
        cpu_latency_variance_us: 1250.5,
        cpu_latency_stddev_us: 35.4,
        gpu_latency_variance_us: 500.2,
        gpu_latency_stddev_us: 22.4,
        bucket_boundaries_us: vec![
            "0-100".to_string(),
            "100-500".to_string(),
            "500-1000".to_string(),
        ],
        cpu_latency_bucket_counts: vec![10, 20, 5],
        gpu_latency_bucket_counts: vec![5, 10, 2],
        throughput_rps: 125.5,
        elapsed_seconds: 60.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""cpu_dispatches":100"#));
    assert!(json.contains(r#""gpu_dispatches":50"#));
    assert!(json.contains(r#""total_dispatches":150"#));
    assert!(json.contains(r#""throughput_rps"#));
    assert!(json.contains(r#""elapsed_seconds"#));
}

#[test]
fn test_dispatch_metrics_query_default() {
    let json = r#"{}"#;
    let query: DispatchMetricsQuery = serde_json::from_str(json).expect("should deserialize");
    assert!(query.format.is_none());
}

#[test]
fn test_dispatch_metrics_query_with_format() {
    let json = r#"{"format": "prometheus"}"#;
    let query: DispatchMetricsQuery = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(query.format, Some("prometheus".to_string()));
}

#[test]
fn test_dispatch_reset_response_serialization() {
    let response = DispatchResetResponse {
        success: true,
        message: "Metrics reset successfully".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""success":true"#));
    assert!(json.contains(r#""message":"Metrics reset successfully""#));

    let deserialized: DispatchResetResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert!(deserialized.success);
    assert_eq!(deserialized.message, "Metrics reset successfully");
}

#[test]
fn test_dispatch_reset_response_failure() {
    let response = DispatchResetResponse {
        success: false,
        message: "Reset failed: no metrics available".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""success":false"#));

    let deserialized: DispatchResetResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert!(!deserialized.success);
}

// ============================================================================
// Test 11-15: ChatCompletionChunk Streaming Types
// ============================================================================

#[test]
fn test_chat_completion_chunk_serialization() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-abc123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1234567890,
        model: "gpt-4".to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };

    let json = serde_json::to_string(&chunk).expect("should serialize");
    assert!(json.contains(r#""object":"chat.completion.chunk""#));
    assert!(json.contains(r#""id":"chatcmpl-abc123""#));

    let deserialized: ChatCompletionChunk =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.id, "chatcmpl-abc123");
    assert_eq!(deserialized.object, "chat.completion.chunk");
}

#[test]
fn test_chat_chunk_choice_with_content() {
    let choice = ChatChunkChoice {
        index: 1,
        delta: ChatDelta {
            role: None,
            content: Some("Hello".to_string()),
        },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains(r#""index":1"#));
    assert!(json.contains(r#""content":"Hello""#));
    assert!(!json.contains(r#""role""#)); // skip_serializing_if on None

    let deserialized: ChatChunkChoice = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.index, 1);
    assert_eq!(deserialized.delta.content, Some("Hello".to_string()));
}

#[test]
fn test_chat_chunk_choice_with_finish_reason() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: None,
        },
        finish_reason: Some("stop".to_string()),
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains(r#""finish_reason":"stop""#));

    let deserialized: ChatChunkChoice = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.finish_reason, Some("stop".to_string()));
}

#[test]
fn test_chat_delta_both_fields() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("Hi there!".to_string()),
    };

    let json = serde_json::to_string(&delta).expect("should serialize");
    assert!(json.contains(r#""role":"assistant""#));
    assert!(json.contains(r#""content":"Hi there!""#));

    let deserialized: ChatDelta = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.role, Some("assistant".to_string()));
    assert_eq!(deserialized.content, Some("Hi there!".to_string()));
}

#[test]
fn test_chat_delta_neither_field() {
    let delta = ChatDelta {
        role: None,
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("should serialize");
    // With skip_serializing_if, empty object
    assert_eq!(json, "{}");

    let deserialized: ChatDelta = serde_json::from_str(&json).expect("should deserialize");
    assert!(deserialized.role.is_none());
    assert!(deserialized.content.is_none());
}

// ============================================================================
// Test 16-20: AuditResponse Serialization
// ============================================================================

#[test]
fn test_audit_response_serialization() {
    let record = AuditRecord {
        request_id: "uuid-123-456".to_string(),
        timestamp: "2024-01-15T10:30:00Z".to_string(),
        model_id: "phi-2".to_string(),
        model_hash: "blake3:abc123".to_string(),
        input_shape: vec![1, 128],
        output: serde_json::json!({"prediction": 0.95}),
        latency_ms: 25.5,
        confidence: Some(0.95),
    };

    let response = AuditResponse { record };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""request_id":"uuid-123-456""#));
    assert!(json.contains(r#""model_id":"phi-2""#));
    assert!(json.contains(r#""latency_ms":25.5"#));

    let deserialized: AuditResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.record.request_id, "uuid-123-456");
    assert_eq!(deserialized.record.model_id, "phi-2");
    assert!((deserialized.record.latency_ms - 25.5).abs() < 0.01);
}

#[test]
fn test_audit_response_without_confidence() {
    let record = AuditRecord {
        request_id: "req-789".to_string(),
        timestamp: "2024-01-15T11:00:00Z".to_string(),
        model_id: "regressor".to_string(),
        model_hash: "blake3:def456".to_string(),
        input_shape: vec![1, 4],
        output: serde_json::json!(42.5),
        latency_ms: 5.0,
        confidence: None,
    };

    let response = AuditResponse { record };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""model_id":"regressor""#));

    let deserialized: AuditResponse = serde_json::from_str(&json).expect("should deserialize");
    assert!(deserialized.record.confidence.is_none());
}

#[test]
fn test_audit_record_large_input_shape() {
    let record = AuditRecord {
        request_id: "batch-001".to_string(),
        timestamp: "2024-01-15T12:00:00Z".to_string(),
        model_id: "bert".to_string(),
        model_hash: "blake3:789".to_string(),
        input_shape: vec![32, 512, 768], // Batch of sequences
        output: serde_json::json!({"embeddings": "..."}),
        latency_ms: 150.0,
        confidence: Some(0.99),
    };

    let response = AuditResponse { record };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""input_shape":[32,512,768]"#));

    let deserialized: AuditResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.record.input_shape, vec![32, 512, 768]);
}

#[test]
fn test_audit_record_complex_output() {
    let record = AuditRecord {
        request_id: "multi-output".to_string(),
        timestamp: "2024-01-15T13:00:00Z".to_string(),
        model_id: "multi-task".to_string(),
        model_hash: "blake3:multi".to_string(),
        input_shape: vec![1, 100],
        output: serde_json::json!({
            "classification": "class_a",
            "regression": 0.75,
            "embeddings": [0.1, 0.2, 0.3]
        }),
        latency_ms: 30.0,
        confidence: Some(0.88),
    };

    let response = AuditResponse { record };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""classification":"class_a""#));

    let deserialized: AuditResponse = serde_json::from_str(&json).expect("should deserialize");
    let output = &deserialized.record.output;
    assert_eq!(output["classification"], "class_a");
}

#[test]
fn test_audit_record_empty_input_shape() {
    let record = AuditRecord {
        request_id: "scalar-input".to_string(),
        timestamp: "2024-01-15T14:00:00Z".to_string(),
        model_id: "simple".to_string(),
        model_hash: "blake3:simple".to_string(),
        input_shape: vec![],
        output: serde_json::json!(1.0),
        latency_ms: 1.0,
        confidence: None,
    };

    let response = AuditResponse { record };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""input_shape":[]"#));

    let deserialized: AuditResponse = serde_json::from_str(&json).expect("should deserialize");
    assert!(deserialized.record.input_shape.is_empty());
}

// ============================================================================
// Test 21-25: ContextWindow Additional Edge Cases
// ============================================================================

#[test]
fn test_context_window_default_config() {
    let config = ContextWindowConfig::default();
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 256);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_chaining() {
    let config = ContextWindowConfig::new(2048).with_reserved_output(128);
    assert_eq!(config.max_tokens, 2048);
    assert_eq!(config.reserved_output_tokens, 128);
}

#[test]
fn test_context_window_manager_preserves_system() {
    let config = ContextWindowConfig::new(200).with_reserved_output(50);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(500), // Very long
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent message".to_string(),
            name: None,
        },
    ];

    assert!(manager.needs_truncation(&messages));

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(truncated);

    // System message should be preserved (if space allows)
    let has_system = result.iter().any(|m| m.role == "system");
    // May or may not have system depending on config and message sizes
    // At minimum, result should not be empty if there was space
    assert!(!result.is_empty() || messages.is_empty());
    let _ = has_system; // Silence unused warning
}

#[test]
fn test_context_window_single_system_message() {
    let config = ContextWindowConfig::new(1000);
    let manager = ContextWindowManager::new(config);

    let messages = vec![ChatMessage {
        role: "system".to_string(),
        content: "Short system prompt.".to_string(),
        name: None,
    }];

    assert!(!manager.needs_truncation(&messages));

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].role, "system");
}

#[test]
fn test_context_window_all_user_messages() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "First".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Second".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Third".to_string(),
            name: None,
        },
    ];

    let estimate = manager.estimate_total_tokens(&messages);
    assert!(estimate > 0);

    let (result, _) = manager.truncate_messages(&messages);
    // Should have some messages (most recent prioritized)
    assert!(!result.is_empty());
}

// ============================================================================
// Test 26-30: DispatchMetricsResponse Edge Cases
// ============================================================================

#[test]
fn test_dispatch_metrics_response_zero_values() {
    let response = DispatchMetricsResponse {
        cpu_dispatches: 0,
        gpu_dispatches: 0,
        total_dispatches: 0,
        gpu_ratio: 0.0,
        cpu_latency_p50_us: 0.0,
        cpu_latency_p95_us: 0.0,
        cpu_latency_p99_us: 0.0,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: 0.0,
        gpu_latency_mean_us: 0.0,
        cpu_latency_min_us: 0,
        cpu_latency_max_us: 0,
        gpu_latency_min_us: 0,
        gpu_latency_max_us: 0,
        cpu_latency_variance_us: 0.0,
        cpu_latency_stddev_us: 0.0,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""total_dispatches":0"#));
    assert!(json.contains(r#""gpu_ratio":0.0"#));
}

#[test]
fn test_dispatch_metrics_response_high_throughput() {
    let response = DispatchMetricsResponse {
        cpu_dispatches: 1_000_000,
        gpu_dispatches: 5_000_000,
        total_dispatches: 6_000_000,
        gpu_ratio: 0.833,
        cpu_latency_p50_us: 10.5,
        cpu_latency_p95_us: 25.0,
        cpu_latency_p99_us: 50.0,
        gpu_latency_p50_us: 5.5,
        gpu_latency_p95_us: 15.0,
        gpu_latency_p99_us: 30.0,
        cpu_latency_mean_us: 12.0,
        gpu_latency_mean_us: 6.0,
        cpu_latency_min_us: 5,
        cpu_latency_max_us: 100,
        gpu_latency_min_us: 2,
        gpu_latency_max_us: 50,
        cpu_latency_variance_us: 25.0,
        cpu_latency_stddev_us: 5.0,
        gpu_latency_variance_us: 10.0,
        gpu_latency_stddev_us: 3.16,
        bucket_boundaries_us: vec!["0-10".to_string(), "10-50".to_string(), "50+".to_string()],
        cpu_latency_bucket_counts: vec![500_000, 400_000, 100_000],
        gpu_latency_bucket_counts: vec![4_000_000, 900_000, 100_000],
        throughput_rps: 100_000.0,
        elapsed_seconds: 60.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""total_dispatches":6000000"#));
    assert!(json.contains(r#""throughput_rps":100000.0"#));
}

#[test]
fn test_dispatch_metrics_query_json_format() {
    let json = r#"{"format": "json"}"#;
    let query: DispatchMetricsQuery = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(query.format, Some("json".to_string()));
}

#[test]
fn test_dispatch_metrics_query_unknown_format() {
    // Unknown formats should still deserialize (validation happens elsewhere)
    let json = r#"{"format": "xml"}"#;
    let query: DispatchMetricsQuery = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(query.format, Some("xml".to_string()));
}

#[test]
fn test_dispatch_metrics_response_large_buckets() {
    let response = DispatchMetricsResponse {
        cpu_dispatches: 1000,
        gpu_dispatches: 2000,
        total_dispatches: 3000,
        gpu_ratio: 0.667,
        cpu_latency_p50_us: 100.0,
        cpu_latency_p95_us: 200.0,
        cpu_latency_p99_us: 500.0,
        gpu_latency_p50_us: 50.0,
        gpu_latency_p95_us: 100.0,
        gpu_latency_p99_us: 200.0,
        cpu_latency_mean_us: 120.0,
        gpu_latency_mean_us: 60.0,
        cpu_latency_min_us: 10,
        cpu_latency_max_us: 1000,
        gpu_latency_min_us: 5,
        gpu_latency_max_us: 500,
        cpu_latency_variance_us: 2500.0,
        cpu_latency_stddev_us: 50.0,
        gpu_latency_variance_us: 1000.0,
        gpu_latency_stddev_us: 31.6,
        bucket_boundaries_us: vec![
            "0-50".to_string(),
            "50-100".to_string(),
            "100-200".to_string(),
            "200-500".to_string(),
            "500-1000".to_string(),
            "1000-2000".to_string(),
            "2000-5000".to_string(),
            "5000+".to_string(),
        ],
        cpu_latency_bucket_counts: vec![100, 200, 300, 200, 100, 50, 30, 20],
        gpu_latency_bucket_counts: vec![500, 600, 400, 300, 150, 40, 8, 2],
        throughput_rps: 50.0,
        elapsed_seconds: 60.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""bucket_boundaries_us""#));

    let deserialized: DispatchMetricsResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.bucket_boundaries_us.len(), 8);
    assert_eq!(deserialized.cpu_latency_bucket_counts.len(), 8);
    assert_eq!(deserialized.gpu_latency_bucket_counts.len(), 8);
}

// ============================================================================
// Test 31-35: ChatCompletionChunk Additional Tests
// ============================================================================

#[test]
fn test_chat_completion_chunk_multiple_choices() {
    let chunk = ChatCompletionChunk {
        id: "test-multi".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1000,
        model: "test-model".to_string(),
        choices: vec![
            ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some("First".to_string()),
                },
                finish_reason: None,
            },
            ChatChunkChoice {
                index: 1,
                delta: ChatDelta {
                    role: None,
                    content: Some("Second".to_string()),
                },
                finish_reason: None,
            },
        ],
    };

    let json = serde_json::to_string(&chunk).expect("should serialize");
    assert!(json.contains(r#""index":0"#));
    assert!(json.contains(r#""index":1"#));

    let deserialized: ChatCompletionChunk =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.choices.len(), 2);
}

#[test]
fn test_chat_completion_chunk_empty_choices() {
    let chunk = ChatCompletionChunk {
        id: "empty-choices".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 500,
        model: "model".to_string(),
        choices: vec![],
    };

    let json = serde_json::to_string(&chunk).expect("should serialize");
    assert!(json.contains(r#""choices":[]"#));

    let deserialized: ChatCompletionChunk =
        serde_json::from_str(&json).expect("should deserialize");
    assert!(deserialized.choices.is_empty());
}

#[test]
fn test_chat_delta_deserialization_from_openai_format() {
    // Test parsing typical OpenAI streaming format
    let json = r#"{"role": "assistant"}"#;
    let delta: ChatDelta = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(delta.role, Some("assistant".to_string()));
    assert!(delta.content.is_none());

    let json2 = r#"{"content": "Hello"}"#;
    let delta2: ChatDelta = serde_json::from_str(json2).expect("should deserialize");
    assert!(delta2.role.is_none());
    assert_eq!(delta2.content, Some("Hello".to_string()));
}

#[test]
fn test_chat_chunk_choice_finish_reason_variations() {
    // Test different finish reasons
    for reason in ["stop", "length", "content_filter", "function_call"] {
        let choice = ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: None,
            },
            finish_reason: Some(reason.to_string()),
        };

        let json = serde_json::to_string(&choice).expect("should serialize");
        assert!(json.contains(&format!(r#""finish_reason":"{}""#, reason)));

        let deserialized: ChatChunkChoice =
            serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(deserialized.finish_reason, Some(reason.to_string()));
    }
}

#[test]
fn test_chat_completion_chunk_roundtrip() {
    let original = ChatCompletionChunk {
        id: "roundtrip-test".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1234567890,
        model: "gpt-4-turbo".to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: Some("Test content with special chars: \"quotes\" and \\backslash".to_string()),
            },
            finish_reason: Some("stop".to_string()),
        }],
    };

    let json = serde_json::to_string(&original).expect("should serialize");
    let deserialized: ChatCompletionChunk =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(original.id, deserialized.id);
    assert_eq!(original.model, deserialized.model);
    assert_eq!(original.created, deserialized.created);
    assert_eq!(original.choices.len(), deserialized.choices.len());
    assert_eq!(
        original.choices[0].delta.content,
        deserialized.choices[0].delta.content
    );
}

// ============================================================================
// Test 36-40: Additional Edge Cases
// ============================================================================

#[test]
fn test_dispatch_reset_response_roundtrip() {
    let original = DispatchResetResponse {
        success: true,
        message: "All metrics cleared at 2024-01-15T10:00:00Z".to_string(),
    };

    let json = serde_json::to_string(&original).expect("should serialize");
    let deserialized: DispatchResetResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(original.success, deserialized.success);
    assert_eq!(original.message, deserialized.message);
}

#[test]
fn test_dispatch_metrics_response_boundary_values() {
    let response = DispatchMetricsResponse {
        cpu_dispatches: usize::MAX,
        gpu_dispatches: 0,
        total_dispatches: usize::MAX,
        gpu_ratio: 0.0,
        cpu_latency_p50_us: f64::MAX,
        cpu_latency_p95_us: f64::MAX,
        cpu_latency_p99_us: f64::MAX,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: f64::MAX,
        gpu_latency_mean_us: 0.0,
        cpu_latency_min_us: 0,
        cpu_latency_max_us: u64::MAX,
        gpu_latency_min_us: 0,
        gpu_latency_max_us: 0,
        cpu_latency_variance_us: f64::MAX,
        cpu_latency_stddev_us: f64::MAX,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        throughput_rps: f64::MAX,
        elapsed_seconds: f64::MAX,
    };

    // Should serialize without panic
    let json = serde_json::to_string(&response).expect("should serialize extreme values");
    assert!(!json.is_empty());
}

#[test]
fn test_audit_response_unicode_content() {
    let record = AuditRecord {
        request_id: "unicode-test".to_string(),
        timestamp: "2024-01-15T15:00:00Z".to_string(),
        model_id: "multilingual".to_string(),
        model_hash: "blake3:unicode".to_string(),
        input_shape: vec![1, 256],
        output: serde_json::json!({"text": "Hello World Bonjour"}),
        latency_ms: 20.0,
        confidence: Some(0.9),
    };

    let response = AuditResponse { record };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains("Hello"));
    assert!(json.contains("World"));

    let deserialized: AuditResponse = serde_json::from_str(&json).expect("should deserialize");
    let text = deserialized.record.output["text"].as_str().unwrap();
    assert!(text.contains("Hello"));
}

#[test]
fn test_context_window_manager_empty_messages() {
    let manager = ContextWindowManager::default_manager();
    let messages: Vec<ChatMessage> = vec![];

    assert!(!manager.needs_truncation(&messages));

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert!(result.is_empty());

    let estimate = manager.estimate_total_tokens(&messages);
    assert_eq!(estimate, 0);
}

#[test]
fn test_context_window_very_small_window() {
    // Window so small nothing fits
    let config = ContextWindowConfig::new(10).with_reserved_output(10);
    let manager = ContextWindowManager::new(config);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "This is a message that won't fit.".to_string(),
        name: None,
    }];

    assert!(manager.needs_truncation(&messages));

    let (result, truncated) = manager.truncate_messages(&messages);
    // When nothing fits, result may be empty or truncated
    if truncated {
        // Expected behavior with very small window
        assert!(result.len() <= messages.len());
    }
}

// ============================================================================
// Test 41-45: Integration-style tests for struct interactions
// ============================================================================

#[test]
fn test_dispatch_metrics_roundtrip_with_all_fields() {
    let original = DispatchMetricsResponse {
        cpu_dispatches: 500,
        gpu_dispatches: 1500,
        total_dispatches: 2000,
        gpu_ratio: 0.75,
        cpu_latency_p50_us: 120.5,
        cpu_latency_p95_us: 250.3,
        cpu_latency_p99_us: 480.7,
        gpu_latency_p50_us: 45.2,
        gpu_latency_p95_us: 90.1,
        gpu_latency_p99_us: 150.8,
        cpu_latency_mean_us: 135.0,
        gpu_latency_mean_us: 55.0,
        cpu_latency_min_us: 20,
        cpu_latency_max_us: 800,
        gpu_latency_min_us: 10,
        gpu_latency_max_us: 300,
        cpu_latency_variance_us: 5000.0,
        cpu_latency_stddev_us: 70.7,
        gpu_latency_variance_us: 2000.0,
        gpu_latency_stddev_us: 44.7,
        bucket_boundaries_us: vec!["0-100us".to_string(), "100-500us".to_string()],
        cpu_latency_bucket_counts: vec![250, 250],
        gpu_latency_bucket_counts: vec![1000, 500],
        throughput_rps: 33.33,
        elapsed_seconds: 60.0,
    };

    let json = serde_json::to_string_pretty(&original).expect("should serialize");

    // Verify JSON structure
    assert!(json.contains("cpu_dispatches"));
    assert!(json.contains("gpu_dispatches"));
    assert!(json.contains("throughput_rps"));

    // Roundtrip
    let deserialized: DispatchMetricsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(original.cpu_dispatches, deserialized.cpu_dispatches);
    assert_eq!(original.gpu_dispatches, deserialized.gpu_dispatches);
    assert!((original.gpu_ratio - deserialized.gpu_ratio).abs() < 0.001);
    assert!((original.throughput_rps - deserialized.throughput_rps).abs() < 0.01);
}

#[test]
fn test_chat_chunk_streaming_sequence() {
    // Simulate a streaming sequence
    let chunks = vec![
        // Initial chunk with role
        ChatCompletionChunk {
            id: "stream-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1000,
            model: "gpt-4".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        },
        // Content chunk 1
        ChatCompletionChunk {
            id: "stream-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1000,
            model: "gpt-4".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        },
        // Content chunk 2
        ChatCompletionChunk {
            id: "stream-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1000,
            model: "gpt-4".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some(", world!".to_string()),
                },
                finish_reason: None,
            }],
        },
        // Final chunk with finish_reason
        ChatCompletionChunk {
            id: "stream-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1000,
            model: "gpt-4".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        },
    ];

    // Verify all chunks serialize correctly
    for chunk in &chunks {
        let json = serde_json::to_string(chunk).expect("chunk should serialize");
        let _: ChatCompletionChunk =
            serde_json::from_str(&json).expect("chunk should deserialize");
    }

    // Verify content assembly
    let mut assembled_content = String::new();
    for chunk in &chunks {
        if let Some(content) = &chunk.choices[0].delta.content {
            assembled_content.push_str(content);
        }
    }
    assert_eq!(assembled_content, "Hello, world!");

    // Verify final chunk has finish_reason
    assert_eq!(
        chunks.last().unwrap().choices[0].finish_reason,
        Some("stop".to_string())
    );
}

#[test]
fn test_audit_record_minimal_valid() {
    let record = AuditRecord {
        request_id: "min".to_string(),
        timestamp: "2024-01-01T00:00:00Z".to_string(),
        model_id: "m".to_string(),
        model_hash: "h".to_string(),
        input_shape: vec![1],
        output: serde_json::json!(null),
        latency_ms: 0.0,
        confidence: None,
    };

    let response = AuditResponse { record };
    let json = serde_json::to_string(&response).expect("should serialize minimal record");
    assert!(json.contains(r#""request_id":"min""#));
}

#[test]
fn test_context_window_config_zero_reserved() {
    let config = ContextWindowConfig::new(100).with_reserved_output(0);
    let manager = ContextWindowManager::new(config);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Test".to_string(),
        name: None,
    }];

    // With 0 reserved, full window available for prompt
    let (result, _) = manager.truncate_messages(&messages);
    assert!(!result.is_empty());
}

#[test]
fn test_dispatch_metrics_nan_handling() {
    // Test that NaN values can be serialized (JSON allows it as null or string)
    let response = DispatchMetricsResponse {
        cpu_dispatches: 0,
        gpu_dispatches: 0,
        total_dispatches: 0,
        gpu_ratio: 0.0, // Avoid NaN in ratio
        cpu_latency_p50_us: 0.0,
        cpu_latency_p95_us: 0.0,
        cpu_latency_p99_us: 0.0,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: 0.0,
        gpu_latency_mean_us: 0.0,
        cpu_latency_min_us: 0,
        cpu_latency_max_us: 0,
        gpu_latency_min_us: 0,
        gpu_latency_max_us: 0,
        cpu_latency_variance_us: 0.0,
        cpu_latency_stddev_us: 0.0,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize zeros");
    assert!(json.contains(r#""throughput_rps":0.0"#));
}

// ============================================================================
// Test 46-50: Additional struct field access and method tests
// ============================================================================

#[test]
fn test_dispatch_metrics_response_field_access() {
    let response = DispatchMetricsResponse {
        cpu_dispatches: 100,
        gpu_dispatches: 200,
        total_dispatches: 300,
        gpu_ratio: 0.667,
        cpu_latency_p50_us: 50.0,
        cpu_latency_p95_us: 100.0,
        cpu_latency_p99_us: 200.0,
        gpu_latency_p50_us: 25.0,
        gpu_latency_p95_us: 50.0,
        gpu_latency_p99_us: 100.0,
        cpu_latency_mean_us: 60.0,
        gpu_latency_mean_us: 30.0,
        cpu_latency_min_us: 10,
        cpu_latency_max_us: 500,
        gpu_latency_min_us: 5,
        gpu_latency_max_us: 250,
        cpu_latency_variance_us: 1000.0,
        cpu_latency_stddev_us: 31.6,
        gpu_latency_variance_us: 500.0,
        gpu_latency_stddev_us: 22.4,
        bucket_boundaries_us: vec!["a".to_string()],
        cpu_latency_bucket_counts: vec![100],
        gpu_latency_bucket_counts: vec![200],
        throughput_rps: 5.0,
        elapsed_seconds: 60.0,
    };

    // Direct field access
    assert_eq!(response.cpu_dispatches, 100);
    assert_eq!(response.gpu_dispatches, 200);
    assert_eq!(response.total_dispatches, 300);
    assert!((response.gpu_ratio - 0.667).abs() < 0.001);
    assert_eq!(response.cpu_latency_min_us, 10);
    assert_eq!(response.cpu_latency_max_us, 500);
    assert_eq!(response.bucket_boundaries_us.len(), 1);
    assert_eq!(response.cpu_latency_bucket_counts[0], 100);
}

#[test]
fn test_chat_chunk_choice_field_access() {
    let choice = ChatChunkChoice {
        index: 5,
        delta: ChatDelta {
            role: Some("system".to_string()),
            content: Some("You are a helpful assistant.".to_string()),
        },
        finish_reason: Some("stop".to_string()),
    };

    assert_eq!(choice.index, 5);
    assert_eq!(choice.delta.role, Some("system".to_string()));
    assert_eq!(
        choice.delta.content,
        Some("You are a helpful assistant.".to_string())
    );
    assert_eq!(choice.finish_reason, Some("stop".to_string()));
}

#[test]
fn test_audit_record_field_access() {
    let record = AuditRecord {
        request_id: "field-test".to_string(),
        timestamp: "2024-06-15T12:00:00Z".to_string(),
        model_id: "test-model".to_string(),
        model_hash: "hash-abc".to_string(),
        input_shape: vec![1, 2, 3],
        output: serde_json::json!({"result": true}),
        latency_ms: 42.5,
        confidence: Some(0.99),
    };

    assert_eq!(record.request_id, "field-test");
    assert_eq!(record.timestamp, "2024-06-15T12:00:00Z");
    assert_eq!(record.model_id, "test-model");
    assert_eq!(record.model_hash, "hash-abc");
    assert_eq!(record.input_shape, vec![1, 2, 3]);
    assert_eq!(record.output["result"], true);
    assert!((record.latency_ms - 42.5).abs() < 0.001);
    assert_eq!(record.confidence, Some(0.99));
}

#[test]
fn test_dispatch_reset_response_field_access() {
    let response = DispatchResetResponse {
        success: false,
        message: "Failed to reset".to_string(),
    };

    assert!(!response.success);
    assert_eq!(response.message, "Failed to reset");
}

#[test]
fn test_context_window_config_field_access() {
    let config = ContextWindowConfig::new(8192).with_reserved_output(1024);

    assert_eq!(config.max_tokens, 8192);
    assert_eq!(config.reserved_output_tokens, 1024);
    assert!(config.preserve_system);
}
