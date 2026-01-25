//! API Tests Part 11: GPU Handlers Coverage
//!
//! Tests for gpu_handlers.rs to improve coverage.
//! Focus: GPU batch requests, warmup, status, request validation, error paths.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app;
use crate::api::{
    BatchGenerateResponse, BatchTokenizeResponse, GenerateResponse, GpuBatchRequest,
    GpuBatchResponse, GpuBatchResult, GpuBatchStats, GpuStatusResponse, GpuWarmupResponse,
    ModelsResponse, TokenizeResponse,
};

// =============================================================================
// GpuBatchRequest Tests
// =============================================================================

#[test]
fn test_gpu_batch_request_serialization() {
    let request = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: vec!["</s>".to_string()],
    };

    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("Hello"));
    assert!(json.contains("World"));
    assert!(json.contains("50"));
    assert!(json.contains("0.7"));
    assert!(json.contains("40"));
    assert!(json.contains("</s>"));
}

#[test]
fn test_gpu_batch_request_deserialization_minimal() {
    let json = r#"{"prompts": ["Test prompt"]}"#;
    let request: GpuBatchRequest = serde_json::from_str(json).expect("deserialize");

    assert_eq!(request.prompts.len(), 1);
    assert_eq!(request.prompts[0], "Test prompt");
    // Check defaults
    assert_eq!(request.max_tokens, 50); // default_max_tokens
    assert_eq!(request.temperature, 0.0); // default
    assert_eq!(request.top_k, 50); // default_top_k
    assert!(request.stop.is_empty()); // default
}

#[test]
fn test_gpu_batch_request_deserialization_full() {
    let json = r#"{
        "prompts": ["prompt1", "prompt2", "prompt3"],
        "max_tokens": 100,
        "temperature": 0.9,
        "top_k": 20,
        "stop": ["stop1", "stop2"]
    }"#;
    let request: GpuBatchRequest = serde_json::from_str(json).expect("deserialize");

    assert_eq!(request.prompts.len(), 3);
    assert_eq!(request.max_tokens, 100);
    assert!((request.temperature - 0.9).abs() < 0.01);
    assert_eq!(request.top_k, 20);
    assert_eq!(request.stop.len(), 2);
}

#[test]
fn test_gpu_batch_request_clone() {
    let request = GpuBatchRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 25,
        temperature: 0.5,
        top_k: 30,
        stop: vec![],
    };

    let cloned = request.clone();
    assert_eq!(cloned.prompts, request.prompts);
    assert_eq!(cloned.max_tokens, request.max_tokens);
    assert_eq!(cloned.temperature, request.temperature);
    assert_eq!(cloned.top_k, request.top_k);
}

#[test]
fn test_gpu_batch_request_debug() {
    let request = GpuBatchRequest {
        prompts: vec!["debug test".to_string()],
        max_tokens: 10,
        temperature: 0.1,
        top_k: 5,
        stop: vec![],
    };

    let debug = format!("{:?}", request);
    assert!(debug.contains("GpuBatchRequest"));
    assert!(debug.contains("debug test"));
}

// =============================================================================
// GpuBatchResponse Tests
// =============================================================================

#[test]
fn test_gpu_batch_response_serialization() {
    let response = GpuBatchResponse {
        results: vec![
            GpuBatchResult {
                index: 0,
                token_ids: vec![1, 2, 3],
                text: "Hello".to_string(),
                num_generated: 3,
            },
            GpuBatchResult {
                index: 1,
                token_ids: vec![4, 5, 6, 7],
                text: "World".to_string(),
                num_generated: 4,
            },
        ],
        stats: GpuBatchStats {
            batch_size: 2,
            gpu_used: true,
            total_tokens: 7,
            processing_time_ms: 15.5,
            throughput_tps: 451.6,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("Hello"));
    assert!(json.contains("World"));
    assert!(json.contains("batch_size"));
    assert!(json.contains("gpu_used"));
    assert!(json.contains("throughput_tps"));
}

#[test]
fn test_gpu_batch_response_deserialization() {
    let json = r#"{
        "results": [
            {"index": 0, "token_ids": [1, 2], "text": "test", "num_generated": 2}
        ],
        "stats": {
            "batch_size": 1,
            "gpu_used": false,
            "total_tokens": 2,
            "processing_time_ms": 10.0,
            "throughput_tps": 200.0
        }
    }"#;
    let response: GpuBatchResponse = serde_json::from_str(json).expect("deserialize");

    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].index, 0);
    assert_eq!(response.stats.batch_size, 1);
    assert!(!response.stats.gpu_used);
}

#[test]
fn test_gpu_batch_response_clone() {
    let response = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1],
            text: "x".to_string(),
            num_generated: 1,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 1,
            processing_time_ms: 5.0,
            throughput_tps: 200.0,
        },
    };

    let cloned = response.clone();
    assert_eq!(cloned.results.len(), response.results.len());
    assert_eq!(cloned.stats.batch_size, response.stats.batch_size);
}

#[test]
fn test_gpu_batch_response_debug() {
    let response = GpuBatchResponse {
        results: vec![],
        stats: GpuBatchStats {
            batch_size: 0,
            gpu_used: false,
            total_tokens: 0,
            processing_time_ms: 0.0,
            throughput_tps: 0.0,
        },
    };

    let debug = format!("{:?}", response);
    assert!(debug.contains("GpuBatchResponse"));
}

// =============================================================================
// GpuBatchResult Tests
// =============================================================================

#[test]
fn test_gpu_batch_result_serialization() {
    let result = GpuBatchResult {
        index: 5,
        token_ids: vec![10, 20, 30, 40, 50],
        text: "Generated text here".to_string(),
        num_generated: 5,
    };

    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains("\"index\":5"));
    assert!(json.contains("Generated text here"));
    assert!(json.contains("num_generated"));
}

#[test]
fn test_gpu_batch_result_deserialization() {
    let json = r#"{"index": 3, "token_ids": [100, 200], "text": "hi", "num_generated": 2}"#;
    let result: GpuBatchResult = serde_json::from_str(json).expect("deserialize");

    assert_eq!(result.index, 3);
    assert_eq!(result.token_ids, vec![100, 200]);
    assert_eq!(result.text, "hi");
    assert_eq!(result.num_generated, 2);
}

#[test]
fn test_gpu_batch_result_clone() {
    let result = GpuBatchResult {
        index: 1,
        token_ids: vec![1, 2, 3],
        text: "clone test".to_string(),
        num_generated: 3,
    };

    let cloned = result.clone();
    assert_eq!(cloned.index, result.index);
    assert_eq!(cloned.token_ids, result.token_ids);
    assert_eq!(cloned.text, result.text);
}

#[test]
fn test_gpu_batch_result_debug() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![],
        text: "debug".to_string(),
        num_generated: 0,
    };

    let debug = format!("{:?}", result);
    assert!(debug.contains("GpuBatchResult"));
}

// =============================================================================
// GpuBatchStats Tests
// =============================================================================

#[test]
fn test_gpu_batch_stats_serialization() {
    let stats = GpuBatchStats {
        batch_size: 32,
        gpu_used: true,
        total_tokens: 1024,
        processing_time_ms: 100.5,
        throughput_tps: 10189.05,
    };

    let json = serde_json::to_string(&stats).expect("serialize");
    assert!(json.contains("32"));
    assert!(json.contains("true"));
    assert!(json.contains("1024"));
    assert!(json.contains("100.5"));
    assert!(json.contains("10189.05"));
}

#[test]
fn test_gpu_batch_stats_deserialization() {
    let json = r#"{
        "batch_size": 64,
        "gpu_used": true,
        "total_tokens": 512,
        "processing_time_ms": 50.0,
        "throughput_tps": 10240.0
    }"#;
    let stats: GpuBatchStats = serde_json::from_str(json).expect("deserialize");

    assert_eq!(stats.batch_size, 64);
    assert!(stats.gpu_used);
    assert_eq!(stats.total_tokens, 512);
    assert!((stats.processing_time_ms - 50.0).abs() < 0.01);
}

#[test]
fn test_gpu_batch_stats_clone() {
    let stats = GpuBatchStats {
        batch_size: 16,
        gpu_used: false,
        total_tokens: 256,
        processing_time_ms: 25.0,
        throughput_tps: 10240.0,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.batch_size, stats.batch_size);
    assert_eq!(cloned.gpu_used, stats.gpu_used);
    assert_eq!(cloned.total_tokens, stats.total_tokens);
}

#[test]
fn test_gpu_batch_stats_debug() {
    let stats = GpuBatchStats {
        batch_size: 8,
        gpu_used: true,
        total_tokens: 128,
        processing_time_ms: 12.5,
        throughput_tps: 10240.0,
    };

    let debug = format!("{:?}", stats);
    assert!(debug.contains("GpuBatchStats"));
    assert!(debug.contains("batch_size"));
}

// =============================================================================
// GpuWarmupResponse Tests
// =============================================================================

#[test]
fn test_gpu_warmup_response_serialization() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 1_073_741_824, // 1GB
        num_layers: 32,
        message: "GPU cache warmed up successfully".to_string(),
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("true"));
    assert!(json.contains("1073741824"));
    assert!(json.contains("32"));
    assert!(json.contains("GPU cache warmed up successfully"));
}

#[test]
fn test_gpu_warmup_response_deserialization() {
    let json = r#"{
        "success": false,
        "memory_bytes": 0,
        "num_layers": 0,
        "message": "Warmup failed"
    }"#;
    let response: GpuWarmupResponse = serde_json::from_str(json).expect("deserialize");

    assert!(!response.success);
    assert_eq!(response.memory_bytes, 0);
    assert_eq!(response.num_layers, 0);
    assert_eq!(response.message, "Warmup failed");
}

#[test]
fn test_gpu_warmup_response_clone() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 500_000_000,
        num_layers: 24,
        message: "OK".to_string(),
    };

    let cloned = response.clone();
    assert_eq!(cloned.success, response.success);
    assert_eq!(cloned.memory_bytes, response.memory_bytes);
    assert_eq!(cloned.num_layers, response.num_layers);
    assert_eq!(cloned.message, response.message);
}

#[test]
fn test_gpu_warmup_response_debug() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 100,
        num_layers: 1,
        message: "test".to_string(),
    };

    let debug = format!("{:?}", response);
    assert!(debug.contains("GpuWarmupResponse"));
}

// =============================================================================
// GpuStatusResponse Tests
// =============================================================================

#[test]
fn test_gpu_status_response_serialization() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 2_147_483_648, // 2GB
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("cache_ready"));
    assert!(json.contains("true"));
    assert!(json.contains("2147483648"));
    assert!(json.contains("batch_threshold"));
    assert!(json.contains("recommended_min_batch"));
}

#[test]
fn test_gpu_status_response_deserialization() {
    let json = r#"{
        "cache_ready": false,
        "cache_memory_bytes": 0,
        "batch_threshold": 32,
        "recommended_min_batch": 32
    }"#;
    let response: GpuStatusResponse = serde_json::from_str(json).expect("deserialize");

    assert!(!response.cache_ready);
    assert_eq!(response.cache_memory_bytes, 0);
    assert_eq!(response.batch_threshold, 32);
    assert_eq!(response.recommended_min_batch, 32);
}

#[test]
fn test_gpu_status_response_clone() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1_000_000,
        batch_threshold: 16,
        recommended_min_batch: 16,
    };

    let cloned = response.clone();
    assert_eq!(cloned.cache_ready, response.cache_ready);
    assert_eq!(cloned.cache_memory_bytes, response.cache_memory_bytes);
    assert_eq!(cloned.batch_threshold, response.batch_threshold);
}

#[test]
fn test_gpu_status_response_debug() {
    let response = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    let debug = format!("{:?}", response);
    assert!(debug.contains("GpuStatusResponse"));
}

// =============================================================================
// BatchConfig Tests (GPU feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_default() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default();
    assert_eq!(config.window_ms, 50);
    assert_eq!(config.min_batch, 4);
    assert_eq!(config.optimal_batch, 32);
    assert_eq!(config.max_batch, 64);
    assert_eq!(config.queue_size, 1024);
    assert_eq!(config.gpu_threshold, 32);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_low_latency() {
    use crate::api::BatchConfig;

    let config = BatchConfig::low_latency();
    assert_eq!(config.window_ms, 5);
    assert_eq!(config.min_batch, 2);
    assert_eq!(config.optimal_batch, 8);
    assert_eq!(config.max_batch, 16);
    assert_eq!(config.queue_size, 512);
    assert_eq!(config.gpu_threshold, 32);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_high_throughput() {
    use crate::api::BatchConfig;

    let config = BatchConfig::high_throughput();
    assert_eq!(config.window_ms, 100);
    assert_eq!(config.min_batch, 8);
    assert_eq!(config.optimal_batch, 32);
    assert_eq!(config.max_batch, 128);
    assert_eq!(config.queue_size, 2048);
    assert_eq!(config.gpu_threshold, 32);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_should_process() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default(); // optimal_batch = 32
    assert!(!config.should_process(0));
    assert!(!config.should_process(16));
    assert!(!config.should_process(31));
    assert!(config.should_process(32));
    assert!(config.should_process(64));
    assert!(config.should_process(100));
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_meets_minimum() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default(); // min_batch = 4
    assert!(!config.meets_minimum(0));
    assert!(!config.meets_minimum(1));
    assert!(!config.meets_minimum(3));
    assert!(config.meets_minimum(4));
    assert!(config.meets_minimum(5));
    assert!(config.meets_minimum(100));
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_clone() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.window_ms, config.window_ms);
    assert_eq!(cloned.min_batch, config.min_batch);
    assert_eq!(cloned.optimal_batch, config.optimal_batch);
    assert_eq!(cloned.max_batch, config.max_batch);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_debug() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("BatchConfig"));
    assert!(debug.contains("window_ms"));
}

// =============================================================================
// ContinuousBatchResponse Tests (GPU feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_single() {
    use crate::api::ContinuousBatchResponse;

    let response = ContinuousBatchResponse::single(
        vec![1, 2, 3, 4, 5], // token_ids
        3,                   // prompt_len
        12.5,                // latency_ms
    );

    assert_eq!(response.token_ids, vec![1, 2, 3, 4, 5]);
    assert_eq!(response.prompt_len, 3);
    assert!(!response.batched);
    assert_eq!(response.batch_size, 1);
    assert!((response.latency_ms - 12.5).abs() < 0.01);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_batched() {
    use crate::api::ContinuousBatchResponse;

    let response = ContinuousBatchResponse::batched(
        vec![10, 20, 30, 40, 50, 60], // token_ids
        4,                            // prompt_len
        16,                           // batch_size
        50.0,                         // latency_ms
    );

    assert_eq!(response.token_ids, vec![10, 20, 30, 40, 50, 60]);
    assert_eq!(response.prompt_len, 4);
    assert!(response.batched);
    assert_eq!(response.batch_size, 16);
    assert!((response.latency_ms - 50.0).abs() < 0.01);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_generated_tokens() {
    use crate::api::ContinuousBatchResponse;

    // Normal case: token_ids longer than prompt_len
    let response = ContinuousBatchResponse::single(vec![1, 2, 3, 4, 5, 6, 7], 3, 10.0);
    let generated = response.generated_tokens();
    assert_eq!(generated, &[4, 5, 6, 7]);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_generated_tokens_empty() {
    use crate::api::ContinuousBatchResponse;

    // Edge case: prompt_len equals token_ids.len()
    let response = ContinuousBatchResponse::single(vec![1, 2, 3], 3, 5.0);
    let generated = response.generated_tokens();
    assert!(generated.is_empty());
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_generated_tokens_overflow() {
    use crate::api::ContinuousBatchResponse;

    // Edge case: prompt_len > token_ids.len() (shouldn't happen but handle gracefully)
    let response = ContinuousBatchResponse::single(vec![1, 2], 10, 5.0);
    let generated = response.generated_tokens();
    assert!(generated.is_empty());
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_clone() {
    use crate::api::ContinuousBatchResponse;

    let response = ContinuousBatchResponse::batched(vec![1, 2, 3], 1, 8, 25.0);
    let cloned = response.clone();

    assert_eq!(cloned.token_ids, response.token_ids);
    assert_eq!(cloned.prompt_len, response.prompt_len);
    assert_eq!(cloned.batched, response.batched);
    assert_eq!(cloned.batch_size, response.batch_size);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_debug() {
    use crate::api::ContinuousBatchResponse;

    let response = ContinuousBatchResponse::single(vec![1], 0, 1.0);
    let debug = format!("{:?}", response);
    assert!(debug.contains("ContinuousBatchResponse"));
}

// =============================================================================
// BatchQueueStats Tests (GPU feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_default() {
    use crate::api::BatchQueueStats;

    let stats = BatchQueueStats::default();
    assert_eq!(stats.total_queued, 0);
    assert_eq!(stats.total_batches, 0);
    assert_eq!(stats.total_single, 0);
    assert!((stats.avg_batch_size - 0.0).abs() < f64::EPSILON);
    assert!((stats.avg_wait_ms - 0.0).abs() < f64::EPSILON);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_clone() {
    use crate::api::BatchQueueStats;

    let stats = BatchQueueStats {
        total_queued: 100,
        total_batches: 10,
        total_single: 20,
        avg_batch_size: 10.0,
        avg_wait_ms: 5.5,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.total_queued, stats.total_queued);
    assert_eq!(cloned.total_batches, stats.total_batches);
    assert_eq!(cloned.total_single, stats.total_single);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_debug() {
    use crate::api::BatchQueueStats;

    let stats = BatchQueueStats::default();
    let debug = format!("{:?}", stats);
    assert!(debug.contains("BatchQueueStats"));
    assert!(debug.contains("total_queued"));
}

// =============================================================================
// BatchProcessResult Tests (GPU feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_process_result_debug() {
    use crate::api::BatchProcessResult;

    let result = BatchProcessResult {
        requests_processed: 5,
        was_batched: true,
        total_time_ms: 50.0,
        avg_latency_ms: 10.0,
    };

    let debug = format!("{:?}", result);
    assert!(debug.contains("BatchProcessResult"));
    assert!(debug.contains("requests_processed"));
    assert!(debug.contains("was_batched"));
}

// =============================================================================
// GPU Endpoint HTTP Tests
// =============================================================================

#[tokio::test]
async fn test_gpu_warmup_endpoint_no_gpu_model() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/gpu/warmup")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Demo app doesn't have GPU model, should return error
    // When GPU feature is enabled, returns 503; when not, also returns 503
    assert!(
        response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_gpu_status_endpoint() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/gpu/status")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    // GPU status always returns OK (even without GPU model)
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let status: GpuStatusResponse = serde_json::from_slice(&body).expect("parse json");

    // Without GPU model, cache_ready should be false
    assert!(!status.cache_ready);
    assert_eq!(status.cache_memory_bytes, 0);
    assert_eq!(status.batch_threshold, 32);
    assert_eq!(status.recommended_min_batch, 32);
}

#[tokio::test]
async fn test_gpu_batch_completions_empty_prompts() {
    let app = create_test_app();

    let request = GpuBatchRequest {
        prompts: vec![], // Empty prompts array - should fail
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: vec![],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&request).expect("serialize"),
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Empty prompts should return 400 Bad Request
    // Note: Non-GPU build might return 503 instead
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_gpu_batch_completions_no_gpu_model() {
    let app = create_test_app();

    let request = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 10,
        temperature: 0.5,
        top_k: 40,
        stop: vec![],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&request).expect("serialize"),
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Demo app doesn't have GPU/cached model, should return error
    assert!(
        response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_gpu_batch_completions_invalid_json() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Invalid JSON should return 400 Bad Request
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_gpu_batch_completions_missing_prompts_field() {
    let app = create_test_app();

    // Missing required 'prompts' field
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"max_tokens": 50}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Missing required field should return 422 Unprocessable Entity
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_gpu_warmup_endpoint_method_not_allowed() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET") // Wrong method, should be POST
                .uri("/v1/gpu/warmup")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    // GET on POST-only endpoint should return 405
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_gpu_status_endpoint_post_method_not_allowed() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST") // Wrong method, should be GET
                .uri("/v1/gpu/status")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // POST on GET-only endpoint should return 405
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// =============================================================================
// Models Endpoint Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_models_handler_demo_mode() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let models: ModelsResponse = serde_json::from_slice(&body).expect("parse json");

    // Demo mode returns default model info
    assert!(!models.models.is_empty());
    assert_eq!(models.models[0].id, "default");
}

// =============================================================================
// Tokenize Handler Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_tokenize_handler_success() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "Hello world"}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let result: TokenizeResponse = serde_json::from_slice(&body).expect("parse json");

    assert!(result.num_tokens > 0);
    assert!(!result.token_ids.is_empty());
}

#[tokio::test]
async fn test_tokenize_handler_with_model_id() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "Test", "model_id": "nonexistent"}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Demo mode falls back to default model, so this should still work
    // or return NOT_FOUND depending on implementation
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND);
}

// =============================================================================
// Generate Handler Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_generate_handler_greedy() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompt": "Hello", "strategy": "greedy", "max_tokens": 5}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let result: GenerateResponse = serde_json::from_slice(&body).expect("parse json");

    assert!(!result.token_ids.is_empty());
    assert!(!result.text.is_empty());
}

#[tokio::test]
async fn test_generate_handler_top_k() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt": "Hello", "strategy": "top_k", "top_k": 10, "max_tokens": 5, "seed": 42}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_generate_handler_top_p() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt": "Hello", "strategy": "top_p", "top_p": 0.9, "max_tokens": 5, "seed": 42}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_generate_handler_empty_prompt() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt": "", "max_tokens": 5}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_generate_handler_invalid_strategy() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompt": "Hello", "strategy": "invalid_strategy", "max_tokens": 5}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// =============================================================================
// Batch Tokenize Handler Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_batch_tokenize_handler_success() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"texts": ["Hello", "World", "Test"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let result: BatchTokenizeResponse = serde_json::from_slice(&body).expect("parse json");

    assert_eq!(result.results.len(), 3);
    for r in &result.results {
        assert!(r.num_tokens > 0);
    }
}

#[tokio::test]
async fn test_batch_tokenize_handler_empty_texts() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"texts": []}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// =============================================================================
// Batch Generate Handler Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_batch_generate_handler_success() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompts": ["Hello", "World"], "max_tokens": 3, "strategy": "greedy"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let result: BatchGenerateResponse = serde_json::from_slice(&body).expect("parse json");

    assert_eq!(result.results.len(), 2);
}

#[tokio::test]
async fn test_batch_generate_handler_empty_prompts() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompts": [], "max_tokens": 5}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_batch_generate_handler_invalid_strategy() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompts": ["test"], "strategy": "invalid"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_batch_generate_handler_with_seed() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompts": ["Test"], "max_tokens": 3, "seed": 12345}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);
}

// =============================================================================
// Edge Cases and Error Paths
// =============================================================================

#[test]
fn test_gpu_batch_request_large_prompts() {
    let large_prompt = "x".repeat(10000);
    let request = GpuBatchRequest {
        prompts: vec![large_prompt.clone()],
        max_tokens: 10,
        temperature: 0.5,
        top_k: 40,
        stop: vec![],
    };

    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.len() > 10000);
}

#[test]
fn test_gpu_batch_request_many_prompts() {
    let prompts: Vec<String> = (0..100).map(|i| format!("Prompt {}", i)).collect();
    let request = GpuBatchRequest {
        prompts,
        max_tokens: 10,
        temperature: 0.5,
        top_k: 40,
        stop: vec![],
    };

    assert_eq!(request.prompts.len(), 100);
}

#[test]
fn test_gpu_batch_stats_zero_values() {
    let stats = GpuBatchStats {
        batch_size: 0,
        gpu_used: false,
        total_tokens: 0,
        processing_time_ms: 0.0,
        throughput_tps: 0.0,
    };

    let json = serde_json::to_string(&stats).expect("serialize");
    let parsed: GpuBatchStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.batch_size, 0);
    assert_eq!(parsed.total_tokens, 0);
}

#[test]
fn test_gpu_batch_stats_large_values() {
    let stats = GpuBatchStats {
        batch_size: 1_000_000,
        gpu_used: true,
        total_tokens: 1_000_000_000,
        processing_time_ms: 100_000.0,
        throughput_tps: 10_000_000.0,
    };

    let json = serde_json::to_string(&stats).expect("serialize");
    let parsed: GpuBatchStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.batch_size, 1_000_000);
}

#[test]
fn test_gpu_warmup_response_large_memory() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 24_000_000_000, // 24GB
        num_layers: 100,
        message: "Warmed up 24GB of VRAM".to_string(),
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("24000000000"));
}

#[test]
fn test_gpu_status_response_large_cache() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 48_000_000_000, // 48GB
        batch_threshold: 64,
        recommended_min_batch: 64,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("48000000000"));
}

// =============================================================================
// Stream Generate Handler Tests
// =============================================================================

#[tokio::test]
async fn test_stream_generate_handler_success() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompt": "Hello", "strategy": "greedy", "max_tokens": 3}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);
    // SSE responses have text/event-stream content type
    let content_type = response.headers().get("content-type");
    assert!(content_type.is_some());
}

#[tokio::test]
async fn test_stream_generate_handler_empty_prompt() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt": "", "max_tokens": 5}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_stream_generate_handler_invalid_strategy() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompt": "Hello", "strategy": "unknown_strategy", "max_tokens": 5}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// =============================================================================
// Additional Response Type Coverage
// =============================================================================

#[test]
fn test_tokenize_response_serialization() {
    let response = TokenizeResponse {
        token_ids: vec![1, 2, 3, 4, 5],
        num_tokens: 5,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("[1,2,3,4,5]"));
    assert!(json.contains("5"));
}

#[test]
fn test_generate_response_serialization() {
    let response = GenerateResponse {
        token_ids: vec![10, 20, 30],
        text: "Hello world".to_string(),
        num_generated: 3,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("Hello world"));
    assert!(json.contains("num_generated"));
}

#[test]
fn test_batch_tokenize_response_serialization() {
    let response = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1],
                num_tokens: 1,
            },
            TokenizeResponse {
                token_ids: vec![2, 3],
                num_tokens: 2,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("results"));
}

#[test]
fn test_batch_generate_response_serialization() {
    let response = BatchGenerateResponse {
        results: vec![
            GenerateResponse {
                token_ids: vec![1],
                text: "a".to_string(),
                num_generated: 1,
            },
            GenerateResponse {
                token_ids: vec![2],
                text: "b".to_string(),
                num_generated: 1,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("results"));
}

#[test]
fn test_models_response_serialization() {
    use crate::registry::ModelInfo;

    let response = ModelsResponse {
        models: vec![ModelInfo {
            id: "model-1".to_string(),
            name: "Model One".to_string(),
            description: "First model".to_string(),
            format: "gguf".to_string(),
            loaded: true,
        }],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("model-1"));
    assert!(json.contains("Model One"));
}
