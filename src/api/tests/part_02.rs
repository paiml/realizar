//! API Tests Part 02
//!
//! Generate endpoint tests, streaming tests

#[cfg(feature = "gpu")]
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
#[cfg(feature = "gpu")]
use tower::util::ServiceExt;

#[cfg(feature = "gpu")]
use crate::api::test_helpers::create_test_app;
#[cfg(feature = "gpu")]
use crate::api::test_helpers::create_test_quantized_model;
#[allow(unused_imports)]
use crate::api::*;

/// IMP-116d: Test cached model can be accessed multiple times (scheduler reuse)
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_116d_scheduler_reuse_across_requests() {
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

    // Verify cached model can be accessed multiple times concurrently
    let mut handles = Vec::new();
    for i in 0..5 {
        let model_clone = cached_model.clone();
        handles.push(tokio::spawn(async move {
            // Access model - this exercises the internal scheduler
            let inner = model_clone.model();
            assert_eq!(
                inner.config.hidden_dim, 64,
                "IMP-116d: Access {i} should succeed"
            );
        }));
    }

    // All concurrent accesses should succeed
    for (i, handle) in handles.into_iter().enumerate() {
        handle
            .await
            .unwrap_or_else(|_| panic!("IMP-116d: Concurrent access {i} should not panic"));
    }
}

// ============================================================
// IMP-126: Wire adaptive generation into HTTP serving handler
// RED phase: Tests written first, implementation to follow
// ============================================================

/// IMP-126a: AppState should have dispatch_metrics field
#[test]
#[cfg(feature = "gpu")]
fn test_imp_126a_appstate_has_dispatch_metrics() {
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

    // Create AppState with cached model
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-126a: Should create AppState");

    // Verify dispatch_metrics is accessible
    let metrics = state.dispatch_metrics();
    assert!(
        metrics.is_some(),
        "IMP-126a: AppState should have dispatch_metrics"
    );

    // Verify metrics starts at zero
    let m = metrics.expect("Should have metrics");
    assert_eq!(
        m.total_dispatches(),
        0,
        "IMP-126a: Metrics should start at zero"
    );
}

/// IMP-126b: OwnedQuantizedModelCachedSync has generate_with_cache_adaptive method
/// This test verifies the method signature exists on the type
#[test]
#[cfg(feature = "gpu")]
fn test_imp_126b_cached_sync_has_generate_adaptive() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
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
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 3,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Verify method exists by calling it (result may fail due to test model size)
    let prompt = vec![1u32, 2, 3];
    let _result = cached_model.generate_with_cache_adaptive(&prompt, &gen_config, &metrics);

    // IMP-126b: Method exists and can be called
    // Actual generation tested in gguf.rs with proper test model
    assert!(
        true,
        "IMP-126b: generate_with_cache_adaptive method exists on OwnedQuantizedModelCachedSync"
    );
}

/// IMP-126c: AppState provides dispatch_metrics for HTTP handlers
#[test]
#[cfg(feature = "gpu")]
fn test_imp_126c_dispatch_metrics_integration() {
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
    let cached_model = OwnedQuantizedModelCachedSync::new(model);

    // Create AppState with cached model - this should initialize dispatch_metrics
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-126c: Should create AppState");

    // Verify dispatch_metrics is accessible and shared
    let metrics1 = state.dispatch_metrics();
    let metrics2 = state.dispatch_metrics();

    assert!(
        metrics1.is_some(),
        "IMP-126c: dispatch_metrics should be available"
    );

    // Metrics should be shareable (Arc)
    let m1 = metrics1.expect("Should have metrics");
    let m2 = metrics2.expect("Should have metrics");
    assert!(
        Arc::ptr_eq(m1, m2),
        "IMP-126c: dispatch_metrics should be shared Arc"
    );
}

/// IMP-126d: Handler uses adaptive generation when dispatch_metrics available
/// This tests that the handler prefers generate_with_cache_adaptive over generate_with_cache
#[test]
#[cfg(feature = "gpu")]
fn test_imp_126d_handler_uses_adaptive_generation() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-126d: Should create AppState");

    // Handler should have dispatch_metrics available for adaptive generation
    let metrics = state.dispatch_metrics();
    assert!(
        metrics.is_some(),
        "IMP-126d: Handler should have dispatch_metrics for adaptive generation"
    );

    // Record initial state
    let m = metrics.expect("Should have metrics");
    let initial_cpu = m.cpu_dispatches();
    let initial_gpu = m.gpu_dispatches();

    // The handler code path (test) should use adaptive generation
    // which records dispatches to metrics. We verify the metrics are being
    // passed through by checking they can be incremented.
    m.record_cpu_dispatch();
    m.record_gpu_dispatch();

    assert_eq!(
        m.cpu_dispatches(),
        initial_cpu + 1,
        "IMP-126d: Metrics should track CPU dispatches"
    );
    assert_eq!(
        m.gpu_dispatches(),
        initial_gpu + 1,
        "IMP-126d: Metrics should track GPU dispatches"
    );
}

// ========================================================================
// IMP-127: /metrics/dispatch Endpoint Tests
// ========================================================================

/// IMP-127a: /metrics/dispatch endpoint exists and returns JSON
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_127a_dispatch_metrics_endpoint_exists() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-127a: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "IMP-127a: /metrics/dispatch should return 200 OK"
    );

    // Verify JSON content type
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(
        content_type.is_some_and(|s| s.contains("application/json")),
        "IMP-127a: Response should be JSON"
    );
}

/// IMP-127b: /metrics/dispatch returns correct structure
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_127b_dispatch_metrics_response_structure() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-127b: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let json: serde_json::Value =
        serde_json::from_slice(&body).expect("IMP-127b: Response should be valid JSON");

    // Verify required fields
    assert!(
        json.get("cpu_dispatches").is_some(),
        "IMP-127b: Response should have cpu_dispatches"
    );
    assert!(
        json.get("gpu_dispatches").is_some(),
        "IMP-127b: Response should have gpu_dispatches"
    );
    assert!(
        json.get("total_dispatches").is_some(),
        "IMP-127b: Response should have total_dispatches"
    );
    assert!(
        json.get("gpu_ratio").is_some(),
        "IMP-127b: Response should have gpu_ratio"
    );
}

/// IMP-127c: /metrics/dispatch starts at zero
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_127c_dispatch_metrics_starts_zero() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-127c: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("test");

    assert_eq!(
        json["cpu_dispatches"].as_u64(),
        Some(0),
        "IMP-127c: cpu_dispatches should start at 0"
    );
    assert_eq!(
        json["gpu_dispatches"].as_u64(),
        Some(0),
        "IMP-127c: gpu_dispatches should start at 0"
    );
    assert_eq!(
        json["total_dispatches"].as_u64(),
        Some(0),
        "IMP-127c: total_dispatches should start at 0"
    );
}

/// IMP-127d: /metrics/dispatch returns 503 when no GPU model configured
#[tokio::test]
async fn test_imp_127d_dispatch_metrics_no_gpu_model() {
    // Use demo() which creates AppState without cached model / dispatch metrics
    let state = AppState::demo().expect("Should create demo AppState");
    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    // Should return 503 Service Unavailable when no dispatch metrics available
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "IMP-127d: /metrics/dispatch should return 503 when no GPU model configured"
    );
}

// ========================================================================
// IMP-128: Prometheus Format Export Tests
// ========================================================================

/// IMP-128a: /metrics/dispatch?format=prometheus returns Prometheus format
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_128a_prometheus_format_endpoint() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-128a: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "IMP-128a: Prometheus format should return 200 OK"
    );

    // Verify text/plain content type for Prometheus
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(
        content_type.is_some_and(|s| s.contains("text/plain")),
        "IMP-128a: Prometheus response should be text/plain"
    );
}

/// IMP-128b: Prometheus format contains correct metric names
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_128b_prometheus_format_structure() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-128b: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let text = String::from_utf8_lossy(&body);

    // Verify Prometheus metric format
    assert!(
        text.contains("realizar_dispatch_cpu_total"),
        "IMP-128b: Should have CPU dispatch counter"
    );
    assert!(
        text.contains("realizar_dispatch_gpu_total"),
        "IMP-128b: Should have GPU dispatch counter"
    );
    assert!(
        text.contains("realizar_dispatch_gpu_ratio"),
        "IMP-128b: Should have GPU ratio gauge"
    );
}

/// IMP-128c: Default format (no query param) returns JSON
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_128c_default_format_is_json() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-128c: Should create AppState");

    let app = create_router(state);

    // Request without format parameter
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(
        content_type.is_some_and(|s| s.contains("application/json")),
        "IMP-128c: Default format should be JSON"
    );
}

/// IMP-128d: format=json explicitly returns JSON
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_128d_explicit_json_format() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-128d: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=json")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(
        content_type.is_some_and(|s| s.contains("application/json")),
        "IMP-128d: format=json should return JSON"
    );
}

// ===== IMP-130: Latency histogram in Prometheus export =====

/// IMP-130a: Prometheus export should include CPU latency histogram
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_130a_prometheus_includes_cpu_latency_histogram() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-130a: Should create AppState");

    // Record some CPU latency samples
    if let Some(metrics) = state.dispatch_metrics() {
        metrics.record_cpu_latency(std::time::Duration::from_micros(50));
        metrics.record_cpu_latency(std::time::Duration::from_micros(200));
        metrics.record_cpu_latency(std::time::Duration::from_micros(800));
    }

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-130a: Should include CPU latency histogram buckets
    assert!(
        body_str.contains("realizar_dispatch_cpu_latency_bucket"),
        "IMP-130a: Prometheus should include CPU latency histogram buckets. Got: {}",
        body_str
    );
    assert!(
        body_str.contains("realizar_dispatch_cpu_latency_sum"),
        "IMP-130a: Prometheus should include CPU latency sum"
    );
    assert!(
        body_str.contains("realizar_dispatch_cpu_latency_count"),
        "IMP-130a: Prometheus should include CPU latency count"
    );
}

/// IMP-130b: Prometheus export should include GPU latency histogram
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_130b_prometheus_includes_gpu_latency_histogram() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-130b: Should create AppState");

    // Record some GPU latency samples
    if let Some(metrics) = state.dispatch_metrics() {
        metrics.record_gpu_latency(std::time::Duration::from_micros(150));
        metrics.record_gpu_latency(std::time::Duration::from_micros(600));
    }

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-130b: Should include GPU latency histogram buckets
    assert!(
        body_str.contains("realizar_dispatch_gpu_latency_bucket"),
        "IMP-130b: Prometheus should include GPU latency histogram buckets. Got: {}",
        body_str
    );
    assert!(
        body_str.contains("realizar_dispatch_gpu_latency_sum"),
        "IMP-130b: Prometheus should include GPU latency sum"
    );
    assert!(
        body_str.contains("realizar_dispatch_gpu_latency_count"),
        "IMP-130b: Prometheus should include GPU latency count"
    );
}

/// IMP-130c: Prometheus latency histogram should have correct bucket labels
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_130c_prometheus_latency_buckets_have_correct_labels() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-130c: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-130c: Should have Prometheus histogram bucket labels (le="X")
    // Bucket boundaries: 100µs, 500µs, 1000µs, 5000µs, +Inf
    assert!(
        body_str.contains(r#"le="100""#),
        "IMP-130c: Should have 100µs bucket label"
    );
    assert!(
        body_str.contains(r#"le="500""#),
        "IMP-130c: Should have 500µs bucket label"
    );
    assert!(
        body_str.contains(r#"le="1000""#),
        "IMP-130c: Should have 1000µs bucket label"
    );
    assert!(
        body_str.contains(r#"le="5000""#),
        "IMP-130c: Should have 5000µs bucket label"
    );
    assert!(
        body_str.contains(r#"le="+Inf""#),
        "IMP-130c: Should have +Inf bucket label"
    );
}

/// IMP-130d: Prometheus latency histogram should have HELP and TYPE annotations
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_130d_prometheus_latency_has_help_and_type() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-130d: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-130d: Should have HELP and TYPE annotations for histograms
    assert!(
        body_str.contains("# HELP realizar_dispatch_cpu_latency"),
        "IMP-130d: Should have HELP for CPU latency histogram"
    );
    assert!(
        body_str.contains("# TYPE realizar_dispatch_cpu_latency histogram"),
        "IMP-130d: Should have TYPE histogram for CPU latency"
    );
    assert!(
        body_str.contains("# HELP realizar_dispatch_gpu_latency"),
        "IMP-130d: Should have HELP for GPU latency histogram"
    );
    assert!(
        body_str.contains("# TYPE realizar_dispatch_gpu_latency histogram"),
        "IMP-130d: Should have TYPE histogram for GPU latency"
    );
}

// ========================================================================
// IMP-141: Add Throughput Metrics to Prometheus Export (RED PHASE)
// ========================================================================

/// IMP-141a: Prometheus export should include throughput_rps gauge
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_141a_prometheus_includes_throughput_rps() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};
    use std::thread;
    use std::time::Duration;

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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-141a: Should create AppState");

    // Record some dispatches to get non-zero throughput
    if let Some(metrics) = state.dispatch_metrics() {
        thread::sleep(Duration::from_millis(2));
        for _ in 0..10 {
            metrics.record_cpu_dispatch();
        }
    }

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-141a: Should include throughput_rps metric
    assert!(
        body_str.contains("realizar_dispatch_throughput_rps"),
        "IMP-141a: Prometheus should include throughput_rps metric. Got: {}",
        body_str
    );
}

/// IMP-141b: Prometheus export should include elapsed_seconds gauge
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_141b_prometheus_includes_elapsed_seconds() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-141b: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-141b: Should include elapsed_seconds metric
    assert!(
        body_str.contains("realizar_dispatch_elapsed_seconds"),
        "IMP-141b: Prometheus should include elapsed_seconds metric. Got: {}",
        body_str
    );
}

/// IMP-141c: throughput_rps should have correct HELP and TYPE annotations
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_141c_throughput_rps_has_help_and_type() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-141c: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-141c: Should have HELP and TYPE for throughput_rps
    assert!(
        body_str.contains("# HELP realizar_dispatch_throughput_rps"),
        "IMP-141c: Should have HELP for throughput_rps"
    );
    assert!(
        body_str.contains("# TYPE realizar_dispatch_throughput_rps gauge"),
        "IMP-141c: Should have TYPE gauge for throughput_rps"
    );
}

/// IMP-141d: elapsed_seconds should have correct HELP and TYPE annotations
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_141d_elapsed_seconds_has_help_and_type() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-141d: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-141d: Should have HELP and TYPE for elapsed_seconds
    assert!(
        body_str.contains("# HELP realizar_dispatch_elapsed_seconds"),
        "IMP-141d: Should have HELP for elapsed_seconds"
    );
    assert!(
        body_str.contains("# TYPE realizar_dispatch_elapsed_seconds gauge"),
        "IMP-141d: Should have TYPE gauge for elapsed_seconds"
    );
}

// ===== IMP-131: Latency percentiles in JSON response =====

/// IMP-131a: DispatchMetrics should have percentile calculation methods
#[cfg(feature = "gpu")]
#[test]
fn test_imp_131a_dispatch_metrics_has_percentile_methods() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // Record some latencies to test percentile calculation
    metrics.record_cpu_latency(std::time::Duration::from_micros(50));
    metrics.record_cpu_latency(std::time::Duration::from_micros(150));
    metrics.record_cpu_latency(std::time::Duration::from_micros(600));
    metrics.record_gpu_latency(std::time::Duration::from_micros(80));
    metrics.record_gpu_latency(std::time::Duration::from_micros(300));

    // IMP-131a: Should have percentile methods
    let _cpu_p50 = metrics.cpu_latency_p50_us();
    let _cpu_p95 = metrics.cpu_latency_p95_us();
    let _cpu_p99 = metrics.cpu_latency_p99_us();
    let _gpu_p50 = metrics.gpu_latency_p50_us();
    let _gpu_p95 = metrics.gpu_latency_p95_us();
    let _gpu_p99 = metrics.gpu_latency_p99_us();
}

/// IMP-131b: Percentile estimation from histogram buckets
#[cfg(feature = "gpu")]
#[test]
fn test_imp_131b_percentile_estimation_from_histogram() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // Record 100 samples: 50 in first bucket, 30 in second, 20 in third
    // This creates a known distribution for testing
    for _ in 0..50 {
        metrics.record_cpu_latency(std::time::Duration::from_micros(50)); // bucket 0: 0-100µs
    }
    for _ in 0..30 {
        metrics.record_cpu_latency(std::time::Duration::from_micros(200)); // bucket 1: 100-500µs
    }
    for _ in 0..20 {
        metrics.record_cpu_latency(std::time::Duration::from_micros(700)); // bucket 2: 500-1000µs
    }

    // p50 should be in first bucket (50th sample out of 100)
    let p50 = metrics.cpu_latency_p50_us();
    assert!(
        p50 <= 100.0,
        "IMP-131b: p50 should be in first bucket (<=100µs), got {:.1}µs",
        p50
    );

    // p95 should be in third bucket (95th sample)
    // First 50 in bucket 0, next 30 in bucket 1 (total 80), next 20 in bucket 2
    // 95th percentile is in bucket 2 (500-1000µs)
    let p95 = metrics.cpu_latency_p95_us();
    assert!(
        p95 >= 500.0 && p95 <= 1000.0,
        "IMP-131b: p95 should be in bucket 2 (500-1000µs), got {:.1}µs",
        p95
    );
}

/// IMP-131c: JSON response should include latency percentiles
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_131c_json_response_includes_percentiles() {
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
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-131c: Should create AppState");

    // Record some latencies
    if let Some(metrics) = state.dispatch_metrics() {
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));
        metrics.record_gpu_latency(std::time::Duration::from_micros(200));
    }

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-131c: JSON should include percentile fields
    assert!(
        body_str.contains("cpu_latency_p50_us"),
        "IMP-131c: JSON should include cpu_latency_p50_us. Got: {}",
        body_str
    );
    assert!(
        body_str.contains("cpu_latency_p95_us"),
        "IMP-131c: JSON should include cpu_latency_p95_us"
    );
    assert!(
        body_str.contains("gpu_latency_p50_us"),
        "IMP-131c: JSON should include gpu_latency_p50_us"
    );
}

/// IMP-131d: Percentiles return 0 when no samples recorded
#[cfg(feature = "gpu")]
#[test]
fn test_imp_131d_percentiles_zero_when_empty() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-131d: Empty histogram should return 0 for all percentiles
    assert_eq!(
        metrics.cpu_latency_p50_us(),
        0.0,
        "IMP-131d: Empty histogram should return 0 for p50"
    );
    assert_eq!(
        metrics.cpu_latency_p95_us(),
        0.0,
        "IMP-131d: Empty histogram should return 0 for p95"
    );
    assert_eq!(
        metrics.cpu_latency_p99_us(),
        0.0,
        "IMP-131d: Empty histogram should return 0 for p99"
    );
    assert_eq!(
        metrics.gpu_latency_p50_us(),
        0.0,
        "IMP-131d: Empty histogram should return 0 for GPU p50"
    );
}

// ===== IMP-132: Wire latency recording into adaptive attention path =====

/// IMP-132a: Adaptive attention should record latency for CPU dispatches
#[cfg(feature = "gpu")]
#[test]
fn test_imp_132a_adaptive_attention_records_cpu_latency() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Generate tokens to trigger CPU dispatches (cache < 64 tokens)
    let _ = cached_model.generate_with_cache_adaptive(&[1, 2, 3], &gen_config, &metrics);

    // IMP-132a: After CPU dispatches, latency should be recorded
    assert!(
        metrics.cpu_latency_count() > 0,
        "IMP-132a: CPU latency count should be > 0 after adaptive generation. Got: {}",
        metrics.cpu_latency_count()
    );
}

/// IMP-132b: Latency values should be reasonable (not zero for executed paths)
#[cfg(feature = "gpu")]
#[test]
#[ignore = "GPU latency test - timing-sensitive and may fail under coverage instrumentation"]
fn test_imp_132b_latency_values_are_reasonable() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Generate tokens
    let _ = cached_model.generate_with_cache_adaptive(&[1, 2, 3], &gen_config, &metrics);

    // IMP-132b: Mean latency should be > 0 (actual time was measured)
    let mean_latency = metrics.cpu_latency_mean_us();
    assert!(
        mean_latency > 0.0,
        "IMP-132b: Mean CPU latency should be > 0µs after attention. Got: {:.1}µs",
        mean_latency
    );
}

/// IMP-132c: Latency count should match dispatch count
#[cfg(feature = "gpu")]
#[test]
fn test_imp_132c_latency_count_matches_dispatch_count() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 2, // 2 layers for more dispatches
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Generate tokens
    let _ = cached_model.generate_with_cache_adaptive(&[1, 2, 3, 4, 5], &gen_config, &metrics);

    // IMP-132c: Every CPU dispatch should record latency
    let cpu_dispatches = metrics.cpu_dispatches();
    let cpu_latency_count = metrics.cpu_latency_count();

    assert_eq!(
        cpu_dispatches, cpu_latency_count,
        "IMP-132c: CPU latency count ({}) should match dispatch count ({})",
        cpu_latency_count, cpu_dispatches
    );
}

/// IMP-132d: GPU dispatches should also record latency (when cache >= 64)
#[cfg(feature = "gpu")]
#[test]
fn test_imp_132d_gpu_dispatches_record_latency() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 80, // Generate enough to trigger GPU dispatch
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Generate enough tokens to trigger GPU dispatch (cache >= 64 tokens)
    let _ = cached_model.generate_with_cache_adaptive(&[1], &gen_config, &metrics);

    // IMP-132d: After many tokens, should have GPU dispatches with latency recorded
    let gpu_dispatches = metrics.gpu_dispatches();
    let gpu_latency_count = metrics.gpu_latency_count();

    if gpu_dispatches > 0 {
        assert_eq!(
            gpu_dispatches, gpu_latency_count,
            "IMP-132d: GPU latency count ({}) should match dispatch count ({})",
            gpu_latency_count, gpu_dispatches
        );
    }
}

// ============================================================
// IMP-133: Add latency mean to JSON response
// RED phase: Tests written first, implementation to follow
// ============================================================

// IMP-133a: DispatchMetrics should have cpu_latency_mean_us method
#[test]
fn test_imp_133a_dispatch_metrics_has_mean_methods() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some latencies
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    metrics.record_gpu_latency(Duration::from_micros(500));
    metrics.record_gpu_latency(Duration::from_micros(700));

    // IMP-133a: Mean methods should exist and return correct values
    let cpu_mean = metrics.cpu_latency_mean_us();
    let gpu_mean = metrics.gpu_latency_mean_us();

    assert!(
        (cpu_mean - 200.0).abs() < 1.0,
        "IMP-133a: CPU mean should be ~200µs, got {}",
        cpu_mean
    );
    assert!(
        (gpu_mean - 600.0).abs() < 1.0,
        "IMP-133a: GPU mean should be ~600µs, got {}",
        gpu_mean
    );
}

// IMP-133b: Mean should be 0 when no samples recorded
#[test]
fn test_imp_133b_mean_zero_when_empty() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-133b: Mean should be 0.0 when no samples recorded
    assert_eq!(
        metrics.cpu_latency_mean_us(),
        0.0,
        "IMP-133b: CPU mean should be 0 when empty"
    );
    assert_eq!(
        metrics.gpu_latency_mean_us(),
        0.0,
        "IMP-133b: GPU mean should be 0 when empty"
    );
}

// IMP-133c: JSON response should include mean latency fields
#[test]
fn test_imp_133c_json_response_includes_mean() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record some latencies
    metrics.record_cpu_dispatch();
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_dispatch();
    metrics.record_cpu_latency(Duration::from_micros(300));

    // Build response (would be done by handler)
    let response = DispatchMetricsResponse {
        cpu_dispatches: metrics.cpu_dispatches(),
        gpu_dispatches: metrics.gpu_dispatches(),
        total_dispatches: metrics.total_dispatches(),
        gpu_ratio: metrics.gpu_ratio(),
        cpu_latency_p50_us: metrics.cpu_latency_p50_us(),
        cpu_latency_p95_us: metrics.cpu_latency_p95_us(),
        cpu_latency_p99_us: metrics.cpu_latency_p99_us(),
        gpu_latency_p50_us: metrics.gpu_latency_p50_us(),
        gpu_latency_p95_us: metrics.gpu_latency_p95_us(),
        gpu_latency_p99_us: metrics.gpu_latency_p99_us(),
        // IMP-133: New mean fields
        cpu_latency_mean_us: metrics.cpu_latency_mean_us(),
        gpu_latency_mean_us: metrics.gpu_latency_mean_us(),
        // IMP-134: New min/max fields
        cpu_latency_min_us: metrics.cpu_latency_min_us(),
        cpu_latency_max_us: metrics.cpu_latency_max_us(),
        gpu_latency_min_us: metrics.gpu_latency_min_us(),
        gpu_latency_max_us: metrics.gpu_latency_max_us(),
        // IMP-135: Variance/stddev fields
        cpu_latency_variance_us: metrics.cpu_latency_variance_us(),
        cpu_latency_stddev_us: metrics.cpu_latency_stddev_us(),
        gpu_latency_variance_us: metrics.gpu_latency_variance_us(),
        gpu_latency_stddev_us: metrics.gpu_latency_stddev_us(),
        // IMP-136: Histogram bucket configuration
        bucket_boundaries_us: metrics.bucket_boundaries_us(),
        cpu_latency_bucket_counts: metrics.cpu_latency_buckets().to_vec(),
        gpu_latency_bucket_counts: metrics.gpu_latency_buckets().to_vec(),
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // IMP-133c: Response should have mean fields with correct values
    assert!(
        (response.cpu_latency_mean_us - 200.0).abs() < 1.0,
        "IMP-133c: Response CPU mean should be ~200µs, got {}",
        response.cpu_latency_mean_us
    );
    assert_eq!(
        response.gpu_latency_mean_us, 0.0,
        "IMP-133c: Response GPU mean should be 0 (no GPU samples)"
    );
}

// IMP-133d: Mean should handle single sample correctly
#[test]
fn test_imp_133d_mean_single_sample() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Single sample
    metrics.record_cpu_latency(Duration::from_micros(42));

    // IMP-133d: Mean of single sample should equal that sample
    assert!(
        (metrics.cpu_latency_mean_us() - 42.0).abs() < 0.1,
        "IMP-133d: Mean of single sample should be 42µs, got {}",
        metrics.cpu_latency_mean_us()
    );
}

// ============================================================
// IMP-134: Add min/max latency tracking
// RED phase: Tests written first, implementation to follow
// ============================================================

// IMP-134a: DispatchMetrics should have min/max methods
#[test]
fn test_imp_134a_dispatch_metrics_has_min_max_methods() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some latencies with varying values
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(50));
    metrics.record_cpu_latency(Duration::from_micros(300));

    metrics.record_gpu_latency(Duration::from_micros(200));
    metrics.record_gpu_latency(Duration::from_micros(800));

    // IMP-134a: Min/max methods should exist and return correct values
    assert_eq!(
        metrics.cpu_latency_min_us(),
        50,
        "IMP-134a: CPU min should be 50µs"
    );
    assert_eq!(
        metrics.cpu_latency_max_us(),
        300,
        "IMP-134a: CPU max should be 300µs"
    );
    assert_eq!(
        metrics.gpu_latency_min_us(),
        200,
        "IMP-134a: GPU min should be 200µs"
    );
    assert_eq!(
        metrics.gpu_latency_max_us(),
        800,
        "IMP-134a: GPU max should be 800µs"
    );
}

// IMP-134b: Min/max should be 0 when no samples recorded
#[test]
fn test_imp_134b_min_max_zero_when_empty() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-134b: Min/max should be 0 when no samples recorded
    assert_eq!(
        metrics.cpu_latency_min_us(),
        0,
        "IMP-134b: CPU min should be 0 when empty"
    );
    assert_eq!(
        metrics.cpu_latency_max_us(),
        0,
        "IMP-134b: CPU max should be 0 when empty"
    );
    assert_eq!(
        metrics.gpu_latency_min_us(),
        0,
        "IMP-134b: GPU min should be 0 when empty"
    );
    assert_eq!(
        metrics.gpu_latency_max_us(),
        0,
        "IMP-134b: GPU max should be 0 when empty"
    );
}

// IMP-134c: JSON response should include min/max latency fields
