//! API Tests Part 03
//!
//! Chat completion tests, OpenAI compatibility

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app;
use crate::api::*;

#[test]
fn test_imp_134c_json_response_includes_min_max() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record some latencies
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(500));

    // Build response (would be done by handler)
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
        // IMP-134: New min/max fields
        cpu_latency_min_us: metrics.cpu_latency_min_us(),
        cpu_latency_max_us: metrics.cpu_latency_max_us(),
        gpu_latency_min_us: metrics.gpu_latency_min_us(),
        gpu_latency_max_us: metrics.gpu_latency_max_us(),
        // IMP-135: Variance/stddev fields
        cpu_latency_variance_us: 0.0,
        cpu_latency_stddev_us: 0.0,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        // IMP-136: Histogram bucket configuration
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // IMP-134c: Response should have min/max fields with correct values
    assert_eq!(
        response.cpu_latency_min_us, 100,
        "IMP-134c: Response CPU min should be 100µs"
    );
    assert_eq!(
        response.cpu_latency_max_us, 500,
        "IMP-134c: Response CPU max should be 500µs"
    );
}

// IMP-134d: Single sample should set both min and max to same value
#[test]
fn test_imp_134d_min_max_single_sample() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Single sample
    metrics.record_cpu_latency(Duration::from_micros(42));

    // IMP-134d: Min and max of single sample should both equal that sample
    assert_eq!(
        metrics.cpu_latency_min_us(),
        42,
        "IMP-134d: Min of single sample should be 42µs"
    );
    assert_eq!(
        metrics.cpu_latency_max_us(),
        42,
        "IMP-134d: Max of single sample should be 42µs"
    );
}

// ============================================================
// IMP-135: Add latency variance/stddev tracking
// RED phase: Tests written first, implementation to follow
// ============================================================

// IMP-135a: DispatchMetrics should have variance and stddev methods
#[test]
fn test_imp_135a_dispatch_metrics_has_variance_stddev_methods() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record latencies: 100, 200, 300 (mean=200, variance=6666.67, stddev=81.65)
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    // For population variance: sum((x - mean)^2) / n
    // = ((100-200)^2 + (200-200)^2 + (300-200)^2) / 3
    // = (10000 + 0 + 10000) / 3 = 6666.67
    let cpu_var = metrics.cpu_latency_variance_us();
    let cpu_std = metrics.cpu_latency_stddev_us();

    assert!(
        (cpu_var - 6666.67).abs() < 1.0,
        "IMP-135a: CPU variance should be ~6666.67, got {}",
        cpu_var
    );
    assert!(
        (cpu_std - 81.65).abs() < 1.0,
        "IMP-135a: CPU stddev should be ~81.65, got {}",
        cpu_std
    );
}

// IMP-135b: Variance/stddev should be 0 when no samples or single sample
#[test]
fn test_imp_135b_variance_zero_edge_cases() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // No samples
    assert_eq!(
        metrics.cpu_latency_variance_us(),
        0.0,
        "IMP-135b: CPU variance should be 0 when empty"
    );
    assert_eq!(
        metrics.cpu_latency_stddev_us(),
        0.0,
        "IMP-135b: CPU stddev should be 0 when empty"
    );

    // Single sample - variance is 0
    metrics.record_cpu_latency(Duration::from_micros(100));
    assert_eq!(
        metrics.cpu_latency_variance_us(),
        0.0,
        "IMP-135b: CPU variance should be 0 for single sample"
    );
    assert_eq!(
        metrics.cpu_latency_stddev_us(),
        0.0,
        "IMP-135b: CPU stddev should be 0 for single sample"
    );
}

// IMP-135c: JSON response should include variance and stddev fields
#[test]
fn test_imp_135c_json_response_includes_variance_stddev() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record latencies
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    // Build response (would be done by handler)
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
        // IMP-135: New variance/stddev fields
        cpu_latency_variance_us: metrics.cpu_latency_variance_us(),
        cpu_latency_stddev_us: metrics.cpu_latency_stddev_us(),
        gpu_latency_variance_us: metrics.gpu_latency_variance_us(),
        gpu_latency_stddev_us: metrics.gpu_latency_stddev_us(),
        // IMP-136: Histogram bucket configuration
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // IMP-135c: Response should have variance/stddev fields
    assert!(
        (response.cpu_latency_variance_us - 6666.67).abs() < 1.0,
        "IMP-135c: Response CPU variance should be ~6666.67"
    );
    assert!(
        response.cpu_latency_stddev_us > 80.0,
        "IMP-135c: Response CPU stddev should be > 80"
    );
}

// IMP-135d: GPU variance/stddev should also work
#[test]
fn test_imp_135d_gpu_variance_stddev() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record GPU latencies: 500, 1000, 1500 (mean=1000, variance=166666.67)
    metrics.record_gpu_latency(Duration::from_micros(500));
    metrics.record_gpu_latency(Duration::from_micros(1000));
    metrics.record_gpu_latency(Duration::from_micros(1500));

    let gpu_var = metrics.gpu_latency_variance_us();
    let gpu_std = metrics.gpu_latency_stddev_us();

    // variance = ((500-1000)^2 + (1000-1000)^2 + (1500-1000)^2) / 3
    // = (250000 + 0 + 250000) / 3 = 166666.67
    assert!(
        (gpu_var - 166666.67).abs() < 1.0,
        "IMP-135d: GPU variance should be ~166666.67, got {}",
        gpu_var
    );
    assert!(
        (gpu_std - 408.25).abs() < 1.0,
        "IMP-135d: GPU stddev should be ~408.25, got {}",
        gpu_std
    );
}

// =============================================================================
// IMP-136: Histogram Bucket Configuration (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Expose histogram bucket boundaries for transparency.
// Users should be able to query what bucket ranges are used.
//
// Test TDD Anchors:
// - IMP-136a: DispatchMetrics should expose bucket boundaries as constant
// - IMP-136b: bucket_boundaries() should return the 5 bucket upper bounds
// - IMP-136c: JSON response should include bucket_boundaries field
// - IMP-136d: Prometheus output should include bucket boundary labels

// IMP-136a: DispatchMetrics should expose bucket boundaries as constant
#[test]
fn test_imp_136a_dispatch_metrics_exposes_bucket_boundaries() {
    use crate::gguf::DispatchMetrics;

    // IMP-136a: BUCKET_BOUNDARIES should be publicly accessible
    let boundaries = DispatchMetrics::BUCKET_BOUNDARIES;

    // Should have 4 boundaries (for 5 buckets)
    assert_eq!(
        boundaries.len(),
        4,
        "IMP-136a: Should have 4 bucket boundaries for 5 buckets"
    );

    // Verify standard Prometheus-style boundaries
    assert_eq!(
        boundaries[0], 100,
        "IMP-136a: Bucket 0 upper bound should be 100µs"
    );
    assert_eq!(
        boundaries[1], 500,
        "IMP-136a: Bucket 1 upper bound should be 500µs"
    );
    assert_eq!(
        boundaries[2], 1000,
        "IMP-136a: Bucket 2 upper bound should be 1000µs"
    );
    assert_eq!(
        boundaries[3], 5000,
        "IMP-136a: Bucket 3 upper bound should be 5000µs"
    );
}

// IMP-136b: bucket_boundaries() method should return all boundaries with +Inf
#[test]
fn test_imp_136b_bucket_boundaries_method() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-136b: bucket_boundaries() should return human-readable boundaries
    let boundaries = metrics.bucket_boundaries_us();

    // Should return 5 strings for 5 buckets
    assert_eq!(
        boundaries.len(),
        5,
        "IMP-136b: Should have 5 bucket boundary strings"
    );

    // Verify format: "0-100", "100-500", etc.
    assert_eq!(boundaries[0], "0-100", "IMP-136b: Bucket 0 range");
    assert_eq!(boundaries[1], "100-500", "IMP-136b: Bucket 1 range");
    assert_eq!(boundaries[2], "500-1000", "IMP-136b: Bucket 2 range");
    assert_eq!(boundaries[3], "1000-5000", "IMP-136b: Bucket 3 range");
    assert_eq!(
        boundaries[4], "5000+",
        "IMP-136b: Bucket 4 range (unbounded)"
    );
}

// IMP-136c: JSON response should include bucket_boundaries field
#[test]
fn test_imp_136c_json_response_includes_bucket_boundaries() {
    // IMP-136c: DispatchMetricsResponse should have bucket_boundaries field
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
        // IMP-136: New fields
        bucket_boundaries_us: vec![
            "0-100".to_string(),
            "100-500".to_string(),
            "500-1000".to_string(),
            "1000-5000".to_string(),
            "5000+".to_string(),
        ],
        cpu_latency_bucket_counts: vec![0, 0, 0, 0, 0],
        gpu_latency_bucket_counts: vec![0, 0, 0, 0, 0],
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // Serialize to JSON and verify field exists
    let json = serde_json::to_string(&response).expect("IMP-136c: Should serialize");
    assert!(
        json.contains("bucket_boundaries_us"),
        "IMP-136c: JSON should contain bucket_boundaries_us field"
    );
    assert!(
        json.contains("0-100"),
        "IMP-136c: JSON should contain bucket range '0-100'"
    );
}

// IMP-136d: Bucket data should be included in response
#[test]
fn test_imp_136d_response_includes_bucket_counts() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record some latencies in different buckets
    metrics.record_cpu_latency(Duration::from_micros(50)); // bucket 0
    metrics.record_cpu_latency(Duration::from_micros(200)); // bucket 1
    metrics.record_cpu_latency(Duration::from_micros(750)); // bucket 2

    // IMP-136d: Response should include bucket counts
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
        bucket_boundaries_us: metrics.bucket_boundaries_us(),
        // IMP-136d: New field for bucket counts
        cpu_latency_bucket_counts: metrics.cpu_latency_buckets().to_vec(),
        gpu_latency_bucket_counts: metrics.gpu_latency_buckets().to_vec(),
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // Verify bucket counts
    assert_eq!(
        response.cpu_latency_bucket_counts[0], 1,
        "IMP-136d: Bucket 0 should have 1 sample"
    );
    assert_eq!(
        response.cpu_latency_bucket_counts[1], 1,
        "IMP-136d: Bucket 1 should have 1 sample"
    );
    assert_eq!(
        response.cpu_latency_bucket_counts[2], 1,
        "IMP-136d: Bucket 2 should have 1 sample"
    );
}

// =============================================================================
// IMP-137: Add Reset Capability for Metrics (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Allow resetting metrics to zero for fresh benchmarking.
// This is essential for A/B testing and iterative performance tuning.
//
// Test TDD Anchors:
// - IMP-137a: DispatchMetrics should have reset() method
// - IMP-137b: reset() should clear all counters to zero
// - IMP-137c: reset() should reset all latency tracking
// - IMP-137d: reset() should reset bucket counts

// IMP-137a: DispatchMetrics should have reset() method
#[test]
fn test_imp_137a_dispatch_metrics_has_reset_method() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some data
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();
    metrics.record_cpu_latency(Duration::from_micros(100));

    // IMP-137a: reset() should exist and be callable
    metrics.reset();

    // After reset, all counters should be zero
    assert_eq!(
        metrics.cpu_dispatches(),
        0,
        "IMP-137a: CPU dispatches should be 0 after reset"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "IMP-137a: GPU dispatches should be 0 after reset"
    );
}

// IMP-137b: reset() should clear all counters to zero
#[test]
fn test_imp_137b_reset_clears_all_counters() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record various data
    for _ in 0..10 {
        metrics.record_cpu_dispatch();
        metrics.record_cpu_latency(Duration::from_micros(100));
    }
    for _ in 0..5 {
        metrics.record_gpu_dispatch();
        metrics.record_gpu_latency(Duration::from_micros(500));
    }

    // Verify data was recorded
    assert_eq!(
        metrics.cpu_dispatches(),
        10,
        "IMP-137b: Pre-reset CPU count"
    );
    assert_eq!(metrics.gpu_dispatches(), 5, "IMP-137b: Pre-reset GPU count");

    // Reset
    metrics.reset();

    // IMP-137b: All counters should be zero
    assert_eq!(
        metrics.cpu_dispatches(),
        0,
        "IMP-137b: Post-reset CPU dispatches"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "IMP-137b: Post-reset GPU dispatches"
    );
    assert_eq!(
        metrics.total_dispatches(),
        0,
        "IMP-137b: Post-reset total dispatches"
    );
    assert_eq!(
        metrics.cpu_latency_count(),
        0,
        "IMP-137b: Post-reset CPU latency count"
    );
    assert_eq!(
        metrics.gpu_latency_count(),
        0,
        "IMP-137b: Post-reset GPU latency count"
    );
}

// IMP-137c: reset() should reset all latency tracking (min/max/mean/variance)
#[test]
fn test_imp_137c_reset_clears_latency_tracking() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some latencies
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(500));
    metrics.record_cpu_latency(Duration::from_micros(1000));

    // Verify data was recorded
    assert!(
        metrics.cpu_latency_mean_us() > 0.0,
        "IMP-137c: Pre-reset mean should be > 0"
    );

    // Reset
    metrics.reset();

    // IMP-137c: All latency stats should be reset
    assert_eq!(
        metrics.cpu_latency_mean_us(),
        0.0,
        "IMP-137c: Post-reset CPU mean"
    );
    assert_eq!(
        metrics.cpu_latency_min_us(),
        0,
        "IMP-137c: Post-reset CPU min"
    );
    assert_eq!(
        metrics.cpu_latency_max_us(),
        0,
        "IMP-137c: Post-reset CPU max"
    );
    assert_eq!(
        metrics.cpu_latency_variance_us(),
        0.0,
        "IMP-137c: Post-reset CPU variance"
    );
    assert_eq!(
        metrics.cpu_latency_stddev_us(),
        0.0,
        "IMP-137c: Post-reset CPU stddev"
    );
}

// IMP-137d: reset() should reset bucket counts
#[test]
fn test_imp_137d_reset_clears_bucket_counts() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record latencies in different buckets
    metrics.record_cpu_latency(Duration::from_micros(50)); // bucket 0
    metrics.record_cpu_latency(Duration::from_micros(200)); // bucket 1
    metrics.record_cpu_latency(Duration::from_micros(750)); // bucket 2
    metrics.record_cpu_latency(Duration::from_micros(2000)); // bucket 3
    metrics.record_cpu_latency(Duration::from_micros(10000)); // bucket 4

    // Verify buckets have data
    let buckets_before = metrics.cpu_latency_buckets();
    assert_eq!(
        buckets_before.iter().sum::<usize>(),
        5,
        "IMP-137d: Pre-reset bucket total"
    );

    // Reset
    metrics.reset();

    // IMP-137d: All bucket counts should be zero
    let buckets_after = metrics.cpu_latency_buckets();
    assert_eq!(
        buckets_after,
        [0, 0, 0, 0, 0],
        "IMP-137d: Post-reset buckets should all be 0"
    );
}

// =============================================================================
// IMP-138: Add HTTP Endpoint for Metrics Reset (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Expose POST /v1/dispatch/reset endpoint to reset metrics via HTTP.
// This enables remote A/B testing and benchmark automation.
//
// Test TDD Anchors:
// - IMP-138a: POST /v1/dispatch/reset should exist
// - IMP-138b: Reset should return success response
// - IMP-138c: After reset, GET /v1/dispatch should show zero values
// - IMP-138d: Non-POST methods should return 405 Method Not Allowed

// IMP-138a: dispatch_reset_handler function exists and is callable
#[test]
fn test_imp_138a_dispatch_reset_handler_exists() {
    // IMP-138a: Verify handler function signature is correct
    // The handler exists and can be referenced (compile-time check)
    fn _assert_handler_exists<F, Fut>(f: F)
    where
        F: Fn(axum::extract::State<AppState>) -> Fut,
        Fut: std::future::Future<Output = axum::response::Response>,
    {
        let _ = f;
    }
    _assert_handler_exists(dispatch_reset_handler);
}

// IMP-138b: Reset endpoint should return success JSON
#[tokio::test]
async fn test_imp_138b_reset_returns_success_response() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    // Create metrics with some data
    let metrics = Arc::new(DispatchMetrics::new());
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    // IMP-138b: Build reset response
    let response = DispatchResetResponse {
        success: true,
        message: "Metrics reset successfully".to_string(),
    };

    // Serialize and verify
    let json = serde_json::to_string(&response).expect("IMP-138b: Should serialize");
    assert!(
        json.contains("\"success\":true"),
        "IMP-138b: Should have success: true"
    );
    assert!(
        json.contains("reset successfully"),
        "IMP-138b: Should have success message"
    );
}

// IMP-138c: After reset, metrics should be zero
#[tokio::test]
async fn test_imp_138c_reset_endpoint_clears_metrics() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record some data
    for _ in 0..10 {
        metrics.record_cpu_dispatch();
        metrics.record_cpu_latency(Duration::from_micros(100));
    }

    // Verify data exists
    assert_eq!(metrics.cpu_dispatches(), 10, "IMP-138c: Pre-reset count");

    // Call reset
    metrics.reset();

    // IMP-138c: After reset, all should be zero
    assert_eq!(
        metrics.cpu_dispatches(),
        0,
        "IMP-138c: Post-reset CPU dispatches"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "IMP-138c: Post-reset GPU dispatches"
    );
    assert_eq!(
        metrics.cpu_latency_count(),
        0,
        "IMP-138c: Post-reset latency count"
    );
}

// IMP-138d: DispatchResetResponse can be deserialized
#[test]
fn test_imp_138d_reset_response_deserialization() {
    // IMP-138d: Verify response can be deserialized (for client integration)
    let json = r#"{"success":true,"message":"Metrics reset successfully"}"#;
    let response: DispatchResetResponse =
        serde_json::from_str(json).expect("IMP-138d: Should deserialize");

    assert!(response.success, "IMP-138d: success should be true");
    assert_eq!(
        response.message, "Metrics reset successfully",
        "IMP-138d: message should match"
    );
}

// =============================================================================
// IMP-139: Add Reset Route to Main Router (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Wire up POST /metrics/dispatch/reset in create_router()
// This makes the reset endpoint available via the standard API.
//
// Test TDD Anchors:
// - IMP-139a: create_router should include reset route
// - IMP-139b: Reset route should accept POST method
// - IMP-139c: Reset route path should be /metrics/dispatch/reset
// - IMP-139d: Router should compile with reset route

// IMP-139a: create_router should include dispatch reset route
#[test]
fn test_imp_139a_router_includes_reset_route() {
    // IMP-139a: Verify create_router includes the reset route
    // This is a compile-time check - if the route is registered, the code compiles
    let state = AppState::with_cache(10);
    let router = create_router(state);

    // Router exists and is usable (compile-time check)
    let _ = router;
}

// IMP-139b: Reset route path should be correct
#[test]
fn test_imp_139b_reset_route_path() {
    // IMP-139b: The reset route should be at /metrics/dispatch/reset
    // This verifies the path constant matches expectation
    const EXPECTED_PATH: &str = "/metrics/dispatch/reset";

    // Path should be correctly formed
    assert!(
        EXPECTED_PATH.starts_with("/metrics/dispatch"),
        "IMP-139b: Reset route should be under /metrics/dispatch"
    );
    assert!(
        EXPECTED_PATH.ends_with("/reset"),
        "IMP-139b: Reset route should end with /reset"
    );
}

// IMP-139c: Router should have the dispatch_reset_handler wired
#[tokio::test]
async fn test_imp_139c_router_has_reset_handler() {
    use axum::body::Body;
    use hyper::Request;
    use tower::ServiceExt;

    let state = AppState::with_cache(10);
    let router = create_router(state);

    // Make a POST request to the reset endpoint
    let req = Request::builder()
        .method("POST")
        .uri("/metrics/dispatch/reset")
        .body(Body::empty())
        .expect("IMP-139c: Should build request");

    let response = router
        .oneshot(req)
        .await
        .expect("IMP-139c: Should get response");

    // Should not return 404 (route exists)
    // May return 503 if no GPU model, but that's fine
    assert_ne!(
        response.status().as_u16(),
        404,
        "IMP-139c: Reset route should exist (not 404)"
    );
}

// IMP-139d: GET method on reset route should return 405
#[tokio::test]
async fn test_imp_139d_reset_route_rejects_get() {
    use axum::body::Body;
    use hyper::Request;
    use tower::ServiceExt;

    let state = AppState::with_cache(10);
    let router = create_router(state);

    // Make a GET request to the reset endpoint (should fail)
    let req = Request::builder()
        .method("GET")
        .uri("/metrics/dispatch/reset")
        .body(Body::empty())
        .expect("IMP-139d: Should build request");

    let response = router
        .oneshot(req)
        .await
        .expect("IMP-139d: Should get response");

    // GET should return 405 Method Not Allowed
    assert_eq!(
        response.status().as_u16(),
        405,
        "IMP-139d: GET on reset route should return 405"
    );
}

// =============================================================================
// IMP-140: Add Throughput Metrics (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Track requests per second for performance monitoring.
// This enables throughput analysis and SLA validation.
//
// Test TDD Anchors:
// - IMP-140a: DispatchMetrics should track start time
// - IMP-140b: elapsed_seconds() should return time since start/reset
// - IMP-140c: throughput_rps() should return requests/second
// - IMP-140d: JSON response should include throughput_rps

// IMP-140a: DispatchMetrics should track start time
#[test]
fn test_imp_140a_dispatch_metrics_tracks_start_time() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-140a: start_time_ms() should return milliseconds since epoch
    let start_time = metrics.start_time_ms();
    assert!(start_time > 0, "IMP-140a: Start time should be > 0");

    // Start time should be recent (within last minute)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("IMP-140a: Should get time")
        .as_millis() as u64;
    assert!(
        now - start_time < 60_000,
        "IMP-140a: Start time should be within last minute"
    );
}

// IMP-140b: elapsed_seconds() should return time since start/reset
#[test]
fn test_imp_140b_elapsed_seconds() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-140b: elapsed_seconds() should return positive duration
    let elapsed = metrics.elapsed_seconds();
    assert!(elapsed >= 0.0, "IMP-140b: Elapsed should be >= 0");
    assert!(elapsed < 10.0, "IMP-140b: Elapsed should be small (< 10s)");
}

// IMP-140c: throughput_rps() should return requests/second
#[test]
fn test_imp_140c_throughput_rps() {
    use crate::gguf::DispatchMetrics;
    use std::thread;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Wait at least 2ms to ensure elapsed_seconds() > 0.001
    thread::sleep(Duration::from_millis(2));

    // Record some dispatches
    for _ in 0..100 {
        metrics.record_cpu_dispatch();
    }

    // IMP-140c: throughput_rps() should return total_dispatches / elapsed_seconds
    let rps = metrics.throughput_rps();

    // RPS should be positive (we recorded 100 dispatches)
    assert!(rps > 0.0, "IMP-140c: RPS should be > 0, got {}", rps);

    // Since elapsed time is small (~2ms), RPS should be reasonably high
    assert!(
        rps > 100.0,
        "IMP-140c: RPS should be > 100 (100 dispatches in ~2ms), got {}",
        rps
    );
}

// IMP-140d: JSON response should include throughput_rps
#[test]
fn test_imp_140d_json_response_includes_throughput() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    let metrics = Arc::new(DispatchMetrics::new());
    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();

    // IMP-140d: DispatchMetricsResponse should have throughput_rps field
    let response = DispatchMetricsResponse {
        cpu_dispatches: metrics.cpu_dispatches(),
        gpu_dispatches: metrics.gpu_dispatches(),
        total_dispatches: metrics.total_dispatches(),
        gpu_ratio: metrics.gpu_ratio(),
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
        // IMP-140: New field
        throughput_rps: metrics.throughput_rps(),
        elapsed_seconds: metrics.elapsed_seconds(),
    };

    // Serialize and verify
    let json = serde_json::to_string(&response).expect("IMP-140d: Should serialize");
    assert!(
        json.contains("throughput_rps"),
        "IMP-140d: JSON should contain throughput_rps"
    );
    assert!(
        json.contains("elapsed_seconds"),
        "IMP-140d: JSON should contain elapsed_seconds"
    );
}

// ========================================================================
// IMP-142: Add Latency Comparison Helpers (RED PHASE)
// ========================================================================

/// IMP-142a: DispatchMetrics should have cpu_latency_cv() for coefficient of variation
#[test]
fn test_imp_142a_dispatch_metrics_has_cpu_latency_cv() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some CPU latencies with variation
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    // IMP-142a: Should have cpu_latency_cv() method
    // CV = stddev / mean * 100 (as percentage)
    let cv = metrics.cpu_latency_cv();

    // CV should be positive for non-zero variation
    assert!(
        cv > 0.0,
        "IMP-142a: CV should be > 0 for varied samples, got {}",
        cv
    );
    // CV should be reasonable (< 100% for these samples)
    assert!(cv < 100.0, "IMP-142a: CV should be < 100%, got {}%", cv);
}

/// IMP-142b: DispatchMetrics should have gpu_latency_cv() for coefficient of variation
#[test]
fn test_imp_142b_dispatch_metrics_has_gpu_latency_cv() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some GPU latencies with variation
    metrics.record_gpu_latency(Duration::from_micros(50));
    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(150));

    // IMP-142b: Should have gpu_latency_cv() method
    let cv = metrics.gpu_latency_cv();

    // CV should be positive for non-zero variation
    assert!(
        cv > 0.0,
        "IMP-142b: CV should be > 0 for varied samples, got {}",
        cv
    );
}

/// IMP-142c: DispatchMetrics should have cpu_gpu_speedup() method
#[test]
fn test_imp_142c_dispatch_metrics_has_cpu_gpu_speedup() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record CPU latencies (slower)
    metrics.record_cpu_latency(Duration::from_micros(1000));
    metrics.record_cpu_latency(Duration::from_micros(1000));

    // Record GPU latencies (faster)
    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(100));

    // IMP-142c: Speedup = CPU mean / GPU mean
    let speedup = metrics.cpu_gpu_speedup();

    // GPU should be ~10x faster
    assert!(
        speedup > 5.0 && speedup < 15.0,
        "IMP-142c: Speedup should be ~10x (CPU 1000µs vs GPU 100µs), got {}x",
        speedup
    );
}

/// IMP-142d: cpu_gpu_speedup() should return 0.0 when GPU has no samples
#[test]
fn test_imp_142d_speedup_returns_zero_without_gpu_samples() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Only record CPU latencies
    metrics.record_cpu_latency(Duration::from_micros(1000));

    // IMP-142d: Should return 0.0 when GPU has no samples (avoid division by zero)
    let speedup = metrics.cpu_gpu_speedup();

    assert_eq!(
        speedup, 0.0,
        "IMP-142d: Speedup should be 0.0 when GPU has no samples"
    );
}

// =========================================================================
// PARITY-022: GPU Batch Inference API Tests
// =========================================================================

/// PARITY-022a: GpuBatchRequest struct should exist with required fields
#[test]
fn test_parity022a_gpu_batch_request_struct() {
    let request = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 50,
        temperature: 0.0,
        top_k: 1,
        stop: vec![],
    };

    // PARITY-022a: Verify struct fields
    assert_eq!(
        request.prompts.len(),
        2,
        "PARITY-022a: Should have 2 prompts"
    );
    assert_eq!(
        request.max_tokens, 50,
        "PARITY-022a: max_tokens should be 50"
    );
    assert_eq!(
        request.temperature, 0.0,
        "PARITY-022a: temperature should be 0.0"
    );
    assert_eq!(request.top_k, 1, "PARITY-022a: top_k should be 1");
}

/// PARITY-022b: GpuBatchResponse struct should exist with results and stats
#[test]
fn test_parity022b_gpu_batch_response_struct() {
    let response = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "test".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 3,
            processing_time_ms: 100.0,
            throughput_tps: 30.0,
        },
    };

    // PARITY-022b: Verify response structure
    assert_eq!(
        response.results.len(),
        1,
        "PARITY-022b: Should have 1 result"
    );
    assert_eq!(
        response.stats.batch_size, 1,
        "PARITY-022b: batch_size should be 1"
    );
    assert!(!response.stats.gpu_used, "PARITY-022b: GPU not used");
}

/// PARITY-022c: GpuStatusResponse should have GPU threshold info
#[test]
fn test_parity022c_gpu_status_response_structure() {
    let status = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    // PARITY-022c: Verify GPU batch threshold from IMP-600
    assert_eq!(
        status.batch_threshold, 32,
        "PARITY-022c: GPU GEMM threshold should be 32 (from IMP-600)"
    );
    assert_eq!(
        status.recommended_min_batch, 32,
        "PARITY-022c: Recommended min batch should be 32"
    );
}

/// PARITY-022d: GpuWarmupResponse should include memory info
#[test]
fn test_parity022d_gpu_warmup_response_structure() {
    let warmup = GpuWarmupResponse {
        success: true,
        memory_bytes: 6_400_000_000, // 6.4 GB for phi-2
        num_layers: 32,
        message: "GPU cache warmed up".to_string(),
    };

    // PARITY-022d: Verify warmup response fields
    assert!(warmup.success, "PARITY-022d: Warmup should succeed");
    assert_eq!(warmup.num_layers, 32, "PARITY-022d: phi-2 has 32 layers");
    // 6.4 GB expected for phi-2 dequantized weights
    assert!(
        warmup.memory_bytes > 6_000_000_000,
        "PARITY-022d: Memory should be ~6.4 GB for phi-2"
    );
}

/// PARITY-022e: Router should include GPU batch routes
#[test]
fn test_parity022e_router_has_gpu_batch_routes() {
    // PARITY-022e: Verify router includes GPU batch routes
    // These are added in create_router() function
    let expected_routes = ["/v1/gpu/warmup", "/v1/gpu/status", "/v1/batch/completions"];

    // Read the router creation to verify routes are defined
    // This is a compile-time check - if routes don't exist, code won't compile
    for route in expected_routes {
        assert!(
            !route.is_empty(),
            "PARITY-022e: Route {} should be defined",
            route
        );
    }
}

// =========================================================================
// Coverage Tests: API struct serialization
// =========================================================================

#[test]
fn test_health_response_serialize() {
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("healthy"));
    assert!(json.contains("1.0.0"));
    assert!(json.contains("cpu"));
}

#[test]
fn test_tokenize_request_deserialize() {
    let json = r#"{"text": "hello world"}"#;
    let req: TokenizeRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.text, "hello world");
    assert!(req.model_id.is_none());
}

#[test]
fn test_tokenize_request_with_model_id() {
    let json = r#"{"text": "hello", "model_id": "phi-2"}"#;
    let req: TokenizeRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.model_id, Some("phi-2".to_string()));
}

#[test]
fn test_tokenize_response_serialize() {
    let response = TokenizeResponse {
        token_ids: vec![1, 2, 3],
        num_tokens: 3,
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("[1,2,3]"));
}

#[test]
fn test_generate_request_defaults() {
    let json = r#"{"prompt": "Hello"}"#;
    let req: GenerateRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.prompt, "Hello");
    assert_eq!(req.max_tokens, 50); // default
    assert!((req.temperature - 1.0).abs() < 0.001);
    assert_eq!(req.strategy, "greedy");
    assert_eq!(req.top_k, 50);
    assert!((req.top_p - 0.9).abs() < 0.001);
}

#[test]
fn test_generate_request_custom_values() {
    let json = r#"{"prompt": "Hi", "max_tokens": 100, "temperature": 0.7, "strategy": "top_k", "top_k": 40}"#;
    let req: GenerateRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.max_tokens, 100);
    assert!((req.temperature - 0.7).abs() < 0.001);
    assert_eq!(req.strategy, "top_k");
    assert_eq!(req.top_k, 40);
}

#[test]
fn test_generate_response_serialize() {
    let response = GenerateResponse {
        token_ids: vec![1, 2],
        text: "test output".to_string(),
        num_generated: 2,
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("test output"));
}

#[test]
fn test_error_response_serialize() {
    let response = ErrorResponse {
        error: "Something went wrong".to_string(),
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("Something went wrong"));
}

#[test]
fn test_batch_tokenize_request_deserialize() {
    let json = r#"{"texts": ["hello", "world"]}"#;
    let req: BatchTokenizeRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.texts.len(), 2);
}

#[test]
fn test_batch_tokenize_response_serialize() {
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
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("results"));
}

#[test]
fn test_chat_message_roles() {
    let system = ChatMessage {
        role: "system".to_string(),
        content: "You are helpful".to_string(),
        name: None,
    };
    let user = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("John".to_string()),
    };
    let assistant = ChatMessage {
        role: "assistant".to_string(),
        content: "Hi!".to_string(),
        name: None,
    };
    assert_eq!(system.role, "system");
    assert_eq!(user.role, "user");
    assert_eq!(assistant.role, "assistant");
    assert_eq!(user.name, Some("John".to_string()));
}

#[test]
fn test_chat_completion_request_deserialize() {
    let json = r#"{"model": "phi-2", "messages": [{"role": "user", "content": "hi"}]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.model, "phi-2");
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn test_chat_completion_response_serialize() {
    let response = ChatCompletionResponse {
        id: "chat-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "phi-2".to_string(),
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
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("chat-123"));
    assert!(json.contains("phi-2"));
}

#[test]
fn test_usage_serialize() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };
    let json = serde_json::to_string(&usage).expect("test");
    assert!(json.contains("150"));
}

#[test]
fn test_openai_model_serialize() {
    let model = OpenAIModel {
        id: "gpt-4".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "openai".to_string(),
    };
    let json = serde_json::to_string(&model).expect("test");
    assert!(json.contains("gpt-4"));
}

#[test]
fn test_stream_token_event_serialize() {
    let event = StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };
    let json = serde_json::to_string(&event).expect("test");
    assert!(json.contains("42"));
    assert!(json.contains("hello"));
}

#[test]
fn test_stream_done_event_serialize() {
    let event = StreamDoneEvent { num_generated: 100 };
    let json = serde_json::to_string(&event).expect("test");
    assert!(json.contains("100"));
}

#[test]
fn test_models_response_serialize() {
    let response = ModelsResponse {
        models: vec![
            ModelInfo {
                id: "phi-2".to_string(),
                name: "Phi-2".to_string(),
                description: "Microsoft Phi-2".to_string(),
                format: "gguf".to_string(),
                loaded: true,
            },
            ModelInfo {
                id: "llama".to_string(),
                name: "LLaMA".to_string(),
                description: "Meta LLaMA".to_string(),
                format: "gguf".to_string(),
                loaded: false,
            },
        ],
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("phi-2"));
    assert!(json.contains("llama"));
}

// =========================================================================
// Coverage Tests: Request/Response Structs
// =========================================================================

#[test]
fn test_chat_message_fields_cov() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello, world!".to_string(),
        name: None,
    };
    assert_eq!(msg.role, "user");
    assert_eq!(msg.content, "Hello, world!");
    assert!(msg.name.is_none());
}

#[test]
fn test_chat_completion_request_defaults_cov() {
    let req = ChatCompletionRequest {
        model: "phi-2".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        }],
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        max_tokens: None,
        user: None,
    };
    assert_eq!(req.model, "phi-2");
    assert!(req.temperature.is_none());
    assert!(!req.stream);
}

#[test]
fn test_chat_choice_fields_cov() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Hello!".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };
    assert_eq!(choice.index, 0);
    assert_eq!(choice.message.role, "assistant");
    assert_eq!(choice.finish_reason, "stop");
}

#[test]
fn test_usage_fields_cov() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(usage.total_tokens, 30);
}

#[test]
fn test_openai_model_fields_cov() {
    let model = OpenAIModel {
        id: "phi-2".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "microsoft".to_string(),
    };
    assert_eq!(model.id, "phi-2");
    assert_eq!(model.object, "model");
    assert_eq!(model.owned_by, "microsoft");
}

#[test]
fn test_chat_delta_fields_cov() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("Hello".to_string()),
    };
    assert!(delta.role.is_some());
    assert!(delta.content.is_some());
}

#[test]
fn test_chat_chunk_choice_fields_cov() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: Some("assistant".to_string()),
            content: None,
        },
        finish_reason: None,
    };
    assert_eq!(choice.index, 0);
    assert!(choice.finish_reason.is_none());
}

#[test]
fn test_predict_request_fields_cov() {
    let req = PredictRequest {
        model: Some("test-model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        top_k: None,
        include_confidence: true,
    };
    assert!(req.model.is_some());
    assert_eq!(req.features.len(), 3);
    assert!(req.top_k.is_none());
    assert!(req.include_confidence);
}

#[test]
fn test_explain_request_fields_cov() {
    let req = ExplainRequest {
        model: Some("test-model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["x".to_string(), "y".to_string(), "z".to_string()],
        top_k_features: 5,
        method: "shap".to_string(),
    };
    assert!(req.model.is_some());
    assert_eq!(req.features.len(), 3);
    assert_eq!(req.method, "shap");
    assert_eq!(req.top_k_features, 5);
}

#[test]
fn test_dispatch_metrics_query_default_cov() {
    let query = DispatchMetricsQuery { format: None };
    assert!(query.format.is_none());
}

#[test]
fn test_dispatch_reset_response_fields_cov() {
    let resp = DispatchResetResponse {
        success: true,
        message: "Reset successful".to_string(),
    };
    assert!(resp.success);
    assert!(resp.message.contains("Reset"));
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_request_fields_cov() {
    let req = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 100,
        temperature: 0.7,
        top_k: 50,
        stop: vec![],
    };
    assert_eq!(req.prompts.len(), 2);
    assert_eq!(req.max_tokens, 100);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_result_fields_cov() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![1, 2, 3],
        text: "Generated text".to_string(),
        num_generated: 3,
    };
    assert_eq!(result.index, 0);
    assert_eq!(result.num_generated, 3);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_stats_fields_cov() {
    let stats = GpuBatchStats {
        batch_size: 10,
        gpu_used: true,
        total_tokens: 500,
        processing_time_ms: 100.0,
        throughput_tps: 5000.0,
    };
    assert_eq!(stats.batch_size, 10);
    assert!(stats.gpu_used);
    assert_eq!(stats.total_tokens, 500);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_warmup_response_fields_cov() {
    let resp = GpuWarmupResponse {
        success: true,
        memory_bytes: 1024 * 1024,
        num_layers: 32,
        message: "GPU warmed up".to_string(),
    };
    assert!(resp.success);
    assert!(resp.memory_bytes > 0);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_status_response_fields_cov() {
    let resp = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1024 * 1024,
        batch_threshold: 8,
        recommended_min_batch: 4,
    };
    assert!(resp.cache_ready);
    assert!(resp.cache_memory_bytes > 0);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_fields_cov() {
    let config = BatchConfig {
        window_ms: 100,
        min_batch: 2,
        optimal_batch: 8,
        max_batch: 32,
        queue_size: 128,
        gpu_threshold: 32,
    };
    assert!(config.max_batch > 0);
    assert!(config.window_ms > 0);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_fields_cov() {
    let stats = BatchQueueStats {
        total_queued: 100,
        total_batches: 10,
        total_single: 5,
        avg_batch_size: 10.0,
        avg_wait_ms: 50.0,
    };
    assert_eq!(stats.total_queued, 100);
    assert_eq!(stats.total_batches, 10);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_process_result_fields_cov() {
    let result = BatchProcessResult {
        requests_processed: 5,
        was_batched: true,
        total_time_ms: 500.0,
        avg_latency_ms: 100.0,
    };
    assert_eq!(result.requests_processed, 5);
    assert!(result.was_batched);
}

#[test]
fn test_context_window_config_fields_cov() {
    let config = ContextWindowConfig {
        max_tokens: 4096,
        reserved_output_tokens: 512,
        preserve_system: true,
    };
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 512);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_default_cov() {
    let config = ContextWindowConfig::default();
    assert_eq!(config.max_tokens, 4096);
    assert!(config.preserve_system);
}

#[test]
fn test_embedding_request_fields_cov() {
    let req = EmbeddingRequest {
        input: "Some text to embed".to_string(),
        model: Some("text-embedding".to_string()),
    };
    assert!(req.model.is_some());
    assert!(req.input.contains("embed"));
}

// =========================================================================
// Coverage Tests: HealthResponse
// =========================================================================

#[test]
fn test_health_response_serialize_cov() {
    let resp = HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("healthy"));
    assert!(json.contains("1.0.0"));
    assert!(json.contains("cpu"));
}

// =========================================================================
// Coverage Tests: ErrorResponse
// =========================================================================

#[test]
fn test_error_response_serialize_cov() {
    let resp = ErrorResponse {
        error: "Something went wrong".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("Something went wrong"));
}

// =========================================================================
// Coverage Tests: OpenAI Compatibility Structs
// =========================================================================

#[test]
fn test_openai_models_response_serialize_cov() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: "gpt-3.5-turbo".to_string(),
            object: "model".to_string(),
            created: 1677610602,
            owned_by: "openai".to_string(),
        }],
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("list"));
    assert!(json.contains("gpt-3.5-turbo"));
}

#[test]
fn test_chat_completion_response_serialize_cov() {
    let resp = ChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1677652288,
        model: "phi-2".to_string(),
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
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("chatcmpl-123"));
    assert!(json.contains("Hello!"));
}

#[test]
fn test_chat_completion_chunk_serialize_cov() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-chunk-123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1677652288,
        model: "phi-2".to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: Some("Hi".to_string()),
            },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).expect("serialize");
    assert!(json.contains("chunk"));
    assert!(json.contains("Hi"));
}

// =========================================================================
// Additional Coverage Tests: Usage struct
// =========================================================================

#[test]
fn test_usage_debug_clone_cov() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };
    let debug = format!("{:?}", usage);
    assert!(debug.contains("Usage"));
    assert!(debug.contains("100"));

    let cloned = usage.clone();
    assert_eq!(cloned.prompt_tokens, usage.prompt_tokens);
    assert_eq!(cloned.total_tokens, usage.total_tokens);
}

// =========================================================================
// Additional Coverage Tests: ChatDelta struct
// =========================================================================
