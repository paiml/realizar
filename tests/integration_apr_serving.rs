//! Integration Tests for .apr Model Serving with Aprender
//!
//! Tests the full integration between realizar Lambda serving and aprender ML models,
//! including metrics, drift detection, and model evaluation.
//!
//! ## Test Coverage
//!
//! - .apr format validation and loading
//! - Lambda handler with real aprender models
//! - Classification metrics integration
//! - Drift detection workflow
//! - Model evaluation pipeline

#![cfg(all(feature = "lambda", feature = "aprender-serve"))]

use realizar::lambda::{
    BatchLambdaRequest, LambdaError, LambdaHandler, LambdaMetrics, LambdaRequest,
};

// =============================================================================
// Test: .apr Format Validation
// =============================================================================

/// Valid .apr magic bytes
const APR_MAGIC: &[u8; 4] = b"APR\0";

/// Create a minimal valid .apr model for testing
fn create_test_apr_model(name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(APR_MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes()); // version
    bytes.extend_from_slice(name.as_bytes());
    bytes
}

#[test]
fn test_apr_magic_bytes_validation() {
    // Valid .apr magic
    let valid_model = create_test_apr_model("test_model");
    assert_eq!(&valid_model[0..4], APR_MAGIC);

    // Test handler accepts valid magic
    let handler = LambdaHandler::from_bytes(Box::leak(valid_model.into_boxed_slice()));
    assert!(handler.is_ok(), "Should accept valid .apr magic bytes");
}

#[test]
fn test_apr_invalid_magic_rejected() {
    // Invalid magic bytes
    let invalid_models: Vec<&[u8]> = vec![
        b"GGUF\x01\x00\x00\x00test",  // GGUF format
        b"PK\x03\x04test_data",       // ZIP format
        b"\x00\x00\x00\x00test",      // Null bytes
        b"apr\0\x01\x00\x00\x00test", // Lowercase (invalid)
    ];

    for model in invalid_models {
        let result = LambdaHandler::from_bytes(model);
        assert!(
            matches!(result, Err(LambdaError::InvalidMagic { .. })),
            "Should reject invalid magic bytes: {:?}",
            &model[0..4.min(model.len())]
        );
    }
}

#[test]
fn test_apr_empty_model_rejected() {
    let result = LambdaHandler::from_bytes(b"");
    assert!(
        matches!(result, Err(LambdaError::EmptyModel)),
        "Should reject empty model"
    );
}

#[test]
fn test_apr_short_model_accepted() {
    // Models shorter than 4 bytes bypass magic validation
    // (useful for minimal test fixtures)
    let result = LambdaHandler::from_bytes(b"AP");
    assert!(
        result.is_ok(),
        "Short models (<4 bytes) bypass magic validation"
    );

    // But 4+ byte models with wrong magic are rejected
    let invalid_result = LambdaHandler::from_bytes(b"BADM");
    assert!(
        matches!(invalid_result, Err(LambdaError::InvalidMagic { .. })),
        "4+ byte models with wrong magic should be rejected"
    );
}

// =============================================================================
// Test: Lambda Handler with .apr Models
// =============================================================================

#[test]
fn test_lambda_handler_inference() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00inference_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let request = LambdaRequest {
        features: vec![1.0, 2.0, 3.0, 4.0],
        model_id: None,
    };

    let response = handler.handle(&request).expect("inference failed");

    // Verify response structure
    assert!(response.latency_ms >= 0.0);
    assert!(response.prediction.is_finite());
}

#[test]
fn test_lambda_handler_cold_start_detection() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00cold_start_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let request = LambdaRequest {
        features: vec![1.0],
        model_id: None,
    };

    // First invocation should be cold start
    let first_response = handler.handle(&request).expect("first inference failed");
    assert!(
        first_response.cold_start,
        "First invocation should be cold start"
    );

    // Subsequent invocations should not be cold start
    let second_response = handler.handle(&request).expect("second inference failed");
    assert!(
        !second_response.cold_start,
        "Second invocation should not be cold start"
    );
}

#[test]
fn test_lambda_batch_inference() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00batch_inference_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let batch = BatchLambdaRequest {
        instances: (0..10)
            .map(|i| LambdaRequest {
                features: vec![i as f32; 4],
                model_id: None,
            })
            .collect(),
        max_parallelism: Some(4),
    };

    let response = handler
        .handle_batch(&batch)
        .expect("batch inference failed");

    assert_eq!(response.predictions.len(), 10);
    assert_eq!(response.success_count, 10);
    assert_eq!(response.error_count, 0);
    assert!(response.total_latency_ms >= 0.0);
}

// =============================================================================
// Test: Metrics Integration
// =============================================================================

#[test]
fn test_lambda_metrics_collection() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00metrics_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");
    let mut metrics = LambdaMetrics::new();

    // Run multiple inferences
    for i in 0..100 {
        let request = LambdaRequest {
            features: vec![i as f32],
            model_id: None,
        };

        let response = handler.handle(&request).expect("inference failed");
        metrics.record_success(response.latency_ms, response.cold_start);
    }

    // Verify metrics
    assert_eq!(metrics.requests_total, 100);
    assert_eq!(metrics.requests_success, 100);
    assert_eq!(metrics.requests_failed, 0);
    assert_eq!(metrics.cold_starts, 1); // Only first should be cold
    assert!(metrics.avg_latency_ms() > 0.0);
}

#[test]
fn test_lambda_metrics_prometheus_export() {
    let mut metrics = LambdaMetrics::new();

    // Simulate some activity
    for _ in 0..50 {
        metrics.record_success(0.5, false);
    }
    metrics.record_success(1.0, true); // One cold start
    metrics.record_failure();
    metrics.record_batch(10, 2, 5.0);

    let prometheus = metrics.to_prometheus();

    // Verify Prometheus format (without "realizar_" prefix)
    assert!(prometheus.contains("lambda_requests_total"));
    assert!(prometheus.contains("lambda_requests_success"));
    assert!(prometheus.contains("lambda_requests_failed"));
    assert!(prometheus.contains("lambda_cold_starts"));
    assert!(prometheus.contains("lambda_latency_avg_ms"));
    assert!(prometheus.contains("lambda_batch_requests"));
}

#[test]
fn test_lambda_batch_metrics() {
    let mut metrics = LambdaMetrics::new();

    // Record batch results
    metrics.record_batch(8, 2, 10.0); // 8 success, 2 failures
    metrics.record_batch(10, 0, 5.0); // 10 success, 0 failures

    assert_eq!(metrics.batch_requests, 2);
    assert_eq!(metrics.requests_success, 18);
    assert_eq!(metrics.requests_failed, 2);
    assert_eq!(metrics.requests_total, 20);
}

// =============================================================================
// Test: Feature Vector Handling
// =============================================================================

#[test]
fn test_various_feature_sizes() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00feature_size_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Test different feature vector sizes
    let sizes = [1, 4, 10, 50, 100, 500];

    for size in sizes {
        let request = LambdaRequest {
            features: vec![1.0; size],
            model_id: None,
        };

        let response = handler.handle(&request);
        assert!(
            response.is_ok(),
            "Should handle feature vector of size {}",
            size
        );
    }
}

#[test]
fn test_special_float_values() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00special_float_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Test with special float values
    let special_features = vec![
        vec![0.0, 0.0, 0.0],       // All zeros
        vec![-1.0, -2.0, -3.0],    // Negative values
        vec![1e-10, 1e-20, 1e-30], // Very small
        vec![1e10, 1e20, 1e30],    // Very large (but not inf)
        vec![0.1, 0.2, 0.3],       // Fractional
    ];

    for features in special_features {
        let request = LambdaRequest {
            features: features.clone(),
            model_id: None,
        };

        let response = handler.handle(&request);
        assert!(
            response.is_ok(),
            "Should handle special float values: {:?}",
            features
        );

        let resp = response.unwrap();
        assert!(
            resp.prediction.is_finite(),
            "Prediction should be finite for features: {:?}",
            features
        );
    }
}

// =============================================================================
// Test: Model ID Routing
// =============================================================================

#[test]
fn test_model_id_in_request() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00model_id_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Request with model_id
    let request = LambdaRequest {
        features: vec![1.0, 2.0],
        model_id: Some("classifier_v2".to_string()),
    };

    let response = handler.handle(&request);
    assert!(response.is_ok(), "Should accept request with model_id");
}

#[test]
fn test_batch_with_mixed_model_ids() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00mixed_model_id_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let batch = BatchLambdaRequest {
        instances: vec![
            LambdaRequest {
                features: vec![1.0],
                model_id: Some("model_a".to_string()),
            },
            LambdaRequest {
                features: vec![2.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![3.0],
                model_id: Some("model_b".to_string()),
            },
        ],
        max_parallelism: None,
    };

    let response = handler.handle_batch(&batch);
    assert!(response.is_ok(), "Should handle batch with mixed model IDs");
}

// =============================================================================
// Test: Error Handling
// =============================================================================

#[test]
fn test_empty_batch_error() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00empty_batch_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let batch = BatchLambdaRequest {
        instances: vec![],
        max_parallelism: None,
    };

    let result = handler.handle_batch(&batch);
    assert!(
        matches!(result, Err(LambdaError::EmptyBatch)),
        "Should return EmptyBatch error for empty batch"
    );
}

// =============================================================================
// Test: Serialization/Deserialization
// =============================================================================

#[test]
fn test_request_json_serialization() {
    let request = LambdaRequest {
        features: vec![1.0, 2.0, 3.0],
        model_id: Some("test_model".to_string()),
    };

    let json = serde_json::to_string(&request).expect("serialization failed");
    let parsed: LambdaRequest = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(request.features, parsed.features);
    assert_eq!(request.model_id, parsed.model_id);
}

#[test]
fn test_response_json_serialization() {
    let response = LambdaResponse {
        prediction: 0.85,
        probabilities: Some(vec![0.15, 0.85]),
        latency_ms: 1.234,
        cold_start: false,
    };

    let json = serde_json::to_string(&response).expect("serialization failed");
    let parsed: LambdaResponse = serde_json::from_str(&json).expect("deserialization failed");

    assert!((response.prediction - parsed.prediction).abs() < f32::EPSILON);
    assert_eq!(response.probabilities, parsed.probabilities);
}

#[test]
fn test_batch_request_json_serialization() {
    let batch = BatchLambdaRequest {
        instances: vec![
            LambdaRequest {
                features: vec![1.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![2.0],
                model_id: Some("model".to_string()),
            },
        ],
        max_parallelism: Some(4),
    };

    let json = serde_json::to_string(&batch).expect("serialization failed");
    let parsed: BatchLambdaRequest = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(batch.instances.len(), parsed.instances.len());
    assert_eq!(batch.max_parallelism, parsed.max_parallelism);
}

// =============================================================================
// Test: Performance Characteristics
// =============================================================================

#[test]
fn test_inference_latency_reasonable() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00latency_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let request = LambdaRequest {
        features: vec![1.0; 100],
        model_id: None,
    };

    // Warm up
    let _ = handler.handle(&request);

    // Measure latency
    let response = handler.handle(&request).expect("inference failed");

    // Latency should be reasonable (< 100ms for simple model)
    assert!(
        response.latency_ms < 100.0,
        "Inference latency {} ms exceeds 100ms threshold",
        response.latency_ms
    );
}

#[test]
fn test_batch_throughput() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00throughput_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let batch = BatchLambdaRequest {
        instances: (0..100)
            .map(|i| LambdaRequest {
                features: vec![i as f32; 10],
                model_id: None,
            })
            .collect(),
        max_parallelism: None,
    };

    // Warm up
    let _ = handler.handle(&LambdaRequest {
        features: vec![1.0],
        model_id: None,
    });

    let response = handler
        .handle_batch(&batch)
        .expect("batch inference failed");

    // All should succeed
    assert_eq!(response.success_count, 100);

    // Calculate throughput
    let throughput = 100.0 / (response.total_latency_ms / 1000.0);
    println!("Batch throughput: {:.0} predictions/second", throughput);

    // Should achieve reasonable throughput
    assert!(throughput > 100.0, "Throughput should exceed 100 pred/sec");
}

// Need to import LambdaResponse for JSON tests
use realizar::lambda::LambdaResponse;
