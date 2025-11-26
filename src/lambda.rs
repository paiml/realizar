//! AWS Lambda Handler for Aprender ML Model Serving
//!
//! Single-binary deployment via `include_bytes!()` for sub-50ms cold starts.
//! Optimized for ARM64 Graviton processors.
//!
//! ## Architecture
//!
//! Per `docs/specifications/serve-deploy-apr.md` Section 7:
//! - Model embedded at compile time via `include_bytes!()`
//! - `OnceLock` for lazy model initialization (amortizes cold start)
//! - ARM64-native binary for Graviton Lambda
//! - Target: <50ms cold start, <10ms warm inference
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::lambda::{LambdaHandler, LambdaRequest, LambdaResponse};
//!
//! // Model bytes embedded at compile time
//! static MODEL_BYTES: &[u8] = include_bytes!("../models/model.apr");
//!
//! let handler = LambdaHandler::from_bytes(MODEL_BYTES)?;
//! ```

use std::{sync::OnceLock, time::Instant};

use serde::{Deserialize, Serialize};

/// Embedded model bytes placeholder
/// In production: `static MODEL_BYTES: &[u8] = include_bytes!("../models/model.apr");`
#[allow(dead_code)]
pub static MODEL_BYTES: OnceLock<&'static [u8]> = OnceLock::new();

/// Lambda request format
///
/// Per spec §5.2: Compatible with `serve::PredictRequest`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaRequest {
    /// Input features for prediction
    pub features: Vec<f32>,
    /// Optional model ID (for multi-model deployments)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

/// Lambda response format
///
/// Per spec §5.2: Compatible with `serve::PredictResponse`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaResponse {
    /// Prediction result
    pub prediction: f32,
    /// Class probabilities (classification models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<Vec<f32>>,
    /// Inference latency in milliseconds
    pub latency_ms: f64,
    /// Cold start indicator (true on first invocation)
    pub cold_start: bool,
}

/// Cold start timing breakdown
///
/// Per spec §7.1: Instrumented cold start measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartMetrics {
    /// Runtime initialization time (ms)
    pub runtime_init_ms: f64,
    /// Model load time (ms)
    pub model_load_ms: f64,
    /// First inference time (ms)
    pub first_inference_ms: f64,
    /// Total cold start time (ms)
    pub total_ms: f64,
}

/// Lambda handler state
///
/// Per spec §6.1: Uses `OnceLock` for lazy initialization
#[derive(Debug)]
pub struct LambdaHandler {
    /// Model bytes (embedded via `include_bytes!()`)
    model_bytes: &'static [u8],
    /// Initialization time tracking
    init_time: OnceLock<Instant>,
    /// Cold start metrics
    cold_start_metrics: OnceLock<ColdStartMetrics>,
}

impl LambdaHandler {
    /// Create handler from embedded model bytes
    ///
    /// Per spec §6.1: `include_bytes!()` pattern
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Embedded .apr model bytes
    ///
    /// # Errors
    ///
    /// Returns `LambdaError::EmptyModel` if model bytes are empty.
    /// Returns `LambdaError::InvalidMagic` if magic bytes don't match .apr format.
    pub fn from_bytes(model_bytes: &'static [u8]) -> Result<Self, LambdaError> {
        if model_bytes.is_empty() {
            return Err(LambdaError::EmptyModel);
        }

        // Validate .apr magic bytes (per aprender model-format-spec.md)
        if model_bytes.len() >= 4 {
            let magic = &model_bytes[0..4];
            // .apr format uses "APR\0" magic bytes
            if magic != b"APR\0" {
                return Err(LambdaError::InvalidMagic {
                    expected: "APR\\0".to_string(),
                    found: format!("{:?}", &model_bytes[0..4.min(model_bytes.len())]),
                });
            }
        }

        Ok(Self {
            model_bytes,
            init_time: OnceLock::new(),
            cold_start_metrics: OnceLock::new(),
        })
    }

    /// Check if this is a cold start (first invocation)
    #[must_use]
    pub fn is_cold_start(&self) -> bool {
        self.init_time.get().is_none()
    }

    /// Get model bytes size
    #[must_use]
    pub fn model_size_bytes(&self) -> usize {
        self.model_bytes.len()
    }

    /// Get cold start metrics (only available after first invocation)
    #[must_use]
    pub fn cold_start_metrics(&self) -> Option<&ColdStartMetrics> {
        self.cold_start_metrics.get()
    }

    /// Handle Lambda invocation
    ///
    /// Per spec §6.1: Lambda handler pattern
    /// Per spec §8.1: Target <10ms warm latency
    ///
    /// # Errors
    ///
    /// Returns `LambdaError::EmptyFeatures` if request features are empty.
    pub fn handle(&self, request: &LambdaRequest) -> Result<LambdaResponse, LambdaError> {
        let start = Instant::now();
        let is_cold = self.is_cold_start();

        // Record initialization time on first invocation
        let _ = self.init_time.get_or_init(|| start);

        // Validate request
        if request.features.is_empty() {
            return Err(LambdaError::EmptyFeatures);
        }

        // TODO: Implement actual model inference
        // Per spec: Load model via OnceLock, run inference with trueno SIMD
        let prediction = self.mock_inference(&request.features)?;

        let latency = start.elapsed();

        // Record cold start metrics on first invocation
        if is_cold {
            let _ = self.cold_start_metrics.get_or_init(|| ColdStartMetrics {
                runtime_init_ms: 0.0, // Would be measured externally
                model_load_ms: 0.0,   // Would include model deserialization
                first_inference_ms: latency.as_secs_f64() * 1000.0,
                total_ms: latency.as_secs_f64() * 1000.0,
            });
        }

        Ok(LambdaResponse {
            prediction,
            probabilities: None,
            latency_ms: latency.as_secs_f64() * 1000.0,
            cold_start: is_cold,
        })
    }

    /// Mock inference for testing
    ///
    /// TODO: Replace with real model inference using trueno SIMD
    #[allow(clippy::unused_self)] // Will use self when real model is implemented
    #[allow(clippy::unnecessary_wraps)] // Will return errors when real model inference is added
    fn mock_inference(&self, features: &[f32]) -> Result<f32, LambdaError> {
        // Placeholder: compute simple sum
        // Real implementation would deserialize model and run inference
        Ok(features.iter().sum())
    }
}

/// Lambda handler errors
#[derive(Debug, Clone, PartialEq)]
pub enum LambdaError {
    /// Model bytes are empty
    EmptyModel,
    /// Invalid .apr magic bytes
    InvalidMagic {
        /// Expected magic bytes
        expected: String,
        /// Found magic bytes
        found: String,
    },
    /// Request features are empty
    EmptyFeatures,
    /// Model deserialization failed
    ModelDeserialize(String),
    /// Inference failed
    InferenceFailed(String),
}

impl std::fmt::Display for LambdaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LambdaError::EmptyModel => write!(f, "Model bytes are empty"),
            LambdaError::InvalidMagic { expected, found } => {
                write!(f, "Invalid magic bytes: expected {expected}, found {found}")
            },
            LambdaError::EmptyFeatures => write!(f, "Request features are empty"),
            LambdaError::ModelDeserialize(msg) => write!(f, "Model deserialization failed: {msg}"),
            LambdaError::InferenceFailed(msg) => write!(f, "Inference failed: {msg}"),
        }
    }
}

impl std::error::Error for LambdaError {}

/// ARM64-specific optimizations
///
/// Per spec §6.1: Graviton optimization
pub mod arm64 {
    /// Check if running on ARM64 architecture
    #[must_use]
    pub const fn is_arm64() -> bool {
        cfg!(target_arch = "aarch64")
    }

    /// Target architecture string
    #[must_use]
    pub const fn target_arch() -> &'static str {
        #[cfg(target_arch = "aarch64")]
        {
            "aarch64"
        }
        #[cfg(target_arch = "x86_64")]
        {
            "x86_64"
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            "unknown"
        }
    }

    /// Optimal SIMD instruction set for current architecture
    #[must_use]
    pub const fn optimal_simd() -> &'static str {
        #[cfg(target_arch = "aarch64")]
        {
            "NEON"
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            "AVX2"
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
        {
            "SSE2"
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            "Scalar"
        }
    }
}

/// Cold start benchmark utilities
///
/// Per spec §7.1: Cold start mitigation measurement
pub mod benchmark {
    use serde::{Deserialize, Serialize};

    use super::{arm64, Instant, LambdaError, LambdaHandler, LambdaRequest};

    /// Target cold start time (per spec §8.1)
    pub const TARGET_COLD_START_MS: f64 = 50.0;

    /// Target warm inference time (per spec §8.1)
    pub const TARGET_WARM_INFERENCE_MS: f64 = 10.0;

    /// Benchmark result
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchmarkResult {
        /// Cold start latency (ms)
        pub cold_start_ms: f64,
        /// Warm inference latency (ms) - median of N invocations
        pub warm_inference_ms: f64,
        /// Number of warm invocations measured
        pub warm_iterations: usize,
        /// Model size (bytes)
        pub model_size_bytes: usize,
        /// Target architecture
        pub target_arch: String,
        /// SIMD instruction set used
        pub simd_backend: String,
        /// Meets cold start target
        pub meets_cold_start_target: bool,
        /// Meets warm inference target
        pub meets_warm_inference_target: bool,
    }

    impl BenchmarkResult {
        /// Check if all targets are met
        #[must_use]
        pub fn meets_all_targets(&self) -> bool {
            self.meets_cold_start_target && self.meets_warm_inference_target
        }
    }

    /// Run cold start benchmark
    ///
    /// Per spec §7.1: Measure cold start with breakdown
    ///
    /// # Errors
    ///
    /// Returns `LambdaError` if handler invocation fails.
    ///
    /// # Panics
    ///
    /// Panics if latencies contain NaN values (should not happen with valid inputs).
    pub fn benchmark_cold_start(
        handler: &LambdaHandler,
        request: &LambdaRequest,
        warm_iterations: usize,
    ) -> Result<BenchmarkResult, LambdaError> {
        // First invocation is cold start
        let cold_start = Instant::now();
        let _cold_response = handler.handle(request)?;
        let cold_start_ms = cold_start.elapsed().as_secs_f64() * 1000.0;

        // Warm invocations
        let mut warm_latencies = Vec::with_capacity(warm_iterations);
        for _ in 0..warm_iterations {
            let start = Instant::now();
            let _response = handler.handle(request)?;
            warm_latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // Compute median warm latency
        warm_latencies.sort_by(|a, b| a.partial_cmp(b).expect("latencies should not contain NaN"));
        let warm_inference_ms = if warm_latencies.is_empty() {
            0.0
        } else {
            warm_latencies[warm_latencies.len() / 2]
        };

        Ok(BenchmarkResult {
            cold_start_ms,
            warm_inference_ms,
            warm_iterations,
            model_size_bytes: handler.model_size_bytes(),
            target_arch: arm64::target_arch().to_string(),
            simd_backend: arm64::optimal_simd().to_string(),
            meets_cold_start_target: cold_start_ms <= TARGET_COLD_START_MS,
            meets_warm_inference_target: warm_inference_ms <= TARGET_WARM_INFERENCE_MS,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // RED PHASE: Failing tests for Lambda handler
    // Per EXTREME TDD: Write tests FIRST, then implement to pass
    // ==========================================================================

    // --------------------------------------------------------------------------
    // Test: LambdaHandler creation
    // --------------------------------------------------------------------------

    #[test]
    fn test_lambda_handler_creation_with_valid_apr_model() {
        // Valid .apr model bytes with magic header
        let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00testmodel";
        let handler = LambdaHandler::from_bytes(model_bytes);
        assert!(handler.is_ok(), "Should accept valid .apr model");
        let handler = handler.unwrap();
        assert_eq!(handler.model_size_bytes(), model_bytes.len());
    }

    #[test]
    fn test_lambda_handler_rejects_empty_model() {
        let model_bytes: &'static [u8] = b"";
        let result = LambdaHandler::from_bytes(model_bytes);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LambdaError::EmptyModel);
    }

    #[test]
    fn test_lambda_handler_rejects_invalid_magic() {
        // Invalid magic bytes (not .apr format)
        let model_bytes: &'static [u8] = b"GGUF\x01\x00\x00\x00testmodel";
        let result = LambdaHandler::from_bytes(model_bytes);
        assert!(result.is_err());
        match result.unwrap_err() {
            LambdaError::InvalidMagic { expected, found } => {
                assert_eq!(expected, "APR\\0");
                // Verify that the found bytes are captured (format: "[71, 71, 85, 70]")
                assert!(!found.is_empty());
                assert!(found.contains("71")); // 'G' = 71 in ASCII
            },
            _ => panic!("Expected InvalidMagic error"),
        }
    }

    // --------------------------------------------------------------------------
    // Test: Lambda request/response serialization
    // --------------------------------------------------------------------------

    #[test]
    fn test_lambda_request_serialization() {
        let request = LambdaRequest {
            features: vec![0.5, 1.2, -0.3, 0.8],
            model_id: Some("sentiment-v1".to_string()),
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("0.5"));
        assert!(json.contains("sentiment-v1"));
    }

    #[test]
    fn test_lambda_request_deserialization() {
        let json = r#"{"features": [1.0, 2.0, 3.0], "model_id": "test-model"}"#;
        let request: LambdaRequest = serde_json::from_str(json).expect("deserialization failed");
        assert_eq!(request.features, vec![1.0, 2.0, 3.0]);
        assert_eq!(request.model_id, Some("test-model".to_string()));
    }

    #[test]
    fn test_lambda_response_serialization() {
        let response = LambdaResponse {
            prediction: 0.85,
            probabilities: Some(vec![0.15, 0.85]),
            latency_ms: 2.3,
            cold_start: true,
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("0.85"));
        assert!(json.contains("cold_start"));
        assert!(json.contains("true"));
    }

    // --------------------------------------------------------------------------
    // Test: Lambda handler invocation
    // --------------------------------------------------------------------------

    #[test]
    fn test_lambda_handler_cold_start_detection() {
        let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00testmodel";
        let handler = LambdaHandler::from_bytes(model_bytes).unwrap();

        // Before first invocation: cold start
        assert!(handler.is_cold_start());

        let request = LambdaRequest {
            features: vec![1.0, 2.0, 3.0],
            model_id: None,
        };

        // First invocation
        let response = handler.handle(&request).unwrap();
        assert!(response.cold_start, "First invocation should be cold start");

        // After first invocation: no longer cold
        assert!(!handler.is_cold_start());

        // Second invocation
        let response2 = handler.handle(&request).unwrap();
        assert!(
            !response2.cold_start,
            "Second invocation should not be cold start"
        );
    }

    #[test]
    fn test_lambda_handler_rejects_empty_features() {
        let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00testmodel";
        let handler = LambdaHandler::from_bytes(model_bytes).unwrap();

        let request = LambdaRequest {
            features: vec![],
            model_id: None,
        };

        let result = handler.handle(&request);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LambdaError::EmptyFeatures);
    }

    #[test]
    fn test_lambda_handler_mock_inference() {
        let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00testmodel";
        let handler = LambdaHandler::from_bytes(model_bytes).unwrap();

        let request = LambdaRequest {
            features: vec![1.0, 2.0, 3.0],
            model_id: None,
        };

        let response = handler.handle(&request).unwrap();
        // Mock inference returns sum of features
        assert!((response.prediction - 6.0).abs() < 0.001);
        assert!(response.latency_ms >= 0.0);
    }

    // --------------------------------------------------------------------------
    // Test: Cold start metrics
    // --------------------------------------------------------------------------

    #[test]
    fn test_cold_start_metrics_recorded() {
        let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00testmodel";
        let handler = LambdaHandler::from_bytes(model_bytes).unwrap();

        // No metrics before first invocation
        assert!(handler.cold_start_metrics().is_none());

        let request = LambdaRequest {
            features: vec![1.0],
            model_id: None,
        };

        let _ = handler.handle(&request).unwrap();

        // Metrics available after first invocation
        let metrics = handler.cold_start_metrics();
        assert!(metrics.is_some());
        let metrics = metrics.unwrap();
        assert!(metrics.total_ms >= 0.0);
        assert!(metrics.first_inference_ms >= 0.0);
    }

    // --------------------------------------------------------------------------
    // Test: ARM64 optimizations
    // --------------------------------------------------------------------------

    #[test]
    fn test_arm64_architecture_detection() {
        let arch = arm64::target_arch();
        assert!(
            arch == "aarch64" || arch == "x86_64" || arch == "unknown",
            "Should detect valid architecture"
        );
    }

    #[test]
    fn test_arm64_simd_detection() {
        let simd = arm64::optimal_simd();
        let valid_simd = ["NEON", "AVX2", "SSE2", "Scalar"];
        assert!(
            valid_simd.contains(&simd),
            "Should detect valid SIMD backend"
        );
    }

    // --------------------------------------------------------------------------
    // Test: Benchmark utilities
    // --------------------------------------------------------------------------

    #[test]
    fn test_benchmark_cold_start() {
        let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00testmodel";
        let handler = LambdaHandler::from_bytes(model_bytes).unwrap();

        let request = LambdaRequest {
            features: vec![1.0, 2.0],
            model_id: None,
        };

        let result = benchmark::benchmark_cold_start(&handler, &request, 10).unwrap();

        assert!(result.cold_start_ms >= 0.0);
        assert!(result.warm_inference_ms >= 0.0);
        assert_eq!(result.warm_iterations, 10);
        assert!(result.model_size_bytes > 0);
    }

    #[test]
    fn test_benchmark_targets() {
        // Verify target constants match spec §8.1
        assert_eq!(benchmark::TARGET_COLD_START_MS, 50.0);
        assert_eq!(benchmark::TARGET_WARM_INFERENCE_MS, 10.0);
    }

    #[test]
    fn test_benchmark_result_meets_targets() {
        let result = benchmark::BenchmarkResult {
            cold_start_ms: 45.0,
            warm_inference_ms: 8.0,
            warm_iterations: 100,
            model_size_bytes: 1000,
            target_arch: "aarch64".to_string(),
            simd_backend: "NEON".to_string(),
            meets_cold_start_target: true,
            meets_warm_inference_target: true,
        };

        assert!(result.meets_all_targets());
    }

    #[test]
    fn test_benchmark_result_fails_targets() {
        let result = benchmark::BenchmarkResult {
            cold_start_ms: 75.0, // Exceeds 50ms target
            warm_inference_ms: 8.0,
            warm_iterations: 100,
            model_size_bytes: 1000,
            target_arch: "aarch64".to_string(),
            simd_backend: "NEON".to_string(),
            meets_cold_start_target: false,
            meets_warm_inference_target: true,
        };

        assert!(!result.meets_all_targets());
    }

    // --------------------------------------------------------------------------
    // Test: Error display
    // --------------------------------------------------------------------------

    #[test]
    fn test_lambda_error_display() {
        assert_eq!(LambdaError::EmptyModel.to_string(), "Model bytes are empty");
        assert_eq!(
            LambdaError::EmptyFeatures.to_string(),
            "Request features are empty"
        );
        assert!(LambdaError::InvalidMagic {
            expected: "APR\\0".to_string(),
            found: "GGUF".to_string()
        }
        .to_string()
        .contains("Invalid magic"));
    }
}
