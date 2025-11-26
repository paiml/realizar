//! Aprender ML Model Serving API
//!
//! HTTP REST API for serving classical ML models (.apr format) from aprender.
//! Separate from the LLM inference API in `api.rs`.
//!
//! ## Endpoints
//!
//! - `GET /health` - Health check / liveness probe
//! - `GET /ready` - Readiness check (model loaded)
//! - `POST /predict` - Single prediction
//! - `POST /predict/batch` - Batch predictions
//! - `GET /models` - List loaded models
//! - `GET /metrics` - Prometheus metrics
//!
//! ## Architecture
//!
//! Per `docs/specifications/serve-deploy-apr.md`:
//! - Pure Rust, WASM-compatible
//! - Sub-10ms p50 latency target
//! - Supports single-binary deployment via `include_bytes!()`
//! - Integrates trueno for SIMD acceleration
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::serve::{create_serve_router, ServeState};
//! use aprender::prelude::*;
//!
//! let model = LinearRegression::new();
//! let state = ServeState::new(model);
//! let app = create_serve_router(state);
//! axum::serve(listener, app).await?;
//! ```

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

/// Application state for aprender model serving
#[derive(Clone)]
pub struct ServeState {
    /// Currently loaded model (placeholder - will support multiple types)
    model_name: String,
    /// Model version
    #[allow(dead_code)] // Will be used when prediction is implemented
    model_version: String,
}

impl ServeState {
    /// Create new serving state
    #[must_use]
    pub fn new(model_name: String, model_version: String) -> Self {
        Self {
            model_name,
            model_version,
        }
    }
}

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Service version
    pub version: String,
}

/// Readiness check response
#[derive(Debug, Serialize, Deserialize)]
pub struct ReadyResponse {
    /// Ready status
    pub ready: bool,
    /// Model loaded
    pub model_loaded: bool,
    /// Model name
    pub model_name: String,
}

/// Prediction request
///
/// Per spec §5.2: Request schema for single prediction
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictRequest {
    /// Model ID (optional, uses default if not specified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Input features as flat array
    pub features: Vec<f32>,
    /// Optional prediction options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<PredictOptions>,
}

/// Prediction options
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictOptions {
    /// Return class probabilities (for classification models)
    #[serde(default)]
    pub return_probabilities: bool,
    /// Return top-k predictions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

/// Prediction response
///
/// Per spec §5.2: Response schema for single prediction
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictResponse {
    /// Predicted value or class
    pub prediction: f32,
    /// Class probabilities (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<Vec<f32>>,
    /// Inference latency in milliseconds
    pub latency_ms: f64,
    /// Model version used
    pub model_version: String,
}

/// Batch prediction request
///
/// Per spec §5.2: Batch request schema
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchPredictRequest {
    /// Model ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Multiple instances to predict
    pub instances: Vec<PredictInstance>,
}

/// Single instance in batch request
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictInstance {
    /// Input features
    pub features: Vec<f32>,
}

/// Batch prediction response
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchPredictResponse {
    /// Predictions for all instances
    pub predictions: Vec<PredictResponse>,
    /// Total batch processing time in milliseconds
    pub total_latency_ms: f64,
}

/// Models list response
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// Available models
    pub models: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model type (e.g., "`LinearRegression`", "`RandomForest`")
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Loaded status
    pub loaded: bool,
}

/// Error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// Error code (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Create router for aprender model serving
///
/// Per spec §5.1: HTTP API endpoint schema
pub fn create_serve_router(state: ServeState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/predict", post(predict_handler))
        .route("/predict/batch", post(batch_predict_handler))
        .route("/models", get(models_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

/// Health check handler
///
/// Per spec §5.1: Liveness probe
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Readiness check handler
///
/// Per spec §5.1: Readiness probe (model loaded)
async fn ready_handler(State(state): State<ServeState>) -> Json<ReadyResponse> {
    Json(ReadyResponse {
        ready: true, // TODO: Check if model is actually loaded
        model_loaded: true,
        model_name: state.model_name,
    })
}

/// Predict handler
///
/// Per spec §5.1: Single prediction endpoint
/// Per spec §8.1: Target p50 latency <10ms
async fn predict_handler(
    State(_state): State<ServeState>,
    Json(_payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    // TODO: Implement actual prediction logic
    // 1. Load model (or get from cache)
    // 2. Validate input dimensions
    // 3. Run inference with trueno SIMD
    // 4. Return prediction

    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "Prediction not yet implemented".to_string(),
            code: Some("E_NOT_IMPLEMENTED".to_string()),
        }),
    ))
}

/// Batch predict handler
///
/// Per spec §5.3: Batch inference
async fn batch_predict_handler(
    State(_state): State<ServeState>,
    Json(_payload): Json<BatchPredictRequest>,
) -> Result<Json<BatchPredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    // TODO: Implement batch prediction
    // Per spec §5.3: Use rayon for parallel inference on native targets

    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "Batch prediction not yet implemented".to_string(),
            code: Some("E_NOT_IMPLEMENTED".to_string()),
        }),
    ))
}

/// Models list handler
///
/// Per spec §5.1: List loaded models endpoint
async fn models_handler(
    State(_state): State<ServeState>,
) -> Result<Json<ModelsResponse>, (StatusCode, Json<ErrorResponse>)> {
    // TODO: Implement models listing from registry

    Ok(Json(ModelsResponse { models: vec![] }))
}

/// Metrics handler
///
/// Per spec §5.1: Prometheus metrics endpoint
async fn metrics_handler(State(_state): State<ServeState>) -> String {
    // TODO: Implement Prometheus metrics collection
    // Track: request count, latency histograms, error rate

    String::from("# HELP requests_total Total number of requests\n# TYPE requests_total counter\nrequests_total 0\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_state_creation() {
        let state = ServeState::new("test-model".to_string(), "v1.0".to_string());
        assert_eq!(state.model_name, "test-model");
        assert_eq!(state.model_version, "v1.0");
    }

    #[test]
    fn test_predict_request_serialization() {
        let request = PredictRequest {
            model_id: Some("sentiment-v1".to_string()),
            features: vec![0.5, 1.2, -0.3, 0.8],
            options: Some(PredictOptions {
                return_probabilities: true,
                top_k: Some(3),
            }),
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("sentiment-v1"));
        assert!(json.contains("0.5"));
        assert!(json.contains("return_probabilities"));
    }

    #[test]
    fn test_predict_response_serialization() {
        let response = PredictResponse {
            prediction: 1.0,
            probabilities: Some(vec![0.12, 0.85, 0.03]),
            latency_ms: 2.3,
            model_version: "v1.2.0".to_string(),
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("v1.2.0"));
        assert!(json.contains("2.3"));
        assert!(json.contains("0.12"));
    }

    #[test]
    fn test_batch_predict_request_serialization() {
        let request = BatchPredictRequest {
            model_id: Some("model-v1".to_string()),
            instances: vec![
                PredictInstance {
                    features: vec![0.5, 1.2],
                },
                PredictInstance {
                    features: vec![0.1, 0.9],
                },
            ],
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("model-v1"));
        assert!(json.contains("instances"));
        assert!(json.contains("0.5"));
        assert!(json.contains("0.9"));
    }

    #[test]
    fn test_health_response_format() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.2.0".to_string(),
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("healthy"));
        assert!(json.contains("0.2.0"));
    }

    #[test]
    fn test_ready_response_format() {
        let response = ReadyResponse {
            ready: true,
            model_loaded: true,
            model_name: "test-model".to_string(),
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("true"));
        assert!(json.contains("test-model"));
    }

    #[test]
    fn test_error_response_format() {
        let response = ErrorResponse {
            error: "Model not found".to_string(),
            code: Some("E404".to_string()),
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("Model not found"));
        assert!(json.contains("E404"));
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await;
        assert_eq!(response.0.status, "healthy");
        assert!(!response.0.version.is_empty());
    }

    #[tokio::test]
    async fn test_ready_handler() {
        let state = ServeState::new("test-model".to_string(), "v1.0".to_string());
        let response = ready_handler(State(state)).await;
        assert!(response.0.ready);
        assert!(response.0.model_loaded);
        assert_eq!(response.0.model_name, "test-model");
    }
}
