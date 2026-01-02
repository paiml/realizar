//! Aprender ML Model Serving API
//!
//! HTTP REST API for serving classical ML models in `.apr` format from aprender.
//! Separate from the LLM inference API in `api.rs`.
//!
//! ## The `.apr` Format
//!
//! Aprender's proprietary binary format with built-in quality (Jidoka):
//! - CRC32 checksum (integrity verification)
//! - Ed25519 signatures (provenance)
//! - AES-256-GCM encryption (confidentiality)
//! - Zstd compression (efficiency)
//! - Quantization support (`Q4_0`, `Q8_0`)
//! - Streaming/mmap (JIT loading)
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
//! - Sub-10ms p50 latency target (actual: ~0.5µs)
//! - Supports single-binary deployment via `include_bytes!()`
//! - 9.6x faster than `PyTorch` (statistically validated)
//!
//! ## Example: Load from `.apr` File
//!
//! ```rust,ignore
//! use realizar::serve::{create_serve_router, ServeState};
//! use aprender::format::{load, ModelType};
//! use aprender::classification::LogisticRegression;
//!
//! // Load trained model from .apr format (with CRC32 verification)
//! let model: LogisticRegression = load("model.apr", ModelType::LogisticRegression).expect("test");
//! let state = ServeState::with_logistic_regression(model, "mnist-v1".to_string(), 784);
//! let app = create_serve_router(state);
//! axum::serve(listener, app).await?;
//! ```
//!
//! ## Example: Embedded Model (Single Binary)
//!
//! ```rust,ignore
//! use realizar::serve::{create_serve_router, ServeState};
//! use aprender::format::{load_from_bytes, ModelType};
//! use aprender::classification::LogisticRegression;
//!
//! // Embed model at compile time
//! const MODEL_BYTES: &[u8] = include_bytes!("../models/sentiment.apr");
//!
//! // Load from embedded bytes (zero-copy where possible)
//! let model: LogisticRegression = load_from_bytes(MODEL_BYTES, ModelType::LogisticRegression).expect("test");
//! let state = ServeState::with_logistic_regression(model, "sentiment-v1".to_string(), 768);
//! ```

use std::{sync::Arc, time::Instant};

#[cfg(feature = "aprender-serve")]
use aprender::{
    classification::{GaussianNB, KNearestNeighbors, LinearSVM, LogisticRegression},
    linear_model::LinearRegression,
    primitives::Matrix,
    tree::{DecisionTreeClassifier, GradientBoostingClassifier, RandomForestClassifier},
    AprenderError, Estimator,
};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

/// Loaded model variant for type-erased serving
///
/// Per spec §5: Supports all 18 APR model types.
/// Currently implemented: 8 most common prediction-capable types.
#[cfg(feature = "aprender-serve")]
#[derive(Clone)]
pub enum LoadedModel {
    // ===== Classification Models =====
    /// Logistic regression model (binary classification)
    LogisticRegression(Arc<LogisticRegression>),
    /// K-nearest neighbors classifier
    KNearestNeighbors(Arc<KNearestNeighbors>),
    /// Gaussian Naive Bayes classifier
    GaussianNB(Arc<GaussianNB>),
    /// Linear SVM classifier
    LinearSVM(Arc<LinearSVM>),
    /// Decision tree classifier (CART)
    DecisionTreeClassifier(Arc<DecisionTreeClassifier>),
    /// Random forest classifier (bagging ensemble)
    RandomForestClassifier(Arc<RandomForestClassifier>),
    /// Gradient boosting classifier
    GradientBoostingClassifier(Arc<GradientBoostingClassifier>),

    // ===== Regression Models =====
    /// Linear regression model (OLS/Ridge/Lasso)
    LinearRegression(Arc<LinearRegression>),
}

/// Application state for aprender model serving
#[derive(Clone)]
pub struct ServeState {
    /// Currently loaded model (type-erased)
    #[cfg(feature = "aprender-serve")]
    model: Option<LoadedModel>,
    /// Model name/identifier
    model_name: String,
    /// Model version
    model_version: String,
    /// Input feature dimension (for validation)
    input_dim: usize,
    /// Request counter for metrics
    request_count: Arc<std::sync::atomic::AtomicU64>,
}

impl ServeState {
    /// Create new serving state without a model (for testing/scaffolding)
    #[must_use]
    pub fn new(model_name: String, model_version: String) -> Self {
        Self {
            #[cfg(feature = "aprender-serve")]
            model: None,
            model_name,
            model_version,
            input_dim: 0,
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Create serving state with a loaded `LogisticRegression` model
    #[cfg(feature = "aprender-serve")]
    #[must_use]
    pub fn with_logistic_regression(
        model: LogisticRegression,
        model_version: String,
        input_dim: usize,
    ) -> Self {
        Self {
            model: Some(LoadedModel::LogisticRegression(Arc::new(model))),
            model_name: "LogisticRegression".to_string(),
            model_version,
            input_dim,
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Load a `LogisticRegression` model from an `.apr` file
    ///
    /// The `.apr` format provides:
    /// - CRC32 integrity verification on load
    /// - Optional Ed25519 signature verification
    /// - Optional AES-256-GCM decryption
    /// - Automatic Zstd decompression
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be read
    /// - CRC32 checksum fails (file corrupted)
    /// - Signature verification fails (if signed)
    /// - Decryption fails (if encrypted, wrong key)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use realizar::serve::ServeState;
    ///
    /// let state = ServeState::load_apr("model.apr", "v1.0".to_string(), 784)?;
    /// ```
    #[cfg(feature = "aprender-serve")]
    pub fn load_apr(
        path: impl AsRef<std::path::Path>,
        model_version: String,
        input_dim: usize,
    ) -> Result<Self, anyhow::Error> {
        use aprender::format::{load, ModelType};

        let model: LogisticRegression = load(path, ModelType::LogisticRegression)?;

        Ok(Self {
            model: Some(LoadedModel::LogisticRegression(Arc::new(model))),
            model_name: "LogisticRegression".to_string(),
            model_version,
            input_dim,
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Load a `LogisticRegression` model from embedded `.apr` bytes
    ///
    /// Use this with `include_bytes!()` for single-binary deployment:
    ///
    /// ```rust,ignore
    /// use realizar::serve::ServeState;
    ///
    /// const MODEL: &[u8] = include_bytes!("../models/sentiment.apr");
    ///
    /// let state = ServeState::load_apr_from_bytes(MODEL, "v1.0".to_string(), 768)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - CRC32 checksum fails (data corrupted)
    /// - Signature verification fails (if signed)
    /// - Decryption fails (if encrypted)
    #[cfg(feature = "aprender-serve")]
    pub fn load_apr_from_bytes(
        bytes: &[u8],
        model_version: String,
        input_dim: usize,
    ) -> Result<Self, anyhow::Error> {
        use aprender::format::{load_from_bytes, ModelType};

        let model: LogisticRegression = load_from_bytes(bytes, ModelType::LogisticRegression)?;

        Ok(Self {
            model: Some(LoadedModel::LogisticRegression(Arc::new(model))),
            model_name: "LogisticRegression".to_string(),
            model_version,
            input_dim,
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Check if a model is loaded
    #[must_use]
    pub fn has_model(&self) -> bool {
        #[cfg(feature = "aprender-serve")]
        {
            self.model.is_some()
        }
        #[cfg(not(feature = "aprender-serve"))]
        {
            false
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
    let model_loaded = state.has_model();
    Json(ReadyResponse {
        ready: model_loaded,
        model_loaded,
        model_name: state.model_name,
    })
}

/// Predict handler
///
/// Per spec §5.1: Single prediction endpoint
/// Per spec §8.1: Target p50 latency <10ms (actual: ~0.5µs)
#[cfg(feature = "aprender-serve")]
async fn predict_handler(
    State(state): State<ServeState>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    use std::sync::atomic::Ordering;

    // Increment request counter
    state.request_count.fetch_add(1, Ordering::Relaxed);

    // Validate model is loaded
    let Some(model) = &state.model else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "No model loaded".to_string(),
                code: Some("E_NO_MODEL".to_string()),
            }),
        ));
    };

    // Validate input dimensions
    if state.input_dim > 0 && payload.features.len() != state.input_dim {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Invalid input dimension: expected {}, got {}",
                    state.input_dim,
                    payload.features.len()
                ),
                code: Some("E_INVALID_INPUT".to_string()),
            }),
        ));
    }

    // Perform inference with timing
    let start = Instant::now();

    // Create input matrix
    let n_features = payload.features.len();
    let input = Matrix::from_vec(1, n_features, payload.features.clone()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Failed to create input matrix: {e}"),
                code: Some("E_MATRIX_ERROR".to_string()),
            }),
        )
    })?;

    let return_probs = payload
        .options
        .as_ref()
        .is_some_and(|o| o.return_probabilities);

    // Helper to map aprender errors to HTTP errors
    let map_err = |e: aprender::AprenderError| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Model inference error: {e}"),
                code: Some("E_INFERENCE_ERROR".to_string()),
            }),
        )
    };

    let (prediction, probabilities) = match model {
        LoadedModel::LogisticRegression(lr) => {
            // LogisticRegression.predict() returns Vec<usize> directly
            let predictions = lr.predict(&input);
            #[allow(clippy::cast_precision_loss)]
            let pred = predictions.first().copied().unwrap_or(0) as f32;

            let probs = if return_probs {
                let prob_vec = lr.predict_proba(&input);
                let p1 = prob_vec.as_slice().first().copied().unwrap_or(0.5);
                Some(vec![1.0 - p1, p1])
            } else {
                None
            };
            (pred, probs)
        },
        LoadedModel::KNearestNeighbors(knn) => {
            // KNN.predict() returns Result<Vec<usize>>
            let predictions = knn.predict(&input).map_err(map_err)?;
            #[allow(clippy::cast_precision_loss)]
            let pred = predictions.first().copied().unwrap_or(0) as f32;
            // KNN doesn't support predict_proba
            (pred, None)
        },
        LoadedModel::GaussianNB(nb) => {
            // GaussianNB.predict() returns Result<Vec<usize>>
            let predictions = nb.predict(&input).map_err(map_err)?;
            #[allow(clippy::cast_precision_loss)]
            let pred = predictions.first().copied().unwrap_or(0) as f32;

            let probs = if return_probs {
                // predict_proba returns Result<Vec<Vec<f32>>>
                let prob_vecs = nb.predict_proba(&input).map_err(map_err)?;
                prob_vecs.first().cloned()
            } else {
                None
            };
            (pred, probs)
        },
        LoadedModel::LinearSVM(svm) => {
            // LinearSVM.predict() returns Result<Vec<usize>>
            let predictions = svm.predict(&input).map_err(map_err)?;
            #[allow(clippy::cast_precision_loss)]
            let pred = predictions.first().copied().unwrap_or(0) as f32;
            // SVM doesn't support predict_proba
            (pred, None)
        },
        LoadedModel::DecisionTreeClassifier(dt) => {
            // DecisionTree.predict() returns Vec<usize> directly
            let predictions = dt.predict(&input);
            #[allow(clippy::cast_precision_loss)]
            let pred = predictions.first().copied().unwrap_or(0) as f32;
            // DecisionTree doesn't support predict_proba in this API
            (pred, None)
        },
        LoadedModel::RandomForestClassifier(rf) => {
            // RandomForest.predict() returns Vec<usize> directly
            let predictions = rf.predict(&input);
            #[allow(clippy::cast_precision_loss)]
            let pred = predictions.first().copied().unwrap_or(0) as f32;

            let probs = if return_probs {
                // predict_proba returns Matrix<f32>
                let prob_matrix = rf.predict_proba(&input);
                let n_classes = prob_matrix.n_cols();
                let mut probs_vec = Vec::with_capacity(n_classes);
                for j in 0..n_classes {
                    probs_vec.push(prob_matrix.get(0, j));
                }
                Some(probs_vec)
            } else {
                None
            };
            (pred, probs)
        },
        LoadedModel::GradientBoostingClassifier(gb) => {
            // GradientBoosting.predict() returns Result<Vec<usize>>
            let predictions = gb.predict(&input).map_err(map_err)?;
            #[allow(clippy::cast_precision_loss)]
            let pred = predictions.first().copied().unwrap_or(0) as f32;

            let probs = if return_probs {
                // predict_proba returns Result<Vec<Vec<f32>>>
                let prob_vecs = gb.predict_proba(&input).map_err(map_err)?;
                prob_vecs.first().cloned()
            } else {
                None
            };
            (pred, probs)
        },
        LoadedModel::LinearRegression(lr) => {
            // LinearRegression.predict() returns Vector<f32> directly
            let predictions = lr.predict(&input);
            let pred = predictions.as_slice().first().copied().unwrap_or(0.0);
            // Regression doesn't have probabilities
            (pred, None)
        },
    };

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(PredictResponse {
        prediction,
        probabilities,
        latency_ms,
        model_version: state.model_version.clone(),
    }))
}

/// Predict handler (fallback when aprender-serve not enabled)
#[cfg(not(feature = "aprender-serve"))]
async fn predict_handler(
    State(_state): State<ServeState>,
    Json(_payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "aprender-serve feature not enabled".to_string(),
            code: Some("E_NOT_IMPLEMENTED".to_string()),
        }),
    ))
}

/// Batch predict handler
///
/// Per spec §5.3: Batch inference
#[cfg(feature = "aprender-serve")]
async fn batch_predict_handler(
    State(state): State<ServeState>,
    Json(payload): Json<BatchPredictRequest>,
) -> Result<Json<BatchPredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    use std::sync::atomic::Ordering;

    // Validate model is loaded
    let Some(model) = &state.model else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "No model loaded".to_string(),
                code: Some("E_NO_MODEL".to_string()),
            }),
        ));
    };

    if payload.instances.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Empty batch".to_string(),
                code: Some("E_EMPTY_BATCH".to_string()),
            }),
        ));
    }

    let batch_start = Instant::now();
    let mut predictions = Vec::with_capacity(payload.instances.len());

    // Increment request counter
    state
        .request_count
        .fetch_add(payload.instances.len() as u64, Ordering::Relaxed);

    for instance in &payload.instances {
        // Validate dimensions
        if state.input_dim > 0 && instance.features.len() != state.input_dim {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "Invalid input dimension: expected {}, got {}",
                        state.input_dim,
                        instance.features.len()
                    ),
                    code: Some("E_INVALID_INPUT".to_string()),
                }),
            ));
        }

        let start = Instant::now();

        // Create input matrix for this instance
        let n_features = instance.features.len();
        let input = Matrix::from_vec(1, n_features, instance.features.clone()).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Failed to create input matrix: {e}"),
                    code: Some("E_MATRIX_ERROR".to_string()),
                }),
            )
        })?;

        // Helper for batch error mapping
        let map_err = |e: AprenderError| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Model inference error: {e}"),
                    code: Some("E_INFERENCE_ERROR".to_string()),
                }),
            )
        };

        // For batch, don't return probabilities to keep response smaller
        let (prediction, probabilities) = match model {
            LoadedModel::LogisticRegression(lr) => {
                let preds = lr.predict(&input);
                #[allow(clippy::cast_precision_loss)]
                let pred = preds.first().copied().unwrap_or(0) as f32;
                (pred, None)
            },
            LoadedModel::KNearestNeighbors(knn) => {
                let preds = knn.predict(&input).map_err(map_err)?;
                #[allow(clippy::cast_precision_loss)]
                let pred = preds.first().copied().unwrap_or(0) as f32;
                (pred, None)
            },
            LoadedModel::GaussianNB(nb) => {
                let preds = nb.predict(&input).map_err(map_err)?;
                #[allow(clippy::cast_precision_loss)]
                let pred = preds.first().copied().unwrap_or(0) as f32;
                (pred, None)
            },
            LoadedModel::LinearSVM(svm) => {
                let preds = svm.predict(&input).map_err(map_err)?;
                #[allow(clippy::cast_precision_loss)]
                let pred = preds.first().copied().unwrap_or(0) as f32;
                (pred, None)
            },
            LoadedModel::DecisionTreeClassifier(dt) => {
                let preds = dt.predict(&input);
                #[allow(clippy::cast_precision_loss)]
                let pred = preds.first().copied().unwrap_or(0) as f32;
                (pred, None)
            },
            LoadedModel::RandomForestClassifier(rf) => {
                let preds = rf.predict(&input);
                #[allow(clippy::cast_precision_loss)]
                let pred = preds.first().copied().unwrap_or(0) as f32;
                (pred, None)
            },
            LoadedModel::GradientBoostingClassifier(gb) => {
                let preds = gb.predict(&input).map_err(map_err)?;
                #[allow(clippy::cast_precision_loss)]
                let pred = preds.first().copied().unwrap_or(0) as f32;
                (pred, None)
            },
            LoadedModel::LinearRegression(lr) => {
                let preds = lr.predict(&input);
                let pred = preds.as_slice().first().copied().unwrap_or(0.0);
                (pred, None)
            },
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        predictions.push(PredictResponse {
            prediction,
            probabilities,
            latency_ms,
            model_version: state.model_version.clone(),
        });
    }

    let total_latency_ms = batch_start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(BatchPredictResponse {
        predictions,
        total_latency_ms,
    }))
}

/// Batch predict handler (fallback when aprender-serve not enabled)
#[cfg(not(feature = "aprender-serve"))]
async fn batch_predict_handler(
    State(_state): State<ServeState>,
    Json(_payload): Json<BatchPredictRequest>,
) -> Result<Json<BatchPredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "aprender-serve feature not enabled".to_string(),
            code: Some("E_NOT_IMPLEMENTED".to_string()),
        }),
    ))
}

/// Models list handler
///
/// Per spec §5.1: List loaded models endpoint
async fn models_handler(
    State(state): State<ServeState>,
) -> Result<Json<ModelsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let models = if state.has_model() {
        vec![ModelInfo {
            id: "default".to_string(),
            model_type: state.model_name.clone(),
            version: state.model_version,
            loaded: true,
        }]
    } else {
        vec![]
    };

    Ok(Json(ModelsResponse { models }))
}

/// Metrics handler
///
/// Per spec §5.1: Prometheus metrics endpoint
async fn metrics_handler(State(state): State<ServeState>) -> String {
    use std::sync::atomic::Ordering;

    let request_count = state.request_count.load(Ordering::Relaxed);
    let model_loaded = i32::from(state.has_model());

    format!(
        "# HELP requests_total Total number of inference requests\n\
         # TYPE requests_total counter\n\
         requests_total {request_count}\n\
         # HELP model_loaded Whether a model is loaded (1=yes, 0=no)\n\
         # TYPE model_loaded gauge\n\
         model_loaded {model_loaded}\n\
         # HELP input_dimension Expected input feature dimension\n\
         # TYPE input_dimension gauge\n\
         input_dimension {}\n",
        state.input_dim
    )
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
    async fn test_ready_handler_no_model() {
        let state = ServeState::new("test-model".to_string(), "v1.0".to_string());
        let response = ready_handler(State(state)).await;
        // Without a model loaded, ready should be false
        assert!(!response.0.ready);
        assert!(!response.0.model_loaded);
        assert_eq!(response.0.model_name, "test-model");
    }

    #[test]
    fn test_serve_state_has_model() {
        let state = ServeState::new("test".to_string(), "v1".to_string());
        assert!(!state.has_model());
    }

    #[test]
    fn test_models_info_serialization() {
        let info = ModelInfo {
            id: "mnist-v1".to_string(),
            model_type: "LogisticRegression".to_string(),
            version: "1.0.0".to_string(),
            loaded: true,
        };

        let json = serde_json::to_string(&info).expect("serialization failed");
        assert!(json.contains("mnist-v1"));
        assert!(json.contains("LogisticRegression"));
    }

    /// Integration test: Train a model, serve it, and make predictions
    #[cfg(feature = "aprender-serve")]
    #[tokio::test]
    async fn test_predict_with_loaded_model() {
        // Train a simple model
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 matrix");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(100);
        model.fit(&x, &y).expect("Training should succeed");

        // Create serve state with model
        let state = ServeState::with_logistic_regression(model, "test-v1".to_string(), 2);
        assert!(state.has_model());

        // Create prediction request
        let request = PredictRequest {
            model_id: None,
            features: vec![0.9, 0.9], // Should predict class 1
            options: Some(PredictOptions {
                return_probabilities: true,
                top_k: None,
            }),
        };

        // Call predict handler
        let result = predict_handler(State(state.clone()), Json(request)).await;
        let response = result.expect("Prediction should succeed");

        // Verify response
        assert_eq!(response.prediction, 1.0); // Should predict class 1
        assert!(response.probabilities.is_some());
        assert!(response.latency_ms < 10.0); // Should be sub-10ms
        assert_eq!(response.model_version, "test-v1");

        // Test class 0 prediction
        let request_0 = PredictRequest {
            model_id: None,
            features: vec![0.0, 0.0], // Should predict class 0
            options: None,
        };
        let result_0 = predict_handler(State(state), Json(request_0)).await;
        let response_0 = result_0.expect("Prediction should succeed");
        assert_eq!(response_0.prediction, 0.0);
    }

    /// Integration test: Batch prediction
    #[cfg(feature = "aprender-serve")]
    #[tokio::test]
    async fn test_batch_predict_with_loaded_model() {
        // Train a simple model
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 matrix");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(100);
        model.fit(&x, &y).expect("Training should succeed");

        let state = ServeState::with_logistic_regression(model, "batch-v1".to_string(), 2);

        // Create batch request
        let request = BatchPredictRequest {
            model_id: None,
            instances: vec![
                PredictInstance {
                    features: vec![0.0, 0.0],
                },
                PredictInstance {
                    features: vec![1.0, 1.0],
                },
            ],
        };

        let result = batch_predict_handler(State(state), Json(request)).await;
        let response = result.expect("Batch prediction should succeed");

        assert_eq!(response.predictions.len(), 2);
        assert_eq!(response.predictions[0].prediction, 0.0); // Class 0
        assert_eq!(response.predictions[1].prediction, 1.0); // Class 1
        assert!(response.total_latency_ms < 10.0);
    }

    /// Test error handling for invalid input dimensions
    #[cfg(feature = "aprender-serve")]
    #[tokio::test]
    async fn test_predict_invalid_dimensions() {
        // Train model with 2 features
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 matrix");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).expect("Training should succeed");

        let state = ServeState::with_logistic_regression(model, "v1".to_string(), 2);

        // Request with wrong number of features
        let request = PredictRequest {
            model_id: None,
            features: vec![1.0, 2.0, 3.0], // 3 features, expected 2
            options: None,
        };

        let result = predict_handler(State(state), Json(request)).await;
        assert!(result.is_err());
        let (status, error) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error.error.contains("Invalid input dimension"));
    }
}
