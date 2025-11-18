//! HTTP API for model inference
//!
//! Provides REST endpoints for tokenization and text generation using axum.
//!
//! ## Endpoints
//!
//! - `GET /health` - Health check
//! - `POST /tokenize` - Tokenize text
//! - `POST /generate` - Generate text from prompt
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::api::{create_router, AppState};
//!
//! let state = AppState::new(model, tokenizer);
//! let app = create_router(state);
//! axum::serve(listener, app).await?;
//! ```

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{
    error::RealizarError,
    generate::{GenerationConfig, SamplingStrategy},
    layers::{Model, ModelConfig},
    tokenizer::BPETokenizer,
};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Model for inference
    model: Arc<Model>,
    /// Tokenizer for encoding/decoding
    tokenizer: Arc<BPETokenizer>,
}

impl AppState {
    /// Create new application state
    ///
    /// # Arguments
    ///
    /// * `model` - Model for inference
    /// * `tokenizer` - Tokenizer for text processing
    #[must_use]
    pub fn new(model: Model, tokenizer: BPETokenizer) -> Self {
        Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
        }
    }

    /// Create a demo state with small model for testing
    ///
    /// # Errors
    ///
    /// Returns error if model or tokenizer creation fails
    pub fn demo() -> Result<Self, RealizarError> {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 64,
            eps: 1e-5,
        };
        let model = Model::new(config)?;

        // Simple demo vocabulary
        let vocab: Vec<String> = (0..100)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        Ok(Self::new(model, tokenizer))
    }
}

/// Health check response
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Service version
    pub version: String,
}

/// Tokenize request
#[derive(Serialize, Deserialize)]
pub struct TokenizeRequest {
    /// Text to tokenize
    pub text: String,
}

/// Tokenize response
#[derive(Serialize, Deserialize)]
pub struct TokenizeResponse {
    /// Token IDs
    pub token_ids: Vec<u32>,
    /// Number of tokens
    pub num_tokens: usize,
}

/// Generate request
#[derive(Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Input prompt (token IDs or text)
    pub prompt: String,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Sampling strategy: "greedy", "`top_k`", or "`top_p`"
    #[serde(default = "default_strategy")]
    pub strategy: String,
    /// Top-k value (if strategy is "`top_k`")
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Top-p value (if strategy is "`top_p`")
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

fn default_max_tokens() -> usize {
    50
}
fn default_temperature() -> f32 {
    1.0
}
fn default_strategy() -> String {
    "greedy".to_string()
}
fn default_top_k() -> usize {
    50
}
fn default_top_p() -> f32 {
    0.9
}

/// Generate response
#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Generated token IDs
    pub token_ids: Vec<u32>,
    /// Decoded text
    pub text: String,
    /// Number of generated tokens
    pub num_generated: usize,
}

/// Error response
#[derive(Serialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
}

/// Create the API router
///
/// # Arguments
///
/// * `state` - Application state with model and tokenizer
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/tokenize", post(tokenize_handler))
        .route("/generate", post(generate_handler))
        .with_state(state)
}

/// Health check handler
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: crate::VERSION.to_string(),
    })
}

/// Tokenize text handler
async fn tokenize_handler(
    State(state): State<AppState>,
    Json(request): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let token_ids = state.tokenizer.encode(&request.text);
    let num_tokens = token_ids.len();

    Ok(Json(TokenizeResponse {
        token_ids,
        num_tokens,
    }))
}

/// Generate text handler
async fn generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Tokenize prompt
    let prompt_ids = state.tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Prompt cannot be empty".to_string(),
            }),
        ));
    }

    // Convert to usize for model
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    // Build generation config
    let strategy = match request.strategy.as_str() {
        "greedy" => SamplingStrategy::Greedy,
        "top_k" => SamplingStrategy::TopK { k: request.top_k },
        "top_p" => SamplingStrategy::TopP { p: request.top_p },
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid strategy: {}", request.strategy),
                }),
            ));
        }
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);

    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    // Generate
    let generated = state
        .model
        .generate(&prompt, &config)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    // Convert back to u32 and decode
    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| u32::try_from(id).unwrap_or(u32::MAX))
        .collect();
    let text = state.tokenizer.decode(&token_ids).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let num_generated = generated.len() - prompt.len();

    Ok(Json(GenerateResponse {
        token_ids,
        text,
        num_generated,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt;

    fn create_test_app() -> Router {
        let state = AppState::demo().unwrap();
        create_router(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_test_app();

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.status, "healthy");
    }

    #[tokio::test]
    async fn test_tokenize_endpoint() {
        let app = create_test_app();

        let request = TokenizeRequest {
            text: "token1 token2".to_string(),
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: TokenizeResponse = serde_json::from_slice(&body).unwrap();
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
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: GenerateResponse = serde_json::from_slice(&body).unwrap();
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
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

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
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

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
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

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
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_app_state_demo() {
        let state = AppState::demo();
        assert!(state.is_ok());
        let state = state.unwrap();
        assert_eq!(state.tokenizer.vocab_size(), 100);
    }
}
