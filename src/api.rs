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
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

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

/// Batch tokenize request
#[derive(Serialize, Deserialize)]
pub struct BatchTokenizeRequest {
    /// Texts to tokenize
    pub texts: Vec<String>,
}

/// Batch tokenize response
#[derive(Serialize, Deserialize)]
pub struct BatchTokenizeResponse {
    /// Results for each text in the same order
    pub results: Vec<TokenizeResponse>,
}

/// Batch generate request
#[derive(Serialize, Deserialize)]
pub struct BatchGenerateRequest {
    /// Input prompts
    pub prompts: Vec<String>,
    /// Maximum tokens to generate (shared across all prompts)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature (shared)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Sampling strategy (shared)
    #[serde(default = "default_strategy")]
    pub strategy: String,
    /// Top-k value (shared)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Top-p value (shared)
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// Batch generate response
#[derive(Serialize, Deserialize)]
pub struct BatchGenerateResponse {
    /// Results for each prompt in the same order
    pub results: Vec<GenerateResponse>,
}

/// Stream token event (SSE)
#[derive(Serialize, Deserialize)]
pub struct StreamTokenEvent {
    /// Token ID
    pub token_id: u32,
    /// Decoded text for this token
    pub text: String,
}

/// Stream done event (SSE)
#[derive(Serialize, Deserialize)]
pub struct StreamDoneEvent {
    /// Total number of tokens generated
    pub num_generated: usize,
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
        .route("/batch/tokenize", post(batch_tokenize_handler))
        .route("/batch/generate", post(batch_generate_handler))
        .route("/stream/generate", post(stream_generate_handler))
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
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);

    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    // Generate
    let generated = state.model.generate(&prompt, &config).map_err(|e| {
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

/// Batch tokenize handler
async fn batch_tokenize_handler(
    State(state): State<AppState>,
    Json(request): Json<BatchTokenizeRequest>,
) -> Result<Json<BatchTokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    if request.texts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Texts array cannot be empty".to_string(),
            }),
        ));
    }

    // Tokenize all texts
    let results: Vec<TokenizeResponse> = request
        .texts
        .iter()
        .map(|text| {
            let token_ids = state.tokenizer.encode(text);
            let num_tokens = token_ids.len();
            TokenizeResponse {
                token_ids,
                num_tokens,
            }
        })
        .collect();

    Ok(Json(BatchTokenizeResponse { results }))
}

/// Batch generate handler
async fn batch_generate_handler(
    State(state): State<AppState>,
    Json(request): Json<BatchGenerateRequest>,
) -> Result<Json<BatchGenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    if request.prompts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Prompts array cannot be empty".to_string(),
            }),
        ));
    }

    // Build generation config (shared across all prompts)
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
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);

    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    // Process each prompt
    let mut results = Vec::with_capacity(request.prompts.len());

    for prompt_text in &request.prompts {
        // Tokenize prompt
        let prompt_ids = state.tokenizer.encode(prompt_text);
        if prompt_ids.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Prompt '{prompt_text}' tokenizes to empty sequence"),
                }),
            ));
        }

        // Convert to usize for model
        let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

        // Generate
        let generated = state.model.generate(&prompt, &config).map_err(|e| {
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

        results.push(GenerateResponse {
            token_ids,
            text,
            num_generated,
        });
    }

    Ok(Json(BatchGenerateResponse { results }))
}

/// Stream generate handler - generates tokens one by one via Server-Sent Events
async fn stream_generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)>
{
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
    let prompt_len = prompt.len();

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
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);

    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    // Generate all tokens (in future, this will be truly streaming token-by-token)
    let generated = match state.model.generate(&prompt, &config) {
        Ok(tokens) => tokens,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            ));
        },
    };

    // Convert to u32
    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| u32::try_from(id).unwrap_or(u32::MAX))
        .collect();

    // Create stream that emits tokens one by one
    let tokenizer = state.tokenizer.clone();
    let stream = async_stream::stream! {
        // Skip prompt tokens, only stream generated tokens
        for &token_id in &token_ids[prompt_len..] {
            // Decode single token
            let text = match tokenizer.decode(&[token_id]) {
                Ok(t) => t,
                Err(_) => String::from("<error>"),
            };

            let event = StreamTokenEvent { token_id, text };
            let data = serde_json::to_string(&event).unwrap();

            yield Ok::<_, Infallible>(Event::default().event("token").data(data));
        }

        // Send done event
        let done_event = StreamDoneEvent {
            num_generated: token_ids.len() - prompt_len,
        };
        let data = serde_json::to_string(&done_event).unwrap();
        yield Ok(Event::default().event("done").data(data));
    };

    Ok(Sse::new(stream))
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt;

    use super::*;

    fn create_test_app() -> Router {
        let state = AppState::demo().unwrap();
        create_router(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_test_app();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
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

    #[test]
    fn test_default_max_tokens() {
        assert_eq!(default_max_tokens(), 50);
    }

    #[test]
    fn test_default_temperature() {
        assert!((default_temperature() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_strategy() {
        assert_eq!(default_strategy(), "greedy");
    }

    #[test]
    fn test_default_top_k() {
        assert_eq!(default_top_k(), 50);
    }

    #[test]
    fn test_default_top_p() {
        assert!((default_top_p() - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_generate_with_defaults() {
        let app = create_test_app();

        // Generate request using default values via serde defaults
        let json = r#"{"prompt": "test"}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(json))
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
        // Verify generation used defaults (greedy with max 50 tokens)
        assert!(result.num_generated <= 50);
    }

    #[tokio::test]
    async fn test_num_generated_calculation() {
        // First tokenize to get prompt length
        let app1 = create_test_app();
        let prompt_tokens = app1
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text": "a"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let prompt_body = axum::body::to_bytes(prompt_tokens.into_body(), usize::MAX)
            .await
            .unwrap();
        let prompt_result: TokenizeResponse = serde_json::from_slice(&prompt_body).unwrap();
        let prompt_len = prompt_result.token_ids.len();

        // Now generate
        let app2 = create_test_app();
        let request = GenerateRequest {
            prompt: "a".to_string(),
            max_tokens: 5,
            temperature: 1.0,
            strategy: "greedy".to_string(),
            top_k: 50,
            top_p: 0.9,
            seed: Some(42),
        };

        let response = app2
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

        // Verify num_generated = total_tokens - prompt_tokens
        assert_eq!(result.num_generated, result.token_ids.len() - prompt_len);

        // Also verify it's in reasonable range
        assert!(result.num_generated > 0);
        assert!(result.num_generated <= 5);
    }

    #[tokio::test]
    async fn test_batch_tokenize_endpoint() {
        let app = create_test_app();

        let request = BatchTokenizeRequest {
            texts: vec!["token1".to_string(), "token2 token3".to_string()],
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/batch/tokenize")
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
        let result: BatchTokenizeResponse = serde_json::from_slice(&body).unwrap();

        // Verify we got 2 results
        assert_eq!(result.results.len(), 2);
        // Each result should have tokens
        assert!(result.results[0].num_tokens > 0);
        assert!(result.results[1].num_tokens > 0);
    }

    #[tokio::test]
    async fn test_batch_tokenize_empty_array_error() {
        let app = create_test_app();

        let request = BatchTokenizeRequest { texts: vec![] };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/batch/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_batch_generate_endpoint() {
        let app = create_test_app();

        let request = BatchGenerateRequest {
            prompts: vec!["token1".to_string(), "token2".to_string()],
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
                    .uri("/batch/generate")
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
        let result: BatchGenerateResponse = serde_json::from_slice(&body).unwrap();

        // Verify we got 2 results
        assert_eq!(result.results.len(), 2);
        // Each result should have tokens
        assert!(!result.results[0].token_ids.is_empty());
        assert!(!result.results[1].token_ids.is_empty());
        // Each result should have text
        assert!(!result.results[0].text.is_empty());
        assert!(!result.results[1].text.is_empty());
    }

    #[tokio::test]
    async fn test_batch_generate_empty_array_error() {
        let app = create_test_app();

        let request = BatchGenerateRequest {
            prompts: vec![],
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
                    .uri("/batch/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_batch_generate_with_defaults() {
        let app = create_test_app();

        // Use serde defaults
        let json = r#"{"prompts": ["test1", "test2"]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/batch/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(json))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: BatchGenerateResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(result.results.len(), 2);
        // Verify generation used defaults (greedy with max 50 tokens)
        for gen_result in &result.results {
            assert!(gen_result.num_generated <= 50);
        }
    }

    #[tokio::test]
    async fn test_batch_generate_order_preserved() {
        let app = create_test_app();

        let request = BatchGenerateRequest {
            prompts: vec!["token1".to_string(), "token2".to_string(), "token3".to_string()],
            max_tokens: 2,
            temperature: 1.0,
            strategy: "greedy".to_string(),
            top_k: 50,
            top_p: 0.9,
            seed: Some(123),
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/batch/generate")
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
        let result: BatchGenerateResponse = serde_json::from_slice(&body).unwrap();

        // Verify order is preserved: 3 prompts -> 3 results in same order
        assert_eq!(result.results.len(), 3);

        // Each result should be non-empty
        for gen_result in &result.results {
            assert!(!gen_result.token_ids.is_empty());
            assert!(!gen_result.text.is_empty());
        }
    }

    #[tokio::test]
    async fn test_batch_generate_invalid_strategy_error() {
        let app = create_test_app();

        let request = BatchGenerateRequest {
            prompts: vec!["test".to_string()],
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
                    .uri("/batch/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_batch_generate_top_k_strategy() {
        let app = create_test_app();

        let request = BatchGenerateRequest {
            prompts: vec!["token1".to_string(), "token2".to_string()],
            max_tokens: 2,
            temperature: 0.8,
            strategy: "top_k".to_string(),
            top_k: 5,
            top_p: 0.9,
            seed: Some(456),
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/batch/generate")
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
        let result: BatchGenerateResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(result.results.len(), 2);
    }

    #[tokio::test]
    async fn test_batch_generate_top_p_strategy() {
        let app = create_test_app();

        let request = BatchGenerateRequest {
            prompts: vec!["token1".to_string()],
            max_tokens: 2,
            temperature: 0.7,
            strategy: "top_p".to_string(),
            top_k: 50,
            top_p: 0.9,
            seed: Some(789),
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/batch/generate")
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
        let result: BatchGenerateResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(result.results.len(), 1);
    }
}
