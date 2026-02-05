//! GPU batch inference handlers
//!
//! Extracted from api/mod.rs (PMAT-802) to reduce module size.
//! Contains batch completions, warmup, and status handlers for GPU inference.

use std::convert::Infallible;

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};

use super::{
    default_max_tokens, default_top_k, AppState, BatchGenerateRequest, BatchGenerateResponse,
    BatchTokenizeRequest, BatchTokenizeResponse, ErrorResponse, GenerateRequest, GenerateResponse,
    ModelsResponse, StreamDoneEvent, StreamTokenEvent, TokenizeRequest, TokenizeResponse,
};
use crate::generate::{GenerationConfig, SamplingStrategy};
use crate::registry::ModelInfo;
use crate::tokenizer::BPETokenizer;

// ============================================================================
// Shared helpers
// ============================================================================

/// Shorthand for the error tuple used across all gpu_handlers endpoints.
type ApiErr = (StatusCode, Json<ErrorResponse>);

/// Build an API error response.
fn api_err(status: StatusCode, msg: impl std::fmt::Display) -> ApiErr {
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
}

/// Get tokenizer from state or return 500.
fn require_tok(state: &AppState) -> Result<std::sync::Arc<BPETokenizer>, ApiErr> {
    state
        .tokenizer
        .clone()
        .ok_or_else(|| api_err(StatusCode::INTERNAL_SERVER_ERROR, "No tokenizer available"))
}

/// Tokenize a prompt, returning error if empty.
fn tokenize_prompt(tokenizer: &BPETokenizer, prompt: &str) -> Result<Vec<u32>, ApiErr> {
    let ids = tokenizer.encode(prompt);
    if ids.is_empty() {
        return Err(api_err(StatusCode::BAD_REQUEST, "Prompt cannot be empty"));
    }
    Ok(ids)
}

/// Get the EOS token id from tokenizer.
fn eos_id(tokenizer: &BPETokenizer) -> u32 {
    tokenizer
        .get_token_id("<|im_end|>")
        .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
        .unwrap_or(151645)
}

// ============================================================================
// PARITY-022: GPU Batch Inference API
// ============================================================================

/// GPU batch completions request (PARITY-022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchRequest {
    /// List of prompts to process in batch
    pub prompts: Vec<String>,
    /// Maximum tokens to generate per prompt
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy)
    #[serde(default)]
    pub temperature: f32,
    /// Top-k sampling (1 = greedy)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Stop tokens (optional)
    #[serde(default)]
    pub stop: Vec<String>,
}

/// GPU batch completions response (PARITY-022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchResponse {
    /// Results for each prompt
    pub results: Vec<GpuBatchResult>,
    /// Batch statistics
    pub stats: GpuBatchStats,
}

/// Single result in GPU batch response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchResult {
    /// Prompt index
    pub index: usize,
    /// Generated token IDs
    pub token_ids: Vec<u32>,
    /// Decoded text
    pub text: String,
    /// Number of tokens generated
    pub num_generated: usize,
}

/// GPU batch statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchStats {
    /// Batch size
    pub batch_size: usize,
    /// Whether GPU was used
    pub gpu_used: bool,
    /// Total tokens generated
    pub total_tokens: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Throughput in tokens per second
    pub throughput_tps: f64,
}

/// GPU warmup response (PARITY-022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuWarmupResponse {
    /// Whether warmup succeeded
    pub success: bool,
    /// Memory used in bytes
    pub memory_bytes: usize,
    /// Number of layers cached
    pub num_layers: usize,
    /// Message
    pub message: String,
}

/// GPU status response (PARITY-022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatusResponse {
    /// Whether GPU cache is warmed up
    pub cache_ready: bool,
    /// Memory used by cache in bytes
    pub cache_memory_bytes: usize,
    /// GPU batch threshold
    pub batch_threshold: usize,
    /// Recommended minimum batch size
    pub recommended_min_batch: usize,
}

// ==================== PARITY-052: Batch Request Queuing ====================
//
// Infrastructure for continuous batch inference via HTTP API.
// Requests are queued and processed in batches for higher throughput.
//
// Architecture:
//   - BatchConfig: Configuration for batch window and size thresholds
//   - ContinuousBatchRequest: Internal request with oneshot response channel
//   - ContinuousBatchResponse: Result returned via oneshot channel
//   - AppState extensions: batch_scheduler, batch_request_tx, batch_config
// ============================================================================

/// Configuration for continuous batch inference (PARITY-052)
#[derive(Debug, Clone)]
#[cfg(feature = "gpu")]
pub struct BatchConfig {
    /// Maximum time to wait for batch to fill (milliseconds)
    pub window_ms: u64,
    /// Minimum batch size to process (below this, use single-request path)
    pub min_batch: usize,
    /// Optimal batch size for M4 parity (process immediately when reached)
    /// PARITY-095: This also controls GPU batch threshold
    pub optimal_batch: usize,
    /// Maximum batch size (GPU memory constraint)
    pub max_batch: usize,
    /// Channel buffer size for request queue
    pub queue_size: usize,
    /// GPU batch threshold (use GPU path when batch >= this)
    /// PARITY-095: GPU GEMM wins at batch >= 32 (from IMP-600 analysis)
    pub gpu_threshold: usize,
}

#[cfg(feature = "gpu")]
impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            window_ms: 50,     // 50ms batch window (allow time for requests to accumulate)
            min_batch: 4,      // Minimum for any batching benefit
            optimal_batch: 32, // PARITY-095: Aligned with GPU threshold for M4 parity
            max_batch: 64,     // Allow larger batches for better GPU utilization
            queue_size: 1024,  // Request queue buffer
            gpu_threshold: 32, // GPU GEMM crossover point (from PARITY-046b)
        }
    }
}

#[cfg(feature = "gpu")]
impl BatchConfig {
    /// Create config optimized for low latency (smaller batches)
    /// Note: GPU batch disabled (threshold > max_batch) for consistent latency
    pub fn low_latency() -> Self {
        Self {
            window_ms: 5,
            min_batch: 2,
            optimal_batch: 8,
            max_batch: 16,
            queue_size: 512,
            gpu_threshold: 32, // Effectively disabled since max_batch=16
        }
    }

    /// Create config optimized for high throughput (larger batches)
    /// PARITY-095: GPU batch enabled for batch >= 32
    pub fn high_throughput() -> Self {
        Self {
            window_ms: 100, // 100ms window for maximum batching
            min_batch: 8,
            optimal_batch: 32, // Trigger processing at GPU threshold
            max_batch: 128,    // Large batches for maximum throughput
            queue_size: 2048,
            gpu_threshold: 32, // GPU GEMM crossover
        }
    }

    /// Check if batch size is sufficient for processing
    pub fn should_process(&self, batch_size: usize) -> bool {
        batch_size >= self.optimal_batch
    }

    /// Check if batch size meets minimum threshold
    pub fn meets_minimum(&self, batch_size: usize) -> bool {
        batch_size >= self.min_batch
    }
}

/// Internal batch request with response channel (PARITY-052)
#[cfg(feature = "gpu")]
pub struct ContinuousBatchRequest {
    /// Tokenized input prompt
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature
    pub temperature: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Channel to send response back to handler
    pub response_tx: tokio::sync::oneshot::Sender<ContinuousBatchResponse>,
    /// Request timestamp for latency tracking
    pub submitted_at: std::time::Instant,
}

/// Response from batch processor (PARITY-052)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ContinuousBatchResponse {
    /// Generated token IDs (includes prompt)
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens (to skip when decoding)
    pub prompt_len: usize,
    /// Whether request was processed in batch or single-request path
    pub batched: bool,
    /// Batch size when processed (1 for single-request)
    pub batch_size: usize,
    /// Processing latency in milliseconds
    pub latency_ms: f64,
}

#[cfg(feature = "gpu")]
impl ContinuousBatchResponse {
    /// Create response for single-request path
    pub fn single(token_ids: Vec<u32>, prompt_len: usize, latency_ms: f64) -> Self {
        Self {
            token_ids,
            prompt_len,
            batched: false,
            batch_size: 1,
            latency_ms,
        }
    }

    /// Create response for batched path
    pub fn batched(
        token_ids: Vec<u32>,
        prompt_len: usize,
        batch_size: usize,
        latency_ms: f64,
    ) -> Self {
        Self {
            token_ids,
            prompt_len,
            batched: true,
            batch_size,
            latency_ms,
        }
    }

    /// Get generated tokens (excluding prompt)
    pub fn generated_tokens(&self) -> &[u32] {
        if self.token_ids.len() > self.prompt_len {
            &self.token_ids[self.prompt_len..]
        } else {
            &[]
        }
    }
}

/// Batch queue statistics (PARITY-052)
#[derive(Debug, Clone, Default)]
#[cfg(feature = "gpu")]
pub struct BatchQueueStats {
    /// Total requests queued
    pub total_queued: u64,
    /// Total batches processed
    pub total_batches: u64,
    /// Total requests processed via single-request path
    pub total_single: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average queue wait time in milliseconds
    pub avg_wait_ms: f64,
}

// ==================== PARITY-053: Batch Processor Background Task ====================
//
// Background task that processes batched inference requests.
// Collects requests until batch is ready (size threshold or timeout), then processes.
//
// Flow:
//   1. Receive requests via mpsc channel
//   2. Accumulate until batch_size >= optimal_batch OR window_ms timeout
//   3. Process batch using model.generate_with_cache() for each request
//   4. Send results via oneshot channels
//
// Note: True batch inference (single forward pass for multiple requests) requires
// additional model infrastructure. This implementation processes requests in
// parallel within a batch window, which still improves throughput under load.
// ==================================================================================

/// Result from batch processing
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct BatchProcessResult {
    /// Number of requests processed
    pub requests_processed: usize,
    /// Whether processed as batch or single
    pub was_batched: bool,
    /// Total processing time in milliseconds
    pub total_time_ms: f64,
    /// Average latency per request in milliseconds
    pub avg_latency_ms: f64,
}

/// Spawn the batch processor background task (PARITY-053)
///
/// Returns the sender channel for submitting requests.
/// The receiver is consumed by the spawned task.
///
/// # Arguments
/// * `model` - The cached model for inference
/// * `config` - Batch configuration
///
/// # Returns
/// * Sender channel for batch requests
#[cfg(feature = "gpu")]
pub fn spawn_batch_processor(
    model: std::sync::Arc<crate::gguf::OwnedQuantizedModelCachedSync>,
    config: BatchConfig,
) -> tokio::sync::mpsc::Sender<ContinuousBatchRequest> {
    let (tx, rx) = tokio::sync::mpsc::channel(config.queue_size);

    // Spawn the background processor task
    tokio::spawn(batch_processor_task(rx, model, config));

    tx
}

/// Background task that processes batched requests (PARITY-053)
///
/// This task runs continuously, collecting requests and processing them in batches.
/// It uses a timeout-based batching strategy:
/// - Process immediately if batch reaches optimal_batch size
/// - Process on timeout (window_ms) if batch has requests
/// - Fall back to single-request processing for very small batches
#[cfg(feature = "gpu")]
async fn batch_processor_task(
    mut rx: tokio::sync::mpsc::Receiver<ContinuousBatchRequest>,
    model: std::sync::Arc<crate::gguf::OwnedQuantizedModelCachedSync>,
    config: BatchConfig,
) {
    use std::time::{Duration, Instant};
    use tokio::time::timeout;

    let mut batch: Vec<ContinuousBatchRequest> = Vec::with_capacity(config.max_batch);
    let mut window_start = Instant::now();

    loop {
        // Calculate remaining time in window
        let elapsed = window_start.elapsed();
        let remaining = Duration::from_millis(config.window_ms).saturating_sub(elapsed);

        // Try to receive with timeout
        match timeout(remaining, rx.recv()).await {
            Ok(Some(request)) => {
                batch.push(request);

                // Process immediately if we hit optimal batch size
                if batch.len() >= config.optimal_batch {
                    process_batch(&model, &config, &mut batch).await;
                    window_start = Instant::now();
                }
            },
            Ok(None) => {
                // Channel closed, process remaining and exit
                if !batch.is_empty() {
                    process_batch(&model, &config, &mut batch).await;
                }
                break;
            },
            Err(_) => {
                // Timeout - process current batch if we have requests
                if !batch.is_empty() {
                    process_batch(&model, &config, &mut batch).await;
                }
                window_start = Instant::now();
            },
        }
    }
}

/// Process a batch of requests (PARITY-053)
///
/// Processes all requests in the batch and sends results via their oneshot channels.
/// Uses tokio::spawn to process requests concurrently within the batch.
#[cfg(feature = "gpu")]
async fn process_batch(
    model: &std::sync::Arc<crate::gguf::OwnedQuantizedModelCachedSync>,
    config: &BatchConfig,
    batch: &mut Vec<ContinuousBatchRequest>,
) {
    use std::time::Instant;

    if batch.is_empty() {
        return;
    }

    let batch_size = batch.len();
    let batch_start = Instant::now();

    // PARITY-095: Use configurable GPU batch threshold
    // GPU GEMM wins at batch >= gpu_threshold (default 32, from IMP-600 analysis)
    let gpu_threshold = config.gpu_threshold;

    // Use true GPU batch inference if batch is large enough and GPU cache is warm
    if batch_size >= gpu_threshold && model.is_gpu_cache_warm() {
        // PARITY-094: True batch inference with GPU FFN
        // Collect all prompts
        let prompts: Vec<Vec<u32>> = batch.iter().map(|r| r.prompt_tokens.clone()).collect();

        // Use first request's config (batch inference assumes similar parameters)
        let first = &batch[0];
        let gen_config = crate::gguf::QuantizedGenerateConfig {
            max_tokens: first.max_tokens,
            temperature: first.temperature,
            top_k: first.top_k,
            stop_tokens: Vec::new(),
            trace: false,
        };

        // Run batch generation with GPU FFN (PARITY-021)
        let results = model.batch_generate_gpu(&prompts, &gen_config);

        let total_latency_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
        let per_request_latency_ms = total_latency_ms / batch_size as f64;

        // Send responses
        match results {
            Ok(all_token_ids) => {
                for (request, token_ids) in batch.drain(..).zip(all_token_ids.into_iter()) {
                    let response = ContinuousBatchResponse {
                        token_ids,
                        prompt_len: request.prompt_tokens.len(),
                        batched: true,
                        batch_size,
                        latency_ms: per_request_latency_ms,
                    };
                    let _ = request.response_tx.send(response);
                }
            },
            Err(_) => {
                // Fallback: return prompts unchanged on error
                for request in batch.drain(..) {
                    let response = ContinuousBatchResponse {
                        token_ids: request.prompt_tokens.clone(),
                        prompt_len: request.prompt_tokens.len(),
                        batched: false,
                        batch_size,
                        latency_ms: per_request_latency_ms,
                    };
                    let _ = request.response_tx.send(response);
                }
            },
        }
    } else {
        // Concurrent single-request processing (for small batches or no GPU cache)
        let mut handles = Vec::with_capacity(batch_size);

        for request in batch.drain(..) {
            let model = model.clone();
            let handle = tokio::spawn(async move {
                let start = Instant::now();

                // Build generation config
                let gen_config = crate::gguf::QuantizedGenerateConfig {
                    max_tokens: request.max_tokens,
                    temperature: request.temperature,
                    top_k: request.top_k,
                    stop_tokens: Vec::new(),
                    trace: false,
                };

                // Generate
                let result = model.generate_with_cache(&request.prompt_tokens, &gen_config);

                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                // Send response
                let response = match result {
                    Ok(token_ids) => ContinuousBatchResponse {
                        token_ids,
                        prompt_len: request.prompt_tokens.len(),
                        batched: false,
                        batch_size: 1,
                        latency_ms,
                    },
                    Err(_) => ContinuousBatchResponse {
                        token_ids: request.prompt_tokens.clone(),
                        prompt_len: request.prompt_tokens.len(),
                        batched: false,
                        batch_size: 1,
                        latency_ms,
                    },
                };

                // Send response (ignore if receiver dropped)
                let _ = request.response_tx.send(response);
            });

            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let _ = handle.await;
        }
    }
}

/// GPU warmup handler (PARITY-022)
/// POST /v1/gpu/warmup - Warmup GPU cache for batch inference
#[cfg(feature = "gpu")]
pub async fn gpu_warmup_handler(
    State(state): State<AppState>,
) -> Result<Json<GpuWarmupResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(cached_model) = state.cached_model() {
        match cached_model.warmup_gpu_cache() {
            Ok((memory_bytes, num_layers)) => Ok(Json(GpuWarmupResponse {
                success: true,
                memory_bytes,
                num_layers,
                message: format!(
                    "GPU cache warmed up: {} layers, {:.2} GB",
                    num_layers,
                    memory_bytes as f64 / 1e9
                ),
            })),
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("GPU warmup failed: {e}"),
                }),
            )),
        }
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "No GPU-capable model loaded. Use with_cached_model() to enable."
                    .to_string(),
            }),
        ))
    }
}

/// GPU warmup handler stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub async fn gpu_warmup_handler(
    State(_state): State<AppState>,
) -> Result<Json<GpuWarmupResponse>, (StatusCode, Json<ErrorResponse>)> {
    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "GPU feature not enabled. Build with --features gpu".to_string(),
        }),
    ))
}

/// GPU status handler (PARITY-022)
/// GET /v1/gpu/status - Check GPU cache status
#[cfg(feature = "gpu")]
pub async fn gpu_status_handler(
    State(state): State<AppState>,
) -> Result<Json<GpuStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(cached_model) = state.cached_model() {
        Ok(Json(GpuStatusResponse {
            cache_ready: cached_model.is_gpu_cache_warm(),
            cache_memory_bytes: cached_model.gpu_cache_memory(),
            batch_threshold: 32, // GPU GEMM threshold from IMP-600
            recommended_min_batch: 32,
        }))
    } else {
        Ok(Json(GpuStatusResponse {
            cache_ready: false,
            cache_memory_bytes: 0,
            batch_threshold: 32,
            recommended_min_batch: 32,
        }))
    }
}

/// GPU status handler stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub async fn gpu_status_handler(
    State(_state): State<AppState>,
) -> Result<Json<GpuStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    Ok(Json(GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    }))
}

/// GPU batch completions handler (PARITY-022)
/// POST /v1/batch/completions - GPU-accelerated batch inference
#[cfg(feature = "gpu")]
pub async fn gpu_batch_completions_handler(
    State(state): State<AppState>,
    Json(request): Json<GpuBatchRequest>,
) -> Result<Json<GpuBatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    use std::time::Instant;

    if request.prompts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Prompts array cannot be empty".to_string(),
            }),
        ));
    }

    let Some(cached_model) = state.cached_model() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "No GPU-capable model loaded".to_string(),
            }),
        ));
    };

    // Check if GPU cache is ready
    let gpu_ready = cached_model.is_gpu_cache_warm();
    let batch_size = request.prompts.len();

    // Tokenize all prompts
    // For GPU batch, we need token IDs as Vec<Vec<u32>>
    let prompts_tokens: Vec<Vec<u32>> = request
        .prompts
        .iter()
        .map(|p| {
            // Simple tokenization for batch - uses model's vocab
            // In production, use a proper tokenizer
            p.bytes().map(|b| b as u32).collect()
        })
        .collect();

    // Create generation config
    let gen_config = crate::gguf::QuantizedGenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: request.top_k,
        stop_tokens: vec![],
        trace: false,
    };

    let start = Instant::now();

    // Decide GPU vs CPU path based on cache readiness and batch size
    let gpu_threshold = 32;
    let use_gpu = gpu_ready && batch_size >= gpu_threshold;

    let results = if use_gpu {
        // GPU batch inference path
        match cached_model.batch_generate_gpu(&prompts_tokens, &gen_config) {
            Ok(generated) => generated,
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("GPU batch generation failed: {e}"),
                    }),
                ));
            },
        }
    } else {
        // CPU sequential path (fallback)
        let mut results = Vec::with_capacity(batch_size);
        for prompt in &prompts_tokens {
            match cached_model.generate_with_cache(prompt, &gen_config) {
                Ok(tokens) => results.push(tokens),
                Err(e) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("Generation failed: {e}"),
                        }),
                    ));
                },
            }
        }
        results
    };

    let elapsed = start.elapsed();
    let total_tokens: usize = results.iter().map(Vec::len).sum();
    let throughput_tps = total_tokens as f64 / elapsed.as_secs_f64();

    // Build response
    let batch_results: Vec<GpuBatchResult> = results
        .into_iter()
        .enumerate()
        .map(|(idx, tokens)| {
            let prompt_len = prompts_tokens.get(idx).map_or(0, Vec::len);
            let num_generated = tokens.len().saturating_sub(prompt_len);
            GpuBatchResult {
                index: idx,
                token_ids: tokens.clone(),
                text: tokens.iter().map(|&t| t as u8 as char).collect(),
                num_generated,
            }
        })
        .collect();

    Ok(Json(GpuBatchResponse {
        results: batch_results,
        stats: GpuBatchStats {
            batch_size,
            gpu_used: use_gpu,
            total_tokens,
            processing_time_ms: elapsed.as_secs_f64() * 1000.0,
            throughput_tps,
        },
    }))
}

/// GPU batch completions handler stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub async fn gpu_batch_completions_handler(
    State(_state): State<AppState>,
    Json(_request): Json<GpuBatchRequest>,
) -> Result<Json<GpuBatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "GPU feature not enabled. Build with --features gpu".to_string(),
        }),
    ))
}

/// Models list handler - returns available models in multi-model mode
pub async fn models_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelsResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(registry) = &state.registry {
        let models = registry.list();
        Ok(Json(ModelsResponse { models }))
    } else {
        // Single model mode - return the single model info
        Ok(Json(ModelsResponse {
            models: vec![ModelInfo {
                id: "default".to_string(),
                name: "Default Model".to_string(),
                description: "Single model deployment".to_string(),
                format: "unknown".to_string(),
                loaded: true,
            }],
        }))
    }
}

/// Tokenize text handler
pub async fn tokenize_handler(
    State(state): State<AppState>,
    Json(request): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (_model, tokenizer) = state.get_model(request.model_id.as_deref()).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let token_ids = tokenizer.encode(&request.text);
    let num_tokens = token_ids.len();

    Ok(Json(TokenizeResponse {
        token_ids,
        num_tokens,
    }))
}

// ── generate_handler backend dispatch ────────────────────────────────

/// Generate text handler
#[cfg(feature = "cuda")]
fn try_cuda_generate(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<Option<GenerateResponse>, ApiErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let cuda_model_lock = match state.cuda_model() {
        Some(l) => l,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;
    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt_tokens = prompt_ids.len();

    let q_config = QuantizedGenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: if request.temperature == 0.0 {
            1
        } else {
            request.top_k
        },
        stop_tokens: vec![eos_id(&tokenizer)],
        trace: false,
    };

    let mut cuda_model = cuda_model_lock.write().map_err(|_| {
        api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to acquire CUDA model lock",
        )
    })?;
    let generated = cuda_model
        .generate_gpu_resident(&prompt_ids, &q_config)
        .map_err(|e| {
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("CUDA generation failed: {e}"),
            )
        })?;
    let text = tokenizer
        .decode(&generated)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Some(GenerateResponse {
        num_generated: generated.len().saturating_sub(prompt_tokens),
        token_ids: generated,
        text,
    }))
}

fn try_quantized_generate(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<Option<GenerateResponse>, ApiErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let quantized_model = match state.quantized_model() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;
    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt_tokens = prompt_ids.len();

    let q_config = QuantizedGenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: if request.temperature == 0.0 {
            1
        } else {
            request.top_k
        },
        stop_tokens: vec![eos_id(&tokenizer)],
        trace: false,
    };

    let generated = quantized_model
        .generate(&prompt_ids, &q_config)
        .map_err(|e| {
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("CPU generation failed: {e}"),
            )
        })?;
    let text = tokenizer
        .decode(&generated)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Some(GenerateResponse {
        num_generated: generated.len().saturating_sub(prompt_tokens),
        token_ids: generated,
        text,
    }))
}

fn try_apr_generate(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<Option<GenerateResponse>, ApiErr> {
    use crate::apr_transformer::GenerateConfig;

    let apr_transformer = match state.apr_transformer() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;
    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt_tokens = prompt_ids.len();

    let gen_config = GenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        ..Default::default()
    };

    let generated = apr_transformer
        .generate_with_cache(&prompt_ids, &gen_config)
        .map_err(|e| {
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("APR generation failed: {e}"),
            )
        })?;
    let text = tokenizer
        .decode(&generated)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Some(GenerateResponse {
        num_generated: generated.len().saturating_sub(prompt_tokens),
        token_ids: generated,
        text,
    }))
}

fn registry_generate(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<GenerateResponse, ApiErr> {
    let (model, tokenizer) = state
        .get_model(request.model_id.as_deref())
        .map_err(|e| api_err(StatusCode::NOT_FOUND, e))?;

    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    let strategy = match request.strategy.as_str() {
        "greedy" => SamplingStrategy::Greedy,
        "top_k" => SamplingStrategy::TopK { k: request.top_k },
        "top_p" => SamplingStrategy::TopP { p: request.top_p },
        other => {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Invalid strategy: {other}"),
            ))
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);
    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    let generated = model
        .generate(&prompt, &config)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| {
                api_err(
                    StatusCode::BAD_REQUEST,
                    format!("Token ID {id} exceeds u32 range"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(GenerateResponse {
        num_generated: generated.len() - prompt.len(),
        token_ids,
        text,
    })
}

pub async fn generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, ApiErr> {
    use std::time::Instant;
    let start = Instant::now();

    if state.is_verbose() {
        eprintln!(
            "[VERBOSE] POST /generate prompt={:?} max_tokens={}",
            &request.prompt.chars().take(50).collect::<String>(),
            request.max_tokens
        );
    }

    #[cfg(feature = "cuda")]
    if let Some(resp) = try_cuda_generate(&state, &request)? {
        state
            .metrics
            .record_success(resp.num_generated, start.elapsed());
        return Ok(Json(resp));
    }

    if let Some(resp) = try_quantized_generate(&state, &request)? {
        state
            .metrics
            .record_success(resp.num_generated, start.elapsed());
        return Ok(Json(resp));
    }

    if let Some(resp) = try_apr_generate(&state, &request)? {
        state
            .metrics
            .record_success(resp.num_generated, start.elapsed());
        return Ok(Json(resp));
    }

    let resp = registry_generate(&state, &request)?;
    state
        .metrics
        .record_success(resp.num_generated, start.elapsed());
    Ok(Json(resp))
}

/// Batch tokenize handler
pub async fn batch_tokenize_handler(
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

    // Get tokenizer (use default model)
    let (_model, tokenizer) = state.get_model(None).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Tokenize all texts
    let results: Vec<TokenizeResponse> = request
        .texts
        .iter()
        .map(|text| {
            let token_ids = tokenizer.encode(text);
            let num_tokens = token_ids.len();
            TokenizeResponse {
                token_ids,
                num_tokens,
            }
        })
        .collect();

    Ok(Json(BatchTokenizeResponse { results }))
}

// ── batch_generate_handler backend dispatch ─────────────────────────

/// Batch generate handler
#[cfg(feature = "cuda")]
fn try_cuda_batch_generate(
    state: &AppState,
    request: &BatchGenerateRequest,
) -> Result<Option<Vec<GenerateResponse>>, ApiErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let cuda_model_lock = match state.cuda_model() {
        Some(l) => l,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;

    let q_config = QuantizedGenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: if request.temperature == 0.0 {
            1
        } else {
            request.top_k
        },
        stop_tokens: vec![eos_id(&tokenizer)],
        trace: false,
    };

    let mut results = Vec::with_capacity(request.prompts.len());
    let mut cuda_model = cuda_model_lock.write().map_err(|_| {
        api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to acquire CUDA model lock",
        )
    })?;

    for prompt_text in &request.prompts {
        let prompt_ids = tokenizer.encode(prompt_text);
        if prompt_ids.is_empty() {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Prompt '{prompt_text}' tokenizes to empty sequence"),
            ));
        }
        let prompt_tokens = prompt_ids.len();
        let generated = cuda_model
            .generate_gpu_resident(&prompt_ids, &q_config)
            .map_err(|e| {
                api_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("CUDA generation failed: {e}"),
                )
            })?;
        let text = tokenizer
            .decode(&generated)
            .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        results.push(GenerateResponse {
            num_generated: generated.len().saturating_sub(prompt_tokens),
            token_ids: generated,
            text,
        });
    }

    Ok(Some(results))
}

fn try_apr_batch_generate(
    state: &AppState,
    request: &BatchGenerateRequest,
) -> Result<Option<Vec<GenerateResponse>>, ApiErr> {
    use crate::apr_transformer::GenerateConfig;

    let apr_transformer = match state.apr_transformer() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;

    let gen_config = GenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        ..Default::default()
    };

    let mut results = Vec::with_capacity(request.prompts.len());

    for prompt_text in &request.prompts {
        let prompt_ids = tokenizer.encode(prompt_text);
        if prompt_ids.is_empty() {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Prompt '{prompt_text}' tokenizes to empty sequence"),
            ));
        }
        let prompt_tokens = prompt_ids.len();
        let generated = apr_transformer
            .generate_with_cache(&prompt_ids, &gen_config)
            .map_err(|e| {
                api_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("APR generation failed: {e}"),
                )
            })?;
        let text = tokenizer
            .decode(&generated)
            .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        results.push(GenerateResponse {
            num_generated: generated.len().saturating_sub(prompt_tokens),
            token_ids: generated,
            text,
        });
    }

    Ok(Some(results))
}

fn registry_batch_generate(
    state: &AppState,
    request: &BatchGenerateRequest,
) -> Result<Vec<GenerateResponse>, ApiErr> {
    let (model, tokenizer) = state
        .get_model(None)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let strategy = match request.strategy.as_str() {
        "greedy" => SamplingStrategy::Greedy,
        "top_k" => SamplingStrategy::TopK { k: request.top_k },
        "top_p" => SamplingStrategy::TopP { p: request.top_p },
        other => {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Invalid strategy: {other}"),
            ))
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);
    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    let mut results = Vec::with_capacity(request.prompts.len());

    for prompt_text in &request.prompts {
        let prompt_ids = tokenizer.encode(prompt_text);
        if prompt_ids.is_empty() {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Prompt '{prompt_text}' tokenizes to empty sequence"),
            ));
        }
        let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();
        let generated = model
            .generate(&prompt, &config)
            .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let token_ids: Vec<u32> = generated
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| {
                    api_err(
                        StatusCode::BAD_REQUEST,
                        format!("Token ID {id} exceeds u32 range"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let text = tokenizer
            .decode(&token_ids)
            .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        results.push(GenerateResponse {
            num_generated: generated.len() - prompt.len(),
            token_ids,
            text,
        });
    }

    Ok(results)
}

pub async fn batch_generate_handler(
    State(state): State<AppState>,
    Json(request): Json<BatchGenerateRequest>,
) -> Result<Json<BatchGenerateResponse>, ApiErr> {
    if request.prompts.is_empty() {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "Prompts array cannot be empty",
        ));
    }

    #[cfg(feature = "cuda")]
    if let Some(results) = try_cuda_batch_generate(&state, &request)? {
        return Ok(Json(BatchGenerateResponse { results }));
    }

    if let Some(results) = try_apr_batch_generate(&state, &request)? {
        return Ok(Json(BatchGenerateResponse { results }));
    }

    let results = registry_batch_generate(&state, &request)?;
    Ok(Json(BatchGenerateResponse { results }))
}

/// Stream generate handler - generates tokens one by one via Server-Sent Events
pub async fn stream_generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    // NOTE: Streaming via CUDA model uses /v1/chat/completions endpoint with stream=true
    // This handler uses the CPU model path; for GPU streaming use OpenAI-compatible endpoint

    // Get model and tokenizer
    let (model, tokenizer) = state.get_model(request.model_id.as_deref()).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Tokenize prompt
    let prompt_ids = tokenizer.encode(&request.prompt);
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
    let generated = match model.generate(&prompt, &config) {
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

    // Convert to u32 with proper overflow handling
    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("Token ID {id} exceeds u32 range"),
                    }),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Create stream that emits tokens one by one
    let tokenizer_clone = tokenizer;
    let stream = async_stream::stream! {
        // Skip prompt tokens, only stream generated tokens
        for &token_id in &token_ids[prompt_len..] {
            // Decode single token
            let text = match tokenizer_clone.decode(&[token_id]) {
                Ok(t) => t,
                Err(_) => String::from("<error>"),
            };

            let event = StreamTokenEvent { token_id, text };
            // Serialization of simple struct should not fail, but handle gracefully
            let data = serde_json::to_string(&event)
                .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string());

            yield Ok::<_, Infallible>(Event::default().event("token").data(data));
        }

        // Send done event
        let done_event = StreamDoneEvent {
            num_generated: token_ids.len() - prompt_len,
        };
        // Serialization of simple struct should not fail, but handle gracefully
        let data = serde_json::to_string(&done_event)
            .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string());
        yield Ok(Event::default().event("done").data(data));
    };

    Ok(Sse::new(stream))
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
#[path = "gpu_handlers_tests.rs"]
mod gpu_handlers_tests;
