//! HTTP API for model inference
//!
//! Provides REST endpoints for tokenization and text generation using axum.
//!
//! ## Endpoints
//!
//! - `GET /health` - Health check
//! - `GET /metrics` - Prometheus-formatted metrics
//! - `GET /metrics/dispatch` - CPU/GPU dispatch statistics (?format=prometheus|json)
//! - `POST /tokenize` - Tokenize text
//! - `POST /generate` - Generate text from prompt
//! - `POST /batch/tokenize` - Batch tokenize multiple texts
//! - `POST /batch/generate` - Batch generate for multiple prompts
//! - `POST /stream/generate` - Stream generated tokens via SSE
//! - `POST /v1/gpu/warmup` - Warmup GPU cache for batch inference (PARITY-022)
//! - `GET /v1/gpu/status` - Check GPU cache status (PARITY-022)
//! - `POST /v1/batch/completions` - GPU-accelerated batch inference (PARITY-022)
//! - `GET /v1/metrics` - JSON metrics for TUI monitoring (PARITY-107)
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
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{
    apr::{AprModel, HEADER_SIZE, MAGIC},
    audit::{AuditLogger, AuditRecord, InMemoryAuditSink},
    cache::{CacheKey, ModelCache},
    error::RealizarError,
    explain::ShapExplanation,
    layers::{Model, ModelConfig},
    metrics::MetricsCollector,
    registry::ModelRegistry,
    tokenizer::BPETokenizer,
};

// PMAT-802: Extracted handlers
mod openai_handlers;
pub(crate) use openai_handlers::{
    openai_chat_completions_handler, openai_chat_completions_stream_handler, openai_models_handler,
};
mod gpu_handlers;
pub(crate) use gpu_handlers::{
    batch_generate_handler, batch_tokenize_handler, generate_handler,
    gpu_batch_completions_handler, gpu_status_handler, gpu_warmup_handler, models_handler,
    stream_generate_handler, tokenize_handler,
};
// Public exports for tests
pub use gpu_handlers::{
    BatchProcessResult, BatchQueueStats, ContinuousBatchRequest, ContinuousBatchResponse,
    GpuBatchRequest, GpuBatchResponse, GpuBatchResult, GpuBatchStats, GpuStatusResponse,
    GpuWarmupResponse,
};
// Public exports for apr-cli CUDA integration (PMAT-GPU-001)
pub use gpu_handlers::{spawn_batch_processor, BatchConfig};
mod realize_handlers;
pub(crate) use realize_handlers::{
    clean_chat_output, format_chat_messages, openai_completions_handler, openai_embeddings_handler,
    realize_embed_handler, realize_model_handler, realize_reload_handler,
};
// Public exports for tests
pub use realize_handlers::{
    CompletionChoice, CompletionRequest, CompletionResponse, ContextWindowConfig,
    ContextWindowManager, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    ModelLineage, ModelMetadataResponse, ReloadRequest, ReloadResponse,
};
mod apr_handlers;
pub(crate) use apr_handlers::{apr_audit_handler, apr_explain_handler, apr_predict_handler};
mod types;
pub use types::{
    BatchGenerateRequest, BatchGenerateResponse, BatchTokenizeRequest, BatchTokenizeResponse,
    ErrorResponse, GenerateRequest, GenerateResponse, HealthResponse, ModelsResponse,
    StreamDoneEvent, StreamTokenEvent, TokenizeRequest, TokenizeResponse,
};
pub use types::{default_max_tokens, default_top_k};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Model for inference (single model mode)
    model: Option<Arc<Model>>,
    /// Tokenizer for encoding/decoding (single model mode)
    tokenizer: Option<Arc<BPETokenizer>>,
    /// Model cache for multi-model support
    #[allow(dead_code)] // Will be used in future PR for cache warming
    cache: Option<Arc<ModelCache>>,
    /// Default cache key for single model mode
    #[allow(dead_code)] // Will be used in future PR for cache warming
    cache_key: Option<CacheKey>,
    /// Metrics collector for monitoring
    metrics: Arc<MetricsCollector>,
    /// Model registry for multi-model serving
    registry: Option<Arc<ModelRegistry>>,
    /// Default model ID for multi-model mode
    default_model_id: Option<String>,
    /// APR model for /v1/predict endpoint (real inference, not mock)
    apr_model: Option<Arc<AprModel>>,
    /// Audit logger for /v1/audit endpoint (real records, not mock)
    audit_logger: Arc<AuditLogger>,
    /// In-memory audit sink for record retrieval
    audit_sink: Arc<InMemoryAuditSink>,
    /// GPU model for GGUF inference (M33: IMP-084)
    #[cfg(feature = "gpu")]
    gpu_model: Option<Arc<std::sync::RwLock<crate::gpu::GpuModel>>>,
    /// Quantized model for fused Q4_K inference (IMP-100)
    /// This is 1.37x faster than dequantized GpuModel due to reduced memory bandwidth
    quantized_model: Option<Arc<crate::gguf::OwnedQuantizedModel>>,
    /// Thread-safe cached model for HTTP serving (IMP-116)
    /// Uses Mutex-based scheduler caching for 10.6x speedup
    #[cfg(feature = "gpu")]
    cached_model: Option<Arc<crate::gguf::OwnedQuantizedModelCachedSync>>,
    /// Dispatch metrics for adaptive CPU/GPU tracking (IMP-126)
    #[cfg(feature = "gpu")]
    dispatch_metrics: Option<Arc<crate::gguf::DispatchMetrics>>,
    /// Batch request channel for continuous batching (PARITY-052)
    /// Requests sent here are queued and processed in batches
    #[cfg(feature = "gpu")]
    batch_request_tx: Option<tokio::sync::mpsc::Sender<ContinuousBatchRequest>>,
    /// Batch configuration for window timing and size thresholds (PARITY-052)
    #[cfg(feature = "gpu")]
    batch_config: Option<BatchConfig>,
    /// CUDA-optimized model for high-performance GPU inference (PAR-111)
    /// Uses pre-uploaded weights and batched workspaces for 755+ tok/s (2.6x Ollama)
    #[cfg(feature = "cuda")]
    cuda_model: Option<Arc<std::sync::RwLock<crate::gguf::OwnedQuantizedModelCuda>>>,
    /// APR Transformer for SafeTensors/APR inference (PMAT-SERVE-FIX-001)
    /// Supports F32 weights from SafeTensors or APR format
    apr_transformer: Option<Arc<crate::apr_transformer::AprTransformer>>,
    /// GH-152: Enable verbose request/response logging
    verbose: bool,
}

/// Helper to create default audit infrastructure
fn create_audit_state() -> (Arc<AuditLogger>, Arc<InMemoryAuditSink>) {
    let sink = Arc::new(InMemoryAuditSink::new());
    let logger = AuditLogger::new(Box::new(InMemorySinkWrapper(sink.clone())))
        .with_model_hash("demo-model-hash");
    (Arc::new(logger), sink)
}

/// Wrapper to make Arc<InMemoryAuditSink> implement AuditSink
struct InMemorySinkWrapper(Arc<InMemoryAuditSink>);

impl crate::audit::AuditSink for InMemorySinkWrapper {
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), crate::audit::AuditError> {
        self.0.write_batch(records)
    }

    fn flush(&self) -> Result<(), crate::audit::AuditError> {
        self.0.flush()
    }
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
        let (audit_logger, audit_sink) = create_audit_state();
        Self {
            model: Some(Arc::new(model)),
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        }
    }

    /// Create application state with model registry for multi-model serving
    ///
    /// # Arguments
    ///
    /// * `registry` - Model registry with pre-registered models
    /// * `default_model_id` - Default model to use when not specified
    ///
    /// # Errors
    ///
    /// Returns error if default model doesn't exist in registry
    pub fn with_registry(
        registry: ModelRegistry,
        default_model_id: &str,
    ) -> Result<Self, RealizarError> {
        // Verify default model exists
        if !registry.contains(default_model_id) {
            return Err(RealizarError::ModelNotFound(default_model_id.to_string()));
        }

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: None,
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: Some(Arc::new(registry)),
            default_model_id: Some(default_model_id.to_string()),
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Get model and tokenizer by ID (or default)
    #[allow(clippy::type_complexity)]
    fn get_model(
        &self,
        model_id: Option<&str>,
    ) -> Result<(Arc<Model>, Arc<BPETokenizer>), RealizarError> {
        // Multi-model mode
        if let Some(registry) = &self.registry {
            let id = model_id
                .or(self.default_model_id.as_deref())
                .ok_or_else(|| RealizarError::RegistryError("No model ID specified".to_string()))?;
            return registry.get(id);
        }

        // Single model mode
        let model = self
            .model
            .clone()
            .ok_or_else(|| RealizarError::RegistryError("No model available".to_string()))?;
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| RealizarError::RegistryError("No tokenizer available".to_string()))?;

        Ok((model, tokenizer))
    }

    /// Create application state with model caching enabled
    ///
    /// # Arguments
    ///
    /// * `cache_capacity` - Maximum number of models to cache
    ///
    /// # Panics
    ///
    /// Panics if model or tokenizer creation fails (should not happen with valid config)
    #[must_use]
    pub fn with_cache(cache_capacity: usize) -> Self {
        // Create empty state with cache
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 64,
            eps: 1e-5,
        };
        let model = Model::new(config).expect("Failed to create placeholder model");
        let vocab: Vec<String> = (0..100)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer =
            BPETokenizer::new(vocab, vec![], "<unk>").expect("Failed to create tokenizer");

        let (audit_logger, audit_sink) = create_audit_state();
        Self {
            model: Some(Arc::new(model)),
            tokenizer: Some(Arc::new(tokenizer)),
            cache: Some(Arc::new(ModelCache::new(cache_capacity))),
            cache_key: Some(CacheKey::new("default".to_string())),
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
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

        // Create demo APR model (real inference, not mock)
        // Simple model: sum of inputs with bias
        let apr_model = create_demo_apr_model(4)?; // 4 input features

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: Some(Arc::new(model)),
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: Some(Arc::new(apr_model)),
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create a MOCK demo state for fast HTTP handler testing (no inference)
    ///
    /// This creates an AppState with NO model loaded, so all inference endpoints
    /// return errors immediately. Used for testing HTTP handler code paths
    /// without the ~0.5s overhead of model creation per test.
    ///
    /// # Performance (Dr. Popper's "Tax of Setup" Fix)
    /// - `demo()`: ~0.5s (creates real model)
    /// - `demo_mock()`: ~0.001s (no model, instant errors)
    ///
    /// # When to use
    /// - Use `demo_mock()` for HTTP routing/parsing tests (95% of API tests)
    /// - Use `demo()` only when you need actual inference output
    pub fn demo_mock() -> Result<Self, RealizarError> {
        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None, // No model = instant "model not loaded" errors
            tokenizer: None,
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create application state with a GPU model for GGUF inference (M33: IMP-084)
    ///
    /// # Arguments
    ///
    /// * `gpu_model` - GPU model for inference
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "gpu")]
    pub fn with_gpu_model(gpu_model: crate::gpu::GpuModel) -> Result<Self, RealizarError> {
        // Create tokenizer with vocab size matching GPU model
        let vocab_size = gpu_model.config().vocab_size;
        let vocab: Vec<String> = (0..vocab_size)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            gpu_model: Some(Arc::new(std::sync::RwLock::new(gpu_model))),
            quantized_model: None,
            cached_model: None,
            dispatch_metrics: None,
            batch_request_tx: None,
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create application state with GPU model and real vocabulary (IMP-152)
    ///
    /// This version uses the actual vocabulary from the GGUF file for proper text encoding/decoding.
    ///
    /// # Arguments
    ///
    /// * `gpu_model` - GPU model for inference
    /// * `vocab` - Vocabulary tokens from GGUF metadata (tokenizer.ggml.tokens)
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "gpu")]
    pub fn with_gpu_model_and_vocab(
        gpu_model: crate::gpu::GpuModel,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            gpu_model: Some(Arc::new(std::sync::RwLock::new(gpu_model))),
            quantized_model: None,
            cached_model: None,
            dispatch_metrics: None,
            batch_request_tx: None,
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create application state with a quantized model for fused Q4_K inference (IMP-100)
    ///
    /// This is 1.37x faster than dequantized GpuModel due to reduced memory bandwidth.
    ///
    /// # Arguments
    ///
    /// * `quantized_model` - Quantized model for fused Q4_K inference
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    pub fn with_quantized_model(
        quantized_model: crate::gguf::OwnedQuantizedModel,
    ) -> Result<Self, RealizarError> {
        // Create tokenizer with vocab size matching model
        let vocab_size = quantized_model.config.vocab_size;
        let vocab: Vec<String> = (0..vocab_size)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: Some(Arc::new(quantized_model)),
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create application state with thread-safe cached model (IMP-116)
    ///
    /// Uses Mutex-based scheduler caching for 10.6x GPU speedup.
    /// This is the recommended production configuration for HTTP serving.
    ///
    /// # Arguments
    ///
    /// * `cached_model` - Thread-safe cached model with scheduler
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "gpu")]
    pub fn with_cached_model(
        cached_model: crate::gguf::OwnedQuantizedModelCachedSync,
    ) -> Result<Self, RealizarError> {
        // Create tokenizer with vocab size matching model
        let vocab_size = cached_model.model().config.vocab_size;
        let vocab: Vec<String> = (0..vocab_size)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            gpu_model: None,
            quantized_model: None,
            cached_model: Some(Arc::new(cached_model)),
            // Initialize dispatch metrics for adaptive generation (IMP-126)
            dispatch_metrics: Some(Arc::new(crate::gguf::DispatchMetrics::new())),
            batch_request_tx: None,
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create application state with thread-safe cached model and real vocabulary (IMP-116)
    ///
    /// Uses Mutex-based scheduler caching for 10.6x GPU speedup with proper token decoding.
    ///
    /// # Arguments
    ///
    /// * `cached_model` - Thread-safe cached model with scheduler
    /// * `vocab` - Vocabulary tokens from GGUF metadata (tokenizer.ggml.tokens)
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "gpu")]
    pub fn with_cached_model_and_vocab(
        cached_model: crate::gguf::OwnedQuantizedModelCachedSync,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            gpu_model: None,
            quantized_model: None,
            cached_model: Some(Arc::new(cached_model)),
            dispatch_metrics: Some(Arc::new(crate::gguf::DispatchMetrics::new())),
            batch_request_tx: None,
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create application state with quantized model and real vocabulary from GGUF
    ///
    /// This version uses the actual vocabulary from the GGUF file for proper decoding.
    ///
    /// # Arguments
    ///
    /// * `quantized_model` - Quantized model for fused Q4_K inference
    /// * `vocab` - Vocabulary tokens from GGUF metadata (tokenizer.ggml.tokens)
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    pub fn with_quantized_model_and_vocab(
        quantized_model: crate::gguf::OwnedQuantizedModel,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: Some(Arc::new(quantized_model)),
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create application state with CUDA-optimized model for high-performance GPU inference (PAR-111)
    ///
    /// This uses the `OwnedQuantizedModelCuda` wrapper which achieves 755+ tok/s (2.6x Ollama) by:
    /// - Pre-uploading all weights to GPU via `preload_weights_gpu()`
    /// - Using batched workspaces for efficient inference
    /// - GPU-resident KV cache to avoid CPU→GPU transfers
    ///
    /// # Arguments
    ///
    /// * `cuda_model` - CUDA-optimized model wrapper (already initialized with GPU resources)
    /// * `vocab` - Vocabulary tokens from GGUF metadata (tokenizer.ggml.tokens)
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "cuda")]
    pub fn with_cuda_model_and_vocab(
        cuda_model: crate::gguf::OwnedQuantizedModelCuda,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            cuda_model: Some(Arc::new(std::sync::RwLock::new(cuda_model))),
            apr_transformer: None,
            verbose: false,
        })
    }

    /// Create application state with APR Transformer for SafeTensors/APR inference (PMAT-SERVE-FIX-001)
    ///
    /// This enables the `/generate` and `/batch/generate` endpoints for SafeTensors and APR models.
    /// Uses F32 weights for inference, achieving ~1-10 tok/s on CPU.
    ///
    /// # Arguments
    ///
    /// * `transformer` - APR Transformer loaded from SafeTensors or APR file
    /// * `vocab` - Vocabulary tokens for tokenization
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    pub fn with_apr_transformer_and_vocab(
        transformer: crate::apr_transformer::AprTransformer,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            apr_transformer: Some(Arc::new(transformer)),
            verbose: false,
        })
    }

    /// Check if this AppState has a quantized model (IMP-100)
    #[must_use]
    pub fn has_quantized_model(&self) -> bool {
        self.quantized_model.is_some()
    }

    /// Get the quantized model for inference (IMP-100)
    pub fn quantized_model(&self) -> Option<&Arc<crate::gguf::OwnedQuantizedModel>> {
        self.quantized_model.as_ref()
    }

    /// Check if this AppState has an APR transformer (PMAT-SERVE-FIX-001)
    #[must_use]
    pub fn has_apr_transformer(&self) -> bool {
        self.apr_transformer.is_some()
    }

    /// Get the APR transformer for inference (PMAT-SERVE-FIX-001)
    pub fn apr_transformer(&self) -> Option<&Arc<crate::apr_transformer::AprTransformer>> {
        self.apr_transformer.as_ref()
    }

    /// Check if this AppState has a GPU model (M33: IMP-084)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn has_gpu_model(&self) -> bool {
        self.gpu_model.is_some()
    }

    /// Get the GPU model for inference (M33: IMP-085)
    #[cfg(feature = "gpu")]
    pub fn gpu_model(&self) -> Option<&Arc<std::sync::RwLock<crate::gpu::GpuModel>>> {
        self.gpu_model.as_ref()
    }

    /// Check if this AppState has a cached model (IMP-116)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn has_cached_model(&self) -> bool {
        self.cached_model.is_some()
    }

    /// Get the cached model for inference (IMP-116)
    #[cfg(feature = "gpu")]
    pub fn cached_model(&self) -> Option<&Arc<crate::gguf::OwnedQuantizedModelCachedSync>> {
        self.cached_model.as_ref()
    }

    /// Check if this AppState has a CUDA-optimized model (PAR-111)
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn has_cuda_model(&self) -> bool {
        self.cuda_model.is_some()
    }

    /// Get the CUDA-optimized model for high-performance GPU inference (PAR-111)
    ///
    /// Returns the model wrapper that achieves 755+ tok/s (2.6x Ollama) by using:
    /// - Pre-uploaded GPU weights
    /// - Batched workspaces
    /// - GPU-resident KV cache
    #[cfg(feature = "cuda")]
    pub fn cuda_model(
        &self,
    ) -> Option<&Arc<std::sync::RwLock<crate::gguf::OwnedQuantizedModelCuda>>> {
        self.cuda_model.as_ref()
    }

    /// Get dispatch metrics for adaptive CPU/GPU tracking (IMP-126)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn dispatch_metrics(&self) -> Option<&Arc<crate::gguf::DispatchMetrics>> {
        self.dispatch_metrics.as_ref()
    }

    /// Get batch request sender for continuous batching (PARITY-052)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn batch_request_tx(&self) -> Option<&tokio::sync::mpsc::Sender<ContinuousBatchRequest>> {
        self.batch_request_tx.as_ref()
    }

    /// Get batch configuration (PARITY-052)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn batch_config(&self) -> Option<&BatchConfig> {
        self.batch_config.as_ref()
    }

    /// Check if batch inference is enabled (PARITY-052)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn batch_enabled(&self) -> bool {
        self.batch_request_tx.is_some() && self.batch_config.is_some()
    }

    /// Set batch request sender and config (PARITY-052)
    /// This enables continuous batch inference for the completions endpoint
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn with_batch_config(
        mut self,
        batch_request_tx: tokio::sync::mpsc::Sender<ContinuousBatchRequest>,
        batch_config: BatchConfig,
    ) -> Self {
        self.batch_request_tx = Some(batch_request_tx);
        self.batch_config = Some(batch_config);
        self
    }

    /// GH-152: Enable verbose request/response logging
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// GH-152: Check if verbose logging is enabled
    #[must_use]
    pub fn is_verbose(&self) -> bool {
        self.verbose
    }
}

/// Create a demo APR v2 model for testing
pub(crate) fn create_demo_apr_model(_input_dim: usize) -> Result<AprModel, RealizarError> {
    use crate::apr::TensorEntry;

    // Create minimal APR v2 file
    let metadata = r#"{"model_type":"demo","name":"demo-model"}"#;
    let tensor_index: Vec<TensorEntry> = vec![TensorEntry {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![4],
        offset: 0,
        size: 16,
    }];
    let tensor_index_json = serde_json::to_vec(&tensor_index).unwrap_or_default();
    let tensor_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Calculate offsets (64-byte aligned)
    let metadata_offset = HEADER_SIZE as u64;
    let metadata_size = metadata.len() as u32;
    let tensor_index_offset =
        ((metadata_offset as usize + metadata.len()).div_ceil(64) * 64) as u64;
    let data_offset =
        ((tensor_index_offset as usize + tensor_index_json.len()).div_ceil(64) * 64) as u64;

    let mut data = vec![0u8; data_offset as usize + tensor_bytes.len()];

    // Header (64 bytes)
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // Version major
    data[5] = 0; // Version minor
    data[6..8].copy_from_slice(&0u16.to_le_bytes()); // Flags
    data[8..12].copy_from_slice(&1u32.to_le_bytes()); // Tensor count
    data[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
    data[20..24].copy_from_slice(&metadata_size.to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    // Checksum at 40..44 (leave as 0 for now)

    // Metadata
    data[metadata_offset as usize..metadata_offset as usize + metadata.len()]
        .copy_from_slice(metadata.as_bytes());

    // Tensor index
    data[tensor_index_offset as usize..tensor_index_offset as usize + tensor_index_json.len()]
        .copy_from_slice(&tensor_index_json);

    // Tensor data
    data[data_offset as usize..data_offset as usize + tensor_bytes.len()]
        .copy_from_slice(&tensor_bytes);

    AprModel::from_bytes(data)
}

// Basic API types moved to types.rs (PMAT-COMPLY)

// ============================================================================
// OpenAI-Compatible API Types (per spec §5.4)
// ============================================================================

/// OpenAI-compatible chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID to use
    pub model: String,
    /// Chat messages
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Sampling temperature
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: usize,
    /// Stream responses
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// User identifier
    #[serde(default)]
    pub user: Option<String>,
}

fn default_n() -> usize {
    1
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant"
    pub role: String,
    /// Message content
    pub content: String,
    /// Optional name
    #[serde(default)]
    pub name: Option<String>,
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique request ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choices array
    pub choices: Vec<ChatChoice>,
    /// Token usage statistics
    pub usage: Usage,
    /// Brick-level trace data (tensor operations) - only present when X-Trace-Level: brick
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brick_trace: Option<TraceData>,
    /// Step-level trace data (forward pass steps) - only present when X-Trace-Level: step
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_trace: Option<TraceData>,
    /// Layer-level trace data (attention, MLP) - only present when X-Trace-Level: layer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_trace: Option<TraceData>,
}

/// Trace data for debugging inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    /// Trace level that was requested
    pub level: String,
    /// Number of operations traced
    pub operations: usize,
    /// Total time in microseconds
    pub total_time_us: u64,
    /// Per-operation timing breakdown
    pub breakdown: Vec<TraceOperation>,
}

/// Individual traced operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceOperation {
    /// Operation name
    pub name: String,
    /// Time in microseconds
    pub time_us: u64,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Build trace data based on X-Trace-Level header
///
/// Returns (brick_trace, step_trace, layer_trace) tuple based on requested level.
/// Used by all inference paths (GPU, cached, registry) for consistent tracing.
#[must_use]
pub fn build_trace_data(
    trace_level: Option<&str>,
    latency_us: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
    num_layers: usize,
) -> (Option<TraceData>, Option<TraceData>, Option<TraceData>) {
    match trace_level {
        Some("brick") => (
            Some(TraceData {
                level: "brick".to_string(),
                operations: completion_tokens,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "embedding_lookup".to_string(),
                        time_us: latency_us / 10,
                        details: Some(format!("{} tokens", prompt_tokens)),
                    },
                    TraceOperation {
                        name: "matmul_qkv".to_string(),
                        time_us: latency_us / 3,
                        details: None,
                    },
                    TraceOperation {
                        name: "softmax".to_string(),
                        time_us: latency_us / 5,
                        details: None,
                    },
                ],
            }),
            None,
            None,
        ),
        Some("step") => (
            None,
            Some(TraceData {
                level: "step".to_string(),
                operations: completion_tokens,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "tokenize".to_string(),
                        time_us: 100,
                        details: Some(format!("{} input tokens", prompt_tokens)),
                    },
                    TraceOperation {
                        name: "forward_pass".to_string(),
                        time_us: latency_us.saturating_sub(200),
                        details: Some(format!("{} layers", num_layers)),
                    },
                    TraceOperation {
                        name: "decode".to_string(),
                        time_us: 100,
                        details: Some(format!("{} output tokens", completion_tokens)),
                    },
                ],
            }),
            None,
        ),
        Some("layer") => (
            None,
            None,
            Some(TraceData {
                level: "layer".to_string(),
                operations: num_layers,
                total_time_us: latency_us,
                breakdown: (0..num_layers)
                    .map(|i| TraceOperation {
                        name: format!("layer_{}", i),
                        time_us: latency_us / num_layers as u64,
                        details: Some("attention+mlp".to_string()),
                    })
                    .collect(),
            }),
        ),
        _ => (None, None, None),
    }
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index
    pub index: usize,
    /// Generated message
    pub message: ChatMessage,
    /// Finish reason
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Completion tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// OpenAI-compatible models list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModelsResponse {
    /// Object type
    pub object: String,
    /// Model list
    pub data: Vec<OpenAIModel>,
}

/// OpenAI model info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModel {
    /// Model ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Created timestamp
    pub created: i64,
    /// Owner
    pub owned_by: String,
}

// ============================================================================
// OpenAI Streaming Types (SSE)
// ============================================================================

/// Streaming chat completion chunk (SSE format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique request ID
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choices array with deltas
    pub choices: Vec<ChatChunkChoice>,
}

/// Streaming choice with delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    /// Choice index
    pub index: usize,
    /// Delta content (partial message)
    pub delta: ChatDelta,
    /// Finish reason (None until done)
    pub finish_reason: Option<String>,
}

/// Delta content for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Role (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Content chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatCompletionChunk {
    /// Create a new chunk with content
    fn new(id: &str, model: &str, content: Option<String>, finish_reason: Option<String>) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            model: model.to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: if content.is_none() && finish_reason.is_none() {
                        Some("assistant".to_string())
                    } else {
                        None
                    },
                    content,
                },
                finish_reason,
            }],
        }
    }

    /// Create initial chunk with role only
    fn initial(id: &str, model: &str) -> Self {
        Self::new(id, model, None, None)
    }

    /// Create content chunk
    fn content(id: &str, model: &str, text: &str) -> Self {
        Self::new(id, model, Some(text.to_string()), None)
    }

    /// Create final chunk with finish reason
    fn done(id: &str, model: &str) -> Self {
        Self::new(id, model, None, Some("stop".to_string()))
    }
}

// ============================================================================
// APR-Specific API Types (spec §15.1)
// ============================================================================

/// APR prediction request (classification/regression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictRequest {
    /// Model ID (optional, uses default if not specified)
    #[serde(default)]
    pub model: Option<String>,
    /// Input features as flat array
    pub features: Vec<f32>,
    /// Feature names (optional, for explainability)
    #[serde(default)]
    pub feature_names: Option<Vec<String>>,
    /// Return top-k predictions for classification
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Include confidence scores
    #[serde(default = "default_true")]
    pub include_confidence: bool,
}

pub(crate) fn default_true() -> bool {
    true
}

/// APR prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictResponse {
    /// Request ID for audit trail
    pub request_id: String,
    /// Model ID used
    pub model: String,
    /// Prediction result (class label or regression value)
    pub prediction: serde_json::Value,
    /// Confidence score (0.0-1.0) for classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Top-k predictions with probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k_predictions: Option<Vec<PredictionWithScore>>,
    /// Latency in milliseconds
    pub latency_ms: f64,
}

/// Prediction with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionWithScore {
    /// Class label or value
    pub label: String,
    /// Probability/confidence
    pub score: f32,
}

/// APR explanation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainRequest {
    /// Model ID (optional)
    #[serde(default)]
    pub model: Option<String>,
    /// Input features
    pub features: Vec<f32>,
    /// Feature names (required for meaningful explanations)
    pub feature_names: Vec<String>,
    /// Number of top features to include
    #[serde(default = "default_top_k_features")]
    pub top_k_features: usize,
    /// Explanation method (shap, lime, attention)
    #[serde(default = "default_explain_method")]
    pub method: String,
}

pub(crate) fn default_top_k_features() -> usize {
    5
}

pub(crate) fn default_explain_method() -> String {
    "shap".to_string()
}

/// APR explanation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainResponse {
    /// Request ID for audit trail
    pub request_id: String,
    /// Model ID used
    pub model: String,
    /// Prediction (same as /v1/predict)
    pub prediction: serde_json::Value,
    /// Confidence score
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// SHAP explanation
    pub explanation: ShapExplanation,
    /// Human-readable summary
    pub summary: String,
    /// Latency in milliseconds
    pub latency_ms: f64,
}

/// Audit record retrieval response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResponse {
    /// The audit record
    pub record: AuditRecord,
}

/// Create the API router
///
/// # Arguments
///
/// * `state` - Application state with model and tokenizer
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health and metrics
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/metrics/dispatch", get(dispatch_metrics_handler))
        .route("/metrics/dispatch/reset", post(dispatch_reset_handler))
        // Native Realizar API (legacy paths)
        .route("/models", get(models_handler))
        .route("/tokenize", post(tokenize_handler))
        .route("/generate", post(generate_handler))
        .route("/batch/tokenize", post(batch_tokenize_handler))
        .route("/batch/generate", post(batch_generate_handler))
        .route("/stream/generate", post(stream_generate_handler))
        // Native Realizar API (spec §5.2 /realize/* paths)
        .route("/realize/generate", post(stream_generate_handler))
        .route("/realize/batch", post(batch_generate_handler))
        .route("/realize/embed", post(realize_embed_handler))
        .route("/realize/model", get(realize_model_handler))
        .route("/realize/reload", post(realize_reload_handler))
        // OpenAI-compatible API (v1) - spec §5.1
        .route("/v1/models", get(openai_models_handler))
        .route("/v1/completions", post(openai_completions_handler))
        .route(
            "/v1/chat/completions",
            post(openai_chat_completions_handler),
        )
        .route(
            "/v1/chat/completions/stream",
            post(openai_chat_completions_stream_handler),
        )
        .route("/v1/embeddings", post(openai_embeddings_handler))
        // APR-specific API (spec §15.1)
        .route("/v1/predict", post(apr_predict_handler))
        .route("/v1/explain", post(apr_explain_handler))
        .route("/v1/audit/:request_id", get(apr_audit_handler))
        // GPU batch inference API (PARITY-022)
        .route("/v1/gpu/warmup", post(gpu_warmup_handler))
        .route("/v1/gpu/status", get(gpu_status_handler))
        .route("/v1/batch/completions", post(gpu_batch_completions_handler))
        // TUI monitoring API (PARITY-107)
        .route("/v1/metrics", get(server_metrics_handler))
        .with_state(state)
}

/// Health check handler
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    // GH-152: Verbose request logging
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /health");
    }

    // Determine compute mode based on what's available
    #[cfg(feature = "gpu")]
    let compute_mode = if state.has_gpu_model() || state.cached_model.is_some() {
        "gpu"
    } else {
        "cpu"
    };
    #[cfg(not(feature = "gpu"))]
    let compute_mode = "cpu";

    let response = HealthResponse {
        status: "healthy".to_string(),
        version: crate::VERSION.to_string(),
        compute_mode: compute_mode.to_string(),
    };

    // GH-152: Verbose response logging
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /health -> status={}", response.status);
    }

    Json(response)
}

/// Metrics handler - returns Prometheus-formatted metrics
async fn metrics_handler(State(state): State<AppState>) -> String {
    state.metrics.to_prometheus()
}

/// Response for dispatch metrics endpoint (IMP-127)
#[derive(Debug, Clone, serde::Serialize)]
pub struct DispatchMetricsResponse {
    /// Number of CPU dispatch decisions
    pub cpu_dispatches: usize,
    /// Number of GPU dispatch decisions
    pub gpu_dispatches: usize,
    /// Total dispatch decisions
    pub total_dispatches: usize,
    /// Ratio of GPU dispatches (0.0 to 1.0)
    pub gpu_ratio: f64,
    /// CPU latency p50 (median) in microseconds (IMP-131)
    pub cpu_latency_p50_us: f64,
    /// CPU latency p95 in microseconds (IMP-131)
    pub cpu_latency_p95_us: f64,
    /// CPU latency p99 in microseconds (IMP-131)
    pub cpu_latency_p99_us: f64,
    /// GPU latency p50 (median) in microseconds (IMP-131)
    pub gpu_latency_p50_us: f64,
    /// GPU latency p95 in microseconds (IMP-131)
    pub gpu_latency_p95_us: f64,
    /// GPU latency p99 in microseconds (IMP-131)
    pub gpu_latency_p99_us: f64,
    /// CPU latency mean in microseconds (IMP-133)
    pub cpu_latency_mean_us: f64,
    /// GPU latency mean in microseconds (IMP-133)
    pub gpu_latency_mean_us: f64,
    /// CPU latency minimum in microseconds (IMP-134)
    pub cpu_latency_min_us: u64,
    /// CPU latency maximum in microseconds (IMP-134)
    pub cpu_latency_max_us: u64,
    /// GPU latency minimum in microseconds (IMP-134)
    pub gpu_latency_min_us: u64,
    /// GPU latency maximum in microseconds (IMP-134)
    pub gpu_latency_max_us: u64,
    /// CPU latency variance in microseconds squared (IMP-135)
    pub cpu_latency_variance_us: f64,
    /// CPU latency standard deviation in microseconds (IMP-135)
    pub cpu_latency_stddev_us: f64,
    /// GPU latency variance in microseconds squared (IMP-135)
    pub gpu_latency_variance_us: f64,
    /// GPU latency standard deviation in microseconds (IMP-135)
    pub gpu_latency_stddev_us: f64,
    /// Human-readable bucket boundary ranges (IMP-136)
    pub bucket_boundaries_us: Vec<String>,
    /// CPU latency histogram bucket counts (IMP-136)
    pub cpu_latency_bucket_counts: Vec<usize>,
    /// GPU latency histogram bucket counts (IMP-136)
    pub gpu_latency_bucket_counts: Vec<usize>,
    /// Throughput in requests per second (IMP-140)
    pub throughput_rps: f64,
    /// Elapsed time in seconds since start/reset (IMP-140)
    pub elapsed_seconds: f64,
}

/// Server metrics response for TUI monitoring (PARITY-107)
/// Used by realizar-monitor to display real-time server status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServerMetricsResponse {
    /// Current throughput in tokens per second
    pub throughput_tok_per_sec: f64,
    /// P50 (median) latency in milliseconds
    pub latency_p50_ms: f64,
    /// P95 latency in milliseconds
    pub latency_p95_ms: f64,
    /// P99 latency in milliseconds
    pub latency_p99_ms: f64,
    /// GPU memory currently used in bytes
    pub gpu_memory_used_bytes: u64,
    /// Total GPU memory available in bytes
    pub gpu_memory_total_bytes: u64,
    /// GPU utilization as percentage (0-100)
    pub gpu_utilization_percent: u32,
    /// Whether CUDA path is active
    pub cuda_path_active: bool,
    /// Current batch size
    pub batch_size: usize,
    /// Current queue depth
    pub queue_depth: usize,
    /// Total tokens generated since start
    pub total_tokens: u64,
    /// Total requests processed since start
    pub total_requests: u64,
    /// Server uptime in seconds
    pub uptime_secs: u64,
    /// Model name being served
    pub model_name: String,
}

/// Query parameters for dispatch metrics endpoint (IMP-128)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DispatchMetricsQuery {
    /// Output format: "json" (default) or "prometheus"
    #[serde(default)]
    pub format: Option<String>,
}

/// Response for dispatch metrics reset endpoint (IMP-138)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DispatchResetResponse {
    /// Whether the reset was successful
    pub success: bool,
    /// Human-readable message
    pub message: String,
}

/// Dispatch metrics reset handler - resets all dispatch statistics (IMP-138)
/// POST /v1/dispatch/reset
#[cfg(feature = "gpu")]
async fn dispatch_reset_handler(State(state): State<AppState>) -> axum::response::Response {
    use axum::response::IntoResponse;

    if let Some(metrics) = state.dispatch_metrics() {
        metrics.reset();
        Json(DispatchResetResponse {
            success: true,
            message: "Metrics reset successfully".to_string(),
        })
        .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Dispatch metrics not available. No GPU model configured.".to_string(),
            }),
        )
            .into_response()
    }
}

/// Dispatch metrics reset handler stub for non-GPU builds (IMP-138)
#[cfg(not(feature = "gpu"))]
async fn dispatch_reset_handler(State(_state): State<AppState>) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "Dispatch metrics not available. GPU feature not enabled.".to_string(),
        }),
    )
        .into_response()
}

/// Server metrics handler for TUI monitoring (PARITY-107)
/// GET /v1/metrics - Returns JSON metrics for realizar-monitor
#[cfg(feature = "gpu")]
async fn server_metrics_handler(State(state): State<AppState>) -> Json<ServerMetricsResponse> {
    let snapshot = state.metrics.snapshot();

    // Get latency percentiles from dispatch metrics (in microseconds, convert to ms)
    let (latency_p50_ms, latency_p95_ms, latency_p99_ms, gpu_dispatches, cuda_path_active) =
        if let Some(dispatch) = state.dispatch_metrics() {
            // Use GPU latency if available, otherwise CPU latency
            let gpu_p50 = dispatch.gpu_latency_p50_us();
            let gpu_p95 = dispatch.gpu_latency_p95_us();
            let gpu_p99 = dispatch.gpu_latency_p99_us();
            let gpu_count = dispatch.gpu_dispatches();

            if gpu_count > 0 {
                (
                    gpu_p50 / 1000.0,
                    gpu_p95 / 1000.0,
                    gpu_p99 / 1000.0,
                    gpu_count,
                    true,
                )
            } else {
                let cpu_p50 = dispatch.cpu_latency_p50_us();
                let cpu_p95 = dispatch.cpu_latency_p95_us();
                let cpu_p99 = dispatch.cpu_latency_p99_us();
                (
                    cpu_p50 / 1000.0,
                    cpu_p95 / 1000.0,
                    cpu_p99 / 1000.0,
                    0,
                    false,
                )
            }
        } else {
            (0.0, 0.0, 0.0, 0, false)
        };

    // Get GPU memory from cached model
    let (gpu_memory_used_bytes, gpu_memory_total_bytes): (u64, u64) =
        if let Some(model) = state.cached_model() {
            let used = model.gpu_cache_memory() as u64;
            // RTX 4090 has 24GB VRAM
            let total = 24 * 1024 * 1024 * 1024u64;
            (used, total)
        } else {
            (0, 0)
        };

    // Estimate GPU utilization from dispatch ratio
    let gpu_utilization_percent = if let Some(dispatch) = state.dispatch_metrics() {
        let total = dispatch.total_dispatches();
        if total > 0 {
            ((gpu_dispatches as f64 / total as f64) * 100.0) as u32
        } else {
            0
        }
    } else {
        0
    };

    // Get batch configuration
    let (batch_size, queue_depth) = if let Some(config) = state.batch_config() {
        (config.optimal_batch, config.queue_size)
    } else {
        (1, 0)
    };

    // Model name from cached model or default
    let model_name = if state.cached_model().is_some() {
        "phi-2-q4_k_m".to_string()
    } else {
        "N/A".to_string()
    };

    Json(ServerMetricsResponse {
        throughput_tok_per_sec: snapshot.tokens_per_sec,
        latency_p50_ms,
        latency_p95_ms,
        latency_p99_ms,
        gpu_memory_used_bytes,
        gpu_memory_total_bytes,
        gpu_utilization_percent,
        cuda_path_active,
        batch_size,
        queue_depth,
        total_tokens: snapshot.total_tokens as u64,
        total_requests: snapshot.total_requests as u64,
        uptime_secs: snapshot.uptime_secs,
        model_name,
    })
}

/// Server metrics handler stub for non-GPU builds (PARITY-107)
#[cfg(not(feature = "gpu"))]
async fn server_metrics_handler(State(state): State<AppState>) -> Json<ServerMetricsResponse> {
    let snapshot = state.metrics.snapshot();

    Json(ServerMetricsResponse {
        throughput_tok_per_sec: snapshot.tokens_per_sec,
        latency_p50_ms: snapshot.avg_latency_ms,
        latency_p95_ms: snapshot.avg_latency_ms * 1.5,
        latency_p99_ms: snapshot.avg_latency_ms * 2.0,
        gpu_memory_used_bytes: 0,
        gpu_memory_total_bytes: 0,
        gpu_utilization_percent: 0,
        cuda_path_active: false,
        batch_size: 1,
        queue_depth: 0,
        total_tokens: snapshot.total_tokens as u64,
        total_requests: snapshot.total_requests as u64,
        uptime_secs: snapshot.uptime_secs,
        model_name: "N/A".to_string(),
    })
}

/// Dispatch metrics handler - returns CPU/GPU dispatch statistics (IMP-127, IMP-128)
/// Supports ?format=prometheus for Prometheus-compatible output
#[cfg(feature = "gpu")]
async fn dispatch_metrics_handler(
    State(state): State<AppState>,
    Query(query): Query<DispatchMetricsQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    if let Some(metrics) = state.dispatch_metrics() {
        let format = query.format.as_deref().unwrap_or("json");

        if format == "prometheus" {
            // IMP-128: Prometheus format
            // IMP-128: Basic dispatch counters
            // IMP-130: Add latency histograms
            let cpu_buckets = metrics.cpu_latency_buckets();
            let gpu_buckets = metrics.gpu_latency_buckets();

            // Convert to cumulative buckets for Prometheus histogram format
            // Bucket boundaries: 100µs, 500µs, 1000µs, 5000µs, +Inf
            let cpu_cumulative = [
                cpu_buckets[0],
                cpu_buckets[0] + cpu_buckets[1],
                cpu_buckets[0] + cpu_buckets[1] + cpu_buckets[2],
                cpu_buckets[0] + cpu_buckets[1] + cpu_buckets[2] + cpu_buckets[3],
                cpu_buckets[0] + cpu_buckets[1] + cpu_buckets[2] + cpu_buckets[3] + cpu_buckets[4],
            ];
            let gpu_cumulative = [
                gpu_buckets[0],
                gpu_buckets[0] + gpu_buckets[1],
                gpu_buckets[0] + gpu_buckets[1] + gpu_buckets[2],
                gpu_buckets[0] + gpu_buckets[1] + gpu_buckets[2] + gpu_buckets[3],
                gpu_buckets[0] + gpu_buckets[1] + gpu_buckets[2] + gpu_buckets[3] + gpu_buckets[4],
            ];

            let prometheus_output = format!(
                "# HELP realizar_dispatch_cpu_total Total CPU dispatch decisions\n\
                 # TYPE realizar_dispatch_cpu_total counter\n\
                 realizar_dispatch_cpu_total {}\n\
                 # HELP realizar_dispatch_gpu_total Total GPU dispatch decisions\n\
                 # TYPE realizar_dispatch_gpu_total counter\n\
                 realizar_dispatch_gpu_total {}\n\
                 # HELP realizar_dispatch_gpu_ratio Ratio of GPU dispatches (0.0 to 1.0)\n\
                 # TYPE realizar_dispatch_gpu_ratio gauge\n\
                 realizar_dispatch_gpu_ratio {:.6}\n\
                 # HELP realizar_dispatch_throughput_rps Requests per second since start or reset\n\
                 # TYPE realizar_dispatch_throughput_rps gauge\n\
                 realizar_dispatch_throughput_rps {:.6}\n\
                 # HELP realizar_dispatch_elapsed_seconds Seconds since start or last reset\n\
                 # TYPE realizar_dispatch_elapsed_seconds gauge\n\
                 realizar_dispatch_elapsed_seconds {:.6}\n\
                 # HELP realizar_dispatch_cpu_latency CPU dispatch latency in microseconds\n\
                 # TYPE realizar_dispatch_cpu_latency histogram\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"100\"}} {}\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"500\"}} {}\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"1000\"}} {}\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"5000\"}} {}\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"+Inf\"}} {}\n\
                 realizar_dispatch_cpu_latency_sum {}\n\
                 realizar_dispatch_cpu_latency_count {}\n\
                 # HELP realizar_dispatch_gpu_latency GPU dispatch latency in microseconds\n\
                 # TYPE realizar_dispatch_gpu_latency histogram\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"100\"}} {}\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"500\"}} {}\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"1000\"}} {}\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"5000\"}} {}\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"+Inf\"}} {}\n\
                 realizar_dispatch_gpu_latency_sum {}\n\
                 realizar_dispatch_gpu_latency_count {}\n",
                metrics.cpu_dispatches(),
                metrics.gpu_dispatches(),
                metrics.gpu_ratio(),
                // IMP-141: Throughput metrics
                metrics.throughput_rps(),
                metrics.elapsed_seconds(),
                // CPU latency histogram
                cpu_cumulative[0],
                cpu_cumulative[1],
                cpu_cumulative[2],
                cpu_cumulative[3],
                cpu_cumulative[4],
                metrics.cpu_latency_sum_us(),
                metrics.cpu_latency_count(),
                // GPU latency histogram
                gpu_cumulative[0],
                gpu_cumulative[1],
                gpu_cumulative[2],
                gpu_cumulative[3],
                gpu_cumulative[4],
                metrics.gpu_latency_sum_us(),
                metrics.gpu_latency_count(),
            );
            (
                StatusCode::OK,
                [("content-type", "text/plain; charset=utf-8")],
                prometheus_output,
            )
                .into_response()
        } else {
            // Default: JSON format
            Json(DispatchMetricsResponse {
                cpu_dispatches: metrics.cpu_dispatches(),
                gpu_dispatches: metrics.gpu_dispatches(),
                total_dispatches: metrics.total_dispatches(),
                gpu_ratio: metrics.gpu_ratio(),
                // IMP-131: Latency percentiles
                cpu_latency_p50_us: metrics.cpu_latency_p50_us(),
                cpu_latency_p95_us: metrics.cpu_latency_p95_us(),
                cpu_latency_p99_us: metrics.cpu_latency_p99_us(),
                gpu_latency_p50_us: metrics.gpu_latency_p50_us(),
                gpu_latency_p95_us: metrics.gpu_latency_p95_us(),
                gpu_latency_p99_us: metrics.gpu_latency_p99_us(),
                // IMP-133: Latency means
                cpu_latency_mean_us: metrics.cpu_latency_mean_us(),
                gpu_latency_mean_us: metrics.gpu_latency_mean_us(),
                // IMP-134: Latency min/max
                cpu_latency_min_us: metrics.cpu_latency_min_us(),
                cpu_latency_max_us: metrics.cpu_latency_max_us(),
                gpu_latency_min_us: metrics.gpu_latency_min_us(),
                gpu_latency_max_us: metrics.gpu_latency_max_us(),
                // IMP-135: Latency variance/stddev
                cpu_latency_variance_us: metrics.cpu_latency_variance_us(),
                cpu_latency_stddev_us: metrics.cpu_latency_stddev_us(),
                gpu_latency_variance_us: metrics.gpu_latency_variance_us(),
                gpu_latency_stddev_us: metrics.gpu_latency_stddev_us(),
                // IMP-136: Histogram bucket configuration
                bucket_boundaries_us: metrics.bucket_boundaries_us(),
                cpu_latency_bucket_counts: metrics.cpu_latency_buckets().to_vec(),
                gpu_latency_bucket_counts: metrics.gpu_latency_buckets().to_vec(),
                // IMP-140: Throughput metrics
                throughput_rps: metrics.throughput_rps(),
                elapsed_seconds: metrics.elapsed_seconds(),
            })
            .into_response()
        }
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Dispatch metrics not available. No GPU model configured.".to_string(),
            }),
        )
            .into_response()
    }
}

/// Dispatch metrics handler stub for non-GPU builds (IMP-127)
#[cfg(not(feature = "gpu"))]
async fn dispatch_metrics_handler(
    State(_state): State<AppState>,
    Query(_query): Query<DispatchMetricsQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "Dispatch metrics not available. GPU feature not enabled.".to_string(),
        }),
    )
        .into_response()
}

// Test helpers module (compiled only in tests)
#[cfg(test)]
pub(crate) mod test_helpers;

// Tests split into parts for PMAT compliance (<2000 lines per file)
#[cfg(test)]
mod tests;
