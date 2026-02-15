
impl AppState {

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

include!("mod_part_02_part_02.rs");
