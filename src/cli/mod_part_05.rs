
// ============================================================================
// Server Commands (extracted from main.rs for testability)
// WAPR-PERF-004: Gated behind "server" feature since depends on crate::api
// ============================================================================
#[cfg(feature = "server")]
mod server_commands {
    use super::Result;

    /// Result of preparing server state (returned by `prepare_serve_state`)
    pub struct PreparedServer {
        /// The prepared AppState for the server
        pub state: crate::api::AppState,
        /// Whether batch mode is enabled
        pub batch_mode_enabled: bool,
        /// Model type that was loaded
        pub model_type: ModelType,
    }

    impl std::fmt::Debug for PreparedServer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PreparedServer")
                .field("batch_mode_enabled", &self.batch_mode_enabled)
                .field("model_type", &self.model_type)
                .finish_non_exhaustive()
        }
    }

    /// Type of model being served
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ModelType {
        /// GGUF quantized model
        Gguf,
        /// SafeTensors model
        SafeTensors,
        /// APR format model
        Apr,
    }

    impl std::fmt::Display for ModelType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ModelType::Gguf => write!(f, "GGUF"),
                ModelType::SafeTensors => write!(f, "SafeTensors"),
                ModelType::Apr => write!(f, "APR"),
            }
        }
    }

    /// Prepare server state by loading a model (GGUF/SafeTensors/APR)
    ///
    /// This function is extracted from `serve_model` for testability.
    /// It handles model loading and AppState creation without starting the server.
    ///
    /// # Arguments
    /// * `model_path` - Path to model file (.gguf, .safetensors, or .apr)
    /// * `batch_mode` - Enable batch processing (requires 'gpu' feature)
    /// * `force_gpu` - Force CUDA backend (requires 'cuda' feature)
    ///
    /// # Returns
    /// A `PreparedServer` containing the AppState and configuration
    /// Load and prepare a GGUF model for serving (CUDA/batch/CPU paths)
    fn prepare_gguf_serve_state(
        model_path: &str,
        batch_mode: bool,
        force_gpu: bool,
    ) -> Result<PreparedServer> {
        use crate::gguf::MappedGGUFModel;

        println!("Parsing GGUF file...");
        let mapped = MappedGGUFModel::from_path(model_path).map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "load_gguf".to_string(),
                reason: format!("Failed to load GGUF: {e}"),
            }
        })?;

        println!("Successfully loaded GGUF model");
        println!("  Tensors: {}", mapped.model.tensors.len());
        println!("  Metadata: {} entries", mapped.model.metadata.len());
        println!();

        // IMP-100: Use OwnedQuantizedModel with fused Q4_K ops (1.37x faster for single-token)
        println!("Creating quantized model (fused Q4_K ops)...");
        let quantized_model =
            crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| {
                crate::error::RealizarError::UnsupportedOperation {
                    operation: "create_quantized".to_string(),
                    reason: format!("Failed to create quantized model: {e}"),
                }
            })?;

        println!("Quantized model created successfully!");
        println!("  Vocab size: {}", quantized_model.config.vocab_size);
        println!("  Hidden dim: {}", quantized_model.config.hidden_dim);
        println!("  Layers: {}", quantized_model.layers.len());

        // Extract vocabulary from GGUF for proper token decoding
        let vocab = mapped.model.vocabulary().unwrap_or_else(|| {
            eprintln!("  Warning: No vocabulary in GGUF, using placeholder tokens");
            (0..quantized_model.config.vocab_size)
                .map(|i| format!("token{i}"))
                .collect()
        });
        println!("  Vocab loaded: {} tokens", vocab.len());
        println!();

        // PARITY-113: Enable CUDA backend via --gpu flag or REALIZAR_BACKEND environment variable
        #[cfg(feature = "cuda")]
        let use_cuda = force_gpu
            || std::env::var("REALIZAR_BACKEND")
                .map(|v| v.eq_ignore_ascii_case("cuda"))
                .unwrap_or(false);

        #[cfg(not(feature = "cuda"))]
        let use_cuda = false;

        #[cfg(not(feature = "cuda"))]
        if force_gpu {
            eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
            eprintln!("Build with: cargo build --features cuda");
            eprintln!();
        }

        let state = if use_cuda && !batch_mode {
            prepare_gguf_cuda_state(quantized_model, vocab, force_gpu)?
        } else if batch_mode {
            prepare_gguf_batch_state(quantized_model, vocab)?
        } else {
            // CPU mode: Use quantized model for serving (fused CPU ops are faster for m=1)
            crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)?
        };

        Ok(PreparedServer {
            state,
            batch_mode_enabled: batch_mode,
            model_type: ModelType::Gguf,
        })
    }

    /// Create CUDA-backed AppState for GGUF serving (PAR-112-FIX)
    fn prepare_gguf_cuda_state(
        quantized_model: crate::gguf::OwnedQuantizedModel,
        vocab: Vec<String>,
        force_gpu: bool,
    ) -> Result<crate::api::AppState> {
        #[cfg(feature = "cuda")]
        {
            use crate::gguf::OwnedQuantizedModelCuda;

            let source = if force_gpu {
                "--gpu flag"
            } else {
                "REALIZAR_BACKEND=cuda"
            };
            println!("Creating CUDA model ({source})...");

            let max_seq_len = 4096; // Support long sequences
            let cuda_model =
                OwnedQuantizedModelCuda::with_max_seq_len(quantized_model, 0, max_seq_len)
                    .map_err(|e| e.error)?;

            println!("  CUDA model created on GPU: {}", cuda_model.device_name());
            println!("  Max sequence length: {}", max_seq_len);
            println!("  TRUE STREAMING: enabled (PAR-112)");
            println!();

            crate::api::AppState::with_cuda_model_and_vocab(cuda_model, vocab)
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = force_gpu;
            crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        }
    }

    /// Create batch-processing AppState for GGUF serving (PARITY-093/094)
    fn prepare_gguf_batch_state(
        quantized_model: crate::gguf::OwnedQuantizedModel,
        vocab: Vec<String>,
    ) -> Result<crate::api::AppState> {
        #[cfg(feature = "gpu")]
        {
            use crate::gguf::OwnedQuantizedModelCachedSync;

            println!("Initializing batch inference mode (PARITY-093/094)...");

            let cached_model = OwnedQuantizedModelCachedSync::new(quantized_model);

            println!("  Warming up GPU cache (dequantizing FFN weights)...");
            match cached_model.warmup_gpu_cache() {
                Ok((memory_bytes, num_layers)) => {
                    println!(
                        "  GPU cache ready: {:.2} GB ({} layers)",
                        memory_bytes as f64 / 1e9,
                        num_layers
                    );
                },
                Err(e) => {
                    eprintln!(
                        "  Warning: GPU cache warmup failed: {}. Falling back to CPU batch.",
                        e
                    );
                },
            }

            let state = crate::api::AppState::with_cached_model_and_vocab(cached_model, vocab)?;

            let cached_model_arc = state
                .cached_model()
                .expect("cached_model should exist")
                .clone();

            let batch_config = crate::api::BatchConfig::default();
            println!("  Batch window: {}ms", batch_config.window_ms);
            println!("  Min batch size: {}", batch_config.min_batch);
            println!("  Optimal batch: {}", batch_config.optimal_batch);
            println!("  Max batch size: {}", batch_config.max_batch);
            println!(
                "  GPU threshold: {} (GPU GEMM for batch >= this)",
                batch_config.gpu_threshold
            );

            let batch_tx =
                crate::api::spawn_batch_processor(cached_model_arc, batch_config.clone());

            println!("  Batch processor: RUNNING");
            println!();

            Ok(state.with_batch_config(batch_tx, batch_config))
        }

        #[cfg(not(feature = "gpu"))]
        {
            eprintln!(
                "Warning: --batch requires 'gpu' feature. Falling back to single-request mode."
            );
            crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        }
    }

    /// Load and prepare a SafeTensors model for serving
    fn prepare_safetensors_serve_state(model_path: &str) -> Result<PreparedServer> {
        use crate::safetensors_infer::SafetensorsToAprConverter;
        use std::path::Path;

        println!("Loading SafeTensors model for serving...");

        let model_path_obj = Path::new(model_path);
        let transformer = SafetensorsToAprConverter::convert(model_path_obj).map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "convert_safetensors".to_string(),
                reason: format!("Failed to convert SafeTensors: {e}"),
            }
        })?;

        println!("  Architecture: {}", transformer.config.architecture);
        println!("  Layers: {}", transformer.config.num_layers);
        println!("  Hidden: {}", transformer.config.hidden_dim);

        #[allow(clippy::map_unwrap_or)]
        let vocab = crate::apr::AprV2Model::load_tokenizer_from_sibling(model_path_obj)
            .map(|(v, _, _)| v)
            .unwrap_or_else(|| {
                println!("  Warning: No tokenizer.json found, using simple vocabulary");
                (0..transformer.config.vocab_size)
                    .map(|i| format!("token{i}"))
                    .collect()
            });

        println!("  Vocab size: {}", vocab.len());
        println!("  Mode: CPU (F32 inference)");

        let state =
            crate::api::AppState::with_apr_transformer_and_vocab(transformer.into_inner(), vocab)?;

        Ok(PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::SafeTensors,
        })
    }

    /// Load and prepare an APR model for serving
    fn prepare_apr_serve_state(model_path: &str) -> Result<PreparedServer> {
        use crate::apr_transformer::AprTransformer;
        use std::path::Path;

        println!("Loading APR model for serving...");

        let file_data = std::fs::read(model_path).map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "read_model_file".to_string(),
                reason: format!("Failed to read {model_path}: {e}"),
            }
        })?;

        let transformer = AprTransformer::from_apr_bytes(&file_data).map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "load_apr".to_string(),
                reason: format!("Failed to load APR: {e}"),
            }
        })?;

        println!("  Architecture: {}", transformer.config.architecture);
        println!("  Layers: {}", transformer.config.num_layers);
        println!("  Hidden: {}", transformer.config.hidden_dim);

        let model_path_obj = Path::new(model_path);
        let vocab = crate::apr::AprV2Model::load_tokenizer_from_sibling(model_path_obj)
            .map(|(v, _, _)| v)
            .or_else(|| {
                crate::apr::AprV2Model::load(model_path_obj)
                    .ok()
                    .and_then(|m| m.load_embedded_tokenizer())
                    .map(|t| t.id_to_token.clone())
            })
            .unwrap_or_else(|| {
                println!("  Warning: No vocabulary found, using simple vocabulary");
                (0..transformer.config.vocab_size)
                    .map(|i| format!("token{i}"))
                    .collect()
            });

        println!("  Vocab size: {}", vocab.len());
        println!("  Mode: CPU (F32 inference)");

        let state = crate::api::AppState::with_apr_transformer_and_vocab(transformer, vocab)?;

        Ok(PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::Apr,
        })
    }

    /// Prepare server state by loading a model (GGUF/SafeTensors/APR)
    ///
    /// Dispatches to format-specific loaders based on file extension.
    pub fn prepare_serve_state(
        model_path: &str,
        batch_mode: bool,
        force_gpu: bool,
    ) -> Result<PreparedServer> {
        println!("Loading model from: {model_path}");
        if batch_mode {
            println!("Mode: BATCH (PARITY-093 M4 parity)");
        } else {
            println!("Mode: SINGLE-REQUEST");
        }
        if force_gpu {
            println!("GPU: FORCED (--gpu flag)");
        }
        println!();

        if model_path.ends_with(".gguf") {
            prepare_gguf_serve_state(model_path, batch_mode, force_gpu)
        } else if model_path.ends_with(".safetensors") {
            prepare_safetensors_serve_state(model_path)
        } else if model_path.ends_with(".apr") {
            prepare_apr_serve_state(model_path)
        } else {
            Err(crate::error::RealizarError::UnsupportedOperation {
                operation: "detect_model_type".to_string(),
                reason: "Unsupported file extension. Expected .gguf, .safetensors, or .apr"
                    .to_string(),
            })
        }
    }

    /// Serve a GGUF/SafeTensors/APR model via HTTP API
    ///
    /// This function was extracted from main.rs (PAR-112-FIX) to enable:
    /// 1. Unit testing of server initialization logic
    /// 2. Coverage measurement (main.rs was at 3.66%)
    /// 3. Reuse from other entry points
    ///
    /// # Arguments
    /// * `host` - Host to bind to (e.g., "0.0.0.0")
    /// * `port` - Port to listen on
    /// * `model_path` - Path to model file (.gguf, .safetensors, or .apr)
    /// * `batch_mode` - Enable batch processing (requires 'gpu' feature)
    /// * `force_gpu` - Force CUDA backend (requires 'cuda' feature)
    pub async fn serve_model(
        host: &str,
        port: u16,
        model_path: &str,
        batch_mode: bool,
        force_gpu: bool,
    ) -> Result<()> {
        // Prepare server state (testable)
        let prepared = prepare_serve_state(model_path, batch_mode, force_gpu)?;

        // Create router
        let app = crate::api::create_router(prepared.state);

        // Parse and validate address
        let addr: std::net::SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Invalid address: {e}"),
            }
        })?;

        // Print server info
        println!("Server listening on http://{addr}");
        println!();
        println!("Endpoints:");
        println!("  GET  /health         - Health check");
        println!("  POST /v1/completions - OpenAI-compatible completions");
        if prepared.batch_mode_enabled {
            println!("  POST /v1/batch/completions - GPU batch completions (PARITY-022)");
            println!("  POST /v1/gpu/warmup  - Warmup GPU cache");
            println!("  GET  /v1/gpu/status  - GPU status");
        }
        println!("  POST /generate       - Generate text (Q4_K fused)");
        println!();

        if prepared.batch_mode_enabled {
            println!("M4 Parity Target: 192 tok/s at concurrency >= 4");
            println!("Benchmark with: wrk -t4 -c4 -d30s http://{addr}/v1/completions");
            println!();
        }

        // Bind and serve
        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "bind".to_string(),
                reason: format!("Failed to bind: {e}"),
            }
        })?;

        axum::serve(listener, app).await.map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "serve".to_string(),
                reason: format!("Server error: {e}"),
            }
        })?;

        Ok(())
    }

    /// Start a demo inference server (no model required)
    ///
    /// This is useful for testing the API without loading a real model.
    pub async fn serve_demo(host: &str, port: u16) -> Result<()> {
        use std::net::SocketAddr;

        println!("Starting Realizar inference server (demo mode)...");

        let state = crate::api::AppState::demo()?;
        let app = crate::api::create_router(state);

        let addr: SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Invalid address: {e}"),
            }
        })?;

        println!("Server listening on http://{addr}");
        println!();
        println!("Endpoints:");
        println!("  GET  /health   - Health check");
        println!("  POST /tokenize - Tokenize text");
        println!("  POST /generate - Generate text");
        println!();
        println!("Example:");
        println!("  curl http://{addr}/health");
        println!();

        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Failed to bind: {e}"),
            }
        })?;

        axum::serve(listener, app).await.map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Server error: {e}"),
            }
        })?;

        Ok(())
    }
} // mod server_commands

#[cfg(feature = "server")]
pub use server_commands::*;
