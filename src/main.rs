//! Realizar CLI - Pure Rust ML inference server
//!
//! An ollama-like experience for the PAIML ML stack.
//!
//! # Commands
//!
//! - `run` - Run a model for inference
//! - `chat` - Interactive chat mode
//! - `list` - List available models
//! - `pull` - Pull a model from registry
//! - `push` - Push a model to registry
//! - `serve` - Start inference server
//! - `bench` - Run benchmarks
//! - `viz` - Visualize benchmark results
//! - `info` - Show version info

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
#[cfg(feature = "registry")]
use pacha::resolver::{ModelResolver, ModelSource};
#[cfg(feature = "registry")]
use pacha::uri::ModelUri;
use realizar::{
    api::{create_router, AppState},
    cli,
    error::Result,
};

/// Realizar - Pure Rust ML inference engine
///
/// A lightweight, fast alternative to ollama for local model inference.
#[derive(Parser)]
#[command(name = "realizar")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a model for inference (like `ollama run`)
    Run {
        /// Model reference (pacha://name:version, hf://org/model, or path)
        #[arg(value_name = "MODEL")]
        model: String,

        /// Optional prompt (interactive mode if omitted)
        #[arg(value_name = "PROMPT")]
        prompt: Option<String>,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature (0.0 = deterministic)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Output format: text, json, or stream
        #[arg(short, long, default_value = "text")]
        format: String,

        /// System prompt for chat template
        #[arg(short, long)]
        system: Option<String>,

        /// Disable chat template formatting (send raw prompt)
        #[arg(long)]
        raw: bool,

        /// Force GPU acceleration (requires CUDA feature)
        #[arg(long)]
        gpu: bool,
    },
    /// Interactive chat mode (like `ollama chat`)
    Chat {
        /// Model reference
        #[arg(value_name = "MODEL")]
        model: String,

        /// System prompt to set context
        #[arg(short, long)]
        system: Option<String>,

        /// History file for conversation persistence
        #[arg(long)]
        history: Option<String>,
    },
    /// List available models (like `ollama list`)
    List {
        /// Show remote registry models
        #[arg(short, long)]
        remote: Option<String>,

        /// Output format: table, json
        #[arg(short, long, default_value = "table")]
        format: String,
    },
    /// Pull a model from registry (like `ollama pull`)
    Pull {
        /// Model reference to pull
        #[arg(value_name = "MODEL")]
        model: String,

        /// Force re-download even if cached
        #[arg(short, long)]
        force: bool,

        /// Quantization format (q4, q8, f16)
        #[arg(short, long)]
        quantize: Option<String>,
    },
    /// Push a model to registry (like `ollama push`)
    Push {
        /// Model to push
        #[arg(value_name = "MODEL")]
        model: String,

        /// Target registry URL
        #[arg(long)]
        to: Option<String>,
    },
    /// Start the inference server (with OpenAI-compatible API)
    Serve {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Path to model file (APR, GGUF, or SafeTensors)
        #[arg(short, long)]
        model: Option<String>,

        /// Use demo model for testing
        #[arg(long)]
        demo: bool,

        /// Enable OpenAI-compatible API at /v1/*
        #[arg(long, default_value = "true")]
        openai_api: bool,

        /// Enable batch inference for M4 parity (PARITY-093)
        /// Uses continuous batching scheduler for 3-4x throughput at high concurrency
        #[arg(long)]
        batch: bool,

        /// Force GPU acceleration (requires CUDA feature)
        #[arg(long)]
        gpu: bool,
    },
    /// Run performance benchmarks (wraps cargo bench)
    Bench {
        /// Benchmark suite to run
        #[arg(value_name = "SUITE")]
        suite: Option<String>,

        /// List available benchmark suites
        #[arg(short, long)]
        list: bool,

        /// Runtime to benchmark (realizar, llama-cpp, vllm, ollama)
        #[arg(long)]
        runtime: Option<String>,

        /// Model path or name for inference benchmarks
        #[arg(long)]
        model: Option<String>,

        /// Server URL for external runtime benchmarking (e.g., http://localhost:11434)
        #[arg(long)]
        url: Option<String>,

        /// Output file for JSON results (v1.1 schema)
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Run convoy test for continuous batching validation (spec 2.4)
    BenchConvoy {
        /// Runtime to benchmark
        #[arg(long)]
        runtime: Option<String>,

        /// Model path for inference
        #[arg(long)]
        model: Option<String>,

        /// Output file for JSON results
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Run saturation stress test (spec 2.5)
    BenchSaturation {
        /// Runtime to benchmark
        #[arg(long)]
        runtime: Option<String>,

        /// Model path for inference
        #[arg(long)]
        model: Option<String>,

        /// Output file for JSON results
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Compare two benchmark result files
    BenchCompare {
        /// First benchmark result file (JSON)
        #[arg(value_name = "FILE1")]
        file1: String,

        /// Second benchmark result file (JSON)
        #[arg(value_name = "FILE2")]
        file2: String,

        /// Significance threshold percentage (default: 5.0)
        #[arg(short, long, default_value = "5.0")]
        threshold: f64,
    },
    /// Detect performance regressions between baseline and current
    BenchRegression {
        /// Baseline benchmark result file (JSON)
        #[arg(value_name = "BASELINE")]
        baseline: String,

        /// Current benchmark result file (JSON)
        #[arg(value_name = "CURRENT")]
        current: String,

        /// Strict mode: fail on any regression
        #[arg(long)]
        strict: bool,
    },
    /// Visualize benchmark results (terminal output)
    Viz {
        /// Use ANSI color output
        #[arg(short, long)]
        color: bool,

        /// Number of samples to generate
        #[arg(short, long, default_value = "100")]
        samples: usize,
    },
    /// Show version and configuration info
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    let parsed = Cli::parse();

    match parsed.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            format,
            system,
            raw,
            gpu,
        } => {
            run_model(
                &model,
                prompt.as_deref(),
                max_tokens,
                temperature,
                &format,
                system.as_deref(),
                raw,
                gpu,
            )
            .await?;
        },
        Commands::Chat {
            model,
            system,
            history,
        } => {
            run_chat(&model, system.as_deref(), history.as_deref()).await?;
        },
        Commands::List { remote, format } => {
            list_models(remote.as_deref(), &format)?;
        },
        Commands::Pull {
            model,
            force,
            quantize,
        } => {
            pull_model(&model, force, quantize.as_deref()).await?;
        },
        Commands::Push { model, to } => {
            push_model(&model, to.as_deref()).await?;
        },
        Commands::Serve {
            host,
            port,
            model,
            demo,
            openai_api: _,
            batch,
            gpu,
        } => {
            if demo {
                serve_demo(&host, port).await?;
            } else if let Some(model_path) = model {
                serve_model(&host, port, &model_path, batch, gpu).await?;
            } else {
                eprintln!("Error: Either --model or --demo must be specified");
                eprintln!();
                eprintln!("Usage:");
                eprintln!("  realizar serve --demo              # Use demo model");
                eprintln!("  realizar serve --model path.gguf   # Load GGUF model");
                eprintln!(
                    "  realizar serve --model path.gguf --batch  # Enable M4 parity batch mode"
                );
                std::process::exit(1);
            }
        },
        Commands::Bench {
            suite,
            list,
            runtime,
            model,
            url,
            output,
        } => {
            cli::run_benchmarks(suite, list, runtime, model, url, output)?;
        },
        Commands::BenchConvoy {
            runtime,
            model,
            output,
        } => {
            cli::run_convoy_test(runtime, model, output)?;
        },
        Commands::BenchSaturation {
            runtime,
            model,
            output,
        } => {
            cli::run_saturation_test(runtime, model, output)?;
        },
        Commands::BenchCompare {
            file1,
            file2,
            threshold,
        } => {
            cli::run_bench_compare(&file1, &file2, threshold)?;
        },
        Commands::BenchRegression {
            baseline,
            current,
            strict,
        } => {
            if cli::run_bench_regression(&baseline, &current, strict).is_err() {
                std::process::exit(1);
            }
        },
        Commands::Viz { color, samples } => {
            cli::run_visualization(color, samples);
        },
        Commands::Info => {
            cli::print_info();
        },
    }

    Ok(())
}

async fn serve_demo(host: &str, port: u16) -> Result<()> {
    println!("Starting Realizar inference server (demo mode)...");

    let state = AppState::demo()?;
    let app = create_router(state);

    let addr: SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
        realizar::error::RealizarError::InvalidShape {
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
        realizar::error::RealizarError::InvalidShape {
            reason: format!("Failed to bind: {e}"),
        }
    })?;

    axum::serve(listener, app)
        .await
        .map_err(|e| realizar::error::RealizarError::InvalidShape {
            reason: format!("Server error: {e}"),
        })?;

    Ok(())
}

async fn serve_model(
    host: &str,
    port: u16,
    model_path: &str,
    batch_mode: bool,
    force_gpu: bool,
) -> Result<()> {
    use realizar::gguf::MappedGGUFModel;

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
        // Load GGUF model
        println!("Parsing GGUF file...");
        let mapped = MappedGGUFModel::from_path(model_path).map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
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
            realizar::gguf::OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| {
                realizar::error::RealizarError::UnsupportedOperation {
                    operation: "create_quantized".to_string(),
                    reason: format!("Failed to create quantized model: {e}"),
                }
            })?;

        println!("Quantized model created successfully!");
        println!("  Vocab size: {}", quantized_model.config.vocab_size);
        println!("  Hidden dim: {}", quantized_model.config.hidden_dim);
        println!("  Layers: {}", quantized_model.layers.len());
        println!();

        // PARITY-113: Enable CUDA backend via --gpu flag or REALIZAR_BACKEND environment variable
        #[cfg(feature = "cuda")]
        let mut quantized_model = quantized_model;
        #[cfg(feature = "cuda")]
        let use_cuda = force_gpu
            || std::env::var("REALIZAR_BACKEND")
                .map(|v| v.eq_ignore_ascii_case("cuda"))
                .unwrap_or(false);
        #[cfg(feature = "cuda")]
        if use_cuda {
            let source = if force_gpu {
                "--gpu flag"
            } else {
                "REALIZAR_BACKEND=cuda"
            };
            println!("Enabling CUDA backend ({source})...");
            match quantized_model.enable_cuda(0) {
                Ok(()) => {
                    println!("  CUDA enabled on GPU 0");
                    println!("  cuda_enabled: {}", quantized_model.cuda_enabled());
                },
                Err(e) => {
                    eprintln!("  Warning: CUDA enable failed: {}. Falling back to CPU.", e);
                },
            }
            println!();
        }
        #[cfg(not(feature = "cuda"))]
        if force_gpu {
            eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
            eprintln!("Build with: cargo build --features cuda");
            eprintln!();
        }

        // PARITY-093: Use cached model with batch support for M4 parity
        let state = {
            #[cfg(feature = "gpu")]
            {
                if batch_mode {
                    use realizar::gguf::OwnedQuantizedModelCachedSync;

                    println!("Initializing batch inference mode (PARITY-093/094)...");

                    // Create cached model for scheduler reuse (10.6x speedup - IMP-112)
                    let cached_model = OwnedQuantizedModelCachedSync::new(quantized_model);

                    // PARITY-094: Warmup GPU cache for batch_generate_gpu
                    // This dequantizes FFN weights to GPU memory (~6GB for phi-2)
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

                    // Create state first (this wraps model in Arc internally)
                    let state = realizar::api::AppState::with_cached_model(cached_model)?;

                    // Get Arc'd model back for batch processor
                    let cached_model_arc = state
                        .cached_model()
                        .expect("cached_model should exist")
                        .clone();

                    // Configure batch processing (PARITY-095: aligned thresholds)
                    let batch_config = realizar::api::BatchConfig::default();
                    println!("  Batch window: {}ms", batch_config.window_ms);
                    println!("  Min batch size: {}", batch_config.min_batch);
                    println!("  Optimal batch: {}", batch_config.optimal_batch);
                    println!("  Max batch size: {}", batch_config.max_batch);
                    println!(
                        "  GPU threshold: {} (GPU GEMM for batch >= this)",
                        batch_config.gpu_threshold
                    );

                    // Spawn batch processor task
                    let batch_tx = realizar::api::spawn_batch_processor(
                        cached_model_arc,
                        batch_config.clone(),
                    );

                    println!("  Batch processor: RUNNING");
                    println!();

                    // Add batch support to state
                    state.with_batch_config(batch_tx, batch_config)
                } else {
                    // Use quantized model for serving (fused CPU ops are faster for m=1)
                    realizar::api::AppState::with_quantized_model(quantized_model)?
                }
            }

            #[cfg(not(feature = "gpu"))]
            {
                if batch_mode {
                    eprintln!(
                        "Warning: --batch requires 'gpu' feature. Falling back to single-request mode."
                    );
                }
                realizar::api::AppState::with_quantized_model(quantized_model)?
            }
        };

        let app = realizar::api::create_router(state);

        let addr: std::net::SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
            realizar::error::RealizarError::InvalidShape {
                reason: format!("Invalid address: {e}"),
            }
        })?;

        println!("Server listening on http://{addr}");
        println!();
        println!("Endpoints:");
        println!("  GET  /health         - Health check");
        println!("  POST /v1/completions - OpenAI-compatible completions");
        if batch_mode {
            println!("  POST /v1/batch/completions - GPU batch completions (PARITY-022)");
            println!("  POST /v1/gpu/warmup  - Warmup GPU cache");
            println!("  GET  /v1/gpu/status  - GPU status");
        }
        println!("  POST /generate       - Generate text (Q4_K fused)");
        println!();

        if batch_mode {
            println!("M4 Parity Target: 192 tok/s at concurrency >= 4");
            println!("Benchmark with: wrk -t4 -c4 -d30s http://{addr}/v1/completions");
            println!();
        }

        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "bind".to_string(),
                reason: format!("Failed to bind: {e}"),
            }
        })?;

        axum::serve(listener, app).await.map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "serve".to_string(),
                reason: format!("Server error: {e}"),
            }
        })?;
    } else if model_path.ends_with(".safetensors") {
        let file_data = std::fs::read(model_path).map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "read_model_file".to_string(),
                reason: format!("Failed to read {model_path}: {e}"),
            }
        })?;
        cli::load_safetensors_model(&file_data)?;
    } else if model_path.ends_with(".apr") {
        let file_data = std::fs::read(model_path).map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "read_model_file".to_string(),
                reason: format!("Failed to read {model_path}: {e}"),
            }
        })?;
        cli::load_apr_model(&file_data)?;
    } else {
        return Err(realizar::error::RealizarError::UnsupportedOperation {
            operation: "detect_model_type".to_string(),
            reason: "Unsupported file extension. Expected .gguf, .safetensors, or .apr".to_string(),
        });
    }

    Ok(())
}

// ============================================================================
// Model Commands (run, chat, list, pull, push)
// ============================================================================

#[cfg(feature = "registry")]
#[allow(clippy::too_many_arguments)]
async fn run_model(
    model_ref: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    system_prompt: Option<&str>,
    raw_mode: bool,
    force_gpu: bool,
) -> Result<()> {
    use realizar::chat_template::{auto_detect_template, ChatMessage};

    println!("Loading model: {model_ref}");
    if force_gpu {
        println!("GPU: FORCED (--gpu flag)");
    }

    let file_data = match ModelUri::parse(model_ref) {
        Ok(uri) => {
            let resolver = ModelResolver::new_default().map_err(|e| {
                realizar::error::RealizarError::UnsupportedOperation {
                    operation: "init_resolver".to_string(),
                    reason: format!("Failed to initialize Pacha resolver: {e}"),
                }
            })?;

            match resolver.resolve(&uri) {
                Ok(resolved) => {
                    match &resolved.source {
                        ModelSource::LocalFile(path) => {
                            println!("  Source: local file ({path})");
                        },
                        ModelSource::PachaLocal { name, version } => {
                            println!("  Source: Pacha registry ({name}:{version})");
                        },
                        ModelSource::PachaRemote {
                            host,
                            name,
                            version,
                        } => {
                            println!("  Source: Remote registry {host} ({name}:{version})");
                        },
                        ModelSource::HuggingFace { repo_id, revision } => {
                            let rev = revision.as_deref().unwrap_or("main");
                            println!("  Source: HuggingFace ({repo_id}@{rev})");
                        },
                    }
                    resolved.data
                },
                Err(e) => {
                    if std::path::Path::new(model_ref).exists() {
                        println!("  Source: local file");
                        std::fs::read(model_ref).map_err(|e| {
                            realizar::error::RealizarError::UnsupportedOperation {
                                operation: "read_model".to_string(),
                                reason: format!("Failed to read {model_ref}: {e}"),
                            }
                        })?
                    } else {
                        return Err(realizar::error::RealizarError::UnsupportedOperation {
                            operation: "resolve_model".to_string(),
                            reason: format!("Failed to resolve model: {e}"),
                        });
                    }
                },
            }
        },
        Err(_) => {
            if !std::path::Path::new(model_ref).exists() {
                return Err(realizar::error::RealizarError::ModelNotFound(
                    model_ref.to_string(),
                ));
            }
            println!("  Source: local file");
            std::fs::read(model_ref).map_err(|e| {
                realizar::error::RealizarError::UnsupportedOperation {
                    operation: "read_model".to_string(),
                    reason: format!("Failed to read {model_ref}: {e}"),
                }
            })?
        },
    };

    cli::display_model_info(model_ref, &file_data)?;
    println!();

    if let Some(prompt_text) = prompt {
        // Apply chat template formatting unless --raw mode
        let formatted_prompt = if raw_mode {
            println!("Prompt (raw): {prompt_text}");
            prompt_text.to_string()
        } else {
            // Auto-detect template from model name
            let template = auto_detect_template(model_ref);
            println!("Chat template: {:?}", template.format());

            // Build messages
            let mut messages = Vec::new();
            if let Some(sys) = system_prompt {
                messages.push(ChatMessage::system(sys));
            }
            messages.push(ChatMessage::user(prompt_text));

            // Format using detected template
            match template.format_conversation(&messages) {
                Ok(formatted) => {
                    println!("Prompt (formatted):");
                    // Show first 200 chars of formatted prompt
                    let preview: String = formatted.chars().take(200).collect();
                    println!(
                        "  {}{}",
                        preview,
                        if formatted.len() > 200 { "..." } else { "" }
                    );
                    formatted
                },
                Err(e) => {
                    eprintln!("Warning: chat template failed ({e}), using raw prompt");
                    prompt_text.to_string()
                },
            }
        };

        println!("Max tokens: {max_tokens}");
        println!("Temperature: {temperature}");
        println!("Format: {format}");
        println!();

        // Run actual GGUF inference with TruenoInferenceEngine
        run_gguf_inference(
            model_ref,
            &file_data,
            &formatted_prompt,
            max_tokens,
            temperature,
            format,
            force_gpu,
        )?;
    } else {
        println!("Interactive mode (Ctrl+D to exit)");
        println!();
        println!("Model loaded ({} bytes)", file_data.len());
        println!("Use a prompt argument:");
        println!("  realizar run {model_ref} \"Your prompt here\"");
    }

    Ok(())
}

/// Run GGUF inference with performance timing
///
/// IMP-130: Zero-copy model loading for <500ms startup time.
/// Uses QuantizedGGUFTransformer with borrowed refs to mmap data.
/// When `force_gpu` is true, uses OwnedQuantizedModel with CUDA acceleration.
fn run_gguf_inference(
    model_ref: &str,
    _file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    force_gpu: bool,
) -> Result<()> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, QuantizedGGUFTransformer};
    use std::time::Instant;

    // Handle --gpu flag warning when CUDA not available
    #[cfg(not(feature = "cuda"))]
    if force_gpu {
        eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
        eprintln!("Build with: cargo build --features cuda");
        eprintln!();
    }
    // Suppress unused warning when cuda feature not enabled
    #[cfg(not(feature = "cuda"))]
    let _ = force_gpu;

    let load_start = Instant::now();

    // Load model using memory-mapped file (same path as working examples)
    let mapped = MappedGGUFModel::from_path(model_ref).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "mmap_gguf".to_string(),
            reason: format!("Failed to mmap GGUF: {e}"),
        }
    })?;

    // GPU path: Use OwnedQuantizedModel with CUDA acceleration
    #[cfg(feature = "cuda")]
    if force_gpu {
        return run_gguf_inference_gpu(
            &mapped,
            prompt,
            max_tokens,
            temperature,
            format,
            load_start,
        );
    }

    // PAR-126: Five-Whys fix - use OwnedQuantizedModel for fast CPU inference
    // Root cause analysis:
    //   Why-1: CPU path was 14 tok/s vs Ollama's 200 tok/s
    //   Why-2: QuantizedGGUFTransformer uses mmap with per-matmul allocations
    //   Why-3: Each of 196 matmuls per token allocates/frees Vec
    //   Why-4: Vec allocation overhead + cache pollution from mmap page faults
    //   Why-5: OwnedQuantizedModel copies weights to RAM but uses _into methods
    // Solution: Use OwnedQuantizedModel - slower loading but faster inference
    let model = realizar::gguf::OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "load_model".to_string(),
            reason: format!("Failed to load model: {e}"),
        }
    })?;

    let load_time = load_start.elapsed();
    println!("Backend: CPU (AVX2 + SIMD)");
    println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);

    // Tokenize prompt using GGUF vocabulary
    let mut prompt_tokens: Vec<u32> = mapped
        .model
        .encode(prompt)
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());

    // Prepend BOS token if available
    if let Some(bos) = mapped.model.bos_token_id() {
        prompt_tokens.insert(0, bos);
    }
    let prompt_len = prompt_tokens.len();

    // Get EOS token for stopping
    let eos_token_id = mapped.model.eos_token_id();

    // Debug: show model info and encoded tokens
    let config = model.config();
    println!(
        "Architecture: {:?}, Hidden: {}, Layers: {}, Heads: {}/{} (KV)",
        mapped.model.architecture(),
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.num_kv_heads
    );
    println!(
        "Prompt tokens: {} (BOS={:?}, EOS={:?})",
        prompt_len,
        mapped.model.bos_token_id(),
        eos_token_id
    );
    println!("Temperature: {:.1}", temperature);
    println!();

    // Run inference with KV cache for O(n) per-token cost
    let gen_start = Instant::now();
    let max_seq_len = prompt_tokens.len() + max_tokens;
    let mut cache = OwnedQuantizedKVCache::from_config(config, max_seq_len);
    let mut all_tokens = prompt_tokens.clone();

    // Prefill: process prompt tokens to populate KV cache
    // We process all tokens but keep the logits from the last one
    let mut logits: Vec<f32> = vec![];
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        logits = model.forward_cached(token_id, &mut cache, pos)?;
    }

    // PAR-051: Diagnostic - show top5 logits after prefill
    // Re-enable by changing `false` to `true` for debugging
    #[allow(clippy::never_loop)]
    if std::env::var("CPU_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        let mut top5: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top5.truncate(5);
        eprintln!("[PAR-051] Prompt tokens: {:?}", prompt_tokens);
        eprintln!("[PAR-051] Logits top5 after prefill: {:?}", top5);
        let greedy_token = top5[0].0 as u32;
        let decoded = mapped.model.decode(&[greedy_token]);
        eprintln!(
            "[PAR-051] Greedy next token: {} = {:?}",
            greedy_token, decoded
        );

        // Check embedding - compare first 5 values
        let last_token = prompt_tokens[prompt_tokens.len() - 1];
        let embed = model.embed(&[last_token]);
        eprintln!(
            "[PAR-051] Last token {} embed[0..5]: {:?}",
            last_token,
            &embed[..5.min(embed.len())]
        );
        eprintln!("[PAR-051] embed sum: {:.6}", embed.iter().sum::<f32>());

        // Check model config
        let cfg = model.config();
        eprintln!(
            "[PAR-051] Config: hidden={}, heads={}/{}, layers={}, vocab={}, eps={:e}",
            cfg.hidden_dim,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.num_layers,
            cfg.vocab_size,
            cfg.eps
        );

        // Check logit at index 29906 (token "2")
        let logit_2 = logits.get(29906).copied().unwrap_or(f32::NAN);
        eprintln!("[PAR-051] Logit for token 29906 ('2'): {:.6}", logit_2);
    }

    // Decode: generate new tokens one at a time
    // First iteration uses logits from prefill, subsequent use new logits
    for i in 0..max_tokens {
        // For first iteration, use logits from prefill; otherwise compute new ones
        if i > 0 {
            let position = prompt_tokens.len() + i - 1;
            let last_token = *all_tokens
                .last()
                .expect("all_tokens should not be empty during generation");
            logits = model.forward_cached(last_token, &mut cache, position)?;
        }

        // Sample next token
        let next_token = if temperature <= 0.01 {
            // Greedy decoding
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32)
        } else {
            // Temperature sampling
            QuantizedGGUFTransformer::sample_topk(&logits, temperature, 40)
        };

        // PERF-002: Debug code removed (was PAR-058-DEBUG and PAR-060)

        // Stop on EOS
        if let Some(eos) = eos_token_id {
            if next_token == eos {
                // PERF-002: eprintln removed for performance
                break;
            }
        }

        all_tokens.push(next_token);
    }

    let generated = all_tokens;
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len() - prompt_len;
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using GGUF vocabulary, replacing SentencePiece markers with spaces
    let output_text = mapped
        .model
        .decode(&generated[prompt_len..])
        .replace('▁', " ");

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time.as_secs_f64() * 1000.0,
                "tokens_per_second": tokens_per_sec,
                "temperature": temperature,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        },
        _ => {
            println!(
                "Generated ({tokens_generated} tokens in {:.2}ms):",
                gen_time.as_secs_f64() * 1000.0
            );
            println!("{prompt}{output_text}");
            println!();
            println!("Performance: {:.1} tok/s", tokens_per_sec);
        },
    }

    Ok(())
}

/// Run GGUF inference with CUDA GPU acceleration
///
/// Uses OwnedQuantizedModel with CUDA backend for high-performance inference.
/// Called when --gpu flag is specified and CUDA feature is enabled.
#[cfg(feature = "cuda")]
fn run_gguf_inference_gpu(
    mapped: &realizar::gguf::MappedGGUFModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    load_start: std::time::Instant,
) -> Result<()> {
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig};
    use std::time::Instant;

    println!("Backend: CUDA (GPU)");
    println!("Creating quantized model with CUDA acceleration...");

    // Create owned quantized model (required for CUDA - can't use borrowed mmap data)
    let quantized_model = OwnedQuantizedModel::from_mapped(mapped).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "create_quantized".to_string(),
            reason: format!("Failed to create quantized model: {e}"),
        }
    })?;

    // Get config info before wrapping
    let vocab_size = quantized_model.config.vocab_size;
    let hidden_dim = quantized_model.config.hidden_dim;
    let num_layers = quantized_model.layers.len();

    // PAR-046: Create OwnedQuantizedModelCuda wrapper for actual GPU acceleration
    // The previous implementation used OwnedQuantizedModel.enable_cuda() which only
    // initialized the executor but forward_cached still used CPU code paths.
    let max_seq_len = 256 + max_tokens; // Allow for prompt + generation
    let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(quantized_model, 0, max_seq_len)
        .map_err(|e| realizar::error::RealizarError::UnsupportedOperation {
            operation: "OwnedQuantizedModelCuda::new".to_string(),
            reason: format!("CUDA initialization failed: {e}"),
        })?;
    println!("  CUDA enabled on GPU: {}", cuda_model.device_name());

    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);

    // Tokenize prompt using GGUF vocabulary
    let mut prompt_tokens: Vec<u32> = mapped
        .model
        .encode(prompt)
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());

    // Prepend BOS token if available
    if let Some(bos) = mapped.model.bos_token_id() {
        prompt_tokens.insert(0, bos);
    }
    let prompt_len = prompt_tokens.len();

    // Get EOS token for stopping
    let eos_token_id = mapped.model.eos_token_id();

    println!(
        "Vocab size: {}, Hidden dim: {}, Layers: {}",
        vocab_size, hidden_dim, num_layers
    );
    println!(
        "Prompt tokens: {} (BOS={:?}, EOS={:?})",
        prompt_len,
        mapped.model.bos_token_id(),
        eos_token_id
    );
    println!("Temperature: {:.1}", temperature);
    println!();

    // PAR-046: Use CUDA-accelerated generation with GPU-resident KV cache
    // This calls generate_cuda_with_cache -> forward_single_cuda_with_cache -> GPU kernels
    let gen_start = Instant::now();

    // Build stop tokens list
    let mut stop_tokens = Vec::new();
    if let Some(eos) = eos_token_id {
        stop_tokens.push(eos);
    }

    // Configure CUDA generation
    let gen_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature <= 0.01 { 1 } else { 40 },
        stop_tokens,
    };

    // PAR-047: Use generate_full_cuda_with_cache for maximum GPU acceleration
    // This path uses:
    // - GPU matmul for QKV, output projection, and FFN
    // - GPU incremental_attention_gpu with GQA support (PAR-021)
    // - Proper SwiGLU activation (PAR-015)
    // PAR-057: Use GPU-resident path for maximum performance (pre-uploads weights, minimal syncs)
    // Falls back to generate_full_cuda_with_cache if architecture not supported
    // PAR-058: Test GPU-resident vs standard CUDA path
    let generated = if cuda_model.supports_gpu_resident() {
        println!("Using GPU-resident path (pre-uploaded weights, ~2 syncs/token)");
        cuda_model
            .generate_gpu_resident(&prompt_tokens, &gen_config)
            .map_err(|e| realizar::error::RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident".to_string(),
                reason: format!("GPU-resident generation failed: {e}"),
            })?
    } else {
        println!("Using standard CUDA path");
        cuda_model
            .generate_full_cuda_with_cache(&prompt_tokens, &gen_config)
            .map_err(|e| realizar::error::RealizarError::UnsupportedOperation {
                operation: "generate_full_cuda_with_cache".to_string(),
                reason: format!("CUDA generation failed: {e}"),
            })?
    };
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using GGUF vocabulary, replacing SentencePiece markers with spaces
    let output_text = mapped
        .model
        .decode(&generated[prompt_len..])
        .replace('▁', " ");

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": "GGUF (CUDA)",
                "backend": "GPU",
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time.as_secs_f64() * 1000.0,
                "tokens_per_second": tokens_per_sec,
                "temperature": temperature,
                "cuda_enabled": true,
                "cuda_device": cuda_model.device_name(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        },
        _ => {
            println!(
                "Generated ({tokens_generated} tokens in {:.2}ms):",
                gen_time.as_secs_f64() * 1000.0
            );
            println!("{prompt}{output_text}");
            println!();
            println!("Performance: {:.1} tok/s (GPU)", tokens_per_sec);
        },
    }

    Ok(())
}

/// Run APR inference with performance timing
#[allow(dead_code)] // APR format support - called when detect_format returns Apr
fn run_apr_inference(
    model_ref: &str,
    file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
) -> Result<()> {
    use realizar::apr_transformer::AprTransformer;
    use std::time::Instant;

    let load_start = Instant::now();

    // Load APR transformer
    let transformer = AprTransformer::from_apr_bytes(file_data).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "parse_apr".to_string(),
            reason: format!("Failed to parse APR: {e}"),
        }
    })?;

    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);

    // Simple tokenization (split by chars for now - real tokenizer would be better)
    let prompt_tokens: Vec<u32> = prompt.chars().map(|c| c as u32).collect();
    let prompt_len = prompt_tokens.len();

    println!("Prompt tokens: {}", prompt_len);
    println!("Temperature: {:.1} (using greedy decoding)", temperature);
    println!();

    // Run inference with timing
    let gen_start = Instant::now();
    let generated = transformer.generate(&prompt_tokens, max_tokens)?;
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output (simple ASCII for now)
    let output_text: String = generated[prompt_len..]
        .iter()
        .map(|&t| char::from_u32(t.min(127)).unwrap_or('?'))
        .collect();

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "format": "APR",
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time.as_secs_f64() * 1000.0,
                "tokens_per_second": tokens_per_sec,
                "temperature": temperature,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        },
        _ => {
            println!(
                "Generated ({tokens_generated} tokens in {:.2}ms):",
                gen_time.as_secs_f64() * 1000.0
            );
            println!("{prompt}{output_text}");
            println!();
            println!("Performance: {:.1} tok/s", tokens_per_sec);
        },
    }

    Ok(())
}

#[cfg(not(feature = "registry"))]
#[allow(clippy::too_many_arguments)]
async fn run_model(
    model_ref: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    system_prompt: Option<&str>,
    raw_mode: bool,
    force_gpu: bool,
) -> Result<()> {
    use realizar::chat_template::{auto_detect_template, ChatMessage};

    println!("Loading model: {model_ref}");
    if force_gpu {
        println!("GPU: FORCED (--gpu flag)");
    }

    if cli::is_local_file_path(model_ref) {
        if !std::path::Path::new(model_ref).exists() {
            return Err(realizar::error::RealizarError::ModelNotFound(
                model_ref.to_string(),
            ));
        }
        println!("  Source: local file");
    } else if model_ref.starts_with("pacha://") || model_ref.contains(':') {
        println!("  Source: Pacha registry");
        println!();
        println!("Enable registry support: --features registry");
        println!("Or use a local file path:");
        println!("  realizar run ./model.gguf \"Your prompt\"");
        return Ok(());
    } else if model_ref.starts_with("hf://") {
        println!("  Source: HuggingFace Hub");
        println!();
        println!("Enable registry support: --features registry");
        println!("Or download manually and use:");
        println!("  realizar run ./model.gguf \"Your prompt\"");
        return Ok(());
    }

    let file_data = std::fs::read(model_ref).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "read_model".to_string(),
            reason: format!("Failed to read {model_ref}: {e}"),
        }
    })?;

    cli::display_model_info(model_ref, &file_data)?;
    println!();

    if let Some(prompt_text) = prompt {
        // Apply chat template formatting unless --raw mode
        let formatted_prompt = if raw_mode {
            println!("Prompt (raw): {prompt_text}");
            prompt_text.to_string()
        } else {
            // Auto-detect template from model name
            let template = auto_detect_template(model_ref);
            println!("Chat template: {:?}", template.format());

            // Build messages
            let mut messages = Vec::new();
            if let Some(sys) = system_prompt {
                messages.push(ChatMessage::system(sys));
            }
            messages.push(ChatMessage::user(prompt_text));

            // Format using detected template
            match template.format_conversation(&messages) {
                Ok(formatted) => {
                    println!("Prompt (formatted):");
                    // Show first 200 chars of formatted prompt
                    let preview: String = formatted.chars().take(200).collect();
                    println!(
                        "  {}{}",
                        preview,
                        if formatted.len() > 200 { "..." } else { "" }
                    );
                    formatted
                },
                Err(e) => {
                    eprintln!("Warning: chat template failed ({e}), using raw prompt");
                    prompt_text.to_string()
                },
            }
        };

        println!("Max tokens: {max_tokens}");
        println!("Temperature: {temperature}");
        println!("Format: {format}");
        println!();

        // Detect format and run appropriate inference
        use realizar::format::{detect_format, ModelFormat};
        let detected_format = detect_format(&file_data).unwrap_or(ModelFormat::Gguf);

        match detected_format {
            ModelFormat::Apr => {
                run_apr_inference(
                    model_ref,
                    &file_data,
                    &formatted_prompt,
                    max_tokens,
                    temperature,
                    format,
                )?;
            },
            ModelFormat::Gguf | ModelFormat::SafeTensors => {
                run_gguf_inference(
                    model_ref,
                    &file_data,
                    &formatted_prompt,
                    max_tokens,
                    temperature,
                    format,
                    force_gpu,
                )?;
            },
        }
    } else {
        println!("Interactive mode (Ctrl+D to exit)");
        println!();
        println!("Model loaded ({} bytes)", file_data.len());
        println!("Use a prompt argument:");
        println!("  realizar run {model_ref} \"Your prompt here\"");
    }

    Ok(())
}

#[cfg(feature = "registry")]
async fn run_chat(
    model_ref: &str,
    system_prompt: Option<&str>,
    history_file: Option<&str>,
) -> Result<()> {
    use std::io::{BufRead, Write};

    println!("Loading model: {model_ref}");

    let file_data = match ModelUri::parse(model_ref) {
        Ok(uri) => {
            let resolver = ModelResolver::new_default().map_err(|e| {
                realizar::error::RealizarError::UnsupportedOperation {
                    operation: "init_resolver".to_string(),
                    reason: format!("Failed to initialize resolver: {e}"),
                }
            })?;

            match resolver.resolve(&uri) {
                Ok(resolved) => {
                    println!("  Source: {:?}", resolved.source);
                    resolved.data
                },
                Err(e) => {
                    if std::path::Path::new(model_ref).exists() {
                        std::fs::read(model_ref).map_err(|e| {
                            realizar::error::RealizarError::UnsupportedOperation {
                                operation: "read_model".to_string(),
                                reason: format!("Failed to read: {e}"),
                            }
                        })?
                    } else {
                        return Err(realizar::error::RealizarError::UnsupportedOperation {
                            operation: "resolve_model".to_string(),
                            reason: format!("Failed to resolve: {e}"),
                        });
                    }
                },
            }
        },
        Err(_) => {
            if !std::path::Path::new(model_ref).exists() {
                return Err(realizar::error::RealizarError::ModelNotFound(
                    model_ref.to_string(),
                ));
            }
            std::fs::read(model_ref).map_err(|e| {
                realizar::error::RealizarError::UnsupportedOperation {
                    operation: "read_model".to_string(),
                    reason: format!("Failed to read: {e}"),
                }
            })?
        },
    };

    cli::display_model_info(model_ref, &file_data)?;
    println!("  Size: {} bytes", file_data.len());
    println!();

    let mut history: Vec<(String, String)> = if let Some(path) = history_file {
        if std::path::Path::new(path).exists() {
            let content = std::fs::read_to_string(path).unwrap_or_default();
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    if let Some(sys) = system_prompt {
        println!("System: {sys}");
        println!();
    }

    println!("Chat mode active. Type 'exit' or Ctrl+D to quit.");
    println!("Commands: /clear (clear history), /history (show history)");
    println!();

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    loop {
        print!(">>> ");
        stdout.flush().ok();

        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => {
                println!();
                break;
            },
            Ok(_) => {
                let input = input.trim();

                if input.is_empty() {
                    continue;
                }

                if input == "exit" || input == "/exit" || input == "/quit" {
                    break;
                }

                if input == "/clear" {
                    history.clear();
                    println!("History cleared.");
                    continue;
                }

                if input == "/history" {
                    if history.is_empty() {
                        println!("No history.");
                    } else {
                        for (i, (user, assistant)) in history.iter().enumerate() {
                            println!("[{}] User: {}", i + 1, user);
                            println!("    Assistant: {}", assistant);
                        }
                    }
                    continue;
                }

                let response = format!("[Model loaded: {} bytes] Echo: {}", file_data.len(), input);

                println!();
                println!("{response}");
                println!();

                history.push((input.to_string(), response));
            },
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            },
        }
    }

    if let Some(path) = history_file {
        if let Ok(json) = serde_json::to_string_pretty(&history) {
            let _ = std::fs::write(path, json);
            println!("History saved to {path}");
        }
    }

    println!("Goodbye!");
    Ok(())
}

#[cfg(not(feature = "registry"))]
async fn run_chat(
    model_ref: &str,
    system_prompt: Option<&str>,
    history_file: Option<&str>,
) -> Result<()> {
    use std::io::{BufRead, Write};

    println!("Loading model: {model_ref}");

    if !std::path::Path::new(model_ref).exists()
        && !model_ref.starts_with("pacha://")
        && !model_ref.starts_with("hf://")
    {
        return Err(realizar::error::RealizarError::ModelNotFound(
            model_ref.to_string(),
        ));
    }

    if model_ref.starts_with("pacha://") || model_ref.starts_with("hf://") {
        println!("Registry URIs require --features registry");
        println!("Use a local file path instead.");
        return Ok(());
    }

    let file_data = std::fs::read(model_ref).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "read_model".to_string(),
            reason: format!("Failed to read: {e}"),
        }
    })?;

    cli::display_model_info(model_ref, &file_data)?;
    println!("  Size: {} bytes", file_data.len());
    println!();

    let mut history: Vec<(String, String)> = if let Some(path) = history_file {
        if std::path::Path::new(path).exists() {
            let content = std::fs::read_to_string(path).unwrap_or_default();
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    if let Some(sys) = system_prompt {
        println!("System: {sys}");
        println!();
    }

    println!("Chat mode active. Type 'exit' or Ctrl+D to quit.");
    println!("Commands: /clear (clear history), /history (show history)");
    println!();

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    loop {
        print!(">>> ");
        stdout.flush().ok();

        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => {
                println!();
                break;
            },
            Ok(_) => {
                let input = input.trim();

                if input.is_empty() {
                    continue;
                }

                if input == "exit" || input == "/exit" || input == "/quit" {
                    break;
                }

                if input == "/clear" {
                    history.clear();
                    println!("History cleared.");
                    continue;
                }

                if input == "/history" {
                    if history.is_empty() {
                        println!("No history.");
                    } else {
                        for (i, (user, assistant)) in history.iter().enumerate() {
                            println!("[{}] User: {}", i + 1, user);
                            println!("    Assistant: {}", assistant);
                        }
                    }
                    continue;
                }

                let response = format!("[Model loaded: {} bytes] Echo: {}", file_data.len(), input);

                println!();
                println!("{response}");
                println!();

                history.push((input.to_string(), response));
            },
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            },
        }
    }

    if let Some(path) = history_file {
        if let Ok(json) = serde_json::to_string_pretty(&history) {
            let _ = std::fs::write(path, json);
            println!("History saved to {path}");
        }
    }

    println!("Goodbye!");
    Ok(())
}

#[cfg(feature = "registry")]
fn list_models(remote: Option<&str>, format: &str) -> Result<()> {
    println!("Available Models");
    println!("================");
    println!();

    if let Some(remote_url) = remote {
        println!("Remote registry: {remote_url}");
        println!();
        println!("Note: Remote registry listing requires --features remote in Pacha.");
        return Ok(());
    }

    let resolver = match ModelResolver::new_default() {
        Ok(r) => r,
        Err(_) => {
            println!("No Pacha registry found.");
            println!();
            println!("Initialize registry:");
            println!("  pacha init");
            println!();
            println!("Or run a local file:");
            println!("  realizar run ./model.gguf \"prompt\"");
            return Ok(());
        },
    };

    if !resolver.has_registry() {
        println!("No Pacha registry found.");
        println!();
        println!("Initialize registry:");
        println!("  pacha init");
        return Ok(());
    }

    let models = match resolver.list_models() {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to list models: {e}");
            return Ok(());
        },
    };

    if models.is_empty() {
        println!("No models found in local registry.");
        println!();
        println!("Pull a model:");
        println!("  realizar pull llama3:8b");
        println!();
        println!("Or run a local file:");
        println!("  realizar run ./model.gguf \"prompt\"");
    } else {
        match format {
            "json" => {
                let json_models: Vec<_> = models
                    .iter()
                    .map(|name| {
                        let versions = resolver.list_versions(name).unwrap_or_default();
                        serde_json::json!({
                            "name": name,
                            "versions": versions.len()
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_models).unwrap_or_default()
                );
            },
            _ => {
                println!("{:<40} {:>12}", "NAME", "VERSIONS");
                println!("{}", "-".repeat(54));
                for name in &models {
                    let versions = resolver.list_versions(name).unwrap_or_default();
                    println!("{:<40} {:>12}", name, versions.len());
                }
            },
        }
    }

    Ok(())
}

#[cfg(not(feature = "registry"))]
fn list_models(remote: Option<&str>, format: &str) -> Result<()> {
    println!("Available Models");
    println!("================");
    println!();

    if let Some(remote_url) = remote {
        println!("Remote registry: {remote_url}");
        println!();
        println!("Note: Remote registry listing requires --features registry.");
        return Ok(());
    }

    let pacha_dir = cli::home_dir()
        .map(|h| h.join(".pacha").join("models"))
        .unwrap_or_else(|| std::path::PathBuf::from(".pacha/models"));

    if !pacha_dir.exists() {
        println!("No models found in local registry.");
        println!();
        println!("Pull a model:");
        println!("  realizar pull llama3:8b");
        println!();
        println!("Or run a local file:");
        println!("  realizar run ./model.gguf \"prompt\"");
        return Ok(());
    }

    let mut models_found = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&pacha_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.ends_with(".gguf")
                    || name.ends_with(".safetensors")
                    || name.ends_with(".apr")
                {
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    models_found.push((name.to_string(), size));
                }
            }
        }
    }

    if models_found.is_empty() {
        println!("No models found in {}", pacha_dir.display());
    } else {
        match format {
            "json" => {
                let json_models: Vec<_> = models_found
                    .iter()
                    .map(|(name, size)| {
                        serde_json::json!({
                            "name": name,
                            "size_bytes": size,
                            "size_human": cli::format_size(*size)
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_models).unwrap_or_default()
                );
            },
            _ => {
                println!("{:<40} {:>12}", "NAME", "SIZE");
                println!("{}", "-".repeat(54));
                for (name, size) in &models_found {
                    println!("{:<40} {:>12}", name, cli::format_size(*size));
                }
            },
        }
    }

    Ok(())
}

#[cfg(feature = "registry")]
async fn pull_model(model_ref: &str, force: bool, quantize: Option<&str>) -> Result<()> {
    println!("Pulling model: {model_ref}");
    if force {
        println!("  Force: re-downloading even if cached");
    }
    if let Some(q) = quantize {
        println!("  Quantize: {q}");
    }
    println!();

    let uri = ModelUri::parse(model_ref).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "parse_uri".to_string(),
            reason: format!("Invalid model reference: {e}"),
        }
    })?;

    let resolver = ModelResolver::new_default().map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "init_resolver".to_string(),
            reason: format!("Failed to initialize Pacha resolver: {e}"),
        }
    })?;

    if !force && resolver.exists(&uri) {
        println!("Model already cached locally.");
        println!("Use --force to re-download.");
        return Ok(());
    }

    println!("Downloading...");
    let resolved = resolver.resolve(&uri).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "pull_model".to_string(),
            reason: format!("Failed to pull model: {e}"),
        }
    })?;

    println!("  Downloaded: {} bytes", resolved.data.len());

    match &resolved.source {
        ModelSource::LocalFile(path) => {
            println!("  Source: local file ({path})");
        },
        ModelSource::PachaLocal { name, version } => {
            println!("  Source: Pacha local ({name}:{version})");
        },
        ModelSource::PachaRemote {
            host,
            name,
            version,
        } => {
            println!("  Source: Remote {host} ({name}:{version})");
            println!("  Cached to local registry.");
        },
        ModelSource::HuggingFace { repo_id, revision } => {
            let rev = revision.as_deref().unwrap_or("main");
            println!("  Source: HuggingFace ({repo_id}@{rev})");
        },
    }

    println!();
    println!("Model ready! Run with:");
    println!("  realizar run {model_ref} \"Your prompt\"");

    Ok(())
}

#[cfg(not(feature = "registry"))]
async fn pull_model(model_ref: &str, force: bool, quantize: Option<&str>) -> Result<()> {
    println!("Pulling model: {model_ref}");
    if force {
        println!("  Force: re-downloading even if cached");
    }
    if let Some(q) = quantize {
        println!("  Quantize: {q}");
    }
    println!();

    if let Some(hf_path) = model_ref.strip_prefix("hf://") {
        println!("Source: HuggingFace Hub");
        println!("Model: {hf_path}");
        println!();
        println!("Enable registry support: --features registry");
        println!("Or manual download:");
        println!("  huggingface-cli download {hf_path}");
    } else if let Some(pacha_path) = model_ref.strip_prefix("pacha://") {
        println!("Source: Pacha Registry");
        println!("Model: {pacha_path}");
        println!();
        println!("Enable registry support: --features registry");
    } else {
        println!("Source: Default registry (Pacha)");
        println!("Model: {model_ref}");
        println!();
        println!("Enable registry support: --features registry");
        println!("Or download manually and use:");
        println!("  realizar run ./downloaded-model.gguf \"prompt\"");
    }

    Ok(())
}

#[cfg(feature = "registry")]
async fn push_model(model_ref: &str, target: Option<&str>) -> Result<()> {
    use pacha::Registry;

    println!("Pushing model: {model_ref}");

    let (name, version_str) = if let Some(idx) = model_ref.rfind(':') {
        (&model_ref[..idx], &model_ref[idx + 1..])
    } else {
        (model_ref, "latest")
    };

    println!("  Name: {name}");
    println!("  Version: {version_str}");

    if let Some(t) = target {
        println!("  Target: {t}");
        println!();
        println!("Remote push requires --features remote in Pacha.");
        println!("Use pacha CLI for remote operations:");
        println!("  pacha push {model_ref} --to {t}");
    } else {
        println!("  Target: local Pacha registry");
        println!();

        let local_path = format!("{name}.gguf");
        if !std::path::Path::new(&local_path).exists() {
            println!("Local file not found: {local_path}");
            println!();
            println!("To push a model to registry:");
            println!("  1. Have the model file: {name}.gguf");
            println!("  2. Run: realizar push {name}:{version_str}");
            return Ok(());
        }

        let data = std::fs::read(&local_path).map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "read_model".to_string(),
                reason: format!("Failed to read {local_path}: {e}"),
            }
        })?;

        let registry = Registry::open_default().map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "open_registry".to_string(),
                reason: format!("Failed to open Pacha registry: {e}"),
            }
        })?;

        let version = parse_model_version(version_str)?;

        let card = pacha::model::ModelCard::new(format!("Model {name} pushed via realizar"));
        registry
            .register_model(name, &version, &data, card)
            .map_err(|e| realizar::error::RealizarError::UnsupportedOperation {
                operation: "register_model".to_string(),
                reason: format!("Failed to register model: {e}"),
            })?;

        println!("Model registered successfully!");
        println!();
        println!("Run with:");
        println!("  realizar run pacha://{name}:{version_str} \"Your prompt\"");
    }

    Ok(())
}

#[cfg(not(feature = "registry"))]
async fn push_model(model_ref: &str, target: Option<&str>) -> Result<()> {
    println!("Pushing model: {model_ref}");
    if let Some(t) = target {
        println!("  Target: {t}");
    } else {
        println!("  Target: default Pacha registry");
    }
    println!();
    println!("Enable registry support: --features registry");
    Ok(())
}

#[cfg(feature = "registry")]
fn parse_model_version(s: &str) -> Result<pacha::model::ModelVersion> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() == 3 {
        let major: u32 =
            parts[0]
                .parse()
                .map_err(|_| realizar::error::RealizarError::UnsupportedOperation {
                    operation: "parse_version".to_string(),
                    reason: format!("Invalid version: {s}"),
                })?;
        let minor: u32 =
            parts[1]
                .parse()
                .map_err(|_| realizar::error::RealizarError::UnsupportedOperation {
                    operation: "parse_version".to_string(),
                    reason: format!("Invalid version: {s}"),
                })?;
        let patch: u32 =
            parts[2]
                .parse()
                .map_err(|_| realizar::error::RealizarError::UnsupportedOperation {
                    operation: "parse_version".to_string(),
                    reason: format!("Invalid version: {s}"),
                })?;
        return Ok(pacha::model::ModelVersion::new(major, minor, patch));
    }

    if s == "latest" {
        return Ok(pacha::model::ModelVersion::new(1, 0, 0));
    }
    if let Ok(major) = s.parse::<u32>() {
        return Ok(pacha::model::ModelVersion::new(major, 0, 0));
    }

    Err(realizar::error::RealizarError::UnsupportedOperation {
        operation: "parse_version".to_string(),
        reason: format!("Invalid version format: {s}. Expected: x.y.z"),
    })
}
