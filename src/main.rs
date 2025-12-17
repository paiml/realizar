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
        } => {
            run_model(&model, prompt.as_deref(), max_tokens, temperature, &format).await?;
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
        } => {
            if demo {
                serve_demo(&host, port).await?;
            } else if let Some(model_path) = model {
                serve_model(&host, port, &model_path, batch).await?;
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

async fn serve_model(host: &str, port: u16, model_path: &str, batch_mode: bool) -> Result<()> {
    use realizar::gguf::MappedGGUFModel;

    println!("Loading model from: {model_path}");
    if batch_mode {
        println!("Mode: BATCH (PARITY-093 M4 parity)");
    } else {
        println!("Mode: SINGLE-REQUEST");
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

        // PARITY-113: Enable CUDA backend via REALIZAR_BACKEND environment variable
        #[cfg(feature = "cuda")]
        let mut quantized_model = quantized_model;
        #[cfg(feature = "cuda")]
        if std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false)
        {
            println!("Enabling CUDA backend (REALIZAR_BACKEND=cuda)...");
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
async fn run_model(
    model_ref: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    format: &str,
) -> Result<()> {
    println!("Loading model: {model_ref}");

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
        println!("Prompt: {prompt_text}");
        println!("Max tokens: {max_tokens}");
        println!("Temperature: {temperature}");
        println!("Format: {format}");
        println!();

        // Run actual GGUF inference with TruenoInferenceEngine
        run_gguf_inference(
            model_ref,
            &file_data,
            prompt_text,
            max_tokens,
            temperature,
            format,
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
fn run_gguf_inference(
    model_ref: &str,
    file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
) -> Result<()> {
    use realizar::gguf::{GGUFModel, GGUFTransformer};
    use realizar::inference::TruenoInferenceEngine;
    use std::time::Instant;

    let load_start = Instant::now();

    // Parse GGUF model first
    let gguf_model = GGUFModel::from_bytes(file_data).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "parse_gguf".to_string(),
            reason: format!("Failed to parse GGUF: {e}"),
        }
    })?;

    // Create transformer from model
    let transformer = GGUFTransformer::from_gguf(&gguf_model, file_data).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "create_transformer".to_string(),
            reason: format!("Failed to create transformer: {e}"),
        }
    })?;

    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);

    // Create inference engine
    let engine = TruenoInferenceEngine::from_gguf_transformer(transformer);

    // Simple tokenization (split by chars for now - real tokenizer would be better)
    let prompt_tokens: Vec<u32> = prompt.chars().map(|c| c as u32).collect();
    let prompt_len = prompt_tokens.len();

    println!("Prompt tokens: {}", prompt_len);
    println!("Temperature: {:.1} (using greedy decoding)", temperature);
    println!();

    // Run inference with timing using KV cache for O(n) complexity
    // Note: temperature is for display only - generate_with_cache uses greedy decoding
    let gen_start = Instant::now();
    let generated = engine.generate_with_cache(&prompt_tokens, max_tokens, None)?;
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len() - prompt_len;
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
async fn run_model(
    model_ref: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    format: &str,
) -> Result<()> {
    println!("Loading model: {model_ref}");

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
        println!("Prompt: {prompt_text}");
        println!("Max tokens: {max_tokens}");
        println!("Temperature: {temperature}");
        println!("Format: {format}");
        println!();

        // Run actual GGUF inference with TruenoInferenceEngine
        run_gguf_inference(
            model_ref,
            &file_data,
            prompt_text,
            max_tokens,
            temperature,
            format,
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
