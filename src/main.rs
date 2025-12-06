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
use realizar::{
    api::{create_router, AppState},
    error::Result,
};

#[cfg(feature = "registry")]
use pacha::resolver::{ModelResolver, ModelSource};
#[cfg(feature = "registry")]
use pacha::uri::ModelUri;
#[cfg(feature = "registry")]
use pacha::remote::RegistryAuth;

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
    ///
    /// Examples:
    ///   realizar run llama3:8b "What is Rust?"
    ///   realizar run pacha://my-model:v1.0 "Hello"
    ///   realizar run ./model.gguf "Prompt"
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
    ///
    /// Examples:
    ///   realizar chat llama3:8b
    ///   realizar chat pacha://assistant:v2.0
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
    ///
    /// Examples:
    ///   realizar list
    ///   realizar list --remote pacha://registry.example.com
    List {
        /// Show remote registry models
        #[arg(short, long)]
        remote: Option<String>,

        /// Output format: table, json
        #[arg(short, long, default_value = "table")]
        format: String,
    },
    /// Pull a model from registry (like `ollama pull`)
    ///
    /// Examples:
    ///   realizar pull llama3:8b
    ///   realizar pull pacha://registry.example.com/model:v1.0
    ///   realizar pull hf://meta-llama/Llama-3-8B
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
    ///
    /// Examples:
    ///   realizar push my-model:v1.0
    ///   realizar push --to pacha://registry.example.com my-model:latest
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

        /// Path to model file (GGUF or Safetensors)
        #[arg(short, long)]
        model: Option<String>,

        /// Use demo model for testing
        #[arg(long)]
        demo: bool,

        /// Enable OpenAI-compatible API at /v1/*
        #[arg(long, default_value = "true")]
        openai_api: bool,
    },
    /// Run performance benchmarks (wraps cargo bench)
    Bench {
        /// Benchmark suite to run (tensor_ops, inference, cache, tokenizer, quantize, lambda)
        #[arg(value_name = "SUITE")]
        suite: Option<String>,

        /// List available benchmark suites
        #[arg(short, long)]
        list: bool,
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
    let cli = Cli::parse();

    match cli.command {
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
        } => {
            if demo {
                serve_demo(&host, port).await?;
            } else if let Some(model_path) = model {
                serve_model(&host, port, &model_path).await?;
            } else {
                eprintln!("Error: Either --model or --demo must be specified");
                eprintln!();
                eprintln!("Usage:");
                eprintln!("  realizar serve --demo              # Use demo model");
                eprintln!("  realizar serve --model path.gguf   # Load GGUF model");
                std::process::exit(1);
            }
        },
        Commands::Bench { suite, list } => {
            run_benchmarks(suite, list)?;
        },
        Commands::Viz { color, samples } => {
            run_visualization(color, samples);
        },
        Commands::Info => {
            println!("Realizar v{}", realizar::VERSION);
            println!("Pure Rust ML inference engine");
            println!();
            println!("Features:");
            println!("  - GGUF and Safetensors model formats");
            println!("  - Transformer inference (LLaMA architecture)");
            println!("  - BPE and SentencePiece tokenizers");
            println!("  - Greedy, top-k, and top-p sampling");
            println!("  - REST API for inference");
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

async fn serve_model(_host: &str, _port: u16, model_path: &str) -> Result<()> {
    println!("Loading model from: {model_path}");
    println!();

    // Read model file
    let file_data = std::fs::read(model_path).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "read_model_file".to_string(),
            reason: format!("Failed to read {model_path}: {e}"),
        }
    })?;

    // Detect file type and parse
    if model_path.ends_with(".gguf") {
        load_gguf_model(&file_data)?;
    } else if model_path.ends_with(".safetensors") {
        load_safetensors_model(&file_data)?;
    } else {
        return Err(realizar::error::RealizarError::UnsupportedOperation {
            operation: "detect_model_type".to_string(),
            reason: "Unsupported file extension. Expected .gguf or .safetensors".to_string(),
        });
    }

    Ok(())
}

fn load_gguf_model(file_data: &[u8]) -> Result<()> {
    use realizar::gguf::GGUFModel;

    println!("Parsing GGUF file...");
    let gguf = GGUFModel::from_bytes(file_data)?;

    println!("✓ Successfully parsed GGUF file");
    println!();
    println!("Model Information:");
    println!("  Version: {}", gguf.header.version);
    println!("  Tensors: {}", gguf.header.tensor_count);
    println!("  Metadata entries: {}", gguf.header.metadata_count);
    println!();

    // Show some metadata
    if !gguf.metadata.is_empty() {
        println!("Metadata (first 5 entries):");
        for (key, _value) in gguf.metadata.iter().take(5) {
            println!("  - {key}");
        }
        if gguf.metadata.len() > 5 {
            println!("  ... and {} more", gguf.metadata.len() - 5);
        }
        println!();
    }

    // Show tensor names
    if !gguf.tensors.is_empty() {
        println!("Tensors (first 10):");
        for tensor in gguf.tensors.iter().take(10) {
            let dims: Vec<String> = tensor.dims.iter().map(|d| d.to_string()).collect();
            println!(
                "  - {} [{}, qtype={}]",
                tensor.name,
                dims.join("×"),
                tensor.qtype
            );
        }
        if gguf.tensors.len() > 10 {
            println!("  ... and {} more", gguf.tensors.len() - 10);
        }
        println!();
    }

    println!("Model loading infrastructure is ready!");
    println!();
    println!("Next steps to complete model loading:");
    println!("  1. Extract ModelConfig from metadata (vocab_size, hidden_dim, etc.)");
    println!("  2. Map tensor names to Model layers (see src/layers.rs docs)");
    println!("  3. Load weights into each layer");
    println!();
    println!("See documentation: cargo doc --open");
    println!("Example: src/layers.rs module documentation");

    Ok(())
}

fn load_safetensors_model(file_data: &[u8]) -> Result<()> {
    use realizar::safetensors::SafetensorsModel;

    println!("Parsing Safetensors file...");
    let safetensors = SafetensorsModel::from_bytes(file_data)?;

    println!("✓ Successfully parsed Safetensors file");
    println!();
    println!("Model Information:");
    println!("  Tensors: {}", safetensors.tensors.len());
    println!("  Data size: {} bytes", safetensors.data.len());
    println!();

    // Show tensor names
    if !safetensors.tensors.is_empty() {
        println!("Tensors (first 10):");
        for (name, tensor_info) in safetensors.tensors.iter().take(10) {
            let shape: Vec<String> = tensor_info.shape.iter().map(|s| s.to_string()).collect();
            println!(
                "  - {} [{}, dtype={:?}]",
                name,
                shape.join("×"),
                tensor_info.dtype
            );
        }
        if safetensors.tensors.len() > 10 {
            println!("  ... and {} more", safetensors.tensors.len() - 10);
        }
        println!();
    }

    println!("Model loading infrastructure is ready!");
    println!();
    println!("Next steps to complete model loading:");
    println!("  1. Extract ModelConfig from tensor shapes");
    println!("  2. Map tensor names to Model layers (see src/layers.rs docs)");
    println!("  3. Load weights into each layer");
    println!();
    println!("See documentation: cargo doc --open");
    println!("Example: src/layers.rs module documentation");

    Ok(())
}

fn run_visualization(use_color: bool, samples: usize) {
    use realizar::viz::{
        print_benchmark_results, render_ascii_histogram, render_sparkline, BenchmarkData,
    };

    println!("Realizar Benchmark Visualization Demo");
    println!("=====================================");
    println!();

    // Generate synthetic benchmark data (simulating inference latencies)
    let mut rng_state = 42u64;
    let latencies: Vec<f64> = (0..samples)
        .map(|_| {
            // Simple LCG for reproducible pseudo-random numbers
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let uniform = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            // Log-normal distribution (typical for latencies)
            let log_mean = 3.0; // ~20us median
            let log_std = 0.5;
            (log_mean + log_std * (2.0 * uniform - 1.0)).exp()
        })
        .collect();

    // Demo 1: Sparkline
    println!("1. Sparkline (latency trend)");
    println!("   {}", render_sparkline(&latencies, 60));
    println!();

    // Demo 2: ASCII histogram
    println!("2. ASCII Histogram (latency distribution)");
    let hist = render_ascii_histogram(&latencies, 12, 50);
    for line in hist.lines() {
        println!("   {line}");
    }
    println!();

    // Demo 3: Full benchmark report
    println!("3. Full Benchmark Report");
    let data = BenchmarkData::new("inference_latency", latencies);
    print_benchmark_results(&data, use_color);
    println!();

    // Demo 4: Multi-benchmark comparison
    println!("4. Multi-Benchmark Comparison");
    println!();

    let benchmarks = [
        ("tensor_add", 15.2, 18.1),
        ("tensor_mul", 16.8, 20.3),
        ("matmul_128", 145.3, 172.1),
        ("softmax", 23.4, 28.9),
        ("attention", 892.1, 1024.5),
    ];

    println!(
        "   {:.<20} {:>10} {:>10} {:>10}",
        "Benchmark", "p50 (us)", "p99 (us)", "Trend"
    );
    println!("   {}", "-".repeat(55));

    for (name, p50, p99) in benchmarks {
        // Generate mini trend data
        let trend: Vec<f64> = (0..20)
            .map(|i| p50 + (i as f64 / 20.0) * (p99 - p50) * 0.3)
            .collect();
        let sparkline = render_sparkline(&trend, 10);
        println!("   {:.<20} {:>10.1} {:>10.1} {}", name, p50, p99, sparkline);
    }
    println!();

    println!("Visualization powered by trueno-viz");
}

/// Available benchmark suites
const BENCHMARK_SUITES: &[(&str, &str)] = &[
    (
        "tensor_ops",
        "Core tensor operations (add, mul, matmul, softmax)",
    ),
    ("inference", "End-to-end inference pipeline benchmarks"),
    ("cache", "KV cache operations and memory management"),
    ("tokenizer", "BPE and SentencePiece tokenization"),
    ("quantize", "Quantization/dequantization (Q4_0, Q8_0)"),
    ("lambda", "AWS Lambda cold start and warm invocation"),
    (
        "comparative",
        "Framework comparison (MNIST, CIFAR-10, Iris)",
    ),
];

fn run_benchmarks(suite: Option<String>, list: bool) -> Result<()> {
    if list {
        println!("Available benchmark suites:");
        println!();
        for (name, description) in BENCHMARK_SUITES {
            println!("  {name:<12} - {description}");
        }
        println!();
        println!("Usage:");
        println!("  realizar bench              # Run all benchmarks");
        println!("  realizar bench tensor_ops   # Run specific suite");
        println!("  realizar bench --list       # List available suites");
        return Ok(());
    }

    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("bench");

    if let Some(ref suite_name) = suite {
        // Validate suite name
        if !BENCHMARK_SUITES.iter().any(|(name, _)| *name == suite_name) {
            eprintln!("Error: Unknown benchmark suite '{suite_name}'");
            eprintln!();
            eprintln!("Available suites:");
            for (name, _) in BENCHMARK_SUITES {
                eprintln!("  {name}");
            }
            std::process::exit(1);
        }
        cmd.arg("--bench").arg(suite_name);
    }

    println!("Running benchmarks...");
    println!();

    let status =
        cmd.status()
            .map_err(|e| realizar::error::RealizarError::UnsupportedOperation {
                operation: "run_benchmarks".to_string(),
                reason: format!("Failed to execute cargo bench: {e}"),
            })?;

    if !status.success() {
        return Err(realizar::error::RealizarError::UnsupportedOperation {
            operation: "run_benchmarks".to_string(),
            reason: format!("Benchmarks failed with exit code: {:?}", status.code()),
        });
    }

    Ok(())
}

// ============================================================================
// New ollama-like Commands
// ============================================================================

/// Run a model for inference (like `ollama run`) - with registry support
#[cfg(feature = "registry")]
async fn run_model(
    model_ref: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    format: &str,
) -> Result<()> {
    println!("Loading model: {model_ref}");

    // Try to resolve via Pacha URI
    let file_data = match ModelUri::parse(model_ref) {
        Ok(uri) => {
            // Use Pacha resolver
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
                        }
                        ModelSource::PachaLocal { name, version } => {
                            println!("  Source: Pacha registry ({name}:{version})");
                        }
                        ModelSource::PachaRemote { host, name, version } => {
                            println!("  Source: Remote registry {host} ({name}:{version})");
                        }
                        ModelSource::HuggingFace { repo_id, revision } => {
                            let rev = revision.as_deref().unwrap_or("main");
                            println!("  Source: HuggingFace ({repo_id}@{rev})");
                        }
                    }
                    resolved.data
                }
                Err(e) => {
                    // Fall back to direct file read for local paths
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
                }
            }
        }
        Err(_) => {
            // Direct file path
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
        }
    };

    // Detect and display model info
    display_model_info(model_ref, &file_data)?;

    println!();

    // Handle prompt
    if let Some(prompt_text) = prompt {
        println!("Prompt: {prompt_text}");
        println!("Max tokens: {max_tokens}");
        println!("Temperature: {temperature}");
        println!("Format: {format}");
        println!();
        println!("Model loaded ({} bytes) - ready for inference!", file_data.len());
    } else {
        // Interactive mode
        println!("Interactive mode (Ctrl+D to exit)");
        println!();
        println!("Model loaded ({} bytes)", file_data.len());
        println!("Use a prompt argument:");
        println!("  realizar run {model_ref} \"Your prompt here\"");
    }

    Ok(())
}

/// Run a model for inference (like `ollama run`) - without registry
#[cfg(not(feature = "registry"))]
async fn run_model(
    model_ref: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    format: &str,
) -> Result<()> {
    println!("Loading model: {model_ref}");

    // Parse model URI
    let is_local_file = model_ref.starts_with("./")
        || model_ref.starts_with('/')
        || model_ref.ends_with(".gguf")
        || model_ref.ends_with(".safetensors")
        || model_ref.ends_with(".apr");

    if is_local_file {
        // Load from file
        if !std::path::Path::new(model_ref).exists() {
            return Err(realizar::error::RealizarError::ModelNotFound(
                model_ref.to_string(),
            ));
        }
        println!("  Source: local file");
    } else if model_ref.starts_with("pacha://") || model_ref.contains(':') {
        // Load from Pacha registry
        println!("  Source: Pacha registry");
        println!();
        println!("Enable registry support: --features registry");
        println!("Or use a local file path:");
        println!("  realizar run ./model.gguf \"Your prompt\"");
        return Ok(());
    } else if model_ref.starts_with("hf://") {
        // HuggingFace Hub
        println!("  Source: HuggingFace Hub");
        println!();
        println!("Enable registry support: --features registry");
        println!("Or download manually and use:");
        println!("  realizar run ./model.gguf \"Your prompt\"");
        return Ok(());
    }

    // Read and parse model file
    let file_data = std::fs::read(model_ref).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "read_model".to_string(),
            reason: format!("Failed to read {model_ref}: {e}"),
        }
    })?;

    // Detect and display model info
    display_model_info(model_ref, &file_data)?;

    println!();

    // Handle prompt
    if let Some(prompt_text) = prompt {
        println!("Prompt: {prompt_text}");
        println!("Max tokens: {max_tokens}");
        println!("Temperature: {temperature}");
        println!("Format: {format}");
        println!();
        println!("Model loaded ({} bytes) - ready for inference!", file_data.len());
    } else {
        // Interactive mode
        println!("Interactive mode (Ctrl+D to exit)");
        println!();
        println!("Model loaded ({} bytes)", file_data.len());
        println!("Use a prompt argument:");
        println!("  realizar run {model_ref} \"Your prompt here\"");
    }

    Ok(())
}

/// Display model information based on file type
fn display_model_info(model_ref: &str, file_data: &[u8]) -> Result<()> {
    if model_ref.ends_with(".gguf") || file_data.starts_with(b"GGUF") {
        use realizar::gguf::GGUFModel;
        let gguf = GGUFModel::from_bytes(file_data)?;
        println!("  Format: GGUF v{}", gguf.header.version);
        println!("  Tensors: {}", gguf.header.tensor_count);
    } else if model_ref.ends_with(".safetensors") {
        use realizar::safetensors::SafetensorsModel;
        let st = SafetensorsModel::from_bytes(file_data)?;
        println!("  Format: SafeTensors");
        println!("  Tensors: {}", st.tensors.len());
    } else {
        println!("  Format: Unknown ({} bytes)", file_data.len());
    }
    Ok(())
}

/// Run interactive chat mode (like `ollama chat`) - with registry support
#[cfg(feature = "registry")]
async fn run_chat(
    model_ref: &str,
    system_prompt: Option<&str>,
    history_file: Option<&str>,
) -> Result<()> {
    use std::io::{BufRead, Write};

    println!("Loading model: {model_ref}");

    // Try to resolve via Pacha URI
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
                }
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
                }
            }
        }
        Err(_) => {
            if !std::path::Path::new(model_ref).exists() {
                return Err(realizar::error::RealizarError::ModelNotFound(model_ref.to_string()));
            }
            std::fs::read(model_ref).map_err(|e| {
                realizar::error::RealizarError::UnsupportedOperation {
                    operation: "read_model".to_string(),
                    reason: format!("Failed to read: {e}"),
                }
            })?
        }
    };

    display_model_info(model_ref, &file_data)?;
    println!("  Size: {} bytes", file_data.len());
    println!();

    // Load history if provided
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
                // EOF (Ctrl+D)
                println!();
                break;
            }
            Ok(_) => {
                let input = input.trim();

                if input.is_empty() {
                    continue;
                }

                // Handle commands
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

                // Simulate response (model inference would go here)
                let response = format!(
                    "[Model loaded: {} bytes] Echo: {}",
                    file_data.len(),
                    input
                );

                println!();
                println!("{response}");
                println!();

                // Add to history
                history.push((input.to_string(), response));
            }
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        }
    }

    // Save history if path provided
    if let Some(path) = history_file {
        if let Ok(json) = serde_json::to_string_pretty(&history) {
            let _ = std::fs::write(path, json);
            println!("History saved to {path}");
        }
    }

    println!("Goodbye!");
    Ok(())
}

/// Run interactive chat mode (like `ollama chat`) - without registry
#[cfg(not(feature = "registry"))]
async fn run_chat(
    model_ref: &str,
    system_prompt: Option<&str>,
    history_file: Option<&str>,
) -> Result<()> {
    use std::io::{BufRead, Write};

    println!("Loading model: {model_ref}");

    // Check file exists
    if !std::path::Path::new(model_ref).exists()
        && !model_ref.starts_with("pacha://")
        && !model_ref.starts_with("hf://")
    {
        return Err(realizar::error::RealizarError::ModelNotFound(model_ref.to_string()));
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

    display_model_info(model_ref, &file_data)?;
    println!("  Size: {} bytes", file_data.len());
    println!();

    // Load history if provided
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
            }
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

                let response = format!(
                    "[Model loaded: {} bytes] Echo: {}",
                    file_data.len(),
                    input
                );

                println!();
                println!("{response}");
                println!();

                history.push((input.to_string(), response));
            }
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
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

/// List available models (like `ollama list`)
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

    // Use Pacha resolver
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
        }
    };

    if !resolver.has_registry() {
        println!("No Pacha registry found.");
        println!();
        println!("Initialize registry:");
        println!("  pacha init");
        return Ok(());
    }

    // List models from registry
    let models = match resolver.list_models() {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to list models: {e}");
            return Ok(());
        }
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
                println!("{}", serde_json::to_string_pretty(&json_models).unwrap_or_default());
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

/// List available models (like `ollama list`) - fallback without registry feature
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

    // List local models from ~/.pacha/
    let pacha_dir = dirs::home_dir()
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

    // Scan for models
    let mut models_found = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&pacha_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.ends_with(".gguf") || name.ends_with(".safetensors") || name.ends_with(".apr") {
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
                            "size_human": format_size(*size)
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&json_models).unwrap_or_default());
            },
            _ => {
                println!("{:<40} {:>12}", "NAME", "SIZE");
                println!("{}", "-".repeat(54));
                for (name, size) in &models_found {
                    println!("{:<40} {:>12}", name, format_size(*size));
                }
            },
        }
    }

    Ok(())
}

/// Format file size in human-readable form
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Pull a model from registry (like `ollama pull`) - with registry support
#[cfg(feature = "registry")]
async fn pull_model(
    model_ref: &str,
    force: bool,
    quantize: Option<&str>,
) -> Result<()> {
    println!("Pulling model: {model_ref}");
    if force {
        println!("  Force: re-downloading even if cached");
    }
    if let Some(q) = quantize {
        println!("  Quantize: {q}");
    }
    println!();

    // Parse URI
    let uri = ModelUri::parse(model_ref).map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "parse_uri".to_string(),
            reason: format!("Invalid model reference: {e}"),
        }
    })?;

    // Initialize resolver
    let resolver = ModelResolver::new_default().map_err(|e| {
        realizar::error::RealizarError::UnsupportedOperation {
            operation: "init_resolver".to_string(),
            reason: format!("Failed to initialize Pacha resolver: {e}"),
        }
    })?;

    // Check if already cached (unless force)
    if !force && resolver.exists(&uri) {
        println!("Model already cached locally.");
        println!("Use --force to re-download.");
        return Ok(());
    }

    // Resolve (pull) the model
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
        }
        ModelSource::PachaLocal { name, version } => {
            println!("  Source: Pacha local ({name}:{version})");
        }
        ModelSource::PachaRemote { host, name, version } => {
            println!("  Source: Remote {host} ({name}:{version})");
            println!("  Cached to local registry.");
        }
        ModelSource::HuggingFace { repo_id, revision } => {
            let rev = revision.as_deref().unwrap_or("main");
            println!("  Source: HuggingFace ({repo_id}@{rev})");
        }
    }

    println!();
    println!("Model ready! Run with:");
    println!("  realizar run {model_ref} \"Your prompt\"");

    Ok(())
}

/// Pull a model from registry (like `ollama pull`) - without registry
#[cfg(not(feature = "registry"))]
async fn pull_model(
    model_ref: &str,
    force: bool,
    quantize: Option<&str>,
) -> Result<()> {
    println!("Pulling model: {model_ref}");
    if force {
        println!("  Force: re-downloading even if cached");
    }
    if let Some(q) = quantize {
        println!("  Quantize: {q}");
    }
    println!();

    // Parse model reference
    if model_ref.starts_with("hf://") {
        let hf_path = &model_ref[5..];
        println!("Source: HuggingFace Hub");
        println!("Model: {hf_path}");
        println!();
        println!("Enable registry support: --features registry");
        println!("Or manual download:");
        println!("  huggingface-cli download {hf_path}");
    } else if model_ref.starts_with("pacha://") {
        let pacha_path = &model_ref[8..];
        println!("Source: Pacha Registry");
        println!("Model: {pacha_path}");
        println!();
        println!("Enable registry support: --features registry");
    } else {
        // Assume short format like "llama3:8b"
        println!("Source: Default registry (Pacha)");
        println!("Model: {model_ref}");
        println!();
        println!("Enable registry support: --features registry");
        println!("Or download manually and use:");
        println!("  realizar run ./downloaded-model.gguf \"prompt\"");
    }

    Ok(())
}

/// Push a model to registry (like `ollama push`) - with registry support
#[cfg(feature = "registry")]
async fn push_model(model_ref: &str, target: Option<&str>) -> Result<()> {
    use pacha::Registry;

    println!("Pushing model: {model_ref}");

    // Parse model reference (name:version)
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

        // Check if local file exists
        let local_path = format!("{name}.gguf");
        if !std::path::Path::new(&local_path).exists() {
            println!("Local file not found: {local_path}");
            println!();
            println!("To push a model to registry:");
            println!("  1. Have the model file: {name}.gguf");
            println!("  2. Run: realizar push {name}:{version_str}");
            return Ok(());
        }

        // Read model data
        let data = std::fs::read(&local_path).map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "read_model".to_string(),
                reason: format!("Failed to read {local_path}: {e}"),
            }
        })?;

        // Open registry
        let registry = Registry::open_default().map_err(|e| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "open_registry".to_string(),
                reason: format!("Failed to open Pacha registry: {e}"),
            }
        })?;

        // Parse version
        let version = parse_model_version(version_str)?;

        // Register model
        let card = pacha::model::ModelCard::new(&format!("Model {name} pushed via realizar"));
        registry
            .register_model(name, &version, &data, card)
            .map_err(|e| {
                realizar::error::RealizarError::UnsupportedOperation {
                    operation: "register_model".to_string(),
                    reason: format!("Failed to register model: {e}"),
                }
            })?;

        println!("Model registered successfully!");
        println!();
        println!("Run with:");
        println!("  realizar run pacha://{name}:{version_str} \"Your prompt\"");
    }

    Ok(())
}

/// Push a model to registry (like `ollama push`) - without registry
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

/// Parse a version string into ModelVersion
#[cfg(feature = "registry")]
fn parse_model_version(s: &str) -> Result<pacha::model::ModelVersion> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() == 3 {
        let major: u32 = parts[0].parse().map_err(|_| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "parse_version".to_string(),
                reason: format!("Invalid version: {s}"),
            }
        })?;
        let minor: u32 = parts[1].parse().map_err(|_| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "parse_version".to_string(),
                reason: format!("Invalid version: {s}"),
            }
        })?;
        let patch: u32 = parts[2].parse().map_err(|_| {
            realizar::error::RealizarError::UnsupportedOperation {
                operation: "parse_version".to_string(),
                reason: format!("Invalid version: {s}"),
            }
        })?;
        return Ok(pacha::model::ModelVersion::new(major, minor, patch));
    }

    // Single number or "latest" -> 1.0.0
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

// Simple home directory resolution (avoids extra dependency)
mod dirs {
    pub(crate) fn home_dir() -> Option<std::path::PathBuf> {
        std::env::var_os("HOME").map(std::path::PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Run Command Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cli_parsing_run_with_prompt() {
        let cli = Cli::parse_from(["realizar", "run", "llama3:8b", "Hello world"]);
        match cli.command {
            Commands::Run {
                model,
                prompt,
                max_tokens,
                temperature,
                format,
            } => {
                assert_eq!(model, "llama3:8b");
                assert_eq!(prompt, Some("Hello world".to_string()));
                assert_eq!(max_tokens, 256); // default
                assert!((temperature - 0.7).abs() < 0.01); // default
                assert_eq!(format, "text"); // default
            },
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_parsing_run_without_prompt() {
        let cli = Cli::parse_from(["realizar", "run", "llama3:8b"]);
        match cli.command {
            Commands::Run { model, prompt, .. } => {
                assert_eq!(model, "llama3:8b");
                assert!(prompt.is_none());
            },
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_parsing_run_with_options() {
        let cli = Cli::parse_from([
            "realizar",
            "run",
            "llama3:8b",
            "Hello",
            "-n",
            "100",
            "-t",
            "0.5",
            "-f",
            "json",
        ]);
        match cli.command {
            Commands::Run {
                max_tokens,
                temperature,
                format,
                ..
            } => {
                assert_eq!(max_tokens, 100);
                assert!((temperature - 0.5).abs() < 0.01);
                assert_eq!(format, "json");
            },
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_parsing_run_pacha_uri() {
        let cli = Cli::parse_from(["realizar", "run", "pacha://model:v1.0", "test"]);
        match cli.command {
            Commands::Run { model, .. } => {
                assert_eq!(model, "pacha://model:v1.0");
            },
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_parsing_run_file_path() {
        let cli = Cli::parse_from(["realizar", "run", "./model.gguf", "test"]);
        match cli.command {
            Commands::Run { model, .. } => {
                assert_eq!(model, "./model.gguf");
            },
            _ => panic!("Expected Run command"),
        }
    }

    // -------------------------------------------------------------------------
    // Chat Command Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cli_parsing_chat() {
        let cli = Cli::parse_from(["realizar", "chat", "llama3:8b"]);
        match cli.command {
            Commands::Chat {
                model,
                system,
                history,
            } => {
                assert_eq!(model, "llama3:8b");
                assert!(system.is_none());
                assert!(history.is_none());
            },
            _ => panic!("Expected Chat command"),
        }
    }

    #[test]
    fn test_cli_parsing_chat_with_system() {
        let cli = Cli::parse_from([
            "realizar",
            "chat",
            "llama3:8b",
            "-s",
            "You are a helpful assistant.",
        ]);
        match cli.command {
            Commands::Chat { system, .. } => {
                assert_eq!(system, Some("You are a helpful assistant.".to_string()));
            },
            _ => panic!("Expected Chat command"),
        }
    }

    // -------------------------------------------------------------------------
    // List Command Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cli_parsing_list() {
        let cli = Cli::parse_from(["realizar", "list"]);
        match cli.command {
            Commands::List { remote, format } => {
                assert!(remote.is_none());
                assert_eq!(format, "table");
            },
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_parsing_list_with_remote() {
        let cli = Cli::parse_from(["realizar", "list", "-r", "pacha://registry.example.com"]);
        match cli.command {
            Commands::List { remote, .. } => {
                assert_eq!(remote, Some("pacha://registry.example.com".to_string()));
            },
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_parsing_list_json() {
        let cli = Cli::parse_from(["realizar", "list", "-f", "json"]);
        match cli.command {
            Commands::List { format, .. } => {
                assert_eq!(format, "json");
            },
            _ => panic!("Expected List command"),
        }
    }

    // -------------------------------------------------------------------------
    // Pull Command Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cli_parsing_pull() {
        let cli = Cli::parse_from(["realizar", "pull", "llama3:8b"]);
        match cli.command {
            Commands::Pull {
                model,
                force,
                quantize,
            } => {
                assert_eq!(model, "llama3:8b");
                assert!(!force);
                assert!(quantize.is_none());
            },
            _ => panic!("Expected Pull command"),
        }
    }

    #[test]
    fn test_cli_parsing_pull_with_force() {
        let cli = Cli::parse_from(["realizar", "pull", "llama3:8b", "-f"]);
        match cli.command {
            Commands::Pull { force, .. } => {
                assert!(force);
            },
            _ => panic!("Expected Pull command"),
        }
    }

    #[test]
    fn test_cli_parsing_pull_with_quantize() {
        let cli = Cli::parse_from(["realizar", "pull", "llama3:8b", "-q", "q4"]);
        match cli.command {
            Commands::Pull { quantize, .. } => {
                assert_eq!(quantize, Some("q4".to_string()));
            },
            _ => panic!("Expected Pull command"),
        }
    }

    #[test]
    fn test_cli_parsing_pull_hf() {
        let cli = Cli::parse_from(["realizar", "pull", "hf://meta-llama/Llama-3-8B"]);
        match cli.command {
            Commands::Pull { model, .. } => {
                assert_eq!(model, "hf://meta-llama/Llama-3-8B");
            },
            _ => panic!("Expected Pull command"),
        }
    }

    // -------------------------------------------------------------------------
    // Push Command Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cli_parsing_push() {
        let cli = Cli::parse_from(["realizar", "push", "my-model:v1.0"]);
        match cli.command {
            Commands::Push { model, to } => {
                assert_eq!(model, "my-model:v1.0");
                assert!(to.is_none());
            },
            _ => panic!("Expected Push command"),
        }
    }

    #[test]
    fn test_cli_parsing_push_with_target() {
        let cli = Cli::parse_from([
            "realizar",
            "push",
            "my-model:v1.0",
            "--to",
            "pacha://registry.example.com",
        ]);
        match cli.command {
            Commands::Push { to, .. } => {
                assert_eq!(to, Some("pacha://registry.example.com".to_string()));
            },
            _ => panic!("Expected Push command"),
        }
    }

    // -------------------------------------------------------------------------
    // Format Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(7 * 1024 * 1024 * 1024), "7.0 GB");
    }

    // -------------------------------------------------------------------------
    // Original Serve Command Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cli_parsing_serve_demo() {
        let cli = Cli::parse_from(["realizar", "serve", "--demo"]);
        match cli.command {
            Commands::Serve { demo, model, .. } => {
                assert!(demo);
                assert!(model.is_none());
            },
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parsing_serve_with_port() {
        let cli = Cli::parse_from(["realizar", "serve", "--demo", "--port", "9090"]);
        match cli.command {
            Commands::Serve { port, demo, .. } => {
                assert_eq!(port, 9090);
                assert!(demo);
            },
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parsing_serve_with_host() {
        let cli = Cli::parse_from(["realizar", "serve", "--demo", "--host", "0.0.0.0"]);
        match cli.command {
            Commands::Serve { host, demo, .. } => {
                assert_eq!(host, "0.0.0.0");
                assert!(demo);
            },
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parsing_serve_with_model() {
        let cli = Cli::parse_from(["realizar", "serve", "--model", "model.gguf"]);
        match cli.command {
            Commands::Serve { model, demo, .. } => {
                assert_eq!(model, Some("model.gguf".to_string()));
                assert!(!demo);
            },
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_cli_parsing_info() {
        let cli = Cli::parse_from(["realizar", "info"]);
        match cli.command {
            Commands::Info => {},
            _ => panic!("Expected Info command"),
        }
    }

    #[test]
    fn test_default_host_and_port() {
        let cli = Cli::parse_from(["realizar", "serve", "--demo"]);
        match cli.command {
            Commands::Serve { host, port, .. } => {
                assert_eq!(host, "127.0.0.1");
                assert_eq!(port, 8080);
            },
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_version_constant_exists() {
        let version = realizar::VERSION;
        assert!(!version.is_empty());
        assert!(version.starts_with("0."));
    }

    #[test]
    fn test_cli_parsing_bench_list() {
        let cli = Cli::parse_from(["realizar", "bench", "--list"]);
        match cli.command {
            Commands::Bench { list, suite } => {
                assert!(list);
                assert!(suite.is_none());
            },
            _ => panic!("Expected Bench command"),
        }
    }

    #[test]
    fn test_cli_parsing_bench_with_suite() {
        let cli = Cli::parse_from(["realizar", "bench", "tensor_ops"]);
        match cli.command {
            Commands::Bench { suite, list } => {
                assert_eq!(suite, Some("tensor_ops".to_string()));
                assert!(!list);
            },
            _ => panic!("Expected Bench command"),
        }
    }

    #[test]
    fn test_cli_parsing_bench_all() {
        let cli = Cli::parse_from(["realizar", "bench"]);
        match cli.command {
            Commands::Bench { suite, list } => {
                assert!(suite.is_none());
                assert!(!list);
            },
            _ => panic!("Expected Bench command"),
        }
    }

    #[test]
    fn test_benchmark_suites_not_empty() {
        assert!(!BENCHMARK_SUITES.is_empty());
        assert!(BENCHMARK_SUITES.len() >= 5);
    }

    #[test]
    fn test_benchmark_suites_have_descriptions() {
        for (name, description) in BENCHMARK_SUITES {
            assert!(!name.is_empty(), "Benchmark name should not be empty");
            assert!(
                !description.is_empty(),
                "Benchmark description should not be empty"
            );
        }
    }

    #[test]
    fn test_cli_parsing_viz_default() {
        let cli = Cli::parse_from(["realizar", "viz"]);
        match cli.command {
            Commands::Viz { color, samples } => {
                assert!(!color);
                assert_eq!(samples, 100);
            },
            _ => panic!("Expected Viz command"),
        }
    }

    #[test]
    fn test_cli_parsing_viz_with_color() {
        let cli = Cli::parse_from(["realizar", "viz", "--color"]);
        match cli.command {
            Commands::Viz { color, samples } => {
                assert!(color);
                assert_eq!(samples, 100);
            },
            _ => panic!("Expected Viz command"),
        }
    }

    #[test]
    fn test_cli_parsing_viz_with_samples() {
        let cli = Cli::parse_from(["realizar", "viz", "--samples", "500"]);
        match cli.command {
            Commands::Viz { color, samples } => {
                assert!(!color);
                assert_eq!(samples, 500);
            },
            _ => panic!("Expected Viz command"),
        }
    }
}
