//! Realizar CLI - Pure Rust ML inference server
//!
//! Run a model inference server or perform single inference.

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use realizar::{
    api::{create_router, AppState},
    error::Result,
};

/// Realizar - Pure Rust ML inference engine
#[derive(Parser)]
#[command(name = "realizar")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
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
        Commands::Serve {
            host,
            port,
            model,
            demo,
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

#[cfg(test)]
mod tests {
    use super::*;

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
