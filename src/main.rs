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
    /// Show version and configuration info
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { host, port, model, demo } => {
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
            println!("  - {} [{}, qtype={}]", tensor.name, dims.join("×"), tensor.qtype);
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
            println!("  - {} [{}, dtype={:?}]", name, shape.join("×"), tensor_info.dtype);
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
}
