//! Realizar CLI - Pure Rust ML inference server
//!
//! Run a model inference server or perform single inference.

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use realizar::api::{create_router, AppState};
use realizar::error::Result;

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
        Commands::Serve { host, port, demo } => {
            if demo {
                serve_demo(&host, port).await?;
            } else {
                eprintln!("Error: Model loading not yet implemented. Use --demo for testing.");
                std::process::exit(1);
            }
        }
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
        }
    }

    Ok(())
}

async fn serve_demo(host: &str, port: u16) -> Result<()> {
    println!("Starting Realizar inference server (demo mode)...");

    let state = AppState::demo()?;
    let app = create_router(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| realizar::error::RealizarError::InvalidShape {
            reason: format!("Invalid address: {e}"),
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

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| realizar::error::RealizarError::InvalidShape {
            reason: format!("Failed to bind: {e}"),
        })?;

    axum::serve(listener, app)
        .await
        .map_err(|e| realizar::error::RealizarError::InvalidShape {
            reason: format!("Server error: {e}"),
        })?;

    Ok(())
}
