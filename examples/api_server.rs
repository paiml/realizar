//! Example: Running the Realizar HTTP API server
//!
//! This example demonstrates how to:
//! 1. Create a demo model
//! 2. Start the HTTP server
//! 3. Make API requests
//!
//! Run with: cargo run --example api_server

use anyhow::Result;
use realizar::api::{create_router, AppState};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Realizar API Server Example ===\n");

    // Create demo model and tokenizer
    println!("Creating demo model...");
    let state = AppState::demo()?;
    println!("  ✓ Model created\n");

    // Create router
    let app = create_router(state);

    // Bind to localhost
    let addr: SocketAddr = "127.0.0.1:3000".parse()?;
    println!("Starting server on http://{}\n", addr);

    println!("Available endpoints:");
    println!("  GET  /health   - Health check");
    println!("  POST /tokenize - Tokenize text");
    println!("  POST /generate - Generate text\n");

    println!("Example requests:");
    println!("  curl http://127.0.0.1:3000/health");
    println!("  curl -X POST http://127.0.0.1:3000/tokenize \\");
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"text\": \"token1 token2\"}}'");
    println!("  curl -X POST http://127.0.0.1:3000/generate \\");
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"prompt\": \"token1\", \"max_tokens\": 5, \"strategy\": \"greedy\"}}'\n");

    println!("Server running... (Press Ctrl+C to stop)\n");

    // Create listener and serve
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
