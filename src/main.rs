//! Realizar CLI - Pure Rust ML inference engine
//!
//! Thin entry point that delegates all logic to the library (T-COV-001).

use clap::Parser;

#[tokio::main]
async fn main() -> realizar::error::Result<()> {
    realizar::cli::entrypoint(realizar::cli::Cli::parse()).await
}
