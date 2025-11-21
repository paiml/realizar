# Realizar Examples

This directory contains example programs demonstrating the capabilities of realizar, a pure Rust ML inference engine.

## Available Examples (5 total)

### 1. `inference.rs` - End-to-End Text Generation

Demonstrates the complete text generation pipeline:
- Model initialization with configuration
- Forward pass through transformer blocks  
- Text generation with various sampling strategies (greedy, top-k, top-p)

**Run:**
```bash
cargo run --example inference
```

### 2. `api_server.rs` - HTTP API Server

Shows how to deploy realizar as a REST API service:
- Create a demo model
- Start the HTTP server (default: http://127.0.0.1:3000)
- Handle tokenization, generation, and batch requests

**Run:**
```bash
cargo run --example api_server
# Server runs at http://127.0.0.1:3000
# Test: curl http://127.0.0.1:3000/health
```

### 3. `tokenization.rs` - Tokenizer Comparison

Compares different tokenization strategies:
- Basic tokenizer (vocabulary-based)
- BPE (Byte Pair Encoding) tokenizer
- SentencePiece tokenizer
- Encoding and decoding workflows

**Run:**
```bash
cargo run --example tokenization
```

### 4. `safetensors_loading.rs` - Model Loading

Demonstrates SafeTensors file format support:
- Load SafeTensors files (aprender compatibility)
- Extract tensor data using helper API
- Inspect model structure and metadata
- Interoperability with aprender-trained models

**Run:**
```bash
cargo run --example safetensors_loading
```

### 5. `model_cache.rs` - Model Caching

Demonstrates ModelCache for efficient model reuse:
- Cache creation with capacity limits
- Model loading with cache hits/misses
- Metrics tracking (hit rate, evictions)
- LRU eviction behavior
- Config-based cache keys

**Run:**
```bash
cargo run --example model_cache
```

## Requirements

All examples use the default features. Some examples (api_server) require the `server` feature:

```toml
[features]
default = ["server", "cli", "gpu"]
```

## Quick Start

Build all examples:
```bash
cargo build --examples
```

Run a specific example:
```bash
cargo run --example <name>
```

## Notes

- Examples use demo/synthetic models for demonstration
- Real model loading requires proper GGUF or SafeTensors files
- API server example runs indefinitely (Ctrl+C to stop)
- All examples follow EXTREME TDD principles with comprehensive testing
