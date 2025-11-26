# Realizar Examples

This directory contains example programs demonstrating the capabilities of realizar, a pure Rust ML inference engine.

## Available Examples (9 total)

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

### 6. `gguf_loading.rs` - GGUF Format Loading

Demonstrates GGUF file format support (llama.cpp/Ollama):
- Load GGUF files (binary format parsing)
- Parse header, metadata, and tensor information
- Inspect model structure (dimensions, quantization types)
- Extract and dequantize tensor data (F32, Q4_0)
- Compatible with llama.cpp and Ollama models

**Run:**
```bash
cargo run --example gguf_loading
```

### 7. `wine_lambda.rs` - AWS Lambda Wine Quality Predictor

Production-ready wine quality rating predictor for AWS Lambda deployment.
Inspired by [paiml/wine-ratings](https://github.com/paiml/wine-ratings).

Features:
- Predicts wine quality (0-10) from 11 physicochemical properties
- Sub-millisecond inference latency
- Cold start detection and metrics
- Prometheus metrics export
- Drift detection for production monitoring
- Ready for ARM64 Graviton deployment

**Run:**
```bash
cargo run --example wine_lambda
```

**Deploy to Lambda:**
```bash
# Build for ARM64
cargo build --release --target aarch64-unknown-linux-gnu --features lambda

# Package and deploy
cp target/aarch64-unknown-linux-gnu/release/wine_lambda bootstrap
zip wine_lambda.zip bootstrap
aws lambda create-function --function-name wine-predictor --runtime provided.al2 \
  --architecture arm64 --handler bootstrap --zip-file fileb://wine_lambda.zip
```

### 8. `data_pipeline.rs` - Alimentar Data Pipeline Integration

End-to-end ML data pipeline demonstrating alimentar + realizar integration:
- Load built-in Iris dataset (embedded, no download)
- Data quality checks
- Transform pipeline (shuffle)
- Train/test split
- Inference with classification
- Drift detection
- DataLoader batching

**Run:**
```bash
cargo run --example data_pipeline --features alimentar-data
```

### 9. `train_model.rs` - Train Real .apr Models

Train actual ML models with aprender and save as .apr format:
- Wine quality linear regression (R² ~0.93)
- Save to .apr binary format
- Load and verify predictions
- Model coefficients extraction

**Run:**
```bash
cargo run --example train_model --features "aprender-serve"
```

**Output:**
- `wine_regressor.apr` (141 bytes, gitignored)

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
