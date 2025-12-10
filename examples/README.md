# Realizar Examples

This directory contains example programs demonstrating the capabilities of realizar, a pure Rust ML inference engine.

## Available Examples (14 total)

### Core Examples (No Extra Features Required)

#### 1. `inference.rs` - End-to-End Text Generation

Demonstrates the complete text generation pipeline:
- Model initialization with configuration
- Forward pass through transformer blocks
- Text generation with various sampling strategies (greedy, top-k, top-p)

**Run:**
```bash
cargo run --example inference
```

#### 2. `api_server.rs` - HTTP API Server

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

#### 3. `tokenization.rs` - Tokenizer Comparison

Compares different tokenization strategies:
- Basic tokenizer (vocabulary-based)
- BPE (Byte Pair Encoding) tokenizer
- SentencePiece tokenizer
- Encoding and decoding workflows

**Run:**
```bash
cargo run --example tokenization
```

#### 4. `safetensors_loading.rs` - SafeTensors Model Loading

Demonstrates SafeTensors file format support:
- Load SafeTensors files (aprender compatibility)
- Extract tensor data using helper API
- Inspect model structure and metadata
- Interoperability with aprender-trained models

**Run:**
```bash
cargo run --example safetensors_loading
```

#### 5. `model_cache.rs` - Model Caching with LRU

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

#### 6. `gguf_loading.rs` - GGUF Format Loading

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

#### 7. `apr_loading.rs` - APR Format Loading (Sovereign Stack)

Demonstrates Aprender's native .apr format - the PRIMARY inference format
for the sovereign AI stack:
- APR format specification (magic, header, flags)
- All supported model types (Linear, NN, MoE, etc.)
- Header parsing and format detection
- Inference with synthetic models
- Batch prediction

**Run:**
```bash
cargo run --example apr_loading
```

#### 8. `observability_demo.rs` - Distributed Tracing & Metrics

Demonstrates the observability stack:
- OpenTelemetry-style tracing with span creation
- W3C TraceContext propagation (traceparent header)
- Prometheus metrics export
- A/B testing with variant tracking
- Custom span attributes and events

**Run:**
```bash
cargo run --example observability_demo
```

### Examples Requiring Features

#### 9. `wine_lambda.rs` - AWS Lambda Wine Quality Predictor

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

#### 10. `data_pipeline.rs` - Alimentar Data Pipeline Integration

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

#### 11. `train_model.rs` - Train Real .apr Models

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

#### 12. `build_mnist_model.rs` - Build MNIST Classifier

Build and save an MNIST digit classifier:
- Create neural network architecture
- Save as .apr format for deployment
- Compatible with serve_mnist example

**Run:**
```bash
cargo run --example build_mnist_model --features "aprender-serve"
```

#### 13. `serve_mnist.rs` - Serve MNIST Model via HTTP

Serve an MNIST classifier via HTTP API:
- Load pre-built MNIST model
- REST API for digit classification
- Batch prediction support

**Run:**
```bash
cargo run --example serve_mnist --features "aprender-serve"
```

#### 14. `mnist_apr_benchmark.rs` - MNIST APR vs PyTorch Benchmark

Benchmark .apr format inference against PyTorch baseline:
- Cold start latency comparison
- Inference throughput measurement
- Memory usage analysis
- Pareto frontier visualization data

**Run:**
```bash
cargo run --example mnist_apr_benchmark --features "aprender-serve"
```

## Quick Reference

| Example | Features Required | Description |
|---------|------------------|-------------|
| `inference` | None | Text generation with sampling |
| `api_server` | None | HTTP REST API server |
| `tokenization` | None | BPE/SentencePiece comparison |
| `safetensors_loading` | None | Load .safetensors files |
| `model_cache` | None | LRU caching demo |
| `gguf_loading` | None | Load llama.cpp models |
| `apr_loading` | None | Load .apr models |
| `observability_demo` | None | Tracing & metrics |
| `wine_lambda` | None | AWS Lambda predictor |
| `data_pipeline` | `alimentar-data` | End-to-end ML pipeline |
| `train_model` | `aprender-serve` | Train & save models |
| `build_mnist_model` | `aprender-serve` | Build MNIST classifier |
| `serve_mnist` | `aprender-serve` | Serve MNIST via HTTP |
| `mnist_apr_benchmark` | `aprender-serve` | Benchmark .apr vs PyTorch |

## Quick Start

Build all examples:
```bash
cargo build --examples
```

Run a specific example:
```bash
cargo run --example <name>
```

Run with features:
```bash
cargo run --example <name> --features "<feature-list>"
```

## Notes

- Examples use demo/synthetic models for demonstration
- Real model loading requires proper GGUF or SafeTensors files
- API server examples run indefinitely (Ctrl+C to stop)
- All examples follow EXTREME TDD principles with comprehensive testing
