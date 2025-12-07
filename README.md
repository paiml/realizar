<div align="center">

<p align="center">
  <img src=".github/realizar-hero.svg" alt="realizar" width="800">
</p>

<h1 align="center">realizar</h1>

<p align="center">
  <b>Pure Rust ML Inference Engine - 9.6x Faster Than PyTorch</b>
</p>

<p align="center">
  <a href="https://github.com/paiml/realizar/actions"><img src="https://github.com/paiml/realizar/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/realizar"><img src="https://img.shields.io/crates/v/realizar.svg" alt="Crates.io"></a>
  <a href="https://docs.rs/realizar"><img src="https://docs.rs/realizar/badge.svg" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="benches/comparative/BENCHMARK_RESULTS.md"><img src="https://img.shields.io/badge/vs_PyTorch-9.6x_faster-brightgreen.svg" alt="Benchmark"></a>
</p>

</div>

---

**Realizar** is a high-performance machine learning inference engine for serving transformer models in production. Built entirely from scratch in Rust with zero external ML dependencies, it delivers 9.6x faster inference than PyTorch for CPU-only deployments while maintaining 94% test coverage and full GGUF/SafeTensors compatibility.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Benchmark: 9.6x Faster Than PyTorch](#benchmark-96x-faster-than-pytorch)
- [AWS Lambda: 53,000x Faster Cold Start](#aws-lambda-53000x-faster-cold-start)
- [Installation](#installation)
- [Usage](#usage)
- [Feature Flags](#feature-flags)
- [Philosophy](#philosophy)
- [Target API](#target-api)
- [Architecture](#architecture)
- [Examples](#examples)
- [Reproducible Benchmarks](#reproducible-benchmarks)
- [Roadmap](#roadmap)
- [Development](#development)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
# Install from crates.io
cargo install realizar

# Start the demo server
realizar serve --demo --port 8080

# Test inference
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10, "strategy": "greedy"}'
```

For production use with custom models, see the [Installation](#installation) and [Usage](#usage) sections.

## Features

**Realizar** is a production-ready ML inference engine built entirely from scratch in pure Rust:

- **🚀 Blazing Fast**: 9.6x faster than PyTorch for CPU-only inference with 0.52µs latency
- **📦 Zero Dependencies**: No external ML libraries - 100% custom implementation using Trueno
- **🔧 Multiple Model Formats**: Native GGUF and SafeTensors parsers built from scratch
- **⚡ Advanced Quantization**: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K support for reduced memory footprint
- **🎯 Production Ready**: REST API server, streaming responses, model caching, Prometheus metrics
- **☁️ Serverless Optimized**: 53,000x faster cold starts for AWS Lambda deployments
- **🔌 Swappable Backends**: Modular HTTP server design (axum default, hyper/actix-web ready)
- **🧪 Extreme Testing**: 260+ tests with 94.61% coverage, property-based and mutation testing

## 📊 Benchmark: 9.6x Faster Than PyTorch

<p align="center">
  <img src="docs/assets/benchmark-comparison.svg" alt="Aprender vs PyTorch Benchmark" width="600">
</p>

For **CPU-only, single-request inference** (AWS Lambda, edge, embedded):

| Metric | Aprender (Rust) | PyTorch (Python) | Winner |
|--------|-----------------|------------------|--------|
| **Inference Latency (p50)** | 0.52 µs | 5.00 µs | **9.6x faster** |
| **Throughput** | 1,898,614/sec | 195,754/sec | **9.7x higher** |
| **Cold Start** | ~5 ms | ~500 ms+ | **100x faster** |
| **Package Size** | ~5 MB | ~500 MB+ | **100x smaller** |
| **Lambda Memory** | 128 MB | 512 MB+ | **4x less** |

**Statistical Validation:** p < 0.001, Cohen's d = 5.19 (large effect), 10,000 iterations

<details>
<summary>📈 Reproduce the benchmark</summary>

```bash
# Run Aprender benchmark
cargo run --example mnist_apr_benchmark --release --features aprender-serve

# Run PyTorch benchmark
cd benches/comparative
uv sync
uv run mnist_benchmark.py

# Generate comparison report
uv run compare_mnist.py
```

See [BENCHMARK_RESULTS.md](benches/comparative/BENCHMARK_RESULTS.md) for full methodology.

</details>

### Why 9.6x Faster?

```
PyTorch (5.00 µs):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Python  │ Bridge  │ Checks  │ COMPUTE │ Alloc   │ Return  │
│ interp  │ FFI     │ dispatch│ (real)  │ tensor  │ to Py   │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                              ↑ Only 10% is actual work

Aprender (0.52 µs):
┌───────────┬───┐
│  COMPUTE  │ret│  ← 77% is actual work
└───────────┴───┘
```

**Bottom line:** For `.apr` models on Lambda/edge, Aprender eliminates Python entirely—faster, smaller, cheaper.

## AWS Lambda: 53,000x Faster Cold Start

<p align="center">
  <img src="docs/assets/lambda-apr-vs-pytorch.svg" alt="Lambda APR vs PyTorch" width="600">
</p>

For serverless deployment, the `.apr` format **dominates** PyTorch:

| Metric | .apr (Rust) | PyTorch | Improvement |
|--------|-------------|---------|-------------|
| **Cold Start** | 15µs | 800ms | **53,000x faster** |
| **Inference** | 0.6µs | 5.0µs | **8.5x faster** |
| **Binary Size** | 3.2KB | >100MB | **30,000x smaller** |
| **Lambda Memory** | 128MB | 512MB+ | **4x less** |

### 100% Reproducible Lambda Deployment

The model file is **checked into git** for byte-for-byte reproducibility:

```bash
# Model is already in the repo
ls -la models/mnist_784x2.apr  # 3,248 bytes

# Build Lambda binary (uses checked-in model)
make lambda-build

# Package for AWS
make lambda-package

# Run locally
make lambda-bench
```

See the [Lambda MNIST Benchmark](book/book/lambda/mnist-benchmark.html) chapter for full details.

<details>
<summary>📣 Share on LinkedIn</summary>

Copy-paste for LinkedIn:

---

**We benchmarked Rust vs Python for ML inference. The results: 9.6x faster.**

For CPU-only, single-request inference (AWS Lambda, edge devices):

- Latency: 0.52µs (Rust) vs 5.0µs (Python) — **9.6x faster**
- Cold start: 5ms vs 500ms+ — **100x faster**
- Package: 5MB vs 500MB — **100x smaller**
- Lambda RAM: 128MB vs 512MB — **4x less**

Why? Python's interpreter + FFI bridge overhead dominates small operations. 90% of PyTorch inference time is overhead, only 10% is actual compute.

Statistically validated: p < 0.001, Cohen's d = 5.19, 10,000 iterations, 100-point QA checklist.

Full methodology + reproducible benchmark: github.com/paiml/realizar

#MachineLearning #Rust #Python #AWS #Lambda #Performance #MLOps

---

</details>

## Installation

```bash
# From crates.io
cargo install realizar

# From source
git clone https://github.com/paiml/realizar
cd realizar
cargo install --path .
```

## Usage

```bash
# Build the binary
cargo build --release

# Start the inference server (demo mode)
./target/release/realizar serve --demo --port 8080

# Test the API
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10, "strategy": "greedy"}'

# View help
./target/release/realizar --help
./target/release/realizar serve --help
```

## ⚙️ Feature Flags

Realizar supports modular compilation through feature flags:

```toml
[dependencies]
realizar = { version = "0.1", default-features = false, features = ["minimal"] }
```

**Available Features:**
- `default` = `["server", "cli", "gpu"]` - Full functionality
- `minimal` = `[]` - Core inference engine only (no server, no CLI)
- `server` - REST API server (requires axum, tokio)
- `cli` - Command-line interface (requires clap)
- `gpu` - GPU acceleration via Trueno
- `full` - Alias for all features

**Examples:**

```bash
# Core inference library only (minimal dependencies)
cargo build --no-default-features --features minimal

# Server without CLI
cargo build --no-default-features --features server,gpu

# Everything enabled
cargo build --features full
```

## 🎯 Philosophy

**Total Control, Zero Compromise**

Build everything ourselves except HTTP infrastructure:
- ✅ **Transformer architecture** - Our code, Trueno-backed
- ✅ **Quantization** - Q4_0, Q8_0, Q4_K from scratch
- ✅ **Model parsing** - GGUF, safetensors native readers
- ✅ **Token encoding** - BPE, SentencePiece in pure Rust
- ✅ **Inference engine** - Every optimization under our control
- 🔧 **HTTP server** - axum (swappable via trait)

## 🚀 Target API

```rust
use realizar::{Model, Server};

// Load model (our loader, our format parsing)
let model = Model::from_gguf("models/llama-3.2-1b.gguf")?;

// Serve (swappable server backend)
Server::new(model)
    .with_gpu()
    .serve("0.0.0.0:8080")?;
```

```bash
# CLI
realizar serve --model llama-3.2-1b.gguf --port 8080

# REST API
curl -X POST http://localhost:8080/generate \
  -d '{"prompt": "Hello", "max_tokens": 100}'

# Metrics (Prometheus format)
curl http://localhost:8080/metrics
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│  HTTP Server (Swappable)           │
│  - axum (default, trait-based)     │
│  - hyper (future)                  │
│  - actix-web (future)              │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Inference Engine (FROM SCRATCH)   │
│  - Transformer (our code)          │
│  - Attention (Trueno-backed)       │
│  - Quantization (our algorithms)   │
│  - KV cache (our management)       │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Model Loader (FROM SCRATCH)       │
│  - GGUF parser (pure Rust)         │
│  - Safetensors reader (pure Rust)  │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Trueno (Compute Primitives)       │
│  - Matrix ops (SIMD/GPU)           │
│  - Vector ops (AVX2/NEON)          │
└─────────────────────────────────────┘
```

## 📦 Dependencies (Minimal)

```toml
[dependencies]
# OUR ecosystem - we control these
trueno = { path = "../trueno" }  # SIMD/GPU compute primitives

# HTTP server ONLY (swappable via trait)
axum = "0.7"
tokio = { version = "1", features = ["rt-multi-thread"] }

# CLI
clap = { version = "4", features = ["derive"] }

# Serialization (for API only, not ML)
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# That's it. NO candle, NO llama-cpp-rs, NO hf-hub
```

## 🔧 What We Build from Scratch

### 1. Model Formats (Pure Rust Parsers)
- **GGUF** - Ollama/llama.cpp format
- **Safetensors** - HuggingFace format
- No external dependencies, complete control

### 2. Transformer Architecture
```rust
pub struct Transformer {
    layers: Vec<TransformerLayer>,
    config: ModelConfig,
}

impl Transformer {
    pub fn forward(&self, tokens: &[u32]) -> Tensor {
        // Our implementation, Trueno ops
        let x = self.embed(tokens);
        for layer in &self.layers {
            x = layer.forward(x);  // We write this
        }
        self.lm_head(x)
    }
}
```

### 3. Attention Mechanism
```rust
pub fn attention(
    q: &Tensor,  // Trueno tensor
    k: &Tensor,
    v: &Tensor,
) -> Tensor {
    // Our attention implementation
    // Uses Trueno for matrix ops (SIMD/GPU)
    let scores = q.matmul(&k.transpose());
    let weights = scores.softmax();
    weights.matmul(v)
}
```

### 4. Quantization
```rust
pub mod quantize {
    // Q4_0 - 4-bit quantization
    pub fn q4_0(weights: &[f32]) -> (Vec<u8>, Vec<f32>) { }

    // Q8_0 - 8-bit quantization
    pub fn q8_0(weights: &[f32]) -> (Vec<i8>, Vec<f32>) { }

    // Q4_K - k-quant 4-bit
    pub fn q4_k(weights: &[f32]) -> Vec<u8> { }

    // Dequantization for inference
    pub fn dequantize(data: &[u8], qtype: QuantType) -> Vec<f32> { }
}
```

### 5. Token Encoding
```rust
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
}

impl Tokenizer {
    // BPE encoding (from scratch)
    pub fn encode(&self, text: &str) -> Vec<u32> { }

    // Decoding
    pub fn decode(&self, tokens: &[u32]) -> String { }
}
```

### 6. KV Cache
```rust
pub struct KVCache {
    keys: Vec<Tensor>,    // Trueno tensors
    values: Vec<Tensor>,
}

impl KVCache {
    // Efficient cache management
    pub fn update(&mut self, layer: usize, k: Tensor, v: Tensor) { }
    pub fn get(&self, layer: usize) -> (&Tensor, &Tensor) { }
}
```

## 🔌 Swappable HTTP Server

```rust
// HTTP server trait (axum is default, can swap)
pub trait HttpServer {
    fn serve(&self, addr: &str) -> Result<()>;
}

// Default: axum
pub struct AxumServer { /* ... */ }
impl HttpServer for AxumServer { /* ... */ }

// Future: hyper, actix-web, custom
pub struct HyperServer { /* ... */ }
impl HttpServer for HyperServer { /* ... */ }

// Usage
let server = Server::new(model)
    .with_backend(AxumServer::new())  // or HyperServer
    .serve("0.0.0.0:8080")?;
```

## 💡 Examples

Realizar includes **6 comprehensive examples** demonstrating all major features:

### 1. End-to-End Inference (`inference.rs`)
Complete text generation pipeline with model initialization, forward pass, and multiple sampling strategies (greedy, top-k, top-p).

```bash
cargo run --example inference
```

### 2. HTTP API Server (`api_server.rs`)
Deploy Realizar as a REST API service with demo model, handling tokenization and generation requests.

```bash
cargo run --example api_server
# Server runs at http://127.0.0.1:3000
# Test: curl http://127.0.0.1:3000/health
```

### 3. Tokenization (`tokenization.rs`)
Compare different tokenization strategies: Basic (vocabulary-based), BPE (Byte Pair Encoding), and SentencePiece.

```bash
cargo run --example tokenization
```

### 4. SafeTensors Loading (`safetensors_loading.rs`)
Load and inspect SafeTensors files (aprender compatibility), extract tensor data, interoperate with aprender-trained models.

```bash
cargo run --example safetensors_loading
```

### 5. Model Caching (`model_cache.rs`)
Demonstrate ModelCache for efficient model reuse with LRU eviction, metrics tracking, and config-based cache keys.

```bash
cargo run --example model_cache
```

### 6. GGUF Format Loading (`gguf_loading.rs`)
Load and inspect GGUF files (llama.cpp/Ollama format), parse headers and metadata, extract tensor data with dequantization support.

```bash
cargo run --example gguf_loading
```

See [`examples/README.md`](examples/README.md) for detailed documentation.

## ⚡ Reproducible Benchmarks

Realizar provides **scientifically rigorous, reproducible benchmarks** following [MLPerf™ Inference](https://mlcommons.org/benchmarks/inference/) methodology. All benchmarks use [Criterion.rs](https://bheisler.github.io/criterion.rs/book/) for statistical analysis with 95% confidence intervals.

### Quick Start

```bash
# Run all Realizar benchmarks
cargo bench

# Run comparative benchmarks (Realizar vs PyTorch)
make bench-comparative

# CLI benchmark commands
./target/release/realizar bench --list
./target/release/realizar bench tensor_ops
./target/release/realizar viz --samples 100
```

### Benchmark Suites

| Suite | Command | Description |
|-------|---------|-------------|
| `tensor_ops` | `cargo bench --bench tensor_ops` | Tensor creation, shape access, indexing |
| `inference` | `cargo bench --bench inference` | End-to-end token generation |
| `cache` | `cargo bench --bench cache` | KV cache hit/miss, eviction |
| `tokenizer` | `cargo bench --bench tokenizer` | BPE/SentencePiece encode/decode |
| `quantize` | `cargo bench --bench quantize` | Q4_0/Q8_0 dequantization |
| `comparative` | `cargo bench --bench comparative` | MNIST, CIFAR-10, Iris vs PyTorch |

### Reproducing Results

**Prerequisites:**

```bash
# Rust toolchain
rustup default stable
rustup update

# Python environment (uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# PyTorch dependencies
cd benches/comparative
uv sync
```

**Hardware Requirements:**

- CPU: x86_64 with AVX2 or ARM64 with NEON
- RAM: 8GB minimum
- Recommended: Disable CPU frequency scaling for stable measurements

```bash
# Linux: Set performance governor
sudo cpupower frequency-set --governor performance
```

**Step-by-Step Reproduction:**

```bash
# 1. Clone and build
git clone https://github.com/paiml/realizar.git
cd realizar
cargo build --release

# 2. Run Realizar benchmarks
cargo bench --bench tensor_ops
cargo bench --bench cache
cargo bench --bench comparative

# 3. Run PyTorch baseline (requires uv)
cd benches/comparative
uv sync
uv run pytorch_baseline.py --all --output pytorch_results.json

# 4. Generate comparison report
uv run run_comparison.py --output comparison_report.md

# 5. View HTML reports
open target/criterion/report/index.html
```

### Datasets

Benchmarks use canonical ML datasets via [Alimentar](https://github.com/paiml/alimentar) for PyTorch parity:

| Dataset | Dimensions | Classes | Features |
|---------|------------|---------|----------|
| **MNIST** | 28×28×1 | 10 | 784 |
| **CIFAR-10** | 32×32×3 | 10 | 3,072 |
| **Fashion-MNIST** | 28×28×1 | 10 | 784 |
| **Iris** | Tabular | 3 | 4 |

### Comparative Framework Testing

We benchmark against PyTorch under equivalent conditions:

| Setting | Value |
|---------|-------|
| Threads | 1 (single-threaded) |
| Batch sizes | 1, 8, 32 |
| Device | CPU only |
| Warm-up | 50 iterations |
| Measurement | 1000 iterations |

**Run comparative benchmarks:**

```bash
# Full comparison (Makefile)
make bench-comparative

# Manual execution
cargo bench --bench comparative
uv run benches/comparative/pytorch_baseline.py --all
uv run benches/comparative/run_comparison.py
```

### Performance Results

**Realizar (v0.2.1) - Intel Core i7, Linux 6.8:**

| Benchmark | Batch | Latency (p50) | Throughput |
|-----------|-------|---------------|------------|
| MNIST inference | 1 | 780 ns | 1.28M samples/s |
| MNIST inference | 32 | 23.8 µs | 1.34M samples/s |
| CIFAR-10 inference | 1 | 1.58 µs | 633K samples/s |
| CIFAR-10 inference | 32 | 49.8 µs | 642K samples/s |
| Iris inference | 32 | 210 ns | 152M samples/s |
| Tensor creation (10) | - | 18 ns | - |
| Tensor creation (10K) | - | 643 ns | - |
| Cache hit | - | 39 ns | - |

### Statistical Methodology

- **Warm-up phase**: Stabilize CPU caches and branch predictors
- **Sample collection**: 100 samples per benchmark (Criterion default)
- **Confidence intervals**: 95% CI reported as [lower, mean, upper]
- **Regression detection**: Automatic comparison against baseline
- **Effect size**: Cohen's d for practical significance

```
tensor_creation/10      time:   [17.887 ns 17.966 ns 18.043 ns]
                                 ^         ^         ^
                              lower      mean      upper
                              bound    estimate    bound
```

### Visualization

```bash
# Terminal visualization
./target/release/realizar viz

# Output includes:
# - Sparklines (trend visualization)
# - ASCII histograms (distribution shape)
# - Statistical summary (mean, std_dev, p50/p95/p99)
# - Multi-benchmark comparison tables
```

### References

1. MLPerf™ Inference Benchmark Suite. MLCommons. https://mlcommons.org/benchmarks/inference/
2. Criterion.rs: Statistics-driven Microbenchmarking. https://bheisler.github.io/criterion.rs/book/
3. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters*. Wiley.
4. Fleming, P. J., & Wallace, J. J. (1986). How not to lie with statistics. *CACM*, 29(3), 218-221.

## 📊 Roadmap

### Phase 1: Core Inference (Weeks 1-8) ✅ COMPLETE

**Build from scratch:**
- ✅ GGUF parser (binary format reader)
- ✅ Safetensors parser (zero-copy reader)
- ✅ Transformer architecture (attention, FFN, LayerNorm, RoPE)
- ✅ Quantization (Q4_0, Q8_0, dequantization)
- ✅ Tokenizer (BPE, SentencePiece)
- ✅ KV cache management
- ✅ Inference engine (generation loop, greedy/top-k/top-p)
- ✅ HTTP server with axum (REST API)
- ✅ CLI: `realizar serve --demo` (model loading in Phase 2)
- ✅ 260 tests (211 unit + 42 property + 7 integration), 94.61% coverage

**Success criteria:**
- ✅ GGUF and Safetensors parsers working
- ✅ Quantization working (Q4_0, Q8_0)
- ✅ REST API with /health, /tokenize, /generate
- ✅ GPU acceleration via Trueno
- ✅ Zero external ML dependencies
- ✅ TDG Score: 93.9/100 (A)

### Phase 2: Optimization (Weeks 9-16) ✅ COMPLETE

- ✅ Advanced quantization (Q4_K, Q5_K, Q6_K)
- ✅ Flash Attention (memory-efficient block-wise computation)
- ✅ Batch inference
- ✅ Streaming responses (SSE)
- ✅ Model caching/warming
- ✅ Benchmarks vs llama.cpp

### Phase 3: Advanced Models (Weeks 17-24)

- ✅ Multi-query attention (MQA)
- ✅ Grouped-query attention (GQA)
- ✅ RoPE position embeddings
- ✅ ALiBi position embeddings
- [ ] Vision models (LLaVA, Qwen-VL)

### Phase 4: Production (Weeks 25-32) ✅ COMPLETE

- ✅ Multi-model serving (ModelRegistry with concurrent access)
- ✅ Request batching (batch tokenize & generate endpoints)
- ✅ Monitoring/metrics (Prometheus-compatible /metrics endpoint)
- ✅ Docker + GPU support (Dockerfile, docker-compose, Kubernetes, AWS ECS)
- ✅ Load testing (Rust-based load test client, 7 scenarios, performance targets)

## 🛠️ Development

```bash
# Build
cargo build --release

# Test
cargo test

# Quality gates
make quality-gates

# Run (when implemented)
cargo run --release -- serve --model llama-3.2-1b.gguf --port 8080
```

## 📚 Documentation

Comprehensive documentation is available as an mdBook:

```bash
# Build and view the book
make book

# Build only
make book-build

# Live reload (for writing docs)
make book-serve

# Open in browser
make book-open
```

The book covers:
- **Core Architecture** - Design philosophy, Trueno integration, feature flags
- **Model Formats** - GGUF and Safetensors parsing from scratch
- **Quantization** - Q4_0, Q8_0, and K-quant algorithms
- **Transformer Architecture** - Attention, RoPE, FFN, KV cache implementation
- **Tokenization** - BPE and SentencePiece without external libraries
- **REST API & CLI** - Production HTTP server and command-line interface
- **GPU Acceleration** - Trueno SIMD/GPU dispatch
- **EXTREME TDD** - Property-based testing, mutation testing methodology
- **Development Phases** - Phase 1-4 roadmap and implementation details

**Note:** Book structure is validated in `make quality-gates` to ensure documentation stays in sync with code.

## 🎓 Learning Resources

We're building everything from scratch. Key papers:
- **[11] TensorFlow** - Model serving architecture
- **[12] PyTorch** - Imperative ML framework design
- **[13] NumPy** - N-dimensional array design
- **[18] BLAS** - Linear algebra API design
- **[19] Strassen** - Fast matrix multiplication
- **[20] Kahan** - Numerical stability

Full spec: [docs/specifications/pure-rust-ml-library-research-spec.md](docs/specifications/pure-rust-ml-library-research-spec.md)

## 🔒 Security

- **Pure Rust** - Memory safe by design
- **Zero unsafe** in public API
- **Minimal deps** - axum + tokio only for HTTP
- `cargo audit` pre-commit
- `cargo-deny` license checks

## 🤝 Contributing

1. Fork repo
2. EXTREME TDD (tests first)
3. `make quality-gates` passes
4. All commits on `master`

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- **[Trueno](https://github.com/paiml/trueno)** - SIMD/GPU compute primitives (our ecosystem)
- **[Aprender](https://github.com/paiml/aprender)** - ML algorithms (Phase 2+)
- **[Renacer](https://github.com/paiml/renacer)** - Profiling
- **[paiml-mcp-agent-toolkit](https://github.com/paiml/paiml-mcp-agent-toolkit)** - Quality gates
- **[bashrs](https://github.com/paiml/bashrs)** - Script enforcement

Developed by [Pragmatic AI Labs](https://paiml.com)

---

**Built from SCRATCH with EXTREME TDD** 🦀⚡
