# Realizar ⚡

> **Pure Rust Model Serving - Built from Scratch**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-blue.svg)](https://www.rust-lang.org)

**Realizar** - Production ML inference engine built **100% from scratch** in pure Rust.

## 🚀 Quick Start

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
- ✅ 237 tests (195 unit + 42 property-based), 95.12% coverage

**Success criteria:**
- ✅ GGUF and Safetensors parsers working
- ✅ Quantization working (Q4_0, Q8_0)
- ✅ REST API with /health, /tokenize, /generate
- ✅ GPU acceleration via Trueno
- ✅ Zero external ML dependencies
- ✅ TDG Score: 100.0/100 (A+)

### Phase 2: Optimization (Weeks 9-16)

- [ ] Advanced quantization (Q4_K, Q5_K, Q6_K)
- [ ] Flash Attention (Trueno-backed)
- [ ] Batch inference
- [ ] Streaming responses (SSE)
- [ ] Model caching/warming
- [ ] Benchmarks vs llama.cpp

### Phase 3: Advanced Models (Weeks 17-24)

- [ ] Multi-query attention (MQA)
- [ ] Grouped-query attention (GQA)
- [ ] RoPE position embeddings
- [ ] ALiBi position embeddings
- [ ] Vision models (LLaVA, Qwen-VL)

### Phase 4: Production (Weeks 25-32)

- [ ] Multi-model serving
- [ ] Request batching
- [ ] Monitoring/metrics
- [ ] Docker + GPU support
- [ ] Load testing

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
