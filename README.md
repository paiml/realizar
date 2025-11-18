# Realizar ⚡

> **Pure Rust Model Serving & ML Library**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-blue.svg)](https://www.rust-lang.org)

**Realizar** - Production model serving (Ollama, HuggingFace) + high-performance ML library in pure Rust.

## 🎯 Mission

**Phase 1 (NOW): Model Serving** - Deploy any SLM/LLM from Ollama or HuggingFace
- Ollama integration (llama, phi, qwen, etc.)
- HuggingFace model loading (safetensors)
- REST API server
- GPU acceleration (CUDA/Metal/Vulkan)

**Phase 2+: ML Library** - SIMD/GPU/WASM compute primitives

## 🚀 Quick Start (Phase 1 Target)

```rust
use realizar::ModelServer;

// Ollama model
let server = ModelServer::ollama("llama3.2:1b")
    .with_gpu()
    .serve("0.0.0.0:8080")?;

// HuggingFace model
let server = ModelServer::huggingface("microsoft/Phi-3-mini-4k-instruct")
    .with_gpu()
    .serve("0.0.0.0:8080")?;
```

```bash
# REST API
curl -X POST http://localhost:8080/generate \
  -d '{"prompt": "Hello", "max_tokens": 100}'
```

## 📊 Roadmap

### Phase 1: Model Serving (Weeks 1-8) 🟡 NOW

- [ ] Ollama integration (llama.cpp bindings)
- [ ] HuggingFace model loading (safetensors)
- [ ] REST API server (axum)
- [ ] GPU support (CUDA/Metal/Vulkan)
- [ ] Streaming responses (SSE)
- [ ] Model caching
- [ ] 100+ tests, 85%+ coverage

### Phase 2: Tensor Operations (Weeks 9-16)

- [ ] Core Tensor API
- [ ] SIMD backend (Trueno)
- [ ] Element-wise ops
- [ ] Matrix operations

### Phase 3: GPU Acceleration (Weeks 17-24)

- [ ] GPU tensor operations (wgpu)
- [ ] Automatic CPU/GPU dispatch

### Phase 4: WASM Support (Weeks 25-32)

- [ ] WASM model serving
- [ ] Browser inference

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│  REST API (axum)               │
│  - /generate                    │
│  - /embed                       │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  Model Backends                 │
│  - Ollama (llama.cpp)          │
│  - HuggingFace (candle)        │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  GPU Acceleration               │
│  - CUDA / Metal / Vulkan        │
└─────────────────────────────────┘
```

## 🛠️ Development

```bash
# Build
cargo build --release

# Test
cargo test

# Quality gates
make quality-gates

# Run server (when implemented)
cargo run --release -- --model llama3.2:1b --port 8080
```

## 📦 Dependencies (Phase 1)

```toml
[dependencies]
# Model serving
candle-core = "0.3"          # HuggingFace models
candle-transformers = "0.3"
hf-hub = "0.3"
llama-cpp-rs = "0.1"         # Ollama models

# Server
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower = "0.4"

# Compute (Phase 2+)
trueno = { path = "../trueno" }
aprender = { path = "../aprender" }
```

## 🔒 Security

- MIT/Apache-2.0 licenses only
- `cargo audit` in pre-commit
- `cargo-deny` checks
- Minimal dependencies

## 📚 Documentation

- **[Research Spec](docs/specifications/pure-rust-ml-library-research-spec.md)** - 25 peer-reviewed citations
- **API Docs:** `cargo doc --open`

## 🤝 Contributing

1. Fork repo
2. EXTREME TDD (tests first)
3. `make quality-gates` passes
4. All commits on `master` branch

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- **[Trueno](https://github.com/paiml/trueno)** - SIMD/GPU compute (Phase 2+)
- **[Aprender](https://github.com/paiml/aprender)** - ML algorithms (Phase 2+)
- **[Renacer](https://github.com/paiml/renacer)** - Profiling
- **[paiml-mcp-agent-toolkit](https://github.com/paiml/paiml-mcp-agent-toolkit)** - Quality gates
- **[bashrs](https://github.com/paiml/bashrs)** - Script enforcement

Developed by [Pragmatic AI Labs](https://paiml.com)

---

**Built with EXTREME TDD** 🦀⚡
