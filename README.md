<div align="center">

# realizar

**Pure Rust ML Inference Engine**

[![CI](https://github.com/paiml/realizar/workflows/CI/badge.svg)](https://github.com/paiml/realizar/actions)
[![Crates.io](https://img.shields.io/crates/v/realizar.svg)](https://crates.io/crates/realizar)
[![Docs](https://docs.rs/realizar/badge.svg)](https://docs.rs/realizar)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

ML inference from scratch in Rust. GGUF/SafeTensors parsing, quantization (Q4_K, Q8_0), transformer inference. SIMD/GPU via [Trueno](https://github.com/paiml/trueno).

## Quick Start

```bash
cargo install realizar
realizar serve --demo --port 8080
curl -X POST http://localhost:8080/generate -d '{"prompt": "Hello", "max_tokens": 10}'
```

## Features

| Category | Details |
|----------|---------|
| Formats | GGUF, SafeTensors, APR (native) |
| Quantization | Q4_0, Q8_0, Q4_K, Q5_K, Q6_K |
| Inference | Transformer, RoPE, KV cache, Flash Attention |
| API | REST, streaming, Prometheus metrics |
| Quality | 1,680+ tests, 95% coverage |

## Benchmarks

### APR Format (Classical ML - Pure Rust)

| Model | Parameters | Latency | Throughput |
|-------|------------|---------|------------|
| Iris | 131 | **103ns** | 9.6M inferences/sec |
| MNIST | 103K | **73µs** | 13.6K inferences/sec |
| Large NN | 1M | **410µs** | 2.4K inferences/sec |

### GGUF Format (LLMs)

| Model | Size | Backend | Throughput |
|-------|------|---------|------------|
| Phi-2 Q4_K_M | 2.7B | RTX 4090 (CUDA) | **477 tok/s** |
| Phi-2 Q4_K_M | 2.7B | CPU (AVX2) | ~15 tok/s |

### Benchmark Matrix (ELI5)

```
┌─────────────────────────────────────────────────────────────────┐
│                  What This Matrix Measures                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Think of it like comparing cars:                                │
│                                                                  │
│  ┌──────────┬──────────────────────────────────────────────┐    │
│  │ Runtime  │ Which "engine" runs your model?              │    │
│  ├──────────┼──────────────────────────────────────────────┤    │
│  │ realizar │ Our pure Rust engine (this project)         │    │
│  │ llama.cpp│ Popular C++ engine (industry standard)      │    │
│  │ Ollama   │ User-friendly wrapper around llama.cpp      │    │
│  └──────────┴──────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────┬──────────────────────────────────────────────┐    │
│  │ Backend  │ Which "fuel" powers the engine?             │    │
│  ├──────────┼──────────────────────────────────────────────┤    │
│  │ CPU      │ Regular processor (slower, always works)    │    │
│  │ CUDA     │ NVIDIA GPU (fastest, needs GPU)             │    │
│  │ WGPU     │ Cross-platform GPU (good balance)           │    │
│  └──────────┴──────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────┬──────────────────────────────────────────────┐    │
│  │ Format   │ Which "fuel type" for your model?           │    │
│  ├──────────┼──────────────────────────────────────────────┤    │
│  │ GGUF     │ Quantized LLMs (smaller, fast)              │    │
│  │ APR      │ Our native format (fastest for small ML)    │    │
│  │ SafeT    │ HuggingFace format (full precision)         │    │
│  └──────────┴──────────────────────────────────────────────┘    │
│                                                                  │
│  Matrix Result = Runtime × Backend × Format                      │
│                                                                  │
│  Example: "llama.cpp + CUDA + GGUF" = 477 tok/s on RTX 4090     │
│           "realizar + CPU + APR"   = 9.6M inf/s for tiny models │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why This Matters:**
- **Small Models (Iris, MNIST)**: Use APR format → nanosecond latency
- **Large Models (LLMs)**: Use GGUF format → GPU acceleration essential
- **Production**: Match your hardware to the right runtime/backend combo

### Run Benchmarks

```bash
# APR format (classical ML)
cargo bench --bench apr_real

# GGUF format (transformers)
cargo bench --bench gguf_real

# Compare APR vs GGUF on same workload
cargo bench --bench comparative

# External servers (llama.cpp, Ollama)
cargo bench --bench external_matrix --features bench-http
```

## Examples

```bash
# All examples
cargo run --example inference          # Basic inference demo
cargo run --example api_server         # HTTP server demo
cargo run --example gguf_loading       # Load GGUF models
cargo run --example apr_loading        # Load APR models
cargo run --example tokenization       # Tokenizer demo
cargo run --example safetensors_loading # SafeTensors demo
cargo run --example observability_demo  # Metrics demo
cargo run --example model_cache        # Caching demo
```

## Usage

```bash
realizar serve --demo --port 8080     # Demo server
curl http://localhost:8080/health     # Health check
curl http://localhost:8080/metrics    # Prometheus
```

## Install

```bash
cargo install realizar                # From crates.io
cargo install --path .                # From source
```

## Feature Flags

- `default` = server + cli + gpu
- `minimal` = Core inference only
- `bench-http` = External server benchmarking

## Architecture

```
realizar/
├── src/
│   ├── gguf.rs         # GGUF parser + transformer inference
│   ├── safetensors.rs  # SafeTensors parser
│   ├── apr.rs          # APR format (native)
│   ├── quantize.rs     # Q4_K, Q8_0 dequantization
│   ├── layers.rs       # Transformer layers
│   ├── tokenizer.rs    # BPE, SentencePiece
│   ├── api.rs          # REST endpoints
│   └── bench_preflight.rs # Deterministic benchmarking
└── benches/
    ├── apr_real.rs     # APR benchmarks
    ├── gguf_real.rs    # GGUF benchmarks
    ├── comparative.rs  # Format comparison
    └── external_matrix.rs # External server benchmarks
```

## License

MIT - [Pragmatic AI Labs](https://paiml.com)
