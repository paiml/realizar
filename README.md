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
| Quality | 1,600+ tests, 95% coverage |

## Benchmarks

| Model | Latency | Throughput |
|-------|---------|------------|
| Iris (131 params) | 103ns | 9.6M/s |
| MNIST (103K params) | 73µs | 13.6K/s |

```bash
cargo bench --bench apr_real
cargo bench --bench comparative  # APR vs GGUF
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

## License

MIT - [Pragmatic AI Labs](https://paiml.com)
