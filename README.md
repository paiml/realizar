# Realizar ⚡

> **Pure Rust, Portable, High-Performance ML Library**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-blue.svg)](https://www.rust-lang.org)
[![Status](https://img.shields.io/badge/status-Phase_1-orange.svg)](docs/specifications/pure-rust-ml-library-research-spec.md)

**Realizar** (Spanish: "to accomplish, to achieve") is a next-generation machine learning library that provides memory-safe, unified API for CPU (SIMD), GPU, and WebAssembly execution.

## 🎯 Mission

Create a production-ready ML library in pure Rust that:
- **Eliminates API fragmentation** between CPU and GPU backends
- **Leverages memory safety** to prevent bugs and vulnerabilities
- **Provides performance portability** across x86-64, ARM, WASM, and GPU
- **Follows EXTREME TDD** with 85%+ coverage and zero tolerance for defects

## 🚀 Quick Start

```rust
use realizar::Tensor;

// Create tensors
let a = Tensor::from_vec(vec![3, 3], vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
]).unwrap();

// Check tensor properties
assert_eq!(a.shape(), &[3, 3]);
assert_eq!(a.ndim(), 2);
assert_eq!(a.size(), 9);
```

### Future Operations (Phase 1+)

```rust
// Element-wise operations (SIMD-accelerated)
let sum = a.add(&b).unwrap();  // Phase 1

// Matrix multiplication (GPU-accelerated for large matrices)
let product = a.matmul(&b).unwrap();  // Phase 2
```

## 📊 Project Status

| Phase | Status | Completion |
|-------|--------|------------|
| **Phase 1: Foundation** | 🟡 In Progress | 10% |
| Phase 2: GPU Acceleration | ⏳ Planned | 0% |
| Phase 3: WASM Support | ⏳ Planned | 0% |
| Phase 4: Advanced Operations | ⏳ Planned | 0% |
| Phase 5: Aprender Integration | ⏳ Planned | 0% |
| Phase 6: Production Deployment | ⏳ Planned | 0% |

**Current Sprint:** 1
**Target:** Phase 1 completion by Week 8

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│  Realizar API (User-Facing)                │
│  - Tensor<T>, DataFrame, Model traits      │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Aprender (ML Algorithms) - NATIVE          │
│  - LinearRegression, KMeans, etc.           │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Trueno (Compute Primitives) - NATIVE       │
│  - Vector, Matrix, SIMD/GPU/WASM backends   │
└─────────────────────────────────────────────┘
```

### Ecosystem Integration

Realizar integrates with the Pragmatic AI Labs ML ecosystem:

| Project | Role | Integration |
|---------|------|-------------|
| **[Trueno](https://github.com/paiml/trueno)** | Compute backend | 100% - Native first-class support |
| **[Aprender](https://github.com/paiml/aprender)** | ML algorithms | 100% - Will be refactored to use Realizar |
| **[Renacer](https://github.com/paiml/renacer)** | Profiling & debugging | Development tool |
| **[paiml-mcp-agent-toolkit](https://github.com/paiml/paiml-mcp-agent-toolkit)** | Quality & roadmap | 100% - PMAT quality gates |
| **[bashrs](https://github.com/paiml/bashrs)** | Script enforcement | 100% - Pre-commit validation |

## 📖 Research Foundation

This project is built on **25 peer-reviewed computer science publications** covering:

- Memory safety and systems programming (Rust, RustBelt)
- SIMD and vectorization (portable SIMD, auto-vectorization)
- GPU computing (Vulkan, WebGPU, MAGMA)
- WebAssembly (WASM spec, SIMD proposal)
- ML system design (TensorFlow, PyTorch, NumPy, TVM)
- Performance analysis (Roofline model, cache-oblivious algorithms)
- Testing and verification (QuickCheck, mutation testing)

**Full specification with citations:** [docs/specifications/pure-rust-ml-library-research-spec.md](docs/specifications/pure-rust-ml-library-research-spec.md)

## 🎓 Key Features

### 1. Unified Backend Dispatch

Operations automatically select the optimal backend:

| Operation | Complexity | Threshold | Backend | Expected Speedup |
|-----------|-----------|-----------|---------|------------------|
| Element-wise add | O(n) | Always | SIMD (AVX2) | 8x |
| Matrix multiply | O(n³) | n > 500 | GPU (wgpu) | 10x |
| Dot product | O(n) | n > 1000 | SIMD (FMA) | 3.4x |
| 2D Convolution | O(n²k²) | output > 10K | GPU | 50x |

### 2. Memory Safety Guarantees

- **Zero unsafe code in public API** - All `unsafe` isolated in backend implementations
- **Compile-time shape validation** - Shape mismatches caught at compile time where possible
- **Runtime size checks** - Runtime validation for dynamic operations
- **No null pointers, no use-after-free** - Rust's ownership model prevents memory bugs

### 3. Production Quality

Following EXTREME TDD methodology from [paiml-mcp-agent-toolkit](https://github.com/paiml/paiml-mcp-agent-toolkit):

| Metric | Target | Current |
|--------|--------|---------|
| Test Coverage | ≥85% | 100% (8/8 tests) |
| Mutation Score | ≥80% | TBD |
| TDG Score | ≥90/100 | TBD |
| Clippy Warnings | 0 | ✅ 0 |
| Cyclomatic Complexity | ≤10 | ✅ Max 5 |

## 🛠️ Development

### Prerequisites

```bash
# Install Rust (1.75+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install development tools
make install-tools
```

### Build and Test

```bash
# Development build
make build

# Run all tests
make test

# Run quality gates (format, clippy, tests)
make quality-gates

# Generate coverage report
make coverage

# Run benchmarks
make bench
```

### Quality Gates

All contributions must pass:

```bash
make quality-gates  # Runs:
  1. cargo fmt --check    # Code formatting
  2. cargo clippy         # Zero warnings
  3. cargo test           # All tests pass
  4. bashrs lint          # Makefile/script validation
```

### Profiling with Renacer

```bash
# Profile benchmarks
make profile

# Profile specific benchmark
renacer --function-time --source -- cargo bench --bench tensor_ops
```

## 📈 Performance Targets

Baseline comparisons (Phase 2+):

| Operation | Size | Realizar (Target) | NumPy (CPU) | PyTorch (GPU) |
|-----------|------|-------------------|-------------|---------------|
| MatMul | 1000×1000 | 10ms | 50ms | 8ms |
| Element-wise add | 1M elements | 0.5ms | 2ms | 1ms |
| Dot product | 10K elements | 3μs | 10μs | N/A |
| 2D Convolution | 512×512 | 20ms | 100ms | 15ms |

## 🗺️ Roadmap

### Phase 1: Foundation (Weeks 1-8) 🟡 Current

- [x] Project structure and quality gates
- [x] Core Tensor API with shape validation
- [ ] Element-wise operations (add, sub, mul, div)
- [ ] SIMD backend integration via Trueno
- [ ] 100 unit tests, 20 property tests
- [ ] TDG score ≥85/100

### Phase 2: GPU Acceleration (Weeks 9-16)

- [ ] GPU dispatch heuristics
- [ ] Matrix multiplication on GPU (wgpu)
- [ ] Automatic CPU fallback
- [ ] 10x speedup for large matrices

### Phase 3: WASM Support (Weeks 17-24)

- [ ] WASM SIMD128 backend
- [ ] JavaScript bindings
- [ ] Browser demo deployment
- [ ] 5x faster than pure JavaScript

### Phase 4-6: Advanced Operations, Integration, Deployment

See [full roadmap](docs/specifications/pure-rust-ml-library-research-spec.md#roadmap-and-milestones)

## 🔒 Security

- **Dependency policy:** MIT/Apache-2.0 licenses only
- **Security audits:** `cargo audit` in pre-commit hooks
- **No known CVEs:** Automated checks via `cargo-deny`
- **Minimal dependencies:** <50 direct dependencies
- **Supply chain verification:** All crates from crates.io only

## 📦 Installation

**Note:** Realizar is in early development (Phase 1). Not yet published to crates.io.

```toml
[dependencies]
realizar = { git = "https://github.com/paiml/realizar" }
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Follow EXTREME TDD (tests first!)
4. Ensure `make quality-gates` passes
5. Submit pull request

**Important:** All commits must be on `master` branch (no feature branches per project policy).

## 📚 Documentation

- **[Research Specification](docs/specifications/pure-rust-ml-library-research-spec.md)** - Full technical spec with 25 peer-reviewed citations
- **API Docs:** `cargo doc --open`
- **Examples:** `examples/` directory (coming in Phase 1)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- **[Trueno](https://github.com/paiml/trueno)** - Multi-target compute library (100% coverage, TDG 96.1/100)
- **[Aprender](https://github.com/paiml/aprender)** - Pure Rust ML library (TDG 94.1/100)
- **[Renacer](https://github.com/paiml/renacer)** - Profiling and debugging (TDG 99.9/100)
- **[paiml-mcp-agent-toolkit](https://github.com/paiml/paiml-mcp-agent-toolkit)** - Quality gates and TDD workflow
- **[bashrs](https://github.com/paiml/bashrs)** - Makefile/script enforcement

Developed by [Pragmatic AI Labs](https://paiml.com)

## 🌐 Deployment

Production deployment to **https://interactive.paiml.com** (Phase 6)

- **S3 Bucket:** `interactive.paiml.com-production-mcb21d5j`
- **CloudFront:** Distribution ID `ELY820FVFXAFF`
- **CI/CD:** GitHub Actions with quality gates
- **Monitoring:** Prometheus + CloudWatch

## 📞 Contact

- **Issues:** [GitHub Issues](https://github.com/paiml/realizar/issues)
- **Discussions:** [GitHub Discussions](https://github.com/paiml/realizar/discussions)
- **Email:** contact@paiml.com

---

**Built with EXTREME TDD and Toyota Way principles** 🦀⚡
