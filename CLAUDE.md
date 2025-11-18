# Claude Code Development Guide for Realizar

## Project Overview

**Realizar** - Pure Rust ML inference engine built from scratch for GGUF and Safetensors model serving.

- **Philosophy:** Total control, zero compromise - build everything ourselves except HTTP infrastructure
- **Architecture:** Model parsers → Inference engine → Trueno compute primitives
- **Methodology:** EXTREME TDD with mutation testing, property-based testing, 85%+ coverage
- **Quality Target:** TDG Score ≥95.0/100 (A+)

## Critical Dependencies - ALWAYS USE LATEST

### Trueno (SIMD/GPU Compute Primitives)

**IMPORTANT:** Trueno is actively developed and frequently updated. **ALWAYS check for the latest version.**

```bash
# Check trueno version before any development work
cd ../trueno && git pull && grep "^version" Cargo.toml
```

**Current Integration:**
- Path: `../trueno`
- Features: `["gpu"]` for GPU acceleration
- Status: v0.2.2 (2024-11-18) - includes abs() SIMD implementation

**Update Workflow:**
1. Pull latest trueno: `cd ../trueno && git pull`
2. Check version: `grep "^version" Cargo.toml`
3. Update realizar's Cargo.toml with new version
4. Test integration: `cargo test --lib`
5. Commit with clear message about trueno version bump

**Trueno Capabilities:**
- Vector operations: add, sub, mul, div, dot, sum, norm_l1, norm_l2
- SIMD backends: AVX2, SSE2, NEON, WASM, Scalar
- GPU backend: wgpu-based (optional feature)
- Activation functions: ReLU, sigmoid, GELU, swish, mish, selu, hardswish
- Performance: 2-11x SIMD speedups on compute-bound operations

### Aprender (ML Library)

**IMPORTANT:** Aprender is actively developed and frequently released. **ALWAYS check for the latest version.**

```bash
# Check aprender version and status
cd ../aprender && git pull && grep "^version" Cargo.toml
```

**Current Status:**
- Version: v0.1.0 (released to crates.io 2024-11-18)
- TDG Score: 95.6/100 (A+)
- Test Coverage: 97.72%
- Path: `../aprender`

**Aprender Primitives (Fallback Option):**
- `Vector<T>` - Generic 1D array with sum, mean, dot, norm, variance
- `Matrix<T>` - Row-major 2D array with matmul, transpose, Cholesky
- **Pure Rust:** Forbids unsafe code entirely
- **Battle-tested:** 149 tests (127 unit + 22 property)

**When to Use Aprender:**
- If trueno has compilation issues (rare)
- For pure Rust fallback without SIMD/GPU
- Can swap implementations transparently

**Update Workflow:**
1. Pull latest aprender: `cd ../aprender && git pull`
2. Check if relevant for inference primitives
3. Consider integration if trueno unavailable
4. Document in commit message

## Development Workflow

### Before Starting Any Work

```bash
# 1. Check ecosystem versions
cd ../trueno && git pull && grep "^version" Cargo.toml
cd ../aprender && git pull && grep "^version" Cargo.toml
cd realizar

# 2. Update dependencies if needed
# Edit Cargo.toml with new versions

# 3. Verify clean build
cargo clean
cargo test --lib

# 4. Check quality baselines
pmat analyze tdg
pmat analyze satd
pmat analyze complexity
```

### EXTREME TDD Methodology

**Follow RED-GREEN-REFACTOR:**

1. **RED:** Write failing tests first
   - Comprehensive test coverage (edge cases, errors, valid inputs)
   - Property-based tests for mathematical correctness
   - Document expected behavior

2. **GREEN:** Minimal implementation to pass tests
   - Focus on correctness, not optimization
   - Use clear, readable code
   - Leverage trueno primitives where applicable

3. **REFACTOR:** Clean up and optimize
   - Fix clippy warnings (zero tolerance)
   - Apply rustfmt formatting
   - Extract helper functions
   - Document with examples

**Quality Gates (all must pass):**
```bash
make fmt-check     # Format check
make clippy        # Zero warnings
make test          # All tests pass
make test-fast     # < 5 minutes
make coverage      # <10 minutes, aim for 85%+
```

### Trueno Integration Patterns

**Prefer Trueno for Compute:**
```rust
// Good: Use trueno for vector operations
use trueno::Vector;

let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
let result = a.dot(&b); // SIMD-accelerated
```

**Matrix Operations:**
```rust
// Good: Use trueno for matrix multiplication
use trueno::Matrix;

let weights = Matrix::from_slice(128, 256, &data);
let input = Matrix::from_slice(1, 128, &input_data);
let output = weights.matmul(&input); // GPU-accelerated if available
```

**Activation Functions:**
```rust
// Good: Use trueno activations for inference
use trueno::Vector;

let logits = Vector::from_slice(&[0.1, -0.5, 0.3]);
let activated = logits.relu(); // SIMD-accelerated ReLU
```

## Phase 1 Roadmap Progress

### Week 1-2: Model Parsers ✅ COMPLETE
- ✅ GGUF parser (header + metadata + tensor_info)
- ✅ Safetensors parser (JSON metadata + zero-copy data)
- ✅ 26 tests passing
- ✅ TDG Score: 96.2/100 (A+)
- ✅ Zero SATD violations

### Week 3-4: Transformer Components ✅ COMPLETE
- ✅ Layer normalization (7 tests, epsilon-based normalization)
- ✅ Linear layer (6 tests, weight/bias loading)
- ✅ GELU activation (5 tests, tanh approximation)
- ✅ Feed-forward networks (FFN) (6 tests, 2-layer with GELU)
- ✅ Softmax activation (6 tests, numerically stable)
- ✅ Attention mechanism (8 tests, scaled dot-product attention)
- ✅ RoPE position embeddings (11 tests, rotary position encoding)
- ✅ KV cache management (10 tests, efficient inference caching)

### Week 5-6: Quantization ✅ COMPLETE
- ✅ Q4_0 dequantization (4-bit, block size 32)
- ✅ Q8_0 dequantization (8-bit, block size 32)
- ✅ Dequantization for inference
- ✅ EXTREME TDD (5 comprehensive tests)
- [ ] Mixed precision support (deferred)

### Week 7-8: Tokenizer & Inference ✅ COMPLETE
- ✅ Basic tokenizer (10 tests, encode/decode)
- ✅ Embedding layer (6 tests, token to vector)
- ✅ Complete Model struct (5 tests, end-to-end inference)
- ✅ Generation loop (6 tests, token sampling)
- ✅ Sampling strategies (16 tests, greedy/top-k/top-p)
- ✅ BPE tokenizer (14 tests, byte pair encoding)
- ✅ SentencePiece tokenizer (14 tests, unigram model)
- ✅ HTTP API with axum (8 tests, REST endpoints)

## Quality Standards

**Mandatory Requirements:**
- **TDG Score:** ≥95.0/100 (A+ grade)
- **Test Coverage:** ≥85%
- **Mutation Score:** ≥80%
- **Cyclomatic Complexity:** ≤10 per function
- **Clippy Warnings:** 0 (zero tolerance)
- **SATD Comments:** 0 (implement or remove TODOs)

**Testing Requirements:**
- Unit tests for all public APIs
- Property-based tests for mathematical operations
- Integration tests for end-to-end workflows
- Benchmark tests for performance-critical paths

## Git Workflow

**Branch Policy:** Work directly on `main` branch (per CLAUDE.md in ~/.claude/)

**Commit Message Format:**
```
<type>: <subject>

<body>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code restructuring
- `test`: Add/update tests
- `docs`: Documentation
- `chore`: Maintenance (deps, config)

## Monitoring Ecosystem Updates

**Daily Checks (if actively developing):**
```bash
# Quick version check
cd ../trueno && git log --oneline -1 && grep "^version" Cargo.toml
cd ../aprender && git log --oneline -1 && grep "^version" Cargo.toml
```

**When to Update Realizar:**
- New trueno version with relevant features (vector ops, activations)
- Bug fixes in trueno that affect realizar
- Performance improvements in trueno SIMD/GPU backends
- New aprender primitives useful for inference

**Testing After Updates:**
1. `cargo clean` - Clear build artifacts
2. `cargo test --lib` - Verify all tests pass
3. `cargo clippy --lib -- -D warnings` - Zero warnings
4. `make quality-gates` - Full quality suite
5. Commit with version bump and rationale

## Architecture Principles

**1. Pure Rust from Scratch:**
- Build all ML components ourselves (parsers, transformer, quantization, tokenizer)
- Use trueno for compute primitives only
- HTTP server is swappable (axum default)

**2. Zero Unsafe in Public API:**
- All unsafe code isolated in trueno/aprender
- Realizar public API is 100% safe Rust

**3. Backend Agnostic:**
- Trueno handles SIMD/GPU dispatch automatically
- Fallback to scalar for unknown architectures
- WASM support via trueno

**4. Swappable HTTP Server:**
```rust
pub trait HttpServer {
    fn serve(&self, addr: &str) -> Result<()>;
}

// Currently: axum
// Future: hyper, actix-web, custom
```

## Performance Targets

**Inference Latency (1B models):**
- p50: <100ms
- p95: <200ms
- p99: <500ms

**Memory Usage:**
- Model: As loaded (no unnecessary copies)
- Runtime: <512MB overhead
- KV cache: Bounded and configurable

**Throughput:**
- Single request: Minimize latency
- Batch inference: Maximize throughput (Phase 2)

## Resources

**Documentation:**
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Safetensors Spec: https://github.com/huggingface/safetensors
- Trueno README: ../trueno/README.md
- Aprender README: ../aprender/README.md

**Related Projects:**
- [Trueno](https://github.com/paiml/trueno) - SIMD/GPU compute primitives
- [Aprender](https://github.com/paiml/aprender) - ML library in pure Rust
- [Renacer](https://github.com/paiml/renacer) - Profiling tools
- [paiml-mcp-agent-toolkit](https://github.com/paiml/paiml-mcp-agent-toolkit) - Quality gates
- [bashrs](https://github.com/paiml/bashrs) - Script enforcement

**Quality Tools:**
- `pmat`: Multi-dimensional analysis (TDG, complexity, SATD, coverage)
- `cargo-mutants`: Mutation testing
- `cargo-llvm-cov`: Code coverage
- `proptest`: Property-based testing

---

**Last Updated:** 2024-11-18
**Realizar Version:** 0.1.0 (Phase 1 Complete)
**Trueno Version:** 0.2.2
**Aprender Version:** 0.1.0
**TDG Score:** 100.0/100 (A+)
**Test Coverage:** 95.12%
**Total Tests:** 237 (195 unit + 42 property-based)
**Benchmarks:** 2 suites (tensor_ops, inference)
**Performance:** <1ms p50 for 5-token generation
**CLI Binary:** ✅ `realizar serve --demo`
**Latest Achievement:** Comprehensive inference benchmarks added
**Completed:** Weeks 1-8 (all major components implemented)
