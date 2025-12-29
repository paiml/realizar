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
- Status: v0.4.2 (2025-11-21) - SIMD attribute compliance, PMAT integration, zero warnings

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

**Trueno GPU Kernels (trueno-gpu crate):**
- `GemmKernel` - Matrix multiplication (naive, tiled, tensor core)
- `AttentionKernel` - FlashAttention-style tiled attention with online softmax
- `SoftmaxKernel` - Numerically stable softmax with warp shuffle
- `LayerNormKernel` - Fused layer normalization
- `QuantizeKernel` - Q4_K dequantization fused with matmul
- `Q5KKernel` - Q5_K dequantization
- `Q6KKernel` - Q6_K dequantization

## ⚠️ CRITICAL ANTI-PATTERN: NO HAND-ROLLED PTX

**NEVER write PTX strings directly in realizar code.**

### Why This Is Forbidden

1. **Trueno exists** - The `trueno-gpu` crate has tested, optimized kernels
2. **PTX is fragile** - Syntax errors, wrong compute capabilities, shared memory limits
3. **Trueno has trueno-explain** - Static analysis tool to find PTX bugs
4. **Maintenance burden** - Hand-rolled PTX must be updated for each GPU generation
5. **Testing** - Trueno kernels have property tests; hand-rolled PTX does not

### The Anti-Pattern (DO NOT DO THIS)

```rust
// ❌ WRONG - Hand-rolled PTX string in realizar
fn generate_attention_ptx(seq_len: u32, head_dim: u32) -> String {
    format!(r"
.version 8.0
.target sm_89
.address_size 64
.visible .entry attention(...) {{
    // 200 lines of hand-written PTX
}}
")
}
```

### The Correct Pattern (DO THIS)

```rust
// ✅ CORRECT - Use trueno-gpu kernels
use trueno_gpu::kernels::{AttentionKernel, Kernel};

let kernel = AttentionKernel::new(seq_len, head_dim)
    .with_causal()
    .with_tiles(64, 64);
let ptx = kernel.emit_ptx();
```

### If Trueno Is Missing a Kernel

1. **Add it to trueno-gpu** - Push to `../trueno`, not realizar
2. **Use the PTX builder API** - `PtxKernel::new().param().build(|ctx| {...})`
3. **Add property tests** - Ensure kernel works for all valid dimensions
4. **Use trueno-explain** - Run `trueno-explain bugs --kernel <name>` to find issues

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

## Hardware Environment

**GPU:** NVIDIA GeForce RTX 4090
- CUDA Compute Capability: 8.9 (Ada Lovelace)
- VRAM: 24GB GDDR6X
- Tensor Cores: 4th Gen (FP16/BF16/INT8)
- CUDA Cores: 16384
- Memory Bandwidth: 1008 GB/s

**⚠️ MANDATORY GPU TESTING:**
```bash
# ALWAYS run GPU tests - RTX 4090 is available
cargo test --lib --features cuda

# DO NOT use #[ignore] for GPU tests
# ALL GPU tests must execute, not be skipped
```

**Benchmark Targets (RTX 4090):**
- Ollama phi2:2.7b: ~225-266 tok/s (baseline)
- llama.cpp CUDA: ~256 tok/s
- Target: <1.25x gap to Ollama

**Development Iteration ("implement using pmat work"):**
1. `pmat analyze satd` - check SATD
2. `cargo clippy --lib --features cuda` - zero warnings
3. `cargo test --lib --features cuda` - **ALL tests including GPU**
4. Update spec with results

---

## CRITICAL: TUI Simulation Debugging (Probar-Style)

**⚠️ MANDATORY FOR ALL GPU/CUDA DEBUGGING**

When debugging GPU scheduler issues (CUDA vs wgpu parity, buffer management, kernel execution),
you MUST use TUI simulation workflow tests. This pattern was proven critical in PARITY-114 where
it detected a **state accumulation bug** that simple unit tests missed.

### Why TUI Simulation is Required

1. **Watches the Flow**: Step-by-step visualization of data through schedulers
2. **Catches State Bugs**: Sequential operations reveal accumulation/leakage issues
3. **Provides Diagnosis**: Automatic analysis of failure ratios (8x = accumulator bug, 4x = tile bug)
4. **Probar Alignment**: Matches probar's proven TUI testing methodology

### TUI Simulation Test Pattern

```rust
/// Example: TUI simulation for scheduler parity testing
#[test]
#[cfg(feature = "cuda")]
fn test_scheduler_parity_tui_simulation() {
    use realizar::gpu::{CudaScheduler, HybridScheduler};

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: Watch Data Flow Through Schedulers                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let mut sim = MatmulSimulator::new();

    // Define steps
    let step_init = sim.add_step("INIT", "Initialize test matrices");
    let step_cpu = sim.add_step("CPU", "Compute reference");
    let step_cuda = sim.add_step("CUDA", "Execute via CudaScheduler");
    let step_check = sim.add_step("CHECK", "Verify parity");

    // Execute with visual feedback
    sim.start_step(step_init);
    println!("  ◐ Initializing...");
    // ... setup code ...
    sim.complete_step(step_init, values, None);
    println!("  ● Complete");

    // Render final TUI frame
    println!("{}", sim.render_final());
}
```

### State Isolation Test Pattern

**CRITICAL**: Always test sequential operations to catch state bugs:

```rust
/// Test for state accumulation bugs
#[test]
fn test_scheduler_state_isolation() {
    let mut scheduler = CudaScheduler::new().unwrap();

    // Same operation twice - results MUST be identical
    let r1 = scheduler.matmul(&a, &b, m, k, n).unwrap();
    let r2 = scheduler.matmul(&a, &b, m, k, n).unwrap();

    assert_eq!(r1[0], r2[0], "State leak detected: first={}, second={}", r1[0], r2[0]);
}
```

### Running TUI Workflow Tests

```bash
# Run all GPU parity workflow tests with visual output
cargo test --test gpu_parity_workflow --features cuda -- --nocapture

# Specific TUI simulation test
cargo test --test gpu_parity_workflow test_parity_114_tui_simulation --features cuda -- --nocapture
```

### Failure Analysis Guide

| Ratio | Diagnosis | Check |
|-------|-----------|-------|
| 8x | Accumulator/tile loop bug | Inner loop iterations, FMA instruction |
| 4x | Partial tile accumulation | n_tiles calculation, tile bounds |
| 2x | Half iterations | Loop termination condition |
| Varies | State accumulation | Output buffer not cleared between calls |

### Bug Discovery: PARITY-114 Case Study

The TUI simulation discovered that **the same operation produced different results**:

```
Op 1: 4×64×8, expected 64, got 8
Op 3: 4×64×8, expected 64, got 16  ← DIFFERENT from Op 1!
```

This proved the output buffer was accumulating between calls rather than being cleared.
Simple unit tests would NOT have caught this - only sequential TUI simulation revealed it.

---

**Last Updated:** 2025-12-29
**Realizar Version:** 0.3.2 (2x Q8_0 SIMD Speedup, Aprender 0.20)
**GPU Spec Version:** v5.1.0 (QA Suite Complete + 95% Coverage)
**Trueno Version:** 0.4.2
**Aprender Version:** 0.20.1
**paiml-mcp-agent-toolkit Version:** v2.200.0 (with Known Defects Scorer, SATD Detector, Defect Analyzer)
**TDG Score:** 93.9/100 (A)
**Rust Project Score:** 137.9/134 (103%, Grade A+)
**Test Coverage:** 92.02% (region), 95.00% (function)
**Total Tests:** 2315 (all passing), 44 GPU-only ignored, 50 QA tests (QA-001 to QA-050)
**Mutation Score:** 100% on api.rs (18/18 viable mutants caught)
**Documentation:** 15.0/15 (100%) ✅ Perfect score!
**Known Defects:** 20.0/20 (100%) ✅ Perfect score!
**Dependency Health:** 10.5/12 (87.5%) - Modular feature flags
**Benchmarks:** 4 suites (tensor_ops, inference, cache, tokenizer)
**Examples:** 6 (inference, api_server, tokenization, safetensors_loading, model_cache, gguf_loading) - all verified working
**Performance:** <1ms p50 for 5-token generation (504µs measured)
**CLI Binary:** ✅ `realizar serve --demo` (65% coverage)
**Quality Improvements:**
  - Added workspace-level lints (unsafe_op_in_unsafe_fn, unreachable_pub, checked_conversions)
  - Created .clippy.toml for cognitive complexity thresholds
  - Fixed critical unwrap() in safetensors.rs (replaced with expect())
  - Updated to latest trueno v0.4.2 with SIMD attribute compliance and PMAT integration
  - Integrated paiml-mcp-agent-toolkit v2.200.0 (Known Defects, SATD, Defect Analysis)
**GPU Performance Parity (M29-M32):**
  - M29: Error Recovery (ErrorRecoveryStrategy, DegradationManager, FailureIsolator)
  - M30: Resource Management (ConnectionPool, ResourceLimiter, ResourceMonitor)
  - M31: Resilience (RetryPolicy, CircuitBreaker, BulkheadManager)
  - M32: Diagnostics (Logger, PhaseTimer, MemoryTracker, DiagnosticsCollector, DebugMode)
**Latest Achievement:** M32 Production Hardening complete, 92.30% coverage
**Completed:** Weeks 1-8 + GPU performance parity milestones M1-M32
