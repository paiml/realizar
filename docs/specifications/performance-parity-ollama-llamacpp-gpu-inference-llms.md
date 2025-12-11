# Performance Parity: Ollama & llama.cpp GPU Inference for LLMs

**Version:** 1.0.0
**Status:** Active
**Authors:** Pragmatic AI Labs
**Date:** 2024-12-11
**Work Item:** PERF-PARITY-001

## Abstract

This specification defines a comprehensive roadmap for achieving performance parity between Realizar and production-grade LLM inference engines (Ollama, llama.cpp) on GPU backends. It establishes KISS (Keep It Simple, Stupid) benchmarking methodology, improvement checklists, and quality assurance protocols aligned with Toyota Production System principles [1] and peer-reviewed benchmarking standards [2-11].

---

## 1. Executive Summary

### 1.1 Current State

| Runtime | Backend | p50 Latency | Throughput | Status |
|---------|---------|-------------|------------|--------|
| llama.cpp | CUDA | 162ms | 256 tok/s | Production |
| Ollama | CUDA | ~120ms | ~260 tok/s | Production |
| Realizar | CPU | ~500ms | ~2 tok/s | Educational |
| Realizar | WGPU | TBD | TBD | In Development |

### 1.2 Target State

| Milestone | Target | Metric |
|-----------|--------|--------|
| M1: CPU Parity | 10x improvement | 20 tok/s CPU |
| M2: WGPU Basic | GPU inference working | Any tok/s |
| M3: WGPU Parity | 50% of llama.cpp | 128 tok/s |
| M4: Full Parity | 90% of llama.cpp | 230+ tok/s |

---

## 2. Toyota Production System Framework

### 2.1 Guiding Principles [1]

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Measure actual performance, not theoretical |
| **Jidoka** | Stop immediately on quality defects |
| **Kaizen** | Continuous, incremental improvement |
| **Poka-yoke** | Error-proof benchmark infrastructure |
| **Heijunka** | Smooth, predictable performance |
| **Standardization** | Consistent methodology across all tests |

### 2.2 EXTREME TDD Integration

All improvements follow RED-GREEN-REFACTOR:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTREME TDD Cycle                             │
├─────────────────────────────────────────────────────────────────┤
│  RED    → Write failing benchmark test with target metric       │
│  GREEN  → Implement minimum code to achieve target              │
│  REFACTOR → Optimize while maintaining correctness              │
│  MEASURE → pmat analyze tdg && pmat analyze satd                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. KISS Benchmarking Architecture

### 3.1 Makefile Target Hierarchy

```
make bench-inference-all          # Master target: ALL inference benchmarks
    ├── make bench-pytorch-inference   # PyTorch vs APR MNIST comparison
    ├── make bench-cpu-inference       # All servers on CPU only
    ├── make bench-wgpu                # WGPU backend (no-op if unavailable)
    ├── make bench-gguf-gpu-inference  # GGUF models on GPU (realizar/ollama/llama.cpp)
    └── make bench-apr-gpu-inference   # APR format on GPU vs GGUF
```

### 3.2 Target Definitions

#### A. `make bench-inference-all`
```makefile
bench-inference-all: ## Run ALL inference benchmarks (master target)
	@echo "$(GREEN)Running complete inference benchmark suite...$(NC)"
	@$(MAKE) bench-pytorch-inference
	@$(MAKE) bench-cpu-inference
	@$(MAKE) bench-wgpu
	@$(MAKE) bench-gguf-gpu-inference
	@$(MAKE) bench-apr-gpu-inference
	@echo "$(GREEN)✅ All inference benchmarks complete$(NC)"
```

#### B. `make bench-pytorch-inference`
```makefile
bench-pytorch-inference: ## PyTorch vs APR MNIST benchmark
	@echo "$(GREEN)Running PyTorch vs APR MNIST comparison...$(NC)"
	cargo bench --bench apr_real
	@if command -v uv >/dev/null 2>&1; then \
		cd benches/comparative && uv run pytorch_baseline.py; \
	fi
```

#### C. `make bench-cpu-inference`
```makefile
bench-cpu-inference: ## All inference servers on CPU only
	@echo "$(GREEN)Running CPU-only inference benchmarks...$(NC)"
	cargo bench --bench gguf_real
	@./scripts/bench-cpu-matrix.sh
```

#### D. `make bench-wgpu`
```makefile
bench-wgpu: ## WGPU backend benchmark (no-op if unavailable)
	@echo "$(GREEN)Running WGPU inference benchmarks...$(NC)"
	@if cargo build --features gpu --quiet 2>/dev/null; then \
		cargo bench --bench gguf_real --features gpu; \
	else \
		echo "$(YELLOW)⚠️  WGPU not available, skipping$(NC)"; \
	fi
```

#### E. `make bench-gguf-gpu-inference`
```makefile
bench-gguf-gpu-inference: ## GGUF GPU inference: realizar/ollama/llama.cpp
	@echo "$(GREEN)Running GGUF GPU inference matrix...$(NC)"
	@./scripts/bench-gguf-gpu-matrix.sh
```

#### F. `make bench-apr-gpu-inference`
```makefile
bench-apr-gpu-inference: ## APR format GPU inference vs GGUF
	@echo "$(GREEN)Running APR vs GGUF GPU comparison...$(NC)"
	cargo bench --bench comparative --features gpu
```

### 3.3 Model Configuration

Models pulled from Hugging Face for reproducibility:

| Model | HF Repository | Size | Quantization |
|-------|---------------|------|--------------|
| Phi-2 | microsoft/phi-2 | 2.7B | Q4_K_M |
| Qwen2.5-Coder | Qwen/Qwen2.5-Coder-1.5B | 1.5B | Q4_K_M |
| DeepSeek-Coder | deepseek-ai/deepseek-coder-1.3b | 1.3B | Q4_K_M |
| TinyLlama | TinyLlama/TinyLlama-1.1B | 1.1B | Q4_K_M |

---

## 4. 25-Point Improvement Checklist

### Phase 1: Foundation (Points 1-5)

- [ ] **IMP-001**: Implement SIMD-accelerated Q4_K dequantization via Trueno
  - Target: 4x speedup over scalar dequantization
  - Test: `cargo test --lib test_q4k_simd_dequantize`
  - Metric: Dequant throughput > 10 GB/s

- [ ] **IMP-002**: Add memory-mapped weight streaming for large models
  - Target: Load 7B models with < 8GB RAM
  - Test: `cargo test --lib test_mmap_weight_streaming`
  - Metric: Memory footprint < model_size + 512MB

- [ ] **IMP-003**: Implement fused attention kernel (Q*K^T*V in single pass)
  - Target: 2x attention speedup
  - Test: `cargo test --lib test_fused_attention`
  - Metric: Attention latency < 10ms for 2K context

- [ ] **IMP-004**: Add KV cache with efficient memory layout
  - Target: 3x decode throughput
  - Test: `cargo test --lib test_kv_cache_layout`
  - Metric: KV cache hit rate > 99%

- [ ] **IMP-005**: Implement batch prefill for prompt processing
  - Target: 5x prefill speedup
  - Test: `cargo test --lib test_batch_prefill`
  - Metric: Prefill throughput > 1000 tok/s

### Phase 2: GPU Backend (Points 6-10)

- [ ] **IMP-006**: Integrate Trueno WGPU backend for matrix operations
  - Target: GPU-accelerated matmul
  - Test: `cargo test --features gpu test_wgpu_matmul`
  - Metric: Matmul TFLOPS > 1.0

- [ ] **IMP-007**: Implement GPU memory management with buffer pooling
  - Target: Zero allocation during inference
  - Test: `cargo test --features gpu test_gpu_buffer_pool`
  - Metric: GPU memory fragmentation < 5%

- [ ] **IMP-008**: Add asynchronous GPU kernel dispatch
  - Target: Hide kernel launch latency
  - Test: `cargo test --features gpu test_async_dispatch`
  - Metric: GPU utilization > 80%

- [ ] **IMP-009**: Implement WGPU compute shaders for transformer layers
  - Target: Full transformer on GPU
  - Test: `cargo test --features gpu test_transformer_gpu`
  - Metric: Layer latency < 5ms

- [ ] **IMP-010**: Add GPU-CPU overlap for streaming generation
  - Target: Continuous token output
  - Test: `cargo test --features gpu test_streaming_overlap`
  - Metric: Token latency jitter < 10%

### Phase 3: Quantization (Points 11-15)

- [ ] **IMP-011**: Implement Q4_K_M fused dequant+matmul kernel
  - Target: No intermediate F32 tensor
  - Test: `cargo test --lib test_fused_q4k_matmul`
  - Metric: Memory bandwidth > 500 GB/s

- [ ] **IMP-012**: Add Q5_K and Q6_K support
  - Target: Quality/speed tradeoff options
  - Test: `cargo test --lib test_q5k_q6k_dequant`
  - Metric: Quality loss < 1% vs F16

- [ ] **IMP-013**: Implement I-quant (integer-only matmul)
  - Target: INT8 inference path
  - Test: `cargo test --lib test_int8_matmul`
  - Metric: 2x throughput vs F32

- [ ] **IMP-014**: Add mixed-precision inference (Q4 weights, F16 activations)
  - Target: Balance quality and speed
  - Test: `cargo test --lib test_mixed_precision`
  - Metric: Perplexity within 0.5 of F16

- [ ] **IMP-015**: Implement weight clustering for cache efficiency
  - Target: Improved memory access patterns
  - Test: `cargo test --lib test_weight_clustering`
  - Metric: L2 cache hit rate > 90%

### Phase 4: Attention Optimization (Points 16-20)

- [ ] **IMP-016**: Implement Flash Attention algorithm
  - Target: O(N) memory for attention
  - Test: `cargo test --lib test_flash_attention`
  - Metric: 4K context with < 100MB attention memory

- [ ] **IMP-017**: Add Grouped-Query Attention (GQA) support
  - Target: Modern model architectures
  - Test: `cargo test --lib test_gqa_inference`
  - Metric: GQA models run correctly

- [ ] **IMP-018**: Implement Sliding Window Attention
  - Target: Long context support
  - Test: `cargo test --lib test_sliding_window`
  - Metric: 32K context viable

- [ ] **IMP-019**: Add ALiBi position encoding
  - Target: Alternative to RoPE
  - Test: `cargo test --lib test_alibi_positions`
  - Metric: ALiBi models run correctly

- [ ] **IMP-020**: Implement sparse attention patterns
  - Target: Efficient long-range attention
  - Test: `cargo test --lib test_sparse_attention`
  - Metric: 50% attention compute reduction

### Phase 5: System Integration (Points 21-25)

- [ ] **IMP-021**: Add continuous batching for concurrent requests
  - Target: Multi-user serving
  - Test: `cargo test --lib test_continuous_batching`
  - Metric: 10 concurrent requests with < 2x latency

- [ ] **IMP-022**: Implement speculative decoding
  - Target: 2x decode throughput
  - Test: `cargo test --lib test_speculative_decode`
  - Metric: Acceptance rate > 70%

- [ ] **IMP-023**: Add tensor parallelism for multi-GPU
  - Target: Scale beyond single GPU
  - Test: `cargo test --features multi-gpu test_tensor_parallel`
  - Metric: 1.8x speedup with 2 GPUs

- [ ] **IMP-024**: Implement model weight caching across requests
  - Target: Zero cold-start after first load
  - Test: `cargo test --lib test_weight_caching`
  - Metric: Warm-start latency < 10ms

- [ ] **IMP-025**: Add ONNX export for deployment portability
  - Target: Cross-platform inference
  - Test: `cargo test --lib test_onnx_export`
  - Metric: ONNX model produces identical output

---

## 5. 50-Point QA Checklist

### Section A: Correctness (Points 1-10)

- [ ] **QA-001**: Output matches llama.cpp for identical inputs (deterministic mode)
- [ ] **QA-002**: Tokenization produces identical token sequences
- [ ] **QA-003**: Attention scores match reference implementation within 1e-5
- [ ] **QA-004**: RoPE embeddings match reference within 1e-6
- [ ] **QA-005**: Softmax outputs sum to 1.0 within 1e-7
- [ ] **QA-006**: Layer norm outputs have unit variance within 1e-4
- [ ] **QA-007**: GELU activation matches PyTorch within 1e-5
- [ ] **QA-008**: SwiGLU activation matches reference within 1e-5
- [ ] **QA-009**: KV cache produces identical results to recomputation
- [ ] **QA-010**: Quantized inference matches F32 within acceptable tolerance

### Section B: Performance (Points 11-20)

- [ ] **QA-011**: Throughput regression < 5% between commits (CI gate)
- [ ] **QA-012**: Latency p99 < 2x p50 (no outliers)
- [ ] **QA-013**: Memory usage < 1.5x model size
- [ ] **QA-014**: GPU utilization > 70% during inference
- [ ] **QA-015**: No memory leaks over 1000 inference cycles
- [ ] **QA-016**: Cold start latency < 5 seconds for 7B model
- [ ] **QA-017**: Warm inference latency within 10% of steady state
- [ ] **QA-018**: Batch inference scales linearly to batch_size=8
- [ ] **QA-019**: Token generation rate stable (CV < 10%)
- [ ] **QA-020**: No performance degradation with context growth

### Section C: Reliability (Points 21-30)

- [ ] **QA-021**: Graceful handling of OOM conditions
- [ ] **QA-022**: Recovery from GPU timeout without crash
- [ ] **QA-023**: Correct behavior on malformed GGUF files
- [ ] **QA-024**: Correct behavior on truncated model files
- [ ] **QA-025**: No panic on empty input sequences
- [ ] **QA-026**: No panic on max context length exceeded
- [ ] **QA-027**: Correct handling of special tokens (BOS, EOS, PAD)
- [ ] **QA-028**: Thread-safe model sharing across inference threads
- [ ] **QA-029**: Deterministic output with fixed seed
- [ ] **QA-030**: Consistent results across CPU/GPU backends

### Section D: Benchmarking Infrastructure (Points 31-40)

- [ ] **QA-031**: CV-based stopping criterion implemented per [2]
- [ ] **QA-032**: Warmup iterations discard JIT/cache effects per [4]
- [ ] **QA-033**: Environment metadata captured per [8]
- [ ] **QA-034**: Outlier detection using MAD per [5]
- [ ] **QA-035**: Results include p50, p95, p99 latencies
- [ ] **QA-036**: Throughput measured in tok/s with variance
- [ ] **QA-037**: Benchmark results versioned and reproducible
- [ ] **QA-038**: Preflight checks validate server availability
- [ ] **QA-039**: Automatic model download from Hugging Face
- [ ] **QA-040**: JSON schema validation for benchmark results

### Section E: Integration (Points 41-50)

- [ ] **QA-041**: `make bench-inference-all` completes without error
- [ ] **QA-042**: `make bench-pytorch-inference` produces comparison report
- [ ] **QA-043**: `make bench-cpu-inference` tests all CPU backends
- [ ] **QA-044**: `make bench-wgpu` gracefully skips if unavailable
- [ ] **QA-045**: `make bench-gguf-gpu-inference` compares all runtimes
- [ ] **QA-046**: `make bench-apr-gpu-inference` produces format comparison
- [ ] **QA-047**: CI pipeline runs benchmarks on every PR
- [ ] **QA-048**: Benchmark results published to metrics dashboard
- [ ] **QA-049**: Historical trend analysis detects regressions
- [ ] **QA-050**: Documentation updated with latest benchmark results

---

## 6. PMAT Integration

### 6.1 Quality Metrics Tracking

```bash
# Run after each improvement implementation
pmat analyze tdg src/           # Technical Debt Grade
pmat analyze satd src/          # Self-Admitted Technical Debt
pmat analyze complexity src/    # Cyclomatic Complexity
```

### 6.2 Quality Gates

| Metric | Threshold | Action on Failure |
|--------|-----------|-------------------|
| TDG Score | ≥ 93.0 | Block merge |
| SATD Count | ≤ 5 | Require resolution |
| Max Complexity | ≤ 15 | Require refactor |
| Test Coverage | ≥ 95% | Block merge |
| Mutation Score | ≥ 80% | Warning |

### 6.3 Continuous Improvement Tracking

```rust
/// PMAT quality checkpoint (run in CI)
pub fn quality_checkpoint() -> QualityReport {
    QualityReport {
        tdg_score: pmat_tdg_score(),
        satd_count: pmat_satd_count(),
        complexity_max: pmat_max_complexity(),
        coverage: cargo_llvm_cov_percentage(),
        mutation_score: cargo_mutants_score(),
    }
}
```

---

## 7. Peer-Reviewed Publications

### 7.1 Benchmarking Methodology

[1] J. K. Liker, *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. New York, NY, USA: McGraw-Hill, 2004. ISBN: 978-0071392310

[2] T. Hoefler and R. Belli, "Scientific Benchmarking of Parallel Computing Systems: Twelve Ways to Tell the Masses When Reporting Performance Results," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC'15)*, Austin, TX, USA, 2015, pp. 1-12. DOI: 10.1145/2807591.2807644

[3] A. Georges, D. Buytaert, and L. Eeckhout, "Statistically Rigorous Java Performance Evaluation," in *Proceedings of the 22nd Annual ACM SIGPLAN Conference on Object-Oriented Programming Systems, Languages and Applications (OOPSLA '07)*, Montreal, Quebec, Canada, 2007, pp. 57-76. DOI: 10.1145/1297027.1297033

[4] T. Mytkowicz, A. Diwan, M. Hauswirth, and P. F. Sweeney, "Producing Wrong Data Without Doing Anything Obviously Wrong!" in *Proceedings of the 14th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS XIV)*, Washington, DC, USA, 2009, pp. 265-276. DOI: 10.1145/1508244.1508275

[5] P. J. Fleming and J. J. Wallace, "How Not to Lie with Statistics: The Correct Way to Summarize Benchmark Results," *Communications of the ACM*, vol. 29, no. 3, pp. 218-221, 1986. DOI: 10.1145/5666.5673

### 7.2 LLM Inference Optimization

[6] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. arXiv: 2205.14135

[7] W. Kwon, Z. Li, S. Zhuang, et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," in *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP '23)*, Koblenz, Germany, 2023, pp. 611-626. DOI: 10.1145/3600006.3613165

[8] J. Vitek and T. Kalibera, "Repeatability, Reproducibility, and Rigor in Systems Research," in *Proceedings of the Ninth ACM International Conference on Embedded Software (EMSOFT '11)*, Taipei, Taiwan, 2011, pp. 33-38. DOI: 10.1145/2038642.2038650

### 7.3 Quantization and Efficiency

[9] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. arXiv: 2208.07339

[10] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2023. arXiv: 2210.17323

### 7.4 Systems Performance

[11] C. Curtsinger and E. D. Berger, "Stabilizer: Statistically Sound Performance Evaluation," in *Proceedings of the 18th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS XVIII)*, Houston, TX, USA, 2013, pp. 219-228. DOI: 10.1145/2451116.2451141

---

## 8. Implementation Roadmap

### 8.1 Phase Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Implementation Roadmap                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 1: Foundation (IMP-001 to IMP-005)                                    │
│   └─ Priority: SIMD dequant, KV cache, batch prefill                       │
│                                                                             │
│ Phase 2: GPU Backend (IMP-006 to IMP-010)                                  │
│   └─ Priority: Trueno WGPU integration, compute shaders                    │
│                                                                             │
│ Phase 3: Quantization (IMP-011 to IMP-015)                                 │
│   └─ Priority: Fused Q4K matmul, mixed precision                           │
│                                                                             │
│ Phase 4: Attention (IMP-016 to IMP-020)                                    │
│   └─ Priority: Flash Attention, GQA support                                │
│                                                                             │
│ Phase 5: Integration (IMP-021 to IMP-025)                                  │
│   └─ Priority: Continuous batching, speculative decode                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Success Metrics

| Milestone | Criteria | Verification |
|-----------|----------|--------------|
| M1 Complete | 10 tok/s CPU | `make bench-cpu-inference` |
| M2 Complete | Any GPU inference | `make bench-wgpu` |
| M3 Complete | 128 tok/s GPU | `make bench-gguf-gpu-inference` |
| M4 Complete | 230 tok/s GPU | Full benchmark matrix |

---

## 9. Appendix A: Benchmark Scripts

### A.1 GPU Matrix Script

```bash
#!/bin/bash
# scripts/bench-gguf-gpu-matrix.sh
# Benchmarks realizar, ollama, and llama.cpp on GPU

set -euo pipefail

MODELS=(
    "phi-2-q4_k_m.gguf"
    "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    "deepseek-coder-1.3b-instruct-q4_k_m.gguf"
)

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          GGUF GPU Inference Benchmark Matrix                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Run benchmarks with CV-based stopping
cargo bench --bench external_matrix --features bench-http
```

### A.2 CPU Matrix Script

```bash
#!/bin/bash
# scripts/bench-cpu-matrix.sh
# Benchmarks all inference servers on CPU only

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          CPU-Only Inference Benchmark Matrix                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Run realizar CPU benchmarks
cargo bench --bench gguf_real

# Run llama.cpp CPU benchmark (if available)
if command -v llama-server &> /dev/null; then
    echo "Starting llama.cpp CPU server..."
    llama-server -m "$MODEL_PATH" --port 8083 -ngl 0 &
    sleep 5
    curl -s -X POST http://localhost:8083/completion \
        -d '{"prompt": "Hello", "n_predict": 30}'
    pkill llama-server || true
fi
```

---

## 10. Appendix B: Toyota Way Checklist

### Pre-Implementation (Genchi Genbutsu)

- [ ] Measured current baseline performance
- [ ] Identified bottleneck via profiling (not assumption)
- [ ] Verified external system state matches expectations
- [ ] Documented environment and configuration

### During Implementation (Jidoka)

- [ ] Tests written before code (RED phase)
- [ ] Minimal implementation to pass tests (GREEN phase)
- [ ] Quality metrics checked after each change
- [ ] Stopped immediately on quality regression

### Post-Implementation (Kaizen)

- [ ] Performance improvement measured and documented
- [ ] Technical debt reduced (TDG score improved)
- [ ] Learnings captured for future improvements
- [ ] Benchmark results added to historical tracking

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-11 | Initial specification |

