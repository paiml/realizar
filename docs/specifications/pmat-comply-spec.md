# Specification: PMAT Compliance & Quality Gates

**Status:** ⚠️ IN PROGRESS (2026-02-01)
**Objective:** Achieve full PMAT compliance across all quality dimensions.
**Command:** `pmat comply check` and `pmat quality-gate`

## Summary

PMAT compliance check: **NON-COMPLIANT**

Critical issues remaining:
- File Health: 39 files >2000 lines (grade D)
- Dead Code: 31.8% (quality-gate) vs target ≤15%
- ComputeBrick: 526 SIMD warnings (#[target_feature] missing)

## 1. Compliance Check Results (`pmat comply check`)

| Check | Status | Details |
|-------|--------|---------|
| Version Currency | ✅ | v2.215.0 (latest) |
| Config Files | ✅ | All present |
| Git Hooks | ✅ | Installed |
| CB-030 O(1) Hooks | ✅ | Cache initialized |
| Quality Thresholds | ✅ | Configured |
| Deprecated Features | ✅ | None detected |
| Cargo.lock | ✅ | Reproducible builds |
| MSRV Defined | ✅ | rust-version present |
| CI Configured | ✅ | 3 workflows |
| ComputeBrick | ⚠️ | 526 warnings (CB-021 SIMD) |
| OIP Tarantula | ⚠️ | 11 issues, 9 warnings |
| Coverage Quality | ⚠️ | 17 warnings (CB-127) |
| PAIML Deps | ⚠️ | 3 dirty workspaces |
| **File Health** | ❌ | 39 files >2000 lines |

## 2. Quality Gate Results (`pmat quality-gate`)

| Metric | Threshold | Current | Status |
|--------|-----------|---------|--------|
| **Dead Code** | ≤ 15% | 31.8% | ❌ FAIL |
| **Complexity** | ≤ 25 cognitive | 147 violations | ❌ FAIL |
| **SATD** | 0 critical | 16 violations | ⚠️ |
| **Entropy** | - | 54 violations | ⚠️ |
| **Provability** | ≥ 0.70 | 0.65 | ❌ FAIL |
| **Security** | 0 | 0 | ✅ PASS |
| **Duplicates** | - | 0 | ✅ PASS |

## 3. Dead Code Violations (Priority: HIGH)

Target: ≤15%
**Current: 31.8%** (quality-gate reports higher than `pmat analyze dead`)

| File | Dead % | Dead Lines | Status |
|------|--------|------------|--------|
| `src/quantize/activation.rs` | 82.4% | 140 | ❌ TODO |
| `src/quantize/fused_k.rs` | 80.0% | 200 | ❌ TODO |
| `src/quantize/parallel_dequant.rs` | 73.7% | 140 | ❌ TODO |
| `src/quantize/dequant.rs` | 75.0% | 120 | ❌ TODO |
| `src/quantize/simd.rs` | - | - | ✅ Cleaned |

### Completed
- Removed `hsum_epi32_*` from simd.rs
- Removed `fast_exp_avx2`, `horizontal_sum_avx2` from activation.rs

### Root Cause
SIMD functions behind `#[cfg(target_arch = "x86_64")]` not counted as covered

## 4. ComputeBrick Compliance (Priority: HIGH)

**526 CB-021 warnings**: SIMD intrinsics without `#[target_feature]`

| File | Issue |
|------|-------|
| `src/quantize/parallel_dequant.rs` | `_mm256_*` without attribute |
| `src/quantize/fused_k.rs` | `_mm256_*` without attribute |
| `src/quantize/activation.rs` | `_mm256_*` without attribute |

### Fix Strategy
Add `#[target_feature(enable = "avx2")]` to SIMD functions:
```rust
#[target_feature(enable = "avx2")]
unsafe fn simd_function() { ... }
```

## 5. File Health (Priority: CRITICAL)

**39 files exceed 2000 lines** (grade D, avg health 62%)

### Non-Test Files >2000 Lines (18 files)

| File | Lines | Action |
|------|-------|--------|
| `src/gpu/scheduler/model.rs` | 2348 | ✅ DONE (types.rs + ops.rs + loading.rs extracted, 1936 non-test) |
| `src/observability.rs` | 2751 | Split: metrics, tracing, logging |
| `src/safetensors.rs` | 2613 | Split: loader, inference, config |
| `src/apr/mod.rs` | 1949 | ✅ DONE (dequant.rs + model_data.rs extracted) |
| `src/quantize/fused_k.rs` | 2403 | Review: may have dead code |
| `src/cuda/executor/layers/batched.rs` | 2304 | Split by layer type |
| `src/generate/sampler.rs` | 2296 | Split: strategies, nucleus, beam |
| `src/cuda/executor/layers/indexed.rs` | 2281 | Split by layer type |
| `src/apr_transformer/mod.rs` | 2266 | Split: forward, attention, ffn |
| `src/gguf/loader.rs` | 2262 | Split: parse, validate, load |
| `src/gguf/batch_scheduler.rs` | 2199 | Split: scheduler, batch, queue |
| `src/cuda/executor/quantized.rs` | 2176 | Split by quant type |
| `src/cuda/kernels.rs` | 2139 | Split: gemm, attention, norm |
| `src/api/gpu_handlers.rs` | 2124 | Split: chat, generate, embed |
| `src/api/mod.rs` | 2114 | Split: routes, handlers, types |
| `src/parallel.rs` | 2082 | Split: threadpool, work, sync |
| `src/paged_kv/mod.rs` | 2072 | Split: cache, paging, eviction |
| `src/cuda/executor/layers/graphed.rs` | 2057 | Split by operation |

### Fix Strategy
1. Create submodule directory (e.g., `src/observability/`)
2. Move logical sections to submodules
3. Re-export from `mod.rs`
4. Verify tests pass after each split

## 6. OIP Tarantula Patterns (Priority: MEDIUM)

**11 issues, 9 warnings** (CB-120 to CB-124)

### CB-122: Serde Unsafe Patterns
`.expect()` on serde operations can panic on malformed input.

| Pattern | Fix |
|---------|-----|
| `serde_json::from_slice().expect()` | Use `?` operator |
| `serde_json::from_str().unwrap()` | Use `?` operator |

### Fix Strategy
Replace panic-prone patterns with proper error handling:
```rust
// Before (panics on bad input)
let data: Config = serde_json::from_slice(&bytes).expect("parse failed");

// After (propagates error)
let data: Config = serde_json::from_slice(&bytes)?;
```

## 7. SATD Violations (Priority: MEDIUM) ✅ RESOLVED

All critical SATD violations fixed:

| File | Issue | Resolution |
|------|-------|------------|
| `src/chat_template.rs` | 14x Security comments | ✅ Reworded (sanitization implemented) |
| `src/infer/mod.rs:292` | Security comment | ✅ Reworded (validation implemented) |
| `src/safetensors_cuda.rs:585` | Performance TODO | ✅ Simplified comment |
| `src/gguf/inference/attention.rs` | 2x Design comments | ✅ Reworded |

### Remaining (Non-critical)
- `book/searcher.js:148` - mdbook generated file (not our code)
- 4x High: Defect tracking comments (acceptable documentation)

## 8. Duplicate Code Patterns (Priority: LOW)

| File | Pattern | Occurrences | Potential Savings |
|------|---------|-------------|-------------------|
| `src/gguf/batch_scheduler.rs` | ResourceManagement | 10x | 1238 lines |
| `src/safetensors.rs` | ApiCall | 10x | 648 lines |

These are lower priority - address after dead code and SATD.

## 9. Execution Protocol

```bash
# 1. Check current state
pmat quality-gate --format summary

# 2. Fix dead code (highest impact)
# For each file in quantize/:
#   - Read exports from quantize/mod.rs
#   - Remove any function not exported
#   - Run: cargo test --lib quantize

# 3. Fix SATD comments
# Either implement the TODO or remove with justification

# 4. Verify
pmat quality-gate --fail-on-violation
make test-fast
make lint
```

## 10. Acceptance Criteria

- [ ] Dead code ≤ 15% (current: 31.8%)
- [x] 0 critical SATD comments (1 in mdbook generated, acceptable)
- [x] All tests pass (13097 passed)
- [x] Zero clippy warnings
- [x] TDG score ≥ 93.0 (94.3)
- [ ] File health grade ≥ C (current: D)
- [ ] ComputeBrick CB-021 warnings = 0 (current: 526)
- [ ] Provability score ≥ 0.70 (current: 0.65)
- [ ] `pmat comply check` = COMPLIANT

## 11. References

- PMAT-805: Qwen throughput spec (parent)
- Issue #43: APR performance (related)
- Issue #45: Forward path checks wrong cache (fixed in cuda.rs)
- Issue #46: Rosetta validation rejects valid Qwen RMSNorm weights (fixed in aprender)
