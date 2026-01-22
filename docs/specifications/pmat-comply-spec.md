# PMAT Comply A+ Specification

**Status:** IN PROGRESS
**Started:** 2025-01-22
**Target:** A+ grade (95%+ coverage, all files <2000 lines)
**Ticket:** PMAT-802

## Executive Summary

Realizar's codebase has grown to ~420K lines across ~200 files. PMAT comply reports **grade E** (54% file health) due to:
- 22 files >2000 lines
- 44 files >1000 lines
- 549 CB-020 warnings (unsafe blocks without SAFETY comments)
- Missing `.pmat-metrics.toml`
- Coverage at ~81% (target: 95%)

## Strategy: "Shattering"

Split large files into modules with tests in separate files (<2000 lines each).

**Pattern:**
```
src/module.rs (15,000 lines)
  ↓ shatter
src/module/
  mod.rs (main code)
  tests/
    mod.rs
    part_01.rs (<2000 lines)
    part_02.rs (<2000 lines)
    ...
```

## Progress Tracker

### Completed Shatters

| File | Original | Result | Tests Pass |
|------|----------|--------|------------|
| `http_client.rs` | 19,721 lines | mod.rs (1,016) + 12 test parts | ✅ 5996 |
| `quantize.rs` | 18,333 lines | mod.rs (7,908) + 6 test parts | ✅ 5996 |
| `gpu.rs` | 15,680 lines | mod.rs (8,719) + 4 test parts | ✅ 5996 |
| `layers.rs` | 15,292 lines | mod.rs (4,256) + 7 test parts | ✅ 5996 |

### In Progress

| File | Original | Status | Blocker |
|------|----------|--------|---------|
| `api.rs` | 14,618 lines | Split attempted | 212 test errors (private access) |
| `gguf_monolith.rs` | 54,882 lines | Split attempted | 165 test errors (private access, HashMap) |

### Remaining Large Files

| File | Lines | Priority |
|------|-------|----------|
| `gguf_monolith.rs` | 54,882 | **CRITICAL** - largest file |
| `api.rs` | 14,618 | High |
| `apr_transformer.rs` | 11,141 | High |
| `cuda/executor/tests.rs` | 10,880 | Medium |
| `bench.rs` | 8,722 | Medium |
| `gpu/mod.rs` | 8,719 | Low (tests split) |
| `apr.rs` | 7,970 | Medium |
| `quantize/mod.rs` | 7,908 | Low (tests split) |
| `cuda/executor/layer.rs` | 6,654 | Medium |
| `generate.rs` | 5,242 | Medium |
| `scheduler.rs` | 4,269 | Low |
| `layers/mod.rs` | 4,256 | Low (tests split) |
| `brick.rs` | 3,999 | Low |
| `cli.rs` | 3,841 | Low |
| `grammar.rs` | 3,588 | Low |
| `paged_kv.rs` | 3,375 | Low |
| `cuda/executor/quantized.rs` | 3,351 | Low |
| `cuda/executor/proptests.rs` | 3,105 | Low |

## Blockers & Five-Whys Analysis

### Blocker 1: Private Function Access in Shattered Tests

**Symptom:** `error[E0425]: cannot find function 'gpt2_unicode_to_byte' in module 'super'`

**Five Whys:**
1. Why? Tests use `super::private_fn()` to access private functions
2. Why? Tests were originally in same file, had access to private scope
3. Why? Rust module privacy - `pub(crate)` needed for cross-module test access
4. Why? Original design didn't anticipate module extraction
5. Why? Monolith grew organically without module boundaries

**Solution Options:**
- A) Make helper functions `pub(crate)` (preferred)
- B) Keep tests in monolith, only split main code
- C) Use `#[cfg(test)]` visibility tricks

### Blocker 2: Missing Imports in Test Files

**Symptom:** `error[E0433]: failed to resolve: use of undeclared type 'HashMap'`

**Root Cause:** Original tests relied on imports at top of monolith file.

**Solution:** Add `use std::collections::HashMap;` to each test part.

### Blocker 3: CB-020 Unsafe SAFETY Comments

**Count:** 549 warnings

**Pattern Needed:**
```rust
// SAFETY: [explanation of why this is safe]
unsafe { ... }
```

**Files Affected:**
- `cuda/executor/layer.rs`
- `cuda/executor/attention.rs`
- Many others in CUDA code

## Next Steps

1. **Revert api.rs split** - restore working state
2. **Fix api.rs private access** - make test helpers `pub(crate)`
3. **Re-attempt api.rs split**
4. **Address gguf_monolith.rs** - largest file, most complex
5. **Add CB-020 SAFETY comments** - 549 locations
6. **Create `.pmat-metrics.toml`**
7. **Run coverage** - verify 95% target

## Git Commits (This Session)

1. `bba9c4e` - refactor(http_client): Shatter into mod.rs + tests.rs
2. `361cc17` - refactor(http_client): Shatter tests.rs into 12 parts
3. `04fb089` - refactor(quantize): Shatter into mod.rs + 6 test parts
4. `8dbc6f1` - refactor(gpu): Shatter into mod.rs + 4 test parts
5. `4dbc86c` - refactor(layers): Shatter into mod.rs + 7 test parts

## Metrics Before/After

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Files >2000 lines | 23 | 22 | 0 |
| Files >1000 lines | 16 | ~44 | <10 |
| File Health Grade | E (58%) | E (54%) | A+ (95%) |
| Test Coverage | 81% | ~81% | 95% |
| CB-020 Warnings | 546 | 549 | 0 |
| All Tests Pass | ✅ | ✅ | ✅ |

## Constraints

- **NEVER push broken code** - other team depends on trunk
- **Push frequently** - incremental progress visible
- **Maintain test pass** - all 5996 tests must pass
- **Track with pmat work** - reference PMAT-802 in commits
- **Five-whys for blockers** - root cause analysis

## Dependencies

- `trueno` - SIMD/GPU compute (path dependency)
- `trueno-gpu` - CUDA kernels
- `pmat` - quality tooling

## Command Reference

```bash
# Check compliance
pmat comply check

# Run tests
cargo test --lib

# Check coverage
make coverage

# Find large files
wc -l src/**/*.rs | sort -rn | head -30

# Find CB-020 violations
pmat comply check 2>&1 | grep CB-020
```
