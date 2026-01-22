# Specification: PMAT Compliance & Codebase Shattering (PMAT-802)

**Status:** IN PROGRESS
**Started:** 2025-01-22
**Target:** A+ grade (95%+ coverage, all files <2000 lines)
**Ticket:** PMAT-802

## 1. Problem Situation
The `realizar` codebase (420K lines) is currently in a state of **Entropic Decay (Grade E)**.
- **Complexity:** 22 files >2000 lines (violating modularity).
- **Safety:** 549 CB-020 warnings (undocumented `unsafe` blocks).
- **Opacity:** Coverage is stuck at 81% (Target: 95%).
- **Metric:** `pmat comply` reports 54% file health.

This structure prevents effective maintenance and hides bugs in "God Objects" like `gguf_monolith.rs` (54k lines).

## 2. Conjectures (Hypotheses)
*   **H1 (The Shattering Hypothesis):** We can decompose "God Objects" (e.g., `gguf_monolith.rs`, `api.rs`) into atomic sub-modules (`mod.rs` + `tests/`) without breaking the 5996 existing tests or reducing performance.
    *   *Prediction:* File health > 90%, compile times reduce by >10%.
*   **H2 (The Safety Documentation Hypothesis):** The 549 CB-020 warnings represent latent memory safety risks.
    *   *Prediction:* Forcing SAFETY comments will reveal at least one actual soundness bug (to be fixed).

## 3. Methodological Rules
1.  **Atomic Shattering:** Split files >2000 lines using the pattern: `src/module.rs` -> `src/module/mod.rs` + `src/module/tests/*.rs`.
2.  **Invariant Preservation:** `cargo test` must PASS after every commit. **Never push broken code.**
3.  **Coverage Ratchet:** Coverage must NOT decrease. Use `explode` strategies if needed to expose hidden paths.
4.  **Traceability:** Every commit must reference `PMAT-802`.
5.  **Root Cause Analysis:** Use "Five Whys" for any blocker (e.g., `api.rs` private access).

## 4. Falsification Protocol
*   **F-SHATTER-01:** If shattering `gguf_monolith.rs` introduces circular dependencies that require >3 days to resolve, **H1 is falsified** for that module. We must pivot to "Internal Partitioning" (inline modules) instead of file extraction.
*   **F-SAFE-01:** If adding SAFETY comments takes >10 minutes per block due to unknown invariants, the code is deemed **Legacy Unsafe** and must be ticketed for rewrite (refuting the assumption it is just "undocumented").

## 5. Execution Plan (The Experiment)

### Phase 1: The "Easy" Shatters (Verified)
- [x] `http_client.rs` -> Shattered (12 parts)
- [x] `quantize.rs` -> Shattered (6 parts)
- [x] `gpu.rs` -> Shattered (4 parts)
- [x] `layers.rs` -> Shattered (7 parts)

### Phase 2: The "Hard" Shatters (Current Focus)
- [ ] **`api.rs` (14k lines):**
    - *Blocker:* Private function access in tests.
    - *Solution:* Expose helpers as `pub(crate)` or move tests to `tests/api_integration.rs`.
- [ ] **`gguf_monolith.rs` (54k lines):**
    - *Status:* CRITICAL. Largest file.
    - *Action:* Break into `gguf/loader.rs`, `gguf/tensor.rs`, `gguf/metadata.rs`.

### Phase 3: Safety & Metrics
- [ ] Fix 549 CB-020 warnings (Add `// SAFETY: ...`).
- [ ] Create `.pmat-metrics.toml`.
- [ ] Verify 95% Coverage.

## 6. Current Metrics & Progress

| Metric | Start | Current | Target |
|--------|-------|---------|--------|
| Files >2000 lines | 23 | 22 | **0** |
| File Health | 58% | 54% | **95%** |
| Tests Passing | 5996 | 5996 | **5996+** |

## 7. Command Reference
```bash
# Verify Health
pmat comply check

# Verify Invariants
cargo test --lib
make coverage
```