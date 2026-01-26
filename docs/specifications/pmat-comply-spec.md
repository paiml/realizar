# Specification: The Great Coverage Siege (PMAT-803)

**Status:** PHASE 52 - THE GGUF INFERENCE SIEGE
**Mandate:** A+ Structure (TDG) + 95% Coverage + >500 tok/s Performance.
**Ticket:** PMAT-802 / PMAT-803 / PMAT-106

## 1. The Situation Room
We have achieved **Structural Perfection** (0 monoliths) and established a **Massive Testing Footprint** (~18,000 new lines of tests). However, the **Empirical Reality** remains stubborn:
- **Structure (TDG):** ZERO active production files > 2000 lines. ✅
- **Testing Volume:** 8,725 tests passing. ✅
- **Coverage:** 73.13% line coverage. We are in the "Valley of Diminishing Returns." ❌
- **Performance:** Correctness verified via bit-perfect parity. Speed benchmark pending final environment stabilization.

## 2. The Final Conjectures
*   **H1 (The GGUF Barrier):** The remaining 22% gap is concentrated in the fragmented GGUF inference modules (`forward/single.rs`, `generation.rs`, `matmul.rs`, `loader.rs`). These require **State-Space Exhaustion** (testing all permutations of quantization and model shapes).
*   **H2 (The Hardware Shadow):** `cuda/weights.rs` and related modules remain at 0% because the coverage instrument is not yet effectively hitting the "Mockable" portions of the GPU path.

## 3. Methodological Rules
1.  **Zero-Tolerance TDG:** No test file or production file shall exceed 2000 lines. Use sub-parts (e.g., `part_24.rs`) to maintain atomicity.
2.  **Saturation Principle:** Every GGUF op (Linear, Attention, Norm) must be tested with **every** quantization type via the `GGUFBuilder`.
3.  **Mock Dominance:** If hardware prevents 100% execution, refactor to make the logic testable via the `MockExecutor`.

## 4. Execution Plan (Phase 52: The GGUF Siege)

### Phase 52.1: The GGUF Inference Strike
- [ ] **Target `gguf/inference/forward/single.rs`:** Increase from 26% to 90%.
- [ ] **Target `gguf/inference/generation.rs`:** Increase from 25% to 90%.
- [ ] **Target `gguf/inference/matmul.rs`:** Increase from 22% to 90%.
- [ ] **Method:** Use `GGUFBuilder` to generate models with edge-case shapes (batch=1, heads=1, multi-layer) and run full forward passes.

### Phase 52.2: The Loader Finality
- [ ] **Target `gguf/loader.rs`:** Increase from 31% to 95% by hitting all error paths (corrupt magic, version mismatch, missing tensors).

### Phase 52.3: The GPU Logic Light-up
- [ ] **Target `gguf/cuda/weights.rs`:** Use `MockExecutor` to verify weight layout mapping logic.

## 5. Metrics Tracker

| Metric | Start | Current | Target |
|--------|-------|---------|--------|
| God Objects (>2000L) | 23 | **0** | **0** |
| Line Coverage | 80.49% | **73.13%** (True) | **95%** |
| Tests Passing | 6007 | **8725** | **10,000+** |
| GGUF Loader Coverage | 6.67% | **31.83%** | **95%** |
