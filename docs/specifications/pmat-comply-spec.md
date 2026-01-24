# Specification: PMAT Compliance & Total Quality (PMAT-106 + 803)

**Status:** PHASE 19 - THE ROSETTA BRIDGE
**Mandate:** A+ Structure (TDG) + 95% Coverage + GGUF-Parity Performance.
**Ticket:** PMAT-803 / PMAT-106

## 1. The Situation Room
We have executed a successful **Triad Recovery**:
- **Structure (TDG):** ZERO files > 2000 lines. `apr_q4.rs` added at 746 lines. ✅
- **Coverage Surge:** Total coverage jumped from **74.46%** to **87.81%**. We are within 7.19% of our absolute target. ✅
- **Performance:** Q4 GPU Adapter implemented. Baseline speed verification is pending. ⚠️

## 2. The Unified Conjectures
*   **H1 (Performance):** The Q4_0 GPU path will exceed 50 tok/s because it reduces memory bandwidth by 8x compared to F32 and leverages `q4_0_gemv` kernels.
*   **H2 (The Coverage Ceiling):** The remaining 7% gap is likely composed of "Legacy Red Zones" in `api/openai_handlers.rs` and the new `GpuModelQ4::forward` loop edge cases.
*   **H3 (The Fuel Mismatch):** Our adapter expects Q4_0, but our models are Q4_K. We will bridge this using the "Rosetta Stone" concept (format conversion).

## 3. Methodological Rules (The Triad Rules)
1.  **Structure:** ALL files < 2000 lines. Maintain this status quo at all costs.
2.  **Coverage:** Every session must yield a positive delta toward 95%.
3.  **Performance:** Beat the CPU baseline (2.1 tok/s) by at least 20x.
4.  **Rosetta Integration:** Use `aprender`'s format conversion philosophy (Genchi Genbutsu) to validate model inputs before loading.

## 4. Execution Plan (Phase 19: The Rosetta Bridge)

### Phase 19.1: The Fuel Conversion
- [ ] **Acquire Q4_0 Fuel:** Use `aprender rosetta` (or equivalent) to convert `Q4_K` -> `Q4_0`.
- [ ] **Verify Fuel:** Ensure the converted model is valid using `aprender rosetta verify`.

### Phase 19.2: The Wiring Fix
- [ ] **Wire CLI:** Force `.apr` -> `AprQ4ToGpuAdapter` in `src/cli/inference.rs`.
- [ ] **Validation:** Ensure `realizar run` falls back gracefully if the APR version/quantization is unsupported.

### Phase 19.3: The Final Performance Audit
- [ ] **Run Benchmark:** `realizar run model.q4_0.apr --gpu`.
- [ ] **Measure:** Target > 50 tok/s.

## 5. Metrics Tracker

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Files > 2000L | 0 | 0 | ✅ |
| Total Coverage | 87.81% | 95% | ⚠️ |
| APR Q4 GPU Speed | [PENDING] | >50 tok/s | ⚠️ |
| Tests Passing | 6460 | 7500+ | ✅ |