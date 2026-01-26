# Specification: PMAT Compliance & Coverage Siege

**Objective:** Achieve **95% Line Coverage** while maintaining **PMAT Compliance** (0 files > 2000 lines).
**Command:** `make coverage`

## 1. Compliance Constraints (PMAT)
*   **Total Decomposition (TDG):** No source file (`src/**/*.rs`) or test file (`tests/**/*.rs`) shall exceed **2000 lines**.
*   **Verification:** `pmat check` (or equivalent structural audit).

## 2. Coverage Targets (Gap Analysis)
Focus on the following high-value, low-coverage modules to bridge the gap from ~73% to 95%:

| Module Path | Strategy |
| :--- | :--- |
| **`gguf/inference/forward/single.rs`** | Exhaustive `GGUFBuilder` scenarios (batch=1, heads=1, multi-layer). |
| **`gguf/inference/generation.rs`** | Test full generation loops with various sampling parameters. |
| **`gguf/inference/matmul.rs`** | Verify all quantization permutations via builder. |
| **`gguf/loader.rs`** | Inject corruption errors (bad magic, version mismatch, missing tensors). |
| **`gguf/cuda/weights.rs`** | Use `MockExecutor` to verify weight layout/mapping without hardware. |

## 3. Execution Protocol
1.  **Run:** `make coverage` to identify current hot spots.
2.  **Refactor:** If a file > 2000 lines, split immediately (e.g., `part_01.rs`).
3.  **Test:** Add high-volume permutation tests using `proptest` and builders.
4.  **Repeat:** Loop until `Line Coverage >= 95.00%`.