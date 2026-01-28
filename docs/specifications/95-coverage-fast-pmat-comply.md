# Specification: Fast O(1) Coverage with PMAT Compliance

**Document ID:** SPEC-COV-95
**Version:** 1.3.0
**Status:** ACTIVE
**Methodology:** The Toyota Way (14 Principles) + Popperian Falsification
**Target:** 95% Production Code Coverage in <10 minutes (Full), O(1) Incremental

---

## 1. Executive Summary

This specification defines a high-integrity coverage measurement system built on the philosophy that **"The right process will produce the right results"** (Toyota Principle 8). It combines rigorous scientific testing with lean manufacturing principles to create a system that is:

1.  **O(1) Incremental Corroboration** - Achieving **One-Piece Flow** in testing, verifying modules in isolation to reduce batch sizes.
2.  **Visual Management (Andon)** - Immediate, inescapable feedback when quality standards are not met.
3.  **Falsification-First QA** - A scientific approach to quality where tests attempt to refute, not verify, the codebase.
4.  **Respect for People** - The system serves the developer by automating the mundane (Jidoka) and providing rapid feedback to prevent frustration.
5.  **Standardized Work** - Defining the current best practice for testing to serve as a baseline for **Kaizen** (Continuous Improvement).

---

## 2. Theoretical Foundation: The Toyota Way

This system explicitly implements specific principles from the 14 Principles of the Toyota Way:

### 2.1 Principle 5: Jidoka (Build Quality In)
**"Stop the line to fix problems. Quality takes precedence over production schedules."**
*   **Implementation:** The `coverage-95` and `pmat quality-gate` commands act as an **Andon Cord**. If coverage drops or a mutant survives, the CI pipeline (the assembly line) stops immediately. No code flows downstream until the defect is corrected.

### 2.2 Principle 2: Continuous Flow
**"Create continuous process flow to bring problems to the surface."**
*   **Implementation:** By breaking the monolith coverage run into O(1) modular targets (`coverage-core`, `coverage-gguf`), we reduce "inventory" (unverified code) and waiting time. This exposes integration issues instantly rather than hiding them in long batch processes.

### 2.3 Principle 12: Genchi Genbutsu
**"Go and see for yourself to thoroughly understand the situation."**
*   **Implementation:** We do not rely solely on aggregate numbers. We use `cargo llvm-cov report --html` to generate visual maps of the code. Developers are expected to "walk the floor" of the report to see *which* specific lines are uncovered, rather than guessing based on a percentage.

### 2.4 Principle 14: Hansei (Reflection) & Kaizen
**"Become a learning organization through relentless reflection and continuous improvement."**
*   **Implementation:** A coverage drop is not just an error; it is an opportunity for **Hansei**. We ask "Why?" five times to understand the root cause of the missing test case, rather than simply patching it.

---

## 3. Architecture

### 3.1 O(1) Incremental Performance (One-Piece Flow)

True O(1) coverage for a *full report* is impossible. We define **O(1) Incremental Corroboration** as:

> The time required to corroborate the quality of a specific module `M` is independent of the size of the rest of the codebase `U - M`.

This is achieved via:
1.  **Modular Isolation**: `make coverage-core` runs only core tests.
2.  **Incremental Compilation**: Only recompile changed modules.
3.  **Profraw Accumulation**: Data is accumulated without merging until the report step.

### 3.2 Module Hierarchy (Heijunka - Leveling)

To level the workload and prevent bottlenecks:

```
coverage (default)     [Composite]
├── coverage-core      (~90s)  - quantize, layers, generate, infer
├── coverage-gguf      (~60s)  - GGUF parser and loader
├── coverage-api       (~60s)  - HTTP API and CLI
└── coverage-cuda      (~120s) - GPU/CUDA operations (Single-threaded)

coverage-all           (~10m)  - All modules combined (Full Report)
coverage-95            (O(1))  - Threshold enforcement (The Quality Gate)
```

### 3.3 Exclusion Strategy (Standardized Work)

Exclusions are explicitly defined to prevent "Muda" (Waste) in the metrics:

| Pattern | Rationale | Toyota Principle |
|---------|-----------|------------------|
| `/tests/` | Test code is not the product | Principle 1 (Long-term philosophy) |
| `/benches/` | Measurement tools, not production parts | - |
| `/examples/` | Documentation, not shipped logic | - |
| `trueno/` | Supplier parts (External dependency) | - |
| `main.rs` | **WARNING:** Must remain a hollow shell. Logic here hides from Jidoka. | Principle 5 (Quality) |

---

## 4. Performance Targets

### 4.1 Time Budgets (Takt Time)

| Target | Takt Time | Tolerance | Falsification |
|--------|-----------|-----------|---------------|
| `make coverage` | <5 min | +30s | `time make coverage` > 330s |
| `make coverage-core` | <2 min | +30s | `time make coverage-core` > 150s |
| Report generation | <30s | +10s | Report step > 40s |

### 4.2 Quality Thresholds

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Line Coverage | 85% | 95% | 98% |
| Function Coverage | 90% | 95% | 99% |
| Region Coverage | 80% | 90% | 95% |
| **Rigor Score** | 60% | 85% | 95% |

*Note: "Perfection Score" has been renamed to "Rigor Score" to reflect that Kaizen is a never-ending journey.*

---

## 5. The 100-Point Falsification Checklist

Each item is a **falsifiable hypothesis**. The item PASSES if no falsification is found.

### Section A: Build System & Flow (20 points)

| # | Hypothesis | Falsification Condition | Points |
|---|------------|------------------------|--------|
| A1 | `make coverage` meets Takt Time (<5 min) | Execution time ≥300s | 2 |
| A2 | `make coverage-core` enables One-Piece Flow | Depends on other coverage targets | 2 |
| A3 | `make coverage-gguf` enables One-Piece Flow | Depends on other coverage targets | 2 |
| A4 | `make coverage-api` enables One-Piece Flow | Depends on other coverage targets | 2 |
| A5 | `make coverage-cuda` enables One-Piece Flow | Depends on other coverage targets | 2 |
| A6 | Incremental builds eliminate waiting (Muda) | Incremental time ≥ clean time | 2 |
| A7 | `--no-report` accumulates data (Batch reduction) | Data lost between runs | 2 |
| A8 | Report generation is <30s | Report time ≥30s | 2 |
| A9 | Makefile has no recursive dependency loops | `make -n` shows circular deps | 2 |
| A10 | System works offline (Self-reliance) | Fails without internet | 2 |

### Section B: Coverage Accuracy (Genchi Genbutsu) (20 points)

| # | Hypothesis | Falsification Condition | Points |
|---|------------|------------------------|--------|
| B1 | Test files are excluded (No noise) | `/tests/` appears in report | 2 |
| B2 | Benchmark files are excluded | `/benches/` appears in report | 2 |
| B3 | Example files are excluded | `/examples/` appears in report | 2 |
| B4 | main.rs is excluded (Hollow Shell) | `main.rs` has non-zero regions | 2 |
| B5 | External deps (trueno) are excluded | `trueno/` appears in report | 2 |
| B6 | All production .rs files are included | Production file missing from report | 2 |
| B7 | Region count matches source lines | Region count = 0 for non-empty file | 2 |
| B8 | Function coverage ≤ 100% | Function coverage > 100% | 2 |
| B9 | Line coverage ≤ region coverage | Line > region (impossible) | 2 |
| B10 | Dead code has 0% coverage | Unreachable code shows >0% | 2 |

### Section C: Test Quality (Built-In Quality) (15 points)

| # | Hypothesis | Falsification Condition | Points |
|---|------------|------------------------|--------|
| C1 | All tests pass before coverage (Jidoka) | Any test failure in coverage run | 2 |
| C2 | No tests are flaky | Same test passes then fails | 2 |
| C3 | Tests are deterministic | Different results with same inputs | 2 |
| C4 | Property tests use sufficient cases | PROPTEST_CASES < 3 | 2 |
| C5 | GPU tests run single-threaded (Safety) | CUDA tests with --test-threads > 1 | 2 |
| C6 | No tests depend on execution order | Tests fail when run in isolation | 2 |
| C7 | Tests clean up resources (5S) | Resource leaks detected | 1 |
| C8 | Tests have timeouts | Any test runs >60s without timeout | 2 |

### Section D: Visual Management (Andon) (15 points)

| # | Hypothesis | Falsification Condition | Points |
|---|------------|------------------------|--------|
| D1 | `coverage-95` stops the line on failure | Exit code 0 when coverage < 95% | 3 |
| D2 | `pmat popper-score` >= 60% | Popper score < 60% | 3 |
| D3 | `pmat quality-gate` signals explicitly | Silent failure / unclear error | 3 |
| D4 | `pmat comply check` shows 100% compliance | Non-compliant status | 3 |
| D5 | `pmat rigor-score` > 160/200 | Rigor score ≤ 160 | 3 |
| D6 | Configuration is standard (`.pmat-gates.toml`) | `coverage_threshold < 95.0` | 3 |

### Section E: Reporting & Artifacts (15 points)

| # | Hypothesis | Falsification Condition | Points |
|---|------------|------------------------|--------|
| E1 | HTML report is generated (Visual Control) | No file at target/coverage/html/index.html | 3 |
| E2 | LCOV report is generated (Standardization) | No file at target/coverage/lcov.info | 3 |
| E3 | Summary shows TOTAL line | No TOTAL in summary output | 3 |
| E4 | Per-file breakdown available | Cannot identify low-coverage files | 3 |
| E5 | PMAT report is generated | `pmat report` fails | 3 |

### Section F: Exclusion Integrity (5 points)

| # | Hypothesis | Falsification Condition | Points |
|---|------------|------------------------|--------|
| F1 | No new ignores without review | pmat-ignore added without PR comment | 1 |
| F2 | Ignores don't hide bugs | Bug found in ignored code | 2 |
| F3 | Exclusions match Makefile regex | File excluded but not in regex | 2 |

### Section G: Mutation & Severe Testing (10 points)

| # | Hypothesis | Falsification Condition | Points |
|---|------------|------------------------|--------|
| G1 | Mutation tests run (`make mutants`) | `make mutants` fails to run | 2 |
| G2 | Mutants are detected (killed) | Mutation score < 60% (if measured) | 2 |
| G3 | Zero coverage triggers alert | 0% coverage treated as passing | 2 |
| G4 | 100% coverage triggers audit | 100% coverage treated as passing without review | 2 |
| G5 | PMAT tools are operational | `pmat diagnose` fails | 2 |

---

## 6. Implementation

### 6.1 Makefile Targets (Standardized Work)

```makefile
# STRICT exclusions: Only count realizar/src/*.rs
COV_EXCLUDE := --ignore-filename-regex='(trueno/|/tests/|_tests\.rs|test_|tui\.rs|bench_viz\.rs|viz\.rs|main\.rs|/benches/|/examples/)'

coverage-core: ## Coverage: core modules only (~90s)
	@cargo llvm-cov test --lib --no-report $(COV_EXCLUDE) \
		-- --test-threads=8 \
		--skip gguf:: --skip api:: --skip cli:: --skip cuda:: --skip gpu:: --skip bench::

coverage: ## DEFAULT: Core coverage with report (Andon Visual)
	$(MAKE) --no-print-directory coverage-core
	cargo llvm-cov report --html --output-dir target/coverage/html $(COV_EXCLUDE)
	cargo llvm-cov report --summary-only $(COV_EXCLUDE) | grep -E "^TOTAL"

coverage-95: ## Enforce 95% threshold (The Gate)
	@COVERAGE=$$(cargo llvm-cov report --summary-only $(COV_EXCLUDE) | grep "TOTAL" | awk '{print $$10}' | sed 's/%//'); \
	if [ -z "$$COVERAGE" ]; then echo "❌ No coverage data"; exit 1; fi; \
	RESULT=$$(echo "$$COVERAGE >= 95" | bc -l); \
	if [ "$$RESULT" = "1" ]; then echo "✅ $$COVERAGE% >= 95%"; else echo "❌ $$COVERAGE% < 95% (STOP THE LINE)"; exit 1; fi
```

### 6.2 PMAT Integration Commands

```bash
# Check Popper Falsifiability Score
pmat popper-score

# Run all quality gates (Jidoka)
pmat quality-gate --fail-on-violation

# Check PMAT Standard Compliance
pmat comply check

# Generate Rigor Report (formerly Perfection)
pmat rigor-score --format markdown > RIGOR.md
```

### 6.3 CI/CD Integration

```yaml
# .github/workflows/quality.yml
quality:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - uses: taiki-e/install-action@cargo-llvm-cov
    
    - name: Run coverage (All)
      run: make coverage-all
      timeout-minutes: 15

    - name: PMAT Quality Gate (Jidoka)
      run: pmat quality-gate --fail-on-violation

    - name: Popper Score Enforcement
      run: |
        SCORE=$(pmat popper-score --format json | jq .total_score)
        if (( $(echo "$SCORE < 80.0" | bc -l) )); then exit 1; fi
```

---

## 7. Hansei (Reflection) & Kaizen (Continuous Improvement)

### 7.1 The 5 Whys Analysis

When coverage drops or a bug slips through, we do not just "fix" it. We apply the 5 Whys:

1.  **Why did the coverage drop?** (e.g., "I added a new error handler.")
2.  **Why was it not covered?** (e.g., "I couldn't trigger the error in a unit test.")
3.  **Why couldn't you trigger it?** (e.g., "The dependency is hard-coded.")
4.  **Why is it hard-coded?** (e.g., "We don't use dependency injection for the logger.")
5.  **Root Cause:** "The architecture lacks testability for infrastructure components." -> **Action:** Refactor for DI, not just write one test.

### 7.2 Improvement Triggers

| Condition | Action |
|-----------|--------|
| Coverage drops >1% | **Stop the Line.** Immediate Hansei meeting. |
| Popper score drops < 80% | Review reproducibility infrastructure. |
| Mutation score < 60% | The tests are weak. Add severe testing. |
| New ignore added | Require peer review (Consensus). |

---

## 8. References

1.  Liker, J.K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill.
2.  Ammann, P., & Offutt, J. (2016). *Introduction to Software Testing*.
3.  Popper, K.R. (1959). *The Logic of Scientific Discovery*.
4.  Fowler, M. (2012). TestPyramid.
5.  Official `cargo-llvm-cov` Documentation.
6.  PMAT (Professional Multi-language Analysis Toolkit) Documentation.

---

## 9. Appendix A: Checklist Scoring

- **Pass:** 85/100 (Compliant)
- **Target:** 95/100 (Exemplary)
- **Fail:** <85/100 (Requires Immediate Action)

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.10.0 | 2026-01-28 | Claude | Added 20 more tests: gguf/config.rs (10), gguf/test_helpers.rs (10). Total tests: 11,499. |
| 1.9.0 | 2026-01-28 | Claude | Added 92 tests: paged_kv/mod.rs (27), bench/statistics.rs (19), tensor.rs (16), gguf/model.rs (9), gpu/backend.rs (10), gpu/adapters/apr.rs (11). Total tests: 11,463. |
| 1.8.0 | 2026-01-28 | Claude | Added 25 tests: layers/attention.rs (Attention, SlidingWindowAttention, FusedQKVAttention, MultiHeadAttention). Total tests: 11,371. |
| 1.7.0 | 2026-01-28 | Claude | Added 29 tests: gpu/metrics.rs (InferenceMetrics, HealthChecker, ShutdownCoordinator, ComputeBackend). Total tests: 11,346. |
| 1.6.0 | 2026-01-28 | Claude | Added 126 tests: ops.rs (32), fused.rs (31), types.rs (21), chunked_prefill.rs (42). Total tests: 11,317. |
| 1.5.0 | 2026-01-28 | Claude | Added 75 tests: fused_q5k_q6k (23), parallel_k (30), fused_k (22). Quantize module now has 1,819 tests. |
| 1.4.0 | 2026-01-28 | Claude | Added 115 tests: activation (32), parallel_dequant (23), types (27), simd (33). Quantize module now has 158 tests. |
| 1.3.0 | 2026-01-27 | Gemini | Integrated Toyota Way (Jidoka, Genchi Genbutsu, Hansei), renamed "Rigor" to "Rigor", clarified "Corroboration". |
| 1.2.0 | 2026-01-27 | Gemini | Full PMAT Tooling integration (Popper Score, Quality Gates) |
| 1.1.0 | 2026-01-27 | Gemini | Enhanced Popperian Falsification, Mutation Testing added, O(1) definition refined |
| 1.0.0 | 2026-01-27 | Claude | Initial specification |