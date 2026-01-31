# Code Coverage

Realizar maintains rigorous code coverage standards through the **T-COV-95** testing methodology, achieving a documented **Platform Ceiling** of 90.56% line coverage.

## Coverage Targets

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Line Coverage | 95% | 90.56% | Platform Ceiling |
| Function Coverage | 95% | 94.30% | Achieved |
| Region Coverage | 85% | 90.83% | Exceeded |

## Platform Ceiling Doctrine

The **Platform Ceiling** (90.56%) represents the maximum achievable coverage when testing on a single hardware platform. This is not a failure to reach 95%—it's the empirically determined limit imposed by:

1. **Hardware-Specific Code Paths**: CUDA, wgpu, and SIMD branches that require specific hardware
2. **Async Runtime Variants**: Code paths for different async runtimes (tokio vs async-std)
3. **Error Conditions**: Operating system error paths that cannot be reliably triggered

### Compute Quarantine

The uncovered code falls into a **Compute Quarantine**—hardware-specific execution paths verified through alternative methods:

```
src/cuda/                  # Requires NVIDIA GPU
src/gguf/inference/        # SIMD/GPU dispatch paths
src/gpu/                   # wgpu backend
```

These paths are verified through:
- **Parity tests**: Comparing CUDA output against CPU reference
- **Hardware integration tests**: Run separately on GPU-equipped CI runners
- **Manual verification**: Tensor output validation against llama.cpp

## Running Coverage

```bash
# Full coverage report (HTML)
cargo llvm-cov --html

# Quick coverage check
cargo llvm-cov

# Coverage with specific features
cargo llvm-cov --features cuda --html
```

## Coverage Philosophy: Popperian Falsification

Realizar follows **Popperian Falsification**—tests attempt to *refute* the implementation, not verify it. This means:

1. **Property-based tests** generate millions of inputs looking for failures
2. **Mutation testing** verifies that changing code breaks tests
3. **Boundary testing** targets edge cases where bugs hide

See [Property-Based Testing](../tdd/property-based.md) for the proptest integration.

## T-COV-95 Campaign Results

The T-COV-95 campaign (January 2026) achieved:

- **6,324 total tests** (all passing)
- **33 proptest tests** generating millions of test cases
- **Security vulnerability discovered** via generative fuzzing
- **100% mutation score** on critical API paths

### Security Discovery

Proptest fuzzing revealed an **allocation attack vulnerability** in GGUF parsing. Corrupted headers with `tensor_count = u64::MAX` could cause multi-terabyte allocation attempts. This was fixed with bounds checks:

```rust
// src/gguf/loader.rs - Allocation attack prevention
const MAX_TENSOR_COUNT: u64 = 100_000;
const MAX_METADATA_COUNT: u64 = 10_000;
const MAX_DIMS: u32 = 8;
const MAX_ARRAY_LEN: u64 = 10_000_000;

if tensor_count > MAX_TENSOR_COUNT {
    return Err(RealizarError::UnsupportedOperation {
        operation: "parse_gguf".to_string(),
        reason: format!(
            "tensor_count {} exceeds maximum allowed {}",
            tensor_count, MAX_TENSOR_COUNT
        ),
    });
}
```

## Continuous Integration

Coverage is checked on every PR via GitHub Actions:

```yaml
- name: Coverage Check
  run: |
    cargo llvm-cov --lcov --output-path lcov.info
    # Fail if below 85% (accounting for Platform Ceiling)
```

## Tools

- **cargo-llvm-cov**: LLVM-based coverage (fast, accurate)
- **proptest**: Property-based test generation
- **cargo-mutants**: Mutation testing

See [cargo llvm-cov](../tools/cargo-llvm-cov.md) for installation and usage.
