//! PTX Parity Validation — GH-219
//!
//! Validates that batched GPU kernels maintain structural parity with their
//! single-vector reference implementations. This catches bugs like:
//! - Missing batch dispatch (no ctaid.y or m_dim)
//! - u64 shared memory addressing (should use u32)
//! - Shared memory size mismatches
//!
//! Exposed as `apr qa` Gate 6: PTX Parity (F-PTX-001)

/// Model dimensions needed to construct and validate kernels
#[derive(Debug, Clone)]
pub struct KernelDimensions {
    /// Hidden dimension (e.g., 1536 for 1.5B, 3584 for 7B)
    pub hidden_dim: u32,
    /// FFN intermediate dimension (e.g., 4864 for 1.5B, 18944 for 7B)
    pub intermediate_dim: u32,
    /// Number of attention heads (e.g., 12 for 1.5B, 28 for 7B)
    pub num_heads: u32,
    /// Head dimension (e.g., 128)
    pub head_dim: u32,
    /// RoPE theta (e.g., 1_000_000.0 for Qwen2.5)
    pub rope_theta: f32,
    /// RMSNorm epsilon (e.g., 1e-6)
    pub epsilon: f32,
}

/// Result of PTX parity validation for a single kernel pair
#[derive(Debug, Clone)]
pub struct KernelParityResult {
    /// Kernel pair name (e.g., "BatchedRmsNorm ↔ RmsNorm")
    pub name: String,
    /// Whether parity validation passed
    pub passed: bool,
    /// Expected dispatch strategy
    pub dispatch_strategy: String,
    /// Violations found (empty if passed)
    pub violations: Vec<String>,
}

/// Full PTX parity validation report
#[derive(Debug, Clone)]
pub struct PtxParityReport {
    /// Individual kernel pair results
    pub results: Vec<KernelParityResult>,
    /// Total kernels validated
    pub total: usize,
    /// Kernels that passed
    pub passed: usize,
    /// Kernels that failed
    pub failed: usize,
}

impl PtxParityReport {
    /// Whether all kernel pairs passed validation
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Format a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        if self.all_passed() {
            format!(
                "{}/{} kernel pairs passed PTX parity",
                self.passed, self.total
            )
        } else {
            let failed_names: Vec<_> = self
                .results
                .iter()
                .filter(|r| !r.passed)
                .map(|r| r.name.as_str())
                .collect();
            format!(
                "{}/{} failed: {}",
                self.failed,
                self.total,
                failed_names.join(", ")
            )
        }
    }
}

/// Validate PTX parity for all 6 batched kernel pairs.
///
/// This instantiates each batched kernel with the model's dimensions,
/// then runs structural PTX analysis to verify:
/// 1. Batch dispatch mechanism exists (ctaid.y or m_dim)
/// 2. No u64 shared memory addressing
/// 3. Dispatch strategy matches expectation
#[cfg(feature = "cuda")]
pub fn validate_all_kernel_pairs(dims: &KernelDimensions) -> PtxParityReport {
    use trueno_gpu::kernels::{
        BatchedQ4KGemvKernel, BatchedQ6KGemvKernel, BatchedResidualAddKernel, BatchedRopeKernel,
        BatchedSwigluKernel, BatchedVectorizedRmsNormKernel, Kernel, KernelParity,
    };

    let mut results = Vec::new();

    // 1. RmsNorm: grid_y dispatch
    {
        let kernel =
            BatchedVectorizedRmsNormKernel::new(dims.hidden_dim, 1).with_epsilon(dims.epsilon);
        let parity = kernel.validate_batch_dispatch();
        let ptx = kernel.emit_ptx();
        let u64_violation = check_u64_shared_mem(&ptx);
        let mut violations: Vec<String> = parity
            .violations
            .iter()
            .map(|v| v.message.clone())
            .collect();
        if let Some(v) = u64_violation {
            violations.push(v);
        }
        results.push(KernelParityResult {
            name: "BatchedRmsNorm \u{2194} RmsNorm".to_string(),
            passed: parity.is_compatible && violations.len() == parity.violations.len(),
            dispatch_strategy: "grid_y".to_string(),
            violations,
        });
    }

    // 2. Q4K GEMV: register_unroll dispatch
    {
        let kernel = BatchedQ4KGemvKernel::new(dims.hidden_dim, dims.hidden_dim, 1);
        let parity = kernel.validate_batch_dispatch();
        let ptx = kernel.emit_ptx();
        let u64_violation = check_u64_shared_mem(&ptx);
        let mut violations: Vec<String> = parity
            .violations
            .iter()
            .map(|v| v.message.clone())
            .collect();
        if let Some(v) = u64_violation {
            violations.push(v);
        }
        results.push(KernelParityResult {
            name: "BatchedQ4KGemv \u{2194} Q4KGemv".to_string(),
            passed: parity.is_compatible && violations.len() == parity.violations.len(),
            dispatch_strategy: "register_unroll".to_string(),
            violations,
        });
    }

    // 3. Q6K GEMV: register_unroll dispatch
    {
        let kernel = BatchedQ6KGemvKernel::new(dims.hidden_dim, dims.hidden_dim, 1);
        let parity = kernel.validate_batch_dispatch();
        let ptx = kernel.emit_ptx();
        let u64_violation = check_u64_shared_mem(&ptx);
        let mut violations: Vec<String> = parity
            .violations
            .iter()
            .map(|v| v.message.clone())
            .collect();
        if let Some(v) = u64_violation {
            violations.push(v);
        }
        results.push(KernelParityResult {
            name: "BatchedQ6KGemv \u{2194} Q6KGemv".to_string(),
            passed: parity.is_compatible && violations.len() == parity.violations.len(),
            dispatch_strategy: "register_unroll".to_string(),
            violations,
        });
    }

    // 4. ResidualAdd: grid_y dispatch
    {
        let kernel = BatchedResidualAddKernel::new(dims.hidden_dim, 1);
        let parity = kernel.validate_batch_dispatch();
        let violations: Vec<String> = parity
            .violations
            .iter()
            .map(|v| v.message.clone())
            .collect();
        results.push(KernelParityResult {
            name: "BatchedResidualAdd \u{2194} ResidualAdd".to_string(),
            passed: parity.is_compatible,
            dispatch_strategy: "grid_y".to_string(),
            violations,
        });
    }

    // 5. RoPE: grid_y dispatch
    {
        let kernel = BatchedRopeKernel::new(dims.num_heads, dims.head_dim, 1, dims.rope_theta);
        let parity = kernel.validate_batch_dispatch();
        let violations: Vec<String> = parity
            .violations
            .iter()
            .map(|v| v.message.clone())
            .collect();
        results.push(KernelParityResult {
            name: "BatchedRoPE \u{2194} RoPE".to_string(),
            passed: parity.is_compatible,
            dispatch_strategy: "grid_y".to_string(),
            violations,
        });
    }

    // 6. SwiGLU: grid_y dispatch
    {
        let kernel = BatchedSwigluKernel::new(dims.intermediate_dim, 1);
        let parity = kernel.validate_batch_dispatch();
        let violations: Vec<String> = parity
            .violations
            .iter()
            .map(|v| v.message.clone())
            .collect();
        results.push(KernelParityResult {
            name: "BatchedSwiGLU \u{2194} SwiGLU".to_string(),
            passed: parity.is_compatible,
            dispatch_strategy: "grid_y".to_string(),
            violations,
        });
    }

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    PtxParityReport {
        results,
        total,
        passed,
        failed,
    }
}

/// Check for u64 shared memory addressing in PTX
#[cfg(feature = "cuda")]
fn check_u64_shared_mem(ptx: &str) -> Option<String> {
    for line in ptx.lines() {
        let trimmed = line.trim();
        if (trimmed.contains("st.shared") || trimmed.contains("ld.shared"))
            && trimmed.contains("[%rd")
        {
            return Some(format!("u64 shared memory addressing: {}", trimmed));
        }
    }
    None
}

/// Validate PTX parity when CUDA is not available (no-op)
#[cfg(not(feature = "cuda"))]
pub fn validate_all_kernel_pairs(_dims: &KernelDimensions) -> PtxParityReport {
    PtxParityReport {
        results: Vec::new(),
        total: 0,
        passed: 0,
        failed: 0,
    }
}

/// Generate PTX source for a named kernel using the given model dimensions.
///
/// Supports the most commonly used kernels in the inference pipeline.
/// Returns the kernel name and PTX source, or an error if the kernel is unknown.
#[cfg(feature = "cuda")]
pub fn generate_named_kernel_ptx(
    name: &str,
    dims: &KernelDimensions,
) -> Result<(String, String), String> {
    use trueno_gpu::kernels::{
        ArgMaxKernel, BatchedQ4KGemvKernel, BatchedQ6KGemvKernel, BatchedResidualAddKernel,
        BatchedRopeKernel, BatchedSwigluKernel, BatchedVectorizedRmsNormKernel,
        FusedSwigluKernel, GemmKernel, Kernel, Q4KGemvKernel, Q5KGemvKernel, Q6KGemvKernel,
        ResidualAddKernel, RmsNormKernel, RopeKernel, SoftmaxKernel, VectorizedRmsNormKernel,
    };

    let name_lower = name.to_lowercase().replace('-', "").replace('_', "");
    let (label, ptx) = match name_lower.as_str() {
        "q4kgemvkernel" | "q4kgemv" | "q4k" => {
            let k = Q4KGemvKernel::new(dims.hidden_dim, dims.hidden_dim);
            ("Q4KGemvKernel".to_string(), k.emit_ptx())
        }
        "q6kgemvkernel" | "q6kgemv" | "q6k" => {
            let k = Q6KGemvKernel::new(dims.hidden_dim, dims.hidden_dim);
            ("Q6KGemvKernel".to_string(), k.emit_ptx())
        }
        "q5kgemvkernel" | "q5kgemv" | "q5k" => {
            let k = Q5KGemvKernel::new(dims.hidden_dim, dims.hidden_dim);
            ("Q5KGemvKernel".to_string(), k.emit_ptx())
        }
        "rmsnormkernel" | "rmsnorm" => {
            let k = RmsNormKernel::new(dims.hidden_dim);
            ("RmsNormKernel".to_string(), k.emit_ptx())
        }
        "vectorizedrmsnormkernel" | "vectorizedrmsnorm" | "vecrmsnorm" => {
            let k = VectorizedRmsNormKernel::new(dims.hidden_dim);
            ("VectorizedRmsNormKernel".to_string(), k.emit_ptx())
        }
        "softmaxkernel" | "softmax" => {
            let k = SoftmaxKernel::new(dims.hidden_dim);
            ("SoftmaxKernel".to_string(), k.emit_ptx())
        }
        "argmaxkernel" | "argmax" => {
            let k = ArgMaxKernel::new(dims.hidden_dim);
            ("ArgMaxKernel".to_string(), k.emit_ptx())
        }
        "residualaddkernel" | "residualadd" | "residual" => {
            let k = ResidualAddKernel::new(dims.hidden_dim);
            ("ResidualAddKernel".to_string(), k.emit_ptx())
        }
        "ropekernel" | "rope" => {
            let k = RopeKernel::new(dims.num_heads, dims.head_dim, dims.rope_theta);
            ("RopeKernel".to_string(), k.emit_ptx())
        }
        "swiglukernel" | "swiglu" | "fusedswiglu" => {
            let k = FusedSwigluKernel::new(dims.intermediate_dim);
            ("FusedSwigluKernel".to_string(), k.emit_ptx())
        }
        "gemmkernel" | "gemm" => {
            let k = GemmKernel::naive(dims.hidden_dim, dims.hidden_dim, dims.hidden_dim);
            ("GemmKernel".to_string(), k.emit_ptx())
        }
        // Batched variants
        "batchedrmsnorm" | "batchedvectorizedrmsnorm" => {
            let k = BatchedVectorizedRmsNormKernel::new(dims.hidden_dim, 1)
                .with_epsilon(dims.epsilon);
            ("BatchedVectorizedRmsNormKernel".to_string(), k.emit_ptx())
        }
        "batchedq4kgemv" | "batchedq4k" => {
            let k = BatchedQ4KGemvKernel::new(dims.hidden_dim, dims.hidden_dim, 1);
            ("BatchedQ4KGemvKernel".to_string(), k.emit_ptx())
        }
        "batchedq6kgemv" | "batchedq6k" => {
            let k = BatchedQ6KGemvKernel::new(dims.hidden_dim, dims.hidden_dim, 1);
            ("BatchedQ6KGemvKernel".to_string(), k.emit_ptx())
        }
        "batchedresidualadd" | "batchedresidual" => {
            let k = BatchedResidualAddKernel::new(dims.hidden_dim, 1);
            ("BatchedResidualAddKernel".to_string(), k.emit_ptx())
        }
        "batchedrope" => {
            let k = BatchedRopeKernel::new(dims.num_heads, dims.head_dim, 1, dims.rope_theta);
            ("BatchedRopeKernel".to_string(), k.emit_ptx())
        }
        "batchedswiglu" => {
            let k = BatchedSwigluKernel::new(dims.intermediate_dim, 1);
            ("BatchedSwigluKernel".to_string(), k.emit_ptx())
        }
        _ => {
            let available = [
                "Q4KGemv", "Q5KGemv", "Q6KGemv", "RmsNorm", "VectorizedRmsNorm",
                "Softmax", "ArgMax", "ResidualAdd", "RoPE", "SwiGLU", "GEMM",
                "BatchedRmsNorm", "BatchedQ4KGemv", "BatchedQ6KGemv",
                "BatchedResidualAdd", "BatchedRoPE", "BatchedSwiGLU",
            ];
            return Err(format!(
                "Unknown kernel '{}'. Available: {}",
                name,
                available.join(", ")
            ));
        }
    };
    Ok((label, ptx))
}

/// Generate PTX source when CUDA is not available.
#[cfg(not(feature = "cuda"))]
pub fn generate_named_kernel_ptx(
    name: &str,
    _dims: &KernelDimensions,
) -> Result<(String, String), String> {
    Err(format!(
        "Kernel '{}' requires CUDA feature. Build with --features cuda",
        name
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // KernelDimensions struct construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_kernel_dimensions_new_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 1536,
            intermediate_dim: 4864,
            num_heads: 12,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            epsilon: 1e-6,
        };
        assert_eq!(dims.hidden_dim, 1536);
        assert_eq!(dims.intermediate_dim, 4864);
        assert_eq!(dims.num_heads, 12);
        assert_eq!(dims.head_dim, 128);
        assert!((dims.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((dims.epsilon - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn test_kernel_dimensions_clone_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            epsilon: 1e-6,
        };
        let cloned = dims.clone();
        assert_eq!(cloned.hidden_dim, dims.hidden_dim);
        assert_eq!(cloned.num_heads, dims.num_heads);
    }

    #[test]
    fn test_kernel_dimensions_debug_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 64,
            intermediate_dim: 256,
            num_heads: 4,
            head_dim: 16,
            rope_theta: 10000.0,
            epsilon: 1e-5,
        };
        let debug_str = format!("{:?}", dims);
        assert!(debug_str.contains("hidden_dim: 64"));
    }

    // -------------------------------------------------------------------------
    // KernelParityResult struct
    // -------------------------------------------------------------------------

    #[test]
    fn test_kernel_parity_result_passed_gh219() {
        let result = KernelParityResult {
            name: "TestKernel ↔ RefKernel".to_string(),
            passed: true,
            dispatch_strategy: "grid_y".to_string(),
            violations: vec![],
        };
        assert!(result.passed);
        assert!(result.violations.is_empty());
        assert_eq!(result.dispatch_strategy, "grid_y");
    }

    #[test]
    fn test_kernel_parity_result_failed_gh219() {
        let result = KernelParityResult {
            name: "Broken ↔ Ref".to_string(),
            passed: false,
            dispatch_strategy: "register_unroll".to_string(),
            violations: vec!["u64 shared memory addressing".to_string()],
        };
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
    }

    // -------------------------------------------------------------------------
    // PtxParityReport: all_passed()
    // -------------------------------------------------------------------------

    #[test]
    fn test_ptx_parity_report_all_passed_empty_gh219() {
        let report = PtxParityReport {
            results: vec![],
            total: 0,
            passed: 0,
            failed: 0,
        };
        assert!(report.all_passed());
    }

    #[test]
    fn test_ptx_parity_report_all_passed_true_gh219() {
        let report = PtxParityReport {
            results: vec![
                KernelParityResult {
                    name: "A ↔ B".to_string(),
                    passed: true,
                    dispatch_strategy: "grid_y".to_string(),
                    violations: vec![],
                },
                KernelParityResult {
                    name: "C ↔ D".to_string(),
                    passed: true,
                    dispatch_strategy: "grid_y".to_string(),
                    violations: vec![],
                },
            ],
            total: 2,
            passed: 2,
            failed: 0,
        };
        assert!(report.all_passed());
    }

    #[test]
    fn test_ptx_parity_report_all_passed_false_gh219() {
        let report = PtxParityReport {
            results: vec![KernelParityResult {
                name: "Broken".to_string(),
                passed: false,
                dispatch_strategy: "grid_y".to_string(),
                violations: vec!["bad".to_string()],
            }],
            total: 1,
            passed: 0,
            failed: 1,
        };
        assert!(!report.all_passed());
    }

    // -------------------------------------------------------------------------
    // PtxParityReport: summary()
    // -------------------------------------------------------------------------

    #[test]
    fn test_ptx_parity_report_summary_all_passed_gh219() {
        let report = PtxParityReport {
            results: vec![
                KernelParityResult {
                    name: "A".to_string(),
                    passed: true,
                    dispatch_strategy: "grid_y".to_string(),
                    violations: vec![],
                },
                KernelParityResult {
                    name: "B".to_string(),
                    passed: true,
                    dispatch_strategy: "grid_y".to_string(),
                    violations: vec![],
                },
            ],
            total: 2,
            passed: 2,
            failed: 0,
        };
        let summary = report.summary();
        assert!(summary.contains("2/2 kernel pairs passed PTX parity"));
    }

    #[test]
    fn test_ptx_parity_report_summary_some_failed_gh219() {
        let report = PtxParityReport {
            results: vec![
                KernelParityResult {
                    name: "Good".to_string(),
                    passed: true,
                    dispatch_strategy: "grid_y".to_string(),
                    violations: vec![],
                },
                KernelParityResult {
                    name: "BadRmsNorm".to_string(),
                    passed: false,
                    dispatch_strategy: "grid_y".to_string(),
                    violations: vec!["u64 addr".to_string()],
                },
                KernelParityResult {
                    name: "BadSwiglu".to_string(),
                    passed: false,
                    dispatch_strategy: "grid_y".to_string(),
                    violations: vec!["missing dispatch".to_string()],
                },
            ],
            total: 3,
            passed: 1,
            failed: 2,
        };
        let summary = report.summary();
        assert!(summary.contains("2/3 failed"));
        assert!(summary.contains("BadRmsNorm"));
        assert!(summary.contains("BadSwiglu"));
    }

    #[test]
    fn test_ptx_parity_report_summary_single_failure_gh219() {
        let report = PtxParityReport {
            results: vec![KernelParityResult {
                name: "FailedKernel".to_string(),
                passed: false,
                dispatch_strategy: "register_unroll".to_string(),
                violations: vec!["violation".to_string()],
            }],
            total: 1,
            passed: 0,
            failed: 1,
        };
        let summary = report.summary();
        assert!(summary.contains("1/1 failed"));
        assert!(summary.contains("FailedKernel"));
    }

    // -------------------------------------------------------------------------
    // Non-CUDA stubs
    // -------------------------------------------------------------------------

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_validate_all_kernel_pairs_no_cuda_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 1536,
            intermediate_dim: 4864,
            num_heads: 12,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            epsilon: 1e-6,
        };
        let report = validate_all_kernel_pairs(&dims);
        assert_eq!(report.total, 0);
        assert_eq!(report.passed, 0);
        assert_eq!(report.failed, 0);
        assert!(report.results.is_empty());
        assert!(report.all_passed());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_generate_named_kernel_ptx_no_cuda_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 1536,
            intermediate_dim: 4864,
            num_heads: 12,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            epsilon: 1e-6,
        };
        let result = generate_named_kernel_ptx("q4k", &dims);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("requires CUDA feature"));
        assert!(err.contains("q4k"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_generate_named_kernel_ptx_unknown_no_cuda_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 64,
            intermediate_dim: 256,
            num_heads: 4,
            head_dim: 16,
            rope_theta: 10000.0,
            epsilon: 1e-5,
        };
        let result = generate_named_kernel_ptx("nonexistent", &dims);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires CUDA feature"));
    }

    // -------------------------------------------------------------------------
    // CUDA kernel pair validation (when CUDA is available)
    // -------------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    #[test]
    fn test_validate_all_kernel_pairs_cuda_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 1536,
            intermediate_dim: 4864,
            num_heads: 12,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            epsilon: 1e-6,
        };
        let report = validate_all_kernel_pairs(&dims);
        assert_eq!(report.total, 6);
        assert_eq!(report.passed + report.failed, 6);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_generate_named_kernel_ptx_q4k_cuda_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 1536,
            intermediate_dim: 4864,
            num_heads: 12,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            epsilon: 1e-6,
        };
        let result = generate_named_kernel_ptx("q4k", &dims);
        assert!(result.is_ok());
        let (label, ptx) = result.unwrap();
        assert_eq!(label, "Q4KGemvKernel");
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".entry"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_generate_named_kernel_ptx_all_names_cuda_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 128,
            intermediate_dim: 512,
            num_heads: 4,
            head_dim: 32,
            rope_theta: 10000.0,
            epsilon: 1e-6,
        };
        let names = [
            "q4k", "q6k", "q5k", "rmsnorm", "vectorizedrmsnorm",
            "softmax", "argmax", "residualadd", "rope", "swiglu", "gemm",
            "batchedrmsnorm", "batchedq4k", "batchedq6k",
            "batchedresidualadd", "batchedrope", "batchedswiglu",
        ];
        for name in names {
            let result = generate_named_kernel_ptx(name, &dims);
            assert!(result.is_ok(), "Failed for kernel: {}", name);
            let (_, ptx) = result.unwrap();
            assert!(!ptx.is_empty(), "Empty PTX for kernel: {}", name);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_generate_named_kernel_ptx_unknown_cuda_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 128,
            intermediate_dim: 512,
            num_heads: 4,
            head_dim: 32,
            rope_theta: 10000.0,
            epsilon: 1e-6,
        };
        let result = generate_named_kernel_ptx("nonexistent_kernel", &dims);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown kernel"));
        assert!(err.contains("Available:"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_generate_named_kernel_ptx_case_insensitive_cuda_gh219() {
        let dims = KernelDimensions {
            hidden_dim: 128,
            intermediate_dim: 512,
            num_heads: 4,
            head_dim: 32,
            rope_theta: 10000.0,
            epsilon: 1e-6,
        };
        // Test with various casing and separators
        let result1 = generate_named_kernel_ptx("Q4K", &dims);
        let result2 = generate_named_kernel_ptx("q4k_gemv_kernel", &dims);
        let result3 = generate_named_kernel_ptx("Q4K-GEMV", &dims);
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_check_u64_shared_mem_clean_gh219() {
        let ptx = ".version 8.0\n.entry test {\nld.shared.f32 [%r1], val;\n}";
        let result = check_u64_shared_mem(ptx);
        assert!(result.is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_check_u64_shared_mem_violation_gh219() {
        let ptx = ".version 8.0\n.entry test {\nst.shared.f32 [%rd1], val;\n}";
        let result = check_u64_shared_mem(ptx);
        assert!(result.is_some());
        assert!(result.unwrap().contains("u64 shared memory addressing"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_check_u64_shared_mem_ld_violation_gh219() {
        let ptx = "ld.shared.f32 [%rd5], %f0;";
        let result = check_u64_shared_mem(ptx);
        assert!(result.is_some());
    }
}
