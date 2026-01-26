//! Phase 43 - GpuExecutorTrait for Dependency Injection
//!
//! Abstracts GPU execution to enable testing without actual CUDA hardware.
//!
//! # Architecture
//!
//! ```text
//! GpuModel
//!    │
//!    └─► Box<dyn GpuExecutorTrait>
//!              │
//!              ├─► CudaExecutorAdapter (production)
//!              └─► MockExecutor (testing)
//! ```
//!
//! # Coverage Impact
//!
//! This trait enables testing of:
//! - `GpuModel::forward()` - Full forward pass logic
//! - `GpuModel::generate()` - Token generation flow
//! - Layer-by-layer computation verification

use crate::error::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Trait for GPU execution backends
///
/// Implementations must be Send + Sync to allow model transfer between threads
/// and safe access in `Arc<RwLock<GpuModel>>` contexts.
pub trait GpuExecutorTrait: Send + Sync {
    /// Perform matrix multiplication: C = A @ B
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix [m, k]
    /// * `b` - Right matrix [k, n]
    /// * `m` - Rows in A
    /// * `k` - Columns in A / Rows in B
    /// * `n` - Columns in B
    ///
    /// # Returns
    ///
    /// Result matrix [m, n]
    fn matmul(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>>;

    /// Check if GPU backend is available
    fn is_available(&self) -> bool;

    /// Get backend name for debugging
    fn name(&self) -> &str;

    /// Synchronize execution (wait for pending operations)
    fn synchronize(&self) -> Result<()>;

    /// Perform matrix multiplication with transposed B: C = A @ B^T
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix [m, k]
    /// * `b` - Right matrix [n, k] (will be transposed to [k, n])
    /// * `m` - Rows in A
    /// * `k` - Columns in A / Columns in B (before transpose)
    /// * `n` - Rows in B (becomes columns after transpose)
    ///
    /// # Returns
    ///
    /// Result matrix [m, n]
    fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>>;
}

/// Call record for MockExecutor
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorCall {
    /// Matrix multiplication call
    Matmul {
        /// Input A dimensions
        a_len: usize,
        /// Input B dimensions
        b_len: usize,
        /// M dimension
        m: usize,
        /// K dimension
        k: usize,
        /// N dimension
        n: usize,
    },
    /// Matrix multiplication with transposed B call
    MatmulTransposeB {
        /// Input A dimensions
        a_len: usize,
        /// Input B dimensions (before transpose)
        b_len: usize,
        /// M dimension
        m: usize,
        /// K dimension
        k: usize,
        /// N dimension
        n: usize,
    },
    /// Synchronize call
    Synchronize,
}

/// Mock executor for testing GpuModel without CUDA
///
/// Records all calls for verification and returns configurable results.
/// Uses interior mutability (Mutex) to allow Sync trait implementation.
pub struct MockExecutor {
    /// Name of this mock
    name: String,
    /// Recorded calls (protected by mutex for thread-safety)
    calls: Mutex<Vec<ExecutorCall>>,
    /// Counter for unique call IDs
    call_counter: AtomicUsize,
    /// Whether to simulate availability
    available: bool,
    /// Custom matmul result (if None, returns zeros)
    matmul_result: Option<Vec<f32>>,
    /// Whether matmul should fail
    matmul_should_fail: bool,
}

impl MockExecutor {
    /// Create a new mock executor
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            calls: Mutex::new(Vec::new()),
            call_counter: AtomicUsize::new(0),
            available: true,
            matmul_result: None,
            matmul_should_fail: false,
        }
    }

    /// Create mock that simulates unavailability
    #[must_use]
    pub fn unavailable(name: &str) -> Self {
        Self {
            name: name.to_string(),
            calls: Mutex::new(Vec::new()),
            call_counter: AtomicUsize::new(0),
            available: false,
            matmul_result: None,
            matmul_should_fail: false,
        }
    }

    /// Set custom matmul result
    #[must_use]
    pub fn with_matmul_result(mut self, result: Vec<f32>) -> Self {
        self.matmul_result = Some(result);
        self
    }

    /// Configure matmul to fail
    #[must_use]
    pub fn with_matmul_failure(mut self) -> Self {
        self.matmul_should_fail = true;
        self
    }

    /// Get all recorded calls (cloned for thread-safety)
    #[must_use]
    pub fn calls(&self) -> Vec<ExecutorCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Get number of calls
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    /// Get number of matmul calls
    #[must_use]
    pub fn matmul_count(&self) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| matches!(c, ExecutorCall::Matmul { .. }))
            .count()
    }

    /// Clear recorded calls
    pub fn clear_calls(&self) {
        self.calls.lock().unwrap().clear();
        self.call_counter.store(0, Ordering::SeqCst);
    }

    /// Check if specific call was made
    #[must_use]
    pub fn has_call(&self, call: &ExecutorCall) -> bool {
        self.calls.lock().unwrap().contains(call)
    }

    /// Get last call (cloned for thread-safety)
    #[must_use]
    pub fn last_call(&self) -> Option<ExecutorCall> {
        self.calls.lock().unwrap().last().cloned()
    }
}

impl GpuExecutorTrait for MockExecutor {
    #[allow(clippy::many_single_char_names)]
    fn matmul(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Record the call (thread-safe via Mutex)
        let call = ExecutorCall::Matmul {
            a_len: a.len(),
            b_len: b.len(),
            m,
            k,
            n,
        };
        self.calls.lock().unwrap().push(call);
        self.call_counter.fetch_add(1, Ordering::SeqCst);

        // Check for configured failure
        if self.matmul_should_fail {
            return Err(crate::error::RealizarError::GpuError {
                reason: "MockExecutor configured to fail".to_string(),
            });
        }

        // Return custom result or zeros
        if let Some(ref result) = self.matmul_result {
            Ok(result.clone())
        } else {
            // Default: return zeros of correct size
            Ok(vec![0.0f32; m * n])
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn synchronize(&self) -> Result<()> {
        // Record sync call (need interior mutability for &self)
        // For testing, we skip recording in synchronize since it takes &self
        Ok(())
    }

    #[allow(clippy::many_single_char_names)]
    fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Record the call (thread-safe via Mutex)
        let call = ExecutorCall::MatmulTransposeB {
            a_len: a.len(),
            b_len: b.len(),
            m,
            k,
            n,
        };
        self.calls.lock().unwrap().push(call);
        self.call_counter.fetch_add(1, Ordering::SeqCst);

        // Check for configured failure
        if self.matmul_should_fail {
            return Err(crate::error::RealizarError::GpuError {
                reason: "MockExecutor configured to fail".to_string(),
            });
        }

        // Return custom result or zeros
        if let Some(ref result) = self.matmul_result {
            Ok(result.clone())
        } else {
            // Default: return zeros of correct size
            Ok(vec![0.0f32; m * n])
        }
    }
}

impl std::fmt::Debug for MockExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockExecutor")
            .field("name", &self.name)
            .field("calls", &self.calls.lock().unwrap().len())
            .field("call_counter", &self.call_counter.load(Ordering::SeqCst))
            .field("available", &self.available)
            .field("matmul_result", &self.matmul_result.is_some())
            .field("matmul_should_fail", &self.matmul_should_fail)
            .finish()
    }
}

/// CPU-based executor for testing and fallback
///
/// Implements actual matrix multiplication on CPU.
pub struct CpuExecutor {
    name: String,
}

impl CpuExecutor {
    /// Create a new CPU executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "CpuExecutor".to_string(),
        }
    }
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuExecutorTrait for CpuExecutor {
    #[allow(clippy::many_single_char_names)]
    fn matmul(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Validate dimensions
        if a.len() != m * k {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!("A size {} != m*k {}", a.len(), m * k),
            });
        }
        if b.len() != k * n {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!("B size {} != k*n {}", b.len(), k * n),
            });
        }

        // Naive matmul (for correctness, not performance)
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn synchronize(&self) -> Result<()> {
        Ok(()) // No-op for CPU
    }

    #[allow(clippy::many_single_char_names)]
    fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Validate dimensions: A is [m, k], B is [n, k] (to be transposed to [k, n])
        if a.len() != m * k {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!("A size {} != m*k {}", a.len(), m * k),
            });
        }
        if b.len() != n * k {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!("B size {} != n*k {}", b.len(), n * k),
            });
        }

        // Naive matmul with B transposed: C = A @ B^T
        // A is [m, k], B is [n, k], C is [m, n]
        // B^T is [k, n], so C[i,j] = sum over p of A[i,p] * B[j,p]
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[j * k + p]; // Note: b[j, p] not b[p, j]
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }
}

impl std::fmt::Debug for CpuExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutor")
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MockExecutor Tests
    // =========================================================================

    #[test]
    fn test_mock_executor_creation() {
        let mock = MockExecutor::new("test");
        assert_eq!(mock.name(), "test");
        assert!(mock.is_available());
        assert_eq!(mock.call_count(), 0);
    }

    #[test]
    fn test_mock_executor_unavailable() {
        let mock = MockExecutor::unavailable("disabled");
        assert!(!mock.is_available());
    }

    #[test]
    fn test_mock_executor_records_matmul() {
        let mut mock = MockExecutor::new("test");
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

        let result = mock.matmul(&a, &b, 2, 2, 2).unwrap();

        assert_eq!(result.len(), 4); // 2x2 output
        assert_eq!(mock.matmul_count(), 1);

        let call = mock.last_call().unwrap();
        assert!(matches!(
            call,
            ExecutorCall::Matmul {
                a_len: 4,
                b_len: 4,
                m: 2,
                k: 2,
                n: 2
            }
        ));
    }

    #[test]
    fn test_mock_executor_custom_result() {
        let mut mock = MockExecutor::new("test").with_matmul_result(vec![1.0, 2.0, 3.0, 4.0]);

        let a = vec![0.0; 4];
        let b = vec![0.0; 4];
        let result = mock.matmul(&a, &b, 2, 2, 2).unwrap();

        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mock_executor_failure() {
        let mut mock = MockExecutor::new("test").with_matmul_failure();

        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let result = mock.matmul(&a, &b, 2, 2, 2);

        assert!(result.is_err());
    }

    #[test]
    fn test_mock_executor_clear_calls() {
        let mut mock = MockExecutor::new("test");
        let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);
        let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);

        assert_eq!(mock.call_count(), 2);

        mock.clear_calls();
        assert_eq!(mock.call_count(), 0);
    }

    #[test]
    fn test_mock_executor_debug() {
        let mock = MockExecutor::new("debug_test");
        let debug_str = format!("{:?}", mock);
        assert!(debug_str.contains("MockExecutor"));
        assert!(debug_str.contains("debug_test"));
    }

    #[test]
    fn test_mock_executor_synchronize() {
        let mock = MockExecutor::new("test");
        assert!(mock.synchronize().is_ok());
    }

    // =========================================================================
    // CpuExecutor Tests
    // =========================================================================

    #[test]
    fn test_cpu_executor_creation() {
        let cpu = CpuExecutor::new();
        assert_eq!(cpu.name(), "CpuExecutor");
        assert!(cpu.is_available());
    }

    #[test]
    fn test_cpu_executor_default() {
        let cpu = CpuExecutor::default();
        assert_eq!(cpu.name(), "CpuExecutor");
    }

    #[test]
    fn test_cpu_executor_matmul_2x2() {
        let mut cpu = CpuExecutor::new();

        // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
        // [3, 4] @ [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let c = cpu.matmul(&a, &b, 2, 2, 2).unwrap();

        assert_eq!(c.len(), 4);
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_executor_matmul_vector() {
        let mut cpu = CpuExecutor::new();

        // [1, 2, 3] @ [[1], [2], [3]] = [1*1 + 2*2 + 3*3] = [14]
        let a = vec![1.0, 2.0, 3.0]; // 1x3
        let b = vec![1.0, 2.0, 3.0]; // 3x1

        let c = cpu.matmul(&a, &b, 1, 3, 1).unwrap();

        assert_eq!(c.len(), 1);
        assert!((c[0] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_executor_matmul_rectangular() {
        let mut cpu = CpuExecutor::new();

        // 2x3 @ 3x4 = 2x4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0; 12]; // 3x4

        let c = cpu.matmul(&a, &b, 2, 3, 4).unwrap();

        assert_eq!(c.len(), 8);
        // First row: [1+2+3, 1+2+3, 1+2+3, 1+2+3] = [6, 6, 6, 6]
        assert!((c[0] - 6.0).abs() < 1e-5);
        // Second row: [4+5+6, ...] = [15, 15, 15, 15]
        assert!((c[4] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_executor_matmul_dimension_error_a() {
        let mut cpu = CpuExecutor::new();
        let a = vec![1.0; 5]; // Wrong size
        let b = vec![1.0; 4];

        let result = cpu.matmul(&a, &b, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_executor_matmul_dimension_error_b() {
        let mut cpu = CpuExecutor::new();
        let a = vec![1.0; 4];
        let b = vec![1.0; 5]; // Wrong size

        let result = cpu.matmul(&a, &b, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_executor_synchronize() {
        let cpu = CpuExecutor::new();
        assert!(cpu.synchronize().is_ok());
    }

    #[test]
    fn test_cpu_executor_debug() {
        let cpu = CpuExecutor::new();
        let debug_str = format!("{:?}", cpu);
        assert!(debug_str.contains("CpuExecutor"));
    }

    // =========================================================================
    // Trait Object Tests
    // =========================================================================

    #[test]
    fn test_executor_trait_object() {
        let mock: Box<dyn GpuExecutorTrait> = Box::new(MockExecutor::new("boxed"));
        assert_eq!(mock.name(), "boxed");
        assert!(mock.is_available());
    }

    #[test]
    fn test_executor_trait_polymorphism() {
        fn run_matmul(executor: &mut dyn GpuExecutorTrait) -> Result<Vec<f32>> {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![1.0; 4];
            executor.matmul(&a, &b, 2, 2, 2)
        }

        let mut mock = MockExecutor::new("mock");
        let mut cpu = CpuExecutor::new();

        let mock_result = run_matmul(&mut mock).unwrap();
        let cpu_result = run_matmul(&mut cpu).unwrap();

        // Mock returns zeros, CPU returns actual computation
        assert_eq!(mock_result, vec![0.0; 4]);
        assert!((cpu_result[0] - 3.0).abs() < 1e-5); // 1*1+2*1 = 3
    }

    // =========================================================================
    // ExecutorCall Tests
    // =========================================================================

    #[test]
    fn test_executor_call_equality() {
        let call1 = ExecutorCall::Matmul {
            a_len: 4,
            b_len: 4,
            m: 2,
            k: 2,
            n: 2,
        };
        let call2 = ExecutorCall::Matmul {
            a_len: 4,
            b_len: 4,
            m: 2,
            k: 2,
            n: 2,
        };
        let call3 = ExecutorCall::Synchronize;

        assert_eq!(call1, call2);
        assert_ne!(call1, call3);
    }

    #[test]
    fn test_executor_call_clone() {
        let call = ExecutorCall::Matmul {
            a_len: 8,
            b_len: 8,
            m: 4,
            k: 2,
            n: 4,
        };
        let cloned = call.clone();
        assert_eq!(call, cloned);
    }

    #[test]
    fn test_executor_call_debug() {
        let call = ExecutorCall::Matmul {
            a_len: 4,
            b_len: 4,
            m: 2,
            k: 2,
            n: 2,
        };
        let debug_str = format!("{:?}", call);
        assert!(debug_str.contains("Matmul"));
    }
}
