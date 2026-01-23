//! CUDA Scheduler Core (PMAT-802)
//!
//! Core CudaScheduler implementation for direct CUDA execution.

#[cfg(feature = "cuda")]
use crate::cuda::CudaExecutor;
use crate::error::{RealizarError, Result};
use super::super::{
    HybridScheduler, StreamingKVCache, exceeds_gpu_buffer_limit,
    cpu_matmul_transposed_simd, cpu_matmul,
};

// Unlike HybridScheduler, this ALWAYS uses CUDA (even for m=1)
// ============================================================================

/// CUDA-native scheduler for GpuModel inference
///
/// Key difference from HybridScheduler:
/// - ALWAYS uses CudaExecutor for matmul (no CPU fallback for m=1)
/// - Direct CUDA kernel execution instead of trueno wgpu
/// - Eliminates the m=1 CPU restriction that causes 1090x slowdown
///
/// ## Performance Impact
///
/// The HybridScheduler forces CPU for m=1 (single-token generation), which
/// causes token-by-token inference to run on CPU. CudaScheduler eliminates
/// this restriction, enabling full GPU acceleration for generation.
#[cfg(feature = "cuda")]
pub struct CudaScheduler {
    executor: crate::cuda::CudaExecutor,
}

#[cfg(feature = "cuda")]
impl CudaScheduler {
    /// Create a new CUDA scheduler
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or initialization fails.
    pub fn new() -> Result<Self> {
        let executor = crate::cuda::CudaExecutor::new(0).map_err(|e| RealizarError::GpuError {
            reason: format!("Failed to create CudaExecutor: {}", e),
        })?;

        Ok(Self { executor })
    }

    /// Check if CUDA is available
    #[must_use]
    pub fn has_cuda(&self) -> bool {
        true // If we have a CudaScheduler, CUDA is available
    }

    /// Check if this scheduler uses CUDA for the given matrix dimensions
    ///
    /// Unlike HybridScheduler, CudaScheduler ALWAYS returns true (uses CUDA).
    #[must_use]
    #[allow(clippy::unused_self)]
    pub fn uses_cuda_for(&self, _m: usize, _k: usize, _n: usize) -> bool {
        true // Always use CUDA - this is the key difference from HybridScheduler
    }

    /// Execute matmul using CUDA
    ///
    /// Same interface as HybridScheduler::matmul for drop-in replacement.
    ///
    /// # Errors
    ///
    /// Returns error if CUDA execution fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; m * n];

        self.executor
            .gemm(a, b, &mut output, m as u32, n as u32, k as u32)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("CUDA GEMM failed: {}", e),
            })?;

        Ok(output)
    }

    /// Get device name
    pub fn device_name(&self) -> Result<String> {
        self.executor
            .device_name()
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Failed to get device name: {}", e),
            })
    }

    /// Cache a weight matrix on GPU (PARITY-120: 10x speedup)
    ///
    /// Weights stay on GPU and are reused for all forward passes.
    ///
    /// # Errors
    ///
    /// Returns error if GPU memory allocation fails.
    pub fn cache_weight(&mut self, name: &str, weight: &[f32]) -> Result<()> {
        self.executor
            .load_weights(name, weight)
            .map(|_| ())
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Failed to cache weight '{}': {}", name, e),
            })
    }

    /// Check if a weight is cached
    #[must_use]
    pub fn has_cached_weight(&self, name: &str) -> bool {
        self.executor.has_weights(name)
    }

    /// Get number of cached weights
    #[must_use]
    pub fn cached_weight_count(&self) -> usize {
        self.executor.cached_weight_count()
    }

    /// Execute matmul using cached weight (PARITY-120: 10x speedup)
    ///
    /// Uses pre-loaded weight on GPU, only transfers input/output.
    /// This is the fast path for single-token generation.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight (from `cache_weight`)
    /// * `x` - Input vector (k elements)
    /// * `k` - Input dimension
    /// * `n` - Output dimension
    ///
    /// # Errors
    ///
    /// Returns error if weight not cached or CUDA fails.
    pub fn matmul_cached(
        &mut self,
        weight_name: &str,
        x: &[f32],
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; n];

        self.executor
            .gemv_cached(weight_name, x, &mut output, k as u32, n as u32)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("CUDA cached GEMV failed: {}", e),
            })?;

        Ok(output)
    }
}
