//! CUDA-accelerated quantized model
//!
//! This module provides GPU-accelerated inference for quantized models
//! using NVIDIA CUDA.
//!
//! # Architecture
//!
//! `OwnedQuantizedModelCuda` wraps an `OwnedQuantizedModel` with a CUDA executor
//! for GPU-accelerated matrix operations. Key features:
//!
//! - GPU-resident KV cache (avoids CPU→GPU transfer per token)
//! - Fused attention kernels
//! - Pre-cached quantized weights
//! - Batch generation support
//!
//! # Module Structure
//!
//! - `backend.rs`: CUDA kernel configuration and PTX generation (CudaBackend)
//! - `forward.rs`: Forward pass methods (single token, cached, GPU-resident)
//! - `generation.rs`: Token generation loops (basic, cached, streaming, batch)
//! - `speculative.rs`: Speculative decoding (self-speculative, draft model)
//! - `weights.rs`: Weight management (pre-caching, GPU upload)
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
//!
//! let model = OwnedQuantizedModel::from_mapped(&mapped)?;
//! let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)?; // GPU 0
//!
//! // GPU-accelerated forward pass
//! let logits = cuda_model.forward_cuda(&tokens)?;
//! ```

mod backend;
mod forward;
mod generation;
mod speculative;
mod weights;

// Re-export types for public API
pub use backend::CudaBackend;

use crate::error::{RealizarError, Result};

// Import types from peer modules (parent of cuda/)
use super::model::OwnedQuantizedModel;
use super::quantized::{OwnedQKVWeights, OwnedQuantizedTensor};
use super::runtime::{OwnedQuantizedKVCache, QuantizedGenerateConfig};
use super::utils::verbose;

// =============================================================================
// IMP-800: CUDA-Accelerated Model Wrapper
// =============================================================================

/// CUDA-accelerated wrapper for `OwnedQuantizedModel` (IMP-800a)
///
/// Provides GPU-accelerated forward pass using NVIDIA CUDA via trueno-gpu.
/// Caches the CudaExecutor to avoid initialization overhead (~50ms) per call.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
///
/// let model = OwnedQuantizedModel::from_mapped(&mapped)?;
/// let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)?; // GPU 0
///
/// // GPU-accelerated forward pass
/// let logits = cuda_model.forward_cuda(&tokens)?;
/// ```
pub struct OwnedQuantizedModelCuda {
    /// Inner model
    pub(crate) model: OwnedQuantizedModel,
    /// Cached CUDA executor
    pub(crate) executor: crate::cuda::CudaExecutor,
    /// GPU device name
    device_name: String,
    /// GPU memory (free, total) in bytes
    memory_info: (usize, usize),
}

impl OwnedQuantizedModelCuda {
    /// Create a new CUDA-accelerated model wrapper
    ///
    /// # Arguments
    ///
    /// * `model` - The quantized model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new(model: OwnedQuantizedModel, device_ordinal: i32) -> Result<Self> {
        Self::with_max_seq_len(model, device_ordinal, 2048)
    }

    /// Create a new CUDA-accelerated model wrapper with custom max sequence length
    ///
    /// # Arguments
    ///
    /// * `model` - The quantized model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    /// * `max_seq_len` - Maximum sequence length for GPU KV cache (PAR-018)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn with_max_seq_len(
        model: OwnedQuantizedModel,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> Result<Self> {
        use crate::cuda::CudaExecutor;

        let mut executor =
            CudaExecutor::new(device_ordinal).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::new".to_string(),
                reason: format!("CUDA initialization failed: {e}"),
            })?;

        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let memory_info = executor.memory_info().unwrap_or((0, 0));

        // PAR-018: Initialize GPU-resident KV cache for attention acceleration
        // This avoids ~66 MB CPU→GPU transfer per token for TinyLlama
        let num_layers = model.layers.len();
        let num_heads = model.config.num_heads;
        let num_kv_heads = model.config.num_kv_heads; // PAR-021 GQA support
        let head_dim = model.config.hidden_dim / num_heads;

        executor
            .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_kv_cache_gpu".to_string(),
                reason: format!("GPU KV cache initialization failed: {e}"),
            })?;

        // PAR-060: Set RoPE theta for position embeddings
        if verbose() {
            eprintln!(
                "[PAR-060] Setting rope_theta = {} for GPU path",
                model.config.rope_theta
            );
        }
        executor.set_rope_theta(model.config.rope_theta);

        // CORRECTNESS-011: Set rope_type for correct RoPE style (NORM vs NEOX)
        if verbose() {
            eprintln!(
                "[CORRECTNESS-011] Setting rope_type = {} for GPU path (0=NORM, 2=NEOX)",
                model.config.rope_type
            );
        }
        executor.set_rope_type(model.config.rope_type);

        Ok(Self {
            model,
            executor,
            device_name,
            memory_info,
        })
    }

    /// Check if CUDA is available
    #[must_use]
    pub fn is_available() -> bool {
        crate::cuda::CudaExecutor::is_available()
    }

    /// Get number of CUDA devices
    #[must_use]
    pub fn num_devices() -> usize {
        crate::cuda::CudaExecutor::num_devices()
    }

    /// Get GPU device name
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get GPU memory info (free, total) in bytes
    #[must_use]
    pub fn memory_info(&self) -> (usize, usize) {
        self.memory_info
    }

    /// Get VRAM usage in MB
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        (self.memory_info.1 / (1024 * 1024)) as u64
    }

    // ========================================================================
    // PAR-073: BrickProfiler API for per-brick timing
    // ========================================================================

    /// Enable per-brick profiling for real timing measurements.
    ///
    /// When enabled, each brick operation is timed individually using
    /// `std::time::Instant` with CUDA sync for accurate GPU timing.
    pub fn enable_profiling(&mut self) {
        self.executor.enable_profiling();
    }

    /// Disable per-brick profiling (default state).
    pub fn disable_profiling(&mut self) {
        self.executor.disable_profiling();
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_profiling_enabled(&self) -> bool {
        self.executor.is_profiling_enabled()
    }

    /// Get the brick profiler for reading statistics.
    #[must_use]
    pub fn profiler(&self) -> &trueno::BrickProfiler {
        self.executor.profiler()
    }

    /// Reset profiler statistics.
    pub fn reset_profiler(&mut self) {
        self.executor.reset_profiler();
    }

    /// Get profiler summary report.
    #[must_use]
    pub fn profiler_summary(&self) -> String {
        self.executor.profiler_summary()
    }

    /// Get reference to inner model
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModel {
        &self.model
    }

    /// PAR-111: Get mutable reference to CUDA executor
    ///
    /// Allows direct access for batched forward path and workspace initialization.
    #[must_use]
    pub fn executor_mut(&mut self) -> &mut crate::cuda::CudaExecutor {
        &mut self.executor
    }

    /// Synchronize CUDA stream (wait for all GPU operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        self.executor
            .synchronize()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::synchronize".to_string(),
                reason: format!("CUDA sync failed: {e}"),
            })
    }
}
