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

/// Error from CUDA model initialization that preserves the unconsumed model.
///
/// When `OwnedQuantizedModelCuda::new()` fails, the model is returned inside this error
/// so callers can fall back to CPU without an expensive 1GB clone.
pub struct CudaInitError {
    /// The initialization error
    pub error: RealizarError,
    /// The unconsumed model, returned for CPU fallback
    model: OwnedQuantizedModel,
}

impl CudaInitError {
    /// Extract the unconsumed model for CPU fallback
    #[must_use]
    pub fn into_model(self) -> OwnedQuantizedModel {
        self.model
    }
}

impl std::fmt::Display for CudaInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::fmt::Debug for CudaInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaInitError({:?})", self.error)
    }
}

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
    /// PAR-083: Pre-allocated embedding buffer to eliminate per-token heap allocation.
    /// Five-Whys root cause: embed() allocates Vec<f32> per token (~14KB for 7B).
    /// Fix: Reuse this buffer with embed_into().
    embed_buf: Vec<f32>,
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
    /// Create a CUDA model wrapper. Model is consumed on failure.
    /// Use `with_max_seq_len` for recoverable errors (returns model on failure).
    pub fn new(model: OwnedQuantizedModel, device_ordinal: i32) -> Result<Self> {
        Self::with_max_seq_len(model, device_ordinal, 2048).map_err(|e| e.error)
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
    /// Returns `CudaInitError` containing both the error and the unconsumed model,
    /// allowing callers to recover the model for CPU fallback without cloning.
    /// GH-280: Check GPU capability before allocation.
    /// Extracted to reduce cognitive complexity of `with_max_seq_len`.
    fn check_gpu_capability(
        model: OwnedQuantizedModel,
    ) -> std::result::Result<OwnedQuantizedModel, CudaInitError> {
        let required = crate::capability::required_ops(&model.config.constraints);
        let supported = crate::capability::gpu_supported_ops();
        if let Err(missing) = crate::capability::check_capability(&required, &supported) {
            let missing_names: Vec<String> = missing.iter().map(ToString::to_string).collect();
            return Err(CudaInitError {
                error: RealizarError::CapabilityMismatch {
                    architecture: model.config.architecture.clone(),
                    missing_ops: missing_names.join(", "),
                    suggestion: "Model will use CPU inference. To add GPU support, \
                                 implement the missing kernels in trueno."
                        .to_string(),
                },
                model,
            });
        }
        Ok(model)
    }

    /// GH-199/PARITY-GATE: Preload GPU weights and verify correctness.
    /// Extracted to reduce cognitive complexity of `with_max_seq_len`.
    fn preload_and_verify(mut self) -> std::result::Result<Self, CudaInitError> {
        if !self.supports_gpu_resident() {
            return Ok(self);
        }

        // GH-199 ROOT CAUSE B: Eagerly preload weights for GPU-resident path.
        if let Err(e) = self.preload_weights_gpu() {
            return Err(CudaInitError {
                error: e,
                model: self.into_model(),
            });
        }

        // PARITY-GATE: Jidoka — stop-the-line if GPU diverges from CPU.
        // Run ONE token through both backends and compare logits.
        // If cosine similarity < 0.99, refuse to construct.
        // Skip gate if SKIP_PARITY_GATE=1 (for debugging the gate itself)
        let skip_gate = std::env::var("SKIP_PARITY_GATE")
            .map(|v| v == "1")
            .unwrap_or(false);

        if !skip_gate {
            if let Err(e) = parity_gate(&mut self) {
                return Err(CudaInitError {
                    error: e,
                    model: self.into_model(),
                });
            }
        }

        Ok(self)
    }

    /// Create a GPU-accelerated inference engine with a custom maximum sequence length.
    pub fn with_max_seq_len(
        model: OwnedQuantizedModel,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> std::result::Result<Self, CudaInitError> {
        use crate::cuda::CudaExecutor;

        // GH-279: Contract gate — validate architecture and dimensions before CUDA init
        if let Err(e) = crate::contract_gate::validate_model_load_basic(
            &model.config.architecture,
            model.config.num_layers,
            model.config.hidden_dim,
            model.config.num_heads,
            model.config.num_kv_heads,
            model.config.intermediate_dim,
            model.config.vocab_size,
        ) {
            return Err(CudaInitError {
                error: crate::contract_gate::gate_error(e),
                model,
            });
        }

        // GH-280: Capability gate — refuse GPU if model requires unsupported ops.
        let model = Self::check_gpu_capability(model)?;

        let mut executor = match CudaExecutor::new(device_ordinal) {
            Ok(e) => e,
            Err(e) => {
                return Err(CudaInitError {
                    error: RealizarError::UnsupportedOperation {
                        operation: "CudaExecutor::new".to_string(),
                        reason: format!("CUDA initialization failed: {e}"),
                    },
                    model,
                });
            },
        };

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

        if let Err(e) =
            executor.init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        {
            return Err(CudaInitError {
                error: RealizarError::UnsupportedOperation {
                    operation: "init_kv_cache_gpu".to_string(),
                    reason: format!("GPU KV cache initialization failed: {e}"),
                },
                model,
            });
        }

        // PAR-118: Initialize Flash Decoding for split-K attention acceleration.
        // Five-Whys: batched_incremental_attention uses Grid=(num_heads,M,1) Block=(32,1,1)
        // = only 896 threads on RTX 4090 for 7B (28 heads). Flash Decoding splits KV cache
        // into chunks processed in parallel, achieving 1.5-2x decode speedup.
        // NOTE: flash_decode_enabled is used by BOTH the graphed decode path
        // (flash_decoding_graphed in attention.rs) and the batched path (batched.rs).
        // The batched path uses threshold 1024 to avoid triggering during normal prefill.
        if let Err(e) = executor.init_flash_decoding(num_heads, head_dim, max_seq_len, 1) {
            if verbose() {
                eprintln!(
                    "[PAR-118] Flash Decoding init failed: {e}, falling back to sequential attention"
                );
            }
            // Non-fatal: sequential attention still works, just slower
        }

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

        // PAR-083: Pre-allocate embedding buffer (hidden_dim f32s) to avoid per-token malloc.
        // Five-Whys: embed() heap alloc per token → eliminated by reusing this buffer.
        let embed_buf = vec![0.0f32; model.config.hidden_dim];

        let cuda_model = Self {
            model,
            executor,
            device_name,
            memory_info,
            embed_buf,
        };

        // GH-199 ROOT CAUSE B + PARITY-GATE: preload weights and verify GPU correctness.
        cuda_model.preload_and_verify()
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

    /// Consume CUDA wrapper and return inner model (for CPU fallback)
    #[must_use]
    pub fn into_model(self) -> OwnedQuantizedModel {
        self.model
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

// =============================================================================
// PARITY GATE: Load-time mathematical proof of GPU/CPU equivalence
// =============================================================================
//
// Toyota Way: Jidoka (自働化) — stop-the-line on defect.
//
// Just as build.rs refuses to compile if ALG-001 through ALG-009 fail,
// this gate refuses to construct `OwnedQuantizedModelCuda` if GPU and CPU
// compute different functions.
//
// An `OwnedQuantizedModelCuda` that passes this gate is PROVEN to produce
// the same output as CPU. One that fails CANNOT be constructed.
//
// Contract: layer-parity-v1.yaml
// Tolerance: cosine_similarity ≥ 0.99 on first-token logits

/// Minimum cosine similarity for parity gate to pass.
/// 0.99 allows for quantized GEMV rounding but catches completely wrong computation.
/// The 1.5B model achieves 0.999997; anything below 0.99 is catastrophically wrong.
const PARITY_GATE_COSINE_MIN: f32 = 0.99;

include!("mod_part_02.rs");
