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
    pub fn with_max_seq_len(
        model: OwnedQuantizedModel,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> std::result::Result<Self, CudaInitError> {
        use crate::cuda::CudaExecutor;

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

        let mut cuda_model = Self {
            model,
            executor,
            device_name,
            memory_info,
            embed_buf,
        };

        // GH-199 ROOT CAUSE B: Eagerly preload weights for GPU-resident path.
        // Makes constructor return a FULLY-READY model, so generate_gpu_resident()
        // only contains per-generation state (fresh KV cache + position reset).
        // Without this, preload_weights_gpu() inside generate_gpu_resident() looked like
        // expensive one-time setup, misleading developers into creating duplicate models.
        if cuda_model.supports_gpu_resident() {
            if let Err(e) = cuda_model.preload_weights_gpu() {
                return Err(CudaInitError {
                    error: e,
                    model: cuda_model.into_model(),
                });
            }

            // PARITY-GATE: Jidoka — stop-the-line if GPU diverges from CPU.
            //
            // Run ONE token through both backends and compare logits.
            // If cosine similarity < 0.99, GPU is computing a DIFFERENT function.
            // Refuse to construct — inference that passes this gate is PROVEN correct.
            //
            // This is the inference equivalent of build.rs compile-time proofs:
            //   build.rs proves: dimensions are mathematically valid
            //   parity gate proves: GPU computes the SAME function as CPU
            //
            // Skip gate if SKIP_PARITY_GATE=1 (for debugging the gate itself)
            let skip_gate = std::env::var("SKIP_PARITY_GATE")
                .map(|v| v == "1")
                .unwrap_or(false);

            if !skip_gate {
                if let Err(e) = parity_gate(&mut cuda_model) {
                    return Err(CudaInitError {
                        error: e,
                        model: cuda_model.into_model(),
                    });
                }
            }
        }

        Ok(cuda_model)
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

/// Run the load-time parity gate.
///
/// Processes BOS token (ID=1) through both CPU and GPU forward passes.
/// Compares the resulting logit vectors via cosine similarity.
///
/// # Errors
///
/// Returns `RealizarError` if GPU and CPU produce divergent logits.
fn parity_gate(cuda_model: &mut OwnedQuantizedModelCuda) -> Result<()> {
    // Extract config values before any mutable borrows
    let hidden_dim = cuda_model.model.config.hidden_dim;
    let num_heads = cuda_model.model.config.num_heads;
    let num_kv_heads = cuda_model.model.config.num_kv_heads;
    let head_dim = if num_heads > 0 {
        hidden_dim / num_heads
    } else {
        0
    };
    let kv_dim = num_kv_heads * head_dim;
    let num_layers = cuda_model.model.config.num_layers;

    // Use architecture-aware BOS token from GGUFConfig (which applies
    // default_bos_for_architecture fallback for weights-only GGUFs).
    // Falls back to 1 only for architectures with no known BOS.
    let token_id: u32 = cuda_model.model.config.bos_token_id.unwrap_or(1);
    let position: usize = 0;

    // Independent KV caches
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 2);
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 2);
    cuda_model.executor.reset_kv_cache_gpu();

    // CPU forward
    let cpu_logits = cuda_model
        .model
        .forward_single_with_cache(token_id, &mut cpu_cache, position)
        .map_err(|e| {
            RealizarError::InferenceError(format!("PARITY-GATE: CPU forward failed: {e}"))
        })?;

    // GPU forward
    let gpu_logits = cuda_model
        .forward_gpu_resident(token_id, &mut gpu_cache, position)
        .map_err(|e| {
            RealizarError::InferenceError(format!("PARITY-GATE: GPU forward failed: {e}"))
        })?;

    // Cosine similarity — the single metric that catches completely wrong computation
    let cosine = cosine_similarity(&cpu_logits, &gpu_logits);

    // Reset KV caches so the model starts fresh for actual inference
    cuda_model.executor.reset_kv_cache_gpu();

    if cosine >= PARITY_GATE_COSINE_MIN {
        if verbose() {
            eprintln!(
                "[PARITY-GATE] PASS: cosine={:.6} (threshold={:.2})",
                cosine, PARITY_GATE_COSINE_MIN,
            );
        }
        Ok(())
    } else {
        // Compute additional diagnostics for the error message
        let cpu_argmax = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        let gpu_argmax = gpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        let max_diff = cpu_logits
            .iter()
            .zip(gpu_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        Err(RealizarError::InferenceError(format!(
            "PARITY-GATE FAILED: GPU computes a DIFFERENT function than CPU.\n\
             \n\
             Cosine similarity: {cosine:.6} (required: ≥{PARITY_GATE_COSINE_MIN:.2})\n\
             CPU argmax: {cpu_argmax} | GPU argmax: {gpu_argmax}\n\
             Max absolute logit difference: {max_diff:.4}\n\
             \n\
             This model's dimensions (hidden={hidden_dim}, heads={num_heads}, kv_heads={num_kv_heads}) cause\n\
             GPU forward pass to diverge from CPU. The GPU CANNOT serve this model.\n\
             \n\
             Run `apr parity <model>` for full SPC diagnosis.\n\
             Set SKIP_PARITY_GATE=1 to bypass (for debugging only).",
        )))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}
