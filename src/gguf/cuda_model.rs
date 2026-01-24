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
//! # Example
//!
//! ```rust,ignore
//! let model = OwnedQuantizedModel::from_mapped(&mapped)?;
//! let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)?; // GPU 0
//!
//! // GPU-accelerated forward pass
//! let logits = cuda_model.forward_cuda(&tokens)?;
//! ```

// Re-export for backward compatibility
#[cfg(feature = "cuda")]
pub use crate::gguf::monolith::OwnedQuantizedModelCuda;

// Note: The full OwnedQuantizedModelCuda struct and impl remain in the monolith
// during the incremental extraction process. This module will eventually contain
// the full implementation once dependencies are resolved.
//
// Current status: Struct definition in monolith (lines 13698-13707)
// Target: Move struct + impl (~3200 lines) here
