//! Inference Engine Module
//!
//! This module contains the active inference logic extracted from gguf/mod.rs
//! using the Strangler Fig pattern. The GGUF module remains a pure parser,
//! while this module handles the actual transformer forward passes.
//!
//! # Architecture
//!
//! - `cpu` - CPU inference with KV caching (forward_cached)
//! - `cuda` - CUDA-accelerated inference (forward_gpu_resident)
//! - `batch` - Batched inference for API serving
//!
//! # Production Methods (The "Survivors")
//!
//! These are the 6 methods identified as production-used:
//! 1. `OwnedQuantizedModel::forward_cached` -> `cpu::forward_cached`
//! 2. `OwnedQuantizedModel::forward_single_with_cache` -> internal
//! 3. `OwnedQuantizedModel::forward_single_with_cache_adaptive` -> internal
//! 4. `OwnedQuantizedModelCachedSync::forward_batch_with_gpu_ffn` -> `batch::forward_batch_with_gpu_ffn`
//! 5. `OwnedQuantizedModelCuda::forward_single_full_cuda_with_cache` -> `cuda::forward_full_cuda_with_cache`
//! 6. `OwnedQuantizedModelCuda::forward_gpu_resident` -> `cuda::forward_gpu_resident`

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "gpu")]
pub mod batch;

// Re-export main types for convenience
pub use cpu::CpuInferenceEngine;

#[cfg(feature = "cuda")]
pub use cuda::CudaInferenceEngine;

#[cfg(feature = "gpu")]
pub use batch::BatchInferenceEngine;
