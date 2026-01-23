//! GPU Scheduler Module (PMAT-802)
//!
//! Shattered from gpu/cuda_scheduler.rs

pub mod batch;
mod core;
mod kv;
mod model;

#[cfg(feature = "cuda")]
pub use core::CudaScheduler;
pub use model::{GpuModel, GpuModelConfig, GpuGenerateConfig, WeightType, AttentionBuffers, BlockWeights};

// Re-export KV cache functions for external use (public API)
#[allow(unused_imports)]
pub use kv::{forward_gpu_with_cache, forward_gpu_incremental, generate_with_cache};
