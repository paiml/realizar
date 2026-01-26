//! GPU Scheduler Module (PMAT-802)
//!
//! Shattered from gpu/cuda_scheduler.rs

pub mod batch;
mod core;
mod kv;
mod model;

#[cfg(feature = "cuda")]
pub use core::CudaScheduler;
pub use model::{
    AttentionBuffers, BlockWeights, GpuGenerateConfig, GpuModel, GpuModelConfig, WeightType,
};

// Re-export KV cache functions for external use (public API)
#[allow(unused_imports)]
pub use kv::{forward_gpu_incremental, forward_gpu_with_cache, generate_with_cache};
