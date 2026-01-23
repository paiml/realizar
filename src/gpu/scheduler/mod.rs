//! GPU Scheduler Module (PMAT-802)
//!
//! Shattered from gpu/cuda_scheduler.rs

mod batch;
mod core;
mod kv;
mod model;

#[cfg(feature = "cuda")]
pub use core::CudaScheduler;
pub use model::{GpuModel, GpuModelConfig, GpuGenerateConfig, WeightType, AttentionBuffers};

// Re-export KV cache functions for external use (public API)
#[allow(unused_imports)]
pub use kv::{forward_gpu_with_cache, forward_gpu_incremental, generate_with_cache};
