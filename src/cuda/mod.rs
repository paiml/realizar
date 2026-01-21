//! CUDA PTX Generation and Execution Module
//!
//! Provides NVIDIA CUDA-specific PTX code generation and execution via `trueno-gpu`.
//! This is an optional backend for maximum performance on NVIDIA hardware.
//!
//! ## Architecture
//!
//! ```text
//! +-----------------------+
//! |   CudaExecutor API    |  <- High-level execution API
//! +-----------------------+
//! |   CudaKernels API     |  <- PTX generation
//! +-----------------------+
//! |   trueno_gpu::driver  |  <- CUDA runtime (context, stream, memory)
//! +-----------------------+
//! |   trueno_gpu::kernels |  <- Hand-optimized PTX kernels
//! +-----------------------+
//! |   trueno_gpu::ptx     |  <- Pure Rust PTX generation
//! +-----------------------+
//! ```
//!
//! ## Module Organization
//!
//! - `kernels`: Kernel type definitions and PTX generation
//! - `memory`: GPU memory pool and staging buffers
//! - `types`: Weight loading types and transformer workspace
//! - `pipeline`: Async pipeline and PTX optimization
//! - `executor`: CUDA execution engine (TODO: split further - currently >800 lines)
//!
//! ## Available Kernels
//!
//! - **GEMM**: Matrix multiplication (naive, tiled, tensor core)
//! - **Softmax**: Numerically stable softmax with warp shuffle
//! - **LayerNorm**: Fused layer normalization
//! - **Attention**: FlashAttention-style tiled attention
//! - **Quantize**: Q4_K/Q5_K/Q6_K dequantization-fused GEMM/GEMV

// Submodules
pub mod kernels;
pub mod memory;
pub mod pipeline;
pub mod types;

// Re-export everything for backwards compatibility
pub use kernels::{CudaKernels, KernelType};
pub use memory::{
    GpuBufferHandle, GpuMemoryPool, PinnedHostBuffer, PoolStats, SizeClass, StagingBufferPool,
    StagingPoolStats, TransferMode,
};
pub use pipeline::{
    presets, AsyncPipeline, BankConflictStrategy, MemoryPattern, PtxOptimizationHints,
    PtxOptimizer, RegisterTiling,
};
pub use types::{IndexedLayerWeights, TransformerWorkspace, WeightQuantType};

// The executor module is included inline for now due to its massive size (~15K lines)
// TODO: Split CudaExecutor into smaller modules:
// - executor/core.rs: Basic context and profiling
// - executor/weights.rs: Weight loading and caching
// - executor/gemm.rs: GEMM/GEMV operations
// - executor/kernels.rs: Activation and attention kernels
// - executor/graph.rs: CUDA graph capture and replay
mod executor;
pub use executor::CudaExecutor;
