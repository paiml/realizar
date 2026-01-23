//! Extracted layer operations for CudaExecutor
//!
//! Split from layer.rs (PMAT-802) to reduce module size while maintaining
//! performance through #[inline(always)] on critical paths.

mod batched;
mod ffn;
mod forward;
mod graphed;
mod indexed;

pub use ffn::{fused_ffn_swiglu_gpu, fused_ffn_swiglu_gpu_true_dp4a};
