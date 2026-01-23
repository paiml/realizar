//! Extracted layer operations for CudaExecutor
//!
//! Split from layer.rs (PMAT-802) to reduce module size while maintaining
//! performance through #[inline(always)] on critical paths.

mod batched;
mod forward;
mod graphed;
mod indexed;
