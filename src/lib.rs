//! # Realizar
//!
//! Pure Rust, portable, high-performance ML library with unified CPU/GPU/WASM support.
//!
//! Realizar (Spanish: "to accomplish, to achieve") provides a unified API for machine learning
//! operations that automatically dispatches to the optimal backend based on data size,
//! operation complexity, and available hardware.
//!
//! ## Features
//!
//! - **Unified API**: Single interface for CPU SIMD, GPU, and WASM execution
//! - **Native Integration**: First-class support for `trueno` and `aprender`
//! - **Memory Safe**: Zero unsafe code in public API, leveraging Rust's type system
//! - **Production Ready**: EXTREME TDD, 85%+ coverage, zero tolerance for defects
//!
//! ## Example
//!
//! ```rust
//! use realizar::Tensor;
//!
//! // Create tensors
//! let a = Tensor::from_vec(vec![3, 3], vec![
//!     1.0, 2.0, 3.0,
//!     4.0, 5.0, 6.0,
//!     7.0, 8.0, 9.0,
//! ]).unwrap();
//!
//! // Check tensor properties
//! assert_eq!(a.shape(), &[3, 3]);
//! assert_eq!(a.ndim(), 2);
//! assert_eq!(a.size(), 9);
//! ```
//!
//! ## Future Operations (Phase 1+)
//!
//! ```rust,ignore
//! // Element-wise operations (SIMD-accelerated) - Coming in Phase 1
//! let sum = a.add(&b).unwrap();
//!
//! // Matrix multiplication (GPU-accelerated for large matrices) - Coming in Phase 2
//! let product = a.matmul(&b).unwrap();
//! ```
//!
//! ## Architecture
//!
//! Realizar is built on top of:
//! - **Trueno**: Low-level compute primitives with SIMD/GPU/WASM backends
//! - **Aprender**: High-level ML algorithms (will be refactored to use Realizar)
//!
//! ## Quality Standards
//!
//! Following EXTREME TDD methodology:
//! - Test Coverage: ≥85%
//! - Mutation Score: ≥80%
//! - TDG Score: ≥90/100
//! - Clippy Warnings: 0 (enforced)
//! - Cyclomatic Complexity: ≤10 per function

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
// Multiple crate versions are acceptable for dependencies
// #![warn(clippy::cargo)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod gguf;
pub mod layers;
pub mod quantize;
pub mod safetensors;
pub mod tensor;
pub mod tokenizer;

// Re-exports for convenience
pub use error::{RealizarError, Result};
pub use tensor::Tensor;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert!(VERSION.starts_with("0."));
    }
}
