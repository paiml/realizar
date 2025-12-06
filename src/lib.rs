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
// Large stack arrays are acceptable in tests for test data
#![allow(clippy::large_stack_arrays)]

#[cfg(feature = "server")]
pub mod api;
pub mod cache;
pub mod error;
pub mod generate;
pub mod gguf;
pub mod layers;
pub mod memory;
#[cfg(feature = "server")]
pub mod metrics;
pub mod moe;
/// Observability: metrics, tracing, and A/B testing
///
/// Safe numeric casts for observability metrics:
/// - Duration microseconds: u128 -> u64 (durations under 584,942 years won't overflow)
/// - Timestamps: u128 -> u64 (Unix epoch nanoseconds/microseconds fit in u64 until ~2554)
/// - Percentages: integer -> f64 (exact for values under 2^53)
#[cfg(feature = "server")]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_sign_loss)]
pub mod observability;
pub mod quantize;
#[cfg(feature = "server")]
pub mod registry;
pub mod safetensors;
#[cfg(feature = "aprender-serve")]
pub mod serve;
pub mod stats;
pub mod tensor;
pub mod viz;
/// Model warm-up and pre-loading
pub mod warmup;

/// AWS Lambda handler for aprender model serving
#[cfg(feature = "lambda")]
pub mod lambda;
/// Multi-target deployment support (Lambda, Docker, WASM)
pub mod target;
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
        // VERSION is a compile-time constant from CARGO_PKG_VERSION, so it's never empty
        assert!(VERSION.starts_with("0."));
        assert!(VERSION.len() >= 3); // At least "0.x"
        assert!(VERSION.contains('.'));
    }
}
