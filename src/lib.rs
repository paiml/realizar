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

// Clippy allows (MUST come after deny/warn to override them)
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::large_stack_arrays)] // Test data
#![allow(clippy::cast_possible_wrap)] // u64 -> i64 for timestamps is safe
#![allow(clippy::cast_precision_loss)] // usize -> f32 precision loss is acceptable
#![allow(clippy::cast_possible_truncation)] // u128 -> u64 etc for metrics is safe
#![allow(clippy::cast_sign_loss)] // Metrics conversions are safe
#![allow(clippy::too_many_lines)] // Some handlers are naturally long
#![allow(clippy::must_use_candidate)] // Not all methods need #[must_use]
#![allow(clippy::doc_markdown)] // Allow technical terms without backticks
#![allow(clippy::redundant_clone)] // Sometimes clarity > performance
#![allow(clippy::uninlined_format_args)] // Prefer explicit format args
#![allow(clippy::single_match_else)] // Sometimes clearer than if-let
#![allow(clippy::unnecessary_to_owned)] // Allow explicit .to_string()
#![allow(clippy::single_char_pattern)] // Allow "x" instead of 'x' in contains()
#![allow(clippy::missing_panics_doc)] // Allow missing Panics doc sections
#![allow(clippy::if_not_else)] // Allow if !condition { } else { }
#![allow(clippy::manual_let_else)] // Allow manual let-else patterns
#![allow(clippy::float_cmp)] // Allow float comparisons in tests
#![allow(clippy::cast_lossless)] // Allow i32 to f64 casts
#![allow(clippy::approx_constant)] // Allow approximate PI
#![allow(clippy::manual_range_contains)] // Allow manual range checks
#![allow(clippy::same_item_push)] // Allow pushing same items in tests

#[cfg(feature = "server")]
pub mod api;
/// Aprender .apr format support (PRIMARY inference format)
///
/// The .apr format is the native format for the sovereign AI stack.
/// GGUF and safetensors are supported as fallback formats.
pub mod apr;
/// Benchmark harness for model runner comparison
///
/// Implements the benchmark specification v1.1 with Toyota Way engineering principles:
/// - Dynamic CV-based stop-rule (Hoefler & Belli)
/// - Thermal throttling protocol
/// - ITL variance measurement
/// - KV-cache fragmentation detection
/// - KL-Divergence quality validation
pub mod bench;
pub mod cache;
/// CLI command implementations (extracted for testability)
pub mod cli;
pub mod error;
pub mod generate;
pub mod gguf;
/// HTTP client for real model server benchmarking
///
/// Implements actual HTTP calls to external servers (vLLM, Ollama, llama.cpp).
/// **NO MOCK DATA** - measures real network latency and inference timing.
#[cfg(feature = "bench-http")]
pub mod http_client;
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
/// Pacha URI scheme support for model loading
pub mod uri;

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
