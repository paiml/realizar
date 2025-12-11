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
/// APR Transformer format for WASM-compatible LLM inference
///
/// Provides F32 transformer weights for fair APR vs GGUF comparison.
/// Designed for WASM compatibility - no SIMD requirements.
pub mod apr_transformer;
/// Audit trail and provenance logging
///
/// Per spec §12: Comprehensive audit record for every inference request.
/// Implements GDPR Article 13/14 and SOC 2 compliance requirements.
/// - Full provenance tracking (model hash, distillation lineage)
/// - Latency breakdown (preprocessing, inference, postprocessing)
/// - Quality gates (Jidoka: NaN check, confidence check)
pub mod audit;
/// Benchmark harness for model runner comparison
///
/// Implements the benchmark specification v1.1 with Toyota Way engineering principles:
/// - Dynamic CV-based stop-rule (Hoefler & Belli)
/// - Thermal throttling protocol
/// - ITL variance measurement
/// - KV-cache fragmentation detection
/// - KL-Divergence quality validation
pub mod bench;
/// Preflight validation protocol for deterministic benchmarking
///
/// Per spec v1.0.1, implements Toyota Way principles:
/// - Jidoka: Fail-fast validation, stop on anomaly
/// - Poka-yoke: Error-proofing through type-safe configurations
/// - Genchi Genbutsu: Verify actual system state
///
/// References:
/// - Hoefler & Belli SC'15: CV-based stopping
/// - Vitek & Kalibera EMSOFT'11: Reproducibility requirements
pub mod bench_preflight;
pub mod cache;
/// CLI command implementations (extracted for testability)
pub mod cli;
/// GGUF to APR Transformer converter
///
/// Converts GGUF models to APR format for fair comparison.
/// All weights are dequantized to F32 for WASM compatibility.
pub mod convert;
/// CUDA PTX generation for NVIDIA GPUs
///
/// Provides native CUDA kernel generation via trueno-gpu.
/// - Pure Rust PTX generation (no LLVM, no nvcc)
/// - Hand-optimized kernels: GEMM, Softmax, LayerNorm, Attention, Q4K
/// - FlashAttention-style tiled attention
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod error;
/// Model explainability (SHAP, Attention)
///
/// Per spec §13: Model explainability for APR classical ML models.
/// Implements SHAP TreeExplainer for tree ensembles and KernelSHAP for any model.
/// - TreeSHAP: O(TLD) complexity for tree-based models
/// - KernelSHAP: Model-agnostic with weighted linear regression
/// - Feature importance: Top-k features by absolute SHAP value
pub mod explain;
/// Unified model format detection (APR, GGUF, SafeTensors)
///
/// Per spec §3: Format Support Matrix - auto-detect from magic bytes.
/// APR is first-class, GGUF and SafeTensors are backwards-compatible.
pub mod format;
pub mod generate;
pub mod gguf;
/// GPU acceleration module (Phase 4: ≥100 tok/s target)
///
/// Implements GPU-accelerated matrix operations via Trueno's wgpu backend.
/// - GPU matmul shader for large matrix multiplications
/// - Hybrid CPU/GPU scheduling based on workload size
/// - Automatic fallback to SIMD when GPU unavailable
#[cfg(feature = "gpu")]
pub mod gpu;
/// Grammar-constrained generation for structured output
///
/// Implements GBNF-style grammar constraints for LLM generation.
/// - JSON schema validation
/// - Custom grammar rules (GBNF format)
/// - Token masking for efficient constrained generation
/// - State machine for tracking grammar state
pub mod grammar;
/// HTTP client for real model server benchmarking
///
/// Implements actual HTTP calls to external servers (vLLM, Ollama, llama.cpp).
/// **NO MOCK DATA** - measures real network latency and inference timing.
#[cfg(feature = "bench-http")]
pub mod http_client;
/// SIMD-accelerated inference engine using trueno
///
/// Provides high-performance transformer inference competing with llama.cpp.
/// Uses trueno's SIMD primitives for matrix operations.
pub mod inference;
pub mod layers;
pub mod memory;
#[cfg(feature = "server")]
pub mod metrics;
/// Unified model loader for APR, GGUF, and SafeTensors
///
/// Per spec §3.2 and §5: Combines format detection with model loading.
/// Supports all 18 APR model types.
pub mod model_loader;
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
/// PagedAttention KV cache management
///
/// Per spec §8.1: Efficient KV cache management based on vLLM's PagedAttention.
/// Reference: [4] Kwon et al. (2023) "Efficient Memory Management for LLM Serving"
/// - Physical pages: Fixed-size memory blocks for KV cache
/// - Page tables: Logical to physical mapping per sequence
/// - Copy-on-Write: Efficient prefix sharing between sequences
pub mod paged_kv;
/// Multi-GPU and Distributed Inference
///
/// Per spec §10: Implements parallelism strategies for 70B+ model inference.
/// Reference: [11] Shoeybi et al. (2019) "Megatron-LM: Training Multi-Billion Parameter LMs"
/// - Tensor Parallelism (TP): Split tensors across GPUs within node (2-8 GPUs)
/// - Pipeline Parallelism (PP): Split layers across GPUs/nodes (2-64 GPUs)
/// - Data Parallelism (DP): Replicate model, split batches
/// - ZeRO-Inference: Memory offload to CPU
pub mod parallel;
pub mod quantize;
#[cfg(feature = "server")]
pub mod registry;
pub mod safetensors;
/// Continuous batching scheduler
///
/// Per spec §8: Implements continuous batching for LLM serving based on vLLM/Orca.
/// Reference: [8] Yu et al. (2022) "Orca: A Distributed Serving System"
/// - Iteration-level scheduling: New requests join batch at any iteration
/// - Preemption: Low-priority requests can be preempted for high-priority
/// - Memory-aware: Respects KV cache limits when scheduling
pub mod scheduler;
#[cfg(feature = "aprender-serve")]
pub mod serve;
/// Speculative decoding for LLM inference acceleration
///
/// Per spec §8.3: Implements speculative decoding based on SGLang/DeepMind research.
/// Reference: [9] Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
/// - Draft model: Small model generates K candidate tokens
/// - Target model: Verifies all K tokens in single forward pass
/// - Rejection sampling: Maintains exact target distribution
/// - Speedup: Up to 3x with well-matched draft/target pairs
pub mod speculative;
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
