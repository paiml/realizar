//! GGUF (GPT-Generated Unified Format) parser
//!
//! Pure Rust implementation of GGUF binary format reader.
//! Used by llama.cpp, Ollama, and compatible tools.
//!
//! Format specification: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//!
//! ## Module Structure
//!
//! This module is being incrementally refactored from a 54K-line monolith
//! into focused submodules for better testability and coverage.

// GGUF Module Structure
//
// Incremental shatter of src/gguf.rs (54K lines) into domain modules.
// Each module should be ≤800 lines for testability.
//
// Shatter Plan (19 modules from 54K lines):
// 🚧 types.rs: Additional tests for constants (~50 lines)
// - header.rs: GGUFHeader, TensorInfo
// - model.rs: GGUFModel, MappedGGUFModel
// - config.rs: GGUFConfig
// - transformer.rs: GGUFTransformer, GGUFTransformerLayer
// - quantized.rs: Quantized tensor types
// - owned.rs: OwnedQuantized* types
// - cached.rs: Cached model variants
// - batching.rs: Batch processing
// - scheduling.rs: Request scheduling
// - gpu_buffer.rs: GPU buffer management
// - prefix_cache.rs: Prefix caching
// - kv_cache.rs: KV cache types
// - inference.rs: OwnedQuantizedModel inference impl
// - cuda.rs: CUDA-specific code
//
// Migration Strategy: Include monolith, gradually extract, re-export all

// Additional tests for constants (will contain extracted code later)
mod types;

// Temporary: include the monolith during migration
// Code will be moved to proper modules incrementally
#[path = "../gguf_monolith.rs"]
mod monolith;

// Re-export everything from monolith for backward compatibility
pub use monolith::*;
