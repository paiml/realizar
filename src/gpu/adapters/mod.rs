//! GPU Model Adapters (PMAT-106)
//!
//! Adapters for converting different model formats to GpuModel.
//!
//! # Supported Formats
//!
//! - **APR F32** - Native `.apr` format with F32 weights
//! - **APR Q4** - GGUF models with Q4_0 quantization
//! - **SafeTensors** - HuggingFace SafeTensors format (planned)
//!
//! # Coverage Impact
//!
//! These adapters drive coverage for:
//! - `apr_transformer/mod.rs` (F32)
//! - `apr_transformer/q4_simd.rs` (Q4)
//! - `gpu/scheduler/batch.rs`
//! - `api/openai_handlers.rs`

mod apr;

pub use apr::{AprF32ToGpuAdapter, AprToGpuAdapter, AprGpuError};
