//! GPU Model Adapters (PMAT-106)
//!
//! Adapters for converting different model formats to GpuModel.
//!
//! # Supported Formats
//!
//! - **APR** - Native APR format with Q4_0 quantization
//! - **SafeTensors** - HuggingFace SafeTensors format (planned)
//!
//! # Coverage Impact
//!
//! These adapters drive coverage for:
//! - `apr_transformer/q4_simd.rs`
//! - `gpu/scheduler/batch.rs`
//! - `api/openai_handlers.rs`

mod apr;

pub use apr::{AprToGpuAdapter, AprGpuError};
