//! Quantized tensor types
//!
//! Types for working with quantized GGUF tensors.
//!
//! NOTE: During migration, types are still defined in monolith.
//! This module re-exports them for organization.

// Re-export from monolith during migration
pub use super::monolith::{
    OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedTensor, QKVWeights, QuantizedTensorRef,
};
