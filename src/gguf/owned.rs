//! Owned quantized model types
//!
//! Models with owned (copied) quantized weights for Arc sharing.
//!
//! NOTE: During migration, types are still defined in monolith.
//! This module re-exports them for organization.

// Re-export from monolith during migration
pub use super::monolith::{
    OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCached,
    OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
};

#[cfg(feature = "cuda")]
pub use super::monolith::OwnedQuantizedModelCuda;
