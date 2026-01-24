//! Owned quantized model types
//!
//! Models with owned (copied) quantized weights for Arc sharing.
//!
//! NOTE: During migration, types are still defined in monolith.
//! This module re-exports them for organization.

// Re-export from monolith during migration
// Note: OwnedQuantizedModel is now in model.rs
// Note: OwnedQuantizedModelCuda is now in cuda_model.rs
// Note: OwnedQuantizedKVCache, QuantizedGenerateConfig accessed via monolith::*
pub use super::monolith::{OwnedQuantizedModelCached, OwnedQuantizedModelCachedSync};
