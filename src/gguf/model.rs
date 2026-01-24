//! GGUF model types
//!
//! Memory-mapped and parsed GGUF models.
//!
//! NOTE: During migration, types are still defined in monolith.
//! This module re-exports them for organization.

// Re-export from monolith during migration
pub use super::monolith::{GGUFTransformer, GGUFTransformerLayer, MappedGGUFModel};
