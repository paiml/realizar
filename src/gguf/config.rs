//! GGUF configuration extraction
//!
//! Extracts model configuration from GGUF metadata.
//!
//! NOTE: During migration, types are still defined in monolith.
//! This module re-exports them for organization.

// Re-export from monolith during migration
pub use super::monolith::GGUFConfig;
