//! Inference support types for quantized model execution
//!
//! This module re-exports inference types from the monolith during migration.
//! These types will be moved here incrementally as the extraction progresses.
//!
//! Types included:
//! - `OwnedInferenceScratchBuffer`: Pre-allocated buffers for zero-allocation forward passes
//! - `ContiguousKVCache`: Contiguous memory KV cache for efficient attention
//! - `DispatchMetrics`: Performance metrics for inference dispatching

// Note: These types are currently accessed via monolith::* re-export.
// This module will contain actual implementations once extraction is complete.
// Re-exports will be enabled when types are moved from monolith to here.
