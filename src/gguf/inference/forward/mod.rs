//! Forward pass implementations for OwnedQuantizedModel
//!
//! This module contains all forward pass variants:
//! - `core.rs`: Basic forward and forward_cached (prefill)
//! - `single.rs`: Single-token forward with cache (decode)
//! - `batch.rs`: Batched forward pass variants

mod batch;
mod core;
mod single;
