//! Indexed layer operations for optimized decode path
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-043: transformer_layer_indexed (hot path for decode)
//! - Private helpers for indexed operations

#![allow(clippy::wildcard_imports)]

use super::super::*;

include!("indexed_part_02.rs");
include!("indexed_part_03.rs");
