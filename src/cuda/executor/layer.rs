//! Transformer layer operations: SwiGLU FFN, full transformer layer, batched processing
//!
//! This module implements:
//! - PAR-023: GPU-Resident SwiGLU FFN
//! - PAR-044: GPU-Resident Transformer Layer
//! - PAR-111: Batched Transformer Layer for multi-sequence processing
//! - PAR-062: CUDA Graph-captured decode
//! - Full forward pass with all layers

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

include!("layer_part_02.rs");
include!("layer_part_03.rs");
