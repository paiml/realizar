//! Activation functions and element-wise GPU operations
//!
//! This module implements:
//! - PAR-023: Activation and Element-wise GPU Operations
//! - GELU, LayerNorm, RMSNorm kernels
//! - Residual add operations
//! - Batched RMSNorm, RoPE, SwiGLU
//! - Host convenience wrappers

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

include!("activations_part_02.rs");
include!("activations_part_03.rs");
