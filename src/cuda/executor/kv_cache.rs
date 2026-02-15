//! KV Cache management for GPU-resident inference
//!
//! This module implements:
//! - PAR-018: GPU-Resident KV Cache initialization
//! - PAR-021: GQA (Grouped Query Attention) support
//! - PAR-119: Batched KV cache for multi-sequence processing
//! - QWEN-007: Q8 quantized KV cache for 4x memory reduction
//! - Cache reset, rollback, and continuation

use super::*;
use crate::quantize::Q8_0Block;

include!("kv_cache_part_02.rs");
include!("kv_cache_part_03.rs");
