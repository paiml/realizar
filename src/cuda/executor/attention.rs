//! Attention mechanisms: incremental attention, flash decoding, tensor core attention
//!
//! This module implements:
//! - PAR-023: GPU-Resident Incremental Attention
//! - PAR-118: Flash Decoding for parallel KV processing
//! - PAR-065: Tensor Core Attention
//! - Batched attention for multi-sequence processing

use super::*;

include!("attention_part_02.rs");
include!("attention_part_03.rs");
