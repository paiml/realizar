//! Attention computation for OwnedQuantizedModel
//!
//! Contains apply_rope (rotary position embeddings), causal_attention,
//! and KV-cache based attention variants with GQA support.

use crate::error::{RealizarError, Result};
use crate::gguf::{GGUFConfig, OwnedQuantizedModel};

include!("attention_part_02.rs");
