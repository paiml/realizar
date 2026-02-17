//! Single-token forward pass with KV cache
//!
//! Contains forward_single_with_cache and forward_single_with_cache_adaptive.
//! These are the decode-phase entry points for autoregressive generation.

use crate::error::Result;
use crate::gguf::ops;
use crate::gguf::{
    DispatchMetrics, InferenceScratchBuffer, OwnedQuantizedKVCache, OwnedQuantizedLayer,
    OwnedQuantizedModel, GGUF_TYPE_Q4_K,
};

include!("single_part_02.rs");
