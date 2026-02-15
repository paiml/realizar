//! Token generation for OwnedQuantizedModel
//!
//! Contains generate, generate_with_cache, generate_with_cache_streaming,
//! generate_with_scratch, and sampling methods.

use crate::error::{RealizarError, Result};
use crate::gguf::ops;
#[cfg(feature = "gpu")]
use crate::gguf::DispatchMetrics;
use crate::gguf::{
    InferenceScratchBuffer, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
};
use rand::Rng;

include!("generation_part_02.rs");
include!("generation_part_03.rs");
