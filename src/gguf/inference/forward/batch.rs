//! Batched forward pass variants
//!
//! Contains forward_batch, forward_batch_gpu, forward_batch_with_cache,
//! and supporting batch matmul/attention helpers.

use crate::error::{RealizarError, Result};
use crate::gguf::ops;
use crate::gguf::{
    DispatchMetrics, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedModel,
    OwnedQuantizedTensor, QuantizedGenerateConfig, TokenBuffer, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K,
    GGUF_TYPE_Q6_K,
};

include!("batch_tiled_causal_owned.rs");
