//! Test helper functions for api tests
//!
//! This module contains shared test utilities used across multiple test parts.
//! Separated for PMAT compliance (<2000 lines per file).

use super::*;
use axum::Router;

/// Create a test application with demo state
pub fn create_test_app() -> Router {
    let state = AppState::demo().expect("test");
    create_router(state)
}

/// Helper to create test quantized model for IMP-116 tests
#[cfg(feature = "gpu")]
pub fn create_test_quantized_model(
    config: &crate::gguf::GGUFConfig,
) -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{
        OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
        GGUF_TYPE_Q4_K,
    };

    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let vocab_size = config.vocab_size;

    // Create Q4_K tensor data helper
    // Q4_K uses row-major storage where each row has ceil(in_dim/256) super-blocks.
    // Each super-block is 144 bytes and covers 256 values.
    fn create_q4k_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let super_blocks_per_row = in_dim.div_ceil(256);
        let bytes_per_row = super_blocks_per_row * 144;
        let data_size = out_dim * bytes_per_row;
        OwnedQuantizedTensor {
            data: vec![0u8; data_size],
            qtype: GGUF_TYPE_Q4_K,
            in_dim,
            out_dim,
        }
    }

    let layers = (0..config.num_layers)
        .map(|_| OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Fused(create_q4k_data(hidden_dim, hidden_dim * 3)),
            qkv_bias: None,
            attn_output_weight: create_q4k_data(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4k_data(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4k_data(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        })
        .collect();

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_data(hidden_dim, vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}
