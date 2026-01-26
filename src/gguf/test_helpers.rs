//! Test helpers for GGUF module testing
//!
//! This module contains shared test utilities that are used across
//! multiple test files in the GGUF module shatter.
//!
//! ## Contents
//!
//! - `create_test_model_with_config`: Creates a minimal OwnedQuantizedModel for testing
//! - `create_q4k_test_data`: Creates Q4_K quantized tensor data for testing
//!
//! ## Usage
//!
//! ```rust,ignore
//! #[cfg(test)]
//! use crate::gguf::test_helpers::{create_test_model_with_config, create_q4k_test_data};
//! ```

use crate::gguf::{
    GGUFConfig, OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
};

/// Create a test model with specific configuration
///
/// This helper creates a minimal `OwnedQuantizedModel` with deterministic
/// test weights for verifying attention, FFN, and other model operations.
///
/// # Arguments
///
/// * `config` - The model configuration to use
///
/// # Returns
///
/// An `OwnedQuantizedModel` with test weights
pub(crate) fn create_test_model_with_config(config: &GGUFConfig) -> OwnedQuantizedModel {
    // Create minimal model weights for testing
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);

    // QKV projection: hidden_dim -> hidden_dim + 2*kv_dim (Q + K + V)
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let qkv_weight = create_q4k_test_data(hidden_dim, qkv_out_dim);

    // Output projection: hidden_dim -> hidden_dim
    let attn_output_weight = create_q4k_test_data(hidden_dim, hidden_dim);

    // FFN weights
    let ffn_up_weight = create_q4k_test_data(hidden_dim, intermediate_dim);
    let ffn_down_weight = create_q4k_test_data(intermediate_dim, hidden_dim);

    // Layer norm weights
    let attn_norm_weight = vec![1.0f32; hidden_dim];

    let layer = OwnedQuantizedLayer {
        attn_norm_weight,
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(qkv_weight),
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
    };

    let token_embedding = vec![0.1f32; vocab_size * hidden_dim];
    let output_norm_weight = vec![1.0f32; hidden_dim];
    let lm_head_weight = create_q4k_test_data(hidden_dim, vocab_size);

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding,
        layers: vec![layer],
        output_norm_weight,
        output_norm_bias: None,
        lm_head_weight,
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create Q4_K test data for given dimensions
///
/// Q4_K uses row-major storage where each row has ceil(in_dim/256) super-blocks.
/// Each super-block is 144 bytes and covers 256 values.
///
/// # Arguments
///
/// * `in_dim` - Input dimension (number of columns)
/// * `out_dim` - Output dimension (number of rows)
///
/// # Returns
///
/// An `OwnedQuantizedTensor` with Q4_K quantized test data
pub(crate) fn create_q4k_test_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Row-major storage: each row needs ceil(in_dim/256) super-blocks
    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;
    let data_size = out_dim * bytes_per_row;
    let mut data = vec![0u8; data_size];

    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let offset = row * bytes_per_row + sb * 144;
            // d=1.0 in f16 format
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            // dmin=0
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
            // Fill scales and quantized values with deterministic test pattern
            for i in 4..144 {
                data[offset + i] = ((row + sb + i) % 16) as u8;
            }
        }
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: 12, // Q4_K
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_q4k_test_data_basic() {
        let tensor = create_q4k_test_data(256, 64);
        assert_eq!(tensor.in_dim, 256);
        assert_eq!(tensor.out_dim, 64);
        assert_eq!(tensor.qtype, 12); // Q4_K
                                      // 1 super-block per row, 144 bytes each, 64 rows
        assert_eq!(tensor.data.len(), 64 * 144);
    }

    #[test]
    fn test_create_q4k_test_data_multi_superblock() {
        // 512 values needs 2 super-blocks per row
        let tensor = create_q4k_test_data(512, 32);
        assert_eq!(tensor.in_dim, 512);
        assert_eq!(tensor.out_dim, 32);
        // 2 super-blocks per row, 144 bytes each, 32 rows
        assert_eq!(tensor.data.len(), 32 * 2 * 144);
    }

    #[test]
    fn test_create_test_model_with_config_basic() {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            rope_theta: 10000.0,
            context_length: 512,
            eps: 1e-5,
            rope_type: 0,
        };

        let model = create_test_model_with_config(&config);
        assert_eq!(model.config.hidden_dim, 64);
        assert_eq!(model.layers.len(), 1);
        assert_eq!(model.token_embedding.len(), 100 * 64);
    }
}
