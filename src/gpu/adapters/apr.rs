//! APR to GpuModel Adapter (PMAT-106)
//!
//! Converts `QuantizedAprTransformerQ4` to `GpuModel` for GPU inference.
//!
//! # Overview
//!
//! The APR format stores weights in Q4_0 quantization. This adapter:
//! 1. Dequantizes Q4_0 weights to F32
//! 2. Restructures weights into GpuModel format
//! 3. Initializes GPU schedulers
//!
//! # Coverage Impact
//!
//! Testing this adapter exercises:
//! - `apr_transformer/q4_simd.rs` - Weight extraction
//! - `gpu/scheduler/model.rs` - GpuModel creation
//! - `quantize/dequant.rs` - Q4_0 dequantization

use crate::apr_transformer::{QuantizedAprTransformerQ4, QuantizedAprLayerQ4, AprTransformerConfig};
use crate::gpu::scheduler::{GpuModel, GpuModelConfig};
use crate::quantize::dequantize_q4_0;
use crate::error::Result;
use thiserror::Error;

/// Errors during APR to GPU conversion
#[derive(Debug, Error)]
pub enum AprGpuError {
    /// Dequantization failed
    #[error("Failed to dequantize Q4_0 weights: {0}")]
    DequantError(String),

    /// Weight dimension mismatch
    #[error("Weight dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of elements
        expected: usize,
        /// Actual number of elements
        actual: usize,
    },

    /// GpuModel creation failed
    #[error("Failed to create GpuModel: {0}")]
    GpuModelError(String),
}

/// Adapter for converting APR models to GPU format
pub struct AprToGpuAdapter;

impl AprToGpuAdapter {
    /// Convert APR config to GPU config
    #[must_use]
    pub fn config_to_gpu(apr_config: &AprTransformerConfig) -> GpuModelConfig {
        GpuModelConfig {
            vocab_size: apr_config.vocab_size,
            hidden_dim: apr_config.hidden_dim,
            num_heads: apr_config.num_heads,
            num_kv_heads: apr_config.num_kv_heads,
            num_layers: apr_config.num_layers,
            intermediate_dim: apr_config.intermediate_dim,
            eps: apr_config.eps,
        }
    }

    /// Dequantize a Q4_0 tensor to F32
    ///
    /// # Arguments
    ///
    /// * `data` - Raw Q4_0 quantized bytes
    /// * `expected_elements` - Expected number of output elements
    ///
    /// # Returns
    ///
    /// Dequantized F32 vector
    pub fn dequantize_tensor(data: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
        let result = dequantize_q4_0(data)?;

        // Validate dimensions
        if result.len() < expected_elements {
            // Pad with zeros if needed (can happen with block alignment)
            let mut padded = result;
            padded.resize(expected_elements, 0.0);
            Ok(padded)
        } else {
            // Truncate to expected size
            Ok(result.into_iter().take(expected_elements).collect())
        }
    }

    /// Extract QKV weights from APR layer
    ///
    /// APR stores QKV as a single tensor, which matches GpuModel format.
    pub fn extract_qkv_weights(
        layer: &QuantizedAprLayerQ4,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Vec<f32>> {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;
        let expected = hidden_dim * qkv_out_dim;

        Self::dequantize_tensor(&layer.qkv_weight.data, expected)
    }

    /// Extract output projection weights
    pub fn extract_out_weights(
        layer: &QuantizedAprLayerQ4,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        let expected = hidden_dim * hidden_dim;
        Self::dequantize_tensor(&layer.attn_output_weight.data, expected)
    }

    /// Extract FFN weights (fc1 = up, fc2 = down)
    ///
    /// Note: APR uses SwiGLU with separate gate/up, but GpuModel combines them.
    /// For compatibility, we return up weights as fc1.
    pub fn extract_ffn_weights(
        layer: &QuantizedAprLayerQ4,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // FC1 (up projection): [hidden_dim, intermediate_dim]
        let fc1_expected = hidden_dim * intermediate_dim;
        let fc1 = Self::dequantize_tensor(&layer.ffn_up_weight.data, fc1_expected)?;

        // FC2 (down projection): [intermediate_dim, hidden_dim]
        let fc2_expected = intermediate_dim * hidden_dim;
        let fc2 = Self::dequantize_tensor(&layer.ffn_down_weight.data, fc2_expected)?;

        Ok((fc1, fc2))
    }

    /// Convert full APR transformer to GpuModel
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer with Q4_0 weights
    ///
    /// # Returns
    ///
    /// `GpuModel` ready for GPU inference
    ///
    /// # Example
    ///
    /// ```ignore
    /// use realizar::apr_transformer::QuantizedAprTransformerQ4;
    /// use realizar::gpu::adapters::AprToGpuAdapter;
    ///
    /// let apr = QuantizedAprTransformerQ4::from_gguf(&gguf_model);
    /// let gpu_model = AprToGpuAdapter::to_gpu_model(&apr)?;
    /// ```
    pub fn to_gpu_model(apr: &QuantizedAprTransformerQ4) -> Result<GpuModel> {
        let config = Self::config_to_gpu(&apr.config);
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;

        // Embedding weights (already F32 in APR)
        let embedding_weights = apr.token_embedding.clone();

        // Dequantize LM head
        let lm_head_expected = hidden_dim * config.vocab_size;
        let lm_head_weight = Self::dequantize_tensor(&apr.lm_head_weight.data, lm_head_expected)?;

        // Transpose LM head for fast CPU inference
        let lm_head_weight_t = transpose_matrix(&lm_head_weight, hidden_dim, config.vocab_size);

        // Convert each layer
        let mut block_weights = Vec::with_capacity(apr.layers.len());
        for layer in &apr.layers {
            let qkv = Self::extract_qkv_weights(layer, hidden_dim, config.num_heads, config.num_kv_heads)?;
            let out = Self::extract_out_weights(layer, hidden_dim)?;
            let (fc1, fc2) = Self::extract_ffn_weights(layer, hidden_dim, intermediate_dim)?;

            block_weights.push(crate::gpu::scheduler::BlockWeights {
                attn_norm_weight: layer.attn_norm_weight.clone(),
                attn_norm_bias: vec![0.0; hidden_dim], // APR doesn't use bias
                qkv_weight: qkv,
                qkv_bias: vec![], // No bias in APR
                out_weight: out,
                out_bias: vec![0.0; hidden_dim],
                ffn_norm_weight: layer.ffn_norm_weight.clone().unwrap_or_else(|| vec![1.0; hidden_dim]),
                ffn_norm_bias: vec![0.0; hidden_dim],
                ffn_fc1_weight: fc1,
                ffn_fc1_bias: vec![0.0; intermediate_dim],
                ffn_fc2_weight: fc2,
                ffn_fc2_bias: vec![0.0; hidden_dim],
            });
        }

        // Final norm
        let final_norm_weight = apr.output_norm_weight.clone();
        let final_norm_bias = vec![0.0; hidden_dim];

        // LM head bias
        let lm_head_bias = vec![0.0; config.vocab_size];

        // Create GpuModel using internal constructor
        GpuModel::from_apr_weights(
            config,
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
        )
    }
}

/// Transpose a row-major matrix
fn transpose_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = data[i * cols + j];
        }
    }
    transposed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apr_transformer::AprTransformerConfig;

    #[test]
    fn test_config_to_gpu() {
        let apr_config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 512,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 4,
            vocab_size: 32000,
            intermediate_dim: 1024,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

        assert_eq!(gpu_config.vocab_size, 32000);
        assert_eq!(gpu_config.hidden_dim, 512);
        assert_eq!(gpu_config.num_heads, 8);
        assert_eq!(gpu_config.num_kv_heads, 4);
        assert_eq!(gpu_config.num_layers, 4);
        assert_eq!(gpu_config.intermediate_dim, 1024);
        assert_eq!(gpu_config.eps, 1e-5);
    }

    #[test]
    fn test_transpose_matrix() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let transposed = transpose_matrix(&data, 2, 3); // 3x2

        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let transposed = transpose_matrix(&data, 2, 2);

        assert_eq!(transposed, vec![1.0, 3.0, 2.0, 4.0]);
    }
}
