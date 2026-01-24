//! APR to GpuModel Adapter (PMAT-106)
//!
//! Converts APR transformers to `GpuModel` for GPU inference.
//!
//! # Overview
//!
//! This module provides adapters for both F32 and Q4 APR formats:
//! - [`AprF32ToGpuAdapter`] - For `.apr` files with F32 weights (direct copy)
//! - [`AprToGpuAdapter`] - For GGUF Q4_0 models (dequantizes to F32)
//!
//! # Coverage Impact
//!
//! Testing these adapters exercises:
//! - `apr_transformer/mod.rs` - F32 weight extraction
//! - `apr_transformer/q4_simd.rs` - Q4 weight extraction
//! - `gpu/scheduler/model.rs` - GpuModel creation
//! - `quantize/dequant.rs` - Q4_0 dequantization

use crate::apr_transformer::{
    AprTransformer, AprTransformerConfig, AprTransformerLayer,
    QuantizedAprTransformerQ4, QuantizedAprLayerQ4,
};
use crate::gpu::scheduler::{GpuModel, GpuModelConfig, BlockWeights};
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

/// Adapter for converting F32 APR models to GPU format
///
/// Used for `.apr` files which contain F32 weights. No dequantization needed.
pub struct AprF32ToGpuAdapter;

impl AprF32ToGpuAdapter {
    /// Convert F32 APR transformer to GpuModel
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer with F32 weights
    ///
    /// # Returns
    ///
    /// `GpuModel` ready for GPU inference
    ///
    /// # Example
    ///
    /// ```ignore
    /// use realizar::apr_transformer::AprTransformer;
    /// use realizar::gpu::adapters::AprF32ToGpuAdapter;
    ///
    /// let apr = AprTransformer::from_apr_bytes(&data)?;
    /// let gpu_model = AprF32ToGpuAdapter::to_gpu_model(&apr)?;
    /// ```
    pub fn to_gpu_model(apr: &AprTransformer) -> Result<GpuModel> {
        let config = AprToGpuAdapter::config_to_gpu(&apr.config);
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;

        // Embedding weights (already F32)
        let embedding_weights = apr.token_embedding.clone();

        // LM head weights (already F32)
        let lm_head_weight = apr.lm_head_weight.clone();

        // Phase 22 FIX: Transpose LM head from APR [vocab_size, hidden_dim] to GPU [hidden_dim, vocab_size]
        // APR stores weights as [out_dim, in_dim], GPU matmul expects [in_dim, out_dim]
        let lm_head_weight_t = transpose_matrix(&lm_head_weight, config.vocab_size, hidden_dim);

        // Convert each layer
        let mut block_weights = Vec::with_capacity(apr.layers.len());
        for layer in &apr.layers {
            block_weights.push(Self::convert_layer(
                layer,
                hidden_dim,
                intermediate_dim,
                config.num_heads,
                config.num_kv_heads,
            ));
        }

        // Final norm
        let final_norm_weight = apr.output_norm_weight.clone();
        let final_norm_bias = apr.output_norm_bias.clone().unwrap_or_else(|| vec![0.0; hidden_dim]);

        // LM head bias
        let lm_head_bias = apr.lm_head_bias.clone().unwrap_or_else(|| vec![0.0; config.vocab_size]);

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

    /// Convert a single F32 layer to BlockWeights
    fn convert_layer(
        layer: &AprTransformerLayer,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> BlockWeights {
        // Phase 21 FIX: APR stores weights as [out_dim, in_dim] row-major,
        // but GPU gemm expects [in_dim, out_dim]. Transpose all projection weights.
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        // Transpose QKV: [qkv_out_dim, hidden_dim] -> [hidden_dim, qkv_out_dim]
        let qkv_weight_t = transpose_matrix(&layer.qkv_weight, qkv_out_dim, hidden_dim);

        // Transpose output projection: [hidden_dim, hidden_dim] -> [hidden_dim, hidden_dim]
        let out_weight_t = transpose_matrix(&layer.attn_output_weight, hidden_dim, hidden_dim);

        // Transpose FFN up (fc1): [intermediate_dim, hidden_dim] -> [hidden_dim, intermediate_dim]
        let fc1_weight_t = transpose_matrix(&layer.ffn_up_weight, intermediate_dim, hidden_dim);

        // Transpose FFN down (fc2): [hidden_dim, intermediate_dim] -> [intermediate_dim, hidden_dim]
        let fc2_weight_t = transpose_matrix(&layer.ffn_down_weight, hidden_dim, intermediate_dim);

        // Transpose gate weight if present: [intermediate_dim, hidden_dim] -> [hidden_dim, intermediate_dim]
        let gate_weight_t = layer.ffn_gate_weight.as_ref().map(|w| {
            transpose_matrix(w, intermediate_dim, hidden_dim)
        });

        BlockWeights {
            attn_norm_weight: layer.attn_norm_weight.clone(),
            attn_norm_bias: layer.attn_norm_bias.clone().unwrap_or_else(|| vec![0.0; hidden_dim]),
            qkv_weight: qkv_weight_t,
            qkv_bias: layer.qkv_bias.clone().unwrap_or_default(),
            out_weight: out_weight_t,
            out_bias: layer.attn_output_bias.clone().unwrap_or_else(|| vec![0.0; hidden_dim]),
            // Use actual FFN norm if available, otherwise identity (Phase 21 fix)
            ffn_norm_weight: layer.ffn_norm_weight.clone().unwrap_or_else(|| vec![1.0; hidden_dim]),
            ffn_norm_bias: layer.ffn_norm_bias.clone().unwrap_or_else(|| vec![0.0; hidden_dim]),
            ffn_fc1_weight: fc1_weight_t,
            ffn_fc1_bias: layer.ffn_up_bias.clone().unwrap_or_else(|| vec![0.0; intermediate_dim]),
            ffn_fc2_weight: fc2_weight_t,
            ffn_fc2_bias: layer.ffn_down_bias.clone().unwrap_or_else(|| vec![0.0; hidden_dim]),
            // SwiGLU gate weight - critical for Qwen/LLaMA models
            ffn_gate_weight: gate_weight_t,
        }
    }
}

/// Adapter for converting Q4 APR models to GPU format
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
            rope_theta: apr_config.rope_theta,
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

        // Phase 22 FIX: Transpose LM head from APR [vocab_size, hidden_dim] to GPU [hidden_dim, vocab_size]
        // APR stores weights as [out_dim, in_dim], GPU matmul expects [in_dim, out_dim]
        let lm_head_weight_t = transpose_matrix(&lm_head_weight, config.vocab_size, hidden_dim);

        // Convert each layer
        let mut block_weights = Vec::with_capacity(apr.layers.len());
        for layer in &apr.layers {
            let qkv = Self::extract_qkv_weights(layer, hidden_dim, config.num_heads, config.num_kv_heads)?;
            let out = Self::extract_out_weights(layer, hidden_dim)?;
            let (fc1, fc2) = Self::extract_ffn_weights(layer, hidden_dim, intermediate_dim)?;

            // Extract gate weight for SwiGLU (optional)
            let ffn_gate_weight = if let Some(ref gate) = layer.ffn_gate_weight {
                let gate_expected = hidden_dim * intermediate_dim;
                Some(Self::dequantize_tensor(&gate.data, gate_expected)?)
            } else {
                None
            };

            block_weights.push(BlockWeights {
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
                ffn_gate_weight,
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
