//! APR Transformer Format for WASM-compatible LLM inference
//!
//! This module provides a WASM-compatible transformer implementation that stores
//! all weights as F32, enabling fair comparison between APR and GGUF formats.
//!
//! ## Design Goals
//!
//! 1. **WASM Compatibility**: Pure F32 weights, no SIMD requirements
//! 2. **Fair Comparison**: Same inference algorithm as GGUFTransformer
//! 3. **Serialization**: APR format with model type `TransformerLM` (0x0050)
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::apr_transformer::AprTransformer;
//! use realizar::gguf::{GGUFModel, GGUFTransformer};
//!
//! // Load GGUF model
//! let gguf_data = std::fs::read("model.gguf")?;
//! let gguf_model = GGUFModel::from_bytes(&gguf_data)?;
//! let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, &gguf_data)?;
//!
//! // Convert to APR format
//! let apr_transformer = AprTransformer::from_gguf_transformer(&gguf_transformer);
//!
//! // Run inference (should match GGUF output)
//! let logits = apr_transformer.forward(&[1, 2, 3, 4])?;
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};

/// Configuration for APR Transformer models
///
/// Mirrors `GGUFConfig` for compatibility but is serializable to APR format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AprTransformerConfig {
    /// Model architecture name (e.g., "phi2", "llama", "qwen2")
    pub architecture: String,
    /// Embedding/hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Maximum context length
    pub context_length: usize,
    /// RoPE theta for position encoding
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
}

impl Default for AprTransformerConfig {
    fn default() -> Self {
        Self {
            architecture: "unknown".to_string(),
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 2048,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }
}

/// Weights for a single transformer layer (all F32)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTransformerLayer {
    /// Attention norm weight [hidden_dim]
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional) [hidden_dim]
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weight [hidden_dim, 3*hidden_dim]
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias (optional) [3*hidden_dim]
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection weight [hidden_dim, hidden_dim]
    pub attn_output_weight: Vec<f32>,
    /// Attention output projection bias (optional) [hidden_dim]
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN gate weight for SwiGLU (optional) [hidden_dim, intermediate_dim]
    pub ffn_gate_weight: Option<Vec<f32>>,
    /// FFN gate bias (optional) [intermediate_dim]
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection weight [hidden_dim, intermediate_dim]
    pub ffn_up_weight: Vec<f32>,
    /// FFN up projection bias (optional) [intermediate_dim]
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection weight [intermediate_dim, hidden_dim]
    pub ffn_down_weight: Vec<f32>,
    /// FFN down projection bias (optional) [hidden_dim]
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (optional) [hidden_dim]
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional) [hidden_dim]
    pub ffn_norm_bias: Option<Vec<f32>>,
}

impl AprTransformerLayer {
    /// Create an empty layer with given dimensions
    pub fn empty(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        }
    }

    /// Get total number of parameters in this layer
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.attn_norm_weight.len();
        count += self.attn_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.qkv_weight.len();
        count += self.qkv_bias.as_ref().map_or(0, Vec::len);
        count += self.attn_output_weight.len();
        count += self.attn_output_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_gate_weight.as_ref().map_or(0, Vec::len);
        count += self.ffn_gate_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_up_weight.len();
        count += self.ffn_up_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_down_weight.len();
        count += self.ffn_down_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_norm_weight.as_ref().map_or(0, Vec::len);
        count += self.ffn_norm_bias.as_ref().map_or(0, Vec::len);
        count
    }
}

/// APR Transformer model with F32 weights
///
/// WASM-compatible format for fair comparison with GGUF.
/// All weights are stored as F32 (dequantized from GGUF if converted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTransformer {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding weights [vocab_size * hidden_dim]
    pub token_embedding: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<AprTransformerLayer>,
    /// Output norm weight [hidden_dim]
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional) [hidden_dim]
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight [hidden_dim * vocab_size]
    pub lm_head_weight: Vec<f32>,
    /// LM head bias (optional) [vocab_size]
    pub lm_head_bias: Option<Vec<f32>>,
}

impl AprTransformer {
    /// Create a new APR transformer with the given configuration
    pub fn new(config: AprTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let intermediate_dim = config.intermediate_dim;

        let layers = (0..config.num_layers)
            .map(|_| AprTransformerLayer::empty(hidden_dim, intermediate_dim))
            .collect();

        Self {
            config,
            token_embedding: vec![0.0; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; hidden_dim * vocab_size],
            lm_head_bias: None,
        }
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.token_embedding.len();
        for layer in &self.layers {
            count += layer.num_parameters();
        }
        count += self.output_norm_weight.len();
        count += self.output_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.lm_head_weight.len();
        count += self.lm_head_bias.as_ref().map_or(0, Vec::len);
        count
    }

    /// Get memory size in bytes (F32 = 4 bytes per param)
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.num_parameters() * 4
    }

    /// Look up token embeddings
    #[must_use]
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                // Out of vocab - return zeros
                embeddings.extend(std::iter::repeat(0.0).take(hidden_dim));
            }
        }

        embeddings
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let seq_len = input.len() / hidden_dim;
        let mut output = Vec::with_capacity(input.len());

        for s in 0..seq_len {
            let start = s * hidden_dim;
            let slice = &input[start..start + hidden_dim];

            // Calculate mean
            let mean: f32 = slice.iter().sum::<f32>() / hidden_dim as f32;

            // Calculate variance
            let variance: f32 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;

            // Normalize
            let std_dev = (variance + eps).sqrt();
            for (i, &x) in slice.iter().enumerate() {
                let normalized = (x - mean) / std_dev;
                let scaled = normalized * weight[i];
                let shifted = if let Some(b) = bias {
                    scaled + b[i]
                } else {
                    scaled
                };
                output.push(shifted);
            }
        }

        output
    }

    /// Matrix multiplication: output[out_dim] = input[in_dim] * weight[in_dim, out_dim]
    #[allow(clippy::unused_self)]
    fn matmul(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let seq_len = input.len() / in_dim;
        let mut output = Vec::with_capacity(seq_len * out_dim);

        for s in 0..seq_len {
            let input_start = s * in_dim;
            let input_slice = &input[input_start..input_start + in_dim];

            for o in 0..out_dim {
                let mut sum = 0.0;
                for (i, &input_val) in input_slice.iter().enumerate() {
                    // Weight layout: [in_dim, out_dim] row-major
                    let weight_idx = i * out_dim + o;
                    if weight_idx < weight.len() {
                        sum += input_val * weight[weight_idx];
                    }
                }
                output.push(sum);
            }
        }

        output
    }

    /// Add bias in-place
    #[allow(clippy::unused_self)]
    fn add_bias(&self, data: &mut [f32], bias: &[f32]) {
        let dim = bias.len();
        for (i, val) in data.iter_mut().enumerate() {
            *val += bias[i % dim];
        }
    }

    /// GELU activation (tanh approximation)
    #[allow(clippy::unused_self)]
    fn gelu(&self, data: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const GELU_COEFF: f32 = 0.044_715;

        for x in data.iter_mut() {
            let x3 = *x * *x * *x;
            let inner = SQRT_2_OVER_PI * (*x + GELU_COEFF * x3);
            *x = 0.5 * *x * (1.0 + inner.tanh());
        }
    }

    /// Forward pass through the transformer
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers
        for layer in &self.layers {
            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection
            let qkv_dim = 3 * hidden_dim;
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // 2c. Simplified attention (matches GGUF implementation)
            let seq_len = token_ids.len();
            let mut attn_out = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                for h in 0..hidden_dim {
                    attn_out.push(qkv[qkv_start + h]); // Use Q for simplified version
                }
            }

            // 2d. Attention output projection
            let mut attn_output =
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim);
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN up projection
            let mut ffn_hidden =
                self.matmul(&hidden, &layer.ffn_up_weight, hidden_dim, intermediate_dim);
            if let Some(ref bias) = layer.ffn_up_bias {
                self.add_bias(&mut ffn_hidden, bias);
            }

            // GELU activation
            self.gelu(&mut ffn_hidden);

            // FFN down projection
            let mut ffn_output = self.matmul(
                &ffn_hidden,
                &layer.ffn_down_weight,
                intermediate_dim,
                hidden_dim,
            );
            if let Some(ref bias) = layer.ffn_down_bias {
                self.add_bias(&mut ffn_output, bias);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self.matmul(
            last_hidden,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Predict next token (greedy decoding)
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Token ID with highest probability
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;

        // Argmax
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;

        Ok(max_idx as u32)
    }
}

/// Convert from `GGUFTransformer` to APR format
///
/// This dequantizes all GGUF weights to F32 for WASM compatibility.
#[cfg(feature = "default")]
impl From<&crate::gguf::GGUFTransformer> for AprTransformer {
    fn from(gguf: &crate::gguf::GGUFTransformer) -> Self {
        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
        };

        let layers = gguf
            .layers
            .iter()
            .map(|l| AprTransformerLayer {
                attn_norm_weight: l.attn_norm_weight.clone(),
                attn_norm_bias: l.attn_norm_bias.clone(),
                qkv_weight: l.qkv_weight.clone(),
                qkv_bias: l.qkv_bias.clone(),
                attn_output_weight: l.attn_output_weight.clone(),
                attn_output_bias: l.attn_output_bias.clone(),
                ffn_gate_weight: l.ffn_gate_weight.clone(),
                ffn_gate_bias: l.ffn_gate_bias.clone(),
                ffn_up_weight: l.ffn_up_weight.clone(),
                ffn_up_bias: l.ffn_up_bias.clone(),
                ffn_down_weight: l.ffn_down_weight.clone(),
                ffn_down_bias: l.ffn_down_bias.clone(),
                ffn_norm_weight: l.ffn_norm_weight.clone(),
                ffn_norm_bias: l.ffn_norm_bias.clone(),
            })
            .collect();

        Self {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            output_norm_bias: gguf.output_norm_bias.clone(),
            lm_head_weight: gguf.lm_head_weight.clone(),
            lm_head_bias: gguf.lm_head_bias.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // Configuration Tests
    // ==========================================================================

    #[test]
    fn test_config_default() {
        let config = AprTransformerConfig::default();
        assert_eq!(config.architecture, "unknown");
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.vocab_size, 32000);
    }

    #[test]
    fn test_config_serialization() {
        let config = AprTransformerConfig {
            architecture: "test_arch".to_string(),
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 1024,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-6,
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let decoded: AprTransformerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, decoded);
    }

    // ==========================================================================
    // Layer Tests
    // ==========================================================================

    #[test]
    fn test_layer_empty() {
        let layer = AprTransformerLayer::empty(64, 256);
        assert_eq!(layer.attn_norm_weight.len(), 64);
        assert_eq!(layer.qkv_weight.len(), 64 * 3 * 64);
        assert_eq!(layer.ffn_up_weight.len(), 64 * 256);
        assert_eq!(layer.ffn_down_weight.len(), 256 * 64);
    }

    #[test]
    fn test_layer_num_parameters() {
        let layer = AprTransformerLayer::empty(64, 256);
        let expected = 64 // attn_norm
            + 64 * 3 * 64 // qkv
            + 64 * 64 // attn_output
            + 64 * 256 // ffn_up
            + 256 * 64; // ffn_down
        assert_eq!(layer.num_parameters(), expected);
    }

    // ==========================================================================
    // Transformer Tests
    // ==========================================================================

    #[test]
    fn test_transformer_new() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        assert_eq!(transformer.layers.len(), 2);
        assert_eq!(transformer.token_embedding.len(), 100 * 64);
        assert_eq!(transformer.output_norm_weight.len(), 64);
        assert_eq!(transformer.lm_head_weight.len(), 64 * 100);
    }

    #[test]
    fn test_transformer_num_parameters() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Should be > 0 and reasonable
        let params = transformer.num_parameters();
        assert!(params > 0);
        assert!(params < 100_000_000); // Less than 100M params for test model
    }

    #[test]
    fn test_transformer_memory_size() {
        let config = AprTransformerConfig {
            hidden_dim: 64,
            num_layers: 1,
            vocab_size: 100,
            intermediate_dim: 128,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let params = transformer.num_parameters();
        let mem = transformer.memory_size();
        assert_eq!(mem, params * 4); // F32 = 4 bytes
    }

    // ==========================================================================
    // Embedding Tests
    // ==========================================================================

    #[test]
    fn test_embed_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            vocab_size: 10,
            ..Default::default()
        };
        let mut transformer = AprTransformer::new(config);

        // Set known embedding for token 3
        transformer.token_embedding[3 * 4..3 * 4 + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let embedded = transformer.embed(&[3]);
        assert_eq!(embedded, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_embed_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            vocab_size: 5,
            ..Default::default()
        };
        let mut transformer = AprTransformer::new(config);

        // Set embeddings
        transformer.token_embedding[0..2].copy_from_slice(&[1.0, 2.0]); // token 0
        transformer.token_embedding[2..4].copy_from_slice(&[3.0, 4.0]); // token 1
        transformer.token_embedding[4..6].copy_from_slice(&[5.0, 6.0]); // token 2

        let embedded = transformer.embed(&[0, 1, 2]);
        assert_eq!(embedded, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_embed_out_of_vocab() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            vocab_size: 5,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // Token 100 is out of vocab (vocab_size=5)
        let embedded = transformer.embed(&[100]);
        assert_eq!(embedded, vec![0.0, 0.0]); // Returns zeros
    }

    // ==========================================================================
    // Layer Norm Tests
    // ==========================================================================

    #[test]
    fn test_layer_norm_identity() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0]; // Identity weight

        let output = transformer.layer_norm(&input, &weight, None, 1e-5);

        // Normalized values should have mean ~0 and var ~1
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!((mean).abs() < 0.001);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 3.0]; // mean=2, var=1
        let weight = vec![1.0, 1.0];
        let bias = vec![10.0, 20.0];

        let output = transformer.layer_norm(&input, &weight, Some(&bias), 1e-5);

        // After norm: [-1, 1], after scale: [-1, 1], after bias: [9, 21]
        assert!((output[0] - 9.0).abs() < 0.01);
        assert!((output[1] - 21.0).abs() < 0.01);
    }

    // ==========================================================================
    // GELU Tests
    // ==========================================================================

    #[test]
    fn test_gelu_zero() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![0.0];
        transformer.gelu(&mut data);
        assert!((data[0]).abs() < 0.0001);
    }

    #[test]
    fn test_gelu_positive() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![1.0];
        transformer.gelu(&mut data);
        // GELU(1) ≈ 0.841
        assert!((data[0] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_gelu_negative() {
        let config = AprTransformerConfig::default();
        let transformer = AprTransformer::new(config);

        let mut data = vec![-1.0];
        transformer.gelu(&mut data);
        // GELU(-1) ≈ -0.159
        assert!((data[0] - (-0.159)).abs() < 0.01);
    }

    // ==========================================================================
    // Matmul Tests
    // ==========================================================================

    #[test]
    fn test_matmul_identity() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let input = vec![1.0, 2.0];
        // Identity matrix [2, 2] in row-major
        let weight = vec![1.0, 0.0, 0.0, 1.0];

        let output = transformer.matmul(&input, &weight, 2, 2);
        assert_eq!(output, vec![1.0, 2.0]);
    }

    #[test]
    fn test_matmul_simple() {
        let config = AprTransformerConfig {
            hidden_dim: 2,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        // input: [1, 2]
        // weight: [[1, 2, 3], [4, 5, 6]] (2x3 row-major)
        // output: [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let output = transformer.matmul(&input, &weight, 2, 3);
        assert_eq!(output, vec![9.0, 12.0, 15.0]);
    }

    // ==========================================================================
    // Forward Tests
    // ==========================================================================

    #[test]
    fn test_forward_empty_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_single_token() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[1]);
        assert!(result.is_ok());

        let logits = result.expect("forward succeeded");
        assert_eq!(logits.len(), 10); // vocab_size
    }

    #[test]
    fn test_forward_multiple_tokens() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.forward(&[1, 2, 3]);
        assert!(result.is_ok());

        let logits = result.expect("forward succeeded");
        assert_eq!(logits.len(), 10); // vocab_size (only last token logits)
    }

    // ==========================================================================
    // Predict Tests
    // ==========================================================================

    #[test]
    fn test_predict_next() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let result = transformer.predict_next(&[1]);
        assert!(result.is_ok());

        let token = result.expect("predict succeeded");
        assert!(token < 10); // Within vocab
    }

    // ==========================================================================
    // Reproducibility Tests
    // ==========================================================================

    #[test]
    fn test_reproducibility_same_input_same_output() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let tokens = vec![1, 2, 3];
        let output1 = transformer.forward(&tokens).expect("forward 1");
        let output2 = transformer.forward(&tokens).expect("forward 2");

        assert_eq!(output1, output2, "Same input should produce same output");
    }

    #[test]
    fn test_reproducibility_predict_deterministic() {
        let config = AprTransformerConfig {
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            ..Default::default()
        };
        let transformer = AprTransformer::new(config);

        let tokens = vec![1, 2, 3];
        let pred1 = transformer.predict_next(&tokens).expect("predict 1");
        let pred2 = transformer.predict_next(&tokens).expect("predict 2");

        assert_eq!(pred1, pred2, "Predictions should be deterministic");
    }

    // ==========================================================================
    // Serialization Tests
    // ==========================================================================

    #[test]
    fn test_transformer_serialization_roundtrip() {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 8,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
        };
        let transformer = AprTransformer::new(config);

        let json = serde_json::to_string(&transformer).expect("serialize");
        let decoded: AprTransformer = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(transformer.config, decoded.config);
        assert_eq!(transformer.token_embedding, decoded.token_embedding);
        assert_eq!(transformer.layers.len(), decoded.layers.len());
    }
}
