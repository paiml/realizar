//! GGUF configuration extraction
//!
//! Extracts model configuration from GGUF metadata.
//!
//! This module defines `GGUFConfig` which holds the transformer
//! architecture parameters needed for inference.

use super::types::GGUFModel;
use crate::error::{RealizarError, Result};

/// Configuration for GGUF transformer inference
#[derive(Debug, Clone)]
pub struct GGUFConfig {
    /// Model architecture (e.g., "phi2", "llama", "qwen2")
    pub architecture: String,
    /// Embedding dimension (hidden size)
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA, often num_heads or num_heads/8)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Context length
    pub context_length: usize,
    /// RoPE theta (position encoding base)
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
    /// RoPE type: 0 = NORM (adjacent pairs), 2 = NEOX (split halves)
    pub rope_type: u32,
}

impl GGUFConfig {
    /// Extract configuration from GGUF model metadata
    ///
    /// # Errors
    ///
    /// Returns an error if required metadata fields are missing from the GGUF model.
    pub fn from_gguf(model: &GGUFModel) -> Result<Self> {
        let architecture = model
            .architecture()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing general.architecture in GGUF metadata".to_string(),
            })?
            .to_string();

        let hidden_dim = model
            .embedding_dim()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing embedding_length in GGUF metadata".to_string(),
            })?;

        let num_layers = model
            .num_layers()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing block_count in GGUF metadata".to_string(),
            })?;

        // Try to get num_heads, default based on hidden_dim if not found
        let num_heads = model.num_heads().unwrap_or(hidden_dim / 64);

        // Get vocab_size from token_embd tensor
        // After dims.reverse(), shape is [vocab_size, hidden_dim] - vocab is at index 0
        let vocab_size = model
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map_or(32000, |t| t.dims.first().copied().unwrap_or(32000) as usize);

        // Infer intermediate_dim from ffn_up tensor
        // After dims.reverse(), shape is [intermediate_dim, hidden_dim] - intermediate is at index 0
        let intermediate_dim = model
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight")
            .map_or(hidden_dim * 4, |t| {
                t.dims.first().copied().unwrap_or(hidden_dim as u64 * 4) as usize
            });

        let context_length = model.context_length().unwrap_or(2048);

        // Read rope_theta from metadata, or use default (10000.0 for LLaMA-style)
        // Qwen2 uses 1000000.0, which is read from qwen2.rope.freq_base
        let rope_theta = model.rope_freq_base().unwrap_or(10000.0);

        // Read RMSNorm epsilon from metadata, or use default (1e-5 for LLaMA-style)
        // Qwen2 uses 1e-6, which is read from qwen2.attention.layer_norm_rms_epsilon
        let eps = model.rms_epsilon().unwrap_or(1e-5);

        // num_kv_heads (for GQA - e.g., Qwen uses fewer KV heads than Q heads)
        let num_kv_heads = model.num_kv_heads().unwrap_or(num_heads);

        // Read rope_type: 0 = NORM (adjacent pairs, default for LLaMA), 2 = NEOX (split halves)
        // LLaMA models use type 0 (adjacent pairs) per llama.cpp's LLAMA_ROPE_TYPE_NORM
        let rope_type = model.rope_type().unwrap_or(0);

        Ok(Self {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
            rope_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_config_creation() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        assert_eq!(config.architecture, "llama");
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.intermediate_dim, 11008);
        assert_eq!(config.context_length, 4096);
        assert!((config.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!((config.eps - 1e-5).abs() < f32::EPSILON);
        assert_eq!(config.rope_type, 0);
    }

    #[test]
    fn test_gguf_config_clone() {
        let config = GGUFConfig {
            architecture: "qwen2".to_string(),
            hidden_dim: 2048,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 5632,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2,
        };

        let cloned = config.clone();
        assert_eq!(cloned.architecture, "qwen2");
        assert_eq!(cloned.hidden_dim, config.hidden_dim);
        assert_eq!(cloned.rope_type, 2);
    }

    #[test]
    fn test_gguf_config_debug() {
        let config = GGUFConfig {
            architecture: "phi2".to_string(),
            hidden_dim: 2560,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            vocab_size: 51200,
            intermediate_dim: 10240,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("phi2"));
        assert!(debug_str.contains("2560"));
    }
}
