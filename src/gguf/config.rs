//! GGUF configuration extraction
//!
//! Extracts model configuration from GGUF metadata.
//!
//! This module defines `GGUFConfig` which holds the transformer
//! architecture parameters needed for inference.

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
