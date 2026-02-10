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
    /// BOS token ID from GGUF metadata (used for GPU validation probe)
    /// None means BOS is unknown — GPU validation will be skipped.
    pub bos_token_id: Option<u32>,
}

/// Architecture-default BOS token IDs for weights-only GGUFs.
///
/// Weights-only GGUF files (e.g., from pacha) contain only 4 metadata keys
/// and lack `tokenizer.ggml.bos_token_id`. This function provides a known-good
/// BOS token ID based on the architecture, enabling GPU validation (F2-VALIDATION)
/// that would otherwise be skipped.
///
/// Sources: `contracts/model-families/*.yaml`
fn default_bos_for_architecture(arch: &str) -> Option<u32> {
    match arch {
        "qwen2" => Some(151_643),
        "llama" => Some(128_000),
        "mistral" => Some(1),
        "gemma" | "gemma2" => Some(2),
        "deepseek" | "deepseek2" => Some(0),
        // phi/phi3: no BOS token (empty string in spec)
        _ => None,
    }
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

        // BOS token ID from GGUF metadata, with architecture-based fallback.
        // Weights-only GGUFs (e.g., from pacha) lack tokenizer.ggml.bos_token_id.
        // Without a BOS token, GPU validation (F2-VALIDATION) is skipped entirely.
        let bos_token_id = model.bos_token_id().or_else(|| {
            let fallback = default_bos_for_architecture(&architecture);
            if fallback.is_some() {
                eprintln!(
                    "[BOS-FALLBACK] No tokenizer.ggml.bos_token_id in GGUF — using architecture default for '{architecture}'"
                );
            }
            fallback
        });

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
            bos_token_id,
        })
    }
}

// ---------------------------------------------------------------------------
// ValidatedModelConfig — newtype Poka-Yoke wrapper (PMAT-235)
// ---------------------------------------------------------------------------

/// A validated model configuration that guarantees structural invariants.
///
/// Wraps `GGUFConfig` and enforces:
/// - `hidden_dim > 0`, `num_layers > 0`, `vocab_size > 0`
/// - `num_heads > 0`, `num_kv_heads > 0`
/// - `hidden_dim % num_heads == 0` (head_dim must divide evenly)
/// - `num_heads % num_kv_heads == 0` (GQA ratio must be an integer)
///
/// The inner `GGUFConfig` is private — access fields via getters or `Deref`.
#[derive(Debug, Clone)]
pub struct ValidatedModelConfig {
    inner: GGUFConfig,
}

impl ValidatedModelConfig {
    /// Validate a `GGUFConfig` and return a `ValidatedModelConfig`.
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidShape` if any structural invariant is violated.
    pub fn validate(config: GGUFConfig) -> Result<Self> {
        if config.hidden_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "hidden_dim must be > 0".to_string(),
            });
        }
        if config.num_layers == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_layers must be > 0".to_string(),
            });
        }
        if config.vocab_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "vocab_size must be > 0".to_string(),
            });
        }
        if config.num_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_heads must be > 0".to_string(),
            });
        }
        if config.num_kv_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_kv_heads must be > 0".to_string(),
            });
        }
        if !config.hidden_dim.is_multiple_of(config.num_heads) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim ({}) must be divisible by num_heads ({}) for head_dim consistency",
                    config.hidden_dim, config.num_heads
                ),
            });
        }
        if !config.num_heads.is_multiple_of(config.num_kv_heads) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "num_heads ({}) must be divisible by num_kv_heads ({}) — GQA ratio must be an integer",
                    config.num_heads, config.num_kv_heads
                ),
            });
        }

        Ok(Self { inner: config })
    }

    /// Load and validate directly from a GGUF model.
    ///
    /// Calls `GGUFConfig::from_gguf()` then validates the result.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata extraction or validation fails.
    pub fn from_gguf(model: &GGUFModel) -> Result<Self> {
        let config = GGUFConfig::from_gguf(model)?;
        Self::validate(config)
    }

    // -- Getters for all fields --

    /// Model architecture (e.g., "llama", "qwen2")
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.inner.architecture
    }

    /// Embedding dimension (hidden size)
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.inner.hidden_dim
    }

    /// Number of transformer layers
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.inner.num_layers
    }

    /// Number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.inner.num_heads
    }

    /// Number of key-value heads (for GQA)
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.inner.num_kv_heads
    }

    /// Vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size
    }

    /// FFN intermediate dimension
    #[must_use]
    pub fn intermediate_dim(&self) -> usize {
        self.inner.intermediate_dim
    }

    /// Context length
    #[must_use]
    pub fn context_length(&self) -> usize {
        self.inner.context_length
    }

    /// RoPE theta (position encoding base)
    #[must_use]
    pub fn rope_theta(&self) -> f32 {
        self.inner.rope_theta
    }

    /// Layer norm epsilon
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.inner.eps
    }

    /// RoPE type: 0 = NORM (adjacent pairs), 2 = NEOX (split halves)
    #[must_use]
    pub fn rope_type(&self) -> u32 {
        self.inner.rope_type
    }

    /// BOS token ID (None if unknown)
    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id
    }

    // -- Derived getters --

    /// Dimension per attention head (`hidden_dim / num_heads`).
    ///
    /// Safe because validation guarantees `hidden_dim % num_heads == 0`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.inner.hidden_dim / self.inner.num_heads
    }

    /// Total KV dimension (`num_kv_heads * head_dim`).
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.inner.num_kv_heads * self.head_dim()
    }

    /// Borrow the inner `GGUFConfig` (backward compatibility escape hatch).
    #[must_use]
    pub fn config(&self) -> &GGUFConfig {
        &self.inner
    }
}

impl std::ops::Deref for ValidatedModelConfig {
    type Target = GGUFConfig;

    fn deref(&self) -> &GGUFConfig {
        &self.inner
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
            bos_token_id: None,
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
            bos_token_id: None,
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
            bos_token_id: None,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("phi2"));
        assert!(debug_str.contains("2560"));
    }

    #[test]
    fn test_gguf_config_gqa_ratio() {
        // Test GQA config (4 Q heads per KV head)
        let config = GGUFConfig {
            architecture: "llama3".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // 4:1 GQA ratio
            vocab_size: 128256,
            intermediate_dim: 14336,
            context_length: 8192,
            rope_theta: 500000.0,
            eps: 1e-5,
            rope_type: 0,
            bos_token_id: None,
        };

        assert_eq!(config.num_heads / config.num_kv_heads, 4);
    }

    #[test]
    fn test_gguf_config_mha() {
        // Test MHA config (num_heads == num_kv_heads)
        let config = GGUFConfig {
            architecture: "phi".to_string(),
            hidden_dim: 2560,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32, // MHA
            vocab_size: 51200,
            intermediate_dim: 10240,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            bos_token_id: None,
        };

        assert_eq!(config.num_heads, config.num_kv_heads);
    }

    #[test]
    fn test_gguf_config_neox_rope() {
        // Test NEOX-style RoPE (type 2)
        let config = GGUFConfig {
            architecture: "gpt-neox".to_string(),
            hidden_dim: 6144,
            num_layers: 44,
            num_heads: 64,
            num_kv_heads: 64,
            vocab_size: 50432,
            intermediate_dim: 24576,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 2, // NEOX
            bos_token_id: None,
        };

        assert_eq!(config.rope_type, 2);
    }

    #[test]
    fn test_gguf_config_high_rope_theta() {
        // Test Qwen-style high rope_theta
        let config = GGUFConfig {
            architecture: "qwen".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 8960,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 0,
            bos_token_id: None,
        };

        assert!(config.rope_theta > 100_000.0);
        assert!(config.context_length > 8192);
    }

    #[test]
    fn test_gguf_config_small_epsilon() {
        // Test different epsilon values
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 50257,
            intermediate_dim: 2048,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-6, // Smaller epsilon
            rope_type: 0,
            bos_token_id: None,
        };

        assert!((config.eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_gguf_config_head_dim() {
        // Verify head_dim calculation
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
            bos_token_id: None,
        };

        let head_dim = config.hidden_dim / config.num_heads;
        assert_eq!(head_dim, 128);
    }

    #[test]
    fn test_gguf_config_kv_dim() {
        // Verify kv_dim calculation for GQA
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
            bos_token_id: None,
        };

        let head_dim = config.hidden_dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;
        assert_eq!(kv_dim, 1024);
    }

    #[test]
    fn test_gguf_config_qkv_out_dim() {
        // Verify QKV output dimension calculation
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
            bos_token_id: None,
        };

        let head_dim = config.hidden_dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;
        let qkv_out_dim = config.hidden_dim + 2 * kv_dim;
        assert_eq!(qkv_out_dim, 4096 + 2 * 1024);
    }

    #[test]
    fn test_gguf_config_debug_contains_all_fields() {
        let config = GGUFConfig {
            architecture: "mistral".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 14336,
            context_length: 8192,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            bos_token_id: None,
        };

        let debug = format!("{:?}", config);
        assert!(debug.contains("architecture"));
        assert!(debug.contains("hidden_dim"));
        assert!(debug.contains("num_layers"));
        assert!(debug.contains("num_heads"));
        assert!(debug.contains("num_kv_heads"));
        assert!(debug.contains("vocab_size"));
        assert!(debug.contains("intermediate_dim"));
        assert!(debug.contains("context_length"));
        assert!(debug.contains("rope_theta"));
        assert!(debug.contains("eps"));
        assert!(debug.contains("rope_type"));
    }

    #[test]
    fn test_gguf_config_clone_deep() {
        let original = GGUFConfig {
            architecture: "test_arch".to_string(),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            bos_token_id: None,
        };

        let cloned = original.clone();

        // Verify all fields are cloned correctly
        assert_eq!(cloned.architecture, original.architecture);
        assert_eq!(cloned.hidden_dim, original.hidden_dim);
        assert_eq!(cloned.num_layers, original.num_layers);
        assert_eq!(cloned.num_heads, original.num_heads);
        assert_eq!(cloned.num_kv_heads, original.num_kv_heads);
        assert_eq!(cloned.vocab_size, original.vocab_size);
        assert_eq!(cloned.intermediate_dim, original.intermediate_dim);
        assert_eq!(cloned.context_length, original.context_length);
        assert!((cloned.rope_theta - original.rope_theta).abs() < f32::EPSILON);
        assert!((cloned.eps - original.eps).abs() < f32::EPSILON);
        assert_eq!(cloned.rope_type, original.rope_type);
    }

    #[test]
    fn test_default_bos_qwen2() {
        assert_eq!(default_bos_for_architecture("qwen2"), Some(151_643));
    }

    #[test]
    fn test_default_bos_llama() {
        assert_eq!(default_bos_for_architecture("llama"), Some(128_000));
    }

    #[test]
    fn test_default_bos_mistral() {
        assert_eq!(default_bos_for_architecture("mistral"), Some(1));
    }

    #[test]
    fn test_default_bos_gemma() {
        assert_eq!(default_bos_for_architecture("gemma"), Some(2));
        assert_eq!(default_bos_for_architecture("gemma2"), Some(2));
    }

    #[test]
    fn test_default_bos_deepseek() {
        assert_eq!(default_bos_for_architecture("deepseek"), Some(0));
        assert_eq!(default_bos_for_architecture("deepseek2"), Some(0));
    }

    #[test]
    fn test_default_bos_unknown_returns_none() {
        assert_eq!(default_bos_for_architecture("phi"), None);
        assert_eq!(default_bos_for_architecture("phi3"), None);
        assert_eq!(default_bos_for_architecture("unknown_arch"), None);
        assert_eq!(default_bos_for_architecture(""), None);
    }

    // -----------------------------------------------------------------------
    // ValidatedModelConfig tests
    // -----------------------------------------------------------------------

    /// Helper: build a valid LLaMA-style GGUFConfig for testing.
    fn valid_llama_config() -> GGUFConfig {
        GGUFConfig {
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
            bos_token_id: Some(128_000),
        }
    }

    #[test]
    fn test_validated_config_accepts_valid() {
        let validated = ValidatedModelConfig::validate(valid_llama_config());
        assert!(validated.is_ok());
    }

    #[test]
    fn test_validated_config_getters() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");

        assert_eq!(v.architecture(), "llama");
        assert_eq!(v.hidden_dim(), 4096);
        assert_eq!(v.num_layers(), 32);
        assert_eq!(v.num_heads(), 32);
        assert_eq!(v.num_kv_heads(), 8);
        assert_eq!(v.vocab_size(), 32000);
        assert_eq!(v.intermediate_dim(), 11008);
        assert_eq!(v.context_length(), 4096);
        assert!((v.rope_theta() - 10000.0).abs() < f32::EPSILON);
        assert!((v.eps() - 1e-5).abs() < f32::EPSILON);
        assert_eq!(v.rope_type(), 0);
        assert_eq!(v.bos_token_id(), Some(128_000));
    }

    #[test]
    fn test_validated_config_head_dim() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        // 4096 / 32 = 128
        assert_eq!(v.head_dim(), 128);
    }

    #[test]
    fn test_validated_config_kv_dim() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        // 8 * 128 = 1024
        assert_eq!(v.kv_dim(), 1024);
    }

    #[test]
    fn test_validated_config_deref() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        // Access via Deref — should reach GGUFConfig fields directly
        assert_eq!(v.hidden_dim, 4096);
        assert_eq!(v.num_heads, 32);
    }

    #[test]
    fn test_validated_config_inner_ref() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        let inner: &GGUFConfig = v.config();
        assert_eq!(inner.architecture, "llama");
        assert_eq!(inner.hidden_dim, 4096);
    }

    #[test]
    fn test_validated_config_clone() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        let cloned = v.clone();
        assert_eq!(cloned.hidden_dim(), v.hidden_dim());
        assert_eq!(cloned.architecture(), v.architecture());
    }

    #[test]
    fn test_validated_config_debug() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        let debug = format!("{v:?}");
        assert!(debug.contains("ValidatedModelConfig"));
        assert!(debug.contains("llama"));
    }

    // -- Validation failure tests --

    #[test]
    fn test_validated_config_rejects_zero_hidden_dim() {
        let mut cfg = valid_llama_config();
        cfg.hidden_dim = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("hidden_dim"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_layers() {
        let mut cfg = valid_llama_config();
        cfg.num_layers = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_layers"));
    }

    #[test]
    fn test_validated_config_rejects_zero_vocab_size() {
        let mut cfg = valid_llama_config();
        cfg.vocab_size = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("vocab_size"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_heads() {
        let mut cfg = valid_llama_config();
        cfg.num_heads = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_heads"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_kv_heads() {
        let mut cfg = valid_llama_config();
        cfg.num_kv_heads = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_kv_heads"));
    }

    #[test]
    fn test_validated_config_rejects_bad_head_dim() {
        let mut cfg = valid_llama_config();
        // 4096 % 33 != 0
        cfg.num_heads = 33;
        cfg.num_kv_heads = 33;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("divisible by num_heads"));
    }

    #[test]
    fn test_validated_config_rejects_bad_gqa_ratio() {
        let mut cfg = valid_llama_config();
        // 32 % 5 != 0
        cfg.num_kv_heads = 5;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("GQA ratio"));
    }

    #[test]
    fn test_validated_config_mha() {
        // MHA: num_heads == num_kv_heads
        let mut cfg = valid_llama_config();
        cfg.num_kv_heads = 32;
        let v = ValidatedModelConfig::validate(cfg).expect("MHA should be valid");
        assert_eq!(v.num_heads(), v.num_kv_heads());
        assert_eq!(v.head_dim(), 128);
        assert_eq!(v.kv_dim(), 4096); // 32 * 128
    }

    #[test]
    fn test_validated_config_qwen_style() {
        // Qwen 1.5B: hidden=1536, heads=12, kv_heads=2
        let cfg = GGUFConfig {
            architecture: "qwen2".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 8960,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2,
            bos_token_id: Some(151_643),
        };
        let v = ValidatedModelConfig::validate(cfg).expect("Qwen config should be valid");
        assert_eq!(v.head_dim(), 128); // 1536 / 12
        assert_eq!(v.kv_dim(), 256); // 2 * 128
    }
}
