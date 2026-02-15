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

include!("config_part_02.rs");
