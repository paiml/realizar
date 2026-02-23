//! GGUF configuration extraction
//!
//! Extracts model configuration from GGUF metadata.
//!
//! This module defines `GGUFConfig` which holds the transformer
//! architecture parameters needed for inference, and `ArchConstraints`
//! which encodes compile-time model family contract data from
//! `aprender/contracts/model-families/*.yaml`.

use super::types::GGUFModel;
use crate::error::{RealizarError, Result};

// ---------------------------------------------------------------------------
// ArchConstraints — contract-driven architecture behavior (GH-278)
// ---------------------------------------------------------------------------

/// Normalization type per model family contract.
///
/// Source: `constraints.norm_type` in `aprender/contracts/model-families/*.yaml`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Standard Layer Normalization with optional bias (GPT-2, phi, BERT, whisper)
    LayerNorm,
    /// Root Mean Square Normalization without bias (LLaMA, Qwen2, Mistral, etc.)
    RmsNorm,
}

/// Activation function per model family contract.
///
/// Source: `constraints.activation` in `aprender/contracts/model-families/*.yaml`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Gaussian Error Linear Unit (GPT-2, BERT, gemma, whisper)
    Gelu,
    /// Sigmoid Linear Unit (LLaMA, Qwen2, Mistral, phi, etc.)
    Silu,
}

/// Positional encoding type per model family contract.
///
/// Source: `constraints.positional_encoding` in `aprender/contracts/model-families/*.yaml`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncoding {
    /// Learned absolute position embeddings (GPT-2, BERT, whisper)
    Absolute,
    /// Rotary Position Embedding (LLaMA, Qwen2, Mistral, phi, etc.)
    Rope,
    /// No positional encoding (mamba, rwkv7)
    None,
}

/// FFN/MLP structure per model family contract.
///
/// Source: `constraints.mlp_type` in `aprender/contracts/model-families/*.yaml`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpType {
    /// Standard GELU MLP: up → GELU → down (GPT-2, BERT, whisper)
    GeluMlp,
    /// SwiGLU: gate ⊙ SiLU(up) → down (LLaMA, Qwen2, Mistral, phi, etc.)
    SwiGlu,
    /// Gated MLP: gate ⊙ GELU(up) → down (gemma, moonshine)
    GatedMlp,
}

/// Weight storage layout per model family contract.
///
/// Source: `constraints.mlp_type` + `shape_template` in contracts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// Standard Linear layout: `[out_features, in_features]` — no transpose needed
    Linear,
    /// Conv1D layout: `[in_features, out_features]` — requires transpose for `y = x @ W^T`
    Conv1D,
}

/// Architecture constraints derived from model family contracts.
///
/// These are compile-time constants per architecture, NOT runtime heuristics.
/// Source of truth: `aprender/contracts/model-families/*.yaml`
///
/// # Usage
///
/// ```ignore
/// let c = ArchConstraints::from_architecture("gpt2");
/// // c.norm_type == NormType::LayerNorm
/// // c.activation == Activation::Gelu
/// // c.positional_encoding == PositionalEncoding::Absolute
/// // c.mlp_type == MlpType::GeluMlp
/// // c.weight_layout == WeightLayout::Conv1D
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchConstraints {
    /// Normalization type (LayerNorm or RMSNorm)
    pub norm_type: NormType,
    /// Activation function (GELU or SiLU)
    pub activation: Activation,
    /// Positional encoding (Absolute, RoPE, or None)
    pub positional_encoding: PositionalEncoding,
    /// FFN structure (GeluMlp, SwiGlu, or GatedMlp)
    pub mlp_type: MlpType,
    /// Weight storage layout (Linear or Conv1D)
    pub weight_layout: WeightLayout,
    /// Whether the architecture has bias terms in attention/FFN layers
    pub has_bias: bool,
    /// Whether embedding and LM head weights are tied
    pub tied_embeddings: bool,
    /// Whether Q and K projections have per-head RMSNorm (GH-279: Qwen3)
    pub has_qk_norm: bool,
    /// Default norm epsilon when GGUF metadata is missing
    pub default_eps: f32,
}

// GH-323: Generated from arch-constraints-v1.yaml by build.rs.
// The include! pulls in from_architecture_generated() which does the actual match.
// Fallback: if build.rs can't find the YAML, it uses arch_constraints_fallback.rs.
include!(concat!(env!("OUT_DIR"), "/arch_constraints_generated.rs"));

impl ArchConstraints {
    /// Look up architecture constraints from the GGUF `general.architecture` value.
    ///
    /// Maps architecture names to their contract-defined behavior per
    /// `provable-contracts/contracts/arch-constraints-v1.yaml`.
    /// Unknown architectures fall back to LLaMA-like defaults.
    ///
    /// AUTO-GENERATED via build.rs from arch-constraints-v1.yaml.
    #[must_use]
    pub fn from_architecture(arch: &str) -> Self {
        from_architecture_generated(arch)
    }

    /// Whether this architecture uses RoPE positional encoding.
    #[must_use]
    pub fn uses_rope(&self) -> bool {
        self.positional_encoding == PositionalEncoding::Rope
    }

    /// Whether this architecture uses RMSNorm (vs LayerNorm).
    #[must_use]
    pub fn uses_rmsnorm(&self) -> bool {
        self.norm_type == NormType::RmsNorm
    }

    /// Whether weight matrices need Conv1D transpose.
    #[must_use]
    pub fn needs_transpose(&self) -> bool {
        self.weight_layout == WeightLayout::Conv1D
    }

    /// Whether this architecture uses a gated FFN (SwiGLU or GatedMLP).
    ///
    /// Gated FFN architectures require `ffn_gate_weight` to be present in model layers.
    /// Non-gated architectures (GeluMlp) use a simple up → activation → down path.
    #[must_use]
    pub fn has_gate_ffn(&self) -> bool {
        !matches!(self.mlp_type, MlpType::GeluMlp)
    }

    /// Whether this architecture uses learned absolute position embeddings.
    ///
    /// Architectures with absolute encoding (GPT-2, BERT, whisper) add learned
    /// position vectors to token embeddings. RoPE-based models skip this.
    #[must_use]
    pub fn uses_absolute_positions(&self) -> bool {
        self.positional_encoding == PositionalEncoding::Absolute
    }
}

/// Infer RoPE type from architecture string.
///
/// Returns 0 for NORM style (adjacent pairs), 2 for NEOX style (split halves).
/// Matches llama.cpp's rope type inference (llama-model.cpp:7763-7811).
///
/// GH-329: Single source of truth — all rope_type inference MUST go through here.
#[must_use]
pub fn infer_rope_type(arch: &str) -> u32 {
    let arch_lower = arch.to_lowercase();
    // NEOX style (type 2): pairs offset by n_rot/2
    // This list matches llama.cpp's llama-model.cpp:7763-7811
    const NEOX_ARCHITECTURES: &[&str] = &[
        "qwen", "qwen2", "qwen3", "qwen3_5", "qwen3.5",
        "stablelm", "phi2", "phi3", "phi",
        "gemma", "gemma2", "gemma3",
        "starcoder2", "gptneox", "gpt_neox",
        "falcon", "falcon_h1",
        "codeshell", "orion",
        "bert", "nomic-bert",
        "dbrx", "olmo2", "olmoe",
        "plamo", "plamo2",
        "openelm", "exaone",
        "minicpm3", "nemotron",
        "internlm2", "deepseek", "deepseek2",
    ];
    for &neox_arch in NEOX_ARCHITECTURES {
        if arch_lower.contains(neox_arch) {
            return 2; // NEOX style
        }
    }
    // NORM style (type 0): adjacent pairs - default for LLaMA, TinyLlama
    0
}

/// Configuration for GGUF transformer inference
#[derive(Debug, Clone)]
pub struct GGUFConfig {
    /// Model architecture (e.g., "phi2", "llama", "qwen2")
    pub architecture: String,
    /// Contract-derived architecture constraints (norm type, activation, etc.)
    pub constraints: ArchConstraints,
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
    /// Explicit per-head dimension for Q/K projections (from GGUF metadata).
    ///
    /// `None` means derive as `hidden_dim / num_heads` (correct for most models).
    /// `Some(128)` for Qwen3-0.6B where `hidden_dim=1024, num_heads=16` but `head_dim=128`
    /// meaning `q_dim = num_heads * head_dim = 2048 ≠ hidden_dim`.
    ///
    /// Source: GGUF metadata `{arch}.attention.key_length`.
    pub explicit_head_dim: Option<usize>,
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
    /// Per-head dimension for Q/K projections.
    ///
    /// Uses explicit value from GGUF metadata if available, otherwise `hidden_dim / num_heads`.
    #[inline]
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.explicit_head_dim.unwrap_or(if self.num_heads > 0 {
            self.hidden_dim / self.num_heads
        } else {
            self.hidden_dim
        })
    }

    /// Total Q projection dimension: `num_heads * head_dim`.
    ///
    /// For most models this equals `hidden_dim`, but Qwen3-0.6B has
    /// `q_dim = 16 * 128 = 2048` while `hidden_dim = 1024`.
    #[inline]
    #[must_use]
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim()
    }

    /// Total KV projection dimension: `num_kv_heads * head_dim`.
    #[inline]
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// GH-305: Infer explicit head_dim from GGUF metadata or tensor shapes.
    ///
    /// Returns `Some(head_dim)` only when it differs from `hidden_dim / num_heads`.
    fn infer_explicit_head_dim(
        model: &GGUFModel,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Option<usize> {
        let default_head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            hidden_dim
        };
        model
            .key_length()
            .or_else(|| {
                // Fallback: infer from blk.0.attn_q.weight shape: [q_dim, hidden_dim] → q_dim / num_heads
                model
                    .tensors
                    .iter()
                    .find(|t| t.name == "blk.0.attn_q.weight")
                    .and_then(|t| {
                        let d0 = t.dims.first().copied()? as usize;
                        if d0 > 0 && num_heads > 0 && d0.is_multiple_of(num_heads) {
                            Some(d0 / num_heads)
                        } else {
                            None
                        }
                    })
            })
            .filter(|&hd| hd != default_head_dim)
    }

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

        // GH-278: Infer intermediate_dim from ffn_down tensor shape (preferred)
        // or ffn_up as fallback.
        //
        // GH-306: ffn_down is more reliable because some architectures (Phi-3.5)
        // fuse gate+up into a single ffn_up tensor with 2x the intermediate dim.
        // ffn_down is always [intermediate_dim, hidden_dim] (Linear layout).
        let _has_ffn_gate = model.tensors.iter().any(|t| t.name == "blk.0.ffn_gate.weight");
        let intermediate_dim = model
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down.weight")
            .map_or_else(
                || {
                    // Fallback: infer from ffn_up
                    model.tensors.iter()
                        .find(|t| t.name == "blk.0.ffn_up.weight")
                        .map_or(hidden_dim * 4, |t| {
                            let d0 = t.dims.first().copied().unwrap_or(hidden_dim as u64 * 4) as usize;
                            let d1 = t.dims.get(1).copied().unwrap_or(hidden_dim as u64) as usize;
                            if d0 == hidden_dim && d1 != hidden_dim {
                                d1
                            } else {
                                d0
                            }
                        })
                },
                |t| {
                    // ffn_down: [intermediate_dim, hidden_dim] after dims.reverse()
                    let d0 = t.dims.first().copied().unwrap_or(hidden_dim as u64 * 4) as usize;
                    let d1 = t.dims.get(1).copied().unwrap_or(hidden_dim as u64) as usize;
                    if d1 == hidden_dim {
                        d0 // Linear layout: [intermediate_dim, hidden_dim]
                    } else if d0 == hidden_dim {
                        d1 // Conv1D layout: [hidden_dim, intermediate_dim]
                    } else {
                        d0 // Best guess
                    }
                },
            );

        let context_length = model.context_length().unwrap_or(2048);

        // Read rope_theta from metadata, or use default (10000.0 for LLaMA-style)
        // Qwen2 uses 1000000.0, which is read from qwen2.rope.freq_base
        let rope_theta = model.rope_freq_base().unwrap_or(10000.0);

        // GH-278: Look up contract constraints for this architecture.
        // This replaces ALL runtime heuristics (tensor presence checks, string matching)
        // with compile-time contract data from aprender/contracts/model-families/*.yaml.
        let constraints = ArchConstraints::from_architecture(&architecture);

        // Read norm epsilon from GGUF metadata, falling back to contract default.
        // The contract default is architecture-specific (e.g., 1e-5 for LLaMA, 1e-6 for Qwen2).
        let eps = model.rms_epsilon().unwrap_or(constraints.default_eps);

        // num_kv_heads (for GQA - e.g., Qwen uses fewer KV heads than Q heads)
        let num_kv_heads = model.num_kv_heads().unwrap_or(num_heads);

        // GH-305: Infer head_dim from GGUF metadata or tensor shapes.
        let explicit_head_dim = Self::infer_explicit_head_dim(model, hidden_dim, num_heads);

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
            constraints,
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
            explicit_head_dim,
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
        // GH-305: When head_dim is explicitly set (from GGUF metadata), hidden_dim may not
        // equal num_heads * head_dim (e.g., Qwen3-0.6B: hidden=1024, heads=16, head_dim=128).
        // Only enforce divisibility when head_dim is NOT explicitly overridden.
        if config.explicit_head_dim.is_none()
            && !config.hidden_dim.is_multiple_of(config.num_heads)
        {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim ({}) must be divisible by num_heads ({}) when head_dim is derived",
                    config.hidden_dim, config.num_heads
                ),
            });
        }
        if config.head_dim() == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
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

    /// Contract-derived architecture constraints
    #[must_use]
    pub fn constraints(&self) -> &ArchConstraints {
        &self.inner.constraints
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

    /// Per-head dimension for Q/K projections.
    ///
    /// From GGUF metadata `{arch}.attention.key_length`, or `hidden_dim / num_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.inner.head_dim()
    }

    /// Total Q projection dimension (`num_heads * head_dim`).
    ///
    /// May differ from `hidden_dim` (e.g., Qwen3-0.6B: q_dim=2048, hidden_dim=1024).
    #[must_use]
    pub fn q_dim(&self) -> usize {
        self.inner.q_dim()
    }

    /// Total KV dimension (`num_kv_heads * head_dim`).
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.inner.kv_dim()
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

include!("config_validated.rs");
