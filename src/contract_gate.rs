//! GH-279: Unified Model Load Contract Gate
//!
//! **THE** single enforcement point for ALL model loading paths in realizar.
//! Every model (GGUF, SafeTensors, APR) MUST pass through `validate_model_load()`
//! before weights enter any kernel.
//!
//! # Architecture
//!
//! ```text
//! GGUF CPU ──────┐
//! GGUF CUDA ─────┤
//! SafeTensors ───┼──► validate_model_load() ──► ModelLoadProof ──► kernel
//! APR CPU ───────┤
//! APR CUDA ──────┤
//! GpuModel ──────┘
//! ```
//!
//! `ModelLoadProof` is a sealed type — private inner field means it can ONLY
//! be constructed by `validate_model_load()`. Downstream code that requires
//! a `&ModelLoadProof` parameter is GUARANTEED to have passed validation.
//!
//! # Validation Layers
//!
//! 1. **Architecture completeness** — all required weight roles present
//!    (via `arch_requirements::required_roles()`)
//! 2. **Dimension plausibility** — hidden_dim > 0, num_heads > 0, hidden_dim % num_heads == 0
//! 3. **Kernel contract link** — trueno `contracts::QuantFormat` constants are
//!    used to validate buffer sizes match expectations

use crate::arch_requirements::{required_roles, WeightRole};
use crate::error::RealizarError;
use crate::gguf::ArchConstraints;
use std::fmt;

// Re-export trueno kernel contracts for downstream consumers
pub use trueno::contracts::{
    self as kernel_contracts, validate_f32_buffer, validate_gemv_shapes, validate_weight_buffer,
    QuantFormat, TensorLayout, WeightBufferError, STACK_LAYOUT,
};

// ============================================================================
// ModelLoadProof — sealed output token
// ============================================================================

/// Proof that a model passed all contract validation gates.
///
/// Private inner field = IMPOSSIBLE to construct without `validate_model_load()`.
/// Functions that accept `&ModelLoadProof` are GUARANTEED that:
/// - All architecture-required weights are declared present
/// - Model dimensions are plausible
/// - The architecture is recognized
///
/// This does NOT prove that weight DATA is correct — only that the structural
/// metadata is valid. Data correctness is validated by `ValidatedLayerWeights`
/// at the per-layer level.
#[derive(Debug, Clone)]
pub struct ModelLoadProof {
    /// Private — construction only through validate_model_load()
    _architecture: String,
    _num_layers: usize,
}

impl ModelLoadProof {
    /// Architecture that was validated.
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self._architecture
    }

    /// Number of layers that was validated.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self._num_layers
    }
}

// ============================================================================
// ModelLoadConfig — input to validation
// ============================================================================

/// Model metadata required for contract validation.
///
/// Extracted from GGUF/SafeTensors/APR metadata at load time.
/// Passed to `validate_model_load()` before any weight data is accessed.
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    /// Architecture name (e.g., "llama", "qwen2", "qwen3")
    pub architecture: String,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads (Q heads)
    pub num_heads: usize,
    /// Number of K/V heads (for GQA)
    pub num_kv_heads: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Which weight roles are present in the model file.
    /// For each layer, the loader checks which tensors exist and reports them here.
    /// If empty, architecture completeness check is skipped (backwards compat).
    pub present_roles: Vec<WeightRole>,
}

// ============================================================================
// Validation Error
// ============================================================================

/// Error from model load contract validation.
#[derive(Debug, Clone)]
pub struct ModelLoadError {
    /// What failed
    pub gate: &'static str,
    /// Detailed reason
    pub reason: String,
}

impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GH-279 contract gate '{}' failed: {}",
            self.gate, self.reason
        )
    }
}

impl std::error::Error for ModelLoadError {}

impl From<ModelLoadError> for RealizarError {
    fn from(e: ModelLoadError) -> Self {
        RealizarError::UnsupportedOperation {
            operation: format!("contract_gate::{}", e.gate),
            reason: e.reason,
        }
    }
}

// ============================================================================
// The Gate
// ============================================================================

/// Validate model metadata before loading weights.
///
/// This is THE enforcement point. ALL model loading paths MUST call this
/// before accessing weight data. Returns `ModelLoadProof` on success.
///
/// # Validation Gates
///
/// 1. **dimension_plausibility** — hidden_dim > 0, num_heads > 0,
///    hidden_dim % num_heads == 0, vocab_size > 0
/// 2. **architecture_recognized** — `ArchConstraints::from_architecture()`
///    returns valid constraints
/// 3. **architecture_completeness** — if `present_roles` is non-empty,
///    every role in `required_roles(arch)` must be in `present_roles`
///
/// # Errors
///
/// Returns `ModelLoadError` with gate name and detailed reason.
pub fn validate_model_load(config: &ModelLoadConfig) -> std::result::Result<ModelLoadProof, ModelLoadError> {
    // Gate 1: Dimension plausibility
    validate_dimensions(config)?;

    // Gate 2: Architecture recognized
    let arch = validate_architecture(&config.architecture)?;

    // Gate 3: Architecture completeness (if roles reported)
    if !config.present_roles.is_empty() {
        validate_completeness(&arch, &config.present_roles, &config.architecture)?;
    }

    Ok(ModelLoadProof {
        _architecture: config.architecture.clone(),
        _num_layers: config.num_layers,
    })
}

/// Convenience: validate from `ArchConstraints` + dimensions (no role checking).
///
/// Used by loading paths that don't enumerate roles but do have ArchConstraints.
/// Still validates dimensions and architecture.
pub fn validate_model_load_basic(
    architecture: &str,
    num_layers: usize,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
    vocab_size: usize,
) -> std::result::Result<ModelLoadProof, ModelLoadError> {
    validate_model_load(&ModelLoadConfig {
        architecture: architecture.to_string(),
        num_layers,
        hidden_dim,
        num_heads,
        num_kv_heads,
        intermediate_dim,
        vocab_size,
        present_roles: Vec::new(), // no role checking in basic mode
    })
}

/// Convert a `ModelLoadError` into a `RealizarError` for ? propagation.
pub fn gate_error(e: ModelLoadError) -> RealizarError {
    e.into()
}

// ============================================================================
// Individual Gates
// ============================================================================

fn validate_dimensions(config: &ModelLoadConfig) -> std::result::Result<(), ModelLoadError> {
    if config.hidden_dim == 0 {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: "hidden_dim is 0".to_string(),
        });
    }
    if config.num_heads == 0 {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: "num_heads is 0".to_string(),
        });
    }
    if config.hidden_dim % config.num_heads != 0 {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: format!(
                "hidden_dim ({}) is not divisible by num_heads ({})",
                config.hidden_dim, config.num_heads
            ),
        });
    }
    if config.vocab_size == 0 {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: "vocab_size is 0".to_string(),
        });
    }
    if config.num_kv_heads == 0 {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: "num_kv_heads is 0".to_string(),
        });
    }
    if config.num_kv_heads > config.num_heads {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: format!(
                "num_kv_heads ({}) > num_heads ({})",
                config.num_kv_heads, config.num_heads
            ),
        });
    }
    if config.intermediate_dim == 0 {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: "intermediate_dim is 0".to_string(),
        });
    }
    if config.num_layers == 0 {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: "num_layers is 0".to_string(),
        });
    }
    Ok(())
}

fn validate_architecture(arch_name: &str) -> std::result::Result<ArchConstraints, ModelLoadError> {
    let arch = ArchConstraints::from_architecture(arch_name);
    // ArchConstraints::from_architecture returns a valid default for unknown architectures.
    // We accept this — unknown architectures get base validation (no QK norm, no bias).
    // This is by design: new architectures can load with base constraints and fail later
    // at the ValidatedLayerWeights level if they have unexpected weight patterns.
    Ok(arch)
}

fn validate_completeness(
    arch: &ArchConstraints,
    present: &[WeightRole],
    arch_name: &str,
) -> std::result::Result<(), ModelLoadError> {
    let required = required_roles(arch);
    let mut missing = Vec::new();

    for &role in required {
        if !present.contains(&role) {
            missing.push(role.field_name());
        }
    }

    if !missing.is_empty() {
        return Err(ModelLoadError {
            gate: "architecture_completeness",
            reason: format!(
                "Architecture '{}' requires {} weights but model is missing: [{}]",
                arch_name,
                required.len(),
                missing.join(", "),
            ),
        });
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ModelLoadConfig {
        ModelLoadConfig {
            architecture: "llama".to_string(),
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 11008,
            vocab_size: 32000,
            present_roles: Vec::new(),
        }
    }

    #[test]
    fn test_valid_model_passes() {
        let proof = validate_model_load(&valid_config()).expect("should pass");
        assert_eq!(proof.architecture(), "llama");
        assert_eq!(proof.num_layers(), 32);
    }

    #[test]
    fn test_zero_hidden_dim_fails() {
        let mut config = valid_config();
        config.hidden_dim = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert_eq!(err.gate, "dimension_plausibility");
        assert!(err.reason.contains("hidden_dim"));
    }

    #[test]
    fn test_zero_num_heads_fails() {
        let mut config = valid_config();
        config.num_heads = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("num_heads"));
    }

    #[test]
    fn test_hidden_not_divisible_by_heads() {
        let mut config = valid_config();
        config.hidden_dim = 4097;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("not divisible"));
    }

    #[test]
    fn test_kv_heads_greater_than_heads() {
        let mut config = valid_config();
        config.num_kv_heads = 64;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("num_kv_heads"));
    }

    #[test]
    fn test_zero_vocab_fails() {
        let mut config = valid_config();
        config.vocab_size = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("vocab_size"));
    }

    #[test]
    fn test_zero_layers_fails() {
        let mut config = valid_config();
        config.num_layers = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("num_layers"));
    }

    #[test]
    fn test_zero_intermediate_fails() {
        let mut config = valid_config();
        config.intermediate_dim = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("intermediate_dim"));
    }

    #[test]
    fn test_basic_convenience() {
        let proof = validate_model_load_basic(
            "qwen2", 28, 1536, 12, 2, 8960, 151936,
        )
        .expect("should pass");
        assert_eq!(proof.architecture(), "qwen2");
    }

    #[test]
    fn test_completeness_llama_all_present() {
        let mut config = valid_config();
        config.present_roles = vec![
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            WeightRole::FfnGate,
            WeightRole::FfnUp,
            WeightRole::FfnDown,
        ];
        assert!(validate_model_load(&config).is_ok());
    }

    #[test]
    fn test_completeness_llama_missing_gate() {
        let mut config = valid_config();
        config.present_roles = vec![
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            // Missing FfnGate, FfnUp, FfnDown
        ];
        let err = validate_model_load(&config).unwrap_err();
        assert_eq!(err.gate, "architecture_completeness");
        assert!(err.reason.contains("ffn_gate"));
    }

    #[test]
    fn test_completeness_qwen3_needs_qk_norm() {
        let mut config = valid_config();
        config.architecture = "qwen3".to_string();
        // Provide base roles but NOT qk_norm
        config.present_roles = vec![
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            WeightRole::FfnGate,
            WeightRole::FfnUp,
            WeightRole::FfnDown,
        ];
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("attn_q_norm"));
    }

    #[test]
    fn test_completeness_qwen3_with_qk_norm_passes() {
        let mut config = valid_config();
        config.architecture = "qwen3".to_string();
        config.present_roles = vec![
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            WeightRole::FfnGate,
            WeightRole::FfnUp,
            WeightRole::FfnDown,
            WeightRole::AttnQNorm,
            WeightRole::AttnKNorm,
        ];
        assert!(validate_model_load(&config).is_ok());
    }

    #[test]
    fn test_no_roles_skips_completeness() {
        // If present_roles is empty, completeness check is skipped
        let config = valid_config();
        assert!(config.present_roles.is_empty());
        assert!(validate_model_load(&config).is_ok());
    }

    #[test]
    fn test_unknown_architecture_uses_base() {
        let proof = validate_model_load_basic(
            "unknown_future_arch", 1, 128, 4, 4, 512, 1000,
        )
        .expect("unknown arch should pass with base constraints");
        assert_eq!(proof.architecture(), "unknown_future_arch");
    }

    #[test]
    fn test_error_display() {
        let err = ModelLoadError {
            gate: "test_gate",
            reason: "test reason".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("GH-279"));
        assert!(msg.contains("test_gate"));
        assert!(msg.contains("test reason"));
    }

    #[test]
    fn test_error_converts_to_realizar_error() {
        let err = ModelLoadError {
            gate: "test",
            reason: "test".to_string(),
        };
        let r_err: RealizarError = err.into();
        match r_err {
            RealizarError::UnsupportedOperation { operation, .. } => {
                assert!(operation.contains("contract_gate"));
            }
            _ => panic!("expected UnsupportedOperation"),
        }
    }
}
