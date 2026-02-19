//! Per-architecture required weight roles.
//!
//! Generated from architecture-requirements-v1.yaml — DO NOT EDIT.
//! See: provable-contracts/contracts/architecture-requirements-v1.yaml
//!
//! UCBD §4 / GH-279: Compile-time enforcement that every loader
//! provides all tensors required by the target architecture.

use crate::gguf::ArchConstraints;

/// Weight roles that may be required for a transformer layer.
/// Each architecture requires a subset of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightRole {
    // Norms (all architectures)
    /// Pre-attention layer normalization gamma
    AttnNorm,
    /// Pre-FFN layer normalization gamma
    FfnNorm,

    // QK norm (Qwen3 family)
    /// Per-head Q RMSNorm gamma (Qwen3)
    AttnQNorm,
    /// Per-head K RMSNorm gamma (Qwen3)
    AttnKNorm,

    // Biases (Qwen2)
    /// Q projection bias (Qwen2, phi)
    AttnQBias,
    /// K projection bias (Qwen2, phi)
    AttnKBias,
    /// V projection bias (Qwen2, phi)
    AttnVBias,

    // Projections (all architectures)
    /// Query projection weights
    QProj,
    /// Key projection weights
    KProj,
    /// Value projection weights
    VProj,
    /// Output projection weights
    OProj,

    // FFN (all SwiGLU architectures)
    /// FFN gate projection (SwiGLU/GatedMLP)
    FfnGate,
    /// FFN up projection
    FfnUp,
    /// FFN down projection
    FfnDown,
}

impl WeightRole {
    /// Returns the field name prefix as it appears in `IndexedLayerWeights`.
    #[must_use]
    pub const fn field_name(&self) -> &'static str {
        match self {
            Self::AttnNorm => "attn_norm",
            Self::FfnNorm => "ffn_norm",
            Self::AttnQNorm => "attn_q_norm",
            Self::AttnKNorm => "attn_k_norm",
            Self::AttnQBias => "attn_q_bias",
            Self::AttnKBias => "attn_k_bias",
            Self::AttnVBias => "attn_v_bias",
            Self::QProj => "attn_q (q_proj)",
            Self::KProj => "attn_k (k_proj)",
            Self::VProj => "attn_v (v_proj)",
            Self::OProj => "attn_output (o_proj)",
            Self::FfnGate => "ffn_gate",
            Self::FfnUp => "ffn_up",
            Self::FfnDown => "ffn_down",
        }
    }
}

/// Base roles required by ALL architectures.
const BASE_ROLES: &[WeightRole] = &[
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

/// Roles for architectures with QK norm but no bias (Qwen3).
const ROLES_QK_NORM_NO_BIAS: &[WeightRole] = &[
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

/// Roles for architectures with bias but no QK norm (Qwen2, phi).
const ROLES_BIAS_NO_QK_NORM: &[WeightRole] = &[
    WeightRole::AttnNorm,
    WeightRole::FfnNorm,
    WeightRole::QProj,
    WeightRole::KProj,
    WeightRole::VProj,
    WeightRole::OProj,
    WeightRole::FfnGate,
    WeightRole::FfnUp,
    WeightRole::FfnDown,
    WeightRole::AttnQBias,
    WeightRole::AttnKBias,
    WeightRole::AttnVBias,
];

/// Roles for architectures with both QK norm AND bias (future).
const ROLES_QK_NORM_AND_BIAS: &[WeightRole] = &[
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
    WeightRole::AttnQBias,
    WeightRole::AttnKBias,
    WeightRole::AttnVBias,
];

/// Returns the required weight roles for a given architecture.
///
/// Exhaustive match on `(has_qk_norm, has_bias)` — adding a new architecture
/// combination without updating this function will still match one of the
/// four arms, but the contract test FALSIFY-ARCH-001 will catch mismatches.
#[must_use]
pub fn required_roles(arch: &ArchConstraints) -> &'static [WeightRole] {
    match (arch.has_qk_norm, arch.has_bias) {
        (true, true) => ROLES_QK_NORM_AND_BIAS,
        (true, false) => ROLES_QK_NORM_NO_BIAS,
        (false, true) => ROLES_BIAS_NO_QK_NORM,
        (false, false) => BASE_ROLES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_roles_count() {
        // Base: 2 norms + 4 attn proj + 3 ffn proj = 9
        assert_eq!(BASE_ROLES.len(), 9);
    }

    #[test]
    fn test_qk_norm_roles_count() {
        // Base 9 + 2 QK norms = 11
        assert_eq!(ROLES_QK_NORM_NO_BIAS.len(), 11);
    }

    #[test]
    fn test_bias_roles_count() {
        // Base 9 + 3 biases = 12
        assert_eq!(ROLES_BIAS_NO_QK_NORM.len(), 12);
    }

    #[test]
    fn test_both_roles_count() {
        // Base 9 + 2 QK norms + 3 biases = 14
        assert_eq!(ROLES_QK_NORM_AND_BIAS.len(), 14);
    }

    #[test]
    fn test_llama_requires_base_only() {
        let arch = ArchConstraints::from_architecture("llama");
        let roles = required_roles(&arch);
        assert_eq!(roles.len(), 9);
        assert!(!roles.contains(&WeightRole::AttnQNorm));
        assert!(!roles.contains(&WeightRole::AttnQBias));
    }

    #[test]
    fn test_qwen2_requires_bias() {
        let arch = ArchConstraints::from_architecture("qwen2");
        let roles = required_roles(&arch);
        assert!(roles.contains(&WeightRole::AttnQBias));
        assert!(roles.contains(&WeightRole::AttnKBias));
        assert!(roles.contains(&WeightRole::AttnVBias));
        assert!(!roles.contains(&WeightRole::AttnQNorm));
    }

    #[test]
    fn test_qwen3_requires_qk_norm() {
        let arch = ArchConstraints::from_architecture("qwen3");
        let roles = required_roles(&arch);
        assert!(roles.contains(&WeightRole::AttnQNorm));
        assert!(roles.contains(&WeightRole::AttnKNorm));
        assert!(!roles.contains(&WeightRole::AttnQBias));
    }

    #[test]
    fn test_mistral_requires_base_only() {
        let arch = ArchConstraints::from_architecture("mistral");
        let roles = required_roles(&arch);
        assert_eq!(roles.len(), 9);
    }

    #[test]
    fn test_all_architectures_have_base_roles() {
        let archs = [
            "llama", "qwen2", "qwen3", "mistral", "gemma", "phi", "phi3", "deepseek",
        ];
        for arch_name in archs {
            let arch = ArchConstraints::from_architecture(arch_name);
            let roles = required_roles(&arch);
            for base_role in BASE_ROLES {
                assert!(
                    roles.contains(base_role),
                    "Architecture '{}' missing base role {:?}",
                    arch_name,
                    base_role
                );
            }
        }
    }

    #[test]
    fn test_all_roles_have_field_names() {
        let arch = ArchConstraints::from_architecture("qwen3");
        for role in required_roles(&arch) {
            assert!(!role.field_name().is_empty());
        }
    }

    #[test]
    fn test_no_duplicate_roles() {
        let mut arch = ArchConstraints::from_architecture("qwen3");
        arch.has_bias = true; // Synthetic: both flags set
        let roles = required_roles(&arch);
        let mut seen = std::collections::HashSet::new();
        for role in roles {
            assert!(seen.insert(role), "Duplicate role: {:?}", role);
        }
    }
}
