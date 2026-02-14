//! Tensor Validation Contract (PMAT-234, PMAT-235)
//!
//! Makes it IMPOSSIBLE to load garbage data from GGUF, APR, or SafeTensors.
//!
//! ## Design Principle
//!
//! Every tensor load MUST pass semantic validation before use.
//! A tensor that parses correctly but contains garbage MUST be rejected.
//!
//! ## Compile-Time Enforcement (PMAT-235)
//!
//! This module implements the Poka-Yoke (mistake-proofing) pattern from the
//! Toyota Production System. The newtype pattern makes invalid tensor states
//! unrepresentable at the type level.
//!
//! ## Theoretical Foundation
//!
//! - Shingo, S. (1986). Zero Quality Control: Source Inspection and the
//!   Poka-Yoke System. Productivity Press.
//! - Brady, E. (2017). Type-Driven Development with Idris. Manning.
//! - Parsons, A. (2019). "Parse, Don't Validate"
//!
//! ## Validation Gates
//!
//! 1. **Density Gate**: Rejects tensors that are mostly zeros (dead weights)
//! 2. **Distribution Gate**: Rejects tensors with abnormal value distributions
//! 3. **Shape Gate**: Rejects tensors with impossible shapes for their role
//! 4. **NaN/Inf Gate**: Rejects tensors containing NaN or Inf values
//!
//! ## Contract
//!
//! See `aprender/contracts/tensor-layout-v1.yaml` for the full specification.

use crate::error::{RealizarError, Result};
use std::fmt;

/// Tensor validation statistics
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Total number of elements
    pub len: usize,
    /// Count of zero values (|x| < 1e-10)
    pub zero_count: usize,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
    /// Minimum value (excluding NaN/Inf)
    pub min: f32,
    /// Maximum value (excluding NaN/Inf)
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// L2 norm (Frobenius norm)
    pub l2_norm: f32,
}

impl TensorStats {
    /// Compute statistics for a tensor
    pub fn compute(data: &[f32]) -> Self {
        let len = data.len();
        if len == 0 {
            return Self {
                len: 0,
                zero_count: 0,
                nan_count: 0,
                inf_count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                l2_norm: 0.0,
            };
        }

        let mut zero_count = 0;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                if v.abs() < 1e-10 {
                    zero_count += 1;
                }
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                sum += v as f64;
                sum_sq += (v as f64) * (v as f64);
            }
        }

        Self {
            len,
            zero_count,
            nan_count,
            inf_count,
            min: if min == f32::INFINITY { 0.0 } else { min },
            max: if max == f32::NEG_INFINITY { 0.0 } else { max },
            mean: (sum / len as f64) as f32,
            l2_norm: (sum_sq.sqrt()) as f32,
        }
    }

    /// Percentage of zeros
    pub fn zero_pct(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        100.0 * self.zero_count as f32 / self.len as f32
    }
}

/// Validation result with detailed diagnostics
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether validation passed all gates
    pub passed: bool,
    /// Computed tensor statistics
    pub stats: TensorStats,
    /// List of failure messages (empty if passed)
    pub failures: Vec<String>,
}

/// Validate an embedding tensor
///
/// Embeddings MUST have:
/// - Less than 50% zeros (dead embeddings = broken model)
/// - No NaN or Inf values
/// - Non-zero L2 norm
/// - Reasonable value range (not all identical)
pub fn validate_embedding(
    name: &str,
    data: &[f32],
    vocab_size: usize,
    hidden_dim: usize,
) -> ValidationResult {
    let stats = TensorStats::compute(data);
    let mut failures = Vec::new();

    // Gate 1: Shape validation
    let expected_len = vocab_size * hidden_dim;
    if data.len() != expected_len {
        failures.push(format!(
            "Shape mismatch: got {} elements, expected {} ({}x{})",
            data.len(),
            expected_len,
            vocab_size,
            hidden_dim
        ));
    }

    // Gate 2: Density validation (CRITICAL - detects incorrect data offsets)
    let zero_pct = stats.zero_pct();
    if zero_pct > 50.0 {
        failures.push(format!(
            "DENSITY FAILURE: {:.1}% zeros (max 50%). Data likely loaded from wrong offset!",
            zero_pct
        ));
    }

    // Gate 3: NaN/Inf validation
    if stats.nan_count > 0 {
        failures.push(format!("Contains {} NaN values", stats.nan_count));
    }
    if stats.inf_count > 0 {
        failures.push(format!("Contains {} Inf values", stats.inf_count));
    }

    // Gate 4: Distribution validation
    if stats.l2_norm < 1e-6 {
        failures.push("L2 norm ~0: tensor is effectively empty".to_string());
    }
    if (stats.max - stats.min).abs() < 1e-10 {
        failures.push("All values identical: tensor is constant".to_string());
    }

    // Gate 5: Sample non-zero tokens (spot check)
    // Check tokens at 10%, 50%, 90% of vocab to ensure data is distributed
    for pct in [10, 50, 90] {
        let token_id = vocab_size * pct / 100;
        let start = token_id * hidden_dim;
        let end = start + hidden_dim;
        if end <= data.len() {
            let token_l2: f32 = data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
            if token_l2 < 1e-6 {
                failures.push(format!(
                    "Token {} ({}% of vocab) has L2=0: embedding data likely corrupted",
                    token_id, pct
                ));
            }
        }
    }

    let passed = failures.is_empty();
    if !passed {
        eprintln!("[VALIDATION FAILED] {}: {:?}", name, failures);
    }

    ValidationResult {
        passed,
        stats,
        failures,
    }
}

/// Validate a weight matrix (linear layer)
pub fn validate_weight(
    name: &str,
    data: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> ValidationResult {
    let stats = TensorStats::compute(data);
    let mut failures = Vec::new();

    // Gate 1: Shape
    let expected_len = out_dim * in_dim;
    if data.len() != expected_len {
        failures.push(format!(
            "Shape mismatch: got {} elements, expected {} ({}x{})",
            data.len(),
            expected_len,
            out_dim,
            in_dim
        ));
    }

    // Gate 2: Density (weights should be mostly non-zero)
    let zero_pct = stats.zero_pct();
    if zero_pct > 80.0 {
        failures.push(format!("DENSITY FAILURE: {:.1}% zeros (max 80%)", zero_pct));
    }

    // Gate 3: NaN/Inf
    if stats.nan_count > 0 {
        failures.push(format!("Contains {} NaN values", stats.nan_count));
    }
    if stats.inf_count > 0 {
        failures.push(format!("Contains {} Inf values", stats.inf_count));
    }

    // Gate 4: Distribution
    if stats.l2_norm < 1e-6 {
        failures.push("L2 norm ~0".to_string());
    }

    let passed = failures.is_empty();
    if !passed {
        eprintln!("[VALIDATION FAILED] {}: {:?}", name, failures);
    }

    ValidationResult {
        passed,
        stats,
        failures,
    }
}

/// Validate a 1D tensor (bias, norm weight)
pub fn validate_vector(_name: &str, data: &[f32], expected_len: usize) -> ValidationResult {
    let stats = TensorStats::compute(data);
    let mut failures = Vec::new();

    if data.len() != expected_len {
        failures.push(format!(
            "Length mismatch: got {}, expected {}",
            data.len(),
            expected_len
        ));
    }

    if stats.nan_count > 0 {
        failures.push(format!("Contains {} NaN values", stats.nan_count));
    }
    if stats.inf_count > 0 {
        failures.push(format!("Contains {} Inf values", stats.inf_count));
    }

    let passed = failures.is_empty();
    ValidationResult {
        passed,
        stats,
        failures,
    }
}

/// Enforce validation - returns error if validation fails
pub fn enforce_embedding_validation(
    name: &str,
    data: &[f32],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<()> {
    let result = validate_embedding(name, data, vocab_size, hidden_dim);
    if !result.passed {
        return Err(RealizarError::FormatError {
            reason: format!(
                "Tensor '{}' failed validation: {}",
                name,
                result.failures.join("; ")
            ),
        });
    }
    Ok(())
}

/// Enforce weight validation
pub fn enforce_weight_validation(
    name: &str,
    data: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    let result = validate_weight(name, data, out_dim, in_dim);
    if !result.passed {
        return Err(RealizarError::FormatError {
            reason: format!(
                "Tensor '{}' failed validation: {}",
                name,
                result.failures.join("; ")
            ),
        });
    }
    Ok(())
}

// =============================================================================
// VALIDATED NEWTYPES - Compile-Time Contract Enforcement (PMAT-235)
// =============================================================================
//
// These types implement the Poka-Yoke pattern: the inner data is private,
// so the ONLY way to construct these types is via the validated constructor.
// This makes it IMPOSSIBLE to use unvalidated tensor data at compile time.
//
// Citation: Shingo, S. (1986). Zero Quality Control: Source Inspection and
//           the Poka-Yoke System. Productivity Press.
// =============================================================================

/// Contract validation error (mirrors aprender::format::ContractValidationError)
#[derive(Debug, Clone)]
pub struct ContractValidationError {
    /// Name of the tensor that failed validation
    pub tensor_name: String,
    /// Contract rule ID that was violated (e.g., "F-DATA-QUALITY-001")
    pub rule_id: String,
    /// Human-readable error message
    pub message: String,
}

impl fmt::Display for ContractValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] Tensor '{}': {}",
            self.rule_id, self.tensor_name, self.message
        )
    }
}

impl std::error::Error for ContractValidationError {}

impl From<ContractValidationError> for RealizarError {
    fn from(e: ContractValidationError) -> Self {
        RealizarError::FormatError {
            reason: e.to_string(),
        }
    }
}

/// Validated embedding tensor - compile-time guarantee of data quality
///
/// This type can ONLY be constructed via `new()`, which enforces:
/// - Correct element count (vocab_size * hidden_dim)
/// - Density check (<50% zeros) - catches PMAT-234 bug
/// - No NaN or Inf values
/// - Non-degenerate distribution (L2 > 1e-6, values vary)
/// - Spot check at 10%/50%/90% of vocab
///
/// # Poka-Yoke Guarantee
///
/// The inner `data` field is private. There is no way to construct this type
/// without passing validation. This makes the PMAT-234 bug (94.5% zeros)
/// impossible at compile time.
#[derive(Debug, Clone)]
pub struct ValidatedEmbedding {
    // PRIVATE - cannot be accessed without going through new()
    data: Vec<f32>,
    vocab_size: usize,
    hidden_dim: usize,
    stats: TensorStats,
}

impl ValidatedEmbedding {
    /// Contract thresholds from tensor-layout-v1.yaml
    const MAX_ZERO_PCT: f32 = 50.0;
    const MIN_L2_NORM: f32 = 1e-6;
    const MIN_TOKEN_L2: f32 = 1e-6;
    const SPOT_CHECK_PCTS: [usize; 3] = [10, 50, 90];

    /// Construct a validated embedding tensor
    ///
    /// This is the ONLY way to create a ValidatedEmbedding. All contract
    /// rules are enforced here.
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` if any validation rule fails.
    pub fn new(
        data: Vec<f32>,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> std::result::Result<Self, ContractValidationError> {
        let name = "embedding";

        // Gate 1: Shape validation (structural)
        let expected_len = vocab_size * hidden_dim;
        if data.len() != expected_len {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
                message: format!(
                    "Shape mismatch: got {} elements, expected {} ({}x{})",
                    data.len(),
                    expected_len,
                    vocab_size,
                    hidden_dim
                ),
            });
        }

        let stats = TensorStats::compute(&data);

        // Gate 2: Density validation (F-DATA-QUALITY-001)
        if stats.zero_pct() > Self::MAX_ZERO_PCT {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-001".to_string(),
                message: format!(
                    "DENSITY FAILURE: {:.1}% zeros (max {}%). Data likely loaded from wrong offset!",
                    stats.zero_pct(),
                    Self::MAX_ZERO_PCT
                ),
            });
        }

        // Gate 3: NaN validation (F-DATA-QUALITY-002)
        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }

        // Gate 4: Inf validation (F-DATA-QUALITY-002)
        if stats.inf_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} Inf values", stats.inf_count),
            });
        }

        // Gate 5: L2 norm validation (F-DATA-QUALITY-003)
        if stats.l2_norm < Self::MIN_L2_NORM {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-003".to_string(),
                message: "L2 norm ~0: tensor is effectively empty".to_string(),
            });
        }

        // Gate 6: Variation validation (F-DATA-QUALITY-003)
        if (stats.max - stats.min).abs() < 1e-10 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-003".to_string(),
                message: "All values identical: tensor is constant".to_string(),
            });
        }

        // Gate 7: Spot check validation (F-DATA-QUALITY-004)
        for pct in Self::SPOT_CHECK_PCTS {
            let token_id = vocab_size * pct / 100;
            let start = token_id * hidden_dim;
            let end = start + hidden_dim;
            if end <= data.len() {
                let token_l2: f32 = data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
                if token_l2 < Self::MIN_TOKEN_L2 {
                    return Err(ContractValidationError {
                        tensor_name: name.to_string(),
                        rule_id: "F-DATA-QUALITY-004".to_string(),
                        message: format!(
                            "Token {} ({}% of vocab) has L2={:.2e}: embedding data likely corrupted or offset",
                            token_id, pct, token_l2
                        ),
                    });
                }
            }
        }

        Ok(Self {
            data,
            vocab_size,
            hidden_dim,
            stats,
        })
    }

    /// Access the validated data
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Consume and return the inner data
    #[must_use]
    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }

    /// Get vocab size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get validation statistics
    #[must_use]
    pub fn stats(&self) -> &TensorStats {
        &self.stats
    }
}

/// Validated weight matrix - compile-time guarantee of data quality
#[derive(Debug, Clone)]
pub struct ValidatedWeight {
    data: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
    name: String,
    stats: TensorStats,
}

impl ValidatedWeight {
    const MAX_ZERO_PCT: f32 = 80.0;
    const MIN_L2_NORM: f32 = 1e-6;

    /// Construct a validated weight matrix
    pub fn new(
        data: Vec<f32>,
        out_dim: usize,
        in_dim: usize,
        name: &str,
    ) -> std::result::Result<Self, ContractValidationError> {
        let expected_len = out_dim * in_dim;
        if data.len() != expected_len {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
                message: format!(
                    "Shape mismatch: got {} elements, expected {} ({}x{})",
                    data.len(),
                    expected_len,
                    out_dim,
                    in_dim
                ),
            });
        }

        let stats = TensorStats::compute(&data);

        if stats.zero_pct() > Self::MAX_ZERO_PCT {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-001".to_string(),
                message: format!(
                    "DENSITY FAILURE: {:.1}% zeros (max {}%)",
                    stats.zero_pct(),
                    Self::MAX_ZERO_PCT
                ),
            });
        }

        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }

        if stats.inf_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} Inf values", stats.inf_count),
            });
        }

        if stats.l2_norm < Self::MIN_L2_NORM {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-003".to_string(),
                message: "L2 norm ~0: tensor is effectively empty".to_string(),
            });
        }

        Ok(Self {
            data,
            out_dim,
            in_dim,
            name: name.to_string(),
            stats,
        })
    }

    /// Access the validated data
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Consume and return the inner data
    #[must_use]
    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }

    /// Get output dimension
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Get input dimension
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Get tensor name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get validation statistics
    #[must_use]
    pub fn stats(&self) -> &TensorStats {
        &self.stats
    }
}

/// Validated 1D tensor (bias, norm weights)
#[derive(Debug, Clone)]
pub struct ValidatedVector {
    data: Vec<f32>,
    name: String,
    stats: TensorStats,
}

impl ValidatedVector {
    /// Construct a validated vector
    pub fn new(
        data: Vec<f32>,
        expected_len: usize,
        name: &str,
    ) -> std::result::Result<Self, ContractValidationError> {
        if data.len() != expected_len {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-LAYOUT-CONTRACT-003".to_string(),
                message: format!(
                    "Length mismatch: got {}, expected {}",
                    data.len(),
                    expected_len
                ),
            });
        }

        let stats = TensorStats::compute(&data);

        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }

        if stats.inf_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} Inf values", stats.inf_count),
            });
        }

        Ok(Self {
            data,
            name: name.to_string(),
            stats,
        })
    }

    /// Access the validated data
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Consume and return the inner data
    #[must_use]
    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }

    /// Get tensor name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get validation statistics
    #[must_use]
    pub fn stats(&self) -> &TensorStats {
        &self.stats
    }
}

// =============================================================================
// VALIDATED APR TRANSFORMER — Compile-Time Tensor Quality Enforcement (PMAT-235)
// =============================================================================
//
// Follows the PreparedTokens pattern from PMAT-236: the inner AprTransformer
// is PRIVATE, so construction ONLY via validate(). This makes it IMPOSSIBLE
// to use an AprTransformer with corrupt tensors in inference.
// =============================================================================

use crate::apr_transformer::{AprTransformer, AprTransformerConfig};

/// Validated APR Transformer — compile-time guarantee of tensor data quality
///
/// This wrapper validates ALL tensors in an `AprTransformer`:
/// - Embedding: density, NaN/Inf, L2 norm, spot check
/// - Layer weights: density, NaN/Inf, L2 norm, shape
/// - Norm vectors: NaN/Inf, length
///
/// The inner `AprTransformer` is private. The ONLY way to construct this type
/// is via `validate()`, which enforces all contract rules.
///
/// Access the inner transformer transparently via `Deref`.
#[derive(Debug, Clone)]
pub struct ValidatedAprTransformer {
    // PRIVATE — only constructible via validate()
    inner: AprTransformer,
}

impl ValidatedAprTransformer {
    /// Validate all tensors in an `AprTransformer`
    ///
    /// This is the ONLY way to create a `ValidatedAprTransformer`.
    /// Every tensor is validated using the existing newtype gates:
    /// - `ValidatedEmbedding` for token_embedding
    /// - `ValidatedWeight` for weight matrices
    /// - `ValidatedVector` for norm weights and biases
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` identifying the first tensor that fails.
    pub fn validate(
        transformer: AprTransformer,
    ) -> std::result::Result<Self, ContractValidationError> {
        let config = &transformer.config;
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let intermediate_dim = config.intermediate_dim;
        let head_dim = if config.num_heads > 0 {
            hidden_dim / config.num_heads
        } else {
            hidden_dim
        };
        let kv_dim = config.num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        // === Global tensors ===

        // token_embedding: [vocab_size * hidden_dim]
        ValidatedEmbedding::new(transformer.token_embedding.clone(), vocab_size, hidden_dim)
            .map_err(|mut e| {
                e.tensor_name = "token_embedding".to_string();
                e
            })?;

        // output_norm_weight: [hidden_dim]
        ValidatedVector::new(
            transformer.output_norm_weight.clone(),
            hidden_dim,
            "output_norm_weight",
        )?;

        // output_norm_bias (optional)
        if let Some(ref bias) = transformer.output_norm_bias {
            ValidatedVector::new(bias.clone(), hidden_dim, "output_norm_bias")?;
        }

        // lm_head_weight: [vocab_size * hidden_dim]
        ValidatedWeight::new(
            transformer.lm_head_weight.clone(),
            vocab_size,
            hidden_dim,
            "lm_head_weight",
        )?;

        // lm_head_bias (optional)
        if let Some(ref bias) = transformer.lm_head_bias {
            ValidatedVector::new(bias.clone(), vocab_size, "lm_head_bias")?;
        }

        // === Per-layer tensors ===
        for (i, layer) in transformer.layers.iter().enumerate() {
            // attn_norm_weight: [hidden_dim]
            ValidatedVector::new(
                layer.attn_norm_weight.clone(),
                hidden_dim,
                &format!("layers.{i}.attn_norm_weight"),
            )?;

            // attn_norm_bias (optional)
            if let Some(ref bias) = layer.attn_norm_bias {
                ValidatedVector::new(
                    bias.clone(),
                    hidden_dim,
                    &format!("layers.{i}.attn_norm_bias"),
                )?;
            }

            // qkv_weight: [qkv_out_dim * hidden_dim]
            ValidatedWeight::new(
                layer.qkv_weight.clone(),
                qkv_out_dim,
                hidden_dim,
                &format!("layers.{i}.qkv_weight"),
            )?;

            // qkv_bias (optional)
            if let Some(ref bias) = layer.qkv_bias {
                ValidatedVector::new(bias.clone(), qkv_out_dim, &format!("layers.{i}.qkv_bias"))?;
            }

            // attn_output_weight: [hidden_dim * hidden_dim]
            ValidatedWeight::new(
                layer.attn_output_weight.clone(),
                hidden_dim,
                hidden_dim,
                &format!("layers.{i}.attn_output_weight"),
            )?;

            // attn_output_bias (optional)
            if let Some(ref bias) = layer.attn_output_bias {
                ValidatedVector::new(
                    bias.clone(),
                    hidden_dim,
                    &format!("layers.{i}.attn_output_bias"),
                )?;
            }

            // ffn_gate_weight (optional): [intermediate_dim * hidden_dim]
            if let Some(ref w) = layer.ffn_gate_weight {
                ValidatedWeight::new(
                    w.clone(),
                    intermediate_dim,
                    hidden_dim,
                    &format!("layers.{i}.ffn_gate_weight"),
                )?;
            }

            // ffn_gate_bias (optional)
            if let Some(ref bias) = layer.ffn_gate_bias {
                ValidatedVector::new(
                    bias.clone(),
                    intermediate_dim,
                    &format!("layers.{i}.ffn_gate_bias"),
                )?;
            }

            // ffn_up_weight: [intermediate_dim * hidden_dim]
            ValidatedWeight::new(
                layer.ffn_up_weight.clone(),
                intermediate_dim,
                hidden_dim,
                &format!("layers.{i}.ffn_up_weight"),
            )?;

            // ffn_up_bias (optional)
            if let Some(ref bias) = layer.ffn_up_bias {
                ValidatedVector::new(
                    bias.clone(),
                    intermediate_dim,
                    &format!("layers.{i}.ffn_up_bias"),
                )?;
            }

            // ffn_down_weight: [hidden_dim * intermediate_dim]
            ValidatedWeight::new(
                layer.ffn_down_weight.clone(),
                hidden_dim,
                intermediate_dim,
                &format!("layers.{i}.ffn_down_weight"),
            )?;

            // ffn_down_bias (optional)
            if let Some(ref bias) = layer.ffn_down_bias {
                ValidatedVector::new(
                    bias.clone(),
                    hidden_dim,
                    &format!("layers.{i}.ffn_down_bias"),
                )?;
            }

            // ffn_norm_weight (optional): [hidden_dim]
            if let Some(ref w) = layer.ffn_norm_weight {
                ValidatedVector::new(
                    w.clone(),
                    hidden_dim,
                    &format!("layers.{i}.ffn_norm_weight"),
                )?;
            }

            // ffn_norm_bias (optional)
            if let Some(ref bias) = layer.ffn_norm_bias {
                ValidatedVector::new(
                    bias.clone(),
                    hidden_dim,
                    &format!("layers.{i}.ffn_norm_bias"),
                )?;
            }
        }

        Ok(Self { inner: transformer })
    }

    /// Access the inner transformer
    #[must_use]
    pub fn transformer(&self) -> &AprTransformer {
        &self.inner
    }

    /// Consume and return the inner transformer
    #[must_use]
    pub fn into_inner(self) -> AprTransformer {
        self.inner
    }

    /// Access the model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.inner.config
    }
}

impl std::ops::Deref for ValidatedAprTransformer {
    type Target = AprTransformer;
    fn deref(&self) -> &AprTransformer {
        &self.inner
    }
}

// =============================================================================
// POPPERIAN FALSIFICATION TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // LEGACY FUNCTION TESTS (validate_embedding, etc.)
    // =========================================================================

    #[test]
    fn test_validates_good_embedding() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(
            result.passed,
            "Good embedding should pass: {:?}",
            result.failures
        );
    }

    #[test]
    fn test_rejects_mostly_zero_embedding() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data = vec![0.0f32; vocab_size * hidden_dim];
        for i in (vocab_size * 95 / 100 * hidden_dim)..(vocab_size * hidden_dim) {
            data[i] = 0.1;
        }

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(!result.passed, "95% zero embedding should fail");
        assert!(result.failures.iter().any(|f| f.contains("DENSITY")));
    }

    #[test]
    fn test_rejects_nan_embedding() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = vec![0.1; vocab_size * hidden_dim];
        data[5] = f32::NAN;

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(!result.passed, "NaN embedding should fail");
    }

    #[test]
    fn test_rejects_wrong_shape() {
        let data = vec![0.1f32; 1000];
        let result = validate_embedding("test", &data, 100, 64);
        assert!(!result.passed, "Wrong shape should fail");
    }

    // =========================================================================
    // POPPERIAN FALSIFICATION TESTS FOR NEWTYPES (PMAT-235)
    // Per Popper (1959), these attempt to DISPROVE the contract works.
    // =========================================================================

    #[test]
    fn falsify_001_validated_embedding_rejects_all_zeros() {
        let bad_data = vec![0.0f32; 100 * 64];
        let result = ValidatedEmbedding::new(bad_data, 100, 64);
        assert!(result.is_err(), "Should reject 100% zeros");
        let err = result.unwrap_err();
        assert!(err.message.contains("DENSITY"), "Error: {}", err.message);
    }

    #[test]
    fn falsify_001_validated_embedding_rejects_94pct_zeros() {
        // Simulate PMAT-234 bug
        let vocab_size = 1000;
        let hidden_dim = 64;
        let mut data = vec![0.0f32; vocab_size * hidden_dim];
        for i in (945 * hidden_dim)..(vocab_size * hidden_dim) {
            data[i] = 0.1;
        }
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject 94.5% zeros");
    }

    #[test]
    fn falsify_001_validated_embedding_accepts_good_data() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(
            result.is_ok(),
            "Should accept good data: {:?}",
            result.err()
        );
    }

    #[test]
    fn falsify_003_validated_embedding_rejects_nan() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        data[5] = f32::NAN;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject NaN");
    }

    #[test]
    fn falsify_004_spot_check_catches_offset_bug() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Zero out token at 10% (token 10)
        let token_10_start = 10 * hidden_dim;
        for i in token_10_start..(token_10_start + hidden_dim) {
            data[i] = 0.0;
        }

        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should catch zero token at 10%");
        assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-004");
    }

    #[test]
    fn falsify_005_rejects_wrong_shape() {
        let data = vec![0.1f32; 1000];
        let result = ValidatedEmbedding::new(data, 100, 64);
        assert!(result.is_err(), "Should reject wrong shape");
    }

    #[test]
    fn validated_weight_rejects_all_zeros() {
        let data = vec![0.0f32; 100];
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_err());
    }

    #[test]
    fn validated_weight_accepts_good_data() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_ok());
    }

    #[test]
    fn validated_vector_rejects_wrong_length() {
        let data = vec![0.1f32; 50];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_err());
    }

    #[test]
    fn validated_vector_accepts_good_data() {
        let data = vec![1.0f32; 100];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_ok());
    }

    // =========================================================================
    // VALIDATED APR TRANSFORMER FALSIFICATION TESTS (PMAT-235)
    // =========================================================================

    /// Helper: create a valid AprTransformer for testing
    fn make_valid_transformer(num_layers: usize) -> AprTransformer {
        use crate::apr_transformer::{AprTransformerConfig, AprTransformerLayer};

        let hidden_dim = 16;
        let num_heads = 4;
        let num_kv_heads = 4;
        let vocab_size = 32;
        let intermediate_dim = 64;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-6,
        };

        // Non-zero sin pattern data
        let make_data = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
                .collect()
        };

        let layers = (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: make_data(qkv_out_dim * hidden_dim),
                qkv_bias: None,
                attn_output_weight: make_data(hidden_dim * hidden_dim),
                attn_output_bias: None,
                ffn_gate_weight: Some(make_data(intermediate_dim * hidden_dim)),
                ffn_gate_bias: None,
                ffn_up_weight: make_data(intermediate_dim * hidden_dim),
                ffn_up_bias: None,
                ffn_down_weight: make_data(hidden_dim * intermediate_dim),
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; hidden_dim]),
                ffn_norm_bias: None,
            })
            .collect();

        AprTransformer {
            config,
            token_embedding: make_data(vocab_size * hidden_dim),
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: make_data(vocab_size * hidden_dim),
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    #[test]
    fn falsify_validated_transformer_rejects_nan_embedding() {
        let mut t = make_valid_transformer(1);
        t.token_embedding[5] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject NaN in embedding");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("embedding"), "Error: {err}");
    }

    #[test]
    fn falsify_validated_transformer_rejects_zero_layer_weight() {
        let mut t = make_valid_transformer(1);
        // Zero out entire qkv_weight → density gate
        let len = t.layers[0].qkv_weight.len();
        t.layers[0].qkv_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject all-zero qkv_weight");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("qkv_weight"), "Error: {err}");
    }

    #[test]
    fn falsify_validated_transformer_rejects_nan_in_deep_layer() {
        let mut t = make_valid_transformer(4);
        t.layers[3].ffn_up_weight[0] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject NaN in layer 3 ffn_up");
        let err = result.unwrap_err();
        assert!(
            err.tensor_name.contains("layers.3.ffn_up_weight"),
            "Error: {err}"
        );
    }

    #[test]
    fn falsify_validated_transformer_identifies_tensor_name() {
        let mut t = make_valid_transformer(2);
        // Corrupt lm_head_weight
        let len = t.lm_head_weight.len();
        t.lm_head_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(
            err.tensor_name, "lm_head_weight",
            "Error should name the tensor: {err}"
        );
    }

    #[test]
    fn validated_transformer_accepts_good_model() {
        let t = make_valid_transformer(2);
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_ok(), "Good model should pass: {:?}", result.err());
    }

    #[test]
    fn validated_transformer_deref_transparent_access() {
        let t = make_valid_transformer(1);
        let validated = ValidatedAprTransformer::validate(t).expect("validation should pass");

        // Access config through Deref
        assert_eq!(validated.config.hidden_dim, 16);
        assert_eq!(validated.config.num_layers, 1);
        assert_eq!(validated.config.vocab_size, 32);

        // Access fields through Deref
        assert!(!validated.token_embedding.is_empty());
        assert_eq!(validated.layers.len(), 1);

        // Access through explicit methods
        assert_eq!(validated.config().hidden_dim, 16);
        assert_eq!(validated.transformer().config.num_layers, 1);

        // into_inner works
        let inner = validated.into_inner();
        assert_eq!(inner.config.hidden_dim, 16);
    }

    // =========================================================================
    // GH-46 FALSIFICATION: Rosetta strict validation boundaries
    // =========================================================================

    /// GH-46: Embedding density gate must reject >50% zeros.
    /// Before the fix, validation was too lenient and passed all-zero embeddings.
    #[test]
    fn test_falsify_gh46_embedding_density_threshold_50pct() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        let total = vocab * hidden;
        // 51% zeros — must FAIL
        let zero_count = (total as f64 * 0.51).ceil() as usize;
        let mut data = vec![1.0_f32; total];
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }
        let result = validate_embedding("test_embed", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: >50% zero embedding must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: Weight density gate must reject >80% zeros.
    /// Before the fix, weights with mostly zeros passed validation silently.
    #[test]
    fn test_falsify_gh46_weight_density_threshold_80pct() {
        let rows = 32_usize;
        let cols = 16_usize;
        let total = rows * cols;
        // 81% zeros — must FAIL
        let zero_count = (total as f64 * 0.81).ceil() as usize;
        let mut data = vec![1.0_f32; total];
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }
        let result = validate_weight("test_weight", &data, rows, cols);
        assert!(
            !result.passed,
            "GH-46: >80% zero weight must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: L2 norm gate must reject flat (constant) tensors.
    /// A constant tensor has zero variance — signals import corruption.
    #[test]
    fn test_falsify_gh46_rejects_flat_tensor() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        // All identical non-zero values — L2 norm > 0 but max-min == 0
        let data = vec![0.5_f32; vocab * hidden];
        let result = validate_embedding("test_flat", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: Flat (constant) embedding must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: NaN gate must catch even a single NaN in embeddings.
    /// Before strict validation, NaN could propagate through inference.
    #[test]
    fn test_falsify_gh46_single_nan_detected() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        let mut data: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        // Inject single NaN at arbitrary position
        data[vocab * hidden / 2] = f32::NAN;
        let result = validate_embedding("test_nan", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: Single NaN must be caught by validation"
        );
        assert!(
            result
                .failures
                .iter()
                .any(|f| f.to_lowercase().contains("nan")),
            "GH-46: Failure message must mention NaN"
        );
    }
}

// T-COV-95 Coverage Bridge (Part 02 - Accessors, error paths, optional biases)
#[cfg(test)]
#[path = "validation_tests_part_02.rs"]
mod validation_tests_part_02;
