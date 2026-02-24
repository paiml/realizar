//! Pure mathematical operations for GGUF inference
//!
//! This module contains standalone math functions used by both CPU and GPU
//! inference paths. By extracting these to a shared module, we enable:
//!
//! - Code reuse between `OwnedQuantizedModel` (CPU) and `OwnedQuantizedModelCuda` (GPU)
//! - Easier testing of mathematical correctness
//! - Clear separation of concerns
//!
//! ## Functions
//!
//! - `rms_norm`: RMSNorm normalization (LLaMA, Qwen, Mistral)
//! - `gelu`: GELU activation function
//! - `silu`: SiLU/Swish activation function
//! - `add_bias`: Add bias vector to output
//! - `argmax`: Find index of maximum value
//! - `softmax`: Numerically stable softmax

use trueno::Vector as TruenoVector;

// =============================================================================
// Normalization Operations
// =============================================================================

/// RMSNorm (Root Mean Square Layer Normalization)
///
/// Used by LLaMA, TinyLlama, Qwen, Mistral instead of LayerNorm.
/// Formula: output = x / sqrt(mean(x^2) + eps) * weight
///
/// # Arguments
/// * `input` - Input tensor [seq_len * hidden_dim]
/// * `weight` - Normalization weights [hidden_dim]
/// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
///
/// # Returns
/// Normalized output [seq_len * hidden_dim]
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());

    let weight_vec = TruenoVector::from_slice(weight);

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let x = &input[start..end];

        let x_vec = TruenoVector::from_slice(x);

        // SIMD: sum of squares
        let sum_sq = x_vec
            .sum_of_squares()
            .unwrap_or_else(|_| x.iter().map(|v| v * v).sum::<f32>());

        let mean_sq = sum_sq / hidden_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        // SIMD: scale by inv_rms, then multiply by weight
        match x_vec
            .scale(inv_rms)
            .and_then(|scaled| scaled.mul(&weight_vec))
        {
            Ok(result) => {
                output.extend_from_slice(result.as_slice());
            },
            Err(_) => {
                // Fallback to scalar
                for j in 0..hidden_dim {
                    output.push(x[j] * inv_rms * weight[j]);
                }
            },
        }
    }

    output
}

/// RMSNorm to pre-allocated buffer (zero-allocation path)
///
/// # Arguments
/// * `input` - Input tensor [hidden_dim] (single position)
/// * `weight` - Normalization weights [hidden_dim]
/// * `eps` - Small constant for numerical stability
/// * `output` - Pre-allocated output buffer [hidden_dim]
pub fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let hidden_dim = weight.len();
    let x = &input[..hidden_dim];

    let x_vec = TruenoVector::from_slice(x);
    let weight_vec = TruenoVector::from_slice(weight);

    let sum_sq = x_vec
        .sum_of_squares()
        .unwrap_or_else(|_| x.iter().map(|v| v * v).sum::<f32>());

    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    match x_vec
        .scale(inv_rms)
        .and_then(|scaled| scaled.mul(&weight_vec))
    {
        Ok(result) => {
            output[..hidden_dim].copy_from_slice(result.as_slice());
        },
        Err(_) => {
            for j in 0..hidden_dim {
                output[j] = x[j] * inv_rms * weight[j];
            }
        },
    }
}

/// True Layer Normalization with optional bias
///
/// GH-278: Implements real LayerNorm with mean subtraction.
/// Formula: output = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
/// Used by GPT-2 and phi-2 (models with attn_norm_bias).
///
/// # Arguments
/// * `input` - Input tensor [seq_len * hidden_dim]
/// * `weight` - Normalization weights [hidden_dim]
/// * `bias` - Optional bias [hidden_dim]
/// * `eps` - Small constant for numerical stability
pub fn layer_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());
    let n = hidden_dim as f32;

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let x = &input[start..end];

        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + eps).sqrt();

        for j in 0..hidden_dim {
            let normalized = (x[j] - mean) * inv_std;
            let mut val = normalized * weight[j];
            if let Some(b) = bias {
                val += b[j];
            }
            output.push(val);
        }
    }

    output
}

/// Layer normalization to pre-allocated buffer
///
/// GH-278: Implements real LayerNorm with mean subtraction.
pub fn layer_norm_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    output: &mut [f32],
) {
    let hidden_dim = weight.len();
    let x = &input[..hidden_dim];
    let n = hidden_dim as f32;

    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();

    for j in 0..hidden_dim {
        let normalized = (x[j] - mean) * inv_std;
        output[j] = normalized * weight[j];
        if let Some(b) = bias {
            output[j] += b[j];
        }
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

/// GELU (Gaussian Error Linear Unit) activation
///
/// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// # Arguments
/// * `input` - Input tensor (modified in-place)
#[inline]
pub fn gelu(input: &mut [f32]) {
    // ONE PATH: Per-element delegates to trueno::gelu_scalar (UCBD §4).
    for x in input.iter_mut() {
        *x = trueno::gelu_scalar(*x);
    }
}

/// SiLU (Sigmoid Linear Unit) / Swish activation
///
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Used in SwiGLU FFN (LLaMA, Mistral, etc.)
///
/// # Arguments
/// * `input` - Input tensor (modified in-place)
#[inline]
pub fn silu(input: &mut [f32]) {
    // ONE PATH: Per-element delegates to trueno::silu_scalar (UCBD §4).
    for x in input.iter_mut() {
        *x = trueno::silu_scalar(*x);
    }
}

// =============================================================================
// Utility Operations
// =============================================================================

/// Add bias vector to output tensor
///
/// # Arguments
/// * `output` - Output tensor [seq_len * out_dim] (modified in-place)
/// * `bias` - Bias vector [out_dim]
#[inline]
pub fn add_bias(output: &mut [f32], bias: &[f32]) {
    let out_dim = bias.len();
    let seq_len = output.len() / out_dim;
    for s in 0..seq_len {
        for o in 0..out_dim {
            output[s * out_dim + o] += bias[o];
        }
    }
}

/// Find index of maximum value (greedy decoding)
///
/// # Arguments
/// * `logits` - Logit values [vocab_size]
///
/// # Returns
/// Index of the maximum value
#[inline]
pub fn argmax(logits: &[f32]) -> u32 {
    let mut max_idx = 0u32;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i as u32;
        }
    }
    max_idx
}

/// Numerically stable softmax
///
/// Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
///
/// # Arguments
/// * `logits` - Input logits (modified in-place to probabilities)
pub fn softmax(logits: &mut [f32]) {
    // Find max for numerical stability
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for x in logits.iter_mut() {
        *x *= inv_sum;
    }
}

/// Per-head RMSNorm for QK normalization (GH-279: Qwen3)
///
/// Applies RMSNorm independently to each attention head's Q or K projection.
/// Weight shape is `[head_dim]` and is shared across all heads.
///
/// Formula per head: `head_out = RMSNorm(head_in, weight, eps)`
///
/// # Arguments
/// * `qk` - Q or K tensor `[num_heads * head_dim]` (modified in-place)
/// * `weight` - Norm weight `[head_dim]`
/// * `num_heads` - Number of heads
/// * `eps` - Epsilon for numerical stability
pub fn apply_per_head_rms_norm(qk: &mut [f32], weight: &[f32], num_heads: usize, eps: f32) {
    let head_dim = weight.len();
    debug_assert_eq!(
        qk.len(),
        num_heads * head_dim,
        "QK norm: expected {} elements, got {}",
        num_heads * head_dim,
        qk.len()
    );

    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let head = &mut qk[start..end];

        // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        let sum_sq: f32 = head.iter().map(|v| v * v).sum();
        let mean_sq = sum_sq / head_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        for (j, val) in head.iter_mut().enumerate() {
            *val = *val * inv_rms * weight[j];
        }
    }
}

include!("ops_gelu_zero_positive.rs");

#[cfg(test)]
mod rmsnorm_contract_tests {
    use super::*;

    // =========================================================================
    // FALSIFY-RN: rmsnorm-kernel-v1.yaml contract (realizar rms_norm)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: realizar had zero FALSIFY-RN-* tests despite 15+ RMSNorm functions
    //   Why 2: ops.rs had no test module at all — tested only via integration
    //   Why 3: no mapping from rmsnorm-kernel-v1.yaml to realizar test names
    //   Why 4: realizar predates the provable-contracts YAML convention
    //   Why 5: rms_norm was tested via end-to-end model runs, not unit contracts
    //
    // References:
    //   - provable-contracts/contracts/rmsnorm-kernel-v1.yaml
    //   - Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
    // =========================================================================

    /// FALSIFY-RN-001: Finiteness — output must be finite for all finite input when eps > 0
    #[test]
    fn falsify_rn_001_finiteness() {
        let weight = vec![1.0f32; 8];
        let eps = 1e-5;

        let test_cases: Vec<(&str, Vec<f32>)> = vec![
            ("normal", vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]),
            ("small", vec![1e-7; 8]),
            ("large", vec![1e6; 8]),
            ("mixed", vec![-1e5, 1e5, -1e-5, 1e-5, 0.0, 0.0, 1.0, -1.0]),
        ];

        for (name, input) in &test_cases {
            let output = rms_norm(input, &weight, eps);

            for (i, &val) in output.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "FALSIFIED RN-001: output[{i}] = {val} not finite for case '{name}'"
                );
            }
        }
    }

    /// FALSIFY-RN-001b: rms_norm_into also produces finite output
    #[test]
    fn falsify_rn_001_into_finiteness() {
        let weight = vec![1.0f32; 4];
        let input = vec![1e-7, 1e7, -1e-7, -1e7];
        let mut output = vec![0.0f32; 4];

        rms_norm_into(&input, &weight, 1e-5, &mut output);

        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "FALSIFIED RN-001: rms_norm_into output[{i}] = {val} not finite"
            );
        }
    }

    /// FALSIFY-RN-002: Scale invariance — RMSNorm(α·x) = sign(α)·RMSNorm(x)
    #[test]
    fn falsify_rn_002_scale_invariance() {
        let weight = vec![1.0f32; 4];
        let eps = 1e-6;
        let x = vec![3.0f32, -1.0, 2.0, -4.0];
        let y_base = rms_norm(&x, &weight, eps);

        for &alpha in &[2.0_f32, 0.5, 100.0, -1.0, -3.0] {
            let x_scaled: Vec<f32> = x.iter().map(|&v| v * alpha).collect();
            let y_scaled = rms_norm(&x_scaled, &weight, eps);

            let sign = alpha.signum();
            for (i, (&ys, &yb)) in y_scaled.iter().zip(y_base.iter()).enumerate() {
                let expected = sign * yb;
                let diff = (ys - expected).abs();
                assert!(
                    diff < 1e-4,
                    "FALSIFIED RN-002: rms_norm({alpha}·x)[{i}] = {ys}, expected {expected}"
                );
            }
        }
    }

    /// FALSIFY-RN-004: Zero vector — RMSNorm(0) = 0 (not NaN)
    #[test]
    fn falsify_rn_004_zero_vector() {
        let weight = vec![1.0f32; 4];
        let x = vec![0.0f32; 4];
        let y = rms_norm(&x, &weight, 1e-5);

        for (i, &val) in y.iter().enumerate() {
            assert!(
                val.is_finite(),
                "FALSIFIED RN-004: rms_norm(0)[{i}] = {val} (expected finite)"
            );
            assert!(
                val.abs() < 1e-3,
                "FALSIFIED RN-004: rms_norm(0)[{i}] = {val} (expected ≈ 0)"
            );
        }
    }

    /// FALSIFY-RN-005: Unit γ normalized RMS ≈ 1
    ///
    /// After RMSNorm with unit weights, RMS of output should be ≈ 1
    #[test]
    fn falsify_rn_005_unit_gamma_normalized_rms() {
        let weight = vec![1.0f32; 8];
        let eps = 1e-6;

        let test_vectors: Vec<Vec<f32>> = vec![
            vec![1.0, -2.0, 3.0, -0.5, 4.0, -1.0, 2.5, -3.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ];

        for (idx, x) in test_vectors.iter().enumerate() {
            let y = rms_norm(x, &weight, eps);

            let rms_out: f32 =
                (y.iter().map(|&v| v * v).sum::<f32>() / y.len() as f32).sqrt();

            assert!(
                (rms_out - 1.0).abs() < 0.01,
                "FALSIFIED RN-005: RMS(rms_norm(x)) = {rms_out}, expected ≈ 1.0 (case {idx})"
            );
        }
    }

    /// FALSIFY-RN-002b: rms_norm and rms_norm_into produce same result
    #[test]
    fn falsify_rn_consistency_norm_vs_norm_into() {
        let weight = vec![1.5f32, 0.5, 2.0, 0.8];
        let input = vec![3.0f32, -1.0, 2.0, -4.0];
        let eps = 1e-5;

        let y_alloc = rms_norm(&input, &weight, eps);
        let mut y_into = vec![0.0f32; 4];
        rms_norm_into(&input, &weight, eps, &mut y_into);

        for (i, (&a, &b)) in y_alloc.iter().zip(y_into.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-6,
                "FALSIFIED: rms_norm vs rms_norm_into mismatch at [{i}]: {a} vs {b}"
            );
        }
    }
}

// =========================================================================
// FALSIFY-GE: gelu-kernel-v1.yaml contract (realizar gelu in-place)
// =========================================================================
#[cfg(test)]
mod gelu_contract_tests {
    use super::*;

    /// FALSIFY-GE-001: Non-negativity — gelu(x) >= 0 for positive x
    #[test]
    fn falsify_ge_001_non_negativity() {
        let mut input = vec![0.001, 0.1, 1.0, 5.0, 10.0, 100.0];
        gelu(&mut input);
        for (i, &val) in input.iter().enumerate() {
            assert!(val >= 0.0, "FALSIFIED GE-001: gelu(positive)[{i}] = {val} < 0");
        }
    }

    /// FALSIFY-GE-002: Monotonicity — ordering preserved for positive inputs
    #[test]
    fn falsify_ge_002_positive_monotonicity() {
        let mut input = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        gelu(&mut input);
        for i in 1..input.len() {
            assert!(
                input[i] > input[i - 1],
                "FALSIFIED GE-002: gelu not monotonic: [{i}]={} not > [{}]={}",
                input[i], i - 1, input[i - 1]
            );
        }
    }

    /// FALSIFY-GE-003: Zero preservation — gelu(0) = 0
    #[test]
    fn falsify_ge_003_zero_preservation() {
        let mut input = vec![0.0];
        gelu(&mut input);
        assert!(input[0].abs() < 1e-7, "FALSIFIED GE-003: gelu(0) = {}", input[0]);
    }

    /// FALSIFY-GE-006: Large input stability
    #[test]
    fn falsify_ge_006_large_input_stability() {
        let mut pos = vec![10.0, 50.0, 100.0];
        let mut neg = vec![-10.0, -50.0, -100.0];
        gelu(&mut pos);
        gelu(&mut neg);

        for (i, (&val, &orig)) in pos.iter().zip([10.0, 50.0, 100.0].iter()).enumerate() {
            assert!(
                (val - orig).abs() < 0.01,
                "FALSIFIED GE-006: gelu({orig}) = {val}, expected ≈ {orig}"
            );
        }
        for (i, &val) in neg.iter().enumerate() {
            assert!(val.abs() < 0.01, "FALSIFIED GE-006: gelu(neg)[{i}] = {val}, expected ≈ 0");
        }
    }
}
