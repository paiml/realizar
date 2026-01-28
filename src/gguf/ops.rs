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

/// Layer normalization with optional bias
///
/// PMAT-094: This is actually RMSNorm for LLaMA-style models.
/// Kept for API compatibility with models that expect layer_norm signature.
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

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let x = &input[start..end];

        // RMSNorm: compute root mean square (no mean subtraction!)
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

        for j in 0..hidden_dim {
            let normalized = x[j] / rms;
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
pub fn layer_norm_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    output: &mut [f32],
) {
    let hidden_dim = weight.len();
    let x = &input[..hidden_dim];

    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

    for j in 0..hidden_dim {
        let normalized = x[j] / rms;
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
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const C: f32 = 0.044_715;

    for x in input.iter_mut() {
        let inner = SQRT_2_OVER_PI * (*x + C * *x * *x * *x);
        *x = 0.5 * *x * (1.0 + inner.tanh());
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
    for x in input.iter_mut() {
        *x = *x * (1.0 / (1.0 + (-*x).exp()));
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_zero() {
        let mut input = vec![0.0];
        gelu(&mut input);
        assert!((input[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let mut input = vec![1.0];
        gelu(&mut input);
        // GELU(1) ≈ 0.8413
        assert!((input[0] - 0.8413).abs() < 0.01);
    }

    #[test]
    fn test_silu_zero() {
        let mut input = vec![0.0];
        silu(&mut input);
        assert!((input[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        let mut input = vec![1.0];
        silu(&mut input);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((input[0] - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn test_argmax_negative() {
        let logits = vec![-1.0, -0.5, -2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_softmax_uniform() {
        let mut logits = vec![0.0, 0.0, 0.0];
        softmax(&mut logits);
        for &p in &logits {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_bias() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2];
        add_bias(&mut output, &bias);
        assert!((output[0] - 1.1).abs() < 1e-6);
        assert!((output[1] - 2.2).abs() < 1e-6);
        assert!((output[2] - 3.1).abs() < 1e-6);
        assert!((output[3] - 4.2).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_unit_weight() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms_norm(&input, &weight, 1e-5);

        // Check output is normalized
        let sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.1); // Approximately unit RMS
    }

    #[test]
    fn test_layer_norm_no_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = layer_norm(&input, &weight, None, 1e-5);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![0.5, 0.5];
        let output = layer_norm(&input, &weight, Some(&bias), 1e-5);
        assert_eq!(output.len(), 2);
        // Output should have bias added
        assert!(output[0] > 0.0);
        assert!(output[1] > 0.0);
    }

    // =========================================================================
    // Additional tests for rms_norm_into (zero-allocation path)
    // =========================================================================

    #[test]
    fn test_rms_norm_into_unit_weight() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        rms_norm_into(&input, &weight, 1e-5, &mut output);

        // Check output is normalized
        let sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_rms_norm_into_zeros() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![999.0; 4];
        rms_norm_into(&input, &weight, 1e-5, &mut output);

        // Output should be near zero (only eps prevents division by zero)
        for &val in &output {
            assert!(val.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_into_with_weight_scaling() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![2.0, 0.5, 1.0, 3.0];
        let mut output = vec![0.0; 4];
        rms_norm_into(&input, &weight, 1e-5, &mut output);

        // Each output should be scaled by weight
        // Input RMS = 1.0, so output ≈ weight
        assert!((output[0] - 2.0).abs() < 0.1);
        assert!((output[1] - 0.5).abs() < 0.1);
        assert!((output[2] - 1.0).abs() < 0.1);
        assert!((output[3] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_rms_norm_into_negative_values() {
        let input = vec![-1.0, -2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        rms_norm_into(&input, &weight, 1e-5, &mut output);

        // Signs should be preserved
        assert!(output[0] < 0.0);
        assert!(output[1] < 0.0);
        assert!(output[2] > 0.0);
        assert!(output[3] > 0.0);
    }

    // =========================================================================
    // Additional tests for layer_norm_into (zero-allocation path)
    // =========================================================================

    #[test]
    fn test_layer_norm_into_no_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        layer_norm_into(&input, &weight, None, 1e-5, &mut output);

        assert_eq!(output.len(), 4);
        // Should be normalized
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_layer_norm_into_with_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![10.0, 20.0];
        let mut output = vec![0.0; 2];
        layer_norm_into(&input, &weight, Some(&bias), 1e-5, &mut output);

        // Bias should be added
        assert!(output[0] > 9.0);
        assert!(output[1] > 19.0);
    }

    #[test]
    fn test_layer_norm_into_zeros() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![999.0; 4];
        layer_norm_into(&input, &weight, None, 1e-5, &mut output);

        // Should handle zeros gracefully
        for &val in &output {
            assert!(val.is_finite());
            assert!(val.abs() < 1e-2);
        }
    }

    // =========================================================================
    // Additional tests for gelu activation
    // =========================================================================

    #[test]
    fn test_gelu_negative() {
        let mut input = vec![-1.0];
        gelu(&mut input);
        // GELU(-1) ≈ -0.1587
        assert!((input[0] - (-0.1587)).abs() < 0.01);
    }

    #[test]
    fn test_gelu_large_positive() {
        let mut input = vec![10.0];
        gelu(&mut input);
        // GELU(x) → x for large positive x
        assert!((input[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_large_negative() {
        let mut input = vec![-10.0];
        gelu(&mut input);
        // GELU(x) → 0 for large negative x
        assert!(input[0].abs() < 0.01);
    }

    #[test]
    fn test_gelu_batch() {
        let mut input = vec![0.0, 1.0, -1.0, 2.0];
        gelu(&mut input);
        assert!((input[0] - 0.0).abs() < 0.01);
        assert!((input[1] - 0.8413).abs() < 0.01);
        assert!((input[2] - (-0.1587)).abs() < 0.01);
        assert!(input[3] > 1.9); // GELU(2) ≈ 1.96
    }

    // =========================================================================
    // Additional tests for silu activation
    // =========================================================================

    #[test]
    fn test_silu_negative() {
        let mut input = vec![-1.0];
        silu(&mut input);
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.2689
        assert!((input[0] - (-0.2689)).abs() < 0.01);
    }

    #[test]
    fn test_silu_large_positive() {
        let mut input = vec![10.0];
        silu(&mut input);
        // SiLU(x) → x for large positive x (sigmoid → 1)
        assert!((input[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_large_negative() {
        let mut input = vec![-10.0];
        silu(&mut input);
        // SiLU(x) → 0 for large negative x (sigmoid → 0)
        assert!(input[0].abs() < 0.001);
    }

    #[test]
    fn test_silu_batch() {
        let mut input = vec![0.0, 1.0, -1.0, 2.0];
        silu(&mut input);
        assert!((input[0] - 0.0).abs() < 0.01);
        assert!((input[1] - 0.7311).abs() < 0.01);
        assert!((input[2] - (-0.2689)).abs() < 0.01);
        assert!(input[3] > 1.7); // SiLU(2) ≈ 1.76
    }

    // =========================================================================
    // Additional tests for argmax
    // =========================================================================

    #[test]
    fn test_argmax_single_element() {
        let logits = vec![42.0];
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn test_argmax_all_same() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        // First occurrence wins
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn test_argmax_infinity() {
        let logits = vec![1.0, f32::INFINITY, 3.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_argmax_with_nan() {
        // NaN should not be selected as max
        let logits = vec![f32::NAN, 1.0, 2.0];
        // NaN comparisons are false, so 2.0 at index 2 should win
        assert_eq!(argmax(&logits), 2);
    }

    // =========================================================================
    // Additional tests for softmax
    // =========================================================================

    #[test]
    fn test_softmax_single_element() {
        let mut logits = vec![5.0];
        softmax(&mut logits);
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_large_values() {
        // Should not overflow due to numerical stability (subtract max)
        let mut logits = vec![1000.0, 1001.0, 1002.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Largest should have highest probability
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut logits = vec![-1.0, -2.0, -3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Least negative should have highest probability
        assert!(logits[0] > logits[1]);
        assert!(logits[1] > logits[2]);
    }

    #[test]
    fn test_softmax_dominance() {
        // One very large value should dominate
        let mut logits = vec![0.0, 0.0, 100.0];
        softmax(&mut logits);
        assert!(logits[2] > 0.99);
        assert!(logits[0] < 0.01);
        assert!(logits[1] < 0.01);
    }

    // =========================================================================
    // Additional tests for add_bias
    // =========================================================================

    #[test]
    fn test_add_bias_single_element() {
        let mut output = vec![1.0];
        let bias = vec![0.5];
        add_bias(&mut output, &bias);
        assert!((output[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_add_bias_multiple_sequences() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 sequences of dim 2
        let bias = vec![0.1, 0.2];
        add_bias(&mut output, &bias);
        assert!((output[0] - 1.1).abs() < 1e-6);
        assert!((output[1] - 2.2).abs() < 1e-6);
        assert!((output[2] - 3.1).abs() < 1e-6);
        assert!((output[3] - 4.2).abs() < 1e-6);
        assert!((output[4] - 5.1).abs() < 1e-6);
        assert!((output[5] - 6.2).abs() < 1e-6);
    }

    #[test]
    fn test_add_bias_negative() {
        let mut output = vec![1.0, 2.0];
        let bias = vec![-0.5, -1.0];
        add_bias(&mut output, &bias);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // Additional tests for rms_norm
    // =========================================================================

    #[test]
    fn test_rms_norm_multiple_sequences() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 sequences of dim 4
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms_norm(&input, &weight, 1e-5);

        assert_eq!(output.len(), 8);
        // Each sequence should be independently normalized
        let seq1_sum_sq: f32 = output[0..4].iter().map(|x| x * x).sum();
        let seq2_sum_sq: f32 = output[4..8].iter().map(|x| x * x).sum();
        let rms1 = (seq1_sum_sq / 4.0).sqrt();
        let rms2 = (seq2_sum_sq / 4.0).sqrt();
        assert!((rms1 - 1.0).abs() < 0.1);
        assert!((rms2 - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_rms_norm_zeros() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms_norm(&input, &weight, 1e-5);

        // Output should be near zero
        for &val in &output {
            assert!(val.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_preserves_sign() {
        let input = vec![-3.0, 2.0, -1.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms_norm(&input, &weight, 1e-5);

        assert!(output[0] < 0.0);
        assert!(output[1] > 0.0);
        assert!(output[2] < 0.0);
        assert!(output[3] > 0.0);
    }

    // =========================================================================
    // Additional tests for layer_norm
    // =========================================================================

    #[test]
    fn test_layer_norm_multiple_sequences() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 sequences of dim 3
        let weight = vec![1.0, 1.0, 1.0];
        let output = layer_norm(&input, &weight, None, 1e-5);

        assert_eq!(output.len(), 6);
        // Each sequence should be independently normalized
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_layer_norm_preserves_sign() {
        let input = vec![-3.0, 2.0, -1.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = layer_norm(&input, &weight, None, 1e-5);

        assert!(output[0] < 0.0);
        assert!(output[1] > 0.0);
        assert!(output[2] < 0.0);
        assert!(output[3] > 0.0);
    }

    #[test]
    fn test_layer_norm_weight_scaling() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![2.0, 0.5, 1.0, 3.0];
        let output = layer_norm(&input, &weight, None, 1e-5);

        // Output ratio should match weight ratio
        assert!((output[0] / output[2] - 2.0).abs() < 0.1);
        assert!((output[1] / output[2] - 0.5).abs() < 0.1);
        assert!((output[3] / output[2] - 3.0).abs() < 0.1);
    }
}
