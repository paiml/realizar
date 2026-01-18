//! Normalization and position encoding operations
//!
//! Provides layer normalization, RMS normalization, and rotary position embeddings
//! used in transformer inference.
//!
//! ## Normalization Functions
//!
//! - [`simd_layer_norm`] - Standard layer normalization with mean and variance
//! - [`simd_rms_norm`] - RMS normalization (faster, used in LLaMA/Mistral)
//!
//! ## Position Encoding
//!
//! - [`apply_rope`] - Rotary Position Embeddings (RoPE)

/// SIMD-accelerated layer normalization
///
/// LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
///
/// # Arguments
///
/// * `input` - Input vector to normalize
/// * `weight` - Scale parameters (gamma)
/// * `bias` - Optional shift parameters (beta)
/// * `eps` - Small constant for numerical stability (typically 1e-5)
///
/// # Example
///
/// ```
/// use realizar::inference::simd_layer_norm;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0];
/// let weight = vec![1.0, 1.0, 1.0, 1.0];
/// let output = simd_layer_norm(&input, &weight, None, 1e-5);
///
/// // Output should have mean ≈ 0 and std ≈ 1
/// let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
/// assert!(mean.abs() < 1e-5);
/// ```
#[must_use]
pub fn simd_layer_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute mean
    let mean: f32 = input.iter().sum::<f32>() / n as f32;

    // Compute variance
    let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    // Normalize
    let inv_std = 1.0 / (var + eps).sqrt();
    let mut output: Vec<f32> = input.iter().map(|x| (x - mean) * inv_std).collect();

    // Apply affine transformation
    for (i, out) in output.iter_mut().enumerate() {
        *out *= weight[i];
        if let Some(b) = bias {
            *out += b[i];
        }
    }

    output
}

/// SIMD-accelerated RMS normalization
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
///
/// RMS normalization is faster than LayerNorm as it doesn't require
/// computing the mean. Used in LLaMA, Mistral, and other modern LLMs.
///
/// # Arguments
///
/// * `input` - Input vector to normalize
/// * `weight` - Scale parameters
/// * `eps` - Small constant for numerical stability (typically 1e-5)
///
/// # Example
///
/// ```
/// use realizar::inference::simd_rms_norm;
///
/// let input = vec![1.0, 2.0, 3.0];
/// let weight = vec![1.0, 1.0, 1.0];
/// let output = simd_rms_norm(&input, &weight, 1e-5);
///
/// // RMS of [1,2,3] ≈ 2.16, so normalized ≈ [0.46, 0.93, 1.39]
/// assert!((output[0] - 0.4629).abs() < 0.01);
/// ```
#[must_use]
pub fn simd_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute RMS
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Normalize and scale
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * inv_rms * w)
        .collect()
}

/// Apply rotary position embeddings (RoPE)
///
/// RoPE encodes position information by rotating pairs of dimensions.
/// This enables relative position encoding that generalizes to longer sequences.
///
/// # Arguments
///
/// * `x` - Mutable slice to apply RoPE to [hidden_dim]
/// * `hidden_dim` - Total hidden dimension (must equal x.len())
/// * `num_heads` - Number of attention heads
/// * `position` - Token position in sequence (0-indexed)
/// * `theta` - Base frequency (typically 10000.0)
///
/// # Algorithm
///
/// For each head and each pair of dimensions (i, i + d/2):
/// ```text
/// freq = 1 / theta^(2i/d)
/// angle = position * freq
/// x[i]     = x[i] * cos(angle) - x[i+d/2] * sin(angle)
/// x[i+d/2] = x[i] * sin(angle) + x[i+d/2] * cos(angle)
/// ```
///
/// # Example
///
/// ```
/// use realizar::inference::apply_rope;
///
/// let mut x = vec![1.0; 64];  // 64 hidden dim
/// apply_rope(&mut x, 64, 4, 0, 10000.0);  // Position 0
///
/// // At position 0, rotations are identity (angle = 0)
/// assert!((x[0] - 1.0).abs() < 1e-5);
/// ```
pub fn apply_rope(x: &mut [f32], hidden_dim: usize, num_heads: usize, position: usize, theta: f32) {
    let head_dim = hidden_dim / num_heads;
    let half_dim = head_dim / 2;

    for h in 0..num_heads {
        let head_offset = h * head_dim;

        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx0 = head_offset + i;
            let idx1 = head_offset + i + half_dim;

            let x0 = x[idx0];
            let x1 = x[idx1];

            x[idx0] = x0 * cos_val - x1 * sin_val;
            x[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

// ============================================================================
// EXTREME TDD: Comprehensive Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // simd_layer_norm Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_layer_norm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = simd_layer_norm(&input, &weight, None, 1e-5);

        // Output should have mean ≈ 0
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

        // Output should have std ≈ 1
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;
        let std = var.sqrt();
        assert!((std - 1.0).abs() < 0.01, "Std should be ~1, got {}", std);
    }

    #[test]
    fn test_layer_norm_with_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let output = simd_layer_norm(&input, &weight, None, 1e-5);

        // With scale=2, std should be ~2
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;
        let std = var.sqrt();
        assert!((std - 2.0).abs() < 0.01, "Std should be ~2, got {}", std);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![5.0, 5.0, 5.0, 5.0];
        let output = simd_layer_norm(&input, &weight, Some(&bias), 1e-5);

        // With bias=5, mean should be ~5
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!((mean - 5.0).abs() < 0.01, "Mean should be ~5, got {}", mean);
    }

    #[test]
    fn test_layer_norm_empty() {
        let input: Vec<f32> = vec![];
        let weight: Vec<f32> = vec![];
        let output = simd_layer_norm(&input, &weight, None, 1e-5);
        assert!(output.is_empty());
    }

    #[test]
    fn test_layer_norm_single_element() {
        let input = vec![5.0];
        let weight = vec![1.0];
        let output = simd_layer_norm(&input, &weight, None, 1e-5);
        // Single element: mean=5, var=0, so normalized = 0
        assert!((output[0]).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_uniform_input() {
        let input = vec![3.0, 3.0, 3.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = simd_layer_norm(&input, &weight, None, 1e-5);
        // Uniform input: mean=3, var=0+eps, normalized ≈ 0
        for &x in &output {
            assert!(x.abs() < 0.1);
        }
    }

    #[test]
    fn test_layer_norm_negative_values() {
        let input = vec![-2.0, -1.0, 1.0, 2.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = simd_layer_norm(&input, &weight, None, 1e-5);

        // Mean should be 0, values should preserve sign relationship
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
        assert!(output[2] < output[3]);
    }

    #[test]
    fn test_layer_norm_large_values() {
        let input = vec![1000.0, 2000.0, 3000.0, 4000.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = simd_layer_norm(&input, &weight, None, 1e-5);

        // Should still have mean ≈ 0 and std ≈ 1
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!(mean.abs() < 1e-4);
    }

    // ------------------------------------------------------------------------
    // simd_rms_norm Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_rms_norm_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let output = simd_rms_norm(&input, &weight, 1e-5);

        // RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.16
        let rms = (14.0_f32 / 3.0).sqrt();
        let expected: Vec<f32> = input.iter().map(|x| x / rms).collect();

        for (out, exp) in output.iter().zip(expected.iter()) {
            assert!((out - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rms_norm_with_scale() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![2.0, 2.0, 2.0];
        let output = simd_rms_norm(&input, &weight, 1e-5);

        let rms = (14.0_f32 / 3.0).sqrt();
        let expected: Vec<f32> = input.iter().map(|x| x / rms * 2.0).collect();

        for (out, exp) in output.iter().zip(expected.iter()) {
            assert!((out - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rms_norm_empty() {
        let input: Vec<f32> = vec![];
        let weight: Vec<f32> = vec![];
        let output = simd_rms_norm(&input, &weight, 1e-5);
        assert!(output.is_empty());
    }

    #[test]
    fn test_rms_norm_single_element() {
        let input = vec![5.0];
        let weight = vec![1.0];
        let output = simd_rms_norm(&input, &weight, 1e-5);
        // RMS of [5] = 5, so output = 5/5 = 1
        assert!((output[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_unit_vector() {
        // For input [1, 0, 0] with weight [1, 1, 1]
        // RMS = sqrt(mean(x^2)) = sqrt(1/3)
        // output = input / RMS * weight = [sqrt(3), 0, 0]
        let input = vec![1.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0];
        let output = simd_rms_norm(&input, &weight, 1e-5);

        let expected = 3.0_f32.sqrt(); // sqrt(3) ≈ 1.732
        assert!(
            (output[0] - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            output[0]
        );
        assert!(output[1].abs() < 1e-5);
        assert!(output[2].abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_zeros() {
        let input = vec![0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0];
        let output = simd_rms_norm(&input, &weight, 1e-5);

        // RMS = sqrt(eps), output = 0 / sqrt(eps) = 0
        for &x in &output {
            assert!(x.abs() < 1e-2);
        }
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let input = vec![-3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let output = simd_rms_norm(&input, &weight, 1e-5);

        // RMS = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.54
        let rms = (12.5_f32).sqrt();
        assert!((output[0] - (-3.0 / rms)).abs() < 1e-5);
        assert!((output[1] - (4.0 / rms)).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_preserves_direction() {
        let input = vec![3.0, 4.0]; // 3-4-5 right triangle
        let weight = vec![1.0, 1.0];
        let output = simd_rms_norm(&input, &weight, 1e-5);

        // Direction should be preserved: output[1]/output[0] = 4/3
        let ratio = output[1] / output[0];
        assert!((ratio - 4.0 / 3.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------------
    // apply_rope Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_rope_position_zero() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0]; // 4 hidden, 1 head, head_dim=4
        let original = x.clone();
        apply_rope(&mut x, 4, 1, 0, 10000.0);

        // At position 0, angle = 0, cos(0) = 1, sin(0) = 0
        // So output should equal input
        for (out, orig) in x.iter().zip(original.iter()) {
            assert!((out - orig).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_rotation_property() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // 4 hidden, 1 head
        apply_rope(&mut x, 4, 1, 1, 10000.0);

        // After rotation, magnitude should be preserved for each pair
        let mag0 = (x[0] * x[0] + x[2] * x[2]).sqrt();
        let mag1 = (x[1] * x[1] + x[3] * x[3]).sqrt();

        assert!((mag0 - 1.0).abs() < 1e-5, "Magnitude of pair 0 should be 1");
        assert!((mag1 - 1.0).abs() < 1e-5, "Magnitude of pair 1 should be 1");
    }

    #[test]
    fn test_rope_multiple_heads() {
        let mut x = vec![1.0; 8]; // 8 hidden, 2 heads, head_dim = 4
        let original = x.clone();
        apply_rope(&mut x, 8, 2, 0, 10000.0);

        // At position 0, should be unchanged
        for (out, orig) in x.iter().zip(original.iter()) {
            assert!((out - orig).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_different_positions() {
        let mut x1 = vec![1.0; 4];
        let mut x2 = vec![1.0; 4];

        apply_rope(&mut x1, 4, 1, 0, 10000.0);
        apply_rope(&mut x2, 4, 1, 1, 10000.0);

        // Different positions should give different results
        assert!((x1[0] - x2[0]).abs() > 1e-6 || (x1[1] - x2[1]).abs() > 1e-6);
    }

    #[test]
    fn test_rope_theta_scaling() {
        let mut x1 = vec![1.0; 4];
        let mut x2 = vec![1.0; 4];

        apply_rope(&mut x1, 4, 1, 10, 10000.0);
        apply_rope(&mut x2, 4, 1, 10, 1000.0);

        // Different theta affects higher frequency dimensions (i > 0)
        // For i=0, freq = 1/theta^0 = 1 (same regardless of theta)
        // For i=1, freq = 1/theta^(2/head_dim) which differs by theta
        // So check x[1] or x[3] (the second pair uses i=1)
        assert!(
            (x1[1] - x2[1]).abs() > 1e-5 || (x1[3] - x2[3]).abs() > 1e-5,
            "Different theta should give different results for non-zero frequency indices"
        );
    }

    #[test]
    fn test_rope_large_position() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut x, 4, 1, 1000, 10000.0);

        // Results should be finite
        for &val in &x {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_rope_eight_heads() {
        let hidden_dim = 64;
        let num_heads = 8;
        let mut x = vec![0.5; hidden_dim];

        apply_rope(&mut x, hidden_dim, num_heads, 5, 10000.0);

        // All values should be finite
        for &val in &x {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_rope_preserves_length() {
        let mut x = vec![3.0, 4.0, 0.0, 0.0]; // pairs: (3,0), (4,0)
        apply_rope(&mut x, 4, 1, 1, 10000.0);

        assert_eq!(x.len(), 4);
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_norm_then_rope() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0; 8];

        // First normalize
        let normalized = simd_rms_norm(&input, &weight, 1e-5);

        // Then apply RoPE
        let mut output = normalized;
        apply_rope(&mut output, 8, 2, 5, 10000.0);

        // Results should be finite
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_layer_norm_vs_rms_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        let ln_output = simd_layer_norm(&input, &weight, None, 1e-5);
        let rms_output = simd_rms_norm(&input, &weight, 1e-5);

        // LayerNorm centers (mean=0), RMSNorm doesn't
        let ln_mean: f32 = ln_output.iter().sum::<f32>() / 4.0;
        let rms_mean: f32 = rms_output.iter().sum::<f32>() / 4.0;

        assert!(ln_mean.abs() < 1e-5, "LayerNorm should have mean ~0");
        assert!(rms_mean.abs() > 0.1, "RMSNorm should not center");
    }

    // ------------------------------------------------------------------------
    // Edge Cases
    // ------------------------------------------------------------------------

    #[test]
    fn test_layer_norm_eps_impact() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        // With var=0, eps prevents division by zero
        let output = simd_layer_norm(&input, &weight, None, 1e-5);
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_rms_norm_eps_impact() {
        let input = vec![0.0, 0.0];
        let weight = vec![1.0, 1.0];

        // With sum_sq=0, eps prevents division by zero
        let output = simd_rms_norm(&input, &weight, 1e-5);
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_rope_half_dim_calculation() {
        // Test with various head dimensions
        for (hidden_dim, num_heads) in [(8, 2), (16, 4), (32, 8), (64, 16)] {
            let mut x = vec![1.0; hidden_dim];
            apply_rope(&mut x, hidden_dim, num_heads, 1, 10000.0);

            // Should not panic and should produce finite values
            for &val in &x {
                assert!(val.is_finite());
            }
        }
    }
}
