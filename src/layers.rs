//! Neural network layers for transformer models
//!
//! Implements core building blocks for transformer architectures:
//! - Layer normalization
//! - Multi-head attention
//! - Feed-forward networks
//! - `RoPE` position embeddings
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::layers::LayerNorm;
//!
//! let layer_norm = LayerNorm::new(512, 1e-5)?;
//! let normalized = layer_norm.forward(&input)?;
//! ```

use crate::error::{RealizarError, Result};
use crate::tensor::Tensor;

/// Layer normalization
///
/// Normalizes activations across the feature dimension using:
/// ```text
/// y = (x - mean(x)) / sqrt(variance(x) + eps) * gamma + beta
/// ```
///
/// # References
///
/// Layer Normalization: <https://arxiv.org/abs/1607.06450>
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Normalized shape (feature dimension)
    normalized_shape: usize,
    /// Epsilon for numerical stability
    eps: f32,
    /// Scale parameter (gamma)
    weight: Vec<f32>,
    /// Shift parameter (beta)
    bias: Vec<f32>,
}

impl LayerNorm {
    /// Create a new layer normalization layer
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Size of the feature dimension to normalize
    /// * `eps` - Small constant for numerical stability (default: `1e-5`)
    ///
    /// # Errors
    ///
    /// Returns error if `normalized_shape` is zero
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let layer_norm = LayerNorm::new(512, 1e-5)?;
    /// ```
    pub fn new(normalized_shape: usize, eps: f32) -> Result<Self> {
        if normalized_shape == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "normalized_shape must be > 0".to_string(),
            });
        }

        // Initialize weight (gamma) to 1.0
        let weight = vec![1.0; normalized_shape];
        // Initialize bias (beta) to 0.0
        let bias = vec![0.0; normalized_shape];

        Ok(Self {
            normalized_shape,
            eps,
            weight,
            bias,
        })
    }

    /// Forward pass through layer normalization
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape `[..., normalized_shape]`
    ///
    /// # Returns
    ///
    /// Normalized tensor with same shape as input
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input is empty
    /// - Last dimension doesn't match `normalized_shape`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let input = Tensor::from_vec(vec![2, 512], data)?;
    /// let output = layer_norm.forward(&input)?;
    /// assert_eq!(output.shape(), &[2, 512]);
    /// ```
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.normalized_shape {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match normalized_shape {}",
                    last_dim, self.normalized_shape
                ),
            });
        }

        let data = input.data();
        let total_size = data.len();
        let num_groups = total_size / self.normalized_shape;

        let mut output = Vec::with_capacity(total_size);

        for group_idx in 0..num_groups {
            let start = group_idx * self.normalized_shape;
            let end = start + self.normalized_shape;
            let group = &data[start..end];

            // Compute mean
            #[allow(clippy::cast_precision_loss)]
            let mean: f32 = group.iter().sum::<f32>() / self.normalized_shape as f32;

            // Compute variance
            #[allow(clippy::cast_precision_loss)]
            let variance: f32 = group
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f32>()
                / self.normalized_shape as f32;

            // Normalize and apply affine transformation
            for (i, &x) in group.iter().enumerate() {
                let normalized = (x - mean) / (variance + self.eps).sqrt();
                let transformed = normalized * self.weight[i] + self.bias[i];
                output.push(transformed);
            }
        }

        Tensor::from_vec(shape.to_vec(), output)
    }

    /// Get the normalized shape
    #[must_use]
    pub fn normalized_shape(&self) -> usize {
        self.normalized_shape
    }

    /// Get epsilon value
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let layer_norm = LayerNorm::new(512, 1e-5).unwrap();
        assert_eq!(layer_norm.normalized_shape(), 512);
        assert_eq!(layer_norm.eps(), 1e-5);
    }

    #[test]
    fn test_layer_norm_zero_shape_error() {
        let result = LayerNorm::new(0, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_forward_simple() {
        // Simple test with known values
        let layer_norm = LayerNorm::new(3, 1e-5).unwrap();

        // Input: [1.0, 2.0, 3.0]
        // Mean: 2.0
        // Variance: ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        // Std: sqrt(2/3 + 1e-5) ≈ 0.8165
        // Normalized: [(1-2)/0.8165, (2-2)/0.8165, (3-2)/0.8165]
        //           ≈ [-1.2247, 0.0, 1.2247]
        let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let output = layer_norm.forward(&input).unwrap();

        let output_data = output.data();
        assert_eq!(output_data.len(), 3);

        // Check that mean is approximately 0
        let mean: f32 = output_data.iter().sum::<f32>() / 3.0;
        assert!((mean - 0.0).abs() < 1e-5);

        // Check that variance is approximately 1
        let variance: f32 = output_data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>()
            / 3.0;
        assert!((variance - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_forward_batched() {
        // Test with batch dimension
        let layer_norm = LayerNorm::new(2, 1e-5).unwrap();

        // Input: [[1.0, 3.0], [2.0, 4.0]]
        let input = Tensor::from_vec(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();
        let output = layer_norm.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 2]);

        let output_data = output.data();
        assert_eq!(output_data.len(), 4);

        // First group [1.0, 3.0]: mean=2.0, normalized should have mean≈0, var≈1
        let group1_mean = (output_data[0] + output_data[1]) / 2.0;
        assert!((group1_mean - 0.0).abs() < 1e-5);

        // Second group [2.0, 4.0]: mean=3.0, normalized should have mean≈0, var≈1
        let group2_mean = (output_data[2] + output_data[3]) / 2.0;
        assert!((group2_mean - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_empty_shape_handling() {
        // LayerNorm should handle validation properly
        // Since Tensor itself doesn't allow empty shapes, we test
        // that the normalized_shape validation works
        let result = LayerNorm::new(0, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_shape_mismatch_error() {
        let layer_norm = LayerNorm::new(3, 1e-5).unwrap();
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap(); // Wrong size
        let result = layer_norm.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_zero_variance() {
        // Test with constant input (zero variance)
        let layer_norm = LayerNorm::new(3, 1e-5).unwrap();
        let input = Tensor::from_vec(vec![3], vec![2.0, 2.0, 2.0]).unwrap();
        let output = layer_norm.forward(&input).unwrap();

        // With zero variance, normalized values should be close to 0
        // (since eps prevents division by zero)
        let output_data = output.data();
        for &val in output_data {
            assert!(val.abs() < 1e-2); // Should be near 0
        }
    }
}
