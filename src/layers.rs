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

/// Apply GELU activation function
///
/// GELU (Gaussian Error Linear Unit): `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// Used in transformer models like BERT, GPT-2, GPT-3.
///
/// # Arguments
///
/// * `input` - Input tensor
///
/// # Returns
///
/// Tensor with GELU applied element-wise
///
/// # Errors
///
/// Returns error if input is empty
///
/// # Examples
///
/// ```rust,ignore
/// let input = Tensor::from_vec(vec![3], vec![-1.0, 0.0, 1.0])?;
/// let output = gelu(&input)?;
/// ```
pub fn gelu(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = input.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Cannot apply GELU to empty tensor".to_string(),
        });
    }

    // Apply GELU activation using approximation
    let output: Vec<f32> = data
        .iter()
        .map(|&x| {
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            let sqrt_2_over_pi = 0.797_884_6; // sqrt(2/π)
            let c = 0.044_715;
            let inner = sqrt_2_over_pi * (x + c * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect();

    Tensor::from_vec(input.shape().to_vec(), output)
}

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

/// Linear transformation layer
///
/// Applies linear transformation: `y = x * W + b`
/// where W is weight matrix and b is bias vector.
///
/// # References
///
/// Standard fully-connected layer used in neural networks.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Weight matrix `[in_features, out_features]`
    weight: Vec<f32>,
    /// Bias vector `[out_features]`
    bias: Vec<f32>,
}

impl Linear {
    /// Create a new linear layer
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Errors
    ///
    /// Returns error if either dimension is zero
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let linear = Linear::new(512, 2048)?;
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "in_features and out_features must be > 0".to_string(),
            });
        }

        // Initialize weights to zero (will be loaded from model)
        let weight = vec![0.0; in_features * out_features];
        // Initialize bias to zero
        let bias = vec![0.0; out_features];

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
        })
    }

    /// Forward pass through linear layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape `[batch, in_features]` or `[in_features]`
    ///
    /// # Returns
    ///
    /// Output tensor with shape `[batch, out_features]` or `[out_features]`
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input last dimension doesn't match `in_features`
    /// - Input is empty
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let input = Tensor::from_vec(vec![2, 512], data)?;
    /// let output = linear.forward(&input)?;
    /// assert_eq!(output.shape(), &[2, 2048]);
    /// ```
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.in_features {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match in_features {}",
                    last_dim, self.in_features
                ),
            });
        }

        let data = input.data();
        let total_size = data.len();
        let num_rows = total_size / self.in_features;

        let mut output = Vec::with_capacity(num_rows * self.out_features);

        // For each input row, compute: output = input * weight + bias
        for row_idx in 0..num_rows {
            let input_start = row_idx * self.in_features;
            let input_row = &data[input_start..input_start + self.in_features];

            // Matrix-vector multiplication: output[j] = sum(input[i] * weight[i][j]) + bias[j]
            for j in 0..self.out_features {
                let mut sum = self.bias[j];
                for (i, &input_val) in input_row.iter().enumerate() {
                    sum += input_val * self.weight[i * self.out_features + j];
                }
                output.push(sum);
            }
        }

        // Construct output shape
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        Tensor::from_vec(output_shape, output)
    }

    /// Get input features
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get weight matrix (for loading from model)
    #[must_use]
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// Get bias vector (for loading from model)
    #[must_use]
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
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

    // Linear layer tests

    #[test]
    fn test_linear_creation() {
        let linear = Linear::new(128, 256).unwrap();
        assert_eq!(linear.in_features(), 128);
        assert_eq!(linear.out_features(), 256);
    }

    #[test]
    fn test_linear_zero_dimensions_error() {
        let result = Linear::new(0, 256);
        assert!(result.is_err());

        let result = Linear::new(128, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_forward_simple() {
        // Simple test: 2 in_features, 3 out_features
        let mut linear = Linear::new(2, 3).unwrap();

        // Set identity-like weights for testing
        // weight[i][j] = 1.0 if i==j, 0.0 otherwise (extended for different dimensions)
        linear.weight_mut()[0] = 1.0; // weight[0][0]
        linear.weight_mut()[1] = 0.0; // weight[0][1]
        linear.weight_mut()[2] = 0.0; // weight[0][2]
        linear.weight_mut()[3] = 0.0; // weight[1][0]
        linear.weight_mut()[4] = 1.0; // weight[1][1]
        linear.weight_mut()[5] = 0.0; // weight[1][2]

        // Bias: all 0.5
        linear.bias_mut()[0] = 0.5;
        linear.bias_mut()[1] = 0.5;
        linear.bias_mut()[2] = 0.5;

        // Input: [2.0, 3.0]
        let input = Tensor::from_vec(vec![2], vec![2.0, 3.0]).unwrap();
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.shape(), &[3]);
        let output_data = output.data();

        // Expected: [2.0*1.0 + 3.0*0.0 + 0.5, 2.0*0.0 + 3.0*1.0 + 0.5, 2.0*0.0 + 3.0*0.0 + 0.5]
        //         = [2.5, 3.5, 0.5]
        assert!((output_data[0] - 2.5).abs() < 1e-5);
        assert!((output_data[1] - 3.5).abs() < 1e-5);
        assert!((output_data[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_linear_forward_batched() {
        // Test with batch dimension: [2, 2] -> [2, 3]
        let mut linear = Linear::new(2, 3).unwrap();

        // Simple weights: all 1.0
        for i in 0..6 {
            linear.weight_mut()[i] = 1.0;
        }
        // Bias: all 0.0
        for i in 0..3 {
            linear.bias_mut()[i] = 0.0;
        }

        // Input: [[1.0, 2.0], [3.0, 4.0]]
        let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        let output_data = output.data();

        // First row: [1.0, 2.0] * all-ones weight + zero bias = [3.0, 3.0, 3.0]
        assert!((output_data[0] - 3.0).abs() < 1e-5);
        assert!((output_data[1] - 3.0).abs() < 1e-5);
        assert!((output_data[2] - 3.0).abs() < 1e-5);

        // Second row: [3.0, 4.0] * all-ones weight + zero bias = [7.0, 7.0, 7.0]
        assert!((output_data[3] - 7.0).abs() < 1e-5);
        assert!((output_data[4] - 7.0).abs() < 1e-5);
        assert!((output_data[5] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_shape_mismatch_error() {
        let linear = Linear::new(3, 2).unwrap();
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap(); // Wrong size (2 vs 3)
        let result = linear.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_weight_bias_mut() {
        let mut linear = Linear::new(2, 3).unwrap();

        // Modify weights
        linear.weight_mut()[0] = 42.0;
        assert_eq!(linear.weight_mut()[0], 42.0);

        // Modify bias
        linear.bias_mut()[0] = 7.0;
        assert_eq!(linear.bias_mut()[0], 7.0);
    }

    // GELU activation tests

    #[test]
    fn test_gelu_zero() {
        let input = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
        let output = gelu(&input).unwrap();
        // GELU(0) = 0
        assert!((output.data()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let input = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
        let output = gelu(&input).unwrap();
        // GELU(1) ≈ 0.841 (approximately x for large positive x)
        assert!(output.data()[0] > 0.8);
        assert!(output.data()[0] < 0.9);
    }

    #[test]
    fn test_gelu_negative() {
        let input = Tensor::from_vec(vec![1], vec![-1.0]).unwrap();
        let output = gelu(&input).unwrap();
        // GELU(-1) is small negative (smooth near zero)
        assert!(output.data()[0] < 0.0);
        assert!(output.data()[0] > -0.2);
    }

    #[test]
    fn test_gelu_batched() {
        let input = Tensor::from_vec(vec![2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
        let output = gelu(&input).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.data().len(), 6);

        // GELU(0) = 0
        assert!((output.data()[2] - 0.0).abs() < 1e-6);
        // Positive values should be positive
        assert!(output.data()[3] > 0.0);
        assert!(output.data()[4] > 0.0);
        assert!(output.data()[5] > 0.0);
    }

    #[test]
    fn test_gelu_preserves_shape() {
        // Test that GELU preserves tensor shape
        let input = Tensor::from_vec(vec![2, 3, 4], vec![0.5; 24]).unwrap();
        let output = gelu(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
        assert_eq!(output.data().len(), 24);
    }
}
