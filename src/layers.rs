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

/// Apply softmax activation function
///
/// Softmax: `y[i] = exp(x[i]) / sum(exp(x[j]))` for all j
///
/// Applies softmax normalization along the last dimension. Uses numerically stable
/// implementation with max subtraction to prevent overflow.
///
/// Used in attention mechanisms for probability distributions.
///
/// # Arguments
///
/// * `input` - Input tensor
///
/// # Returns
///
/// Tensor with softmax applied along last dimension (values sum to 1.0)
///
/// # Errors
///
/// Returns error if input is empty
///
/// # Examples
///
/// ```rust,ignore
/// let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0])?;
/// let output = softmax(&input)?;
/// // output sums to 1.0
/// ```
pub fn softmax(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = input.data();
    let shape = input.shape();

    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Cannot apply softmax to empty tensor".to_string(),
        });
    }

    if shape.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Cannot apply softmax to tensor with empty shape".to_string(),
        });
    }

    let last_dim = shape[shape.len() - 1];
    let num_groups = data.len() / last_dim;
    let mut output = Vec::with_capacity(data.len());

    // Apply softmax to each group (row) independently
    for group_idx in 0..num_groups {
        let start = group_idx * last_dim;
        let end = start + last_dim;
        let group = &data[start..end];

        // Find max for numerical stability
        let max_val = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = group.iter().map(|&x| (x - max_val).exp()).collect();

        // Sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();

        // Normalize to get probabilities
        for &exp_val in &exp_vals {
            output.push(exp_val / sum_exp);
        }
    }

    Tensor::from_vec(shape.to_vec(), output)
}

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

/// Feed-forward network (FFN)
///
/// Two-layer feed-forward network with GELU activation:
/// ```text
/// FFN(x) = Linear2(GELU(Linear1(x)))
/// ```
///
/// Typically used in transformer blocks with:
/// - `hidden_dim` = model dimension (e.g., 768, 512)
/// - `intermediate_dim` = expansion (typically 4x `hidden_dim`)
///
/// # References
///
/// Standard transformer FFN from "Attention is All You Need"
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// First linear layer (expansion)
    fc1: Linear,
    /// Second linear layer (projection)
    fc2: Linear,
    /// Hidden dimension
    hidden_dim: usize,
    /// Intermediate dimension
    intermediate_dim: usize,
}

impl FeedForward {
    /// Create a new feed-forward network
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Input/output dimension
    /// * `intermediate_dim` - Intermediate dimension (typically 4x `hidden_dim`)
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are zero
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let ffn = FeedForward::new(768, 3072)?; // GPT-2 style (4x expansion)
    /// ```
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Result<Self> {
        let fc1 = Linear::new(hidden_dim, intermediate_dim)?;
        let fc2 = Linear::new(intermediate_dim, hidden_dim)?;

        Ok(Self {
            fc1,
            fc2,
            hidden_dim,
            intermediate_dim,
        })
    }

    /// Forward pass through feed-forward network
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape `[..., hidden_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor with shape `[..., hidden_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if input shape doesn't match `hidden_dim`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let input = Tensor::from_vec(vec![2, 768], data)?;
    /// let output = ffn.forward(&input)?;
    /// assert_eq!(output.shape(), &[2, 768]);
    /// ```
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // fc1: [hidden_dim] -> [intermediate_dim]
        let hidden = self.fc1.forward(input)?;

        // GELU activation
        let activated = gelu(&hidden)?;

        // fc2: [intermediate_dim] -> [hidden_dim]
        self.fc2.forward(&activated)
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get intermediate dimension
    #[must_use]
    pub fn intermediate_dim(&self) -> usize {
        self.intermediate_dim
    }

    /// Get mutable reference to first linear layer (for loading weights)
    #[must_use]
    pub fn fc1_mut(&mut self) -> &mut Linear {
        &mut self.fc1
    }

    /// Get mutable reference to second linear layer (for loading weights)
    #[must_use]
    pub fn fc2_mut(&mut self) -> &mut Linear {
        &mut self.fc2
    }
}

/// Scaled dot-product attention
///
/// Computes attention as:
/// ```text
/// Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
/// ```
///
/// This is a building block for multi-head attention.
///
/// # References
///
/// "Attention is All You Need" - Vaswani et al., 2017
#[derive(Debug, Clone)]
pub struct Attention {
    /// Head dimension (`d_k` = `d_model` / `num_heads`)
    head_dim: usize,
    /// Scale factor: 1 / `sqrt(head_dim)`
    scale: f32,
}

impl Attention {
    /// Create a new attention layer
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head
    ///
    /// # Errors
    ///
    /// Returns error if `head_dim` is zero
    pub fn new(head_dim: usize) -> Result<Self> {
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self { head_dim, scale })
    }

    /// Compute scaled dot-product attention
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor `[seq_len, head_dim]`
    /// * `key` - Key tensor `[seq_len, head_dim]`
    /// * `value` - Value tensor `[seq_len, head_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if shapes don't match
    pub fn forward(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
    ) -> Result<Tensor<f32>> {
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        // Validate shapes
        if q_shape.is_empty() || k_shape.is_empty() || v_shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Query, key, value tensors must have at least 1 dimension".to_string(),
            });
        }

        let q_last = q_shape[q_shape.len() - 1];
        let k_last = k_shape[k_shape.len() - 1];
        let v_last = v_shape[v_shape.len() - 1];

        if q_last != self.head_dim || k_last != self.head_dim || v_last != self.head_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Expected head_dim={}, got Q={}, K={}, V={}",
                    self.head_dim, q_last, k_last, v_last
                ),
            });
        }

        // Get sequence lengths
        let q_seq_len = if q_shape.len() > 1 { q_shape[0] } else { 1 };
        let k_seq_len = if k_shape.len() > 1 { k_shape[0] } else { 1 };
        let v_seq_len = if v_shape.len() > 1 { v_shape[0] } else { 1 };

        if k_seq_len != v_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!("Key seq_len {k_seq_len} != Value seq_len {v_seq_len}"),
            });
        }

        let q_data = query.data();
        let k_data = key.data();
        let v_data = value.data();

        // Compute attention scores: Q @ K.T
        // scores[i][j] = sum(Q[i][k] * K[j][k]) for all k
        let mut scores = Vec::with_capacity(q_seq_len * k_seq_len);
        for i in 0..q_seq_len {
            for j in 0..k_seq_len {
                let mut dot = 0.0;
                for k in 0..self.head_dim {
                    dot += q_data[i * self.head_dim + k] * k_data[j * self.head_dim + k];
                }
                scores.push(dot * self.scale);
            }
        }

        // Apply softmax to each row of scores
        let scores_tensor = Tensor::from_vec(vec![q_seq_len, k_seq_len], scores)?;
        let attn_weights = softmax(&scores_tensor)?;
        let attn_data = attn_weights.data();

        // Compute output: attn_weights @ V
        // output[i][k] = sum(attn_weights[i][j] * V[j][k]) for all j
        let mut output = Vec::with_capacity(q_seq_len * self.head_dim);
        for i in 0..q_seq_len {
            for k in 0..self.head_dim {
                let mut sum = 0.0;
                for j in 0..k_seq_len {
                    sum += attn_data[i * k_seq_len + j] * v_data[j * self.head_dim + k];
                }
                output.push(sum);
            }
        }

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get scale factor
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let layer_norm = LayerNorm::new(512, 1e-5).unwrap();
        assert_eq!(layer_norm.normalized_shape(), 512);
        assert!((layer_norm.eps() - 1e-5).abs() < 1e-10);
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
        assert!((linear.weight_mut()[0] - 42.0).abs() < 1e-6);

        // Modify bias
        linear.bias_mut()[0] = 7.0;
        assert!((linear.bias_mut()[0] - 7.0).abs() < 1e-6);
    }

    // Softmax activation tests

    #[test]
    fn test_softmax_simple() {
        // Simple softmax: [0, 0, 0] -> [1/3, 1/3, 1/3]
        let input = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
        let output = softmax(&input).unwrap();

        assert_eq!(output.shape(), &[3]);
        // All equal inputs -> equal probabilities
        assert!((output.data()[0] - 0.333_333).abs() < 1e-5);
        assert!((output.data()[1] - 0.333_333).abs() < 1e-5);
        assert!((output.data()[2] - 0.333_333).abs() < 1e-5);

        // Sum should be 1.0
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_probabilities_sum_to_one() {
        let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = softmax(&input).unwrap();

        // Sum should be exactly 1.0
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All values should be positive
        for &val in output.data() {
            assert!(val > 0.0);
            assert!(val < 1.0);
        }
    }

    #[test]
    fn test_softmax_max_dominates() {
        // When one value is much larger, it should dominate
        let input = Tensor::from_vec(vec![3], vec![0.0, 0.0, 10.0]).unwrap();
        let output = softmax(&input).unwrap();

        // Last element should be close to 1.0
        assert!(output.data()[2] > 0.999);
        // Others should be very small
        assert!(output.data()[0] < 0.001);
        assert!(output.data()[1] < 0.001);
    }

    #[test]
    fn test_softmax_batched() {
        // Batched: [[1, 2], [3, 4]]
        let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = softmax(&input).unwrap();

        assert_eq!(output.shape(), &[2, 2]);

        // Each row should sum to 1.0
        let row1_sum = output.data()[0] + output.data()[1];
        let row2_sum = output.data()[2] + output.data()[3];
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not overflow
        let input = Tensor::from_vec(vec![3], vec![1000.0, 1001.0, 1002.0]).unwrap();
        let output = softmax(&input).unwrap();

        // Should not be NaN or Inf
        for &val in output.data() {
            assert!(val.is_finite());
        }

        // Sum should still be 1.0
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_preserves_shape() {
        let input = Tensor::from_vec(vec![2, 3, 4], vec![1.0; 24]).unwrap();
        let output = softmax(&input).unwrap();

        assert_eq!(output.shape(), &[2, 3, 4]);
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

    // FeedForward (FFN) tests

    #[test]
    fn test_ffn_creation() {
        let ffn = FeedForward::new(512, 2048).unwrap();
        assert_eq!(ffn.hidden_dim(), 512);
        assert_eq!(ffn.intermediate_dim(), 2048);
    }

    #[test]
    fn test_ffn_zero_dimensions_error() {
        let result = FeedForward::new(0, 2048);
        assert!(result.is_err());

        let result = FeedForward::new(512, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffn_forward_shape() {
        // Test that FFN preserves hidden_dim
        let ffn = FeedForward::new(4, 16).unwrap(); // Small sizes for testing
        let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).unwrap();
        let output = ffn.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    fn test_ffn_forward_computation() {
        // Test FFN with known weights
        let mut ffn = FeedForward::new(2, 4).unwrap();

        // Set fc1 weights to identity-like (for simplicity)
        for i in 0..8 {
            ffn.fc1_mut().weight_mut()[i] = 0.1;
        }
        for i in 0..4 {
            ffn.fc1_mut().bias_mut()[i] = 0.0;
        }

        // Set fc2 weights
        for i in 0..8 {
            ffn.fc2_mut().weight_mut()[i] = 0.1;
        }
        for i in 0..2 {
            ffn.fc2_mut().bias_mut()[i] = 0.0;
        }

        // Input: [1.0, 2.0]
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
        let output = ffn.forward(&input).unwrap();

        // Output should be valid (not NaN, not Inf)
        assert_eq!(output.shape(), &[2]);
        assert!(output.data()[0].is_finite());
        assert!(output.data()[1].is_finite());
    }

    #[test]
    fn test_ffn_batched() {
        let ffn = FeedForward::new(3, 12).unwrap();

        // Batched input: [2, 3]
        let input = Tensor::from_vec(vec![2, 3], vec![0.5; 6]).unwrap();
        let output = ffn.forward(&input).unwrap();

        // Output shape should match input
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.data().len(), 6);
    }

    #[test]
    fn test_ffn_weight_access() {
        let mut ffn = FeedForward::new(2, 4).unwrap();

        // Modify fc1 weights
        ffn.fc1_mut().weight_mut()[0] = 42.0;
        assert!((ffn.fc1_mut().weight_mut()[0] - 42.0).abs() < 1e-6);

        // Modify fc2 bias
        ffn.fc2_mut().bias_mut()[0] = 7.0;
        assert!((ffn.fc2_mut().bias_mut()[0] - 7.0).abs() < 1e-6);
    }

    // Attention tests

    #[test]
    fn test_attention_creation() {
        let attn = Attention::new(64).unwrap();
        assert_eq!(attn.head_dim(), 64);
        // scale = 1 / sqrt(64) = 1/8 = 0.125
        assert!((attn.scale() - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_attention_zero_head_dim_error() {
        let result = Attention::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_forward_shape() {
        let attn = Attention::new(4).unwrap();

        // Q, K, V all have shape [3, 4] (seq_len=3, head_dim=4)
        let q = Tensor::from_vec(vec![3, 4], vec![0.1; 12]).unwrap();
        let k = Tensor::from_vec(vec![3, 4], vec![0.2; 12]).unwrap();
        let v = Tensor::from_vec(vec![3, 4], vec![0.3; 12]).unwrap();

        let output = attn.forward(&q, &k, &v).unwrap();

        // Output should have shape [3, 4]
        assert_eq!(output.shape(), &[3, 4]);
        assert_eq!(output.data().len(), 12);
    }

    #[test]
    fn test_attention_forward_computation() {
        let attn = Attention::new(2).unwrap();

        // Simple 2x2 case for manual verification
        // Q = [[1, 0], [0, 1]]
        // K = [[1, 0], [0, 1]]
        // V = [[1, 2], [3, 4]]
        let q = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let output = attn.forward(&q, &k, &v).unwrap();

        // Output should be valid (not NaN, not Inf)
        assert_eq!(output.shape(), &[2, 2]);
        for &val in output.data() {
            assert!(val.is_finite());
        }

        // First row of Q=[1,0] has dot products: with K[0]=[1,0] -> 1, with K[1]=[0,1] -> 0
        // After scaling and softmax, should attend more to first position
        // So output[0] should be closer to V[0]=[1,2] than V[1]=[3,4]
        assert!(output.data()[0] < 2.0); // Closer to 1 than 3
        assert!(output.data()[1] < 3.0); // Closer to 2 than 4
    }

    #[test]
    fn test_attention_shape_mismatch_error() {
        let attn = Attention::new(4).unwrap();

        // Q has head_dim=4, K has head_dim=3
        let q = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).unwrap();
        let k = Tensor::from_vec(vec![2, 3], vec![0.2; 6]).unwrap();
        let v = Tensor::from_vec(vec![2, 4], vec![0.3; 8]).unwrap();

        let result = attn.forward(&q, &k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_kv_seq_len_mismatch_error() {
        let attn = Attention::new(4).unwrap();

        // K has seq_len=3, V has seq_len=2
        let q = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).unwrap();
        let k = Tensor::from_vec(vec![3, 4], vec![0.2; 12]).unwrap();
        let v = Tensor::from_vec(vec![2, 4], vec![0.3; 8]).unwrap();

        let result = attn.forward(&q, &k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_softmax_weights_sum() {
        // Verify that attention is using softmax correctly
        // by checking output is weighted combination of values
        let attn = Attention::new(3).unwrap();

        // All equal Q and K means uniform attention
        let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        // V = [[1, 2, 3], [4, 5, 6]]
        let v = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let output = attn.forward(&q, &k, &v).unwrap();

        // With uniform attention, output should be average of V rows
        // Each output row should be close to [2.5, 3.5, 4.5]
        let expected = [2.5, 3.5, 4.5];
        for row in 0..2 {
            for (col, &exp) in expected.iter().enumerate() {
                let actual = output.data()[row * 3 + col];
                assert!(
                    (actual - exp).abs() < 0.01,
                    "row={row}, col={col}: expected {exp}, got {actual}",
                );
            }
        }
    }

    #[test]
    fn test_attention_single_position() {
        // Test with single position (seq_len=1)
        let attn = Attention::new(4).unwrap();

        let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let v = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let output = attn.forward(&q, &k, &v).unwrap();

        // With single position, output should equal V
        assert_eq!(output.shape(), &[1, 4]);
        for i in 0..4 {
            assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
        }
    }
}
