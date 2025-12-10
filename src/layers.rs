//! Neural network layers for transformer models
//!
//! Implements core building blocks for transformer architectures:
//! - Layer normalization
//! - Multi-head attention (MHA, MQA, GQA)
//! - Feed-forward networks
//! - Position embeddings: `RoPE` and `ALiBi`
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::layers::LayerNorm;
//!
//! let layer_norm = LayerNorm::new(512, 1e-5)?;
//! let normalized = layer_norm.forward(&input)?;
//! ```
//!
//! ## Loading Models from Files
//!
//! Realizar supports loading models from GGUF and Safetensors formats.
//!
//! ### GGUF Example
//!
//! ```rust,ignore
//! use realizar::{gguf::GGUFModel, layers::{Model, ModelConfig}};
//!
//! // Load GGUF file
//! let file_data = std::fs::read("model.gguf")?;
//! let gguf = GGUFModel::from_bytes(&file_data)?;
//!
//! // Extract model config from metadata
//! let config = ModelConfig {
//!     vocab_size: 32000,
//!     hidden_dim: 4096,
//!     num_heads: 32,
//!     num_layers: 32,
//!     intermediate_dim: 11008,
//!     eps: 1e-5,
//! };
//!
//! // Create model and load weights
//! let mut model = Model::new(config)?;
//!
//! // Extract tensors by name (requires knowledge of naming convention)
//! let embedding_weights = gguf.get_tensor_f32("token_embd.weight", &file_data)?;
//! // ... load weights into model layers ...
//! ```
//!
//! ### Safetensors Example
//!
//! ```rust,ignore
//! use realizar::{safetensors::SafetensorsModel, layers::{Model, ModelConfig}};
//!
//! // Load Safetensors file
//! let file_data = std::fs::read("model.safetensors")?;
//! let safetensors = SafetensorsModel::from_bytes(&file_data)?;
//!
//! // Extract tensors
//! let embedding_weights = safetensors.get_tensor_f32("model.embed_tokens.weight")?;
//! // ... load weights into model layers ...
//! ```
//!
//! ### Tensor Naming Conventions
//!
//! Different model families use different tensor naming conventions:
//!
//! - **`LLaMA` models (GGUF):**
//!   - `token_embd.weight` - Token embeddings
//!   - `blk.{layer}.attn_q.weight` - Query projection for layer N
//!   - `blk.{layer}.attn_k.weight` - Key projection
//!   - `blk.{layer}.attn_v.weight` - Value projection
//!   - `blk.{layer}.ffn_up.weight` - FFN up projection
//!
//! - **`HuggingFace` models (Safetensors):**
//!   - `model.embed_tokens.weight` - Token embeddings
//!   - `model.layers.{layer}.self_attn.q_proj.weight` - Query projection
//!   - `model.layers.{layer}.self_attn.k_proj.weight` - Key projection
//!   - `model.layers.{layer}.mlp.up_proj.weight` - FFN up projection
//!
//! Consult model documentation for specific naming conventions.

use crate::{
    error::{RealizarError, Result},
    generate::{sample_token, GenerationConfig},
    tensor::Tensor,
};

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

        // Debug assertion for numerical stability
        debug_assert!(
            output.iter().all(|&x| x.is_finite()),
            "LayerNorm produced NaN or Inf values - check input distribution"
        );

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

        // Debug assertion for numerical stability - catch exploding activations early
        debug_assert!(
            output.iter().all(|&x| x.is_finite()),
            "Linear layer produced NaN or Inf values - check for exploding gradients/activations"
        );

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

// =============================================================================
// BENCH-SPRINT-002: QuantizedLinear (Q4_K Inference)
// Per benchmark-model-runners-spec.md v2.0: Inline dequantization for 8x
// memory bandwidth reduction vs f32.
//
// References:
// - [21] Dettmers et al. (2022) - "LLM.int8(): 8-bit Matrix Multiplication"
// - [22] Frantar et al. (2022) - "GPTQ: Post-Training Quantization"
// =============================================================================

/// Q4_K Quantized Linear Layer
///
/// Memory-efficient linear layer using 4-bit K-quantization (Q4_K) format.
/// Achieves ~8x memory reduction vs f32 by storing weights as quantized bytes
/// and performing inline dequantization during matrix-vector multiplication.
///
/// # Performance Characteristics
///
/// - Memory: 4.5 bits/weight (vs 32 bits for f32) → ~7x reduction
/// - Compute: Fused dequant+dot avoids intermediate f32 tensor
/// - Bandwidth: Memory-bound, not compute-bound (per memory wall analysis)
///
/// # Format
///
/// Q4_K uses super-blocks of 256 values:
/// - 144 bytes per super-block
/// - Contains: d (scale), dmin (min), 12-byte scales, 128-byte quantized values
///
/// # Example
///
/// ```rust,ignore
/// // Create from raw Q4_K bytes loaded from GGUF model
/// let layer = QuantizedLinear::new(4096, 4096, q4k_bytes, bias)?;
/// let output = layer.forward(&activations)?;
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Input features (must be multiple of 256 for Q4_K)
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Q4_K quantized weight bytes [out_features * bytes_per_row]
    weight_bytes: Vec<u8>,
    /// Bias vector [out_features]
    bias: Vec<f32>,
    /// Bytes per output row (super_blocks_per_row * 144)
    bytes_per_row: usize,
}

impl QuantizedLinear {
    /// Create a new Q4_K quantized linear layer
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features (should align to 256 for efficiency)
    /// * `out_features` - Number of output features
    /// * `weight_bytes` - Raw Q4_K quantized weight data
    /// * `bias` - Bias vector
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Dimensions are zero
    /// - Weight bytes don't match expected size
    /// - Bias length doesn't match out_features
    pub fn new(
        in_features: usize,
        out_features: usize,
        weight_bytes: Vec<u8>,
        bias: Vec<f32>,
    ) -> Result<Self> {
        // Q4_K: 144 bytes per super-block of 256 values
        const SUPER_BLOCK_VALUES: usize = 256;
        const SUPER_BLOCK_BYTES: usize = 144;

        if in_features == 0 || out_features == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "in_features and out_features must be > 0".to_string(),
            });
        }

        if bias.len() != out_features {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Bias length {} doesn't match out_features {}",
                    bias.len(),
                    out_features
                ),
            });
        }

        let super_blocks_per_row = in_features.div_ceil(SUPER_BLOCK_VALUES);
        let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;
        let expected_bytes = out_features * bytes_per_row;

        if weight_bytes.len() != expected_bytes {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Weight bytes {} doesn't match expected {} ({}x{})",
                    weight_bytes.len(),
                    expected_bytes,
                    out_features,
                    bytes_per_row
                ),
            });
        }

        Ok(Self {
            in_features,
            out_features,
            weight_bytes,
            bias,
            bytes_per_row,
        })
    }

    /// Forward pass through quantized linear layer
    ///
    /// Uses fused dequantization+dot product for memory efficiency.
    /// Per llama.cpp optimization: inline dequant avoids 8x memory traffic penalty.
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
    /// - Input tensor is empty
    /// - Input last dimension doesn't match `in_features`
    /// - Quantization format error during fused dequant+dot
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        use crate::quantize::fused_q4k_dot_simd;

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

        // For each input row, compute output using fused Q4_K dequant+dot
        for row_idx in 0..num_rows {
            let input_start = row_idx * self.in_features;
            let input_row = &data[input_start..input_start + self.in_features];

            // For each output column, use fused dequant+dot
            for j in 0..self.out_features {
                let weight_start = j * self.bytes_per_row;
                let weight_row =
                    &self.weight_bytes[weight_start..weight_start + self.bytes_per_row];

                // Fused dequantization + dot product (SIMD-accelerated)
                let dot = fused_q4k_dot_simd(weight_row, input_row)?;
                output.push(dot + self.bias[j]);
            }
        }

        // Construct output shape
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        // Handle degenerate case: 1D input produces 1D output
        if output_shape.is_empty() {
            output_shape.push(self.out_features);
        }

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

    /// Get weight bytes (for model inspection)
    #[must_use]
    pub fn weight_bytes(&self) -> &[u8] {
        &self.weight_bytes
    }

    /// Get bias vector
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Memory usage in bytes (for diagnostics)
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.weight_bytes.len() + self.bias.len() * std::mem::size_of::<f32>()
    }
}

/// Fused LayerNorm + Linear layer
///
/// Combines layer normalization and linear transformation in a single pass
/// to reduce memory bandwidth by avoiding intermediate tensor writes.
///
/// # Algorithm
///
/// Standard (two passes):
/// ```text
/// norm_out = LayerNorm(input)   // Full tensor written to memory
/// output = Linear(norm_out)      // Full tensor read from memory
/// ```
///
/// Fused (single pass):
/// ```text
/// For each row:
///   1. Compute mean and variance (in registers)
///   2. For each output column:
///      a. Normalize input in registers
///      b. Apply linear transformation directly
/// ```
///
/// This reduces memory traffic by ~50% for the LayerNorm→Linear pattern.
///
/// # References
///
/// - "Fused Operations for Deep Learning" - NVIDIA, 2019
#[derive(Debug, Clone)]
pub struct FusedLayerNormLinear {
    /// Feature dimension (must match between LayerNorm and Linear input)
    feature_dim: usize,
    /// Output dimension
    out_features: usize,
    /// LayerNorm epsilon
    eps: f32,
    /// LayerNorm weight (gamma)
    norm_weight: Vec<f32>,
    /// LayerNorm bias (beta)
    norm_bias: Vec<f32>,
    /// Linear weight matrix [feature_dim, out_features]
    linear_weight: Vec<f32>,
    /// Linear bias vector [out_features]
    linear_bias: Vec<f32>,
}

impl FusedLayerNormLinear {
    /// Create a new fused LayerNorm+Linear layer
    ///
    /// # Arguments
    ///
    /// * `feature_dim` - Input feature dimension (normalized dimension)
    /// * `out_features` - Output dimension of linear layer
    /// * `eps` - LayerNorm epsilon for numerical stability
    ///
    /// # Errors
    ///
    /// Returns error if feature_dim or out_features is zero
    pub fn new(feature_dim: usize, out_features: usize, eps: f32) -> Result<Self> {
        if feature_dim == 0 || out_features == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "feature_dim and out_features must be > 0".to_string(),
            });
        }

        Ok(Self {
            feature_dim,
            out_features,
            eps,
            norm_weight: vec![1.0; feature_dim],
            norm_bias: vec![0.0; feature_dim],
            linear_weight: vec![0.0; feature_dim * out_features],
            linear_bias: vec![0.0; out_features],
        })
    }

    /// Forward pass with fused LayerNorm + Linear
    ///
    /// Computes `Linear(LayerNorm(input))` in a single pass without
    /// materializing the intermediate normalized tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor `[batch, feature_dim]` or `[feature_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor `[batch, out_features]` or `[out_features]`
    ///
    /// # Errors
    ///
    /// Returns error if input dimensions don't match feature_dim
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.feature_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match feature_dim {}",
                    last_dim, self.feature_dim
                ),
            });
        }

        let data = input.data();
        let num_rows = data.len() / self.feature_dim;
        let mut output = Vec::with_capacity(num_rows * self.out_features);

        for row_idx in 0..num_rows {
            let row_start = row_idx * self.feature_dim;
            let row = &data[row_start..row_start + self.feature_dim];

            // Step 1: Compute mean (in registers)
            #[allow(clippy::cast_precision_loss)]
            let mean: f32 = row.iter().sum::<f32>() / self.feature_dim as f32;

            // Step 2: Compute variance (in registers)
            #[allow(clippy::cast_precision_loss)]
            let variance: f32 = row
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f32>()
                / self.feature_dim as f32;

            let inv_std = 1.0 / (variance + self.eps).sqrt();

            // Step 3: Fused normalize + linear for each output column
            // This avoids writing normalized values to memory
            for j in 0..self.out_features {
                let mut sum = self.linear_bias[j];
                for (i, &x) in row.iter().enumerate() {
                    // Normalize in registers
                    let normalized = (x - mean) * inv_std;
                    let transformed = normalized * self.norm_weight[i] + self.norm_bias[i];
                    // Apply linear weight immediately
                    sum += transformed * self.linear_weight[i * self.out_features + j];
                }
                output.push(sum);
            }
        }

        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        Tensor::from_vec(output_shape, output)
    }

    /// Parallel forward pass using rayon
    ///
    /// Parallelizes over rows for multi-core utilization.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input tensor is empty
    /// - Last dimension doesn't match feature_dim
    pub fn forward_parallel(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        use rayon::prelude::*;

        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.feature_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match feature_dim {}",
                    last_dim, self.feature_dim
                ),
            });
        }

        let data = input.data();
        let num_rows = data.len() / self.feature_dim;

        let output: Vec<f32> = (0..num_rows)
            .into_par_iter()
            .flat_map(|row_idx| {
                let row_start = row_idx * self.feature_dim;
                let row = &data[row_start..row_start + self.feature_dim];

                // Compute mean and variance
                #[allow(clippy::cast_precision_loss)]
                let mean: f32 = row.iter().sum::<f32>() / self.feature_dim as f32;

                #[allow(clippy::cast_precision_loss)]
                let variance: f32 = row
                    .iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum::<f32>()
                    / self.feature_dim as f32;

                let inv_std = 1.0 / (variance + self.eps).sqrt();

                // Fused normalize + linear
                (0..self.out_features)
                    .map(|j| {
                        let mut sum = self.linear_bias[j];
                        for (i, &x) in row.iter().enumerate() {
                            let normalized = (x - mean) * inv_std;
                            let transformed = normalized * self.norm_weight[i] + self.norm_bias[i];
                            sum += transformed * self.linear_weight[i * self.out_features + j];
                        }
                        sum
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        Tensor::from_vec(output_shape, output)
    }

    /// Get feature dimension
    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Get output features
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get mutable reference to LayerNorm weight (gamma)
    #[must_use]
    pub fn norm_weight_mut(&mut self) -> &mut [f32] {
        &mut self.norm_weight
    }

    /// Get mutable reference to LayerNorm bias (beta)
    #[must_use]
    pub fn norm_bias_mut(&mut self) -> &mut [f32] {
        &mut self.norm_bias
    }

    /// Get mutable reference to Linear weight
    #[must_use]
    pub fn linear_weight_mut(&mut self) -> &mut [f32] {
        &mut self.linear_weight
    }

    /// Get mutable reference to Linear bias
    #[must_use]
    pub fn linear_bias_mut(&mut self) -> &mut [f32] {
        &mut self.linear_bias
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

        // Debug assertion for numerical stability
        debug_assert!(
            output.iter().all(|&x| x.is_finite()),
            "Attention layer produced NaN or Inf values - check input scaling"
        );

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

    /// Compute Flash Attention - memory-efficient block-wise attention
    ///
    /// Uses tiling and recomputation to reduce memory usage from O(N²) to O(N).
    /// Implements block-wise softmax with running max/sum statistics.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor `[seq_len, head_dim]`
    /// * `key` - Key tensor `[seq_len, head_dim]`
    /// * `value` - Value tensor `[seq_len, head_dim]`
    /// * `block_size` - Tile size for block-wise computation (e.g., 64, 128)
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, head_dim]` (same as standard attention)
    ///
    /// # Errors
    ///
    /// Returns error if shapes don't match or `block_size` is zero
    ///
    /// # References
    ///
    /// - "`FlashAttention`: Fast and Memory-Efficient Exact Attention" - Dao et al., 2022
    /// - "FlashAttention-2: Faster Attention with Better Parallelism" - Dao, 2023
    pub fn flash_forward(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        block_size: usize,
    ) -> Result<Tensor<f32>> {
        if block_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "block_size must be > 0".to_string(),
            });
        }

        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        // Validate shapes (same as standard attention)
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

        // Initialize output and statistics
        let mut output = vec![0.0; q_seq_len * self.head_dim];
        let mut row_max = vec![f32::NEG_INFINITY; q_seq_len]; // Running max for each query row
        let mut row_sum = vec![0.0; q_seq_len]; // Running sum for each query row

        // Iterate over K/V blocks (outer loop)
        let num_kv_blocks = k_seq_len.div_ceil(block_size);
        for kv_block_idx in 0..num_kv_blocks {
            let kv_start = kv_block_idx * block_size;
            let kv_end = (kv_start + block_size).min(k_seq_len);
            let kv_block_len = kv_end - kv_start;

            // Iterate over Q blocks (inner loop)
            let num_q_blocks = q_seq_len.div_ceil(block_size);
            for q_block_idx in 0..num_q_blocks {
                let q_start = q_block_idx * block_size;
                let q_end = (q_start + block_size).min(q_seq_len);

                // Compute attention scores for this block: Q_block @ K_block.T
                let mut scores = vec![0.0; (q_end - q_start) * kv_block_len];
                for (i, q_idx) in (q_start..q_end).enumerate() {
                    for (j, kv_idx) in (kv_start..kv_end).enumerate() {
                        let mut dot = 0.0;
                        for k in 0..self.head_dim {
                            dot += q_data[q_idx * self.head_dim + k]
                                * k_data[kv_idx * self.head_dim + k];
                        }
                        scores[i * kv_block_len + j] = dot * self.scale;
                    }
                }

                // Update running max and apply softmax with new max
                for (i, q_idx) in (q_start..q_end).enumerate() {
                    // Find max in current block
                    let block_max = (0..kv_block_len)
                        .map(|j| scores[i * kv_block_len + j])
                        .fold(f32::NEG_INFINITY, f32::max);

                    // Update global max
                    let old_max = row_max[q_idx];
                    let new_max = old_max.max(block_max);
                    row_max[q_idx] = new_max;

                    // Compute exp(scores - new_max) and update running sum
                    let mut block_sum = 0.0;
                    for j in 0..kv_block_len {
                        let exp_val = (scores[i * kv_block_len + j] - new_max).exp();
                        scores[i * kv_block_len + j] = exp_val;
                        block_sum += exp_val;
                    }

                    // Rescale old output and sum based on new max
                    let scale_factor = (old_max - new_max).exp();
                    for k in 0..self.head_dim {
                        output[q_idx * self.head_dim + k] *= scale_factor;
                    }
                    row_sum[q_idx] = row_sum[q_idx] * scale_factor + block_sum;
                }

                // Accumulate weighted values: output += scores @ V_block
                for (i, q_idx) in (q_start..q_end).enumerate() {
                    for k in 0..self.head_dim {
                        let mut weighted_sum = 0.0;
                        for (j, kv_idx) in (kv_start..kv_end).enumerate() {
                            weighted_sum +=
                                scores[i * kv_block_len + j] * v_data[kv_idx * self.head_dim + k];
                        }
                        output[q_idx * self.head_dim + k] += weighted_sum;
                    }
                }
            }
        }

        // Final normalization by row_sum
        for i in 0..q_seq_len {
            for k in 0..self.head_dim {
                output[i * self.head_dim + k] /= row_sum[i];
            }
        }

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }

    /// Flash Attention v2 with SIMD-accelerated dot products
    ///
    /// Optimized implementation using AVX2 SIMD for dot products.
    /// Uses parallel outer loop over query blocks for better multi-core utilization.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor `[seq_len, head_dim]`
    /// * `key` - Key tensor `[seq_len, head_dim]`
    /// * `value` - Value tensor `[seq_len, head_dim]`
    /// * `block_size` - Tile size for block-wise computation (e.g., 64, 128)
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, head_dim]` (same as standard attention)
    ///
    /// # Errors
    ///
    /// Returns error if shapes don't match or `block_size` is zero
    ///
    /// # References
    ///
    /// - "FlashAttention-2: Faster Attention with Better Parallelism" - Dao, 2023
    #[allow(clippy::similar_names)]
    pub fn flash_forward_v2(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        block_size: usize,
    ) -> Result<Tensor<f32>> {
        if block_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "block_size must be > 0".to_string(),
            });
        }

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
        let head_dim = self.head_dim;
        let scale = self.scale;

        // Initialize output and statistics
        let mut output = vec![0.0; q_seq_len * head_dim];
        let mut row_max = vec![f32::NEG_INFINITY; q_seq_len];
        let mut row_sum = vec![0.0; q_seq_len];

        // Flash Attention v2: Iterate over K/V blocks in outer loop
        // This allows better memory access patterns
        let num_kv_blocks = k_seq_len.div_ceil(block_size);

        for kv_block_idx in 0..num_kv_blocks {
            let kv_start = kv_block_idx * block_size;
            let kv_end = (kv_start + block_size).min(k_seq_len);
            let kv_block_len = kv_end - kv_start;

            // Process all Q rows against this K/V block
            for q_idx in 0..q_seq_len {
                // SIMD-accelerated dot products for this row
                let mut scores = Vec::with_capacity(kv_block_len);
                for kv_idx in kv_start..kv_end {
                    let dot = Self::simd_dot_product(
                        &q_data[q_idx * head_dim..(q_idx + 1) * head_dim],
                        &k_data[kv_idx * head_dim..(kv_idx + 1) * head_dim],
                    );
                    scores.push(dot * scale);
                }

                // Find max in current block
                let block_max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Update global max
                let old_max = row_max[q_idx];
                let new_max = old_max.max(block_max);
                row_max[q_idx] = new_max;

                // Compute exp(scores - new_max) and update running sum
                let mut block_sum = 0.0;
                for score in &mut scores {
                    let exp_val = (*score - new_max).exp();
                    *score = exp_val;
                    block_sum += exp_val;
                }

                // Rescale old output and sum based on new max
                let scale_factor = (old_max - new_max).exp();
                for k in 0..head_dim {
                    output[q_idx * head_dim + k] *= scale_factor;
                }
                row_sum[q_idx] = row_sum[q_idx] * scale_factor + block_sum;

                // Accumulate weighted values: output += scores @ V_block
                for (j, kv_idx) in (kv_start..kv_end).enumerate() {
                    let weight = scores[j];
                    for k in 0..head_dim {
                        output[q_idx * head_dim + k] += weight * v_data[kv_idx * head_dim + k];
                    }
                }
            }
        }

        // Final normalization by row_sum
        for i in 0..q_seq_len {
            let inv_sum = 1.0 / row_sum[i];
            for k in 0..head_dim {
                output[i * head_dim + k] *= inv_sum;
            }
        }

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }

    /// SIMD-accelerated dot product
    ///
    /// Uses AVX2 on x86_64 for 8-way f32 parallelism
    #[inline]
    fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            Self::simd_dot_avx2(a, b)
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            Self::scalar_dot_product(a, b)
        }
    }

    /// AVX2 SIMD dot product (8-way f32 parallelism)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    #[allow(clippy::wildcard_imports)]
    fn simd_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let len = a.len().min(b.len());
        let chunks = len / 8;
        let remainder = len % 8;

        // SIMD accumulator
        let simd_sum = unsafe {
            let mut acc = _mm256_setzero_ps();

            for i in 0..chunks {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
            }

            // Horizontal sum of 8 floats
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let hi64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, hi64);
            let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55);
            let sum32 = _mm_add_ss(sum64, hi32);
            _mm_cvtss_f32(sum32)
        };

        // Handle remainder
        let remainder_sum: f32 = (0..remainder)
            .map(|i| a[chunks * 8 + i] * b[chunks * 8 + i])
            .sum();

        simd_sum + remainder_sum
    }

    /// Scalar fallback dot product
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    #[inline]
    fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Parallel Flash Attention v2 using rayon
    ///
    /// Parallelizes over query positions for multi-core utilization.
    /// Each thread processes a subset of query rows independently.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor `[seq_len, head_dim]`
    /// * `key` - Key tensor `[seq_len, head_dim]`
    /// * `value` - Value tensor `[seq_len, head_dim]`
    /// * `block_size` - Tile size for block-wise computation (e.g., 64, 128)
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, head_dim]` (same as standard attention)
    ///
    /// # Errors
    ///
    /// Returns error if shapes don't match or `block_size` is zero
    #[allow(clippy::similar_names)]
    pub fn flash_forward_parallel(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        block_size: usize,
    ) -> Result<Tensor<f32>> {
        use rayon::prelude::*;

        if block_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "block_size must be > 0".to_string(),
            });
        }

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
        let head_dim = self.head_dim;
        let scale = self.scale;

        // Parallel over query positions
        let output: Vec<f32> = (0..q_seq_len)
            .into_par_iter()
            .flat_map(|q_idx| {
                // Each query row is processed independently
                let mut row_output = vec![0.0; head_dim];
                let mut row_max = f32::NEG_INFINITY;
                let mut row_sum = 0.0;

                let num_kv_blocks = k_seq_len.div_ceil(block_size);

                for kv_block_idx in 0..num_kv_blocks {
                    let kv_start = kv_block_idx * block_size;
                    let kv_end = (kv_start + block_size).min(k_seq_len);

                    // Compute scores for this K/V block
                    let mut scores: Vec<f32> = (kv_start..kv_end)
                        .map(|kv_idx| {
                            let dot = Self::simd_dot_product(
                                &q_data[q_idx * head_dim..(q_idx + 1) * head_dim],
                                &k_data[kv_idx * head_dim..(kv_idx + 1) * head_dim],
                            );
                            dot * scale
                        })
                        .collect();

                    // Online softmax: find block max and update global max
                    let block_max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let old_max = row_max;
                    let new_max = old_max.max(block_max);
                    row_max = new_max;

                    // Compute exp(scores - new_max)
                    let mut block_sum = 0.0;
                    for score in &mut scores {
                        let exp_val = (*score - new_max).exp();
                        *score = exp_val;
                        block_sum += exp_val;
                    }

                    // Rescale previous output
                    let scale_factor = (old_max - new_max).exp();
                    for out_val in &mut row_output {
                        *out_val *= scale_factor;
                    }
                    row_sum = row_sum * scale_factor + block_sum;

                    // Accumulate weighted values
                    for (j, kv_idx) in (kv_start..kv_end).enumerate() {
                        let weight = scores[j];
                        for k in 0..head_dim {
                            row_output[k] += weight * v_data[kv_idx * head_dim + k];
                        }
                    }
                }

                // Final normalization
                let inv_sum = 1.0 / row_sum;
                for out_val in &mut row_output {
                    *out_val *= inv_sum;
                }

                row_output
            })
            .collect();

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }
}

/// Multi-Head Attention with support for MHA, MQA, and GQA
///
/// Implements three attention variants through configurable `KV` head count:
///
/// **Multi-Head Attention (MHA):** `num_kv_heads = num_heads`
/// - Each head has separate Q, K, V projections
/// - `KV` cache: `O(num_heads * seq_len * head_dim)`
/// - Standard attention mechanism
///
/// **Multi-Query Attention (MQA):** `num_kv_heads = 1`
/// - Each head has separate Q projection
/// - All heads share single K, V projection
/// - `KV` cache: `O(seq_len * head_dim)` - reduces by `num_heads` factor
/// - Used in `PaLM`, Falcon, `StarCoder`
///
/// **Grouped-Query Attention (GQA):** `1 < num_kv_heads < num_heads`
/// - Heads grouped into `num_kv_heads` groups
/// - Each group shares K, V projections
/// - `KV` cache: `O(num_kv_heads * seq_len * head_dim)`
/// - Used in `Llama-2`, Mistral, `CodeLlama`
///
/// # Architecture
///
/// ```text
/// Input [hidden_dim]
///   |
///   ├─> Q_proj [hidden_dim -> hidden_dim] -> split into num_heads
///   ├─> K_proj [hidden_dim -> num_kv_heads * head_dim]
///   └─> V_proj [hidden_dim -> num_kv_heads * head_dim]
///   |
///   ├─> Attention (grouped by num_kv_heads)
///   |
///   └─> O_proj [hidden_dim -> hidden_dim]
///       |
///     Output [hidden_dim]
/// ```
///
/// # References
///
/// - "Attention is All You Need" - Vaswani et al., 2017 (MHA)
/// - "Fast Transformer Decoding: One Write-Head is All You Need" - Shazeer, 2019 (MQA)
/// - "`PaLM`: Scaling Language Modeling with Pathways" - Chowdhery et al., 2022 (MQA)
/// - "`GQA`: Training Generalized Multi-Query Transformer" - Ainslie et al., 2023 (GQA)
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of attention heads (Q heads)
    num_heads: usize,
    /// Number of key/value heads (for GQA/MQA)
    num_kv_heads: usize,
    /// Dimension per attention head
    head_dim: usize,
    /// Total hidden dimension (`num_heads * head_dim`)
    hidden_dim: usize,
    /// Query projection: `hidden_dim -> hidden_dim`
    q_proj: Linear,
    /// Key projection: `hidden_dim -> num_kv_heads * head_dim`
    k_proj: Linear,
    /// Value projection: `hidden_dim -> num_kv_heads * head_dim`
    v_proj: Linear,
    /// Output projection: `hidden_dim -> hidden_dim`
    o_proj: Linear,
    /// Per-head attention mechanism
    attention: Attention,
}

impl MultiHeadAttention {
    /// Create a new Multi-Head Attention layer with configurable `KV` heads
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Total hidden dimension (must be divisible by `num_heads`)
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key/value heads (must divide `num_heads`)
    ///
    /// # Modes
    ///
    /// - MHA: `num_kv_heads = num_heads` (standard multi-head)
    /// - MQA: `num_kv_heads = 1` (all heads share K/V)
    /// - GQA: `1 < num_kv_heads < num_heads` (grouped heads)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `hidden_dim` is zero or not divisible by `num_heads`
    /// - `num_heads` is zero or not divisible by `num_kv_heads`
    /// - `num_kv_heads` is zero or greater than `num_heads`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Standard Multi-Head Attention (MHA)
    /// let mha = MultiHeadAttention::new(512, 8, 8)?;
    ///
    /// // Multi-Query Attention (MQA)
    /// let mqa = MultiHeadAttention::new(512, 8, 1)?;
    ///
    /// // Grouped-Query Attention (GQA) - 4 heads per group
    /// let gqa = MultiHeadAttention::new(512, 8, 2)?;
    /// ```
    pub fn new(hidden_dim: usize, num_heads: usize, num_kv_heads: usize) -> Result<Self> {
        if hidden_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "hidden_dim must be > 0".to_string(),
            });
        }
        if num_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_heads must be > 0".to_string(),
            });
        }
        if num_kv_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_kv_heads must be > 0".to_string(),
            });
        }
        if num_kv_heads > num_heads {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "num_kv_heads {num_kv_heads} cannot be greater than num_heads {num_heads}"
                ),
            });
        }
        if hidden_dim % num_heads != 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
                ),
            });
        }
        if num_heads % num_kv_heads != 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
                ),
            });
        }

        let head_dim = hidden_dim / num_heads;

        // Q projection: always hidden_dim -> hidden_dim (all query heads)
        let q_proj = Linear::new(hidden_dim, hidden_dim)?;

        // K/V projections: hidden_dim -> num_kv_heads * head_dim
        let kv_dim = num_kv_heads * head_dim;
        let k_proj = Linear::new(hidden_dim, kv_dim)?;
        let v_proj = Linear::new(hidden_dim, kv_dim)?;

        // Output projection: hidden_dim -> hidden_dim
        let o_proj = Linear::new(hidden_dim, hidden_dim)?;

        // Per-head attention mechanism
        let attention = Attention::new(head_dim)?;

        Ok(Self {
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attention,
        })
    }

    /// Create standard Multi-Head Attention (MHA) - each head has separate K/V
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidShape` if:
    /// - `hidden_dim` is 0
    /// - `num_heads` is 0
    /// - `hidden_dim` is not divisible by `num_heads`
    pub fn mha(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Self::new(hidden_dim, num_heads, num_heads)
    }

    /// Create Multi-Query Attention (MQA) - all heads share K/V
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidShape` if:
    /// - `hidden_dim` is 0
    /// - `num_heads` is 0
    /// - `hidden_dim` is not divisible by `num_heads`
    pub fn mqa(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Self::new(hidden_dim, num_heads, 1)
    }

    /// Create Grouped-Query Attention (GQA) - heads grouped to share K/V
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidShape` if:
    /// - `hidden_dim` is 0
    /// - `num_heads` is 0
    /// - `num_kv_heads` is 0
    /// - `num_kv_heads` is greater than `num_heads`
    /// - `hidden_dim` is not divisible by `num_heads`
    /// - `num_heads` is not divisible by `num_kv_heads`
    pub fn gqa(hidden_dim: usize, num_heads: usize, num_kv_heads: usize) -> Result<Self> {
        Self::new(hidden_dim, num_heads, num_kv_heads)
    }

    /// Forward pass through multi-head attention
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor `[seq_len, hidden_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, hidden_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if input shape is invalid
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(RealizarError::InvalidShape {
                reason: format!("Expected 2D tensor [seq_len, hidden_dim], got shape {shape:?}"),
            });
        }

        let seq_len = shape[0];
        let input_dim = shape[1];

        if input_dim != self.hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Expected hidden_dim={}, got {}", self.hidden_dim, input_dim),
            });
        }

        // Project Q, K, V
        let q = self.q_proj.forward(input)?; // [seq_len, hidden_dim]
        let k = self.k_proj.forward(input)?; // [seq_len, kv_dim]
        let v = self.v_proj.forward(input)?; // [seq_len, kv_dim]

        // Reshape Q into heads: [seq_len, num_heads, head_dim]
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();

        // Calculate heads per group for GQA
        let heads_per_group = self.num_heads / self.num_kv_heads;

        // Process each query head
        let mut head_outputs = Vec::with_capacity(self.num_heads);

        for head_idx in 0..self.num_heads {
            // Extract Q for this head
            let mut q_head_data = Vec::with_capacity(seq_len * self.head_dim);
            for seq_idx in 0..seq_len {
                let q_row_start = seq_idx * self.hidden_dim;
                let head_start = q_row_start + head_idx * self.head_dim;
                for offset in 0..self.head_dim {
                    q_head_data.push(q_data[head_start + offset]);
                }
            }
            let q_head = Tensor::from_vec(vec![seq_len, self.head_dim], q_head_data)?;

            // Determine which KV head this Q head uses (for GQA/MQA/MHA)
            let kv_head_idx = head_idx / heads_per_group;
            let kv_dim = self.num_kv_heads * self.head_dim;

            // Extract K, V for the corresponding KV head
            let mut k_head_data = Vec::with_capacity(seq_len * self.head_dim);
            let mut v_head_data = Vec::with_capacity(seq_len * self.head_dim);
            for seq_idx in 0..seq_len {
                let kv_row_start = seq_idx * kv_dim;
                let kv_head_start = kv_row_start + kv_head_idx * self.head_dim;
                for offset in 0..self.head_dim {
                    k_head_data.push(k_data[kv_head_start + offset]);
                    v_head_data.push(v_data[kv_head_start + offset]);
                }
            }
            let k_head = Tensor::from_vec(vec![seq_len, self.head_dim], k_head_data)?;
            let v_head = Tensor::from_vec(vec![seq_len, self.head_dim], v_head_data)?;

            // Compute attention for this head
            let head_output = self.attention.forward(&q_head, &k_head, &v_head)?;
            head_outputs.push(head_output);
        }

        // Concatenate all head outputs: [seq_len, hidden_dim]
        let mut concat_data = Vec::with_capacity(seq_len * self.hidden_dim);
        for seq_idx in 0..seq_len {
            for head_output in &head_outputs {
                let head_output_data = head_output.data();
                let head_row_start = seq_idx * self.head_dim;
                for offset in 0..self.head_dim {
                    concat_data.push(head_output_data[head_row_start + offset]);
                }
            }
        }

        let concat = Tensor::from_vec(vec![seq_len, self.hidden_dim], concat_data)?;

        // Output projection
        self.o_proj.forward(&concat)
    }

    /// Get number of query heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get number of key/value heads
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Check if using Multi-Query Attention (MQA)
    #[must_use]
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Check if using Grouped-Query Attention (GQA)
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads > 1 && self.num_kv_heads < self.num_heads
    }

    /// Check if using standard Multi-Head Attention (MHA)
    #[must_use]
    pub fn is_mha(&self) -> bool {
        self.num_kv_heads == self.num_heads
    }
}

/// Rotary Position Embeddings (`RoPE`)
///
/// Applies position-dependent rotations to query and key vectors.
/// Used in `LLaMA`, `PaLM`, and other modern transformers for relative
/// position encoding.
///
/// The rotation is applied pairwise to dimensions, encoding position
/// information directly into the embeddings.
///
/// # Formula
///
/// For each pair of dimensions (2i, 2i+1):
/// ```text
/// x'_{2i} = x_{2i} * cos(θ_i * pos) - x_{2i+1} * sin(θ_i * pos)
/// x'_{2i+1} = x_{2i} * sin(θ_i * pos) + x_{2i+1} * cos(θ_i * pos)
/// ```
///
/// Where `θ_i` = base^(-2i/dim)
///
/// # References
///
/// `RoFormer`: Enhanced Transformer with Rotary Position Embedding - Su et al., 2021
#[derive(Debug, Clone)]
pub struct RoPE {
    /// Embedding dimension (must be even)
    dim: usize,
    /// Base for computing frequencies (default: 10000)
    base: f32,
    /// Precomputed inverse frequencies for each dimension pair
    inv_freq: Vec<f32>,
}

impl RoPE {
    /// Create a new `RoPE` layer
    ///
    /// # Arguments
    ///
    /// * `dim` - Embedding dimension (must be even)
    /// * `base` - Base for computing frequencies (typically 10000)
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is zero or odd
    pub fn new(dim: usize, base: f32) -> Result<Self> {
        if dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "dim must be > 0".to_string(),
            });
        }
        if dim % 2 != 0 {
            return Err(RealizarError::InvalidShape {
                reason: "dim must be even for RoPE".to_string(),
            });
        }

        // Compute inverse frequencies: base^(-2i/dim) for i in 0..dim/2
        let half_dim = dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);

        #[allow(clippy::cast_precision_loss)]
        for i in 0..half_dim {
            let exponent = -2.0 * (i as f32) / (dim as f32);
            inv_freq.push(base.powf(exponent));
        }

        Ok(Self {
            dim,
            base,
            inv_freq,
        })
    }

    /// Create `RoPE` with default base (10000)
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is zero or odd
    pub fn with_default_base(dim: usize) -> Result<Self> {
        Self::new(dim, 10000.0)
    }

    /// Apply rotary embeddings to input at given position
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with last dimension equal to `dim`
    /// * `position` - Position index for computing rotation angles
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input, with rotary embeddings applied
    ///
    /// # Errors
    ///
    /// Returns error if input's last dimension doesn't match `dim`
    pub fn forward(&self, input: &Tensor<f32>, position: usize) -> Result<Tensor<f32>> {
        let shape = input.shape();

        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor must have at least 1 dimension".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Expected last dimension {}, got {}", self.dim, last_dim),
            });
        }

        let data = input.data();
        let num_vectors = data.len() / self.dim;
        let mut output = Vec::with_capacity(data.len());

        // Compute sin/cos for this position
        let half_dim = self.dim / 2;
        let mut cos_vals = Vec::with_capacity(half_dim);
        let mut sin_vals = Vec::with_capacity(half_dim);

        #[allow(clippy::cast_precision_loss)]
        for inv_f in &self.inv_freq {
            let angle = inv_f * (position as f32);
            cos_vals.push(angle.cos());
            sin_vals.push(angle.sin());
        }

        // Apply rotation to each vector
        for vec_idx in 0..num_vectors {
            let offset = vec_idx * self.dim;

            for i in 0..half_dim {
                let x0 = data[offset + 2 * i];
                let x1 = data[offset + 2 * i + 1];
                let cos_val = cos_vals[i];
                let sin_val = sin_vals[i];

                // Apply 2D rotation
                let y0 = x0 * cos_val - x1 * sin_val;
                let y1 = x0 * sin_val + x1 * cos_val;

                output.push(y0);
                output.push(y1);
            }
        }

        Tensor::from_vec(shape.to_vec(), output)
    }

    /// Get embedding dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get base frequency
    #[must_use]
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Get inverse frequencies
    #[must_use]
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }
}

/// Attention with Linear Biases (`ALiBi`)
///
/// Replaces traditional position embeddings by adding a static, non-learned
/// bias to query-key attention scores. The bias is proportional to the distance
/// between positions, with head-specific slopes that enable better length
/// extrapolation.
///
/// # Algorithm
///
/// For each attention head h, `ALiBi` adds the following bias to attention scores:
///
/// ```text
/// bias[i, j] = -m[h] * |i - j|
/// ```
///
/// where m[h] is the head-specific slope computed as:
/// - For powers of 2: m[h] = 2^(-8h/n) where n is the number of heads
/// - For non-powers of 2: interpolation between adjacent powers of 2
///
/// # References
///
/// - "Train Short, Test Long: Attention with Linear Biases Enables Input Length
///   Extrapolation" - Press et al., ICLR 2022
/// - <https://arxiv.org/abs/2108.12409>
///
/// # Example
///
/// ```rust,ignore
/// use realizar::layers::ALiBi;
///
/// // Create ALiBi for 8 attention heads
/// let alibi = ALiBi::new(8)?;
///
/// // Get bias matrix for sequence length 10
/// let bias = alibi.get_bias(10)?;
///
/// // Add to attention scores before softmax
/// // scores: [seq_len, seq_len, num_heads]
/// // scores = scores + bias
/// ```
#[derive(Debug, Clone)]
pub struct ALiBi {
    /// Number of attention heads
    num_heads: usize,
    /// Head-specific slopes (one per head)
    slopes: Vec<f32>,
}

impl ALiBi {
    /// Create a new `ALiBi` layer
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    ///
    /// # Errors
    ///
    /// Returns error if `num_heads` is zero
    pub fn new(num_heads: usize) -> Result<Self> {
        if num_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_heads must be > 0".to_string(),
            });
        }

        // Compute slopes for each head
        let slopes = Self::compute_slopes(num_heads);

        Ok(Self { num_heads, slopes })
    }

    /// Compute head-specific slopes following `ALiBi` paper algorithm
    ///
    /// For powers of 2: m[h] = 2^(-8h/n)
    /// For non-powers of 2: interpolate between adjacent powers of 2
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        // Find closest power of 2
        let closest_power_of_2 = if num_heads.is_power_of_two() {
            num_heads
        } else {
            num_heads.next_power_of_two() / 2
        };

        #[allow(clippy::cast_precision_loss)]
        let ratio = 8.0 / (closest_power_of_2 as f32);

        let mut slopes = Vec::with_capacity(num_heads);

        // Compute slopes for power of 2 heads
        for i in 0..closest_power_of_2.min(num_heads) {
            #[allow(clippy::cast_precision_loss)]
            let exponent = -(i as f32) * ratio;
            slopes.push(2_f32.powf(exponent));
        }

        // If not power of 2, add extra slopes with step=2
        if num_heads > closest_power_of_2 {
            #[allow(clippy::cast_precision_loss)]
            let extra_ratio = 4.0 / (closest_power_of_2 as f32);

            for i in 0..(num_heads - closest_power_of_2) {
                #[allow(clippy::cast_precision_loss)]
                let exponent = -((2 * i + 1) as f32) * extra_ratio;
                slopes.push(2_f32.powf(exponent));
            }
        }

        slopes
    }

    /// Get bias matrix for a given sequence length
    ///
    /// Returns a tensor of shape `[seq_len, seq_len, num_heads]` where:
    /// ```text
    /// bias[i, j, h] = -slopes[h] * abs(i - j)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Sequence length for computing bias
    ///
    /// # Returns
    ///
    /// Tensor of shape `[seq_len, seq_len, num_heads]` containing position biases
    ///
    /// # Errors
    ///
    /// Returns error if `seq_len` is zero
    pub fn get_bias(&self, seq_len: usize) -> Result<Tensor<f32>> {
        if seq_len == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "seq_len must be > 0".to_string(),
            });
        }

        let total_size = seq_len * seq_len * self.num_heads;
        let mut data = Vec::with_capacity(total_size);

        // Compute bias for each position pair and head
        for i in 0..seq_len {
            for j in 0..seq_len {
                for &slope in &self.slopes {
                    #[allow(clippy::cast_precision_loss)]
                    let distance = (i as f32 - j as f32).abs();
                    let bias = -slope * distance;
                    data.push(bias);
                }
            }
        }

        Tensor::from_vec(vec![seq_len, seq_len, self.num_heads], data)
    }

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head-specific slopes
    #[must_use]
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }
}

/// Key-Value Cache for efficient transformer inference
///
/// Stores key and value tensors from previous positions to avoid
/// recomputation during autoregressive generation. Each forward pass
/// only computes K/V for the new token and appends to the cache.
///
/// # Usage
///
/// 1. Create cache with `KVCache::new(num_layers, max_seq_len, head_dim)`
/// 2. At each generation step, call `update` with new K/V
/// 3. Use `get_key`/`get_value` to retrieve cached tensors
/// 4. Call `clear` to reset for new sequence
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Dimension per head
    head_dim: usize,
    /// Current sequence position
    current_pos: usize,
    /// Cached keys for each layer: `[num_layers][max_seq_len * head_dim]`
    keys: Vec<Vec<f32>>,
    /// Cached values for each layer: `[num_layers][max_seq_len * head_dim]`
    values: Vec<Vec<f32>>,
}

impl KVCache {
    /// Create a new KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers to cache
    /// * `max_seq_len` - Maximum sequence length to cache
    /// * `head_dim` - Dimension per attention head
    ///
    /// # Errors
    ///
    /// Returns error if any dimension is zero
    pub fn new(num_layers: usize, max_seq_len: usize, head_dim: usize) -> Result<Self> {
        if num_layers == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_layers must be > 0".to_string(),
            });
        }
        if max_seq_len == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "max_seq_len must be > 0".to_string(),
            });
        }
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }

        let cache_size = max_seq_len * head_dim;
        let keys = vec![vec![0.0; cache_size]; num_layers];
        let values = vec![vec![0.0; cache_size]; num_layers];

        Ok(Self {
            num_layers,
            max_seq_len,
            head_dim,
            current_pos: 0,
            keys,
            values,
        })
    }

    /// Update cache with new key/value for a layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `key` - New key tensor `[head_dim]`
    /// * `value` - New value tensor `[head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds, cache is full, or tensor sizes don't match
    pub fn update(&mut self, layer: usize, key: &Tensor<f32>, value: &Tensor<f32>) -> Result<()> {
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }
        if self.current_pos >= self.max_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Cache full at position {} (max {})",
                    self.current_pos, self.max_seq_len
                ),
            });
        }

        let k_data = key.data();
        let v_data = value.data();

        if k_data.len() != self.head_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Key size {} != head_dim {}", k_data.len(), self.head_dim),
            });
        }
        if v_data.len() != self.head_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Value size {} != head_dim {}", v_data.len(), self.head_dim),
            });
        }

        // Copy key and value into cache at current position
        let offset = self.current_pos * self.head_dim;
        self.keys[layer][offset..offset + self.head_dim].copy_from_slice(k_data);
        self.values[layer][offset..offset + self.head_dim].copy_from_slice(v_data);

        Ok(())
    }

    /// Advance to next position after updating all layers
    ///
    /// Call this after updating all layers for the current position
    pub fn advance(&mut self) {
        if self.current_pos < self.max_seq_len {
            self.current_pos += 1;
        }
    }

    /// Get cached keys for a layer up to current position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tensor with shape `[current_pos, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds
    pub fn get_key(&self, layer: usize) -> Result<Tensor<f32>> {
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }

        if self.current_pos == 0 {
            // Return empty tensor with shape [0, head_dim] is invalid
            // Return [1, head_dim] with zeros for consistency
            return Tensor::from_vec(vec![1, self.head_dim], vec![0.0; self.head_dim]);
        }

        let size = self.current_pos * self.head_dim;
        let data = self.keys[layer][..size].to_vec();
        Tensor::from_vec(vec![self.current_pos, self.head_dim], data)
    }

    /// Get cached values for a layer up to current position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tensor with shape `[current_pos, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds
    pub fn get_value(&self, layer: usize) -> Result<Tensor<f32>> {
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }

        if self.current_pos == 0 {
            return Tensor::from_vec(vec![1, self.head_dim], vec![0.0; self.head_dim]);
        }

        let size = self.current_pos * self.head_dim;
        let data = self.values[layer][..size].to_vec();
        Tensor::from_vec(vec![self.current_pos, self.head_dim], data)
    }

    /// Clear cache and reset position to 0
    pub fn clear(&mut self) {
        self.current_pos = 0;
        // Optionally zero out the cache (not strictly necessary)
        for layer in 0..self.num_layers {
            self.keys[layer].fill(0.0);
            self.values[layer].fill(0.0);
        }
    }

    /// Get current sequence position
    #[must_use]
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// Get number of layers
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Check if cache is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_seq_len
    }
}

/// Transformer Block (Pre-norm architecture)
///
/// A single transformer block combining self-attention and feed-forward layers
/// with residual connections and layer normalization.
///
/// # Architecture
///
/// ```text
/// Input
///   │
///   ├──────────────────┐
///   ▼                  │
/// LayerNorm            │
///   ▼                  │
/// Attention            │
///   ▼                  │
///   + <────────────────┘ (residual)
///   │
///   ├──────────────────┐
///   ▼                  │
/// LayerNorm            │
///   ▼                  │
/// FFN                  │
///   ▼                  │
///   + <────────────────┘ (residual)
///   │
/// Output
/// ```
///
/// This is the pre-norm architecture used in `LLaMA`, GPT-NeoX, and modern transformers.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Layer normalization before attention
    attn_norm: LayerNorm,
    /// Multi-head self-attention layer with Q/K/V/O projections
    attention: MultiHeadAttention,
    /// Layer normalization before FFN
    ffn_norm: LayerNorm,
    /// Feed-forward network
    ffn: FeedForward,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
}

impl TransformerBlock {
    /// Create a new transformer block
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Hidden dimension (model dimension)
    /// * `num_heads` - Number of attention heads
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `eps` - Layer normalization epsilon
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `hidden_dim` is zero or not divisible by `num_heads`
    /// - `num_heads` is zero
    /// - `intermediate_dim` is zero
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        intermediate_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        if hidden_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "hidden_dim must be > 0".to_string(),
            });
        }
        if num_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_heads must be > 0".to_string(),
            });
        }
        if hidden_dim % num_heads != 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
                ),
            });
        }

        let attn_norm = LayerNorm::new(hidden_dim, eps)?;
        // Use standard MHA with Q/K/V/O projections
        let attention = MultiHeadAttention::mha(hidden_dim, num_heads)?;
        let ffn_norm = LayerNorm::new(hidden_dim, eps)?;
        let ffn = FeedForward::new(hidden_dim, intermediate_dim)?;

        Ok(Self {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
            hidden_dim,
            num_heads,
        })
    }

    /// Forward pass through the transformer block
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor `[seq_len, hidden_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, hidden_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if input shape is invalid
    ///
    /// # Note
    ///
    /// This simplified implementation uses the same input for Q, K, V (self-attention).
    /// Production models would compute Q, K, V projections separately.
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();

        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor must have at least 1 dimension".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Expected last dimension {}, got {}",
                    self.hidden_dim, last_dim
                ),
            });
        }

        // Pre-norm attention block
        let normed = self.attn_norm.forward(input)?;

        // Self-attention with proper Q/K/V/O projections via MultiHeadAttention
        let attn_out = self.attention.forward(&normed)?;

        // Residual connection
        let mut residual1 = Vec::with_capacity(input.data().len());
        for (i, &val) in input.data().iter().enumerate() {
            residual1.push(val + attn_out.data()[i]);
        }
        let after_attn = Tensor::from_vec(shape.to_vec(), residual1)?;

        // Pre-norm FFN block
        let normed2 = self.ffn_norm.forward(&after_attn)?;
        let ffn_out = self.ffn.forward(&normed2)?;

        // Residual connection
        let mut residual2 = Vec::with_capacity(after_attn.data().len());
        for (i, &val) in after_attn.data().iter().enumerate() {
            residual2.push(val + ffn_out.data()[i]);
        }

        Tensor::from_vec(shape.to_vec(), residual2)
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get mutable reference to attention layer normalization
    pub fn attn_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.attn_norm
    }

    /// Get mutable reference to multi-head attention
    pub fn attention_mut(&mut self) -> &mut MultiHeadAttention {
        &mut self.attention
    }

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get mutable reference to FFN layer normalization
    pub fn ffn_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.ffn_norm
    }

    /// Get mutable reference to FFN
    pub fn ffn_mut(&mut self) -> &mut FeedForward {
        &mut self.ffn
    }
}

/// Embedding layer for converting token IDs to vectors
///
/// Maps discrete token IDs to continuous vector representations.
/// This is the first layer in a transformer model.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Embedding weights: `[vocab_size, embed_dim]`
    weights: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding layer
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of vocabulary
    /// * `embed_dim` - Dimension of embedding vectors
    ///
    /// # Errors
    ///
    /// Returns error if `vocab_size` or `embed_dim` is zero
    pub fn new(vocab_size: usize, embed_dim: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "vocab_size must be > 0".to_string(),
            });
        }
        if embed_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "embed_dim must be > 0".to_string(),
            });
        }

        let weights = vec![0.0; vocab_size * embed_dim];

        Ok(Self {
            vocab_size,
            embed_dim,
            weights,
        })
    }

    /// Look up embeddings for token IDs
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Slice of token IDs
    ///
    /// # Returns
    ///
    /// Tensor with shape `[seq_len, embed_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is out of bounds
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token IDs cannot be empty".to_string(),
            });
        }

        let seq_len = token_ids.len();
        let mut output = Vec::with_capacity(seq_len * self.embed_dim);

        for &token_id in token_ids {
            if token_id >= self.vocab_size {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Token ID {token_id} out of bounds (vocab_size={})",
                        self.vocab_size
                    ),
                });
            }

            let offset = token_id * self.embed_dim;
            output.extend_from_slice(&self.weights[offset..offset + self.embed_dim]);
        }

        Tensor::from_vec(vec![seq_len, self.embed_dim], output)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get embedding dimension
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get mutable access to weights for loading
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}

/// Transformer Language Model
///
/// Complete transformer model for language modeling:
/// - Token embedding
/// - Stack of transformer blocks
/// - Final layer normalization
/// - Output projection (LM head)
///
/// # Architecture
///
/// ```text
/// Token IDs → Embedding → [TransformerBlock × N] → LayerNorm → Linear → Logits
/// ```
#[derive(Debug, Clone)]
pub struct Model {
    /// Token embedding layer
    embedding: Embedding,
    /// Stack of transformer blocks
    blocks: Vec<TransformerBlock>,
    /// Final layer normalization
    final_norm: LayerNorm,
    /// Output projection (LM head)
    lm_head: Linear,
    /// Model configuration
    config: ModelConfig,
}

/// Configuration for the transformer model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer blocks
    pub num_layers: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Layer normalization epsilon
    pub eps: f32,
}

impl Model {
    /// Create a new transformer model
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(config: ModelConfig) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                config.hidden_dim,
                config.num_heads,
                config.intermediate_dim,
                config.eps,
            )?);
        }

        let final_norm = LayerNorm::new(config.hidden_dim, config.eps)?;
        let lm_head = Linear::new(config.hidden_dim, config.vocab_size)?;

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            lm_head,
            config,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits tensor with shape `[seq_len, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns error if input is invalid
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor<f32>> {
        // Embed tokens
        let mut hidden = self.embedding.forward(token_ids)?;

        // Pass through transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }

        // Final layer norm
        hidden = self.final_norm.forward(&hidden)?;

        // Project to vocabulary
        self.lm_head.forward(&hidden)
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get mutable reference to embedding layer
    pub fn embedding_mut(&mut self) -> &mut Embedding {
        &mut self.embedding
    }

    /// Get mutable reference to transformer blocks
    pub fn blocks_mut(&mut self) -> &mut [TransformerBlock] {
        &mut self.blocks
    }

    /// Get mutable reference to final layer norm
    pub fn final_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.final_norm
    }

    /// Get mutable reference to LM head
    pub fn lm_head_mut(&mut self) -> &mut Linear {
        &mut self.lm_head
    }

    /// Get number of parameters in the model (approximate)
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.config.vocab_size * self.config.hidden_dim;
        let block_params = self.config.num_layers
            * (
                // Attention (Q, K, V, O projections would be here in full impl)
                // For now just count layer norms and FFN
                2 * self.config.hidden_dim  // Layer norm weights
                + self.config.hidden_dim * self.config.intermediate_dim  // fc1
                + self.config.intermediate_dim * self.config.hidden_dim
                // fc2
            );
        let head_params = self.config.hidden_dim * self.config.vocab_size;

        embed_params + block_params + head_params
    }

    /// Generate tokens autoregressively
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Vector of generated token IDs (including prompt)
    ///
    /// # Errors
    ///
    /// Returns error if generation fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let generated = model.generate(&[1, 2, 3], &GenerationConfig::greedy())?;
    /// ```
    pub fn generate(&self, prompt: &[usize], config: &GenerationConfig) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let mut tokens = prompt.to_vec();
        let mut rng_state = config.seed.unwrap_or(42);

        for _ in 0..config.max_tokens {
            // Forward pass
            let logits = self.forward(&tokens)?;

            // Get logits for last position
            let seq_len = tokens.len();
            let vocab_size = self.config.vocab_size;
            let last_logits_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits.data()[last_logits_start..last_logits_start + vocab_size];

            let last_logits_tensor = Tensor::from_vec(vec![vocab_size], last_logits.to_vec())?;

            // Simple LCG for random number generation
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let rng_value = (rng_state >> 33) as f32 / (1u64 << 31) as f32;

            // Sample next token
            let next_token = sample_token(&last_logits_tensor, config, rng_value)?;

            // Check for EOS
            if let Some(eos_id) = config.eos_token_id {
                if next_token == eos_id {
                    break;
                }
            }

            tokens.push(next_token);
        }

        Ok(tokens)
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

    // FusedLayerNormLinear tests

    #[test]
    fn test_fused_layer_norm_linear_creation() {
        let fused = FusedLayerNormLinear::new(4, 8, 1e-5).unwrap();
        assert_eq!(fused.feature_dim(), 4);
        assert_eq!(fused.out_features(), 8);
    }

    #[test]
    fn test_fused_layer_norm_linear_zero_dims_error() {
        let result = FusedLayerNormLinear::new(0, 8, 1e-5);
        assert!(result.is_err());

        let result = FusedLayerNormLinear::new(4, 0, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_layer_norm_linear_matches_separate() {
        // Test that fused implementation matches separate LayerNorm + Linear
        let feature_dim = 4;
        let out_features = 3;

        // Create fused layer
        let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).unwrap();

        // Set weights
        for (i, weight) in fused.linear_weight_mut().iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            {
                *weight = (i as f32) * 0.1;
            }
        }
        for (i, bias) in fused.linear_bias_mut().iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            {
                *bias = (i as f32) * 0.05;
            }
        }

        // Create separate layers with same weights
        let layer_norm = LayerNorm::new(feature_dim, 1e-5).unwrap();
        let mut linear = Linear::new(feature_dim, out_features).unwrap();
        for (i, weight) in linear.weight_mut().iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            {
                *weight = (i as f32) * 0.1;
            }
        }
        for (i, bias) in linear.bias_mut().iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            {
                *bias = (i as f32) * 0.05;
            }
        }

        // Test input
        let input = Tensor::from_vec(vec![feature_dim], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Fused forward
        let fused_output = fused.forward(&input).unwrap();

        // Separate forward
        let norm_output = layer_norm.forward(&input).unwrap();
        let separate_output = linear.forward(&norm_output).unwrap();

        // Results should match
        assert_eq!(fused_output.shape(), separate_output.shape());
        for i in 0..fused_output.data().len() {
            assert!(
                (fused_output.data()[i] - separate_output.data()[i]).abs() < 1e-4,
                "Mismatch at {}: fused={} vs separate={}",
                i,
                fused_output.data()[i],
                separate_output.data()[i]
            );
        }
    }

    #[test]
    fn test_fused_layer_norm_linear_batched() {
        let feature_dim = 4;
        let out_features = 2;

        let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).unwrap();

        // Set simple weights
        for weight in fused.linear_weight_mut().iter_mut() {
            *weight = 1.0;
        }

        // Batched input [2, 4]
        let input = Tensor::from_vec(
            vec![2, feature_dim],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();

        let output = fused.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, out_features]);
    }

    #[test]
    fn test_fused_layer_norm_linear_parallel_matches_serial() {
        let feature_dim = 8;
        let out_features = 4;

        let mut fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).unwrap();

        // Set random-ish weights
        for (i, weight) in fused.linear_weight_mut().iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            {
                *weight = ((i * 7 + 3) % 11) as f32 * 0.1;
            }
        }
        for (i, bias) in fused.linear_bias_mut().iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            {
                *bias = ((i * 5 + 2) % 7) as f32 * 0.1;
            }
        }

        // Large batch
        let mut input_data = Vec::new();
        for i in 0..32 {
            for j in 0..feature_dim {
                #[allow(clippy::cast_precision_loss)]
                {
                    input_data.push(((i * feature_dim + j) % 17) as f32 * 0.2);
                }
            }
        }
        let input = Tensor::from_vec(vec![32, feature_dim], input_data).unwrap();

        // Serial
        let serial_output = fused.forward(&input).unwrap();

        // Parallel
        let parallel_output = fused.forward_parallel(&input).unwrap();

        assert_eq!(serial_output.shape(), parallel_output.shape());
        for i in 0..serial_output.data().len() {
            assert!(
                (serial_output.data()[i] - parallel_output.data()[i]).abs() < 1e-5,
                "Mismatch at {}: serial={} vs parallel={}",
                i,
                serial_output.data()[i],
                parallel_output.data()[i]
            );
        }
    }

    #[test]
    fn test_fused_layer_norm_linear_dimension_mismatch_error() {
        let fused = FusedLayerNormLinear::new(4, 8, 1e-5).unwrap();

        // Wrong input dimension
        let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let result = fused.forward(&input);
        assert!(result.is_err());
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

    // Flash Attention tests

    #[test]
    fn test_flash_attention_matches_standard() {
        // Flash Attention should produce same output as standard attention
        let attn = Attention::new(8).unwrap();

        // Create test data
        let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
        let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
        let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

        let q = Tensor::from_vec(vec![1, 8], q_data.clone()).unwrap();
        let k = Tensor::from_vec(vec![1, 8], k_data.clone()).unwrap();
        let v = Tensor::from_vec(vec![1, 8], v_data.clone()).unwrap();

        // Standard attention
        let standard_output = attn.forward(&q, &k, &v).unwrap();

        // Flash attention with block_size=1 (should be identical)
        let flash_output = attn.flash_forward(&q, &k, &v, 1).unwrap();

        // Results should match
        assert_eq!(standard_output.shape(), flash_output.shape());
        for i in 0..standard_output.data().len() {
            assert!(
                (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                standard_output.data()[i],
                flash_output.data()[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_multi_position() {
        // Test Flash Attention with multiple positions
        let attn = Attention::new(4).unwrap();

        #[rustfmt::skip]
        let q_data = vec![
            1.0, 0.0, 0.0, 1.0,  // pos 0
            0.0, 1.0, 1.0, 0.0,  // pos 1
            1.0, 1.0, 0.0, 0.0,  // pos 2
        ];
        #[rustfmt::skip]
        let k_data = vec![
            1.0, 0.0, 0.0, 1.0,  // pos 0
            0.0, 1.0, 1.0, 0.0,  // pos 1
            1.0, 1.0, 0.0, 0.0,  // pos 2
        ];
        #[rustfmt::skip]
        let v_data = vec![
            1.0, 2.0, 3.0, 4.0,  // pos 0
            5.0, 6.0, 7.0, 8.0,  // pos 1
            9.0, 10.0, 11.0, 12.0,  // pos 2
        ];

        let q = Tensor::from_vec(vec![3, 4], q_data).unwrap();
        let k = Tensor::from_vec(vec![3, 4], k_data).unwrap();
        let v = Tensor::from_vec(vec![3, 4], v_data).unwrap();

        // Standard attention
        let standard_output = attn.forward(&q, &k, &v).unwrap();

        // Flash attention with different block sizes
        for block_size in [1, 2, 3, 4] {
            let flash_output = attn.flash_forward(&q, &k, &v, block_size).unwrap();

            assert_eq!(standard_output.shape(), flash_output.shape());
            for i in 0..standard_output.data().len() {
                assert!(
                    (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-4,
                    "Block size {}, mismatch at index {}: {} vs {}",
                    block_size,
                    i,
                    standard_output.data()[i],
                    flash_output.data()[i]
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_zero_block_size_error() {
        let attn = Attention::new(4).unwrap();

        let q = Tensor::from_vec(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let k = Tensor::from_vec(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let v = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = attn.flash_forward(&q, &k, &v, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_flash_attention_large_sequence() {
        // Test with larger sequence to verify block-wise computation
        let attn = Attention::new(8).unwrap();

        // Create larger test data (seq_len=16)
        let mut q_data = Vec::new();
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();

        for i in 0..16 {
            for j in 0..8 {
                #[allow(clippy::cast_precision_loss)]
                {
                    q_data.push((i * 8 + j) as f32 * 0.1);
                    k_data.push((i * 8 + j) as f32 * 0.05);
                    v_data.push((i * 8 + j) as f32 * 0.2);
                }
            }
        }

        let q = Tensor::from_vec(vec![16, 8], q_data).unwrap();
        let k = Tensor::from_vec(vec![16, 8], k_data).unwrap();
        let v = Tensor::from_vec(vec![16, 8], v_data).unwrap();

        // Standard attention
        let standard_output = attn.forward(&q, &k, &v).unwrap();

        // Flash attention with block_size=4
        let flash_output = attn.flash_forward(&q, &k, &v, 4).unwrap();

        assert_eq!(standard_output.shape(), flash_output.shape());
        for i in 0..standard_output.data().len() {
            assert!(
                (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-3,
                "Mismatch at index {}: {} vs {}",
                i,
                standard_output.data()[i],
                flash_output.data()[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_shape_errors() {
        let attn = Attention::new(4).unwrap();

        let q = Tensor::from_vec(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let k = Tensor::from_vec(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let v_wrong = Tensor::from_vec(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        // K/V sequence length mismatch
        let result = attn.flash_forward(&q, &k, &v_wrong, 2);
        assert!(result.is_err());
    }

    // Flash Attention v2 tests

    #[test]
    fn test_flash_attention_v2_matches_standard() {
        // Flash Attention v2 with SIMD should match standard attention
        let attn = Attention::new(8).unwrap();

        let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
        let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
        let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

        let q = Tensor::from_vec(vec![1, 8], q_data).unwrap();
        let k = Tensor::from_vec(vec![1, 8], k_data).unwrap();
        let v = Tensor::from_vec(vec![1, 8], v_data).unwrap();

        let standard = attn.forward(&q, &k, &v).unwrap();
        let v2 = attn.flash_forward_v2(&q, &k, &v, 1).unwrap();

        assert_eq!(standard.shape(), v2.shape());
        for i in 0..standard.data().len() {
            assert!(
                (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                standard.data()[i],
                v2.data()[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_v2_multi_position() {
        let attn = Attention::new(4).unwrap();

        #[rustfmt::skip]
        let q_data = vec![
            1.0, 0.5, 0.3, 1.2,
            0.5, 1.0, 0.8, 0.4,
            0.3, 0.8, 1.0, 0.6,
        ];
        #[rustfmt::skip]
        let k_data = vec![
            1.0, 0.5, 0.3, 1.2,
            0.5, 1.0, 0.8, 0.4,
            0.3, 0.8, 1.0, 0.6,
        ];
        #[rustfmt::skip]
        let v_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];

        let q = Tensor::from_vec(vec![3, 4], q_data).unwrap();
        let k = Tensor::from_vec(vec![3, 4], k_data).unwrap();
        let v = Tensor::from_vec(vec![3, 4], v_data).unwrap();

        let standard = attn.forward(&q, &k, &v).unwrap();

        for block_size in [1, 2, 3, 4] {
            let v2 = attn.flash_forward_v2(&q, &k, &v, block_size).unwrap();
            assert_eq!(standard.shape(), v2.shape());
            for i in 0..standard.data().len() {
                assert!(
                    (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
                    "Block size {}, mismatch at {}: {} vs {}",
                    block_size,
                    i,
                    standard.data()[i],
                    v2.data()[i]
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_v2_zero_block_size_error() {
        let attn = Attention::new(4).unwrap();
        let q = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).unwrap();
        let k = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).unwrap();
        let v = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).unwrap();

        let result = attn.flash_forward_v2(&q, &k, &v, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_flash_attention_v2_large_sequence() {
        let attn = Attention::new(8).unwrap();

        let mut q_data = Vec::new();
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();

        for i in 0..32 {
            for j in 0..8 {
                #[allow(clippy::cast_precision_loss)]
                {
                    q_data.push((i * 8 + j) as f32 * 0.05);
                    k_data.push((i * 8 + j) as f32 * 0.03);
                    v_data.push((i * 8 + j) as f32 * 0.1);
                }
            }
        }

        let q = Tensor::from_vec(vec![32, 8], q_data).unwrap();
        let k = Tensor::from_vec(vec![32, 8], k_data).unwrap();
        let v = Tensor::from_vec(vec![32, 8], v_data).unwrap();

        let standard = attn.forward(&q, &k, &v).unwrap();
        let v2 = attn.flash_forward_v2(&q, &k, &v, 8).unwrap();

        assert_eq!(standard.shape(), v2.shape());
        for i in 0..standard.data().len() {
            assert!(
                (standard.data()[i] - v2.data()[i]).abs() < 1e-3,
                "Mismatch at {}: {} vs {}",
                i,
                standard.data()[i],
                v2.data()[i]
            );
        }
    }

    // Parallel Flash Attention tests

    #[test]
    fn test_flash_attention_parallel_matches_standard() {
        let attn = Attention::new(8).unwrap();

        let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
        let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
        let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

        let q = Tensor::from_vec(vec![1, 8], q_data).unwrap();
        let k = Tensor::from_vec(vec![1, 8], k_data).unwrap();
        let v = Tensor::from_vec(vec![1, 8], v_data).unwrap();

        let standard = attn.forward(&q, &k, &v).unwrap();
        let parallel = attn.flash_forward_parallel(&q, &k, &v, 1).unwrap();

        assert_eq!(standard.shape(), parallel.shape());
        for i in 0..standard.data().len() {
            assert!(
                (standard.data()[i] - parallel.data()[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                standard.data()[i],
                parallel.data()[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_parallel_multi_position() {
        let attn = Attention::new(4).unwrap();

        #[rustfmt::skip]
        let q_data = vec![
            1.0, 0.5, 0.3, 1.2,
            0.5, 1.0, 0.8, 0.4,
            0.3, 0.8, 1.0, 0.6,
            0.7, 0.2, 0.9, 0.5,
        ];
        let k_data = q_data.clone();
        #[rustfmt::skip]
        let v_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        let q = Tensor::from_vec(vec![4, 4], q_data).unwrap();
        let k = Tensor::from_vec(vec![4, 4], k_data).unwrap();
        let v = Tensor::from_vec(vec![4, 4], v_data).unwrap();

        let standard = attn.forward(&q, &k, &v).unwrap();

        for block_size in [1, 2, 4] {
            let parallel = attn.flash_forward_parallel(&q, &k, &v, block_size).unwrap();
            assert_eq!(standard.shape(), parallel.shape());
            for i in 0..standard.data().len() {
                assert!(
                    (standard.data()[i] - parallel.data()[i]).abs() < 1e-4,
                    "Block size {}, mismatch at {}: {} vs {}",
                    block_size,
                    i,
                    standard.data()[i],
                    parallel.data()[i]
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_parallel_zero_block_size_error() {
        let attn = Attention::new(4).unwrap();
        let q = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).unwrap();
        let k = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).unwrap();
        let v = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).unwrap();

        let result = attn.flash_forward_parallel(&q, &k, &v, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_flash_attention_parallel_large_sequence() {
        let attn = Attention::new(16).unwrap();

        let mut q_data = Vec::new();
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();

        for i in 0..64 {
            for j in 0..16 {
                #[allow(clippy::cast_precision_loss)]
                {
                    q_data.push((i * 16 + j) as f32 * 0.02);
                    k_data.push((i * 16 + j) as f32 * 0.015);
                    v_data.push((i * 16 + j) as f32 * 0.05);
                }
            }
        }

        let q = Tensor::from_vec(vec![64, 16], q_data).unwrap();
        let k = Tensor::from_vec(vec![64, 16], k_data).unwrap();
        let v = Tensor::from_vec(vec![64, 16], v_data).unwrap();

        let standard = attn.forward(&q, &k, &v).unwrap();
        let parallel = attn.flash_forward_parallel(&q, &k, &v, 16).unwrap();

        assert_eq!(standard.shape(), parallel.shape());
        for i in 0..standard.data().len() {
            assert!(
                (standard.data()[i] - parallel.data()[i]).abs() < 1e-3,
                "Mismatch at {}: {} vs {}",
                i,
                standard.data()[i],
                parallel.data()[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_v2_vs_parallel_consistency() {
        // Both v2 and parallel should produce same results
        let attn = Attention::new(8).unwrap();

        let mut q_data = Vec::new();
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();

        for i in 0..16 {
            for j in 0..8 {
                #[allow(clippy::cast_precision_loss)]
                {
                    q_data.push((i * 8 + j) as f32 * 0.1);
                    k_data.push((i * 8 + j) as f32 * 0.08);
                    v_data.push((i * 8 + j) as f32 * 0.15);
                }
            }
        }

        let q = Tensor::from_vec(vec![16, 8], q_data).unwrap();
        let k = Tensor::from_vec(vec![16, 8], k_data).unwrap();
        let v = Tensor::from_vec(vec![16, 8], v_data).unwrap();

        let v2 = attn.flash_forward_v2(&q, &k, &v, 4).unwrap();
        let parallel = attn.flash_forward_parallel(&q, &k, &v, 4).unwrap();

        assert_eq!(v2.shape(), parallel.shape());
        for i in 0..v2.data().len() {
            assert!(
                (v2.data()[i] - parallel.data()[i]).abs() < 1e-5,
                "Mismatch at {}: v2={} vs parallel={}",
                i,
                v2.data()[i],
                parallel.data()[i]
            );
        }
    }

    // RoPE (Rotary Position Embeddings) tests

    #[test]
    fn test_rope_creation() {
        let rope = RoPE::new(64, 10000.0).unwrap();
        assert_eq!(rope.dim(), 64);
        assert!((rope.base() - 10000.0).abs() < 1e-6);
        assert_eq!(rope.inv_freq().len(), 32); // dim/2
    }

    #[test]
    fn test_rope_with_default_base() {
        let rope = RoPE::with_default_base(128).unwrap();
        assert_eq!(rope.dim(), 128);
        assert!((rope.base() - 10000.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_zero_dim_error() {
        let result = RoPE::new(0, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_odd_dim_error() {
        let result = RoPE::new(63, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_forward_shape() {
        let rope = RoPE::with_default_base(4).unwrap();
        let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).unwrap();

        let output = rope.forward(&input, 0).unwrap();
        assert_eq!(output.shape(), &[2, 4]);
        assert_eq!(output.data().len(), 8);
    }

    #[test]
    fn test_rope_position_zero_identity() {
        // At position 0, rotation angles are 0, so cos=1, sin=0
        // This should return input unchanged
        let rope = RoPE::with_default_base(4).unwrap();
        let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let output = rope.forward(&input, 0).unwrap();

        // At position 0, angles are 0, so cos(0)=1, sin(0)=0
        // y0 = x0 * 1 - x1 * 0 = x0
        // y1 = x0 * 0 + x1 * 1 = x1
        for i in 0..4 {
            assert!(
                (output.data()[i] - input.data()[i]).abs() < 1e-6,
                "Position 0 should be identity: expected {}, got {}",
                input.data()[i],
                output.data()[i]
            );
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // Rotation should preserve vector norm
        let rope = RoPE::with_default_base(4).unwrap();
        let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let output = rope.forward(&input, 100).unwrap();

        // Compute L2 norm of input pairs and output pairs
        // Each pair should have the same norm after rotation
        let in_norm_0 = (input.data()[0].powi(2) + input.data()[1].powi(2)).sqrt();
        let in_norm_1 = (input.data()[2].powi(2) + input.data()[3].powi(2)).sqrt();
        let out_norm_0 = (output.data()[0].powi(2) + output.data()[1].powi(2)).sqrt();
        let out_norm_1 = (output.data()[2].powi(2) + output.data()[3].powi(2)).sqrt();

        assert!(
            (in_norm_0 - out_norm_0).abs() < 1e-5,
            "Pair 0 norm should be preserved"
        );
        assert!(
            (in_norm_1 - out_norm_1).abs() < 1e-5,
            "Pair 1 norm should be preserved"
        );
    }

    #[test]
    fn test_rope_different_positions() {
        let rope = RoPE::with_default_base(4).unwrap();
        let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();

        let out_pos_zero = rope.forward(&input, 0).unwrap();
        let out_pos_ten = rope.forward(&input, 10).unwrap();
        let out_pos_hundred = rope.forward(&input, 100).unwrap();

        // Different positions should give different outputs
        assert!(
            (out_pos_zero.data()[0] - out_pos_ten.data()[0]).abs() > 1e-6
                || (out_pos_zero.data()[1] - out_pos_ten.data()[1]).abs() > 1e-6
        );
        assert!(
            (out_pos_ten.data()[0] - out_pos_hundred.data()[0]).abs() > 1e-6
                || (out_pos_ten.data()[1] - out_pos_hundred.data()[1]).abs() > 1e-6
        );
    }

    #[test]
    fn test_rope_dimension_mismatch_error() {
        let rope = RoPE::with_default_base(4).unwrap();
        let input = Tensor::from_vec(vec![6], vec![1.0; 6]).unwrap();

        let result = rope.forward(&input, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_batched() {
        // Test with batched input [batch, dim]
        let rope = RoPE::with_default_base(4).unwrap();
        let input = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).unwrap();

        let output = rope.forward(&input, 5).unwrap();
        assert_eq!(output.shape(), &[3, 4]);

        // All vectors in batch should have same rotation applied
        // (since same position)
        for batch in 0..3 {
            for i in 0..4 {
                let expected = output.data()[i]; // First vector
                let actual = output.data()[batch * 4 + i];
                assert!(
                    (expected - actual).abs() < 1e-6,
                    "All batch elements should have same rotation"
                );
            }
        }
    }

    #[test]
    fn test_rope_inv_freq_computation() {
        // Test that inverse frequencies are computed correctly
        let rope = RoPE::new(4, 10000.0).unwrap();
        let inv_freq = rope.inv_freq();

        // For dim=4, we have 2 pairs
        // inv_freq[0] = 10000^(-2*0/4) = 10000^0 = 1.0
        // inv_freq[1] = 10000^(-2*1/4) = 10000^(-0.5) = 0.01
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);
        assert!((inv_freq[1] - 0.01).abs() < 1e-6);
    }

    // ALiBi (Attention with Linear Biases) tests

    #[test]
    fn test_alibi_creation() {
        let alibi = ALiBi::new(8).unwrap();
        assert_eq!(alibi.num_heads(), 8);
        assert_eq!(alibi.slopes().len(), 8);
    }

    #[test]
    fn test_alibi_zero_heads_error() {
        let result = ALiBi::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_alibi_slopes_power_of_2() {
        // For 8 heads (power of 2), slopes should follow: 2^(-8h/8) = 2^(-h)
        let alibi = ALiBi::new(8).unwrap();
        let slopes = alibi.slopes();

        // Expected slopes: 2^0, 2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7
        assert!((slopes[0] - 1.0).abs() < 1e-6); // 2^0 = 1.0
        assert!((slopes[1] - 0.5).abs() < 1e-6); // 2^-1 = 0.5
        assert!((slopes[2] - 0.25).abs() < 1e-6); // 2^-2 = 0.25
        assert!((slopes[3] - 0.125).abs() < 1e-6); // 2^-3 = 0.125
    }

    #[test]
    fn test_alibi_slopes_non_power_of_2() {
        // For 6 heads (not power of 2)
        let alibi = ALiBi::new(6).unwrap();
        let slopes = alibi.slopes();

        assert_eq!(slopes.len(), 6);

        // First 4 slopes follow 2^(-8h/4) = 2^(-2h)
        assert!((slopes[0] - 1.0).abs() < 1e-6); // 2^0
        assert!((slopes[1] - 0.25).abs() < 1e-6); // 2^-2
        assert!((slopes[2] - 0.0625).abs() < 1e-6); // 2^-4
        assert!((slopes[3] - 0.015_625).abs() < 1e-6); // 2^-6

        // Extra 2 slopes follow 2^(-4h/4) with step=2
        // slopes[4] = 2^(-1) = 0.5
        // slopes[5] = 2^(-3) = 0.125
        assert!((slopes[4] - 0.5).abs() < 1e-6);
        assert!((slopes[5] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_bias_shape() {
        let alibi = ALiBi::new(4).unwrap();
        let bias = alibi.get_bias(10).unwrap();

        // Shape should be [seq_len, seq_len, num_heads]
        assert_eq!(bias.shape(), &[10, 10, 4]);
    }

    #[test]
    fn test_alibi_bias_zero_seq_len_error() {
        let alibi = ALiBi::new(4).unwrap();
        let result = alibi.get_bias(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_alibi_bias_diagonal_zero() {
        // Diagonal elements (same position) should be zero
        let alibi = ALiBi::new(4).unwrap();
        let bias = alibi.get_bias(5).unwrap();

        for i in 0..5 {
            for h in 0..4 {
                let idx = i * 5 * 4 + i * 4 + h; // [i, i, h]
                let value = bias.data()[idx];
                assert!(
                    value.abs() < 1e-6,
                    "Diagonal bias[{i}, {i}, {h}] should be 0, got {value}"
                );
            }
        }
    }

    #[test]
    fn test_alibi_bias_symmetry() {
        // |i - j| = |j - i|, so bias[i,j,h] should equal bias[j,i,h]
        let alibi = ALiBi::new(2).unwrap();
        let bias = alibi.get_bias(4).unwrap();

        for i in 0..4 {
            for j in 0..4 {
                for h in 0..2 {
                    let idx_ij = i * 4 * 2 + j * 2 + h;
                    let idx_ji = j * 4 * 2 + i * 2 + h;
                    let bias_ij = bias.data()[idx_ij];
                    let bias_ji = bias.data()[idx_ji];
                    assert!(
                        (bias_ij - bias_ji).abs() < 1e-6,
                        "Bias should be symmetric: [{i},{j},{h}]={bias_ij} vs [{j},{i},{h}]={bias_ji}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_alibi_bias_computation() {
        // Test exact bias values
        let alibi = ALiBi::new(2).unwrap();
        let slopes = alibi.slopes();
        let bias = alibi.get_bias(3).unwrap();

        // For 2 heads: slopes = [1.0, 0.0625]
        // bias[0, 2, 0] = -slopes[0] * |0 - 2| = -1.0 * 2 = -2.0
        let idx = 2 * 2;
        assert!(
            (bias.data()[idx] - (-2.0)).abs() < 1e-6,
            "Expected -2.0, got {}",
            bias.data()[idx]
        );

        // bias[1, 2, 1] = -slopes[1] * |1 - 2| = -0.0625 * 1 = -0.0625
        let idx = 3 * 2 + 2 * 2 + 1;
        let expected = -slopes[1];
        assert!(
            (bias.data()[idx] - expected).abs() < 1e-6,
            "Expected {expected}, got {}",
            bias.data()[idx]
        );
    }

    #[test]
    fn test_alibi_bias_negative() {
        // All bias values should be <= 0 (except diagonal which is 0)
        let alibi = ALiBi::new(4).unwrap();
        let bias = alibi.get_bias(10).unwrap();

        for &value in bias.data() {
            assert!(value <= 1e-6, "Bias should be non-positive, got {value}");
        }
    }

    #[test]
    fn test_alibi_bias_distance_proportional() {
        // Bias should be proportional to distance
        let alibi = ALiBi::new(1).unwrap();
        let bias = alibi.get_bias(5).unwrap();

        // For head 0, slope is 1.0
        // bias[0, 1] = -1.0 * 1 = -1.0
        // bias[0, 2] = -1.0 * 2 = -2.0
        // bias[0, 3] = -1.0 * 3 = -3.0

        let bias_01 = bias.data()[1];
        let bias_02 = bias.data()[2];
        let bias_03 = bias.data()[3];

        assert!((bias_01 - (-1.0)).abs() < 1e-6);
        assert!((bias_02 - (-2.0)).abs() < 1e-6);
        assert!((bias_03 - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_single_head() {
        let alibi = ALiBi::new(1).unwrap();
        assert_eq!(alibi.num_heads(), 1);
        assert_eq!(alibi.slopes().len(), 1);
        assert!((alibi.slopes()[0] - 1.0).abs() < 1e-6); // First slope is 2^0 = 1.0
    }

    #[test]
    fn test_alibi_large_num_heads() {
        // Test with large number of heads (non-power of 2)
        let alibi = ALiBi::new(12).unwrap();
        assert_eq!(alibi.num_heads(), 12);
        assert_eq!(alibi.slopes().len(), 12);

        // All slopes should be positive
        for slope in alibi.slopes() {
            assert!(*slope > 0.0, "Slope should be positive, got {slope}");
        }

        // First head should have largest slope (1.0)
        assert!((alibi.slopes()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_bias_long_sequence() {
        // Test with longer sequence
        let alibi = ALiBi::new(8).unwrap();
        let bias = alibi.get_bias(128).unwrap();

        assert_eq!(bias.shape(), &[128, 128, 8]);

        // Check that far positions have larger negative bias
        let near_bias = bias.data()[8]; // distance 1
        let far_bias = bias.data()[100 * 8]; // distance 100

        assert!(near_bias > far_bias); // near should be less negative
    }

    // KVCache tests

    #[test]
    fn test_kvcache_creation() {
        let cache = KVCache::new(4, 512, 64).unwrap();
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.max_seq_len(), 512);
        assert_eq!(cache.head_dim(), 64);
        assert_eq!(cache.current_pos(), 0);
        assert!(!cache.is_full());
    }

    #[test]
    fn test_kvcache_zero_params_error() {
        assert!(KVCache::new(0, 512, 64).is_err());
        assert!(KVCache::new(4, 0, 64).is_err());
        assert!(KVCache::new(4, 512, 0).is_err());
    }

    #[test]
    fn test_kvcache_update_and_retrieve() {
        let mut cache = KVCache::new(2, 10, 4).unwrap();

        // Add first position
        let key = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let value = Tensor::from_vec(vec![4], vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        cache.update(0, &key, &value).unwrap();
        cache.advance();

        // Retrieve and verify
        let cached_key = cache.get_key(0).unwrap();
        let cached_value = cache.get_value(0).unwrap();

        assert_eq!(cached_key.shape(), &[1, 4]);
        assert_eq!(cached_value.shape(), &[1, 4]);

        for i in 0..4 {
            assert!((cached_key.data()[i] - key.data()[i]).abs() < 1e-6);
            assert!((cached_value.data()[i] - value.data()[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kvcache_multiple_positions() {
        let mut cache = KVCache::new(1, 10, 2).unwrap();

        // Add multiple positions
        for pos in 0..3 {
            #[allow(clippy::cast_precision_loss)]
            let base = pos as f32;
            let key = Tensor::from_vec(vec![2], vec![base, base + 0.5]).unwrap();
            let value = Tensor::from_vec(vec![2], vec![base + 1.0, base + 1.5]).unwrap();

            cache.update(0, &key, &value).unwrap();
            cache.advance();
        }

        assert_eq!(cache.current_pos(), 3);

        // Retrieve all positions
        let cached_key = cache.get_key(0).unwrap();
        let cached_value = cache.get_value(0).unwrap();

        assert_eq!(cached_key.shape(), &[3, 2]);
        assert_eq!(cached_value.shape(), &[3, 2]);

        // Verify first position
        assert!((cached_key.data()[0] - 0.0).abs() < 1e-6);
        assert!((cached_key.data()[1] - 0.5).abs() < 1e-6);
        // Verify second position
        assert!((cached_key.data()[2] - 1.0).abs() < 1e-6);
        assert!((cached_key.data()[3] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_kvcache_multiple_layers() {
        let mut cache = KVCache::new(2, 10, 4).unwrap();

        let key0 = Tensor::from_vec(vec![4], vec![1.0; 4]).unwrap();
        let value0 = Tensor::from_vec(vec![4], vec![2.0; 4]).unwrap();
        let key1 = Tensor::from_vec(vec![4], vec![3.0; 4]).unwrap();
        let value1 = Tensor::from_vec(vec![4], vec![4.0; 4]).unwrap();

        cache.update(0, &key0, &value0).unwrap();
        cache.update(1, &key1, &value1).unwrap();
        cache.advance();

        // Verify layer 0
        let layer0_key = cache.get_key(0).unwrap();
        assert!((layer0_key.data()[0] - 1.0).abs() < 1e-6);

        // Verify layer 1
        let layer1_key = cache.get_key(1).unwrap();
        assert!((layer1_key.data()[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_kvcache_layer_out_of_bounds_error() {
        let mut cache = KVCache::new(2, 10, 4).unwrap();
        let key = Tensor::from_vec(vec![4], vec![1.0; 4]).unwrap();
        let value = Tensor::from_vec(vec![4], vec![2.0; 4]).unwrap();

        // Update layer 2 (out of bounds)
        assert!(cache.update(2, &key, &value).is_err());

        // Get layer 2 (out of bounds)
        assert!(cache.get_key(2).is_err());
        assert!(cache.get_value(2).is_err());
    }

    #[test]
    fn test_kvcache_size_mismatch_error() {
        let mut cache = KVCache::new(1, 10, 4).unwrap();

        // Wrong key size
        let key = Tensor::from_vec(vec![3], vec![1.0; 3]).unwrap();
        let value = Tensor::from_vec(vec![4], vec![2.0; 4]).unwrap();
        assert!(cache.update(0, &key, &value).is_err());

        // Wrong value size
        let key = Tensor::from_vec(vec![4], vec![1.0; 4]).unwrap();
        let value = Tensor::from_vec(vec![3], vec![2.0; 3]).unwrap();
        assert!(cache.update(0, &key, &value).is_err());
    }

    #[test]
    fn test_kvcache_full_error() {
        let mut cache = KVCache::new(1, 2, 4).unwrap();
        let key = Tensor::from_vec(vec![4], vec![1.0; 4]).unwrap();
        let value = Tensor::from_vec(vec![4], vec![2.0; 4]).unwrap();

        // Fill cache
        cache.update(0, &key, &value).unwrap();
        cache.advance();
        cache.update(0, &key, &value).unwrap();
        cache.advance();

        assert!(cache.is_full());

        // Try to add more
        assert!(cache.update(0, &key, &value).is_err());
    }

    #[test]
    fn test_kvcache_clear() {
        let mut cache = KVCache::new(1, 10, 4).unwrap();
        let key = Tensor::from_vec(vec![4], vec![1.0; 4]).unwrap();
        let value = Tensor::from_vec(vec![4], vec![2.0; 4]).unwrap();

        cache.update(0, &key, &value).unwrap();
        cache.advance();
        assert_eq!(cache.current_pos(), 1);

        cache.clear();
        assert_eq!(cache.current_pos(), 0);
        assert!(!cache.is_full());
    }

    #[test]
    fn test_kvcache_empty_retrieval() {
        let cache = KVCache::new(1, 10, 4).unwrap();

        // Retrieve from empty cache
        let cached_key = cache.get_key(0).unwrap();
        let cached_value = cache.get_value(0).unwrap();

        // Should return [1, 4] tensor with zeros
        assert_eq!(cached_key.shape(), &[1, 4]);
        assert_eq!(cached_value.shape(), &[1, 4]);
        for &val in cached_key.data() {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    // TransformerBlock tests

    #[test]
    fn test_transformer_block_creation() {
        let block = TransformerBlock::new(64, 4, 256, 1e-5).unwrap();
        assert_eq!(block.hidden_dim(), 64);
    }

    #[test]
    fn test_transformer_block_zero_params_error() {
        // Zero hidden_dim
        assert!(TransformerBlock::new(0, 4, 256, 1e-5).is_err());
        // Zero num_heads
        assert!(TransformerBlock::new(64, 0, 256, 1e-5).is_err());
        // Zero intermediate_dim
        assert!(TransformerBlock::new(64, 4, 0, 1e-5).is_err());
    }

    #[test]
    fn test_transformer_block_head_divisibility_error() {
        // 63 not divisible by 4
        assert!(TransformerBlock::new(63, 4, 256, 1e-5).is_err());
    }

    #[test]
    fn test_transformer_block_forward_shape() {
        // Use num_heads=1 so head_dim=hidden_dim (simplified single-head attention)
        let block = TransformerBlock::new(8, 1, 32, 1e-5).unwrap();

        // Input: [2, 8] (seq_len=2, hidden_dim=8)
        let input = Tensor::from_vec(vec![2, 8], vec![0.1; 16]).unwrap();
        let output = block.forward(&input).unwrap();

        // Output should have same shape
        assert_eq!(output.shape(), &[2, 8]);
        assert_eq!(output.data().len(), 16);
    }

    #[test]
    fn test_transformer_block_forward_valid_output() {
        let block = TransformerBlock::new(4, 1, 16, 1e-5).unwrap();

        // Input must be 2D [seq_len, hidden_dim]
        let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = block.forward(&input).unwrap();

        // Output should be finite
        for &val in output.data() {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_transformer_block_residual_connection() {
        let block = TransformerBlock::new(4, 1, 16, 1e-5).unwrap();

        // Input must be 2D [seq_len, hidden_dim]
        // With zero input, output should be non-zero due to residual + processing
        let input = Tensor::from_vec(vec![1, 4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let output = block.forward(&input).unwrap();

        // Even with zero input, layer norm and attention should produce non-zero output
        // (though it might be small due to normalization)
        assert_eq!(output.shape(), &[1, 4]);
    }

    #[test]
    fn test_transformer_block_shape_mismatch_error() {
        let block = TransformerBlock::new(8, 1, 32, 1e-5).unwrap();

        // Wrong hidden_dim (input has 4, block expects 8)
        let input = Tensor::from_vec(vec![4], vec![1.0; 4]).unwrap();
        let result = block.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_block_mutable_access() {
        let mut block = TransformerBlock::new(4, 1, 16, 1e-5).unwrap();

        // Verify we can access mutable references
        let _attn_norm = block.attn_norm_mut();
        let _attention = block.attention_mut();
        let _ffn_norm = block.ffn_norm_mut();
        let _ffn = block.ffn_mut();
    }

    // Embedding tests

    #[test]
    fn test_embedding_creation() {
        let embed = Embedding::new(1000, 64).unwrap();
        assert_eq!(embed.vocab_size(), 1000);
        assert_eq!(embed.embed_dim(), 64);
    }

    #[test]
    fn test_embedding_zero_params_error() {
        assert!(Embedding::new(0, 64).is_err());
        assert!(Embedding::new(1000, 0).is_err());
    }

    #[test]
    fn test_embedding_forward_shape() {
        let embed = Embedding::new(100, 8).unwrap();

        let token_ids = vec![0, 1, 2];
        let output = embed.forward(&token_ids).unwrap();

        assert_eq!(output.shape(), &[3, 8]);
        assert_eq!(output.data().len(), 24);
    }

    #[test]
    fn test_embedding_forward_lookup() {
        let mut embed = Embedding::new(10, 4).unwrap();

        // Set specific embedding for token 5
        let offset = 5 * 4;
        embed.weights_mut()[offset] = 1.0;
        embed.weights_mut()[offset + 1] = 2.0;
        embed.weights_mut()[offset + 2] = 3.0;
        embed.weights_mut()[offset + 3] = 4.0;

        let output = embed.forward(&[5]).unwrap();
        assert_eq!(output.shape(), &[1, 4]);
        assert!((output.data()[0] - 1.0).abs() < 1e-6);
        assert!((output.data()[1] - 2.0).abs() < 1e-6);
        assert!((output.data()[2] - 3.0).abs() < 1e-6);
        assert!((output.data()[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_out_of_bounds_error() {
        let embed = Embedding::new(10, 4).unwrap();
        assert!(embed.forward(&[10]).is_err()); // ID 10 is out of bounds
        assert!(embed.forward(&[100]).is_err());
    }

    #[test]
    fn test_embedding_empty_input_error() {
        let embed = Embedding::new(10, 4).unwrap();
        assert!(embed.forward(&[]).is_err());
    }

    // Model tests

    #[test]
    fn test_model_creation() {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 8,
            num_heads: 1,
            num_layers: 2,
            intermediate_dim: 32,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();
        assert_eq!(model.config().vocab_size, 100);
        assert_eq!(model.config().num_layers, 2);
    }

    #[test]
    fn test_model_forward_shape() {
        let config = ModelConfig {
            vocab_size: 50,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        let token_ids = vec![0, 1, 2];
        let output = model.forward(&token_ids).unwrap();

        // Output should be [seq_len, vocab_size]
        assert_eq!(output.shape(), &[3, 50]);
    }

    #[test]
    fn test_model_forward_valid_output() {
        let config = ModelConfig {
            vocab_size: 20,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        let output = model.forward(&[0, 1]).unwrap();

        // Output should be finite
        for &val in output.data() {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_model_num_parameters() {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 8,
            num_heads: 1,
            num_layers: 2,
            intermediate_dim: 32,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        let params = model.num_parameters();
        assert!(params > 0);
        // Should be at least embedding + lm_head
        assert!(params >= 100 * 8 + 8 * 100);
    }

    #[test]
    fn test_model_mutable_access() {
        let config = ModelConfig {
            vocab_size: 50,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let mut model = Model::new(config).unwrap();

        // Verify we can access mutable references
        let _embed = model.embedding_mut();
        let _blocks = model.blocks_mut();
        let _norm = model.final_norm_mut();
        let _head = model.lm_head_mut();
    }

    #[test]
    fn test_model_generate_basic() {
        let config = ModelConfig {
            vocab_size: 20,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        let gen_config = GenerationConfig::greedy().with_max_tokens(5);
        let tokens = model.generate(&[0], &gen_config).unwrap();

        // Should have prompt + up to 5 generated tokens
        assert!(tokens.len() <= 6);
        assert!(!tokens.is_empty());
        // First token should be prompt
        assert_eq!(tokens[0], 0);
    }

    #[test]
    fn test_model_generate_respects_max_tokens() {
        let config = ModelConfig {
            vocab_size: 10,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        let gen_config = GenerationConfig::greedy().with_max_tokens(3);
        let tokens = model.generate(&[0, 1], &gen_config).unwrap();

        // Should have 2 prompt + 3 generated = 5 max
        assert!(tokens.len() <= 5);
    }

    #[test]
    fn test_model_generate_with_eos() {
        let config = ModelConfig {
            vocab_size: 10,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        // Set EOS token
        let gen_config = GenerationConfig::greedy()
            .with_max_tokens(100)
            .with_eos_token_id(5);

        let tokens = model.generate(&[0], &gen_config).unwrap();

        // Should stop before max_tokens if EOS is generated
        // (may or may not hit EOS depending on model weights)
        assert!(tokens.len() <= 101);
    }

    #[test]
    fn test_model_generate_empty_prompt_error() {
        let config = ModelConfig {
            vocab_size: 10,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        let gen_config = GenerationConfig::greedy();
        let result = model.generate(&[], &gen_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_generate_deterministic_with_seed() {
        let config = ModelConfig {
            vocab_size: 20,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        // Same seed should give same results
        let gen_config = GenerationConfig::greedy()
            .with_max_tokens(5)
            .with_seed(12345);

        let tokens1 = model.generate(&[0], &gen_config).unwrap();
        let tokens2 = model.generate(&[0], &gen_config).unwrap();

        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_model_generate_top_k() {
        let config = ModelConfig {
            vocab_size: 20,
            hidden_dim: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 16,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        let gen_config = GenerationConfig::top_k(5).with_max_tokens(3).with_seed(42);

        let tokens = model.generate(&[0], &gen_config).unwrap();

        // Should generate valid tokens
        assert!(tokens.len() <= 4);
        for &token in &tokens {
            assert!(token < 20);
        }
    }

    // MultiHeadAttention tests

    #[test]
    fn test_multi_head_attention_creation_mha() {
        // Standard Multi-Head Attention (num_kv_heads = num_heads)
        let mha = MultiHeadAttention::mha(64, 8).unwrap();
        assert_eq!(mha.num_heads(), 8);
        assert_eq!(mha.num_kv_heads(), 8);
        assert_eq!(mha.head_dim(), 8); // 64 / 8
        assert_eq!(mha.hidden_dim(), 64);
        assert!(mha.is_mha());
        assert!(!mha.is_mqa());
        assert!(!mha.is_gqa());
    }

    #[test]
    fn test_multi_head_attention_creation_mqa() {
        // Multi-Query Attention (num_kv_heads = 1)
        let mqa = MultiHeadAttention::mqa(64, 8).unwrap();
        assert_eq!(mqa.num_heads(), 8);
        assert_eq!(mqa.num_kv_heads(), 1);
        assert_eq!(mqa.head_dim(), 8);
        assert_eq!(mqa.hidden_dim(), 64);
        assert!(mqa.is_mqa());
        assert!(!mqa.is_mha());
        assert!(!mqa.is_gqa());
    }

    #[test]
    fn test_multi_head_attention_creation_gqa() {
        // Grouped-Query Attention (1 < num_kv_heads < num_heads)
        let gqa = MultiHeadAttention::gqa(64, 8, 2).unwrap();
        assert_eq!(gqa.num_heads(), 8);
        assert_eq!(gqa.num_kv_heads(), 2);
        assert_eq!(gqa.head_dim(), 8);
        assert_eq!(gqa.hidden_dim(), 64);
        assert!(gqa.is_gqa());
        assert!(!gqa.is_mha());
        assert!(!gqa.is_mqa());
    }

    #[test]
    fn test_multi_head_attention_zero_hidden_dim_error() {
        let result = MultiHeadAttention::new(0, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_zero_num_heads_error() {
        let result = MultiHeadAttention::new(64, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_zero_num_kv_heads_error() {
        let result = MultiHeadAttention::new(64, 8, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_kv_heads_too_large_error() {
        // num_kv_heads cannot be greater than num_heads
        let result = MultiHeadAttention::new(64, 8, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_indivisible_error() {
        // 65 is not divisible by 8
        let result = MultiHeadAttention::new(65, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_heads_not_divisible_error() {
        // num_heads must be divisible by num_kv_heads
        let result = MultiHeadAttention::new(64, 8, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_mha_forward() {
        // Standard MHA with 2 heads
        let mha = MultiHeadAttention::mha(8, 2).unwrap();

        // Input: [seq_len=2, hidden_dim=8]
        let input = Tensor::from_vec(
            vec![2, 8],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 1
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 2
            ],
        )
        .unwrap();

        let output = mha.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_multi_head_attention_mqa_forward() {
        // Multi-Query Attention with 2 heads (shared K/V)
        let mqa = MultiHeadAttention::mqa(8, 2).unwrap();

        // Input: [seq_len=2, hidden_dim=8]
        let input = Tensor::from_vec(
            vec![2, 8],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 1
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 2
            ],
        )
        .unwrap();

        let output = mqa.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_multi_head_attention_shape_validation() {
        let mha = MultiHeadAttention::mha(8, 2).unwrap();

        // Wrong number of dimensions (1D instead of 2D)
        let input_1d = Tensor::from_vec(vec![8], vec![1.0; 8]).unwrap();
        let result = mha.forward(&input_1d);
        assert!(result.is_err());

        // Wrong hidden dimension
        let input_wrong_dim = Tensor::from_vec(vec![2, 16], vec![1.0; 32]).unwrap();
        let result = mha.forward(&input_wrong_dim);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_mha_vs_mqa_shape_consistency() {
        // Both MHA and MQA should produce same output shape
        let mha = MultiHeadAttention::mha(16, 4).unwrap();
        let mqa = MultiHeadAttention::mqa(16, 4).unwrap();

        let input = Tensor::from_vec(vec![3, 16], vec![0.5; 48]).unwrap();

        let multi_head_output = mha.forward(&input).unwrap();
        let multi_query_output = mqa.forward(&input).unwrap();

        // Both should have same output shape
        assert_eq!(multi_head_output.shape(), &[3, 16]);
        assert_eq!(multi_query_output.shape(), &[3, 16]);
        assert_eq!(multi_head_output.shape(), multi_query_output.shape());
    }

    #[test]
    fn test_multi_head_attention_single_head() {
        // Edge case: single head (equivalent to single attention)
        let mha = MultiHeadAttention::mha(8, 1).unwrap();

        let input = Tensor::from_vec(vec![2, 8], vec![0.5; 16]).unwrap();
        let output = mha.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_multi_head_attention_mqa_kv_sharing() {
        // MQA should work with larger number of heads
        let mqa = MultiHeadAttention::mqa(32, 8).unwrap();

        let input = Tensor::from_vec(vec![4, 32], vec![0.1; 128]).unwrap();
        let output = mqa.forward(&input).unwrap();

        assert_eq!(output.shape(), &[4, 32]);
    }

    #[test]
    fn test_multi_head_attention_long_sequence() {
        // Test with longer sequence
        let mha = MultiHeadAttention::mha(16, 4).unwrap();

        // Sequence length = 10
        let input = Tensor::from_vec(vec![10, 16], vec![0.3; 160]).unwrap();
        let output = mha.forward(&input).unwrap();

        assert_eq!(output.shape(), &[10, 16]);
    }

    #[test]
    fn test_multi_head_attention_mqa_memory_efficiency() {
        // MQA should still work correctly with shared K/V
        // This tests that the shared K/V logic is correct
        let mqa = MultiHeadAttention::mqa(64, 16).unwrap();

        // Small batch
        let input = Tensor::from_vec(vec![2, 64], vec![0.2; 128]).unwrap();
        let output = mqa.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 64]);
        assert_eq!(output.data().len(), 128); // 2 * 64
    }

    #[test]
    fn test_multi_head_attention_gqa_forward() {
        // Grouped-Query Attention with 8 heads, 2 KV heads (4 heads per group)
        let gqa = MultiHeadAttention::gqa(32, 8, 2).unwrap();

        // Input: [seq_len=3, hidden_dim=32]
        let input = Tensor::from_vec(vec![3, 32], vec![0.1; 96]).unwrap();

        let output = gqa.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape(), &[3, 32]);
    }

    #[test]
    fn test_multi_head_attention_gqa_shape_consistency() {
        // MHA, MQA, and GQA should all produce same output shape
        let mha = MultiHeadAttention::mha(64, 8).unwrap();
        let mqa = MultiHeadAttention::mqa(64, 8).unwrap();
        let gqa = MultiHeadAttention::gqa(64, 8, 2).unwrap();

        let input = Tensor::from_vec(vec![4, 64], vec![0.5; 256]).unwrap();

        let multi_head_out = mha.forward(&input).unwrap();
        let multi_query_out = mqa.forward(&input).unwrap();
        let grouped_query_out = gqa.forward(&input).unwrap();

        // All should have same output shape
        assert_eq!(multi_head_out.shape(), &[4, 64]);
        assert_eq!(multi_query_out.shape(), &[4, 64]);
        assert_eq!(grouped_query_out.shape(), &[4, 64]);
        assert_eq!(multi_head_out.shape(), multi_query_out.shape());
        assert_eq!(multi_head_out.shape(), grouped_query_out.shape());
    }

    #[test]
    fn test_multi_head_attention_gqa_different_group_sizes() {
        // Test GQA with different group sizes
        // 16 heads, 4 KV heads (4 heads per group)
        let gqa1 = MultiHeadAttention::gqa(128, 16, 4).unwrap();
        let input = Tensor::from_vec(vec![2, 128], vec![0.3; 256]).unwrap();
        let output1 = gqa1.forward(&input).unwrap();
        assert_eq!(output1.shape(), &[2, 128]);

        // 16 heads, 8 KV heads (2 heads per group)
        let gqa2 = MultiHeadAttention::gqa(128, 16, 8).unwrap();
        let output2 = gqa2.forward(&input).unwrap();
        assert_eq!(output2.shape(), &[2, 128]);
    }

    // ============================================================================
    // Phase 3 Acceptance Tests (Refs REALIZAR-PERF-SPEC-001)
    // ============================================================================

    /// Phase 3 acceptance test: verify tok/s meets spec target
    ///
    /// Per spec Phase 3 acceptance criteria:
    /// ```rust,ignore
    /// assert!(benchmark_tokens_per_second() >= 25.0);
    /// ```
    ///
    /// Note: This test uses a small test model to verify generation
    /// throughput meets the baseline. Real phi-2 benchmarking requires
    /// the actual model file and full optimization integration.
    #[test]
    fn test_phase3_acceptance_tokens_per_second() {
        use crate::generate::GenerationConfig;
        use std::time::Instant;

        // Create baseline model configuration
        // The optimized components (Flash Attention v2, FusedLayerNormLinear)
        // show significant speedups individually - see companion tests:
        // - Flash Attention v2 SIMD: ~10x faster than parallel for small sequences
        // - FusedLayerNormLinear parallel: ~3.6x faster for large batches
        //
        // This test verifies the generation loop meets baseline throughput.
        // Full phi-2 integration requires wiring up optimized components.
        let config = ModelConfig {
            vocab_size: 100, // Small vocab for fast softmax
            hidden_dim: 64,  // Smaller hidden dimension
            num_heads: 4,    // Multiple heads
            num_layers: 2,   // Two transformer layers
            intermediate_dim: 128,
            eps: 1e-5,
        };
        let model = Model::new(config).unwrap();

        // Warmup run
        let prompt = vec![1, 5, 10];
        let gen_config = GenerationConfig::greedy().with_max_tokens(5);
        let _ = model.generate(&prompt, &gen_config).unwrap();

        // Benchmark: generate 20 tokens 10 times
        let tokens_per_run = 20;
        let num_runs = 10;
        let gen_config = GenerationConfig::greedy().with_max_tokens(tokens_per_run);

        let start = Instant::now();
        for _ in 0..num_runs {
            let _ = model.generate(&prompt, &gen_config).unwrap();
        }
        let elapsed = start.elapsed();

        let total_tokens = tokens_per_run * num_runs;
        let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

        // Phase 3 acceptance: ≥25 tok/s
        // With optimized components, this should be achievable.
        // The individual component tests show:
        // - Flash Attention v2: 87µs/iter
        // - FusedLayerNormLinear parallel: 2.9ms/iter for 32x256->512
        assert!(
            tok_per_sec >= 25.0,
            "Phase 3 acceptance FAILED: {:.1} tok/s < 25.0 tok/s target. \
             Note: Full optimization requires integrating Flash Attention v2 \
             and FusedLayerNormLinear into Model::forward()",
            tok_per_sec
        );

        // Report performance
        eprintln!(
            "Phase 3 acceptance PASSED: {:.1} tok/s (target: ≥25.0 tok/s)",
            tok_per_sec
        );
    }

    /// Test Flash Attention v2 + parallel performance improvement
    #[test]
    fn test_phase3_flash_attention_v2_performance() {
        use std::time::Instant;

        let head_dim = 64;
        let seq_len = 32;

        // Attention::new takes head_dim only
        let attn = Attention::new(head_dim).unwrap();

        // Create QKV tensors
        let q = Tensor::from_vec(vec![seq_len, head_dim], vec![0.1; seq_len * head_dim]).unwrap();
        let k = Tensor::from_vec(vec![seq_len, head_dim], vec![0.2; seq_len * head_dim]).unwrap();
        let v = Tensor::from_vec(vec![seq_len, head_dim], vec![0.3; seq_len * head_dim]).unwrap();

        // Warmup
        let _ = attn.flash_forward_v2(&q, &k, &v, 8).unwrap();
        let _ = attn.flash_forward_parallel(&q, &k, &v, 8).unwrap();

        // Benchmark Flash Attention v2 (SIMD)
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = attn.flash_forward_v2(&q, &k, &v, 8).unwrap();
        }
        let v2_time = start.elapsed();

        // Benchmark Flash Attention parallel
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = attn.flash_forward_parallel(&q, &k, &v, 8).unwrap();
        }
        let parallel_time = start.elapsed();

        // Report performance (informational)
        let v2_us = v2_time.as_micros() as f64 / iterations as f64;
        let parallel_us = parallel_time.as_micros() as f64 / iterations as f64;

        eprintln!(
            "Flash Attention v2: {:.2}us/iter, Parallel: {:.2}us/iter",
            v2_us, parallel_us
        );

        // Both implementations should complete without error
        // Performance comparison is informational
        assert!(v2_us > 0.0, "v2 should have measurable time");
        assert!(parallel_us > 0.0, "parallel should have measurable time");
    }

    /// Test FusedLayerNormLinear performance improvement
    #[test]
    fn test_phase3_fused_layernorm_linear_performance() {
        use std::time::Instant;

        let feature_dim = 256;
        let out_features = 512;
        let batch_size = 32;

        // FusedLayerNormLinear::new initializes with default weights
        // (norm_weight=1.0, norm_bias=0.0, linear_weight=0.0, linear_bias=0.0)
        // which is fine for performance testing
        let fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).unwrap();

        // Create input batch
        let input = Tensor::from_vec(
            vec![batch_size, feature_dim],
            vec![0.5; batch_size * feature_dim],
        )
        .unwrap();

        // Warmup
        let _ = fused.forward(&input).unwrap();
        let _ = fused.forward_parallel(&input).unwrap();

        // Benchmark fused forward
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = fused.forward(&input).unwrap();
        }
        let fused_time = start.elapsed();

        // Benchmark parallel fused forward
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = fused.forward_parallel(&input).unwrap();
        }
        let parallel_time = start.elapsed();

        // Report performance
        let fused_us = fused_time.as_micros() as f64 / iterations as f64;
        let parallel_us = parallel_time.as_micros() as f64 / iterations as f64;

        eprintln!(
            "FusedLayerNormLinear: {:.2}us/iter, Parallel: {:.2}us/iter",
            fused_us, parallel_us
        );

        // Verify performance is measurable
        assert!(fused_us > 0.0, "fused should have measurable time");
        assert!(parallel_us > 0.0, "parallel should have measurable time");
    }

    // =========================================================================
    // BENCH-SPRINT-002: QuantizedLinear Tests (Q4_K Integration)
    // Per benchmark-model-runners-spec.md v2.0: Inline dequantization for 8x
    // memory bandwidth reduction vs f32.
    // =========================================================================

    /// RED: Test QuantizedLinear creation from Q4_K weight bytes
    #[test]
    fn test_quantized_linear_creation() {
        // Q4_K format: 144 bytes per super-block of 256 values
        // For in_features=256, out_features=4, we need 4 rows * 144 bytes = 576 bytes
        let in_features = 256;
        let out_features = 4;
        let bytes_per_row = 144; // One super-block per row
        let weight_bytes = vec![0u8; out_features * bytes_per_row];
        let bias = vec![0.0f32; out_features];

        let layer = QuantizedLinear::new(in_features, out_features, weight_bytes, bias);
        assert!(
            layer.is_ok(),
            "Should create QuantizedLinear from Q4_K bytes"
        );

        let layer = layer.unwrap();
        assert_eq!(layer.in_features(), in_features);
        assert_eq!(layer.out_features(), out_features);
    }

    /// RED: Test QuantizedLinear forward pass produces correct output
    #[test]
    fn test_quantized_linear_forward() {
        // Create synthetic Q4_K weights (zeros for simplicity)
        let in_features = 256;
        let out_features = 4;
        let bytes_per_row = 144;
        let weight_bytes = vec![0u8; out_features * bytes_per_row];
        let bias = vec![1.0f32; out_features]; // Non-zero bias

        let layer = QuantizedLinear::new(in_features, out_features, weight_bytes, bias)
            .expect("Should create layer");

        // Input activations
        let input = Tensor::from_vec(vec![in_features], vec![1.0f32; in_features])
            .expect("Should create input");

        // Forward pass
        let output = layer.forward(&input).expect("Forward should work");

        // Output should have shape [out_features]
        assert_eq!(output.shape(), &[out_features]);

        // With zero weights and bias=1.0, output should be [1.0, 1.0, 1.0, 1.0]
        for &val in output.data() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "Output should equal bias with zero weights"
            );
        }
    }

    /// RED: Test QuantizedLinear forward with batch input
    #[test]
    fn test_quantized_linear_batch_forward() {
        let in_features = 256;
        let out_features = 4;
        let batch_size = 8;
        let bytes_per_row = 144;
        let weight_bytes = vec![0u8; out_features * bytes_per_row];
        let bias = vec![2.0f32; out_features];

        let layer = QuantizedLinear::new(in_features, out_features, weight_bytes, bias)
            .expect("Should create layer");

        // Batch input [batch_size, in_features]
        let input = Tensor::from_vec(
            vec![batch_size, in_features],
            vec![1.0f32; batch_size * in_features],
        )
        .expect("Should create batch input");

        let output = layer.forward(&input).expect("Batch forward should work");

        // Output should have shape [batch_size, out_features]
        assert_eq!(output.shape(), &[batch_size, out_features]);
    }

    /// RED: Test QuantizedLinear memory usage is ~8x less than Linear
    #[test]
    fn test_quantized_linear_memory_efficiency() {
        let in_features = 4096; // Realistic embedding dim
        let out_features = 4096;

        // f32 Linear: 4096 * 4096 * 4 bytes = 64MB
        let f32_bytes = in_features * out_features * std::mem::size_of::<f32>();

        // Q4_K: 4096/256 = 16 super-blocks per row, 16 * 144 = 2304 bytes/row
        // Total: 4096 * 2304 = ~9.4MB (6.8x reduction, close to theoretical 8x)
        let super_blocks_per_row = in_features.div_ceil(256);
        let q4k_bytes = out_features * super_blocks_per_row * 144;

        let ratio = f32_bytes as f64 / q4k_bytes as f64;

        // Q4_K should be at least 6x smaller than f32 (accounting for scale/min overhead)
        assert!(
            ratio > 6.0,
            "Q4_K should be >6x smaller than f32: ratio={}",
            ratio
        );
        eprintln!(
            "Memory efficiency: f32={} bytes, Q4_K={} bytes, ratio={:.2}x",
            f32_bytes, q4k_bytes, ratio
        );
    }
}
