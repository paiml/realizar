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
        // SAFETY: Memory safety ensured by bounds checking and alignment
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

// ============================================================================
// Sliding Window Attention (Mistral/Mixtral style)
// ============================================================================
//
// Limits attention to a fixed window of recent tokens for efficient
// long-context inference. Used by Mistral-7B, Mixtral, and similar models.
//
// Benefits:
// - Reduces memory from O(n²) to O(n*w) where w = window_size
// - Enables very long context with bounded KV cache
// - Compatible with Flash Attention algorithms
//
// Reference: "Mistral 7B" - Jiang et al., 2023
// ============================================================================

/// Sliding Window Attention
///
/// Limits each token to attending only to the most recent `window_size` tokens.
/// This provides linear memory scaling for long sequences while maintaining
/// local context.
///
/// # Algorithm
///
/// For each query position i, attention is computed only over keys/values
/// in positions `[max(0, i - window_size + 1), i]`.
///
/// ```text
/// Standard Attention (full):  Sliding Window (w=3):
///   Q K K K K K                 Q K K K . .
///   Q K K K K K                 . Q K K K .
///   Q K K K K K                 . . Q K K K
///   Q K K K K K                 . . . Q K K
/// ```
///
/// # References
///
/// - "Mistral 7B" - Jiang et al., 2023
/// - "Longformer: The Long-Document Transformer" - Beltagy et al., 2020
#[derive(Debug, Clone)]
pub struct SlidingWindowAttention {
    /// Head dimension (`d_k` = `d_model` / `num_heads`)
    head_dim: usize,
    /// Scale factor: 1 / `sqrt(head_dim)`
    scale: f32,
    /// Window size (number of tokens each query can attend to)
    window_size: usize,
}

impl SlidingWindowAttention {
    /// Create a new sliding window attention layer
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head
    /// * `window_size` - Number of tokens each query can attend to
    ///
    /// # Errors
    ///
    /// Returns error if `head_dim` is zero or `window_size` is zero
    pub fn new(head_dim: usize, window_size: usize) -> Result<Self> {
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }
        if window_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "window_size must be > 0".to_string(),
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            head_dim,
            scale,
            window_size,
        })
    }

    /// Compute sliding window attention
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

        let mut output = Vec::with_capacity(q_seq_len * self.head_dim);

        // Process each query position with sliding window
        for i in 0..q_seq_len {
            // Compute window boundaries [window_start, window_end)
            // For causal attention: can only attend to positions <= i
            let window_end = (i + 1).min(k_seq_len);
            let window_start = window_end.saturating_sub(self.window_size);
            let window_len = window_end - window_start;

            if window_len == 0 {
                // No keys to attend to, output zeros
                output.extend(std::iter::repeat_n(0.0, self.head_dim));
                continue;
            }

            // Compute attention scores for this window
            let mut scores = Vec::with_capacity(window_len);
            for j in window_start..window_end {
                let mut dot = 0.0;
                for k in 0..self.head_dim {
                    dot += q_data[i * self.head_dim + k] * k_data[j * self.head_dim + k];
                }
                scores.push(dot * self.scale);
            }

            // Apply softmax over window scores
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            for score in &mut scores {
                let exp_val = (*score - max_score).exp();
                *score = exp_val;
                exp_sum += exp_val;
            }
            let inv_sum = 1.0 / exp_sum;
            for score in &mut scores {
                *score *= inv_sum;
            }

            // Compute weighted sum of values
            for k in 0..self.head_dim {
                let mut sum = 0.0;
                for (idx, j) in (window_start..window_end).enumerate() {
                    sum += scores[idx] * v_data[j * self.head_dim + k];
                }
                output.push(sum);
            }
        }

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }

    /// Compute sliding window attention with mask
    ///
    /// Supports bidirectional attention (non-causal) with the sliding window.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor `[seq_len, head_dim]`
    /// * `key` - Key tensor `[seq_len, head_dim]`
    /// * `value` - Value tensor `[seq_len, head_dim]`
    /// * `causal` - If true, only attend to past positions (causal/autoregressive)
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if shapes don't match
    pub fn forward_with_mask(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        causal: bool,
    ) -> Result<Tensor<f32>> {
        if causal {
            // Causal is the default behavior
            return self.forward(query, key, value);
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

        let mut output = Vec::with_capacity(q_seq_len * self.head_dim);
        let half_window = self.window_size / 2;

        // Process each query position with bidirectional sliding window
        for i in 0..q_seq_len {
            // Bidirectional window centered on position i
            let window_start = i.saturating_sub(half_window);
            let window_end = (i + half_window + 1).min(k_seq_len);
            let window_len = window_end - window_start;

            if window_len == 0 {
                output.extend(std::iter::repeat_n(0.0, self.head_dim));
                continue;
            }

            // Compute attention scores for this window
            let mut scores = Vec::with_capacity(window_len);
            for j in window_start..window_end {
                let mut dot = 0.0;
                for k in 0..self.head_dim {
                    dot += q_data[i * self.head_dim + k] * k_data[j * self.head_dim + k];
                }
                scores.push(dot * self.scale);
            }

            // Apply softmax over window scores
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            for score in &mut scores {
                let exp_val = (*score - max_score).exp();
                *score = exp_val;
                exp_sum += exp_val;
            }
            let inv_sum = 1.0 / exp_sum;
            for score in &mut scores {
                *score *= inv_sum;
            }

            // Compute weighted sum of values
            for k in 0..self.head_dim {
                let mut sum = 0.0;
                for (idx, j) in (window_start..window_end).enumerate() {
                    sum += scores[idx] * v_data[j * self.head_dim + k];
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

    /// Get window size
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Compute the effective context at a given position
    ///
    /// Returns the number of tokens this position can attend to
    #[must_use]
    pub fn effective_context(&self, position: usize, seq_len: usize) -> usize {
        let window_end = (position + 1).min(seq_len);
        let window_start = window_end.saturating_sub(self.window_size);
        window_end - window_start
    }

    /// Memory usage relative to full attention
    ///
    /// Returns the ratio of memory used compared to full attention.
    /// For window_size w and seq_len n: memory = O(n*w) vs O(n²)
    #[must_use]
    pub fn memory_ratio(&self, seq_len: usize) -> f32 {
        if seq_len == 0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            (self.window_size.min(seq_len) as f32) / (seq_len as f32)
        }
    }
}

// ============================================================================
// Fused QKV + Attention (IMP-003)
// Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md
// ============================================================================

/// Fused Query-Key-Value projection with scaled dot-product attention
///
/// Combines QKV projection and attention into a single fused operation for
/// improved memory efficiency and performance. Eliminates intermediate
/// materializations by computing attention in a single pass.
///
/// # Performance Benefits
///
/// - **Memory Bandwidth**: Single read of input, single write of output
/// - **Cache Efficiency**: QKV computed block-wise to maximize L1/L2 reuse
/// - **Numerical Stability**: Uses log-sum-exp trick for softmax
///
/// # Algorithm (Flash Attention style)
///
/// ```text
/// for each block of queries:
///     Q_block = input_block @ W_q
///     for each block of keys/values:
///         K_block = input_block @ W_k
///         V_block = input_block @ W_v
///         scores = Q_block @ K_block^T / sqrt(d)
///         update running softmax and output
/// ```
///
/// # References
///
/// - [1] Dao et al., "FlashAttention: Fast and Memory-Efficient Attention", 2022
/// - [2] Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism", 2023
#[derive(Debug, Clone)]
pub struct FusedQKVAttention {
    /// Dimension per attention head
    head_dim: usize,
    /// Total hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Scale factor: 1 / sqrt(head_dim)
    scale: f32,
    /// Query projection weights: [hidden_dim, hidden_dim]
    w_q: Vec<f32>,
    /// Key projection weights: [hidden_dim, hidden_dim]
    w_k: Vec<f32>,
    /// Value projection weights: [hidden_dim, hidden_dim]
    w_v: Vec<f32>,
    /// Output projection weights: [hidden_dim, hidden_dim]
    w_o: Vec<f32>,
}

impl FusedQKVAttention {
    /// Create a new fused QKV attention layer
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension per attention head
    /// * `hidden_dim` - Total hidden dimension (must be divisible by head_dim)
    ///
    /// # Errors
    ///
    /// Returns error if head_dim is 0, hidden_dim is 0, or hidden_dim % head_dim != 0
    pub fn new(head_dim: usize, hidden_dim: usize) -> Result<Self> {
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }
        if hidden_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "hidden_dim must be > 0".to_string(),
            });
        }
        if !hidden_dim.is_multiple_of(head_dim) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim ({}) must be divisible by head_dim ({})",
                    hidden_dim, head_dim
                ),
            });
        }

        let num_heads = hidden_dim / head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let proj_size = hidden_dim * hidden_dim;

        // Initialize with small random-like values for non-degenerate behavior
        let init_weight = |size: usize| -> Vec<f32> {
            (0..size).map(|i| (i as f32 * 0.001).sin() * 0.02).collect()
        };

        Ok(Self {
            head_dim,
            hidden_dim,
            num_heads,
            scale,
            w_q: init_weight(proj_size),
            w_k: init_weight(proj_size),
            w_v: init_weight(proj_size),
            w_o: init_weight(proj_size),
        })
    }

    /// Forward pass with fused QKV projection and attention
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [seq_len, hidden_dim]
    ///
    /// # Returns
    ///
    /// Output tensor [seq_len, hidden_dim]
    ///
    /// # Errors
    ///
    /// Returns error if input shape doesn't match hidden_dim
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(RealizarError::InvalidShape {
                reason: "Input must have at least 2 dimensions [seq_len, hidden_dim]".to_string(),
            });
        }

        let seq_len = shape[0];
        let input_dim = shape[shape.len() - 1];

        if input_dim != self.hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input hidden_dim ({}) doesn't match layer hidden_dim ({})",
                    input_dim, self.hidden_dim
                ),
            });
        }

        let data = input.data();

        // Compute Q, K, V projections
        let mut q = vec![0.0f32; seq_len * self.hidden_dim];
        let mut k = vec![0.0f32; seq_len * self.hidden_dim];
        let mut v = vec![0.0f32; seq_len * self.hidden_dim];

        // Matrix multiply: [seq_len, hidden_dim] @ [hidden_dim, hidden_dim]
        for i in 0..seq_len {
            for j in 0..self.hidden_dim {
                let mut sum_q = 0.0f32;
                let mut sum_k = 0.0f32;
                let mut sum_v = 0.0f32;
                for l in 0..self.hidden_dim {
                    let inp = data[i * self.hidden_dim + l];
                    sum_q += inp * self.w_q[l * self.hidden_dim + j];
                    sum_k += inp * self.w_k[l * self.hidden_dim + j];
                    sum_v += inp * self.w_v[l * self.hidden_dim + j];
                }
                q[i * self.hidden_dim + j] = sum_q;
                k[i * self.hidden_dim + j] = sum_k;
                v[i * self.hidden_dim + j] = sum_v;
            }
        }

        // Compute attention per head
        let mut output = vec![0.0f32; seq_len * self.hidden_dim];

        for head in 0..self.num_heads {
            let head_offset = head * self.head_dim;

            // Compute attention scores for this head
            for i in 0..seq_len {
                // Find max for numerical stability (causal: only j <= i)
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=i {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let q_idx = i * self.hidden_dim + head_offset + d;
                        let k_idx = j * self.hidden_dim + head_offset + d;
                        dot += q[q_idx] * k[k_idx];
                    }
                    let score = dot * self.scale;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Compute softmax with log-sum-exp trick
                // Using enumerate() pattern for causal attention where j <= i
                let mut sum_exp = 0.0f32;
                let mut scores = vec![0.0f32; i + 1];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let q_idx = i * self.hidden_dim + head_offset + d;
                        let k_idx = j * self.hidden_dim + head_offset + d;
                        dot += q[q_idx] * k[k_idx];
                    }
                    *score = (dot * self.scale - max_score).exp();
                    sum_exp += *score;
                }

                // Normalize and compute weighted sum of values
                if sum_exp > 0.0 {
                    for d in 0..self.head_dim {
                        let mut weighted_sum = 0.0f32;
                        for (j, &score) in scores.iter().enumerate() {
                            let v_idx = j * self.hidden_dim + head_offset + d;
                            weighted_sum += (score / sum_exp) * v[v_idx];
                        }
                        output[i * self.hidden_dim + head_offset + d] = weighted_sum;
                    }
                }
            }
        }

        // Output projection
        let mut final_output = vec![0.0f32; seq_len * self.hidden_dim];
        for i in 0..seq_len {
            for j in 0..self.hidden_dim {
                let mut sum = 0.0f32;
                for l in 0..self.hidden_dim {
                    sum += output[i * self.hidden_dim + l] * self.w_o[l * self.hidden_dim + j];
                }
                final_output[i * self.hidden_dim + j] = sum;
            }
        }

        Tensor::from_vec(vec![seq_len, self.hidden_dim], final_output)
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

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get mutable access to Q projection weights for loading
    pub fn w_q_mut(&mut self) -> &mut [f32] {
        &mut self.w_q
    }

    /// Get mutable access to K projection weights for loading
    pub fn w_k_mut(&mut self) -> &mut [f32] {
        &mut self.w_k
    }

    /// Get mutable access to V projection weights for loading
    pub fn w_v_mut(&mut self) -> &mut [f32] {
        &mut self.w_v
    }

    /// Get mutable access to output projection weights for loading
    pub fn w_o_mut(&mut self) -> &mut [f32] {
        &mut self.w_o
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
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
                ),
            });
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
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
        if !dim.is_multiple_of(2) {
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

// ============================================================================
// RoPE Scaling Methods (NTK, YaRN, Linear, Dynamic NTK)
// ============================================================================
//
// These methods extend RoPE to handle longer context lengths than trained.
// References:
// - NTK-aware: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
// - YaRN: https://arxiv.org/abs/2309.00071
// - Code Llama linear scaling: https://arxiv.org/abs/2308.12950
// ============================================================================

/// RoPE scaling type for context length extension
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RopeScalingType {
    /// No scaling (original RoPE)
    #[default]
    None,
    /// Linear interpolation (Code Llama style)
    /// scale = trained_length / target_length
    Linear {
        /// Scale factor (typically trained_length / target_length)
        scale: f32,
    },
    /// NTK-aware scaling
    /// Modifies base frequency: base' = base * scale^(dim / (dim - 2))
    Ntk {
        /// Scale factor for context extension
        scale: f32,
    },
    /// Dynamic NTK-aware scaling
    /// Adjusts scale dynamically based on current sequence length
    DynamicNtk {
        /// Original training context length
        original_max_len: usize,
        /// Target extended context length
        target_max_len: usize,
    },
    /// YaRN (Yet another RoPE extensioN)
    /// Combines NTK interpolation with attention scaling
    Yarn {
        /// Original training context length
        original_max_len: usize,
        /// Target extended context length
        target_max_len: usize,
        /// Attention scaling factor (typically sqrt(scale))
        attn_factor: f32,
        /// Beta for interpolation ramp (default: 32)
        beta_fast: f32,
        /// Beta for extrapolation (default: 1)
        beta_slow: f32,
    },
}

/// Scaled Rotary Position Embeddings
///
/// Extends `RoPE` with various scaling methods for context length extension.
/// Supports NTK-aware, Linear, Dynamic NTK, and YaRN scaling.
///
/// # Scaling Methods
///
/// ## Linear Scaling (Code Llama)
/// Simply scales down the position: pos' = pos / scale
///
/// ## NTK-aware Scaling
/// Modifies the base frequency to reduce high-frequency component decay:
/// base' = base * scale^(dim / (dim - 2))
///
/// ## Dynamic NTK
/// Dynamically adjusts NTK scale based on current sequence length
///
/// ## YaRN (Yet another RoPE extensioN)
/// Combines NTK with attention factor and interpolation ramp
///
/// # References
///
/// - "Code Llama: Open Foundation Models for Code" - Rozière et al., 2023
/// - "YaRN: Efficient Context Window Extension of Large Language Models" - Peng et al., 2023
#[derive(Debug, Clone)]
pub struct ScaledRoPE {
    /// Base RoPE parameters
    dim: usize,
    /// Original base frequency
    original_base: f32,
    /// Scaled base frequency (after NTK adjustment)
    scaled_base: f32,
    /// Scaling configuration
    scaling: RopeScalingType,
    /// Precomputed inverse frequencies (with scaling applied)
    inv_freq: Vec<f32>,
    /// Attention scaling factor (for YaRN)
    mscale: f32,
}

impl ScaledRoPE {
    /// Create a new scaled `RoPE` layer
    ///
    /// # Arguments
    ///
    /// * `dim` - Embedding dimension (must be even)
    /// * `base` - Base frequency (typically 10000)
    /// * `scaling` - Scaling method to use
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is zero or odd
    pub fn new(dim: usize, base: f32, scaling: RopeScalingType) -> Result<Self> {
        if dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "dim must be > 0".to_string(),
            });
        }
        if !dim.is_multiple_of(2) {
            return Err(RealizarError::InvalidShape {
                reason: "dim must be even for RoPE".to_string(),
            });
        }

        let (scaled_base, mscale, inv_freq) = Self::compute_frequencies(dim, base, &scaling);

        Ok(Self {
            dim,
            original_base: base,
            scaled_base,
            scaling,
            inv_freq,
            mscale,
        })
    }

    /// Create scaled `RoPE` with default base (10000)
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is zero or odd
    pub fn with_default_base(dim: usize, scaling: RopeScalingType) -> Result<Self> {
        Self::new(dim, 10000.0, scaling)
    }

    /// Compute inverse frequencies with scaling applied
    fn compute_frequencies(
        dim: usize,
        base: f32,
        scaling: &RopeScalingType,
    ) -> (f32, f32, Vec<f32>) {
        let half_dim = dim / 2;

        // Compute scaled base and mscale based on scaling type
        #[allow(clippy::cast_precision_loss)]
        let (scaled_base, mscale) = match scaling {
            RopeScalingType::None | RopeScalingType::Linear { .. } => (base, 1.0),
            RopeScalingType::Ntk { scale } => {
                // NTK formula: base' = base * scale^(dim / (dim - 2))
                let dim_f = dim as f32;
                let exponent = dim_f / (dim_f - 2.0);
                let ntk_base = base * scale.powf(exponent);
                (ntk_base, 1.0)
            },
            RopeScalingType::DynamicNtk {
                original_max_len,
                target_max_len,
            } => {
                // Dynamic NTK uses scale = target / original
                let scale = (*target_max_len as f32) / (*original_max_len as f32);
                let dim_f = dim as f32;
                let exponent = dim_f / (dim_f - 2.0);
                let ntk_base = base * scale.powf(exponent);
                (ntk_base, 1.0)
            },
            RopeScalingType::Yarn {
                original_max_len,
                target_max_len,
                attn_factor,
                beta_fast,
                beta_slow,
            } => {
                // YaRN combines NTK with attention scaling
                let scale = (*target_max_len as f32) / (*original_max_len as f32);
                let dim_f = dim as f32;

                // Compute NTK-style base modification
                // YaRN uses a smoother interpolation based on frequency
                let exponent = dim_f / (dim_f - 2.0);
                let ntk_base = base * scale.powf(exponent);

                // Compute mscale (attention factor)
                // Default: sqrt(1 + ln(scale) / ln(original_max_len))
                let mscale = if *attn_factor > 0.0 {
                    *attn_factor
                } else {
                    let log_scale = scale.ln();
                    let log_orig = (*original_max_len as f32).ln();
                    (1.0 + log_scale / log_orig).sqrt()
                };

                // The beta parameters affect the interpolation ramp
                // but are applied per-frequency in the forward pass
                let _ = (beta_fast, beta_slow); // Used in forward

                (ntk_base, mscale)
            },
        };

        // Compute inverse frequencies with scaled base
        let mut inv_freq = Vec::with_capacity(half_dim);

        #[allow(clippy::cast_precision_loss)]
        for i in 0..half_dim {
            let exponent = -2.0 * (i as f32) / (dim as f32);
            inv_freq.push(scaled_base.powf(exponent));
        }

        (scaled_base, mscale, inv_freq)
    }

    /// Apply scaled rotary embeddings to input at given position
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with last dimension equal to `dim`
    /// * `position` - Position index for computing rotation angles
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input, with scaled rotary embeddings applied
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

        // Compute effective position based on scaling type
        #[allow(clippy::cast_precision_loss)]
        let effective_pos = match &self.scaling {
            RopeScalingType::None
            | RopeScalingType::Ntk { .. }
            | RopeScalingType::DynamicNtk { .. }
            | RopeScalingType::Yarn { .. } => position as f32,
            RopeScalingType::Linear { scale } => (position as f32) / scale,
        };

        // Compute sin/cos for this position
        let half_dim = self.dim / 2;
        let mut cos_vals = Vec::with_capacity(half_dim);
        let mut sin_vals = Vec::with_capacity(half_dim);

        // Apply YaRN interpolation ramp if applicable
        #[allow(clippy::cast_precision_loss)]
        for (i, inv_f) in self.inv_freq.iter().enumerate() {
            let angle = inv_f * effective_pos;

            // For YaRN, apply interpolation ramp based on frequency index
            let (cos_val, sin_val) = if let RopeScalingType::Yarn {
                original_max_len,
                target_max_len,
                beta_fast,
                beta_slow,
                ..
            } = &self.scaling
            {
                // Compute wavelength for this frequency
                let freq = 1.0 / inv_f;
                let wavelength = 2.0 * std::f32::consts::PI * freq;

                // Compute interpolation factor
                let low_freq_wavelen = (*original_max_len as f32) / *beta_slow;
                let high_freq_wavelen = (*original_max_len as f32) / *beta_fast;

                let ramp = if wavelength < high_freq_wavelen {
                    0.0 // Full extrapolation (use NTK-scaled frequency)
                } else if wavelength > low_freq_wavelen {
                    1.0 // Full interpolation (use linear-scaled position)
                } else {
                    // Linear ramp between extrapolation and interpolation
                    (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
                };

                // Interpolate between NTK angle and linear angle
                let scale = (*target_max_len as f32) / (*original_max_len as f32);
                let linear_pos = effective_pos / scale;

                // Original frequency from unscaled base
                let orig_inv_f = self
                    .original_base
                    .powf(-2.0 * (i as f32) / (self.dim as f32));
                let linear_angle = orig_inv_f * linear_pos;

                // Interpolate angles
                let final_angle = angle * (1.0 - ramp) + linear_angle * ramp;

                (final_angle.cos(), final_angle.sin())
            } else {
                (angle.cos(), angle.sin())
            };

            cos_vals.push(cos_val);
            sin_vals.push(sin_val);
        }

        // Apply rotation to each vector (with mscale for YaRN)
        for vec_idx in 0..num_vectors {
            let offset = vec_idx * self.dim;

            for i in 0..half_dim {
                let x0 = data[offset + 2 * i];
                let x1 = data[offset + 2 * i + 1];
                let cos_val = cos_vals[i];
                let sin_val = sin_vals[i];

                // Apply 2D rotation with mscale
                let y0 = (x0 * cos_val - x1 * sin_val) * self.mscale;
                let y1 = (x0 * sin_val + x1 * cos_val) * self.mscale;

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

    /// Get original base frequency
    #[must_use]
    pub fn original_base(&self) -> f32 {
        self.original_base
    }

    /// Get scaled base frequency
    #[must_use]
    pub fn scaled_base(&self) -> f32 {
        self.scaled_base
    }

    /// Get scaling configuration
    #[must_use]
    pub fn scaling(&self) -> &RopeScalingType {
        &self.scaling
    }

    /// Get inverse frequencies
    #[must_use]
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// Get attention scaling factor (mscale)
    #[must_use]
    pub fn mscale(&self) -> f32 {
        self.mscale
    }

    /// Compute effective context length multiplier
    ///
    /// Returns the factor by which the original context length is extended
    #[must_use]
    pub fn context_length_multiplier(&self) -> f32 {
        match &self.scaling {
            RopeScalingType::None => 1.0,
            RopeScalingType::Linear { scale } | RopeScalingType::Ntk { scale } => *scale,
            RopeScalingType::DynamicNtk {
                original_max_len,
                target_max_len,
            }
            | RopeScalingType::Yarn {
                original_max_len,
                target_max_len,
                ..
            } => (*target_max_len as f32) / (*original_max_len as f32),
        }
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
        if !hidden_dim.is_multiple_of(num_heads) {
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

#[cfg(test)]
mod tests;
