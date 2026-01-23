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

// PMAT-802: Extracted modules
mod position;
pub use position::{RoPE, RopeScalingType, ScaledRoPE, ALiBi};
mod model;
pub use model::{KVCache, TransformerBlock, Embedding, Model, ModelConfig};
mod attention;
pub use attention::{Attention, SlidingWindowAttention, FusedQKVAttention, MultiHeadAttention};

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

#[cfg(test)]

#[cfg(test)]
mod tests;
