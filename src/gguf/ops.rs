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

include!("ops_part_02.rs");
