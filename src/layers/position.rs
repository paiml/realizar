//! Position embeddings for transformer models
//!
//! Extracted from layers/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - RoPE: Rotary Position Embeddings (RoFormer, LLaMA, PaLM)
//! - RopeScalingType: Context length extension methods (NTK, YaRN, Linear)
//! - ScaledRoPE: RoPE with scaling for extended context
//! - ALiBi: Attention with Linear Biases

use crate::{
    error::{RealizarError, Result},
    tensor::Tensor,
};

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

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== RoPE Tests ====================

    #[test]
    fn test_rope_new_valid_dim() {
        let rope = RoPE::new(64, 10000.0).unwrap();
        assert_eq!(rope.dim(), 64);
        assert_eq!(rope.base(), 10000.0);
        assert_eq!(rope.inv_freq().len(), 32);
    }

    #[test]
    fn test_rope_new_small_dim() {
        let rope = RoPE::new(2, 10000.0).unwrap();
        assert_eq!(rope.dim(), 2);
        assert_eq!(rope.inv_freq().len(), 1);
    }

    #[test]
    fn test_rope_new_zero_dim_error() {
        let result = RoPE::new(0, 10000.0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("dim must be > 0"));
    }

    #[test]
    fn test_rope_new_odd_dim_error() {
        let result = RoPE::new(3, 10000.0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("dim must be even"));
    }

    #[test]
    fn test_rope_with_default_base() {
        let rope = RoPE::with_default_base(64).unwrap();
        assert_eq!(rope.base(), 10000.0);
        assert_eq!(rope.dim(), 64);
    }

    #[test]
    fn test_rope_with_default_base_error() {
        let result = RoPE::with_default_base(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_inv_freq_values() {
        let rope = RoPE::new(4, 10000.0).unwrap();
        let inv_freq = rope.inv_freq();
        // inv_freq[0] = 10000^(-2*0/4) = 10000^0 = 1.0
        // inv_freq[1] = 10000^(-2*1/4) = 10000^(-0.5) = 1/sqrt(10000) = 0.01
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);
        assert!((inv_freq[1] - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_rope_forward_identity_at_position_zero() {
        let rope = RoPE::new(4, 10000.0).unwrap();
        let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let output = rope.forward(&input, 0).unwrap();
        // At position 0, all angles are 0, so cos=1, sin=0
        // Output should equal input for x*1 - y*0 = x, x*0 + y*1 = y
        let data = output.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
        assert!((data[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_forward_rotation() {
        let rope = RoPE::new(2, 1.0).unwrap();
        // With base=1, inv_freq[0]=1, so angle at pos=1 is 1 radian
        let input = Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap();
        let output = rope.forward(&input, 1).unwrap();
        let data = output.data();
        // Expected: [cos(1), sin(1)]
        assert!((data[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((data[1] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_rope_forward_multiple_vectors() {
        let rope = RoPE::new(2, 10000.0).unwrap();
        let input = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let output = rope.forward(&input, 0).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
        // At position 0, output == input
        let data = output.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_forward_empty_input_error() {
        let rope = RoPE::new(4, 10000.0).unwrap();
        // Tensor::from_vec with empty shape returns error, so we test with 0-size valid tensor
        // Create a 1D tensor with wrong dimension (0 elements in last dim)
        // Actually, empty tensors are rejected by Tensor::from_vec itself
        // So we test shape validation by using a mismatched dimension
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
        let result = rope.forward(&input, 0);
        // Expecting error due to dimension mismatch (dim 2 != expected 4)
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_forward_dim_mismatch_error() {
        let rope = RoPE::new(4, 10000.0).unwrap();
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
        let result = rope.forward(&input, 0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Expected last dimension 4"));
    }

    // ==================== RopeScalingType Tests ====================

    #[test]
    fn test_rope_scaling_type_default() {
        let scaling: RopeScalingType = Default::default();
        assert_eq!(scaling, RopeScalingType::None);
    }

    #[test]
    fn test_rope_scaling_type_linear() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        if let RopeScalingType::Linear { scale } = scaling {
            assert!((scale - 2.0).abs() < 1e-6);
        } else {
            panic!("Expected Linear variant");
        }
    }

    #[test]
    fn test_rope_scaling_type_ntk() {
        let scaling = RopeScalingType::Ntk { scale: 4.0 };
        if let RopeScalingType::Ntk { scale } = scaling {
            assert!((scale - 4.0).abs() < 1e-6);
        } else {
            panic!("Expected Ntk variant");
        }
    }

    #[test]
    fn test_rope_scaling_type_dynamic_ntk() {
        let scaling = RopeScalingType::DynamicNtk {
            original_max_len: 2048,
            target_max_len: 8192,
        };
        if let RopeScalingType::DynamicNtk {
            original_max_len,
            target_max_len,
        } = scaling
        {
            assert_eq!(original_max_len, 2048);
            assert_eq!(target_max_len, 8192);
        } else {
            panic!("Expected DynamicNtk variant");
        }
    }

    #[test]
    fn test_rope_scaling_type_yarn() {
        let scaling = RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 8192,
            attn_factor: 1.5,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        if let RopeScalingType::Yarn {
            original_max_len,
            target_max_len,
            attn_factor,
            ..
        } = scaling
        {
            assert_eq!(original_max_len, 2048);
            assert_eq!(target_max_len, 8192);
            assert!((attn_factor - 1.5).abs() < 1e-6);
        } else {
            panic!("Expected Yarn variant");
        }
    }

    #[test]
    fn test_rope_scaling_type_clone() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        let cloned = scaling;
        assert_eq!(scaling, cloned);
    }

    #[test]
    fn test_rope_scaling_type_copy() {
        let scaling = RopeScalingType::Ntk { scale: 4.0 };
        let copied: RopeScalingType = scaling;
        assert_eq!(copied, RopeScalingType::Ntk { scale: 4.0 });
    }

    // ==================== ScaledRoPE Tests ====================

    #[test]
    fn test_scaled_rope_new_none_scaling() {
        let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).unwrap();
        assert_eq!(scaled.dim(), 64);
        assert_eq!(scaled.original_base(), 10000.0);
        assert_eq!(scaled.scaled_base(), 10000.0);
        assert!((scaled.mscale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_new_zero_dim_error() {
        let result = ScaledRoPE::new(0, 10000.0, RopeScalingType::None);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_rope_new_odd_dim_error() {
        let result = ScaledRoPE::new(7, 10000.0, RopeScalingType::None);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_rope_with_default_base() {
        let scaled = ScaledRoPE::with_default_base(64, RopeScalingType::None).unwrap();
        assert_eq!(scaled.original_base(), 10000.0);
    }

    #[test]
    fn test_scaled_rope_linear_scaling() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        // Linear scaling doesn't change base
        assert_eq!(scaled.scaled_base(), 10000.0);
        assert!((scaled.mscale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_ntk_scaling() {
        let scaling = RopeScalingType::Ntk { scale: 4.0 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        // NTK: base' = base * scale^(dim / (dim - 2))
        // 64 / 62 ≈ 1.032, so base' = 10000 * 4^1.032 ≈ 41600
        assert!(scaled.scaled_base() > 10000.0);
        assert!((scaled.mscale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_dynamic_ntk_scaling() {
        let scaling = RopeScalingType::DynamicNtk {
            original_max_len: 2048,
            target_max_len: 8192,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        // scale = 8192/2048 = 4.0, so base should increase like NTK
        assert!(scaled.scaled_base() > 10000.0);
    }

    #[test]
    fn test_scaled_rope_yarn_scaling() {
        let scaling = RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 8192,
            attn_factor: 1.5,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        // YaRN should have mscale set to attn_factor
        assert!((scaled.mscale() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_yarn_auto_attn_factor() {
        let scaling = RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 8192,
            attn_factor: 0.0, // auto-compute
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        // Auto-computed mscale should be > 1.0
        assert!(scaled.mscale() > 1.0);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_none() {
        let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).unwrap();
        assert!((scaled.context_length_multiplier() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_linear() {
        let scaling = RopeScalingType::Linear { scale: 2.5 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        assert!((scaled.context_length_multiplier() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_ntk() {
        let scaling = RopeScalingType::Ntk { scale: 3.0 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        assert!((scaled.context_length_multiplier() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_dynamic_ntk() {
        let scaling = RopeScalingType::DynamicNtk {
            original_max_len: 1024,
            target_max_len: 4096,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        assert!((scaled.context_length_multiplier() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_yarn() {
        let scaling = RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 16384,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        assert!((scaled.context_length_multiplier() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_forward_none_scaling() {
        let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).unwrap();
        let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let output = scaled.forward(&input, 0).unwrap();
        let data = output.data();
        // At position 0, output == input (cos=1, sin=0)
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_forward_linear_scaling() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        let scaled = ScaledRoPE::new(2, 1.0, scaling).unwrap();
        let input = Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap();
        // With linear scaling, effective_pos = pos / scale = 2 / 2 = 1
        let output = scaled.forward(&input, 2).unwrap();
        let data = output.data();
        // Expected: [cos(1), sin(1)]
        assert!((data[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((data[1] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_forward_empty_input_error() {
        let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).unwrap();
        // Empty tensors rejected by Tensor::from_vec, so test dimension mismatch instead
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
        let result = scaled.forward(&input, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_rope_forward_dim_mismatch_error() {
        let scaled = ScaledRoPE::new(8, 10000.0, RopeScalingType::None).unwrap();
        let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = scaled.forward(&input, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_rope_inv_freq() {
        let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).unwrap();
        assert_eq!(scaled.inv_freq().len(), 2);
    }

    #[test]
    fn test_scaled_rope_scaling_getter() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).unwrap();
        assert_eq!(*scaled.scaling(), RopeScalingType::Linear { scale: 2.0 });
    }

    // ==================== ALiBi Tests ====================

    #[test]
    fn test_alibi_new_power_of_two() {
        let alibi = ALiBi::new(8).unwrap();
        assert_eq!(alibi.num_heads(), 8);
        assert_eq!(alibi.slopes().len(), 8);
    }

    #[test]
    fn test_alibi_new_non_power_of_two() {
        let alibi = ALiBi::new(6).unwrap();
        assert_eq!(alibi.num_heads(), 6);
        assert_eq!(alibi.slopes().len(), 6);
    }

    #[test]
    fn test_alibi_new_single_head() {
        let alibi = ALiBi::new(1).unwrap();
        assert_eq!(alibi.num_heads(), 1);
        assert_eq!(alibi.slopes().len(), 1);
    }

    #[test]
    fn test_alibi_new_zero_heads_error() {
        let result = ALiBi::new(0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("num_heads must be > 0"));
    }

    #[test]
    fn test_alibi_slopes_power_of_two() {
        let alibi = ALiBi::new(4).unwrap();
        let slopes = alibi.slopes();
        // For n=4: m[h] = 2^(-8h/4) = 2^(-2h)
        // slopes = [2^0, 2^-2, 2^-4, 2^-6] = [1.0, 0.25, 0.0625, 0.015625]
        assert!((slopes[0] - 1.0).abs() < 1e-6);
        assert!((slopes[1] - 0.25).abs() < 1e-6);
        assert!((slopes[2] - 0.0625).abs() < 1e-6);
        assert!((slopes[3] - 0.015625).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_slopes_8_heads() {
        let alibi = ALiBi::new(8).unwrap();
        let slopes = alibi.slopes();
        // For n=8: m[h] = 2^(-8h/8) = 2^(-h)
        // slopes[0] = 2^0 = 1.0
        // slopes[1] = 2^-1 = 0.5
        // slopes[7] = 2^-7 = 0.0078125
        assert!((slopes[0] - 1.0).abs() < 1e-6);
        assert!((slopes[1] - 0.5).abs() < 1e-6);
        assert!((slopes[7] - 0.0078125).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_get_bias_shape() {
        let alibi = ALiBi::new(4).unwrap();
        let bias = alibi.get_bias(8).unwrap();
        assert_eq!(bias.shape(), &[8, 8, 4]);
    }

    #[test]
    fn test_alibi_get_bias_zero_seq_error() {
        let alibi = ALiBi::new(4).unwrap();
        let result = alibi.get_bias(0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("seq_len must be > 0"));
    }

    #[test]
    fn test_alibi_get_bias_diagonal_is_zero() {
        let alibi = ALiBi::new(2).unwrap();
        let bias = alibi.get_bias(3).unwrap();
        let data = bias.data();
        // At (i, i), distance = 0, so bias = 0
        // Bias is [seq_len, seq_len, num_heads], linearized as [i][j][h]
        // (0,0,0): data[0*3*2 + 0*2 + 0] = data[0]
        // (1,1,0): data[1*3*2 + 1*2 + 0] = data[8]
        // (2,2,0): data[2*3*2 + 2*2 + 0] = data[16]
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[8] - 0.0).abs() < 1e-6);
        assert!((data[16] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_get_bias_values() {
        let alibi = ALiBi::new(1).unwrap();
        let bias = alibi.get_bias(3).unwrap();
        let data = bias.data();
        let slope = alibi.slopes()[0];
        // With 1 head: bias[i,j,0] = -slope * |i-j|
        // (0,0): 0, (0,1): -slope*1, (0,2): -slope*2
        // (1,0): -slope*1, (1,1): 0, (1,2): -slope*1
        // (2,0): -slope*2, (2,1): -slope*1, (2,2): 0
        assert!((data[0] - 0.0).abs() < 1e-6); // (0,0)
        assert!((data[1] - (-slope * 1.0)).abs() < 1e-6); // (0,1)
        assert!((data[2] - (-slope * 2.0)).abs() < 1e-6); // (0,2)
        assert!((data[3] - (-slope * 1.0)).abs() < 1e-6); // (1,0)
        assert!((data[4] - 0.0).abs() < 1e-6); // (1,1)
        assert!((data[5] - (-slope * 1.0)).abs() < 1e-6); // (1,2)
    }

    #[test]
    fn test_alibi_bias_symmetry() {
        let alibi = ALiBi::new(2).unwrap();
        let bias = alibi.get_bias(4).unwrap();
        let data = bias.data();
        // bias[i,j,h] should have |bias[i,j,h]| == |bias[j,i,h]|
        // Since we use |i-j|, bias[i,j,h] == bias[j,i,h] (both negative)
        for i in 0..4 {
            for j in 0..4 {
                for h in 0..2 {
                    let idx_ij = i * 4 * 2 + j * 2 + h;
                    let idx_ji = j * 4 * 2 + i * 2 + h;
                    assert!((data[idx_ij] - data[idx_ji]).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_alibi_clone() {
        let alibi = ALiBi::new(4).unwrap();
        let cloned = alibi.clone();
        assert_eq!(alibi.num_heads(), cloned.num_heads());
        assert_eq!(alibi.slopes(), cloned.slopes());
    }

    #[test]
    fn test_alibi_large_seq_len() {
        let alibi = ALiBi::new(8).unwrap();
        let bias = alibi.get_bias(256).unwrap();
        assert_eq!(bias.shape(), &[256, 256, 8]);
        // Check a corner case: (0, 255) should be -slope * 255
        let data = bias.data();
        let idx = 255 * 8; // (0, 255, head=0)
        let expected = -alibi.slopes()[0] * 255.0;
        assert!((data[idx] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_alibi_non_power_of_two_slopes() {
        // For 6 heads: closest power of 2 is 4
        // First 4 slopes: 2^(-8h/4) = 2^(-2h) for h=0..4
        // Extra 2 slopes: 2^(-(2h+1)*4/4) = 2^(-2h-1) for h=0..2
        let alibi = ALiBi::new(6).unwrap();
        let slopes = alibi.slopes();
        assert_eq!(slopes.len(), 6);
        // First 4: [1.0, 0.25, 0.0625, 0.015625]
        assert!((slopes[0] - 1.0).abs() < 1e-6);
        assert!((slopes[3] - 0.015625).abs() < 1e-6);
        // Extra 2 have different computation
        assert!(slopes[4] > 0.0);
        assert!(slopes[5] > 0.0);
    }
}
