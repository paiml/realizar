//! Attention mechanisms for transformer models
//!
//! Extracted from layers/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - Attention: Basic scaled dot-product attention
//! - SlidingWindowAttention: Efficient attention with fixed window size
//! - FusedQKVAttention: FlashAttention-style tiled attention
//! - MultiHeadAttention: Full multi-head attention with Q/K/V projections

use crate::{
    error::{RealizarError, Result},
    tensor::Tensor,
};

use super::{softmax, Linear};

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
