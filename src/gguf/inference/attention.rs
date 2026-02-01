//! Attention computation for OwnedQuantizedModel
//!
//! Contains apply_rope (rotary position embeddings), causal_attention,
//! and KV-cache based attention variants with GQA support.

use crate::error::{RealizarError, Result};
use crate::gguf::{GGUFConfig, OwnedQuantizedModel};

impl OwnedQuantizedModel {
    /// Apply Rotary Position Embedding (RoPE) to Q or K vectors
    ///
    /// Supports two RoPE styles:
    /// - NORM (type 0): Adjacent pairs rotation (LLaMA default)
    /// - NEOX (type 2): Split halves rotation (GPT-NeoX)
    ///
    /// # Arguments
    /// * `x` - Vector to rotate in-place [num_heads_in_x * head_dim]
    /// * `position` - Position index for frequency calculation
    /// * `num_heads_in_x` - Number of heads in x (num_heads for Q, num_kv_heads for K)
    ///
    /// # GQA Support
    /// For GQA models, pass num_heads for Q vectors and num_kv_heads for K vectors.
    pub(crate) fn apply_rope(&self, x: &mut [f32], position: usize, num_heads_in_x: usize) {
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let rope_type = self.config.rope_type;

        // Stack-based buffers (max 128 = 256 head_dim, covers all common models)
        // Avoids heap allocation on every call
        let mut cos_vals: [f32; 128] = [0.0; 128];
        let mut sin_vals: [f32; 128] = [0.0; 128];

        // Pre-compute cos/sin for this position (reused across all heads)
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;
        for i in 0..half_dim.min(128) {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_v, cos_v) = angle.sin_cos();
            cos_vals[i] = cos_v;
            sin_vals[i] = sin_v;
        }

        // Apply rotation to each head
        for h in 0..num_heads_in_x {
            let head_start = h * head_dim;

            if head_start + head_dim > x.len() {
                continue;
            }

            if rope_type == 2 {
                // NEOX style: split halves (x[0..half], x[half..])
                // Used by GPT-NeoX and some newer models
                let (first_half, second_half) =
                    x[head_start..head_start + head_dim].split_at_mut(half_dim);
                crate::quantize::apply_rope_rotation_simd(
                    first_half,
                    second_half,
                    &cos_vals[..half_dim],
                    &sin_vals[..half_dim],
                );
            } else {
                // NORM style (type 0): adjacent pairs (x[0], x[1]), (x[2], x[3]), ...
                // This is the default for LLaMA-family models
                let head_slice = &mut x[head_start..head_start + head_dim];
                for i in 0..half_dim {
                    let x0 = head_slice[2 * i];
                    let x1 = head_slice[2 * i + 1];
                    let cos_v = cos_vals[i];
                    let sin_v = sin_vals[i];
                    head_slice[2 * i] = x0 * cos_v - x1 * sin_v;
                    head_slice[2 * i + 1] = x0 * sin_v + x1 * cos_v;
                }
            }
        }
    }

    /// Compute scaled dot-product attention with causal mask (IMP-101b)
    ///
    /// Computes: softmax(QK^T / sqrt(d_k)) * V with causal masking
    ///
    /// # Arguments
    /// * `q` - Query vectors [seq_len, q_dim] where q_dim = num_heads * head_dim
    /// * `k` - Key vectors [seq_len, kv_dim] where kv_dim = num_kv_heads * head_dim
    /// * `v` - Value vectors [seq_len, kv_dim] where kv_dim = num_kv_heads * head_dim
    ///
    /// # Returns
    /// Attention output [seq_len, q_dim] where q_dim = num_heads * head_dim
    ///
    /// # GQA (Grouped Query Attention) Support
    /// For models where num_kv_heads < num_heads (e.g., TinyLlama: 4 vs 32),
    /// multiple Q heads share the same K/V head. The group size is num_heads / num_kv_heads.
    pub(crate) fn causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // GQA: multiple Q heads share each KV head
        // group_size = num_heads / num_kv_heads (e.g., 32/4 = 8 for TinyLlama)
        let group_size = num_heads / num_kv_heads;

        // Q has num_heads heads, K/V have num_kv_heads heads
        let q_dim = num_heads * head_dim; // e.g., 32 * 64 = 2048
        let kv_dim = num_kv_heads * head_dim; // e.g., 4 * 64 = 256

        let mut output = vec![0.0f32; seq_len * q_dim];

        // Process each Q head independently
        for head in 0..num_heads {
            // Map Q head to corresponding KV head (GQA grouping)
            let kv_head = head / group_size;

            let q_head_offset = head * head_dim;
            let kv_head_offset = kv_head * head_dim;

            // Process each query position
            for i in 0..seq_len {
                // Compute attention scores for this query against all keys up to position i (causal)
                let mut scores = Vec::with_capacity(i + 1);
                let q_start = i * q_dim + q_head_offset;

                for j in 0..=i {
                    // Only attend to positions 0..=i (causal mask)
                    let k_start = j * kv_dim + kv_head_offset;

                    // Dot product Q[i] · K[j]
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q[q_start + d] * k[k_start + d];
                    }
                    scores.push(score * scale);
                }

                // Softmax (SIMD-optimized)
                crate::quantize::softmax_simd(&mut scores);

                // Weighted sum of values
                let out_start = i * q_dim + q_head_offset;
                for (j, &weight) in scores.iter().enumerate() {
                    let v_start = j * kv_dim + kv_head_offset;
                    for d in 0..head_dim {
                        output[out_start + d] += weight * v[v_start + d];
                    }
                }
            }
        }

        output
    }

    /// Get model configuration
    pub fn config(&self) -> &GGUFConfig {
        &self.config
    }

    /// Check if CUDA is enabled
    #[cfg(feature = "cuda")]
    pub fn cuda_enabled(&self) -> bool {
        self.cuda_executor.is_some()
    }

    /// Check if CUDA acceleration is enabled (stub when cuda feature disabled)
    #[cfg(not(feature = "cuda"))]
    pub fn cuda_enabled(&self) -> bool {
        false
    }

    // ============================================================================
    // SIMD Helper Methods
    // ============================================================================

    /// SIMD-optimized dot product for f32 slices
    #[inline]
    fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: We've verified AVX2+FMA support
                unsafe { Self::simd_dot_f32_avx2(a, b) }
            } else {
                Self::simd_dot_f32_scalar(a, b)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::simd_dot_f32_scalar(a, b)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn simd_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            use std::arch::x86_64::{
                _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
                _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps,
                _mm_movehl_ps,
            };

            let len = a.len().min(b.len());
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;

            // Process 16 floats at a time (2x unrolled for better ILP)
            while i + 16 <= len {
                let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
                let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                acc = _mm256_fmadd_ps(va0, vb0, acc);
                acc = _mm256_fmadd_ps(va1, vb1, acc);
                i += 16;
            }
            // Handle remaining 8-float chunk
            if i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                acc = _mm256_fmadd_ps(va, vb, acc);
                i += 8;
            }

            // Horizontal sum
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result = _mm_add_ss(sums, shuf2);
            let mut sum = _mm_cvtss_f32(result);

            // Handle remaining elements
            while i < len {
                sum += a[i] * b[i];
                i += 1;
            }

            sum
        }
    }

    #[inline]
    fn simd_dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// SIMD-optimized scaled accumulation: out[i] += weight * val[i]
    #[inline]
    fn simd_axpy_f32(out: &mut [f32], weight: f32, val: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: We've verified AVX2 support
                unsafe { Self::simd_axpy_f32_avx2(out, weight, val) }
            } else {
                Self::simd_axpy_f32_scalar(out, weight, val);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::simd_axpy_f32_scalar(out, weight, val);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn simd_axpy_f32_avx2(out: &mut [f32], weight: f32, val: &[f32]) {
        use std::arch::x86_64::{
            _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
        };

        let len = out.len().min(val.len());
        let w = _mm256_set1_ps(weight);
        let mut i = 0;

        // Process 8 floats at a time
        while i + 8 <= len {
            // SAFETY: bounds checked above, pointers valid
            let v_out = unsafe { _mm256_loadu_ps(out.as_ptr().add(i)) };
            // SAFETY: Memory safety ensured by bounds checking and alignment
            let v_val = unsafe { _mm256_loadu_ps(val.as_ptr().add(i)) };
            let result = _mm256_fmadd_ps(w, v_val, v_out);
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), result) };
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            out[i] += weight * val[i];
            i += 1;
        }
    }

    #[inline]
    fn simd_axpy_f32_scalar(out: &mut [f32], weight: f32, val: &[f32]) {
        for (o, v) in out.iter_mut().zip(val.iter()) {
            *o += weight * *v;
        }
    }

    // ============================================================================
    // KV Cache Attention Methods
    // ============================================================================

    /// Attention with KV cache for autoregressive generation
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Key for current position [hidden_dim]
    /// * `current_v` - Value for current position [hidden_dim]
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    pub(crate) fn attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / hidden_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; hidden_dim];

        // Process each head
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Compute attention scores against all positions (cached + current)
            let mut scores = Vec::with_capacity(total_len);

            // Scores against cached positions (SIMD-optimized)
            for pos in 0..cache_len {
                let k_start = pos * hidden_dim + head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];
                let score = Self::simd_dot_f32(q_head, cached_key) * scale;
                scores.push(score);
            }

            // Score against current position (SIMD-optimized)
            let curr_key = &current_k[head_offset..head_offset + head_dim];
            let current_score = Self::simd_dot_f32(q_head, curr_key) * scale;
            scores.push(current_score);

            // Softmax (SIMD-optimized)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            let out_head = &mut output[head_offset..head_offset + head_dim];

            // Sum over cached values (SIMD-optimized)
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                let v_start = pos * hidden_dim + head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];
                Self::simd_axpy_f32(out_head, weight, cached_val);
            }

            // Add current value (SIMD-optimized)
            let curr_val = &current_v[head_offset..head_offset + head_dim];
            let current_weight = scores[cache_len];
            Self::simd_axpy_f32(out_head, current_weight, curr_val);
        }

        output
    }

    /// Compute attention with Grouped Query Attention (GQA) support (IMP-105)
    ///
    /// GQA uses fewer KV heads than Q heads, with multiple Q heads sharing each KV head.
    /// This reduces memory bandwidth and KV cache size for large models.
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim] (num_heads Q heads)
    /// * `k_cache` - Cached keys [cache_len, kv_dim] (num_kv_heads KV heads)
    /// * `v_cache` - Cached values [cache_len, kv_dim] (num_kv_heads KV heads)
    /// * `current_k` - Key for current position [kv_dim]
    /// * `current_v` - Value for current position [kv_dim]
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    ///
    /// # GQA Mapping
    /// Q head i uses KV head (i * num_kv_heads / num_heads)
    /// Example: 8 Q heads, 2 KV heads → Q heads 0-3 use KV head 0, Q heads 4-7 use KV head 1
    pub fn attention_with_cache_gqa(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Number of Q heads that share each KV head
        let q_per_kv = num_heads / num_kv_heads;

        // Total sequence length = cached + 1 (current)
        let cache_len = if kv_dim > 0 {
            k_cache.len() / kv_dim
        } else {
            0
        };
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; hidden_dim];

        // Score buffer for the current group.
        // Size: q_per_kv * total_len.
        // We reuse this buffer for each KV group to minimize allocation.
        let mut group_scores = vec![0.0f32; q_per_kv * total_len];

        // Process each KV head group (OPTIMIZATION: Scan KV cache once per group)
        for kv_head in 0..num_kv_heads {
            let kv_head_offset = kv_head * head_dim;

            // 1. Compute Scores (Scan K Cache Once)
            for pos in 0..cache_len {
                let k_start = pos * kv_dim + kv_head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];

                // For each Q head in this group
                for i in 0..q_per_kv {
                    let q_head_idx = kv_head * q_per_kv + i;
                    let q_head_offset = q_head_idx * head_dim;
                    let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

                    let score = Self::simd_dot_f32(q_head_data, cached_key) * scale;
                    group_scores[i * total_len + pos] = score;
                }
            }

            // Handle current position K
            let curr_key = &current_k[kv_head_offset..kv_head_offset + head_dim];
            for i in 0..q_per_kv {
                let q_head_idx = kv_head * q_per_kv + i;
                let q_head_offset = q_head_idx * head_dim;
                let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

                let score = Self::simd_dot_f32(q_head_data, curr_key) * scale;
                group_scores[i * total_len + cache_len] = score;
            }

            // 2. Softmax (Per Q Head)
            for i in 0..q_per_kv {
                let start = i * total_len;
                let end = start + total_len;
                crate::quantize::softmax_simd(&mut group_scores[start..end]);
            }

            // 3. Accumulate Values (Scan V Cache Once)
            for pos in 0..cache_len {
                let v_start = pos * kv_dim + kv_head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];

                for i in 0..q_per_kv {
                    let weight = group_scores[i * total_len + pos];
                    let q_head_idx = kv_head * q_per_kv + i;
                    let out_offset = q_head_idx * head_dim;
                    let out_head = &mut output[out_offset..out_offset + head_dim];

                    Self::simd_axpy_f32(out_head, weight, cached_val);
                }
            }

            // Handle current position V
            let curr_val = &current_v[kv_head_offset..kv_head_offset + head_dim];
            for i in 0..q_per_kv {
                let weight = group_scores[i * total_len + cache_len];
                let q_head_idx = kv_head * q_per_kv + i;
                let out_offset = q_head_idx * head_dim;
                let out_head = &mut output[out_offset..out_offset + head_dim];

                Self::simd_axpy_f32(out_head, weight, curr_val);
            }
        }

        output
    }

    /// Attention with cache - writes to pre-allocated buffer (IMP-131)
    pub fn attention_with_cache_gqa_into(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        output: &mut [f32],
    ) {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_per_kv = num_heads / num_kv_heads;

        let cache_len = if kv_dim > 0 {
            k_cache.len() / kv_dim
        } else {
            0
        };
        let total_len = cache_len + 1;

        // Zero output buffer
        output[..hidden_dim].iter_mut().for_each(|x| *x = 0.0);

        // Score buffer for the current group.
        // Size: q_per_kv * total_len.
        // We reuse this buffer for each KV group to minimize allocation.
        let mut group_scores = vec![0.0f32; q_per_kv * total_len];

        // Process each KV head group (OPTIMIZATION: Scan KV cache once per group)
        for kv_head in 0..num_kv_heads {
            let kv_head_offset = kv_head * head_dim;

            // 1. Compute Scores (Scan K Cache Once)
            for pos in 0..cache_len {
                let k_start = pos * kv_dim + kv_head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];

                // For each Q head in this group
                for i in 0..q_per_kv {
                    let q_head_idx = kv_head * q_per_kv + i;
                    let q_head_offset = q_head_idx * head_dim;
                    let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

                    let score = Self::simd_dot_f32(q_head_data, cached_key) * scale;
                    group_scores[i * total_len + pos] = score;
                }
            }

            // Handle current position K
            let curr_key = &current_k[kv_head_offset..kv_head_offset + head_dim];
            for i in 0..q_per_kv {
                let q_head_idx = kv_head * q_per_kv + i;
                let q_head_offset = q_head_idx * head_dim;
                let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

                let score = Self::simd_dot_f32(q_head_data, curr_key) * scale;
                group_scores[i * total_len + cache_len] = score;
            }

            // 2. Softmax (Per Q Head)
            for i in 0..q_per_kv {
                let start = i * total_len;
                let end = start + total_len;
                crate::quantize::softmax_simd(&mut group_scores[start..end]);
            }

            // 3. Accumulate Values (Scan V Cache Once)
            for pos in 0..cache_len {
                let v_start = pos * kv_dim + kv_head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];

                for i in 0..q_per_kv {
                    let weight = group_scores[i * total_len + pos];
                    let q_head_idx = kv_head * q_per_kv + i;
                    let out_offset = q_head_idx * head_dim;
                    let out_head = &mut output[out_offset..out_offset + head_dim];

                    Self::simd_axpy_f32(out_head, weight, cached_val);
                }
            }

            // Handle current position V
            let curr_val = &current_v[kv_head_offset..kv_head_offset + head_dim];
            for i in 0..q_per_kv {
                let weight = group_scores[i * total_len + cache_len];
                let q_head_idx = kv_head * q_per_kv + i;
                let out_offset = q_head_idx * head_dim;
                let out_head = &mut output[out_offset..out_offset + head_dim];

                Self::simd_axpy_f32(out_head, weight, curr_val);
            }
        }
    }

    /// Adaptive attention with KV cache - auto-selects CPU or GPU backend (IMP-122)
    ///
    /// For short cache lengths (< 64), uses efficient CPU implementation.
    /// For long cache lengths (>= 64), uses GPU-accelerated computation.
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Key for current position [hidden_dim]
    /// * `current_v` - Value for current position [hidden_dim]
    ///
    /// # Returns
    /// Result containing attention output [hidden_dim]
    ///
    /// # Errors
    /// Returns error if GPU operations fail (for GPU path)
    #[cfg(feature = "gpu")]
    pub fn adaptive_attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // Calculate cache length
        let cache_len = if hidden_dim > 0 {
            k_cache.len() / hidden_dim
        } else {
            0
        };

        // Threshold for GPU dispatch (matches IMP-119)
        const GPU_CACHE_LEN_THRESHOLD: usize = 64;

        if cache_len >= GPU_CACHE_LEN_THRESHOLD {
            // GPU path for long sequences
            self.gpu_attention_with_cache(q, k_cache, v_cache, current_k, current_v)
        } else {
            // CPU path for short sequences - use existing implementation
            Ok(self.attention_with_cache(q, k_cache, v_cache, current_k, current_v))
        }
    }

    /// CPU-only version of adaptive attention
    #[cfg(not(feature = "gpu"))]
    pub fn adaptive_attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Result<Vec<f32>> {
        Ok(self.attention_with_cache(q, k_cache, v_cache, current_k, current_v))
    }

    /// GPU-accelerated attention with KV cache (IMP-122)
    ///
    /// Uses GPU for Q@K^T computation when cache is large enough.
    #[cfg(feature = "gpu")]
    fn gpu_attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / hidden_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; hidden_dim];

        // Create scheduler for GPU operations
        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "gpu_attention_with_cache".to_string(),
                reason: format!("Failed to create scheduler: {e}"),
            }
        })?;

        // Process each head
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Build full K matrix for this head: [total_len, head_dim]
            let mut k_full = Vec::with_capacity(total_len * head_dim);
            for pos in 0..cache_len {
                let k_start = pos * hidden_dim + head_offset;
                k_full.extend_from_slice(&k_cache[k_start..k_start + head_dim]);
            }
            k_full.extend_from_slice(&current_k[head_offset..head_offset + head_dim]);

            // Transpose K to [head_dim, total_len] for matmul
            let mut k_t = vec![0.0f32; head_dim * total_len];
            for pos in 0..total_len {
                for d in 0..head_dim {
                    k_t[d * total_len + pos] = k_full[pos * head_dim + d];
                }
            }

            // GPU matmul: Q[1, head_dim] @ K_T[head_dim, total_len] -> [1, total_len]
            let scores_raw = scheduler
                .matmul(q_head, &k_t, 1, head_dim, total_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "gpu_attention_with_cache".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Scale scores
            let mut scores: Vec<f32> = scores_raw.iter().map(|&s| s * scale).collect();

            // Softmax (SIMD-optimized)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            let out_head = &mut output[head_offset..head_offset + head_dim];

            // Cached values
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                let v_start = pos * hidden_dim + head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];
                for d in 0..head_dim {
                    out_head[d] += weight * cached_val[d];
                }
            }

            // Current value
            let curr_val = &current_v[head_offset..head_offset + head_dim];
            let current_weight = scores[cache_len];
            for d in 0..head_dim {
                out_head[d] += current_weight * curr_val[d];
            }
        }

        Ok(output)
    }

    /// FlashAttention: Tiled attention with O(N) memory (PARITY-026)
    ///
    /// Implements the FlashAttention algorithm from Dao et al. for memory-efficient attention.
    /// Uses online softmax to process attention in tiles without materializing the N×N matrix.
    ///
    /// # Key Properties
    /// - Memory: O(N) instead of O(N²)
    /// - Numerically equivalent to standard attention
    /// - 10-100x memory savings for long sequences
    ///
    /// # Arguments
    /// * `q` - Query vector [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Current key [hidden_dim]
    /// * `current_v` - Current value [hidden_dim]
    /// * `block_size` - Tile size for tiled computation (default: 64)
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn flash_attention_tiled(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        block_size: usize,
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / hidden_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; hidden_dim];

        // Process each head with FlashAttention tiling
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Online softmax state for this head
            let mut m_i = f32::NEG_INFINITY; // Running max
            let mut l_i = 0.0f32; // Running sum of exp(score - max)
            let mut o_i = vec![0.0f32; head_dim]; // Accumulated output

            // Process KV cache in tiles
            let num_tiles = total_len.div_ceil(block_size);

            for tile_idx in 0..num_tiles {
                let tile_start = tile_idx * block_size;
                let tile_end = (tile_start + block_size).min(total_len);
                let tile_len = tile_end - tile_start;

                // Compute scores for this tile
                let mut tile_scores = Vec::with_capacity(tile_len);
                let mut tile_values: Vec<&[f32]> = Vec::with_capacity(tile_len);

                for pos in tile_start..tile_end {
                    if pos < cache_len {
                        // From cache
                        let k_start = pos * hidden_dim + head_offset;
                        let cached_key = &k_cache[k_start..k_start + head_dim];

                        // Compute Q·K score
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_head[d] * cached_key[d];
                        }
                        tile_scores.push(score * scale);

                        let v_start = pos * hidden_dim + head_offset;
                        tile_values.push(&v_cache[v_start..v_start + head_dim]);
                    } else {
                        // Current position
                        let curr_key = &current_k[head_offset..head_offset + head_dim];

                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_head[d] * curr_key[d];
                        }
                        tile_scores.push(score * scale);

                        tile_values.push(&current_v[head_offset..head_offset + head_dim]);
                    }
                }

                // Find max in this tile
                let m_tile = tile_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Update running max
                let m_new = m_i.max(m_tile);

                // Rescale factors for online softmax
                let scale_old = (m_i - m_new).exp();
                let scale_tile = (m_tile - m_new).exp();

                // Compute local softmax sum for this tile
                let l_tile: f32 = tile_scores.iter().map(|&s| (s - m_tile).exp()).sum();

                // Update running sum
                l_i = l_i * scale_old + l_tile * scale_tile;

                // Update output: rescale old output and add new contribution
                for o in &mut o_i {
                    *o *= scale_old;
                }

                // Add weighted values from this tile
                for (j, &score) in tile_scores.iter().enumerate() {
                    let attn_weight = (score - m_tile).exp() * scale_tile;
                    let v = tile_values[j];
                    for d in 0..head_dim {
                        o_i[d] += attn_weight * v[d];
                    }
                }

                m_i = m_new;
            }

            // Finalize: divide by sum
            if l_i > 0.0 {
                for d in 0..head_dim {
                    output[head_offset + d] = o_i[d] / l_i;
                }
            }
        }

        output
    }

    /// CPU fallback for flash_attention_tiled - uses standard attention
    #[cfg(not(feature = "gpu"))]
    #[allow(unused_variables)]
    pub fn flash_attention_tiled(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        block_size: usize,
    ) -> Vec<f32> {
        // FlashAttention is a GPU optimization; CPU path uses standard attention
        self.attention_with_cache(q, k_cache, v_cache, current_k, current_v)
    }
}
