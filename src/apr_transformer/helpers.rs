//! APR Transformer Helper Functions (PMAT-802)
//!
//! Row-major matmul wrappers and SIMD primitives for APR inference.

use crate::error::Result;
use crate::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};

/// Row-major Q4K matmul wrapper (LAYOUT-001)
///
/// Wraps `fused_q4k_parallel_matvec` with dimension order matching the old API.
/// OLD API: `matmul_q4k_rowmajor(bytes, input, out_dim, in_dim)` - column-major, WRONG
/// NEW API: `matmul_q4k_rowmajor(bytes, input, out_dim, in_dim)` - row-major, CORRECT
///
/// FORBIDDEN: Never use `trueno::backends::q4k::matmul_q4k_f32_colmajor*` for GGUF/APR.
///
/// # Errors
///
/// Returns error if tensor dimensions are mismatched or data is corrupted.
#[inline]
pub(crate) fn matmul_q4k_rowmajor(
    q4k_bytes: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    // fused_q4k_parallel_matvec expects (bytes, input, in_dim, out_dim) - swap order!
    // AUDIT-301 FIX: Propagate error instead of expect()
    fused_q4k_parallel_matvec(q4k_bytes, input, in_dim, out_dim)
}

/// Row-major Q6K matmul wrapper (LAYOUT-001)
///
/// # Errors
///
/// Returns error if tensor dimensions are mismatched or data is corrupted.
#[inline]
pub(crate) fn matmul_q6k_rowmajor(
    q6k_bytes: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    // AUDIT-301 FIX: Propagate error instead of expect()
    fused_q6k_parallel_matvec(q6k_bytes, input, in_dim, out_dim)
}

// ============================================================================
// PMAT-103: SIMD Attention Primitives for 5.0+ tok/s target
// ============================================================================

/// SIMD dot product with AVX2 acceleration (PMAT-103)
///
/// Computes the dot product of two f32 slices using AVX2 when available.
/// Falls back to scalar when AVX2 is not supported or slices are small.
#[inline]
pub(crate) fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "SIMD dot: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && a.len() >= 8 {
            return unsafe { simd_dot_f32_avx2(a, b) };
        }
    }

    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// AVX2 dot product implementation (PMAT-103)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: Memory safety ensured by bounds checking before SIMD operations
    unsafe {
        use std::arch::x86_64::{
            _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
            _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
        };

        let n = a.len();
        let mut acc = _mm256_setzero_ps();

        // Process 8 elements at a time
        let chunks = n / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        // Horizontal sum of 8 floats
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let sum128 = _mm_hadd_ps(sum128, sum128);
        let sum128 = _mm_hadd_ps(sum128, sum128);
        let mut result = _mm_cvtss_f32(sum128);

        // Handle remaining elements
        let remainder = n % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..n {
                result += a[i] * b[i];
            }
        }

        result
    }
}

/// SIMD weighted accumulation: out[i] += weight * val[i] (PMAT-103)
///
/// Uses AVX2 FMA for efficient multiply-accumulate operations.
#[inline]
pub(crate) fn simd_add_weighted(out: &mut [f32], val: &[f32], weight: f32) {
    debug_assert_eq!(out.len(), val.len(), "SIMD add_weighted: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && out.len() >= 8 {
            // SAFETY: is_x86_feature_detected! ensures CPU supports AVX2/FMA before calling
            unsafe { simd_add_weighted_avx2(out, val, weight) };
            return;
        }
    }

    // Scalar fallback
    for (o, v) in out.iter_mut().zip(val.iter()) {
        *o += weight * v;
    }
}

/// AVX2 weighted accumulation implementation (PMAT-103)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_add_weighted_avx2(out: &mut [f32], val: &[f32], weight: f32) {
    // SAFETY: Memory safety ensured by bounds checking before SIMD operations
    unsafe {
        use std::arch::x86_64::{
            _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
        };

        let n = out.len();
        let w = _mm256_set1_ps(weight);

        // Process 8 elements at a time
        let chunks = n / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let v_out = _mm256_loadu_ps(out.as_ptr().add(offset));
            let v_val = _mm256_loadu_ps(val.as_ptr().add(offset));
            let result = _mm256_fmadd_ps(w, v_val, v_out);
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
        }

        // Handle remaining elements
        let remainder = n % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..n {
                out[i] += weight * val[i];
            }
        }
    }
}

// ============================================================================
// F32 Compute Helpers (PMAT-COMPLY: extracted from mod.rs)
// ============================================================================

/// F32 matrix-vector multiplication: output[out_dim] = weight[out_dim, in_dim] @ input[in_dim]
///
/// PMAT-095: Weights stored in matvec-optimal [out_dim, in_dim] format.
/// PMAT-103: 4-wide unrolled dot product for cache utilization.
pub(crate) fn f32_matmul(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let seq_len = input.len() / in_dim;
    let expected_size = in_dim * out_dim;

    if weight.len() != expected_size {
        return f32_matmul_scalar(input, weight, in_dim, out_dim);
    }

    let mut output = vec![0.0f32; seq_len * out_dim];

    for s in 0..seq_len {
        let input_start = s * in_dim;
        let input_slice = &input[input_start..input_start + in_dim];
        let out_start = s * out_dim;

        // PMAT-103: Unrolled dot product for better cache utilization
        let out_chunks = out_dim / 4;
        let out_remainder = out_dim % 4;

        for o_chunk in 0..out_chunks {
            let o_base = o_chunk * 4;
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;

            let w0_start = (o_base) * in_dim;
            let w1_start = (o_base + 1) * in_dim;
            let w2_start = (o_base + 2) * in_dim;
            let w3_start = (o_base + 3) * in_dim;

            for i in 0..in_dim {
                let x = input_slice[i];
                sum0 += x * weight[w0_start + i];
                sum1 += x * weight[w1_start + i];
                sum2 += x * weight[w2_start + i];
                sum3 += x * weight[w3_start + i];
            }

            output[out_start + o_base] = sum0;
            output[out_start + o_base + 1] = sum1;
            output[out_start + o_base + 2] = sum2;
            output[out_start + o_base + 3] = sum3;
        }

        // Handle remainder
        for o in (out_dim - out_remainder)..out_dim {
            let w_start = o * in_dim;
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += input_slice[i] * weight[w_start + i];
            }
            output[out_start + o] = sum;
        }
    }

    output
}

/// Scalar fallback for matmul (PMAT-095: weight is [out_dim, in_dim] row-major)
pub(crate) fn f32_matmul_scalar(
    input: &[f32],
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let seq_len = input.len() / in_dim;
    let mut output = Vec::with_capacity(seq_len * out_dim);

    for s in 0..seq_len {
        let input_start = s * in_dim;
        let input_slice = &input[input_start..input_start + in_dim];

        for o in 0..out_dim {
            let mut sum = 0.0;
            for (i, &input_val) in input_slice.iter().enumerate() {
                let weight_idx = o * in_dim + i;
                if weight_idx < weight.len() {
                    sum += input_val * weight[weight_idx];
                }
            }
            output.push(sum);
        }
    }

    output
}

/// Add bias in-place
pub(crate) fn add_bias_inplace(data: &mut [f32], bias: &[f32]) {
    let dim = bias.len();
    for (i, val) in data.iter_mut().enumerate() {
        *val += bias[i % dim];
    }
}

/// GELU activation in-place (tanh approximation)
pub(crate) fn gelu_inplace(data: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_COEFF: f32 = 0.044_715;

    for x in data.iter_mut() {
        let x3 = *x * *x * *x;
        let inner = SQRT_2_OVER_PI * (*x + GELU_COEFF * x3);
        *x = 0.5 * *x * (1.0 + inner.tanh());
    }
}

/// Apply Rotary Position Embedding (RoPE) to Q or K vectors
///
/// RoPE encodes position information by rotating pairs of elements
/// with position-dependent angles.
pub(crate) fn apply_rope_f32(
    x: &mut [f32],
    position: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    let half_dim = head_dim / 2;
    let pos_f32 = position as f32;
    let head_dim_f32 = head_dim as f32;

    for h in 0..num_heads {
        let head_start = h * head_dim;
        let idx2_start = head_start + half_dim;

        if idx2_start + half_dim > x.len() {
            continue;
        }

        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();

            let x1 = x[head_start + i];
            let x2 = x[idx2_start + i];

            x[head_start + i] = x1 * cos_val - x2 * sin_val;
            x[idx2_start + i] = x1 * sin_val + x2 * cos_val;
        }
    }
}

/// RMSNorm (Root Mean Square Layer Normalization)
///
/// PMAT-094 FIX: Qwen2, LLaMA, Mistral use RMSNorm, NOT LayerNorm.
/// Formula: output = x / sqrt(mean(x^2) + eps) * weight + bias
#[allow(clippy::cast_precision_loss)]
pub(crate) fn rms_norm(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());

    for s in 0..seq_len {
        let start = s * hidden_dim;
        let slice = &input[start..start + hidden_dim];

        let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

        for (i, &x) in slice.iter().enumerate() {
            let normalized = x / rms;
            let scaled = normalized * weight[i];
            let shifted = if let Some(b) = bias {
                scaled + b[i]
            } else {
                scaled
            };
            output.push(shifted);
        }
    }

    output
}

include!("helpers_part_02.rs");
