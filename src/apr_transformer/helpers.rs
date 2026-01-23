//! APR Transformer Helper Functions (PMAT-802)
//!
//! Row-major matmul wrappers and SIMD primitives for APR inference.

use crate::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};


/// Row-major Q4K matmul wrapper (LAYOUT-001)
///
/// Wraps `fused_q4k_parallel_matvec` with dimension order matching the old API.
/// OLD API: `matmul_q4k_rowmajor(bytes, input, out_dim, in_dim)` - column-major, WRONG
/// NEW API: `matmul_q4k_rowmajor(bytes, input, out_dim, in_dim)` - row-major, CORRECT
///
/// FORBIDDEN: Never use `trueno::backends::q4k::matmul_q4k_f32_colmajor*` for GGUF/APR.
#[inline]
pub(crate) fn matmul_q4k_rowmajor(q4k_bytes: &[u8], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    // fused_q4k_parallel_matvec expects (bytes, input, in_dim, out_dim) - swap order!
    fused_q4k_parallel_matvec(q4k_bytes, input, in_dim, out_dim)
        .expect("Q4K matmul failed - check tensor dimensions")
}

/// Row-major Q6K matmul wrapper (LAYOUT-001)
#[inline]
pub(crate) fn matmul_q6k_rowmajor(q6k_bytes: &[u8], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    fused_q6k_parallel_matvec(q6k_bytes, input, in_dim, out_dim)
        .expect("Q6K matmul failed - check tensor dimensions")
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
        use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

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
