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
// Tests for SIMD Helpers (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // simd_dot_f32 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_dot_f32_basic() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = simd_dot_f32(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((result - 70.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_zeros() {
        let a = vec![0.0f32; 16];
        let b = vec![1.0f32; 16];
        let result = simd_dot_f32(&a, &b);
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_ones() {
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 16];
        let result = simd_dot_f32(&a, &b);
        assert!((result - 16.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_large_vector() {
        // Test with vector size > 8 to exercise AVX2 path
        let n = 64;
        let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let b = vec![1.0f32; n];
        let result = simd_dot_f32(&a, &b);
        // Sum of 1 to 64 = 64*65/2 = 2080
        assert!((result - 2080.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_remainder() {
        // Test with size not divisible by 8 to exercise remainder handling
        let n = 13;
        let a = vec![2.0f32; n];
        let b = vec![3.0f32; n];
        let result = simd_dot_f32(&a, &b);
        // 2*3*13 = 78
        assert!((result - 78.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_small_vector() {
        // Test with size < 8 to exercise scalar fallback
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = simd_dot_f32(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_negative() {
        let a = vec![-1.0f32, 2.0, -3.0, 4.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0];
        let result = simd_dot_f32(&a, &b);
        // -1 + 2 - 3 + 4 = 2
        assert!((result - 2.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // simd_add_weighted Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_add_weighted_basic() {
        let mut out = vec![1.0f32, 2.0, 3.0, 4.0];
        let val = vec![10.0f32, 20.0, 30.0, 40.0];
        simd_add_weighted(&mut out, &val, 0.5);
        // out[i] = out[i] + 0.5 * val[i]
        assert!((out[0] - 6.0).abs() < 0.001); // 1 + 0.5*10 = 6
        assert!((out[1] - 12.0).abs() < 0.001); // 2 + 0.5*20 = 12
        assert!((out[2] - 18.0).abs() < 0.001); // 3 + 0.5*30 = 18
        assert!((out[3] - 24.0).abs() < 0.001); // 4 + 0.5*40 = 24
    }

    #[test]
    fn test_simd_add_weighted_zero_weight() {
        let mut out = vec![1.0f32, 2.0, 3.0, 4.0];
        let val = vec![100.0f32; 4];
        simd_add_weighted(&mut out, &val, 0.0);
        // out should remain unchanged
        assert!((out[0] - 1.0).abs() < 0.001);
        assert!((out[1] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_add_weighted_large_vector() {
        // Test with vector size > 8 to exercise AVX2 path
        let n = 64;
        let mut out = vec![1.0f32; n];
        let val = vec![2.0f32; n];
        simd_add_weighted(&mut out, &val, 3.0);
        // out[i] = 1 + 3*2 = 7
        for &v in &out {
            assert!((v - 7.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_simd_add_weighted_remainder() {
        // Test with size not divisible by 8
        let n = 11;
        let mut out = vec![0.0f32; n];
        let val: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        simd_add_weighted(&mut out, &val, 2.0);
        // out[i] = 0 + 2*val[i]
        for (i, &v) in out.iter().enumerate() {
            let expected = 2.0 * (i + 1) as f32;
            assert!((v - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_simd_add_weighted_small_vector() {
        // Test with size < 8 to exercise scalar fallback
        let mut out = vec![5.0f32, 10.0];
        let val = vec![1.0f32, 1.0];
        simd_add_weighted(&mut out, &val, -2.0);
        // out[0] = 5 + (-2)*1 = 3
        // out[1] = 10 + (-2)*1 = 8
        assert!((out[0] - 3.0).abs() < 0.001);
        assert!((out[1] - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_add_weighted_accumulate() {
        // Test multiple accumulations
        let mut out = vec![0.0f32; 16];
        let val1 = vec![1.0f32; 16];
        let val2 = vec![2.0f32; 16];

        simd_add_weighted(&mut out, &val1, 1.0);
        simd_add_weighted(&mut out, &val2, 0.5);
        // out = 0 + 1*1 + 0.5*2 = 2
        for &v in &out {
            assert!((v - 2.0).abs() < 0.001);
        }
    }
}
