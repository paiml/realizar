
/// AVX2 SIMD dequantization for a single Q8_0 block
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_q8_0_block_avx2(block_data: &[u8]) -> Vec<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let mut result = vec![0.0f32; 32];

    // Read scale (f16 -> f32)
    let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
    let scale = f16_to_f32(scale_bits);

    // SAFETY: AVX2 availability verified by caller's target_feature
    unsafe {
        let scale_vec = _mm256_set1_ps(scale);

        // Process 32 i8 values in 4 iterations of 8
        for chunk in 0..4 {
            let byte_start = 2 + chunk * 8; // Start at offset 2 (after f16 scale)

            // Load 8 i8 values and sign-extend to i32
            let q0 = block_data[byte_start] as i8 as i32;
            let q1 = block_data[byte_start + 1] as i8 as i32;
            let q2 = block_data[byte_start + 2] as i8 as i32;
            let q3 = block_data[byte_start + 3] as i8 as i32;
            let q4 = block_data[byte_start + 4] as i8 as i32;
            let q5 = block_data[byte_start + 5] as i8 as i32;
            let q6 = block_data[byte_start + 6] as i8 as i32;
            let q7 = block_data[byte_start + 7] as i8 as i32;

            let q_vec = _mm256_setr_epi32(q0, q1, q2, q3, q4, q5, q6, q7);
            let q_f32 = _mm256_cvtepi32_ps(q_vec);

            // Multiply by scale
            let dequant = _mm256_mul_ps(scale_vec, q_f32);

            // Store 8 results
            _mm256_storeu_ps(result.as_mut_ptr().add(chunk * 8), dequant);
        }
    }

    result
}

// DequantStats, SimdBackend, detect_simd_backend moved to types.rs (PMAT-802)

/// SIMD-optimized RoPE rotation for a single head
///
/// Applies rotary position embedding rotation to a single attention head:
/// x1[i] = x1[i] * cos[i] - x2[i] * sin[i]
/// x2[i] = x1[i] * sin[i] + x2[i] * cos[i]
///
/// # Arguments
/// * `x1` - First half of head (will be modified in-place)
/// * `x2` - Second half of head (will be modified in-place)
/// * `cos_vals` - Precomputed cosine values
/// * `sin_vals` - Precomputed sine values
#[inline]
pub fn apply_rope_rotation_simd(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    debug_assert_eq!(x1.len(), x2.len());
    debug_assert_eq!(x1.len(), cos_vals.len());
    debug_assert_eq!(x1.len(), sin_vals.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                apply_rope_rotation_avx512(x1, x2, cos_vals, sin_vals);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                apply_rope_rotation_avx2(x1, x2, cos_vals, sin_vals);
            }
            return;
        }
    }

    // Scalar fallback
    apply_rope_rotation_scalar(x1, x2, cos_vals, sin_vals);
}

/// Scalar fallback for RoPE rotation
///
/// Exposed as `pub(crate)` for direct testing on AVX2 machines.
#[inline]
pub(crate) fn apply_rope_rotation_scalar(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    for i in 0..x1.len() {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn apply_rope_rotation_avx2(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
    };

    let n = x1.len();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= n {
        let v1 = _mm256_loadu_ps(x1.as_ptr().add(i));
        let v2 = _mm256_loadu_ps(x2.as_ptr().add(i));
        let cos_v = _mm256_loadu_ps(cos_vals.as_ptr().add(i));
        let sin_v = _mm256_loadu_ps(sin_vals.as_ptr().add(i));

        // r1 = v1 * cos - v2 * sin
        let v1_cos = _mm256_mul_ps(v1, cos_v);
        let r1 = _mm256_fnmadd_ps(v2, sin_v, v1_cos);

        // r2 = v1 * sin + v2 * cos
        let v1_sin = _mm256_mul_ps(v1, sin_v);
        let r2 = _mm256_fmadd_ps(v2, cos_v, v1_sin);

        _mm256_storeu_ps(x1.as_mut_ptr().add(i), r1);
        _mm256_storeu_ps(x2.as_mut_ptr().add(i), r2);

        i += 8;
    }

    // Handle remainder
    while i < n {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn apply_rope_rotation_avx512(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_storeu_ps,
    };

    let n = x1.len();
    let mut i = 0;

    // Process 16 elements at a time with AVX-512
    while i + 16 <= n {
        let v1 = _mm512_loadu_ps(x1.as_ptr().add(i));
        let v2 = _mm512_loadu_ps(x2.as_ptr().add(i));
        let cos_v = _mm512_loadu_ps(cos_vals.as_ptr().add(i));
        let sin_v = _mm512_loadu_ps(sin_vals.as_ptr().add(i));

        // r1 = v1 * cos - v2 * sin
        let v1_cos = _mm512_mul_ps(v1, cos_v);
        let r1 = _mm512_fnmadd_ps(v2, sin_v, v1_cos);

        // r2 = v1 * sin + v2 * cos
        let v1_sin = _mm512_mul_ps(v1, sin_v);
        let r2 = _mm512_fmadd_ps(v2, cos_v, v1_sin);

        _mm512_storeu_ps(x1.as_mut_ptr().add(i), r1);
        _mm512_storeu_ps(x2.as_mut_ptr().add(i), r2);

        i += 16;
    }

    // Handle remainder with AVX2 or scalar
    while i < n {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
        i += 1;
    }
}
