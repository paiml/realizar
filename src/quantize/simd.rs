//! Quantization SIMD Helpers (PMAT-802)
//!
//! Extracted from quantize/mod.rs - Shared SIMD utility functions.
//!
//! ## Contents
//! - f16 conversion: `f16_to_f32`, `read_f16`
//! - Scale extraction: `extract_scale_min`, `extract_scale_min_from_slice`
//! - Horizontal sum helpers: `hsum_epi32_128`, `hsum_epi32_256`, etc.
//! - SIMD activations: `softmax_simd`, `fused_swiglu_simd`, `apply_rope_rotation_simd`

// ============================================================================
// f16 Conversion (Manual Implementation)
// ============================================================================

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
///
/// Handles normal values, subnormals, infinities, and NaN.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mantissa = h & 0x3FF;

    if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            // Zero (preserve sign)
            if sign == 1 {
                -0.0
            } else {
                0.0
            }
        } else {
            // Subnormal: (mantissa / 1024) * 2^-14
            let value = (mantissa as f32 / 1024.0) * (2.0_f32).powi(-14);
            if sign == 1 {
                -value
            } else {
                value
            }
        }
    } else if exp == 31 {
        // Infinity or NaN
        if mantissa == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        // Normal value: (1 + mantissa/1024) * 2^(exp-15)
        let value = (1.0 + mantissa as f32 / 1024.0) * (2.0_f32).powi(exp as i32 - 15);
        if sign == 1 {
            -value
        } else {
            value
        }
    }
}

/// Helper: Read f16 from bytes and convert to f32
#[inline]
pub fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

// ============================================================================
// Scale Extraction for K-Quantization
// ============================================================================

/// Extract 6-bit scale and min values from packed scales array
///
/// PAR-001 FIX: Matches llama.cpp's get_scale_min_k4 packing scheme:
/// - Blocks 0-3: scale = q[j] & 63, min = q[j+4] & 63
/// - Blocks 4-7: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
///   min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
#[inline]
pub fn extract_scale_min(scales: &[u8; 12], block_idx: usize) -> (f32, f32) {
    let j = block_idx;
    let (scale_bits, min_bits) = if j < 4 {
        // First 4 blocks: simple layout
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        // Last 4 blocks: packed layout using high bits from first 4 bytes
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    };

    // Return raw 6-bit values as floats
    // The GGUF header's d/dmin values already include the /63 normalization
    let scale = f32::from(scale_bits);
    let min = f32::from(min_bits);

    (scale, min)
}

/// Extract scale and min from packed 6-bit scales (helper for InterleavedQ4K)
pub fn extract_scale_min_from_slice(scales: &[u8], idx: usize) -> (f32, f32) {
    // Same logic as extract_scale_min but works with slice
    let scale_idx = idx / 2;
    let min_idx = idx / 2 + 4;

    let (scale_raw, min_raw) = if idx.is_multiple_of(2) {
        (scales[scale_idx] & 0x3F, scales[min_idx] & 0x3F)
    } else {
        (
            (scales[scale_idx] >> 6) | ((scales[scale_idx + 2] & 0x0F) << 2),
            (scales[min_idx] >> 6) | ((scales[min_idx + 2] & 0x0F) << 2),
        )
    };

    (scale_raw as f32, min_raw as f32)
}

// ============================================================================
// x86_64 SIMD Horizontal Sum Helpers
// ============================================================================

/// Fast horizontal sum of 4 i32 in __m128i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_epi32_128(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::{_mm_cvtsi128_si32, _mm_hadd_epi32};
    let sum64 = _mm_hadd_epi32(v, v);
    let sum32 = _mm_hadd_epi32(sum64, sum64);
    _mm_cvtsi128_si32(sum32)
}

/// Fast horizontal sum of 8 i32 in __m256i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_epi32_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{_mm256_castsi256_si128, _mm256_extracti128_si256, _mm_add_epi32};
    // SAFETY: Unsafe operation with validated invariants
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        hsum_epi32_128(_mm_add_epi32(lo, hi))
    }
}

/// Helper: horizontal sum of 8 int32 values in a 256-bit register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn horizontal_sum_epi32_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm_add_epi32, _mm_cvtsi128_si32,
        _mm_hadd_epi32,
    };

    // Add high 128 bits to low 128 bits
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);

    // Horizontal add within 128 bits
    let sum64 = _mm_hadd_epi32(sum128, sum128);
    let sum32 = _mm_hadd_epi32(sum64, sum64);

    _mm_cvtsi128_si32(sum32)
}

/// Helper: horizontal sum of 16 int16 values in a 256-bit register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn horizontal_sum_epi16_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{_mm256_madd_epi16, _mm256_set1_epi16};

    // Use madd to sum pairs of i16 to i32
    let ones = _mm256_set1_epi16(1);
    let sum_i32 = _mm256_madd_epi16(v, ones);

    // Now sum the 8 i32 values
    horizontal_sum_epi32_256(sum_i32)
}

/// Helper: Horizontal sum of 8 i32 values to single i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // All intrinsics are unsafe and we're in an unsafe fn with target_feature
    let sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_10_10));
    _mm_cvtsi128_si32(sum32)
}

// ============================================================================
// SIMD Activation Functions
// ============================================================================

/// SIMD-accelerated in-place softmax
///
/// Uses AVX2 for parallel max-finding and normalization.
/// Falls back to scalar on non-x86_64 or when AVX2 is unavailable.
///
/// # Arguments
/// * `x` - Input/output slice, modified in-place to contain softmax values
pub fn softmax_simd(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && x.len() >= 8 {
            // SAFETY: AVX2 is available and slice is large enough
            unsafe {
                softmax_simd_avx2(x);
            }
            return;
        }
    }

    // Scalar fallback
    softmax_scalar(x);
}

/// Scalar softmax implementation
fn softmax_scalar(x: &mut [f32]) {
    // Find max for numerical stability
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for val in x.iter_mut() {
        *val *= inv_sum;
    }
}

/// AVX2 softmax implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn softmax_simd_avx2(x: &mut [f32]) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let len = x.len();

    // Phase 1: Find max using SIMD
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let chunks = len / 8;

    for i in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        max_vec = _mm256_max_ps(max_vec, v);
    }

    // Horizontal max
    let max_128 = _mm_max_ps(_mm256_castps256_ps128(max_vec), _mm256_extractf128_ps(max_vec, 1));
    let max_64 = _mm_max_ps(max_128, _mm_movehl_ps(max_128, max_128));
    let max_32 = _mm_max_ss(max_64, _mm_shuffle_ps(max_64, max_64, 1));
    let mut max_val = _mm_cvtss_f32(max_32);

    // Check remaining elements
    for i in (chunks * 8)..len {
        max_val = max_val.max(x[i]);
    }

    // Phase 2: Compute exp(x - max) and sum using SIMD
    let max_vec = _mm256_set1_ps(max_val);
    let mut sum_vec = _mm256_setzero_ps();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(v, max_vec);

        // Use polynomial approximation for exp (faster but less accurate)
        // For production, could use vectorized exp from a math library
        // Here we fall back to scalar exp for accuracy
        let mut exp_vals = [0.0f32; 8];
        let diff_arr: [f32; 8] = std::mem::transmute(diff);
        for (j, &d) in diff_arr.iter().enumerate() {
            exp_vals[j] = d.exp();
        }
        let exp_vec = _mm256_loadu_ps(exp_vals.as_ptr());

        _mm256_storeu_ps(x.as_mut_ptr().add(i * 8), exp_vec);
        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
    }

    // Handle remaining elements
    let mut sum_scalar = 0.0f32;
    for i in (chunks * 8)..len {
        x[i] = (x[i] - max_val).exp();
        sum_scalar += x[i];
    }

    // Horizontal sum
    let sum_128 = _mm_add_ps(_mm256_castps256_ps128(sum_vec), _mm256_extractf128_ps(sum_vec, 1));
    let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
    let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));
    let sum = _mm_cvtss_f32(sum_32) + sum_scalar;

    // Phase 3: Normalize
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = _mm256_set1_ps(inv_sum);

    for i in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        let normalized = _mm256_mul_ps(v, inv_sum_vec);
        _mm256_storeu_ps(x.as_mut_ptr().add(i * 8), normalized);
    }

    for i in (chunks * 8)..len {
        x[i] *= inv_sum;
    }
}

/// Fused SiLU (Swish) with gating: gate = gate * sigmoid(gate) * up
///
/// Modifies `gate` in-place: gate[i] = silu(gate[i]) * up[i]
///
/// # Arguments
/// * `gate` - Gate values, modified in-place
/// * `up` - Up-projection values (must be same length as gate)
///
/// # Panics
/// Panics if `gate.len() != up.len()`
pub fn fused_swiglu_simd(gate: &mut [f32], up: &[f32]) {
    assert_eq!(
        gate.len(),
        up.len(),
        "fused_swiglu_simd: gate and up must have same length"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && gate.len() >= 8 {
            // SAFETY: AVX2+FMA available, slices are same length
            unsafe {
                fused_swiglu_simd_avx2(gate, up);
            }
            return;
        }
    }

    // Scalar fallback
    fused_swiglu_scalar(gate, up);
}

/// Scalar SwiGLU implementation
fn fused_swiglu_scalar(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        let silu = *g / (1.0 + (-*g).exp());
        *g = silu * u;
    }
}

/// AVX2 SwiGLU implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_swiglu_simd_avx2(gate: &mut [f32], up: &[f32]) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let len = gate.len();
    let chunks = len / 8;
    let one = _mm256_set1_ps(1.0);

    for i in 0..chunks {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i * 8));
        let u = _mm256_loadu_ps(up.as_ptr().add(i * 8));

        // Compute sigmoid using scalar (for accuracy)
        // A fully vectorized version would use polynomial approximation
        let mut sigmoid_vals = [0.0f32; 8];
        let g_arr: [f32; 8] = std::mem::transmute(g);
        for (j, &gv) in g_arr.iter().enumerate() {
            sigmoid_vals[j] = 1.0 / (1.0 + (-gv).exp());
        }
        let sigmoid = _mm256_loadu_ps(sigmoid_vals.as_ptr());

        // SiLU = g * sigmoid(g)
        let silu = _mm256_mul_ps(g, sigmoid);

        // Result = silu * up
        let result = _mm256_mul_ps(silu, u);

        _mm256_storeu_ps(gate.as_mut_ptr().add(i * 8), result);
    }

    // Handle remaining elements
    for i in (chunks * 8)..len {
        let g = gate[i];
        let sigmoid = 1.0 / (1.0 + (-g).exp());
        gate[i] = g * sigmoid * up[i];
    }
}

/// Apply RoPE (Rotary Position Embeddings) rotation with SIMD
///
/// Applies rotary position encoding to query/key vectors.
///
/// # Arguments
/// * `x` - Input tensor, modified in-place (shape: [seq_len, num_heads, head_dim])
/// * `freqs_cis` - Complex exponentials (cos, sin) for each position
/// * `head_dim` - Dimension per head (must be even)
pub fn apply_rope_rotation_simd(x: &mut [f32], freqs_cos: &[f32], freqs_sin: &[f32], head_dim: usize) {
    debug_assert!(head_dim.is_multiple_of(2), "head_dim must be even");
    debug_assert_eq!(freqs_cos.len(), freqs_sin.len());

    let half_dim = head_dim / 2;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && half_dim >= 8 {
            // SAFETY: AVX2+FMA available
            unsafe {
                apply_rope_rotation_avx2(x, freqs_cos, freqs_sin, head_dim);
            }
            return;
        }
    }

    // Scalar fallback
    apply_rope_rotation_scalar(x, freqs_cos, freqs_sin, head_dim);
}

/// Scalar RoPE implementation
fn apply_rope_rotation_scalar(x: &mut [f32], freqs_cos: &[f32], freqs_sin: &[f32], head_dim: usize) {
    let half_dim = head_dim / 2;

    // Process pairs of values
    for i in 0..half_dim {
        if i >= freqs_cos.len() {
            break;
        }

        let cos = freqs_cos[i];
        let sin = freqs_sin[i];

        let x0 = x[i];
        let x1 = x[i + half_dim];

        // Apply rotation: [cos -sin; sin cos] @ [x0; x1]
        x[i] = x0 * cos - x1 * sin;
        x[i + half_dim] = x0 * sin + x1 * cos;
    }
}

/// AVX2 RoPE implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn apply_rope_rotation_avx2(
    x: &mut [f32],
    freqs_cos: &[f32],
    freqs_sin: &[f32],
    head_dim: usize,
) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let half_dim = head_dim / 2;
    let chunks = half_dim / 8;

    for i in 0..chunks {
        let offset = i * 8;

        // Load x0 and x1 (first and second half of dimension)
        let x0 = _mm256_loadu_ps(x.as_ptr().add(offset));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(offset + half_dim));

        // Load cos and sin values
        let cos = _mm256_loadu_ps(freqs_cos.as_ptr().add(offset));
        let sin = _mm256_loadu_ps(freqs_sin.as_ptr().add(offset));

        // Compute: x0' = x0 * cos - x1 * sin
        let x0_new = _mm256_fmsub_ps(x0, cos, _mm256_mul_ps(x1, sin));

        // Compute: x1' = x0 * sin + x1 * cos
        let x1_new = _mm256_fmadd_ps(x0, sin, _mm256_mul_ps(x1, cos));

        // Store results
        _mm256_storeu_ps(x.as_mut_ptr().add(offset), x0_new);
        _mm256_storeu_ps(x.as_mut_ptr().add(offset + half_dim), x1_new);
    }

    // Handle remaining elements
    let remaining_start = chunks * 8;
    for i in remaining_start..half_dim {
        if i >= freqs_cos.len() {
            break;
        }

        let cos = freqs_cos[i];
        let sin = freqs_sin[i];

        let x0 = x[i];
        let x1 = x[i + half_dim];

        x[i] = x0 * cos - x1 * sin;
        x[i + half_dim] = x0 * sin + x1 * cos;
    }
}
