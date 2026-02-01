//! Quantization SIMD Helpers (PMAT-802)
//!
//! Extracted from quantize/mod.rs - Shared SIMD utility functions.
//!
//! ## Contents
//! - f16 conversion: `f16_to_f32`, `read_f16`
//! - Scale extraction: `extract_scale_min`, `extract_scale_min_from_slice`
//! - Horizontal sum helpers: `hsum_epi32_128`, `hsum_epi32_256`, etc.
//!
//! Note: SIMD activations (`softmax_simd`, `fused_swiglu_simd`, `apply_rope_rotation_simd`)
//! are exported from `activation.rs` and `parallel_dequant.rs` respectively.

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
///
/// # Safety
/// Requires AVX2 support. Caller must verify CPU feature availability.
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
///
/// # Safety
/// Requires AVX2 support. Caller must verify CPU feature availability.
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
///
/// # Safety
/// Requires AVX2 support. Caller must verify CPU feature availability.
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
///
/// # Safety
/// Requires AVX2 support. Caller must verify CPU feature availability.
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
///
/// # Safety
/// Requires AVX2 support. Caller must verify CPU feature availability.
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

#[cfg(test)]
mod tests {
    use super::*;

    // ============= f16_to_f32 tests =============

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        let val = f16_to_f32(0x8000);
        assert!(val == 0.0 && val.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_one() {
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        let val = f16_to_f32(0xBC00);
        assert!((val - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_two() {
        let val = f16_to_f32(0x4000);
        assert!((val - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_half() {
        let val = f16_to_f32(0x3800);
        assert!((val - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn test_f16_to_f32_neg_infinity() {
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        // Smallest positive subnormal: 2^-24
        let val = f16_to_f32(0x0001);
        assert!(val > 0.0);
        assert!(val < 0.0001);
    }

    // ============= read_f16 tests =============

    #[test]
    fn test_read_f16_one() {
        let bytes = 0x3C00u16.to_le_bytes();
        let val = read_f16(&bytes);
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_read_f16_two() {
        let bytes = 0x4000u16.to_le_bytes();
        let val = read_f16(&bytes);
        assert!((val - 2.0).abs() < 0.001);
    }

    // ============= extract_scale_min tests =============

    #[test]
    fn test_extract_scale_min_first_block() {
        let scales: [u8; 12] = [10, 20, 30, 40, 5, 15, 25, 35, 0, 0, 0, 0];
        let (scale, min) = extract_scale_min(&scales, 0);
        assert_eq!(scale, 10.0); // scales[0] & 63
        assert_eq!(min, 5.0); // scales[4] & 63
    }

    #[test]
    fn test_extract_scale_min_second_block() {
        let scales: [u8; 12] = [10, 20, 30, 40, 5, 15, 25, 35, 0, 0, 0, 0];
        let (scale, min) = extract_scale_min(&scales, 1);
        assert_eq!(scale, 20.0);
        assert_eq!(min, 15.0);
    }

    #[test]
    fn test_extract_scale_min_packed_block() {
        // Block 4+ uses packed layout
        let mut scales: [u8; 12] = [0; 12];
        scales[0] = 0b11_000010; // high bits for block 4
        scales[8] = 0b0011_0001; // low bits: scale=0x31, min high nibble=0x3

        let (scale, min) = extract_scale_min(&scales, 4);
        // scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = 1 | (3 << 4) = 1 | 48 = 49
        // min = (scales[8] >> 4) | ((scales[4] >> 6) << 4) = 3 | 0 = 3
        assert_eq!(scale, 49.0);
        assert_eq!(min, 3.0);
    }

    // ============= extract_scale_min_from_slice tests =============

    #[test]
    fn test_extract_scale_min_from_slice_even() {
        let scales: [u8; 8] = [10, 20, 30, 40, 5, 15, 25, 35];
        let (scale, min) = extract_scale_min_from_slice(&scales, 0);
        assert_eq!(scale, 10.0);
        assert_eq!(min, 5.0);
    }

    // ============= hsum tests (x86_64 only) =============

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_hsum_epi32_128() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            use std::arch::x86_64::_mm_setr_epi32;
            let v = _mm_setr_epi32(1, 2, 3, 4);
            let sum = hsum_epi32_128(v);
            assert_eq!(sum, 10);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_hsum_epi32_256() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            use std::arch::x86_64::_mm256_setr_epi32;
            let v = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
            let sum = hsum_epi32_256(v);
            assert_eq!(sum, 36);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_horizontal_sum_epi32_256() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            use std::arch::x86_64::_mm256_setr_epi32;
            let v = _mm256_setr_epi32(10, 20, 30, 40, 50, 60, 70, 80);
            let sum = horizontal_sum_epi32_256(v);
            assert_eq!(sum, 360);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_horizontal_sum_epi16_256() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            use std::arch::x86_64::_mm256_setr_epi16;
            let v = _mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
            let sum = horizontal_sum_epi16_256(v);
            assert_eq!(sum, 136); // 1+2+...+16 = 136
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_hsum_epi32() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            use std::arch::x86_64::_mm256_setr_epi32;
            let v = _mm256_setr_epi32(-1, 2, -3, 4, -5, 6, -7, 8);
            let sum = hsum_epi32(v);
            assert_eq!(sum, 4); // (-1+2-3+4-5+6-7+8) = 4
        }
    }
}
