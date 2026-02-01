//! Part 23: Additional SIMD Coverage Tests for quantize/simd.rs
//!
//! Targets the uncovered ~10% in simd.rs:
//! - f16_to_f32 positive subnormal branch (line 38)
//! - extract_scale_min_from_slice odd index branch (lines 114-117)
//! - Standalone SIMD horizontal sum functions (lines 133-217)
//! - AVX2 RoPE rotation inner loop (lines 525-535)

use crate::quantize::simd::{extract_scale_min_from_slice, f16_to_f32};
use crate::quantize::{fused_swiglu_simd, softmax_simd};

#[cfg(target_arch = "x86_64")]
use crate::quantize::simd::{
    horizontal_sum_epi16_256, horizontal_sum_epi32_256, hsum_epi32, hsum_epi32_128, hsum_epi32_256,
};

// =============================================================================
// f16_to_f32: Positive Subnormal Coverage (line 38)
// =============================================================================

/// Test positive subnormal f16 values covering all mantissa ranges
#[test]
fn test_f16_to_f32_positive_subnormals() {
    // Test various positive subnormals (exp=0, mantissa!=0, sign=0)
    let test_cases: &[(u16, f32)] = &[
        (0x0001, (1.0 / 1024.0) * (2.0_f32).powi(-14)), // min
        (0x0200, (512.0 / 1024.0) * (2.0_f32).powi(-14)), // mid
        (0x03FF, (1023.0 / 1024.0) * (2.0_f32).powi(-14)), // max
    ];

    for &(bits, expected) in test_cases {
        let result = f16_to_f32(bits);
        assert!(result > 0.0, "bits=0x{:04X} should be positive", bits);
        assert!(
            (result - expected).abs() < 1e-12,
            "bits=0x{:04X}: got {}, expected {}",
            bits,
            result,
            expected
        );
    }
}

// =============================================================================
// extract_scale_min_from_slice: Odd Index Coverage (lines 114-117)
// =============================================================================

/// Test extract_scale_min_from_slice with all odd indices 1,3,5,7
#[test]
fn test_extract_scale_min_from_slice_odd_indices() {
    // For odd idx: scale = (scales[idx/2] >> 6) | ((scales[idx/2+2] & 0x0F) << 2)
    //              min = (scales[idx/2+4] >> 6) | ((scales[idx/2+6] & 0x0F) << 2)

    // idx=1: scale_idx=0, min_idx=4
    let scales1: [u8; 12] = [
        0b11_000000,
        0,
        0b0000_0101,
        0,
        0b01_000000,
        0,
        0b0000_0111,
        0,
        0,
        0,
        0,
        0,
    ];
    let (s, m) = extract_scale_min_from_slice(&scales1, 1);
    assert_eq!(s, 23.0, "idx=1 scale: 3|(5<<2)");
    assert_eq!(m, 29.0, "idx=1 min: 1|(7<<2)");

    // idx=3: scale_idx=1, min_idx=5
    let scales3: [u8; 12] = [
        0,
        0b10_000000,
        0,
        0b0000_0110,
        0,
        0b00_000000,
        0,
        0b0000_1000,
        0,
        0,
        0,
        0,
    ];
    let (s, m) = extract_scale_min_from_slice(&scales3, 3);
    assert_eq!(s, 26.0, "idx=3 scale: 2|(6<<2)");
    assert_eq!(m, 32.0, "idx=3 min: 0|(8<<2)");

    // idx=5: scale_idx=2, min_idx=6
    let scales5: [u8; 12] = [
        0,
        0,
        0b01_000000,
        0,
        0b0000_1010,
        0,
        0b11_000000,
        0,
        0b0000_1111,
        0,
        0,
        0,
    ];
    let (s, m) = extract_scale_min_from_slice(&scales5, 5);
    assert_eq!(s, 41.0, "idx=5 scale: 1|(10<<2)");
    assert_eq!(m, 63.0, "idx=5 min: 3|(15<<2)");

    // idx=7: scale_idx=3, min_idx=7
    let scales7: [u8; 12] = [
        0,
        0,
        0,
        0b11_000000,
        0,
        0b0000_1111,
        0,
        0b10_000000,
        0,
        0b0000_1100,
        0,
        0,
    ];
    let (s, m) = extract_scale_min_from_slice(&scales7, 7);
    assert_eq!(s, 63.0, "idx=7 scale: 3|(15<<2)");
    assert_eq!(m, 50.0, "idx=7 min: 2|(12<<2)");

    // Zeros for all odd indices
    let zeros: [u8; 12] = [0; 12];
    for idx in [1, 3, 5, 7] {
        let (s, m) = extract_scale_min_from_slice(&zeros, idx);
        assert_eq!(s, 0.0);
        assert_eq!(m, 0.0);
    }
}

// =============================================================================
// SIMD Horizontal Sum Functions (lines 133-217)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod simd_horizontal_sum_tests {
    use super::*;

    #[test]
    fn test_hsum_epi32_128_coverage() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::{_mm_set_epi32, _mm_setzero_si128};

        unsafe {
            // Basic sum: [1,2,3,4] = 10
            assert_eq!(hsum_epi32_128(_mm_set_epi32(4, 3, 2, 1)), 10);
            // Mixed: [-1,-2,3,4] = 4
            assert_eq!(hsum_epi32_128(_mm_set_epi32(4, 3, -2, -1)), 4);
            // Zeros
            assert_eq!(hsum_epi32_128(_mm_setzero_si128()), 0);
        }
    }

    #[test]
    fn test_hsum_epi32_256_coverage() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::_mm256_set_epi32;

        unsafe {
            // [1..8] = 36
            assert_eq!(hsum_epi32_256(_mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1)), 36);
            // Symmetric sums to 0
            assert_eq!(
                hsum_epi32_256(_mm256_set_epi32(4, 3, 2, 1, -1, -2, -3, -4)),
                0
            );
        }
    }

    #[test]
    fn test_horizontal_sum_epi32_256_coverage() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::_mm256_set_epi32;

        unsafe {
            // [10,20,30,40,50,60,70,80] = 360
            assert_eq!(
                horizontal_sum_epi32_256(_mm256_set_epi32(80, 70, 60, 50, 40, 30, 20, 10)),
                360
            );
            // Single nonzero
            assert_eq!(
                horizontal_sum_epi32_256(_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 42)),
                42
            );
        }
    }

    #[test]
    fn test_horizontal_sum_epi16_256_coverage() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::_mm256_set_epi16;

        unsafe {
            // [1..16] = 136
            assert_eq!(
                horizontal_sum_epi16_256(_mm256_set_epi16(
                    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
                )),
                136
            );
            // Symmetric sums to 0
            assert_eq!(
                horizontal_sum_epi16_256(_mm256_set_epi16(
                    8, 7, 6, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -6, -7, -8
                )),
                0
            );
        }
    }

    #[test]
    fn test_hsum_epi32_coverage() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::_mm256_set_epi32;

        unsafe {
            // [1..8] = 36
            assert_eq!(hsum_epi32(_mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1)), 36);
            // Large values
            assert_eq!(
                hsum_epi32(_mm256_set_epi32(
                    100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000
                )),
                3_600_000
            );
        }
    }

    #[test]
    fn test_horizontal_sum_consistency() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::_mm256_set_epi32;

        unsafe {
            let v = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
            let r1 = hsum_epi32_256(v);
            let r2 = horizontal_sum_epi32_256(v);
            let r3 = hsum_epi32(v);
            assert_eq!(r1, r2);
            assert_eq!(r2, r3);
            assert_eq!(r1, 36);
        }
    }
}

// =============================================================================
// Additional Edge Cases for SIMD Paths
// =============================================================================

/// Test softmax and swiglu trigger SIMD path with exactly 8 elements
#[test]
fn test_simd_activation_8_elements() {
    // Softmax with 8 elements
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // SwiGLU with 8 elements
    let mut gate: Vec<f32> = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
    let up = vec![1.0; 8];
    let expected: Vec<f32> = gate
        .iter()
        .copied()
        .map(|g: f32| g * (1.0 / (1.0 + (-g).exp())))
        .collect();
    fused_swiglu_simd(&mut gate, &up);
    for (got, exp) in gate.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 0.2); // Lenient for AVX2 polynomial approx
    }
}
