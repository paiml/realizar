//! Part 23: Additional SIMD Coverage Tests for quantize/simd.rs
//!
//! Targets the uncovered ~10% in simd.rs:
//! - f16_to_f32 positive subnormal branch (line 38)
//! - extract_scale_min_from_slice odd index branch (lines 114-117)
//! - Standalone SIMD horizontal sum functions (lines 133-217)
//! - AVX2 RoPE rotation inner loop (lines 525-535)
//!
//! Focus: Direct testing of unsafe SIMD primitives and edge cases.

use crate::quantize::simd::{
    apply_rope_rotation_simd, extract_scale_min_from_slice, f16_to_f32, fused_swiglu_simd,
    softmax_simd,
};

#[cfg(target_arch = "x86_64")]
use crate::quantize::simd::{
    horizontal_sum_epi16_256, horizontal_sum_epi32_256, hsum_epi32, hsum_epi32_128, hsum_epi32_256,
};

// =============================================================================
// f16_to_f32: Positive Subnormal Coverage (line 38)
// =============================================================================

/// Test positive subnormal f16 values (exp=0, mantissa!=0, sign=0)
#[test]
fn test_f16_to_f32_positive_subnormal_smallest() {
    // 0x0001 = positive smallest subnormal (sign=0, exp=0, mantissa=1)
    // Value = (1/1024) * 2^-14 = 2^-24 ≈ 5.96e-8
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0, "Positive subnormal should be positive: {}", result);

    let expected = (1.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-12,
        "Positive subnormal 0x0001: got {}, expected {}",
        result,
        expected
    );
}

/// Test positive subnormal with mid-range mantissa
#[test]
fn test_f16_to_f32_positive_subnormal_mid() {
    // 0x0200 = positive subnormal with mantissa=512 (sign=0, exp=0, mantissa=512)
    // Value = (512/1024) * 2^-14 = 0.5 * 2^-14 ≈ 3.05e-5
    let result = f16_to_f32(0x0200);
    assert!(result > 0.0, "Positive subnormal should be positive: {}", result);

    let expected = (512.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-10,
        "Positive subnormal 0x0200: got {}, expected {}",
        result,
        expected
    );
}

/// Test positive subnormal with max mantissa
#[test]
fn test_f16_to_f32_positive_subnormal_max() {
    // 0x03FF = positive largest subnormal (sign=0, exp=0, mantissa=1023)
    // Value = (1023/1024) * 2^-14 ≈ 6.10e-5
    let result = f16_to_f32(0x03FF);
    assert!(result > 0.0, "Positive max subnormal should be positive: {}", result);

    let expected = (1023.0 / 1024.0) * (2.0_f32).powi(-14);
    assert!(
        (result - expected).abs() < 1e-10,
        "Positive max subnormal: got {}, expected {}",
        result,
        expected
    );
}

/// Test various positive subnormal patterns
#[test]
fn test_f16_to_f32_positive_subnormal_various() {
    let test_cases: &[(u16, &str)] = &[
        (0x0001, "min"),
        (0x0100, "mantissa=256"),
        (0x0200, "mantissa=512"),
        (0x0300, "mantissa=768"),
        (0x03FF, "max"),
    ];

    for &(bits, desc) in test_cases {
        let result = f16_to_f32(bits);
        assert!(
            result > 0.0 && result.is_finite(),
            "Positive subnormal {}: got {}",
            desc,
            result
        );

        // Verify against formula: (mantissa/1024) * 2^-14
        let mantissa = (bits & 0x3FF) as f32;
        let expected = (mantissa / 1024.0) * (2.0_f32).powi(-14);
        assert!(
            (result - expected).abs() < 1e-12,
            "Positive subnormal {}: got {}, expected {}",
            desc,
            result,
            expected
        );
    }
}

// =============================================================================
// extract_scale_min_from_slice: Odd Index Coverage (lines 114-117)
// =============================================================================

/// Test extract_scale_min_from_slice with idx=1 (odd)
#[test]
fn test_extract_scale_min_from_slice_idx_1() {
    // For odd idx=1:
    // scale_idx = 0, min_idx = 4
    // scale = (scales[0] >> 6) | ((scales[2] & 0x0F) << 2)
    // min = (scales[4] >> 6) | ((scales[6] & 0x0F) << 2)

    let scales: [u8; 12] = [
        0b11_000000, // byte 0: high bits = 3
        0,
        0b0000_0101, // byte 2: low nibble = 5
        0,
        0b01_000000, // byte 4: high bits = 1
        0,
        0b0000_0111, // byte 6: low nibble = 7
        0,
        0,
        0,
        0,
        0,
    ];

    let (s, m) = extract_scale_min_from_slice(&scales, 1);
    // scale = 3 | (5 << 2) = 3 | 20 = 23
    // min = 1 | (7 << 2) = 1 | 28 = 29
    assert_eq!(s, 23.0, "idx=1 scale");
    assert_eq!(m, 29.0, "idx=1 min");
}

/// Test extract_scale_min_from_slice with idx=3 (odd)
#[test]
fn test_extract_scale_min_from_slice_idx_3() {
    // For odd idx=3:
    // scale_idx = 1, min_idx = 5
    // scale = (scales[1] >> 6) | ((scales[3] & 0x0F) << 2)
    // min = (scales[5] >> 6) | ((scales[7] & 0x0F) << 2)

    let scales: [u8; 12] = [
        0,
        0b10_000000, // byte 1: high bits = 2
        0,
        0b0000_0110, // byte 3: low nibble = 6
        0,
        0b00_000000, // byte 5: high bits = 0
        0,
        0b0000_1000, // byte 7: low nibble = 8
        0,
        0,
        0,
        0,
    ];

    let (s, m) = extract_scale_min_from_slice(&scales, 3);
    // scale = 2 | (6 << 2) = 2 | 24 = 26
    // min = 0 | (8 << 2) = 0 | 32 = 32
    assert_eq!(s, 26.0, "idx=3 scale");
    assert_eq!(m, 32.0, "idx=3 min");
}

/// Test extract_scale_min_from_slice with idx=5 (odd)
#[test]
fn test_extract_scale_min_from_slice_idx_5() {
    // For odd idx=5:
    // scale_idx = 2, min_idx = 6
    // scale = (scales[2] >> 6) | ((scales[4] & 0x0F) << 2)
    // min = (scales[6] >> 6) | ((scales[8] & 0x0F) << 2)

    let scales: [u8; 12] = [
        0,
        0,
        0b01_000000, // byte 2: high bits = 1
        0,
        0b0000_1010, // byte 4: low nibble = 10
        0,
        0b11_000000, // byte 6: high bits = 3
        0,
        0b0000_1111, // byte 8: low nibble = 15
        0,
        0,
        0,
    ];

    let (s, m) = extract_scale_min_from_slice(&scales, 5);
    // scale = 1 | (10 << 2) = 1 | 40 = 41
    // min = 3 | (15 << 2) = 3 | 60 = 63
    assert_eq!(s, 41.0, "idx=5 scale");
    assert_eq!(m, 63.0, "idx=5 min");
}

/// Test extract_scale_min_from_slice with idx=7 (odd)
#[test]
fn test_extract_scale_min_from_slice_idx_7() {
    // For odd idx=7:
    // scale_idx = 3, min_idx = 7
    // scale = (scales[3] >> 6) | ((scales[5] & 0x0F) << 2)
    // min = (scales[7] >> 6) | ((scales[9] & 0x0F) << 2)

    let scales: [u8; 12] = [
        0,
        0,
        0,
        0b11_000000, // byte 3: high bits = 3
        0,
        0b0000_1111, // byte 5: low nibble = 15
        0,
        0b10_000000, // byte 7: high bits = 2
        0,
        0b0000_1100, // byte 9: low nibble = 12
        0,
        0,
    ];

    let (s, m) = extract_scale_min_from_slice(&scales, 7);
    // scale = 3 | (15 << 2) = 3 | 60 = 63
    // min = 2 | (12 << 2) = 2 | 48 = 50
    assert_eq!(s, 63.0, "idx=7 scale");
    assert_eq!(m, 50.0, "idx=7 min");
}

/// Test all odd indices 1,3,5,7 with zero bytes
#[test]
fn test_extract_scale_min_from_slice_odd_indices_zeros() {
    let scales: [u8; 12] = [0; 12];

    for idx in [1, 3, 5, 7] {
        let (s, m) = extract_scale_min_from_slice(&scales, idx);
        assert_eq!(s, 0.0, "idx={} scale with zeros", idx);
        assert_eq!(m, 0.0, "idx={} min with zeros", idx);
    }
}

// =============================================================================
// SIMD Horizontal Sum Functions (lines 133-217)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod simd_horizontal_sum_tests {
    use super::*;

    /// Test hsum_epi32_128 with known values
    #[test]
    fn test_hsum_epi32_128_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm_set_epi32;

        // [1, 2, 3, 4] should sum to 10
        unsafe {
            let v = _mm_set_epi32(4, 3, 2, 1); // Note: reverse order in _mm_set
            let result = hsum_epi32_128(v);
            assert_eq!(result, 10, "hsum_epi32_128([1,2,3,4])");
        }
    }

    /// Test hsum_epi32_128 with negative values
    #[test]
    fn test_hsum_epi32_128_negative() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm_set_epi32;

        // [-1, -2, 3, 4] should sum to 4
        unsafe {
            let v = _mm_set_epi32(4, 3, -2, -1);
            let result = hsum_epi32_128(v);
            assert_eq!(result, 4, "hsum_epi32_128([-1,-2,3,4])");
        }
    }

    /// Test hsum_epi32_128 with zeros
    #[test]
    fn test_hsum_epi32_128_zeros() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm_setzero_si128;

        unsafe {
            let v = _mm_setzero_si128();
            let result = hsum_epi32_128(v);
            assert_eq!(result, 0, "hsum_epi32_128 zeros");
        }
    }

    /// Test hsum_epi32_256 with known values
    #[test]
    fn test_hsum_epi32_256_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm256_set_epi32;

        // [1, 2, 3, 4, 5, 6, 7, 8] should sum to 36
        unsafe {
            let v = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
            let result = hsum_epi32_256(v);
            assert_eq!(result, 36, "hsum_epi32_256([1..8])");
        }
    }

    /// Test hsum_epi32_256 with negative values
    #[test]
    fn test_hsum_epi32_256_mixed() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm256_set_epi32;

        // [-4, -3, -2, -1, 1, 2, 3, 4] should sum to 0
        unsafe {
            let v = _mm256_set_epi32(4, 3, 2, 1, -1, -2, -3, -4);
            let result = hsum_epi32_256(v);
            assert_eq!(result, 0, "hsum_epi32_256 mixed");
        }
    }

    /// Test horizontal_sum_epi32_256 with known values
    #[test]
    fn test_horizontal_sum_epi32_256_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm256_set_epi32;

        // [10, 20, 30, 40, 50, 60, 70, 80] should sum to 360
        unsafe {
            let v = _mm256_set_epi32(80, 70, 60, 50, 40, 30, 20, 10);
            let result = horizontal_sum_epi32_256(v);
            assert_eq!(result, 360, "horizontal_sum_epi32_256");
        }
    }

    /// Test horizontal_sum_epi32_256 with single nonzero
    #[test]
    fn test_horizontal_sum_epi32_256_single() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm256_set_epi32;

        unsafe {
            let v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 42);
            let result = horizontal_sum_epi32_256(v);
            assert_eq!(result, 42, "horizontal_sum_epi32_256 single");
        }
    }

    /// Test horizontal_sum_epi16_256 with known values
    #[test]
    fn test_horizontal_sum_epi16_256_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm256_set_epi16;

        // 16 values: 1..16, sum = 136
        unsafe {
            let v = _mm256_set_epi16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            let result = horizontal_sum_epi16_256(v);
            assert_eq!(result, 136, "horizontal_sum_epi16_256([1..16])");
        }
    }

    /// Test horizontal_sum_epi16_256 with negative values
    #[test]
    fn test_horizontal_sum_epi16_256_negative() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm256_set_epi16;

        // [-8..8] (excluding 0) should sum to 0
        unsafe {
            let v = _mm256_set_epi16(8, 7, 6, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -6, -7, -8);
            let result = horizontal_sum_epi16_256(v);
            assert_eq!(result, 0, "horizontal_sum_epi16_256 symmetric");
        }
    }

    /// Test hsum_epi32 with known values (uses shuffle instead of hadd)
    #[test]
    fn test_hsum_epi32_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm256_set_epi32;

        // [1, 2, 3, 4, 5, 6, 7, 8] should sum to 36
        unsafe {
            let v = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
            let result = hsum_epi32(v);
            assert_eq!(result, 36, "hsum_epi32([1..8])");
        }
    }

    /// Test hsum_epi32 with large values
    #[test]
    fn test_hsum_epi32_large() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use std::arch::x86_64::_mm256_set_epi32;

        // Large values that don't overflow
        unsafe {
            let v = _mm256_set_epi32(
                100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000,
            );
            let result = hsum_epi32(v);
            assert_eq!(result, 3_600_000, "hsum_epi32 large");
        }
    }

    /// Test all horizontal sum functions agree on same input
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

            assert_eq!(r1, 36, "hsum_epi32_256");
            assert_eq!(r2, 36, "horizontal_sum_epi32_256");
            assert_eq!(r3, 36, "hsum_epi32");
            assert_eq!(r1, r2);
            assert_eq!(r2, r3);
        }
    }
}

// =============================================================================
// AVX2 RoPE Rotation Inner Loop Coverage (lines 525-535)
// =============================================================================

/// Test RoPE with head_dim=16 (half_dim=8, exactly 1 SIMD iteration)
#[test]
fn test_rope_avx2_single_iteration() {
    let head_dim = 16; // half_dim = 8, chunks = 1
    let mut x: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();

    let freqs_cos: Vec<f32> = (0..8).map(|i| (i as f32 * 0.1).cos()).collect();
    let freqs_sin: Vec<f32> = (0..8).map(|i| (i as f32 * 0.1).sin()).collect();

    let expected = rope_reference(&x, &freqs_cos, &freqs_sin, head_dim);
    apply_rope_rotation_simd(&mut x, &freqs_cos, &freqs_sin, head_dim);

    for (i, (got, exp)) in x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "head_dim=16: mismatch at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Test RoPE with head_dim=32 (half_dim=16, 2 SIMD iterations)
#[test]
fn test_rope_avx2_two_iterations() {
    let head_dim = 32; // half_dim = 16, chunks = 2
    let mut x: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32 * 0.1).collect();

    let freqs_cos: Vec<f32> = (0..16).map(|i| (i as f32 * 0.05).cos()).collect();
    let freqs_sin: Vec<f32> = (0..16).map(|i| (i as f32 * 0.05).sin()).collect();

    let expected = rope_reference(&x, &freqs_cos, &freqs_sin, head_dim);
    apply_rope_rotation_simd(&mut x, &freqs_cos, &freqs_sin, head_dim);

    for (i, (got, exp)) in x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "head_dim=32: mismatch at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Test RoPE with head_dim=64 (half_dim=32, 4 SIMD iterations)
#[test]
fn test_rope_avx2_four_iterations() {
    let head_dim = 64; // half_dim = 32, chunks = 4
    let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32 - 32.0) * 0.1).collect();

    let freqs_cos: Vec<f32> = (0..32).map(|i| (i as f32 * 0.02).cos()).collect();
    let freqs_sin: Vec<f32> = (0..32).map(|i| (i as f32 * 0.02).sin()).collect();

    let expected = rope_reference(&x, &freqs_cos, &freqs_sin, head_dim);
    apply_rope_rotation_simd(&mut x, &freqs_cos, &freqs_sin, head_dim);

    for (i, (got, exp)) in x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "head_dim=64: mismatch at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Test RoPE with head_dim=128 (half_dim=64, 8 SIMD iterations)
#[test]
fn test_rope_avx2_many_iterations() {
    let head_dim = 128; // half_dim = 64, chunks = 8
    let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.01).collect();

    let freqs_cos: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).cos()).collect();
    let freqs_sin: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();

    let expected = rope_reference(&x, &freqs_cos, &freqs_sin, head_dim);
    apply_rope_rotation_simd(&mut x, &freqs_cos, &freqs_sin, head_dim);

    for (i, (got, exp)) in x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "head_dim=128: mismatch at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Test RoPE rotation with identity (cos=1, sin=0) at SIMD scale
#[test]
fn test_rope_avx2_identity_rotation() {
    let head_dim = 32;
    let mut x: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let original = x.clone();

    let freqs_cos = vec![1.0; 16];
    let freqs_sin = vec![0.0; 16];

    apply_rope_rotation_simd(&mut x, &freqs_cos, &freqs_sin, head_dim);

    for (i, (got, orig)) in x.iter().zip(original.iter()).enumerate() {
        assert!(
            (got - orig).abs() < 1e-5,
            "Identity rotation at {}: got {}, expected {}",
            i,
            got,
            orig
        );
    }
}

/// Test RoPE rotation with 90-degree (cos=0, sin=1) at SIMD scale
#[test]
fn test_rope_avx2_90_degree_rotation() {
    let head_dim = 16;
    let mut x: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();

    let freqs_cos = vec![0.0; 8];
    let freqs_sin = vec![1.0; 8];

    let expected = rope_reference(&x, &freqs_cos, &freqs_sin, head_dim);
    apply_rope_rotation_simd(&mut x, &freqs_cos, &freqs_sin, head_dim);

    for (i, (got, exp)) in x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "90-degree at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Test RoPE preserves L2 norm (orthogonal rotation property)
#[test]
fn test_rope_avx2_preserves_norm() {
    let head_dim = 64;
    let mut x: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32 * 0.1).collect();

    let orig_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    let freqs_cos: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).cos()).collect();
    let freqs_sin: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut x, &freqs_cos, &freqs_sin, head_dim);

    let new_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    assert!(
        (orig_norm - new_norm).abs() < 1e-3,
        "RoPE should preserve norm: orig={}, new={}",
        orig_norm,
        new_norm
    );
}

// =============================================================================
// Additional Edge Cases
// =============================================================================

/// Test softmax triggers SIMD path with exactly 8 elements
#[test]
fn test_softmax_simd_exactly_8() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax of 8 elements: sum={}",
        sum
    );
}

/// Test fused_swiglu_simd triggers SIMD path with exactly 8 elements
#[test]
fn test_swiglu_simd_exactly_8() {
    let mut gate = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
    let up = vec![1.0; 8];

    let expected: Vec<f32> = gate
        .iter()
        .map(|g| {
            let sigmoid = 1.0 / (1.0 + (-g).exp());
            g * sigmoid
        })
        .collect();

    fused_swiglu_simd(&mut gate, &up);

    for (i, (got, exp)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "SwiGLU at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

// =============================================================================
// Helper Function
// =============================================================================

/// Reference implementation of RoPE rotation
fn rope_reference(x: &[f32], freqs_cos: &[f32], freqs_sin: &[f32], head_dim: usize) -> Vec<f32> {
    let mut result = x.to_vec();
    let half_dim = head_dim / 2;

    for i in 0..half_dim.min(freqs_cos.len()) {
        if i + half_dim >= x.len() {
            break;
        }
        let x0 = x[i];
        let x1 = x[i + half_dim];
        let cos = freqs_cos[i];
        let sin = freqs_sin[i];

        result[i] = x0 * cos - x1 * sin;
        result[i + half_dim] = x0 * sin + x1 * cos;
    }
    result
}
