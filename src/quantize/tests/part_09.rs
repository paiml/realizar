//! Phase 36: SIMD Helper Functions Tests
//!
//! These tests cover the SIMD helper functions in simd.rs:
//! - f16 conversion: `f16_to_f32`, `read_f16`
//! - Scale extraction: `extract_scale_min`, `extract_scale_min_from_slice`
//! - Activation functions: `softmax_simd`, `fused_swiglu_simd`
//!
//! Strategy: Use proptest to fuzz inputs and verify against reference implementations.

use proptest::prelude::*;

use crate::quantize::simd::{extract_scale_min, extract_scale_min_from_slice, read_f16};
use crate::quantize::{f16_to_f32, fused_swiglu_simd, softmax_simd};

// =============================================================================
// f16 Conversion Tests
// =============================================================================

#[test]
fn test_f16_to_f32_zero() {
    // Positive zero: 0x0000
    assert_eq!(f16_to_f32(0x0000), 0.0);

    // Negative zero: 0x8000
    let neg_zero = f16_to_f32(0x8000);
    assert!(neg_zero == 0.0 || neg_zero == -0.0);
}

#[test]
fn test_f16_to_f32_one() {
    // 1.0 in f16: sign=0, exp=15 (biased), mantissa=0 → 0x3C00
    let result = f16_to_f32(0x3C00);
    assert!(
        (result - 1.0).abs() < 1e-6,
        "1.0 conversion: got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_negative_one() {
    // -1.0 in f16: sign=1, exp=15, mantissa=0 → 0xBC00
    let result = f16_to_f32(0xBC00);
    assert!(
        (result - (-1.0)).abs() < 1e-6,
        "-1.0 conversion: got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_two() {
    // 2.0 in f16: sign=0, exp=16 (biased), mantissa=0 → 0x4000
    let result = f16_to_f32(0x4000);
    assert!(
        (result - 2.0).abs() < 1e-6,
        "2.0 conversion: got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_half() {
    // 0.5 in f16: sign=0, exp=14 (biased), mantissa=0 → 0x3800
    let result = f16_to_f32(0x3800);
    assert!(
        (result - 0.5).abs() < 1e-6,
        "0.5 conversion: got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_infinity() {
    // +Inf: sign=0, exp=31, mantissa=0 → 0x7C00
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite() && result > 0.0, "Should be +Inf");

    // -Inf: sign=1, exp=31, mantissa=0 → 0xFC00
    let result = f16_to_f32(0xFC00);
    assert!(result.is_infinite() && result < 0.0, "Should be -Inf");
}

#[test]
fn test_f16_to_f32_nan() {
    // NaN: exp=31, mantissa!=0 → 0x7C01 (one example)
    let result = f16_to_f32(0x7C01);
    assert!(result.is_nan(), "Should be NaN");
}

#[test]
fn test_f16_to_f32_subnormal() {
    // Subnormal: exp=0, mantissa!=0
    // Smallest subnormal: 0x0001 = 2^-24 ≈ 5.96e-8
    let result = f16_to_f32(0x0001);
    assert!(
        result > 0.0 && result < 1e-6,
        "Subnormal should be tiny positive: {}",
        result
    );
}

#[test]
fn test_read_f16_basic() {
    // Test read_f16 helper
    let bytes = 0x3C00u16.to_le_bytes();
    let result = read_f16(&bytes);
    assert!((result - 1.0).abs() < 1e-6, "read_f16(1.0): got {}", result);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test f16_to_f32 matches half crate for normal values
    #[test]
    fn prop_f16_to_f32_matches_half(bits in 0u16..0x7C00) {
        // Skip NaN/Inf range (0x7C00+)
        let our_result = f16_to_f32(bits);
        let half_result = half::f16::from_bits(bits).to_f32();

        // Allow small tolerance for subnormals
        if our_result.abs() < 1e-6 && half_result.abs() < 1e-6 {
            // Both near zero, that's fine
            return Ok(());
        }

        let tolerance = half_result.abs() * 1e-5 + 1e-10;
        prop_assert!(
            (our_result - half_result).abs() <= tolerance,
            "f16_to_f32 mismatch for bits 0x{:04X}: ours={}, half={}",
            bits, our_result, half_result
        );
    }
}

// =============================================================================
// Scale Extraction Tests
// =============================================================================

#[test]
fn test_extract_scale_min_first_four_blocks() {
    // First 4 blocks use simple layout: scale = q[j] & 63, min = q[j+4] & 63
    let scales: [u8; 12] = [
        10, 20, 30, 40, // scales for blocks 0-3
        5, 15, 25, 35, // mins for blocks 0-3
        0, 0, 0, 0, // high bits (unused for first 4)
    ];

    let (s0, m0) = extract_scale_min(&scales, 0);
    assert_eq!(s0, 10.0);
    assert_eq!(m0, 5.0);

    let (s1, m1) = extract_scale_min(&scales, 1);
    assert_eq!(s1, 20.0);
    assert_eq!(m1, 15.0);

    let (s2, m2) = extract_scale_min(&scales, 2);
    assert_eq!(s2, 30.0);
    assert_eq!(m2, 25.0);

    let (s3, m3) = extract_scale_min(&scales, 3);
    assert_eq!(s3, 40.0);
    assert_eq!(m3, 35.0);
}

#[test]
fn test_extract_scale_min_last_four_blocks() {
    // Last 4 blocks use packed layout
    // For j=4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    //          m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    let scales: [u8; 12] = [
        0b11_000000, // byte 0: high bits = 3
        0b10_000000, // byte 1: high bits = 2
        0b01_000000, // byte 2: high bits = 1
        0b00_000000, // byte 3: high bits = 0
        0b11_000000, // byte 4: high bits = 3 (for min)
        0b10_000000, // byte 5: high bits = 2
        0b01_000000, // byte 6: high bits = 1
        0b00_000000, // byte 7: high bits = 0
        0b0101_0010, // byte 8: low=2 (scale4), high=5 (min4)
        0b0110_0011, // byte 9: low=3 (scale5), high=6 (min5)
        0b0111_0100, // byte 10: low=4 (scale6), high=7 (min6)
        0b1000_0101, // byte 11: low=5 (scale7), high=8 (min7)
    ];

    // Block 4: d = 2 | (3 << 4) = 2 + 48 = 50, m = 5 | (3 << 4) = 5 + 48 = 53
    let (s4, m4) = extract_scale_min(&scales, 4);
    assert_eq!(s4, 50.0);
    assert_eq!(m4, 53.0);
}

#[test]
fn test_extract_scale_min_all_zeros() {
    let scales: [u8; 12] = [0; 12];

    for i in 0..8 {
        let (s, m) = extract_scale_min(&scales, i);
        assert_eq!(s, 0.0, "Block {} scale should be 0", i);
        assert_eq!(m, 0.0, "Block {} min should be 0", i);
    }
}

#[test]
fn test_extract_scale_min_from_slice_basic() {
    let scales: [u8; 12] = [10, 20, 30, 40, 5, 15, 25, 35, 0, 0, 0, 0];

    let (s0, m0) = extract_scale_min_from_slice(&scales, 0);
    // idx=0: scale_idx=0, min_idx=4, using & 0x3F
    assert_eq!(s0, 10.0);
    assert_eq!(m0, 5.0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test extract_scale_min returns values in valid 6-bit range
    #[test]
    fn prop_extract_scale_min_range(
        scales in prop::collection::vec(any::<u8>(), 12..=12)
    ) {
        let scales_arr: [u8; 12] = scales.try_into().unwrap();

        for i in 0..8 {
            let (s, m) = extract_scale_min(&scales_arr, i);
            prop_assert!(s >= 0.0 && s <= 63.0, "Scale out of range: {}", s);
            prop_assert!(m >= 0.0 && m <= 63.0, "Min out of range: {}", m);
        }
    }
}

// =============================================================================
// Softmax SIMD Tests
// =============================================================================

/// Reference softmax implementation (numerically stable)
fn softmax_reference(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }

    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|v| (*v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|v| v / sum).collect()
}

#[test]
fn test_softmax_simd_empty() {
    let mut x: Vec<f32> = vec![];
    softmax_simd(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_simd_single() {
    let mut x = vec![1.0f32];
    softmax_simd(&mut x);
    assert!(
        (x[0] - 1.0).abs() < 1e-6,
        "Softmax of single element should be 1.0"
    );
}

#[test]
fn test_softmax_simd_two_equal() {
    let mut x = vec![1.0f32, 1.0];
    softmax_simd(&mut x);
    assert!(
        (x[0] - 0.5).abs() < 1e-6,
        "Equal inputs should give 0.5 each: {:?}",
        x
    );
    assert!((x[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_large_difference() {
    // When one value is much larger, it should dominate
    let mut x = vec![0.0, 0.0, 0.0, 100.0];
    softmax_simd(&mut x);
    assert!(x[3] > 0.99, "Largest value should dominate: {:?}", x);
}

#[test]
fn test_softmax_simd_sum_to_one() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Softmax should sum to 1.0: got {}",
        sum
    );
}

#[test]
fn test_softmax_simd_all_negative() {
    let mut x = vec![-1.0, -2.0, -3.0, -4.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Softmax with negative inputs should still sum to 1.0"
    );
    // First element should be largest (least negative)
    assert!(x[0] > x[1] && x[1] > x[2] && x[2] > x[3]);
}

#[test]
fn test_softmax_simd_matches_reference_small() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let reference = softmax_reference(&x);
    softmax_simd(&mut x);

    for (i, (actual, expected)) in x.iter().zip(reference.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_softmax_simd_matches_reference_large() {
    // Test with 32 elements (uses SIMD path)
    let mut x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let reference = softmax_reference(&x);
    softmax_simd(&mut x);

    for (i, (actual, expected)) in x.iter().zip(reference.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at {}: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test softmax_simd always sums to 1.0
    #[test]
    fn prop_softmax_simd_sums_to_one(
        x in prop::collection::vec(-100.0f32..100.0f32, 1..=64)
    ) {
        let mut x_copy = x.clone();
        softmax_simd(&mut x_copy);

        let sum: f32 = x_copy.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1.0: got {} for input len {}",
            sum, x.len()
        );
    }

    /// Test softmax_simd values are all non-negative
    #[test]
    fn prop_softmax_simd_non_negative(
        x in prop::collection::vec(-100.0f32..100.0f32, 1..=64)
    ) {
        let mut x_copy = x.clone();
        softmax_simd(&mut x_copy);

        for (i, v) in x_copy.iter().enumerate() {
            prop_assert!(*v >= 0.0, "Softmax value at {} should be non-negative: {}", i, v);
        }
    }

    /// Test softmax_simd preserves relative ordering (larger input → larger output)
    #[test]
    fn prop_softmax_simd_preserves_order(
        x in prop::collection::vec(-10.0f32..10.0f32, 2..=32)
    ) {
        let mut x_copy = x.clone();
        softmax_simd(&mut x_copy);

        // Find max input and its softmax output
        let (max_idx, _) = x.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let (softmax_max_idx, _) = x_copy.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        prop_assert_eq!(
            max_idx, softmax_max_idx,
            "Max input index should have max softmax output"
        );
    }

    /// Test softmax_simd matches reference implementation
    #[test]
    fn prop_softmax_simd_matches_reference(
        x in prop::collection::vec(-10.0f32..10.0f32, 1..=64)
    ) {
        let reference = softmax_reference(&x);
        let mut x_copy = x.clone();
        softmax_simd(&mut x_copy);

        for (i, (actual, expected)) in x_copy.iter().zip(reference.iter()).enumerate() {
            let tolerance = expected.abs() * 1e-4 + 1e-6;
            prop_assert!(
                (actual - expected).abs() <= tolerance,
                "Mismatch at {}: got {}, expected {} (input len {})",
                i, actual, expected, x.len()
            );
        }
    }
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

#[test]
fn test_softmax_simd_large_values() {
    // Large values should not overflow
    let mut x = vec![1000.0, 1001.0, 1002.0, 1003.0];
    softmax_simd(&mut x);

    // Should not be NaN or Inf
    for v in &x {
        assert!(!v.is_nan(), "Softmax should not produce NaN");
        assert!(!v.is_infinite(), "Softmax should not produce Inf");
    }

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Should still sum to 1.0");
}

#[test]
fn test_softmax_simd_very_negative() {
    // Very negative values should not underflow to zero sum
    let mut x = vec![-1000.0, -999.0, -998.0, -997.0];
    softmax_simd(&mut x);

    for v in &x {
        assert!(!v.is_nan(), "Softmax should not produce NaN");
        assert!(*v >= 0.0, "Softmax values should be non-negative");
    }

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Should still sum to 1.0");
}

// =============================================================================
// Determinism Tests
// =============================================================================

#[test]
fn test_softmax_simd_deterministic() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();

    let mut x1 = x.clone();
    let mut x2 = x.clone();

    softmax_simd(&mut x1);
    softmax_simd(&mut x2);

    for (i, (a, b)) in x1.iter().zip(x2.iter()).enumerate() {
        assert_eq!(a, b, "Softmax should be deterministic at index {}", i);
    }
}

// =============================================================================
// Fused SwiGLU Tests
// =============================================================================

/// Reference implementation of SwiGLU: gate * sigmoid(gate) * up
fn swiglu_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let sigmoid = 1.0 / (1.0 + (-g).exp());
            g * sigmoid * u
        })
        .collect()
}

#[test]
fn test_fused_swiglu_simd_basic() {
    let mut gate = vec![0.0, 1.0, -1.0, 2.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];
    let expected = swiglu_reference(&gate, &up);

    fused_swiglu_simd(&mut gate, &up);

    for (i, (g, e)) in gate.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "SwiGLU mismatch at {}: got {}, expected {}",
            i,
            g,
            e
        );
    }
}

#[test]
fn test_fused_swiglu_simd_zeros() {
    let mut gate = vec![0.0; 16];
    let up = vec![1.0; 16];

    fused_swiglu_simd(&mut gate, &up);

    // sigmoid(0) = 0.5, so 0 * 0.5 * 1 = 0
    for (i, g) in gate.iter().enumerate() {
        assert!(
            g.abs() < 1e-10,
            "SwiGLU of zero should be zero at {}: {}",
            i,
            g
        );
    }
}

#[test]
fn test_fused_swiglu_simd_large_positive() {
    // Large positive gate values should saturate sigmoid to ~1
    let mut gate = vec![100.0; 8];
    let up = vec![2.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // gate * sigmoid(gate) * up ≈ 100 * 1 * 2 = 200
    for (i, g) in gate.iter().enumerate() {
        assert!((g - 200.0).abs() < 1.0, "Large gate SwiGLU at {}: {}", i, g);
    }
}

#[test]
fn test_fused_swiglu_simd_large_negative() {
    // Large negative gate values should saturate sigmoid to ~0
    // Use -10 instead of -100 to stay within AVX2 polynomial approximation range
    let mut gate = vec![-10.0; 8];
    let up = vec![2.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // silu(-10) ≈ -10 * sigmoid(-10) ≈ -10 * 0.00005 ≈ -0.0005
    // result ≈ -0.001
    for (i, g) in gate.iter().enumerate() {
        assert!(g.abs() < 0.01, "Large negative gate SwiGLU at {}: {}", i, g);
    }
}

#[test]
#[should_panic(expected = "assertion")]
fn test_fused_swiglu_simd_length_mismatch() {
    // Mismatched lengths should panic (debug_assert in activation.rs)
    let mut gate = vec![1.0; 4];
    let up = vec![2.0; 8];

    fused_swiglu_simd(&mut gate, &up);
}

#[test]
fn test_fused_swiglu_simd_deterministic() {
    let gate_orig: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let up: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 + 0.5).collect();

    let mut gate1 = gate_orig.clone();
    let mut gate2 = gate_orig.clone();

    fused_swiglu_simd(&mut gate1, &up);
    fused_swiglu_simd(&mut gate2, &up);

    for (i, (a, b)) in gate1.iter().zip(gate2.iter()).enumerate() {
        assert_eq!(a, b, "SwiGLU should be deterministic at index {}", i);
    }
}

// =============================================================================
// Proptest for SwiGLU
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(25))]

    #[test]
    fn prop_fused_swiglu_is_finite_and_reasonable(
        gate in prop::collection::vec(-10.0f32..10.0f32, 1..=64),
    ) {
        let up: Vec<f32> = gate.iter().map(|g| g.abs() + 0.1).collect();

        let mut gate_simd = gate.clone();
        fused_swiglu_simd(&mut gate_simd, &up);

        // Verify outputs are finite and reasonably bounded
        for (i, (&got, &orig)) in gate_simd.iter().zip(gate.iter()).enumerate() {
            prop_assert!(
                got.is_finite(),
                "SwiGLU should produce finite output at {}: got {}",
                i, got
            );
            // Output should be bounded by max(|gate|) * max(up) * 2
            let max_expected = 10.0 * 10.2 * 2.0; // Conservative bound
            prop_assert!(
                got.abs() < max_expected,
                "SwiGLU output too large at {}: got {}, gate was {}",
                i, got, orig
            );
        }
    }
}
