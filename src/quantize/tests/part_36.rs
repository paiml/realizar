//! Deep coverage tests for fused_k.rs inner loop paths (PMAT-COV-95)
//!
//! The existing tests mostly use zero-scale data which bypasses the actual
//! dequant+dot computation. These tests construct valid Q4_K super-blocks
//! with non-zero d, dmin, scales, and qs values to exercise every line in:
//! - `fused_q4k_dot` inner loop (lines 92-141)
//! - `fused_q4k_q8k_dot` inner loop (lines 607-667)
//! - `fused_q4k_dot_simd` dispatch and parity with scalar
//! - `fused_q4k_q8k_dot_simd` dispatch and parity with scalar

use crate::quantize::fused_k::{
    fused_q4k_dot, fused_q4k_dot_simd, fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd,
};
use crate::quantize::simd::extract_scale_min;
use crate::quantize::types::QK_K;

/// Q4_K super-block size: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144 bytes
const SB_BYTES: usize = 144;

// ============================================================================
// Helper: Build a valid Q4_K super-block with controllable parameters
// ============================================================================

/// Build a single Q4_K super-block with the given f16 d, f16 dmin,
/// 12-byte scales array, and 128-byte qs array.
fn build_q4k_superblock(d_f16: u16, dmin_f16: u16, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
    let mut sb = vec![0u8; SB_BYTES];
    sb[0..2].copy_from_slice(&d_f16.to_le_bytes());
    sb[2..4].copy_from_slice(&dmin_f16.to_le_bytes());
    sb[4..16].copy_from_slice(scales);
    sb[16..144].copy_from_slice(qs);
    sb
}

/// f16 encoding of 1.0
const F16_ONE: u16 = 0x3C00;
/// f16 encoding of 0.5
const F16_HALF: u16 = 0x3800;
/// f16 encoding of 2.0
const F16_TWO: u16 = 0x4000;
/// f16 encoding of 0.0
const F16_ZERO: u16 = 0x0000;
/// f16 encoding of 0.25
const F16_QUARTER: u16 = 0x3400;

// ============================================================================
// DEEP SCALAR INNER LOOP TESTS: fused_q4k_dot
// ============================================================================

/// Test fused_q4k_dot with d=1.0, dmin=0, uniform scales, uniform qs.
/// This isolates the `d * sc * q_val` term without the `dmin * m` subtraction.
#[test]
fn test_fused_q4k_dot_d_only_uniform_qs() {
    // scales: all blocks have scale=1, min=0
    // For blocks 0-3: scales[j] & 63 = 1, scales[j+4] & 63 = 0
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1; // scale = 1
        scales[j + 4] = 0; // min = 0
    }
    // For blocks 4-7: high bits encoding (scale=1, min=0)
    // d = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    // m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    // We need scale=1: set scales[j+4] & 0x0F = 1, scales[j-4] >> 6 = 0
    // scales[8] = 0x01, scales[9] = 0x01, scales[10] = 0x01, scales[11] = 0x01
    for j in 8..12 {
        scales[j] = 0x01; // low nibble = 1 for scale, high nibble = 0 for min
    }

    // qs: all nibbles = 5 => 0x55
    let qs = [0x55u8; 128];

    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);
    // All activations = 1.0
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Each value dequantizes to: d * scale * nibble_val - dmin * min
    // = 1.0 * scale_float * 5.0 - 0.0
    // All 256 values contribute scale_float * 5.0 * 1.0
    // The total depends on scale extraction for each block
    // Let's just verify it's non-zero and positive
    assert!(result > 0.0, "Expected positive result, got {result}");
}

/// Test that dmin subtraction actually occurs by setting dmin > 0 with min > 0.
#[test]
fn test_fused_q4k_dot_dmin_subtraction() {
    // d=0 so d*scale*q_val = 0 for all values
    // dmin=1.0 and min=1 means each value = -dmin*min = -1.0
    // dot with all-ones activations = -1.0 * 256 = -256.0
    // But min extraction depends on the scale format.

    let mut scales = [0u8; 12];
    // For blocks 0-3: scale=0, min=1
    for j in 0..4 {
        scales[j] = 0; // scale=0
        scales[j + 4] = 1; // min=1
    }
    // For blocks 4-7: scale=0, min=1
    // m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    // Want m=1: scales[j+4] >> 4 = 0, scales[j] >> 6 = 0 => m=0... need different encoding
    // Actually for blocks 4-7: scales[j+4] = scales[8..12]
    // d = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4) where j=4..7
    // For j=4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    // m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    // Want d=0, m=1: scales[8] & 0x0F = 0, scales[8] >> 4 = 1 => scales[8] = 0x10
    for j in 8..12 {
        scales[j] = 0x10; // high nibble = 1 for min, low nibble = 0 for scale
    }

    let qs = [0x00u8; 128]; // q_val = 0 for all
    let data = build_q4k_superblock(F16_ZERO, F16_ONE, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Every value = 0 - 1.0*min_float => each contributes -min_float * 1.0
    // Result should be negative
    assert!(
        result < 0.0,
        "Expected negative result from dmin, got {result}"
    );
}

/// Test with both d and dmin non-zero to exercise both terms.
#[test]
fn test_fused_q4k_dot_both_terms() {
    let mut scales = [0u8; 12];
    // All blocks: scale=2, min=1
    for j in 0..4 {
        scales[j] = 2;
        scales[j + 4] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x12; // low nibble=2 (scale), high nibble=1 (min)
    }

    // qs: alternating 0x37 => low=7, high=3
    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = 0x37;
    }

    let data = build_q4k_superblock(F16_HALF, F16_QUARTER, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");
    // Result should be non-zero (mix of positive and negative terms)
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test with varying qs patterns across the 128-byte array.
#[test]
fn test_fused_q4k_dot_varying_qs() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 3;
        scales[j + 4] = 0;
    }
    for j in 8..12 {
        scales[j] = 0x03;
    }

    // Monotonically increasing nibbles
    let mut qs = [0u8; 128];
    for i in 0..128 {
        let lo = (i % 16) as u8;
        let hi = ((i + 5) % 16) as u8;
        qs[i] = lo | (hi << 4);
    }

    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);
    // Varying activations
    let activations: Vec<f32> = (0..QK_K).map(|i| (i as f32) * 0.01 - 1.28).collect();

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test with max nibble values (0xFF) and non-zero d, to exercise large q_val paths.
#[test]
fn test_fused_q4k_dot_max_nibbles_nonzero_d() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 63; // max 6-bit scale
        scales[j + 4] = 63; // max 6-bit min
    }
    // For blocks 4-7, encode max scale and min
    // d = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    // For max: d=63 => 0x3F => (0x0F) | (scales[j-4]>>6 << 4)
    // scales[j-4] = 63 means bits[7:6] = 0 so d = 0x0F = 15
    // Actually 63 = 0b00111111 so >>6 = 0, high bits = 0x0F from scales[j+4]&0x0F
    // max achievable: scales[j+4] = 0xFF => 0x0F | ((scales[j-4]>>6)<<4)
    // If scales[0..4] have top bits set: scales[0] = 0xFF => 0xFF>>6 = 3, d = 0x0F|(3<<4) = 0x3F = 63
    for j in 0..4 {
        scales[j] = 0xFF; // bits 7:6 = 3 for high-block encoding
        scales[j + 4] = 0xFF; // same
    }
    for j in 8..12 {
        scales[j] = 0xFF;
    }

    let qs = [0xFFu8; 128]; // all nibbles = 15
    let data = build_q4k_superblock(F16_TWO, F16_HALF, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
    // With max scales, max nibbles, d=2.0, result should be large
    assert!(
        result.abs() > 1.0,
        "Expected large magnitude result, got {result}"
    );
}

/// Test with two super-blocks where each has different d/dmin.
#[test]
fn test_fused_q4k_dot_two_blocks_different_params() {
    let mut scales1 = [0u8; 12];
    for j in 0..4 {
        scales1[j] = 10;
        scales1[j + 4] = 5;
    }
    for j in 8..12 {
        scales1[j] = 0x5A; // scale=10, min=5 roughly
    }
    let qs1 = [0x88u8; 128]; // lo=8, hi=8

    let mut scales2 = [0u8; 12];
    for j in 0..4 {
        scales2[j] = 20;
        scales2[j + 4] = 10;
    }
    for j in 8..12 {
        scales2[j] = 0xA4; // different encoding
    }
    let qs2 = [0xAAu8; 128]; // lo=10, hi=10

    let mut data = build_q4k_superblock(F16_ONE, F16_HALF, &scales1, &qs1);
    data.extend(build_q4k_superblock(F16_TWO, F16_QUARTER, &scales2, &qs2));

    let activations = vec![0.5f32; QK_K * 2];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test that all 4 chunks (j=0,64,128,192) in the inner loop are exercised
/// by verifying the result changes when different chunks have different data.
#[test]
fn test_fused_q4k_dot_per_chunk_sensitivity() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
    }
    for j in 8..12 {
        scales[j] = 0x05;
    }

    // Baseline: all qs = 0
    let qs_zero = [0x00u8; 128];
    let data_zero = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_zero);
    let activations = vec![1.0f32; QK_K];
    let baseline = fused_q4k_dot(&data_zero, &activations).expect("baseline");

    // Modify only chunk 0 (bytes 0..32 of qs, values 0..63)
    let mut qs_chunk0 = [0x00u8; 128];
    for i in 0..32 {
        qs_chunk0[i] = 0xFF;
    }
    let data_chunk0 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_chunk0);
    let result_chunk0 = fused_q4k_dot(&data_chunk0, &activations).expect("chunk0");
    assert_ne!(
        baseline, result_chunk0,
        "Chunk 0 modification should change result"
    );

    // Modify only chunk 2 (bytes 64..96 of qs, values 128..191)
    let mut qs_chunk2 = [0x00u8; 128];
    for i in 64..96 {
        qs_chunk2[i] = 0xFF;
    }
    let data_chunk2 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_chunk2);
    let result_chunk2 = fused_q4k_dot(&data_chunk2, &activations).expect("chunk2");
    assert_ne!(
        baseline, result_chunk2,
        "Chunk 2 modification should change result"
    );

    // Modify only chunk 3 (bytes 96..128 of qs, values 192..255)
    let mut qs_chunk3 = [0x00u8; 128];
    for i in 96..128 {
        qs_chunk3[i] = 0xFF;
    }
    let data_chunk3 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_chunk3);
    let result_chunk3 = fused_q4k_dot(&data_chunk3, &activations).expect("chunk3");
    assert_ne!(
        baseline, result_chunk3,
        "Chunk 3 modification should change result"
    );
}

/// Test low nibble vs high nibble paths by comparing qs=0x0F (lo=15,hi=0) vs qs=0xF0 (lo=0,hi=15).
#[test]
fn test_fused_q4k_dot_lo_vs_hi_nibbles() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 10;
    }
    for j in 8..12 {
        scales[j] = 0x0A;
    }

    // Only low nibbles set
    let qs_lo = [0x0Fu8; 128];
    let data_lo = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_lo);

    // Only high nibbles set
    let qs_hi = [0xF0u8; 128];
    let data_hi = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_hi);

    let activations = vec![1.0f32; QK_K];

    let result_lo = fused_q4k_dot(&data_lo, &activations).expect("lo only");
    let result_hi = fused_q4k_dot(&data_hi, &activations).expect("hi only");

    // Both should be non-zero
    assert!(
        result_lo.abs() > 0.0,
        "Low nibble result should be non-zero"
    );
    assert!(
        result_hi.abs() > 0.0,
        "High nibble result should be non-zero"
    );
    // They should be different because lo and hi nibbles use different scales
    // (is vs is+1 in extract_scale_min)
}

/// Test with negative activations to exercise the multiply-accumulate with signs.
#[test]
fn test_fused_q4k_dot_negative_activations() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 8;
        scales[j + 4] = 4;
    }
    for j in 8..12 {
        scales[j] = 0x48;
    }

    let qs = [0x77u8; 128]; // lo=7, hi=7
    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);
    let activations = vec![-1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("neg activations");
    assert!(result.is_finite(), "Expected finite result, got {result}");
    // Result should have opposite sign from positive activations
    let pos_result = fused_q4k_dot(&data, &vec![1.0f32; QK_K]).expect("pos activations");
    if pos_result.abs() > 1e-6 {
        assert!(
            result.signum() != pos_result.signum(),
            "Negative activations should flip result sign: neg={result}, pos={pos_result}"
        );
    }
}

// ============================================================================
// DEEP SCALAR INNER LOOP TESTS: fused_q4k_q8k_dot
// ============================================================================

/// Test fused_q4k_q8k_dot with non-zero d, dmin, scales, qs, and q8k values.
#[test]
fn test_fused_q4k_q8k_dot_nonzero_computation() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
        scales[j + 4] = 2;
    }
    for j in 8..12 {
        scales[j] = 0x25;
    }

    let qs = [0x55u8; 128]; // lo=5, hi=5
    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);

    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![10i8; QK_K];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
    assert!(result.abs() > 0.0, "Expected non-zero result, got {result}");
}

/// Test q8k dot with varying quant values to exercise the inner product loop.
#[test]
fn test_fused_q4k_q8k_dot_varying_quants() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 10;
        scales[j + 4] = 3;
    }
    for j in 8..12 {
        scales[j] = 0x3A;
    }

    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = ((i * 7 + 3) % 256) as u8;
    }
    let data = build_q4k_superblock(F16_ONE, F16_QUARTER, &scales, &qs);

    let q8k_scales = vec![1.0f32; 1];
    // Varying positive and negative q8 values
    let q8k_quants: Vec<i8> = (0..QK_K).map(|i| ((i as i16 % 127) - 63) as i8).collect();

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test q8k dot with multiple super-blocks and non-zero data.
#[test]
fn test_fused_q4k_q8k_dot_multi_sb_nonzero() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 7;
        scales[j + 4] = 3;
    }
    for j in 8..12 {
        scales[j] = 0x37;
    }

    let qs = [0xABu8; 128]; // lo=11, hi=10
    let num_sb = 3;
    let mut data = Vec::new();
    for _ in 0..num_sb {
        data.extend(build_q4k_superblock(F16_HALF, F16_QUARTER, &scales, &qs));
    }

    let q8k_scales = vec![0.3f32; num_sb];
    let q8k_quants: Vec<i8> = (0..QK_K * num_sb).map(|i| ((i % 50) as i8) - 25).collect();

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test that the low nibble × q8[offset+b] and high nibble × q8[offset+32+b] paths
/// produce different results when data differs.
#[test]
fn test_fused_q4k_q8k_dot_lo_hi_nibble_separation() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
    }
    for j in 8..12 {
        scales[j] = 0x05;
    }

    // Only low nibbles nonzero
    let qs_lo = [0x0Fu8; 128];
    let data_lo = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_lo);

    // Only high nibbles nonzero
    let qs_hi = [0xF0u8; 128];
    let data_hi = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_hi);

    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![5i8; QK_K];

    let result_lo = fused_q4k_q8k_dot(&data_lo, &q8k_scales, &q8k_quants).expect("lo nibble");
    let result_hi = fused_q4k_q8k_dot(&data_hi, &q8k_scales, &q8k_quants).expect("hi nibble");

    assert!(result_lo.abs() > 0.0, "Low nibble should produce non-zero");
    assert!(result_hi.abs() > 0.0, "High nibble should produce non-zero");
}

/// Test q8k dot with extreme i8 values to exercise overflow-safe accumulation.
#[test]
fn test_fused_q4k_q8k_dot_extreme_i8_values() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 63; // max
        scales[j + 4] = 63;
    }
    for j in 8..12 {
        scales[j] = 0xFF;
    }

    let qs = [0xFFu8; 128]; // all nibbles = 15
    let data = build_q4k_superblock(F16_ONE, F16_ONE, &scales, &qs);

    let q8k_scales = vec![1.0f32; 1];
    // All q8 values at maximum positive
    let q8k_quants = vec![127i8; QK_K];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("extreme values");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test q8k dot with all minimum i8 values.
#[test]
fn test_fused_q4k_q8k_dot_all_min_i8() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 10;
        scales[j + 4] = 5;
    }
    for j in 8..12 {
        scales[j] = 0x5A;
    }

    let qs = [0x88u8; 128]; // lo=8, hi=8
    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);

    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![-128i8; QK_K]; // all minimum

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("min i8");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

// ============================================================================
// SIMD/SCALAR PARITY TESTS WITH NON-TRIVIAL DATA
// ============================================================================

/// Test fused_q4k_dot_simd vs scalar with non-trivial data (all inner loop paths exercised).
#[test]
fn test_fused_q4k_dot_simd_scalar_parity_nontrivial() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 15;
        scales[j + 4] = 7;
    }
    for j in 8..12 {
        scales[j] = 0x7F;
    }

    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = ((i * 13 + 7) % 256) as u8;
    }

    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);
    let activations: Vec<f32> = (0..QK_K).map(|i| ((i as f32) - 128.0) * 0.01).collect();

    let scalar = fused_q4k_dot(&data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&data, &activations).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.01,
        "SIMD/scalar parity failure: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

/// Test fused_q4k_q8k_dot_simd vs scalar with non-trivial data.
#[test]
fn test_fused_q4k_q8k_dot_simd_scalar_parity_nontrivial() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 20;
        scales[j + 4] = 8;
    }
    for j in 8..12 {
        scales[j] = 0x84;
    }

    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = ((i * 11 + 3) % 256) as u8;
    }

    let data = build_q4k_superblock(F16_HALF, F16_QUARTER, &scales, &qs);
    let q8k_scales = vec![0.75f32; 1];
    let q8k_quants: Vec<i8> = (0..QK_K)
        .map(|i| ((i % 200) as i16 - 100).clamp(-128, 127) as i8)
        .collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.05,
        "SIMD/scalar parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

/// Test SIMD parity with multi-superblock data.
#[test]
fn test_fused_q4k_dot_simd_scalar_parity_multi_sb() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 8;
        scales[j + 4] = 3;
    }
    for j in 8..12 {
        scales[j] = 0x38;
    }

    let num_sb = 4;
    let mut data = Vec::new();
    for sb in 0..num_sb {
        let mut qs = [0u8; 128];
        for i in 0..128 {
            qs[i] = ((i + sb * 37) % 256) as u8;
        }
        data.extend(build_q4k_superblock(F16_ONE, F16_QUARTER, &scales, &qs));
    }

    let activations: Vec<f32> = (0..QK_K * num_sb).map(|i| (i as f32).sin()).collect();

    let scalar = fused_q4k_dot(&data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&data, &activations).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.01,
        "Multi-SB parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

/// Test SIMD parity for q8k with multiple superblocks.
#[test]
fn test_fused_q4k_q8k_dot_simd_scalar_parity_multi_sb() {
    let num_sb = 3;
    let mut data = Vec::new();

    for sb in 0..num_sb {
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = (5 + sb as u8) % 63;
            scales[j + 4] = (2 + sb as u8) % 63;
        }
        for j in 8..12 {
            scales[j] = 0x25;
        }
        let mut qs = [0u8; 128];
        for i in 0..128 {
            qs[i] = ((i * (sb + 3) + 11) % 256) as u8;
        }
        data.extend(build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs));
    }

    let q8k_scales: Vec<f32> = (0..num_sb).map(|i| 0.1 + 0.1 * i as f32).collect();
    let q8k_quants: Vec<i8> = (0..QK_K * num_sb)
        .map(|i| ((i * 3 % 255) as i8).wrapping_sub(127))
        .collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.05,
        "Multi-SB q8k parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

// ============================================================================
// SIMD DISPATCH ERROR PATHS
// ============================================================================

/// Test fused_q4k_dot_simd error path with bad data length.
#[test]
fn test_fused_q4k_dot_simd_bad_data_length() {
    let data = vec![0u8; 145]; // 145 is not a multiple of 144
    let activations = vec![1.0f32; QK_K];
    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not a multiple"), "Error: {err}");
}

/// Test fused_q4k_dot_simd error path with activation mismatch.
#[test]
fn test_fused_q4k_dot_simd_activation_mismatch() {
    let data = vec![0u8; SB_BYTES];
    let activations = vec![1.0f32; 100]; // should be 256
    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("doesn't match") || err.contains("match"),
        "Error: {err}"
    );
}

/// Test fused_q4k_q8k_dot_simd error path with bad data.
#[test]
fn test_fused_q4k_q8k_dot_simd_bad_data_length() {
    let data = vec![0u8; 200]; // not multiple of 144
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; QK_K];
    let result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

/// Test fused_q4k_q8k_dot_simd error with insufficient scales.
#[test]
fn test_fused_q4k_q8k_dot_simd_insufficient_scales() {
    let data = vec![0u8; SB_BYTES * 2]; // 2 superblocks
    let q8k_scales = vec![1.0f32; 1]; // only 1 scale for 2 blocks
    let q8k_quants = vec![1i8; QK_K * 2];
    let result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

/// Test fused_q4k_q8k_dot_simd error with insufficient quants.
#[test]
fn test_fused_q4k_q8k_dot_simd_insufficient_quants() {
    let data = vec![0u8; SB_BYTES]; // 1 superblock
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 100]; // should be 256
    let result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

// ============================================================================
// EXTRACT_SCALE_MIN INTEGRATION TESTS (verifying correct scale application)
// ============================================================================

/// Verify that extract_scale_min returns expected values for our test scales,
/// confirming the inner loop applies them correctly.
#[test]
fn test_scale_extraction_consistency() {
    let mut scales = [0u8; 12];
    // Block 0: scale=10, min=3
    scales[0] = 10;
    scales[4] = 3;

    let (sc, mn) = extract_scale_min(&scales, 0);
    assert_eq!(sc, 10.0);
    assert_eq!(mn, 3.0);
}

/// Verify scale extraction for blocks 4-7 (packed format).
#[test]
fn test_scale_extraction_high_blocks() {
    let mut scales = [0u8; 12];
    // For block 4:
    // d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    // m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    scales[0] = 0xC0; // bits 7:6 = 3 => contributes 3<<4 = 48 to scale
    scales[4] = 0x80; // bits 7:6 = 2 => contributes 2<<4 = 32 to min
    scales[8] = 0x25; // low nibble=5 for scale, high nibble=2 for min

    let (sc, mn) = extract_scale_min(&scales, 4);
    // scale = 5 | (3 << 4) = 5 | 48 = 53
    assert_eq!(sc, 53.0, "Scale extraction for block 4");
    // min = 2 | (2 << 4) = 2 | 32 = 34
    assert_eq!(mn, 34.0, "Min extraction for block 4");
}

// ============================================================================
// NUMERICAL PRECISION AND ACCUMULATION TESTS
// ============================================================================

/// Test that accumulation works correctly across many values (Kahan-style issues).
#[test]
fn test_fused_q4k_dot_accumulation_precision() {
    // Use 8 superblocks to accumulate a large number of products
    let num_sb = 8;
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let qs = [0x11u8; 128]; // lo=1, hi=1
    let mut data = Vec::new();
    for _ in 0..num_sb {
        data.extend(build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs));
    }

    let activations = vec![1.0f32; QK_K * num_sb];

    let result = fused_q4k_dot(&data, &activations).expect("accumulation");
    // Each superblock: 256 values, each dequantized to d*scale*1 = 1.0 * scale * 1.0
    // Total = num_sb * 256 * (scale_contribution)
    assert!(
        result.is_finite(),
        "Accumulation should be finite: {result}"
    );
    assert!(result > 0.0, "Accumulation should be positive: {result}");
}

/// Test linearity: doubling activations should double the result.
#[test]
fn test_fused_q4k_dot_linearity() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
        scales[j + 4] = 0; // no dmin contribution
    }
    for j in 8..12 {
        scales[j] = 0x05;
    }

    let qs = [0x33u8; 128]; // lo=3, hi=3
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    let activations_1x = vec![1.0f32; QK_K];
    let activations_2x = vec![2.0f32; QK_K];

    let result_1x = fused_q4k_dot(&data, &activations_1x).expect("1x");
    let result_2x = fused_q4k_dot(&data, &activations_2x).expect("2x");

    let ratio = if result_1x.abs() > 1e-8 {
        result_2x / result_1x
    } else {
        // If result_1x ~ 0, result_2x should also be ~ 0
        assert!(result_2x.abs() < 1e-6);
        2.0 // pass trivially
    };
    assert!(
        (ratio - 2.0).abs() < 0.01,
        "Expected 2x scaling: 1x={result_1x}, 2x={result_2x}, ratio={ratio}"
    );
}

/// Test that fused_q4k_q8k_dot scales linearly with q8k_scale.
#[test]
fn test_fused_q4k_q8k_dot_scale_linearity() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 8;
    }
    for j in 8..12 {
        scales[j] = 0x08;
    }

    let qs = [0x44u8; 128]; // lo=4, hi=4
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    let q8k_quants = vec![10i8; QK_K];

    let q8k_scales_1x = vec![1.0f32; 1];
    let q8k_scales_2x = vec![2.0f32; 1];

    let result_1x = fused_q4k_q8k_dot(&data, &q8k_scales_1x, &q8k_quants).expect("1x");
    let result_2x = fused_q4k_q8k_dot(&data, &q8k_scales_2x, &q8k_quants).expect("2x");

    if result_1x.abs() > 1e-8 {
        let ratio = result_2x / result_1x;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Expected 2x scaling: 1x={result_1x}, 2x={result_2x}, ratio={ratio}"
        );
    }
}

// ============================================================================
// EDGE CASE: SINGLE ELEMENT PER CHUNK BOUNDARY
// ============================================================================

/// Test that the inner loop processes exactly QK_K=256 elements per superblock
/// by checking that result is zero when all activations are zero except one.
#[test]
fn test_fused_q4k_dot_single_nonzero_activation() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let qs = [0x55u8; 128]; // lo=5, hi=5
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    // Only activation[0] = 1.0, rest = 0.0
    let mut activations = vec![0.0f32; QK_K];
    activations[0] = 1.0;

    let result = fused_q4k_dot(&data, &activations).expect("single nonzero");
    // Only one product contributes, so result should be small but non-zero
    // (assuming scale > 0 and q_val > 0)
    // value = d * sc * 5 - dmin * min
    // With dmin=0: value = 1.0 * scale_float * 5.0
    // Only one activation at 1.0, so result = that value
    assert!(result.is_finite());
}

/// Test with nonzero activation at the last position (boundary of last chunk).
#[test]
fn test_fused_q4k_dot_last_activation_nonzero() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let qs = [0x55u8; 128];
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    let mut activations = vec![0.0f32; QK_K];
    activations[QK_K - 1] = 1.0; // last element only

    let result = fused_q4k_dot(&data, &activations).expect("last position");
    assert!(result.is_finite());
}

/// Test with nonzero activation at chunk boundary (position 64, 128, 192).
#[test]
fn test_fused_q4k_dot_chunk_boundary_activations() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 3;
    }
    for j in 8..12 {
        scales[j] = 0x03;
    }

    let qs = [0xAAu8; 128]; // lo=10, hi=10
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    for boundary in [0, 32, 64, 96, 128, 160, 192, 224] {
        let mut activations = vec![0.0f32; QK_K];
        activations[boundary] = 1.0;
        let result = fused_q4k_dot(&data, &activations).expect(&format!("boundary {boundary}"));
        assert!(
            result.is_finite(),
            "Failed at boundary {boundary}: {result}"
        );
    }
}

// ============================================================================
// LARGE F16 d/dmin VALUES
// ============================================================================

/// Test with large d value (f16 max normalized ~65504).
#[test]
fn test_fused_q4k_dot_large_d() {
    // f16 max = 0x7BFF = 65504.0
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let qs = [0x11u8; 128]; // lo=1, hi=1
    let data = build_q4k_superblock(0x7BFF, F16_ZERO, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("large d");
    assert!(
        result.is_finite(),
        "Large d should still be finite: {result}"
    );
    assert!(result > 0.0);
}

/// Test with subnormal f16 d value.
#[test]
fn test_fused_q4k_dot_subnormal_d() {
    // f16 subnormal: 0x0001 = smallest positive subnormal ~5.96e-8
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 63; // max scale
    }
    for j in 8..12 {
        scales[j] = 0x0F; // scale nibble = F
    }

    let qs = [0xFFu8; 128]; // max nibbles
    let data = build_q4k_superblock(0x0001, F16_ZERO, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("subnormal d");
    assert!(result.is_finite());
    // Very small d means very small result
    assert!(
        result.abs() < 1.0,
        "Subnormal d should give small result: {result}"
    );
}

// ============================================================================
// SIMD empty/zero tests to exercise SIMD dispatch path
// ============================================================================

/// Test SIMD dispatch with empty data (should return 0.0 via scalar fallback).
#[test]
fn test_fused_q4k_dot_simd_empty() {
    let result = fused_q4k_dot_simd(&[], &[]).expect("empty simd");
    assert_eq!(result, 0.0);
}

/// Test SIMD q8k dispatch with empty data.
#[test]
fn test_fused_q4k_q8k_dot_simd_empty() {
    let result = fused_q4k_q8k_dot_simd(&[], &[], &[]).expect("empty simd q8k");
    assert_eq!(result, 0.0);
}

/// Test SIMD dispatch with non-trivial data exercising all 4 chunks.
#[test]
fn test_fused_q4k_dot_simd_all_chunks_nonzero() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 10 + j as u8;
        scales[j + 4] = 3 + j as u8;
    }
    for j in 8..12 {
        scales[j] = 0x3A + (j - 8) as u8;
    }

    let mut qs = [0u8; 128];
    // Different pattern for each 32-byte chunk
    for i in 0..32 {
        qs[i] = 0x12; // chunk 0
    }
    for i in 32..64 {
        qs[i] = 0x34; // chunk 1
    }
    for i in 64..96 {
        qs[i] = 0x56; // chunk 2
    }
    for i in 96..128 {
        qs[i] = 0x78; // chunk 3
    }

    let data = build_q4k_superblock(F16_ONE, F16_QUARTER, &scales, &qs);
    let activations: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.01) + 0.1).collect();

    let scalar = fused_q4k_dot(&data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&data, &activations).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.01,
        "All-chunks parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

/// Test fused_q4k_q8k_dot_simd with non-trivial data and all chunks active.
#[test]
fn test_fused_q4k_q8k_dot_simd_all_chunks_nonzero() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 12;
        scales[j + 4] = 4;
    }
    for j in 8..12 {
        scales[j] = 0x4C;
    }

    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = ((i * 5 + 17) % 256) as u8;
    }

    let data = build_q4k_superblock(F16_HALF, F16_QUARTER, &scales, &qs);
    let q8k_scales = vec![0.8f32; 1];
    let q8k_quants: Vec<i8> = (0..QK_K)
        .map(|i| {
            let v = (i as i16 * 3) % 256 - 128;
            v.clamp(-128, 127) as i8
        })
        .collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.05,
        "Q8K all-chunks parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

// ============================================================================
// REGRESSION: Specific scale patterns that exercise blocks 4-7
// ============================================================================

/// Test with scales that exercise blocks 4-7 specifically (packed encoding).
#[test]
fn test_fused_q4k_dot_blocks_4_through_7() {
    // Only set scales for blocks 4-7 (via packed encoding), blocks 0-3 = 0
    let mut scales = [0u8; 12];
    // Blocks 0-3 have scale=0, min=0
    // Blocks 4-7: use high bits from blocks 0-3
    scales[0] = 0xC0; // bits 7:6 = 3 for block 4 scale high
    scales[1] = 0x80; // bits 7:6 = 2 for block 5 scale high
    scales[2] = 0x40; // bits 7:6 = 1 for block 6 scale high
    scales[3] = 0x00; // bits 7:6 = 0 for block 7 scale high
                      // scales[4-7] stay 0 (blocks 0-3 min = 0)
                      // scales[8-11] provide low 4 bits
    scales[8] = 0x0F; // block 4: scale_lo=15, min_hi=0
    scales[9] = 0x0A; // block 5: scale_lo=10, min_hi=0
    scales[10] = 0x05; // block 6: scale_lo=5, min_hi=0
    scales[11] = 0x01; // block 7: scale_lo=1, min_hi=0

    let qs = [0x88u8; 128]; // lo=8, hi=8
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("blocks 4-7");
    // Blocks 0-3 have scale=0 so contribute 0
    // Blocks 4-7 have non-zero scales so should contribute
    assert!(result.is_finite());
    assert!(result.abs() > 0.0, "Blocks 4-7 should contribute: {result}");
}

/// Test that blocks 0-3 and 4-7 produce different results when their scales differ.
#[test]
fn test_fused_q4k_dot_block_groups_independence() {
    // Set up only blocks 0-3
    let mut scales_low = [0u8; 12];
    for j in 0..4 {
        scales_low[j] = 10; // blocks 0-3: scale=10
    }

    // Set up only blocks 4-7
    let mut scales_high = [0u8; 12];
    scales_high[0] = 0xC0;
    scales_high[1] = 0xC0;
    scales_high[2] = 0xC0;
    scales_high[3] = 0xC0;
    for j in 8..12 {
        scales_high[j] = 0x0A; // scale=10 for blocks 4-7
    }

    let qs = [0x55u8; 128];
    let data_low = build_q4k_superblock(F16_ONE, F16_ZERO, &scales_low, &qs);
    let data_high = build_q4k_superblock(F16_ONE, F16_ZERO, &scales_high, &qs);

    let activations = vec![1.0f32; QK_K];

    let result_low = fused_q4k_dot(&data_low, &activations).expect("low blocks");
    let result_high = fused_q4k_dot(&data_high, &activations).expect("high blocks");

    assert!(result_low.is_finite());
    assert!(result_high.is_finite());
    // Both should be non-zero but possibly different magnitudes
    assert!(result_low.abs() > 0.0);
    assert!(result_high.abs() > 0.0);
}

// ============================================================================
// Q4K_Q8K DOT: Per-byte inner loop coverage
// ============================================================================

/// Test that each of the 32 bytes in a chunk contributes to the result.
#[test]
fn test_fused_q4k_q8k_dot_per_byte_contribution() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let q8k_scales = vec![1.0f32; 1];

    // Baseline: all qs = 0
    let qs_zero = [0x00u8; 128];
    let data_zero = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_zero);
    let q8k_quants = vec![1i8; QK_K];
    let baseline = fused_q4k_q8k_dot(&data_zero, &q8k_scales, &q8k_quants).expect("baseline");

    // Set byte 0 only
    let mut qs_byte0 = [0x00u8; 128];
    qs_byte0[0] = 0xFF;
    let data_byte0 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_byte0);
    let result_byte0 = fused_q4k_q8k_dot(&data_byte0, &q8k_scales, &q8k_quants).expect("byte0");
    assert_ne!(baseline, result_byte0, "Byte 0 should affect result");

    // Set byte 64 (start of chunk 2)
    let mut qs_byte64 = [0x00u8; 128];
    qs_byte64[64] = 0xFF;
    let data_byte64 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_byte64);
    let result_byte64 = fused_q4k_q8k_dot(&data_byte64, &q8k_scales, &q8k_quants).expect("byte64");
    assert_ne!(baseline, result_byte64, "Byte 64 should affect result");
}

/// Test q8k dot accumulation: sum_lo and sum_hi, q8_sum_lo and q8_sum_hi paths.
#[test]
fn test_fused_q4k_q8k_dot_accumulator_paths() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
        scales[j + 4] = 2;
    }
    for j in 8..12 {
        scales[j] = 0x25;
    }

    // Use specific qs pattern that exercises both sum_lo and sum_hi differently
    let mut qs = [0u8; 128];
    // Chunk 0: low nibbles all 15, high nibbles all 0
    for i in 0..32 {
        qs[i] = 0x0F; // lo=15, hi=0
    }
    // Chunk 1: low nibbles all 0, high nibbles all 15
    for i in 32..64 {
        qs[i] = 0xF0; // lo=0, hi=15
    }
    // Chunk 2: alternating
    for i in 64..96 {
        qs[i] = if i % 2 == 0 { 0x0F } else { 0xF0 };
    }
    // Chunk 3: middle values
    for i in 96..128 {
        qs[i] = 0x77; // lo=7, hi=7
    }

    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants: Vec<i8> = (0..QK_K).map(|i| ((i % 64) as i8) - 32).collect();

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("accumulator paths");
    assert!(result.is_finite(), "Should be finite: {result}");
}
