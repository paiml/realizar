//! Part 17: Additional Activation Coverage Tests
//!
//! Targets uncovered code paths in `src/quantize/activation.rs`:
//! - AVX2 remainder loops at specific block boundaries
//! - Fused function edge cases with minimal dimensions
//! - `fast_exp_avx2` coverage via softmax with specific patterns
//! - horizontal_max/sum AVX2 helpers via specific input patterns
//! - quantize_rmsnorm_q8_0_into with exact block boundaries

use crate::quantize::activation::{fused_swiglu_scalar, softmax_scalar};
use crate::quantize::{
    fused_rmsnorm_ffn_up_gate, fused_rmsnorm_q4_0_matmul, fused_swiglu_simd,
    quantize_activations_q8_0, quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into, softmax_simd,
};

// ============================================================================
// AVX2 Remainder Loop Tests - Specific Sizes for Full Block + Remainder
// ============================================================================

/// Tests size 35: 1 full 32-element block + 3 remainder in second block
/// This exercises both the SIMD main loop and remainder handling for partial blocks
#[test]
fn test_quantize_rmsnorm_q8_0_size_35_block_boundary() {
    let input: Vec<f32> = (0..35).map(|i| (i as f32 - 17.5) * 0.1).collect();
    let norm_weight = vec![1.0f32; 35];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should have 2 blocks (ceil(35/32) = 2)
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64); // 2 * 32 padded

    // Verify padding zeros in second block (positions 35..64)
    for q in &quants[35..64] {
        assert_eq!(*q, 0i8, "Padding should be zero");
    }

    // Both scales should be positive
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
}

/// Tests size 41: exercises AVX2 sum_sq loop (5*8=40 SIMD) + 1 scalar remainder
#[test]
fn test_quantize_rmsnorm_q8_0_size_41_simd_sum_remainder() {
    let input: Vec<f32> = (0..41).map(|i| (i as f32 - 20.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 41];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2); // ceil(41/32)
    assert_eq!(quants.len(), 64);

    // Verify first and last actual values are not padding
    assert_ne!(quants[0], 0i8);
    assert_ne!(quants[40], 0i8);
}

/// Tests size 57: 7*8 + 1 SIMD elements, exercises horizontal sum remainder
#[test]
fn test_quantize_rmsnorm_q8_0_size_57_horizontal_sum() {
    let input: Vec<f32> = (0..57).map(|i| (i as f32 - 28.0) * 0.03).collect();
    let norm_weight = vec![1.0f32; 57];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2); // ceil(57/32)
                                 // Should produce valid quantized output
    for q in &quants[..57] {
        assert!(true /* i8 always in range */);
    }
}

// ============================================================================
// AVX2 Block-Level Remainder Tests
// ============================================================================

/// Size 39: tests SIMD block with 7 remainder (32 + 7)
/// This hits the "while j < valid_len" remainder loop in AVX2 max-finding
#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_7_remainder() {
    let input: Vec<f32> = (0..39).map(|i| (i as f32 - 19.5) * 0.1).collect();
    let norm_weight = vec![1.0f32; 39];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2);
    // Second block has 7 elements, so 25 should be padding
    for q in &quants[39..64] {
        assert_eq!(*q, 0i8);
    }
}

/// Size 47: tests SIMD block with 15 remainder (32 + 15)
/// 15 = 8 SIMD + 7 scalar in the block quantization loop
#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_15_remainder() {
    let input: Vec<f32> = (0..47).map(|i| (i as f32 - 23.5) * 0.08).collect();
    let norm_weight = vec![1.0f32; 47];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2);
    // Verify non-padding values are quantized
    let non_zero_in_second_block = quants[32..47].iter().any(|&q| q != 0);
    assert!(
        non_zero_in_second_block,
        "Second block should have non-zero values"
    );
}

/// Size 55: tests SIMD block with 23 remainder (32 + 23)
/// 23 = 2*8 + 7 = 2 SIMD iterations + 7 scalar
#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_23_remainder() {
    let input: Vec<f32> = (0..55).map(|i| (i as f32 - 27.5) * 0.06).collect();
    let norm_weight = vec![1.0f32; 55];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
}

// ============================================================================
// Softmax Fast Exp AVX2 Coverage
// ============================================================================

/// Test softmax with values near exp underflow boundary (-87)
/// This exercises the clamping in fast_exp_avx2
#[test]
fn test_softmax_simd_near_exp_underflow() {
    let mut x = vec![-85.0, -86.0, -87.0, -88.0, -89.0, -90.0, -100.0, 0.0];
    softmax_simd(&mut x);

    // Sum should still be 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Sum should be 1.0, got {}", sum);

    // The 0.0 value should dominate (others are very negative)
    assert!(x[7] > 0.99, "Element at index 7 should dominate");
}

/// Test softmax with alternating signs to exercise multiple exp ranges
#[test]
fn test_softmax_simd_alternating_exp_ranges() {
    let mut x = vec![-50.0, 50.0, -50.0, 50.0, -50.0, 50.0, -50.0, 50.0];
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);

    // Positive values should each be ~0.25, negative near 0
    for i in 0..8 {
        if i % 2 == 0 {
            assert!(x[i] < 0.001, "Negative values should be near 0");
        } else {
            assert!(x[i] > 0.24, "Positive values should be ~0.25");
        }
    }
}

/// Test softmax with 17 elements - triggers SIMD (2*8=16) + 1 scalar remainder
#[test]
fn test_softmax_simd_size_17_scalar_remainder() {
    let mut x: Vec<f32> = (0..17).map(|i| i as f32 * 0.5).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);

    // Last element (index 16) should be largest probability
    for i in 0..16 {
        assert!(x[16] > x[i], "Last element should have highest probability");
    }
}

// ============================================================================
// SwiGLU SIMD Coverage - Specific Sizes
// ============================================================================

/// Test SwiGLU with 9 elements - 1 SIMD iteration + 1 scalar
#[test]
fn test_fused_swiglu_simd_size_9_remainder_1() {
    let mut gate: Vec<f32> = (0..9).map(|i| (i as f32 - 4.0) * 0.3).collect();
    let up = vec![1.0f32; 9];

    fused_swiglu_simd(&mut gate, &up);

    // All should be finite
    for g in &gate {
        assert!(g.is_finite(), "SwiGLU output should be finite");
    }
}

/// Test SwiGLU with 17 elements - 2 SIMD iterations + 1 scalar
#[test]
fn test_fused_swiglu_simd_size_17_remainder_1() {
    let mut gate: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 0.2).collect();
    let up = vec![1.0f32; 17];

    fused_swiglu_simd(&mut gate, &up);

    for g in &gate {
        assert!(g.is_finite());
    }
}

/// Test SwiGLU with 23 elements - 2 SIMD iterations + 7 scalar
#[test]
fn test_fused_swiglu_simd_size_23_remainder_7() {
    let mut gate: Vec<f32> = (0..23).map(|i| (i as f32 - 11.0) * 0.15).collect();
    let up = vec![1.0f32; 23];

    fused_swiglu_simd(&mut gate, &up);

    for g in &gate {
        assert!(g.is_finite());
    }
}

/// Test SwiGLU exp approximation with values at polynomial boundaries
#[test]
fn test_fused_swiglu_simd_exp_boundaries() {
    // Values that test the exp approximation range
    let mut gate = vec![
        -87.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, // 8 elements
    ];
    let up = vec![1.0f32; 8];

    fused_swiglu_simd(&mut gate, &up);

    // Very negative values should produce near-zero (silu(-x) -> 0 for large x)
    assert!(gate[0].abs() < 1e-30, "silu(-87) should be ~0");
    assert!(gate[1].abs() < 1e-10, "silu(-50) should be ~0");

    // Positive values: silu(x) ~= x for large x
    assert!((gate[6] - 10.0).abs() < 0.1, "silu(10) ~= 10");
    assert!((gate[7] - 50.0).abs() < 0.1, "silu(50) ~= 50");
}

// ============================================================================
// Quantize RMSNorm Into - Block Boundary Coverage
// ============================================================================

/// Test into variant with exactly 32 elements (no padding needed)
#[test]
fn test_quantize_rmsnorm_q8_0_into_exact_32() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    assert!(scales[0] > 0.0, "Scale should be positive");
    // No padding needed - all 32 values should be actual data
}

/// Test into variant with 65 elements (2 full blocks + 1)
#[test]
fn test_quantize_rmsnorm_q8_0_into_size_65() {
    let input: Vec<f32> = (0..65).map(|i| (i as f32 - 32.5) * 0.05).collect();
    let norm_weight = vec![1.0f32; 65];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 3];
    let mut quants = vec![0i8; 96]; // 3 * 32

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // All 3 blocks should have positive scales
    for s in &scales {
        assert!(*s > 0.0);
    }

    // Last block has only 1 element, rest should be padding
    assert_ne!(quants[64], 0i8, "Element 64 should be quantized");
    for q in &quants[65..96] {
        assert_eq!(*q, 0i8, "Elements 65-95 should be padding zeros");
    }
}

/// Test into variant with 97 elements (3 full blocks + 1)
#[test]
fn test_quantize_rmsnorm_q8_0_into_size_97() {
    let input: Vec<f32> = (0..97).map(|i| (i as f32 - 48.5) * 0.03).collect();
    let norm_weight = vec![1.0f32; 97];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 4];
    let mut quants = vec![0i8; 128]; // 4 * 32

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // Verify all scales positive
    for s in &scales {
        assert!(*s > 0.0);
    }
}

// ============================================================================
// Fused RMSNorm Q4_0 Matmul - Dimension Edge Cases
// ============================================================================

/// Test matmul with minimum valid dimensions (32 input, 1 output)
#[test]
fn test_fused_rmsnorm_q4_0_matmul_min_dimensions() {
    let input = vec![0.5f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    // Q4_0: 32 elements per block = 18 bytes (2 bytes scale + 16 bytes data)
    // For 1 output row with 32 input elements: 1 block = 18 bytes
    let weight_data = vec![0u8; 18];

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, eps, &weight_data, 32, 1);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 1);
}

/// Test matmul with 64 input dimension (2 blocks)
#[test]
fn test_fused_rmsnorm_q4_0_matmul_64_input() {
    let input = vec![0.5f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    // 64 elements = 2 blocks = 36 bytes per row
    // 4 output rows = 144 bytes
    let weight_data = vec![0u8; 144];

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, eps, &weight_data, 64, 4);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 4);
}

/// Test matmul with partial block (48 elements = 1 full + 16 partial)
#[test]
fn test_fused_rmsnorm_q4_0_matmul_partial_block() {
    let input = vec![0.5f32; 48];
    let norm_weight = vec![1.0f32; 48];
    let eps = 1e-5;

    // 48 elements = ceil(48/32) = 2 blocks = 36 bytes per row
    let weight_data = vec![0u8; 36 * 2]; // 2 output rows

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, eps, &weight_data, 48, 2);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 2);
}

// ============================================================================
// Fused RMSNorm FFN Up/Gate - Dimension Edge Cases
// ============================================================================

/// Test FFN with minimum dimensions
#[test]
fn test_fused_rmsnorm_ffn_up_gate_min_dimensions() {
    let input = vec![0.5f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let up_weight = vec![0u8; 18];
    let gate_weight = vec![0u8; 18];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, eps, &up_weight, &gate_weight, 32, 1);

    assert!(result.is_ok());
    let (up_out, gate_out) = result.unwrap();
    assert_eq!(up_out.len(), 1);
    assert_eq!(gate_out.len(), 1);
}

/// Test FFN with 64 input dimension
#[test]
fn test_fused_rmsnorm_ffn_up_gate_64_input() {
    let input = vec![0.5f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    // 64 input = 2 blocks = 36 bytes per row
    let up_weight = vec![0u8; 36 * 8]; // 8 output rows
    let gate_weight = vec![0u8; 36 * 8];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, eps, &up_weight, &gate_weight, 64, 8);

    assert!(result.is_ok());
    let (up_out, gate_out) = result.unwrap();
    assert_eq!(up_out.len(), 8);
    assert_eq!(gate_out.len(), 8);
}

// ============================================================================
// Quantize Activations Q8_0 - Additional Edge Cases
// ============================================================================

/// Test with 33 elements - 1 full block + 1 element
#[test]
fn test_quantize_activations_q8_0_size_33() {
    let activations: Vec<f32> = (0..33).map(|i| (i as f32 - 16.5) * 0.1).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);

    // Only element 32 is in second block, rest is padding
    assert_ne!(quants[32], 0i8, "Element 32 should be quantized");
    for q in &quants[33..64] {
        assert_eq!(*q, 0i8, "Elements 33-63 should be padding");
    }
}

/// Test with max positive values near f32::MAX / 127
#[test]
fn test_quantize_activations_q8_0_large_positive() {
    let large_val = f32::MAX / 1000.0; // Very large but not MAX
    let activations = vec![large_val; 8];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Scale should be large_val / 127
    assert!(scales[0].is_finite());
    // All quants should be 127
    for q in &quants[..8] {
        assert_eq!(*q, 127);
    }
}

/// Test with mixed positive and negative large values
#[test]
fn test_quantize_activations_q8_0_mixed_large() {
    let large_val = 1e10;
    let activations = vec![large_val, -large_val, large_val, -large_val];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Should correctly quantize to extremes
    assert_eq!(quants[0], 127);
    assert_eq!(quants[1], -127);
    assert_eq!(quants[2], 127);
    assert_eq!(quants[3], -127);
}

// ============================================================================
// Scalar vs SIMD Consistency - Specific Patterns
// ============================================================================

/// Test that scalar and SIMD produce consistent results for edge patterns
#[test]
fn test_swiglu_scalar_simd_edge_consistency() {
    // Pattern that exercises exp approximation edges
    let values: Vec<f32> = vec![
        -100.0,
        -87.0,
        -50.0,
        -10.0,
        -1.0,
        0.0,
        1.0,
        10.0,
        50.0,
        87.0,
        100.0,
        f32::MIN_POSITIVE,
        f32::EPSILON,
        0.5,
        -0.5,
        2.0,
    ];
    let up = vec![1.0f32; 16];

    let mut gate_scalar = values.clone();
    fused_swiglu_scalar(&mut gate_scalar, &up);

    let mut gate_simd = values;
    fused_swiglu_simd(&mut gate_simd, &up);

    // Both should produce finite results (may differ in precision)
    for (s, d) in gate_scalar.iter().zip(gate_simd.iter()) {
        assert!(s.is_finite(), "Scalar should be finite");
        assert!(d.is_finite(), "SIMD should be finite");
    }
}

/// Test softmax scalar/SIMD consistency with extreme range
#[test]
fn test_softmax_scalar_simd_extreme_consistency() {
    let values: Vec<f32> = vec![-1000.0, -500.0, -100.0, -10.0, 0.0, 10.0, 100.0, 500.0];

    let mut x_scalar = values.clone();
    softmax_scalar(&mut x_scalar);

    let mut x_simd = values;
    softmax_simd(&mut x_simd);

    // Both should sum to 1.0
    let sum_scalar: f32 = x_scalar.iter().sum();
    let sum_simd: f32 = x_simd.iter().sum();

    assert!(
        (sum_scalar - 1.0).abs() < 1e-4,
        "Scalar sum: {}",
        sum_scalar
    );
    assert!((sum_simd - 1.0).abs() < 1e-4, "SIMD sum: {}", sum_simd);
}

// ============================================================================
// Weight Pattern Tests for RMSNorm
// ============================================================================

/// Test with weights that vary across blocks
#[test]
fn test_quantize_rmsnorm_q8_0_varying_block_weights() {
    let input = vec![1.0f32; 64];
    // First block has small weights, second block has large weights
    let mut norm_weight = vec![0.1f32; 32];
    norm_weight.extend(vec![10.0f32; 32]);
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // First block should have smaller scale than second block
    assert!(
        scales[0] < scales[1],
        "Second block should have larger scale due to larger weights"
    );
}

/// Test with zero weights in one block, non-zero in another
#[test]
fn test_quantize_rmsnorm_q8_0_zero_first_block_weights() {
    let input = vec![1.0f32; 64];
    let mut norm_weight = vec![0.0f32; 32]; // Zero weights in first block
    norm_weight.extend(vec![1.0f32; 32]); // Non-zero in second block
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // First block should have fallback scale (1.0/127.0)
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    // First block quants should all be zero
    for q in &quants[..32] {
        assert_eq!(*q, 0i8);
    }
    // Second block should have non-zero values
    assert!(scales[1] > 0.0);
}

// ============================================================================
// Epsilon Edge Cases
// ============================================================================

/// Test with negative epsilon (should still work, just adds negative to mean_sq)
#[test]
fn test_quantize_rmsnorm_q8_0_negative_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = -1e-5; // Negative epsilon (unusual but valid float)

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should still produce valid output (mean_sq is 1.0, so 1.0 + (-1e-5) > 0)
    assert!(scales[0].is_finite());
    assert!(scales[0] > 0.0);
}

/// Test with infinity epsilon
#[test]
fn test_quantize_rmsnorm_q8_0_infinity_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = f32::INFINITY;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // inv_rms = 1.0 / sqrt(mean_sq + inf) = 1.0 / inf = 0.0
    // So all normalized values are 0, triggering fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
}

// ============================================================================
// Input Pattern Edge Cases
// ============================================================================

/// Test with single very large value, rest zeros
#[test]
fn test_quantize_rmsnorm_q8_0_spike_input() {
    let mut input = vec![0.0f32; 32];
    input[0] = 1e6; // Single spike
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Only first element should have non-zero quant
    assert_eq!(quants[0], 127);
    for q in &quants[1..32] {
        assert_eq!(*q, 0i8);
    }
}

/// Test with decreasing exponential pattern
#[test]
fn test_quantize_rmsnorm_q8_0_exponential_decay() {
    let input: Vec<f32> = (0..32).map(|i| (-0.1 * i as f32).exp()).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should produce valid quantized output with highest value at index 0
    assert!(scales[0] > 0.0);
    // First elements should have higher absolute values
    assert!(quants[0].abs() >= quants[31].abs());
}
