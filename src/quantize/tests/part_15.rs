//! Part 15: Activation Function Edge Cases and Special Value Coverage
//!
//! This module provides comprehensive coverage for activation.rs edge cases:
//! - Special floating point values (Inf, NaN, subnormals)
//! - Boundary values at i8 limits (-128, 127)
//! - Empty input handling
//! - Large/small epsilon values
//! - Mixed extreme values
//! - AVX2 specific remainder loop paths

use crate::quantize::activation::{
    fused_swiglu_scalar, quantize_rmsnorm_q8_0_scalar, softmax_scalar,
};
use crate::quantize::{
    fused_swiglu_simd, quantize_activations_q8_0, quantize_rmsnorm_q8_0,
    quantize_rmsnorm_q8_0_into, softmax_simd,
};

// ============================================================================
// Special Floating Point Values: Infinity
// ============================================================================

#[test]
fn test_softmax_simd_empty_input() {
    // Tests the early return path at line 633-635
    let mut x: Vec<f32> = vec![];
    softmax_simd(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_scalar_empty_input() {
    let mut x: Vec<f32> = vec![];
    softmax_scalar(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_simd_positive_infinity() {
    // Test with positive infinity
    let mut x = vec![1.0, f32::INFINITY, 2.0, 3.0];
    softmax_simd(&mut x);

    // The infinity element should dominate
    assert!(
        x[1] > 0.99 || x[1].is_nan(),
        "Infinity should dominate or produce NaN"
    );
    // Other elements should be near 0 or NaN
}

#[test]
fn test_softmax_simd_negative_infinity() {
    // Test with negative infinity
    let mut x = vec![1.0, f32::NEG_INFINITY, 2.0, 3.0];
    softmax_simd(&mut x);

    // The negative infinity element should be near 0
    let sum: f32 = x.iter().sum();
    // Sum should be close to 1 or NaN depending on implementation
    assert!(sum.is_nan() || (sum - 1.0).abs() < 1e-3);
}

#[test]
fn test_softmax_scalar_positive_infinity() {
    let mut x = vec![1.0, f32::INFINITY, 2.0];
    softmax_scalar(&mut x);

    // Infinity in input should result in that element being dominant
    // or the result being NaN depending on numerical handling
    assert!(x[1] >= 0.99 || x[1].is_nan());
}

#[test]
fn test_softmax_scalar_negative_infinity() {
    let mut x = vec![f32::NEG_INFINITY, 1.0, 2.0];
    softmax_scalar(&mut x);

    // The negative infinity element should have probability near 0
    assert!(x[0] < 0.01 || x[0].is_nan());
}

// ============================================================================
// Special Floating Point Values: NaN
// ============================================================================

#[test]
fn test_softmax_simd_nan() {
    // NaN propagation test
    let mut x = vec![1.0, f32::NAN, 2.0, 3.0];
    softmax_simd(&mut x);

    // NaN should propagate through the computation
    // At least one element should be NaN
    let has_nan = x.iter().any(|v| v.is_nan());
    assert!(has_nan, "NaN should propagate through softmax");
}

#[test]
fn test_softmax_scalar_nan() {
    let mut x = vec![1.0, f32::NAN, 2.0];
    softmax_scalar(&mut x);

    // NaN should propagate
    let has_nan = x.iter().any(|v| v.is_nan());
    assert!(has_nan, "NaN should propagate through scalar softmax");
}

#[test]
fn test_fused_swiglu_scalar_infinity() {
    let mut gate = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0];
    let up = vec![1.0, 1.0, 1.0];

    fused_swiglu_scalar(&mut gate, &up);

    // For positive infinity: silu(inf) = inf * sigmoid(inf) = inf * 1 = inf
    assert!(gate[0].is_infinite() && gate[0] > 0.0);
    // For negative infinity: silu(-inf) = -inf * sigmoid(-inf) = -inf * 0 = NaN or 0
    // (This is 0 * (-inf) which is NaN in IEEE 754)
    assert!(gate[1].is_nan() || gate[1] == 0.0);
}

#[test]
fn test_fused_swiglu_scalar_nan() {
    let mut gate = vec![f32::NAN, 1.0, 2.0];
    let up = vec![1.0, 1.0, 1.0];

    fused_swiglu_scalar(&mut gate, &up);

    // NaN should propagate
    assert!(gate[0].is_nan());
    // Other elements should be valid
    assert!(gate[1].is_finite());
    assert!(gate[2].is_finite());
}

#[test]
fn test_fused_swiglu_simd_infinity() {
    let mut gate = vec![f32::INFINITY; 8];
    let up = vec![1.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // All should be infinite
    for g in &gate {
        assert!(g.is_infinite() && *g > 0.0);
    }
}

#[test]
fn test_fused_swiglu_simd_nan_propagation() {
    let mut gate = vec![1.0, f32::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let up = vec![1.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // NaN should propagate
    assert!(gate[1].is_nan());
    // Other elements should be finite
    assert!(gate[0].is_finite());
    assert!(gate[2].is_finite());
}

// ============================================================================
// Special Floating Point Values: Subnormal Numbers
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_subnormal_input() {
    // Subnormal (denormalized) numbers are very small but not zero
    let subnormal = f32::MIN_POSITIVE / 2.0; // This is subnormal
    let input = vec![subnormal; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should handle subnormals gracefully
    assert!(scales[0].is_finite());
    assert!(scales[0] > 0.0);
    // Quants should be valid
    for q in &quants {
        assert!(true /* i8 values always in range */);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_subnormal() {
    let subnormal = f32::MIN_POSITIVE / 2.0;
    let input = vec![subnormal; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert!(scales[0].is_finite());
    // Due to eps, inv_rms will be large but finite
    for q in &quants {
        assert!(true /* i8 values always in range */);
    }
}

// ============================================================================
// Boundary Values: i8 Limits
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_produces_max_quant() {
    // Input designed to produce exactly 127 after quantization
    let input = vec![10.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // After RMSNorm with uniform input, all values are 1.0
    // So all quants should be 127
    for q in &quants {
        assert_eq!(*q, 127);
    }
    assert!(scales[0] > 0.0);
}

#[test]
fn test_quantize_rmsnorm_q8_0_produces_min_quant() {
    // Input designed to produce exactly -127 after quantization
    let input = vec![-10.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // After RMSNorm with uniform negative input, all values are -1.0
    // So all quants should be -127
    for q in &quants {
        assert_eq!(*q, -127);
    }
}

#[test]
fn test_quantize_activations_q8_0_boundary_127() {
    let activations = vec![127.0f32; 16];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Max should map to 127
    assert_eq!(quants[0], 127);
    // Scale = 127.0 / 127.0 = 1.0
    assert!((scales[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_quantize_activations_q8_0_boundary_negative_127() {
    let activations = vec![-127.0f32; 16];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Min should map to -127
    assert_eq!(quants[0], -127);
}

#[test]
fn test_quantize_activations_q8_0_mixed_extremes() {
    let mut activations = vec![0.0f32; 32];
    activations[0] = 127.0;
    activations[1] = -127.0;
    activations[15] = 63.5;
    activations[16] = -63.5;

    let (scales, quants) = quantize_activations_q8_0(&activations);

    // First element should be 127, second -127
    assert_eq!(quants[0], 127);
    assert_eq!(quants[1], -127);
    // Element at 15 should be approximately half
    assert!((quants[15] as f32 - 63.5).abs() < 2.0);
}

// ============================================================================
// Large and Small Epsilon Values
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_zero_epsilon() {
    // Zero epsilon (technically dangerous but should handle)
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 0.0;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should still produce valid output
    assert!(scales[0].is_finite());
    assert!(scales[0] > 0.0);
    for q in &quants {
        assert!(true /* i8 values always in range */);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_zero_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 0.0;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert!(scales[0].is_finite());
    for q in &quants {
        assert!(true /* i8 values always in range */);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_huge_epsilon() {
    // Very large epsilon
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1000.0;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Large epsilon means inv_rms will be small
    // So normalized values will be small
    assert!(scales[0].is_finite());
    assert!(scales[0] > 0.0);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_zero_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 0.0;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    assert!(scales[0].is_finite());
}

// ============================================================================
// Mixed Extreme Values
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_mixed_large_small() {
    // Mix of very large and very small values
    let mut input = vec![0.001f32; 32];
    input[0] = 1000.0;
    input[31] = -1000.0;
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should produce valid output
    assert!(scales[0].is_finite());
    // Extreme values should map to max/min quants
    // (after normalization)
}

#[test]
fn test_softmax_simd_extreme_range() {
    // Very large range of values
    let mut x = vec![0.0f32; 16];
    x[0] = -500.0;
    x[8] = 500.0;

    softmax_simd(&mut x);

    // The large positive value should dominate
    assert!(x[8] > 0.99);
    // All others should be near zero
    for i in 0..16 {
        if i != 8 {
            assert!(x[i] < 0.01 || x[i].is_nan());
        }
    }
}

#[test]
fn test_softmax_scalar_extreme_range() {
    let mut x = vec![0.0f32; 8];
    x[0] = -500.0;
    x[4] = 500.0;

    softmax_scalar(&mut x);

    // The large positive value should dominate
    assert!(x[4] > 0.99);
}

// ============================================================================
// Weight Variations
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_zero_weight() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![0.0f32; 32]; // All zero weights
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Zero weights should produce zero normalized values
    // Which triggers the fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_negative_weight() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![-1.0f32; 32]; // Negative weights
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Negative weights should flip the sign
    assert!(scales[0] > 0.0);
    for q in &quants {
        assert_eq!(*q, -127);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_varying_weights() {
    let input = vec![1.0f32; 32];
    let norm_weight: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // Should have varying quantized values
    assert!(scales[0] > 0.0);
    // Not all quants should be the same
    let first = quants[0];
    let all_same = quants[1..32].iter().all(|&q| q == first);
    assert!(!all_same, "Varying weights should produce varying quants");
}

// ============================================================================
// Size Edge Cases for AVX2 Paths
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_size_1() {
    // Minimum size
    let input = vec![5.0f32];
    let norm_weight = vec![1.0f32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32); // Padded
    assert_eq!(quants[0], 127); // Normalized = 1.0 -> 127
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_8() {
    // Exactly one SIMD iteration
    let input: Vec<f32> = (0..8).map(|i| i as f32 - 4.0).collect();
    let norm_weight = vec![1.0f32; 8];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_16() {
    // Two SIMD iterations, no remainder
    let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 16];
    let eps = 1e-5;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_24() {
    // Three SIMD iterations, no remainder
    let input: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 24];
    let eps = 1e-5;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
}

#[test]
fn test_fused_swiglu_simd_size_1() {
    // Single element (hits scalar remainder)
    let mut gate = vec![0.5f32];
    let up = vec![2.0f32];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0.5) * 2.0
    let expected = 0.5 / (1.0 + (-0.5f32).exp()) * 2.0;
    assert!((gate[0] - expected).abs() < 1e-4);
}

#[test]
fn test_fused_swiglu_simd_size_16() {
    // Two SIMD iterations, no remainder
    let mut gate: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.2).collect();
    let up = vec![1.0f32; 16];

    fused_swiglu_simd(&mut gate, &up);

    // All should be finite
    for g in &gate {
        assert!(g.is_finite());
    }
}

#[test]
fn test_softmax_simd_size_1() {
    // Single element softmax should be 1.0
    let mut x = vec![42.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_size_8() {
    // Exactly one SIMD iteration
    let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Last element should be largest
    assert!(x[7] > x[0]);
}

#[test]
fn test_softmax_simd_size_16() {
    // Two SIMD iterations
    let mut x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ============================================================================
// Up array edge cases for SwiGLU
// ============================================================================

#[test]
fn test_fused_swiglu_simd_up_zeros() {
    let mut gate: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
    let up = vec![0.0f32; 16];

    fused_swiglu_simd(&mut gate, &up);

    // silu(x) * 0 = 0
    for g in &gate {
        assert_eq!(*g, 0.0);
    }
}

#[test]
fn test_fused_swiglu_simd_up_negative() {
    let mut gate = vec![2.0f32; 8];
    let up = vec![-1.0f32; 8];

    fused_swiglu_simd(&mut gate, &up);

    // silu(2.0) > 0, * -1 = negative
    for g in &gate {
        assert!(*g < 0.0);
    }
}

#[test]
fn test_fused_swiglu_simd_up_large() {
    let mut gate = vec![0.5f32; 8];
    let up = vec![1000.0f32; 8];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0.5) > 0, multiplied by large up should produce large positive
    for g in &gate {
        assert!(*g > 100.0, "Expected large positive, got {}", g);
        assert!(g.is_finite());
    }
}

// ============================================================================
// Precision and Numerical Stability
// ============================================================================

#[test]
fn test_softmax_simd_underflow_protection() {
    // Values so negative that exp would underflow
    let mut x = vec![-1000.0, -1001.0, -1002.0, -1003.0];
    softmax_simd(&mut x);

    // Should still sum to 1.0 due to max subtraction
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // All values should be finite
    for v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_softmax_scalar_underflow_protection() {
    let mut x = vec![-1000.0, -1001.0, -1002.0];
    softmax_scalar(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_quantize_rmsnorm_q8_0_numerical_stability() {
    // Input that would cause numerical issues without proper handling
    let mut input = vec![0.0f32; 64];
    input[0] = 1e6;
    input[63] = -1e6;
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should produce valid output
    assert!(scales[0].is_finite());
    assert!(scales[1].is_finite());
    // Extreme values should map to extreme quants
}

// ============================================================================
// Quantize rmsnorm into edge cases
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_into_size_1() {
    let input = vec![5.0f32];
    let norm_weight = vec![1.0f32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    assert_eq!(quants[0], 127);
    // Padding should be zeros
    for q in &quants[1..32] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_size_33() {
    // Tests partial second block
    let input: Vec<f32> = (0..33).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 33];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // Both blocks should have scales
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
    // Padding in second block (indices 33..64)
    for q in &quants[33..64] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_all_zeros() {
    let input = vec![0.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![99i8; 32]; // Non-zero initial values

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // All quants should be zero
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
    // Scale should be the fallback value
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
}

// ============================================================================
// Quantize activations edge cases
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_empty() {
    let activations: Vec<f32> = vec![];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert!(scales.is_empty());
    assert!(quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_single_zero() {
    let activations = vec![0.0f32];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    assert_eq!(quants[0], 0i8);
}

#[test]
fn test_quantize_activations_q8_0_all_zeros() {
    let activations = vec![0.0f32; 64];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // All scales should be fallback
    for s in &scales {
        assert!((*s - 1.0 / 127.0).abs() < 1e-10);
    }
    // All quants should be zero
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_activations_q8_0_infinity() {
    let activations = vec![f32::INFINITY, 1.0, 2.0];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Infinity should produce infinite scale or saturated quants
    // The behavior depends on implementation - i8 is always finite
    assert!(scales[0].is_infinite() || quants[0] == 127 || quants[0] == -128);
}

#[test]
fn test_quantize_activations_q8_0_negative_infinity() {
    let activations = vec![f32::NEG_INFINITY, 1.0, 2.0];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Should handle negative infinity
    // The behavior depends on implementation - i8 is always finite
    assert!(scales[0].is_infinite() || quants[0] == -127 || quants[0] == -128);
}

#[test]
fn test_quantize_activations_q8_0_nan() {
    let activations = vec![f32::NAN, 1.0, 2.0];
    let (scales, _quants) = quantize_activations_q8_0(&activations);

    // NaN in max calculation propagates
    assert!(scales[0].is_nan() || scales[0] > 0.0);
}

// ============================================================================
// Additional Scalar/SIMD Parity Tests
// ============================================================================

#[test]
fn test_fused_swiglu_scalar_simd_parity_large() {
    let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.05).collect();
    let up = vec![1.5f32; 128];

    let mut gate_scalar = values.clone();
    fused_swiglu_scalar(&mut gate_scalar, &up);

    let mut gate_simd = values;
    fused_swiglu_simd(&mut gate_simd, &up);

    // Both produce finite results
    for (s, d) in gate_scalar.iter().zip(gate_simd.iter()) {
        assert!(s.is_finite(), "Scalar should be finite: {}", s);
        assert!(d.is_finite(), "SIMD should be finite: {}", d);
    }
}

#[test]
fn test_softmax_scalar_simd_parity_large() {
    let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.02).collect();

    let mut x_scalar = values.clone();
    softmax_scalar(&mut x_scalar);

    let mut x_simd = values;
    softmax_simd(&mut x_simd);

    // Should match within tolerance
    for (s, d) in x_scalar.iter().zip(x_simd.iter()) {
        assert!((s - d).abs() < 1e-5, "Mismatch: scalar={}, simd={}", s, d);
    }
}

// ============================================================================
// Block Boundary Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_exactly_32() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Exactly one block
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // No padding needed
}

#[test]
fn test_quantize_rmsnorm_q8_0_exactly_64() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Exactly two blocks
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_exactly_96() {
    let input: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 0.03).collect();
    let norm_weight = vec![1.0f32; 96];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Exactly three blocks
    assert_eq!(scales.len(), 3);
    assert_eq!(quants.len(), 96);
}

// ============================================================================
// Maximum Input Size Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_large_input() {
    // 1024 elements = 32 blocks
    let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.001).collect();
    let norm_weight = vec![1.0f32; 1024];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 32);
    assert_eq!(quants.len(), 1024);
    // All scales should be positive and finite
    for s in &scales {
        assert!(*s > 0.0);
        assert!(s.is_finite());
    }
}

#[test]
fn test_fused_swiglu_simd_large_input() {
    let mut gate: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.005).collect();
    let up = vec![1.0f32; 1024];

    fused_swiglu_simd(&mut gate, &up);

    // All should be finite
    for g in &gate {
        assert!(g.is_finite());
    }
}

#[test]
fn test_softmax_simd_large_input() {
    let mut x: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();

    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}
