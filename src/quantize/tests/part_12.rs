//! Part 12: Deep Coverage Tests for activation.rs
//!
//! Targets uncovered lines in `src/quantize/activation.rs` including:
//! - Scalar fallback paths for all activation functions
//! - Edge cases: zeros, near-zeros, extreme values, boundary sizes
//! - Partial block handling in quantization
//! - `quantize_rmsnorm_q8_0_into` edge cases
//! - AVX2 remainder loop paths

use crate::quantize::activation::{
    fused_swiglu_scalar, quantize_rmsnorm_q8_0_scalar, softmax_scalar,
};
use crate::quantize::{
    fused_rmsnorm_ffn_up_gate, fused_rmsnorm_q4_0_matmul, fused_swiglu_simd,
    quantize_activations_q8_0, quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into, softmax_simd,
};

// ============================================================================
// quantize_rmsnorm_q8_0_scalar: Direct Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_basic() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // One block of 32
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // Scale should be positive
    assert!(scales[0] > 0.0);
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_partial_block() {
    // 40 elements = 1 full block + 8 partial
    let input: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 40];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // Should have 2 blocks (ceil(40/32) = 2)
    assert_eq!(scales.len(), 2);
    // Quants padded to 64 (2 * 32)
    assert_eq!(quants.len(), 64);
    // Last 24 values should be padding zeros
    for q in &quants[40..64] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_near_zero_max() {
    // All values extremely small -> triggers near-zero handling path
    let input = vec![1e-12f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // Scale should be positive and finite (either fallback or computed)
    assert!(scales[0] > 0.0, "Scale should be positive");
    assert!(scales[0].is_finite(), "Scale should be finite");
    // All quantized values are i8, so range is guaranteed
    assert!(!quants.is_empty(), "Should have quantized values");
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_zeros() {
    let input = vec![0.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // inv_rms will be very large due to eps, but zero * inv_rms = 0
    // So max_abs = 0, triggering fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    for q in quants {
        assert_eq!(q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_extreme_values() {
    let input: Vec<f32> = vec![127.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // After RMSNorm, all values are 1.0 (same input, normalized)
    // Scale = 1.0 / 127.0
    assert!(scales[0] > 0.0);
    // All quants should be equal
    let first = quants[0];
    for q in &quants[1..32] {
        assert_eq!(*q, first);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_negative_values() {
    let input: Vec<f32> = (0..32).map(|i| -((i + 1) as f32)).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    // Most quantized values should be negative
    let negative_count = quants[..32].iter().filter(|&&q| q < 0).count();
    assert!(negative_count > 16);
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_weight_variation() {
    let input = vec![1.0f32; 32];
    let norm_weight: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let eps = 1e-5;

    let (_scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // Quantized values should vary due to weights
    assert_ne!(quants[0], quants[31]);
}

// ============================================================================
// fused_swiglu_scalar: Direct Tests
// ============================================================================

#[test]
fn test_fused_swiglu_scalar_basic() {
    let mut gate = vec![0.0, 1.0, -1.0, 2.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];

    fused_swiglu_scalar(&mut gate, &up);

    // silu(0) * 1 = 0
    assert!((gate[0] - 0.0).abs() < 1e-5);
    // silu(1) * 1 = 1 / (1 + exp(-1)) ~= 0.7311
    assert!((gate[1] - 0.7310586).abs() < 1e-4);
    // silu(-1) * 1 = -1 / (1 + exp(1)) ~= -0.2689
    assert!((gate[2] - (-0.2689414)).abs() < 1e-4);
}

#[test]
fn test_fused_swiglu_scalar_zeros() {
    let mut gate = vec![0.0; 8];
    let up = vec![1.0; 8];

    fused_swiglu_scalar(&mut gate, &up);

    for g in gate {
        assert_eq!(g, 0.0);
    }
}

#[test]
fn test_fused_swiglu_scalar_up_zeros() {
    let mut gate = vec![1.0, 2.0, 3.0, 4.0];
    let up = vec![0.0; 4];

    fused_swiglu_scalar(&mut gate, &up);

    // silu(x) * 0 = 0
    for g in gate {
        assert_eq!(g, 0.0);
    }
}

#[test]
fn test_fused_swiglu_scalar_large_positive() {
    let mut gate = vec![10.0, 20.0, 50.0];
    let up = vec![1.0; 3];

    fused_swiglu_scalar(&mut gate, &up);

    // For large positive x, silu(x) ~= x (sigmoid(x) ~= 1)
    assert!((gate[0] - 10.0).abs() < 0.01);
    assert!((gate[1] - 20.0).abs() < 0.01);
    assert!((gate[2] - 50.0).abs() < 0.01);
}

#[test]
fn test_fused_swiglu_scalar_large_negative() {
    let mut gate = vec![-10.0, -20.0, -50.0];
    let up = vec![1.0; 3];

    fused_swiglu_scalar(&mut gate, &up);

    // For large negative x, silu(x) ~= 0 (sigmoid(x) ~= 0)
    assert!(gate[0].abs() < 0.001);
    assert!(gate[1].abs() < 1e-6);
    assert!(gate[2].abs() < 1e-10);
}

#[test]
fn test_fused_swiglu_scalar_empty() {
    let mut gate: Vec<f32> = vec![];
    let up: Vec<f32> = vec![];

    fused_swiglu_scalar(&mut gate, &up);

    assert!(gate.is_empty());
}

#[test]
fn test_fused_swiglu_scalar_single() {
    let mut gate = vec![0.5];
    let up = vec![2.0];

    fused_swiglu_scalar(&mut gate, &up);

    // silu(0.5) = 0.5 / (1 + exp(-0.5)) ~= 0.3112
    // silu(0.5) * 2.0 ~= 0.6225
    assert!((gate[0] - 0.6224593).abs() < 1e-4);
}

// ============================================================================
// softmax_scalar: Direct Tests
// ============================================================================

#[test]
fn test_softmax_scalar_basic() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    softmax_scalar(&mut x);

    // Sum should be 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Values should be ordered
    assert!(x[3] > x[2]);
    assert!(x[2] > x[1]);
    assert!(x[1] > x[0]);
}

#[test]
fn test_softmax_scalar_uniform() {
    let mut x = vec![5.0; 8];
    softmax_scalar(&mut x);

    // All values should be equal = 1/8
    for v in &x {
        assert!((v - 0.125).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_scalar_large_values() {
    let mut x = vec![1000.0, 1001.0, 1002.0];
    softmax_scalar(&mut x);

    // Should be numerically stable
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All values should be valid (no NaN/Inf)
    for v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_softmax_scalar_very_negative() {
    let mut x = vec![-1000.0, -1001.0, -1002.0];
    softmax_scalar(&mut x);

    // Should still sum to 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_scalar_single() {
    let mut x = vec![42.0];
    softmax_scalar(&mut x);

    // Single element softmax = 1.0
    assert!((x[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_scalar_dominant() {
    let mut x = vec![0.0, 0.0, 100.0, 0.0];
    softmax_scalar(&mut x);

    // The large value should dominate
    assert!(x[2] > 0.99);
    assert!(x[0] < 0.01);
    assert!(x[1] < 0.01);
    assert!(x[3] < 0.01);
}

// ============================================================================
// quantize_rmsnorm_q8_0: SIMD/Scalar dispatch tests (sizes that hit remainders)
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_size_7_remainder() {
    // Size 7 hits remainder loop (not divisible by 8)
    let input: Vec<f32> = (0..7).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 7];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // 1 block (ceil(7/32) = 1)
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32); // Padded to 32
                                  // Padding should be zeros
    for q in &quants[7..32] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_33_remainder() {
    // Size 33 = 1 full block + 1 element
    let input: Vec<f32> = (0..33).map(|i| i as f32 * 0.05).collect();
    let norm_weight = vec![1.0f32; 33];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_63() {
    // 63 = tests the AVX2 loop (7 * 8 = 56 processed) + 7 remainder
    let input: Vec<f32> = (0..63).map(|i| (i as f32 - 31.5) * 0.1).collect();
    let norm_weight = vec![1.0f32; 63];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2); // ceil(63/32)
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_256() {
    // Large size for AVX2 main loop testing
    let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let norm_weight = vec![1.0f32; 256];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 8); // 256/32
    assert_eq!(quants.len(), 256);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_260() {
    // 260 = 8 full blocks + 4 elements (tests partial block in AVX2 path)
    let input: Vec<f32> = (0..260).map(|i| i as f32 * 0.001).collect();
    let norm_weight = vec![1.0f32; 260];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 9); // ceil(260/32)
    assert_eq!(quants.len(), 288); // 9 * 32
}

// ============================================================================
// quantize_rmsnorm_q8_0_into: Edge cases
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_into_basic() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    assert!(scales[0] > 0.0);
    // Values should be non-zero (except possibly zeros from input)
    let nonzero = quants.iter().filter(|&&q| q != 0).count();
    assert!(nonzero > 0);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_partial_block() {
    // 50 elements = 1 full block + 18 partial
    let input: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 50];
    let eps = 1e-5;

    // Need 2 scales and 64 quants (2 blocks)
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
    // Padding should be zeros (indices 50..64)
    for q in &quants[50..64] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_near_zero() {
    let input = vec![1e-12f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // With near-zero input, scale should be small but non-zero (fallback or computed)
    assert!(scales[0] > 0.0, "Scale should be positive");
    assert!(scales[0].is_finite(), "Scale should be finite");
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_matches_allocating_version() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    // Allocating version
    let (expected_scales, expected_quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Into version
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];
    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // Results should match
    assert_eq!(scales, expected_scales);
    assert_eq!(quants, expected_quants);
}

// ============================================================================
// fused_swiglu_simd: Size-based coverage
// ============================================================================

#[test]
fn test_fused_swiglu_simd_size_7() {
    // Not divisible by 8 - hits remainder path
    let mut gate: Vec<f32> = (0..7).map(|i| i as f32 * 0.5).collect();
    let up = vec![1.0f32; 7];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0) = 0
    assert!((gate[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_fused_swiglu_simd_size_15() {
    // 8 + 7 remainder
    let mut gate: Vec<f32> = (0..15).map(|i| (i as f32 - 7.0) * 0.2).collect();
    let up = vec![1.0f32; 15];

    fused_swiglu_simd(&mut gate, &up);

    // All should be finite
    for g in &gate {
        assert!(g.is_finite());
    }
}

#[test]
fn test_fused_swiglu_simd_size_64() {
    // Perfectly divisible by 8
    let mut gate: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let up = vec![1.0f32; 64];

    fused_swiglu_simd(&mut gate, &up);

    // Should have proper sigmoid-like distribution
    assert!(gate.iter().all(|g| g.is_finite()));
}

// ============================================================================
// softmax_simd: Size-based coverage
// ============================================================================

#[test]
fn test_softmax_simd_size_7() {
    // Not divisible by 8
    let mut x: Vec<f32> = (0..7).map(|i| i as f32).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_size_15() {
    // 8 + 7 remainder
    let mut x: Vec<f32> = (0..15).map(|i| i as f32 * 0.1).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_size_64() {
    let mut x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_size_100() {
    // 12 * 8 + 4 remainder
    let mut x: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ============================================================================
// quantize_activations_q8_0: Edge cases
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_size_1() {
    let activations = vec![42.0f32];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // 1 block
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32); // Padded

    // First element should be 127 (max value maps to max quant)
    assert_eq!(quants[0], 127);
    // Padding should be zeros
    for q in &quants[1..32] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_activations_q8_0_symmetric() {
    let activations = vec![-10.0, 0.0, 10.0];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // Scale = 10.0 / 127.0
    // quants: -127, 0, 127
    assert_eq!(quants[0], -127);
    assert_eq!(quants[1], 0);
    assert_eq!(quants[2], 127);
}

#[test]
fn test_quantize_activations_q8_0_near_zero_max() {
    let activations = vec![1e-12f32; 10];
    let (scales, _quants) = quantize_activations_q8_0(&activations);

    // Fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
}

#[test]
fn test_quantize_activations_q8_0_exact_block() {
    let activations: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // Max value is 31, so quants[31] = 127
    assert_eq!(quants[31], 127);
}

#[test]
fn test_quantize_activations_q8_0_multi_block() {
    let activations: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.5).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // 4 blocks (ceil(100/32) = 4)
    assert_eq!(scales.len(), 4);
    assert_eq!(quants.len(), 128);
}

// ============================================================================
// fused_rmsnorm_q4_0_matmul: Error paths and edge cases
// ============================================================================

#[test]
fn test_fused_rmsnorm_q4_0_matmul_input_dim_mismatch() {
    let input = vec![1.0f32; 16]; // Wrong size
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 18]; // 1 block

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_weight_too_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 10]; // Too small (need 18 bytes for 1 block)

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_zero_out_dim() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 18];

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 0);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

// ============================================================================
// fused_rmsnorm_ffn_up_gate: Error paths and edge cases
// ============================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_input_dim_mismatch() {
    let input = vec![1.0f32; 16]; // Wrong size
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18];
    let gate_weight = vec![0u8; 18];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_up_weight_too_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 10]; // Too small
    let gate_weight = vec![0u8; 18];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_gate_weight_too_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18];
    let gate_weight = vec![0u8; 10]; // Too small

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_zero_out_dim() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18];
    let gate_weight = vec![0u8; 18];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 0);
    assert!(result.is_ok());
    let (up, gate) = result.unwrap();
    assert!(up.is_empty());
    assert!(gate.is_empty());
}

// ============================================================================
// Scalar vs SIMD parity tests
// ============================================================================

#[test]
fn test_swiglu_scalar_produces_output() {
    let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.2).collect();
    let up = vec![1.0f32; 32];

    // Test that scalar swiglu produces valid output
    let mut gate_scalar = values.clone();
    fused_swiglu_scalar(&mut gate_scalar, &up);

    // All values should be finite
    for v in &gate_scalar {
        assert!(v.is_finite(), "SwiGLU output should be finite");
    }

    // Output should differ from input (transformation applied)
    let different = gate_scalar
        .iter()
        .zip(values.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(different, "SwiGLU should transform input");
}

#[test]
fn test_softmax_scalar_simd_parity() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

    // Scalar
    let mut x_scalar = values.clone();
    softmax_scalar(&mut x_scalar);

    // SIMD (via dispatch)
    let mut x_simd = values.clone();
    softmax_simd(&mut x_simd);

    // Should match within tolerance
    for (s, d) in x_scalar.iter().zip(x_simd.iter()) {
        assert!((s - d).abs() < 1e-5, "Mismatch: scalar={}, simd={}", s, d);
    }
}

#[test]
fn test_quantize_rmsnorm_scalar_simd_parity() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    // Scalar
    let (scales_scalar, quants_scalar) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // SIMD (via dispatch) - may use AVX2
    let (scales_simd, quants_simd) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Scales should be very close
    for (s, d) in scales_scalar.iter().zip(scales_simd.iter()) {
        assert!(
            (s - d).abs() < 1e-5,
            "Scale mismatch: scalar={}, simd={}",
            s,
            d
        );
    }

    // Quants may differ by 1 due to rounding
    for (s, d) in quants_scalar.iter().zip(quants_simd.iter()) {
        assert!(
            (*s as i32 - *d as i32).abs() <= 1,
            "Quant mismatch: scalar={}, simd={}",
            s,
            d
        );
    }
}

// ============================================================================
// Additional edge case tests for full coverage
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_size_1() {
    let input = vec![1.0f32];
    let norm_weight = vec![1.0f32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // First element should be 127 (normalized = 1.0)
    assert_eq!(quants[0], 127);
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_size_31() {
    // Just under block boundary
    let input: Vec<f32> = (0..31).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 31];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // Padding at position 31
    assert_eq!(quants[31], 0i8);
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_size_64() {
    // Exactly 2 blocks
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
    // Both blocks should have valid scales
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
}

#[test]
fn test_fused_swiglu_scalar_symmetry() {
    // silu(-x) * 1 = -silu(x) for silu
    let mut gate_pos = vec![1.0, 2.0, 3.0];
    let mut gate_neg = vec![-1.0, -2.0, -3.0];
    let up = vec![1.0, 1.0, 1.0];

    fused_swiglu_scalar(&mut gate_pos, &up);
    fused_swiglu_scalar(&mut gate_neg, &up);

    // silu(-x) = -x * sigmoid(-x), silu(x) = x * sigmoid(x)
    // They are NOT equal in magnitude, but have opposite signs
    for i in 0..3 {
        assert!(gate_pos[i] > 0.0);
        assert!(gate_neg[i] < 0.0);
    }
}

#[test]
fn test_softmax_scalar_two_elements() {
    let mut x = vec![0.0, 0.0];
    softmax_scalar(&mut x);

    // Equal inputs should give equal outputs
    assert!((x[0] - 0.5).abs() < 1e-5);
    assert!((x[1] - 0.5).abs() < 1e-5);
}

#[test]
fn test_softmax_scalar_diff_10() {
    // Large difference should make smaller almost 0
    let mut x = vec![0.0, 10.0];
    softmax_scalar(&mut x);

    assert!(x[0] < 0.001);
    assert!(x[1] > 0.999);
}

#[test]
fn test_quantize_activations_q8_0_all_negative() {
    let activations = vec![-5.0f32; 16];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // All quants should be -127 (max negative)
    for q in &quants[..16] {
        assert_eq!(*q, -127);
    }
    // Padding should be 0
    for q in &quants[16..32] {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_quantize_activations_q8_0_alternating() {
    let activations = vec![1.0, -1.0, 1.0, -1.0];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // Scale = 1.0 / 127.0
    // quants should alternate 127, -127
    assert_eq!(quants[0], 127);
    assert_eq!(quants[1], -127);
    assert_eq!(quants[2], 127);
    assert_eq!(quants[3], -127);
}

// ============================================================================
// Test clamping behavior
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_clamping() {
    // Create input that would exceed i8 range after quantization
    let input = vec![1000.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // All quants should be 127 (clamped max)
    for q in quants {
        assert_eq!(q, 127);
    }
    assert!(scales[0] > 0.0);
}

#[test]
fn test_quantize_activations_q8_0_clamping_negative() {
    // Very large negative values
    let activations = vec![-1000.0f32; 8];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // All should clamp to -127 (not -128)
    for q in &quants[..8] {
        assert_eq!(*q, -127);
    }
}

// ============================================================================
// Test various epsilon values
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_tiny_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-15; // Very small epsilon

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert!(scales[0] > 0.0);
    // All quants should be equal (uniform input)
    let first = quants[0];
    for q in &quants[1..32] {
        assert_eq!(*q, first);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_large_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1.0; // Large epsilon

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    // Large epsilon affects the inv_rms calculation
    assert!(scales[0] > 0.0);
}

// ============================================================================
// AVX2 specific path tests (block sizes that exercise remainder loops)
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_size_9() {
    // 9 elements in block = 1 SIMD iteration + 1 remainder
    let input: Vec<f32> = (0..9).map(|i| (i as f32) * 0.1).collect();
    let norm_weight = vec![1.0f32; 9];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_size_17() {
    // 17 elements in block = 2 SIMD iterations + 1 remainder
    let input: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 17];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_size_25() {
    // 25 elements in block = 3 SIMD iterations + 1 remainder
    let input: Vec<f32> = (0..25).map(|i| (i as f32 - 12.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 25];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

// ============================================================================
// Test with specific patterns that might cause numerical issues
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_alternating_signs() {
    let input: Vec<f32> = (0..32)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (_scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // All normalized values have same magnitude, so should quantize to same abs value
    let first_abs = quants[0].abs();
    for q in &quants[1..32] {
        assert_eq!(q.abs(), first_abs);
    }
}

#[test]
fn test_fused_swiglu_simd_very_large_values() {
    let mut gate = vec![100.0f32; 16];
    let up = vec![1.0f32; 16];

    fused_swiglu_simd(&mut gate, &up);

    // For large positive, silu(x) ~= x
    for g in &gate {
        assert!((g - 100.0).abs() < 0.1);
    }
}

#[test]
fn test_softmax_simd_all_same_large() {
    let mut x = vec![500.0f32; 16];
    softmax_simd(&mut x);

    // All same -> uniform distribution
    for v in &x {
        assert!((v - 1.0 / 16.0).abs() < 1e-5);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_multi_block() {
    // 128 elements = 4 blocks
    let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let norm_weight = vec![1.0f32; 128];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 4];
    let mut quants = vec![0i8; 128];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // All 4 blocks should have positive scales
    for s in &scales {
        assert!(*s > 0.0);
    }
}
