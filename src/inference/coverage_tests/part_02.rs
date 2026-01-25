//! Part 02: SIMD operations and BF16/F16 conversion tests
//!
//! Additional coverage for simd.rs including:
//! - Edge cases for SIMD operations
//! - BF16/F16 conversion edge cases
//! - Streaming matmul tests
//! - Large input handling

use crate::inference::{
    simd_add, simd_bf16_dot, simd_bf16_matmul, simd_bf16_to_f32, simd_dot, simd_f16_to_f32,
    simd_gelu, simd_matmul, simd_mul, simd_silu, simd_softmax,
};

// ============================================================================
// simd_matmul Additional Edge Cases
// ============================================================================

#[test]
fn test_simd_matmul_single_element() {
    let input = vec![3.0];
    let weight = vec![2.0];
    let output = simd_matmul(&input, &weight, 1, 1);

    assert_eq!(output.len(), 1);
    assert!((output[0] - 6.0).abs() < 1e-5);
}

#[test]
fn test_simd_matmul_row_vector_output() {
    // 1x4 input, 4x1 weight -> 1x1 output
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0]; // Single row that sums
    let output = simd_matmul(&input, &weight, 4, 1);

    assert_eq!(output.len(), 1);
    assert!((output[0] - 10.0).abs() < 1e-5);
}

#[test]
fn test_simd_matmul_zeros_input() {
    let input = vec![0.0; 4];
    let weight = vec![1.0; 8]; // 2x4 matrix
    let output = simd_matmul(&input, &weight, 4, 2);

    assert_eq!(output.len(), 2);
    for &v in &output {
        assert!((v).abs() < 1e-5);
    }
}

#[test]
fn test_simd_matmul_zeros_weight() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.0; 8]; // 2x4 matrix of zeros
    let output = simd_matmul(&input, &weight, 4, 2);

    assert_eq!(output.len(), 2);
    for &v in &output {
        assert!((v).abs() < 1e-5);
    }
}

#[test]
fn test_simd_matmul_negative_values() {
    let input = vec![-1.0, -2.0];
    let weight = vec![
        -1.0, -1.0, // row 0: sum = -(-1) + -(-2) = 3
        1.0, 1.0, // row 1: sum = -1 + -2 = -3
    ];
    let output = simd_matmul(&input, &weight, 2, 2);

    assert_eq!(output.len(), 2);
    assert!((output[0] - 3.0).abs() < 1e-5);
    assert!((output[1] - (-3.0)).abs() < 1e-5);
}

#[test]
fn test_simd_matmul_non_square() {
    // 3 input -> 5 output
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![
        1.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, // row 1
        0.0, 0.0, 1.0, // row 2
        1.0, 1.0, 0.0, // row 3
        0.0, 1.0, 1.0, // row 4
    ];
    let output = simd_matmul(&input, &weight, 3, 5);

    assert_eq!(output.len(), 5);
    assert!((output[0] - 1.0).abs() < 1e-5); // x
    assert!((output[1] - 2.0).abs() < 1e-5); // y
    assert!((output[2] - 3.0).abs() < 1e-5); // z
    assert!((output[3] - 3.0).abs() < 1e-5); // x + y
    assert!((output[4] - 5.0).abs() < 1e-5); // y + z
}

#[test]
fn test_simd_matmul_large_dimensions() {
    // Test with dimensions that span multiple tiles
    let in_dim = 257; // Just over TILE_SIZE of 64 * 4
    let out_dim = 129; // Just over TILE_SIZE of 64 * 2
    let input: Vec<f32> = (0..in_dim).map(|i| (i % 10) as f32 * 0.1).collect();
    let weight = vec![0.0; out_dim * in_dim];

    let output = simd_matmul(&input, &weight, in_dim, out_dim);

    assert_eq!(output.len(), out_dim);
    for &v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_simd_matmul_tile_boundary() {
    // Test exactly at tile boundary (64)
    let in_dim = 64;
    let out_dim = 64;
    let input: Vec<f32> = vec![1.0; in_dim];

    // Weight: identity matrix at position 0
    let mut weight = vec![0.0; out_dim * in_dim];
    for i in 0..out_dim {
        weight[i * in_dim + i] = 1.0;
    }

    let output = simd_matmul(&input, &weight, in_dim, out_dim);

    assert_eq!(output.len(), out_dim);
    for &v in &output {
        assert!((v - 1.0).abs() < 1e-5);
    }
}

// ============================================================================
// simd_dot Additional Edge Cases
// ============================================================================

#[test]
fn test_simd_dot_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let result = simd_dot(&a, &b);
    assert!((result).abs() < 1e-5);
}

#[test]
fn test_simd_dot_single() {
    let a = vec![3.0];
    let b = vec![4.0];
    let result = simd_dot(&a, &b);
    assert!((result - 12.0).abs() < 1e-5);
}

#[test]
fn test_simd_dot_alternating_signs() {
    let a = vec![1.0, -1.0, 1.0, -1.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let result = simd_dot(&a, &b);
    assert!((result).abs() < 1e-5); // Should cancel out
}

#[test]
fn test_simd_dot_large_values() {
    let a = vec![1e10, 1e10];
    let b = vec![1e10, 1e10];
    let result = simd_dot(&a, &b);
    assert!(result.is_finite());
    assert!((result - 2e20).abs() < 1e15); // Some tolerance for large values
}

#[test]
fn test_simd_dot_small_values() {
    let a = vec![1e-10, 1e-10];
    let b = vec![1e-10, 1e-10];
    let result = simd_dot(&a, &b);
    assert!(result.is_finite());
    assert!((result - 2e-20).abs() < 1e-25);
}

// ============================================================================
// simd_add/simd_mul Edge Cases
// ============================================================================

#[test]
fn test_simd_add_empty() {
    let mut a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    simd_add(&mut a, &b);
    assert!(a.is_empty());
}

#[test]
fn test_simd_add_single() {
    let mut a = vec![1.5];
    let b = vec![2.5];
    simd_add(&mut a, &b);
    assert!((a[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_simd_add_large_vector() {
    let mut a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let b: Vec<f32> = vec![1.0; 1000];
    simd_add(&mut a, &b);

    for (i, &v) in a.iter().enumerate() {
        assert!((v - (i as f32 + 1.0)).abs() < 1e-5);
    }
}

#[test]
fn test_simd_mul_empty() {
    let mut a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    simd_mul(&mut a, &b);
    assert!(a.is_empty());
}

#[test]
fn test_simd_mul_single() {
    let mut a = vec![3.0];
    let b = vec![4.0];
    simd_mul(&mut a, &b);
    assert!((a[0] - 12.0).abs() < 1e-5);
}

#[test]
fn test_simd_mul_large_vector() {
    let mut a: Vec<f32> = vec![2.0; 1000];
    let b: Vec<f32> = vec![3.0; 1000];
    simd_mul(&mut a, &b);

    for &v in &a {
        assert!((v - 6.0).abs() < 1e-5);
    }
}

// ============================================================================
// simd_silu Edge Cases
// ============================================================================

#[test]
fn test_simd_silu_empty() {
    let mut data: Vec<f32> = vec![];
    simd_silu(&mut data);
    assert!(data.is_empty());
}

#[test]
fn test_simd_silu_single() {
    let mut data = vec![0.0];
    simd_silu(&mut data);
    assert!((data[0]).abs() < 1e-5);
}

#[test]
fn test_simd_silu_symmetry() {
    // SiLU is NOT symmetric: silu(-x) != -silu(x)
    let mut pos = vec![2.0];
    let mut neg = vec![-2.0];
    simd_silu(&mut pos);
    simd_silu(&mut neg);

    // |silu(2)| should be much larger than |silu(-2)|
    assert!(pos[0].abs() > neg[0].abs() * 5.0);
}

#[test]
fn test_simd_silu_gradient_like() {
    // Test that silu(x) increases as x increases for positive x
    let mut data: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
    simd_silu(&mut data);

    // For positive x, silu should be monotonically increasing
    for i in 1..data.len() {
        assert!(data[i] >= data[i - 1]);
    }
}

// ============================================================================
// simd_gelu Edge Cases
// ============================================================================

#[test]
fn test_simd_gelu_empty() {
    let mut data: Vec<f32> = vec![];
    simd_gelu(&mut data);
    assert!(data.is_empty());
}

#[test]
fn test_simd_gelu_single() {
    let mut data = vec![0.0];
    simd_gelu(&mut data);
    assert!((data[0]).abs() < 1e-5);
}

#[test]
fn test_simd_gelu_large_positive() {
    let mut data = vec![5.0];
    simd_gelu(&mut data);
    // gelu(5) should be very close to 5
    assert!((data[0] - 5.0).abs() < 0.01);
}

#[test]
fn test_simd_gelu_large_negative() {
    let mut data = vec![-5.0];
    simd_gelu(&mut data);
    // gelu(-5) should be very close to 0
    assert!((data[0]).abs() < 0.01);
}

#[test]
fn test_simd_gelu_batch_consistency() {
    // Verify batch processing gives same result as individual
    let mut batch = vec![0.0, 1.0, -1.0, 2.0, -2.0];
    let mut individual_results = vec![0.0; 5];

    for (i, &v) in [0.0, 1.0, -1.0, 2.0, -2.0].iter().enumerate() {
        let mut single = vec![v];
        simd_gelu(&mut single);
        individual_results[i] = single[0];
    }

    simd_gelu(&mut batch);

    for (batch_val, ind_val) in batch.iter().zip(individual_results.iter()) {
        assert!((batch_val - ind_val).abs() < 1e-5);
    }
}

// ============================================================================
// simd_softmax Edge Cases
// ============================================================================

#[test]
fn test_simd_softmax_two_elements() {
    let mut data = vec![0.0, 0.0];
    simd_softmax(&mut data);

    // Uniform input -> uniform output
    assert!((data[0] - 0.5).abs() < 1e-5);
    assert!((data[1] - 0.5).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_dominant_element() {
    let mut data = vec![0.0, 100.0, 0.0];
    simd_softmax(&mut data);

    // Middle element should dominate
    assert!(data[1] > 0.99);
    assert!(data[0] < 0.01);
    assert!(data[2] < 0.01);
}

#[test]
fn test_simd_softmax_negative_dominant() {
    let mut data = vec![-100.0, 0.0, -100.0];
    simd_softmax(&mut data);

    // Middle element should dominate
    assert!(data[1] > 0.99);
}

#[test]
fn test_simd_softmax_all_same_large() {
    let mut data = vec![1000.0; 4];
    simd_softmax(&mut data);

    // All same -> uniform distribution
    for &v in &data {
        assert!((v - 0.25).abs() < 1e-5);
    }
}

#[test]
fn test_simd_softmax_preserves_sum() {
    let mut data = vec![0.5, -0.5, 1.0, -1.0, 0.0];
    simd_softmax(&mut data);

    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ============================================================================
// BF16/F16 Conversion Edge Cases
// ============================================================================

#[test]
fn test_simd_bf16_to_f32_odd_count() {
    // Odd number of bytes (incomplete last value)
    let bf16_bytes = vec![0u8; 5]; // 2 complete values + 1 incomplete byte
    let result = simd_bf16_to_f32(&bf16_bytes);

    // Should only convert complete pairs (5/2 = 2)
    assert_eq!(result.len(), 2);
}

#[test]
fn test_simd_bf16_to_f32_special_values() {
    // Test zero
    let zero_bytes = half::bf16::from_f32(0.0).to_le_bytes();
    let result = simd_bf16_to_f32(&zero_bytes);
    assert!((result[0]).abs() < 1e-10);

    // Test negative zero
    let neg_zero_bytes = half::bf16::from_f32(-0.0).to_le_bytes();
    let result = simd_bf16_to_f32(&neg_zero_bytes);
    assert!((result[0]).abs() < 1e-10);
}

#[test]
fn test_simd_bf16_to_f32_infinity() {
    let inf_bytes = half::bf16::from_f32(f32::INFINITY).to_le_bytes();
    let result = simd_bf16_to_f32(&inf_bytes);
    assert!(result[0].is_infinite() && result[0].is_sign_positive());

    let neg_inf_bytes = half::bf16::from_f32(f32::NEG_INFINITY).to_le_bytes();
    let result = simd_bf16_to_f32(&neg_inf_bytes);
    assert!(result[0].is_infinite() && result[0].is_sign_negative());
}

#[test]
fn test_simd_bf16_to_f32_nan() {
    let nan_bytes = half::bf16::from_f32(f32::NAN).to_le_bytes();
    let result = simd_bf16_to_f32(&nan_bytes);
    assert!(result[0].is_nan());
}

#[test]
fn test_simd_bf16_to_f32_exact_8_values() {
    // Test exactly 8 values (one SIMD chunk on AVX2)
    let values: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut bf16_bytes = Vec::with_capacity(16);
    for &v in &values {
        bf16_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }

    let result = simd_bf16_to_f32(&bf16_bytes);
    assert_eq!(result.len(), 8);

    for (i, (&expected, &actual)) in values.iter().zip(result.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_simd_f16_to_f32_empty() {
    let result = simd_f16_to_f32(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_simd_f16_to_f32_special_values() {
    let zero_bytes = half::f16::from_f32(0.0).to_le_bytes();
    let result = simd_f16_to_f32(&zero_bytes);
    assert!((result[0]).abs() < 1e-10);

    let inf_bytes = half::f16::from_f32(f32::INFINITY).to_le_bytes();
    let result = simd_f16_to_f32(&inf_bytes);
    assert!(result[0].is_infinite());
}

#[test]
fn test_simd_f16_to_f32_subnormal() {
    // Test a very small F16 value (subnormal)
    let small = half::f16::from_f32(1e-7);
    let bytes = small.to_le_bytes();
    let result = simd_f16_to_f32(&bytes);
    assert!(result[0] >= 0.0);
    assert!(result[0] < 1e-3);
}

// ============================================================================
// BF16 Dot Product Edge Cases
// ============================================================================

#[test]
fn test_simd_bf16_dot_empty() {
    let result = simd_bf16_dot(&[], &[]);
    assert!((result).abs() < 1e-5);
}

#[test]
fn test_simd_bf16_dot_single_value() {
    let a_bytes = half::bf16::from_f32(3.0).to_le_bytes().to_vec();
    let b_bytes = half::bf16::from_f32(4.0).to_le_bytes().to_vec();
    let result = simd_bf16_dot(&a_bytes, &b_bytes);
    assert!((result - 12.0).abs() < 0.1);
}

#[test]
fn test_simd_bf16_dot_mismatched_lengths() {
    let mut a_bytes = Vec::new();
    let mut b_bytes = Vec::new();

    // a has 3 values, b has 2 values
    for v in [1.0, 2.0, 3.0] {
        a_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    for v in [1.0, 1.0] {
        b_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }

    // Should use minimum length
    let result = simd_bf16_dot(&a_bytes, &b_bytes);
    // Only first 2 values used: 1*1 + 2*1 = 3
    assert!((result - 3.0).abs() < 0.1);
}

#[test]
fn test_simd_bf16_dot_chunk_boundary() {
    // Test with exactly 64 values (one chunk)
    let n = 64;
    let mut a_bytes = Vec::with_capacity(n * 2);
    let mut b_bytes = Vec::with_capacity(n * 2);

    for i in 0..n {
        let v = ((i % 5) as f32) * 0.2;
        a_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        b_bytes.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
    }

    let result = simd_bf16_dot(&a_bytes, &b_bytes);
    // Sum of 0.0, 0.2, 0.4, 0.6, 0.8, 0.0, ... (13 complete cycles = 2.0 each, + 3 values)
    // 64/5 = 12 complete + 4 remainder (0.0, 0.2, 0.4, 0.6 = 1.2)
    let expected = 12.0 * 2.0 + 1.2;
    assert!(
        (result - expected).abs() < 1.0,
        "Expected ~{}, got {}",
        expected,
        result
    );
}

// ============================================================================
// BF16 Matmul Edge Cases
// ============================================================================

#[test]
fn test_simd_bf16_matmul_single_element() {
    let input = vec![2.0];
    let weight_bytes = half::bf16::from_f32(3.0).to_le_bytes().to_vec();

    let output = simd_bf16_matmul(&input, &weight_bytes, 1, 1);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 6.0).abs() < 0.1);
}

#[test]
fn test_simd_bf16_matmul_zeros() {
    let input = vec![0.0; 4];
    let mut weight_bytes = Vec::new();
    for _ in 0..8 {
        weight_bytes.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
    }

    let output = simd_bf16_matmul(&input, &weight_bytes, 4, 2);
    for &v in &output {
        assert!(v.abs() < 0.01);
    }
}

#[test]
fn test_simd_bf16_matmul_large() {
    let in_dim = 128;
    let out_dim = 64;
    let input: Vec<f32> = (0..in_dim).map(|i| (i % 10) as f32 * 0.1).collect();

    // Create identity-like weight in BF16
    let mut weight_bytes = Vec::with_capacity(out_dim * in_dim * 2);
    for row in 0..out_dim {
        for col in 0..in_dim {
            let val = if row == col { 1.0 } else { 0.0 };
            weight_bytes.extend_from_slice(&half::bf16::from_f32(val).to_le_bytes());
        }
    }

    let output = simd_bf16_matmul(&input, &weight_bytes, in_dim, out_dim);
    assert_eq!(output.len(), out_dim);

    // First out_dim elements should approximately match input
    for i in 0..out_dim {
        let expected = (i % 10) as f32 * 0.1;
        assert!(
            (output[i] - expected).abs() < 0.05,
            "Index {}: expected {}, got {}",
            i,
            expected,
            output[i]
        );
    }
}

// ============================================================================
// Integration: Activation Chains
// ============================================================================

#[test]
fn test_matmul_silu_chain() {
    let input = vec![1.0, 2.0];
    let weight = vec![
        1.0, 1.0, // sum: 3
        1.0, -1.0, // diff: -1
    ];
    let mut output = simd_matmul(&input, &weight, 2, 2);
    simd_silu(&mut output);

    // silu(3) and silu(-1)
    assert!(output[0] > 2.5); // silu(3) > 2.5
    assert!(output[1] < 0.0); // silu(-1) < 0
}

#[test]
fn test_matmul_gelu_softmax_chain() {
    let input = vec![0.5, 0.5, 0.5, 0.5];
    let weight = vec![
        1.0, 1.0, 1.0, 1.0, // sum: 2
        1.0, 0.0, 0.0, 0.0, // just first: 0.5
        0.0, 0.0, 0.0, 1.0, // just last: 0.5
    ];
    let mut output = simd_matmul(&input, &weight, 4, 3);
    simd_gelu(&mut output);
    simd_softmax(&mut output);

    // Output should sum to 1
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // First element (gelu(2)) should have highest probability
    assert!(output[0] > output[1]);
    assert!(output[0] > output[2]);
}

#[test]
fn test_residual_with_activation() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![
        0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1,
    ];

    let proj = simd_matmul(&input, &weight, 4, 4);
    let mut proj_activated = proj.clone();
    simd_gelu(&mut proj_activated);

    let mut residual = input.clone();
    simd_add(&mut residual, &proj_activated);

    // Should be approximately input + gelu(0.1 * input)
    for (&r, &i) in residual.iter().zip(input.iter()) {
        assert!(r > i); // Residual should be larger than input
        assert!(r < i * 1.5); // But not too much larger
    }
}
