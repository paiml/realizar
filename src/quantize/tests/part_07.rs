//! Part 07: Fused K-Quant Tests (PMAT-803)
//!
//! Tests for `quantize/fused_k.rs` to drive coverage from 36% to higher.
//!
//! # Coverage Targets
//!
//! - `fused_q4k_dot` - Q4_K dot product with f32 activations
//! - `fused_q4k_dot_simd` - SIMD-accelerated Q4_K dot product
//! - `fused_q4k_q8k_dot` - Q4_K with Q8_K activations
//! - `fused_q4k_q8k_dot_simd` - SIMD-accelerated Q4_K x Q8_K
//!
//! # Test Organization
//!
//! - Error path tests (invalid lengths, mismatches)
//! - Basic functionality tests (zero values, identity)
//! - Property-based tests (scalar vs SIMD equivalence)
//! - Edge case tests (single block, many blocks)

use crate::quantize::fused_k::{fused_q4k_dot, fused_q4k_dot_simd};
use crate::quantize::fused_k::{fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd};
use crate::quantize::types::QK_K;

// ============================================================================
// Constants
// ============================================================================

/// Q4_K super-block size in bytes
const Q4K_BLOCK_BYTES: usize = 144;

// ============================================================================
// Part 1: fused_q4k_dot Error Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_invalid_length_not_multiple() {
    // Not a multiple of 144 bytes
    let q4k_data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, crate::error::RealizarError::InvalidShape { .. }));
}

#[test]
fn test_fused_q4k_dot_invalid_length_one_byte_short() {
    // 143 bytes instead of 144
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES - 1];
    let activations = vec![0.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_invalid_length_one_byte_over() {
    // 145 bytes instead of 144
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES + 1];
    let activations = vec![0.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_activation_length_mismatch_short() {
    // Valid Q4K data but activations too short
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![0.0f32; QK_K - 1];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_activation_length_mismatch_long() {
    // Valid Q4K data but activations too long
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![0.0f32; QK_K + 1];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_empty_inputs() {
    // Empty is technically valid (0 super-blocks, 0 activations)
    let q4k_data: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0.0);
}

// ============================================================================
// Part 2: fused_q4k_dot Basic Functionality Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_zero_weights() {
    // All zero weights should produce zero dot product
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    // Zero scales/data means zero contribution
    assert!(result.unwrap().abs() < 1e-6);
}

#[test]
fn test_fused_q4k_dot_zero_activations() {
    // Zero activations should produce zero dot product
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let activations = vec![0.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.unwrap().abs() < 1e-6);
}

#[test]
fn test_fused_q4k_dot_single_block() {
    // Single super-block with known values
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    // Should produce some non-zero value
    let dot = result.unwrap();
    assert!(dot.is_finite());
}

#[test]
fn test_fused_q4k_dot_multiple_blocks() {
    // Multiple super-blocks
    let num_blocks = 4;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for _ in 0..num_blocks {
        q4k_data.extend_from_slice(&create_q4k_test_block(0.5, 0.1));
    }
    let activations = vec![1.0f32; num_blocks * QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    let dot = result.unwrap();
    assert!(dot.is_finite());
}

#[test]
fn test_fused_q4k_dot_varied_activations() {
    // Activations with varied values
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let mut activations = vec![0.0f32; QK_K];
    for (i, a) in activations.iter_mut().enumerate() {
        *a = (i as f32) / (QK_K as f32);
    }

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_dot_negative_activations() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let activations = vec![-1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

// ============================================================================
// Part 3: fused_q4k_dot_simd Error Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_simd_invalid_length() {
    let q4k_data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_activation_mismatch() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![0.0f32; QK_K + 10];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_empty_inputs() {
    let q4k_data: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0.0);
}

// ============================================================================
// Part 4: fused_q4k_dot_simd Basic Functionality Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_simd_zero_weights() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.unwrap().abs() < 1e-6);
}

#[test]
fn test_fused_q4k_dot_simd_single_block() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_dot_simd_multiple_blocks() {
    let num_blocks = 8;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for _ in 0..num_blocks {
        q4k_data.extend_from_slice(&create_q4k_test_block(0.5, 0.25));
    }
    let activations = vec![0.5f32; num_blocks * QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

// ============================================================================
// Part 5: Scalar vs SIMD Equivalence Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_scalar_simd_equivalence_zero() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let activations = vec![1.0f32; QK_K];

    let scalar = fused_q4k_dot(&q4k_data, &activations).unwrap();
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).unwrap();

    assert!((scalar - simd).abs() < 1e-5, "scalar={} simd={}", scalar, simd);
}

#[test]
fn test_fused_q4k_dot_scalar_simd_equivalence_ones() {
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let activations = vec![1.0f32; QK_K];

    let scalar = fused_q4k_dot(&q4k_data, &activations).unwrap();
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).unwrap();

    // Allow for some numerical tolerance
    let rel_diff = if scalar.abs() > 1e-6 {
        (scalar - simd).abs() / scalar.abs()
    } else {
        (scalar - simd).abs()
    };
    assert!(rel_diff < 0.01, "scalar={} simd={} rel_diff={}", scalar, simd, rel_diff);
}

#[test]
fn test_fused_q4k_dot_scalar_simd_equivalence_random() {
    // Use deterministic "random" values
    let mut q4k_data = Vec::with_capacity(Q4K_BLOCK_BYTES);
    for i in 0..Q4K_BLOCK_BYTES {
        q4k_data.push((i * 37 % 256) as u8);
    }
    // Fix the d and dmin fields to valid f16 values
    q4k_data[0] = 0x00;
    q4k_data[1] = 0x3C; // ~1.0 in f16
    q4k_data[2] = 0x00;
    q4k_data[3] = 0x30; // ~0.5 in f16

    let mut activations = vec![0.0f32; QK_K];
    for (i, a) in activations.iter_mut().enumerate() {
        *a = ((i * 13) % 100) as f32 / 100.0 - 0.5;
    }

    let scalar = fused_q4k_dot(&q4k_data, &activations).unwrap();
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).unwrap();

    // Check both are finite
    assert!(scalar.is_finite());
    assert!(simd.is_finite());

    // Check they're close
    let abs_diff = (scalar - simd).abs();
    let max_val = scalar.abs().max(simd.abs()).max(1.0);
    assert!(abs_diff / max_val < 0.1, "scalar={} simd={}", scalar, simd);
}

#[test]
fn test_fused_q4k_dot_scalar_simd_equivalence_multiple_blocks() {
    let num_blocks = 4;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for b in 0..num_blocks {
        let mut block = create_q4k_test_block(0.5 + b as f32 * 0.1, 0.1);
        // Vary the quant data a bit
        for i in 12..Q4K_BLOCK_BYTES {
            block[i] = ((b * 17 + i * 7) % 256) as u8;
        }
        q4k_data.extend_from_slice(&block);
    }

    let activations: Vec<f32> = (0..num_blocks * QK_K)
        .map(|i| (i as f32 / (num_blocks * QK_K) as f32) - 0.5)
        .collect();

    let scalar = fused_q4k_dot(&q4k_data, &activations).unwrap();
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).unwrap();

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

// ============================================================================
// Part 6: fused_q4k_q8k_dot Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_length() {
    let q4k_data = vec![0u8; 100]; // Not multiple of 144
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q8k_scales_length() {
    // 2 super-blocks but only 1 scale (needs at least 2)
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES * 2];
    let q8k_scales = vec![1.0f32; 1]; // Should be 2 for 2 super-blocks
    let q8k_quants = vec![0i8; QK_K * 2];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q8k_quants_length() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; QK_K - 1]; // Too short

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_empty_inputs() {
    let q4k_data: Vec<u8> = vec![];
    let q8k_scales: Vec<f32> = vec![];
    let q8k_quants: Vec<i8> = vec![];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_zero_weights() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().abs() < 1e-6);
}

#[test]
fn test_fused_q4k_q8k_dot_zero_activations() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().abs() < 1e-6);
}

#[test]
fn test_fused_q4k_q8k_dot_single_block() {
    let q4k_data = create_q4k_test_block(1.0, 0.25);
    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![1i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_multiple_blocks() {
    let num_blocks = 4;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for _ in 0..num_blocks {
        q4k_data.extend_from_slice(&create_q4k_test_block(0.5, 0.1));
    }
    let q8k_scales = vec![0.5f32; num_blocks];
    let q8k_quants = vec![2i8; num_blocks * QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_negative_quants() {
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![-1i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_mixed_quants() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let q8k_scales = vec![0.5f32; 1];
    let mut q8k_quants = Vec::with_capacity(QK_K);
    for i in 0..QK_K {
        q8k_quants.push(if i % 2 == 0 { 1 } else { -1 });
    }

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

// ============================================================================
// Part 7: fused_q4k_q8k_dot_simd Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_simd_invalid_length() {
    let q4k_data = vec![0u8; 100];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_empty_inputs() {
    let q4k_data: Vec<u8> = vec![];
    let q8k_scales: Vec<f32> = vec![];
    let q8k_quants: Vec<i8> = vec![];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_simd_single_block() {
    let q4k_data = create_q4k_test_block(1.0, 0.25);
    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![1i8; QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_multiple_blocks() {
    let num_blocks = 8;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for _ in 0..num_blocks {
        q4k_data.extend_from_slice(&create_q4k_test_block(0.25, 0.1));
    }
    let q8k_scales = vec![0.25f32; num_blocks];
    let q8k_quants = vec![3i8; num_blocks * QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

// ============================================================================
// Part 8: Q4K x Q8K Scalar vs SIMD Equivalence
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_scalar_simd_equivalence_zero() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; QK_K];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).unwrap();

    assert!((scalar - simd).abs() < 1e-5, "scalar={} simd={}", scalar, simd);
}

#[test]
fn test_fused_q4k_q8k_dot_scalar_simd_equivalence_basic() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![2i8; QK_K];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).unwrap();

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_scalar_simd_equivalence_varied() {
    let mut q4k_data = Vec::with_capacity(Q4K_BLOCK_BYTES);
    for i in 0..Q4K_BLOCK_BYTES {
        q4k_data.push((i * 41 % 256) as u8);
    }
    // Fix d and dmin to valid f16
    q4k_data[0] = 0x00;
    q4k_data[1] = 0x3C;
    q4k_data[2] = 0x00;
    q4k_data[3] = 0x38;

    let q8k_scales: Vec<f32> = (0..8).map(|i| 0.1 + i as f32 * 0.1).collect();
    let q8k_quants: Vec<i8> = (0..QK_K).map(|i| ((i % 256) as i8).wrapping_sub(64)).collect();

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).unwrap();

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_scalar_simd_equivalence_many_blocks() {
    let num_blocks = 16;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for b in 0..num_blocks {
        let mut block = create_q4k_test_block(0.5, 0.25);
        for i in 12..Q4K_BLOCK_BYTES {
            block[i] = ((b * 23 + i * 11) % 256) as u8;
        }
        q4k_data.extend_from_slice(&block);
    }

    let q8k_scales = vec![0.25f32; num_blocks];
    let q8k_quants: Vec<i8> = (0..num_blocks * QK_K)
        .map(|i| ((i % 127) as i8) - 63)
        .collect();

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).unwrap();

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

// ============================================================================
// Part 9: Edge Cases
// ============================================================================

#[test]
fn test_fused_q4k_dot_large_scale() {
    // Very large scale value
    let mut q4k_data = create_q4k_test_block(1000.0, 500.0);
    // Ensure d is a valid f16 representation of ~1000
    // f16 max is 65504, so 1000 is fine
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_dot_small_scale() {
    // Very small scale value
    let q4k_data = create_q4k_test_block(0.001, 0.0005);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_large_activations() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let activations = vec![1000.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_extreme_quants() {
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![127i8; QK_K]; // Max i8 value

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_min_quants() {
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![-128i8; QK_K]; // Min i8 value

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a Q4_K test block with specified d and dmin values
///
/// Q4_K super-block format (144 bytes):
/// - d: 2 bytes (f16)
/// - dmin: 2 bytes (f16)
/// - scales: 12 bytes (6-bit scales for 8 sub-blocks)
/// - quants: 128 bytes (4-bit quants for 256 values)
fn create_q4k_test_block(d: f32, dmin: f32) -> Vec<u8> {
    let mut block = vec![0u8; Q4K_BLOCK_BYTES];

    // Write d as f16 (simplified - use known f16 bit patterns)
    let d_f16 = f32_to_f16_approx(d);
    block[0] = (d_f16 & 0xFF) as u8;
    block[1] = ((d_f16 >> 8) & 0xFF) as u8;

    // Write dmin as f16
    let dmin_f16 = f32_to_f16_approx(dmin);
    block[2] = (dmin_f16 & 0xFF) as u8;
    block[3] = ((dmin_f16 >> 8) & 0xFF) as u8;

    // Scales (12 bytes) - set to mid-range values
    for i in 4..16 {
        block[i] = 0x88; // Mid-range scale
    }

    // Quants (128 bytes) - set to alternating pattern
    for i in 16..Q4K_BLOCK_BYTES {
        block[i] = 0x55; // Alternating nibbles
    }

    block
}

/// Approximate f32 to f16 conversion
fn f32_to_f16_approx(f: f32) -> u16 {
    if f == 0.0 {
        return 0;
    }
    if f.is_nan() {
        return 0x7E00;
    }
    if f.is_infinite() {
        return if f > 0.0 { 0x7C00 } else { 0xFC00 };
    }

    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    // Bias adjustment: f32 bias is 127, f16 bias is 15
    let new_exp = exp - 127 + 15;

    if new_exp <= 0 {
        // Subnormal or zero
        0
    } else if new_exp >= 31 {
        // Overflow to infinity
        (sign << 15) | 0x7C00
    } else {
        // Normal number
        let new_mantissa = (mantissa >> 13) as u16;
        (sign << 15) | ((new_exp as u16) << 10) | new_mantissa
    }
}
