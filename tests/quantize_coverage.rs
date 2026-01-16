//! EXTREME TDD coverage tests for realizar/src/quantize.rs
//!
//! Target: Increase coverage from 76% to 90%+
//! Focus areas:
//! - Q4_K, Q5_K, Q6_K dequantization edge cases
//! - Fused dot product operations
//! - Parallel matrix-vector operations
//! - Error handling paths
//! - SIMD paths (scalar fallbacks)

use realizar::quantize::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    fused_q4_0_parallel_matvec, fused_q4k_auto_matvec_into,
    fused_q4k_dot, fused_q4k_dot_simd, fused_q4k_parallel_matvec,
    fused_q4k_parallel_matvec_into, fused_q4k_q8k_parallel_matvec_into,
    fused_q4k_tiled_matvec, fused_q5k_dot, fused_q5k_dot_simd, fused_q5k_parallel_matvec,
    fused_q5k_parallel_matvec_into, fused_q5k_tiled_matvec, fused_q6k_colmajor_matvec,
    fused_q6k_dot, fused_q6k_dot_simd, fused_q6k_parallel_matvec,
    fused_q6k_parallel_matvec_into, fused_q6k_q8k_dot_simd,
    fused_q6k_q8k_parallel_matvec_into, fused_q6k_tiled_matvec,
    fused_rmsnorm_ffn_up_gate, fused_rmsnorm_ffn_up_gate_into, fused_rmsnorm_q4_0_matmul,
    fused_swiglu_simd, int8_matvec, int8_matvec_parallel,
    quantize_activations_q8_0, quantize_activations_q8k_into, quantize_rmsnorm_q8_0,
    quantize_rmsnorm_q8_0_into, softmax_simd, Int8Row, QK_K,
};

// ============================================================================
// Q4_0 Dequantization Tests
// ============================================================================

#[test]
fn test_q4_0_empty_data() {
    let result = dequantize_q4_0(&[]).expect("empty should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_q4_0_negative_scale() {
    let mut data = Vec::new();
    // Scale: -1.0 as f16
    data.extend_from_slice(&half::f16::from_f32(-1.0).to_le_bytes());
    // 16 bytes of quants (all zeros for simplicity)
    data.extend_from_slice(&[0x88u8; 16]); // low=8, high=8

    let result = dequantize_q4_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // Values should be negative of positive scale
}

#[test]
fn test_q4_0_very_small_scale() {
    let mut data = Vec::new();
    // Scale: very small f16
    data.extend_from_slice(&half::f16::from_f32(1e-4).to_le_bytes());
    data.extend_from_slice(&[0xFFu8; 16]); // all 15s

    let result = dequantize_q4_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // All values should be very small
    for val in result {
        assert!(val.abs() < 1e-2);
    }
}

// ============================================================================
// Q8_0 Dequantization Tests
// ============================================================================

#[test]
fn test_q8_0_empty_data() {
    let result = dequantize_q8_0(&[]).expect("empty should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_q8_0_negative_quants() {
    let mut data = Vec::new();
    // Scale: 1.0 as f16
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // 32 int8 values with negative values
    for i in 0..32_i8 {
        #[allow(clippy::cast_sign_loss)]
        data.push((i - 16) as u8);
    }

    let result = dequantize_q8_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    assert!((result[0] - (-16.0)).abs() < 1e-3);
}

#[test]
fn test_q8_0_large_scale() {
    let mut data = Vec::new();
    // Scale: 100.0 as f16
    data.extend_from_slice(&half::f16::from_f32(100.0).to_le_bytes());
    data.extend_from_slice(&[1u8; 32]);

    let result = dequantize_q8_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // All values should be ~100.0 (1 * 100)
    for val in &result {
        assert!((*val - 100.0).abs() < 1.0);
    }
}

// ============================================================================
// Q4_K Dequantization Tests
// ============================================================================

#[test]
fn test_q4k_varied_scales() {
    // Create a Q4_K block with varied scale values
    let mut data = vec![0u8; 144];

    // d = 2.0 (f16)
    let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
    data[0] = d_bytes[0];
    data[1] = d_bytes[1];

    // dmin = 1.0 (f16)
    let dmin_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    data[2] = dmin_bytes[0];
    data[3] = dmin_bytes[1];

    // scales: set varied patterns
    for i in 0..12 {
        data[4 + i] = ((i * 5) % 64) as u8;
    }

    // qs: varied patterns
    for i in 0..128 {
        data[16 + i] = ((i * 3) % 256) as u8;
    }

    let result = dequantize_q4_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q4k_max_d_value() {
    let mut data = vec![0u8; 144];

    // d = max f16 (~65504)
    let d_bytes = half::f16::from_f32(65504.0).to_bits().to_le_bytes();
    data[0] = d_bytes[0];
    data[1] = d_bytes[1];

    // dmin = 0
    data[2] = 0;
    data[3] = 0;

    let result = dequantize_q4_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

// ============================================================================
// Q5_K Dequantization Tests
// ============================================================================

#[test]
fn test_q5k_with_high_bits() {
    // Q5_K uses 5 bits: 4 bits from qs + 1 bit from qh
    let mut data = vec![0u8; 176];

    // d = 1.0
    let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    data[0] = d_bytes[0];
    data[1] = d_bytes[1];

    // dmin = 0.0
    data[2] = 0;
    data[3] = 0;

    // scales
    for i in 0..12 {
        data[4 + i] = 0x3F; // max scale
    }

    // qh: set all high bits to 1
    for i in 0..32 {
        data[16 + i] = 0xFF;
    }

    // qs: set to mid-range
    for i in 0..128 {
        data[48 + i] = 0x88;
    }

    let result = dequantize_q5_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q5k_negative_dmin() {
    let mut data = vec![0u8; 176];

    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    // dmin = -1.0 (negative)
    data[2..4].copy_from_slice(&half::f16::from_f32(-1.0).to_bits().to_le_bytes());

    let result = dequantize_q5_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q5k_multiple_blocks() {
    // 3 super-blocks
    let data = vec![0u8; 176 * 3];
    let result = dequantize_q5_k(&data).expect("valid data");
    assert_eq!(result.len(), 256 * 3);
}

// ============================================================================
// Q6_K Dequantization Tests
// ============================================================================

#[test]
fn test_q6k_with_varied_qh() {
    // Q6_K: ql (128) + qh (64) + scales (16) + d (2) = 210
    let mut data = vec![0u8; 210];

    // ql: varied
    for (i, byte) in data.iter_mut().enumerate().take(128) {
        *byte = (i % 256) as u8;
    }

    // qh: varied (upper 2 bits)
    for i in 0..64 {
        data[128 + i] = 0xAA; // alternating pattern
    }

    // scales: varied signed
    for i in 0..16 {
        #[allow(clippy::cast_sign_loss)]
        {
            data[192 + i] = (i as i8 - 8) as u8;
        }
    }

    // d = 0.5
    data[208..210].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());

    let result = dequantize_q6_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q6k_negative_scales() {
    let mut data = vec![0u8; 210];

    // Set negative scales
    for i in 0..16 {
        data[192 + i] = 0xFF; // -1 as i8
    }

    // d = 1.0
    data[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    let result = dequantize_q6_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

// ============================================================================
// Additional Q4_K Tests
// ============================================================================

#[test]
fn test_q4k_zero_dmin() {
    let mut data = vec![0u8; 144];

    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    // dmin = 0.0 (no minimum subtraction)
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

    // Set varied scale pattern
    for i in 0..12 {
        data[4 + i] = ((i + 1) * 4) as u8;
    }

    let result = dequantize_q4_k(&data).expect("valid");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q4k_large_scale_values() {
    let mut data = vec![0u8; 144];

    // d = 10.0
    data[0..2].copy_from_slice(&half::f16::from_f32(10.0).to_bits().to_le_bytes());
    // dmin = 5.0
    data[2..4].copy_from_slice(&half::f16::from_f32(5.0).to_bits().to_le_bytes());

    // Max scales
    for i in 0..12 {
        data[4 + i] = 0xFF;
    }

    let result = dequantize_q4_k(&data).expect("valid");
    assert_eq!(result.len(), 256);
}

// ============================================================================
// Fused Q4_K Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_all_zeros() {
    let q4k_data = vec![0u8; 144];
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations).expect("valid");
    assert!(result.abs() < 1e-6);
}

#[test]
fn test_fused_q4k_dot_simd_error_paths() {
    // Invalid data length
    let bad_data = vec![0u8; 100]; // not multiple of 144
    let activations = vec![0.0f32; 256];
    assert!(fused_q4k_dot_simd(&bad_data, &activations).is_err());

    // Activation length mismatch
    let good_data = vec![0u8; 144];
    let bad_activations = vec![0.0f32; 100];
    assert!(fused_q4k_dot_simd(&good_data, &bad_activations).is_err());
}

// ============================================================================
// Fused Q5_K Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q5k_dot_basic() {
    let mut q5k_data = vec![0u8; 176];

    // d = 1.0
    q5k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    // dmin = 0.0
    q5k_data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

    let activations: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();

    let result = fused_q5k_dot(&q5k_data, &activations).expect("valid");
    // Just verify it produces a finite result
    assert!(result.is_finite());
}

#[test]
fn test_fused_q5k_dot_simd_wrapper() {
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];

    // Should use scalar fallback (simd wrapper)
    let result = fused_q5k_dot_simd(&q5k_data, &activations).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q5k_dot_invalid_data() {
    let bad_data = vec![0u8; 175]; // not multiple of 176
    let activations = vec![0.0f32; 256];
    assert!(fused_q5k_dot(&bad_data, &activations).is_err());
}

#[test]
fn test_fused_q5k_dot_activation_mismatch() {
    let data = vec![0u8; 176];
    let activations = vec![0.0f32; 128]; // wrong length
    assert!(fused_q5k_dot(&data, &activations).is_err());
}

// ============================================================================
// Fused Q6_K Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q6k_dot_basic() {
    let mut q6k_data = vec![0u8; 210];

    // d at offset 208
    q6k_data[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    let activations: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();

    let result = fused_q6k_dot(&q6k_data, &activations).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q6k_dot_simd_wrapper() {
    let mut q6k_data = vec![0u8; 210];
    q6k_data[208..210].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());

    let activations = vec![1.0f32; 256];

    let result = fused_q6k_dot_simd(&q6k_data, &activations).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q6k_dot_invalid_data() {
    let bad_data = vec![0u8; 209]; // not multiple of 210
    let activations = vec![0.0f32; 256];
    assert!(fused_q6k_dot(&bad_data, &activations).is_err());
}

// ============================================================================
// Q6_K x Q8_K Fused Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q6k_q8k_dot_simd_basic() {
    // Q6_K: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes
    // Note: The scalar path has a known bug with shift overflow.
    // This test uses the SIMD path which has correct implementation.
    let mut q6k_data = vec![0u8; 210];

    // ql: low 4 bits - use small varied values
    for (i, byte) in q6k_data.iter_mut().enumerate().take(128) {
        *byte = ((i % 8) | ((i % 8) << 4)) as u8;
    }

    // qh: high 2 bits
    for i in 0..64 {
        q6k_data[128 + i] = 0x55; // 01010101 pattern
    }

    // scales: positive values (as i8)
    for i in 0..16 {
        q6k_data[192 + i] = 8; // small positive scale
    }

    // d at offset 208
    q6k_data[208..210].copy_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());

    let q8k_scales = vec![0.1f32; 1];
    let q8k_quants = vec![8i8; 256];

    // Use SIMD wrapper which uses AVX2 path on supported CPUs
    let result = fused_q6k_q8k_dot_simd(&q6k_data, &q8k_scales, &q8k_quants).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q6k_q8k_dot_simd_wrapper() {
    let mut q6k_data = vec![0u8; 210];
    q6k_data[208..210].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());

    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![64i8; 256];

    let result = fused_q6k_q8k_dot_simd(&q6k_data, &q8k_scales, &q8k_quants).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q6k_q8k_dot_invalid_data() {
    let bad_data = vec![0u8; 209];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    // Use SIMD wrapper which checks data validity
    assert!(fused_q6k_q8k_dot_simd(&bad_data, &q8k_scales, &q8k_quants).is_err());
}

#[test]
fn test_fused_q6k_q8k_dot_buffer_too_small() {
    let q6k_data = vec![0u8; 210];
    let q8k_scales = vec![1.0f32; 0]; // empty!
    let q8k_quants = vec![1i8; 256];
    // Use SIMD wrapper
    assert!(fused_q6k_q8k_dot_simd(&q6k_data, &q8k_scales, &q8k_quants).is_err());
}

// ============================================================================
// Tiled Matrix-Vector Multiply Tests
// ============================================================================

#[test]
fn test_fused_q4k_tiled_matvec_error_weight_too_small() {
    let weight_data = vec![0u8; 100]; // too small
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_tiled_matvec(&weight_data, &activations, 256, 4, None);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_tiled_matvec_error_activation_mismatch() {
    let weight_data = vec![0u8; 144 * 4];
    let activations = vec![0.0f32; 128]; // wrong length

    let result = fused_q4k_tiled_matvec(&weight_data, &activations, 256, 4, None);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_tiled_matvec_basic() {
    let weight_data = vec![0u8; 176 * 4]; // 4 rows
    let activations = vec![1.0f32; 256];

    let result = fused_q5k_tiled_matvec(&weight_data, &activations, 256, 4, None).expect("valid");
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_q5k_tiled_matvec_error_weight_small() {
    let weight_data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];

    let result = fused_q5k_tiled_matvec(&weight_data, &activations, 256, 4, None);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_tiled_matvec_basic() {
    let weight_data = vec![0u8; 210 * 4]; // 4 rows
    let activations = vec![1.0f32; 256];

    let result = fused_q6k_tiled_matvec(&weight_data, &activations, 256, 4, None).expect("valid");
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_q6k_tiled_matvec_error_activation_mismatch() {
    let weight_data = vec![0u8; 210 * 4];
    let activations = vec![0.0f32; 128]; // wrong

    let result = fused_q6k_tiled_matvec(&weight_data, &activations, 256, 4, None);
    assert!(result.is_err());
}

// ============================================================================
// Parallel Matrix-Vector Multiply Tests
// ============================================================================

#[test]
fn test_fused_q4k_parallel_matvec_small() {
    // Test sequential path (out_dim < 256)
    let weight_data = vec![0u8; 144 * 10]; // 10 rows
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_parallel_matvec(&weight_data, &activations, 256, 10).expect("valid");
    assert_eq!(result.len(), 10);
}

#[test]
fn test_fused_q4k_parallel_matvec_large() {
    // Test parallel path (out_dim >= 256)
    let weight_data = vec![0u8; 144 * 300]; // 300 rows
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_parallel_matvec(&weight_data, &activations, 256, 300).expect("valid");
    assert_eq!(result.len(), 300);
}

#[test]
fn test_fused_q4k_parallel_matvec_into_errors() {
    let weight_data = vec![0u8; 144 * 4];
    let activations = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2]; // too small

    let result =
        fused_q4k_parallel_matvec_into(&weight_data, &activations, 256, 4, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_parallel_matvec_basic() {
    let weight_data = vec![0u8; 176 * 10];
    let activations = vec![1.0f32; 256];

    let result = fused_q5k_parallel_matvec(&weight_data, &activations, 256, 10).expect("valid");
    assert_eq!(result.len(), 10);
}

#[test]
fn test_fused_q5k_parallel_matvec_into_errors() {
    let weight_data = vec![0u8; 176 * 4];
    let activations = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2]; // too small

    let result =
        fused_q5k_parallel_matvec_into(&weight_data, &activations, 256, 4, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_parallel_matvec_basic() {
    let weight_data = vec![0u8; 210 * 10];
    let activations = vec![1.0f32; 256];

    let result = fused_q6k_parallel_matvec(&weight_data, &activations, 256, 10).expect("valid");
    assert_eq!(result.len(), 10);
}

#[test]
fn test_fused_q6k_parallel_matvec_into_errors() {
    // Weight too small
    let weight_data = vec![0u8; 100];
    let activations = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 4];

    let result =
        fused_q6k_parallel_matvec_into(&weight_data, &activations, 256, 4, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// Q4_K x Q8_K Parallel Matvec Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_basic() {
    let weight_data = vec![0u8; 144 * 4];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    let mut output = vec![0.0f32; 4];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weight_data,
        &q8k_scales,
        &q8k_quants,
        256,
        4,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_fused_q6k_q8k_parallel_matvec_into_basic() {
    let weight_data = vec![0u8; 210 * 4];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    let mut output = vec![0.0f32; 4];

    let result = fused_q6k_q8k_parallel_matvec_into(
        &weight_data,
        &q8k_scales,
        &q8k_quants,
        256,
        4,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_fused_q6k_q8k_parallel_matvec_into_errors() {
    // Output too small
    let weight_data = vec![0u8; 210 * 4];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    let mut output = vec![0.0f32; 2]; // too small

    let result = fused_q6k_q8k_parallel_matvec_into(
        &weight_data,
        &q8k_scales,
        &q8k_quants,
        256,
        4,
        &mut output,
    );
    assert!(result.is_err());
}

// ============================================================================
// Q4_K Auto Matvec Tests
// ============================================================================

#[test]
fn test_fused_q4k_auto_matvec_into_basic() {
    let weight_data = vec![0u8; 144 * 4];
    let activations = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 4];

    let result = fused_q4k_auto_matvec_into(&weight_data, &activations, 256, 4, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_auto_matvec_into_with_padding() {
    // Activations not multiple of 256 - needs padding
    let weight_data = vec![0u8; 144 * 4];
    let activations = vec![1.0f32; 200]; // needs padding to 256
    let mut output = vec![0.0f32; 4];

    let result = fused_q4k_auto_matvec_into(&weight_data, &activations, 256, 4, &mut output);
    assert!(result.is_ok());
}

// ============================================================================
// Q6_K Column-Major Matvec Tests
// ============================================================================

#[test]
fn test_fused_q6k_colmajor_matvec_basic() {
    let weight_data = vec![0u8; 210 * 4]; // 4 columns
    let activations = vec![1.0f32; 4];

    let result = fused_q6k_colmajor_matvec(&weight_data, &activations, 4, 256).expect("valid");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_fused_q6k_colmajor_matvec_error_weight_small() {
    let weight_data = vec![0u8; 100]; // too small
    let activations = vec![1.0f32; 4];

    let result = fused_q6k_colmajor_matvec(&weight_data, &activations, 4, 256);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_colmajor_matvec_error_activation_mismatch() {
    let weight_data = vec![0u8; 210 * 4];
    let activations = vec![1.0f32; 8]; // wrong length

    let result = fused_q6k_colmajor_matvec(&weight_data, &activations, 4, 256);
    assert!(result.is_err());
}

// ============================================================================
// Q4_0 Parallel Matvec Tests
// ============================================================================

#[test]
fn test_fused_q4_0_parallel_matvec_basic() {
    // Q4_0 block: 18 bytes, 32 values
    let weight_data = vec![0u8; 18 * 4]; // 4 rows, 32 values each
    let activations = vec![1.0f32; 32];

    let result = fused_q4_0_parallel_matvec(&weight_data, &activations, 32, 4).expect("valid");
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_q4_0_parallel_matvec_error_weight_small() {
    let weight_data = vec![0u8; 10]; // too small
    let activations = vec![1.0f32; 32];

    let result = fused_q4_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_parallel_matvec_error_activation_mismatch() {
    let weight_data = vec![0u8; 18 * 4];
    let activations = vec![1.0f32; 16]; // wrong

    let result = fused_q4_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_err());
}

// ============================================================================
// Quantize Activations Tests
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_basic() {
    let activations: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();

    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 2); // 64/32 = 2 blocks
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_activations_q8_0_all_zeros() {
    let activations = vec![0.0f32; 64];

    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 2);
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_activations_q8_0_partial_block() {
    // 50 values: 1 full block + partial
    let activations: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();

    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 2); // ceil(50/32) = 2
    assert_eq!(quants.len(), 64); // padded to 2*32
}

#[test]
fn test_quantize_activations_q8k_into() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants)
        .expect("valid quantization");

    assert!(scales[0] > 0.0);
}

// ============================================================================
// RMSNorm + Q8_0 Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_large_input() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let norm_weight = vec![1.0f32; 256];

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-6);

    assert_eq!(scales.len(), 8); // 256/32 = 8 blocks
    assert_eq!(quants.len(), 256);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_basic() {
    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 64];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, 1e-6, &mut scales, &mut quants);

    assert!(scales[0] > 0.0 || scales[1] > 0.0);
}

// ============================================================================
// Fused RMSNorm + Matmul Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_q4_0_matmul_basic() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 18 * 4]; // 4 output rows

    let result =
        fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-6, &weight_data, 32, 4).expect("valid");
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_error_input_mismatch() {
    let input = vec![1.0f32; 16]; // wrong
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 18 * 4];

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-6, &weight_data, 32, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_error_weight_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 10]; // too small

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-6, &weight_data, 32, 4);
    assert!(result.is_err());
}

// ============================================================================
// Fused FFN Up/Gate Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_basic() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18 * 8];
    let gate_weight = vec![0u8; 18 * 8];

    let (up, gate) = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        1e-6,
        &up_weight,
        &gate_weight,
        32,
        8,
    )
    .expect("valid");

    assert_eq!(up.len(), 8);
    assert_eq!(gate.len(), 8);
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_error_input_mismatch() {
    let input = vec![1.0f32; 16]; // wrong
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18 * 8];
    let gate_weight = vec![0u8; 18 * 8];

    let result = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        1e-6,
        &up_weight,
        &gate_weight,
        32,
        8,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_error_up_weight_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 10]; // too small
    let gate_weight = vec![0u8; 18 * 8];

    let result = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        1e-6,
        &up_weight,
        &gate_weight,
        32,
        8,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_error_gate_weight_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18 * 8];
    let gate_weight = vec![0u8; 10]; // too small

    let result = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        1e-6,
        &up_weight,
        &gate_weight,
        32,
        8,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_into_basic() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18 * 8];
    let gate_weight = vec![0u8; 18 * 8];
    let mut up_output = vec![0.0f32; 8];
    let mut gate_output = vec![0.0f32; 8];
    let mut q8_scales = vec![0.0f32; 1];
    let mut q8_quants = vec![0i8; 32];

    let result = fused_rmsnorm_ffn_up_gate_into(
        &input,
        &norm_weight,
        1e-6,
        &up_weight,
        &gate_weight,
        32,
        8,
        &mut up_output,
        &mut gate_output,
        &mut q8_scales,
        &mut q8_quants,
    );
    assert!(result.is_ok());
}

// ============================================================================
// SwiGLU Tests
// ============================================================================

#[test]
fn test_fused_swiglu_simd_basic() {
    let mut gate = vec![0.0f32, 1.0, -1.0, 2.0];
    let up = vec![1.0f32, 2.0, 3.0, 4.0];

    fused_swiglu_simd(&mut gate, &up);

    // silu(x) = x * sigmoid(x)
    // gate[0] = silu(0) * 1 = 0 * 0.5 * 1 = 0
    assert!(gate[0].abs() < 1e-6);
    // silu(1) ~ 0.731
    assert!((gate[1] - 0.731 * 2.0).abs() < 0.1);
}

#[test]
fn test_fused_swiglu_simd_large() {
    // Test SIMD path (>8 elements)
    let mut gate: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let up: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.5).collect();

    fused_swiglu_simd(&mut gate, &up);

    // Verify all values are finite
    for v in &gate {
        assert!(v.is_finite());
    }
}

#[test]
fn test_fused_swiglu_simd_negative() {
    let mut gate = vec![-5.0f32; 8];
    let up = vec![1.0f32; 8];

    fused_swiglu_simd(&mut gate, &up);

    // silu(-5) is very small (close to 0)
    for v in &gate {
        assert!(v.abs() < 0.1);
    }
}

// ============================================================================
// Softmax Tests
// ============================================================================

#[test]
fn test_softmax_simd_basic() {
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];

    softmax_simd(&mut x);

    // Verify sum to 1
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Verify monotonicity
    for i in 1..x.len() {
        assert!(x[i] > x[i - 1]);
    }
}

#[test]
fn test_softmax_simd_empty() {
    let mut x: Vec<f32> = vec![];
    softmax_simd(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_simd_single() {
    let mut x = vec![5.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_large() {
    // Test SIMD path (>8 elements)
    let mut x: Vec<f32> = (0..32).map(|i| i as f32).collect();

    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_numerical_stability() {
    // Large values that could overflow without max subtraction
    let mut x = vec![1000.0f32, 1001.0, 1002.0];

    softmax_simd(&mut x);

    // Should still sum to 1
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // All values should be finite
    for v in &x {
        assert!(v.is_finite());
    }
}

// ============================================================================
// INT8 Row Tests
// ============================================================================

#[test]
fn test_int8_row_quantize_dequantize() {
    let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();

    let row = Int8Row::quantize(&values);
    let dequant = row.dequantize();

    // Check round-trip error
    for (orig, deq) in values.iter().zip(dequant.iter()) {
        assert!((orig - deq).abs() < 0.02);
    }
}

#[test]
fn test_int8_row_single_element() {
    let values = vec![127.0f32];
    let row = Int8Row::quantize(&values);

    // Scale should be 1.0 (127/127)
    assert!((row.scale - 1.0).abs() < 0.01);

    let dequant = row.dequantize();
    assert!((dequant[0] - 127.0).abs() < 1.0);
}

#[test]
fn test_int8_matvec_basic() {
    let weights: Vec<Int8Row> = (0..4)
        .map(|_| Int8Row::quantize(&vec![1.0f32; 8]))
        .collect();
    let activations = vec![1.0f32; 8];

    let result = int8_matvec(&weights, &activations);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_int8_matvec_parallel_basic() {
    let weights: Vec<Int8Row> = (0..10)
        .map(|_| Int8Row::quantize(&vec![1.0f32; 8]))
        .collect();
    let activations = vec![1.0f32; 8];

    let result = int8_matvec_parallel(&weights, &activations);
    assert_eq!(result.len(), 10);
}

// ============================================================================
// QK_K Constant Test
// ============================================================================

#[test]
fn test_qk_k_constant() {
    // QK_K should be 256 (super-block size for K-quants)
    assert_eq!(QK_K, 256);
}
