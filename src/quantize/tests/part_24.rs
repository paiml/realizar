//! Part 24: Comprehensive coverage for quantize/mod.rs functions
//!
//! Target: 90% coverage for quantize/mod.rs
//! Focus on:
//! - quantize_activations_q8k_into error branches
//! - quantize_to_q8_blocks error branches
//! - InterleavedQ4K all paths
//! - fused_q4_0_q8_0_parallel_matvec all paths
//! - fused_q8_0_q8_0_parallel_matvec all paths
//! - f16_to_f32_lut

use crate::quantize::{
    dequantize_q8_blocks, f16_to_f32_lut, fused_q4_0_q8_0_parallel_matvec,
    fused_q4_0_q8_0_parallel_matvec_into, fused_q8_0_q8_0_parallel_matvec,
    fused_q8_0_q8_0_parallel_matvec_into, quantize_activations_q8k_into, quantize_to_q8_blocks,
    InterleavedQ4K, Q8_0Block,
};

// Import internal functions for direct testing
use crate::quantize::{
    extract_scale_min, extract_scale_min_from_slice,
    fused_q4_0_q8_0_dot_scalar, fused_q8_0_q8_0_dot_scalar,
};

// =============================================================================
// f16_to_f32_lut tests
// =============================================================================

#[test]
fn test_f16_lut_zero() {
    let result = f16_to_f32_lut(0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_f16_lut_one() {
    // f16 representation of 1.0 is 0x3C00
    let result = f16_to_f32_lut(0x3C00);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_lut_negative_one() {
    // f16 representation of -1.0 is 0xBC00
    let result = f16_to_f32_lut(0xBC00);
    assert!((result - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_f16_lut_half() {
    // f16 representation of 0.5 is 0x3800
    let result = f16_to_f32_lut(0x3800);
    assert!((result - 0.5).abs() < 1e-6);
}

#[test]
fn test_f16_lut_various_values() {
    // Test a range of values to ensure LUT is correctly populated
    let test_cases = [
        (0x0000u16, 0.0f32),      // Zero
        (0x8000, -0.0),           // Negative zero
        (0x3C00, 1.0),            // One
        (0x4000, 2.0),            // Two
        (0x4200, 3.0),            // Three
        (0x3E00, 1.5),            // 1.5
        (0x4400, 4.0),            // Four
        (0x4500, 5.0),            // Five
    ];

    for (bits, expected) in test_cases {
        let result = f16_to_f32_lut(bits);
        assert!(
            (result - expected).abs() < 1e-3,
            "f16 bits {:#06x}: expected {}, got {}",
            bits,
            expected,
            result
        );
    }
}

#[test]
fn test_f16_lut_max_value() {
    // Max normal f16 value
    let result = f16_to_f32_lut(0x7BFF);
    assert!(result > 60000.0 && result < 70000.0);
}

#[test]
fn test_f16_lut_infinity() {
    // f16 positive infinity
    let result = f16_to_f32_lut(0x7C00);
    assert!(result.is_infinite() && result.is_sign_positive());
}

#[test]
fn test_f16_lut_nan() {
    // f16 NaN (any value with exp=0x1F and mantissa!=0)
    let result = f16_to_f32_lut(0x7C01);
    assert!(result.is_nan());
}

// =============================================================================
// quantize_activations_q8k_into tests
// =============================================================================

#[test]
fn test_q8k_into_not_multiple_of_256() {
    let activations = vec![0.0f32; 255]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 255];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("multiple of 256"));
}

#[test]
fn test_q8k_into_scales_too_small() {
    let activations = vec![0.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 1]; // Only 1 scale, need 2
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Scales buffer too small"));
}

#[test]
fn test_q8k_into_quants_too_small() {
    let activations = vec![0.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 256]; // Only 256, need 512

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Quants buffer too small"));
}

#[test]
fn test_q8k_into_success() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    // Scale should be non-zero for non-zero input
    assert!(scales[0].abs() > 0.0);
}

#[test]
fn test_q8k_into_two_superblocks() {
    let activations = vec![0.5f32; 512];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0].abs() > 0.0);
    assert!(scales[1].abs() > 0.0);
}

// =============================================================================
// quantize_to_q8_blocks tests
// =============================================================================

#[test]
fn test_quantize_to_q8_blocks_not_multiple_of_32() {
    let values = vec![0.0f32; 33]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("multiple of 32"));
}

#[test]
fn test_quantize_to_q8_blocks_success() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.unwrap();
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_empty() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.unwrap();
    assert!(blocks.is_empty());
}

#[test]
fn test_quantize_to_q8_blocks_single_block() {
    let values = vec![0.5f32; 32];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.unwrap();
    assert_eq!(blocks.len(), 1);
}

// =============================================================================
// dequantize_q8_blocks tests
// =============================================================================

#[test]
fn test_dequantize_q8_blocks_roundtrip() {
    let original = vec![0.5f32; 64];
    let blocks = quantize_to_q8_blocks(&original).unwrap();
    let dequantized = dequantize_q8_blocks(&blocks);

    assert_eq!(dequantized.len(), original.len());

    // Check approximate equality (quantization introduces error)
    for (o, d) in original.iter().zip(dequantized.iter()) {
        assert!((o - d).abs() < 0.1, "original={}, dequantized={}", o, d);
    }
}

#[test]
fn test_dequantize_q8_blocks_empty() {
    let blocks: Vec<Q8_0Block> = vec![];
    let result = dequantize_q8_blocks(&blocks);
    assert!(result.is_empty());
}

// =============================================================================
// InterleavedQ4K tests
// =============================================================================

#[test]
fn test_interleaved_q4k_invalid_size() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not a multiple of super-block size"));
}

#[test]
fn test_interleaved_q4k_empty() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.unwrap();
    assert_eq!(interleaved.num_super_blocks, 0);
    assert_eq!(interleaved.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_single_superblock() {
    // Create valid Q4_K super-block data (144 bytes)
    let mut data = vec![0u8; 144];
    // Set d (f16 at offset 0-1) to some value
    data[0] = 0x00;
    data[1] = 0x3C; // f16 1.0

    // Set dmin (f16 at offset 2-3)
    data[2] = 0x00;
    data[3] = 0x00; // f16 0.0

    // Scales at offset 4-15 (12 bytes)
    // Quantized values at offset 16-143 (128 bytes)

    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.unwrap();
    assert_eq!(interleaved.num_super_blocks, 1);
    assert_eq!(interleaved.num_values(), 256);
    assert_eq!(interleaved.d.len(), 1);
    assert_eq!(interleaved.dmin.len(), 1);
    assert_eq!(interleaved.scales.len(), 12);
    assert_eq!(interleaved.qs.len(), 128);
}

#[test]
fn test_interleaved_q4k_two_superblocks() {
    let data = vec![0u8; 288]; // 2 super-blocks
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.unwrap();
    assert_eq!(interleaved.num_super_blocks, 2);
    assert_eq!(interleaved.num_values(), 512);
}

#[test]
fn test_interleaved_q4k_dot_wrong_length() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();

    // Wrong activation length
    let activations = vec![1.0f32; 100]; // Should be 256
    let result = interleaved.dot(&activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("doesn't match"));
}

#[test]
fn test_interleaved_q4k_dot_success() {
    // Create Q4_K data with known values
    let mut data = vec![0u8; 144];
    // Set d to 1.0 (f16)
    data[0] = 0x00;
    data[1] = 0x3C;
    // Set some quantized values
    for i in 16..144 {
        data[i] = 0x55; // Arbitrary pattern
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
    // Result should be finite
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_interleaved_q4k_dot_zero_activations() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();

    let activations = vec![0.0f32; 256];
    let result = interleaved.dot(&activations).unwrap();
    assert_eq!(result, 0.0);
}

// =============================================================================
// fused_q4_0_q8_0_parallel_matvec tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_weight_too_small() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("weight data too small"));
}

#[test]
fn test_fused_q4_0_q8_0_activation_mismatch() {
    // Q4_0: 18 bytes per block of 32
    let weight_data = vec![0u8; 18]; // 1 row, 1 block
    let activations = vec![1.0f32; 64]; // Wrong size

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("doesn't match in_dim"));
}

#[test]
fn test_fused_q4_0_q8_0_sequential_path() {
    // Small matrix - sequential path (out_dim < 1024)
    let in_dim = 32;
    let out_dim = 4;
    let blocks_per_row = 1;
    let bytes_per_row = 18; // 1 block

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q4_0_q8_0_parallel_path() {
    // Large matrix - parallel path (out_dim >= 1024)
    let in_dim = 32;
    let out_dim = 2048;
    let bytes_per_row = 18; // 1 block

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q4_0_q8_0_boundary_1024() {
    // Exactly at threshold
    let in_dim = 32;
    let out_dim = 1024;
    let bytes_per_row = 18;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

// =============================================================================
// fused_q4_0_q8_0_parallel_matvec_into tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_into_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];

    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_into_activation_mismatch() {
    let weight_data = vec![0u8; 18];
    let activations = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 1];

    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_into_success() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; out_dim * 18];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_ok());
}

// =============================================================================
// fused_q8_0_q8_0_parallel_matvec tests
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("weight data too small"));
}

#[test]
fn test_fused_q8_0_q8_0_activation_mismatch() {
    // Q8_0: 34 bytes per block of 32
    let weight_data = vec![0u8; 34];
    let activations = vec![1.0f32; 64];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("doesn't match in_dim"));
}

#[test]
fn test_fused_q8_0_q8_0_sequential_path() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_path() {
    let in_dim = 32;
    let out_dim = 2048;
    let bytes_per_row = 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.5f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

// =============================================================================
// fused_q8_0_q8_0_parallel_matvec_into tests
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_into_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];

    let result = fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 1, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_into_activation_mismatch() {
    let weight_data = vec![0u8; 34];
    let activations = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 1];

    let result = fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 1, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_into_success() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; out_dim * 34];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q8_0_q8_0_into_large() {
    let in_dim = 64;
    let out_dim = 128;
    let blocks_per_row = 2;
    let bytes_per_row = blocks_per_row * 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.25f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

// =============================================================================
// Additional coverage for edge cases
// =============================================================================

#[test]
fn test_q4_0_matvec_multiple_blocks_per_row() {
    // 3 blocks per row (96 values)
    let in_dim = 96;
    let out_dim = 2;
    let blocks_per_row = 3;
    let bytes_per_row = blocks_per_row * 18;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q8_0_matvec_multiple_blocks_per_row() {
    let in_dim = 96;
    let out_dim = 2;
    let blocks_per_row = 3;
    let bytes_per_row = blocks_per_row * 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_q8k_into_multiple_superblocks() {
    let activations = vec![0.5f32; 1024]; // 4 superblocks
    let mut scales = vec![0.0f32; 4];
    let mut quants = vec![0i8; 1024];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());

    // All scales should be computed
    for s in &scales {
        assert!(s.abs() > 0.0);
    }
}

#[test]
fn test_interleaved_q4k_varied_values() {
    // Create Q4_K data with varied values
    let mut data = vec![0u8; 144];

    // Set d to 0.5 (f16 0x3800)
    data[0] = 0x00;
    data[1] = 0x38;

    // Set dmin to 0.25 (f16 0x3400)
    data[2] = 0x00;
    data[3] = 0x34;

    // Set some scale values
    for i in 4..16 {
        data[i] = (i as u8) * 5;
    }

    // Set varied quantized values
    for i in 16..144 {
        data[i] = (i as u8) & 0xFF;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();

    // Check d and dmin were parsed
    assert!(interleaved.d[0] > 0.0);
    assert!(interleaved.dmin[0] > 0.0);

    let activations: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
    let result = interleaved.dot(&activations).unwrap();
    assert!(result.is_finite());
}

#[test]
fn test_q4_0_matvec_with_non_zero_weights() {
    let in_dim = 32;
    let out_dim = 2;

    // Create Q4_0 data with non-zero scale
    let mut weight_data = vec![0u8; out_dim * 18];

    // Set f16 scale to 1.0 (0x3C00) for first row
    weight_data[0] = 0x00;
    weight_data[1] = 0x3C;

    // Set some quantized values
    for i in 2..18 {
        weight_data[i] = 0x88; // Both nibbles = 8, centered
    }

    // Second row
    weight_data[18] = 0x00;
    weight_data[19] = 0x3C;
    for i in 20..36 {
        weight_data[i] = 0x44;
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();

    // With non-zero weights and activations, output should be non-zero
    // (though exact values depend on quantization details)
    assert!(output.iter().all(|v| v.is_finite()));
}

// =============================================================================
// extract_scale_min tests
// =============================================================================

#[test]
fn test_extract_scale_min_block_0() {
    let scales: [u8; 12] = [0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min(&scales, 0);
    assert_eq!(scale, 63.0); // 0x3F & 63 = 63
    assert_eq!(min, 42.0);   // 0x2A & 63 = 42
}

#[test]
fn test_extract_scale_min_block_1() {
    let scales: [u8; 12] = [0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min(&scales, 1);
    assert_eq!(scale, 31.0); // 0x1F & 63 = 31
    assert_eq!(min, 21.0);   // 0x15 & 63 = 21
}

#[test]
fn test_extract_scale_min_block_2() {
    let scales: [u8; 12] = [0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min(&scales, 2);
    assert_eq!(scale, 15.0); // 0x0F & 63 = 15
    assert_eq!(min, 10.0);   // 0x0A & 63 = 10
}

#[test]
fn test_extract_scale_min_block_3() {
    let scales: [u8; 12] = [0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min(&scales, 3);
    assert_eq!(scale, 7.0);  // 0x07 & 63 = 7
    assert_eq!(min, 5.0);    // 0x05 & 63 = 5
}

#[test]
fn test_extract_scale_min_block_4() {
    // Block 4-7 use packed layout
    let scales: [u8; 12] = [0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min(&scales, 4);
    // scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = (0x12 & 0x0F) | ((0xC0 >> 6) << 4) = 2 | (3 << 4) = 2 | 48 = 50
    // min = (scales[8] >> 4) | ((scales[4] >> 6) << 4) = (0x12 >> 4) | ((0 >> 6) << 4) = 1 | 0 = 1
    assert_eq!(scale, 50.0);
    assert_eq!(min, 1.0);
}

#[test]
fn test_extract_scale_min_block_5() {
    let scales: [u8; 12] = [0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00];
    let (scale, min) = extract_scale_min(&scales, 5);
    // scale = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4) = (0x34 & 0x0F) | ((0xC0 >> 6) << 4) = 4 | 48 = 52
    // min = (scales[9] >> 4) | ((scales[5] >> 6) << 4) = (0x34 >> 4) | 0 = 3
    assert_eq!(scale, 52.0);
    assert_eq!(min, 3.0);
}

#[test]
fn test_extract_scale_min_block_6() {
    let scales: [u8; 12] = [0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x56, 0x00];
    let (scale, min) = extract_scale_min(&scales, 6);
    // scale = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4) = (0x56 & 0x0F) | ((0xC0 >> 6) << 4) = 6 | 48 = 54
    // min = (scales[10] >> 4) | ((scales[6] >> 6) << 4) = (0x56 >> 4) | 0 = 5
    assert_eq!(scale, 54.0);
    assert_eq!(min, 5.0);
}

#[test]
fn test_extract_scale_min_block_7() {
    let scales: [u8; 12] = [0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x78];
    let (scale, min) = extract_scale_min(&scales, 7);
    // scale = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4) = (0x78 & 0x0F) | ((0xC0 >> 6) << 4) = 8 | 48 = 56
    // min = (scales[11] >> 4) | ((scales[7] >> 6) << 4) = (0x78 >> 4) | 0 = 7
    assert_eq!(scale, 56.0);
    assert_eq!(min, 7.0);
}

// =============================================================================
// extract_scale_min_from_slice tests
// =============================================================================

#[test]
fn test_extract_scale_min_from_slice_even_idx() {
    // idx = 0 (even)
    let scales: Vec<u8> = vec![0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min_from_slice(&scales, 0);
    // scale_idx = 0, min_idx = 4
    // scale = scales[0] & 0x3F = 0x3F = 63
    // min = scales[4] & 0x3F = 0x2A = 42
    assert_eq!(scale, 63.0);
    assert_eq!(min, 42.0);
}

#[test]
fn test_extract_scale_min_from_slice_even_idx_2() {
    let scales: Vec<u8> = vec![0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min_from_slice(&scales, 2);
    // scale_idx = 1, min_idx = 5
    // scale = scales[1] & 0x3F = 0x1F = 31
    // min = scales[5] & 0x3F = 0x15 = 21
    assert_eq!(scale, 31.0);
    assert_eq!(min, 21.0);
}

#[test]
fn test_extract_scale_min_from_slice_odd_idx() {
    // idx = 1 (odd) - uses different extraction logic
    let scales: Vec<u8> = vec![0xC0, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min_from_slice(&scales, 1);
    // scale_idx = 0, min_idx = 4
    // scale = (scales[0] >> 6) | ((scales[2] & 0x0F) << 2) = (0xC0 >> 6) | ((0x0F & 0x0F) << 2) = 3 | 60 = 63
    // min = (scales[4] >> 6) | ((scales[6] & 0x0F) << 2) = (0 >> 6) | ((0x0F & 0x0F) << 2) = 0 | 60 = 60
    assert_eq!(scale, 63.0);
    assert_eq!(min, 60.0);
}

#[test]
fn test_extract_scale_min_from_slice_odd_idx_3() {
    let scales: Vec<u8> = vec![0x00, 0xC0, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00];
    let (scale, min) = extract_scale_min_from_slice(&scales, 3);
    // scale_idx = 1, min_idx = 5
    // scale = (scales[1] >> 6) | ((scales[3] & 0x0F) << 2) = (0xC0 >> 6) | ((0x0F) << 2) = 3 | 60 = 63
    // min = (scales[5] >> 6) | ((scales[7] & 0x0F) << 2) = 0 | 60 = 60
    assert_eq!(scale, 63.0);
    assert_eq!(min, 60.0);
}

// =============================================================================
// fused_q4_0_q8_0_dot_scalar tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_zero() {
    // Q4_0 block: 2 bytes f16 scale + 16 bytes quants
    let q4_data = vec![0u8; 18];
    let q8_scales = vec![0.0f32];
    let q8_quants = vec![0i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_basic() {
    // Create Q4_0 data with scale = 1.0 (f16 0x3C00)
    let mut q4_data = vec![0u8; 18];
    q4_data[0] = 0x00;
    q4_data[1] = 0x3C; // f16 1.0

    // Set quantized values to 8 (centered at 0 after -8 offset)
    for i in 2..18 {
        q4_data[i] = 0x88; // Both nibbles = 8
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    // Result should be 0 since (8-8)=0 for all values
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_nonzero() {
    // Create Q4_0 data
    let mut q4_data = vec![0u8; 18];
    q4_data[0] = 0x00;
    q4_data[1] = 0x3C; // f16 1.0

    // Set quantized values to various values
    for i in 2..18 {
        q4_data[i] = 0xF0; // Low nibble = 0 (-8), high nibble = 15 (7)
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![127i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
    // Result should be non-zero
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_multiple_blocks() {
    // 2 blocks = 64 elements
    let q4_data = vec![0u8; 36]; // 2 * 18 bytes
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 64];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    assert!(result.is_finite());
}

// =============================================================================
// fused_q8_0_q8_0_dot_scalar tests
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_zero() {
    // Q8_0 block: 2 bytes f16 scale + 32 bytes quants = 34 bytes
    let q8_weight = vec![0u8; 34];
    let q8_scales = vec![0.0f32];
    let q8_quants = vec![0i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_basic() {
    // Create Q8_0 weight with scale = 1.0
    let mut q8_weight = vec![0u8; 34];
    q8_weight[0] = 0x00;
    q8_weight[1] = 0x3C; // f16 1.0

    // Set quantized values
    for i in 2..34 {
        q8_weight[i] = 1; // i8 value 1
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
    // Should be 32 * 1 * 1 * 1 * 1 = 32
    assert!((result - 32.0).abs() < 1.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_negative() {
    let mut q8_weight = vec![0u8; 34];
    q8_weight[0] = 0x00;
    q8_weight[1] = 0x3C; // f16 1.0

    // Set quantized values to -1 (0xFF as i8)
    for i in 2..34 {
        q8_weight[i] = 0xFF; // -1 as i8
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
    // Should be 32 * (-1) * 1 * 1 * 1 = -32
    assert!((result - (-32.0)).abs() < 1.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_multiple_blocks() {
    // 2 blocks = 64 elements
    let q8_weight = vec![0u8; 68]; // 2 * 34 bytes
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 64];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 64);
    assert!(result.is_finite());
}

// =============================================================================
// Additional edge case tests for high coverage
// =============================================================================

#[test]
fn test_q4_0_dot_scalar_partial_block() {
    // Test with in_dim not multiple of 32 (partial final block)
    let q4_data = vec![0u8; 36]; // 2 blocks worth
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 50]; // Only 50 elements, not 64

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 50);
    assert!(result.is_finite());
}

#[test]
fn test_q8_0_dot_scalar_partial_block() {
    let q8_weight = vec![0u8; 68];
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 50];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 50);
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_three_superblocks() {
    let data = vec![0u8; 432]; // 3 super-blocks
    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();
    assert_eq!(interleaved.num_super_blocks, 3);
    assert_eq!(interleaved.num_values(), 768);
}

#[test]
fn test_q8k_into_exact_buffer_sizes() {
    let activations = vec![0.5f32; 256];
    let mut scales = vec![0.0f32; 1]; // Exactly 1 needed
    let mut quants = vec![0i8; 256]; // Exactly 256 needed

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

#[test]
fn test_extract_scale_min_all_blocks() {
    // Test all 8 blocks to ensure both branches are covered
    let scales: [u8; 12] = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

    for i in 0..8 {
        let (scale, min) = extract_scale_min(&scales, i);
        assert!(scale >= 0.0);
        assert!(min >= 0.0);
    }
}
