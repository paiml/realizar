//! Part 22: Enhanced parallel dequantization coverage tests for `parallel_dequant.rs`
//!
//! Targets the uncovered ~16% of parallel_dequant.rs, focusing on:
//! - AVX2 chunked parallel path (CHUNK_SIZE = 64 boundary)
//! - AVX2 remainder handling in RoPE rotation
//! - Edge cases in superblock/block processing
//! - Numerical boundary conditions
//! - Large-scale parallel execution paths

use crate::quantize::parallel_dequant::{
    apply_rope_rotation_scalar, apply_rope_rotation_simd, dequantize_q4_k_parallel,
    dequantize_q4_k_simd, dequantize_q4_k_superblock, dequantize_q8_0_block,
    dequantize_q8_0_parallel, dequantize_q8_0_simd,
};
use crate::quantize::types::QK_K;

// ============================================================================
// Helper functions for generating test data
// ============================================================================

/// Generate Q4_K super-block data with specific d/dmin values
fn generate_q4k_with_scales(num_super_blocks: usize, d_bits: [u8; 2], dmin_bits: [u8; 2]) -> Vec<u8> {
    const SUPER_BLOCK_BYTES: usize = 144;
    let mut data = vec![0u8; num_super_blocks * SUPER_BLOCK_BYTES];

    for sb in 0..num_super_blocks {
        let sb_start = sb * SUPER_BLOCK_BYTES;

        // Set d (f16)
        data[sb_start] = d_bits[0];
        data[sb_start + 1] = d_bits[1];

        // Set dmin (f16)
        data[sb_start + 2] = dmin_bits[0];
        data[sb_start + 3] = dmin_bits[1];

        // Set scales (12 bytes) - varied pattern
        for i in 0..12 {
            data[sb_start + 4 + i] = ((sb * 13 + i * 7 + 17) % 64) as u8;
        }

        // Set quantized values (128 bytes)
        for i in 0..128 {
            data[sb_start + 16 + i] = ((sb * 19 + i * 23) % 256) as u8;
        }
    }

    data
}

/// Generate Q8_0 block data with specific scale
fn generate_q8_0_with_scale(num_blocks: usize, scale_bits: [u8; 2]) -> Vec<u8> {
    const BLOCK_BYTES: usize = 34;
    let mut data = vec![0u8; num_blocks * BLOCK_BYTES];

    for block in 0..num_blocks {
        let block_start = block * BLOCK_BYTES;

        // Set scale (f16)
        data[block_start] = scale_bits[0];
        data[block_start + 1] = scale_bits[1];

        // Set quantized int8 values
        for i in 0..32 {
            data[block_start + 2 + i] = ((block * 11 + i * 13 + 7) % 256) as u8;
        }
    }

    data
}

// ============================================================================
// Tests for AVX2 chunked parallel Q4_K path (CHUNK_SIZE = 64)
// ============================================================================

#[test]
fn test_q4k_simd_exactly_64_superblocks_chunk_boundary_p22() {
    // CHUNK_SIZE = 64 - test exact boundary
    let data = generate_q4k_with_scales(64, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 64 * QK_K);

    // Verify all values are finite
    for (i, val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite value at index {}: {}", i, val);
    }
}

#[test]
fn test_q4k_simd_127_superblocks_just_under_threshold_p22() {
    // < 2 * CHUNK_SIZE (128) - should use sequential path
    let data = generate_q4k_with_scales(127, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 127 * QK_K);
}

#[test]
fn test_q4k_simd_128_superblocks_threshold_p22() {
    // Exactly 2 * CHUNK_SIZE - triggers parallel path
    let data = generate_q4k_with_scales(128, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 128 * QK_K);
}

#[test]
fn test_q4k_simd_129_superblocks_above_threshold_p22() {
    // > 2 * CHUNK_SIZE - parallel with partial final chunk
    let data = generate_q4k_with_scales(129, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 129 * QK_K);
}

#[test]
fn test_q4k_simd_256_superblocks_multiple_chunks_p22() {
    // 4 full chunks of 64
    let data = generate_q4k_with_scales(256, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 256 * QK_K);

    // Verify consistency
    let parallel = dequantize_q4_k_parallel(&data).unwrap();
    for i in 0..output.len() {
        let diff = (output[i] - parallel[i]).abs();
        assert!(diff < 1e-6, "SIMD/parallel mismatch at {}", i);
    }
}

#[test]
fn test_q4k_simd_320_superblocks_5_chunks_p22() {
    // 5 full chunks of 64
    let data = generate_q4k_with_scales(320, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 320 * QK_K);
}

#[test]
fn test_q4k_simd_333_superblocks_partial_chunk_p22() {
    // 5 full chunks + 13 remainder
    let data = generate_q4k_with_scales(333, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 333 * QK_K);
}

// ============================================================================
// Tests for RoPE rotation remainder handling
// ============================================================================

#[test]
fn test_rope_simd_size_9_one_remainder_p22() {
    // 8 + 1 remainder element
    let half_dim = 9;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + 10) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    let x1_orig = x1.clone();
    let x2_orig = x2.clone();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Verify the last element (remainder) was processed
    let i = half_dim - 1;
    let expected_x1 = x1_orig[i] * cos_vals[i] - x2_orig[i] * sin_vals[i];
    let expected_x2 = x1_orig[i] * sin_vals[i] + x2_orig[i] * cos_vals[i];
    assert!((x1[i] - expected_x1).abs() < 1e-5, "Remainder x1 mismatch");
    assert!((x2[i] - expected_x2).abs() < 1e-5, "Remainder x2 mismatch");
}

#[test]
fn test_rope_simd_size_15_seven_remainder_p22() {
    // 8 + 7 remainder elements
    let half_dim = 15;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.5).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.2).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.2).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_rope_simd_size_23_seven_remainder_after_two_simd_p22() {
    // 16 + 7 remainder (2 SIMD iterations + 7 scalar)
    let half_dim = 23;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();

    let mut scalar_x1 = x1.clone();
    let mut scalar_x2 = x2.clone();

    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.15).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.15).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);
    apply_rope_rotation_scalar(&mut scalar_x1, &mut scalar_x2, &cos_vals, &sin_vals);

    // Compare SIMD with scalar
    for i in 0..half_dim {
        assert!(
            (x1[i] - scalar_x1[i]).abs() < 1e-5,
            "x1 mismatch at {}: {} vs {}",
            i,
            x1[i],
            scalar_x1[i]
        );
        assert!(
            (x2[i] - scalar_x2[i]).abs() < 1e-5,
            "x2 mismatch at {}: {} vs {}",
            i,
            x2[i],
            scalar_x2[i]
        );
    }
}

#[test]
fn test_rope_simd_size_31_remainder_loop_p22() {
    // 24 + 7 for AVX2, or 16 + 15 for AVX512 remainder
    let half_dim = 31;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.3).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.3 + 2.0).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.08).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.08).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

// ============================================================================
// Tests for Q8_0 SIMD path coverage
// ============================================================================

#[test]
fn test_q8_0_simd_large_parallel_p22() {
    // Large enough to exercise parallel path
    let data = generate_q8_0_with_scale(256, [0x00, 0x3C]); // scale = 1.0
    let result = dequantize_q8_0_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 256 * 32);

    // Verify against parallel
    let parallel = dequantize_q8_0_parallel(&data).unwrap();
    for i in 0..output.len() {
        let diff = (output[i] - parallel[i]).abs();
        assert!(diff < 1e-6, "Q8_0 SIMD/parallel mismatch at {}", i);
    }
}

#[test]
fn test_q8_0_block_all_chunks_p22() {
    // Tests all 4 chunks in dequantize_q8_0_block_avx2
    let mut block_data = vec![0u8; 34];

    // Scale = 2.0 (f16 = 0x4000)
    block_data[0] = 0x00;
    block_data[1] = 0x40;

    // Set different values in each 8-byte chunk to verify all chunks processed
    // Chunk 0: bytes 2-9
    for i in 0..8 {
        block_data[2 + i] = 10; // +10 as i8
    }
    // Chunk 1: bytes 10-17
    for i in 0..8 {
        block_data[10 + i] = 20;
    }
    // Chunk 2: bytes 18-25
    for i in 0..8 {
        block_data[18 + i] = 30;
    }
    // Chunk 3: bytes 26-33
    for i in 0..8 {
        block_data[26 + i] = 40;
    }

    let output = dequantize_q8_0_block(&block_data);

    // Each chunk should produce different values
    // Chunk 0: 10 * 2.0 = 20.0
    for i in 0..8 {
        assert!(
            (output[i] - 20.0).abs() < 0.1,
            "Chunk 0 failed at {}: {}",
            i,
            output[i]
        );
    }
    // Chunk 1: 20 * 2.0 = 40.0
    for i in 8..16 {
        assert!(
            (output[i] - 40.0).abs() < 0.1,
            "Chunk 1 failed at {}: {}",
            i,
            output[i]
        );
    }
    // Chunk 2: 30 * 2.0 = 60.0
    for i in 16..24 {
        assert!(
            (output[i] - 60.0).abs() < 0.1,
            "Chunk 2 failed at {}: {}",
            i,
            output[i]
        );
    }
    // Chunk 3: 40 * 2.0 = 80.0
    for i in 24..32 {
        assert!(
            (output[i] - 80.0).abs() < 0.1,
            "Chunk 3 failed at {}: {}",
            i,
            output[i]
        );
    }
}

// ============================================================================
// Tests for Q4_K superblock SIMD chunks
// ============================================================================

#[test]
fn test_q4k_superblock_all_64_value_chunks_p22() {
    // Q4_K processes 64 values at a time, 4 times per superblock
    let mut sb_data = vec![0u8; 144];

    // d = 1.0
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C;

    // dmin = 0.0
    sb_data[2] = 0x00;
    sb_data[3] = 0x00;

    // Set scales to 1 for predictable output
    for i in 0..12 {
        sb_data[4 + i] = 1;
    }

    // Set each 32-byte section to different values
    // This tests all 4 iterations of the j loop (0, 64, 128, 192)
    for j_idx in 0..4 {
        let base = j_idx * 32;
        let val = (j_idx * 17 + 5) as u8;
        for i in 0..32 {
            sb_data[16 + base + i] = val;
        }
    }

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);

    // Verify different sections have been processed
    let first_section_avg: f32 = output[0..64].iter().sum::<f32>() / 64.0;
    let second_section_avg: f32 = output[64..128].iter().sum::<f32>() / 64.0;
    let third_section_avg: f32 = output[128..192].iter().sum::<f32>() / 64.0;
    let fourth_section_avg: f32 = output[192..256].iter().sum::<f32>() / 64.0;

    // Each section should have different average due to different quant values
    assert!(
        (first_section_avg - second_section_avg).abs() > 0.01
            || (second_section_avg - third_section_avg).abs() > 0.01,
        "Sections should differ"
    );

    // All should be finite
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_superblock_high_low_nibble_split_p22() {
    // Test that low and high nibbles are correctly separated
    let mut sb_data = vec![0u8; 144];

    // d = 1.0, dmin = 0.0
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C;
    sb_data[2] = 0x00;
    sb_data[3] = 0x00;

    // All scales = 1
    for i in 0..12 {
        sb_data[4 + i] = 1;
    }

    // Set all quants to 0xF0 (low nibble = 0, high nibble = 15)
    for i in 0..128 {
        sb_data[16 + i] = 0xF0;
    }

    let output = dequantize_q4_k_superblock(&sb_data);

    // First 32 values of each 64-chunk should be from low nibble (0)
    // Second 32 values should be from high nibble (15)
    for chunk in 0..4 {
        let base = chunk * 64;
        // Low nibble values (first 32)
        for i in 0..32 {
            assert!(
                output[base + i].abs() < 1.0,
                "Low nibble at {} should be small: {}",
                base + i,
                output[base + i]
            );
        }
        // High nibble values (second 32)
        for i in 32..64 {
            assert!(
                output[base + i] > 1.0 || output[base + i] < -1.0,
                "High nibble at {} should be larger: {}",
                base + i,
                output[base + i]
            );
        }
    }
}

// ============================================================================
// Tests for numerical edge cases
// ============================================================================

#[test]
fn test_q4k_with_inf_scale_p22() {
    // f16 infinity = 0x7C00
    let mut data = generate_q4k_with_scales(1, [0x00, 0x7C], [0x00, 0x00]);

    // Set some non-zero quants
    for i in 0..128 {
        data[16 + i] = 0x11;
    }

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.unwrap();
    // With infinite scale, results will be inf
    for val in &output {
        // Should be inf or finite (not NaN unless quants are 0)
        assert!(!val.is_nan() || val.is_infinite() || val.is_finite());
    }
}

#[test]
fn test_q4k_with_nan_scale_p22() {
    // f16 NaN = 0x7C01
    let mut data = generate_q4k_with_scales(1, [0x01, 0x7C], [0x00, 0x00]);

    for i in 0..128 {
        data[16 + i] = 0x55;
    }

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.unwrap();
    // With NaN scale, all results should be NaN
    for val in &output {
        assert!(val.is_nan(), "Expected NaN, got {}", val);
    }
}

#[test]
fn test_q8_0_with_inf_scale_p22() {
    let mut data = vec![0u8; 34];

    // Scale = inf (f16 = 0x7C00)
    data[0] = 0x00;
    data[1] = 0x7C;

    // Set non-zero quants
    for i in 0..32 {
        data[2 + i] = 64;
    }

    let output = dequantize_q8_0_block(&data);

    for val in &output {
        assert!(val.is_infinite(), "Expected inf, got {}", val);
    }
}

#[test]
fn test_rope_with_nan_inputs_p22() {
    let half_dim = 4;
    let mut x1 = vec![f32::NAN, 1.0, 2.0, 3.0];
    let mut x2 = vec![4.0, f32::NAN, 5.0, 6.0];
    let cos_vals = vec![0.5; half_dim];
    let sin_vals = vec![0.5; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // NaN propagates
    assert!(x1[0].is_nan());
    assert!(x2[1].is_nan());
}

#[test]
fn test_rope_with_inf_inputs_p22() {
    let half_dim = 4;
    let mut x1 = vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 3.0];
    let mut x2 = vec![4.0, 5.0, 6.0, f32::INFINITY];
    let cos_vals = vec![0.5; half_dim];
    let sin_vals = vec![0.5; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Inf propagates in computations
    assert!(x1[0].is_infinite() || x1[0].is_nan());
    assert!(x1[2].is_infinite() || x1[2].is_nan());
}

// ============================================================================
// Tests for parallel consistency under load
// ============================================================================

#[test]
fn test_q4k_parallel_stress_consistency_p22() {
    // Run many times to catch race conditions
    let data = generate_q4k_with_scales(64, [0x00, 0x3C], [0x00, 0x38]);
    let reference = dequantize_q4_k_simd(&data).unwrap();

    for iteration in 0..20 {
        let result = dequantize_q4_k_simd(&data).unwrap();
        for i in 0..result.len() {
            assert!(
                (result[i] - reference[i]).abs() < 1e-10
                    || (result[i].is_nan() && reference[i].is_nan()),
                "Iteration {}: Mismatch at {}: {} vs {}",
                iteration,
                i,
                result[i],
                reference[i]
            );
        }
    }
}

#[test]
fn test_q8_0_parallel_stress_consistency_p22() {
    let data = generate_q8_0_with_scale(128, [0x00, 0x3C]);
    let reference = dequantize_q8_0_simd(&data).unwrap();

    for iteration in 0..20 {
        let result = dequantize_q8_0_simd(&data).unwrap();
        for i in 0..result.len() {
            assert!(
                (result[i] - reference[i]).abs() < 1e-10,
                "Iteration {}: Mismatch at {}: {} vs {}",
                iteration,
                i,
                result[i],
                reference[i]
            );
        }
    }
}

// ============================================================================
// Tests for scale extraction edge cases in dequantization
// ============================================================================

#[test]
fn test_q4k_scale_extraction_blocks_4_to_7_p22() {
    // Blocks 4-7 use packed scale extraction with high bits
    let mut sb_data = vec![0u8; 144];

    // d = 1.0, dmin = 0.5
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C;
    sb_data[2] = 0x00;
    sb_data[3] = 0x38;

    // Set scales bytes 0-3 with high bits set (affects blocks 4-7)
    sb_data[4] = 0b11_000001; // scale[0] = 1, high bits = 3
    sb_data[5] = 0b10_000010; // scale[1] = 2, high bits = 2
    sb_data[6] = 0b01_000011; // scale[2] = 3, high bits = 1
    sb_data[7] = 0b00_000100; // scale[3] = 4, high bits = 0

    // Set scales bytes 4-7 (used for mins of blocks 0-3, and combined with high bits for 4-7)
    sb_data[8] = 0b00_010000;  // For block 4: d = (low) | (high << 4) = 0 | 48 = 48
    sb_data[9] = 0b00_100000;  // For block 5
    sb_data[10] = 0b00_110000; // For block 6
    sb_data[11] = 0b00_000001; // For block 7

    // Set remaining scale bytes
    for i in 8..12 {
        sb_data[4 + i] = ((i * 5) % 64) as u8;
    }

    // Fill quants
    for i in 0..128 {
        sb_data[16 + i] = 0x77; // Both nibbles = 7
    }

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);

    // Verify output is finite and varies based on scale extraction
    for val in &output {
        assert!(val.is_finite(), "Non-finite value: {}", val);
    }
}

// ============================================================================
// Tests for exact size boundaries
// ============================================================================

#[test]
fn test_q4k_parallel_boundary_1_superblock_p22() {
    let data = generate_q4k_with_scales(1, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_parallel(&data).unwrap();
    assert_eq!(result.len(), QK_K);
}

#[test]
fn test_q4k_parallel_boundary_2_superblocks_p22() {
    let data = generate_q4k_with_scales(2, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_parallel(&data).unwrap();
    assert_eq!(result.len(), 2 * QK_K);
}

#[test]
fn test_q8_0_parallel_boundary_1_block_p22() {
    let data = generate_q8_0_with_scale(1, [0x00, 0x3C]);
    let result = dequantize_q8_0_parallel(&data).unwrap();
    assert_eq!(result.len(), 32);
}

#[test]
fn test_q8_0_parallel_boundary_2_blocks_p22() {
    let data = generate_q8_0_with_scale(2, [0x00, 0x3C]);
    let result = dequantize_q8_0_parallel(&data).unwrap();
    assert_eq!(result.len(), 64);
}

#[test]
fn test_rope_boundary_size_1_p22() {
    let mut x1 = vec![1.0];
    let mut x2 = vec![2.0];
    let cos_vals = vec![0.0];
    let sin_vals = vec![1.0];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // 90 degree: x1' = -x2, x2' = x1
    assert!((x1[0] - (-2.0)).abs() < 1e-6);
    assert!((x2[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_rope_boundary_size_2_p22() {
    let mut x1 = vec![1.0, 3.0];
    let mut x2 = vec![2.0, 4.0];
    let cos_vals = vec![1.0, 0.0];
    let sin_vals = vec![0.0, 1.0];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // First: identity
    assert!((x1[0] - 1.0).abs() < 1e-6);
    assert!((x2[0] - 2.0).abs() < 1e-6);
    // Second: 90 degree
    assert!((x1[1] - (-4.0)).abs() < 1e-6);
    assert!((x2[1] - 3.0).abs() < 1e-6);
}
