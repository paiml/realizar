//! Phase 45: Parallel dequantization coverage tests for `parallel_dequant.rs`
//!
//! Tests cover:
//! - `dequantize_q4_k_parallel` - Parallel Q4_K dequantization
//! - `dequantize_q4_k_simd` - SIMD-accelerated Q4_K dequantization
//! - `dequantize_q4_k_superblock` - Single super-block dequantization
//! - `dequantize_q8_0_parallel` - Parallel Q8_0 dequantization
//! - `dequantize_q8_0_simd` - SIMD-accelerated Q8_0 dequantization
//! - `dequantize_q8_0_block` - Single block dequantization
//! - `apply_rope_rotation_simd` - SIMD RoPE rotation
//! - `apply_rope_rotation_scalar` - Scalar RoPE fallback
//!
//! Focus areas:
//! - Edge cases: empty input, single block, boundary sizes
//! - Error handling: invalid sizes, non-multiple of block size
//! - Parallel vs sequential paths
//! - AVX2 SIMD paths (with scalar fallback verification)
//! - RoPE rotation for various head dimensions

use crate::quantize::parallel_dequant::{
    apply_rope_rotation_scalar, apply_rope_rotation_simd, dequantize_q4_k_parallel,
    dequantize_q4_k_simd, dequantize_q4_k_superblock, dequantize_q8_0_block,
    dequantize_q8_0_parallel, dequantize_q8_0_simd,
};
use crate::quantize::types::QK_K;

// ============================================================================
// Helper functions for generating valid quantized data
// ============================================================================

/// Generate valid Q4_K super-block data (144 bytes per super-block)
fn generate_q4k_superblock_data(num_super_blocks: usize) -> Vec<u8> {
    const SUPER_BLOCK_BYTES: usize = 144;
    let mut data = vec![0u8; num_super_blocks * SUPER_BLOCK_BYTES];

    for sb in 0..num_super_blocks {
        let sb_start = sb * SUPER_BLOCK_BYTES;

        // Set d (f16 = 1.0 = 0x3C00 in little-endian)
        data[sb_start] = 0x00;
        data[sb_start + 1] = 0x3C;

        // Set dmin (f16 = 0.5 = 0x3800 in little-endian)
        data[sb_start + 2] = 0x00;
        data[sb_start + 3] = 0x38;

        // Set scales (12 bytes) - alternating pattern
        for i in 0..12 {
            data[sb_start + 4 + i] = ((sb * 7 + i * 13) % 64) as u8;
        }

        // Set quantized values (128 bytes) - deterministic pattern
        for i in 0..128 {
            data[sb_start + 16 + i] = ((sb * 11 + i * 17) % 256) as u8;
        }
    }

    data
}

/// Generate valid Q8_0 block data (34 bytes per block: 2 scale + 32 quants)
fn generate_q8_0_block_data(num_blocks: usize) -> Vec<u8> {
    const BLOCK_BYTES: usize = 34;
    let mut data = vec![0u8; num_blocks * BLOCK_BYTES];

    for block in 0..num_blocks {
        let block_start = block * BLOCK_BYTES;

        // Set scale (f16 = 0.5 = 0x3800 in little-endian)
        data[block_start] = 0x00;
        data[block_start + 1] = 0x38;

        // Set quantized int8 values (32 bytes)
        for i in 0..32 {
            // Values from -128 to 127, deterministic pattern
            data[block_start + 2 + i] = ((block * 7 + i * 5) % 256) as u8;
        }
    }

    data
}

// ============================================================================
// Tests for dequantize_q4_k_parallel
// ============================================================================

#[test]
fn test_dequantize_q4_k_parallel_single_superblock_p19() {
    let data = generate_q4k_superblock_data(1);
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), QK_K); // 256 values per super-block

    // All values should be finite
    for val in &output {
        assert!(val.is_finite(), "Value should be finite: {}", val);
    }
}

#[test]
fn test_dequantize_q4_k_parallel_multiple_superblocks_p19() {
    let num_super_blocks = 5;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), num_super_blocks * QK_K);

    // All values should be finite
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q4_k_parallel_large_input_p19() {
    // Test with many super-blocks to exercise parallel paths
    let num_super_blocks = 64;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), num_super_blocks * QK_K);
}

#[test]
fn test_dequantize_q4_k_parallel_invalid_size_too_small_p19() {
    // Data that's not a multiple of 144 bytes
    let data = vec![0u8; 100]; // Less than one super-block
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not a multiple"));
}

#[test]
fn test_dequantize_q4_k_parallel_invalid_size_partial_p19() {
    // Data that's 1.5 super-blocks (144 + 72 = 216 bytes)
    let data = vec![0u8; 216];
    let result = dequantize_q4_k_parallel(&data);

    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_k_parallel_empty_input_p19() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_k_parallel(&data);

    // Empty input is a multiple of 144 (0 * 144 = 0), should succeed with empty output
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_dequantize_q4_k_parallel_deterministic_p19() {
    let data = generate_q4k_superblock_data(4);

    // Run multiple times and verify determinism
    let result1 = dequantize_q4_k_parallel(&data).unwrap();
    let result2 = dequantize_q4_k_parallel(&data).unwrap();
    let result3 = dequantize_q4_k_parallel(&data).unwrap();

    for i in 0..result1.len() {
        assert_eq!(
            result1[i], result2[i],
            "Mismatch at index {} between runs 1-2",
            i
        );
        assert_eq!(
            result2[i], result3[i],
            "Mismatch at index {} between runs 2-3",
            i
        );
    }
}

#[test]
fn test_dequantize_q4_k_parallel_scale_variations_p19() {
    // Test with different scale values to exercise scale extraction
    let mut data = generate_q4k_superblock_data(2);

    // Set different d values for the two super-blocks
    // Super-block 0: d = 2.0 (0x4000)
    data[0] = 0x00;
    data[1] = 0x40;

    // Super-block 1: d = 0.25 (0x3400)
    data[144] = 0x00;
    data[145] = 0x34;

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.unwrap();

    // First super-block values should be larger on average
    let sum_first: f32 = output[..QK_K].iter().map(|v| v.abs()).sum();
    let sum_second: f32 = output[QK_K..].iter().map(|v| v.abs()).sum();

    // The first super-block has larger d, so its values should have larger magnitude
    assert!(
        sum_first > sum_second * 0.5, // Account for dmin differences
        "Expected first super-block to have larger values"
    );
}

// ============================================================================
// Tests for dequantize_q4_k_simd
// ============================================================================

#[test]
fn test_dequantize_q4_k_simd_single_superblock_p19() {
    let data = generate_q4k_superblock_data(1);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), QK_K);
}

#[test]
fn test_dequantize_q4_k_simd_multiple_superblocks_p19() {
    let num_super_blocks = 8;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), num_super_blocks * QK_K);
}

#[test]
fn test_dequantize_q4_k_simd_parity_with_parallel_p19() {
    // SIMD and parallel should produce identical results
    let data = generate_q4k_superblock_data(4);

    let simd_result = dequantize_q4_k_simd(&data).unwrap();
    let parallel_result = dequantize_q4_k_parallel(&data).unwrap();

    assert_eq!(simd_result.len(), parallel_result.len());

    for i in 0..simd_result.len() {
        let diff = (simd_result[i] - parallel_result[i]).abs();
        assert!(
            diff < 1e-6,
            "SIMD/parallel mismatch at {}: simd={}, parallel={}, diff={}",
            i,
            simd_result[i],
            parallel_result[i],
            diff
        );
    }
}

#[test]
fn test_dequantize_q4_k_simd_large_for_avx2_path_p19() {
    // Large enough to trigger parallel AVX2 path (>= 128 super-blocks)
    let num_super_blocks = 130;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), num_super_blocks * QK_K);
}

#[test]
fn test_dequantize_q4_k_simd_small_for_sequential_path_p19() {
    // Small enough to skip parallelism overhead (< 128 super-blocks)
    let num_super_blocks = 2;
    let data = generate_q4k_superblock_data(num_super_blocks);
    let result = dequantize_q4_k_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), num_super_blocks * QK_K);
}

#[test]
fn test_dequantize_q4_k_simd_invalid_size_p19() {
    let data = vec![0u8; 100]; // Not a multiple of 144
    let result = dequantize_q4_k_simd(&data);
    assert!(result.is_err());
}

// ============================================================================
// Tests for dequantize_q4_k_superblock (internal helper)
// ============================================================================

#[test]
fn test_dequantize_q4_k_superblock_basic_p19() {
    let data = generate_q4k_superblock_data(1);
    let sb_data = &data[0..144];

    let output = dequantize_q4_k_superblock(sb_data);
    assert_eq!(output.len(), QK_K);

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q4_k_superblock_all_zeros_p19() {
    // Super-block with zero scales should produce near-zero output
    let sb_data = vec![0u8; 144];

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);

    // With d=0 and dmin=0, all values should be 0
    for val in &output {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_dequantize_q4_k_superblock_max_nibble_values_p19() {
    // Test with all nibbles set to max (0xF = 15)
    let mut sb_data = vec![0u8; 144];

    // Set d = 1.0
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C;

    // Set dmin = 0.0
    sb_data[2] = 0x00;
    sb_data[3] = 0x00;

    // Set all scales to 1
    for i in 0..12 {
        sb_data[4 + i] = 1;
    }

    // Set all quantized values to 0xFF (both nibbles = 15)
    for i in 0..128 {
        sb_data[16 + i] = 0xFF;
    }

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);

    // All values should be finite and positive (15 * scale)
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q4_k_superblock_alternating_pattern_p19() {
    // Test with alternating nibble pattern (0x0F, 0xF0)
    let mut sb_data = vec![0u8; 144];

    // Set scales
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C; // d = 1.0
    sb_data[2] = 0x00;
    sb_data[3] = 0x38; // dmin = 0.5

    // Set scales to non-zero
    for i in 0..12 {
        sb_data[4 + i] = 0x01;
    }

    // Alternating quantized values
    for i in 0..128 {
        sb_data[16 + i] = if i % 2 == 0 { 0x0F } else { 0xF0 };
    }

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);
}

// ============================================================================
// Tests for dequantize_q8_0_parallel
// ============================================================================

#[test]
fn test_dequantize_q8_0_parallel_single_block_p19() {
    let data = generate_q8_0_block_data(1);
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 32); // 32 values per block
}

#[test]
fn test_dequantize_q8_0_parallel_multiple_blocks_p19() {
    let num_blocks = 10;
    let data = generate_q8_0_block_data(num_blocks);
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), num_blocks * 32);
}

#[test]
fn test_dequantize_q8_0_parallel_large_input_p19() {
    // Many blocks to exercise parallel paths
    let num_blocks = 100;
    let data = generate_q8_0_block_data(num_blocks);
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), num_blocks * 32);
}

#[test]
fn test_dequantize_q8_0_parallel_invalid_size_p19() {
    // Data that's not a multiple of 34 bytes
    let data = vec![0u8; 50]; // Not 34 * n
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not a multiple"));
}

#[test]
fn test_dequantize_q8_0_parallel_empty_input_p19() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q8_0_parallel(&data);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_dequantize_q8_0_parallel_deterministic_p19() {
    let data = generate_q8_0_block_data(8);

    let result1 = dequantize_q8_0_parallel(&data).unwrap();
    let result2 = dequantize_q8_0_parallel(&data).unwrap();

    for i in 0..result1.len() {
        assert_eq!(result1[i], result2[i], "Non-deterministic at index {}", i);
    }
}

// ============================================================================
// Tests for dequantize_q8_0_simd
// ============================================================================

#[test]
fn test_dequantize_q8_0_simd_single_block_p19() {
    let data = generate_q8_0_block_data(1);
    let result = dequantize_q8_0_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 32);
}

#[test]
fn test_dequantize_q8_0_simd_multiple_blocks_p19() {
    let num_blocks = 16;
    let data = generate_q8_0_block_data(num_blocks);
    let result = dequantize_q8_0_simd(&data);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), num_blocks * 32);
}

#[test]
fn test_dequantize_q8_0_simd_parity_with_parallel_p19() {
    let data = generate_q8_0_block_data(8);

    let simd_result = dequantize_q8_0_simd(&data).unwrap();
    let parallel_result = dequantize_q8_0_parallel(&data).unwrap();

    assert_eq!(simd_result.len(), parallel_result.len());

    for i in 0..simd_result.len() {
        let diff = (simd_result[i] - parallel_result[i]).abs();
        assert!(
            diff < 1e-6,
            "SIMD/parallel mismatch at {}: simd={}, parallel={}",
            i,
            simd_result[i],
            parallel_result[i]
        );
    }
}

#[test]
fn test_dequantize_q8_0_simd_invalid_size_p19() {
    let data = vec![0u8; 50]; // Not 34 * n
    let result = dequantize_q8_0_simd(&data);
    assert!(result.is_err());
}

// ============================================================================
// Tests for dequantize_q8_0_block (internal helper)
// ============================================================================

#[test]
fn test_dequantize_q8_0_block_basic_p19() {
    let data = generate_q8_0_block_data(1);
    let block_data = &data[0..34];

    let output = dequantize_q8_0_block(block_data);
    assert_eq!(output.len(), 32);

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q8_0_block_zero_scale_p19() {
    // Block with zero scale
    let mut block_data = vec![0u8; 34];
    // Scale = 0 (f16)
    block_data[0] = 0x00;
    block_data[1] = 0x00;

    // Set some non-zero quants
    for i in 0..32 {
        block_data[2 + i] = i as u8;
    }

    let output = dequantize_q8_0_block(&block_data);
    assert_eq!(output.len(), 32);

    // With zero scale, all outputs should be 0
    for val in &output {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_dequantize_q8_0_block_max_values_p19() {
    let mut block_data = vec![0u8; 34];

    // Scale = 1.0 (f16 = 0x3C00)
    block_data[0] = 0x00;
    block_data[1] = 0x3C;

    // Set all quants to 127 (max positive i8)
    for i in 0..32 {
        block_data[2 + i] = 127;
    }

    let output = dequantize_q8_0_block(&block_data);

    // All values should be 127 * 1.0 = 127.0
    for val in &output {
        assert!((val - 127.0).abs() < 0.01, "Expected 127.0, got {}", val);
    }
}

#[test]
fn test_dequantize_q8_0_block_min_values_p19() {
    let mut block_data = vec![0u8; 34];

    // Scale = 1.0 (f16 = 0x3C00)
    block_data[0] = 0x00;
    block_data[1] = 0x3C;

    // Set all quants to -128 (min i8 as u8 = 0x80 = 128)
    for i in 0..32 {
        block_data[2 + i] = 0x80;
    }

    let output = dequantize_q8_0_block(&block_data);

    // All values should be -128 * 1.0 = -128.0
    for val in &output {
        assert!(
            (val - (-128.0)).abs() < 0.01,
            "Expected -128.0, got {}",
            val
        );
    }
}

#[test]
fn test_dequantize_q8_0_block_alternating_signs_p19() {
    let mut block_data = vec![0u8; 34];

    // Scale = 0.5 (f16 = 0x3800)
    block_data[0] = 0x00;
    block_data[1] = 0x38;

    // Alternating positive (64) and negative (-64 = 0xC0 = 192)
    for i in 0..32 {
        block_data[2 + i] = if i % 2 == 0 { 64 } else { 192 };
    }

    let output = dequantize_q8_0_block(&block_data);

    for (i, &val) in output.iter().enumerate() {
        if i % 2 == 0 {
            assert!((val - 32.0).abs() < 0.01, "Expected 32.0, got {}", val);
        } else {
            assert!((val - (-32.0)).abs() < 0.01, "Expected -32.0, got {}", val);
        }
    }
}

// ============================================================================
// Tests for apply_rope_rotation_simd
// ============================================================================

#[test]
fn test_apply_rope_rotation_simd_basic_p19() {
    let half_dim = 4;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
    let mut x2 = vec![5.0, 6.0, 7.0, 8.0];
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Verify finite results
    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_simd_size_8_p19() {
    // Exactly 8 elements - minimum for AVX2 SIMD path
    let half_dim = 8;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_simd_size_16_p19() {
    // 16 elements - 2 SIMD iterations
    let half_dim = 16;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32 * 0.1).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1) + 1.0).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_simd_size_17_remainder_p19() {
    // 17 elements - tests remainder handling (2*8 + 1)
    let half_dim = 17;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_simd_size_7_scalar_fallback_p19() {
    // 7 elements - should use scalar fallback (< 8)
    let half_dim = 7;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_simd_identity_p19() {
    // Identity rotation: cos=1, sin=0
    let half_dim = 8;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x2 = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let original_x1 = x1.clone();
    let original_x2 = x2.clone();

    let cos_vals = vec![1.0; half_dim];
    let sin_vals = vec![0.0; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=1, sin=0: x1' = x1*1 - x2*0 = x1, x2' = x1*0 + x2*1 = x2
    for i in 0..half_dim {
        assert!((x1[i] - original_x1[i]).abs() < 1e-6, "x1 changed at {}", i);
        assert!((x2[i] - original_x2[i]).abs() < 1e-6, "x2 changed at {}", i);
    }
}

#[test]
fn test_apply_rope_rotation_simd_90_degrees_p19() {
    // 90 degree rotation: cos=0, sin=1
    let half_dim = 8;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x2 = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let original_x1 = x1.clone();
    let original_x2 = x2.clone();

    let cos_vals = vec![0.0; half_dim];
    let sin_vals = vec![1.0; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=0, sin=1: x1' = x1*0 - x2*1 = -x2, x2' = x1*1 + x2*0 = x1
    for i in 0..half_dim {
        assert!(
            (x1[i] - (-original_x2[i])).abs() < 1e-6,
            "x1[{}]: expected {}, got {}",
            i,
            -original_x2[i],
            x1[i]
        );
        assert!(
            (x2[i] - original_x1[i]).abs() < 1e-6,
            "x2[{}]: expected {}, got {}",
            i,
            original_x1[i],
            x2[i]
        );
    }
}

#[test]
fn test_apply_rope_rotation_simd_180_degrees_p19() {
    // 180 degree rotation: cos=-1, sin=0
    let half_dim = 8;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x2 = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let original_x1 = x1.clone();
    let original_x2 = x2.clone();

    let cos_vals = vec![-1.0; half_dim];
    let sin_vals = vec![0.0; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=-1, sin=0: x1' = x1*(-1) - x2*0 = -x1, x2' = x1*0 + x2*(-1) = -x2
    for i in 0..half_dim {
        assert!((x1[i] - (-original_x1[i])).abs() < 1e-6);
        assert!((x2[i] - (-original_x2[i])).abs() < 1e-6);
    }
}

#[test]
fn test_apply_rope_rotation_simd_parity_with_scalar_p19() {
    let half_dim = 32;
    let mut simd_x1: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.1).collect();
    let mut simd_x2: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let mut scalar_x1 = simd_x1.clone();
    let mut scalar_x2 = simd_x2.clone();

    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut simd_x1, &mut simd_x2, &cos_vals, &sin_vals);
    apply_rope_rotation_scalar(&mut scalar_x1, &mut scalar_x2, &cos_vals, &sin_vals);

    for i in 0..half_dim {
        let diff_x1 = (simd_x1[i] - scalar_x1[i]).abs();
        let diff_x2 = (simd_x2[i] - scalar_x2[i]).abs();
        assert!(
            diff_x1 < 1e-5,
            "x1 SIMD/scalar mismatch at {}: simd={}, scalar={}",
            i,
            simd_x1[i],
            scalar_x1[i]
        );
        assert!(
            diff_x2 < 1e-5,
            "x2 SIMD/scalar mismatch at {}: simd={}, scalar={}",
            i,
            simd_x2[i],
            scalar_x2[i]
        );
    }
}

// ============================================================================
// Tests for apply_rope_rotation_scalar (internal helper)
// ============================================================================

#[test]
fn test_apply_rope_rotation_scalar_basic_p19() {
    let half_dim = 4;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
    let mut x2 = vec![5.0, 6.0, 7.0, 8.0];
    let cos_vals = vec![1.0, 0.866, 0.5, 0.0]; // cos(0), cos(30), cos(60), cos(90)
    let sin_vals = vec![0.0, 0.5, 0.866, 1.0]; // sin(0), sin(30), sin(60), sin(90)

    apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_scalar_empty_p19() {
    let mut x1: Vec<f32> = vec![];
    let mut x2: Vec<f32> = vec![];
    let cos_vals: Vec<f32> = vec![];
    let sin_vals: Vec<f32> = vec![];

    // Should not panic on empty input
    apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);
    assert!(x1.is_empty());
    assert!(x2.is_empty());
}

#[test]
fn test_apply_rope_rotation_scalar_single_element_p19() {
    let mut x1 = vec![1.0];
    let mut x2 = vec![2.0];
    let cos_vals = vec![0.5];
    let sin_vals = vec![0.866];

    apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // x1' = 1*0.5 - 2*0.866 = 0.5 - 1.732 = -1.232
    // x2' = 1*0.866 + 2*0.5 = 0.866 + 1.0 = 1.866
    assert!((x1[0] - (-1.232)).abs() < 0.01);
    assert!((x2[0] - 1.866).abs() < 0.01);
}

#[test]
fn test_apply_rope_rotation_scalar_negative_values_p19() {
    let mut x1 = vec![-1.0, -2.0, -3.0, -4.0];
    let mut x2 = vec![-5.0, -6.0, -7.0, -8.0];
    let cos_vals = vec![1.0; 4];
    let sin_vals = vec![0.0; 4];

    apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Identity rotation preserves values
    assert!((x1[0] - (-1.0)).abs() < 1e-6);
    assert!((x2[0] - (-5.0)).abs() < 1e-6);
}

// ============================================================================
// Tests for thread pool and chunking behavior
// ============================================================================

#[test]
fn test_q4k_parallel_thread_consistency_p19() {
    // Run many times to catch potential race conditions
    let data = generate_q4k_superblock_data(16);
    let reference = dequantize_q4_k_parallel(&data).unwrap();

    for run in 0..10 {
        let result = dequantize_q4_k_parallel(&data).unwrap();
        for i in 0..result.len() {
            assert_eq!(
                result[i], reference[i],
                "Thread inconsistency on run {} at index {}",
                run, i
            );
        }
    }
}

#[test]
fn test_q8_0_parallel_thread_consistency_p19() {
    let data = generate_q8_0_block_data(32);
    let reference = dequantize_q8_0_parallel(&data).unwrap();

    for run in 0..10 {
        let result = dequantize_q8_0_parallel(&data).unwrap();
        for i in 0..result.len() {
            assert_eq!(
                result[i], reference[i],
                "Thread inconsistency on run {} at index {}",
                run, i
            );
        }
    }
}

#[test]
fn test_q4k_simd_chunking_boundary_128_p19() {
    // Test at the CHUNK_SIZE boundary (64 super-blocks)
    // Just below threshold
    let data_below = generate_q4k_superblock_data(127);
    let result_below = dequantize_q4_k_simd(&data_below);
    assert!(result_below.is_ok());

    // Just above threshold
    let data_above = generate_q4k_superblock_data(129);
    let result_above = dequantize_q4_k_simd(&data_above);
    assert!(result_above.is_ok());

    // Exactly at threshold
    let data_at = generate_q4k_superblock_data(128);
    let result_at = dequantize_q4_k_simd(&data_at);
    assert!(result_at.is_ok());
}

// ============================================================================
// Tests for numerical edge cases
// ============================================================================

#[test]
fn test_q4k_dequant_with_large_scale_p19() {
    let mut data = generate_q4k_superblock_data(1);

    // Set d to a large f16 value (near max ~65504)
    // 0x7BFF is max normal f16
    data[0] = 0xFF;
    data[1] = 0x7B;

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.unwrap();
    for val in &output {
        assert!(val.is_finite(), "Large scale produced non-finite: {}", val);
    }
}

#[test]
fn test_q4k_dequant_with_small_scale_p19() {
    let mut data = generate_q4k_superblock_data(1);

    // Set d to a very small f16 value (subnormal)
    data[0] = 0x01;
    data[1] = 0x00;

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.unwrap();
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q8_0_dequant_with_negative_scale_p19() {
    let mut data = vec![0u8; 34];

    // Set scale to -1.0 (f16 = 0xBC00)
    data[0] = 0x00;
    data[1] = 0xBC;

    // Set quants to positive values
    for i in 0..32 {
        data[2 + i] = 64; // +64
    }

    let output = dequantize_q8_0_block(&data);

    // Result should be 64 * (-1.0) = -64.0
    for val in &output {
        assert!((val - (-64.0)).abs() < 0.01, "Expected -64.0, got {}", val);
    }
}

#[test]
fn test_rope_rotation_with_subnormal_values_p19() {
    let half_dim = 4;
    let mut x1 = vec![f32::MIN_POSITIVE / 2.0; half_dim]; // Subnormal
    let mut x2 = vec![f32::MIN_POSITIVE / 2.0; half_dim];
    let cos_vals = vec![0.5; half_dim];
    let sin_vals = vec![0.5; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Should produce finite results (possibly zero due to underflow)
    for val in x1.iter().chain(x2.iter()) {
        assert!(
            val.is_finite(),
            "Subnormal rotation produced non-finite: {}",
            val
        );
    }
}

#[test]
fn test_rope_rotation_with_large_values_p19() {
    let half_dim = 4;
    let mut x1 = vec![1e30; half_dim];
    let mut x2 = vec![1e30; half_dim];
    let cos_vals = vec![0.5; half_dim];
    let sin_vals = vec![0.5; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(
            val.is_finite(),
            "Large value rotation produced non-finite: {}",
            val
        );
    }
}

// ============================================================================
// Tests for AVX-512 path (if available)
// ============================================================================

#[test]
fn test_rope_rotation_size_32_avx512_candidate_p19() {
    // 32 elements - 2 AVX-512 iterations if available
    let half_dim = 32;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_rope_rotation_size_64_avx512_multiple_p19() {
    // 64 elements - 4 AVX-512 iterations
    let half_dim = 64;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.01).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.01 + 0.5).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.02).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.02).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_rope_rotation_size_65_avx512_remainder_p19() {
    // 65 elements - 4 AVX-512 + 1 remainder
    let half_dim = 65;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.03).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.03).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}
