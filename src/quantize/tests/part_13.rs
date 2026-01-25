//! Part 13: Additional coverage for quantize/mod.rs uncovered paths
//!
//! Focus areas:
//! - InterleavedQ4K::dot with various input sizes (mod.rs impl)
//! - extract_scale_min and extract_scale_min_from_slice edge cases
//! - fused_q4_0_q8_0_dot_scalar with edge cases
//! - fused_q8_0_q8_0_dot_scalar with edge cases
//! - fused_q4_0_q8_0_parallel_matvec_into error paths
//! - fused_q8_0_q8_0_parallel_matvec_into error paths

use crate::quantize::{
    dequantize_q8_blocks, extract_scale_min, extract_scale_min_from_slice,
    fused_q4_0_q8_0_dot_scalar, fused_q4_0_q8_0_parallel_matvec,
    fused_q4_0_q8_0_parallel_matvec_into, fused_q8_0_q8_0_dot_scalar,
    fused_q8_0_q8_0_parallel_matvec, fused_q8_0_q8_0_parallel_matvec_into,
    quantize_activations_q8k_into, quantize_to_q8_blocks, InterleavedQ4K, Q8KSuperBlock, Q8_0Block,
};

// =============================================================================
// InterleavedQ4K::dot Tests - Exercising scalar fallback path (mod.rs impl)
// =============================================================================

#[test]
fn test_interleaved_q4k_mod_dot_single_superblock_zeros() {
    // 1 super-block with all zeros
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");

    let activations = vec![1.0f32; 256];
    let result = interleaved.dot(&activations).expect("dot should work");

    // All zeros in weights means result should be approximately 0
    assert!(result.is_finite(), "Result should be finite: {}", result);
}

#[test]
fn test_interleaved_q4k_mod_dot_single_superblock_ones() {
    // Create super-block with d=1.0, all scales=1, all qs=0x11
    let mut data = vec![0u8; 144];

    // d = 1.0 (f16: 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // dmin = 0.0
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

    // Set scale byte 0 to 1 (for first block scale)
    data[4] = 1;

    // Set qs to 0x11 (low=1, high=1)
    for i in 16..144 {
        data[i] = 0x11;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_mod_dot_two_superblocks() {
    // 2 super-blocks
    let mut data = vec![0u8; 288];

    // Super-block 0: d=1.0, dmin=0.0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

    // Super-block 1: d=2.0, dmin=0.0
    data[144..146].copy_from_slice(&0x4000u16.to_le_bytes());
    data[146..148].copy_from_slice(&0x0000u16.to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![0.5f32; 512];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_mod_dot_length_error() {
    let data = vec![0u8; 144]; // 256 values
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");

    // Wrong length
    let activations = vec![1.0f32; 128];
    let result = interleaved.dot(&activations);
    assert!(result.is_err(), "Should error on length mismatch");

    let activations = vec![1.0f32; 300];
    let result = interleaved.dot(&activations);
    assert!(result.is_err(), "Should error on length mismatch");
}

#[test]
fn test_interleaved_q4k_mod_dot_varied_activations() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0

    // Set scales
    for i in 4..16 {
        data[i] = 10;
    }

    // Set varied qs
    for i in 16..144 {
        data[i] = ((i - 16) % 16) as u8 | (((i - 16 + 1) % 16) << 4) as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");

    // Varied activations
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_mod_dot_negative_activations() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[4] = 32; // scale

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![-1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_mod_dot_with_dmin() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    data[2..4].copy_from_slice(&0x3C00u16.to_le_bytes()); // dmin=1.0 (nonzero!)

    // Set mins in scales (bytes 4-7 are for mins of first 4 blocks)
    data[4] = 0; // scale for block 0
    data[8] = 10; // min for block 0

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    // With dmin nonzero, we subtract dmin*m from the result
    assert!(result.is_finite());
}

// =============================================================================
// extract_scale_min Tests - All block indices
// =============================================================================

#[test]
fn test_extract_scale_min_mod_all_blocks() {
    let scales: [u8; 12] = [
        0b00_000001, // byte 0: scale0=1, high bits=0
        0b00_000010, // byte 1: scale1=2
        0b00_000011, // byte 2: scale2=3
        0b00_000100, // byte 3: scale3=4
        0b00_000101, // byte 4: min0=5
        0b00_000110, // byte 5: min1=6
        0b00_000111, // byte 6: min2=7
        0b00_001000, // byte 7: min3=8
        0b0010_0001, // byte 8: low=1 (scale4), high=2 (min4)
        0b0100_0011, // byte 9: low=3 (scale5), high=4 (min5)
        0b0110_0101, // byte 10: low=5 (scale6), high=6 (min6)
        0b1000_0111, // byte 11: low=7 (scale7), high=8 (min7)
    ];

    // Blocks 0-3: simple extraction
    let (s0, m0) = extract_scale_min(&scales, 0);
    assert_eq!(s0, 1.0);
    assert_eq!(m0, 5.0);

    let (s1, m1) = extract_scale_min(&scales, 1);
    assert_eq!(s1, 2.0);
    assert_eq!(m1, 6.0);

    let (s2, m2) = extract_scale_min(&scales, 2);
    assert_eq!(s2, 3.0);
    assert_eq!(m2, 7.0);

    let (s3, m3) = extract_scale_min(&scales, 3);
    assert_eq!(s3, 4.0);
    assert_eq!(m3, 8.0);

    // Blocks 4-7: packed extraction
    // Block 4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = 1 | (0 << 4) = 1
    //          m = (scales[8] >> 4) | ((scales[4] >> 6) << 4) = 2 | (0 << 4) = 2
    let (s4, m4) = extract_scale_min(&scales, 4);
    assert_eq!(s4, 1.0);
    assert_eq!(m4, 2.0);

    let (s5, m5) = extract_scale_min(&scales, 5);
    assert_eq!(s5, 3.0);
    assert_eq!(m5, 4.0);

    let (s6, m6) = extract_scale_min(&scales, 6);
    assert_eq!(s6, 5.0);
    assert_eq!(m6, 6.0);

    let (s7, m7) = extract_scale_min(&scales, 7);
    assert_eq!(s7, 7.0);
    assert_eq!(m7, 8.0);
}

#[test]
fn test_extract_scale_min_mod_high_bits_contribution() {
    // Test that high bits from bytes 0-3 contribute to blocks 4-7
    let mut scales: [u8; 12] = [0; 12];

    // Set high bits in byte 0 (contributes to scale4)
    scales[0] = 0b11_000000; // high bits = 3
                             // Set low bits in byte 8 (contributes to scale4)
    scales[8] = 0b0000_0001; // low = 1

    // Block 4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = 1 | (3 << 4) = 49
    let (s4, _) = extract_scale_min(&scales, 4);
    assert_eq!(s4, 49.0, "Scale4 with high bits contribution");
}

#[test]
fn test_extract_scale_min_from_slice_mod_all_indices() {
    let scales: [u8; 12] = [
        10, 20, 30, 40, // bytes 0-3
        5, 15, 25, 35, // bytes 4-7
        0, 0, 0, 0, // bytes 8-11
    ];

    // Even indices use simple extraction
    // idx=0: scale_idx=0, min_idx=4
    let (s0, m0) = extract_scale_min_from_slice(&scales, 0);
    assert_eq!(s0, 10.0);
    assert_eq!(m0, 5.0);

    // idx=2: scale_idx=1, min_idx=5
    let (s2, m2) = extract_scale_min_from_slice(&scales, 2);
    assert_eq!(s2, 20.0);
    assert_eq!(m2, 15.0);

    // idx=4: scale_idx=2, min_idx=6
    let (s4, m4) = extract_scale_min_from_slice(&scales, 4);
    assert_eq!(s4, 30.0);
    assert_eq!(m4, 25.0);

    // idx=6: scale_idx=3, min_idx=7
    let (s6, m6) = extract_scale_min_from_slice(&scales, 6);
    assert_eq!(s6, 40.0);
    assert_eq!(m6, 35.0);
}

#[test]
fn test_extract_scale_min_from_slice_mod_odd_indices_formula() {
    // For odd indices:
    // scale_idx = idx / 2
    // min_idx = scale_idx + 4
    // scale = (scales[scale_idx] >> 6) | ((scales[scale_idx + 2] & 0x0F) << 2)
    // min = (scales[min_idx] >> 6) | ((scales[min_idx + 2] & 0x0F) << 2)

    let mut scales: [u8; 12] = [0; 12];
    // For idx=1: scale_idx=0, uses bytes 0 and 2
    scales[0] = 0b11_000000; // high 2 bits = 3
    scales[2] = 0b0000_1111; // low 4 bits = 15
                             // scale = 3 | (15 << 2) = 3 + 60 = 63

    scales[4] = 0b10_000000; // high 2 bits = 2
    scales[6] = 0b0000_0101; // low 4 bits = 5
                             // min = 2 | (5 << 2) = 2 + 20 = 22

    let (s1, m1) = extract_scale_min_from_slice(&scales, 1);
    assert_eq!(s1, 63.0, "idx=1 scale");
    assert_eq!(m1, 22.0, "idx=1 min");
}

// =============================================================================
// fused_q4_0_q8_0_dot_scalar Edge Cases
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_empty() {
    let q4_data: Vec<u8> = vec![];
    let q8_scales: Vec<f32> = vec![];
    let q8_quants: Vec<i8> = vec![];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_partial_block() {
    // Q4_0 block is 18 bytes (2 scale + 16 quants)
    // If we only have 10 bytes, it's a partial block
    let mut q4_data = vec![0u8; 10];
    q4_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    // Should not panic, processes what's available
    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_two_complete_blocks() {
    // 2 Q4_0 blocks = 36 bytes
    let mut q4_data = vec![0u8; 36];

    // Block 0: scale = 1.0
    q4_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // quants centered at 8, so 0x88 gives q=0 after -8 offset
    for i in 2..18 {
        q4_data[i] = 0x88;
    }

    // Block 1: scale = 2.0
    q4_data[18..20].copy_from_slice(&0x4000u16.to_le_bytes());
    for i in 20..36 {
        q4_data[i] = 0x88;
    }

    let q8_scales = vec![1.0f32, 1.0f32];
    let q8_quants = vec![1i8; 64];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    // With centered quants (0x88 = q_val 0 after -8), result should be ~0
    assert!(result.abs() < 1.0, "Expected near 0, got {}", result);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_negative_quants() {
    let mut q4_data = vec![0u8; 18];
    q4_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
                                                             // Set q4 quants to 0x00 (low=0, high=0 -> values = -8 after offset)
    for i in 2..18 {
        q4_data[i] = 0x00;
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![10i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    // Each dequantized Q4 value is -8 (0 - 8 offset), scaled by 1.0
    assert!(result < 0.0, "Should be negative with 0x00 q4 quants");
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_large_scale() {
    let mut q4_data = vec![0u8; 18];
    // scale = 1000.0 in f16 (approximate: 0x63D0)
    q4_data[0..2].copy_from_slice(&0x63D0u16.to_le_bytes());
    for i in 2..18 {
        q4_data[i] = 0xFF; // max q4 values
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
}

// =============================================================================
// fused_q8_0_q8_0_dot_scalar Edge Cases
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_empty() {
    let q8_weight_data: Vec<u8> = vec![];
    let q8_act_scales: Vec<f32> = vec![];
    let q8_act_quants: Vec<i8> = vec![];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_partial_block() {
    // Q8_0 block is 34 bytes (2 scale + 32 quants)
    let mut q8_weight_data = vec![0u8; 20]; // partial
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_two_blocks() {
    // 2 Q8_0 blocks = 68 bytes
    let mut q8_weight_data = vec![0u8; 68];

    // Block 0: scale = 1.0, all quants = 5
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for i in 2..34 {
        q8_weight_data[i] = 5u8;
    }

    // Block 1: scale = 0.5, all quants = 10
    q8_weight_data[34..36].copy_from_slice(&0x3800u16.to_le_bytes());
    for i in 36..68 {
        q8_weight_data[i] = 10u8;
    }

    let q8_act_scales = vec![1.0f32, 1.0f32];
    let q8_act_quants = vec![2i8; 64];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 64);
    // Block 0: 1.0 * 1.0 * (5 * 2 * 32) = 320
    // Block 1: 0.5 * 1.0 * (10 * 2 * 32) = 320
    // Total: 640
    assert!(
        (result - 640.0).abs() < 10.0,
        "Expected ~640, got {}",
        result
    );
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_mixed_signs() {
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0

    // Mix of positive and negative quants
    for i in 2..18 {
        q8_weight_data[i] = 10u8; // positive
    }
    for i in 18..34 {
        q8_weight_data[i] = (-10i8) as u8; // negative
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![5i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // 16 positive: 10 * 5 * 16 = 800
    // 16 negative: -10 * 5 * 16 = -800
    // Total: 0
    assert!(
        result.abs() < 10.0,
        "Mixed signs should nearly cancel: {}",
        result
    );
}

// =============================================================================
// Parallel matvec error path tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_mod_weight_short() {
    // Weights too short for dimensions
    let in_dim = 64;
    let out_dim = 4;
    let bytes_per_row = 36; // 2 blocks per row
    let needed = out_dim * bytes_per_row; // 144 bytes
    let weight_data = vec![0u8; needed - 1]; // 1 byte short

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_mod_dim_mismatch() {
    let in_dim = 32;
    let bytes_per_row = 18;
    let weight_data = vec![0u8; 4 * bytes_per_row];
    let mut output = vec![0.0f32; 4];

    // Wrong activation length
    let activations = vec![1.0f32; 64];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_mod_weight_short() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;
    let needed = out_dim * bytes_per_row;
    let weight_data = vec![0u8; needed - 1];

    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_mod_dim_mismatch() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let mut output = vec![0.0f32; out_dim];

    // Wrong activation length
    let activations = vec![1.0f32; 64];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_mod_output_short() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    // Output too short
    let mut output = vec![0.0f32; 2];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}

// =============================================================================
// Sequential path tests (small dimensions)
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_mod_sequential_path() {
    // Small dimensions to trigger sequential path (out_dim < 512)
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 18;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    // Set scales to 1.0 for each row
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        weight_data[row_start..row_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // Set quants to centered (0x88)
        for i in 2..18 {
            weight_data[row_start + i] = 0x88;
        }
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_mod_parallel_path() {
    // Large dimensions to trigger parallel path (out_dim >= 512)
    let in_dim = 32;
    let out_dim = 600;
    let bytes_per_row = 18;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    // Set all scales to 1.0
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        weight_data[row_start..row_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

// =============================================================================
// Q8_0Block additional tests
// =============================================================================

#[test]
fn test_q8_0block_mod_quantize_extreme_values() {
    let mut values = [0.0f32; 32];
    values[0] = f32::MAX / 2.0;
    values[1] = -f32::MAX / 2.0;

    let block = Q8_0Block::quantize(&values);
    assert!(block.scale.is_finite());

    let dequant = block.dequantize();
    for v in &dequant {
        assert!(v.is_finite());
    }
}

#[test]
fn test_q8_0block_mod_roundtrip_small_values() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.001);
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();

    // Small values may lose precision
    for (&orig, &deq) in values.iter().zip(dequant.iter()) {
        assert!((orig - deq).abs() < 0.01, "orig={}, deq={}", orig, deq);
    }
}

// =============================================================================
// Q8KSuperBlock additional tests
// =============================================================================

#[test]
fn test_q8k_superblock_mod_quantize_alternating() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }

    let block = Q8KSuperBlock::quantize(&values);

    // Check that alternating pattern is preserved in sign
    for i in 0..256 {
        if i % 2 == 0 {
            assert!(block.quants[i] > 0, "Even index should be positive");
        } else {
            assert!(block.quants[i] < 0, "Odd index should be negative");
        }
    }
}

#[test]
fn test_q8k_superblock_mod_quantize_into_with_overflow_values() {
    // Values that would overflow if not clamped
    let values = [500.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    // All should clamp to 127
    for q in &quants {
        assert_eq!(*q, 127);
    }
}

// =============================================================================
// quantize_to_q8_blocks and dequantize_q8_blocks additional tests
// =============================================================================

#[test]
fn test_quantize_to_q8_blocks_mod_exact_multiple() {
    // Exactly 2 blocks = 64 values
    let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 16.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("valid");

    assert_eq!(blocks.len(), 2);

    // Dequantize and verify
    let dequant = dequantize_q8_blocks(&blocks);
    assert_eq!(dequant.len(), 64);

    for (&orig, &deq) in values.iter().zip(dequant.iter()) {
        assert!((orig - deq).abs() < 0.5, "orig={}, deq={}", orig, deq);
    }
}

#[test]
fn test_quantize_to_q8_blocks_mod_not_multiple() {
    // 50 values (not multiple of 32)
    let values = vec![1.0f32; 50];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_blocks_mod_preserves_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let blocks = vec![block];

    let dequant = dequantize_q8_blocks(&blocks);

    for v in &dequant {
        assert!(v.abs() < 1e-6, "Should be near zero: {}", v);
    }
}

// =============================================================================
// quantize_activations_q8k_into additional tests
// =============================================================================

#[test]
fn test_quantize_activations_q8k_into_mod_not_multiple() {
    let activations = vec![1.0f32; 300]; // Not multiple of 256
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 300];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_mod_scales_too_small() {
    let activations = vec![1.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Only 1, need 2
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_mod_quants_too_small() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 128]; // Only 128, need 256

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_mod_success() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants).expect("should work");

    assert!(scales[0] > 0.0);

    // Check that signs are preserved
    for i in 0..128 {
        assert!(quants[i] <= 0, "First half should be negative or zero");
    }
    for i in 128..256 {
        assert!(quants[i] >= 0, "Second half should be positive or zero");
    }
}
