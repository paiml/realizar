use super::*;

// ============================================================================
// FUSED Q4_K DOT PRODUCT TESTS (SCALAR)
// ============================================================================

#[test]
fn test_fused_q4k_dot_invalid_data_length() {
    // Q4_K super-block is 144 bytes
    let data = vec![0u8; 100]; // Not a multiple of 144
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not a multiple"));
}

#[test]
fn test_fused_q4k_dot_activation_length_mismatch() {
    // One super-block = 144 bytes = 256 values
    let data = vec![0u8; 144];
    let activations = vec![0.0f32; 128]; // Should be 256

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("doesn't match"));
}

#[test]
fn test_fused_q4k_dot_zero_data() {
    // All zeros should produce zero dot product
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations).unwrap();
    // d = 0, so all values dequantize to 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4k_dot_single_super_block() {
    // Create valid Q4_K data: d (2) + dmin (2) + scales (12) + qs (128)
    let mut data = vec![0u8; 144];

    // Set d = 1.0 (f16: 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // Set dmin = 0.5 (f16: 0x3800)
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set scales (12 bytes packed)
    for i in 0..12 {
        data[4 + i] = 0x11;
    }

    // Set qs values to a pattern
    for i in 0..128 {
        data[16 + i] = 0x55; // Pattern: low=5, high=5
    }

    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_multiple_super_blocks() {
    // Two super-blocks = 288 bytes = 512 values
    let mut data = vec![0u8; 288];

    // Set d = 0.5 for both blocks
    data[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
    data[144..146].copy_from_slice(&0x3800u16.to_le_bytes());

    let activations = vec![0.5f32; 512];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_empty_data() {
    let data = vec![];
    let activations = vec![];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0.0);
}

// ============================================================================
// FUSED Q4_K DOT PRODUCT TESTS (SIMD)
// ============================================================================

#[test]
fn test_fused_q4k_dot_simd_invalid_input() {
    let data = vec![0u8; 100]; // Invalid length
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_matches_scalar() {
    // Create valid Q4_K data
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0.5
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set scales
    for i in 0..12 {
        data[4 + i] = 0x11;
    }

    // Set qs to deterministic pattern
    for i in 0..128 {
        data[16 + i] = ((i * 3) % 256) as u8;
    }

    let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

    let scalar_result = fused_q4k_dot(&data, &activations).unwrap();
    let simd_result = fused_q4k_dot_simd(&data, &activations).unwrap();

    // Allow small tolerance for SIMD vs scalar (FMA ordering differences)
    let rel_err = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        rel_err < 0.001,
        "scalar={} simd={} rel_err={}",
        scalar_result,
        simd_result,
        rel_err
    );
}

#[test]
fn test_fused_q4k_dot_simd_zero_activations() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot_simd(&data, &activations).unwrap();
    // Product of anything with zero should be zero
    assert_eq!(result, 0.0);
}

// ============================================================================
// FUSED Q4_K × Q8_K DOT PRODUCT TESTS
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_invalid_data_length() {
    let data = vec![0u8; 100]; // Not a multiple of 144
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_scales_too_small() {
    let data = vec![0u8; 144]; // 1 super-block
    let q8k_scales = vec![]; // Should be >= 1
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_quants_too_small() {
    let data = vec![0u8; 144]; // 1 super-block = 256 values
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![10i8; 128]; // Should be 256

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_zero_data() {
    let data = vec![0u8; 144];
    let q8k_scales = vec![0.0f32; 1]; // Zero scale
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).unwrap();
    // With zero Q4_K d and zero Q8_K scale, result should be 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_basic() {
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0.5
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set scales
    for i in 0..12 {
        data[4 + i] = 0x11;
    }

    // Set qs
    for i in 0..128 {
        data[16 + i] = 0x55;
    }

    let q8k_scales = vec![0.1f32; 1];
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_invalid_input() {
    let data = vec![0u8; 100]; // Invalid
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_matches_scalar() {
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0.5
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set scales
    for i in 0..12 {
        data[4 + i] = 0x11;
    }

    // Set qs to pattern
    for i in 0..128 {
        data[16 + i] = ((i * 7) % 256) as u8;
    }

    let q8k_scales = vec![0.1f32; 1];
    let q8k_quants: Vec<i8> = (0..256).map(|i| ((i % 64) - 32) as i8).collect();

    let scalar_result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).unwrap();
    let simd_result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).unwrap();

    // Allow tolerance for SIMD vs scalar
    let rel_err = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        rel_err < 0.02,
        "scalar={} simd={} rel_err={}",
        scalar_result,
        simd_result,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_multiple_super_blocks() {
    // Two super-blocks = 288 bytes = 512 values
    let mut data = vec![0u8; 288];

    // Set d for both blocks
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[144..146].copy_from_slice(&0x3C00u16.to_le_bytes());

    let q8k_scales = vec![0.1f32; 2];
    let q8k_quants = vec![5i8; 512];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_fused_q4k_dot_max_nibble_values() {
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // Set qs to max nibble (0xFF = 15 low, 15 high)
    for i in 0..128 {
        data[16 + i] = 0xFF;
    }

    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_negative_quants() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    let q8k_scales = vec![0.1f32; 1];
    let q8k_quants = vec![-10i8; 256]; // All negative

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_simd_large_input() {
    // 8 super-blocks = 1152 bytes = 2048 values
    let mut data = vec![0u8; 1152];

    for sb in 0..8 {
        let offset = sb * 144;
        // Set d = 0.1
        data[offset..offset + 2].copy_from_slice(&0x2E66u16.to_le_bytes());
    }

    let activations = vec![0.5f32; 2048];

    let scalar_result = fused_q4k_dot(&data, &activations).unwrap();
    let simd_result = fused_q4k_dot_simd(&data, &activations).unwrap();

    let rel_err = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        rel_err < 0.001,
        "scalar={} simd={} rel_err={}",
        scalar_result,
        simd_result,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_mixed_signs() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set qs to alternating pattern
    for i in 0..128 {
        data[16 + i] = if i % 2 == 0 { 0x0F } else { 0xF0 };
    }

    let q8k_scales = vec![0.1f32; 1];
    // Alternating positive and negative quants
    let q8k_quants: Vec<i8> = (0..256)
        .map(|i| if i % 2 == 0 { 10 } else { -10 })
        .collect();

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_scale_extraction() {
    // Test that scales are correctly extracted and applied
    let mut data = vec![0u8; 144];

    // Set d = 2.0 (f16: 0x4000)
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());

    // Set dmin = 0 (no offset)
    data[2..4].copy_from_slice(&[0x00, 0x00]);

    // Set first scale to max (63) - packed format
    data[4] = 0x3F; // First scale = 63

    // Set qs to 1 (low nibble only)
    for i in 0..128 {
        data[16 + i] = 0x01;
    }

    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
    // Result should be non-zero since d > 0, scale > 0, qs > 0
    assert!(result.unwrap().abs() > 0.0);
}

// ============================================================================
// T-COV-95: ADDITIONAL COVERAGE FOR fused_k.rs
// ============================================================================

// --- fused_q4k_dot: known-value computation verification ---

#[test]
fn test_fused_q4k_dot_known_value_single_block_uniform() {
    // Build a Q4_K super-block with known properties:
    // d = 1.0, dmin = 0, all scales = 1, all mins = 0, all qs = 0x33 (low=3, high=3)
    let mut data = vec![0u8; 144];

    // d = 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // dmin = 0.0
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    // scales[0..3] = 1 (first 4 blocks use simple layout: scale = scales[j] & 63)
    data[4] = 1;
    data[5] = 1;
    data[6] = 1;
    data[7] = 1;
    // scales[4..7] = 0 (mins for first 4 blocks: scales[j+4] & 63)
    // scales[8..11] = 0 (packed upper bits)

    // Set all qs to 0x33: low nibble = 3, high nibble = 3
    for i in 0..128 {
        data[16 + i] = 0x33;
    }

    // Activations: all 1.0
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Manually compute:
    // 4 chunks of 64 values, each chunk has 32 low nibbles + 32 high nibbles
    // For blocks 0-3 (is=0..7, but first 4 have scale=1):
    //   Chunk 0 (j=0): is=0 -> sc1=scales[0]&63=1, m1=scales[4]&63=0
    //                   is=1 -> sc2=scales[1]&63=1, m2=scales[5]&63=0
    //     d1 = 1.0 * 1 = 1.0, dm1 = 0
    //     d2 = 1.0 * 1 = 1.0, dm2 = 0
    //     Low nibbles (32 values): q_val=3, value=1.0*3-0=3.0, sum=32*3.0=96.0
    //     High nibbles (32 values): q_val=3, value=1.0*3-0=3.0, sum=32*3.0=96.0
    //   Chunk 1 (j=64): is=2 -> sc1=scales[2]&63=1, sc2=scales[3]&63=1
    //     Same pattern: 96.0 + 96.0 = 192.0
    //   Chunk 2 (j=128): is=4 -> packed layout, scales byte at data[12]=0, data[0]>>6=0
    //     sc1 = (scales[8]&0x0F)|((scales[0]>>6)<<4) = 0
    //     So all values are 0 for blocks 4-7
    //   Chunk 3 (j=192): is=6 -> same, packed = 0
    //
    // Total = 96 + 96 + 96 + 96 + 0 + 0 + 0 + 0 = 384.0
    assert!(
        (result - 384.0).abs() < 1.0,
        "Expected about 384.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_dot_known_value_with_dmin() {
    // Test that dmin is correctly subtracted
    let mut data = vec![0u8; 144];

    // d = 1.0, dmin = 0.5
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // scales[0] = 2 (scale for block 0), scales[4] = 3 (min for block 0)
    data[4] = 2;
    data[8] = 3;
    // All other scales/mins = 0

    // qs all zeros: low nibble = 0, high nibble = 0
    // (all qs are already 0)

    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Chunk 0 (j=0): is=0 -> sc1=scales[0]&63=2, m1=scales[4]&63=3
    //                 is=1 -> sc2=scales[1]&63=0, m2=scales[5]&63=0
    //   d1=1.0*2=2.0, dm1=0.5*3=1.5
    //   Low nibbles (32 values): val = 2.0*0 - 1.5 = -1.5, sum = 32 * (-1.5) = -48.0
    //   d2=1.0*0=0, dm2=0.5*0=0
    //   High nibbles (32 values): val = 0*0 - 0 = 0, sum = 0
    // Chunks 1-3: all scales=0, so vals = 0 - 0 = 0
    //
    // Total = -48.0
    assert!(
        (result - (-48.0)).abs() < 0.1,
        "Expected about -48.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_dot_varied_nibble_pattern() {
    // Verify correct nibble extraction: qs byte 0xAB = low=0xB=11, high=0xA=10
    let mut data = vec![0u8; 144];

    // d = 1.0, dmin = 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // scale for block 0 = 1
    data[4] = 1;

    // Set first qs byte = 0xAB, rest = 0
    data[16] = 0xAB;

    // Activations: all 1.0
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Chunk 0, block 0 (sc1=1, m1=0):
    //   Low nibble byte[0] = 0xB = 11, val = 1.0*1*11 - 0 = 11.0
    //   Low nibble bytes[1..31] = 0, val = 0
    //   -> Low nibble sum = 11.0
    //
    // Chunk 0, block 1 (sc2=scales[1]&63=0):
    //   High nibble byte[0] = 0xA = 10, val = 1.0*0*10 - 0 = 0
    //   -> High nibble sum = 0
    //
    // All other chunks: scales = 0, so 0
    //
    // Total = 11.0
    assert!(
        (result - 11.0).abs() < 0.01,
        "Expected 11.0, got {}",
        result
    );
}

// --- fused_q4k_q8k_dot: known-value computation verification ---

#[test]
fn test_fused_q4k_q8k_dot_known_value_simple() {
    // Build a Q4_K block with known properties and Q8_K activations
    let mut data = vec![0u8; 144];

    // d = 1.0, dmin = 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

    // scale for block 0 = 1
    data[4] = 1;

    // All qs = 0x11 (low=1, high=1)
    for i in 0..128 {
        data[16 + i] = 0x11;
    }

    // Q8_K: scale = 1.0, all quants = 2
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![2i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");

    // Chunk 0 (j=0): sc1=1, m1=0 -> d_sc1_q8=1.0*1*1.0=1.0, dm1_q8=0
    //   sc2=0 (scales[1]=0) -> d_sc2_q8=0, dm2_q8=0
    //   32 bytes: q4_lo = 1, q8_lo = 2 -> sum_lo = 32 * (1*2) = 64
    //   q8_sum_lo = 32 * 2 = 64
    //   result += 1.0 * 64 - 0 * 64 = 64.0
    //   High nibble: q4_hi = 1, but sc2=0 so no contribution
    //   result += 0 * ... - 0 * ... = 0
    // Chunks 1-3: scales=0 so result += 0 * ... = 0
    // Total = 64.0
    assert!(
        (result - 64.0).abs() < 1.0,
        "Expected about 64.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_known_value_with_dmin() {
    let mut data = vec![0u8; 144];

    // d = 2.0, dmin = 1.0
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());

    // scale[0] = 3, min[0] = 2
    data[4] = 3; // scales[0] & 63 = 3
    data[8] = 2; // scales[4] & 63 = 2

    // All qs = 0 (low=0, high=0)

    // Q8_K: scale = 0.5, quants all = 10
    let q8k_scales = vec![0.5f32];
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");

    // Chunk 0 (j=0): sc1=3, m1=2
    //   d_sc1_q8 = 2.0 * 3 * 0.5 = 3.0
    //   dm1_q8 = 1.0 * 2 * 0.5 = 1.0
    //   Low nibbles: q4_lo=0 for all 32 -> sum_lo=0
    //   q8_sum_lo = 32 * 10 = 320
    //   contrib1 = 3.0 * 0 - 1.0 * 320 = -320.0
    //
    //   sc2=scales[1]&63=0, m2=scales[5]&63=0
    //   contrib2 = 0 - 0 = 0
    //
    // Other chunks: all zero scales/mins
    // Total = -320.0
    assert!(
        (result - (-320.0)).abs() < 1.0,
        "Expected about -320.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_two_super_blocks_known() {
    // 2 super-blocks: first with d=1.0, second with d=0.5
    let mut data = vec![0u8; 288];

    // Block 0: d=1.0, dmin=0, scale[0]=1, qs all = 0x22 (lo=2, hi=2)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[4] = 1;
    for i in 0..128 {
        data[16 + i] = 0x22;
    }

    // Block 1: d=0.5, dmin=0, scale[0]=1, qs all = 0x33 (lo=3, hi=3)
    data[144..146].copy_from_slice(&0x3800u16.to_le_bytes());
    data[148] = 1;
    for i in 0..128 {
        data[160 + i] = 0x33;
    }

    let q8k_scales = vec![1.0f32; 2];
    let q8k_quants = vec![1i8; 512];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");

    // Block 0, Chunk 0: sc1=1, m1=0
    //   sum_lo = 32 * (2*1) = 64
    //   contrib = 1.0*1*1.0 * 64 = 64.0
    //   High nibble: sc2=0 -> 0
    // Block 0 total = 64.0
    //
    // Block 1, Chunk 0: sc1=1, m1=0
    //   sum_lo = 32 * (3*1) = 96
    //   contrib = 0.5*1*1.0 * 96 = 48.0
    // Block 1 total = 48.0
    //
    // Grand total = 64.0 + 48.0 = 112.0
    assert!(
        (result - 112.0).abs() < 1.0,
        "Expected about 112.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_negative_q8_quants() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    data[4] = 1; // scale=1

    // All qs = 0x11 (lo=1, hi=1)
    for i in 0..128 {
        data[16 + i] = 0x11;
    }

    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![-5i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("should succeed");

    // Chunk 0: sc1=1, m1=0
    //   sum_lo = 32 * (1 * (-5)) = -160
    //   q8_sum_lo = 32 * (-5) = -160
    //   contrib = 1.0 * (-160) - 0 = -160.0
    // Total = -160.0
    assert!(
        (result - (-160.0)).abs() < 1.0,
        "Expected about -160.0, got {}",
        result
    );
}

// --- SIMD vs scalar parity with known values ---

#[test]
fn test_fused_q4k_dot_simd_vs_scalar_varied_activations() {
    // Use varied activations and qs patterns to stress-test parity
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin=0.5

    // Set varied scales
    data[4] = 5;
    data[5] = 10;
    data[6] = 15;
    data[7] = 20;
    data[8] = 3;
    data[9] = 7;
    data[10] = 11;
    data[11] = 14;

    // Varied qs pattern
    for i in 0..128 {
        data[16 + i] = ((i * 13 + 7) % 256) as u8;
    }

    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

    let scalar_result = fused_q4k_dot(&data, &activations).expect("scalar should succeed");
    let simd_result = fused_q4k_dot_simd(&data, &activations).expect("simd should succeed");

    let rel_err = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        rel_err < 0.01,
        "SIMD/scalar parity failed: scalar={}, simd={}, rel_err={}",
        scalar_result,
        simd_result,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_simd_vs_scalar_varied() {
    // Varied data to exercise all code paths
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin=0.5

    // Set all scales to small known values
    for i in 0..12 {
        data[4 + i] = ((i + 1) * 3) as u8;
    }

    // Varied qs
    for i in 0..128 {
        data[16 + i] = ((i * 7 + 11) % 256) as u8;
    }

    let q8k_scales = vec![0.3f32];
    let q8k_quants: Vec<i8> = (0..256).map(|i| (i % 100) as i8 - 50).collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let rel_err = if scalar.abs() > 1e-6 {
        (simd - scalar).abs() / scalar.abs()
    } else {
        (simd - scalar).abs()
    };
    assert!(
        rel_err < 0.02,
        "Q4K×Q8K SIMD/scalar parity: scalar={}, simd={}, rel_err={}",
        scalar,
        simd,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_simd_multiple_super_blocks_parity() {
    // 4 super-blocks to exercise prefetch paths and multi-block accumulation
    let mut data = vec![0u8; 576]; // 4 * 144

    for sb in 0..4 {
        let offset = sb * 144;
        // d = 0.5
        data[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
        // dmin = 0.25
        data[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
        // Set varied scales
        for i in 0..12 {
            data[offset + 4 + i] = ((sb * 5 + i * 3 + 1) % 63) as u8;
        }
        // Set varied qs
        for i in 0..128 {
            data[offset + 16 + i] = ((sb * 17 + i * 11 + 3) % 256) as u8;
        }
    }

    let q8k_scales = vec![0.2f32; 4];
    let q8k_quants: Vec<i8> = (0..1024)
        .map(|i| (((i * 3 + 7) % 200) as i32 - 100) as i8)
        .collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let rel_err = if scalar.abs() > 1e-6 {
        (simd - scalar).abs() / scalar.abs()
    } else {
        (simd - scalar).abs()
    };
    assert!(
        rel_err < 0.02,
        "4-superblock parity: scalar={}, simd={}, rel_err={}",
        scalar,
        simd,
        rel_err
    );
}

// --- fused_q4k_dot: additional error path tests ---

#[test]
fn test_fused_q4k_dot_error_message_contains_details() {
    let data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];
    let err = fused_q4k_dot(&data, &activations).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("100"),
        "Error should contain data length: {}",
        msg
    );
    assert!(
        msg.contains("144"),
        "Error should contain block size: {}",
        msg
    );
}

#[test]
fn test_fused_q4k_dot_activation_error_message() {
    let data = vec![0u8; 144];
    let activations = vec![0.0f32; 100]; // Wrong
    let err = fused_q4k_dot(&data, &activations).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("100"),
        "Error should contain act length: {}",
        msg
    );
    assert!(
        msg.contains("256"),
        "Error should contain expected count: {}",
        msg
    );
}

#[test]
fn test_fused_q4k_q8k_dot_error_messages() {
    // Test each error path explicitly
    let data = vec![0u8; 100];
    let err = fused_q4k_q8k_dot(&data, &[1.0], &[1; 256]).unwrap_err();
    assert!(err.to_string().contains("not a multiple"));

    let data = vec![0u8; 144];
    let err = fused_q4k_q8k_dot(&data, &[], &[1; 256]).unwrap_err();
    assert!(err.to_string().contains("scales"));

    let err = fused_q4k_q8k_dot(&data, &[1.0], &[1; 100]).unwrap_err();
    assert!(err.to_string().contains("quants"));
}

// --- fused_q4k_q8k_dot_simd error paths ---

#[test]
fn test_fused_q4k_q8k_dot_simd_error_paths() {
    // Invalid data length
    let err = fused_q4k_q8k_dot_simd(&[0u8; 100], &[1.0], &[1i8; 256]).unwrap_err();
    assert!(err.to_string().contains("not a multiple"));
}

#[test]
fn test_fused_q4k_dot_simd_error_paths() {
    // Activation length mismatch via simd path
    let data = vec![0u8; 144];
    let activations = vec![0.0f32; 100];
    let err = fused_q4k_dot_simd(&data, &activations).unwrap_err();
    assert!(err.to_string().contains("doesn't match"));
}

// --- fused_q4k_dot: packed scale blocks (blocks 4-7) ---

#[test]
fn test_fused_q4k_dot_packed_scale_blocks() {
    // Exercise blocks 4-7 which use packed scale layout
    let mut data = vec![0u8; 144];

    // d = 1.0, dmin = 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // For block 4 (is=4): packed layout
    // scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    // min = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    //
    // Set scales[0] = 0b11_000000 (high 2 bits = 3, low 6 bits = 0)
    // Set scales[8] = 0b0010_0101 (high 4 = 2, low 4 = 5)
    // scale = 5 | (3 << 4) = 5 | 48 = 53
    // min = 2 | (0 << 4) = 2
    data[4] = 0b1100_0000; // scales[0]
    data[12] = 0b0010_0101; // scales[8]

    // Set qs for chunk 2 (j=128, which reads qs[64..96])
    // These are for blocks 4 and 5
    for i in 64..96 {
        data[16 + i] = 0x22; // low=2, high=2
    }

    let activations = vec![1.0f32; 256];
    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Chunk 2 (j=128):
    //   is=4: sc1 = (scales[8]&0x0F)|((scales[0]>>6)<<4) = 5 | 48 = 53
    //         m1  = (scales[8]>>4)|((scales[4]>>6)<<4)    = 2 | 0 = 2
    //   is=5: sc2 = (scales[9]&0x0F)|((scales[1]>>6)<<4) = 0
    //
    //   d1 = 1.0 * 53 = 53.0, dm1 = 0 * 2 = 0 (dmin=0)
    //   Low nibbles (32): val = 53.0 * 2 - 0 = 106.0, sum = 32 * 106 = 3392.0
    //   High nibbles: sc2=0 so 0
    //
    // Total = 3392.0
    assert!(
        (result - 3392.0).abs() < 1.0,
        "Expected about 3392.0, got {}",
        result
    );
}

// --- Symmetry and sign tests ---

#[test]
fn test_fused_q4k_dot_sign_reversal() {
    // If we negate all activations, result should negate
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[4] = 1;
    for i in 0..128 {
        data[16 + i] = 0x55;
    }

    let pos_act = vec![1.0f32; 256];
    let neg_act = vec![-1.0f32; 256];

    let pos_result = fused_q4k_dot(&data, &pos_act).expect("pos");
    let neg_result = fused_q4k_dot(&data, &neg_act).expect("neg");

    assert!(
        (pos_result + neg_result).abs() < 0.01,
        "Negating activations should negate result: {} vs {}",
        pos_result,
        neg_result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_sign_reversal() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[4] = 1;
    for i in 0..128 {
        data[16 + i] = 0x55;
    }

    let q8k_scales = vec![1.0f32];
    let pos_quants = vec![10i8; 256];
    let neg_quants = vec![-10i8; 256];

    let pos_result = fused_q4k_q8k_dot(&data, &q8k_scales, &pos_quants).expect("pos");
    let neg_result = fused_q4k_q8k_dot(&data, &q8k_scales, &neg_quants).expect("neg");

    // With dmin=0, negating quants should negate result
    // (When dmin != 0, the min correction also inverts, so this holds)
    assert!(
        (pos_result + neg_result).abs() < 1.0,
        "Negating quants should negate result: {} vs {}",
        pos_result,
        neg_result
    );
}

// --- Large multi-block SIMD parity ---

#[test]
fn test_fused_q4k_dot_simd_16_super_blocks() {
    // 16 super-blocks = 2304 bytes = 4096 values
    let mut data = vec![0u8; 16 * 144];

    for sb in 0..16 {
        let offset = sb * 144;
        // d = 0.1 (f16 ~ 0x2E66)
        data[offset..offset + 2].copy_from_slice(&0x2E66u16.to_le_bytes());
        data[offset + 2..offset + 4].copy_from_slice(&0x2800u16.to_le_bytes());

        // Varied scales
        for i in 0..12 {
            data[offset + 4 + i] = ((sb + i * 5 + 1) % 63) as u8;
        }

        // Varied qs
        for i in 0..128 {
            data[offset + 16 + i] = ((sb * 37 + i * 23 + 5) % 256) as u8;
        }
    }

    let activations: Vec<f32> = (0..4096)
        .map(|i| ((i * 7 + 3) % 200) as f32 * 0.005 - 0.5)
        .collect();

    let scalar = fused_q4k_dot(&data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&data, &activations).expect("simd");

    let rel_err = if scalar.abs() > 1e-6 {
        (simd - scalar).abs() / scalar.abs()
    } else {
        (simd - scalar).abs()
    };
    assert!(
        rel_err < 0.01,
        "16-superblock parity: scalar={}, simd={}, rel_err={}",
        scalar,
        simd,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_empty() {
    let result = fused_q4k_q8k_dot(&[], &[], &[]).expect("empty should work");
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_simd_empty() {
    let result = fused_q4k_q8k_dot_simd(&[], &[], &[]).expect("empty should work");
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4k_dot_simd_empty() {
    let result = fused_q4k_dot_simd(&[], &[]).expect("empty should work");
    assert_eq!(result, 0.0);
}

// --- dequantize_q4_k consistency check ---

/// PMAT-170: Q4K Layout Consistency Test
///
/// Verifies that apr::dequantize_q4_k produces the same element ordering
/// as fused_q4k_parallel_matvec. This was the root cause of GPU explosion bug #170.
#[test]
fn test_q4k_layout_consistency_pmat170() {
    use crate::apr::dequantize_q4_k;
    use crate::quantize::fused_q4k_parallel_matvec;

    // Use 256x256 test matrix (1 super-block per row)
    let in_dim = 256;
    let out_dim = 256;
    let num_elements = in_dim * out_dim;

    // Create reproducible Q4K test data (144 bytes per row)
    let bytes_per_row = 144;
    let total_bytes = out_dim * bytes_per_row;
    let q4k_bytes: Vec<u8> = (0..total_bytes)
        .map(|i| ((i * 17 + 37) % 256) as u8)
        .collect();

    // Method 1: Direct dequantization
    let dequant = dequantize_q4_k(&q4k_bytes, num_elements);

    // Method 2: Extract columns via fused matmul with basis vectors
    let mut fused_matrix = vec![0.0f32; num_elements];
    for col in 0..in_dim {
        // Basis vector: e_col = [0, ..., 0, 1, 0, ..., 0]
        let mut basis = vec![0.0f32; in_dim];
        basis[col] = 1.0;

        // fused_q4k_parallel_matvec produces W @ basis = column col of W
        if let Ok(column) = fused_q4k_parallel_matvec(&q4k_bytes, &basis, in_dim, out_dim) {
            for row in 0..out_dim {
                fused_matrix[row * in_dim + col] = column[row];
            }
        }
    }

    // Compare element by element
    let mut mismatches = 0;
    let mut max_diff = 0.0f32;

    for i in 0..num_elements {
        let diff = (dequant[i] - fused_matrix[i]).abs();
        if diff > 1e-5 {
            mismatches += 1;
            max_diff = max_diff.max(diff);
        }
    }

    assert_eq!(
        mismatches, 0,
        "Q4K layout mismatch: {} elements differ (max diff: {}). \
             This indicates dequantize_q4_k has different element ordering \
             than fused_q4k_parallel_matvec, which would cause GPU explosion.",
        mismatches, max_diff
    );
}

// ============================================================================
// FUSED Q4_K × Q8_K DOT PRODUCT — AVX2 COVERAGE TESTS
// ============================================================================
// These tests call the unsafe fused_q4k_q8k_dot_avx2 directly to cover
// the AVX2 code path (which is unreachable through the public API on
// machines with AVX-512 VNNI).

/// Build valid Q4K super-block data for testing.
/// Each super-block: [d:f16(2), dmin:f16(2), scales:12, quants:128] = 144 bytes
fn build_q4k_test_block(d: f32, dmin: f32, nibble_val: u8) -> [u8; 144] {
    let mut block = [0u8; 144];
    // d as f16
    let d_bits = half::f16::from_f32(d).to_bits();
    block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    // dmin as f16
    let dmin_bits = half::f16::from_f32(dmin).to_bits();
    block[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
    // scales: set all to give scale=1, min=0 (6-bit encoded)
    // For extract_scale_min, lower 4 bits = scale, upper 2 bits = min
    for i in 0..12 {
        block[4 + i] = 0x01; // scale=1, min=0 in packed format
    }
    // quants: 128 bytes, each byte has lo and hi nibble
    let packed = (nibble_val & 0x0F) | ((nibble_val & 0x0F) << 4);
    for i in 0..128 {
        block[16 + i] = packed;
    }
    block
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return; // Skip on non-AVX2 hardware
    }

    // Build 1 super-block
    let block = build_q4k_test_block(1.0, 0.0, 3);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - avx2).abs();
    assert!(
        diff < 1.0,
        "scalar={scalar} vs avx2={avx2}, diff={diff} exceeds tolerance"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_zero_quants() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 0);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![0i8; 256];

    let result = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();
    assert!(
        result.abs() < 1e-6,
        "zero × zero should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_multi_superblock() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 4 super-blocks
    let block = build_q4k_test_block(1.0, 0.0, 5);
    let mut q4k_data = Vec::with_capacity(144 * 4);
    for _ in 0..4 {
        q4k_data.extend_from_slice(&block);
    }
    let q8k_scales = vec![1.0f32; 4];
    let q8k_quants = vec![2i8; 256 * 4];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - avx2).abs();
    // Allow larger tolerance for multi-block accumulation
    let rel_tolerance = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < rel_tolerance,
        "4-block: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_negative_quants() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 7);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![-3i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - avx2).abs();
    let rel_tolerance = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < rel_tolerance,
        "neg quants: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_with_dmin() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // Non-zero dmin affects the min-subtraction path
    let block = build_q4k_test_block(1.0, 0.5, 4);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![2.0f32];
    let q8k_quants = vec![5i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - avx2).abs();
    let rel_tolerance = scalar.abs().max(1.0) * 0.05;
    assert!(
        diff < rel_tolerance,
        "dmin: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_invalid_data_length() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let q4k_data = vec![0u8; 100]; // Not a multiple of 144
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let result = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err(), "should fail for non-144-aligned data");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_buffer_too_small() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 1);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 128]; // Too small (need 256)

    let result = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err(), "should fail for too-small Q8K buffer");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_varying_scales() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 2 super-blocks with different Q8K scales
    let block = build_q4k_test_block(1.0, 0.0, 8);
    let mut q4k_data = Vec::with_capacity(144 * 2);
    q4k_data.extend_from_slice(&block);
    q4k_data.extend_from_slice(&block);
    let q8k_scales = vec![0.5f32, 2.0f32];
    let q8k_quants = vec![3i8; 256 * 2];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - avx2).abs();
    let rel_tolerance = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < rel_tolerance,
        "varying scales: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

// ============================================================================
// FUSED Q4_K × Q8_K DOT PRODUCT — AVX-512 VNNI COVERAGE TESTS
// ============================================================================
// These tests call the unsafe fused_q4k_q8k_dot_avx512vnni and
// fused_q4k_q8k_dot_avx512vnni_opt directly to cover both AVX-512 code paths.

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 3);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let vnni =
        unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - vnni).abs();
    assert!(
        diff < 1.0,
        "scalar={scalar} vs avx512vnni={vnni}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_zero_quants() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 0);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![0i8; 256];

    let result =
        unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();
    assert!(
        result.abs() < 1e-6,
        "zero × zero should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_multi_superblock() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 5);
    let mut q4k_data = Vec::with_capacity(144 * 3);
    for _ in 0..3 {
        q4k_data.extend_from_slice(&block);
    }
    let q8k_scales = vec![1.0f32; 3];
    let q8k_quants = vec![2i8; 256 * 3];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let vnni =
        unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - vnni).abs();
    let tol = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < tol,
        "3-block: scalar={scalar} vs avx512vnni={vnni}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_invalid_data_length() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let q4k_data = vec![0u8; 100]; // Not a multiple of 144
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let result = unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err(), "should fail for invalid data length");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_buffer_too_small() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 1);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 128]; // Too small

    let result = unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err(), "should fail for too-small Q8K buffer");
}

// --- fused_q4k_q8k_dot_avx512vnni_opt tests ---

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 3);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let vnni_opt =
        unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - vnni_opt).abs();
    assert!(
        diff < 1.0,
        "scalar={scalar} vs avx512vnni_opt={vnni_opt}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_zero() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 0);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![0i8; 256];

    let result =
        unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();
    assert!(
        result.abs() < 1e-6,
        "zero × zero should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_multi_superblock() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 8);
    let mut q4k_data = Vec::with_capacity(144 * 4);
    for _ in 0..4 {
        q4k_data.extend_from_slice(&block);
    }
    let q8k_scales = vec![0.5f32, 1.0, 2.0, 0.25];
    let q8k_quants = vec![3i8; 256 * 4];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let vnni_opt =
        unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - vnni_opt).abs();
    let tol = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < tol,
        "4-block varying: scalar={scalar} vs vnni_opt={vnni_opt}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_invalid_data_length() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let q4k_data = vec![0u8; 100];
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let result = unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_buffer_too_small() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 1);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 128]; // Too small

    let result = unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_negative_quants() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(2.0, 0.5, 7);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![-3i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).unwrap();
    let vnni_opt =
        unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) }.unwrap();

    let diff = (scalar - vnni_opt).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "negative quants: scalar={scalar} vs vnni_opt={vnni_opt}, diff={diff}"
    );
}

// --- fused_q4k_dot_avx512_vnni tests (Q4K × f32 activations) ---

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_dot_exercises_code_path() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 5);
    let q4k_data = block.to_vec();
    let activations = vec![1.0f32; 256];

    // AVX512 VNNI Q4K dot uses internal int8 quantization of activations,
    // so results may differ from the AVX2 public path. Just verify it runs
    // and produces a finite value.
    let result = unsafe { fused_q4k_dot_avx512_vnni(&q4k_data, &activations) }.unwrap();
    assert!(
        result.is_finite(),
        "should produce finite result, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_dot_zero_activations() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 8);
    let q4k_data = block.to_vec();
    let activations = vec![0.0f32; 256];

    let result = unsafe { fused_q4k_dot_avx512_vnni(&q4k_data, &activations) }.unwrap();
    assert!(
        result.abs() < 1e-3,
        "zero activations should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_dot_invalid_data_length() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let q4k_data = vec![0u8; 100]; // Not multiple of 144
    let activations = vec![1.0f32; 256];

    let result = unsafe { fused_q4k_dot_avx512_vnni(&q4k_data, &activations) };
    assert!(result.is_err());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_dot_multi_superblock() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(0.5, 0.1, 6);
    let mut q4k_data = Vec::with_capacity(144 * 2);
    q4k_data.extend_from_slice(&block);
    q4k_data.extend_from_slice(&block);
    let activations: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();

    // Exercises multi-block AVX512 VNNI path with varied activations
    let result = unsafe { fused_q4k_dot_avx512_vnni(&q4k_data, &activations) }.unwrap();
    assert!(
        result.is_finite(),
        "should produce finite result, got {result}"
    );
}
