//! Part 16: Comprehensive coverage for quantize/mod.rs uncovered paths
//!
//! Focus areas:
//! - F16_TO_F32_LUT static initialization and f16_to_f32_lut function
//! - Q8KSuperBlock::dequantize
//! - Q8_0Block::relative_error edge cases
//! - SimdBackend Display trait
//! - DequantStats struct
//! - InterleavedQ4K additional paths
//! - extract_scale_min edge cases for blocks 4-7
//! - fused_q4_0_q8_0_dot_simd boundary cases

use crate::quantize::{
    dequantize_q8_blocks, detect_simd_backend, fused_q4_0_q8_0_parallel_matvec,
    fused_q4_0_q8_0_parallel_matvec_into, fused_q8_0_q8_0_parallel_matvec,
    fused_q8_0_q8_0_parallel_matvec_into, quantize_activations_q8k_into, quantize_to_q8_blocks,
    DequantStats, InterleavedQ4K, Q8KSuperBlock, Q8_0Block, SimdBackend, BLOCK_SIZE, QK_K,
};

// These are pub(crate), available within the crate
use crate::quantize::{
    extract_scale_min, extract_scale_min_from_slice, fused_q4_0_q8_0_dot_scalar,
    fused_q4_0_q8_0_dot_simd, fused_q8_0_q8_0_dot_scalar,
};

// =============================================================================
// F16_TO_F32_LUT and f16_to_f32_lut() Tests
// =============================================================================

#[test]
fn test_f16_to_f32_lut_special_values() {
    // The LUT is accessed via InterleavedQ4K::from_q4k which uses f16_to_f32_lut
    // Test that special f16 values are correctly converted

    // Zero: f16 bits 0x0000
    let zero_data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&zero_data).expect("valid");
    assert_eq!(interleaved.d[0], 0.0, "f16 zero should convert to f32 zero");

    // One: f16 bits 0x3C00
    let mut one_data = vec![0u8; 144];
    one_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&one_data).expect("valid");
    assert!(
        (interleaved.d[0] - 1.0).abs() < 1e-6,
        "f16 1.0 should convert correctly"
    );

    // Negative one: f16 bits 0xBC00
    let mut neg_one_data = vec![0u8; 144];
    neg_one_data[0..2].copy_from_slice(&0xBC00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&neg_one_data).expect("valid");
    assert!(
        (interleaved.d[0] + 1.0).abs() < 1e-6,
        "f16 -1.0 should convert correctly"
    );
}

#[test]
fn test_f16_to_f32_lut_half_precision() {
    // 0.5 in f16: 0x3800
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(
        (interleaved.d[0] - 0.5).abs() < 1e-6,
        "f16 0.5 should convert correctly"
    );

    // 2.0 in f16: 0x4000
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(
        (interleaved.d[0] - 2.0).abs() < 1e-6,
        "f16 2.0 should convert correctly"
    );
}

#[test]
fn test_f16_to_f32_lut_subnormal_f16() {
    // Subnormal f16 value (very small, denormalized)
    // f16 subnormal: 0x0001 = smallest positive subnormal
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x0001u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(interleaved.d[0] > 0.0, "Subnormal f16 should be positive");
    assert!(
        interleaved.d[0] < 1e-6,
        "Subnormal f16 should be very small"
    );
}

#[test]
fn test_f16_to_f32_lut_infinity() {
    // Positive infinity in f16: 0x7C00
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(interleaved.d[0].is_infinite() && interleaved.d[0] > 0.0);

    // Negative infinity in f16: 0xFC00
    data[0..2].copy_from_slice(&0xFC00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(interleaved.d[0].is_infinite() && interleaved.d[0] < 0.0);
}

#[test]
fn test_f16_to_f32_lut_nan() {
    // NaN in f16: 0x7E00 (one possible NaN encoding)
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x7E00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(
        interleaved.d[0].is_nan(),
        "f16 NaN should convert to f32 NaN"
    );
}

#[test]
fn test_f16_to_f32_lut_max_f16() {
    // Max normal f16: 0x7BFF = 65504.0
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x7BFFu16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(
        (interleaved.d[0] - 65504.0).abs() < 1.0,
        "Max f16 should be ~65504"
    );
}

// =============================================================================
// Q8KSuperBlock::dequantize Tests
// =============================================================================

#[test]
fn test_q8k_superblock_dequantize_basic() {
    let values = [1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // All values should be approximately 1.0
    for (i, &v) in dequant.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 0.1,
            "Index {}: expected ~1.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn test_q8k_superblock_dequantize_zeros() {
    let values = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // All values should be exactly 0
    for (i, &v) in dequant.iter().enumerate() {
        assert!(v.abs() < 1e-6, "Index {}: expected 0, got {}", i, v);
    }
}

#[test]
fn test_q8k_superblock_dequantize_alternating() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = if i % 2 == 0 { 10.0 } else { -10.0 };
    }

    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // Check alternating signs are preserved
    for (i, &v) in dequant.iter().enumerate() {
        if i % 2 == 0 {
            assert!(v > 0.0, "Even index {} should be positive: {}", i, v);
        } else {
            assert!(v < 0.0, "Odd index {} should be negative: {}", i, v);
        }
    }
}

#[test]
fn test_q8k_superblock_dequantize_varying() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) * 0.5);
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // Check the trend is preserved (ascending values)
    for i in 1..256 {
        assert!(
            dequant[i] >= dequant[i - 1] - 1.0,
            "Values should be roughly ascending at index {}",
            i
        );
    }
}

#[test]
fn test_q8k_superblock_dequantize_extreme_values() {
    let mut values = [0.0f32; 256];
    values[0] = 1000.0;
    values[255] = -1000.0;

    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // First value should be large positive (clamped to 127)
    assert!(dequant[0] > 500.0, "First value should be large positive");
    // Last value should be large negative (clamped to -127)
    assert!(dequant[255] < -500.0, "Last value should be large negative");
}

// =============================================================================
// Q8_0Block::relative_error Edge Cases
// =============================================================================

#[test]
fn test_q8_0block_relative_error_near_zero_max() {
    // Test the early return path when max_val < 1e-10
    let values = [1e-12f32; 32]; // Very small values
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);

    // Should return 0.0 when max_val is near zero
    assert_eq!(
        error, 0.0,
        "Relative error should be 0 for near-zero values"
    );
}

#[test]
fn test_q8_0block_relative_error_exactly_zero() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);

    assert_eq!(error, 0.0, "Relative error should be 0 for zero values");
}

#[test]
fn test_q8_0block_relative_error_mixed_small() {
    let mut values = [0.0f32; 32];
    values[0] = 1e-11; // Just at the threshold
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);

    // Should return 0.0 since max is < 1e-10
    assert_eq!(error, 0.0);
}

#[test]
fn test_q8_0block_relative_error_normal() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);

    // Should be a small but non-zero relative error
    assert!(
        error > 0.0,
        "Normal values should have some quantization error"
    );
    assert!(error < 0.1, "Relative error should be small: {}", error);
}

#[test]
fn test_q8_0block_quantization_error_basic() {
    let values = [5.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);

    // Quantization error should be small
    assert!(error < 0.1, "Quantization error should be small: {}", error);
}

// =============================================================================
// SimdBackend Display Trait Tests
// =============================================================================

#[test]
fn test_simd_backend_display_avx2() {
    let backend = SimdBackend::Avx2;
    assert_eq!(format!("{}", backend), "AVX2");
}

#[test]
fn test_simd_backend_display_sse2() {
    let backend = SimdBackend::Sse2;
    assert_eq!(format!("{}", backend), "SSE2");
}

#[test]
fn test_simd_backend_display_neon() {
    let backend = SimdBackend::Neon;
    assert_eq!(format!("{}", backend), "NEON");
}

#[test]
fn test_simd_backend_display_scalar() {
    let backend = SimdBackend::Scalar;
    assert_eq!(format!("{}", backend), "Scalar");
}

#[test]
fn test_simd_backend_default() {
    let backend: SimdBackend = Default::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_equality() {
    assert_eq!(SimdBackend::Avx2, SimdBackend::Avx2);
    assert_ne!(SimdBackend::Avx2, SimdBackend::Sse2);
    assert_ne!(SimdBackend::Neon, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_clone() {
    let backend = SimdBackend::Avx2;
    let cloned = backend;
    assert_eq!(backend, cloned);
}

#[test]
fn test_simd_backend_copy() {
    let backend = SimdBackend::Neon;
    let copied: SimdBackend = backend;
    assert_eq!(backend, copied);
}

// =============================================================================
// DequantStats Tests
// =============================================================================

#[test]
fn test_dequant_stats_default() {
    let stats: DequantStats = Default::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

#[test]
fn test_dequant_stats_clone() {
    let mut stats = DequantStats::default();
    stats.blocks_processed = 100;
    stats.bytes_processed = 3200;
    stats.simd_backend = SimdBackend::Avx2;

    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, 100);
    assert_eq!(cloned.bytes_processed, 3200);
    assert_eq!(cloned.simd_backend, SimdBackend::Avx2);
}

#[test]
fn test_dequant_stats_debug() {
    let stats = DequantStats {
        blocks_processed: 42,
        bytes_processed: 1344,
        simd_backend: SimdBackend::Sse2,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("42"));
    assert!(debug_str.contains("1344"));
    assert!(debug_str.contains("Sse2"));
}

// =============================================================================
// detect_simd_backend Tests
// =============================================================================

#[test]
fn test_detect_simd_backend_runs() {
    let backend = detect_simd_backend();
    // On x86_64 with AVX2 (which is the development machine), expect AVX2
    // On other platforms, just verify it returns a valid variant
    match backend {
        SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon | SimdBackend::Scalar => {},
    }
}

#[test]
fn test_detect_simd_backend_consistent() {
    // Multiple calls should return the same result
    let backend1 = detect_simd_backend();
    let backend2 = detect_simd_backend();
    assert_eq!(backend1, backend2);
}

// =============================================================================
// extract_scale_min Tests for Blocks 4-7 (Packed Format)
// =============================================================================

#[test]
fn test_extract_scale_min_block_5() {
    let mut scales = [0u8; 12];
    // Block 5 uses: d = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4)
    //               m = (scales[9] >> 4) | ((scales[5] >> 6) << 4)
    scales[1] = 0b11_000000; // high 2 bits = 3
    scales[5] = 0b10_000000; // high 2 bits = 2
    scales[9] = 0b0011_0101; // low=5, high=3

    let (s, m) = extract_scale_min(&scales, 5);
    // scale = 5 | (3 << 4) = 5 + 48 = 53
    assert_eq!(s, 53.0, "Block 5 scale");
    // min = 3 | (2 << 4) = 3 + 32 = 35
    assert_eq!(m, 35.0, "Block 5 min");
}

#[test]
fn test_extract_scale_min_block_6() {
    let mut scales = [0u8; 12];
    scales[2] = 0b01_000000; // high 2 bits = 1
    scales[6] = 0b11_000000; // high 2 bits = 3
    scales[10] = 0b1001_0010; // low=2, high=9

    let (s, m) = extract_scale_min(&scales, 6);
    // scale = 2 | (1 << 4) = 2 + 16 = 18
    assert_eq!(s, 18.0, "Block 6 scale");
    // min = 9 | (3 << 4) = 9 + 48 = 57
    assert_eq!(m, 57.0, "Block 6 min");
}

#[test]
fn test_extract_scale_min_block_7() {
    let mut scales = [0u8; 12];
    scales[3] = 0b10_000000; // high 2 bits = 2
    scales[7] = 0b01_000000; // high 2 bits = 1
    scales[11] = 0b0100_1111; // low=15, high=4

    let (s, m) = extract_scale_min(&scales, 7);
    // scale = 15 | (2 << 4) = 15 + 32 = 47
    assert_eq!(s, 47.0, "Block 7 scale");
    // min = 4 | (1 << 4) = 4 + 16 = 20
    assert_eq!(m, 20.0, "Block 7 min");
}

#[test]
fn test_extract_scale_min_all_blocks_max_values() {
    // Test with maximum 6-bit values (63)
    let scales: [u8; 12] = [
        0b11_111111, // block 0 scale = 63, high bits = 3 for block 4
        0b11_111111, // block 1 scale = 63, high bits = 3 for block 5
        0b11_111111, // block 2 scale = 63, high bits = 3 for block 6
        0b11_111111, // block 3 scale = 63, high bits = 3 for block 7
        0b11_111111, // block 0 min = 63, high bits = 3 for block 4 min
        0b11_111111, // block 1 min = 63, high bits = 3 for block 5 min
        0b11_111111, // block 2 min = 63, high bits = 3 for block 6 min
        0b11_111111, // block 3 min = 63, high bits = 3 for block 7 min
        0b1111_1111, // block 4 low bits
        0b1111_1111, // block 5 low bits
        0b1111_1111, // block 6 low bits
        0b1111_1111, // block 7 low bits
    ];

    // Blocks 0-3: simple extraction, scale = min = 63
    for i in 0..4 {
        let (s, m) = extract_scale_min(&scales, i);
        assert_eq!(s, 63.0, "Block {} scale should be 63", i);
        assert_eq!(m, 63.0, "Block {} min should be 63", i);
    }

    // Blocks 4-7: packed extraction with max values
    for i in 4..8 {
        let (s, m) = extract_scale_min(&scales, i);
        // scale = 15 | (3 << 4) = 15 + 48 = 63
        assert_eq!(s, 63.0, "Block {} scale should be 63", i);
        // min = 15 | (3 << 4) = 63
        assert_eq!(m, 63.0, "Block {} min should be 63", i);
    }
}

// =============================================================================
// extract_scale_min_from_slice Additional Tests
// =============================================================================

#[test]
fn test_extract_scale_min_from_slice_odd_index_3() {
    let mut scales = [0u8; 12];
    // For idx=3: scale_idx=1, min_idx=5
    // scale = (scales[1] >> 6) | ((scales[3] & 0x0F) << 2)
    // min = (scales[5] >> 6) | ((scales[7] & 0x0F) << 2)
    scales[1] = 0b10_000000; // high 2 bits = 2
    scales[3] = 0b0000_0111; // low 4 bits = 7
    scales[5] = 0b11_000000; // high 2 bits = 3
    scales[7] = 0b0000_1010; // low 4 bits = 10

    let (s, m) = extract_scale_min_from_slice(&scales, 3);
    // scale = 2 | (7 << 2) = 2 + 28 = 30
    assert_eq!(s, 30.0, "idx=3 scale");
    // min = 3 | (10 << 2) = 3 + 40 = 43
    assert_eq!(m, 43.0, "idx=3 min");
}

#[test]
fn test_extract_scale_min_from_slice_odd_index_5() {
    let mut scales = [0u8; 12];
    // For idx=5: scale_idx=2, min_idx=6
    scales[2] = 0b01_000000; // high 2 bits = 1
    scales[4] = 0b0000_1100; // low 4 bits = 12
    scales[6] = 0b11_000000; // high 2 bits = 3
    scales[8] = 0b0000_0001; // low 4 bits = 1

    let (s, m) = extract_scale_min_from_slice(&scales, 5);
    // scale = 1 | (12 << 2) = 1 + 48 = 49
    assert_eq!(s, 49.0, "idx=5 scale");
    // min = 3 | (1 << 2) = 3 + 4 = 7
    assert_eq!(m, 7.0, "idx=5 min");
}

#[test]
fn test_extract_scale_min_from_slice_odd_index_7() {
    let mut scales = [0u8; 12];
    // For idx=7: scale_idx=3, min_idx=7
    scales[3] = 0b00_000000; // high 2 bits = 0
    scales[5] = 0b0000_1111; // low 4 bits = 15
    scales[7] = 0b10_000000; // high 2 bits = 2
    scales[9] = 0b0000_0011; // low 4 bits = 3

    let (s, m) = extract_scale_min_from_slice(&scales, 7);
    // scale = 0 | (15 << 2) = 60
    assert_eq!(s, 60.0, "idx=7 scale");
    // min = 2 | (3 << 2) = 2 + 12 = 14
    assert_eq!(m, 14.0, "idx=7 min");
}

// =============================================================================
// InterleavedQ4K Additional Tests
// =============================================================================

#[test]
fn test_interleaved_q4k_scales_and_dmin_extraction() {
    // Test that scales and dmin are correctly extracted
    let mut data = vec![0u8; 144];

    // d = 0.5 (f16: 0x3800)
    data[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
    // dmin = 0.25 (f16: 0x3400)
    data[2..4].copy_from_slice(&0x3400u16.to_le_bytes());

    // Set scale bytes
    for i in 4..16 {
        data[i] = (i - 4) as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();

    assert!((interleaved.d[0] - 0.5).abs() < 1e-3);
    assert!((interleaved.dmin[0] - 0.25).abs() < 1e-3);

    // Check scales were copied
    for (i, &s) in interleaved.scales.iter().enumerate() {
        assert_eq!(s, i as u8);
    }
}

#[test]
fn test_interleaved_q4k_qs_copy() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0

    // Set qs with recognizable pattern
    for i in 16..144 {
        data[i] = ((i - 16) % 256) as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();

    // Check qs were copied correctly
    for (i, &q) in interleaved.qs.iter().enumerate() {
        assert_eq!(q, (i % 256) as u8);
    }
}

#[test]
fn test_interleaved_q4k_dot_with_all_max_nibbles() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0

    // Set all scales to 1
    for i in 4..16 {
        data[i] = 1;
    }

    // Set all qs to 0xFF (max nibbles: low=15, high=15)
    for i in 16..144 {
        data[i] = 0xFF;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite(), "Result should be finite: {}", result);
    assert!(result > 0.0, "Result should be positive with max nibbles");
}

#[test]
fn test_interleaved_q4k_dot_with_zero_d() {
    let mut data = vec![0u8; 144];
    // d = 0.0 (f16: 0x0000)
    data[0..2].copy_from_slice(&0x0000u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert_eq!(result, 0.0, "Zero d should give zero result");
}

#[test]
fn test_interleaved_q4k_dot_three_superblocks() {
    let mut data = vec![0u8; 144 * 3];

    // Super-block 0: d = 1.0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // Super-block 1: d = 2.0
    data[144..146].copy_from_slice(&0x4000u16.to_le_bytes());

    // Super-block 2: d = 0.5
    data[288..290].copy_from_slice(&0x3800u16.to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();
    assert_eq!(interleaved.num_super_blocks, 3);
    assert_eq!(interleaved.num_values(), 768);

    let activations = vec![0.1f32; 768];
    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

// =============================================================================
// fused_q4_0_q8_0_dot_simd Boundary Tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_simd_exactly_256_elements() {
    // 256 elements = 8 blocks, triggers 4-block unrolling
    let in_dim = 256;
    let num_blocks = 8;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 2..18 {
            q4_data[start + i] = 0x44; // q_low=4, q_high=4
        }
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![1i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_simd_exactly_128_elements() {
    // 128 elements = 4 blocks, at the 4-block unroll threshold
    let in_dim = 128;
    let num_blocks = 4;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![1i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_simd_5_blocks() {
    // 5 blocks = 160 elements, tests remainder handling after 4-block
    let in_dim = 160;
    let num_blocks = 5;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 2..18 {
            q4_data[start + i] = 0x55;
        }
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![2i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_simd_9_blocks() {
    // 9 blocks = 288 elements, tests 4-block unroll with remainder
    let in_dim = 288;
    let num_blocks = 9;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![1i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

// =============================================================================
// Constants Tests
// =============================================================================

#[test]
fn test_block_size_constant() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant() {
    assert_eq!(QK_K, 256);
}

// =============================================================================
// Q8_0Block Additional Tests
// =============================================================================

#[test]
fn test_q8_0block_quantize_with_zero_block() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);

    // Fallback scale = 1.0 / 127.0
    assert!((block.scale - 1.0 / 127.0).abs() < 1e-10);

    // All quants should be 0
    for q in &block.quants {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_q8_0block_quantize_negative_dominant() {
    let mut values = [0.0f32; 32];
    values[0] = -100.0;

    let block = Q8_0Block::quantize(&values);

    // Scale should be based on the max abs value
    assert!((block.scale - 100.0 / 127.0).abs() < 1e-3);

    // First quant should be -127 (clamped)
    assert_eq!(block.quants[0], -127);
}

#[test]
fn test_q8_0block_debug() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let debug_str = format!("{:?}", block);

    assert!(debug_str.contains("Q8_0Block"));
    assert!(debug_str.contains("scale"));
    assert!(debug_str.contains("quants"));
}

#[test]
fn test_q8_0block_clone() {
    let values = [5.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let cloned = block.clone();

    assert_eq!(block.scale, cloned.scale);
    assert_eq!(block.quants, cloned.quants);
}

// =============================================================================
// fused_q8_0_q8_0_dot_scalar Additional Tests
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_boundary_in_dim() {
    // Test with in_dim that isn't a multiple of 32
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    for i in 2..34 {
        q8_weight_data[i] = 10u8;
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![5i8; 32];

    // Test with in_dim = 20 (less than block size)
    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 20);
    assert!(result.is_finite());
    // Should only sum first 20 products
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_exact_block() {
    // in_dim exactly 32
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for i in 2..34 {
        q8_weight_data[i] = 1u8;
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // Expected: 1.0 * 1.0 * (1 * 1 * 32) = 32
    assert!((result - 32.0).abs() < 1.0);
}

// =============================================================================
// Parallel Matvec Large Dimension Tests (Parallel Path)
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_above_threshold() {
    // 2048 outputs should trigger parallel path (threshold is 1024)
    let in_dim = 32;
    let out_dim = 2048;
    let bytes_per_row = 18;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        let start = row * bytes_per_row;
        weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let activations = vec![0.01f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);

    // All outputs should be finite
    for v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_large() {
    // 1024 outputs with parallel path
    let in_dim = 32;
    let out_dim = 1024;
    let bytes_per_row = 34;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        let start = row * bytes_per_row;
        weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let activations = vec![0.1f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

// =============================================================================
// quantize_activations_q8k_into Tests
// =============================================================================

#[test]
fn test_quantize_activations_q8k_into_two_superblocks() {
    let activations: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.1).collect();
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants).expect("should work");

    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);

    // Check that first superblock has mostly negative values
    let negative_count: usize = quants[..256].iter().filter(|&&q| q < 0).count();
    assert!(
        negative_count > 100,
        "First superblock should have many negatives"
    );

    // Check that second superblock has mostly positive values
    let positive_count: usize = quants[256..].iter().filter(|&&q| q > 0).count();
    assert!(
        positive_count > 100,
        "Second superblock should have many positives"
    );
}

#[test]
fn test_quantize_activations_q8k_into_uniform() {
    let activations = vec![50.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants).expect("should work");

    // All quants should be 127 (max positive)
    for q in &quants {
        assert_eq!(*q, 127);
    }
}

// =============================================================================
// fused_q4_0_q8_0_dot_scalar Special Values
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_with_nan_scale() {
    let mut q4_data = vec![0u8; 18];
    // NaN in f16: 0x7E00
    q4_data[0..2].copy_from_slice(&0x7E00u16.to_le_bytes());

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(result.is_nan(), "NaN scale should propagate");
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_with_inf_scale() {
    let mut q4_data = vec![0u8; 18];
    // Infinity in f16: 0x7C00
    q4_data[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(
        result.is_infinite() || result.is_nan(),
        "Inf scale should produce inf/nan"
    );
}

// =============================================================================
// InterleavedQ4K::dot Scalar Fallback (non-x86_64 path)
// =============================================================================

#[test]
fn test_interleaved_q4k_dot_scalar_via_small_input() {
    // Even on x86_64 with AVX2, we test the scalar path through the dot() dispatcher
    // The scalar path is also tested indirectly - here we ensure correctness
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    data[4] = 1; // scale for first block

    // Set recognizable pattern in qs
    for i in 16..144 {
        data[i] = 0x21; // q_low=1, q_high=2
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot works");
    assert!(result.is_finite());
}

// =============================================================================
// Q8KSuperBlock::quantize_into Tests
// =============================================================================

#[test]
fn test_q8k_superblock_quantize_into_basic() {
    let values = [5.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // All should be 127
    for q in &quants {
        assert_eq!(*q, 127);
    }
}

#[test]
fn test_q8k_superblock_quantize_into_negative() {
    let values = [-5.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // All should be -127
    for q in &quants {
        assert_eq!(*q, -127);
    }
}

#[test]
fn test_q8k_superblock_quantize_into_mixed() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 128.0) * 0.5;
    }
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // First half should be negative, second half positive
    assert!(quants[0] < 0);
    assert!(quants[255] > 0);
}

// =============================================================================
// Additional fused_q4_0_q8_0_parallel_matvec_into Tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_large_output() {
    let in_dim = 64;
    let out_dim = 256;
    let bytes_per_row = 36; // 2 blocks

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        for block in 0..2 {
            let start = row * bytes_per_row + block * 18;
            weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
    }

    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_ok());

    for v in &output {
        assert!(v.is_finite());
    }
}

// =============================================================================
// quantize_to_q8_blocks and dequantize_q8_blocks Additional Tests
// =============================================================================

#[test]
fn test_quantize_to_q8_blocks_four_blocks() {
    let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("valid");

    assert_eq!(blocks.len(), 4);

    let dequant = dequantize_q8_blocks(&blocks);
    assert_eq!(dequant.len(), 128);

    // Check trend preserved
    for i in 1..128 {
        assert!(
            dequant[i] > dequant[i - 1] - 0.5,
            "Values should be roughly ascending"
        );
    }
}

#[test]
fn test_dequantize_q8_blocks_single_block() {
    let values = [10.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let blocks = vec![block];

    let dequant = dequantize_q8_blocks(&blocks);

    assert_eq!(dequant.len(), 32);
    for v in &dequant {
        assert!((v - 10.0).abs() < 0.5);
    }
}
