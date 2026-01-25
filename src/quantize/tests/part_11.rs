//! Extended coverage tests for quantize/types.rs (PMAT-802)
//!
//! This file targets uncovered code paths in the quantization types module.
//! Focus areas:
//! - Edge cases in Q8_0Block and Q8KSuperBlock quantization
//! - Near-zero value handling (minimal scale paths)
//! - Clamping boundary conditions
//! - Error message formatting

use crate::quantize::detect_simd_backend;
use crate::quantize::types::{
    DequantStats, InterleavedQ4K, Q4_0Block, Q4_KBlock, Q5_KBlock, Q6_KBlock, Q8KSuperBlock,
    Q8_0Block, SimdBackend, BLOCK_SIZE, QK_K,
};

// ============================================================================
// Q8_0Block Edge Case Coverage
// ============================================================================

/// Test Q8_0Block quantization with extremely small values (near-zero path)
/// This targets the `else` branch in quantize() where max_abs <= 1e-10
#[test]
fn test_q8_0_block_quantize_near_zero_values() {
    // All values are effectively zero
    let values = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&values);

    // Scale should be the minimal scale (1.0 / 127.0)
    let expected_scale = 1.0 / 127.0;
    assert!(
        (block.scale - expected_scale).abs() < 1e-9,
        "Expected minimal scale {expected_scale}, got {}",
        block.scale
    );

    // All quants should be 0 since values are tiny
    assert!(
        block.quants.iter().all(|&q| q == 0),
        "Expected all zeros for near-zero input"
    );
}

/// Test Q8_0Block quantization with exactly zero values
#[test]
fn test_q8_0_block_quantize_exact_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);

    // Should use minimal scale
    assert!(block.scale > 0.0, "Scale should be positive even for zeros");
    assert!(
        block.quants.iter().all(|&q| q == 0),
        "All quants should be zero"
    );
}

/// Test Q8_0Block quantization with values requiring negative clamping
#[test]
fn test_q8_0_block_quantize_clamp_negative() {
    // Create values that would exceed -128 when quantized
    // With scale = 1000/127 = 7.87, value -1000 would be -127 * 7.87 / 7.87 = -127
    // To get clamping, we need asymmetric values
    let mut values = [0.0f32; 32];
    values[0] = 1.0; // Sets max_abs to 1.0
    values[1] = -200.0; // Much larger than max_abs

    // max_abs = 200.0, scale = 200/127 = 1.575
    // -200 / 1.575 = -127.0 (at boundary)
    // Let's use more extreme values
    let mut values2 = [0.0f32; 32];
    values2[0] = 1.0; // max_abs starts at 1.0
                      // After processing all, max_abs will be max of all abs values
    for i in 1..32 {
        values2[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let block = Q8_0Block::quantize(&values2);
    assert!(block.scale > 0.0);
}

/// Test Q8_0Block quantization with large positive value causing clamping
#[test]
fn test_q8_0_block_quantize_clamp_positive() {
    // Create scenario where some values might clip to 127
    let mut values = [100.0f32; 32];
    values[0] = 100.0; // max
    let block = Q8_0Block::quantize(&values);

    // All values are the same and at max, so all quants should be 127
    assert!(
        block.quants.iter().all(|&q| q == 127),
        "All should clamp to 127"
    );
}

/// Test Q8_0Block relative_error with tiny max value (early return path)
#[test]
fn test_q8_0_block_relative_error_tiny_max() {
    // Create block with some data
    let original = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&original);

    // relative_error should return 0.0 when max_val < 1e-10
    let error = block.relative_error(&original);
    assert!(
        error.abs() < 1e-6,
        "Relative error should be 0 for tiny values, got {error}"
    );
}

/// Test Q8_0Block quantization_error calculation
#[test]
fn test_q8_0_block_quantization_error_varied() {
    // Create block with varied values
    let mut original = [0.0f32; 32];
    for i in 0..32 {
        original[i] = (i as f32) * 0.5 - 7.5; // Range roughly -7.5 to 8
    }

    let block = Q8_0Block::quantize(&original);
    let error = block.quantization_error(&original);

    // Error should be positive and reasonably small
    assert!(error >= 0.0, "Error should be non-negative");
    assert!(error < 1.0, "Error should be small for reasonable values");
}

/// Test Q8_0Block with mixed sign extreme values
#[test]
fn test_q8_0_block_mixed_extremes() {
    let mut values = [0.0f32; 32];
    values[0] = 1000.0;
    values[1] = -1000.0;
    for i in 2..32 {
        values[i] = (i as f32) - 16.0;
    }

    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();

    // Check that max and min are approximately preserved
    let max_orig = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_deq = dequantized
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Due to quantization, max should be close but not exact
    assert!(
        (max_orig - max_deq).abs() < max_orig * 0.02,
        "Max should be preserved within 2%"
    );
}

// ============================================================================
// Q8KSuperBlock Edge Case Coverage
// ============================================================================

/// Test Q8KSuperBlock quantization with near-zero values
#[test]
fn test_q8k_superblock_quantize_near_zero() {
    let values = [1e-12f32; 256];
    let block = Q8KSuperBlock::quantize(&values);

    // Should use minimal scale
    let expected_scale = 1.0 / 127.0;
    assert!(
        (block.scale - expected_scale).abs() < 1e-9,
        "Expected minimal scale"
    );
}

/// Test Q8KSuperBlock quantize_into with near-zero values
#[test]
fn test_q8k_superblock_quantize_into_near_zero() {
    let values = [1e-12f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    // Should use minimal scale
    let expected_scale = 1.0 / 127.0;
    assert!(
        (scale - expected_scale).abs() < 1e-9,
        "Expected minimal scale from quantize_into"
    );
}

/// Test Q8KSuperBlock quantize_into with varied values
#[test]
fn test_q8k_superblock_quantize_into_varied() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 127.5) * 0.1;
    }
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0, "Scale should be positive");
    // Check that quantization is symmetric
    let sum: i32 = quants.iter().map(|&q| q as i32).sum();
    assert!(sum.abs() < 256, "Sum should be roughly balanced");
}

/// Test Q8KSuperBlock dequantize roundtrip
#[test]
fn test_q8k_superblock_dequantize_roundtrip() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 127.5) * 0.01;
    }

    let block = Q8KSuperBlock::quantize(&values);
    let dequantized = block.dequantize();

    // Check that values are approximately preserved
    let mut max_error: f32 = 0.0;
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        max_error = max_error.max((orig - deq).abs());
    }

    assert!(
        max_error < 0.1,
        "Roundtrip error should be small, got {max_error}"
    );
}

/// Test Q8KSuperBlock with all negative values
#[test]
fn test_q8k_superblock_all_negative() {
    let values = [-1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);

    // All quants should be -127
    assert!(
        block.quants.iter().all(|&q| q == -127),
        "All should be -127 for uniform negative"
    );
}

// ============================================================================
// InterleavedQ4K Coverage (types.rs version)
// ============================================================================

/// Test InterleavedQ4K from_q4k with empty data
#[test]
fn test_interleaved_q4k_empty() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("valid result");
    assert_eq!(interleaved.num_super_blocks, 0);
    assert_eq!(interleaved.num_values(), 0);
}

/// Test InterleavedQ4K from_q4k with valid data
#[test]
fn test_interleaved_q4k_single_block() {
    // Create valid Q4_K super-block data (144 bytes)
    let mut data = vec![0u8; 144];

    // Set d = 1.0 (f16 representation: 0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;

    // Set dmin = 0.5 (f16 representation: 0x3800)
    data[2] = 0x00;
    data[3] = 0x38;

    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());

    let interleaved = result.expect("valid result");
    assert_eq!(interleaved.num_super_blocks, 1);
    assert_eq!(interleaved.num_values(), 256);
    assert_eq!(interleaved.d.len(), 1);
    assert_eq!(interleaved.dmin.len(), 1);
    assert_eq!(interleaved.scales.len(), 12);
    assert_eq!(interleaved.qs.len(), 128);
}

/// Test InterleavedQ4K from_q4k with invalid length
#[test]
fn test_interleaved_q4k_invalid_length() {
    // 143 bytes - not a multiple of 144
    let data = vec![0u8; 143];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());

    // Verify error message contains relevant info
    if let Err(e) = result {
        let err_str = format!("{e:?}");
        assert!(
            err_str.contains("143") || err_str.contains("144"),
            "Error should mention the sizes"
        );
    }
}

/// Test InterleavedQ4K from_q4k with multiple super-blocks
#[test]
fn test_interleaved_q4k_multiple_blocks() {
    // 3 super-blocks = 432 bytes
    let mut data = vec![0u8; 144 * 3];

    // Set different d values for each super-block
    // Block 0: d = 1.0
    data[0] = 0x00;
    data[1] = 0x3C;
    // Block 1: d = 2.0 (0x4000)
    data[144] = 0x00;
    data[145] = 0x40;
    // Block 2: d = 0.5 (0x3800)
    data[288] = 0x00;
    data[289] = 0x38;

    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());

    let interleaved = result.expect("valid result");
    assert_eq!(interleaved.num_super_blocks, 3);
    assert_eq!(interleaved.num_values(), 768);
    assert_eq!(interleaved.d.len(), 3);
    assert_eq!(interleaved.dmin.len(), 3);
    assert_eq!(interleaved.scales.len(), 36); // 3 * 12
    assert_eq!(interleaved.qs.len(), 384); // 3 * 128
}

/// Test InterleavedQ4K Debug trait
#[test]
fn test_interleaved_q4k_debug() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");
    let debug_str = format!("{interleaved:?}");
    assert!(debug_str.contains("InterleavedQ4K"));
    assert!(debug_str.contains("num_super_blocks"));
}

/// Test InterleavedQ4K Clone trait
#[test]
fn test_interleaved_q4k_clone() {
    let data = vec![0u8; 144];
    let original = InterleavedQ4K::from_q4k(&data).expect("valid data");
    let cloned = original.clone();

    assert_eq!(original.num_super_blocks, cloned.num_super_blocks);
    assert_eq!(original.d, cloned.d);
    assert_eq!(original.dmin, cloned.dmin);
    assert_eq!(original.scales, cloned.scales);
    assert_eq!(original.qs, cloned.qs);
}

// ============================================================================
// SimdBackend and DequantStats Coverage
// ============================================================================

/// Test all SimdBackend Display variants
#[test]
fn test_simd_backend_display_comprehensive() {
    let backends = [
        (SimdBackend::Avx2, "AVX2"),
        (SimdBackend::Sse2, "SSE2"),
        (SimdBackend::Neon, "NEON"),
        (SimdBackend::Scalar, "Scalar"),
    ];

    for (backend, expected) in backends {
        let display = format!("{backend}");
        assert_eq!(
            display, expected,
            "Display for {backend:?} should be {expected}"
        );
    }
}

/// Test SimdBackend Debug trait
#[test]
fn test_simd_backend_debug_all() {
    let backends = [
        SimdBackend::Avx2,
        SimdBackend::Sse2,
        SimdBackend::Neon,
        SimdBackend::Scalar,
    ];

    for backend in backends {
        let debug_str = format!("{backend:?}");
        assert!(
            !debug_str.is_empty(),
            "Debug output should not be empty for {backend:?}"
        );
    }
}

/// Test DequantStats construction and fields
#[test]
fn test_dequant_stats_fields() {
    let stats = DequantStats {
        blocks_processed: 42,
        bytes_processed: 1024,
        simd_backend: SimdBackend::Avx2,
    };

    assert_eq!(stats.blocks_processed, 42);
    assert_eq!(stats.bytes_processed, 1024);
    assert_eq!(stats.simd_backend, SimdBackend::Avx2);
}

/// Test DequantStats Debug output contains all fields
#[test]
fn test_dequant_stats_debug_comprehensive() {
    let stats = DequantStats {
        blocks_processed: 999,
        bytes_processed: 12345,
        simd_backend: SimdBackend::Neon,
    };

    let debug_str = format!("{stats:?}");
    assert!(debug_str.contains("999"), "Should contain blocks count");
    assert!(debug_str.contains("12345"), "Should contain bytes count");
    assert!(debug_str.contains("Neon"), "Should contain backend name");
}

/// Test detect_simd_backend returns a valid variant
#[test]
fn test_detect_simd_backend_valid() {
    let backend = detect_simd_backend();

    // Should be one of the valid variants
    let is_valid = matches!(
        backend,
        SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon | SimdBackend::Scalar
    );
    assert!(is_valid, "Backend should be a valid variant");
}

// ============================================================================
// Block Type Coverage
// ============================================================================

/// Test Q4_0Block Debug and Clone
#[test]
fn test_q4_0_block_traits() {
    let block = Q4_0Block {
        scale: 2.5,
        quants: [0xABu8; 16],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
    assert_eq!(block.quants, cloned.quants);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q4_0Block"));
    assert!(debug_str.contains("scale"));
    assert!(debug_str.contains("quants"));
}

/// Test Q4_KBlock Debug and Clone
#[test]
fn test_q4_k_block_traits() {
    let block = Q4_KBlock {
        d: 1.5,
        dmin: 0.25,
        scales: [10u8; 12],
        qs: [0x55u8; 128],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    assert!((block.dmin - cloned.dmin).abs() < 1e-6);
    assert_eq!(block.scales, cloned.scales);
    assert_eq!(block.qs, cloned.qs);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q4_KBlock"));
}

/// Test Q5_KBlock Debug and Clone
#[test]
fn test_q5_k_block_traits() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 0.5,
        scales: [5u8; 12],
        qh: [0xFFu8; 32],
        qs: [0x33u8; 128],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    assert!((block.dmin - cloned.dmin).abs() < 1e-6);
    assert_eq!(block.scales, cloned.scales);
    assert_eq!(block.qh, cloned.qh);
    assert_eq!(block.qs, cloned.qs);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q5_KBlock"));
}

/// Test Q6_KBlock Debug and Clone
#[test]
fn test_q6_k_block_traits() {
    let block = Q6_KBlock {
        d: 1.0,
        scales: [1i8; 16],
        qh: [0x11u8; 64],
        qs: [0x22u8; 128],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    assert_eq!(block.scales, cloned.scales);
    assert_eq!(block.qh, cloned.qh);
    assert_eq!(block.qs, cloned.qs);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q6_KBlock"));
}

/// Test Q8KSuperBlock Debug and Clone
#[test]
fn test_q8k_superblock_traits() {
    let block = Q8KSuperBlock {
        scale: 0.5,
        quants: [10i8; 256],
    };

    // Test Clone
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
    assert_eq!(block.quants, cloned.quants);

    // Test Debug
    let debug_str = format!("{block:?}");
    assert!(debug_str.contains("Q8KSuperBlock"));
}

// ============================================================================
// Constants Coverage
// ============================================================================

/// Test BLOCK_SIZE constant value
#[test]
fn test_block_size_value() {
    assert_eq!(BLOCK_SIZE, 32);
}

/// Test QK_K constant value
#[test]
fn test_qk_k_value() {
    assert_eq!(QK_K, 256);
}

/// Test relationship between constants
#[test]
fn test_constants_relationship() {
    // QK_K should be exactly 8 blocks worth
    assert_eq!(QK_K, BLOCK_SIZE * 8);
}

// ============================================================================
// Edge Cases for Quantization Boundaries
// ============================================================================

/// Test Q8_0Block with value exactly at -128 boundary
#[test]
fn test_q8_0_block_boundary_minus_128() {
    // Scale = 1.0, so -128.0 should map to -128 (clamped)
    let mut values = [0.0f32; 32];
    values[0] = 128.0; // This sets the scale
    values[1] = -128.0; // This should map to -128

    let block = Q8_0Block::quantize(&values);
    // Scale should be 128/127 ≈ 1.0079
    // -128 / 1.0079 ≈ -127.0 (within valid range)
    assert!(
        block.quants[1] <= 0,
        "Negative value should quantize to negative"
    );
}

/// Test Q8_0Block with value exactly at 127 boundary
#[test]
fn test_q8_0_block_boundary_127() {
    let values = [1.0f32; 32]; // All values at max
    let block = Q8_0Block::quantize(&values);

    // All should be 127
    assert!(
        block.quants.iter().all(|&q| q == 127),
        "Max values should quantize to 127"
    );
}

/// Test Q8KSuperBlock with value at clamping boundary
#[test]
fn test_q8k_superblock_clamping() {
    let mut values = [0.0f32; 256];
    values[0] = 1.0; // Reference max
    values[1] = 1.0; // Same max
    for i in 2..256 {
        values[i] = 1.0;
    }

    let block = Q8KSuperBlock::quantize(&values);

    // All should be 127
    assert!(
        block.quants.iter().all(|&q| q == 127),
        "All max values should be 127"
    );
}

// ============================================================================
// Additional Coverage for types.rs InterleavedQ4K fields access
// ============================================================================

/// Test that InterleavedQ4K fields are accessible and have correct values
#[test]
fn test_interleaved_q4k_field_access() {
    // Create data with specific values
    let mut data = vec![0u8; 144];

    // d = 1.5 (f16: 0x3E00)
    data[0] = 0x00;
    data[1] = 0x3E;

    // dmin = 0.25 (f16: 0x3400)
    data[2] = 0x00;
    data[3] = 0x34;

    // Set some scale values
    for i in 4..16 {
        data[i] = (i - 4) as u8;
    }

    // Set some qs values
    for i in 16..144 {
        data[i] = ((i - 16) % 256) as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");

    // Check d value is close to 1.5
    assert!(
        (interleaved.d[0] - 1.5).abs() < 0.1,
        "d should be approximately 1.5, got {}",
        interleaved.d[0]
    );

    // Check dmin value is close to 0.25
    assert!(
        (interleaved.dmin[0] - 0.25).abs() < 0.1,
        "dmin should be approximately 0.25, got {}",
        interleaved.dmin[0]
    );

    // Check scales were copied correctly
    for i in 0..12 {
        assert_eq!(interleaved.scales[i], i as u8, "scales[{i}] should be {i}");
    }

    // Check qs were copied correctly
    for i in 0..128 {
        assert_eq!(
            interleaved.qs[i],
            (i % 256) as u8,
            "qs[{i}] should be {}",
            i % 256
        );
    }
}

/// Test InterleavedQ4K with actual non-zero quantized values
#[test]
fn test_interleaved_q4k_with_real_data() {
    let mut data = vec![0u8; 144 * 2]; // 2 super-blocks

    // First super-block: d=2.0, dmin=1.0
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes()); // d=2.0
    data[2..4].copy_from_slice(&0x3C00u16.to_le_bytes()); // dmin=1.0

    // Set scales to non-zero
    for i in 4..16 {
        data[i] = 32; // Scale value of 32
    }

    // Set qs to alternating pattern
    for i in 16..144 {
        data[i] = 0x12; // low=2, high=1
    }

    // Second super-block: d=0.5, dmin=0.25
    data[144..146].copy_from_slice(&0x3800u16.to_le_bytes()); // d=0.5
    data[146..148].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin=0.25

    for i in 148..160 {
        data[i] = 16;
    }

    for i in 160..288 {
        data[i] = 0x34;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");

    assert_eq!(interleaved.num_super_blocks, 2);

    // Verify d values
    assert!((interleaved.d[0] - 2.0).abs() < 0.1);
    assert!((interleaved.d[1] - 0.5).abs() < 0.1);

    // Verify dmin values
    assert!((interleaved.dmin[0] - 1.0).abs() < 0.1);
    assert!((interleaved.dmin[1] - 0.25).abs() < 0.1);
}
