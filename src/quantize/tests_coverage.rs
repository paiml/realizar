//! T-COV-95 Coverage tests for quantize/mod.rs
//!
//! Covers:
//! - quantize_activations_q8k_into: success, error paths (not multiple of 256, buffer too small)
//! - quantize_to_q8_blocks: success, error path (not multiple of 32)
//! - dequantize_q8_blocks: round-trip
//! - InterleavedQ4K::from_q4k: success, error path
//! - InterleavedQ4K::num_values
//! - InterleavedQ4K::dot (scalar)
//! - fused_q4_0_q8_0_dot_scalar: known values
//! - fused_q4_0_q8_0_parallel_matvec: success and error paths
//! - fused_q4_0_q8_0_parallel_matvec_into: success and error paths
//! - f16_to_f32_lut: known values
//! - SimdBackend Display + Default
//! - DequantStats Default
//! - detect_simd_backend
//! - Q8_0Block::quantize, dequantize, quantization_error, relative_error
//! - Q8KSuperBlock::quantize, quantize_into, dequantize

use super::*;

// ============================================================================
// quantize_activations_q8k_into
// ============================================================================

#[test]
fn test_quantize_activations_q8k_into_success() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0, "Scale should be positive");
    // All same value -> all quants should be the same
    assert!(quants.iter().all(|&q| q == quants[0]));
}

#[test]
fn test_quantize_activations_q8k_into_two_superblocks() {
    let activations = vec![0.5f32; 512];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
}

#[test]
fn test_quantize_activations_q8k_into_not_multiple_of_256() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("256"), "Error should mention 256: {}", err);
}

#[test]
fn test_quantize_activations_q8k_into_scales_too_small() {
    let activations = vec![1.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 1]; // Only room for 1
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Scales") || err.contains("scales") || err.contains("too small"),
        "Error should mention scales buffer: {}",
        err
    );
}

#[test]
fn test_quantize_activations_q8k_into_quants_too_small() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Quants") || err.contains("quants") || err.contains("too small"),
        "Error should mention quants buffer: {}",
        err
    );
}

// ============================================================================
// quantize_to_q8_blocks / dequantize_q8_blocks
// ============================================================================

#[test]
fn test_quantize_to_q8_blocks_success() {
    let values = vec![1.0f32; 64]; // 2 blocks of 32
    let blocks = quantize_to_q8_blocks(&values).expect("should quantize");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_not_multiple_of_32() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("32"), "Error should mention 32: {}", err);
}

#[test]
fn test_quantize_dequantize_round_trip() {
    let original = vec![0.5f32; 32];
    let blocks = quantize_to_q8_blocks(&original).expect("should quantize");
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);

    // Check approximate round-trip (quantization introduces error)
    for (o, d) in original.iter().zip(dequantized.iter()) {
        assert!(
            (o - d).abs() < 0.01,
            "Round-trip error too large: {} vs {}",
            o,
            d
        );
    }
}

#[test]
fn test_quantize_dequantize_varied_values() {
    let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let blocks = quantize_to_q8_blocks(&original).expect("should quantize");
    let dequantized = dequantize_q8_blocks(&blocks);

    for (o, d) in original.iter().zip(dequantized.iter()) {
        assert!(
            (o - d).abs() < 0.02,
            "Round-trip error too large: {} vs {}",
            o,
            d
        );
    }
}

#[test]
fn test_dequantize_q8_blocks_empty() {
    let blocks: Vec<Q8_0Block> = vec![];
    let output = dequantize_q8_blocks(&blocks);
    assert!(output.is_empty());
}

// ============================================================================
// InterleavedQ4K
// ============================================================================

#[test]
fn test_interleaved_q4k_from_q4k_success() {
    // 144 bytes per super-block
    let data = vec![0u8; 144];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let iq = result.unwrap();
    assert_eq!(iq.num_super_blocks, 1);
    assert_eq!(iq.num_values(), 256);
}

#[test]
fn test_interleaved_q4k_from_q4k_two_superblocks() {
    let data = vec![0u8; 288]; // 2 super-blocks
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let iq = result.unwrap();
    assert_eq!(iq.num_super_blocks, 2);
    assert_eq!(iq.num_values(), 512);
    assert_eq!(iq.d.len(), 2);
    assert_eq!(iq.dmin.len(), 2);
    assert_eq!(iq.scales.len(), 24);
    assert_eq!(iq.qs.len(), 256);
}

#[test]
fn test_interleaved_q4k_from_q4k_invalid_length() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("144"), "Error should mention 144: {}", err);
}

#[test]
fn test_interleaved_q4k_from_q4k_empty() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let iq = result.unwrap();
    assert_eq!(iq.num_super_blocks, 0);
    assert_eq!(iq.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_dot_scalar() {
    // Create a super-block with known values
    let mut data = vec![0u8; 144];
    // d = 1.0 as f16 (bits = 0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    // dmin = 0.0 as f16
    data[2] = 0x00;
    data[3] = 0x00;
    // scales: all zeros (results in zero output)
    // qs: all zeros

    let iq = InterleavedQ4K::from_q4k(&data).unwrap();
    let activations = vec![1.0f32; 256];
    let result = iq.dot(&activations);
    assert!(result.is_ok());
    // With zero scales and zero qs, result should be 0 or very small
    let val = result.unwrap();
    assert!(
        val.abs() < 0.1,
        "Dot product with zero data should be near zero, got {}",
        val
    );
}

// ============================================================================
// fused_q4_0_q8_0_dot_scalar
// ============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_zero() {
    // Q4_0 block: 18 bytes (2 scale + 16 quants)
    // All zeros
    let q4_data = vec![0u8; 18];
    let q8_scales = vec![1.0f32];
    let q8_quants = vec![0i8; 32];
    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_empty() {
    let result = fused_q4_0_q8_0_dot_scalar(&[], &[], &[], 0);
    assert_eq!(result, 0.0);
}

// ============================================================================
// fused_q4_0_q8_0_parallel_matvec
// ============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_weight_too_small() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 2);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("too small"),
        "Error should mention too small: {}",
        err
    );
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_activation_mismatch() {
    // Need enough weight data for out_dim=1, in_dim=32 -> 1 block -> 18 bytes
    let weight_data = vec![0u8; 18];
    let activations = vec![1.0f32; 64]; // Wrong size
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Activation") || err.contains("activation"),
        "Error should mention activation mismatch: {}",
        err
    );
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_success() {
    // 1 row, 32 elements = 1 Q4_0 block (18 bytes)
    let weight_data = vec![0u8; 18];
    let activations = vec![0.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 1);
}

// ============================================================================
// fused_q4_0_q8_0_parallel_matvec_into
// ============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_activation_mismatch() {
    let weight_data = vec![0u8; 18];
    let activations = vec![1.0f32; 64]; // Wrong
    let mut output = vec![0.0f32; 1];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_success() {
    let weight_data = vec![0u8; 18];
    let activations = vec![0.0f32; 32];
    let mut output = vec![0.0f32; 1];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_ok());
    assert_eq!(output.len(), 1);
}

// ============================================================================
// f16_to_f32_lut
// ============================================================================

#[test]
fn test_f16_to_f32_lut_zero() {
    let val = f16_to_f32_lut(0);
    assert_eq!(val, 0.0);
}

#[test]
fn test_f16_to_f32_lut_one() {
    // f16 1.0 = 0x3C00
    let val = f16_to_f32_lut(0x3C00);
    assert!((val - 1.0).abs() < 0.001, "Expected 1.0, got {}", val);
}

#[test]
fn test_f16_to_f32_lut_negative_one() {
    // f16 -1.0 = 0xBC00
    let val = f16_to_f32_lut(0xBC00);
    assert!((val - (-1.0)).abs() < 0.001, "Expected -1.0, got {}", val);
}

#[test]
fn test_f16_to_f32_lut_max() {
    // f16 max positive = 0x7BFF (~65504)
    let val = f16_to_f32_lut(0x7BFF);
    assert!(val > 65000.0, "Expected large value, got {}", val);
}

// ============================================================================
// SimdBackend
// ============================================================================

#[test]
fn test_simd_backend_display() {
    assert_eq!(format!("{}", SimdBackend::Avx2), "AVX2");
    assert_eq!(format!("{}", SimdBackend::Sse2), "SSE2");
    assert_eq!(format!("{}", SimdBackend::Neon), "NEON");
    assert_eq!(format!("{}", SimdBackend::Scalar), "Scalar");
}

#[test]
fn test_simd_backend_default() {
    let backend: SimdBackend = Default::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_eq() {
    assert_eq!(SimdBackend::Avx2, SimdBackend::Avx2);
    assert_ne!(SimdBackend::Avx2, SimdBackend::Sse2);
}

#[test]
fn test_simd_backend_clone() {
    let backend = SimdBackend::Avx2;
    let cloned = backend;
    assert_eq!(backend, cloned);
}

#[test]
fn test_simd_backend_debug() {
    let debug = format!("{:?}", SimdBackend::Avx2);
    assert!(debug.contains("Avx2"));
}

// ============================================================================
// DequantStats
// ============================================================================

#[test]
fn test_dequant_stats_default() {
    let stats: DequantStats = Default::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

#[test]
fn test_dequant_stats_construction() {
    let stats = DequantStats {
        blocks_processed: 100,
        bytes_processed: 1800,
        simd_backend: SimdBackend::Avx2,
    };
    assert_eq!(stats.blocks_processed, 100);
    assert_eq!(stats.bytes_processed, 1800);
    assert_eq!(stats.simd_backend, SimdBackend::Avx2);
}

#[test]
fn test_dequant_stats_clone() {
    let stats = DequantStats {
        blocks_processed: 50,
        bytes_processed: 900,
        simd_backend: SimdBackend::Sse2,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, 50);
    assert_eq!(cloned.simd_backend, SimdBackend::Sse2);
}

#[test]
fn test_dequant_stats_debug() {
    let stats = DequantStats::default();
    let debug = format!("{:?}", stats);
    assert!(debug.contains("DequantStats"));
}

// ============================================================================
// detect_simd_backend
// ============================================================================

#[test]
fn test_detect_simd_backend_returns_valid() {
    let backend = detect_simd_backend();
    // On x86_64, should be at least SSE2 or AVX2
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            backend == SimdBackend::Avx2 || backend == SimdBackend::Sse2,
            "On x86_64, should detect AVX2 or SSE2, got {:?}",
            backend
        );
    }
    // On any platform, should return a valid variant
    let display = format!("{}", backend);
    assert!(!display.is_empty());
}

// ============================================================================
// Q8_0Block
// ============================================================================

#[test]
fn test_q8_0_block_quantize_ones() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    assert!(block.quants.iter().all(|&q| q == 127)); // All max
}

#[test]
fn test_q8_0_block_quantize_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    // Near-zero values -> minimal scale, all quants zero
    assert!(block.quants.iter().all(|&q| q == 0));
}

#[test]
fn test_q8_0_block_quantize_mixed() {
    let mut values = [0.0f32; 32];
    values[0] = 1.0;
    values[1] = -1.0;
    values[2] = 0.5;
    let block = Q8_0Block::quantize(&values);
    assert!(block.quants[0] > 0);
    assert!(block.quants[1] < 0);
    assert!(block.quants[2] > 0);
}

#[test]
fn test_q8_0_block_dequantize() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();
    for &v in dequantized.iter() {
        assert!(
            (v - 1.0).abs() < 0.02,
            "Dequantized should be near 1.0, got {}",
            v
        );
    }
}

#[test]
fn test_q8_0_block_quantization_error() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    assert!(
        error < 0.02,
        "Quantization error should be small, got {}",
        error
    );
}

#[test]
fn test_q8_0_block_relative_error() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert!(
        rel_error < 0.02,
        "Relative error should be small, got {}",
        rel_error
    );
}

#[test]
fn test_q8_0_block_relative_error_near_zero() {
    let values = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert_eq!(
        rel_error, 0.0,
        "Near-zero values should return 0 relative error"
    );
}

// ============================================================================
// Q8KSuperBlock
// ============================================================================

#[test]
fn test_q8k_superblock_quantize() {
    let values = [0.5f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0);
    // All same value -> all quants should be the same
    assert!(block.quants.iter().all(|&q| q == block.quants[0]));
}

#[test]
fn test_q8k_superblock_quantize_zeros() {
    let values = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.quants.iter().all(|&q| q == 0));
}

#[test]
fn test_q8k_superblock_dequantize() {
    let values = [0.5f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let dequantized = block.dequantize();
    for &v in dequantized.iter() {
        assert!(
            (v - 0.5).abs() < 0.01,
            "Dequantized should be near 0.5, got {}",
            v
        );
    }
}

#[test]
fn test_q8k_superblock_quantize_into() {
    let values = vec![0.3f32; 256];
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    assert!(quants.iter().all(|&q| q == quants[0]));
}

// ============================================================================
// Constants
// ============================================================================

#[test]
fn test_block_size_constant() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant() {
    assert_eq!(QK_K, 256);
}

// ============================================================================
// T-COV-95: Additional coverage for quantize/mod.rs pure functions
// ============================================================================

// --- fused_q4_0_q8_0_dot_scalar: known value computation ---

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_known_values() {
    // Q4_0 block: 2 bytes (f16 scale) + 16 bytes (quants) = 18 bytes
    // f16 for 1.0 = 0x3C00
    let mut q4_data = vec![0u8; 18];
    q4_data[0] = 0x00;
    q4_data[1] = 0x3C; // scale = 1.0

    // Set quants: byte 0 = 0x98 -> low nibble=8, high nibble=9
    // Q4_0 dequant: (nibble - 8) * scale
    // low_quant = 8 - 8 = 0, high_quant = 9 - 8 = 1
    q4_data[2] = 0x98;
    // All other quant bytes are 0 => (0 - 8) = -8 for both nibbles

    // Q8 activations: scale = 1.0, quants all = 1
    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);

    // Manually compute:
    // byte 0: low_quant = (8 - 8) = 0, act[0] = 1 => 0*1 = 0
    //         high_quant = (9 - 8) = 1, act[16] = 1 => 1*1 = 1
    // bytes 1-15: low nibble = 0-8 = -8, high = 0-8 = -8
    //   15 * (-8 * 1) + 15 * (-8 * 1) = -240
    // total integer sum = 0 + 1 + (-240) = -239
    // combined_scale = 1.0 * 1.0 = 1.0
    // result = 1.0 * (-239.0) = -239.0
    assert!(
        (result - (-239.0)).abs() < 0.01,
        "Expected -239.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_multi_block() {
    // 2 blocks = 36 bytes, 64 elements
    let mut q4_data = vec![0u8; 36];
    // Block 0: scale = 0.5 (f16 = 0x3800)
    q4_data[0] = 0x00;
    q4_data[1] = 0x38;
    // All zero quants => all nibbles = 0, offset by -8 => val = -8

    // Block 1: scale = 0
    // All zeros => no contribution

    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 64];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    // Block 0: 32 values, all = (0 - 8) * 0.5 = -4.0 as weight, act = 1
    // Sum of q4*q8 (integer) = 32 * (-8 * 1) = -256
    // scale = 0.5 * 1.0 = 0.5, block_sum = 0.5 * -256 = -128.0
    // Block 1: scale = 0, so 0 contribution
    assert!(
        (result - (-128.0)).abs() < 0.1,
        "Expected about -128.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_negative_quants() {
    let mut q4_data = vec![0u8; 18];
    q4_data[0] = 0x00;
    q4_data[1] = 0x3C; // scale = 1.0
                       // All quant bytes = 0xFF => low nibble = 15, high nibble = 15
                       // (15 - 8) = 7 for each
    for i in 2..18 {
        q4_data[i] = 0xFF;
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![-1i8; 32]; // All negative activations

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    // All q4 values = 7, all q8 = -1
    // block_sum = 32 * (7 * -1) = -224
    // result = 1.0 * (-224) = -224.0
    assert!(
        (result - (-224.0)).abs() < 0.01,
        "Expected -224.0, got {}",
        result
    );
}

// --- fused_q8_0_q8_0_dot_scalar ---

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_zero() {
    // Q8_0 weight block: 34 bytes (2 scale + 32 quants)
    let q8_weight_data = vec![0u8; 34];
    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![0i8; 32];
    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_empty() {
    let result = fused_q8_0_q8_0_dot_scalar(&[], &[], &[], 0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_known_values() {
    let mut q8_weight_data = vec![0u8; 34];
    // weight scale = 1.0 (f16 = 0x3C00)
    q8_weight_data[0] = 0x00;
    q8_weight_data[1] = 0x3C;
    // weight quants: all = 10 (as i8 bytes)
    for i in 0..32 {
        q8_weight_data[2 + i] = 10u8; // i8 value = 10
    }

    let q8_act_scales = vec![2.0f32];
    let q8_act_quants = vec![5i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // block_sum = sum(10 * 5) for 32 values = 32 * 50 = 1600
    // combined_scale = 1.0 * 2.0 = 2.0
    // result = 2.0 * 1600 = 3200.0
    assert!(
        (result - 3200.0).abs() < 1.0,
        "Expected about 3200.0, got {}",
        result
    );
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_negative_weights() {
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0] = 0x00;
    q8_weight_data[1] = 0x3C; // scale = 1.0
                              // weight quants: all = -5 (0xFB as u8)
    for i in 0..32 {
        #[allow(clippy::cast_sign_loss)]
        {
            q8_weight_data[2 + i] = (-5i8) as u8;
        }
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![3i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // block_sum = 32 * (-5 * 3) = 32 * -15 = -480
    // combined_scale = 1.0 * 1.0 = 1.0
    // result = -480.0
    assert!(
        (result - (-480.0)).abs() < 1.0,
        "Expected about -480.0, got {}",
        result
    );
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_multi_block() {
    // 2 blocks = 68 bytes, 64 elements
    let mut q8_weight_data = vec![0u8; 68];
    // Block 0: scale = 1.0, quants all = 1
    q8_weight_data[0] = 0x00;
    q8_weight_data[1] = 0x3C;
    for i in 0..32 {
        q8_weight_data[2 + i] = 1u8;
    }
    // Block 1: scale = 2.0 (f16 = 0x4000), quants all = 2
    q8_weight_data[34] = 0x00;
    q8_weight_data[35] = 0x40;
    for i in 0..32 {
        q8_weight_data[36 + i] = 2u8;
    }

    let q8_act_scales = vec![1.0f32; 2];
    let q8_act_quants = vec![1i8; 64];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 64);
    // Block 0: combined = 1.0 * 1.0 = 1.0, sum = 32 * (1*1) = 32, contrib = 32.0
    // Block 1: combined = 2.0 * 1.0 = 2.0, sum = 32 * (2*1) = 64, contrib = 128.0
    // total = 160.0
    assert!(
        (result - 160.0).abs() < 1.0,
        "Expected about 160.0, got {}",
        result
    );
}

// --- fused_q4_0_q8_0_dot_simd vs scalar parity ---

#[test]
fn test_fused_q4_0_q8_0_dot_simd_vs_scalar() {
    // Build deterministic Q4_0 data: 4 blocks = 72 bytes, 128 elements
    let mut q4_data = vec![0u8; 72]; // 4 blocks * 18 bytes
    for b in 0..4 {
        let offset = b * 18;
        // scale = 0.5 (f16 = 0x3800)
        q4_data[offset] = 0x00;
        q4_data[offset + 1] = 0x38;
        for i in 0..16 {
            q4_data[offset + 2 + i] = ((b * 17 + i * 3) % 256) as u8;
        }
    }

    let activations: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let (q8_scales, q8_quants) =
        crate::quantize::activation::quantize_activations_q8_0(&activations);

    let scalar_result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 128);
    let simd_result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, 128);

    let tol = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        tol < 0.01,
        "scalar={} simd={} rel_err={}",
        scalar_result,
        simd_result,
        tol
    );
}

// --- InterleavedQ4K::dot correctness ---

#[test]
fn test_interleaved_q4k_dot_with_nonzero_data() {
    let mut data = vec![0u8; 144];
    // d = 1.0 (f16 = 0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    // dmin = 0.0
    data[2] = 0x00;
    data[3] = 0x00;
    // Set scales: first scale byte = 1 (scale=1, min=0 for block 0)
    data[4] = 0x01;
    // qs: set to pattern 0x11 (low=1, high=1)
    for i in 0..128 {
        data[16 + i] = 0x11;
    }

    let iq = InterleavedQ4K::from_q4k(&data).expect("valid data");
    let activations = vec![1.0f32; 256];
    let result = iq.dot(&activations).expect("dot should succeed");

    // With scale[0]=1, min=0: values are d*1*1 = 1.0 for low nibble group (32 values)
    // Other blocks have scale=0 so they produce -dmin*min which is 0
    // Should produce a positive non-zero value
    assert!(
        result.abs() > 0.0,
        "Expected non-zero result, got {}",
        result
    );
}

#[test]
fn test_interleaved_q4k_dot_activation_length_mismatch() {
    let data = vec![0u8; 144];
    let iq = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![1.0f32; 128]; // Should be 256
    let result = iq.dot(&activations);
    assert!(result.is_err());
}

// --- quantize_to_q8_blocks: varied inputs ---

#[test]
fn test_quantize_to_q8_blocks_large_values() {
    let mut values = vec![0.0f32; 32];
    for i in 0..32 {
        values[i] = (i as f32 - 16.0) * 100.0;
    }
    let blocks = quantize_to_q8_blocks(&values).expect("should work");
    assert_eq!(blocks.len(), 1);

    let dequant = dequantize_q8_blocks(&blocks);
    // Verify approximate round-trip
    for (o, d) in values.iter().zip(dequant.iter()) {
        let diff = (o - d).abs();
        assert!(
            diff < blocks[0].scale * 2.0,
            "Too large error: {} vs {}",
            o,
            d
        );
    }
}

#[test]
fn test_quantize_to_q8_blocks_zeros() {
    let values = vec![0.0f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("should work");
    assert_eq!(blocks.len(), 1);
    for q in &blocks[0].quants {
        assert_eq!(*q, 0);
    }
}

// --- quantize_activations_q8k_into: edge cases ---

#[test]
fn test_quantize_activations_q8k_into_large_values() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 100.0).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
    // The max abs value should map to approximately 127
    let max_quant = quants.iter().map(|q| q.unsigned_abs()).max().unwrap_or(0);
    assert!(
        max_quant >= 126,
        "Max quant should be near 127, got {}",
        max_quant
    );
}

// --- fused_q4_0_q8_0_parallel_matvec: larger parallel path ---

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_multi_row() {
    // 4 rows of 32 elements each = 4 Q4_0 blocks (72 bytes)
    let weight_data = vec![0u8; 72]; // 4 rows * 18 bytes/row
    let activations = vec![0.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 4);
    // With zero activations, all outputs should be zero
    for &v in &output {
        assert_eq!(v, 0.0);
    }
}

// --- fused_q8_0_q8_0_parallel_matvec: success and error paths ---

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_weight_too_small() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 2);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_activation_mismatch() {
    let weight_data = vec![0u8; 34]; // 1 row * 34 bytes
    let activations = vec![1.0f32; 64]; // Wrong size (should be 32)
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_success() {
    let weight_data = vec![0u8; 34]; // 1 row, 32 elements
    let activations = vec![0.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_multi_row() {
    // 4 rows of 32 elements each => 4 * 34 = 136 bytes
    let weight_data = vec![0u8; 136];
    let activations = vec![0.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 4);
}

// --- fused_q8_0_q8_0_parallel_matvec_into: success and error paths ---

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 2, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_activation_mismatch() {
    let weight_data = vec![0u8; 34];
    let activations = vec![1.0f32; 64]; // Wrong
    let mut output = vec![0.0f32; 1];
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 1, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_output_too_small() {
    let weight_data = vec![0u8; 68]; // 2 rows * 34
    let activations = vec![0.0f32; 32];
    let mut output = vec![0.0f32; 1]; // Need 2
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 2, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_success() {
    let weight_data = vec![0u8; 34];
    let activations = vec![0.0f32; 32];
    let mut output = vec![0.0f32; 1];
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 1, &mut output);
    assert!(result.is_ok());
}

// --- f16_to_f32_lut: additional edge cases ---

#[test]
fn test_f16_to_f32_lut_half() {
    // f16 0.5 = 0x3800
    let val = f16_to_f32_lut(0x3800);
    assert!((val - 0.5).abs() < 0.001, "Expected 0.5, got {}", val);
}

#[test]
fn test_f16_to_f32_lut_two() {
    // f16 2.0 = 0x4000
    let val = f16_to_f32_lut(0x4000);
    assert!((val - 2.0).abs() < 0.001, "Expected 2.0, got {}", val);
}

#[test]
fn test_f16_to_f32_lut_infinity() {
    // f16 infinity = 0x7C00
    let val = f16_to_f32_lut(0x7C00);
    assert!(val.is_infinite() && val > 0.0);
}

#[test]
fn test_f16_to_f32_lut_neg_infinity() {
    // f16 neg infinity = 0xFC00
    let val = f16_to_f32_lut(0xFC00);
    assert!(val.is_infinite() && val < 0.0);
}

#[test]
fn test_f16_to_f32_lut_nan() {
    // f16 NaN = 0x7C01
    let val = f16_to_f32_lut(0x7C01);
    assert!(val.is_nan());
}

#[test]
fn test_f16_to_f32_lut_negative_zero() {
    let val = f16_to_f32_lut(0x8000);
    assert!(val == 0.0 && val.is_sign_negative());
}

// --- InterleavedQ4K: d and dmin extraction ---

#[test]
fn test_interleaved_q4k_extracts_d_dmin() {
    let mut data = vec![0u8; 144];
    // d = 2.0 (f16 = 0x4000)
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
    // dmin = 0.25 (f16 = 0x3400)
    data[2..4].copy_from_slice(&0x3400u16.to_le_bytes());

    let iq = InterleavedQ4K::from_q4k(&data).expect("valid data");
    assert!(
        (iq.d[0] - 2.0).abs() < 0.01,
        "d should be 2.0, got {}",
        iq.d[0]
    );
    assert!(
        (iq.dmin[0] - 0.25).abs() < 0.01,
        "dmin should be 0.25, got {}",
        iq.dmin[0]
    );
}

// --- Q8_0Block: clamping behavior ---

#[test]
fn test_q8_0_block_quantize_extreme_values() {
    // Values that require clamping
    let mut values = [0.0f32; 32];
    values[0] = 1000.0;
    values[1] = -1000.0;
    let block = Q8_0Block::quantize(&values);
    assert_eq!(block.quants[0], 127); // Clamped to max
    assert_eq!(block.quants[1], -127); // Clamped to min (symmetric)
}

// --- Q8KSuperBlock: roundtrip with varied values ---

#[test]
fn test_q8k_superblock_roundtrip_varied() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 128.0) * 0.5;
    }
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();
    for (orig, deq) in values.iter().zip(dequant.iter()) {
        let diff = (orig - deq).abs();
        assert!(
            diff < block.scale * 2.0,
            "Roundtrip error too large: orig={}, deq={}, diff={}, scale={}",
            orig,
            deq,
            diff,
            block.scale
        );
    }
}

// --- Q4_KBlock, Q5_KBlock, Q6_KBlock: clone and debug ---

#[test]
fn test_q4_k_block_clone() {
    let block = Q4_KBlock {
        d: 1.5,
        dmin: 0.3,
        scales: [7; 12],
        qs: [0xAB; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, 1.5);
    assert_eq!(cloned.dmin, 0.3);
    assert_eq!(cloned.scales, [7; 12]);
    assert_eq!(cloned.qs, [0xAB; 128]);
}

#[test]
fn test_q4_k_block_debug() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0; 12],
        qs: [0; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q4_KBlock"));
}

#[test]
fn test_q5_k_block_clone() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 0.1,
        scales: [1; 12],
        qh: [0xFF; 32],
        qs: [0x55; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, 2.0);
    assert_eq!(cloned.qh, [0xFF; 32]);
}

#[test]
fn test_q5_k_block_debug() {
    let block = Q5_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0; 12],
        qh: [0; 32],
        qs: [0; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q5_KBlock"));
}

#[test]
fn test_q6_k_block_clone() {
    let block = Q6_KBlock {
        d: 0.5,
        scales: [3; 16],
        qh: [0xAA; 64],
        qs: [0x33; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, 0.5);
    assert_eq!(cloned.scales, [3; 16]);
}

#[test]
fn test_q6_k_block_debug() {
    let block = Q6_KBlock {
        d: 1.0,
        scales: [0; 16],
        qh: [0; 64],
        qs: [0; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q6_KBlock"));
}

// --- InterleavedQ4K: clone and debug ---

#[test]
fn test_interleaved_q4k_clone_debug() {
    let data = vec![0u8; 144];
    let iq = InterleavedQ4K::from_q4k(&data).expect("valid");
    let cloned = iq.clone();
    assert_eq!(cloned.num_super_blocks, 1);
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("InterleavedQ4K"));
}

// --- Q4_0Block: clone and debug ---

#[test]
fn test_q4_0_block_clone_debug() {
    let block = Q4_0Block {
        scale: 1.0,
        quants: [0x55; 16],
    };
    let cloned = block.clone();
    assert_eq!(cloned.scale, 1.0);
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("Q4_0Block"));
}

// --- DequantStats: clone, debug ---

#[test]
fn test_dequant_stats_clone_round_trip() {
    let stats = DequantStats {
        blocks_processed: 42,
        bytes_processed: 756,
        simd_backend: SimdBackend::Neon,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, 42);
    assert_eq!(cloned.bytes_processed, 756);
    assert_eq!(cloned.simd_backend, SimdBackend::Neon);
}

// ============================================================================
// FUSED Q4_0 × Q8_0 DOT PRODUCT — AVX2 DIRECT COVERAGE TESTS
// ============================================================================
// On AVX-512 VNNI machines, the public API dispatches to the AVX512 path,
// making these AVX2 functions unreachable. Test them directly.

/// Build a Q4_0 block: 2 bytes (f16 scale) + 16 bytes (nibbles for 32 values) = 18 bytes.
fn build_q4_0_test_block(scale: f32, nibble_val: u8) -> [u8; 18] {
    let mut block = [0u8; 18];
    let scale_bits = half::f16::from_f32(scale).to_bits();
    block[0..2].copy_from_slice(&scale_bits.to_le_bytes());
    let packed = (nibble_val & 0x0F) | ((nibble_val & 0x0F) << 4);
    for i in 0..16 {
        block[2 + i] = packed;
    }
    block
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 4 blocks = 128 elements (< 256, so avx2 2-block path)
    let block = build_q4_0_test_block(1.0, 5);
    let mut q4_data = Vec::with_capacity(18 * 4);
    for _ in 0..4 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales = vec![1.0f32; 4];
    let q8_quants = vec![2i8; 128];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 128);
    let avx2 = unsafe { fused_q4_0_q8_0_dot_avx2(&q4_data, &q8_scales, &q8_quants, 128) };

    let diff = (scalar - avx2).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(diff < tol, "scalar={scalar} vs avx2={avx2}, diff={diff}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_dot_zero_quants() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4_0_test_block(1.0, 8); // nibble=8 → 8-8=0 after offset
    let mut q4_data = Vec::with_capacity(18 * 2);
    for _ in 0..2 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![0i8; 64];

    let result = unsafe { fused_q4_0_q8_0_dot_avx2(&q4_data, &q8_scales, &q8_quants, 64) };
    assert!(
        result.abs() < 1e-3,
        "zero × zero should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_dot_negative_activations() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4_0_test_block(1.0, 0xF); // nibble=15 → 15-8=7
    let mut q4_data = Vec::with_capacity(18 * 2);
    for _ in 0..2 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![-3i8; 64];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    let avx2 = unsafe { fused_q4_0_q8_0_dot_avx2(&q4_data, &q8_scales, &q8_quants, 64) };

    let diff = (scalar - avx2).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "negative act: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_4block_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 8 blocks = 256 elements (≥ 256, triggers 4-block unrolling)
    let block = build_q4_0_test_block(0.5, 3);
    let mut q4_data = Vec::with_capacity(18 * 8);
    for _ in 0..8 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales = vec![1.0f32; 8];
    let q8_quants = vec![4i8; 256];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 256);
    let avx2_4b = unsafe { fused_q4_0_q8_0_dot_avx2_4block(&q4_data, &q8_scales, &q8_quants, 256) };

    let diff = (scalar - avx2_4b).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "4block: scalar={scalar} vs avx2_4b={avx2_4b}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_4block_dot_large_dim() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 16 blocks = 512 elements
    let block = build_q4_0_test_block(1.0, 10);
    let mut q4_data = Vec::with_capacity(18 * 16);
    for _ in 0..16 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.1).collect();
    let q8_quants = vec![1i8; 512];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 512);
    let avx2_4b = unsafe { fused_q4_0_q8_0_dot_avx2_4block(&q4_data, &q8_scales, &q8_quants, 512) };

    let diff = (scalar - avx2_4b).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "large dim: scalar={scalar} vs avx2_4b={avx2_4b}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_4block_dot_varying_scales() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 12 blocks = 384 elements (tests non-power-of-2 with 4-block unrolling)
    let block = build_q4_0_test_block(2.0, 7);
    let mut q4_data = Vec::with_capacity(18 * 12);
    for _ in 0..12 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales: Vec<f32> = (0..12)
        .map(|i| if i % 2 == 0 { 1.0 } else { -0.5 })
        .collect();
    let q8_quants = vec![5i8; 384];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 384);
    let avx2_4b = unsafe { fused_q4_0_q8_0_dot_avx2_4block(&q4_data, &q8_scales, &q8_quants, 384) };

    let diff = (scalar - avx2_4b).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "varying: scalar={scalar} vs avx2_4b={avx2_4b}, diff={diff}"
    );
}
