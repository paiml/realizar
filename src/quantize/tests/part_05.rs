use crate::quantize::*;
#[test]
fn test_dequantize_q4_0_two_blocks_ext_cov() {
    let mut data = vec![0u8; 36]; // 2 blocks at 18 bytes each
                                  // First block scale = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Second block scale = 2.0
    data[18..20].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 64);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q8_0 success path
// =========================================================================

#[test]
fn test_dequantize_q8_0_one_block_ext_cov() {
    // Q8_0 block = 2 byte f16 scale + 32 byte quants = 34 bytes
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // All zero quants
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 32);
}

#[test]
fn test_dequantize_q8_0_with_values_ext_cov() {
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
    // Set some quants (as signed i8 but stored as u8)
    for i in 0..32 {
        data[2 + i] = 10; // int8 value 10
    }
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 32);
    // Each value should be approximately 10 * 0.1 = 1.0
    for v in &vals {
        assert!((v - 1.0).abs() < 0.05);
    }
}

// =========================================================================
// Extended Coverage Tests for dequantize_q4_k success path
// =========================================================================

#[test]
fn test_dequantize_q4_k_one_superblock_cov() {
    // Q4_K super-block = 144 bytes
    let data = vec![0u8; 144];
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 256);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q5_k success path
// =========================================================================

#[test]
fn test_dequantize_q5_k_one_superblock_cov() {
    // Q5_K super-block = 176 bytes
    let data = vec![0u8; 176];
    let result = dequantize_q5_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 256);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q6_k success path
// =========================================================================

#[test]
fn test_dequantize_q6_k_one_superblock_cov() {
    // Q6_K super-block = 210 bytes
    let data = vec![0u8; 210];
    let result = dequantize_q6_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 256);
}

// =========================================================================
// Extended Coverage Tests for Q4_KBlock struct
// =========================================================================

#[test]
fn test_q4_k_block_full_init_cov() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    assert!((block.d - 1.0).abs() < 1e-6);
    assert!((block.dmin - 0.5).abs() < 1e-6);
    assert_eq!(block.scales.len(), 12);
    assert_eq!(block.qs.len(), 128);
}

// =========================================================================
// Extended Coverage Tests for Q5_KBlock struct
// =========================================================================

#[test]
fn test_q5_k_block_full_init_cov() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 1.0,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    assert!((block.d - 2.0).abs() < 1e-6);
    assert!((block.dmin - 1.0).abs() < 1e-6);
    assert_eq!(block.qh.len(), 32);
}

// =========================================================================
// Extended Coverage Tests for Q6_KBlock struct
// =========================================================================

#[test]
fn test_q6_k_block_full_init_cov() {
    let block = Q6_KBlock {
        d: 3.0,
        scales: [0i8; 16],
        qh: [0u8; 64],
        qs: [0u8; 128],
    };
    assert!((block.d - 3.0).abs() < 1e-6);
    assert_eq!(block.scales.len(), 16);
    assert_eq!(block.qh.len(), 64);
    assert_eq!(block.qs.len(), 128);
}

// =========================================================================
// Extended Coverage Tests for f16_to_f32
// =========================================================================

#[test]
fn test_f16_to_f32_zero_ext_cov() {
    let zero = f16_to_f32(0);
    assert!((zero - 0.0).abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_negative_zero_ext_cov() {
    // Negative zero: sign bit set, everything else zero
    let neg_zero = f16_to_f32(0x8000);
    assert!(neg_zero == 0.0 || neg_zero == -0.0);
}

#[test]
fn test_f16_to_f32_one_ext_cov() {
    let one = half::f16::from_f32(1.0).to_bits();
    let result = f16_to_f32(one);
    assert!((result - 1.0).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_negative_one_ext_cov() {
    let neg_one = half::f16::from_f32(-1.0).to_bits();
    let result = f16_to_f32(neg_one);
    assert!((result - (-1.0)).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_infinity_ext_cov() {
    // Positive infinity: exp=31, mantissa=0
    let pos_inf = 0x7C00;
    assert!(f16_to_f32(pos_inf).is_infinite());
    assert!(f16_to_f32(pos_inf) > 0.0);
}

#[test]
fn test_f16_to_f32_neg_infinity_ext_cov() {
    // Negative infinity: sign=1, exp=31, mantissa=0
    let neg_inf = 0xFC00;
    assert!(f16_to_f32(neg_inf).is_infinite());
    assert!(f16_to_f32(neg_inf) < 0.0);
}

#[test]
fn test_f16_to_f32_nan_ext_cov() {
    // NaN: exp=31, mantissa!=0
    let nan = 0x7C01;
    assert!(f16_to_f32(nan).is_nan());
}

#[test]
fn test_f16_to_f32_subnormal_ext_cov() {
    // Subnormal: exp=0, mantissa!=0
    let subnormal = 0x0001; // Smallest positive subnormal
    let result = f16_to_f32(subnormal);
    assert!(result > 0.0);
    assert!(result < 1e-5);
}

#[test]
fn test_f16_to_f32_neg_subnormal_ext_cov() {
    // Negative subnormal
    let neg_subnormal = 0x8001;
    let result = f16_to_f32(neg_subnormal);
    assert!(result < 0.0);
}

#[test]
fn test_f16_to_f32_max_normal_ext_cov() {
    // Max normal: exp=30, mantissa=all 1s (just below infinity)
    let max_normal = 0x7BFF;
    let result = f16_to_f32(max_normal);
    assert!(result > 60000.0);
    assert!(result < 70000.0);
}

// =========================================================================
// Extended Coverage Tests for f16_to_f32_lut
// =========================================================================

#[test]
fn test_f16_to_f32_lut_zero_ext_cov() {
    let result = f16_to_f32_lut(0);
    assert!((result - 0.0).abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_lut_one_ext_cov() {
    let one_bits = half::f16::from_f32(1.0).to_bits();
    let result = f16_to_f32_lut(one_bits);
    assert!((result - 1.0).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_lut_all_values_ext_cov() {
    // Test that LUT returns reasonable values for all inputs
    for i in 0..1000u16 {
        let val = f16_to_f32_lut(i);
        assert!(!val.is_nan() || i >= 0x7C01);
    }
}

// =========================================================================
// Extended Coverage Tests for Q8_0Block methods
// =========================================================================

#[test]
fn test_q8_0_block_quantize_zeros_ext_cov() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    // Scale should be minimal for zeros
    assert!(block.scale < 0.01);
}

#[test]
fn test_q8_0_block_quantize_uniform_ext_cov() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    // All quants should be 127 (max) since all values are equal to max
    assert!(block.quants.iter().all(|&q| q == 127));
}

#[test]
fn test_q8_0_block_quantize_negative_ext_cov() {
    let values = [-1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    // All quants should be -127 (min + 1) since all values are negative max
    assert!(block.quants.iter().all(|&q| q == -127));
}

#[test]
fn test_q8_0_block_quantize_mixed_ext_cov() {
    let mut values = [0.0f32; 32];
    for i in 0..32 {
        values[i] = (i as f32 - 15.5) * 0.1;
    }
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
}

#[test]
fn test_q8_0_block_dequantize_ext_cov() {
    let block = Q8_0Block {
        scale: 0.1,
        quants: [10i8; 32],
    };
    let result = block.dequantize();
    assert_eq!(result.len(), 32);
    for val in result {
        assert!((val - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_q8_0_block_quantization_error_cov2() {
    let original = [1.0f32; 32];
    let block = Q8_0Block::quantize(&original);
    let error = block.quantization_error(&original);
    // Error should be small for uniform values
    assert!(error < 0.02);
}

#[test]
fn test_q8_0_block_relative_error_zeros_cov2() {
    let original = [0.0f32; 32];
    let block = Q8_0Block::quantize(&original);
    let error = block.relative_error(&original);
    assert!((error - 0.0).abs() < 1e-6);
}

#[test]
fn test_q8_0_block_relative_error_nonzero_cov() {
    let mut original = [0.0f32; 32];
    for i in 0..32 {
        original[i] = (i as f32) * 0.1;
    }
    let block = Q8_0Block::quantize(&original);
    let error = block.relative_error(&original);
    // Relative error should be small
    assert!(error < 0.1);
}

#[test]
fn test_q8_0_block_clone_cov2() {
    let block = Q8_0Block {
        scale: 0.5,
        quants: [1i8; 32],
    };
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
    assert_eq!(block.quants, cloned.quants);
}

#[test]
fn test_q8_0_block_debug_cov2() {
    let block = Q8_0Block {
        scale: 0.5,
        quants: [0i8; 32],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("scale"));
    assert!(debug_str.contains("quants"));
}

// =========================================================================
// Extended Coverage Tests for Q4_0Block
// =========================================================================

#[test]
fn test_q4_0_block_clone_cov2() {
    let block = Q4_0Block {
        scale: 2.0,
        quants: [0x12u8; 16],
    };
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
    assert_eq!(block.quants, cloned.quants);
}

#[test]
fn test_q4_0_block_debug_cov2() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0u8; 16],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("scale"));
    assert!(debug_str.contains("quants"));
}

// =========================================================================
// Extended Coverage Tests for Q8KSuperBlock
// =========================================================================

#[test]
fn test_q8k_superblock_quantize_zeros_ext_cov() {
    let values = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale < 0.01);
}

#[test]
fn test_q8k_superblock_quantize_uniform_ext_cov() {
    let values = [1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.quants.iter().all(|&q| q == 127));
}

#[test]
fn test_q8k_superblock_quantize_mixed_ext_cov() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 127.5) * 0.01;
    }
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0);
}

#[test]
fn test_q8k_superblock_dequantize_ext_cov() {
    let block = Q8KSuperBlock {
        scale: 0.1,
        quants: [10i8; 256],
    };
    let result = block.dequantize();
    assert_eq!(result.len(), 256);
    for val in result {
        assert!((val - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_q8k_superblock_quantize_into_ext_cov() {
    let values = [1.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    assert!(quants.iter().all(|&q| q == 127));
}

#[test]
fn test_q8k_superblock_clone_ext_cov() {
    let block = Q8KSuperBlock {
        scale: 0.5,
        quants: [1i8; 256],
    };
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
}

#[test]
fn test_q8k_superblock_debug_ext_cov() {
    let block = Q8KSuperBlock {
        scale: 1.0,
        quants: [0i8; 256],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("scale"));
}

// =========================================================================
// Extended Coverage Tests for quantize_activations_q8k_into
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_success_ext_cov() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

#[test]
fn test_quantize_activations_q8k_into_not_multiple_ext_cov() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_scales_too_small_ext_cov() {
    let activations = vec![1.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 1]; // Only 1 scale, need 2
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_quants_too_small_ext_cov() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_multi_superblock_ext_cov() {
    let activations = vec![1.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

// =========================================================================
// Extended Coverage Tests for quantize_to_q8_blocks
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_success_ext_cov() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_not_multiple_ext_cov() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_single_block_ext_cov() {
    let values = vec![1.0f32; 32];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 1);
}

#[test]
fn test_quantize_to_q8_blocks_empty_ext_cov() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q8_blocks
// =========================================================================

#[test]
fn test_dequantize_q8_blocks_ext_cov() {
    let blocks = vec![
        Q8_0Block {
            scale: 0.1,
            quants: [10i8; 32],
        },
        Q8_0Block {
            scale: 0.2,
            quants: [5i8; 32],
        },
    ];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_dequantize_q8_blocks_empty_ext_cov() {
    let blocks: Vec<Q8_0Block> = vec![];
    let result = dequantize_q8_blocks(&blocks);
    assert!(result.is_empty());
}

// =========================================================================
// Extended Coverage Tests for InterleavedQ4K
// =========================================================================

#[test]
fn test_interleaved_q4k_from_q4k_success_ext_cov() {
    let data = vec![0u8; 144]; // 1 superblock
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("quantization failed");
    assert_eq!(interleaved.num_super_blocks, 1);
}

#[test]
fn test_interleaved_q4k_from_q4k_invalid_ext_cov() {
    let data = vec![0u8; 143]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_num_values_ext_cov() {
    let data = vec![0u8; 288]; // 2 superblocks
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    assert_eq!(interleaved.num_values(), 512);
}

#[test]
fn test_interleaved_q4k_clone_ext_cov() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let cloned = interleaved.clone();
    assert_eq!(interleaved.num_super_blocks, cloned.num_super_blocks);
}

#[test]
fn test_interleaved_q4k_debug_ext_cov() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let debug_str = format!("{:?}", interleaved);
    assert!(debug_str.contains("num_super_blocks"));
}

#[test]
fn test_interleaved_q4k_dot_success_ext_cov() {
    let data = vec![0u8; 144]; // 1 superblock
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 256];
    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
}

#[test]
fn test_interleaved_q4k_dot_wrong_size_ext_cov() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = interleaved.dot(&activations);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests for extract_scale_min_from_slice
// =========================================================================

#[test]
fn test_extract_scale_min_from_slice_even_idx_ext_cov() {
    let scales = [31u8, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0];
    let (scale, min) = extract_scale_min_from_slice(&scales, 0);
    assert!((scale - 31.0).abs() < 1e-6);
    assert!((min - 15.0).abs() < 1e-6);
}

#[test]
fn test_extract_scale_min_from_slice_odd_idx_ext_cov() {
    let mut scales = [0u8; 12];
    scales[0] = 0b11_000000; // high 2 bits contribute to scale[1]
    scales[2] = 0b0000_0011; // low 4 bits contribute to scale[1]
    let (scale, _) = extract_scale_min_from_slice(&scales, 1);
    // scale = (scales[0] >> 6) | ((scales[2] & 0x0F) << 2)
    // = (0b11_000000 >> 6) | ((0x03 & 0x0F) << 2) = 3 | 12 = 15
    assert!((scale - 15.0).abs() < 1e-6);
}

// =========================================================================
// Extended Coverage Tests for Q4_KBlock/Q5_KBlock/Q6_KBlock Clone/Debug
// =========================================================================

#[test]
fn test_q4_k_block_clone_debug_ext_cov() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q4_KBlock"));
}

#[test]
fn test_q5_k_block_clone_debug_ext_cov() {
    let block = Q5_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q5_KBlock"));
}

#[test]
fn test_q6_k_block_clone_debug_ext_cov() {
    let block = Q6_KBlock {
        d: 1.0,
        scales: [0i8; 16],
        qh: [0u8; 64],
        qs: [0u8; 128],
    };
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q6_KBlock"));
}

// =========================================================================
// Extended Coverage Tests for constants
// =========================================================================

#[test]
fn test_block_size_constant_ext_cov() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant_ext_cov() {
    assert_eq!(QK_K, 256);
}

// =========================================================================
// Extended Coverage Tests: DequantStats and SimdBackend
// =========================================================================

#[test]
fn test_dequant_stats_default_ext_cov() {
    let stats = DequantStats::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

#[test]
fn test_dequant_stats_clone_ext_cov() {
    let stats = DequantStats {
        blocks_processed: 100,
        bytes_processed: 5000,
        simd_backend: SimdBackend::Avx2,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, 100);
    assert_eq!(cloned.bytes_processed, 5000);
}

#[test]
fn test_dequant_stats_debug_ext_cov() {
    let stats = DequantStats {
        blocks_processed: 42,
        bytes_processed: 1024,
        simd_backend: SimdBackend::Sse2,
    };
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("DequantStats"));
    assert!(debug_str.contains("42"));
    assert!(debug_str.contains("1024"));
}

#[test]
fn test_simd_backend_all_variants_ext_cov() {
    let variants = [
        SimdBackend::Avx2,
        SimdBackend::Sse2,
        SimdBackend::Neon,
        SimdBackend::Scalar,
    ];
    for v in variants {
        let _ = format!("{:?}", v);
    }
}

#[test]
fn test_simd_backend_display_all_ext_cov() {
    assert_eq!(format!("{}", SimdBackend::Avx2), "AVX2");
    assert_eq!(format!("{}", SimdBackend::Sse2), "SSE2");
    assert_eq!(format!("{}", SimdBackend::Neon), "NEON");
    assert_eq!(format!("{}", SimdBackend::Scalar), "Scalar");
}

#[test]
fn test_simd_backend_clone_ext_cov() {
    let backend = SimdBackend::Avx2;
    let cloned = backend;
    assert_eq!(backend, cloned);
}

#[test]
fn test_simd_backend_eq_ext_cov() {
    assert_eq!(SimdBackend::Scalar, SimdBackend::Scalar);
    assert_ne!(SimdBackend::Scalar, SimdBackend::Avx2);
    assert_ne!(SimdBackend::Sse2, SimdBackend::Neon);
}

#[test]
fn test_simd_backend_default_ext_cov() {
    let backend: SimdBackend = SimdBackend::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_copy_ext_cov() {
    let backend = SimdBackend::Neon;
    let copied = backend;
    assert_eq!(copied, SimdBackend::Neon);
}

// =========================================================================
// Extended Coverage Tests: Error paths for dequantize functions
// =========================================================================

#[test]
fn test_dequantize_q4_0_empty_input_ext_cov() {
    let result = dequantize_q4_0(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_empty_input_ext_cov() {
    let result = dequantize_q8_0(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_f16_empty_input_ext_cov() {
    let result = dequantize_f16(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_1_empty_input_ext_cov() {
    let result = dequantize_q4_1(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q5_0_empty_input_ext_cov() {
    let result = dequantize_q5_0(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q5_1_empty_input_ext_cov() {
    let result = dequantize_q5_1(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_k_empty_input_ext_cov() {
    let result = dequantize_q4_k(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q5_k_empty_input_ext_cov() {
    let result = dequantize_q5_k(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q6_k_empty_input_ext_cov() {
    let result = dequantize_q6_k(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Extended Coverage Tests: fused matvec error paths
// =========================================================================

#[test]
fn test_fused_q4k_parallel_matvec_into_invalid_output_ext_cov() {
    let q4k_data = vec![0u8; 144]; // 1 super-block
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let mut output = vec![0.0f32; 0]; // Wrong size
    let result =
        fused_q4k_q8k_parallel_matvec_into(&q4k_data, &scales, &quants, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_parallel_matvec_into_invalid_input_ext_cov() {
    let q4k_data = vec![0u8; 100]; // Invalid size
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let mut output = vec![0.0f32; 1];
    let result =
        fused_q4k_q8k_parallel_matvec_into(&q4k_data, &scales, &quants, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_parallel_matvec_into_invalid_input_ext_cov() {
    let data = vec![0u8; 100]; // Invalid
    let activations = vec![0.0f32; 256];
    let mut output = vec![0.0f32; 1];
    let result = fused_q5k_parallel_matvec_into(&data, &activations, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_parallel_matvec_into_dim_mismatch_ext_cov() {
    let data = vec![0u8; 176]; // Valid for 1 row
    let activations = vec![0.0f32; 128]; // Wrong dimension
    let mut output = vec![0.0f32; 1];
    let result = fused_q5k_parallel_matvec_into(&data, &activations, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_parallel_matvec_into_invalid_input_ext_cov() {
    let data = vec![0u8; 100]; // Invalid
    let activations = vec![0.0f32; 256];
    let mut output = vec![0.0f32; 1];
    let result = fused_q6k_parallel_matvec_into(&data, &activations, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_parallel_matvec_into_dim_mismatch_ext_cov() {
    let data = vec![0u8; 210]; // Valid for 1 row
    let activations = vec![0.0f32; 128]; // Wrong dimension
    let mut output = vec![0.0f32; 1];
    let result = fused_q6k_parallel_matvec_into(&data, &activations, 1, 256, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: fused dot error paths
// =========================================================================

#[test]
fn test_fused_q6k_dot_invalid_data_ext_cov() {
    let data = vec![0u8; 100]; // Invalid size
    let activations = vec![0.0f32; 256];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_dot_invalid_data_ext_cov() {
    let data = vec![0u8; 100]; // Invalid size
    let activations = vec![0.0f32; 256];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_dot_simd_basic_ext_cov() {
    let data = vec![0u8; 210]; // Valid Q6_K super-block
    let activations = vec![0.0f32; 256];
    let result = fused_q6k_dot_simd(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q5k_dot_simd_basic_ext_cov() {
    let data = vec![0u8; 176]; // Valid Q5_K super-block
    let activations = vec![0.0f32; 256];
    let result = fused_q5k_dot_simd(&data, &activations);
    assert!(result.is_ok());
}

// =========================================================================
// Extended Coverage Tests: Q8K/Q4K fused operations
// =========================================================================

#[test]
fn test_fused_q4k_q8k_dot_simd_dim_mismatch_ext_cov() {
    let q4k_data = vec![0u8; 144]; // 1 super-block
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 128]; // Wrong size
    let result = fused_q4k_q8k_dot_simd(&q4k_data, &scales, &quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_data_ext_cov() {
    let q4k_data = vec![0u8; 100]; // Invalid
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: parallel dequantize
// =========================================================================

#[test]
fn test_dequantize_q4_k_parallel_empty_ext_cov() {
    let result = dequantize_q4_k_parallel(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_k_simd_empty_ext_cov() {
    let result = dequantize_q4_k_simd(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_parallel_empty_ext_cov() {
    let result = dequantize_q8_0_parallel(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_simd_empty_ext_cov() {
    let result = dequantize_q8_0_simd(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Extended Coverage Tests: Q4_0/Q8_0 parallel matvec
// =========================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_invalid_weights_ext_cov() {
    let weights = vec![0u8; 10]; // Invalid
    let activations = vec![1.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_dim_mismatch_ext_cov() {
    let weights = vec![0u8; 18]; // 1 block for 32 values
    let activations = vec![1.0f32; 16]; // Wrong size
    let result = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_invalid_weights_ext_cov() {
    let weights = vec![0u8; 10]; // Invalid
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&weights, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_invalid_weights_ext_cov() {
    let weights = vec![0u8; 10]; // Invalid
    let activations = vec![1.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weights, &activations, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_invalid_weights_ext_cov() {
    let weights = vec![0u8; 10]; // Invalid
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];
    let result = fused_q8_0_q8_0_parallel_matvec_into(&weights, &activations, 32, 32, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: FFN up/gate fused operations
// =========================================================================

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_invalid_up_ext_cov() {
    let up_data = vec![0u8; 100]; // Invalid
    let gate_data = vec![0u8; 144];
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];
    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_data,
        &gate_data,
        &scales,
        &quants,
        1,
        256,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_invalid_gate_ext_cov() {
    let up_data = vec![0u8; 144];
    let gate_data = vec![0u8; 100]; // Invalid
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];
    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_data,
        &gate_data,
        &scales,
        &quants,
        1,
        256,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: tiled matvec error paths
// =========================================================================

#[test]
fn test_fused_q4k_tiled_matvec_invalid_data_ext_cov() {
    let data = vec![0u8; 100]; // Invalid
    let activations = vec![0.0f32; 256];
    let result = fused_q4k_tiled_matvec(&data, &activations, 1, 256, Some(64));
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_tiled_matvec_dim_mismatch_ext_cov() {
    let data = vec![0u8; 144]; // Valid for 1 row
    let activations = vec![0.0f32; 128]; // Wrong dimension
    let result = fused_q4k_tiled_matvec(&data, &activations, 1, 256, Some(64));
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_parallel_matvec_invalid_data_ext_cov() {
    let data = vec![0u8; 100]; // Invalid
    let activations = vec![0.0f32; 256];
    let result = fused_q4k_parallel_matvec(&data, &activations, 1, 256);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_parallel_matvec_dim_mismatch_ext_cov() {
    let data = vec![0u8; 144]; // Valid for 1 row
    let activations = vec![0.0f32; 128]; // Wrong dimension
    let result = fused_q4k_parallel_matvec(&data, &activations, 1, 256);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: rmsnorm fused operations
// =========================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_weight_too_small_ext_cov() {
    let input = vec![1.0f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let up_weights = vec![0u8; 10]; // Too small
    let gate_weights = vec![0u8; 10];

    let result = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        0.00001,
        &up_weights,
        &gate_weights,
        64,
        1,
    );
    assert!(result.is_err());
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_partial_block_ext_cov() {
    let input = vec![1.0f32; 48]; // Not multiple of 32
    let norm_weight = vec![1.0f32; 48];
    let eps = 1e-6;
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);
    assert_eq!(scales.len(), 2); // 2 blocks needed for 48 values
    assert_eq!(quants.len(), 64); // 2 blocks * 32 quants
}

// =========================================================================
// Extended Coverage Tests: rope rotation edge cases
// =========================================================================

#[test]
fn test_apply_rope_rotation_simd_empty_ext_cov() {
    let mut x1: Vec<f32> = vec![];
    let mut x2: Vec<f32> = vec![];
    let cos_vals: Vec<f32> = vec![];
    let sin_vals: Vec<f32> = vec![];
    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);
    assert!(x1.is_empty());
    assert!(x2.is_empty());
}

#[test]
fn test_apply_rope_rotation_simd_single_ext_cov() {
    let mut x1 = vec![1.0f32];
    let mut x2 = vec![0.0f32];
    let cos_vals = vec![1.0f32];
    let sin_vals = vec![0.0f32];
    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);
    assert!((x1[0] - 1.0).abs() < 1e-6);
}

// =========================================================================
// Extended Coverage Tests: swiglu and softmax edge cases
// =========================================================================

#[test]
fn test_fused_swiglu_simd_empty_ext_cov() {
    let mut gate: Vec<f32> = vec![];
    let up: Vec<f32> = vec![];
    fused_swiglu_simd(&mut gate, &up);
    assert!(gate.is_empty());
}

#[test]
fn test_fused_swiglu_simd_single_ext_cov() {
    let mut gate = vec![0.0f32];
    let up = vec![1.0f32];
    fused_swiglu_simd(&mut gate, &up);
    // silu(0) = 0 / (1 + exp(-0)) = 0 / 2 = 0
    // result = 0 * 1 = 0
    assert!((gate[0]).abs() < 0.01);
}

#[test]
fn test_softmax_simd_empty_ext_cov() {
    let mut x: Vec<f32> = vec![];
    softmax_simd(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_simd_single_ext_cov() {
    let mut x = vec![5.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-6); // Single element should be 1.0
}

// =========================================================================
// Extended Coverage Tests: Q8_0Block and Q4_0Block operations
// =========================================================================

#[test]
fn test_q8_0_block_quantize_large_values_ext_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32) * 10.0);
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();
    let error = values
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 32.0;
    assert!(error < 2.0); // Acceptable quantization error
}

#[test]
fn test_q8_0_block_quantize_negative_large_ext_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| -(i as f32) * 5.0);
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();
    assert_eq!(dequant.len(), 32);
}

#[test]
fn test_interleaved_q4k_dot_multiple_blocks_ext_cov() {
    let q4k_data = vec![0u8; 288]; // 2 super-blocks
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");
    let activations = vec![0.0f32; 512];
    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
}

// =========================================================================
// Extended Coverage Tests: Q8K SuperBlock operations
// =========================================================================

#[test]
fn test_q8k_superblock_quantize_large_ext_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32) * 0.5);
    let sb = Q8KSuperBlock::quantize(&values);
    let dequant = sb.dequantize();
    assert_eq!(dequant.len(), 256);
}

#[test]
fn test_q8k_superblock_quantize_all_same_ext_cov() {
    let values = [7.5f32; 256];
    let sb = Q8KSuperBlock::quantize(&values);
    assert!(sb.scale > 0.0);
    assert_eq!(sb.quants.len(), 256);
}

#[test]
fn test_q8k_superblock_quantize_into_multiple_ext_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| if i < 128 { 1.0 } else { -1.0 });
    let values_slice = values.as_slice();
    let mut out_scale = 0.0f32;
    let mut out_quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(values_slice, &mut out_scale, &mut out_quants);
    assert!(out_scale > 0.0);
}

// =========================================================================
// Deep Coverage Tests: InterleavedQ4K
// =========================================================================

#[test]
fn test_interleaved_q4k_from_q4k_invalid_length_cov() {
    // Not a multiple of 144
    let data = vec![0u8; 143];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_from_q4k_empty_deep3() {
    let result = InterleavedQ4K::from_q4k(&[]);
    assert!(result.is_ok());
    let iq = result.expect("quantization failed");
    assert_eq!(iq.num_super_blocks, 0);
    assert_eq!(iq.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_num_values_cov() {
    // One super-block = 256 values
    let data = vec![0u8; 144];
    let iq = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    assert_eq!(iq.num_values(), 256);
}

#[test]
fn test_interleaved_q4k_dot_dim_mismatch_cov() {
    let data = vec![0u8; 144];
    let iq = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = iq.dot(&activations);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_dot_valid_cov() {
    // Create valid Q4_K data
    let mut data = vec![0u8; 144];
    // Set d=1.0 as f16
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    data[0] = d_bytes[0];
    data[1] = d_bytes[1];
    // dmin = 0.0
    data[2] = 0;
    data[3] = 0;

    let iq = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 256];
    let result = iq.dot(&activations);
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: Q8_0Block additional methods
// =========================================================================

#[test]
fn test_q8_0_block_quantize_all_zeros_cov() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    // Scale should be minimal (1/127)
    assert!(block.scale > 0.0);
    assert!(block.scale < 0.01);
}

#[test]
fn test_q8_0_block_relative_error_near_zero_cov() {
    let values = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_err = block.relative_error(&values);
    // Should return 0.0 for near-zero inputs
    assert_eq!(rel_err, 0.0);
}

#[test]
fn test_q8_0_block_quantization_error_deep2() {
    let values: [f32; 32] = std::array::from_fn(|i| i as f32 - 16.0);
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Error should be small for linear values
    assert!(error < 0.5);
}

#[test]
fn test_q8_0_block_dequantize_roundtrip_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 15.5) * 2.0);
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();
    // Check roundtrip error is reasonable
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        let err = (orig - deq).abs();
        assert!(err < 1.0, "Error too large: {} vs {}", orig, deq);
    }
}

// =========================================================================
// Deep Coverage Tests: Q8KSuperBlock
// =========================================================================

#[test]
fn test_q8k_superblock_quantize_alternating_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| if i % 2 == 0 { 10.0 } else { -10.0 });
    let sb = Q8KSuperBlock::quantize(&values);
    assert!(sb.scale > 0.0);
    // Check that quants alternate in sign
    assert!(sb.quants[0] > 0);
    assert!(sb.quants[1] < 0);
}

#[test]
fn test_q8k_superblock_quantize_increasing_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) / 10.0);
    let sb = Q8KSuperBlock::quantize(&values);
    assert!(sb.scale > 0.0);
    // First quant should be negative, last should be positive
    assert!(sb.quants[0] < 0);
    assert!(sb.quants[255] > 0);
}

// =========================================================================
// Deep Coverage Tests: quantize_to_q8_blocks
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_exact_blocks_cov() {
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantization failed");
    assert_eq!(blocks.len(), 2); // 64 values = 2 blocks
}

#[test]
fn test_quantize_to_q8_blocks_partial_block_cov() {
    // Function requires multiple of 32, so 50 should error
    let values: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let result = quantize_to_q8_blocks(&values);
    // Should error because 50 is not a multiple of 32
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_empty_deep2() {
    let values: Vec<f32> = vec![];
    let blocks = quantize_to_q8_blocks(&values).expect("quantization failed");
    assert!(blocks.is_empty());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q8_blocks
// =========================================================================

#[test]
fn test_dequantize_q8_blocks_roundtrip_deep2() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantization failed");
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);
    // Check roundtrip error
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        let err = (orig - deq).abs();
        assert!(err < 1.0);
    }
}

// =========================================================================
// Deep Coverage Tests: f16_to_f32
// =========================================================================

#[test]
fn test_f16_to_f32_special_values_cov() {
    // Zero
    assert_eq!(f16_to_f32(0x0000), 0.0);
    // One (f16 representation of 1.0)
    let one = half::f16::from_f32(1.0).to_bits();
    assert!((f16_to_f32(one) - 1.0).abs() < 1e-3);
    // Negative one
    let neg_one = half::f16::from_f32(-1.0).to_bits();
    assert!((f16_to_f32(neg_one) - (-1.0)).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_small_values_cov() {
    let small = half::f16::from_f32(0.001).to_bits();
    let result = f16_to_f32(small);
    assert!((result - 0.001).abs() < 1e-4);
}

// =========================================================================
// Deep Coverage Tests: dequantize_f16
// =========================================================================

#[test]
fn test_dequantize_f16_valid_deep2() {
    // 4 bytes = 2 f16 values
    let one = half::f16::from_f32(1.0).to_le_bytes();
    let two = half::f16::from_f32(2.0).to_le_bytes();
    let data = [one[0], one[1], two[0], two[1]];
    let result = dequantize_f16(&data).expect("quantization failed");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 1e-3);
    assert!((result[1] - 2.0).abs() < 1e-3);
}

#[test]
fn test_dequantize_f16_odd_length_cov() {
    let data = [0u8; 3]; // Not a multiple of 2
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q4_1
// =========================================================================

#[test]
fn test_dequantize_q4_1_valid_cov() {
    // Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes quants = 20 bytes
    let mut data = vec![0u8; 20];
    // scale = 1.0 as f16
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    data[0] = scale[0];
    data[1] = scale[1];
    // min = 0.0 as f16
    data[2] = 0;
    data[3] = 0;
    // quants: all zeros

    let result = dequantize_q4_1(&data).expect("quantization failed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q4_1_invalid_length_deep2() {
    let data = vec![0u8; 19]; // Not a multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q5_0
// =========================================================================

#[test]
fn test_dequantize_q5_0_valid_cov() {
    // Q5_0 block: 2 bytes scale + 4 bytes high bits + 16 bytes low quants = 22 bytes
    let mut data = vec![0u8; 22];
    // scale = 1.0 as f16
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    data[0] = scale[0];
    data[1] = scale[1];

    let result = dequantize_q5_0(&data).expect("quantization failed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_0_invalid_length_deep2() {
    let data = vec![0u8; 21]; // Not a multiple of 22
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q5_1
// =========================================================================

#[test]
fn test_dequantize_q5_1_valid_deep2() {
    // Q5_1 block: 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes low = 24 bytes
    let mut data = vec![0u8; 24];
    // scale = 1.0 as f16
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    data[0] = scale[0];
    data[1] = scale[1];

    let result = dequantize_q5_1(&data).expect("quantization failed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_1_invalid_length_deep2() {
    let data = vec![0u8; 23]; // Not a multiple of 24
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q5_k
// =========================================================================

#[test]
fn test_dequantize_q5_k_valid_cov() {
    // Q5_K super-block: 176 bytes
    let data = vec![0u8; 176];
    let result = dequantize_q5_k(&data).expect("quantization failed");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q5_k_invalid_length_cov() {
    let data = vec![0u8; 175]; // Not a multiple of 176
    let result = dequantize_q5_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: dequantize_q6_k
// =========================================================================

#[test]
fn test_dequantize_q6_k_valid_cov() {
    // Q6_K super-block: 210 bytes
    let data = vec![0u8; 210];
    let result = dequantize_q6_k(&data).expect("quantization failed");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q6_k_invalid_length_cov() {
    let data = vec![0u8; 209]; // Not a multiple of 210
    let result = dequantize_q6_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: fused dot products
// =========================================================================

#[test]
fn test_fused_q4k_dot_valid_cov() {
    let data = vec![0u8; 144]; // One super-block
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_invalid_data_cov() {
    let data = vec![0u8; 143]; // Invalid length
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_dim_mismatch_cov() {
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_valid_cov() {
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q6k_dot_valid_cov() {
    let data = vec![0u8; 210];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q5k_dot_valid_cov() {
    let data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_q8_dot
// =========================================================================

#[test]
fn test_fused_q4k_q8_dot_valid_cov() {
    let q4k_data = vec![0u8; 144];
    let q8_blocks: Vec<Q8_0Block> = (0..8)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [0i8; 32],
        })
        .collect();
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8_dot_invalid_q4k_cov() {
    let q4k_data = vec![0u8; 143]; // Invalid length
    let q8_blocks: Vec<Q8_0Block> = (0..8)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [0i8; 32],
        })
        .collect();
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_q8k_dot
// =========================================================================

#[test]
fn test_fused_q4k_q8k_dot_valid_cov() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1]; // One scale per super-block
    let q8k_quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_cov() {
    let q4k_data = vec![0u8; 143];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

// =========================================================================
// Deep Coverage Tests: quantize_activations_q8k_into
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_valid_cov() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();
    let mut scales = vec![0.0f32; 1]; // One scale per 256 values
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
}

#[test]
fn test_quantize_activations_q8k_into_zeros_cov() {
    let activations = vec![0.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    // Should succeed even with zeros
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: quantize_activations_q8_0
// =========================================================================

// =========================================================================
// PMAT-097: Reference Comparison Tests for Q4_1, Q5_0, Q5_1
// These tests verify CORRECTNESS of dequantized values, not just code coverage.
// Based on GGML/llama.cpp reference implementation and candle layout specification.
// =========================================================================

/// Q4_1 Reference Test: Verify candle layout with known values
/// GGUF candle layout for Q4_1:
/// - Positions 0-15: low nibbles (byte & 0xF) from bytes 0-15
/// - Positions 16-31: high nibbles (byte >> 4) from bytes 0-15
///
/// Formula: value = scale * nibble + min
#[test]
fn test_q4_1_candle_layout_correctness() {
    // Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes quants = 20 bytes
    let mut block = vec![0u8; 20];

    // Scale = 1.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Min = 0.0
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    // Set quants with known pattern:
    // byte[0] = 0x10 -> low=0, high=1
    // byte[1] = 0x32 -> low=2, high=3
    // byte[2] = 0x54 -> low=4, high=5
    // byte[3] = 0x76 -> low=6, high=7
    block[4] = 0x10; // low=0, high=1
    block[5] = 0x32; // low=2, high=3
    block[6] = 0x54; // low=4, high=5
    block[7] = 0x76; // low=6, high=7
    block[8] = 0x98; // low=8, high=9
    block[9] = 0xBA; // low=10 (0xA), high=11 (0xB)
    block[10] = 0xDC; // low=12 (0xC), high=13 (0xD)
    block[11] = 0xFE; // low=14 (0xE), high=15 (0xF)
                      // Remaining bytes are zeros

    let result = dequantize_q4_1(&block).expect("dequantization failed");

    assert_eq!(result.len(), 32);

    // Verify candle layout - low nibbles at positions 0-15
    assert!(
        (result[0] - 0.0).abs() < 1e-5,
        "pos 0: expected 0.0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 2.0).abs() < 1e-5,
        "pos 1: expected 2.0, got {}",
        result[1]
    );
    assert!(
        (result[2] - 4.0).abs() < 1e-5,
        "pos 2: expected 4.0, got {}",
        result[2]
    );
    assert!(
        (result[3] - 6.0).abs() < 1e-5,
        "pos 3: expected 6.0, got {}",
        result[3]
    );
    assert!(
        (result[4] - 8.0).abs() < 1e-5,
        "pos 4: expected 8.0, got {}",
        result[4]
    );
    assert!(
        (result[5] - 10.0).abs() < 1e-5,
        "pos 5: expected 10.0, got {}",
        result[5]
    );
    assert!(
        (result[6] - 12.0).abs() < 1e-5,
        "pos 6: expected 12.0, got {}",
        result[6]
    );
    assert!(
        (result[7] - 14.0).abs() < 1e-5,
        "pos 7: expected 14.0, got {}",
        result[7]
    );

    // Verify candle layout - high nibbles at positions 16-31
    assert!(
        (result[16] - 1.0).abs() < 1e-5,
        "pos 16: expected 1.0, got {}",
        result[16]
    );
    assert!(
        (result[17] - 3.0).abs() < 1e-5,
        "pos 17: expected 3.0, got {}",
        result[17]
    );
    assert!(
        (result[18] - 5.0).abs() < 1e-5,
        "pos 18: expected 5.0, got {}",
        result[18]
    );
    assert!(
        (result[19] - 7.0).abs() < 1e-5,
        "pos 19: expected 7.0, got {}",
        result[19]
    );
    assert!(
        (result[20] - 9.0).abs() < 1e-5,
        "pos 20: expected 9.0, got {}",
        result[20]
    );
    assert!(
        (result[21] - 11.0).abs() < 1e-5,
        "pos 21: expected 11.0, got {}",
        result[21]
    );
    assert!(
        (result[22] - 13.0).abs() < 1e-5,
        "pos 22: expected 13.0, got {}",
        result[22]
    );
    assert!(
        (result[23] - 15.0).abs() < 1e-5,
        "pos 23: expected 15.0, got {}",
        result[23]
    );
}

/// Q4_1 Reference Test: Verify scale and min application
#[test]
fn test_q4_1_scale_and_min_correctness() {
    let mut block = vec![0u8; 20];

    // Scale = 2.0, Min = 10.0
    block[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(10.0).to_le_bytes());

    // byte[0] = 0x31 -> low=1, high=3
    block[4] = 0x31;

    let result = dequantize_q4_1(&block).expect("dequantization failed");

    // Position 0: 2.0 * 1 + 10.0 = 12.0
    assert!(
        (result[0] - 12.0).abs() < 0.1,
        "pos 0: expected 12.0, got {}",
        result[0]
    );

    // Position 16: 2.0 * 3 + 10.0 = 16.0
    assert!(
        (result[16] - 16.0).abs() < 0.1,
        "pos 16: expected 16.0, got {}",
        result[16]
    );
}

/// Q5_0 Reference Test: Verify candle layout with 5th bit
/// GGUF candle layout for Q5_0:
/// - Positions 0-15: (low nibble | (qh bit << 4)) - 16
/// - Positions 16-31: (high nibble | (qh bit << 4)) - 16
///
/// Formula: value = scale * ((nibble | high_bit) - 16)
#[test]
fn test_q5_0_candle_layout_correctness() {
    // Q5_0 block: 2 bytes scale + 4 bytes qh + 16 bytes quants = 22 bytes
    let mut block = vec![0u8; 22];

    // Scale = 1.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // qh = 0 (no high bits set) for simplicity
    block[2..6].copy_from_slice(&[0, 0, 0, 0]);

    // Set quants with pattern
    // With qh=0 and offset=-16, nibble 0 -> -16, nibble 8 -> -8, nibble 15 -> -1
    block[6] = 0x80; // low=0, high=8
    block[7] = 0xF1; // low=1, high=15

    let result = dequantize_q5_0(&block).expect("dequantization failed");

    assert_eq!(result.len(), 32);

    // Position 0: 1.0 * (0 | 0 - 16) = -16.0
    assert!(
        (result[0] - (-16.0)).abs() < 1e-5,
        "pos 0: expected -16.0, got {}",
        result[0]
    );

    // Position 1: 1.0 * (1 | 0 - 16) = -15.0
    assert!(
        (result[1] - (-15.0)).abs() < 1e-5,
        "pos 1: expected -15.0, got {}",
        result[1]
    );

    // Position 16: 1.0 * (8 | 0 - 16) = -8.0
    assert!(
        (result[16] - (-8.0)).abs() < 1e-5,
        "pos 16: expected -8.0, got {}",
        result[16]
    );

    // Position 17: 1.0 * (15 | 0 - 16) = -1.0
    assert!(
        (result[17] - (-1.0)).abs() < 1e-5,
        "pos 17: expected -1.0, got {}",
        result[17]
    );
}

/// Q5_0 Reference Test: Verify high bit handling
/// qh bits 0-15 apply to positions 0-15, bits 16-31 apply to positions 16-31
#[test]
fn test_q5_0_high_bit_correctness() {
    let mut block = vec![0u8; 22];

    // Scale = 1.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // qh = 0x00010001 -> bit 0 and bit 16 are set
    // This means positions 0 and 16 get +16 from the high bit
    block[2..6].copy_from_slice(&0x00010001u32.to_le_bytes());

    // Quants: byte[0] = 0x00 -> low=0, high=0
    block[6] = 0x00;

    let result = dequantize_q5_0(&block).expect("dequantization failed");

    // Position 0: nibble=0, high_bit=1 -> (0 | 16) - 16 = 0
    assert!(
        (result[0] - 0.0).abs() < 1e-5,
        "pos 0: expected 0.0, got {}",
        result[0]
    );

    // Position 16: nibble=0, high_bit=1 -> (0 | 16) - 16 = 0
    assert!(
        (result[16] - 0.0).abs() < 1e-5,
        "pos 16: expected 0.0, got {}",
        result[16]
    );
}

/// Q5_1 Reference Test: Verify candle layout with scale and min
/// Formula: value = scale * (nibble | high_bit) + min
#[test]
fn test_q5_1_candle_layout_correctness() {
    // Q5_1 block: 2 bytes scale + 2 bytes min + 4 bytes qh + 16 bytes quants = 24 bytes
    let mut block = vec![0u8; 24];

    // Scale = 1.0, Min = 0.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    // qh = 0 (no high bits set)
    block[4..8].copy_from_slice(&[0, 0, 0, 0]);

    // Quants with known pattern
    block[8] = 0x10; // low=0, high=1
    block[9] = 0x32; // low=2, high=3

    let result = dequantize_q5_1(&block).expect("dequantization failed");

    assert_eq!(result.len(), 32);

    // Position 0: 1.0 * (0 | 0) + 0 = 0.0
    assert!(
        (result[0] - 0.0).abs() < 1e-5,
        "pos 0: expected 0.0, got {}",
        result[0]
    );

    // Position 1: 1.0 * (2 | 0) + 0 = 2.0
    assert!(
        (result[1] - 2.0).abs() < 1e-5,
        "pos 1: expected 2.0, got {}",
        result[1]
    );

    // Position 16: 1.0 * (1 | 0) + 0 = 1.0
    assert!(
        (result[16] - 1.0).abs() < 1e-5,
        "pos 16: expected 1.0, got {}",
        result[16]
    );

    // Position 17: 1.0 * (3 | 0) + 0 = 3.0
    assert!(
        (result[17] - 3.0).abs() < 1e-5,
        "pos 17: expected 3.0, got {}",
        result[17]
    );
}

/// Q5_1 Reference Test: Verify scale, min, and high bit together
#[test]
fn test_q5_1_scale_min_high_bit_correctness() {
    let mut block = vec![0u8; 24];

    // Scale = 2.0, Min = 5.0
    block[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(5.0).to_le_bytes());

    // qh = 0x00000001 -> only bit 0 set
    block[4..8].copy_from_slice(&0x00000001u32.to_le_bytes());

    // byte[0] = 0x21 -> low=1, high=2
    block[8] = 0x21;

    let result = dequantize_q5_1(&block).expect("dequantization failed");

    // Position 0: 2.0 * (1 | 16) + 5.0 = 2.0 * 17 + 5 = 39.0
    assert!(
        (result[0] - 39.0).abs() < 0.1,
        "pos 0: expected 39.0, got {}",
        result[0]
    );

    // Position 16: 2.0 * (2 | 0) + 5.0 = 2.0 * 2 + 5 = 9.0
    assert!(
        (result[16] - 9.0).abs() < 0.1,
        "pos 16: expected 9.0, got {}",
        result[16]
    );
}

/// Q4_0 Reference Test: Verify candle layout (for comparison - this one works)
/// Q4_0 uses signed offset: value = scale * (nibble - 8)
#[test]
fn test_q4_0_candle_layout_reference() {
    let mut block = vec![0u8; 18];

    // Scale = 1.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // Quants with known pattern
    block[2] = 0x80; // low=0, high=8
    block[3] = 0xF1; // low=1, high=15

    let result = dequantize_q4_0(&block).expect("dequantization failed");

    // Position 0: 1.0 * (0 - 8) = -8.0
    assert!(
        (result[0] - (-8.0)).abs() < 1e-5,
        "pos 0: expected -8.0, got {}",
        result[0]
    );

    // Position 1: 1.0 * (1 - 8) = -7.0
    assert!(
        (result[1] - (-7.0)).abs() < 1e-5,
        "pos 1: expected -7.0, got {}",
        result[1]
    );

    // Position 16: 1.0 * (8 - 8) = 0.0
    assert!(
        (result[16] - 0.0).abs() < 1e-5,
        "pos 16: expected 0.0, got {}",
        result[16]
    );

    // Position 17: 1.0 * (15 - 8) = 7.0
    assert!(
        (result[17] - 7.0).abs() < 1e-5,
        "pos 17: expected 7.0, got {}",
        result[17]
    );
}

/// Multi-block test: Verify Q4_1 works across block boundaries
#[test]
fn test_q4_1_multi_block_correctness() {
    // Two Q4_1 blocks = 40 bytes
    let mut data = vec![0u8; 40];

    // Block 0: scale=1.0, min=0.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data[4] = 0x21; // low=1, high=2

    // Block 1: scale=2.0, min=10.0
    data[20..22].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    data[22..24].copy_from_slice(&half::f16::from_f32(10.0).to_le_bytes());
    data[24] = 0x43; // low=3, high=4

    let result = dequantize_q4_1(&data).expect("dequantization failed");

    assert_eq!(result.len(), 64); // 2 blocks * 32 values

    // Block 0, Position 0: 1.0 * 1 + 0.0 = 1.0
    assert!(
        (result[0] - 1.0).abs() < 1e-5,
        "block 0 pos 0: expected 1.0, got {}",
        result[0]
    );

    // Block 0, Position 16: 1.0 * 2 + 0.0 = 2.0
    assert!(
        (result[16] - 2.0).abs() < 1e-5,
        "block 0 pos 16: expected 2.0, got {}",
        result[16]
    );

    // Block 1, Position 0: 2.0 * 3 + 10.0 = 16.0
    assert!(
        (result[32] - 16.0).abs() < 0.1,
        "block 1 pos 0: expected 16.0, got {}",
        result[32]
    );

    // Block 1, Position 16: 2.0 * 4 + 10.0 = 18.0
    assert!(
        (result[48] - 18.0).abs() < 0.1,
        "block 1 pos 16: expected 18.0, got {}",
        result[48]
    );
}

/// Integration test: Q4_1 matmul produces correct output
/// This tests the entire Q4_1 dequantize + trueno Matrix path
#[test]
fn test_q4_1_matmul_integration() {
    use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

    // Create a small 2x2 matmul (2 output rows, each with 32 elements = 1 block)
    let in_dim = 32;
    let out_dim = 2;

    // Create Q4_1 data for 2 rows (2 blocks × 20 bytes = 40 bytes)
    // Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes quants = 20 bytes
    let mut data = vec![0u8; 40];

    // Row 0: scale=1.0, min=0.0, byte[0]=0x10 -> pos 0=0, pos 16=1
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data[4] = 0x10; // low=0 (pos 0), high=1 (pos 16)
                    // Rest of quants are 0

    // Row 1: scale=1.0, min=0.0, all quants give 1.0 for all positions
    data[20..22].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[22..24].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // Fill row 1 quants with 0x11 -> all positions get value 1
    for i in 24..40 {
        data[i] = 0x11;
    }

    // Dequantize
    let weights_f32 = dequantize_q4_1(&data).expect("dequantization failed");
    assert_eq!(weights_f32.len(), out_dim * in_dim);

    // Debug: Print first few values of each row
    eprintln!("Row 0 (first 20): {:?}", &weights_f32[0..20]);
    eprintln!("Row 0 positions 16-20: {:?}", &weights_f32[16..20]);
    eprintln!("Row 1 (first 20): {:?}", &weights_f32[32..52]);

    // Create Matrix [out_dim, in_dim] = [2, 32]
    let weight_matrix = TruenoMatrix::from_vec(out_dim, in_dim, weights_f32.clone())
        .expect("matrix creation failed");

    // Create activation vector: all 1.0
    let activations = vec![1.0f32; in_dim];
    let x_vec = TruenoVector::from_slice(&activations);

    // Compute matmul
    let result = weight_matrix.matvec(&x_vec).expect("matvec failed");

    eprintln!("Matmul result: {:?}", result.as_slice());

    // Compute expected by manual sum
    let expected_row0: f32 = weights_f32[0..32].iter().sum();
    let expected_row1: f32 = weights_f32[32..64].iter().sum();
    eprintln!("Expected row 0 sum (manual): {}", expected_row0);
    eprintln!("Expected row 1 sum (manual): {}", expected_row1);

    // Row 0: only position 16 has value 1.0, rest are 0
    // Sum should be 1.0
    let row0_sum = result.as_slice()[0];

    // Row 1: all positions have value 1.0
    // Sum of [1, 1, ..., 1] = 32.0
    let row1_sum = result.as_slice()[1];

    assert!(
        (row0_sum - expected_row0).abs() < 0.1,
        "Row 0 sum should be {}, got {}",
        expected_row0,
        row0_sum
    );
    assert!(
        (row1_sum - expected_row1).abs() < 0.1,
        "Row 1 sum should be {}, got {}",
        expected_row1,
        row1_sum
    );
}

/// Integration test: Q4_1 matmul with realistic dimensions (896x896)
#[test]
fn test_q4_1_matmul_large_dimensions() {
    use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

    // Qwen2-0.5B dimensions: hidden_dim=896
    let in_dim = 896;
    let out_dim = 896;

    // blocks_per_row = 896 / 32 = 28
    // bytes_per_row = 28 * 20 = 560
    let blocks_per_row = in_dim / 32;
    let bytes_per_row = blocks_per_row * 20;
    let total_bytes = out_dim * bytes_per_row;

    // Create Q4_1 data: all zeros except row 0 has scale=1.0
    let mut data = vec![0u8; total_bytes];

    // Row 0: scale=1.0, min=0.0, quants all 0x11 (value 1 for all positions)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    for i in 4..bytes_per_row {
        if i >= 4 && (i - 4) % 20 < 16 {
            // Only set quant bytes, not scale/min of subsequent blocks
            data[i] = 0x11;
        }
    }

    // Dequantize
    let weights_f32 = dequantize_q4_1(&data).expect("dequantization failed");

    // Verify dimensions
    assert_eq!(
        weights_f32.len(),
        out_dim * in_dim,
        "Dequantized size mismatch: {} vs {}",
        weights_f32.len(),
        out_dim * in_dim
    );

    // Create Matrix
    let weight_matrix =
        TruenoMatrix::from_vec(out_dim, in_dim, weights_f32).expect("matrix creation failed");

    // Create activation: all 1.0
    let activations = vec![1.0f32; in_dim];
    let x_vec = TruenoVector::from_slice(&activations);

    // Compute matmul
    let result = weight_matrix.matvec(&x_vec).expect("matvec failed");

    assert_eq!(result.len(), out_dim);

    // Row 0 should have some non-zero sum (from the first block's values)
    // Other rows should be ~0 (scale=0 from zero-initialized data)
    let row0_sum = result.as_slice()[0];
    assert!(
        row0_sum.abs() > 0.1,
        "Row 0 should have non-zero output, got {}",
        row0_sum
    );

    // Verify no NaN or Inf
    for (i, &v) in result.as_slice().iter().enumerate() {
        assert!(v.is_finite(), "Output at {} is not finite: {}", i, v);
    }
}

/// Falsification test: Q4_1 should NOT use interleaved layout
/// If this test fails, the code is using wrong (interleaved) layout
#[test]
fn test_q4_1_not_interleaved_layout() {
    let mut block = vec![0u8; 20];

    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    // byte[0] = 0x10 -> low=0, high=1
    // In WRONG interleaved layout: pos 0 = 0, pos 1 = 1
    // In CORRECT candle layout: pos 0 = 0, pos 16 = 1
    block[4] = 0x10;

    let result = dequantize_q4_1(&block).expect("dequantization failed");

    // If interleaved (WRONG): result[1] would be 1.0
    // If candle (CORRECT): result[1] should be 0.0 (from byte 1 which is 0)
    assert!(
        (result[1] - 0.0).abs() < 1e-5,
        "INTERLEAVED LAYOUT DETECTED! pos 1 should be 0.0 (candle), got {} (interleaved would give 1.0)",
        result[1]
    );

    // Verify high nibble goes to position 16
    assert!(
        (result[16] - 1.0).abs() < 1e-5,
        "CANDLE LAYOUT BROKEN! pos 16 should be 1.0, got {}",
        result[16]
    );
}

// =========================================================================
// BUG-GGUF-001: Q4_0 Falsification Tests
// Compare fused Q4_0 × Q8_0 kernel against reference (dequantize + naive)
// =========================================================================

/// Falsification test: fused Q4_0 matmul vs dequantize + TruenoMatrix path
/// If these produce different results, one of the paths has a bug.
#[test]
fn test_q4_0_fused_vs_dequantize_matmul() {
    use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

    // Small test: 2 output rows, 32 input dims (1 block per row)
    let in_dim = 32;
    let out_dim = 2;

    // Q4_0 block: 2 bytes scale + 16 bytes quants = 18 bytes per 32 elements
    let bytes_per_row = 18;
    let total_bytes = out_dim * bytes_per_row;
    let mut data = vec![0u8; total_bytes];

    // Row 0: scale=1.0, quants pattern [0,1,2,...,15] for low nibbles
    //        high nibbles will be [0,0,...,0]
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Set quant bytes: 0x10, 0x32, 0x54, ... for positions 0-15 = 0,2,4,...
    // Q4_0 uses signed: value = (nibble - 8) * scale
    for i in 0..16 {
        // Put increasing value in low nibble (0,1,2,...,15)
        // Put 8 in high nibble so high_quant = 8 - 8 = 0
        data[2 + i] = (8 << 4) | (i as u8);
    }

    // Row 1: scale=2.0, all quants = 0x99 (low=9, high=9)
    // Signed: 9 - 8 = 1, so value = 2.0 * 1 = 2.0 for all positions
    data[18..20].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    for i in 0..16 {
        data[20 + i] = 0x99;
    }

    // Path 1: Fused Q4_0 × Q8_0 matmul
    let activations = vec![1.0f32; in_dim];
    let fused_result = fused_q4_0_q8_0_parallel_matvec(&data, &activations, in_dim, out_dim)
        .expect("fused matmul failed");

    // Path 2: Dequantize + TruenoMatrix matmul (reference)
    let weights_f32 = dequantize_q4_0(&data).expect("dequantize failed");
    let weight_matrix = TruenoMatrix::from_vec(out_dim, in_dim, weights_f32.clone())
        .expect("matrix creation failed");
    let x_vec = TruenoVector::from_slice(&activations);
    let reference_result = weight_matrix.matvec(&x_vec).expect("matvec failed");

    eprintln!("Q4_0 Fused result: {:?}", fused_result);
    eprintln!("Q4_0 Reference result: {:?}", reference_result.as_slice());
    eprintln!("Q4_0 Row 0 weights (first 20): {:?}", &weights_f32[0..20]);
    eprintln!("Q4_0 Row 0 weights (pos 16-20): {:?}", &weights_f32[16..20]);

    // Compare results
    for i in 0..out_dim {
        let fused_val = fused_result[i];
        let ref_val = reference_result.as_slice()[i];
        let diff = (fused_val - ref_val).abs();

        // Allow small tolerance for quantization noise
        assert!(
            diff < 1.0,
            "Q4_0 MISMATCH at row {}: fused={}, reference={}, diff={}",
            i,
            fused_val,
            ref_val,
            diff
        );
    }
}

/// Falsification test: Q4_0 fused matmul with Qwen2-0.5B dimensions
#[test]
fn test_q4_0_fused_matmul_qwen_dimensions() {
    use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

    // Qwen2-0.5B: hidden_dim=896
    let in_dim: usize = 896;
    let out_dim: usize = 128; // Smaller for faster test

    // Q4_0: 18 bytes per 32 elements
    let blocks_per_row = in_dim.div_ceil(32);
    let bytes_per_row = blocks_per_row * 18;
    let total_bytes = out_dim * bytes_per_row;

    // Create deterministic test data
    let mut data = vec![0u8; total_bytes];
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        for block in 0..blocks_per_row {
            let block_start = row_start + block * 18;

            // Scale varies by row
            let scale = 0.1 + (row as f32) * 0.01;
            data[block_start..block_start + 2]
                .copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());

            // Fill quants with deterministic pattern
            for i in 0..16 {
                data[block_start + 2 + i] =
                    (((row + block + i) % 16) << 4 | ((row + i) % 16)) as u8;
            }
        }
    }

    // Create random-ish activations (deterministic)
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Path 1: Fused matmul
    let fused_result = fused_q4_0_q8_0_parallel_matvec(&data, &activations, in_dim, out_dim)
        .expect("fused matmul failed");

    // Path 2: Reference path
    let weights_f32 = dequantize_q4_0(&data).expect("dequantize failed");
    let weight_matrix =
        TruenoMatrix::from_vec(out_dim, in_dim, weights_f32).expect("matrix creation failed");
    let x_vec = TruenoVector::from_slice(&activations);
    let reference_result = weight_matrix.matvec(&x_vec).expect("matvec failed");

    // Compare results
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..out_dim {
        let diff = (fused_result[i] - reference_result.as_slice()[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    eprintln!(
        "Q4_0 Qwen dims: max diff = {} at row {}",
        max_diff, max_diff_idx
    );
    eprintln!("Fused[{}] = {}", max_diff_idx, fused_result[max_diff_idx]);
    eprintln!(
        "Reference[{}] = {}",
        max_diff_idx,
        reference_result.as_slice()[max_diff_idx]
    );

    // Q8_0 quantization introduces error, so allow larger tolerance
    // But systematic bugs would cause HUGE differences
    assert!(
        max_diff < 50.0,
        "Q4_0 fused vs reference max diff {} is too large (indicates bug)",
        max_diff
    );
}
