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
