use crate::quantize::*;
#[test]
fn test_dequantize_q8_blocks_roundtrip_cov() {
    let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let blocks = quantize_to_q8_blocks(&original).expect("quantization failed");
    let restored = dequantize_q8_blocks(&blocks);
    assert_eq!(restored.len(), 64);
    // Should be close to original
    for (o, r) in original.iter().zip(restored.iter()) {
        assert!((o - r).abs() < 0.1);
    }
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock quantize_into
// =========================================================================

#[test]
fn test_q8ksuperblock_quantize_into_cov() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
}

// =========================================================================
// Coverage Tests: dequantize_f16 additional error paths
// =========================================================================

#[test]
fn test_dequantize_f16_size_3_cov() {
    let data = vec![0u8; 3]; // Odd size - invalid
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_f16_vec_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: f16_to_f32 edge cases
// =========================================================================

#[test]
fn test_f16_to_f32_subnormal_cov() {
    // Subnormal f16 value (exponent 0, non-zero mantissa)
    let subnormal = 0x0001u16; // Smallest positive subnormal
    let f32_val = f16_to_f32(subnormal);
    assert!(f32_val > 0.0);
    assert!(f32_val < 1e-5);
}

#[test]
fn test_f16_to_f32_infinity_cov() {
    // Positive infinity (exponent all 1s, mantissa 0)
    let pos_inf = 0x7C00u16;
    let f32_val = f16_to_f32(pos_inf);
    assert!(f32_val.is_infinite() && f32_val > 0.0);
}

#[test]
fn test_f16_to_f32_neg_infinity_cov() {
    // Negative infinity
    let neg_inf = 0xFC00u16;
    let f32_val = f16_to_f32(neg_inf);
    assert!(f32_val.is_infinite() && f32_val < 0.0);
}

// ============================================================================
// Coverage Tests for Dequantize Functions (Refs PMAT-802)
// ============================================================================

#[test]
fn test_cov_dequantize_f16_valid() {
    // Create F16 bytes for values: 1.0, 2.0
    let f16_one = half::f16::from_f32(1.0).to_le_bytes();
    let f16_two = half::f16::from_f32(2.0).to_le_bytes();
    let data = [f16_one[0], f16_one[1], f16_two[0], f16_two[1]];

    let result = dequantize_f16(&data).expect("should succeed");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[1] - 2.0).abs() < 0.001);
}

#[test]
fn test_cov_dequantize_f16_invalid_odd() {
    let data = [0u8; 3]; // Odd length
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_dequantize_f16_empty() {
    let data: [u8; 0] = [];
    let result = dequantize_f16(&data).expect("empty should succeed");
    assert_eq!(result.len(), 0);
}

#[test]
fn test_cov_dequantize_q4_1_invalid() {
    // Q4_1 block is 20 bytes, test with 19
    let data = vec![0u8; 19];
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_dequantize_q4_1_one_block() {
    // Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes data = 20 bytes
    let mut data = Vec::new();
    // Scale = 1.0 as f16
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Min = 0.0 as f16
    data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // 16 bytes of zeros (32 4-bit values)
    data.extend_from_slice(&[0u8; 16]);

    let result = dequantize_q4_1(&data).expect("should succeed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_cov_dequantize_q4_1_two_blocks() {
    let mut data = Vec::new();
    // Two blocks of 20 bytes each
    for _ in 0..2 {
        data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
        data.extend_from_slice(&[0u8; 16]);
    }

    let result = dequantize_q4_1(&data).expect("should succeed");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_cov_dequantize_q5_0_invalid() {
    // Q5_0 block is 22 bytes, test with 21
    let data = vec![0u8; 21];
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_dequantize_q5_0_one_block() {
    // Q5_0 block: 2 bytes scale + 4 bytes high bits + 16 bytes low bits = 22 bytes
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&[0u8; 4]); // qh (high bits)
    data.extend_from_slice(&[0u8; 16]); // qs (low bits)

    let result = dequantize_q5_0(&data).expect("should succeed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_cov_dequantize_q5_0_two_blocks() {
    // Two blocks = 44 bytes -> 64 floats
    let data = vec![0u8; 44];
    let result = dequantize_q5_0(&data).expect("should succeed");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_cov_dequantize_q5_1_invalid() {
    // Q5_1 block is 24 bytes, test with 23
    let data = vec![0u8; 23];
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_dequantize_q5_1_one_block() {
    // Q5_1 block: 2 bytes scale + 2 bytes min + 4 bytes high + 16 bytes low = 24 bytes
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data.extend_from_slice(&[0u8; 4]); // qh
    data.extend_from_slice(&[0u8; 16]); // qs

    let result = dequantize_q5_1(&data).expect("should succeed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_cov_dequantize_q5_1_two_blocks() {
    // Two blocks = 48 bytes -> 64 floats
    let data = vec![0u8; 48];
    let result = dequantize_q5_1(&data).expect("should succeed");
    assert_eq!(result.len(), 64);
}

// =========================================================================
// IMP-XXX: Direct Scalar Fallback Tests
// =========================================================================
// These tests directly call the scalar implementations, bypassing the
// SIMD dispatcher. This ensures the scalar fallbacks are tested even on
// machines with AVX2 support.

#[test]
fn test_scalar_quantize_rmsnorm_q8_0_direct() {
    // Direct test of scalar implementation
    let input = vec![1.0f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);

    // Should produce 2 blocks of scales (64 / 32 = 2)
    assert_eq!(scales.len(), 2);
    // Should produce 64 quantized values
    assert_eq!(quants.len(), 64);
    // Scales should be positive
    assert!(scales.iter().all(|&s| s >= 0.0));
}

#[test]
fn test_scalar_quantize_rmsnorm_q8_0_various_sizes() {
    for size in [32, 64, 128, 256, 512] {
        let input = vec![0.5f32; size];
        let norm_weight = vec![2.0f32; size];
        let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-6);

        assert_eq!(scales.len(), size / 32, "scales for size {}", size);
        assert_eq!(quants.len(), size, "quants for size {}", size);
    }
}

#[test]
fn test_scalar_fused_swiglu_direct() {
    // Direct test of scalar SwiGLU
    let mut gate = vec![1.0f32; 64];
    let up = vec![2.0f32; 64];

    fused_swiglu_simd(&mut gate, &up);

    // silu(1.0) ≈ 0.7311 (sigmoid(1) = ~0.7311)
    // result = silu(gate) * up ≈ 0.7311 * 2.0 ≈ 1.4623
    // Note: SIMD approximation may differ slightly
    for g in &gate {
        assert!(*g > 1.4 && *g < 1.6, "got {}", g);
    }
}

#[test]
fn test_scalar_fused_swiglu_zero() {
    let mut gate = vec![0.0f32; 32];
    let up = vec![1.0f32; 32];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    for g in &gate {
        assert!(g.abs() < 1e-6, "expected near zero, got {}", g);
    }
}

#[test]
fn test_scalar_softmax_direct() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    softmax_simd(&mut x);

    // Sum should be 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {}", sum);

    // Values should be monotonically increasing (since input was increasing)
    assert!(x[0] < x[1] && x[1] < x[2] && x[2] < x[3]);
}

#[test]
fn test_scalar_softmax_large_values() {
    // Test numerical stability with large values
    let mut x = vec![1000.0, 1001.0, 1002.0, 1003.0];
    softmax_simd(&mut x);

    // Should not overflow/underflow
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {}", sum);
    assert!(
        x.iter().all(|&v| v.is_finite()),
        "all values should be finite"
    );
}

#[test]
fn test_scalar_apply_rope_rotation_direct() {
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
    let mut x2 = vec![5.0, 6.0, 7.0, 8.0];
    let cos_vals = vec![0.5f32; 4];
    let sin_vals = vec![0.866f32; 4]; // sin(60°) ≈ 0.866

    let x1_orig = x1.clone();
    let x2_orig = x2.clone();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Rotation formula: x1' = x1*cos - x2*sin, x2' = x1*sin + x2*cos
    for i in 0..4 {
        let expected_x1 = x1_orig[i] * 0.5 - x2_orig[i] * 0.866;
        let expected_x2 = x1_orig[i] * 0.866 + x2_orig[i] * 0.5;
        assert!(
            (x1[i] - expected_x1).abs() < 1e-5,
            "x1[{}]: expected {}, got {}",
            i,
            expected_x1,
            x1[i]
        );
        assert!(
            (x2[i] - expected_x2).abs() < 1e-5,
            "x2[{}]: expected {}, got {}",
            i,
            expected_x2,
            x2[i]
        );
    }
}

#[test]
fn test_scalar_fused_q4_0_q8_0_dot_direct() {
    // Create minimal Q4_0 data (1 block = 18 bytes)
    // Layout: 2 bytes f16 scale + 16 bytes quants
    let mut q4_data = vec![0u8; 18];
    // Set scale to 1.0 (f16 encoding)
    let scale_bits = half::f16::from_f32(1.0).to_bits();
    q4_data[0] = scale_bits as u8;
    q4_data[1] = (scale_bits >> 8) as u8;
    // Set quants to some pattern (each byte = 2 values)
    for i in 0..16 {
        q4_data[2 + i] = 0x88; // Both nibbles = 8 (after -8 offset = 0)
    }

    // Create Q8_0 activations (1 block = 32 values)
    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);

    // With our setup, all dequantized values are 0, so dot product should be 0
    assert!(result.abs() < 1e-5, "expected near zero, got {}", result);
}

#[test]
fn test_scalar_fused_q8_0_q8_0_dot_direct() {
    // Create Q8_0 weight data (1 block = 34 bytes: 2 bytes scale + 32 bytes quants)
    let mut q8_weight_data = vec![0u8; 34];
    // Set scale to 1.0 (f16 encoding)
    let scale_bits = half::f16::from_f32(1.0).to_bits();
    q8_weight_data[0] = scale_bits as u8;
    q8_weight_data[1] = (scale_bits >> 8) as u8;
    // Set quants to 1
    for i in 0..32 {
        q8_weight_data[2 + i] = 1;
    }

    // Create Q8_0 activations
    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);

    // Dot product of 32 ones with scale 1.0 * 1.0 = 32
    assert!((result - 32.0).abs() < 1e-3, "expected ~32, got {}", result);
}

// =========================================================================
// Direct Tests for Dequantization Helpers
// =========================================================================

#[test]
fn test_dequantize_q4_k_cov_direct() {
    // Create a minimal Q4_K super-block (144 bytes)
    // Layout: d (2) + dmin (2) + scales (12) + qs (128) = 144 bytes
    let mut sb_data = vec![0u8; 144];

    // Set d = 1.0 (f16 encoding)
    let d_bits = half::f16::from_f32(1.0).to_bits();
    sb_data[0] = d_bits as u8;
    sb_data[1] = (d_bits >> 8) as u8;

    // Set dmin = 0.0 (no minimum offset)
    sb_data[2] = 0;
    sb_data[3] = 0;

    // Set scales to simple pattern (all 1s for testing)
    for i in 0..12 {
        sb_data[4 + i] = 0x11; // scale = 1, min = 1 (packed)
    }

    // Set qs to all zeros for simplicity
    for i in 0..128 {
        sb_data[16 + i] = 0x00;
    }

    let result = dequantize_q4_k(&sb_data).expect("test");

    // Should produce QK_K (256) values
    assert_eq!(
        result.len(),
        QK_K,
        "expected {} values, got {}",
        QK_K,
        result.len()
    );
    // All values should be finite
    assert!(
        result.iter().all(|&v| v.is_finite()),
        "all values should be finite"
    );
}

#[test]
fn test_dequantize_q4_k_cov_nonzero_quants() {
    // Test with non-zero quantized values
    let mut sb_data = vec![0u8; 144];

    // d = 0.5
    let d_bits = half::f16::from_f32(0.5).to_bits();
    sb_data[0] = d_bits as u8;
    sb_data[1] = (d_bits >> 8) as u8;

    // dmin = 0.25
    let dmin_bits = half::f16::from_f32(0.25).to_bits();
    sb_data[2] = dmin_bits as u8;
    sb_data[3] = (dmin_bits >> 8) as u8;

    // scales = 0x22 (scale=2, min=2 for each pair)
    for i in 0..12 {
        sb_data[4 + i] = 0x22;
    }

    // qs = 0x88 (nibbles = 8, 8)
    for i in 0..128 {
        sb_data[16 + i] = 0x88;
    }

    let result = dequantize_q4_k(&sb_data).expect("test");

    assert_eq!(result.len(), QK_K);
    // With our setup, values should follow pattern: d * scale * q - dmin * m
    // The exact values depend on scale extraction logic, but they should be consistent
    assert!(result.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_dequantize_q8_0_direct() {
    // Create a Q8_0 block (34 bytes): 2 bytes scale + 32 bytes quants
    let mut block_data = vec![0u8; 34];

    // Set scale = 1.0 (f16 encoding)
    let scale_bits = half::f16::from_f32(1.0).to_bits();
    block_data[0] = scale_bits as u8;
    block_data[1] = (scale_bits >> 8) as u8;

    // Set quants to sequential values 0-31 (as i8)
    for i in 0..32 {
        block_data[2 + i] = i as u8;
    }

    let result = dequantize_q8_0(&block_data).expect("test");

    // Should produce 32 values
    assert_eq!(result.len(), 32, "expected 32 values, got {}", result.len());

    // Values should be 0.0, 1.0, 2.0, ... 31.0 (scale * quant = 1.0 * i)
    for (i, &val) in result.iter().enumerate() {
        let expected = i as f32;
        assert!(
            (val - expected).abs() < 0.01,
            "result[{}] = {}, expected {}",
            i,
            val,
            expected
        );
    }
}

#[test]
fn test_dequantize_q8_0_negative_quants() {
    // Test with negative quantized values
    let mut block_data = vec![0u8; 34];

    // Set scale = 2.0
    let scale_bits = half::f16::from_f32(2.0).to_bits();
    block_data[0] = scale_bits as u8;
    block_data[1] = (scale_bits >> 8) as u8;

    // Set quants to -128, -64, 0, 64, 127 pattern repeated
    let pattern: [i8; 5] = [-128, -64, 0, 64, 127];
    for i in 0..32 {
        block_data[2 + i] = pattern[i % 5] as u8;
    }

    let result = dequantize_q8_0(&block_data).expect("test");

    assert_eq!(result.len(), 32);

    // Check first few values: scale * quant = 2.0 * i8
    assert!(
        (result[0] - (-256.0)).abs() < 0.1,
        "result[0] = {}",
        result[0]
    ); // 2.0 * -128
    assert!(
        (result[1] - (-128.0)).abs() < 0.1,
        "result[1] = {}",
        result[1]
    ); // 2.0 * -64
    assert!((result[2] - 0.0).abs() < 0.1, "result[2] = {}", result[2]); // 2.0 * 0
    assert!((result[3] - 128.0).abs() < 0.1, "result[3] = {}", result[3]); // 2.0 * 64
    assert!((result[4] - 254.0).abs() < 0.1, "result[4] = {}", result[4]); // 2.0 * 127
}

#[test]
fn test_dequantize_q8_0_zero_scale() {
    // Test with zero scale (edge case)
    let mut block_data = vec![0u8; 34];

    // Set scale = 0.0
    block_data[0] = 0;
    block_data[1] = 0;

    // Set quants to 127 (max)
    for i in 0..32 {
        block_data[2 + i] = 127;
    }

    let result = dequantize_q8_0(&block_data).expect("test");

    // All values should be 0 (0.0 * 127 = 0)
    for (i, &val) in result.iter().enumerate() {
        assert!(val.abs() < 0.001, "result[{}] should be 0, got {}", i, val);
    }
}

// ============================================================================
// DIRECT TESTS FOR EXPOSED HELPER FUNCTIONS (Force-Path Coverage)
// ============================================================================

#[test]
fn test_f16_to_f32_lut_zero() {
    // Test zero conversion
    let result = f16_to_f32_lut(0);
    assert!(
        result.abs() < 1e-10,
        "Zero should convert to 0.0, got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_lut_one() {
    // Test 1.0 conversion (f16 bits for 1.0 = 0x3C00)
    let result = f16_to_f32_lut(0x3C00);
    assert!((result - 1.0).abs() < 1e-6, "Should be 1.0, got {}", result);
}

#[test]
fn test_f16_to_f32_lut_negative_one() {
    // Test -1.0 conversion (f16 bits for -1.0 = 0xBC00)
    let result = f16_to_f32_lut(0xBC00);
    assert!(
        (result - (-1.0)).abs() < 1e-6,
        "Should be -1.0, got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_lut_small_value() {
    // Test small value (0.5 = 0x3800 in f16)
    let result = f16_to_f32_lut(0x3800);
    assert!((result - 0.5).abs() < 1e-6, "Should be 0.5, got {}", result);
}

#[test]
fn test_read_f16_zero() {
    // Test reading f16 zero
    let bytes = [0x00, 0x00]; // f16 zero
    let result = read_f16(&bytes);
    assert!(result.abs() < 1e-10, "Should be 0.0, got {}", result);
}

#[test]
fn test_read_f16_one() {
    // Test reading f16 1.0
    let bytes = [0x00, 0x3C]; // f16 1.0 in little-endian
    let result = read_f16(&bytes);
    assert!((result - 1.0).abs() < 1e-6, "Should be 1.0, got {}", result);
}

#[test]
fn test_read_f16_negative() {
    // Test reading f16 -1.0
    let bytes = [0x00, 0xBC]; // f16 -1.0 in little-endian
    let result = read_f16(&bytes);
    assert!(
        (result - (-1.0)).abs() < 1e-6,
        "Should be -1.0, got {}",
        result
    );
}

#[test]
fn test_extract_scale_min_from_slice_first_block() {
    // Test extraction for index 0 (first block - simple layout)
    let scales: [u8; 8] = [0x3F, 0x1F, 0x0F, 0x07, 0x20, 0x10, 0x08, 0x04];
    let (scale, min) = extract_scale_min_from_slice(&scales, 0);
    // idx=0: scale = scales[0] & 0x3F = 0x3F = 63
    // idx=0: min = scales[4] & 0x3F = 0x20 = 32
    assert!(
        (scale - 63.0).abs() < 0.001,
        "scale should be 63, got {}",
        scale
    );
    assert!((min - 32.0).abs() < 0.001, "min should be 32, got {}", min);
}

#[test]
fn test_extract_scale_min_from_slice_second_block() {
    // Test extraction for index 2 (even, simple layout)
    let scales: [u8; 8] = [0x3F, 0x1F, 0x0F, 0x07, 0x20, 0x10, 0x08, 0x04];
    let (scale, min) = extract_scale_min_from_slice(&scales, 2);
    // idx=2: scale_idx = 1, scale = scales[1] & 0x3F = 0x1F = 31
    // idx=2: min_idx = 5, min = scales[5] & 0x3F = 0x10 = 16
    assert!(
        (scale - 31.0).abs() < 0.001,
        "scale should be 31, got {}",
        scale
    );
    assert!((min - 16.0).abs() < 0.001, "min should be 16, got {}", min);
}

#[test]
fn test_extract_scale_min_first_blocks() {
    // Test extract_scale_min for first 4 blocks (simple layout)
    let scales: [u8; 12] = [
        0x3F, 0x2F, 0x1F, 0x0F, // First 4 scale values (low 6 bits)
        0x20, 0x18, 0x10, 0x08, // First 4 min values (low 6 bits)
        0x00, 0x00, 0x00, 0x00, // High bits (unused for first 4)
    ];

    // Block 0: scale = 0x3F & 63 = 63, min = 0x20 & 63 = 32
    let (scale, min) = extract_scale_min(&scales, 0);
    assert!(
        (scale - 63.0).abs() < 0.001,
        "Block 0 scale should be 63, got {}",
        scale
    );
    assert!(
        (min - 32.0).abs() < 0.001,
        "Block 0 min should be 32, got {}",
        min
    );

    // Block 1: scale = 0x2F & 63 = 47, min = 0x18 & 63 = 24
    let (scale, min) = extract_scale_min(&scales, 1);
    assert!(
        (scale - 47.0).abs() < 0.001,
        "Block 1 scale should be 47, got {}",
        scale
    );
    assert!(
        (min - 24.0).abs() < 0.001,
        "Block 1 min should be 24, got {}",
        min
    );
}

#[test]
fn test_extract_scale_min_packed_blocks() {
    // Test extract_scale_min for blocks 4-7 (packed layout)
    // The packed layout uses high bits from bytes 0-3 combined with bytes 8-11
    let scales: [u8; 12] = [
        0x40, 0x80, 0xC0, 0x00, // Low 6 bits + high 2 bits for blocks 4-7
        0x00, 0x00, 0x00, 0x00, // Min values for blocks 0-3
        0x12, 0x34, 0x56, 0x78, // Packed values for blocks 4-7
    ];

    // Block 4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = (0x12 & 0x0F) | (1 << 4) = 2 | 16 = 18
    // Block 4: m = (scales[8] >> 4) | ((scales[4] >> 6) << 4) = (0x12 >> 4) | (0 << 4) = 1 | 0 = 1
    let (scale, min) = extract_scale_min(&scales, 4);
    assert!(
        (scale - 18.0).abs() < 0.001,
        "Block 4 scale should be 18, got {}",
        scale
    );
    assert!(
        (min - 1.0).abs() < 0.001,
        "Block 4 min should be 1, got {}",
        min
    );
}

#[test]
fn test_extract_scale_min_all_zeros() {
    // Test with all zeros
    let scales: [u8; 12] = [0; 12];

    for block_idx in 0..8 {
        let (scale, min) = extract_scale_min(&scales, block_idx);
        assert!(
            scale.abs() < 0.001,
            "Block {} scale should be 0, got {}",
            block_idx,
            scale
        );
        assert!(
            min.abs() < 0.001,
            "Block {} min should be 0, got {}",
            block_idx,
            min
        );
    }
}

// =========================================================================
// Dequantization Edge Cases (Coverage: error paths)
// =========================================================================

#[test]
fn test_dequantize_f16_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_f16(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_f16_odd_length_edge() {
    // Odd number of bytes (not valid for f16)
    let data = [0u8; 3];
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_0_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q4_0(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q8_0(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_k_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q4_k(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q5_k_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q5_k(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q6_k_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q6_k(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Dequantize Q4_1 Tests (Coverage: Q4_1 format)
// =========================================================================

#[test]
fn test_dequantize_q4_1_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q4_1(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Dequantize Q5_0 Tests (Coverage: Q5_0 format)
// =========================================================================

#[test]
fn test_dequantize_q5_0_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q5_0(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Dequantize Q5_1 Tests (Coverage: Q5_1 format)
// =========================================================================

#[test]
fn test_dequantize_q5_1_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q5_1(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Block Size Constants Tests (Coverage: constants)
// =========================================================================

#[test]
fn test_block_size_constant_check() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant_check() {
    assert_eq!(QK_K, 256);
}

// =========================================================================
// Q4_0Block Tests (Coverage: Q4_0Block struct)
// =========================================================================

#[test]
fn test_q4_0_block_struct_fields_check() {
    let block = Q4_0Block {
        scale: 2.5,
        quants: [0x12; 16],
    };
    assert!((block.scale - 2.5).abs() < 1e-6);
    assert_eq!(block.quants[0], 0x12);
}

#[test]
fn test_q4_0_block_clone_check() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0xAB; 16],
    };
    let cloned = block.clone();
    assert!((cloned.scale - block.scale).abs() < 1e-6);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q4_0_block_debug_check() {
    let block = Q4_0Block {
        scale: 0.5,
        quants: [0; 16],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q4_0Block"));
    assert!(debug.contains("scale"));
}

// =========================================================================
// Q8_0Block Tests (Coverage: Q8_0Block struct)
// =========================================================================

#[test]
fn test_q8_0_block_struct_fields_check() {
    let block = Q8_0Block {
        scale: 0.125,
        quants: [64i8; 32],
    };
    assert!((block.scale - 0.125).abs() < 1e-6);
    assert_eq!(block.quants[0], 64);
}

#[test]
fn test_q8_0_block_clone_check() {
    let block = Q8_0Block {
        scale: 0.25,
        quants: [-10i8; 32],
    };
    let cloned = block.clone();
    assert!((cloned.scale - block.scale).abs() < 1e-6);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8_0_block_debug_check() {
    let block = Q8_0Block {
        scale: 1.0,
        quants: [0; 32],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q8_0Block"));
    assert!(debug.contains("scale"));
}

// =========================================================================
// f16_to_f32_lut Additional Tests (Coverage: LUT edge cases)
// =========================================================================

#[test]
fn test_f16_to_f32_lut_max_positive_check() {
    // f16 max positive = 0x7BFF = ~65504
    let result = f16_to_f32_lut(0x7BFF);
    assert!(result > 60000.0);
}

#[test]
fn test_f16_to_f32_lut_max_negative_check() {
    // f16 max negative = 0xFBFF = ~-65504
    let result = f16_to_f32_lut(0xFBFF);
    assert!(result < -60000.0);
}

#[test]
fn test_f16_to_f32_lut_infinity_check() {
    // f16 positive infinity = 0x7C00
    let result = f16_to_f32_lut(0x7C00);
    assert!(result.is_infinite());
    assert!(result > 0.0);
}

#[test]
fn test_f16_to_f32_lut_neg_infinity_check() {
    // f16 negative infinity = 0xFC00
    let result = f16_to_f32_lut(0xFC00);
    assert!(result.is_infinite());
    assert!(result < 0.0);
}

#[test]
fn test_f16_to_f32_lut_nan_check() {
    // f16 NaN = 0x7C01 (exponent all 1s, nonzero mantissa)
    let result = f16_to_f32_lut(0x7C01);
    assert!(result.is_nan());
}

#[test]
fn test_f16_to_f32_lut_subnormal_check() {
    // f16 smallest subnormal = 0x0001
    let result = f16_to_f32_lut(0x0001);
    assert!(result > 0.0);
    assert!(result < 1e-6);
}

// =========================================================================
// Softmax SIMD Tests (Coverage: softmax_simd function - extended)
// =========================================================================

#[test]
fn test_softmax_simd_single_element_extended() {
    let mut values = [1.0f32];
    softmax_simd(&mut values);
    assert!((values[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_two_equal_elements_extended() {
    let mut values = [1.0f32, 1.0];
    softmax_simd(&mut values);
    assert!((values[0] - 0.5).abs() < 1e-6);
    assert!((values[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_dominant_element_extended() {
    let mut values = [100.0f32, 0.0, 0.0];
    softmax_simd(&mut values);
    assert!(values[0] > 0.99); // Dominant
    assert!(values[1] < 0.01);
    assert!(values[2] < 0.01);
}

#[test]
fn test_softmax_simd_negative_values_extended() {
    let mut values = [-1.0f32, -2.0, -3.0];
    softmax_simd(&mut values);
    // Sum should be 1
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // First element should be largest
    assert!(values[0] > values[1]);
    assert!(values[1] > values[2]);
}

#[test]
fn test_softmax_simd_large_values_extended() {
    let mut values = [1000.0f32, 1000.0, 1000.0];
    softmax_simd(&mut values);
    // Should still sum to 1 despite large inputs
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_empty_extended() {
    let mut values: [f32; 0] = [];
    softmax_simd(&mut values);
    // Should not panic
}

// =========================================================================
// Fused Q4_0 Q8_0 Parallel Matvec Tests (Coverage: fused operations)
// =========================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_empty_input_check() {
    let result = fused_q4_0_q8_0_parallel_matvec(&[], &[], 0, 0);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_empty_check() {
    let mut output: Vec<f32> = vec![];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&[], &[], 0, &mut output);
    assert!(result.is_ok());
}

// =========================================================================
// Quantize Activations Tests (Coverage: activation quantization)
// =========================================================================

#[test]
fn test_quantize_activations_q8_0_returns_tuple() {
    let (scales, quants) = quantize_activations_q8_0(&[1.0, 2.0, 3.0, 4.0]);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_empty_returns_tuple() {
    let (scales, quants) = quantize_activations_q8_0(&[]);
    assert!(scales.is_empty());
    assert!(quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_uniform_values() {
    let input = vec![2.0f32; 64];
    let (scales, quants) = quantize_activations_q8_0(&input);
    assert!(!scales.is_empty());
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_activations_q8_0_zeros_values() {
    let input = vec![0.0f32; 32];
    let (scales, quants) = quantize_activations_q8_0(&input);
    // Should handle zeros gracefully
    let _ = (scales, quants);
}

// =========================================================================
// Extract Scale Min from Slice Tests (Coverage: scale extraction helpers)
// =========================================================================

#[test]
fn test_extract_scale_min_from_slice_all_max_check() {
    let scales: [u8; 8] = [0x3F; 8];
    let (scale, _min) = extract_scale_min_from_slice(&scales, 0);
    assert!((scale - 63.0).abs() < 0.001);
}

#[test]
fn test_extract_scale_min_from_slice_alternating_check() {
    let scales: [u8; 8] = [0x15, 0x2A, 0x15, 0x2A, 0x15, 0x2A, 0x15, 0x2A];
    let (scale0, _min0) = extract_scale_min_from_slice(&scales, 0);
    let (scale2, _min2) = extract_scale_min_from_slice(&scales, 2);
    // Should be consistent
    assert!((scale0 - 21.0).abs() < 0.001);
    assert!((scale2 - 42.0).abs() < 0.001);
}

// =========================================================================
// Coverage Tests: Q4_0Block struct
// =========================================================================

#[test]
fn test_q4_0_block_debug_cov() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0u8; 16],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q4_0Block"));
}

#[test]
fn test_q4_0_block_clone_cov() {
    let block = Q4_0Block {
        scale: 2.5,
        quants: [0x12; 16],
    };
    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q4_0_block_zero_scale_cov() {
    let block = Q4_0Block {
        scale: 0.0,
        quants: [0xFF; 16],
    };
    // Q4_0Block stores raw bytes, verify fields
    assert_eq!(block.scale, 0.0);
    assert_eq!(block.quants[0], 0xFF);
}

// =========================================================================
// Coverage Tests: Q8_0Block struct
// =========================================================================

#[test]
fn test_q8_0_block_debug_cov() {
    let block = Q8_0Block {
        scale: 0.5,
        quants: [0i8; 32],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q8_0Block"));
}

#[test]
fn test_q8_0_block_clone_cov() {
    let block = Q8_0Block {
        scale: 1.0,
        quants: [127i8; 32],
    };
    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8_0_block_negative_quants_cov() {
    let mut quants = [0i8; 32];
    for i in 0..32 {
        quants[i] = -((i % 128) as i8);
    }
    let block = Q8_0Block { scale: 0.1, quants };
    let deq = block.dequantize();
    assert_eq!(deq.len(), 32);
    // First non-zero should be negative
    assert!(deq[1] < 0.0);
}

// =========================================================================
// Coverage Tests: Q4_KBlock struct
// =========================================================================

#[test]
fn test_q4_k_block_debug_cov() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q4_KBlock"));
}

#[test]
fn test_q4_k_block_clone_cov() {
    let block = Q4_KBlock {
        d: 2.0,
        dmin: 1.0,
        scales: [0x3F; 12],
        qs: [0xAA; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.dmin, block.dmin);
    assert_eq!(cloned.scales, block.scales);
    assert_eq!(cloned.qs, block.qs);
}

// =========================================================================
// Coverage Tests: Q5_KBlock struct
// =========================================================================

#[test]
fn test_q5_k_block_debug_cov() {
    let block = Q5_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q5_KBlock"));
}

#[test]
fn test_q5_k_block_clone_cov() {
    let block = Q5_KBlock {
        d: 3.0,
        dmin: 1.5,
        scales: [0x55; 12],
        qh: [0xFF; 32],
        qs: [0x55; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.qh, block.qh);
}

// =========================================================================
// Coverage Tests: Q6_KBlock struct
// =========================================================================

#[test]
fn test_q6_k_block_debug_cov() {
    let block = Q6_KBlock {
        d: 1.0,
        scales: [0i8; 16],
        qh: [0u8; 64],
        qs: [0u8; 128],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q6_KBlock"));
}

#[test]
fn test_q6_k_block_clone_cov() {
    let block = Q6_KBlock {
        d: 4.0,
        scales: [127i8; 16],
        qh: [0xAA; 64],
        qs: [0x55; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.scales, block.scales);
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock struct
// =========================================================================

#[test]
fn test_q8k_superblock_debug_cov() {
    let sb = Q8KSuperBlock {
        scale: 1.0,
        quants: [0i8; 256],
    };
    let debug_str = format!("{:?}", sb);
    assert!(debug_str.contains("Q8KSuperBlock"));
}

#[test]
fn test_q8k_superblock_clone_cov() {
    let sb = Q8KSuperBlock {
        scale: 2.0,
        quants: [64i8; 256],
    };
    let cloned = sb.clone();
    assert_eq!(cloned.scale, sb.scale);
    assert_eq!(cloned.quants[0], sb.quants[0]);
}

#[test]
fn test_q8k_superblock_quantize_zeros_cov() {
    let values = [0.0f32; 256];
    let sb = Q8KSuperBlock::quantize(&values);
    // All zeros should produce near-zero quants
    for q in &sb.quants {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_q8k_superblock_quantize_max_values_cov() {
    let values = [127.0f32; 256];
    let sb = Q8KSuperBlock::quantize(&values);
    // Scale should handle max values
    assert!(sb.scale > 0.0);
    // All quants should be at max
    for q in &sb.quants {
        assert_eq!(*q, 127);
    }
}

#[test]
fn test_q8k_superblock_dequantize_roundtrip_cov() {
    let mut values = [0.0f32; 256];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32 - 128.0) * 0.1;
    }
    let sb = Q8KSuperBlock::quantize(&values);
    let deq = sb.dequantize();
    // Check roundtrip is approximate
    for (orig, deq_val) in values.iter().zip(deq.iter()) {
        let diff = (orig - deq_val).abs();
        assert!(diff < 0.2); // Quantization error tolerance
    }
}

// =========================================================================
// Coverage Tests: InterleavedQ4K struct
// =========================================================================

#[test]
fn test_interleaved_q4k_debug_cov() {
    let iq4k = InterleavedQ4K {
        d: vec![1.0],
        dmin: vec![0.5],
        scales: vec![0u8; 12],
        qs: vec![0u8; 128],
        num_super_blocks: 1,
    };
    let debug_str = format!("{:?}", iq4k);
    assert!(debug_str.contains("InterleavedQ4K"));
}

#[test]
fn test_interleaved_q4k_clone_cov() {
    let iq4k = InterleavedQ4K {
        d: vec![2.0, 3.0],
        dmin: vec![1.0, 1.5],
        scales: vec![0x55; 24],
        qs: vec![0xAA; 256],
        num_super_blocks: 2,
    };
    let cloned = iq4k.clone();
    assert_eq!(cloned.num_super_blocks, iq4k.num_super_blocks);
    assert_eq!(cloned.d, iq4k.d);
}

#[test]
fn test_interleaved_q4k_from_q4k_invalid_size_cov() {
    // Not a multiple of 144
    let data = vec![0u8; 100];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_from_q4k_empty_deep2() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let iq4k = result.expect("quantization failed");
    assert_eq!(iq4k.num_super_blocks, 0);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8k_into error paths (extended)
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_invalid_length_ext2_cov() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_scales_too_small_ext2_cov() {
    let activations = vec![1.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Only space for 1
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_quants_too_small_ext2_cov() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_success_ext2_cov() {
    let activations = vec![1.5f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: quantize_to_q8_blocks (extended)
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_invalid_length_ext2_cov() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_success_ext2_cov() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("quantization failed");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_empty_ext2_cov() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q8_blocks (extended)
// =========================================================================

#[test]
fn test_dequantize_q8_blocks_multiple_ext2_cov() {
    let blocks = vec![
        Q8_0Block {
            scale: 1.0,
            quants: [10i8; 32],
        },
        Q8_0Block {
            scale: 2.0,
            quants: [5i8; 32],
        },
    ];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 64);
    // First block values
    assert!((result[0] - 10.0).abs() < 0.01);
    // Second block values
    assert!((result[32] - 10.0).abs() < 0.01);
}

// =========================================================================
// Coverage Tests: dequantize_q4_1 error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q4_1_invalid_length_ext_cov() {
    let data = vec![0u8; 10]; // Not multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_1_empty_ext_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q5_0 error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q5_0_invalid_length_ext_cov() {
    let data = vec![0u8; 15]; // Not multiple of 22
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_0_empty_ext_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q5_1 error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q5_1_invalid_length_ext_cov() {
    let data = vec![0u8; 15]; // Not multiple of 24
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_empty_ext_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q4_k error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q4_k_invalid_length_ext_cov() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = dequantize_q4_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_q5_k error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q5_k_invalid_length_ext_cov() {
    let data = vec![0u8; 100]; // Not multiple of 176
    let result = dequantize_q5_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_q6_k error path (extended)
// =========================================================================

#[test]
fn test_dequantize_q6_k_invalid_length_ext_cov() {
    let data = vec![0u8; 100]; // Not multiple of 210
    let result = dequantize_q6_k(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock::quantize_into
// =========================================================================

#[test]
fn test_q8k_superblock_quantize_into_cov() {
    let values = vec![1.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    // All same positive values should produce same quants
    assert_eq!(quants[0], quants[255]);
}

#[test]
fn test_q8k_superblock_quantize_into_varied_cov() {
    let mut values = vec![0.0f32; 256];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32) * 0.5 - 64.0;
    }
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    // Should have varied quant values
    assert_ne!(quants[0], quants[200]);
}

// =========================================================================
// Coverage Tests: f16_to_f32 and f16_to_f32_lut
// =========================================================================

#[test]
fn test_f16_to_f32_lut_one_cov() {
    // f16 representation of 1.0 = 0x3C00
    let result = f16_to_f32_lut(0x3C00);
    assert!((result - 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_lut_negative_one_cov() {
    // f16 representation of -1.0 = 0xBC00
    let result = f16_to_f32_lut(0xBC00);
    assert!((result + 1.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_lut_half_cov() {
    // f16 representation of 0.5 = 0x3800
    let result = f16_to_f32_lut(0x3800);
    assert!((result - 0.5).abs() < 0.001);
}

// =========================================================================
// Coverage Tests: Block size and QK_K constants
// =========================================================================

#[test]
fn test_block_size_constant_cov() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant_cov() {
    assert_eq!(QK_K, 256);
}

// =========================================================================
// Extended Coverage Tests for Q8_0Block methods
// =========================================================================

#[test]
fn test_q8_0_block_quantization_error_ext_cov() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Error should be small for uniform values
    assert!(error < 0.1);
}

#[test]
fn test_q8_0_block_quantization_error_zeros_ext_cov() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    assert!(error < 1e-5);
}

#[test]
fn test_q8_0_block_relative_error_ext_cov() {
    let values = [10.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // Relative error should be small
    assert!(rel_error < 0.01);
}

#[test]
fn test_q8_0_block_relative_error_zeros_ext_cov() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // For zeros, relative error should be 0
    assert!(rel_error.abs() < 1e-5);
}

#[test]
fn test_q8_0_block_relative_error_varied_ext_cov() {
    let mut values = [0.0f32; 32];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32 - 16.0) * 2.0;
    }
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert!(rel_error < 0.05);
}

#[test]
fn test_q8_0_block_dequantize_roundtrip_ext_cov() {
    let mut values = [0.0f32; 32];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32) * 0.5 - 8.0;
    }
    let block = Q8_0Block::quantize(&values);
    let deq = block.dequantize();
    for (orig, d) in values.iter().zip(deq.iter()) {
        let diff = (orig - d).abs();
        assert!(diff < 0.5); // Quantization tolerance
    }
}

// =========================================================================
// Extended Coverage Tests for f16_to_f32_lut
// =========================================================================

#[test]
fn test_f16_to_f32_lut_zero_cov() {
    let result = f16_to_f32_lut(0);
    assert!(result.abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_lut_negative_zero_cov() {
    // Negative zero in f16 is 0x8000
    let result = f16_to_f32_lut(0x8000);
    assert!(result.abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_lut_infinity_cov() {
    // Positive infinity in f16 is 0x7C00
    let result = f16_to_f32_lut(0x7C00);
    assert!(result.is_infinite() && result > 0.0);
}

#[test]
fn test_f16_to_f32_lut_negative_infinity_cov() {
    // Negative infinity in f16 is 0xFC00
    let result = f16_to_f32_lut(0xFC00);
    assert!(result.is_infinite() && result < 0.0);
}

#[test]
fn test_f16_to_f32_lut_small_positive_cov() {
    // f16 smallest positive normal: 0x0400
    let result = f16_to_f32_lut(0x0400);
    assert!(result > 0.0 && result < 1.0);
}

// =========================================================================
// Extended Coverage Tests for Q4_0Block
// =========================================================================

#[test]
fn test_q4_0_block_debug_ext_cov() {
    let block = Q4_0Block {
        scale: 1.0,
        quants: [0u8; 16],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q4_0Block"));
}

#[test]
fn test_q4_0_block_clone_ext_cov() {
    let block = Q4_0Block {
        scale: 2.5,
        quants: [0xFF; 16],
    };
    let cloned = block.clone();
    assert!((cloned.scale - block.scale).abs() < 1e-6);
    assert_eq!(cloned.quants, block.quants);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q4_0 success path
// =========================================================================

#[test]
fn test_dequantize_q4_0_one_block_ext_cov() {
    // Q4_0 block = 2 byte f16 scale + 16 byte quants = 18 bytes
    let mut data = vec![0u8; 18];
    // Scale = 1.0 in f16 little-endian: 0x3C00
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // All quants zero = all zeros out
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 32);
}
