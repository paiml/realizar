use crate::quantize::*;
/// Test Q8KSuperBlock quantize_into
#[test]
fn test_q8k_superblock_quantize_into() {
    let values = [2.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    // All values should be quantized to max (127)
    for &q in &quants {
        assert_eq!(q, 127);
    }
}

/// Test Q8KSuperBlock dequantize round-trip
#[test]
fn test_q8k_superblock_dequantize_roundtrip() {
    let mut original = [0.0f32; 256];
    for i in 0..256 {
        original[i] = (i as f32 - 128.0) / 10.0;
    }
    let block = Q8KSuperBlock::quantize(&original);
    let recovered = block.dequantize();

    // Check round-trip error is reasonable
    let max_error: f32 = original
        .iter()
        .zip(recovered.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 0.2); // Q8 has ~1% error
}

/// Test Q8KSuperBlock Debug and Clone traits
#[test]
fn test_q8k_superblock_traits() {
    let values = [1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("scale"));
    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8k_into error paths
// =========================================================================

/// Test quantize_activations_q8k_into with invalid length
#[test]
fn test_quantize_activations_q8k_into_invalid_length() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test quantize_activations_q8k_into with too small scales buffer
#[test]
fn test_quantize_activations_q8k_into_small_scales() {
    let activations = vec![1.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Need 2
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test quantize_activations_q8k_into with too small quants buffer
#[test]
fn test_quantize_activations_q8k_into_small_quants() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Need 256
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

/// Test quantize_activations_q8k_into success case
#[test]
fn test_quantize_activations_q8k_into_success() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: fused_rmsnorm_ffn_up_gate error paths
// =========================================================================

/// Test fused_rmsnorm_ffn_up_gate input dimension error
#[test]
fn test_fused_rmsnorm_ffn_up_gate_input_dim_error() {
    let input = vec![1.0f32; 16]; // Wrong size
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 36];
    let gate_weight = vec![0u8; 36];
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 1);
    assert!(result.is_err());
}

/// Test fused_rmsnorm_ffn_up_gate up_weight too small
#[test]
fn test_fused_rmsnorm_ffn_up_gate_up_weight_error() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 10]; // Too small
    let gate_weight = vec![0u8; 36];
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 2);
    assert!(result.is_err());
}

/// Test fused_rmsnorm_ffn_up_gate gate_weight too small
#[test]
fn test_fused_rmsnorm_ffn_up_gate_gate_weight_error() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 36];
    let gate_weight = vec![0u8; 10]; // Too small
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 2);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_rmsnorm_q4_0_matmul error paths
// =========================================================================

/// Test fused_rmsnorm_q4_0_matmul input dimension error
#[test]
fn test_fused_rmsnorm_q4_0_matmul_input_dim_error() {
    let input = vec![1.0f32; 16]; // Wrong size
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 36];
    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 1);
    assert!(result.is_err());
}

/// Test fused_rmsnorm_q4_0_matmul weight data too small
#[test]
fn test_fused_rmsnorm_q4_0_matmul_weight_data_error() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 10]; // Too small
    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 2);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Q8_0Block methods
// =========================================================================

/// Test Q8_0Block quantize basic
#[test]
fn test_q8_0_block_quantize_basic() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    for &q in &block.quants {
        assert_eq!(q, 127);
    }
}

/// Test Q8_0Block quantize near-zero
#[test]
fn test_q8_0_block_quantize_near_zero() {
    let values = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!((block.scale - 1.0 / 127.0).abs() < 1e-6);
}

/// Test Q8_0Block dequantize
#[test]
fn test_q8_0_block_dequantize() {
    let mut original = [0.0f32; 32];
    for i in 0..32 {
        original[i] = (i as f32 - 16.0) / 10.0;
    }
    let block = Q8_0Block::quantize(&original);
    let recovered = block.dequantize();

    let max_error: f32 = original
        .iter()
        .zip(recovered.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 0.1);
}

/// Test Q8_0Block quantization_error
#[test]
fn test_q8_0_block_quantization_error() {
    let values = [1.5f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    assert!(error >= 0.0);
    assert!(error < 0.1);
}

/// Test Q8_0Block relative_error
#[test]
fn test_q8_0_block_relative_error() {
    let values = [2.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);
    assert!(error >= 0.0);
    assert!(error < 0.05);
}

// =========================================================================
// Coverage Tests: fused_q5k_dot_simd
// =========================================================================

/// Test fused_q5k_dot_simd basic
#[test]
fn test_fused_q5k_dot_simd_basic() {
    // Q5_K super-block is 176 bytes for 256 values
    let mut q5k_data = vec![0u8; 176];
    q5k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot_simd(&q5k_data, &activations);
    assert!(result.is_ok());
}

/// Test fused_q5k_dot_simd invalid length
#[test]
fn test_fused_q5k_dot_simd_invalid_length() {
    let q5k_data = vec![0u8; 100]; // Not multiple of 176
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot_simd(&q5k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q6k_dot_simd error handling
// =========================================================================

/// Test fused_q6k_dot_simd invalid length
#[test]
fn test_fused_q6k_dot_simd_invalid_length() {
    let q6k_data = vec![0u8; 100]; // Not multiple of 210
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot_simd(&q6k_data, &activations);
    assert!(result.is_err());
}

/// Test fused_q6k_dot_simd dimension mismatch
#[test]
fn test_fused_q6k_dot_simd_dim_mismatch() {
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 128]; // Should be 256
    let result = fused_q6k_dot_simd(&q6k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_rmsnorm_q8_0 with various sizes
// =========================================================================

/// Test quantize_rmsnorm_q8_0 with partial block
#[test]
fn test_quantize_rmsnorm_q8_0_partial_block() {
    let input = vec![1.0f32; 48]; // 1 full block + 16 partial
    let norm_weight = vec![1.0f32; 48];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64); // Padded to 32-element blocks
}

/// Test quantize_rmsnorm_q8_0 with large input
#[test]
fn test_quantize_rmsnorm_q8_0_large() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let norm_weight = vec![1.0f32; 256];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 8);
    assert_eq!(quants.len(), 256);
    // Verify quants array is populated (values in i8 range by type)
    assert!(!quants.is_empty());
    // Ensure we have some non-zero values
    assert!(quants.iter().any(|&q| q != 0));
}

/// Test quantize_rmsnorm_q8_0 with near-zero max abs
#[test]
fn test_quantize_rmsnorm_q8_0_near_zero_max() {
    let input = vec![1e-12f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    // Should use fallback scale
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: dequantize_q8_blocks
// =========================================================================

/// Test dequantize_q8_blocks with multiple blocks
#[test]
fn test_dequantize_q8_blocks_multiple() {
    let blocks = vec![
        Q8_0Block {
            scale: 1.0,
            quants: [64i8; 32],
        },
        Q8_0Block {
            scale: 2.0,
            quants: [32i8; 32],
        },
    ];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 64);
    assert!((result[0] - 64.0).abs() < 1e-5);
    assert!((result[32] - 64.0).abs() < 1e-5); // 32 * 2.0
}

// =========================================================================
// Coverage Tests: fused_q5k_dot scalar
// =========================================================================

/// Test fused_q5k_dot basic
#[test]
fn test_fused_q5k_dot_basic() {
    let mut q5k_data = vec![0u8; 176];
    q5k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&q5k_data, &activations);
    assert!(result.is_ok());
}

/// Test fused_q5k_dot invalid length
#[test]
fn test_fused_q5k_dot_invalid_length() {
    let q5k_data = vec![0u8; 100];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&q5k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q6k_dot scalar
// =========================================================================

/// Test fused_q6k_dot dimension mismatch
#[test]
fn test_fused_q6k_dot_dim_mismatch() {
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 128]; // Should be 256
    let result = fused_q6k_dot(&q6k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Tiled matvec error handling
// =========================================================================

/// Test fused_q4k_tiled_matvec dimension mismatch
#[test]
fn test_fused_q4k_tiled_matvec_dim_mismatch() {
    let q4k_data = vec![0u8; 144]; // 1 super-block
    let activations = vec![1.0f32; 128]; // Should be 256
    let result = fused_q4k_tiled_matvec(&q4k_data, &activations, 256, 1, Some(64));
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Parallel matvec into variants
// =========================================================================

/// Test fused_q4k_parallel_matvec_into dimension validation
#[test]
fn test_fused_q4k_parallel_matvec_into_dim_error() {
    let q4k_data = vec![0u8; 144];
    let activations = vec![1.0f32; 128]; // Wrong size
    let mut output = vec![0.0f32; 1];
    let result = fused_q4k_parallel_matvec_into(&q4k_data, &activations, 256, 1, &mut output);
    assert!(result.is_err());
}

/// Test fused_q5k_parallel_matvec_into dimension validation
#[test]
fn test_fused_q5k_parallel_matvec_into_dim_error() {
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    let result = fused_q5k_parallel_matvec_into(&q5k_data, &activations, 256, 1, &mut output);
    assert!(result.is_err());
}

/// Test fused_q6k_parallel_matvec_into dimension validation
#[test]
fn test_fused_q6k_parallel_matvec_into_dim_error() {
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    let result = fused_q6k_parallel_matvec_into(&q6k_data, &activations, 256, 1, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: InterleavedQ4K
// =========================================================================

/// Test InterleavedQ4K from_q4k invalid length
#[test]
fn test_interleaved_q4k_invalid_length() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

/// Test InterleavedQ4K from_q4k success
#[test]
fn test_interleaved_q4k_success() {
    let mut data = vec![0u8; 144];
    // Set d = 1.0 (f16)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("quantization failed");
    assert_eq!(interleaved.num_values(), 256);
}

/// Test InterleavedQ4K dot product
#[test]
fn test_interleaved_q4k_dot() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 256];
    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
}

// =========================================================================
// Coverage Tests: detect_simd_backend
// =========================================================================

/// Test detect_simd_backend returns valid backend
#[test]
fn test_detect_simd_backend_returns_valid() {
    let backend = detect_simd_backend();
    // Backend should be one of the valid variants
    match backend {
        SimdBackend::Scalar | SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon => {},
    }
}

// =========================================================================
// Coverage Tests: apply_rope_rotation_simd
// =========================================================================

/// Test apply_rope_rotation_simd basic
#[test]
fn test_apply_rope_rotation_simd_basic_coverage() {
    // x1, x2, cos, sin must all have same length
    let mut x1 = vec![1.0f32; 32];
    let mut x2 = vec![1.0f32; 32];
    let cos = vec![1.0f32; 32];
    let sin = vec![0.0f32; 32];
    apply_rope_rotation_simd(&mut x1, &mut x2, &cos, &sin);
    // With cos=1, sin=0, values should be unchanged
    for &v in &x1 {
        assert!((v - 1.0).abs() < 1e-5);
    }
}

/// Test apply_rope_rotation_simd with rotation
#[test]
fn test_apply_rope_rotation_simd_with_rotation() {
    let mut x1 = vec![1.0f32; 32];
    let mut x2 = vec![0.0f32; 32];
    let cos = vec![0.0f32; 32]; // cos(90 degrees)
    let sin = vec![1.0f32; 32]; // sin(90 degrees)
    apply_rope_rotation_simd(&mut x1, &mut x2, &cos, &sin);
    // x1_new = x1 * cos - x2 * sin = 1.0 * 0 - 0 * 1 = 0
    // x2_new = x1 * sin + x2 * cos = 1.0 * 1 + 0 * 0 = 1
    assert!(x1[0].abs() < 1e-4);
    assert!((x2[0] - 1.0).abs() < 1e-4);
}

// =========================================================================
// Coverage Tests: fused_swiglu_simd
// =========================================================================

/// Test fused_swiglu_simd various sizes
#[test]
fn test_fused_swiglu_simd_various_sizes() {
    for size in [16, 32, 64, 100] {
        let mut gate = vec![1.0f32; size];
        let up = vec![2.0f32; size];
        fused_swiglu_simd(&mut gate, &up);
        // Check all values are modified
        for &v in &gate {
            assert!(v.is_finite());
        }
    }
}

/// Test fused_swiglu_simd negative values
#[test]
fn test_fused_swiglu_simd_negative_values() {
    let mut gate = vec![-1.0f32; 32];
    let up = vec![2.0f32; 32];
    fused_swiglu_simd(&mut gate, &up);
    // Check SiLU(-1) * 2.0 is computed
    for &v in &gate {
        assert!(v.is_finite());
    }
}

// =========================================================================
// Coverage Tests: softmax_simd
// =========================================================================

/// Test softmax_simd with negative values
#[test]
fn test_softmax_simd_negative_values() {
    let mut x = vec![-1.0f32, -2.0, -3.0, -4.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

/// Test softmax_simd with large values
#[test]
fn test_softmax_simd_large_values() {
    let mut x = vec![100.0f32, 200.0, 300.0, 400.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

/// Test softmax_simd with mixed values
#[test]
fn test_softmax_simd_mixed_values() {
    let mut x = vec![-10.0f32, 0.0, 10.0, 20.0];
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Largest input should have largest probability
    assert!(x[3] > x[2]);
    assert!(x[2] > x[1]);
    assert!(x[1] > x[0]);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8_0
// =========================================================================

/// Test quantize_activations_q8_0 various sizes
#[test]
fn test_quantize_activations_q8_0_various_sizes() {
    for size in [32, 64, 128, 256] {
        let activations: Vec<f32> = (0..size).map(|i| i as f32 / 10.0).collect();
        let (scales, quants) = quantize_activations_q8_0(&activations);
        assert_eq!(scales.len(), size / 32);
        assert_eq!(quants.len(), size);
    }
}

/// Test quantize_activations_q8_0 negative values
#[test]
fn test_quantize_activations_q8_0_negative_values() {
    let activations: Vec<f32> = (0..32).map(|i| -i as f32 / 10.0).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    assert!(scales[0] > 0.0);
    // First value should be 0, others negative
    for &q in &quants[1..] {
        assert!(q <= 0);
    }
}

// =========================================================================
// Coverage Tests: fused_q4_0_q8_0_parallel_matvec
// =========================================================================

/// Test fused_q4_0_q8_0_parallel_matvec weight too small
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_weight_error() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 2);
    assert!(result.is_err());
}

/// Test fused_q4_0_q8_0_parallel_matvec_into weight error
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_weight_error() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q8_0_q8_0_parallel_matvec
// =========================================================================

/// Test fused_q8_0_q8_0_parallel_matvec weight error
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_weight_error() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 2);
    assert!(result.is_err());
}

/// Test fused_q8_0_q8_0_parallel_matvec_into weight error
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_weight_error() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 2, &mut output);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize parallel/simd variants
// =========================================================================

/// Test dequantize_q4_k_parallel invalid length
#[test]
fn test_dequantize_q4_k_parallel_invalid() {
    let data = vec![0u8; 100]; // Not multiple of 144
    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_err());
}

/// Test dequantize_q4_k_simd invalid length
#[test]
fn test_dequantize_q4_k_simd_invalid() {
    let data = vec![0u8; 100];
    let result = dequantize_q4_k_simd(&data);
    assert!(result.is_err());
}

/// Test dequantize_q8_0_parallel invalid length
#[test]
fn test_dequantize_q8_0_parallel_invalid() {
    let data = vec![0u8; 10]; // Not multiple of 34
    let result = dequantize_q8_0_parallel(&data);
    assert!(result.is_err());
}

/// Test dequantize_q8_0_simd invalid length
#[test]
fn test_dequantize_q8_0_simd_invalid() {
    let data = vec![0u8; 10];
    let result = dequantize_q8_0_simd(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Q8_0Block methods (cov suffix)
// =========================================================================

#[test]
fn test_q8_0_block_quantize_cov() {
    let values: [f32; 32] = [1.0; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    assert_eq!(block.quants.len(), 32);
}

#[test]
fn test_q8_0_block_quantize_zeros_cov() {
    let values: [f32; 32] = [0.0; 32];
    let block = Q8_0Block::quantize(&values);
    // Should use minimal scale for near-zero blocks
    assert!(block.scale > 0.0);
    for q in &block.quants {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_q8_0_block_quantize_mixed_cov() {
    let mut values: [f32; 32] = [0.0; 32];
    for i in 0..32 {
        values[i] = (i as f32 - 16.0) * 0.1;
    }
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
}

#[test]
fn test_q8_0_block_dequantize_cov() {
    let values: [f32; 32] = [1.0; 32];
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();
    assert_eq!(dequantized.len(), 32);
    for v in &dequantized {
        assert!((*v - 1.0).abs() < 0.1); // Within quantization error
    }
}

#[test]
fn test_q8_0_block_quantization_error_cov() {
    let values: [f32; 32] = [1.0; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Q8_0 should have small error for uniform values
    assert!(error < 0.1);
}

#[test]
fn test_q8_0_block_relative_error_cov() {
    let values: [f32; 32] = [1.0; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert!(rel_error < 0.1);
}

#[test]
fn test_q8_0_block_relative_error_zeros_cov() {
    let values: [f32; 32] = [0.0; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert_eq!(rel_error, 0.0);
}

// =========================================================================
// Coverage Tests: f16_to_f32 (cov suffix)
// =========================================================================

#[test]
fn test_f16_to_f32_zero_cov() {
    let result = f16_to_f32(0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_f16_to_f32_one_cov() {
    // f16 1.0 = 0x3C00
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_negative_cov() {
    // f16 -1.0 = 0xBC00
    let result = f16_to_f32(0xBC00);
    assert!((result + 1.0).abs() < 1e-3);
}

// =========================================================================
// Coverage Tests: dequantize valid paths (cov suffix)
// =========================================================================

#[test]
fn test_dequantize_q4_0_valid_cov() {
    // Q4_0: 18 bytes per block (2 for f16 scale + 16 for quants)
    let mut data = vec![0u8; 18];
    // Set scale to 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test");
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequantize_q8_0_valid_cov() {
    // Q8_0: 34 bytes per block (2 for f16 scale + 32 for quants)
    let mut data = vec![0u8; 34];
    // Set scale to 1.0 as f16 (0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test");
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequantize_f16_valid_cov() {
    // f16: 2 bytes per value
    let mut data = vec![0u8; 4];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
    data[2..4].copy_from_slice(&0x4000u16.to_le_bytes()); // 2.0
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    let values = result.expect("test");
    assert_eq!(values.len(), 2);
}

#[test]
fn test_dequantize_f16_invalid_odd_length_cov() {
    let data = vec![0u8; 3]; // Not multiple of 2
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_to_q8_blocks (cov suffix)
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_valid_cov() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("test");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_invalid_length_cov() {
    let values = vec![1.0f32; 33]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_blocks_cov() {
    let values = vec![1.0f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("test");
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);
}

// =========================================================================
// Coverage Tests: softmax_simd (cov suffix)
// =========================================================================

#[test]
fn test_softmax_simd_basic_cov() {
    let mut x = vec![1.0f32, 2.0, 3.0];
    softmax_simd(&mut x);
    // Sum should be ~1
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // x[2] should be largest
    assert!(x[2] > x[1]);
    assert!(x[1] > x[0]);
}

#[test]
fn test_softmax_simd_uniform_cov() {
    let mut x = vec![1.0f32; 4];
    softmax_simd(&mut x);
    // All values should be equal (0.25)
    for v in &x {
        assert!((*v - 0.25).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_simd_large_values_cov() {
    let mut x = vec![100.0f32, 200.0, 300.0];
    softmax_simd(&mut x);
    // Should not overflow
    for v in &x {
        assert!(v.is_finite());
    }
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

// =========================================================================
// Coverage Tests: fused_swiglu_simd (cov suffix)
// =========================================================================

#[test]
fn test_fused_swiglu_simd_basic_cov() {
    let mut gate = vec![0.0f32, 1.0, 2.0, -1.0];
    let up = vec![1.0f32, 1.0, 1.0, 1.0];
    fused_swiglu_simd(&mut gate, &up);
    // gate[0] = 0 * sigmoid(0) * 1 = 0 * 0.5 * 1 = 0
    assert!(gate[0].abs() < 0.01);
    // gate[1] = 1 * sigmoid(1) * 1 ≈ 0.731
    assert!((gate[1] - 0.731).abs() < 0.05);
}

#[test]
fn test_fused_swiglu_simd_larger_cov() {
    // Test with larger vectors for better coverage
    let mut gate = vec![1.0f32; 16];
    let up = vec![1.0f32; 16];
    fused_swiglu_simd(&mut gate, &up);
    // gate = 1 * sigmoid(1) * 1 ≈ 0.731
    for v in &gate {
        assert!((*v - 0.731).abs() < 0.05);
    }
}

// =========================================================================
// Coverage Tests: quantize_activations_q8_0 (cov suffix)
// =========================================================================

#[test]
fn test_quantize_activations_q8_0_basic_cov() {
    let activations = vec![1.0f32; 64];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 2); // 64 / 32 = 2 blocks
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_activations_q8_0_zeros_cov() {
    let activations = vec![0.0f32; 32];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    // Quants should all be zero
    for q in &quants {
        assert_eq!(*q, 0);
    }
}

// =========================================================================
// Coverage Tests: Struct Debug/Clone implementations
// =========================================================================

#[test]
fn test_q4_0block_debug_clone() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0u8; 16],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q4_0Block"));
    assert!(debug.contains("1.5"));

    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8_0block_debug_clone() {
    let block = Q8_0Block {
        scale: 2.5,
        quants: [1i8; 32],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q8_0Block"));
    assert!(debug.contains("2.5"));

    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8_0block_relative_error() {
    let original = [1.0f32; 32];
    let block = Q8_0Block::quantize(&original);
    let rel_err = block.relative_error(&original);
    assert!(
        rel_err < 0.01,
        "Relative error should be small for uniform values"
    );

    // Test with zeros
    let zeros = [0.0f32; 32];
    let block_zeros = Q8_0Block::quantize(&zeros);
    let rel_err_zeros = block_zeros.relative_error(&zeros);
    assert_eq!(rel_err_zeros, 0.0, "Relative error for zeros should be 0");
}

#[test]
fn test_q8_0block_quantization_error() {
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) / 10.0);
    let block = Q8_0Block::quantize(&original);
    let error = block.quantization_error(&original);
    // Max error should be bounded
    assert!(error < 0.05, "Quantization error should be bounded");
}

#[test]
fn test_q8ksuperblock_debug_clone() {
    let block = Q8KSuperBlock {
        scale: 3.5,
        quants: [1i8; 256],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q8KSuperBlock"));
    assert!(debug.contains("3.5"));

    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8ksuperblock_dequantize() {
    let mut block = Q8KSuperBlock {
        scale: 0.1,
        quants: [0i8; 256],
    };
    // Set some values
    block.quants[0] = 127;
    block.quants[1] = -128;
    block.quants[255] = 50;

    let dequant = block.dequantize();
    assert_eq!(dequant.len(), 256);
    assert!((dequant[0] - 12.7).abs() < 0.01); // 127 * 0.1
    assert!((dequant[1] - (-12.8)).abs() < 0.01); // -128 * 0.1
    assert!((dequant[255] - 5.0).abs() < 0.01); // 50 * 0.1
}

#[test]
fn test_q4_kblock_debug_clone() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q4_KBlock"));

    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.dmin, block.dmin);
}

#[test]
fn test_q5_kblock_debug_clone() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 0.25,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q5_KBlock"));

    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.dmin, block.dmin);
}

#[test]
fn test_q6_kblock_debug_clone() {
    let block = Q6_KBlock {
        d: 1.5,
        scales: [0i8; 16],
        qh: [0u8; 64],
        qs: [0u8; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q6_KBlock"));

    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
}

#[test]
fn test_interleaved_q4k_debug_clone() {
    // Create minimal valid Q4K data (144 bytes)
    let mut q4k_data = vec![0u8; 144];
    // Set d and dmin as f16
    q4k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4k_data[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("test");
    let debug = format!("{:?}", interleaved);
    assert!(debug.contains("InterleavedQ4K"));

    let cloned = interleaved.clone();
    assert_eq!(cloned.num_super_blocks, interleaved.num_super_blocks);
    assert_eq!(cloned.num_values(), 256);
}

#[test]
fn test_interleaved_q4k_invalid_length_cov() {
    let bad_data = vec![0u8; 143]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&bad_data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: f16_to_f32 edge cases
// =========================================================================

#[test]
fn test_f16_to_f32_positive_zero() {
    let result = f16_to_f32(0x0000);
    assert_eq!(result, 0.0);
    assert!(!result.is_sign_negative());
}

#[test]
fn test_f16_to_f32_negative_zero() {
    let result = f16_to_f32(0x8000);
    assert_eq!(result, -0.0);
    assert!(result.is_sign_negative());
}

#[test]
fn test_f16_to_f32_positive_infinity() {
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite());
    assert!(result.is_sign_positive());
}

#[test]
fn test_f16_to_f32_negative_infinity() {
    let result = f16_to_f32(0xFC00);
    assert!(result.is_infinite());
    assert!(result.is_sign_negative());
}

#[test]
fn test_f16_to_f32_nan_cov() {
    let result = f16_to_f32(0x7C01); // NaN with non-zero mantissa
    assert!(result.is_nan());
}

#[test]
fn test_f16_to_f32_subnormal() {
    // Smallest positive subnormal: 2^-24 ≈ 5.96e-8
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0);
    assert!(result < 1e-6);

    // Negative subnormal
    let result_neg = f16_to_f32(0x8001);
    assert!(result_neg < 0.0);
    assert!(result_neg > -1e-6);
}

#[test]
fn test_f16_to_f32_normal_values() {
    // Test 1.0
    let one = f16_to_f32(0x3C00);
    assert!((one - 1.0).abs() < 1e-5);

    // Test -1.0
    let neg_one = f16_to_f32(0xBC00);
    assert!((neg_one - (-1.0)).abs() < 1e-5);

    // Test 2.0
    let two = f16_to_f32(0x4000);
    assert!((two - 2.0).abs() < 1e-5);
}

// =========================================================================
// Coverage Tests: dequantize_q4_1 valid case
// =========================================================================

#[test]
fn test_dequantize_q4_1_valid() {
    // Create valid Q4_1 data: 20 bytes per block
    let mut data = vec![0u8; 20];
    // Scale = 1.0 (f16)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Min = 0.0 (f16)
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // 16 bytes of quants (all zeros = 0s)
    // zeros should produce: d * 0 + min = 0 for low nibble, d * 0 + min = 0 for high nibble

    let result = dequantize_q4_1(&data).expect("test");
    assert_eq!(result.len(), 32);
    // With zero quants and zero min, all values should be 0
    for v in &result {
        assert!(v.abs() < 0.001);
    }
}

#[test]
fn test_dequantize_q4_1_invalid_length_cov() {
    let data = vec![0u8; 19]; // Not multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_q5_0 and dequantize_q5_1
// =========================================================================

#[test]
fn test_dequantize_q5_0_valid() {
    // Q5_0: 22 bytes per block
    let mut data = vec![0u8; 22];
    // Scale = 1.0 (f16)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // High bits (4 bytes)
    // Low bits (16 bytes)
    // All zeros = 0 - 16 = -16 (per dequant formula)

    let result = dequantize_q5_0(&data).expect("test");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_0_invalid_length_cov() {
    let data = vec![0u8; 21]; // Not multiple of 22
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_valid() {
    // Q5_1: 24 bytes per block
    let mut data = vec![0u8; 24];
    // Scale = 1.0 (f16)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Min = 0.0 (f16)
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // High bits (4 bytes), low bits (16 bytes)

    let result = dequantize_q5_1(&data).expect("test");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_1_invalid_length_cov() {
    let data = vec![0u8; 23]; // Not multiple of 24
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_f16
// =========================================================================

#[test]
fn test_dequantize_f16_valid() {
    // Create f16 data for value 1.0
    let f16_one = half::f16::from_f32(1.0).to_le_bytes();
    let f16_two = half::f16::from_f32(2.0).to_le_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&f16_one);
    data.extend_from_slice(&f16_two);

    let result = dequantize_f16(&data).expect("test");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[1] - 2.0).abs() < 0.001);
}

#[test]
fn test_dequantize_f16_invalid_length_cov() {
    let data = vec![0u8; 3]; // Not multiple of 2
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: DequantStats and SimdBackend
// =========================================================================

#[test]
fn test_dequant_stats_debug_clone() {
    let stats = DequantStats {
        blocks_processed: 100,
        bytes_processed: 800,
        simd_backend: SimdBackend::Avx2,
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("DequantStats"));
    assert!(debug.contains("100"));

    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, stats.blocks_processed);
    assert_eq!(cloned.bytes_processed, stats.bytes_processed);
}

#[test]
fn test_simd_backend_debug_clone() {
    let backend = SimdBackend::Avx2;
    let debug = format!("{:?}", backend);
    assert!(debug.contains("Avx2"));

    let cloned = backend;
    assert_eq!(cloned, backend);

    // Test equality
    assert_eq!(SimdBackend::Sse2, SimdBackend::Sse2);
    assert_ne!(SimdBackend::Avx2, SimdBackend::Scalar);
}

#[test]
fn test_detect_simd_backend_cov() {
    let backend = detect_simd_backend();
    let debug = format!("{:?}", backend);
    // Just verify it returns something valid
    assert!(
        backend == SimdBackend::Avx2
            || backend == SimdBackend::Sse2
            || backend == SimdBackend::Neon
            || backend == SimdBackend::Scalar
    );
    println!("Detected SIMD backend: {}", debug);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8k_into error paths
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_invalid_length_cov() {
    let activations = vec![0.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_small_scales_buffer() {
    let activations = vec![0.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Too small
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_small_quants_buffer() {
    let activations = vec![0.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_to_q8_blocks
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_valid() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let blocks = quantize_to_q8_blocks(&values).expect("test");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_error_cov() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_blocks() {
    let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 10.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("test");
    let dequant = dequantize_q8_blocks(&blocks);

    assert_eq!(dequant.len(), 32);
    for (o, d) in values.iter().zip(dequant.iter()) {
        assert!((o - d).abs() < 0.02);
    }
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock::quantize_into
// =========================================================================

#[test]
fn test_q8ksuperblock_quantize_into() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();
    let mut scale: f32 = 0.0;
    let mut quants = vec![0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // Verify some quants are non-zero
    let non_zero: usize = quants.iter().filter(|&&q| q != 0).count();
    assert!(non_zero > 200);
}

// =========================================================================
// Coverage Tests: f16_to_f32 additional edge cases
// =========================================================================

#[test]
fn test_f16_to_f32_half_cov() {
    // f16 representation of 0.5 is 0x3800
    let half = f16_to_f32(0x3800);
    assert!((half - 0.5).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_two_cov() {
    // f16 representation of 2.0 is 0x4000
    let two = f16_to_f32(0x4000);
    assert!((two - 2.0).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_small_positive_cov() {
    // f16 representation of small positive number
    let small = f16_to_f32(0x0001);
    assert!(small > 0.0 && small < 0.0001);
}

// =========================================================================
// Coverage Tests: dequantize_q4_0 error paths
// =========================================================================

#[test]
fn test_dequantize_q4_0_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_0(&data);
    // Empty data should work but return empty
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_0_invalid_size_cov() {
    // Q4_0 block is 18 bytes (2 for scale + 16 for 32 nibbles)
    let data = vec![0u8; 17]; // Invalid size
    let result = dequantize_q4_0(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_q8_0 error paths
// =========================================================================

#[test]
fn test_dequantize_q8_0_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_invalid_size_cov() {
    // Q8_0 block is 34 bytes (2 for scale + 32 for int8)
    let data = vec![0u8; 33]; // Invalid size
    let result = dequantize_q8_0(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_f16 error paths
// =========================================================================

#[test]
fn test_dequantize_f16_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_f16_odd_size_cov() {
    // f16 requires 2 bytes per value
    let data = vec![0u8; 3]; // Odd, invalid
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: Q8_0Block methods
// =========================================================================

#[test]
fn test_q8_0block_quantization_error_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Error should be small
    assert!(error < 0.1);
}

#[test]
fn test_q8_0block_relative_error_cov() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 + 1.0) * 0.1);
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // Relative error should be small
    assert!(rel_error < 0.1);
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock methods
// =========================================================================

#[test]
fn test_q8ksuperblock_quantize_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) * 0.01);
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0 || values.iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn test_q8ksuperblock_dequantize_cov() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) * 0.01);
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();
    assert_eq!(dequant.len(), 256);
    // Should roughly match original
    for (o, d) in values.iter().zip(dequant.iter()) {
        assert!((o - d).abs() < 0.05);
    }
}

// =========================================================================
// Coverage Tests: InterleavedQ4K methods
// =========================================================================

#[test]
fn test_interleavedq4k_num_values_cov() {
    // Create a minimal Q4K data (256 values per super-block)
    // Q4_K super-block is 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data);
    assert!(interleaved.is_ok());
    let interleaved = interleaved.expect("quantization failed");
    assert_eq!(interleaved.num_values(), 256);
}

// =========================================================================
// Coverage Tests: fused_q4k_dot error paths
// =========================================================================

#[test]
fn test_fused_q4k_dot_length_mismatch_cov() {
    let q4k_data = vec![0u8; 144]; // 1 super-block = 256 values
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q6k_dot error paths
// =========================================================================

#[test]
fn test_fused_q6k_dot_length_mismatch_cov() {
    // Q6_K super-block is 210 bytes for 256 values
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = fused_q6k_dot(&q6k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q5k_dot error paths
// =========================================================================

#[test]
fn test_fused_q5k_dot_length_mismatch_cov() {
    // Q5_K super-block is 176 bytes for 256 values
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = fused_q5k_dot(&q5k_data, &activations);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_activations_q8k_into success path
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_success_cov() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
}

// =========================================================================
// Coverage Tests: dequantize_q4_1 error paths
// =========================================================================

#[test]
fn test_dequantize_q4_1_invalid_size_cov() {
    // Q4_1 block is 20 bytes (4 bytes min + 4 bytes max + 16 nibbles) for 32 values
    let data = vec![0u8; 19]; // Invalid - not a multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_1_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q5_0 error paths
// =========================================================================

#[test]
fn test_dequantize_q5_0_invalid_size_cov() {
    // Q5_0 block is 22 bytes (2 scale + 4 high bits + 16 nibbles) for 32 values
    let data = vec![0u8; 21]; // Invalid
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_0_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: dequantize_q5_1 error paths
// =========================================================================

#[test]
fn test_dequantize_q5_1_invalid_size_cov() {
    // Q5_1 block is 24 bytes
    let data = vec![0u8; 23]; // Invalid
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: fused_q4k_q8_dot error paths
// =========================================================================

#[test]
fn test_fused_q4k_q8_dot_invalid_q4k_size_cov() {
    let q4k_data = vec![0u8; 100]; // Invalid - not a multiple of 144
    let values: [f32; 32] = [0.0; 32];
    let q8_blocks = vec![Q8_0Block::quantize(&values); 8];
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8_dot_block_mismatch_cov() {
    let q4k_data = vec![0u8; 144]; // 1 super-block = 256 values
    let values: [f32; 32] = [0.0; 32];
    let q8_blocks = vec![Q8_0Block::quantize(&values); 4]; // 4 * 32 = 128 values - mismatch
    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: fused_q4k_q8k_dot error paths
// =========================================================================

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_size_cov() {
    let q4k_data = vec![0u8; 100]; // Invalid
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_scale_too_few_cov() {
    let q4k_data = vec![0u8; 288]; // 2 super-blocks = 512 values
    let scales = vec![1.0f32; 1]; // Only 1 scale but need 2
    let quants = vec![0i8; 512];
    let result = fused_q4k_q8k_dot(&q4k_data, &scales, &quants);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: quantize_to_q8_blocks error paths
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_not_multiple_cov() {
    let values = vec![1.0f32; 30]; // Not a multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_empty_cov() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_quantize_to_q8_blocks_one_block_cov() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 1);
}

// =========================================================================
// Coverage Tests: dequantize_q8_blocks
// =========================================================================

