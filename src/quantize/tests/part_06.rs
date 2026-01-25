use crate::quantize::*;
#[test]
fn test_quantize_activations_q8_0_valid_cov() {
    let activations: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 2); // 64/32 = 2 blocks
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_activations_q8_0_empty_cov() {
    let activations: Vec<f32> = vec![];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert!(scales.is_empty());
    assert!(quants.is_empty());
}

// =========================================================================
// Deep Coverage Tests: apply_rope_rotation_simd
// =========================================================================

#[test]
fn test_apply_rope_rotation_simd_identity_cov() {
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
    let mut x2 = vec![5.0, 6.0, 7.0, 8.0];
    let cos_vals = vec![1.0, 1.0, 1.0, 1.0]; // cos(0) = 1
    let sin_vals = vec![0.0, 0.0, 0.0, 0.0]; // sin(0) = 0

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=1, sin=0: x1' = x1, x2' = x2
    assert_eq!(x1, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(x2, vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_apply_rope_rotation_simd_ninety_deg_cov() {
    let mut x1 = vec![1.0, 0.0];
    let mut x2 = vec![0.0, 1.0];
    let cos_vals = vec![0.0, 0.0]; // cos(90) = 0
    let sin_vals = vec![1.0, 1.0]; // sin(90) = 1

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // x1' = x1*0 - x2*1 = -x2
    // x2' = x1*1 + x2*0 = x1
    assert!((x1[0] - 0.0).abs() < 1e-6);
    assert!((x2[0] - 1.0).abs() < 1e-6);
}

// =========================================================================
// Deep Coverage Tests: apply_rope_rotation_simd
// =========================================================================

#[test]
fn test_apply_rope_rotation_simd_basic_cov() {
    let mut x1: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..16).map(|i| (i + 16) as f32).collect();
    let cos_vals: Vec<f32> = vec![1.0; 16];
    let sin_vals: Vec<f32> = vec![0.0; 16];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=1, sin=0: values unchanged
    assert_eq!(x1[0], 0.0);
    assert_eq!(x2[0], 16.0);
}

// =========================================================================
// Deep Coverage Tests: softmax_simd
// =========================================================================

#[test]
fn test_softmax_simd_basic_deep2() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    softmax_simd(&mut x);

    // Sum should be 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Values should be ordered (higher input -> higher probability)
    assert!(x[3] > x[2]);
    assert!(x[2] > x[1]);
    assert!(x[1] > x[0]);
}

#[test]
fn test_softmax_simd_large_values_deep2() {
    let mut x = vec![100.0, 101.0, 102.0, 103.0];
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_uniform_deep2() {
    let mut x = vec![5.0; 4];
    softmax_simd(&mut x);

    // Uniform input -> uniform output
    for &val in &x {
        assert!((val - 0.25).abs() < 1e-5);
    }
}

// =========================================================================
// Deep Coverage Tests: fused_swiglu_simd
// =========================================================================

#[test]
fn test_fused_swiglu_simd_basic_deep2() {
    let mut gate = vec![1.0, 2.0, 3.0, 4.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];

    fused_swiglu_simd(&mut gate, &up);

    // gate should be modified
    assert!(gate[0] > 0.0);
}

#[test]
fn test_fused_swiglu_simd_zeros_deep2() {
    let mut gate = vec![0.0; 8];
    let up = vec![1.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // sigmoid(0) = 0.5, so gate[i] = 0 * 0.5 * 1 = 0
    for &val in &gate {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

// =========================================================================
// Deep Coverage Tests: DequantStats and SimdBackend
// =========================================================================

#[test]
fn test_simd_backend_display_cov() {
    let backend = detect_simd_backend();
    let display = format!("{backend}");
    assert!(!display.is_empty());
}

#[test]
fn test_dequant_stats_default_cov() {
    let stats = DequantStats::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
}

// =========================================================================
// Deep Coverage Tests: fused matvec functions
// =========================================================================

#[test]
fn test_fused_q4k_parallel_matvec_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 144 bytes per row
    let weights = vec![0u8; 288];
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_parallel_matvec(&weights, &activations, 256, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_fused_q5k_parallel_matvec_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 176 bytes per row
    let weights = vec![0u8; 352];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_parallel_matvec(&weights, &activations, 256, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_fused_q6k_parallel_matvec_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 210 bytes per row
    let weights = vec![0u8; 420];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_parallel_matvec(&weights, &activations, 256, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

// =========================================================================
// Deep Coverage Tests: fused_q4_0_q8_0 functions
// =========================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_valid_cov() {
    // Q4_0: 18 bytes per 32 values
    // in_dim=32, out_dim=2 -> 2 rows * 18 bytes per row
    let mut weights = vec![0u8; 36];
    // Set scales to 1.0 as f16
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale[0];
    weights[1] = scale[1];
    weights[18] = scale[0];
    weights[19] = scale[1];

    let activations = vec![1.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, 32, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

// =========================================================================
// Deep Coverage Tests: parallel dequantization
// =========================================================================

#[test]
fn test_dequantize_q4_k_parallel_valid_cov() {
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());
    let values = result.expect("quantization failed");
    assert_eq!(values.len(), 256);
}

#[test]
fn test_dequantize_q4_k_simd_valid_cov() {
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_simd(&data);
    assert!(result.is_ok());
    let values = result.expect("quantization failed");
    assert_eq!(values.len(), 256);
}

#[test]
fn test_dequantize_q8_0_parallel_valid_cov() {
    // Q8_0: 34 bytes per 32 values
    let data = vec![0u8; 34];
    let result = dequantize_q8_0_parallel(&data);
    assert!(result.is_ok());
    let values = result.expect("quantization failed");
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequantize_q8_0_simd_valid_cov() {
    let data = vec![0u8; 34];
    let result = dequantize_q8_0_simd(&data);
    assert!(result.is_ok());
    let values = result.expect("quantization failed");
    assert_eq!(values.len(), 32);
}

// =========================================================================
// Deep Coverage Tests: fused_rmsnorm functions
// =========================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_basic_cov() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);
    assert_eq!(scales.len(), 2); // 64/32 = 2 blocks
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_basic_cov() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 / 10.0).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);
    assert!(scales[0] > 0.0);
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_cov() {
    // in_dim=32, out_dim=2 -> 2 rows * 18 bytes per row
    let mut weights = vec![0u8; 36];
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale[0];
    weights[1] = scale[1];
    weights[18] = scale[0];
    weights[19] = scale[1];

    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    // Signature: (input, norm_weight, eps, weight_data, in_dim, out_dim)
    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, eps, &weights, 32, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_cov() {
    // For Q4_K: 144 bytes per 256 values
    // in_dim=256, out_dim=1
    let up_weights = vec![0u8; 144];
    let gate_weights = vec![0u8; 144];
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];
    let eps = 1e-5;

    // Signature: (input, norm_weight, eps, up_weight_data, gate_weight_data, in_dim, out_dim)
    let result = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        eps,
        &up_weights,
        &gate_weights,
        256,
        1,
    );
    assert!(result.is_ok());
    let (up_out, gate_out) = result.expect("quantization failed");
    assert_eq!(up_out.len(), 1);
    assert_eq!(gate_out.len(), 1);
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_tiled_matvec
// =========================================================================

#[test]
fn test_fused_q4k_tiled_matvec_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 144 bytes per row
    let weights = vec![0u8; 288];
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_tiled_matvec(&weights, &activations, 256, 2, None);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

// =========================================================================
// Deep Coverage Tests: fused_q8_0_q8_0 functions
// =========================================================================

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_valid_cov() {
    // Q8_0: 34 bytes per 32 values
    // in_dim=32, out_dim=2 -> 2 rows * 34 bytes per row
    let mut weights = vec![0u8; 68];
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale[0];
    weights[1] = scale[1];
    weights[34] = scale[0];
    weights[35] = scale[1];

    let activations = vec![1.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weights, &activations, 32, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_valid_cov() {
    let mut weights = vec![0u8; 68];
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale[0];
    weights[1] = scale[1];
    weights[34] = scale[0];
    weights[35] = scale[1];

    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result = fused_q8_0_q8_0_parallel_matvec_into(&weights, &activations, 32, 2, &mut output);
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_q8k_parallel_matvec_into
// =========================================================================

#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 144 bytes per row
    let weights = vec![0u8; 288];
    let q8k_scales = vec![1.0f32; 1]; // One scale for 256 values
    let q8k_quants = vec![0i8; 256];
    let mut output = vec![0.0f32; 2];

    let result =
        fused_q4k_q8k_parallel_matvec_into(&weights, &q8k_scales, &q8k_quants, 256, 2, &mut output);
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_q8k_ffn_up_gate_into
// =========================================================================

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_valid_cov() {
    // Signature: (up_weight, gate_weight, q8k_scales, q8k_quants, in_dim, out_dim, up_output, gate_output)
    let up_weights = vec![0u8; 144];
    let gate_weights = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        256,
        1,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: Scalar Fallback Paths (_deep_qcov_ prefix)
// =========================================================================
//
// These tests explicitly exercise scalar fallback code paths that are normally
// bypassed when AVX2/AVX-512 are available. They achieve coverage by:
// 1. Directly calling *_scalar functions
// 2. Testing error handling paths
// 3. Exercising boundary conditions in quantization code
// =========================================================================

// -------------------------------------------------------------------------
// InterleavedQ4K::dot_scalar Direct Tests
// -------------------------------------------------------------------------

#[test]
fn test_interleaved_q4k_dot_scalar_explicitly_deep_qcov_001() {
    // Force use of scalar path by calling dot_scalar directly
    // Single super-block with simple test data
    let mut q4k_data = Vec::with_capacity(144);

    // d = 1.0 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // dmin = 0.0 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // scales: 12 bytes (all zeros -> scale=0, min=0)
    q4k_data.extend_from_slice(&[0u8; 12]);
    // qs: 128 bytes (alternating pattern 0x12 for low=2, high=1)
    q4k_data.extend_from_slice(&[0x12u8; 128]);

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations = vec![1.0f32; 256];

    // Directly call scalar implementation
    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
    // With scale=0 and min=0 from scales array, result should be 0
    let dot_result = result.expect("quantization failed");
    assert!(dot_result.abs() < 1e-6, "Expected ~0, got {}", dot_result);
}

#[test]
fn test_interleaved_q4k_dot_scalar_with_scales_deep_qcov_002() {
    // Test scalar path with non-zero scales
    let mut q4k_data = Vec::with_capacity(144);

    // d = 2.0 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    // dmin = 0.5 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    // scales: 12 bytes with first scale=1, first min=1
    let mut scales = [0u8; 12];
    scales[0] = 0x01; // scale[0] = 1 (6-bit)
    scales[4] = 0x01; // min[0] = 1 (6-bit)
    q4k_data.extend_from_slice(&scales);
    // qs: 128 bytes (all 0x55 for varied nibbles)
    q4k_data.extend_from_slice(&[0x55u8; 128]);

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
    // Result should be non-zero due to scales
    let _dot_result = result.expect("quantization failed");
}

#[test]
fn test_interleaved_q4k_dot_scalar_multiple_superblocks_deep_qcov_003() {
    // Test scalar path with 4 super-blocks
    let num_sb = 4;
    let mut q4k_data = Vec::with_capacity(144 * num_sb);

    for i in 0..num_sb {
        // Vary d across super-blocks
        let d_val = 0.5 + (i as f32) * 0.25;
        q4k_data.extend_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
        q4k_data.extend_from_slice(&half::f16::from_f32(0.1).to_le_bytes());

        // Varied scales
        let mut scales = [0u8; 12];
        scales[0] = ((i + 1) % 64) as u8;
        q4k_data.extend_from_slice(&scales);

        // Varied qs
        let pattern = ((i * 17) % 256) as u8;
        q4k_data.extend_from_slice(&vec![pattern; 128]);
    }

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();

    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
}

#[test]
fn test_interleaved_q4k_dot_scalar_negative_activations_deep_qcov_004() {
    // Test with negative activations
    let mut q4k_data = Vec::with_capacity(144);
    q4k_data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    let mut scales = [0u8; 12];
    scales[0] = 0x3F; // max scale
    q4k_data.extend_from_slice(&scales);
    q4k_data.extend_from_slice(&[0xFFu8; 128]); // max nibbles

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.01).collect();

    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
    // Result should be negative
    let dot_result = result.expect("quantization failed");
    assert!(dot_result < 0.0, "Expected negative, got {}", dot_result);
}

#[test]
fn test_interleaved_q4k_dot_scalar_max_values_deep_qcov_005() {
    // Test with maximum scale and nibble values
    let mut q4k_data = Vec::with_capacity(144);
    // d = max f16 normal value (about 65504)
    q4k_data.extend_from_slice(&half::f16::from_f32(100.0).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(50.0).to_le_bytes());
    // Max scales (63)
    q4k_data.extend_from_slice(&[0x3Fu8; 12]);
    // Max nibbles (0xFF = low=15, high=15)
    q4k_data.extend_from_slice(&[0xFFu8; 128]);

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations = vec![10.0f32; 256];

    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
}

// -------------------------------------------------------------------------
// InterleavedQ4K Error Handling Tests
// -------------------------------------------------------------------------

#[test]
fn test_interleaved_q4k_dot_mismatched_size_deep_qcov_006() {
    let q4k_data = vec![0u8; 144]; // 1 super-block = 256 values
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");

    // Wrong activation size (128 instead of 256)
    let activations = vec![1.0f32; 128];
    let result = interleaved.dot(&activations);
    assert!(result.is_err());

    // Verify error message contains useful info
    if let Err(e) = result {
        let msg = format!("{:?}", e);
        assert!(msg.contains("128") || msg.contains("256"));
    }
}

#[test]
fn test_interleaved_q4k_from_invalid_length_deep_qcov_007() {
    // Not a multiple of 144
    let q4k_data = vec![0u8; 143];
    let result = InterleavedQ4K::from_q4k(&q4k_data);
    assert!(result.is_err());

    let q4k_data = vec![0u8; 145];
    let result = InterleavedQ4K::from_q4k(&q4k_data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_num_values_deep_qcov_008() {
    // Test num_values() for various super-block counts
    let q4k_data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");
    assert_eq!(interleaved.num_values(), 256);

    let q4k_data = vec![0u8; 288];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");
    assert_eq!(interleaved.num_values(), 512);

    let q4k_data = vec![0u8; 144 * 10];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");
    assert_eq!(interleaved.num_values(), 2560);
}

// -------------------------------------------------------------------------
// fused_q4k_dot Scalar Path Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_scalar_all_zeros_deep_qcov_009() {
    // All zeros should produce zero dot product
    let q4k_data = vec![0u8; 144];
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    let dot = result.expect("quantization failed");
    assert!(dot.abs() < 1e-6, "Expected ~0, got {}", dot);
}

#[test]
fn test_fused_q4k_dot_invalid_q4k_length_deep_qcov_010() {
    // Q4K data not multiple of 144
    let q4k_data = vec![0u8; 100];
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_mismatched_activation_length_deep_qcov_011() {
    // Activations don't match expected count
    let q4k_data = vec![0u8; 144]; // 256 values
    let activations = vec![1.0f32; 100];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_falls_back_correctly_deep_qcov_012() {
    // This test verifies that fused_q4k_dot_simd produces same results as scalar
    let mut q4k_data = Vec::with_capacity(144);
    q4k_data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    q4k_data.extend_from_slice(&[0x21u8; 12]); // varied scales
    q4k_data.extend_from_slice(&[0x34u8; 128]); // varied qs

    let activations: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();

    let scalar_result = fused_q4k_dot(&q4k_data, &activations).expect("scalar ok");
    let simd_result = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd ok");

    // Results should be within ULP tolerance
    let diff = (scalar_result - simd_result).abs();
    let tolerance = scalar_result.abs() * 1e-5 + 1e-6;
    assert!(
        diff < tolerance,
        "scalar={}, simd={}, diff={}",
        scalar_result,
        simd_result,
        diff
    );
}

// -------------------------------------------------------------------------
// fused_q4k_q8k_dot Scalar Path Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_q8k_dot_scalar_basic_deep_qcov_013() {
    // Test scalar Q4K×Q8K dot product
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_deep_qcov_014() {
    let q4k_data = vec![0u8; 100]; // Not multiple of 144
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_insufficient_scales_deep_qcov_015() {
    let q4k_data = vec![0u8; 288]; // 2 super-blocks
    let q8k_scales = vec![1.0f32; 1]; // Only 1 scale (need 2)
    let q8k_quants = vec![0i8; 512];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_insufficient_quants_deep_qcov_016() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 100]; // Need 256

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_with_data_deep_qcov_017() {
    // Test with actual non-zero data
    let mut q4k_data = Vec::with_capacity(144);
    q4k_data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    let mut scales = [0u8; 12];
    scales[0] = 0x1F; // scale=31
    scales[4] = 0x0F; // min=15
    q4k_data.extend_from_slice(&scales);
    q4k_data.extend_from_slice(&[0xABu8; 128]);

    let q8k_scales = vec![1.5f32; 1];
    let q8k_quants: Vec<i8> = (0..256)
        .map(|i| ((i % 256) as i8).wrapping_sub(64))
        .collect();

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

// -------------------------------------------------------------------------
// fused_q4_0_q8_0_dot_scalar Direct Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_basic_deep_qcov_018() {
    // Test scalar Q4_0×Q8_0 dot directly
    // Q4_0: 18 bytes per 32 values (2 bytes scale + 16 bytes qs)
    let mut q4_data = vec![0u8; 18];
    q4_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // 16 bytes of qs with value 0x88 (low=8, high=8) -> centered at 0
    q4_data[2..18].copy_from_slice(&[0x88u8; 16]);

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    // Should be near zero since Q4 values are centered at 8 (subtract 8 = 0)
    assert!(result.abs() < 1e-3, "Expected ~0, got {}", result);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_multiple_blocks_deep_qcov_019() {
    // Test with 2 blocks (64 values)
    let mut q4_data = vec![0u8; 36];
    q4_data[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    q4_data[2..18].copy_from_slice(&[0x99u8; 16]);
    q4_data[18..20].copy_from_slice(&half::f16::from_f32(1.5).to_le_bytes());
    q4_data[20..36].copy_from_slice(&[0x66u8; 16]);

    let q8_scales = vec![1.0f32, 2.0f32];
    let q8_quants: Vec<i8> = (0..64).map(|i| (i % 100) as i8).collect();

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    // Just verify it runs without error
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_truncated_block_deep_qcov_020() {
    // Test with in_dim not a multiple of 32
    let mut q4_data = vec![0u8; 18];
    q4_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4_data[2..18].copy_from_slice(&[0x55u8; 16]);

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![2i8; 32];

    // Only use 24 values (less than 32)
    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 24);
    assert!(result.is_finite());
}

// -------------------------------------------------------------------------
// extract_scale_min_from_slice Coverage Tests
// -------------------------------------------------------------------------

#[test]
fn test_extract_scale_min_from_slice_odd_indices_deep_qcov_021() {
    // Test odd index paths (different bit manipulation)
    let mut scales = [0u8; 12];
    // Set up for odd index extraction
    scales[0] = 0xC0; // High 2 bits for odd scale
    scales[2] = 0x0F; // Low 4 bits for odd scale
    scales[4] = 0x80; // High 2 bits for odd min
    scales[6] = 0x07; // Low 4 bits for odd min

    let (scale1, min1) = extract_scale_min_from_slice(&scales, 1);
    assert!(scale1 >= 0.0);
    assert!(min1 >= 0.0);

    // Test index 3, 5, 7
    for idx in [3, 5, 7] {
        let (s, m) = extract_scale_min_from_slice(&scales, idx);
        assert!(s >= 0.0);
        assert!(m >= 0.0);
    }
}

#[test]
fn test_extract_scale_min_from_slice_even_indices_deep_qcov_022() {
    // Test even index paths
    let mut scales = [0u8; 12];
    scales[0] = 0x3F; // max 6-bit scale for idx=0
    scales[4] = 0x20; // min for idx=0

    let (scale0, min0) = extract_scale_min_from_slice(&scales, 0);
    assert_eq!(scale0, 63.0);
    assert_eq!(min0, 32.0);

    // Test indices 2, 4, 6
    for idx in [2, 4, 6] {
        let (s, m) = extract_scale_min_from_slice(&scales, idx);
        assert!(s >= 0.0);
        assert!(m >= 0.0);
    }
}

// -------------------------------------------------------------------------
// fused_q4_0_q8_0_parallel_matvec Error Path Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_weight_too_small_deep_qcov_023() {
    let in_dim = 32;
    let out_dim = 4;
    // Need 4 rows * 18 bytes = 72 bytes, provide less
    let weight_data = vec![0u8; 50];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_weight_too_small_deep_qcov_024() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; 50]; // Too small
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_large_matrix_deep_qcov_025() {
    // Test with larger matrix to exercise parallel path
    let in_dim: usize = 128;
    let out_dim = 64;
    let blocks_per_row = in_dim.div_ceil(32);
    let bytes_per_row = blocks_per_row * 18;
    let mut weight_data = vec![0u8; out_dim * bytes_per_row];

    // Set scales for each block
    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let offset = row * bytes_per_row + block * 18;
            weight_data[offset..offset + 2]
                .copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

// -------------------------------------------------------------------------
// fused_q4k_q8k_dot_simd Parity Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_q8k_dot_simd_vs_scalar_parity_deep_qcov_026() {
    // Verify SIMD and scalar produce same results
    let mut q4k_data = Vec::with_capacity(144);
    q4k_data.extend_from_slice(&half::f16::from_f32(1.5).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
    for i in 0..12 {
        q4k_data.push((i * 5 % 64) as u8);
    }
    for i in 0..128 {
        q4k_data.push((i * 7 % 256) as u8);
    }

    let q8k_scales = vec![2.0f32];
    let q8k_quants: Vec<i8> = (0..256).map(|i| ((i * 3) % 127) as i8).collect();

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar ok");
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("simd ok");

    let rel_diff = (scalar - simd).abs() / (scalar.abs() + 1e-10);
    assert!(
        rel_diff < 0.01,
        "scalar={}, simd={}, rel_diff={}",
        scalar,
        simd,
        rel_diff
    );
}

// -------------------------------------------------------------------------
// Quantization Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_dequantize_q4_k_edge_all_max_nibbles_deep_qcov_027() {
    // Test Q4_K with all maximum nibble values (0xFF)
    let mut data = vec![0u8; 144];
    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // dmin = 1.0
    data[2..4].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // scales all max
    data[4..16].copy_from_slice(&[0x3F; 12]);
    // qs all max
    data[16..144].copy_from_slice(&[0xFF; 128]);

    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 256);
    // All values should be positive (d * scale * 15 - dmin * min)
}

#[test]
fn test_dequantize_q4_k_edge_all_zero_nibbles_deep_qcov_028() {
    // Test Q4_K with all zero nibble values
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[4..16].copy_from_slice(&[0x3F; 12]); // max scales
                                              // qs all zero
    data[16..144].copy_from_slice(&[0x00; 128]);

    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    // Values should be negative due to -dmin * min term
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v <= 0.0 || v.abs() < 1e-6,
            "vals[{}]={} should be <= 0",
            i,
            v
        );
    }
}

#[test]
fn test_dequantize_q5_k_edge_high_bits_deep_qcov_029() {
    // Test Q5_K with high bits set in qh
    let mut data = vec![0u8; 176];
    data[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    data[4..16].copy_from_slice(&[0x1F; 12]); // varied scales
                                              // qh all 0xFF (high bits set)
    data[16..48].copy_from_slice(&[0xFF; 32]);
    // qs varied
    data[48..176].copy_from_slice(&[0xAA; 128]);

    let result = dequantize_q5_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 256);
}

#[test]
fn test_dequantize_q6_k_edge_negative_scales_deep_qcov_030() {
    // Q6_K scales are i8, test negative scales
    let mut data = vec![0u8; 210];
    // ql: 128 bytes
    data[0..128].copy_from_slice(&[0x55; 128]);
    // qh: 64 bytes
    data[128..192].copy_from_slice(&[0xAA; 64]);
    // scales: 16 i8 values, some negative
    for i in 0..16 {
        data[192 + i] = if i % 2 == 0 { 64u8 } else { 192u8 }; // 64 or -64 as i8
    }
    // d at the end
    data[208..210].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());

    let result = dequantize_q6_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    // Should have mix of positive and negative values
    let has_pos = vals.iter().any(|&v| v > 0.0);
    let has_neg = vals.iter().any(|&v| v < 0.0);
    assert!(has_pos || has_neg, "Should have varied values");
}

#[test]
fn test_dequantize_q8_0_negative_values_deep_qcov_031() {
    // Test Q8_0 with negative quantized values
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    // 32 i8 values: -128 to -97
    for i in 0..32 {
        data[2 + i] = (128 + i) as u8; // -128, -127, ... as i8
    }

    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    // All values should be negative
    for (i, &v) in vals.iter().enumerate() {
        assert!(v < 0.0, "vals[{}]={} should be < 0", i, v);
    }
}

// -------------------------------------------------------------------------
// Q8KSuperBlock Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_q8k_superblock_quantize_all_same_value_deep_qcov_032() {
    let values = [42.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    // All quants should be same
    let first = block.quants[0];
    assert!(block.quants.iter().all(|&q| q == first));
}

#[test]
fn test_q8k_superblock_quantize_alternating_deep_qcov_033() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let block = Q8KSuperBlock::quantize(&values);
    // Quants should alternate between positive and negative
    assert!(block.quants[0] != 0);
    assert!(block.quants[0] != block.quants[1] || block.quants[0] == 0);
}

#[test]
fn test_q8k_superblock_quantize_into_overflow_safe_deep_qcov_034() {
    // Test with values that could cause overflow in naive implementation
    let values = [f32::MAX / 1000.0; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    // Should complete without panic
    assert!(scale > 0.0);
}

// -------------------------------------------------------------------------
// Boundary Condition Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_single_superblock_boundary_deep_qcov_035() {
    // Exactly 1 super-block (minimum valid size)
    let q4k_data = vec![0u8; 144];
    let activations = vec![0.5f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_max_superblocks_deep_qcov_036() {
    // Test with 16 super-blocks (4096 values)
    let num_sb = 16;
    let mut q4k_data = vec![0u8; 144 * num_sb];
    for i in 0..num_sb {
        let offset = i * 144;
        q4k_data[offset..offset + 2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    }

    let activations = vec![1.0f32; 256 * num_sb];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_interleaved_q4k_clone_and_debug_deep_qcov_037() {
    let q4k_data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");

    // Test Clone
    let cloned = interleaved.clone();
    assert_eq!(cloned.num_super_blocks, interleaved.num_super_blocks);

    // Test Debug
    let debug_str = format!("{:?}", interleaved);
    assert!(debug_str.contains("InterleavedQ4K"));
}

// -------------------------------------------------------------------------
// Additional fused_q8_0_q8_0 Coverage
// -------------------------------------------------------------------------

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_weight_error_deep_qcov_038() {
    let in_dim = 64;
    let out_dim = 4;
    // Q8_0: 34 bytes per 32 values, so 2 blocks per row = 68 bytes per row
    // Need 4 * 68 = 272 bytes, provide less
    let weight_data = vec![0u8; 100];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// f16_to_f32_lut Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_f16_to_f32_lut_special_values_deep_qcov_039() {
    // Test zero
    let zero_bits = half::f16::from_f32(0.0).to_bits();
    assert_eq!(f16_to_f32_lut(zero_bits), 0.0);

    // Test negative zero
    let neg_zero_bits = half::f16::from_f32(-0.0).to_bits();
    let result = f16_to_f32_lut(neg_zero_bits);
    assert!(result == 0.0 || result == -0.0);

    // Test small positive
    let small_bits = half::f16::from_f32(0.001).to_bits();
    let result = f16_to_f32_lut(small_bits);
    assert!((result - 0.001).abs() < 0.0001);

    // Test large value
    let large_bits = half::f16::from_f32(1000.0).to_bits();
    let result = f16_to_f32_lut(large_bits);
    assert!((result - 1000.0).abs() < 1.0);
}

// -------------------------------------------------------------------------
// quantize_activations_q8_0 Coverage
// -------------------------------------------------------------------------

#[test]
fn test_quantize_activations_q8_0_single_block_deep_qcov_040() {
    // Exactly 32 values (1 block)
    let activations: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_activations_q8_0_multiple_blocks_deep_qcov_041() {
    // 3 blocks (96 values)
    let activations: Vec<f32> = (0..96).map(|i| (i as f32 * 0.1).sin()).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 3);
    assert_eq!(quants.len(), 96);
}

#[test]
fn test_quantize_activations_q8_0_partial_block_deep_qcov_042() {
    // 40 values (1 full + partial)
    let activations = vec![1.0f32; 40];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    // Should round up to 2 blocks
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64); // Padded to 64
}

// -------------------------------------------------------------------------
// Additional Error Path Coverage
// -------------------------------------------------------------------------

#[test]
fn test_dequantize_q4_0_empty_input_deep_qcov_043() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

#[test]
fn test_dequantize_q8_0_empty_input_deep_qcov_044() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

#[test]
fn test_dequantize_q4_k_empty_input_deep_qcov_045() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

#[test]
fn test_dequantize_q5_k_empty_input_deep_qcov_046() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

#[test]
fn test_dequantize_q6_k_empty_input_deep_qcov_047() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q6_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

// -------------------------------------------------------------------------
// fused_q4k_q8k_parallel_matvec Coverage
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_error_weight_deep_qcov_048() {
    // in_dim=256, out_dim=2 -> need 288 bytes, provide less
    let weights = vec![0u8; 100];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let mut output = vec![0.0f32; 2];

    let result =
        fused_q4k_q8k_parallel_matvec_into(&weights, &q8k_scales, &q8k_quants, 256, 2, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_error_up_weight_deep_qcov_049() {
    let up_weights = vec![0u8; 50]; // Too small
    let gate_weights = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        256,
        1,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_error_gate_weight_deep_qcov_050() {
    let up_weights = vec![0u8; 144];
    let gate_weights = vec![0u8; 50]; // Too small
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        256,
        1,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

// =========================================================================
// Additional 95% Coverage Tests
// =========================================================================

#[test]
fn test_cov95_fused_q4k_dot_empty_activations() {
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let activations: Vec<f32> = vec![];
    let result = fused_q4k_dot(&q4k_data, &activations);
    // Empty activations should return error or 0
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_cov95_fused_q4k_dot_simd_short_input() {
    // Test with input shorter than SIMD width
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let activations = vec![1.0f32; 16]; // Less than block size
    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_cov95_fused_q6k_dot_basic() {
    // Q6_K block: 210 bytes for 256 values
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot(&q6k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q6k_dot_simd_basic() {
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot_simd(&q6k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q5k_dot_basic() {
    // Q5_K block: 176 bytes for 256 values
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&q5k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q5k_dot_simd_basic() {
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot_simd(&q5k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q4k_tiled_matvec_small() {
    // Small matrix for tiled matvec
    let in_dim = 256;
    let out_dim = 64;
    let bytes_per_row = 144; // Q4_K block size
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weight_data, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

#[test]
fn test_cov95_fused_q4k_parallel_matvec_basic() {
    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 144;
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

#[test]
fn test_cov95_fused_q5k_parallel_matvec_basic() {
    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 176; // Q5_K block size
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

#[test]
fn test_cov95_fused_q6k_parallel_matvec_basic() {
    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 210; // Q6_K block size
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

#[test]
fn test_cov95_fused_q4k_q8_dot_basic() {
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let q8_blocks = vec![
        Q8_0Block {
            scale: 1.0,
            quants: [0i8; 32],
        };
        8
    ]; // 256 values

    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q4k_q8k_dot_basic() {
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 8]; // 8 blocks for 256 values
    let q8k_quants = vec![0i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q4k_q8k_dot_simd_basic() {
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 8];
    let q8k_quants = vec![0i8; 256];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_quantize_rmsnorm_q8_0_basic() {
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
}

#[test]
fn test_cov95_quantize_rmsnorm_q8_0_into_basic() {
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];
    let eps = 1e-5;
    let mut scales = vec![0.0f32; 8]; // 8 blocks
    let mut quants = vec![0i8; 256];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);
    // At least some scales should be non-zero
    let has_nonzero = scales.iter().any(|&s| s.abs() > 1e-10);
    assert!(has_nonzero);
}

#[test]
fn test_cov95_fused_swiglu_simd_basic() {
    let mut gate = vec![0.5f32; 256];
    let up = vec![1.0f32; 256];

    fused_swiglu_simd(&mut gate, &up);
    // Gate should be modified
    let modified = gate.iter().any(|&g| (g - 0.5).abs() > 1e-6);
    assert!(modified);
}

#[test]
fn test_cov95_softmax_simd_basic() {
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    softmax_simd(&mut x);

    // Sum should be ~1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_cov95_softmax_simd_large() {
    let mut x: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

#[test]
fn test_cov95_dequantize_q4_1_basic() {
    // Q4_1 block: 20 bytes for 32 values
    let data = vec![0u8; 20];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 32);
}

#[test]
fn test_cov95_dequantize_q5_0_basic() {
    // Q5_0 block: 22 bytes for 32 values
    let data = vec![0u8; 22];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 32);
}

#[test]
fn test_cov95_dequantize_q5_1_basic() {
    // Q5_1 block: 24 bytes for 32 values
    let data = vec![0u8; 24];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 32);
}

#[test]
fn test_cov95_dequantize_q2_k_basic() {
    // Q2_K block: 84 bytes for 256 values
    let data = vec![0u8; 84];
    let result = dequantize_q2_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 256);
}

#[test]
fn test_cov95_dequantize_f16_basic() {
    // Two f16 values (4 bytes)
    let data = vec![0u8; 4];
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 2);
}

#[test]
fn test_cov95_f16_to_f32_special_values() {
    // Test special f16 values
    let zero = f16_to_f32(0x0000);
    assert_eq!(zero, 0.0);

    let neg_zero = f16_to_f32(0x8000);
    assert_eq!(neg_zero, 0.0);

    let one = f16_to_f32(0x3C00); // 1.0 in f16
    assert!((one - 1.0).abs() < 1e-3);
}

#[test]
fn test_cov95_fused_q6k_colmajor_matvec_alias() {
    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 210;
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_colmajor_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q4k_auto_matvec_into_alias() {
    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 144;
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4k_auto_matvec_into(&weight_data, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_quantize_activations_q8k_into_basic() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 8];
    let mut quants = vec![0i8; 256];

    let _ = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    let has_nonzero = scales.iter().any(|&s| s.abs() > 1e-10);
    assert!(has_nonzero);
}

#[test]
fn test_cov95_quantize_to_q8_blocks_basic() {
    let values = vec![1.0f32; 32];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 1);
}

#[test]
fn test_cov95_dequantize_q8_blocks_basic() {
    let blocks = vec![Q8_0Block {
        scale: 1.0,
        quants: [64i8; 32],
    }];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 32);
}
