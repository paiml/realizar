//! Deep coverage tests for quantize module

use realizar::quantize::{
    detect_simd_backend, dequantize_f16, dequantize_q4_0, dequantize_q4_0_parallel,
    dequantize_q4_0_simd, dequantize_q4_1, dequantize_q4_k, dequantize_q4_k_parallel,
    dequantize_q4_k_simd, dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k,
    dequantize_q8_0, dequantize_q8_0_parallel, dequantize_q8_0_simd,
    dequantize_q8_0_simd_optimized, dequantize_q8_blocks, f16_to_f32,
    fused_q4_0_q8_0_parallel_matvec, fused_q4_0_q8_0_parallel_matvec_into,
    fused_q4_0_q8_0_parallel_matvec_prequant, fused_q4k_auto_matvec_into, fused_q4k_dot,
    fused_q4k_dot_simd, fused_q4k_parallel_matvec, fused_q4k_parallel_matvec_into,
    fused_q4k_q8_dot, fused_q4k_q8_dot_simd, fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd,
    fused_q4k_q8k_ffn_up_gate_into, fused_q4k_q8k_parallel_matvec_into,
    fused_q4k_tiled_matvec, fused_q5k_dot, fused_q5k_dot_simd, fused_q5k_parallel_matvec,
    fused_q5k_parallel_matvec_into, fused_q5k_tiled_matvec, fused_q6k_colmajor_matvec,
    fused_q6k_dot, fused_q6k_dot_simd, fused_q6k_parallel_matvec,
    fused_q6k_parallel_matvec_into, fused_q6k_q8k_dot_simd, fused_q6k_q8k_parallel_matvec_into,
    fused_q6k_tiled_matvec, fused_q8_0_q8_0_parallel_matvec,
    fused_q8_0_q8_0_parallel_matvec_into, fused_rmsnorm_ffn_up_gate,
    fused_rmsnorm_q4_0_matmul, fused_swiglu_simd,
    int8_matvec, int8_matvec_parallel, quantize_activations_q8_0, quantize_activations_q8k_into,
    quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into, quantize_to_q8_blocks, softmax_simd,
    apply_rope_rotation_simd, DequantStats, Int8Row, InterleavedQ4K, Q8KSuperBlock, Q8_0Block,
    SimdBackend, BLOCK_SIZE, QK_K,
};

fn f16_bytes(val: f32) -> [u8; 2] {
    half::f16::from_f32(val).to_le_bytes()
}

// Constants
#[test]
fn test_block_size_constant() { assert_eq!(BLOCK_SIZE, 32); }

#[test]
fn test_qk_k_constant() { assert_eq!(QK_K, 256); }

#[test]
fn test_simd_backend_enum_variants() {
    let _ = SimdBackend::Avx2;
    let _ = SimdBackend::Sse2;
    let _ = SimdBackend::Neon;
    let _ = SimdBackend::Scalar;
}

#[test]
fn test_detect_simd_backend() {
    let _ = detect_simd_backend();
}

#[test]
fn test_dequant_stats_struct() {
    let stats = DequantStats { blocks_processed: 10, bytes_processed: 340, simd_backend: SimdBackend::Avx2 };
    assert_eq!(stats.blocks_processed, 10);
}

// Q8_0Block
#[test]
fn test_q8_0_block_quantize_basic() {
    let block = Q8_0Block::quantize(&[1.0f32; 32]);
    assert!(block.scale > 0.0);
    assert_eq!(block.quants[0], 127);
}

#[test]
fn test_q8_0_block_quantize_zeros() {
    let block = Q8_0Block::quantize(&[0.0f32; 32]);
    assert!(block.quants.iter().all(|&q| q == 0));
}

#[test]
fn test_q8_0_block_dequantize() {
    let block = Q8_0Block::quantize(&[0.5f32; 32]);
    let deq = block.dequantize();
    assert!(deq.iter().all(|&v| (v - 0.5).abs() < 0.1));
}

#[test]
fn test_q8_0_block_quantization_error() {
    let values: [f32; 32] = core::array::from_fn(|i| i as f32 / 10.0);
    let block = Q8_0Block::quantize(&values);
    assert!(block.quantization_error(&values) < 0.1);
}

#[test]
fn test_q8_0_block_relative_error() {
    let values: [f32; 32] = core::array::from_fn(|i| (i + 1) as f32);
    let block = Q8_0Block::quantize(&values);
    assert!(block.relative_error(&values) < 0.05);
}

// Q8KSuperBlock
#[test]
fn test_q8k_superblock_quantize() {
    let block = Q8KSuperBlock::quantize(&[1.0f32; 256]);
    assert!(block.scale > 0.0);
    assert_eq!(block.quants[0], 127);
}

#[test]
fn test_q8k_superblock_dequantize() {
    let block = Q8KSuperBlock::quantize(&[0.5f32; 256]);
    let deq = block.dequantize();
    assert!(deq.iter().all(|&v| (v - 0.5).abs() < 0.1));
}

#[test]
fn test_q8k_superblock_quantize_into() {
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(&[1.0f32; 256], &mut scale, &mut quants);
    assert!(scale > 0.0);
}

// InterleavedQ4K
#[test]
fn test_interleaved_q4k_from_q4k() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    data[2..4].copy_from_slice(&f16_bytes(0.0));
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
}

// Int8Row
#[test]
fn test_int8_row_quantize() {
    let row = Int8Row::quantize(&(0..32).map(|i| i as f32 / 10.0).collect::<Vec<_>>());
    assert_eq!(row.weights.len(), 32);
    assert!(row.scale > 0.0);
}

#[test]
fn test_int8_matvec_basic() {
    let row = Int8Row::quantize(&vec![1.0f32; 32]);
    let result = int8_matvec(&[row], &vec![1.0f32; 32]);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_int8_matvec_parallel() {
    let row = Int8Row::quantize(&vec![1.0f32; 32]);
    let result = int8_matvec_parallel(&vec![row; 8], &vec![1.0f32; 32]);
    assert_eq!(result.len(), 8);
}

// F16 conversion
#[test]
fn test_f16_to_f32_zero() { assert_eq!(f16_to_f32(0), 0.0); }

#[test]
fn test_f16_to_f32_one() {
    let one = half::f16::from_f32(1.0).to_bits();
    assert!((f16_to_f32(one) - 1.0).abs() < 1e-5);
}

#[test]
fn test_dequantize_f16_empty() { assert!(dequantize_f16(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_f16_single() {
    let bytes = f16_bytes(2.75);  // Arbitrary non-PI value
    let result = dequantize_f16(&bytes).unwrap();
    assert!((result[0] - 2.75).abs() < 0.01);
}

#[test]
fn test_dequantize_f16_odd_bytes() { assert!(dequantize_f16(&[0, 0, 0]).is_err()); }

// Q4_0
#[test]
fn test_dequantize_q4_0_empty() { assert!(dequantize_q4_0(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_q4_0_single() {
    let mut data = vec![0u8; 18];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q4_0(&data).unwrap().len(), 32);
}

#[test]
fn test_dequantize_q4_0_invalid() { assert!(dequantize_q4_0(&[0; 10]).is_err()); }

#[test]
fn test_dequantize_q4_0_simd() {
    let mut data = vec![0u8; 18];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q4_0_simd(&data).unwrap().len(), 32);
}

#[test]
fn test_dequantize_q4_0_parallel() {
    let mut data = vec![0u8; 18 * 16];
    for i in 0..16 { data[i * 18..i * 18 + 2].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(dequantize_q4_0_parallel(&data).unwrap().len(), 512);
}

// Q4_1
#[test]
fn test_dequantize_q4_1_empty() { assert!(dequantize_q4_1(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_q4_1_single() {
    let mut data = vec![0u8; 20];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q4_1(&data).unwrap().len(), 32);
}

#[test]
fn test_dequantize_q4_1_invalid() { assert!(dequantize_q4_1(&[0; 10]).is_err()); }

// Q5_0
#[test]
fn test_dequantize_q5_0_empty() { assert!(dequantize_q5_0(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_q5_0_single() {
    let mut data = vec![0u8; 22];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q5_0(&data).unwrap().len(), 32);
}

#[test]
fn test_dequantize_q5_0_invalid() { assert!(dequantize_q5_0(&[0; 10]).is_err()); }

// Q5_1
#[test]
fn test_dequantize_q5_1_empty() { assert!(dequantize_q5_1(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_q5_1_single() {
    let mut data = vec![0u8; 24];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q5_1(&data).unwrap().len(), 32);
}

#[test]
fn test_dequantize_q5_1_invalid() { assert!(dequantize_q5_1(&[0; 10]).is_err()); }

// Q8_0
#[test]
fn test_dequantize_q8_0_empty() { assert!(dequantize_q8_0(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_q8_0_single() {
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q8_0(&data).unwrap().len(), 32);
}

#[test]
fn test_dequantize_q8_0_invalid() { assert!(dequantize_q8_0(&[0; 10]).is_err()); }

#[test]
fn test_dequantize_q8_0_simd() {
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q8_0_simd(&data).unwrap().len(), 32);
}

#[test]
fn test_dequantize_q8_0_simd_optimized() {
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q8_0_simd_optimized(&data).unwrap().len(), 32);
}

#[test]
fn test_dequantize_q8_0_parallel() {
    let mut data = vec![0u8; 34 * 16];
    for i in 0..16 { data[i * 34..i * 34 + 2].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(dequantize_q8_0_parallel(&data).unwrap().len(), 512);
}

// Q4_K
#[test]
fn test_dequantize_q4_k_empty() { assert!(dequantize_q4_k(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_q4_k_single() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q4_k(&data).unwrap().len(), 256);
}

#[test]
fn test_dequantize_q4_k_invalid() { assert!(dequantize_q4_k(&[0; 100]).is_err()); }

#[test]
fn test_dequantize_q4_k_simd() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q4_k_simd(&data).unwrap().len(), 256);
}

#[test]
fn test_dequantize_q4_k_parallel() {
    let mut data = vec![0u8; 144 * 8];
    for i in 0..8 { data[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(dequantize_q4_k_parallel(&data).unwrap().len(), 2048);
}

// Q5_K
#[test]
fn test_dequantize_q5_k_empty() { assert!(dequantize_q5_k(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_q5_k_single() {
    let mut data = vec![0u8; 176];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q5_k(&data).unwrap().len(), 256);
}

#[test]
fn test_dequantize_q5_k_invalid() { assert!(dequantize_q5_k(&[0; 100]).is_err()); }

// Q6_K
#[test]
fn test_dequantize_q6_k_empty() { assert!(dequantize_q6_k(&[]).unwrap().is_empty()); }

#[test]
fn test_dequantize_q6_k_single() {
    let mut data = vec![0u8; 210];
    data[208..210].copy_from_slice(&f16_bytes(1.0));
    assert_eq!(dequantize_q6_k(&data).unwrap().len(), 256);
}

#[test]
fn test_dequantize_q6_k_invalid() { assert!(dequantize_q6_k(&[0; 100]).is_err()); }

// Quantize to Q8 blocks
#[test]
fn test_quantize_to_q8_blocks_empty() { assert!(quantize_to_q8_blocks(&[]).unwrap().is_empty()); }

#[test]
fn test_quantize_to_q8_blocks_single() {
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    assert_eq!(quantize_to_q8_blocks(&values).unwrap().len(), 1);
}

#[test]
fn test_quantize_to_q8_blocks_invalid() { assert!(quantize_to_q8_blocks(&[0.0; 33]).is_err()); }

#[test]
fn test_dequantize_q8_blocks_roundtrip() {
    let values: Vec<f32> = (0..64).map(|i| i as f32 / 10.0).collect();
    let blocks = quantize_to_q8_blocks(&values).unwrap();
    let deq = dequantize_q8_blocks(&blocks);
    assert_eq!(deq.len(), 64);
}

// Quantize activations Q8_0
#[test]
fn test_quantize_activations_q8_0() {
    let (scales, quants) = quantize_activations_q8_0(&(0..64).map(|i| i as f32 / 10.0).collect::<Vec<_>>());
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
}

// Quantize activations Q8_K
#[test]
fn test_quantize_activations_q8k_into() {
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    quantize_activations_q8k_into(&[1.0f32; 256], &mut scales, &mut quants).unwrap();
    assert!(scales[0] > 0.0);
}

#[test]
fn test_quantize_activations_q8k_into_invalid() {
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    assert!(quantize_activations_q8k_into(&[1.0f32; 100], &mut scales, &mut quants).is_err());
}

// Fused Q4_K dot
#[test]
fn test_fused_q4k_dot() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q4k_dot(&data, &[1.0f32; 256]).unwrap().is_finite());
}

#[test]
fn test_fused_q4k_dot_simd() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q4k_dot_simd(&data, &[1.0f32; 256]).unwrap().is_finite());
}

// Fused Q4_K Q8_0 dot
#[test]
fn test_fused_q4k_q8_dot() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    let blocks: Vec<Q8_0Block> = (0..8).map(|_| Q8_0Block::quantize(&[1.0f32; 32])).collect();
    assert!(fused_q4k_q8_dot(&data, &blocks).unwrap().is_finite());
}

#[test]
fn test_fused_q4k_q8_dot_simd() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    let blocks: Vec<Q8_0Block> = (0..8).map(|_| Q8_0Block::quantize(&[1.0f32; 32])).collect();
    assert!(fused_q4k_q8_dot_simd(&data, &blocks).unwrap().is_finite());
}

// Fused Q4_K Q8_K dot
#[test]
fn test_fused_q4k_q8k_dot() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q4k_q8k_dot(&data, &[1.0f32; 1], &[1i8; 256]).unwrap().is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_simd() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q4k_q8k_dot_simd(&data, &[1.0f32; 1], &[1i8; 256]).unwrap().is_finite());
}

// Fused Q5_K dot
#[test]
fn test_fused_q5k_dot() {
    let mut data = vec![0u8; 176];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q5k_dot(&data, &[1.0f32; 256]).unwrap().is_finite());
}

#[test]
fn test_fused_q5k_dot_simd() {
    let mut data = vec![0u8; 176];
    data[0..2].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q5k_dot_simd(&data, &[1.0f32; 256]).unwrap().is_finite());
}

// Fused Q6_K dot
#[test]
fn test_fused_q6k_dot() {
    let mut data = vec![0u8; 210];
    data[208..210].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q6k_dot(&data, &[1.0f32; 256]).unwrap().is_finite());
}

#[test]
fn test_fused_q6k_dot_simd() {
    let mut data = vec![0u8; 210];
    data[208..210].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q6k_dot_simd(&data, &[1.0f32; 256]).unwrap().is_finite());
}

// Fused Q6_K Q8_K dot
#[test]
fn test_fused_q6k_q8k_dot_simd() {
    let mut data = vec![0u8; 210];
    data[208..210].copy_from_slice(&f16_bytes(1.0));
    assert!(fused_q6k_q8k_dot_simd(&data, &[1.0f32; 1], &[1i8; 256]).unwrap().is_finite());
}

// Tiled matvec
#[test]
fn test_fused_q4k_tiled_matvec() {
    let mut weights = vec![0u8; 144 * 2];
    for i in 0..2 { weights[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(fused_q4k_tiled_matvec(&weights, &[1.0f32; 256], 256, 2, None).unwrap().len(), 2);
}

#[test]
fn test_fused_q5k_tiled_matvec() {
    let mut weights = vec![0u8; 176 * 2];
    for i in 0..2 { weights[i * 176..i * 176 + 2].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(fused_q5k_tiled_matvec(&weights, &[1.0f32; 256], 256, 2, None).unwrap().len(), 2);
}

#[test]
fn test_fused_q6k_tiled_matvec() {
    let mut weights = vec![0u8; 210 * 2];
    for i in 0..2 { weights[i * 210 + 208..i * 210 + 210].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(fused_q6k_tiled_matvec(&weights, &[1.0f32; 256], 256, 2, None).unwrap().len(), 2);
}

// Parallel matvec
#[test]
fn test_fused_q4k_parallel_matvec() {
    let mut weights = vec![0u8; 144 * 8];
    for i in 0..8 { weights[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(fused_q4k_parallel_matvec(&weights, &[1.0f32; 256], 256, 8).unwrap().len(), 8);
}

#[test]
fn test_fused_q4k_parallel_matvec_into() {
    let mut weights = vec![0u8; 144 * 8];
    for i in 0..8 { weights[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0)); }
    let mut output = vec![0.0f32; 8];
    fused_q4k_parallel_matvec_into(&weights, &[1.0f32; 256], 256, 8, &mut output).unwrap();
    assert_eq!(output.len(), 8);
}

#[test]
fn test_fused_q5k_parallel_matvec() {
    let mut weights = vec![0u8; 176 * 8];
    for i in 0..8 { weights[i * 176..i * 176 + 2].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(fused_q5k_parallel_matvec(&weights, &[1.0f32; 256], 256, 8).unwrap().len(), 8);
}

#[test]
fn test_fused_q5k_parallel_matvec_into() {
    let mut weights = vec![0u8; 176 * 8];
    for i in 0..8 { weights[i * 176..i * 176 + 2].copy_from_slice(&f16_bytes(1.0)); }
    let mut output = vec![0.0f32; 8];
    fused_q5k_parallel_matvec_into(&weights, &[1.0f32; 256], 256, 8, &mut output).unwrap();
}

#[test]
fn test_fused_q6k_parallel_matvec() {
    let mut weights = vec![0u8; 210 * 8];
    for i in 0..8 { weights[i * 210 + 208..i * 210 + 210].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(fused_q6k_parallel_matvec(&weights, &[1.0f32; 256], 256, 8).unwrap().len(), 8);
}

#[test]
fn test_fused_q6k_parallel_matvec_into() {
    let mut weights = vec![0u8; 210 * 8];
    for i in 0..8 { weights[i * 210 + 208..i * 210 + 210].copy_from_slice(&f16_bytes(1.0)); }
    let mut output = vec![0.0f32; 8];
    fused_q6k_parallel_matvec_into(&weights, &[1.0f32; 256], 256, 8, &mut output).unwrap();
}

// Q4_K Q8_K parallel matvec
#[test]
fn test_fused_q4k_q8k_parallel_matvec_into() {
    let mut weights = vec![0u8; 144 * 8];
    for i in 0..8 { weights[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0)); }
    let mut output = vec![0.0f32; 8];
    fused_q4k_q8k_parallel_matvec_into(&weights, &[1.0f32; 1], &[1i8; 256], 256, 8, &mut output).unwrap();
}

#[test]
fn test_fused_q6k_q8k_parallel_matvec_into() {
    let mut weights = vec![0u8; 210 * 8];
    for i in 0..8 { weights[i * 210 + 208..i * 210 + 210].copy_from_slice(&f16_bytes(1.0)); }
    let mut output = vec![0.0f32; 8];
    fused_q6k_q8k_parallel_matvec_into(&weights, &[1.0f32; 1], &[1i8; 256], 256, 8, &mut output).unwrap();
}

// Q4_K FFN up gate
#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into() {
    let mut up = vec![0u8; 144 * 8];
    let mut gate = vec![0u8; 144 * 8];
    for i in 0..8 {
        up[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0));
        gate[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0));
    }
    let mut up_out = vec![0.0f32; 8];
    let mut gate_out = vec![0.0f32; 8];
    fused_q4k_q8k_ffn_up_gate_into(&up, &gate, &[1.0f32; 1], &[1i8; 256], 256, 8, &mut up_out, &mut gate_out).unwrap();
}

// Q4_K auto matvec
#[test]
fn test_fused_q4k_auto_matvec_into() {
    let mut weights = vec![0u8; 144 * 8];
    for i in 0..8 { weights[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0)); }
    let mut output = vec![0.0f32; 8];
    fused_q4k_auto_matvec_into(&weights, &[1.0f32; 256], 256, 8, &mut output).unwrap();
}

// Q6_K colmajor matvec
// Column-major: in_dim columns, each column has one Q6_K super-block (210 bytes, 256 values)
// out_dim is fixed at 256 (Q6_K super-block size)
#[test]
fn test_fused_q6k_colmajor_matvec() {
    let in_dim = 4;  // 4 input columns
    let out_dim = 256;  // Q6_K super-block size
    let mut weights = vec![0u8; 210 * in_dim];  // in_dim super-blocks
    for i in 0..in_dim { weights[i * 210 + 208..i * 210 + 210].copy_from_slice(&f16_bytes(1.0)); }
    assert_eq!(fused_q6k_colmajor_matvec(&weights, &[1.0f32; 4], in_dim, out_dim).unwrap().len(), out_dim);
}

// Q4_0 Q8_0 parallel matvec
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec() {
    let bpr = 18 * 8;  // 8 blocks per row (256 / 32 = 8), 18 bytes per block
    let mut weights = vec![0u8; bpr * 8];  // 8 output rows
    for r in 0..8 { for b in 0..8 { weights[r * bpr + b * 18..r * bpr + b * 18 + 2].copy_from_slice(&f16_bytes(1.0)); } }
    assert_eq!(fused_q4_0_q8_0_parallel_matvec(&weights, &[1.0f32; 256], 256, 8).unwrap().len(), 8);
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into() {
    let bpr = 18 * 8;
    let mut weights = vec![0u8; bpr * 8];
    for r in 0..8 { for b in 0..8 { weights[r * bpr + b * 18..r * bpr + b * 18 + 2].copy_from_slice(&f16_bytes(1.0)); } }
    let mut output = vec![0.0f32; 8];
    fused_q4_0_q8_0_parallel_matvec_into(&weights, &[1.0f32; 256], 256, &mut output).unwrap();
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_prequant() {
    let bpr = 18 * 8;  // 8 blocks per row (256 / 32 = 8), 18 bytes per block
    let mut weights = vec![0u8; bpr * 8];  // 8 output rows
    for r in 0..8 { for b in 0..8 { weights[r * bpr + b * 18..r * bpr + b * 18 + 2].copy_from_slice(&f16_bytes(1.0)); } }
    assert_eq!(fused_q4_0_q8_0_parallel_matvec_prequant(&weights, &[1.0f32; 8], &[1i8; 256], 256, 8).unwrap().len(), 8);
}

// Q8_0 Q8_0 parallel matvec
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec() {
    let bpr = 34 * 8;  // 8 blocks per row (256 / 32 = 8), 34 bytes per Q8_0 block
    let mut weights = vec![0u8; bpr * 8];  // 8 output rows
    for r in 0..8 { for b in 0..8 { weights[r * bpr + b * 34..r * bpr + b * 34 + 2].copy_from_slice(&f16_bytes(1.0)); } }
    assert_eq!(fused_q8_0_q8_0_parallel_matvec(&weights, &[1.0f32; 256], 256, 8).unwrap().len(), 8);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into() {
    let bpr = 34 * 8;
    let mut weights = vec![0u8; bpr * 8];
    for r in 0..8 { for b in 0..8 { weights[r * bpr + b * 34..r * bpr + b * 34 + 2].copy_from_slice(&f16_bytes(1.0)); } }
    let mut output = vec![0.0f32; 8];
    fused_q8_0_q8_0_parallel_matvec_into(&weights, &[1.0f32; 256], 256, 8, &mut output).unwrap();
}

// RMSNorm quantization
#[test]
fn test_quantize_rmsnorm_q8_0() {
    let (scales, quants) = quantize_rmsnorm_q8_0(&[1.0f32; 64], &[1.0f32; 64], 1e-5);
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into() {
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];
    quantize_rmsnorm_q8_0_into(&[1.0f32; 64], &[1.0f32; 64], 1e-5, &mut scales, &mut quants);
}

// Fused RMSNorm
#[test]
fn test_fused_rmsnorm_q4_0_matmul() {
    let bpr = 18 * 8;  // 8 blocks per row (256 / 32 = 8), 18 bytes per block
    let mut weights = vec![0u8; bpr * 8];  // 8 output rows
    for r in 0..8 { for b in 0..8 { weights[r * bpr + b * 18..r * bpr + b * 18 + 2].copy_from_slice(&f16_bytes(1.0)); } }
    assert_eq!(fused_rmsnorm_q4_0_matmul(&[1.0f32; 256], &[1.0f32; 256], 1e-5, &weights, 256, 8).unwrap().len(), 8);
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate() {
    let bpr = 18 * 8;
    let mut up = vec![0u8; bpr * 8];
    let mut gate = vec![0u8; bpr * 8];
    for r in 0..8 { for b in 0..8 {
        up[r * bpr + b * 18..r * bpr + b * 18 + 2].copy_from_slice(&f16_bytes(1.0));
        gate[r * bpr + b * 18..r * bpr + b * 18 + 2].copy_from_slice(&f16_bytes(1.0));
    } }
    let result = fused_rmsnorm_ffn_up_gate(&[1.0f32; 256], &[1.0f32; 256], 1e-5, &up, &gate, 256, 8).unwrap();
    assert_eq!(result.0.len(), 8);
    assert_eq!(result.1.len(), 8);
}

// SwiGLU and Softmax
#[test]
fn test_fused_swiglu_simd() {
    let mut gate = vec![1.0f32; 64];
    fused_swiglu_simd(&mut gate, &[1.0f32; 64]);
    assert!(gate.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_softmax_simd() {
    let mut x = vec![1.0f32; 64];
    softmax_simd(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_single() {
    let mut x = vec![5.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-5);
}

// RoPE rotation
#[test]
fn test_apply_rope_rotation_simd() {
    let mut q = vec![1.0f32; 64];
    let mut k = vec![1.0f32; 64];
    apply_rope_rotation_simd(&mut q, &mut k, &[1.0f32; 64], &[0.0f32; 64]);
    assert!(q.iter().all(|&v| (v - 1.0).abs() < 1e-5));
}

// Consistency tests
#[test]
fn test_q4_0_scalar_vs_simd() {
    let mut data = vec![0u8; 18 * 4];
    for i in 0..4 {
        data[i * 18..i * 18 + 2].copy_from_slice(&f16_bytes(1.0 + i as f32));
        for j in 0..16 { data[i * 18 + 2 + j] = ((i * 16 + j) % 256) as u8; }
    }
    let scalar = dequantize_q4_0(&data).unwrap();
    let simd = dequantize_q4_0_simd(&data).unwrap();
    assert_eq!(scalar.len(), simd.len());
    for (s, si) in scalar.iter().zip(simd.iter()) { assert!((s - si).abs() < 1e-5); }
}

#[test]
fn test_q8_0_scalar_vs_simd() {
    let mut data = vec![0u8; 34 * 4];
    for i in 0..4 {
        data[i * 34..i * 34 + 2].copy_from_slice(&f16_bytes(1.0 + i as f32));
        for j in 0..32 { data[i * 34 + 2 + j] = ((i * 32 + j) % 256) as u8; }
    }
    let scalar = dequantize_q8_0(&data).unwrap();
    let simd = dequantize_q8_0_simd(&data).unwrap();
    assert_eq!(scalar.len(), simd.len());
}

#[test]
fn test_q4_k_scalar_vs_simd() {
    let mut data = vec![0u8; 144 * 2];
    for i in 0..2 {
        data[i * 144..i * 144 + 2].copy_from_slice(&f16_bytes(1.0 + i as f32));
        data[i * 144 + 2..i * 144 + 4].copy_from_slice(&f16_bytes(0.5));
    }
    let scalar = dequantize_q4_k(&data).unwrap();
    let simd = dequantize_q4_k_simd(&data).unwrap();
    assert_eq!(scalar.len(), simd.len());
}
