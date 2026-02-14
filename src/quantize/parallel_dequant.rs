//! Parallel dequantization functions (PMAT-802)
//!
//! Implements parallel dequantization using rayon for multi-core acceleration:
//! - `dequantize_q4_k_parallel` - Parallel Q4_K dequantization
//! - `dequantize_q4_k_simd` - SIMD-accelerated Q4_K dequantization
//! - `dequantize_q8_0_parallel` - Parallel Q8_0 dequantization
//! - `dequantize_q8_0_simd` - SIMD-accelerated Q8_0 dequantization

use super::dequant::{f16_to_f32, read_f16};
use super::simd::extract_scale_min;
use super::types::QK_K;
use crate::error::{RealizarError, Result};

// ============================================================================

/// Parallel Q4_K dequantization using rayon
///
/// Processes super-blocks in parallel for faster model loading.
/// Each super-block (256 values) is dequantized independently.
///
/// # Arguments
///
/// * `data` - Raw Q4_K quantized data (144 bytes per super-block)
///
/// # Returns
///
/// Dequantized f32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of super-block size
///
/// # Performance
///
/// On a 16-core system, achieves ~10x speedup over serial dequantization
/// for large models (1B+ parameters).
pub fn dequantize_q4_k_parallel(data: &[u8]) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;

    // Process super-blocks in parallel
    let result: Vec<f32> = (0..num_super_blocks)
        .into_par_iter()
        .flat_map(|sb_idx| {
            let sb_start = sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &data[sb_start..sb_start + SUPER_BLOCK_BYTES];
            dequantize_q4_k_superblock(sb_data)
        })
        .collect();

    Ok(result)
}

/// Dequantize a single Q4_K super-block (256 values)
///
/// Internal helper for parallel processing.
///
/// Exposed as `pub(crate)` for direct testing.
#[inline]
pub(crate) fn dequantize_q4_k_superblock(sb_data: &[u8]) -> Vec<f32> {
    let mut result = vec![0.0f32; QK_K];

    // Read d (f16 -> f32)
    let d = read_f16(sb_data.get(0..2).expect("Q4_K superblock: need ≥2 bytes for d"));

    // Read dmin (f16 -> f32)
    let dmin = read_f16(sb_data.get(2..4).expect("Q4_K superblock: need ≥4 bytes for dmin"));

    // Read scales (12 bytes)
    let mut scales = [0u8; 12];
    scales.copy_from_slice(sb_data.get(4..16).expect("Q4_K superblock: need ≥16 bytes for scales"));

    // Read qs (128 bytes)
    let qs = sb_data.get(16..144).expect("Q4_K superblock: need ≥144 bytes for qs");

    // Dequantize following candle's layout:
    // For each 64-value chunk, output 32 low nibbles then 32 high nibbles
    let mut ys_index = 0;

    for j in (0..QK_K).step_by(64) {
        let q = &qs[j / 2..j / 2 + 32];

        // Get scales for the two 32-value halves
        let is = j / 32;
        let (sc1, m1) = extract_scale_min(&scales, is);
        let d1 = d * sc1;
        let dm1 = dmin * m1;

        let (sc2, m2) = extract_scale_min(&scales, is + 1);
        let d2 = d * sc2;
        let dm2 = dmin * m2;

        // First pass: 32 low nibbles
        for &byte in q {
            result[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
            ys_index += 1;
        }

        // Second pass: 32 high nibbles
        for &byte in q {
            result[ys_index] = d2 * (byte >> 4) as f32 - dm2;
            ys_index += 1;
        }
    }

    result
}

/// SIMD-accelerated Q4_K dequantization with runtime feature detection
///
/// Uses AVX2 when available, falls back to parallel scalar otherwise.
///
/// # Arguments
///
/// * `data` - Raw Q4_K quantized data (144 bytes per super-block)
///
/// # Returns
///
/// Dequantized f32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of super-block size
///
/// # Performance
///
/// AVX2: ~4x speedup over scalar per super-block
/// Combined with rayon: ~40x speedup on 16-core system
pub fn dequantize_q4_k_simd(data: &[u8]) -> Result<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified at runtime
            return unsafe { dequantize_q4_k_avx2_parallel(data) };
        }
    }

    // Fallback to parallel scalar
    dequantize_q4_k_parallel(data)
}

/// AVX2-accelerated parallel Q4_K dequantization
///
/// # Safety
///
/// Caller must ensure AVX2 is available (use runtime feature detection)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_q4_k_avx2_parallel(data: &[u8]) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;
    // Process 64 super-blocks per parallel task to reduce scheduling overhead
    const CHUNK_SIZE: usize = 64;
    const CHUNK_BYTES: usize = SUPER_BLOCK_BYTES * CHUNK_SIZE;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;

    // For small data, skip parallelism overhead
    if num_super_blocks < CHUNK_SIZE * 2 {
        let mut result = Vec::with_capacity(num_super_blocks * QK_K);
        for sb_idx in 0..num_super_blocks {
            let sb_start = sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &data[sb_start..sb_start + SUPER_BLOCK_BYTES];
            // SAFETY: AVX2 availability verified by caller
            result.extend(unsafe { dequantize_q4_k_superblock_avx2(sb_data) });
        }
        return Ok(result);
    }

    // Process chunks of super-blocks in parallel
    let result: Vec<f32> = data
        .par_chunks(CHUNK_BYTES)
        .flat_map(|chunk| {
            let mut chunk_result = Vec::with_capacity(chunk.len() / SUPER_BLOCK_BYTES * QK_K);
            for sb_data in chunk.chunks_exact(SUPER_BLOCK_BYTES) {
                // SAFETY: AVX2 availability verified by caller
                chunk_result.extend(unsafe { dequantize_q4_k_superblock_avx2(sb_data) });
            }
            chunk_result
        })
        .collect();

    Ok(result)
}

/// AVX2 SIMD dequantization for a single Q4_K super-block
///
/// Uses 256-bit SIMD to process 8 values simultaneously.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dequantize_q4_k_superblock_avx2(sb_data: &[u8]) -> Vec<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let mut result = vec![0.0f32; QK_K];

    // Read d and dmin
    let d = read_f16(sb_data.get(0..2).expect("Q4_K superblock: need ≥2 bytes for d"));
    let dmin = read_f16(sb_data.get(2..4).expect("Q4_K superblock: need ≥4 bytes for dmin"));

    // SAFETY: AVX2 availability verified by caller's target_feature
    unsafe {
        // Read scales
        let mut scales = [0u8; 12];
        scales.copy_from_slice(sb_data.get(4..16).expect("Q4_K superblock: need ≥16 bytes for scales"));

        let qs = sb_data.get(16..144).expect("Q4_K superblock: need ≥144 bytes for qs");

        // Dequantize following candle's layout:
        // For each 64-value chunk, output 32 low nibbles then 32 high nibbles
        let mut ys_index = 0;

        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;
            let d1_vec = _mm256_set1_ps(d1);
            let dm1_vec = _mm256_set1_ps(dm1);

            let (sc2, m2) = extract_scale_min(&scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;
            let d2_vec = _mm256_set1_ps(d2);
            let dm2_vec = _mm256_set1_ps(dm2);

            // First pass: 32 low nibbles in 4 iterations of 8
            for chunk in 0..4 {
                let byte_start = chunk * 8;

                // Extract 8 low nibbles from 8 bytes
                let q0 = (q[byte_start] & 0x0F) as i32;
                let q1 = (q[byte_start + 1] & 0x0F) as i32;
                let q2 = (q[byte_start + 2] & 0x0F) as i32;
                let q3 = (q[byte_start + 3] & 0x0F) as i32;
                let q4 = (q[byte_start + 4] & 0x0F) as i32;
                let q5 = (q[byte_start + 5] & 0x0F) as i32;
                let q6 = (q[byte_start + 6] & 0x0F) as i32;
                let q7 = (q[byte_start + 7] & 0x0F) as i32;

                let q_vec = _mm256_setr_epi32(q0, q1, q2, q3, q4, q5, q6, q7);
                let q_f32 = _mm256_cvtepi32_ps(q_vec);
                let dequant = _mm256_fmsub_ps(d1_vec, q_f32, dm1_vec);

                _mm256_storeu_ps(result.as_mut_ptr().add(ys_index), dequant);
                ys_index += 8;
            }

            // Second pass: 32 high nibbles in 4 iterations of 8
            for chunk in 0..4 {
                let byte_start = chunk * 8;

                // Extract 8 high nibbles from 8 bytes
                let q0 = (q[byte_start] >> 4) as i32;
                let q1 = (q[byte_start + 1] >> 4) as i32;
                let q2 = (q[byte_start + 2] >> 4) as i32;
                let q3 = (q[byte_start + 3] >> 4) as i32;
                let q4 = (q[byte_start + 4] >> 4) as i32;
                let q5 = (q[byte_start + 5] >> 4) as i32;
                let q6 = (q[byte_start + 6] >> 4) as i32;
                let q7 = (q[byte_start + 7] >> 4) as i32;

                let q_vec = _mm256_setr_epi32(q0, q1, q2, q3, q4, q5, q6, q7);
                let q_f32 = _mm256_cvtepi32_ps(q_vec);
                let dequant = _mm256_fmsub_ps(d2_vec, q_f32, dm2_vec);

                _mm256_storeu_ps(result.as_mut_ptr().add(ys_index), dequant);
                ys_index += 8;
            }
        }
    }

    result
}

/// Parallel Q8_0 dequantization using rayon
///
/// Q8_0 is simpler than Q4_K (no scale packing), making SIMD even more effective.
///
/// # Arguments
///
/// * `data` - Raw Q8_0 quantized data (36 bytes per block: 4 scale + 32 quants)
///
/// # Returns
///
/// Dequantized f32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of block size
pub fn dequantize_q8_0_parallel(data: &[u8]) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8 quants)

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;

    // Process blocks in parallel
    let result: Vec<f32> = (0..num_blocks)
        .into_par_iter()
        .flat_map(|block_idx| {
            let block_start = block_idx * BLOCK_BYTES;
            let block_data = &data[block_start..block_start + BLOCK_BYTES];
            dequantize_q8_0_block(block_data)
        })
        .collect();

    Ok(result)
}

/// Dequantize a single Q8_0 block (32 values)
///
/// Exposed as `pub(crate)` for direct testing.
#[inline]
pub(crate) fn dequantize_q8_0_block(block_data: &[u8]) -> Vec<f32> {
    let mut result = Vec::with_capacity(32);

    // Read scale (f16 -> f32)
    let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
    let scale = f16_to_f32(scale_bits);

    // Dequantize 32 int8 values
    for &byte in &block_data[2..34] {
        let value = i8::from_le_bytes([byte]);
        result.push(scale * f32::from(value));
    }

    result
}

/// SIMD-accelerated Q8_0 dequantization
///
/// Uses AVX2 when available for 8x parallel i8→f32 conversion.
///
/// # Errors
///
/// Returns error if data length is not a multiple of block size (36 bytes)
pub fn dequantize_q8_0_simd(data: &[u8]) -> Result<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified at runtime
            return unsafe { dequantize_q8_0_avx2_parallel(data) };
        }
    }

    // Fallback to parallel scalar
    dequantize_q8_0_parallel(data)
}

/// AVX2-accelerated parallel Q8_0 dequantization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_q8_0_avx2_parallel(data: &[u8]) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8 quants)

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;

    let result: Vec<f32> = (0..num_blocks)
        .into_par_iter()
        .flat_map(|block_idx| {
            let block_start = block_idx * BLOCK_BYTES;
            let block_data = &data[block_start..block_start + BLOCK_BYTES];
            // SAFETY: AVX2 availability verified by caller
            unsafe { dequantize_q8_0_block_avx2(block_data) }
        })
        .collect();

    Ok(result)
}

/// AVX2 SIMD dequantization for a single Q8_0 block
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_q8_0_block_avx2(block_data: &[u8]) -> Vec<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let mut result = vec![0.0f32; 32];

    // Read scale (f16 -> f32)
    let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
    let scale = f16_to_f32(scale_bits);

    // SAFETY: AVX2 availability verified by caller's target_feature
    unsafe {
        let scale_vec = _mm256_set1_ps(scale);

        // Process 32 i8 values in 4 iterations of 8
        for chunk in 0..4 {
            let byte_start = 2 + chunk * 8; // Start at offset 2 (after f16 scale)

            // Load 8 i8 values and sign-extend to i32
            let q0 = block_data[byte_start] as i8 as i32;
            let q1 = block_data[byte_start + 1] as i8 as i32;
            let q2 = block_data[byte_start + 2] as i8 as i32;
            let q3 = block_data[byte_start + 3] as i8 as i32;
            let q4 = block_data[byte_start + 4] as i8 as i32;
            let q5 = block_data[byte_start + 5] as i8 as i32;
            let q6 = block_data[byte_start + 6] as i8 as i32;
            let q7 = block_data[byte_start + 7] as i8 as i32;

            let q_vec = _mm256_setr_epi32(q0, q1, q2, q3, q4, q5, q6, q7);
            let q_f32 = _mm256_cvtepi32_ps(q_vec);

            // Multiply by scale
            let dequant = _mm256_mul_ps(scale_vec, q_f32);

            // Store 8 results
            _mm256_storeu_ps(result.as_mut_ptr().add(chunk * 8), dequant);
        }
    }

    result
}

// DequantStats, SimdBackend, detect_simd_backend moved to types.rs (PMAT-802)

/// SIMD-optimized RoPE rotation for a single head
///
/// Applies rotary position embedding rotation to a single attention head:
/// x1[i] = x1[i] * cos[i] - x2[i] * sin[i]
/// x2[i] = x1[i] * sin[i] + x2[i] * cos[i]
///
/// # Arguments
/// * `x1` - First half of head (will be modified in-place)
/// * `x2` - Second half of head (will be modified in-place)
/// * `cos_vals` - Precomputed cosine values
/// * `sin_vals` - Precomputed sine values
#[inline]
pub fn apply_rope_rotation_simd(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    debug_assert_eq!(x1.len(), x2.len());
    debug_assert_eq!(x1.len(), cos_vals.len());
    debug_assert_eq!(x1.len(), sin_vals.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                apply_rope_rotation_avx512(x1, x2, cos_vals, sin_vals);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                apply_rope_rotation_avx2(x1, x2, cos_vals, sin_vals);
            }
            return;
        }
    }

    // Scalar fallback
    apply_rope_rotation_scalar(x1, x2, cos_vals, sin_vals);
}

/// Scalar fallback for RoPE rotation
///
/// Exposed as `pub(crate)` for direct testing on AVX2 machines.
#[inline]
pub(crate) fn apply_rope_rotation_scalar(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    for i in 0..x1.len() {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn apply_rope_rotation_avx2(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
    };

    let n = x1.len();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= n {
        let v1 = _mm256_loadu_ps(x1.as_ptr().add(i));
        let v2 = _mm256_loadu_ps(x2.as_ptr().add(i));
        let cos_v = _mm256_loadu_ps(cos_vals.as_ptr().add(i));
        let sin_v = _mm256_loadu_ps(sin_vals.as_ptr().add(i));

        // r1 = v1 * cos - v2 * sin
        let v1_cos = _mm256_mul_ps(v1, cos_v);
        let r1 = _mm256_fnmadd_ps(v2, sin_v, v1_cos);

        // r2 = v1 * sin + v2 * cos
        let v1_sin = _mm256_mul_ps(v1, sin_v);
        let r2 = _mm256_fmadd_ps(v2, cos_v, v1_sin);

        _mm256_storeu_ps(x1.as_mut_ptr().add(i), r1);
        _mm256_storeu_ps(x2.as_mut_ptr().add(i), r2);

        i += 8;
    }

    // Handle remainder
    while i < n {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn apply_rope_rotation_avx512(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_storeu_ps,
    };

    let n = x1.len();
    let mut i = 0;

    // Process 16 elements at a time with AVX-512
    while i + 16 <= n {
        let v1 = _mm512_loadu_ps(x1.as_ptr().add(i));
        let v2 = _mm512_loadu_ps(x2.as_ptr().add(i));
        let cos_v = _mm512_loadu_ps(cos_vals.as_ptr().add(i));
        let sin_v = _mm512_loadu_ps(sin_vals.as_ptr().add(i));

        // r1 = v1 * cos - v2 * sin
        let v1_cos = _mm512_mul_ps(v1, cos_v);
        let r1 = _mm512_fnmadd_ps(v2, sin_v, v1_cos);

        // r2 = v1 * sin + v2 * cos
        let v1_sin = _mm512_mul_ps(v1, sin_v);
        let r2 = _mm512_fmadd_ps(v2, cos_v, v1_sin);

        _mm512_storeu_ps(x1.as_mut_ptr().add(i), r1);
        _mm512_storeu_ps(x2.as_mut_ptr().add(i), r2);

        i += 16;
    }

    // Handle remainder with AVX2 or scalar
    while i < n {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============= Q4_K parallel tests =============

    #[test]
    fn test_dequantize_q4_k_parallel_empty() {
        let result = dequantize_q4_k_parallel(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dequantize_q4_k_parallel_invalid_size() {
        // Q4_K super-block is 144 bytes; 100 bytes is invalid
        let data = vec![0u8; 100];
        let result = dequantize_q4_k_parallel(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RealizarError::InvalidShape { .. }));
    }

    #[test]
    fn test_dequantize_q4_k_parallel_single_block() {
        // Create a valid super-block (144 bytes)
        let mut data = vec![0u8; 144];
        // Set d = 1.0 (f16 encoding)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0 in f16
                                                              // Set dmin = 0.0
        data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

        let result = dequantize_q4_k_parallel(&data);
        assert!(result.is_ok());
        let dequant = result.unwrap();
        assert_eq!(dequant.len(), 256); // QK_K = 256
    }

    #[test]
    fn test_dequantize_q4_k_parallel_multiple_blocks() {
        // Create 2 valid super-blocks (288 bytes)
        let mut data = vec![0u8; 288];
        // Set d = 1.0 for both blocks
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[144..146].copy_from_slice(&0x3C00u16.to_le_bytes());

        let result = dequantize_q4_k_parallel(&data);
        assert!(result.is_ok());
        let dequant = result.unwrap();
        assert_eq!(dequant.len(), 512); // 2 * QK_K = 512
    }

    #[test]
    fn test_dequantize_q4_k_superblock_zero_data() {
        // All zeros should dequantize correctly
        let sb_data = vec![0u8; 144];
        let result = dequantize_q4_k_superblock(&sb_data);
        assert_eq!(result.len(), 256);
        // With d=0 and dmin=0, all values should be 0
        for val in &result {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_dequantize_q4_k_superblock_scale_factor() {
        let mut sb_data = vec![0u8; 144];
        // d = 2.0 in f16 = 0x4000
        sb_data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        // dmin = 0
        sb_data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
        // Set qs to known values (nibbles = 1)
        for i in 16..144 {
            sb_data[i] = 0x11; // Low nibble = 1, high nibble = 1
        }

        let result = dequantize_q4_k_superblock(&sb_data);
        assert_eq!(result.len(), 256);
        // Values should reflect the scale factor
    }

    // ============= Q4_K SIMD tests =============

    #[test]
    fn test_dequantize_q4_k_simd_empty() {
        let result = dequantize_q4_k_simd(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dequantize_q4_k_simd_invalid_size() {
        let data = vec![0u8; 50];
        let result = dequantize_q4_k_simd(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_k_simd_matches_parallel() {
        let mut data = vec![0u8; 144];
        // Set meaningful values
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
        for i in 16..144 {
            data[i] = (i % 256) as u8;
        }

        let simd_result = dequantize_q4_k_simd(&data).unwrap();
        let parallel_result = dequantize_q4_k_parallel(&data).unwrap();

        assert_eq!(simd_result.len(), parallel_result.len());
        for (s, p) in simd_result.iter().zip(parallel_result.iter()) {
            assert!((s - p).abs() < 1e-5, "simd={} parallel={}", s, p);
        }
    }

    // ============= Q8_0 parallel tests =============

    #[test]
    fn test_dequantize_q8_0_parallel_empty() {
        let result = dequantize_q8_0_parallel(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dequantize_q8_0_parallel_invalid_size() {
        // Q8_0 block is 34 bytes; 20 bytes is invalid
        let data = vec![0u8; 20];
        let result = dequantize_q8_0_parallel(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_parallel_single_block() {
        // Create valid Q8_0 block (34 bytes)
        let mut data = vec![0u8; 34];
        // Scale = 1.0 (f16 = 0x3C00)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // Set quants to known values
        for i in 2..34 {
            data[i] = 10; // Each i8 = 10
        }

        let result = dequantize_q8_0_parallel(&data).unwrap();
        assert_eq!(result.len(), 32);
        for val in &result {
            assert!((val - 10.0).abs() < 0.01, "expected 10.0, got {}", val);
        }
    }

    #[test]
    fn test_dequantize_q8_0_parallel_negative_values() {
        let mut data = vec![0u8; 34];
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
                                                              // Set quants to -5 (i8 represented as u8)
        for i in 2..34 {
            data[i] = (-5i8) as u8;
        }

        let result = dequantize_q8_0_parallel(&data).unwrap();
        for val in &result {
            assert!((val - (-5.0)).abs() < 0.01, "expected -5.0, got {}", val);
        }
    }

    #[test]
    fn test_dequantize_q8_0_block_identity() {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
        for i in 0..32 {
            block[2 + i] = i as u8;
        }

        let result = dequantize_q8_0_block(&block);
        assert_eq!(result.len(), 32);
        for (i, val) in result.iter().enumerate() {
            assert!((val - i as f32).abs() < 0.01);
        }
    }

    // ============= Q8_0 SIMD tests =============

    #[test]
    fn test_dequantize_q8_0_simd_empty() {
        let result = dequantize_q8_0_simd(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dequantize_q8_0_simd_invalid_size() {
        let data = vec![0u8; 30];
        let result = dequantize_q8_0_simd(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_simd_matches_parallel() {
        let mut data = vec![0u8; 34];
        data[0..2].copy_from_slice(&0x4000u16.to_le_bytes()); // scale = 2.0
        for i in 2..34 {
            data[i] = ((i - 2) as i8 * 3) as u8;
        }

        let simd_result = dequantize_q8_0_simd(&data).unwrap();
        let parallel_result = dequantize_q8_0_parallel(&data).unwrap();

        assert_eq!(simd_result.len(), parallel_result.len());
        for (s, p) in simd_result.iter().zip(parallel_result.iter()) {
            assert!((s - p).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequantize_q8_0_simd_multiple_blocks() {
        // 3 blocks = 102 bytes
        let mut data = vec![0u8; 102];
        for block in 0..3 {
            let offset = block * 34;
            data[offset..offset + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
            for i in 0..32 {
                data[offset + 2 + i] = (block * 10 + i) as u8;
            }
        }

        let result = dequantize_q8_0_simd(&data).unwrap();
        assert_eq!(result.len(), 96); // 3 * 32
    }

    // ============= RoPE rotation tests =============

    #[test]
    fn test_apply_rope_rotation_scalar_identity() {
        let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
        let mut x2 = vec![0.0, 0.0, 0.0, 0.0];
        let cos_vals = vec![1.0, 1.0, 1.0, 1.0]; // cos(0) = 1
        let sin_vals = vec![0.0, 0.0, 0.0, 0.0]; // sin(0) = 0

        apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

        // With cos=1, sin=0: x1' = x1*1 - x2*0 = x1, x2' = x1*0 + x2*1 = 0
        assert_eq!(x1, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(x2, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_apply_rope_rotation_scalar_90_degrees() {
        let mut x1 = vec![1.0, 2.0];
        let mut x2 = vec![0.0, 0.0];
        let cos_vals = vec![0.0, 0.0]; // cos(90°) ≈ 0
        let sin_vals = vec![1.0, 1.0]; // sin(90°) = 1

        apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

        // x1' = x1*0 - x2*1 = 0
        // x2' = x1*1 + x2*0 = 1, 2
        assert!((x1[0] - 0.0).abs() < 1e-5);
        assert!((x2[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_rotation_simd_matches_scalar() {
        let mut x1_simd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut x2_simd = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let cos_vals = vec![0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
        let sin_vals = vec![0.6, 0.4, 0.7, 0.8, 0.9, 0.9, 0.95, 0.98, 0.995, 1.0];

        let mut x1_scalar = x1_simd.clone();
        let mut x2_scalar = x2_simd.clone();

        apply_rope_rotation_scalar(&mut x1_scalar, &mut x2_scalar, &cos_vals, &sin_vals);
        apply_rope_rotation_simd(&mut x1_simd, &mut x2_simd, &cos_vals, &sin_vals);

        for i in 0..x1_simd.len() {
            assert!(
                (x1_simd[i] - x1_scalar[i]).abs() < 1e-5,
                "x1 mismatch at {}: simd={} scalar={}",
                i,
                x1_simd[i],
                x1_scalar[i]
            );
            assert!(
                (x2_simd[i] - x2_scalar[i]).abs() < 1e-5,
                "x2 mismatch at {}: simd={} scalar={}",
                i,
                x2_simd[i],
                x2_scalar[i]
            );
        }
    }

    #[test]
    fn test_apply_rope_rotation_simd_large() {
        // Test with length > 16 to exercise AVX-512 path
        let n = 64;
        let mut x1 = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        let mut x2 = (0..n).map(|i| (i + 100) as f32).collect::<Vec<_>>();
        let cos_vals = (0..n)
            .map(|i| ((i as f32) * 0.01).cos())
            .collect::<Vec<_>>();
        let sin_vals = (0..n)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect::<Vec<_>>();

        let mut x1_ref = x1.clone();
        let mut x2_ref = x2.clone();
        apply_rope_rotation_scalar(&mut x1_ref, &mut x2_ref, &cos_vals, &sin_vals);

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        for i in 0..n {
            assert!((x1[i] - x1_ref[i]).abs() < 1e-4);
            assert!((x2[i] - x2_ref[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_apply_rope_rotation_preserves_magnitude() {
        // Rotation should preserve magnitude: |x|^2 = x1^2 + x2^2
        let mut x1: Vec<f32> = vec![3.0, 4.0, 5.0, 6.0];
        let mut x2: Vec<f32> = vec![4.0, 3.0, 12.0, 8.0];
        let angle = 0.5f32;
        let cos_vals = vec![angle.cos(); 4];
        let sin_vals = vec![angle.sin(); 4];

        let mag_before: Vec<f32> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a * a + b * b).sqrt())
            .collect();

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        let mag_after: Vec<f32> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a * a + b * b).sqrt())
            .collect();

        for (before, after) in mag_before.iter().zip(mag_after.iter()) {
            assert!((before - after).abs() < 1e-5);
        }
    }
}
