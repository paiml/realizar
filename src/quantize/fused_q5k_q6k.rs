//! Fused Q5K and Q6K dot product operations (PMAT-802)
//!
//! Implements fused dequant+dot operations for Q5_K and Q6_K formats:
//! - `fused_q6k_dot`, `fused_q6k_dot_simd` - Q6_K dot products
//! - `fused_q5k_dot`, `fused_q5k_dot_simd` - Q5_K dot products

use super::dequant::read_f16;
use super::simd::extract_scale_min;
use super::types::{Q8_0Block, QK_K};
use crate::error::{RealizarError, Result};

/// Fused Q6_K dequantize + dot product
///
/// Computes the dot product of Q6_K quantized weights with f32 activations.
pub fn fused_q6k_dot(q6k_data: &[u8], activations: &[f32]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 210;

    // Validate Q6_K data length
    if !q6k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q6_K data length {} is not a multiple of super-block size {}",
                q6k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q6k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    // Validate activation length matches
    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q6_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Accumulator for dot product result
    let mut acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let act_start = sb_idx * QK_K;

        // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2)
        let ql = &q6k_data[sb_start..sb_start + 128];
        let qh = &q6k_data[sb_start + 128..sb_start + 192];

        // Read scales (16 bytes, i8)
        let mut scales = [0i8; 16];
        for (i, scale) in scales.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_wrap)]
            {
                *scale = q6k_data[sb_start + 192 + i] as i8;
            }
        }

        // Read d (f16 -> f32) at offset 208
        let d = read_f16(&q6k_data[sb_start + 208..sb_start + 210]);

        // Fused dequant+dot following candle's exact layout
        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            for l in 0..32 {
                let is = l / 16; // Scale index selector

                // Extract 4 values per iteration (at positions l, l+32, l+64, l+96)
                let q1 = ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32;

                // Dequantize and accumulate dot product
                let v1 = d * (sc[is] as f32) * (q1 as f32);
                let v2 = d * (sc[is + 2] as f32) * (q2 as f32);
                let v3 = d * (sc[is + 4] as f32) * (q3 as f32);
                let v4 = d * (sc[is + 6] as f32) * (q4 as f32);

                acc += v1 * activations[act_start + n + l];
                acc += v2 * activations[act_start + n + l + 32];
                acc += v3 * activations[act_start + n + l + 64];
                acc += v4 * activations[act_start + n + l + 96];
            }
        }
    }

    Ok(acc)
}

/// SIMD-accelerated fused Q6_K dequant+dot (with scalar fallback)
///
/// Per Williams et al. (2009) roofline model, memory bandwidth is the bottleneck.
/// This function provides a unified interface with runtime feature detection.
/// Currently uses scalar implementation; SIMD Q6_K optimization can be added later.
///
/// # Arguments
///
/// * `q6k_data` - Raw Q6_K quantized data (210 bytes per super-block)
/// * `activations` - Input activations (256 values per super-block)
///
/// # Returns
///
/// Dot product result as f32
///
/// # Errors
///
/// Returns error if data sizes don't match or are malformed
pub fn fused_q6k_dot_simd(q6k_data: &[u8], activations: &[f32]) -> Result<f32> {
    // PAR-126: AVX2 SIMD implementation for Q6_K
    // Critical optimization: Q6_K scalar was 9x slower than Q4_K SIMD
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We've verified AVX2 and FMA are available at runtime
            return unsafe { fused_q6k_dot_avx2(q6k_data, activations) };
        }
    }
    // pmat-ignore: hardware-path (scalar fallback tested directly via fused_q6k_dot)
    // Fallback to scalar implementation
    fused_q6k_dot(q6k_data, activations)
}

/// PAR-126: AVX2 SIMD implementation for Q6_K dot product
///
/// Uses AVX2 + FMA to achieve ~8x speedup over scalar.
/// Q6_K layout: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes
///
/// # Safety
/// Requires AVX2 and FMA instruction sets
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_q6k_dot_avx2(q6k_data: &[u8], activations: &[f32]) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 210;

    if !q6k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q6_K data length {} is not a multiple of super-block size {}",
                q6k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q6k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q6_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // 4 independent accumulators to hide FMA latency
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    // Masks for 6-bit extraction (reserved for optimized path)
    let _mask_0f = _mm256_set1_epi8(0x0F_i8);
    let _mask_03 = _mm256_set1_epi8(0x03_i8);
    let offset_32 = _mm256_set1_epi32(32);

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let act_start = sb_idx * QK_K;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            let next_sb = (sb_idx + 1) * SUPER_BLOCK_BYTES;
            _mm_prefetch(q6k_data.as_ptr().add(next_sb).cast::<i8>(), _MM_HINT_T0);
        }

        // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2)
        let ql_ptr = q6k_data.as_ptr().add(sb_start);
        let qh_ptr = q6k_data.as_ptr().add(sb_start + 128);
        let scales_ptr = q6k_data.as_ptr().add(sb_start + 192);

        // Read d (f16 -> f32)
        let d = read_f16(&q6k_data[sb_start + 208..sb_start + 210]);
        let d_vec = _mm256_set1_ps(d);

        // Read all 16 scales (as i8, will use in inner loop)
        let mut scales = [0i8; 16];
        std::ptr::copy_nonoverlapping(scales_ptr, scales.as_mut_ptr().cast::<u8>(), 16);

        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = ql_ptr.add(64 * idx);
            let qh_slice = qh_ptr.add(32 * idx);
            let act_base = activations.as_ptr().add(act_start + n);

            // Process 32 values at a time using AVX2
            // Each iteration handles l=0..8, extracting 4 values each (32 total)
            for l_base in (0..32).step_by(8) {
                // Load 8 bytes of ql[l], ql[l+32], qh[l]
                let ql_lo_64 = std::ptr::read_unaligned(ql_slice.add(l_base).cast::<u64>());
                let ql_hi_64 = std::ptr::read_unaligned(ql_slice.add(l_base + 32).cast::<u64>());
                let qh_64 = std::ptr::read_unaligned(qh_slice.add(l_base).cast::<u64>());

                // Convert to SIMD vectors (expand u8 to i32 for arithmetic)
                let ql_lo = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, ql_lo_64 as i64));
                let ql_hi = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, ql_hi_64 as i64));
                let qh = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, qh_64 as i64));

                // Extract 6-bit values (4 values per input byte)
                // q1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4) - 32
                let q1_lo = _mm256_and_si256(ql_lo, _mm256_set1_epi32(0x0F));
                let q1_hi = _mm256_slli_epi32(_mm256_and_si256(qh, _mm256_set1_epi32(0x03)), 4);
                let q1 = _mm256_sub_epi32(_mm256_or_si256(q1_lo, q1_hi), offset_32);

                // q2 = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32
                let q2_lo = _mm256_and_si256(ql_hi, _mm256_set1_epi32(0x0F));
                let q2_hi = _mm256_slli_epi32(
                    _mm256_and_si256(_mm256_srli_epi32(qh, 2), _mm256_set1_epi32(0x03)),
                    4,
                );
                let q2 = _mm256_sub_epi32(_mm256_or_si256(q2_lo, q2_hi), offset_32);

                // q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
                let q3_lo = _mm256_srli_epi32(ql_lo, 4);
                let q3_hi = _mm256_slli_epi32(
                    _mm256_and_si256(_mm256_srli_epi32(qh, 4), _mm256_set1_epi32(0x03)),
                    4,
                );
                let q3 = _mm256_sub_epi32(_mm256_or_si256(q3_lo, q3_hi), offset_32);

                // q4 = (ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
                let q4_lo = _mm256_srli_epi32(ql_hi, 4);
                let q4_hi = _mm256_slli_epi32(_mm256_srli_epi32(qh, 6), 4);
                let q4 = _mm256_sub_epi32(_mm256_or_si256(q4_lo, q4_hi), offset_32);

                // Determine scale index: is = l / 16 (0 for l<16, 1 for l>=16)
                let is = l_base / 16;

                // Get scales for each of the 4 output values
                let sc1 = sc[is] as f32;
                let sc2 = sc[is + 2] as f32;
                let sc3 = sc[is + 4] as f32;
                let sc4 = sc[is + 6] as f32;

                // Convert quantized values to f32 and multiply by d*scale
                let q1_f32 = _mm256_cvtepi32_ps(q1);
                let q2_f32 = _mm256_cvtepi32_ps(q2);
                let q3_f32 = _mm256_cvtepi32_ps(q3);
                let q4_f32 = _mm256_cvtepi32_ps(q4);

                let dequant1 = _mm256_mul_ps(_mm256_mul_ps(d_vec, _mm256_set1_ps(sc1)), q1_f32);
                let dequant2 = _mm256_mul_ps(_mm256_mul_ps(d_vec, _mm256_set1_ps(sc2)), q2_f32);
                let dequant3 = _mm256_mul_ps(_mm256_mul_ps(d_vec, _mm256_set1_ps(sc3)), q3_f32);
                let dequant4 = _mm256_mul_ps(_mm256_mul_ps(d_vec, _mm256_set1_ps(sc4)), q4_f32);

                // Load activations
                let act1 = _mm256_loadu_ps(act_base.add(l_base));
                let act2 = _mm256_loadu_ps(act_base.add(l_base + 32));
                let act3 = _mm256_loadu_ps(act_base.add(l_base + 64));
                let act4 = _mm256_loadu_ps(act_base.add(l_base + 96));

                // FMA: acc += dequant * act
                acc0 = _mm256_fmadd_ps(dequant1, act1, acc0);
                acc1 = _mm256_fmadd_ps(dequant2, act2, acc1);
                acc2 = _mm256_fmadd_ps(dequant3, act3, acc2);
                acc3 = _mm256_fmadd_ps(dequant4, act4, acc3);
            }
        }
    }

    // Combine 4 accumulators
    let acc_01 = _mm256_add_ps(acc0, acc1);
    let acc_23 = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc_01, acc_23);

    // Horizontal sum
    let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
    let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    let result = _mm_cvtss_f32(temp);

    Ok(result)
}

/// Fused Q5_K dequantize + dot product
///
/// Computes the dot product of Q5_K quantized weights with f32 activations
/// WITHOUT allocating an intermediate f32 buffer. Dequantization happens
/// inline, accumulating directly into a register.
///
/// # Arguments
///
/// * `q5k_data` - Raw Q5_K quantized data (super-blocks of 176 bytes)
/// * `activations` - f32 activation values (must match dequantized length)
///
/// # Returns
///
/// The dot product as f32
///
/// # Errors
///
/// Returns error if:
/// - `q5k_data` length is not a multiple of 176 bytes (super-block size)
/// - `activations` length doesn't match the number of quantized values
///
/// # Examples
///
/// ```rust,ignore
/// let weights_q5k = load_q5k_weights();
/// let activations = get_layer_activations();
/// let result = fused_q5k_dot(&weights_q5k, &activations)?;
/// ```
#[allow(clippy::similar_names)]
pub fn fused_q5k_dot(q5k_data: &[u8], activations: &[f32]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 176;

    // Validate Q5_K data length
    if !q5k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_K data length {} is not a multiple of super-block size {}",
                q5k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q5k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    // Validate activation length matches
    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q5_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Accumulator for dot product result
    let mut acc = 0.0f32;
    let mut activation_idx = 0;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Read d (f16 -> f32)
        let d = read_f16(&q5k_data[sb_start..sb_start + 2]);

        // Read dmin (f16 -> f32)
        let dmin = read_f16(&q5k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q5k_data[sb_start + 4..sb_start + 16]);

        // Read qh - high bits (32 bytes)
        let qh_start = sb_start + 16;
        let qh = &q5k_data[qh_start..qh_start + 32];

        // Read qs - low 4 bits (128 bytes)
        let qs_start = sb_start + 48;
        let qs = &q5k_data[qs_start..qs_start + 128];

        // Fused dequant+dot for 8 blocks of 32 values each
        for block_idx in 0..8 {
            // Extract 6-bit scale and min for this block
            let (scale, min) = extract_scale_min(&scales, block_idx);

            // Process 32 values
            let block_start = block_idx * 16;
            let qh_block_start = block_idx * 4;

            for byte_idx in 0..16 {
                let qs_byte = qs[block_start + byte_idx];
                let high_bits_byte = qh[qh_block_start + byte_idx / 4];
                let bit_offset = (byte_idx % 4) * 2;

                // Low value: dequantize and accumulate
                let q_low_4bit = qs_byte & 0x0F;
                let q_low_high_bit = (high_bits_byte >> bit_offset) & 0x01;
                #[allow(clippy::cast_possible_wrap)]
                let q_low = ((q_low_high_bit << 4) | q_low_4bit) as i8;
                let value_low = d * scale * f32::from(q_low) - dmin * min;
                acc += value_low * activations[activation_idx];
                activation_idx += 1;

                // High value: dequantize and accumulate
                let q_high_4bit = (qs_byte >> 4) & 0x0F;
                let q_high_high_bit = (high_bits_byte >> (bit_offset + 1)) & 0x01;
                #[allow(clippy::cast_possible_wrap)]
                let q_high = ((q_high_high_bit << 4) | q_high_4bit) as i8;
                let value_high = d * scale * f32::from(q_high) - dmin * min;
                acc += value_high * activations[activation_idx];
                activation_idx += 1;
            }
        }
    }

    Ok(acc)
}

/// SIMD-accelerated fused Q5_K dequant+dot (with scalar fallback)
///
/// Provides unified interface with runtime feature detection.
/// Currently uses scalar implementation; SIMD Q5_K can be added later.
///
/// # Errors
///
/// Returns error if data sizes don't match or are malformed.
/// See [`fused_q5k_dot`] for details.
pub fn fused_q5k_dot_simd(q5k_data: &[u8], activations: &[f32]) -> Result<f32> {
    // Q5_K SIMD optimization deferred to Phase 2
    fused_q5k_dot(q5k_data, activations)
}

/// Fused Q4_K with Q8 blocks dot product
///
/// Computes the dot product of Q4_K quantized weights with Q8_0 quantized activations.
pub fn fused_q4k_q8_dot(q4k_data: &[u8], q8_blocks: &[Q8_0Block]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 144;

    // Validate Q4_K data length
    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K; // 256 values per super-block
    let expected_q8_blocks = expected_values / 32;

    // Validate Q8 block count matches
    if q8_blocks.len() != expected_q8_blocks {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 block count {} doesn't match expected {} (for {} Q4_K values)",
                q8_blocks.len(),
                expected_q8_blocks,
                expected_values
            ),
        });
    }

    // Accumulator for dot product result
    let mut acc = 0.0f32;
    let mut q8_block_idx = 0;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Read d (f16 -> f32) - super-block scale
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);

        // Read dmin (f16 -> f32) - super-block min
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes) - packed 6-bit scales for 8 blocks
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Read qs (128 bytes) - 256 4-bit quantized values
        let qs_start = sb_start + 16;
        let qs = &q4k_data[qs_start..qs_start + 128];

        // Process 8 blocks of 32 values each
        for block_idx in 0..8 {
            // Extract 6-bit scale and min for this block
            let (scale, min) = extract_scale_min(&scales, block_idx);

            // Get the Q8 block for this 32-value chunk
            let q8_block = &q8_blocks[q8_block_idx];
            let q8_scale = q8_block.scale;
            q8_block_idx += 1;

            // Process 32 values (16 bytes, 2 4-bit values per byte)
            let block_start = block_idx * 16;
            for byte_idx in 0..16 {
                let byte = qs[block_start + byte_idx];
                let q8_idx = byte_idx * 2;

                // Low 4 bits: fused dequant and accumulate
                #[allow(clippy::cast_possible_wrap)]
                let q4_low = (byte & 0x0F) as i8;
                let w_low = d * scale * f32::from(q4_low) - dmin * min;
                let a_low = q8_scale * f32::from(q8_block.quants[q8_idx]);
                acc += w_low * a_low;

                // High 4 bits: fused dequant and accumulate
                #[allow(clippy::cast_possible_wrap)]
                let q4_high = ((byte >> 4) & 0x0F) as i8;
                let w_high = d * scale * f32::from(q4_high) - dmin * min;
                let a_high = q8_scale * f32::from(q8_block.quants[q8_idx + 1]);
                acc += w_high * a_high;
            }
        }
    }

    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Q6_K FUSED DOT PRODUCT TESTS
    // ============================================================================

    #[test]
    fn test_fused_q6k_dot_invalid_data_length() {
        // Q6_K super-block is 210 bytes, test with non-multiple
        let data = vec![0u8; 100]; // Not a multiple of 210
        let activations = vec![0.0f32; 256];

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a multiple"));
    }

    #[test]
    fn test_fused_q6k_dot_activation_length_mismatch() {
        // One super-block = 210 bytes = 256 values
        let data = vec![0u8; 210];
        let activations = vec![0.0f32; 128]; // Should be 256

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_fused_q6k_dot_zero_data() {
        // All zeros should produce zero dot product
        let mut data = vec![0u8; 210];
        // Set d to 0 (f16 zero)
        data[208..210].copy_from_slice(&[0x00, 0x00]);
        let activations = vec![1.0f32; 256];

        let result = fused_q6k_dot(&data, &activations).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_fused_q6k_dot_single_super_block() {
        // Create valid Q6_K data: ql (128) + qh (64) + scales (16) + d (2)
        let mut data = vec![0u8; 210];

        // Set d = 1.0 (f16: 0x3C00)
        data[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());

        // Set some scales (signed i8)
        for i in 0..16 {
            data[192 + i] = 1; // Small positive scale
        }

        // Set ql values to small patterns
        for i in 0..128 {
            data[i] = ((i % 16) as u8) | (((i % 16) as u8) << 4);
        }

        let activations = vec![1.0f32; 256];

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q6k_dot_simd_matches_scalar() {
        // Create valid Q6_K data for comparison
        let mut data = vec![0u8; 210];

        // Set d = 1.0 (f16: 0x3C00)
        data[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());

        // Set scales to 1
        for i in 0..16 {
            data[192 + i] = 1;
        }

        // Set ql/qh to deterministic pattern
        for i in 0..128 {
            data[i] = (i % 256) as u8;
        }
        for i in 0..64 {
            data[128 + i] = (i % 256) as u8;
        }

        let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        let scalar_result = fused_q6k_dot(&data, &activations).unwrap();
        let simd_result = fused_q6k_dot_simd(&data, &activations).unwrap();

        // Allow 1% tolerance for SIMD vs scalar differences
        let rel_err = if scalar_result.abs() > 1e-6 {
            (simd_result - scalar_result).abs() / scalar_result.abs()
        } else {
            (simd_result - scalar_result).abs()
        };
        assert!(
            rel_err < 0.01,
            "scalar={} simd={} rel_err={}",
            scalar_result,
            simd_result,
            rel_err
        );
    }

    #[test]
    fn test_fused_q6k_dot_simd_invalid_input() {
        let data = vec![0u8; 100]; // Invalid length
        let activations = vec![0.0f32; 256];

        let result = fused_q6k_dot_simd(&data, &activations);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q6k_dot_multiple_super_blocks() {
        // Two super-blocks = 420 bytes = 512 values
        let mut data = vec![0u8; 420];

        // Set d = 0.5 for both blocks (f16: 0x3800)
        data[208..210].copy_from_slice(&0x3800u16.to_le_bytes());
        data[418..420].copy_from_slice(&0x3800u16.to_le_bytes());

        // Set scales
        for i in 0..16 {
            data[192 + i] = 2;
            data[402 + i] = 2;
        }

        let activations = vec![0.5f32; 512];

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    // ============================================================================
    // Q5_K FUSED DOT PRODUCT TESTS
    // ============================================================================

    #[test]
    fn test_fused_q5k_dot_invalid_data_length() {
        // Q5_K super-block is 176 bytes
        let data = vec![0u8; 100]; // Not a multiple of 176
        let activations = vec![0.0f32; 256];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a multiple"));
    }

    #[test]
    fn test_fused_q5k_dot_activation_length_mismatch() {
        // One super-block = 176 bytes = 256 values
        let data = vec![0u8; 176];
        let activations = vec![0.0f32; 128]; // Should be 256

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_fused_q5k_dot_zero_data() {
        // All zeros should produce zero dot product
        let data = vec![0u8; 176];
        // d = 0, dmin = 0
        let activations = vec![1.0f32; 256];

        let result = fused_q5k_dot(&data, &activations).unwrap();
        // With zero scales, result should be close to zero
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_fused_q5k_dot_single_super_block() {
        // Q5_K layout: d (2) + dmin (2) + scales (12) + qh (32) + qs (128)
        let mut data = vec![0u8; 176];

        // Set d = 1.0 (f16: 0x3C00)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

        // Set dmin = 0.5 (f16: 0x3800)
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

        // Set qs values to pattern
        for i in 0..128 {
            data[48 + i] = ((i % 16) as u8) | (((i + 1) % 16) << 4) as u8;
        }

        let activations = vec![1.0f32; 256];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q5k_dot_simd_matches_scalar() {
        // Create valid Q5_K data
        let mut data = vec![0u8; 176];

        // Set d = 1.0, dmin = 0.5
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

        // Set scales (12 bytes packed)
        for i in 0..12 {
            data[4 + i] = 0x11; // Small positive scales
        }

        // Set qh and qs
        for i in 0..32 {
            data[16 + i] = (i % 256) as u8;
        }
        for i in 0..128 {
            data[48 + i] = ((i * 3) % 256) as u8;
        }

        let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        let scalar_result = fused_q5k_dot(&data, &activations).unwrap();
        let simd_result = fused_q5k_dot_simd(&data, &activations).unwrap();

        // SIMD currently uses scalar, so should be exact match
        assert!(
            (scalar_result - simd_result).abs() < 1e-6,
            "scalar={} simd={}",
            scalar_result,
            simd_result
        );
    }

    #[test]
    fn test_fused_q5k_dot_simd_invalid_input() {
        let data = vec![0u8; 100]; // Invalid length
        let activations = vec![0.0f32; 256];

        let result = fused_q5k_dot_simd(&data, &activations);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q5k_dot_multiple_super_blocks() {
        // Two super-blocks = 352 bytes = 512 values
        let mut data = vec![0u8; 352];

        // Set d and dmin for both blocks
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        data[176..178].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[178..180].copy_from_slice(&0x3800u16.to_le_bytes());

        let activations = vec![0.5f32; 512];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    // ============================================================================
    // Q4_K × Q8 DOT PRODUCT TESTS
    // ============================================================================

    /// Helper to create a Q8_0Block with zero values
    fn zero_q8_block() -> Q8_0Block {
        Q8_0Block {
            scale: 0.0,
            quants: [0i8; 32],
        }
    }

    /// Helper to create a Q8_0Block with specified scale and quant value
    fn make_q8_block(scale: f32, quant_val: i8) -> Q8_0Block {
        Q8_0Block {
            scale,
            quants: [quant_val; 32],
        }
    }

    #[test]
    fn test_fused_q4k_q8_dot_invalid_data_length() {
        // Q4_K super-block is 144 bytes
        let data = vec![0u8; 100]; // Not a multiple of 144
        let q8_blocks: Vec<Q8_0Block> = (0..8).map(|_| zero_q8_block()).collect();

        let result = fused_q4k_q8_dot(&data, &q8_blocks);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a multiple"));
    }

    #[test]
    fn test_fused_q4k_q8_dot_block_count_mismatch() {
        // One super-block = 144 bytes = 256 values = 8 Q8 blocks
        let data = vec![0u8; 144];
        let q8_blocks: Vec<Q8_0Block> = (0..4).map(|_| zero_q8_block()).collect(); // Should be 8

        let result = fused_q4k_q8_dot(&data, &q8_blocks);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_fused_q4k_q8_dot_zero_data() {
        // All zeros should produce zero dot product
        let data = vec![0u8; 144];
        let q8_blocks: Vec<Q8_0Block> = (0..8).map(|_| zero_q8_block()).collect();

        let result = fused_q4k_q8_dot(&data, &q8_blocks).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_fused_q4k_q8_dot_single_super_block() {
        // Create valid Q4_K data: d (2) + dmin (2) + scales (12) + qs (128)
        let mut data = vec![0u8; 144];

        // Set d = 1.0, dmin = 0.5
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

        // Set qs values
        for i in 0..128 {
            data[16 + i] = 0x55; // Pattern: low=5, high=5
        }

        // Create Q8 blocks with some values
        let q8_blocks: Vec<Q8_0Block> = (0..8).map(|_| make_q8_block(0.1, 10)).collect();

        let result = fused_q4k_q8_dot(&data, &q8_blocks);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q4k_q8_dot_multiple_super_blocks() {
        // Two super-blocks = 288 bytes = 512 values = 16 Q8 blocks
        let mut data = vec![0u8; 288];

        // Set headers for both blocks
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[144..146].copy_from_slice(&0x3C00u16.to_le_bytes());

        let q8_blocks: Vec<Q8_0Block> = (0..16).map(|_| zero_q8_block()).collect();

        let result = fused_q4k_q8_dot(&data, &q8_blocks);
        assert!(result.is_ok());
    }

    // ============================================================================
    // EDGE CASE TESTS
    // ============================================================================

    #[test]
    fn test_fused_q6k_dot_negative_scales() {
        let mut data = vec![0u8; 210];

        // Set d = 1.0
        data[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());

        // Set negative scales (i8)
        #[allow(clippy::cast_sign_loss)]
        for i in 0..16 {
            data[192 + i] = (-5i8) as u8;
        }

        let activations = vec![1.0f32; 256];

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q5k_dot_with_high_bits() {
        let mut data = vec![0u8; 176];

        // Set d = 1.0, dmin = 0.1
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x2E66u16.to_le_bytes()); // ~0.1

        // Set qh (high bits) to all 1s
        for i in 0..32 {
            data[16 + i] = 0xFF;
        }

        // Set qs
        for i in 0..128 {
            data[48 + i] = 0xF0; // high nibbles
        }

        let activations = vec![1.0f32; 256];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q6k_dot_empty_data() {
        let data = vec![];
        let activations = vec![];

        // Empty is valid (0 super-blocks)
        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn test_fused_q5k_dot_empty_data() {
        let data = vec![];
        let activations = vec![];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
    }
}
