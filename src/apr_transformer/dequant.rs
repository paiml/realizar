//! GGUF K-quant Dequantization Helpers (PMAT-802)
//!
//! Dequantization routines for APR Q4_K/Q6_K support.

// ============================================================================
// GGUF K-quant Dequantization Helpers (for APR Q4_K/Q6_K support)
// =============================================================================

/// Convert IEEE 754 half-precision (f16) bits to f32
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal - convert to normalized f32
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        // Normal number
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Extract scale and min from Q4_K 12-byte packed scales
///
/// PAR-001 FIX: Matches llama.cpp's get_scale_min_k4 packing scheme:
/// - Blocks 0-3: scale = q[j] & 63, min = q[j+4] & 63
/// - Blocks 4-7: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
///   min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
#[inline]
pub(crate) fn extract_scale_min_apr(scales: &[u8], block_idx: usize) -> (f32, f32) {
    let j = block_idx;
    let (scale_bits, min_bits) = if j < 4 {
        // First 4 blocks: simple layout
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        // Last 4 blocks: packed layout using high bits from first 4 bytes
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    };

    (f32::from(scale_bits), f32::from(min_bits))
}

/// Dequantize Q4_K format (K-quants) for APR tensors
/// Q4_K: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
///
/// PMAT-086 FIX: Correct implementation matching llama.cpp/candle layout:
/// - For each 64-value chunk, output 32 low nibbles THEN 32 high nibbles
/// - Use sc1/dm1 for low nibbles, sc2/dm2 for high nibbles (different scales per half)
pub(crate) fn dequantize_q4_k_apr(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256; // Super-block size
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

    let num_blocks = num_elements.div_ceil(QK_K);
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if total_bytes > data.len() {
        // Return zeros if data is insufficient
        return vec![0.0; num_elements];
    }

    let mut result = vec![0.0f32; num_blocks * QK_K];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        // Read d (f16 scale) and dmin (f16 min)
        let d = f16_to_f32(u16::from_le_bytes([data[sb_start], data[sb_start + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[sb_start + 2], data[sb_start + 3]]));

        // Read scales (12 bytes)
        let scales = &data[sb_start + 4..sb_start + 16];

        // Read qs (128 bytes)
        let qs = &data[sb_start + 16..sb_start + 144];

        // Dequantize following candle's layout:
        // For each 64-value chunk, output 32 low nibbles then 32 high nibbles
        let mut ys_index = out_start;

        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min_apr(scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;

            let (sc2, m2) = extract_scale_min_apr(scales, is + 1);
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
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q6_K format (K-quants) for APR tensors
/// Q6_K super-block layout (per llama.cpp block_q6_K and candle):
/// - ql: 128 bytes (low 4 bits, 256 values, 2 per byte)
/// - qh: 64 bytes (high 2 bits, 256 values, 4 per byte)
/// - scales: 16 bytes (i8 signed scales for 16 blocks)
/// - d: 2 bytes (f16)
///
/// Total: 128 + 64 + 16 + 2 = 210 bytes
pub(crate) fn dequantize_q6_k_apr(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    let num_blocks = num_elements.div_ceil(QK_K);
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if total_bytes > data.len() {
        return vec![0.0; num_elements];
    }

    let mut result = vec![0.0f32; num_blocks * QK_K];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        // Read ql - low 4 bits (128 bytes) at offset 0
        let ql = &data[sb_start..sb_start + 128];

        // Read qh - high 2 bits (64 bytes) at offset 128
        let qh = &data[sb_start + 128..sb_start + 192];

        // Read scales (16 bytes, i8) at offset 192
        let mut scales = [0i8; 16];
        #[allow(clippy::cast_possible_wrap)]
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = data[sb_start + 192 + i] as i8;
        }

        // Read d (f16 -> f32) at offset 208 (last 2 bytes)
        let d = f16_to_f32(u16::from_le_bytes([
            data[sb_start + 208],
            data[sb_start + 209],
        ]));

        // Dequantize 256 values following candle's exact layout
        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            for l in 0..32 {
                let is = l / 16; // Scale index selector (0 or 1 within this 128-block)

                // Extract 4 values per iteration (at positions l, l+32, l+64, l+96)
                // q1: low 4 bits of ql[l] + bits 0-1 of qh[l]
                let q1 = ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32;
                // q2: low 4 bits of ql[l+32] + bits 2-3 of qh[l]
                let q2 = ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32;
                // q3: high 4 bits of ql[l] + bits 4-5 of qh[l]
                let q3 = ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32;
                // q4: high 4 bits of ql[l+32] + bits 6-7 of qh[l]
                let q4 = ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32;

                // Write to output with correct scale indexing
                result[out_start + n + l] = d * (sc[is] as f32) * (q1 as f32);
                result[out_start + n + l + 32] = d * (sc[is + 2] as f32) * (q2 as f32);
                result[out_start + n + l + 64] = d * (sc[is + 4] as f32) * (q3 as f32);
                result[out_start + n + l + 96] = d * (sc[is + 6] as f32) * (q4 as f32);
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q8_0 format for APR tensors (GH-191)
///
/// Q8_0 block layout (per GGML):
/// - scale: 2 bytes (f16)
/// - quants: 32 bytes (i8 values)
///   Total: 34 bytes per block, 32 elements per block
pub(crate) fn dequantize_q8_0_apr(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 34; // 2 (f16 scale) + 32 (i8 quants)
    const ELEMENTS_PER_BLOCK: usize = 32;

    let num_blocks = num_elements.div_ceil(ELEMENTS_PER_BLOCK);
    let total_bytes = num_blocks * BLOCK_SIZE;

    if total_bytes > data.len() {
        return vec![0.0; num_elements];
    }

    let mut result = vec![0.0f32; num_blocks * ELEMENTS_PER_BLOCK];

    for blk in 0..num_blocks {
        let blk_start = blk * BLOCK_SIZE;
        let out_start = blk * ELEMENTS_PER_BLOCK;

        // Read scale (f16 → f32)
        let scale = f16_to_f32(u16::from_le_bytes([data[blk_start], data[blk_start + 1]]));

        // Dequantize 32 i8 values
        #[allow(clippy::cast_possible_wrap)]
        for i in 0..ELEMENTS_PER_BLOCK {
            let q = data[blk_start + 2 + i] as i8;
            result[out_start + i] = scale * (q as f32);
        }
    }

    result.truncate(num_elements);
    result
}

// ============================================================================
// Tests for Dequantization Helpers (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // f16_to_f32 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_f16_to_f32_zero() {
        // +0.0 in f16 = 0x0000
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 0.0001);
        // -0.0 in f16 = 0x8000
        assert!((f16_to_f32(0x8000) - (-0.0)).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.0001);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_half() {
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_two() {
        // 2.0 in f16 = 0x4000
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        // +Inf in f16 = 0x7C00
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00) > 0.0);
        // -Inf in f16 = 0xFC00
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00) < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        // NaN in f16: exp=31, mantissa!=0
        assert!(f16_to_f32(0x7C01).is_nan());
        assert!(f16_to_f32(0x7FFF).is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        // Smallest positive subnormal: 0x0001 = 2^-24
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0);
        assert!(result < 0.001); // Very small
    }

    // -------------------------------------------------------------------------
    // extract_scale_min_apr Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_scale_min_apr_first_four_blocks() {
        // 12 bytes of scales
        let scales = [10, 20, 30, 40, 5, 15, 25, 35, 0, 0, 0, 0];

        // Block 0: scale = scales[0] & 63 = 10, min = scales[4] & 63 = 5
        let (s, m) = extract_scale_min_apr(&scales, 0);
        assert!((s - 10.0).abs() < 0.001);
        assert!((m - 5.0).abs() < 0.001);

        // Block 1: scale = scales[1] & 63 = 20, min = scales[5] & 63 = 15
        let (s, m) = extract_scale_min_apr(&scales, 1);
        assert!((s - 20.0).abs() < 0.001);
        assert!((m - 15.0).abs() < 0.001);

        // Block 3: scale = scales[3] & 63 = 40, min = scales[7] & 63 = 35
        let (s, m) = extract_scale_min_apr(&scales, 3);
        assert!((s - 40.0).abs() < 0.001);
        assert!((m - 35.0).abs() < 0.001);
    }

    #[test]
    fn test_extract_scale_min_apr_last_four_blocks() {
        // 12 bytes of scales with specific values for testing packed layout
        // For block 4+: d = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
        //               m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
        let scales = [0, 0, 0, 0, 0, 0, 0, 0, 0x12, 0x34, 0x56, 0x78];

        // Block 4: j=4, uses scales[8] and scales[0]
        let (s, m) = extract_scale_min_apr(&scales, 4);
        // scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
        //       = (0x12 & 0x0F) | ((0 >> 6) << 4) = 0x02 = 2
        // min = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
        //     = (0x12 >> 4) | ((0 >> 6) << 4) = 0x01 = 1
        assert!((s - 2.0).abs() < 0.001);
        assert!((m - 1.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // dequantize_q4_k_apr Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q4_k_apr_empty() {
        let data: Vec<u8> = vec![];
        let result = dequantize_q4_k_apr(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4_k_apr_insufficient_data() {
        let data: Vec<u8> = vec![0; 10]; // Less than 144 bytes
        let result = dequantize_q4_k_apr(&data, 256);
        // Should return zeros when data is insufficient
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q4_k_apr_zeros() {
        // 144 bytes of zeros (one super-block)
        let data = vec![0u8; 144];
        let result = dequantize_q4_k_apr(&data, 256);
        assert_eq!(result.len(), 256);
        // With d=0.0, all values should be 0.0
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q4_k_apr_truncation() {
        // Request fewer elements than super-block size
        let data = vec![0u8; 144];
        let result = dequantize_q4_k_apr(&data, 100);
        assert_eq!(result.len(), 100);
    }

    // -------------------------------------------------------------------------
    // dequantize_q6_k_apr Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q6_k_apr_empty() {
        let data: Vec<u8> = vec![];
        let result = dequantize_q6_k_apr(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6_k_apr_insufficient_data() {
        let data: Vec<u8> = vec![0; 100]; // Less than 210 bytes
        let result = dequantize_q6_k_apr(&data, 256);
        // Should return zeros when data is insufficient
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q6_k_apr_zeros() {
        // 210 bytes of zeros (one super-block)
        let data = vec![0u8; 210];
        let result = dequantize_q6_k_apr(&data, 256);
        assert_eq!(result.len(), 256);
        // With d=0.0, all values should be 0.0
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q6_k_apr_truncation() {
        // Request fewer elements than super-block size
        let data = vec![0u8; 210];
        let result = dequantize_q6_k_apr(&data, 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_dequantize_q6_k_apr_multiple_blocks() {
        // Two super-blocks (420 bytes)
        let data = vec![0u8; 420];
        let result = dequantize_q6_k_apr(&data, 512);
        assert_eq!(result.len(), 512);
    }

    // -------------------------------------------------------------------------
    // dequantize_q8_0_apr Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q8_0_apr_zeros() {
        // Q8_0: 34 bytes per block (2 f16 scale + 32 i8 quants)
        // All zeros => scale=0, so all outputs should be 0
        let data = vec![0u8; 34];
        let result = dequantize_q8_0_apr(&data, 32);
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q8_0_apr_single_block() {
        // scale = 1.0 in f16 = 0x3C00 little-endian
        let mut data = vec![0u8; 34];
        data[0] = 0x00; // f16 1.0 low byte
        data[1] = 0x3C; // f16 1.0 high byte
        // Set quants: all i8 = 1 (unsigned byte 1)
        for i in 2..34 {
            data[i] = 1;
        }
        let result = dequantize_q8_0_apr(&data, 32);
        assert_eq!(result.len(), 32);
        // Each value should be scale * q = 1.0 * 1 = 1.0
        for &v in &result {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_dequantize_q8_0_apr_negative_quants() {
        let mut data = vec![0u8; 34];
        data[0] = 0x00;
        data[1] = 0x3C; // scale = 1.0
        // Set quants: all i8 = -1 (0xFF as u8)
        for i in 2..34 {
            data[i] = 0xFF;
        }
        let result = dequantize_q8_0_apr(&data, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - (-1.0)).abs() < 0.01, "expected ~-1.0, got {v}");
        }
    }

    #[test]
    fn test_dequantize_q8_0_apr_truncation() {
        // Request fewer elements than block size
        let data = vec![0u8; 34];
        let result = dequantize_q8_0_apr(&data, 16);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_dequantize_q8_0_apr_multiple_blocks() {
        // Two blocks (68 bytes)
        let data = vec![0u8; 68];
        let result = dequantize_q8_0_apr(&data, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_q8_0_apr_insufficient_data() {
        // Not enough data for even one block
        let data = vec![0u8; 10];
        let result = dequantize_q8_0_apr(&data, 32);
        assert_eq!(result.len(), 32);
        // Should return zeros when data is insufficient
        assert!(result.iter().all(|&x| x == 0.0));
    }
}
