//! Dequantization functions for GGUF quantization formats (PMAT-802)
//!
//! Extracted from quantize/mod.rs - Basic dequantization for Q4_0, Q8_0, F16,
//! Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q2_K formats.
//!
//! ## Contents
//! - `dequantize_q4_0` - 4-bit quantization (block size 32)
//! - `dequantize_q8_0` - 8-bit quantization (block size 32)
//! - `dequantize_f16` - Half-precision to f32
//! - `dequantize_q4_1` - 4-bit with scale and min
//! - `dequantize_q5_0` - 5-bit quantization
//! - `dequantize_q5_1` - 5-bit with scale and min
//! - `dequantize_q4_k` - K-quantization 4-bit (super-block 256)
//! - `dequantize_q5_k` - K-quantization 5-bit (super-block 256)
//! - `dequantize_q6_k` - K-quantization 6-bit (super-block 256)
//! - `dequantize_q2_k` - K-quantization 2-bit (super-block 256)

use crate::error::{RealizarError, Result};
use crate::quantize::{BLOCK_SIZE, QK_K};

// Re-use helpers from simd module
use super::simd::extract_scale_min;

/// Dequantize `Q4_0` format weights
///
/// # Arguments
///
/// * `data` - Raw `Q4_0` quantized data (blocks of scale + 16 bytes)
///
/// # Returns
///
/// Dequantized float32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of block size
///
/// # Examples
///
/// ```rust,ignore
/// let quantized = load_q4_0_weights();
/// let weights = dequantize_q4_0(&quantized)?;
/// ```
pub fn dequantize_q4_0(data: &[u8]) -> Result<Vec<f32>> {
    // Q4_0 block: 2 bytes (f16 scale) + 16 bytes (quants) = 18 bytes
    // GGML spec: typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;
    const BLOCK_BYTES: usize = 2 + 16;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        // Read scale (f16, per GGML spec)
        let scale_bytes = &data[block_start..block_start + 2];
        let scale = half::f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();

        // Read quantized values (16 bytes)
        let quants_start = block_start + 2;
        let quants = &data[quants_start..quants_start + 16];

        // Dequantize following candle's layout:
        // - Positions 0-15: low nibbles of bytes 0-15
        // - Positions 16-31: high nibbles of bytes 0-15
        for (j, &byte) in quants.iter().enumerate() {
            // Low 4 bits go to position j
            #[allow(clippy::cast_possible_wrap)]
            let low = (byte & 0x0F) as i16 - 8;
            result[out_start + j] = scale * (low as f32);

            // High 4 bits go to position j + 16
            #[allow(clippy::cast_possible_wrap)]
            let high = (byte >> 4) as i16 - 8;
            result[out_start + j + 16] = scale * (high as f32);
        }
    }

    Ok(result)
}

/// Dequantize `Q8_0` format weights
///
/// # Arguments
///
/// * `data` - Raw `Q8_0` quantized data (blocks of scale + 32 int8 values)
///
/// # Returns
///
/// Dequantized float32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of block size
pub fn dequantize_q8_0(data: &[u8]) -> Result<Vec<f32>> {
    // Q8_0 block: 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes
    // Note: GGML spec uses f16 for scale, not f32!
    const BLOCK_BYTES: usize = 2 + 32;

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
    let mut result = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;

        // Read scale (f16 -> f32)
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);

        // Read quantized values (32 int8 values)
        let quants_start = block_start + 2;
        let quants = &data[quants_start..quants_start + 32];

        // Dequantize
        for &byte in quants {
            let value = i8::from_le_bytes([byte]);
            result.push(scale * f32::from(value));
        }
    }

    Ok(result)
}

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
///
/// Handles normal values, subnormals, infinities, and NaN.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mantissa = h & 0x3FF;

    if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            // Zero (preserve sign)
            if sign == 1 {
                -0.0
            } else {
                0.0
            }
        } else {
            // Subnormal: (mantissa / 1024) * 2^-14
            let value = (mantissa as f32 / 1024.0) * (2.0_f32).powi(-14);
            if sign == 1 {
                -value
            } else {
                value
            }
        }
    } else if exp == 31 {
        // Infinity or NaN
        if mantissa == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        // Normal value: (1 + mantissa/1024) * 2^(exp-15)
        let value = (1.0 + mantissa as f32 / 1024.0) * (2.0_f32).powi(exp as i32 - 15);
        if sign == 1 {
            -value
        } else {
            value
        }
    }
}

/// Dequantize `F16` format weights to `F32`
///
/// # Arguments
///
/// * `data` - Raw F16 data (2 bytes per value)
///
/// # Returns
///
/// Dequantized float32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of 2 bytes
pub fn dequantize_f16(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(2) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "F16 data length {} is not a multiple of 2 bytes",
                data.len()
            ),
        });
    }

    let num_values = data.len() / 2;
    let mut result = Vec::with_capacity(num_values);

    for chunk in data.chunks_exact(2) {
        let h = u16::from_le_bytes([chunk[0], chunk[1]]);
        result.push(f16_to_f32(h));
    }

    Ok(result)
}

/// Dequantize `Q4_1` format weights
///
/// Q4_1 format: 2 bytes (f16 scale) + 2 bytes (f16 min) + 16 bytes (quants) = 20 bytes
/// GGUF/candle layout:
/// - Positions 0-15: low nibbles of bytes 0-15
/// - Positions 16-31: high nibbles of bytes 0-15
pub fn dequantize_q4_1(data: &[u8]) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 20;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_1 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    // Pre-allocate with correct size for candle layout
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        let d_bytes = &data[block_start..block_start + 2];
        let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));

        let min_bytes = &data[block_start + 2..block_start + 4];
        let min = f16_to_f32(u16::from_le_bytes([min_bytes[0], min_bytes[1]]));

        let quants = &data[block_start + 4..block_start + 20];

        // Use candle layout (same as Q4_0):
        // - Low nibbles (byte & 0xF) at positions 0-15
        // - High nibbles (byte >> 4) at positions 16-31
        for (j, &byte) in quants.iter().enumerate() {
            // Low 4 bits go to position j (0-15)
            let low = byte & 0x0F;
            result[out_start + j] = d * f32::from(low) + min;

            // High 4 bits go to position j + 16 (16-31)
            let high = (byte >> 4) & 0x0F;
            result[out_start + j + 16] = d * f32::from(high) + min;
        }
    }

    Ok(result)
}

/// Dequantize `Q5_0` format weights
///
/// Q5_0 format: 2 bytes (f16 scale) + 4 bytes (high bits) + 16 bytes (quants) = 22 bytes
/// GGUF/candle layout:
/// - Positions 0-15: low nibbles + high bits from qh
/// - Positions 16-31: high nibbles + high bits from qh
pub fn dequantize_q5_0(data: &[u8]) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 22;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    // Pre-allocate with correct size for candle layout
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        let d_bytes = &data[block_start..block_start + 2];
        let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));

        let qh = u32::from_le_bytes([
            data[block_start + 2],
            data[block_start + 3],
            data[block_start + 4],
            data[block_start + 5],
        ]);

        let qs = &data[block_start + 6..block_start + 22];

        // Use candle layout:
        // - Low nibbles (byte & 0xF) at positions 0-15
        // - High nibbles (byte >> 4) at positions 16-31
        for (i, &byte) in qs.iter().enumerate() {
            // Low 4 bits + 5th bit go to position i (0-15)
            let low_q = byte & 0x0F;
            let high_bit_low = ((qh >> i) & 1) as u8;
            let q_low = low_q | (high_bit_low << 4);
            #[allow(clippy::cast_possible_wrap)]
            let value_low = q_low as i8 - 16;
            result[out_start + i] = d * f32::from(value_low);

            // High 4 bits + 5th bit go to position i + 16 (16-31)
            let high_q = (byte >> 4) & 0x0F;
            let high_bit_high = ((qh >> (i + 16)) & 1) as u8;
            let q_high = high_q | (high_bit_high << 4);
            #[allow(clippy::cast_possible_wrap)]
            let value_high = q_high as i8 - 16;
            result[out_start + i + 16] = d * f32::from(value_high);
        }
    }

    Ok(result)
}

/// Dequantize `Q5_1` format weights
///
/// Q5_1 format: 2 bytes (f16 scale) + 2 bytes (f16 min) + 4 bytes (high bits) + 16 bytes (quants) = 24 bytes
/// GGUF/candle layout:
/// - Positions 0-15: low nibbles + high bits from qh
/// - Positions 16-31: high nibbles + high bits from qh
pub fn dequantize_q5_1(data: &[u8]) -> Result<Vec<f32>> {
    const BLOCK_BYTES: usize = 24;

    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_1 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    // Pre-allocate with correct size for candle layout
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        let d_bytes = &data[block_start..block_start + 2];
        let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));

        let min_bytes = &data[block_start + 2..block_start + 4];
        let min = f16_to_f32(u16::from_le_bytes([min_bytes[0], min_bytes[1]]));

        let qh = u32::from_le_bytes([
            data[block_start + 4],
            data[block_start + 5],
            data[block_start + 6],
            data[block_start + 7],
        ]);

        let qs = &data[block_start + 8..block_start + 24];

        // Use candle layout:
        // - Low nibbles (byte & 0xF) at positions 0-15
        // - High nibbles (byte >> 4) at positions 16-31
        for (i, &byte) in qs.iter().enumerate() {
            // Low 4 bits + 5th bit go to position i (0-15)
            let low_q = byte & 0x0F;
            let high_bit_low = ((qh >> i) & 1) as u8;
            let q_low = low_q | (high_bit_low << 4);
            result[out_start + i] = d * f32::from(q_low) + min;

            // High 4 bits + 5th bit go to position i + 16 (16-31)
            let high_q = (byte >> 4) & 0x0F;
            let high_bit_high = ((qh >> (i + 16)) & 1) as u8;
            let q_high = high_q | (high_bit_high << 4);
            result[out_start + i + 16] = d * f32::from(q_high) + min;
        }
    }

    Ok(result)
}

/// Dequantize `Q4_K` format weights
pub fn dequantize_q4_k(data: &[u8]) -> Result<Vec<f32>> {
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
    let mut result = vec![0.0f32; num_super_blocks * QK_K];

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        let d = read_f16(&data[sb_start..sb_start + 2]);
        let dmin = read_f16(&data[sb_start + 2..sb_start + 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[sb_start + 4..sb_start + 16]);

        let qs_start = sb_start + 16;
        let qs = &data[qs_start..qs_start + 128];

        let mut ys_index = out_start;

        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;

            let (sc2, m2) = extract_scale_min(&scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;

            for &byte in q {
                result[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
                ys_index += 1;
            }

            for &byte in q {
                result[ys_index] = d2 * (byte >> 4) as f32 - dm2;
                ys_index += 1;
            }
        }
    }

    Ok(result)
}

/// Dequantize `Q5_K` format weights
pub fn dequantize_q5_k(data: &[u8]) -> Result<Vec<f32>> {
    const SUPER_BLOCK_BYTES: usize = 176;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = Vec::with_capacity(num_super_blocks * QK_K);

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        let d = read_f16(&data[sb_start..sb_start + 2]);
        let dmin = read_f16(&data[sb_start + 2..sb_start + 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[sb_start + 4..sb_start + 16]);

        let qh_start = sb_start + 16;
        let qh = &data[qh_start..qh_start + 32];

        let qs_low_start = sb_start + 48;
        let qs = &data[qs_low_start..qs_low_start + 128];

        for block_idx in 0..8 {
            let (scale, min) = extract_scale_min(&scales, block_idx);

            let block_start = block_idx * 16;
            let qh_block_start = block_idx * 4;

            for byte_idx in 0..16 {
                let qs_byte = qs[block_start + byte_idx];

                let high_bits_byte = qh[qh_block_start + byte_idx / 4];
                let bit_offset = (byte_idx % 4) * 2;

                let q_low_4bit = qs_byte & 0x0F;
                let q_low_high_bit = (high_bits_byte >> bit_offset) & 0x01;
                #[allow(clippy::cast_possible_wrap)]
                let q_low = ((q_low_high_bit << 4) | q_low_4bit) as i8;
                let value_low = d * scale * f32::from(q_low) - dmin * min;
                result.push(value_low);

                let q_high_4bit = (qs_byte >> 4) & 0x0F;
                let q_high_high_bit = (high_bits_byte >> (bit_offset + 1)) & 0x01;
                #[allow(clippy::cast_possible_wrap)]
                let q_high = ((q_high_high_bit << 4) | q_high_4bit) as i8;
                let value_high = d * scale * f32::from(q_high) - dmin * min;
                result.push(value_high);
            }
        }
    }

    Ok(result)
}

/// Dequantize `Q6_K` format weights
pub fn dequantize_q6_k(data: &[u8]) -> Result<Vec<f32>> {
    const SUPER_BLOCK_BYTES: usize = 210;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q6_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = vec![0.0f32; num_super_blocks * QK_K];

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        let ql = &data[sb_start..sb_start + 128];
        let qh = &data[sb_start + 128..sb_start + 192];

        let mut scales = [0i8; 16];
        for (i, scale) in scales.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_wrap)]
            {
                *scale = data[sb_start + 192 + i] as i8;
            }
        }

        let d = read_f16(&data[sb_start + 208..sb_start + 210]);

        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            for l in 0..32 {
                let is = l / 16;

                let q1 = ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32;

                result[out_start + n + l] = d * (sc[is] as f32) * (q1 as f32);
                result[out_start + n + l + 32] = d * (sc[is + 2] as f32) * (q2 as f32);
                result[out_start + n + l + 64] = d * (sc[is + 4] as f32) * (q3 as f32);
                result[out_start + n + l + 96] = d * (sc[is + 6] as f32) * (q4 as f32);
            }
        }
    }

    Ok(result)
}

/// Dequantize `Q2_K` format weights
pub fn dequantize_q2_k(data: &[u8]) -> Result<Vec<f32>> {
    const SUPER_BLOCK_BYTES: usize = 84;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q2_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = vec![0.0f32; num_super_blocks * QK_K];

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        let scales_data = &data[sb_start..sb_start + 16];
        let qs = &data[sb_start + 16..sb_start + 80];

        let d = read_f16(&data[sb_start + 80..sb_start + 82]);
        let dmin = read_f16(&data[sb_start + 82..sb_start + 84]);

        for j in 0..16 {
            let sc = (scales_data[j] & 0x0F) as f32;
            let m = (scales_data[j] >> 4) as f32;

            let d_sc = d * sc;
            let dm = dmin * m;

            let qs_offset = j * 4;

            for k in 0..4 {
                let q_byte = qs[qs_offset + k];
                let q0 = (q_byte & 0x03) as f32;
                let q1 = ((q_byte >> 2) & 0x03) as f32;
                let q2 = ((q_byte >> 4) & 0x03) as f32;
                let q3 = ((q_byte >> 6) & 0x03) as f32;

                let base_idx = out_start + j * 16 + k * 4;
                result[base_idx] = d_sc * q0 - dm;
                result[base_idx + 1] = d_sc * q1 - dm;
                result[base_idx + 2] = d_sc * q2 - dm;
                result[base_idx + 3] = d_sc * q3 - dm;
            }
        }
    }

    Ok(result)
}

/// Helper: Read f16 from bytes and convert to f32
#[inline]
pub(crate) fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

// ============================================================================
// Tests for Dequantization Functions (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // f16_to_f32 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_f16_to_f32_zero() {
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        let result = f16_to_f32(0x8000);
        assert!(result == 0.0 || result == -0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00) > 0.0);
    }

    #[test]
    fn test_f16_to_f32_neg_infinity() {
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00) < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0);
        assert!(result < 0.001);
    }

    // -------------------------------------------------------------------------
    // dequantize_q4_0 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q4_0_empty() {
        let result = dequantize_q4_0(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4_0_invalid_length() {
        let result = dequantize_q4_0(&[0; 17]); // Not multiple of 18
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_0_zeros() {
        // One block: 2 (scale) + 16 (quants) = 18 bytes
        let data = vec![0u8; 18];
        let result = dequantize_q4_0(&data).unwrap();
        assert_eq!(result.len(), 32);
        // With scale=0, all values should be 0 (or close due to -8 offset)
    }

    #[test]
    fn test_dequantize_q4_0_one_block() {
        // scale = 1.0 (f16: 0x3C00)
        let mut data = vec![0u8; 18];
        data[0] = 0x00;
        data[1] = 0x3C; // f16 for 1.0
        let result = dequantize_q4_0(&data).unwrap();
        assert_eq!(result.len(), 32);
    }

    // -------------------------------------------------------------------------
    // dequantize_q8_0 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q8_0_empty() {
        let result = dequantize_q8_0(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q8_0_invalid_length() {
        let result = dequantize_q8_0(&[0; 33]); // Not multiple of 34
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_zeros() {
        // One block: 2 (scale) + 32 (quants) = 34 bytes
        let data = vec![0u8; 34];
        let result = dequantize_q8_0(&data).unwrap();
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q8_0_one_block() {
        // scale = 1.0 (f16: 0x3C00)
        let mut data = vec![0u8; 34];
        data[0] = 0x00;
        data[1] = 0x3C; // f16 for 1.0
                        // quants[0] = 1 (as i8)
        data[2] = 1;
        let result = dequantize_q8_0(&data).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 1.0).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // dequantize_f16 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_f16_empty() {
        let result = dequantize_f16(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_f16_invalid_length() {
        let result = dequantize_f16(&[0]); // Not multiple of 2
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_f16_zeros() {
        let data = vec![0u8; 4]; // 2 values
        let result = dequantize_f16(&data).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_f16_one() {
        // f16 for 1.0 = 0x3C00
        let data = vec![0x00, 0x3C];
        let result = dequantize_f16(&data).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 0.0001);
    }

    // -------------------------------------------------------------------------
    // dequantize_q4_1 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q4_1_empty() {
        let result = dequantize_q4_1(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4_1_invalid_length() {
        let result = dequantize_q4_1(&[0; 19]); // Not multiple of 20
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_1_zeros() {
        let data = vec![0u8; 20];
        let result = dequantize_q4_1(&data).unwrap();
        assert_eq!(result.len(), 32);
    }

    // -------------------------------------------------------------------------
    // dequantize_q5_0 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q5_0_empty() {
        let result = dequantize_q5_0(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q5_0_invalid_length() {
        let result = dequantize_q5_0(&[0; 21]); // Not multiple of 22
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_0_zeros() {
        let data = vec![0u8; 22];
        let result = dequantize_q5_0(&data).unwrap();
        assert_eq!(result.len(), 32);
    }

    // -------------------------------------------------------------------------
    // dequantize_q5_1 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q5_1_empty() {
        let result = dequantize_q5_1(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q5_1_invalid_length() {
        let result = dequantize_q5_1(&[0; 23]); // Not multiple of 24
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_1_zeros() {
        let data = vec![0u8; 24];
        let result = dequantize_q5_1(&data).unwrap();
        assert_eq!(result.len(), 32);
    }

    // -------------------------------------------------------------------------
    // dequantize_q4_k Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q4_k_empty() {
        let result = dequantize_q4_k(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4_k_invalid_length() {
        let result = dequantize_q4_k(&[0; 100]); // Not multiple of 144
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_k_zeros() {
        let data = vec![0u8; 144]; // One super-block
        let result = dequantize_q4_k(&data).unwrap();
        assert_eq!(result.len(), QK_K);
    }

    // -------------------------------------------------------------------------
    // dequantize_q5_k Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q5_k_empty() {
        let result = dequantize_q5_k(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q5_k_invalid_length() {
        let result = dequantize_q5_k(&[0; 100]); // Not multiple of 176
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_k_zeros() {
        let data = vec![0u8; 176]; // One super-block
        let result = dequantize_q5_k(&data).unwrap();
        assert_eq!(result.len(), QK_K);
    }

    // -------------------------------------------------------------------------
    // dequantize_q6_k Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q6_k_empty() {
        let result = dequantize_q6_k(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6_k_invalid_length() {
        let result = dequantize_q6_k(&[0; 100]); // Not multiple of 210
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q6_k_zeros() {
        let data = vec![0u8; 210]; // One super-block
        let result = dequantize_q6_k(&data).unwrap();
        assert_eq!(result.len(), QK_K);
    }

    // -------------------------------------------------------------------------
    // dequantize_q2_k Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q2_k_empty() {
        let result = dequantize_q2_k(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q2_k_invalid_length() {
        let result = dequantize_q2_k(&[0; 50]); // Not multiple of 84
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q2_k_zeros() {
        let data = vec![0u8; 84]; // One super-block
        let result = dequantize_q2_k(&data).unwrap();
        assert_eq!(result.len(), QK_K);
    }

    // -------------------------------------------------------------------------
    // read_f16 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_read_f16_zero() {
        let result = read_f16(&[0x00, 0x00]);
        assert!((result - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_read_f16_one() {
        let result = read_f16(&[0x00, 0x3C]); // f16 for 1.0
        assert!((result - 1.0).abs() < 0.0001);
    }
}
