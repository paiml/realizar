//! Quantization and dequantization for model weights
//!
//! Implements quantization formats used by GGUF models:
//! - `Q4_0`: 4-bit quantization (block size 32)
//! - `Q8_0`: 8-bit quantization (block size 32)
//! - `Q4_K`: 4-bit K-quantization (super-block size 256)
//! - `Q5_K`: 5-bit K-quantization (super-block size 256)
//!
//! ## `Q4_0` Format
//!
//! `Q4_0` stores weights in blocks of 32 values:
//! - 1 float32 scale factor per block
//! - 16 bytes of 4-bit quantized values (2 values per byte)
//! - Dequantization: `value = scale * quantized_value`
//!
//! ## `Q8_0` Format
//!
//! `Q8_0` stores weights in blocks of 32 values:
//! - 1 float32 scale factor per block
//! - 32 int8 quantized values
//! - Dequantization: `value = scale * quantized_value`
//!
//! ## `Q4_K` Format
//!
//! `Q4_K` uses super-blocks of 256 values divided into 8 blocks of 32 values:
//! - 1 half-precision super-block scale (`d`)
//! - 1 half-precision super-block min (`dmin`)
//! - 12 bytes of 6-bit block scales (packed)
//! - 128 bytes of 4-bit quantized values
//! - Dequantization: `value = d * scale * quantized - dmin * min`
//! - Achieves 4.5 bits per weight with better quality than `Q4_0`
//!
//! ## `Q5_K` Format
//!
//! `Q5_K` uses super-blocks of 256 values divided into 8 blocks of 32 values:
//! - 1 half-precision super-block scale (`d`)
//! - 1 half-precision super-block min (`dmin`)
//! - 12 bytes of 6-bit block scales (packed)
//! - 32 bytes of high bits (1 bit per value for 5-bit quantization)
//! - 128 bytes of low 4-bit quantized values
//! - Dequantization: `value = d * scale * quantized - dmin * min`
//! - Achieves 5.5 bits per weight (higher quality than `Q4_K`)

use crate::error::{RealizarError, Result};

/// Block size for `Q4_0` and `Q8_0` quantization
pub const BLOCK_SIZE: usize = 32;

/// Super-block size for K-quantization formats (`Q4_K`, `Q5_K`, `Q6_K`)
pub const QK_K: usize = 256;

/// `Q4_0` quantized block
///
/// Each block contains:
/// - 1 float32 scale factor
/// - 16 bytes (32 4-bit values, 2 per byte)
#[derive(Debug, Clone)]
pub struct Q4_0Block {
    /// Scale factor for dequantization
    pub scale: f32,
    /// Quantized values (16 bytes = 32 4-bit values)
    pub quants: [u8; 16],
}

/// `Q8_0` quantized block
///
/// Each block contains:
/// - 1 float32 scale factor
/// - 32 int8 values
#[derive(Debug, Clone)]
pub struct Q8_0Block {
    /// Scale factor for dequantization
    pub scale: f32,
    /// Quantized values (32 int8 values)
    pub quants: [i8; 32],
}

/// `Q4_K` quantized super-block
///
/// K-quantization uses super-blocks of 256 values (8 blocks of 32 each).
/// Achieves 4.5 bits per weight with better quality than `Q4_0`.
///
/// Each super-block contains:
/// - 1 half-precision scale factor (`d`)
/// - 1 half-precision min factor (`dmin`)
/// - 12 bytes of 6-bit quantized scales (for 8 blocks)
/// - 128 bytes of 4-bit quantized values (256 values)
///
/// Total: 2 + 2 + 12 + 128 = 144 bytes per super-block of 256 values
/// = 4.5 bits per weight
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Q4_KBlock {
    /// Super-block scale factor (f16)
    pub d: f32, // Stored as f16, loaded as f32
    /// Super-block min factor (f16)
    pub dmin: f32, // Stored as f16, loaded as f32
    /// 6-bit quantized scales (12 bytes for 8 blocks)
    pub scales: [u8; 12],
    /// 4-bit quantized values (128 bytes = 256 4-bit values)
    pub qs: [u8; 128],
}

/// `Q5_K` quantized super-block
///
/// K-quantization uses super-blocks of 256 values (8 blocks of 32 each).
/// Achieves 5.5 bits per weight with higher quality than `Q4_K`.
///
/// Each super-block contains:
/// - 1 half-precision scale factor (`d`)
/// - 1 half-precision min factor (`dmin`)
/// - 12 bytes of 6-bit quantized scales (for 8 blocks)
/// - 32 bytes of high bits (1 bit per value for 5-bit quantization)
/// - 128 bytes of low 4-bit quantized values
///
/// Total: 2 + 2 + 12 + 32 + 128 = 176 bytes per super-block of 256 values
/// = 5.5 bits per weight
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Q5_KBlock {
    /// Super-block scale factor (f16)
    pub d: f32, // Stored as f16, loaded as f32
    /// Super-block min factor (f16)
    pub dmin: f32, // Stored as f16, loaded as f32
    /// 6-bit quantized scales (12 bytes for 8 blocks)
    pub scales: [u8; 12],
    /// High bits for 5-bit quantization (32 bytes = 256 bits)
    pub qh: [u8; 32],
    /// Low 4-bit quantized values (128 bytes = 256 4-bit values)
    pub qs: [u8; 128],
}

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
    // Q4_0 block: 4 bytes (f32 scale) + 16 bytes (quants) = 20 bytes
    const BLOCK_BYTES: usize = 4 + 16;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_0 data length {} is not a multiple of block size {}",
                data.len(),
                BLOCK_BYTES
            ),
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    let mut result = Vec::with_capacity(num_blocks * BLOCK_SIZE);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;

        // Read scale (f32)
        let scale_bytes = &data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read quantized values (16 bytes)
        let quants_start = block_start + 4;
        let quants = &data[quants_start..quants_start + 16];

        // Dequantize: 2 4-bit values per byte
        for &byte in quants {
            // Low 4 bits
            // SAFETY: Intentional wrap for 4-bit quantization: u8 [0-15] → i8 [-8,7]
            #[allow(clippy::cast_possible_wrap)]
            let low = (byte & 0x0F) as i8 - 8; // Convert to signed [-8, 7]
            result.push(scale * f32::from(low));

            // High 4 bits
            #[allow(clippy::cast_possible_wrap)]
            let high = ((byte >> 4) & 0x0F) as i8 - 8;
            result.push(scale * f32::from(high));
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
    // Q8_0 block: 4 bytes (f32 scale) + 32 bytes (int8 quants) = 36 bytes
    const BLOCK_BYTES: usize = 4 + 32;

    if data.len() % BLOCK_BYTES != 0 {
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

        // Read scale (f32)
        let scale_bytes = &data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read quantized values (32 int8 values)
        let quants_start = block_start + 4;
        let quants = &data[quants_start..quants_start + 32];

        // Dequantize
        for &byte in quants {
            let value = i8::from_le_bytes([byte]);
            result.push(scale * f32::from(value));
        }
    }

    Ok(result)
}

/// Dequantize `Q4_K` format weights
///
/// # Arguments
///
/// * `data` - Raw `Q4_K` quantized data (super-blocks of 144 bytes each)
///
/// # Returns
///
/// Dequantized float32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of super-block size (144 bytes)
///
/// # Examples
///
/// ```rust,ignore
/// let quantized = load_q4_k_weights();
/// let weights = dequantize_q4_k(&quantized)?;
/// ```
pub fn dequantize_q4_k(data: &[u8]) -> Result<Vec<f32>> {
    // Q4_K super-block: 2 + 2 + 12 + 128 = 144 bytes
    const SUPER_BLOCK_BYTES: usize = 144;

    if data.len() % SUPER_BLOCK_BYTES != 0 {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = Vec::with_capacity(num_super_blocks * QK_K);

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Read d (f16 -> f32)
        let d = read_f16(&data[sb_start..sb_start + 2]);

        // Read dmin (f16 -> f32)
        let dmin = read_f16(&data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[sb_start + 4..sb_start + 16]);

        // Read qs (128 bytes)
        let qs_start = sb_start + 16;
        let qs = &data[qs_start..qs_start + 128];

        // Dequantize 8 blocks of 32 values each
        for block_idx in 0..8 {
            // Extract 6-bit scale and min for this block
            let (scale, min) = extract_scale_min(&scales, block_idx);

            // Process 32 values (16 bytes, 2 4-bit values per byte)
            let block_start = block_idx * 16;
            for byte_idx in 0..16 {
                let byte = qs[block_start + byte_idx];

                // Low 4 bits
                // SAFETY: Intentional for 4-bit quantization: u8 [0-15] → i8 [0,15]
                #[allow(clippy::cast_possible_wrap)]
                let q_low = (byte & 0x0F) as i8;
                let value_low = d * scale * f32::from(q_low) - dmin * min;
                result.push(value_low);

                // High 4 bits
                #[allow(clippy::cast_possible_wrap)]
                let q_high = ((byte >> 4) & 0x0F) as i8;
                let value_high = d * scale * f32::from(q_high) - dmin * min;
                result.push(value_high);
            }
        }
    }

    Ok(result)
}

/// Dequantize `Q5_K` format weights
///
/// # Arguments
///
/// * `data` - Raw `Q5_K` quantized data (super-blocks of 176 bytes each)
///
/// # Returns
///
/// Dequantized float32 values
///
/// # Errors
///
/// Returns error if data length is not a multiple of super-block size (176 bytes)
///
/// # Examples
///
/// ```rust,ignore
/// let quantized = load_q5_k_weights();
/// let weights = dequantize_q5_k(&quantized)?;
/// ```
pub fn dequantize_q5_k(data: &[u8]) -> Result<Vec<f32>> {
    // Q5_K super-block: 2 + 2 + 12 + 32 + 128 = 176 bytes
    const SUPER_BLOCK_BYTES: usize = 176;

    if data.len() % SUPER_BLOCK_BYTES != 0 {
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

        // Read d (f16 -> f32)
        let d = read_f16(&data[sb_start..sb_start + 2]);

        // Read dmin (f16 -> f32)
        let dmin = read_f16(&data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[sb_start + 4..sb_start + 16]);

        // Read qh - high bits (32 bytes)
        let qh_start = sb_start + 16;
        let qh = &data[qh_start..qh_start + 32];

        // Read qs - low 4 bits (128 bytes)
        let qs_low_start = sb_start + 48;
        let qs = &data[qs_low_start..qs_low_start + 128];

        // Dequantize 8 blocks of 32 values each
        for block_idx in 0..8 {
            // Extract 6-bit scale and min for this block
            let (scale, min) = extract_scale_min(&scales, block_idx);

            // Process 32 values (16 bytes of low bits + 4 bytes of high bits)
            let block_start = block_idx * 16;
            let qh_block_start = block_idx * 4; // 32 bits = 4 bytes per block

            for byte_idx in 0..16 {
                let qs_byte = qs[block_start + byte_idx];

                // Get high bits for this pair of values
                let high_bits_byte = qh[qh_block_start + byte_idx / 4];
                let bit_offset = (byte_idx % 4) * 2;

                // Low value (first 4 bits + high bit)
                let q_low_4bit = qs_byte & 0x0F;
                let q_low_high_bit = (high_bits_byte >> bit_offset) & 0x01;
                // SAFETY: Intentional for 5-bit quantization: u8 [0-31] → i8 [0,31]
                #[allow(clippy::cast_possible_wrap)]
                let q_low = ((q_low_high_bit << 4) | q_low_4bit) as i8;
                let value_low = d * scale * f32::from(q_low) - dmin * min;
                result.push(value_low);

                // High value (second 4 bits + high bit)
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

/// Helper: Read f16 from bytes and convert to f32
#[inline]
fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

/// Helper: Extract 6-bit scale and min for a block from the packed scales array
#[inline]
fn extract_scale_min(scales: &[u8; 12], block_idx: usize) -> (f32, f32) {
    // Each block has 6-bit scale and 6-bit min (12 bits total)
    // 8 blocks * 12 bits = 96 bits = 12 bytes
    let bit_offset = block_idx * 12;
    let byte_offset = bit_offset / 8;
    let bit_in_byte = bit_offset % 8;

    // Extract 12 bits across potentially 2-3 bytes
    let bits = if bit_in_byte <= 4 {
        // Fits in 2 bytes
        let b0 = u16::from(scales[byte_offset]);
        let b1 = u16::from(scales[byte_offset + 1]);
        ((b1 << 8) | b0) >> bit_in_byte
    } else {
        // Spans 3 bytes
        let b0 = u32::from(scales[byte_offset]);
        let b1 = u32::from(scales[byte_offset + 1]);
        let b2 = u32::from(scales[byte_offset + 2]);
        // SAFETY: We only extract 12 bits, which fits in u16
        #[allow(clippy::cast_possible_truncation)]
        {
            (((b2 << 16) | (b1 << 8) | b0) >> bit_in_byte) as u16
        }
    };

    // Extract 6-bit scale and 6-bit min
    let scale_bits = (bits & 0x3F) as u8; // Lower 6 bits
    let min_bits = ((bits >> 6) & 0x3F) as u8; // Upper 6 bits

    // Convert 6-bit values to floats (normalize to [0, 1] range)
    let scale = f32::from(scale_bits) / 63.0;
    let min = f32::from(min_bits) / 63.0;

    (scale, min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q4_0_single_block() {
        // Single Q4_0 block: scale=2.0, values=0,1,2,3,...,31 (mapped to -8,-7,-6,...,23)
        let mut data = Vec::new();

        // Scale: 2.0
        data.extend_from_slice(&2.0f32.to_le_bytes());

        // 16 bytes of quantized values (0x01, 0x23, 0x45, ...)
        for i in 0..16 {
            let low = i * 2;
            let high = i * 2 + 1;
            data.push((high << 4) | low);
        }

        let result = dequantize_q4_0(&data).unwrap();
        assert_eq!(result.len(), 32);

        // Check first few values: scale * (value - 8)
        // value 0 -> -8, scale 2.0 -> -16.0
        assert!((result[0] - (-16.0)).abs() < 1e-6);
        // value 1 -> -7, scale 2.0 -> -14.0
        assert!((result[1] - (-14.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_q4_0_invalid_length() {
        // 19 bytes (not a multiple of 20)
        let data = vec![0u8; 19];
        let result = dequantize_q4_0(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_single_block() {
        // Single Q8_0 block: scale=0.5, values=-128,-127,...,127
        let mut data = Vec::new();

        // Scale: 0.5
        data.extend_from_slice(&0.5f32.to_le_bytes());

        // 32 int8 values
        #[allow(clippy::cast_possible_truncation)]
        for i in 0..32_i8 {
            data.push(i.to_le_bytes()[0]);
        }

        let result = dequantize_q8_0(&data).unwrap();
        assert_eq!(result.len(), 32);

        // Check first few values
        assert!((result[0] - 0.0).abs() < 1e-6); // 0 * 0.5 = 0.0
        assert!((result[1] - 0.5).abs() < 1e-6); // 1 * 0.5 = 0.5
        assert!((result[31] - 15.5).abs() < 1e-6); // 31 * 0.5 = 15.5
    }

    #[test]
    fn test_dequantize_q8_0_invalid_length() {
        // 35 bytes (not a multiple of 36)
        let data = vec![0u8; 35];
        let result = dequantize_q8_0(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_0_multiple_blocks() {
        let mut data = Vec::new();

        // Block 1: scale=1.0
        data.extend_from_slice(&1.0f32.to_le_bytes());
        for i in 0..16 {
            data.push((i << 4) | i);
        }

        // Block 2: scale=3.0
        data.extend_from_slice(&3.0f32.to_le_bytes());
        for i in 0..16 {
            data.push((i << 4) | i);
        }

        let result = dequantize_q4_0(&data).unwrap();
        assert_eq!(result.len(), 64); // 2 blocks * 32 values
    }

    #[test]
    fn test_dequantize_q4_k_invalid_length() {
        // 143 bytes (not a multiple of 144)
        let data = vec![0u8; 143];
        let result = dequantize_q4_k(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_k_single_super_block() {
        // Single Q4_K super-block: 144 bytes total
        let mut data = Vec::new();

        // d = 1.0 (f16)
        data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

        // dmin = 0.0 (f16)
        data.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

        // scales: 12 bytes (8 blocks * 12 bits = 96 bits, packed)
        // For simplicity, use simple encoding
        data.extend_from_slice(&[0x00; 12]);

        // qs: 128 bytes (256 4-bit values)
        data.extend_from_slice(&[0x00; 128]);

        let result = dequantize_q4_k(&data).unwrap();
        assert_eq!(result.len(), 256); // 1 super-block * 256 values
    }

    #[test]
    fn test_dequantize_q4_k_output_size() {
        // 2 super-blocks: 2 * 144 = 288 bytes
        let data = vec![0u8; 288];
        let result = dequantize_q4_k(&data).unwrap();
        assert_eq!(result.len(), 512); // 2 super-blocks * 256 values each
    }

    #[test]
    fn test_read_f16() {
        // Test f16 reading
        let f16_1 = half::f16::from_f32(1.0);
        let bytes = f16_1.to_bits().to_le_bytes();
        let result = read_f16(&bytes);
        assert!((result - 1.0).abs() < 1e-3);

        let f16_half = half::f16::from_f32(0.5);
        let bytes = f16_half.to_bits().to_le_bytes();
        let result = read_f16(&bytes);
        assert!((result - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_extract_scale_min() {
        // Test scale/min extraction
        let mut scales = [0u8; 12];

        // Block 0: scale=31 (0x1F), min=0 (first 12 bits = 0x01F)
        scales[0] = 0x1F; // Lower 8 bits of scale
        scales[1] = 0x00; // Upper 4 bits of scale + lower 2 bits of min

        let (scale, min) = extract_scale_min(&scales, 0);
        assert!((scale - 31.0 / 63.0).abs() < 1e-6);
        assert!((min - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_q5_k_invalid_length() {
        // 175 bytes (not a multiple of 176)
        let data = vec![0u8; 175];
        let result = dequantize_q5_k(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_k_single_super_block() {
        // Single Q5_K super-block: 176 bytes total
        let mut data = Vec::new();

        // d = 1.0 (f16)
        data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

        // dmin = 0.0 (f16)
        data.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

        // scales: 12 bytes (8 blocks * 12 bits = 96 bits, packed)
        data.extend_from_slice(&[0x00; 12]);

        // qh: 32 bytes (high bits)
        data.extend_from_slice(&[0x00; 32]);

        // qs: 128 bytes (low 4 bits)
        data.extend_from_slice(&[0x00; 128]);

        let result = dequantize_q5_k(&data).unwrap();
        assert_eq!(result.len(), 256); // 1 super-block * 256 values
    }

    #[test]
    fn test_dequantize_q5_k_output_size() {
        // 2 super-blocks: 2 * 176 = 352 bytes
        let data = vec![0u8; 352];
        let result = dequantize_q5_k(&data).unwrap();
        assert_eq!(result.len(), 512); // 2 super-blocks * 256 values each
    }

    #[test]
    fn test_dequantize_q5_k_with_data() {
        // Test Q5_K with some non-zero data
        let mut data = Vec::new();

        // d = 2.0 (f16)
        data.extend_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());

        // dmin = 0.5 (f16)
        data.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());

        // scales: 12 bytes (set first scale to max)
        let mut scales = [0u8; 12];
        scales[0] = 0x3F; // scale=63 (6 bits)
        data.extend_from_slice(&scales);

        // qh: 32 bytes (all zeros for simplicity)
        data.extend_from_slice(&[0x00; 32]);

        // qs: 128 bytes (all zeros)
        data.extend_from_slice(&[0x00; 128]);

        let result = dequantize_q5_k(&data).unwrap();
        assert_eq!(result.len(), 256);
        // Values should be computed based on formula: d * scale * q - dmin * min
    }
}
