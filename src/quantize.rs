//! Quantization and dequantization for model weights
//!
//! Implements quantization formats used by GGUF models:
//! - `Q4_0`: 4-bit quantization (block size 32)
//! - `Q8_0`: 8-bit quantization (block size 32)
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

use crate::error::{RealizarError, Result};

/// Block size for `Q4_0` and `Q8_0` quantization
pub const BLOCK_SIZE: usize = 32;

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
        for i in 0..32 {
            data.push((i as i8).to_le_bytes()[0]);
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
}
