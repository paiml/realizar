//! Property-based tests for quantization/dequantization
//!
//! These tests use proptest to verify quantization properties.

use proptest::prelude::*;
use realizar::quantize::{dequantize_q4_0, dequantize_q8_0, BLOCK_SIZE};

/// Strategy for generating valid Q4_0 blocks
fn q4_0_block_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q4_0 block: 2 bytes scale (f16) + 16 bytes quants = 18 bytes
    (any::<u16>(), prop::collection::vec(any::<u8>(), 16..=16)).prop_map(|(scale_bits, quants)| {
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&scale_bits.to_le_bytes()); // f16 as u16 bits
        block.extend(quants);
        block
    })
}

/// Strategy for generating valid Q8_0 blocks
fn q8_0_block_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q8_0 block: 2 bytes scale (f16) + 32 bytes quants = 34 bytes
    (any::<u16>(), prop::collection::vec(any::<u8>(), 32..=32)).prop_map(|(scale_bits, quants)| {
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&scale_bits.to_le_bytes()); // f16 as u16 bits
        block.extend(quants);
        block
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Q4_0 dequantization produces exactly BLOCK_SIZE values per block
    #[test]
    fn test_q4_0_output_size(block in q4_0_block_strategy()) {
        let result = dequantize_q4_0(&block).unwrap();
        prop_assert_eq!(result.len(), BLOCK_SIZE);
    }

    /// Q4_0 dequantization with multiple blocks
    #[test]
    fn test_q4_0_multiple_blocks(num_blocks in 1usize..10) {
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            // Scale: f16 1.0 = 0x3C00
            data.extend_from_slice(&0x3C00u16.to_le_bytes());
            // 16 bytes of quants
            data.extend_from_slice(&[0u8; 16]);
        }

        let result = dequantize_q4_0(&data).unwrap();
        prop_assert_eq!(result.len(), num_blocks * BLOCK_SIZE);
    }

    /// Q4_0 with zero scale produces all zeros
    #[test]
    fn test_q4_0_zero_scale(quants in prop::collection::vec(any::<u8>(), 16..=16)) {
        let mut data = Vec::new();
        // f16 zero = 0x0000
        data.extend_from_slice(&0x0000u16.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q4_0(&data).unwrap();
        for val in result {
            prop_assert!((val - 0.0).abs() < 1e-10);
        }
    }

    /// Q4_0 invalid length always errors
    #[test]
    fn test_q4_0_invalid_length(len in 1usize..100) {
        // Exclude lengths that are valid (multiples of 18)
        if len % 18 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q4_0(&data);
            prop_assert!(result.is_err());
        }
    }

    /// Q4_0 dequantization values are bounded by scale
    #[test]
    fn test_q4_0_values_bounded(
        // Use a finite f16 range (0x0000-0x7BFF for positive, 0x8000-0xFBFF for negative)
        // Exclude inf (0x7C00) and NaN (0x7C01+)
        scale_bits in prop::num::u16::ANY.prop_filter("finite f16", |&b| {
            let exp = (b >> 10) & 0x1F;
            exp != 0x1F // Filter out inf/nan (exponent == 31)
        }),
        quants in prop::collection::vec(any::<u8>(), 16..=16)
    ) {
        let mut data = Vec::new();
        // f16 scale as u16 bits
        data.extend_from_slice(&scale_bits.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q4_0(&data).unwrap();

        // Convert f16 to f32 for bound check
        let scale = half::f16::from_bits(scale_bits).to_f32();
        // Q4_0 values range from -8 to 7 after offset
        let max_abs = scale.abs() * 8.0;
        for val in result {
            // Allow some tolerance for f16 precision loss
            prop_assert!(val.abs() <= max_abs + 0.1, "val {} > max {}", val, max_abs);
        }
    }

    /// Q8_0 dequantization produces exactly BLOCK_SIZE values per block
    #[test]
    fn test_q8_0_output_size(block in q8_0_block_strategy()) {
        let result = dequantize_q8_0(&block).unwrap();
        prop_assert_eq!(result.len(), BLOCK_SIZE);
    }

    /// Q8_0 dequantization with multiple blocks
    #[test]
    fn test_q8_0_multiple_blocks(num_blocks in 1usize..10) {
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            // Scale: f16 1.0 = 0x3C00
            data.extend_from_slice(&0x3C00u16.to_le_bytes());
            // 32 bytes of quants
            data.extend_from_slice(&[0u8; 32]);
        }

        let result = dequantize_q8_0(&data).unwrap();
        prop_assert_eq!(result.len(), num_blocks * BLOCK_SIZE);
    }

    /// Q8_0 with zero scale produces all zeros
    #[test]
    fn test_q8_0_zero_scale(quants in prop::collection::vec(any::<u8>(), 32..=32)) {
        let mut data = Vec::new();
        // f16 zero = 0x0000
        data.extend_from_slice(&0x0000u16.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q8_0(&data).unwrap();
        for val in result {
            prop_assert!((val - 0.0).abs() < 1e-10);
        }
    }

    /// Q8_0 invalid length always errors
    #[test]
    fn test_q8_0_invalid_length(len in 1usize..100) {
        // Exclude lengths that are valid (multiples of 34)
        if len % 34 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q8_0(&data);
            prop_assert!(result.is_err());
        }
    }

    /// Q8_0 dequantization values are bounded by scale
    #[test]
    fn test_q8_0_values_bounded(
        // Use a finite f16 range (0x0000-0x7BFF for positive, 0x8000-0xFBFF for negative)
        // Exclude inf (0x7C00) and NaN (0x7C01+)
        scale_bits in prop::num::u16::ANY.prop_filter("finite f16", |&b| {
            let exp = (b >> 10) & 0x1F;
            exp != 0x1F // Filter out inf/nan (exponent == 31)
        }),
        quants in prop::collection::vec(any::<u8>(), 32..=32)
    ) {
        let mut data = Vec::new();
        // f16 scale as u16 bits
        data.extend_from_slice(&scale_bits.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q8_0(&data).unwrap();

        // Convert f16 to f32 for bound check
        let scale = half::f16::from_bits(scale_bits).to_f32();
        // Q8_0 values range from -128 to 127
        let max_abs = scale.abs() * 128.0;
        for val in result {
            // Allow some tolerance for f16 precision loss
            prop_assert!(val.abs() <= max_abs + 0.1, "val {} > max {}", val, max_abs);
        }
    }

    /// Empty data returns empty result
    #[test]
    fn test_empty_data(_ in Just(())) {
        let result_q4 = dequantize_q4_0(&[]).unwrap();
        prop_assert!(result_q4.is_empty());

        let result_q8 = dequantize_q8_0(&[]).unwrap();
        prop_assert!(result_q8.is_empty());
    }
}

/// Unit tests for fused RMSNorm + Q8_0 quantization
#[cfg(test)]
mod fused_rmsnorm_tests {
    use realizar::quantize::quantize_rmsnorm_q8_0;

    #[test]
    fn test_fused_rmsnorm_q8_0_basic() {
        // Input: [1.0, 2.0, 3.0, ...] (32 elements to fill one Q8_0 block)
        let input: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        let norm_weight: Vec<f32> = vec![1.0; 32];
        let eps = 1e-6;

        let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

        // Should produce 1 block (32 elements)
        assert_eq!(scales.len(), 1);
        assert_eq!(quants.len(), 32);

        // Verify output makes sense
        // RMSNorm should produce normalized values with unit-ish scale
        assert!(scales[0] > 0.0, "Scale should be positive");
        assert!(scales[0] < 10.0, "Scale should be reasonable");
    }

    #[test]
    fn test_fused_rmsnorm_q8_0_zeros() {
        let input: Vec<f32> = vec![0.0; 32];
        let norm_weight: Vec<f32> = vec![1.0; 32];
        let eps = 1e-6;

        let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

        // All zeros should produce zero quants
        for &q in &quants {
            assert_eq!(q, 0, "Zero input should produce zero quants");
        }
    }

    #[test]
    fn test_fused_rmsnorm_q8_0_constant() {
        // Constant input: all same value
        let input: Vec<f32> = vec![5.0; 32];
        let norm_weight: Vec<f32> = vec![1.0; 32];
        let eps = 1e-6;

        let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

        // Constant input after RMSNorm should be all 1.0 (or close)
        // Then quantized to Q8_0
        assert_eq!(scales.len(), 1);
        assert_eq!(quants.len(), 32);

        // All quants should be the same for constant input
        let first = quants[0];
        for &q in &quants[1..] {
            assert_eq!(q, first, "Constant input should produce constant quants");
        }
    }
}
