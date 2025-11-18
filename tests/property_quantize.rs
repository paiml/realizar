//! Property-based tests for quantization/dequantization
//!
//! These tests use proptest to verify quantization properties.

use proptest::prelude::*;
use realizar::quantize::{dequantize_q4_0, dequantize_q8_0, BLOCK_SIZE};

/// Strategy for generating valid Q4_0 blocks
fn q4_0_block_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q4_0 block: 4 bytes scale + 16 bytes quants = 20 bytes
    (any::<f32>(), prop::collection::vec(any::<u8>(), 16..=16)).prop_map(|(scale, quants)| {
        let mut block = Vec::with_capacity(20);
        block.extend_from_slice(&scale.to_le_bytes());
        block.extend(quants);
        block
    })
}

/// Strategy for generating valid Q8_0 blocks
fn q8_0_block_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q8_0 block: 4 bytes scale + 32 bytes quants = 36 bytes
    (any::<f32>(), prop::collection::vec(any::<u8>(), 32..=32)).prop_map(|(scale, quants)| {
        let mut block = Vec::with_capacity(36);
        block.extend_from_slice(&scale.to_le_bytes());
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
            // Scale: 1.0
            data.extend_from_slice(&1.0f32.to_le_bytes());
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
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q4_0(&data).unwrap();
        for val in result {
            prop_assert!((val - 0.0).abs() < 1e-10);
        }
    }

    /// Q4_0 invalid length always errors
    #[test]
    fn test_q4_0_invalid_length(len in 1usize..100) {
        // Exclude lengths that are valid (multiples of 20)
        if len % 20 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q4_0(&data);
            prop_assert!(result.is_err());
        }
    }

    /// Q4_0 dequantization values are bounded by scale
    #[test]
    fn test_q4_0_values_bounded(
        scale in -100.0f32..100.0f32,
        quants in prop::collection::vec(any::<u8>(), 16..=16)
    ) {
        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q4_0(&data).unwrap();

        // Q4_0 values range from -8 to 7 after offset
        let max_abs = scale.abs() * 8.0;
        for val in result {
            prop_assert!(val.abs() <= max_abs + 1e-6);
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
            // Scale: 1.0
            data.extend_from_slice(&1.0f32.to_le_bytes());
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
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q8_0(&data).unwrap();
        for val in result {
            prop_assert!((val - 0.0).abs() < 1e-10);
        }
    }

    /// Q8_0 invalid length always errors
    #[test]
    fn test_q8_0_invalid_length(len in 1usize..100) {
        // Exclude lengths that are valid (multiples of 36)
        if len % 36 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q8_0(&data);
            prop_assert!(result.is_err());
        }
    }

    /// Q8_0 dequantization values are bounded by scale
    #[test]
    fn test_q8_0_values_bounded(
        scale in -100.0f32..100.0f32,
        quants in prop::collection::vec(any::<u8>(), 32..=32)
    ) {
        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q8_0(&data).unwrap();

        // Q8_0 values range from -128 to 127
        let max_abs = scale.abs() * 128.0;
        for val in result {
            prop_assert!(val.abs() <= max_abs + 1e-6);
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
