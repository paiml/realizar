//! Phase 39: Parallel K-quantization coverage tests for `parallel_k.rs`
//!
//! Tests cover:
//! - `fused_q4k_tiled_matvec` - L2-aware tiled matmul
//! - `fused_q4k_parallel_matvec` / `_into` - Parallel Q4_K
//! - `fused_q5k_parallel_matvec` / `_into` - Parallel Q5_K
//! - `fused_q6k_parallel_matvec` / `_into` - Parallel Q6_K
//! - `fused_q4k_q8k_parallel_matvec_into` - Q4_K x Q8_K with TCB tiling
//! - `fused_q4k_q8k_ffn_up_gate_into` - Fused FFN up+gate
//! - Backward-compat aliases
//!
//! Focus areas:
//! - Sequential vs parallel paths (PARALLEL_THRESHOLD = 256)
//! - TCB midi-tile (64 rows) and micro-tile (4 rows) chunking
//! - Edge cases: single row, tile boundaries, partial tiles
//! - Error handling: invalid dimensions, buffer sizes

use crate::quantize::parallel_k::{
    fused_q4k_auto_matvec_into, fused_q4k_parallel_matvec, fused_q4k_parallel_matvec_into,
    fused_q4k_q8k_ffn_up_gate_into, fused_q4k_q8k_parallel_matvec_into, fused_q4k_tiled_matvec,
    fused_q5k_parallel_matvec, fused_q5k_parallel_matvec_into, fused_q6k_colmajor_matvec,
    fused_q6k_parallel_matvec, fused_q6k_parallel_matvec_into,
};
use crate::quantize::types::QK_K;

// ============================================================================
// Helper functions for generating valid quantized weight data
// ============================================================================

/// Generates valid Q4_K weight data for testing.
/// Q4_K format: 144 bytes per super-block (256 elements).
fn generate_q4k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144;
    let total_bytes = out_dim * bytes_per_row;

    let mut data = vec![0u8; total_bytes];

    // Fill with deterministic pattern that produces valid Q4_K blocks
    for (i, byte) in data.iter_mut().enumerate() {
        // Create semi-random but deterministic pattern
        *byte = ((i * 17 + 31) % 256) as u8;
    }

    // Set valid scale values (d, dmin) at appropriate offsets in each block
    // Q4_K layout: d (2 bytes), dmin (2 bytes), scales (12 bytes), qs (128 bytes)
    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let block_start = row * bytes_per_row + sb * 144;

            // Set d as small positive value (half-precision)
            // 0x3c00 = 1.0 in FP16
            data[block_start] = 0x00;
            data[block_start + 1] = 0x3c;

            // Set dmin as small positive value
            data[block_start + 2] = 0x00;
            data[block_start + 3] = 0x38; // ~0.5 in FP16
        }
    }

    data
}

/// Generates valid Q5_K weight data for testing.
/// Q5_K format: 176 bytes per super-block (256 elements).
fn generate_q5k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 176;
    let total_bytes = out_dim * bytes_per_row;

    let mut data = vec![0u8; total_bytes];

    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i * 19 + 37) % 256) as u8;
    }

    // Set valid scale values for Q5_K
    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let block_start = row * bytes_per_row + sb * 176;
            data[block_start] = 0x00;
            data[block_start + 1] = 0x3c;
            data[block_start + 2] = 0x00;
            data[block_start + 3] = 0x38;
        }
    }

    data
}

/// Generates valid Q6_K weight data for testing.
/// Q6_K format: 210 bytes per super-block (256 elements).
fn generate_q6k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 210;
    let total_bytes = out_dim * bytes_per_row;

    let mut data = vec![0u8; total_bytes];

    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i * 23 + 41) % 256) as u8;
    }

    // Set valid scale value for Q6_K (d at offset 208)
    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let block_start = row * bytes_per_row + sb * 210;
            // d at offset 208 (2 bytes FP16)
            data[block_start + 208] = 0x00;
            data[block_start + 209] = 0x3c;
        }
    }

    data
}

/// Generates Q8_K quantized activations for testing.
fn generate_q8k_activations(in_dim: usize) -> (Vec<f32>, Vec<i8>) {
    let super_blocks = in_dim.div_ceil(QK_K);
    let scales: Vec<f32> = (0..super_blocks).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let quants: Vec<i8> = (0..in_dim)
        .map(|i| ((i % 256) as i8).wrapping_sub(64))
        .collect();
    (scales, quants)
}

// ============================================================================
// Tests for fused_q4k_tiled_matvec
// ============================================================================

#[test]
fn test_q4k_tiled_matvec_single_row_pk14() {
    let in_dim = 256; // 1 super-block
    let out_dim = 1;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
    // Should produce a finite value
    assert!(output[0].is_finite());
}

#[test]
fn test_q4k_tiled_matvec_multiple_rows_pk14() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 32;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_tiled_matvec_custom_tile_size_pk14() {
    let in_dim = 256;
    let out_dim = 128;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];

    // Custom tile size = 16
    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, Some(16));
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_tiled_matvec_partial_last_tile_pk14() {
    let in_dim = 256;
    let out_dim = 100; // Not a multiple of default tile size (64)
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_tiled_matvec_many_tiles_pk14() {
    let in_dim = 256;
    let out_dim = 256; // 4 tiles of size 64
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.1f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_tiled_matvec_weight_too_small_pk14() {
    let in_dim = 256;
    let out_dim = 10;
    let weights = vec![0u8; 100]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("weight data too small"));
}

#[test]
fn test_q4k_tiled_matvec_activation_mismatch_pk14() {
    let in_dim = 256;
    let out_dim = 10;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; 128]; // Wrong size

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("doesn't match in_dim"));
}

#[test]
fn test_q4k_tiled_matvec_tile_size_1_pk14() {
    let in_dim = 256;
    let out_dim = 8;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    // Extreme: tile_size = 1
    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, Some(1));
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_tiled_matvec_tile_larger_than_out_pk14() {
    let in_dim = 256;
    let out_dim = 10;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    // tile_size > out_dim - should still work (single partial tile)
    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, Some(128));
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

// ============================================================================
// Tests for fused_q4k_parallel_matvec (sequential path: out_dim < 256)
// ============================================================================

#[test]
fn test_q4k_parallel_matvec_sequential_path_pk14() {
    let in_dim = 256;
    let out_dim = 128; // < 256 threshold, uses sequential path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_parallel_matvec_at_threshold_pk14() {
    let in_dim = 256;
    let out_dim = 256; // Exactly at threshold, uses parallel path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_parallel_matvec_parallel_path_pk14() {
    let in_dim = 512;
    let out_dim = 512; // > 256 threshold, uses parallel path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.1f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_parallel_matvec_single_row_pk14() {
    let in_dim = 256;
    let out_dim = 1;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), 1);
}

#[test]
fn test_q4k_parallel_matvec_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_q4k_parallel_matvec_activation_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; 64]; // Wrong size

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q4k_parallel_matvec_into
// ============================================================================

#[test]
fn test_q4k_parallel_matvec_into_sequential_pk14() {
    let in_dim = 256;
    let out_dim = 128; // Sequential path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_parallel_matvec_into_parallel_pk14() {
    let in_dim = 256;
    let out_dim = 512; // Parallel path with midi-tiles
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_parallel_matvec_into_midi_tile_boundary_pk14() {
    let in_dim = 256;
    let out_dim = 64; // Exactly one midi-tile (MIDI_TILE_M = 64)
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    // But out_dim < 256, so this uses sequential path
    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q4k_parallel_matvec_into_partial_midi_tile_pk14() {
    let in_dim = 256;
    let out_dim = 300; // 4 midi-tiles + 44 remainder (300 = 64*4 + 44)
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_parallel_matvec_into_output_buffer_too_small_pk14() {
    let in_dim = 256;
    let out_dim = 128;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; 64]; // Too small

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Output buffer too small"));
}

#[test]
fn test_q4k_parallel_matvec_into_larger_buffer_pk14() {
    let in_dim = 256;
    let out_dim = 32;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![999.0f32; 64]; // Larger than needed

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    // First out_dim elements should be updated
    for val in &output[..out_dim] {
        assert!(val.is_finite());
        assert_ne!(*val, 999.0); // Should have been overwritten
    }

    // Rest should remain unchanged
    for val in &output[out_dim..] {
        assert_eq!(*val, 999.0);
    }
}

// ============================================================================
// Tests for fused_q5k_parallel_matvec
// ============================================================================

#[test]
fn test_q5k_parallel_matvec_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q5k_parallel_matvec_large_pk14() {
    let in_dim = 512;
    let out_dim = 512;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q5k_parallel_matvec_single_row_pk14() {
    let in_dim = 256;
    let out_dim = 1;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_q5k_parallel_matvec_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Q5_K"));
}

#[test]
fn test_q5k_parallel_matvec_activation_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; 128]; // Wrong size

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q5k_parallel_matvec_into
// ============================================================================

#[test]
fn test_q5k_parallel_matvec_into_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q5k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q5k_parallel_matvec_into_large_pk14() {
    let in_dim = 512;
    let out_dim = 256;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q5k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q5k_parallel_matvec_into_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; 32]; // Too small

    let result =
        fused_q5k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q6k_parallel_matvec
// ============================================================================

#[test]
fn test_q6k_parallel_matvec_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q6k_parallel_matvec_large_pk14() {
    let in_dim = 512;
    let out_dim = 512;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q6k_parallel_matvec_single_row_pk14() {
    let in_dim = 256;
    let out_dim = 1;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_q6k_parallel_matvec_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Q6_K"));
}

#[test]
fn test_q6k_parallel_matvec_activation_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; 128]; // Wrong size

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q6k_parallel_matvec_into (with TCB tiling)
// ============================================================================

#[test]
fn test_q6k_parallel_matvec_into_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q6k_parallel_matvec_into_midi_tile_boundary_pk14() {
    let in_dim = 256;
    let out_dim = 128; // 2 midi-tiles
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q6k_parallel_matvec_into_partial_midi_tile_pk14() {
    let in_dim = 256;
    let out_dim = 200; // 3 midi-tiles + 8 remainder
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q6k_parallel_matvec_into_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; 32]; // Too small

    let result =
        fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q4k_q8k_parallel_matvec_into (TCB tiling with micro-tiles)
// ============================================================================

#[test]
fn test_q4k_q8k_parallel_matvec_into_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_micro_tile_boundary_pk14() {
    let in_dim = 256;
    let out_dim = 8; // 2 micro-tiles of 4 rows
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_with_remainder_pk14() {
    let in_dim = 256;
    let out_dim = 70; // 17 micro-tiles + 2 remainder rows
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_single_row_pk14() {
    let in_dim = 256;
    let out_dim = 1; // Less than one micro-tile
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_3_rows_pk14() {
    let in_dim = 256;
    let out_dim = 3; // Less than one micro-tile (remainder only)
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_large_pk14() {
    let in_dim = 512;
    let out_dim = 256; // Multiple midi-tiles
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = vec![0u8; 10]; // Too small
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; 32]; // Too small

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q4k_q8k_ffn_up_gate_into
// ============================================================================

#[test]
fn test_q4k_q8k_ffn_up_gate_into_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());

    for val in &up_output {
        assert!(val.is_finite());
    }
    for val in &gate_output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_large_pk14() {
    let in_dim = 512;
    let out_dim = 256; // Multiple midi-tiles
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_partial_midi_tile_pk14() {
    let in_dim = 256;
    let out_dim = 100; // 1 midi-tile + 36 remainder
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_up_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = vec![0u8; 10]; // Too small
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_gate_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = vec![0u8; 10]; // Too small
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_up_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; 32]; // Too small
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_gate_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; 32]; // Too small

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

// ============================================================================
// Tests for backward-compat aliases
// ============================================================================

#[test]
fn test_fused_q6k_colmajor_matvec_alias_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    // fused_q6k_colmajor_matvec should be an alias for fused_q6k_parallel_matvec
    let result1 = fused_q6k_colmajor_matvec(&weights, &activations, in_dim, out_dim);
    let result2 = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);

    assert!(result1.is_ok());
    assert!(result2.is_ok());

    let out1 = result1.unwrap();
    let out2 = result2.unwrap();

    assert_eq!(out1.len(), out2.len());
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_fused_q4k_auto_matvec_into_alias_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output1 = vec![0.0f32; out_dim];
    let mut output2 = vec![0.0f32; out_dim];

    // fused_q4k_auto_matvec_into should be an alias for fused_q4k_parallel_matvec_into
    let result1 = fused_q4k_auto_matvec_into(&weights, &activations, in_dim, out_dim, &mut output1);
    let result2 =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output2);

    assert!(result1.is_ok());
    assert!(result2.is_ok());

    for (a, b) in output1.iter().zip(output2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

// ============================================================================
// Determinism and consistency tests
// ============================================================================

#[test]
fn test_q4k_parallel_matvec_deterministic_pk14() {
    let in_dim = 256;
    let out_dim = 512; // Parallel path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    // Run multiple times and check determinism
    let result1 = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).unwrap();
    let result2 = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).unwrap();
    let result3 = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).unwrap();

    for i in 0..out_dim {
        assert_eq!(
            result1[i], result2[i],
            "Mismatch at index {} between run 1 and 2",
            i
        );
        assert_eq!(
            result2[i], result3[i],
            "Mismatch at index {} between run 2 and 3",
            i
        );
    }
}

#[test]
fn test_q4k_parallel_vs_tiled_consistency_pk14() {
    let in_dim = 256;
    let out_dim = 128; // Sequential path for parallel version
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let tiled_result =
        fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None).unwrap();
    let parallel_result =
        fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).unwrap();

    // Both should produce the same output
    for i in 0..out_dim {
        let diff = (tiled_result[i] - parallel_result[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: tiled={}, parallel={}, diff={}",
            i,
            tiled_result[i],
            parallel_result[i],
            diff
        );
    }
}

#[test]
fn test_q4k_matvec_vs_matvec_into_consistency_pk14() {
    let in_dim = 256;
    let out_dim = 128;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let alloc_result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).unwrap();

    let mut into_result = vec![0.0f32; out_dim];
    fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut into_result)
        .unwrap();

    for i in 0..out_dim {
        let diff = (alloc_result[i] - into_result[i]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at index {}: alloc={}, into={}, diff={}",
            i,
            alloc_result[i],
            into_result[i],
            diff
        );
    }
}

// ============================================================================
// Edge case tests for dimension handling
// ============================================================================

#[test]
fn test_q4k_multiple_superblocks_per_row_pk14() {
    let in_dim = 768; // 3 super-blocks per row (768 / 256 = 3)
    let out_dim = 32;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.1f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_non_multiple_of_qkk_in_dim_pk14() {
    // in_dim not a multiple of QK_K (256) - uses div_ceil
    let in_dim = 300; // ceil(300/256) = 2 super-blocks
    let out_dim = 16;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_q5k_multiple_superblocks_pk14() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_q6k_multiple_superblocks_pk14() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

// ============================================================================
// Large scale tests (for parallel path coverage)
// ============================================================================

#[test]
fn test_q4k_large_parallel_execution_pk14() {
    let in_dim = 1024; // 4 super-blocks
    let out_dim = 1024; // Well above 256 threshold - uses parallel path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.01f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), out_dim);

    // All outputs should be finite
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_q8k_large_parallel_pk14() {
    let in_dim = 512;
    let out_dim = 512;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_ffn_up_gate_large_parallel_pk14() {
    let in_dim = 512;
    let out_dim = 512;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());

    for val in &up_output {
        assert!(val.is_finite());
    }
    for val in &gate_output {
        assert!(val.is_finite());
    }
}
