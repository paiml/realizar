//! Parallel and tiled matrix-vector operations for K-quantization (PMAT-802)
//!
//! Implements L2-aware tiled and parallel matvec operations:
//! - `fused_q4k_tiled_matvec` - L2-aware tiled matmul
//! - `fused_q4k_parallel_matvec`, `fused_q4k_parallel_matvec_into` - Parallel Q4_K
//! - `fused_q5k_parallel_matvec`, `fused_q5k_parallel_matvec_into` - Parallel Q5_K
//! - `fused_q6k_parallel_matvec`, `fused_q6k_parallel_matvec_into` - Parallel Q6_K
//!
//! Per Goto & Van Geijn "Anatomy of High-Performance Matrix Multiplication":
//! - GEBP (General Block Panel) tiling maximizes cache reuse
//! - Tile size should fit in L2 cache (~256KB-512KB typically)

use crate::error::{RealizarError, Result};
use super::fused_k::{fused_q4k_dot_simd, fused_q4k_q8k_dot_simd, fused_q4k_q8k_dot_4rows_avx512vnni};
use super::fused_q5k_q6k::{fused_q5k_dot_simd, fused_q6k_dot_simd};
use super::types::QK_K;

// ============================================================================

/// Default tile size for L2-aware tiled matmul
///
/// Chosen to fit in L2 cache while maximizing parallelism:
/// - Typical L2 size: 256KB-512KB
/// - Q4_K row size for hidden_dim=2560: ~1440 bytes
/// - 64 rows = ~92KB of weight data, plus activations
const DEFAULT_OUTPUT_TILE_SIZE: usize = 64;

/// Fused Q4_K matrix-vector multiply with L2-aware tiling
///
/// Processes outputs in tiles to maximize L2 cache reuse.
/// Each tile loads weight data once and computes multiple outputs.
///
/// # Arguments
///
/// * `weight_data` - Raw Q4_K quantized weight data
/// * `activations` - Input activations [in_dim]
/// * `in_dim` - Input dimension (must be multiple of 256 for Q4_K)
/// * `out_dim` - Output dimension
/// * `tile_size` - Number of outputs to process per tile (default: 64)
///
/// # Returns
///
/// Output vector [out_dim]
///
/// # Errors
///
/// Returns error if dimensions don't match weight data
///
/// # Performance
///
/// - **L2-aware**: Tiles fit in L2 cache, reducing DRAM traffic
/// - **Fused**: Dequantize inline with dot product (8x bandwidth reduction)
/// - **SIMD**: Uses AVX2 when available for 4-8x compute speedup
#[allow(clippy::similar_names)]
pub fn fused_q4k_tiled_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    tile_size: Option<usize>,
) -> Result<Vec<f32>> {
    let tile_size = tile_size.unwrap_or(DEFAULT_OUTPUT_TILE_SIZE);

    // Calculate bytes per output row
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144; // Q4_K: 144 bytes per super-block

    // Validate dimensions
    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    let mut output = vec![0.0f32; out_dim];

    // Process outputs in tiles for L2 cache efficiency
    let num_tiles = out_dim.div_ceil(tile_size);

    for tile_idx in 0..num_tiles {
        let tile_start = tile_idx * tile_size;
        let tile_end = (tile_start + tile_size).min(out_dim);

        // Prefetch next tile's weight data (if available)
        #[cfg(target_arch = "x86_64")]
        if tile_idx + 1 < num_tiles {
            let next_tile_start = (tile_idx + 1) * tile_size;
            let next_row_start = next_tile_start * bytes_per_row;
            if next_row_start < weight_data.len() {
                // SAFETY: Prefetch is a hint, no memory safety requirements
                unsafe {
                    use std::arch::x86_64::_mm_prefetch;
                    use std::arch::x86_64::_MM_HINT_T0;
                    let ptr = weight_data.as_ptr().add(next_row_start);
                    _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
                }
            }
        }

        // Process tile: compute dot products for tile_start..tile_end
        for (idx, out_slot) in output[tile_start..tile_end].iter_mut().enumerate() {
            let o = tile_start + idx;
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];

            // Fused dequant + dot product
            *out_slot = fused_q4k_dot_simd(row_data, activations)?;
        }
    }

    Ok(output)
}

// ============================================================================
// PARALLEL TILED MATRIX-VECTOR MULTIPLICATION (Phase 2 + 3)
// ============================================================================
//
// Per Blumofe & Leiserson [6] "Scheduling Multithreaded Computations by Work Stealing":
// - Work-stealing schedulers like rayon maximize CPU utilization
// - Each output row is independent → trivially parallelizable
// - Expected speedup: ~Nx on N-core systems for memory-bound workloads
// ============================================================================

/// Parallel fused Q4_K matrix-vector multiply with L2-aware tiling
///
/// Uses rayon parallel iterators for multi-core acceleration.
/// Per Valiant's BSP model [14], synchronization happens at tile boundaries.
///
/// # Performance
///
/// - **Multi-core**: Linear speedup up to memory bandwidth saturation
/// - **L2-aware**: Tiles fit in L2 cache
/// - **Fused**: 8x memory bandwidth reduction
/// - **SIMD**: AVX2 when available
/// - **Adaptive parallelism**: Sequential for small matrices, parallel for large (IMP-103)
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
#[allow(clippy::similar_names)]
pub fn fused_q4k_parallel_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    // PAR-126: Five-Whys fix - parallel threshold was too high
    // OLD: PARALLEL_THRESHOLD=4096 meant FFN down (out_dim=1536) used sequential path
    // PROBLEM: 1.5B model was 11 tok/s instead of 200 tok/s due to single-threaded matmuls
    //
    // ANALYSIS (for 32-core system with in_dim=8960):
    // - Per-row time: 8960/256 superblocks × ~50ns/superblock = ~1.75µs
    // - Rayon overhead: ~10µs (reduced with work-stealing)
    // - Break-even: 10µs / (1.75µs/32) = ~183 rows
    // SOLUTION: Lower threshold to 256 to enable parallelism for all practical matmuls
    const PARALLEL_THRESHOLD: usize = 256;

    // Calculate bytes per output row
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144; // Q4_K: 144 bytes per super-block

    // Validate dimensions
    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    if out_dim < PARALLEL_THRESHOLD {
        // Sequential path: avoids rayon overhead for small matrices
        let output: Vec<f32> = (0..out_dim)
            .map(|o| {
                let row_start = o * bytes_per_row;
                let row_end = row_start + bytes_per_row;
                let row_data = &weight_data[row_start..row_end];
                fused_q4k_dot_simd(row_data, activations).unwrap_or(0.0)
            })
            .collect();

        Ok(output)
    } else {
        // Parallel path: better for large matrices
        use rayon::prelude::*;

        // Use chunked parallel iteration with optimal chunk size
        // Chunk size tuned for L2 cache (~256KB): process ~64 rows per chunk
        const CHUNK_SIZE: usize = 64;

        let output: Vec<f32> = (0..out_dim)
            .into_par_iter()
            .with_min_len(CHUNK_SIZE)
            .map(|o| {
                let row_start = o * bytes_per_row;
                let row_end = row_start + bytes_per_row;
                let row_data = &weight_data[row_start..row_end];
                fused_q4k_dot_simd(row_data, activations).unwrap_or(0.0)
            })
            .collect();

        Ok(output)
    }
}

/// Parallel fused Q4_K matrix-vector multiply - writes to pre-allocated buffer
///
/// IMP-131: Zero-allocation variant for hot-path inference.
/// This avoids Vec allocation overhead that causes 30-40% performance loss.
///
/// # Arguments
/// * `weight_data` - Raw Q4_K quantized weights [out_dim, in_dim]
/// * `activations` - Input activations [in_dim]
/// * `in_dim` - Input dimension (must match activations length)
/// * `out_dim` - Output dimension (must match output buffer length)
/// * `output` - Pre-allocated output buffer [out_dim]
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
/// - Output buffer length doesn't match out_dim
#[allow(clippy::similar_names)]
pub fn fused_q4k_parallel_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    // PAR-126: Match threshold from allocating version (was 4096, caused 25% perf loss)
    // Analysis: For 32-core system with in_dim=8960:
    // - Per-row time: ~1.75µs, Rayon overhead: ~10µs
    // - Break-even: ~183 rows, so 256 is safe threshold
    const PARALLEL_THRESHOLD: usize = 256;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    if output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Output buffer too small: need {}, have {}",
                out_dim,
                output.len()
            ),
        });
    }

    if out_dim < PARALLEL_THRESHOLD {
        // Sequential path
        for o in 0..out_dim {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];
            output[o] = fused_q4k_dot_simd(row_data, activations).unwrap_or(0.0);
        }
    } else {
        // Parallel path with TCB-style midi-tile chunking
        // Process rows in 64-row chunks to maximize activation cache reuse
        use rayon::prelude::*;
        const MIDI_TILE_M: usize = 64;

        output[..out_dim]
            .par_chunks_mut(MIDI_TILE_M)
            .enumerate()
            .for_each(|(midi_idx, midi_chunk)| {
                let midi_start = midi_idx * MIDI_TILE_M;

                // Process each row in this midi-tile
                // All rows share the same activation vector (kept in L2 cache)
                for (local_idx, out) in midi_chunk.iter_mut().enumerate() {
                    let row = midi_start + local_idx;
                    let row_start = row * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    let row_data = &weight_data[row_start..row_end];
                    *out = fused_q4k_dot_simd(row_data, activations).unwrap_or(0.0);
                }
            });
    }

    Ok(())
}

/// Parallel fused Q5_K matrix-vector multiply
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
#[allow(clippy::similar_names)]
pub fn fused_q5k_parallel_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 176; // Q5_K: 176 bytes per super-block

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_K weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    let output: Vec<f32> = (0..out_dim)
        .into_par_iter()
        .map(|o| {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];

            fused_q5k_dot_simd(row_data, activations).unwrap_or(0.0)
        })
        .collect();

    Ok(output)
}

/// Parallel fused Q5_K matrix-vector multiply - writes to pre-allocated buffer
///
/// IMP-131: Zero-allocation variant for hot-path inference.
#[allow(clippy::similar_names)]
pub fn fused_q5k_parallel_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 176;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_K weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    if output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Output buffer too small: need {}, have {}",
                out_dim,
                output.len()
            ),
        });
    }

    output[..out_dim]
        .par_iter_mut()
        .enumerate()
        .for_each(|(o, out)| {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];
            *out = fused_q5k_dot_simd(row_data, activations).unwrap_or(0.0);
        });

    Ok(())
}

/// Parallel fused Q6_K matrix-vector multiply
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
#[allow(clippy::similar_names)]
pub fn fused_q6k_parallel_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 210; // Q6_K: 210 bytes per super-block

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q6_K weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    let output: Vec<f32> = (0..out_dim)
        .into_par_iter()
        .map(|o| {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];

            fused_q6k_dot_simd(row_data, activations).unwrap_or(0.0)
        })
        .collect();

    Ok(output)
}

/// Parallel fused Q6_K matrix-vector multiply - writes to pre-allocated buffer
///
/// IMP-131: Zero-allocation variant for hot-path inference.
#[allow(clippy::similar_names)]
pub fn fused_q6k_parallel_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 210;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q6_K weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    if output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Output buffer too small: need {}, have {}",
                out_dim,
                output.len()
            ),
        });
    }

    // TCB Tiling: Process rows in midi-tiles (64 rows) to maximize activation cache reuse
    // While Q6K×f32 doesn't have integer Q8K, the f32 activation vector still benefits
    // from being kept in L2 cache while processing multiple output rows.
    const MIDI_TILE_M: usize = 64;

    output[..out_dim]
        .par_chunks_mut(MIDI_TILE_M)
        .enumerate()
        .for_each(|(midi_idx, midi_chunk)| {
            let midi_start = midi_idx * MIDI_TILE_M;

            // Process each row in this midi-tile
            // All rows share the same activation vector (kept in L2 cache)
            for (local_idx, out) in midi_chunk.iter_mut().enumerate() {
                let row = midi_start + local_idx;
                let row_start = row * bytes_per_row;
                let row_end = row_start + bytes_per_row;
                let row_data = &weight_data[row_start..row_end];
                *out = fused_q6k_dot_simd(row_data, activations).unwrap_or(0.0);
            }
        });

    Ok(())
}

// ============================================================================

/// Backwards-compatible alias for `fused_q6k_parallel_matvec`.
///
/// The column-major layout is now the default for the parallel implementation.
#[inline]
pub fn fused_q6k_colmajor_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    fused_q6k_parallel_matvec(weight_data, activations, in_dim, out_dim)
}

/// Backwards-compatible alias for `fused_q4k_parallel_matvec_into`.
///
/// The "auto" naming referred to automatic thread dispatch which is now the default.
#[inline]
pub fn fused_q4k_auto_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    fused_q4k_parallel_matvec_into(weight_data, activations, in_dim, out_dim, output)
}

/// Parallel Q4_K × Q8_K matrix-vector multiply with TCB tiling
///
/// Uses rayon parallel iterators for multi-core acceleration and TCB tiling
/// pattern for cache optimization.
pub fn fused_q4k_q8k_parallel_matvec_into(
    weight_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Weight data too small: need {} bytes, have {}",
                expected_weight_bytes,
                weight_data.len()
            ),
        });
    }

    if output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Output buffer too small: need {}, have {}",
                out_dim,
                output.len()
            ),
        });
    }

    // TCB Tiling Parameters (from trueno::tiling::TilingConfig::cpu_avx512_vnni_q4k_q8k)
    // - Midi-tile: 64 rows (fits in L2 cache with Q8K input)
    // - Micro-tile: 4 rows (processed simultaneously sharing Q8K loads)
    const MIDI_TILE_M: usize = 64;
    const MICRO_TILE_M: usize = 4;

    // Check if we can use the optimized 4-row micro-kernel
    #[cfg(target_arch = "x86_64")]
    let use_4row_kernel =
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni");
    #[cfg(not(target_arch = "x86_64"))]
    let use_4row_kernel = false;

    if use_4row_kernel && out_dim >= MICRO_TILE_M {
        // TCB-style tiled execution with 4-row micro-kernel
        // Process MIDI_TILE_M rows per parallel chunk to maximize Q8K sharing

        // Split output into midi-tiles for rayon parallelism
        output[..out_dim]
            .par_chunks_mut(MIDI_TILE_M)
            .enumerate()
            .for_each(|(midi_idx, midi_chunk)| {
                let midi_start = midi_idx * MIDI_TILE_M;
                let midi_rows = midi_chunk.len();

                // Process micro-tiles (4 rows at a time) within this midi-tile
                let full_micro_tiles = midi_rows / MICRO_TILE_M;
                let remainder = midi_rows % MICRO_TILE_M;

                for micro_idx in 0..full_micro_tiles {
                    let row_base = midi_start + micro_idx * MICRO_TILE_M;

                    // Build row pointers for 4-row kernel
                    let row_ptrs: [*const u8; 4] = [
                        weight_data.as_ptr().wrapping_add(row_base * bytes_per_row),
                        weight_data
                            .as_ptr()
                            .wrapping_add((row_base + 1) * bytes_per_row),
                        weight_data
                            .as_ptr()
                            .wrapping_add((row_base + 2) * bytes_per_row),
                        weight_data
                            .as_ptr()
                            .wrapping_add((row_base + 3) * bytes_per_row),
                    ];

                    // SAFETY: AVX-512 VNNI detected, pointers are within weight_data bounds
                    #[cfg(target_arch = "x86_64")]
                    let outputs = unsafe {
                        fused_q4k_q8k_dot_4rows_avx512vnni(
                            row_ptrs,
                            bytes_per_row,
                            q8k_scales,
                            q8k_quants,
                        )
                    };

                    #[cfg(not(target_arch = "x86_64"))]
                    let outputs = [0.0f32; 4];

                    let local_base = micro_idx * MICRO_TILE_M;
                    midi_chunk[local_base] = outputs[0];
                    midi_chunk[local_base + 1] = outputs[1];
                    midi_chunk[local_base + 2] = outputs[2];
                    midi_chunk[local_base + 3] = outputs[3];
                }

                // Handle remainder rows (< 4) with single-row kernel
                for r in 0..remainder {
                    let row = midi_start + full_micro_tiles * MICRO_TILE_M + r;
                    let row_start = row * bytes_per_row;
                    let row_data = &weight_data[row_start..row_start + bytes_per_row];
                    let local_idx = full_micro_tiles * MICRO_TILE_M + r;
                    midi_chunk[local_idx] =
                        fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0);
                }
            });
    } else {
        // Fallback: per-row execution (no TCB optimization)
        output[..out_dim]
            .par_iter_mut()
            .enumerate()
            .for_each(|(o, out)| {
                let row_start = o * bytes_per_row;
                let row_data = &weight_data[row_start..row_start + bytes_per_row];
                *out = fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0);
            });
    }

    Ok(())
}

/// Fused FFN up+gate projection in single parallel region
///
/// Eliminates rayon::join overhead by processing both up and gate weights
/// in a single par_chunks_mut call. Both projections share the same Q8K
/// quantized input, so we only load it once per midi-tile.
///
/// # Performance
///
/// Reduces parallel region spawns from 2 to 1 per FFN layer, saving ~10-50µs
/// per layer. For 28 layers, this is 280-1400µs per token.
///
/// # Arguments
///
/// * `up_weight` - Q4K weight data for FFN up projection
/// * `gate_weight` - Q4K weight data for FFN gate projection
/// * `q8k_scales` - Pre-quantized activation scales
/// * `q8k_quants` - Pre-quantized activation values
/// * `in_dim` - Input dimension (hidden_dim)
/// * `out_dim` - Output dimension (intermediate_dim)
/// * `up_output` - Output buffer for up projection
/// * `gate_output` - Output buffer for gate projection
#[allow(clippy::too_many_arguments)]
pub fn fused_q4k_q8k_ffn_up_gate_into(
    up_weight: &[u8],
    gate_weight: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    in_dim: usize,
    out_dim: usize,
    up_output: &mut [f32],
    gate_output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;
    const MIDI_TILE_M: usize = 64;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if up_weight.len() < expected_weight_bytes || gate_weight.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Weight data too small: need {} bytes",
                expected_weight_bytes
            ),
        });
    }

    if up_output.len() < out_dim || gate_output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!("Output buffers too small: need {}", out_dim),
        });
    }

    // Process both up and gate in a single parallel region
    // Each thread handles a midi-tile of rows for BOTH projections
    // We use zip + par_chunks_mut to ensure thread-safe non-overlapping access
    up_output[..out_dim]
        .par_chunks_mut(MIDI_TILE_M)
        .zip(gate_output[..out_dim].par_chunks_mut(MIDI_TILE_M))
        .enumerate()
        .for_each(|(midi_idx, (up_chunk, gate_chunk))| {
            let midi_start = midi_idx * MIDI_TILE_M;

            for (local_row, (up_out, gate_out)) in
                up_chunk.iter_mut().zip(gate_chunk.iter_mut()).enumerate()
            {
                let row = midi_start + local_row;
                let row_start = row * bytes_per_row;

                // Compute up projection for this row
                let up_row = &up_weight[row_start..row_start + bytes_per_row];
                *up_out = fused_q4k_q8k_dot_simd(up_row, q8k_scales, q8k_quants).unwrap_or(0.0);

                // Compute gate projection for this row
                let gate_row = &gate_weight[row_start..row_start + bytes_per_row];
                *gate_out = fused_q4k_q8k_dot_simd(gate_row, q8k_scales, q8k_quants).unwrap_or(0.0);
            }
        });

    Ok(())
}
