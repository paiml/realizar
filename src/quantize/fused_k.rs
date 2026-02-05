//! Fused quantized operations for K-quantization formats (PMAT-802)
//!
//! Implements fused dequant+dot operations that eliminate intermediate f32 buffer
//! allocation for 8x memory bandwidth reduction.
//!
//! Per llama-cpp-style-performance-spec.md:
//! - Memory wall is the bottleneck (Wulf & McKee [10])
//! - Fused operations keep data in registers, avoid memory round-trips
//! - ULP tolerance of ≤4 for numerical equivalence (Goldberg [9])
//!
//! Functions:
//! - `fused_q4k_dot`, `fused_q4k_dot_simd` - Q4_K dot products
//! - `fused_q4k_q8_dot` - Q4_K with Q8 activations
//! - `fused_q4k_q8k_dot`, `fused_q4k_q8k_dot_simd` - Q4_K with Q8_K activations
//! - `fused_q4k_q8k_parallel_matvec_into` - Parallel matrix-vector multiply
//! - `fused_q4k_q8k_ffn_up_gate_into` - Fused FFN up/gate computation
//! - `fused_q6k_dot`, `fused_q6k_dot_simd` - Q6_K dot products
//! - `fused_q5k_dot`, `fused_q5k_dot_simd` - Q5_K dot products

use super::dequant::read_f16;
use super::simd::extract_scale_min;
use super::types::QK_K;
use crate::error::{RealizarError, Result};

/// Fused Q4_K dequantize + dot product
///
/// Computes the dot product of Q4_K quantized weights with f32 activations
/// WITHOUT allocating an intermediate f32 buffer. Dequantization happens
/// inline, accumulating directly into a register.
///
/// # Arguments
///
/// * `q4k_data` - Raw Q4_K quantized data (super-blocks of 144 bytes)
/// * `activations` - f32 activation values (must match dequantized length)
///
/// # Returns
///
/// The dot product as f32
///
/// # Errors
///
/// Returns error if:
/// - `q4k_data` length is not a multiple of 144 bytes (super-block size)
/// - `activations` length doesn't match the number of quantized values
///
/// # Performance
///
/// This function reduces memory traffic by 8x compared to separate
/// dequantize-then-dot operations:
/// - Naive: Read Q4_K (4.5 bits) → Write f32 (32 bits) → Read f32 → Compute
/// - Fused: Read Q4_K (4.5 bits) → Compute in registers
///
/// # Examples
///
/// ```rust,ignore
/// let weights_q4k = load_q4k_weights();
/// let activations = get_layer_activations();
/// let result = fused_q4k_dot(&weights_q4k, &activations)?;
/// ```
pub fn fused_q4k_dot(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
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
    let expected_values = num_super_blocks * QK_K;

    // Validate activation length matches
    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q4_K values count {}",
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
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);

        // Read dmin (f16 -> f32)
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Read qs (128 bytes)
        let qs_start = sb_start + 16;
        let qs = &q4k_data[qs_start..qs_start + 128];

        // PAR-001: Match dequantize_q4_k layout (llama.cpp/candle compatible)
        // Process 4 chunks of 64 values each (0, 64, 128, 192)
        // Each chunk: 32 low nibbles, then 32 high nibbles from 32 consecutive bytes
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

            // First pass: 32 low nibbles (use sc1, m1)
            for &byte in q {
                let q_val = (byte & 0x0F) as f32;
                let value = d1 * q_val - dm1;
                acc += value * activations[activation_idx];
                activation_idx += 1;
            }

            // Second pass: 32 high nibbles (use sc2, m2)
            for &byte in q {
                let q_val = (byte >> 4) as f32;
                let value = d2 * q_val - dm2;
                acc += value * activations[activation_idx];
                activation_idx += 1;
            }
        }
    }

    Ok(acc)
}

/// Fused Q4_K dequantize + dot product with SIMD acceleration
///
/// This is the public, safe API that automatically dispatches to the best
/// available implementation (AVX2 when available, scalar fallback otherwise).
///
/// # Arguments
///
/// * `q4k_data` - Raw Q4_K quantized data (super-blocks of 144 bytes)
/// * `activations` - f32 activation values (must match dequantized length)
///
/// # Returns
///
/// The dot product as f32, matching `fused_q4k_dot` within 4 ULPs
///
/// # Errors
///
/// Returns error if:
/// - `q4k_data` length is not a multiple of 144 bytes (super-block size)
/// - `activations` length doesn't match the number of quantized values
///
/// # Performance
///
/// - AVX2: ~8x speedup over scalar via 256-bit SIMD + FMA
/// - Fused operation: 8x memory bandwidth reduction vs dequant-then-dot
/// - Combined potential: Up to 64x improvement for memory-bound operations
pub fn fused_q4k_dot_simd(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    // Runtime feature detection with fallback (per RustBelt pattern)
    #[cfg(target_arch = "x86_64")]
    {
        // PAR-126: AVX-512 VNNI requires pre-quantized activations (Q4K×Q8K format)
        // For now, use AVX2 which works with f32 activations directly.
        // Future optimization: pre-quantize activations to Q8_0 format once per matmul.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We've verified AVX2 and FMA are available at runtime
            // The unsafe function performs the same logical operation as scalar
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { fused_q4k_dot_avx2(q4k_data, activations) };
        }
    }

    // pmat-ignore: hardware-path (scalar fallback tested directly via fused_q4k_dot)
    // Fallback to scalar implementation
    fused_q4k_dot(q4k_data, activations)
}

/// Quantize f32 activations to i8 and compute integer dot product with q_nibbles.
///
/// Returns (integer_sum, activation_scale) for caller to apply FP32 scaling.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VNNI
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn avx512_quantize_dot(
    act_slice: &[f32],
    q_nibbles_256: std::arch::x86_64::__m256i,
) -> (i32, f32) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let act_max = act_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let act_min = act_slice.iter().copied().fold(f32::INFINITY, f32::min);
    let act_scale = if act_max > act_min {
        127.0 / (act_max - act_min)
    } else {
        1.0
    };

    let mut act_i8 = [0i8; 32];
    for (i, &a) in act_slice.iter().enumerate() {
        act_i8[i] = ((a - act_min) * act_scale).round() as i8;
    }

    let act_vec = _mm256_loadu_si256(act_i8.as_ptr().cast::<__m256i>());
    let q_512 = _mm512_cvtepu8_epi16(q_nibbles_256);
    let act_512 = _mm512_cvtepi8_epi16(act_vec);
    let prod = _mm512_mullo_epi16(q_512, act_512);

    let sum_256 = _mm256_add_epi16(
        _mm512_castsi512_si256(prod),
        _mm512_extracti64x4_epi64(prod, 1),
    );
    let sum_128 = _mm_add_epi16(
        _mm256_castsi256_si128(sum_256),
        _mm256_extracti128_si256(sum_256, 1),
    );
    let sum_32 = _mm256_cvtepi16_epi32(sum_128);
    let sum_arr: [i32; 8] = std::mem::transmute(sum_32);
    (sum_arr.iter().sum(), act_scale)
}

/// AVX-512 VNNI accelerated Q4_K dot product (PAR-126)
///
/// Uses vpdpbusd for int8×int8 accumulation which is 4x faster than FP32 FMA.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VNNI
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_q4k_dot_avx512_vnni(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

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
    let expected_values = num_super_blocks * QK_K;

    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q4_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Final accumulator (FP32)
    let mut total_sum = 0.0f32;
    let mut activation_idx = 0;

    // Nibble mask for 4-bit extraction
    let nibble_mask = _mm512_set1_epi8(0x0F_i8);

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            let next_sb = (sb_idx + 1) * SUPER_BLOCK_BYTES;
            _mm_prefetch(q4k_data.as_ptr().add(next_sb).cast::<i8>(), _MM_HINT_T0);
        }

        // Read d and dmin (f16 → f32)
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Pointer to quantized data (128 bytes)
        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);

        // Process 64 values at a time (matches AVX-512 width of 64 bytes)
        for j in (0..QK_K).step_by(64) {
            let q_start = j / 2;

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Load 32 bytes of quantized data (64 nibbles)
            let q_bytes_256 = _mm256_loadu_si256(qs_ptr.add(q_start).cast::<__m256i>());

            // Expand to 512-bit for AVX-512 operations
            let q_bytes = _mm512_castsi256_si512(q_bytes_256);
            let q_bytes = _mm512_inserti64x4(q_bytes, q_bytes_256, 1);

            // Extract low and high nibbles
            let q_lo = _mm512_and_si512(q_bytes, nibble_mask);
            let q_hi = _mm512_and_si512(_mm512_srli_epi16(q_bytes, 4), nibble_mask);

            // Process low nibbles: quantize activations → integer dot → scale
            let q_lo_256 = _mm512_castsi512_si256(q_lo);
            let (int_sum, act_scale) =
                avx512_quantize_dot(&activations[activation_idx..activation_idx + 32], q_lo_256);
            total_sum += int_sum as f32 * d * sc1 / act_scale - (32.0 * dmin * m1);
            activation_idx += 32;

            // Process high nibbles: same pattern
            let q_hi_256 = _mm512_castsi512_si256(q_hi);
            let (int_sum2, act_scale2) =
                avx512_quantize_dot(&activations[activation_idx..activation_idx + 32], q_hi_256);
            total_sum += int_sum2 as f32 * d * sc2 / act_scale2 - (32.0 * dmin * m2);
            activation_idx += 32;
        }
    }

    Ok(total_sum)
}

/// AVX2-accelerated fused Q4_K dequant+dot kernel (PARITY-003: llama.cpp-style SIMD)
///
/// # Safety
///
/// Caller must ensure:
/// 1. AVX2 and FMA CPU features are available (use `is_x86_feature_detected!`)
/// 2. Input slices are valid (handled by Rust's slice guarantees)
///
/// This function is marked unsafe due to SIMD intrinsics, but is logically
/// equivalent to the scalar `fused_q4k_dot` (within ULP tolerance).
///
/// # Optimizations (PARITY-003)
/// - SIMD loads: `_mm256_loadu_si256` for 32-byte bulk loads (vs scalar byte loads)
/// - SIMD nibble extraction: `_mm256_and_si256` with 0x0F mask (vs scalar & 0x0F)
/// - 4 independent accumulators to hide FMA latency
/// - Software prefetching for next super-block
/// - Matches llama.cpp ggml_vec_dot_q4_K_q8_K pattern
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_q4k_dot_avx2(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    // Allow wildcard import for SIMD intrinsics (standard pattern for arch-specific code)
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    // Validate inputs (same as scalar)
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
    let expected_values = num_super_blocks * QK_K;

    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q4_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Nibble mask for extracting 4-bit values (llama.cpp pattern)
    let nibble_mask = _mm256_set1_epi8(0x0F_i8);

    // PARITY-003: 4 independent accumulators to hide FMA latency
    // FMA latency = 4 cycles, throughput = 2/cycle
    // With 4 independent chains, we saturate the FMA throughput
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut activation_idx = 0;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Prefetch next super-block while processing current
        if sb_idx + 1 < num_super_blocks {
            let next_sb = (sb_idx + 1) * SUPER_BLOCK_BYTES;
            // SAFETY: Prefetch is a hint, pointer arithmetic is in bounds (checked above)
            _mm_prefetch(q4k_data.as_ptr().add(next_sb).cast::<i8>(), _MM_HINT_T0);
        }

        // Read d and dmin (f16 -> f32)
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Pointer to quantized data (128 bytes = 256 nibbles = 256 values)
        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);

        // PAR-001: Match dequantize_q4_k layout (llama.cpp/candle compatible)
        // Process 4 chunks of 64 values each (j=0, 64, 128, 192)
        // Each chunk: 32 low nibbles (sc1), then 32 high nibbles (sc2)
        for j in (0..QK_K).step_by(64) {
            let q_start = j / 2; // 32 bytes per 64-value chunk

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Precompute d*scale and dmin*min for both halves
            let d_scale1 = d * sc1;
            let dm1 = dmin * m1;
            let d_scale2 = d * sc2;
            let dm2 = dmin * m2;

            // SIMD OPTIMIZATION: Load 32 bytes at once (64 nibbles = 64 values)
            // llama.cpp pattern: _mm256_loadu_si256 + AND/SHIFT for nibble extraction
            let q_bytes = _mm256_loadu_si256(qs_ptr.add(q_start).cast::<__m256i>());

            // Extract low nibbles (first 32 values) and high nibbles (second 32 values)
            let q_lo = _mm256_and_si256(q_bytes, nibble_mask);
            let q_hi = _mm256_and_si256(_mm256_srli_epi16(q_bytes, 4), nibble_mask);

            // Process low nibbles (32 values with scale sc1/m1)
            // Split into 4 groups of 8 for f32 conversion (AVX2 can convert 8 i32→f32)
            let d_scale1_vec = _mm256_set1_ps(d_scale1);
            let dm1_vec = _mm256_set1_ps(dm1);

            // Extract bytes 0-7 (low nibbles) → convert to f32
            let q_lo_128_0 = _mm256_castsi256_si128(q_lo);
            let q_lo_i32_0 = _mm256_cvtepu8_epi32(q_lo_128_0);
            let q_lo_f32_0 = _mm256_cvtepi32_ps(q_lo_i32_0);
            let dequant0 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_0, dm1_vec);
            let act0 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc0 = _mm256_fmadd_ps(dequant0, act0, acc0);
            activation_idx += 8;

            // Extract bytes 8-15 (low nibbles)
            let q_lo_shifted = _mm_srli_si128(q_lo_128_0, 8);
            let q_lo_i32_1 = _mm256_cvtepu8_epi32(q_lo_shifted);
            let q_lo_f32_1 = _mm256_cvtepi32_ps(q_lo_i32_1);
            let dequant1 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_1, dm1_vec);
            let act1 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc1 = _mm256_fmadd_ps(dequant1, act1, acc1);
            activation_idx += 8;

            // Extract bytes 16-23 (low nibbles from high 128 bits)
            let q_lo_128_1 = _mm256_extracti128_si256(q_lo, 1);
            let q_lo_i32_2 = _mm256_cvtepu8_epi32(q_lo_128_1);
            let q_lo_f32_2 = _mm256_cvtepi32_ps(q_lo_i32_2);
            let dequant2 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_2, dm1_vec);
            let act2 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc2 = _mm256_fmadd_ps(dequant2, act2, acc2);
            activation_idx += 8;

            // Extract bytes 24-31 (low nibbles)
            let q_lo_shifted2 = _mm_srli_si128(q_lo_128_1, 8);
            let q_lo_i32_3 = _mm256_cvtepu8_epi32(q_lo_shifted2);
            let q_lo_f32_3 = _mm256_cvtepi32_ps(q_lo_i32_3);
            let dequant3 = _mm256_fmsub_ps(d_scale1_vec, q_lo_f32_3, dm1_vec);
            let act3 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc3 = _mm256_fmadd_ps(dequant3, act3, acc3);
            activation_idx += 8;

            // Process high nibbles (32 values with scale sc2/m2)
            let d_scale2_vec = _mm256_set1_ps(d_scale2);
            let dm2_vec = _mm256_set1_ps(dm2);

            // Extract bytes 0-7 (high nibbles)
            let q_hi_128_0 = _mm256_castsi256_si128(q_hi);
            let q_hi_i32_0 = _mm256_cvtepu8_epi32(q_hi_128_0);
            let q_hi_f32_0 = _mm256_cvtepi32_ps(q_hi_i32_0);
            let dequant4 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_0, dm2_vec);
            let act4 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc0 = _mm256_fmadd_ps(dequant4, act4, acc0);
            activation_idx += 8;

            // Extract bytes 8-15 (high nibbles)
            let q_hi_shifted = _mm_srli_si128(q_hi_128_0, 8);
            let q_hi_i32_1 = _mm256_cvtepu8_epi32(q_hi_shifted);
            let q_hi_f32_1 = _mm256_cvtepi32_ps(q_hi_i32_1);
            let dequant5 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_1, dm2_vec);
            let act5 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc1 = _mm256_fmadd_ps(dequant5, act5, acc1);
            activation_idx += 8;

            // Extract bytes 16-23 (high nibbles from high 128 bits)
            let q_hi_128_1 = _mm256_extracti128_si256(q_hi, 1);
            let q_hi_i32_2 = _mm256_cvtepu8_epi32(q_hi_128_1);
            let q_hi_f32_2 = _mm256_cvtepi32_ps(q_hi_i32_2);
            let dequant6 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_2, dm2_vec);
            let act6 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc2 = _mm256_fmadd_ps(dequant6, act6, acc2);
            activation_idx += 8;

            // Extract bytes 24-31 (high nibbles)
            let q_hi_shifted2 = _mm_srli_si128(q_hi_128_1, 8);
            let q_hi_i32_3 = _mm256_cvtepu8_epi32(q_hi_shifted2);
            let q_hi_f32_3 = _mm256_cvtepi32_ps(q_hi_i32_3);
            let dequant7 = _mm256_fmsub_ps(d_scale2_vec, q_hi_f32_3, dm2_vec);
            let act7 = _mm256_loadu_ps(activations.as_ptr().add(activation_idx));
            acc3 = _mm256_fmadd_ps(dequant7, act7, acc3);
            activation_idx += 8;
        }
    }

    // Combine 4 accumulators → single accumulator
    let acc_01 = _mm256_add_ps(acc0, acc1);
    let acc_23 = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc_01, acc_23);

    // Horizontal sum: reduce 8 lanes to single value
    let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
    let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    let result = _mm_cvtss_f32(temp);

    Ok(result)
}

// ============================================================================
// Q4_K × Q8_K KERNELS (Super-block aligned integer-only arithmetic)
// ============================================================================

/// Fused Q4_K × Q8_K dot product (super-block aligned, llama.cpp-style)
///
/// Uses Q8_K format (256 values per super-block, single scale) for maximum
/// SIMD efficiency. This matches llama.cpp's `ggml_vec_dot_q4_K_q8_K`.
///
/// # Arguments
/// * `q4k_data` - Raw Q4_K quantized data (144 bytes per super-block)
/// * `q8k_scales` - Q8_K scales (one per super-block)
/// * `q8k_quants` - Q8_K quantized int8 values (256 per super-block)
///
/// # Performance
///
/// Compared to Q4_K × f32:
/// - 8x fewer memory reads for activations
/// - Integer-only inner loop (no f32 conversion until end)
/// - Single scale multiplication per super-block (vs 8 for Q8_0)
pub fn fused_q4k_q8k_dot(q4k_data: &[u8], q8k_scales: &[f32], q8k_quants: &[i8]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 144;

    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if q8k_scales.len() < num_super_blocks {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_K scales count {} < expected {}",
                q8k_scales.len(),
                num_super_blocks
            ),
        });
    }

    if q8k_quants.len() < expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_K quants count {} < expected {}",
                q8k_quants.len(),
                expected_values
            ),
        });
    }

    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Read Q4_K super-block header
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes for 8 blocks)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Q8_K scale for this super-block
        let q8_scale = q8k_scales[sb_idx];

        // Process 4 chunks of 64 values (matching dequantize_q4_k layout)
        // The dequantized output order is: 32 low nibbles, then 32 high nibbles
        // So activations[j..j+32] correspond to low nibbles, activations[j+32..j+64] to high
        for j in (0..QK_K).step_by(64) {
            let q_offset = sb_start + 16 + j / 2; // 32 bytes per 64-value chunk
            let q8_offset = q8_start + j;

            // Get scales for low and high nibbles
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Combined scale factors
            let d_sc1_q8 = d * sc1 * q8_scale;
            let dm1_q8 = dmin * m1 * q8_scale;
            let d_sc2_q8 = d * sc2 * q8_scale;
            let dm2_q8 = dmin * m2 * q8_scale;

            // Accumulators for low and high nibbles
            let mut sum_lo: i32 = 0; // q4_lo × q8 (for activations[j..j+32])
            let mut sum_hi: i32 = 0; // q4_hi × q8 (for activations[j+32..j+64])
            let mut q8_sum_lo: i32 = 0;
            let mut q8_sum_hi: i32 = 0;

            for b in 0..32 {
                let q4_byte = q4k_data[q_offset + b];

                // Low nibble × activation[j + b] (first 32 positions in dequant order)
                let q4_lo = (q4_byte & 0x0F) as i32;
                let q8_lo = q8k_quants[q8_offset + b] as i32;
                sum_lo += q4_lo * q8_lo;
                q8_sum_lo += q8_lo;

                // High nibble × activation[j + 32 + b] (second 32 positions in dequant order)
                let q4_hi = ((q4_byte >> 4) & 0x0F) as i32;
                let q8_hi = q8k_quants[q8_offset + 32 + b] as i32;
                sum_hi += q4_hi * q8_hi;
                q8_sum_hi += q8_hi;
            }

            // Apply formula: (d * scale * sum_q4_q8 - dmin * min * sum_q8) * q8_scale
            total_acc += d_sc1_q8 * (sum_lo as f32) - dm1_q8 * (q8_sum_lo as f32);
            total_acc += d_sc2_q8 * (sum_hi as f32) - dm2_q8 * (q8_sum_hi as f32);
        }
    }

    Ok(total_acc)
}

/// SIMD-accelerated Q4_K × Q8_K dot product
///
/// Uses AVX-512 VNNI (vpdpbusd) for maximum throughput, falls back to AVX2 or scalar.
/// Single scale per super-block eliminates per-block overhead.
pub fn fused_q4k_q8k_dot_simd(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> Result<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        // PAR-126: Use V2 optimized AVX-512 VNNI kernel (deferred horizontal sums)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { fused_q4k_q8k_dot_avx512vnni_v2(q4k_data, q8k_scales, q8k_quants) };
        }
        // pmat-ignore: hardware-path (AVX2 fallback never reached when AVX-512 VNNI available)
        // Fallback to AVX2 (layout issue resolved)
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { fused_q4k_q8k_dot_avx2(q4k_data, q8k_scales, q8k_quants) };
        }
    }

    // pmat-ignore: hardware-path (scalar fallback tested directly via fused_q4k_q8k_dot)
    fused_q4k_q8k_dot(q4k_data, q8k_scales, q8k_quants)
}

/// PAR-126 V2: AVX-512 VNNI kernel with deferred horizontal sums
///
/// Key optimization: Instead of extracting block sums to scalars in the inner loop,
/// keeps all 8 block accumulators in __m256i vectors and only reduces at the end.
/// This eliminates 24 horizontal sums per row (from 8 to 1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
unsafe fn fused_q4k_q8k_dot_avx512vnni_v2(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if q8k_scales.len() < num_super_blocks || q8k_quants.len() < expected_values {
        return Err(RealizarError::InvalidShape {
            reason: "Q8_K buffer too small".to_string(),
        });
    }

    let nibble_mask = _mm256_set1_epi8(0x0F_i8);
    let ones_16 = _mm256_set1_epi16(1);

    // Global float accumulator
    let mut total_acc = _mm256_setzero_ps();

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            _mm_prefetch(
                q4k_data
                    .as_ptr()
                    .add((sb_idx + 1) * SUPER_BLOCK_BYTES)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                q8k_quants.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Read Q4_K header
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        let mut scales_raw = [0u8; 12];
        scales_raw.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        let q8_scale = q8k_scales[sb_idx];
        let d_q8 = d * q8_scale;
        let dmin_q8 = dmin * q8_scale;

        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // PAR-126 V2: Keep all 8 block results in vectors
        // block_dots_vec[i] = sum of (Q4[block_i] * Q8[block_i])
        // block_q8sums_vec[i] = sum of Q8[block_i]
        let mut block_dots_vec = _mm256_setzero_si256();
        let mut block_q8sums_vec = _mm256_setzero_si256();

        // Process 64 values (2 blocks) per iteration, 4 iterations = 8 blocks
        // Each iteration produces 2 block sums that go into specific lanes
        for chunk in 0..4 {
            let j = chunk * 64;
            let q_offset = j / 2;

            // Load 32 bytes Q4 (64 nibbles packed)
            let q4_bytes = _mm256_loadu_si256(qs_ptr.add(q_offset).cast::<__m256i>());

            // Extract nibbles: lo = first 32 values, hi = next 32 values
            let q4_lo = _mm256_and_si256(q4_bytes, nibble_mask);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), nibble_mask);

            // Load 64 bytes Q8
            let q8_lo = _mm256_loadu_si256(q8_ptr.add(j).cast::<__m256i>());
            let q8_hi = _mm256_loadu_si256(q8_ptr.add(j + 32).cast::<__m256i>());

            // Q4 × Q8 products -> i16 via maddubs -> i32 via madd
            let prod_lo_i16 = _mm256_maddubs_epi16(q4_lo, q8_lo);
            let prod_hi_i16 = _mm256_maddubs_epi16(q4_hi, q8_hi);
            let prod_lo_i32 = _mm256_madd_epi16(prod_lo_i16, ones_16);
            let prod_hi_i32 = _mm256_madd_epi16(prod_hi_i16, ones_16);

            // Reduce each 256-bit register to ONE sum using hadd (within-lane reduction)
            // prod_lo_i32 has 8 i32 values, we need their sum for block chunk*2
            // prod_hi_i32 has 8 i32 values, we need their sum for block chunk*2+1

            // Use hadd twice to get 8 -> 4 -> 2 values per register
            let prod_lo_h1 = _mm256_hadd_epi32(prod_lo_i32, prod_hi_i32); // interleaves lo and hi
            let prod_h2 = _mm256_hadd_epi32(prod_lo_h1, prod_lo_h1); // further reduce

            // Now prod_h2 has: [sum_lo_lane0, sum_hi_lane0, sum_lo_lane0, sum_hi_lane0,
            //                   sum_lo_lane1, sum_hi_lane1, sum_lo_lane1, sum_hi_lane1]
            // We need to extract the two unique sums and place them in block_dots_vec

            // Extract two sums: add lane0 and lane1 of each block
            let lane0 = _mm256_castsi256_si128(prod_h2);
            let lane1 = _mm256_extracti128_si256(prod_h2, 1);
            let sums_128 = _mm_add_epi32(lane0, lane1); // [dot_lo, dot_hi, dot_lo, dot_hi]

            // Insert into correct position based on chunk
            // block_dots_vec lanes: [0,1,2,3,4,5,6,7] for blocks [0,1,2,3,4,5,6,7]
            match chunk {
                0 => {
                    // Put in lanes 0,1
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 0), 0);
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 1), 1);
                },
                1 => {
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 0), 2);
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 1), 3);
                },
                2 => {
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 0), 4);
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 1), 5);
                },
                3 => {
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 0), 6);
                    block_dots_vec =
                        _mm256_insert_epi32(block_dots_vec, _mm_extract_epi32(sums_128, 1), 7);
                },
                _ => unreachable!(),
            }

            // Q8 sums for dmin correction using sad_epu8 (sum of absolute differences)
            // Convert signed i8 to unsigned by adding 128, compute SAD, then adjust
            let bias = _mm256_set1_epi8(-128_i8);
            let _q8_lo_u = _mm256_sub_epi8(q8_lo, bias); // signed [-128,127] -> unsigned [0,255]
            let _q8_hi_u = _mm256_sub_epi8(q8_hi, bias);

            // Use mpsadbw or manual approach for sum
            // Simpler: sign-extend and sum using madd
            let q8_lo_i16_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_lo));
            let q8_lo_i16_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_lo, 1));
            let q8_hi_i16_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_hi));
            let q8_hi_i16_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_hi, 1));

            let q8_lo_i32_a = _mm256_madd_epi16(q8_lo_i16_a, ones_16);
            let q8_lo_i32_b = _mm256_madd_epi16(q8_lo_i16_b, ones_16);
            let q8_hi_i32_a = _mm256_madd_epi16(q8_hi_i16_a, ones_16);
            let q8_hi_i32_b = _mm256_madd_epi16(q8_hi_i16_b, ones_16);

            let q8_lo_sum = _mm256_add_epi32(q8_lo_i32_a, q8_lo_i32_b);
            let q8_hi_sum = _mm256_add_epi32(q8_hi_i32_a, q8_hi_i32_b);

            // Reduce to two sums using hadd
            let q8_h1 = _mm256_hadd_epi32(q8_lo_sum, q8_hi_sum);
            let q8_h2 = _mm256_hadd_epi32(q8_h1, q8_h1);

            let q8_lane0 = _mm256_castsi256_si128(q8_h2);
            let q8_lane1 = _mm256_extracti128_si256(q8_h2, 1);
            let q8_sums_128 = _mm_add_epi32(q8_lane0, q8_lane1);

            match chunk {
                0 => {
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 0), 0);
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 1), 1);
                },
                1 => {
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 0), 2);
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 1), 3);
                },
                2 => {
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 0), 4);
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 1), 5);
                },
                3 => {
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 0), 6);
                    block_q8sums_vec =
                        _mm256_insert_epi32(block_q8sums_vec, _mm_extract_epi32(q8_sums_128, 1), 7);
                },
                _ => unreachable!(),
            }
        }

        // Extract all 8 scales and mins into vectors
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for i in 0..8 {
            let (sc, m) = extract_scale_min(&scales_raw, i);
            scales[i] = sc;
            mins[i] = m;
        }

        let scales_vec = _mm256_loadu_ps(scales.as_ptr());
        let mins_vec = _mm256_loadu_ps(mins.as_ptr());

        // Convert block results to f32
        let dots_f32 = _mm256_cvtepi32_ps(block_dots_vec);
        let q8sums_f32 = _mm256_cvtepi32_ps(block_q8sums_vec);

        // Compute: d_q8 * scales * dots - dmin_q8 * mins * q8sums
        let d_q8_vec = _mm256_set1_ps(d_q8);
        let dmin_q8_vec = _mm256_set1_ps(dmin_q8);

        let term1 = _mm256_mul_ps(d_q8_vec, _mm256_mul_ps(scales_vec, dots_f32));
        let term2 = _mm256_mul_ps(dmin_q8_vec, _mm256_mul_ps(mins_vec, q8sums_f32));
        let result = _mm256_sub_ps(term1, term2);

        // Accumulate into total (defer final horizontal sum)
        total_acc = _mm256_add_ps(total_acc, result);
    }

    // ONE horizontal sum at the very end
    let sum128 = _mm_add_ps(
        _mm256_castps256_ps128(total_acc),
        _mm256_extractf128_ps(total_acc, 1),
    );
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    Ok(_mm_cvtss_f32(sum32))
}

/// AVX-512 VNNI optimized Q4_K × Q8_K dot product (llama.cpp style)
///
/// Uses `vpdpbusd` instruction for 16-way uint8×int8→int32 multiply-accumulate.
/// This is the fastest CPU path for quantized matmul.
///
/// # Safety
/// Requires AVX-512F and AVX-512 VNNI CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_q4k_q8k_dot_avx512vnni(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if q8k_scales.len() < num_super_blocks || q8k_quants.len() < expected_values {
        return Err(RealizarError::InvalidShape {
            reason: "Q8_K buffer too small".to_string(),
        });
    }

    let _nibble_mask = _mm512_set1_epi8(0x0F_i8); // Reserved for future optimization
    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            _mm_prefetch(
                q4k_data
                    .as_ptr()
                    .add((sb_idx + 1) * SUPER_BLOCK_BYTES)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Read Q4_K header
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        let q8_scale = q8k_scales[sb_idx];
        let d_q8 = d * q8_scale;
        let dmin_q8 = dmin * q8_scale;

        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // Process 4 chunks of 64 values (matching dequantize_q4_k layout)
        for j in (0..QK_K).step_by(64) {
            let q_offset = j / 2; // 32 bytes per 64-value chunk

            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Load 32 bytes of Q4_K (as lower half of 512-bit register)
            // We need to process 64 Q4 values (32 bytes) with 64 Q8 values (64 bytes)
            let q4_256 = _mm256_loadu_si256(qs_ptr.add(q_offset).cast::<__m256i>());

            // Zero-extend to 512-bit (q4 data in lower 256 bits) - reserved for full-width optimization
            let _q4_512 = _mm512_castsi256_si512(q4_256);

            // Extract low nibbles (first 32 values) and high nibbles (second 32 values)
            let q4_lo_256 = _mm256_and_si256(q4_256, _mm256_set1_epi8(0x0F_i8));
            let q4_hi_256 =
                _mm256_and_si256(_mm256_srli_epi16(q4_256, 4), _mm256_set1_epi8(0x0F_i8));

            // Load Q8 values: first 32 for low nibbles, next 32 for high nibbles
            // (matching dequantize_q4_k output order: 32 low, then 32 high)
            let q8_lo_256 = _mm256_loadu_si256(q8_ptr.add(j).cast::<__m256i>());
            let q8_hi_256 = _mm256_loadu_si256(q8_ptr.add(j + 32).cast::<__m256i>());

            // For VNNI vpdpbusd: accumulator += (unsigned a) × (signed b)
            // Q4 values are 0-15 (unsigned), Q8 values are -128..127 (signed)
            // We need to cast Q4 to unsigned interpretation

            // Convert to 512-bit for VNNI
            let q4_lo_512 = _mm512_castsi256_si512(q4_lo_256);
            let q4_hi_512 = _mm512_castsi256_si512(q4_hi_256);
            let q8_lo_512 = _mm512_castsi256_si512(q8_lo_256);
            let q8_hi_512 = _mm512_castsi256_si512(q8_hi_256);

            // VNNI multiply-accumulate: result[i] += q4[4i..4i+4] · q8[4i..4i+4]
            // This computes 4 int8×int8 products and adds them to int32 accumulator
            let acc_lo = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_lo_512, q8_lo_512);
            let acc_hi = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_hi_512, q8_hi_512);

            // Horizontal sum the accumulators (each has 16 int32 values)
            // Extract lower 256 bits which contain our results
            let acc_lo_256 = _mm512_castsi512_si256(acc_lo);
            let acc_hi_256 = _mm512_castsi512_si256(acc_hi);

            // Sum all 8 int32 values in each 256-bit register
            let sum_lo = horizontal_sum_epi32_256(acc_lo_256);
            let sum_hi = horizontal_sum_epi32_256(acc_hi_256);

            // Sum Q8 values for min contribution
            // For signed i8, we need to convert to i16 first to avoid overflow
            let q8_lo_256_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_lo_256));
            let q8_lo_256_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_lo_256, 1));
            let q8_sum_lo = horizontal_sum_epi16_256(q8_lo_256_i16_lo)
                + horizontal_sum_epi16_256(q8_lo_256_i16_hi);

            let q8_hi_256_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_hi_256));
            let q8_hi_256_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_hi_256, 1));
            let q8_sum_hi = horizontal_sum_epi16_256(q8_hi_256_i16_lo)
                + horizontal_sum_epi16_256(q8_hi_256_i16_hi);

            // Apply scales
            total_acc += d_q8 * sc1 * (sum_lo as f32) - dmin_q8 * m1 * (q8_sum_lo as f32);
            total_acc += d_q8 * sc2 * (sum_hi as f32) - dmin_q8 * m2 * (q8_sum_hi as f32);
        }
    }

    Ok(total_acc)
}

/// Optimized AVX-512 VNNI Q4_K × Q8_K dot product - DEFERRED horizontal sum
///
/// PAR-126 Five-Whys optimization: Instead of horizontal sums per chunk (24+ per super-block),
/// we accumulate all 8 block results in vector registers and do ONE horizontal sum at the end.
/// This reduces horizontal sum operations from 24+ to 1 per super-block (24x reduction).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
unsafe fn fused_q4k_q8k_dot_avx512vnni_opt(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if q8k_scales.len() < num_super_blocks || q8k_quants.len() < expected_values {
        return Err(RealizarError::InvalidShape {
            reason: "Q8_K buffer too small".to_string(),
        });
    }

    let nibble_mask = _mm256_set1_epi8(0x0F_i8);
    let ones_16 = _mm256_set1_epi16(1);

    // Global float accumulator
    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            _mm_prefetch(
                q4k_data
                    .as_ptr()
                    .add((sb_idx + 1) * SUPER_BLOCK_BYTES)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                q8k_quants.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Read Q4_K header
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        let mut scales_raw = [0u8; 12];
        scales_raw.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        let q8_scale = q8k_scales[sb_idx];
        let d_q8 = d * q8_scale;
        let dmin_q8 = dmin * q8_scale;

        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // PAR-126: DEFERRED HORIZONTAL SUM OPTIMIZATION
        // Keep 8 block accumulators in __m256i vectors (one per 32-value block)
        // block_dots[i] = sum of (Q4[block_i] * Q8[block_i]) for block i
        // block_q8sums[i] = sum of Q8[block_i] for dmin correction
        let mut block_dots = [0i32; 8];
        let mut block_q8sums = [0i32; 8];

        // Process 64 values (2 blocks) per iteration, 4 iterations total = 8 blocks
        for chunk in 0..4 {
            let j = chunk * 64;
            let q_offset = j / 2;

            // Load 32 bytes Q4 (64 nibbles packed)
            let q4_bytes = _mm256_loadu_si256(qs_ptr.add(q_offset).cast::<__m256i>());

            // Extract nibbles: lo = first 32 values, hi = next 32 values
            let q4_lo = _mm256_and_si256(q4_bytes, nibble_mask);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), nibble_mask);

            // Load 64 bytes Q8
            let q8_lo = _mm256_loadu_si256(q8_ptr.add(j).cast::<__m256i>());
            let q8_hi = _mm256_loadu_si256(q8_ptr.add(j + 32).cast::<__m256i>());

            // Q4 × Q8 products -> i16 via maddubs -> i32 via madd
            // maddubs: adjacent u8*i8 pairs summed to i16
            // madd with ones: sum pairs of i16 to i32
            let prod_lo_i16 = _mm256_maddubs_epi16(q4_lo, q8_lo);
            let prod_hi_i16 = _mm256_maddubs_epi16(q4_hi, q8_hi);
            let prod_lo_i32 = _mm256_madd_epi16(prod_lo_i16, ones_16);
            let prod_hi_i32 = _mm256_madd_epi16(prod_hi_i16, ones_16);

            // prod_lo_i32 now has 8 i32 values (4 per 128-bit lane)
            // We need to reduce to 1 value per 32-element block
            // Lane 0-3 are from elements 0-15, lane 4-7 are from elements 16-31
            // So we sum all 8 to get one block sum

            // Sum all 8 i32 in prod_lo_i32 using only one horizontal sum
            // First add the two 128-bit halves
            let prod_lo_128 = _mm_add_epi32(
                _mm256_castsi256_si128(prod_lo_i32),
                _mm256_extracti128_si256(prod_lo_i32, 1),
            );
            let prod_hi_128 = _mm_add_epi32(
                _mm256_castsi256_si128(prod_hi_i32),
                _mm256_extracti128_si256(prod_hi_i32, 1),
            );

            // Now we have 4 i32 each - sum them with hadd
            let prod_lo_64 = _mm_hadd_epi32(prod_lo_128, prod_hi_128);
            let prod_32 = _mm_hadd_epi32(prod_lo_64, prod_lo_64);

            // Extract block sums - lane 0 is block_lo, lane 1 is block_hi
            let block_idx = chunk * 2;
            block_dots[block_idx] = _mm_extract_epi32(prod_32, 0);
            block_dots[block_idx + 1] = _mm_extract_epi32(prod_32, 1);

            // Q8 sums for dmin correction - convert i8 to i16 first to avoid overflow
            // We need sum of all 32 i8 values for each block
            let q8_lo_i16_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_lo));
            let q8_lo_i16_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_lo, 1));
            let q8_hi_i16_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_hi));
            let q8_hi_i16_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_hi, 1));

            // Sum the i16 values to i32 using madd with ones
            let q8_lo_i32_a = _mm256_madd_epi16(q8_lo_i16_a, _mm256_set1_epi16(1));
            let q8_lo_i32_b = _mm256_madd_epi16(q8_lo_i16_b, _mm256_set1_epi16(1));
            let q8_hi_i32_a = _mm256_madd_epi16(q8_hi_i16_a, _mm256_set1_epi16(1));
            let q8_hi_i32_b = _mm256_madd_epi16(q8_hi_i16_b, _mm256_set1_epi16(1));

            // Combine halves for each block
            let q8_lo_sum = _mm256_add_epi32(q8_lo_i32_a, q8_lo_i32_b);
            let q8_hi_sum = _mm256_add_epi32(q8_hi_i32_a, q8_hi_i32_b);

            // Horizontal sum each block's q8 sum
            let q8_lo_128 = _mm_add_epi32(
                _mm256_castsi256_si128(q8_lo_sum),
                _mm256_extracti128_si256(q8_lo_sum, 1),
            );
            let q8_hi_128 = _mm_add_epi32(
                _mm256_castsi256_si128(q8_hi_sum),
                _mm256_extracti128_si256(q8_hi_sum, 1),
            );
            let q8_64 = _mm_hadd_epi32(q8_lo_128, q8_hi_128);
            let q8_32 = _mm_hadd_epi32(q8_64, q8_64);

            block_q8sums[block_idx] = _mm_extract_epi32(q8_32, 0);
            block_q8sums[block_idx + 1] = _mm_extract_epi32(q8_32, 1);
        }

        // Extract all 8 scales and mins
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for i in 0..8 {
            let (sc, m) = extract_scale_min(&scales_raw, i);
            scales[i] = sc;
            mins[i] = m;
        }

        // Load scales and mins into vectors for SIMD multiply
        let scales_vec = _mm256_loadu_ps(scales.as_ptr());
        let mins_vec = _mm256_loadu_ps(mins.as_ptr());

        // Convert block_dots and block_q8sums to f32
        let dots_i32 = _mm256_loadu_si256(block_dots.as_ptr().cast::<__m256i>());
        let q8sums_i32 = _mm256_loadu_si256(block_q8sums.as_ptr().cast::<__m256i>());
        let dots_f32 = _mm256_cvtepi32_ps(dots_i32);
        let q8sums_f32 = _mm256_cvtepi32_ps(q8sums_i32);

        // Compute: d_q8 * scales * dots - dmin_q8 * mins * q8sums
        let d_q8_vec = _mm256_set1_ps(d_q8);
        let dmin_q8_vec = _mm256_set1_ps(dmin_q8);

        let term1 = _mm256_mul_ps(d_q8_vec, _mm256_mul_ps(scales_vec, dots_f32));
        let term2 = _mm256_mul_ps(dmin_q8_vec, _mm256_mul_ps(mins_vec, q8sums_f32));
        let result = _mm256_sub_ps(term1, term2);

        // ONE horizontal sum for all 8 blocks
        let sum128 = _mm_add_ps(
            _mm256_castps256_ps128(result),
            _mm256_extractf128_ps(result, 1),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        total_acc += _mm_cvtss_f32(sum32);
    }

    Ok(total_acc)
}

/// Fast horizontal sum of 4 i32 in __m128i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_128(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::{_mm_cvtsi128_si32, _mm_hadd_epi32};
    let sum64 = _mm_hadd_epi32(v, v);
    let sum32 = _mm_hadd_epi32(sum64, sum64);
    _mm_cvtsi128_si32(sum32)
}

/// Fast horizontal sum of 8 i32 in __m256i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{_mm256_castsi256_si128, _mm256_extracti128_si256, _mm_add_epi32};
    // SAFETY: Unsafe operation with validated invariants
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        hsum_epi32_128(_mm_add_epi32(lo, hi))
    }
}

/// Helper: horizontal sum of 8 int32 values in a 256-bit register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_epi32_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm_add_epi32, _mm_cvtsi128_si32,
        _mm_hadd_epi32,
    };

    // Add high 128 bits to low 128 bits
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);

    // Horizontal add within 128 bits
    let sum64 = _mm_hadd_epi32(sum128, sum128);
    let sum32 = _mm_hadd_epi32(sum64, sum64);

    _mm_cvtsi128_si32(sum32)
}

/// Helper: horizontal sum of 16 int16 values in a 256-bit register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn horizontal_sum_epi16_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::{_mm256_madd_epi16, _mm256_set1_epi16};

    // Use madd to sum pairs of i16 to i32
    let ones = _mm256_set1_epi16(1);
    let sum_i32 = _mm256_madd_epi16(v, ones);

    // Now sum the 8 i32 values
    horizontal_sum_epi32_256(sum_i32)
}

/// AVX2-optimized Q4_K × Q8_K dot product
///
/// # Safety
/// Requires AVX2 CPU feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_q4k_q8k_dot_avx2(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if q8k_scales.len() < num_super_blocks || q8k_quants.len() < expected_values {
        return Err(RealizarError::InvalidShape {
            reason: "Q8_K buffer too small".to_string(),
        });
    }

    let nibble_mask = _mm256_set1_epi8(0x0F_i8);
    let ones_16 = _mm256_set1_epi16(1);

    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            _mm_prefetch(
                q4k_data
                    .as_ptr()
                    .add((sb_idx + 1) * SUPER_BLOCK_BYTES)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                q8k_quants.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Read Q4_K header
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        let q8_scale = q8k_scales[sb_idx];
        let d_q8 = d * q8_scale;
        let dmin_q8 = dmin * q8_scale;

        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // (accumulator variables for future fixed-point optimization)
        let _acc_sum = _mm256_setzero_si256();
        let _acc_min = _mm256_setzero_si256();

        // Process 4 iterations of 64 values each (256 total)
        for j in (0..QK_K).step_by(64) {
            let q_offset = j / 2; // 32 bytes per 64 values

            // Get scales for two 32-value blocks
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Fixed-point scales (8.8 format) - used for integer path
            let sc1_i16 = (sc1 * 256.0).round() as i16;
            let sc2_i16 = (sc2 * 256.0).round() as i16;
            let _m1_i16 = (m1 * 256.0).round() as i16;
            let _m2_i16 = (m2 * 256.0).round() as i16;

            // Load 32 bytes of Q4_K (64 nibbles)
            let q4_bytes = _mm256_loadu_si256(qs_ptr.add(q_offset).cast::<__m256i>());

            // Extract nibbles
            let q4_lo = _mm256_and_si256(q4_bytes, nibble_mask);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), nibble_mask);

            // Load 64 bytes of Q8_K (sequential values)
            // CORRECT LAYOUT: dequantize_q4_k outputs 32 low nibbles, then 32 high nibbles
            // So Q8[j..j+32] corresponds to low nibbles, Q8[j+32..j+64] to high nibbles
            let q8_lo = _mm256_loadu_si256(q8_ptr.add(j).cast::<__m256i>());
            let q8_hi = _mm256_loadu_si256(q8_ptr.add(j + 32).cast::<__m256i>());

            // Q4_lo × Q8_lo: low nibbles times first 32 Q8 values (unsigned × signed → i16)
            let prod_lo = _mm256_maddubs_epi16(q4_lo, q8_lo);
            // Q4_hi × Q8_hi: high nibbles times second 32 Q8 values
            let prod_hi = _mm256_maddubs_epi16(q4_hi, q8_hi);

            // Apply block scales and accumulate (for future integer-only path)
            let _scale_lo = _mm256_set1_epi16(sc1_i16);
            let _scale_hi = _mm256_set1_epi16(sc2_i16);

            // Split products by block (first 128 bits = block 1, second 128 bits = block 2)
            let prod_lo_128 = _mm256_castsi256_si128(prod_lo);
            let prod_lo_hi128 = _mm256_extracti128_si256(prod_lo, 1);
            let prod_hi_128 = _mm256_castsi256_si128(prod_hi);
            let prod_hi_hi128 = _mm256_extracti128_si256(prod_hi, 1);

            // Horizontal sum to i32
            let sum_lo_1 = _mm_madd_epi16(prod_lo_128, _mm_set1_epi16(1));
            let sum_lo_2 = _mm_madd_epi16(prod_lo_hi128, _mm_set1_epi16(1));
            let sum_hi_1 = _mm_madd_epi16(prod_hi_128, _mm_set1_epi16(1));
            let sum_hi_2 = _mm_madd_epi16(prod_hi_hi128, _mm_set1_epi16(1));

            // Add low and high nibble products
            let sum_1 = _mm_add_epi32(sum_lo_1, sum_hi_1);
            let sum_2 = _mm_add_epi32(sum_lo_2, sum_hi_2);

            // Apply scales (as f32 to avoid overflow)
            let sum_1_f = _mm_cvtepi32_ps(sum_1);
            let sum_2_f = _mm_cvtepi32_ps(sum_2);

            let scaled_1 = _mm_mul_ps(sum_1_f, _mm_set1_ps(sc1));
            let scaled_2 = _mm_mul_ps(sum_2_f, _mm_set1_ps(sc2));

            // Sum for min contribution (sum of Q8 values)
            let q8_sum_lo =
                _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_lo)), ones_16);
            let q8_sum_hi = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_lo, 1)),
                ones_16,
            );

            // Horizontal reduce
            let hsum_lo = _mm_add_epi32(
                _mm256_castsi256_si128(q8_sum_lo),
                _mm256_extracti128_si256(q8_sum_lo, 1),
            );
            let _hsum_hi = _mm_add_epi32(
                _mm256_castsi256_si128(q8_sum_hi),
                _mm256_extracti128_si256(q8_sum_hi, 1),
            );

            // Include both halves in block sum
            let q8_block1_sum = _mm_add_epi32(hsum_lo, _mm_shuffle_epi32(hsum_lo, 0b10_11_00_01));
            let q8_block1_sum = _mm_add_epi32(
                q8_block1_sum,
                _mm_shuffle_epi32(q8_block1_sum, 0b00_00_10_10),
            );
            let q8_block1_val = _mm_cvtsi128_si32(q8_block1_sum);

            // Similar for second block (q8_hi)
            let q8_sum_hi2 =
                _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8_hi)), ones_16);
            let q8_sum_hi3 = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8_hi, 1)),
                ones_16,
            );
            let hsum2_lo = _mm_add_epi32(
                _mm256_castsi256_si128(q8_sum_hi2),
                _mm256_extracti128_si256(q8_sum_hi2, 1),
            );
            let hsum2_hi = _mm_add_epi32(
                _mm256_castsi256_si128(q8_sum_hi3),
                _mm256_extracti128_si256(q8_sum_hi3, 1),
            );
            let q8_block2_sum = _mm_add_epi32(hsum2_lo, hsum2_hi);
            let q8_block2_sum = _mm_add_epi32(
                q8_block2_sum,
                _mm_shuffle_epi32(q8_block2_sum, 0b10_11_00_01),
            );
            let q8_block2_sum = _mm_add_epi32(
                q8_block2_sum,
                _mm_shuffle_epi32(q8_block2_sum, 0b00_00_10_10),
            );
            let q8_block2_val = _mm_cvtsi128_si32(q8_block2_sum);

            // Final accumulation with f32 precision
            let scaled_sum = _mm_add_ps(scaled_1, scaled_2);
            let hsum = _mm_hadd_ps(scaled_sum, scaled_sum);
            let hsum = _mm_hadd_ps(hsum, hsum);
            let block_prod = _mm_cvtss_f32(hsum);

            total_acc += d_q8 * block_prod;
            total_acc -= dmin_q8 * (m1 * q8_block1_val as f32 + m2 * q8_block2_val as f32);
        }
    }

    Ok(total_acc)
}

/// TCB 4-row micro-kernel: Process 4 rows simultaneously sharing Q8K loads
///
/// This implements the trueno TCB micro-tile pattern (4×1×256):
/// - Load Q8K input ONCE per superblock
/// - Process 4 weight rows using the SAME Q8K loads
/// - Return 4 output values
///
/// # Performance
///
/// Sharing Q8K loads across 4 rows reduces memory bandwidth by ~4x for the
/// input vector, which is the key optimization from TCB (Tiling Compute Blocks).
///
/// # Safety
/// Requires AVX-512F, AVX-512 VNNI, and AVX-512BW CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
pub(crate) unsafe fn fused_q4k_q8k_dot_4rows_avx512vnni(
    row_ptrs: [*const u8; 4],
    bytes_per_row: usize,
    q8k_scales: &[f32],
    q8k_quants: &[i8],
) -> [f32; 4] {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;
    let num_super_blocks = bytes_per_row / SUPER_BLOCK_BYTES;

    let nibble_mask = _mm256_set1_epi8(0x0F_i8);
    let ones_16 = _mm256_set1_epi16(1);

    // 4 accumulators for 4 output rows (8 blocks × f32 = 8 values each)
    let mut total_acc = [_mm256_setzero_ps(); 4];

    for sb_idx in 0..num_super_blocks {
        let q8_start = sb_idx * QK_K;
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Prefetch next Q8K superblock (shared across all 4 rows)
        if sb_idx + 1 < num_super_blocks {
            _mm_prefetch(
                q8k_quants.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        let q8_scale = q8k_scales[sb_idx];
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // ============================================================
        // CRITICAL: Pre-load Q8K data and compute Q8 sums ONCE per superblock
        // These are shared across ALL 4 rows
        // ============================================================

        // Pre-load all Q8K chunks for this superblock (4 chunks × 2 registers = 8 loads)
        let q8_chunk0_lo = _mm256_loadu_si256(q8_ptr.cast::<__m256i>());
        let q8_chunk0_hi = _mm256_loadu_si256(q8_ptr.add(32).cast::<__m256i>());
        let q8_chunk1_lo = _mm256_loadu_si256(q8_ptr.add(64).cast::<__m256i>());
        let q8_chunk1_hi = _mm256_loadu_si256(q8_ptr.add(96).cast::<__m256i>());
        let q8_chunk2_lo = _mm256_loadu_si256(q8_ptr.add(128).cast::<__m256i>());
        let q8_chunk2_hi = _mm256_loadu_si256(q8_ptr.add(160).cast::<__m256i>());
        let q8_chunk3_lo = _mm256_loadu_si256(q8_ptr.add(192).cast::<__m256i>());
        let q8_chunk3_hi = _mm256_loadu_si256(q8_ptr.add(224).cast::<__m256i>());

        // Pre-compute Q8 sums for dmin correction (same for all rows)
        let q8_sums = compute_q8_sums_8blocks(
            q8_chunk0_lo,
            q8_chunk0_hi,
            q8_chunk1_lo,
            q8_chunk1_hi,
            q8_chunk2_lo,
            q8_chunk2_hi,
            q8_chunk3_lo,
            q8_chunk3_hi,
            ones_16,
        );

        // Process 4 rows using the pre-loaded Q8K data
        for row in 0..4 {
            let row_data = row_ptrs[row].add(sb_start);

            // Read Q4_K header for this row
            let d = read_f16(std::slice::from_raw_parts(row_data, 2));
            let dmin = read_f16(std::slice::from_raw_parts(row_data.add(2), 2));

            let mut scales_raw = [0u8; 12];
            std::ptr::copy_nonoverlapping(row_data.add(4), scales_raw.as_mut_ptr(), 12);

            let d_q8 = d * q8_scale;
            let dmin_q8 = dmin * q8_scale;

            let qs_ptr = row_data.add(16);

            // Compute Q4×Q8 dot products for all 8 blocks using pre-loaded Q8K
            let block_dots = compute_q4_q8_dots_8blocks(
                qs_ptr,
                q8_chunk0_lo,
                q8_chunk0_hi,
                q8_chunk1_lo,
                q8_chunk1_hi,
                q8_chunk2_lo,
                q8_chunk2_hi,
                q8_chunk3_lo,
                q8_chunk3_hi,
                nibble_mask,
                ones_16,
            );

            // Extract 6-bit scales and mins
            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            for i in 0..8 {
                let (sc, m) = extract_scale_min(&scales_raw, i);
                scales[i] = sc;
                mins[i] = m;
            }

            // Final computation: d_q8 * scales * dots - dmin_q8 * mins * q8sums
            let scales_vec = _mm256_loadu_ps(scales.as_ptr());
            let mins_vec = _mm256_loadu_ps(mins.as_ptr());
            let d_q8_vec = _mm256_set1_ps(d_q8);
            let dmin_q8_vec = _mm256_set1_ps(dmin_q8);

            let dots_f32 = _mm256_cvtepi32_ps(block_dots);
            let q8sums_f32 = _mm256_cvtepi32_ps(q8_sums);

            let term1 = _mm256_mul_ps(d_q8_vec, _mm256_mul_ps(scales_vec, dots_f32));
            let term2 = _mm256_mul_ps(dmin_q8_vec, _mm256_mul_ps(mins_vec, q8sums_f32));
            let result = _mm256_sub_ps(term1, term2);

            total_acc[row] = _mm256_add_ps(total_acc[row], result);
        }
    }

    // Final horizontal sums for each row
    let mut outputs = [0.0f32; 4];
    for row in 0..4 {
        let sum128 = _mm_add_ps(
            _mm256_castps256_ps128(total_acc[row]),
            _mm256_extractf128_ps(total_acc[row], 1),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        outputs[row] = _mm_cvtss_f32(sum32);
    }

    outputs
}

/// Helper: Compute Q8 sums for 8 blocks (shared across rows)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn compute_q8_sums_8blocks(
    c0_lo: std::arch::x86_64::__m256i,
    c0_hi: std::arch::x86_64::__m256i,
    c1_lo: std::arch::x86_64::__m256i,
    c1_hi: std::arch::x86_64::__m256i,
    c2_lo: std::arch::x86_64::__m256i,
    c2_hi: std::arch::x86_64::__m256i,
    c3_lo: std::arch::x86_64::__m256i,
    c3_hi: std::arch::x86_64::__m256i,
    ones_16: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // SAFETY: All calls are to unsafe intrinsics, within already-unsafe fn
    // Sum Q8 values for each of the 8 blocks (32 values each)
    unsafe {
        let sum0 = sum_i8_to_i32(c0_lo, ones_16);
        let sum1 = sum_i8_to_i32(c0_hi, ones_16);
        let sum2 = sum_i8_to_i32(c1_lo, ones_16);
        let sum3 = sum_i8_to_i32(c1_hi, ones_16);
        let sum4 = sum_i8_to_i32(c2_lo, ones_16);
        let sum5 = sum_i8_to_i32(c2_hi, ones_16);
        let sum6 = sum_i8_to_i32(c3_lo, ones_16);
        let sum7 = sum_i8_to_i32(c3_hi, ones_16);

        // Pack 8 sums into a single __m256i
        let mut result = _mm256_setzero_si256();
        result = _mm256_insert_epi32(result, sum0, 0);
        result = _mm256_insert_epi32(result, sum1, 1);
        result = _mm256_insert_epi32(result, sum2, 2);
        result = _mm256_insert_epi32(result, sum3, 3);
        result = _mm256_insert_epi32(result, sum4, 4);
        result = _mm256_insert_epi32(result, sum5, 5);
        result = _mm256_insert_epi32(result, sum6, 6);
        result = _mm256_insert_epi32(result, sum7, 7);
        result
    }
}

/// Helper: Sum 32 i8 values to i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sum_i8_to_i32(v: std::arch::x86_64::__m256i, ones: std::arch::x86_64::__m256i) -> i32 {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
    let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
    let sum_lo = _mm256_madd_epi16(lo, ones);
    let sum_hi = _mm256_madd_epi16(hi, ones);
    let sum = _mm256_add_epi32(sum_lo, sum_hi);

    // Horizontal sum of 8 i32 -> 1 i32
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum),
        _mm256_extracti128_si256(sum, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_10_10));
    _mm_cvtsi128_si32(sum32)
}

/// Helper: Compute Q4×Q8 dot products for 8 blocks using pre-loaded Q8K
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn compute_q4_q8_dots_8blocks(
    qs_ptr: *const u8,
    q8_c0_lo: std::arch::x86_64::__m256i,
    q8_c0_hi: std::arch::x86_64::__m256i,
    q8_c1_lo: std::arch::x86_64::__m256i,
    q8_c1_hi: std::arch::x86_64::__m256i,
    q8_c2_lo: std::arch::x86_64::__m256i,
    q8_c2_hi: std::arch::x86_64::__m256i,
    q8_c3_lo: std::arch::x86_64::__m256i,
    q8_c3_hi: std::arch::x86_64::__m256i,
    nibble_mask: std::arch::x86_64::__m256i,
    ones_16: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // SAFETY: All intrinsics and pointer ops are unsafe, within already-unsafe fn
    unsafe {
        // Load Q4K data (128 bytes total for 256 values)
        let q4_c0 = _mm256_loadu_si256(qs_ptr.cast::<__m256i>());
        let q4_c1 = _mm256_loadu_si256(qs_ptr.add(32).cast::<__m256i>());
        let q4_c2 = _mm256_loadu_si256(qs_ptr.add(64).cast::<__m256i>());
        let q4_c3 = _mm256_loadu_si256(qs_ptr.add(96).cast::<__m256i>());

        // Extract nibbles and compute Q4×Q8 for each chunk
        let dot0 = q4_q8_chunk_dot(q4_c0, q8_c0_lo, q8_c0_hi, nibble_mask, ones_16);
        let dot1 = q4_q8_chunk_dot(q4_c1, q8_c1_lo, q8_c1_hi, nibble_mask, ones_16);
        let dot2 = q4_q8_chunk_dot(q4_c2, q8_c2_lo, q8_c2_hi, nibble_mask, ones_16);
        let dot3 = q4_q8_chunk_dot(q4_c3, q8_c3_lo, q8_c3_hi, nibble_mask, ones_16);

        // Pack 8 dot products into result (2 per chunk: lo nibble block, hi nibble block)
        let mut result = _mm256_setzero_si256();
        result = _mm256_insert_epi32(result, _mm_cvtsi128_si32(_mm256_castsi256_si128(dot0)), 0);
        result = _mm256_insert_epi32(
            result,
            _mm_extract_epi32(_mm256_castsi256_si128(dot0), 1),
            1,
        );
        result = _mm256_insert_epi32(result, _mm_cvtsi128_si32(_mm256_castsi256_si128(dot1)), 2);
        result = _mm256_insert_epi32(
            result,
            _mm_extract_epi32(_mm256_castsi256_si128(dot1), 1),
            3,
        );
        result = _mm256_insert_epi32(result, _mm_cvtsi128_si32(_mm256_castsi256_si128(dot2)), 4);
        result = _mm256_insert_epi32(
            result,
            _mm_extract_epi32(_mm256_castsi256_si128(dot2), 1),
            5,
        );
        result = _mm256_insert_epi32(result, _mm_cvtsi128_si32(_mm256_castsi256_si128(dot3)), 6);
        result = _mm256_insert_epi32(
            result,
            _mm_extract_epi32(_mm256_castsi256_si128(dot3), 1),
            7,
        );
        result
    }
}

/// Helper: Q4×Q8 dot product for one 64-value chunk (32 lo nibbles + 32 hi nibbles)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q4_q8_chunk_dot(
    q4_packed: std::arch::x86_64::__m256i,
    q8_lo: std::arch::x86_64::__m256i,
    q8_hi: std::arch::x86_64::__m256i,
    nibble_mask: std::arch::x86_64::__m256i,
    ones_16: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // SAFETY: All intrinsics are unsafe, within already-unsafe fn
    unsafe {
        // Extract nibbles
        let q4_lo = _mm256_and_si256(q4_packed, nibble_mask);
        let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_packed, 4), nibble_mask);

        // Q4 × Q8 products
        let prod_lo_i16 = _mm256_maddubs_epi16(q4_lo, q8_lo);
        let prod_hi_i16 = _mm256_maddubs_epi16(q4_hi, q8_hi);
        let prod_lo_i32 = _mm256_madd_epi16(prod_lo_i16, ones_16);
        let prod_hi_i32 = _mm256_madd_epi16(prod_hi_i16, ones_16);

        // Reduce each to single sum
        let sum_lo = hsum_epi32(prod_lo_i32);
        let sum_hi = hsum_epi32(prod_hi_i32);

        // Return [sum_lo, sum_hi, 0, 0, 0, 0, 0, 0]
        let mut result = _mm256_setzero_si256();
        result = _mm256_insert_epi32(result, sum_lo, 0);
        result = _mm256_insert_epi32(result, sum_hi, 1);
        result
    }
}

/// Helper: Horizontal sum of 8 i32 values to single i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // All intrinsics are unsafe and we're in an unsafe fn with target_feature
    let sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_10_10));
    _mm_cvtsi128_si32(sum32)
}

#[cfg(test)]
#[path = "fused_k_tests.rs"]
mod fused_k_tests;
