//! Fused activation functions for quantized inference (PMAT-802)
//!
//! Implements fused operations combining normalization with quantization:
//! - `quantize_rmsnorm_q8_0` - RMSNorm with Q8_0 quantization
//! - `quantize_rmsnorm_q8_0_into` - Zero-allocation variant
//! - `fused_rmsnorm_q4_0_matmul` - RMSNorm + matmul fusion
//! - `fused_rmsnorm_ffn_up_gate` - RMSNorm + FFN up/gate fusion
//! - `fused_swiglu_simd` - SIMD-accelerated SwiGLU activation
//! - `softmax_simd` - SIMD-accelerated softmax

use crate::error::{RealizarError, Result};
use super::types::{BLOCK_SIZE, Q4_0Block};
use super::fused_q4_0_q8_0_dot_simd;

// ============================================================================
// Key insight: llama.cpp quantizes activations to Q8_0 and uses integer
// multiply-accumulate (maddubs_epi16), which is 4-5x faster than f32 FMA.

/// Fused RMSNorm + Q8_0 quantization
///
/// Computes RMSNorm and quantizes in a single pass:
/// normalized[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
/// Then quantizes to Q8_0 format.
///
/// This avoids allocating an intermediate normalized vector.
#[inline]
pub fn quantize_rmsnorm_q8_0(input: &[f32], norm_weight: &[f32], eps: f32) -> (Vec<f32>, Vec<i8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { quantize_rmsnorm_q8_0_avx2(input, norm_weight, eps) };
        }
    }
    quantize_rmsnorm_q8_0_scalar(input, norm_weight, eps)
}

/// Scalar implementation of fused RMSNorm + Q8_0 quantization
///
/// This is exposed as `pub(crate)` for direct testing. The production code
/// uses the public `quantize_rmsnorm_q8_0` wrapper which dispatches to AVX2
/// when available.
pub(crate) fn quantize_rmsnorm_q8_0_scalar(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<i8>) {
    let hidden_dim = input.len();
    debug_assert_eq!(hidden_dim, norm_weight.len());

    // Compute sum of squares for RMSNorm
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    // Now quantize the normalized values directly
    let num_blocks = hidden_dim.div_ceil(32);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut quants = Vec::with_capacity(num_blocks * 32);

    for block_idx in 0..num_blocks {
        let start = block_idx * 32;
        let end = (start + 32).min(hidden_dim);

        // Find max absolute value of normalized values for this block
        let mut max_abs = 0.0f32;
        for i in start..end {
            // Fused: x[i] * inv_rms * weight[i]
            let normalized = input[i] * inv_rms * norm_weight[i];
            let abs = normalized.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        // Compute scale
        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0 / 127.0
        };
        let inv_scale = 1.0 / scale;
        scales.push(scale);

        // Quantize normalized values
        for i in start..end {
            let normalized = input[i] * inv_rms * norm_weight[i];
            let q = (normalized * inv_scale).round();
            quants.push(q.clamp(-128.0, 127.0) as i8);
        }
        // Pad to 32 if partial block
        for _ in end..(start + 32) {
            quants.push(0i8);
        }
    }

    (scales, quants)
}

/// AVX2-accelerated fused RMSNorm + Q8_0 quantization
///
/// Processes 8 floats at a time using SIMD for:
/// - Sum of squares computation
/// - Max abs finding per block
/// - Normalization and quantization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn quantize_rmsnorm_q8_0_avx2(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<i8>) {
    // SAFETY: Memory safety ensured by bounds checking and alignment
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_and_ps, _mm256_andnot_ps, _mm256_castps256_ps128,
            _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cvtps_epi32, _mm256_extractf128_ps,
            _mm256_extracti128_si256, _mm256_floor_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
            _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps, _mm256_set1_epi32,
            _mm256_set1_ps, _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm_max_ps,
            _mm_movehl_ps, _mm_packs_epi16, _mm_packs_epi32, _mm_shuffle_ps, _mm_storel_epi64,
        };

        let hidden_dim = input.len();
        debug_assert_eq!(hidden_dim, norm_weight.len());

        // SIMD sum of squares
        let mut sum_sq_vec = _mm256_setzero_ps();
        let mut i = 0;

        // Process 8 floats at a time
        while i + 8 <= hidden_dim {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
            i += 8;
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum_sq_vec, 1);
        let lo = _mm256_castps256_ps128(sum_sq_vec);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);
        let mut sum_sq = _mm_cvtss_f32(sum32);

        // Handle remaining elements
        while i < hidden_dim {
            sum_sq += input[i] * input[i];
            i += 1;
        }

        let mean_sq = sum_sq / hidden_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let inv_rms_vec = _mm256_set1_ps(inv_rms);

        // Quantize with SIMD
        let num_blocks = hidden_dim.div_ceil(32);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut quants = vec![0i8; num_blocks * 32];

        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF_u32 as i32));
        let round_const = _mm256_set1_ps(0.5);
        let clamp_min = _mm256_set1_ps(-128.0);
        let clamp_max = _mm256_set1_ps(127.0);

        for block_idx in 0..num_blocks {
            let start = block_idx * 32;
            let block_end = (start + 32).min(hidden_dim);
            let valid_len = block_end - start;

            // Find max abs in this block using SIMD
            let mut max_vec = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= valid_len {
                let idx = start + j;
                let inp = _mm256_loadu_ps(input.as_ptr().add(idx));
                let wgt = _mm256_loadu_ps(norm_weight.as_ptr().add(idx));
                let normalized = _mm256_mul_ps(_mm256_mul_ps(inp, inv_rms_vec), wgt);
                let abs_val = _mm256_and_ps(normalized, abs_mask);
                max_vec = _mm256_max_ps(max_vec, abs_val);
                j += 8;
            }

            // Horizontal max
            let max_hi = _mm256_extractf128_ps(max_vec, 1);
            let max_lo = _mm256_castps256_ps128(max_vec);
            let max_128 = _mm_max_ps(max_lo, max_hi);
            let max_64 = _mm_max_ps(max_128, _mm_movehl_ps(max_128, max_128));
            let max_32 = _mm_max_ps(max_64, _mm_shuffle_ps(max_64, max_64, 1));
            let mut max_abs = _mm_cvtss_f32(max_32);

            // Handle remaining elements in block
            while j < valid_len {
                let normalized = input[start + j] * inv_rms * norm_weight[start + j];
                let abs = normalized.abs();
                if abs > max_abs {
                    max_abs = abs;
                }
                j += 1;
            }

            // Compute scale
            let scale = if max_abs > 1e-10 {
                max_abs / 127.0
            } else {
                1.0 / 127.0
            };
            let inv_scale = 1.0 / scale;
            let inv_scale_vec = _mm256_set1_ps(inv_scale);
            scales.push(scale);

            // Quantize with SIMD
            let quant_ptr = quants.as_mut_ptr().add(block_idx * 32);
            let mut k = 0;
            while k + 8 <= valid_len {
                let idx = start + k;
                let inp = _mm256_loadu_ps(input.as_ptr().add(idx));
                let wgt = _mm256_loadu_ps(norm_weight.as_ptr().add(idx));
                let normalized = _mm256_mul_ps(_mm256_mul_ps(inp, inv_rms_vec), wgt);
                let scaled = _mm256_mul_ps(normalized, inv_scale_vec);

                // Round to nearest (add 0.5 and truncate, handle sign)
                let sign = _mm256_and_ps(
                    scaled,
                    _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000_u32 as i32)),
                );
                let abs_scaled = _mm256_andnot_ps(sign, scaled);
                let rounded = _mm256_or_ps(
                    _mm256_floor_ps(_mm256_add_ps(abs_scaled, round_const)),
                    sign,
                );

                // Clamp to [-128, 127]
                let clamped = _mm256_max_ps(clamp_min, _mm256_min_ps(clamp_max, rounded));

                // Convert to int32 and extract to i8
                let int32 = _mm256_cvtps_epi32(clamped);

                // Pack i32 -> i16 -> i8 (only need lower 8 values)
                let lo128 = _mm256_castsi256_si128(int32);
                let hi128 = _mm256_extracti128_si256(int32, 1);
                let packed16 = _mm_packs_epi32(lo128, hi128);
                let packed8 = _mm_packs_epi16(packed16, packed16);

                // Store 8 i8 values
                _mm_storel_epi64(quant_ptr.add(k).cast(), packed8);
                k += 8;
            }

            // Handle remaining elements
            while k < valid_len {
                let normalized = input[start + k] * inv_rms * norm_weight[start + k];
                let q = (normalized * inv_scale).round();
                *quant_ptr.add(k) = q.clamp(-128.0, 127.0) as i8;
                k += 1;
            }
        }

        (scales, quants)
    }
}

/// Fused RMSNorm + Q4_0 matmul
///
/// Combines RMSNorm normalization with quantized matmul in one operation:
/// 1. Computes inv_rms = 1 / sqrt(mean(x^2) + eps)
/// 2. Quantizes (x * inv_rms * norm_weight) to Q8_0
/// 3. Performs Q4_0 × Q8_0 integer matmul
///
/// This eliminates the intermediate normalized vector allocation.
#[allow(clippy::similar_names)]
pub fn fused_rmsnorm_q4_0_matmul(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
    weight_data: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    const Q4_0_BLOCK_BYTES: usize = 18;
    const Q4_0_BLOCK_SIZE: usize = 32;

    if input.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Input length {} doesn't match in_dim {}",
                input.len(),
                in_dim
            ),
        });
    }

    let blocks_per_row = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_0 weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    // Fused RMSNorm + Q8_0 quantization (single pass, no intermediate allocation)
    let (q8_scales, q8_quants) = quantize_rmsnorm_q8_0(input, norm_weight, eps);

    // Parallel matmul with chunking
    const CHUNK_SIZE: usize = 64;
    let output: Vec<f32> = (0..out_dim)
        .into_par_iter()
        .with_min_len(CHUNK_SIZE)
        .map(|o| {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];
            fused_q4_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim)
        })
        .collect();

    Ok(output)
}

/// Fused RMSNorm + parallel FFN up/gate projections
///
/// For SwiGLU models, FFN has two parallel matmuls (up and gate) that share
/// the same normalized input. This function:
/// 1. Computes inv_rms once
/// 2. Quantizes normalized input to Q8_0 once
/// 3. Runs both up and gate matmuls in parallel
///
/// Eliminates: 1 RMSNorm pass, 1 intermediate allocation, 1 Q8_0 quantization
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
pub fn fused_rmsnorm_ffn_up_gate(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
    up_weight_data: &[u8],
    gate_weight_data: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use rayon::prelude::*;

    const Q4_0_BLOCK_BYTES: usize = 18;
    const Q4_0_BLOCK_SIZE: usize = 32;

    if input.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Input length {} doesn't match in_dim {}",
                input.len(),
                in_dim
            ),
        });
    }

    let blocks_per_row = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
    let expected_weight_bytes = out_dim * bytes_per_row;

    if up_weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "FFN up weight data too small: need {} bytes, have {}",
                expected_weight_bytes,
                up_weight_data.len()
            ),
        });
    }
    if gate_weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "FFN gate weight data too small: need {} bytes, have {}",
                expected_weight_bytes,
                gate_weight_data.len()
            ),
        });
    }

    // Fused RMSNorm + Q8_0 quantization - computed ONCE for both matmuls
    let (q8_scales, q8_quants) = quantize_rmsnorm_q8_0(input, norm_weight, eps);

    // Run both matmuls in parallel using rayon::join
    // Each matmul uses parallel iteration with chunking to reduce overhead
    let (up_output, gate_output) = rayon::join(
        || {
            const CHUNK_SIZE: usize = 64;
            (0..out_dim)
                .into_par_iter()
                .with_min_len(CHUNK_SIZE)
                .map(|o| {
                    let row_start = o * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    let row_data = &up_weight_data[row_start..row_end];
                    fused_q4_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim)
                })
                .collect::<Vec<f32>>()
        },
        || {
            const CHUNK_SIZE: usize = 64;
            (0..out_dim)
                .into_par_iter()
                .with_min_len(CHUNK_SIZE)
                .map(|o| {
                    let row_start = o * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    let row_data = &gate_weight_data[row_start..row_end];
                    fused_q4_0_q8_0_dot_simd(row_data, &q8_scales, &q8_quants, in_dim)
                })
                .collect::<Vec<f32>>()
        },
    );

    Ok((up_output, gate_output))
}

/// Zero-allocation variant of quantize_rmsnorm_q8_0
///
/// Writes results directly into pre-allocated output buffers.
pub fn quantize_rmsnorm_q8_0_into(
    input: &[f32],
    norm_weight: &[f32],
    eps: f32,
    scales: &mut [f32],
    quants: &mut [i8],
) {
    let hidden_dim = input.len();
    debug_assert_eq!(hidden_dim, norm_weight.len());

    // Compute sum of squares for RMSNorm
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    let num_blocks = hidden_dim.div_ceil(32);

    for block_idx in 0..num_blocks {
        let start = block_idx * 32;
        let end = (start + 32).min(hidden_dim);

        // Find max absolute value of normalized values for this block
        let mut max_abs = 0.0f32;
        for i in start..end {
            let normalized = input[i] * inv_rms * norm_weight[i];
            let abs = normalized.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        // Compute scale
        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0 / 127.0
        };
        let inv_scale = 1.0 / scale;
        scales[block_idx] = scale;

        // Quantize normalized values
        let quant_start = block_idx * 32;
        for i in start..end {
            let normalized = input[i] * inv_rms * norm_weight[i];
            let q = (normalized * inv_scale).round();
            quants[quant_start + (i - start)] = q.clamp(-128.0, 127.0) as i8;
        }
        // Pad to 32 if partial block
        for j in (end - start)..32 {
            quants[quant_start + j] = 0i8;
        }
    }
}

/// SIMD-accelerated fused SwiGLU activation: silu(gate) * up
///
/// Combines silu activation and element-wise multiply in a single pass
/// for better cache locality. Uses AVX2/AVX-512 SIMD where available.
///
/// # Arguments
/// * `gate` - Gate values, modified in-place to contain result
/// * `up` - Up projection values
pub fn fused_swiglu_simd(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA verified at runtime
            unsafe {
                fused_swiglu_avx2(gate, up);
            }
            return;
        }
    }

    // Scalar fallback
    fused_swiglu_scalar(gate, up);
}

/// Scalar fused SwiGLU: silu(gate) * up
///
/// Exposed as `pub(crate)` for direct testing on AVX2 machines.
#[inline]
pub(crate) fn fused_swiglu_scalar(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let silu_g = *g / (1.0 + (-*g).exp());
        *g = silu_g * u;
    }
}

/// AVX2 SIMD fused SwiGLU with FMA
///
/// Computes silu(gate) * up using:
/// - Polynomial approximation for exp(-x)
/// - FMA for efficient multiply-add
/// - 8-wide AVX2 vectors
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
#[allow(clippy::many_single_char_names)]
unsafe fn fused_swiglu_avx2(gate: &mut [f32], up: &[f32]) {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtps_epi32, _mm256_floor_ps,
        _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps,
        _mm256_rcp_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_slli_epi32,
        _mm256_storeu_ps, _mm256_sub_ps,
    };

    // SAFETY: Memory safety ensured by bounds checking and alignment
    unsafe {
        let n = gate.len();
        let mut i = 0;

        // Constants for exp approximation (polynomial coefficients)
        // Using 5th-degree polynomial approximation for exp(x) on [-87, 0]
        let one = _mm256_set1_ps(1.0);
        let ln2_inv = _mm256_set1_ps(1.442_695); // 1/ln(2)
        let ln2 = _mm256_set1_ps(0.693_147_2);
        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(0.693_147_2); // ln(2)
        let c2 = _mm256_set1_ps(0.240_226_5); // ln(2)^2 / 2!
        let c3 = _mm256_set1_ps(0.055_504_11); // ln(2)^3 / 3!
        let c4 = _mm256_set1_ps(0.009_618_13); // ln(2)^4 / 4!
        let c5 = _mm256_set1_ps(0.001_333_36); // ln(2)^5 / 5!
        let min_exp = _mm256_set1_ps(-87.0); // Minimum input to avoid underflow
        let two = _mm256_set1_ps(2.0); // For Newton-Raphson

        // Process 8 elements at a time
        while i + 8 <= n {
            // Load gate and up values
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let u = _mm256_loadu_ps(up.as_ptr().add(i));

            // Compute -g for sigmoid
            let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);

            // Clamp to avoid exp underflow
            let neg_g_clamped = _mm256_max_ps(neg_g, min_exp);

            // Fast exp approximation using 2^(x/ln2) = 2^n * 2^f where n=floor, f=frac
            // n = floor(x * 1/ln2)
            let xln2 = _mm256_mul_ps(neg_g_clamped, ln2_inv);
            let n_f = _mm256_floor_ps(xln2);
            let n_i = _mm256_cvtps_epi32(n_f);

            // f = x - n * ln2 (fractional part scaled back)
            let f = _mm256_fnmadd_ps(n_f, ln2, neg_g_clamped);

            // Horner's method: c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))
            let p = _mm256_fmadd_ps(f, c5, c4);
            let p = _mm256_fmadd_ps(f, p, c3);
            let p = _mm256_fmadd_ps(f, p, c2);
            let p = _mm256_fmadd_ps(f, p, c1);
            let p = _mm256_fmadd_ps(f, p, c0);

            // Scale by 2^n using integer bit manipulation
            // 2^n = reinterpret((n + 127) << 23) as float
            let bias = _mm256_set1_epi32(127);
            let n_biased = _mm256_add_epi32(n_i, bias);
            let exp_scale = _mm256_slli_epi32::<23>(n_biased);
            let exp_scale_f = _mm256_castsi256_ps(exp_scale);

            // exp(-g) = 2^n * p(f)
            let exp_neg_g = _mm256_mul_ps(p, exp_scale_f);

            // sigmoid(-(-g)) = 1 / (1 + exp(-g))
            // Use fast reciprocal approximation with Newton-Raphson refinement
            let denom = _mm256_add_ps(one, exp_neg_g);
            let rcp = _mm256_rcp_ps(denom); // ~12-bit precision
                                            // One Newton-Raphson iteration: x' = x * (2 - d*x)
            let sigmoid = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));

            // silu(g) = g * sigmoid(g)
            let silu_g = _mm256_mul_ps(g, sigmoid);

            // Result = silu(g) * u
            let result = _mm256_mul_ps(silu_g, u);

            // Store result
            _mm256_storeu_ps(gate.as_mut_ptr().add(i), result);

            i += 8;
        }

        // Handle remainder with scalar code
        while i < n {
            let g = gate[i];
            let silu_g = g / (1.0 + (-g).exp());
            gate[i] = silu_g * up[i];
            i += 1;
        }
    }
}

/// SIMD-optimized in-place softmax
///
/// Computes softmax(x) = exp(x - max) / sum(exp(x - max))
/// Uses AVX2/AVX-512 for vectorized exp and horizontal operations.
///
/// # Arguments
/// * `x` - Slice to softmax in-place
#[inline]
pub fn softmax_simd(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                softmax_avx2(x);
            }
            return;
        }
    }

    // Scalar fallback
    softmax_scalar(x);
}

/// Scalar softmax
///
/// Exposed as `pub(crate)` for direct testing on AVX2 machines.
#[inline]
pub(crate) fn softmax_scalar(x: &mut [f32]) {
    // Find max for numerical stability
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// AVX2 SIMD softmax - only SIMD for max-find and normalization
/// (exp() uses libm which is faster than polynomial for short vectors)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn softmax_avx2(x: &mut [f32]) {
    use std::arch::x86_64::{
        _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };

    let n = x.len();
    if n == 0 {
        return;
    }

    // ============= Phase 1: Find max (SIMD) =============
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;

    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        max_vec = _mm256_max_ps(max_vec, v);
        i += 8;
    }

    let mut max_scalar = horizontal_max_avx2(max_vec);
    for j in i..n {
        max_scalar = max_scalar.max(x[j]);
    }

    // ============= Phase 2: Compute exp(x - max) (scalar libm) =============
    let mut sum_scalar = 0.0f32;
    for j in 0..n {
        let exp_v = (x[j] - max_scalar).exp();
        x[j] = exp_v;
        sum_scalar += exp_v;
    }

    // ============= Phase 3: Normalize (SIMD) =============
    let inv_sum = _mm256_set1_ps(1.0 / sum_scalar);

    i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let normalized = _mm256_mul_ps(v, inv_sum);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), normalized);
        i += 8;
    }

    let inv_sum_scalar = 1.0 / sum_scalar;
    for j in i..n {
        x[j] *= inv_sum_scalar;
    }
}

/// Fast exp approximation using polynomial (AVX2)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fast_exp_avx2(
    x: std::arch::x86_64::__m256,
    ln2_inv: std::arch::x86_64::__m256,
    ln2: std::arch::x86_64::__m256,
    c0: std::arch::x86_64::__m256,
    c1: std::arch::x86_64::__m256,
    c2: std::arch::x86_64::__m256,
    c3: std::arch::x86_64::__m256,
    c4: std::arch::x86_64::__m256,
    c5: std::arch::x86_64::__m256,
    min_exp: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_castsi256_ps, _mm256_cvtps_epi32, _mm256_floor_ps,
        _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_epi32,
        _mm256_slli_epi32,
    };

    {
        // Clamp to avoid underflow
        let x_clamped = _mm256_max_ps(x, min_exp);

        // n = floor(x / ln2)
        let xln2 = _mm256_mul_ps(x_clamped, ln2_inv);
        let n_f = _mm256_floor_ps(xln2);
        let n_i = _mm256_cvtps_epi32(n_f);

        // f = x - n * ln2
        let f = _mm256_fnmadd_ps(n_f, ln2, x_clamped);

        // Polynomial: c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))
        let p = _mm256_fmadd_ps(f, c5, c4);
        let p = _mm256_fmadd_ps(f, p, c3);
        let p = _mm256_fmadd_ps(f, p, c2);
        let p = _mm256_fmadd_ps(f, p, c1);
        let p = _mm256_fmadd_ps(f, p, c0);

        // 2^n via bit manipulation
        let bias = _mm256_set1_epi32(127);
        let n_biased = _mm256_add_epi32(n_i, bias);
        let exp_scale = _mm256_slli_epi32::<23>(n_biased);
        let exp_scale_f = _mm256_castsi256_ps(exp_scale);

        _mm256_mul_ps(p, exp_scale_f)
    }
}

/// Horizontal max of 8-wide AVX2 vector
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn horizontal_max_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::{
        _mm256_extractf128_ps, _mm_cvtss_f32, _mm_max_ps, _mm_max_ss, _mm_movehl_ps, _mm_shuffle_ps,
    };

    {
        // Extract high and low 128-bit lanes
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_extractf128_ps::<0>(v);
        let max128 = _mm_max_ps(hi, lo);

        // Reduce 4 to 2
        let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));

        // Reduce 2 to 1
        let max32 = _mm_max_ss(max64, _mm_shuffle_ps::<0x55>(max64, max64));

        _mm_cvtss_f32(max32)
    }
}

/// Horizontal sum of 8-wide AVX2 vector
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::{
        _mm256_extractf128_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    {
        // Extract high and low 128-bit lanes
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_extractf128_ps::<0>(v);
        let sum128 = _mm_add_ps(hi, lo);

        // Reduce 4 to 2
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));

        // Reduce 2 to 1
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<0x55>(sum64, sum64));

        _mm_cvtss_f32(sum32)
    }
}

/// Quantize f32 activations to Q8_0 format for fast integer matmul
///
/// Returns (scales, quantized_values) where each block of 32 values
/// has one f32 scale and 32 int8 quantized values.
#[inline]
pub fn quantize_activations_q8_0(activations: &[f32]) -> (Vec<f32>, Vec<i8>) {
    let num_blocks = activations.len().div_ceil(32);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut quants = Vec::with_capacity(num_blocks * 32);

    for block_idx in 0..num_blocks {
        let start = block_idx * 32;
        let end = (start + 32).min(activations.len());

        // Find max absolute value for symmetric quantization
        let mut max_abs = 0.0f32;
        for i in start..end {
            let abs = activations[i].abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        // Compute scale (avoid division by zero)
        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0 / 127.0
        };
        let inv_scale = 1.0 / scale;
        scales.push(scale);

        // Quantize values
        for i in start..end {
            let q = (activations[i] * inv_scale).round();
            quants.push(q.clamp(-128.0, 127.0) as i8);
        }
        // Pad to 32 if partial block
        for _ in end..(start + 32) {
            quants.push(0i8);
        }
    }

    (scales, quants)
}
