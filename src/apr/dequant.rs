//! Dequantization helpers for APR quantized tensor formats (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.

/// Convert F16 (IEEE 754 half-precision) to F32
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
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

/// Dequantize F16 data to F32
pub fn dequantize_f16(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_elements);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        result.push(f16_to_f32(bits));
    }
    result.truncate(num_elements);
    result
}

/// Dequantize Q8_0 format (GGUF compatible)
/// Q8_0: blocks of 32 elements, each block has 2-byte f16 scale + 32 bytes of int8 quants
pub fn dequantize_q8_0(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 32; // f16 scale + 32 int8 values

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + BLOCK_BYTES <= bytes.len() {
        // Read scale (f16)
        let scale_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
        offset += 2;

        // Read 32 int8 values
        for i in 0..BLOCK_SIZE {
            if result.len() >= num_elements {
                break;
            }
            let v = f32::from(bytes[offset + i] as i8);
            result.push(v * scale);
        }
        offset += 32;
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q4_K format (GGUF K-quants)
/// Q4_K: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
///
/// LAYOUT-001 FIX: Element ordering must match fused_q4k_dot (PAR-001):
/// - 4 chunks of 64 values each (at offsets 0, 64, 128, 192)
/// - Each chunk: 32 low nibbles (scale is), then 32 high nibbles (scale is+1)
/// - NOT interleaved (L0, H0, L1, H1...) - that was the bug!
pub fn dequantize_q4_k(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + SUPER_BLOCK_BYTES <= bytes.len() {
        // Read d (f16 scale) and dmin (f16 min)
        let d = f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]));
        offset += 4;

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&bytes[offset..offset + 12]);
        offset += 12;

        // Read 128 bytes = 256 4-bit quantized values
        let qs = &bytes[offset..offset + 128];
        offset += 128;

        // PAR-001: Match fused_q4k_dot layout (llama.cpp/candle compatible)
        // Process 4 chunks of 64 values each (0, 64, 128, 192)
        // Each chunk: 32 low nibbles, then 32 high nibbles from 32 consecutive bytes
        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min_q4k(&scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;

            let (sc2, m2) = extract_scale_min_q4k(&scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;

            // First pass: 32 low nibbles (use sc1, m1)
            for &byte in q {
                if result.len() >= num_elements {
                    break;
                }
                let q_val = (byte & 0x0F) as f32;
                result.push(d1 * q_val - dm1);
            }

            // Second pass: 32 high nibbles (use sc2, m2)
            for &byte in q {
                if result.len() >= num_elements {
                    break;
                }
                let q_val = (byte >> 4) as f32;
                result.push(d2 * q_val - dm2);
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Extract scale and min for Q4_K block index (0-7)
/// Matches fused_k.rs extract_scale_min exactly
#[inline]
fn extract_scale_min_q4k(scales: &[u8; 12], block_idx: usize) -> (f32, f32) {
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

/// Dequantize Q6_K format (GGUF K-quants)
/// Q6_K: super blocks of 256 elements
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
///
/// LAYOUT-001 FIX: Element ordering must match fused_q6k_dot (candle compatible):
/// - Process 128 values at a time (n=0, n=128)
/// - For each l in 0..32, extract 4 values at positions n+l, n+l+32, n+l+64, n+l+96
#[allow(clippy::cast_possible_wrap)]
pub fn dequantize_q6_k(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 128 + 64 + 16 + 2; // 210 bytes

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + SUPER_BLOCK_BYTES <= bytes.len() {
        // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2)
        let ql = &bytes[offset..offset + 128];
        offset += 128;

        let qh = &bytes[offset..offset + 64];
        offset += 64;

        // Read scales (16 bytes, signed i8)
        let mut scales = [0i8; 16];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = bytes[offset + i] as i8;
        }
        offset += 16;

        // Read d (f16 -> f32)
        let d = f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        offset += 2;

        // Match fused_q6k_dot layout exactly
        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            // Output positions n+0 to n+31 (q1 values)
            for l in 0..32 {
                if result.len() >= num_elements {
                    break;
                }
                let is = l / 16;
                let q1 = ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32;
                result.push(d * (sc[is] as f32) * (q1 as f32));
            }

            // Output positions n+32 to n+63 (q2 values)
            for l in 0..32 {
                if result.len() >= num_elements {
                    break;
                }
                let is = l / 16;
                let q2 = ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32;
                result.push(d * (sc[is + 2] as f32) * (q2 as f32));
            }

            // Output positions n+64 to n+95 (q3 values)
            for l in 0..32 {
                if result.len() >= num_elements {
                    break;
                }
                let is = l / 16;
                let q3 = ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32;
                result.push(d * (sc[is + 4] as f32) * (q3 as f32));
            }

            // Output positions n+96 to n+127 (q4 values)
            for l in 0..32 {
                if result.len() >= num_elements {
                    break;
                }
                let is = l / 16;
                let q4 = ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32;
                result.push(d * (sc[is + 6] as f32) * (q4 as f32));
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Map APR dtype string to GGML quantization type ID.
///
/// These IDs are used by `load_quantized_weights_with_type()` to select
/// the correct GPU dequantization kernel (Q4K GEMV, Q6K GEMV, etc.).
#[inline]
pub fn dtype_to_ggml_qtype(dtype: &str) -> Option<u32> {
    match dtype {
        "Q4_K" | "q4_k" => Some(12), // GGML_TYPE_Q4_K
        "Q5_K" | "q5_k" => Some(13), // GGML_TYPE_Q5_K
        "Q6_K" | "q6_k" => Some(14), // GGML_TYPE_Q6_K
        "Q8_0" | "q8_0" => Some(8),  // GGML_TYPE_Q8_0
        "Q4_0" | "q4_0" => Some(2),  // GGML_TYPE_Q4_0
        "Q4_1" | "q4_1" => Some(3),  // GGML_TYPE_Q4_1
        "Q5_0" | "q5_0" => Some(6),  // GGML_TYPE_Q5_0
        _ => None,                   // F32/F16 are not quantized
    }
}

/// Check if dtype is a quantized format that can use GPU dequant kernels.
#[inline]
pub fn is_quantized_dtype(dtype: &str) -> bool {
    dtype_to_ggml_qtype(dtype).is_some()
}
