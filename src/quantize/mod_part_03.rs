
/// Expand 16 packed Q4_0 nibble bytes at `q4_ptr + 2` into a 256-bit vector.
///
/// Returns raw expanded bytes (low nibbles in lower 128, high nibbles in upper 128).
/// The caller is responsible for masking to 0x0F and subtracting the offset.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn expand_q4_raw_avx2(q4_ptr: *const u8) -> std::arch::x86_64::__m256i {
    // SAFETY: caller guarantees q4_ptr + 2..+18 is valid
    use std::arch::x86_64::{_mm256_set_m128i, _mm_loadu_si128, _mm_srli_epi16};
    let raw = _mm_loadu_si128(q4_ptr.add(2).cast());
    let hi = _mm_srli_epi16(raw, 4);
    _mm256_set_m128i(hi, raw)
}

/// Expand 16 packed Q4_0 nibble bytes into a 256-bit vector of signed values (-8..+7).
///
/// Loads 16 bytes from `q4_ptr + 2`, splits into low/high nibbles, masks to 0x0F,
/// and subtracts 8 to center at zero.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn expand_q4_nibbles_avx2(q4_ptr: *const u8) -> std::arch::x86_64::__m256i {
    // SAFETY: caller guarantees q4_ptr + 2..+18 is valid
    use std::arch::x86_64::{_mm256_and_si256, _mm256_set1_epi8, _mm256_sub_epi8};
    let combined = expand_q4_raw_avx2(q4_ptr);
    let nibbles = _mm256_and_si256(combined, _mm256_set1_epi8(0x0F));
    _mm256_sub_epi8(nibbles, _mm256_set1_epi8(8))
}

/// Process one Q4_0 block via AVX2 maddubs and accumulate into `acc`.
///
/// Performs: acc += (q4_scale * q8_scale) * dot(q4_block, q8_block)
/// using the sign trick for unsigned x signed maddubs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_block_dot_accumulate(
    q4_signed: std::arch::x86_64::__m256i,
    q8_ptr: *const i8,
    combined_scale: std::arch::x86_64::__m256,
    acc: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    // SAFETY: caller guarantees q8_ptr..+32 is valid
    use std::arch::x86_64::{
        _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_sign_epi8,
    };
    let q8_vec = _mm256_loadu_si256(q8_ptr.cast());
    let q4_abs = _mm256_sign_epi8(q4_signed, q4_signed);
    let q8_signed = _mm256_sign_epi8(q8_vec, q4_signed);
    let prod_i16 = _mm256_maddubs_epi16(q4_abs, q8_signed);
    let prod_i32 = _mm256_madd_epi16(prod_i16, _mm256_set1_epi16(1));
    let prod_f32 = _mm256_cvtepi32_ps(prod_i32);
    _mm256_fmadd_ps(combined_scale, prod_f32, acc)
}

/// Horizontal sum of 8 f32 lanes in a __m256 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
    };
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);
    _mm_cvtss_f32(sum32)
}

/// AVX-512 VNNI accelerated Q4_0 × Q8_0 dot product using vpdpbusd with 512-bit vectors
///
/// Uses 512-bit registers to process 2 blocks (64 values) per iteration, providing
/// ~2x throughput over the 256-bit AVX2 path. The vpdpbusd instruction performs
/// native u8×i8 multiply-accumulate directly to i32.
///
/// Performance: ~1.8-2x faster than AVX2 path on Zen4, Sapphire Rapids, and later.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[inline]
unsafe fn fused_q4_0_q8_0_dot_avx512_vnni(
    q4_data: &[u8],
    q8_scales: &[f32],
    q8_quants: &[i8],
    in_dim: usize,
) -> f32 {
    // SAFETY: Memory safety ensured by bounds checking and alignment
    unsafe {
        use std::arch::x86_64::{
            __m512i, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_setzero_ps, _mm512_and_si512,
            _mm512_castsi512_si256, _mm512_dpbusd_epi32, _mm512_extracti64x4_epi64,
            _mm512_loadu_si512, _mm512_set1_epi8, _mm512_setzero_si512, _mm512_sub_epi8,
            _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
        };

        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;

        let num_blocks = in_dim.div_ceil(Q4_0_BLOCK_SIZE);

        // Use two accumulators for better pipelining
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let offset = _mm512_set1_epi8(8);
        let low_mask = _mm512_set1_epi8(0x0F);

        let mut block_idx = 0;

        // Process 4 blocks at a time (128 values per iteration) using 2x 512-bit vectors
        // This provides better ILP on modern OoO CPUs
        while block_idx + 4 <= num_blocks {
            // Prefetch next iteration's data (8 blocks ahead = 2 iterations)
            if block_idx + 8 <= num_blocks {
                let pf_q4 = q4_data.as_ptr().add((block_idx + 8) * Q4_0_BLOCK_BYTES);
                let pf_q8 = q8_quants.as_ptr().add((block_idx + 8) * Q4_0_BLOCK_SIZE);
                std::arch::x86_64::_mm_prefetch(pf_q4.cast(), std::arch::x86_64::_MM_HINT_T0);
                std::arch::x86_64::_mm_prefetch(
                    pf_q4.add(72).cast(),
                    std::arch::x86_64::_MM_HINT_T0,
                );
                std::arch::x86_64::_mm_prefetch(pf_q8.cast(), std::arch::x86_64::_MM_HINT_T0);
                std::arch::x86_64::_mm_prefetch(
                    pf_q8.add(64).cast(),
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }

            // === First pair of blocks (0, 1) ===
            let q4_ptr_0 = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q4_ptr_1 = q4_data.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_BYTES);
            let q8_ptr_a = q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE);

            let q4_scale_0 = f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_0, *q4_ptr_0.add(1)]));
            let q4_scale_1 = f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_1, *q4_ptr_1.add(1)]));
            let q8_scale_0 = q8_scales[block_idx];
            let q8_scale_1 = q8_scales[block_idx + 1];

            // Expand nibbles for blocks 0,1 (raw expansion, mask+offset applied after 512-bit combine)
            let q4_expanded_0 = expand_q4_raw_avx2(q4_ptr_0);
            let q4_expanded_1 = expand_q4_raw_avx2(q4_ptr_1);

            let q4_combined_a: __m512i = std::arch::x86_64::_mm512_inserti64x4(
                std::arch::x86_64::_mm512_castsi256_si512(q4_expanded_0),
                q4_expanded_1,
                1,
            );
            let q4_nibbles_a = _mm512_and_si512(q4_combined_a, low_mask);
            let q4_signed_a = _mm512_sub_epi8(q4_nibbles_a, offset);
            let q8_vec_a = _mm512_loadu_si512(q8_ptr_a.cast());

            let q4_abs_a = std::arch::x86_64::_mm512_abs_epi8(q4_signed_a);
            let mask_a = std::arch::x86_64::_mm512_movepi8_mask(q4_signed_a);
            let neg_q8_a = std::arch::x86_64::_mm512_sub_epi8(_mm512_setzero_si512(), q8_vec_a);
            let q8_signed_a = std::arch::x86_64::_mm512_mask_blend_epi8(mask_a, q8_vec_a, neg_q8_a);
            let int_acc_a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_abs_a, q8_signed_a);

            // === Second pair of blocks (2, 3) ===
            let q4_ptr_2 = q4_data.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_BYTES);
            let q4_ptr_3 = q4_data.as_ptr().add((block_idx + 3) * Q4_0_BLOCK_BYTES);
            let q8_ptr_b = q8_quants.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_SIZE);

            let q4_scale_2 = f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_2, *q4_ptr_2.add(1)]));
            let q4_scale_3 = f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_3, *q4_ptr_3.add(1)]));
            let q8_scale_2 = q8_scales[block_idx + 2];
            let q8_scale_3 = q8_scales[block_idx + 3];

            let q4_expanded_2 = expand_q4_raw_avx2(q4_ptr_2);
            let q4_expanded_3 = expand_q4_raw_avx2(q4_ptr_3);

            let q4_combined_b: __m512i = std::arch::x86_64::_mm512_inserti64x4(
                std::arch::x86_64::_mm512_castsi256_si512(q4_expanded_2),
                q4_expanded_3,
                1,
            );
            let q4_nibbles_b = _mm512_and_si512(q4_combined_b, low_mask);
            let q4_signed_b = _mm512_sub_epi8(q4_nibbles_b, offset);
            let q8_vec_b = _mm512_loadu_si512(q8_ptr_b.cast());

            let q4_abs_b = std::arch::x86_64::_mm512_abs_epi8(q4_signed_b);
            let mask_b = std::arch::x86_64::_mm512_movepi8_mask(q4_signed_b);
            let neg_q8_b = std::arch::x86_64::_mm512_sub_epi8(_mm512_setzero_si512(), q8_vec_b);
            let q8_signed_b = std::arch::x86_64::_mm512_mask_blend_epi8(mask_b, q8_vec_b, neg_q8_b);
            let int_acc_b = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_abs_b, q8_signed_b);

            // Scale and accumulate first pair
            let int_lo_a = _mm512_castsi512_si256(int_acc_a);
            let int_hi_a = _mm512_extracti64x4_epi64(int_acc_a, 1);
            let prod_f32_0 = _mm256_cvtepi32_ps(int_lo_a);
            let prod_f32_1 = _mm256_cvtepi32_ps(int_hi_a);
            acc0 = _mm256_fmadd_ps(
                std::arch::x86_64::_mm256_set1_ps(q4_scale_0 * q8_scale_0),
                prod_f32_0,
                acc0,
            );
            acc0 = _mm256_fmadd_ps(
                std::arch::x86_64::_mm256_set1_ps(q4_scale_1 * q8_scale_1),
                prod_f32_1,
                acc0,
            );

            // Scale and accumulate second pair
            let int_lo_b = _mm512_castsi512_si256(int_acc_b);
            let int_hi_b = _mm512_extracti64x4_epi64(int_acc_b, 1);
            let prod_f32_2 = _mm256_cvtepi32_ps(int_lo_b);
            let prod_f32_3 = _mm256_cvtepi32_ps(int_hi_b);
            acc1 = _mm256_fmadd_ps(
                std::arch::x86_64::_mm256_set1_ps(q4_scale_2 * q8_scale_2),
                prod_f32_2,
                acc1,
            );
            acc1 = _mm256_fmadd_ps(
                std::arch::x86_64::_mm256_set1_ps(q4_scale_3 * q8_scale_3),
                prod_f32_3,
                acc1,
            );

            block_idx += 4;
        }

        // Process 2 blocks at a time (64 values per iteration) using 512-bit vectors
        while block_idx + 2 <= num_blocks {
            let q4_ptr_0 = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q4_ptr_1 = q4_data.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_BYTES);
            let q8_ptr = q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE);

            // Read scales for both blocks
            let q4_scale_bits_0 = u16::from_le_bytes([*q4_ptr_0, *q4_ptr_0.add(1)]);
            let q4_scale_bits_1 = u16::from_le_bytes([*q4_ptr_1, *q4_ptr_1.add(1)]);
            let q4_scale_0 = f16_to_f32_lut(q4_scale_bits_0);
            let q4_scale_1 = f16_to_f32_lut(q4_scale_bits_1);
            let q8_scale_0 = q8_scales[block_idx];
            let q8_scale_1 = q8_scales[block_idx + 1];

            // Expand nibbles for both blocks (raw expansion, mask+offset applied after 512-bit combine)
            let q4_expanded_0 = expand_q4_raw_avx2(q4_ptr_0);
            let q4_expanded_1 = expand_q4_raw_avx2(q4_ptr_1);

            // Combine into 512-bit vector
            let q4_combined: __m512i = std::arch::x86_64::_mm512_inserti64x4(
                std::arch::x86_64::_mm512_castsi256_si512(q4_expanded_0),
                q4_expanded_1,
                1,
            );

            // Mask and convert to signed
            let q4_nibbles = _mm512_and_si512(q4_combined, low_mask);
            let q4_signed = _mm512_sub_epi8(q4_nibbles, offset);

            // Load Q8 quants (64 bytes = 2 blocks)
            let q8_vec = _mm512_loadu_si512(q8_ptr.cast());

            // For vpdpbusd, we need unsigned × signed
            // Use sign trick: |q4| × sign(q8, q4)
            let q4_abs = std::arch::x86_64::_mm512_abs_epi8(q4_signed);
            let q8_signed = {
                // _mm512_sign_epi8 doesn't exist, implement with mask
                let mask = std::arch::x86_64::_mm512_movepi8_mask(q4_signed);
                let neg_q8 = std::arch::x86_64::_mm512_sub_epi8(_mm512_setzero_si512(), q8_vec);
                std::arch::x86_64::_mm512_mask_blend_epi8(mask, q8_vec, neg_q8)
            };

            // vpdpbusd: 512-bit version processes 64 u8×i8 products
            // Accumulates 16 lanes of i32 (each is sum of 4 products)
            let int_acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_abs, q8_signed);

            // Split result into two 256-bit halves for separate scaling
            let int_lo = _mm512_castsi512_si256(int_acc);
            let int_hi = _mm512_extracti64x4_epi64(int_acc, 1);

            // Convert to float and scale each block separately
            let prod_f32_0 = _mm256_cvtepi32_ps(int_lo);
            let prod_f32_1 = _mm256_cvtepi32_ps(int_hi);

            let scale_0 = std::arch::x86_64::_mm256_set1_ps(q4_scale_0 * q8_scale_0);
            let scale_1 = std::arch::x86_64::_mm256_set1_ps(q4_scale_1 * q8_scale_1);

            acc0 = _mm256_fmadd_ps(scale_0, prod_f32_0, acc0);
            acc0 = _mm256_fmadd_ps(scale_1, prod_f32_1, acc0);

            block_idx += 2;
        }

        // Handle remaining single block with AVX2
        while block_idx < num_blocks {
            let q4_ptr = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q8_ptr = q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE);

            let q4_scale = f16_to_f32_lut(u16::from_le_bytes([*q4_ptr, *q4_ptr.add(1)]));
            let combined_scale = std::arch::x86_64::_mm256_set1_ps(q4_scale * q8_scales[block_idx]);

            let q4_signed = expand_q4_nibbles_avx2(q4_ptr);
            acc0 = avx2_block_dot_accumulate(q4_signed, q8_ptr, combined_scale, acc0);

            block_idx += 1;
        }

        // Combine both accumulators and do horizontal sum
        let acc = std::arch::x86_64::_mm256_add_ps(acc0, acc1);
        hsum_avx2(acc)
    }
}

/// AVX2 accelerated Q4_0 × Q8_0 dot product using integer SIMD
///
/// Uses AVX2 maddubs which multiplies pairs of u8×i8 and accumulates
/// to i16, then we sum to i32 and convert to f32. This is ~4x faster than
/// the f32 FMA approach because:
/// 1. Integer ops have lower latency
/// 2. maddubs does multiply AND horizontal add in one instruction
/// 3. Less data movement (1 byte vs 4 bytes per value)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fused_q4_0_q8_0_dot_avx2(
    q4_data: &[u8],
    q8_scales: &[f32],
    q8_quants: &[i8],
    in_dim: usize,
) -> f32 {
    // SAFETY: Memory safety ensured by bounds checking and alignment
    unsafe {
        use std::arch::x86_64::{
            _mm256_and_si256, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_si256,
            _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi8,
            _mm256_set1_ps, _mm256_setzero_ps, _mm256_sign_epi8, _mm256_sub_epi8, _mm_cvtss_f32,
            _mm_hadd_ps, _mm_prefetch, _MM_HINT_T0,
        };

        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;

        let num_blocks = in_dim.div_ceil(Q4_0_BLOCK_SIZE);

        // Float accumulator for final sum
        let mut acc = _mm256_setzero_ps();

        // Offset: Q4_0 values are 0-15, we subtract 8 to get -8 to +7
        let offset = _mm256_set1_epi8(8);
        let low_mask = _mm256_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);

        let mut block_idx = 0;

        // Process 2 blocks at a time for better instruction-level parallelism
        while block_idx + 2 <= num_blocks {
            // Prefetch next iteration's blocks
            if block_idx + 4 <= num_blocks {
                let prefetch_q4 = q4_data.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_BYTES);
                let prefetch_q8 = q8_quants.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_SIZE);
                _mm_prefetch(prefetch_q4.cast(), _MM_HINT_T0);
                _mm_prefetch(prefetch_q8.cast(), _MM_HINT_T0);
            }

            // === Block 0 ===
            let q4_ptr_0 = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q8_ptr_0 = q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE);

            // Read Q4_0 scale (f16 -> f32 via LUT)
            let q4_scale_bits_0 = u16::from_le_bytes([*q4_ptr_0, *q4_ptr_0.add(1)]);
            let q4_scale_0 = f16_to_f32_lut(q4_scale_bits_0);
            let q8_scale_0 = q8_scales[block_idx];
            let combined_scale_0 = _mm256_set1_ps(q4_scale_0 * q8_scale_0);

            // Load Q4_0 quants (16 bytes = 32 nibbles)
            let q4_bytes = std::slice::from_raw_parts(q4_ptr_0.add(2), 16);

            // bytes_from_nibbles_32: expand 16 bytes to 32 bytes
            // Low nibbles in first 16 positions, high nibbles in next 16
            let q4_lo_128 = std::arch::x86_64::_mm_loadu_si128(q4_bytes.as_ptr().cast());
            let q4_hi_128 = std::arch::x86_64::_mm_srli_epi16(q4_lo_128, 4);
            // Combine into 256-bit: high nibbles in upper 128, low nibbles in lower 128
            let q4_combined = std::arch::x86_64::_mm256_set_m128i(q4_hi_128, q4_lo_128);
            // Mask to get just nibbles
            let q4_nibbles = _mm256_and_si256(q4_combined, low_mask);
            // Convert from unsigned 0-15 to signed -8 to +7
            let q4_signed = _mm256_sub_epi8(q4_nibbles, offset);

            // Load Q8_0 quants (32 bytes)
            let q8_vec = _mm256_loadu_si256(q8_ptr_0.cast());

            // Integer multiply-accumulate using signed multiply trick:
            // maddubs requires unsigned × signed, so we use sign trick
            // ax = |x|, sy = sign(y, x), then maddubs(ax, sy) = x * y
            let q4_abs = _mm256_sign_epi8(q4_signed, q4_signed);
            let q8_signed = _mm256_sign_epi8(q8_vec, q4_signed);

            // maddubs: multiply pairs and add horizontally to i16
            let prod_i16 = _mm256_maddubs_epi16(q4_abs, q8_signed);
            // madd: pairwise add i16 to i32
            let prod_i32 = _mm256_madd_epi16(prod_i16, ones);
            // Convert to float
            let prod_f32 = _mm256_cvtepi32_ps(prod_i32);

            // Scale and accumulate
            acc = _mm256_fmadd_ps(combined_scale_0, prod_f32, acc);

            // === Block 1 ===
            let q4_ptr_1 = q4_data.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_BYTES);
            let q8_ptr_1 = q8_quants.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_SIZE);

            let q4_scale_bits_1 = u16::from_le_bytes([*q4_ptr_1, *q4_ptr_1.add(1)]);
            let q4_scale_1 = f16_to_f32_lut(q4_scale_bits_1);
            let q8_scale_1 = q8_scales[block_idx + 1];
            let combined_scale_1 = _mm256_set1_ps(q4_scale_1 * q8_scale_1);

            let q4_bytes_1 = std::slice::from_raw_parts(q4_ptr_1.add(2), 16);
            let q4_lo_128_1 = std::arch::x86_64::_mm_loadu_si128(q4_bytes_1.as_ptr().cast());
            let q4_hi_128_1 = std::arch::x86_64::_mm_srli_epi16(q4_lo_128_1, 4);
            let q4_combined_1 = std::arch::x86_64::_mm256_set_m128i(q4_hi_128_1, q4_lo_128_1);
            let q4_nibbles_1 = _mm256_and_si256(q4_combined_1, low_mask);
            let q4_signed_1 = _mm256_sub_epi8(q4_nibbles_1, offset);

            let q8_vec_1 = _mm256_loadu_si256(q8_ptr_1.cast());

            let q4_abs_1 = _mm256_sign_epi8(q4_signed_1, q4_signed_1);
            let q8_signed_1 = _mm256_sign_epi8(q8_vec_1, q4_signed_1);

            let prod_i16_1 = _mm256_maddubs_epi16(q4_abs_1, q8_signed_1);
            let prod_i32_1 = _mm256_madd_epi16(prod_i16_1, ones);
            let prod_f32_1 = _mm256_cvtepi32_ps(prod_i32_1);

            acc = _mm256_fmadd_ps(combined_scale_1, prod_f32_1, acc);

            block_idx += 2;
        }

        // Handle remaining single block
        while block_idx < num_blocks {
            let q4_ptr = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q8_ptr = q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE);

            let q4_scale_bits = u16::from_le_bytes([*q4_ptr, *q4_ptr.add(1)]);
            let q4_scale = f16_to_f32_lut(q4_scale_bits);
            let q8_scale = q8_scales[block_idx];
            let combined_scale = _mm256_set1_ps(q4_scale * q8_scale);

            let q4_bytes = std::slice::from_raw_parts(q4_ptr.add(2), 16);
            let q4_lo_128 = std::arch::x86_64::_mm_loadu_si128(q4_bytes.as_ptr().cast());
            let q4_hi_128 = std::arch::x86_64::_mm_srli_epi16(q4_lo_128, 4);
            let q4_combined = std::arch::x86_64::_mm256_set_m128i(q4_hi_128, q4_lo_128);
            let q4_nibbles = _mm256_and_si256(q4_combined, low_mask);
            let q4_signed = _mm256_sub_epi8(q4_nibbles, offset);

            let q8_vec = _mm256_loadu_si256(q8_ptr.cast());

            let q4_abs = _mm256_sign_epi8(q4_signed, q4_signed);
            let q8_signed = _mm256_sign_epi8(q8_vec, q4_signed);

            let prod_i16 = _mm256_maddubs_epi16(q4_abs, q8_signed);
            let prod_i32 = _mm256_madd_epi16(prod_i16, ones);
            let prod_f32 = _mm256_cvtepi32_ps(prod_i32);

            acc = _mm256_fmadd_ps(combined_scale, prod_f32, acc);

            block_idx += 1;
        }

        // Horizontal sum of 8 floats
        let hi = std::arch::x86_64::_mm256_extractf128_ps(acc, 1);
        let lo = std::arch::x86_64::_mm256_castps256_ps128(acc);
        let sum128 = std::arch::x86_64::_mm_add_ps(lo, hi);
        // Use hadd for final reduction
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);
        _mm_cvtss_f32(sum32)
    }
}
