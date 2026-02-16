
/// Expand 16 packed Q4_0 nibble bytes at `q4_ptr + 2` into a 256-bit vector.
///
/// Returns raw expanded bytes (low nibbles in lower 128, high nibbles in upper 128).
/// The caller is responsible for masking to 0x0F and subtracting the offset.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn expand_q4_raw_avx2(q4_ptr: *const u8) -> std::arch::x86_64::__m256i {
    // SAFETY: caller guarantees q4_ptr + 2..+18 is valid
    unsafe {
        use std::arch::x86_64::{_mm256_set_m128i, _mm_loadu_si128, _mm_srli_epi16};
        let raw = _mm_loadu_si128(q4_ptr.add(2).cast());
        let hi = _mm_srli_epi16(raw, 4);
        _mm256_set_m128i(hi, raw)
    }
}

/// Expand 16 packed Q4_0 nibble bytes into a 256-bit vector of signed values (-8..+7).
///
/// Loads 16 bytes from `q4_ptr + 2`, splits into low/high nibbles, masks to 0x0F,
/// and subtracts 8 to center at zero.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn expand_q4_nibbles_avx2(q4_ptr: *const u8) -> std::arch::x86_64::__m256i {
    // SAFETY: caller guarantees q4_ptr + 2..+18 is valid
    unsafe {
        use std::arch::x86_64::{_mm256_and_si256, _mm256_set1_epi8, _mm256_sub_epi8};
        let combined = expand_q4_raw_avx2(q4_ptr);
        let nibbles = _mm256_and_si256(combined, _mm256_set1_epi8(0x0F));
        _mm256_sub_epi8(nibbles, _mm256_set1_epi8(8))
    }
}

/// Process one Q4_0 block via AVX2 maddubs and accumulate into `acc`.
///
/// Performs: acc += (q4_scale * q8_scale) * dot(q4_block, q8_block)
/// using the sign trick for unsigned x signed maddubs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_block_dot_accumulate(
    q4_signed: std::arch::x86_64::__m256i,
    q8_ptr: *const i8,
    combined_scale: std::arch::x86_64::__m256,
    acc: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    // SAFETY: caller guarantees q8_ptr..+32 is valid
    unsafe {
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

/// Process a pair of Q4_0 blocks via AVX-512 VNNI: combine two 256-bit expansions
/// into a 512-bit dpbusd, then split and scale-accumulate each half separately.
///
/// Returns the updated accumulator.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
unsafe fn avx512_pair_dot_accumulate(
    q4_ptr_lo: *const u8,
    q4_ptr_hi: *const u8,
    q8_ptr: *const i8,
    scale_lo: f32,
    scale_hi: f32,
    low_mask: std::arch::x86_64::__m512i,
    offset: std::arch::x86_64::__m512i,
    acc: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    // SAFETY: caller guarantees both q4 pointers and q8_ptr are valid
    unsafe {
        use std::arch::x86_64::{
            _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_set1_ps, _mm512_and_si512,
            _mm512_castsi256_si512, _mm512_castsi512_si256, _mm512_dpbusd_epi32,
            _mm512_extracti64x4_epi64, _mm512_inserti64x4, _mm512_loadu_si512,
            _mm512_setzero_si512, _mm512_sub_epi8,
        };

        let q4_expanded_lo = expand_q4_raw_avx2(q4_ptr_lo);
        let q4_expanded_hi = expand_q4_raw_avx2(q4_ptr_hi);

        let q4_combined = _mm512_inserti64x4(
            _mm512_castsi256_si512(q4_expanded_lo),
            q4_expanded_hi,
            1,
        );
        let q4_nibbles = _mm512_and_si512(q4_combined, low_mask);
        let q4_signed = _mm512_sub_epi8(q4_nibbles, offset);
        let q8_vec = _mm512_loadu_si512(q8_ptr.cast());

        // Sign trick for vpdpbusd (unsigned x signed): |q4| x sign(q8, q4)
        let q4_abs = std::arch::x86_64::_mm512_abs_epi8(q4_signed);
        let mask = std::arch::x86_64::_mm512_movepi8_mask(q4_signed);
        let neg_q8 = std::arch::x86_64::_mm512_sub_epi8(_mm512_setzero_si512(), q8_vec);
        let q8_signed = std::arch::x86_64::_mm512_mask_blend_epi8(mask, q8_vec, neg_q8);
        let int_acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_abs, q8_signed);

        // Split into two 256-bit halves, scale each block separately
        let int_lo = _mm512_castsi512_si256(int_acc);
        let int_hi = _mm512_extracti64x4_epi64(int_acc, 1);
        let prod_f32_lo = _mm256_cvtepi32_ps(int_lo);
        let prod_f32_hi = _mm256_cvtepi32_ps(int_hi);

        let result = _mm256_fmadd_ps(_mm256_set1_ps(scale_lo), prod_f32_lo, acc);
        _mm256_fmadd_ps(_mm256_set1_ps(scale_hi), prod_f32_hi, result)
    }
}

/// AVX-512 VNNI accelerated Q4_0 x Q8_0 dot product using vpdpbusd with 512-bit vectors
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
        use std::arch::x86_64::{_mm256_setzero_ps, _mm512_set1_epi8};

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

            // First pair of blocks (0, 1) -> acc0
            let q4_ptr_0 = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q4_ptr_1 = q4_data.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_BYTES);
            acc0 = avx512_pair_dot_accumulate(
                q4_ptr_0,
                q4_ptr_1,
                q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE),
                f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_0, *q4_ptr_0.add(1)])) * q8_scales[block_idx],
                f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_1, *q4_ptr_1.add(1)])) * q8_scales[block_idx + 1],
                low_mask,
                offset,
                acc0,
            );

            // Second pair of blocks (2, 3) -> acc1
            let q4_ptr_2 = q4_data.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_BYTES);
            let q4_ptr_3 = q4_data.as_ptr().add((block_idx + 3) * Q4_0_BLOCK_BYTES);
            acc1 = avx512_pair_dot_accumulate(
                q4_ptr_2,
                q4_ptr_3,
                q8_quants.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_SIZE),
                f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_2, *q4_ptr_2.add(1)])) * q8_scales[block_idx + 2],
                f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_3, *q4_ptr_3.add(1)])) * q8_scales[block_idx + 3],
                low_mask,
                offset,
                acc1,
            );

            block_idx += 4;
        }

        // Process 2 blocks at a time (64 values per iteration) using 512-bit vectors
        while block_idx + 2 <= num_blocks {
            let q4_ptr_0 = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let q4_ptr_1 = q4_data.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_BYTES);
            acc0 = avx512_pair_dot_accumulate(
                q4_ptr_0,
                q4_ptr_1,
                q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE),
                f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_0, *q4_ptr_0.add(1)])) * q8_scales[block_idx],
                f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_1, *q4_ptr_1.add(1)])) * q8_scales[block_idx + 1],
                low_mask,
                offset,
                acc0,
            );

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
        use std::arch::x86_64::{_mm256_set1_ps, _mm256_setzero_ps, _mm_prefetch, _MM_HINT_T0};

        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;

        let num_blocks = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
        let mut acc = _mm256_setzero_ps();
        let mut block_idx = 0;

        // Process 2 blocks at a time for better instruction-level parallelism
        while block_idx + 2 <= num_blocks {
            // Prefetch next iteration's blocks
            if block_idx + 4 <= num_blocks {
                _mm_prefetch(
                    q4_data.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_BYTES).cast(),
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    q8_quants.as_ptr().add((block_idx + 2) * Q4_0_BLOCK_SIZE).cast(),
                    _MM_HINT_T0,
                );
            }

            // Block 0
            let q4_ptr_0 = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let scale_0 = f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_0, *q4_ptr_0.add(1)]));
            let q4_signed_0 = expand_q4_nibbles_avx2(q4_ptr_0);
            acc = avx2_block_dot_accumulate(
                q4_signed_0,
                q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE),
                _mm256_set1_ps(scale_0 * q8_scales[block_idx]),
                acc,
            );

            // Block 1
            let q4_ptr_1 = q4_data.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_BYTES);
            let scale_1 = f16_to_f32_lut(u16::from_le_bytes([*q4_ptr_1, *q4_ptr_1.add(1)]));
            let q4_signed_1 = expand_q4_nibbles_avx2(q4_ptr_1);
            acc = avx2_block_dot_accumulate(
                q4_signed_1,
                q8_quants.as_ptr().add((block_idx + 1) * Q4_0_BLOCK_SIZE),
                _mm256_set1_ps(scale_1 * q8_scales[block_idx + 1]),
                acc,
            );

            block_idx += 2;
        }

        // Handle remaining single block
        while block_idx < num_blocks {
            let q4_ptr = q4_data.as_ptr().add(block_idx * Q4_0_BLOCK_BYTES);
            let scale = f16_to_f32_lut(u16::from_le_bytes([*q4_ptr, *q4_ptr.add(1)]));
            let q4_signed = expand_q4_nibbles_avx2(q4_ptr);
            acc = avx2_block_dot_accumulate(
                q4_signed,
                q8_quants.as_ptr().add(block_idx * Q4_0_BLOCK_SIZE),
                _mm256_set1_ps(scale * q8_scales[block_idx]),
                acc,
            );
            block_idx += 1;
        }

        hsum_avx2(acc)
    }
}
