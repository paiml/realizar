// PMAT-298: True AVX-512 VNNI Q4K x Q8K dot product helper.
// Uses 512-bit operations (_mm512_*) instead of 256-bit AVX2.
// Processes 64 bytes per operation (2x throughput).
// Key instruction: _mm512_dpbusd_epi32 (VPDPBUSD)

/// Compute Q4xQ8 dot products for 8 blocks using 512-bit AVX-512 VNNI.
///
/// Each super-block has 256 Q4K nibble-packed bytes (128 bytes) and 256 Q8K values.
/// We process 2 chunks at a time using 512-bit registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn compute_q4_q8_dots_8blocks_avx512(
    qs_ptr: *const u8,
    q8_ptr: *const i8,
) -> [i32; 8] {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let nibble_mask = _mm512_set1_epi8(0x0F_i8);

    // Load ALL Q4K data (128 bytes) into 2 x 512-bit registers
    let q4_01 = _mm512_loadu_si512(qs_ptr.cast::<__m512i>());
    let q4_23 = _mm512_loadu_si512(qs_ptr.add(64).cast::<__m512i>());

    // Load ALL Q8K data (256 bytes) into 4 x 512-bit registers
    let q8_01 = _mm512_loadu_si512(q8_ptr.cast::<__m512i>());
    let q8_23 = _mm512_loadu_si512(q8_ptr.add(64).cast::<__m512i>());
    let q8_45 = _mm512_loadu_si512(q8_ptr.add(128).cast::<__m512i>());
    let q8_67 = _mm512_loadu_si512(q8_ptr.add(192).cast::<__m512i>());

    // Extract low and high nibbles from Q4K
    let q4_01_lo = _mm512_and_si512(q4_01, nibble_mask);
    let q4_01_hi = _mm512_and_si512(_mm512_srli_epi16(q4_01, 4), nibble_mask);
    let q4_23_lo = _mm512_and_si512(q4_23, nibble_mask);
    let q4_23_hi = _mm512_and_si512(_mm512_srli_epi16(q4_23, 4), nibble_mask);

    // VPDPBUSD: Q4 (unsigned) × Q8 (signed) dot product
    // Each instruction processes 64 bytes and accumulates into 16 i32 lanes
    let dot_01_lo = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_01_lo, q8_01);
    let dot_01_hi = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_01_hi, q8_23);
    let dot_23_lo = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_23_lo, q8_45);
    let dot_23_hi = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_23_hi, q8_67);

    // Each 512-bit dot result has 16 i32 partial sums.
    // We need to reduce each to a single i32 per Q8 block (32 values).
    // Each 512-bit register covers 64 values = 2 Q8 blocks.
    // Split: low 256 bits = block 0, high 256 bits = block 1.

    let mut dots = [0i32; 8];

    // Block 0,1: from dot_01_lo (Q4 chunks 0,1 low nibbles × Q8 chunks 0,1)
    let lo_256 = _mm512_castsi512_si256(dot_01_lo);
    let hi_256 = _mm512_extracti64x4_epi64(dot_01_lo, 1);
    dots[0] = hsum_256(lo_256);
    dots[1] = hsum_256(hi_256);

    // Block 2,3: from dot_01_hi (Q4 chunks 0,1 high nibbles × Q8 chunks 2,3)
    let lo_256 = _mm512_castsi512_si256(dot_01_hi);
    let hi_256 = _mm512_extracti64x4_epi64(dot_01_hi, 1);
    dots[2] = hsum_256(lo_256);
    dots[3] = hsum_256(hi_256);

    // Block 4,5: from dot_23_lo
    let lo_256 = _mm512_castsi512_si256(dot_23_lo);
    let hi_256 = _mm512_extracti64x4_epi64(dot_23_lo, 1);
    dots[4] = hsum_256(lo_256);
    dots[5] = hsum_256(hi_256);

    // Block 6,7: from dot_23_hi
    let lo_256 = _mm512_castsi512_si256(dot_23_hi);
    let hi_256 = _mm512_extracti64x4_epi64(dot_23_hi, 1);
    dots[6] = hsum_256(lo_256);
    dots[7] = hsum_256(hi_256);

    dots
}

/// Horizontal sum of 8 i32 values in a 256-bit register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_10_10));
    _mm_cvtsi128_si32(sum32)
}

/// Compute Q8K block sums using 512-bit operations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
pub(crate) unsafe fn compute_q8_sums_avx512(q8_ptr: *const i8) -> [i32; 8] {
    use std::arch::x86_64::*;

    let ones = _mm512_set1_epi8(1);
    let mut sums = [0i32; 8];

    // Process 4 chunks of 64 bytes each (256 values total)
    for chunk in 0..4 {
        let q8 = _mm512_loadu_si512(q8_ptr.add(chunk * 64).cast::<__m512i>());
        // dpbusd with ones gives sum of signed bytes
        let sum_vec = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, q8);

        // Each 512-bit sum covers 2 Q8 blocks (32 values each)
        let lo_256 = _mm512_castsi512_si256(sum_vec);
        let hi_256 = _mm512_extracti64x4_epi64(sum_vec, 1);
        sums[chunk * 2] = hsum_256(lo_256);
        sums[chunk * 2 + 1] = hsum_256(hi_256);
    }

    sums
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_available() {
        // Just check if the test compiles and AVX-512 detection works
        let has_avx512 =
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni");
        eprintln!("AVX-512 VNNI available: {has_avx512}");
    }
}
