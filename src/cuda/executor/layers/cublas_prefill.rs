//! PMAT-024/026/031: cuBLAS GEMM for prefill
//!
//! Prefill path evolution:
//!   v1 (PMAT-024): Dequant Q4K/Q6K → FP32 scratch → cuBLAS SGEMM (per-request dequant)
//!   v2 (PMAT-031): Cache FP16 weights + cuBLAS HGEMM with tensor cores
//!
//! PMAT-031 eliminates per-request dequant cost and uses tensor cores for 2-4x speedup.
//! On first prefill, weights are dequantized to FP32, converted to FP16, and cached.
//! Subsequent prefills use cached FP16 weights directly with HGEMM.

use super::super::*;

/// Minimum M (batch/sequence length) to use cuBLAS GEMM instead of batched GEMV.
/// Below this threshold, batched GEMV is faster due to lower overhead.
///
/// Override with CUBLAS_GEMM_THRESHOLD env var (e.g., =1 for HGEMM decode on high-BW GPUs).
/// Minimum batch size M for cuBLAS SGEMM to beat batched GEMV in decode.
///
/// GH-141 Five-Whys: At M=4, cuBLAS SGEMM dequants Q4K → FP32 (4 B/elem)
/// then runs SGEMM. Batched GEMV reads Q4K directly (0.5625 B/elem) — 7.1x
/// less bandwidth. SGEMM only wins at large M where compute dominates.
///
/// Default=4: cuBLAS SGEMM beats batched GEMV at M=4 (51 vs 35 tok/s).
/// Batched Q4K GEMV at M<=8 uses single warp (32 threads/block) — insufficient
/// parallelism. Multi-warp specializations only exist for M=16/32.
/// TODO: Add M=4 multi-warp kernel, then raise threshold to 8+.
pub(crate) fn cublas_gemm_threshold() -> u32 {
    std::env::var("CUBLAS_GEMM_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

/// PMAT-053: Inline PTX for FP32→FP8 E4M3 element-wise conversion.
/// Works on sm_75+ via manual bit manipulation (no cvt.e4m3x2 which needs sm_90).
/// Each thread converts 1 FP32 → 1 E4M3 byte. Block size 256.
///
/// FP8 E4M3: sign(1) + exponent(4) + mantissa(3), bias=7, range ±448.
/// FP32:     sign(1) + exponent(8) + mantissa(23), bias=127.
///
/// Algorithm: clamp |x| to [2^-9, 448], rebias exponent (127→7), round mantissa
/// (23→3 bits with RNE), pack sign+exp+mantissa into 8 bits.
/// NaN/Inf → max finite (0x7E = 448.0). Zero → 0x00.
const F32_TO_E4M3_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_e4m3(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<16>;
    .reg .f32 %f<4>;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];

    // Global thread index
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;

    // Bounds check
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;

    // Load FP32
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];

    // Reinterpret as u32
    mov.b32 %r4, %f0;

    // Extract sign bit (bit 31) -> %r5
    shr.u32 %r5, %r4, 31;

    // Extract FP32 exponent (bits 30:23) -> %r6
    bfe.u32 %r6, %r4, 23, 8;

    // Extract FP32 mantissa (bits 22:0) -> %r7
    and.b32 %r7, %r4, 0x007FFFFF;

    // Check for zero/denorm (exp == 0) -> output 0x00
    setp.eq.u32 %p1, %r6, 0;
    @%p1 bra L_ZERO;

    // Check for NaN/Inf (exp == 255) -> output sign | 0x7E (max finite)
    setp.eq.u32 %p2, %r6, 255;
    @%p2 bra L_NANINF;

    // Rebias exponent: e4m3_exp = fp32_exp - 127 + 7 = fp32_exp - 120
    // E4M3 valid biased exp range: 1..15 (unbiased -6..+8)
    sub.u32 %r8, %r6, 120;

    // Check underflow (fp32_exp < 121 -> e4m3_exp < 1 -> denorm/zero)
    setp.lt.s32 %p1, %r8, 1;
    @%p1 bra L_ZERO;

    // Check overflow (e4m3_exp > 15 -> clamp to max finite)
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF;

    // Round mantissa: 23 bits -> 3 bits (drop 20 low bits, RNE)
    // Round bit = bit 19, sticky = bits 18:0
    shr.u32 %r9, %r7, 20;          // top 3 mantissa bits
    bfe.u32 %r10, %r7, 19, 1;      // round bit
    and.b32 %r11, %r7, 0x0007FFFF; // sticky bits (bits 18:0)

    // RNE: round up if (round && (sticky || lsb_of_result))
    and.b32 %r12, %r9, 1;          // lsb of truncated mantissa
    or.b32 %r13, %r11, %r12;       // sticky | lsb
    setp.ne.u32 %p3, %r13, 0;
    and.b32 %r14, %r10, 1;         // round bit
    // Only round up if round bit is set AND (sticky|lsb)
    selp.u32 %r14, %r14, 0, %p3;
    add.u32 %r9, %r9, %r14;

    // Handle mantissa overflow (0b1000 -> increment exponent)
    setp.gt.u32 %p3, %r9, 7;
    @!%p3 bra L_PACK;
    mov.u32 %r9, 0;
    add.u32 %r8, %r8, 1;
    // If exponent overflows to 16 -> max finite
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF;

L_PACK:
    // Pack: sign(1) | exp(4) | mantissa(3)
    shl.b32 %r5, %r5, 7;
    shl.b32 %r8, %r8, 3;
    or.b32 %r5, %r5, %r8;
    or.b32 %r5, %r5, %r9;
    bra L_STORE;

L_ZERO:
    mov.u32 %r5, 0;
    bra L_STORE;

L_NANINF:
    // sign | 0x7E (max finite = 448.0)
    shl.b32 %r5, %r5, 7;
    or.b32 %r5, %r5, 0x7E;

L_STORE:
    // Store 1 byte
    add.u64 %rd4, %rd0, %rd2;
    st.global.u8 [%rd4], %r5;

L_DONE:
    ret;
}
"#;

/// PMAT-053b: Scaled FP32→FP8 E4M3 conversion kernel.
/// Same as F32_TO_E4M3_PTX but multiplies input by a per-tensor scale factor first.
/// quant_scale = 448 / absmax — normalizes input range to E4M3 representable range.
const F32_TO_E4M3_SCALED_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_e4m3_scaled(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count,
    .param .f32 param_scale
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<16>;
    .reg .f32 %f<4>;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    ld.param.f32 %f3, [param_scale];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;

    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE_S;

    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];
    mul.f32 %f0, %f0, %f3;

    mov.b32 %r4, %f0;
    shr.u32 %r5, %r4, 31;
    bfe.u32 %r6, %r4, 23, 8;
    and.b32 %r7, %r4, 0x007FFFFF;

    setp.eq.u32 %p1, %r6, 0;
    @%p1 bra L_ZERO_S;
    setp.eq.u32 %p2, %r6, 255;
    @%p2 bra L_NANINF_S;

    sub.u32 %r8, %r6, 120;
    setp.lt.s32 %p1, %r8, 1;
    @%p1 bra L_ZERO_S;
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF_S;

    shr.u32 %r9, %r7, 20;
    bfe.u32 %r10, %r7, 19, 1;
    and.b32 %r11, %r7, 0x0007FFFF;
    and.b32 %r12, %r9, 1;
    or.b32 %r13, %r11, %r12;
    setp.ne.u32 %p3, %r13, 0;
    and.b32 %r14, %r10, 1;
    selp.u32 %r14, %r14, 0, %p3;
    add.u32 %r9, %r9, %r14;

    setp.gt.u32 %p3, %r9, 7;
    @!%p3 bra L_PACK_S;
    mov.u32 %r9, 0;
    add.u32 %r8, %r8, 1;
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF_S;

L_PACK_S:
    shl.b32 %r5, %r5, 7;
    shl.b32 %r8, %r8, 3;
    or.b32 %r5, %r5, %r8;
    or.b32 %r5, %r5, %r9;
    bra L_STORE_S;

L_ZERO_S:
    mov.u32 %r5, 0;
    bra L_STORE_S;

L_NANINF_S:
    shl.b32 %r5, %r5, 7;
    or.b32 %r5, %r5, 0x7E;

L_STORE_S:
    add.u64 %rd4, %rd0, %rd2;
    st.global.u8 [%rd4], %r5;

L_DONE_S:
    ret;
}
"#;

/// PMAT-079: Device-scaled FP32→FP8 E4M3 conversion kernel.
/// Reads absmax from a device pointer (no CPU sync needed).
/// Thread 0 of block 0 also writes dequant_scale = absmax/448 to a device buffer
/// for use as cuBLASLt A_SCALE_POINTER.
const F32_TO_E4M3_DEVICE_SCALED_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_e4m3_device_scaled(
    .param .u64 param_dst,           // FP8 output buffer
    .param .u64 param_src,           // FP32 input buffer
    .param .u32 param_count,         // number of elements
    .param .u64 param_absmax_ptr,    // device ptr to u32 (absmax as IEEE 754 bits)
    .param .u64 param_dequant_ptr    // device ptr to f32 where to write absmax/448
) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<16>;
    .reg .f32 %f<7>;
    .reg .pred %p<5>;

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    ld.param.u64 %rd4, [param_absmax_ptr];
    ld.param.u64 %rd5, [param_dequant_ptr];

    // Read absmax from device memory (u32 bits to f32)
    ld.global.u32 %r15, [%rd4];
    mov.b32 %f4, %r15;

    // Handle absmax == 0: use 1.0 to avoid div-by-zero
    setp.eq.f32 %p3, %f4, 0f00000000;
    @%p3 mov.f32 %f4, 0f3F800000;

    // Compute quant_scale = 448 / absmax (f5 = 448.0, reused below)
    mov.f32 %f5, 0f43E00000;
    div.rn.f32 %f3, %f5, %f4;

    // Thread 0 of block 0: write dequant_scale = absmax / 448 to device buffer
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    or.b32 %r14, %r1, %r2;
    setp.ne.u32 %p4, %r14, 0;
    @%p4 bra L_SKIP_DEQUANT;
    div.rn.f32 %f6, %f4, %f5;
    st.global.f32 [%rd5], %f6;
L_SKIP_DEQUANT:

    // Grid-stride loop for FP8 conversion
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;

    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE_DS;

    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];
    mul.f32 %f0, %f0, %f3;

    // FP32 to FP8 E4M3 conversion (same logic as f32_to_e4m3_scaled)
    mov.b32 %r4, %f0;
    shr.u32 %r5, %r4, 31;
    bfe.u32 %r6, %r4, 23, 8;
    and.b32 %r7, %r4, 0x007FFFFF;

    setp.eq.u32 %p1, %r6, 0;
    @%p1 bra L_ZERO_DS;
    setp.eq.u32 %p2, %r6, 255;
    @%p2 bra L_NANINF_DS;

    sub.u32 %r8, %r6, 120;
    setp.lt.s32 %p1, %r8, 1;
    @%p1 bra L_ZERO_DS;
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF_DS;

    shr.u32 %r9, %r7, 20;
    bfe.u32 %r10, %r7, 19, 1;
    and.b32 %r11, %r7, 0x0007FFFF;
    and.b32 %r12, %r9, 1;
    or.b32 %r13, %r11, %r12;
    setp.ne.u32 %p3, %r13, 0;
    and.b32 %r14, %r10, 1;
    selp.u32 %r14, %r14, 0, %p3;
    add.u32 %r9, %r9, %r14;

    setp.gt.u32 %p3, %r9, 7;
    @!%p3 bra L_PACK_DS;
    mov.u32 %r9, 0;
    add.u32 %r8, %r8, 1;
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF_DS;

L_PACK_DS:
    shl.b32 %r5, %r5, 7;
    shl.b32 %r8, %r8, 3;
    or.b32 %r5, %r5, %r8;
    or.b32 %r5, %r5, %r9;
    bra L_STORE_DS;

L_ZERO_DS:
    mov.u32 %r5, 0;
    bra L_STORE_DS;

L_NANINF_DS:
    shl.b32 %r5, %r5, 7;
    or.b32 %r5, %r5, 0x7E;

L_STORE_DS:
    add.u64 %rd4, %rd0, %rd2;
    st.global.u8 [%rd4], %r5;

L_DONE_DS:
    ret;
}
"#;

/// PMAT-053b: GPU absmax reduction kernel.
/// Each block computes a local absmax via shared memory, then atomicMax to global result.
/// Output buffer must be pre-zeroed (single u32 interpreted as FP32 via IEEE 754 ordering).
/// For positive floats, u32 ordering matches f32 ordering, so atom.max.u32 is correct.
const ABSMAX_REDUCE_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry absmax_reduce(
    .param .u64 param_output,
    .param .u64 param_input,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<10>;
    .reg .f32 %f<4>;
    .reg .pred %p<2>;
    .shared .align 4 .b32 sdata[256];

    ld.param.u64 %rd0, [param_output];
    ld.param.u64 %rd1, [param_input];
    ld.param.u32 %r0, [param_count];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r4, %r2, %r3, %r1;

    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r5, %r5, %r3;

    mov.b32 %f0, 0x00000000;

L_LOOP:
    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra L_REDUCE;

    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];
    abs.f32 %f1, %f1;
    max.f32 %f0, %f0, %f1;

    add.u32 %r4, %r4, %r5;
    bra L_LOOP;

L_REDUCE:
    // Store thread-local absmax into shared memory
    // %r7 = &sdata[tid.x] (reused as this thread's shared addr)
    mov.u32 %r6, sdata;
    shl.b32 %r9, %r1, 2;
    add.u32 %r7, %r6, %r9;
    st.shared.f32 [%r7], %f0;
    bar.sync 0;

    // Tree reduction in shared memory
    mov.u32 %r8, 128;
L_RED_LOOP:
    setp.ge.u32 %p1, %r1, %r8;
    @%p1 bra L_RED_DONE;
    add.u32 %r9, %r1, %r8;
    shl.b32 %r9, %r9, 2;
    add.u32 %r9, %r6, %r9;
    ld.shared.f32 %f2, [%r9];
    max.f32 %f0, %f0, %f2;
    st.shared.f32 [%r7], %f0;
L_RED_DONE:
    bar.sync 0;
    shr.u32 %r8, %r8, 1;
    setp.ne.u32 %p1, %r8, 0;
    @%p1 bra L_RED_LOOP;

    // Thread 0 does atomic max into global output
    setp.ne.u32 %p0, %r1, 0;
    @%p0 bra L_EXIT;
    mov.b32 %r9, %f0;
    atom.global.max.u32 %r9, [%rd0], %r9;

L_EXIT:
    ret;
}
"#;

/// PMAT-031: Inline PTX for FP32→FP16 element-wise conversion.
/// Block size 256, one element per thread. Trivially memory-bound (~1μs for 160K elements).
const F32_TO_F16_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_f16(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<4>;
    .reg .f32 %f0;
    .reg .b16 %h0;
    .reg .pred %p0;
    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];
    cvt.rn.f16.f32 %h0, %f0;
    shl.b64 %rd4, %rd2, 1;
    add.u64 %rd4, %rd0, %rd4;
    st.global.b16 [%rd4], %h0;
L_DONE:
    ret;
}
"#;

/// PMAT-053: Inline PTX for FP16->FP32 element-wise conversion.
/// Block size 256, one element per thread. Uses hardware cvt.f32.f16.
const F16_TO_F32_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f16_to_f32(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<4>;
    .reg .f32 %f0;
    .reg .b16 %h0;
    .reg .pred %p0;
    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;
    // Load FP16 (16 bits)
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 1;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.b16 %h0, [%rd3];
    // Convert FP16 -> FP32 (hardware instruction)
    cvt.f32.f16 %f0, %h0;
    // Store as FP32 (4 bytes)
    shl.b64 %rd4, %rd2, 2;
    add.u64 %rd4, %rd0, %rd4;
    st.global.f32 [%rd4], %f0;
L_DONE:
    ret;
}
"#;

/// PMAT-079: FP16 to FP32 conversion with device-side activation dequant scaling.
/// Reads a single FP32 scale from a device pointer (act_dequant = act_absmax/448),
/// multiplies each converted element by it.
/// Weight dequant is already folded into the GEMM alpha — only act_dequant varies per request.
const F16_TO_F32_ACT_SCALED_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f16_to_f32_act_scaled(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count,
    .param .u64 param_act_dequant_ptr
) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<2>;
    .reg .b16 %h0;
    .reg .pred %p0;
    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    ld.param.u64 %rd5, [param_act_dequant_ptr];

    // Read act_dequant scale from device (same for all elements)
    ld.global.f32 %f1, [%rd5];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE_AS;
    // Load FP16 (16 bits)
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 1;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.b16 %h0, [%rd3];
    // Convert FP16 to FP32 then scale by act_dequant
    cvt.f32.f16 %f0, %h0;
    mul.f32 %f0, %f0, %f1;
    // Store as FP32 (4 bytes)
    shl.b64 %rd4, %rd2, 2;
    add.u64 %rd4, %rd0, %rd4;
    st.global.f32 [%rd4], %f0;
L_DONE_AS:
    ret;
}
"#;

impl CudaExecutor {
    /// Initialize cuBLAS handle for prefill GEMM (lazy, called once)
    pub(crate) fn ensure_cublas(&mut self) -> Result<(), GpuError> {
        if self.cublas_handle.is_some() {
            return Ok(());
        }
        let handle = trueno_gpu::driver::CublasHandle::new(&self.context)?;
        handle.set_stream(&self.stream)?;
        self.cublas_handle = Some(handle);
        Ok(())
    }

    /// PMAT-063: Pre-allocate cuBLAS workspace for CUDA graph capture.
    ///
    /// cuBLAS internally allocates workspace for fast algorithm selection.
    /// During CUDA graph capture, dynamic allocation is forbidden, so cuBLAS
    /// falls back to workspace-free algorithms (7x slower on RTX 4060L).
    ///
    /// This allocates a 32 MB GPU buffer and registers it with cuBLAS via
    /// `cublasSetWorkspace`. Must be called before prefill graph capture.
    pub(crate) fn ensure_cublas_workspace(&mut self) -> Result<(), GpuError> {
        if self.cublas_workspace.is_some() {
            return Ok(());
        }
        self.ensure_cublas()?;

        // 32 MB workspace — covers all standard cuBLAS GEMM shapes
        const WORKSPACE_SIZE: usize = 32 * 1024 * 1024;
        let workspace = GpuBuffer::<u8>::new(&self.context, WORKSPACE_SIZE)?;
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        handle.set_workspace(workspace.as_ptr(), WORKSPACE_SIZE)?;
        eprintln!(
            "[PMAT-063] cuBLAS workspace: {} MB pre-allocated for graph capture",
            WORKSPACE_SIZE / 1024 / 1024
        );
        self.cublas_workspace = Some(workspace);
        Ok(())
    }

    /// Ensure dequant scratch buffer is large enough for N×K FP32 elements
    fn ensure_dequant_scratch(&mut self, n: u32, k: u32) -> Result<(), GpuError> {
        let needed = n as usize * k as usize;
        if self.dequant_scratch_size >= needed {
            return Ok(());
        }
        self.dequant_scratch = Some(GpuBuffer::new(&self.context, needed)?);
        self.dequant_scratch_size = needed;
        Ok(())
    }

    /// PMAT-031: Ensure FP16 activation scratch is large enough
    fn ensure_fp16_activation_scratch(&mut self, count: usize) -> Result<(), GpuError> {
        if self.fp16_activation_scratch_size >= count {
            return Ok(());
        }
        self.fp16_activation_scratch = Some(GpuBuffer::new(&self.context, count)?);
        self.fp16_activation_scratch_size = count;
        Ok(())
    }

    /// PMAT-031: Convert FP32 GPU data to FP16 using inline PTX kernel
    fn convert_f32_to_f16(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
    ) -> Result<(), GpuError> {
        // Compile conversion kernel once
        if !self.modules.contains_key("f32_to_f16") {
            let module = self.compile_ptx(F32_TO_F16_PTX)?;
            self.modules.insert("f32_to_f16".to_string(), module);
        }

        let module = self.modules.get_mut("f32_to_f16").expect("just inserted");
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        // SAFETY: src_ptr and dst_ptr are valid GPU allocations, count verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_f16",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-053: Ensure FP8 activation scratch is large enough
    fn ensure_fp8_activation_scratch(&mut self, count: usize) -> Result<(), GpuError> {
        if self.fp8_activation_scratch_size >= count {
            return Ok(());
        }
        self.fp8_activation_scratch = Some(GpuBuffer::new(&self.context, count)?);
        self.fp8_activation_scratch_size = count;
        // PMAT-084: Reallocation invalidates cached FP8 activation data
        self.fp8_activation_cache_key = None;
        Ok(())
    }

    /// PMAT-053: Convert FP32 GPU data to FP8 E4M3 using inline PTX kernel (sm_75+)
    fn convert_f32_to_e4m3(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
    ) -> Result<(), GpuError> {
        if !self.modules.contains_key("f32_to_e4m3") {
            let module = self.compile_ptx(F32_TO_E4M3_PTX)?;
            self.modules.insert("f32_to_e4m3".to_string(), module);
        }

        let module = self.modules.get_mut("f32_to_e4m3").expect("just inserted");
        // Each thread processes 1 element
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        // SAFETY: src_ptr and dst_ptr are valid GPU allocations
        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_e4m3",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-053: Convert FP16 GPU data to FP32 using hardware cvt.f32.f16
    fn convert_f16_to_f32(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
    ) -> Result<(), GpuError> {
        if !self.modules.contains_key("f16_to_f32") {
            let module = self.compile_ptx(F16_TO_F32_PTX)?;
            self.modules.insert("f16_to_f32".to_string(), module);
        }

        let module = self.modules.get_mut("f16_to_f32").expect("just inserted");
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        unsafe {
            self.stream.launch_kernel(
                module,
                "f16_to_f32",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-079: Convert FP16→FP32 with device-side activation dequant scaling.
    ///
    /// Reads act_dequant (act_absmax/448) from a device pointer and multiplies
    /// each converted element by it. Weight dequant is folded into GEMM alpha.
    fn convert_f16_to_f32_act_scaled(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
        act_dequant_ptr: u64,
    ) -> Result<(), GpuError> {
        let cache_key = "f16_to_f32_act_scaled";
        if !self.modules.contains_key(cache_key) {
            let module = self.compile_ptx(F16_TO_F32_ACT_SCALED_PTX)?;
            self.modules.insert(cache_key.to_string(), module);
        }

        let module = self.modules.get_mut(cache_key).expect("just inserted");
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;
        let mut act_deq = act_dequant_ptr;

        unsafe {
            self.stream.launch_kernel(
                module,
                "f16_to_f32_act_scaled",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut act_deq) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-053b: Launch scaled FP32→FP8 E4M3 conversion kernel.
    /// Multiplies each element by quant_scale before E4M3 conversion.
    fn convert_f32_to_e4m3_scaled(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
        quant_scale: f32,
    ) -> Result<(), GpuError> {
        let cache_key = "f32_to_e4m3_scaled";
        if !self.modules.contains_key(cache_key) {
            let module = self.compile_ptx(F32_TO_E4M3_SCALED_PTX)?;
            self.modules.insert(cache_key.to_string(), module);
        }

        let module = self.modules.get_mut(cache_key).expect("just inserted");
        let num_blocks = (count + 255) / 256;
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;
        let mut scale = quant_scale;

        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_e4m3_scaled",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut scale) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-053b: Compute absmax of a GPU FP32 buffer via reduction kernel.
    /// Returns the absolute maximum value as f32.
    fn gpu_absmax(&mut self, src_ptr: u64, count: u32) -> Result<f32, GpuError> {
        if !self.modules.contains_key("absmax_reduce") {
            let module = self.compile_ptx(ABSMAX_REDUCE_PTX)?;
            self.modules.insert("absmax_reduce".to_string(), module);
        }

        // Allocate single u32 result buffer and zero it explicitly
        let mut result_buf = GpuBuffer::<u32>::new(&self.context, 1)?;
        let result_ptr = result_buf.as_ptr();
        // cuMemAlloc does NOT guarantee zeroed memory — zero explicitly
        result_buf.copy_from_host(&[0u32])?;

        let module = self
            .modules
            .get_mut("absmax_reduce")
            .expect("just inserted");
        let num_blocks = ((count + 255) / 256).min(256);
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 256 * 4,
        };

        let mut out = result_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        unsafe {
            self.stream.launch_kernel(
                module,
                "absmax_reduce",
                &config,
                &mut [
                    std::ptr::from_mut(&mut out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;

        // Read result back (u32 reinterpreted as f32)
        let mut result_u32 = [0u32; 1];
        result_buf.copy_to_host(&mut result_u32)?;
        let absmax = f32::from_bits(result_u32[0]);

        Ok(absmax)
    }

    /// PMAT-079: Compute absmax on GPU, keep result on device. No CPU sync.
    ///
    /// Returns the device pointer to the absmax result (u32 = f32 bits).
    /// The persistent buffer `fp8_absmax_buf` is reused across calls.
    fn gpu_absmax_device(&mut self, src_ptr: u64, count: u32) -> Result<u64, GpuError> {
        if !self.modules.contains_key("absmax_reduce") {
            let module = self.compile_ptx(ABSMAX_REDUCE_PTX)?;
            self.modules.insert("absmax_reduce".to_string(), module);
        }

        // Allocate or reuse persistent absmax buffer
        if self.fp8_absmax_buf.is_none() {
            self.fp8_absmax_buf = Some(GpuBuffer::<u32>::new(&self.context, 1)?);
        }
        let absmax_buf = self.fp8_absmax_buf.as_mut().expect("just allocated");
        let result_ptr = absmax_buf.as_ptr();
        // Zero the buffer on self.stream (NOT stream 0!) to avoid race with previous
        // matmul's conversion kernel that reads from this same buffer.
        // SAFETY: zero_val lives until stream.synchronize or kernel completion.
        // The zero is consumed by absmax_reduce (same stream, ordered after).
        let zero_val = [0u32; 1];
        unsafe {
            absmax_buf.copy_from_host_async(&zero_val, &self.stream)?;
        }

        let module = self
            .modules
            .get_mut("absmax_reduce")
            .expect("just inserted");
        let num_blocks = ((count + 255) / 256).min(256);
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 256 * 4,
        };

        let mut out = result_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        unsafe {
            self.stream.launch_kernel(
                module,
                "absmax_reduce",
                &config,
                &mut [
                    std::ptr::from_mut(&mut out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // No sync! Result stays on device for the next kernel to read.
        Ok(result_ptr)
    }

    /// PMAT-079: Launch device-scaled FP32→FP8 E4M3 conversion kernel.
    ///
    /// Reads absmax from a device pointer (output of `gpu_absmax_device`).
    /// Also writes `absmax/448` to `dequant_ptr` for cuBLASLt A_SCALE_POINTER.
    fn convert_f32_to_e4m3_device_scaled(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
        absmax_ptr: u64,
        dequant_ptr: u64,
    ) -> Result<(), GpuError> {
        let cache_key = "f32_to_e4m3_device_scaled";
        if !self.modules.contains_key(cache_key) {
            let module = self.compile_ptx(F32_TO_E4M3_DEVICE_SCALED_PTX)?;
            self.modules.insert(cache_key.to_string(), module);
        }

        let module = self.modules.get_mut(cache_key).expect("just inserted");
        let num_blocks = (count + 255) / 256;
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;
        let mut abs_ptr = absmax_ptr;
        let mut deq_ptr = dequant_ptr;

        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_e4m3_device_scaled",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut abs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut deq_ptr) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-053b: Get cached FP8 E4M3 weight with per-tensor scaling.
    ///
    /// On cache miss: dequant Q4K/Q6K → FP32 → absmax → scaled FP8 E4M3 → cache.
    /// Also stores the dequant scale (absmax/448) in fp8_weight_scales for cuBLASLt.
    fn get_or_cache_fp8_weight(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        if let Some(buf) = self.fp8_weight_cache.get(&weight_ptr) {
            return Ok(buf.as_ptr());
        }

        // Cache miss: dequant → FP32 scratch
        let f32_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "get_or_cache_fp8_weight: unsupported qtype {:?}",
                    qtype
                )))
            },
        };

        let count = n as usize * k as usize;

        // PMAT-053b: Compute per-tensor absmax for scaling
        let absmax = self.gpu_absmax(f32_ptr, count as u32)?;
        let absmax = if absmax == 0.0 { 1.0 } else { absmax };
        let quant_scale = 448.0 / absmax;
        let dequant_scale = absmax / 448.0;

        // Allocate persistent FP8 buffer [N × K] — 1 byte per element
        let fp8_buf = GpuBuffer::<u8>::new(&self.context, count)?;
        let fp8_ptr = fp8_buf.as_ptr();

        // Convert FP32 → scaled FP8 E4M3
        self.convert_f32_to_e4m3_scaled(f32_ptr, fp8_ptr, count as u32, quant_scale)?;

        // Store dequant scale as CPU float — used as GEMM alpha (constant, no sync needed)
        self.fp8_weight_scales.insert(weight_ptr, dequant_scale);

        self.fp8_weight_cache.insert(weight_ptr, fp8_buf);
        Ok(fp8_ptr)
    }

    /// PMAT-079: Fully async FP8 E4M3 GEMM — zero CPU syncs.
    ///
    /// Pipeline (all on device, no CPU readback):
    ///   1. absmax_reduce → device absmax_buf (no sync)
    ///   2. f32_to_e4m3_device_scaled → reads absmax from device, writes FP8 + act_dequant
    ///   3. gemm_fp8_e4m3_to_f16 → unscaled GEMM with alpha=1.0
    ///   4. f16_to_f32_device_scaled → reads act_dequant × weight_dequant from device
    ///
    /// The GEMM computes raw FP8 dot products (no scaling). The dequant is applied
    /// during the FP16→FP32 conversion: output = f16_val × (act_absmax/448) × (w_absmax/448).
    /// This avoids both the GPU→CPU absmax sync AND cuBLASLt scale pointer issues.
    #[allow(clippy::too_many_arguments)]
    fn cublas_prefill_fp8_gemm(
        &mut self,
        w_fp8_ptr: u64,
        weight_key: u64, // original weight_ptr used as key into fp8_weight_scales
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32, // sequence/batch length (tokens)
        n: u32, // output dimension
        k: u32, // input dimension
    ) -> Result<(), GpuError> {
        let detail_trace = std::env::var("PREFILL_DETAIL_TRACE").is_ok();
        let t0 = if detail_trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // cuBLASLt FP8 requires batch dimension aligned to 16
        let m_padded = (m + 15) & !15;

        // Step 1+2: Device-side absmax + FP8 conversion (zero CPU syncs)
        // PMAT-084: Cache FP8 activation — skip redundant absmax+convert when
        // multiple GEMMs share the same input (QKV phase, FFN gate+up).
        // Saves 84 kernel pairs per prefill (3 per layer × 28 layers).
        let input_actual_count = (m as usize * k as usize) as u32;
        let input_padded_count = m_padded as usize * k as usize;
        self.ensure_fp8_activation_scratch(input_padded_count)?;
        let input_fp8_ptr = self
            .fp8_activation_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();

        // Ensure persistent dequant buffer exists
        if self.fp8_act_dequant_buf.is_none() {
            self.fp8_act_dequant_buf = Some(GpuBuffer::<f32>::new(&self.context, 1)?);
        }
        let act_dequant_ptr = self
            .fp8_act_dequant_buf
            .as_ref()
            .expect("just allocated")
            .as_ptr();

        let cache_key = (packed_input_ptr, input_actual_count);
        if self.fp8_activation_cache_key == Some(cache_key) {
            // PMAT-084: Reuse cached FP8 activation + dequant scale.
            // QKV phase: Q computes, K+V reuse. FFN: gate computes, up reuses.
            // 3 hits/layer × 28 layers = 84 saved absmax+convert pairs.
            if detail_trace {
                eprintln!("[PMAT-084] FP8 activation cache HIT ptr={packed_input_ptr:#x} count={input_actual_count}");
            }
        } else {
            let absmax_ptr = self.gpu_absmax_device(packed_input_ptr, input_actual_count)?;
            self.convert_f32_to_e4m3_device_scaled(
                packed_input_ptr,
                input_fp8_ptr,
                input_actual_count,
                absmax_ptr,
                act_dequant_ptr,
            )?;
            self.fp8_activation_cache_key = Some(cache_key);
        }

        // Look up weight dequant scale (CPU float, constant per weight, no sync needed)
        let weight_dequant = *self.fp8_weight_scales.get(&weight_key).ok_or_else(|| {
            GpuError::InvalidParameter(format!(
                "FP8 weight scale not found for key {weight_key:#x}"
            ))
        })?;

        let t1 = if detail_trace {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Step 3: cuBLASLt FP8 GEMM with alpha=weight_dequant → FP16 output
        // weight_dequant is a constant CPU float (computed once at weight cache time).
        // This partially dequants: D = (w_max/448) × FP8(A) × FP8(B)
        // = (448/act_max) × true_result. The act_dequant (act_max/448) is applied in step 4.
        let output_padded_count = n as usize * m_padded as usize;
        self.ensure_fp16_activation_scratch(output_padded_count)?;
        let f16_output_ptr = self
            .fp16_activation_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();

        if self.cublaslt_handle.is_none() {
            self.cublaslt_handle = Some(trueno_gpu::driver::CublasLtHandle::new()?);
        }
        // PMAT-086: Use cached GEMM to avoid per-call descriptor creation.
        // 168 GEMMs per prefill × ~30μs descriptor overhead = ~5ms savings.
        let lt_handle = self.cublaslt_handle.as_mut().expect("just created");
        lt_handle.gemm_fp8_e4m3_to_f16_cached(
            n as i32,
            m_padded as i32,
            k as i32,
            weight_dequant, // alpha = w_absmax/448 (constant, no sync needed)
            w_fp8_ptr,
            k as i32,
            input_fp8_ptr,
            k as i32,
            0.0,
            f16_output_ptr,
            n as i32,
            &self.stream,
        )?;

        let t2 = if detail_trace {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Step 4: Convert FP16→FP32 with device-side act_dequant scaling.
        // Reads act_dequant (act_absmax/448) from device, multiplies each element by it.
        // Combined with step 3 alpha: D_f32 = f16_val × act_dequant = true_result.
        let output_actual_count = n as usize * m as usize;
        self.convert_f16_to_f32_act_scaled(
            f16_output_ptr,
            packed_output_ptr,
            output_actual_count as u32,
            act_dequant_ptr,
        )?;

        if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
            self.stream.synchronize()?;
            let t3 = std::time::Instant::now();
            eprintln!(
                "[FP8-TRACE] M={} (pad={}) N={} K={}: absmax+convert={:.3}ms gemm={:.3}ms f16->f32+scale={:.3}ms total={:.3}ms",
                m,
                m_padded,
                n,
                k,
                t1.duration_since(t0).as_secs_f64() * 1000.0,
                t2.duration_since(t1).as_secs_f64() * 1000.0,
                t3.duration_since(t2).as_secs_f64() * 1000.0,
                t3.duration_since(t0).as_secs_f64() * 1000.0,
            );
        }

        Ok(())
    }

    /// Launch Q4K dequant kernel to an arbitrary output buffer
    fn launch_dequant_q4k(
        &mut self,
        weight_ptr: u64,
        output_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q4k_dequant_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q4KDequant { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = output_ptr;
        let mut ptr_w = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All pointers are valid GPU allocations, dimensions verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "q4k_dequant_to_f32",
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-065: Launch Q4K → FP16 direct dequant kernel
    ///
    /// Dequants Q4K super-blocks directly to FP16 output (no F32 intermediate).
    /// Half the output bandwidth of launch_dequant_q4k (2 B/elem vs 4 B/elem).
    fn launch_dequant_q4k_fp16(
        &mut self,
        weight_ptr: u64,
        output_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q4k_dequant_fp16_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q4KDequantFp16 { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = output_ptr;
        let mut ptr_w = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        unsafe {
            self.stream.launch_kernel(
                module,
                "q4k_dequant_to_f16",
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-065: Dequant Q4K → FP16 temp buffer for L2-cached HGEMM
    ///
    /// Per-matmul dequant: Q4K weights (DRAM) → FP16 temp (L2-hot) → cuBLAS HGEMM.
    /// Reads 3.56x less from DRAM vs cached FP16 HGEMM (0.5625 vs 2.0 B/elem).
    /// The FP16 temp buffer (≤27.5 MB for largest matrix) fits in RTX 4060's 32 MB L2,
    /// so cuBLAS reads from L2 instead of DRAM.
    ///
    /// Uses a separate `fp16_dequant_temp` buffer (not `fp16_activation_scratch`,
    /// which is used for input activation conversion in cublas_prefill_hgemm).
    fn dequant_q4k_fp16_temp(&mut self, weight_ptr: u64, n: u32, k: u32) -> Result<u64, GpuError> {
        let count = n as usize * k as usize;
        // Ensure temp buffer is large enough
        let need_alloc = match &self.fp16_dequant_temp {
            Some(buf) => buf.len() < count,
            None => true,
        };
        if need_alloc {
            self.fp16_dequant_temp = Some(GpuBuffer::<u16>::new(&self.context, count)?);
        }
        let fp16_ptr = self
            .fp16_dequant_temp
            .as_ref()
            .expect("temp just allocated")
            .as_ptr();
        self.launch_dequant_q4k_fp16(weight_ptr, fp16_ptr, n, k)?;
        Ok(fp16_ptr)
    }

    /// Launch Q6K dequant kernel to an arbitrary output buffer
    fn launch_dequant_q6k(
        &mut self,
        weight_ptr: u64,
        output_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q6k_dequant_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q6KDequant { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = output_ptr;
        let mut ptr_w = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All pointers are valid GPU allocations, dimensions verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "q6k_dequant_to_f32",
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// Dequantize Q4K weights on GPU into FP32 scratch buffer
    fn dequant_q4k_to_scratch(&mut self, weight_ptr: u64, n: u32, k: u32) -> Result<u64, GpuError> {
        self.ensure_dequant_scratch(n, k)?;
        let scratch_ptr = self
            .dequant_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.launch_dequant_q4k(weight_ptr, scratch_ptr, n, k)?;
        Ok(scratch_ptr)
    }

    /// PMAT-026: Dequantize Q6K weights on GPU into FP32 scratch buffer
    fn dequant_q6k_to_scratch(&mut self, weight_ptr: u64, n: u32, k: u32) -> Result<u64, GpuError> {
        self.ensure_dequant_scratch(n, k)?;
        let scratch_ptr = self
            .dequant_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.launch_dequant_q6k(weight_ptr, scratch_ptr, n, k)?;
        Ok(scratch_ptr)
    }

    /// PMAT-031: Get cached FP16 weight or dequant+convert+cache on first access.
    ///
    /// On cache miss: dequant Q4K/Q6K → FP32 scratch → convert to FP16 → cache.
    /// On cache hit: return cached FP16 pointer directly (zero dequant cost).
    fn get_or_cache_fp16_weight(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        if let Some(buf) = self.fp16_weight_cache.get(&weight_ptr) {
            return Ok(buf.as_ptr());
        }

        // Cache miss: dequant → FP32 scratch
        let f32_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "get_or_cache_fp16_weight: unsupported qtype {:?}",
                    qtype
                )))
            },
        };

        // Allocate persistent FP16 buffer [N × K]
        let count = n as usize * k as usize;
        let fp16_buf = GpuBuffer::<u16>::new(&self.context, count)?;
        let fp16_ptr = fp16_buf.as_ptr();

        // Convert FP32 → FP16 (same stream, ordered after dequant)
        self.convert_f32_to_f16(f32_ptr, fp16_ptr, count as u32)?;

        self.fp16_weight_cache.insert(weight_ptr, fp16_buf);
        Ok(fp16_ptr)
    }

    /// PMAT-031: cuBLAS HGEMM prefill — cached FP16 weights + tensor cores
    ///
    /// C[M×N] = Input_fp16[M×K] @ W_fp16[N×K]^T → C is FP32
    ///
    /// Uses gemm_f16_to_f32: FP16 inputs, FP32 output, FP32 accumulation, tensor cores.
    #[allow(clippy::too_many_arguments)]
    fn cublas_prefill_hgemm(
        &mut self,
        w_fp16_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let detail_trace = std::env::var("PREFILL_DETAIL_TRACE").is_ok();
        let t0 = if detail_trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Convert FP32 activations → FP16
        let input_count = m as usize * k as usize;
        self.ensure_fp16_activation_scratch(input_count)?;
        let input_fp16_ptr = self
            .fp16_activation_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.convert_f32_to_f16(packed_input_ptr, input_fp16_ptr, input_count as u32)?;

        let t1 = if detail_trace {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // HGEMM: FP16 weights × FP16 activations → FP32 output (tensor cores)
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        let result = handle.gemm_f16_to_f32(
            trueno_gpu::driver::GemmOp::Trans,
            trueno_gpu::driver::GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            w_fp16_ptr,
            k as i32,
            input_fp16_ptr,
            k as i32,
            0.0,
            packed_output_ptr,
            n as i32,
        );

        if let (Some(t0), Some(t1)) = (t0, t1) {
            self.stream.synchronize()?;
            let t2 = std::time::Instant::now();
            eprintln!(
                "[HGEMM-TRACE] M={} N={} K={}: cvt={:.3}ms cublas={:.3}ms total={:.3}ms",
                m,
                n,
                k,
                t1.duration_since(t0).as_secs_f64() * 1000.0,
                t2.duration_since(t1).as_secs_f64() * 1000.0,
                t2.duration_since(t0).as_secs_f64() * 1000.0,
            );
        }

        result
    }

    /// PMAT-024/026/031/053/064/GH-182: cuBLAS GEMM (or fused Q4K GEMM) for prefill
    ///
    /// C[M×N] = Input[M×K] @ W[N×K]^T
    ///
    /// Priority:
    /// 0. Q4K_WMMA_PREFILL=1 + Q4K → WMMA tensor core Q4K GEMM (3.56x BW savings + tensor cores)
    /// 1. FUSED_Q4K_PREFILL=1 + Q4K → tiled fused Q4K GEMM (reads Q4K directly, scalar FMA)
    /// 2. FP8_PREFILL=1 + sm_89+ → cached FP8 E4M3 weights + cuBLAS FP8 GEMM (1 B/elem, 2x vs HGEMM)
    /// 3. L2_PREFILL=1 + Q4K → per-matmul Q4K→FP16 dequant + L2-cached HGEMM (3.56x less DRAM BW)
    /// 4. HGEMM_PREFILL!=0 (default) → cached FP16 weights + cuBLAS HGEMM + tensor cores
    /// 5. HGEMM_PREFILL=0 → per-request dequant + cuBLAS SGEMM
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cublas_prefill_gemm(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PMAT-066: DP4A Q4K×Q8 GEMM — no FP16 dequant, 3.56x BW reduction
        if qtype == WeightQuantType::Q4K && std::env::var("DP4A_GEMM_PREFILL").as_deref() == Ok("1")
        {
            return self.launch_dp4a_q4k_gemm(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // PMAT-045: Multi-warp Q4K WMMA GEMM — 4 warps, 32×32 tiles, maxnreg(96)
        if qtype == WeightQuantType::Q4K && std::env::var("MW_WMMA_PREFILL").as_deref() == Ok("1") {
            return self.launch_mw_q4k_wmma_kernel(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // PMAT-064: Q4K WMMA GEMM — tensor cores with direct Q4K weight reads
        // Dequant Q4K→FP16 in SHMEM, WMMA 16×16×16 matmul. 3.56x less BW than HGEMM.
        if qtype == WeightQuantType::Q4K && std::env::var("Q4K_WMMA_PREFILL").as_deref() == Ok("1")
        {
            return self.q4k_wmma_gemm_prefill(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // GH-182: Fused Q4K GEMM — reads Q4K directly (0.5625 B/elem vs 2 B/elem HGEMM)
        if qtype == WeightQuantType::Q4K && std::env::var("FUSED_Q4K_PREFILL").as_deref() == Ok("1")
        {
            return self.fused_q4k_gemm_prefill(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        self.ensure_cublas()?;

        // PMAT-053/067: FP8 E4M3 GEMM — 1 byte/elem (2x BW savings vs HGEMM)
        // Auto-enabled on sm_89+ (Ada Lovelace). Override: FP8_PREFILL=0 to disable.
        if self.gpu_profile.fp8_prefill && self.gpu_profile.cc >= 89 {
            let w_fp8_ptr = self.get_or_cache_fp8_weight(qtype, weight_ptr, n, k)?;
            return self.cublas_prefill_fp8_gemm(
                w_fp8_ptr,
                weight_ptr, // key into fp8_weight_scales
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // PMAT-065: L2-cached HGEMM — per-matmul Q4K→FP16 dequant + HGEMM from L2
        // Reads Q4K from DRAM (0.5625 B/elem), writes FP16 to temp buffer (L2-hot),
        // cuBLAS reads FP16 from L2 instead of DRAM. 3.56x less DRAM bandwidth.
        // Enable with L2_PREFILL=1. Eliminates need for 2944 MB FP16 weight cache.
        if qtype == WeightQuantType::Q4K && std::env::var("L2_PREFILL").as_deref() == Ok("1") {
            let w_fp16_ptr = self.dequant_q4k_fp16_temp(weight_ptr, n, k)?;
            return self.cublas_prefill_hgemm(
                w_fp16_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // PMAT-031: HGEMM path with cached FP16 weights (default)
        // GH-141: Skip HGEMM when FP16 cache was cleared (batched mode frees it
        // to make room for batched KV caches on 8GB GPUs). Uses SGEMM instead.
        if std::env::var("HGEMM_PREFILL").as_deref() != Ok("0")
            && !self.fp16_weight_cache.is_empty()
        {
            let w_fp16_ptr = self.get_or_cache_fp16_weight(qtype, weight_ptr, n, k)?;
            return self.cublas_prefill_hgemm(
                w_fp16_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // Fallback: dequant + SGEMM (original PMAT-024/026 path)
        let w_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "cublas_prefill_gemm: unsupported qtype {:?}",
                    qtype
                )))
            },
        };

        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        handle.gemm_f32(
            trueno_gpu::driver::GemmOp::Trans,
            trueno_gpu::driver::GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            w_ptr,
            k as i32,
            packed_input_ptr,
            k as i32,
            0.0,
            packed_output_ptr,
            n as i32,
        )
    }

    /// PMAT-064: Q4K WMMA GEMM for prefill — tensor cores + direct Q4K reads
    ///
    /// C[M×N] = A[M×K] @ B_q4k[N×(K/256)×144B]^T
    ///
    /// Dequantizes Q4K super-blocks to FP16 in shared memory, uses WMMA
    /// 16×16×16 tensor core tiles for compute. 3.56× less bandwidth than
    /// HGEMM (0.5625 B/elem vs 2 B/elem for FP16).
    ///
    /// Grid: (ceil(N/16), ceil(M/16)), Block: 32 threads (1 warp per WMMA tile)
    #[allow(clippy::too_many_arguments)]
    fn q4k_wmma_gemm_prefill(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        self.launch_q4k_wmma_kernel(weight_ptr, packed_input_ptr, packed_output_ptr, m, n, k)
    }

    /// Launch the Q4K WMMA GEMM kernel
    ///
    /// WMMA stores full 16×16 tiles, so when M or N isn't a multiple of 16,
    /// edge tiles write past the output buffer. To avoid corrupting adjacent
    /// GPU memory, we allocate a padded temporary buffer and copy back.
    #[allow(clippy::too_many_arguments)]
    fn launch_q4k_wmma_kernel(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Pad M and N to multiples of 16 for WMMA tile safety
        let m_padded = (m + 15) & !15;
        let n_padded = (n + 15) & !15;
        let needs_padding = m_padded != m || n_padded != n;

        // Use padded dimensions for kernel compilation (n_const in store stride)
        let kernel_type = KernelType::TensorCoreQ4KGemm {
            m: m_padded,
            n: n_padded,
            k,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("tensor_core_q4k_gemm_{m_padded}_{n_padded}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        // If padding needed, allocate temp buffer BEFORE borrowing modules
        let actual_output_ptr = if needs_padding {
            let padded_count = m_padded as usize * n_padded as usize;
            self.ensure_wmma_scratch(padded_count)?;
            self.wmma_scratch
                .as_ref()
                .expect("wmma scratch allocated")
                .as_ptr()
        } else {
            packed_output_ptr
        };

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/16), ceil(M/16)), Block: 32 (1 warp for WMMA)
        let grid_x = n_padded / 16;
        let grid_y = m_padded / 16;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 32, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = weight_ptr;
        let mut ptr_c = actual_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Copy valid [M, N] from padded buffer to actual output
        if needs_padding {
            // Synchronize stream to ensure WMMA kernel completes before D2D copy.
            // cuMemcpyDtoD is host-synchronous but NOT stream-ordered — it races
            // with async kernel launches without this sync.
            self.stream.synchronize()?;
            // Copy row by row: each row has N valid elements out of N_padded
            for row in 0..m {
                let src_offset = row as u64 * n_padded as u64 * 4;
                let dst_offset = row as u64 * n as u64 * 4;
                self.stream.memcpy_dtod_sync(
                    packed_output_ptr + dst_offset,
                    actual_output_ptr + src_offset,
                    n as usize * 4,
                )?;
            }
        }

        Ok(())
    }

    /// PMAT-045: Multi-Warp Q4K WMMA GEMM — 4 warps, 32×32 output tiles
    ///
    /// C[M×N] = A[M×K] @ B_q4k[N×(K/256)×144B]^T
    ///
    /// 4 warps per block (128 threads), each warp handles a 16×16 WMMA tile.
    /// Grid: (ceil(N/32), ceil(M/32)). SHMEM: 2048 bytes.
    /// maxnreg(96) limits register pressure for better occupancy.
    #[allow(clippy::too_many_arguments)]
    fn launch_mw_q4k_wmma_kernel(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Pad M and N to multiples of 32 for 2×2 WMMA tile safety
        let m_padded = (m + 31) & !31;
        let n_padded = (n + 31) & !31;
        let needs_padding = m_padded != m || n_padded != n;

        let kernel_type = KernelType::MultiWarpTensorCoreQ4KGemm {
            m: m_padded,
            n: n_padded,
            k,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("mw_tensor_core_q4k_gemm_{m_padded}_{n_padded}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        // If padding needed, allocate temp buffer
        let actual_output_ptr = if needs_padding {
            let padded_count = m_padded as usize * n_padded as usize;
            self.ensure_wmma_scratch(padded_count)?;
            self.wmma_scratch
                .as_ref()
                .expect("wmma scratch allocated")
                .as_ptr()
        } else {
            packed_output_ptr
        };

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/32), ceil(M/32)), Block: 128 (4 warps for 2×2 WMMA tiles)
        let grid_x = n_padded / 32;
        let grid_y = m_padded / 32;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 128, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = weight_ptr;
        let mut ptr_c = actual_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Copy valid [M, N] from padded buffer to actual output
        if needs_padding {
            self.stream.synchronize()?;
            for row in 0..m {
                let src_offset = row as u64 * n_padded as u64 * 4;
                let dst_offset = row as u64 * n as u64 * 4;
                self.stream.memcpy_dtod_sync(
                    packed_output_ptr + dst_offset,
                    actual_output_ptr + src_offset,
                    n as usize * 4,
                )?;
            }
        }

        Ok(())
    }

    /// PMAT-066: DP4A Q4K×Q8 GEMM — dequant-free prefill
    ///
    /// Pipeline:
    /// 1. Q8 quantize: f32 activations → Q8_1 format (36 bytes per 32 values)
    /// 2. DP4A GEMM: Q4K weights × Q8 activations → f32 output
    ///
    /// No FP16 dequantization. 3.56x memory bandwidth reduction vs HGEMM.
    #[allow(clippy::too_many_arguments)]
    fn launch_dp4a_q4k_gemm(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let total_f32_elements = m * k;
        let num_q8_blocks = total_f32_elements / 32;
        let q8_bytes = num_q8_blocks as usize * 36;

        // Ensure Q8 scratch buffer is large enough
        let need_alloc = match &self.dp4a_q8_scratch {
            Some(buf) => buf.len() < q8_bytes,
            None => true,
        };
        if need_alloc {
            self.dp4a_q8_scratch = Some(GpuBuffer::<u8>::new(&self.context, q8_bytes)?);
        }
        let q8_ptr = self
            .dp4a_q8_scratch
            .as_ref()
            .expect("q8 scratch allocated")
            .as_ptr();

        // Step 1: Q8 quantize M*K f32 activations → Q8_1
        {
            let kernel_type = KernelType::Q8Quantize {
                n: total_f32_elements,
            };
            let kernel_name = self.kernels.kernel_name(&kernel_type);
            let cache_key = format!("q8_quantize_{total_f32_elements}");

            if !self.modules.contains_key(&cache_key) {
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(cache_key.clone(), module);
            }

            let module = self
                .modules
                .get_mut(&cache_key)
                .expect("module just inserted");
            let config = LaunchConfig::grid_2d(num_q8_blocks, 1, 32, 1);
            let mut out = q8_ptr;
            let mut inp = packed_input_ptr;
            let mut n_val = total_f32_elements;

            unsafe {
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut out) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut inp) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Step 2: DP4A Q4K×Q8 GEMM
        let num_warps: u32 = 4;
        let num_half_warps = num_warps * 2;
        let tile_m: u32 = 4;

        let kernel_type = KernelType::Dp4aQ4KGemm { m, n, k };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("dp4a_q4k_gemm_{m}_{n}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let grid_x = (n + num_half_warps - 1) / num_half_warps;
        let grid_y = (m + tile_m - 1) / tile_m;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, num_warps * 32, 1);

        let mut ptr_y = packed_output_ptr;
        let mut ptr_w = weight_ptr;
        let mut ptr_q8 = q8_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_y) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_q8) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// Ensure WMMA scratch buffer is large enough
    fn ensure_wmma_scratch(&mut self, count: usize) -> Result<(), GpuError> {
        let need_alloc = match &self.wmma_scratch {
            Some(buf) => buf.len() < count,
            None => true,
        };
        if need_alloc {
            self.wmma_scratch = Some(GpuBuffer::<f32>::new(&self.context, count)?);
        }
        Ok(())
    }

    /// GH-182: Fused tiled Q4K GEMM for prefill — reads Q4K weights directly
    ///
    /// C[M×N] = A[M×K] @ B_q4k[N×(K/256)×144B]^T
    ///
    /// Each thread computes tile_m output rows for one column, loading weight
    /// super-blocks once and reusing across rows. 3.56x bandwidth reduction
    /// vs HGEMM (0.5625 B/elem vs 2 B/elem).
    #[allow(clippy::too_many_arguments)]
    fn fused_q4k_gemm_prefill(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let tile_m: u32 = std::env::var("FUSED_Q4K_TILE_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

        let kernel_type = KernelType::QuantizedGemmGgmlTiled { m, n, k, tile_m };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_gemm_ggml_tiled_{m}_{n}_{k}_{tile_m}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/block_threads), ceil(M/tile_m))
        let block_threads = 128u32;
        let grid_x = (n + block_threads - 1) / block_threads;
        let grid_y = (m + tile_m - 1) / tile_m;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, block_threads, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = weight_ptr;
        let mut ptr_c = packed_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    // ========================================================================
    // PMAT-032: Parallel Prefill Attention via cuBLAS Strided Batched GEMM
    // ========================================================================

    /// PMAT-032: Inline PTX for causal mask + row-wise softmax.
    ///
    /// Grid: (num_heads, M, 1), Block: (32, 1, 1)
    /// Each block processes one (head, token) pair.
    ///
    /// Params: scores_ptr, M, total_len, base_seq_len, num_heads
    /// scores layout: [num_heads, M, total_len] row-major
    ///
    /// For token i (ctaid.y), valid positions are j < base_seq_len + i + 1.
    /// Sets invalid positions to -inf, then computes in-place softmax.
    const CAUSAL_MASK_SOFTMAX_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry causal_mask_softmax(
    .param .u64 param_scores,
    .param .u32 param_m,
    .param .u32 param_total_len,
    .param .u32 param_base_seq_len,
    .param .u32 param_num_heads
) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<16>;
    .reg .f32 %f<8>;
    .reg .pred %p<4>;

    // head_idx = ctaid.x, token_idx = ctaid.y, tid = tid.x
    ld.param.u64 %rd0, [param_scores];
    ld.param.u32 %r0, [param_m];
    ld.param.u32 %r1, [param_total_len];
    ld.param.u32 %r2, [param_base_seq_len];
    ld.param.u32 %r3, [param_num_heads];

    mov.u32 %r4, %ctaid.x;   // head_idx
    mov.u32 %r5, %ctaid.y;   // token_idx
    mov.u32 %r6, %tid.x;     // lane_id
    mov.u32 %r7, %ntid.x;    // block_size (32)

    // valid_len = base_seq_len + token_idx + 1
    add.u32 %r8, %r2, %r5;
    add.u32 %r8, %r8, 1;

    // row_offset = (head_idx * M + token_idx) * total_len
    mad.lo.u32 %r9, %r4, %r0, %r5;
    mul.lo.u32 %r9, %r9, %r1;

    // row_ptr = scores + row_offset * 4
    cvt.u64.u32 %rd1, %r9;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd1, %rd0, %rd1;

    // === Pass 1: Apply causal mask + find row max ===
    mov.f32 %f0, 0fFF800000;  // max = -inf

    mov.u32 %r10, %r6;  // j = tid
LOOP_MAX:
    setp.ge.u32 %p0, %r10, %r1;  // j >= total_len?
    @%p0 bra DONE_MAX;

    // addr = row_ptr + j * 4
    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    // If j >= valid_len, mask to -inf
    setp.ge.u32 %p1, %r10, %r8;
    @%p1 bra STORE_NEG_INF;

    // Valid position: load and track max
    ld.global.f32 %f1, [%rd2];
    max.f32 %f0, %f0, %f1;
    bra NEXT_MAX;

STORE_NEG_INF:
    mov.f32 %f1, 0fFF800000;
    st.global.f32 [%rd2], %f1;

NEXT_MAX:
    add.u32 %r10, %r10, %r7;  // j += blockDim
    bra LOOP_MAX;

DONE_MAX:
    // Warp reduce max (butterfly)
    shfl.sync.bfly.b32 %f1, %f0, 16, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 8, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 4, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 2, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 1, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    // %f0 = row_max (broadcast to all lanes via shfl)

    // === Pass 2: exp(x - max) and sum ===
    mov.f32 %f2, 0f00000000;  // sum = 0

    mov.u32 %r10, %r6;
LOOP_EXP:
    setp.ge.u32 %p0, %r10, %r1;
    @%p0 bra DONE_EXP;

    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    ld.global.f32 %f1, [%rd2];
    sub.f32 %f1, %f1, %f0;       // x - max
    // Clamp to prevent overflow: if x-max < -88, set to 0
    mov.f32 %f3, 0fC2B00000;     // -88.0
    setp.lt.f32 %p2, %f1, %f3;
    @%p2 mov.f32 %f1, 0fC2B00000;

    // exp approximation using ex2 (exp2(x * log2(e)))
    mul.f32 %f1, %f1, 0f3FB8AA3B;  // x * log2(e) = x * 1.4426950408889634
    ex2.approx.f32 %f1, %f1;
    st.global.f32 [%rd2], %f1;
    add.f32 %f2, %f2, %f1;

    add.u32 %r10, %r10, %r7;
    bra LOOP_EXP;

DONE_EXP:
    // Warp reduce sum
    shfl.sync.bfly.b32 %f1, %f2, 16, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 8, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 4, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 2, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 1, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    // %f2 = sum_exp

    // inv_sum = 1.0 / sum_exp
    rcp.approx.f32 %f2, %f2;

    // === Pass 3: normalize ===
    mov.u32 %r10, %r6;
LOOP_NORM:
    setp.ge.u32 %p0, %r10, %r1;
    @%p0 bra DONE_NORM;

    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    ld.global.f32 %f1, [%rd2];
    mul.f32 %f1, %f1, %f2;
    st.global.f32 [%rd2], %f1;

    add.u32 %r10, %r10, %r7;
    bra LOOP_NORM;

DONE_NORM:
    ret;
}
"#;

    /// PMAT-032: Ensure attention score scratch buffer is large enough
    fn ensure_attn_score_scratch(&mut self, size: usize) -> Result<(), GpuError> {
        if self.prefill_attn_scores_size >= size {
            return Ok(());
        }
        self.prefill_attn_scores = Some(GpuBuffer::new(&self.context, size)?);
        self.prefill_attn_scores_size = size;
        Ok(())
    }

    /// PMAT-032: Bulk scatter M tokens' K/V to cache positions [cache_len..cache_len+M]
    ///
    /// Uses stream-ordered async D2D copies on self.stream, so cuBLAS HGEMM writes
    /// (which also run on self.stream) are guaranteed to complete before the copies
    /// read the K/V buffers. No explicit stream.synchronize() needed.
    fn bulk_scatter_kv(
        &mut self,
        layer_idx: usize,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        m: u32,
        kv_dim: u32,
    ) -> Result<usize, GpuError> {
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let stream_handle = self.stream.raw();

        let new_len = cache_len + m as usize;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PMAT-032: KV cache overflow - max_len={max_len}, trying to add {m} at position {cache_len}"
            )));
        }

        let k_key = format!("kv_{layer_idx}_k");
        let v_key = format!("kv_{layer_idx}_v");

        // Scatter K and V for all M tokens using stream-ordered async copies
        for seq_idx in 0..m as usize {
            let position = cache_len + seq_idx;

            // Scatter K
            {
                let k_cache = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-032: K cache not initialized for layer {layer_idx}"
                    ))
                })?;
                for kv_head in 0..num_kv_heads {
                    let src_offset = kv_head * head_dim + seq_idx * kv_dim as usize;
                    let dst_offset = kv_head * (max_len * head_dim) + position * head_dim;
                    // SAFETY: k_buf_ptr is valid GPU buffer [M, kv_dim], offsets bounded,
                    // src_view lifetime managed manually (forget below)
                    let src_view = unsafe {
                        GpuBuffer::<f32>::from_raw_parts(
                            k_buf_ptr + (src_offset * std::mem::size_of::<f32>()) as u64,
                            head_dim,
                        )
                    };
                    // SAFETY: Both buffers valid until stream sync, stream_handle is valid
                    unsafe {
                        k_cache.copy_from_buffer_at_async_raw(
                            &src_view,
                            dst_offset,
                            0,
                            head_dim,
                            stream_handle,
                        )?;
                    }
                    std::mem::forget(src_view);
                }
            }

            // Scatter V
            {
                let v_cache = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-032: V cache not initialized for layer {layer_idx}"
                    ))
                })?;
                for kv_head in 0..num_kv_heads {
                    let src_offset = kv_head * head_dim + seq_idx * kv_dim as usize;
                    let dst_offset = kv_head * (max_len * head_dim) + position * head_dim;
                    // SAFETY: v_buf_ptr is valid GPU buffer [M, kv_dim], offsets bounded
                    let src_view = unsafe {
                        GpuBuffer::<f32>::from_raw_parts(
                            v_buf_ptr + (src_offset * std::mem::size_of::<f32>()) as u64,
                            head_dim,
                        )
                    };
                    // SAFETY: Both buffers valid until stream sync, stream_handle is valid
                    unsafe {
                        v_cache.copy_from_buffer_at_async_raw(
                            &src_view,
                            dst_offset,
                            0,
                            head_dim,
                            stream_handle,
                        )?;
                    }
                    std::mem::forget(src_view);
                }
            }
        }

        // Update cache length once for all M tokens
        self.kv_cache_lengths.insert(layer_idx, new_len);

        Ok(cache_len)
    }

    /// PMAT-032: Parallel prefill attention using cuBLAS strided batched GEMM.
    ///
    /// Replaces M sequential `incremental_attention_into_for_capture` calls with:
    /// 1. Bulk KV scatter (M tokens at once)
    /// 2. QK^T via cuBLAS strided batched GEMM (handles GQA)
    /// 3. Causal mask + softmax (single PTX kernel)
    /// 4. Attn × V via cuBLAS strided batched GEMM
    ///
    /// For M=8, 28 layers: 672 kernel launches → 196 (3.4× reduction).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_attention_cublas(
        &mut self,
        layer_idx: usize,
        _q_buf: &GpuBuffer<f32>,
        _k_buf: &GpuBuffer<f32>,
        _v_buf: &GpuBuffer<f32>,
        _attn_out_buf: &GpuBuffer<f32>,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        m: u32,
        q_dim: u32,
        kv_dim: u32,
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);

        // PMAT-052: Zero-copy path for fresh prefill (cache_len == 0).
        //
        // Five-Whys: c=1 TTFT has ~14,000 D2D copies from bulk_scatter_kv.
        // Why? M tokens × num_kv_heads × 2 (K/V) per layer × 28 layers.
        // Why? Packed K/V [M, kv_dim] needs rearranging to [num_kv_heads, max_len, head_dim].
        // Why? cuBLAS reads cache with lda=head_dim.
        // Fix: Read K/V directly from packed buffer (lda=kv_dim). Scatter to cache afterward
        // via single PTX kernel per layer (28 launches replaces 14,000 D2D copies).
        if cache_len == 0 {
            let total_len = m as usize;

            // PMAT-069: Fused prefill attention — 1 kernel/layer replaces 5 cuBLAS + PTX launches.
            // Opt-in: FUSED_PREFILL_ATTN=1 (currently 2.8x slower than cuBLAS due to
            // single-warp-per-head design — needs multi-block tiling to be competitive).
            let use_fused_attn = std::env::var("FUSED_PREFILL_ATTN")
                .map(|v| v == "1")
                .unwrap_or(false);

            if use_fused_attn {
                return self.prefill_attention_fused(
                    layer_idx,
                    q_buf_ptr,
                    k_buf_ptr,
                    v_buf_ptr,
                    attn_out_ptr,
                    m,
                    q_dim,
                    kv_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    heads_per_kv,
                    max_len,
                );
            }

            // Score scratch buffer
            self.ensure_cublas()?;
            let score_size = num_heads * m as usize * total_len;
            self.ensure_attn_score_scratch(score_size)?;
            let score_ptr = self
                .prefill_attn_scores
                .as_ref()
                .expect("score scratch just allocated")
                .as_ptr();

            // 1. QK^T — reading K directly from packed buffer (lda=kv_dim)
            let handle = self.cublas_handle.as_ref().expect("cublas initialized");

            for kv_group in 0..num_kv_heads {
                let first_q_head = kv_group * heads_per_kv;
                let k_head_ptr =
                    k_buf_ptr + (kv_group * head_dim * std::mem::size_of::<f32>()) as u64;
                let q_head_ptr =
                    q_buf_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;
                let s_head_ptr = score_ptr
                    + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;

                handle.gemm_f32_strided_batched(
                    trueno_gpu::driver::GemmOp::Trans,   // K^T
                    trueno_gpu::driver::GemmOp::NoTrans, // Q
                    total_len as i32,                    // m (rows of C)
                    m as i32,                            // n (cols of C)
                    head_dim as i32,                     // k
                    scale,                               // alpha = 1/sqrt(d)
                    k_head_ptr,                          // A = packed K
                    kv_dim as i32,                       // lda = kv_dim (NOT head_dim)
                    0,                                   // stride_a = 0 (shared K)
                    q_head_ptr,                          // B = Q
                    q_dim as i32,                        // ldb = q_dim
                    head_dim as i64,                     // stride_b = head_dim
                    0.0,                                 // beta
                    s_head_ptr,                          // C = score
                    total_len as i32,                    // ldc
                    (m as usize * total_len) as i64,     // stride_c
                    heads_per_kv as i32,                 // batch
                )?;
            }

            // 2. Causal mask + softmax
            if !self.modules.contains_key("causal_mask_softmax") {
                let module = self.compile_ptx(Self::CAUSAL_MASK_SOFTMAX_PTX)?;
                self.modules
                    .insert("causal_mask_softmax".to_string(), module);
            }

            {
                let module = self
                    .modules
                    .get_mut("causal_mask_softmax")
                    .expect("just inserted");
                let config = LaunchConfig {
                    grid: (num_heads as u32, m, 1),
                    block: (32, 1, 1),
                    shared_mem: 0,
                };

                let mut s_ptr = score_ptr;
                let mut m_val = m;
                let mut tl_val = total_len as u32;
                let mut base_val = 0u32; // No prior cache
                let mut nh_val = num_heads as u32;

                // SAFETY: score buffer valid, dimensions verified
                unsafe {
                    self.stream.launch_kernel(
                        module,
                        "causal_mask_softmax",
                        &config,
                        &mut [
                            std::ptr::from_mut(&mut s_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut tl_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut base_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut nh_val) as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 3. Attn × V — reading V directly from packed buffer (lda=kv_dim)
            let handle = self.cublas_handle.as_ref().expect("cublas initialized");

            for kv_group in 0..num_kv_heads {
                let first_q_head = kv_group * heads_per_kv;
                let v_head_ptr =
                    v_buf_ptr + (kv_group * head_dim * std::mem::size_of::<f32>()) as u64;
                let p_head_ptr = score_ptr
                    + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;
                let o_head_ptr =
                    attn_out_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;

                handle.gemm_f32_strided_batched(
                    trueno_gpu::driver::GemmOp::NoTrans, // V
                    trueno_gpu::driver::GemmOp::NoTrans, // P
                    head_dim as i32,                     // m (rows of C)
                    m as i32,                            // n (cols of C)
                    total_len as i32,                    // k (inner dim)
                    1.0,                                 // alpha
                    v_head_ptr,                          // A = packed V
                    kv_dim as i32,                       // lda = kv_dim (NOT head_dim)
                    0,                                   // stride_a = 0 (shared V)
                    p_head_ptr,                          // B = P (softmax output)
                    total_len as i32,                    // ldb
                    (m as usize * total_len) as i64,     // stride_b
                    0.0,                                 // beta
                    o_head_ptr,                          // C = output
                    q_dim as i32,                        // ldc = q_dim
                    head_dim as i64,                     // stride_c
                    heads_per_kv as i32,                 // batch
                )?;
            }

            // 4. Scatter K/V from packed buffer to single KV cache via PTX kernel
            let k_key = format!("kv_{layer_idx}_k");
            let v_key = format!("kv_{layer_idx}_v");
            let k_cache_ptr = self
                .kv_cache_gpu
                .get(&k_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?
                .as_ptr();
            let v_cache_ptr = self
                .kv_cache_gpu
                .get(&v_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?
                .as_ptr();

            if !self.modules.contains_key("scatter_packed_kv") {
                let module = self.compile_ptx(Self::SCATTER_PACKED_KV_PTX)?;
                self.modules.insert("scatter_packed_kv".to_string(), module);
            }

            let module = self
                .modules
                .get_mut("scatter_packed_kv")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (m, num_kv_heads as u32, 1),
                block: (head_dim as u32, 1, 1),
                shared_mem: 0,
            };

            let mut k_src = k_buf_ptr;
            let mut v_src = v_buf_ptr;
            let mut k_dst = k_cache_ptr;
            let mut v_dst = v_cache_ptr;
            let mut kv_dim_val = kv_dim;
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;

            // SAFETY: All GPU buffers valid, grid/block dimensions verified
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "scatter_packed_kv",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut k_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut kv_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }

            // Update single KV cache length
            self.kv_cache_lengths.insert(layer_idx, m as usize);

            Ok(())
        } else {
            // Incremental path: cache_len > 0, need data already in cache + new tokens.
            // Fall back to bulk_scatter_kv (scatter first, then read from cache).
            self.bulk_scatter_kv(layer_idx, k_buf_ptr, v_buf_ptr, m, kv_dim as u32)?;
            let total_len = cache_len + m as usize;

            self.ensure_cublas()?;
            let score_size = num_heads * m as usize * total_len;
            self.ensure_attn_score_scratch(score_size)?;
            let score_ptr = self
                .prefill_attn_scores
                .as_ref()
                .expect("score scratch just allocated")
                .as_ptr();

            let k_key = format!("kv_{layer_idx}_k");
            let v_key = format!("kv_{layer_idx}_v");
            let k_cache_ptr = self
                .kv_cache_gpu
                .get(&k_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?
                .as_ptr();
            let v_cache_ptr = self
                .kv_cache_gpu
                .get(&v_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?
                .as_ptr();

            let handle = self.cublas_handle.as_ref().expect("cublas initialized");

            for kv_group in 0..num_kv_heads {
                let first_q_head = kv_group * heads_per_kv;
                let k_head_ptr = k_cache_ptr
                    + (kv_group * max_len * head_dim * std::mem::size_of::<f32>()) as u64;
                let q_head_ptr =
                    q_buf_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;
                let s_head_ptr = score_ptr
                    + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;

                handle.gemm_f32_strided_batched(
                    trueno_gpu::driver::GemmOp::Trans,   // K^T
                    trueno_gpu::driver::GemmOp::NoTrans, // Q
                    total_len as i32,                    // m (rows of C)
                    m as i32,                            // n (cols of C)
                    head_dim as i32,                     // k
                    scale,                               // alpha = 1/sqrt(d)
                    k_head_ptr,                          // A = K cache
                    head_dim as i32,                     // lda
                    0,
                    q_head_ptr,
                    q_dim as i32,
                    head_dim as i64,
                    0.0,
                    s_head_ptr,
                    total_len as i32,
                    (m as usize * total_len) as i64,
                    heads_per_kv as i32,
                )?;
            }

            // Causal mask + softmax
            if !self.modules.contains_key("causal_mask_softmax") {
                let module = self.compile_ptx(Self::CAUSAL_MASK_SOFTMAX_PTX)?;
                self.modules
                    .insert("causal_mask_softmax".to_string(), module);
            }

            {
                let module = self
                    .modules
                    .get_mut("causal_mask_softmax")
                    .expect("just inserted");
                let config = LaunchConfig {
                    grid: (num_heads as u32, m, 1),
                    block: (32, 1, 1),
                    shared_mem: 0,
                };

                let mut s_ptr = score_ptr;
                let mut m_val = m;
                let mut tl_val = total_len as u32;
                let mut base_val = cache_len as u32;
                let mut nh_val = num_heads as u32;

                // SAFETY: score buffer valid, dimensions verified
                unsafe {
                    self.stream.launch_kernel(
                        module,
                        "causal_mask_softmax",
                        &config,
                        &mut [
                            std::ptr::from_mut(&mut s_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut tl_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut base_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut nh_val) as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            let handle = self.cublas_handle.as_ref().expect("cublas initialized");

            for kv_group in 0..num_kv_heads {
                let first_q_head = kv_group * heads_per_kv;
                let v_head_ptr = v_cache_ptr
                    + (kv_group * max_len * head_dim * std::mem::size_of::<f32>()) as u64;
                let p_head_ptr = score_ptr
                    + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;
                let o_head_ptr =
                    attn_out_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;

                handle.gemm_f32_strided_batched(
                    trueno_gpu::driver::GemmOp::NoTrans,
                    trueno_gpu::driver::GemmOp::NoTrans,
                    head_dim as i32,
                    m as i32,
                    total_len as i32,
                    1.0,
                    v_head_ptr,
                    head_dim as i32,
                    0,
                    p_head_ptr,
                    total_len as i32,
                    (m as usize * total_len) as i64,
                    0.0,
                    o_head_ptr,
                    q_dim as i32,
                    (head_dim) as i64,
                    heads_per_kv as i32,
                )?;
            }

            Ok(())
        }
    }

    /// PMAT-051: Scatter K/V from packed QKV buffer directly to batched KV cache.
    ///
    /// Five-Whys: bulk_scatter_kv makes 56,000 D2D copies → 128.7ms Attn+Scatter.
    /// Why? Each token×head needs cuMemcpyDtoDAsync (interleaved→contiguous).
    /// Fix: Single PTX kernel per prompt per layer scatters all K/V in one launch.
    /// Grid=(seq_len, num_kv_heads), Block=(head_dim). Each thread copies K+V f32.
    ///
    /// Replaces scatter_single_kv_to_batched_layer (which required intermediate single cache).
    const SCATTER_PACKED_KV_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry scatter_packed_kv(
    .param .u64 p_k_src,
    .param .u64 p_v_src,
    .param .u64 p_k_dst,
    .param .u64 p_v_dst,
    .param .u32 p_kv_dim,
    .param .u32 p_head_dim,
    .param .u32 p_max_len
) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd0, [p_k_src];
    ld.param.u64 %rd1, [p_v_src];
    ld.param.u64 %rd2, [p_k_dst];
    ld.param.u64 %rd3, [p_v_dst];
    ld.param.u32 %r0, [p_kv_dim];
    ld.param.u32 %r1, [p_head_dim];
    ld.param.u32 %r2, [p_max_len];

    mov.u32 %r3, %ctaid.x;   // token_idx
    mov.u32 %r4, %ctaid.y;   // head_idx
    mov.u32 %r5, %tid.x;     // dim_idx

    // src_offset = token * kv_dim + head * head_dim + dim
    mad.lo.u32 %r6, %r3, %r0, %r5;
    mad.lo.u32 %r6, %r4, %r1, %r6;

    // dst_offset = (head * max_len + token) * head_dim + dim
    mad.lo.u32 %r7, %r4, %r2, %r3;
    mad.lo.u32 %r7, %r7, %r1, %r5;

    // Convert to byte offsets
    cvt.u64.u32 %rd4, %r6;
    shl.b64 %rd4, %rd4, 2;
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;

    // K: packed to cache
    add.u64 %rd6, %rd0, %rd4;
    ld.global.f32 %f0, [%rd6];
    add.u64 %rd7, %rd2, %rd5;
    st.global.f32 [%rd7], %f0;

    // V: packed to cache
    add.u64 %rd8, %rd1, %rd4;
    ld.global.f32 %f1, [%rd8];
    add.u64 %rd9, %rd3, %rd5;
    st.global.f32 [%rd9], %f1;

    ret;
}
"#;

    /// PMAT-051: Prefill attention reading K/V directly from packed QKV buffer.
    ///
    /// Five-Whys: Attn+Scatter=128.7ms (58% of multi-prompt prefill).
    /// Why? bulk_scatter_kv makes 56,000 individual cuMemcpyDtoDAsync calls.
    /// Why? Packed K/V [M, kv_dim] needs rearranging to [num_kv_heads, max_len, head_dim].
    /// Why? cuBLAS expected lda=head_dim from cache layout.
    /// Fix: Use lda=kv_dim to read directly from packed buffer. Zero D2D copies.
    ///
    /// Then scatter K/V from packed buffer directly to batched KV cache via PTX kernel,
    /// bypassing the intermediate single KV cache entirely.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_attention_from_packed(
        &mut self,
        layer_idx: usize,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        seq_len: u32,
        q_dim: u32,
        kv_dim: u32,
        slot_idx: usize,
    ) -> Result<(), GpuError> {
        if seq_len == 0 {
            return Ok(());
        }

        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_len = seq_len as usize; // No prior cache — fresh prompt

        // Score scratch buffer
        self.ensure_cublas()?;
        let score_size = num_heads * seq_len as usize * total_len;
        self.ensure_attn_score_scratch(score_size)?;
        let score_ptr = self
            .prefill_attn_scores
            .as_ref()
            .expect("score scratch just allocated")
            .as_ptr();

        // 1. QK^T via cuBLAS — reading K directly from packed buffer (lda=kv_dim)
        //
        // K packed: [S, kv_dim] row-major → col-major per head: [head_dim, S], lda = kv_dim
        // (heads are interleaved with kv_dim stride between tokens)
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");

        for kv_group in 0..num_kv_heads {
            let first_q_head = kv_group * heads_per_kv;
            let k_head_ptr = k_buf_ptr + (kv_group * head_dim * std::mem::size_of::<f32>()) as u64;
            let q_head_ptr =
                q_buf_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;
            let s_head_ptr = score_ptr
                + (first_q_head * seq_len as usize * total_len * std::mem::size_of::<f32>()) as u64;

            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::Trans,     // K^T
                trueno_gpu::driver::GemmOp::NoTrans,   // Q
                total_len as i32,                      // m (rows of C)
                seq_len as i32,                        // n (cols of C)
                head_dim as i32,                       // k
                scale,                                 // alpha = 1/sqrt(d)
                k_head_ptr,                            // A = packed K
                kv_dim as i32,                         // lda = kv_dim (NOT head_dim)
                0,                                     // stride_a = 0 (shared K)
                q_head_ptr,                            // B = Q
                q_dim as i32,                          // ldb = q_dim
                head_dim as i64,                       // stride_b = head_dim
                0.0,                                   // beta
                s_head_ptr,                            // C = score
                total_len as i32,                      // ldc
                (seq_len as usize * total_len) as i64, // stride_c
                heads_per_kv as i32,                   // batch
            )?;
        }

        // 2. Causal mask + softmax
        if !self.modules.contains_key("causal_mask_softmax") {
            let module = self.compile_ptx(Self::CAUSAL_MASK_SOFTMAX_PTX)?;
            self.modules
                .insert("causal_mask_softmax".to_string(), module);
        }

        {
            let module = self
                .modules
                .get_mut("causal_mask_softmax")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (num_heads as u32, seq_len, 1),
                block: (32, 1, 1),
                shared_mem: 0,
            };

            let mut s_ptr = score_ptr;
            let mut m_val = seq_len;
            let mut tl_val = total_len as u32;
            let mut base_val = 0u32; // No prior cache
            let mut nh_val = num_heads as u32;

            // SAFETY: score buffer valid, dimensions verified
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "causal_mask_softmax",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut s_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut tl_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut base_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut nh_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 3. Attn × V — reading V directly from packed buffer (lda=kv_dim)
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");

        for kv_group in 0..num_kv_heads {
            let first_q_head = kv_group * heads_per_kv;
            let v_head_ptr = v_buf_ptr + (kv_group * head_dim * std::mem::size_of::<f32>()) as u64;
            let p_head_ptr = score_ptr
                + (first_q_head * seq_len as usize * total_len * std::mem::size_of::<f32>()) as u64;
            let o_head_ptr =
                attn_out_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;

            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::NoTrans,   // V
                trueno_gpu::driver::GemmOp::NoTrans,   // P
                head_dim as i32,                       // m (rows of C)
                seq_len as i32,                        // n (cols of C)
                total_len as i32,                      // k (inner dim)
                1.0,                                   // alpha
                v_head_ptr,                            // A = packed V
                kv_dim as i32,                         // lda = kv_dim (NOT head_dim)
                0,                                     // stride_a = 0 (shared V)
                p_head_ptr,                            // B = P (softmax output)
                total_len as i32,                      // ldb
                (seq_len as usize * total_len) as i64, // stride_b
                0.0,                                   // beta
                o_head_ptr,                            // C = output
                q_dim as i32,                          // ldc = q_dim
                head_dim as i64,                       // stride_c
                heads_per_kv as i32,                   // batch
            )?;
        }

        // 4. Scatter K/V from packed buffer directly to batched KV cache
        let stride = self.batched_kv_stride;
        if stride > 0 {
            let max_len = self.kv_cache_max_len;
            let slot_offset_bytes = (slot_idx * stride * std::mem::size_of::<f32>()) as u64;

            let batched_k_ptr = self
                .batched_kv_k_caches
                .get(&layer_idx)
                .ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-051: batched K cache layer {} not found",
                        layer_idx
                    ))
                })?
                .as_ptr();
            let batched_v_ptr = self
                .batched_kv_v_caches
                .get(&layer_idx)
                .ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-051: batched V cache layer {} not found",
                        layer_idx
                    ))
                })?
                .as_ptr();

            // Compile scatter kernel on first use
            if !self.modules.contains_key("scatter_packed_kv") {
                let module = self.compile_ptx(Self::SCATTER_PACKED_KV_PTX)?;
                self.modules.insert("scatter_packed_kv".to_string(), module);
            }

            let module = self
                .modules
                .get_mut("scatter_packed_kv")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (seq_len, num_kv_heads as u32, 1),
                block: (head_dim as u32, 1, 1),
                shared_mem: 0,
            };

            let mut k_src = k_buf_ptr;
            let mut v_src = v_buf_ptr;
            let mut k_dst = batched_k_ptr + slot_offset_bytes;
            let mut v_dst = batched_v_ptr + slot_offset_bytes;
            let mut kv_dim_val = kv_dim;
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;

            // SAFETY: All GPU buffers valid, grid/block dimensions verified
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "scatter_packed_kv",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut k_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut kv_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }

            // Update batched KV length for this slot
            if slot_idx < self.batched_kv_lengths.len() {
                self.batched_kv_lengths[slot_idx] = seq_len as usize;
            }
        }

        Ok(())
    }

    /// PMAT-024/026: Batched GEMV with cuBLAS GEMM fallback for prefill
    ///
    /// When M >= threshold and weights are Q4K or Q6K, uses cuBLAS GEMM
    /// (HGEMM or SGEMM) instead of batched GEMV.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_gemv_or_gemm(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        packed_input: &GpuBuffer<f32>,
        packed_output: &GpuBuffer<f32>,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n_per_seq: u32,
        k_per_seq: u32,
    ) -> Result<(), GpuError> {
        // PMAT-091: W4A16 coalesced WMMA GEMM for batched decode (M>=2).
        // Reads Q4K (0.5625 B/elem) with FP16 tensor cores — best of both worlds.
        // Interleaved weight layout fixes 864B cross-column coalescing problem.
        // At 70% WMMA efficiency: est. ~355 tok/s c=4 aggregate.
        if self.gpu_profile.w4a16_interleaved
            && m >= 2
            && !self.is_capturing
            && qtype == WeightQuantType::Q4K
            && self.interleaved_weight_cache.contains_key(&weight_ptr)
        {
            return self.interleaved_wmma_q4k_gemm(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // PMAT-090: FP8 cuBLASLt GEMM for batched decode on sm_89+.
        // PMAT-093: FP8 threshold raised from M>=2 to M>=5.
        // At M=4: FP8 conversion overhead (absmax+convert per projection) cancels
        // tensor core gains. DP4A with fused QKV+gate+up has lower launch count.
        // Benchmarks (yoga 4060L, 1900MHz, 60s, isolated):
        //   c=4 FP8: 235.0 aggregate, 16.5ms ITL
        //   c=4 DP4A: 242.1 aggregate, 15.9ms ITL (+3%)
        //   c=8 FP8: 405.6 aggregate, 18.9ms ITL
        //   c=8 DP4A: 329.1 aggregate, 23.4ms ITL (-19%)
        // FP8 crossover point: ~M=5 where tensor core BW dominates conversion cost.
        // FP8 weight cache persists from prefill — no additional warmup needed.
        if self.gpu_profile.fp8_decode
            && m >= 5
            && !self.is_capturing
            && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
        {
            return self.cublas_prefill_gemm(
                qtype,
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // PMAT-061: cuBLAS HGEMM for M>1 decode when FP16 weight cache is available.
        // Five-Whys: c=4 decode 0.56x (23.4ms/step vs llama.cpp 13.1ms).
        // Why? Batched Q4K GEMV is compute-bound at M=4 (3.25x instruction scaling).
        // Why? Each weight SB requires M activation loads + M DP4A chains.
        // Why? Dequant uses ~20 bitmask ops/SB — kernel is ALU-heavy at any M.
        // Fix: cuBLAS HGEMM uses tensor cores (memory-bound, ~1x scaling with M).
        // FP16 reads 3.5x more data but tensor cores hide M×compute.
        // Estimated: HGEMM M=4 ~15.6ms (BW-limited) vs GEMV M=4 ~23.4ms (compute-limited).
        if self.hgemm_batched_decode_active
            && m >= 2
            && !self.is_capturing
            && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
            && self.fp16_weight_cache.contains_key(&weight_ptr)
            && std::env::var("HGEMM_BATCHED_DECODE").as_deref() != Ok("0")
        {
            return self.cublas_prefill_gemm(
                qtype,
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // GH-141: Batched HW DP4A for Q4K on DP4A-capable GPUs (sm_75+).
        // Reads Q4K weights (0.5625 B/elem) + Q8_1 activations (1.125 B/elem)
        // = 1.69 B/elem total, vs cuBLAS SGEMM 8 B/elem. 4.7x less bandwidth.
        // PMAT-056: Removed !self.is_capturing guard — DP4A kernels are pure GPU
        // kernels (no H2D copies), graph-capturable. Old guard forced FP32 fallback.
        let use_batched_dp4a = qtype == WeightQuantType::Q4K
            && m >= 2
            && m <= 8
            && self.gpu_profile.q4k == crate::cuda::gpu_profile::Q4kVariant::HwDp4a
            && !self.is_prefilling
            && std::env::var("BATCHED_DP4A").as_deref() != Ok("0");

        if use_batched_dp4a {
            return self.batched_hw_dp4a_q4k_gemv_into(
                weight_ptr,
                packed_input,
                packed_output,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // GH-141: Never use cuBLAS during CUDA graph capture. cuBLAS calls
        // are not reliably capturable — they may use internal workspace or
        // stream management that the graph infrastructure cannot replay.
        // Use batched GEMV (custom PTX) instead, which is always capturable.
        let use_cublas = m >= cublas_gemm_threshold()
            && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
            && std::env::var("CUBLAS_PREFILL").as_deref() != Ok("0")
            && !self.is_capturing;

        if use_cublas {
            self.cublas_prefill_gemm(
                qtype,
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            )
        } else {
            self.batched_gemv_with_fallback(
                qtype,
                weight_ptr,
                packed_input,
                packed_output,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            )
        }
    }

    /// PMAT-061: Check if FP16 weight cache is populated (for HGEMM batched decode routing).
    pub(crate) fn has_fp16_weight_cache(&self) -> bool {
        !self.fp16_weight_cache.is_empty()
    }

    /// GH-141: Clear FP16 weight cache to free VRAM for batched KV caches.
    ///
    /// FP16 weights are only needed during HGEMM prefill (not GEMV decode).
    /// Clearing them before batched decode frees ~2944 MB on Qwen2.5 1.5B,
    /// allowing batched KV cache allocation on 8GB GPUs.
    /// The cache is re-warmed lazily on next prefill via get_or_cache_fp16_weight.
    pub(crate) fn clear_fp16_weight_cache(&mut self) {
        let cache_mb = self
            .fp16_weight_cache
            .values()
            .map(|b| b.len() * 2)
            .sum::<usize>() as f64
            / 1_048_576.0;
        let count = self.fp16_weight_cache.len();
        self.fp16_weight_cache.clear();
        // Also clear FP16 activation scratch (tied to HGEMM path)
        self.fp16_activation_scratch = None;
        self.fp16_activation_scratch_size = 0;
        if count > 0 {
            eprintln!(
                "[GH-141] Cleared FP16 weight cache: {} matrices ({:.1} MB freed)",
                count, cache_mb,
            );
        }
    }

    /// GH-141: Clear prefill CUDA graphs to free VRAM.
    /// Prefill graphs are not used during batched decode and can be recaptured later.
    pub(crate) fn clear_prefill_graphs(&mut self) {
        let count = self.prefill_graphs.len();
        self.prefill_graphs.clear();
        self.prefill_graph_input_buf = None;
        // PMAT-059: Reset prefill capture flag so graphs can be recaptured
        self.prefill_graph_capture_failed = false;
        if count > 0 {
            eprintln!("[GH-141] Cleared {} prefill graphs", count);
        }
    }

    /// PMAT-037: Pre-populate FP16 weight cache for HGEMM decode.
    ///
    /// Must be called BEFORE CUDA graph capture, because graph capture doesn't
    /// allow dynamic GPU memory allocation. Also warms up cuBLAS workspace.
    pub(crate) fn warmup_hgemm_cache(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        let start = std::time::Instant::now();
        let mut cached = 0usize;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                break;
            }
            let lw = self.get_indexed_layer(layer_idx).clone();

            // Cache all 7 weight matrices per layer
            let weights = [
                (lw.attn_q_qtype, lw.attn_q_ptr, q_dim, hidden_dim), // Q: [q_dim, hidden]
                (lw.attn_k_qtype, lw.attn_k_ptr, kv_dim, hidden_dim), // K: [kv_dim, hidden]
                (lw.attn_v_qtype, lw.attn_v_ptr, kv_dim, hidden_dim), // V: [kv_dim, hidden]
                (lw.attn_output_qtype, lw.attn_output_ptr, hidden_dim, q_dim), // O: [hidden, q_dim]
                (
                    lw.ffn_gate_qtype,
                    lw.ffn_gate_ptr,
                    intermediate_dim,
                    hidden_dim,
                ), // gate
                (lw.ffn_up_qtype, lw.ffn_up_ptr, intermediate_dim, hidden_dim), // up
                (
                    lw.ffn_down_qtype,
                    lw.ffn_down_ptr,
                    hidden_dim,
                    intermediate_dim,
                ), // down
            ];

            for (qtype, ptr, n, k) in weights {
                if ptr != 0 && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K) {
                    if self.fp16_weight_cache.get(&ptr).is_none() {
                        self.get_or_cache_fp16_weight(qtype, ptr, n, k)?;
                        cached += 1;
                    }
                }
            }
        }

        // Also cache LM head
        if self.lm_head_ptr != 0
            && (self.lm_head_qtype == WeightQuantType::Q4K
                || self.lm_head_qtype == WeightQuantType::Q6K)
        {
            let lm_ptr = self.lm_head_ptr;
            let lm_qtype = self.lm_head_qtype;
            if self.fp16_weight_cache.get(&lm_ptr).is_none() {
                self.get_or_cache_fp16_weight(lm_qtype, lm_ptr, vocab_size, hidden_dim)?;
                cached += 1;
            }
        }

        // PMAT-063: Pre-allocate cuBLAS workspace for CUDA graph capture.
        // Without this, graph capture forces workspace-free algorithms (7x slower).
        // 32 MB workspace covers all standard GEMM shapes for Qwen 1.5B.
        if cached > 0 {
            self.ensure_cublas_workspace()?;

            // PMAT-063: Warm up cuBLAS JIT for common prefill sequence lengths.
            // First cuBLAS GEMM per (M,N,K) takes ~42ms (algorithm selection);
            // subsequent calls take <0.1ms. Pre-warming all common M values
            // moves this one-time cost from first-request to model load.
            //
            // Measured shapes for Qwen2.5-Coder-1.5B (M=seq_len):
            //   (N=1536, K=1536): Q/O projections
            //   (N=256,  K=1536): K/V projections (GQA, 2 KV heads)
            //   (N=8960, K=1536): gate/up projections
            //   (N=1536, K=8960): down projection
            // Each unique M triggers ~42ms JIT for first shape + ~5ms for remaining.
            let max_m = 512i32; // Cover sequences up to 512 tokens
            let max_dim = std::cmp::max(hidden_dim, intermediate_dim) as usize;
            self.ensure_fp16_activation_scratch(max_m as usize * max_dim)?;
            let dummy_input = GpuBuffer::<u16>::new(&self.context, max_m as usize * max_dim)?;
            let dummy_out = GpuBuffer::<f32>::new(&self.context, max_m as usize * max_dim)?;

            if let Some((&_ptr, fp16_buf)) = self.fp16_weight_cache.iter().next() {
                let handle = self.cublas_handle.as_ref().expect("cublas initialized");

                let shapes = [
                    (hidden_dim as i32, hidden_dim as i32), // Q/O
                    (
                        (self.kv_num_kv_heads * self.kv_head_dim) as i32,
                        hidden_dim as i32,
                    ), // K/V
                    (intermediate_dim as i32, hidden_dim as i32), // gate/up
                    (hidden_dim as i32, intermediate_dim as i32), // down
                ];

                // Warm common M values: powers of 2 + typical chat lengths
                let m_values: &[i32] = &[8, 16, 32, 64, 125, 128, 256, 512];
                let mut warmed = 0u32;
                for &m in m_values {
                    for &(n, k) in &shapes {
                        if n > 0 && k > 0 {
                            let _ = handle.gemm_f16_to_f32(
                                trueno_gpu::driver::GemmOp::Trans,
                                trueno_gpu::driver::GemmOp::NoTrans,
                                n,
                                m,
                                k,
                                0.0,
                                fp16_buf.as_ptr(),
                                k,
                                dummy_input.as_ptr(),
                                k,
                                0.0,
                                dummy_out.as_ptr(),
                                n,
                            );
                            warmed += 1;
                        }
                    }
                }
                self.stream.synchronize()?;
                eprintln!(
                    "[PMAT-063] cuBLAS JIT warmed: {} shapes ({}×{} M×NK combos)",
                    warmed,
                    m_values.len(),
                    shapes.len()
                );
            }
        }

        let elapsed = start.elapsed();
        let cache_mb = self
            .fp16_weight_cache
            .values()
            .map(|b| b.len() * 2)
            .sum::<usize>() as f64
            / 1_048_576.0;
        eprintln!(
            "[PMAT-037] FP16 weight cache: {} matrices cached ({:.1} MB) in {:.1}ms",
            cached,
            cache_mb,
            elapsed.as_secs_f64() * 1000.0,
        );

        Ok(())
    }

    /// PMAT-053: Pre-populate FP8 E4M3 weight cache.
    ///
    /// Similar to `warmup_hgemm_cache` but caches as FP8 (1 byte/elem = 50% of FP16).
    /// Must be called BEFORE CUDA graph capture. Requires sm_89+.
    pub(crate) fn warmup_fp8_cache(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        if self.gpu_profile.cc < 89 {
            eprintln!(
                "[PMAT-053] FP8 cache skipped: requires sm_89+ (have {})",
                self.gpu_profile.sm_target
            );
            return Ok(());
        }

        let start = std::time::Instant::now();
        self.ensure_cublas()?;
        let mut cached = 0usize;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                break;
            }
            let lw = self.get_indexed_layer(layer_idx).clone();

            let weights = [
                (lw.attn_q_qtype, lw.attn_q_ptr, q_dim, hidden_dim),
                (lw.attn_k_qtype, lw.attn_k_ptr, kv_dim, hidden_dim),
                (lw.attn_v_qtype, lw.attn_v_ptr, kv_dim, hidden_dim),
                (lw.attn_output_qtype, lw.attn_output_ptr, hidden_dim, q_dim),
                (
                    lw.ffn_gate_qtype,
                    lw.ffn_gate_ptr,
                    intermediate_dim,
                    hidden_dim,
                ),
                (lw.ffn_up_qtype, lw.ffn_up_ptr, intermediate_dim, hidden_dim),
                (
                    lw.ffn_down_qtype,
                    lw.ffn_down_ptr,
                    hidden_dim,
                    intermediate_dim,
                ),
            ];

            for (qtype, ptr, n, k) in weights {
                if ptr != 0 && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K) {
                    if !self.fp8_weight_cache.contains_key(&ptr) {
                        self.get_or_cache_fp8_weight(qtype, ptr, n, k)?;
                        cached += 1;
                    }
                }
            }
        }

        // Also cache LM head
        if self.lm_head_ptr != 0
            && (self.lm_head_qtype == WeightQuantType::Q4K
                || self.lm_head_qtype == WeightQuantType::Q6K)
        {
            let lm_ptr = self.lm_head_ptr;
            let lm_qtype = self.lm_head_qtype;
            if !self.fp8_weight_cache.contains_key(&lm_ptr) {
                self.get_or_cache_fp8_weight(lm_qtype, lm_ptr, vocab_size, hidden_dim)?;
                cached += 1;
            }
        }

        self.stream.synchronize()?;

        // PMAT-082: cuBLASLt FP8 JIT warmup — eliminate 3ms first-request latency.
        // First cuBLASLt FP8 GEMM triggers JIT compilation; warm it here at model load.
        if cached > 0 {
            if self.cublaslt_handle.is_none() {
                self.cublaslt_handle = Some(trueno_gpu::driver::CublasLtHandle::new()?);
            }

            // Use smallest realistic M (padded to 16) with the first cached FP8 weight.
            // One GEMM is enough — cuBLASLt JIT is per-handle, not per-shape.
            let m_warmup: i32 = 16;
            let n_warmup = hidden_dim as i32;
            let k_warmup = hidden_dim as i32;
            let warmup_input_count = m_warmup as usize * k_warmup as usize;
            let warmup_output_count = n_warmup as usize * m_warmup as usize;

            // Allocate scratch BEFORE borrowing lt_handle (avoid borrow conflict)
            self.ensure_fp8_activation_scratch(warmup_input_count)?;
            self.ensure_fp16_activation_scratch(warmup_output_count)?;
            let input_ptr = self
                .fp8_activation_scratch
                .as_ref()
                .expect("scratch just allocated")
                .as_ptr();
            let output_ptr = self
                .fp16_activation_scratch
                .as_ref()
                .expect("scratch just allocated")
                .as_ptr();

            // Extract weight ptr before borrowing lt_handle
            let fp8_weight_ptr = self.fp8_weight_cache.values().next().map(|b| b.as_ptr());

            if let Some(w_ptr) = fp8_weight_ptr {
                let lt_handle = self.cublaslt_handle.as_ref().expect("just created");
                let _ = lt_handle.gemm_fp8_e4m3_to_f16(
                    trueno_gpu::driver::GemmOp::Trans,
                    trueno_gpu::driver::GemmOp::NoTrans,
                    n_warmup,
                    m_warmup,
                    k_warmup,
                    1.0, // dummy alpha
                    w_ptr,
                    k_warmup,
                    input_ptr,
                    k_warmup,
                    0.0,
                    output_ptr,
                    n_warmup,
                    &self.stream,
                );
                self.stream.synchronize()?;
                eprintln!("[PMAT-082] cuBLASLt FP8 JIT warmed ({n_warmup}×{m_warmup}×{k_warmup})");
            }
        }

        let elapsed = start.elapsed();
        let cache_mb = self
            .fp8_weight_cache
            .values()
            .map(|b| b.len())
            .sum::<usize>() as f64
            / 1_048_576.0;
        eprintln!(
            "[PMAT-053] FP8 weight cache: {} matrices cached ({:.1} MB) in {:.1}ms",
            cached,
            cache_mb,
            elapsed.as_secs_f64() * 1000.0,
        );

        Ok(())
    }

    /// PMAT-053: Clear FP8 weight cache to free VRAM.
    pub(crate) fn clear_fp8_weight_cache(&mut self) {
        let cache_mb = self
            .fp8_weight_cache
            .values()
            .map(|b| b.len())
            .sum::<usize>() as f64
            / 1_048_576.0;
        let count = self.fp8_weight_cache.len();
        self.fp8_weight_cache.clear();
        self.fp8_activation_scratch = None;
        self.fp8_activation_scratch_size = 0;
        if count > 0 {
            eprintln!(
                "[PMAT-053] Cleared FP8 weight cache: {} matrices ({:.1} MB freed)",
                count, cache_mb,
            );
        }
    }

    /// PMAT-069: Fused prefill attention — single PTX kernel per layer.
    ///
    /// Replaces 5 cuBLAS + PTX launches (2 QK^T + 1 softmax + 2 Attn×V) with
    /// 1 fused kernel launch using online softmax (no N×N materialization).
    /// For 28 layers: 140 → 28 launches. Saves ~7.5ms TTFT.
    ///
    /// After attention, scatters K/V to KV cache for subsequent decode tokens.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_attention_fused(
        &mut self,
        layer_idx: usize,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        m: u32,
        q_dim: u32,
        kv_dim: u32,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        heads_per_kv: usize,
        max_len: usize,
    ) -> Result<(), GpuError> {
        use trueno_gpu::kernels::{Kernel as _, PrefillAttentionKernel};

        // Compile and cache fused attention kernel
        if !self.modules.contains_key("fused_prefill_attn") {
            let kernel = PrefillAttentionKernel::new(head_dim as u32, heads_per_kv as u32);
            let ptx = kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&ptx)?;
            self.modules
                .insert("fused_prefill_attn".to_string(), module);
            eprintln!(
                "[PMAT-069] Compiled fused prefill attention: head_dim={}, heads_per_kv={}",
                head_dim, heads_per_kv,
            );
        }

        // Launch fused attention kernel
        {
            let module = self
                .modules
                .get_mut("fused_prefill_attn")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (num_heads as u32, 1, 1),
                block: (32, 1, 1),
                shared_mem: 0,
            };

            let mut q = q_buf_ptr;
            let mut k = k_buf_ptr;
            let mut v = v_buf_ptr;
            let mut o = attn_out_ptr;
            let mut m_val = m;
            let mut qs = q_dim;
            let mut kvs = kv_dim;
            let mut nqh = num_heads as u32;

            // SAFETY: GPU buffers valid, grid covers all Q heads, block is 1 warp
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "fused_prefill_attention_causal",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut q) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut o) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut qs) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut kvs) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut nqh) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Scatter K/V from packed buffer to KV cache (needed for subsequent decode)
        let k_key = format!("kv_{layer_idx}_k");
        let v_key = format!("kv_{layer_idx}_v");
        let k_cache_ptr = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?
            .as_ptr();
        let v_cache_ptr = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?
            .as_ptr();

        if !self.modules.contains_key("scatter_packed_kv") {
            let module = self.compile_ptx(Self::SCATTER_PACKED_KV_PTX)?;
            self.modules.insert("scatter_packed_kv".to_string(), module);
        }

        let module = self
            .modules
            .get_mut("scatter_packed_kv")
            .expect("just inserted");
        let config = LaunchConfig {
            grid: (m, num_kv_heads as u32, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem: 0,
        };

        let mut k_src = k_buf_ptr;
        let mut v_src = v_buf_ptr;
        let mut k_dst = k_cache_ptr;
        let mut v_dst = v_cache_ptr;
        let mut kv_dim_val = kv_dim;
        let mut head_dim_val = head_dim as u32;
        let mut max_len_val = max_len as u32;

        // SAFETY: All GPU buffers valid, grid/block dimensions verified
        unsafe {
            self.stream.launch_kernel(
                module,
                "scatter_packed_kv",
                &config,
                &mut [
                    std::ptr::from_mut(&mut k_src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut kv_dim_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Update single KV cache length
        self.kv_cache_lengths.insert(layer_idx, m as usize);

        Ok(())
    }

    // ========================================================================
    // PMAT-091: Interleaved Q4K weight cache for coalesced WMMA GEMM
    // ========================================================================

    /// Warmup interleaved Q4K weight cache at model init.
    ///
    /// Downloads Q4K weights from GPU, repacks to column-interleaved tile layout
    /// on CPU, re-uploads to GPU. Only runs once per model load.
    /// Total overhead: ~200ms for 197 matrices (Q4K data ~850MB, PCIe limited).
    pub(crate) fn warmup_interleaved_cache(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        _vocab_size: u32,
    ) -> Result<(), GpuError> {
        let start = std::time::Instant::now();
        let mut cached = 0usize;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                break;
            }
            let lw = self.get_indexed_layer(layer_idx).clone();

            let weights = [
                (lw.attn_q_qtype, lw.attn_q_ptr, q_dim, hidden_dim),
                (lw.attn_k_qtype, lw.attn_k_ptr, kv_dim, hidden_dim),
                (lw.attn_v_qtype, lw.attn_v_ptr, kv_dim, hidden_dim),
                (lw.attn_output_qtype, lw.attn_output_ptr, hidden_dim, q_dim),
                (
                    lw.ffn_gate_qtype,
                    lw.ffn_gate_ptr,
                    intermediate_dim,
                    hidden_dim,
                ),
                (lw.ffn_up_qtype, lw.ffn_up_ptr, intermediate_dim, hidden_dim),
                (
                    lw.ffn_down_qtype,
                    lw.ffn_down_ptr,
                    hidden_dim,
                    intermediate_dim,
                ),
            ];

            for (qtype, ptr, n, k) in weights {
                if ptr != 0 && qtype == WeightQuantType::Q4K {
                    if !self.interleaved_weight_cache.contains_key(&ptr) {
                        self.get_or_cache_interleaved_weight(ptr, n, k)?;
                        cached += 1;
                    }
                }
            }
        }

        // NOTE: LM head skipped — WMMA for decode only, LM head uses existing path

        self.stream.synchronize()?;
        eprintln!(
            "[PMAT-091] Interleaved Q4K cache warmed: {} matrices in {:?}",
            cached,
            start.elapsed()
        );
        Ok(())
    }

    /// Get or create interleaved Q4K weight buffer for a given weight pointer.
    ///
    /// Downloads Q4K bytes from GPU, repacks to column-interleaved tile layout
    /// via `repack_q4k_interleaved()`, uploads interleaved data to new GPU buffer.
    fn get_or_cache_interleaved_weight(
        &mut self,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        if let Some(buf) = self.interleaved_weight_cache.get(&weight_ptr) {
            return Ok(buf.as_ptr());
        }

        let n_usize = n as usize;
        let k_usize = k as usize;
        let num_sb = k_usize / 256;
        let q4k_byte_count = n_usize * num_sb * 144;

        // Step 1: Download Q4K bytes from GPU to CPU
        let weight_view = unsafe { GpuBuffer::<u8>::from_raw_parts(weight_ptr, q4k_byte_count) };
        let mut q4k_host = vec![0u8; q4k_byte_count];
        weight_view.copy_to_host(&mut q4k_host)?;
        std::mem::forget(weight_view); // Don't free — we don't own this memory

        // Step 2: Repack on CPU
        let interleaved = trueno_gpu::kernels::repack_q4k_interleaved(&q4k_host, n_usize, k_usize);

        // Step 3: Upload interleaved data to GPU
        let il_buf = GpuBuffer::<u8>::from_host(&self.context, &interleaved)?;
        let il_ptr = il_buf.as_ptr();

        self.interleaved_weight_cache.insert(weight_ptr, il_buf);
        Ok(il_ptr)
    }

    /// PMAT-091: Launch interleaved WMMA Q4K GEMM for batched decode.
    ///
    /// Same grid/block as multi-warp WMMA but reads from column-interleaved
    /// weight layout. Adjacent threads access adjacent qs bytes — perfect coalescing.
    #[allow(clippy::too_many_arguments)]
    fn interleaved_wmma_q4k_gemm(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached interleaved weight pointer
        let il_ptr = self
            .interleaved_weight_cache
            .get(&weight_ptr)
            .expect("interleaved weight should be cached by warmup")
            .as_ptr();

        // Pad M and N to multiples of 32 for 2×2 WMMA tile safety
        let m_padded = (m + 31) & !31;
        let n_padded = (n + 31) & !31;
        let needs_padding = m_padded != m || n_padded != n;

        let kernel_type = KernelType::InterleavedWmmaQ4KGemm {
            m: m_padded,
            n: n_padded,
            k,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("interleaved_wmma_q4k_gemm_{m_padded}_{n_padded}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        // If padding needed, use WMMA scratch buffer
        let actual_output_ptr = if needs_padding {
            let padded_count = m_padded as usize * n_padded as usize;
            self.ensure_wmma_scratch(padded_count)?;
            self.wmma_scratch
                .as_ref()
                .expect("wmma scratch allocated")
                .as_ptr()
        } else {
            packed_output_ptr
        };

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/32), ceil(M/32)), Block: 128 (4 warps for 2×2 WMMA tiles)
        let grid_x = n_padded / 32;
        let grid_y = m_padded / 32;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 128, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = il_ptr;
        let mut ptr_c = actual_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Copy valid [M, N] from padded buffer to actual output
        if needs_padding {
            self.stream.synchronize()?;
            for row in 0..m {
                let src_offset = row as u64 * n_padded as u64 * 4;
                let dst_offset = row as u64 * n as u64 * 4;
                self.stream.memcpy_dtod_sync(
                    packed_output_ptr + dst_offset,
                    actual_output_ptr + src_offset,
                    n as usize * 4,
                )?;
            }
        }

        Ok(())
    }
}
