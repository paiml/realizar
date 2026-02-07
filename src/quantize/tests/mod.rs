mod part_01;
mod part_02;
mod part_03;
mod part_04;
mod part_05a;
mod part_05b;
mod part_06;
mod part_07;
mod part_08; // Phase 36: Fused K-quantization math kernel tests (proptest)
mod part_09; // Phase 36: SIMD helper function tests (proptest)
mod part_10; // Phase 37: Additional SIMD coverage tests
mod part_11; // Phase 38: Extended coverage for quantize/types.rs
mod part_12; // Phase 38: Deep coverage tests for activation.rs
mod part_13; // Phase 39: Additional coverage for quantize/mod.rs uncovered paths
mod part_14; // Phase 39: Parallel K-quantization coverage tests for parallel_k.rs
mod part_15; // Activation function edge cases: special floats, boundaries, SIMD paths
mod part_16; // F16 LUT, Q8K dequantize, SimdBackend Display, DequantStats, extract_scale_min blocks 4-7
mod part_17; // Additional activation.rs coverage: AVX2 remainder loops, fused function edge cases
mod part_18; // Phase 40: SIMD coverage enhancement (simd.rs edge cases, alignment, fallback paths)
mod part_19; // Phase 45: Parallel dequantization coverage (parallel_dequant.rs)
mod part_21; // Phase 46: Comprehensive activation.rs coverage
mod part_23; // Additional SIMD coverage: f16 subnormal, extract_scale_min odd idx, horizontal sums, AVX2 RoPE
mod part_24; // Comprehensive coverage for quantize/mod.rs functions (f16 LUT, Q8K into, InterleavedQ4K, fused matvec)
mod part_25; // Popperian SIMD Falsification: performance tests, SIMD/scalar parity, path verification
mod part_26; // T-COV-001: Error path and edge case coverage tests
mod part_27; // T-COV-95 Directive 3: Scalar exhaustion for fused_k.rs
mod part_28; // T-COV-95 Directive 4: Performance Falsification Gate (SIMD vs scalar)
mod part_29; // T-COV-95 Coverage Bridge B7 (Q4_1, Q5_0, Q5_1, Q2_K dequantization)
mod part_30; // T-COV-95 Deep Coverage Bridge (q8k_into, q8_blocks, InterleavedQ4K, f16 LUT)
mod part_31; // T-COV-95 Coverage Bridge (fused Q4_0/Q8_0 matvec, extract_scale_min blocks 4-7)
mod part_32; // T-COV-95 Coverage Bridge (q8k_into, dequant q4_0/q8_0/q4_1/q5_0/q5_1)
mod part_33; // T-COV-95 Extended Coverage (Q8 blocks, dequant edge cases, block boundaries)
mod part_34; // T-COV-95 Phase 50: Deep coverage for quantize/mod.rs and activation.rs
mod part_35; // T-COV-95 Phase 50: Deep coverage for fused_q5k_q6k.rs and fused_k.rs
mod part_36; // T-COV-95: Deep inner-loop coverage for fused_k.rs (scalar + SIMD parity)
