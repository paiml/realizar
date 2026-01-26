//! APR Transformer Tests (PMAT-803)
//!
//! Split into parts for PMAT compliance (<2000 lines per file).
//!
//! Part organization:
//! - apr_transformer: Core AprTransformer tests (initialization, forward, error handling)
//! - q4_simd: Quantized Q4_0 SIMD transformer tests
//! - coverage: Comprehensive coverage tests for loader, dequant, config, helpers
//! - part_02: Additional coverage for forward pass, attention, KV cache edge cases

mod apr_transformer;
mod coverage;
mod part_02;
mod q4_simd;
