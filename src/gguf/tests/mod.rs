//! GGUF module tests - shattered from gguf_monolith.rs (PMAT-802)
//!
//! This module organizes tests from the monolith into focused part files
//! for better maintainability. Monolith reduced from 54,792 to 29,894 lines.
//!
//! ## Part Files
//!
//! - `part_01.rs`: Config + IMP-101 (RoPE, Causal Attention) + IMP-105 (GQA)
//! - `part_02.rs`: IMP-106 + IMP-107 + IMP-108 (Batch operations)
//! - `part_03.rs`: IMP-109 + IMP-110 (Fused kernels, parallel attention)
//! - `part_04.rs`: IMP-112, IMP-111, IMP-113, IMP-114, PARITY-114
//! - `part_05.rs`: IMP-115 - IMP-119
//! - `part_06.rs`: IMP-121-125, IMP-123, IMP-129, PARITY-002
//! - `part_07.rs`: PARITY-005 - PARITY-010 (Contiguous KV Cache, Batch Processing, Benchmarks)
//! - `part_08.rs`: PARITY-011 - PARITY-012 (Integration QA, GPU Optimization)
//! - `part_09.rs`: PARITY-013 - PARITY-017 (GPU Optimization, Multi-Request Batching)
//! - `part_10.rs`: PARITY-050 - PARITY-054 (Batch Inference Analysis & API Integration)
//! - `part_11.rs`: PARITY-055 - PARITY-063 (Batch Throughput Benchmarking & Speculative Phase 2)
//! - `part_12.rs`: PARITY-070 - PARITY-077 (Phase 3: Quantized Attention)
//! - `part_13.rs`: PARITY-078 - PARITY-086 (Phase 4/5: FlashAttention-2 & Stream-K)
//! - `part_14.rs`: PARITY-018 - PARITY-025 (GPU Batch FFN & Request Infrastructure)
//! - `part_15.rs`: PARITY-026 - PARITY-034 (FlashAttention & Infrastructure)
//! - `part_16.rs`: PARITY-035 (Chunked Prefill for Long Contexts)
//! - `part_17.rs`: Phase 33 - Forward Pass Coverage (forward/core.rs)
//! - `part_18.rs`: Phase 33 - GGUF Loader Coverage (loader.rs)
//! - `part_19.rs`: Phase 34 - Thread-safe Cache (cached/sync.rs)
//! - `part_20.rs`: Phase 34 - Matmul Coverage (matmul.rs)
//! - `part_21.rs`: Phase 34 - Transformer Structure (transformer.rs)
//! - `part_22.rs`: Phase 35 - GGUF Edge Cases (loader.rs, types.rs)
//! - `part_23.rs`: Phase 36 - GGUF Parsing and Error Handling (mod.rs coverage)

mod part_01;
mod part_02;
mod part_03;
mod part_04;
mod part_05;
mod part_06;
mod part_07;
mod part_08;
mod part_09;
mod part_10;
mod part_11;
mod part_12;
mod part_13;
mod part_14;
mod part_15;
mod part_16;
mod part_17;
mod part_18;
mod part_19;
mod part_20;
mod part_21;
mod part_22;
mod part_23;
mod part_24;
