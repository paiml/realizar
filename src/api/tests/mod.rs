//! API Tests
//!
//! Split into parts for PMAT compliance (<2000 lines per file).
//!
//! Part organization:
//! - part_01: Unit tests, clean_chat_output, health/metrics endpoints (IMP-144-152)
//! - part_02: Generate endpoint tests, streaming tests
//! - part_03: Chat completion tests, OpenAI compatibility
//! - part_04: GPU inference tests (IMP-116+)
//! - part_05: Additional coverage tests
//! - part_06: Error response coverage (PMAT-803)
//! - part_07: Realize handlers coverage (Phase 37 - Scenario Blitz)
//! - part_08: OpenAI handlers coverage
//! - part_09: OpenAI handlers extended coverage
//! - part_10: Realize handlers extended coverage (ModelLineage, ReloadResponse, etc.)
//! - part_11: GPU handlers coverage (GpuBatchRequest, GpuBatchResponse, BatchConfig, etc.)
//! - part_12: OpenAI/Realize handlers - Request/Response type serialization
//! - part_13: OpenAI/Realize handlers - HTTP endpoint error paths and streaming
//! - part_14: Additional coverage tests
//! - part_15: T-COV-95 Directive 2: In-Process API Falsification (GPU/CUDA/quantized paths)
//! - part_16: T-COV-95 Popper Phase 2: Combinatorial API Sweep (stream/temp/max_tokens/invalid)

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
mod part_18; // T-COV-95 Coverage Bridge B2+B3 (GPU handlers, Realize/OpenAI handlers, AppState)
mod part_19; // T-COV-95 Deep Coverage Bridge (BatchConfig, ContinuousBatchResponse, streaming types, endpoints)
mod part_20; // T-COV-95 Deep Coverage Bridge (ContextWindow, format_chat, clean_chat, HTTP handlers, serde)
mod part_21; // T-COV-95 Extended Coverage (build_trace_data, streaming types, request/response serde)
mod part_22; // T-COV-95 APR handlers coverage (predict, explain, audit, serde, error paths)
mod part_23; // T-COV-95 gpu_handlers + realize_handlers coverage (BatchConfig, ContextWindow, format_chat)
mod part_24; // T-COV-95 Protocol Falsification: Potemkin Village GPU Mocks
mod part_25; // T-COV-95 Chaotic Citizens: GPU Batch Resilience Falsification
