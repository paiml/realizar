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

mod part_01;
mod part_02;
mod part_03;
mod part_04;
mod part_05;
mod part_06;
mod part_07;
