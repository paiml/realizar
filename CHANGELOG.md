# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CHANGELOG.md for tracking version history

## [0.1.0] - 2024-11-18

### Added
- Pure Rust ML inference engine from scratch
- GGUF format parser (v3 support)
  - Full metadata parsing (all value types including Arrays)
  - Tensor information parsing
  - Quantization type support (Q4_0, Q8_0)
- Safetensors format parser
  - JSON header parsing
  - Tensor data loading
  - Zero-copy tensor access
- Transformer model implementation (LLaMA architecture)
  - Multi-head attention with RoPE
  - Feed-forward networks with SwiGLU
  - RMSNorm layer normalization
  - Configurable layers, heads, dimensions
- Tokenization support
  - Basic character-level tokenizer
  - BPE (Byte Pair Encoding) tokenizer
  - SentencePiece tokenizer with unigram model
- Text generation strategies
  - Greedy sampling
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Temperature scaling
  - Configurable generation parameters
- REST API with Axum
  - `/health` - Health check endpoint
  - `/tokenize` - Text tokenization endpoint
  - `/generate` - Text generation endpoint
  - Demo mode for testing
- CLI binary (`realizar`)
  - `serve` command with `--demo` flag
  - `info` command for version information
  - Configurable host and port
- Comprehensive test suite
  - 260 total tests (211 unit + 42 property + 7 integration)
  - 95.46% code coverage (region)
  - 100% mutation score on api.rs
  - Property-based tests with proptest
  - Integration tests for CLI
- Performance benchmarks
  - Tensor operations benchmark suite
  - Inference benchmark suite
  - Sub-millisecond generation (<1ms p50)
- Examples
  - `inference.rs` - Model inference demonstration
  - `tokenization.rs` - Tokenizer comparison
  - `api_server.rs` - HTTP server demo
- GPU acceleration via Trueno (optional feature)

### Performance
- Forward pass (1 token): ~17.5 µs
- 5-token generation: ~504 µs
- 10-token generation: ~1.54 ms
- 20-token generation: ~5.52 ms

### Quality Metrics
- Test coverage: 95.46% (region), 91.33% (function)
- TDG Score: 93.9/100 (A grade)
- Mutation score: 100% (api.rs)
- Clippy: 0 warnings
- Rustfmt: compliant

### Dependencies
- Trueno v0.2.2 - SIMD/GPU compute primitives
- Axum v0.7 - HTTP server framework
- Tokio v1 - Async runtime
- Clap v4 - CLI argument parsing
- Serde v1 - Serialization
- Thiserror v1 - Error handling

[Unreleased]: https://github.com/paiml/realizar/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/realizar/releases/tag/v0.1.0
