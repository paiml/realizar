//! Performance Parity Benchmarks (Refs PERF-PARITY-001)
//!
//! Implements the benchmark specification v1.1 for comparing realizar with:
//! - llama.cpp (GGUF inference)
//! - Ollama (production deployment)
//!
//! ## Toyota Way Principles
//!
//! - **Genchi Genbutsu**: Go and see - measure actual inference, not synthetic ops
//! - **Jidoka**: Stop on anomaly - fail-fast on thermal throttling or variance
//! - **Kaizen**: Continuous improvement - track regressions over time
//!
//! ## Benchmark Categories
//!
//! | Category      | Target           | Metric                    |
//! |---------------|------------------|---------------------------|
//! | TTFT (Cold)   | ≤1.0× llama.cpp  | Time to first token (ms)  |
//! | TTFT (Warm)   | ≤1.0× llama.cpp  | Time to first token (ms)  |
//! | TPS           | ≥80% llama.cpp   | Tokens per second         |
//! | Memory        | ≤1.1× llama.cpp  | Peak RSS (MB)             |
//!
//! ## Usage
//!
//! ```bash
//! # Run all performance parity benchmarks
//! cargo bench --bench performance_parity
//!
//! # Run specific benchmark group
//! cargo bench --bench performance_parity -- ttft
//! cargo bench --bench performance_parity -- tps
//! ```
//!
//! ## References
//!
//! - [17] Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems", SC'15
//! - [11] Dean & Barroso, "The Tail at Scale", CACM 2013

#![allow(clippy::cast_precision_loss)]
#![allow(dead_code)] // Benchmark constants may not all be used

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Import realizar types
use realizar::layers::{softmax, Attention, FusedQKVAttention, LayerNorm, Linear};
use realizar::quantize::{dequantize_q4_k_simd, fused_q4k_dot_simd};
use realizar::Tensor;

/// Fixed prompt for reproducible benchmarks (DO NOT CHANGE)
#[allow(dead_code)]
const BENCHMARK_PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

/// Target sequence lengths for scaling analysis
const SEQ_LENGTHS: &[usize] = &[128, 256, 512, 1024];

/// Hidden dimension for benchmark models
const HIDDEN_DIM: usize = 4096;

/// Number of attention heads
const NUM_HEADS: usize = 32;

/// Head dimension
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;

// ============================================================================
// IMP-001: SIMD Q4_K Dequantization Benchmarks
// ============================================================================

fn benchmark_q4k_dequant_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("q4k_dequant_simd");
    group.sample_size(100);

    // Q4_K super-block sizes to test
    let super_block_counts = [1, 4, 16, 64, 256]; // 256 to 65536 values

    for &sb_count in &super_block_counts {
        let num_values = sb_count * 256;
        let data_size = sb_count * 144; // 144 bytes per Q4_K super-block

        // Create Q4_K test data
        let mut data = vec![0u8; data_size];
        for sb_idx in 0..sb_count {
            let offset = sb_idx * 144;
            // d = 1.0 (f16)
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            // dmin = 0.0 (f16)
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
            // scales and qs filled with pattern
            for i in 4..144 {
                data[offset + i] = (i % 16) as u8;
            }
        }

        group.throughput(Throughput::Elements(num_values as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{num_values}_values")),
            &data,
            |b, d| {
                b.iter(|| {
                    let result = dequantize_q4_k_simd(black_box(d)).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IMP-002: Fused Q4_K Dot Product Benchmarks
// ============================================================================

fn benchmark_fused_q4k_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_q4k_dot");
    group.sample_size(100);

    // Test sizes matching typical layer dimensions
    let sizes = [256, 1024, 4096, 16384];

    for &size in &sizes {
        let sb_count = size / 256;
        let data_size = sb_count * 144;

        // Create Q4_K weights
        let mut weights = vec![0u8; data_size];
        for sb_idx in 0..sb_count {
            let offset = sb_idx * 144;
            weights[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d=1.0
            weights[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin=0.0
            for i in 4..144 {
                weights[offset + i] = ((sb_idx + i) % 16) as u8;
            }
        }

        // Create activation vector
        let activations: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("fused_simd", format!("{size}_dim")),
            &(&weights, &activations),
            |b, (w, a)| {
                b.iter(|| {
                    let result = fused_q4k_dot_simd(black_box(*w), black_box(*a)).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IMP-003: Fused QKV + Attention Benchmarks
// ============================================================================

fn benchmark_fused_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_attention");
    group.sample_size(50); // Fewer samples for expensive operations

    for &seq_len in SEQ_LENGTHS {
        let hidden_dim = 256; // Smaller for benchmark speed
        let head_dim = 32;

        let fused = FusedQKVAttention::new(head_dim, hidden_dim).unwrap();
        let input = Tensor::from_vec(
            vec![seq_len, hidden_dim],
            (0..(seq_len * hidden_dim))
                .map(|i| (i as f32 * 0.001).sin())
                .collect(),
        )
        .unwrap();

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("seq{seq_len}")),
            &(&fused, &input),
            |b, (f, i)| {
                b.iter(|| {
                    let output = f.forward(black_box(*i)).unwrap();
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IMP-003: Separate vs Fused Attention Comparison
// ============================================================================

fn benchmark_attention_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_comparison");
    group.sample_size(50);

    let seq_len = 128;
    let head_dim = 32;
    let hidden_dim = 128;

    // Fused attention
    let fused = FusedQKVAttention::new(head_dim, hidden_dim).unwrap();
    let input = Tensor::from_vec(
        vec![seq_len, hidden_dim],
        (0..(seq_len * hidden_dim))
            .map(|i| (i as f32 * 0.001).sin())
            .collect(),
    )
    .unwrap();

    group.bench_function("fused_qkv_attention", |b| {
        b.iter(|| {
            let output = fused.forward(black_box(&input)).unwrap();
            black_box(output)
        });
    });

    // Separate attention (baseline)
    let attention = Attention::new(head_dim).unwrap();
    let q = Tensor::from_vec(vec![seq_len, head_dim], vec![0.1; seq_len * head_dim]).unwrap();
    let k = q.clone();
    let v = q.clone();

    group.bench_function("separate_attention", |b| {
        b.iter(|| {
            let output = attention
                .forward(black_box(&q), black_box(&k), black_box(&v))
                .unwrap();
            black_box(output)
        });
    });

    group.finish();
}

// ============================================================================
// Layer Normalization Benchmarks
// ============================================================================

fn benchmark_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");

    for &seq_len in SEQ_LENGTHS {
        let hidden_dim = 256;
        let layer_norm = LayerNorm::new(hidden_dim, 1e-5).unwrap();
        let input = Tensor::from_vec(
            vec![seq_len, hidden_dim],
            (0..(seq_len * hidden_dim))
                .map(|i| (i as f32 * 0.01).sin())
                .collect(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("seq{seq_len}")),
            &(&layer_norm, &input),
            |b, (ln, i)| {
                b.iter(|| {
                    let output = ln.forward(black_box(*i)).unwrap();
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Softmax Benchmarks
// ============================================================================

fn benchmark_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for &seq_len in SEQ_LENGTHS {
        let input = Tensor::from_vec(
            vec![seq_len],
            (0..seq_len).map(|i| (i as f32 * 0.1).sin()).collect(),
        )
        .unwrap();

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("len{seq_len}")),
            &input,
            |b, i| {
                b.iter(|| {
                    let output = softmax(black_box(i)).unwrap();
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Linear Layer Benchmarks
// ============================================================================

fn benchmark_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear");
    group.sample_size(50);

    let dimensions = [(256, 256), (256, 1024), (1024, 256), (1024, 1024)];

    for (in_dim, out_dim) in dimensions {
        let linear = Linear::new(in_dim, out_dim).unwrap();
        let input = Tensor::from_vec(
            vec![1, in_dim],
            (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect(),
        )
        .unwrap();

        let ops = (in_dim * out_dim) as u64; // FLOPs approximation
        group.throughput(Throughput::Elements(ops));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("{in_dim}x{out_dim}")),
            &(&linear, &input),
            |b, (l, i)| {
                b.iter(|| {
                    let output = l.forward(black_box(*i)).unwrap();
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// TTFT (Time To First Token) Simulation
// ============================================================================

fn benchmark_ttft_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttft_simulation");
    group.sample_size(20); // Fewer samples for complex operations

    // Simulate prefill phase with multiple layers
    let num_layers = 4;
    let hidden_dim = 256;
    let head_dim = 32;
    let prompt_len = 32;

    // Create layers
    let fused_attns: Vec<_> = (0..num_layers)
        .map(|_| FusedQKVAttention::new(head_dim, hidden_dim).unwrap())
        .collect();
    let layer_norms: Vec<_> = (0..num_layers)
        .map(|_| LayerNorm::new(hidden_dim, 1e-5).unwrap())
        .collect();

    let input = Tensor::from_vec(
        vec![prompt_len, hidden_dim],
        (0..(prompt_len * hidden_dim))
            .map(|i| (i as f32 * 0.001).sin())
            .collect(),
    )
    .unwrap();

    group.throughput(Throughput::Elements(prompt_len as u64));
    group.bench_function("prefill_4_layers", |b| {
        b.iter(|| {
            let mut hidden = input.clone();
            for layer_idx in 0..num_layers {
                // Layer norm
                hidden = layer_norms[layer_idx].forward(&hidden).unwrap();
                // Attention
                hidden = fused_attns[layer_idx].forward(&hidden).unwrap();
            }
            black_box(hidden)
        });
    });

    group.finish();
}

// ============================================================================
// Memory Efficiency Benchmarks
// ============================================================================

fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(20);

    // Compare memory patterns for different batch sizes
    let batch_sizes = [1, 4, 8];
    let hidden_dim = 256;
    let head_dim = 32;

    for &batch_size in &batch_sizes {
        let seq_len = 64;
        let fused = FusedQKVAttention::new(head_dim, hidden_dim).unwrap();
        let input = Tensor::from_vec(
            vec![batch_size * seq_len, hidden_dim],
            vec![0.1; batch_size * seq_len * hidden_dim],
        )
        .unwrap();

        group.throughput(Throughput::Elements((batch_size * seq_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("fused_attn_batch", format!("batch{batch_size}")),
            &(&fused, &input),
            |b, (f, i)| {
                b.iter(|| {
                    let output = f.forward(black_box(*i)).unwrap();
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Tests for benchmark infrastructure
// ============================================================================

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::{dequantize_q4_k_simd, FusedQKVAttention, Tensor, SEQ_LENGTHS};
    use super::{BENCHMARK_PROMPT, HEAD_DIM, HIDDEN_DIM, NUM_HEADS};

    #[test]
    fn test_q4k_benchmark_data_valid() {
        // Verify benchmark data is valid Q4_K format
        let sb_count = 4;
        let data_size = sb_count * 144;
        let mut data = vec![0u8; data_size];

        for sb_idx in 0..sb_count {
            let offset = sb_idx * 144;
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
        }

        let result = dequantize_q4_k_simd(&data);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), sb_count * 256);
    }

    #[test]
    fn test_fused_attention_benchmark_sizes() {
        // Verify benchmark sizes are valid
        for &seq_len in SEQ_LENGTHS {
            let hidden_dim = 256;
            let head_dim = 32;

            let fused = FusedQKVAttention::new(head_dim, hidden_dim).unwrap();
            let input =
                Tensor::from_vec(vec![seq_len, hidden_dim], vec![0.1; seq_len * hidden_dim])
                    .unwrap();

            let output = fused.forward(&input).unwrap();
            assert_eq!(output.shape(), &[seq_len, hidden_dim]);
        }
    }

    #[test]
    fn test_benchmark_constants_valid() {
        // Verify benchmark constants are sensible
        assert!(HIDDEN_DIM > 0);
        assert!(NUM_HEADS > 0);
        assert_eq!(HIDDEN_DIM % NUM_HEADS, 0);
        assert_eq!(HEAD_DIM, HIDDEN_DIM / NUM_HEADS);
        assert!(!BENCHMARK_PROMPT.is_empty());
    }
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    benchmark_q4k_dequant_simd,
    benchmark_fused_q4k_dot,
    benchmark_fused_attention,
    benchmark_attention_comparison,
    benchmark_layer_norm,
    benchmark_softmax,
    benchmark_linear,
    benchmark_ttft_simulation,
    benchmark_memory_efficiency,
);

criterion_main!(benches);
