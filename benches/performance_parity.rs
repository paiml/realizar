#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::excessive_precision)]

//! Performance Parity Benchmarks (Refs PERF-PARITY-001)
//!
//! Implements the benchmark specification v1.1 for comparing realizar with:
//! - llama.cpp (GGUF inference)
//! - Ollama (production deployment)
//!
//! ## Toyota Way Principles
//!
//! - **Genchi Genbutsu**: Go and see - measure actual inference, not test ops
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
use realizar::quantize::{dequantize_q4_k_simd, fused_q4k_dot_simd, fused_q4k_parallel_matvec};
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
                    let result = dequantize_q4_k_simd(black_box(d)).expect("test");
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
                    let result = fused_q4k_dot_simd(black_box(*w), black_box(*a)).expect("test");
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

        let fused = FusedQKVAttention::new(head_dim, hidden_dim).expect("test");
        let input = Tensor::from_vec(
            vec![seq_len, hidden_dim],
            (0..(seq_len * hidden_dim))
                .map(|i| (i as f32 * 0.001).sin())
                .collect(),
        )
        .expect("test");

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("seq{seq_len}")),
            &(&fused, &input),
            |b, (f, i)| {
                b.iter(|| {
                    let output = f.forward(black_box(*i)).expect("test");
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
    let fused = FusedQKVAttention::new(head_dim, hidden_dim).expect("test");
    let input = Tensor::from_vec(
        vec![seq_len, hidden_dim],
        (0..(seq_len * hidden_dim))
            .map(|i| (i as f32 * 0.001).sin())
            .collect(),
    )
    .expect("test");

    group.bench_function("fused_qkv_attention", |b| {
        b.iter(|| {
            let output = fused.forward(black_box(&input)).expect("test");
            black_box(output)
        });
    });

    // Separate attention (baseline)
    let attention = Attention::new(head_dim).expect("test");
    let q = Tensor::from_vec(vec![seq_len, head_dim], vec![0.1; seq_len * head_dim]).expect("test");
    let k = q.clone();
    let v = q.clone();

    group.bench_function("separate_attention", |b| {
        b.iter(|| {
            let output = attention
                .forward(black_box(&q), black_box(&k), black_box(&v))
                .expect("test");
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
        let layer_norm = LayerNorm::new(hidden_dim, 1e-5).expect("test");
        let input = Tensor::from_vec(
            vec![seq_len, hidden_dim],
            (0..(seq_len * hidden_dim))
                .map(|i| (i as f32 * 0.01).sin())
                .collect(),
        )
        .expect("test");

        group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("seq{seq_len}")),
            &(&layer_norm, &input),
            |b, (ln, i)| {
                b.iter(|| {
                    let output = ln.forward(black_box(*i)).expect("test");
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
        .expect("test");

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("len{seq_len}")),
            &input,
            |b, i| {
                b.iter(|| {
                    let output = softmax(black_box(i)).expect("test");
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
        let linear = Linear::new(in_dim, out_dim).expect("test");
        let input = Tensor::from_vec(
            vec![1, in_dim],
            (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect(),
        )
        .expect("test");

        let ops = (in_dim * out_dim) as u64; // FLOPs approximation
        group.throughput(Throughput::Elements(ops));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("{in_dim}x{out_dim}")),
            &(&linear, &input),
            |b, (l, i)| {
                b.iter(|| {
                    let output = l.forward(black_box(*i)).expect("test");
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
        .map(|_| FusedQKVAttention::new(head_dim, hidden_dim).expect("test"))
        .collect();
    let layer_norms: Vec<_> = (0..num_layers)
        .map(|_| LayerNorm::new(hidden_dim, 1e-5).expect("test"))
        .collect();

    let input = Tensor::from_vec(
        vec![prompt_len, hidden_dim],
        (0..(prompt_len * hidden_dim))
            .map(|i| (i as f32 * 0.001).sin())
            .collect(),
    )
    .expect("test");

    group.throughput(Throughput::Elements(prompt_len as u64));
    group.bench_function("prefill_4_layers", |b| {
        b.iter(|| {
            let mut hidden = input.clone();
            for layer_idx in 0..num_layers {
                // Layer norm
                hidden = layer_norms[layer_idx].forward(&hidden).expect("test");
                // Attention
                hidden = fused_attns[layer_idx].forward(&hidden).expect("test");
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
        let fused = FusedQKVAttention::new(head_dim, hidden_dim).expect("test");
        let input = Tensor::from_vec(
            vec![batch_size * seq_len, hidden_dim],
            vec![0.1; batch_size * seq_len * hidden_dim],
        )
        .expect("test");

        group.throughput(Throughput::Elements((batch_size * seq_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("fused_attn_batch", format!("batch{batch_size}")),
            &(&fused, &input),
            |b, (f, i)| {
                b.iter(|| {
                    let output = f.forward(black_box(*i)).expect("test");
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
        assert_eq!(result.expect("test").len(), sb_count * 256);
    }

    #[test]
    fn test_fused_attention_benchmark_sizes() {
        // Verify benchmark sizes are valid
        for &seq_len in SEQ_LENGTHS {
            let hidden_dim = 256;
            let head_dim = 32;

            let fused = FusedQKVAttention::new(head_dim, hidden_dim).expect("test");
            let input =
                Tensor::from_vec(vec![seq_len, hidden_dim], vec![0.1; seq_len * hidden_dim])
                    .expect("test");

            let output = fused.forward(&input).expect("test");
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
// IMP-100c: Quantized vs Dequantized Throughput Benchmark
// ============================================================================

fn benchmark_quantized_vs_dequantized(c: &mut Criterion) {
    let mut group = c.benchmark_group("imp_100c_quantized_vs_dequantized");
    group.sample_size(100);

    // Test LLM-typical dimensions: (in_dim, out_dim)
    // Qwen 1.5B: hidden=1536, intermediate=4096
    // Phi-2: hidden=2560, intermediate=10240
    let dimensions = [
        (1024, 1024),  // Square baseline
        (1536, 4096),  // Qwen-like FFN up
        (4096, 1536),  // Qwen-like FFN down
        (2560, 10240), // Phi-2-like FFN up
    ];

    for (in_dim, out_dim) in dimensions {
        // Calculate Q4_K sizes: 144 bytes per 256 values (super-block)
        let weight_values = in_dim * out_dim;
        let sb_count = (weight_values + 255) / 256;
        let q4k_size = sb_count * 144;

        // Create Q4_K weight matrix (row-major: out_dim rows of in_dim values each)
        let mut weights = vec![0u8; q4k_size];
        for sb_idx in 0..sb_count {
            let offset = sb_idx * 144;
            weights[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d=1.0
            weights[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin=0.0
            for i in 4..144 {
                weights[offset + i] = ((sb_idx + i) % 16) as u8;
            }
        }

        // Create input activation vector (batch=1, m=1)
        let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

        // Benchmark 1: Fused Q4_K parallel matvec (IMP-100)
        group.throughput(Throughput::Elements(weight_values as u64));
        group.bench_with_input(
            BenchmarkId::new("fused_q4k_matvec", format!("{in_dim}x{out_dim}")),
            &(&weights, &activations, in_dim, out_dim),
            |b, (w, a, in_d, out_d)| {
                b.iter(|| {
                    let result =
                        fused_q4k_parallel_matvec(black_box(*w), black_box(*a), *in_d, *out_d)
                            .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark 2: Dequantize then f32 matvec (baseline)
        // This simulates what a naive implementation would do
        group.bench_with_input(
            BenchmarkId::new("dequant_then_matvec", format!("{in_dim}x{out_dim}")),
            &(&weights, &activations, in_dim, out_dim),
            |b, (w, a, in_d, out_d)| {
                b.iter(|| {
                    // Step 1: Dequantize Q4_K to f32
                    let dequantized = dequantize_q4_k_simd(black_box(*w)).expect("test");

                    // Step 2: Manual matmul (out_dim x in_dim) * (in_dim,) -> (out_dim,)
                    let mut output = vec![0.0f32; *out_d];
                    for row in 0..*out_d {
                        let row_start = row * *in_d;
                        let row_end = row_start + *in_d;
                        if row_end <= dequantized.len() {
                            let row_slice = &dequantized[row_start..row_end];
                            output[row] = row_slice.iter().zip(a.iter()).map(|(w, x)| w * x).sum();
                        }
                    }
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IMP-101d: KV Cache vs Full Recompute Benchmark
// ============================================================================

/// Benchmark comparing O(n) KV cache attention vs O(n²) full recompute
///
/// This validates the KV cache integration (IMP-101c) provides expected speedup
/// for incremental token generation.
fn benchmark_kv_cache_attention(c: &mut Criterion) {
    use realizar::gguf::OwnedQuantizedKVCache;

    let mut group = c.benchmark_group("imp_101d_kv_cache_attention");
    group.sample_size(50);

    // Test different sequence lengths to show O(n) vs O(n²) scaling
    let seq_lengths = [32, 64, 128, 256];
    let hidden_dim = 256;
    let num_heads = 4;
    let head_dim = hidden_dim / num_heads;
    let num_layers = 4;

    for &seq_len in &seq_lengths {
        // Setup: Pre-filled KV cache simulating seq_len-1 tokens already processed
        let mut cache = OwnedQuantizedKVCache::new(num_layers, hidden_dim, seq_len + 64);

        // Pre-fill cache with seq_len-1 positions
        for pos in 0..(seq_len - 1) {
            for layer in 0..num_layers {
                let k: Vec<f32> = (0..hidden_dim)
                    .map(|i| ((pos * hidden_dim + i) as f32 * 0.001).sin())
                    .collect();
                let v: Vec<f32> = (0..hidden_dim)
                    .map(|i| ((pos * hidden_dim + i) as f32 * 0.002).cos())
                    .collect();
                cache.append(layer, &k, &v);
            }
            cache.advance();
        }

        // Current token's Q, K, V
        let q: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.003).sin()).collect();
        let current_k: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.004).cos()).collect();
        let current_v: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.005).sin()).collect();

        // Benchmark: Attention with KV cache (O(n) per token)
        // Note: This is a simplified standalone benchmark without full model
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("kv_cache_attention", format!("seq{seq_len}")),
            &(&q, &cache, &current_k, &current_v, num_heads, head_dim),
            |b, (q, cache, cur_k, cur_v, n_heads, h_dim)| {
                b.iter(|| {
                    // Compute attention scores: Q * K^T / sqrt(d_k)
                    let scale = 1.0 / (*h_dim as f32).sqrt();
                    let k_cache = cache.get_k(0); // Use layer 0 for benchmark
                    let v_cache = cache.get_v(0);
                    let cache_len = k_cache.len() / hidden_dim;

                    let mut output = vec![0.0f32; hidden_dim];

                    // Process each head
                    for head in 0..*n_heads {
                        let head_offset = head * *h_dim;
                        let q_head = &q[head_offset..head_offset + *h_dim];

                        // Compute scores against cached + current positions
                        let mut scores = Vec::with_capacity(cache_len + 1);

                        // Cached positions
                        for pos in 0..cache_len {
                            let k_start = pos * hidden_dim + head_offset;
                            let k_head = &k_cache[k_start..k_start + *h_dim];
                            let score: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                            scores.push(score * scale);
                        }

                        // Current position
                        let cur_k_head = &cur_k[head_offset..head_offset + *h_dim];
                        let cur_score: f32 =
                            q_head.iter().zip(cur_k_head).map(|(a, b)| a * b).sum();
                        scores.push(cur_score * scale);

                        // Softmax
                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut exp_sum = 0.0f32;
                        for s in &mut scores {
                            *s = (*s - max_score).exp();
                            exp_sum += *s;
                        }
                        for s in &mut scores {
                            *s /= exp_sum;
                        }

                        // Weighted sum of values
                        let out_head = &mut output[head_offset..head_offset + *h_dim];
                        for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                            let v_start = pos * hidden_dim + head_offset;
                            let v_head = &v_cache[v_start..v_start + *h_dim];
                            for (i, &val) in v_head.iter().enumerate() {
                                out_head[i] += weight * val;
                            }
                        }
                        // Current value
                        let cur_v_head = &cur_v[head_offset..head_offset + *h_dim];
                        let cur_weight = scores[cache_len];
                        for (i, &val) in cur_v_head.iter().enumerate() {
                            out_head[i] += cur_weight * val;
                        }
                    }

                    black_box(output)
                });
            },
        );

        // Benchmark: Full recompute (O(n²) per token) - baseline for comparison
        // This simulates what happens without KV cache
        let all_k: Vec<f32> = (0..(seq_len * hidden_dim))
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let all_v: Vec<f32> = (0..(seq_len * hidden_dim))
            .map(|i| (i as f32 * 0.002).cos())
            .collect();
        let all_q: Vec<f32> = (0..(seq_len * hidden_dim))
            .map(|i| (i as f32 * 0.003).sin())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("full_recompute", format!("seq{seq_len}")),
            &(&all_q, &all_k, &all_v, num_heads, head_dim, seq_len),
            |b, (q, k, v, n_heads, h_dim, s_len)| {
                b.iter(|| {
                    let scale = 1.0 / (*h_dim as f32).sqrt();
                    let mut output = vec![0.0f32; *s_len * hidden_dim];

                    // Process each head
                    for head in 0..*n_heads {
                        let head_offset = head * *h_dim;

                        // Process each query position
                        for i in 0..*s_len {
                            let q_start = i * hidden_dim + head_offset;
                            let q_head = &q[q_start..q_start + *h_dim];

                            // Compute scores against ALL positions (causal: 0..=i)
                            let mut scores = Vec::with_capacity(i + 1);
                            for j in 0..=i {
                                let k_start = j * hidden_dim + head_offset;
                                let k_head = &k[k_start..k_start + *h_dim];
                                let score: f32 =
                                    q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                                scores.push(score * scale);
                            }

                            // Softmax
                            let max_score =
                                scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mut exp_sum = 0.0f32;
                            for s in &mut scores {
                                *s = (*s - max_score).exp();
                                exp_sum += *s;
                            }
                            for s in &mut scores {
                                *s /= exp_sum;
                            }

                            // Weighted sum of values
                            let out_start = i * hidden_dim + head_offset;
                            let out_head = &mut output[out_start..out_start + *h_dim];
                            for (j, &weight) in scores.iter().enumerate() {
                                let v_start = j * hidden_dim + head_offset;
                                let v_head = &v[v_start..v_start + *h_dim];
                                for (d, &val) in v_head.iter().enumerate() {
                                    out_head[d] += weight * val;
                                }
                            }
                        }
                    }

                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IMP-102a: End-to-End Generation Benchmark (generate vs generate_with_cache)
// ============================================================================

/// Benchmark comparing full generation with and without KV cache
///
/// This tests the complete inference path including:
/// - Token embedding lookup
/// - Transformer layer processing
/// - LM head projection
/// - Token sampling
fn benchmark_e2e_generation(c: &mut Criterion) {
    use realizar::gguf::{
        GGUFConfig, OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel,
        OwnedQuantizedTensor, QuantizedGenerateConfig,
    };

    let mut group = c.benchmark_group("imp_102a_e2e_generation");
    group.sample_size(20); // Fewer samples for expensive operations

    // Create a minimal but realistic model configuration
    let hidden_dim = 256;
    let intermediate_dim = 512;
    let num_heads = 4;
    let num_layers = 2;
    let vocab_size = 1000;
    let _head_dim = hidden_dim / num_heads;

    let config = GGUFConfig {
        architecture: "benchmark".to_string(),
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_heads,
        num_kv_heads: num_heads,
        vocab_size,
        context_length: 2048,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    // Create Q4_K quantized weights (144 bytes per 256 values)
    let create_q4k_tensor = |in_dim: usize, out_dim: usize| -> OwnedQuantizedTensor {
        let total_values = in_dim * out_dim;
        let sb_count = (total_values + 255) / 256;
        let data_size = sb_count * 144;
        let mut data = vec![0u8; data_size];

        for sb_idx in 0..sb_count {
            let offset = sb_idx * 144;
            // d = 0.1 (small scale for stable outputs)
            data[offset..offset + 2].copy_from_slice(&0x2E66_u16.to_le_bytes()); // ~0.1 in f16
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin = 0
            for i in 4..144 {
                data[offset + i] = ((sb_idx + i) % 16) as u8;
            }
        }

        OwnedQuantizedTensor {
            data,
            in_dim,
            out_dim,
            qtype: 12, // Q4_K (GGUF_TYPE_Q4_K)
        }
    };

    // Create token embeddings (vocab_size x hidden_dim)
    let token_embedding: Vec<f32> = (0..(vocab_size * hidden_dim))
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();

    // Create transformer layers
    let layers: Vec<OwnedQuantizedLayer> = (0..num_layers)
        .map(|_| OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Fused(create_q4k_tensor(hidden_dim, hidden_dim * 3)),
            qkv_bias: None,
            attn_output_weight: create_q4k_tensor(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4k_tensor(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4k_tensor(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        })
        .collect();

    let model = OwnedQuantizedModel::new_for_benchmark(
        config.clone(),
        token_embedding,
        layers,
        vec![1.0; hidden_dim],
        None,
        create_q4k_tensor(hidden_dim, vocab_size),
        None,
    );

    // Test different generation lengths
    let gen_configs = [
        (4, 4),   // Short: 4 prompt + 4 generate
        (8, 8),   // Medium: 8 prompt + 8 generate
        (16, 16), // Longer: 16 prompt + 16 generate
    ];

    for (prompt_len, gen_len) in gen_configs {
        let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % vocab_size) as u32).collect();
        let config = QuantizedGenerateConfig::deterministic(gen_len);

        // Benchmark: generate() - O(n²) per token (full recompute)
        group.throughput(Throughput::Elements(gen_len as u64));
        group.bench_with_input(
            BenchmarkId::new("generate_no_cache", format!("p{prompt_len}_g{gen_len}")),
            &(&model, &prompt, &config),
            |b, (m, p, cfg)| {
                b.iter(|| {
                    let result = m.generate(black_box(*p), black_box(*cfg)).expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark: generate_with_cache() - O(n) per token (KV cache)
        group.bench_with_input(
            BenchmarkId::new("generate_with_cache", format!("p{prompt_len}_g{gen_len}")),
            &(&model, &prompt, &config),
            |b, (m, p, cfg)| {
                b.iter(|| {
                    let result = m
                        .generate_with_cache(black_box(*p), black_box(*cfg))
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IMP-102c: Component-Level Profiling Benchmark
// ============================================================================

/// Benchmark breaking down time spent in each component of generate_with_cache
///
/// This identifies the next bottleneck after KV cache integration by measuring:
/// - Token embedding lookup
/// - Layer norm operations
/// - Fused Q4_K matvec (QKV, output, FFN up, FFN down, LM head)
/// - Attention with cache (scores, softmax, value aggregation)
/// - GELU activation
///
/// Per Toyota Way: Genchi Genbutsu - measure actual component times, not assumptions
fn benchmark_component_profiling(c: &mut Criterion) {
    use std::time::Instant;

    let mut group = c.benchmark_group("imp_102c_component_profiling");
    group.sample_size(50);

    // Model configuration (realistic but small for fast benchmarking)
    let hidden_dim = 512;
    let intermediate_dim = 1024;
    let num_heads = 8;
    let head_dim = hidden_dim / num_heads;
    let vocab_size = 2000;
    let seq_len = 64; // Simulate 64 tokens in KV cache

    // Create Q4_K weight data
    let create_q4k_data = |in_dim: usize, out_dim: usize| -> Vec<u8> {
        let total_values = in_dim * out_dim;
        let sb_count = (total_values + 255) / 256;
        let data_size = sb_count * 144;
        let mut data = vec![0u8; data_size];

        for sb_idx in 0..sb_count {
            let offset = sb_idx * 144;
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d=1.0
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin=0
            for i in 4..144 {
                data[offset + i] = ((sb_idx + i) % 16) as u8;
            }
        }
        data
    };

    // Component 1: Token embedding lookup
    let embeddings: Vec<f32> = (0..(vocab_size * hidden_dim))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    group.bench_function("1_embedding_lookup", |b| {
        let token_id = 42u32;
        b.iter(|| {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            let hidden: Vec<f32> = embeddings[start..end].to_vec();
            black_box(hidden)
        });
    });

    // Component 2: Layer norm
    let hidden_vec: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let norm_weight: Vec<f32> = vec![1.0; hidden_dim];
    let eps = 1e-5f32;

    group.bench_function("2_layer_norm", |b| {
        b.iter(|| {
            let input = black_box(&hidden_vec);
            let weight = black_box(&norm_weight);

            // Compute mean
            let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;

            // Compute variance
            let var: f32 =
                input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
            let std = (var + eps).sqrt();

            // Normalize
            let output: Vec<f32> = input
                .iter()
                .zip(weight.iter())
                .map(|(x, w)| ((x - mean) / std) * w)
                .collect();
            black_box(output)
        });
    });

    // Component 3: Fused Q4_K matvec (critical path)
    let qkv_weights = create_q4k_data(hidden_dim, hidden_dim * 3);
    let input_vec: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.001).sin()).collect();

    group.bench_function("3_fused_q4k_qkv_projection", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&qkv_weights),
                black_box(&input_vec),
                hidden_dim,
                hidden_dim * 3,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Component 4: RoPE (rotary position embedding)
    let qk_vec: Vec<f32> = (0..(hidden_dim * 2))
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let position = 32usize;

    group.bench_function("4_rope_apply", |b| {
        b.iter(|| {
            let q = &qk_vec[..hidden_dim];
            let k = &qk_vec[hidden_dim..];
            let pos = black_box(position);

            let mut q_rot = q.to_vec();
            let mut k_rot = k.to_vec();

            // Apply RoPE per head
            for head in 0..num_heads {
                let head_offset = head * head_dim;
                for i in 0..(head_dim / 2) {
                    let theta = 10000.0f32.powf(-2.0 * (i as f32) / (head_dim as f32));
                    let angle = (pos as f32) * theta;
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();

                    let j = head_offset + i;
                    let j2 = head_offset + head_dim / 2 + i;

                    // Rotate Q
                    let q0 = q_rot[j];
                    let q1 = q_rot[j2];
                    q_rot[j] = q0 * cos_val - q1 * sin_val;
                    q_rot[j2] = q0 * sin_val + q1 * cos_val;

                    // Rotate K
                    let k0 = k_rot[j];
                    let k1 = k_rot[j2];
                    k_rot[j] = k0 * cos_val - k1 * sin_val;
                    k_rot[j2] = k0 * sin_val + k1 * cos_val;
                }
            }
            black_box((q_rot, k_rot))
        });
    });

    // Component 5: Attention with cache (main bottleneck candidate)
    let q: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k_cache: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let v_cache: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| (i as f32 * 0.002).sin())
        .collect();
    let cur_k: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.003).cos()).collect();
    let cur_v: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.004).sin()).collect();

    group.bench_function("5_attention_with_cache", |b| {
        b.iter(|| {
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut output = vec![0.0f32; hidden_dim];

            for head in 0..num_heads {
                let head_offset = head * head_dim;
                let q_head = &q[head_offset..head_offset + head_dim];

                // Compute scores against cached positions
                let mut scores = Vec::with_capacity(seq_len + 1);
                for pos in 0..seq_len {
                    let k_start = pos * hidden_dim + head_offset;
                    let k_head = &k_cache[k_start..k_start + head_dim];
                    let score: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                    scores.push(score * scale);
                }
                // Current position
                let cur_k_head = &cur_k[head_offset..head_offset + head_dim];
                let cur_score: f32 = q_head.iter().zip(cur_k_head).map(|(a, b)| a * b).sum();
                scores.push(cur_score * scale);

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }
                for s in &mut scores {
                    *s /= exp_sum;
                }

                // Weighted sum of values
                let out_head = &mut output[head_offset..head_offset + head_dim];
                for (pos, &weight) in scores.iter().enumerate().take(seq_len) {
                    let v_start = pos * hidden_dim + head_offset;
                    let v_head = &v_cache[v_start..v_start + head_dim];
                    for (i, &val) in v_head.iter().enumerate() {
                        out_head[i] += weight * val;
                    }
                }
                // Current value
                let cur_v_head = &cur_v[head_offset..head_offset + head_dim];
                let cur_weight = scores[seq_len];
                for (i, &val) in cur_v_head.iter().enumerate() {
                    out_head[i] += cur_weight * val;
                }
            }
            black_box(output)
        });
    });

    // Component 6: Output projection (fused Q4_K matvec)
    let output_weights = create_q4k_data(hidden_dim, hidden_dim);

    group.bench_function("6_fused_q4k_output_proj", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&output_weights),
                black_box(&input_vec),
                hidden_dim,
                hidden_dim,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Component 7: FFN up projection (fused Q4_K matvec - largest)
    let ffn_up_weights = create_q4k_data(hidden_dim, intermediate_dim);

    group.bench_function("7_fused_q4k_ffn_up", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&ffn_up_weights),
                black_box(&input_vec),
                hidden_dim,
                intermediate_dim,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Component 8: GELU activation
    let ffn_hidden: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    group.bench_function("8_gelu_activation", |b| {
        b.iter(|| {
            let input = black_box(&ffn_hidden);
            let output: Vec<f32> = input
                .iter()
                .map(|&x| {
                    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let sqrt_2_pi = 0.7978845608f32;
                    let coeff = 0.044715f32;
                    x * 0.5 * (1.0 + (sqrt_2_pi * (x + coeff * x * x * x)).tanh())
                })
                .collect();
            black_box(output)
        });
    });

    // Component 9: FFN down projection
    let ffn_down_weights = create_q4k_data(intermediate_dim, hidden_dim);
    let ffn_intermediate: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    group.bench_function("9_fused_q4k_ffn_down", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&ffn_down_weights),
                black_box(&ffn_intermediate),
                intermediate_dim,
                hidden_dim,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Component 10: LM head projection (vocab projection - potentially large)
    let lm_head_weights = create_q4k_data(hidden_dim, vocab_size);

    group.bench_function("10_fused_q4k_lm_head", |b| {
        b.iter(|| {
            let result = fused_q4k_parallel_matvec(
                black_box(&lm_head_weights),
                black_box(&input_vec),
                hidden_dim,
                vocab_size,
            )
            .expect("test");
            black_box(result)
        });
    });

    // Summary: Full single-token forward pass (all components)
    group.bench_function("TOTAL_single_token_forward", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Embedding lookup
                let token_id = 42usize;
                let emb_start = token_id * hidden_dim;
                let _hidden: Vec<f32> = embeddings[emb_start..emb_start + hidden_dim].to_vec();

                // Layer norm
                let mean: f32 = input_vec.iter().sum::<f32>() / hidden_dim as f32;
                let var: f32 =
                    input_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
                let _normed: Vec<f32> = input_vec
                    .iter()
                    .zip(norm_weight.iter())
                    .map(|(x, w)| ((x - mean) / (var + eps).sqrt()) * w)
                    .collect();

                // QKV projection
                let _qkv =
                    fused_q4k_parallel_matvec(&qkv_weights, &input_vec, hidden_dim, hidden_dim * 3)
                        .expect("test");

                // Attention (simplified)
                let scale = 1.0 / (head_dim as f32).sqrt();
                let mut output = vec![0.0f32; hidden_dim];
                for head in 0..num_heads {
                    let head_offset = head * head_dim;
                    let q_head = &q[head_offset..head_offset + head_dim];
                    let mut scores = Vec::with_capacity(seq_len + 1);
                    for pos in 0..seq_len {
                        let k_start = pos * hidden_dim + head_offset;
                        let k_head = &k_cache[k_start..k_start + head_dim];
                        scores.push(
                            q_head.iter().zip(k_head).map(|(a, b)| a * b).sum::<f32>() * scale,
                        );
                    }
                    scores.push(
                        q_head
                            .iter()
                            .zip(&cur_k[head_offset..head_offset + head_dim])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale,
                    );
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        exp_sum += *s;
                    }
                    for s in &mut scores {
                        *s /= exp_sum;
                    }
                    let out_head = &mut output[head_offset..head_offset + head_dim];
                    for (pos, &w) in scores.iter().enumerate().take(seq_len) {
                        let v_start = pos * hidden_dim + head_offset;
                        for (i, &v) in v_cache[v_start..v_start + head_dim].iter().enumerate() {
                            out_head[i] += w * v;
                        }
                    }
                }

                // Output projection
                let _out =
                    fused_q4k_parallel_matvec(&output_weights, &input_vec, hidden_dim, hidden_dim)
                        .expect("test");

                // FFN up
                let _ffn_up = fused_q4k_parallel_matvec(
                    &ffn_up_weights,
                    &input_vec,
                    hidden_dim,
                    intermediate_dim,
                )
                .expect("test");

                // FFN down
                let _ffn_down = fused_q4k_parallel_matvec(
                    &ffn_down_weights,
                    &ffn_intermediate,
                    intermediate_dim,
                    hidden_dim,
                )
                .expect("test");

                // LM head
                let _logits =
                    fused_q4k_parallel_matvec(&lm_head_weights, &input_vec, hidden_dim, vocab_size)
                        .expect("test");

                black_box(());
            }
            start.elapsed()
        });
    });

    group.finish();
}

// ============================================================================
// IMP-103a: SIMD-Optimized Q4_K Matvec Benchmark
// ============================================================================

/// Benchmark for fused Q4_K matvec optimization (IMP-103)
///
/// Measures throughput of fused_q4k_parallel_matvec at different dimensions
/// to identify optimization opportunities.
///
/// Target: 2x speedup via improved SIMD utilization
fn benchmark_q4k_matvec_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("imp_103_q4k_matvec_optimization");
    group.sample_size(50);

    // Dimensions from IMP-102c profiling (most time-consuming):
    // LM head: 512 -> 2000 (21.4% of time)
    // QKV: 512 -> 1536 (19.5% of time)
    // FFN up: 512 -> 1024 (15.4% of time)
    let dimensions = [
        (512, 512, "output_proj"),
        (512, 1024, "ffn_up"),
        (1024, 512, "ffn_down"),
        (512, 1536, "qkv_proj"),
        (512, 2000, "lm_head"),
        (1024, 4096, "large_ffn"),
    ];

    for (in_dim, out_dim, name) in dimensions {
        // Create Q4_K weight data
        let total_values = in_dim * out_dim;
        let sb_count = (total_values + 255) / 256;
        let data_size = sb_count * 144;
        let mut weights = vec![0u8; data_size];

        for sb_idx in 0..sb_count {
            let offset = sb_idx * 144;
            weights[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d=1.0
            weights[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin=0
            for i in 4..144 {
                weights[offset + i] = ((sb_idx + i) % 16) as u8;
            }
        }

        // Create activations
        let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

        // Measure throughput in GFLOPS (2 * in_dim * out_dim for matvec)
        let flops = 2 * in_dim * out_dim;
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(
            BenchmarkId::new("fused_q4k_matvec", format!("{name}_{in_dim}x{out_dim}")),
            &(&weights, &activations, in_dim, out_dim),
            |b, (w, a, in_d, out_d)| {
                b.iter(|| {
                    let result =
                        fused_q4k_parallel_matvec(black_box(*w), black_box(*a), *in_d, *out_d)
                            .expect("test");
                    black_box(result)
                });
            },
        );
    }

    // Also benchmark the inner dot product to isolate single-row performance
    let in_dim = 512;
    let sb_count = in_dim / 256;
    let data_size = sb_count * 144;
    let mut row_weights = vec![0u8; data_size];
    for sb_idx in 0..sb_count {
        let offset = sb_idx * 144;
        row_weights[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        row_weights[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
        for i in 4..144 {
            row_weights[offset + i] = ((sb_idx + i) % 16) as u8;
        }
    }
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

    group.bench_function("single_row_dot_512", |b| {
        b.iter(|| {
            let result =
                fused_q4k_dot_simd(black_box(&row_weights), black_box(&activations)).expect("test");
            black_box(result)
        });
    });

    // Larger single row for better measurement
    let in_dim = 4096;
    let sb_count = in_dim / 256;
    let data_size = sb_count * 144;
    let mut row_weights = vec![0u8; data_size];
    for sb_idx in 0..sb_count {
        let offset = sb_idx * 144;
        row_weights[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        row_weights[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
        for i in 4..144 {
            row_weights[offset + i] = ((sb_idx + i) % 16) as u8;
        }
    }
    let activations_large: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

    group.bench_function("single_row_dot_4096", |b| {
        b.iter(|| {
            let result = fused_q4k_dot_simd(black_box(&row_weights), black_box(&activations_large))
                .expect("test");
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// IMP-106: Batch Prefill Benchmarks
// ============================================================================

/// Benchmark batch prefill vs sequential prefill (IMP-106c)
///
/// Compares:
/// - Sequential: for each token, forward_single_with_cache
/// - Batch: prefill_batch (processes all tokens)
fn benchmark_batch_prefill(c: &mut Criterion) {
    use realizar::gguf::{GGUFConfig, OwnedQuantizedKVCache};

    let mut group = c.benchmark_group("imp_106_batch_prefill");
    group.sample_size(50);

    // Test different prompt lengths
    let prompt_lengths = [4, 8, 16, 32];

    for &prompt_len in &prompt_lengths {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim: 256,
            intermediate_dim: 512,
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 1000,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        // Create test model
        let model = create_benchmark_model(&config);
        let prompt: Vec<u32> = (0..prompt_len as u32).collect();

        // Benchmark sequential prefill
        group.bench_function(BenchmarkId::new("sequential", prompt_len), |b| {
            b.iter(|| {
                let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);
                for (pos, &token_id) in prompt.iter().enumerate() {
                    let _ = model.forward_single_with_cache(black_box(token_id), &mut cache, pos);
                }
                black_box(cache.len())
            });
        });

        // Benchmark batch prefill
        group.bench_function(BenchmarkId::new("batch", prompt_len), |b| {
            b.iter(|| {
                let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);
                let _ = model.prefill_batch(black_box(&prompt), &mut cache);
                black_box(cache.len())
            });
        });
    }

    group.finish();
}

/// Create a benchmark model with proper Q4_K weights
fn create_benchmark_model(
    config: &realizar::gguf::GGUFConfig,
) -> realizar::gguf::OwnedQuantizedModel {
    use realizar::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let vocab_size = config.vocab_size;
    let head_dim = hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let mut layers = Vec::new();
    for _ in 0..config.num_layers {
        let layer = OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Fused(create_bench_q4k_data(hidden_dim, qkv_out_dim)),
            qkv_bias: None,
            attn_output_weight: create_bench_q4k_data(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_bench_q4k_data(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_bench_q4k_data(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        };
        layers.push(layer);
    }

    OwnedQuantizedModel::new_for_benchmark(
        config.clone(),
        vec![0.1f32; vocab_size * hidden_dim],
        layers,
        vec![1.0f32; hidden_dim],
        None,
        create_bench_q4k_data(hidden_dim, vocab_size),
        None,
    )
}

/// Create Q4_K benchmark data
fn create_bench_q4k_data(in_dim: usize, out_dim: usize) -> realizar::gguf::OwnedQuantizedTensor {
    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;
    let data_size = out_dim * bytes_per_row;
    let mut data = vec![0u8; data_size];

    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let offset = row * bytes_per_row + sb * 144;
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d=1.0
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin=0
            for i in 4..144 {
                data[offset + i] = ((row + sb + i) % 16) as u8;
            }
        }
    }

    realizar::gguf::OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: 12, // Q4_K
    }
}

// ============================================================================
// IMP-107: GPU Batch Matmul Benchmarks
// ============================================================================

/// Benchmark GPU vs CPU matmul crossover point (IMP-107c)
///
/// Measures when GPU batch matmul becomes faster than CPU.
/// Per spec: GPU should win for batch_size > 1 and large matrices.
#[cfg(feature = "gpu")]
fn benchmark_gpu_batch_matmul(c: &mut Criterion) {
    use realizar::gpu::HybridScheduler;

    let mut group = c.benchmark_group("gpu_batch_matmul");
    group.sample_size(50);

    // Test various batch sizes and matrix dimensions
    // Format: (batch_size, k_dim, n_dim)
    let test_cases = [
        (1, 256, 256),   // Single token - CPU should win
        (1, 512, 512),   // Single token, larger - CPU should still win
        (4, 256, 256),   // Small batch - crossover point
        (8, 256, 512),   // Medium batch - GPU should start winning
        (16, 512, 512),  // Large batch - GPU should dominate
        (32, 512, 1024), // Very large batch - GPU territory
    ];

    for (batch_size, k, n) in test_cases {
        let m = batch_size;
        let ops = m * k * n;

        // Create test matrices
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        group.throughput(Throughput::Elements(ops as u64));

        // Benchmark CPU path
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{m}x{k}x{n}")),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let result = cpu_matmul_bench(black_box(a), black_box(b), m, k, n);
                    black_box(result)
                });
            },
        );

        // Benchmark HybridScheduler (GPU when appropriate)
        group.bench_with_input(
            BenchmarkId::new("hybrid", format!("{m}x{k}x{n}")),
            &(&a, &b),
            |bencher, (a, b)| {
                let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");
                bencher.iter(|| {
                    let result = scheduler
                        .matmul(black_box(a), black_box(b), m, k, n)
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// CPU reference matmul for benchmark comparison
#[cfg(feature = "gpu")]
fn cpu_matmul_bench(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ============================================================================
// IMP-108: Batched Causal Attention Benchmarks
// ============================================================================

/// Benchmark batched causal attention vs sequential (IMP-108c)
///
/// Measures the speedup of batched causal attention with GPU acceleration
/// compared to sequential per-position attention.
#[cfg(feature = "gpu")]
fn benchmark_batched_causal_attention(c: &mut Criterion) {
    use realizar::gguf::GGUFConfig;

    let mut group = c.benchmark_group("batched_causal_attention");
    group.sample_size(30);

    // Test various sequence lengths
    let seq_lengths = [4, 8, 16, 32, 64];

    for seq_len in seq_lengths {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim: 256, // Realistic hidden dim
            intermediate_dim: 512,
            num_layers: 1,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 1000,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let model = create_bench_model_with_config(&config);

        // Create test Q, K, V
        let hidden_dim = config.hidden_dim;
        let num_heads = config.num_heads;
        let q: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();

        group.throughput(Throughput::Elements(
            (seq_len * seq_len * hidden_dim) as u64,
        ));

        // Benchmark CPU sequential causal attention (reference)
        group.bench_with_input(
            BenchmarkId::new("cpu_sequential", seq_len),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = causal_attention_cpu_ref(
                        black_box(q),
                        black_box(k),
                        black_box(v),
                        seq_len,
                        hidden_dim,
                        num_heads,
                    );
                    black_box(result)
                });
            },
        );

        // Benchmark batched GPU causal attention
        group.bench_with_input(
            BenchmarkId::new("batched_gpu", seq_len),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .batched_causal_attention_gpu(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// CPU reference implementation of causal attention for benchmarking
#[cfg(feature = "gpu")]
fn causal_attention_cpu_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
) -> Vec<f32> {
    let head_dim = hidden_dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * hidden_dim];

    for head in 0..num_heads {
        let head_offset = head * head_dim;

        for i in 0..seq_len {
            let mut scores = Vec::with_capacity(i + 1);
            let q_start = i * hidden_dim + head_offset;

            for j in 0..=i {
                let k_start = j * hidden_dim + head_offset;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_start + d] * k[k_start + d];
                }
                scores.push(score * scale);
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }

            // Weighted sum
            let out_start = i * hidden_dim + head_offset;
            for (j, &weight) in scores.iter().enumerate() {
                let v_start = j * hidden_dim + head_offset;
                for d in 0..head_dim {
                    output[out_start + d] += weight * v[v_start + d];
                }
            }
        }
    }

    output
}

/// Benchmark fused batch matmul vs separate operations (IMP-109)
///
/// Measures the performance benefit of fusing dequantization with matmul
/// for FFN projections in transformer layers.
#[cfg(feature = "gpu")]
fn benchmark_fused_batch_matmul(c: &mut Criterion) {
    use realizar::gguf::GGUFConfig;
    use realizar::gpu::HybridScheduler;
    use realizar::quantize::{dequantize_q4_k_simd, QK_K};

    let mut group = c.benchmark_group("fused_batch_matmul_imp109");
    group.sample_size(30);

    // Test various batch sizes and dimensions
    let configs = [
        (4, 256, 512, "small_4x256x512"),
        (8, 256, 512, "small_8x256x512"),
        (4, 512, 1024, "medium_4x512x1024"),
        (8, 512, 1024, "medium_8x512x1024"),
        (16, 256, 512, "batch_16x256x512"),
        (32, 256, 512, "batch_32x256x512"),
    ];

    for (batch_size, hidden_dim, intermediate_dim, label) in configs {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim,
            intermediate_dim,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let model = create_bench_model_with_config(&config);

        // Create input activations
        let activations: Vec<f32> = (0..batch_size * hidden_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();

        let weight = &model.layers[0].ffn_up_weight;

        // Pre-dequantize weight for "separate" benchmark
        let weight_f32 = {
            let in_dim = weight.in_dim;
            let out_dim = weight.out_dim;
            let super_blocks_per_row = in_dim.div_ceil(QK_K);
            let mut output = Vec::with_capacity(in_dim * out_dim);
            for row in 0..out_dim {
                let row_start = row * super_blocks_per_row * 144;
                let row_end = row_start + super_blocks_per_row * 144;
                let row_data = &weight.data[row_start..row_end];
                let row_dequant = dequantize_q4_k_simd(row_data).expect("test");
                output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
            }
            output
        };

        group.throughput(Throughput::Elements(
            (batch_size * hidden_dim * intermediate_dim) as u64,
        ));

        // Benchmark fused batch matmul (dequant + matmul combined)
        group.bench_with_input(
            BenchmarkId::new("fused", label),
            &(&activations, weight),
            |bencher, (act, w)| {
                bencher.iter(|| {
                    let result = model
                        .fused_batch_matmul_gpu(black_box(act), black_box(w), batch_size)
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark separate: pre-dequantized weight + matmul
        let weight_clone = weight_f32.clone();
        group.bench_with_input(
            BenchmarkId::new("separate_predequant", label),
            &(&activations, &weight_clone),
            |bencher, (act, w)| {
                bencher.iter(|| {
                    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");
                    let result = scheduler
                        .matmul(
                            black_box(act),
                            black_box(w),
                            batch_size,
                            hidden_dim,
                            intermediate_dim,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark separate: dequant each time + matmul (worst case)
        group.bench_with_input(
            BenchmarkId::new("separate_redequant", label),
            &(&activations, weight),
            |bencher, (act, w)| {
                bencher.iter(|| {
                    // Dequantize weight each time (simulates no caching)
                    let in_dim = w.in_dim;
                    let out_dim = w.out_dim;
                    let super_blocks_per_row = in_dim.div_ceil(QK_K);
                    let mut w_f32 = Vec::with_capacity(in_dim * out_dim);
                    for row in 0..out_dim {
                        let row_start = row * super_blocks_per_row * 144;
                        let row_end = row_start + super_blocks_per_row * 144;
                        let row_data = &w.data[row_start..row_end];
                        let row_dequant = dequantize_q4_k_simd(row_data).expect("test");
                        w_f32.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
                    }

                    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");
                    let result = scheduler
                        .matmul(black_box(act), &w_f32, batch_size, in_dim, out_dim)
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel multi-head attention vs sequential (IMP-110)
///
/// Measures the performance of processing all attention heads in parallel
/// vs the sequential head-by-head approach.
#[cfg(feature = "gpu")]
fn benchmark_parallel_multihead_attention(c: &mut Criterion) {
    use realizar::gguf::GGUFConfig;

    let mut group = c.benchmark_group("parallel_multihead_attention_imp110");
    group.sample_size(30);

    // Test various sequence lengths and head counts
    let configs = [
        (4, 64, 4, "seq4_h4"),
        (8, 64, 4, "seq8_h4"),
        (16, 64, 4, "seq16_h4"),
        (4, 128, 8, "seq4_h8"),
        (8, 128, 8, "seq8_h8"),
        (16, 128, 8, "seq16_h8"),
        (32, 256, 8, "seq32_h8"),
    ];

    for (seq_len, hidden_dim, num_heads, label) in configs {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim,
            intermediate_dim: hidden_dim * 2,
            num_layers: 1,
            num_heads,
            num_kv_heads: num_heads,
            vocab_size: 100,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let model = create_bench_model_with_config(&config);

        // Create Q, K, V tensors [seq_len, hidden_dim]
        let q: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();

        let head_dim = hidden_dim / num_heads;
        // Ops: num_heads * (QK^T: seq*seq*head_dim + softmax: seq*seq + AV: seq*seq*head_dim)
        let ops = num_heads * seq_len * seq_len * head_dim * 2;
        group.throughput(Throughput::Elements(ops as u64));

        // Benchmark sequential multi-head attention (existing implementation)
        group.bench_with_input(
            BenchmarkId::new("sequential", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .batched_causal_attention_gpu(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark parallel multi-head attention (IMP-110)
        group.bench_with_input(
            BenchmarkId::new("parallel", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .parallel_multihead_attention_gpu(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark tiled attention vs standard attention (IMP-111)
///
/// Measures the performance and memory efficiency of Flash Attention-style
/// tiled computation vs full materialization of attention matrix.
///
/// Key insight: Tiled attention uses O(tile_size) memory per row vs O(seq_len).
/// For long sequences, this enables processing without OOM.
#[cfg(feature = "gpu")]
fn benchmark_tiled_attention(c: &mut Criterion) {
    use realizar::gguf::GGUFConfig;

    let mut group = c.benchmark_group("tiled_attention_imp111");
    group.sample_size(30);

    // Test various sequence lengths to show memory vs speed tradeoff
    let configs = [
        (32, 64, 4, 8, "seq32_tile8"),
        (64, 64, 4, 8, "seq64_tile8"),
        (64, 64, 4, 16, "seq64_tile16"),
        (128, 64, 4, 16, "seq128_tile16"),
        (128, 64, 4, 32, "seq128_tile32"),
        (256, 64, 4, 32, "seq256_tile32"),
    ];

    for (seq_len, hidden_dim, num_heads, tile_size, label) in configs {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim,
            intermediate_dim: hidden_dim * 2,
            num_layers: 1,
            num_heads,
            num_kv_heads: num_heads,
            vocab_size: 100,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let model = create_bench_model_with_config(&config);
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create single-head Q, K, V for fair comparison
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();

        // Ops: QK^T: seq*seq*head_dim + softmax: seq*seq + AV: seq*seq*head_dim
        let ops = seq_len * seq_len * head_dim * 2;
        group.throughput(Throughput::Elements(ops as u64));

        // Benchmark standard attention (full materialization)
        group.bench_with_input(
            BenchmarkId::new("standard", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .standard_single_head_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark tiled attention (O(tile_size) memory)
        group.bench_with_input(
            BenchmarkId::new("tiled", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .tiled_single_head_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                            tile_size,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark tiled causal attention (autoregressive inference)
        group.bench_with_input(
            BenchmarkId::new("tiled_causal", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = model
                        .tiled_causal_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                            tile_size,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scheduler caching vs uncached (IMP-112)
///
/// Measures the performance benefit of caching the HybridScheduler
/// across multiple forward passes instead of recreating it each time.
#[cfg(feature = "gpu")]
fn benchmark_scheduler_caching(c: &mut Criterion) {
    use realizar::gguf::{GGUFConfig, OwnedQuantizedModelCached};

    let mut group = c.benchmark_group("scheduler_caching_imp112");
    group.sample_size(20); // Fewer samples since operations are slow

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let model = create_bench_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model.clone());

    let tokens = vec![1u32, 5, 10, 20];

    group.throughput(Throughput::Elements(
        (tokens.len() * config.vocab_size) as u64,
    ));

    // Warm up cached model (initialize scheduler)
    let _ = cached_model
        .forward_batch_gpu_cached(&tokens)
        .expect("test");

    // Benchmark uncached (creates new scheduler each call - ~300ms overhead)
    group.bench_function("uncached_forward", |bencher| {
        bencher.iter(|| {
            let result = model.forward_batch_gpu(black_box(&tokens)).expect("test");
            black_box(result)
        });
    });

    // Benchmark cached (reuses scheduler - ~0ms overhead)
    group.bench_function("cached_forward", |bencher| {
        bencher.iter(|| {
            let result = cached_model
                .forward_batch_gpu_cached(black_box(&tokens))
                .expect("test");
            black_box(result)
        });
    });

    // Multiple calls to show amortization benefit
    group.bench_function("cached_5x_forward", |bencher| {
        bencher.iter(|| {
            for _ in 0..5 {
                let result = cached_model
                    .forward_batch_gpu_cached(black_box(&tokens))
                    .expect("test");
                black_box(&result);
            }
        });
    });

    group.finish();
}

/// Benchmark single-dispatch vs multi-dispatch attention (IMP-113)
///
/// Compares the performance of processing all attention heads with:
/// - Multi-dispatch: Loop over heads, one matmul dispatch per head
/// - Single-dispatch: Batched operations with cached scheduler
#[cfg(feature = "gpu")]
fn benchmark_single_dispatch_attention(c: &mut Criterion) {
    use realizar::gguf::{GGUFConfig, OwnedQuantizedModelCached};

    let mut group = c.benchmark_group("single_dispatch_attention_imp113");
    group.sample_size(30);

    // Test various configurations
    let configs = [
        (8, 64, 4, "seq8_h4"),
        (16, 64, 4, "seq16_h4"),
        (16, 128, 8, "seq16_h8"),
        (32, 128, 8, "seq32_h8"),
        (32, 256, 8, "seq32_h8_hd256"),
    ];

    for (seq_len, hidden_dim, num_heads, label) in configs {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim,
            intermediate_dim: hidden_dim * 2,
            num_layers: 1,
            num_heads,
            num_kv_heads: num_heads,
            vocab_size: 100,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let model = create_bench_model_with_config(&config);
        let cached_model = OwnedQuantizedModelCached::new(model.clone());

        // Create Q, K, V tensors [seq_len, hidden_dim]
        let q: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();

        let head_dim = hidden_dim / num_heads;
        let ops = num_heads * seq_len * seq_len * head_dim * 2;
        group.throughput(Throughput::Elements(ops as u64));

        // Warm up cached model
        let _ = cached_model.parallel_multihead_attention_gpu_cached(&q, &k, &v, seq_len);

        // Benchmark multi-dispatch (existing implementation)
        group.bench_with_input(
            BenchmarkId::new("multi_dispatch", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .parallel_multihead_attention_gpu_cached(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark single-dispatch (IMP-113)
        group.bench_with_input(
            BenchmarkId::new("single_dispatch", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .single_dispatch_multihead_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark flattened vs loop-based batched GEMM (IMP-114)
///
/// Compares the flattened batched GEMM approach against the loop-based
/// approach from IMP-113.
#[cfg(feature = "gpu")]
fn benchmark_flattened_batched_gemm(c: &mut Criterion) {
    use realizar::gguf::{GGUFConfig, OwnedQuantizedModelCached};

    let mut group = c.benchmark_group("flattened_batched_gemm_imp114");
    group.sample_size(30);

    // Test various batch sizes and matrix dimensions
    let configs = [
        (4, 8, 16, 8, "b4_m8_k16_n8"),
        (8, 16, 8, 16, "b8_m16_k8_n16"),
        (8, 32, 16, 32, "b8_m32_k16_n32"),
        (16, 16, 8, 16, "b16_m16_k8_n16"),
        (16, 8, 8, 8, "b16_m8_k8_n8"),
    ];

    for (batch_size, m, k, n, label) in configs {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let model = create_bench_model_with_config(&config);
        let cached_model = OwnedQuantizedModelCached::new(model);

        // Create batched matrices
        let batched_a: Vec<f32> = (0..batch_size * m * k)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let batched_b: Vec<f32> = (0..batch_size * k * n)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();

        let ops = batch_size * m * k * n * 2;
        group.throughput(Throughput::Elements(ops as u64));

        // Warm up
        let _ =
            cached_model.batched_gemm_single_dispatch(&batched_a, &batched_b, batch_size, m, k, n);

        // Benchmark loop-based (IMP-113)
        group.bench_with_input(
            BenchmarkId::new("loop_based", label),
            &(&batched_a, &batched_b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let result = cached_model
                        .batched_gemm_single_dispatch(
                            black_box(a),
                            black_box(b),
                            batch_size,
                            m,
                            k,
                            n,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark flattened (IMP-114)
        group.bench_with_input(
            BenchmarkId::new("flattened", label),
            &(&batched_a, &batched_b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let result = cached_model
                        .flattened_batched_gemm(black_box(a), black_box(b), batch_size, m, k, n)
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// IMP-115: Benchmark fused kernel attention vs separate operations
#[cfg(feature = "gpu")]
fn benchmark_fused_kernel_attention(c: &mut Criterion) {
    use realizar::gguf::{GGUFConfig, OwnedQuantizedModelCached};

    let mut group = c.benchmark_group("fused_kernel_attention_imp115");
    group.sample_size(30);

    // Test various configurations
    let configs = [
        (4, 8, 16, "h4_seq8_d16"),   // 4 heads, seq 8, dim 16
        (8, 8, 16, "h8_seq8_d16"),   // 8 heads, seq 8, dim 16
        (8, 16, 16, "h8_seq16_d16"), // 8 heads, seq 16, dim 16
        (8, 32, 16, "h8_seq32_d16"), // 8 heads, seq 32, dim 16
    ];

    for (num_heads, seq_len, head_dim, label) in configs {
        let hidden_dim = num_heads * head_dim;
        let config = GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim,
            intermediate_dim: hidden_dim * 4,
            num_layers: 1,
            num_heads,
            num_kv_heads: num_heads,
            vocab_size: 100,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        let model = create_bench_model_with_config(&config);
        let cached_model = OwnedQuantizedModelCached::new(model);

        // Create Q, K, V tensors
        let q: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();

        // Approximate ops: num_heads * (seq * seq * head_dim * 2 + seq * seq + seq * head_dim * seq)
        let ops = num_heads * seq_len * seq_len * head_dim * 4;
        group.throughput(Throughput::Elements(ops as u64));

        // Warm up
        let _ = cached_model.flattened_multihead_attention(&q, &k, &v, seq_len);

        // Benchmark flattened (separate matmul + softmax + matmul)
        group.bench_with_input(
            BenchmarkId::new("separate_ops", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .flattened_multihead_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark fused kernel (IMP-115)
        group.bench_with_input(
            BenchmarkId::new("fused_kernel", label),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .fused_multihead_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IMP-120: GPU vs CPU Fused Attention Crossover Benchmark
// ============================================================================

/// Benchmark GPU vs CPU fused attention to determine optimal crossover point
///
/// This benchmark measures:
/// - CPU fused attention (IMP-115) latency across sequence lengths
/// - GPU fused attention (IMP-119) latency across sequence lengths
/// - Identifies the sequence length where GPU becomes faster
#[cfg(feature = "gpu")]
fn benchmark_gpu_cpu_crossover(c: &mut Criterion) {
    use realizar::gguf::{GGUFConfig, OwnedQuantizedModelCached};

    let mut group = c.benchmark_group("gpu_cpu_crossover_imp120");
    group.sample_size(20); // Fewer samples for longer benchmarks

    // Test sequence lengths around expected crossover (~64 tokens)
    let seq_lengths = [8, 16, 32, 64, 128, 256];
    let head_dim = 64;
    let num_heads = 8;
    let hidden_dim = num_heads * head_dim;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim,
        intermediate_dim: hidden_dim * 4,
        num_layers: 1,
        num_heads,
        num_kv_heads: num_heads,
        vocab_size: 100,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
    };

    let model = create_bench_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    for seq_len in seq_lengths {
        // Create single-head Q, K, V tensors for fair comparison
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Ops: seq * seq * head_dim * 2 (Q@K^T) + seq * seq (softmax) + seq * seq * head_dim (attn@V)
        let ops = seq_len * seq_len * head_dim * 3 + seq_len * seq_len;
        group.throughput(Throughput::Elements(ops as u64));

        // Warm up both paths
        let _ = cached_model.fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale);
        let _ = cached_model.gpu_fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale);

        // Benchmark CPU fused attention
        group.bench_with_input(
            BenchmarkId::new("cpu_fused", format!("seq{seq_len}")),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .fused_causal_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark GPU fused attention
        group.bench_with_input(
            BenchmarkId::new("gpu_fused", format!("seq{seq_len}")),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .gpu_fused_causal_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );

        // Benchmark adaptive (auto-selects based on seq_len)
        group.bench_with_input(
            BenchmarkId::new("adaptive", format!("seq{seq_len}")),
            &(&q, &k, &v),
            |bencher, (q, k, v)| {
                bencher.iter(|| {
                    let result = cached_model
                        .adaptive_fused_attention(
                            black_box(q),
                            black_box(k),
                            black_box(v),
                            seq_len,
                            head_dim,
                            scale,
                        )
                        .expect("test");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Create benchmark model with config
#[cfg(feature = "gpu")]
fn create_bench_model_with_config(
    config: &realizar::gguf::GGUFConfig,
) -> realizar::gguf::OwnedQuantizedModel {
    use realizar::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let vocab_size = config.vocab_size;

    // Create minimal test weights
    let layers = (0..config.num_layers)
        .map(|_| OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Fused(create_bench_q4k_data(hidden_dim, hidden_dim * 3)),
            qkv_bias: None,
            attn_output_weight: create_bench_q4k_data(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_bench_q4k_data(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_bench_q4k_data(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
        })
        .collect();

    OwnedQuantizedModel::new_for_benchmark(
        config.clone(),
        vec![0.1f32; vocab_size * hidden_dim],
        layers,
        vec![1.0f32; hidden_dim],
        None,
        create_bench_q4k_data(hidden_dim, vocab_size),
        None,
    )
}

// ============================================================================
// Criterion Groups
// ============================================================================

#[cfg(feature = "gpu")]
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
    benchmark_fused_batch_matmul,
    benchmark_quantized_vs_dequantized,
    benchmark_kv_cache_attention,
    benchmark_e2e_generation,
    benchmark_component_profiling,
    benchmark_q4k_matvec_optimization,
    benchmark_batch_prefill,
    benchmark_gpu_batch_matmul,
    benchmark_batched_causal_attention,
    benchmark_parallel_multihead_attention,
    benchmark_tiled_attention,
    benchmark_scheduler_caching,
    benchmark_single_dispatch_attention,
    benchmark_flattened_batched_gemm,
    benchmark_fused_kernel_attention,
    benchmark_gpu_cpu_crossover,
);

#[cfg(not(feature = "gpu"))]
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
    benchmark_quantized_vs_dequantized,
    benchmark_kv_cache_attention,
    benchmark_e2e_generation,
    benchmark_component_profiling,
    benchmark_q4k_matvec_optimization,
    benchmark_batch_prefill,
);

criterion_main!(benches);
