//! Benchmark suite for quantization/dequantization operations
//!
//! Measures dequantization performance for different quantization formats:
//! - Q4_0 (4-bit, block size 32)
//! - Q8_0 (8-bit, block size 32)
//! - Q4_K (4-bit K-quant, super-block size 256)
//! - Q5_K (5-bit K-quant, super-block size 256)
//! - Q6_K (6-bit K-quant, super-block size 256)
//!
//! Performance targets:
//! - Q4_0/Q8_0: <1μs per block (32 values)
//! - Q4_K/Q5_K/Q6_K: <10μs per super-block (256 values)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use realizar::quantize::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};

/// Create Q4_0 test data (20 bytes per block = 2 bytes scale + 16 bytes quantized + 2 bytes padding)
fn create_q4_0_data(num_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_blocks * 20);
    for _ in 0..num_blocks {
        // Scale (f16 as 2 bytes, simplified)
        data.extend_from_slice(&1.0f32.to_le_bytes()[..2]);
        // 16 bytes of 4-bit quantized data (32 values packed)
        for _ in 0..16 {
            data.push(0x12); // Demo data
        }
        // 2 bytes padding for alignment
        data.extend_from_slice(&[0, 0]);
    }
    data
}

/// Create Q8_0 test data (36 bytes per block = 2 bytes scale + 32 bytes quantized + 2 bytes padding)
fn create_q8_0_data(num_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_blocks * 36);
    for _ in 0..num_blocks {
        // Scale (f16 as 2 bytes, simplified)
        data.extend_from_slice(&0.5f32.to_le_bytes()[..2]);
        // 32 bytes of 8-bit quantized data
        for i in 0..32 {
            data.push(i as u8);
        }
        // 2 bytes padding for alignment
        data.extend_from_slice(&[0, 0]);
    }
    data
}

/// Create Q4_K test data (144 bytes per super-block)
fn create_q4_k_data(num_super_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_super_blocks * 144);
    for _ in 0..num_super_blocks {
        // Scales and mins (2 bytes each, 12 scales + 12 mins = 48 bytes)
        for _ in 0..24 {
            data.extend_from_slice(&1.0f32.to_le_bytes()[..2]);
        }
        // d (4 bytes f32) + dmin (4 bytes f32) = 8 bytes
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // Quantized data (256 values * 4 bits / 8 = 128 bytes - 48 scales/mins - 8 d/dmin = 72 bytes)
        // Actually: 144 - 48 - 8 = 88 bytes for quantized data
        for _ in 0..88 {
            data.push(0x12);
        }
    }
    data
}

/// Create Q5_K test data (176 bytes per super-block)
fn create_q5_k_data(num_super_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_super_blocks * 176);
    for _ in 0..num_super_blocks {
        // Similar structure to Q4_K but with 5-bit encoding
        // Scales (12 * 2 = 24 bytes)
        for _ in 0..12 {
            data.extend_from_slice(&1.0f32.to_le_bytes()[..2]);
        }
        // Mins (12 * 2 = 24 bytes)
        for _ in 0..12 {
            data.extend_from_slice(&0.0f32.to_le_bytes()[..2]);
        }
        // d (4 bytes) + dmin (4 bytes) = 8 bytes
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // Quantized data (176 - 24 - 24 - 8 = 120 bytes)
        for _ in 0..120 {
            data.push(0x12);
        }
    }
    data
}

/// Create Q6_K test data (210 bytes per super-block)
fn create_q6_k_data(num_super_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_super_blocks * 210);
    for _ in 0..num_super_blocks {
        // Scales (16 * 2 = 32 bytes for Q6_K)
        for _ in 0..16 {
            data.extend_from_slice(&1.0f32.to_le_bytes()[..2]);
        }
        // d (4 bytes f32)
        data.extend_from_slice(&1.0f32.to_le_bytes());
        // Quantized data (210 - 32 - 4 = 174 bytes)
        for _ in 0..174 {
            data.push(0x12);
        }
    }
    data
}

// Benchmark: Q4_0 dequantization
fn benchmark_q4_0_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q4_0_dequantize");

    for num_blocks in [1, 10, 100, 1000].iter() {
        let data = create_q4_0_data(*num_blocks);
        let num_values = num_blocks * 32; // 32 values per block

        group.bench_with_input(BenchmarkId::new("blocks", num_blocks), &data, |b, data| {
            b.iter(|| {
                let result = dequantize_q4_0(black_box(data)).expect("Dequantization failed");
                black_box(result)
            });
        });

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q4_0(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Q8_0 dequantization
fn benchmark_q8_0_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q8_0_dequantize");

    for num_blocks in [1, 10, 100, 1000].iter() {
        let data = create_q8_0_data(*num_blocks);
        let num_values = num_blocks * 32;

        group.bench_with_input(BenchmarkId::new("blocks", num_blocks), &data, |b, data| {
            b.iter(|| {
                let result = dequantize_q8_0(black_box(data)).expect("Dequantization failed");
                black_box(result)
            });
        });

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q8_0(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Q4_K dequantization
fn benchmark_q4_k_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q4_k_dequantize");

    for num_super_blocks in [1, 10, 100].iter() {
        let data = create_q4_k_data(*num_super_blocks);
        let num_values = num_super_blocks * 256; // 256 values per super-block

        group.bench_with_input(
            BenchmarkId::new("super_blocks", num_super_blocks),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = dequantize_q4_k(black_box(data)).expect("Dequantization failed");
                    black_box(result)
                });
            },
        );

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q4_k(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Q5_K dequantization
fn benchmark_q5_k_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q5_k_dequantize");

    for num_super_blocks in [1, 10, 100].iter() {
        let data = create_q5_k_data(*num_super_blocks);
        let num_values = num_super_blocks * 256;

        group.bench_with_input(
            BenchmarkId::new("super_blocks", num_super_blocks),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = dequantize_q5_k(black_box(data)).expect("Dequantization failed");
                    black_box(result)
                });
            },
        );

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q5_k(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Q6_K dequantization
fn benchmark_q6_k_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q6_k_dequantize");

    for num_super_blocks in [1, 10, 100].iter() {
        let data = create_q6_k_data(*num_super_blocks);
        let num_values = num_super_blocks * 256;

        group.bench_with_input(
            BenchmarkId::new("super_blocks", num_super_blocks),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = dequantize_q6_k(black_box(data)).expect("Dequantization failed");
                    black_box(result)
                });
            },
        );

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q6_k(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Compare all quantization formats
fn benchmark_quantization_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_comparison");

    // All benchmarks dequantize ~1000 values for fair comparison
    let q4_0_data = create_q4_0_data(32); // 32 blocks * 32 values = 1024 values
    let q8_0_data = create_q8_0_data(32); // 32 blocks * 32 values = 1024 values
    let q4_k_data = create_q4_k_data(4); // 4 super-blocks * 256 values = 1024 values
    let q5_k_data = create_q5_k_data(4); // 4 super-blocks * 256 values = 1024 values
    let q6_k_data = create_q6_k_data(4); // 4 super-blocks * 256 values = 1024 values

    group.bench_function("q4_0_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q4_0(black_box(&q4_0_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.bench_function("q8_0_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q8_0(black_box(&q8_0_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.bench_function("q4_k_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q4_k(black_box(&q4_k_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.bench_function("q5_k_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q5_k(black_box(&q5_k_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.bench_function("q6_k_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q6_k(black_box(&q6_k_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_q4_0_dequantize,
    benchmark_q8_0_dequantize,
    benchmark_q4_k_dequantize,
    benchmark_q5_k_dequantize,
    benchmark_q6_k_dequantize,
    benchmark_quantization_comparison
);
criterion_main!(benches);
