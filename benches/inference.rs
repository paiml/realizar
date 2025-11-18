//! Benchmark suite for inference operations
//!
//! Measures end-to-end inference latency for model generation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use realizar::generate::GenerationConfig;
use realizar::layers::{Model, ModelConfig};

fn create_test_model() -> Model {
    let config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 1,
        num_layers: 2,
        intermediate_dim: 64,
        eps: 1e-5,
    };
    Model::new(config).unwrap()
}

fn benchmark_model_forward(c: &mut Criterion) {
    let model = create_test_model();
    let mut group = c.benchmark_group("model_forward");

    for seq_len in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &seq_len| {
                let tokens: Vec<usize> = (0..seq_len).map(|i| i % 100).collect();
                b.iter(|| {
                    let logits = model.forward(black_box(&tokens)).unwrap();
                    black_box(logits)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_generation_greedy(c: &mut Criterion) {
    let model = create_test_model();
    let config = GenerationConfig::greedy().with_max_tokens(5);

    c.bench_function("generation_greedy_5_tokens", |b| {
        let prompt = vec![1, 5, 10];
        b.iter(|| {
            let result = model.generate(black_box(&prompt), black_box(&config)).unwrap();
            black_box(result)
        });
    });
}

fn benchmark_generation_top_k(c: &mut Criterion) {
    let model = create_test_model();
    let config = GenerationConfig::top_k(5)
        .with_max_tokens(5)
        .with_seed(42);

    c.bench_function("generation_top_k_5_tokens", |b| {
        let prompt = vec![1, 5, 10];
        b.iter(|| {
            let result = model.generate(black_box(&prompt), black_box(&config)).unwrap();
            black_box(result)
        });
    });
}

fn benchmark_generation_top_p(c: &mut Criterion) {
    let model = create_test_model();
    let config = GenerationConfig::top_p(0.9)
        .with_max_tokens(5)
        .with_seed(42);

    c.bench_function("generation_top_p_5_tokens", |b| {
        let prompt = vec![1, 5, 10];
        b.iter(|| {
            let result = model.generate(black_box(&prompt), black_box(&config)).unwrap();
            black_box(result)
        });
    });
}

fn benchmark_generation_varying_length(c: &mut Criterion) {
    let model = create_test_model();
    let mut group = c.benchmark_group("generation_varying_length");

    for max_tokens in [1, 5, 10, 20].iter() {
        let config = GenerationConfig::greedy().with_max_tokens(*max_tokens);
        group.bench_with_input(
            BenchmarkId::from_parameter(max_tokens),
            max_tokens,
            |b, _| {
                let prompt = vec![1, 5, 10];
                b.iter(|| {
                    let result = model.generate(black_box(&prompt), black_box(&config)).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_sampling_strategies(c: &mut Criterion) {
    let model = create_test_model();
    let mut group = c.benchmark_group("sampling_strategies");

    let strategies = vec![
        ("greedy", GenerationConfig::greedy()),
        ("top_k_5", GenerationConfig::top_k(5).with_seed(42)),
        ("top_k_50", GenerationConfig::top_k(50).with_seed(42)),
        ("top_p_0.9", GenerationConfig::top_p(0.9).with_seed(42)),
        ("top_p_0.95", GenerationConfig::top_p(0.95).with_seed(42)),
    ];

    for (name, mut config) in strategies {
        config = config.with_max_tokens(10);
        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, config| {
            let prompt = vec![1, 5, 10];
            b.iter(|| {
                let result = model.generate(black_box(&prompt), black_box(config)).unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_model_forward,
    benchmark_generation_greedy,
    benchmark_generation_top_k,
    benchmark_generation_top_p,
    benchmark_generation_varying_length,
    benchmark_sampling_strategies,
);
criterion_main!(benches);
