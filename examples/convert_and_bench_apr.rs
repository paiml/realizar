//! Convert GGUF to APR and benchmark all formats
//!
//! Compares:
//! 1. GGUF (Q4_0 quantized) - baseline
//! 2. APR Q4_0 (quantized) - parallel matmul
//! 3. APR Q4_0 + KV Cache - context-aware generation
//!
//! Usage:
//!   GGUF_MODEL=/path/to/model.gguf cargo run --example convert_and_bench_apr --release

use realizar::apr_transformer::QuantizedAprTransformerQ4;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::time::Instant;

fn main() {
    let model_path = std::env::var("GGUF_MODEL")
        .unwrap_or_else(|_| "/mnt/ssd/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf".to_string());

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           GGUF vs APR Inference Benchmark                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // === Load GGUF Model ===
    println!("1. Loading GGUF model: {}", model_path);
    let start = Instant::now();
    let gguf_data = std::fs::read(&model_path).expect("Failed to read GGUF file");
    println!("   File size: {:.1} MB", gguf_data.len() as f64 / 1e6);

    let mapped = MappedGGUFModel::from_path(&model_path).expect("Failed to load GGUF");
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");
    println!("   GGUF loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!(
        "   Config: hidden_dim={}, vocab_size={}, layers={}",
        gguf_model.config().hidden_dim,
        gguf_model.config().vocab_size,
        gguf_model.config().num_layers
    );

    // === Convert to APR Q4_0 ===
    println!("\n2. Creating APR Q4_0...");
    let start = Instant::now();
    let apr_q4 = QuantizedAprTransformerQ4::from_gguf(&gguf_model);
    println!("   Conversion completed in {:.2}s", start.elapsed().as_secs_f32());

    // === Benchmark GGUF (single token, no context) ===
    println!("\n3. Benchmarking GGUF (single token)...");
    println!("   Warming up...");
    for _ in 0..10 {
        let _ = gguf_model.forward(&[1]);
    }
    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = gguf_model.forward(&[1]);
    }
    let gguf_avg = start.elapsed() / iterations;
    let gguf_tps = 1.0 / gguf_avg.as_secs_f64();
    println!("   Throughput: {:.1} tok/s ({:.1}ms/tok)", gguf_tps, gguf_avg.as_secs_f64() * 1000.0);

    // === Benchmark APR Q4_0 (single token, no context) ===
    println!("\n4. Benchmarking APR Q4_0 (single token, parallel)...");
    println!("   Warming up...");
    for _ in 0..10 {
        let _ = apr_q4.forward(&[1]);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = apr_q4.forward(&[1]);
    }
    let apr_q4_avg = start.elapsed() / iterations;
    let apr_q4_tps = 1.0 / apr_q4_avg.as_secs_f64();
    println!("   Throughput: {:.1} tok/s ({:.1}ms/tok)", apr_q4_tps, apr_q4_avg.as_secs_f64() * 1000.0);

    // === Benchmark APR Q4_0 with KV Cache (context-aware generation) ===
    println!("\n5. Benchmarking APR Q4_0 + KV Cache (context-aware)...");

    // First: Measure per-token cost with growing context
    let mut cache = apr_q4.create_kv_cache();

    // Warmup
    for i in 0..5 {
        let _ = apr_q4.forward_with_cache(&[(i % 100) as u32], &mut cache);
    }
    cache.clear();

    // Measure generation with context building up
    let gen_tokens = 20u32;
    let start = Instant::now();
    for i in 0..gen_tokens {
        let _ = apr_q4.forward_with_cache(&[(i % 100) as u32], &mut cache);
    }
    let cache_total = start.elapsed();
    let cache_avg = cache_total / gen_tokens;
    let cache_tps = gen_tokens as f64 / cache_total.as_secs_f64();
    println!("   Generated {} tokens in {:.0}ms", gen_tokens, cache_total.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.1} tok/s ({:.1}ms/tok avg)", cache_tps, cache_avg.as_secs_f64() * 1000.0);

    // === Summary ===
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                           Summary                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Format           │ Throughput  │ Latency   │ vs GGUF │ Context      ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ GGUF Q4_0        │ {:>7.1} tok/s │ {:>6.1}ms │ 1.00x   │ None         ║",
        gguf_tps, gguf_avg.as_secs_f64() * 1000.0
    );
    println!(
        "║ APR Q4_0         │ {:>7.1} tok/s │ {:>6.1}ms │ {:.2}x   │ None         ║",
        apr_q4_tps, apr_q4_avg.as_secs_f64() * 1000.0, apr_q4_tps / gguf_tps
    );
    println!(
        "║ APR + KV Cache   │ {:>7.1} tok/s │ {:>6.1}ms │ {:.2}x   │ {} tokens   ║",
        cache_tps, cache_avg.as_secs_f64() * 1000.0, cache_tps / gguf_tps, gen_tokens
    );
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // Analysis
    println!("\nKey Insights:");
    println!("  • Single-token (no context): APR is {:.1}x faster than GGUF", apr_q4_tps / gguf_tps);
    println!("  • Context-aware generation: {:.1} tok/s with KV cache", cache_tps);
    println!("  • KV cache enables attending to past tokens efficiently");

    let llama_cpp_tps = 42.0;
    println!("\n  vs llama.cpp (42 tok/s):");
    println!("    • Single-token APR: {:.0}%", apr_q4_tps / llama_cpp_tps * 100.0);
    println!("    • Context-aware APR: {:.0}%", cache_tps / llama_cpp_tps * 100.0);
}
