//! Convert GGUF to APR and benchmark all formats
//!
//! Compares three formats:
//! 1. GGUF (Q4_0 quantized) - baseline
//! 2. APR F32 (dequantized) - memory bandwidth limited
//! 3. APR Q4_0 (quantized) - should match GGUF performance
//!
//! Usage:
//!   GGUF_MODEL=/path/to/model.gguf cargo run --example convert_and_bench_apr --release

use realizar::apr_transformer::QuantizedAprTransformerQ4;
use realizar::convert::GgufToAprConverter;
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

    // === Convert to APR F32 ===
    println!("\n2. Converting GGUF to APR F32 (dequantize)...");
    let start = Instant::now();
    let apr_f32 = GgufToAprConverter::convert(&gguf_data).expect("Failed to convert");
    println!("   Conversion completed in {:.2}s", start.elapsed().as_secs_f32());

    let stats = GgufToAprConverter::stats(&apr_f32);
    println!("   F32 memory: {:.1} MB ({:.1}x vs Q4_0)",
        stats.memory_mb(),
        stats.memory_mb() / (gguf_data.len() as f64 / 1e6));

    // === Convert to APR Q4_0 ===
    println!("\n3. Creating APR Q4_0 (keeps quantization)...");
    let start = Instant::now();
    let apr_q4 = QuantizedAprTransformerQ4::from_gguf(&gguf_model);
    println!("   Conversion completed in {:.2}s", start.elapsed().as_secs_f32());
    println!("   Q4_0 memory: {:.1} MB", apr_q4.memory_size() as f64 / 1e6);

    // === Benchmark GGUF ===
    println!("\n4. Benchmarking GGUF (Q4_0 quantized)...");
    for _ in 0..3 {
        let _ = gguf_model.forward(&[1]);
    }
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = gguf_model.forward(&[1]);
    }
    let gguf_avg = start.elapsed() / iterations;
    let gguf_tps = 1.0 / gguf_avg.as_secs_f64();
    println!("   Average forward: {:.1}ms", gguf_avg.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.1} tok/s", gguf_tps);

    // === Benchmark APR F32 ===
    println!("\n5. Benchmarking APR F32 (dequantized)...");
    for _ in 0..3 {
        let _ = apr_f32.forward(&[1]);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = apr_f32.forward(&[1]);
    }
    let apr_f32_avg = start.elapsed() / iterations;
    let apr_f32_tps = 1.0 / apr_f32_avg.as_secs_f64();
    println!("   Average forward: {:.1}ms", apr_f32_avg.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.1} tok/s", apr_f32_tps);

    // === Benchmark APR Q4_0 ===
    println!("\n6. Benchmarking APR Q4_0 (SIMD quantized)...");
    for _ in 0..3 {
        let _ = apr_q4.forward(&[1]);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = apr_q4.forward(&[1]);
    }
    let apr_q4_avg = start.elapsed() / iterations;
    let apr_q4_tps = 1.0 / apr_q4_avg.as_secs_f64();
    println!("   Average forward: {:.1}ms", apr_q4_avg.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.1} tok/s", apr_q4_tps);

    // === Summary ===
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         Summary                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ Format    │ Size (MB) │ Forward (ms) │ Throughput │ vs GGUF      ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║ GGUF Q4_0 │ {:>9.1} │ {:>12.1} │ {:>7.1} tok/s │ 1.00x        ║",
        gguf_data.len() as f64 / 1e6,
        gguf_avg.as_secs_f64() * 1000.0,
        gguf_tps
    );
    println!(
        "║ APR F32   │ {:>9.1} │ {:>12.1} │ {:>7.1} tok/s │ {:.2}x        ║",
        stats.memory_mb(),
        apr_f32_avg.as_secs_f64() * 1000.0,
        apr_f32_tps,
        apr_f32_tps / gguf_tps
    );
    println!(
        "║ APR Q4_0  │ {:>9.1} │ {:>12.1} │ {:>7.1} tok/s │ {:.2}x        ║",
        apr_q4.memory_size() as f64 / 1e6,
        apr_q4_avg.as_secs_f64() * 1000.0,
        apr_q4_tps,
        apr_q4_tps / gguf_tps
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\nKey findings:");
    println!("  • APR F32 is {:.0}x slower than GGUF (memory bandwidth limited)",
        gguf_tps / apr_f32_tps);
    println!("  • APR Q4_0 achieves {:.0}% of GGUF performance (same SIMD matmul)",
        (apr_q4_tps / gguf_tps) * 100.0);
    println!("  • Quantization reduces memory by {:.1}x ({:.1} MB vs {:.1} MB)",
        stats.memory_mb() / (gguf_data.len() as f64 / 1e6),
        stats.memory_mb(),
        gguf_data.len() as f64 / 1e6);
}
