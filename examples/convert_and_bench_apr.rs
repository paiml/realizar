//! Convert GGUF to APR and benchmark both formats
//!
//! Usage:
//!   GGUF_MODEL=/path/to/model.gguf cargo run --example convert_and_bench_apr --release --features aprender-serve

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

    // === Convert to APR ===
    println!("\n2. Converting GGUF to APR (dequantize to F32)...");
    let start = Instant::now();
    let apr_transformer = GgufToAprConverter::convert(&gguf_data).expect("Failed to convert");
    let convert_time = start.elapsed();
    println!("   Conversion completed in {:.2}s", convert_time.as_secs_f32());

    // Get stats
    let stats = GgufToAprConverter::stats(&apr_transformer);
    println!("   Parameters: {:.1}M ({:.2}B)", stats.parameters_m(), stats.parameters_b());
    println!("   F32 memory: {:.1} MB ({:.2} GB)", stats.memory_mb(), stats.memory_gb());
    println!("   Expansion: {:.1}x vs Q4_0", stats.memory_mb() / (gguf_data.len() as f64 / 1e6));

    // === Skip APR file save (JSON serialization is too large) ===
    // The APR JSON format would be ~12GB for TinyLlama (3x the F32 binary size)
    // In production, use binary serialization instead
    println!("\n3. Skipping APR file save (JSON too large for 1B+ models)");
    let apr_file_size_estimate = stats.memory_bytes_f32 as f64 * 3.0; // JSON ~3x binary
    println!("   Estimated APR JSON size: {:.1} GB", apr_file_size_estimate / 1e9);

    // === Benchmark GGUF ===
    println!("\n4. Benchmarking GGUF (Q4_0 quantized)...");

    // Warmup
    for _ in 0..3 {
        let _ = gguf_model.forward(&[1]);
    }

    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = gguf_model.forward(&[1]);
    }
    let gguf_total = start.elapsed();
    let gguf_avg = gguf_total / iterations;
    let gguf_tps = 1.0 / gguf_avg.as_secs_f64();

    println!("   Average forward: {:.1}ms", gguf_avg.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.1} tok/s", gguf_tps);

    // === Benchmark APR ===
    println!("\n5. Benchmarking APR (F32 dequantized)...");

    // Debug: print layer weight dimensions
    if let Some(layer) = apr_transformer.layers.first() {
        println!("   Layer 0 weight sizes:");
        println!("     qkv_weight: {} (hidden_dim={}, expected qkv_dim={})",
            layer.qkv_weight.len(),
            apr_transformer.config.hidden_dim,
            apr_transformer.config.hidden_dim * 3);
        println!("     attn_output_weight: {}", layer.attn_output_weight.len());
        println!("     ffn_up_weight: {}", layer.ffn_up_weight.len());
        println!("     ffn_down_weight: {}", layer.ffn_down_weight.len());
    }

    // Warmup
    for _ in 0..3 {
        let _ = apr_transformer.forward(&[1]);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = apr_transformer.forward(&[1]);
    }
    let apr_total = start.elapsed();
    let apr_avg = apr_total / iterations;
    let apr_tps = 1.0 / apr_avg.as_secs_f64();

    println!("   Average forward: {:.1}ms", apr_avg.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.1} tok/s", apr_tps);

    // === Summary ===
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         Summary                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ Format │ Size (MB) │ Forward (ms) │ Throughput │ vs GGUF         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║ GGUF   │ {:>9.1} │ {:>12.1} │ {:>7.1} tok/s │ 1.00x           ║",
        gguf_data.len() as f64 / 1e6,
        gguf_avg.as_secs_f64() * 1000.0,
        gguf_tps
    );
    println!(
        "║ APR    │ {:>9.1} │ {:>12.1} │ {:>7.1} tok/s │ {:.2}x           ║",
        stats.memory_mb(),
        apr_avg.as_secs_f64() * 1000.0,
        apr_tps,
        apr_tps / gguf_tps
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\nNote: APR uses F32 weights ({:.1}x larger than Q4_0)", stats.memory_mb() / (gguf_data.len() as f64 / 1e6));
}
