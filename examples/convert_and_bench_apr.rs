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
    println!(
        "   GQA: num_heads={}, num_kv_heads={}",
        gguf_model.config().num_heads,
        gguf_model.config().num_kv_heads
    );

    // === Convert to APR Q4_0 ===
    println!("\n2. Creating APR Q4_0 (keeps quantization)...");
    let start = Instant::now();
    let apr_q4 = QuantizedAprTransformerQ4::from_gguf(&gguf_model);
    println!("   Conversion completed in {:.2}s", start.elapsed().as_secs_f32());
    println!("   Q4_0 memory: {:.1} MB", apr_q4.memory_size() as f64 / 1e6);

    // Print layer 0 weight dimensions for debugging
    if let Some(layer) = apr_q4.layers.first() {
        println!("\n   Layer 0 weight dimensions:");
        println!("     qkv_weight: in={}, out={}, bytes={}",
            layer.qkv_weight.in_dim, layer.qkv_weight.out_dim, layer.qkv_weight.data.len());
        println!("     attn_output_weight: in={}, out={}, bytes={}",
            layer.attn_output_weight.in_dim, layer.attn_output_weight.out_dim, layer.attn_output_weight.data.len());
        println!("     ffn_up_weight: in={}, out={}, bytes={}",
            layer.ffn_up_weight.in_dim, layer.ffn_up_weight.out_dim, layer.ffn_up_weight.data.len());
        println!("     ffn_down_weight: in={}, out={}, bytes={}",
            layer.ffn_down_weight.in_dim, layer.ffn_down_weight.out_dim, layer.ffn_down_weight.data.len());
        if let Some(gate) = &layer.ffn_gate_weight {
            println!("     ffn_gate_weight: in={}, out={}, bytes={}",
                gate.in_dim, gate.out_dim, gate.data.len());
        }
    }

    // === Benchmark GGUF ===
    println!("\n3. Benchmarking GGUF (Q4_0 quantized)...");
    // More warmup for stable results
    println!("   Warming up (10 iterations)...");
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
    println!("   Average forward: {:.1}ms", gguf_avg.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.1} tok/s", gguf_tps);

    // === Benchmark APR Q4_0 ===
    println!("\n4. Benchmarking APR Q4_0 (SIMD quantized)...");
    println!("   Warming up (10 iterations)...");
    for _ in 0..10 {
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
        "║ APR Q4_0  │ {:>9.1} │ {:>12.1} │ {:>7.1} tok/s │ {:.2}x        ║",
        apr_q4.memory_size() as f64 / 1e6,
        apr_q4_avg.as_secs_f64() * 1000.0,
        apr_q4_tps,
        apr_q4_tps / gguf_tps
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Analysis
    let overhead_ms = apr_q4_avg.as_secs_f64() * 1000.0 - gguf_avg.as_secs_f64() * 1000.0;
    let overhead_per_layer = overhead_ms / gguf_model.config().num_layers as f64;
    println!("\nPerformance Analysis:");
    println!("  • Total overhead: {:.1}ms ({:.1}ms per layer)", overhead_ms, overhead_per_layer);
    println!("  • Speedup needed: {:.1}x to match GGUF", apr_q4_avg.as_secs_f64() / gguf_avg.as_secs_f64());
}
