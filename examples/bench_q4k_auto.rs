//! Benchmark Q4K×Q8K with pre-quantized activations
use std::time::Instant;
use realizar::quantize::fused_q4k_auto_matvec_into;

fn main() {
    let hidden: usize = 1536;
    let inter: usize = 8960;
    
    let super_blocks_per_row = hidden.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;
    let weight_bytes = inter * bytes_per_row;
    
    println!("Q4K Auto (with Q8K quantization) Benchmark");
    println!("  Hidden: {}, Intermediate: {}", hidden, inter);
    println!("  Weight bytes: {} ({:.1} MB)", weight_bytes, weight_bytes as f64 / 1e6);
    
    // Create test data
    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 / hidden as f32) * 2.0 - 1.0)
        .collect();
    let mut output = vec![0.0f32; inter];
    
    // Warmup
    for _ in 0..10 {
        let _ = fused_q4k_auto_matvec_into(&weights, &activations, hidden, inter, &mut output);
    }
    
    // Benchmark
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = fused_q4k_auto_matvec_into(&weights, &activations, hidden, inter, &mut output);
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / iters as f64;
    
    println!("\nResults (FFN up/gate: {}→{}):", hidden, inter);
    println!("  Per matmul: {:.1} µs", per_iter_us);
    
    let matmuls_per_layer = 7;
    let layers = 28;
    let estimated_token_ms = per_iter_us * matmuls_per_layer as f64 * layers as f64 / 1000.0;
    println!("  Estimated per token ({} layers × {} matmuls): {:.1} ms = {:.1} tok/s", 
        layers, matmuls_per_layer, estimated_token_ms, 1000.0 / estimated_token_ms);
    
    let ollama_tok_s = 290.0;
    let target_token_ms = 1000.0 / ollama_tok_s;
    println!("\nTarget: {:.1} ms/token ({:.0} tok/s)", target_token_ms, ollama_tok_s);
    println!("Gap: {:.1}x slower", estimated_token_ms / target_token_ms);
    
    let ops = hidden * inter;
    let gflops = (ops as f64 * 2.0) / (per_iter_us * 1e3);
    println!("\nThroughput: {:.1} GFLOPS", gflops);
}
