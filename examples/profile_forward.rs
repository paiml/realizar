//! Profile forward pass to identify bottlenecks

use std::env;
use std::fs;
use std::time::Instant;
use realizar::gguf::GGUFModel;
use realizar::Model;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });
    
    println!("Loading model: {}", path);
    let start = Instant::now();
    let model = Model::load(&path).expect("Failed to load model");
    println!("Model loaded in {:?}", start.elapsed());
    
    // Run warmup
    let input_ids = vec![151643u32, 2182, 374, 264, 1296]; // "This is a test"
    println!("\nWarmup with {} tokens...", input_ids.len());
    let _ = model.forward(&input_ids, 0);
    
    // Profile forward passes
    println!("\nProfiling 10 forward passes...");
    let iters = 10;
    let start = Instant::now();
    for i in 0..iters {
        let _ = model.forward(&input_ids, i);
    }
    let total = start.elapsed();
    let avg_ms = total.as_millis() as f64 / iters as f64;
    
    println!("\nResults:");
    println!("  Total time: {:?}", total);
    println!("  Avg per forward: {:.1} ms", avg_ms);
    println!("  Throughput: {:.1} tok/s", 1000.0 / avg_ms);
    
    // Also test single token generation
    println!("\nSingle token generation (decode):");
    let single_token = vec![2182u32]; // Single token
    let start = Instant::now();
    for i in 0..iters {
        let _ = model.forward(&single_token, i + 10);
    }
    let decode_total = start.elapsed();
    let decode_avg_ms = decode_total.as_millis() as f64 / iters as f64;
    println!("  Avg per token: {:.1} ms", decode_avg_ms);
    println!("  Decode throughput: {:.1} tok/s", 1000.0 / decode_avg_ms);
}
