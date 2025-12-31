//! Profile forward pass breakdown to find bottlenecks
//!
//! Run with: cargo run --release --example profile_forward_breakdown -- <model.gguf>

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
use std::time::Instant;
use std::{env, sync::atomic::{AtomicU64, Ordering}};

// Global timing accumulators
static EMBED_TIME: AtomicU64 = AtomicU64::new(0);
static NORM_TIME: AtomicU64 = AtomicU64::new(0);
static QKV_TIME: AtomicU64 = AtomicU64::new(0);
static ROPE_TIME: AtomicU64 = AtomicU64::new(0);
static ATTN_TIME: AtomicU64 = AtomicU64::new(0);
static OUT_PROJ_TIME: AtomicU64 = AtomicU64::new(0);
static FFN_TIME: AtomicU64 = AtomicU64::new(0);
static LM_HEAD_TIME: AtomicU64 = AtomicU64::new(0);
static OTHER_TIME: AtomicU64 = AtomicU64::new(0);

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/noah/.cache/gguf/qwen2.5-0.5b-q4_0.gguf");

    println!("=== Forward Pass Breakdown Profiling ===\n");

    // Load model
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();
    let load_time = start.elapsed();

    let config = model.config();
    let model_name = path.split('/').last().unwrap_or(path);

    println!("Model: {}", model_name);
    println!("Load time: {:.2?}", load_time);
    println!("Config: {} layers, {} hidden, {} heads, {} kv_heads",
        config.num_layers, config.hidden_dim, config.num_heads, config.num_kv_heads);
    println!();

    // Create KV cache
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let max_seq_len = 128;
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, kv_dim, max_seq_len);

    // Warmup
    println!("Warming up...");
    for pos in 0..10 {
        let _ = model.forward_single_with_cache(1, &mut cache, pos);
    }

    // Reset cache
    cache = OwnedQuantizedKVCache::new(config.num_layers, kv_dim, max_seq_len);

    // Profile multiple tokens
    let num_tokens = 30;
    println!("Profiling {} tokens...\n", num_tokens);

    let mut total_times = Vec::new();
    for pos in 0..num_tokens {
        let start = Instant::now();
        let _ = model.forward_single_with_cache(1, &mut cache, pos);
        total_times.push(start.elapsed());
    }

    // Calculate statistics
    let total_us: Vec<u128> = total_times.iter().map(|t| t.as_micros()).collect();
    let sum: u128 = total_us.iter().sum();
    let avg = sum / total_us.len() as u128;

    let mut sorted = total_us.clone();
    sorted.sort();
    let median = sorted[sorted.len() / 2];

    println!("=== Per-Token Latency ===");
    println!("  Median: {} µs ({:.2} ms)", median, median as f64 / 1000.0);
    println!("  Avg:    {} µs ({:.2} ms)", avg, avg as f64 / 1000.0);
    println!("  Throughput: {:.1} tok/s", 1_000_000.0 / median as f64);
    println!();

    // Estimate breakdown based on model dimensions
    println!("=== Estimated Time Breakdown (theoretical) ===\n");

    let hidden = config.hidden_dim as f64;
    let layers = config.num_layers as f64;
    let inter = config.intermediate_dim as f64;
    let vocab = config.vocab_size as f64;
    let kv_heads = config.num_kv_heads as f64;
    let num_heads = config.num_heads as f64;
    let head_d = hidden / num_heads;

    // Weight sizes (Q4_0 = 0.5 bytes per weight)
    let bytes_per_weight = 0.5;

    // Per-layer matmuls
    let qkv_bytes = hidden * (hidden + 2.0 * kv_heads * head_d) * bytes_per_weight;
    let out_bytes = hidden * hidden * bytes_per_weight;
    let ffn_bytes = 3.0 * hidden * inter * bytes_per_weight; // up, gate, down
    let lm_head_bytes = hidden * vocab * bytes_per_weight;

    // Assume 25 GB/s effective bandwidth (from profiling)
    let bandwidth = 25.0; // GB/s

    let qkv_ms = (qkv_bytes / 1e9) / bandwidth * 1000.0 * layers;
    let out_ms = (out_bytes / 1e9) / bandwidth * 1000.0 * layers;
    let ffn_ms = (ffn_bytes / 1e9) / bandwidth * 1000.0 * layers;
    let lm_head_ms = (lm_head_bytes / 1e9) / bandwidth * 1000.0;
    let total_matmul_ms = qkv_ms + out_ms + ffn_ms + lm_head_ms;

    println!("Matmul times @ {} GB/s:", bandwidth);
    println!("  QKV projection:   {:>6.2} ms ({:>5.1}%)", qkv_ms, 100.0 * qkv_ms / total_matmul_ms);
    println!("  Output projection: {:>6.2} ms ({:>5.1}%)", out_ms, 100.0 * out_ms / total_matmul_ms);
    println!("  FFN (up+gate+down): {:>6.2} ms ({:>5.1}%)", ffn_ms, 100.0 * ffn_ms / total_matmul_ms);
    println!("  LM head:           {:>6.2} ms ({:>5.1}%)", lm_head_ms, 100.0 * lm_head_ms / total_matmul_ms);
    println!("  Total matmul:      {:>6.2} ms", total_matmul_ms);
    println!();

    // Attention cost estimation
    let avg_cache_len = num_tokens as f64 / 2.0; // Average cache length during profiling
    let attn_flops_per_head = 2.0 * avg_cache_len * head_d; // QK dot + softmax + weighted sum
    let total_attn_flops = attn_flops_per_head * num_heads * layers;
    let gflops_capacity = 50.0; // Rough estimate for scalar attention
    let attn_ms = (total_attn_flops / 1e9) / gflops_capacity * 1000.0;

    println!("Attention estimate (avg cache len = {:.0}):", avg_cache_len);
    println!("  Attention compute: {:>6.2} ms", attn_ms);
    println!();

    // Compare to actual
    let actual_ms = median as f64 / 1000.0;
    let estimated_ms = total_matmul_ms + attn_ms;
    let overhead_ms = actual_ms - estimated_ms;

    println!("=== Actual vs Estimated ===");
    println!("  Estimated (matmul + attn): {:>6.2} ms", estimated_ms);
    println!("  Actual (median):           {:>6.2} ms", actual_ms);
    println!("  Overhead:                  {:>6.2} ms ({:.1}%)",
             overhead_ms, 100.0 * overhead_ms / actual_ms);
    println!();

    if overhead_ms > 5.0 {
        println!("=== Overhead Sources ===");
        println!("  - Layer normalization (RMSNorm)");
        println!("  - RoPE position embeddings");
        println!("  - Memory allocation for intermediate vectors");
        println!("  - Thread synchronization between operations");
        println!("  - Activation quantization overhead");
    }

    // Scaling analysis
    println!("\n=== Attention Scaling Analysis ===");
    let first_10_avg: f64 = total_times[0..10].iter().map(|t| t.as_secs_f64()).sum::<f64>() / 10.0;
    let last_10_avg: f64 = total_times[20..30].iter().map(|t| t.as_secs_f64()).sum::<f64>() / 10.0;
    let scaling = last_10_avg / first_10_avg;

    println!("  First 10 tokens avg: {:.2} ms", first_10_avg * 1000.0);
    println!("  Last 10 tokens avg:  {:.2} ms", last_10_avg * 1000.0);
    println!("  Scaling factor:      {:.2}x", scaling);

    if scaling > 1.3 {
        println!("\n  ⚠️  Attention is scaling significantly!");
        println!("  Consider implementing tiled/flash attention.");
    } else {
        println!("\n  ✓ Attention scales reasonably.");
    }
}
