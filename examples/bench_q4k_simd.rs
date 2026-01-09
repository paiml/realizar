//! Benchmark: Q4_K dot product SIMD kernel throughput
//!
//! Measures the raw performance of fused_q4k_dot_simd to identify
//! remaining optimization opportunities.

use realizar::quantize::{
    fused_q4k_dot, fused_q4k_dot_simd, fused_q4k_q8_dot, fused_q4k_q8_dot_simd,
    quantize_to_q8_blocks, QK_K,
};
use std::time::Instant;

fn main() {
    // Create test data matching TinyLlama hidden_dim = 2048
    let hidden_dim = 2048;
    let num_super_blocks = hidden_dim / QK_K; // 2048 / 256 = 8 super-blocks
    let bytes_per_superblock = 144;

    let mut q4k_data = Vec::with_capacity(num_super_blocks * bytes_per_superblock);

    for sb_idx in 0..num_super_blocks {
        // d (f16)
        let d = 0.5 + (sb_idx as f32) * 0.1;
        q4k_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());

        // dmin (f16)
        q4k_data.extend_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());

        // scales: 12 bytes
        for i in 0..12 {
            q4k_data.push(((sb_idx * 7 + i) % 64) as u8);
        }

        // qs: 128 bytes
        for i in 0..128 {
            q4k_data.push(((sb_idx * 13 + i) % 256) as u8);
        }
    }

    let activations: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.017).sin()).collect();

    // Verify correctness
    let scalar_result = fused_q4k_dot(&q4k_data, &activations).expect("scalar");
    let simd_result = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd");
    let diff = (scalar_result - simd_result).abs();
    println!("Scalar result: {:.6}", scalar_result);
    println!("SIMD result:   {:.6}", simd_result);
    println!("Difference:    {:.9} (should be < 1e-4)", diff);
    // Allow 0.1% relative error for SIMD FP reassociation
    let rel_diff = diff / scalar_result.abs().max(1e-10);
    assert!(
        rel_diff < 1e-3,
        "SIMD result diverged too much: rel_diff={}",
        rel_diff
    );

    // Warmup
    for _ in 0..1000 {
        let _ = fused_q4k_dot_simd(&q4k_data, &activations);
    }

    // Benchmark scalar
    let iterations = 100_000;
    println!(
        "\n=== Benchmark ({} iterations, {} values per dot) ===",
        iterations, hidden_dim
    );

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_dot(&q4k_data, &activations);
    }
    let scalar_time = start.elapsed();
    let scalar_ns = scalar_time.as_nanos() as f64 / iterations as f64;
    let scalar_gflops = (2.0 * hidden_dim as f64) / scalar_ns; // 2 FLOPs per element (mul + add)

    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_dot_simd(&q4k_data, &activations);
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() as f64 / iterations as f64;
    let simd_gflops = (2.0 * hidden_dim as f64) / simd_ns; // 2 FLOPs per element (mul + add)

    println!(
        "Scalar: {:.1}ns per dot, {:.2} GFLOP/s",
        scalar_ns, simd_gflops
    );
    println!(
        "SIMD:   {:.1}ns per dot, {:.2} GFLOP/s",
        simd_ns, simd_gflops
    );
    println!("Speedup: {:.2}x", scalar_ns / simd_ns);

    // Calculate throughput in terms of model matmul operations
    // TinyLlama: 22 layers, each with multiple matmuls
    // Attention: Q, K, V (3x), O (1x) = 4 matmuls per layer
    // FFN: gate (1x), up (1x), down (1x) = 3 matmuls per layer
    // Total: 7 matmuls per layer × 22 layers = 154 matmuls per token
    let matmuls_per_token = 154;
    let ns_per_matmul = simd_ns * (hidden_dim as f64 / 1.0); // scale by typical matmul size
    let ns_per_token = ns_per_matmul * matmuls_per_token as f64;
    let estimated_toks = 1e9 / ns_per_token;

    println!("\n=== Estimated Performance ===");
    println!("If matmul were the only cost: {:.1} tok/s", estimated_toks);
    println!("Current actual:               9.7 tok/s");
    println!("Gap suggests other bottlenecks (memory bandwidth, attention, etc.)");

    // Memory bandwidth analysis
    let q4k_bytes_per_element = 4.5 / 8.0; // Q4_K is ~4.5 bits per element
    let q4k_bytes = hidden_dim as f64 * q4k_bytes_per_element;
    let act_bytes = hidden_dim as f64 * 4.0; // f32 activations
    let total_bytes = q4k_bytes + act_bytes;
    let bandwidth_gb_s = (total_bytes * iterations as f64) / (simd_time.as_secs_f64() * 1e9);
    println!("\nMemory bandwidth: {:.1} GB/s", bandwidth_gb_s);
    println!("DDR4-3200 theoretical max: ~25 GB/s");

    // =========================================================================
    // Q4_K × Q8_0 Integer SIMD Benchmark (llama.cpp pattern)
    // =========================================================================
    println!("\n=== Q4_K × Q8_0 Integer SIMD Benchmark (llama.cpp pattern) ===");

    // Quantize activations to Q8_0
    let q8_blocks = quantize_to_q8_blocks(&activations).expect("Q8 quantization");
    println!(
        "Quantized {} f32 activations to {} Q8_0 blocks",
        hidden_dim,
        q8_blocks.len()
    );

    // Verify correctness
    let scalar_q8_result = fused_q4k_q8_dot(&q4k_data, &q8_blocks).expect("scalar q8");
    let simd_q8_result = fused_q4k_q8_dot_simd(&q4k_data, &q8_blocks).expect("simd q8");
    let q8_diff = (scalar_q8_result - simd_q8_result).abs();
    let q8_rel_diff = q8_diff / scalar_q8_result.abs().max(1e-10);
    println!("Scalar Q8 result: {:.6}", scalar_q8_result);
    println!("SIMD Q8 result:   {:.6}", simd_q8_result);
    println!("Relative diff:    {:.9}", q8_rel_diff);

    // Warmup
    for _ in 0..1000 {
        let _ = fused_q4k_q8_dot_simd(&q4k_data, &q8_blocks);
    }

    // Benchmark scalar Q8
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    }
    let scalar_q8_time = start.elapsed();
    let scalar_q8_ns = scalar_q8_time.as_nanos() as f64 / iterations as f64;

    // Benchmark SIMD Q8
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_q8_dot_simd(&q4k_data, &q8_blocks);
    }
    let simd_q8_time = start.elapsed();
    let simd_q8_ns = simd_q8_time.as_nanos() as f64 / iterations as f64;

    println!("\nQ4_K × Q8_0 Results:");
    println!("Scalar Q8: {:.1}ns per dot", scalar_q8_ns);
    println!("SIMD Q8:   {:.1}ns per dot", simd_q8_ns);
    println!(
        "Q8 SIMD speedup over Q8 scalar: {:.2}x",
        scalar_q8_ns / simd_q8_ns
    );
    println!(
        "Q8 SIMD speedup over f32 SIMD:  {:.2}x",
        simd_ns / simd_q8_ns
    );

    // Compare all paths
    println!("\n=== Summary ===");
    println!("f32 scalar: {:.1}ns", scalar_ns);
    println!(
        "f32 SIMD:   {:.1}ns (speedup: {:.2}x)",
        simd_ns,
        scalar_ns / simd_ns
    );
    println!("Q8  scalar: {:.1}ns", scalar_q8_ns);
    println!(
        "Q8  SIMD:   {:.1}ns (speedup vs f32 scalar: {:.2}x)",
        simd_q8_ns,
        scalar_ns / simd_q8_ns
    );
}
