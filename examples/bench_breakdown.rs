//! Benchmark breakdown: measure each component separately
//! to identify the true bottleneck

use std::time::Instant;

fn main() {
    let hidden = 1536;
    let inter = 8960;
    let super_blocks = hidden / 256;
    let bytes_per_row = super_blocks * 144;
    let weight_bytes = inter * bytes_per_row;

    println!("Performance Breakdown (1536 -> 8960 matmul)");
    println!("============================================\n");

    // Create test data
    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 / hidden as f32) * 2.0 - 1.0)
        .collect();

    // Pre-quantized Q8K
    let (q8k_scales, q8k_quants) = quantize_to_q8k(&activations);

    // 1. Measure Q8K quantization time
    let iters = 1000;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = quantize_to_q8k(&activations);
    }
    let quant_us = start.elapsed().as_micros() as f64 / iters as f64;
    println!("1. Q8K quantization: {:.2} µs", quant_us);

    // 2. Measure single row dot product (with pre-quantized activations)
    let row_data = &weights[0..bytes_per_row];
    let start = Instant::now();
    let iters2 = 100000;
    for _ in 0..iters2 {
        let _ = realizar::quantize::fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants);
    }
    let dot_ns = start.elapsed().as_nanos() as f64 / iters2 as f64;
    println!("2. Single dot product: {:.1} ns", dot_ns);

    // 3. Measure parallel matmul with pre-quantized (no reallocation)
    let mut output = vec![0.0f32; inter];
    let start = Instant::now();
    let iters3 = 100;
    for _ in 0..iters3 {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &weights, &q8k_scales, &q8k_quants, hidden, inter, &mut output
        );
    }
    let matmul_prequant_us = start.elapsed().as_micros() as f64 / iters3 as f64;
    println!("3. Parallel matmul (pre-quantized): {:.1} µs", matmul_prequant_us);

    // 4. Measure parallel matmul with auto-quantization (includes allocation)
    let start = Instant::now();
    for _ in 0..iters3 {
        let _ = realizar::quantize::fused_q4k_auto_matvec_into(
            &weights, &activations, hidden, inter, &mut output
        );
    }
    let matmul_auto_us = start.elapsed().as_micros() as f64 / iters3 as f64;
    println!("4. Parallel matmul (auto-quant): {:.1} µs", matmul_auto_us);

    // Analysis
    println!("\n--- Analysis ---");
    let sequential_estimate = dot_ns * inter as f64 / 1000.0;
    println!("Sequential estimate (dot × rows): {:.1} µs", sequential_estimate);
    println!("Parallel speedup: {:.1}x", sequential_estimate / matmul_prequant_us);
    println!("Auto-quant overhead: {:.1} µs ({:.0}%)",
        matmul_auto_us - matmul_prequant_us,
        (matmul_auto_us - matmul_prequant_us) / matmul_prequant_us * 100.0);

    // Target comparison
    let target_tok_s = 290.0;
    let matmuls_per_token = 28 * 7;
    let target_matmul_us = 1e6 / target_tok_s / matmuls_per_token as f64;
    println!("\nTarget: {:.1} µs per matmul ({:.0} tok/s)", target_matmul_us, target_tok_s);
    println!("Gap (pre-quant): {:.1}x", matmul_prequant_us / target_matmul_us);
    println!("Gap (auto-quant): {:.1}x", matmul_auto_us / target_matmul_us);

    // Theoretical analysis
    println!("\n--- Theoretical Limits ---");
    let flops_per_matmul = 2 * hidden * inter;  // mul + add
    let gflops = flops_per_matmul as f64 / matmul_prequant_us / 1000.0;
    println!("Achieved: {:.1} GFLOPS", gflops);

    // Memory bandwidth limit
    let bytes_read = weight_bytes + hidden * 4;  // weights + activations
    let bandwidth_gb_s = bytes_read as f64 / matmul_prequant_us / 1000.0;
    println!("Bandwidth used: {:.1} GB/s", bandwidth_gb_s);
    println!("DDR5 theoretical max: ~200 GB/s");

    // Thread efficiency
    let num_threads = rayon::current_num_threads();
    let thread_efficiency = sequential_estimate / matmul_prequant_us / num_threads as f64;
    println!("\nThreads: {}", num_threads);
    println!("Thread efficiency: {:.1}%", thread_efficiency * 100.0);
}

fn quantize_to_q8k(values: &[f32]) -> (Vec<f32>, Vec<i8>) {
    const QK_K: usize = 256;
    let num_sb = values.len().div_ceil(QK_K);
    let padded_len = num_sb * QK_K;

    let mut scales = Vec::with_capacity(num_sb);
    let mut quants = vec![0i8; padded_len];

    for sb in 0..num_sb {
        let start = sb * QK_K;
        let end = (start + QK_K).min(values.len());
        let chunk = &values[start..end];

        let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let inv_scale = if scale > 0.0 { 127.0 / amax } else { 0.0 };

        scales.push(scale);

        for (i, v) in chunk.iter().enumerate() {
            quants[start + i] = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (scales, quants)
}
