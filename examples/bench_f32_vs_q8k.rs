//! PAR-126: Compare Q4K×f32 (AVX2) vs Q4K×Q8K (VNNI) paths
//! This explains the actual inference performance vs benchmarks

use std::time::Instant;

fn main() {
    println!("Q4K×f32 vs Q4K×Q8K Comparison");
    println!("==============================\n");

    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .ok();

    let hidden = 1536;
    let inter = 8960;
    let super_blocks = hidden / 256;
    let bytes_per_row = super_blocks * 144;
    let weight_bytes = inter * bytes_per_row;

    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 / hidden as f32) * 2.0 - 1.0)
        .collect();

    // Pre-quantize for Q8K path
    let (q8k_scales, q8k_quants) = quantize_to_q8k(&activations);
    let mut output = vec![0.0f32; inter];

    // Warmup both paths
    for _ in 0..10 {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &weights,
            &activations,
            hidden,
            inter,
            &mut output,
        )
        .ok();
        realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &weights,
            &q8k_scales,
            &q8k_quants,
            hidden,
            inter,
            &mut output,
        )
        .ok();
    }

    // === Q4K×f32 path (what GGUF inference uses) ===
    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &weights,
            &activations,
            hidden,
            inter,
            &mut output,
        )
        .ok();
    }
    let f32_us = start.elapsed().as_micros() as f64 / iters as f64;
    let f32_gflops = 2.0 * hidden as f64 * inter as f64 / f32_us / 1000.0;

    // === Q4K×Q8K path (what benchmarks used) ===
    let start = Instant::now();
    for _ in 0..iters {
        realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &weights,
            &q8k_scales,
            &q8k_quants,
            hidden,
            inter,
            &mut output,
        )
        .ok();
    }
    let q8k_us = start.elapsed().as_micros() as f64 / iters as f64;
    let q8k_gflops = 2.0 * hidden as f64 * inter as f64 / q8k_us / 1000.0;

    // === Q4K×f32 with auto-quant (includes Q8K quantization cost) ===
    let start = Instant::now();
    for _ in 0..iters {
        realizar::quantize::fused_q4k_auto_matvec_into(
            &weights,
            &activations,
            hidden,
            inter,
            &mut output,
        )
        .ok();
    }
    let auto_us = start.elapsed().as_micros() as f64 / iters as f64;
    let auto_gflops = 2.0 * hidden as f64 * inter as f64 / auto_us / 1000.0;

    println!("FFN up matmul ({} → {}):", hidden, inter);
    println!();
    println!(
        "Q4K×f32 (AVX2):     {:6.1} µs ({:5.1} GFLOPS)",
        f32_us, f32_gflops
    );
    println!(
        "Q4K×Q8K (VNNI):     {:6.1} µs ({:5.1} GFLOPS)",
        q8k_us, q8k_gflops
    );
    println!(
        "Q4K×auto (quant+VNNI): {:6.1} µs ({:5.1} GFLOPS)",
        auto_us, auto_gflops
    );
    println!();
    println!("Speedup VNNI/AVX2: {:.2}x", f32_us / q8k_us);
    println!("Speedup auto/AVX2: {:.2}x", f32_us / auto_us);

    // Per-token estimates
    let layers = 28;
    let matmuls_per_layer = 5;

    let f32_per_layer = f32_us * matmuls_per_layer as f64;
    let q8k_per_layer = q8k_us * matmuls_per_layer as f64;
    let auto_per_layer = auto_us * matmuls_per_layer as f64;

    let f32_per_token = f32_per_layer * layers as f64 / 1000.0; // ms
    let q8k_per_token = q8k_per_layer * layers as f64 / 1000.0;
    let auto_per_token = auto_per_layer * layers as f64 / 1000.0;

    println!("\n=== Per-Token Estimate (matmuls only) ===");
    println!(
        "Q4K×f32:   {:.1} ms = {:.0} tok/s",
        f32_per_token,
        1000.0 / f32_per_token
    );
    println!(
        "Q4K×Q8K:   {:.1} ms = {:.0} tok/s",
        q8k_per_token,
        1000.0 / q8k_per_token
    );
    println!(
        "Q4K×auto:  {:.1} ms = {:.0} tok/s",
        auto_per_token,
        1000.0 / auto_per_token
    );

    println!("\nOllama reference: ~270 tok/s");
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
