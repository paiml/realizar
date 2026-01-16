//! Benchmark tiled vs non-tiled Q4_K matmul
//!
//! Tests the L2-aware tiled implementation for FFN down (1536×8960)

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let layer = &model.layers[0];
    let intermediate_dim = model.config().intermediate_dim;
    let hidden_dim = model.config().hidden_dim;

    // FFN down: 1536x8960 (the slow one due to cache thrashing)
    let weight = &layer.ffn_down_weight;
    println!("\nFFN down weight: {}x{}", weight.out_dim, weight.in_dim);

    // Create test activations (intermediate_dim = 8960)
    let activations: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let mut output_orig = vec![0.0f32; weight.out_dim];
    let mut output_tiled = vec![0.0f32; weight.out_dim];

    // Warmup
    println!("\nWarming up...");
    for _ in 0..5 {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &weight.data,
            &activations,
            weight.in_dim,
            weight.out_dim,
            &mut output_orig,
        )?;
        realizar::quantize::fused_q4k_auto_matvec_into(
            &weight.data,
            &activations,
            weight.in_dim,
            weight.out_dim,
            &mut output_tiled,
        )?;
    }

    let iterations = 100;

    // Benchmark original
    println!("\nBenchmarking original (non-tiled)...");
    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &weight.data,
            &activations,
            weight.in_dim,
            weight.out_dim,
            &mut output_orig,
        )?;
    }
    let orig_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Benchmark tiled
    println!("Benchmarking L2-tiled...");
    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_auto_matvec_into(
            &weight.data,
            &activations,
            weight.in_dim,
            weight.out_dim,
            &mut output_tiled,
        )?;
    }
    let tiled_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Verify correctness
    let max_diff = output_orig
        .iter()
        .zip(output_tiled.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\n{}", "=".repeat(60));
    println!("RESULTS: FFN Down ({}x{})", weight.out_dim, weight.in_dim);
    println!("{}", "=".repeat(60));
    println!("Original (non-tiled): {:>8.1} us", orig_us);
    println!("L2-tiled:             {:>8.1} us", tiled_us);
    println!("Speedup:              {:>8.2}x", orig_us / tiled_us);
    println!("Max diff:             {:>8.6}", max_diff);

    // Also test FFN up for comparison (should be similar since it's already cache-friendly)
    let weight_up = &layer.ffn_up_weight;
    println!("\n{}", "=".repeat(60));
    println!("FFN Up ({}) - should be ~same speed", weight_up.out_dim);
    println!("{}", "=".repeat(60));

    let act_up: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.017).sin()).collect();
    let mut out_up_orig = vec![0.0f32; weight_up.out_dim];
    let mut out_up_tiled = vec![0.0f32; weight_up.out_dim];

    // Warmup
    for _ in 0..5 {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &weight_up.data,
            &act_up,
            weight_up.in_dim,
            weight_up.out_dim,
            &mut out_up_orig,
        )?;
        realizar::quantize::fused_q4k_auto_matvec_into(
            &weight_up.data,
            &act_up,
            weight_up.in_dim,
            weight_up.out_dim,
            &mut out_up_tiled,
        )?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &weight_up.data,
            &act_up,
            weight_up.in_dim,
            weight_up.out_dim,
            &mut out_up_orig,
        )?;
    }
    let up_orig_us = start.elapsed().as_micros() as f64 / iterations as f64;

    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_auto_matvec_into(
            &weight_up.data,
            &act_up,
            weight_up.in_dim,
            weight_up.out_dim,
            &mut out_up_tiled,
        )?;
    }
    let up_tiled_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("Original: {:>8.1} us", up_orig_us);
    println!("L2-tiled: {:>8.1} us", up_tiled_us);
    println!("Speedup:  {:>8.2}x", up_orig_us / up_tiled_us);

    // Projected improvement
    println!("\n{}", "=".repeat(60));
    println!("PROJECTED FORWARD PASS IMPROVEMENT");
    println!("{}", "=".repeat(60));

    let num_layers = model.layers.len();
    let old_ffn_down_total_ms = orig_us * num_layers as f64 / 1000.0;
    let new_ffn_down_total_ms = tiled_us * num_layers as f64 / 1000.0;
    let savings_ms = old_ffn_down_total_ms - new_ffn_down_total_ms;

    println!(
        "Old FFN down total (28 layers): {:>8.1} ms",
        old_ffn_down_total_ms
    );
    println!(
        "New FFN down total (28 layers): {:>8.1} ms",
        new_ffn_down_total_ms
    );
    println!("Savings:                        {:>8.1} ms", savings_ms);

    let old_forward_ms = 102.0; // Measured
    let new_forward_ms = old_forward_ms - savings_ms;
    println!(
        "\nOld forward pass: {:.1} ms = {:.1} tok/s",
        old_forward_ms,
        1000.0 / old_forward_ms
    );
    println!(
        "New forward pass: {:.1} ms = {:.1} tok/s",
        new_forward_ms,
        1000.0 / new_forward_ms
    );
    println!("Improvement: {:.2}x", old_forward_ms / new_forward_ms);

    Ok(())
}
