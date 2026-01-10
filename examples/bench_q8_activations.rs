//! Benchmark Q8_0 activation quantization speedup
//!
//! Compares f32 activations vs Q8_0 pre-quantized activations for Q4_K matmul

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{
    fused_q4k_parallel_matvec_into, fused_q4k_q8_parallel_matvec_into, quantize_to_q8_blocks_fast,
};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;

    println!(
        "Model: hidden_dim={}, intermediate_dim={}",
        hidden_dim, intermediate_dim
    );

    // Create test activations
    let activations: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.017).sin()).collect();

    // Pre-quantize activations to Q8_0
    let q8_blocks = quantize_to_q8_blocks_fast(&activations);
    println!(
        "Q8 blocks: {} (expected {})",
        q8_blocks.len(),
        hidden_dim / 32
    );

    // Get first layer FFN up weight for testing
    let layer = &model.layers[0];
    let ffn_up_weight = &layer.ffn_up_weight;

    println!(
        "FFN Up weight: {}x{} (Q4_K)",
        ffn_up_weight.out_dim, ffn_up_weight.in_dim
    );

    let in_dim = ffn_up_weight.in_dim;
    let out_dim = ffn_up_weight.out_dim;

    // Pre-allocate output
    let mut output_f32 = vec![0.0f32; out_dim];
    let mut output_q8 = vec![0.0f32; out_dim];

    // Warmup
    println!("\nWarming up...");
    for _ in 0..10 {
        fused_q4k_parallel_matvec_into(
            &ffn_up_weight.data,
            &activations,
            in_dim,
            out_dim,
            &mut output_f32,
        )?;
        fused_q4k_q8_parallel_matvec_into(
            &ffn_up_weight.data,
            &q8_blocks,
            in_dim,
            out_dim,
            &mut output_q8,
        )?;
    }

    let iterations = 100;

    // Benchmark f32 activations
    println!(
        "\nBenchmarking f32 activations ({} iterations)...",
        iterations
    );
    let start = Instant::now();
    for _ in 0..iterations {
        fused_q4k_parallel_matvec_into(
            &ffn_up_weight.data,
            &activations,
            in_dim,
            out_dim,
            &mut output_f32,
        )?;
    }
    let f32_time = start.elapsed();
    let f32_us = f32_time.as_micros() as f64 / iterations as f64;

    // Benchmark Q8_0 activations
    println!(
        "Benchmarking Q8_0 activations ({} iterations)...",
        iterations
    );
    let start = Instant::now();
    for _ in 0..iterations {
        fused_q4k_q8_parallel_matvec_into(
            &ffn_up_weight.data,
            &q8_blocks,
            in_dim,
            out_dim,
            &mut output_q8,
        )?;
    }
    let q8_time = start.elapsed();
    let q8_us = q8_time.as_micros() as f64 / iterations as f64;

    // Include quantization time
    let start = Instant::now();
    for _ in 0..iterations {
        let q8_blocks = quantize_to_q8_blocks_fast(&activations);
        fused_q4k_q8_parallel_matvec_into(
            &ffn_up_weight.data,
            &q8_blocks,
            in_dim,
            out_dim,
            &mut output_q8,
        )?;
    }
    let q8_with_quant_time = start.elapsed();
    let q8_with_quant_us = q8_with_quant_time.as_micros() as f64 / iterations as f64;

    // Results
    println!("\n{}", "=".repeat(60));
    println!("RESULTS: Q4_K × Activation Matmul Performance");
    println!("{}", "=".repeat(60));
    println!("Matrix size: {}x{}", out_dim, in_dim);
    println!();
    println!("f32 activations:           {:>8.1} µs per matmul", f32_us);
    println!("Q8_0 activations (pre):    {:>8.1} µs per matmul", q8_us);
    println!(
        "Q8_0 + quantization:       {:>8.1} µs per matmul",
        q8_with_quant_us
    );
    println!();
    println!("Speedup (Q8 pre-quant):    {:.2}x", f32_us / q8_us);
    println!(
        "Speedup (Q8 with quant):   {:.2}x",
        f32_us / q8_with_quant_us
    );

    // Calculate theoretical token throughput
    let matmuls_per_token = 155;
    let f32_ms_per_token = f32_us * matmuls_per_token as f64 / 1000.0;
    let q8_ms_per_token = q8_us * matmuls_per_token as f64 / 1000.0;

    println!();
    println!(
        "Theoretical Token Performance (matmul only, {} matmuls/token):",
        matmuls_per_token
    );
    println!(
        "f32: {:.1} ms/token = {:.1} tok/s",
        f32_ms_per_token,
        1000.0 / f32_ms_per_token
    );
    println!(
        "Q8:  {:.1} ms/token = {:.1} tok/s",
        q8_ms_per_token,
        1000.0 / q8_ms_per_token
    );

    // Verify correctness
    let max_diff = output_f32
        .iter()
        .zip(output_q8.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let avg_diff = output_f32
        .iter()
        .zip(output_q8.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / out_dim as f32;

    println!();
    println!("Correctness Check:");
    println!("Max diff:  {:.6}", max_diff);
    println!("Avg diff:  {:.6}", avg_diff);

    if max_diff > 1.0 {
        println!("⚠️  WARNING: Large difference detected - may need quantization tuning");
    } else {
        println!("✅ Outputs match within acceptable tolerance");
    }

    Ok(())
}
