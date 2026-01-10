//! Benchmark rayon dispatch overhead
//!
//! Measures the cost of rayon parallel dispatch vs single-threaded execution
//! to quantify P1 optimization opportunity (batch layer dispatch)

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.layers.len();

    println!("Model: {} layers, hidden_dim={}", num_layers, hidden_dim);

    // Create test activations
    let activations: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.017).sin()).collect();

    // Count matmuls per forward pass (for decode/single token):
    // - Per layer: QKV (1-3), attn_out (1), ffn_up (1), ffn_gate (1), ffn_down (1) = 5-7
    // - Final: lm_head (1)
    // For Qwen with separate QKV: 7 × 28 + 1 = 197 dispatches
    let matmuls_per_layer = 7; // QKV(3) + attn_out + up + gate + down
    let total_matmuls = matmuls_per_layer * num_layers + 1; // +1 for lm_head

    println!("\nEstimated matmul dispatches per token: {}", total_matmuls);

    // Benchmark rayon dispatch overhead using empty work
    let iterations = 1000;

    println!("\n=== Measuring Pure Rayon Dispatch Overhead ===");

    // Single dispatch
    let start = Instant::now();
    for _ in 0..iterations {
        let _: Vec<f32> = (0..100).into_iter().map(|i| i as f32).collect();
    }
    let serial_time = start.elapsed();

    // Using rayon
    use rayon::prelude::*;
    let start = Instant::now();
    for _ in 0..iterations {
        let _: Vec<f32> = (0..100).into_par_iter().map(|i| i as f32).collect();
    }
    let parallel_time = start.elapsed();

    let overhead_per_dispatch_us =
        (parallel_time.as_micros() as f64 - serial_time.as_micros() as f64) / iterations as f64;

    println!(
        "Serial {} iterations:   {:>8.1} µs ({:.1} µs/iter)",
        iterations,
        serial_time.as_micros() as f64,
        serial_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "Parallel {} iterations: {:>8.1} µs ({:.1} µs/iter)",
        iterations,
        parallel_time.as_micros() as f64,
        parallel_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "Rayon overhead per dispatch: {:.1} µs",
        overhead_per_dispatch_us.max(0.0)
    );

    // Benchmark with realistic matmul-sized work
    println!("\n=== Measuring Dispatch Overhead with Matmul-sized Work ===");

    let ffn_up_weight = &model.layers[0].ffn_up_weight;
    let in_dim = ffn_up_weight.in_dim;
    let out_dim = ffn_up_weight.out_dim;

    println!("FFN Up weight: {}x{}", out_dim, in_dim);

    // Warm up
    let mut output = vec![0.0f32; out_dim];
    for _ in 0..5 {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &ffn_up_weight.data,
            &activations,
            in_dim,
            out_dim,
            &mut output,
        )?;
    }

    // Single matmul timing
    let single_iterations = 100;
    let start = Instant::now();
    for _ in 0..single_iterations {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &ffn_up_weight.data,
            &activations,
            in_dim,
            out_dim,
            &mut output,
        )?;
    }
    let single_matmul_us = start.elapsed().as_micros() as f64 / single_iterations as f64;

    println!("Single matmul time: {:.1} µs", single_matmul_us);

    // Multiple dispatches (simulating forward pass)
    let dispatch_iterations = 20;
    let mut total_dispatch_time_us = 0.0;

    // Use separate input/output buffers to avoid borrow conflicts
    let mut input_buf = activations.clone();
    let mut output_buf = vec![0.0f32; out_dim.max(hidden_dim)];

    for _ in 0..dispatch_iterations {
        let start = Instant::now();

        // Simulate forward pass: many independent matmul dispatches
        for layer in &model.layers {
            // QKV
            realizar::quantize::fused_q4k_parallel_matvec_into(
                &layer.ffn_up_weight.data,
                &input_buf,
                in_dim,
                out_dim,
                &mut output_buf,
            )?;
            // attn_out - use ffn_down as proxy (similar size)
            input_buf[..hidden_dim].copy_from_slice(&output_buf[..hidden_dim]);
            realizar::quantize::fused_q4k_parallel_matvec_into(
                &layer.ffn_down_weight.data,
                &input_buf[..hidden_dim],
                hidden_dim,
                hidden_dim,
                &mut output_buf,
            )?;
            // ffn_up
            realizar::quantize::fused_q4k_parallel_matvec_into(
                &layer.ffn_up_weight.data,
                &activations,
                in_dim,
                out_dim,
                &mut output_buf,
            )?;
            // ffn_gate
            if let Some(ref gate) = layer.ffn_gate_weight {
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &gate.data,
                    &activations,
                    in_dim,
                    out_dim,
                    &mut output_buf,
                )?;
            }
            // ffn_down
            input_buf[..hidden_dim].copy_from_slice(&output_buf[..hidden_dim]);
            realizar::quantize::fused_q4k_parallel_matvec_into(
                &layer.ffn_down_weight.data,
                &input_buf[..hidden_dim],
                hidden_dim,
                hidden_dim,
                &mut output_buf,
            )?;
        }

        total_dispatch_time_us += start.elapsed().as_micros() as f64;
    }

    let avg_forward_us = total_dispatch_time_us / dispatch_iterations as f64;
    let avg_forward_ms = avg_forward_us / 1000.0;

    println!("\n=== Forward Pass Timing (Matmul Only) ===");
    println!(
        "Average forward time (matmul only): {:.1} ms",
        avg_forward_ms
    );
    println!(
        "Theoretical single-dispatch time: {:.1} ms",
        single_matmul_us * (5 * num_layers) as f64 / 1000.0
    );
    println!(
        "Dispatch overhead estimate: {:.1} ms",
        avg_forward_ms - single_matmul_us * (5 * num_layers) as f64 / 1000.0
    );

    // Test batched approach using rayon::scope
    println!("\n=== Testing Batched Dispatch (rayon::scope) ===");

    let batched_iterations = 20;
    let mut total_batched_time_us = 0.0;

    for _ in 0..batched_iterations {
        let start = Instant::now();

        // Single rayon scope for all layers
        rayon::scope(|_s| {
            for layer in &model.layers {
                // Note: These still dispatch internally, but we can measure scope overhead
                let _ = realizar::quantize::fused_q4k_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &activations,
                    in_dim,
                    out_dim,
                    &mut output_buf,
                );
                input_buf[..hidden_dim].copy_from_slice(&output_buf[..hidden_dim]);
                let _ = realizar::quantize::fused_q4k_parallel_matvec_into(
                    &layer.ffn_down_weight.data,
                    &input_buf[..hidden_dim],
                    hidden_dim,
                    hidden_dim,
                    &mut output_buf,
                );
                let _ = realizar::quantize::fused_q4k_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &activations,
                    in_dim,
                    out_dim,
                    &mut output_buf,
                );
                if let Some(ref gate) = layer.ffn_gate_weight {
                    let _ = realizar::quantize::fused_q4k_parallel_matvec_into(
                        &gate.data,
                        &activations,
                        in_dim,
                        out_dim,
                        &mut output_buf,
                    );
                }
                input_buf[..hidden_dim].copy_from_slice(&output_buf[..hidden_dim]);
                let _ = realizar::quantize::fused_q4k_parallel_matvec_into(
                    &layer.ffn_down_weight.data,
                    &input_buf[..hidden_dim],
                    hidden_dim,
                    hidden_dim,
                    &mut output_buf,
                );
            }
        });

        total_batched_time_us += start.elapsed().as_micros() as f64;
    }

    let avg_batched_us = total_batched_time_us / batched_iterations as f64;
    let avg_batched_ms = avg_batched_us / 1000.0;

    println!("Batched forward time: {:.1} ms", avg_batched_ms);
    println!("Improvement: {:.2}x", avg_forward_ms / avg_batched_ms);

    // Calculate theoretical token throughput
    println!("\n{}", "=".repeat(60));
    println!("PROJECTED PERFORMANCE");
    println!("{}", "=".repeat(60));

    let current_forward_ms = 102.0; // From benchmark_matrix results
    let dispatch_overhead_ms = avg_forward_ms - single_matmul_us * (5 * num_layers) as f64 / 1000.0;
    let projected_forward_ms = current_forward_ms - dispatch_overhead_ms;

    println!(
        "Current forward: {:.1} ms/token = {:.1} tok/s",
        current_forward_ms,
        1000.0 / current_forward_ms
    );
    println!(
        "Estimated dispatch overhead: {:.1} ms",
        dispatch_overhead_ms
    );
    println!(
        "Projected after P1 fix: {:.1} ms/token = {:.1} tok/s",
        projected_forward_ms,
        1000.0 / projected_forward_ms
    );
    println!(
        "Expected speedup: {:.2}x",
        current_forward_ms / projected_forward_ms
    );

    Ok(())
}
