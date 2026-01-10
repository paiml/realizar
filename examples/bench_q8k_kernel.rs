//! Benchmark Q4_K × Q8_K kernel vs Q4_K × f32
//!
//! Tests the new super-block aligned Q8_K format for integer matmul.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, GGUF_TYPE_Q4_K};
use realizar::quantize::{
    fused_q4k_auto_matvec_into, fused_q4k_parallel_matvec_into, fused_q4k_q8k_dot,
    fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Find a Q4_K weight
    let mut test_weight = None;
    for layer in &model.layers {
        if layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K {
            test_weight = Some(&layer.ffn_up_weight);
            break;
        }
    }

    let weight = match test_weight {
        Some(w) => w,
        None => {
            println!("No Q4_K ffn_up weights found, trying attn_output_weight...");
            let layer = &model.layers[0];
            if layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K {
                &layer.attn_output_weight
            } else {
                println!("No Q4_K weights available for benchmark, using ffn_down");
                &layer.ffn_down_weight
            }
        },
    };

    println!(
        "\nTest weight: {}×{} (qtype={})",
        weight.out_dim, weight.in_dim, weight.qtype
    );

    // Prepare test data
    let in_dim = weight.in_dim;
    let out_dim = weight.out_dim;
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Prepare Q8_K buffers
    let padded_len = in_dim.next_multiple_of(256);
    let num_superblocks = padded_len / 256;
    let mut q8k_scales = vec![0.0f32; num_superblocks];
    let mut q8k_quants = vec![0i8; padded_len];

    // Pad activations if needed
    let mut padded_activations = activations.clone();
    padded_activations.resize(padded_len, 0.0);

    // Pre-quantize activations
    quantize_activations_q8k_into(&padded_activations, &mut q8k_scales, &mut q8k_quants)?;

    let mut output_f32 = vec![0.0f32; out_dim];
    let mut output_q8k = vec![0.0f32; out_dim];
    let mut output_auto = vec![0.0f32; out_dim];

    let iterations = 100;

    // Warmup
    for _ in 0..5 {
        fused_q4k_parallel_matvec_into(
            &weight.data,
            &activations,
            in_dim,
            out_dim,
            &mut output_f32,
        )?;
    }

    // Benchmark Q4_K × f32 (current path)
    let start = Instant::now();
    for _ in 0..iterations {
        fused_q4k_parallel_matvec_into(
            &weight.data,
            &activations,
            in_dim,
            out_dim,
            &mut output_f32,
        )?;
    }
    let f32_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Warmup Q8K
    for _ in 0..5 {
        fused_q4k_q8k_parallel_matvec_into(
            &weight.data,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output_q8k,
        )?;
    }

    // Benchmark Q4_K × Q8_K (new path with pre-quantized activations)
    let start = Instant::now();
    for _ in 0..iterations {
        fused_q4k_q8k_parallel_matvec_into(
            &weight.data,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output_q8k,
        )?;
    }
    let q8k_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Benchmark auto path (quantization included)
    let start = Instant::now();
    for _ in 0..iterations {
        fused_q4k_auto_matvec_into(
            &weight.data,
            &activations,
            in_dim,
            out_dim,
            &mut output_auto,
        )?;
    }
    let auto_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Measure quantization overhead
    let start = Instant::now();
    for _ in 0..iterations {
        quantize_activations_q8k_into(&padded_activations, &mut q8k_scales, &mut q8k_quants)?;
    }
    let quant_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("\n=== Q4_K Matmul Benchmark ({}×{}) ===", out_dim, in_dim);
    println!("Q4_K × f32:           {:>8.1} µs", f32_us);
    println!(
        "Q4_K × Q8_K (pre-quant): {:>8.1} µs ({:.2}x)",
        q8k_us,
        f32_us / q8k_us
    );
    println!(
        "Q4_K × Q8_K (auto):   {:>8.1} µs ({:.2}x)",
        auto_us,
        f32_us / auto_us
    );
    println!("Q8_K quantization:    {:>8.1} µs", quant_us);

    // Verify correctness - test scalar kernel for single row
    println!("\n=== Correctness Check ===");

    // Test scalar Q8K kernel directly for first row
    let bytes_per_row = (in_dim.div_ceil(256)) * 144;
    let row0_data = &weight.data[..bytes_per_row];
    let scalar_result = fused_q4k_q8k_dot(row0_data, &q8k_scales, &q8k_quants)?;
    println!("Row 0 - f32 kernel:     {:.6}", output_f32[0]);
    println!("Row 0 - Q8K scalar:     {:.6}", scalar_result);
    println!("Row 0 - Q8K parallel:   {:.6}", output_q8k[0]);

    let max_diff = output_f32
        .iter()
        .zip(output_q8k.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let max_val = output_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel_error = if max_val > 1e-10 {
        max_diff / max_val
    } else {
        0.0
    };

    println!("\nMax absolute diff: {:.6}", max_diff);
    println!("Max relative error: {:.4}%", rel_error * 100.0);
    println!("First 5 f32 outputs:  {:?}", &output_f32[..5.min(out_dim)]);
    println!("First 5 Q8K outputs:  {:?}", &output_q8k[..5.min(out_dim)]);

    // Compute effective throughput
    let ops_per_matmul = 2 * out_dim * in_dim; // multiply-adds
    let gflops_f32 = ops_per_matmul as f64 / f32_us / 1000.0;
    let gflops_q8k = ops_per_matmul as f64 / q8k_us / 1000.0;

    println!("\n=== Effective Throughput ===");
    println!("Q4_K × f32:  {:.2} GFLOPS", gflops_f32);
    println!("Q4_K × Q8_K: {:.2} GFLOPS", gflops_q8k);

    // Memory bandwidth
    let weight_bytes = weight.data.len();
    let activation_bytes_f32 = in_dim * 4; // f32
    let activation_bytes_q8k = padded_len + num_superblocks * 4; // i8 + scales

    let bw_f32 = (weight_bytes + activation_bytes_f32) as f64 / f32_us / 1000.0; // GB/s
    let bw_q8k = (weight_bytes + activation_bytes_q8k) as f64 / q8k_us / 1000.0;

    println!("\n=== Memory Bandwidth ===");
    println!(
        "Q4_K × f32:  {:.2} GB/s ({}KB data)",
        bw_f32,
        (weight_bytes + activation_bytes_f32) / 1024
    );
    println!(
        "Q4_K × Q8_K: {:.2} GB/s ({}KB data)",
        bw_q8k,
        (weight_bytes + activation_bytes_q8k) / 1024
    );

    Ok(())
}
