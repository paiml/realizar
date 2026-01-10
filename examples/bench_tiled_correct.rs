//! Benchmark tiled vs non-tiled matmul with correct qtype dispatch

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K,
};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let layer = &model.layers[0];

    // Check all weight qtypes
    println!("\nWeight qtypes:");
    println!("  FFN up:   qtype={}", layer.ffn_up_weight.qtype);
    println!(
        "  FFN gate: qtype={}",
        layer.ffn_gate_weight.as_ref().map(|w| w.qtype).unwrap_or(0)
    );
    println!("  FFN down: qtype={}", layer.ffn_down_weight.qtype);

    // FFN down is Q6_K (14), let's find a Q4_K weight for testing
    // Try ffn_up which might be Q4_K
    let weight = &layer.ffn_up_weight;
    println!(
        "\nUsing FFN up weight: {}x{}, qtype={}",
        weight.out_dim, weight.in_dim, weight.qtype
    );

    if weight.qtype != GGUF_TYPE_Q4_K {
        println!(
            "Warning: FFN up is not Q4_K (qtype={}), skipping tiled test",
            weight.qtype
        );

        // Let's at least benchmark the existing Q4_K matmul if we can find one
        println!("\nSearching for Q4_K weights...");
        for (i, layer) in model.layers.iter().enumerate() {
            if layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K {
                println!("Found Q4_K at layer {} ffn_up", i);
                break;
            }
            if layer.ffn_down_weight.qtype == GGUF_TYPE_Q4_K {
                println!("Found Q4_K at layer {} ffn_down", i);
                break;
            }
        }

        // Just test with the Q6_K weights using correct kernel
        println!("\n=== Testing Q6_K FFN down ===");
        let weight = &layer.ffn_down_weight;
        let intermediate_dim = model.config().intermediate_dim;
        let activations: Vec<f32> = (0..intermediate_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let mut output = vec![0.0f32; weight.out_dim];

        // Use Q6_K kernel
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            realizar::quantize::fused_q6k_parallel_matvec_into(
                &weight.data,
                &activations,
                weight.in_dim,
                weight.out_dim,
                &mut output,
            )?;
        }
        let q6k_us = start.elapsed().as_micros() as f64 / iterations as f64;
        println!(
            "Q6_K FFN down ({}x{}): {:.1} us",
            weight.out_dim, weight.in_dim, q6k_us
        );
        println!("First 5 outputs: {:?}", &output[..5]);

        return Ok(());
    }

    // If we have Q4_K, test the tiled implementation
    let activations: Vec<f32> = (0..weight.in_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let mut output_orig = vec![0.0f32; weight.out_dim];
    let mut output_tiled = vec![0.0f32; weight.out_dim];

    let iterations = 100;

    // Benchmark original
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
    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_tiled_matvec_into(
            &weight.data,
            &activations,
            weight.in_dim,
            weight.out_dim,
            &mut output_tiled,
        )?;
    }
    let tiled_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("\n=== Q4_K Matmul Comparison ===");
    println!("Original: {:.1} us", orig_us);
    println!("Tiled:    {:.1} us", tiled_us);
    println!("Speedup:  {:.2}x", orig_us / tiled_us);

    Ok(())
}
