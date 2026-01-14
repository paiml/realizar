//! PAR-126: Profile Q4K×Q8K matmuls through ALL 28 layers
//! Uses the same path as actual forward pass

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K};
use realizar::quantize::{fused_q4k_q8k_parallel_matvec_into, fused_q6k_parallel_matvec_into, quantize_activations_q8k_into};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config.hidden_dim;
    let intermediate_dim = model.config.intermediate_dim;
    let num_layers = model.config.num_layers;

    println!("=== PAR-126: Profile Q4K×Q8K ALL {} Layers ===\n", num_layers);
    println!("Model: hidden={}, intermediate={}", hidden_dim, intermediate_dim);

    // Pre-allocate all buffers
    let mut q_out = vec![0.0f32; hidden_dim];
    let mut k_out = vec![0.0f32; hidden_dim / 6];
    let mut v_out = vec![0.0f32; hidden_dim / 6];
    let mut attn_out_buf = vec![0.0f32; hidden_dim];
    let mut up_out = vec![0.0f32; intermediate_dim];
    let mut gate_out = vec![0.0f32; intermediate_dim];
    let mut down_out = vec![0.0f32; hidden_dim];
    
    // Create f32 inputs
    let hidden_input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 / hidden_dim as f32) * 2.0 - 1.0).collect();
    let ffn_input: Vec<f32> = (0..intermediate_dim).map(|i| (i as f32 / intermediate_dim as f32) * 2.0 - 1.0).collect();
    
    // Pre-quantize inputs to Q8K (like forward pass does)
    let hidden_padded = hidden_dim.next_multiple_of(256);
    let hidden_sb = hidden_padded / 256;
    let mut hidden_q8k_scales = vec![0.0f32; hidden_sb];
    let mut hidden_q8k_quants = vec![0i8; hidden_padded];
    quantize_activations_q8k_into(&hidden_input, &mut hidden_q8k_scales, &mut hidden_q8k_quants)?;
    
    let ffn_padded = intermediate_dim.next_multiple_of(256);
    let ffn_sb = ffn_padded / 256;
    let mut ffn_q8k_scales = vec![0.0f32; ffn_sb];
    let mut ffn_q8k_quants = vec![0i8; ffn_padded];
    quantize_activations_q8k_into(&ffn_input, &mut ffn_q8k_scales, &mut ffn_q8k_quants)?;

    // Warmup
    println!("Warming up...");
    for layer in &model.layers {
        match &layer.qkv_weight {
            realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
                if q.qtype == GGUF_TYPE_Q4_K {
                    fused_q4k_q8k_parallel_matvec_into(&q.data, &hidden_q8k_scales, &hidden_q8k_quants, q.in_dim, q.out_dim, &mut q_out)?;
                }
                if k.qtype == GGUF_TYPE_Q6_K {
                    fused_q6k_parallel_matvec_into(&k.data, &hidden_input, k.in_dim, k.out_dim, &mut k_out)?;
                }
                if v.qtype == GGUF_TYPE_Q6_K {
                    fused_q6k_parallel_matvec_into(&v.data, &hidden_input, v.in_dim, v.out_dim, &mut v_out)?;
                }
            }
            _ => {}
        }
        if layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_q8k_parallel_matvec_into(&layer.attn_output_weight.data, &hidden_q8k_scales, &hidden_q8k_quants,
                layer.attn_output_weight.in_dim, layer.attn_output_weight.out_dim, &mut attn_out_buf)?;
        }
        if layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_q8k_parallel_matvec_into(&layer.ffn_up_weight.data, &hidden_q8k_scales, &hidden_q8k_quants,
                layer.ffn_up_weight.in_dim, layer.ffn_up_weight.out_dim, &mut up_out)?;
        }
        if let Some(ref w) = layer.ffn_gate_weight {
            if w.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_q8k_parallel_matvec_into(&w.data, &hidden_q8k_scales, &hidden_q8k_quants,
                    w.in_dim, w.out_dim, &mut gate_out)?;
            }
        }
        if layer.ffn_down_weight.qtype == GGUF_TYPE_Q6_K {
            fused_q6k_parallel_matvec_into(&layer.ffn_down_weight.data, &ffn_input,
                layer.ffn_down_weight.in_dim, layer.ffn_down_weight.out_dim, &mut down_out)?;
        }
    }

    // Measure: Run through all layers multiple times
    let runs = 20;
    let mut total_us = 0.0f64;

    println!("\nMeasuring {} runs through all {} layers (Q4K×Q8K path)...", runs, num_layers);
    for _ in 0..runs {
        let start = Instant::now();

        for layer in &model.layers {
            // QKV
            match &layer.qkv_weight {
                realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
                    if q.qtype == GGUF_TYPE_Q4_K {
                        fused_q4k_q8k_parallel_matvec_into(&q.data, &hidden_q8k_scales, &hidden_q8k_quants, q.in_dim, q.out_dim, &mut q_out)?;
                    }
                    if k.qtype == GGUF_TYPE_Q6_K {
                        fused_q6k_parallel_matvec_into(&k.data, &hidden_input, k.in_dim, k.out_dim, &mut k_out)?;
                    }
                    if v.qtype == GGUF_TYPE_Q6_K {
                        fused_q6k_parallel_matvec_into(&v.data, &hidden_input, v.in_dim, v.out_dim, &mut v_out)?;
                    }
                }
                _ => {}
            }

            // Attn out
            if layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_q8k_parallel_matvec_into(&layer.attn_output_weight.data, &hidden_q8k_scales, &hidden_q8k_quants,
                    layer.attn_output_weight.in_dim, layer.attn_output_weight.out_dim, &mut attn_out_buf)?;
            }

            // FFN up, gate
            if layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_q8k_parallel_matvec_into(&layer.ffn_up_weight.data, &hidden_q8k_scales, &hidden_q8k_quants,
                    layer.ffn_up_weight.in_dim, layer.ffn_up_weight.out_dim, &mut up_out)?;
            }
            if let Some(ref w) = layer.ffn_gate_weight {
                if w.qtype == GGUF_TYPE_Q4_K {
                    fused_q4k_q8k_parallel_matvec_into(&w.data, &hidden_q8k_scales, &hidden_q8k_quants,
                        w.in_dim, w.out_dim, &mut gate_out)?;
                }
            }

            // FFN down
            if layer.ffn_down_weight.qtype == GGUF_TYPE_Q6_K {
                fused_q6k_parallel_matvec_into(&layer.ffn_down_weight.data, &ffn_input,
                    layer.ffn_down_weight.in_dim, layer.ffn_down_weight.out_dim, &mut down_out)?;
            }
        }

        total_us += start.elapsed().as_micros() as f64;
    }

    let per_forward_ms = total_us / runs as f64 / 1000.0;
    let per_layer_us = total_us / runs as f64 / num_layers as f64;

    println!("\n=== Results (V2 AVX-512 Kernel) ===");
    println!("All-layers matmuls: {:.1} ms/forward", per_forward_ms);
    println!("Per-layer average:  {:.0} µs", per_layer_us);

    // Compare to old (f32) path
    let old_matmul_ms = 35.7;  // From profile_all_layers
    println!("\n=== Comparison ===");
    println!("Old (f32 path):   {:.1} ms/forward", old_matmul_ms);
    println!("New (Q8K path):   {:.1} ms/forward", per_forward_ms);
    println!("Speedup:          {:.2}x", old_matmul_ms / per_forward_ms);

    // Ollama comparison
    let ollama_ms = 14.05;
    let estimated_total_ms = per_forward_ms + 1.2 + 0.5 + 0.2; // matmul + attention + other + q8k
    println!("\n=== vs Ollama ===");
    println!("Ollama:    {:.2} ms/tok (71.2 tok/s)", ollama_ms);
    println!("realizar:  {:.2} ms/tok ({:.1} tok/s)", estimated_total_ms, 1000.0 / estimated_total_ms);
    println!("Gap:       {:.2}x", estimated_total_ms / ollama_ms);

    // ComputeBlocks/sec calculation
    let compute_blocks_per_token = num_layers;
    let compute_blocks_per_sec = compute_blocks_per_token as f64 * 1000.0 / estimated_total_ms;
    println!("\n=== ComputeBlocks/sec ===");
    println!("Current: {:.0} CB/s", compute_blocks_per_sec);
    println!("Target (2x Ollama): {:.0} CB/s", compute_blocks_per_token as f64 * 142.3);

    Ok(())
}
