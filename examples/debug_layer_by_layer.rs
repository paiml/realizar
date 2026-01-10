//! Debug transformer layer by layer
//!
//! Trace the hidden state through each layer to find where it diverges
//! from expected values.
//!
//! Run: cd /home/noah/src/realizar && cargo run --release --example debug_layer_by_layer

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Layer-by-Layer Debug ===\n");

    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    let hidden_dim = model.config.hidden_dim;
    let num_layers = model.config.num_layers;

    println!("Model: {} layers, {} hidden dim", num_layers, hidden_dim);
    println!(
        "  num_heads: {}, num_kv_heads: {}",
        model.config.num_heads, model.config.num_kv_heads
    );
    println!("  RoPE theta: {}", model.config.rope_theta);

    // Test with BOS token
    let bos_token = 151643u32;
    println!("\nTesting with BOS token: {}", bos_token);

    // Create KV cache
    let mut cache = OwnedQuantizedKVCache::new(num_layers, hidden_dim, model.config.num_kv_heads);

    // Get embedding
    let emb_start = bos_token as usize * hidden_dim;
    let hidden: Vec<f32> = model.token_embedding[emb_start..emb_start + hidden_dim].to_vec();

    println!("\nInitial embedding:");
    println!("  sum: {:.4}", hidden.iter().sum::<f32>());
    println!(
        "  norm: {:.4}",
        hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!("  first 8: {:?}", &hidden[..8]);

    // Process through layers manually
    let mut hidden = hidden;

    for layer_idx in 0..num_layers {
        let layer = &model.layers[layer_idx];

        // RMSNorm
        let normed = model.rms_norm(&hidden, &layer.attn_norm_weight, model.config.eps);

        if layer_idx == 0 || layer_idx == 12 || layer_idx == 23 {
            println!("\nLayer {} - after attention norm:", layer_idx);
            println!("  normed sum: {:.4}", normed.iter().sum::<f32>());
            println!(
                "  normed norm: {:.4}",
                normed.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
        }

        // QKV projection
        let q_dim = layer.qkv_weight.q_dim();
        let k_dim = match &layer.qkv_weight {
            realizar::gguf::OwnedQKVWeights::Fused(_) => q_dim,
            realizar::gguf::OwnedQKVWeights::Separate { k, .. } => k.out_dim,
        };
        let v_dim = match &layer.qkv_weight {
            realizar::gguf::OwnedQKVWeights::Fused(_) => q_dim,
            realizar::gguf::OwnedQKVWeights::Separate { v, .. } => v.out_dim,
        };

        let mut qkv = model.qkv_matmul(&normed, &layer.qkv_weight)?;
        if let Some(ref bias) = layer.qkv_bias {
            model.add_bias(&mut qkv, bias);
        }

        // Extract Q, K, V
        let mut q = qkv[0..q_dim].to_vec();
        let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
        let v = qkv[q_dim + k_dim..q_dim + k_dim + v_dim].to_vec();

        if layer_idx == 0 {
            println!("\nLayer 0 - Q/K/V before RoPE:");
            println!(
                "  Q sum: {:.4}, first 4: {:?}",
                q.iter().sum::<f32>(),
                &q[..4]
            );
            println!(
                "  K sum: {:.4}, first 4: {:?}",
                k.iter().sum::<f32>(),
                &k[..4]
            );
            println!(
                "  V sum: {:.4}, first 4: {:?}",
                v.iter().sum::<f32>(),
                &v[..4]
            );
        }

        // Apply RoPE at position 0
        model.apply_rope(&mut q, 0, model.config.num_heads);
        model.apply_rope(&mut k, 0, model.config.num_kv_heads);

        if layer_idx == 0 {
            println!("\nLayer 0 - Q/K after RoPE (position 0):");
            println!(
                "  Q sum: {:.4}, first 4: {:?}",
                q.iter().sum::<f32>(),
                &q[..4]
            );
            println!(
                "  K sum: {:.4}, first 4: {:?}",
                k.iter().sum::<f32>(),
                &k[..4]
            );
        }

        // Attention: for position 0, it's just V (expanded for GQA)
        let k_cache = cache.get_k(layer_idx);

        let attn_out = if k_cache.is_empty() {
            // First token - just use V (expanded for GQA)
            if model.config.num_kv_heads < model.config.num_heads {
                let head_dim = hidden_dim / model.config.num_heads;
                let group_size = model.config.num_heads / model.config.num_kv_heads;
                (0..model.config.num_heads)
                    .flat_map(|h| {
                        let kv_head = h / group_size;
                        let start = kv_head * head_dim;
                        v[start..start + head_dim].iter().copied()
                    })
                    .collect::<Vec<_>>()
            } else {
                v.clone()
            }
        } else {
            panic!("KV cache should be empty for position 0");
        };

        // Store in cache
        cache.append(layer_idx, &k, &v);

        if layer_idx == 0 {
            println!("\nLayer 0 - attention output (V expanded for GQA):");
            println!(
                "  attn_out len: {} (expected: {})",
                attn_out.len(),
                hidden_dim
            );
            println!("  attn_out sum: {:.4}", attn_out.iter().sum::<f32>());
            println!("  attn_out first 8: {:?}", &attn_out[..8]);
        }

        // Output projection
        let mut attn_output = model.fused_matmul(&attn_out, &layer.attn_output_weight)?;
        if let Some(ref bias) = layer.attn_output_bias {
            model.add_bias(&mut attn_output, bias);
        }

        if layer_idx == 0 {
            println!("\nLayer 0 - after attn_output projection:");
            println!("  attn_output sum: {:.4}", attn_output.iter().sum::<f32>());
            println!("  attn_output first 8: {:?}", &attn_output[..8]);
        }

        // First residual
        for i in 0..hidden_dim {
            hidden[i] += attn_output[i];
        }

        if layer_idx == 0 {
            println!("\nLayer 0 - after first residual:");
            println!("  hidden sum: {:.4}", hidden.iter().sum::<f32>());
        }

        // Pre-FFN norm
        let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
            model.rms_norm(&hidden, ffn_norm, model.config.eps)
        } else {
            hidden.clone()
        };

        // SwiGLU FFN
        let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
            let mut ffn_up = model.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                model.add_bias(&mut ffn_up, bias);
            }

            let mut ffn_gate = model.fused_matmul(&ffn_input, gate_weight)?;
            if let Some(ref bias) = layer.ffn_gate_bias {
                model.add_bias(&mut ffn_gate, bias);
            }

            // SiLU on gate
            for x in ffn_gate.iter_mut() {
                *x = *x * (1.0 / (1.0 + (-*x).exp()));
            }

            // Gate * up
            for i in 0..ffn_gate.len() {
                ffn_gate[i] *= ffn_up[i];
            }

            let mut output = model.fused_matmul(&ffn_gate, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                model.add_bias(&mut output, bias);
            }
            output
        } else {
            // GELU path
            let mut ffn_hidden = model.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                model.add_bias(&mut ffn_hidden, bias);
            }
            // GELU
            for x in ffn_hidden.iter_mut() {
                let cdf = 0.5 * (1.0 + (*x / 1.414213562373095).tanh());
                *x = *x * cdf;
            }
            let mut output = model.fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                model.add_bias(&mut output, bias);
            }
            output
        };

        // Second residual
        for i in 0..hidden_dim {
            hidden[i] += ffn_output[i];
        }

        if layer_idx == 0 || layer_idx == 12 || layer_idx == 23 {
            println!("\nLayer {} - after second residual:", layer_idx);
            println!("  hidden sum: {:.4}", hidden.iter().sum::<f32>());
            println!(
                "  hidden norm: {:.4}",
                hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
            println!("  hidden first 8: {:?}", &hidden[..8]);
        }
    }

    // Final norm
    let normed = model.rms_norm(&hidden, &model.output_norm_weight, model.config.eps);
    println!("\nFinal output norm:");
    println!("  normed sum: {:.4}", normed.iter().sum::<f32>());
    println!("  normed first 8: {:?}", &normed[..8]);

    // LM head
    let logits = model.fused_matmul(&normed, &model.lm_head_weight)?;

    println!("\nLogits:");
    println!("  len: {}", logits.len());
    println!("  logits[0] (\"!\"): {:.4}", logits[0]);
    println!("  logits[19] (\"4\"): {:.4}", logits[19]);

    // Find argmax
    let (argmax_idx, argmax_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let argmax_str = vocab.get(argmax_idx).map(|s| s.as_str()).unwrap_or("?");
    println!(
        "  Argmax: token {} ({:?}) with logit {:.4}",
        argmax_idx, argmax_str, argmax_val
    );

    // Compare with model.forward_cached
    println!("\n=== Compare with forward_cached ===");
    let mut cache2 = OwnedQuantizedKVCache::new(num_layers, hidden_dim, model.config.num_kv_heads);
    let logits2 = model.forward_cached(bos_token, &mut cache2, 0)?;
    println!("forward_cached logits[0]: {:.4}", logits2[0]);
    println!("forward_cached logits[19]: {:.4}", logits2[19]);

    Ok(())
}
