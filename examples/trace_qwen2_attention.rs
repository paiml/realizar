//! Trace Qwen2 attention patterns for multi-token sequence
//! Check if attention weights look reasonable
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Qwen2 Attention Pattern Trace ===\n");
    println!("Model config:");
    println!("  hidden_dim: {}", model.config.hidden_dim);
    println!("  num_heads: {}", model.config.num_heads);
    println!("  num_kv_heads: {}", model.config.num_kv_heads);
    println!(
        "  head_dim: {}",
        model.config.hidden_dim / model.config.num_heads
    );
    println!("  rope_type: {}", model.config.rope_type);

    let head_dim = model.config.hidden_dim / model.config.num_heads;
    let num_heads = model.config.num_heads;
    let num_kv_heads = model.config.num_kv_heads;
    let group_size = num_heads / num_kv_heads;

    println!("  group_size (Q heads per KV head): {}", group_size);

    // Tokens: 17 (+), 10 (2), 17 (+), 28 (=) - but we want "2+2="
    // Actually from previous traces: 17="+", 10="2", so "2+2=" is [10, 17, 10, 28]
    // Wait, let me check the vocab again
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("\nVocab check:");
    for tok in [10, 17, 19, 28] {
        let s = vocab.get(tok).map(|s| s.as_str()).unwrap_or("?");
        println!("  Token {}: {:?}", tok, s);
    }

    // Based on previous traces:
    // Token 17 = "2", Token 10 = "+", Token 28 = "="
    // So "2+2=" = [17, 10, 17, 28]
    let tokens = vec![17u32, 10, 17, 28]; // "2+2="
    let seq_len = tokens.len();

    println!("\nInput sequence: {:?}", tokens);
    println!("Sequence meaning: 2 + 2 =\n");

    // Get embeddings for all tokens
    let hidden_dim = model.config.hidden_dim;
    let mut hidden: Vec<f32> = Vec::with_capacity(seq_len * hidden_dim);
    for &tok in &tokens {
        let start = tok as usize * hidden_dim;
        let end = start + hidden_dim;
        hidden.extend_from_slice(&model.token_embedding[start..end]);
    }

    println!("Initial hidden norms per position:");
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let norm: f32 = hidden[start..start + hidden_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!("  Position {} (token {}): {:.4}", pos, tokens[pos], norm);
    }

    // Apply layer 0 attention manually to trace
    let layer = &model.layers[0];

    // RMSNorm on hidden
    let eps = model.config.rms_norm_eps;
    let mut normed = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        let ss: f32 = hidden[start..end].iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        for i in 0..hidden_dim {
            normed[start + i] = hidden[start + i] * scale * layer.attn_norm_weight[i];
        }
    }

    println!("\nAfter RMSNorm (layer 0) norms per position:");
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let norm: f32 = normed[start..start + hidden_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!("  Position {}: {:.4}", pos, norm);
    }

    // Compute Q, K, V for all positions
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let mut q_all = vec![0.0f32; seq_len * q_dim];
    let mut k_all = vec![0.0f32; seq_len * kv_dim];
    let mut v_all = vec![0.0f32; seq_len * kv_dim];

    // Q projection
    for pos in 0..seq_len {
        let in_start = pos * hidden_dim;
        let out_start = pos * q_dim;
        layer.attn_q_weight.matvec(
            &normed[in_start..in_start + hidden_dim],
            &mut q_all[out_start..out_start + q_dim],
        );
        // Add bias
        if let Some(ref bias) = layer.attn_q_bias {
            for i in 0..q_dim {
                q_all[out_start + i] += bias[i];
            }
        }
    }

    // K projection
    for pos in 0..seq_len {
        let in_start = pos * hidden_dim;
        let out_start = pos * kv_dim;
        layer.attn_k_weight.matvec(
            &normed[in_start..in_start + hidden_dim],
            &mut k_all[out_start..out_start + kv_dim],
        );
        // Add bias
        if let Some(ref bias) = layer.attn_k_bias {
            for i in 0..kv_dim {
                k_all[out_start + i] += bias[i];
            }
        }
    }

    // V projection
    for pos in 0..seq_len {
        let in_start = pos * hidden_dim;
        let out_start = pos * kv_dim;
        layer.attn_v_weight.matvec(
            &normed[in_start..in_start + hidden_dim],
            &mut v_all[out_start..out_start + kv_dim],
        );
        // Add bias
        if let Some(ref bias) = layer.attn_v_bias {
            for i in 0..kv_dim {
                v_all[out_start + i] += bias[i];
            }
        }
    }

    println!("\nQKV norms per position (before RoPE):");
    for pos in 0..seq_len {
        let q_norm: f32 = q_all[pos * q_dim..(pos + 1) * q_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let k_norm: f32 = k_all[pos * kv_dim..(pos + 1) * kv_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let v_norm: f32 = v_all[pos * kv_dim..(pos + 1) * kv_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!(
            "  Position {}: Q={:.2}, K={:.2}, V={:.4}",
            pos, q_norm, k_norm, v_norm
        );
    }

    // Apply RoPE (NEOX style, rope_type=2)
    let rope_theta = model.config.rope_theta;
    println!("\nApplying RoPE (NEOX style, theta={})...", rope_theta);

    // NEOX RoPE: pairs are (x[i], x[i+half_dim])
    let half_dim = head_dim / 2;

    // Apply RoPE to Q
    for pos in 0..seq_len {
        for h in 0..num_heads {
            let base = pos * q_dim + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_v = angle.cos();
                let sin_v = angle.sin();

                let x1 = q_all[base + i];
                let x2 = q_all[base + half_dim + i];
                q_all[base + i] = x1 * cos_v - x2 * sin_v;
                q_all[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
            }
        }
    }

    // Apply RoPE to K
    for pos in 0..seq_len {
        for h in 0..num_kv_heads {
            let base = pos * kv_dim + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_v = angle.cos();
                let sin_v = angle.sin();

                let x1 = k_all[base + i];
                let x2 = k_all[base + half_dim + i];
                k_all[base + i] = x1 * cos_v - x2 * sin_v;
                k_all[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
            }
        }
    }

    println!("\nQKV norms per position (after RoPE):");
    for pos in 0..seq_len {
        let q_norm: f32 = q_all[pos * q_dim..(pos + 1) * q_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let k_norm: f32 = k_all[pos * kv_dim..(pos + 1) * kv_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let v_norm: f32 = v_all[pos * kv_dim..(pos + 1) * kv_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!(
            "  Position {}: Q={:.2}, K={:.2}, V={:.4}",
            pos, q_norm, k_norm, v_norm
        );
    }

    // Now compute attention scores for position 3 (the "=" token)
    // This is what determines the next token prediction
    let query_pos = 3; // Position of "="
    let scale = 1.0 / (head_dim as f32).sqrt();

    println!(
        "\n=== Attention Analysis for Position {} (token '=') ===",
        query_pos
    );
    println!("Scale factor: {:.6}", scale);

    // Compute attention for each Q head
    for head in 0..num_heads.min(3) {
        // Just first 3 heads
        let kv_head = head / group_size;
        let q_offset = query_pos * q_dim + head * head_dim;

        println!("\nHead {} (maps to KV head {}):", head, kv_head);

        // Compute attention scores against all K positions (causal: 0..=query_pos)
        let mut scores = vec![0.0f32; query_pos + 1];
        for key_pos in 0..=query_pos {
            let k_offset = key_pos * kv_dim + kv_head * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_all[q_offset + d] * k_all[k_offset + d];
            }
            scores[key_pos] = dot * scale;
        }

        println!("  Raw attention scores: {:?}", scores);

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        println!("  Softmax weights: {:?}", weights);
        println!(
            "  Attending to: pos0={:.1}%, pos1={:.1}%, pos2={:.1}%, pos3={:.1}%",
            weights[0] * 100.0,
            weights[1] * 100.0,
            weights[2] * 100.0,
            weights[3] * 100.0
        );
    }

    // Compare with TinyLlama's attention pattern
    println!("\n=== For Comparison ===");
    println!("A well-behaved model for '2+2=' should:");
    println!("  - Have reasonable attention spread across positions");
    println!("  - Not have extreme attention to any single position");
    println!("  - Have attention weights that sum to 1.0");

    // Check if Q or K have extreme values that could cause issues
    println!("\n=== Extreme Value Check ===");
    let q_min = q_all.iter().cloned().fold(f32::INFINITY, f32::min);
    let q_max = q_all.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let k_min = k_all.iter().cloned().fold(f32::INFINITY, f32::min);
    let k_max = k_all.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("Q range: [{:.2}, {:.2}]", q_min, q_max);
    println!("K range: [{:.2}, {:.2}]", k_min, k_max);

    // Check for potential overflow in attention computation
    let max_dot = q_max.abs() * k_max.abs() * head_dim as f32 * scale;
    println!("\nPotential max attention score magnitude: {:.2}", max_dot);
    println!("(If this is >100, softmax may saturate to 0 or 1)");

    Ok(())
}
