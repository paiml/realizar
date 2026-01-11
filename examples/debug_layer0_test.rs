use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let token = 13048u32; // "Hi"
    let position = 0usize;

    // Get embedding
    let hidden = model.embed(&[token]);
    println!(
        "Embedding: first 5: {:?}, sum: {:.4}",
        &hidden[..5],
        hidden.iter().sum::<f32>()
    );

    // Get layer 0
    let layer = &model.model.layers[0];

    // RMSNorm before attention
    let normed = model.rms_norm(&hidden, &layer.attn_norm_weight, model.config.eps);
    println!(
        "After attn_norm: first 5: {:?}, sum: {:.4}",
        &normed[..5],
        normed.iter().sum::<f32>()
    );

    // QKV projection
    let qkv = model.qkv_matmul(&normed, &layer.qkv_weight)?;
    println!("QKV len: {}", qkv.len());

    let hidden_dim = model.config.hidden_dim;
    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = hidden_dim / model.config.num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let q = &qkv[0..hidden_dim];
    let k = &qkv[hidden_dim..hidden_dim + kv_dim];
    let v = &qkv[hidden_dim + kv_dim..];

    println!(
        "Q: first 5: {:?}, sum: {:.4}",
        &q[..5],
        q.iter().sum::<f32>()
    );
    println!(
        "K: first 5: {:?}, sum: {:.4}",
        &k[..5],
        k.iter().sum::<f32>()
    );
    println!(
        "V: first 5: {:?}, sum: {:.4}",
        &v[..5],
        v.iter().sum::<f32>()
    );

    // Apply RoPE
    let mut q_rope = q.to_vec();
    let mut k_rope = k.to_vec();
    model.apply_rope(&mut q_rope, position, model.config.num_heads);
    model.apply_rope(&mut k_rope, position, num_kv_heads);

    println!(
        "Q after RoPE (pos={}): first 5: {:?}, sum: {:.4}",
        position,
        &q_rope[..5],
        q_rope.iter().sum::<f32>()
    );
    println!(
        "K after RoPE (pos={}): first 5: {:?}, sum: {:.4}",
        position,
        &k_rope[..5],
        k_rope.iter().sum::<f32>()
    );

    // For position 0, attention output is just V expanded to all heads
    let q_per_kv = model.config.num_heads / num_kv_heads;
    let mut attn_out = vec![0.0f32; hidden_dim];
    for q_head in 0..model.config.num_heads {
        let kv_head = q_head / q_per_kv;
        let v_start = kv_head * head_dim;
        let out_start = q_head * head_dim;
        attn_out[out_start..out_start + head_dim].copy_from_slice(&v[v_start..v_start + head_dim]);
    }
    println!(
        "Attn out (V expanded): first 5: {:?}, sum: {:.4}",
        &attn_out[..5],
        attn_out.iter().sum::<f32>()
    );

    // Attention output projection
    let attn_proj = model.fused_matmul(&attn_out, &layer.attn_output_weight)?;
    println!(
        "Attn proj: first 5: {:?}, sum: {:.4}",
        &attn_proj[..5],
        attn_proj.iter().sum::<f32>()
    );

    // Residual
    let mut hidden_after_attn = hidden.clone();
    for i in 0..hidden_dim {
        hidden_after_attn[i] += attn_proj[i];
    }
    println!(
        "After attn residual: first 5: {:?}, sum: {:.4}",
        &hidden_after_attn[..5],
        hidden_after_attn.iter().sum::<f32>()
    );

    // FFN norm
    let ffn_normed = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
        model.rms_norm(&hidden_after_attn, ffn_norm, model.config.eps)
    } else {
        hidden_after_attn.clone()
    };
    println!(
        "FFN normed: first 5: {:?}, sum: {:.4}",
        &ffn_normed[..5],
        ffn_normed.iter().sum::<f32>()
    );

    // SwiGLU FFN
    let mut ffn_up = model.fused_matmul(&ffn_normed, &layer.ffn_up_weight)?;
    if let Some(ref gate_weight) = layer.ffn_gate_weight {
        let mut ffn_gate = model.fused_matmul(&ffn_normed, gate_weight)?;
        // SiLU on gate
        for x in ffn_gate.iter_mut() {
            *x = *x * (1.0 / (1.0 + (-*x).exp())); // SiLU = x * sigmoid(x)
        }
        // Element-wise multiply
        for i in 0..ffn_gate.len() {
            ffn_gate[i] *= ffn_up[i];
        }
        ffn_up = ffn_gate;
    }
    println!(
        "FFN activated: first 5: {:?}, sum: {:.4}",
        &ffn_up[..5],
        ffn_up.iter().sum::<f32>()
    );

    // FFN down
    let ffn_down = model.fused_matmul(&ffn_up, &layer.ffn_down_weight)?;
    println!(
        "FFN down: first 5: {:?}, sum: {:.4}",
        &ffn_down[..5],
        ffn_down.iter().sum::<f32>()
    );

    // Final residual
    let mut hidden_after_ffn = hidden_after_attn.clone();
    for i in 0..hidden_dim {
        hidden_after_ffn[i] += ffn_down[i];
    }
    println!(
        "After FFN residual: first 5: {:?}, sum: {:.4}",
        &hidden_after_ffn[..5],
        hidden_after_ffn.iter().sum::<f32>()
    );

    Ok(())
}
