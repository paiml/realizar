//! Trace first layer to find bug
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config.hidden_dim;

    println!("=== Layer 0 Trace ===\n");

    // Get embedding for token 17 ("2")
    let tok = 17u32;
    let emb_start = tok as usize * hidden_dim;
    let emb = model.token_embedding[emb_start..emb_start + hidden_dim].to_vec();

    let emb_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Initial embedding: norm={:.4}", emb_norm);
    println!("  first 8: {:?}", &emb[..8]);

    // Check layer 0 structure
    let layer = &model.layers[0];

    println!("\nLayer 0 attn_norm_weight:");
    let norm_w_norm: f32 = layer
        .attn_norm_weight
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    println!("  norm: {:.4}", norm_w_norm);
    println!("  first 8: {:?}", &layer.attn_norm_weight[..8]);

    // Apply RMSNorm
    let sum_sq: f32 = emb.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + model.config.eps).sqrt();

    let mut normed = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        normed[i] = emb[i] * inv_rms * layer.attn_norm_weight[i];
    }

    let normed_norm: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "\nAfter attn RMSNorm: norm={:.4}, inv_rms={:.4}",
        normed_norm, inv_rms
    );
    println!("  first 8: {:?}", &normed[..8]);

    // Check QKV bias
    println!("\nQKV bias:");
    if let Some(ref bias) = layer.qkv_bias {
        println!("  length: {}", bias.len());
        println!("  Q bias sum: {:.4}", bias[0..896].iter().sum::<f32>());
        println!(
            "  K bias sum: {:.4}",
            bias[896..896 + 128].iter().sum::<f32>()
        );
        println!("  V bias sum: {:.4}", bias[896 + 128..].iter().sum::<f32>());
        println!("  Q bias first 8: {:?}", &bias[..8]);
        println!("  K bias first 8: {:?}", &bias[896..896 + 8]);
        println!("  V bias first 8: {:?}", &bias[896 + 128..896 + 128 + 8]);
    } else {
        println!("  None");
    }

    Ok(())
}
