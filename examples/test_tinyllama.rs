//! Test TinyLlama to verify model is working
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        println!("TinyLlama model not found at {}", path);
        return Ok(());
    }

    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("Model config:");
    println!("  hidden_dim: {}", model.config.hidden_dim);
    println!("  num_layers: {}", model.config.num_layers);
    println!("  num_heads: {}", model.config.num_heads);
    println!("  num_kv_heads: {}", model.config.num_kv_heads);
    println!("  rope_theta: {}", model.config.rope_theta);
    println!("  rope_type: {}", model.config.rope_type);

    // BOS token for TinyLlama is 1
    let bos = 1u32;

    println!("\nForward with BOS token ({})...", bos);
    let logits = model.forward(&[bos])?;

    println!("Top 10 predictions:");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok).map(|s| s.as_str()).unwrap_or("?");
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
    }

    Ok(())
}
