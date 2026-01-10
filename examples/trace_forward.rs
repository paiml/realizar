//! Trace forward pass to find bug
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    let bos = 151643u32;
    let hidden_dim = model.config.hidden_dim;

    // Get initial embedding
    let emb_start = bos as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding[emb_start..emb_start + hidden_dim].to_vec();

    let emb_sum: f32 = embedding.iter().sum();
    let emb_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Initial embedding (BOS):");
    println!("  sum={:.4}, norm={:.4}", emb_sum, emb_norm);
    println!("  first 8: {:?}", &embedding[..8]);

    // Check token 0's embedding for comparison
    let tok0_emb: Vec<f32> = model.token_embedding[..hidden_dim].to_vec();
    let tok0_sum: f32 = tok0_emb.iter().sum();
    let tok0_norm: f32 = tok0_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nToken 0 (\"!\") embedding:");
    println!("  sum={:.4}, norm={:.4}", tok0_sum, tok0_norm);

    // Check cosine similarity between BOS embedding and token 0 embedding
    let dot: f32 = embedding
        .iter()
        .zip(tok0_emb.iter())
        .map(|(a, b)| a * b)
        .sum();
    let cos_sim = dot / (emb_norm * tok0_norm);
    println!("\nCosine similarity(BOS, token_0) = {:.4}", cos_sim);

    // Run forward and check final hidden state before LM head
    // We can't access intermediate states easily, but we can check the LM head input
    // by computing the full forward and then looking at what the final norm produces

    // Full forward to get logits
    let logits = model.forward(&[bos])?;

    // Check logits for BOS position (only 1 token, so logits are for position 0)
    println!("\nLogits for position 0 (after BOS):");
    println!("  Token 0 (\"!\"): {:.4}", logits[0]);
    println!("  Token 19 (\"4\"): {:.4}", logits[19]);

    // Check if token 0's embedding dot product with hidden would give high logit
    // If hidden ≈ token_0_embedding * scale, then logit ≈ scale * ||token_0_emb||²
    // For tok0_norm = 0.47, that would give logit ≈ scale * 0.22
    // But we're seeing logit = 15.24, which would need scale ≈ 69!

    // Let's check what hidden state would produce such logits
    // logit[i] = hidden · lm_head_row[i]
    // For logit[0] = 15.24 and lm_head_row[0] = token_0_embedding (tied weights)
    // hidden · token_0_emb = 15.24
    // If hidden were similar to token_0_emb with some scale:
    // scale * ||token_0_emb||² = 15.24
    // scale = 15.24 / 0.22 ≈ 69

    println!("\nAnalysis:");
    println!(
        "  If logit = hidden · token_emb, and token_0 norm² = {:.4}",
        tok0_norm * tok0_norm
    );
    println!(
        "  Then for logit=15.24: scale = {:.4}",
        15.24 / (tok0_norm * tok0_norm)
    );
    println!("  This suggests hidden has large norm or is aligned with token_0");

    Ok(())
}
