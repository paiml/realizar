//! Benchmark forward pass timing
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::time::Instant;

fn main() {
    println!("Loading model...");
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").unwrap();
    let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    let config = model.config();
    println!("Config: hidden_dim={}, vocab_size={}, num_layers={}",
             config.hidden_dim, config.vocab_size, config.num_layers);

    // Test with single token
    println!("\nTesting single-token forward pass...");
    let start = Instant::now();
    let logits = model.forward(&[1]).unwrap();
    println!("Single token forward: {:.3}s", start.elapsed().as_secs_f32());
    println!("Logits len: {}, first 5: {:?}", logits.len(), &logits[..5]);

    // Find top prediction
    let (best_idx, best_logit) = logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    let vocab = mapped.model.vocabulary().unwrap();
    let tok = if best_idx < vocab.len() { &vocab[best_idx] } else { "?" };
    println!("Top prediction: {} '{}' = {:.4}", best_idx, tok, best_logit);
}
