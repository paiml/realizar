//! Quick generation test
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let mapped = MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").unwrap();
    let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();
    let vocab = mapped.model.vocabulary().unwrap();

    // Simple greedy generation
    let prompt = "Once upon a time";
    let tokens = mapped.model.encode(prompt).unwrap();
    println!("Prompt: '{}'", prompt);
    println!("Tokens: {:?}", tokens);

    let mut all_tokens = tokens.clone();
    for _ in 0..20 {
        let logits = model.forward(&all_tokens).unwrap();

        // Greedy: pick highest logit
        let (best_idx, best_logit) = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let tok_str = if best_idx < vocab.len() { &vocab[best_idx] } else { "?" };
        print!("{}", tok_str.replace("▁", " "));
        all_tokens.push(best_idx as u32);
    }
    println!("\n");

    // Show full generated sequence
    println!("Full tokens: {:?}", all_tokens);
}
