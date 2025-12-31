//! Quick generation test
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf");
    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();
    let vocab = mapped.model.vocabulary().unwrap();

    // Simple greedy generation
    let prompt = "Once upon a time";
    let tokens = mapped.model.encode(prompt).unwrap();
    println!("Prompt: '{}'", prompt);
    println!("Tokens: {:?}", tokens);

    let mut all_tokens = tokens;
    for _ in 0..20 {
        let logits = model.forward(&all_tokens).unwrap();

        // Greedy: pick highest logit
        let (best_idx, _best_logit) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let tok_str = if best_idx < vocab.len() {
            &vocab[best_idx]
        } else {
            "?"
        };
        // Handle both GPT-2 style (Ġ) and SentencePiece style (▁) space tokens
        print!("{}", tok_str.replace("▁", " ").replace('\u{0120}', " "));
        all_tokens.push(best_idx as u32);
    }
    println!("\n");

    // Show full generated sequence
    println!("Full tokens: {:?}", all_tokens);
}
