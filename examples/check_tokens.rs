use realizar::gguf::{GGUFModel, MappedGGUFModel, OwnedQuantizedModel};
use std::fs;

fn main() {
    let mapped =
        MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            .unwrap();

    let prompt = "The capital of France is";
    let tokens = mapped.model.encode(prompt).unwrap();

    println!("Prompt: '{}'", prompt);
    println!("Tokens: {:?}", tokens);
    println!("Num tokens: {}", tokens.len());

    // Decode back
    let decoded = mapped.model.decode(&tokens);
    println!("Decoded: '{}'", decoded);

    // Check vocabulary
    if let Some(vocab) = mapped.model.vocabulary() {
        println!("\nVocabulary sample:");
        for (i, tok) in vocab.iter().enumerate().take(10) {
            println!("  {}: '{}'", i, tok);
        }
        println!("  ...");

        // Check specific token IDs
        for &tid in &tokens {
            if (tid as usize) < vocab.len() {
                println!("Token {}: '{}'", tid, vocab[tid as usize]);
            }
        }
    }
}
