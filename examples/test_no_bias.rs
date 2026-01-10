//! Test Qwen2 without biases to isolate the issue
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let mut model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Test Without Biases ===\n");

    // Remove all QKV biases
    for layer in &mut model.layers {
        layer.qkv_bias = None;
    }

    println!("Removed all QKV biases.\n");

    // Test with 2+2=
    let tokens = vec![17, 10, 17, 28]; // "2+2="
    let logits = model.forward(&tokens)?;

    println!("Input: 2+2= (tokens: {:?})", tokens);
    println!("\nTop 10 predictions (WITHOUT biases):");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok).map(|s| s.as_str()).unwrap_or("?");
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
    }

    println!("\nSpecific tokens:");
    println!("  Token 19 (\"4\"): logit={:.4}", logits[19]);
    println!("  Token 0 (\"!\"): logit={:.4}", logits[0]);

    Ok(())
}
