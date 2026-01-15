//! Check if Qwen2.5 has QKV bias

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Checking QKV Bias ===");

    for (i, layer) in model.layers.iter().enumerate().take(3) {
        println!("\nLayer {}:", i);
        if let Some(ref bias) = layer.qkv_bias {
            println!("  QKV bias: len={}", bias.len());
            println!("  First 10 values: {:?}", &bias[..10.min(bias.len())]);
            println!("  Sum: {:.6}", bias.iter().sum::<f32>());
        } else {
            println!("  QKV bias: None");
        }
    }

    Ok(())
}
