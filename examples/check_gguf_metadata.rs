//! Check GGUF file metadata
use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let data = fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").unwrap();
    let model = GGUFModel::from_bytes(&data).unwrap();

    println!("GGUF Metadata:");
    for (key, value) in &model.metadata {
        // Print up to 100 chars of each value
        let value_str = format!("{:?}", value);
        let truncated = if value_str.len() > 100 {
            format!("{}...", &value_str[..100])
        } else {
            value_str
        };
        println!("  {}: {}", key, truncated);
    }
}
