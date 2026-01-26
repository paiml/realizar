//! Check if biases exist in both models
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bias Check ===\n");

    for (name, path) in [
        ("Qwen2-0.5B", "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf"),
        ("Qwen2.5-Coder-1.5B", "/home/noah/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    ] {
        let mapped = MappedGGUFModel::from_path(path)?;
        let model = OwnedQuantizedModel::from_mapped(&mapped)?;
        
        println!("=== {} ===", name);
        let layer0 = &model.layers[0];
        println!("  qkv_bias: {:?}", layer0.qkv_bias.as_ref().map(|b| b.len()));
        println!("  attn_norm_bias: {:?}", layer0.attn_norm_bias.as_ref().map(|b| b.len()));
        println!("  attn_output_bias: {:?}", layer0.attn_output_bias.as_ref().map(|b| b.len()));
        println!("  ffn_up_bias: {:?}", layer0.ffn_up_bias.as_ref().map(|b| b.len()));
        println!("  ffn_down_bias: {:?}", layer0.ffn_down_bias.as_ref().map(|b| b.len()));
        println!("  ffn_gate_bias: {:?}", layer0.ffn_gate_bias.as_ref().map(|b| b.len()));
        println!("  ffn_norm_bias: {:?}", layer0.ffn_norm_bias.as_ref().map(|b| b.len()));
        
        println!("  output_norm_bias: {:?}", model.output_norm_bias.as_ref().map(|b| b.len()));
        println!("  lm_head_bias: {:?}", model.lm_head_bias.as_ref().map(|b| b.len()));
        
        // Check first few values of qkv_bias if present
        if let Some(ref bias) = layer0.qkv_bias {
            println!("  qkv_bias first 5: {:?}", &bias[..5.min(bias.len())]);
        }
        println!();
    }

    Ok(())
}
