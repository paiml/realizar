use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let data = fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").unwrap();
    let model = GGUFModel::from_bytes(&data).unwrap();

    println!("Key tensor dimensions (after dims.reverse()):");
    for t in &model.tensors {
        if t.name == "token_embd.weight"
            || t.name == "output.weight"
            || t.name.contains("blk.0.attn_q")
            || t.name.contains("blk.0.attn_k")
            || t.name.contains("blk.0.attn_v")
            || t.name.contains("blk.0.attn_output")
            || t.name.contains("blk.0.ffn")
        {
            println!("  {}: dims={:?}", t.name, t.dims);
        }
    }
}
