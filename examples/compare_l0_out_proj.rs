//! Compare CPU vs GPU output projection for layer 0

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config.hidden_dim;
    let num_heads = cpu_model.config.num_heads;
    let num_kv_heads = cpu_model.config.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;
    let eps = cpu_model.config.eps;

    let test_token: u32 = 791;
    let embedding = cpu_model.embed(&[test_token]);

    // CPU: RMSNorm + QKV projection
    let layer = &cpu_model.layers[0];
    let ss: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (ss / hidden_dim as f32 + eps).sqrt();
    let cpu_normed: Vec<f32> = embedding
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| (x / rms) * w)
        .collect();

    let qkv_cpu = cpu_model.qkv_matmul(&cpu_normed, &layer.qkv_weight)?;
    let v_cpu = &qkv_cpu[q_dim + kv_dim..];

    // At position 0, attention output = V
    // Now we need to apply output projection
    // Output projection: hidden_dim x hidden_dim (or kv_heads * head_dim -> hidden_dim for GQA)

    // The attention output needs to be reshaped for output projection
    // For GQA with num_heads=12, num_kv_heads=2:
    // - Each KV head is shared by 6 Q heads
    // - Attention output per head: head_dim
    // - Total attention output: num_heads * head_dim = hidden_dim

    // But wait, for position 0 with single token, V has kv_dim elements (not hidden_dim)
    // The attention mechanism expands KV heads to match Q heads

    eprintln!("\nV (kv_dim={}): {:?}", kv_dim, &v_cpu[..5]);

    // For GQA, need to replicate KV heads to match Q heads
    let heads_per_kv = num_heads / num_kv_heads;
    let mut attention_output = vec![0.0f32; hidden_dim];
    for q_head in 0..num_heads {
        let kv_head = q_head / heads_per_kv;
        for d in 0..head_dim {
            attention_output[q_head * head_dim + d] = v_cpu[kv_head * head_dim + d];
        }
    }

    eprintln!(
        "Expanded attention output (hidden_dim={}): first 5: {:?}",
        hidden_dim,
        &attention_output[..5]
    );

    // Apply output projection
    let out_proj = cpu_model.matmul(&attention_output, &layer.attn_output_weight)?;
    eprintln!("\n=== CPU Output Projection ===");
    eprintln!("Out proj first 5: {:?}", &out_proj[..5]);

    eprintln!("\n=== GPU Output Projection (from debug) ===");
    eprintln!("Out proj first 3: [-0.95968825, 0.011617601, -0.059631474]");

    // Compare
    let gpu_out = [-0.95968825f32, 0.011617601, -0.059631474];
    eprintln!("\n=== Comparison ===");
    for i in 0..3 {
        let diff = (out_proj[i] - gpu_out[i]).abs();
        eprintln!(
            "  [{}]: CPU={:.6}, GPU={:.6}, diff={:.6}",
            i, out_proj[i], gpu_out[i], diff
        );
    }

    // Check output projection weight shape
    eprintln!(
        "\nOutput weight len: {}",
        layer.attn_output_weight.data().len()
    );
    eprintln!(
        "Expected: {} x {} = {} bytes",
        hidden_dim,
        hidden_dim,
        hidden_dim * hidden_dim * 4 / 8
    ); // Q4K approximate

    Ok(())
}
