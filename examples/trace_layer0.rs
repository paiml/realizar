//! Trace layer 0 in detail on CPU to compare with GPU

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
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
    eprintln!("Embedding first 5: {:?}", &embedding[..5]);

    let layer = &cpu_model.layers[0];

    // RMSNorm
    let ss: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (ss / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = embedding
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| (x / rms) * w)
        .collect();
    eprintln!("\n[CPU-L0] RMSNorm first 3: {:?}", &normed[..3]);

    // QKV projection
    let qkv = cpu_model.qkv_matmul(&normed, &layer.qkv_weight)?;
    let v = &qkv[q_dim + kv_dim..];
    eprintln!("[CPU-L0] V first 5: {:?}", &v[..5]);

    // At position 0, attention output = V (expanded for GQA)
    let q_per_kv = num_heads / num_kv_heads;
    let mut attn_out = vec![0.0f32; hidden_dim];
    for q_head in 0..num_heads {
        let kv_head = q_head / q_per_kv;
        let v_start = kv_head * head_dim;
        let out_start = q_head * head_dim;
        attn_out[out_start..out_start + head_dim].copy_from_slice(&v[v_start..v_start + head_dim]);
    }
    eprintln!("[CPU-L0] Attn out first 3: {:?}", &attn_out[..3]);

    // Output projection
    let out_proj = cpu_model.fused_matmul(&attn_out, &layer.attn_output_weight)?;
    eprintln!("[CPU-L0] Output proj first 3: {:?}", &out_proj[..3]);

    // Compare with GPU
    eprintln!(
        "\n[GPU-L0] Output proj first 3 (from debug): [-0.95968825, 0.011617601, -0.059631474]"
    );

    // Diff
    let gpu_out_proj = [-0.95968825f32, 0.011617601, -0.059631474];
    eprintln!("\n=== Output Projection Comparison ===");
    for i in 0..3 {
        let diff = out_proj[i] - gpu_out_proj[i];
        eprintln!(
            "  [{}]: CPU={:.6}, GPU={:.6}, diff={:.6}",
            i, out_proj[i], gpu_out_proj[i], diff
        );
    }

    // Residual
    let mut residual1: Vec<f32> = embedding
        .iter()
        .zip(out_proj.iter())
        .map(|(&e, &o)| e + o)
        .collect();
    eprintln!("\n[CPU-L0] Residual1 first 3: {:?}", &residual1[..3]);
    eprintln!("[GPU-L0] Residual1 first 3 (from debug): [-0.9881397, 0.023871362, -0.047377713]");

    Ok(())
}
