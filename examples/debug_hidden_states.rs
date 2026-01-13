//! CORRECTNESS-002: Compare hidden states between CPU and GPU
//!
//! Tests that hidden states match at each layer
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_hidden_states

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Get config
    let hidden_dim = cpu_model.config.hidden_dim;
    let num_layers = cpu_model.config.num_layers;
    let num_heads = cpu_model.config.num_heads;
    let num_kv_heads = cpu_model.config.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let vocab_size = cpu_model.config.vocab_size;

    let test_token: u32 = 791;
    eprintln!("\n=== Testing token {} ===", test_token);

    // Get CPU embedding
    let cpu_embed = cpu_model.embed(&[test_token]);
    let cpu_embed_sum: f32 = cpu_embed.iter().sum();
    let cpu_embed_rms: f32 =
        (cpu_embed.iter().map(|x| x * x).sum::<f32>() / cpu_embed.len() as f32).sqrt();
    eprintln!(
        "[CPU] Embedding: sum={:.4}, rms={:.4}, first 5={:?}",
        cpu_embed_sum,
        cpu_embed_rms,
        &cpu_embed[..5]
    );

    // Now test LM head directly with CPU embedding
    // This bypasses all transformer layers and just tests embedding -> LM head
    eprintln!("\n=== Direct LM head test ===");

    // CPU: Apply output norm to embedding (as a simple test vector)
    let cpu_normed = cpu_model.layer_norm(
        &cpu_embed,
        &cpu_model.output_norm_weight,
        None,
        cpu_model.config.eps,
    );
    let cpu_normed_sum: f32 = cpu_normed.iter().sum();
    let cpu_normed_rms: f32 =
        (cpu_normed.iter().map(|x| x * x).sum::<f32>() / cpu_normed.len() as f32).sqrt();
    eprintln!(
        "[CPU] Normed embed: sum={:.4}, rms={:.4}, first 5={:?}",
        cpu_normed_sum,
        cpu_normed_rms,
        &cpu_normed[..5]
    );

    // CPU: Apply LM head
    let cpu_lm_logits = cpu_model.matmul_q6k_vec(&cpu_model.lm_head_weight, &cpu_normed)?;
    let cpu_lm_sum: f32 = cpu_lm_logits.iter().sum();
    let cpu_lm_argmax = cpu_lm_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU] Direct LM head: sum={:.4}, argmax={}, first 5={:?}",
        cpu_lm_sum,
        cpu_lm_argmax,
        &cpu_lm_logits[..5]
    );

    // GPU: Test same path
    eprintln!("\nInitializing GPU...");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    // GPU: Upload normed embedding and run LM head only
    let gpu_lm_logits = cuda_model.lm_head_only(&cpu_normed)?;
    let gpu_lm_sum: f32 = gpu_lm_logits.iter().sum();
    let gpu_lm_argmax = gpu_lm_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[GPU] Direct LM head: sum={:.4}, argmax={}, first 5={:?}",
        gpu_lm_sum,
        gpu_lm_argmax,
        &gpu_lm_logits[..5]
    );

    // Compare
    let mut dot = 0.0f64;
    let mut cpu_sq = 0.0f64;
    let mut gpu_sq = 0.0f64;
    for i in 0..vocab_size {
        let c = cpu_lm_logits[i] as f64;
        let g = gpu_lm_logits[i] as f64;
        dot += c * g;
        cpu_sq += c * c;
        gpu_sq += g * g;
    }
    let corr = dot / (cpu_sq.sqrt() * gpu_sq.sqrt());
    eprintln!("\nLM head correlation: {:.6}", corr);

    if corr > 0.99 {
        eprintln!("[OK] LM head GPU matches CPU");
    } else if corr < 0.0 {
        eprintln!("[FAIL] LM head has NEGATIVE correlation!");
    } else {
        eprintln!("[FAIL] LM head diverges (corr={:.4})", corr);
    }

    Ok(())
}
