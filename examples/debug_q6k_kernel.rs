//! CORRECTNESS-002: Test Q6K GEMV kernel directly with synthetic data
//!
//! Compares GPU Q6K kernel output with CPU reference implementation
//!
//! Run with: cargo run --release --features cuda --example debug_q6k_kernel

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};
use realizar::quantize::fused_q6k_parallel_matvec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model to get LM head weights...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config.hidden_dim;
    let vocab_size = model.config.vocab_size;

    eprintln!(
        "Model config: hidden_dim={}, vocab_size={}",
        hidden_dim, vocab_size
    );
    eprintln!(
        "LM head: qtype={}, data_len={}",
        model.lm_head_weight.qtype,
        model.lm_head_weight.data.len()
    );

    // Create synthetic input (ones vector)
    let input_ones: Vec<f32> = vec![1.0; hidden_dim];

    // CPU reference: compute LM head with ones input
    eprintln!("\n=== CPU Reference (fused_q6k_parallel_matvec) ===");
    let cpu_logits = fused_q6k_parallel_matvec(
        &model.lm_head_weight.data,
        &input_ones,
        hidden_dim,
        vocab_size,
    )?;

    let cpu_sum: f32 = cpu_logits.iter().sum();
    let cpu_max = cpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU] Logits: sum={:.4}, max={:.4}, argmax={}",
        cpu_sum, cpu_max, cpu_argmax
    );
    eprintln!("[CPU] First 10: {:?}", &cpu_logits[..10]);

    // GPU: Initialize and run the same computation
    eprintln!("\n=== GPU Q6K GEMV ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;

    // Preload LM head weights
    cuda_model.preload_weights_gpu()?;

    // Run GPU LM head with ones input
    let gpu_logits = cuda_model.lm_head_forward_raw(&input_ones)?;

    let gpu_sum: f32 = gpu_logits.iter().sum();
    let gpu_max = gpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[GPU] Logits: sum={:.4}, max={:.4}, argmax={}",
        gpu_sum, gpu_max, gpu_argmax
    );
    eprintln!("[GPU] First 10: {:?}", &gpu_logits[..10]);

    // Element-wise comparison
    eprintln!("\n=== Comparison ===");
    let mut dot = 0.0f64;
    let mut cpu_sq = 0.0f64;
    let mut gpu_sq = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for i in 0..vocab_size {
        let c = cpu_logits[i] as f64;
        let g = gpu_logits[i] as f64;
        dot += c * g;
        cpu_sq += c * c;
        gpu_sq += g * g;
        let diff = (c - g).abs() as f32;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    let corr = dot / (cpu_sq.sqrt() * gpu_sq.sqrt());
    eprintln!("Correlation: {:.6}", corr);
    eprintln!(
        "Max diff: {:.6} at idx {} (CPU={:.4}, GPU={:.4})",
        max_diff, max_diff_idx, cpu_logits[max_diff_idx], gpu_logits[max_diff_idx]
    );

    // Sum ratio
    eprintln!("Sum ratio GPU/CPU: {:.6}", gpu_sum / cpu_sum);

    if corr > 0.99 {
        eprintln!("\n[OK] GPU Q6K kernel matches CPU reference");
    } else if corr < 0.0 {
        eprintln!("\n[FAIL] Q6K kernel has NEGATIVE correlation - major bug in kernel!");
    } else {
        eprintln!("\n[FAIL] Q6K kernel diverges from CPU (corr={:.4})", corr);
    }

    Ok(())
}
