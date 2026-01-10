//! Quick test to verify RoPE is being applied correctly on GPU path
//! Run: cd /home/noah/src/realizar && cargo run --release --example test_rope_fix

use realizar::gguf::{GGUFModel, GpuInferenceEngine};
use realizar::inference::TruenoInferenceEngine;
use std::path::Path;

const MODEL_PATH: &str = "/home/noah/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     PAR-060: RoPE GPU/CPU Comparison Test                  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Load model
    println!("Loading model...");
    let model_path = Path::new(MODEL_PATH);
    let model = match GGUFModel::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        },
    };

    println!(
        "Model: {} layers, hidden_dim={}, vocab_size={}, rope_theta={}",
        model.layers.len(),
        model.config.hidden_dim,
        model.config.vocab_size,
        model.config.rope_theta
    );

    // Tokenize test prompt
    let prompt = "Hi";
    let tokens = match model.tokenize(prompt) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Tokenization failed: {}", e);
            return;
        },
    };
    println!("\nPrompt: {:?}", prompt);
    println!("Tokens: {:?}", tokens);

    // === CPU Inference ===
    println!("\n--- CPU Inference ---");
    let cpu_engine = TruenoInferenceEngine::from_gguf(&model);
    let cpu_logits = cpu_engine
        .forward_batch(&tokens)
        .expect("CPU forward failed");
    let last_cpu_logits = &cpu_logits[cpu_logits.len() - model.config.vocab_size..];

    // Get top token from CPU
    let (cpu_top_idx, cpu_top_val) = last_cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!(
        "CPU top token: {} with logit {:.4}",
        cpu_top_idx, cpu_top_val
    );
    println!("CPU first 5 logits: {:?}", &last_cpu_logits[..5]);

    // === GPU Inference ===
    println!("\n--- GPU Inference ---");
    let mut gpu_engine = match GpuInferenceEngine::new(&model, 2048) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("GPU init failed: {}", e);
            return;
        },
    };

    // Prepare output buffer
    let mut gpu_logits = vec![0.0f32; model.config.vocab_size];

    // Run GPU forward
    if let Err(e) = gpu_engine.forward_to_logits(&tokens, &mut gpu_logits) {
        eprintln!("GPU forward failed: {}", e);
        return;
    }

    // Get top token from GPU
    let (gpu_top_idx, gpu_top_val) = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!(
        "GPU top token: {} with logit {:.4}",
        gpu_top_idx, gpu_top_val
    );
    println!("GPU first 5 logits: {:?}", &gpu_logits[..5]);

    // === Comparison ===
    println!("\n--- Comparison ---");

    // Check if top tokens match
    if cpu_top_idx == gpu_top_idx {
        println!(
            "✅ TOP TOKENS MATCH: {} (CPU={:.4}, GPU={:.4})",
            cpu_top_idx, cpu_top_val, gpu_top_val
        );
    } else {
        println!(
            "❌ TOP TOKENS DIFFER: CPU={} vs GPU={}",
            cpu_top_idx, gpu_top_idx
        );
    }

    // Calculate cosine similarity
    let dot: f32 = last_cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| a * b)
        .sum();
    let cpu_norm: f32 = last_cpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
    let gpu_norm: f32 = gpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine_sim = dot / (cpu_norm * gpu_norm);
    println!("Cosine similarity: {:.6}", cosine_sim);

    // Calculate max absolute difference
    let max_diff = last_cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    println!("Max absolute difference: {:.6}", max_diff);

    // PASS/FAIL determination
    println!("\n═══════════════════════════════════════════════════════════════");
    if cpu_top_idx == gpu_top_idx && cosine_sim > 0.99 {
        println!("✅ PASS: GPU output matches CPU (RoPE fix verified)");
    } else {
        println!("❌ FAIL: GPU output still differs from CPU");
        println!("   This may indicate:");
        println!("   - RoPE not being applied correctly");
        println!("   - Other numerical differences (precision, etc.)");
    }
}
