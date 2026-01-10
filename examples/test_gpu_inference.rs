//! Quick test of GPU inference with Qwen model
//!
//! Run with: cargo run --release --features cuda --example test_gpu_inference

use realizar::cuda::CudaExecutor;
use realizar::gguf::GGUFTransformer;
use std::time::Instant;

fn main() -> realizar::Result<()> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model from {}", model_path);
    let start = Instant::now();
    let transformer = GGUFTransformer::from_gguf_file(model_path)?;
    println!("Model loaded in {:?}", start.elapsed());

    println!("\nModel config:");
    println!("  hidden_dim: {}", transformer.config.hidden_dim);
    println!(
        "  intermediate_dim: {}",
        transformer.config.intermediate_dim
    );
    println!("  num_layers: {}", transformer.config.num_layers);
    println!("  num_heads: {}", transformer.config.num_heads);
    println!("  num_kv_heads: {}", transformer.config.num_kv_heads);
    println!(
        "  head_dim: {}",
        transformer.config.hidden_dim / transformer.config.num_heads
    );
    println!("  vocab_size: {}", transformer.config.vocab_size);
    println!("  rope_theta: {}", transformer.config.rope_theta);
    println!("  rope_type: {}", transformer.config.rope_type);

    // Initialize GPU
    println!("\nInitializing GPU...");
    let mut executor = CudaExecutor::new(0)?;

    // Simple prompt: "2+2="
    let prompt = "What is 2+2? Answer:";
    println!("\nPrompt: {}", prompt);

    // Get tokenizer and tokenize
    let tokens = transformer.tokenize(prompt)?;
    println!(
        "Input tokens ({} tokens): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(10)]
    );

    // Initialize KV cache
    let max_seq_len = 64;
    let num_layers = transformer.config.num_layers;
    let num_heads = transformer.config.num_heads;
    let num_kv_heads = transformer.config.num_kv_heads;
    let head_dim = transformer.config.hidden_dim / num_heads;

    executor.init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)?;
    executor.set_rope_theta(transformer.config.rope_theta);

    // Index weights for GPU
    println!("\nIndexing weights for GPU...");
    executor.index_weights_gpu(&transformer)?;

    // Initialize workspace
    let hidden_dim = transformer.config.hidden_dim;
    let intermediate_dim = transformer.config.intermediate_dim;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    executor.init_workspace(hidden_dim, intermediate_dim, q_dim, kv_dim)?;

    // Run prefill
    println!("\nRunning prefill...");
    let start = Instant::now();
    let prefill_logits = executor.prefill_gpu(&tokens, &transformer)?;
    executor.sync()?;
    println!("Prefill took {:?}", start.elapsed());

    // Get top 5 predictions from last position
    let last_logits = &prefill_logits[prefill_logits.len() - transformer.config.vocab_size..];
    let mut sorted: Vec<(usize, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 predictions after prefill:");
    for (i, (token_id, logit)) in sorted.iter().take(5).enumerate() {
        let token_str = transformer
            .decode_token(*token_id as u32)
            .unwrap_or("<unk>".to_string());
        println!(
            "  {}. token {} ({:?}): {:.4}",
            i + 1,
            token_id,
            token_str,
            logit
        );
    }

    // Greedy decode a few tokens
    println!("\nGreedy decode (GPU):");
    let mut generated_tokens = vec![];
    for i in 0..8 {
        let logits = if i == 0 {
            // Use prefill logits for first decode
            last_logits.to_vec()
        } else {
            // Decode step
            let last_token = generated_tokens.last().copied().unwrap();
            executor.decode_step_gpu(last_token, &transformer)?
        };

        // Argmax
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        generated_tokens.push(next_token);
        let token_str = transformer
            .decode_token(next_token)
            .unwrap_or("<unk>".to_string());
        print!("{}", token_str);
        std::io::Write::flush(&mut std::io::stdout())?;

        // Stop on EOS
        if next_token == 151645 || next_token == 151643 {
            break;
        }
    }
    println!();

    // Compare with CPU
    println!("\n--- CPU Reference ---");
    let cpu_logits = transformer.forward_with_cache(&tokens)?;
    let cpu_last = &cpu_logits[cpu_logits.len() - transformer.config.vocab_size..];
    let mut cpu_sorted: Vec<(usize, f32)> =
        cpu_last.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    cpu_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 predictions (CPU):");
    for (i, (token_id, logit)) in cpu_sorted.iter().take(5).enumerate() {
        let token_str = transformer
            .decode_token(*token_id as u32)
            .unwrap_or("<unk>".to_string());
        println!(
            "  {}. token {} ({:?}): {:.4}",
            i + 1,
            token_id,
            token_str,
            logit
        );
    }

    // Compare hidden states
    let gpu_sum: f32 = last_logits.iter().sum();
    let cpu_sum: f32 = cpu_last.iter().sum();
    println!(
        "\nLogit sums - GPU: {:.4}, CPU: {:.4}, ratio: {:.4}",
        gpu_sum,
        cpu_sum,
        gpu_sum / cpu_sum
    );

    Ok(())
}
