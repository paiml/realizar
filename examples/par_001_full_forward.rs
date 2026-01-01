//! PAR-001: Full forward pass comparison
//!
//! Run a complete forward pass and compare with llama.cpp output

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Full Forward Pass ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();
    let vocab = mapped.model.vocabulary().unwrap();

    println!(
        "Model: {} layers, {} heads, {} kv_heads",
        model.config.num_layers, model.config.num_heads, model.config.num_kv_heads
    );

    // Test tokens - "Once upon a time"
    let tokens = [26222u32, 2501, 263, 931]; // Once upon a time (approx)
    println!("\nTest prompt tokens: {:?}", tokens);
    for &t in &tokens {
        let s = vocab.get(t as usize).map(|s| s.as_str()).unwrap_or("?");
        println!("  {}: '{}'", t, s);
    }

    // Run forward pass for each token position
    let kv_dim = model.config.num_kv_heads * (model.config.hidden_dim / model.config.num_heads);
    let max_seq_len = 128;
    let mut cache = OwnedQuantizedKVCache::new(model.config.num_layers, kv_dim, max_seq_len);

    println!("\n=== Forward passes ===");
    for (pos, &token) in tokens.iter().enumerate() {
        let logits = model
            .forward_cached(token, &mut cache, pos)
            .expect("forward failed");

        let l2 = l2_norm(&logits);
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Find top 5 tokens
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!(
            "\nPosition {}: token {} ('{}')",
            pos,
            token,
            vocab.get(token as usize).map(|s| s.as_str()).unwrap_or("?")
        );
        println!("  Logits: L2={:.2}, min={:.4}, max={:.4}", l2, min, max);
        println!("  Top 5 next tokens:");
        for (rank, (idx, score)) in indexed.iter().take(5).enumerate() {
            let tok_str = vocab.get(*idx).map(|s| s.as_str()).unwrap_or("?");
            println!(
                "    {}: token {} = {:.4} ('{}')",
                rank + 1,
                idx,
                score,
                tok_str
            );
        }

        // Also show what greedy decoding would pick
        let greedy_idx = indexed[0].0;
        let greedy_str = vocab.get(greedy_idx).map(|s| s.as_str()).unwrap_or("?");
        println!("  → Greedy next: {} ('{}')", greedy_idx, greedy_str);
    }

    // Now generate a few tokens
    println!("\n=== Generation (greedy) ===");
    let mut generated = tokens.to_vec();
    for _ in 0..10 {
        let pos = generated.len() - 1;
        let token = generated[pos];
        let logits = model
            .forward_cached(token, &mut cache, pos)
            .expect("forward failed");

        // Greedy select
        let (next_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        generated.push(next_idx as u32);

        let tok_str = vocab.get(next_idx).map(|s| s.as_str()).unwrap_or("?");
        print!("{}", tok_str);
    }
    println!();

    // Print full generated sequence
    println!("\nFull sequence:");
    for &t in &generated {
        let s = vocab.get(t as usize).map(|s| s.as_str()).unwrap_or("?");
        print!("{}", s);
    }
    println!();

    println!("\n=== Complete ===");
}
