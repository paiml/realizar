//! End-to-end inference example for realizar
//!
//! Demonstrates the complete text generation pipeline:
//! - Model initialization with configuration
//! - Forward pass through transformer blocks
//! - Text generation with various sampling strategies

use realizar::{
    generate::GenerationConfig,
    layers::{Model, ModelConfig},
};

fn main() {
    println!("=== Realizar Inference Example ===\n");

    // Define a small model configuration for demonstration
    let config = ModelConfig {
        vocab_size: 100,      // Small vocabulary for demo
        hidden_dim: 32,       // Reduced dimensions for speed
        num_heads: 1,         // Single attention head
        num_layers: 2,        // Two transformer blocks
        intermediate_dim: 64, // FFN intermediate size
        eps: 1e-5,            // Layer normalization epsilon
    };

    // Create the model (initializes with random weights)
    let model = Model::new(config.clone()).expect("Failed to create model");

    println!("Model Configuration:");
    println!("  - Vocabulary Size: {}", config.vocab_size);
    println!("  - Hidden Dimension: {}", config.hidden_dim);
    println!("  - Number of Heads: {}", config.num_heads);
    println!("  - Number of Layers: {}", config.num_layers);
    println!("  - Intermediate Dim: {}", config.intermediate_dim);
    println!(
        "  - Approx Parameters: {}",
        format_params(model.num_parameters())
    );
    println!();

    // Example 1: Forward pass
    println!("--- Forward Pass ---");
    let prompt_tokens = vec![1, 5, 10]; // Example token IDs
    let logits = model.forward(&prompt_tokens).expect("Forward pass failed");
    println!("Input tokens: {:?}", prompt_tokens);
    println!(
        "Output logits shape: [{}, {}]",
        prompt_tokens.len(),
        config.vocab_size
    );
    println!(
        "Last position max logit: {:.4}",
        find_max(&logits.data()[(prompt_tokens.len() - 1) * config.vocab_size..])
    );
    println!();

    // Example 2: Greedy generation
    println!("--- Greedy Generation ---");
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(5)
        .with_eos_token_id(99); // Use token 99 as EOS

    let generated = model
        .generate(&prompt_tokens, &gen_config)
        .expect("Generation failed");
    println!("Prompt: {:?}", prompt_tokens);
    println!("Generated: {:?}", generated);
    println!("New tokens: {:?}", &generated[prompt_tokens.len()..]);
    println!();

    // Example 3: Top-k sampling
    println!("--- Top-K Sampling (k=5) ---");
    let gen_config = GenerationConfig::top_k(5)
        .with_temperature(0.8)
        .with_max_tokens(5)
        .with_seed(42); // Deterministic for reproducibility

    let generated = model
        .generate(&prompt_tokens, &gen_config)
        .expect("Generation failed");
    println!("Prompt: {:?}", prompt_tokens);
    println!("Generated: {:?}", generated);
    println!();

    // Example 4: Top-p (nucleus) sampling
    println!("--- Top-P Sampling (p=0.9) ---");
    let gen_config = GenerationConfig::top_p(0.9)
        .with_temperature(0.7)
        .with_max_tokens(5)
        .with_seed(123);

    let generated = model
        .generate(&prompt_tokens, &gen_config)
        .expect("Generation failed");
    println!("Prompt: {:?}", prompt_tokens);
    println!("Generated: {:?}", generated);
    println!();

    // Example 5: Early stopping with EOS
    println!("--- Early Stopping Test ---");
    // Set EOS to a common token to see early stopping
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(20)
        .with_eos_token_id(50); // Will stop if token 50 is generated

    let generated = model
        .generate(&prompt_tokens, &gen_config)
        .expect("Generation failed");
    println!("Max tokens: 20");
    println!("Generated length: {}", generated.len());
    println!(
        "Stopped early: {}",
        if generated.len() < prompt_tokens.len() + 20 {
            "yes"
        } else {
            "no"
        }
    );
    println!();

    println!("=== Inference Complete ===");
}

/// Format parameter count with K/M suffixes
fn format_params(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}

/// Find maximum value in slice
fn find_max(slice: &[f32]) -> f32 {
    slice
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).expect("Comparison failed: NaN values not allowed in logits"))
        .unwrap_or(0.0)
}
