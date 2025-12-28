//! Trace forward pass step by step
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn stats(x: &[f32]) -> (f32, f32, f32, f32) {
    let min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let std: f32 = (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32).sqrt();
    (min, max, mean, std)
}

fn main() {
    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").unwrap();
    let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();

    let hidden_dim = model.config.hidden_dim;
    
    // Get embedding for token 1 (BOS)
    let token_id = 1u32;
    let embed = &model.token_embedding[token_id as usize * hidden_dim..(token_id as usize + 1) * hidden_dim];
    
    println!("\n1. Embedding for BOS (token 1):");
    let (min, max, mean, std) = stats(embed);
    println!("   shape: [{}]", hidden_dim);
    println!("   first 5: {:?}", &embed[..5]);
    println!("   stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", min, max, mean, std);

    // Get first layer
    let layer = &model.layers[0];
    
    // Apply RMSNorm manually to understand what's happening
    println!("\n2. First layer attn_norm_weight:");
    let (min, max, mean, std) = stats(&layer.attn_norm_weight);
    println!("   shape: [{}]", layer.attn_norm_weight.len());
    println!("   first 5: {:?}", &layer.attn_norm_weight[..5]);
    println!("   stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", min, max, mean, std);

    // Compute RMSNorm of embedding
    let eps = model.config.eps;
    let mean_sq: f32 = embed.iter().map(|v| v * v).sum::<f32>() / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    
    let normed: Vec<f32> = embed.iter()
        .zip(&layer.attn_norm_weight)
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();
    
    println!("\n3. After RMSNorm (embed * inv_rms * weight):");
    let (min, max, mean, std) = stats(&normed);
    println!("   mean_sq={:.6}, inv_rms={:.4}", mean_sq, inv_rms);
    println!("   first 5: {:?}", &normed[..5]);
    println!("   stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", min, max, mean, std);

    // What if we just normalized WITHOUT the weight?
    let normed_no_weight: Vec<f32> = embed.iter()
        .map(|&x| x * inv_rms)
        .collect();
    
    println!("\n4. RMSNorm WITHOUT weight multiplication:");
    let (min, max, mean, std) = stats(&normed_no_weight);
    println!("   first 5: {:?}", &normed_no_weight[..5]);
    println!("   stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", min, max, mean, std);

    // What if weight is 1.0 + stored_weight?
    let normed_with_offset: Vec<f32> = embed.iter()
        .zip(&layer.attn_norm_weight)
        .map(|(&x, &w)| x * inv_rms * (1.0 + w))
        .collect();
    
    println!("\n5. RMSNorm with weight=1.0+stored:");
    let (min, max, mean, std) = stats(&normed_with_offset);
    println!("   first 5: {:?}", &normed_with_offset[..5]);
    println!("   stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", min, max, mean, std);

    // Check the output_norm_weight for comparison
    println!("\n6. output_norm_weight (for comparison):");
    let (min, max, mean, std) = stats(&model.output_norm_weight);
    println!("   first 5: {:?}", &model.output_norm_weight[..5]);
    println!("   stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", min, max, mean, std);

    // Try full forward pass
    println!("\n7. Running full forward pass on [1] (BOS)...");
    let logits = model.forward(&[1]).unwrap();
    let (min, max, mean, std) = stats(&logits);
    println!("   logits stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}", min, max, mean, std);
}
