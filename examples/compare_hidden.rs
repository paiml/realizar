//! Compare CPU vs GPU hidden states
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");
    
    let eps = model.config.eps;
    let hidden_dim = model.config.hidden_dim;
    
    // CPU forward with full tracing
    let token_id = 791u32;
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding[start..start + hidden_dim].to_vec();
    
    println!("Token {} embedding first 5: {:?}", token_id, &hidden[..5]);
    
    // Process all layers
    for layer_idx in 0..model.config.num_layers {
        // Run CPU layer forward
        let layer = &model.layers[layer_idx];
        
        // Pre-attention RMSNorm
        let rms = (hidden.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32 + eps).sqrt();
        let normed: Vec<f32> = hidden.iter().zip(layer.attn_norm_weight.iter())
            .map(|(x, w)| (x / rms) * w).collect();
        
        // Full layer processing (using model's built-in method)
        hidden = model.forward_layer(layer_idx, &hidden).expect("layer forward");
    }
    
    println!("\nCPU final hidden (before output norm):");
    println!("  first 5: {:?}", &hidden[..5]);
    println!("  sum: {:.4}", hidden.iter().sum::<f32>());
    println!("  rms: {:.4}", (hidden.iter().map(|x|x*x).sum::<f32>() / hidden.len() as f32).sqrt());
    
    // Apply output norm
    let output_rms = (hidden.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = hidden.iter().zip(model.output_norm_weight.iter())
        .map(|(x, w)| (x / output_rms) * w).collect();
    
    println!("\nCPU normed hidden:");
    println!("  first 5: {:?}", &normed[..5]);
    println!("  sum: {:.4}", normed.iter().sum::<f32>());
    
    // Compare with GPU values from debug output
    println!("\nGPU values (from debug):");
    println!("  Hidden: [1.072186, 7.35783, -18.59197, 22.212904, -23.266695], sum=483.2961");
    println!("  Normed: [0.11976412, 0.8568495, -1.5612367, 2.604381, -2.6726382]");
    
    // Compute differences
    let gpu_hidden = [1.072186f32, 7.35783, -18.59197, 22.212904, -23.266695];
    println!("\nDifferences in final hidden (first 5):");
    for i in 0..5 {
        let diff = (hidden[i] - gpu_hidden[i]).abs();
        println!("  [{}] CPU={:.4}, GPU={:.4}, diff={:.4}", i, hidden[i], gpu_hidden[i], diff);
    }
}
