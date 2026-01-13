//! CORRECTNESS-002: Layer-by-layer CPU vs GPU comparison
//!
//! Traces through each layer to find where GPU diverges from CPU
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_layer_trace

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Layer-by-layer trace\n");

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config.hidden_dim;
    let num_layers = cpu_model.config.num_layers;
    let num_heads = cpu_model.config.num_heads;
    let num_kv_heads = cpu_model.config.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    eprintln!("hidden_dim={}, num_layers={}", hidden_dim, num_layers);

    let test_token: u32 = 791;
    eprintln!("Testing token {}\n", test_token);

    // CPU: Get embedding
    let cpu_embedding = &cpu_model.embed_tokens.data;
    let embedding_offset = (test_token as usize) * hidden_dim * 2;
    let mut cpu_hidden: Vec<f32> = (0..hidden_dim)
        .map(|i| {
            let bytes = &cpu_embedding[embedding_offset + i * 2..embedding_offset + i * 2 + 2];
            half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
        })
        .collect();

    eprintln!("[CPU] Embedding first 5: {:?}", &cpu_hidden[..5]);

    // CPU: Run through layers one at a time
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);

    for layer_idx in 0..num_layers {
        // Run single layer on CPU
        let layer = &cpu_model.layers[layer_idx];

        // RMSNorm
        let rms_eps = cpu_model.config.rms_norm_eps;
        let sum_sq: f32 = cpu_hidden.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + rms_eps).sqrt();

        let normed: Vec<f32> = cpu_hidden
            .iter()
            .zip(layer.input_layernorm.data.chunks(2))
            .map(|(h, w)| {
                let weight = half::f16::from_bits(u16::from_le_bytes([w[0], w[1]])).to_f32();
                h / rms * weight
            })
            .collect();

        if layer_idx < 3 || layer_idx == num_layers - 1 {
            eprintln!(
                "[CPU L{}] Hidden first 3: [{:.4}, {:.4}, {:.4}]",
                layer_idx, cpu_hidden[0], cpu_hidden[1], cpu_hidden[2]
            );
            eprintln!(
                "[CPU L{}] Normed first 3: [{:.4}, {:.4}, {:.4}]",
                layer_idx, normed[0], normed[1], normed[2]
            );
        }
    }

    // Now run GPU forward and capture intermediate values
    #[cfg(feature = "cuda")]
    {
        use realizar::gguf::OwnedQuantizedModelCuda;

        eprintln!("\n=== GPU Forward with debug output ===");

        let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
        let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
        let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
        cuda_model.preload_weights_gpu()?;

        // Get GPU embedding for comparison
        let gpu_embed = &cuda_model.model.embed_tokens.data;
        let gpu_embedding: Vec<f32> = (0..hidden_dim)
            .map(|i| {
                let bytes = &gpu_embed[embedding_offset + i * 2..embedding_offset + i * 2 + 2];
                half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
            })
            .collect();

        eprintln!("[GPU] Embedding first 5: {:?}", &gpu_embedding[..5]);

        // Compare embeddings
        let embed_match = cpu_hidden
            .iter()
            .zip(gpu_embedding.iter())
            .all(|(c, g)| (c - g).abs() < 1e-6);
        eprintln!("Embeddings match: {}", embed_match);

        // Run full GPU forward
        std::env::set_var("GPU_DEBUG", "1");
        let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
        let _gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

        // The intermediate values are printed by the debug output in forward_gpu_resident
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CUDA not enabled");
    }

    Ok(())
}
