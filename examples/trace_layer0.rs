//! PAR-001b: Debug trace for layer 0 to find divergence point
//! Traces embedding -> RMS norm -> QKV matmul step by step

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf");

    println!("=== PAR-001b: Layer 0 Trace Debug ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("Model config:");
    println!("  hidden_dim: {}", model.config.hidden_dim);
    println!("  num_heads: {}", model.config.num_heads);
    println!("  num_kv_heads: {}", model.config.num_kv_heads);
    println!("  num_layers: {}", model.config.num_layers);
    println!("  vocab_size: {}", model.config.vocab_size);
    println!("  rope_theta: {}", model.config.rope_theta);
    println!("  eps: {}", model.config.eps);
    println!();

    // Test with token "Once" = 26222
    let token_id: u32 = 26222;
    println!("Test token: {} (id={})", "Once", token_id);

    // Step 1: Embedding lookup
    let embedding = model.embed(&[token_id]);
    println!("\n1. Embedding lookup:");
    println!("   Shape: [{}]", embedding.len());
    println!("   First 10: {:?}", &embedding[..10]);
    println!("   Last 10: {:?}", &embedding[embedding.len() - 10..]);
    println!(
        "   L2 norm: {}",
        (embedding.iter().map(|x| x * x).sum::<f32>()).sqrt()
    );
    println!(
        "   Mean: {}",
        embedding.iter().sum::<f32>() / embedding.len() as f32
    );
    println!(
        "   Min: {}, Max: {}",
        embedding.iter().cloned().fold(f32::INFINITY, f32::min),
        embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Step 2: RMS norm (layer 0 attention norm)
    let layer0 = &model.layers[0];
    println!("\n2. Layer 0 attention norm weight:");
    println!("   Shape: [{}]", layer0.attn_norm_weight.len());
    println!("   First 10: {:?}", &layer0.attn_norm_weight[..10]);
    println!(
        "   L2 norm: {}",
        (layer0.attn_norm_weight.iter().map(|x| x * x).sum::<f32>()).sqrt()
    );

    // Compute RMS norm manually
    let hidden_dim = model.config.hidden_dim;
    let eps = model.config.eps;
    let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    println!("\n   RMS computation:");
    println!("   sum_sq = {}", sum_sq);
    println!("   rms = {}", rms);
    println!("   inv_rms = {}", inv_rms);

    let normed: Vec<f32> = embedding
        .iter()
        .zip(layer0.attn_norm_weight.iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();
    println!("\n3. RMS-normed embedding:");
    println!("   Shape: [{}]", normed.len());
    println!("   First 10: {:?}", &normed[..10]);
    println!("   Last 10: {:?}", &normed[normed.len() - 10..]);
    println!(
        "   L2 norm: {}",
        (normed.iter().map(|x| x * x).sum::<f32>()).sqrt()
    );

    // Step 3: QKV weight info
    println!("\n4. Layer 0 QKV weight info:");
    match &layer0.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Fused(w) => {
            println!("   Type: Fused");
            println!("   in_dim: {}, out_dim: {}", w.in_dim, w.out_dim);
            println!("   qtype: {}", w.qtype);
            println!("   data len: {} bytes", w.data.len());
        },
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!("   Type: Separate Q/K/V");
            println!(
                "   Q: in={}, out={}, qtype={}",
                q.in_dim, q.out_dim, q.qtype
            );
            println!(
                "   K: in={}, out={}, qtype={}",
                k.in_dim, k.out_dim, k.qtype
            );
            println!(
                "   V: in={}, out={}, qtype={}",
                v.in_dim, v.out_dim, v.qtype
            );
        },
    }

    // Step 4: Try QKV matmul
    println!("\n5. QKV projection:");
    match model.qkv_matmul(&normed, &layer0.qkv_weight) {
        Ok(qkv) => {
            println!("   Shape: [{}]", qkv.len());
            println!("   First 10: {:?}", &qkv[..10.min(qkv.len())]);

            // Expected: Q (hidden_dim) + K (kv_dim) + V (kv_dim)
            let head_dim = hidden_dim / model.config.num_heads;
            let kv_dim = model.config.num_kv_heads * head_dim;
            let expected_qkv_dim = hidden_dim + 2 * kv_dim;
            println!(
                "   Expected dim: {} (Q:{} + K:{} + V:{})",
                expected_qkv_dim, hidden_dim, kv_dim, kv_dim
            );

            println!(
                "   L2 norm: {}",
                (qkv.iter().map(|x| x * x).sum::<f32>()).sqrt()
            );
            println!(
                "   Min: {}, Max: {}",
                qkv.iter().cloned().fold(f32::INFINITY, f32::min),
                qkv.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );

            // Check for NaN/Inf
            let nan_count = qkv.iter().filter(|x| x.is_nan()).count();
            let inf_count = qkv.iter().filter(|x| x.is_infinite()).count();
            println!("   NaN count: {}, Inf count: {}", nan_count, inf_count);

            // Show Q, K, V segments
            if qkv.len() >= expected_qkv_dim {
                println!("\n   Q segment (first 10): {:?}", &qkv[..10]);
                println!(
                    "   K segment (first 10): {:?}",
                    &qkv[hidden_dim..hidden_dim + 10]
                );
                println!(
                    "   V segment (first 10): {:?}",
                    &qkv[hidden_dim + kv_dim..hidden_dim + kv_dim + 10]
                );
            }
        },
        Err(e) => {
            println!("   ERROR: {:?}", e);
        },
    }

    // Step 5: Check LM head weights
    println!("\n6. LM Head weight info:");
    println!("   in_dim: {}", model.lm_head_weight.in_dim);
    println!("   out_dim: {} (vocab_size)", model.lm_head_weight.out_dim);
    println!("   qtype: {}", model.lm_head_weight.qtype);
    println!("   data len: {} bytes", model.lm_head_weight.data.len());

    // Step 6: Check output norm weight
    println!("\n7. Output norm weight:");
    println!("   Shape: [{}]", model.output_norm_weight.len());
    println!("   First 10: {:?}", &model.output_norm_weight[..10]);
    println!(
        "   L2 norm: {}",
        (model.output_norm_weight.iter().map(|x| x * x).sum::<f32>()).sqrt()
    );

    // Step 7: Run a full forward pass and print logits
    println!("\n8. Full forward pass (first token only):");
    use realizar::gguf::OwnedQuantizedKVCache;
    let head_dim = hidden_dim / model.config.num_heads;
    let kv_dim = model.config.num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(model.config.num_layers, kv_dim, 256);

    match model.forward_single_with_cache(token_id, &mut cache, 0) {
        Ok(logits) => {
            println!("   Logits shape: [{}]", logits.len());
            println!("   First 10 logits: {:?}", &logits[..10.min(logits.len())]);
            println!(
                "   L2 norm: {}",
                (logits.iter().map(|x| x * x).sum::<f32>()).sqrt()
            );
            println!(
                "   Min: {}, Max: {}",
                logits.iter().cloned().fold(f32::INFINITY, f32::min),
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );

            // Find top 5 tokens
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("\n   Top 5 predicted tokens:");
            let vocab = mapped.model.vocabulary().expect("test");
            for (idx, score) in indexed.iter().take(5) {
                let token_str = if *idx < vocab.len() {
                    vocab[*idx].replace('▁', " ")
                } else {
                    format!("<{}>", idx)
                };
                println!("     {} (id={}): {:.4}", token_str, idx, score);
            }

            // Check for NaN/Inf
            let nan_count = logits.iter().filter(|x| x.is_nan()).count();
            let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                println!("\n   WARNING: NaN={}, Inf={}", nan_count, inf_count);
            }
        },
        Err(e) => {
            println!("   ERROR: {:?}", e);
        },
    }

    println!("\n=== Debug trace complete ===");
}
