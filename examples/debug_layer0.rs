//! Debug Layer 0 output - compare CPU vs GPU for first transformer layer
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

fn main() {
    // PAR-069: Set CUDA_GRAPH_DISABLE BEFORE any CUDA operations
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let model_path = "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("load");
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped).expect("cpu model");

    println!("Model config:");
    println!("  hidden_dim: {}", cpu_model.config.hidden_dim);
    println!("  num_heads: {}", cpu_model.config.num_heads);
    println!("  num_kv_heads: {}", cpu_model.config.num_kv_heads);

    println!("\nCreating CUDA model...");
    let mut cuda_model = OwnedQuantizedModelCuda::new(cpu_model.clone(), 0).expect("cuda model");

    // Single token test
    let token_id = 9707u32; // "Hello"
    println!("\nTesting with token {} ('Hello')", token_id);

    // 1. Compare embeddings (both use CPU for this)
    let embedding = cpu_model.embed(&[token_id]);
    println!("Embedding[0..5]: {:?}", &embedding[..5]);
    println!("Embedding sum: {:.6}", embedding.iter().sum::<f32>());

    // 2. Run CPU forward through ONLY layer 0
    // We need to manually call the layer operations
    let layer0 = &cpu_model.layers[0];

    // 2a. RMSNorm
    let mut normed = embedding.clone();
    cpu_model.rmsnorm(&mut normed, &layer0.attn_norm_gamma);
    println!("\nCPU After attn_norm (layer 0):");
    println!("  normed[0..5]: {:?}", &normed[..5]);
    println!("  normed sum: {:.6}", normed.iter().sum::<f32>());

    // 2b. QKV projection (separate Q, K, V for Qwen2.5)
    let q_dim =
        cpu_model.config.num_heads * (cpu_model.config.hidden_dim / cpu_model.config.num_heads);
    let kv_dim =
        cpu_model.config.num_kv_heads * (cpu_model.config.hidden_dim / cpu_model.config.num_heads);

    let (q_weight, k_weight, v_weight) = match &layer0.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => (q, k, v),
        _ => panic!("Expected separate Q/K/V weights"),
    };

    let q_cpu = cpu_model.q4k_matvec(&q_weight, &normed);
    let k_cpu = cpu_model.q4k_matvec(&k_weight, &normed);
    let v_cpu = cpu_model.q4k_matvec(&v_weight, &normed);

    println!("\nCPU Q projection (layer 0, q_dim={}):", q_dim);
    println!("  Q[0..5]: {:?}", &q_cpu[..5]);
    println!("  Q sum: {:.6}", q_cpu.iter().sum::<f32>());

    println!("\nCPU K projection (layer 0, kv_dim={}):", kv_dim);
    println!("  K[0..5]: {:?}", &k_cpu[..5]);
    println!("  K sum: {:.6}", k_cpu.iter().sum::<f32>());

    println!("\nCPU V projection (layer 0):");
    println!("  V[0..5]: {:?}", &v_cpu[..5]);
    println!("  V sum: {:.6}", v_cpu.iter().sum::<f32>());

    // 3. Now get GPU intermediate values via debug mode
    // We need to enable skip_debug=false to see the intermediate values
    // For now, let's just run the forward and compare final output

    // Preload and run GPU forward
    cuda_model.preload_weights_gpu().expect("preload");

    let kv_dim_full =
        cpu_model.config.num_kv_heads * (cpu_model.config.hidden_dim / cpu_model.config.num_heads);
    let mut cache =
        realizar::gguf::OwnedQuantizedKVCache::new(cpu_model.config.num_layers, kv_dim_full, 16);

    let gpu_logits = cuda_model
        .forward_gpu_resident(token_id, &mut cache, 0)
        .expect("gpu forward");

    // CPU full forward for comparison
    let cpu_logits = cpu_model.forward(&[token_id]).expect("cpu forward");

    // Compare
    println!("\n=== FINAL LOGITS COMPARISON ===");
    let cpu_top: (usize, f32) = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();
    let gpu_top: (usize, f32) = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    println!("CPU argmax: {} (logit {:.4})", cpu_top.0, cpu_top.1);
    println!("GPU argmax: {} (logit {:.4})", gpu_top.0, gpu_top.1);

    if cpu_top.0 == gpu_top.0 {
        println!("✅ CPU and GPU agree!");
    } else {
        println!("❌ CPU and GPU DISAGREE!");

        // More detailed comparison
        let mut diff_sum = 0.0f32;
        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;
        for i in 0..cpu_logits.len().min(gpu_logits.len()) {
            let diff = (cpu_logits[i] - gpu_logits[i]).abs();
            diff_sum += diff;
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }
        println!("\nMean diff: {:.6}", diff_sum / cpu_logits.len() as f32);
        println!("Max diff: {:.6} at idx {}", max_diff, max_diff_idx);
    }
}
