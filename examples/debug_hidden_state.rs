//! Compare hidden state before output norm between CPU and GPU
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

fn main() {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");
    std::env::set_var("GPU_DEBUG", "1");
    std::env::set_var("REALIZAR_DEBUG_FORWARD", "1"); // Enable CPU debug output

    let model_path = "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("load");
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped).expect("cpu model");

    println!("Creating CUDA model...");
    let mut cuda_model = OwnedQuantizedModelCuda::new(cpu_model.clone(), 0).expect("cuda model");
    cuda_model.preload_weights_gpu().expect("preload");

    let token_id = 9707u32; // "Hello"

    let kv_dim =
        cpu_model.config.num_kv_heads * (cpu_model.config.hidden_dim / cpu_model.config.num_heads);
    let mut cpu_cache =
        realizar::gguf::OwnedQuantizedKVCache::new(cpu_model.config.num_layers, kv_dim, 16);

    println!("\n=== CPU Forward ===");
    let cpu_logits = cpu_model
        .forward_cached(token_id, &mut cpu_cache, 0)
        .expect("cpu forward");

    println!("\n=== GPU Forward ===");
    let mut gpu_cache =
        realizar::gguf::OwnedQuantizedKVCache::new(cpu_model.config.num_layers, kv_dim, 16);
    let gpu_logits = cuda_model
        .forward_gpu_resident(token_id, &mut gpu_cache, 0)
        .expect("gpu forward");

    // Compare logits
    println!("\n=== Logits Comparison ===");
    let cpu_top = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();
    let gpu_top = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    println!("CPU argmax: {} (logit {:.4})", cpu_top.0, cpu_top.1);
    println!("GPU argmax: {} (logit {:.4})", gpu_top.0, gpu_top.1);
}
