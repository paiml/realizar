//! Phase 15b: Step-by-Step Forward Parity Test
//!
//! Compares CPU and GPU forward pass results to isolate the divergence point.

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig};

    /// Test full forward pass comparison
    #[test]
    fn test_forward_parity() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 15b: CPU vs GPU FORWARD PARITY                                ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path = std::path::Path::new(env!("HOME"))
            .join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found");
            return;
        }

        let mapped = MappedGGUFModel::from_path(model_path.to_str().unwrap())
            .expect("Failed to mmap GGUF");

        // Test with 2 tokens: BOS + "Once"
        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        let once_tokens = mapped.model.encode("Once").unwrap_or(vec![9038]);
        let tokens = vec![bos_token, once_tokens[0]];

        eprintln!("Test tokens: {:?}", tokens);
        for (i, &tok) in tokens.iter().enumerate() {
            let decoded = mapped.model.decode(&[tok]);
            eprintln!("  [{}] {} = {:?}", i, tok, decoded);
        }

        // CPU forward
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("CPU FORWARD PASS");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)
            .expect("Failed to load CPU model");
        let mut cpu_cache = realizar::gguf::OwnedQuantizedKVCache::from_config(
            cpu_model.config(), 20
        );

        let mut cpu_logits = vec![];
        for (pos, &tok) in tokens.iter().enumerate() {
            cpu_logits = cpu_model.forward_cached(tok, &mut cpu_cache, pos)
                .expect("CPU forward failed");
            let l2 = cpu_logits.iter().map(|x| x*x).sum::<f32>().sqrt();
            let top1 = cpu_logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            let decoded = mapped.model.decode(&[top1]);
            eprintln!("Position {}: L2={:.4}, top1={} ({:?})", pos, l2, top1, decoded);
        }

        // GPU forward
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("GPU FORWARD PASS (generate_full_cuda_with_cache)");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        std::env::set_var("SKIP_CUDA_GRAPH", "1");

        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(), 0, 20
        ).expect("Failed to create CUDA model");

        let gen_config = QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![],
        };

        let gpu_result = cuda_model.generate_full_cuda_with_cache(&tokens, &gen_config)
            .expect("GPU forward failed");

        std::env::remove_var("SKIP_CUDA_GRAPH");

        let gpu_next_token = if gpu_result.len() > tokens.len() {
            gpu_result[tokens.len()]
        } else {
            0
        };
        let gpu_decoded = mapped.model.decode(&[gpu_next_token]);

        eprintln!("Generated: {:?}", gpu_result);
        eprintln!("Next token: {} ({:?})", gpu_next_token, gpu_decoded);

        // Compare
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("COMPARISON");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let cpu_top1 = cpu_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        let cpu_decoded = mapped.model.decode(&[cpu_top1]);

        eprintln!("CPU next token: {} ({:?})", cpu_top1, cpu_decoded);
        eprintln!("GPU next token: {} ({:?})", gpu_next_token, gpu_decoded);

        if cpu_top1 == gpu_next_token {
            eprintln!("\n✅ MATCH: CPU and GPU agree!");
        } else {
            eprintln!("\n❌ DIVERGENCE DETECTED!");
            eprintln!("\nAnalysis:");
            eprintln!("  The generate_full_cuda_with_cache path uses:");
            eprintln!("  - CPU embedding lookup");
            eprintln!("  - GPU QKV projection (matmul)");
            eprintln!("  - CPU RoPE");
            eprintln!("  - CPU attention");
            eprintln!("  - GPU output/FFN projections (matmul)");
            eprintln!("  - GPU LM head projection (matmul)");
            eprintln!("\n  Since CPU RoPE and attention are shared, the bug is in GPU matmul.");
        }
    }

    /// Single token test - should match since no KV cache involved
    #[test]
    fn test_single_token_parity() {
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("SINGLE TOKEN PARITY TEST");
        eprintln!("═══════════════════════════════════════════════════════════════════════\n");

        let model_path = std::path::Path::new(env!("HOME"))
            .join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found");
            return;
        }

        let mapped = MappedGGUFModel::from_path(model_path.to_str().unwrap())
            .expect("Failed to mmap GGUF");

        // Single token: just BOS
        let tokens = vec![mapped.model.bos_token_id().unwrap_or(1)];
        eprintln!("Token: {:?}", tokens);

        // CPU
        let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)
            .expect("Failed to load CPU model");
        let mut cpu_cache = realizar::gguf::OwnedQuantizedKVCache::from_config(
            cpu_model.config(), 10
        );
        let cpu_logits = cpu_model.forward_cached(tokens[0], &mut cpu_cache, 0)
            .expect("CPU forward failed");
        let cpu_top1 = cpu_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        // GPU
        std::env::set_var("SKIP_CUDA_GRAPH", "1");
        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(), 0, 10
        ).expect("Failed to create CUDA model");

        let gen_config = QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![],
        };

        let gpu_result = cuda_model.generate_full_cuda_with_cache(&tokens, &gen_config)
            .expect("GPU forward failed");
        std::env::remove_var("SKIP_CUDA_GRAPH");

        let gpu_top1 = if gpu_result.len() > 1 { gpu_result[1] } else { 0 };

        let cpu_decoded = mapped.model.decode(&[cpu_top1]);
        let gpu_decoded = mapped.model.decode(&[gpu_top1]);

        eprintln!("CPU: {} ({:?})", cpu_top1, cpu_decoded);
        eprintln!("GPU: {} ({:?})", gpu_top1, gpu_decoded);

        if cpu_top1 == gpu_top1 {
            eprintln!("✅ Single token: MATCH");
        } else {
            eprintln!("❌ Single token: DIVERGE (unexpected for position 0!)");
        }
    }
}
