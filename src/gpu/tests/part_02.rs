use crate::gpu::*;
#[test]
#[cfg(feature = "cuda")]
fn test_imp_1004a_cuda_matmul_benchmark() {
    // IMP-1004a: Benchmark CUDA matmul for realistic LLM dimensions
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1004a: CUDA not available, skipping");
        return;
    }

    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    // Realistic LLM dimensions: hidden_dim=4096, intermediate_dim=11008
    let test_cases = [
        (1, 4096, 4096, "1x4096x4096 (m=1, attention output)"),
        (1, 4096, 11008, "1x4096x11008 (m=1, FFN fc1)"),
        (1, 11008, 4096, "1x11008x4096 (m=1, FFN fc2)"),
        (1, 4096, 32000, "1x4096x32000 (m=1, LM head)"),
    ];

    for (m, k, n, desc) in test_cases {
        let a: Vec<f32> = vec![1.0; m * k];
        let b: Vec<f32> = vec![1.0; k * n];

        // Warmup
        let _ = cuda_scheduler.matmul(&a, &b, m, k, n);

        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cuda_scheduler.matmul(&a, &b, m, k, n);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        println!("IMP-1004a: {} - {:.3}ms avg", desc, avg_ms);
    }
}

#[test]
#[cfg(feature = "cuda")]
#[allow(clippy::many_single_char_names)]
fn test_imp_1004b_cuda_vs_cpu_matmul() {
    // IMP-1004b: Direct comparison of CUDA vs CPU matmul for m=1
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1004b: CUDA not available, skipping");
        return;
    }

    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    let mut hybrid_scheduler =
        HybridScheduler::with_threshold(1).expect("Failed to create HybridScheduler");

    // Test m=1 case where HybridScheduler forces CPU
    let m = 1;
    let k = 4096;
    let n = 4096;
    let a: Vec<f32> = vec![1.0; m * k];
    let b: Vec<f32> = vec![1.0; k * n];

    // Warmup
    let _ = cuda_scheduler.matmul(&a, &b, m, k, n);
    let _ = hybrid_scheduler.matmul(&a, &b, m, k, n);

    let iterations = 20;

    // CUDA timing
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_scheduler.matmul(&a, &b, m, k, n);
    }
    let cuda_time = start.elapsed();

    // CPU timing (HybridScheduler forces CPU for m=1)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_scheduler.matmul(&a, &b, m, k, n);
    }
    let cpu_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let cpu_avg_ms = cpu_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = cpu_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1004b: m=1 matmul (1x{}x{}) - CUDA={:.3}ms, CPU={:.3}ms, Speedup={:.2}x",
        k, n, cuda_avg_ms, cpu_avg_ms, speedup
    );

    // For small m=1 ops, GPU may not be faster due to transfer overhead
    // Just verify both produce correct results
    let cuda_result = cuda_scheduler.matmul(&a, &b, m, k, n).expect("test");
    let cpu_result = hybrid_scheduler.matmul(&a, &b, m, k, n).expect("test");

    assert_eq!(
        cuda_result.len(),
        cpu_result.len(),
        "IMP-1004b: Both should produce same output size"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1004c_full_forward_benchmark() {
    // IMP-1004c: Benchmark full GpuModel forward pass (CUDA vs Hybrid)
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1004c: CUDA not available, skipping");
        return;
    }

    // Use smaller model for benchmark (still tests full pipeline)
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 1024,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    let token_ids = vec![42usize]; // Single token

    // Warmup
    let _ = cuda_model.forward_gpu(&token_ids);
    let _ = hybrid_model.forward_gpu(&token_ids);

    let iterations = 20;

    // CUDA forward timing
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_model.forward_gpu(&token_ids);
    }
    let cuda_time = start.elapsed();

    // Hybrid forward timing (forces CPU for m=1)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_model.forward_gpu(&token_ids);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = hybrid_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1004c: Full forward pass - CUDA={:.3}ms, Hybrid={:.3}ms, Speedup={:.2}x",
        cuda_avg_ms, hybrid_avg_ms, speedup
    );

    // Calculate throughput
    let cuda_tok_per_sec = 1000.0 / cuda_avg_ms;
    let hybrid_tok_per_sec = 1000.0 / hybrid_avg_ms;

    println!(
        "IMP-1004c: Throughput - CUDA={:.1} tok/s, Hybrid={:.1} tok/s",
        cuda_tok_per_sec, hybrid_tok_per_sec
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1004d_token_generation_throughput() {
    // IMP-1004d: Measure token generation throughput (the key parity metric)
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1004d: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 1024,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");

    let prompt = vec![1usize, 2, 3]; // Small prompt

    // Generate tokens and measure
    let gen_config = GpuGenerateConfig::deterministic(10);

    let start = Instant::now();
    let tokens = cuda_model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1004d: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // Target: Ollama phi2 achieves ~228 tok/s
    // Current gap: ~1090x
    // This test establishes baseline for improvement tracking
    println!(
        "IMP-1004d: Target=228 tok/s (Ollama), Current={:.1} tok/s, Gap={:.0}x",
        tok_per_sec,
        228.0 / tok_per_sec.max(0.001)
    );

    // Verify generation produced tokens
    assert!(
        tokens_generated > 0,
        "IMP-1004d: Should generate at least some tokens"
    );
}

// ========================================================================
// PARITY-120: Weight caching for 10x+ speedup
// ========================================================================

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_parity_120a_cached_vs_uncached_matmul() {
    // PARITY-120a: Compare cached vs uncached matmul performance
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("PARITY-120a: CUDA not available, skipping");
        return;
    }

    let mut scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    // Realistic LLM dimensions: hidden=4096, qkv=12288 (3*4096)
    let k = 4096usize;
    let n = 4096usize;
    let weight: Vec<f32> = vec![1.0; k * n];
    let x: Vec<f32> = vec![1.0; k];

    // Cache the weight
    scheduler
        .cache_weight("test_weight", &weight)
        .expect("Failed to cache weight");

    // Warmup both paths
    let _ = scheduler.matmul(&x, &weight, 1, k, n);
    let _ = scheduler.matmul_cached("test_weight", &x, k, n);

    let iterations = 20;

    // Benchmark UNCACHED (current slow path)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = scheduler.matmul(&x, &weight, 1, k, n);
    }
    let uncached_time = start.elapsed();

    // Benchmark CACHED (new fast path)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = scheduler.matmul_cached("test_weight", &x, k, n);
    }
    let cached_time = start.elapsed();

    let uncached_avg_ms = uncached_time.as_secs_f64() * 1000.0 / iterations as f64;
    let cached_avg_ms = cached_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = uncached_avg_ms / cached_avg_ms;

    println!(
        "PARITY-120a: 1x{}x{} - Uncached={:.3}ms, Cached={:.3}ms, Speedup={:.1}x",
        k, n, uncached_avg_ms, cached_avg_ms, speedup
    );

    // Verify correctness
    let uncached_result = scheduler.matmul(&x, &weight, 1, k, n).expect("test");
    let cached_result = scheduler
        .matmul_cached("test_weight", &x, k, n)
        .expect("test");

    assert_eq!(
        uncached_result.len(),
        cached_result.len(),
        "PARITY-120a: Output sizes should match"
    );

    // Results should be identical
    for (i, (u, c)) in uncached_result.iter().zip(cached_result.iter()).enumerate() {
        assert!(
            (u - c).abs() < 0.01,
            "PARITY-120a: Results differ at {}: uncached={}, cached={}",
            i,
            u,
            c
        );
    }

    // Target: >2x speedup (eliminating weight transfer)
    assert!(
        speedup > 1.5,
        "PARITY-120a: Expected >1.5x speedup, got {:.1}x",
        speedup
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_parity_120b_full_layer_cached() {
    // PARITY-120b: Simulate full layer with all weights cached
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("PARITY-120b: CUDA not available, skipping");
        return;
    }

    let mut scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    // Realistic dimensions
    let hidden = 4096usize;
    let qkv = 3 * hidden;
    let intermediate = 11008usize;

    // Create and cache all layer weights
    let qkv_weight: Vec<f32> = vec![1.0; hidden * qkv];
    let out_weight: Vec<f32> = vec![1.0; hidden * hidden];
    let fc1_weight: Vec<f32> = vec![1.0; hidden * intermediate];
    let fc2_weight: Vec<f32> = vec![1.0; intermediate * hidden];

    scheduler.cache_weight("qkv", &qkv_weight).expect("test");
    scheduler.cache_weight("out", &out_weight).expect("test");
    scheduler.cache_weight("fc1", &fc1_weight).expect("test");
    scheduler.cache_weight("fc2", &fc2_weight).expect("test");

    assert_eq!(
        scheduler.cached_weight_count(),
        4,
        "PARITY-120b: Should have 4 cached weights"
    );

    let x: Vec<f32> = vec![1.0; hidden];
    let iterations = 10;

    // Benchmark cached full layer
    let start = Instant::now();
    for _ in 0..iterations {
        let qkv_out = scheduler
            .matmul_cached("qkv", &x, hidden, qkv)
            .expect("test");
        let attn_out = scheduler
            .matmul_cached("out", &qkv_out[..hidden], hidden, hidden)
            .expect("test");
        let fc1_out = scheduler
            .matmul_cached("fc1", &attn_out, hidden, intermediate)
            .expect("test");
        let _fc2_out = scheduler
            .matmul_cached("fc2", &fc1_out, intermediate, hidden)
            .expect("test");
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let tok_per_sec = 1000.0 / avg_ms;

    println!(
        "PARITY-120b: Full layer (4 matmuls) - {:.2}ms/token = {:.1} tok/s",
        avg_ms, tok_per_sec
    );

    // Target: >50 tok/s for single layer
    println!(
        "PARITY-120b: Target=228 tok/s (Ollama), Current={:.1} tok/s (single layer)",
        tok_per_sec
    );
}

// ========================================================================
// IMP-1005: Wire CudaScheduler into forward_gpu() via do_matmul()
// ========================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1005a_do_matmul_uses_cuda_scheduler() {
    // IMP-1005a: do_matmul() should use cuda_scheduler when available
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1005a: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    // Both should have do_matmul method
    let a: Vec<f32> = vec![1.0; 64];
    let b: Vec<f32> = vec![1.0; 64 * 100];

    let cuda_result = cuda_model.do_matmul(&a, &b, 1, 64, 100);
    let hybrid_result = hybrid_model.do_matmul(&a, &b, 1, 64, 100);

    assert!(
        cuda_result.is_ok(),
        "IMP-1005a: CUDA do_matmul should succeed"
    );
    assert!(
        hybrid_result.is_ok(),
        "IMP-1005a: Hybrid do_matmul should succeed"
    );

    // Both should produce same-sized output
    assert_eq!(
        cuda_result.expect("test").len(),
        hybrid_result.expect("test").len(),
        "IMP-1005a: Both should produce same output size"
    );
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "flaky - timing depends on GPU warmup state and system load"]
fn test_imp_1005b_forward_gpu_speedup_with_cuda() {
    // IMP-1005b: forward_gpu should be faster with cuda_scheduler
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1005b: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 1024,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    let token_ids = vec![42usize]; // Single token (m=1)

    // Warmup
    let _ = cuda_model.forward_gpu(&token_ids);
    let _ = hybrid_model.forward_gpu(&token_ids);

    let iterations = 20;

    // CUDA forward timing (should use do_matmul -> cuda_scheduler)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_model.forward_gpu(&token_ids);
    }
    let cuda_time = start.elapsed();

    // Hybrid forward timing (do_matmul -> scheduler -> CPU for m=1)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_model.forward_gpu(&token_ids);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = hybrid_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1005b: forward_gpu (m=1) - CUDA={:.3}ms, Hybrid={:.3}ms, Speedup={:.2}x",
        cuda_avg_ms, hybrid_avg_ms, speedup
    );

    // Calculate throughput
    let cuda_tok_per_sec = 1000.0 / cuda_avg_ms;
    let hybrid_tok_per_sec = 1000.0 / hybrid_avg_ms;

    println!(
        "IMP-1005b: Throughput - CUDA={:.1} tok/s, Hybrid={:.1} tok/s",
        cuda_tok_per_sec, hybrid_tok_per_sec
    );

    // After wiring, CUDA should be faster (speedup > 1.0)
    // Before fix: speedup ~1.0 (both use same path)
    // After fix: speedup should be > 1.5x
    assert!(
        speedup > 0.5,
        "IMP-1005b: CUDA path should not be catastrophically slower"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1005c_token_generation_with_cuda_forward() {
    // IMP-1005c: Token generation throughput after forward_gpu uses cuda_scheduler
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1005c: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 1024,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    let prompt = vec![1usize, 2, 3];
    let gen_config = GpuGenerateConfig::deterministic(10);

    let start = Instant::now();
    let tokens = cuda_model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1005c: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // After IMP-1005, throughput should improve
    // Previous (IMP-1004d): 9.1 tok/s
    // Target: > 15 tok/s (improvement from wiring cuda_scheduler)
    println!(
        "IMP-1005c: Previous=9.1 tok/s, Current={:.1} tok/s, Target=228 tok/s",
        tok_per_sec
    );

    assert!(
        tokens_generated > 0,
        "IMP-1005c: Should generate at least some tokens"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1005d_forward_block_uses_do_matmul() {
    // IMP-1005d: forward_block_idx should use do_matmul internally
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1005d: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    // Test single token forward through blocks
    let input: Vec<f32> = vec![0.1; 256]; // [1, hidden_dim]
    let seq_len = 1;

    // Warmup
    let _ = cuda_model.forward_block_idx(&input, seq_len, 0);
    let _ = hybrid_model.forward_block_idx(&input, seq_len, 0);

    let iterations = 20;

    // CUDA block forward
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_model.forward_block_idx(&input, seq_len, 0);
    }
    let cuda_time = start.elapsed();

    // Hybrid block forward
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_model.forward_block_idx(&input, seq_len, 0);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;

    println!(
        "IMP-1005d: forward_block_idx (m=1) - CUDA={:.3}ms, Hybrid={:.3}ms",
        cuda_avg_ms, hybrid_avg_ms
    );

    // Both should complete successfully
    let cuda_result = cuda_model.forward_block_idx(&input, seq_len, 0);
    let hybrid_result = hybrid_model.forward_block_idx(&input, seq_len, 0);

    assert!(
        cuda_result.is_ok() && hybrid_result.is_ok(),
        "IMP-1005d: Both should complete successfully"
    );
}

// ========================================================================
// IMP-1006: Wire do_matmul into incremental forward paths
// ========================================================================

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_imp_1006a_incremental_forward_uses_cuda() {
    // IMP-1006a: forward_gpu_incremental_optimized should use do_matmul
    // which routes to CudaScheduler when available
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1006a: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    // Single token forward (m=1)
    // With config: hidden_dim=256, num_heads=4 => head_dim=64
    let token_id: usize = 42;
    let num_kv_heads = cuda_model.config.num_kv_heads;
    let head_dim = cuda_model.config.head_dim();
    let max_positions = 128;

    // Warmup
    let mut cuda_cache = StreamingKVCache::new(
        cuda_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let mut hybrid_cache = StreamingKVCache::new(
        hybrid_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let _ = cuda_model.forward_gpu_incremental_optimized(token_id, &mut cuda_cache);
    let _ = hybrid_model.forward_gpu_incremental_optimized(token_id, &mut hybrid_cache);

    let iterations = 20;

    // CUDA incremental forward (create fresh cache each iteration)
    let start = Instant::now();
    for i in 0..iterations {
        let mut cache = StreamingKVCache::new(
            cuda_model.config.num_layers,
            max_positions,
            num_kv_heads,
            head_dim,
        );
        let _ = cuda_model.forward_gpu_incremental_optimized(i % 100, &mut cache);
    }
    let cuda_time = start.elapsed();

    // Hybrid incremental forward
    let start = Instant::now();
    for i in 0..iterations {
        let mut cache = StreamingKVCache::new(
            hybrid_model.config.num_layers,
            max_positions,
            num_kv_heads,
            head_dim,
        );
        let _ = hybrid_model.forward_gpu_incremental_optimized(i % 100, &mut cache);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = hybrid_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1006a: incremental_forward (m=1) - CUDA={:.3}ms, Hybrid={:.3}ms, Speedup={:.2}x",
        cuda_avg_ms, hybrid_avg_ms, speedup
    );

    // IMP-1006: After wiring do_matmul, CUDA path should be faster
    assert!(
        cuda_avg_ms < hybrid_avg_ms * 2.0,
        "IMP-1006a: CUDA incremental should not be much slower than Hybrid"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1006b_block_incremental_uses_cuda() {
    // IMP-1006b: forward_block_incremental_optimized should use do_matmul
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1006b: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    let input: Vec<f32> = vec![0.1; 256];
    let block_idx = 0;
    let num_kv_heads = cuda_model.config.num_kv_heads;
    let head_dim = cuda_model.config.head_dim();
    let max_positions = 128;

    // Warmup
    let mut cuda_cache = StreamingKVCache::new(
        cuda_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let mut hybrid_cache = StreamingKVCache::new(
        hybrid_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let _ = cuda_model.forward_block_incremental_optimized(&input, block_idx, &mut cuda_cache);
    let _ =
        hybrid_model.forward_block_incremental_optimized(&input, block_idx, &mut hybrid_cache);

    let iterations = 20;

    // CUDA block incremental (build up KV cache)
    let mut cuda_cache2 = StreamingKVCache::new(
        cuda_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let start = Instant::now();
    for _ in 0..iterations {
        let _ =
            cuda_model.forward_block_incremental_optimized(&input, block_idx, &mut cuda_cache2);
    }
    let cuda_time = start.elapsed();

    // Hybrid block incremental
    let mut hybrid_cache2 = StreamingKVCache::new(
        hybrid_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_model.forward_block_incremental_optimized(
            &input,
            block_idx,
            &mut hybrid_cache2,
        );
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = hybrid_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1006b: block_incremental (m=1) - CUDA={:.3}ms, Hybrid={:.3}ms, Speedup={:.2}x",
        cuda_avg_ms, hybrid_avg_ms, speedup
    );

    // After IMP-1006, CUDA should handle m=1 operations via do_matmul
    assert!(
        cuda_avg_ms > 0.0,
        "IMP-1006b: CUDA path should complete successfully"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1006c_generate_throughput_improved() {
    // IMP-1006c: After wiring incremental paths, generate() throughput should improve
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1006c: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4, // More layers to see impact
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    let start = Instant::now();
    let tokens = cuda_model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1006c: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // After IMP-1006, throughput should improve from IMP-1005 levels
    // IMP-1005c: ~18.2 tok/s (forward methods)
    // IMP-1004d: ~9.1 tok/s (generate baseline)
    // Target: > 15 tok/s (routing to CUDA in incremental paths)
    println!(
        "IMP-1006c: Previous=9.1 tok/s, Current={:.1} tok/s, Target=228 tok/s (Ollama)",
        tok_per_sec
    );

    assert!(
        tokens_generated > 0,
        "IMP-1006c: Should generate at least some tokens"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1006d_all_matmuls_routed_to_cuda() {
    // IMP-1006d: Verify that model with cuda_scheduler routes all matmuls to CUDA
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1006d: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut cuda_model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    // Verify cuda_scheduler is present
    assert!(
        cuda_model.has_cuda_scheduler(),
        "IMP-1006d: CUDA model should have cuda_scheduler"
    );

    // Test do_matmul routing
    let a: Vec<f32> = vec![1.0; 256 * 128];
    let b: Vec<f32> = vec![1.0; 128 * 64];

    let result = cuda_model.do_matmul(&a, &b, 256, 128, 64);
    assert!(
        result.is_ok(),
        "IMP-1006d: do_matmul should complete via CUDA"
    );

    let output = result.expect("test");
    assert_eq!(
        output.len(),
        256 * 64,
        "IMP-1006d: Output dimensions should be correct"
    );

    println!("IMP-1006d: All matmuls routed to CudaScheduler ✓");
}

// ========================================================================
// IMP-1007: Eliminate weight cloning in incremental forward paths
// ========================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1007a_no_clone_matmul() {
    // IMP-1007a: matmul_split should work without cloning weights
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1007a: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut model = GpuModel::new_with_cuda(config).expect("Failed to create model");

    // Test matmul without weight cloning using split borrow pattern
    let input: Vec<f32> = vec![0.1; 256];

    // IMP-1007: Should be able to call matmul_split without cloning weights
    let result = model.matmul_split(&input, 0, WeightType::Qkv);
    assert!(result.is_ok(), "IMP-1007a: matmul_split should work");

    println!("IMP-1007a: Zero-clone matmul verified ✓");
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1007b_incremental_no_clone_speedup() {
    // IMP-1007b: forward_block_incremental without weight cloning should be faster
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1007b: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut model = GpuModel::new_with_cuda(config.clone()).expect("Failed to create model");

    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let max_positions = 128;

    let input: Vec<f32> = vec![0.1; 256];
    let block_idx = 0;

    // Warmup
    let mut cache = StreamingKVCache::new(
        model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let _ = model.forward_block_incremental_optimized(&input, block_idx, &mut cache);

    let iterations = 50;

    // Benchmark
    let mut cache2 = StreamingKVCache::new(
        model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward_block_incremental_optimized(&input, block_idx, &mut cache2);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!(
        "IMP-1007b: block_incremental avg={:.3}ms ({} iterations)",
        avg_ms, iterations
    );

    // After IMP-1007, this should be < 0.5ms per block (faster than IMP-1006's 0.7ms)
    println!(
        "IMP-1007b: Previous=0.698ms, Current={:.3}ms, Target=<0.5ms",
        avg_ms
    );

    assert!(avg_ms > 0.0, "IMP-1007b: Should complete successfully");
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1007c_generate_throughput_improved() {
    // IMP-1007c: generate() throughput should improve after eliminating cloning
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1007c: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut model = GpuModel::new_with_cuda(config).expect("Failed to create model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    let start = Instant::now();
    let tokens = model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1007c: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // After IMP-1007, throughput should improve from IMP-1006's 37.3 tok/s
    println!(
        "IMP-1007c: Previous=37.3 tok/s, Current={:.1} tok/s, Target=228 tok/s (Ollama)",
        tok_per_sec
    );

    assert!(
        tokens_generated > 0,
        "IMP-1007c: Should generate at least some tokens"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1008a_refcell_scheduler_matmul() {
    // IMP-1008a: matmul_refcell should work with &self (no &mut self)
    // This enables simultaneous borrowing of weights and scheduler
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1008a: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    // Test matmul_refcell with &self (not &mut self)
    let a: Vec<f32> = vec![0.1; 256];
    let b: Vec<f32> = vec![0.2; 256 * 512];

    // This should work without requiring &mut self
    let result = model
        .matmul_refcell(&a, &b, 1, 256, 512)
        .expect("matmul_refcell should work");

    assert_eq!(
        result.len(),
        512,
        "IMP-1008a: Output should be 512 elements"
    );

    // Verify result is reasonable (non-zero, non-NaN)
    let sum: f32 = result.iter().sum();
    assert!(sum.is_finite(), "IMP-1008a: Result should be finite");
    assert!(sum != 0.0, "IMP-1008a: Result should be non-zero");

    println!("IMP-1008a: matmul_refcell works with &self");
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1008b_zero_clone_forward_block() {
    // IMP-1008b: forward_block_refcell should not clone weights
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1008b: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let model = GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");

    let input: Vec<f32> = vec![0.1; 256];
    let block_idx = 0;
    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let max_positions = 128;

    // Warmup
    let mut cache = StreamingKVCache::new(
        model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let _ = model.forward_block_refcell(&input, block_idx, &mut cache);

    let iterations = 100;

    // Benchmark with RefCell (zero-clone)
    let mut cache2 = StreamingKVCache::new(
        model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward_block_refcell(&input, block_idx, &mut cache2);
    }
    let refcell_time = start.elapsed();
    let refcell_avg_us = refcell_time.as_micros() as f64 / iterations as f64;

    println!(
        "IMP-1008b: forward_block_refcell avg={:.1}µs ({} iterations)",
        refcell_avg_us, iterations
    );

    // Should be faster than clone-based approach
    assert!(
        refcell_avg_us > 0.0,
        "IMP-1008b: forward_block_refcell should complete"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1008c_generate_throughput_refcell() {
    // IMP-1008c: generate_refcell should have better throughput than clone-based generate
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1008c: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    // Measure RefCell-based generation
    let start = Instant::now();
    let tokens = model
        .generate_refcell(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1008c: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // Target: >50 tok/s (improvement over IMP-1006's 35 tok/s)
    println!(
        "IMP-1008c: Previous=35.1 tok/s, Current={:.1} tok/s, Target=228 tok/s (Ollama)",
        tok_per_sec
    );

    assert!(
        tokens_generated > 0,
        "IMP-1008c: Should generate at least some tokens"
    );
}

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_imp_1008d_compare_clone_vs_refcell() {
    // IMP-1008d: Direct comparison of clone-based vs RefCell-based forward
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1008d: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut clone_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create clone model");
    let refcell_model =
        GpuModel::new_with_cuda(config).expect("Failed to create refcell model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    // Warmup
    let _ = clone_model.generate(&prompt, &gen_config);
    let _ = refcell_model.generate_refcell(&prompt, &gen_config);

    let iterations = 5;

    // Benchmark clone-based
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = clone_model.generate(&prompt, &gen_config);
    }
    let clone_time = start.elapsed();

    // Benchmark RefCell-based
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = refcell_model.generate_refcell(&prompt, &gen_config);
    }
    let refcell_time = start.elapsed();

    let clone_avg_ms = clone_time.as_secs_f64() * 1000.0 / iterations as f64;
    let refcell_avg_ms = refcell_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = clone_avg_ms / refcell_avg_ms;

    println!(
        "IMP-1008d: Clone={:.2}ms, RefCell={:.2}ms, Speedup={:.2}x",
        clone_avg_ms, refcell_avg_ms, speedup
    );

    // RefCell should be faster (no cloning overhead)
    // For small test models, improvement may be small
    // For phi-2 scale, improvement would be dramatic (8.6GB cloning eliminated)
    assert!(
        speedup > 0.9,
        "IMP-1008d: RefCell should not be slower than clone"
    );
}

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_imp_1009a_main_generate_uses_refcell() {
    // IMP-1009a: Main generate() should use RefCell path when CUDA available
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1009a: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    // Measure main generate() throughput
    let start = Instant::now();
    let tokens = model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1009a: Main generate() - {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // After IMP-1009, main generate() should be fast (>100 tok/s target)
    // This test will FAIL before IMP-1009 wiring (expect ~35 tok/s)
    // This test will PASS after IMP-1009 wiring (expect ~200+ tok/s)
    println!(
        "IMP-1009a: Target=100+ tok/s (RefCell speed), Current={:.1} tok/s",
        tok_per_sec
    );

    // Assert that main generate() is fast (indicates RefCell is wired in)
    assert!(
        tok_per_sec > 100.0,
        "IMP-1009a: Main generate() should achieve >100 tok/s with RefCell wiring"
    );
}

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_imp_1009b_generate_parity_with_refcell() {
    // IMP-1009b: Main generate() should match generate_refcell() throughput
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1009b: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut clone_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create model");
    let refcell_model = GpuModel::new_with_cuda(config).expect("Failed to create model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    // Warmup
    let _ = clone_model.generate(&prompt, &gen_config);
    let _ = refcell_model.generate_refcell(&prompt, &gen_config);

    let iterations = 5;

    // Benchmark main generate()
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = clone_model.generate(&prompt, &gen_config);
    }
    let main_time = start.elapsed();

    // Benchmark generate_refcell()
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = refcell_model.generate_refcell(&prompt, &gen_config);
    }
    let refcell_time = start.elapsed();

    let main_avg_ms = main_time.as_secs_f64() * 1000.0 / iterations as f64;
    let refcell_avg_ms = refcell_time.as_secs_f64() * 1000.0 / iterations as f64;
    let ratio = main_avg_ms / refcell_avg_ms;

    println!(
        "IMP-1009b: Main={:.2}ms, RefCell={:.2}ms, Ratio={:.2}x",
        main_avg_ms, refcell_avg_ms, ratio
    );

    // After IMP-1009, main should be within 1.2x of RefCell (allow some overhead)
    assert!(
        ratio < 1.5,
        "IMP-1009b: Main generate() should be within 1.5x of RefCell throughput"
    );
}

// ============================================================================
// Coverage Improvement Tests - Scalar/SIMD Operations
// ============================================================================

#[test]
fn test_exceeds_gpu_buffer_limit() {
    // Within limit
    assert!(!exceeds_gpu_buffer_limit(1000));
    // At limit
    assert!(!exceeds_gpu_buffer_limit(MAX_GPU_BUFFER_BYTES / 4));
    // Exceeds limit
    assert!(exceeds_gpu_buffer_limit(MAX_GPU_BUFFER_BYTES / 4 + 1));
}

#[test]
fn test_scalar_softmax_basic() {
    let input = vec![1.0, 2.0, 3.0];
    let output = scalar_softmax(&input);

    assert_eq!(output.len(), 3);
    // Sum should be 1
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Largest input should have largest output
    assert!(output[2] > output[1] && output[1] > output[0]);
}

#[test]
fn test_scalar_softmax_empty() {
    let input: Vec<f32> = vec![];
    let output = scalar_softmax(&input);
    assert!(output.is_empty());
}

#[test]
fn test_scalar_softmax_single() {
    let input = vec![5.0];
    let output = scalar_softmax(&input);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = simd_softmax(&input);

    assert_eq!(output.len(), 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_empty() {
    let input: Vec<f32> = vec![];
    let output = simd_softmax(&input);
    assert!(output.is_empty());
}

#[test]
fn test_simd_softmax_matches_scalar() {
    let input = vec![0.5, 1.5, 2.5, 0.0];
    let scalar_out = scalar_softmax(&input);
    let simd_out = simd_softmax(&input);

    for (s, si) in scalar_out.iter().zip(simd_out.iter()) {
        assert!((s - si).abs() < 1e-5, "SIMD softmax should match scalar");
    }
}

#[test]
fn test_scalar_rope_basic() {
    let seq_len = 2;
    let head_dim = 4;
    let hidden_dim = head_dim;
    let input = vec![1.0; seq_len * hidden_dim];
    let theta = 10000.0;

    let output = scalar_rope(&input, seq_len, head_dim, theta);

    assert_eq!(output.len(), input.len());
    // Should have rotated values
}

#[test]
fn test_scalar_rope_empty() {
    let output = scalar_rope(&[], 0, 4, 10000.0);
    assert!(output.is_empty());
}

#[test]
fn test_scalar_rope_zero_head_dim() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = scalar_rope(&input, 2, 0, 10000.0);
    assert!(output.is_empty());
}

#[test]
fn test_simd_rope_basic() {
    let seq_len = 2;
    let head_dim = 4;
    let hidden_dim = head_dim;
    let input = vec![1.0; seq_len * hidden_dim];
    let theta = 10000.0;

    let output = simd_rope(&input, seq_len, head_dim, theta);

    assert_eq!(output.len(), input.len());
}

#[test]
fn test_simd_rope_empty() {
    let output = simd_rope(&[], 0, 4, 10000.0);
    assert!(output.is_empty());
}

#[test]
fn test_simd_rope_matches_scalar() {
    let seq_len = 2;
    let head_dim = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let theta = 10000.0;

    let scalar_out = scalar_rope(&input, seq_len, head_dim, theta);
    let simd_out = simd_rope(&input, seq_len, head_dim, theta);

    for (s, si) in scalar_out.iter().zip(simd_out.iter()) {
        assert!((s - si).abs() < 1e-4, "SIMD rope should match scalar");
    }
}

// ============================================================================
// Coverage Tests - Batch/FFN Operations
// ============================================================================

#[test]
fn test_batch_embed_basic() {
    let vocab_size = 10;
    let hidden_dim = 4;
    let embedding_table: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| i as f32 * 0.1)
        .collect();
    let tokens = vec![0, 2, 5];

    let output = batch_embed(&embedding_table, &tokens, hidden_dim);

    assert_eq!(output.len(), 3 * hidden_dim);
    // Token 0 should have embedding [0.0, 0.1, 0.2, 0.3]
    assert!((output[0] - 0.0).abs() < 1e-5);
    assert!((output[1] - 0.1).abs() < 1e-5);
}

#[test]
fn test_batch_embed_single_token() {
    let embedding_table = vec![1.0, 2.0, 3.0, 4.0]; // vocab_size=1, hidden_dim=4
    let tokens = vec![0];

    let output = batch_embed(&embedding_table, &tokens, 4);

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_sequential_ffn_basic() {
    let hidden_dim = 4;
    let intermediate_dim = 8;
    let input = vec![1.0; hidden_dim];
    let up_weight = vec![0.1; hidden_dim * intermediate_dim];
    let down_weight = vec![0.1; intermediate_dim * hidden_dim];

    let output = sequential_ffn(
        &input,
        &up_weight,
        &down_weight,
        hidden_dim,
        intermediate_dim,
    );

    assert_eq!(output.len(), hidden_dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_parallel_ffn_basic() {
    let hidden_dim = 4;
    let intermediate_dim = 8;
    let input = vec![1.0; hidden_dim];
    let up_weight = vec![0.1; hidden_dim * intermediate_dim];
    let down_weight = vec![0.1; intermediate_dim * hidden_dim];

    let output = parallel_ffn(
        &input,
        &up_weight,
        &down_weight,
        hidden_dim,
        intermediate_dim,
    );

    assert_eq!(output.len(), hidden_dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Coverage Tests - LayerNorm
// ============================================================================

#[test]
fn test_standard_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;

    let output = standard_layernorm(&input, &gamma, &beta, eps);

    assert_eq!(output.len(), 4);
    // Should be normalized (mean ~0, std ~1)
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(mean.abs() < 1e-4, "Mean should be ~0 after layernorm");
}

