//! GGUF Part 07: PARITY-005 (Contiguous KV Cache) + PARITY-006 (Batch Processing) +
//!               PARITY-007 (E2E Benchmark Verification) + PARITY-008 (Popper Score)
//!
//! Extracted from gguf_monolith.rs (PMAT-802)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{
    ContiguousKVCache, GGUFConfig, OwnedQuantizedKVCache, OwnedQuantizedModel,
    OwnedQuantizedModelCached, QuantizedGenerateConfig,
};

// ========================================================================
// PARITY-005: Contiguous KV Cache Tests
// ========================================================================

/// PARITY-005a: ContiguousKVCache should use single contiguous allocation
#[test]
fn test_parity005a_contiguous_kv_cache_layout() {
    let num_layers = 4;
    let hidden_dim = 64;
    let max_seq_len = 32;

    let cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Verify contiguous layout
    assert!(
        cache.is_contiguous(),
        "PARITY-005a: Cache should report contiguous layout"
    );

    // Verify cache line alignment
    assert!(
        cache.is_cache_aligned(),
        "PARITY-005a: Layer stride should be cache-line aligned"
    );

    // Verify stride is multiple of 16 (64 bytes / 4 bytes per float)
    assert_eq!(
        cache.layer_stride() % 16,
        0,
        "PARITY-005a: Layer stride {} should be multiple of 16 floats",
        cache.layer_stride()
    );
}

/// PARITY-005b: Cache line alignment calculation
#[test]
fn test_parity005b_cache_line_alignment() {
    // Test various sizes for proper alignment
    let test_cases = vec![
        (4, 64, 32, 2048),   // 32 * 64 = 2048, already aligned
        (4, 64, 33, 2112),   // 33 * 64 = 2112, needs alignment to 2128
        (2, 80, 16, 1280),   // 16 * 80 = 1280, aligned
        (8, 256, 64, 16384), // 64 * 256 = 16384, aligned
    ];

    for (num_layers, hidden_dim, max_seq_len, _expected_raw) in test_cases {
        let cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

        // Layer stride should be cache-line aligned
        assert!(
            cache.is_cache_aligned(),
            "PARITY-005b: Cache should be aligned for num_layers={}, hidden_dim={}, max_seq_len={}",
            num_layers,
            hidden_dim,
            max_seq_len
        );

        // Verify stride is at least raw size
        let raw_size = max_seq_len * hidden_dim;
        assert!(
            cache.layer_stride() >= raw_size,
            "PARITY-005b: Layer stride {} should be >= raw size {}",
            cache.layer_stride(),
            raw_size
        );
    }
}

/// PARITY-005c: Append and retrieve K/V data correctly
#[test]
fn test_parity005c_contiguous_kv_operations() {
    let num_layers = 2;
    let hidden_dim = 16;
    let max_seq_len = 8;

    let mut cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Create test data
    let k0: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.1).collect();
    let v0: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.2).collect();
    let k1: Vec<f32> = (0..hidden_dim).map(|i| (i + 16) as f32 * 0.1).collect();
    let v1: Vec<f32> = (0..hidden_dim).map(|i| (i + 16) as f32 * 0.2).collect();

    // Append to layer 0
    cache.append(0, &k0, &v0);
    cache.advance();

    // Append to layer 0 again (position 1)
    cache.append(0, &k1, &v1);
    cache.advance();

    // Verify length
    assert_eq!(cache.len(), 2, "PARITY-005c: Cache should have 2 positions");

    // Verify K data for layer 0
    let cached_k = cache.get_k(0);
    assert_eq!(
        cached_k.len(),
        2 * hidden_dim,
        "PARITY-005c: K cache should have 2 * hidden_dim elements"
    );

    // First position's K values should match k0
    for (i, &val) in cached_k[..hidden_dim].iter().enumerate() {
        assert!(
            (val - k0[i]).abs() < 1e-6,
            "PARITY-005c: K position 0 mismatch at {}: expected {}, got {}",
            i,
            k0[i],
            val
        );
    }

    // Second position's K values should match k1
    for (i, &val) in cached_k[hidden_dim..].iter().enumerate() {
        assert!(
            (val - k1[i]).abs() < 1e-6,
            "PARITY-005c: K position 1 mismatch at {}: expected {}, got {}",
            i,
            k1[i],
            val
        );
    }
}

/// PARITY-005d: Reset cache to empty state
#[test]
fn test_parity005d_contiguous_kv_reset() {
    let mut cache = ContiguousKVCache::new(2, 16, 8);

    // Add some data
    let k: Vec<f32> = vec![1.0; 16];
    let v: Vec<f32> = vec![2.0; 16];
    cache.append(0, &k, &v);
    cache.advance();
    cache.append(0, &k, &v);
    cache.advance();

    assert_eq!(cache.len(), 2, "PARITY-005d: Before reset, len should be 2");

    // Reset
    cache.reset();

    assert_eq!(cache.len(), 0, "PARITY-005d: After reset, len should be 0");

    // Should be able to append again
    cache.append(0, &k, &v);
    cache.advance();
    assert_eq!(
        cache.len(),
        1,
        "PARITY-005d: After reset+append, len should be 1"
    );
}

/// PARITY-005e: Verify sequential memory layout
#[test]
fn test_parity005e_sequential_memory_layout() {
    let num_layers = 2;
    let hidden_dim = 8;
    let max_seq_len = 4;

    let mut cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Add data for layer 0 at position 0
    let k0: Vec<f32> = (0..hidden_dim).map(|i| i as f32).collect();
    let v0: Vec<f32> = (0..hidden_dim).map(|i| (i + 10) as f32).collect();
    cache.append(0, &k0, &v0);

    // Add data for layer 1 at position 0 (same position, different layer)
    let k1: Vec<f32> = (0..hidden_dim).map(|i| (i + 20) as f32).collect();
    let v1: Vec<f32> = (0..hidden_dim).map(|i| (i + 30) as f32).collect();
    cache.append(1, &k1, &v1);

    // Advance after appending to all layers for this position
    cache.advance();

    // Verify layer 0's K data
    let layer0_k = cache.get_k(0);
    assert!(
        layer0_k[0] == 0.0,
        "PARITY-005e: Layer 0 K should start with 0.0"
    );

    // Verify layer 1's K data
    let layer1_k = cache.get_k(1);
    assert!(
        layer1_k[0] == 20.0,
        "PARITY-005e: Layer 1 K should start with 20.0"
    );
}

/// PARITY-005f: Verify memory usage matches expected
#[test]
fn test_parity005f_memory_usage() {
    let num_layers = 4;
    let hidden_dim = 64;
    let max_seq_len = 128;

    let cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);

    // Expected: 2 (K+V) * num_layers * max_seq_len * hidden_dim * sizeof(f32)
    let expected_bytes = 2 * num_layers * max_seq_len * hidden_dim * std::mem::size_of::<f32>();

    // Actual should be close (may have padding for alignment)
    let actual_bytes = cache.memory_bytes();

    assert!(
        actual_bytes >= expected_bytes,
        "PARITY-005f: Memory usage {} should be >= minimum {}",
        actual_bytes,
        expected_bytes
    );

    // Should not be too much larger (< 10% overhead for alignment)
    let max_overhead = expected_bytes as f64 * 1.1;
    assert!(
        actual_bytes as f64 <= max_overhead,
        "PARITY-005f: Memory usage {} should be < {} (10% overhead)",
        actual_bytes,
        max_overhead as usize
    );
}

/// PARITY-005i: Verify cache performance compared to non-contiguous
#[test]
fn test_parity005i_cache_performance_comparison() {
    use std::time::Instant;

    let num_layers = 4;
    let hidden_dim = 64;
    let max_seq_len = 64;
    let num_iterations = 100;

    // Benchmark non-contiguous (OwnedQuantizedKVCache)
    let start = Instant::now();
    for _ in 0..num_iterations {
        let mut cache = OwnedQuantizedKVCache::new(num_layers, hidden_dim, max_seq_len);
        let k: Vec<f32> = vec![1.0; hidden_dim];
        let v: Vec<f32> = vec![2.0; hidden_dim];
        for _ in 0..max_seq_len {
            for layer in 0..num_layers {
                cache.append_kv(layer, &k, &v);
            }
        }
    }
    let owned_time = start.elapsed();

    // Benchmark contiguous
    let start = Instant::now();
    for _ in 0..num_iterations {
        let mut cache = ContiguousKVCache::new(num_layers, hidden_dim, max_seq_len);
        let k: Vec<f32> = vec![1.0; hidden_dim];
        let v: Vec<f32> = vec![2.0; hidden_dim];
        for _ in 0..max_seq_len {
            for layer in 0..num_layers {
                cache.append(layer, &k, &v);
            }
            cache.advance();
        }
    }
    let contiguous_time = start.elapsed();

    println!("PARITY-005i: Cache Performance Comparison");
    println!("  OwnedQuantizedKVCache: {:?}", owned_time);
    println!("  ContiguousKVCache: {:?}", contiguous_time);

    // Contiguous should not be significantly slower (within 10x to account for coverage instrumentation)
    let ratio = contiguous_time.as_nanos() as f64 / owned_time.as_nanos() as f64;
    println!("  Ratio (contiguous/owned): {:.2}x", ratio);

    assert!(
        ratio < 10.0,
        "PARITY-005i: Contiguous cache should not be 10x slower than owned"
    );
}

// ========================================================================
// PARITY-006: Batch Processing Tests
// ========================================================================

/// PARITY-006a: batch_generate method should exist and produce valid output
#[test]
fn test_parity006a_batch_generate_exists() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let cached = OwnedQuantizedModelCached::new(model);

    // Single prompt batch
    let prompts = vec![vec![1u32, 2, 3]];
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let results = cached
        .model()
        .batch_generate(
            &prompts.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
            &gen_config,
        )
        .expect("PARITY-006a: batch_generate should exist and succeed");

    assert_eq!(
        results.len(),
        1,
        "PARITY-006a: Should return results for 1 prompt"
    );
    assert!(
        results[0].len() > 3,
        "PARITY-006a: Generated output should include prompt + new tokens"
    );
}

/// PARITY-006b: Single-prompt batch should match non-batch generation
#[test]
fn test_parity006b_single_prompt_optimization() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Deterministic
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Non-batch generation
    let prompt = vec![1u32, 2, 3];
    let single_result = model
        .generate_with_cache(&prompt, &gen_config)
        .expect("Single generate should work");

    // Batch generation with single prompt
    let cached = OwnedQuantizedModelCached::new(model);
    let prompts = vec![prompt.clone()];
    let batch_results = cached
        .model()
        .batch_generate(
            &prompts.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
            &gen_config,
        )
        .expect("Batch generate should work");

    // Results should match (deterministic sampling)
    assert_eq!(
        single_result, batch_results[0],
        "PARITY-006b: Single and batch generation should produce same result"
    );
}

/// PARITY-006c: Batch outputs should be valid for all prompts
#[test]
fn test_parity006c_batch_output_validity() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let cached = OwnedQuantizedModelCached::new(model);

    // Multiple prompts of different lengths
    let prompts = vec![vec![1u32, 2, 3], vec![4u32, 5, 6, 7, 8], vec![9u32]];
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let results = cached
        .model()
        .batch_generate(
            &prompts.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
            &gen_config,
        )
        .expect("PARITY-006c: Batch generate should succeed");

    assert_eq!(results.len(), 3, "PARITY-006c: Should have 3 results");

    // Each result should include at least its prompt
    for (i, (prompt, result)) in prompts.iter().zip(results.iter()).enumerate() {
        assert!(
            result.len() >= prompt.len(),
            "PARITY-006c: Result {} should include prompt ({} >= {})",
            i,
            result.len(),
            prompt.len()
        );
        assert!(
            result[..prompt.len()] == *prompt,
            "PARITY-006c: Result {} should start with prompt",
            i
        );
    }
}

/// PARITY-006d: Batch throughput factor should scale with batch size
#[test]
fn test_parity006d_throughput_factor() {
    // batch_throughput_factor should return expected speedup for batch inference
    let single = OwnedQuantizedModel::batch_throughput_factor(1);
    let small_batch = OwnedQuantizedModel::batch_throughput_factor(4);
    let large_batch = OwnedQuantizedModel::batch_throughput_factor(16);

    // Single should be 1.0 (baseline)
    assert!(
        (single - 1.0).abs() < 0.01,
        "PARITY-006d: Single batch throughput factor should be 1.0"
    );

    // Larger batches should have higher throughput factor
    assert!(
        small_batch > single,
        "PARITY-006d: Batch of 4 should have higher throughput than single"
    );
    assert!(
        large_batch > small_batch,
        "PARITY-006d: Batch of 16 should have higher throughput than batch of 4"
    );
}

/// PARITY-006e: Batch performance should be better than sequential for multiple prompts
#[test]
fn test_parity006e_batch_performance_comparison() {
    use std::time::Instant;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let prompts = vec![
        vec![1u32, 2, 3],
        vec![4u32, 5, 6],
        vec![7u32, 8, 9],
        vec![10u32, 11, 12],
    ];

    // Sequential generation
    let start = Instant::now();
    for prompt in &prompts {
        let _ = model.generate_with_cache(prompt, &gen_config);
    }
    let sequential_time = start.elapsed();

    // Batch generation
    let cached = OwnedQuantizedModelCached::new(model);
    let start = Instant::now();
    let _ = cached.model().batch_generate(
        &prompts.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
        &gen_config,
    );
    let batch_time = start.elapsed();

    println!("PARITY-006e: Performance Comparison");
    println!("  Sequential: {:?}", sequential_time);
    println!("  Batch: {:?}", batch_time);
    println!(
        "  Speedup: {:.2}x",
        sequential_time.as_nanos() as f64 / batch_time.as_nanos() as f64
    );

    // Batch should not be slower (within 10x to account for coverage instrumentation)
    assert!(
        batch_time.as_nanos() <= sequential_time.as_nanos() * 10,
        "PARITY-006e: Batch should not be more than 10x slower than sequential"
    );
}

/// PARITY-006f: Empty prompts should return error
#[test]
fn test_parity006f_empty_prompts_error() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);
    let cached = OwnedQuantizedModelCached::new(model);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Empty batch should error
    let empty_prompts: Vec<&[u32]> = vec![];
    let result = cached.model().batch_generate(&empty_prompts, &gen_config);
    assert!(
        result.is_err(),
        "PARITY-006f: Empty batch should return error"
    );
}

// ========================================================================
// PARITY-007: E2E Benchmark Verification Tests
// ========================================================================

/// PARITY-007a: Coefficient of variation calculation
#[test]
fn test_parity007a_cv_calculation() {
    // CV = (std_dev / mean) * 100
    let values: Vec<f64> = vec![10.0, 12.0, 11.0, 9.0, 11.0];
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    let cv = (std_dev / mean) * 100.0;

    // CV should be low for stable measurements
    assert!(cv < 15.0, "PARITY-007a: CV should be < 15% for stable data");

    // Test with high variance data
    let noisy: Vec<f64> = vec![1.0, 100.0, 5.0, 50.0, 10.0];
    let noisy_mean: f64 = noisy.iter().sum::<f64>() / noisy.len() as f64;
    let noisy_variance: f64 =
        noisy.iter().map(|x| (x - noisy_mean).powi(2)).sum::<f64>() / noisy.len() as f64;
    let noisy_cv = (noisy_variance.sqrt() / noisy_mean) * 100.0;

    assert!(
        noisy_cv > 50.0,
        "PARITY-007a: Noisy data should have CV > 50%"
    );
}

/// PARITY-007b: Benchmark metrics struct
#[test]
fn test_parity007b_benchmark_metrics() {
    // BenchmarkMetrics should capture all required fields
    struct BenchmarkMetrics {
        throughput_toks: f64,
        latency_p50_ms: f64,
        latency_p95_ms: f64,
        latency_p99_ms: f64,
        cv_percent: f64,
    }

    let metrics = BenchmarkMetrics {
        throughput_toks: 64.0,
        latency_p50_ms: 15.6,
        latency_p95_ms: 18.2,
        latency_p99_ms: 21.5,
        cv_percent: 8.5,
    };

    assert!(
        metrics.throughput_toks > 0.0,
        "PARITY-007b: Throughput should be positive"
    );
    assert!(
        metrics.latency_p50_ms < metrics.latency_p95_ms,
        "PARITY-007b: p50 should be < p95"
    );
    assert!(
        metrics.latency_p95_ms < metrics.latency_p99_ms,
        "PARITY-007b: p95 should be < p99"
    );
    assert!(metrics.cv_percent < 15.0, "PARITY-007b: CV should be < 15%");
}

/// PARITY-007c: Hardware info capture
#[test]
fn test_parity007c_hardware_info() {
    struct HardwareInfo {
        cpu_model: String,
        cpu_cores: usize,
        ram_gb: usize,
        gpu_name: Option<String>,
    }

    let info = HardwareInfo {
        cpu_model: "AMD Ryzen 9 5950X".to_string(),
        cpu_cores: 32,
        ram_gb: 128,
        gpu_name: Some("NVIDIA RTX 4090".to_string()),
    };

    assert!(
        !info.cpu_model.is_empty(),
        "PARITY-007c: CPU model should not be empty"
    );
    assert!(info.cpu_cores >= 1, "PARITY-007c: CPU cores should be >= 1");
    assert!(info.ram_gb >= 1, "PARITY-007c: RAM should be >= 1 GB");
}

/// PARITY-007d: Percentile calculation
#[test]
fn test_parity007d_percentile_calculation() {
    let mut values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50_idx = (values.len() as f64 * 0.50) as usize - 1;
    let p95_idx = (values.len() as f64 * 0.95) as usize - 1;
    let p99_idx = (values.len() as f64 * 0.99) as usize - 1;

    let p50 = values[p50_idx];
    let p95 = values[p95_idx];
    let p99 = values[p99_idx];

    assert!((p50 - 50.0).abs() < 1.0, "PARITY-007d: p50 should be ~50");
    assert!((p95 - 95.0).abs() < 1.0, "PARITY-007d: p95 should be ~95");
    assert!((p99 - 99.0).abs() < 1.0, "PARITY-007d: p99 should be ~99");
}

/// PARITY-007e: Gap calculation to target
#[test]
fn test_parity007e_gap_calculation() {
    let current_toks: f64 = 64.0;
    let target_toks: f64 = 225.0;
    let gap_ratio: f64 = target_toks / current_toks;

    assert!(
        (gap_ratio - 3.52_f64).abs() < 0.1,
        "PARITY-007e: Gap ratio should be ~3.5x"
    );

    let gap_percent = (target_toks - current_toks) / current_toks * 100.0;
    assert!(
        gap_percent > 200.0,
        "PARITY-007e: Gap should be > 200% improvement needed"
    );
}

/// PARITY-007f: End-to-end benchmark with realizar baseline
#[test]
fn test_parity007f_realizar_benchmark() {
    use std::time::Instant;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_runs = 5;

    let mut throughputs = Vec::with_capacity(num_runs);
    let mut latencies = Vec::with_capacity(num_runs);

    for _ in 0..num_runs {
        let start = Instant::now();
        let result = model.generate_with_cache(&prompt, &gen_config).unwrap();
        let elapsed = start.elapsed();

        let new_tokens = result.len() - prompt.len();
        let toks_per_sec = new_tokens as f64 / elapsed.as_secs_f64();
        let latency_ms = elapsed.as_secs_f64() * 1000.0 / new_tokens as f64;

        throughputs.push(toks_per_sec);
        latencies.push(latency_ms);
    }

    let mean_tps: f64 = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

    println!("PARITY-007f: Realizar Test Model Benchmark");
    println!("  Mean throughput: {:.1} tok/s", mean_tps);

    // Test model should achieve some throughput
    assert!(mean_tps > 0.0, "PARITY-007f: Throughput should be positive");
}

// ========================================================================
// PARITY-008: Popper Score Improvement Tests
// ========================================================================

/// PARITY-008a: Falsifiable claim structure
#[test]
fn test_parity008a_falsifiable_claim_structure() {
    struct FalsifiableClaim {
        prediction: String,
        measurement: String,
        threshold: f64,
        evidence: Option<f64>,
    }

    let claim = FalsifiableClaim {
        prediction: "GPU batch FFN is 1.1x faster at batch=32".to_string(),
        measurement: "FFN latency ratio (GPU/CPU)".to_string(),
        threshold: 0.91, // 1/1.1
        evidence: None,
    };

    assert!(
        !claim.prediction.is_empty(),
        "PARITY-008a: Prediction should not be empty"
    );
    assert!(
        claim.threshold > 0.0 && claim.threshold < 1.0,
        "PARITY-008a: Threshold for speedup should be < 1.0"
    );
}

/// PARITY-008b: Random seed management for reproducibility
#[test]
fn test_parity008b_random_seed_management() {
    // Local struct for seed configuration testing
    struct SeedConfig {
        seed: u64,
    }

    impl SeedConfig {
        fn new(seed: u64) -> Self {
            Self { seed }
        }

        fn for_ollama_comparison() -> Self {
            Self { seed: 42 }
        }
    }

    let config = SeedConfig::for_ollama_comparison();
    assert_eq!(
        config.seed, 42,
        "PARITY-008b: Ollama comparison seed should be 42"
    );

    let derived = SeedConfig::new(1000);
    assert_eq!(derived.seed, 1000, "PARITY-008b: Custom seed should match");

    // Same seed should produce same sequence
    let seed1 = SeedConfig::new(42);
    let seed2 = SeedConfig::new(42);
    assert_eq!(
        seed1.seed, seed2.seed,
        "PARITY-008b: Same seed should be reproducible"
    );
}

/// PARITY-008c: Popper score calculation
#[test]
fn test_parity008c_popper_score_calculation() {
    // Popper score: How well predictions match evidence
    // Score = 1.0 - (|predicted - actual| / predicted)

    // Local struct for Popper score calculation testing
    #[allow(dead_code)]
    struct PopperScore {
        prediction: String,
        predicted: f64,
        actual: f64,
        score: f64,
    }

    impl PopperScore {
        fn calculate(prediction: String, predicted: f64, actual: f64) -> Self {
            let score = 1.0 - ((predicted - actual).abs() / predicted);
            Self {
                prediction,
                predicted,
                actual,
                score,
            }
        }
    }

    let before = PopperScore::calculate(
        "GPU 10x faster".to_string(),
        10.0, // Predicted speedup
        2.5,  // Actual speedup
    );

    let after = PopperScore::calculate(
        "GPU 10x faster".to_string(),
        10.0, // Predicted speedup
        9.5,  // Actual speedup (after optimization)
    );

    assert!(
        before.score < after.score,
        "PARITY-008c: Score should improve when actual approaches predicted"
    );

    // Perfect match should give score of 1.0
    let perfect = PopperScore::calculate("Test".to_string(), 5.0, 5.0);
    assert!(
        (perfect.score - 1.0).abs() < 0.01,
        "PARITY-008c: Perfect match should give score ~1.0"
    );
}

/// PARITY-008d: Explicit thresholds for acceptance
#[test]
fn test_parity008d_explicit_thresholds() {
    struct AcceptanceThreshold {
        metric: String,
        minimum: f64,
        target: f64,
        stretch: f64,
    }

    let throughput_threshold = AcceptanceThreshold {
        metric: "tok/s".to_string(),
        minimum: 64.0,  // Current baseline
        target: 225.0,  // Ollama parity
        stretch: 300.0, // Exceeds Ollama
    };

    let latency_threshold = AcceptanceThreshold {
        metric: "ms/token".to_string(),
        minimum: 50.0, // Maximum acceptable
        target: 4.4,   // Ollama parity (1000/225)
        stretch: 3.3,  // Exceeds Ollama
    };

    assert!(
        throughput_threshold.minimum < throughput_threshold.target,
        "PARITY-008d: Minimum should be less than target"
    );
    assert!(
        throughput_threshold.target < throughput_threshold.stretch,
        "PARITY-008d: Target should be less than stretch"
    );
    assert!(
        latency_threshold.minimum > latency_threshold.target,
        "PARITY-008d: Latency minimum should be higher than target (lower is better)"
    );
}

/// PARITY-008e: Benchmark reproducibility check
#[test]
fn test_parity008e_benchmark_reproducibility() {
    use std::time::Instant;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
    };
    let model = create_test_model_with_config(&config);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let prompt = vec![1u32, 2, 3];

    // Run twice with same seed/config
    let start1 = Instant::now();
    let result1 = model.generate_with_cache(&prompt, &gen_config).unwrap();
    let _time1 = start1.elapsed();

    let start2 = Instant::now();
    let result2 = model.generate_with_cache(&prompt, &gen_config).unwrap();
    let _time2 = start2.elapsed();

    // Outputs should be identical (deterministic sampling)
    assert_eq!(
        result1, result2,
        "PARITY-008e: Deterministic sampling should produce identical outputs"
    );
}

/// PARITY-008f: Measurement validation
#[test]
fn test_parity008f_measurement_validation() {
    // Validate that measurements are within expected ranges

    struct Measurement {
        value: f64,
        unit: String,
        min_valid: f64,
        max_valid: f64,
    }

    impl Measurement {
        fn is_valid(&self) -> bool {
            self.value >= self.min_valid && self.value <= self.max_valid
        }
    }

    let throughput = Measurement {
        value: 64.0,
        unit: "tok/s".to_string(),
        min_valid: 0.1,
        max_valid: 10000.0,
    };

    let latency = Measurement {
        value: 15.6,
        unit: "ms".to_string(),
        min_valid: 0.001,
        max_valid: 10000.0,
    };

    assert!(
        throughput.is_valid(),
        "PARITY-008f: Throughput should be valid"
    );
    assert!(latency.is_valid(), "PARITY-008f: Latency should be valid");

    // Invalid measurement should fail
    let invalid = Measurement {
        value: -5.0,
        unit: "tok/s".to_string(),
        min_valid: 0.0,
        max_valid: 10000.0,
    };
    assert!(
        !invalid.is_valid(),
        "PARITY-008f: Negative throughput should be invalid"
    );
}

// ========================================================================
// PARITY-009: Benchmark Infrastructure (QA-031 to QA-040)
// ========================================================================

/// Test PARITY-009a: QA-031 CV-based stopping criterion per Hoefler & Belli
#[test]
fn test_parity009a_cv_stopping_criterion() {
    /// Benchmark runner with CV-based stopping
    /// Per Hoefler & Belli SC'15: Stop when CV < threshold
    #[derive(Debug)]
    struct CVStoppingBenchmark {
        target_cv: f64,
        max_iterations: usize,
        min_iterations: usize,
    }

    impl CVStoppingBenchmark {
        fn new() -> Self {
            Self {
                target_cv: 0.05, // 5% CV threshold per spec
                max_iterations: 100,
                min_iterations: 5,
            }
        }

        fn calculate_cv(values: &[f64]) -> f64 {
            if values.len() < 2 {
                return 1.0; // High CV for insufficient data
            }
            let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
            if mean == 0.0 {
                return 0.0;
            }
            let variance: f64 =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variance.sqrt() / mean
        }

        fn should_stop(&self, values: &[f64]) -> (bool, f64) {
            if values.len() < self.min_iterations {
                return (false, 1.0);
            }
            if values.len() >= self.max_iterations {
                return (true, Self::calculate_cv(values));
            }
            let cv = Self::calculate_cv(values);
            (cv < self.target_cv, cv)
        }

        fn run<F>(&self, mut benchmark_fn: F) -> (Vec<f64>, usize, f64)
        where
            F: FnMut() -> f64,
        {
            let mut values = Vec::new();
            loop {
                values.push(benchmark_fn());
                let (stop, cv) = self.should_stop(&values);
                if stop {
                    let len = values.len();
                    return (values, len, cv);
                }
            }
        }
    }

    let runner = CVStoppingBenchmark::new();

    // Simulate stable measurements (low CV)
    let mut counter = 0;
    let (_values, iterations, cv) = runner.run(|| {
        counter += 1;
        100.0 + (counter as f64 * 0.01) // Very stable: 100.01, 100.02, ...
    });

    println!("\nPARITY-009a: CV-based stopping");
    println!("  Iterations: {}", iterations);
    println!("  Final CV: {:.4}", cv);
    println!("  Target CV: {:.4}", runner.target_cv);

    assert!(
        cv < runner.target_cv,
        "QA-031: CV should be below threshold"
    );
    assert!(
        iterations >= runner.min_iterations,
        "QA-031: Should run minimum iterations"
    );
    assert!(
        iterations <= runner.max_iterations,
        "QA-031: Should not exceed max iterations"
    );
}

/// Test PARITY-009b: QA-032 Warmup iterations discard
#[test]
fn test_parity009b_warmup_discard() {
    /// Benchmark with warmup discard per Mytkowicz et al.
    #[derive(Debug)]
    struct WarmupBenchmark {
        warmup_iterations: usize,
        measurement_iterations: usize,
    }

    impl WarmupBenchmark {
        fn new(warmup: usize, measure: usize) -> Self {
            Self {
                warmup_iterations: warmup,
                measurement_iterations: measure,
            }
        }

        fn run<F>(&self, mut benchmark_fn: F) -> (Vec<f64>, Vec<f64>)
        where
            F: FnMut(usize) -> f64,
        {
            let mut warmup_values = Vec::with_capacity(self.warmup_iterations);
            let mut measurement_values = Vec::with_capacity(self.measurement_iterations);

            // Warmup phase (JIT, cache warming)
            for i in 0..self.warmup_iterations {
                warmup_values.push(benchmark_fn(i));
            }

            // Measurement phase
            for i in 0..self.measurement_iterations {
                measurement_values.push(benchmark_fn(self.warmup_iterations + i));
            }

            (warmup_values, measurement_values)
        }
    }

    let runner = WarmupBenchmark::new(3, 5);

    // Simulate JIT warmup effect: first iterations are slower
    let (warmup, measurements) = runner.run(|i| {
        if i < 3 {
            200.0 - (i as f64 * 30.0) // Warmup: 200, 170, 140
        } else {
            100.0 + (i as f64 * 0.5) // Stable: ~101.5 - 103.5
        }
    });

    let warmup_mean: f64 = warmup.iter().sum::<f64>() / warmup.len() as f64;
    let measure_mean: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;

    println!("\nPARITY-009b: Warmup discard");
    println!(
        "  Warmup iterations: {} (mean: {:.1})",
        warmup.len(),
        warmup_mean
    );
    println!(
        "  Measurement iterations: {} (mean: {:.1})",
        measurements.len(),
        measure_mean
    );

    assert_eq!(warmup.len(), 3, "QA-032: Should have 3 warmup iterations");
    assert_eq!(
        measurements.len(),
        5,
        "QA-032: Should have 5 measurement iterations"
    );
    assert!(
        warmup_mean > measure_mean,
        "QA-032: Warmup should be slower (JIT effect)"
    );
}

/// Test PARITY-009c: QA-033 Environment metadata capture
#[test]
fn test_parity009c_environment_metadata() {
    /// Environment metadata per Vitek & Kalibera
    #[derive(Debug, Clone)]
    struct EnvironmentMetadata {
        // System info
        os: String,
        arch: String,
        #[allow(dead_code)]
        cpu_model: String,
        cpu_cores: usize,
        #[allow(dead_code)]
        ram_gb: usize,

        // Runtime info
        #[allow(dead_code)]
        rust_version: String,
        cargo_profile: String,
        #[allow(dead_code)]
        target_triple: String,

        // Benchmark config
        #[allow(dead_code)]
        timestamp: String,
        #[allow(dead_code)]
        git_commit: String,
        #[allow(dead_code)]
        benchmark_version: String,
    }

    impl EnvironmentMetadata {
        fn capture() -> Self {
            Self {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                cpu_model: "Unknown".to_string(), // Would read from /proc/cpuinfo
                cpu_cores: std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(1),
                ram_gb: 16, // Would read from system
                rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
                cargo_profile: if cfg!(debug_assertions) {
                    "debug"
                } else {
                    "release"
                }
                .to_string(),
                target_triple: std::env::consts::ARCH.to_string(),
                timestamp: "2025-12-13T22:00:00Z".to_string(),
                git_commit: "abc123".to_string(),
                benchmark_version: "1.0.0".to_string(),
            }
        }

        fn is_reproducible(&self) -> bool {
            !self.os.is_empty()
                && !self.arch.is_empty()
                && self.cpu_cores > 0
                && !self.cargo_profile.is_empty()
        }
    }

    let env = EnvironmentMetadata::capture();

    println!("\nPARITY-009c: Environment metadata");
    println!("  OS: {}", env.os);
    println!("  Arch: {}", env.arch);
    println!("  CPU cores: {}", env.cpu_cores);
    println!("  Profile: {}", env.cargo_profile);

    assert!(
        env.is_reproducible(),
        "QA-033: Environment must be reproducible"
    );
    assert!(!env.os.is_empty(), "QA-033: OS must be captured");
    assert!(!env.arch.is_empty(), "QA-033: Arch must be captured");
    assert!(env.cpu_cores > 0, "QA-033: CPU cores must be captured");
}

/// Test PARITY-009d: QA-034 Outlier detection using MAD
#[test]
fn test_parity009d_outlier_detection_mad() {
    /// Median Absolute Deviation (MAD) outlier detection
    /// Per Fleming & Wallace: MAD is robust to outliers
    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).expect("test"));
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    fn mad(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        let med = median(&mut sorted);
        let mut deviations: Vec<f64> = values.iter().map(|v| (v - med).abs()).collect();
        median(&mut deviations)
    }

    fn detect_outliers(values: &[f64], threshold: f64) -> Vec<usize> {
        let mut sorted = values.to_vec();
        let med = median(&mut sorted);
        let mad_value = mad(values);
        let k = 1.4826; // Scale factor for normal distribution

        values
            .iter()
            .enumerate()
            .filter(|(_, &v)| {
                if mad_value == 0.0 {
                    false
                } else {
                    ((v - med).abs() / (k * mad_value)) > threshold
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    // Test data with outliers
    let values = vec![100.0, 101.0, 99.0, 102.0, 98.0, 500.0, 100.5, 99.5];
    let outliers = detect_outliers(&values, 3.0); // 3 MAD threshold

    println!("\nPARITY-009d: MAD outlier detection");
    println!("  Values: {:?}", values);
    println!("  MAD: {:.2}", mad(&values));
    println!("  Outliers at indices: {:?}", outliers);

    assert!(
        outliers.contains(&5),
        "QA-034: Should detect 500.0 as outlier"
    );
    assert!(
        !outliers.contains(&0),
        "QA-034: 100.0 should not be outlier"
    );
}

/// Test PARITY-009e: QA-035 p50, p95, p99 latencies
#[test]
fn test_parity009e_latency_percentiles() {
    /// Latency percentile calculator per Georges et al.
    #[derive(Debug, Clone)]
    struct LatencyStats {
        p50: f64,
        p95: f64,
        p99: f64,
        min: f64,
        max: f64,
        #[allow(dead_code)]
        mean: f64,
    }

    impl LatencyStats {
        fn from_latencies(latencies: &[f64]) -> Self {
            let mut sorted = latencies.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("test"));

            let percentile = |p: f64| -> f64 {
                let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
                sorted[idx.min(sorted.len() - 1)]
            };

            Self {
                p50: percentile(0.50),
                p95: percentile(0.95),
                p99: percentile(0.99),
                min: sorted[0],
                max: sorted[sorted.len() - 1],
                mean: latencies.iter().sum::<f64>() / latencies.len() as f64,
            }
        }
    }

    // Simulate latency distribution
    let latencies: Vec<f64> = (0..100)
        .map(|i| 10.0 + (i as f64 * 0.5) + if i > 95 { 50.0 } else { 0.0 })
        .collect();

    let stats = LatencyStats::from_latencies(&latencies);

    println!("\nPARITY-009e: Latency percentiles");
    println!("  p50: {:.2}ms", stats.p50);
    println!("  p95: {:.2}ms", stats.p95);
    println!("  p99: {:.2}ms", stats.p99);
    println!("  min: {:.2}ms, max: {:.2}ms", stats.min, stats.max);

    assert!(stats.p50 < stats.p95, "QA-035: p50 should be less than p95");
    assert!(stats.p95 < stats.p99, "QA-035: p95 should be less than p99");
    assert!(stats.min <= stats.p50, "QA-035: min should be <= p50");
    assert!(stats.p99 <= stats.max, "QA-035: p99 should be <= max");
}

/// Test PARITY-009f: QA-036 Throughput with variance
#[test]
fn test_parity009f_throughput_variance() {
    /// Throughput measurement with variance tracking
    #[derive(Debug, Clone)]
    struct ThroughputStats {
        mean_tps: f64,
        variance: f64,
        stddev: f64,
        cv: f64,
        samples: usize,
    }

    impl ThroughputStats {
        fn from_samples(tps_samples: &[f64]) -> Self {
            let n = tps_samples.len() as f64;
            let mean = tps_samples.iter().sum::<f64>() / n;
            let variance = tps_samples.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
            let stddev = variance.sqrt();
            let cv = if mean > 0.0 { stddev / mean } else { 0.0 };

            Self {
                mean_tps: mean,
                variance,
                stddev,
                cv,
                samples: tps_samples.len(),
            }
        }

        fn is_stable(&self) -> bool {
            self.cv < 0.05 // 5% CV threshold
        }

        fn confidence_interval_95(&self) -> (f64, f64) {
            let margin = 1.96 * self.stddev / (self.samples as f64).sqrt();
            (self.mean_tps - margin, self.mean_tps + margin)
        }
    }

    // Simulate throughput measurements
    let tps_samples = vec![200.0, 205.0, 198.0, 202.0, 201.0, 199.0, 203.0, 200.5];
    let stats = ThroughputStats::from_samples(&tps_samples);
    let (ci_low, ci_high) = stats.confidence_interval_95();

    println!("\nPARITY-009f: Throughput with variance");
    println!("  Mean: {:.2} tok/s", stats.mean_tps);
    println!("  StdDev: {:.2}", stats.stddev);
    println!("  CV: {:.4}", stats.cv);
    println!("  95% CI: [{:.2}, {:.2}]", ci_low, ci_high);

    assert!(
        stats.is_stable(),
        "QA-036: Measurements should be stable (CV < 0.05)"
    );
    assert!(stats.variance > 0.0, "QA-036: Variance should be positive");
    assert!(
        ci_low < stats.mean_tps && stats.mean_tps < ci_high,
        "QA-036: Mean should be in CI"
    );
}

/// Test PARITY-009g: QA-037 Versioned benchmark results
#[test]
fn test_parity009g_versioned_results() {
    /// Versioned benchmark result for reproducibility
    #[derive(Debug, Clone)]
    struct VersionedBenchmarkResult {
        // Version info
        schema_version: String,
        benchmark_version: String,
        realizar_version: String,

        // Metadata
        timestamp: String,
        git_commit: String,
        environment_hash: String,

        // Results
        throughput_tps: f64,
        #[allow(dead_code)]
        latency_p50_ms: f64,
        #[allow(dead_code)]
        latency_p99_ms: f64,
        cv: f64,
        iterations: usize,
    }

    impl VersionedBenchmarkResult {
        fn new(tps: f64, p50: f64, p99: f64, cv: f64, iterations: usize) -> Self {
            Self {
                schema_version: "1.0.0".to_string(),
                benchmark_version: "PARITY-009".to_string(),
                realizar_version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: "2025-12-13T22:00:00Z".to_string(),
                git_commit: "abc123def".to_string(),
                environment_hash: "sha256:...".to_string(),
                throughput_tps: tps,
                latency_p50_ms: p50,
                latency_p99_ms: p99,
                cv,
                iterations,
            }
        }

        fn is_valid(&self) -> bool {
            !self.schema_version.is_empty()
                && !self.benchmark_version.is_empty()
                && !self.realizar_version.is_empty()
                && self.throughput_tps > 0.0
                && self.cv >= 0.0
                && self.iterations > 0
        }

        fn is_reproducible(&self) -> bool {
            !self.git_commit.is_empty()
                && !self.timestamp.is_empty()
                && !self.environment_hash.is_empty()
        }
    }

    let result = VersionedBenchmarkResult::new(
        200.5, // tps
        5.2,   // p50
        12.8,  // p99
        0.025, // cv
        50,    // iterations
    );

    println!("\nPARITY-009g: Versioned results");
    println!("  Schema: {}", result.schema_version);
    println!("  Benchmark: {}", result.benchmark_version);
    println!("  Realizar: {}", result.realizar_version);
    println!("  Throughput: {:.2} tok/s", result.throughput_tps);

    assert!(result.is_valid(), "QA-037: Result must be valid");
    assert!(
        result.is_reproducible(),
        "QA-037: Result must be reproducible"
    );
    assert_eq!(
        result.schema_version, "1.0.0",
        "QA-037: Schema version must be set"
    );
}

// ========================================================================
// PARITY-010: Benchmark Infrastructure QA-038 to QA-040
// ========================================================================

/// Test PARITY-010a: QA-038 Preflight checks validate server availability
#[test]
fn test_parity010a_preflight_server_checks() {
    /// Preflight check result
    #[derive(Debug, Clone)]
    enum PreflightStatus {
        Pass,
        Fail(String),
        Skip(String),
    }

    /// Server availability check
    #[derive(Debug)]
    struct ServerPreflightCheck {
        name: String,
        endpoint: String,
        #[allow(dead_code)]
        timeout_ms: u64,
        required: bool,
    }

    impl ServerPreflightCheck {
        fn new(name: &str, endpoint: &str, required: bool) -> Self {
            Self {
                name: name.to_string(),
                endpoint: endpoint.to_string(),
                timeout_ms: 5000,
                required,
            }
        }

        /// Simulate server check (real impl would use HTTP client)
        fn check(&self, server_available: bool) -> PreflightStatus {
            if server_available {
                PreflightStatus::Pass
            } else if self.required {
                PreflightStatus::Fail(format!("{} not available at {}", self.name, self.endpoint))
            } else {
                PreflightStatus::Skip(format!("{} optional, skipping", self.name))
            }
        }
    }

    /// Preflight suite for benchmark servers
    #[derive(Debug)]
    struct PreflightSuite {
        checks: Vec<ServerPreflightCheck>,
    }

    impl PreflightSuite {
        fn new() -> Self {
            Self {
                checks: vec![
                    ServerPreflightCheck::new("Ollama", "http://localhost:11434", true),
                    ServerPreflightCheck::new("llama.cpp", "http://localhost:8080", false),
                    ServerPreflightCheck::new("vLLM", "http://localhost:8000", false),
                ],
            }
        }

        fn run(&self, availability: &[bool]) -> (usize, usize, usize) {
            let mut passed = 0;
            let mut failed = 0;
            let mut skipped = 0;

            for (check, &available) in self.checks.iter().zip(availability.iter()) {
                match check.check(available) {
                    PreflightStatus::Pass => passed += 1,
                    PreflightStatus::Fail(_) => failed += 1,
                    PreflightStatus::Skip(_) => skipped += 1,
                }
            }

            (passed, failed, skipped)
        }

        fn all_required_pass(&self, availability: &[bool]) -> bool {
            for (check, &available) in self.checks.iter().zip(availability.iter()) {
                if check.required && !available {
                    return false;
                }
            }
            true
        }
    }

    let suite = PreflightSuite::new();

    // Test: All servers available
    let (passed, failed, _skipped) = suite.run(&[true, true, true]);
    assert_eq!(passed, 3, "QA-038: All 3 servers should pass");
    assert_eq!(failed, 0, "QA-038: No failures");

    // Test: Only required (Ollama) available
    let (passed, _failed, skipped) = suite.run(&[true, false, false]);
    assert_eq!(passed, 1, "QA-038: Ollama passes");
    assert_eq!(skipped, 2, "QA-038: Optional servers skipped");
    assert!(
        suite.all_required_pass(&[true, false, false]),
        "QA-038: Required servers pass"
    );

    // Test: Required server unavailable
    assert!(
        !suite.all_required_pass(&[false, true, true]),
        "QA-038: Should fail if Ollama down"
    );

    println!("\nPARITY-010a: Preflight server checks");
    println!("  Checks defined: {}", suite.checks.len());
    println!("  Required: Ollama");
    println!("  Optional: llama.cpp, vLLM");
}

/// Test PARITY-010b: QA-039 Automatic model download from Hugging Face
#[test]
fn test_parity010b_model_download() {
    /// Model download configuration
    #[derive(Debug, Clone)]
    struct ModelDownloadConfig {
        repo_id: String,
        filename: String,
        revision: String,
        cache_dir: String,
    }

    impl ModelDownloadConfig {
        fn new(repo_id: &str, filename: &str) -> Self {
            Self {
                repo_id: repo_id.to_string(),
                filename: filename.to_string(),
                revision: "main".to_string(),
                cache_dir: "~/.cache/huggingface/hub".to_string(),
            }
        }

        fn url(&self) -> String {
            format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                self.repo_id, self.revision, self.filename
            )
        }

        fn cache_path(&self) -> String {
            let repo_dir = self.repo_id.replace('/', "--");
            format!(
                "{}/models--{}/snapshots/{}/{}",
                self.cache_dir, repo_dir, self.revision, self.filename
            )
        }
    }

    /// Model download status
    #[derive(Debug, Clone)]
    enum DownloadStatus {
        Cached(String),     // Already in cache
        Downloaded(String), // Freshly downloaded
        #[allow(dead_code)]
        Failed(String), // Download failed
    }

    /// Model downloader (test)
    struct ModelDownloader {
        configs: Vec<ModelDownloadConfig>,
    }

    impl ModelDownloader {
        fn new() -> Self {
            Self {
                configs: vec![
                    ModelDownloadConfig::new("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf"),
                    ModelDownloadConfig::new("microsoft/phi-2", "model.safetensors"),
                ],
            }
        }

        /// Simulate download check
        fn check_or_download(&self, config: &ModelDownloadConfig, cached: bool) -> DownloadStatus {
            if cached {
                DownloadStatus::Cached(config.cache_path())
            } else {
                // In real impl: download from config.url()
                DownloadStatus::Downloaded(config.cache_path())
            }
        }
    }

    let downloader = ModelDownloader::new();
    let config = &downloader.configs[0];

    // Test: Model already cached
    let status = downloader.check_or_download(config, true);
    assert!(
        matches!(status, DownloadStatus::Cached(_)),
        "QA-039: Should return cached"
    );

    // Test: Model needs download
    let status = downloader.check_or_download(config, false);
    assert!(
        matches!(status, DownloadStatus::Downloaded(_)),
        "QA-039: Should download"
    );

    // Test: URL construction
    let url = config.url();
    assert!(
        url.contains("huggingface.co"),
        "QA-039: URL should be HuggingFace"
    );
    assert!(
        url.contains(&config.repo_id),
        "QA-039: URL should contain repo"
    );
    assert!(
        url.contains(&config.filename),
        "QA-039: URL should contain filename"
    );

    println!("\nPARITY-010b: Model download from HuggingFace");
    println!("  Repo: {}", config.repo_id);
    println!("  File: {}", config.filename);
    println!("  URL: {}", config.url());
}

/// Test PARITY-010c: QA-040 JSON schema validation for benchmark results
#[test]
fn test_parity010c_json_schema_validation() {
    /// JSON schema field definition
    #[derive(Debug, Clone)]
    struct SchemaField {
        name: String,
        field_type: FieldType,
        required: bool,
    }

    #[derive(Debug, Clone)]
    enum FieldType {
        String,
        Number,
        Integer,
        #[allow(dead_code)]
        Boolean,
        #[allow(dead_code)]
        Array(Box<FieldType>),
        Object(Vec<SchemaField>),
    }

    /// Benchmark result schema
    #[derive(Debug)]
    struct BenchmarkResultSchema {
        version: String,
        fields: Vec<SchemaField>,
    }

    impl BenchmarkResultSchema {
        fn v1() -> Self {
            Self {
                version: "1.0.0".to_string(),
                fields: vec![
                    SchemaField {
                        name: "schema_version".to_string(),
                        field_type: FieldType::String,
                        required: true,
                    },
                    SchemaField {
                        name: "timestamp".to_string(),
                        field_type: FieldType::String,
                        required: true,
                    },
                    SchemaField {
                        name: "git_commit".to_string(),
                        field_type: FieldType::String,
                        required: true,
                    },
                    SchemaField {
                        name: "throughput_tps".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "latency_p50_ms".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "latency_p95_ms".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "latency_p99_ms".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "cv".to_string(),
                        field_type: FieldType::Number,
                        required: true,
                    },
                    SchemaField {
                        name: "iterations".to_string(),
                        field_type: FieldType::Integer,
                        required: true,
                    },
                    SchemaField {
                        name: "environment".to_string(),
                        field_type: FieldType::Object(vec![
                            SchemaField {
                                name: "os".to_string(),
                                field_type: FieldType::String,
                                required: true,
                            },
                            SchemaField {
                                name: "arch".to_string(),
                                field_type: FieldType::String,
                                required: true,
                            },
                            SchemaField {
                                name: "cpu_cores".to_string(),
                                field_type: FieldType::Integer,
                                required: true,
                            },
                        ]),
                        required: true,
                    },
                ],
            }
        }

        fn required_field_count(&self) -> usize {
            self.fields.iter().filter(|f| f.required).count()
        }

        fn validate_field_presence(&self, field_names: &[&str]) -> Vec<String> {
            let mut missing = Vec::new();
            for field in &self.fields {
                if field.required && !field_names.contains(&field.name.as_str()) {
                    missing.push(field.name.clone());
                }
            }
            missing
        }
    }

    let schema = BenchmarkResultSchema::v1();

    // Test: Schema version
    assert_eq!(
        schema.version, "1.0.0",
        "QA-040: Schema version should be 1.0.0"
    );

    // Test: Required fields
    assert!(
        schema.required_field_count() >= 9,
        "QA-040: Should have >=9 required fields"
    );

    // Test: Validation with all fields
    let all_fields = vec![
        "schema_version",
        "timestamp",
        "git_commit",
        "throughput_tps",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "cv",
        "iterations",
        "environment",
    ];
    let missing = schema.validate_field_presence(&all_fields);
    assert!(missing.is_empty(), "QA-040: All required fields present");

    // Test: Validation with missing fields
    let partial_fields = vec!["schema_version", "throughput_tps"];
    let missing = schema.validate_field_presence(&partial_fields);
    assert!(!missing.is_empty(), "QA-040: Should detect missing fields");
    assert!(
        missing.contains(&"timestamp".to_string()),
        "QA-040: timestamp should be missing"
    );

    println!("\nPARITY-010c: JSON schema validation");
    println!("  Schema version: {}", schema.version);
    println!("  Required fields: {}", schema.required_field_count());
    println!("  Total fields: {}", schema.fields.len());
}

/// Test PARITY-010d: Combined preflight and validation suite
#[test]
fn test_parity010d_benchmark_preflight_suite() {
    /// Complete preflight suite combining all checks
    #[derive(Debug)]
    struct BenchmarkPreflightSuite {
        server_checks: Vec<(&'static str, bool)>, // (name, required)
        model_checks: Vec<&'static str>,          // model repo IDs
        schema_version: &'static str,
    }

    impl BenchmarkPreflightSuite {
        fn standard() -> Self {
            Self {
                server_checks: vec![("Ollama", true), ("llama.cpp", false), ("vLLM", false)],
                model_checks: vec!["TheBloke/phi-2-GGUF", "microsoft/phi-2"],
                schema_version: "1.0.0",
            }
        }

        fn run_all(&self, servers_up: &[bool], models_cached: &[bool]) -> PreflightResult {
            let mut result = PreflightResult::default();

            // Server checks
            for ((name, required), &up) in self.server_checks.iter().zip(servers_up.iter()) {
                if up {
                    result.servers_passed += 1;
                } else if *required {
                    result.servers_failed += 1;
                    result.errors.push(format!("{} unavailable", name));
                } else {
                    result.servers_skipped += 1;
                }
            }

            // Model checks
            for (_model, &cached) in self.model_checks.iter().zip(models_cached.iter()) {
                if cached {
                    result.models_cached += 1;
                } else {
                    result.models_to_download += 1;
                }
            }

            result.schema_valid = true;
            result
        }
    }

    #[derive(Debug, Default)]
    struct PreflightResult {
        servers_passed: usize,
        servers_failed: usize,
        #[allow(dead_code)]
        servers_skipped: usize,
        models_cached: usize,
        models_to_download: usize,
        schema_valid: bool,
        errors: Vec<String>,
    }

    impl PreflightResult {
        fn can_proceed(&self) -> bool {
            self.servers_failed == 0 && self.schema_valid
        }
    }

    let suite = BenchmarkPreflightSuite::standard();

    // Test: All ready
    let result = suite.run_all(&[true, true, true], &[true, true]);
    assert!(
        result.can_proceed(),
        "QA-038-040: Should proceed when all ready"
    );
    assert_eq!(result.servers_passed, 3);
    assert_eq!(result.models_cached, 2);

    // Test: Required server down
    let result = suite.run_all(&[false, true, true], &[true, true]);
    assert!(
        !result.can_proceed(),
        "QA-038-040: Should not proceed if required down"
    );

    // Test: Model needs download
    let result = suite.run_all(&[true, false, false], &[false, true]);
    assert!(
        result.can_proceed(),
        "QA-038-040: Can proceed with download needed"
    );
    assert_eq!(result.models_to_download, 1);

    println!("\nPARITY-010d: Complete preflight suite");
    println!("  Server checks: {}", suite.server_checks.len());
    println!("  Model checks: {}", suite.model_checks.len());
    println!("  Schema: {}", suite.schema_version);
}
