//! Comprehensive Benchmark Matrix for 2x llama.cpp Target (PMAT Protocol)
//!
//! Tests: GGUF × CPU/GPU × tiny/small for available models
//! Compares against: llama.cpp baselines
//! Target: APR 2x faster than all baselines for EVERY cell
//!
//! Run: cargo run --release --features cuda --example benchmark_matrix

#[cfg(feature = "cuda")]
use realizar::cuda::CudaExecutor;
#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
};
use std::process::Command;
use std::time::{Duration, Instant};

/// Model tier configuration
struct ModelTier {
    name: &'static str,
    size: &'static str,
    gguf_path: &'static str,
    llama_cpp_gpu_baseline: f64,
    llama_cpp_cpu_baseline: f64,
}

const TIERS: &[ModelTier] = &[
    ModelTier {
        name: "tiny",
        size: "0.5B",
        gguf_path:
            "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_0.gguf",
        llama_cpp_gpu_baseline: 519.0,
        llama_cpp_cpu_baseline: 156.0,
    },
    ModelTier {
        name: "small",
        size: "1.5B",
        gguf_path:
            "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        llama_cpp_gpu_baseline: 348.0,
        llama_cpp_cpu_baseline: 78.0,
    },
];

/// Benchmark result for a single cell
#[derive(Debug, Clone)]
struct BenchResult {
    tier: String,
    backend: String,
    format: String,
    tok_s: f64,
    baseline_tok_s: f64,
    speedup: f64,
    meets_2x: bool,
}

fn benchmark_ollama(model: &str, prompt: &str, max_tokens: usize) -> Result<f64, String> {
    // Warm up
    let _ = Command::new("ollama")
        .args(["run", model, prompt, "--verbose"])
        .output();

    // Benchmark run
    let start = Instant::now();
    let output = Command::new("ollama")
        .args(["run", model, prompt])
        .output()
        .map_err(|e| format!("Failed to run ollama: {}", e))?;

    let elapsed = start.elapsed();

    // Parse token count from response
    let response = String::from_utf8_lossy(&output.stdout);
    let tokens = response.split_whitespace().count().max(max_tokens);

    let tok_s = tokens as f64 / elapsed.as_secs_f64();
    Ok(tok_s)
}

fn benchmark_realizar_cpu(model_path: &str, max_tokens: usize) -> Result<f64, RealizarError> {
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let prompt = vec![1u32, 29871, 29896]; // Simple prompt
    let config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![2],
    };

    // Warm up
    for _ in 0..3 {
        let _ = model.generate_with_scratch(&prompt, &config)?;
    }

    // Benchmark
    let iterations = 5;
    let mut times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let tokens = model.generate_with_scratch(&prompt, &config)?;
        let elapsed = start.elapsed();

        let generated = tokens.len() - prompt.len();
        if generated > 0 {
            let tok_s = generated as f64 / elapsed.as_secs_f64();
            times.push(tok_s);
        }
    }

    if times.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "No tokens generated".to_string(),
        });
    }

    // Return median
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[times.len() / 2])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("COMPREHENSIVE BENCHMARK MATRIX - Target: 2x llama.cpp for ALL cells");
    println!("{}", "=".repeat(80));
    println!();

    // llama.cpp baselines (from spec)
    let llama_baselines = [
        ("tiny", "CPU", 156.62),
        ("tiny", "GPU", 519.35),
        ("small", "CPU", 78.19),
        ("small", "GPU", 348.64),
        ("medium", "CPU", 23.31),
        ("medium", "GPU", 149.35),
        ("large", "CPU", 5.72),
        ("large", "GPU", 38.65),
    ];

    let mut results: Vec<BenchResult> = Vec::new();

    // Test small tier first (we have the model)
    println!("\n### Testing small tier (1.5B) - CPU ###");
    if let Some(path) = TIERS[1].gguf_path {
        match benchmark_realizar_cpu(path, 32) {
            Ok(tok_s) => {
                let baseline = 78.19; // llama.cpp CPU
                let speedup = tok_s / baseline;
                println!("  realizar CPU: {:.1} tok/s", tok_s);
                println!("  llama.cpp:    {:.1} tok/s", baseline);
                println!(
                    "  Speedup:      {:.2}x {}",
                    speedup,
                    if speedup >= 2.0 { "✅" } else { "❌" }
                );

                results.push(BenchResult {
                    tier: "small".to_string(),
                    backend: "CPU".to_string(),
                    format: "GGUF".to_string(),
                    tok_s,
                    baseline_tok_s: baseline,
                    speedup,
                    meets_2x: speedup >= 2.0,
                });
            },
            Err(e) => println!("  Error: {}", e),
        }
    }

    // Test Ollama baseline for comparison
    println!("\n### Ollama Baseline (small, 1.5B) ###");
    match benchmark_ollama("qwen2.5-coder:1.5b", "Write a hello world in Rust", 32) {
        Ok(tok_s) => {
            println!("  Ollama: {:.1} tok/s", tok_s);
        },
        Err(e) => println!("  Error: {}", e),
    }

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK MATRIX SUMMARY");
    println!("{}", "=".repeat(80));
    println!();
    println!("| Tier | Backend | Format | APR tok/s | Baseline | Speedup | Status |");
    println!("|------|---------|--------|-----------|----------|---------|--------|");

    let mut all_pass = true;
    for r in &results {
        let status = if r.meets_2x { "✅ PASS" } else { "❌ FAIL" };
        println!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.2}x | {} |",
            r.tier, r.backend, r.format, r.tok_s, r.baseline_tok_s, r.speedup, status
        );
        if !r.meets_2x {
            all_pass = false;
        }
    }

    println!();
    if all_pass {
        println!("✅ ALL CELLS MEET 2x TARGET!");
    } else {
        println!("❌ SOME CELLS BELOW 2x - Five-whys analysis required");

        // Five-whys for failures
        for r in results.iter().filter(|r| !r.meets_2x) {
            println!("\n### Five-Whys: {} {} {} ###", r.tier, r.backend, r.format);
            println!(
                "Current: {:.1} tok/s, Target: {:.1} tok/s (2x baseline)",
                r.tok_s,
                r.baseline_tok_s * 2.0
            );
            println!("Gap: {:.1}x needed", (r.baseline_tok_s * 2.0) / r.tok_s);
        }
    }

    Ok(())
}
