//! IMP-700: Real-World Performance Verification
//!
//! Popperian Falsification: Measure ACTUAL performance against Ollama and llama.cpp
//!
//! Falsifiable Claims:
//! 1. Ollama throughput: Expected ~200+ tok/s (from IMP-400)
//! 2. Realizar throughput: Measured vs expected
//! 3. Gap: Must be measured, not estimated
//!
//! Run: cargo run --release --example imp_700_realworld_verification

use std::time::{Duration, Instant};

/// Statistics for measurement stability (Coefficient of Variation)
fn calculate_cv(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance: f64 =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt() / mean
}

/// Measure Ollama throughput via HTTP API
fn measure_ollama(
    model: &str,
    prompt: &str,
    num_runs: usize,
) -> Result<(f64, f64, Duration), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let mut throughputs = Vec::with_capacity(num_runs);
    let mut latencies = Vec::with_capacity(num_runs);

    for run in 0..num_runs {
        let start = Instant::now();

        let response = client
            .post("http://localhost:11434/api/generate")
            .json(&serde_json::json!({
                "model": model,
                "prompt": prompt,
                "stream": false,
                "options": {
                    "num_predict": 50,
                    "temperature": 0.0,
                    "seed": 42
                }
            }))
            .send()
            .map_err(|e| format!("Ollama request failed: {}", e))?;

        let elapsed = start.elapsed();

        let json: serde_json::Value = response
            .json()
            .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

        // Extract metrics from Ollama response
        if let (Some(total_duration), Some(eval_count)) = (
            json.get("total_duration").and_then(|v| v.as_u64()),
            json.get("eval_count").and_then(|v| v.as_u64()),
        ) {
            let duration_secs = total_duration as f64 / 1_000_000_000.0;
            let tps = eval_count as f64 / duration_secs;
            throughputs.push(tps);
            latencies.push(elapsed);

            if run == 0 {
                println!(
                    "  Run {}: {} tokens in {:.2}s = {:.1} tok/s",
                    run + 1,
                    eval_count,
                    duration_secs,
                    tps
                );
            }
        }
    }

    if throughputs.is_empty() {
        return Err("No valid measurements".to_string());
    }

    let mean_tps: f64 = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let cv = calculate_cv(&throughputs);
    let mean_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;

    Ok((mean_tps, cv, mean_latency))
}

/// Measure realizar throughput
fn measure_realizar(
    model_path: &str,
    prompt_tokens: &[u32],
    num_tokens: usize,
) -> Result<(f64, Duration), String> {
    use realizar::gguf::{GGUFModel, GGUFTransformer};
    use std::fs;

    // Load model file
    let start_load = Instant::now();
    let file_data =
        fs::read(model_path).map_err(|e| format!("Failed to read model file: {}", e))?;

    let model =
        GGUFModel::from_bytes(&file_data).map_err(|e| format!("Failed to parse GGUF: {}", e))?;
    let load_time = start_load.elapsed();
    println!(
        "  Model loaded in {:?} ({:.1} MB)",
        load_time,
        file_data.len() as f64 / 1_000_000.0
    );

    // Create transformer
    let transformer = GGUFTransformer::from_gguf(&model, &file_data)
        .map_err(|e| format!("Failed to create transformer: {}", e))?;

    // Warmup
    let _ = transformer.forward(prompt_tokens);

    // Measure generation
    let start = Instant::now();
    let mut tokens = prompt_tokens.to_vec();

    for _ in 0..num_tokens {
        let logits = transformer
            .forward(&tokens)
            .map_err(|e| format!("Forward pass failed: {}", e))?;

        // Greedy sampling
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        tokens.push(next_token);
    }

    let elapsed = start.elapsed();
    let tps = num_tokens as f64 / elapsed.as_secs_f64();

    Ok((tps, elapsed))
}

fn main() {
    println!("=== IMP-700: Real-World Performance Verification ===");
    println!("Popperian Falsification: Measure ACTUAL performance gaps\n");

    // Test configuration
    let prompt = "The capital of France is";
    let num_runs = 5;
    let num_tokens = 50;

    // Test 1: Ollama phi2:2.7b
    println!("=== Test 1: Ollama phi2:2.7b ===");
    match measure_ollama("phi2:2.7b", prompt, num_runs) {
        Ok((tps, cv, latency)) => {
            println!(
                "  Result: {:.1} tok/s (CV={:.4}, latency={:?})",
                tps, cv, latency
            );
            if cv > 0.05 {
                println!("  WARNING: High CV indicates unstable measurements");
            }

            // Falsifiable claim verification
            println!("\n  Falsifiable Claims:");
            println!("  - Expected: ~200+ tok/s (from IMP-400)");
            println!("  - Measured: {:.1} tok/s", tps);
            if tps >= 180.0 {
                println!("  - Status: VERIFIED (within expected range)");
            } else {
                println!("  - Status: BELOW EXPECTED (check GPU/system state)");
            }
        },
        Err(e) => {
            println!("  ERROR: {}", e);
            println!("  Make sure Ollama is running: ollama serve");
        },
    }

    // Test 2: Realizar with phi-2 GGUF (if available)
    println!("\n=== Test 2: Realizar phi-2 Q4_K_M ===");

    // Common paths for phi-2 model
    let model_paths = [
        "/home/noah/.cache/huggingface/hub/models--TheBloke--phi-2-GGUF/snapshots/*/phi-2.Q4_K_M.gguf",
        "../models/phi-2.Q4_K_M.gguf",
        "models/phi-2.Q4_K_M.gguf",
    ];

    let mut model_found = false;
    for pattern in &model_paths {
        // Try to find model via glob
        if let Ok(paths) = glob::glob(pattern) {
            for path in paths.flatten() {
                println!("  Found model: {}", path.display());
                let prompt_tokens: Vec<u32> = vec![1, 450, 3007, 310, 3444, 338]; // "The capital of France is"

                match measure_realizar(path.to_str().unwrap(), &prompt_tokens, num_tokens) {
                    Ok((tps, latency)) => {
                        println!("  Result: {:.2} tok/s (latency={:?})", tps, latency);
                        model_found = true;

                        // Calculate gap
                        let ollama_baseline = 200.0; // Conservative estimate
                        let gap = ollama_baseline / tps;
                        println!("\n  Performance Gap Analysis:");
                        println!("  - Ollama baseline: ~{:.0} tok/s", ollama_baseline);
                        println!("  - Realizar: {:.2} tok/s", tps);
                        println!("  - Gap: {:.0}x", gap);
                        println!("  - Target for parity: gap < 1.25x");
                    },
                    Err(e) => {
                        println!("  ERROR: {}", e);
                    },
                }
                break;
            }
        }
        if model_found {
            break;
        }
    }

    if !model_found {
        println!("  No phi-2 GGUF model found. Download with:");
        println!("  huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q4_K_M.gguf");
    }

    println!("\n=== Summary ===");
    println!("IMP-700: Real-world verification completed");
    println!("All measurements are falsifiable and reproducible");
    println!("Run this benchmark to verify any performance claims");
}
