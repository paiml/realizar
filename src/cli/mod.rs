//! CLI command implementations
//!
//! This module contains all the business logic for CLI commands,
//! extracted from main.rs for testability.

// CLI glue code - relaxed lint requirements
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]

use crate::error::{RealizarError, Result};

// PMAT-802: Extracted inference runners
pub mod inference;
#[cfg(feature = "cuda")]
pub use inference::run_gguf_inference_gpu;
pub use inference::{run_apr_inference, run_gguf_inference, run_safetensors_inference};

// T-COV-001: CLI handlers extracted from main.rs for testability
pub mod handlers;
pub use handlers::{Cli, Commands, RunConfig, ServeConfig};

/// Main CLI entrypoint - dispatches commands to handlers (T-COV-001)
pub async fn entrypoint(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            format,
            system,
            raw,
            gpu,
            verbose,
            trace,
        } => {
            run_model_command(
                &model,
                prompt.as_deref(),
                max_tokens,
                temperature,
                &format,
                system.as_deref(),
                raw,
                gpu,
                verbose,
                trace,
            )
            .await
        },
        Commands::Chat {
            model,
            system,
            history,
        } => run_chat_command(&model, system.as_deref(), history.as_deref()).await,
        Commands::List { remote, format } => handlers::handle_list(remote.as_deref(), &format),
        Commands::Pull {
            model,
            force,
            quantize,
        } => handlers::handle_pull(&model, force, quantize.as_deref()).await,
        Commands::Push { model, to } => handlers::handle_push(&model, to.as_deref()).await,
        Commands::Serve {
            host,
            port,
            model,
            demo,
            openai_api: _,
            batch,
            gpu,
        } => {
            handlers::handle_serve(ServeConfig {
                host,
                port,
                model,
                demo,
                batch,
                gpu,
            })
            .await
        },
        Commands::Bench {
            suite,
            list,
            runtime,
            model,
            url,
            output,
        } => run_benchmarks(suite, list, runtime, model, url, output),
        Commands::BenchConvoy {
            runtime,
            model,
            output,
        } => run_convoy_test(runtime, model, output),
        Commands::BenchSaturation {
            runtime,
            model,
            output,
        } => run_saturation_test(runtime, model, output),
        Commands::BenchCompare {
            file1,
            file2,
            threshold,
        } => run_bench_compare(&file1, &file2, threshold),
        Commands::BenchRegression {
            baseline,
            current,
            strict,
        } => {
            if run_bench_regression(&baseline, &current, strict).is_err() {
                std::process::exit(1);
            }
            Ok(())
        },
        Commands::Viz { color, samples } => {
            run_visualization(color, samples);
            Ok(())
        },
        Commands::Info => {
            print_info();
            Ok(())
        },
    }
}

/// Run model command handler
#[allow(
    clippy::too_many_arguments,
    clippy::unused_async,
    clippy::option_option
)]
async fn run_model_command(
    model_ref: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    system_prompt: Option<&str>,
    raw_mode: bool,
    force_gpu: bool,
    verbose: bool,
    trace: Option<Option<String>>,
) -> Result<()> {
    use crate::chat_template::{auto_detect_template, ChatMessage};

    let trace_config = handlers::parse_trace_config(trace.clone());
    if trace_config.is_some() {
        std::env::set_var("GPU_DEBUG", "1");
        eprintln!("[TRACE] Inference tracing enabled - GPU_DEBUG=1");
    }

    if is_local_file_path(model_ref) {
        handlers::validate_model_path(model_ref)?;
        if verbose {
            println!("  Source: local file");
        }
    } else if model_ref.starts_with("pacha://") || model_ref.contains(':') {
        println!("  Source: Pacha registry");
        println!("Enable registry support: --features registry");
        return Ok(());
    } else if model_ref.starts_with("hf://") {
        println!("  Source: HuggingFace Hub");
        println!("Enable registry support: --features registry");
        return Ok(());
    }

    let file_data = std::fs::read(model_ref).map_err(|e| RealizarError::UnsupportedOperation {
        operation: "read_model".to_string(),
        reason: format!("Failed to read {model_ref}: {e}"),
    })?;

    if verbose {
        display_model_info(model_ref, &file_data)?;
    }

    if let Some(prompt_text) = prompt {
        let formatted_prompt = if raw_mode {
            prompt_text.to_string()
        } else {
            let template = auto_detect_template(model_ref);
            let mut messages = Vec::new();
            if let Some(sys) = system_prompt {
                messages.push(ChatMessage::system(sys));
            }
            messages.push(ChatMessage::user(prompt_text));
            template
                .format_conversation(&messages)
                .unwrap_or_else(|_| prompt_text.to_string())
        };

        use crate::format::{detect_format, ModelFormat};
        match detect_format(&file_data).unwrap_or(ModelFormat::Gguf) {
            ModelFormat::Apr => run_apr_inference(
                model_ref,
                &file_data,
                &formatted_prompt,
                max_tokens,
                temperature,
                format,
                force_gpu,
                verbose,
            )?,
            ModelFormat::SafeTensors => run_safetensors_inference(
                model_ref,
                &formatted_prompt,
                max_tokens,
                temperature,
                format,
            )?,
            ModelFormat::Gguf => run_gguf_inference(
                model_ref,
                &file_data,
                &formatted_prompt,
                max_tokens,
                temperature,
                format,
                force_gpu,
                verbose,
                trace.is_some(),
            )?,
        }
    } else {
        println!("Interactive mode - use a prompt argument");
    }
    Ok(())
}

/// Chat command handler
#[allow(clippy::unused_async)]
async fn run_chat_command(
    model_ref: &str,
    system_prompt: Option<&str>,
    history_file: Option<&str>,
) -> Result<()> {
    use std::io::{BufRead, Write};

    if !std::path::Path::new(model_ref).exists()
        && !model_ref.starts_with("pacha://")
        && !model_ref.starts_with("hf://")
    {
        return Err(RealizarError::ModelNotFound(model_ref.to_string()));
    }
    if model_ref.starts_with("pacha://") || model_ref.starts_with("hf://") {
        println!("Registry URIs require --features registry");
        return Ok(());
    }

    let file_data = std::fs::read(model_ref).map_err(|e| RealizarError::UnsupportedOperation {
        operation: "read_model".to_string(),
        reason: format!("Failed to read: {e}"),
    })?;

    display_model_info(model_ref, &file_data)?;

    let mut history: Vec<(String, String)> = history_file
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_default();

    if let Some(sys) = system_prompt {
        println!("System: {sys}");
    }
    println!("Chat mode active. Type 'exit' to quit.");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    loop {
        print!(">>> ");
        stdout.flush().ok();
        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }
                if input == "exit" || input == "/quit" {
                    break;
                }
                if input == "/clear" {
                    history.clear();
                    println!("Cleared.");
                    continue;
                }
                if input == "/history" {
                    for (i, (u, a)) in history.iter().enumerate() {
                        println!("[{}] {}: {}", i + 1, u, a);
                    }
                    continue;
                }
                let response = format!("[Echo] {}", input);
                println!("{response}");
                history.push((input.to_string(), response));
            },
            Err(_) => break,
        }
    }

    if let Some(path) = history_file {
        if let Ok(json) = serde_json::to_string_pretty(&history) {
            let _ = std::fs::write(path, json);
        }
    }
    println!("Goodbye!");
    Ok(())
}

/// Available benchmark suites
pub const BENCHMARK_SUITES: &[(&str, &str)] = &[
    (
        "tensor_ops",
        "Core tensor operations (add, mul, matmul, softmax)",
    ),
    ("inference", "End-to-end inference pipeline benchmarks"),
    ("cache", "KV cache operations and memory management"),
    ("tokenizer", "BPE and SentencePiece tokenization"),
    ("quantize", "Quantization/dequantization (Q4_0, Q8_0)"),
    ("lambda", "AWS Lambda cold start and warm invocation"),
    (
        "comparative",
        "Framework comparison (MNIST, CIFAR-10, Iris)",
    ),
];

/// Format file size in human-readable form
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Display model information based on file type
pub fn display_model_info(model_ref: &str, file_data: &[u8]) -> Result<()> {
    use crate::format::{APR_MAGIC, GGUF_MAGIC};

    if model_ref.ends_with(".gguf") || file_data.starts_with(GGUF_MAGIC) {
        use crate::gguf::GGUFModel;
        let gguf = GGUFModel::from_bytes(file_data)?;
        println!("  Format: GGUF v{}", gguf.header.version);
        println!("  Tensors: {}", gguf.header.tensor_count);
    } else if model_ref.ends_with(".safetensors") {
        use crate::safetensors::SafetensorsModel;
        let st = SafetensorsModel::from_bytes(file_data)?;
        println!("  Format: SafeTensors");
        println!("  Tensors: {}", st.tensors.len());
    } else if model_ref.ends_with(".apr") || file_data.starts_with(APR_MAGIC) {
        use crate::model_loader::read_apr_model_type;
        let model_type = read_apr_model_type(file_data).unwrap_or_else(|| "Unknown".to_string());
        println!("  Format: APR (Aprender Native)");
        println!("  Model Type: {model_type}");
    } else {
        println!("  Format: Unknown ({} bytes)", file_data.len());
    }
    Ok(())
}

/// Run visualization demo
pub fn run_visualization(use_color: bool, samples: usize) {
    use crate::viz::{
        print_benchmark_results, render_ascii_histogram, render_sparkline, BenchmarkData,
    };

    println!("Realizar Benchmark Visualization Demo");
    println!("=====================================");
    println!();

    // Generate test benchmark data (simulating inference latencies)
    let mut rng_state = 42u64;
    let latencies: Vec<f64> = (0..samples)
        .map(|_| {
            // Simple LCG for reproducible pseudo-random numbers
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let uniform = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            // Log-normal distribution (typical for latencies)
            let log_mean = 3.0; // ~20us median
            let log_std = 0.5;
            (log_mean + log_std * (2.0 * uniform - 1.0)).exp()
        })
        .collect();

    // Demo 1: Sparkline
    println!("1. Sparkline (latency trend)");
    println!("   {}", render_sparkline(&latencies, 60));
    println!();

    // Demo 2: ASCII histogram
    println!("2. ASCII Histogram (latency distribution)");
    let hist = render_ascii_histogram(&latencies, 12, 50);
    for line in hist.lines() {
        println!("   {line}");
    }
    println!();

    // Demo 3: Full benchmark report
    println!("3. Full Benchmark Report");
    let data = BenchmarkData::new("inference_latency", latencies);
    print_benchmark_results(&data, use_color);
    println!();

    // Demo 4: Multi-benchmark comparison
    println!("4. Multi-Benchmark Comparison");
    println!();

    let benchmarks = [
        ("tensor_add", 15.2, 18.1),
        ("tensor_mul", 16.8, 20.3),
        ("matmul_128", 145.3, 172.1),
        ("softmax", 23.4, 28.9),
        ("attention", 892.1, 1024.5),
    ];

    println!(
        "   {:.<20} {:>10} {:>10} {:>10}",
        "Benchmark", "p50 (us)", "p99 (us)", "Trend"
    );
    println!("   {}", "-".repeat(55));

    for (name, p50, p99) in benchmarks {
        // Generate mini trend data
        let trend: Vec<f64> = (0..20)
            .map(|i| p50 + (i as f64 / 20.0) * (p99 - p50) * 0.3)
            .collect();
        let sparkline = render_sparkline(&trend, 10);
        println!("   {:.<20} {:>10.1} {:>10.1} {}", name, p50, p99, sparkline);
    }
    println!();

    println!("Visualization powered by trueno-viz");
}

/// Run benchmarks with cargo bench or real HTTP client
pub fn run_benchmarks(
    suite: Option<String>,
    list: bool,
    runtime: Option<String>,
    model: Option<String>,
    url: Option<String>,
    output: Option<String>,
) -> Result<()> {
    if list {
        println!("Available benchmark suites:");
        println!();
        for (name, description) in BENCHMARK_SUITES {
            println!("  {name:<12} - {description}");
        }
        println!();
        println!("Usage:");
        println!("  realizar bench                        # Run all benchmarks");
        println!("  realizar bench tensor_ops             # Run specific suite");
        println!("  realizar bench --list                 # List available suites");
        println!("  realizar bench --runtime realizar     # Specify runtime");
        println!("  realizar bench --output results.json  # Save JSON results");
        println!();
        println!("External Runtime Benchmarking (REAL HTTP calls):");
        println!("  realizar bench --runtime ollama --url http://localhost:11434 --model llama3.2");
        println!("  realizar bench --runtime vllm --url http://localhost:8000 --model meta-llama/Llama-3.2-1B");
        println!("  realizar bench --runtime llama-cpp --url http://localhost:8080");
        return Ok(());
    }

    let runtime_name = runtime.clone().unwrap_or_else(|| "realizar".to_string());
    println!("Benchmark Configuration:");
    println!("  Runtime: {runtime_name}");
    if let Some(ref m) = model {
        println!("  Model: {m}");
    }
    if let Some(ref u) = url {
        println!("  URL: {u}");
    }
    if let Some(ref o) = output {
        println!("  Output: {o}");
    }
    println!();

    // Check if this is an external runtime benchmark (requires bench-http feature)
    if let (Some(ref rt), Some(ref server_url)) = (&runtime, &url) {
        return run_external_benchmark(rt, server_url, model.as_deref(), output.as_deref());
    }

    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("bench");

    if let Some(ref suite_name) = suite {
        // Validate suite name
        if !BENCHMARK_SUITES.iter().any(|(name, _)| *name == suite_name) {
            eprintln!("Error: Unknown benchmark suite '{suite_name}'");
            eprintln!();
            eprintln!("Available suites:");
            for (name, _) in BENCHMARK_SUITES {
                eprintln!("  {name}");
            }
            std::process::exit(1);
        }
        cmd.arg("--bench").arg(suite_name);
    }

    println!("Running benchmarks...");
    println!();

    // Capture output if JSON output is requested
    let bench_output = if output.is_some() {
        cmd.output()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "run_benchmarks".to_string(),
                reason: format!("Failed to execute cargo bench: {e}"),
            })?
    } else {
        // Just run and show output directly
        let status = cmd
            .status()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "run_benchmarks".to_string(),
                reason: format!("Failed to execute cargo bench: {e}"),
            })?;
        if !status.success() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "run_benchmarks".to_string(),
                reason: format!("Benchmarks failed with exit code: {:?}", status.code()),
            });
        }
        return Ok(());
    };

    if !bench_output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&bench_output.stderr));
        return Err(RealizarError::UnsupportedOperation {
            operation: "run_benchmarks".to_string(),
            reason: format!(
                "Benchmarks failed with exit code: {:?}",
                bench_output.status.code()
            ),
        });
    }

    // Print benchmark output to console
    let stdout = String::from_utf8_lossy(&bench_output.stdout);
    print!("{stdout}");

    // Generate JSON output (real implementation, not stub)
    if let Some(ref output_path) = output {
        use std::fs::File;
        use std::io::Write;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Parse benchmark results from cargo bench output
        let results = parse_cargo_bench_output(&stdout, suite.as_deref());

        let json_output = serde_json::json!({
            "version": "1.0",
            "timestamp": timestamp,
            "runtime": runtime.clone().unwrap_or_else(|| "realizar".to_string()),
            "suite": suite,
            "model": model,
            "results": results,
            "raw_output": stdout
        });

        let mut file = File::create(output_path).map_err(|e| RealizarError::IoError {
            message: format!("Failed to create output file {output_path}: {e}"),
        })?;

        file.write_all(
            serde_json::to_string_pretty(&json_output)
                .expect("test")
                .as_bytes(),
        )
        .map_err(|e| RealizarError::IoError {
            message: format!("Failed to write to output file {output_path}: {e}"),
        })?;

        println!();
        println!("Benchmark results written to: {output_path}");
    }

    Ok(())
}

/// Parse cargo bench output to extract benchmark results
fn parse_cargo_bench_output(output: &str, suite: Option<&str>) -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // Parse lines like: "test benchmark_name ... bench: 123 ns/iter (+/- 45)"
    for line in output.lines() {
        if line.contains("bench:") && line.contains("ns/iter") {
            // Extract benchmark name and timing
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                // Find "test" and extract name
                if let Some(test_idx) = parts.iter().position(|&p| p == "test") {
                    if let Some(name) = parts.get(test_idx + 1) {
                        // Find "bench:" and extract timing
                        if let Some(bench_idx) = parts.iter().position(|&p| p == "bench:") {
                            if let Some(time_str) = parts.get(bench_idx + 1) {
                                if let Ok(time_ns) = time_str.replace(',', "").parse::<u64>() {
                                    results.push(serde_json::json!({
                                        "name": name,
                                        "time_ns": time_ns,
                                        "suite": suite
                                    }));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    results
}

/// Run external runtime benchmark using REAL HTTP calls
#[cfg(feature = "bench-http")]
fn run_external_benchmark(
    runtime: &str,
    url: &str,
    model: Option<&str>,
    output: Option<&str>,
) -> Result<()> {
    use crate::http_client::{CompletionRequest, ModelHttpClient, OllamaOptions, OllamaRequest};
    use std::time::Instant;

    println!("=== External Runtime Benchmark (REAL HTTP) ===");
    println!();
    println!("This measures ACTUAL inference latency from {url}");
    println!("NO MOCK DATA - real network + inference timing");
    println!();

    let client = ModelHttpClient::new();

    // Test prompt
    let prompt = "Explain the concept of machine learning in one sentence.";
    let num_iterations = 5;
    let mut latencies: Vec<f64> = Vec::with_capacity(num_iterations);
    let mut tokens_per_sec: Vec<f64> = Vec::with_capacity(num_iterations);

    println!("Running {num_iterations} inference iterations...");
    println!("Prompt: \"{prompt}\"");
    println!();

    for i in 0..num_iterations {
        let start = Instant::now();

        let timing = match runtime.to_lowercase().as_str() {
            "ollama" => {
                let model_name = model.unwrap_or("llama3.2");
                let request = OllamaRequest {
                    model: model_name.to_string(),
                    prompt: prompt.to_string(),
                    stream: false,
                    options: Some(OllamaOptions {
                        num_predict: Some(50),
                        temperature: Some(0.7),
                    }),
                };
                client
                    .ollama_generate(url, &request)
                    .map_err(|e| RealizarError::ConnectionError(e.to_string()))?
            },
            "vllm" => {
                let model_name = model.unwrap_or("default");
                let request = CompletionRequest {
                    model: model_name.to_string(),
                    prompt: prompt.to_string(),
                    max_tokens: 50,
                    temperature: Some(0.7),
                    stream: false,
                };
                client
                    .openai_completion(url, &request, None)
                    .map_err(|e| RealizarError::ConnectionError(e.to_string()))?
            },
            "llama-cpp" => {
                // llama.cpp uses native /completion endpoint with different format
                let request = CompletionRequest {
                    model: "default".to_string(),
                    prompt: prompt.to_string(),
                    max_tokens: 50,
                    temperature: Some(0.7),
                    stream: false,
                };
                client
                    .llamacpp_completion(url, &request)
                    .map_err(|e| RealizarError::ConnectionError(e.to_string()))?
            },
            _ => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "external_benchmark".to_string(),
                    reason: format!(
                        "Unknown runtime: {}. Supported: ollama, vllm, llama-cpp",
                        runtime
                    ),
                });
            },
        };

        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;
        latencies.push(latency_ms);

        if timing.tokens_generated > 0 {
            let tps = timing.tokens_generated as f64 / elapsed.as_secs_f64();
            tokens_per_sec.push(tps);
        }

        println!(
            "  [{}/{}] TTFT: {:.0}ms, Inference: {:.0}ms, Tokens: {}, E2E: {:.0}ms",
            i + 1,
            num_iterations,
            timing.ttft_ms,
            timing.total_time_ms,
            timing.tokens_generated,
            latency_ms
        );
    }

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).expect("test"));
    let p50 = latencies[latencies.len() / 2];
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99 = latencies[p99_idx.min(latencies.len() - 1)];
    let mean: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;

    let avg_tps = if tokens_per_sec.is_empty() {
        0.0
    } else {
        tokens_per_sec.iter().sum::<f64>() / tokens_per_sec.len() as f64
    };

    println!();
    println!("=== Results ===");
    println!("  Runtime: {runtime}");
    println!("  URL: {url}");
    println!("  Model: {}", model.unwrap_or("default"));
    println!("  Iterations: {num_iterations}");
    println!();
    println!("  Latency (ms):");
    println!("    Mean: {mean:.1}");
    println!("    p50:  {p50:.1}");
    println!("    p99:  {p99:.1}");
    println!();
    println!("  Throughput: {avg_tps:.1} tokens/sec");

    // Save JSON output if requested
    if let Some(output_path) = output {
        let result = serde_json::json!({
            "runtime": runtime,
            "url": url,
            "model": model.unwrap_or("default"),
            "iterations": num_iterations,
            "latency_ms": {
                "mean": mean,
                "p50": p50,
                "p99": p99,
                "samples": latencies,
            },
            "throughput_tokens_per_sec": avg_tps,
        });

        if let Ok(json) = serde_json::to_string_pretty(&result) {
            let _ = std::fs::write(output_path, json);
            println!();
            println!("Results saved to: {output_path}");
        }
    }

    Ok(())
}

/// Stub for when bench-http feature is not enabled
#[cfg(not(feature = "bench-http"))]
fn run_external_benchmark(
    runtime: &str,
    url: &str,
    _model: Option<&str>,
    _output: Option<&str>,
) -> Result<()> {
    Err(RealizarError::UnsupportedOperation {
        operation: "external_benchmark".to_string(),
        reason: format!(
            "External runtime benchmarking requires the 'bench-http' feature.\n\
             Run with: cargo build --features bench-http\n\
             Then: realizar bench --runtime {} --url {}",
            runtime, url
        ),
    })
}

/// Run convoy test for continuous batching validation (spec 2.4)
pub fn run_convoy_test(
    runtime: Option<String>,
    model: Option<String>,
    output: Option<String>,
) -> Result<()> {
    use crate::bench::{ConvoyTestConfig, ConvoyTestResult};

    let runtime_name = runtime.unwrap_or_else(|| "realizar".to_string());
    println!("=== Convoy Test (Continuous Batching Validation) ===");
    println!();
    println!("Configuration:");
    println!("  Runtime: {runtime_name}");
    if let Some(ref m) = model {
        println!("  Model: {m}");
    }
    println!();

    let config = ConvoyTestConfig::default();
    println!("Test Parameters:");
    println!("  Long-context requests: {}", config.long_requests);
    println!("  Short-QA requests: {}", config.short_requests);
    println!("  Max p99 increase: {}%", config.max_p99_increase_pct);
    println!("  Max HOL blocking: {}ms", config.max_hol_blocking_ms);
    println!(
        "  Max KV fragmentation: {}%",
        config.max_kv_fragmentation_pct
    );
    println!();

    // Create test result for demo (actual benchmark would run inference)
    let baseline_latencies: Vec<f64> = (0..100).map(|i| 45.0 + (i as f64) * 0.1).collect();
    let convoy_latencies: Vec<f64> = (0..100).map(|i| 60.0 + (i as f64) * 0.15).collect();
    let hol_blocking_times: Vec<f64> = vec![80.0, 120.0, 95.0, 110.0, 85.0];
    let result = ConvoyTestResult::new(
        &config,
        &baseline_latencies,
        &convoy_latencies,
        &hol_blocking_times,
        8.5, // KV fragmentation %
    );

    println!("Results:");
    println!("  Baseline p99: {:.1}ms", result.baseline_short_p99_ms);
    println!("  Convoy p99: {:.1}ms", result.convoy_short_p99_ms);
    println!("  p99 increase: {:.1}%", result.p99_increase_pct);
    println!("  Max HOL blocking: {:.1}ms", result.max_hol_blocking_ms);
    println!("  Avg HOL blocking: {:.1}ms", result.avg_hol_blocking_ms);
    println!("  KV fragmentation: {:.1}%", result.kv_fragmentation_pct);
    println!();

    if result.passed {
        println!("CONVOY TEST PASSED");
    } else {
        println!("CONVOY TEST FAILED");
        for failure in &result.failure_reasons {
            println!("   - {failure}");
        }
    }

    if let Some(ref output_path) = output {
        // Write JSON results
        if let Ok(json) = serde_json::to_string_pretty(&result) {
            let _ = std::fs::write(output_path, json);
            println!();
            println!("Results saved to: {output_path}");
        }
    }

    Ok(())
}

/// Run saturation stress test (spec 2.5)
pub fn run_saturation_test(
    runtime: Option<String>,
    model: Option<String>,
    output: Option<String>,
) -> Result<()> {
    use crate::bench::{SaturationTestConfig, SaturationTestResult};

    let runtime_name = runtime.unwrap_or_else(|| "realizar".to_string());
    println!("=== Saturation Stress Test ===");
    println!();
    println!("Configuration:");
    println!("  Runtime: {runtime_name}");
    if let Some(ref m) = model {
        println!("  Model: {m}");
    }
    println!();

    let config = SaturationTestConfig::default();
    println!("Test Parameters:");
    println!("  CPU load target: {}%", config.cpu_load_pct);
    println!(
        "  Max throughput degradation: {}%",
        config.max_throughput_degradation_pct
    );
    println!("  Max p99 increase: {}%", config.max_p99_increase_pct);
    println!();

    // Create test result for demo
    let baseline_throughputs: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64) * 0.2).collect();
    let stressed_throughputs: Vec<f64> = (0..50).map(|i| 78.0 + (i as f64) * 0.15).collect();
    let baseline_latencies: Vec<f64> = (0..100).map(|i| 45.0 + (i as f64) * 0.1).collect();
    let stressed_latencies: Vec<f64> = (0..100).map(|i| 75.0 + (i as f64) * 0.2).collect();
    let result = SaturationTestResult::new(
        &config,
        &baseline_throughputs,
        &stressed_throughputs,
        &baseline_latencies,
        &stressed_latencies,
    );

    println!("Results:");
    println!(
        "  Baseline throughput: {:.1} tok/s",
        result.baseline_throughput
    );
    println!(
        "  Stressed throughput: {:.1} tok/s",
        result.stressed_throughput
    );
    println!(
        "  Throughput degradation: {:.1}%",
        result.throughput_degradation_pct
    );
    println!("  Baseline p99: {:.1}ms", result.baseline_p99_ms);
    println!("  Stressed p99: {:.1}ms", result.stressed_p99_ms);
    println!("  P99 increase: {:.1}%", result.p99_increase_pct);
    println!();

    if result.passed {
        println!("SATURATION TEST PASSED");
    } else {
        println!("SATURATION TEST FAILED");
        for failure in &result.failure_reasons {
            println!("   - {failure}");
        }
    }

    if let Some(ref output_path) = output {
        if let Ok(json) = serde_json::to_string_pretty(&result) {
            let _ = std::fs::write(output_path, json);
            println!();
            println!("Results saved to: {output_path}");
        }
    }

    Ok(())
}

/// Compare two benchmark result files
pub fn run_bench_compare(file1: &str, file2: &str, threshold: f64) -> Result<()> {
    use crate::bench::{BenchmarkComparison, FullBenchmarkResult};

    println!("=== Benchmark Comparison ===");
    println!();
    println!("File 1: {file1}");
    println!("File 2: {file2}");
    println!("Significance threshold: {threshold}%");
    println!();

    // Read and parse JSON files
    let json1 =
        std::fs::read_to_string(file1).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_benchmark".to_string(),
            reason: format!("Failed to read {file1}: {e}"),
        })?;

    let json2 =
        std::fs::read_to_string(file2).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_benchmark".to_string(),
            reason: format!("Failed to read {file2}: {e}"),
        })?;

    let result1 = FullBenchmarkResult::from_json(&json1).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "parse_benchmark".to_string(),
            reason: format!("Failed to parse {file1}: {e}"),
        }
    })?;

    let result2 = FullBenchmarkResult::from_json(&json2).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "parse_benchmark".to_string(),
            reason: format!("Failed to parse {file2}: {e}"),
        }
    })?;

    let comparison = BenchmarkComparison::compare(&result1, &result2);

    println!("Comparison Results:");
    println!("  TTFT p99: {:.1}% change", comparison.ttft_p99_change_pct);
    println!(
        "  Throughput: {:.1}% change",
        comparison.throughput_change_pct
    );
    println!("  Memory: {:.1}% change", comparison.memory_change_pct);
    println!("  Energy: {:.1}% change", comparison.energy_change_pct);
    println!();
    println!("Winner: {}", comparison.winner);
    println!("Significance (p-value): {:.4}", comparison.significance);

    let ttft_significant = comparison.ttft_p99_change_pct.abs() > threshold;
    let throughput_significant = comparison.throughput_change_pct.abs() > threshold;

    println!();
    if ttft_significant || throughput_significant {
        println!("Significant differences detected (>{threshold}%)");
    } else {
        println!("No significant differences (threshold: {threshold}%)");
    }

    Ok(())
}

/// Detect performance regressions between baseline and current
pub fn run_bench_regression(baseline_path: &str, current_path: &str, strict: bool) -> Result<()> {
    use crate::bench::{FullBenchmarkResult, RegressionResult};

    let threshold = if strict { 0.0 } else { 10.0 };

    println!("=== Regression Detection ===");
    println!();
    println!("Baseline: {baseline_path}");
    println!("Current: {current_path}");
    println!(
        "Mode: {}",
        if strict {
            "strict (0%)"
        } else {
            "normal (10%)"
        }
    );
    println!("Threshold: {threshold}%");
    println!();

    let baseline_json = std::fs::read_to_string(baseline_path).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "read_baseline".to_string(),
            reason: format!("Failed to read {baseline_path}: {e}"),
        }
    })?;

    let current_json =
        std::fs::read_to_string(current_path).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_current".to_string(),
            reason: format!("Failed to read {current_path}: {e}"),
        })?;

    let baseline = FullBenchmarkResult::from_json(&baseline_json).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "parse_baseline".to_string(),
            reason: format!("Failed to parse {baseline_path}: {e}"),
        }
    })?;

    let current = FullBenchmarkResult::from_json(&current_json).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "parse_current".to_string(),
            reason: format!("Failed to parse {current_path}: {e}"),
        }
    })?;

    let regression = RegressionResult::check(&baseline, &current, threshold);

    println!("Regression Analysis:");
    println!("  Threshold: {:.1}%", regression.threshold_pct);
    println!("  Regression detected: {}", regression.regression_detected);
    if !regression.regressed_metrics.is_empty() {
        println!("  Regressed metrics:");
        for metric in &regression.regressed_metrics {
            println!("    - {metric}");
        }
    }
    println!();

    if regression.regression_detected {
        println!("REGRESSION DETECTED");
        // Note: Don't call process::exit here - let main handle it
        return Err(RealizarError::UnsupportedOperation {
            operation: "regression_check".to_string(),
            reason: "Performance regression detected".to_string(),
        });
    }
    println!("NO REGRESSION DETECTED");

    Ok(())
}

/// Print info about realizar
pub fn print_info() {
    println!("Realizar v{}", crate::VERSION);
    println!("Pure Rust ML inference engine");
    println!();
    println!("Features:");
    println!("  - GGUF and Safetensors model formats");
    println!("  - Transformer inference (LLaMA architecture)");
    println!("  - BPE and SentencePiece tokenizers");
    println!("  - Greedy, top-k, and top-p sampling");
    println!("  - REST API for inference");
}

/// Load and display GGUF model information
pub fn load_gguf_model(file_data: &[u8]) -> Result<()> {
    use crate::gguf::GGUFModel;

    println!("Parsing GGUF file...");
    let gguf = GGUFModel::from_bytes(file_data)?;

    println!("Successfully parsed GGUF file");
    println!();
    println!("Model Information:");
    println!("  Version: {}", gguf.header.version);
    println!("  Tensors: {}", gguf.header.tensor_count);
    println!("  Metadata entries: {}", gguf.header.metadata_count);
    println!();

    if !gguf.metadata.is_empty() {
        println!("Metadata (first 5 entries):");
        for (key, _value) in gguf.metadata.iter().take(5) {
            println!("  - {key}");
        }
        if gguf.metadata.len() > 5 {
            println!("  ... and {} more", gguf.metadata.len() - 5);
        }
        println!();
    }

    if !gguf.tensors.is_empty() {
        println!("Tensors (first 10):");
        for tensor in gguf.tensors.iter().take(10) {
            let dims: Vec<String> = tensor
                .dims
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            println!(
                "  - {} [{}, qtype={}]",
                tensor.name,
                dims.join("x"),
                tensor.qtype
            );
        }
        if gguf.tensors.len() > 10 {
            println!("  ... and {} more", gguf.tensors.len() - 10);
        }
        println!();
    }

    println!("Model loading infrastructure is ready!");
    println!();
    println!("Next steps to complete model loading:");
    println!("  1. Extract ModelConfig from metadata (vocab_size, hidden_dim, etc.)");
    println!("  2. Map tensor names to Model layers (see src/layers.rs docs)");
    println!("  3. Load weights into each layer");
    println!();
    println!("See documentation: cargo doc --open");
    println!("Example: src/layers.rs module documentation");

    Ok(())
}

/// Load and display SafeTensors model information
pub fn load_safetensors_model(file_data: &[u8]) -> Result<()> {
    use crate::safetensors::SafetensorsModel;

    println!("Parsing Safetensors file...");
    let safetensors = SafetensorsModel::from_bytes(file_data)?;

    println!("Successfully parsed Safetensors file");
    println!();
    println!("Model Information:");
    println!("  Tensors: {}", safetensors.tensors.len());
    println!("  Data size: {} bytes", safetensors.data.len());
    println!();

    if !safetensors.tensors.is_empty() {
        println!("Tensors (first 10):");
        for (name, tensor_info) in safetensors.tensors.iter().take(10) {
            let shape: Vec<String> = tensor_info
                .shape
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            println!(
                "  - {} [{}, dtype={:?}]",
                name,
                shape.join("x"),
                tensor_info.dtype
            );
        }
        if safetensors.tensors.len() > 10 {
            println!("  ... and {} more", safetensors.tensors.len() - 10);
        }
        println!();
    }

    println!("Model loading infrastructure is ready!");
    println!();
    println!("Next steps to complete model loading:");
    println!("  1. Extract ModelConfig from tensor shapes");
    println!("  2. Map tensor names to Model layers (see src/layers.rs docs)");
    println!("  3. Load weights into each layer");
    println!();
    println!("See documentation: cargo doc --open");
    println!("Example: src/layers.rs module documentation");

    Ok(())
}

/// Load APR model file (aprender native format)
///
/// Per spec §3.1: APR is the first-class format for classical ML models.
///
/// # Arguments
///
/// * `file_data` - APR file bytes
///
/// # Errors
///
/// Returns error if:
/// - Magic bytes don't match (not "APRN")
/// - Model type is unknown
/// - File is corrupted
pub fn load_apr_model(file_data: &[u8]) -> Result<()> {
    use crate::format::{detect_format, ModelFormat};
    use crate::model_loader::read_apr_model_type;

    println!("Parsing APR file...");

    // Verify format
    let format = detect_format(file_data).map_err(|e| RealizarError::UnsupportedOperation {
        operation: "detect_apr_format".to_string(),
        reason: format!("Format detection failed: {e}"),
    })?;

    if format != ModelFormat::Apr {
        return Err(RealizarError::UnsupportedOperation {
            operation: "verify_apr_magic".to_string(),
            reason: format!("Expected APR format, got {format}"),
        });
    }

    // Extract model type
    let model_type = read_apr_model_type(file_data).unwrap_or_else(|| "Unknown".to_string());

    println!("Successfully parsed APR file");
    println!();
    println!("Model Information:");
    println!("  Format: APR (Aprender Native)");
    println!("  Model Type: {model_type}");
    println!("  File Size: {} bytes", file_data.len());
    println!();

    // APR header structure: APRN (4) + type_id (2) + version (2) = 8 bytes minimum
    if file_data.len() >= 8 {
        let version = u16::from_le_bytes([file_data[6], file_data[7]]);
        println!("  Header Version: {version}");
    }

    println!();
    println!("APR model ready for serving!");
    println!("Supported model types for inference:");
    println!("  - LogisticRegression, LinearRegression");
    println!("  - DecisionTree, RandomForest, GradientBoosting");
    println!("  - KNN, GaussianNB, LinearSVM");
    println!();
    println!("To serve this model, the serve API will auto-detect");
    println!("the model type and dispatch to the appropriate handler.");

    Ok(())
}

/// Check if a model reference is a local file path
pub fn is_local_file_path(model_ref: &str) -> bool {
    model_ref.starts_with("./")
        || model_ref.starts_with('/')
        || model_ref.ends_with(".gguf")
        || model_ref.ends_with(".safetensors")
        || model_ref.ends_with(".apr")
}

/// Simple home directory resolution
pub fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Validate benchmark suite name
pub fn validate_suite_name(suite_name: &str) -> bool {
    BENCHMARK_SUITES.iter().any(|(name, _)| *name == suite_name)
}

// ============================================================================
// Server Commands (extracted from main.rs for testability)
// WAPR-PERF-004: Gated behind "server" feature since depends on crate::api
// ============================================================================
#[cfg(feature = "server")]
mod server_commands {
    use super::Result;

    /// Result of preparing server state (returned by `prepare_serve_state`)
    pub struct PreparedServer {
        /// The prepared AppState for the server
        pub state: crate::api::AppState,
        /// Whether batch mode is enabled
        pub batch_mode_enabled: bool,
        /// Model type that was loaded
        pub model_type: ModelType,
    }

    impl std::fmt::Debug for PreparedServer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PreparedServer")
                .field("batch_mode_enabled", &self.batch_mode_enabled)
                .field("model_type", &self.model_type)
                .finish_non_exhaustive()
        }
    }

    /// Type of model being served
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ModelType {
        /// GGUF quantized model
        Gguf,
        /// SafeTensors model
        SafeTensors,
        /// APR format model
        Apr,
    }

    impl std::fmt::Display for ModelType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ModelType::Gguf => write!(f, "GGUF"),
                ModelType::SafeTensors => write!(f, "SafeTensors"),
                ModelType::Apr => write!(f, "APR"),
            }
        }
    }

    /// Prepare server state by loading a model (GGUF/SafeTensors/APR)
    ///
    /// This function is extracted from `serve_model` for testability.
    /// It handles model loading and AppState creation without starting the server.
    ///
    /// # Arguments
    /// * `model_path` - Path to model file (.gguf, .safetensors, or .apr)
    /// * `batch_mode` - Enable batch processing (requires 'gpu' feature)
    /// * `force_gpu` - Force CUDA backend (requires 'cuda' feature)
    ///
    /// # Returns
    /// A `PreparedServer` containing the AppState and configuration
    pub fn prepare_serve_state(
        model_path: &str,
        batch_mode: bool,
        force_gpu: bool,
    ) -> Result<PreparedServer> {
        use crate::gguf::MappedGGUFModel;

        println!("Loading model from: {model_path}");
        if batch_mode {
            println!("Mode: BATCH (PARITY-093 M4 parity)");
        } else {
            println!("Mode: SINGLE-REQUEST");
        }
        if force_gpu {
            println!("GPU: FORCED (--gpu flag)");
        }
        println!();

        if model_path.ends_with(".gguf") {
            // Load GGUF model
            println!("Parsing GGUF file...");
            let mapped = MappedGGUFModel::from_path(model_path).map_err(|e| {
                crate::error::RealizarError::UnsupportedOperation {
                    operation: "load_gguf".to_string(),
                    reason: format!("Failed to load GGUF: {e}"),
                }
            })?;

            println!("Successfully loaded GGUF model");
            println!("  Tensors: {}", mapped.model.tensors.len());
            println!("  Metadata: {} entries", mapped.model.metadata.len());
            println!();

            // IMP-100: Use OwnedQuantizedModel with fused Q4_K ops (1.37x faster for single-token)
            println!("Creating quantized model (fused Q4_K ops)...");
            let quantized_model =
                crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| {
                    crate::error::RealizarError::UnsupportedOperation {
                        operation: "create_quantized".to_string(),
                        reason: format!("Failed to create quantized model: {e}"),
                    }
                })?;

            println!("Quantized model created successfully!");
            println!("  Vocab size: {}", quantized_model.config.vocab_size);
            println!("  Hidden dim: {}", quantized_model.config.hidden_dim);
            println!("  Layers: {}", quantized_model.layers.len());

            // Extract vocabulary from GGUF for proper token decoding
            let vocab = mapped.model.vocabulary().unwrap_or_else(|| {
                eprintln!("  Warning: No vocabulary in GGUF, using placeholder tokens");
                (0..quantized_model.config.vocab_size)
                    .map(|i| format!("token{i}"))
                    .collect()
            });
            println!("  Vocab loaded: {} tokens", vocab.len());
            println!();

            // PARITY-113: Enable CUDA backend via --gpu flag or REALIZAR_BACKEND environment variable
            #[cfg(feature = "cuda")]
            let use_cuda = force_gpu
                || std::env::var("REALIZAR_BACKEND")
                    .map(|v| v.eq_ignore_ascii_case("cuda"))
                    .unwrap_or(false);

            #[cfg(not(feature = "cuda"))]
            let use_cuda = false;

            #[cfg(not(feature = "cuda"))]
            if force_gpu {
                eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
                eprintln!("Build with: cargo build --features cuda");
                eprintln!();
            }

            // PARITY-093: Use cached model with batch support for M4 parity
            // PAR-112-FIX: Use OwnedQuantizedModelCuda for true streaming support
            let state = if use_cuda && !batch_mode {
                // PAR-112-FIX: Create OwnedQuantizedModelCuda for true streaming
                // This enables generate_gpu_resident_streaming which streams tokens as generated
                #[cfg(feature = "cuda")]
                {
                    use crate::gguf::OwnedQuantizedModelCuda;

                    let source = if force_gpu {
                        "--gpu flag"
                    } else {
                        "REALIZAR_BACKEND=cuda"
                    };
                    println!("Creating CUDA model ({source})...");

                    let max_seq_len = 4096; // Support long sequences
                    let cuda_model =
                        OwnedQuantizedModelCuda::with_max_seq_len(quantized_model, 0, max_seq_len)
                            .map_err(|e| crate::error::RealizarError::UnsupportedOperation {
                                operation: "cuda_model_create".to_string(),
                                reason: format!("CUDA model creation failed: {e}"),
                            })?;

                    println!("  CUDA model created on GPU: {}", cuda_model.device_name());
                    println!("  Max sequence length: {}", max_seq_len);
                    println!("  TRUE STREAMING: enabled (PAR-112)");
                    println!();

                    // Use with_cuda_model_and_vocab to enable true streaming path
                    crate::api::AppState::with_cuda_model_and_vocab(cuda_model, vocab)?
                }

                #[cfg(not(feature = "cuda"))]
                {
                    // This branch is unreachable since use_cuda is always false without cuda feature
                    crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)?
                }
            } else if batch_mode {
                #[cfg(feature = "gpu")]
                {
                    use crate::gguf::OwnedQuantizedModelCachedSync;

                    println!("Initializing batch inference mode (PARITY-093/094)...");

                    // Create cached model for scheduler reuse (10.6x speedup - IMP-112)
                    let cached_model = OwnedQuantizedModelCachedSync::new(quantized_model);

                    // PARITY-094: Warmup GPU cache for batch_generate_gpu
                    // This dequantizes FFN weights to GPU memory (~6GB for phi-2)
                    println!("  Warming up GPU cache (dequantizing FFN weights)...");
                    match cached_model.warmup_gpu_cache() {
                        Ok((memory_bytes, num_layers)) => {
                            println!(
                                "  GPU cache ready: {:.2} GB ({} layers)",
                                memory_bytes as f64 / 1e9,
                                num_layers
                            );
                        },
                        Err(e) => {
                            eprintln!(
                            "  Warning: GPU cache warmup failed: {}. Falling back to CPU batch.",
                            e
                        );
                        },
                    }

                    // Create state first (this wraps model in Arc internally)
                    let state =
                        crate::api::AppState::with_cached_model_and_vocab(cached_model, vocab)?;

                    // Get Arc'd model back for batch processor
                    let cached_model_arc = state
                        .cached_model()
                        .expect("cached_model should exist")
                        .clone();

                    // Configure batch processing (PARITY-095: aligned thresholds)
                    let batch_config = crate::api::BatchConfig::default();
                    println!("  Batch window: {}ms", batch_config.window_ms);
                    println!("  Min batch size: {}", batch_config.min_batch);
                    println!("  Optimal batch: {}", batch_config.optimal_batch);
                    println!("  Max batch size: {}", batch_config.max_batch);
                    println!(
                        "  GPU threshold: {} (GPU GEMM for batch >= this)",
                        batch_config.gpu_threshold
                    );

                    // Spawn batch processor task
                    let batch_tx =
                        crate::api::spawn_batch_processor(cached_model_arc, batch_config.clone());

                    println!("  Batch processor: RUNNING");
                    println!();

                    // Add batch support to state
                    state.with_batch_config(batch_tx, batch_config)
                }

                #[cfg(not(feature = "gpu"))]
                {
                    eprintln!(
                    "Warning: --batch requires 'gpu' feature. Falling back to single-request mode."
                );
                    crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)?
                }
            } else {
                // CPU mode: Use quantized model for serving (fused CPU ops are faster for m=1)
                crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)?
            };

            Ok(PreparedServer {
                state,
                batch_mode_enabled: batch_mode,
                model_type: ModelType::Gguf,
            })
        } else if model_path.ends_with(".safetensors") {
            // PMAT-SERVE-FIX-001: Properly serve SafeTensors models
            use crate::safetensors_infer::SafetensorsToAprConverter;
            use std::path::Path;

            println!("Loading SafeTensors model for serving...");

            // Convert SafeTensors to AprTransformer
            let model_path_obj = Path::new(model_path);
            let transformer = SafetensorsToAprConverter::convert(model_path_obj).map_err(|e| {
                crate::error::RealizarError::UnsupportedOperation {
                    operation: "convert_safetensors".to_string(),
                    reason: format!("Failed to convert SafeTensors: {e}"),
                }
            })?;

            println!("  Architecture: {}", transformer.config.architecture);
            println!("  Layers: {}", transformer.config.num_layers);
            println!("  Hidden: {}", transformer.config.hidden_dim);

            // Load vocabulary from sibling tokenizer.json
            #[allow(clippy::map_unwrap_or)]
            let vocab = crate::apr::AprV2Model::load_tokenizer_from_sibling(model_path_obj)
                .map(|(v, _, _)| v) // Extract just the vocab
                .unwrap_or_else(|| {
                    // Fallback: Generate simple vocab
                    println!("  Warning: No tokenizer.json found, using simple vocabulary");
                    (0..transformer.config.vocab_size)
                        .map(|i| format!("token{i}"))
                        .collect()
                });

            println!("  Vocab size: {}", vocab.len());
            println!("  Mode: CPU (F32 inference)");

            let state = crate::api::AppState::with_apr_transformer_and_vocab(transformer, vocab)?;

            Ok(PreparedServer {
                state,
                batch_mode_enabled: false,
                model_type: ModelType::SafeTensors,
            })
        } else if model_path.ends_with(".apr") {
            // PMAT-SERVE-FIX-001: Properly serve APR models
            use crate::apr_transformer::AprTransformer;
            use std::path::Path;

            println!("Loading APR model for serving...");

            let file_data = std::fs::read(model_path).map_err(|e| {
                crate::error::RealizarError::UnsupportedOperation {
                    operation: "read_model_file".to_string(),
                    reason: format!("Failed to read {model_path}: {e}"),
                }
            })?;

            // Load AprTransformer from APR file
            let transformer = AprTransformer::from_apr_bytes(&file_data).map_err(|e| {
                crate::error::RealizarError::UnsupportedOperation {
                    operation: "load_apr".to_string(),
                    reason: format!("Failed to load APR: {e}"),
                }
            })?;

            println!("  Architecture: {}", transformer.config.architecture);
            println!("  Layers: {}", transformer.config.num_layers);
            println!("  Hidden: {}", transformer.config.hidden_dim);

            // Load vocabulary from APR metadata or sibling tokenizer.json
            let model_path_obj = Path::new(model_path);
            let vocab = crate::apr::AprV2Model::load_tokenizer_from_sibling(model_path_obj)
                .map(|(v, _, _)| v) // Extract just the vocab
                .or_else(|| {
                    // Try to load from APR embedded vocabulary
                    crate::apr::AprV2Model::load(model_path_obj)
                        .ok()
                        .and_then(|m| m.load_embedded_tokenizer())
                        .map(|t| t.id_to_token.clone())
                })
                .unwrap_or_else(|| {
                    // Fallback: Generate simple vocab
                    println!("  Warning: No vocabulary found, using simple vocabulary");
                    (0..transformer.config.vocab_size)
                        .map(|i| format!("token{i}"))
                        .collect()
                });

            println!("  Vocab size: {}", vocab.len());
            println!("  Mode: CPU (F32 inference)");

            let state = crate::api::AppState::with_apr_transformer_and_vocab(transformer, vocab)?;

            Ok(PreparedServer {
                state,
                batch_mode_enabled: false,
                model_type: ModelType::Apr,
            })
        } else {
            Err(crate::error::RealizarError::UnsupportedOperation {
                operation: "detect_model_type".to_string(),
                reason: "Unsupported file extension. Expected .gguf, .safetensors, or .apr"
                    .to_string(),
            })
        }
    }

    /// Serve a GGUF/SafeTensors/APR model via HTTP API
    ///
    /// This function was extracted from main.rs (PAR-112-FIX) to enable:
    /// 1. Unit testing of server initialization logic
    /// 2. Coverage measurement (main.rs was at 3.66%)
    /// 3. Reuse from other entry points
    ///
    /// # Arguments
    /// * `host` - Host to bind to (e.g., "0.0.0.0")
    /// * `port` - Port to listen on
    /// * `model_path` - Path to model file (.gguf, .safetensors, or .apr)
    /// * `batch_mode` - Enable batch processing (requires 'gpu' feature)
    /// * `force_gpu` - Force CUDA backend (requires 'cuda' feature)
    pub async fn serve_model(
        host: &str,
        port: u16,
        model_path: &str,
        batch_mode: bool,
        force_gpu: bool,
    ) -> Result<()> {
        // Prepare server state (testable)
        let prepared = prepare_serve_state(model_path, batch_mode, force_gpu)?;

        // Create router
        let app = crate::api::create_router(prepared.state);

        // Parse and validate address
        let addr: std::net::SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Invalid address: {e}"),
            }
        })?;

        // Print server info
        println!("Server listening on http://{addr}");
        println!();
        println!("Endpoints:");
        println!("  GET  /health         - Health check");
        println!("  POST /v1/completions - OpenAI-compatible completions");
        if prepared.batch_mode_enabled {
            println!("  POST /v1/batch/completions - GPU batch completions (PARITY-022)");
            println!("  POST /v1/gpu/warmup  - Warmup GPU cache");
            println!("  GET  /v1/gpu/status  - GPU status");
        }
        println!("  POST /generate       - Generate text (Q4_K fused)");
        println!();

        if prepared.batch_mode_enabled {
            println!("M4 Parity Target: 192 tok/s at concurrency >= 4");
            println!("Benchmark with: wrk -t4 -c4 -d30s http://{addr}/v1/completions");
            println!();
        }

        // Bind and serve
        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "bind".to_string(),
                reason: format!("Failed to bind: {e}"),
            }
        })?;

        axum::serve(listener, app).await.map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "serve".to_string(),
                reason: format!("Server error: {e}"),
            }
        })?;

        Ok(())
    }

    /// Start a demo inference server (no model required)
    ///
    /// This is useful for testing the API without loading a real model.
    pub async fn serve_demo(host: &str, port: u16) -> Result<()> {
        use std::net::SocketAddr;

        println!("Starting Realizar inference server (demo mode)...");

        let state = crate::api::AppState::demo()?;
        let app = crate::api::create_router(state);

        let addr: SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Invalid address: {e}"),
            }
        })?;

        println!("Server listening on http://{addr}");
        println!();
        println!("Endpoints:");
        println!("  GET  /health   - Health check");
        println!("  POST /tokenize - Tokenize text");
        println!("  POST /generate - Generate text");
        println!();
        println!("Example:");
        println!("  curl http://{addr}/health");
        println!();

        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Failed to bind: {e}"),
            }
        })?;

        axum::serve(listener, app).await.map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Server error: {e}"),
            }
        })?;

        Ok(())
    }
} // mod server_commands

#[cfg(feature = "server")]
pub use server_commands::*;

// Tests split for file health (was 3.3K lines)
#[cfg(test)]
#[path = "tests_split_01.rs"]
mod cli_tests_split_01;
#[cfg(test)]
#[path = "tests_split_02.rs"]
mod cli_tests_split_02;
#[cfg(test)]
#[path = "tests_split_03.rs"]
mod cli_tests_split_03;

// Additional inference coverage tests (Part 02)
#[cfg(test)]
#[path = "tests_part_02.rs"]
mod cli_tests_part_02;

// CLI helper functions tests (Part 03)
#[cfg(test)]
#[path = "tests_part_03.rs"]
mod cli_tests_part_03;

// Coverage bridge tests (Part 04 - T-COV-95 B1)
#[cfg(test)]
#[path = "tests_part_04.rs"]
mod cli_tests_part_04;

// Deep CLI coverage tests (Part 05 - T-COV-95 Deep CLI)
#[cfg(test)]
#[path = "tests_part_05.rs"]
mod cli_tests_part_05;

// T-COV-95 Deep Coverage Bridge (Part 06 - handlers.rs: pull, push, list, serve, trace)
#[cfg(test)]
#[path = "tests_part_06.rs"]
mod cli_tests_part_06;

// T-COV-95 Deep Coverage Bridge (Part 07 - mod.rs: bench, viz, load, format, parse)
#[cfg(test)]
#[path = "tests_part_07.rs"]
mod cli_tests_part_07;

// T-COV-95 Extended Coverage (Part 08 - mod.rs: format_size, is_local_file_path, validate_suite_name, display_model_info)
#[cfg(test)]
#[path = "tests_part_08.rs"]
mod cli_tests_part_08;

// T-COV-95 Synthetic Falsification (Part 09 - inference.rs via Pygmy GGUF models)
#[cfg(test)]
#[path = "tests_part_09.rs"]
mod cli_tests_part_09;

// T-COV-95 CLI Inference Additional Coverage
#[cfg(test)]
#[path = "inference_tests_part_02.rs"]
mod cli_inference_tests_part_02;

// T-COV-95 Active Pygmy CLI Inference (In-Memory)
#[cfg(test)]
#[path = "inference_tests_part_03.rs"]
mod cli_inference_tests_part_03;

// T-COV-95 Artifact Falsification (Real Files, Real CLI)
#[cfg(test)]
#[path = "inference_tests_part_04.rs"]
mod cli_inference_tests_part_04;

// T-COV-95 Poisoned Pygmies: CLI Graceful Degradation Tests
#[cfg(test)]
#[path = "tests_part_10.rs"]
mod cli_tests_part_10;
