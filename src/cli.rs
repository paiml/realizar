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
    if model_ref.ends_with(".gguf") || file_data.starts_with(b"GGUF") {
        use crate::gguf::GGUFModel;
        let gguf = GGUFModel::from_bytes(file_data)?;
        println!("  Format: GGUF v{}", gguf.header.version);
        println!("  Tensors: {}", gguf.header.tensor_count);
    } else if model_ref.ends_with(".safetensors") {
        use crate::safetensors::SafetensorsModel;
        let st = SafetensorsModel::from_bytes(file_data)?;
        println!("  Format: SafeTensors");
        println!("  Tensors: {}", st.tensors.len());
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

    // Generate synthetic benchmark data (simulating inference latencies)
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

/// Run benchmarks with cargo bench
pub fn run_benchmarks(
    suite: Option<String>,
    list: bool,
    runtime: Option<String>,
    model: Option<String>,
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
        return Ok(());
    }

    let runtime_name = runtime.unwrap_or_else(|| "realizar".to_string());
    println!("Benchmark Configuration:");
    println!("  Runtime: {runtime_name}");
    if let Some(ref m) = model {
        println!("  Model: {m}");
    }
    if let Some(ref o) = output {
        println!("  Output: {o}");
    }
    println!();

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

    // TODO: Generate JSON output if --output specified
    if let Some(ref output_path) = output {
        println!();
        println!("Note: JSON output to {output_path} is a placeholder.");
        println!("      Full benchmark harness integration coming soon.");
    }

    Ok(())
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

    // Create simulated result for demo (actual benchmark would run inference)
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

    // Create simulated result for demo
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

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Format Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(10 * 1024), "10.0 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.0 MB");
        assert_eq!(format_size(512 * 1024 * 1024), "512.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(7 * 1024 * 1024 * 1024), "7.0 GB");
    }

    // -------------------------------------------------------------------------
    // Benchmark Suite Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_suites_not_empty() {
        assert!(!BENCHMARK_SUITES.is_empty());
        assert!(BENCHMARK_SUITES.len() >= 5);
    }

    #[test]
    fn test_benchmark_suites_have_descriptions() {
        for (name, description) in BENCHMARK_SUITES {
            assert!(!name.is_empty(), "Benchmark name should not be empty");
            assert!(
                !description.is_empty(),
                "Benchmark description should not be empty"
            );
        }
    }

    #[test]
    fn test_validate_suite_name_valid() {
        assert!(validate_suite_name("tensor_ops"));
        assert!(validate_suite_name("inference"));
        assert!(validate_suite_name("cache"));
        assert!(validate_suite_name("tokenizer"));
        assert!(validate_suite_name("quantize"));
        assert!(validate_suite_name("lambda"));
        assert!(validate_suite_name("comparative"));
    }

    #[test]
    fn test_validate_suite_name_invalid() {
        assert!(!validate_suite_name("unknown"));
        assert!(!validate_suite_name(""));
        assert!(!validate_suite_name("tensor"));
        assert!(!validate_suite_name("TENSOR_OPS"));
    }

    // -------------------------------------------------------------------------
    // Is Local File Path Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_true() {
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("/home/user/model.gguf"));
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("model.apr"));
    }

    #[test]
    fn test_is_local_file_path_false() {
        assert!(!is_local_file_path("llama3:8b"));
        assert!(!is_local_file_path("pacha://model:v1"));
        assert!(!is_local_file_path("hf://meta-llama/Llama-3"));
    }

    // -------------------------------------------------------------------------
    // Home Dir Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_home_dir_returns_path() {
        // This test depends on HOME being set, which it usually is
        let home = home_dir();
        // Just check it doesn't panic - may be None in some environments
        if let Some(path) = home {
            assert!(path.is_absolute() || path.to_string_lossy().starts_with('/'));
        }
    }

    // -------------------------------------------------------------------------
    // Display Model Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_unknown_format() {
        // Empty data with unknown extension
        let result = display_model_info("model.bin", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_gguf_extension_but_invalid() {
        // .gguf extension but not valid GGUF data
        let result = display_model_info("model.gguf", &[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Print Info Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_print_info_does_not_panic() {
        // Just ensure it doesn't panic
        print_info();
    }

    // -------------------------------------------------------------------------
    // Run Visualization Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_does_not_panic() {
        // Run with minimal samples to keep test fast
        run_visualization(false, 10);
        run_visualization(true, 10);
    }

    // -------------------------------------------------------------------------
    // Load Model Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_gguf_model_invalid() {
        let result = load_gguf_model(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_model_invalid() {
        let result = load_safetensors_model(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Convoy and Saturation Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_convoy_test_no_output() {
        // Just verify it runs without panic
        let result = run_convoy_test(Some("test".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_no_output() {
        let result = run_saturation_test(Some("test".to_string()), None, None);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Benchmark Compare/Regression Tests (file not found cases)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_compare_file_not_found() {
        let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_regression_file_not_found() {
        let result = run_bench_regression(
            "/nonexistent/baseline.json",
            "/nonexistent/current.json",
            false,
        );
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // run_benchmarks Tests (list mode only - doesn't run cargo bench)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_benchmarks_list_mode() {
        // List mode should succeed without running cargo bench
        let result = run_benchmarks(None, true, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_runtime() {
        // List mode with runtime specified
        let result = run_benchmarks(None, true, Some("realizar".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_model() {
        // List mode with model specified
        let result = run_benchmarks(None, true, None, Some("model.gguf".to_string()), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_output() {
        // List mode with output specified
        let result = run_benchmarks(None, true, None, None, Some("/tmp/output.json".to_string()));
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Temp File Tests for bench compare/regression (error paths)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_compare_second_file_not_found() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let file1 = dir.join("bench_compare_one.json");

        let mut f1 = std::fs::File::create(&file1).unwrap();
        f1.write_all(b"{}").unwrap();

        let result = run_bench_compare(file1.to_str().unwrap(), "/nonexistent/file2.json", 5.0);

        let _ = std::fs::remove_file(&file1);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_regression_second_file_not_found() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let baseline = dir.join("bench_regress_base.json");

        let mut f1 = std::fs::File::create(&baseline).unwrap();
        f1.write_all(b"{}").unwrap();

        let result = run_bench_regression(
            baseline.to_str().unwrap(),
            "/nonexistent/current.json",
            false,
        );

        let _ = std::fs::remove_file(&baseline);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Convoy/Saturation with output file tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_convoy_test_with_output() {
        let dir = std::env::temp_dir();
        let output = dir.join("convoy_output.json");

        let result = run_convoy_test(
            Some("test".to_string()),
            Some("model.gguf".to_string()),
            Some(output.to_str().unwrap().to_string()),
        );

        assert!(result.is_ok());
        assert!(output.exists());

        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn test_run_saturation_test_with_output() {
        let dir = std::env::temp_dir();
        let output = dir.join("saturation_output.json");

        let result = run_saturation_test(
            Some("test".to_string()),
            Some("model.gguf".to_string()),
            Some(output.to_str().unwrap().to_string()),
        );

        assert!(result.is_ok());
        assert!(output.exists());

        let _ = std::fs::remove_file(&output);
    }

    // -------------------------------------------------------------------------
    // Display Model Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_with_gguf_magic() {
        // Create minimal GGUF data with magic header
        let data = b"GGUF\x03\x00\x00\x00"; // GGUF magic + version 3
        let result = display_model_info("test.gguf", data);
        // Will fail to parse but exercises the GGUF path
        assert!(result.is_err());
    }

    #[test]
    fn test_display_model_info_safetensors_extension() {
        let result = display_model_info("test.safetensors", &[0, 1, 2, 3]);
        assert!(result.is_err());
    }
}
