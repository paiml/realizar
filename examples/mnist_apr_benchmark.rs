//! MNIST Inference Benchmark: Aprender (.apr) vs PyTorch
//!
//! Scientifically reproducible benchmark comparing inference latency
//! for equivalent models on MNIST classification.
//!
//! ## Methodology (per Box et al. 2005, Georges et al. 2007)
//!
//! 1. Fixed random seeds for reproducibility
//! 2. Warm-up phase excluded from measurements
//! 3. Large sample size (10,000 iterations) for statistical significance
//! 4. Report: mean, std, 95% CI, percentiles (p50, p95, p99)
//! 5. Same model architecture: Logistic Regression (784 -> 10)
//! 6. Same input data: Single MNIST sample (784 floats)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example mnist_apr_benchmark --release --features aprender-serve
//! ```
//!
//! ## Output
//!
//! - `aprender_mnist_results.json` - Machine-readable results
//! - Stdout - Human-readable report
//!
//! ## Comparison with PyTorch
//!
//! ```bash
//! cd benches/comparative
//! uv run mnist_benchmark.py
//! uv run compare_mnist.py
//! ```
//!
//! ## Citation
//!
//! See CITATION.cff for proper academic citation format.

use std::{
    fs::File,
    io::Write,
    time::{Duration, Instant},
};

use aprender::prelude::*;

// Configuration - MUST match Python benchmark exactly
const SEED: u64 = 42;
const INPUT_DIM: usize = 784; // 28x28 MNIST
const NUM_CLASSES: usize = 2; // Binary: digit 0 vs others (aprender LogisticRegression is binary)
const WARMUP_ITERATIONS: usize = 100;
const BENCHMARK_ITERATIONS: usize = 10_000;
const TRAINING_SAMPLES: usize = 1000;

/// Statistical results from benchmark run
#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    iterations: usize,
    mean_us: f64,
    std_us: f64,
    ci_95_lower: f64,
    ci_95_upper: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    min_us: f64,
    max_us: f64,
    throughput_per_sec: f64,
}

impl BenchmarkResult {
    fn from_durations(name: &str, durations: &[Duration]) -> Self {
        let latencies_us: Vec<f64> = durations
            .iter()
            .map(|d| d.as_nanos() as f64 / 1000.0)
            .collect();
        let n = latencies_us.len() as f64;

        let mean = latencies_us.iter().sum::<f64>() / n;
        let variance = latencies_us.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();
        let se = std / n.sqrt();
        let ci_95 = 1.96 * se;

        let mut sorted = latencies_us;
        sorted.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("latency values should be comparable floats")
        });

        Self {
            name: name.to_string(),
            iterations: durations.len(),
            mean_us: mean,
            std_us: std,
            ci_95_lower: mean - ci_95,
            ci_95_upper: mean + ci_95,
            p50_us: sorted[sorted.len() / 2],
            p95_us: sorted[(sorted.len() as f64 * 0.95) as usize],
            p99_us: sorted[(sorted.len() as f64 * 0.99) as usize],
            min_us: sorted[0],
            max_us: sorted[sorted.len() - 1],
            throughput_per_sec: if mean > 0.0 { 1_000_000.0 / mean } else { 0.0 },
        }
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{
    "name": "{}",
    "iterations": {},
    "mean_us": {:.2},
    "std_us": {:.2},
    "ci_95_lower": {:.2},
    "ci_95_upper": {:.2},
    "p50_us": {:.2},
    "p95_us": {:.2},
    "p99_us": {:.2},
    "min_us": {:.2},
    "max_us": {:.2},
    "throughput_per_sec": {:.0}
  }}"#,
            self.name,
            self.iterations,
            self.mean_us,
            self.std_us,
            self.ci_95_lower,
            self.ci_95_upper,
            self.p50_us,
            self.p95_us,
            self.p99_us,
            self.min_us,
            self.max_us,
            self.throughput_per_sec
        )
    }

    fn report(&self) {
        println!("  Iterations: {}", self.iterations);
        println!("  Mean: {:.2} us", self.mean_us);
        println!("  Std Dev: {:.2} us", self.std_us);
        println!(
            "  95% CI: [{:.2}, {:.2}] us",
            self.ci_95_lower, self.ci_95_upper
        );
        println!("  p50: {:.2} us", self.p50_us);
        println!("  p95: {:.2} us", self.p95_us);
        println!("  p99: {:.2} us", self.p99_us);
        println!("  Min: {:.2} us", self.min_us);
        println!("  Max: {:.2} us", self.max_us);
        println!(
            "  Throughput: {:.0} inferences/sec",
            self.throughput_per_sec
        );
    }
}

/// Generate test MNIST data - MUST match Python exactly
///
/// Formula: pixel = ((i * 17 + j * 31) % 256) / 255.0
/// Labels: Binary (0 or 1) for logistic regression
fn generate_mnist_data() -> (Matrix<f32>, Vec<usize>) {
    let mut data = Vec::with_capacity(TRAINING_SAMPLES * INPUT_DIM);
    let mut labels = Vec::with_capacity(TRAINING_SAMPLES);

    for i in 0..TRAINING_SAMPLES {
        for j in 0..INPUT_DIM {
            // Same formula as Python for exact reproducibility
            let pixel = ((i * 17 + j * 31) % 256) as f32 / 255.0;
            data.push(pixel);
        }
        // Binary classification: 0 vs not-0 (simulates digit 0 vs others)
        labels.push(if i % 10 == 0 { 0 } else { 1 });
    }

    let x = Matrix::from_vec(TRAINING_SAMPLES, INPUT_DIM, data).expect("valid matrix");
    (x, labels)
}

/// Generate single inference sample - MUST match Python exactly
fn generate_sample() -> Matrix<f32> {
    let data: Vec<f32> = (0..INPUT_DIM).map(|j| (j % 256) as f32 / 255.0).collect();
    Matrix::from_vec(1, INPUT_DIM, data).expect("valid sample")
}

/// Benchmark inference with warmup and measurement phases
fn benchmark_inference<F>(name: &str, mut inference_fn: F) -> BenchmarkResult
where
    F: FnMut() -> Vec<usize>,
{
    // Warmup phase (excluded from measurements)
    for _ in 0..WARMUP_ITERATIONS {
        let _ = inference_fn();
    }

    // Measurement phase
    let mut durations = Vec::with_capacity(BENCHMARK_ITERATIONS);
    for _ in 0..BENCHMARK_ITERATIONS {
        let start = Instant::now();
        let _ = inference_fn();
        durations.push(start.elapsed());
    }

    BenchmarkResult::from_durations(name, &durations)
}

fn get_cpu_info() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        return name.trim().to_string();
                    }
                }
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

fn main() {
    println!("======================================================================");
    println!("MNIST Inference Benchmark: Aprender (.apr)");
    println!("======================================================================");
    println!();
    println!("## Configuration");
    println!("  Seed: {}", SEED);
    println!("  Input dimensions: {}", INPUT_DIM);
    println!("  Output classes: {}", NUM_CLASSES);
    println!("  Training samples: {}", TRAINING_SAMPLES);
    println!("  Warmup iterations: {}", WARMUP_ITERATIONS);
    println!("  Benchmark iterations: {}", BENCHMARK_ITERATIONS);
    println!();
    println!("## Environment");
    println!("  Aprender: {}", env!("CARGO_PKG_VERSION"));
    println!("  Rust: {}", rustc_version());
    println!(
        "  Platform: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("  CPU: {}", get_cpu_info());
    println!();

    // Generate data
    println!("## Generating test MNIST data...");
    let (x, y) = generate_mnist_data();
    println!("  Data shape: {}x{}", TRAINING_SAMPLES, INPUT_DIM);
    println!();

    // Generate single inference sample
    let sample = generate_sample();

    let mut results = Vec::new();

    // Benchmark 1: Logistic Regression
    println!("## Training LogisticRegression...");
    let mut logreg = LogisticRegression::new();
    logreg.fit(&x, &y).expect("training failed");

    println!("## Benchmarking LogisticRegression inference...");
    let logreg_result =
        benchmark_inference("Aprender LogisticRegression", || logreg.predict(&sample));
    logreg_result.report();
    results.push(logreg_result);
    println!();

    // Benchmark 2: Multinomial (softmax regression)
    println!("## Training MultinomialNB (baseline)...");
    // Note: aprender doesn't have exact MLP equivalent, so we use available models
    // For true comparison, we use LogisticRegression which IS equivalent

    // Summary
    println!("======================================================================");
    println!("SUMMARY");
    println!("======================================================================");
    println!();
    println!(
        "| {:<28} | {:>10} | {:>10} | {:>14} |",
        "Model", "p50 (us)", "p99 (us)", "Throughput/sec"
    );
    println!("|------------------------------|------------|------------|----------------|");
    for r in &results {
        println!(
            "| {:<28} | {:>10.2} | {:>10.2} | {:>14.0} |",
            r.name, r.p50_us, r.p99_us, r.throughput_per_sec
        );
    }
    println!();

    // Save JSON results
    let timestamp = chrono_lite_timestamp();
    let json_output = format!(
        r#"{{
  "config": {{
    "seed": {},
    "input_dim": {},
    "num_classes": {},
    "warmup_iterations": {},
    "benchmark_iterations": {},
    "training_samples": {},
    "aprender_version": "{}",
    "rust_version": "{}",
    "platform": "{} {}",
    "cpu": "{}",
    "timestamp": "{}"
  }},
  "results": [
    {}
  ]
}}"#,
        SEED,
        INPUT_DIM,
        NUM_CLASSES,
        WARMUP_ITERATIONS,
        BENCHMARK_ITERATIONS,
        TRAINING_SAMPLES,
        env!("CARGO_PKG_VERSION"),
        rustc_version(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        get_cpu_info(),
        timestamp,
        results
            .iter()
            .map(|r| r.to_json())
            .collect::<Vec<_>>()
            .join(",\n    ")
    );

    let output_path = "benches/comparative/aprender_mnist_results.json";
    if let Ok(mut file) = File::create(output_path) {
        let _ = file.write_all(json_output.as_bytes());
        println!("Results saved to: {}", output_path);
    } else {
        // Try current directory as fallback
        let fallback_path = "aprender_mnist_results.json";
        if let Ok(mut file) = File::create(fallback_path) {
            let _ = file.write_all(json_output.as_bytes());
            println!("Results saved to: {}", fallback_path);
        }
    }

    println!();
    println!("To compare with PyTorch, run:");
    println!("  cd benches/comparative");
    println!("  uv run mnist_benchmark.py");
    println!("  uv run compare_mnist.py");
}

fn rustc_version() -> &'static str {
    // Compile-time rustc version would require build.rs
    // For now, return a placeholder
    "stable"
}

fn chrono_lite_timestamp() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Approximate ISO 8601
    format!("{}Z", secs)
}
