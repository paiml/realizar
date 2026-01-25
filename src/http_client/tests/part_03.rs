use crate::http_client::*;
// =========================================================================

/// IMP-158a: Benchmark result schema
#[derive(Debug, Clone)]
pub struct BenchmarkResultSchema {
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
}

impl BenchmarkResultSchema {
    pub fn standard() -> Self {
        Self {
            required_fields: vec![
                "throughput_tps".to_string(),
                "iterations".to_string(),
                "cv_achieved".to_string(),
                "timestamp".to_string(),
            ],
            optional_fields: vec![
                "latency_p50_ms".to_string(),
                "latency_p95_ms".to_string(),
                "latency_p99_ms".to_string(),
                "model_path".to_string(),
                "environment".to_string(),
            ],
        }
    }

    pub fn validate(&self, json: &str) -> std::result::Result<(), Vec<String>> {
        let parsed: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => return Err(vec![format!("Invalid JSON: {}", e)]),
        };

        let mut missing = Vec::new();
        for field in &self.required_fields {
            if !Self::has_field(&parsed, field) {
                missing.push(format!("Missing required field: {}", field));
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }

    fn has_field(value: &serde_json::Value, field: &str) -> bool {
        // Check top-level and nested in "results"
        if value.get(field).is_some() {
            return true;
        }
        if let Some(results) = value.get("results") {
            if results.get(field).is_some() {
                return true;
            }
        }
        false
    }
}

/// IMP-158a: Test schema validation
#[test]
fn test_imp_158a_schema_validation() {
    let schema = BenchmarkResultSchema::standard();

    // Valid result
    let valid_json = r#"{
        "throughput_tps": 150.0,
        "iterations": 25,
        "cv_achieved": 0.08,
        "timestamp": "2025-12-13T10:00:00Z"
    }"#;

    assert!(
        schema.validate(valid_json).is_ok(),
        "IMP-158a: Valid JSON should pass validation"
    );

    // Invalid result (missing throughput)
    let invalid_json = r#"{
        "iterations": 25,
        "cv_achieved": 0.08,
        "timestamp": "2025-12-13T10:00:00Z"
    }"#;

    let errors = schema.validate(invalid_json).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("throughput_tps")),
        "IMP-158a: Should report missing throughput_tps"
    );

    println!("\nIMP-158a: Schema Validation:");
    println!("  Required fields: {:?}", schema.required_fields);
    println!("  Valid JSON: PASS");
    println!("  Invalid JSON errors: {:?}", errors);
}

/// IMP-158b: Result range validation
#[derive(Debug, Clone)]
pub struct RangeValidator {
    pub field: String,
    pub min: f64,
    pub max: f64,
}

impl RangeValidator {
    pub fn new(field: &str, min: f64, max: f64) -> Self {
        Self {
            field: field.to_string(),
            min,
            max,
        }
    }

    pub fn validate(&self, value: f64) -> std::result::Result<(), String> {
        if value < self.min {
            Err(format!(
                "{} = {} is below minimum {}",
                self.field, value, self.min
            ))
        } else if value > self.max {
            Err(format!(
                "{} = {} exceeds maximum {}",
                self.field, value, self.max
            ))
        } else {
            Ok(())
        }
    }
}

/// IMP-158b: Test range validation
#[test]
fn test_imp_158b_range_validation() {
    // Throughput: 0-10000 tok/s reasonable
    let throughput_validator = RangeValidator::new("throughput_tps", 0.0, 10000.0);
    assert!(
        throughput_validator.validate(150.0).is_ok(),
        "IMP-158b: 150 in range"
    );
    assert!(
        throughput_validator.validate(-1.0).is_err(),
        "IMP-158b: Negative should fail"
    );
    assert!(
        throughput_validator.validate(50000.0).is_err(),
        "IMP-158b: 50000 too high"
    );

    // CV: 0-1.0
    let cv_validator = RangeValidator::new("cv_achieved", 0.0, 1.0);
    assert!(
        cv_validator.validate(0.08).is_ok(),
        "IMP-158b: 0.08 CV in range"
    );
    assert!(
        cv_validator.validate(1.5).is_err(),
        "IMP-158b: 1.5 CV too high"
    );

    println!("\nIMP-158b: Range Validation:");
    println!(
        "  throughput_tps: {:.0}-{:.0}",
        throughput_validator.min, throughput_validator.max
    );
    println!(
        "  cv_achieved: {:.2}-{:.2}",
        cv_validator.min, cv_validator.max
    );
}

/// IMP-158c: Complete result validation
pub struct CompleteValidator {
    pub schema: BenchmarkResultSchema,
    pub range_validators: Vec<RangeValidator>,
}

impl CompleteValidator {
    pub fn standard() -> Self {
        Self {
            schema: BenchmarkResultSchema::standard(),
            range_validators: vec![
                RangeValidator::new("throughput_tps", 0.0, 10000.0),
                RangeValidator::new("cv_achieved", 0.0, 1.0),
                RangeValidator::new("iterations", 1.0, 1000.0),
            ],
        }
    }

    pub fn validate_json(&self, json: &str) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Schema validation
        if let Err(schema_errors) = self.schema.validate(json) {
            errors.extend(schema_errors);
        }

        // Parse for range validation
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json) {
            for validator in &self.range_validators {
                if let Some(value) = Self::get_field_value(&parsed, &validator.field) {
                    if let Err(e) = validator.validate(value) {
                        errors.push(e);
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn get_field_value(value: &serde_json::Value, field: &str) -> Option<f64> {
        value
            .get(field)
            .and_then(serde_json::Value::as_f64)
            .or_else(|| {
                value
                    .get("results")
                    .and_then(|r| r.get(field))
                    .and_then(serde_json::Value::as_f64)
            })
    }
}

/// IMP-158c: Test complete validation
#[test]
fn test_imp_158c_complete_validation() {
    let validator = CompleteValidator::standard();

    let valid = r#"{
        "throughput_tps": 150.0,
        "iterations": 25,
        "cv_achieved": 0.08,
        "timestamp": "2025-12-13T10:00:00Z"
    }"#;

    assert!(
        validator.validate_json(valid).is_ok(),
        "IMP-158c: Valid result should pass"
    );

    let invalid = r#"{
        "throughput_tps": -50.0,
        "iterations": 25,
        "cv_achieved": 2.0,
        "timestamp": "2025-12-13T10:00:00Z"
    }"#;

    let errors = validator.validate_json(invalid).unwrap_err();
    assert!(errors.len() >= 2, "IMP-158c: Should have multiple errors");

    println!("\nIMP-158c: Complete Validation:");
    println!("  Valid JSON: PASS");
    println!("  Invalid JSON errors: {:?}", errors);
}

/// IMP-158d: Comparison result validation
#[derive(Debug, Clone)]
pub struct ComparisonResultValidator;

impl ComparisonResultValidator {
    pub fn validate_comparison(
        realizar_tps: f64,
        reference_tps: f64,
    ) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if realizar_tps <= 0.0 {
            errors.push("Realizar throughput must be positive".to_string());
        }
        if reference_tps <= 0.0 {
            errors.push("Reference throughput must be positive".to_string());
        }

        // Sanity check: throughput shouldn't differ by more than 1000x
        if realizar_tps > 0.0 && reference_tps > 0.0 {
            let ratio = (realizar_tps / reference_tps).max(reference_tps / realizar_tps);
            if ratio > 1000.0 {
                errors.push(format!("Throughput ratio {}x seems unreasonable", ratio));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// IMP-158d: Test comparison validation
#[test]
fn test_imp_158d_comparison_validation() {
    assert!(
        ComparisonResultValidator::validate_comparison(150.0, 256.0).is_ok(),
        "IMP-158d: Valid comparison should pass"
    );

    let errors = ComparisonResultValidator::validate_comparison(-10.0, 256.0).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("positive")),
        "IMP-158d: Should reject negative throughput"
    );

    let ratio_errors = ComparisonResultValidator::validate_comparison(1.0, 100000.0).unwrap_err();
    assert!(
        ratio_errors.iter().any(|e| e.contains("unreasonable")),
        "IMP-158d: Should flag extreme ratios"
    );

    println!("\nIMP-158d: Comparison Validation:");
    println!("  Valid (150 vs 256): PASS");
    println!("  Negative value: {:?}", errors);
    println!("  Extreme ratio: {:?}", ratio_errors);
}

// =========================================================================
// IMP-159: Throughput Variance Tracking (QA-036, EXTREME TDD)
// =========================================================================
// Per spec QA-036: Track throughput variance for statistical confidence.
// CV-based stopping criterion per Hoefler & Belli SC'15.
// Run with: cargo test test_imp_159 --lib --features bench-http

/// IMP-159a: Throughput measurement with variance tracking
#[derive(Debug, Clone)]
pub struct ThroughputWithVariance {
    /// Mean throughput in tokens/second
    pub mean_tps: f64,
    /// Standard deviation of throughput
    pub stddev_tps: f64,
    /// Coefficient of variation (CV = stddev/mean)
    pub cv: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Individual samples for analysis
    pub samples: Vec<f64>,
    /// 95% confidence interval (mean ± margin)
    pub ci_95_margin: f64,
}

impl ThroughputWithVariance {
    /// Create from a vector of throughput samples
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self {
                mean_tps: 0.0,
                stddev_tps: 0.0,
                cv: 0.0,
                sample_count: 0,
                samples: Vec::new(),
                ci_95_margin: 0.0,
            };
        }

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let stddev = variance.sqrt();
        let cv = if mean > 0.0 { stddev / mean } else { 0.0 };

        // 95% CI margin: t * (stddev / sqrt(n)), using t ≈ 1.96 for large n
        let t_value = if n >= 30 { 1.96 } else { 2.0 };
        let ci_margin = t_value * stddev / (n as f64).sqrt();

        Self {
            mean_tps: mean,
            stddev_tps: stddev,
            cv,
            sample_count: n,
            samples: samples.to_vec(),
            ci_95_margin: ci_margin,
        }
    }

    /// Check if measurement meets CV threshold for reliability
    pub fn meets_cv_threshold(&self, threshold: f64) -> bool {
        self.cv <= threshold && self.sample_count >= 5
    }

    /// Get 95% confidence interval as (lower, upper)
    pub fn confidence_interval(&self) -> (f64, f64) {
        (
            self.mean_tps - self.ci_95_margin,
            self.mean_tps + self.ci_95_margin,
        )
    }
}

/// IMP-159a: Test throughput variance calculation
#[test]
fn test_imp_159a_throughput_variance_calculation() {
    // Stable throughput samples (low variance)
    let stable_samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0,
    ];
    let stable = ThroughputWithVariance::from_samples(&stable_samples);

    // IMP-159a: Mean should be ~100
    assert!(
        (stable.mean_tps - 100.1).abs() < 0.5,
        "IMP-159a: Mean should be ~100, got {:.2}",
        stable.mean_tps
    );

    // IMP-159a: CV should be low (< 5%)
    assert!(
        stable.cv < 0.05,
        "IMP-159a: CV for stable samples should be < 5%, got {:.4}",
        stable.cv
    );

    // IMP-159a: Should meet CV threshold
    assert!(
        stable.meets_cv_threshold(0.10),
        "IMP-159a: Stable samples should meet 10% CV threshold"
    );

    println!("\nIMP-159a: Throughput Variance Calculation:");
    println!("  Samples: {:?}", stable_samples);
    println!("  Mean: {:.2} tok/s", stable.mean_tps);
    println!("  Stddev: {:.2} tok/s", stable.stddev_tps);
    println!("  CV: {:.4} ({:.1}%)", stable.cv, stable.cv * 100.0);
    println!(
        "  95% CI: ({:.2}, {:.2})",
        stable.confidence_interval().0,
        stable.confidence_interval().1
    );
}

/// IMP-159b: Variance-aware throughput comparison
#[derive(Debug, Clone)]
pub struct VarianceAwareComparison {
    /// First measurement (e.g., Realizar)
    pub measurement_a: ThroughputWithVariance,
    /// Second measurement (e.g., llama.cpp)
    pub measurement_b: ThroughputWithVariance,
    /// Ratio of means (B/A)
    pub mean_ratio: f64,
    /// Whether difference is statistically significant
    pub statistically_significant: bool,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
}

impl VarianceAwareComparison {
    /// Compare two measurements with statistical analysis
    pub fn compare(a: &ThroughputWithVariance, b: &ThroughputWithVariance) -> Self {
        let mean_ratio = if a.mean_tps > 0.0 {
            b.mean_tps / a.mean_tps
        } else {
            1.0
        };

        // Cohen's d effect size
        let pooled_stddev = f64::midpoint(a.stddev_tps.powi(2), b.stddev_tps.powi(2)).sqrt();
        let effect_size = if pooled_stddev > 0.0 {
            (b.mean_tps - a.mean_tps).abs() / pooled_stddev
        } else {
            0.0
        };

        // Statistical significance: CI don't overlap
        let (a_lower, a_upper) = a.confidence_interval();
        let (b_lower, b_upper) = b.confidence_interval();
        let statistically_significant = a_upper < b_lower || b_upper < a_lower;

        Self {
            measurement_a: a.clone(),
            measurement_b: b.clone(),
            mean_ratio,
            statistically_significant,
            effect_size,
        }
    }

    /// Check if B is significantly faster than A
    pub fn b_significantly_faster(&self) -> bool {
        self.statistically_significant && self.mean_ratio > 1.0
    }

    /// Get effect size interpretation (small/medium/large per Cohen)
    pub fn effect_interpretation(&self) -> &'static str {
        if self.effect_size < 0.2 {
            "negligible"
        } else if self.effect_size < 0.5 {
            "small"
        } else if self.effect_size < 0.8 {
            "medium"
        } else {
            "large"
        }
    }
}

/// IMP-159b: Test variance-aware comparison
#[test]
fn test_imp_159b_variance_aware_comparison() {
    // Realizar samples: ~80 tok/s
    let realizar_samples = vec![78.0, 82.0, 80.0, 79.0, 81.0, 80.0, 77.0, 83.0, 80.0, 79.0];
    let realizar = ThroughputWithVariance::from_samples(&realizar_samples);

    // llama.cpp samples: ~256 tok/s
    let llamacpp_samples = vec![
        250.0, 260.0, 255.0, 252.0, 258.0, 256.0, 248.0, 262.0, 254.0, 257.0,
    ];
    let llamacpp = ThroughputWithVariance::from_samples(&llamacpp_samples);

    let comparison = VarianceAwareComparison::compare(&realizar, &llamacpp);

    // IMP-159b: Ratio should be ~3.2x
    assert!(
        comparison.mean_ratio > 3.0 && comparison.mean_ratio < 3.5,
        "IMP-159b: Ratio should be ~3.2x, got {:.2}x",
        comparison.mean_ratio
    );

    // IMP-159b: Difference should be statistically significant
    assert!(
        comparison.statistically_significant,
        "IMP-159b: 3.2x difference should be statistically significant"
    );

    // IMP-159b: Effect size should be large
    assert!(
        comparison.effect_size > 0.8,
        "IMP-159b: Effect size should be large (>0.8), got {:.2}",
        comparison.effect_size
    );

    println!("\nIMP-159b: Variance-Aware Comparison:");
    println!(
        "  Realizar: {:.2} ± {:.2} tok/s (CV={:.4})",
        realizar.mean_tps, realizar.ci_95_margin, realizar.cv
    );
    println!(
        "  llama.cpp: {:.2} ± {:.2} tok/s (CV={:.4})",
        llamacpp.mean_tps, llamacpp.ci_95_margin, llamacpp.cv
    );
    println!("  Ratio: {:.2}x", comparison.mean_ratio);
    println!("  Significant: {}", comparison.statistically_significant);
    println!(
        "  Effect size: {:.2} ({})",
        comparison.effect_size,
        comparison.effect_interpretation()
    );
}

/// IMP-159c: CV-based stopping criterion per Hoefler & Belli
#[derive(Debug, Clone)]
pub struct AdaptiveSampler {
    /// Target CV threshold
    pub target_cv: f64,
    /// Minimum samples before checking CV
    pub min_samples: usize,
    /// Maximum samples (hard limit)
    pub max_samples: usize,
    /// Current samples
    samples: Vec<f64>,
}

impl AdaptiveSampler {
    pub fn new(target_cv: f64, min_samples: usize, max_samples: usize) -> Self {
        Self {
            target_cv,
            min_samples,
            max_samples,
            samples: Vec::new(),
        }
    }

    /// Add a sample and check if we should stop
    pub fn add_sample(&mut self, value: f64) -> bool {
        self.samples.push(value);

        // Check stopping criterion
        if self.samples.len() < self.min_samples {
            return false; // Need more samples
        }

        if self.samples.len() >= self.max_samples {
            return true; // Hit max limit
        }

        // Check CV
        let stats = ThroughputWithVariance::from_samples(&self.samples);
        stats.cv <= self.target_cv
    }

    /// Get current statistics
    pub fn current_stats(&self) -> ThroughputWithVariance {
        ThroughputWithVariance::from_samples(&self.samples)
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// IMP-159c: Test adaptive sampling with CV stopping
#[test]
fn test_imp_159c_adaptive_cv_stopping() {
    // Scenario 1: Stable measurements should stop early
    let mut sampler = AdaptiveSampler::new(0.05, 5, 20);
    let stable_values = [100.0, 101.0, 99.0, 100.0, 100.0, 101.0, 99.0, 100.0];

    let mut stopped_at = 0;
    for (i, &value) in stable_values.iter().enumerate() {
        if sampler.add_sample(value) {
            stopped_at = i + 1;
            break;
        }
    }

    // IMP-159c: Should stop early with stable values
    assert!(
        stopped_at >= 5 && stopped_at <= 8,
        "IMP-159c: Stable values should stop at 5-8 samples, stopped at {}",
        stopped_at
    );

    let final_stats = sampler.current_stats();
    assert!(
        final_stats.cv <= 0.05,
        "IMP-159c: Final CV should be <= 5%, got {:.4}",
        final_stats.cv
    );

    println!("\nIMP-159c: Adaptive CV Stopping:");
    println!("  Target CV: {:.2}%", sampler.target_cv * 100.0);
    println!("  Stopped at: {} samples", stopped_at);
    println!(
        "  Final CV: {:.4} ({:.2}%)",
        final_stats.cv,
        final_stats.cv * 100.0
    );
    println!("  Mean: {:.2} tok/s", final_stats.mean_tps);
}

/// IMP-159d: Real-world variance tracking with llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_159d_realworld_variance_tracking() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Count from 1 to 10:".to_string(),
        max_tokens: 30,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect samples with adaptive stopping
    let mut sampler = AdaptiveSampler::new(0.10, 5, 15);
    let mut iteration = 0;

    while !sampler.add_sample(0.0) && iteration < 15 {
        let start = std::time::Instant::now();
        if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
            let elapsed = start.elapsed().as_secs_f64();
            let tokens = result.text.split_whitespace().count();
            let throughput = tokens as f64 / elapsed;

            // Replace the dummy 0.0 with actual throughput
            sampler.samples.pop();
            sampler.samples.push(throughput);
        }
        iteration += 1;
    }

    let stats = sampler.current_stats();

    // IMP-159d: Verify we got meaningful measurements
    assert!(
        stats.sample_count >= 5,
        "IMP-159d: Should collect at least 5 samples, got {}",
        stats.sample_count
    );

    assert!(
        stats.mean_tps > 10.0,
        "IMP-159d: Mean throughput should be > 10 tok/s, got {:.2}",
        stats.mean_tps
    );

    println!("\nIMP-159d: Real-World Variance Tracking (llama.cpp):");
    println!("  Samples collected: {}", stats.sample_count);
    println!("  Mean throughput: {:.2} tok/s", stats.mean_tps);
    println!("  Stddev: {:.2} tok/s", stats.stddev_tps);
    println!("  CV: {:.4} ({:.2}%)", stats.cv, stats.cv * 100.0);
    println!(
        "  95% CI: ({:.2}, {:.2})",
        stats.confidence_interval().0,
        stats.confidence_interval().1
    );
    println!(
        "  Meets 10% CV threshold: {}",
        stats.meets_cv_threshold(0.10)
    );
}

// =========================================================================
// IMP-160: Multi-Run Statistical Benchmark Analysis (EXTREME TDD)
// =========================================================================
// Per spec: Scientific benchmarking requires multiple independent runs.
// Implements bootstrap confidence intervals and effect size analysis.
// Run with: cargo test test_imp_160 --lib --features bench-http

/// IMP-160a: Multi-run benchmark result aggregation
#[derive(Debug, Clone)]
pub struct MultiRunBenchmark {
    /// Server name being benchmarked
    pub server_name: String,
    /// Number of complete benchmark runs
    pub run_count: usize,
    /// Results from each run (each run has its own stats)
    pub run_results: Vec<ThroughputWithVariance>,
    /// Aggregated mean across all runs
    pub aggregate_mean_tps: f64,
    /// Standard deviation of run means
    pub run_mean_stddev: f64,
    /// CV of run means (variability between runs)
    pub between_run_cv: f64,
    /// Overall sample count (sum of all runs)
    pub total_samples: usize,
}

impl MultiRunBenchmark {
    /// Create from multiple benchmark runs
    pub fn from_runs(server_name: &str, runs: Vec<ThroughputWithVariance>) -> Self {
        let run_count = runs.len();
        if run_count == 0 {
            return Self {
                server_name: server_name.to_string(),
                run_count: 0,
                run_results: Vec::new(),
                aggregate_mean_tps: 0.0,
                run_mean_stddev: 0.0,
                between_run_cv: 0.0,
                total_samples: 0,
            };
        }

        // Collect run means for aggregation
        let run_means: Vec<f64> = runs.iter().map(|r| r.mean_tps).collect();
        let total_samples: usize = runs.iter().map(|r| r.sample_count).sum();

        // Aggregate statistics
        let aggregate_mean = run_means.iter().sum::<f64>() / run_count as f64;
        let variance = if run_count > 1 {
            run_means
                .iter()
                .map(|x| (x - aggregate_mean).powi(2))
                .sum::<f64>()
                / (run_count - 1) as f64
        } else {
            0.0
        };
        let run_stddev = variance.sqrt();
        let cv = if aggregate_mean > 0.0 {
            run_stddev / aggregate_mean
        } else {
            0.0
        };

        Self {
            server_name: server_name.to_string(),
            run_count,
            run_results: runs,
            aggregate_mean_tps: aggregate_mean,
            run_mean_stddev: run_stddev,
            between_run_cv: cv,
            total_samples,
        }
    }

    /// Check if results are reproducible (low between-run variance)
    pub fn is_reproducible(&self, cv_threshold: f64) -> bool {
        self.run_count >= 3 && self.between_run_cv <= cv_threshold
    }

    /// Get bootstrap 95% CI from run means
    pub fn bootstrap_ci(&self) -> (f64, f64) {
        if self.run_count < 3 {
            return (self.aggregate_mean_tps, self.aggregate_mean_tps);
        }
        // Simple percentile bootstrap approximation
        let t_value = 2.0; // Approximate for small samples
        let margin = t_value * self.run_mean_stddev / (self.run_count as f64).sqrt();
        (
            self.aggregate_mean_tps - margin,
            self.aggregate_mean_tps + margin,
        )
    }
}

/// IMP-160a: Test multi-run aggregation
#[test]
fn test_imp_160a_multirun_aggregation() {
    // Simulate 5 benchmark runs for llama.cpp
    let run1 = ThroughputWithVariance::from_samples(&[254.0, 258.0, 252.0, 256.0, 255.0]);
    let run2 = ThroughputWithVariance::from_samples(&[260.0, 262.0, 258.0, 261.0, 259.0]);
    let run3 = ThroughputWithVariance::from_samples(&[248.0, 252.0, 250.0, 249.0, 251.0]);
    let run4 = ThroughputWithVariance::from_samples(&[255.0, 257.0, 254.0, 256.0, 256.0]);
    let run5 = ThroughputWithVariance::from_samples(&[250.0, 252.0, 251.0, 253.0, 249.0]);

    let multirun = MultiRunBenchmark::from_runs("llama.cpp", vec![run1, run2, run3, run4, run5]);

    // IMP-160a: Should have 5 runs
    assert_eq!(multirun.run_count, 5, "IMP-160a: Should have 5 runs");

    // IMP-160a: Aggregate mean should be ~255
    assert!(
        (multirun.aggregate_mean_tps - 254.0).abs() < 3.0,
        "IMP-160a: Aggregate mean should be ~254, got {:.2}",
        multirun.aggregate_mean_tps
    );

    // IMP-160a: Between-run CV should be low (reproducible)
    assert!(
        multirun.between_run_cv < 0.05,
        "IMP-160a: Between-run CV should be < 5%, got {:.4}",
        multirun.between_run_cv
    );

    // IMP-160a: Should be reproducible
    assert!(
        multirun.is_reproducible(0.10),
        "IMP-160a: Results should be reproducible"
    );

    println!("\nIMP-160a: Multi-Run Aggregation:");
    println!("  Server: {}", multirun.server_name);
    println!("  Runs: {}", multirun.run_count);
    println!("  Total samples: {}", multirun.total_samples);
    println!("  Aggregate mean: {:.2} tok/s", multirun.aggregate_mean_tps);
    println!(
        "  Between-run stddev: {:.2} tok/s",
        multirun.run_mean_stddev
    );
    println!(
        "  Between-run CV: {:.4} ({:.2}%)",
        multirun.between_run_cv,
        multirun.between_run_cv * 100.0
    );
    println!(
        "  Bootstrap 95% CI: ({:.2}, {:.2})",
        multirun.bootstrap_ci().0,
        multirun.bootstrap_ci().1
    );
}

/// IMP-160b: Multi-run comparison between servers
#[derive(Debug, Clone)]
pub struct MultiRunComparison {
    /// Server A (e.g., Realizar)
    pub server_a: MultiRunBenchmark,
    /// Server B (e.g., llama.cpp)
    pub server_b: MultiRunBenchmark,
    /// Ratio of aggregate means (B/A)
    pub aggregate_ratio: f64,
    /// Whether difference is reproducibly significant
    pub reproducibly_significant: bool,
    /// Minimum observed ratio across runs
    pub min_ratio: f64,
    /// Maximum observed ratio across runs
    pub max_ratio: f64,
}

impl MultiRunComparison {
    pub fn compare(a: MultiRunBenchmark, b: MultiRunBenchmark) -> Self {
        let aggregate_ratio = if a.aggregate_mean_tps > 0.0 {
            b.aggregate_mean_tps / a.aggregate_mean_tps
        } else {
            1.0
        };

        // Calculate min/max ratio from individual runs
        let ratios: Vec<f64> = a
            .run_results
            .iter()
            .zip(b.run_results.iter())
            .map(|(ra, rb)| {
                if ra.mean_tps > 0.0 {
                    rb.mean_tps / ra.mean_tps
                } else {
                    1.0
                }
            })
            .collect();

        let min_ratio = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ratio = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Reproducibly significant if CIs don't overlap and both are reproducible
        let (a_lower, a_upper) = a.bootstrap_ci();
        let (b_lower, b_upper) = b.bootstrap_ci();
        let ci_separated = a_upper < b_lower || b_upper < a_lower;
        let reproducibly_significant =
            ci_separated && a.is_reproducible(0.15) && b.is_reproducible(0.15);

        Self {
            server_a: a,
            server_b: b,
            aggregate_ratio,
            reproducibly_significant,
            min_ratio,
            max_ratio,
        }
    }
}

/// IMP-160b: Test multi-run comparison
#[test]
fn test_imp_160b_multirun_comparison() {
    // Realizar runs: ~80 tok/s
    let r1 = ThroughputWithVariance::from_samples(&[78.0, 82.0, 80.0, 79.0, 81.0]);
    let r2 = ThroughputWithVariance::from_samples(&[80.0, 81.0, 79.0, 80.0, 80.0]);
    let r3 = ThroughputWithVariance::from_samples(&[77.0, 83.0, 80.0, 78.0, 82.0]);
    let realizar = MultiRunBenchmark::from_runs("Realizar", vec![r1, r2, r3]);

    // llama.cpp runs: ~256 tok/s
    let l1 = ThroughputWithVariance::from_samples(&[254.0, 258.0, 256.0, 255.0, 257.0]);
    let l2 = ThroughputWithVariance::from_samples(&[260.0, 262.0, 258.0, 259.0, 261.0]);
    let l3 = ThroughputWithVariance::from_samples(&[248.0, 252.0, 250.0, 251.0, 249.0]);
    let llamacpp = MultiRunBenchmark::from_runs("llama.cpp", vec![l1, l2, l3]);

    let comparison = MultiRunComparison::compare(realizar, llamacpp);

    // IMP-160b: Aggregate ratio should be ~3.2x
    assert!(
        comparison.aggregate_ratio > 3.0 && comparison.aggregate_ratio < 3.5,
        "IMP-160b: Aggregate ratio should be ~3.2x, got {:.2}x",
        comparison.aggregate_ratio
    );

    // IMP-160b: Difference should be reproducibly significant
    assert!(
        comparison.reproducibly_significant,
        "IMP-160b: 3.2x gap should be reproducibly significant"
    );

    println!("\nIMP-160b: Multi-Run Comparison:");
    println!(
        "  Realizar: {:.2} tok/s ({} runs)",
        comparison.server_a.aggregate_mean_tps, comparison.server_a.run_count
    );
    println!(
        "  llama.cpp: {:.2} tok/s ({} runs)",
        comparison.server_b.aggregate_mean_tps, comparison.server_b.run_count
    );
    println!("  Aggregate ratio: {:.2}x", comparison.aggregate_ratio);
    println!(
        "  Ratio range: [{:.2}x, {:.2}x]",
        comparison.min_ratio, comparison.max_ratio
    );
    println!(
        "  Reproducibly significant: {}",
        comparison.reproducibly_significant
    );
}

/// IMP-160c: Statistical power analysis for benchmark design
#[derive(Debug, Clone)]
pub struct BenchmarkPowerAnalysis {
    /// Minimum detectable effect size (Cohen's d)
    pub min_effect_size: f64,
    /// Statistical power achieved (0-1)
    pub power: f64,
    /// Sample size per group
    pub sample_size: usize,
    /// Significance level (alpha)
    pub alpha: f64,
    /// Recommended sample size for desired power
    pub recommended_n: usize,
}

impl BenchmarkPowerAnalysis {
    /// Estimate power for given effect size and sample size
    /// Uses simplified power calculation (normal approximation)
    pub fn estimate(effect_size: f64, sample_size: usize, alpha: f64, _desired_power: f64) -> Self {
        // Z-score for alpha (two-tailed)
        let z_alpha = 1.96; // For alpha = 0.05

        // Estimated power (simplified)
        let sqrt_n = (sample_size as f64).sqrt();
        let noncentrality = effect_size * sqrt_n / 2.0_f64.sqrt();
        let power = 1.0 - (1.0 / (1.0 + (noncentrality - z_alpha).exp())); // Logistic approx

        // Sample size needed for desired power
        let z_beta = 0.84; // For power = 0.80
        let recommended_n = if effect_size > 0.0 {
            let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
            n.ceil() as usize
        } else {
            100 // Default if no effect
        };

        Self {
            min_effect_size: effect_size,
            power,
            sample_size,
            alpha,
            recommended_n,
        }
    }

    /// Check if power is adequate for reliable detection
    pub fn is_adequately_powered(&self) -> bool {
        self.power >= 0.80
    }
}

/// IMP-160c: Test power analysis
#[test]
fn test_imp_160c_power_analysis() {
    // Large effect (d=2.0) with small sample - should be well powered
    let large_effect = BenchmarkPowerAnalysis::estimate(2.0, 10, 0.05, 0.80);
    assert!(
        large_effect.power > 0.70,
        "IMP-160c: Large effect with n=10 should have power > 70%, got {:.2}",
        large_effect.power
    );

    // Small effect (d=0.2) with small sample - underpowered
    let small_effect = BenchmarkPowerAnalysis::estimate(0.2, 10, 0.05, 0.80);
    assert!(
        small_effect.power < 0.50,
        "IMP-160c: Small effect with n=10 should have low power, got {:.2}",
        small_effect.power
    );

    // Recommended n for small effect should be large
    assert!(
        small_effect.recommended_n > 50,
        "IMP-160c: Small effect should need many samples, got n={}",
        small_effect.recommended_n
    );

    println!("\nIMP-160c: Power Analysis:");
    println!("  Large effect (d=2.0, n=10):");
    println!("    Power: {:.2}", large_effect.power);
    println!(
        "    Adequately powered: {}",
        large_effect.is_adequately_powered()
    );
    println!("  Small effect (d=0.2, n=10):");
    println!("    Power: {:.2}", small_effect.power);
    println!(
        "    Recommended n for 80% power: {}",
        small_effect.recommended_n
    );
}

/// IMP-160d: Real-world multi-run benchmark against llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_160d_realworld_multirun() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "What is 2+2?".to_string(),
        max_tokens: 20,
        temperature: Some(0.0),
        stream: false,
    };

    // Perform 3 runs, 5 samples each
    let mut runs: Vec<ThroughputWithVariance> = Vec::new();

    for run_idx in 0..3 {
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
                let elapsed = start.elapsed().as_secs_f64();
                let tokens = result.text.split_whitespace().count().max(1);
                samples.push(tokens as f64 / elapsed);
            }
        }
        if !samples.is_empty() {
            runs.push(ThroughputWithVariance::from_samples(&samples));
        }
        println!(
            "  Run {}: {} samples, mean {:.2} tok/s",
            run_idx + 1,
            samples.len(),
            runs.last().map_or(0.0, |r| r.mean_tps)
        );
    }

    let multirun = MultiRunBenchmark::from_runs("llama.cpp", runs);

    // IMP-160d: Verify multi-run results
    assert!(
        multirun.run_count >= 2,
        "IMP-160d: Should complete at least 2 runs, got {}",
        multirun.run_count
    );

    assert!(
        multirun.aggregate_mean_tps > 10.0,
        "IMP-160d: Aggregate mean should be > 10 tok/s, got {:.2}",
        multirun.aggregate_mean_tps
    );

    println!("\nIMP-160d: Real-World Multi-Run Benchmark (llama.cpp):");
    println!("  Completed runs: {}", multirun.run_count);
    println!("  Total samples: {}", multirun.total_samples);
    println!("  Aggregate mean: {:.2} tok/s", multirun.aggregate_mean_tps);
    println!(
        "  Between-run CV: {:.4} ({:.2}%)",
        multirun.between_run_cv,
        multirun.between_run_cv * 100.0
    );
    println!("  Reproducible: {}", multirun.is_reproducible(0.15));
    println!(
        "  Bootstrap 95% CI: ({:.2}, {:.2})",
        multirun.bootstrap_ci().0,
        multirun.bootstrap_ci().1
    );
}

// =========================================================================
// IMP-161: Warmup Detection and JIT Filtering (QA-032, EXTREME TDD)
// =========================================================================
// Per Vitek & Kalibera EMSOFT'11: Detect and remove warmup iterations.
// JIT compilation causes initial measurements to be non-representative.
// Run with: cargo test test_imp_161 --lib --features bench-http

/// IMP-161a: Warmup detection using changepoint analysis
#[derive(Debug, Clone)]
pub struct WarmupDetector {
    /// Minimum iterations before checking for warmup end
    pub min_iterations: usize,
    /// Maximum warmup iterations allowed
    pub max_warmup: usize,
    /// Threshold for detecting stable state (ratio of variance)
    pub stability_threshold: f64,
    /// Window size for moving average
    pub window_size: usize,
}

impl WarmupDetector {
    pub fn new(min_iterations: usize, max_warmup: usize, stability_threshold: f64) -> Self {
        Self {
            min_iterations,
            max_warmup,
            stability_threshold,
            window_size: 5,
        }
    }

    /// Default detector per Vitek & Kalibera recommendations
    pub fn default_detector() -> Self {
        Self::new(3, 10, 0.20)
    }

    /// Detect warmup end using variance ratio method
    /// Returns (warmup_iterations, steady_state_samples)
    pub fn detect_warmup(&self, samples: &[f64]) -> WarmupResult {
        let n = samples.len();
        if n < self.min_iterations + self.window_size {
            return WarmupResult {
                warmup_iterations: 0,
                steady_state_samples: samples.to_vec(),
                warmup_detected: false,
                variance_ratio: 1.0,
            };
        }

        // Calculate variance of first window vs later windows
        let mut best_split = 0;
        let mut best_ratio = f64::MAX;

        for split in self.min_iterations..n.saturating_sub(self.window_size).min(self.max_warmup) {
            let warmup = &samples[..split];
            let steady = &samples[split..];

            if warmup.len() < 2 || steady.len() < 2 {
                continue;
            }

            let warmup_var = Self::variance(warmup);
            let steady_var = Self::variance(steady);

            // If steady state has much lower variance, we found warmup end
            if warmup_var > 0.0 && steady_var > 0.0 {
                let ratio = steady_var / warmup_var;
                if ratio < best_ratio {
                    best_ratio = ratio;
                    best_split = split;
                }
            }
        }

        // Check if we detected significant warmup
        let warmup_detected = best_ratio < self.stability_threshold && best_split > 0;

        let (warmup_iters, steady_samples) = if warmup_detected {
            (best_split, samples[best_split..].to_vec())
        } else {
            (0, samples.to_vec())
        };

        WarmupResult {
            warmup_iterations: warmup_iters,
            steady_state_samples: steady_samples,
            warmup_detected,
            variance_ratio: best_ratio,
        }
    }

    fn variance(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64
    }
}

/// IMP-161a: Result of warmup detection
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Number of warmup iterations detected
    pub warmup_iterations: usize,
    /// Samples after warmup removal
    pub steady_state_samples: Vec<f64>,
    /// Whether warmup was detected
    pub warmup_detected: bool,
    /// Variance ratio (steady/warmup)
    pub variance_ratio: f64,
}

impl WarmupResult {
    /// Get statistics from steady state only
    pub fn steady_state_stats(&self) -> ThroughputWithVariance {
        ThroughputWithVariance::from_samples(&self.steady_state_samples)
    }
}

/// IMP-161a: Test warmup detection
#[test]
fn test_imp_161a_warmup_detection() {
    // Simulate warmup: first 5 samples are slow (JIT not warmed up)
    let samples = vec![
        50.0, 55.0, 60.0, 70.0, 80.0, // Warmup phase (improving)
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0, 100.0, 102.0, // Steady state
    ];

    let detector = WarmupDetector::default_detector();
    let result = detector.detect_warmup(&samples);

    // IMP-161a: Should detect warmup
    assert!(
        result.warmup_detected,
        "IMP-161a: Should detect warmup in ramping data"
    );

    // IMP-161a: Warmup should be 3-10 iterations (algorithm finds optimal variance split)
    assert!(
        result.warmup_iterations >= 3 && result.warmup_iterations <= 10,
        "IMP-161a: Warmup should be 3-10 iterations, got {}",
        result.warmup_iterations
    );

    // IMP-161a: Steady state should have higher mean
    let steady_stats = result.steady_state_stats();
    assert!(
        steady_stats.mean_tps > 90.0,
        "IMP-161a: Steady state mean should be >90, got {:.2}",
        steady_stats.mean_tps
    );

    println!("\nIMP-161a: Warmup Detection:");
    println!("  Raw samples: {:?}", samples);
    println!("  Warmup detected: {}", result.warmup_detected);
    println!("  Warmup iterations: {}", result.warmup_iterations);
    println!("  Variance ratio: {:.4}", result.variance_ratio);
    println!("  Steady state mean: {:.2} tok/s", steady_stats.mean_tps);
    println!("  Steady state CV: {:.4}", steady_stats.cv);
}

/// IMP-161b: JIT-aware benchmark runner
#[derive(Debug, Clone)]
pub struct JitAwareBenchmark {
    /// Warmup detector configuration
    pub detector: WarmupDetector,
    /// Results before warmup removal
    pub raw_stats: ThroughputWithVariance,
    /// Results after warmup removal
    pub filtered_stats: ThroughputWithVariance,
    /// Warmup detection result
    pub warmup_result: WarmupResult,
    /// Improvement from filtering (percentage)
    pub improvement_percent: f64,
}

impl JitAwareBenchmark {
    pub fn analyze(samples: &[f64]) -> Self {
        let detector = WarmupDetector::default_detector();
        let raw_stats = ThroughputWithVariance::from_samples(samples);
        let warmup_result = detector.detect_warmup(samples);
        let filtered_stats = warmup_result.steady_state_stats();

        let improvement = if raw_stats.mean_tps > 0.0 {
            ((filtered_stats.mean_tps - raw_stats.mean_tps) / raw_stats.mean_tps) * 100.0
        } else {
            0.0
        };

        Self {
            detector,
            raw_stats,
            filtered_stats,
            warmup_result,
            improvement_percent: improvement,
        }
    }

    /// Check if JIT filtering made a significant difference
    pub fn filtering_significant(&self) -> bool {
        self.warmup_result.warmup_detected && self.improvement_percent.abs() > 5.0
    }
}

/// IMP-161b: Test JIT-aware benchmark analysis
#[test]
fn test_imp_161b_jit_aware_benchmark() {
    // Simulate JIT warmup scenario
    let samples = vec![
        40.0, 60.0, 80.0, 90.0, 95.0, // JIT warming up
        100.0, 98.0, 102.0, 99.0, 101.0, 100.0, 99.0, 101.0, 100.0, 98.0, // JIT hot
    ];

    let analysis = JitAwareBenchmark::analyze(&samples);

    // IMP-161b: Filtered mean should be higher than raw
    assert!(
        analysis.filtered_stats.mean_tps > analysis.raw_stats.mean_tps,
        "IMP-161b: Filtered mean should be higher after removing warmup"
    );

    // IMP-161b: Should show significant improvement
    assert!(
        analysis.improvement_percent > 5.0,
        "IMP-161b: Should show >5% improvement, got {:.2}%",
        analysis.improvement_percent
    );

    // IMP-161b: Filtering should be significant
    assert!(
        analysis.filtering_significant(),
        "IMP-161b: JIT filtering should be significant"
    );

    println!("\nIMP-161b: JIT-Aware Benchmark:");
    println!(
        "  Raw mean: {:.2} tok/s (n={})",
        analysis.raw_stats.mean_tps, analysis.raw_stats.sample_count
    );
    println!(
        "  Filtered mean: {:.2} tok/s (n={})",
        analysis.filtered_stats.mean_tps, analysis.filtered_stats.sample_count
    );
    println!("  Improvement: {:.2}%", analysis.improvement_percent);
    println!(
        "  Warmup removed: {} iterations",
        analysis.warmup_result.warmup_iterations
    );
    println!(
        "  Filtering significant: {}",
        analysis.filtering_significant()
    );
}

/// IMP-161c: Cold vs warm start detection
#[derive(Debug, Clone)]
pub struct ColdWarmComparison {
    /// Cold start measurement (first request)
    pub cold_latency_ms: f64,
    /// Warm start measurement (subsequent average)
    pub warm_latency_ms: f64,
    /// Cold start penalty ratio
    pub cold_penalty_ratio: f64,
    /// Whether cold start penalty is significant (>2x)
    pub significant_cold_penalty: bool,
}

impl ColdWarmComparison {
    pub fn analyze(latencies: &[f64]) -> Self {
        if latencies.is_empty() {
            return Self {
                cold_latency_ms: 0.0,
                warm_latency_ms: 0.0,
                cold_penalty_ratio: 1.0,
                significant_cold_penalty: false,
            };
        }

        let cold_latency = latencies[0];
        let warm_latency = if latencies.len() > 1 {
            latencies[1..].iter().sum::<f64>() / (latencies.len() - 1) as f64
        } else {
            cold_latency
        };

        let penalty_ratio = if warm_latency > 0.0 {
            cold_latency / warm_latency
        } else {
            1.0
        };

        Self {
            cold_latency_ms: cold_latency,
            warm_latency_ms: warm_latency,
            cold_penalty_ratio: penalty_ratio,
            significant_cold_penalty: penalty_ratio > 2.0,
        }
    }
}

/// IMP-161c: Test cold/warm start detection
#[test]
fn test_imp_161c_cold_warm_detection() {
    // Simulate cold start: first request is slow
    let latencies = vec![
        500.0, // Cold start (model loading, JIT compilation)
        100.0, 105.0, 98.0, 102.0, 99.0, 101.0, 100.0, 103.0, 97.0, // Warm
    ];

    let analysis = ColdWarmComparison::analyze(&latencies);

    // IMP-161c: Cold start should be ~500ms
    assert!(
        (analysis.cold_latency_ms - 500.0).abs() < 1.0,
        "IMP-161c: Cold latency should be 500ms, got {:.2}",
        analysis.cold_latency_ms
    );

    // IMP-161c: Warm latency should be ~100ms
    assert!(
        (analysis.warm_latency_ms - 100.5).abs() < 5.0,
        "IMP-161c: Warm latency should be ~100ms, got {:.2}",
        analysis.warm_latency_ms
    );

    // IMP-161c: Cold penalty should be significant (~5x)
    assert!(
        analysis.significant_cold_penalty,
        "IMP-161c: Cold start penalty should be significant"
    );

    assert!(
        analysis.cold_penalty_ratio > 4.0 && analysis.cold_penalty_ratio < 6.0,
        "IMP-161c: Cold penalty ratio should be ~5x, got {:.2}x",
        analysis.cold_penalty_ratio
    );

    println!("\nIMP-161c: Cold/Warm Start Detection:");
    println!("  Cold start latency: {:.2} ms", analysis.cold_latency_ms);
    println!("  Warm average latency: {:.2} ms", analysis.warm_latency_ms);
    println!("  Cold penalty ratio: {:.2}x", analysis.cold_penalty_ratio);
    println!(
        "  Significant penalty: {}",
        analysis.significant_cold_penalty
    );
}

/// IMP-161d: Real-world warmup detection with llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_161d_realworld_warmup_detection() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Say hello:".to_string(),
        max_tokens: 10,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect 15 samples to detect warmup
    let mut latencies_ms = Vec::new();
    for _ in 0..15 {
        let start = std::time::Instant::now();
        if client
            .llamacpp_completion("http://127.0.0.1:8082", &request)
            .is_ok()
        {
            latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if latencies_ms.len() < 10 {
        println!("IMP-161d: Not enough samples collected");
        return;
    }

    // Convert to throughput (approximated from latency)
    let throughputs: Vec<f64> = latencies_ms
        .iter()
        .map(|lat| if *lat > 0.0 { 10000.0 / lat } else { 0.0 }) // ~10 tokens
        .collect();

    let analysis = JitAwareBenchmark::analyze(&throughputs);
    let cold_warm = ColdWarmComparison::analyze(&latencies_ms);

    println!("\nIMP-161d: Real-World Warmup Detection (llama.cpp):");
    println!("  Samples collected: {}", latencies_ms.len());
    println!("  Raw mean: {:.2} tok/s", analysis.raw_stats.mean_tps);
    println!(
        "  Filtered mean: {:.2} tok/s",
        analysis.filtered_stats.mean_tps
    );
    println!(
        "  Warmup iterations: {}",
        analysis.warmup_result.warmup_iterations
    );
    println!(
        "  Filtering improvement: {:.2}%",
        analysis.improvement_percent
    );
    println!("  Cold start latency: {:.2} ms", cold_warm.cold_latency_ms);
    println!("  Warm latency: {:.2} ms", cold_warm.warm_latency_ms);
    println!("  Cold penalty: {:.2}x", cold_warm.cold_penalty_ratio);
}

// =========================================================================
// IMP-162: MAD Outlier Detection Verification (QA-034, EXTREME TDD)
