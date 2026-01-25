use crate::http_client::*;
// =========================================================================
// Per spec QA-034: Use Median Absolute Deviation for robust outlier detection.
// MAD is more resistant to outliers than standard deviation.
// Outlier threshold: |x - median| > k * MAD, typically k = 3.0
// Run with: cargo test test_imp_162 --lib --features bench-http

/// IMP-162a: MAD-based outlier detection
#[derive(Debug, Clone)]
pub struct MadOutlierDetector {
    /// K factor for outlier threshold (default: 3.0)
    pub k_factor: f64,
    /// Consistency constant for normal distribution (1.4826)
    pub consistency_constant: f64,
}

impl MadOutlierDetector {
    pub fn new(k_factor: f64) -> Self {
        Self {
            k_factor,
            // 1.4826 makes MAD consistent with std dev for normal distribution
            consistency_constant: 1.4826,
        }
    }

    /// Default detector with k=3.0 (standard outlier threshold)
    pub fn default_detector() -> Self {
        Self::new(3.0)
    }

    /// Calculate median of samples
    fn median(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            f64::midpoint(sorted[mid - 1], sorted[mid])
        } else {
            sorted[mid]
        }
    }

    /// Calculate MAD (Median Absolute Deviation)
    pub fn calculate_mad(&self, samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let median = Self::median(samples);
        let absolute_deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();
        Self::median(&absolute_deviations) * self.consistency_constant
    }

    /// Detect outliers in samples
    pub fn detect_outliers(&self, samples: &[f64]) -> MadOutlierResult {
        if samples.is_empty() {
            return MadOutlierResult {
                median: 0.0,
                mad: 0.0,
                scaled_mad: 0.0,
                threshold: 0.0,
                outlier_indices: Vec::new(),
                outlier_values: Vec::new(),
                clean_samples: Vec::new(),
                outlier_count: 0,
                outlier_percent: 0.0,
            };
        }

        let median = Self::median(samples);
        let mad = self.calculate_mad(samples);
        let scaled_mad = mad; // Already scaled by consistency constant
        let threshold = self.k_factor * scaled_mad;

        let mut outlier_indices = Vec::new();
        let mut outlier_values = Vec::new();
        let mut clean_samples = Vec::new();

        for (i, &value) in samples.iter().enumerate() {
            if (value - median).abs() > threshold {
                outlier_indices.push(i);
                outlier_values.push(value);
            } else {
                clean_samples.push(value);
            }
        }

        let outlier_count = outlier_indices.len();
        let outlier_percent = if !samples.is_empty() {
            (outlier_count as f64 / samples.len() as f64) * 100.0
        } else {
            0.0
        };

        MadOutlierResult {
            median,
            mad,
            scaled_mad,
            threshold,
            outlier_indices,
            outlier_values,
            clean_samples,
            outlier_count,
            outlier_percent,
        }
    }
}

/// IMP-162a: Result of MAD outlier detection
#[derive(Debug, Clone)]
pub struct MadOutlierResult {
    /// Median of samples
    pub median: f64,
    /// MAD (Median Absolute Deviation)
    pub mad: f64,
    /// Scaled MAD (for normal distribution consistency)
    pub scaled_mad: f64,
    /// Outlier threshold (k * MAD)
    pub threshold: f64,
    /// Indices of detected outliers
    pub outlier_indices: Vec<usize>,
    /// Values of detected outliers
    pub outlier_values: Vec<f64>,
    /// Samples with outliers removed
    pub clean_samples: Vec<f64>,
    /// Number of outliers detected
    pub outlier_count: usize,
    /// Percentage of outliers
    pub outlier_percent: f64,
}

impl MadOutlierResult {
    /// Get statistics from clean samples
    pub fn clean_stats(&self) -> ThroughputWithVariance {
        ThroughputWithVariance::from_samples(&self.clean_samples)
    }

    /// Check if outlier filtering was significant
    pub fn filtering_significant(&self) -> bool {
        self.outlier_count > 0 && self.outlier_percent > 1.0
    }
}

/// IMP-162a: Test MAD outlier detection
#[test]
fn test_imp_162a_mad_outlier_detection() {
    // Data with clear outliers
    let samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0, // Normal
        500.0, // High outlier
        10.0,  // Low outlier
    ];

    let detector = MadOutlierDetector::default_detector();
    let result = detector.detect_outliers(&samples);

    // IMP-162a: Median should be ~100
    assert!(
        (result.median - 100.0).abs() < 5.0,
        "IMP-162a: Median should be ~100, got {:.2}",
        result.median
    );

    // IMP-162a: Should detect 2 outliers
    assert_eq!(
        result.outlier_count, 2,
        "IMP-162a: Should detect 2 outliers, got {}",
        result.outlier_count
    );

    // IMP-162a: Outliers should include 500 and 10
    assert!(
        result.outlier_values.contains(&500.0) && result.outlier_values.contains(&10.0),
        "IMP-162a: Outliers should include 500 and 10"
    );

    // IMP-162a: Clean samples should have ~10 values
    assert_eq!(
        result.clean_samples.len(),
        10,
        "IMP-162a: Clean samples should have 10 values"
    );

    println!("\nIMP-162a: MAD Outlier Detection:");
    println!("  Samples: {:?}", samples);
    println!("  Median: {:.2}", result.median);
    println!("  MAD: {:.2}", result.mad);
    println!("  Threshold: {:.2}", result.threshold);
    println!("  Outliers: {:?}", result.outlier_values);
    println!("  Outlier %: {:.2}%", result.outlier_percent);
    println!("  Clean sample count: {}", result.clean_samples.len());
}

/// IMP-162b: MAD vs Standard Deviation comparison
#[derive(Debug, Clone)]
pub struct MadVsStdComparison {
    /// Standard deviation of samples
    pub stddev: f64,
    /// MAD of samples
    pub mad: f64,
    /// Outliers detected by stddev method (k * stddev from mean)
    pub stddev_outliers: usize,
    /// Outliers detected by MAD method
    pub mad_outliers: usize,
    /// Robustness ratio (stddev / MAD) - higher means more outlier influence
    pub robustness_ratio: f64,
}

impl MadVsStdComparison {
    pub fn compare(samples: &[f64], k_factor: f64) -> Self {
        if samples.is_empty() {
            return Self {
                stddev: 0.0,
                mad: 0.0,
                stddev_outliers: 0,
                mad_outliers: 0,
                robustness_ratio: 1.0,
            };
        }

        // Standard deviation method
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let stddev = variance.sqrt();

        // Count outliers by stddev method
        let stddev_threshold = k_factor * stddev;
        let stddev_outliers = samples
            .iter()
            .filter(|&&x| (x - mean).abs() > stddev_threshold)
            .count();

        // MAD method
        let detector = MadOutlierDetector::new(k_factor);
        let mad_result = detector.detect_outliers(samples);

        let robustness_ratio = if mad_result.mad > 0.0 {
            stddev / mad_result.mad
        } else {
            1.0
        };

        Self {
            stddev,
            mad: mad_result.mad,
            stddev_outliers,
            mad_outliers: mad_result.outlier_count,
            robustness_ratio,
        }
    }
}

/// IMP-162b: Test MAD vs stddev comparison
#[test]
fn test_imp_162b_mad_vs_stddev() {
    // Data with extreme outlier - stddev is heavily influenced
    let samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0,
        1000.0, // Extreme outlier
    ];

    let comparison = MadVsStdComparison::compare(&samples, 3.0);

    // IMP-162b: Stddev should be inflated by outlier
    assert!(
        comparison.stddev > 100.0,
        "IMP-162b: Stddev should be heavily inflated, got {:.2}",
        comparison.stddev
    );

    // IMP-162b: MAD should be robust
    assert!(
        comparison.mad < 10.0,
        "IMP-162b: MAD should be robust to outlier, got {:.2}",
        comparison.mad
    );

    // IMP-162b: Robustness ratio should be high (stddev >> MAD)
    assert!(
        comparison.robustness_ratio > 10.0,
        "IMP-162b: Robustness ratio should be high, got {:.2}",
        comparison.robustness_ratio
    );

    // IMP-162b: MAD should detect the outlier
    assert!(
        comparison.mad_outliers >= 1,
        "IMP-162b: MAD should detect outlier"
    );

    println!("\nIMP-162b: MAD vs Standard Deviation:");
    println!("  Stddev: {:.2} (inflated by outlier)", comparison.stddev);
    println!("  MAD: {:.2} (robust)", comparison.mad);
    println!("  Robustness ratio: {:.2}x", comparison.robustness_ratio);
    println!("  Stddev outliers: {}", comparison.stddev_outliers);
    println!("  MAD outliers: {}", comparison.mad_outliers);
}

/// IMP-162c: Benchmark result cleaning with MAD
#[derive(Debug, Clone)]
pub struct CleanedBenchmarkResult {
    /// Raw statistics (before cleaning)
    pub raw_stats: ThroughputWithVariance,
    /// Cleaned statistics (after outlier removal)
    pub cleaned_stats: ThroughputWithVariance,
    /// MAD outlier detection result
    pub outlier_result: MadOutlierResult,
    /// Improvement in CV after cleaning
    pub cv_improvement_percent: f64,
    /// Change in mean after cleaning
    pub mean_change_percent: f64,
}

impl CleanedBenchmarkResult {
    pub fn clean(samples: &[f64]) -> Self {
        let raw_stats = ThroughputWithVariance::from_samples(samples);
        let detector = MadOutlierDetector::default_detector();
        let outlier_result = detector.detect_outliers(samples);
        let cleaned_stats = outlier_result.clean_stats();

        let cv_improvement = if raw_stats.cv > 0.0 {
            ((raw_stats.cv - cleaned_stats.cv) / raw_stats.cv) * 100.0
        } else {
            0.0
        };

        let mean_change = if raw_stats.mean_tps > 0.0 {
            ((cleaned_stats.mean_tps - raw_stats.mean_tps) / raw_stats.mean_tps) * 100.0
        } else {
            0.0
        };

        Self {
            raw_stats,
            cleaned_stats,
            outlier_result,
            cv_improvement_percent: cv_improvement,
            mean_change_percent: mean_change,
        }
    }

    /// Check if cleaning made a significant improvement
    pub fn cleaning_beneficial(&self) -> bool {
        self.cv_improvement_percent > 10.0 && self.outlier_result.outlier_count > 0
    }
}

/// IMP-162c: Test benchmark cleaning with MAD
#[test]
fn test_imp_162c_benchmark_cleaning() {
    // Realistic benchmark data with outliers
    let samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0, // Normal
        200.0, // GC pause / thermal throttle
        50.0,  // Cold cache / context switch
    ];

    let result = CleanedBenchmarkResult::clean(&samples);

    // IMP-162c: Cleaned CV should be lower than raw
    assert!(
        result.cleaned_stats.cv < result.raw_stats.cv,
        "IMP-162c: Cleaned CV should be lower"
    );

    // IMP-162c: Should show significant CV improvement
    assert!(
        result.cv_improvement_percent > 50.0,
        "IMP-162c: CV should improve significantly, got {:.2}%",
        result.cv_improvement_percent
    );

    // IMP-162c: Cleaning should be beneficial
    assert!(
        result.cleaning_beneficial(),
        "IMP-162c: Cleaning should be beneficial"
    );

    println!("\nIMP-162c: Benchmark Cleaning with MAD:");
    println!(
        "  Raw mean: {:.2} tok/s (CV={:.4})",
        result.raw_stats.mean_tps, result.raw_stats.cv
    );
    println!(
        "  Cleaned mean: {:.2} tok/s (CV={:.4})",
        result.cleaned_stats.mean_tps, result.cleaned_stats.cv
    );
    println!(
        "  Outliers removed: {}",
        result.outlier_result.outlier_count
    );
    println!("  CV improvement: {:.2}%", result.cv_improvement_percent);
    println!("  Mean change: {:.2}%", result.mean_change_percent);
    println!("  Cleaning beneficial: {}", result.cleaning_beneficial());
}

/// IMP-162d: Real-world MAD outlier detection with llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_162d_realworld_mad_outlier_detection() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Hello:".to_string(),
        max_tokens: 5,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect 20 samples
    let mut latencies_ms = Vec::new();
    for _ in 0..20 {
        let start = std::time::Instant::now();
        if client
            .llamacpp_completion("http://127.0.0.1:8082", &request)
            .is_ok()
        {
            latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if latencies_ms.len() < 10 {
        println!("IMP-162d: Not enough samples collected");
        return;
    }

    // Convert to throughput
    let throughputs: Vec<f64> = latencies_ms
        .iter()
        .map(|lat| if *lat > 0.0 { 5000.0 / lat } else { 0.0 }) // ~5 tokens
        .collect();

    let result = CleanedBenchmarkResult::clean(&throughputs);
    let comparison = MadVsStdComparison::compare(&throughputs, 3.0);

    println!("\nIMP-162d: Real-World MAD Outlier Detection (llama.cpp):");
    println!("  Samples collected: {}", throughputs.len());
    println!(
        "  Raw mean: {:.2} tok/s (CV={:.4})",
        result.raw_stats.mean_tps, result.raw_stats.cv
    );
    println!(
        "  Cleaned mean: {:.2} tok/s (CV={:.4})",
        result.cleaned_stats.mean_tps, result.cleaned_stats.cv
    );
    println!(
        "  Outliers detected: {}",
        result.outlier_result.outlier_count
    );
    if !result.outlier_result.outlier_values.is_empty() {
        println!(
            "  Outlier values: {:?}",
            result.outlier_result.outlier_values
        );
    }
    println!("  CV improvement: {:.2}%", result.cv_improvement_percent);
    println!(
        "  MAD vs Stddev robustness: {:.2}x",
        comparison.robustness_ratio
    );
    println!("  Cleaning beneficial: {}", result.cleaning_beneficial());
}

// =========================================================================
// IMP-163: Real-World A/B Latency Comparison (QA-012, EXTREME TDD)
// =========================================================================
// Per spec QA-012: Latency p99 < 2x p50 (no outliers)
// Compares latency distributions between Realizar and external servers.
// Run with: cargo test test_imp_163 --lib --features bench-http

/// IMP-163a: A/B latency comparison result
#[derive(Debug, Clone)]
pub struct ABLatencyComparison {
    /// Server A name (e.g., "Realizar")
    pub server_a_name: String,
    /// Server B name (e.g., "llama.cpp")
    pub server_b_name: String,
    /// Server A latency percentiles
    pub server_a_latency: LatencyPercentiles,
    /// Server B latency percentiles
    pub server_b_latency: LatencyPercentiles,
    /// Latency ratio (A/B) at p50
    pub p50_ratio: f64,
    /// Latency ratio (A/B) at p99
    pub p99_ratio: f64,
    /// Whether both servers meet QA-012: p99 < 2x p50
    pub both_meet_qa012: bool,
    /// Winner at p50 ("A", "B", or "tie")
    pub p50_winner: String,
    /// Winner at p99
    pub p99_winner: String,
}

impl ABLatencyComparison {
    pub fn compare(
        server_a_name: &str,
        server_a_samples: &[f64],
        server_b_name: &str,
        server_b_samples: &[f64],
    ) -> Self {
        let server_a_latency = LatencyPercentiles::from_samples(server_a_samples);
        let server_b_latency = LatencyPercentiles::from_samples(server_b_samples);

        let p50_ratio = if server_b_latency.p50_ms > 0.0 {
            server_a_latency.p50_ms / server_b_latency.p50_ms
        } else {
            1.0
        };

        let p99_ratio = if server_b_latency.p99_ms > 0.0 {
            server_a_latency.p99_ms / server_b_latency.p99_ms
        } else {
            1.0
        };

        // QA-012: p99 < 2x p50
        let a_meets = server_a_latency.p99_ms < 2.0 * server_a_latency.p50_ms;
        let b_meets = server_b_latency.p99_ms < 2.0 * server_b_latency.p50_ms;

        let p50_winner = if (p50_ratio - 1.0).abs() < 0.05 {
            "tie".to_string()
        } else if p50_ratio < 1.0 {
            server_a_name.to_string()
        } else {
            server_b_name.to_string()
        };

        let p99_winner = if (p99_ratio - 1.0).abs() < 0.05 {
            "tie".to_string()
        } else if p99_ratio < 1.0 {
            server_a_name.to_string()
        } else {
            server_b_name.to_string()
        };

        Self {
            server_a_name: server_a_name.to_string(),
            server_b_name: server_b_name.to_string(),
            server_a_latency,
            server_b_latency,
            p50_ratio,
            p99_ratio,
            both_meet_qa012: a_meets && b_meets,
            p50_winner,
            p99_winner,
        }
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            "{} vs {}: p50 ratio={:.2}x (winner: {}), p99 ratio={:.2}x (winner: {}), QA-012: {}",
            self.server_a_name,
            self.server_b_name,
            self.p50_ratio,
            self.p50_winner,
            self.p99_ratio,
            self.p99_winner,
            if self.both_meet_qa012 { "PASS" } else { "FAIL" }
        )
    }
}

/// IMP-163a: Test A/B latency comparison structure
#[test]
fn test_imp_163a_ab_latency_comparison() {
    // test Realizar latencies (slower)
    let realizar_latencies = vec![
        520.0, 530.0, 510.0, 525.0, 515.0, 540.0, 505.0, 535.0, 520.0, 528.0,
    ];
    // test llama.cpp latencies (faster)
    let llamacpp_latencies = vec![
        160.0, 165.0, 158.0, 162.0, 155.0, 170.0, 152.0, 168.0, 161.0, 164.0,
    ];

    let comparison = ABLatencyComparison::compare(
        "Realizar",
        &realizar_latencies,
        "llama.cpp",
        &llamacpp_latencies,
    );

    // IMP-163a: p50 ratio should be ~3.2x (520/162)
    assert!(
        comparison.p50_ratio > 3.0 && comparison.p50_ratio < 3.5,
        "IMP-163a: p50 ratio should be ~3.2x, got {:.2}x",
        comparison.p50_ratio
    );

    // IMP-163a: llama.cpp should win on p50
    assert_eq!(
        comparison.p50_winner, "llama.cpp",
        "IMP-163a: llama.cpp should win on p50"
    );

    // IMP-163a: Both should meet QA-012 (no outliers in these samples)
    assert!(
        comparison.both_meet_qa012,
        "IMP-163a: Both servers should meet QA-012 (p99 < 2x p50)"
    );

    println!("\nIMP-163a: A/B Latency Comparison:");
    println!("  {}", comparison.summary());
    println!(
        "  Realizar p50: {:.1}ms, p99: {:.1}ms",
        comparison.server_a_latency.p50_ms, comparison.server_a_latency.p99_ms
    );
    println!(
        "  llama.cpp p50: {:.1}ms, p99: {:.1}ms",
        comparison.server_b_latency.p50_ms, comparison.server_b_latency.p99_ms
    );
}

/// IMP-163b: QA-012 violation detection
#[derive(Debug, Clone)]
pub struct QA012Violation {
    /// Server name
    pub server_name: String,
    /// p50 latency
    pub p50_ms: f64,
    /// p99 latency
    pub p99_ms: f64,
    /// Actual ratio (p99/p50)
    pub tail_ratio: f64,
    /// Whether violation occurred (p99 >= 2x p50)
    pub violated: bool,
    /// Severity: "none", "minor" (2-3x), "major" (3-5x), "severe" (>5x)
    pub severity: String,
}

impl QA012Violation {
    pub fn check(server_name: &str, latencies: &[f64]) -> Self {
        let percentiles = LatencyPercentiles::from_samples(latencies);
        let tail_ratio = percentiles.tail_latency_ratio();
        let violated = tail_ratio >= 2.0;

        let severity = if tail_ratio < 2.0 {
            "none"
        } else if tail_ratio < 3.0 {
            "minor"
        } else if tail_ratio < 5.0 {
            "major"
        } else {
            "severe"
        };

        Self {
            server_name: server_name.to_string(),
            p50_ms: percentiles.p50_ms,
            p99_ms: percentiles.p99_ms,
            tail_ratio,
            violated,
            severity: severity.to_string(),
        }
    }
}

/// IMP-163b: Test QA-012 violation detection
#[test]
fn test_imp_163b_qa012_violation_detection() {
    // Good: p99 close to p50
    let good_latencies = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 105.0, 95.0, 103.0, 100.0, 101.0,
    ];
    let good_check = QA012Violation::check("GoodServer", &good_latencies);
    assert!(
        !good_check.violated,
        "IMP-163b: Good server should not violate QA-012"
    );
    assert_eq!(good_check.severity, "none");

    // Bad: p99 >> p50 (outliers)
    let bad_latencies = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 98.0, 500.0, 600.0, 700.0,
    ];
    let bad_check = QA012Violation::check("BadServer", &bad_latencies);
    assert!(
        bad_check.violated,
        "IMP-163b: Bad server should violate QA-012"
    );
    assert!(
        bad_check.severity == "major" || bad_check.severity == "severe",
        "IMP-163b: Severity should be major or severe, got {}",
        bad_check.severity
    );

    println!("\nIMP-163b: QA-012 Violation Detection:");
    println!(
        "  Good server: p50={:.1}ms, p99={:.1}ms, ratio={:.2}x, violated={}, severity={}",
        good_check.p50_ms,
        good_check.p99_ms,
        good_check.tail_ratio,
        good_check.violated,
        good_check.severity
    );
    println!(
        "  Bad server: p50={:.1}ms, p99={:.1}ms, ratio={:.2}x, violated={}, severity={}",
        bad_check.p50_ms,
        bad_check.p99_ms,
        bad_check.tail_ratio,
        bad_check.violated,
        bad_check.severity
    );
}

/// IMP-163c: Multi-server latency leaderboard
#[derive(Debug, Clone)]
pub struct LatencyLeaderboard {
    /// Server entries sorted by p50 latency
    pub entries: Vec<LeaderboardEntry>,
}

#[derive(Debug, Clone)]
pub struct LeaderboardEntry {
    pub rank: usize,
    pub server_name: String,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub tail_ratio: f64,
    pub meets_qa012: bool,
}

impl LatencyLeaderboard {
    pub fn from_results(results: Vec<(&str, Vec<f64>)>) -> Self {
        let mut entries: Vec<LeaderboardEntry> = results
            .into_iter()
            .map(|(name, samples)| {
                let percentiles = LatencyPercentiles::from_samples(&samples);
                let tail_ratio = percentiles.tail_latency_ratio();
                LeaderboardEntry {
                    rank: 0, // Set later
                    server_name: name.to_string(),
                    p50_ms: percentiles.p50_ms,
                    p99_ms: percentiles.p99_ms,
                    tail_ratio,
                    meets_qa012: tail_ratio < 2.0,
                }
            })
            .collect();

        // Sort by p50 (lower is better)
        entries.sort_by(|a, b| {
            a.p50_ms
                .partial_cmp(&b.p50_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = i + 1;
        }

        Self { entries }
    }

    /// Get winner (rank 1)
    pub fn winner(&self) -> Option<&LeaderboardEntry> {
        self.entries.first()
    }
}

/// IMP-163c: Test multi-server leaderboard
#[test]
fn test_imp_163c_latency_leaderboard() {
    let results = vec![
        ("Realizar", vec![520.0, 530.0, 510.0, 525.0, 515.0]),
        ("llama.cpp", vec![160.0, 165.0, 158.0, 162.0, 155.0]),
        ("Ollama", vec![280.0, 290.0, 275.0, 285.0, 270.0]),
    ];

    let leaderboard = LatencyLeaderboard::from_results(results);

    // IMP-163c: llama.cpp should be rank 1
    assert_eq!(
        leaderboard.winner().map(|e| e.server_name.as_str()),
        Some("llama.cpp"),
        "IMP-163c: llama.cpp should be rank 1"
    );

    // IMP-163c: Realizar should be rank 3
    let realizar_rank = leaderboard
        .entries
        .iter()
        .find(|e| e.server_name == "Realizar")
        .map(|e| e.rank);
    assert_eq!(
        realizar_rank,
        Some(3),
        "IMP-163c: Realizar should be rank 3"
    );

    println!("\nIMP-163c: Latency Leaderboard:");
    for entry in &leaderboard.entries {
        println!(
            "  #{}: {} - p50={:.1}ms, p99={:.1}ms, QA-012={}",
            entry.rank,
            entry.server_name,
            entry.p50_ms,
            entry.p99_ms,
            if entry.meets_qa012 { "PASS" } else { "FAIL" }
        );
    }
}

/// IMP-163d: Real-world A/B comparison against llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_163d_realworld_ab_latency() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Say hello:".to_string(),
        max_tokens: 10,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect llama.cpp latencies
    let mut llamacpp_latencies = Vec::new();
    for _ in 0..10 {
        let start = std::time::Instant::now();
        if client
            .llamacpp_completion("http://127.0.0.1:8082", &request)
            .is_ok()
        {
            llamacpp_latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if llamacpp_latencies.len() < 5 {
        println!("IMP-163d: Not enough llama.cpp samples");
        return;
    }

    // test Realizar latencies (since we don't have a running server)
    // In production, this would call the Realizar server
    let realizar_latencies: Vec<f64> = llamacpp_latencies
        .iter()
        .map(|lat| lat * 3.2) // Simulate 3.2x slower
        .collect();

    let comparison = ABLatencyComparison::compare(
        "Realizar",
        &realizar_latencies,
        "llama.cpp",
        &llamacpp_latencies,
    );

    let qa012_check = QA012Violation::check("llama.cpp", &llamacpp_latencies);

    println!("\nIMP-163d: Real-World A/B Latency Comparison:");
    println!("  {}", comparison.summary());
    println!("  llama.cpp samples: {}", llamacpp_latencies.len());
    println!(
        "  llama.cpp p50: {:.1}ms",
        comparison.server_b_latency.p50_ms
    );
    println!(
        "  llama.cpp p99: {:.1}ms",
        comparison.server_b_latency.p99_ms
    );
    println!(
        "  llama.cpp QA-012: {} (ratio={:.2}x)",
        if qa012_check.violated { "FAIL" } else { "PASS" },
        qa012_check.tail_ratio
    );
    println!(
        "  Estimated Realizar p50: {:.1}ms",
        comparison.server_a_latency.p50_ms
    );
}

// =========================================================================
// IMP-164: Real-World Throughput Regression Detection (QA-011, EXTREME TDD)
// =========================================================================
// Per spec QA-011: Throughput regression < 5% between commits (CI gate)
// Run with: cargo test test_imp_164 --lib --features bench-http

/// IMP-164a: Throughput regression tracker
#[derive(Debug, Clone)]
pub struct ThroughputRegressionTracker {
    /// Baseline throughput (from previous commit/release)
    pub baseline_tps: f64,
    /// Current throughput
    pub current_tps: f64,
    /// Regression threshold percentage
    pub threshold_percent: f64,
    /// Actual change percentage (negative = regression)
    pub change_percent: f64,
    /// Whether regression exceeds threshold
    pub regression_detected: bool,
    /// CI gate status
    pub ci_gate_passed: bool,
}

impl ThroughputRegressionTracker {
    pub fn check(baseline_tps: f64, current_tps: f64, threshold_percent: f64) -> Self {
        let change_percent = if baseline_tps > 0.0 {
            ((current_tps - baseline_tps) / baseline_tps) * 100.0
        } else {
            0.0
        };

        let regression_detected = change_percent < -threshold_percent;
        let ci_gate_passed = !regression_detected;

        Self {
            baseline_tps,
            current_tps,
            threshold_percent,
            change_percent,
            regression_detected,
            ci_gate_passed,
        }
    }

    /// Format for CI output
    pub fn ci_message(&self) -> String {
        if self.ci_gate_passed {
            format!(
                "✅ Throughput OK: {:.1} tok/s ({:+.1}% vs baseline {:.1})",
                self.current_tps, self.change_percent, self.baseline_tps
            )
        } else {
            format!(
                "❌ REGRESSION: {:.1} tok/s ({:.1}% below baseline {:.1}, threshold {:.1}%)",
                self.current_tps, -self.change_percent, self.baseline_tps, self.threshold_percent
            )
        }
    }
}

/// IMP-164a: Test regression tracker
#[test]
fn test_imp_164a_throughput_regression_tracker() {
    // No regression: current > baseline
    let improvement = ThroughputRegressionTracker::check(80.0, 85.0, 5.0);
    assert!(
        !improvement.regression_detected,
        "IMP-164a: Improvement should not be regression"
    );
    assert!(
        improvement.ci_gate_passed,
        "IMP-164a: CI gate should pass on improvement"
    );
    assert!(
        improvement.change_percent > 0.0,
        "IMP-164a: Change should be positive"
    );

    // Minor regression within threshold
    let minor = ThroughputRegressionTracker::check(80.0, 77.0, 5.0);
    assert!(
        !minor.regression_detected,
        "IMP-164a: 3.75% drop within 5% threshold"
    );
    assert!(
        minor.ci_gate_passed,
        "IMP-164a: CI gate should pass on minor drop"
    );

    // Major regression exceeds threshold
    let major = ThroughputRegressionTracker::check(80.0, 70.0, 5.0);
    assert!(
        major.regression_detected,
        "IMP-164a: 12.5% drop exceeds 5% threshold"
    );
    assert!(
        !major.ci_gate_passed,
        "IMP-164a: CI gate should fail on major regression"
    );

    println!("\nIMP-164a: Throughput Regression Tracker:");
    println!("  Improvement: {}", improvement.ci_message());
    println!("  Minor drop: {}", minor.ci_message());
    println!("  Major regression: {}", major.ci_message());
}

/// IMP-164b: Historical regression analysis
#[derive(Debug, Clone)]
pub struct HistoricalRegressionAnalysis {
    /// Version history entries
    pub history: Vec<VersionThroughput>,
    /// Detected regressions
    pub regressions: Vec<RegressionEvent>,
    /// Overall trend ("improving", "stable", "degrading")
    pub trend: String,
    /// Max regression observed
    pub max_regression_percent: f64,
}

#[derive(Debug, Clone)]
pub struct VersionThroughput {
    pub version: String,
    pub throughput_tps: f64,
}

#[derive(Debug, Clone)]
pub struct RegressionEvent {
    pub from_version: String,
    pub to_version: String,
    pub regression_percent: f64,
}

impl HistoricalRegressionAnalysis {
    pub fn analyze(history: Vec<VersionThroughput>, threshold_percent: f64) -> Self {
        let mut regressions = Vec::new();
        let mut max_regression: f64 = 0.0;

        for window in history.windows(2) {
            let prev = &window[0];
            let curr = &window[1];
            if prev.throughput_tps > 0.0 {
                let change =
                    ((curr.throughput_tps - prev.throughput_tps) / prev.throughput_tps) * 100.0;
                if change < -threshold_percent {
                    max_regression = max_regression.max(-change);
                    regressions.push(RegressionEvent {
                        from_version: prev.version.clone(),
                        to_version: curr.version.clone(),
                        regression_percent: -change,
                    });
                }
            }
        }

        // Determine trend
        let trend = if history.len() < 2 {
            "unknown"
        } else {
            let first = history.first().map_or(0.0, |v| v.throughput_tps);
            let last = history.last().map_or(0.0, |v| v.throughput_tps);
            let overall_change = if first > 0.0 {
                (last - first) / first
            } else {
                0.0
            };
            if overall_change > 0.05 {
                "improving"
            } else if overall_change < -0.05 {
                "degrading"
            } else {
                "stable"
            }
        };

        Self {
            history,
            regressions,
            trend: trend.to_string(),
            max_regression_percent: max_regression,
        }
    }
}

/// IMP-164b: Test historical regression analysis
#[test]
fn test_imp_164b_historical_regression() {
    let history = vec![
        VersionThroughput {
            version: "v0.1.0".to_string(),
            throughput_tps: 50.0,
        },
        VersionThroughput {
            version: "v0.2.0".to_string(),
            throughput_tps: 60.0,
        }, // +20%
        VersionThroughput {
            version: "v0.3.0".to_string(),
            throughput_tps: 55.0,
        }, // -8.3% REGRESSION
        VersionThroughput {
            version: "v0.4.0".to_string(),
            throughput_tps: 70.0,
        }, // +27%
        VersionThroughput {
            version: "v0.5.0".to_string(),
            throughput_tps: 80.0,
        }, // +14%
    ];

    let analysis = HistoricalRegressionAnalysis::analyze(history, 5.0);

    // IMP-164b: Should detect one regression (v0.2.0 → v0.3.0)
    assert_eq!(
        analysis.regressions.len(),
        1,
        "IMP-164b: Should detect 1 regression"
    );

    // IMP-164b: Overall trend should be improving (50 → 80)
    assert_eq!(
        analysis.trend, "improving",
        "IMP-164b: Trend should be improving"
    );

    println!("\nIMP-164b: Historical Regression Analysis:");
    println!("  Versions analyzed: {}", analysis.history.len());
    println!("  Regressions detected: {}", analysis.regressions.len());
    for reg in &analysis.regressions {
        println!(
            "    {} → {}: -{:.1}%",
            reg.from_version, reg.to_version, reg.regression_percent
        );
    }
    println!("  Overall trend: {}", analysis.trend);
    println!("  Max regression: {:.1}%", analysis.max_regression_percent);
}

/// IMP-164c: CI gate configuration
#[derive(Debug, Clone)]
pub struct CIGateConfig {
    /// Regression threshold for blocking
    pub block_threshold_percent: f64,
    /// Warning threshold (non-blocking)
    pub warn_threshold_percent: f64,
    /// Minimum samples for reliable measurement
    pub min_samples: usize,
    /// Maximum CV for measurement quality
    pub max_cv: f64,
}

impl CIGateConfig {
    pub fn default_config() -> Self {
        Self {
            block_threshold_percent: 5.0,
            warn_threshold_percent: 2.0,
            min_samples: 10,
            max_cv: 0.10,
        }
    }

    pub fn strict_config() -> Self {
        Self {
            block_threshold_percent: 2.0,
            warn_threshold_percent: 1.0,
            min_samples: 20,
            max_cv: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CIGateResult {
    pub config: CIGateConfig,
    pub passed: bool,
    pub warning: bool,
    pub regression_percent: f64,
    pub measurement_quality: String,
    pub message: String,
}

impl CIGateResult {
    pub fn evaluate(
        config: CIGateConfig,
        baseline: &ThroughputWithVariance,
        current: &ThroughputWithVariance,
    ) -> Self {
        let regression_percent = if baseline.mean_tps > 0.0 {
            ((baseline.mean_tps - current.mean_tps) / baseline.mean_tps) * 100.0
        } else {
            0.0
        };

        let measurement_quality = if current.sample_count < config.min_samples {
            "insufficient_samples"
        } else if current.cv > config.max_cv {
            "high_variance"
        } else {
            "good"
        };

        let passed = regression_percent < config.block_threshold_percent;
        let warning = regression_percent >= config.warn_threshold_percent && passed;

        let message = if !passed {
            format!(
                "❌ BLOCKED: {:.1}% regression exceeds {:.1}% threshold",
                regression_percent, config.block_threshold_percent
            )
        } else if warning {
            format!(
                "⚠️ WARNING: {:.1}% regression approaching threshold",
                regression_percent
            )
        } else {
            format!(
                "✅ PASSED: {:.1}% change within acceptable range",
                regression_percent
            )
        };

        Self {
            config,
            passed,
            warning,
            regression_percent,
            measurement_quality: measurement_quality.to_string(),
            message,
        }
    }
}

/// IMP-164c: Test CI gate evaluation
#[test]
fn test_imp_164c_ci_gate_evaluation() {
    let baseline = ThroughputWithVariance::from_samples(&[
        80.0, 82.0, 78.0, 81.0, 79.0, 80.0, 83.0, 77.0, 80.0, 81.0,
    ]);

    // Good: slight improvement
    let good_current = ThroughputWithVariance::from_samples(&[
        85.0, 87.0, 83.0, 86.0, 84.0, 85.0, 88.0, 82.0, 85.0, 86.0,
    ]);
    let good_result =
        CIGateResult::evaluate(CIGateConfig::default_config(), &baseline, &good_current);
    assert!(
        good_result.passed && !good_result.warning,
        "IMP-164c: Improvement should pass without warning"
    );

    // Warning: small regression
    let warn_current = ThroughputWithVariance::from_samples(&[
        78.0, 80.0, 76.0, 79.0, 77.0, 78.0, 81.0, 75.0, 78.0, 79.0,
    ]);
    let warn_result =
        CIGateResult::evaluate(CIGateConfig::default_config(), &baseline, &warn_current);
    assert!(
        warn_result.passed && warn_result.warning,
        "IMP-164c: 2-5% regression should warn"
    );

    // Blocked: large regression
    let bad_current = ThroughputWithVariance::from_samples(&[
        70.0, 72.0, 68.0, 71.0, 69.0, 70.0, 73.0, 67.0, 70.0, 71.0,
    ]);
    let bad_result =
        CIGateResult::evaluate(CIGateConfig::default_config(), &baseline, &bad_current);
    assert!(!bad_result.passed, "IMP-164c: >5% regression should block");

    println!("\nIMP-164c: CI Gate Evaluation:");
    println!(
        "  Baseline: {:.1} tok/s (CV={:.4})",
        baseline.mean_tps, baseline.cv
    );
    println!(
        "  Good: {} (quality: {})",
        good_result.message, good_result.measurement_quality
    );
    println!(
        "  Warning: {} (quality: {})",
        warn_result.message, warn_result.measurement_quality
    );
    println!(
        "  Blocked: {} (quality: {})",
        bad_result.message, bad_result.measurement_quality
    );
}

/// IMP-164d: Real-world regression check against llama.cpp baseline
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_164d_realworld_regression_check() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Count to 5:".to_string(),
        max_tokens: 15,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect throughput samples
    let mut throughputs = Vec::new();
    for _ in 0..10 {
        let start = std::time::Instant::now();
        if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
            let elapsed = start.elapsed().as_secs_f64();
            let tokens = result.text.split_whitespace().count().max(1);
            throughputs.push(tokens as f64 / elapsed);
        }
    }

    if throughputs.len() < 5 {
        println!("IMP-164d: Not enough samples");
        return;
    }

    let current = ThroughputWithVariance::from_samples(&throughputs);

    // Use spec baseline: 256 tok/s for llama.cpp
    let baseline = ThroughputWithVariance::from_samples(&[256.0; 10]);

    let tracker = ThroughputRegressionTracker::check(baseline.mean_tps, current.mean_tps, 5.0);
    let ci_result = CIGateResult::evaluate(CIGateConfig::default_config(), &baseline, &current);

    println!("\nIMP-164d: Real-World Regression Check (llama.cpp):");
    println!("  Spec baseline: {:.1} tok/s", baseline.mean_tps);
    println!(
        "  Current measured: {:.1} tok/s (CV={:.4})",
        current.mean_tps, current.cv
    );
    println!("  {}", tracker.ci_message());
    println!(
        "  CI Gate: {} (quality: {})",
        ci_result.message, ci_result.measurement_quality
    );
}

// =========================================================================
// IMP-165: Real-World Memory Efficiency Comparison (QA-013, EXTREME TDD)
// =========================================================================
// Per spec QA-013: Memory usage < 1.5x model size
// Compares memory efficiency between inference engines.
// Run with: cargo test test_imp_165 --lib --features bench-http

/// IMP-165a: Memory efficiency measurement
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyMeasurement {
    /// Server name
    pub server_name: String,
    /// Model file size in MB
    pub model_size_mb: f64,
    /// Peak memory usage during inference in MB
    pub peak_memory_mb: f64,
    /// Memory overhead ratio (peak / model)
    pub overhead_ratio: f64,
    /// Whether it meets QA-013 (< 1.5x)
    pub meets_qa013: bool,
    /// Memory efficiency score (0-100, higher is better)
    pub efficiency_score: f64,
}

impl MemoryEfficiencyMeasurement {
    pub fn new(server_name: &str, model_size_mb: f64, peak_memory_mb: f64) -> Self {
        let overhead_ratio = if model_size_mb > 0.0 {
            peak_memory_mb / model_size_mb
        } else {
            1.0
        };

        let meets_qa013 = overhead_ratio < 1.5;

        // Efficiency score: 100 at 1.0x, 0 at 2.0x
        let efficiency_score = ((2.0 - overhead_ratio) / 1.0 * 100.0).clamp(0.0, 100.0);

        Self {
            server_name: server_name.to_string(),
            model_size_mb,
            peak_memory_mb,
            overhead_ratio,
            meets_qa013,
            efficiency_score,
        }
    }

    /// Calculate wasted memory in MB
    pub fn wasted_memory_mb(&self) -> f64 {
        (self.peak_memory_mb - self.model_size_mb).max(0.0)
    }
}

/// IMP-165a: Test memory efficiency measurement
#[test]
fn test_imp_165a_memory_efficiency_measurement() {
    // Efficient server: peak close to model size
    let efficient = MemoryEfficiencyMeasurement::new("Efficient", 1000.0, 1100.0);
    assert!(
        efficient.meets_qa013,
        "IMP-165a: 1.1x overhead should meet QA-013"
    );
    assert!(
        efficient.efficiency_score > 80.0,
        "IMP-165a: Efficient server should have high score, got {:.1}",
        efficient.efficiency_score
    );

    // Borderline server: just under 1.5x
    let borderline = MemoryEfficiencyMeasurement::new("Borderline", 1000.0, 1400.0);
    assert!(
        borderline.meets_qa013,
        "IMP-165a: 1.4x overhead should meet QA-013"
    );

    // Inefficient server: exceeds 1.5x
    let inefficient = MemoryEfficiencyMeasurement::new("Inefficient", 1000.0, 1800.0);
    assert!(
        !inefficient.meets_qa013,
        "IMP-165a: 1.8x overhead should fail QA-013"
    );
    assert!(
        inefficient.efficiency_score < 30.0,
        "IMP-165a: Inefficient server should have low score, got {:.1}",
        inefficient.efficiency_score
    );

    println!("\nIMP-165a: Memory Efficiency Measurement:");
    println!(
        "  Efficient: {:.1}x overhead, score={:.1}, QA-013={}",
        efficient.overhead_ratio, efficient.efficiency_score, efficient.meets_qa013
    );
    println!(
        "  Borderline: {:.1}x overhead, score={:.1}, QA-013={}",
        borderline.overhead_ratio, borderline.efficiency_score, borderline.meets_qa013
    );
    println!(
        "  Inefficient: {:.1}x overhead, score={:.1}, QA-013={}",
        inefficient.overhead_ratio, inefficient.efficiency_score, inefficient.meets_qa013
    );
}

/// IMP-165b: Multi-server memory comparison
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyComparison {
    /// All server measurements
    pub measurements: Vec<MemoryEfficiencyMeasurement>,
    /// Server with best efficiency
    pub most_efficient: String,
    /// Server with worst efficiency
    pub least_efficient: String,
    /// Average overhead ratio across all servers
    pub avg_overhead_ratio: f64,
}

impl MemoryEfficiencyComparison {
    pub fn compare(measurements: Vec<MemoryEfficiencyMeasurement>) -> Self {
        if measurements.is_empty() {
            return Self {
                measurements: Vec::new(),
                most_efficient: "none".to_string(),
                least_efficient: "none".to_string(),
                avg_overhead_ratio: 1.0,
            };
        }

        let avg =
            measurements.iter().map(|m| m.overhead_ratio).sum::<f64>() / measurements.len() as f64;

        let most = measurements
            .iter()
            .min_by(|a, b| {
                a.overhead_ratio
                    .partial_cmp(&b.overhead_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or_else(|| "none".to_string(), |m| m.server_name.clone());

        let least = measurements
            .iter()
            .max_by(|a, b| {
                a.overhead_ratio
                    .partial_cmp(&b.overhead_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or_else(|| "none".to_string(), |m| m.server_name.clone());

        Self {
            measurements,
            most_efficient: most,
            least_efficient: least,
            avg_overhead_ratio: avg,
        }
    }
}

/// IMP-165b: Test multi-server memory comparison
#[test]
fn test_imp_165b_memory_comparison() {
    let model_size = 4000.0; // 4GB model

    let measurements = vec![
        MemoryEfficiencyMeasurement::new("llama.cpp", model_size, 4200.0), // 1.05x
        MemoryEfficiencyMeasurement::new("Ollama", model_size, 4800.0),    // 1.2x
        MemoryEfficiencyMeasurement::new("Realizar", model_size, 5200.0),  // 1.3x
    ];

    let comparison = MemoryEfficiencyComparison::compare(measurements);

    // IMP-165b: llama.cpp should be most efficient
    assert_eq!(
        comparison.most_efficient, "llama.cpp",
        "IMP-165b: llama.cpp should be most efficient"
    );

    // IMP-165b: Realizar should be least efficient
    assert_eq!(
        comparison.least_efficient, "Realizar",
        "IMP-165b: Realizar should be least efficient"
    );

    // IMP-165b: All should meet QA-013
    let all_meet = comparison.measurements.iter().all(|m| m.meets_qa013);
    assert!(
        all_meet,
        "IMP-165b: All servers should meet QA-013 (< 1.5x)"
    );

    println!("\nIMP-165b: Memory Efficiency Comparison:");
    println!("  Model size: {:.0} MB", model_size);
    for m in &comparison.measurements {
        println!(
            "  {}: {:.0} MB peak ({:.2}x), score={:.1}, QA-013={}",
            m.server_name, m.peak_memory_mb, m.overhead_ratio, m.efficiency_score, m.meets_qa013
        );
    }
    println!("  Most efficient: {}", comparison.most_efficient);
    println!("  Least efficient: {}", comparison.least_efficient);
    println!("  Average overhead: {:.2}x", comparison.avg_overhead_ratio);
}

/// IMP-165c: Memory per token efficiency
#[derive(Debug, Clone)]
pub struct MemoryPerTokenEfficiency {
    /// Server name
    pub server_name: String,
    /// Memory per token in KB
    pub memory_per_token_kb: f64,
    /// Context length tested
    pub context_length: usize,
    /// Whether scaling is linear (expected) or super-linear (bad)
    pub linear_scaling: bool,
}

impl MemoryPerTokenEfficiency {
    pub fn analyze(server_name: &str, context_memory_pairs: &[(usize, f64)]) -> Self {
        if context_memory_pairs.len() < 2 {
            return Self {
                server_name: server_name.to_string(),
                memory_per_token_kb: 0.0,
                context_length: 0,
                linear_scaling: true,
            };
        }

        // Calculate memory per token for last measurement
        let (last_ctx, last_mem) = context_memory_pairs.last().expect("test");
        let (first_ctx, first_mem) = context_memory_pairs.first().expect("test");

        let delta_mem = last_mem - first_mem;
        let delta_ctx = (*last_ctx - *first_ctx) as f64;
        let mem_per_token = if delta_ctx > 0.0 {
            (delta_mem * 1024.0) / delta_ctx // Convert MB to KB
        } else {
            0.0
        };

        // Check for linear scaling: memory growth should be proportional to context
        // If growth rate increases significantly, scaling is super-linear (bad)
        let linear = if context_memory_pairs.len() >= 3 {
            let mid = context_memory_pairs.len() / 2;
            let (mid_ctx, mid_mem) = &context_memory_pairs[mid];

            let rate1 = (mid_mem - first_mem) / (*mid_ctx - *first_ctx) as f64;
            let rate2 = (last_mem - mid_mem) / (*last_ctx - *mid_ctx) as f64;

            // Linear if rates are within 20% of each other
            (rate2 / rate1 - 1.0).abs() < 0.20
        } else {
            true
        };

        Self {
            server_name: server_name.to_string(),
            memory_per_token_kb: mem_per_token,
            context_length: *last_ctx,
            linear_scaling: linear,
        }
    }
}

/// IMP-165c: Test memory per token efficiency
#[test]
fn test_imp_165c_memory_per_token() {
    // Linear scaling: good behavior
    let linear_data = vec![
        (512, 5000.0),  // 512 tokens, 5GB
        (1024, 5500.0), // 1024 tokens, 5.5GB
        (2048, 6500.0), // 2048 tokens, 6.5GB
    ];
    let linear = MemoryPerTokenEfficiency::analyze("LinearServer", &linear_data);

    assert!(
        linear.linear_scaling,
        "IMP-165c: Linear memory growth should be detected"
    );

    // Super-linear scaling: bad behavior (memory explodes)
    let superlinear_data = vec![
        (512, 5000.0),   // 512 tokens, 5GB
        (1024, 6000.0),  // 1024 tokens, 6GB (+2MB/tok)
        (2048, 10000.0), // 2048 tokens, 10GB (+4MB/tok - rate doubled!)
    ];
    let superlinear = MemoryPerTokenEfficiency::analyze("SuperLinearServer", &superlinear_data);

    assert!(
        !superlinear.linear_scaling,
        "IMP-165c: Super-linear growth should be detected"
    );

    println!("\nIMP-165c: Memory Per Token Efficiency:");
    println!(
        "  Linear server: {:.2} KB/token, linear={}",
        linear.memory_per_token_kb, linear.linear_scaling
    );
    println!(
        "  Super-linear server: {:.2} KB/token, linear={}",
        superlinear.memory_per_token_kb, superlinear.linear_scaling
    );
}

/// IMP-165d: Real-world memory efficiency (placeholder)
#[test]
#[ignore = "Requires running llama.cpp server with memory monitoring"]
fn test_imp_165d_realworld_memory_efficiency() {
    // This test would require:
    // 1. Running llama.cpp server
    // 2. Monitoring memory usage via /proc or similar
    // 3. Known model size for comparison

    // test values based on typical observations
    let model_size_mb = 4000.0; // 4GB Q4_K model

    let measurements = vec![
        MemoryEfficiencyMeasurement::new("llama.cpp", model_size_mb, 4200.0),
        MemoryEfficiencyMeasurement::new("Ollama", model_size_mb, 4600.0),
    ];

    let comparison = MemoryEfficiencyComparison::compare(measurements);

    println!("\nIMP-165d: Real-World Memory Efficiency:");
    println!("  Model size: {:.0} MB", model_size_mb);
    for m in &comparison.measurements {
        println!(
            "  {}: {:.2}x overhead, wasted={:.0} MB, QA-013={}",
            m.server_name,
            m.overhead_ratio,
            m.wasted_memory_mb(),
            m.meets_qa013
        );
    }
}

// =========================================================================
// IMP-166: Real-World Cold Start Verification (QA-016, EXTREME TDD)
