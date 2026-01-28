//! Statistical measurement and analysis for benchmarking
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - BENCH-004: MeasurementProtocol
//! - BENCH-005: LatencyStatistics
//! - BENCH-006: Outlier Detection (MAD-based)
//! - BENCH-007: Regression Detection
//! - BENCH-008: Welch's t-test for Statistical Significance

#![allow(clippy::cast_precision_loss)]

use std::time::Duration;

// ============================================================================
// BENCH-004: MeasurementProtocol (following SPEC-BENCH-001)
// ============================================================================

/// Complete measurement protocol for benchmarking
///
/// Follows MLPerf™ Inference benchmarking principles for scientific rigor.
#[derive(Debug, Clone)]
pub struct MeasurementProtocol {
    /// Number of latency samples to collect
    pub latency_samples: usize,
    /// Percentiles to compute (e.g., 50, 90, 95, 99, 99.9)
    pub latency_percentiles: Vec<f64>,
    /// Duration for throughput measurement
    pub throughput_duration: Duration,
    /// Ramp-up time before throughput measurement
    pub throughput_ramp_up: Duration,
    /// Number of memory samples to collect
    pub memory_samples: usize,
    /// Interval between memory samples
    pub memory_interval: Duration,
}

impl Default for MeasurementProtocol {
    fn default() -> Self {
        Self {
            latency_samples: 100,
            latency_percentiles: vec![50.0, 90.0, 95.0, 99.0, 99.9],
            throughput_duration: Duration::from_secs(60),
            throughput_ramp_up: Duration::from_secs(10),
            memory_samples: 10,
            memory_interval: Duration::from_secs(1),
        }
    }
}

impl MeasurementProtocol {
    /// Create a new measurement protocol with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of latency samples
    #[must_use]
    pub fn with_latency_samples(mut self, samples: usize) -> Self {
        self.latency_samples = samples;
        self
    }

    /// Set the percentiles to compute
    #[must_use]
    pub fn with_percentiles(mut self, percentiles: Vec<f64>) -> Self {
        self.latency_percentiles = percentiles;
        self
    }

    /// Set the throughput measurement duration
    #[must_use]
    pub fn with_throughput_duration(mut self, duration: Duration) -> Self {
        self.throughput_duration = duration;
        self
    }

    /// Set the number of memory samples
    #[must_use]
    pub fn with_memory_samples(mut self, samples: usize) -> Self {
        self.memory_samples = samples;
        self
    }
}

// ============================================================================
// BENCH-005: LatencyStatistics (following SPEC-BENCH-001 Section 7.1)
// ============================================================================

/// Comprehensive latency statistics following MLPerf™ reporting standards
#[derive(Debug, Clone)]
pub struct LatencyStatistics {
    /// Mean latency
    pub mean: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum latency
    pub min: Duration,
    /// Maximum latency
    pub max: Duration,
    /// 50th percentile (median)
    pub p50: Duration,
    /// 90th percentile
    pub p90: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// 99.9th percentile (tail latency)
    pub p999: Duration,
    /// Number of samples
    pub samples: usize,
    /// 95% confidence interval (lower, upper)
    pub confidence_interval_95: (Duration, Duration),
}

impl LatencyStatistics {
    /// Compute statistics from a slice of duration samples
    ///
    /// # Panics
    /// Panics if samples is empty
    #[must_use]
    pub fn from_samples(samples: &[Duration]) -> Self {
        assert!(!samples.is_empty(), "samples must not be empty");

        let n = samples.len();
        let n_f64 = n as f64;

        // Compute mean
        let sum_nanos: u128 = samples.iter().map(Duration::as_nanos).sum();
        let mean_nanos = sum_nanos / n as u128;
        let mean = Duration::from_nanos(mean_nanos as u64);

        // Compute standard deviation
        let variance: f64 = samples
            .iter()
            .map(|s| {
                let diff = s.as_nanos() as f64 - mean_nanos as f64;
                diff * diff
            })
            .sum::<f64>()
            / (n_f64 - 1.0).max(1.0);
        let std_dev_nanos = variance.sqrt();
        let std_dev = Duration::from_nanos(std_dev_nanos as u64);

        // Sort for percentile computation
        let mut sorted: Vec<Duration> = samples.to_vec();
        sorted.sort();

        // Min/max
        let min = sorted[0];
        let max = sorted[n - 1];

        // Percentiles using nearest-rank method
        let percentile = |p: f64| -> Duration {
            let idx = ((p / 100.0) * n_f64).ceil() as usize;
            sorted[idx.saturating_sub(1).min(n - 1)]
        };

        let p50 = percentile(50.0);
        let p90 = percentile(90.0);
        let p95 = percentile(95.0);
        let p99 = percentile(99.0);
        let p999 = percentile(99.9);

        // 95% confidence interval using t-distribution approximation
        // For large n, t ≈ 1.96
        let t_value = if n >= 30 { 1.96 } else { 2.0 + 4.0 / n_f64 };
        let margin = std_dev_nanos * t_value / n_f64.sqrt();
        let lower = Duration::from_nanos((mean_nanos as f64 - margin).max(0.0) as u64);
        let upper = Duration::from_nanos((mean_nanos as f64 + margin) as u64);

        Self {
            mean,
            std_dev,
            min,
            max,
            p50,
            p90,
            p95,
            p99,
            p999,
            samples: n,
            confidence_interval_95: (lower, upper),
        }
    }
}

// ============================================================================
// BENCH-006: Outlier Detection (MAD-based)
// ============================================================================

/// Detect outliers using Median Absolute Deviation (MAD) method
///
/// More robust than standard deviation for non-normal distributions.
/// Uses the modified Z-score method with configurable threshold.
///
/// # Arguments
/// * `samples` - Slice of f64 samples
/// * `threshold` - Modified Z-score threshold (typically 3.5 for strict, 2.0 for lenient)
///
/// # Returns
/// Vector of indices that are considered outliers
pub fn detect_outliers(samples: &[f64], threshold: f64) -> Vec<usize> {
    if samples.len() < 3 {
        return Vec::new();
    }

    // Calculate median
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len().is_multiple_of(2) {
        f64::midpoint(sorted[sorted.len() / 2 - 1], sorted[sorted.len() / 2])
    } else {
        sorted[sorted.len() / 2]
    };

    // Calculate MAD (Median Absolute Deviation)
    let mut deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if deviations.len().is_multiple_of(2) {
        f64::midpoint(
            deviations[deviations.len() / 2 - 1],
            deviations[deviations.len() / 2],
        )
    } else {
        deviations[deviations.len() / 2]
    };

    // Avoid division by zero
    if mad < f64::EPSILON {
        return Vec::new();
    }

    // Constant for normal distribution approximation
    let k = 1.4826;

    // Find outliers using modified Z-score
    samples
        .iter()
        .enumerate()
        .filter(|(_, &x)| {
            let modified_z = (x - median) / (k * mad);
            modified_z.abs() > threshold
        })
        .map(|(i, _)| i)
        .collect()
}

// ============================================================================
// BENCH-007: Regression Detection
// ============================================================================

/// Single benchmark metric for comparison
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    /// Metric name
    pub name: String,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Number of samples
    pub samples: usize,
}

/// Individual regression item
#[derive(Debug, Clone)]
pub struct Regression {
    /// Metric that regressed
    pub metric: String,
    /// Baseline value
    pub baseline: f64,
    /// Current value
    pub current: f64,
    /// Percentage change
    pub change_percent: f64,
}

/// Report from regression analysis
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Metrics that exceeded failure threshold
    pub regressions: Vec<Regression>,
    /// Metrics that exceeded warning threshold
    pub warnings: Vec<Regression>,
    /// Metrics that improved significantly
    pub improvements: Vec<Regression>,
    /// Overall pass/fail (no regressions)
    pub passed: bool,
}

/// Performance regression detector
///
/// Compares baseline and current benchmark results to detect
/// performance regressions, warnings, and improvements.
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    /// Warning threshold (default: 2%)
    pub warning_threshold: f64,
    /// Failure threshold (default: 5%)
    pub failure_threshold: f64,
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self {
            warning_threshold: 0.02, // 2%
            failure_threshold: 0.05, // 5%
        }
    }
}

impl RegressionDetector {
    /// Compare baseline and current metrics
    pub fn compare(
        &self,
        baseline: &BenchmarkMetrics,
        current: &BenchmarkMetrics,
    ) -> RegressionReport {
        let mut regressions = Vec::new();
        let mut warnings = Vec::new();
        let mut improvements = Vec::new();

        // Calculate percentage change (positive = regression for latency-like metrics)
        let change = (current.mean - baseline.mean) / baseline.mean;

        let item = Regression {
            metric: baseline.name.clone(),
            baseline: baseline.mean,
            current: current.mean,
            change_percent: change * 100.0,
        };

        if change > self.failure_threshold {
            regressions.push(item);
        } else if change > self.warning_threshold {
            warnings.push(item);
        } else if change < -self.warning_threshold {
            improvements.push(item);
        }

        RegressionReport {
            passed: regressions.is_empty(),
            regressions,
            warnings,
            improvements,
        }
    }
}

// ============================================================================
// BENCH-008: Welch's t-test for Statistical Significance
// Per Hoefler & Belli [17], statistical testing is required for valid comparisons
// ============================================================================

/// Result of Welch's t-test for comparing two sample means
#[derive(Debug, Clone)]
pub struct WelchTTestResult {
    /// Calculated t-statistic
    pub t_statistic: f64,
    /// Welch-Satterthwaite degrees of freedom
    pub degrees_of_freedom: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Whether the difference is statistically significant at given alpha
    pub significant: bool,
}

/// Perform Welch's t-test to compare two sample means
///
/// Welch's t-test is used when samples may have unequal variances.
/// Returns statistical significance information.
///
/// # Arguments
/// * `sample_a` - First sample
/// * `sample_b` - Second sample
/// * `alpha` - Significance level (e.g., 0.05 for 95% confidence)
///
/// # Example
/// ```
/// use realizar::bench::welch_t_test;
///
/// let a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
/// let b = vec![20.0, 21.0, 20.5, 20.2, 20.8];
/// let result = welch_t_test(&a, &b, 0.05);
/// assert!(result.significant); // Clearly different means
/// ```
pub fn welch_t_test(sample_a: &[f64], sample_b: &[f64], alpha: f64) -> WelchTTestResult {
    let n1 = sample_a.len() as f64;
    let n2 = sample_b.len() as f64;

    // Calculate means
    let mean1 = sample_a.iter().sum::<f64>() / n1;
    let mean2 = sample_b.iter().sum::<f64>() / n2;

    // Calculate sample variances (using n-1 for unbiased estimator)
    let var1 = if n1 > 1.0 {
        sample_a.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0)
    } else {
        0.0
    };
    let var2 = if n2 > 1.0 {
        sample_b.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0)
    } else {
        0.0
    };

    // Handle zero variance case
    let se1 = var1 / n1;
    let se2 = var2 / n2;
    let se_diff = (se1 + se2).sqrt();

    if se_diff < f64::EPSILON {
        // Both samples have zero variance - cannot compute t-statistic
        return WelchTTestResult {
            t_statistic: 0.0,
            degrees_of_freedom: n1 + n2 - 2.0,
            p_value: 1.0,
            significant: false,
        };
    }

    // Calculate t-statistic
    let t_stat = (mean1 - mean2) / se_diff;

    // Welch-Satterthwaite degrees of freedom
    let df_num = (se1 + se2).powi(2);
    let df_denom = if n1 > 1.0 && se1 > f64::EPSILON {
        se1.powi(2) / (n1 - 1.0)
    } else {
        0.0
    } + if n2 > 1.0 && se2 > f64::EPSILON {
        se2.powi(2) / (n2 - 1.0)
    } else {
        0.0
    };

    let df = if df_denom > f64::EPSILON {
        df_num / df_denom
    } else {
        n1 + n2 - 2.0
    };

    // Approximate p-value using normal distribution for large df
    // For small df, we use a more conservative approximation
    let p_value = approximate_t_pvalue(t_stat.abs(), df);

    WelchTTestResult {
        t_statistic: t_stat,
        degrees_of_freedom: df,
        p_value,
        significant: p_value < alpha,
    }
}

/// Approximate two-tailed p-value from t-distribution
///
/// Uses normal approximation for large df, conservative approximation for small df
fn approximate_t_pvalue(t_abs: f64, df: f64) -> f64 {
    // For very large df, use normal approximation
    if df > 100.0 {
        // Use error function approximation for normal CDF
        let z = t_abs;
        let p = erfc_approx(z / std::f64::consts::SQRT_2);
        return p;
    }

    // For smaller df, use a polynomial approximation of t-distribution CDF
    // Based on Abramowitz and Stegun approximation
    let ratio = df / (df + t_abs * t_abs);
    incomplete_beta_approx(ratio, df / 2.0, 0.5)
}

/// Approximate complementary error function
fn erfc_approx(x: f64) -> f64 {
    // Horner form coefficients for erfc approximation
    // From Abramowitz and Stegun, formula 7.1.26
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    if sign < 0.0 {
        2.0 - y
    } else {
        y
    }
}

/// Approximate incomplete beta function (simplified for t-test)
fn incomplete_beta_approx(x: f64, a: f64, b: f64) -> f64 {
    // Use continued fraction expansion for better accuracy
    // Simplified approximation suitable for t-distribution p-values
    if x < (a + 1.0) / (a + b + 2.0) {
        let beta_factor =
            gamma_ln(a + b) - gamma_ln(a) - gamma_ln(b) + a * x.ln() + b * (1.0 - x).ln();
        let beta_factor = beta_factor.exp();
        beta_factor * cf_beta(x, a, b) / a
    } else {
        1.0 - incomplete_beta_approx(1.0 - x, b, a)
    }
}

/// Continued fraction for incomplete beta
#[allow(clippy::many_single_char_names)] // Standard math notation for beta function
fn cf_beta(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;
    let tiny = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Approximate log-gamma function (Stirling's approximation)
#[allow(clippy::excessive_precision)] // Lanczos coefficients require high precision
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Lanczos approximation coefficients
    let g = 7.0;
    let c = [
        0.999_999_999_999_81,
        676.520_368_121_885,
        -1_259.139_216_722_403,
        771.323_428_777_653,
        -176.615_029_162_141,
        12.507_343_278_687,
        -0.138_571_095_265_72,
        9.984_369_578_02e-6,
        1.505_632_735_15e-7,
    ];

    let x = x - 1.0;
    let mut sum = c[0];
    for (i, &coef) in c.iter().enumerate().skip(1) {
        sum += coef / (x + i as f64);
    }

    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

// ============================================================================
// Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MeasurementProtocol Tests
    // =========================================================================

    #[test]
    fn test_measurement_protocol_default() {
        let protocol = MeasurementProtocol::default();
        assert_eq!(protocol.latency_samples, 100);
        assert_eq!(protocol.latency_percentiles, vec![50.0, 90.0, 95.0, 99.0, 99.9]);
        assert_eq!(protocol.throughput_duration, Duration::from_secs(60));
        assert_eq!(protocol.throughput_ramp_up, Duration::from_secs(10));
        assert_eq!(protocol.memory_samples, 10);
        assert_eq!(protocol.memory_interval, Duration::from_secs(1));
    }

    #[test]
    fn test_measurement_protocol_new() {
        let protocol = MeasurementProtocol::new();
        assert_eq!(protocol.latency_samples, 100);
    }

    #[test]
    fn test_measurement_protocol_with_latency_samples() {
        let protocol = MeasurementProtocol::new().with_latency_samples(200);
        assert_eq!(protocol.latency_samples, 200);
    }

    #[test]
    fn test_measurement_protocol_with_percentiles() {
        let protocol = MeasurementProtocol::new().with_percentiles(vec![50.0, 99.0]);
        assert_eq!(protocol.latency_percentiles, vec![50.0, 99.0]);
    }

    #[test]
    fn test_measurement_protocol_with_throughput_duration() {
        let protocol = MeasurementProtocol::new()
            .with_throughput_duration(Duration::from_secs(120));
        assert_eq!(protocol.throughput_duration, Duration::from_secs(120));
    }

    #[test]
    fn test_measurement_protocol_with_memory_samples() {
        let protocol = MeasurementProtocol::new().with_memory_samples(20);
        assert_eq!(protocol.memory_samples, 20);
    }

    #[test]
    fn test_measurement_protocol_builder_chain() {
        let protocol = MeasurementProtocol::new()
            .with_latency_samples(50)
            .with_percentiles(vec![90.0, 99.0])
            .with_throughput_duration(Duration::from_secs(30))
            .with_memory_samples(5);

        assert_eq!(protocol.latency_samples, 50);
        assert_eq!(protocol.latency_percentiles, vec![90.0, 99.0]);
        assert_eq!(protocol.throughput_duration, Duration::from_secs(30));
        assert_eq!(protocol.memory_samples, 5);
    }

    // =========================================================================
    // LatencyStatistics Tests
    // =========================================================================

    #[test]
    fn test_latency_statistics_from_samples_uniform() {
        let samples: Vec<Duration> = (1..=10)
            .map(|i| Duration::from_millis(i * 10))
            .collect();

        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.samples, 10);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(100));
        // Mean = (10+20+...+100)/10 = 550/10 = 55ms
        assert_eq!(stats.mean.as_millis(), 55);
    }

    #[test]
    fn test_latency_statistics_from_samples_single() {
        let samples = vec![Duration::from_millis(100)];
        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.samples, 1);
        assert_eq!(stats.min, Duration::from_millis(100));
        assert_eq!(stats.max, Duration::from_millis(100));
        assert_eq!(stats.mean, Duration::from_millis(100));
        assert_eq!(stats.p50, Duration::from_millis(100));
    }

    #[test]
    fn test_latency_statistics_percentiles() {
        // 100 samples from 1ms to 100ms
        let samples: Vec<Duration> = (1..=100)
            .map(|i| Duration::from_millis(i))
            .collect();

        let stats = LatencyStatistics::from_samples(&samples);

        // p50 should be around 50ms
        assert!(stats.p50.as_millis() >= 49 && stats.p50.as_millis() <= 51);
        // p90 should be around 90ms
        assert!(stats.p90.as_millis() >= 89 && stats.p90.as_millis() <= 91);
        // p99 should be around 99ms
        assert!(stats.p99.as_millis() >= 98 && stats.p99.as_millis() <= 100);
    }

    #[test]
    #[should_panic(expected = "samples must not be empty")]
    fn test_latency_statistics_empty_samples_panics() {
        let samples: Vec<Duration> = vec![];
        let _ = LatencyStatistics::from_samples(&samples);
    }

    // =========================================================================
    // RegressionDetector Tests
    // =========================================================================

    #[test]
    fn test_regression_detector_default() {
        let detector = RegressionDetector::default();
        assert!((detector.warning_threshold - 0.02).abs() < f64::EPSILON);
        assert!((detector.failure_threshold - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_regression_detector_clone() {
        let detector = RegressionDetector::default();
        let cloned = detector.clone();
        assert!((detector.warning_threshold - cloned.warning_threshold).abs() < f64::EPSILON);
        assert!((detector.failure_threshold - cloned.failure_threshold).abs() < f64::EPSILON);
    }

    // =========================================================================
    // WelchTTestResult Tests
    // =========================================================================

    #[test]
    fn test_welch_t_test_result_fields() {
        // Create a result directly
        let result = WelchTTestResult {
            t_statistic: 2.5,
            degrees_of_freedom: 18.0,
            p_value: 0.02,
            significant: true,
        };
        assert!((result.t_statistic - 2.5).abs() < f64::EPSILON);
        assert!((result.degrees_of_freedom - 18.0).abs() < f64::EPSILON);
        assert!((result.p_value - 0.02).abs() < f64::EPSILON);
        assert!(result.significant);
    }

    #[test]
    fn test_welch_t_test_same_samples() {
        // Same distributions should have high p-value (not significant)
        let a: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();
        let b: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();

        let result = welch_t_test(&a, &b, 0.05);
        // p-value should be > 0.05 (not significant)
        assert!(result.p_value > 0.05);
        assert!(!result.significant);
    }

    #[test]
    fn test_welch_t_test_different_samples() {
        // Clearly different distributions with some variance
        let a: Vec<f64> = (1..=30).map(|i| 100.0 + (i as f64 % 5.0)).collect();
        let b: Vec<f64> = (1..=30).map(|i| 200.0 + (i as f64 % 5.0)).collect();

        let result = welch_t_test(&a, &b, 0.05);
        // Should be significant difference (100 unit difference with small variance)
        assert!(result.significant);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_welch_t_test_different_alpha() {
        // Same samples, test with different alpha values
        let a: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();
        let b: Vec<f64> = (1..=20).map(|i| 110.0 + i as f64).collect();

        let result_05 = welch_t_test(&a, &b, 0.05);
        let result_01 = welch_t_test(&a, &b, 0.01);

        // p-value should be the same, but significance may differ
        assert!((result_05.p_value - result_01.p_value).abs() < 0.001);
    }

    // =========================================================================
    // Statistical Function Tests
    // =========================================================================

    #[test]
    fn test_gamma_ln_positive() {
        // gamma(1) = 1, ln(1) = 0
        let result = gamma_ln(1.0);
        assert!(result.abs() < 0.01);

        // gamma(2) = 1, ln(1) = 0
        let result = gamma_ln(2.0);
        assert!(result.abs() < 0.01);

        // gamma(5) = 24, ln(24) ≈ 3.178
        let result = gamma_ln(5.0);
        assert!((result - 3.178).abs() < 0.1);
    }

    #[test]
    fn test_gamma_ln_negative_or_zero() {
        assert!(gamma_ln(0.0).is_infinite());
        assert!(gamma_ln(-1.0).is_infinite());
    }
}
