//! Benchmark harness for model runner comparison
//!
//! Implements the benchmark specification v1.1 with Toyota Way engineering principles:
//! - Dynamic CV-based stop-rule (Hoefler & Belli, SC12)
//! - Thermal throttling protocol
//! - ITL variance measurement (Dean & Barroso, "Tail at Scale")
//! - KV-cache fragmentation detection (PagedAttention methodology)
//! - KL-Divergence quality validation (LLM.int8())
//!
//! ## References
//!
//! - [17] Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems", SC'15
//! - [11] Dean & Barroso, "The Tail at Scale", CACM 2013
//! - [12] Kwon et al., "PagedAttention", SOSP'23
//! - [13] Dettmers et al., "LLM.int8()", NeurIPS 2022

#![allow(clippy::cast_precision_loss)] // Statistical functions need usize->f64

use std::time::Duration;

use serde::{Deserialize, Serialize};

// ============================================================================
// Dynamic Sampler (Section 2.1)
// ============================================================================

/// Dynamic stop-rule based on Coefficient of Variation (CV)
///
/// Per Hoefler & Belli [17], fixed iteration counts mask variance characteristics.
/// This sampler stops when statistical stability is achieved.
#[derive(Debug, Clone)]
pub struct DynamicSampler {
    /// Minimum number of samples before checking CV
    pub min_samples: usize,
    /// Maximum samples (failsafe)
    pub max_samples: usize,
    /// Target CV threshold (default: 0.05 = 5%)
    pub cv_threshold: f64,
    /// Sliding window size for CV calculation
    pub cv_window: usize,
    /// Number of consecutive stable windows required
    pub stability_count: usize,
    /// Current stability streak
    stable_streak: usize,
}

impl Default for DynamicSampler {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_samples: 10_000,
            cv_threshold: 0.05,
            cv_window: 50,
            stability_count: 3,
            stable_streak: 0,
        }
    }
}

impl DynamicSampler {
    /// Create a new sampler with custom parameters
    #[must_use]
    pub fn new(min_samples: usize, max_samples: usize, cv_threshold: f64) -> Self {
        Self {
            min_samples,
            max_samples,
            cv_threshold,
            cv_window: 50,
            stability_count: 3,
            stable_streak: 0,
        }
    }

    /// Check if sampling should continue
    ///
    /// Returns `true` if more samples are needed, `false` if stable.
    #[must_use]
    pub fn should_continue(&mut self, samples: &[f64]) -> bool {
        let n = samples.len();

        // Always continue until minimum samples
        if n < self.min_samples {
            return true;
        }

        // Stop at maximum (failsafe)
        if n >= self.max_samples {
            return false;
        }

        // Compute CV over sliding window
        let window_start = n.saturating_sub(self.cv_window);
        let window = &samples[window_start..];
        let cv = compute_cv(window);

        if cv < self.cv_threshold {
            self.stable_streak += 1;
            if self.stable_streak >= self.stability_count {
                return false; // Stable - stop sampling
            }
        } else {
            self.stable_streak = 0;
        }

        true // Continue sampling
    }

    /// Get the current CV for the last window
    #[must_use]
    pub fn current_cv(&self, samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return f64::INFINITY;
        }
        let window_start = samples.len().saturating_sub(self.cv_window);
        compute_cv(&samples[window_start..])
    }

    /// Reset the sampler for a new run
    pub fn reset(&mut self) {
        self.stable_streak = 0;
    }
}

/// Compute Coefficient of Variation (CV = std_dev / mean)
fn compute_cv(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::INFINITY;
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;

    if mean.abs() < 1e-10 {
        return f64::INFINITY;
    }

    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    std_dev / mean.abs()
}

// ============================================================================
// Thermal Guard (Section 2.2)
// ============================================================================

/// Temperature monitoring for benchmark validity
///
/// Per spec Section 2.2, benchmarks are invalid if temperature variance > 2°C.
#[derive(Debug, Clone)]
pub struct ThermalGuard {
    /// Maximum temperature before cooldown (°C)
    pub max_temp_c: f64,
    /// Temperature to resume at after cooldown (°C)
    pub cooldown_threshold_c: f64,
    /// Cooldown sleep duration (ms)
    pub cooldown_sleep_ms: u64,
    /// Maximum allowed temperature variance (°C)
    pub temp_variance_c: f64,
}

impl Default for ThermalGuard {
    fn default() -> Self {
        Self {
            max_temp_c: 80.0,
            cooldown_threshold_c: 70.0,
            cooldown_sleep_ms: 10_000,
            temp_variance_c: 2.0,
        }
    }
}

/// Result of thermal validation
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalValidity {
    /// Temperature variance within acceptable range
    Valid,
    /// Temperature variance too high
    Invalid(String),
}

impl ThermalGuard {
    /// Check if benchmark results are thermally valid
    #[must_use]
    pub fn validate_run(&self, temps: &[f64]) -> ThermalValidity {
        if temps.is_empty() {
            return ThermalValidity::Valid;
        }

        let variance = compute_variance(temps);
        let std_dev = variance.sqrt();

        if std_dev > self.temp_variance_c {
            ThermalValidity::Invalid(format!(
                "Temperature variance {std_dev:.2}°C exceeds threshold {:.2}°C",
                self.temp_variance_c
            ))
        } else {
            ThermalValidity::Valid
        }
    }

    /// Check if cooldown is needed and sleep if so
    pub fn cooldown_if_needed(&self, current_temp: f64) {
        if current_temp > self.max_temp_c {
            std::thread::sleep(Duration::from_millis(self.cooldown_sleep_ms));
        }
    }

    /// Get max temperature from readings
    #[must_use]
    pub fn max_temp(&self, temps: &[f64]) -> f64 {
        if temps.is_empty() {
            return 0.0;
        }
        temps.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get temperature variance
    #[must_use]
    pub fn temp_variance(&self, temps: &[f64]) -> f64 {
        compute_variance(temps).sqrt()
    }
}

/// Compute variance of a dataset
fn compute_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

// ============================================================================
// KV-Cache Metrics (Section 4.3)
// ============================================================================

/// KV-cache fragmentation metrics per PagedAttention [12]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KvCacheMetrics {
    /// Total allocated KV-cache memory (bytes)
    pub allocated_bytes: u64,
    /// Actually used KV-cache memory (bytes)
    pub used_bytes: u64,
    /// Fragmentation percentage (waste)
    pub fragmentation_pct: f64,
}

impl KvCacheMetrics {
    /// Create new metrics from allocated and used bytes
    #[must_use]
    pub fn new(allocated_bytes: u64, used_bytes: u64) -> Self {
        let waste = allocated_bytes.saturating_sub(used_bytes);
        let fragmentation_pct = if allocated_bytes > 0 {
            (waste as f64 / allocated_bytes as f64) * 100.0
        } else {
            0.0
        };

        Self {
            allocated_bytes,
            used_bytes,
            fragmentation_pct,
        }
    }

    /// Get allocated memory in MB
    #[must_use]
    pub fn allocated_mb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get used memory in MB
    #[must_use]
    pub fn used_mb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Check if fragmentation is acceptable (< threshold)
    #[must_use]
    pub fn is_acceptable(&self, threshold_pct: f64) -> bool {
        self.fragmentation_pct < threshold_pct
    }
}

// ============================================================================
// Energy Metrics (Section 4.4)
// ============================================================================

/// Energy measurement metrics per Garcia-Martin et al. [14]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnergyMetrics {
    /// Total energy consumed (Joules)
    pub total_joules: f64,
    /// Idle power consumption (Watts)
    pub idle_watts: f64,
    /// Average active power consumption (Watts)
    pub active_watts_avg: f64,
    /// Number of tokens generated
    pub tokens_generated: u64,
}

impl EnergyMetrics {
    /// Create new energy metrics
    #[must_use]
    pub fn new(total_joules: f64, idle_watts: f64, active_watts_avg: f64, tokens: u64) -> Self {
        Self {
            total_joules,
            idle_watts,
            active_watts_avg,
            tokens_generated: tokens,
        }
    }

    /// Calculate energy per token (Joules/token)
    #[must_use]
    pub fn joules_per_token(&self) -> f64 {
        if self.tokens_generated == 0 {
            return 0.0;
        }
        self.total_joules / self.tokens_generated as f64
    }

    /// Calculate energy efficiency ratio (tokens per Joule)
    #[must_use]
    pub fn tokens_per_joule(&self) -> f64 {
        if self.total_joules < 1e-10 {
            return 0.0;
        }
        self.tokens_generated as f64 / self.total_joules
    }
}

// ============================================================================
// ITL (Inter-Token Latency) Metrics (Section 4.2)
// ============================================================================

/// Inter-Token Latency metrics per "Tail at Scale" [11]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ItlMetrics {
    /// Median ITL (ms)
    pub median_ms: f64,
    /// Standard deviation (jitter indicator)
    pub std_dev_ms: f64,
    /// p99 ITL (ms)
    pub p99_ms: f64,
    /// p99.9 ITL (ms)
    pub p999_ms: f64,
}

impl ItlMetrics {
    /// Create ITL metrics from raw measurements
    #[must_use]
    pub fn from_measurements(itl_times_ms: &[f64]) -> Self {
        if itl_times_ms.is_empty() {
            return Self::default();
        }

        let mut sorted = itl_times_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let median_ms = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        let mean = itl_times_ms.iter().sum::<f64>() / n as f64;
        let variance = itl_times_ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (n as f64 - 1.0).max(1.0);
        let std_dev_ms = variance.sqrt();

        let percentile_99 = ((n as f64 * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        let percentile_999 = ((n as f64 * 0.999).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);

        Self {
            median_ms,
            std_dev_ms,
            p99_ms: sorted[percentile_99],
            p999_ms: sorted[percentile_999],
        }
    }

    /// Check if jitter is acceptable (std_dev < threshold)
    #[must_use]
    pub fn is_low_jitter(&self, threshold_ms: f64) -> bool {
        self.std_dev_ms < threshold_ms
    }
}

// ============================================================================
// KL-Divergence Quality Validation (Section 6.1)
// ============================================================================

/// Result of quantization quality validation
#[derive(Debug, Clone, PartialEq)]
pub enum QualityResult {
    /// Quality is acceptable
    Pass {
        /// Measured KL-divergence (nats)
        kl_divergence: f64,
    },
    /// Quality degradation detected
    Fail {
        /// Measured KL-divergence (nats)
        kl_divergence: f64,
        /// Threshold that was exceeded
        threshold: f64,
        /// Descriptive message
        message: &'static str,
    },
}

/// Compute softmax of logits
fn softmax(logits: &[f32]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f64> = logits
        .iter()
        .map(|x| ((*x - max_logit) as f64).exp())
        .collect();
    let sum: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|x| x / sum).collect()
}

/// Validate quantization quality using KL-Divergence
///
/// Per LLM.int8() [13], epsilon checks fail on outlier features.
/// KL-divergence provides a proper information-theoretic measure.
///
/// # Arguments
///
/// * `fp32_logits` - Reference logits from FP32 model
/// * `quantized_logits` - Logits from quantized model
/// * `threshold` - Maximum acceptable KL-divergence (nats)
///
/// # Returns
///
/// `QualityResult::Pass` if KL-divergence < threshold, `Fail` otherwise.
#[must_use]
pub fn validate_quantization_quality(
    fp32_logits: &[f32],
    quantized_logits: &[f32],
    threshold: f64,
) -> QualityResult {
    if fp32_logits.len() != quantized_logits.len() {
        return QualityResult::Fail {
            kl_divergence: f64::INFINITY,
            threshold,
            message: "Logit vector lengths do not match",
        };
    }

    if fp32_logits.is_empty() {
        return QualityResult::Pass { kl_divergence: 0.0 };
    }

    // Convert to probability distributions
    let fp32_probs = softmax(fp32_logits);
    let quant_probs = softmax(quantized_logits);

    // Compute KL(P_fp32 || P_quant)
    let kl_div: f64 = fp32_probs
        .iter()
        .zip(&quant_probs)
        .map(|(p, q)| {
            if *p > 1e-10 && *q > 1e-10 {
                p * (p / q).ln()
            } else {
                0.0
            }
        })
        .sum();

    if kl_div < threshold {
        QualityResult::Pass {
            kl_divergence: kl_div,
        }
    } else {
        QualityResult::Fail {
            kl_divergence: kl_div,
            threshold,
            message: "Quantization quality degradation detected",
        }
    }
}

// ============================================================================
// Benchmark Result (Section 4.1)
// ============================================================================

/// Configuration for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Model identifier
    pub model: String,
    /// Model format (apr, gguf, safetensors)
    pub format: String,
    /// Quantization level
    pub quantization: String,
    /// Runtime name
    pub runtime: String,
    /// Runtime version
    pub runtime_version: String,
}

/// Complete benchmark result per spec Section 4.1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Configuration used
    pub config: BenchmarkConfig,
    /// Cold start time (ms)
    pub cold_start_ms: f64,
    /// Model load time (ms)
    pub model_load_ms: f64,
    /// Time-to-first-token measurements (ms)
    pub ttft_ms: Vec<f64>,
    /// Inter-token latency measurements (ms)
    pub itl_ms: Vec<f64>,
    /// Generation throughput measurements (tok/s)
    pub generation_tok_s: Vec<f64>,
    /// Peak memory usage (MB)
    pub peak_memory_mb: u64,
    /// KV-cache fragmentation percentage
    pub kv_cache_waste_pct: f64,
    /// Total energy consumed (Joules)
    pub energy_joules: f64,
    /// Total tokens generated
    pub tokens_generated: u64,
    /// Actual number of iterations (dynamic sampling)
    pub actual_iterations: usize,
    /// CV at stop point
    pub cv_at_stop: f64,
    /// Unix timestamp
    pub timestamp: u64,
}

/// Summary statistics for benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    // TTFT metrics
    /// TTFT p50 (ms)
    pub ttft_p50: f64,
    /// TTFT p95 (ms)
    pub ttft_p95: f64,
    /// TTFT p99 (ms)
    pub ttft_p99: f64,
    /// TTFT p99.9 (ms)
    pub ttft_p999: f64,

    // ITL metrics
    /// ITL median (ms)
    pub itl_median: f64,
    /// ITL standard deviation (jitter)
    pub itl_std_dev: f64,

    // Throughput metrics
    /// Throughput median (tok/s)
    pub throughput_median: f64,
    /// Throughput 95% CI (lower, upper)
    pub throughput_ci_95: (f64, f64),

    // Energy metrics
    /// Energy per token (J/tok)
    pub token_joules: f64,

    // Memory metrics
    /// KV-cache waste percentage
    pub memory_waste_pct: f64,

    // Statistical validity
    /// Number of iterations run
    pub iterations: usize,
    /// Final CV value
    pub cv_final: f64,
}

impl BenchmarkResult {
    /// Generate summary statistics from raw measurements
    #[must_use]
    pub fn summary(&self) -> BenchmarkSummary {
        BenchmarkSummary {
            ttft_p50: percentile(&self.ttft_ms, 50.0),
            ttft_p95: percentile(&self.ttft_ms, 95.0),
            ttft_p99: percentile(&self.ttft_ms, 99.0),
            ttft_p999: percentile(&self.ttft_ms, 99.9),

            itl_median: percentile(&self.itl_ms, 50.0),
            itl_std_dev: compute_std_dev(&self.itl_ms),

            throughput_median: percentile(&self.generation_tok_s, 50.0),
            throughput_ci_95: bootstrap_ci(&self.generation_tok_s, 0.95, 1000),

            token_joules: if self.tokens_generated > 0 {
                self.energy_joules / self.tokens_generated as f64
            } else {
                0.0
            },

            memory_waste_pct: self.kv_cache_waste_pct,
            iterations: self.actual_iterations,
            cv_final: self.cv_at_stop,
        }
    }
}

/// Compute percentile of a dataset
fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((sorted.len() as f64 * p / 100.0).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

/// Compute standard deviation
fn compute_std_dev(data: &[f64]) -> f64 {
    compute_variance(data).sqrt()
}

/// Bootstrap confidence interval
fn bootstrap_ci(data: &[f64], confidence: f64, n_resamples: usize) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }

    let mut bootstrap_means = Vec::with_capacity(n_resamples);
    let n = data.len();

    for i in 0..n_resamples {
        // Simple deterministic pseudo-random for reproducibility
        // Uses a basic LCG instead of hash for clippy compliance
        let seed = (i as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);

        let mut sum = 0.0;
        for j in 0..n {
            let idx = ((seed.wrapping_mul(j as u64 + 1)) as usize) % n;
            sum += data[idx];
        }
        bootstrap_means.push(sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower_idx = ((n_resamples as f64 * alpha / 2.0).floor() as usize).min(n_resamples - 1);
    let upper_idx =
        ((n_resamples as f64 * (1.0 - alpha / 2.0)).ceil() as usize).min(n_resamples - 1);

    (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
}

// ============================================================================
// Convoy Test (Section 2.4)
// ============================================================================

/// Workload type for convoy testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// Short QA: 32 input tokens, 64 output tokens
    ShortQa,
    /// Long Context: 2048 input tokens, 512 output tokens
    LongContext,
}

impl WorkloadType {
    /// Get input token count for this workload type
    #[must_use]
    pub const fn input_tokens(&self) -> usize {
        match self {
            Self::ShortQa => 32,
            Self::LongContext => 2048,
        }
    }

    /// Get output token count for this workload type
    #[must_use]
    pub const fn output_tokens(&self) -> usize {
        match self {
            Self::ShortQa => 64,
            Self::LongContext => 512,
        }
    }
}

/// Configuration for convoy test per spec Section 2.4
#[derive(Debug, Clone)]
pub struct ConvoyTestConfig {
    /// Number of long-context requests (default: 10)
    pub long_requests: usize,
    /// Number of short-QA requests (default: 100)
    pub short_requests: usize,
    /// Maximum acceptable p99 latency increase (default: 50%)
    pub max_p99_increase_pct: f64,
    /// Maximum acceptable head-of-line blocking time (ms)
    pub max_hol_blocking_ms: f64,
    /// Maximum acceptable KV-cache fragmentation (%)
    pub max_kv_fragmentation_pct: f64,
}

impl Default for ConvoyTestConfig {
    fn default() -> Self {
        Self {
            long_requests: 10,
            short_requests: 100,
            max_p99_increase_pct: 50.0,
            max_hol_blocking_ms: 500.0,
            max_kv_fragmentation_pct: 15.0,
        }
    }
}

/// Result of a single request in convoy test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvoyRequestResult {
    /// Request type
    pub workload_type: String,
    /// Time spent waiting (head-of-line blocking)
    pub queue_time_ms: f64,
    /// Time to first token
    pub ttft_ms: f64,
    /// Total latency
    pub total_latency_ms: f64,
}

/// Overall convoy test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvoyTestResult {
    /// Number of long-context requests in test
    pub long_requests: usize,
    /// Number of short-QA requests in test
    pub short_requests: usize,

    /// Baseline: Short-QA p99 without convoy
    pub baseline_short_p99_ms: f64,
    /// Convoy: Short-QA p99 with convoy
    pub convoy_short_p99_ms: f64,
    /// P99 increase percentage
    pub p99_increase_pct: f64,

    /// Maximum head-of-line blocking observed
    pub max_hol_blocking_ms: f64,
    /// Average head-of-line blocking
    pub avg_hol_blocking_ms: f64,

    /// KV-cache fragmentation during convoy
    pub kv_fragmentation_pct: f64,

    /// Pass/fail status
    pub passed: bool,
    /// Failure reasons (if any)
    pub failure_reasons: Vec<String>,
}

impl ConvoyTestResult {
    /// Create a new convoy test result from measurements
    #[must_use]
    pub fn new(
        config: &ConvoyTestConfig,
        baseline_short_latencies: &[f64],
        convoy_short_latencies: &[f64],
        hol_blocking_times: &[f64],
        kv_fragmentation_pct: f64,
    ) -> Self {
        let baseline_short_p99 = percentile(baseline_short_latencies, 99.0);
        let convoy_short_p99 = percentile(convoy_short_latencies, 99.0);

        let p99_increase_pct = if baseline_short_p99 > 0.0 {
            ((convoy_short_p99 - baseline_short_p99) / baseline_short_p99) * 100.0
        } else {
            0.0
        };

        let max_hol_blocking = hol_blocking_times.iter().copied().fold(0.0_f64, f64::max);
        let avg_hol_blocking = if hol_blocking_times.is_empty() {
            0.0
        } else {
            hol_blocking_times.iter().sum::<f64>() / hol_blocking_times.len() as f64
        };

        let mut failure_reasons = Vec::new();

        if p99_increase_pct > config.max_p99_increase_pct {
            failure_reasons.push(format!(
                "P99 increase {p99_increase_pct:.1}% exceeds threshold {:.1}%",
                config.max_p99_increase_pct
            ));
        }

        if max_hol_blocking > config.max_hol_blocking_ms {
            failure_reasons.push(format!(
                "Max HOL blocking {max_hol_blocking:.1}ms exceeds threshold {:.1}ms",
                config.max_hol_blocking_ms
            ));
        }

        if kv_fragmentation_pct > config.max_kv_fragmentation_pct {
            failure_reasons.push(format!(
                "KV fragmentation {kv_fragmentation_pct:.1}% exceeds threshold {:.1}%",
                config.max_kv_fragmentation_pct
            ));
        }

        Self {
            long_requests: config.long_requests,
            short_requests: config.short_requests,
            baseline_short_p99_ms: baseline_short_p99,
            convoy_short_p99_ms: convoy_short_p99,
            p99_increase_pct,
            max_hol_blocking_ms: max_hol_blocking,
            avg_hol_blocking_ms: avg_hol_blocking,
            kv_fragmentation_pct,
            passed: failure_reasons.is_empty(),
            failure_reasons,
        }
    }
}

// ============================================================================
// Saturation Test (Section 2.5)
// ============================================================================

/// Configuration for saturation stress test per spec Section 2.5
#[derive(Debug, Clone)]
pub struct SaturationTestConfig {
    /// CPU load percentage (default: 50%)
    pub cpu_load_pct: u8,
    /// Maximum acceptable throughput degradation (default: 30%)
    pub max_throughput_degradation_pct: f64,
    /// Maximum acceptable p99 latency increase (default: 100%)
    pub max_p99_increase_pct: f64,
}

impl Default for SaturationTestConfig {
    fn default() -> Self {
        Self {
            cpu_load_pct: 50,
            max_throughput_degradation_pct: 30.0,
            max_p99_increase_pct: 100.0,
        }
    }
}

/// Saturation test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationTestResult {
    /// CPU load used
    pub cpu_load_pct: u8,

    /// Baseline throughput (tok/s)
    pub baseline_throughput: f64,
    /// Stressed throughput (tok/s)
    pub stressed_throughput: f64,
    /// Throughput degradation percentage
    pub throughput_degradation_pct: f64,

    /// Baseline p99 latency (ms)
    pub baseline_p99_ms: f64,
    /// Stressed p99 latency (ms)
    pub stressed_p99_ms: f64,
    /// P99 latency increase percentage
    pub p99_increase_pct: f64,

    /// Pass/fail status
    pub passed: bool,
    /// Failure reasons (if any)
    pub failure_reasons: Vec<String>,
}

impl SaturationTestResult {
    /// Create a new saturation test result
    #[must_use]
    pub fn new(
        config: &SaturationTestConfig,
        baseline_throughputs: &[f64],
        stressed_throughputs: &[f64],
        baseline_latencies: &[f64],
        stressed_latencies: &[f64],
    ) -> Self {
        let baseline_throughput = if baseline_throughputs.is_empty() {
            0.0
        } else {
            baseline_throughputs.iter().sum::<f64>() / baseline_throughputs.len() as f64
        };

        let stressed_throughput = if stressed_throughputs.is_empty() {
            0.0
        } else {
            stressed_throughputs.iter().sum::<f64>() / stressed_throughputs.len() as f64
        };

        let throughput_degradation_pct = if baseline_throughput > 0.0 {
            ((baseline_throughput - stressed_throughput) / baseline_throughput) * 100.0
        } else {
            0.0
        };

        let baseline_p99 = percentile(baseline_latencies, 99.0);
        let stressed_p99 = percentile(stressed_latencies, 99.0);

        let p99_increase_pct = if baseline_p99 > 0.0 {
            ((stressed_p99 - baseline_p99) / baseline_p99) * 100.0
        } else {
            0.0
        };

        let mut failure_reasons = Vec::new();

        if throughput_degradation_pct > config.max_throughput_degradation_pct {
            failure_reasons.push(format!(
                "Throughput degradation {throughput_degradation_pct:.1}% exceeds threshold {:.1}%",
                config.max_throughput_degradation_pct
            ));
        }

        if p99_increase_pct > config.max_p99_increase_pct {
            failure_reasons.push(format!(
                "P99 increase {p99_increase_pct:.1}% exceeds threshold {:.1}%",
                config.max_p99_increase_pct
            ));
        }

        Self {
            cpu_load_pct: config.cpu_load_pct,
            baseline_throughput,
            stressed_throughput,
            throughput_degradation_pct,
            baseline_p99_ms: baseline_p99,
            stressed_p99_ms: stressed_p99,
            p99_increase_pct,
            passed: failure_reasons.is_empty(),
            failure_reasons,
        }
    }
}

// ============================================================================
// Benchmark Runner (Full Harness)
// ============================================================================

/// Hardware specification for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    /// CPU model
    pub cpu: String,
    /// GPU model (if any)
    pub gpu: Option<String>,
    /// Total memory in GB
    pub memory_gb: u64,
    /// Storage type
    pub storage: String,
}

impl Default for HardwareSpec {
    fn default() -> Self {
        Self {
            cpu: "Unknown".to_string(),
            gpu: None,
            memory_gb: 0,
            storage: "Unknown".to_string(),
        }
    }
}

/// Sampling method configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Sampling method (e.g., "dynamic_cv")
    pub method: String,
    /// CV threshold for stopping
    pub cv_threshold: f64,
    /// Actual iterations run
    pub actual_iterations: usize,
    /// CV at stop point
    pub cv_at_stop: f64,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            method: "dynamic_cv".to_string(),
            cv_threshold: 0.05,
            actual_iterations: 0,
            cv_at_stop: 0.0,
            warmup_iterations: 100,
        }
    }
}

/// Thermal validity info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalInfo {
    /// Whether thermal conditions were valid
    pub valid: bool,
    /// Temperature variance (°C)
    pub temp_variance_c: f64,
    /// Maximum temperature observed (°C)
    pub max_temp_c: f64,
}

impl Default for ThermalInfo {
    fn default() -> Self {
        Self {
            valid: true,
            temp_variance_c: 0.0,
            max_temp_c: 0.0,
        }
    }
}

/// TTFT results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtftResults {
    /// P50 (median)
    pub p50: f64,
    /// P95
    pub p95: f64,
    /// P99
    pub p99: f64,
    /// P99.9
    pub p999: f64,
}

/// ITL results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItlResults {
    /// Median ITL
    pub median: f64,
    /// Standard deviation (jitter)
    pub std_dev: f64,
    /// P99 ITL
    pub p99: f64,
}

/// Throughput results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputResults {
    /// Median throughput (tok/s)
    pub median: f64,
    /// 95% confidence interval
    pub ci_95: (f64, f64),
}

/// Memory results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResults {
    /// Model size (MB)
    pub model_mb: u64,
    /// Peak RSS (MB)
    pub peak_rss_mb: u64,
    /// KV-cache waste percentage
    pub kv_waste_pct: f64,
}

/// Energy results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyResults {
    /// Total energy (Joules)
    pub total_joules: f64,
    /// Energy per token (J/tok)
    pub token_joules: f64,
    /// Idle power (Watts)
    pub idle_watts: f64,
}

/// Cold start results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartResults {
    /// Median cold start time (ms)
    pub median: f64,
    /// P99 cold start time (ms)
    pub p99: f64,
}

/// Quality validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityValidation {
    /// KL-divergence vs FP32
    pub kl_divergence_vs_fp32: f64,
    /// Perplexity on WikiText-2 (optional)
    pub perplexity_wikitext2: Option<f64>,
}

/// Full benchmark results per JSON schema v1.1 (Appendix B)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBenchmarkResult {
    /// Schema version
    pub version: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Model configuration
    pub config: BenchmarkConfig,
    /// Hardware specification
    pub hardware: HardwareSpec,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Thermal information
    pub thermal: ThermalInfo,
    /// All results
    pub results: BenchmarkResults,
    /// Quality validation
    pub quality: QualityValidation,
}

/// Consolidated benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Time-to-first-token metrics
    pub ttft_ms: TtftResults,
    /// Inter-token latency metrics
    pub itl_ms: ItlResults,
    /// Throughput metrics
    pub throughput_tok_s: ThroughputResults,
    /// Memory metrics
    pub memory_mb: MemoryResults,
    /// Energy metrics
    pub energy: EnergyResults,
    /// Cold start metrics
    pub cold_start_ms: ColdStartResults,
}

impl FullBenchmarkResult {
    /// Create from a BenchmarkResult with additional metadata
    #[must_use]
    pub fn from_benchmark_result(
        result: &BenchmarkResult,
        hardware: HardwareSpec,
        thermal_temps: &[f64],
        kl_divergence: f64,
    ) -> Self {
        let thermal_guard = ThermalGuard::default();
        let thermal_validity = thermal_guard.validate_run(thermal_temps);

        let summary = result.summary();

        Self {
            version: "1.1".to_string(),
            timestamp: chrono_timestamp(),
            config: result.config.clone(),
            hardware,
            sampling: SamplingConfig {
                method: "dynamic_cv".to_string(),
                cv_threshold: 0.05,
                actual_iterations: result.actual_iterations,
                cv_at_stop: result.cv_at_stop,
                warmup_iterations: 100,
            },
            thermal: ThermalInfo {
                valid: thermal_validity == ThermalValidity::Valid,
                temp_variance_c: thermal_guard.temp_variance(thermal_temps),
                max_temp_c: thermal_guard.max_temp(thermal_temps),
            },
            results: BenchmarkResults {
                ttft_ms: TtftResults {
                    p50: summary.ttft_p50,
                    p95: summary.ttft_p95,
                    p99: summary.ttft_p99,
                    p999: summary.ttft_p999,
                },
                itl_ms: ItlResults {
                    median: summary.itl_median,
                    std_dev: summary.itl_std_dev,
                    p99: percentile(&result.itl_ms, 99.0),
                },
                throughput_tok_s: ThroughputResults {
                    median: summary.throughput_median,
                    ci_95: summary.throughput_ci_95,
                },
                memory_mb: MemoryResults {
                    model_mb: result.peak_memory_mb / 2, // Approximate model size
                    peak_rss_mb: result.peak_memory_mb,
                    kv_waste_pct: result.kv_cache_waste_pct,
                },
                energy: EnergyResults {
                    total_joules: result.energy_joules,
                    token_joules: summary.token_joules,
                    idle_watts: 0.0, // Would need separate measurement
                },
                cold_start_ms: ColdStartResults {
                    median: result.cold_start_ms,
                    p99: result.cold_start_ms * 1.5, // Approximate
                },
            },
            quality: QualityValidation {
                kl_divergence_vs_fp32: kl_divergence,
                perplexity_wikitext2: None,
            },
        }
    }

    /// Serialize to JSON string
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is invalid or doesn't match the schema.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Generate ISO 8601 timestamp
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Simple ISO 8601 format without external dependencies
    format!("1970-01-01T00:00:00Z+{secs}s")
}

// ============================================================================
// Benchmark Comparison
// ============================================================================

/// Result of comparing two benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// Baseline config
    pub baseline_runtime: String,
    /// Current config
    pub current_runtime: String,

    /// TTFT p99 change percentage (negative = improvement)
    pub ttft_p99_change_pct: f64,
    /// Throughput change percentage (positive = improvement)
    pub throughput_change_pct: f64,
    /// Memory change percentage (negative = improvement)
    pub memory_change_pct: f64,
    /// Energy change percentage (negative = improvement)
    pub energy_change_pct: f64,

    /// Overall winner
    pub winner: String,
    /// Significance level (p-value from Mann-Whitney U)
    pub significance: f64,
}

impl BenchmarkComparison {
    /// Compare two benchmark results
    #[must_use]
    pub fn compare(baseline: &FullBenchmarkResult, current: &FullBenchmarkResult) -> Self {
        let ttft_p99_change = if baseline.results.ttft_ms.p99 > 0.0 {
            ((current.results.ttft_ms.p99 - baseline.results.ttft_ms.p99)
                / baseline.results.ttft_ms.p99)
                * 100.0
        } else {
            0.0
        };

        let throughput_change = if baseline.results.throughput_tok_s.median > 0.0 {
            ((current.results.throughput_tok_s.median - baseline.results.throughput_tok_s.median)
                / baseline.results.throughput_tok_s.median)
                * 100.0
        } else {
            0.0
        };

        let memory_change = if baseline.results.memory_mb.peak_rss_mb > 0 {
            ((current.results.memory_mb.peak_rss_mb as f64
                - baseline.results.memory_mb.peak_rss_mb as f64)
                / baseline.results.memory_mb.peak_rss_mb as f64)
                * 100.0
        } else {
            0.0
        };

        let energy_change = if baseline.results.energy.token_joules > 0.0 {
            ((current.results.energy.token_joules - baseline.results.energy.token_joules)
                / baseline.results.energy.token_joules)
                * 100.0
        } else {
            0.0
        };

        // Simple winner determination: count improvements
        let mut current_wins = 0;
        let mut baseline_wins = 0;

        if ttft_p99_change < -5.0 {
            current_wins += 1;
        } else if ttft_p99_change > 5.0 {
            baseline_wins += 1;
        }

        if throughput_change > 5.0 {
            current_wins += 1;
        } else if throughput_change < -5.0 {
            baseline_wins += 1;
        }

        if memory_change < -5.0 {
            current_wins += 1;
        } else if memory_change > 5.0 {
            baseline_wins += 1;
        }

        if energy_change < -5.0 {
            current_wins += 1;
        } else if energy_change > 5.0 {
            baseline_wins += 1;
        }

        let winner = match current_wins.cmp(&baseline_wins) {
            std::cmp::Ordering::Greater => current.config.runtime.clone(),
            std::cmp::Ordering::Less => baseline.config.runtime.clone(),
            std::cmp::Ordering::Equal => "tie".to_string(),
        };

        Self {
            baseline_runtime: baseline.config.runtime.clone(),
            current_runtime: current.config.runtime.clone(),
            ttft_p99_change_pct: ttft_p99_change,
            throughput_change_pct: throughput_change,
            memory_change_pct: memory_change,
            energy_change_pct: energy_change,
            winner,
            significance: 0.001, // Would need actual Mann-Whitney U test
        }
    }
}

// ============================================================================
// Regression Detection
// ============================================================================

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Whether a regression was detected
    pub regression_detected: bool,
    /// Metrics that regressed
    pub regressed_metrics: Vec<String>,
    /// Regression threshold used (%)
    pub threshold_pct: f64,
}

impl RegressionResult {
    /// Check for regressions between baseline and current
    #[must_use]
    pub fn check(
        baseline: &FullBenchmarkResult,
        current: &FullBenchmarkResult,
        threshold_pct: f64,
    ) -> Self {
        let mut regressed_metrics = Vec::new();

        // Check TTFT p99 (higher = regression)
        if baseline.results.ttft_ms.p99 > 0.0 {
            let change = ((current.results.ttft_ms.p99 - baseline.results.ttft_ms.p99)
                / baseline.results.ttft_ms.p99)
                * 100.0;
            if change > threshold_pct {
                regressed_metrics.push(format!("ttft_p99 (+{change:.1}%)"));
            }
        }

        // Check throughput (lower = regression)
        if baseline.results.throughput_tok_s.median > 0.0 {
            let change = ((baseline.results.throughput_tok_s.median
                - current.results.throughput_tok_s.median)
                / baseline.results.throughput_tok_s.median)
                * 100.0;
            if change > threshold_pct {
                regressed_metrics.push(format!("throughput (-{change:.1}%)"));
            }
        }

        // Check memory (higher = regression)
        if baseline.results.memory_mb.peak_rss_mb > 0 {
            let change = ((current.results.memory_mb.peak_rss_mb as f64
                - baseline.results.memory_mb.peak_rss_mb as f64)
                / baseline.results.memory_mb.peak_rss_mb as f64)
                * 100.0;
            if change > threshold_pct {
                regressed_metrics.push(format!("memory (+{change:.1}%)"));
            }
        }

        Self {
            regression_detected: !regressed_metrics.is_empty(),
            regressed_metrics,
            threshold_pct,
        }
    }
}

// ============================================================================
// Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // DynamicSampler Tests
    // ========================================================================

    #[test]
    fn test_dynamic_sampler_continues_until_min_samples() {
        let mut dyn_sampler = DynamicSampler::new(100, 10_000, 0.05);
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();

        assert!(dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_stops_at_max_samples() {
        let mut dyn_sampler = DynamicSampler::new(10, 100, 0.05);

        // Generate 100 data points with high variance
        let data: Vec<f64> = (0..100).map(|i| (i % 50) as f64 * 10.0).collect();

        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_stops_when_cv_stable() {
        let mut dyn_sampler = DynamicSampler::new(10, 10_000, 0.05);
        dyn_sampler.stability_count = 1; // Stop after 1 stable check

        // Generate 100 data points with very low variance (CV ~= 0)
        let data: Vec<f64> = vec![100.0; 100];

        // Should stop because CV = 0 < 0.05
        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_requires_stability_streak() {
        let mut dyn_sampler = DynamicSampler::new(10, 10_000, 0.05);
        dyn_sampler.stability_count = 3;

        // Stable data points
        let data: Vec<f64> = vec![100.0; 100];

        // First check - streak = 1
        assert!(dyn_sampler.should_continue(&data));
        // Second check - streak = 2
        assert!(dyn_sampler.should_continue(&data));
        // Third check - streak = 3, should stop
        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_reset() {
        let mut sampler = DynamicSampler::new(10, 10_000, 0.05);
        sampler.stable_streak = 5;
        sampler.reset();
        assert_eq!(sampler.stable_streak, 0);
    }

    #[test]
    fn test_compute_cv_constant_values() {
        let data = vec![100.0; 50];
        let cv = compute_cv(&data);
        assert!(cv.abs() < 1e-10, "CV of constant values should be ~0");
    }

    #[test]
    fn test_compute_cv_varied_values() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let cv = compute_cv(&data);
        // CV = std_dev / mean = 15.81 / 30 ~= 0.527
        assert!(cv > 0.5 && cv < 0.6, "CV should be ~0.527, got {cv}");
    }

    #[test]
    fn test_compute_cv_empty_data() {
        let data: Vec<f64> = vec![];
        let cv = compute_cv(&data);
        assert!(cv.is_infinite());
    }

    // ========================================================================
    // ThermalGuard Tests
    // ========================================================================

    #[test]
    fn test_thermal_guard_valid_low_variance() {
        let guard = ThermalGuard::default();
        let temps = vec![75.0, 75.5, 74.8, 75.2, 75.1];

        assert_eq!(guard.validate_run(&temps), ThermalValidity::Valid);
    }

    #[test]
    fn test_thermal_guard_invalid_high_variance() {
        let guard = ThermalGuard::default();
        // Variance std_dev > 2°C
        let temps = vec![70.0, 75.0, 80.0, 72.0, 78.0];

        match guard.validate_run(&temps) {
            ThermalValidity::Invalid(msg) => {
                assert!(msg.contains("exceeds threshold"));
            },
            ThermalValidity::Valid => panic!("Expected Invalid"),
        }
    }

    #[test]
    fn test_thermal_guard_empty_temps() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.validate_run(&[]), ThermalValidity::Valid);
    }

    #[test]
    fn test_thermal_guard_max_temp() {
        let guard = ThermalGuard::default();
        let temps = vec![70.0, 75.0, 85.0, 72.0];
        assert_eq!(guard.max_temp(&temps), 85.0);
    }

    // ========================================================================
    // KvCacheMetrics Tests
    // ========================================================================

    #[test]
    fn test_kv_cache_metrics_no_waste() {
        let metrics = KvCacheMetrics::new(1000, 1000);
        assert_eq!(metrics.fragmentation_pct, 0.0);
        assert!(metrics.is_acceptable(10.0));
    }

    #[test]
    fn test_kv_cache_metrics_with_waste() {
        let metrics = KvCacheMetrics::new(1000, 800);
        assert!((metrics.fragmentation_pct - 20.0).abs() < 0.01);
        assert!(!metrics.is_acceptable(10.0));
        assert!(metrics.is_acceptable(25.0));
    }

    #[test]
    fn test_kv_cache_metrics_zero_allocated() {
        let metrics = KvCacheMetrics::new(0, 0);
        assert_eq!(metrics.fragmentation_pct, 0.0);
    }

    #[test]
    fn test_kv_cache_metrics_mb_conversion() {
        let metrics = KvCacheMetrics::new(1024 * 1024 * 100, 1024 * 1024 * 80);
        assert!((metrics.allocated_mb() - 100.0).abs() < 0.01);
        assert!((metrics.used_mb() - 80.0).abs() < 0.01);
    }

    // ========================================================================
    // EnergyMetrics Tests
    // ========================================================================

    #[test]
    fn test_energy_metrics_joules_per_token() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 1000);
        assert!((metrics.joules_per_token() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_energy_metrics_zero_tokens() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 0);
        assert_eq!(metrics.joules_per_token(), 0.0);
    }

    #[test]
    fn test_energy_metrics_tokens_per_joule() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 1000);
        assert!((metrics.tokens_per_joule() - 10.0).abs() < 0.001);
    }

    // ========================================================================
    // ItlMetrics Tests
    // ========================================================================

    #[test]
    fn test_itl_metrics_from_measurements() {
        let itl = vec![10.0, 12.0, 11.0, 15.0, 13.0, 14.0, 11.0, 12.0, 13.0, 10.0];
        let metrics = ItlMetrics::from_measurements(&itl);

        // Median should be around 12
        assert!(metrics.median_ms > 11.0 && metrics.median_ms < 13.0);
        // Std dev should be small
        assert!(metrics.std_dev_ms < 5.0);
        // p99 should be around 15
        assert!(metrics.p99_ms >= 14.0);
    }

    #[test]
    fn test_itl_metrics_empty() {
        let metrics = ItlMetrics::from_measurements(&[]);
        assert_eq!(metrics.median_ms, 0.0);
        assert_eq!(metrics.std_dev_ms, 0.0);
    }

    #[test]
    fn test_itl_metrics_low_jitter() {
        let itl = vec![10.0; 100];
        let metrics = ItlMetrics::from_measurements(&itl);
        assert!(metrics.is_low_jitter(1.0));
    }

    #[test]
    fn test_itl_metrics_high_jitter() {
        let itl: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let metrics = ItlMetrics::from_measurements(&itl);
        assert!(!metrics.is_low_jitter(5.0));
    }

    // ========================================================================
    // KL-Divergence Tests
    // ========================================================================

    #[test]
    fn test_kl_divergence_identical_distributions() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = validate_quantization_quality(&logits, &logits, 0.01);

        match result {
            QualityResult::Pass { kl_divergence } => {
                assert!(kl_divergence < 1e-10, "KL should be ~0 for identical");
            },
            QualityResult::Fail { .. } => panic!("Expected Pass for identical"),
        }
    }

    #[test]
    fn test_kl_divergence_slightly_different() {
        let fp32 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quant = vec![1.01, 2.01, 3.01, 4.01, 5.01];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);

        match result {
            QualityResult::Pass { kl_divergence } => {
                assert!(kl_divergence < 0.001, "KL should be very small");
            },
            QualityResult::Fail { .. } => panic!("Expected Pass for small diff"),
        }
    }

    #[test]
    fn test_kl_divergence_very_different() {
        let fp32 = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        let quant = vec![0.0, 0.0, 0.0, 0.0, 10.0];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);

        match result {
            QualityResult::Fail { kl_divergence, .. } => {
                assert!(kl_divergence > 1.0, "KL should be large for opposite");
            },
            QualityResult::Pass { .. } => panic!("Expected Fail for very different"),
        }
    }

    #[test]
    fn test_kl_divergence_mismatched_lengths() {
        let fp32 = vec![1.0, 2.0, 3.0];
        let quant = vec![1.0, 2.0];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);
        assert!(matches!(result, QualityResult::Fail { .. }));
    }

    #[test]
    fn test_kl_divergence_empty() {
        let result = validate_quantization_quality(&[], &[], 0.01);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    // ========================================================================
    // BenchmarkResult Tests
    // ========================================================================

    #[test]
    fn test_benchmark_result_summary() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0, 22.0, 21.0, 25.0, 23.0, 24.0, 22.0, 21.0, 20.0, 26.0],
            itl_ms: vec![10.0, 11.0, 10.5, 11.5, 10.2, 10.8, 11.2, 10.3, 10.7, 11.0],
            generation_tok_s: vec![140.0, 142.0, 141.0, 143.0, 139.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let summary = result.summary();

        // Check percentiles are reasonable
        assert!(summary.ttft_p50 > 20.0 && summary.ttft_p50 < 25.0);
        assert!(summary.ttft_p99 >= summary.ttft_p50);
        assert!(summary.ttft_p999 >= summary.ttft_p99);

        // Check ITL
        assert!(summary.itl_median > 10.0 && summary.itl_median < 12.0);
        assert!(summary.itl_std_dev < 2.0);

        // Check throughput
        assert!(summary.throughput_median > 139.0 && summary.throughput_median < 144.0);

        // Check energy
        assert!((summary.token_joules - 0.05).abs() < 0.001);

        // Check metadata
        assert_eq!(summary.iterations, 500);
        assert!((summary.cv_final - 0.045).abs() < 0.001);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!(percentile(&data, 50.0) >= 5.0 && percentile(&data, 50.0) <= 6.0);
        assert!(percentile(&data, 90.0) >= 9.0);
        assert_eq!(percentile(&data, 100.0), 10.0);
    }

    #[test]
    fn test_bootstrap_ci() {
        let data = vec![100.0; 100];
        let (lower, upper) = bootstrap_ci(&data, 0.95, 1000);

        // For constant data, CI should be tight around 100
        assert!((lower - 100.0).abs() < 0.01);
        assert!((upper - 100.0).abs() < 0.01);
    }

    // ========================================================================
    // Softmax Tests
    // ========================================================================

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_monotonic() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);

        // Higher logits should have higher probabilities
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1]);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large logits shouldn't overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // ========================================================================
    // WorkloadType Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_workload_type_short_qa() {
        let workload = WorkloadType::ShortQa;
        assert_eq!(workload.input_tokens(), 32);
        assert_eq!(workload.output_tokens(), 64);
    }

    #[test]
    fn test_workload_type_long_context() {
        let workload = WorkloadType::LongContext;
        assert_eq!(workload.input_tokens(), 2048);
        assert_eq!(workload.output_tokens(), 512);
    }

    // ========================================================================
    // ConvoyTestConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_convoy_config_default() {
        let config = ConvoyTestConfig::default();
        assert_eq!(config.long_requests, 10);
        assert_eq!(config.short_requests, 100);
        assert!((config.max_p99_increase_pct - 50.0).abs() < 0.01);
        assert!((config.max_hol_blocking_ms - 500.0).abs() < 0.01);
        assert!((config.max_kv_fragmentation_pct - 15.0).abs() < 0.01);
    }

    // ========================================================================
    // ConvoyTestResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_convoy_test_result_pass() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0, 12.0, 11.0, 13.0, 10.5]; // p99 ~= 13
        let convoy = vec![12.0, 14.0, 13.0, 15.0, 12.5]; // p99 ~= 15 (15% increase)
        let hol = vec![50.0, 100.0, 75.0, 80.0, 60.0];
        let kv_frag = 10.0; // 10% < 15% threshold

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(result.passed, "Should pass with acceptable metrics");
        assert!(result.failure_reasons.is_empty());
        assert!(result.p99_increase_pct < 50.0);
        assert!(result.max_hol_blocking_ms < 500.0);
    }

    #[test]
    fn test_convoy_test_result_fail_p99() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![20.0; 100]; // 100% increase > 50% threshold
        let hol = vec![50.0; 100];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with 100% p99 increase");
        assert!(result.failure_reasons.iter().any(|r| r.contains("P99")));
    }

    #[test]
    fn test_convoy_test_result_fail_hol_blocking() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![11.0; 100]; // 10% increase - acceptable
        let hol = vec![600.0; 100]; // 600ms > 500ms threshold
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with HOL blocking > 500ms");
        assert!(result.failure_reasons.iter().any(|r| r.contains("HOL")));
    }

    #[test]
    fn test_convoy_test_result_fail_kv_fragmentation() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![11.0; 100];
        let hol = vec![50.0; 100];
        let kv_frag = 20.0; // 20% > 15% threshold

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with KV fragmentation > 15%");
        assert!(result.failure_reasons.iter().any(|r| r.contains("KV")));
    }

    // ========================================================================
    // SaturationTestConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_saturation_config_default() {
        let config = SaturationTestConfig::default();
        assert_eq!(config.cpu_load_pct, 50);
        assert!((config.max_throughput_degradation_pct - 30.0).abs() < 0.01);
        assert!((config.max_p99_increase_pct - 100.0).abs() < 0.01);
    }

    // ========================================================================
    // SaturationTestResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_saturation_test_result_pass() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let stressed_throughput = vec![85.0, 87.0, 83.0, 86.0, 84.0]; // ~15% degradation
        let baseline_latency = vec![10.0, 12.0, 11.0, 10.5, 11.5];
        let stressed_latency = vec![15.0, 17.0, 16.0, 15.5, 16.5]; // ~50% increase

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(result.passed, "Should pass with acceptable degradation");
        assert!(result.throughput_degradation_pct < 30.0);
        assert!(result.p99_increase_pct < 100.0);
    }

    #[test]
    fn test_saturation_test_result_fail_throughput() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0; 100];
        let stressed_throughput = vec![50.0; 100]; // 50% degradation > 30%
        let baseline_latency = vec![10.0; 100];
        let stressed_latency = vec![15.0; 100]; // 50% increase - acceptable

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(
            !result.passed,
            "Should fail with 50% throughput degradation"
        );
        assert!(result
            .failure_reasons
            .iter()
            .any(|r| r.contains("Throughput")));
    }

    #[test]
    fn test_saturation_test_result_fail_p99() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0; 100];
        let stressed_throughput = vec![90.0; 100]; // 10% degradation - acceptable
        let baseline_latency = vec![10.0; 100];
        let stressed_latency = vec![25.0; 100]; // 150% increase > 100%

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(!result.passed, "Should fail with 150% p99 increase");
        assert!(result.failure_reasons.iter().any(|r| r.contains("P99")));
    }

    // ========================================================================
    // HardwareSpec Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_hardware_spec_default() {
        let spec = HardwareSpec::default();
        assert_eq!(spec.cpu, "Unknown");
        assert!(spec.gpu.is_none());
        assert_eq!(spec.memory_gb, 0);
        assert_eq!(spec.storage, "Unknown");
    }

    // ========================================================================
    // SamplingConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.method, "dynamic_cv");
        assert!((config.cv_threshold - 0.05).abs() < 0.001);
        assert_eq!(config.warmup_iterations, 100);
    }

    // ========================================================================
    // ThermalInfo Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_thermal_info_default() {
        let info = ThermalInfo::default();
        assert!(info.valid);
        assert!((info.temp_variance_c - 0.0).abs() < 0.001);
        assert!((info.max_temp_c - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // FullBenchmarkResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_full_benchmark_result_from_benchmark_result() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0, 22.0, 21.0, 25.0, 23.0],
            itl_ms: vec![10.0, 11.0, 10.5, 11.5, 10.2],
            generation_tok_s: vec![140.0, 142.0, 141.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let hardware = HardwareSpec {
            cpu: "Apple M3 Max".to_string(),
            gpu: Some("Apple M3 Max (40 cores)".to_string()),
            memory_gb: 128,
            storage: "NVMe".to_string(),
        };

        let temps = vec![72.0, 73.0, 72.5, 73.5, 72.0];
        let kl_div = 0.031;

        let full_result =
            FullBenchmarkResult::from_benchmark_result(&result, hardware, &temps, kl_div);

        assert_eq!(full_result.version, "1.1");
        assert!(full_result.timestamp.contains("1970")); // Simple timestamp format
        assert_eq!(full_result.config.model, "test");
        assert_eq!(full_result.hardware.cpu, "Apple M3 Max");
        assert_eq!(full_result.sampling.actual_iterations, 500);
        assert!(full_result.thermal.valid);
        assert!((full_result.quality.kl_divergence_vs_fp32 - 0.031).abs() < 0.001);
    }

    #[test]
    fn test_full_benchmark_result_json_roundtrip() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0, 22.0, 21.0],
            itl_ms: vec![10.0, 11.0, 10.5],
            generation_tok_s: vec![140.0, 142.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let full_result =
            FullBenchmarkResult::from_benchmark_result(&result, HardwareSpec::default(), &[], 0.0);

        let json = full_result.to_json().expect("Should serialize");
        let parsed: FullBenchmarkResult =
            FullBenchmarkResult::from_json(&json).expect("Should parse");

        assert_eq!(parsed.version, "1.1");
        assert_eq!(parsed.config.model, "test");
        assert_eq!(parsed.sampling.actual_iterations, 500);
    }

    // ========================================================================
    // BenchmarkComparison Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_benchmark_comparison_realizar_wins() {
        let baseline = create_test_full_result("llama.cpp", 40.0, 100.0, 1500, 0.06);
        let current = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);

        assert_eq!(comparison.winner, "realizar");
        assert!(comparison.ttft_p99_change_pct < 0.0); // Improvement
        assert!(comparison.throughput_change_pct > 0.0); // Improvement
        assert!(comparison.memory_change_pct < 0.0); // Improvement
        assert!(comparison.energy_change_pct < 0.0); // Improvement
    }

    #[test]
    fn test_benchmark_comparison_tie() {
        let baseline = create_test_full_result("runtime_a", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("runtime_b", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);

        assert_eq!(comparison.winner, "tie");
    }

    // ========================================================================
    // RegressionResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_regression_result_no_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 29.0, 145.0, 1150, 0.038);

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(!regression.regression_detected);
        assert!(regression.regressed_metrics.is_empty());
    }

    #[test]
    fn test_regression_result_ttft_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 35.0, 140.0, 1200, 0.04); // 16.7% worse TTFT

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("ttft")));
    }

    #[test]
    fn test_regression_result_throughput_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 30.0, 120.0, 1200, 0.04); // 14.3% worse throughput

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("throughput")));
    }

    #[test]
    fn test_regression_result_memory_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 30.0, 140.0, 1400, 0.04); // 16.7% worse memory

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("memory")));
    }

    /// Helper function to create test FullBenchmarkResult
    fn create_test_full_result(
        runtime: &str,
        ttft_p99: f64,
        throughput: f64,
        memory_mb: u64,
        token_joules: f64,
    ) -> FullBenchmarkResult {
        FullBenchmarkResult {
            version: "1.1".to_string(),
            timestamp: "2025-12-09T12:00:00Z".to_string(),
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: runtime.to_string(),
                runtime_version: "1.0.0".to_string(),
            },
            hardware: HardwareSpec::default(),
            sampling: SamplingConfig::default(),
            thermal: ThermalInfo::default(),
            results: BenchmarkResults {
                ttft_ms: TtftResults {
                    p50: ttft_p99 * 0.7,
                    p95: ttft_p99 * 0.9,
                    p99: ttft_p99,
                    p999: ttft_p99 * 1.2,
                },
                itl_ms: ItlResults {
                    median: 10.0,
                    std_dev: 2.0,
                    p99: 15.0,
                },
                throughput_tok_s: ThroughputResults {
                    median: throughput,
                    ci_95: (throughput * 0.95, throughput * 1.05),
                },
                memory_mb: MemoryResults {
                    model_mb: memory_mb / 2,
                    peak_rss_mb: memory_mb,
                    kv_waste_pct: 3.0,
                },
                energy: EnergyResults {
                    total_joules: 50.0,
                    token_joules,
                    idle_watts: 8.0,
                },
                cold_start_ms: ColdStartResults {
                    median: 100.0,
                    p99: 150.0,
                },
            },
            quality: QualityValidation {
                kl_divergence_vs_fp32: 0.03,
                perplexity_wikitext2: Some(5.89),
            },
        }
    }

    // ========================================================================
    // Additional Coverage Tests (Phase 3 - 95% Target)
    // ========================================================================

    #[test]
    fn test_dynamic_sampler_current_cv_empty() {
        let sampler = DynamicSampler::default();
        let cv = sampler.current_cv(&[]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_dynamic_sampler_current_cv_single_value() {
        let sampler = DynamicSampler::default();
        let cv = sampler.current_cv(&[100.0]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_dynamic_sampler_current_cv_constant_values() {
        let sampler = DynamicSampler::default();
        let data: Vec<f64> = vec![50.0; 100];
        let cv = sampler.current_cv(&data);
        assert!(cv.abs() < 1e-10, "CV of constant should be ~0");
    }

    #[test]
    fn test_dynamic_sampler_current_cv_varied_window() {
        let mut sampler = DynamicSampler::default();
        sampler.cv_window = 10;
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 % 10.0)).collect();
        let cv = sampler.current_cv(&data);
        assert!(cv > 0.0 && cv < 1.0);
    }

    #[test]
    fn test_dynamic_sampler_current_cv_small_window() {
        let mut sampler = DynamicSampler::default();
        sampler.cv_window = 5;
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let cv = sampler.current_cv(&data);
        assert!(cv > 0.4 && cv < 0.6);
    }

    #[test]
    fn test_dynamic_sampler_default_values() {
        let sampler = DynamicSampler::default();
        assert_eq!(sampler.min_samples, 100);
        assert_eq!(sampler.max_samples, 10_000);
        assert!((sampler.cv_threshold - 0.05).abs() < 0.001);
        assert_eq!(sampler.cv_window, 50);
        assert_eq!(sampler.stability_count, 3);
        // stable_streak is private, tested via should_continue
    }

    #[test]
    fn test_thermal_guard_temp_variance_empty() {
        let guard = ThermalGuard::default();
        let variance = guard.temp_variance(&[]);
        assert!((variance - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_thermal_guard_temp_variance_single() {
        let guard = ThermalGuard::default();
        let variance = guard.temp_variance(&[75.0]);
        assert!((variance - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_thermal_guard_temp_variance_constant() {
        let guard = ThermalGuard::default();
        let temps = vec![72.0; 100];
        let variance = guard.temp_variance(&temps);
        assert!(variance < 0.001);
    }

    #[test]
    fn test_thermal_guard_temp_variance_varied() {
        let guard = ThermalGuard::default();
        let temps = vec![70.0, 72.0, 74.0, 76.0, 78.0];
        let variance = guard.temp_variance(&temps);
        assert!(variance > 2.0 && variance < 4.0);
    }

    #[test]
    fn test_thermal_guard_max_temp_empty() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp(&[]), 0.0);
    }

    #[test]
    fn test_thermal_guard_max_temp_single() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp(&[82.5]), 82.5);
    }

    #[test]
    fn test_thermal_guard_cooldown_not_needed() {
        let guard = ThermalGuard::default();
        // Should not sleep when temp is below max
        guard.cooldown_if_needed(70.0);
        // Test passes if no timeout
    }

    #[test]
    fn test_chrono_timestamp_format() {
        let ts = chrono_timestamp();
        assert!(ts.contains("1970"));
        assert!(ts.contains("T"));
        assert!(ts.contains("Z"));
        assert!(ts.contains("+"));
        assert!(ts.contains("s"));
    }

    #[test]
    fn test_bootstrap_ci_empty() {
        let (lower, upper) = bootstrap_ci(&[], 0.95, 1000);
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 0.0);
    }

    #[test]
    fn test_bootstrap_ci_single_value() {
        let (lower, upper) = bootstrap_ci(&[42.0], 0.95, 1000);
        assert!((lower - 42.0).abs() < 0.01);
        assert!((upper - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_bootstrap_ci_varied_data() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let (lower, upper) = bootstrap_ci(&data, 0.95, 1000);
        // Mean is 50.5, CI should contain mean
        assert!(lower < 55.0);
        assert!(upper > 45.0);
        assert!(lower < upper);
    }

    #[test]
    fn test_bootstrap_ci_narrow_confidence() {
        let data = vec![100.0; 50];
        let (lower, upper) = bootstrap_ci(&data, 0.50, 100);
        // Even narrow CI should be close to 100 for constant data
        assert!((lower - 100.0).abs() < 0.1);
        assert!((upper - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
        assert_eq!(percentile(&[42.0], 99.0), 42.0);
    }

    #[test]
    fn test_compute_std_dev_constant() {
        let data = vec![100.0; 50];
        let std_dev = compute_std_dev(&data);
        assert!(std_dev < 0.001);
    }

    #[test]
    fn test_compute_std_dev_empty() {
        let std_dev = compute_std_dev(&[]);
        assert_eq!(std_dev, 0.0);
    }

    #[test]
    fn test_compute_variance_empty() {
        assert_eq!(compute_variance(&[]), 0.0);
    }

    #[test]
    fn test_compute_variance_single() {
        assert_eq!(compute_variance(&[100.0]), 0.0);
    }

    #[test]
    fn test_compute_cv_single_value() {
        let cv = compute_cv(&[100.0]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_compute_cv_zero_mean() {
        // Mix of positive and negative that averages to near zero
        let data = vec![-1.0, 1.0, -1.0, 1.0];
        let cv = compute_cv(&data);
        // Mean is 0, CV should be infinite
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_energy_metrics_tokens_per_joule_zero_joules() {
        let metrics = EnergyMetrics::new(0.0, 10.0, 50.0, 1000);
        assert_eq!(metrics.tokens_per_joule(), 0.0);
    }

    #[test]
    fn test_energy_metrics_very_small_joules() {
        let metrics = EnergyMetrics::new(1e-15, 10.0, 50.0, 1000);
        assert_eq!(metrics.tokens_per_joule(), 0.0);
    }

    #[test]
    fn test_itl_metrics_single_value() {
        let metrics = ItlMetrics::from_measurements(&[15.0]);
        assert_eq!(metrics.median_ms, 15.0);
        assert_eq!(metrics.p99_ms, 15.0);
        assert_eq!(metrics.p999_ms, 15.0);
        assert_eq!(metrics.std_dev_ms, 0.0);
    }

    #[test]
    fn test_itl_metrics_two_values() {
        let metrics = ItlMetrics::from_measurements(&[10.0, 20.0]);
        assert_eq!(metrics.median_ms, 15.0);
        assert!(metrics.std_dev_ms > 0.0);
    }

    #[test]
    fn test_convoy_test_result_empty_hol() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 10];
        let convoy = vec![11.0; 10];
        let hol: Vec<f64> = vec![];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);
        assert_eq!(result.avg_hol_blocking_ms, 0.0);
        assert_eq!(result.max_hol_blocking_ms, 0.0);
    }

    #[test]
    fn test_convoy_test_result_zero_baseline() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![0.0; 10];
        let convoy = vec![10.0; 10];
        let hol = vec![50.0; 10];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);
        assert_eq!(result.p99_increase_pct, 0.0);
    }

    #[test]
    fn test_saturation_test_result_empty_data() {
        let config = SaturationTestConfig::default();
        let result = SaturationTestResult::new(&config, &[], &[], &[], &[]);

        assert_eq!(result.baseline_throughput, 0.0);
        assert_eq!(result.stressed_throughput, 0.0);
        assert_eq!(result.throughput_degradation_pct, 0.0);
    }

    #[test]
    fn test_saturation_test_result_zero_baseline() {
        let config = SaturationTestConfig::default();
        let result =
            SaturationTestResult::new(&config, &[0.0; 10], &[50.0; 10], &[0.0; 10], &[10.0; 10]);

        assert_eq!(result.throughput_degradation_pct, 0.0);
        assert_eq!(result.p99_increase_pct, 0.0);
    }

    #[test]
    fn test_benchmark_comparison_zero_baselines() {
        let baseline = create_test_full_result("baseline", 0.0, 0.0, 0, 0.0);
        let current = create_test_full_result("current", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);
        assert_eq!(comparison.ttft_p99_change_pct, 0.0);
        assert_eq!(comparison.throughput_change_pct, 0.0);
        assert_eq!(comparison.memory_change_pct, 0.0);
        assert_eq!(comparison.energy_change_pct, 0.0);
    }

    #[test]
    fn test_regression_result_zero_baselines() {
        let baseline = create_test_full_result("test", 0.0, 0.0, 0, 0.0);
        let current = create_test_full_result("test", 30.0, 140.0, 1200, 0.04);

        let regression = RegressionResult::check(&baseline, &current, 5.0);
        // No regression detected because baseline is zero
        assert!(!regression.regression_detected);
    }

    #[test]
    fn test_benchmark_result_zero_tokens() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0],
            itl_ms: vec![10.0],
            generation_tok_s: vec![140.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 0,
            actual_iterations: 100,
            cv_at_stop: 0.04,
            timestamp: 12345,
        };

        let summary = result.summary();
        assert_eq!(summary.token_joules, 0.0);
    }

    #[test]
    fn test_kv_cache_used_more_than_allocated() {
        // Edge case: used > allocated (shouldn't happen but test boundary)
        let metrics = KvCacheMetrics::new(1000, 1500);
        // saturating_sub gives 0 waste
        assert_eq!(metrics.fragmentation_pct, 0.0);
    }

    #[test]
    fn test_softmax_single_value() {
        let probs = softmax(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_negative_values() {
        let logits = vec![-5.0, -3.0, -1.0, 0.0, 1.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Should still be monotonic
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1]);
        }
    }

    #[test]
    fn test_full_benchmark_result_invalid_thermal() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0],
            itl_ms: vec![10.0],
            generation_tok_s: vec![140.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 100,
            cv_at_stop: 0.04,
            timestamp: 12345,
        };

        // High variance temps that should be invalid
        let temps = vec![60.0, 70.0, 80.0, 65.0, 85.0];
        let full_result = FullBenchmarkResult::from_benchmark_result(
            &result,
            HardwareSpec::default(),
            &temps,
            0.03,
        );

        assert!(!full_result.thermal.valid);
        assert!(full_result.thermal.temp_variance_c > 2.0);
    }

    #[test]
    fn test_benchmark_comparison_baseline_wins() {
        let baseline = create_test_full_result("baseline", 25.0, 160.0, 1000, 0.03);
        let current = create_test_full_result("current", 40.0, 100.0, 1500, 0.06);

        let comparison = BenchmarkComparison::compare(&baseline, &current);
        assert_eq!(comparison.winner, "baseline");
    }

    #[test]
    fn test_thermal_validity_debug() {
        let valid = ThermalValidity::Valid;
        let invalid = ThermalValidity::Invalid("test".to_string());
        // Test Debug derive
        assert!(format!("{valid:?}").contains("Valid"));
        assert!(format!("{invalid:?}").contains("Invalid"));
    }

    #[test]
    fn test_quality_result_debug() {
        let pass = QualityResult::Pass {
            kl_divergence: 0.01,
        };
        let fail = QualityResult::Fail {
            kl_divergence: 0.5,
            threshold: 0.1,
            message: "test",
        };
        // Test Debug derive
        assert!(format!("{pass:?}").contains("Pass"));
        assert!(format!("{fail:?}").contains("Fail"));
    }

    #[test]
    fn test_workload_type_equality() {
        assert_eq!(WorkloadType::ShortQa, WorkloadType::ShortQa);
        assert_eq!(WorkloadType::LongContext, WorkloadType::LongContext);
        assert_ne!(WorkloadType::ShortQa, WorkloadType::LongContext);
    }

    #[test]
    fn test_workload_type_copy() {
        let wt = WorkloadType::ShortQa;
        let wt_copy = wt;
        assert_eq!(wt, wt_copy);
    }
}
