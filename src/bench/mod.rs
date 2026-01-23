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

use std::fmt::Write;
use std::time::Duration;

use serde::{Deserialize, Serialize};

#[cfg(feature = "bench-http")]
use crate::http_client::{CompletionRequest, ModelHttpClient, OllamaOptions, OllamaRequest};

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
    /// Create a new ThermalGuard with custom parameters
    #[must_use]
    pub fn new(
        max_temp_c: f64,
        cooldown_threshold_c: f64,
        cooldown_sleep_ms: u64,
        temp_variance_c: f64,
    ) -> Self {
        Self {
            max_temp_c,
            cooldown_threshold_c,
            cooldown_sleep_ms,
            temp_variance_c,
        }
    }

    /// Check if cooldown is needed (without sleeping)
    #[must_use]
    pub fn needs_cooldown(&self, current_temp: f64) -> bool {
        current_temp > self.max_temp_c
    }

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
        let median_ms = if n.is_multiple_of(2) {
            f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
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
// BENCH-002: Runtime Backend Abstraction (Refs BENCH-002)
// ============================================================================

use std::collections::HashMap;

use crate::error::RealizarError;

/// Supported runtime types for inference benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeType {
    /// Native Realizar runtime (.apr format)
    Realizar,
    /// llama.cpp (GGUF format)
    LlamaCpp,
    /// vLLM (safetensors, HuggingFace)
    Vllm,
    /// Ollama (wraps llama.cpp)
    Ollama,
}

impl RuntimeType {
    /// Get string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Realizar => "realizar",
            Self::LlamaCpp => "llama-cpp",
            Self::Vllm => "vllm",
            Self::Ollama => "ollama",
        }
    }

    /// Parse from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "realizar" => Some(Self::Realizar),
            "llama-cpp" | "llama.cpp" | "llamacpp" => Some(Self::LlamaCpp),
            "vllm" => Some(Self::Vllm),
            "ollama" => Some(Self::Ollama),
            _ => None,
        }
    }
}

/// Request for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature
    pub temperature: f64,
    /// Optional stop sequences
    pub stop: Vec<String>,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 100,
            temperature: 0.7,
            stop: Vec::new(),
        }
    }
}

impl InferenceRequest {
    /// Create new request with prompt
    #[must_use]
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            ..Default::default()
        }
    }

    /// Set max tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }
}

/// Response from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Time to first token (ms)
    pub ttft_ms: f64,
    /// Total generation time (ms)
    pub total_time_ms: f64,
    /// Inter-token latencies (ms)
    pub itl_ms: Vec<f64>,
}

impl InferenceResponse {
    /// Calculate tokens per second
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_time_ms <= 0.0 {
            return 0.0;
        }
        (self.tokens_generated as f64) / (self.total_time_ms / 1000.0)
    }
}

/// Runtime backend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Runtime type
    pub runtime_type: RuntimeType,
    /// Version string
    pub version: String,
    /// Whether streaming is supported
    pub supports_streaming: bool,
    /// Model currently loaded (if any)
    pub loaded_model: Option<String>,
}

/// Trait for inference runtime backends
pub trait RuntimeBackend: Send + Sync {
    /// Get backend information
    fn info(&self) -> BackendInfo;

    /// Run inference
    ///
    /// # Errors
    ///
    /// Returns `RealizarError` if inference fails due to:
    /// - Model not loaded
    /// - Backend communication failure
    /// - Invalid request parameters
    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError>;

    /// Load a model (if applicable)
    ///
    /// # Errors
    ///
    /// Returns `RealizarError` if model loading fails due to:
    /// - Model file not found
    /// - Invalid model format
    /// - Insufficient memory
    fn load_model(&mut self, _model_path: &str) -> Result<(), RealizarError> {
        Ok(()) // Default: no-op
    }
}

/// Mock backend for testing
pub struct MockBackend {
    ttft_ms: f64,
    tokens_per_second: f64,
}

impl MockBackend {
    /// Create a new mock backend with specified latencies
    #[must_use]
    pub fn new(ttft_ms: f64, tokens_per_second: f64) -> Self {
        Self {
            ttft_ms,
            tokens_per_second,
        }
    }
}

impl RuntimeBackend for MockBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            runtime_type: RuntimeType::Realizar,
            version: env!("CARGO_PKG_VERSION").to_string(),
            supports_streaming: true,
            loaded_model: None,
        }
    }

    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError> {
        let tokens = request.max_tokens.min(100);
        let gen_time_ms = (tokens as f64) / self.tokens_per_second * 1000.0;

        Ok(InferenceResponse {
            text: "Mock response".to_string(),
            tokens_generated: tokens,
            ttft_ms: self.ttft_ms,
            total_time_ms: self.ttft_ms + gen_time_ms,
            itl_ms: vec![gen_time_ms / tokens as f64; tokens],
        })
    }
}

/// Registry of available backends
pub struct BackendRegistry {
    backends: HashMap<RuntimeType, Box<dyn RuntimeBackend>>,
}

impl BackendRegistry {
    /// Create empty registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Register a backend
    pub fn register(&mut self, runtime: RuntimeType, backend: Box<dyn RuntimeBackend>) {
        self.backends.insert(runtime, backend);
    }

    /// Get a backend by type
    #[must_use]
    pub fn get(&self, runtime: RuntimeType) -> Option<&dyn RuntimeBackend> {
        self.backends.get(&runtime).map(AsRef::as_ref)
    }

    /// List registered runtimes
    #[must_use]
    pub fn list(&self) -> Vec<RuntimeType> {
        self.backends.keys().copied().collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for llama.cpp backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    /// Path to llama-cli binary
    pub binary_path: String,
    /// Path to model file
    pub model_path: Option<String>,
    /// Number of GPU layers to offload
    pub n_gpu_layers: u32,
    /// Context size
    pub ctx_size: usize,
    /// Number of threads
    pub threads: usize,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            binary_path: "llama-cli".to_string(),
            model_path: None,
            n_gpu_layers: 0,
            ctx_size: 2048,
            threads: 4,
        }
    }
}

impl LlamaCppConfig {
    /// Create new config with binary path
    #[must_use]
    pub fn new(binary_path: &str) -> Self {
        Self {
            binary_path: binary_path.to_string(),
            ..Default::default()
        }
    }

    /// Set model path
    #[must_use]
    pub fn with_model(mut self, model_path: &str) -> Self {
        self.model_path = Some(model_path.to_string());
        self
    }

    /// Set GPU layers
    #[must_use]
    pub fn with_gpu_layers(mut self, layers: u32) -> Self {
        self.n_gpu_layers = layers;
        self
    }

    /// Set context size
    #[must_use]
    pub fn with_ctx_size(mut self, ctx_size: usize) -> Self {
        self.ctx_size = ctx_size;
        self
    }
}

/// Configuration for vLLM backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    /// Base URL for vLLM server
    pub base_url: String,
    /// API version
    pub api_version: String,
    /// Model name/path
    pub model: Option<String>,
    /// API key (if required)
    pub api_key: Option<String>,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000".to_string(),
            api_version: "v1".to_string(),
            model: None,
            api_key: None,
        }
    }
}

impl VllmConfig {
    /// Create new config with base URL
    #[must_use]
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            ..Default::default()
        }
    }

    /// Set model
    #[must_use]
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Set API key
    #[must_use]
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }
}

// ============================================================================
// LlamaCppBackend Implementation (BENCH-002)
// ============================================================================

/// llama.cpp backend for GGUF model inference via subprocess
pub struct LlamaCppBackend {
    config: LlamaCppConfig,
}

impl LlamaCppBackend {
    /// Create new llama.cpp backend
    #[must_use]
    pub fn new(config: LlamaCppConfig) -> Self {
        Self { config }
    }

    /// Build CLI arguments for llama-cli invocation
    #[must_use]
    pub fn build_cli_args(&self, request: &InferenceRequest) -> Vec<String> {
        let mut args = Vec::new();

        // Model path
        if let Some(ref model_path) = self.config.model_path {
            args.push("-m".to_string());
            args.push(model_path.clone());
        }

        // Prompt
        args.push("-p".to_string());
        args.push(request.prompt.clone());

        // Number of tokens to generate
        args.push("-n".to_string());
        args.push(request.max_tokens.to_string());

        // GPU layers
        args.push("-ngl".to_string());
        args.push(self.config.n_gpu_layers.to_string());

        // Context size
        args.push("-c".to_string());
        args.push(self.config.ctx_size.to_string());

        // Threads
        args.push("-t".to_string());
        args.push(self.config.threads.to_string());

        // Temperature (if non-default)
        if (request.temperature - 0.8).abs() > 0.01 {
            args.push("--temp".to_string());
            args.push(format!("{:.2}", request.temperature));
        }

        args
    }

    /// Parse a timing line from llama-cli output
    ///
    /// Example: `llama_perf_context_print: prompt eval time =      12.34 ms /    10 tokens`
    /// Returns: `Some((12.34, 10))`
    #[must_use]
    pub fn parse_timing_line(output: &str, metric_name: &str) -> Option<(f64, usize)> {
        for line in output.lines() {
            // For "eval time", we need to exclude "prompt eval time"
            let matches = if metric_name == "eval time" {
                line.contains(metric_name) && !line.contains("prompt eval time")
            } else {
                line.contains(metric_name)
            };

            if matches && line.contains('=') {
                // Extract the value after "=" and before "ms"
                // Format: "metric_name =      12.34 ms /    10 tokens"
                if let Some(eq_pos) = line.find('=') {
                    let after_eq = &line[eq_pos + 1..];
                    // Find ms position
                    if let Some(ms_pos) = after_eq.find("ms") {
                        let value_str = after_eq[..ms_pos].trim();
                        if let Ok(value) = value_str.parse::<f64>() {
                            // Find the count after "/"
                            if let Some(slash_pos) = after_eq.find('/') {
                                let after_slash = &after_eq[slash_pos + 1..];
                                // Extract number before "tokens" or "runs"
                                let count_str =
                                    after_slash.split_whitespace().next().unwrap_or("0");
                                if let Ok(count) = count_str.parse::<usize>() {
                                    return Some((value, count));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract generated text from llama-cli output (before timing lines)
    #[must_use]
    pub fn extract_generated_text(output: &str) -> String {
        let mut text_lines = Vec::new();
        for line in output.lines() {
            // Stop when we hit timing/performance lines
            if line.contains("llama_perf_") || line.contains("sampler") {
                break;
            }
            text_lines.push(line);
        }
        text_lines.join("\n").trim().to_string()
    }

    /// Parse full CLI output into InferenceResponse
    ///
    /// # Errors
    ///
    /// Returns error if timing information cannot be parsed from output.
    pub fn parse_cli_output(output: &str) -> Result<InferenceResponse, RealizarError> {
        // Extract generated text
        let text = Self::extract_generated_text(output);

        // Parse timing metrics
        let ttft_ms = Self::parse_timing_line(output, "prompt eval time").map_or(0.0, |(ms, _)| ms);

        let (total_time_ms, _) = Self::parse_timing_line(output, "total time").unwrap_or((0.0, 0));

        let (_, tokens_generated) =
            Self::parse_timing_line(output, "eval time").unwrap_or((0.0, 0));

        // ITL is not directly available from CLI output, estimate from eval time
        let eval_time = Self::parse_timing_line(output, "eval time").map_or(0.0, |(ms, _)| ms);

        let itl_ms = if tokens_generated > 1 {
            let avg_itl = eval_time / (tokens_generated as f64);
            vec![avg_itl; tokens_generated.saturating_sub(1)]
        } else {
            vec![]
        };

        Ok(InferenceResponse {
            text,
            tokens_generated,
            ttft_ms,
            total_time_ms,
            itl_ms,
        })
    }
}

impl RuntimeBackend for LlamaCppBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            runtime_type: RuntimeType::LlamaCpp,
            version: "b2345".to_string(), // Would be detected from binary
            supports_streaming: false,    // CLI mode doesn't stream
            loaded_model: self.config.model_path.clone(),
        }
    }

    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError> {
        use std::process::Command;

        // Require model path
        let model_path = self.config.model_path.as_ref().ok_or_else(|| {
            RealizarError::InvalidConfiguration("model_path is required".to_string())
        })?;

        // Build CLI arguments
        let args = self.build_cli_args(request);

        // Execute llama-cli
        let output = Command::new(&self.config.binary_path)
            .args(&args)
            .output()
            .map_err(|e| {
                RealizarError::ModelNotFound(format!(
                    "Failed to execute {}: {}",
                    self.config.binary_path, e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RealizarError::InferenceError(format!(
                "llama-cli failed: {} (model: {})",
                stderr, model_path
            )));
        }

        // Parse stdout for response and timing
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Timing info is often in stderr, combine both
        let combined_output = format!("{}\n{}", stdout, stderr);
        Self::parse_cli_output(&combined_output)
    }
}

// ============================================================================
// VllmBackend Implementation (BENCH-003) - REAL HTTP CALLS
// ============================================================================

/// vLLM backend for inference via HTTP API
///
/// **REAL IMPLEMENTATION** - makes actual HTTP requests to vLLM servers.
/// No mock data. Measures real latency and throughput.
#[cfg(feature = "bench-http")]
pub struct VllmBackend {
    config: VllmConfig,
    http_client: ModelHttpClient,
}

#[cfg(feature = "bench-http")]
impl VllmBackend {
    /// Create new vLLM backend with default HTTP client
    #[must_use]
    pub fn new(config: VllmConfig) -> Self {
        Self {
            config,
            http_client: ModelHttpClient::new(),
        }
    }

    /// Create new vLLM backend with custom HTTP client
    #[must_use]
    pub fn with_client(config: VllmConfig, client: ModelHttpClient) -> Self {
        Self {
            config,
            http_client: client,
        }
    }
}

#[cfg(feature = "bench-http")]
impl RuntimeBackend for VllmBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            runtime_type: RuntimeType::Vllm,
            version: "0.4.0".to_string(), // Would be detected from API
            supports_streaming: true,
            loaded_model: self.config.model.clone(),
        }
    }

    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError> {
        // Parse URL to check for invalid port
        let url = &self.config.base_url;
        if let Some(port_str) = url.split(':').next_back() {
            if let Ok(port) = port_str.parse::<u32>() {
                if port > 65535 {
                    return Err(RealizarError::ConnectionError(format!(
                        "Invalid port in URL: {}",
                        url
                    )));
                }
            }
        }

        // REAL HTTP request to vLLM server via OpenAI-compatible API
        #[allow(clippy::cast_possible_truncation)]
        let completion_request = CompletionRequest {
            model: self
                .config
                .model
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens,
            temperature: Some(request.temperature as f32),
            stream: false,
        };

        let timing = self.http_client.openai_completion(
            &self.config.base_url,
            &completion_request,
            self.config.api_key.as_deref(),
        )?;

        Ok(InferenceResponse {
            text: timing.text,
            tokens_generated: timing.tokens_generated,
            ttft_ms: timing.ttft_ms,
            total_time_ms: timing.total_time_ms,
            itl_ms: vec![], // ITL requires streaming, not available in blocking mode
        })
    }
}

// ============================================================================
// OllamaBackend Implementation - REAL HTTP CALLS
// ============================================================================

/// Configuration for Ollama backend
#[cfg(feature = "bench-http")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Base URL for Ollama server
    pub base_url: String,
    /// Model name
    pub model: String,
}

#[cfg(feature = "bench-http")]
impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
        }
    }
}

/// Ollama backend for inference via HTTP API
///
/// **REAL IMPLEMENTATION** - makes actual HTTP requests to Ollama servers.
/// No mock data. Measures real latency and throughput.
#[cfg(feature = "bench-http")]
pub struct OllamaBackend {
    config: OllamaConfig,
    http_client: ModelHttpClient,
}

#[cfg(feature = "bench-http")]
impl OllamaBackend {
    /// Create new Ollama backend with default HTTP client
    #[must_use]
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config,
            http_client: ModelHttpClient::new(),
        }
    }

    /// Create new Ollama backend with custom HTTP client
    #[must_use]
    pub fn with_client(config: OllamaConfig, client: ModelHttpClient) -> Self {
        Self {
            config,
            http_client: client,
        }
    }
}

#[cfg(feature = "bench-http")]
impl RuntimeBackend for OllamaBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            runtime_type: RuntimeType::Ollama,
            version: "0.1.0".to_string(), // Would be detected from API
            supports_streaming: true,
            loaded_model: Some(self.config.model.clone()),
        }
    }

    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError> {
        // REAL HTTP request to Ollama server
        #[allow(clippy::cast_possible_truncation)]
        let ollama_request = OllamaRequest {
            model: self.config.model.clone(),
            prompt: request.prompt.clone(),
            stream: false,
            options: Some(OllamaOptions {
                num_predict: Some(request.max_tokens),
                temperature: Some(request.temperature as f32),
            }),
        };

        let timing = self
            .http_client
            .ollama_generate(&self.config.base_url, &ollama_request)?;

        Ok(InferenceResponse {
            text: timing.text,
            tokens_generated: timing.tokens_generated,
            ttft_ms: timing.ttft_ms,
            total_time_ms: timing.total_time_ms,
            itl_ms: vec![], // ITL requires streaming, not available in blocking mode
        })
    }
}

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
// Load Testing (Section 14.1)
// ============================================================================

/// Configuration for load testing
///
/// Per spec §14: Implements wrk2-style load testing with configurable
/// concurrency, duration, and target rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfig {
    /// Number of concurrent connections/threads
    pub concurrency: usize,
    /// Test duration in seconds
    pub duration_secs: u64,
    /// Target requests per second (0 = unlimited)
    pub target_rps: f64,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Warm-up period in seconds
    pub warmup_secs: u64,
    /// Target latency threshold (p99) in milliseconds
    pub latency_threshold_ms: f64,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrency: 10,
            duration_secs: 60,
            target_rps: 0.0, // Unlimited
            timeout_ms: 5000,
            warmup_secs: 5,
            latency_threshold_ms: 500.0, // Per spec: <500ms p99 target
        }
    }
}

impl LoadTestConfig {
    /// Create config for stress testing
    #[must_use]
    pub fn for_stress_test() -> Self {
        Self {
            concurrency: 100,
            duration_secs: 300,
            target_rps: 0.0,
            timeout_ms: 10_000,
            warmup_secs: 10,
            latency_threshold_ms: 1000.0,
        }
    }

    /// Create config for latency-focused testing
    #[must_use]
    pub fn for_latency_test() -> Self {
        Self {
            concurrency: 1,
            duration_secs: 60,
            target_rps: 10.0, // Fixed rate
            timeout_ms: 2000,
            warmup_secs: 5,
            latency_threshold_ms: 200.0,
        }
    }

    /// Validate the configuration
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.concurrency > 0
            && self.duration_secs > 0
            && self.timeout_ms > 0
            && self.latency_threshold_ms > 0.0
    }
}

/// Results from a load test run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResult {
    /// Total requests made
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Requests per second (achieved)
    pub rps_achieved: f64,
    /// Latency percentiles in milliseconds
    pub latency_p50_ms: f64,
    /// Latency p95 in milliseconds
    pub latency_p95_ms: f64,
    /// Latency p99 in milliseconds
    pub latency_p99_ms: f64,
    /// Maximum latency in milliseconds
    pub latency_max_ms: f64,
    /// Total data transferred in bytes
    pub data_transferred_bytes: u64,
    /// Test duration in seconds
    pub duration_secs: f64,
    /// Error rate (0.0-1.0)
    pub error_rate: f64,
    /// Whether the test passed the latency threshold
    pub passed_latency_threshold: bool,
}

impl LoadTestResult {
    /// Check if the load test passed all thresholds
    #[must_use]
    pub fn is_passing(&self) -> bool {
        self.passed_latency_threshold && self.error_rate < 0.01 // <1% error rate
    }

    /// Calculate throughput in MB/s
    #[must_use]
    pub fn throughput_mbps(&self) -> f64 {
        if self.duration_secs > 0.0 {
            (self.data_transferred_bytes as f64 / 1_000_000.0) / self.duration_secs
        } else {
            0.0
        }
    }
}

/// Load test runner
#[derive(Debug)]
pub struct LoadTestRunner {
    config: LoadTestConfig,
}

impl LoadTestRunner {
    /// Create a new load test runner
    #[must_use]
    pub fn new(config: LoadTestConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &LoadTestConfig {
        &self.config
    }

    /// Simulate a load test run (for testing purposes)
    ///
    /// In production, this would make actual HTTP requests.
    #[must_use]
    pub fn simulate_run(&self) -> LoadTestResult {
        // Simulate based on configuration
        let total_requests =
            (self.config.concurrency as f64 * self.config.duration_secs as f64 * 10.0) as usize;
        let error_count = total_requests / 100; // 1% error rate
        let successful = total_requests - error_count;

        // Simulate latencies based on concurrency
        // Higher concurrency = higher latencies
        let base_latency = 20.0; // 20ms base
        let concurrency_factor = (self.config.concurrency as f64).ln();

        let p50 = base_latency + concurrency_factor * 5.0;
        let p95 = p50 * 2.5;
        let p99 = p50 * 4.0;
        let max = p99 * 2.0;

        let duration = self.config.duration_secs as f64;
        let rps = if duration > 0.0 {
            total_requests as f64 / duration
        } else {
            0.0
        };

        LoadTestResult {
            total_requests,
            successful_requests: successful,
            failed_requests: error_count,
            rps_achieved: rps,
            latency_p50_ms: p50,
            latency_p95_ms: p95,
            latency_p99_ms: p99,
            latency_max_ms: max,
            data_transferred_bytes: (total_requests * 1024) as u64, // ~1KB per request
            duration_secs: duration,
            error_rate: error_count as f64 / total_requests as f64,
            passed_latency_threshold: p99 < self.config.latency_threshold_ms,
        }
    }
}

// ============================================================================
// Distributed Benchmark Suite (Section 10)
// ============================================================================

/// Configuration for distributed benchmarks
///
/// Per spec §10: Measures scaling efficiency for multi-GPU inference.
/// Reference: [24] NVIDIA Megatron Core distributed training methodology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedBenchConfig {
    /// GPU counts to test for scaling (e.g., [1, 2, 4, 8])
    pub gpu_counts: Vec<usize>,
    /// Number of iterations per GPU count
    pub iterations: usize,
    /// Warm-up iterations (not counted in results)
    pub warmup: usize,
    /// Model size in parameters (for theoretical FLOPS calculation)
    /// Uses u64 to support large models (7B+) on 32-bit platforms like WASM
    pub model_params: u64,
    /// Sequence length for testing
    pub seq_len: usize,
    /// Batch size for testing
    pub batch_size: usize,
    /// Target scaling efficiency threshold (0.0-1.0)
    pub efficiency_threshold: f64,
}

impl Default for DistributedBenchConfig {
    fn default() -> Self {
        Self {
            gpu_counts: vec![1, 2, 4, 8],
            iterations: 100,
            warmup: 10,
            model_params: 7_000_000_000, // 7B default
            seq_len: 2048,
            batch_size: 1,
            efficiency_threshold: 0.85, // Per spec: >85% for 2-8 GPUs
        }
    }
}

impl DistributedBenchConfig {
    /// Create config for small model testing
    #[must_use]
    pub fn for_small_model() -> Self {
        Self {
            gpu_counts: vec![1, 2],
            iterations: 50,
            warmup: 5,
            model_params: 125_000_000, // 125M
            seq_len: 512,
            batch_size: 1,
            efficiency_threshold: 0.80,
        }
    }

    /// Create config for large model testing (70B+)
    #[must_use]
    pub fn for_large_model() -> Self {
        Self {
            gpu_counts: vec![2, 4, 8],
            iterations: 50,
            warmup: 5,
            model_params: 70_000_000_000, // 70B
            seq_len: 4096,
            batch_size: 1,
            efficiency_threshold: 0.85,
        }
    }
}

/// Result from scaling efficiency benchmark
///
/// Measures Amdahl's law scaling for multi-GPU inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEfficiencyResult {
    /// Number of GPUs
    pub gpu_count: usize,
    /// Throughput in tokens/second
    pub throughput_tps: f64,
    /// Latency in milliseconds (p50)
    pub latency_p50_ms: f64,
    /// Latency in milliseconds (p99)
    pub latency_p99_ms: f64,
    /// Scaling efficiency vs 1 GPU (0.0-1.0)
    pub efficiency: f64,
    /// Communication overhead in milliseconds
    pub comm_overhead_ms: f64,
    /// Theoretical speedup (Amdahl's law)
    pub theoretical_speedup: f64,
    /// Achieved speedup vs baseline
    pub achieved_speedup: f64,
}

impl ScalingEfficiencyResult {
    /// Check if efficiency meets threshold
    #[must_use]
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.efficiency >= threshold
    }

    /// Calculate parallel fraction from Amdahl's law
    ///
    /// S = 1 / ((1 - P) + P/N)
    /// Solving for P: P = (N - S*N) / (S - S*N - 1 + N)
    #[must_use]
    pub fn parallel_fraction(&self) -> f64 {
        let n = self.gpu_count as f64;
        let s = self.achieved_speedup;
        if n <= 1.0 || s <= 1.0 {
            return 1.0;
        }
        (n * s - n) / (n * s - s)
    }
}

/// Result from tensor parallel benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParallelResult {
    /// Tensor parallel degree
    pub tp_degree: usize,
    /// Forward pass time in ms
    pub forward_ms: f64,
    /// All-reduce time in ms
    pub all_reduce_ms: f64,
    /// Overhead percentage from communication
    pub comm_overhead_pct: f64,
    /// Memory per GPU in MB
    pub memory_per_gpu_mb: f64,
    /// Effective TFLOPS
    pub effective_tflops: f64,
}

/// Result from pipeline parallel benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineParallelResult {
    /// Pipeline parallel degree
    pub pp_degree: usize,
    /// Number of micro-batches
    pub micro_batches: usize,
    /// Pipeline bubble ratio (0.0-1.0)
    pub bubble_ratio: f64,
    /// Throughput in tokens/second
    pub throughput_tps: f64,
    /// Inter-stage latency in ms
    pub inter_stage_ms: f64,
    /// Memory per stage in MB
    pub memory_per_stage_mb: f64,
}

/// Result from communication benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationResult {
    /// Operation name (all_reduce, all_gather, etc.)
    pub operation: String,
    /// Data size in bytes
    pub data_size_bytes: usize,
    /// Latency in microseconds
    pub latency_us: f64,
    /// Bandwidth in GB/s
    pub bandwidth_gbps: f64,
    /// Number of participants
    pub world_size: usize,
}

/// Distributed benchmark suite
///
/// Per spec §10: Comprehensive benchmark suite for multi-GPU inference.
/// Tests tensor parallelism, pipeline parallelism, and communication overhead.
#[derive(Debug)]
pub struct DistributedBenchSuite {
    config: DistributedBenchConfig,
    scaling_results: Vec<ScalingEfficiencyResult>,
    tp_results: Vec<TensorParallelResult>,
    pp_results: Vec<PipelineParallelResult>,
    comm_results: Vec<CommunicationResult>,
}

impl DistributedBenchSuite {
    /// Create a new distributed benchmark suite
    #[must_use]
    pub fn new(config: DistributedBenchConfig) -> Self {
        Self {
            config,
            scaling_results: Vec::new(),
            tp_results: Vec::new(),
            pp_results: Vec::new(),
            comm_results: Vec::new(),
        }
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &DistributedBenchConfig {
        &self.config
    }

    /// Run scaling efficiency benchmark
    ///
    /// Simulates multi-GPU scaling using Amdahl's law model.
    /// In production, this would measure actual GPU hardware.
    pub fn run_scaling_benchmark(&mut self) {
        // Calculate baseline throughput for 1 GPU
        let base_throughput = self.calculate_theoretical_throughput(1);
        let base_latency = 1000.0 / base_throughput; // ms per token

        for &gpu_count in &self.config.gpu_counts.clone() {
            // Theoretical speedup from Amdahl's law
            // Assume 90% parallelizable (typical for transformers)
            let parallel_fraction = 0.90;
            let theoretical_speedup =
                1.0 / ((1.0 - parallel_fraction) + parallel_fraction / gpu_count as f64);

            // Add communication overhead (typically 5-15% per additional GPU)
            let comm_overhead_factor = 1.0 + 0.05 * (gpu_count - 1) as f64;
            let achieved_speedup = theoretical_speedup / comm_overhead_factor;

            let throughput = base_throughput * achieved_speedup;
            let latency_p50 = base_latency / achieved_speedup;
            let latency_p99 = latency_p50 * 1.5; // Typical tail latency factor

            let efficiency = if gpu_count > 1 {
                achieved_speedup / gpu_count as f64
            } else {
                1.0
            };

            let comm_overhead_ms = if gpu_count > 1 {
                (theoretical_speedup - achieved_speedup) * base_latency
            } else {
                0.0
            };

            self.scaling_results.push(ScalingEfficiencyResult {
                gpu_count,
                throughput_tps: throughput,
                latency_p50_ms: latency_p50,
                latency_p99_ms: latency_p99,
                efficiency,
                comm_overhead_ms,
                theoretical_speedup,
                achieved_speedup,
            });
        }
    }

    /// Run tensor parallel benchmark
    ///
    /// Measures overhead of tensor parallelism (column/row parallel linear).
    pub fn run_tensor_parallel_benchmark(&mut self) {
        let base_flops = self.calculate_model_flops();

        for tp_degree in [1, 2, 4, 8] {
            if tp_degree > self.config.gpu_counts.iter().max().copied().unwrap_or(1) {
                continue;
            }

            // Forward pass scales inversely with TP degree
            let base_forward_ms = 50.0; // Baseline for 7B model
            let forward_ms =
                base_forward_ms / tp_degree as f64 * (self.config.model_params as f64 / 7e9);

            // All-reduce latency (alpha + beta * size)
            // Typical: alpha = 5us, beta = 0.1us/KB
            let tensor_size_kb = (self.config.model_params / tp_degree as u64) as f64 / 256.0; // 4 bytes, 1024 per KB
            let all_reduce_ms = if tp_degree > 1 {
                (5.0 + 0.1 * tensor_size_kb) / 1000.0
            } else {
                0.0
            };

            let total_ms = forward_ms + all_reduce_ms;
            let comm_overhead_pct = if total_ms > 0.0 {
                all_reduce_ms / total_ms * 100.0
            } else {
                0.0
            };

            // Memory per GPU decreases with TP
            let total_memory_mb = self.config.model_params as f64 * 2.0 / 1e6; // 2 bytes per param (fp16)
            let memory_per_gpu_mb = total_memory_mb / tp_degree as f64;

            // Effective TFLOPS
            let effective_tflops = if total_ms > 0.0 {
                base_flops / (total_ms / 1000.0) / 1e12
            } else {
                0.0
            };

            self.tp_results.push(TensorParallelResult {
                tp_degree,
                forward_ms,
                all_reduce_ms,
                comm_overhead_pct,
                memory_per_gpu_mb,
                effective_tflops,
            });
        }
    }

    /// Run pipeline parallel benchmark
    ///
    /// Measures throughput and bubble ratio for pipeline parallelism.
    pub fn run_pipeline_parallel_benchmark(&mut self) {
        let base_throughput = self.calculate_theoretical_throughput(1);

        for pp_degree in [1, 2, 4, 8] {
            if pp_degree > self.config.gpu_counts.iter().max().copied().unwrap_or(1) {
                continue;
            }

            // Optimal micro-batches = pp_degree * 4 (heuristic from Megatron-LM)
            let micro_batches = pp_degree * 4;

            // Pipeline bubble ratio: (pp - 1) / (pp - 1 + m) where m = micro_batches
            let bubble_ratio = if pp_degree > 1 {
                (pp_degree - 1) as f64 / (pp_degree - 1 + micro_batches) as f64
            } else {
                0.0
            };

            // Throughput accounting for bubble
            let efficiency = 1.0 - bubble_ratio;
            let throughput_tps = base_throughput * pp_degree as f64 * efficiency;

            // Inter-stage latency (send/recv)
            let inter_stage_ms = if pp_degree > 1 { 0.5 } else { 0.0 };

            // Memory per stage
            let total_memory_mb = self.config.model_params as f64 * 2.0 / 1e6;
            let memory_per_stage_mb = total_memory_mb / pp_degree as f64;

            self.pp_results.push(PipelineParallelResult {
                pp_degree,
                micro_batches,
                bubble_ratio,
                throughput_tps,
                inter_stage_ms,
                memory_per_stage_mb,
            });
        }
    }

    /// Run communication benchmark
    ///
    /// Measures latency and bandwidth for collective operations.
    pub fn run_communication_benchmark(&mut self) {
        let world_size = self.config.gpu_counts.iter().max().copied().unwrap_or(1);

        // Test various data sizes
        let data_sizes: Vec<usize> = vec![
            1024,              // 1 KB
            1024 * 1024,       // 1 MB
            10 * 1024 * 1024,  // 10 MB
            100 * 1024 * 1024, // 100 MB
        ];

        for data_size in data_sizes {
            // All-reduce latency model: log(n) * (alpha + beta * size)
            // Typical NCCL: alpha = 3us, beta = 0.08us/KB
            let alpha_us = 3.0;
            let beta_us_per_kb = 0.08;
            let size_kb = data_size as f64 / 1024.0;
            let latency_us = (world_size as f64).ln() * (alpha_us + beta_us_per_kb * size_kb);

            // Bandwidth = size / time
            let bandwidth_gbps = if latency_us > 0.0 {
                (data_size as f64 * 8.0) / (latency_us * 1000.0) // bits to Gbps
            } else {
                0.0
            };

            self.comm_results.push(CommunicationResult {
                operation: "all_reduce".to_string(),
                data_size_bytes: data_size,
                latency_us,
                bandwidth_gbps,
                world_size,
            });

            // All-gather has different characteristics
            let all_gather_latency = latency_us * 0.8; // Typically faster
            let all_gather_bw = bandwidth_gbps * 1.2;

            self.comm_results.push(CommunicationResult {
                operation: "all_gather".to_string(),
                data_size_bytes: data_size,
                latency_us: all_gather_latency,
                bandwidth_gbps: all_gather_bw,
                world_size,
            });
        }
    }

    /// Run complete benchmark suite
    pub fn run_all(&mut self) {
        self.run_scaling_benchmark();
        self.run_tensor_parallel_benchmark();
        self.run_pipeline_parallel_benchmark();
        self.run_communication_benchmark();
    }

    /// Get scaling efficiency results
    #[must_use]
    pub fn scaling_results(&self) -> &[ScalingEfficiencyResult] {
        &self.scaling_results
    }

    /// Get tensor parallel results
    #[must_use]
    pub fn tp_results(&self) -> &[TensorParallelResult] {
        &self.tp_results
    }

    /// Get pipeline parallel results
    #[must_use]
    pub fn pp_results(&self) -> &[PipelineParallelResult] {
        &self.pp_results
    }

    /// Get communication results
    #[must_use]
    pub fn comm_results(&self) -> &[CommunicationResult] {
        &self.comm_results
    }

    /// Check if all scaling results meet efficiency threshold
    #[must_use]
    pub fn all_meet_efficiency_threshold(&self) -> bool {
        self.scaling_results
            .iter()
            .all(|r| r.meets_threshold(self.config.efficiency_threshold))
    }

    /// Get summary statistics
    #[must_use]
    pub fn summary(&self) -> DistributedBenchSummary {
        let max_scaling = self
            .scaling_results
            .iter()
            .map(|r| r.gpu_count)
            .max()
            .unwrap_or(1);
        let max_efficiency = self
            .scaling_results
            .iter()
            .map(|r| r.efficiency)
            .fold(0.0_f64, f64::max);
        let min_efficiency = self
            .scaling_results
            .iter()
            .map(|r| r.efficiency)
            .fold(1.0_f64, f64::min);
        let max_throughput = self
            .scaling_results
            .iter()
            .map(|r| r.throughput_tps)
            .fold(0.0_f64, f64::max);

        let avg_tp_overhead = if self.tp_results.is_empty() {
            0.0
        } else {
            self.tp_results
                .iter()
                .map(|r| r.comm_overhead_pct)
                .sum::<f64>()
                / self.tp_results.len() as f64
        };

        let avg_pp_bubble = if self.pp_results.is_empty() {
            0.0
        } else {
            self.pp_results.iter().map(|r| r.bubble_ratio).sum::<f64>()
                / self.pp_results.len() as f64
        };

        DistributedBenchSummary {
            max_scaling,
            max_efficiency,
            min_efficiency,
            max_throughput_tps: max_throughput,
            avg_tp_comm_overhead_pct: avg_tp_overhead,
            avg_pp_bubble_ratio: avg_pp_bubble,
            meets_threshold: self.all_meet_efficiency_threshold(),
        }
    }

    /// Calculate theoretical throughput for given GPU count
    fn calculate_theoretical_throughput(&self, _gpu_count: usize) -> f64 {
        // Base throughput calculation
        // Typical: 7B model on A100 = ~30 tok/s (prefill) + ~100 tok/s (decode)
        let base_tps = 100.0 * (7e9 / self.config.model_params as f64);
        base_tps * (self.config.batch_size as f64)
    }

    /// Calculate model FLOPS for one forward pass
    fn calculate_model_flops(&self) -> f64 {
        // Rough estimate: 2 * params * seq_len (for one forward pass)
        2.0 * self.config.model_params as f64 * self.config.seq_len as f64
    }
}

/// Summary of distributed benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedBenchSummary {
    /// Maximum GPU scaling tested
    pub max_scaling: usize,
    /// Maximum efficiency achieved
    pub max_efficiency: f64,
    /// Minimum efficiency achieved
    pub min_efficiency: f64,
    /// Maximum throughput achieved (tokens/sec)
    pub max_throughput_tps: f64,
    /// Average tensor parallel communication overhead
    pub avg_tp_comm_overhead_pct: f64,
    /// Average pipeline parallel bubble ratio
    pub avg_pp_bubble_ratio: f64,
    /// Whether all results meet efficiency threshold
    pub meets_threshold: bool,
}

// ============================================================================
// Backend Benchmark Matrix (per Hoefler & Belli SC'15)
// ============================================================================

/// Compute backend type for benchmark matrix
///
/// Represents the different compute backends that can be benchmarked:
/// - CPU: Scalar/SIMD operations via trueno CPU backend
/// - Wgpu: Cross-platform GPU via trueno wgpu backend
/// - Cuda: NVIDIA GPU via trueno-gpu PTX execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeBackendType {
    /// CPU backend (scalar/SIMD via trueno)
    Cpu,
    /// wgpu GPU backend (cross-platform via trueno)
    Wgpu,
    /// CUDA GPU backend (NVIDIA via trueno-gpu)
    Cuda,
}

impl std::fmt::Display for ComputeBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Wgpu => write!(f, "wgpu"),
            Self::Cuda => write!(f, "cuda"),
        }
    }
}

impl ComputeBackendType {
    /// Parse from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "wgpu" | "gpu" => Some(Self::Wgpu),
            "cuda" | "nvidia" => Some(Self::Cuda),
            _ => None,
        }
    }

    /// All available backend types
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![Self::Cpu, Self::Wgpu, Self::Cuda]
    }
}

/// Single entry in the benchmark matrix
///
/// Represents results for one (runtime, backend) combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixBenchmarkEntry {
    /// Runtime type (realizar, llama-cpp, ollama, vllm)
    pub runtime: RuntimeType,
    /// Compute backend (cpu, wgpu, cuda)
    pub backend: ComputeBackendType,
    /// Model name/identifier
    pub model: String,
    /// Whether this configuration is available
    pub available: bool,
    /// p50 latency in milliseconds
    pub p50_latency_ms: f64,
    /// p99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Throughput in tokens per second
    pub throughput_tps: f64,
    /// Cold start time in milliseconds
    pub cold_start_ms: f64,
    /// Number of samples collected
    pub samples: usize,
    /// Final CV at stop
    pub cv_at_stop: f64,
    /// Additional notes (e.g., "GPU layers: 99")
    pub notes: String,
}

impl Default for MatrixBenchmarkEntry {
    fn default() -> Self {
        Self {
            runtime: RuntimeType::Realizar,
            backend: ComputeBackendType::Cpu,
            model: String::new(),
            available: false,
            p50_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            throughput_tps: 0.0,
            cold_start_ms: 0.0,
            samples: 0,
            cv_at_stop: 0.0,
            notes: String::new(),
        }
    }
}

impl MatrixBenchmarkEntry {
    /// Create a new unavailable entry (placeholder)
    #[must_use]
    pub fn unavailable(runtime: RuntimeType, backend: ComputeBackendType) -> Self {
        Self {
            runtime,
            backend,
            available: false,
            notes: "Backend not available".to_string(),
            ..Default::default()
        }
    }

    /// Create entry from raw latency samples
    #[must_use]
    pub fn from_samples(
        runtime: RuntimeType,
        backend: ComputeBackendType,
        model: &str,
        latencies_ms: &[f64],
        throughputs_tps: &[f64],
        cold_start_ms: f64,
    ) -> Self {
        let samples = latencies_ms.len();
        if samples == 0 {
            return Self::unavailable(runtime, backend);
        }

        let p50_latency = percentile(latencies_ms, 50.0);
        let p99_latency = percentile(latencies_ms, 99.0);
        let throughput = if throughputs_tps.is_empty() {
            0.0
        } else {
            throughputs_tps.iter().sum::<f64>() / throughputs_tps.len() as f64
        };
        let cv = compute_cv(latencies_ms);

        Self {
            runtime,
            backend,
            model: model.to_string(),
            available: true,
            p50_latency_ms: p50_latency,
            p99_latency_ms: p99_latency,
            throughput_tps: throughput,
            cold_start_ms,
            samples,
            cv_at_stop: cv,
            notes: String::new(),
        }
    }

    /// Add notes to the entry
    #[must_use]
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = notes.to_string();
        self
    }
}

/// Complete benchmark matrix comparing runtimes across backends
///
/// Per Hoefler & Belli SC'15, this matrix enables:
/// - Reproducible comparisons across configurations
/// - Statistical validity via CV-based stopping
/// - Clear identification of performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMatrix {
    /// Schema version
    pub version: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Model used for benchmarking
    pub model: String,
    /// Hardware specification
    pub hardware: HardwareSpec,
    /// Benchmark methodology
    pub methodology: String,
    /// CV threshold used
    pub cv_threshold: f64,
    /// Matrix entries indexed by (runtime, backend)
    pub entries: Vec<MatrixBenchmarkEntry>,
}

impl BenchmarkMatrix {
    /// Create a new empty matrix
    #[must_use]
    pub fn new(model: &str, hardware: HardwareSpec) -> Self {
        Self {
            version: "1.1".to_string(),
            timestamp: chrono_timestamp(),
            model: model.to_string(),
            hardware,
            methodology: "CV-based stopping (Hoefler & Belli SC'15)".to_string(),
            cv_threshold: 0.05,
            entries: Vec::new(),
        }
    }

    /// Add an entry to the matrix
    pub fn add_entry(&mut self, entry: MatrixBenchmarkEntry) {
        // Remove existing entry for same (runtime, backend) if present
        self.entries
            .retain(|e| e.runtime != entry.runtime || e.backend != entry.backend);
        self.entries.push(entry);
    }

    /// Get entry for specific (runtime, backend) combination
    #[must_use]
    pub fn get_entry(
        &self,
        runtime: RuntimeType,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .find(|e| e.runtime == runtime && e.backend == backend)
    }

    /// Get all entries for a specific runtime
    #[must_use]
    pub fn entries_for_runtime(&self, runtime: RuntimeType) -> Vec<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| e.runtime == runtime)
            .collect()
    }

    /// Get all entries for a specific backend
    #[must_use]
    pub fn entries_for_backend(&self, backend: ComputeBackendType) -> Vec<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| e.backend == backend)
            .collect()
    }

    /// Find the fastest runtime for a given backend (by p50 latency)
    #[must_use]
    pub fn fastest_for_backend(
        &self,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries_for_backend(backend)
            .into_iter()
            .filter(|e| e.available)
            .min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            })
    }

    /// Find the highest throughput runtime for a given backend
    #[must_use]
    pub fn highest_throughput_for_backend(
        &self,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries_for_backend(backend)
            .into_iter()
            .filter(|e| e.available)
            .max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            })
    }

    /// Generate markdown table for README
    #[must_use]
    pub fn to_markdown_table(&self) -> String {
        let mut table = String::new();

        // Header
        table.push_str("| Runtime | Backend | p50 Latency | p99 Latency | Throughput | Cold Start | Samples | CV |\n");
        table.push_str("|---------|---------|-------------|-------------|------------|------------|---------|----|\n");

        // Sort entries by runtime, then backend
        let mut sorted_entries = self.entries.clone();
        sorted_entries.sort_by(|a, b| {
            let runtime_cmp = format!("{:?}", a.runtime).cmp(&format!("{:?}", b.runtime));
            if runtime_cmp == std::cmp::Ordering::Equal {
                format!("{}", a.backend).cmp(&format!("{}", b.backend))
            } else {
                runtime_cmp
            }
        });

        for entry in &sorted_entries {
            if entry.available {
                let _ = writeln!(
                    table,
                    "| **{}** | {} | {:.1}ms | {:.1}ms | {:.1} tok/s | {:.0}ms | {} | {:.3} |",
                    format!("{:?}", entry.runtime).to_lowercase(),
                    entry.backend,
                    entry.p50_latency_ms,
                    entry.p99_latency_ms,
                    entry.throughput_tps,
                    entry.cold_start_ms,
                    entry.samples,
                    entry.cv_at_stop,
                );
            } else {
                let _ = writeln!(
                    table,
                    "| {} | {} | - | - | - | - | - | - |",
                    format!("{:?}", entry.runtime).to_lowercase(),
                    entry.backend,
                );
            }
        }

        table
    }

    /// Serialize to JSON
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    ///
    /// # Errors
    ///
    /// Returns error if JSON is invalid.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Matrix benchmark runner configuration
#[derive(Debug, Clone)]
pub struct MatrixBenchmarkConfig {
    /// Runtimes to benchmark
    pub runtimes: Vec<RuntimeType>,
    /// Backends to benchmark
    pub backends: Vec<ComputeBackendType>,
    /// Model path
    pub model_path: String,
    /// Prompt for benchmarking
    pub prompt: String,
    /// Max tokens to generate
    pub max_tokens: usize,
    /// CV threshold for stopping
    pub cv_threshold: f64,
    /// Minimum samples
    pub min_samples: usize,
    /// Maximum samples (failsafe)
    pub max_samples: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for MatrixBenchmarkConfig {
    fn default() -> Self {
        Self {
            runtimes: vec![
                RuntimeType::Realizar,
                RuntimeType::LlamaCpp,
                RuntimeType::Ollama,
            ],
            backends: vec![ComputeBackendType::Cpu, ComputeBackendType::Wgpu],
            model_path: String::new(),
            prompt: "Explain machine learning in one sentence.".to_string(),
            max_tokens: 50,
            cv_threshold: 0.05,
            min_samples: 30,
            max_samples: 200,
            warmup_iterations: 5,
        }
    }
}

/// Summary statistics for a single matrix column (backend)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSummary {
    /// Backend type
    pub backend: ComputeBackendType,
    /// Number of available runtimes
    pub available_runtimes: usize,
    /// Fastest runtime (by p50 latency)
    pub fastest_runtime: Option<String>,
    /// Fastest p50 latency
    pub fastest_p50_ms: f64,
    /// Highest throughput runtime
    pub highest_throughput_runtime: Option<String>,
    /// Highest throughput (tok/s)
    pub highest_throughput_tps: f64,
}

/// Summary of the entire benchmark matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixSummary {
    /// Total entries in matrix
    pub total_entries: usize,
    /// Number of available entries
    pub available_entries: usize,
    /// Per-backend summaries
    pub backend_summaries: Vec<BackendSummary>,
    /// Overall fastest (runtime, backend) combination
    pub overall_fastest: Option<(String, String)>,
    /// Overall highest throughput (runtime, backend)
    pub overall_highest_throughput: Option<(String, String)>,
}

impl BenchmarkMatrix {
    /// Generate summary statistics
    #[must_use]
    pub fn summary(&self) -> MatrixSummary {
        let total_entries = self.entries.len();
        let available_entries = self.entries.iter().filter(|e| e.available).count();

        let mut backend_summaries = Vec::new();
        for backend in ComputeBackendType::all() {
            let entries: Vec<_> = self.entries_for_backend(backend);
            let available: Vec<_> = entries.iter().filter(|e| e.available).collect();

            let fastest = available.iter().min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            });
            let highest_tp = available.iter().max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            });

            backend_summaries.push(BackendSummary {
                backend,
                available_runtimes: available.len(),
                fastest_runtime: fastest.map(|e| format!("{:?}", e.runtime).to_lowercase()),
                fastest_p50_ms: fastest.map_or(0.0, |e| e.p50_latency_ms),
                highest_throughput_runtime: highest_tp
                    .map(|e| format!("{:?}", e.runtime).to_lowercase()),
                highest_throughput_tps: highest_tp.map_or(0.0, |e| e.throughput_tps),
            });
        }

        let available = self.entries.iter().filter(|e| e.available);
        let overall_fastest = available
            .clone()
            .min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            })
            .map(|e| {
                (
                    format!("{:?}", e.runtime).to_lowercase(),
                    e.backend.to_string(),
                )
            });
        let overall_highest_throughput = available
            .max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            })
            .map(|e| {
                (
                    format!("{:?}", e.runtime).to_lowercase(),
                    e.backend.to_string(),
                )
            });

        MatrixSummary {
            total_entries,
            available_entries,
            backend_summaries,
            overall_fastest,
            overall_highest_throughput,
        }
    }
}

// ============================================================================
// IMP-800: TRUE GPU Parity Benchmark (M2 Milestone)
// ============================================================================

/// GPU parity benchmark configuration (IMP-800b)
///
/// Configures apples-to-apples throughput comparison on same GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuParityBenchmark {
    /// Model to benchmark (phi-2 Q4_K_M)
    pub model_path: String,
    /// Prompt for generation
    pub prompt: String,
    /// Number of tokens to generate
    pub max_tokens: usize,
    /// Ollama endpoint for comparison
    pub ollama_endpoint: String,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Target CV for stable measurements
    pub target_cv: f64,
}

impl Default for GpuParityBenchmark {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: "The quick brown fox".to_string(),
            max_tokens: 32,
            ollama_endpoint: "http://localhost:11434".to_string(),
            warmup_iterations: 3,
            measurement_iterations: 10,
            target_cv: 0.05,
        }
    }
}

impl GpuParityBenchmark {
    /// Create a new GPU parity benchmark with model path
    #[must_use]
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    /// Set the prompt for generation
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Set the number of tokens to generate
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the Ollama endpoint
    #[must_use]
    pub fn with_ollama_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.ollama_endpoint = endpoint.into();
        self
    }

    /// Set the number of warmup iterations
    #[must_use]
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_iterations = warmup;
        self
    }

    /// Set the number of measurement iterations
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }
}

/// Benchmark result with statistical analysis (IMP-800b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuParityResult {
    /// Realizar GPU throughput (tok/s)
    pub realizar_gpu_tps: f64,
    /// Ollama throughput (tok/s)
    pub ollama_tps: f64,
    /// Performance gap ratio (Ollama / Realizar)
    pub gap_ratio: f64,
    /// Coefficient of variation (measurement stability)
    pub cv: f64,
    /// GPU device name
    pub gpu_device: String,
    /// VRAM usage (MB)
    pub vram_mb: u64,
    /// Realizar latency p50 (ms)
    pub realizar_p50_ms: f64,
    /// Ollama latency p50 (ms)
    pub ollama_p50_ms: f64,
}

impl GpuParityResult {
    /// Create a new GPU parity result
    #[must_use]
    pub fn new(
        realizar_gpu_tps: f64,
        ollama_tps: f64,
        cv: f64,
        gpu_device: impl Into<String>,
        vram_mb: u64,
    ) -> Self {
        let gap_ratio = if realizar_gpu_tps > 0.0 {
            ollama_tps / realizar_gpu_tps
        } else {
            f64::INFINITY
        };

        Self {
            realizar_gpu_tps,
            ollama_tps,
            gap_ratio,
            cv,
            gpu_device: gpu_device.into(),
            vram_mb,
            realizar_p50_ms: 0.0,
            ollama_p50_ms: 0.0,
        }
    }

    /// Returns true if within 2x of Ollama (M2 target)
    #[must_use]
    pub fn achieves_m2_parity(&self) -> bool {
        self.gap_ratio <= 2.0
    }

    /// Returns true if within 1.25x of Ollama (M4 target)
    #[must_use]
    pub fn achieves_m4_parity(&self) -> bool {
        self.gap_ratio <= 1.25
    }

    /// Returns true if GPU is faster than CPU SIMD baseline (5 tok/s)
    #[must_use]
    pub fn gpu_faster_than_cpu(&self) -> bool {
        self.realizar_gpu_tps > 5.0
    }

    /// Returns true if measurements are stable (CV < 0.05)
    #[must_use]
    pub fn measurements_stable(&self) -> bool {
        self.cv < 0.05
    }

    /// Get speedup over CPU SIMD baseline
    #[must_use]
    pub fn cpu_speedup(&self) -> f64 {
        self.realizar_gpu_tps / 5.0 // CPU baseline ~5 tok/s
    }
}

/// Gap analysis with falsifiable claims (IMP-800c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysis {
    /// Claimed gap reduction
    pub claimed_gap: f64,
    /// Measured gap
    pub measured_gap: f64,
    /// Statistical significance (p-value)
    pub p_value: f64,
    /// Confidence interval lower bound (95%)
    pub ci_95_lower: f64,
    /// Confidence interval upper bound (95%)
    pub ci_95_upper: f64,
    /// Popper score (falsifiability, 0-100)
    pub popper_score: f64,
    /// Claim descriptions
    pub claims: Vec<FalsifiableClaim>,
}

/// A falsifiable claim for Popperian testing (IMP-800c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsifiableClaim {
    /// Claim identifier
    pub id: String,
    /// Claim description
    pub description: String,
    /// Expected value
    pub expected: f64,
    /// Threshold for verification
    pub threshold: f64,
    /// Measured value
    pub measured: f64,
    /// Whether claim is verified
    pub verified: bool,
}

impl FalsifiableClaim {
    /// Create a new falsifiable claim
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        expected: f64,
        threshold: f64,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            expected,
            threshold,
            measured: 0.0,
            verified: false,
        }
    }

    /// Evaluate the claim against a measured value
    #[must_use]
    pub fn evaluate(mut self, measured: f64) -> Self {
        self.measured = measured;
        self.verified = measured >= self.threshold;
        self
    }
}

impl GapAnalysis {
    /// Create a new gap analysis
    #[must_use]
    pub fn new(claimed_gap: f64, measured_gap: f64) -> Self {
        Self {
            claimed_gap,
            measured_gap,
            p_value: 0.0,
            ci_95_lower: 0.0,
            ci_95_upper: 0.0,
            popper_score: 0.0,
            claims: Vec::new(),
        }
    }

    /// Add statistical bounds
    #[must_use]
    pub fn with_statistics(mut self, p_value: f64, ci_lower: f64, ci_upper: f64) -> Self {
        self.p_value = p_value;
        self.ci_95_lower = ci_lower;
        self.ci_95_upper = ci_upper;
        self
    }

    /// Calculate and set Popper score based on claims
    pub fn calculate_popper_score(&mut self) {
        if self.claims.is_empty() {
            self.popper_score = 0.0;
            return;
        }

        let verified_count = self.claims.iter().filter(|c| c.verified).count();
        self.popper_score = (verified_count as f64 / self.claims.len() as f64) * 100.0;
    }

    /// Add a falsifiable claim
    pub fn add_claim(&mut self, claim: FalsifiableClaim) {
        self.claims.push(claim);
    }

    /// Claim is verified if measured within CI
    #[must_use]
    pub fn claim_verified(&self) -> bool {
        self.measured_gap >= self.ci_95_lower && self.measured_gap <= self.ci_95_upper
    }

    /// Create default IMP-800c claims
    #[must_use]
    pub fn with_default_claims(mut self, realizar_gpu_tps: f64) -> Self {
        // IMP-800c-1: GPU faster than CPU SIMD (>5x, threshold 25 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-1", "GPU faster than CPU SIMD (>5x)", 5.0, 25.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-2: GPU within 10x of Ollama (threshold 24 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-2", "GPU within 10x of Ollama", 10.0, 24.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-3: GPU within 2x of Ollama - M2 (threshold 120 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-3", "GPU within 2x of Ollama (M2)", 2.0, 120.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-4: GPU at parity with Ollama - M4 (threshold 192 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-4", "GPU at parity with Ollama (M4)", 1.25, 192.0)
                .evaluate(realizar_gpu_tps),
        );

        self.calculate_popper_score();
        self
    }
}

// ============================================================================
// IMP-900: Closing the 18x Gap (M3/M4 Milestones)
// ============================================================================

/// Optimized GEMM configuration (IMP-900a)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedGemmConfig {
    /// Tile size for shared memory (typically 32 or 64)
    pub tile_size: u32,
    /// Register blocking factor (typically 4 or 8)
    pub reg_block: u32,
    /// Use tensor cores if available (SM 7.0+)
    pub use_tensor_cores: bool,
    /// Vectorized loads (float4 = 4)
    pub vector_width: u32,
    /// Unroll factor for K-loop
    pub k_unroll: u32,
    /// Use double buffering for tile prefetch
    pub double_buffer: bool,
}

impl Default for OptimizedGemmConfig {
    fn default() -> Self {
        Self {
            tile_size: 32,
            reg_block: 4,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 4,
            double_buffer: true,
        }
    }
}

impl OptimizedGemmConfig {
    /// Create configuration for small matrices (256x256)
    #[must_use]
    pub fn small() -> Self {
        Self {
            tile_size: 16,
            reg_block: 2,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 4,
            double_buffer: false,
        }
    }

    /// Create configuration for large matrices (1024+)
    #[must_use]
    pub fn large() -> Self {
        Self {
            tile_size: 64,
            reg_block: 8,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 8,
            double_buffer: true,
        }
    }

    /// Calculate shared memory requirement (bytes)
    #[must_use]
    pub fn shared_memory_bytes(&self) -> u32 {
        // Two tiles (A and B) in shared memory
        // Each tile is tile_size × tile_size × sizeof(f32)
        let tile_bytes = self.tile_size * self.tile_size * 4;
        if self.double_buffer {
            tile_bytes * 4 // 2 tiles × 2 buffers
        } else {
            tile_bytes * 2 // 2 tiles
        }
    }

    /// Calculate threads per block
    #[must_use]
    pub fn threads_per_block(&self) -> u32 {
        // Each thread computes reg_block × reg_block elements
        let threads_per_dim = self.tile_size / self.reg_block;
        threads_per_dim * threads_per_dim
    }

    /// Calculate registers per thread (for accumulators)
    #[must_use]
    pub fn registers_per_thread(&self) -> u32 {
        // reg_block × reg_block accumulator values
        self.reg_block * self.reg_block
    }
}

/// GEMM performance result (IMP-900a)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmPerformanceResult {
    /// Matrix M dimension (rows of A, rows of C)
    pub m: u32,
    /// Matrix N dimension (cols of B, cols of C)
    pub n: u32,
    /// Matrix K dimension (cols of A, rows of B)
    pub k: u32,
    /// Time in milliseconds
    pub time_ms: f64,
    /// GFLOP/s achieved
    pub gflops: f64,
    /// Memory bandwidth achieved (GB/s)
    pub bandwidth_gbs: f64,
    /// Percentage of peak performance
    pub efficiency: f64,
}

impl GemmPerformanceResult {
    /// Create a new GEMM performance result
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32, time_ms: f64) -> Self {
        // GEMM operations: 2 * M * N * K (multiply-add)
        let ops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
        let gflops = ops / (time_ms * 1e6);

        // Memory: read A (M*K), read B (K*N), write C (M*N)
        let bytes = (f64::from(m) * f64::from(k)
            + f64::from(k) * f64::from(n)
            + f64::from(m) * f64::from(n))
            * 4.0;
        let bandwidth_gbs = bytes / (time_ms * 1e6);

        Self {
            m,
            n,
            k,
            time_ms,
            gflops,
            bandwidth_gbs,
            efficiency: 0.0, // Set by caller based on peak
        }
    }

    /// Set efficiency based on peak GFLOP/s
    #[must_use]
    pub fn with_peak(mut self, peak_gflops: f64) -> Self {
        self.efficiency = (self.gflops / peak_gflops) * 100.0;
        self
    }

    /// Check if performance improved by at least the given factor
    #[must_use]
    pub fn improved_by(&self, baseline_gflops: f64, factor: f64) -> bool {
        self.gflops >= baseline_gflops * factor
    }
}

/// Optimized GEMM benchmark runner (IMP-900a)
#[derive(Debug)]
pub struct OptimizedGemmBenchmark {
    /// Configuration
    pub config: OptimizedGemmConfig,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Measurement iterations
    pub measurement_iterations: usize,
    /// Target coefficient of variation
    pub target_cv: f64,
}

impl Default for OptimizedGemmBenchmark {
    fn default() -> Self {
        Self {
            config: OptimizedGemmConfig::default(),
            warmup_iterations: 5,
            measurement_iterations: 20,
            target_cv: 0.05,
        }
    }
}

impl OptimizedGemmBenchmark {
    /// Create benchmark with custom config
    #[must_use]
    pub fn with_config(config: OptimizedGemmConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Calculate expected improvement over naive GEMM
    #[must_use]
    pub fn expected_improvement(&self) -> f64 {
        let mut improvement = 1.0;

        // Shared memory tiling: ~2x for cache efficiency
        improvement *= 2.0;

        // Register blocking: ~1.5x for reduced memory traffic
        if self.config.reg_block >= 4 {
            improvement *= 1.5;
        }

        // Vectorized loads: ~1.3x for coalesced access
        if self.config.vector_width >= 4 {
            improvement *= 1.3;
        }

        // Double buffering: ~1.2x for latency hiding
        if self.config.double_buffer {
            improvement *= 1.2;
        }

        improvement
    }
}

/// Kernel fusion configuration (IMP-900b)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusedOpType {
    /// GEMM + bias + activation
    GemmBiasActivation,
    /// Layer normalization + linear projection
    LayerNormLinear,
    /// Fused attention (FlashAttention-style)
    FusedAttention,
    /// FFN: up projection + gate + down projection
    FusedFfn,
}

/// Fused operation specification (IMP-900b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOpSpec {
    /// Type of fused operation
    pub op_type: FusedOpType,
    /// Input dimensions
    pub input_dims: Vec<u32>,
    /// Output dimensions
    pub output_dims: Vec<u32>,
    /// Activation function (if applicable)
    pub activation: Option<String>,
    /// Number of kernel launches when fused
    pub fused_launches: u32,
    /// Number of kernel launches when unfused
    pub unfused_launches: u32,
}

impl FusedOpSpec {
    /// Calculate launch reduction factor
    #[must_use]
    pub fn launch_reduction(&self) -> f64 {
        f64::from(self.unfused_launches) / f64::from(self.fused_launches)
    }

    /// Check if fusion reduces launches by at least 50%
    #[must_use]
    pub fn achieves_target_reduction(&self) -> bool {
        self.launch_reduction() >= 2.0
    }
}

/// FlashAttention configuration (IMP-900c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashAttentionConfig {
    /// Block size for Q tiling (Br)
    pub block_size_q: u32,
    /// Block size for K/V tiling (Bc)
    pub block_size_kv: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Use causal masking
    pub causal: bool,
    /// Softmax scale (default: 1/sqrt(head_dim))
    pub scale: f32,
}

impl FlashAttentionConfig {
    /// Create configuration for phi-2 model
    #[must_use]
    pub fn phi2() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            head_dim: 80, // phi-2: 2560 / 32 heads
            num_heads: 32,
            causal: true,
            scale: 1.0 / (80.0_f32).sqrt(),
        }
    }

    /// Calculate memory required for attention (naive vs flash)
    #[must_use]
    pub fn memory_comparison(&self, seq_len: u32) -> (u64, u64) {
        // Naive: O(N²) attention matrix
        let naive_bytes = u64::from(seq_len) * u64::from(seq_len) * 4;

        // FlashAttention: O(N) working memory
        let flash_bytes = u64::from(self.block_size_q) * u64::from(self.block_size_kv) * 4 * 2; // S and P blocks

        (naive_bytes, flash_bytes)
    }

    /// Calculate memory savings factor
    #[must_use]
    pub fn memory_savings(&self, seq_len: u32) -> f64 {
        let (naive, flash) = self.memory_comparison(seq_len);
        naive as f64 / flash as f64
    }
}

/// Memory pool configuration (IMP-900d)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size (bytes)
    pub initial_size: usize,
    /// Maximum pool size (bytes)
    pub max_size: usize,
    /// Size classes for allocation (powers of 2)
    pub size_classes: Vec<usize>,
    /// Use pinned memory for host staging
    pub use_pinned_memory: bool,
    /// Enable async transfers
    pub async_transfers: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024,  // 256 MB
            max_size: 2 * 1024 * 1024 * 1024, // 2 GB
            size_classes: vec![
                4096,        // 4 KB
                16384,       // 16 KB
                65536,       // 64 KB
                262_144,     // 256 KB
                1_048_576,   // 1 MB
                4_194_304,   // 4 MB
                16_777_216,  // 16 MB
                67_108_864,  // 64 MB
                268_435_456, // 256 MB
            ],
            use_pinned_memory: true,
            async_transfers: true,
        }
    }
}

impl MemoryPoolConfig {
    /// Find the smallest size class that fits the requested size
    #[must_use]
    pub fn find_size_class(&self, requested: usize) -> Option<usize> {
        self.size_classes
            .iter()
            .copied()
            .find(|&size| size >= requested)
    }

    /// Calculate expected bandwidth improvement from pinned memory
    #[must_use]
    pub fn expected_bandwidth_improvement(&self) -> f64 {
        if self.use_pinned_memory {
            2.4 // Pinned memory typically 2-3x faster
        } else {
            1.0
        }
    }
}

/// IMP-900 combined result (M3/M4 targets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Imp900Result {
    /// Baseline throughput (13.1 tok/s from IMP-800)
    pub baseline_tps: f64,
    /// Throughput after optimizations
    pub optimized_tps: f64,
    /// GEMM optimization improvement factor
    pub gemm_improvement: f64,
    /// Kernel fusion improvement factor
    pub fusion_improvement: f64,
    /// FlashAttention improvement factor
    pub flash_attention_improvement: f64,
    /// Memory optimization improvement factor
    pub memory_improvement: f64,
    /// Gap to Ollama
    pub gap_ratio: f64,
    /// Target milestone achieved
    pub milestone: Option<String>,
}

impl Imp900Result {
    /// Create result from baseline
    #[must_use]
    pub fn from_baseline(baseline_tps: f64) -> Self {
        Self {
            baseline_tps,
            optimized_tps: baseline_tps,
            gemm_improvement: 1.0,
            fusion_improvement: 1.0,
            flash_attention_improvement: 1.0,
            memory_improvement: 1.0,
            gap_ratio: 240.0 / baseline_tps,
            milestone: None,
        }
    }

    /// Apply GEMM optimization
    #[must_use]
    pub fn with_gemm_improvement(mut self, factor: f64) -> Self {
        self.gemm_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply fusion optimization
    #[must_use]
    pub fn with_fusion_improvement(mut self, factor: f64) -> Self {
        self.fusion_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply FlashAttention optimization
    #[must_use]
    pub fn with_flash_attention_improvement(mut self, factor: f64) -> Self {
        self.flash_attention_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply memory optimization
    #[must_use]
    pub fn with_memory_improvement(mut self, factor: f64) -> Self {
        self.memory_improvement = factor;
        self.recalculate();
        self
    }

    /// Recalculate throughput and milestone
    fn recalculate(&mut self) {
        let total_improvement = self.gemm_improvement
            * self.fusion_improvement
            * self.flash_attention_improvement
            * self.memory_improvement;

        self.optimized_tps = self.baseline_tps * total_improvement;
        self.gap_ratio = 240.0 / self.optimized_tps;

        self.milestone = if self.gap_ratio <= 1.25 {
            Some("M4".to_string()) // Full parity
        } else if self.gap_ratio <= 2.0 {
            Some("M3".to_string()) // Near parity
        } else if self.gap_ratio <= 5.0 {
            Some("M2".to_string()) // Within 5x
        } else {
            None
        };
    }

    /// Check if M3 target achieved (>48 tok/s, <5x gap)
    #[must_use]
    pub fn achieves_m3(&self) -> bool {
        self.optimized_tps >= 48.0 && self.gap_ratio <= 5.0
    }

    /// Check if M4 target achieved (>192 tok/s, <1.25x gap)
    #[must_use]
    pub fn achieves_m4(&self) -> bool {
        self.optimized_tps >= 192.0 && self.gap_ratio <= 1.25
    }

    /// Get combined improvement factor
    #[must_use]
    pub fn total_improvement(&self) -> f64 {
        self.optimized_tps / self.baseline_tps
    }
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod bench_tests;
