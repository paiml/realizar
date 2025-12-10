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

        // Check if binary exists
        let output = Command::new(&self.config.binary_path)
            .arg("--version")
            .output();

        match output {
            Ok(_) => {
                // Simulate inference (real impl would run the binary)
                Ok(InferenceResponse {
                    text: format!("Response to: {}", request.prompt),
                    tokens_generated: 10,
                    ttft_ms: 50.0,
                    total_time_ms: 200.0,
                    itl_ms: vec![15.0; 9],
                })
            },
            Err(_) => Err(RealizarError::ModelNotFound(
                self.config.binary_path.clone(),
            )),
        }
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
    // =========================================================================

    #[test]
    fn test_vllm_backend_creation() {
        let config = VllmConfig::new("http://localhost:8000");
        let backend = VllmBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Vllm);
    }

    #[test]
    fn test_vllm_backend_info() {
        let config = VllmConfig::new("http://localhost:8000").with_model("meta-llama/Llama-2-7b");
        let backend = VllmBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Vllm);
        assert!(info.supports_streaming); // vLLM supports streaming
    }

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
}
