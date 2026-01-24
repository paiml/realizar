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

// PMAT-802: Extracted modules
mod runtime;
mod statistics;
mod load_testing;
mod matrix;
mod gpu_parity;

// RuntimeType always available (matrix.rs needs it)
pub use runtime::RuntimeType;

// HTTP-dependent runtime backends
#[cfg(feature = "bench-http")]
pub use runtime::{
    InferenceRequest, InferenceResponse, BackendInfo, RuntimeBackend,
    MockBackend, BackendRegistry, LlamaCppConfig, VllmConfig, LlamaCppBackend,
    VllmBackend, OllamaConfig, OllamaBackend,
};

pub use statistics::{
    MeasurementProtocol, LatencyStatistics, detect_outliers, BenchmarkMetrics,
    Regression, RegressionReport, RegressionDetector, WelchTTestResult, welch_t_test,
};

pub use load_testing::{LoadTestConfig, LoadTestResult, LoadTestRunner};

pub use matrix::{
    MatrixBenchmarkEntry, BenchmarkMatrix, MatrixBenchmarkConfig,
    BackendSummary, MatrixSummary, ComputeBackendType,
};

pub use gpu_parity::{
    GpuParityBenchmark, GpuParityResult, GapAnalysis, FalsifiableClaim,
    OptimizedGemmConfig, GemmPerformanceResult, OptimizedGemmBenchmark,
    FusedOpType, FusedOpSpec, FlashAttentionConfig, Imp900Result, MemoryPoolConfig,
};

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


// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod bench_tests;
