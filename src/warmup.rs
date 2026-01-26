//! Model Warm-up and Pre-loading
//!
//! Reduces cold start latency by pre-loading models and running warm-up inference.
//!
//! ## Features
//!
//! - Pre-load models into memory before serving
//! - Run warm-up inference to JIT compile and optimize
//! - Validate model integrity before accepting traffic
//! - Background model loading for zero-downtime updates
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::warmup::{WarmupConfig, ModelWarmer};
//!
//! let config = WarmupConfig::new()
//!     .with_warmup_iterations(3)
//!     .with_timeout(Duration::from_secs(30));
//!
//! let warmer = ModelWarmer::new(config);
//! warmer.warm_up(&model, &tokenizer).await?;
//! ```
//!
//! ## Toyota Way Principles
//!
//! - Heijunka: Level loading by pre-warming
//! - Jidoka: Validate model before serving
//! - Poka-Yoke: Prevent cold start errors

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

// ============================================================================
// WARM-001: Configuration
// ============================================================================

/// Configuration for model warm-up
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Number of warm-up inference iterations
    pub warmup_iterations: usize,
    /// Timeout for warm-up process
    pub timeout: Duration,
    /// Sample prompt for warm-up inference
    pub sample_prompt: String,
    /// Maximum tokens for warm-up generation
    pub sample_max_tokens: usize,
    /// Validate model output during warm-up
    pub validate_output: bool,
    /// Run garbage collection after warm-up
    pub gc_after_warmup: bool,
    /// Log warm-up progress
    pub verbose: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            timeout: Duration::from_secs(60),
            sample_prompt: "Hello, world!".to_string(),
            sample_max_tokens: 10,
            validate_output: true,
            gc_after_warmup: true,
            verbose: false,
        }
    }
}

impl WarmupConfig {
    /// Create a new warm-up configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of warm-up iterations
    #[must_use]
    pub fn with_warmup_iterations(mut self, n: usize) -> Self {
        self.warmup_iterations = n.max(1);
        self
    }

    /// Set warm-up timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set sample prompt for warm-up
    #[must_use]
    pub fn with_sample_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.sample_prompt = prompt.into();
        self
    }

    /// Set maximum tokens for warm-up generation
    #[must_use]
    pub fn with_sample_max_tokens(mut self, n: usize) -> Self {
        self.sample_max_tokens = n;
        self
    }

    /// Enable/disable output validation
    #[must_use]
    pub fn with_validate_output(mut self, validate: bool) -> Self {
        self.validate_output = validate;
        self
    }

    /// Enable/disable garbage collection after warm-up
    #[must_use]
    pub fn with_gc_after_warmup(mut self, gc: bool) -> Self {
        self.gc_after_warmup = gc;
        self
    }

    /// Enable/disable verbose logging
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

// ============================================================================
// WARM-002: Warm-up Status
// ============================================================================

/// Status of warm-up process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarmupStatus {
    /// Not yet started
    NotStarted,
    /// Currently warming up
    InProgress,
    /// Warm-up completed successfully
    Ready,
    /// Warm-up failed
    Failed,
    /// Warm-up timed out
    TimedOut,
}

impl WarmupStatus {
    /// Check if model is ready to serve
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Check if warm-up is still in progress
    #[must_use]
    pub fn is_in_progress(&self) -> bool {
        matches!(self, Self::InProgress)
    }

    /// Check if warm-up failed
    #[must_use]
    pub fn has_failed(&self) -> bool {
        matches!(self, Self::Failed | Self::TimedOut)
    }
}

// ============================================================================
// WARM-003: Warm-up Result
// ============================================================================

/// Result of warm-up process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupResult {
    /// Final status
    pub status: WarmupStatus,
    /// Number of iterations completed
    pub iterations_completed: usize,
    /// Total warm-up duration
    pub total_duration: Duration,
    /// Average inference latency during warm-up
    pub avg_latency: Duration,
    /// First inference latency (cold)
    pub first_latency: Duration,
    /// Last inference latency (warm)
    pub last_latency: Duration,
    /// Speedup factor (first / last)
    pub speedup_factor: f64,
    /// Error message if failed
    pub error: Option<String>,
}

impl WarmupResult {
    /// Create a successful result
    #[must_use]
    pub fn success(iterations: usize, duration: Duration, latencies: &[Duration]) -> Self {
        let first = latencies.first().copied().unwrap_or(Duration::ZERO);
        let last = latencies.last().copied().unwrap_or(Duration::ZERO);
        let avg = if latencies.is_empty() {
            Duration::ZERO
        } else {
            Duration::from_nanos(
                latencies.iter().map(|d| d.as_nanos() as u64).sum::<u64>() / latencies.len() as u64,
            )
        };

        let speedup = if last.as_nanos() > 0 {
            first.as_nanos() as f64 / last.as_nanos() as f64
        } else {
            1.0
        };

        Self {
            status: WarmupStatus::Ready,
            iterations_completed: iterations,
            total_duration: duration,
            avg_latency: avg,
            first_latency: first,
            last_latency: last,
            speedup_factor: speedup,
            error: None,
        }
    }

    /// Create a failed result
    #[must_use]
    pub fn failed(error: impl Into<String>, iterations: usize, duration: Duration) -> Self {
        Self {
            status: WarmupStatus::Failed,
            iterations_completed: iterations,
            total_duration: duration,
            avg_latency: Duration::ZERO,
            first_latency: Duration::ZERO,
            last_latency: Duration::ZERO,
            speedup_factor: 0.0,
            error: Some(error.into()),
        }
    }

    /// Create a timed out result
    #[must_use]
    pub fn timed_out(iterations: usize, duration: Duration) -> Self {
        Self {
            status: WarmupStatus::TimedOut,
            iterations_completed: iterations,
            total_duration: duration,
            avg_latency: Duration::ZERO,
            first_latency: Duration::ZERO,
            last_latency: Duration::ZERO,
            speedup_factor: 0.0,
            error: Some("Warm-up timed out".to_string()),
        }
    }
}

// ============================================================================
// WARM-004: Model Health
// ============================================================================

/// Model health status for readiness probes
#[derive(Debug, Clone)]
pub struct ModelHealth {
    /// Whether model is ready
    ready: Arc<AtomicBool>,
    /// Warm-up status
    status: Arc<std::sync::RwLock<WarmupStatus>>,
    /// Total requests served
    requests_served: Arc<AtomicU64>,
    /// Failed requests
    requests_failed: Arc<AtomicU64>,
    /// Last health check timestamp
    last_health_check: Arc<std::sync::RwLock<Instant>>,
    /// Model load timestamp
    loaded_at: Instant,
}

impl Default for ModelHealth {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelHealth {
    /// Create new health tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
            status: Arc::new(std::sync::RwLock::new(WarmupStatus::NotStarted)),
            requests_served: Arc::new(AtomicU64::new(0)),
            requests_failed: Arc::new(AtomicU64::new(0)),
            last_health_check: Arc::new(std::sync::RwLock::new(Instant::now())),
            loaded_at: Instant::now(),
        }
    }

    /// Check if model is ready
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Set ready status
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    /// Get current status
    #[must_use]
    pub fn status(&self) -> WarmupStatus {
        *self.status.read().expect("test")
    }

    /// Set status
    pub fn set_status(&self, status: WarmupStatus) {
        *self.status.write().expect("test") = status;
        if status == WarmupStatus::Ready {
            self.set_ready(true);
        }
    }

    /// Record a successful request
    pub fn record_success(&self) {
        self.requests_served.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total requests served
    #[must_use]
    pub fn total_requests(&self) -> u64 {
        self.requests_served.load(Ordering::Relaxed)
    }

    /// Get failed requests
    #[must_use]
    pub fn failed_requests(&self) -> u64 {
        self.requests_failed.load(Ordering::Relaxed)
    }

    /// Get error rate
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total = self.total_requests();
        let failed = self.failed_requests();
        if total == 0 {
            0.0
        } else {
            failed as f64 / total as f64
        }
    }

    /// Update health check timestamp
    pub fn touch(&self) {
        *self.last_health_check.write().expect("test") = Instant::now();
    }

    /// Get uptime since model was loaded
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.loaded_at.elapsed()
    }

    /// Get time since last health check
    #[must_use]
    pub fn time_since_last_check(&self) -> Duration {
        self.last_health_check.read().expect("test").elapsed()
    }

    /// Generate health report
    #[must_use]
    pub fn report(&self) -> HealthReport {
        HealthReport {
            ready: self.is_ready(),
            status: self.status(),
            uptime_secs: self.uptime().as_secs_f64(),
            total_requests: self.total_requests(),
            failed_requests: self.failed_requests(),
            error_rate: self.error_rate(),
            time_since_last_check_secs: self.time_since_last_check().as_secs_f64(),
        }
    }
}

/// Health report for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Is model ready to serve
    pub ready: bool,
    /// Current warm-up status
    pub status: WarmupStatus,
    /// Uptime in seconds
    pub uptime_secs: f64,
    /// Total requests served
    pub total_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Error rate (0.0 - 1.0)
    pub error_rate: f64,
    /// Time since last health check in seconds
    pub time_since_last_check_secs: f64,
}

// ============================================================================
// WARM-005: Warm-up Executor
// ============================================================================

/// Executes model warm-up process
#[derive(Debug, Clone)]
pub struct WarmupExecutor {
    config: WarmupConfig,
}

impl WarmupExecutor {
    /// Create a new warm-up executor
    #[must_use]
    pub fn new(config: WarmupConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &WarmupConfig {
        &self.config
    }

    /// Simulate warm-up (for testing without actual model)
    ///
    /// This runs the warm-up process with test inference delays.
    #[must_use]
    pub fn simulate_warmup(&self) -> WarmupResult {
        let start = Instant::now();
        let mut latencies = Vec::with_capacity(self.config.warmup_iterations);

        // Simulate decreasing latency as model warms up
        for i in 0..self.config.warmup_iterations {
            // First iteration is "cold" (slower)
            let base_latency_us = if i == 0 { 1000 } else { 100 };
            let jitter = (i * 10) as u64;
            let latency = Duration::from_micros(base_latency_us - jitter.min(50));
            latencies.push(latency);
        }

        WarmupResult::success(self.config.warmup_iterations, start.elapsed(), &latencies)
    }

    /// Check if timeout has been exceeded
    #[allow(dead_code)]
    fn check_timeout(&self, start: Instant, iterations: usize) -> Option<WarmupResult> {
        if start.elapsed() > self.config.timeout {
            Some(WarmupResult::timed_out(iterations, start.elapsed()))
        } else {
            None
        }
    }
}

impl Default for WarmupExecutor {
    fn default() -> Self {
        Self::new(WarmupConfig::default())
    }
}

// ============================================================================
// WARM-006: Pre-load Configuration
// ============================================================================

/// Configuration for model pre-loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreloadConfig {
    /// Models to pre-load on startup
    pub models: Vec<PreloadModelConfig>,
    /// Load models in parallel
    pub parallel_loading: bool,
    /// Maximum concurrent model loads
    pub max_concurrent: usize,
    /// Fail startup if any model fails to load
    pub fail_fast: bool,
}

impl Default for PreloadConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            parallel_loading: true,
            max_concurrent: 4,
            fail_fast: false,
        }
    }
}

impl PreloadConfig {
    /// Create new pre-load configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a model to pre-load
    #[must_use]
    pub fn with_model(mut self, model: PreloadModelConfig) -> Self {
        self.models.push(model);
        self
    }

    /// Set parallel loading
    #[must_use]
    pub fn with_parallel_loading(mut self, parallel: bool) -> Self {
        self.parallel_loading = parallel;
        self
    }

    /// Set maximum concurrent loads
    #[must_use]
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }

    /// Set fail-fast behavior
    #[must_use]
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }
}

/// Configuration for a single model to pre-load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreloadModelConfig {
    /// Model identifier
    pub model_id: String,
    /// Model URI (pacha://, file://, hf://)
    pub uri: String,
    /// Priority (lower = load first)
    pub priority: u32,
    /// Run warm-up after loading
    pub warmup: bool,
    /// Warm-up configuration
    pub warmup_config: Option<WarmupConfig>,
}

impl PreloadModelConfig {
    /// Create new model pre-load config
    #[must_use]
    pub fn new(model_id: impl Into<String>, uri: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            uri: uri.into(),
            priority: 100,
            warmup: true,
            warmup_config: None,
        }
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Enable/disable warm-up
    #[must_use]
    pub fn with_warmup(mut self, warmup: bool) -> Self {
        self.warmup = warmup;
        self
    }

    /// Set warm-up configuration
    #[must_use]
    pub fn with_warmup_config(mut self, config: WarmupConfig) -> Self {
        self.warmup_config = Some(config);
        self.warmup = true;
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // WARM-001: Configuration Tests
    #[test]
    fn test_warmup_config_default_and_builder() {
        let def = WarmupConfig::default();
        assert_eq!(def.warmup_iterations, 3);
        assert_eq!(def.timeout, Duration::from_secs(60));
        assert!(def.validate_output);

        let built = WarmupConfig::new()
            .with_warmup_iterations(5)
            .with_timeout(Duration::from_secs(120))
            .with_sample_prompt("Test")
            .with_sample_max_tokens(20)
            .with_validate_output(false)
            .with_gc_after_warmup(false)
            .with_verbose(true);
        assert_eq!(built.warmup_iterations, 5);
        assert_eq!(built.sample_prompt, "Test");
        assert!(!built.validate_output);
        assert!(!built.gc_after_warmup);
        assert!(built.verbose);

        // Min iterations clamping
        assert_eq!(
            WarmupConfig::new()
                .with_warmup_iterations(0)
                .warmup_iterations,
            1
        );
    }

    // WARM-002: Status Tests
    #[test]
    fn test_warmup_status_methods() {
        assert!(!WarmupStatus::NotStarted.is_ready());
        assert!(WarmupStatus::Ready.is_ready());
        assert!(WarmupStatus::InProgress.is_in_progress());
        assert!(!WarmupStatus::Ready.is_in_progress());
        assert!(WarmupStatus::Failed.has_failed());
        assert!(WarmupStatus::TimedOut.has_failed());
        assert!(!WarmupStatus::Ready.has_failed());
    }

    // WARM-003: Result Tests
    #[test]
    fn test_warmup_result_variants() {
        // Success with latencies
        let latencies = vec![
            Duration::from_millis(100),
            Duration::from_millis(50),
            Duration::from_millis(25),
        ];
        let success = WarmupResult::success(3, Duration::from_millis(200), &latencies);
        assert_eq!(success.status, WarmupStatus::Ready);
        assert_eq!(success.first_latency, Duration::from_millis(100));
        assert_eq!(success.last_latency, Duration::from_millis(25));
        assert!(success.speedup_factor > 1.0);
        assert!(success.error.is_none());

        // Failed
        let failed = WarmupResult::failed("Test error", 2, Duration::from_secs(5));
        assert_eq!(failed.status, WarmupStatus::Failed);
        assert_eq!(failed.error, Some("Test error".to_string()));

        // Timed out
        let timeout = WarmupResult::timed_out(1, Duration::from_secs(60));
        assert_eq!(timeout.status, WarmupStatus::TimedOut);
        assert!(timeout.error.expect("err").contains("timed out"));

        // Empty latencies
        let empty = WarmupResult::success(0, Duration::ZERO, &[]);
        assert_eq!(empty.first_latency, Duration::ZERO);
        assert!((empty.speedup_factor - 1.0).abs() < f64::EPSILON);

        // Zero last latency edge case
        let zero = WarmupResult::success(
            2,
            Duration::from_millis(100),
            &[Duration::from_millis(100), Duration::ZERO],
        );
        assert!((zero.speedup_factor - 1.0).abs() < f64::EPSILON);

        // Average latency calculation
        let avg_test = WarmupResult::success(
            3,
            Duration::from_millis(600),
            &[
                Duration::from_millis(100),
                Duration::from_millis(200),
                Duration::from_millis(300),
            ],
        );
        assert_eq!(avg_test.avg_latency, Duration::from_millis(200));
    }

    // WARM-004: Health Tests
    #[test]
    fn test_model_health_basic() {
        let health = ModelHealth::new();
        assert!(!health.is_ready());
        assert_eq!(health.status(), WarmupStatus::NotStarted);
        assert_eq!(health.total_requests(), 0);

        health.set_ready(true);
        assert!(health.is_ready());
        health.set_ready(false);
        assert!(!health.is_ready());

        health.set_status(WarmupStatus::Ready);
        assert!(health.is_ready());
        assert_eq!(health.status(), WarmupStatus::Ready);

        health.record_success();
        health.record_success();
        health.record_failure();
        assert_eq!(health.total_requests(), 2);
        assert_eq!(health.failed_requests(), 1);
        assert!((health.error_rate() - 0.5).abs() < f64::EPSILON);

        // Default trait
        let def = ModelHealth::default();
        assert!(!def.is_ready());
    }

    #[test]
    fn test_model_health_timing_and_report() {
        let health = ModelHealth::new();
        std::thread::sleep(Duration::from_millis(5));
        assert!(health.uptime() >= Duration::from_millis(5));

        let before = health.time_since_last_check();
        health.touch();
        assert!(health.time_since_last_check() < before);

        health.set_status(WarmupStatus::Ready);
        health.record_success();
        let report = health.report();
        assert!(report.ready);
        assert_eq!(report.status, WarmupStatus::Ready);
        assert_eq!(report.total_requests, 1);
    }

    #[test]
    fn test_model_health_clone_shares_state() {
        let health = ModelHealth::new();
        health.set_status(WarmupStatus::Ready);
        health.record_success();
        let cloned = health.clone();
        assert!(cloned.is_ready());
        health.record_success();
        assert_eq!(cloned.total_requests(), 2); // Shared Arc state
    }

    // WARM-005: Executor Tests
    #[test]
    fn test_warmup_executor() {
        let executor = WarmupExecutor::new(WarmupConfig::new().with_warmup_iterations(3));
        assert_eq!(executor.config().warmup_iterations, 3);

        let result = executor.simulate_warmup();
        assert_eq!(result.status, WarmupStatus::Ready);
        assert!(result.first_latency > result.last_latency);

        // Default trait
        assert_eq!(WarmupExecutor::default().config().warmup_iterations, 3);

        // Timeout check
        let timeout_exec =
            WarmupExecutor::new(WarmupConfig::new().with_timeout(Duration::from_millis(1)));
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        assert!(timeout_exec.check_timeout(start, 0).is_some());

        // No timeout
        let long_exec =
            WarmupExecutor::new(WarmupConfig::new().with_timeout(Duration::from_secs(60)));
        assert!(long_exec.check_timeout(Instant::now(), 5).is_none());

        // Many iterations (jitter clamping)
        let many = WarmupExecutor::new(WarmupConfig::new().with_warmup_iterations(10));
        let res = many.simulate_warmup();
        assert_eq!(res.iterations_completed, 10);
        assert!(res.avg_latency > Duration::ZERO);
    }

    // WARM-006: Preload Config Tests
    #[test]
    fn test_preload_config() {
        let def = PreloadConfig::default();
        assert!(def.models.is_empty());
        assert!(def.parallel_loading);
        assert_eq!(def.max_concurrent, 4);

        let model = PreloadModelConfig::new("llama", "pacha://llama:7b");
        let config = PreloadConfig::new()
            .with_model(model)
            .with_parallel_loading(false)
            .with_max_concurrent(2)
            .with_fail_fast(true);
        assert_eq!(config.models.len(), 1);
        assert!(!config.parallel_loading);
        assert!(config.fail_fast);

        // Min concurrent
        assert_eq!(
            PreloadConfig::new().with_max_concurrent(0).max_concurrent,
            1
        );

        // Multiple models
        let multi = PreloadConfig::new()
            .with_model(PreloadModelConfig::new("m1", "f://1").with_priority(10))
            .with_model(PreloadModelConfig::new("m2", "f://2").with_priority(5));
        assert_eq!(multi.models.len(), 2);
    }

    #[test]
    fn test_preload_model_config() {
        let basic = PreloadModelConfig::new("gpt2", "hf://gpt2");
        assert_eq!(basic.model_id, "gpt2");
        assert_eq!(basic.priority, 100);
        assert!(basic.warmup);
        assert!(basic.warmup_config.is_none());

        let built = PreloadModelConfig::new("llama", "file://model.gguf")
            .with_priority(10)
            .with_warmup(false)
            .with_warmup_config(WarmupConfig::new().with_warmup_iterations(5));
        assert_eq!(built.priority, 10);
        assert!(built.warmup); // with_warmup_config enables warmup
        assert_eq!(built.warmup_config.expect("cfg").warmup_iterations, 5);
    }

    // Serialization Tests
    #[test]
    fn test_serialization() {
        // WarmupConfig roundtrip
        let config = WarmupConfig::new()
            .with_warmup_iterations(8)
            .with_sample_prompt("Test");
        let json = serde_json::to_string(&config).expect("ser");
        let deser: WarmupConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(deser.warmup_iterations, 8);

        // All WarmupStatus variants
        for status in [
            WarmupStatus::NotStarted,
            WarmupStatus::InProgress,
            WarmupStatus::Ready,
            WarmupStatus::Failed,
            WarmupStatus::TimedOut,
        ] {
            let j = serde_json::to_string(&status).expect("ser");
            let d: WarmupStatus = serde_json::from_str(&j).expect("de");
            assert_eq!(d, status);
        }

        // WarmupResult
        let result =
            WarmupResult::success(3, Duration::from_millis(100), &[Duration::from_millis(50)]);
        assert!(serde_json::to_string(&result)
            .expect("ser")
            .contains("Ready"));

        // PreloadConfig roundtrip
        let model = PreloadModelConfig::new("test", "file://test.gguf")
            .with_warmup_config(WarmupConfig::new().with_warmup_iterations(7));
        let pc = PreloadConfig::new().with_model(model).with_fail_fast(true);
        let pc_json = serde_json::to_string(&pc).expect("ser");
        let pc_de: PreloadConfig = serde_json::from_str(&pc_json).expect("de");
        assert_eq!(
            pc_de.models[0]
                .warmup_config
                .as_ref()
                .expect("cfg")
                .warmup_iterations,
            7
        );

        // HealthReport
        let report = HealthReport {
            ready: true,
            status: WarmupStatus::Ready,
            uptime_secs: 100.0,
            total_requests: 1000,
            failed_requests: 5,
            error_rate: 0.005,
            time_since_last_check_secs: 1.5,
        };
        assert!(serde_json::to_string(&report)
            .expect("ser")
            .contains("1000"));

        let json2 = r#"{"ready":false,"status":"InProgress","uptime_secs":42.5,"total_requests":500,"failed_requests":10,"error_rate":0.02,"time_since_last_check_secs":0.5}"#;
        let r2: HealthReport = serde_json::from_str(json2).expect("de");
        assert_eq!(r2.total_requests, 500);
    }

    // Debug trait coverage
    #[test]
    fn test_debug_traits() {
        assert!(format!(
            "{:?}",
            WarmupResult::failed("err", 1, Duration::from_secs(1))
        )
        .contains("Failed"));
        assert!(format!("{:?}", WarmupConfig::new()).contains("warmup_iterations"));
        assert!(format!("{:?}", WarmupExecutor::default()).contains("WarmupExecutor"));
    }
}
