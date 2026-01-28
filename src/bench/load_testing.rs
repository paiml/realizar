//! Load testing framework for stress testing inference servers
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - LoadTestConfig for wrk2-style load testing configuration
//! - LoadTestResult for test outcomes
//! - LoadTestRunner for executing load tests

#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

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
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // LoadTestConfig tests
    // =========================================================================

    #[test]
    fn test_load_test_config_default() {
        let config = LoadTestConfig::default();
        assert_eq!(config.concurrency, 10);
        assert_eq!(config.duration_secs, 60);
        assert_eq!(config.target_rps, 0.0);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.warmup_secs, 5);
        assert!((config.latency_threshold_ms - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_config_for_stress_test() {
        let config = LoadTestConfig::for_stress_test();
        assert_eq!(config.concurrency, 100);
        assert_eq!(config.duration_secs, 300);
        assert_eq!(config.timeout_ms, 10_000);
        assert_eq!(config.warmup_secs, 10);
        assert!((config.latency_threshold_ms - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_config_for_latency_test() {
        let config = LoadTestConfig::for_latency_test();
        assert_eq!(config.concurrency, 1);
        assert_eq!(config.duration_secs, 60);
        assert!((config.target_rps - 10.0).abs() < f64::EPSILON);
        assert_eq!(config.timeout_ms, 2000);
        assert!((config.latency_threshold_ms - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_config_is_valid() {
        let config = LoadTestConfig::default();
        assert!(config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_zero_concurrency() {
        let config = LoadTestConfig {
            concurrency: 0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_zero_duration() {
        let config = LoadTestConfig {
            duration_secs: 0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_zero_timeout() {
        let config = LoadTestConfig {
            timeout_ms: 0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_zero_threshold() {
        let config = LoadTestConfig {
            latency_threshold_ms: 0.0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_is_valid_negative_threshold() {
        let config = LoadTestConfig {
            latency_threshold_ms: -1.0,
            ..Default::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_load_test_config_clone() {
        let config = LoadTestConfig::for_stress_test();
        let cloned = config.clone();
        assert_eq!(config.concurrency, cloned.concurrency);
        assert_eq!(config.duration_secs, cloned.duration_secs);
    }

    #[test]
    fn test_load_test_config_debug() {
        let config = LoadTestConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("LoadTestConfig"));
        assert!(debug.contains("concurrency"));
    }

    #[test]
    fn test_load_test_config_serialization() {
        let config = LoadTestConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        assert!(json.contains("concurrency"));
        assert!(json.contains("60")); // duration_secs
    }

    #[test]
    fn test_load_test_config_deserialization() {
        let json = r#"{"concurrency": 50, "duration_secs": 120, "target_rps": 100.0, "timeout_ms": 3000, "warmup_secs": 10, "latency_threshold_ms": 300.0}"#;
        let config: LoadTestConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(config.concurrency, 50);
        assert_eq!(config.duration_secs, 120);
        assert!((config.target_rps - 100.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // LoadTestResult tests
    // =========================================================================

    #[test]
    fn test_load_test_result_is_passing_true() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 995,
            failed_requests: 5,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.005, // 0.5% < 1%
            passed_latency_threshold: true,
        };
        assert!(result.is_passing());
    }

    #[test]
    fn test_load_test_result_is_passing_false_latency() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 995,
            failed_requests: 5,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.005,
            passed_latency_threshold: false,
        };
        assert!(!result.is_passing());
    }

    #[test]
    fn test_load_test_result_is_passing_false_error_rate() {
        let result = LoadTestResult {
            total_requests: 100,
            successful_requests: 80,
            failed_requests: 20,
            rps_achieved: 10.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 100_000,
            duration_secs: 10.0,
            error_rate: 0.20, // 20% > 1%
            passed_latency_threshold: true,
        };
        assert!(!result.is_passing());
    }

    #[test]
    fn test_load_test_result_throughput_mbps() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 10_000_000, // 10 MB
            duration_secs: 10.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        // 10 MB / 10 sec = 1 MB/s
        assert!((result.throughput_mbps() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_result_throughput_mbps_zero_duration() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            rps_achieved: 0.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 10_000_000,
            duration_secs: 0.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        assert!((result.throughput_mbps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_result_clone() {
        let result = LoadTestResult {
            total_requests: 500,
            successful_requests: 490,
            failed_requests: 10,
            rps_achieved: 50.0,
            latency_p50_ms: 15.0,
            latency_p95_ms: 25.0,
            latency_p99_ms: 35.0,
            latency_max_ms: 55.0,
            data_transferred_bytes: 500_000,
            duration_secs: 10.0,
            error_rate: 0.02,
            passed_latency_threshold: true,
        };
        let cloned = result.clone();
        assert_eq!(result.total_requests, cloned.total_requests);
        assert!((result.latency_p50_ms - cloned.latency_p50_ms).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_test_result_debug() {
        let result = LoadTestResult {
            total_requests: 100,
            successful_requests: 100,
            failed_requests: 0,
            rps_achieved: 10.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 40.0,
            data_transferred_bytes: 100_000,
            duration_secs: 10.0,
            error_rate: 0.0,
            passed_latency_threshold: true,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("LoadTestResult"));
        assert!(debug.contains("total_requests"));
    }

    #[test]
    fn test_load_test_result_serialization() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 990,
            failed_requests: 10,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 50.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.01,
            passed_latency_threshold: true,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("total_requests"));
        assert!(json.contains("1000"));
    }

    // =========================================================================
    // LoadTestRunner tests
    // =========================================================================

    #[test]
    fn test_load_test_runner_new() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config.clone());
        assert_eq!(runner.config().concurrency, config.concurrency);
    }

    #[test]
    fn test_load_test_runner_config() {
        let config = LoadTestConfig::for_stress_test();
        let runner = LoadTestRunner::new(config);
        assert_eq!(runner.config().concurrency, 100);
    }

    #[test]
    fn test_load_test_runner_debug() {
        let runner = LoadTestRunner::new(LoadTestConfig::default());
        let debug = format!("{:?}", runner);
        assert!(debug.contains("LoadTestRunner"));
    }

    #[test]
    fn test_load_test_runner_simulate_run_basic() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        assert!(result.total_requests > 0);
        assert!(result.successful_requests <= result.total_requests);
        assert_eq!(
            result.total_requests,
            result.successful_requests + result.failed_requests
        );
        assert!(result.rps_achieved > 0.0);
    }

    #[test]
    fn test_load_test_runner_simulate_run_stress() {
        let config = LoadTestConfig::for_stress_test();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Stress test should have more requests
        assert!(result.total_requests > 10_000);
        assert!(result.latency_p50_ms > 0.0);
        assert!(result.latency_p99_ms >= result.latency_p50_ms);
    }

    #[test]
    fn test_load_test_runner_simulate_run_latency() {
        let config = LoadTestConfig::for_latency_test();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Latency test has concurrency=1, so fewer requests
        assert!(result.total_requests > 0);
        // Lower concurrency should mean lower latencies
        assert!(result.latency_p50_ms < 50.0);
    }

    #[test]
    fn test_load_test_runner_simulate_run_error_rate() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Simulated error rate is ~1%
        assert!(result.error_rate >= 0.0);
        assert!(result.error_rate <= 0.02);
    }

    #[test]
    fn test_load_test_runner_simulate_run_latency_percentiles_order() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Percentiles should be in ascending order
        assert!(result.latency_p50_ms <= result.latency_p95_ms);
        assert!(result.latency_p95_ms <= result.latency_p99_ms);
        assert!(result.latency_p99_ms <= result.latency_max_ms);
    }

    #[test]
    fn test_load_test_runner_simulate_run_data_transferred() {
        let config = LoadTestConfig::default();
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        // Data should be approximately total_requests * 1KB
        let expected_data = (result.total_requests * 1024) as u64;
        assert_eq!(result.data_transferred_bytes, expected_data);
    }

    #[test]
    fn test_load_test_runner_simulate_run_duration() {
        let config = LoadTestConfig {
            duration_secs: 30,
            ..Default::default()
        };
        let runner = LoadTestRunner::new(config);
        let result = runner.simulate_run();

        assert!((result.duration_secs - 30.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_load_test_config_extreme_values() {
        let config = LoadTestConfig {
            concurrency: 10000,
            duration_secs: 3600,
            target_rps: 1_000_000.0,
            timeout_ms: 60_000,
            warmup_secs: 60,
            latency_threshold_ms: 10.0,
        };
        assert!(config.is_valid());
    }

    #[test]
    fn test_load_test_result_boundary_error_rate() {
        // Exactly 1% error rate - should still fail (< not <=)
        let result = LoadTestResult {
            total_requests: 100,
            successful_requests: 99,
            failed_requests: 1,
            rps_achieved: 10.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 40.0,
            data_transferred_bytes: 100_000,
            duration_secs: 10.0,
            error_rate: 0.01, // Exactly 1%
            passed_latency_threshold: true,
        };
        // 0.01 is NOT < 0.01, so should NOT pass
        assert!(!result.is_passing());
    }

    #[test]
    fn test_load_test_result_just_under_error_threshold() {
        let result = LoadTestResult {
            total_requests: 1000,
            successful_requests: 991,
            failed_requests: 9,
            rps_achieved: 100.0,
            latency_p50_ms: 10.0,
            latency_p95_ms: 20.0,
            latency_p99_ms: 30.0,
            latency_max_ms: 40.0,
            data_transferred_bytes: 1_000_000,
            duration_secs: 10.0,
            error_rate: 0.009, // Just under 1%
            passed_latency_threshold: true,
        };
        assert!(result.is_passing());
    }
}
