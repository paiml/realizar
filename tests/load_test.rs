//! Load testing for Realizar HTTP API
//!
//! Tests the system under various load patterns:
//! - Concurrent requests
//! - Sustained load
//! - Spike traffic
//! - Batch operations
//!
//! Metrics tracked:
//! - Latency percentiles (p50, p95, p99)
//! - Throughput (requests/sec)
//! - Error rates
//! - Resource usage

use realizar::api::{BatchGenerateRequest, BatchTokenizeRequest, GenerateRequest, TokenizeRequest};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

/// Type alias for test results
type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Load test configuration
#[derive(Debug, Clone)]
struct LoadTestConfig {
    /// Number of concurrent clients
    concurrency: usize,
    /// Total number of requests to send
    total_requests: usize,
    /// Request timeout
    timeout: Duration,
    /// Base URL for API
    base_url: String,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrency: 10,
            total_requests: 100,
            timeout: Duration::from_secs(30),
            base_url: "http://127.0.0.1:8080".to_string(),
        }
    }
}

/// Metrics collected during load test
#[derive(Debug, Default)]
struct LoadTestMetrics {
    /// Request latencies in microseconds
    latencies: Vec<u64>,
    /// Number of successful requests
    successes: Arc<AtomicU64>,
    /// Number of failed requests
    failures: Arc<AtomicU64>,
    /// Total duration
    duration: Duration,
}

impl LoadTestMetrics {
    fn new() -> Self {
        Self {
            latencies: Vec::new(),
            successes: Arc::new(AtomicU64::new(0)),
            failures: Arc::new(AtomicU64::new(0)),
            duration: Duration::default(),
        }
    }

    /// Calculate percentile from sorted latencies
    fn percentile(&self, p: f64) -> u64 {
        if self.latencies.is_empty() {
            return 0;
        }
        let idx = ((p / 100.0) * self.latencies.len() as f64).ceil() as usize - 1;
        self.latencies[idx.min(self.latencies.len() - 1)]
    }

    /// Calculate requests per second
    fn throughput(&self) -> f64 {
        let total = self.successes.load(Ordering::Relaxed) + self.failures.load(Ordering::Relaxed);
        total as f64 / self.duration.as_secs_f64()
    }

    /// Calculate error rate percentage
    fn error_rate(&self) -> f64 {
        let total = self.successes.load(Ordering::Relaxed) + self.failures.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        (self.failures.load(Ordering::Relaxed) as f64 / total as f64) * 100.0
    }

    /// Print summary report
    fn report(&self) {
        println!("\n=== Load Test Results ===");
        println!("Total requests: {}", self.latencies.len());
        println!("Successes: {}", self.successes.load(Ordering::Relaxed));
        println!("Failures: {}", self.failures.load(Ordering::Relaxed));
        println!("Duration: {:.2}s", self.duration.as_secs_f64());
        println!("Throughput: {:.2} req/s", self.throughput());
        println!("Error rate: {:.2}%", self.error_rate());
        println!("\nLatency Percentiles:");
        println!("  p50: {:.2}ms", self.percentile(50.0) as f64 / 1000.0);
        println!("  p95: {:.2}ms", self.percentile(95.0) as f64 / 1000.0);
        println!("  p99: {:.2}ms", self.percentile(99.0) as f64 / 1000.0);
        println!(
            "  max: {:.2}ms",
            self.latencies.iter().max().unwrap_or(&0) / 1000
        );
    }
}

/// Run health check load test
async fn load_test_health(
    config: &LoadTestConfig,
    client: &reqwest::Client,
) -> Result<LoadTestMetrics, Box<dyn std::error::Error>> {
    let mut metrics = LoadTestMetrics::new();
    let start_time = Instant::now();

    // Calculate requests per worker
    let requests_per_worker = config.total_requests / config.concurrency;
    let remainder = config.total_requests % config.concurrency;

    let mut handles: Vec<JoinHandle<Vec<u64>>> = Vec::new();

    for worker_id in 0..config.concurrency {
        let worker_requests = if worker_id < remainder {
            requests_per_worker + 1
        } else {
            requests_per_worker
        };

        let client = client.clone();
        let url = format!("{}/health", config.base_url);
        let successes = Arc::clone(&metrics.successes);
        let failures = Arc::clone(&metrics.failures);
        let timeout = config.timeout;

        let handle = tokio::spawn(async move {
            let mut latencies = Vec::new();

            for _ in 0..worker_requests {
                let start = Instant::now();
                match tokio::time::timeout(timeout, client.get(&url).send()).await {
                    Ok(Ok(resp)) if resp.status().is_success() => {
                        successes.fetch_add(1, Ordering::Relaxed);
                        latencies.push(start.elapsed().as_micros() as u64);
                    },
                    _ => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    },
                }
            }

            latencies
        });

        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        if let Ok(worker_latencies) = handle.await {
            metrics.latencies.extend(worker_latencies);
        }
    }

    metrics.duration = start_time.elapsed();
    metrics.latencies.sort_unstable();

    Ok(metrics)
}

/// Run tokenize endpoint load test
async fn load_test_tokenize(
    config: &LoadTestConfig,
    client: &reqwest::Client,
) -> Result<LoadTestMetrics, Box<dyn std::error::Error>> {
    let mut metrics = LoadTestMetrics::new();
    let start_time = Instant::now();

    let requests_per_worker = config.total_requests / config.concurrency;
    let remainder = config.total_requests % config.concurrency;

    let mut handles: Vec<JoinHandle<Vec<u64>>> = Vec::new();

    for worker_id in 0..config.concurrency {
        let worker_requests = if worker_id < remainder {
            requests_per_worker + 1
        } else {
            requests_per_worker
        };

        let client = client.clone();
        let url = format!("{}/tokenize", config.base_url);
        let successes = Arc::clone(&metrics.successes);
        let failures = Arc::clone(&metrics.failures);
        let timeout = config.timeout;

        let handle = tokio::spawn(async move {
            let mut latencies = Vec::new();

            for i in 0..worker_requests {
                let request = TokenizeRequest {
                    text: format!("Hello world test message {}", i),
                    model_id: None,
                };

                let start = Instant::now();
                match tokio::time::timeout(timeout, client.post(&url).json(&request).send()).await {
                    Ok(Ok(resp)) if resp.status().is_success() => {
                        successes.fetch_add(1, Ordering::Relaxed);
                        latencies.push(start.elapsed().as_micros() as u64);
                    },
                    _ => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    },
                }
            }

            latencies
        });

        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        if let Ok(worker_latencies) = handle.await {
            metrics.latencies.extend(worker_latencies);
        }
    }

    metrics.duration = start_time.elapsed();
    metrics.latencies.sort_unstable();

    Ok(metrics)
}

/// Run generate endpoint load test
async fn load_test_generate(
    config: &LoadTestConfig,
    client: &reqwest::Client,
) -> Result<LoadTestMetrics, Box<dyn std::error::Error>> {
    let mut metrics = LoadTestMetrics::new();
    let start_time = Instant::now();

    let requests_per_worker = config.total_requests / config.concurrency;
    let remainder = config.total_requests % config.concurrency;

    let mut handles: Vec<JoinHandle<Vec<u64>>> = Vec::new();

    for worker_id in 0..config.concurrency {
        let worker_requests = if worker_id < remainder {
            requests_per_worker + 1
        } else {
            requests_per_worker
        };

        let client = client.clone();
        let url = format!("{}/generate", config.base_url);
        let successes = Arc::clone(&metrics.successes);
        let failures = Arc::clone(&metrics.failures);
        let timeout = config.timeout;

        let handle = tokio::spawn(async move {
            let mut latencies = Vec::new();

            for i in 0..worker_requests {
                let request = GenerateRequest {
                    prompt: format!("Hello {}", i),
                    max_tokens: 5,
                    temperature: 1.0,
                    strategy: "greedy".to_string(),
                    top_k: 50,
                    top_p: 0.9,
                    seed: None,
                    model_id: None,
                };

                let start = Instant::now();
                match tokio::time::timeout(timeout, client.post(&url).json(&request).send()).await {
                    Ok(Ok(resp)) if resp.status().is_success() => {
                        successes.fetch_add(1, Ordering::Relaxed);
                        latencies.push(start.elapsed().as_micros() as u64);
                    },
                    _ => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    },
                }
            }

            latencies
        });

        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        if let Ok(worker_latencies) = handle.await {
            metrics.latencies.extend(worker_latencies);
        }
    }

    metrics.duration = start_time.elapsed();
    metrics.latencies.sort_unstable();

    Ok(metrics)
}

/// Run batch tokenize load test
async fn load_test_batch_tokenize(
    config: &LoadTestConfig,
    client: &reqwest::Client,
) -> Result<LoadTestMetrics, Box<dyn std::error::Error>> {
    let mut metrics = LoadTestMetrics::new();
    let start_time = Instant::now();

    let requests_per_worker = config.total_requests / config.concurrency;
    let remainder = config.total_requests % config.concurrency;

    let mut handles: Vec<JoinHandle<Vec<u64>>> = Vec::new();

    for worker_id in 0..config.concurrency {
        let worker_requests = if worker_id < remainder {
            requests_per_worker + 1
        } else {
            requests_per_worker
        };

        let client = client.clone();
        let url = format!("{}/batch/tokenize", config.base_url);
        let successes = Arc::clone(&metrics.successes);
        let failures = Arc::clone(&metrics.failures);
        let timeout = config.timeout;

        let handle = tokio::spawn(async move {
            let mut latencies = Vec::new();

            for i in 0..worker_requests {
                let request = BatchTokenizeRequest {
                    texts: vec![
                        format!("Text 1 batch {}", i),
                        format!("Text 2 batch {}", i),
                        format!("Text 3 batch {}", i),
                    ],
                };

                let start = Instant::now();
                match tokio::time::timeout(timeout, client.post(&url).json(&request).send()).await {
                    Ok(Ok(resp)) if resp.status().is_success() => {
                        successes.fetch_add(1, Ordering::Relaxed);
                        latencies.push(start.elapsed().as_micros() as u64);
                    },
                    _ => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    },
                }
            }

            latencies
        });

        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        if let Ok(worker_latencies) = handle.await {
            metrics.latencies.extend(worker_latencies);
        }
    }

    metrics.duration = start_time.elapsed();
    metrics.latencies.sort_unstable();

    Ok(metrics)
}

/// Run batch generate load test
async fn load_test_batch_generate(
    config: &LoadTestConfig,
    client: &reqwest::Client,
) -> Result<LoadTestMetrics, Box<dyn std::error::Error>> {
    let mut metrics = LoadTestMetrics::new();
    let start_time = Instant::now();

    let requests_per_worker = config.total_requests / config.concurrency;
    let remainder = config.total_requests % config.concurrency;

    let mut handles: Vec<JoinHandle<Vec<u64>>> = Vec::new();

    for worker_id in 0..config.concurrency {
        let worker_requests = if worker_id < remainder {
            requests_per_worker + 1
        } else {
            requests_per_worker
        };

        let client = client.clone();
        let url = format!("{}/batch/generate", config.base_url);
        let successes = Arc::clone(&metrics.successes);
        let failures = Arc::clone(&metrics.failures);
        let timeout = config.timeout;

        let handle = tokio::spawn(async move {
            let mut latencies = Vec::new();

            for i in 0..worker_requests {
                let request = BatchGenerateRequest {
                    prompts: vec![
                        format!("Prompt 1 batch {}", i),
                        format!("Prompt 2 batch {}", i),
                    ],
                    max_tokens: 5,
                    temperature: 1.0,
                    strategy: "greedy".to_string(),
                    top_k: 50,
                    top_p: 0.9,
                    seed: None,
                };

                let start = Instant::now();
                match tokio::time::timeout(timeout, client.post(&url).json(&request).send()).await {
                    Ok(Ok(resp)) if resp.status().is_success() => {
                        successes.fetch_add(1, Ordering::Relaxed);
                        latencies.push(start.elapsed().as_micros() as u64);
                    },
                    _ => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    },
                }
            }

            latencies
        });

        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        if let Ok(worker_latencies) = handle.await {
            metrics.latencies.extend(worker_latencies);
        }
    }

    metrics.duration = start_time.elapsed();
    metrics.latencies.sort_unstable();

    Ok(metrics)
}

// Tests are only enabled with --features load-test-enabled
#[cfg(feature = "load-test-enabled")]
mod enabled_tests {
    use super::*;

    /// Helper to start test server
    async fn start_test_server() -> TestResult {
        // Give server time to start if not already running
        tokio::time::sleep(Duration::from_secs(1)).await;
        Ok(())
    }

    #[tokio::test]
    async fn test_health_endpoint_load() -> TestResult {
        start_test_server().await?;

        let config = LoadTestConfig {
            concurrency: 10,
            total_requests: 100,
            ..Default::default()
        };

        let client = reqwest::Client::new();
        let metrics = load_test_health(&config, &client).await?;

        metrics.report();

        // Assertions
        assert!(metrics.error_rate() < 5.0, "Error rate too high");
        assert!(metrics.percentile(95.0) < 1_000_000, "p95 latency > 1s");

        Ok(())
    }

    #[tokio::test]
    async fn test_tokenize_endpoint_load() -> TestResult {
        start_test_server().await?;

        let config = LoadTestConfig {
            concurrency: 5,
            total_requests: 50,
            ..Default::default()
        };

        let client = reqwest::Client::new();
        let metrics = load_test_tokenize(&config, &client).await?;

        metrics.report();

        assert!(metrics.error_rate() < 10.0, "Error rate too high");

        Ok(())
    }

    #[tokio::test]
    async fn test_generate_endpoint_load() -> TestResult {
        start_test_server().await?;

        let config = LoadTestConfig {
            concurrency: 5,
            total_requests: 25,
            timeout: Duration::from_secs(60),
            ..Default::default()
        };

        let client = reqwest::Client::new();
        let metrics = load_test_generate(&config, &client).await?;

        metrics.report();

        assert!(metrics.error_rate() < 10.0, "Error rate too high");

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_tokenize_load() -> TestResult {
        start_test_server().await?;

        let config = LoadTestConfig {
            concurrency: 5,
            total_requests: 25,
            ..Default::default()
        };

        let client = reqwest::Client::new();
        let metrics = load_test_batch_tokenize(&config, &client).await?;

        metrics.report();

        assert!(metrics.error_rate() < 10.0, "Error rate too high");

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_generate_load() -> TestResult {
        start_test_server().await?;

        let config = LoadTestConfig {
            concurrency: 3,
            total_requests: 15,
            timeout: Duration::from_secs(60),
            ..Default::default()
        };

        let client = reqwest::Client::new();
        let metrics = load_test_batch_generate(&config, &client).await?;

        metrics.report();

        assert!(metrics.error_rate() < 10.0, "Error rate too high");

        Ok(())
    }

    #[tokio::test]
    async fn test_sustained_load() -> TestResult {
        start_test_server().await?;

        // Run sustained load for 30 seconds
        let config = LoadTestConfig {
            concurrency: 10,
            total_requests: 200,
            timeout: Duration::from_secs(30),
            ..Default::default()
        };

        let client = reqwest::Client::new();
        let metrics = load_test_health(&config, &client).await?;

        println!("\n=== Sustained Load Test (30s) ===");
        metrics.report();

        assert!(metrics.error_rate() < 5.0, "Error rate too high");
        assert!(
            metrics.throughput() > 5.0,
            "Throughput too low: {} req/s",
            metrics.throughput()
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_spike_traffic() -> TestResult {
        start_test_server().await?;

        // Baseline load
        let baseline_config = LoadTestConfig {
            concurrency: 2,
            total_requests: 10,
            ..Default::default()
        };

        let client = reqwest::Client::new();
        let baseline_metrics = load_test_health(&baseline_config, &client).await?;

        println!("\n=== Baseline Load ===");
        baseline_metrics.report();

        // Spike to 10x load
        let spike_config = LoadTestConfig {
            concurrency: 20,
            total_requests: 100,
            ..Default::default()
        };

        let spike_metrics = load_test_health(&spike_config, &client).await?;

        println!("\n=== Spike Load (10x) ===");
        spike_metrics.report();

        // System should handle spike gracefully
        assert!(
            spike_metrics.error_rate() < 15.0,
            "Error rate too high under spike"
        );

        Ok(())
    }
}

// Placeholder test when load testing is disabled
#[cfg(not(feature = "load-test-enabled"))]
#[test]
fn load_tests_disabled() {
    println!("Load tests are disabled. Enable with --features load-test-enabled");
    println!("Note: Requires a running server at http://127.0.0.1:8080");
}
