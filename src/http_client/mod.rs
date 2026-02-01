//! HTTP Client for Real Model Server Benchmarking
//!
//! This module implements REAL HTTP calls to external model servers.
//! **NO MOCK DATA** - actual network requests with timing measurements.
//!
//! ## Supported Backends
//! - vLLM: OpenAI-compatible `/v1/completions` endpoint
//! - Ollama: `/api/generate` endpoint
//! - llama.cpp: OpenAI-compatible `/v1/completions` endpoint
//!
//! ## Quality Features
//! - Preflight validation per Toyota Way principles (Jidoka, Poka-yoke)
//! - CV-based stopping criterion per Hoefler & Belli SC'15
//! - MAD-based outlier detection per Chen et al.
//!
//! ## References
//! - [1] OpenAI API Spec: https://platform.openai.com/docs/api-reference
//! - [2] Ollama API Spec: https://github.com/ollama/ollama/blob/main/docs/api.md
//! - [3] Hoefler & Belli SC'15: CV-based stopping for reproducible benchmarks
//! - [4] Chen et al.: Robust outlier detection using MAD

use std::time::Instant;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::bench_preflight::{
    canonical_inputs, CvStoppingCriterion, OutlierDetector, PreflightRunner, QualityMetrics,
    ServerAvailabilityCheck, StopDecision,
};
use crate::error::{RealizarError, Result};

/// OpenAI-compatible completion request (vLLM, llama.cpp)
#[derive(Debug, Clone, Serialize)]
pub struct CompletionRequest {
    /// Model identifier
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
}

/// OpenAI-compatible completion response
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionResponse {
    /// Response ID
    pub id: String,
    /// Completion choices
    pub choices: Vec<CompletionChoice>,
    /// Usage statistics
    pub usage: Option<UsageStats>,
}

/// A single completion choice
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionChoice {
    /// Generated text
    pub text: String,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// Token usage statistics
#[derive(Debug, Clone, Deserialize)]
pub struct UsageStats {
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Completion tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// Ollama generate request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaRequest {
    /// Model name
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Whether to stream
    #[serde(default)]
    pub stream: bool,
    /// Generation options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
}

/// Ollama generation options
#[derive(Debug, Clone, Serialize)]
pub struct OllamaOptions {
    /// Maximum tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<usize>,
    /// Temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// Ollama generate response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaResponse {
    /// Model used
    pub model: String,
    /// Generated response
    pub response: String,
    /// Whether generation is done
    pub done: bool,
    /// Total duration in nanoseconds
    #[serde(default)]
    pub total_duration: u64,
    /// Load duration in nanoseconds
    #[serde(default)]
    pub load_duration: u64,
    /// Prompt evaluation count
    #[serde(default)]
    pub prompt_eval_count: usize,
    /// Prompt evaluation duration in nanoseconds
    #[serde(default)]
    pub prompt_eval_duration: u64,
    /// Evaluation count (tokens generated)
    #[serde(default)]
    pub eval_count: usize,
    /// Evaluation duration in nanoseconds
    #[serde(default)]
    pub eval_duration: u64,
}

/// llama.cpp native completion response (different from OpenAI format)
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaCppResponse {
    /// Generated content
    pub content: String,
    /// Model path
    #[serde(default)]
    pub model: String,
    /// Number of tokens predicted
    #[serde(default)]
    pub tokens_predicted: usize,
    /// Number of tokens evaluated (prompt)
    #[serde(default)]
    pub tokens_evaluated: usize,
    /// Whether generation stopped
    #[serde(default)]
    pub stop: bool,
    /// Timing information
    #[serde(default)]
    pub timings: Option<LlamaCppTimings>,
}

/// llama.cpp timing information
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaCppTimings {
    /// Prompt tokens
    #[serde(default)]
    pub prompt_n: usize,
    /// Prompt processing time in ms
    #[serde(default)]
    pub prompt_ms: f64,
    /// Predicted tokens
    #[serde(default)]
    pub predicted_n: usize,
    /// Prediction time in ms
    #[serde(default)]
    pub predicted_ms: f64,
    /// Tokens per second for generation
    #[serde(default)]
    pub predicted_per_second: f64,
}

/// Timing measurements from an HTTP inference request
#[derive(Debug, Clone)]
pub struct InferenceTiming {
    /// Time to first byte (TTFT) in milliseconds
    pub ttft_ms: f64,
    /// Total request time in milliseconds
    pub total_time_ms: f64,
    /// Tokens generated
    pub tokens_generated: usize,
    /// Generated text
    pub text: String,
}

/// HTTP client for model server communication
pub struct ModelHttpClient {
    client: Client,
    timeout_secs: u64,
}

impl Default for ModelHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelHttpClient {
    /// Create a new HTTP client with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            timeout_secs: 60,
        }
    }

    /// Create a new HTTP client with custom timeout
    #[must_use]
    pub fn with_timeout(timeout_secs: u64) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .build()
                .expect("Failed to create HTTP client"),
            timeout_secs,
        }
    }

    /// Get the configured timeout
    #[must_use]
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Call OpenAI-compatible `/v1/completions` endpoint (vLLM, llama.cpp)
    ///
    /// # Errors
    /// Returns error if network request fails or response parsing fails
    pub fn openai_completion(
        &self,
        base_url: &str,
        request: &CompletionRequest,
        api_key: Option<&str>,
    ) -> Result<InferenceTiming> {
        let url = format!("{}/v1/completions", base_url.trim_end_matches('/'));
        let start = Instant::now();

        let mut req_builder = self.client.post(&url).json(request);

        if let Some(key) = api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder
            .send()
            .map_err(|e| RealizarError::ConnectionError(format!("HTTP request failed: {}", e)))?;

        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(RealizarError::ConnectionError(format!(
                "HTTP {} from {}: {}",
                status, url, body
            )));
        }

        let completion: CompletionResponse =
            response.json().map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to parse completion response: {}", e),
            })?;

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let text = completion
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        let tokens_generated = completion.usage.map_or(0, |u| u.completion_tokens);

        Ok(InferenceTiming {
            ttft_ms,
            total_time_ms,
            tokens_generated,
            text,
        })
    }

    /// Call Ollama `/api/generate` endpoint
    ///
    /// # Errors
    /// Returns error if network request fails or response parsing fails
    pub fn ollama_generate(
        &self,
        base_url: &str,
        request: &OllamaRequest,
    ) -> Result<InferenceTiming> {
        let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
        let start = Instant::now();

        let response =
            self.client.post(&url).json(request).send().map_err(|e| {
                RealizarError::ConnectionError(format!("HTTP request failed: {}", e))
            })?;

        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(RealizarError::ConnectionError(format!(
                "HTTP {} from {}: {}",
                status, url, body
            )));
        }

        let ollama_resp: OllamaResponse =
            response.json().map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to parse Ollama response: {}", e),
            })?;

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Use eval_count if available, otherwise estimate from response text
        // Ollama doesn't always return eval_count (depends on model/config)
        let tokens_generated = if ollama_resp.eval_count > 0 {
            ollama_resp.eval_count
        } else {
            // Estimate: ~4 chars per token (common for LLMs)
            // This is a reasonable fallback for scientific reproducibility
            (ollama_resp.response.len() / 4).max(1)
        };

        Ok(InferenceTiming {
            ttft_ms,
            total_time_ms,
            tokens_generated,
            text: ollama_resp.response,
        })
    }

    /// Call llama.cpp native `/completion` endpoint
    ///
    /// Note: llama.cpp also has `/v1/completions` but it returns a non-OpenAI format.
    /// This method uses the native endpoint which is more reliable.
    ///
    /// # Errors
    /// Returns error if network request fails or response parsing fails
    pub fn llamacpp_completion(
        &self,
        base_url: &str,
        request: &CompletionRequest,
    ) -> Result<InferenceTiming> {
        // llama.cpp uses /completion for native format, /v1/completions for OpenAI-ish
        let url = format!("{}/completion", base_url.trim_end_matches('/'));
        let start = Instant::now();

        // llama.cpp expects slightly different field names
        let body = serde_json::json!({
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature.unwrap_or(0.8),
            "stream": false
        });

        let response =
            self.client.post(&url).json(&body).send().map_err(|e| {
                RealizarError::ConnectionError(format!("HTTP request failed: {}", e))
            })?;

        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(RealizarError::ConnectionError(format!(
                "HTTP {} from {}: {}",
                status, url, body
            )));
        }

        let llama_resp: LlamaCppResponse =
            response.json().map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to parse llama.cpp response: {}", e),
            })?;

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(InferenceTiming {
            ttft_ms,
            total_time_ms,
            tokens_generated: llama_resp.tokens_predicted,
            text: llama_resp.content,
        })
    }

    /// Health check for OpenAI-compatible server
    ///
    /// # Errors
    /// Returns error if server is not reachable
    pub fn health_check_openai(&self, base_url: &str) -> Result<bool> {
        let url = format!("{}/v1/models", base_url.trim_end_matches('/'));

        let response =
            self.client.get(&url).send().map_err(|e| {
                RealizarError::ConnectionError(format!("Health check failed: {}", e))
            })?;

        Ok(response.status().is_success())
    }

    /// Health check for Ollama server
    ///
    /// # Errors
    /// Returns error if server is not reachable
    pub fn health_check_ollama(&self, base_url: &str) -> Result<bool> {
        let url = format!("{}/api/tags", base_url.trim_end_matches('/'));

        let response =
            self.client.get(&url).send().map_err(|e| {
                RealizarError::ConnectionError(format!("Health check failed: {}", e))
            })?;

        Ok(response.status().is_success())
    }
}

// ============================================================================
// HTTP Benchmark Runner with CV-based Stopping (Hoefler & Belli SC'15)
// ============================================================================

/// Configuration for HTTP benchmark runs
///
/// Per spec v1.0.1, uses canonical inputs and CV-based stopping criterion.
#[derive(Debug, Clone)]
pub struct HttpBenchmarkConfig {
    /// CV-based stopping criterion (per Hoefler & Belli SC'15)
    pub cv_criterion: CvStoppingCriterion,
    /// Warmup iterations (not counted in stats)
    pub warmup_iterations: usize,
    /// Prompt for inference (uses canonical input by default)
    pub prompt: String,
    /// Max tokens to generate (uses canonical input by default)
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// Whether to run preflight validation
    pub run_preflight: bool,
    /// Whether to filter outliers using MAD
    pub filter_outliers: bool,
    /// Outlier k-factor (3.0 = 99.7% for normal distribution)
    pub outlier_k_factor: f64,
}

impl Default for HttpBenchmarkConfig {
    fn default() -> Self {
        Self {
            cv_criterion: CvStoppingCriterion::default(), // 5% CV, 5-30 samples
            warmup_iterations: 2,
            prompt: canonical_inputs::LATENCY_PROMPT.to_string(),
            max_tokens: canonical_inputs::MAX_TOKENS,
            temperature: 0.0, // Deterministic by default
            run_preflight: true,
            filter_outliers: true,
            outlier_k_factor: 3.0,
        }
    }
}

impl HttpBenchmarkConfig {
    /// Create config with relaxed CV threshold (for quicker benchmarks)
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            cv_criterion: CvStoppingCriterion::new(3, 10, 0.20), // 20% CV
            warmup_iterations: 1,
            run_preflight: false,
            filter_outliers: false,
            ..Default::default()
        }
    }

    /// Create config optimized for reproducibility (strict CV)
    #[must_use]
    pub fn reproducible() -> Self {
        Self {
            cv_criterion: CvStoppingCriterion::new(10, 50, 0.03), // 3% CV, more samples
            warmup_iterations: 3,
            run_preflight: true,
            filter_outliers: true,
            outlier_k_factor: 2.5, // Stricter outlier detection
            ..Default::default()
        }
    }

    /// Backward-compatible accessors for min_samples
    #[must_use]
    pub fn min_samples(&self) -> usize {
        self.cv_criterion.min_samples
    }

    /// Backward-compatible accessor for max_samples
    #[must_use]
    pub fn max_samples(&self) -> usize {
        self.cv_criterion.max_samples
    }

    /// Backward-compatible accessor for cv_threshold
    #[must_use]
    pub fn cv_threshold(&self) -> f64 {
        self.cv_criterion.cv_threshold
    }
}

/// Results from a benchmark run
///
/// Per spec v1.0.1, includes quality metrics for reproducibility assessment.
#[derive(Debug, Clone)]
pub struct HttpBenchmarkResult {
    /// Collected latency samples (ms) - raw, before outlier filtering
    pub latency_samples: Vec<f64>,
    /// Filtered latency samples (ms) - after outlier removal
    pub latency_samples_filtered: Vec<f64>,
    /// Mean latency (ms) - computed from filtered samples
    pub mean_latency_ms: f64,
    /// P50 latency (ms)
    pub p50_latency_ms: f64,
    /// P99 latency (ms)
    pub p99_latency_ms: f64,
    /// Standard deviation (ms)
    pub std_dev_ms: f64,
    /// Coefficient of variation at stop
    pub cv_at_stop: f64,
    /// Throughput (tokens/sec)
    pub throughput_tps: f64,
    /// Cold start latency (first iteration after warmup)
    pub cold_start_ms: f64,
    /// Number of samples collected (raw)
    pub sample_count: usize,
    /// Number of samples after outlier filtering
    pub filtered_sample_count: usize,
    /// Whether CV threshold was achieved
    pub cv_converged: bool,
    /// Quality metrics for reproducibility assessment
    pub quality_metrics: QualityMetrics,
}

/// HTTP benchmark runner with CV-based stopping and preflight validation
///
/// Per spec v1.0.1, implements Toyota Way principles:
/// - Jidoka: Fail-fast via preflight validation
/// - Poka-yoke: Type-safe configuration
/// - Genchi Genbutsu: Verify actual server state before benchmark
pub struct HttpBenchmarkRunner {
    client: ModelHttpClient,
    config: HttpBenchmarkConfig,
    preflight_runner: Option<PreflightRunner>,
    outlier_detector: OutlierDetector,
}

impl HttpBenchmarkRunner {
    /// Create a new benchmark runner
    #[must_use]
    pub fn new(config: HttpBenchmarkConfig) -> Self {
        let outlier_detector = OutlierDetector::new(config.outlier_k_factor);
        Self {
            client: ModelHttpClient::with_timeout(120), // 2 minute timeout for benchmarks
            config,
            preflight_runner: None,
            outlier_detector,
        }
    }

    /// Create with default configuration (5% CV, preflight enabled, outlier filtering)
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(HttpBenchmarkConfig::default())
    }

    /// Create with relaxed configuration (20% CV, no preflight, no outlier filtering)
    #[must_use]
    pub fn with_relaxed() -> Self {
        Self::new(HttpBenchmarkConfig::relaxed())
    }

    /// Create with reproducible configuration (3% CV, strict preflight, outlier filtering)
    #[must_use]
    pub fn with_reproducible() -> Self {
        Self::new(HttpBenchmarkConfig::reproducible())
    }

    /// Run preflight validation for llama.cpp server
    ///
    /// # Errors
    /// Returns error if preflight validation fails
    pub fn run_preflight_llamacpp(&mut self, base_url: &str) -> Result<Vec<String>> {
        let mut runner = PreflightRunner::new();

        // Add server availability check
        let url_parts: Vec<&str> = base_url.trim_end_matches('/').split(':').collect();
        let port = url_parts
            .last()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(8080);

        let mut server_check = ServerAvailabilityCheck::llama_cpp(port);

        // Perform actual health check
        match self.client.health_check_openai(base_url) {
            Ok(true) => server_check.set_health_status(200),
            Ok(false) => server_check.set_health_status(500),
            Err(_) => server_check.set_health_status(0),
        }

        runner.add_check(Box::new(server_check));

        let passed = runner
            .run()
            .map_err(|e| RealizarError::ConnectionError(format!("Preflight failed: {}", e)))?;

        self.preflight_runner = Some(runner);
        Ok(passed)
    }

    /// Run preflight validation for Ollama server
    ///
    /// # Errors
    /// Returns error if preflight validation fails
    pub fn run_preflight_ollama(&mut self, base_url: &str) -> Result<Vec<String>> {
        let mut runner = PreflightRunner::new();

        // Add server availability check
        let url_parts: Vec<&str> = base_url.trim_end_matches('/').split(':').collect();
        let port = url_parts
            .last()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(11434);

        let mut server_check = ServerAvailabilityCheck::ollama(port);

        // Perform actual health check
        match self.client.health_check_ollama(base_url) {
            Ok(true) => server_check.set_health_status(200),
            Ok(false) => server_check.set_health_status(500),
            Err(_) => server_check.set_health_status(0),
        }

        runner.add_check(Box::new(server_check));

        let passed = runner
            .run()
            .map_err(|e| RealizarError::ConnectionError(format!("Preflight failed: {}", e)))?;

        self.preflight_runner = Some(runner);
        Ok(passed)
    }

    /// Get list of passed preflight checks
    #[must_use]
    pub fn preflight_checks_passed(&self) -> Vec<String> {
        self.preflight_runner
            .as_ref()
            .map(|r| r.passed_checks().to_vec())
            .unwrap_or_default()
    }

    /// Run benchmark against llama.cpp server
    ///
    /// Per spec v1.0.1:
    /// 1. Runs optional preflight validation (Jidoka)
    /// 2. Uses CV-based stopping criterion (Hoefler & Belli SC'15)
    /// 3. Applies MAD-based outlier detection
    /// 4. Returns quality metrics for reproducibility assessment
    ///
    /// # Errors
    /// Returns error if preflight fails or server is unreachable
    pub fn benchmark_llamacpp(&mut self, base_url: &str) -> Result<HttpBenchmarkResult> {
        // Preflight validation (Jidoka: fail-fast)
        if self.config.run_preflight {
            self.run_preflight_llamacpp(base_url)?;
        }

        let mut latencies = Vec::with_capacity(self.config.max_samples());
        let mut throughputs = Vec::with_capacity(self.config.max_samples());
        let mut cold_start_ms = 0.0;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let request = CompletionRequest {
                model: "default".to_string(),
                prompt: self.config.prompt.clone(),
                max_tokens: self.config.max_tokens,
                temperature: Some(self.config.temperature),
                stream: false,
            };
            // CB-121: Warmup - errors expected and harmless before server is ready
            drop(self.client.llamacpp_completion(base_url, &request));
        }

        // Measurement loop with CV-based stopping (per Hoefler & Belli SC'15)
        for i in 0..self.config.max_samples() {
            let request = CompletionRequest {
                model: "default".to_string(),
                prompt: self.config.prompt.clone(),
                max_tokens: self.config.max_tokens,
                temperature: Some(self.config.temperature),
                stream: false,
            };

            let timing = self.client.llamacpp_completion(base_url, &request)?;

            latencies.push(timing.total_time_ms);
            if timing.tokens_generated > 0 {
                let tps = (timing.tokens_generated as f64) / (timing.total_time_ms / 1000.0);
                throughputs.push(tps);
            }

            // First measurement is cold start
            if i == 0 {
                cold_start_ms = timing.total_time_ms;
            }

            // Check CV-based stopping criterion
            if let StopDecision::Stop(_) = self.config.cv_criterion.should_stop(&latencies) {
                break;
            }
        }

        // Apply outlier detection if configured
        let (filtered_latencies, outliers_detected, outliers_excluded) =
            if self.config.filter_outliers {
                let outliers = self.outlier_detector.detect(&latencies);
                let outlier_count = outliers.iter().filter(|&&x| x).count();
                let filtered = self.outlier_detector.filter(&latencies);
                (filtered, outlier_count, outlier_count)
            } else {
                (latencies.clone(), 0, 0)
            };

        self.compute_results_with_quality(
            &latencies,
            &filtered_latencies,
            &throughputs,
            cold_start_ms,
            outliers_detected,
            outliers_excluded,
        )
    }

    /// Run benchmark against Ollama server
    ///
    /// Per spec v1.0.1:
    /// 1. Runs optional preflight validation (Jidoka)
    /// 2. Uses CV-based stopping criterion (Hoefler & Belli SC'15)
    /// 3. Applies MAD-based outlier detection
    /// 4. Returns quality metrics for reproducibility assessment
    ///
    /// # Errors
    /// Returns error if preflight fails or server is unreachable
    pub fn benchmark_ollama(&mut self, base_url: &str, model: &str) -> Result<HttpBenchmarkResult> {
        // Preflight validation (Jidoka: fail-fast)
        if self.config.run_preflight {
            self.run_preflight_ollama(base_url)?;
        }

        let mut latencies = Vec::with_capacity(self.config.max_samples());
        let mut throughputs = Vec::with_capacity(self.config.max_samples());
        let mut cold_start_ms = 0.0;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let request = OllamaRequest {
                model: model.to_string(),
                prompt: self.config.prompt.clone(),
                stream: false,
                options: Some(OllamaOptions {
                    num_predict: Some(self.config.max_tokens),
                    temperature: Some(self.config.temperature),
                }),
            };
            // CB-121: Warmup - errors expected and harmless before server is ready
            drop(self.client.ollama_generate(base_url, &request));
        }

        // Measurement loop with CV-based stopping (per Hoefler & Belli SC'15)
        for i in 0..self.config.max_samples() {
            let request = OllamaRequest {
                model: model.to_string(),
                prompt: self.config.prompt.clone(),
                stream: false,
                options: Some(OllamaOptions {
                    num_predict: Some(self.config.max_tokens),
                    temperature: Some(self.config.temperature),
                }),
            };

            let timing = self.client.ollama_generate(base_url, &request)?;

            latencies.push(timing.total_time_ms);
            if timing.tokens_generated > 0 {
                let tps = (timing.tokens_generated as f64) / (timing.total_time_ms / 1000.0);
                throughputs.push(tps);
            }

            if i == 0 {
                cold_start_ms = timing.total_time_ms;
            }

            // Check CV-based stopping criterion
            if let StopDecision::Stop(_) = self.config.cv_criterion.should_stop(&latencies) {
                break;
            }
        }

        // Apply outlier detection if configured
        let (filtered_latencies, outliers_detected, outliers_excluded) =
            if self.config.filter_outliers {
                let outliers = self.outlier_detector.detect(&latencies);
                let outlier_count = outliers.iter().filter(|&&x| x).count();
                let filtered = self.outlier_detector.filter(&latencies);
                (filtered, outlier_count, outlier_count)
            } else {
                (latencies.clone(), 0, 0)
            };

        self.compute_results_with_quality(
            &latencies,
            &filtered_latencies,
            &throughputs,
            cold_start_ms,
            outliers_detected,
            outliers_excluded,
        )
    }

    /// Calculate coefficient of variation (kept for backward compatibility in tests)
    #[cfg(test)]
    fn calculate_cv(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return f64::MAX;
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;

        if mean.abs() < f64::EPSILON {
            return f64::MAX;
        }

        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

        let std_dev = variance.sqrt();
        std_dev / mean
    }

    /// Compute final benchmark results with quality metrics
    ///
    /// Per spec v1.0.1: Uses filtered samples for statistics but reports both raw and filtered
    fn compute_results_with_quality(
        &self,
        raw_latencies: &[f64],
        filtered_latencies: &[f64],
        throughputs: &[f64],
        cold_start_ms: f64,
        outliers_detected: usize,
        outliers_excluded: usize,
    ) -> Result<HttpBenchmarkResult> {
        // Use filtered samples for statistics
        let latencies = if filtered_latencies.is_empty() {
            raw_latencies
        } else {
            filtered_latencies
        };

        if latencies.is_empty() {
            return Err(RealizarError::InferenceError(
                "No valid samples collected".to_string(),
            ));
        }

        let n = latencies.len() as f64;
        let mean_latency_ms = latencies.iter().sum::<f64>() / n;

        let variance = latencies
            .iter()
            .map(|x| (x - mean_latency_ms).powi(2))
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        let std_dev_ms = variance.sqrt();

        let cv = if mean_latency_ms.abs() > f64::EPSILON {
            std_dev_ms / mean_latency_ms
        } else {
            f64::MAX
        };

        // Sort for percentiles
        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50_idx = latencies.len() / 2;
        let p99_idx = (latencies.len() * 99 / 100).min(latencies.len().saturating_sub(1));

        let p50_latency_ms = sorted_latencies.get(p50_idx).copied().unwrap_or(0.0);
        let p99_latency_ms = sorted_latencies.get(p99_idx).copied().unwrap_or(0.0);

        let throughput_tps = if throughputs.is_empty() {
            0.0
        } else {
            throughputs.iter().sum::<f64>() / throughputs.len() as f64
        };

        let cv_converged = cv < self.config.cv_threshold();

        // Build quality metrics
        let quality_metrics = QualityMetrics {
            cv_at_stop: cv,
            cv_converged,
            outliers_detected,
            outliers_excluded,
            preflight_checks_passed: self.preflight_checks_passed(),
        };

        Ok(HttpBenchmarkResult {
            latency_samples: raw_latencies.to_vec(),
            latency_samples_filtered: filtered_latencies.to_vec(),
            mean_latency_ms,
            p50_latency_ms,
            p99_latency_ms,
            std_dev_ms,
            cv_at_stop: cv,
            throughput_tps,
            cold_start_ms,
            sample_count: raw_latencies.len(),
            filtered_sample_count: filtered_latencies.len(),
            cv_converged,
            quality_metrics,
        })
    }

    /// Backward-compatible compute_results (for tests)
    #[cfg(test)]
    fn compute_results(
        latencies: &[f64],
        throughputs: &[f64],
        cold_start_ms: f64,
        cv_threshold: f64,
    ) -> HttpBenchmarkResult {
        let n = latencies.len() as f64;
        let mean_latency_ms = if n > 0.0 {
            latencies.iter().sum::<f64>() / n
        } else {
            0.0
        };

        let variance = if n > 1.0 {
            latencies
                .iter()
                .map(|x| (x - mean_latency_ms).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };
        let std_dev_ms = variance.sqrt();

        let cv = if mean_latency_ms.abs() > f64::EPSILON {
            std_dev_ms / mean_latency_ms
        } else {
            f64::MAX
        };

        // Sort for percentiles
        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50_idx = latencies.len() / 2;
        let p99_idx = (latencies.len() * 99 / 100).min(latencies.len().saturating_sub(1));

        let p50_latency_ms = sorted_latencies.get(p50_idx).copied().unwrap_or(0.0);
        let p99_latency_ms = sorted_latencies.get(p99_idx).copied().unwrap_or(0.0);

        let throughput_tps = if throughputs.is_empty() {
            0.0
        } else {
            throughputs.iter().sum::<f64>() / throughputs.len() as f64
        };

        let cv_converged = cv < cv_threshold;

        HttpBenchmarkResult {
            latency_samples: latencies.to_vec(),
            latency_samples_filtered: latencies.to_vec(), // Same as raw when no filtering
            mean_latency_ms,
            p50_latency_ms,
            p99_latency_ms,
            std_dev_ms,
            cv_at_stop: cv,
            throughput_tps,
            cold_start_ms,
            sample_count: latencies.len(),
            filtered_sample_count: latencies.len(),
            cv_converged,
            quality_metrics: QualityMetrics {
                cv_at_stop: cv,
                cv_converged,
                outliers_detected: 0,
                outliers_excluded: 0,
                preflight_checks_passed: Vec::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests;
