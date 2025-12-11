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
//! ## References
//! - [1] OpenAI API Spec: https://platform.openai.com/docs/api-reference
//! - [2] Ollama API Spec: https://github.com/ollama/ollama/blob/main/docs/api.md

use std::time::Instant;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

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

        Ok(InferenceTiming {
            ttft_ms,
            total_time_ms,
            tokens_generated: ollama_resp.eval_count,
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
#[derive(Debug, Clone)]
pub struct HttpBenchmarkConfig {
    /// Minimum samples before CV check
    pub min_samples: usize,
    /// Maximum samples to collect
    pub max_samples: usize,
    /// Target coefficient of variation (e.g., 0.05 = 5%)
    pub cv_threshold: f64,
    /// Warmup iterations (not counted in stats)
    pub warmup_iterations: usize,
    /// Prompt for inference
    pub prompt: String,
    /// Max tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
}

impl Default for HttpBenchmarkConfig {
    fn default() -> Self {
        Self {
            min_samples: 5,
            max_samples: 30,
            cv_threshold: 0.10, // 10% CV target
            warmup_iterations: 2,
            prompt: "Explain machine learning in one sentence.".to_string(),
            max_tokens: 50,
            temperature: 0.7,
        }
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct HttpBenchmarkResult {
    /// Collected latency samples (ms)
    pub latency_samples: Vec<f64>,
    /// Mean latency (ms)
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
    /// Number of samples collected
    pub sample_count: usize,
    /// Whether CV threshold was achieved
    pub cv_converged: bool,
}

/// HTTP benchmark runner with CV-based stopping
pub struct HttpBenchmarkRunner {
    client: ModelHttpClient,
    config: HttpBenchmarkConfig,
}

impl HttpBenchmarkRunner {
    /// Create a new benchmark runner
    #[must_use]
    pub fn new(config: HttpBenchmarkConfig) -> Self {
        Self {
            client: ModelHttpClient::with_timeout(120), // 2 minute timeout for benchmarks
            config,
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(HttpBenchmarkConfig::default())
    }

    /// Run benchmark against llama.cpp server
    ///
    /// # Errors
    /// Returns error if server is unreachable or all requests fail
    pub fn benchmark_llamacpp(&self, base_url: &str) -> Result<HttpBenchmarkResult> {
        let mut latencies = Vec::with_capacity(self.config.max_samples);
        let mut throughputs = Vec::with_capacity(self.config.max_samples);
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
            let _ = self.client.llamacpp_completion(base_url, &request);
        }

        // Measurement loop with CV-based stopping
        for i in 0..self.config.max_samples {
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

            // Check CV after minimum samples
            if latencies.len() >= self.config.min_samples {
                let cv = Self::calculate_cv(&latencies);
                if cv < self.config.cv_threshold {
                    break;
                }
            }
        }

        Ok(Self::compute_results(
            &latencies,
            &throughputs,
            cold_start_ms,
            self.config.cv_threshold,
        ))
    }

    /// Run benchmark against Ollama server
    ///
    /// # Errors
    /// Returns error if server is unreachable or all requests fail
    pub fn benchmark_ollama(&self, base_url: &str, model: &str) -> Result<HttpBenchmarkResult> {
        let mut latencies = Vec::with_capacity(self.config.max_samples);
        let mut throughputs = Vec::with_capacity(self.config.max_samples);
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
            let _ = self.client.ollama_generate(base_url, &request);
        }

        // Measurement loop with CV-based stopping
        for i in 0..self.config.max_samples {
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

            if latencies.len() >= self.config.min_samples {
                let cv = Self::calculate_cv(&latencies);
                if cv < self.config.cv_threshold {
                    break;
                }
            }
        }

        Ok(Self::compute_results(
            &latencies,
            &throughputs,
            cold_start_ms,
            self.config.cv_threshold,
        ))
    }

    /// Calculate coefficient of variation
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

    /// Compute final benchmark results
    fn compute_results(
        latencies: &[f64],
        throughputs: &[f64],
        cold_start_ms: f64,
        cv_threshold: f64,
    ) -> HttpBenchmarkResult {
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

        HttpBenchmarkResult {
            latency_samples: latencies.to_vec(),
            mean_latency_ms,
            p50_latency_ms,
            p99_latency_ms,
            std_dev_ms,
            cv_at_stop: cv,
            throughput_tps,
            cold_start_ms,
            sample_count: latencies.len(),
            cv_converged: cv < cv_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Unit Tests (No network required)
    // =========================================================================

    #[test]
    fn test_client_creation() {
        let client = ModelHttpClient::new();
        assert_eq!(client.timeout_secs(), 60);
    }

    #[test]
    fn test_client_custom_timeout() {
        let client = ModelHttpClient::with_timeout(120);
        assert_eq!(client.timeout_secs(), 120);
    }

    #[test]
    fn test_completion_request_serialization() {
        let request = CompletionRequest {
            model: "llama2".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 100,
            temperature: Some(0.7),
            stream: false,
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("\"model\":\"llama2\""));
        assert!(json.contains("\"prompt\":\"Hello\""));
        assert!(json.contains("\"max_tokens\":100"));
    }

    #[test]
    fn test_ollama_request_serialization() {
        let request = OllamaRequest {
            model: "llama2".to_string(),
            prompt: "Hello".to_string(),
            stream: false,
            options: Some(OllamaOptions {
                num_predict: Some(100),
                temperature: Some(0.7),
            }),
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("\"model\":\"llama2\""));
        assert!(json.contains("\"prompt\":\"Hello\""));
    }

    #[test]
    fn test_completion_response_deserialization() {
        let json = r#"{
            "id": "cmpl-123",
            "choices": [{"text": "World!", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;

        let response: CompletionResponse =
            serde_json::from_str(json).expect("deserialization failed");

        assert_eq!(response.id, "cmpl-123");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].text, "World!");
    }

    #[test]
    fn test_ollama_response_deserialization() {
        let json = r#"{
            "model": "llama2",
            "response": "Hello back!",
            "done": true,
            "total_duration": 5000000000,
            "load_duration": 1000000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 2000000000,
            "eval_count": 5,
            "eval_duration": 2000000000
        }"#;

        let response: OllamaResponse = serde_json::from_str(json).expect("deserialization failed");

        assert_eq!(response.model, "llama2");
        assert_eq!(response.response, "Hello back!");
        assert!(response.done);
        assert_eq!(response.eval_count, 5);
    }

    // =========================================================================
    // Integration Tests (Require running servers)
    // Mark with #[ignore] for CI - run manually with: cargo test -- --ignored
    // =========================================================================

    #[test]
    #[ignore = "Requires vLLM server at localhost:8000"]
    fn test_vllm_real_inference() {
        let client = ModelHttpClient::new();

        let request = CompletionRequest {
            model: "meta-llama/Llama-2-7b-hf".to_string(),
            prompt: "The capital of France is".to_string(),
            max_tokens: 20,
            temperature: Some(0.1),
            stream: false,
        };

        let result = client.openai_completion("http://localhost:8000", &request, None);

        // This MUST succeed with a real server
        let timing = result.expect("vLLM inference failed - is server running?");

        // Verify we got REAL data, not mock data
        assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
        assert!(
            timing.total_time_ms > 0.0,
            "Total time must be positive (real latency)"
        );
        assert!(!timing.text.is_empty(), "Must get actual generated text");

        println!("vLLM Real Inference:");
        println!("  TTFT: {:.2}ms", timing.ttft_ms);
        println!("  Total: {:.2}ms", timing.total_time_ms);
        println!("  Tokens: {}", timing.tokens_generated);
        println!("  Text: {}", timing.text);
    }

    #[test]
    #[ignore = "Requires Ollama server at localhost:11434"]
    fn test_ollama_real_inference() {
        let client = ModelHttpClient::new();

        let request = OllamaRequest {
            model: "llama2".to_string(),
            prompt: "The capital of France is".to_string(),
            stream: false,
            options: Some(OllamaOptions {
                num_predict: Some(20),
                temperature: Some(0.1),
            }),
        };

        let result = client.ollama_generate("http://localhost:11434", &request);

        // This MUST succeed with a real server
        let timing = result.expect("Ollama inference failed - is server running?");

        // Verify we got REAL data, not mock data
        assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
        assert!(
            timing.total_time_ms > 0.0,
            "Total time must be positive (real latency)"
        );
        assert!(!timing.text.is_empty(), "Must get actual generated text");

        println!("Ollama Real Inference:");
        println!("  TTFT: {:.2}ms", timing.ttft_ms);
        println!("  Total: {:.2}ms", timing.total_time_ms);
        println!("  Tokens: {}", timing.tokens_generated);
        println!("  Text: {}", timing.text);
    }

    #[test]
    #[ignore = "Requires llama.cpp server at localhost:8080"]
    fn test_llamacpp_real_inference() {
        let client = ModelHttpClient::new();

        let request = CompletionRequest {
            model: "default".to_string(), // llama.cpp uses loaded model
            prompt: "The capital of France is".to_string(),
            max_tokens: 20,
            temperature: Some(0.1),
            stream: false,
        };

        let result = client.openai_completion("http://localhost:8080", &request, None);

        let timing = result.expect("llama.cpp inference failed - is server running?");

        assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
        assert!(
            timing.total_time_ms > 0.0,
            "Total time must be positive (real latency)"
        );

        println!("llama.cpp Real Inference:");
        println!("  TTFT: {:.2}ms", timing.ttft_ms);
        println!("  Total: {:.2}ms", timing.total_time_ms);
        println!("  Tokens: {}", timing.tokens_generated);
        println!("  Text: {}", timing.text);
    }

    #[test]
    fn test_connection_error_handling() {
        let client = ModelHttpClient::with_timeout(1); // 1 second timeout

        let request = CompletionRequest {
            model: "test".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: None,
            stream: false,
        };

        // This should fail because no server is running on this port
        let result = client.openai_completion("http://localhost:59999", &request, None);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            RealizarError::ConnectionError(msg) => {
                assert!(msg.contains("HTTP request failed"));
            },
            other => panic!("Expected ConnectionError, got: {:?}", other),
        }
    }

    #[test]
    fn test_ollama_connection_error() {
        let client = ModelHttpClient::with_timeout(1);

        let request = OllamaRequest {
            model: "test".to_string(),
            prompt: "Hello".to_string(),
            stream: false,
            options: None,
        };

        let result = client.ollama_generate("http://localhost:59998", &request);

        assert!(result.is_err());
    }

    #[test]
    fn test_llamacpp_response_deserialization() {
        let json = r#"{
            "content": "Machine learning is a subset of AI.",
            "model": "/path/to/model.gguf",
            "tokens_predicted": 8,
            "tokens_evaluated": 5,
            "stop": true,
            "timings": {
                "prompt_n": 5,
                "prompt_ms": 10.5,
                "predicted_n": 8,
                "predicted_ms": 25.3,
                "predicted_per_second": 316.2
            }
        }"#;

        let response: LlamaCppResponse =
            serde_json::from_str(json).expect("deserialization failed");

        assert_eq!(response.content, "Machine learning is a subset of AI.");
        assert_eq!(response.tokens_predicted, 8);
        assert_eq!(response.tokens_evaluated, 5);
        assert!(response.stop);

        let timings = response.timings.expect("timings should be present");
        assert_eq!(timings.prompt_n, 5);
        assert_eq!(timings.predicted_n, 8);
        assert!((timings.predicted_per_second - 316.2).abs() < 0.1);
    }

    #[test]
    fn test_llamacpp_response_minimal() {
        // llama.cpp response with only required field
        let json = r#"{"content": "Hello world"}"#;

        let response: LlamaCppResponse =
            serde_json::from_str(json).expect("deserialization failed");

        assert_eq!(response.content, "Hello world");
        assert_eq!(response.tokens_predicted, 0); // default
        assert!(response.timings.is_none());
    }

    #[test]
    fn test_llamacpp_connection_error() {
        let client = ModelHttpClient::with_timeout(1);

        let request = CompletionRequest {
            model: "test".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: None,
            stream: false,
        };

        let result = client.llamacpp_completion("http://localhost:59997", &request);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            RealizarError::ConnectionError(msg) => {
                assert!(msg.contains("HTTP request failed"));
            },
            other => panic!("Expected ConnectionError, got: {:?}", other),
        }
    }

    // =========================================================================
    // HTTP Benchmark Runner Tests
    // =========================================================================

    #[test]
    fn test_benchmark_config_default() {
        let config = HttpBenchmarkConfig::default();
        assert_eq!(config.min_samples, 5);
        assert_eq!(config.max_samples, 30);
        assert!((config.cv_threshold - 0.10).abs() < 0.001);
        assert_eq!(config.warmup_iterations, 2);
    }

    #[test]
    fn test_benchmark_runner_creation() {
        let runner = HttpBenchmarkRunner::with_defaults();
        assert_eq!(runner.config.min_samples, 5);
    }

    #[test]
    fn test_cv_calculation_identical_values() {
        // Identical values should have CV = 0
        let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let cv = HttpBenchmarkRunner::calculate_cv(&samples);
        assert!(
            cv < 0.001,
            "CV of identical values should be ~0, got {}",
            cv
        );
    }

    #[test]
    fn test_cv_calculation_varied_values() {
        // Known CV case: mean=100, std=10, CV=0.1
        let samples = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let cv = HttpBenchmarkRunner::calculate_cv(&samples);
        // CV should be around 0.079 (std ~7.9, mean 100)
        assert!(
            cv > 0.05 && cv < 0.15,
            "CV should be reasonable, got {}",
            cv
        );
    }

    #[test]
    fn test_cv_calculation_single_value() {
        let samples = vec![100.0];
        let cv = HttpBenchmarkRunner::calculate_cv(&samples);
        assert_eq!(cv, f64::MAX, "Single value should return MAX CV");
    }

    #[test]
    fn test_cv_calculation_empty() {
        let samples: Vec<f64> = vec![];
        let cv = HttpBenchmarkRunner::calculate_cv(&samples);
        assert_eq!(cv, f64::MAX, "Empty samples should return MAX CV");
    }

    #[test]
    fn test_compute_results_basic() {
        let latencies = vec![100.0, 110.0, 90.0, 105.0, 95.0];
        let throughputs = vec![50.0, 45.0, 55.0, 48.0, 52.0];
        let cold_start = 120.0;
        let cv_threshold = 0.10;

        let result = HttpBenchmarkRunner::compute_results(
            &latencies,
            &throughputs,
            cold_start,
            cv_threshold,
        );

        assert_eq!(result.sample_count, 5);
        assert!((result.mean_latency_ms - 100.0).abs() < 0.01);
        assert!(result.p50_latency_ms > 0.0);
        assert!(result.p99_latency_ms >= result.p50_latency_ms);
        assert!(result.throughput_tps > 0.0);
        assert_eq!(result.cold_start_ms, 120.0);
    }

    #[test]
    fn test_compute_results_percentiles() {
        // Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let latencies: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let throughputs = vec![];
        let cold_start = 1.0;
        let cv_threshold = 0.10;

        let result = HttpBenchmarkRunner::compute_results(
            &latencies,
            &throughputs,
            cold_start,
            cv_threshold,
        );

        // p50 at index 5 = 6.0
        assert!((result.p50_latency_ms - 6.0).abs() < 0.1);
        // p99 at index 9 = 10.0
        assert!((result.p99_latency_ms - 10.0).abs() < 0.1);
    }

    #[test]
    #[ignore = "Requires llama.cpp server at localhost:8082"]
    fn test_benchmark_runner_llamacpp() {
        let config = HttpBenchmarkConfig {
            min_samples: 3,
            max_samples: 5,
            cv_threshold: 0.50, // Relaxed for test
            warmup_iterations: 1,
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.1,
        };

        let runner = HttpBenchmarkRunner::new(config);
        let result = runner
            .benchmark_llamacpp("http://localhost:8082")
            .expect("Benchmark failed - is llama.cpp running?");

        assert!(result.sample_count >= 3);
        assert!(result.mean_latency_ms > 0.0);
        assert!(result.throughput_tps > 0.0);

        println!("llama.cpp Benchmark Results:");
        println!("  Samples: {}", result.sample_count);
        println!("  Mean: {:.2}ms", result.mean_latency_ms);
        println!("  P50: {:.2}ms", result.p50_latency_ms);
        println!("  P99: {:.2}ms", result.p99_latency_ms);
        println!("  TPS: {:.2}", result.throughput_tps);
        println!("  CV: {:.4}", result.cv_at_stop);
        println!("  Converged: {}", result.cv_converged);
    }
}
