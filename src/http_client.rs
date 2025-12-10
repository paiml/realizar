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
}
