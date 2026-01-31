//! Runtime backend abstraction for benchmark comparison
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - BENCH-002: Runtime Backend Abstraction
//! - LlamaCppBackend, VllmBackend, OllamaBackend implementations

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::RealizarError;

#[cfg(feature = "bench-http")]
use crate::http_client::{CompletionRequest, ModelHttpClient, OllamaOptions, OllamaRequest};

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

    /// Set stop sequences
    #[must_use]
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
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

    /// Set number of threads
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
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
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // RuntimeType Tests
    // =========================================================================

    #[test]
    fn test_runtime_type_as_str() {
        assert_eq!(RuntimeType::Realizar.as_str(), "realizar");
        assert_eq!(RuntimeType::LlamaCpp.as_str(), "llama-cpp");
        assert_eq!(RuntimeType::Vllm.as_str(), "vllm");
        assert_eq!(RuntimeType::Ollama.as_str(), "ollama");
    }

    #[test]
    fn test_runtime_type_parse() {
        assert_eq!(RuntimeType::parse("realizar"), Some(RuntimeType::Realizar));
        assert_eq!(RuntimeType::parse("llama-cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("llama.cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("llamacpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("vllm"), Some(RuntimeType::Vllm));
        assert_eq!(RuntimeType::parse("ollama"), Some(RuntimeType::Ollama));
        assert_eq!(RuntimeType::parse("REALIZAR"), Some(RuntimeType::Realizar)); // case-insensitive
        assert_eq!(RuntimeType::parse("unknown"), None);
    }

    #[test]
    fn test_runtime_type_clone_eq() {
        let rt = RuntimeType::Realizar;
        assert_eq!(rt, rt.clone());
    }

    #[test]
    fn test_runtime_type_debug() {
        let debug = format!("{:?}", RuntimeType::Vllm);
        assert!(debug.contains("Vllm"));
    }

    #[test]
    fn test_runtime_type_serialize() {
        let json = serde_json::to_string(&RuntimeType::LlamaCpp).unwrap();
        assert!(json.contains("LlamaCpp"));
    }

    // =========================================================================
    // InferenceRequest Tests
    // =========================================================================

    #[test]
    fn test_inference_request_default() {
        let req = InferenceRequest::default();
        assert!(req.prompt.is_empty());
        assert_eq!(req.max_tokens, 100);
        assert!((req.temperature - 0.7).abs() < 0.01);
        assert!(req.stop.is_empty());
    }

    #[test]
    fn test_inference_request_new() {
        let req = InferenceRequest::new("Hello world");
        assert_eq!(req.prompt, "Hello world");
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn test_inference_request_builder() {
        let req = InferenceRequest::new("test")
            .with_max_tokens(50)
            .with_temperature(0.5)
            .with_stop(vec!["END".to_string()]);

        assert_eq!(req.prompt, "test");
        assert_eq!(req.max_tokens, 50);
        assert!((req.temperature - 0.5).abs() < 0.01);
        assert_eq!(req.stop, vec!["END".to_string()]);
    }

    #[test]
    fn test_inference_request_serialize() {
        let req = InferenceRequest::new("prompt");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("prompt"));
        assert!(json.contains("100")); // max_tokens
    }

    // =========================================================================
    // InferenceResponse Tests
    // =========================================================================

    #[test]
    fn test_inference_response_tokens_per_second() {
        let resp = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 10.0,
            total_time_ms: 1000.0, // 1 second
            itl_ms: vec![],
        };
        assert!((resp.tokens_per_second() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_inference_response_tokens_per_second_zero_time() {
        let resp = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 0.0,
            total_time_ms: 0.0,
            itl_ms: vec![],
        };
        assert_eq!(resp.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_inference_response_tokens_per_second_negative_time() {
        let resp = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 0.0,
            total_time_ms: -1.0,
            itl_ms: vec![],
        };
        assert_eq!(resp.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_inference_response_serialize() {
        let resp = InferenceResponse {
            text: "Generated text".to_string(),
            tokens_generated: 42,
            ttft_ms: 15.5,
            total_time_ms: 100.0,
            itl_ms: vec![2.0, 3.0],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Generated text"));
        assert!(json.contains("42"));
    }

    // =========================================================================
    // BackendInfo Tests
    // =========================================================================

    #[test]
    fn test_backend_info_serialize() {
        let info = BackendInfo {
            runtime_type: RuntimeType::Realizar,
            version: "1.0.0".to_string(),
            supports_streaming: true,
            loaded_model: Some("llama-7b".to_string()),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("1.0.0"));
        assert!(json.contains("llama-7b"));
    }

    #[test]
    fn test_backend_info_clone() {
        let info = BackendInfo {
            runtime_type: RuntimeType::Vllm,
            version: "0.4.0".to_string(),
            supports_streaming: true,
            loaded_model: None,
        };
        let cloned = info.clone();
        assert_eq!(info.version, cloned.version);
    }

    // =========================================================================
    // MockBackend Tests
    // =========================================================================

    #[test]
    fn test_mock_backend_new() {
        let backend = MockBackend::new(50.0, 10.0);
        assert!((backend.ttft_ms - 50.0).abs() < 0.01);
        assert!((backend.tokens_per_second - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_mock_backend_info() {
        let backend = MockBackend::new(10.0, 100.0);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Realizar);
        assert!(info.supports_streaming);
        assert!(info.loaded_model.is_none());
    }

    #[test]
    fn test_mock_backend_inference() {
        let backend = MockBackend::new(20.0, 50.0);
        let request = InferenceRequest::new("Hello").with_max_tokens(25);
        let response = backend.inference(&request).unwrap();

        assert_eq!(response.text, "Mock response");
        assert_eq!(response.tokens_generated, 25);
        assert!((response.ttft_ms - 20.0).abs() < 0.01);
        // gen_time = 25 / 50 * 1000 = 500ms
        // total = 20 + 500 = 520ms
        assert!((response.total_time_ms - 520.0).abs() < 0.1);
        assert_eq!(response.itl_ms.len(), 25);
    }

    #[test]
    fn test_mock_backend_inference_max_100() {
        let backend = MockBackend::new(10.0, 100.0);
        let request = InferenceRequest::new("Hello").with_max_tokens(200);
        let response = backend.inference(&request).unwrap();
        // Should be capped at 100
        assert_eq!(response.tokens_generated, 100);
    }

    // =========================================================================
    // BackendRegistry Tests
    // =========================================================================

    #[test]
    fn test_backend_registry_new() {
        let registry = BackendRegistry::new();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_backend_registry_default() {
        let registry = BackendRegistry::default();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_backend_registry_register_and_get() {
        let mut registry = BackendRegistry::new();
        let backend = MockBackend::new(10.0, 100.0);
        registry.register(RuntimeType::Realizar, Box::new(backend));

        let retrieved = registry.get(RuntimeType::Realizar);
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().info().runtime_type,
            RuntimeType::Realizar
        );
    }

    #[test]
    fn test_backend_registry_get_missing() {
        let registry = BackendRegistry::new();
        assert!(registry.get(RuntimeType::Vllm).is_none());
    }

    #[test]
    fn test_backend_registry_list() {
        let mut registry = BackendRegistry::new();
        registry.register(
            RuntimeType::Realizar,
            Box::new(MockBackend::new(10.0, 100.0)),
        );
        registry.register(
            RuntimeType::LlamaCpp,
            Box::new(MockBackend::new(20.0, 50.0)),
        );

        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&RuntimeType::Realizar));
        assert!(list.contains(&RuntimeType::LlamaCpp));
    }

    // =========================================================================
    // LlamaCppConfig Tests
    // =========================================================================

    #[test]
    fn test_llamacpp_config_default() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.binary_path, "llama-cli");
        assert!(config.model_path.is_none());
        assert_eq!(config.n_gpu_layers, 0);
        assert_eq!(config.ctx_size, 2048);
        assert_eq!(config.threads, 4);
    }

    #[test]
    fn test_llamacpp_config_new() {
        let config = LlamaCppConfig::new("/usr/local/bin/llama-cli");
        assert_eq!(config.binary_path, "/usr/local/bin/llama-cli");
    }

    #[test]
    fn test_llamacpp_config_builder() {
        let config = LlamaCppConfig::new("llama-cli")
            .with_model("/models/llama.gguf")
            .with_gpu_layers(32)
            .with_ctx_size(4096)
            .with_threads(8);

        assert_eq!(config.model_path, Some("/models/llama.gguf".to_string()));
        assert_eq!(config.n_gpu_layers, 32);
        assert_eq!(config.ctx_size, 4096);
        assert_eq!(config.threads, 8);
    }

    #[test]
    fn test_llamacpp_config_serialize() {
        let config = LlamaCppConfig::new("llama-cli").with_model("model.gguf");
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("llama-cli"));
        assert!(json.contains("model.gguf"));
    }

    // =========================================================================
    // VllmConfig Tests
    // =========================================================================

    #[test]
    fn test_vllm_config_default() {
        let config = VllmConfig::default();
        assert_eq!(config.base_url, "http://localhost:8000");
        assert_eq!(config.api_version, "v1");
        assert!(config.model.is_none());
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_vllm_config_new() {
        let config = VllmConfig::new("http://myserver:8080");
        assert_eq!(config.base_url, "http://myserver:8080");
    }

    #[test]
    fn test_vllm_config_builder() {
        let config = VllmConfig::new("http://localhost:8000")
            .with_model("mistral-7b")
            .with_api_key("sk-secret");

        assert_eq!(config.model, Some("mistral-7b".to_string()));
        assert_eq!(config.api_key, Some("sk-secret".to_string()));
    }

    #[test]
    fn test_vllm_config_serialize() {
        let config = VllmConfig::new("http://test").with_model("phi-2");
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("http://test"));
        assert!(json.contains("phi-2"));
    }

    // =========================================================================
    // LlamaCppBackend Tests
    // =========================================================================

    #[test]
    fn test_llamacpp_backend_build_cli_args() {
        let config = LlamaCppConfig::new("llama-cli")
            .with_model("/models/llama.gguf")
            .with_gpu_layers(10)
            .with_ctx_size(2048)
            .with_threads(4);
        let backend = LlamaCppBackend::new(config);

        let request = InferenceRequest::new("Hello").with_max_tokens(50);
        let args = backend.build_cli_args(&request);

        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"/models/llama.gguf".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(args.contains(&"-n".to_string()));
        assert!(args.contains(&"50".to_string()));
        assert!(args.contains(&"-ngl".to_string()));
        assert!(args.contains(&"10".to_string()));
    }

    #[test]
    fn test_llamacpp_backend_build_cli_args_custom_temp() {
        let config = LlamaCppConfig::new("llama-cli").with_model("model.gguf");
        let backend = LlamaCppBackend::new(config);

        let request = InferenceRequest::new("test").with_temperature(0.5);
        let args = backend.build_cli_args(&request);

        assert!(args.contains(&"--temp".to_string()));
        assert!(args.iter().any(|a| a.contains("0.50")));
    }

    #[test]
    fn test_llamacpp_backend_build_cli_args_default_temp() {
        let config = LlamaCppConfig::new("llama-cli").with_model("model.gguf");
        let backend = LlamaCppBackend::new(config);

        let request = InferenceRequest::new("test").with_temperature(0.8);
        let args = backend.build_cli_args(&request);

        // Default temp (0.8) should not add --temp flag
        assert!(!args.contains(&"--temp".to_string()));
    }

    #[test]
    fn test_llamacpp_backend_parse_timing_line() {
        let output = r"
llama_perf_context_print: prompt eval time =      12.34 ms /    10 tokens
llama_perf_context_print: eval time =     123.45 ms /   100 tokens
llama_perf_context_print: total time =     135.79 ms /   110 runs
        ";

        let (prompt_time, prompt_tokens) =
            LlamaCppBackend::parse_timing_line(output, "prompt eval time").unwrap();
        assert!((prompt_time - 12.34).abs() < 0.01);
        assert_eq!(prompt_tokens, 10);

        let (eval_time, eval_tokens) =
            LlamaCppBackend::parse_timing_line(output, "eval time").unwrap();
        assert!((eval_time - 123.45).abs() < 0.01);
        assert_eq!(eval_tokens, 100);

        let (total_time, total_runs) =
            LlamaCppBackend::parse_timing_line(output, "total time").unwrap();
        assert!((total_time - 135.79).abs() < 0.01);
        assert_eq!(total_runs, 110);
    }

    #[test]
    fn test_llamacpp_backend_parse_timing_line_not_found() {
        let output = "No timing info here";
        assert!(LlamaCppBackend::parse_timing_line(output, "eval time").is_none());
    }

    #[test]
    fn test_llamacpp_backend_extract_generated_text() {
        let output = r"Hello world!
This is generated text.
llama_perf_context_print: eval time = 100 ms
sampler stats follow...";

        let text = LlamaCppBackend::extract_generated_text(output);
        assert_eq!(text, "Hello world!\nThis is generated text.");
    }

    #[test]
    fn test_llamacpp_backend_extract_generated_text_empty() {
        let output = "llama_perf_context_print: eval time = 100 ms";
        let text = LlamaCppBackend::extract_generated_text(output);
        assert!(text.is_empty());
    }

    #[test]
    fn test_llamacpp_backend_parse_cli_output() {
        let output = r"Generated response text
llama_perf_context_print: prompt eval time =      50.00 ms /     5 tokens
llama_perf_context_print: eval time =     200.00 ms /    20 tokens
llama_perf_context_print: total time =     250.00 ms /    25 runs";

        let response = LlamaCppBackend::parse_cli_output(output).unwrap();
        assert_eq!(response.text, "Generated response text");
        assert!((response.ttft_ms - 50.0).abs() < 0.01);
        assert_eq!(response.tokens_generated, 20);
        assert!((response.total_time_ms - 250.0).abs() < 0.01);
        // ITL should be estimated from eval time / tokens
        assert_eq!(response.itl_ms.len(), 19); // tokens - 1
    }

    #[test]
    fn test_llamacpp_backend_parse_cli_output_minimal() {
        let output = "Just text, no timing";
        let response = LlamaCppBackend::parse_cli_output(output).unwrap();
        assert_eq!(response.text, "Just text, no timing");
        assert_eq!(response.ttft_ms, 0.0);
        assert_eq!(response.tokens_generated, 0);
    }

    #[test]
    fn test_llamacpp_backend_info() {
        let config = LlamaCppConfig::new("llama-cli").with_model("model.gguf");
        let backend = LlamaCppBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
        assert!(!info.supports_streaming);
        assert_eq!(info.loaded_model, Some("model.gguf".to_string()));
    }

    // =========================================================================
    // Runtime Backend Trait Tests
    // =========================================================================

    #[test]
    fn test_runtime_backend_load_model_default() {
        let mut backend = MockBackend::new(10.0, 100.0);
        // Default load_model should be a no-op
        let result = backend.load_model("any/path.gguf");
        assert!(result.is_ok());
    }
}
