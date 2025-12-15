//! Preflight Validation Protocol for Deterministic Benchmarking
//!
//! Implements the Toyota Way principles for benchmark quality:
//! - Jidoka: Fail-fast validation, stop on anomaly
//! - Poka-yoke: Error-proofing through type-safe configurations
//! - Genchi Genbutsu: Verify actual system state
//!
//! References:
//! - Hoefler & Belli SC'15 [1]: CV-based stopping
//! - Vitek & Kalibera EMSOFT'11 [7]: Reproducibility requirements
//! - Liker [9]: Toyota Way principles

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Specification Constants (per spec v1.0.1 Section 4.3)
// ============================================================================

/// Canonical benchmark inputs - versioned for reproducibility
pub mod canonical_inputs {
    /// Version: Changing inputs MUST increment this version
    pub const VERSION: &str = "1.0.0";

    /// Fixed prompt for latency benchmarks
    pub const LATENCY_PROMPT: &str = "Explain the concept of machine learning in one sentence.";

    /// Fixed token sequence for throughput benchmarks
    pub const THROUGHPUT_TOKENS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8];

    /// Fixed max tokens for generation benchmarks
    pub const MAX_TOKENS: usize = 50;

    /// Specification version for metadata
    pub const SPEC_VERSION: &str = "1.0.1";
}

// ============================================================================
// Error Types (Poka-yoke: Make errors explicit and typed)
// ============================================================================

/// Preflight validation errors with detailed context for diagnosis
#[derive(Debug, Error)]
pub enum PreflightError {
    /// Server is not reachable at the specified URL
    #[error("Server unreachable at {url}: {reason}")]
    ServerUnreachable {
        /// The URL that was unreachable
        url: String,
        /// The reason for the failure
        reason: String,
    },

    /// Health check endpoint failed
    #[error("Health check failed at {url}: HTTP {status}")]
    HealthCheckFailed {
        /// The health check URL
        url: String,
        /// HTTP status code returned
        status: u16,
    },

    /// Required model not found in backend
    #[error("Model not found: requested '{requested}', available: {available:?}")]
    ModelNotFound {
        /// Model name that was requested
        requested: String,
        /// List of available models
        available: Vec<String>,
    },

    /// Response schema does not match expected format
    #[error("Schema mismatch: missing field '{missing_field}'")]
    SchemaMismatch {
        /// The field that was expected but missing
        missing_field: String,
    },

    /// Field type does not match expected type
    #[error("Field type mismatch: '{field}' expected {expected}, got {actual}")]
    FieldTypeMismatch {
        /// Field name with type mismatch
        field: String,
        /// Expected type name
        expected: String,
        /// Actual type received
        actual: String,
    },

    /// Response parsing failed
    #[error("Response parse error: {reason}")]
    ResponseParseError {
        /// Description of the parse failure
        reason: String,
    },

    /// Timeout during preflight check
    #[error("Timeout after {duration:?} during {operation}")]
    Timeout {
        /// How long the operation waited before timing out
        duration: Duration,
        /// Name of the operation that timed out
        operation: String,
    },

    /// Configuration error
    #[error("Configuration error: {reason}")]
    ConfigError {
        /// Description of the configuration error
        reason: String,
    },
}

/// Result type for preflight operations
pub type PreflightResult<T> = Result<T, PreflightError>;

// ============================================================================
// Preflight Check Trait (Jidoka: Quality built into every step)
// ============================================================================

/// Trait for preflight validation checks
///
/// Implements Jidoka principle: each check validates one aspect of system state
/// and fails fast if the condition is not met.
pub trait PreflightCheck: fmt::Debug + Send + Sync {
    /// Unique identifier for this check (for logging and metrics)
    fn name(&self) -> &'static str;

    /// Validate the condition, returning Ok(()) if passed
    ///
    /// # Errors
    /// Returns `PreflightError` with detailed context if validation fails
    fn validate(&self) -> PreflightResult<()>;

    /// Optional: Description of what this check validates
    fn description(&self) -> &'static str {
        "Preflight validation check"
    }
}

// ============================================================================
// Deterministic Inference Configuration (per spec Section 4.1)
// ============================================================================

/// Configuration for deterministic inference
///
/// Per Fleming & Wallace [5], deterministic benchmarks require seed control
/// and elimination of randomness sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicInferenceConfig {
    /// Temperature = 0.0 disables sampling randomness
    pub temperature: f64,
    /// Fixed seed for any remaining randomness
    pub seed: u64,
    /// Top-k = 1 forces greedy decoding
    pub top_k: usize,
    /// Top-p = 1.0 disables nucleus sampling
    pub top_p: f64,
}

impl Default for DeterministicInferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0, // Greedy decoding
            seed: 42,         // Fixed, reproducible seed
            top_k: 1,         // Deterministic token selection
            top_p: 1.0,       // Disable nucleus sampling
        }
    }
}

impl DeterministicInferenceConfig {
    /// Create a new deterministic config with custom seed
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Validate that this config is truly deterministic
    ///
    /// # Errors
    /// Returns `PreflightError::ConfigError` if any parameter allows non-determinism:
    /// - temperature > 0.0 (allows sampling randomness)
    /// - top_k != 1 (allows multiple token choices)
    pub fn validate_determinism(&self) -> PreflightResult<()> {
        if self.temperature > 0.0 {
            return Err(PreflightError::ConfigError {
                reason: format!(
                    "Temperature {} > 0.0 allows randomness; set to 0.0 for determinism",
                    self.temperature
                ),
            });
        }
        if self.top_k != 1 {
            return Err(PreflightError::ConfigError {
                reason: format!(
                    "top_k {} != 1 allows multiple token choices; set to 1 for determinism",
                    self.top_k
                ),
            });
        }
        Ok(())
    }
}

// ============================================================================
// CV-Based Stopping Criterion (per spec Section 3.1)
// ============================================================================

/// Reason for stopping benchmark iteration
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// CV threshold achieved (statistically sufficient)
    CvConverged(f64),
    /// Maximum samples reached (bounded resource usage)
    MaxSamples,
    /// Minimum samples not yet reached
    Continue,
}

/// Decision from stopping criterion
#[derive(Debug, Clone, PartialEq)]
pub enum StopDecision {
    /// Continue collecting samples
    Continue,
    /// Stop with given reason
    Stop(StopReason),
}

/// CV-based stopping criterion per Hoefler & Belli SC'15 [1]
#[derive(Debug, Clone)]
pub struct CvStoppingCriterion {
    /// Minimum samples before CV check (prevents premature stopping)
    pub min_samples: usize,
    /// Maximum samples (bounded resource usage)
    pub max_samples: usize,
    /// Target CV threshold (e.g., 0.05 = 5%)
    pub cv_threshold: f64,
}

impl Default for CvStoppingCriterion {
    fn default() -> Self {
        Self {
            min_samples: 5,
            max_samples: 30,
            cv_threshold: 0.05, // 5% CV target per SC'15
        }
    }
}

impl CvStoppingCriterion {
    /// Create new criterion with custom parameters
    #[must_use]
    pub fn new(min_samples: usize, max_samples: usize, cv_threshold: f64) -> Self {
        Self {
            min_samples,
            max_samples,
            cv_threshold,
        }
    }

    /// Evaluate whether to stop based on current samples
    #[must_use]
    pub fn should_stop(&self, samples: &[f64]) -> StopDecision {
        if samples.len() < self.min_samples {
            return StopDecision::Continue;
        }
        if samples.len() >= self.max_samples {
            return StopDecision::Stop(StopReason::MaxSamples);
        }

        let cv = self.calculate_cv(samples);
        if cv < self.cv_threshold {
            StopDecision::Stop(StopReason::CvConverged(cv))
        } else {
            StopDecision::Continue
        }
    }

    /// Calculate coefficient of variation (std_dev / mean)
    #[must_use]
    pub fn calculate_cv(&self, samples: &[f64]) -> f64 {
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
}

// ============================================================================
// Outlier Detection (per spec Section 3.3)
// ============================================================================

/// Outlier detection using Median Absolute Deviation (MAD)
///
/// Per Chen et al. [4], MAD is more robust than standard deviation
/// for samples with potential outliers.
pub struct OutlierDetector {
    /// k-factor for threshold (3.0 = ~99.7% for normal distribution)
    pub k_factor: f64,
}

impl Default for OutlierDetector {
    fn default() -> Self {
        Self { k_factor: 3.0 }
    }
}

impl OutlierDetector {
    /// Create detector with custom k-factor
    #[must_use]
    pub fn new(k_factor: f64) -> Self {
        Self { k_factor }
    }

    /// Detect outliers in samples
    ///
    /// Returns a boolean vector where `true` indicates an outlier
    #[must_use]
    pub fn detect(&self, samples: &[f64]) -> Vec<bool> {
        if samples.len() < 3 {
            return vec![false; samples.len()];
        }

        let median = Self::percentile(samples, 50.0);
        let deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();
        let mad = Self::percentile(&deviations, 50.0);

        // 1.4826 scales MAD to equivalent std dev for normal distribution
        let threshold = self.k_factor * mad * 1.4826;

        samples
            .iter()
            .map(|x| (x - median).abs() > threshold)
            .collect()
    }

    /// Filter outliers from samples, returning clean samples
    #[must_use]
    pub fn filter(&self, samples: &[f64]) -> Vec<f64> {
        let outliers = self.detect(samples);
        samples
            .iter()
            .zip(outliers.iter())
            .filter(|(_, is_outlier)| !**is_outlier)
            .map(|(sample, _)| *sample)
            .collect()
    }

    /// Calculate percentile of samples
    fn percentile(samples: &[f64], p: f64) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

// ============================================================================
// Quality Metrics (per spec Section 6.1)
// ============================================================================

/// Quality metrics for benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// CV at the point where benchmark stopped
    pub cv_at_stop: f64,
    /// Whether CV threshold was achieved
    pub cv_converged: bool,
    /// Number of outliers detected
    pub outliers_detected: usize,
    /// Number of outliers excluded from statistics
    pub outliers_excluded: usize,
    /// List of preflight checks that passed
    pub preflight_checks_passed: Vec<String>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            cv_at_stop: f64::MAX,
            cv_converged: false,
            outliers_detected: 0,
            outliers_excluded: 0,
            preflight_checks_passed: Vec::new(),
        }
    }
}

// ============================================================================
// Preflight Check Implementations
// ============================================================================

/// Check that a configuration is deterministic
#[derive(Debug)]
pub struct DeterminismCheck {
    config: DeterministicInferenceConfig,
}

impl DeterminismCheck {
    /// Create a new determinism check for the given config
    #[must_use]
    pub fn new(config: DeterministicInferenceConfig) -> Self {
        Self { config }
    }
}

impl PreflightCheck for DeterminismCheck {
    fn name(&self) -> &'static str {
        "determinism_check"
    }

    fn description(&self) -> &'static str {
        "Validates inference configuration ensures deterministic output"
    }

    fn validate(&self) -> PreflightResult<()> {
        self.config.validate_determinism()
    }
}

/// Server availability check - validates server URL and health endpoint response
///
/// Per spec Section 4.2: Verify server is reachable before benchmark
#[derive(Debug)]
pub struct ServerAvailabilityCheck {
    /// Server URL to check
    url: String,
    /// Health endpoint path (e.g., "/health")
    health_path: String,
    /// Cached health check result (status code, None if not yet checked)
    health_status: Option<u16>,
}

impl ServerAvailabilityCheck {
    /// Create a new server availability check
    #[must_use]
    pub fn new(url: String, health_path: String) -> Self {
        Self {
            url,
            health_path,
            health_status: None,
        }
    }

    /// Create with llama.cpp defaults (port 8082, /health)
    #[must_use]
    pub fn llama_cpp(port: u16) -> Self {
        Self::new(format!("http://127.0.0.1:{port}"), "/health".to_string())
    }

    /// Create with Ollama defaults (port 11434, /api/tags)
    #[must_use]
    pub fn ollama(port: u16) -> Self {
        Self::new(format!("http://127.0.0.1:{port}"), "/api/tags".to_string())
    }

    /// Set the health check result (called after HTTP request)
    pub fn set_health_status(&mut self, status: u16) {
        self.health_status = Some(status);
    }

    /// Get the full health URL
    #[must_use]
    pub fn health_url(&self) -> String {
        format!("{}{}", self.url, self.health_path)
    }

    /// Check if URL is well-formed
    fn validate_url(&self) -> PreflightResult<()> {
        if self.url.is_empty() {
            return Err(PreflightError::ConfigError {
                reason: "Server URL cannot be empty".to_string(),
            });
        }
        if !self.url.starts_with("http://") && !self.url.starts_with("https://") {
            return Err(PreflightError::ConfigError {
                reason: format!(
                    "Server URL must start with http:// or https://, got: {}",
                    self.url
                ),
            });
        }
        Ok(())
    }
}

impl PreflightCheck for ServerAvailabilityCheck {
    fn name(&self) -> &'static str {
        "server_availability_check"
    }

    fn description(&self) -> &'static str {
        "Validates server is reachable at the configured URL"
    }

    fn validate(&self) -> PreflightResult<()> {
        // First validate URL format
        self.validate_url()?;

        // Check if health status has been set
        match self.health_status {
            Some(status) if status >= 200 && status < 300 => Ok(()),
            Some(status) => Err(PreflightError::HealthCheckFailed {
                url: self.health_url(),
                status,
            }),
            None => Err(PreflightError::ConfigError {
                reason: "Health check not performed - call set_health_status() first".to_string(),
            }),
        }
    }
}

/// Model availability check - validates requested model exists
///
/// Per spec Section 4.2: Verify model is loaded before benchmark
#[derive(Debug)]
pub struct ModelAvailabilityCheck {
    /// Model name that is requested
    requested_model: String,
    /// List of available models (populated after query)
    available_models: Vec<String>,
}

impl ModelAvailabilityCheck {
    /// Create a new model availability check
    #[must_use]
    pub fn new(requested_model: String) -> Self {
        Self {
            requested_model,
            available_models: Vec::new(),
        }
    }

    /// Set the list of available models (called after querying server)
    pub fn set_available_models(&mut self, models: Vec<String>) {
        self.available_models = models;
    }

    /// Get the requested model name
    #[must_use]
    pub fn requested_model(&self) -> &str {
        &self.requested_model
    }
}

impl PreflightCheck for ModelAvailabilityCheck {
    fn name(&self) -> &'static str {
        "model_availability_check"
    }

    fn description(&self) -> &'static str {
        "Validates requested model is available on the server"
    }

    fn validate(&self) -> PreflightResult<()> {
        if self.requested_model.is_empty() {
            return Err(PreflightError::ConfigError {
                reason: "Model name cannot be empty".to_string(),
            });
        }

        if self.available_models.is_empty() {
            return Err(PreflightError::ConfigError {
                reason: "Available models list not set - call set_available_models() first"
                    .to_string(),
            });
        }

        // Check for exact match or partial match (model:tag format)
        let found = self.available_models.iter().any(|m| {
            m == &self.requested_model
                || m.starts_with(&format!("{}:", self.requested_model))
                || self.requested_model.starts_with(&format!("{m}:"))
        });

        if found {
            Ok(())
        } else {
            Err(PreflightError::ModelNotFound {
                requested: self.requested_model.clone(),
                available: self.available_models.clone(),
            })
        }
    }
}

/// Response schema check - validates JSON response has required fields
///
/// Per spec Section 4.3: Verify response format matches expected schema
#[derive(Debug)]
pub struct ResponseSchemaCheck {
    /// Required field names that must be present
    required_fields: Vec<String>,
    /// Optional field type constraints (field_name -> expected_type)
    field_types: std::collections::HashMap<String, String>,
}

impl ResponseSchemaCheck {
    /// Create a new response schema check with required fields
    #[must_use]
    pub fn new(required_fields: Vec<String>) -> Self {
        Self {
            required_fields,
            field_types: std::collections::HashMap::new(),
        }
    }

    /// Create schema check for llama.cpp /completion response
    #[must_use]
    pub fn llama_cpp_completion() -> Self {
        let mut check = Self::new(vec![
            "content".to_string(),
            "tokens_predicted".to_string(),
            "timings".to_string(),
        ]);
        check
            .field_types
            .insert("tokens_predicted".to_string(), "number".to_string());
        check
            .field_types
            .insert("content".to_string(), "string".to_string());
        check
    }

    /// Create schema check for Ollama /api/generate response
    #[must_use]
    pub fn ollama_generate() -> Self {
        let mut check = Self::new(vec!["response".to_string(), "done".to_string()]);
        check
            .field_types
            .insert("response".to_string(), "string".to_string());
        check
            .field_types
            .insert("done".to_string(), "boolean".to_string());
        check
    }

    /// Add a type constraint for a field
    #[must_use]
    pub fn with_type_constraint(mut self, field: String, expected_type: String) -> Self {
        self.field_types.insert(field, expected_type);
        self
    }

    /// Validate a JSON value against this schema
    ///
    /// # Errors
    /// Returns `PreflightError` if:
    /// - The JSON is not an object
    /// - A required field is missing
    /// - A field has an unexpected type
    pub fn validate_json(&self, json: &serde_json::Value) -> PreflightResult<()> {
        let obj = json
            .as_object()
            .ok_or_else(|| PreflightError::ResponseParseError {
                reason: "Expected JSON object at root".to_string(),
            })?;

        // Check required fields exist
        for field in &self.required_fields {
            if !obj.contains_key(field) {
                return Err(PreflightError::SchemaMismatch {
                    missing_field: field.clone(),
                });
            }
        }

        // Check field types
        for (field, expected_type) in &self.field_types {
            if let Some(value) = obj.get(field) {
                let actual_type = match value {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "boolean",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Object(_) => "object",
                };

                if actual_type != expected_type {
                    return Err(PreflightError::FieldTypeMismatch {
                        field: field.clone(),
                        expected: expected_type.clone(),
                        actual: actual_type.to_string(),
                    });
                }
            }
        }

        Ok(())
    }
}

impl PreflightCheck for ResponseSchemaCheck {
    fn name(&self) -> &'static str {
        "response_schema_check"
    }

    fn description(&self) -> &'static str {
        "Validates response JSON matches expected schema"
    }

    fn validate(&self) -> PreflightResult<()> {
        // Standalone validation - just checks configuration is valid
        if self.required_fields.is_empty() {
            return Err(PreflightError::ConfigError {
                reason: "At least one required field must be specified".to_string(),
            });
        }
        Ok(())
    }
}

/// Preflight validation runner - executes all checks in sequence
///
/// Per Jidoka principle: Stop immediately on first failure
#[derive(Debug, Default)]
pub struct PreflightRunner {
    /// Checks to execute in order
    checks: Vec<Box<dyn PreflightCheck>>,
    /// Names of passed checks (populated during run)
    passed: Vec<String>,
}

impl PreflightRunner {
    /// Create a new preflight runner
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a check to the runner
    pub fn add_check(&mut self, check: Box<dyn PreflightCheck>) {
        self.checks.push(check);
    }

    /// Run all checks, stopping on first failure (Jidoka)
    ///
    /// Returns list of passed check names on success
    ///
    /// # Errors
    /// Returns the `PreflightError` from the first check that fails.
    /// Per Jidoka principle, execution stops immediately on first failure.
    pub fn run(&mut self) -> PreflightResult<Vec<String>> {
        self.passed.clear();

        for check in &self.checks {
            check.validate()?;
            self.passed.push(check.name().to_string());
        }

        Ok(self.passed.clone())
    }

    /// Get passed checks (after run)
    #[must_use]
    pub fn passed_checks(&self) -> &[String] {
        &self.passed
    }
}

// ============================================================================
// Tests (EXTREME TDD: Tests written FIRST)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Canonical Inputs Tests
    // =========================================================================

    #[test]
    fn test_canonical_inputs_version_is_semver() {
        let version = canonical_inputs::VERSION;
        let parts: Vec<&str> = version.split('.').collect();
        assert_eq!(
            parts.len(),
            3,
            "Version should be semver (major.minor.patch)"
        );
        for part in parts {
            assert!(
                part.parse::<u32>().is_ok(),
                "Version part '{}' should be numeric",
                part
            );
        }
    }

    #[test]
    fn test_canonical_inputs_prompt_not_empty() {
        // Verify prompt has reasonable length for benchmarking
        let prompt_len = canonical_inputs::LATENCY_PROMPT.len();
        assert!(
            prompt_len >= 10,
            "Latency prompt should have at least 10 chars, got {}",
            prompt_len
        );
    }

    #[test]
    fn test_canonical_inputs_tokens_not_empty() {
        // Verify we have enough tokens for throughput testing
        let token_count = canonical_inputs::THROUGHPUT_TOKENS.len();
        assert!(
            token_count >= 4,
            "Throughput tokens should have at least 4 tokens, got {}",
            token_count
        );
    }

    #[test]
    fn test_canonical_inputs_max_tokens_reasonable() {
        // Verify max tokens is in a sensible range
        let max_tokens = canonical_inputs::MAX_TOKENS;
        assert!(
            max_tokens > 0,
            "Max tokens should be positive, got {}",
            max_tokens
        );
        assert!(
            max_tokens <= 1000,
            "Max tokens should be <= 1000, got {}",
            max_tokens
        );
    }

    // =========================================================================
    // DeterministicInferenceConfig Tests
    // =========================================================================

    #[test]
    fn test_deterministic_config_default_is_deterministic() {
        let config = DeterministicInferenceConfig::default();
        assert!(
            config.validate_determinism().is_ok(),
            "Default config should be deterministic"
        );
    }

    #[test]
    fn test_deterministic_config_default_values() {
        let config = DeterministicInferenceConfig::default();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.seed, 42);
        assert_eq!(config.top_k, 1);
        assert_eq!(config.top_p, 1.0);
    }

    #[test]
    fn test_deterministic_config_with_seed() {
        let config = DeterministicInferenceConfig::with_seed(12345);
        assert_eq!(config.seed, 12345);
        assert!(config.validate_determinism().is_ok());
    }

    #[test]
    fn test_deterministic_config_rejects_nonzero_temperature() {
        let config = DeterministicInferenceConfig {
            temperature: 0.7,
            ..Default::default()
        };
        let result = config.validate_determinism();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, PreflightError::ConfigError { .. }));
    }

    #[test]
    fn test_deterministic_config_rejects_topk_not_one() {
        let config = DeterministicInferenceConfig {
            top_k: 50,
            ..Default::default()
        };
        let result = config.validate_determinism();
        assert!(result.is_err());
    }

    // =========================================================================
    // CvStoppingCriterion Tests
    // =========================================================================

    #[test]
    fn test_cv_criterion_default_values() {
        let criterion = CvStoppingCriterion::default();
        assert_eq!(criterion.min_samples, 5);
        assert_eq!(criterion.max_samples, 30);
        assert!((criterion.cv_threshold - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_cv_criterion_continues_below_min_samples() {
        let criterion = CvStoppingCriterion::new(5, 30, 0.05);
        let samples = vec![100.0, 100.0, 100.0]; // Only 3 samples
        assert_eq!(criterion.should_stop(&samples), StopDecision::Continue);
    }

    #[test]
    fn test_cv_criterion_stops_at_max_samples() {
        let criterion = CvStoppingCriterion::new(5, 10, 0.01); // Very tight CV
        let samples: Vec<f64> = (1..=10).map(|x| x as f64 * 10.0).collect();
        assert_eq!(
            criterion.should_stop(&samples),
            StopDecision::Stop(StopReason::MaxSamples)
        );
    }

    #[test]
    fn test_cv_criterion_converges_on_identical_values() {
        let criterion = CvStoppingCriterion::new(5, 30, 0.05);
        let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let cv = criterion.calculate_cv(&samples);
        assert!(
            cv < 0.001,
            "CV of identical values should be ~0, got {}",
            cv
        );

        match criterion.should_stop(&samples) {
            StopDecision::Stop(StopReason::CvConverged(cv)) => {
                assert!(cv < 0.05);
            },
            other => panic!("Expected CvConverged, got {:?}", other),
        }
    }

    #[test]
    fn test_cv_criterion_continues_on_high_variance() {
        let criterion = CvStoppingCriterion::new(5, 30, 0.05);
        // High variance: 10, 100, 10, 100, 10 - CV >> 0.05
        let samples = vec![10.0, 100.0, 10.0, 100.0, 10.0];
        assert_eq!(criterion.should_stop(&samples), StopDecision::Continue);
    }

    #[test]
    fn test_cv_calculation_single_value() {
        let criterion = CvStoppingCriterion::default();
        let samples = vec![100.0];
        let cv = criterion.calculate_cv(&samples);
        assert_eq!(cv, f64::MAX);
    }

    #[test]
    fn test_cv_calculation_empty() {
        let criterion = CvStoppingCriterion::default();
        let samples: Vec<f64> = vec![];
        let cv = criterion.calculate_cv(&samples);
        assert_eq!(cv, f64::MAX);
    }

    #[test]
    fn test_cv_calculation_known_values() {
        let criterion = CvStoppingCriterion::default();
        // Mean = 100, values deviate by ~7.9, so CV ~0.079
        let samples = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let cv = criterion.calculate_cv(&samples);
        assert!(cv > 0.07 && cv < 0.09, "Expected CV ~0.079, got {}", cv);
    }

    // =========================================================================
    // OutlierDetector Tests
    // =========================================================================

    #[test]
    fn test_outlier_detector_default_k_factor() {
        let detector = OutlierDetector::default();
        assert!((detector.k_factor - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_outlier_detector_no_outliers_uniform() {
        let detector = OutlierDetector::default();
        let samples = vec![100.0, 101.0, 99.0, 100.5, 99.5];
        let outliers = detector.detect(&samples);
        assert!(
            !outliers.iter().any(|&x| x),
            "Uniform samples should have no outliers"
        );
    }

    #[test]
    fn test_outlier_detector_finds_extreme_outlier() {
        let detector = OutlierDetector::default();
        let samples = vec![100.0, 101.0, 99.0, 100.0, 1000.0]; // 1000 is extreme
        let outliers = detector.detect(&samples);
        assert!(outliers[4], "1000.0 should be detected as outlier");
    }

    #[test]
    fn test_outlier_detector_filter_removes_outliers() {
        let detector = OutlierDetector::default();
        let samples = vec![100.0, 101.0, 99.0, 100.0, 1000.0];
        let filtered = detector.filter(&samples);
        assert!(
            !filtered.contains(&1000.0),
            "Filtered should not contain outlier"
        );
        assert_eq!(filtered.len(), 4);
    }

    #[test]
    fn test_outlier_detector_handles_small_samples() {
        let detector = OutlierDetector::default();
        let samples = vec![100.0, 200.0]; // Only 2 samples
        let outliers = detector.detect(&samples);
        assert_eq!(
            outliers,
            vec![false, false],
            "Should not detect outliers with < 3 samples"
        );
    }

    #[test]
    fn test_outlier_detector_percentile() {
        // Uses nearest-rank method: idx = round((p/100) * (n-1))
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // p50 of 10 elements: idx = round(0.5 * 9) = round(4.5) = 5 → value 6.0
        let p50 = OutlierDetector::percentile(&samples, 50.0);
        assert!(
            (p50 - 5.5).abs() < 1.0,
            "p50 should be ~5.5 (nearest rank gives 6), got {}",
            p50
        );

        // p99: idx = round(0.99 * 9) = round(8.91) = 9 → value 10.0
        let p99 = OutlierDetector::percentile(&samples, 99.0);
        assert!((p99 - 10.0).abs() < 0.5, "p99 should be ~10.0, got {}", p99);
    }

    // =========================================================================
    // QualityMetrics Tests
    // =========================================================================

    #[test]
    fn test_quality_metrics_default() {
        let metrics = QualityMetrics::default();
        assert_eq!(metrics.cv_at_stop, f64::MAX);
        assert!(!metrics.cv_converged);
        assert_eq!(metrics.outliers_detected, 0);
        assert!(metrics.preflight_checks_passed.is_empty());
    }

    #[test]
    fn test_quality_metrics_serialization() {
        let metrics = QualityMetrics {
            cv_at_stop: 0.03,
            cv_converged: true,
            outliers_detected: 2,
            outliers_excluded: 1,
            preflight_checks_passed: vec!["server_check".to_string()],
        };
        let json = serde_json::to_string(&metrics).expect("serialization");
        assert!(json.contains("0.03"));
        assert!(json.contains("server_check"));
    }

    // =========================================================================
    // DeterminismCheck Tests
    // =========================================================================

    #[test]
    fn test_determinism_check_trait_impl() {
        let config = DeterministicInferenceConfig::default();
        let check = DeterminismCheck::new(config);
        assert_eq!(check.name(), "determinism_check");
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_determinism_check_fails_on_bad_config() {
        let config = DeterministicInferenceConfig {
            temperature: 0.5,
            ..Default::default()
        };
        let check = DeterminismCheck::new(config);
        assert!(check.validate().is_err());
    }

    // =========================================================================
    // PreflightError Tests
    // =========================================================================

    #[test]
    fn test_preflight_error_display() {
        let err = PreflightError::ModelNotFound {
            requested: "phi".to_string(),
            available: vec!["phi2:2.7b".to_string(), "llama2".to_string()],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("phi"));
        assert!(msg.contains("phi2:2.7b"));
    }

    #[test]
    fn test_preflight_error_schema_mismatch() {
        let err = PreflightError::SchemaMismatch {
            missing_field: "eval_count".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("eval_count"));
    }

    #[test]
    fn test_preflight_error_type_mismatch() {
        let err = PreflightError::FieldTypeMismatch {
            field: "tokens".to_string(),
            expected: "number".to_string(),
            actual: "string".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("tokens"));
        assert!(msg.contains("number"));
    }

    // =========================================================================
    // ServerAvailabilityCheck Tests
    // =========================================================================

    #[test]
    fn test_server_check_llama_cpp_defaults() {
        let check = ServerAvailabilityCheck::llama_cpp(8082);
        assert_eq!(check.health_url(), "http://127.0.0.1:8082/health");
        assert_eq!(check.name(), "server_availability_check");
    }

    #[test]
    fn test_server_check_ollama_defaults() {
        let check = ServerAvailabilityCheck::ollama(11434);
        assert_eq!(check.health_url(), "http://127.0.0.1:11434/api/tags");
    }

    #[test]
    fn test_server_check_validates_url_format() {
        let check = ServerAvailabilityCheck::new("invalid-url".to_string(), "/health".to_string());
        let result = check.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_server_check_rejects_empty_url() {
        let check = ServerAvailabilityCheck::new(String::new(), "/health".to_string());
        let result = check.validate();
        assert!(matches!(result, Err(PreflightError::ConfigError { .. })));
    }

    #[test]
    fn test_server_check_requires_health_status() {
        let check = ServerAvailabilityCheck::llama_cpp(8082);
        // No health status set yet
        let result = check.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_server_check_accepts_200_status() {
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);
        check.set_health_status(200);
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_server_check_accepts_204_status() {
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);
        check.set_health_status(204); // No Content is valid
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_server_check_rejects_500_status() {
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);
        check.set_health_status(500);
        let result = check.validate();
        assert!(matches!(
            result,
            Err(PreflightError::HealthCheckFailed { status: 500, .. })
        ));
    }

    #[test]
    fn test_server_check_rejects_404_status() {
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);
        check.set_health_status(404);
        let result = check.validate();
        assert!(result.is_err());
    }

    // =========================================================================
    // ModelAvailabilityCheck Tests
    // =========================================================================

    #[test]
    fn test_model_check_finds_exact_match() {
        let mut check = ModelAvailabilityCheck::new("phi2:2.7b".to_string());
        check.set_available_models(vec!["phi2:2.7b".to_string(), "llama2".to_string()]);
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_model_check_finds_partial_match() {
        let mut check = ModelAvailabilityCheck::new("phi2".to_string());
        check.set_available_models(vec!["phi2:2.7b".to_string(), "llama2".to_string()]);
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_model_check_fails_on_missing_model() {
        let mut check = ModelAvailabilityCheck::new("gpt4".to_string());
        check.set_available_models(vec!["phi2:2.7b".to_string(), "llama2".to_string()]);
        let result = check.validate();
        assert!(matches!(result, Err(PreflightError::ModelNotFound { .. })));
    }

    #[test]
    fn test_model_check_rejects_empty_model_name() {
        let check = ModelAvailabilityCheck::new(String::new());
        let result = check.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_model_check_requires_available_models() {
        let check = ModelAvailabilityCheck::new("phi2".to_string());
        // No available models set
        let result = check.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_model_check_name() {
        let check = ModelAvailabilityCheck::new("phi2".to_string());
        assert_eq!(check.name(), "model_availability_check");
        assert_eq!(check.requested_model(), "phi2");
    }

    // =========================================================================
    // ResponseSchemaCheck Tests
    // =========================================================================

    #[test]
    fn test_schema_check_llama_cpp_completion() {
        let check = ResponseSchemaCheck::llama_cpp_completion();
        assert_eq!(check.name(), "response_schema_check");
        assert!(check.validate().is_ok()); // Has required fields
    }

    #[test]
    fn test_schema_check_ollama_generate() {
        let check = ResponseSchemaCheck::ollama_generate();
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_schema_check_validates_required_fields() {
        let check = ResponseSchemaCheck::new(vec!["content".to_string(), "tokens".to_string()]);
        let json: serde_json::Value = serde_json::json!({
            "content": "Hello",
            "tokens": 5
        });
        assert!(check.validate_json(&json).is_ok());
    }

    #[test]
    fn test_schema_check_fails_on_missing_field() {
        let check = ResponseSchemaCheck::new(vec!["content".to_string(), "tokens".to_string()]);
        let json: serde_json::Value = serde_json::json!({
            "content": "Hello"
            // missing "tokens"
        });
        let result = check.validate_json(&json);
        assert!(
            matches!(result, Err(PreflightError::SchemaMismatch { missing_field }) if missing_field == "tokens")
        );
    }

    #[test]
    fn test_schema_check_validates_field_types() {
        let check = ResponseSchemaCheck::new(vec!["count".to_string()])
            .with_type_constraint("count".to_string(), "number".to_string());

        // Correct type
        let json: serde_json::Value = serde_json::json!({ "count": 42 });
        assert!(check.validate_json(&json).is_ok());

        // Wrong type
        let json: serde_json::Value = serde_json::json!({ "count": "42" });
        let result = check.validate_json(&json);
        assert!(matches!(
            result,
            Err(PreflightError::FieldTypeMismatch { .. })
        ));
    }

    #[test]
    fn test_schema_check_rejects_non_object() {
        let check = ResponseSchemaCheck::new(vec!["content".to_string()]);
        let json: serde_json::Value = serde_json::json!("not an object");
        let result = check.validate_json(&json);
        assert!(matches!(
            result,
            Err(PreflightError::ResponseParseError { .. })
        ));
    }

    #[test]
    fn test_schema_check_rejects_empty_required_fields() {
        let check = ResponseSchemaCheck::new(vec![]);
        let result = check.validate();
        assert!(result.is_err());
    }

    // =========================================================================
    // PreflightRunner Tests
    // =========================================================================

    #[test]
    fn test_runner_runs_all_checks() {
        let mut runner = PreflightRunner::new();

        // Add a passing check
        let config = DeterministicInferenceConfig::default();
        runner.add_check(Box::new(DeterminismCheck::new(config)));

        // Add another passing check
        let schema = ResponseSchemaCheck::new(vec!["foo".to_string()]);
        runner.add_check(Box::new(schema));

        let result = runner.run();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_runner_stops_on_first_failure_jidoka() {
        let mut runner = PreflightRunner::new();

        // Add a passing check
        let config = DeterministicInferenceConfig::default();
        runner.add_check(Box::new(DeterminismCheck::new(config)));

        // Add a failing check (empty required fields)
        let schema = ResponseSchemaCheck::new(vec![]);
        runner.add_check(Box::new(schema));

        // Add another check that won't run
        let config2 = DeterministicInferenceConfig::default();
        runner.add_check(Box::new(DeterminismCheck::new(config2)));

        let result = runner.run();
        assert!(result.is_err());
        // Only first check passed before failure
        assert_eq!(runner.passed_checks().len(), 1);
    }

    #[test]
    fn test_runner_empty_passes() {
        let mut runner = PreflightRunner::new();
        let result = runner.run();
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_runner_clears_passed_on_rerun() {
        let mut runner = PreflightRunner::new();

        let config = DeterministicInferenceConfig::default();
        runner.add_check(Box::new(DeterminismCheck::new(config)));

        // First run
        let _ = runner.run();
        assert_eq!(runner.passed_checks().len(), 1);

        // Second run should clear and repopulate
        let _ = runner.run();
        assert_eq!(runner.passed_checks().len(), 1);
    }

    // =========================================================================
    // IMP-143: Real-World Server Verification Tests (EXTREME TDD)
    // =========================================================================
    // These tests verify actual connectivity to external servers.
    // Run with: cargo test test_imp_143 --lib -- --ignored

    /// IMP-143a: Verify llama.cpp server preflight check works with real server
    #[test]
    #[ignore = "Requires running llama.cpp server on port 8082"]
    fn test_imp_143a_llamacpp_real_server_check() {
        // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);

        // Attempt real connection
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("IMP-143a: Should create HTTP client");

        let health_url = check.health_url();
        match client.get(&health_url).send() {
            Ok(response) => {
                let status = response.status().as_u16();
                check.set_health_status(status);

                // IMP-143a: If server is running, check should pass
                let result = check.validate();
                assert!(
                    result.is_ok(),
                    "IMP-143a: llama.cpp server check should pass when server is running. Status: {}, Error: {:?}",
                    status,
                    result.err()
                );
            },
            Err(e) => {
                panic!(
                    "IMP-143a: Could not connect to llama.cpp server at {}. \
                    Start with: llama-server -m model.gguf --host 127.0.0.1 --port 8082. \
                    Error: {}",
                    health_url, e
                );
            },
        }
    }

    /// IMP-143b: Verify Ollama server preflight check works with real server
    #[test]
    #[ignore = "Requires running Ollama server on port 11434"]
    fn test_imp_143b_ollama_real_server_check() {
        // This test requires: ollama serve
        let mut check = ServerAvailabilityCheck::ollama(11434);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("IMP-143b: Should create HTTP client");

        let health_url = check.health_url();
        match client.get(&health_url).send() {
            Ok(response) => {
                let status = response.status().as_u16();
                check.set_health_status(status);

                // IMP-143b: If server is running, check should pass
                let result = check.validate();
                assert!(
                    result.is_ok(),
                    "IMP-143b: Ollama server check should pass when server is running. Status: {}, Error: {:?}",
                    status,
                    result.err()
                );
            },
            Err(e) => {
                panic!(
                    "IMP-143b: Could not connect to Ollama server at {}. \
                    Start with: ollama serve. \
                    Error: {}",
                    health_url, e
                );
            },
        }
    }

    /// IMP-143c: Preflight runner should detect server availability
    #[test]
    fn test_imp_143c_preflight_detects_unavailable_server() {
        // Use a port that's unlikely to have a server running
        let mut check = ServerAvailabilityCheck::llama_cpp(59999);
        check.set_health_status(0); // Connection refused test

        // IMP-143c: Check should fail for unavailable server
        let result = check.validate();
        assert!(
            result.is_err(),
            "IMP-143c: Preflight should detect unavailable server"
        );
    }

    /// IMP-143d: Preflight reports correct error for connection failures
    #[test]
    fn test_imp_143d_preflight_error_reporting() {
        let mut check = ServerAvailabilityCheck::llama_cpp(59998);
        check.set_health_status(503); // Service unavailable

        let result = check.validate();
        match result {
            Err(PreflightError::HealthCheckFailed { status, url, .. }) => {
                assert_eq!(status, 503, "IMP-143d: Should report correct status code");
                assert!(url.contains("59998"), "IMP-143d: Should report correct URL");
            },
            _ => panic!("IMP-143d: Should return HealthCheckFailed error"),
        }
    }
}
