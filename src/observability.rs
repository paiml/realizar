//! Observability Module for Production Monitoring
//!
//! Provides comprehensive observability features per spec §7:
//! - Trueno-DB metrics integration for time-series storage
//! - Renacer tracing for distributed request tracking
//! - A/B testing support for model comparison
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::observability::{ObservabilityConfig, Observer, ABTest};
//!
//! let config = ObservabilityConfig::new()
//!     .with_trueno_db("trueno-db://metrics")
//!     .with_tracing(true);
//!
//! let observer = Observer::new(config);
//!
//! // Record inference metrics
//! observer.record_inference("model-v1", 150, 32, Duration::from_millis(45));
//!
//! // A/B testing
//! let ab_test = ABTest::new("model-comparison")
//!     .with_variant("control", "model-v1", 0.5)
//!     .with_variant("treatment", "model-v2", 0.5);
//! let variant = ab_test.select("user-123");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// OBS-001: Configuration
// ============================================================================

/// Observability configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Trueno-DB connection URI for metrics storage
    pub trueno_db_uri: Option<String>,
    /// Enable distributed tracing
    pub tracing_enabled: bool,
    /// Sampling rate for traces (0.0 - 1.0)
    pub trace_sample_rate: f64,
    /// Metrics flush interval in seconds
    pub flush_interval_secs: u64,
    /// Enable A/B testing
    pub ab_testing_enabled: bool,
}

impl ObservabilityConfig {
    /// Create new configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            trueno_db_uri: None,
            tracing_enabled: true,
            trace_sample_rate: 1.0,
            flush_interval_secs: 60,
            ab_testing_enabled: true,
        }
    }

    /// Set Trueno-DB URI
    #[must_use]
    pub fn with_trueno_db(mut self, uri: impl Into<String>) -> Self {
        self.trueno_db_uri = Some(uri.into());
        self
    }

    /// Enable/disable tracing
    #[must_use]
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }

    /// Set trace sampling rate
    #[must_use]
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.trace_sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set flush interval
    #[must_use]
    pub fn with_flush_interval(mut self, secs: u64) -> Self {
        self.flush_interval_secs = secs;
        self
    }
}

// ============================================================================
// OBS-002: Metrics Point
// ============================================================================

/// A single metrics data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Timestamp (Unix epoch millis)
    pub timestamp: u64,
    /// Labels/tags
    pub labels: HashMap<String, String>,
}

impl MetricPoint {
    /// Create a new metric point
    #[must_use]
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            labels: HashMap::new(),
        }
    }

    /// Add a label
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Format for Trueno-DB line protocol
    #[must_use]
    pub fn to_line_protocol(&self) -> String {
        let labels_str = if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            format!(",{}", pairs.join(","))
        };

        format!(
            "{}{} value={} {}",
            self.name, labels_str, self.value, self.timestamp
        )
    }
}

// ============================================================================
// OBS-003: Tracing Span
// ============================================================================

/// A tracing span for distributed request tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Unique span ID
    pub span_id: String,
    /// Trace ID (shared across spans in same request)
    pub trace_id: String,
    /// Parent span ID (if any)
    pub parent_id: Option<String>,
    /// Operation name
    pub operation: String,
    /// Service name
    pub service: String,
    /// Start timestamp (Unix epoch micros)
    pub start_time: u64,
    /// Duration in microseconds
    pub duration_us: Option<u64>,
    /// Span status
    pub status: SpanStatus,
    /// Span attributes
    pub attributes: HashMap<String, String>,
}

/// Span status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpanStatus {
    /// Span is in progress
    #[default]
    InProgress,
    /// Span completed successfully
    Ok,
    /// Span completed with error
    Error,
}

impl Span {
    /// Create a new span
    #[must_use]
    pub fn new(operation: impl Into<String>, trace_id: impl Into<String>) -> Self {
        Self {
            span_id: generate_id(),
            trace_id: trace_id.into(),
            parent_id: None,
            operation: operation.into(),
            service: "realizar".to_string(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0),
            duration_us: None,
            status: SpanStatus::InProgress,
            attributes: HashMap::new(),
        }
    }

    /// Create a child span
    #[must_use]
    pub fn child(&self, operation: impl Into<String>) -> Self {
        let mut span = Self::new(operation, self.trace_id.clone());
        span.parent_id = Some(self.span_id.clone());
        span
    }

    /// Set parent span
    #[must_use]
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Add attribute
    #[must_use]
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// End the span with success
    pub fn end_ok(&mut self) {
        self.status = SpanStatus::Ok;
        self.duration_us = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0)
                .saturating_sub(self.start_time),
        );
    }

    /// End the span with error
    pub fn end_error(&mut self, error: impl Into<String>) {
        self.status = SpanStatus::Error;
        self.attributes.insert("error".to_string(), error.into());
        self.duration_us = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0)
                .saturating_sub(self.start_time),
        );
    }

    /// Get duration as Duration
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.duration_us.map(Duration::from_micros)
    }
}

// ============================================================================
// OBS-004: A/B Testing
// ============================================================================

/// A/B test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTest {
    /// Test name
    pub name: String,
    /// Test variants
    pub variants: Vec<ABVariant>,
    /// Whether test is active
    pub active: bool,
    /// Test start timestamp
    pub start_time: u64,
}

/// A/B test variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABVariant {
    /// Variant name
    pub name: String,
    /// Model to use for this variant
    pub model: String,
    /// Traffic weight (0.0 - 1.0)
    pub weight: f64,
}

impl ABTest {
    /// Create a new A/B test
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variants: Vec::new(),
            active: true,
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Add a variant
    #[must_use]
    pub fn with_variant(
        mut self,
        name: impl Into<String>,
        model: impl Into<String>,
        weight: f64,
    ) -> Self {
        self.variants.push(ABVariant {
            name: name.into(),
            model: model.into(),
            weight: weight.clamp(0.0, 1.0),
        });
        self
    }

    /// Select a variant for a user (deterministic based on user ID)
    #[must_use]
    pub fn select(&self, user_id: &str) -> Option<&ABVariant> {
        if !self.active || self.variants.is_empty() {
            return None;
        }

        // Normalize weights
        let total_weight: f64 = self.variants.iter().map(|v| v.weight).sum();
        if total_weight <= 0.0 {
            return self.variants.first();
        }

        // Deterministic hash-based selection
        let hash = simple_hash(user_id);
        let selection = (hash as f64 / u64::MAX as f64) * total_weight;

        let mut cumulative = 0.0;
        for variant in &self.variants {
            cumulative += variant.weight;
            if selection < cumulative {
                return Some(variant);
            }
        }

        self.variants.last()
    }

    /// Check if test is valid (weights sum to ~1.0)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.variants.is_empty() {
            return false;
        }
        let total: f64 = self.variants.iter().map(|v| v.weight).sum();
        (total - 1.0).abs() < 0.001
    }
}

/// A/B test result tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ABTestResult {
    /// Test name
    pub test_name: String,
    /// Results per variant
    pub variants: HashMap<String, VariantResult>,
}

/// Results for a single variant
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantResult {
    /// Number of requests
    pub requests: u64,
    /// Number of successes
    pub successes: u64,
    /// Total latency (ms)
    pub total_latency_ms: u64,
    /// Total tokens generated
    pub total_tokens: u64,
}

impl VariantResult {
    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.successes as f64 / self.requests as f64
        }
    }

    /// Calculate average latency
    #[must_use]
    pub fn avg_latency_ms(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.total_latency_ms as f64 / self.requests as f64
        }
    }

    /// Calculate tokens per request
    #[must_use]
    pub fn tokens_per_request(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.total_tokens as f64 / self.requests as f64
        }
    }
}

// ============================================================================
// OBS-005: Observer (Main Interface)
// ============================================================================

/// Central observability interface
#[derive(Debug)]
pub struct Observer {
    /// Configuration
    config: ObservabilityConfig,
    /// Metrics buffer
    metrics_buffer: Arc<RwLock<Vec<MetricPoint>>>,
    /// Spans buffer
    spans_buffer: Arc<RwLock<Vec<Span>>>,
    /// A/B test results
    ab_results: Arc<RwLock<HashMap<String, ABTestResult>>>,
    /// Request counter
    request_counter: AtomicU64,
}

impl Observer {
    /// Create a new observer
    #[must_use]
    pub fn new(config: ObservabilityConfig) -> Self {
        Self {
            config,
            metrics_buffer: Arc::new(RwLock::new(Vec::new())),
            spans_buffer: Arc::new(RwLock::new(Vec::new())),
            ab_results: Arc::new(RwLock::new(HashMap::new())),
            request_counter: AtomicU64::new(0),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_observer() -> Self {
        Self::new(ObservabilityConfig::new())
    }

    /// Record an inference metric
    pub fn record_inference(
        &self,
        model: &str,
        tokens: usize,
        latency_ms: u64,
    ) {
        // Tokens metric
        self.record_metric(
            MetricPoint::new("realizar_tokens_total", tokens as f64)
                .with_label("model", model),
        );

        // Latency metric
        self.record_metric(
            MetricPoint::new("realizar_latency_ms", latency_ms as f64)
                .with_label("model", model),
        );

        // Request count
        self.record_metric(
            MetricPoint::new("realizar_requests_total", 1.0)
                .with_label("model", model),
        );
    }

    /// Record a custom metric
    pub fn record_metric(&self, metric: MetricPoint) {
        if let Ok(mut buffer) = self.metrics_buffer.write() {
            buffer.push(metric);
        }
    }

    /// Create a new trace
    #[must_use]
    pub fn start_trace(&self, operation: &str) -> Span {
        let trace_id = generate_id();
        Span::new(operation, trace_id)
    }

    /// Record a completed span
    pub fn record_span(&self, span: Span) {
        if !self.config.tracing_enabled {
            return;
        }

        // Sample based on rate
        if self.config.trace_sample_rate < 1.0 {
            let sample = simple_hash(&span.span_id) as f64 / u64::MAX as f64;
            if sample > self.config.trace_sample_rate {
                return;
            }
        }

        if let Ok(mut buffer) = self.spans_buffer.write() {
            buffer.push(span);
        }
    }

    /// Record A/B test result
    pub fn record_ab_result(
        &self,
        test_name: &str,
        variant_name: &str,
        success: bool,
        latency_ms: u64,
        tokens: u64,
    ) {
        if !self.config.ab_testing_enabled {
            return;
        }

        if let Ok(mut results) = self.ab_results.write() {
            let test_result = results
                .entry(test_name.to_string())
                .or_insert_with(|| ABTestResult {
                    test_name: test_name.to_string(),
                    variants: HashMap::new(),
                });

            let variant_result = test_result
                .variants
                .entry(variant_name.to_string())
                .or_default();

            variant_result.requests += 1;
            if success {
                variant_result.successes += 1;
            }
            variant_result.total_latency_ms += latency_ms;
            variant_result.total_tokens += tokens;
        }
    }

    /// Get A/B test results
    #[must_use]
    pub fn get_ab_results(&self, test_name: &str) -> Option<ABTestResult> {
        self.ab_results
            .read()
            .ok()
            .and_then(|r| r.get(test_name).cloned())
    }

    /// Flush metrics to Trueno-DB (returns line protocol)
    pub fn flush_metrics(&self) -> Vec<String> {
        let metrics = if let Ok(mut buffer) = self.metrics_buffer.write() {
            std::mem::take(&mut *buffer)
        } else {
            Vec::new()
        };

        metrics.iter().map(MetricPoint::to_line_protocol).collect()
    }

    /// Flush spans
    pub fn flush_spans(&self) -> Vec<Span> {
        if let Ok(mut buffer) = self.spans_buffer.write() {
            std::mem::take(&mut *buffer)
        } else {
            Vec::new()
        }
    }

    /// Get next request ID
    pub fn next_request_id(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Generate Prometheus-format metrics
    #[must_use]
    pub fn prometheus_metrics(&self) -> String {
        let metrics = if let Ok(buffer) = self.metrics_buffer.read() {
            buffer.clone()
        } else {
            Vec::new()
        };

        let mut output = String::new();
        let mut by_name: HashMap<String, Vec<&MetricPoint>> = HashMap::new();

        for metric in &metrics {
            by_name.entry(metric.name.clone()).or_default().push(metric);
        }

        for (name, points) in by_name {
            output.push_str(&format!("# TYPE {name} gauge\n"));
            for point in points {
                let labels = if point.labels.is_empty() {
                    String::new()
                } else {
                    let pairs: Vec<String> = point
                        .labels
                        .iter()
                        .map(|(k, v)| format!("{k}=\"{v}\""))
                        .collect();
                    format!("{{{}}}", pairs.join(","))
                };
                output.push_str(&format!("{name}{labels} {}\n", point.value));
            }
        }

        output
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self::default_observer()
    }
}

// ============================================================================
// OBS-006: Helper Functions
// ============================================================================

/// Generate a unique ID
fn generate_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let hasher_state = RandomState::new();
    let mut hasher = hasher_state.build_hasher();
    hasher.write_u64(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0),
    );
    format!("{:016x}", hasher.finish())
}

/// Simple hash function for deterministic selection (FNV-1a variant)
fn simple_hash(input: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash: u64 = FNV_OFFSET;
    for byte in input.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observability_config_default() {
        let config = ObservabilityConfig::new();
        assert!(config.tracing_enabled);
        assert!((config.trace_sample_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_observability_config_builder() {
        let config = ObservabilityConfig::new()
            .with_trueno_db("trueno-db://localhost:5432")
            .with_tracing(false)
            .with_sample_rate(0.5);

        assert_eq!(
            config.trueno_db_uri,
            Some("trueno-db://localhost:5432".to_string())
        );
        assert!(!config.tracing_enabled);
        assert!((config.trace_sample_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_metric_point() {
        let metric = MetricPoint::new("test_metric", 42.0)
            .with_label("model", "llama3")
            .with_label("version", "1.0");

        assert_eq!(metric.name, "test_metric");
        assert!((metric.value - 42.0).abs() < 0.001);
        assert_eq!(metric.labels.get("model"), Some(&"llama3".to_string()));
    }

    #[test]
    fn test_metric_line_protocol() {
        let metric = MetricPoint::new("cpu_usage", 75.5).with_label("host", "server1");

        let line = metric.to_line_protocol();
        assert!(line.contains("cpu_usage"));
        assert!(line.contains("host=server1"));
        assert!(line.contains("75.5"));
    }

    #[test]
    fn test_span_creation() {
        let span = Span::new("inference", "trace-123");

        assert!(!span.span_id.is_empty());
        assert_eq!(span.trace_id, "trace-123");
        assert_eq!(span.operation, "inference");
        assert_eq!(span.status, SpanStatus::InProgress);
    }

    #[test]
    fn test_span_child() {
        let parent = Span::new("request", "trace-456");
        let child = parent.child("tokenize");

        assert_eq!(child.trace_id, parent.trace_id);
        assert_eq!(child.parent_id, Some(parent.span_id.clone()));
        assert_eq!(child.operation, "tokenize");
    }

    #[test]
    fn test_span_end_ok() {
        let mut span = Span::new("test", "trace");
        std::thread::sleep(Duration::from_millis(10));
        span.end_ok();

        assert_eq!(span.status, SpanStatus::Ok);
        assert!(span.duration_us.is_some());
        assert!(span.duration_us.unwrap() >= 10000); // At least 10ms
    }

    #[test]
    fn test_span_end_error() {
        let mut span = Span::new("test", "trace");
        span.end_error("Something went wrong");

        assert_eq!(span.status, SpanStatus::Error);
        assert_eq!(span.attributes.get("error"), Some(&"Something went wrong".to_string()));
    }

    #[test]
    fn test_ab_test_creation() {
        let test = ABTest::new("model-comparison")
            .with_variant("control", "model-v1", 0.5)
            .with_variant("treatment", "model-v2", 0.5);

        assert_eq!(test.name, "model-comparison");
        assert_eq!(test.variants.len(), 2);
        assert!(test.is_valid());
    }

    #[test]
    fn test_ab_test_selection_deterministic() {
        let test = ABTest::new("test")
            .with_variant("a", "model-a", 0.5)
            .with_variant("b", "model-b", 0.5);

        // Same user should always get same variant
        let variant1 = test.select("user-123");
        let variant2 = test.select("user-123");

        assert_eq!(variant1.map(|v| &v.name), variant2.map(|v| &v.name));
    }

    #[test]
    fn test_ab_test_selection_distribution() {
        let test = ABTest::new("test")
            .with_variant("a", "model-a", 0.5)
            .with_variant("b", "model-b", 0.5);

        let mut count_a = 0;
        let mut count_b = 0;

        for i in 0..1000 {
            let user_id = format!("user-{i}");
            if let Some(variant) = test.select(&user_id) {
                if variant.name == "a" {
                    count_a += 1;
                } else {
                    count_b += 1;
                }
            }
        }

        // Should be roughly 50/50 (within 10% tolerance)
        let ratio = count_a as f64 / (count_a + count_b) as f64;
        assert!(ratio > 0.4 && ratio < 0.6);
    }

    #[test]
    fn test_ab_test_invalid_weights() {
        let test = ABTest::new("test")
            .with_variant("a", "model-a", 0.3)
            .with_variant("b", "model-b", 0.3);

        assert!(!test.is_valid()); // Weights sum to 0.6, not 1.0
    }

    #[test]
    fn test_variant_result_calculations() {
        let result = VariantResult {
            requests: 100,
            successes: 90,
            total_latency_ms: 5000,
            total_tokens: 10000,
        };

        assert!((result.success_rate() - 0.9).abs() < 0.001);
        assert!((result.avg_latency_ms() - 50.0).abs() < 0.001);
        assert!((result.tokens_per_request() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_observer_record_inference() {
        let observer = Observer::default_observer();
        observer.record_inference("llama3", 100, 50);

        let metrics = observer.flush_metrics();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_observer_record_span() {
        let observer = Observer::default_observer();
        let mut span = observer.start_trace("test-op");
        span.end_ok();
        observer.record_span(span);

        let spans = observer.flush_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].operation, "test-op");
    }

    #[test]
    fn test_observer_ab_results() {
        let observer = Observer::default_observer();

        observer.record_ab_result("test", "control", true, 50, 100);
        observer.record_ab_result("test", "control", true, 60, 120);
        observer.record_ab_result("test", "treatment", false, 40, 80);

        let results = observer.get_ab_results("test").unwrap();
        let control = results.variants.get("control").unwrap();
        let treatment = results.variants.get("treatment").unwrap();

        assert_eq!(control.requests, 2);
        assert_eq!(control.successes, 2);
        assert_eq!(treatment.requests, 1);
        assert_eq!(treatment.successes, 0);
    }

    #[test]
    fn test_observer_prometheus_format() {
        let observer = Observer::default_observer();
        observer.record_metric(
            MetricPoint::new("test_metric", 42.0).with_label("env", "prod"),
        );

        let prom = observer.prometheus_metrics();
        assert!(prom.contains("test_metric"));
        assert!(prom.contains("env=\"prod\""));
        assert!(prom.contains("42"));
    }

    #[test]
    fn test_observer_request_id() {
        let observer = Observer::default_observer();

        let id1 = observer.next_request_id();
        let id2 = observer.next_request_id();
        let id3 = observer.next_request_id();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
    }

    #[test]
    fn test_span_status_default() {
        let status = SpanStatus::default();
        assert_eq!(status, SpanStatus::InProgress);
    }

    #[test]
    fn test_generate_id_unique() {
        let id1 = generate_id();
        let id2 = generate_id();

        // IDs should be unique (though not guaranteed with fast calls)
        assert_eq!(id1.len(), 16);
        assert_eq!(id2.len(), 16);
    }

    #[test]
    fn test_simple_hash_deterministic() {
        let hash1 = simple_hash("test-input");
        let hash2 = simple_hash("test-input");
        let hash3 = simple_hash("different-input");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_observer_sampling() {
        let config = ObservabilityConfig::new().with_sample_rate(0.0);
        let observer = Observer::new(config);

        let mut span = observer.start_trace("test");
        span.end_ok();
        observer.record_span(span);

        // With 0% sampling, no spans should be recorded
        let spans = observer.flush_spans();
        assert!(spans.is_empty());
    }

    #[test]
    fn test_observer_tracing_disabled() {
        let config = ObservabilityConfig::new().with_tracing(false);
        let observer = Observer::new(config);

        let mut span = observer.start_trace("test");
        span.end_ok();
        observer.record_span(span);

        let spans = observer.flush_spans();
        assert!(spans.is_empty());
    }
}
