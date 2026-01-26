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

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

// ============================================================================
// W3C Trace Context (per OpenTelemetry specification)
// ============================================================================

/// W3C Trace Context for distributed tracing
/// Implements <https://www.w3.org/TR/trace-context/>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Trace ID (32 hex chars, 16 bytes)
    pub trace_id: String,
    /// Parent Span ID (16 hex chars, 8 bytes)
    pub parent_span_id: Option<String>,
    /// Trace flags (sampled, etc.)
    pub trace_flags: u8,
    /// Trace state (vendor-specific data)
    pub trace_state: Option<String>,
}

impl TraceContext {
    /// Create a new trace context with a fresh trace ID
    #[must_use]
    pub fn new() -> Self {
        Self {
            trace_id: generate_trace_id(),
            parent_span_id: None,
            trace_flags: 0x01, // Sampled
            trace_state: None,
        }
    }

    /// Create child context with new span ID
    #[must_use]
    pub fn child(&self, parent_span_id: &str) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            parent_span_id: Some(parent_span_id.to_string()),
            trace_flags: self.trace_flags,
            trace_state: self.trace_state.clone(),
        }
    }

    /// Parse from W3C traceparent header
    /// Format: {version}-{trace_id}-{parent_id}-{flags}
    /// Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
    #[must_use]
    pub fn from_traceparent(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let version = parts[0];
        if version != "00" {
            return None; // Only version 00 supported
        }

        let trace_id = parts[1];
        let parent_span_id = parts[2];
        let flags = u8::from_str_radix(parts[3], 16).ok()?;

        // Validate lengths
        if trace_id.len() != 32 || parent_span_id.len() != 16 {
            return None;
        }

        Some(Self {
            trace_id: trace_id.to_string(),
            parent_span_id: Some(parent_span_id.to_string()),
            trace_flags: flags,
            trace_state: None,
        })
    }

    /// Generate W3C traceparent header
    #[must_use]
    pub fn to_traceparent(&self, span_id: &str) -> String {
        format!("00-{}-{}-{:02x}", self.trace_id, span_id, self.trace_flags)
    }

    /// Set tracestate header value
    #[must_use]
    pub fn with_tracestate(mut self, state: impl Into<String>) -> Self {
        self.trace_state = Some(state.into());
        self
    }

    /// Check if trace is sampled
    #[must_use]
    pub fn is_sampled(&self) -> bool {
        self.trace_flags & 0x01 != 0
    }

    /// Set sampled flag
    pub fn set_sampled(&mut self, sampled: bool) {
        if sampled {
            self.trace_flags |= 0x01;
        } else {
            self.trace_flags &= !0x01;
        }
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Latency Histogram for Percentile Calculations
// ============================================================================

/// Histogram for tracking latency distributions and calculating percentiles
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// Bucket boundaries in microseconds
    buckets: Vec<u64>,
    /// Count per bucket
    counts: Vec<u64>,
    /// Total count
    total: u64,
    /// Sum of all values (for mean calculation)
    sum: u64,
    /// Min value seen
    min: Option<u64>,
    /// Max value seen
    max: Option<u64>,
}

impl LatencyHistogram {
    /// Create histogram with default buckets (exponential: 1ms to 60s)
    #[must_use]
    pub fn new() -> Self {
        // Buckets: 1ms, 2ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s, 60s
        let buckets = vec![
            1_000,      // 1ms
            2_000,      // 2ms
            5_000,      // 5ms
            10_000,     // 10ms
            25_000,     // 25ms
            50_000,     // 50ms
            100_000,    // 100ms
            250_000,    // 250ms
            500_000,    // 500ms
            1_000_000,  // 1s
            2_500_000,  // 2.5s
            5_000_000,  // 5s
            10_000_000, // 10s
            30_000_000, // 30s
            60_000_000, // 60s
        ];
        let counts = vec![0; buckets.len() + 1]; // +1 for overflow bucket
        Self {
            buckets,
            counts,
            total: 0,
            sum: 0,
            min: None,
            max: None,
        }
    }

    /// Create histogram with custom buckets (in microseconds)
    #[must_use]
    pub fn with_buckets(mut buckets: Vec<u64>) -> Self {
        buckets.sort_unstable();
        let counts = vec![0; buckets.len() + 1];
        Self {
            buckets,
            counts,
            total: 0,
            sum: 0,
            min: None,
            max: None,
        }
    }

    /// Record a latency value in microseconds
    pub fn observe(&mut self, value_us: u64) {
        self.total += 1;
        self.sum += value_us;

        // Update min/max
        self.min = Some(self.min.map_or(value_us, |m| m.min(value_us)));
        self.max = Some(self.max.map_or(value_us, |m| m.max(value_us)));

        // Find bucket
        let bucket_idx = self.buckets.iter().position(|&b| value_us <= b);
        match bucket_idx {
            Some(idx) => self.counts[idx] += 1,
            None => *self.counts.last_mut().unwrap_or(&mut 0) += 1,
        }
    }

    /// Record latency from Duration
    pub fn observe_duration(&mut self, duration: Duration) {
        self.observe(duration.as_micros() as u64);
    }

    /// Get percentile value (0-100)
    #[must_use]
    pub fn percentile(&self, p: f64) -> Option<u64> {
        if self.total == 0 || !(0.0..=100.0).contains(&p) {
            return None;
        }

        let target = ((p / 100.0) * self.total as f64).ceil() as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return if i < self.buckets.len() {
                    Some(self.buckets[i])
                } else {
                    self.max // Overflow bucket
                };
            }
        }

        self.max
    }

    /// Get p50 (median)
    #[must_use]
    pub fn p50(&self) -> Option<u64> {
        self.percentile(50.0)
    }

    /// Get p95
    #[must_use]
    pub fn p95(&self) -> Option<u64> {
        self.percentile(95.0)
    }

    /// Get p99
    #[must_use]
    pub fn p99(&self) -> Option<u64> {
        self.percentile(99.0)
    }

    /// Get mean latency
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.total == 0 {
            None
        } else {
            Some(self.sum as f64 / self.total as f64)
        }
    }

    /// Get total count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.total
    }

    /// Get min value
    #[must_use]
    pub fn min(&self) -> Option<u64> {
        self.min
    }

    /// Get max value
    #[must_use]
    pub fn max_val(&self) -> Option<u64> {
        self.max
    }

    /// Export as Prometheus histogram format
    #[must_use]
    pub fn to_prometheus(&self, name: &str, labels: &str) -> String {
        use std::fmt::Write;
        let mut output = String::new();

        // Bucket counters (cumulative)
        let mut cumulative = 0u64;
        for (i, &boundary) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i];
            let le = boundary as f64 / 1_000_000.0; // Convert to seconds
            let _ = writeln!(
                output,
                "{name}_bucket{{le=\"{le:.6}\",{labels}}} {cumulative}"
            );
        }
        // +Inf bucket
        cumulative += self.counts.last().copied().unwrap_or(0);
        let _ = writeln!(output, "{name}_bucket{{le=\"+Inf\",{labels}}} {cumulative}");

        // Sum and count
        let sum_secs = self.sum as f64 / 1_000_000.0;
        let _ = writeln!(output, "{name}_sum{{{labels}}} {sum_secs:.6}");
        let _ = writeln!(output, "{name}_count{{{labels}}} {}", self.total);

        output
    }
}

// ============================================================================
// OpenTelemetry-Compatible Span Export
// ============================================================================

/// OpenTelemetry-compatible span for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelSpan {
    /// Trace ID (hex string)
    #[serde(rename = "traceId")]
    pub trace_id: String,
    /// Span ID (hex string)
    #[serde(rename = "spanId")]
    pub span_id: String,
    /// Parent Span ID (optional)
    #[serde(rename = "parentSpanId", skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// Operation name
    #[serde(rename = "operationName")]
    pub operation_name: String,
    /// Service name
    #[serde(rename = "serviceName")]
    pub service_name: String,
    /// Start time (Unix epoch microseconds)
    #[serde(rename = "startTimeUnixNano")]
    pub start_time: u64,
    /// End time (Unix epoch microseconds)
    #[serde(rename = "endTimeUnixNano")]
    pub end_time: u64,
    /// Span kind
    #[serde(rename = "kind")]
    pub kind: SpanKind,
    /// Status
    pub status: OtelStatus,
    /// Attributes
    pub attributes: Vec<OtelAttribute>,
}

/// OpenTelemetry span kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SpanKind {
    /// Internal operation
    #[default]
    Internal,
    /// Server-side of RPC
    Server,
    /// Client-side of RPC
    Client,
    /// Message producer
    Producer,
    /// Message consumer
    Consumer,
}

/// OpenTelemetry status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelStatus {
    /// Status code
    pub code: OtelStatusCode,
    /// Optional message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// OpenTelemetry status code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OtelStatusCode {
    /// Unset
    #[default]
    Unset,
    /// OK
    Ok,
    /// Error
    Error,
}

/// OpenTelemetry attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelAttribute {
    /// Attribute key
    pub key: String,
    /// Attribute value
    pub value: OtelValue,
}

/// OpenTelemetry attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OtelValue {
    /// String value
    String {
        /// The string value
        string_value: String,
    },
    /// Integer value
    Int {
        /// The integer value
        int_value: i64,
    },
    /// Float value
    Float {
        /// The double/float value
        double_value: f64,
    },
    /// Boolean value
    Bool {
        /// The boolean value
        bool_value: bool,
    },
}

impl From<&str> for OtelValue {
    fn from(s: &str) -> Self {
        OtelValue::String {
            string_value: s.to_string(),
        }
    }
}

impl From<String> for OtelValue {
    fn from(s: String) -> Self {
        OtelValue::String { string_value: s }
    }
}

impl From<i64> for OtelValue {
    fn from(v: i64) -> Self {
        OtelValue::Int { int_value: v }
    }
}

impl From<f64> for OtelValue {
    fn from(v: f64) -> Self {
        OtelValue::Float { double_value: v }
    }
}

impl From<bool> for OtelValue {
    fn from(v: bool) -> Self {
        OtelValue::Bool { bool_value: v }
    }
}

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

    /// Set span kind
    #[must_use]
    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.attributes
            .insert("span.kind".to_string(), format!("{kind:?}"));
        self
    }

    /// Convert to OpenTelemetry-compatible span format
    #[must_use]
    pub fn to_otel(&self) -> OtelSpan {
        let end_time = self.start_time + self.duration_us.unwrap_or(0);

        let status = match self.status {
            SpanStatus::Ok => OtelStatus {
                code: OtelStatusCode::Ok,
                message: None,
            },
            SpanStatus::Error => OtelStatus {
                code: OtelStatusCode::Error,
                message: self.attributes.get("error").cloned(),
            },
            SpanStatus::InProgress => OtelStatus {
                code: OtelStatusCode::Unset,
                message: None,
            },
        };

        let attributes: Vec<OtelAttribute> = self
            .attributes
            .iter()
            .map(|(k, v)| OtelAttribute {
                key: k.clone(),
                value: OtelValue::from(v.as_str()),
            })
            .collect();

        let kind = self
            .attributes
            .get("span.kind")
            .map_or(SpanKind::Internal, |k| match k.as_str() {
                "Server" => SpanKind::Server,
                "Client" => SpanKind::Client,
                "Producer" => SpanKind::Producer,
                "Consumer" => SpanKind::Consumer,
                _ => SpanKind::Internal,
            });

        OtelSpan {
            trace_id: self.trace_id.clone(),
            span_id: self.span_id.clone(),
            parent_span_id: self.parent_id.clone(),
            operation_name: self.operation.clone(),
            service_name: self.service.clone(),
            start_time: self.start_time * 1000, // Convert to nanoseconds
            end_time: end_time * 1000,
            kind,
            status,
            attributes,
        }
    }

    /// Get trace context for propagation
    #[must_use]
    pub fn trace_context(&self) -> TraceContext {
        TraceContext {
            trace_id: self.trace_id.clone(),
            parent_span_id: Some(self.span_id.clone()),
            trace_flags: 0x01, // Sampled
            trace_state: None,
        }
    }

    /// Generate traceparent header for this span
    #[must_use]
    pub fn traceparent(&self) -> String {
        format!("00-{}-{}-01", self.trace_id, self.span_id)
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
    pub fn record_inference(&self, model: &str, tokens: usize, latency_ms: u64) {
        // Tokens metric
        self.record_metric(
            MetricPoint::new("realizar_tokens_total", tokens as f64).with_label("model", model),
        );

        // Latency metric
        self.record_metric(
            MetricPoint::new("realizar_latency_ms", latency_ms as f64).with_label("model", model),
        );

        // Request count
        self.record_metric(
            MetricPoint::new("realizar_requests_total", 1.0).with_label("model", model),
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
            let test_result =
                results
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
        use std::fmt::Write;
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
            let _ = writeln!(output, "# TYPE {name} gauge");
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
                let _ = writeln!(output, "{name}{labels} {}", point.value);
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

/// Generate a unique ID (16 hex chars / 8 bytes for span ID)
fn generate_id() -> String {
    use std::{
        collections::hash_map::RandomState,
        hash::{BuildHasher, Hasher},
    };

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

/// Generate a trace ID (32 hex chars / 16 bytes per W3C spec)
fn generate_trace_id() -> String {
    use std::{
        collections::hash_map::RandomState,
        hash::{BuildHasher, Hasher},
    };

    let hasher_state = RandomState::new();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let mut hasher1 = hasher_state.build_hasher();
    hasher1.write_u64(now);
    let high = hasher1.finish();

    let mut hasher2 = hasher_state.build_hasher();
    hasher2.write_u64(now.wrapping_add(1));
    let low = hasher2.finish();

    format!("{high:016x}{low:016x}")
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
        assert!(span.duration_us.expect("test") >= 10000); // At least 10ms
    }

    #[test]
    fn test_span_end_error() {
        let mut span = Span::new("test", "trace");
        span.end_error("Something went wrong");

        assert_eq!(span.status, SpanStatus::Error);
        assert_eq!(
            span.attributes.get("error"),
            Some(&"Something went wrong".to_string())
        );
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

        let results = observer.get_ab_results("test").expect("test");
        let control = results.variants.get("control").expect("test");
        let treatment = results.variants.get("treatment").expect("test");

        assert_eq!(control.requests, 2);
        assert_eq!(control.successes, 2);
        assert_eq!(treatment.requests, 1);
        assert_eq!(treatment.successes, 0);
    }

    #[test]
    fn test_observer_prometheus_format() {
        let observer = Observer::default_observer();
        observer.record_metric(MetricPoint::new("test_metric", 42.0).with_label("env", "prod"));

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

    // =========================================================================
    // W3C Trace Context Tests
    // =========================================================================

    #[test]
    fn test_trace_context_new() {
        let ctx = TraceContext::new();
        assert_eq!(ctx.trace_id.len(), 32);
        assert!(ctx.parent_span_id.is_none());
        assert_eq!(ctx.trace_flags, 0x01); // Sampled by default
        assert!(ctx.trace_state.is_none());
    }

    #[test]
    fn test_trace_context_child() {
        let parent_ctx = TraceContext::new();
        let child_ctx = parent_ctx.child("abcdef0123456789");

        assert_eq!(child_ctx.trace_id, parent_ctx.trace_id);
        assert_eq!(
            child_ctx.parent_span_id,
            Some("abcdef0123456789".to_string())
        );
        assert_eq!(child_ctx.trace_flags, parent_ctx.trace_flags);
    }

    #[test]
    fn test_trace_context_from_traceparent_valid() {
        let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::from_traceparent(header).expect("test");

        assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(ctx.parent_span_id, Some("b7ad6b7169203331".to_string()));
        assert_eq!(ctx.trace_flags, 0x01);
    }

    #[test]
    fn test_trace_context_from_traceparent_not_sampled() {
        let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00";
        let ctx = TraceContext::from_traceparent(header).expect("test");

        assert_eq!(ctx.trace_flags, 0x00);
        assert!(!ctx.is_sampled());
    }

    #[test]
    fn test_trace_context_from_traceparent_invalid_version() {
        let header = "01-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        assert!(TraceContext::from_traceparent(header).is_none());
    }

    #[test]
    fn test_trace_context_from_traceparent_invalid_format() {
        assert!(TraceContext::from_traceparent("invalid").is_none());
        assert!(TraceContext::from_traceparent("00-abc-def-01").is_none());
        assert!(TraceContext::from_traceparent("").is_none());
    }

    #[test]
    fn test_trace_context_to_traceparent() {
        let ctx = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            parent_span_id: None,
            trace_flags: 0x01,
            trace_state: None,
        };

        let traceparent = ctx.to_traceparent("b7ad6b7169203331");
        assert_eq!(
            traceparent,
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        );
    }

    #[test]
    fn test_trace_context_with_tracestate() {
        let ctx = TraceContext::new().with_tracestate("vendor=value");
        assert_eq!(ctx.trace_state, Some("vendor=value".to_string()));
    }

    #[test]
    fn test_trace_context_sampled_flag() {
        let mut ctx = TraceContext::new();
        assert!(ctx.is_sampled());

        ctx.set_sampled(false);
        assert!(!ctx.is_sampled());
        assert_eq!(ctx.trace_flags, 0x00);

        ctx.set_sampled(true);
        assert!(ctx.is_sampled());
        assert_eq!(ctx.trace_flags, 0x01);
    }

    #[test]
    fn test_trace_context_default() {
        let ctx = TraceContext::default();
        assert_eq!(ctx.trace_id.len(), 32);
    }

    // =========================================================================
    // Latency Histogram Tests
    // =========================================================================

    #[test]
    fn test_latency_histogram_new() {
        let hist = LatencyHistogram::new();
        assert_eq!(hist.count(), 0);
        assert!(hist.min().is_none());
        assert!(hist.max_val().is_none());
        assert!(hist.mean().is_none());
    }

    #[test]
    fn test_latency_histogram_observe() {
        let mut hist = LatencyHistogram::new();
        hist.observe(1000); // 1ms
        hist.observe(5000); // 5ms
        hist.observe(10000); // 10ms

        assert_eq!(hist.count(), 3);
        assert_eq!(hist.min(), Some(1000));
        assert_eq!(hist.max_val(), Some(10000));
    }

    #[test]
    fn test_latency_histogram_mean() {
        let mut hist = LatencyHistogram::new();
        hist.observe(1000);
        hist.observe(2000);
        hist.observe(3000);

        let mean = hist.mean().expect("test");
        assert!((mean - 2000.0).abs() < 0.001);
    }

    #[test]
    fn test_latency_histogram_observe_duration() {
        let mut hist = LatencyHistogram::new();
        hist.observe_duration(Duration::from_millis(5));

        assert_eq!(hist.count(), 1);
        assert_eq!(hist.min(), Some(5000)); // 5ms = 5000us
    }

    #[test]
    fn test_latency_histogram_percentiles() {
        let mut hist = LatencyHistogram::new();
        // Add 100 observations: 1ms, 2ms, 3ms, ... 100ms
        for i in 1..=100 {
            hist.observe(i * 1000);
        }

        // p50 should be around 50ms
        let p50 = hist.p50().expect("test");
        assert!(p50 >= 25_000 && p50 <= 100_000);

        // p95 should be around 95ms
        let p95 = hist.p95().expect("test");
        assert!(p95 >= 50000);

        // p99 should be around 99ms
        let p99 = hist.p99().expect("test");
        assert!(p99 >= 50000);
    }

    #[test]
    fn test_latency_histogram_percentile_empty() {
        let hist = LatencyHistogram::new();
        assert!(hist.percentile(50.0).is_none());
        assert!(hist.p50().is_none());
        assert!(hist.p95().is_none());
        assert!(hist.p99().is_none());
    }

    #[test]
    fn test_latency_histogram_percentile_invalid() {
        let mut hist = LatencyHistogram::new();
        hist.observe(1000);

        assert!(hist.percentile(-1.0).is_none());
        assert!(hist.percentile(101.0).is_none());
    }

    #[test]
    fn test_latency_histogram_custom_buckets() {
        let buckets = vec![100, 500, 1000, 5000, 10000];
        let mut hist = LatencyHistogram::with_buckets(buckets);

        hist.observe(50); // bucket 0 (<=100)
        hist.observe(200); // bucket 1 (<=500)
        hist.observe(750); // bucket 2 (<=1000)
        hist.observe(20000); // overflow bucket

        assert_eq!(hist.count(), 4);
        assert_eq!(hist.min(), Some(50));
        assert_eq!(hist.max_val(), Some(20000));
    }

    #[test]
    fn test_latency_histogram_to_prometheus() {
        let mut hist = LatencyHistogram::new();
        hist.observe(1000); // 1ms
        hist.observe(50000); // 50ms

        let prom = hist.to_prometheus("request_latency", "service=\"api\"");

        assert!(prom.contains("request_latency_bucket"));
        assert!(prom.contains("le="));
        assert!(prom.contains("service=\"api\""));
        assert!(prom.contains("request_latency_sum"));
        assert!(prom.contains("request_latency_count"));
        assert!(prom.contains("} 2")); // count = 2
    }

    // =========================================================================
    // OpenTelemetry Export Tests
    // =========================================================================

    #[test]
    fn test_span_to_otel_ok() {
        let mut span = Span::new("test-op", "trace123456789012345678901234");
        span.end_ok();

        let otel = span.to_otel();

        assert_eq!(otel.trace_id, "trace123456789012345678901234");
        assert_eq!(otel.operation_name, "test-op");
        assert_eq!(otel.service_name, "realizar");
        assert_eq!(otel.status.code, OtelStatusCode::Ok);
        assert!(otel.status.message.is_none());
    }

    #[test]
    fn test_span_to_otel_error() {
        let mut span = Span::new("failing-op", "trace123456789012345678901234");
        span.end_error("Connection timeout");

        let otel = span.to_otel();

        assert_eq!(otel.status.code, OtelStatusCode::Error);
        assert_eq!(otel.status.message, Some("Connection timeout".to_string()));
    }

    #[test]
    fn test_span_to_otel_with_parent() {
        let parent = Span::new("parent-op", "trace123456789012345678901234");
        let mut child = parent.child("child-op");
        child.end_ok();

        let otel = child.to_otel();

        assert_eq!(otel.parent_span_id, Some(parent.span_id.clone()));
    }

    #[test]
    fn test_span_to_otel_with_attributes() {
        let mut span = Span::new("test-op", "trace123456789012345678901234")
            .with_attribute("model", "llama3")
            .with_attribute("tokens", "256");
        span.end_ok();

        let otel = span.to_otel();

        assert!(otel.attributes.iter().any(|a| a.key == "model"));
        assert!(otel.attributes.iter().any(|a| a.key == "tokens"));
    }

    #[test]
    fn test_span_to_otel_with_kind() {
        let mut span =
            Span::new("server-op", "trace123456789012345678901234").with_kind(SpanKind::Server);
        span.end_ok();

        let otel = span.to_otel();
        assert_eq!(otel.kind, SpanKind::Server);
    }

    #[test]
    fn test_span_to_otel_timestamps() {
        let mut span = Span::new("test-op", "trace123456789012345678901234");
        std::thread::sleep(Duration::from_millis(5));
        span.end_ok();

        let otel = span.to_otel();

        // End time should be >= start time
        assert!(otel.end_time >= otel.start_time);
        // Timestamps should be in nanoseconds
        assert!(otel.start_time > 0);
    }

    #[test]
    fn test_span_trace_context() {
        let span = Span::new("test-op", "0af7651916cd43dd8448eb211c80319c");
        let ctx = span.trace_context();

        assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(ctx.parent_span_id, Some(span.span_id.clone()));
        assert_eq!(ctx.trace_flags, 0x01);
    }

    #[test]
    fn test_span_traceparent() {
        let span = Span::new("test-op", "0af7651916cd43dd8448eb211c80319c");
        let traceparent = span.traceparent();

        assert!(traceparent.starts_with("00-"));
        assert!(traceparent.contains("0af7651916cd43dd8448eb211c80319c"));
        assert!(traceparent.ends_with("-01"));

        // Should have format: 00-{trace_id}-{span_id}-01
        let parts: Vec<&str> = traceparent.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "00");
        assert_eq!(parts[1], "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(parts[3], "01");
    }

    // =========================================================================
    // OtelValue Tests
    // =========================================================================

    #[test]
    fn test_otel_value_from_str() {
        let val: OtelValue = "test".into();
        match val {
            OtelValue::String { string_value } => assert_eq!(string_value, "test"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_otel_value_from_string() {
        let val: OtelValue = String::from("test").into();
        match val {
            OtelValue::String { string_value } => assert_eq!(string_value, "test"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_otel_value_from_i64() {
        let val: OtelValue = 42i64.into();
        match val {
            OtelValue::Int { int_value } => assert_eq!(int_value, 42),
            _ => panic!("Expected Int variant"),
        }
    }

    #[test]
    fn test_otel_value_from_f64() {
        let val: OtelValue = 3.14f64.into();
        match val {
            OtelValue::Float { double_value } => assert!((double_value - 3.14).abs() < 0.001),
            _ => panic!("Expected Float variant"),
        }
    }

    #[test]
    fn test_otel_value_from_bool() {
        let val: OtelValue = true.into();
        match val {
            OtelValue::Bool { bool_value } => assert!(bool_value),
            _ => panic!("Expected Bool variant"),
        }
    }

    // =========================================================================
    // SpanKind Tests
    // =========================================================================

    #[test]
    fn test_span_kind_default() {
        let kind = SpanKind::default();
        assert_eq!(kind, SpanKind::Internal);
    }

    #[test]
    fn test_otel_status_code_default() {
        let code = OtelStatusCode::default();
        assert_eq!(code, OtelStatusCode::Unset);
    }

    // =========================================================================
    // Generate Trace ID Tests
    // =========================================================================

    #[test]
    fn test_generate_trace_id_length() {
        let id = generate_trace_id();
        assert_eq!(id.len(), 32); // 16 bytes = 32 hex chars
    }

    #[test]
    fn test_generate_trace_id_hex() {
        let id = generate_trace_id();
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // =========================================================================
    // Additional Coverage Tests - Edge Cases
    // =========================================================================

    #[test]
    fn test_metric_line_protocol_no_labels() {
        let metric = MetricPoint::new("simple_metric", 100.0);
        let line = metric.to_line_protocol();

        // Should not contain commas (no labels)
        assert!(line.starts_with("simple_metric value=100"));
        // Verify no extra comma before "value"
        assert!(!line.contains(",value"));
    }

    #[test]
    fn test_span_with_parent() {
        let span = Span::new("child-op", "trace-abc").with_parent("parent-span-123");

        assert_eq!(span.parent_id, Some("parent-span-123".to_string()));
    }

    #[test]
    fn test_span_duration() {
        let mut span = Span::new("test", "trace");

        // Before ending, duration should be None
        assert!(span.duration().is_none());

        std::thread::sleep(Duration::from_millis(5));
        span.end_ok();

        // After ending, duration should be Some
        let dur = span.duration().expect("should have duration");
        assert!(dur >= Duration::from_millis(5));
    }

    #[test]
    fn test_span_to_otel_in_progress() {
        let span = Span::new("in-progress-op", "trace123456789012345678901234");
        // Don't end the span - leave it in progress

        let otel = span.to_otel();

        assert_eq!(otel.status.code, OtelStatusCode::Unset);
        assert!(otel.status.message.is_none());
    }

    #[test]
    fn test_span_to_otel_client_kind() {
        let mut span =
            Span::new("client-op", "trace123456789012345678901234").with_kind(SpanKind::Client);
        span.end_ok();

        let otel = span.to_otel();
        assert_eq!(otel.kind, SpanKind::Client);
    }

    #[test]
    fn test_span_to_otel_producer_kind() {
        let mut span =
            Span::new("producer-op", "trace123456789012345678901234").with_kind(SpanKind::Producer);
        span.end_ok();

        let otel = span.to_otel();
        assert_eq!(otel.kind, SpanKind::Producer);
    }

    #[test]
    fn test_span_to_otel_consumer_kind() {
        let mut span =
            Span::new("consumer-op", "trace123456789012345678901234").with_kind(SpanKind::Consumer);
        span.end_ok();

        let otel = span.to_otel();
        assert_eq!(otel.kind, SpanKind::Consumer);
    }

    #[test]
    fn test_span_to_otel_unknown_kind() {
        // Test with an unknown kind string in attributes
        let mut span = Span::new("test-op", "trace123456789012345678901234");
        span.attributes
            .insert("span.kind".to_string(), "Unknown".to_string());
        span.end_ok();

        let otel = span.to_otel();
        assert_eq!(otel.kind, SpanKind::Internal); // Falls back to Internal
    }

    #[test]
    fn test_variant_result_zero_requests() {
        let result = VariantResult::default();

        assert_eq!(result.requests, 0);
        assert!((result.success_rate() - 0.0).abs() < 0.001);
        assert!((result.avg_latency_ms() - 0.0).abs() < 0.001);
        assert!((result.tokens_per_request() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_ab_test_select_inactive() {
        let mut test = ABTest::new("inactive-test")
            .with_variant("a", "model-a", 0.5)
            .with_variant("b", "model-b", 0.5);
        test.active = false;

        assert!(test.select("user-123").is_none());
    }

    #[test]
    fn test_ab_test_select_empty_variants() {
        let test = ABTest::new("empty-test");

        assert!(test.select("user-123").is_none());
    }

    #[test]
    fn test_ab_test_select_zero_weight() {
        let test = ABTest::new("zero-weight-test")
            .with_variant("a", "model-a", 0.0)
            .with_variant("b", "model-b", 0.0);

        // With zero total weight, should return first variant
        let variant = test.select("user-123");
        assert!(variant.is_some());
        assert_eq!(variant.unwrap().name, "a");
    }

    #[test]
    fn test_ab_test_is_valid_empty() {
        let test = ABTest::new("empty-test");
        assert!(!test.is_valid());
    }

    #[test]
    fn test_ab_test_weight_clamping() {
        let test = ABTest::new("clamped-test")
            .with_variant("a", "model-a", 1.5) // Should clamp to 1.0
            .with_variant("b", "model-b", -0.5); // Should clamp to 0.0

        assert!((test.variants[0].weight - 1.0).abs() < 0.001);
        assert!((test.variants[1].weight - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_observer_ab_testing_disabled() {
        let config = ObservabilityConfig {
            ab_testing_enabled: false,
            ..ObservabilityConfig::new()
        };
        let observer = Observer::new(config);

        observer.record_ab_result("test", "control", true, 50, 100);

        // Results should not be recorded
        assert!(observer.get_ab_results("test").is_none());
    }

    #[test]
    fn test_observability_config_flush_interval() {
        let config = ObservabilityConfig::new().with_flush_interval(120);

        assert_eq!(config.flush_interval_secs, 120);
    }

    #[test]
    fn test_observability_config_sample_rate_clamping() {
        let config_high = ObservabilityConfig::new().with_sample_rate(2.0);
        assert!((config_high.trace_sample_rate - 1.0).abs() < 0.001);

        let config_low = ObservabilityConfig::new().with_sample_rate(-0.5);
        assert!((config_low.trace_sample_rate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_latency_histogram_overflow_bucket() {
        let mut hist = LatencyHistogram::new();
        // Add a value larger than all buckets (60s = 60_000_000 us)
        hist.observe(100_000_000); // 100s - definitely in overflow

        assert_eq!(hist.count(), 1);
        assert_eq!(hist.max_val(), Some(100_000_000));

        // p100 should return max value (overflow bucket)
        let p100 = hist.percentile(100.0);
        assert_eq!(p100, Some(100_000_000));
    }

    #[test]
    fn test_latency_histogram_default() {
        let hist = LatencyHistogram::default();
        assert_eq!(hist.count(), 0);
    }

    #[test]
    fn test_latency_histogram_with_buckets_unsorted() {
        // Buckets should be sorted internally
        let buckets = vec![1000, 100, 500, 200];
        let mut hist = LatencyHistogram::with_buckets(buckets);

        hist.observe(150);
        hist.observe(350);

        assert_eq!(hist.count(), 2);
    }

    #[test]
    fn test_observer_prometheus_multiple_metrics_same_name() {
        let observer = Observer::default_observer();

        observer.record_metric(
            MetricPoint::new("request_count", 10.0).with_label("endpoint", "/api/v1"),
        );
        observer.record_metric(
            MetricPoint::new("request_count", 20.0).with_label("endpoint", "/api/v2"),
        );

        let prom = observer.prometheus_metrics();

        // Should have TYPE header
        assert!(prom.contains("# TYPE request_count gauge"));
        // Should have both metrics
        assert!(prom.contains("endpoint=\"/api/v1\""));
        assert!(prom.contains("endpoint=\"/api/v2\""));
    }

    #[test]
    fn test_observer_prometheus_no_labels() {
        let observer = Observer::default_observer();
        observer.record_metric(MetricPoint::new("simple_count", 5.0));

        let prom = observer.prometheus_metrics();

        assert!(prom.contains("simple_count 5"));
    }

    #[test]
    fn test_trace_context_from_traceparent_invalid_hex_flags() {
        // Invalid hex in flags field
        let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-GG";
        assert!(TraceContext::from_traceparent(header).is_none());
    }

    #[test]
    fn test_trace_context_from_traceparent_wrong_trace_id_length() {
        // trace_id too short (only 16 chars instead of 32)
        let header = "00-0af7651916cd43dd-b7ad6b7169203331-01";
        assert!(TraceContext::from_traceparent(header).is_none());
    }

    #[test]
    fn test_trace_context_from_traceparent_wrong_span_id_length() {
        // span_id too short (only 8 chars instead of 16)
        let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b71-01";
        assert!(TraceContext::from_traceparent(header).is_none());
    }

    #[test]
    fn test_otel_span_serialization() {
        let mut span =
            Span::new("test-op", "0af7651916cd43dd8448eb211c80319c").with_attribute("key", "value");
        span.end_ok();

        let otel = span.to_otel();

        // Should be serializable to JSON
        let json = serde_json::to_string(&otel).expect("should serialize");
        assert!(json.contains("traceId"));
        assert!(json.contains("spanId"));
        assert!(json.contains("operationName"));
    }

    #[test]
    fn test_ab_test_result_default() {
        let result = ABTestResult::default();
        assert!(result.test_name.is_empty());
        assert!(result.variants.is_empty());
    }

    #[test]
    fn test_metric_point_with_multiple_labels() {
        let metric = MetricPoint::new("multi_label", 42.0)
            .with_label("region", "us-east")
            .with_label("env", "prod")
            .with_label("version", "1.0");

        let line = metric.to_line_protocol();

        assert!(line.contains("region=us-east"));
        assert!(line.contains("env=prod"));
        assert!(line.contains("version=1.0"));
    }

    #[test]
    fn test_observer_flush_metrics_clears_buffer() {
        let observer = Observer::default_observer();

        observer.record_metric(MetricPoint::new("test", 1.0));
        let first_flush = observer.flush_metrics();
        assert_eq!(first_flush.len(), 1);

        // Second flush should return empty
        let second_flush = observer.flush_metrics();
        assert!(second_flush.is_empty());
    }

    #[test]
    fn test_observer_flush_spans_clears_buffer() {
        let observer = Observer::default_observer();

        let mut span = observer.start_trace("test-op");
        span.end_ok();
        observer.record_span(span);

        let first_flush = observer.flush_spans();
        assert_eq!(first_flush.len(), 1);

        // Second flush should return empty
        let second_flush = observer.flush_spans();
        assert!(second_flush.is_empty());
    }

    #[test]
    fn test_ab_test_select_last_variant() {
        // Test that selection can hit the last variant
        let test = ABTest::new("test")
            .with_variant("a", "model-a", 0.1)
            .with_variant("b", "model-b", 0.9);

        // With many users, we should hit both variants
        let mut hit_b = false;
        for i in 0..100 {
            let user_id = format!("user-{i}");
            if let Some(variant) = test.select(&user_id) {
                if variant.name == "b" {
                    hit_b = true;
                    break;
                }
            }
        }
        assert!(hit_b, "Should hit variant b with 90% weight");
    }

    #[test]
    fn test_latency_histogram_single_value_percentiles() {
        let mut hist = LatencyHistogram::new();
        hist.observe(5000); // 5ms

        // All percentiles should return the same bucket
        assert!(hist.p50().is_some());
        assert!(hist.p95().is_some());
        assert!(hist.p99().is_some());
    }

    #[test]
    fn test_span_with_attribute_chain() {
        let span = Span::new("test", "trace")
            .with_attribute("a", "1")
            .with_attribute("b", "2")
            .with_attribute("c", "3");

        assert_eq!(span.attributes.len(), 3);
        assert_eq!(span.attributes.get("a"), Some(&"1".to_string()));
        assert_eq!(span.attributes.get("b"), Some(&"2".to_string()));
        assert_eq!(span.attributes.get("c"), Some(&"3".to_string()));
    }

    #[test]
    fn test_trace_context_set_sampled_preserves_other_flags() {
        let mut ctx = TraceContext {
            trace_id: "test".to_string(),
            parent_span_id: None,
            trace_flags: 0xFF, // All flags set
            trace_state: None,
        };

        // Setting sampled to false should only clear bit 0
        ctx.set_sampled(false);
        assert_eq!(ctx.trace_flags, 0xFE);

        // Setting sampled to true should only set bit 0
        ctx.set_sampled(true);
        assert_eq!(ctx.trace_flags, 0xFF);
    }

    #[test]
    fn test_simple_hash_empty_string() {
        let hash = simple_hash("");
        // Should not panic and should return a consistent value
        let hash2 = simple_hash("");
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_generate_id_format() {
        let id = generate_id();
        // Should be 16 hex characters
        assert_eq!(id.len(), 16);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // =========================================================================
    // Additional Edge Case Tests for Coverage
    // =========================================================================

    #[test]
    fn test_latency_histogram_percentile_boundary() {
        // Test percentile at exact boundary values
        let mut hist = LatencyHistogram::new();
        hist.observe(1000);

        // p0 should work
        let p0 = hist.percentile(0.0);
        assert!(p0.is_some());

        // p100 should work
        let p100 = hist.percentile(100.0);
        assert!(p100.is_some());
    }

    #[test]
    fn test_latency_histogram_to_prometheus_empty_labels() {
        let mut hist = LatencyHistogram::new();
        hist.observe(5000);

        // Empty labels string
        let prom = hist.to_prometheus("latency", "");
        assert!(prom.contains("latency_bucket"));
        assert!(prom.contains("le=\"+Inf\""));
    }

    #[test]
    fn test_latency_histogram_min_max_update() {
        let mut hist = LatencyHistogram::new();

        // First observation sets both min and max
        hist.observe(5000);
        assert_eq!(hist.min(), Some(5000));
        assert_eq!(hist.max_val(), Some(5000));

        // Lower value updates min
        hist.observe(1000);
        assert_eq!(hist.min(), Some(1000));
        assert_eq!(hist.max_val(), Some(5000));

        // Higher value updates max
        hist.observe(10000);
        assert_eq!(hist.min(), Some(1000));
        assert_eq!(hist.max_val(), Some(10000));
    }

    #[test]
    fn test_span_to_otel_internal_kind_default() {
        // Span without explicit kind should be Internal
        let mut span = Span::new("test-op", "trace123456789012345678901234");
        span.end_ok();

        let otel = span.to_otel();
        assert_eq!(otel.kind, SpanKind::Internal);
    }

    #[test]
    fn test_otel_attribute_construction() {
        let attr = OtelAttribute {
            key: "test_key".to_string(),
            value: OtelValue::from("test_value"),
        };

        assert_eq!(attr.key, "test_key");
        match attr.value {
            OtelValue::String { string_value } => assert_eq!(string_value, "test_value"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_otel_status_construction() {
        let status_ok = OtelStatus {
            code: OtelStatusCode::Ok,
            message: None,
        };
        assert_eq!(status_ok.code, OtelStatusCode::Ok);

        let status_err = OtelStatus {
            code: OtelStatusCode::Error,
            message: Some("failed".to_string()),
        };
        assert_eq!(status_err.message, Some("failed".to_string()));
    }

    #[test]
    fn test_ab_variant_construction() {
        let variant = ABVariant {
            name: "control".to_string(),
            model: "model-v1".to_string(),
            weight: 0.5,
        };

        assert_eq!(variant.name, "control");
        assert_eq!(variant.model, "model-v1");
        assert!((variant.weight - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_observer_default_trait() {
        let observer: Observer = Default::default();
        let id = observer.next_request_id();
        assert_eq!(id, 0);
    }

    #[test]
    fn test_observer_record_metric_multiple() {
        let observer = Observer::default_observer();

        // Record multiple metrics of different types
        observer.record_metric(MetricPoint::new("counter", 1.0));
        observer.record_metric(MetricPoint::new("gauge", 50.0));
        observer.record_metric(MetricPoint::new("histogram", 100.0));

        let metrics = observer.flush_metrics();
        assert_eq!(metrics.len(), 3);
    }

    #[test]
    fn test_ab_test_result_construction() {
        let mut result = ABTestResult {
            test_name: "my-test".to_string(),
            variants: HashMap::new(),
        };

        result.variants.insert(
            "control".to_string(),
            VariantResult {
                requests: 10,
                successes: 8,
                total_latency_ms: 500,
                total_tokens: 1000,
            },
        );

        assert_eq!(result.test_name, "my-test");
        assert_eq!(result.variants.len(), 1);
        assert!(result.variants.contains_key("control"));
    }

    #[test]
    fn test_serde_trace_context_roundtrip() {
        let ctx = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            parent_span_id: Some("b7ad6b7169203331".to_string()),
            trace_flags: 0x01,
            trace_state: Some("vendor=value".to_string()),
        };

        let json = serde_json::to_string(&ctx).expect("serialize");
        let ctx2: TraceContext = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(ctx.trace_id, ctx2.trace_id);
        assert_eq!(ctx.parent_span_id, ctx2.parent_span_id);
        assert_eq!(ctx.trace_flags, ctx2.trace_flags);
        assert_eq!(ctx.trace_state, ctx2.trace_state);
    }

    #[test]
    fn test_serde_latency_histogram_roundtrip() {
        let mut hist = LatencyHistogram::new();
        hist.observe(1000);
        hist.observe(5000);

        let json = serde_json::to_string(&hist).expect("serialize");
        let hist2: LatencyHistogram = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(hist.count(), hist2.count());
        assert_eq!(hist.min(), hist2.min());
        assert_eq!(hist.max_val(), hist2.max_val());
    }

    #[test]
    fn test_serde_metric_point_roundtrip() {
        let metric = MetricPoint::new("test_metric", 42.5).with_label("env", "prod");

        let json = serde_json::to_string(&metric).expect("serialize");
        let metric2: MetricPoint = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(metric.name, metric2.name);
        assert!((metric.value - metric2.value).abs() < 0.001);
        assert_eq!(metric.labels.get("env"), metric2.labels.get("env"));
    }

    #[test]
    fn test_serde_span_roundtrip() {
        let mut span = Span::new("test-op", "trace-123").with_attribute("key", "value");
        span.end_ok();

        let json = serde_json::to_string(&span).expect("serialize");
        let span2: Span = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(span.trace_id, span2.trace_id);
        assert_eq!(span.operation, span2.operation);
        assert_eq!(span.status, span2.status);
    }

    #[test]
    fn test_serde_ab_test_roundtrip() {
        let test = ABTest::new("test")
            .with_variant("a", "model-a", 0.5)
            .with_variant("b", "model-b", 0.5);

        let json = serde_json::to_string(&test).expect("serialize");
        let test2: ABTest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(test.name, test2.name);
        assert_eq!(test.variants.len(), test2.variants.len());
    }

    #[test]
    fn test_serde_span_kind_roundtrip() {
        let kinds = vec![
            SpanKind::Internal,
            SpanKind::Server,
            SpanKind::Client,
            SpanKind::Producer,
            SpanKind::Consumer,
        ];

        for kind in kinds {
            let json = serde_json::to_string(&kind).expect("serialize");
            let kind2: SpanKind = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(kind, kind2);
        }
    }

    #[test]
    fn test_serde_otel_status_code_roundtrip() {
        let codes = vec![
            OtelStatusCode::Unset,
            OtelStatusCode::Ok,
            OtelStatusCode::Error,
        ];

        for code in codes {
            let json = serde_json::to_string(&code).expect("serialize");
            let code2: OtelStatusCode = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(code, code2);
        }
    }

    #[test]
    fn test_serde_span_status_roundtrip() {
        let statuses = vec![SpanStatus::InProgress, SpanStatus::Ok, SpanStatus::Error];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let status2: SpanStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(status, status2);
        }
    }

    #[test]
    fn test_serde_otel_value_variants() {
        // String variant
        let s: OtelValue = "test".into();
        let json = serde_json::to_string(&s).expect("serialize");
        assert!(json.contains("string_value"));

        // Int variant
        let i: OtelValue = 42i64.into();
        let json = serde_json::to_string(&i).expect("serialize");
        assert!(json.contains("int_value"));

        // Float variant
        let f: OtelValue = 3.14f64.into();
        let json = serde_json::to_string(&f).expect("serialize");
        assert!(json.contains("double_value"));

        // Bool variant
        let b: OtelValue = true.into();
        let json = serde_json::to_string(&b).expect("serialize");
        assert!(json.contains("bool_value"));
    }

    #[test]
    fn test_observability_config_default_values() {
        let config = ObservabilityConfig::default();

        assert!(config.trueno_db_uri.is_none());
        assert!(!config.tracing_enabled); // Default is false
        assert!((config.trace_sample_rate - 0.0).abs() < 0.001);
        assert_eq!(config.flush_interval_secs, 0);
        assert!(!config.ab_testing_enabled);
    }

    #[test]
    fn test_concurrent_request_id_generation() {
        use std::sync::Arc;
        use std::thread;

        let observer = Arc::new(Observer::default_observer());
        let mut handles = vec![];

        // Spawn multiple threads to get request IDs
        for _ in 0..10 {
            let obs = Arc::clone(&observer);
            handles.push(thread::spawn(move || obs.next_request_id()));
        }

        let mut ids: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        ids.sort();

        // All IDs should be unique (0-9)
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(*id, i as u64);
        }
    }

    #[test]
    fn test_concurrent_metric_recording() {
        use std::sync::Arc;
        use std::thread;

        let observer = Arc::new(Observer::default_observer());
        let mut handles = vec![];

        for i in 0..5 {
            let obs = Arc::clone(&observer);
            handles.push(thread::spawn(move || {
                obs.record_metric(MetricPoint::new(format!("metric_{i}"), i as f64));
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let metrics = observer.flush_metrics();
        assert_eq!(metrics.len(), 5);
    }

    #[test]
    fn test_concurrent_span_recording() {
        use std::sync::Arc;
        use std::thread;

        let observer = Arc::new(Observer::default_observer());
        let mut handles = vec![];

        for i in 0..5 {
            let obs = Arc::clone(&observer);
            handles.push(thread::spawn(move || {
                let mut span = obs.start_trace(&format!("op-{i}"));
                span.end_ok();
                obs.record_span(span);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let spans = observer.flush_spans();
        assert_eq!(spans.len(), 5);
    }

    #[test]
    fn test_observability_config_serde_roundtrip() {
        let config = ObservabilityConfig::new()
            .with_trueno_db("trueno://localhost")
            .with_tracing(true)
            .with_sample_rate(0.75)
            .with_flush_interval(30);

        let json = serde_json::to_string(&config).expect("serialize");
        let config2: ObservabilityConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(config.trueno_db_uri, config2.trueno_db_uri);
        assert_eq!(config.tracing_enabled, config2.tracing_enabled);
        assert!((config.trace_sample_rate - config2.trace_sample_rate).abs() < 0.001);
        assert_eq!(config.flush_interval_secs, config2.flush_interval_secs);
    }

    #[test]
    fn test_metric_point_timestamp_is_set() {
        let metric = MetricPoint::new("test", 1.0);
        // Timestamp should be non-zero (set to current time)
        assert!(metric.timestamp > 0);
    }

    #[test]
    fn test_span_start_time_is_set() {
        let span = Span::new("test", "trace-id");
        // Start time should be non-zero
        assert!(span.start_time > 0);
    }

    #[test]
    fn test_ab_test_start_time_is_set() {
        let test = ABTest::new("test");
        // Start time should be non-zero
        assert!(test.start_time > 0);
    }

    #[test]
    fn test_latency_histogram_with_empty_buckets() {
        let hist = LatencyHistogram::with_buckets(vec![]);

        // Empty buckets - only overflow bucket exists
        assert_eq!(hist.buckets.len(), 0);
        // counts should have 1 element (overflow bucket)
        assert_eq!(hist.counts.len(), 1);
    }

    #[test]
    fn test_latency_histogram_observe_into_first_bucket() {
        let mut hist = LatencyHistogram::with_buckets(vec![1000, 5000, 10000]);

        // Value exactly at first bucket boundary
        hist.observe(1000);
        assert_eq!(hist.count(), 1);

        // p50 should return the first bucket
        let p50 = hist.p50();
        assert_eq!(p50, Some(1000));
    }

    #[test]
    fn test_span_service_name_default() {
        let span = Span::new("op", "trace");
        assert_eq!(span.service, "realizar");
    }

    #[test]
    fn test_otel_span_timestamps_in_nanoseconds() {
        let mut span = Span::new("test", "trace123456789012345678901234");
        span.end_ok();

        let otel = span.to_otel();

        // OTel timestamps should be in nanoseconds (start_time * 1000 from microseconds)
        // They should be much larger than the original microsecond values
        assert!(otel.start_time > span.start_time);
    }

    #[test]
    fn test_observer_get_ab_results_missing() {
        let observer = Observer::default_observer();
        assert!(observer.get_ab_results("nonexistent").is_none());
    }

    #[test]
    fn test_ab_test_select_full_weight_first_variant() {
        let test = ABTest::new("test").with_variant("only", "model-only", 1.0);

        // Should always select the only variant
        for i in 0..10 {
            let variant = test.select(&format!("user-{i}"));
            assert!(variant.is_some());
            assert_eq!(variant.unwrap().name, "only");
        }
    }
}
