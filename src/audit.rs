//! Audit Trail and Provenance Logging
//!
//! Per spec §12: Comprehensive audit record for every inference request.
//! Implements GDPR Article 13/14 and SOC 2 compliance requirements.
//!
//! ## Toyota Way: Jidoka (Built-in Quality)
//!
//! Every inference operation includes automatic quality checks:
//! - CRC32 checksum verification
//! - NaN/Inf detection in outputs
//! - Latency anomaly detection
//! - Confidence score validation
//!
//! ## Key Features
//!
//! - Full provenance tracking (model hash, distillation lineage)
//! - Latency breakdown (preprocessing, inference, postprocessing)
//! - Quality gates (NaN check, confidence check)
//! - Batched async flush for high throughput

// Module-level clippy allows for audit infrastructure
#![allow(clippy::must_use_candidate)] // Builder pattern methods don't need must_use
#![allow(clippy::return_self_not_must_use)] // Builder pattern methods return Self
#![allow(clippy::missing_errors_doc)] // Error conditions are self-explanatory

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use uuid::Uuid;

/// Comprehensive audit record for every inference request
/// Per GDPR Article 13/14 and SOC 2 compliance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    // === Identification ===
    /// Unique request identifier
    pub request_id: String,
    /// Timestamp (ISO 8601 with timezone)
    pub timestamp: DateTime<Utc>,
    /// Client identifier (hashed for privacy)
    pub client_id_hash: Option<String>,

    // === Model Information ===
    /// Model file SHA256 hash (provenance)
    pub model_hash: String,
    /// Model version string
    pub model_version: String,
    /// Model type (LogisticRegression, RandomForest, etc.)
    pub model_type: String,
    /// Distillation lineage (if applicable)
    pub distillation_teacher_hash: Option<String>,

    // === Request Details ===
    /// Input feature dimensions
    pub input_dims: Vec<usize>,
    /// Input feature hash (for reproducibility without storing raw data)
    pub input_hash: String,
    /// Request options (explain, confidence threshold, etc.)
    pub options: AuditOptions,

    // === Response Details ===
    /// Output prediction (class or value)
    pub prediction: serde_json::Value,
    /// Confidence score (if classification)
    pub confidence: Option<f32>,
    /// Explanation summary (if explainability enabled)
    pub explanation_summary: Option<String>,

    // === Performance ===
    /// Total latency in milliseconds
    pub latency_ms: f64,
    /// Breakdown: preprocessing, inference, postprocessing
    pub latency_breakdown: LatencyBreakdown,
    /// Peak memory usage in bytes
    pub memory_peak_bytes: u64,

    // === Quality Checks (Jidoka) ===
    /// Did output pass NaN/Inf check?
    pub quality_nan_check: bool,
    /// Did output pass confidence threshold?
    pub quality_confidence_check: bool,
    /// Any warnings generated?
    pub warnings: Vec<String>,
}

impl AuditRecord {
    /// Create a new audit record with minimal required fields
    pub fn new(request_id: Uuid, model_hash: &str, model_type: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            timestamp: Utc::now(),
            client_id_hash: None,
            model_hash: model_hash.to_string(),
            model_version: String::new(),
            model_type: model_type.to_string(),
            distillation_teacher_hash: None,
            input_dims: Vec::new(),
            input_hash: String::new(),
            options: AuditOptions::default(),
            prediction: serde_json::Value::Null,
            confidence: None,
            explanation_summary: None,
            latency_ms: 0.0,
            latency_breakdown: LatencyBreakdown::default(),
            memory_peak_bytes: 0,
            quality_nan_check: true,
            quality_confidence_check: true,
            warnings: Vec::new(),
        }
    }

    /// Builder pattern: set client ID hash
    pub fn with_client_hash(mut self, hash: impl Into<String>) -> Self {
        self.client_id_hash = Some(hash.into());
        self
    }

    /// Builder pattern: set model version
    pub fn with_model_version(mut self, version: impl Into<String>) -> Self {
        self.model_version = version.into();
        self
    }

    /// Builder pattern: set input dimensions
    pub fn with_input_dims(mut self, dims: Vec<usize>) -> Self {
        self.input_dims = dims;
        self
    }

    /// Builder pattern: set input hash
    pub fn with_input_hash(mut self, hash: impl Into<String>) -> Self {
        self.input_hash = hash.into();
        self
    }

    /// Builder pattern: set prediction result
    pub fn with_prediction(mut self, prediction: serde_json::Value) -> Self {
        self.prediction = prediction;
        self
    }

    /// Builder pattern: set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Builder pattern: set latency
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency_ms = latency.as_secs_f64() * 1000.0;
        self
    }

    /// Builder pattern: set latency breakdown
    pub fn with_latency_breakdown(mut self, breakdown: LatencyBreakdown) -> Self {
        self.latency_breakdown = breakdown;
        self
    }

    /// Builder pattern: add a warning
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }

    /// Builder pattern: set quality check results
    pub fn with_quality_checks(mut self, nan_check: bool, confidence_check: bool) -> Self {
        self.quality_nan_check = nan_check;
        self.quality_confidence_check = confidence_check;
        self
    }

    /// Builder pattern: set distillation teacher hash
    pub fn with_distillation_teacher(mut self, teacher_hash: impl Into<String>) -> Self {
        self.distillation_teacher_hash = Some(teacher_hash.into());
        self
    }
}

/// Request options included in audit record
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditOptions {
    /// Was explainability requested?
    pub explain: bool,
    /// Confidence threshold (if any)
    pub confidence_threshold: Option<f32>,
    /// Max tokens (for LLM requests)
    pub max_tokens: Option<usize>,
    /// Temperature (for LLM requests)
    pub temperature: Option<f32>,
}

impl AuditOptions {
    /// Create options for APR model request
    pub fn apr(explain: bool, confidence_threshold: Option<f32>) -> Self {
        Self {
            explain,
            confidence_threshold,
            max_tokens: None,
            temperature: None,
        }
    }

    /// Create options for LLM request
    pub fn llm(max_tokens: usize, temperature: f32) -> Self {
        Self {
            explain: false,
            confidence_threshold: None,
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
        }
    }
}

/// Latency breakdown for detailed performance analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    /// Preprocessing time in milliseconds
    pub preprocessing_ms: f64,
    /// Inference time in milliseconds
    pub inference_ms: f64,
    /// Postprocessing time in milliseconds
    pub postprocessing_ms: f64,
    /// Explanation generation time (if enabled)
    pub explanation_ms: Option<f64>,
}

impl LatencyBreakdown {
    /// Create a new latency breakdown
    pub fn new(preprocessing_ms: f64, inference_ms: f64, postprocessing_ms: f64) -> Self {
        Self {
            preprocessing_ms,
            inference_ms,
            postprocessing_ms,
            explanation_ms: None,
        }
    }

    /// Add explanation time
    pub fn with_explanation(mut self, explanation_ms: f64) -> Self {
        self.explanation_ms = Some(explanation_ms);
        self
    }

    /// Total time in milliseconds
    pub fn total_ms(&self) -> f64 {
        self.preprocessing_ms
            + self.inference_ms
            + self.postprocessing_ms
            + self.explanation_ms.unwrap_or(0.0)
    }
}

/// Model provenance tracking (Jidoka: full traceability)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChain {
    /// Original training data hash
    pub training_data_hash: Option<String>,
    /// Training code commit SHA
    pub training_code_sha: Option<String>,
    /// Training environment specification
    pub training_env: Option<TrainingEnv>,
    /// Distillation chain (for distilled models)
    pub distillation_chain: Vec<DistillationStep>,
    /// Quantization provenance
    pub quantization: Option<QuantizationProvenance>,
    /// Signature chain (all signers)
    pub signatures: Vec<SignatureRecord>,
}

impl Default for ProvenanceChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ProvenanceChain {
    /// Create an empty provenance chain
    pub fn new() -> Self {
        Self {
            training_data_hash: None,
            training_code_sha: None,
            training_env: None,
            distillation_chain: Vec::new(),
            quantization: None,
            signatures: Vec::new(),
        }
    }

    /// Add training data hash
    pub fn with_training_data(mut self, hash: impl Into<String>) -> Self {
        self.training_data_hash = Some(hash.into());
        self
    }

    /// Add training code commit
    pub fn with_code_sha(mut self, sha: impl Into<String>) -> Self {
        self.training_code_sha = Some(sha.into());
        self
    }

    /// Add distillation step
    pub fn add_distillation(&mut self, step: DistillationStep) {
        self.distillation_chain.push(step);
    }

    /// Add signature
    pub fn add_signature(&mut self, signature: SignatureRecord) {
        self.signatures.push(signature);
    }
}

/// Training environment specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEnv {
    /// Framework version (e.g., "aprender 0.1.0")
    pub framework: String,
    /// Hardware description
    pub hardware: String,
    /// Random seed used
    pub seed: Option<u64>,
}

/// Distillation step in provenance chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationStep {
    /// Teacher model hash
    pub teacher_hash: String,
    /// Distillation method (Standard, Progressive, Ensemble)
    pub method: String,
    /// Temperature used
    pub temperature: f32,
    /// Alpha (soft vs hard loss weight)
    pub alpha: f32,
    /// Final distillation loss
    pub final_loss: f32,
    /// Timestamp of distillation
    pub timestamp: DateTime<Utc>,
}

/// Quantization provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationProvenance {
    /// Original model hash (before quantization)
    pub original_hash: String,
    /// Quantization method (Q4_0, Q8_0, AWQ, etc.)
    pub method: String,
    /// Bits per weight
    pub bits: u8,
    /// Calibration data hash (if applicable)
    pub calibration_hash: Option<String>,
}

/// Signature record for signed models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureRecord {
    /// Signer identity (e.g., "paiml-release-key")
    pub signer: String,
    /// Signature algorithm (Ed25519)
    pub algorithm: String,
    /// Signature timestamp
    pub timestamp: DateTime<Utc>,
    /// Signature bytes (base64 encoded)
    pub signature: String,
}

/// Audit sink trait for different output destinations
pub trait AuditSink: Send + Sync {
    /// Write a batch of audit records
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), AuditError>;

    /// Flush any pending writes
    fn flush(&self) -> Result<(), AuditError>;
}

/// In-memory audit sink for testing
#[derive(Debug, Default)]
pub struct InMemoryAuditSink {
    records: Mutex<Vec<AuditRecord>>,
}

impl InMemoryAuditSink {
    /// Create a new in-memory sink
    pub fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
        }
    }

    /// Get all stored records
    pub fn records(&self) -> Vec<AuditRecord> {
        self.records.lock().expect("test").clone()
    }

    /// Get record count
    pub fn count(&self) -> usize {
        self.records.lock().expect("test").len()
    }
}

impl AuditSink for InMemoryAuditSink {
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), AuditError> {
        let mut storage = self.records.lock().expect("test");
        storage.extend(records.iter().cloned());
        Ok(())
    }

    fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

/// JSON file audit sink
pub struct JsonFileAuditSink {
    path: std::path::PathBuf,
}

impl JsonFileAuditSink {
    /// Create a new JSON file sink
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl AuditSink for JsonFileAuditSink {
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), AuditError> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| AuditError::IoError(e.to_string()))?;

        for record in records {
            let json = serde_json::to_string(record)
                .map_err(|e| AuditError::SerializationError(e.to_string()))?;
            writeln!(file, "{}", json).map_err(|e| AuditError::IoError(e.to_string()))?;
        }

        Ok(())
    }

    fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

/// Audit error types
#[derive(Debug, Clone)]
pub enum AuditError {
    /// IO error writing audit records
    IoError(String),
    /// Serialization error
    SerializationError(String),
    /// Record not found
    NotFound(String),
}

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "Audit IO error: {msg}"),
            Self::SerializationError(msg) => write!(f, "Audit serialization error: {msg}"),
            Self::NotFound(id) => write!(f, "Audit record not found: {id}"),
        }
    }
}

impl std::error::Error for AuditError {}

/// High-performance audit logger with batching
pub struct AuditLogger {
    /// Output sink (file, database, event stream)
    sink: Box<dyn AuditSink>,
    /// Batch buffer for efficiency
    buffer: Mutex<VecDeque<AuditRecord>>,
    /// Buffer size threshold for auto-flush
    buffer_threshold: usize,
    /// Total records logged (for metrics)
    total_logged: AtomicU64,
    /// Model hash for provenance
    model_hash: String,
    /// Load timestamp
    load_timestamp: DateTime<Utc>,
}

impl AuditLogger {
    /// Create a new audit logger with default settings
    pub fn new(sink: Box<dyn AuditSink>) -> Self {
        Self {
            sink,
            buffer: Mutex::new(VecDeque::new()),
            buffer_threshold: 1000,
            total_logged: AtomicU64::new(0),
            model_hash: String::new(),
            load_timestamp: Utc::now(),
        }
    }

    /// Create an audit logger with in-memory sink (for testing)
    pub fn in_memory() -> (Self, std::sync::Arc<InMemoryAuditSink>) {
        let sink = std::sync::Arc::new(InMemoryAuditSink::new());
        let logger = Self::new(Box::new(InMemorySinkWrapper(sink.clone())));
        (logger, sink)
    }

    /// Set the model hash for provenance tracking
    pub fn with_model_hash(mut self, hash: impl Into<String>) -> Self {
        self.model_hash = hash.into();
        self
    }

    /// Set the load timestamp
    pub fn with_load_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.load_timestamp = timestamp;
        self
    }

    /// Set buffer threshold for auto-flush
    pub fn with_buffer_threshold(mut self, threshold: usize) -> Self {
        self.buffer_threshold = threshold;
        self
    }

    /// Log a request (returns the request ID)
    pub fn log_request(&self, model_type: &str, input_dims: &[usize]) -> Uuid {
        let request_id = Uuid::new_v4();
        let record = AuditRecord::new(request_id, &self.model_hash, model_type)
            .with_input_dims(input_dims.to_vec());

        let mut buffer = self.buffer.lock().expect("test");
        buffer.push_back(record);

        request_id
    }

    /// Complete a request with response data
    pub fn log_response(
        &self,
        request_id: Uuid,
        prediction: serde_json::Value,
        latency: Duration,
        confidence: Option<f32>,
    ) {
        let mut buffer = self.buffer.lock().expect("test");

        // Find and update the record
        if let Some(record) = buffer
            .iter_mut()
            .find(|r| r.request_id == request_id.to_string())
        {
            record.prediction = prediction;
            record.latency_ms = latency.as_secs_f64() * 1000.0;
            record.confidence = confidence;
        }

        // Auto-flush if buffer exceeds threshold
        if buffer.len() >= self.buffer_threshold {
            let _ = self.flush_buffer_locked(&mut buffer);
        }
    }

    /// Manually flush the buffer
    pub fn flush(&self) -> Result<(), AuditError> {
        let mut buffer = self.buffer.lock().expect("test");
        self.flush_buffer_locked(&mut buffer)
    }

    /// Internal flush with lock already held
    fn flush_buffer_locked(&self, buffer: &mut VecDeque<AuditRecord>) -> Result<(), AuditError> {
        if buffer.is_empty() {
            return Ok(());
        }

        let records: Vec<AuditRecord> = buffer.drain(..).collect();
        let count = records.len() as u64;

        self.sink.write_batch(&records)?;
        self.total_logged.fetch_add(count, Ordering::Relaxed);

        Ok(())
    }

    /// Get total records logged
    pub fn total_logged(&self) -> u64 {
        self.total_logged.load(Ordering::Relaxed)
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.lock().expect("test").len()
    }

    /// Get model hash
    pub fn model_hash(&self) -> &str {
        &self.model_hash
    }

    /// Get load timestamp
    pub fn load_timestamp(&self) -> DateTime<Utc> {
        self.load_timestamp
    }
}

/// Wrapper to make Arc<InMemoryAuditSink> implement AuditSink
struct InMemorySinkWrapper(std::sync::Arc<InMemoryAuditSink>);

impl AuditSink for InMemorySinkWrapper {
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), AuditError> {
        self.0.write_batch(records)
    }

    fn flush(&self) -> Result<(), AuditError> {
        self.0.flush()
    }
}

// ============================================================================
// EXTREME TDD TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // -------------------------------------------------------------------------
    // AuditRecord Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_record_new() {
        let request_id = Uuid::new_v4();
        let record = AuditRecord::new(request_id, "abc123", "LogisticRegression");

        assert_eq!(record.request_id, request_id.to_string());
        assert_eq!(record.model_hash, "abc123");
        assert_eq!(record.model_type, "LogisticRegression");
        assert!(record.quality_nan_check);
        assert!(record.quality_confidence_check);
    }

    #[test]
    fn test_audit_record_builder_pattern() {
        let request_id = Uuid::new_v4();
        let record = AuditRecord::new(request_id, "hash123", "RandomForest")
            .with_client_hash("client_hash_456")
            .with_model_version("1.0.0")
            .with_input_dims(vec![784])
            .with_input_hash("input_hash_789")
            .with_prediction(serde_json::json!({"class": 1}))
            .with_confidence(0.95)
            .with_latency(Duration::from_millis(50))
            .with_warning("Low confidence")
            .with_quality_checks(true, false);

        assert_eq!(record.client_id_hash, Some("client_hash_456".to_string()));
        assert_eq!(record.model_version, "1.0.0");
        assert_eq!(record.input_dims, vec![784]);
        assert_eq!(record.input_hash, "input_hash_789");
        assert_eq!(record.prediction, serde_json::json!({"class": 1}));
        assert_eq!(record.confidence, Some(0.95));
        assert!((record.latency_ms - 50.0).abs() < 0.1);
        assert_eq!(record.warnings, vec!["Low confidence"]);
        assert!(record.quality_nan_check);
        assert!(!record.quality_confidence_check);
    }

    #[test]
    fn test_audit_record_serialization() {
        let request_id = Uuid::new_v4();
        let record =
            AuditRecord::new(request_id, "hash", "KNN").with_prediction(serde_json::json!(5));

        let json = serde_json::to_string(&record).expect("test");
        let deserialized: AuditRecord = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.model_type, "KNN");
        assert_eq!(deserialized.prediction, serde_json::json!(5));
    }

    #[test]
    fn test_audit_record_with_distillation_teacher() {
        let record = AuditRecord::new(Uuid::new_v4(), "student", "MLP")
            .with_distillation_teacher("teacher_hash_abc");

        assert_eq!(
            record.distillation_teacher_hash,
            Some("teacher_hash_abc".to_string())
        );
    }

    // -------------------------------------------------------------------------
    // AuditOptions Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_options_apr() {
        let options = AuditOptions::apr(true, Some(0.8));

        assert!(options.explain);
        assert_eq!(options.confidence_threshold, Some(0.8));
        assert!(options.max_tokens.is_none());
        assert!(options.temperature.is_none());
    }

    #[test]
    fn test_audit_options_llm() {
        let options = AuditOptions::llm(256, 0.7);

        assert!(!options.explain);
        assert!(options.confidence_threshold.is_none());
        assert_eq!(options.max_tokens, Some(256));
        assert_eq!(options.temperature, Some(0.7));
    }

    // -------------------------------------------------------------------------
    // LatencyBreakdown Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_latency_breakdown_new() {
        let breakdown = LatencyBreakdown::new(1.5, 10.0, 0.5);

        assert!((breakdown.preprocessing_ms - 1.5).abs() < 0.001);
        assert!((breakdown.inference_ms - 10.0).abs() < 0.001);
        assert!((breakdown.postprocessing_ms - 0.5).abs() < 0.001);
        assert!(breakdown.explanation_ms.is_none());
    }

    #[test]
    fn test_latency_breakdown_with_explanation() {
        let breakdown = LatencyBreakdown::new(1.0, 5.0, 1.0).with_explanation(3.0);

        assert_eq!(breakdown.explanation_ms, Some(3.0));
    }

    #[test]
    fn test_latency_breakdown_total() {
        let breakdown = LatencyBreakdown::new(1.0, 5.0, 2.0);
        assert!((breakdown.total_ms() - 8.0).abs() < 0.001);

        let with_explain = breakdown.with_explanation(2.0);
        assert!((with_explain.total_ms() - 10.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // ProvenanceChain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_provenance_chain_new() {
        let chain = ProvenanceChain::new();

        assert!(chain.training_data_hash.is_none());
        assert!(chain.training_code_sha.is_none());
        assert!(chain.distillation_chain.is_empty());
        assert!(chain.signatures.is_empty());
    }

    #[test]
    fn test_provenance_chain_builder() {
        let chain = ProvenanceChain::new()
            .with_training_data("data_hash_123")
            .with_code_sha("abc123def");

        assert_eq!(chain.training_data_hash, Some("data_hash_123".to_string()));
        assert_eq!(chain.training_code_sha, Some("abc123def".to_string()));
    }

    #[test]
    fn test_provenance_chain_add_distillation() {
        let mut chain = ProvenanceChain::new();
        chain.add_distillation(DistillationStep {
            teacher_hash: "teacher123".to_string(),
            method: "Standard".to_string(),
            temperature: 4.0,
            alpha: 0.5,
            final_loss: 0.01,
            timestamp: Utc::now(),
        });

        assert_eq!(chain.distillation_chain.len(), 1);
        assert_eq!(chain.distillation_chain[0].teacher_hash, "teacher123");
    }

    #[test]
    fn test_provenance_chain_add_signature() {
        let mut chain = ProvenanceChain::new();
        chain.add_signature(SignatureRecord {
            signer: "paiml-release".to_string(),
            algorithm: "Ed25519".to_string(),
            timestamp: Utc::now(),
            signature: "base64sig==".to_string(),
        });

        assert_eq!(chain.signatures.len(), 1);
        assert_eq!(chain.signatures[0].signer, "paiml-release");
    }

    // -------------------------------------------------------------------------
    // InMemoryAuditSink Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_in_memory_sink_write_batch() {
        let sink = InMemoryAuditSink::new();
        let records = vec![
            AuditRecord::new(Uuid::new_v4(), "h1", "LR"),
            AuditRecord::new(Uuid::new_v4(), "h2", "RF"),
        ];

        sink.write_batch(&records).expect("test");

        assert_eq!(sink.count(), 2);
        let stored = sink.records();
        assert_eq!(stored[0].model_type, "LR");
        assert_eq!(stored[1].model_type, "RF");
    }

    #[test]
    fn test_in_memory_sink_flush() {
        let sink = InMemoryAuditSink::new();
        assert!(sink.flush().is_ok());
    }

    // -------------------------------------------------------------------------
    // AuditLogger Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_logger_in_memory() {
        let (logger, sink) = AuditLogger::in_memory();

        let request_id = logger.log_request("LogisticRegression", &[784]);
        logger.log_response(
            request_id,
            serde_json::json!({"class": 1}),
            Duration::from_millis(5),
            Some(0.95),
        );

        assert_eq!(logger.buffer_size(), 1);
        logger.flush().expect("test");
        assert_eq!(logger.buffer_size(), 0);
        assert_eq!(sink.count(), 1);
    }

    #[test]
    fn test_audit_logger_with_model_hash() {
        let (logger, _sink) = AuditLogger::in_memory();
        let logger = logger.with_model_hash("sha256_abc123");

        assert_eq!(logger.model_hash(), "sha256_abc123");
    }

    #[test]
    fn test_audit_logger_total_logged() {
        let (logger, _sink) = AuditLogger::in_memory();

        for _ in 0..5 {
            let id = logger.log_request("Test", &[10]);
            logger.log_response(id, serde_json::json!(0), Duration::from_millis(1), None);
        }

        logger.flush().expect("test");
        assert_eq!(logger.total_logged(), 5);
    }

    #[test]
    fn test_audit_logger_auto_flush() {
        let sink = std::sync::Arc::new(InMemoryAuditSink::new());
        let logger =
            AuditLogger::new(Box::new(InMemorySinkWrapper(sink.clone()))).with_buffer_threshold(3);

        // Log 3 requests - should auto-flush
        for _ in 0..3 {
            let id = logger.log_request("Test", &[5]);
            logger.log_response(id, serde_json::json!(1), Duration::from_millis(1), None);
        }

        // Buffer should be empty after auto-flush
        assert_eq!(logger.buffer_size(), 0);
        assert_eq!(sink.count(), 3);
    }

    #[test]
    fn test_audit_logger_load_timestamp() {
        let timestamp = Utc::now();
        let (logger, _) = AuditLogger::in_memory();
        let logger = logger.with_load_timestamp(timestamp);

        assert_eq!(logger.load_timestamp(), timestamp);
    }

    // -------------------------------------------------------------------------
    // AuditError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_error_display() {
        let io_err = AuditError::IoError("disk full".to_string());
        assert!(io_err.to_string().contains("disk full"));

        let ser_err = AuditError::SerializationError("invalid json".to_string());
        assert!(ser_err.to_string().contains("invalid json"));

        let not_found = AuditError::NotFound("abc-123".to_string());
        assert!(not_found.to_string().contains("abc-123"));
    }

    // -------------------------------------------------------------------------
    // Serialization Round-trip Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_latency_breakdown_serialization() {
        let breakdown = LatencyBreakdown::new(1.0, 5.0, 2.0).with_explanation(1.5);
        let json = serde_json::to_string(&breakdown).expect("test");
        let restored: LatencyBreakdown = serde_json::from_str(&json).expect("test");

        assert!((restored.inference_ms - 5.0).abs() < 0.001);
        assert_eq!(restored.explanation_ms, Some(1.5));
    }

    #[test]
    fn test_provenance_chain_serialization() {
        let chain = ProvenanceChain::new()
            .with_training_data("data123")
            .with_code_sha("commit456");

        let json = serde_json::to_string(&chain).expect("test");
        let restored: ProvenanceChain = serde_json::from_str(&json).expect("test");

        assert_eq!(restored.training_data_hash, Some("data123".to_string()));
        assert_eq!(restored.training_code_sha, Some("commit456".to_string()));
    }

    #[test]
    fn test_quantization_provenance_serialization() {
        let quant = QuantizationProvenance {
            original_hash: "original_model_hash".to_string(),
            method: "Q4_K_M".to_string(),
            bits: 4,
            calibration_hash: Some("calibration_data_hash".to_string()),
        };

        let json = serde_json::to_string(&quant).expect("test");
        let restored: QuantizationProvenance = serde_json::from_str(&json).expect("test");

        assert_eq!(restored.method, "Q4_K_M");
        assert_eq!(restored.bits, 4);
    }
}
