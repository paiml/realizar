//! GPU Diagnostics Module (PMAT-802)
//!
//! Extracted from gpu/mod.rs - Logging, metrics, and debugging infrastructure.
//!
//! ## Contents
//! - `Logger`, `LogConfig`, `LogLevel`, `LogEntry` - Structured logging (IMP-079)
//! - `PhaseTimer`, `MemoryTracker`, `MemoryReport` - Performance tracking (IMP-080)
//! - `DiagnosticsCollector`, `DebugMode`, `RequestCapture`, `StateDump` - Debug tools (IMP-081)

use std::collections::HashMap;

// ============================================================================
// M32: Production Logging & Diagnostics (IMP-079, IMP-080, IMP-081)
// ============================================================================

/// Log level (IMP-079)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Trace level (most verbose)
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warn,
    /// Error level
    Error,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

/// Log entry (IMP-079)
#[derive(Debug, Clone)]
pub struct LogEntry {
    level: LogLevel,
    message: String,
    timestamp: u64,
    correlation_id: Option<String>,
    fields: HashMap<String, String>,
}

impl LogEntry {
    /// Create new log entry
    #[must_use]
    pub fn new(level: LogLevel, message: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            level,
            message: message.to_string(),
            timestamp,
            correlation_id: None,
            fields: HashMap::new(),
        }
    }

    /// Set correlation ID
    #[must_use]
    pub fn with_correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }

    /// Add custom field
    #[must_use]
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }

    /// Get correlation ID
    #[must_use]
    pub fn correlation_id(&self) -> Option<&str> {
        self.correlation_id.as_deref()
    }

    /// Get log level
    #[must_use]
    pub fn level(&self) -> LogLevel {
        self.level
    }

    /// Get timestamp
    #[must_use]
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Convert to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        use std::fmt::Write;

        let mut json = format!(
            "{{\"level\":\"{}\",\"message\":\"{}\",\"timestamp\":{}",
            self.level.as_str(),
            self.message,
            self.timestamp
        );

        if let Some(ref id) = self.correlation_id {
            let _ = write!(json, ",\"correlation_id\":\"{}\"", id);
        }

        for (key, value) in &self.fields {
            let _ = write!(json, ",\"{}\":\"{}\"", key, value);
        }

        json.push('}');
        json
    }
}

/// Log configuration (IMP-079)
#[derive(Debug, Clone)]
pub struct LogConfig {
    default_level: LogLevel,
    json_format: bool,
    module_levels: HashMap<String, LogLevel>,
}

impl LogConfig {
    /// Create new log config
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_level: LogLevel::Info,
            json_format: false,
            module_levels: HashMap::new(),
        }
    }

    /// Set default log level
    #[must_use]
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.default_level = level;
        self
    }

    /// Enable JSON format
    #[must_use]
    pub fn with_json_format(mut self, enabled: bool) -> Self {
        self.json_format = enabled;
        self
    }

    /// Set module-specific log level
    #[must_use]
    pub fn with_module_level(mut self, module: &str, level: LogLevel) -> Self {
        self.module_levels.insert(module.to_string(), level);
        self
    }
}

impl Default for LogConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Logger (IMP-079)
pub struct Logger {
    config: LogConfig,
}

impl Logger {
    /// Create new logger
    #[must_use]
    pub fn new(config: LogConfig) -> Self {
        Self { config }
    }

    /// Check if log level is enabled for module
    #[must_use]
    pub fn is_enabled(&self, level: LogLevel, module: &str) -> bool {
        let min_level = self
            .config
            .module_levels
            .get(module)
            .copied()
            .unwrap_or(self.config.default_level);
        level >= min_level
    }
}

/// Phase timer for latency breakdown (IMP-080)
pub struct PhaseTimer {
    phases: std::sync::Mutex<HashMap<String, (Option<std::time::Instant>, u64)>>,
}

impl PhaseTimer {
    /// Create new phase timer
    #[must_use]
    pub fn new() -> Self {
        Self {
            phases: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Start timing a phase
    pub fn start_phase(&self, name: &str) {
        let mut phases = self.phases.lock().expect("mutex poisoned");
        phases.insert(name.to_string(), (Some(std::time::Instant::now()), 0));
    }

    /// End timing a phase
    pub fn end_phase(&self, name: &str) {
        let mut phases = self.phases.lock().expect("mutex poisoned");
        if let Some((Some(start_time), _)) = phases.get(name) {
            let elapsed = start_time.elapsed().as_micros() as u64;
            phases.insert(name.to_string(), (None, elapsed));
        }
    }

    /// Get timing breakdown
    #[must_use]
    pub fn breakdown(&self) -> HashMap<String, u64> {
        let phases = self.phases.lock().expect("mutex poisoned");
        phases.iter().map(|(k, (_, v))| (k.clone(), *v)).collect()
    }
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory report (IMP-080)
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Peak memory usage in bytes
    pub peak_bytes: u64,
    /// Current memory usage in bytes
    pub current_bytes: u64,
    /// Total allocation count
    pub allocation_count: u64,
}

/// Memory tracker (IMP-080)
pub struct MemoryTracker {
    current: std::sync::atomic::AtomicU64,
    peak: std::sync::atomic::AtomicU64,
    allocation_count: std::sync::atomic::AtomicU64,
}

impl MemoryTracker {
    /// Create new memory tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: std::sync::atomic::AtomicU64::new(0),
            peak: std::sync::atomic::AtomicU64::new(0),
            allocation_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&self, _name: &str, bytes: u64) {
        let new_current = self
            .current
            .fetch_add(bytes, std::sync::atomic::Ordering::SeqCst)
            + bytes;
        self.allocation_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Update peak if necessary
        let mut peak = self.peak.load(std::sync::atomic::Ordering::SeqCst);
        while new_current > peak {
            match self.peak.compare_exchange_weak(
                peak,
                new_current,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current_peak) => peak = current_peak,
            }
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&self, _name: &str, bytes: u64) {
        self.current
            .fetch_sub(bytes, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get memory report
    #[must_use]
    pub fn report(&self) -> MemoryReport {
        MemoryReport {
            peak_bytes: self.peak.load(std::sync::atomic::Ordering::SeqCst),
            current_bytes: self.current.load(std::sync::atomic::Ordering::SeqCst),
            allocation_count: self
                .allocation_count
                .load(std::sync::atomic::Ordering::SeqCst),
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Diagnostics summary (IMP-080)
#[derive(Debug, Clone)]
pub struct DiagnosticsSummary {
    /// Number of requests tracked
    pub request_count: u64,
}

/// Diagnostics collector (IMP-080)
pub struct DiagnosticsCollector {
    /// Request count (pub(crate) for test access)
    pub(crate) request_count: std::sync::atomic::AtomicU64,
    #[allow(dead_code)]
    timings: std::sync::Mutex<Vec<HashMap<String, u64>>>,
    #[allow(dead_code)]
    memory_snapshots: std::sync::Mutex<Vec<MemoryReport>>,
}

impl DiagnosticsCollector {
    /// Create new diagnostics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            request_count: std::sync::atomic::AtomicU64::new(0),
            timings: std::sync::Mutex::new(Vec::new()),
            memory_snapshots: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Record request timing
    pub fn record_request_timing(&self, _request_id: &str, timing: HashMap<String, u64>) {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.timings.lock().expect("mutex poisoned").push(timing);
    }

    /// Record memory snapshot
    pub fn record_memory_snapshot(&self, report: MemoryReport) {
        self.memory_snapshots
            .lock()
            .expect("mutex poisoned")
            .push(report);
    }

    /// Get diagnostics summary
    #[must_use]
    pub fn summary(&self) -> DiagnosticsSummary {
        DiagnosticsSummary {
            request_count: self.request_count.load(std::sync::atomic::Ordering::SeqCst),
        }
    }
}

impl Default for DiagnosticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug mode controller (IMP-081)
pub struct DebugMode {
    enabled: std::sync::atomic::AtomicBool,
}

impl DebugMode {
    /// Create new debug mode
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Check if debug mode is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Enable debug mode
    pub fn enable(&self) {
        self.enabled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Disable debug mode
    #[allow(dead_code)]
    pub fn disable(&self) {
        self.enabled
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Default for DebugMode {
    fn default() -> Self {
        Self::new()
    }
}

/// Request capture for replay (IMP-081)
#[derive(Debug, Clone)]
pub struct RequestCapture {
    input: String,
    params: HashMap<String, String>,
}

impl RequestCapture {
    /// Create new request capture
    #[must_use]
    pub fn new() -> Self {
        Self {
            input: String::new(),
            params: HashMap::new(),
        }
    }

    /// Set input
    #[must_use]
    pub fn with_input(mut self, input: &str) -> Self {
        self.input = input.to_string();
        self
    }

    /// Add parameter
    #[must_use]
    pub fn with_params(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }

    /// Get input
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Get params
    #[must_use]
    pub fn params(&self) -> &HashMap<String, String> {
        &self.params
    }

    /// Serialize to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        let params_json: Vec<String> = self
            .params
            .iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
            .collect();
        format!(
            "{{\"input\":\"{}\",\"params\":{{{}}}}}",
            self.input,
            params_json.join(",")
        )
    }

    /// Deserialize from JSON (simple implementation)
    ///
    /// # Errors
    /// Returns error if JSON is malformed or missing input field.
    pub fn from_json(json: &str) -> std::result::Result<Self, &'static str> {
        // Simple extraction - production would use serde
        let input_start = json.find("\"input\":\"").ok_or("Missing input")?;
        let input_rest = &json[input_start + 9..];
        let input_end = input_rest.find('"').ok_or("Invalid input")?;
        let input = &input_rest[..input_end];

        Ok(Self {
            input: input.to_string(),
            params: HashMap::new(),
        })
    }
}

impl Default for RequestCapture {
    fn default() -> Self {
        Self::new()
    }
}

/// State dump for debugging (IMP-081)
#[derive(Debug, Clone)]
pub struct StateDump {
    error: String,
    stack_trace: String,
    state: HashMap<String, String>,
}

impl StateDump {
    /// Create new state dump
    #[must_use]
    pub fn new() -> Self {
        Self {
            error: String::new(),
            stack_trace: String::new(),
            state: HashMap::new(),
        }
    }

    /// Set error
    #[must_use]
    pub fn with_error(mut self, error: &str) -> Self {
        self.error = error.to_string();
        self
    }

    /// Set stack trace
    #[must_use]
    pub fn with_stack_trace(mut self, trace: &str) -> Self {
        self.stack_trace = trace.to_string();
        self
    }

    /// Add state
    #[must_use]
    pub fn with_state(mut self, key: &str, value: &str) -> Self {
        self.state.insert(key.to_string(), value.to_string());
        self
    }

    /// Get error
    #[must_use]
    pub fn error(&self) -> &str {
        &self.error
    }

    /// Get stack trace
    #[must_use]
    pub fn stack_trace(&self) -> &str {
        &self.stack_trace
    }

    /// Get state
    #[must_use]
    pub fn state(&self) -> &HashMap<String, String> {
        &self.state
    }

    /// Convert to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        let state_json: Vec<String> = self
            .state
            .iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
            .collect();
        format!(
            "{{\"error\":\"{}\",\"stack_trace\":\"{}\",\"state\":{{{}}}}}",
            self.error,
            self.stack_trace.replace('\n', "\\n"),
            state_json.join(",")
        )
    }
}

impl Default for StateDump {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== LogLevel Tests ====================

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Trace.as_str(), "TRACE");
        assert_eq!(LogLevel::Debug.as_str(), "DEBUG");
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Warn.as_str(), "WARN");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
    }

    #[test]
    fn test_log_level_clone_copy() {
        let level = LogLevel::Info;
        let cloned = level.clone();
        let copied: LogLevel = level;
        assert_eq!(level, cloned);
        assert_eq!(level, copied);
    }

    #[test]
    fn test_log_level_eq() {
        assert_eq!(LogLevel::Error, LogLevel::Error);
        assert_ne!(LogLevel::Error, LogLevel::Warn);
    }

    // ==================== LogEntry Tests ====================

    #[test]
    fn test_log_entry_new() {
        let entry = LogEntry::new(LogLevel::Info, "Test message");
        assert_eq!(entry.level(), LogLevel::Info);
        assert!(entry.timestamp() > 0);
        assert!(entry.correlation_id().is_none());
    }

    #[test]
    fn test_log_entry_with_correlation_id() {
        let entry = LogEntry::new(LogLevel::Debug, "msg")
            .with_correlation_id("req-123");
        assert_eq!(entry.correlation_id(), Some("req-123"));
    }

    #[test]
    fn test_log_entry_with_field() {
        let entry = LogEntry::new(LogLevel::Warn, "warning")
            .with_field("key1", "value1")
            .with_field("key2", "value2");
        let json = entry.to_json();
        assert!(json.contains("\"key1\":\"value1\""));
        assert!(json.contains("\"key2\":\"value2\""));
    }

    #[test]
    fn test_log_entry_to_json_basic() {
        let entry = LogEntry::new(LogLevel::Error, "error message");
        let json = entry.to_json();
        assert!(json.contains("\"level\":\"ERROR\""));
        assert!(json.contains("\"message\":\"error message\""));
        assert!(json.contains("\"timestamp\":"));
    }

    #[test]
    fn test_log_entry_to_json_with_correlation() {
        let entry = LogEntry::new(LogLevel::Info, "msg")
            .with_correlation_id("corr-456");
        let json = entry.to_json();
        assert!(json.contains("\"correlation_id\":\"corr-456\""));
    }

    #[test]
    fn test_log_entry_clone() {
        let entry = LogEntry::new(LogLevel::Debug, "test")
            .with_correlation_id("id")
            .with_field("k", "v");
        let cloned = entry.clone();
        assert_eq!(cloned.level(), entry.level());
        assert_eq!(cloned.correlation_id(), entry.correlation_id());
    }

    // ==================== LogConfig Tests ====================

    #[test]
    fn test_log_config_new() {
        let config = LogConfig::new();
        assert!(!config.json_format);
    }

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert!(!config.json_format);
    }

    #[test]
    fn test_log_config_with_level() {
        let config = LogConfig::new().with_level(LogLevel::Debug);
        assert_eq!(config.default_level, LogLevel::Debug);
    }

    #[test]
    fn test_log_config_with_json_format() {
        let config = LogConfig::new().with_json_format(true);
        assert!(config.json_format);
    }

    #[test]
    fn test_log_config_with_module_level() {
        let config = LogConfig::new()
            .with_module_level("gpu", LogLevel::Trace)
            .with_module_level("inference", LogLevel::Error);
        assert_eq!(config.module_levels.get("gpu"), Some(&LogLevel::Trace));
        assert_eq!(config.module_levels.get("inference"), Some(&LogLevel::Error));
    }

    #[test]
    fn test_log_config_clone() {
        let config = LogConfig::new()
            .with_level(LogLevel::Warn)
            .with_json_format(true);
        let cloned = config.clone();
        assert_eq!(cloned.default_level, config.default_level);
        assert_eq!(cloned.json_format, config.json_format);
    }

    // ==================== Logger Tests ====================

    #[test]
    fn test_logger_new() {
        let config = LogConfig::new().with_level(LogLevel::Info);
        let logger = Logger::new(config);
        assert!(logger.is_enabled(LogLevel::Error, "any_module"));
    }

    #[test]
    fn test_logger_is_enabled_default_level() {
        let config = LogConfig::new().with_level(LogLevel::Info);
        let logger = Logger::new(config);

        assert!(!logger.is_enabled(LogLevel::Trace, "module"));
        assert!(!logger.is_enabled(LogLevel::Debug, "module"));
        assert!(logger.is_enabled(LogLevel::Info, "module"));
        assert!(logger.is_enabled(LogLevel::Warn, "module"));
        assert!(logger.is_enabled(LogLevel::Error, "module"));
    }

    #[test]
    fn test_logger_is_enabled_module_level() {
        let config = LogConfig::new()
            .with_level(LogLevel::Info)
            .with_module_level("gpu", LogLevel::Trace);
        let logger = Logger::new(config);

        // GPU module should allow Trace
        assert!(logger.is_enabled(LogLevel::Trace, "gpu"));
        // Other modules use default Info level
        assert!(!logger.is_enabled(LogLevel::Debug, "other"));
    }

    // ==================== PhaseTimer Tests ====================

    #[test]
    fn test_phase_timer_new() {
        let timer = PhaseTimer::new();
        let breakdown = timer.breakdown();
        assert!(breakdown.is_empty());
    }

    #[test]
    fn test_phase_timer_default() {
        let timer = PhaseTimer::default();
        assert!(timer.breakdown().is_empty());
    }

    #[test]
    fn test_phase_timer_start_end_phase() {
        let timer = PhaseTimer::new();
        timer.start_phase("attention");
        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.end_phase("attention");

        let breakdown = timer.breakdown();
        assert!(breakdown.contains_key("attention"));
        assert!(breakdown["attention"] > 0);
    }

    #[test]
    fn test_phase_timer_multiple_phases() {
        let timer = PhaseTimer::new();
        timer.start_phase("phase1");
        timer.end_phase("phase1");
        timer.start_phase("phase2");
        timer.end_phase("phase2");

        let breakdown = timer.breakdown();
        assert_eq!(breakdown.len(), 2);
    }

    #[test]
    fn test_phase_timer_end_without_start() {
        let timer = PhaseTimer::new();
        timer.end_phase("nonexistent");
        // Should not panic, just ignore
        assert!(timer.breakdown().is_empty());
    }

    // ==================== MemoryReport Tests ====================

    #[test]
    fn test_memory_report_fields() {
        let report = MemoryReport {
            peak_bytes: 1024,
            current_bytes: 512,
            allocation_count: 10,
        };
        assert_eq!(report.peak_bytes, 1024);
        assert_eq!(report.current_bytes, 512);
        assert_eq!(report.allocation_count, 10);
    }

    #[test]
    fn test_memory_report_clone() {
        let report = MemoryReport {
            peak_bytes: 2048,
            current_bytes: 1024,
            allocation_count: 5,
        };
        let cloned = report.clone();
        assert_eq!(cloned.peak_bytes, report.peak_bytes);
    }

    // ==================== MemoryTracker Tests ====================

    #[test]
    fn test_memory_tracker_new() {
        let tracker = MemoryTracker::new();
        let report = tracker.report();
        assert_eq!(report.current_bytes, 0);
        assert_eq!(report.peak_bytes, 0);
        assert_eq!(report.allocation_count, 0);
    }

    #[test]
    fn test_memory_tracker_default() {
        let tracker = MemoryTracker::default();
        let report = tracker.report();
        assert_eq!(report.current_bytes, 0);
    }

    #[test]
    fn test_memory_tracker_record_allocation() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation("buffer1", 1024);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 1024);
        assert_eq!(report.peak_bytes, 1024);
        assert_eq!(report.allocation_count, 1);
    }

    #[test]
    fn test_memory_tracker_multiple_allocations() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation("a", 100);
        tracker.record_allocation("b", 200);
        tracker.record_allocation("c", 300);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 600);
        assert_eq!(report.peak_bytes, 600);
        assert_eq!(report.allocation_count, 3);
    }

    #[test]
    fn test_memory_tracker_record_deallocation() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation("buf", 1000);
        tracker.record_deallocation("buf", 400);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 600);
        assert_eq!(report.peak_bytes, 1000); // Peak unchanged
    }

    #[test]
    fn test_memory_tracker_peak_tracking() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation("a", 500);
        tracker.record_allocation("b", 500);
        // Peak is now 1000
        tracker.record_deallocation("a", 500);
        tracker.record_allocation("c", 200);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 700);
        assert_eq!(report.peak_bytes, 1000); // Peak was 1000
    }

    // ==================== DiagnosticsCollector Tests ====================

    #[test]
    fn test_diagnostics_collector_new() {
        let collector = DiagnosticsCollector::new();
        let summary = collector.summary();
        assert_eq!(summary.request_count, 0);
    }

    #[test]
    fn test_diagnostics_collector_default() {
        let collector = DiagnosticsCollector::default();
        assert_eq!(collector.summary().request_count, 0);
    }

    #[test]
    fn test_diagnostics_collector_record_request_timing() {
        let collector = DiagnosticsCollector::new();
        let mut timing = HashMap::new();
        timing.insert("attention".to_string(), 100);
        timing.insert("ffn".to_string(), 50);

        collector.record_request_timing("req-1", timing);

        let summary = collector.summary();
        assert_eq!(summary.request_count, 1);
    }

    #[test]
    fn test_diagnostics_collector_multiple_requests() {
        let collector = DiagnosticsCollector::new();

        collector.record_request_timing("r1", HashMap::new());
        collector.record_request_timing("r2", HashMap::new());
        collector.record_request_timing("r3", HashMap::new());

        assert_eq!(collector.summary().request_count, 3);
    }

    #[test]
    fn test_diagnostics_collector_record_memory_snapshot() {
        let collector = DiagnosticsCollector::new();
        let report = MemoryReport {
            peak_bytes: 1024,
            current_bytes: 512,
            allocation_count: 2,
        };

        collector.record_memory_snapshot(report);
        // No assertion needed - just verify no panic
    }

    // ==================== DebugMode Tests ====================

    #[test]
    fn test_debug_mode_new() {
        let debug = DebugMode::new();
        assert!(!debug.is_enabled());
    }

    #[test]
    fn test_debug_mode_default() {
        let debug = DebugMode::default();
        assert!(!debug.is_enabled());
    }

    #[test]
    fn test_debug_mode_enable() {
        let debug = DebugMode::new();
        assert!(!debug.is_enabled());
        debug.enable();
        assert!(debug.is_enabled());
    }

    #[test]
    fn test_debug_mode_disable() {
        let debug = DebugMode::new();
        debug.enable();
        assert!(debug.is_enabled());
        debug.disable();
        assert!(!debug.is_enabled());
    }

    // ==================== RequestCapture Tests ====================

    #[test]
    fn test_request_capture_new() {
        let capture = RequestCapture::new();
        assert_eq!(capture.input(), "");
        assert!(capture.params().is_empty());
    }

    #[test]
    fn test_request_capture_default() {
        let capture = RequestCapture::default();
        assert_eq!(capture.input(), "");
    }

    #[test]
    fn test_request_capture_with_input() {
        let capture = RequestCapture::new().with_input("Hello, world!");
        assert_eq!(capture.input(), "Hello, world!");
    }

    #[test]
    fn test_request_capture_with_params() {
        let capture = RequestCapture::new()
            .with_params("temp", "0.7")
            .with_params("max_tokens", "100");

        assert_eq!(capture.params().get("temp"), Some(&"0.7".to_string()));
        assert_eq!(capture.params().get("max_tokens"), Some(&"100".to_string()));
    }

    #[test]
    fn test_request_capture_to_json() {
        let capture = RequestCapture::new()
            .with_input("prompt")
            .with_params("temp", "0.5");
        let json = capture.to_json();

        assert!(json.contains("\"input\":\"prompt\""));
        assert!(json.contains("\"temp\":\"0.5\""));
    }

    #[test]
    fn test_request_capture_from_json() {
        let json = "{\"input\":\"test prompt\",\"params\":{}}";
        let capture = RequestCapture::from_json(json).unwrap();
        assert_eq!(capture.input(), "test prompt");
    }

    #[test]
    fn test_request_capture_from_json_error() {
        let result = RequestCapture::from_json("{\"no_input\":\"value\"}");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Missing input");
    }

    #[test]
    fn test_request_capture_from_json_invalid() {
        let result = RequestCapture::from_json("{\"input\":}");
        assert!(result.is_err());
    }

    #[test]
    fn test_request_capture_clone() {
        let capture = RequestCapture::new()
            .with_input("test")
            .with_params("k", "v");
        let cloned = capture.clone();
        assert_eq!(cloned.input(), capture.input());
    }

    // ==================== StateDump Tests ====================

    #[test]
    fn test_state_dump_new() {
        let dump = StateDump::new();
        assert_eq!(dump.error(), "");
        assert_eq!(dump.stack_trace(), "");
        assert!(dump.state().is_empty());
    }

    #[test]
    fn test_state_dump_default() {
        let dump = StateDump::default();
        assert_eq!(dump.error(), "");
    }

    #[test]
    fn test_state_dump_with_error() {
        let dump = StateDump::new().with_error("OutOfMemory");
        assert_eq!(dump.error(), "OutOfMemory");
    }

    #[test]
    fn test_state_dump_with_stack_trace() {
        let dump = StateDump::new().with_stack_trace("at func1\nat func2");
        assert_eq!(dump.stack_trace(), "at func1\nat func2");
    }

    #[test]
    fn test_state_dump_with_state() {
        let dump = StateDump::new()
            .with_state("batch_size", "32")
            .with_state("seq_len", "256");

        assert_eq!(dump.state().get("batch_size"), Some(&"32".to_string()));
        assert_eq!(dump.state().get("seq_len"), Some(&"256".to_string()));
    }

    #[test]
    fn test_state_dump_to_json() {
        let dump = StateDump::new()
            .with_error("Error")
            .with_stack_trace("trace")
            .with_state("key", "value");
        let json = dump.to_json();

        assert!(json.contains("\"error\":\"Error\""));
        assert!(json.contains("\"stack_trace\":\"trace\""));
        assert!(json.contains("\"key\":\"value\""));
    }

    #[test]
    fn test_state_dump_to_json_newline_escape() {
        let dump = StateDump::new().with_stack_trace("line1\nline2");
        let json = dump.to_json();
        assert!(json.contains("\\n"));
    }

    #[test]
    fn test_state_dump_clone() {
        let dump = StateDump::new()
            .with_error("err")
            .with_stack_trace("trace");
        let cloned = dump.clone();
        assert_eq!(cloned.error(), dump.error());
        assert_eq!(cloned.stack_trace(), dump.stack_trace());
    }

    // ==================== DiagnosticsSummary Tests ====================

    #[test]
    fn test_diagnostics_summary_fields() {
        let summary = DiagnosticsSummary { request_count: 42 };
        assert_eq!(summary.request_count, 42);
    }

    #[test]
    fn test_diagnostics_summary_clone() {
        let summary = DiagnosticsSummary { request_count: 10 };
        let cloned = summary.clone();
        assert_eq!(cloned.request_count, summary.request_count);
    }
}
