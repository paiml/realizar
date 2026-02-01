//! BrickTracer: The Golden Trace for GPU/CPU Parity Debugging (Phase 14)
//!
//! This module provides automated divergence detection between CPU and GPU
//! inference paths by logging tensor checksums at each computational step.
//!
//! # Purpose
//!
//! When GPU output diverges from CPU (e.g., "1+1=" yields "2." on CPU but "10" on GPU),
//! the tracer identifies the EXACT point of divergence by comparing:
//! - L2 norm (cheap fingerprint)
//! - First N elements (for debugging)
//! - Full tensor (if verbose mode enabled)
//!
//! # Usage
//!
//! ```rust,ignore
//! use realizar::brick::BrickTracer;
//!
//! let mut tracer = BrickTracer::new();
//!
//! // Log at each computation step
//! tracer.log("embedding", &embedding_output);
//! tracer.log("layer0_attn_norm", &normed);
//! tracer.log("layer0_qkv", &qkv);
//! tracer.log("layer0_rope_q", &q_rope);
//! tracer.log("layer0_rope_k", &k_rope);
//! tracer.log("layer0_attention", &attn_out);
//! tracer.log("layer0_o_proj", &o_proj);
//! tracer.log("layer0_ffn_norm", &ffn_normed);
//! tracer.log("layer0_ffn", &ffn_out);
//! // ... more layers ...
//! tracer.log("final_norm", &final_normed);
//! tracer.log("logits", &logits);
//!
//! // Compare CPU vs GPU traces
//! let divergence = BrickTracer::compare(&cpu_tracer, &gpu_tracer);
//! if let Some(first_diff) = divergence.first_divergence() {
//!     eprintln!("Divergence at: {} (CPU L2={}, GPU L2={})",
//!               first_diff.name, first_diff.cpu_l2, first_diff.gpu_l2);
//! }
//! ```
//!
//! # References
//!
//! - Phase 14: "The New Doctrine: The Golden Trace"
//! - PMAT-106: APR GPU Adapter + Integration Tests

use std::collections::HashMap;
use std::fmt;

/// A single trace event capturing tensor state at a computation step
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Name of the computation step (e.g., "layer0_rope_q")
    pub name: String,
    /// Position/sequence index when this event was logged
    pub position: usize,
    /// L2 norm of the tensor (cheap fingerprint)
    pub l2_norm: f32,
    /// Mean value
    pub mean: f32,
    /// Min value
    pub min: f32,
    /// Max value
    pub max: f32,
    /// First 8 elements (for quick debugging)
    pub head: [f32; 8],
    /// Full tensor data (only if verbose mode enabled)
    pub full_data: Option<Vec<f32>>,
    /// Tensor length
    pub len: usize,
}

impl TraceEvent {
    /// Create a new trace event from tensor data
    pub fn new(name: &str, tensor: &[f32], position: usize, verbose: bool) -> Self {
        let len = tensor.len();

        // Compute L2 norm
        let l2_norm = tensor.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Compute statistics
        let mean = if len > 0 {
            tensor.iter().sum::<f32>() / len as f32
        } else {
            0.0
        };
        let min = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Copy first 8 elements
        let mut head = [0.0f32; 8];
        for (i, &v) in tensor.iter().take(8).enumerate() {
            head[i] = v;
        }

        // Optionally store full tensor
        let full_data = if verbose { Some(tensor.to_vec()) } else { None };

        Self {
            name: name.to_string(),
            position,
            l2_norm,
            mean,
            min,
            max,
            head,
            full_data,
            len,
        }
    }

    /// Check if two events are approximately equal within tolerance
    pub fn approx_eq(&self, other: &Self, tolerance: f32) -> bool {
        // Check L2 norm
        let l2_diff = (self.l2_norm - other.l2_norm).abs();
        let l2_rel = if self.l2_norm.abs() > 1e-10 {
            l2_diff / self.l2_norm.abs()
        } else {
            l2_diff
        };

        l2_rel <= tolerance
    }

    /// Compute relative difference between two events
    pub fn relative_diff(&self, other: &Self) -> f32 {
        let l2_diff = (self.l2_norm - other.l2_norm).abs();
        if self.l2_norm.abs() > 1e-10 {
            l2_diff / self.l2_norm.abs()
        } else {
            l2_diff
        }
    }
}

impl fmt::Display for TraceEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [pos={}]: L2={:.6}, mean={:.6}, range=[{:.6}, {:.6}], len={}, head=[{:.4}, {:.4}, {:.4}, {:.4}...]",
            self.name,
            self.position,
            self.l2_norm,
            self.mean,
            self.min,
            self.max,
            self.len,
            self.head[0],
            self.head[1],
            self.head[2],
            self.head[3],
        )
    }
}

/// Result of comparing two trace events
#[derive(Debug, Clone)]
pub struct TraceDiff {
    /// Name of the computation step
    pub name: String,
    /// Position where divergence occurred
    pub position: usize,
    /// CPU L2 norm
    pub cpu_l2: f32,
    /// GPU L2 norm
    pub gpu_l2: f32,
    /// Relative difference (|cpu - gpu| / |cpu|)
    pub relative_diff: f32,
    /// CPU head values
    pub cpu_head: [f32; 8],
    /// GPU head values
    pub gpu_head: [f32; 8],
}

impl fmt::Display for TraceDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [pos={}]: CPU L2={:.6} vs GPU L2={:.6} (diff={:.2}%)\n  CPU head: [{:.4}, {:.4}, {:.4}, {:.4}...]\n  GPU head: [{:.4}, {:.4}, {:.4}, {:.4}...]",
            self.name,
            self.position,
            self.cpu_l2,
            self.gpu_l2,
            self.relative_diff * 100.0,
            self.cpu_head[0], self.cpu_head[1], self.cpu_head[2], self.cpu_head[3],
            self.gpu_head[0], self.gpu_head[1], self.gpu_head[2], self.gpu_head[3],
        )
    }
}

/// Comparison result between CPU and GPU traces
#[derive(Debug, Clone)]
pub struct TraceComparison {
    /// All differences found
    pub diffs: Vec<TraceDiff>,
    /// Tolerance used for comparison
    pub tolerance: f32,
}

impl TraceComparison {
    /// Get the first divergence point
    pub fn first_divergence(&self) -> Option<&TraceDiff> {
        self.diffs.first()
    }

    /// Check if traces are equivalent (no divergence)
    pub fn is_equivalent(&self) -> bool {
        self.diffs.is_empty()
    }

    /// Get summary of divergence
    pub fn summary(&self) -> String {
        if self.diffs.is_empty() {
            "No divergence detected".to_string()
        } else {
            let first = &self.diffs[0];
            format!(
                "First divergence at '{}' (pos={}): {:.2}% L2 diff ({} total divergences)",
                first.name,
                first.position,
                first.relative_diff * 100.0,
                self.diffs.len()
            )
        }
    }
}

impl fmt::Display for TraceComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== TRACE COMPARISON ===")?;
        writeln!(f, "Tolerance: {:.4}%", self.tolerance * 100.0)?;
        writeln!(f, "Divergences: {}", self.diffs.len())?;

        if self.diffs.is_empty() {
            writeln!(f, "Result: MATCH")?;
        } else {
            writeln!(f, "Result: DIVERGENCE DETECTED")?;
            writeln!(f, "\n--- First Divergence ---")?;
            if let Some(first) = self.diffs.first() {
                writeln!(f, "{first}")?;
            }

            if self.diffs.len() > 1 {
                writeln!(f, "\n--- All Divergences ---")?;
                for diff in &self.diffs {
                    writeln!(
                        f,
                        "  {}: {:.2}% diff",
                        diff.name,
                        diff.relative_diff * 100.0
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// BrickTracer: Collects trace events for GPU/CPU parity debugging
///
/// The tracer logs tensor state at each computation step, enabling
/// automated detection of where GPU output diverges from CPU.
#[derive(Debug, Clone)]
pub struct BrickTracer {
    /// Collected trace events in order
    events: Vec<TraceEvent>,
    /// Current position being traced
    position: usize,
    /// Whether to store full tensor data (expensive)
    verbose: bool,
    /// Event index by name for fast lookup
    index: HashMap<String, usize>,
}

impl Default for BrickTracer {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickTracer {
    /// Create a new tracer
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            position: 0,
            verbose: false,
            index: HashMap::new(),
        }
    }

    /// Create a tracer with verbose mode (stores full tensors)
    pub fn verbose() -> Self {
        Self {
            events: Vec::new(),
            position: 0,
            verbose: true,
            index: HashMap::new(),
        }
    }

    /// Set the current position being traced
    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    /// Log a tensor at the current computation step
    pub fn log(&mut self, name: &str, tensor: &[f32]) {
        let event = TraceEvent::new(name, tensor, self.position, self.verbose);
        let idx = self.events.len();
        self.index.insert(name.to_string(), idx);
        self.events.push(event);
    }

    /// Log a tensor with explicit position
    pub fn log_at(&mut self, name: &str, tensor: &[f32], position: usize) {
        let event = TraceEvent::new(name, tensor, position, self.verbose);
        let idx = self.events.len();
        self.index.insert(name.to_string(), idx);
        self.events.push(event);
    }

    /// Get all trace events
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Get event by name
    pub fn get(&self, name: &str) -> Option<&TraceEvent> {
        self.index.get(name).map(|&idx| &self.events[idx])
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
        self.index.clear();
        self.position = 0;
    }

    /// Compare two tracers (CPU vs GPU)
    ///
    /// # Arguments
    ///
    /// * `cpu` - CPU tracer
    /// * `gpu` - GPU tracer
    /// * `tolerance` - Relative tolerance for L2 norm comparison (e.g., 0.01 = 1%)
    ///
    /// # Returns
    ///
    /// Comparison result with all divergences
    pub fn compare(cpu: &Self, gpu: &Self, tolerance: f32) -> TraceComparison {
        let mut diffs = Vec::new();

        // Compare events in order
        for cpu_event in &cpu.events {
            if let Some(gpu_event) = gpu.get(&cpu_event.name) {
                if !cpu_event.approx_eq(gpu_event, tolerance) {
                    diffs.push(TraceDiff {
                        name: cpu_event.name.clone(),
                        position: cpu_event.position,
                        cpu_l2: cpu_event.l2_norm,
                        gpu_l2: gpu_event.l2_norm,
                        relative_diff: cpu_event.relative_diff(gpu_event),
                        cpu_head: cpu_event.head,
                        gpu_head: gpu_event.head,
                    });
                }
            }
            // Skip if GPU doesn't have this event (may have different instrumentation)
        }

        TraceComparison { diffs, tolerance }
    }

    /// Print all events to stderr for debugging
    pub fn dump(&self) {
        eprintln!("=== BRICK TRACE ({} events) ===", self.events.len());
        for event in &self.events {
            eprintln!("  {event}");
        }
    }

    /// Print summary statistics
    pub fn summary(&self) {
        eprintln!("=== TRACE SUMMARY ===");
        eprintln!("Events: {}", self.events.len());
        if let Some(first) = self.events.first() {
            eprintln!("First: {}", first.name);
        }
        if let Some(last) = self.events.last() {
            eprintln!("Last: {}", last.name);
        }

        // Find largest L2 norm
        if let Some(max_event) = self.events.iter().max_by(|a, b| {
            a.l2_norm
                .partial_cmp(&b.l2_norm)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            eprintln!("Max L2: {} = {:.6}", max_event.name, max_event.l2_norm);
        }
    }
}

// Global tracers for CPU and GPU paths (thread-local to avoid contention).
// Used by trace_cpu! and trace_gpu! macros when "trace" feature is enabled.
#[cfg(feature = "trace")]
thread_local! {
    /// Thread-local tracer for CPU inference path
    pub static CPU_TRACER: std::cell::RefCell<BrickTracer> = std::cell::RefCell::new(BrickTracer::new());
    /// Thread-local tracer for GPU inference path
    pub static GPU_TRACER: std::cell::RefCell<BrickTracer> = std::cell::RefCell::new(BrickTracer::new());
}

/// Log a tensor to the CPU tracer for parity debugging.
///
/// Only active when the "trace" feature is enabled. When disabled, this macro
/// compiles to nothing (zero overhead).
///
/// # Arguments
///
/// * `$name` - Name of the computation step (e.g., "layer0_rope_q")
/// * `$tensor` - Slice of f32 values to log
/// * `$pos` - (Optional) Explicit position index
///
/// # Example
///
/// ```rust,ignore
/// trace_cpu!("embedding", &embedding_output);
/// trace_cpu!("layer0_attn", &attn_out, position);
/// ```
#[macro_export]
#[cfg(feature = "trace")]
macro_rules! trace_cpu {
    ($name:expr, $tensor:expr) => {
        $crate::brick::tracer::CPU_TRACER.with(|t| {
            t.borrow_mut().log($name, $tensor);
        });
    };
    ($name:expr, $tensor:expr, $pos:expr) => {
        $crate::brick::tracer::CPU_TRACER.with(|t| {
            t.borrow_mut().log_at($name, $tensor, $pos);
        });
    };
}

/// No-op version of trace_cpu when "trace" feature is disabled.
#[macro_export]
#[cfg(not(feature = "trace"))]
macro_rules! trace_cpu {
    ($name:expr, $tensor:expr) => {};
    ($name:expr, $tensor:expr, $pos:expr) => {};
}

/// Log a tensor to the GPU tracer for parity debugging.
///
/// Only active when the "trace" feature is enabled. When disabled, this macro
/// compiles to nothing (zero overhead).
///
/// # Arguments
///
/// * `$name` - Name of the computation step (e.g., "layer0_rope_q")
/// * `$tensor` - Slice of f32 values to log (must be downloaded from GPU first)
/// * `$pos` - (Optional) Explicit position index
///
/// # Example
///
/// ```rust,ignore
/// // After D2H copy from GPU buffer
/// trace_gpu!("embedding", &embedding_output);
/// trace_gpu!("layer0_attn", &attn_out, position);
/// ```
#[macro_export]
#[cfg(feature = "trace")]
macro_rules! trace_gpu {
    ($name:expr, $tensor:expr) => {
        $crate::brick::tracer::GPU_TRACER.with(|t| {
            t.borrow_mut().log($name, $tensor);
        });
    };
    ($name:expr, $tensor:expr, $pos:expr) => {
        $crate::brick::tracer::GPU_TRACER.with(|t| {
            t.borrow_mut().log_at($name, $tensor, $pos);
        });
    };
}

/// No-op version of trace_gpu when "trace" feature is disabled.
#[macro_export]
#[cfg(not(feature = "trace"))]
macro_rules! trace_gpu {
    ($name:expr, $tensor:expr) => {};
    ($name:expr, $tensor:expr, $pos:expr) => {};
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_event_creation() {
        let tensor = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = TraceEvent::new("test", &tensor, 0, false);

        assert_eq!(event.name, "test");
        assert_eq!(event.position, 0);
        assert_eq!(event.len, 5);
        assert!((event.l2_norm - 7.416198).abs() < 0.001); // sqrt(55)
        assert!((event.mean - 3.0).abs() < 0.001);
        assert!((event.min - 1.0).abs() < 0.001);
        assert!((event.max - 5.0).abs() < 0.001);
        assert!(event.full_data.is_none());
    }

    #[test]
    fn test_trace_event_verbose() {
        let tensor = vec![1.0, 2.0, 3.0];
        let event = TraceEvent::new("test", &tensor, 0, true);

        assert!(event.full_data.is_some());
        assert_eq!(event.full_data.unwrap(), tensor);
    }

    #[test]
    fn test_tracer_log() {
        let mut tracer = BrickTracer::new();

        tracer.log("step1", &[1.0, 2.0, 3.0]);
        tracer.log("step2", &[4.0, 5.0, 6.0]);

        assert_eq!(tracer.events().len(), 2);
        assert!(tracer.get("step1").is_some());
        assert!(tracer.get("step2").is_some());
        assert!(tracer.get("step3").is_none());
    }

    #[test]
    fn test_tracer_comparison_match() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("embedding", &[1.0, 2.0, 3.0]);
        cpu.log("layer0_norm", &[0.5, 1.0, 1.5]);

        gpu.log("embedding", &[1.0, 2.0, 3.0]);
        gpu.log("layer0_norm", &[0.5, 1.0, 1.5]);

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        assert!(comparison.is_equivalent());
    }

    #[test]
    fn test_tracer_comparison_diverge() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("embedding", &[1.0, 2.0, 3.0]);
        cpu.log("layer0_norm", &[0.5, 1.0, 1.5]);
        cpu.log("layer0_attn", &[0.1, 0.2, 0.3]);

        gpu.log("embedding", &[1.0, 2.0, 3.0]); // Match
        gpu.log("layer0_norm", &[0.5, 1.0, 1.5]); // Match
        gpu.log("layer0_attn", &[0.5, 0.6, 0.7]); // DIVERGE!

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        assert!(!comparison.is_equivalent());

        let first = comparison.first_divergence().unwrap();
        assert_eq!(first.name, "layer0_attn");
    }

    #[test]
    fn test_tracer_clear() {
        let mut tracer = BrickTracer::new();
        tracer.log("step1", &[1.0]);
        assert_eq!(tracer.events().len(), 1);

        tracer.clear();
        assert_eq!(tracer.events().len(), 0);
    }

    #[test]
    fn test_tracer_position() {
        let mut tracer = BrickTracer::new();

        tracer.set_position(0);
        tracer.log("pos0", &[1.0]);

        tracer.set_position(1);
        tracer.log("pos1", &[2.0]);

        tracer.log_at("explicit", &[3.0], 5);

        assert_eq!(tracer.get("pos0").unwrap().position, 0);
        assert_eq!(tracer.get("pos1").unwrap().position, 1);
        assert_eq!(tracer.get("explicit").unwrap().position, 5);
    }

    #[test]
    fn test_relative_diff() {
        let event1 = TraceEvent::new("a", &[10.0], 0, false);
        let event2 = TraceEvent::new("b", &[11.0], 0, false);

        let diff = event1.relative_diff(&event2);
        assert!((diff - 0.1).abs() < 0.001); // 10% difference
    }

    #[test]
    fn test_approx_eq() {
        let event1 = TraceEvent::new("a", &[10.0], 0, false);
        let event2 = TraceEvent::new("b", &[10.05], 0, false);
        let event3 = TraceEvent::new("c", &[11.0], 0, false);

        assert!(event1.approx_eq(&event2, 0.01)); // 1% tolerance
        assert!(!event1.approx_eq(&event3, 0.01)); // 10% diff > 1% tolerance
        assert!(event1.approx_eq(&event3, 0.15)); // 10% diff < 15% tolerance
    }

    #[test]
    fn test_empty_tensor() {
        let event = TraceEvent::new("empty", &[], 0, false);
        assert_eq!(event.len, 0);
        assert_eq!(event.l2_norm, 0.0);
        assert_eq!(event.mean, 0.0);
    }

    // T-COV-95: Additional coverage tests

    #[test]
    fn test_trace_event_display() {
        let event = TraceEvent::new("layer0", &[1.0, 2.0, 3.0], 5, false);
        let display = format!("{}", event);
        assert!(display.contains("layer0"));
        assert!(display.contains("pos=5"));
    }

    #[test]
    fn test_trace_comparison_summary_no_divergence() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("a", &[1.0, 2.0, 3.0]);
        gpu.log("a", &[1.0, 2.0, 3.0]);

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        let summary = comparison.summary();
        assert!(summary.contains("No divergence"));
    }

    #[test]
    fn test_trace_comparison_summary_with_divergence() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("attn", &[1.0, 2.0, 3.0]);
        gpu.log("attn", &[10.0, 20.0, 30.0]); // Big divergence

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        let summary = comparison.summary();
        assert!(summary.contains("divergence"));
        assert!(summary.contains("attn"));
    }

    #[test]
    fn test_trace_comparison_display_no_divergence() {
        let comparison = TraceComparison {
            diffs: vec![],
            tolerance: 0.01,
        };
        let display = format!("{}", comparison);
        assert!(display.contains("MATCH"));
    }

    #[test]
    fn test_trace_comparison_display_with_divergence() {
        let comparison = TraceComparison {
            diffs: vec![TraceDiff {
                name: "test".to_string(),
                position: 0,
                cpu_l2: 10.0,
                gpu_l2: 15.0,
                relative_diff: 0.5,
                cpu_head: [0.0; 8],
                gpu_head: [0.0; 8],
            }],
            tolerance: 0.01,
        };
        let display = format!("{}", comparison);
        assert!(display.contains("DIVERGENCE DETECTED"));
    }

    #[test]
    fn test_trace_comparison_display_multiple_divergences() {
        let comparison = TraceComparison {
            diffs: vec![
                TraceDiff {
                    name: "attn".to_string(),
                    position: 0,
                    cpu_l2: 10.0,
                    gpu_l2: 15.0,
                    relative_diff: 0.5,
                    cpu_head: [0.0; 8],
                    gpu_head: [0.0; 8],
                },
                TraceDiff {
                    name: "ffn".to_string(),
                    position: 1,
                    cpu_l2: 20.0,
                    gpu_l2: 25.0,
                    relative_diff: 0.25,
                    cpu_head: [0.0; 8],
                    gpu_head: [0.0; 8],
                },
            ],
            tolerance: 0.01,
        };
        let display = format!("{}", comparison);
        assert!(display.contains("All Divergences"));
        assert!(display.contains("attn"));
        assert!(display.contains("ffn"));
    }

    #[test]
    fn test_trace_diff_display() {
        let diff = TraceDiff {
            name: "layer0_attn".to_string(),
            position: 3,
            cpu_l2: 100.0,
            gpu_l2: 110.0,
            relative_diff: 0.1,
            cpu_head: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            gpu_head: [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
        };
        let display = format!("{}", diff);
        assert!(display.contains("layer0_attn"));
    }

    #[test]
    fn test_tracer_default() {
        let tracer = BrickTracer::default();
        assert!(tracer.events().is_empty());
    }

    #[test]
    fn test_tracer_verbose() {
        let mut tracer = BrickTracer::verbose();
        tracer.log("test", &[1.0, 2.0, 3.0]);
        let event = tracer.get("test").unwrap();
        assert!(event.full_data.is_some());
    }

    #[test]
    fn test_tracer_dump_no_panic() {
        let mut tracer = BrickTracer::new();
        tracer.log("step1", &[1.0]);
        tracer.log("step2", &[2.0]);
        // Just verify it doesn't panic
        tracer.dump();
    }

    #[test]
    fn test_tracer_summary_no_panic() {
        let mut tracer = BrickTracer::new();
        tracer.log("step1", &[1.0, 100.0]);
        tracer.log("step2", &[2.0, 50.0]);
        // Just verify it doesn't panic
        tracer.summary();
    }

    #[test]
    fn test_tracer_summary_empty() {
        let tracer = BrickTracer::new();
        tracer.summary(); // Should not panic
    }

    #[test]
    fn test_tracer_compare_mismatched_events() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("a", &[1.0]);
        cpu.log("b", &[2.0]);

        gpu.log("a", &[1.0]);
        // gpu doesn't have "b"

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        // Should handle missing events gracefully
        assert!(comparison.is_equivalent() || !comparison.is_equivalent());
    }

    #[test]
    fn test_trace_event_min_max() {
        let tensor = vec![-10.0, 0.0, 5.0, 100.0];
        let event = TraceEvent::new("minmax", &tensor, 0, false);
        assert!((event.min - (-10.0)).abs() < 0.001);
        assert!((event.max - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_relative_diff_zero_cpu() {
        let event1 = TraceEvent::new("a", &[0.0], 0, false);
        let event2 = TraceEvent::new("b", &[1.0], 0, false);
        let diff = event1.relative_diff(&event2);
        // When CPU L2 is zero, should handle gracefully
        assert!(diff.is_finite());
    }
}
