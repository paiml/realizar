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

include!("tracer_impl.rs");
include!("tracer_part_03.rs");
