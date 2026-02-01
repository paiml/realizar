//! Backend benchmark matrix for comparing compute implementations
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - ComputeBackendType enum (CPU, wgpu, CUDA)
//! - BenchmarkMatrix for cross-backend comparison
//! - MatrixBenchmarkConfig and related types

#![allow(clippy::cast_precision_loss)]

use std::fmt::Write;

use serde::{Deserialize, Serialize};

use super::{chrono_timestamp, compute_cv, percentile, HardwareSpec, RuntimeType};

// ============================================================================
// Backend Benchmark Matrix (per Hoefler & Belli SC'15)
// ============================================================================

/// Compute backend type for benchmark matrix
///
/// Represents the different compute backends that can be benchmarked:
/// - CPU: Scalar/SIMD operations via trueno CPU backend
/// - Wgpu: Cross-platform GPU via trueno wgpu backend
/// - Cuda: NVIDIA GPU via trueno-gpu PTX execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeBackendType {
    /// CPU backend (scalar/SIMD via trueno)
    Cpu,
    /// wgpu GPU backend (cross-platform via trueno)
    Wgpu,
    /// CUDA GPU backend (NVIDIA via trueno-gpu)
    Cuda,
}

impl std::fmt::Display for ComputeBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Wgpu => write!(f, "wgpu"),
            Self::Cuda => write!(f, "cuda"),
        }
    }
}

impl ComputeBackendType {
    /// Parse from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "wgpu" | "gpu" => Some(Self::Wgpu),
            "cuda" | "nvidia" => Some(Self::Cuda),
            _ => None,
        }
    }

    /// All available backend types
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![Self::Cpu, Self::Wgpu, Self::Cuda]
    }
}

/// Single entry in the benchmark matrix
///
/// Represents results for one (runtime, backend) combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixBenchmarkEntry {
    /// Runtime type (realizar, llama-cpp, ollama, vllm)
    pub runtime: RuntimeType,
    /// Compute backend (cpu, wgpu, cuda)
    pub backend: ComputeBackendType,
    /// Model name/identifier
    pub model: String,
    /// Whether this configuration is available
    pub available: bool,
    /// p50 latency in milliseconds
    pub p50_latency_ms: f64,
    /// p99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Throughput in tokens per second
    pub throughput_tps: f64,
    /// Cold start time in milliseconds
    pub cold_start_ms: f64,
    /// Number of samples collected
    pub samples: usize,
    /// Final CV at stop
    pub cv_at_stop: f64,
    /// Additional notes (e.g., "GPU layers: 99")
    pub notes: String,
}

impl Default for MatrixBenchmarkEntry {
    fn default() -> Self {
        Self {
            runtime: RuntimeType::Realizar,
            backend: ComputeBackendType::Cpu,
            model: String::new(),
            available: false,
            p50_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            throughput_tps: 0.0,
            cold_start_ms: 0.0,
            samples: 0,
            cv_at_stop: 0.0,
            notes: String::new(),
        }
    }
}

impl MatrixBenchmarkEntry {
    /// Create a new unavailable entry (placeholder)
    #[must_use]
    pub fn unavailable(runtime: RuntimeType, backend: ComputeBackendType) -> Self {
        Self {
            runtime,
            backend,
            available: false,
            notes: "Backend not available".to_string(),
            ..Default::default()
        }
    }

    /// Create entry from raw latency samples
    #[must_use]
    pub fn from_samples(
        runtime: RuntimeType,
        backend: ComputeBackendType,
        model: &str,
        latencies_ms: &[f64],
        throughputs_tps: &[f64],
        cold_start_ms: f64,
    ) -> Self {
        let samples = latencies_ms.len();
        if samples == 0 {
            return Self::unavailable(runtime, backend);
        }

        let p50_latency = percentile(latencies_ms, 50.0);
        let p99_latency = percentile(latencies_ms, 99.0);
        let throughput = if throughputs_tps.is_empty() {
            0.0
        } else {
            throughputs_tps.iter().sum::<f64>() / throughputs_tps.len() as f64
        };
        let cv = compute_cv(latencies_ms);

        Self {
            runtime,
            backend,
            model: model.to_string(),
            available: true,
            p50_latency_ms: p50_latency,
            p99_latency_ms: p99_latency,
            throughput_tps: throughput,
            cold_start_ms,
            samples,
            cv_at_stop: cv,
            notes: String::new(),
        }
    }

    /// Add notes to the entry
    #[must_use]
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = notes.to_string();
        self
    }
}

/// Complete benchmark matrix comparing runtimes across backends
///
/// Per Hoefler & Belli SC'15, this matrix enables:
/// - Reproducible comparisons across configurations
/// - Statistical validity via CV-based stopping
/// - Clear identification of performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMatrix {
    /// Schema version
    pub version: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Model used for benchmarking
    pub model: String,
    /// Hardware specification
    pub hardware: HardwareSpec,
    /// Benchmark methodology
    pub methodology: String,
    /// CV threshold used
    pub cv_threshold: f64,
    /// Matrix entries indexed by (runtime, backend)
    pub entries: Vec<MatrixBenchmarkEntry>,
}

impl BenchmarkMatrix {
    /// Create a new empty matrix
    #[must_use]
    pub fn new(model: &str, hardware: HardwareSpec) -> Self {
        Self {
            version: "1.1".to_string(),
            timestamp: chrono_timestamp(),
            model: model.to_string(),
            hardware,
            methodology: "CV-based stopping (Hoefler & Belli SC'15)".to_string(),
            cv_threshold: 0.05,
            entries: Vec::new(),
        }
    }

    /// Add an entry to the matrix
    pub fn add_entry(&mut self, entry: MatrixBenchmarkEntry) {
        // Remove existing entry for same (runtime, backend) if present
        self.entries
            .retain(|e| e.runtime != entry.runtime || e.backend != entry.backend);
        self.entries.push(entry);
    }

    /// Get entry for specific (runtime, backend) combination
    #[must_use]
    pub fn get_entry(
        &self,
        runtime: RuntimeType,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .find(|e| e.runtime == runtime && e.backend == backend)
    }

    /// Get all entries for a specific runtime
    #[must_use]
    pub fn entries_for_runtime(&self, runtime: RuntimeType) -> Vec<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| e.runtime == runtime)
            .collect()
    }

    /// Get all entries for a specific backend
    #[must_use]
    pub fn entries_for_backend(&self, backend: ComputeBackendType) -> Vec<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| e.backend == backend)
            .collect()
    }

    /// Find the fastest runtime for a given backend (by p50 latency)
    #[must_use]
    pub fn fastest_for_backend(
        &self,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries_for_backend(backend)
            .into_iter()
            .filter(|e| e.available)
            .min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            })
    }

    /// Find the highest throughput runtime for a given backend
    #[must_use]
    pub fn highest_throughput_for_backend(
        &self,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries_for_backend(backend)
            .into_iter()
            .filter(|e| e.available)
            .max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            })
    }

    /// Generate markdown table for README
    #[must_use]
    pub fn to_markdown_table(&self) -> String {
        let mut table = String::new();

        // Header
        table.push_str("| Runtime | Backend | p50 Latency | p99 Latency | Throughput | Cold Start | Samples | CV |\n");
        table.push_str("|---------|---------|-------------|-------------|------------|------------|---------|----|\n");

        // Sort entries by runtime, then backend
        let mut sorted_entries = self.entries.clone();
        sorted_entries.sort_by(|a, b| {
            let runtime_cmp = format!("{:?}", a.runtime).cmp(&format!("{:?}", b.runtime));
            if runtime_cmp == std::cmp::Ordering::Equal {
                format!("{}", a.backend).cmp(&format!("{}", b.backend))
            } else {
                runtime_cmp
            }
        });

        for entry in &sorted_entries {
            if entry.available {
                let _ = writeln!(
                    table,
                    "| **{}** | {} | {:.1}ms | {:.1}ms | {:.1} tok/s | {:.0}ms | {} | {:.3} |",
                    format!("{:?}", entry.runtime).to_lowercase(),
                    entry.backend,
                    entry.p50_latency_ms,
                    entry.p99_latency_ms,
                    entry.throughput_tps,
                    entry.cold_start_ms,
                    entry.samples,
                    entry.cv_at_stop,
                );
            } else {
                let _ = writeln!(
                    table,
                    "| {} | {} | - | - | - | - | - | - |",
                    format!("{:?}", entry.runtime).to_lowercase(),
                    entry.backend,
                );
            }
        }

        table
    }

    /// Serialize to JSON
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    ///
    /// # Errors
    ///
    /// Returns error if JSON is invalid.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Matrix benchmark runner configuration
#[derive(Debug, Clone)]
pub struct MatrixBenchmarkConfig {
    /// Runtimes to benchmark
    pub runtimes: Vec<RuntimeType>,
    /// Backends to benchmark
    pub backends: Vec<ComputeBackendType>,
    /// Model path
    pub model_path: String,
    /// Prompt for benchmarking
    pub prompt: String,
    /// Max tokens to generate
    pub max_tokens: usize,
    /// CV threshold for stopping
    pub cv_threshold: f64,
    /// Minimum samples
    pub min_samples: usize,
    /// Maximum samples (failsafe)
    pub max_samples: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for MatrixBenchmarkConfig {
    fn default() -> Self {
        Self {
            runtimes: vec![
                RuntimeType::Realizar,
                RuntimeType::LlamaCpp,
                RuntimeType::Ollama,
            ],
            backends: vec![ComputeBackendType::Cpu, ComputeBackendType::Wgpu],
            model_path: String::new(),
            prompt: "Explain machine learning in one sentence.".to_string(),
            max_tokens: 50,
            cv_threshold: 0.05,
            min_samples: 30,
            max_samples: 200,
            warmup_iterations: 5,
        }
    }
}

/// Summary statistics for a single matrix column (backend)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSummary {
    /// Backend type
    pub backend: ComputeBackendType,
    /// Number of available runtimes
    pub available_runtimes: usize,
    /// Fastest runtime (by p50 latency)
    pub fastest_runtime: Option<String>,
    /// Fastest p50 latency
    pub fastest_p50_ms: f64,
    /// Highest throughput runtime
    pub highest_throughput_runtime: Option<String>,
    /// Highest throughput (tok/s)
    pub highest_throughput_tps: f64,
}

/// Summary of the entire benchmark matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixSummary {
    /// Total entries in matrix
    pub total_entries: usize,
    /// Number of available entries
    pub available_entries: usize,
    /// Per-backend summaries
    pub backend_summaries: Vec<BackendSummary>,
    /// Overall fastest (runtime, backend) combination
    pub overall_fastest: Option<(String, String)>,
    /// Overall highest throughput (runtime, backend)
    pub overall_highest_throughput: Option<(String, String)>,
}

impl BenchmarkMatrix {
    /// Generate summary statistics
    #[must_use]
    pub fn summary(&self) -> MatrixSummary {
        let total_entries = self.entries.len();
        let available_entries = self.entries.iter().filter(|e| e.available).count();

        let mut backend_summaries = Vec::new();
        for backend in ComputeBackendType::all() {
            let entries: Vec<_> = self.entries_for_backend(backend);
            let available: Vec<_> = entries.iter().filter(|e| e.available).collect();

            let fastest = available.iter().min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            });
            let highest_tp = available.iter().max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            });

            backend_summaries.push(BackendSummary {
                backend,
                available_runtimes: available.len(),
                fastest_runtime: fastest.map(|e| format!("{:?}", e.runtime).to_lowercase()),
                fastest_p50_ms: fastest.map_or(0.0, |e| e.p50_latency_ms),
                highest_throughput_runtime: highest_tp
                    .map(|e| format!("{:?}", e.runtime).to_lowercase()),
                highest_throughput_tps: highest_tp.map_or(0.0, |e| e.throughput_tps),
            });
        }

        let available = self.entries.iter().filter(|e| e.available);
        let overall_fastest = available
            .clone()
            .min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            })
            .map(|e| {
                (
                    format!("{:?}", e.runtime).to_lowercase(),
                    e.backend.to_string(),
                )
            });
        let overall_highest_throughput = available
            .max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            })
            .map(|e| {
                (
                    format!("{:?}", e.runtime).to_lowercase(),
                    e.backend.to_string(),
                )
            });

        MatrixSummary {
            total_entries,
            available_entries,
            backend_summaries,
            overall_fastest,
            overall_highest_throughput,
        }
    }
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ComputeBackendType Tests
    // =========================================================================

    #[test]
    fn test_compute_backend_type_display() {
        assert_eq!(format!("{}", ComputeBackendType::Cpu), "cpu");
        assert_eq!(format!("{}", ComputeBackendType::Wgpu), "wgpu");
        assert_eq!(format!("{}", ComputeBackendType::Cuda), "cuda");
    }

    #[test]
    fn test_compute_backend_type_parse() {
        assert_eq!(
            ComputeBackendType::parse("cpu"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("wgpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("gpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("cuda"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(
            ComputeBackendType::parse("nvidia"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(
            ComputeBackendType::parse("CPU"),
            Some(ComputeBackendType::Cpu)
        ); // case-insensitive
        assert_eq!(ComputeBackendType::parse("unknown"), None);
    }

    #[test]
    fn test_compute_backend_type_all() {
        let all = ComputeBackendType::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&ComputeBackendType::Cpu));
        assert!(all.contains(&ComputeBackendType::Wgpu));
        assert!(all.contains(&ComputeBackendType::Cuda));
    }

    #[test]
    fn test_compute_backend_type_clone_eq() {
        let backend = ComputeBackendType::Cuda;
        assert_eq!(backend, backend.clone());
    }

    #[test]
    fn test_compute_backend_type_serialize() {
        let json = serde_json::to_string(&ComputeBackendType::Wgpu).unwrap();
        assert!(json.contains("Wgpu"));
    }

    // =========================================================================
    // MatrixBenchmarkEntry Tests
    // =========================================================================

    #[test]
    fn test_matrix_benchmark_entry_default() {
        let entry = MatrixBenchmarkEntry::default();
        assert_eq!(entry.runtime, RuntimeType::Realizar);
        assert_eq!(entry.backend, ComputeBackendType::Cpu);
        assert!(!entry.available);
        assert_eq!(entry.samples, 0);
    }

    #[test]
    fn test_matrix_benchmark_entry_unavailable() {
        let entry = MatrixBenchmarkEntry::unavailable(RuntimeType::Vllm, ComputeBackendType::Cuda);
        assert_eq!(entry.runtime, RuntimeType::Vllm);
        assert_eq!(entry.backend, ComputeBackendType::Cuda);
        assert!(!entry.available);
        assert!(entry.notes.contains("not available"));
    }

    #[test]
    fn test_matrix_benchmark_entry_from_samples() {
        let latencies = vec![10.0, 12.0, 11.0, 13.0, 9.0];
        let throughputs = vec![100.0, 90.0, 95.0, 85.0, 105.0];
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "llama-7b",
            &latencies,
            &throughputs,
            50.0,
        );

        assert_eq!(entry.runtime, RuntimeType::LlamaCpp);
        assert_eq!(entry.backend, ComputeBackendType::Wgpu);
        assert_eq!(entry.model, "llama-7b");
        assert!(entry.available);
        assert_eq!(entry.samples, 5);
        assert!((entry.cold_start_ms - 50.0).abs() < 0.01);
        // p50 of [9, 10, 11, 12, 13] should be around 11
        assert!(entry.p50_latency_ms > 0.0);
        // Average throughput should be 95
        assert!((entry.throughput_tps - 95.0).abs() < 0.01);
    }

    #[test]
    fn test_matrix_benchmark_entry_from_empty_samples() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Ollama,
            ComputeBackendType::Cpu,
            "model",
            &[],
            &[],
            0.0,
        );
        assert!(!entry.available);
    }

    #[test]
    fn test_matrix_benchmark_entry_from_samples_empty_throughput() {
        let latencies = vec![10.0, 12.0];
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model",
            &latencies,
            &[],
            0.0,
        );
        assert!(entry.available);
        assert_eq!(entry.throughput_tps, 0.0);
    }

    #[test]
    fn test_matrix_benchmark_entry_with_notes() {
        let entry = MatrixBenchmarkEntry::default().with_notes("GPU layers: 32");
        assert_eq!(entry.notes, "GPU layers: 32");
    }

    #[test]
    fn test_matrix_benchmark_entry_serialize() {
        let entry = MatrixBenchmarkEntry {
            runtime: RuntimeType::LlamaCpp,
            backend: ComputeBackendType::Cuda,
            model: "phi-2".to_string(),
            available: true,
            p50_latency_ms: 25.5,
            p99_latency_ms: 45.0,
            throughput_tps: 200.0,
            cold_start_ms: 100.0,
            samples: 30,
            cv_at_stop: 0.03,
            notes: "test".to_string(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("phi-2"));
        assert!(json.contains("200"));
    }

    // =========================================================================
    // BenchmarkMatrix Tests
    // =========================================================================

    fn make_hardware_spec() -> HardwareSpec {
        HardwareSpec {
            cpu: "Intel i7".to_string(),
            gpu: Some("RTX 4090".to_string()),
            memory_gb: 32,
            storage: "NVMe SSD".to_string(),
        }
    }

    #[test]
    fn test_benchmark_matrix_new() {
        let hw = make_hardware_spec();
        let matrix = BenchmarkMatrix::new("llama-7b", hw);

        assert_eq!(matrix.model, "llama-7b");
        assert_eq!(matrix.version, "1.1");
        assert!((matrix.cv_threshold - 0.05).abs() < 0.001);
        assert!(matrix.entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_add_entry() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cpu);
        matrix.add_entry(entry);

        assert_eq!(matrix.entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_add_entry_replaces_existing() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let entry1 =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cpu);
        matrix.add_entry(entry1);

        let entry2 = MatrixBenchmarkEntry {
            runtime: RuntimeType::Realizar,
            backend: ComputeBackendType::Cpu,
            available: true,
            p50_latency_ms: 10.0,
            ..Default::default()
        };
        matrix.add_entry(entry2);

        assert_eq!(matrix.entries.len(), 1);
        assert!(matrix.entries[0].available);
    }

    #[test]
    fn test_benchmark_matrix_get_entry() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let entry = MatrixBenchmarkEntry::unavailable(RuntimeType::Vllm, ComputeBackendType::Cuda);
        matrix.add_entry(entry);

        let found = matrix.get_entry(RuntimeType::Vllm, ComputeBackendType::Cuda);
        assert!(found.is_some());

        let not_found = matrix.get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_benchmark_matrix_entries_for_runtime() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Vllm,
            ComputeBackendType::Cuda,
        ));

        let realizar_entries = matrix.entries_for_runtime(RuntimeType::Realizar);
        assert_eq!(realizar_entries.len(), 2);

        let vllm_entries = matrix.entries_for_runtime(RuntimeType::Vllm);
        assert_eq!(vllm_entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_entries_for_backend() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cuda,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
        ));

        let cuda_entries = matrix.entries_for_backend(ComputeBackendType::Cuda);
        assert_eq!(cuda_entries.len(), 2);

        let cpu_entries = matrix.entries_for_backend(ComputeBackendType::Cpu);
        assert_eq!(cpu_entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let mut entry1 = MatrixBenchmarkEntry::default();
        entry1.runtime = RuntimeType::Realizar;
        entry1.backend = ComputeBackendType::Cpu;
        entry1.available = true;
        entry1.p50_latency_ms = 20.0;
        matrix.add_entry(entry1);

        let mut entry2 = MatrixBenchmarkEntry::default();
        entry2.runtime = RuntimeType::LlamaCpp;
        entry2.backend = ComputeBackendType::Cpu;
        entry2.available = true;
        entry2.p50_latency_ms = 15.0; // faster
        matrix.add_entry(entry2);

        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cpu);
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend_none_available() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cuda);
        assert!(fastest.is_none());
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_for_backend() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let mut entry1 = MatrixBenchmarkEntry::default();
        entry1.runtime = RuntimeType::Realizar;
        entry1.backend = ComputeBackendType::Wgpu;
        entry1.available = true;
        entry1.throughput_tps = 100.0;
        matrix.add_entry(entry1);

        let mut entry2 = MatrixBenchmarkEntry::default();
        entry2.runtime = RuntimeType::Ollama;
        entry2.backend = ComputeBackendType::Wgpu;
        entry2.available = true;
        entry2.throughput_tps = 150.0; // higher
        matrix.add_entry(entry2);

        let highest = matrix.highest_throughput_for_backend(ComputeBackendType::Wgpu);
        assert!(highest.is_some());
        assert_eq!(highest.unwrap().runtime, RuntimeType::Ollama);
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_table() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let mut entry = MatrixBenchmarkEntry::default();
        entry.runtime = RuntimeType::Realizar;
        entry.backend = ComputeBackendType::Cpu;
        entry.available = true;
        entry.p50_latency_ms = 10.5;
        entry.p99_latency_ms = 25.0;
        entry.throughput_tps = 95.5;
        entry.cold_start_ms = 50.0;
        entry.samples = 30;
        entry.cv_at_stop = 0.045;
        matrix.add_entry(entry);

        let md = matrix.to_markdown_table();
        assert!(md.contains("| Runtime |"));
        assert!(md.contains("realizar"));
        assert!(md.contains("cpu"));
        assert!(md.contains("10.5ms"));
        assert!(md.contains("95.5 tok/s"));
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_table_unavailable() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Vllm,
            ComputeBackendType::Cuda,
        ));

        let md = matrix.to_markdown_table();
        assert!(md.contains("vllm"));
        assert!(md.contains("| - | - | - |"));
    }

    #[test]
    fn test_benchmark_matrix_to_json() {
        let hw = make_hardware_spec();
        let matrix = BenchmarkMatrix::new("llama-7b", hw);

        let json = matrix.to_json().unwrap();
        assert!(json.contains("llama-7b"));
        assert!(json.contains("version"));
    }

    #[test]
    fn test_benchmark_matrix_from_json() {
        let hw = make_hardware_spec();
        let matrix = BenchmarkMatrix::new("phi-2", hw);
        let json = matrix.to_json().unwrap();

        let parsed = BenchmarkMatrix::from_json(&json).unwrap();
        assert_eq!(parsed.model, "phi-2");
    }

    // =========================================================================
    // MatrixBenchmarkConfig Tests
    // =========================================================================

    #[test]
    fn test_matrix_benchmark_config_default() {
        let config = MatrixBenchmarkConfig::default();
        assert_eq!(config.runtimes.len(), 3);
        assert!(config.runtimes.contains(&RuntimeType::Realizar));
        assert_eq!(config.backends.len(), 2);
        assert_eq!(config.max_tokens, 50);
        assert_eq!(config.min_samples, 30);
        assert_eq!(config.max_samples, 200);
    }

    #[test]
    fn test_matrix_benchmark_config_debug() {
        let config = MatrixBenchmarkConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("MatrixBenchmarkConfig"));
    }

    // =========================================================================
    // BackendSummary Tests
    // =========================================================================

    #[test]
    fn test_backend_summary_serialize() {
        let summary = BackendSummary {
            backend: ComputeBackendType::Cuda,
            available_runtimes: 2,
            fastest_runtime: Some("realizar".to_string()),
            fastest_p50_ms: 15.0,
            highest_throughput_runtime: Some("llama-cpp".to_string()),
            highest_throughput_tps: 200.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("Cuda")); // Serializes as enum variant name
        assert!(json.contains("realizar"));
    }

    // =========================================================================
    // MatrixSummary Tests
    // =========================================================================

    #[test]
    fn test_benchmark_matrix_summary() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let mut entry1 = MatrixBenchmarkEntry::default();
        entry1.runtime = RuntimeType::Realizar;
        entry1.backend = ComputeBackendType::Cpu;
        entry1.available = true;
        entry1.p50_latency_ms = 20.0;
        entry1.throughput_tps = 100.0;
        matrix.add_entry(entry1);

        let mut entry2 = MatrixBenchmarkEntry::default();
        entry2.runtime = RuntimeType::LlamaCpp;
        entry2.backend = ComputeBackendType::Cpu;
        entry2.available = true;
        entry2.p50_latency_ms = 10.0;
        entry2.throughput_tps = 150.0;
        matrix.add_entry(entry2);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Vllm,
            ComputeBackendType::Cuda,
        ));

        let summary = matrix.summary();
        assert_eq!(summary.total_entries, 3);
        assert_eq!(summary.available_entries, 2);

        // Overall fastest should be llama-cpp (10ms)
        assert!(summary.overall_fastest.is_some());
        assert_eq!(summary.overall_fastest.as_ref().unwrap().0, "llamacpp");

        // Overall highest throughput should be llama-cpp (150 tok/s)
        assert!(summary.overall_highest_throughput.is_some());
        assert_eq!(
            summary.overall_highest_throughput.as_ref().unwrap().0,
            "llamacpp"
        );
    }

    #[test]
    fn test_matrix_summary_serialize() {
        let summary = MatrixSummary {
            total_entries: 5,
            available_entries: 3,
            backend_summaries: vec![],
            overall_fastest: Some(("realizar".to_string(), "cuda".to_string())),
            overall_highest_throughput: None,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("total_entries"));
        assert!(json.contains("realizar"));
    }
}
