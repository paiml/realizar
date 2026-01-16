//! Benchmark Visualization Module (PAR-040)
//!
//! Creates 2×3 grid visualizations for inference benchmark comparisons
//! and generates profiling logs suitable for chat paste debugging.
//!
//! ## Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │              GGUF Inference Comparison (tok/s GPU)                  │
//! ├─────────────────────┬─────────────────────┬─────────────────────────┤
//! │   APR serve GGUF    │      Ollama         │      llama.cpp          │
//! ├─────────────────────┴─────────────────────┴─────────────────────────┤
//! │              APR Server Format Comparison (tok/s GPU)               │
//! ├─────────────────────┬─────────────────────┬─────────────────────────┤
//! │   APR serve .apr    │  APR serve GGUF     │ Ollama / llama.cpp      │
//! └─────────────────────┴─────────────────────┴─────────────────────────┘
//! ```

use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};

// ============================================================================
// Benchmark Result Types
// ============================================================================

/// Single benchmark measurement
#[derive(Debug, Clone)]
pub struct BenchMeasurement {
    /// Engine name (APR, Ollama, llama.cpp)
    pub engine: String,
    /// Format (GGUF, APR)
    pub format: String,
    /// Throughput in tokens/second
    pub tokens_per_sec: f64,
    /// Time to first token in milliseconds
    pub ttft_ms: f64,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Total duration
    pub duration: Duration,
    /// GPU utilization percentage (if available)
    pub gpu_util: Option<f64>,
    /// GPU memory used in MB (if available)
    pub gpu_mem_mb: Option<f64>,
}

impl BenchMeasurement {
    /// Create a new benchmark measurement
    pub fn new(engine: &str, format: &str) -> Self {
        Self {
            engine: engine.to_string(),
            format: format.to_string(),
            tokens_per_sec: 0.0,
            ttft_ms: 0.0,
            tokens_generated: 0,
            duration: Duration::ZERO,
            gpu_util: None,
            gpu_mem_mb: None,
        }
    }

    /// Set throughput
    #[must_use]
    pub fn with_throughput(mut self, tps: f64) -> Self {
        self.tokens_per_sec = tps;
        self
    }

    /// Set TTFT
    #[must_use]
    pub fn with_ttft(mut self, ttft_ms: f64) -> Self {
        self.ttft_ms = ttft_ms;
        self
    }

    /// Set tokens generated
    #[must_use]
    pub fn with_tokens(mut self, count: usize, duration: Duration) -> Self {
        self.tokens_generated = count;
        self.duration = duration;
        if duration.as_secs_f64() > 0.0 {
            self.tokens_per_sec = count as f64 / duration.as_secs_f64();
        }
        self
    }

    /// Set GPU metrics
    #[must_use]
    pub fn with_gpu(mut self, util: f64, mem_mb: f64) -> Self {
        self.gpu_util = Some(util);
        self.gpu_mem_mb = Some(mem_mb);
        self
    }
}

/// Profiling hotspot for debugging
#[derive(Debug, Clone)]
pub struct ProfilingHotspot {
    /// Component name
    pub component: String,
    /// Time spent
    pub time: Duration,
    /// Percentage of total
    pub percentage: f64,
    /// Call count
    pub call_count: u64,
    /// Average time per call
    pub avg_per_call: Duration,
    /// Explanation/recommendation
    pub explanation: String,
    /// Is this expected for inference?
    pub is_expected: bool,
}

impl ProfilingHotspot {
    /// Format as single-line report
    pub fn to_line(&self) -> String {
        let marker = if self.is_expected { "✓" } else { "⚠" };
        format!(
            "{} {:20} {:>6.1}% {:>8.2}ms ({:>6} calls, {:>6.2}µs/call)",
            marker,
            self.component,
            self.percentage,
            self.time.as_secs_f64() * 1000.0,
            self.call_count,
            self.avg_per_call.as_secs_f64() * 1_000_000.0
        )
    }
}

// ============================================================================
// Benchmark Grid (2×3)
// ============================================================================

/// 2×3 Benchmark comparison grid
#[derive(Debug, Clone, Default)]
pub struct BenchmarkGrid {
    /// Row 1, Col 1: APR server serving GGUF format
    pub gguf_apr: Option<BenchMeasurement>,
    /// Row 1, Col 2: Ollama serving GGUF format
    pub gguf_ollama: Option<BenchMeasurement>,
    /// Row 1, Col 3: llama.cpp serving GGUF format
    pub gguf_llamacpp: Option<BenchMeasurement>,

    /// Row 2, Col 1: APR server serving native .apr format
    pub apr_native: Option<BenchMeasurement>,
    /// Row 2, Col 2: APR server serving GGUF (for comparison)
    pub apr_gguf: Option<BenchMeasurement>,
    /// Row 2, Col 3: Baseline measurement (Ollama/llama.cpp)
    pub apr_baseline: Option<BenchMeasurement>,

    /// Profiling hotspots
    pub hotspots: Vec<ProfilingHotspot>,

    /// Model name
    pub model_name: String,
    /// Model parameters (e.g., "0.5B")
    pub model_params: String,
    /// Quantization type (e.g., "Q4_K_M")
    pub quantization: String,

    /// GPU name
    pub gpu_name: String,
    /// GPU VRAM in GB
    pub gpu_vram_gb: f64,
}

impl BenchmarkGrid {
    /// Create new benchmark grid
    pub fn new() -> Self {
        Self::default()
    }

    /// Set model info
    #[must_use]
    pub fn with_model(mut self, name: &str, params: &str, quant: &str) -> Self {
        self.model_name = name.to_string();
        self.model_params = params.to_string();
        self.quantization = quant.to_string();
        self
    }

    /// Set GPU info
    #[must_use]
    pub fn with_gpu(mut self, name: &str, vram_gb: f64) -> Self {
        self.gpu_name = name.to_string();
        self.gpu_vram_gb = vram_gb;
        self
    }

    /// Add GGUF row measurements
    pub fn set_gguf_row(
        &mut self,
        apr: BenchMeasurement,
        ollama: BenchMeasurement,
        llamacpp: BenchMeasurement,
    ) {
        self.gguf_apr = Some(apr);
        self.gguf_ollama = Some(ollama);
        self.gguf_llamacpp = Some(llamacpp);
    }

    /// Add APR row measurements
    pub fn set_apr_row(
        &mut self,
        native: BenchMeasurement,
        gguf: BenchMeasurement,
        baseline: BenchMeasurement,
    ) {
        self.apr_native = Some(native);
        self.apr_gguf = Some(gguf);
        self.apr_baseline = Some(baseline);
    }

    /// Add profiling hotspot
    pub fn add_hotspot(&mut self, hotspot: ProfilingHotspot) {
        self.hotspots.push(hotspot);
    }

    // ========================================================================
    // Terminal Visualization (ASCII)
    // ========================================================================

    /// Render as ASCII grid for terminal
    pub fn render_ascii(&self) -> String {
        let mut out = String::new();

        // Header
        writeln!(
            out,
            "╔═══════════════════════════════════════════════════════════════════════╗"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║           INFERENCE BENCHMARK COMPARISON (tok/s GPU)                  ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║  Model: {:30} Quant: {:10}         ║",
            truncate(&self.model_name, 30),
            truncate(&self.quantization, 10)
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║  GPU: {:35} VRAM: {:5.1}GB              ║",
            truncate(&self.gpu_name, 35),
            self.gpu_vram_gb
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════════════════════════════════════════════════════╣"
        )
        .expect("failed to write benchmark output");

        // Row 1: GGUF comparison
        writeln!(
            out,
            "║                    GGUF Format Inference                              ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════╦═══════════════════════╦═══════════════════════╣"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║    APR serve GGUF     ║       Ollama          ║      llama.cpp        ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════╬═══════════════════════╬═══════════════════════╣"
        )
        .expect("failed to write benchmark output");

        let gguf_apr_tps = self.gguf_apr.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let gguf_ollama_tps = self.gguf_ollama.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let gguf_llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, |m| m.tokens_per_sec);

        writeln!(
            out,
            "║  {:>8.1} tok/s      ║  {:>8.1} tok/s      ║  {:>8.1} tok/s      ║",
            gguf_apr_tps, gguf_ollama_tps, gguf_llamacpp_tps
        )
        .expect("failed to write benchmark output");

        // Bar visualization
        let max_tps = [gguf_apr_tps, gguf_ollama_tps, gguf_llamacpp_tps]
            .iter()
            .cloned()
            .fold(1.0, f64::max);

        writeln!(
            out,
            "║  {}  ║  {}  ║  {}  ║",
            render_bar(gguf_apr_tps, max_tps, 17),
            render_bar(gguf_ollama_tps, max_tps, 17),
            render_bar(gguf_llamacpp_tps, max_tps, 17)
        )
        .expect("failed to write benchmark output");

        // TTFT
        let gguf_apr_ttft = self.gguf_apr.as_ref().map_or(0.0, |m| m.ttft_ms);
        let gguf_ollama_ttft = self.gguf_ollama.as_ref().map_or(0.0, |m| m.ttft_ms);
        let gguf_llamacpp_ttft = self.gguf_llamacpp.as_ref().map_or(0.0, |m| m.ttft_ms);

        writeln!(
            out,
            "║  TTFT: {:>6.1}ms      ║  TTFT: {:>6.1}ms      ║  TTFT: {:>6.1}ms      ║",
            gguf_apr_ttft, gguf_ollama_ttft, gguf_llamacpp_ttft
        )
        .expect("failed to write benchmark output");

        // Row 2: APR server comparison
        writeln!(
            out,
            "╠═══════════════════════╩═══════════════════════╩═══════════════════════╣"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║                   APR Server Format Comparison                        ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════╦═══════════════════════╦═══════════════════════╣"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "║   APR serve .apr      ║   APR serve GGUF      ║  Ollama (baseline)    ║"
        )
        .expect("failed to write benchmark output");
        writeln!(
            out,
            "╠═══════════════════════╬═══════════════════════╬═══════════════════════╣"
        )
        .expect("failed to write benchmark output");

        let apr_native_tps = self.apr_native.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let apr_gguf_tps = self.apr_gguf.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let apr_baseline_tps = self.apr_baseline.as_ref().map_or(0.0, |m| m.tokens_per_sec);

        writeln!(
            out,
            "║  {:>8.1} tok/s      ║  {:>8.1} tok/s      ║  {:>8.1} tok/s      ║",
            apr_native_tps, apr_gguf_tps, apr_baseline_tps
        )
        .expect("failed to write benchmark output");

        let max_tps2 = [apr_native_tps, apr_gguf_tps, apr_baseline_tps]
            .iter()
            .cloned()
            .fold(1.0, f64::max);

        writeln!(
            out,
            "║  {}  ║  {}  ║  {}  ║",
            render_bar(apr_native_tps, max_tps2, 17),
            render_bar(apr_gguf_tps, max_tps2, 17),
            render_bar(apr_baseline_tps, max_tps2, 17)
        )
        .expect("failed to write benchmark output");

        // Speedup vs baseline
        let speedup_native = if apr_baseline_tps > 0.0 {
            apr_native_tps / apr_baseline_tps
        } else {
            0.0
        };
        let speedup_gguf = if apr_baseline_tps > 0.0 {
            apr_gguf_tps / apr_baseline_tps
        } else {
            0.0
        };

        writeln!(
            out,
            "║  vs Ollama: {:>5.2}x   ║  vs Ollama: {:>5.2}x   ║  (baseline)           ║",
            speedup_native, speedup_gguf
        )
        .expect("failed to write benchmark output");

        writeln!(
            out,
            "╚═══════════════════════╩═══════════════════════╩═══════════════════════╝"
        )
        .expect("failed to write benchmark output");

        out
    }

    // ========================================================================
    // Profiling Log for Chat Paste
    // ========================================================================

    /// Generate profiling log suitable for chat paste
    pub fn render_profiling_log(&self) -> String {
        let mut out = String::new();

        writeln!(out, "```").expect("failed to write benchmark output");
        writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "INFERENCE PROFILING REPORT").expect("failed to write benchmark output");
        writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        )
        .expect("failed to write benchmark output");
        writeln!(out).expect("failed to write benchmark output");

        // Model & Hardware
        writeln!(out, "MODEL: {} ({})", self.model_name, self.model_params)
            .expect("failed to write benchmark output");
        writeln!(out, "QUANT: {}", self.quantization).expect("failed to write benchmark output");
        writeln!(
            out,
            "GPU:   {} ({:.1}GB VRAM)",
            self.gpu_name, self.gpu_vram_gb
        )
        .expect("failed to write benchmark output");
        writeln!(out).expect("failed to write benchmark output");

        // Performance Summary
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "THROUGHPUT COMPARISON (tok/s)").expect("failed to write benchmark output");
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");

        if let Some(ref m) = self.gguf_apr {
            writeln!(
                out,
                "APR GGUF:      {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                m.tokens_per_sec, m.ttft_ms
            )
            .expect("failed to write benchmark output");
        }
        if let Some(ref m) = self.apr_native {
            writeln!(
                out,
                "APR .apr:      {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                m.tokens_per_sec, m.ttft_ms
            )
            .expect("failed to write benchmark output");
        }
        if let Some(ref m) = self.gguf_ollama {
            writeln!(
                out,
                "Ollama:        {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                m.tokens_per_sec, m.ttft_ms
            )
            .expect("failed to write benchmark output");
        }
        if let Some(ref m) = self.gguf_llamacpp {
            writeln!(
                out,
                "llama.cpp:     {:>8.1} tok/s  (TTFT: {:>6.1}ms)",
                m.tokens_per_sec, m.ttft_ms
            )
            .expect("failed to write benchmark output");
        }
        writeln!(out).expect("failed to write benchmark output");

        // Speedup Analysis
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "SPEEDUP ANALYSIS").expect("failed to write benchmark output");
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");

        let ollama_tps = self
            .gguf_ollama
            .as_ref()
            .map_or(318.0, |m| m.tokens_per_sec);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(200.0, |m| m.tokens_per_sec);

        if let Some(ref m) = self.gguf_apr {
            let vs_ollama = m.tokens_per_sec / ollama_tps;
            let vs_llamacpp = m.tokens_per_sec / llamacpp_tps;
            writeln!(
                out,
                "APR GGUF vs Ollama:     {:>5.2}x  {}",
                vs_ollama,
                if vs_ollama >= 1.0 { "✓" } else { "⚠" }
            )
            .expect("failed to write benchmark output");
            writeln!(
                out,
                "APR GGUF vs llama.cpp:  {:>5.2}x  {}",
                vs_llamacpp,
                if vs_llamacpp >= 1.25 {
                    "✓ Point 41 PASS"
                } else {
                    "⚠ Point 41 FAIL"
                }
            )
            .expect("failed to write benchmark output");
        }

        if let Some(ref m) = self.apr_native {
            let vs_ollama = m.tokens_per_sec / ollama_tps;
            writeln!(
                out,
                "APR .apr vs Ollama:     {:>5.2}x  {}",
                vs_ollama,
                if vs_ollama >= 2.0 {
                    "✓ 2x target"
                } else {
                    ""
                }
            )
            .expect("failed to write benchmark output");
        }
        writeln!(out).expect("failed to write benchmark output");

        // Profiling Hotspots
        if !self.hotspots.is_empty() {
            writeln!(
                out,
                "───────────────────────────────────────────────────────────────────────"
            )
            .expect("failed to write benchmark output");
            writeln!(out, "PROFILING HOTSPOTS (>5% of execution time)")
                .expect("failed to write benchmark output");
            writeln!(
                out,
                "───────────────────────────────────────────────────────────────────────"
            )
            .expect("failed to write benchmark output");

            for hotspot in &self.hotspots {
                writeln!(out, "{}", hotspot.to_line()).expect("failed to write benchmark output");
                if !hotspot.explanation.is_empty() {
                    writeln!(out, "   └─ {}", hotspot.explanation)
                        .expect("failed to write benchmark output");
                }
            }
            writeln!(out).expect("failed to write benchmark output");
        }

        // GPU Metrics
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "GPU METRICS").expect("failed to write benchmark output");
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");

        if let Some(ref m) = self.gguf_apr {
            if let (Some(util), Some(mem)) = (m.gpu_util, m.gpu_mem_mb) {
                writeln!(
                    out,
                    "APR GGUF:   GPU Util: {:>5.1}%  VRAM: {:>6.0}MB",
                    util, mem
                )
                .expect("failed to write benchmark output");
            }
        }
        if let Some(ref m) = self.apr_native {
            if let (Some(util), Some(mem)) = (m.gpu_util, m.gpu_mem_mb) {
                writeln!(
                    out,
                    "APR .apr:   GPU Util: {:>5.1}%  VRAM: {:>6.0}MB",
                    util, mem
                )
                .expect("failed to write benchmark output");
            }
        }
        writeln!(out).expect("failed to write benchmark output");

        // Recommendations
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "OPTIMIZATION RECOMMENDATIONS").expect("failed to write benchmark output");
        writeln!(
            out,
            "───────────────────────────────────────────────────────────────────────"
        )
        .expect("failed to write benchmark output");

        let unexpected: Vec<_> = self.hotspots.iter().filter(|h| !h.is_expected).collect();
        if unexpected.is_empty() {
            writeln!(out, "✓ No unexpected hotspots detected")
                .expect("failed to write benchmark output");
        } else {
            for h in unexpected {
                writeln!(out, "⚠ {}: {}", h.component, h.explanation)
                    .expect("failed to write benchmark output");
            }
        }

        // Phase 2 status
        let apr_tps = self.gguf_apr.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        if apr_tps < 500.0 {
            writeln!(out).expect("failed to write benchmark output");
            writeln!(out, "Phase 2 Optimizations (projected 3.28x improvement):")
                .expect("failed to write benchmark output");
            writeln!(out, "  PAR-036: Persistent threads      (1.3x)")
                .expect("failed to write benchmark output");
            writeln!(out, "  PAR-037: CUDA graph capture      (1.5x)")
                .expect("failed to write benchmark output");
            writeln!(out, "  PAR-038: Multi-stream pipeline   (1.2x)")
                .expect("failed to write benchmark output");
            writeln!(out, "  PAR-039: Megakernel fusion       (1.4x)")
                .expect("failed to write benchmark output");
            writeln!(
                out,
                "  Projected: {:.1} × 3.28 = {:.1} tok/s",
                apr_tps,
                apr_tps * 3.28
            )
            .expect("failed to write benchmark output");
        }

        writeln!(
            out,
            "═══════════════════════════════════════════════════════════════════════"
        )
        .expect("failed to write benchmark output");
        writeln!(out, "```").expect("failed to write benchmark output");

        out
    }

    /// Generate compact one-liner for quick comparison
    pub fn render_compact(&self) -> String {
        let apr_tps = self.gguf_apr.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let ollama_tps = self.gguf_ollama.as_ref().map_or(0.0, |m| m.tokens_per_sec);
        let llamacpp_tps = self
            .gguf_llamacpp
            .as_ref()
            .map_or(0.0, |m| m.tokens_per_sec);

        format!(
            "APR:{:.0} Ollama:{:.0} llama.cpp:{:.0} tok/s | APR vs Ollama:{:.2}x vs llama.cpp:{:.2}x",
            apr_tps, ollama_tps, llamacpp_tps,
            apr_tps / ollama_tps.max(1.0),
            apr_tps / llamacpp_tps.max(1.0)
        )
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Benchmark runner with profiling
#[derive(Debug)]
pub struct BenchmarkRunner {
    /// Results grid
    pub grid: BenchmarkGrid,
    /// Profiling start time
    start_time: Option<Instant>,
    /// Component timings
    component_times: Vec<(String, Duration, u64)>,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkRunner {
    /// Create new benchmark runner
    pub fn new() -> Self {
        Self {
            grid: BenchmarkGrid::new(),
            start_time: None,
            component_times: Vec::new(),
        }
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Record component timing
    pub fn record_component(&mut self, name: &str, duration: Duration, calls: u64) {
        self.component_times
            .push((name.to_string(), duration, calls));
    }

    /// Measure a component
    pub fn measure<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        self.record_component(name, start.elapsed(), 1);
        result
    }

    /// Finalize and compute hotspots
    pub fn finalize(&mut self) {
        let total_time: Duration = self.component_times.iter().map(|(_, d, _)| *d).sum();
        let total_nanos = total_time.as_nanos() as f64;

        if total_nanos == 0.0 {
            return;
        }

        for (name, duration, calls) in &self.component_times {
            let percentage = (duration.as_nanos() as f64 / total_nanos) * 100.0;

            if percentage > 5.0 {
                let avg_per_call = if *calls > 0 {
                    Duration::from_nanos((duration.as_nanos() / *calls as u128) as u64)
                } else {
                    Duration::ZERO
                };

                let (explanation, is_expected) = explain_inference_hotspot(name, percentage);

                self.grid.add_hotspot(ProfilingHotspot {
                    component: name.clone(),
                    time: *duration,
                    percentage,
                    call_count: *calls,
                    avg_per_call,
                    explanation,
                    is_expected,
                });
            }
        }

        // Sort by percentage descending
        self.grid.hotspots.sort_by(|a, b| {
            b.percentage
                .partial_cmp(&a.percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Render ASCII bar
fn render_bar(value: f64, max: f64, width: usize) -> String {
    let ratio = if max > 0.0 { value / max } else { 0.0 };
    let filled = ((ratio * width as f64) as usize).min(width);
    let empty = width - filled;

    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

/// Truncate string to max length
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
    }
}

/// Explain inference hotspot
fn explain_inference_hotspot(component: &str, percentage: f64) -> (String, bool) {
    match component {
        "Q4K_GEMV" | "MatMul" | "GEMM" => (
            format!(
                "Matrix ops dominate ({:.1}%) - expected for transformer inference",
                percentage
            ),
            true,
        ),
        "Attention" | "FlashAttention" => (
            format!(
                "Attention at {:.1}% - normal for autoregressive decoding",
                percentage
            ),
            true,
        ),
        "KV_Cache" | "KVCache" => {
            if percentage > 20.0 {
                (
                    "KV cache overhead high - consider FP16 cache or graph capture".to_string(),
                    false,
                )
            } else {
                ("KV cache within normal range".to_string(), true)
            }
        },
        "Softmax" => {
            if percentage > 10.0 {
                (
                    "Softmax unusually high - check for redundant computations".to_string(),
                    false,
                )
            } else {
                ("Softmax within normal range".to_string(), true)
            }
        },
        "RMSNorm" | "LayerNorm" => {
            if percentage > 15.0 {
                (
                    "Normalization overhead high - consider fused kernels".to_string(),
                    false,
                )
            } else {
                ("Normalization within normal range".to_string(), true)
            }
        },
        "MemcpyH2D" | "MemcpyD2H" | "Transfer" => (
            "Memory transfer - consider persistent GPU buffers".to_string(),
            false,
        ),
        "KernelLaunch" => (
            "Kernel launch overhead - consider CUDA graphs or megakernels".to_string(),
            false,
        ),
        "Embedding" => (
            "Embedding lookup - expected at start of inference".to_string(),
            true,
        ),
        "Sampling" | "TopK" | "TopP" => (
            "Sampling overhead - expected for token generation".to_string(),
            true,
        ),
        _ => {
            if percentage > 20.0 {
                (
                    format!("Unknown component at {:.1}% - investigate", percentage),
                    false,
                )
            } else {
                (String::new(), true)
            }
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_grid_ascii() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Qwen2.5-Coder-0.5B", "0.5B", "Q4_K_M")
            .with_gpu("RTX 4090", 24.0);

        grid.set_gguf_row(
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput(500.0)
                .with_ttft(7.0),
            BenchMeasurement::new("Ollama", "GGUF")
                .with_throughput(318.0)
                .with_ttft(50.0),
            BenchMeasurement::new("llama.cpp", "GGUF")
                .with_throughput(200.0)
                .with_ttft(30.0),
        );

        grid.set_apr_row(
            BenchMeasurement::new("APR", ".apr")
                .with_throughput(600.0)
                .with_ttft(5.0),
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput(500.0)
                .with_ttft(7.0),
            BenchMeasurement::new("Ollama", "GGUF")
                .with_throughput(318.0)
                .with_ttft(50.0),
        );

        let ascii = grid.render_ascii();
        assert!(ascii.contains("APR serve GGUF"));
        assert!(ascii.contains("Ollama"));
        assert!(ascii.contains("llama.cpp"));
        assert!(ascii.contains("500.0 tok/s"));
    }

    #[test]
    fn test_profiling_log() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Qwen2.5-Coder-0.5B", "0.5B", "Q4_K_M")
            .with_gpu("RTX 4090", 24.0);

        grid.gguf_apr = Some(
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput(500.0)
                .with_ttft(7.0)
                .with_gpu(95.0, 2048.0),
        );

        grid.add_hotspot(ProfilingHotspot {
            component: "Q4K_GEMV".to_string(),
            time: Duration::from_millis(150),
            percentage: 45.0,
            call_count: 1000,
            avg_per_call: Duration::from_micros(150),
            explanation: "Matrix ops dominate - expected".to_string(),
            is_expected: true,
        });

        let log = grid.render_profiling_log();
        assert!(log.contains("PROFILING REPORT"));
        assert!(log.contains("Q4K_GEMV"));
        assert!(log.contains("45.0%"));
    }

    #[test]
    fn test_compact_output() {
        let mut grid = BenchmarkGrid::new();
        grid.gguf_apr = Some(BenchMeasurement::new("APR", "GGUF").with_throughput(500.0));
        grid.gguf_ollama = Some(BenchMeasurement::new("Ollama", "GGUF").with_throughput(318.0));
        grid.gguf_llamacpp =
            Some(BenchMeasurement::new("llama.cpp", "GGUF").with_throughput(200.0));

        let compact = grid.render_compact();
        assert!(compact.contains("APR:500"));
        assert!(compact.contains("vs llama.cpp:2.50x"));
    }

    #[test]
    fn test_runner_profiling() {
        let mut runner = BenchmarkRunner::new();
        runner.start();

        runner.record_component("Q4K_GEMV", Duration::from_millis(100), 500);
        runner.record_component("Attention", Duration::from_millis(50), 500);
        runner.record_component("Other", Duration::from_millis(10), 100);

        runner.finalize();

        assert!(!runner.grid.hotspots.is_empty());
        assert_eq!(runner.grid.hotspots[0].component, "Q4K_GEMV");
    }

    #[test]
    fn test_render_bar() {
        let bar = render_bar(50.0, 100.0, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 5);
        assert_eq!(bar.chars().filter(|c| *c == '░').count(), 5);
    }

    // =========================================================================
    // Coverage Tests: BenchMeasurement
    // =========================================================================

    #[test]
    fn test_bench_measurement_new() {
        let m = BenchMeasurement::new("TestEngine", "TestFormat");
        assert_eq!(m.engine, "TestEngine");
        assert_eq!(m.format, "TestFormat");
        assert_eq!(m.tokens_per_sec, 0.0);
        assert_eq!(m.ttft_ms, 0.0);
        assert_eq!(m.tokens_generated, 0);
        assert!(m.gpu_util.is_none());
        assert!(m.gpu_mem_mb.is_none());
    }

    #[test]
    fn test_bench_measurement_with_throughput() {
        let m = BenchMeasurement::new("APR", "GGUF").with_throughput(100.0);
        assert_eq!(m.tokens_per_sec, 100.0);
    }

    #[test]
    fn test_bench_measurement_with_ttft() {
        let m = BenchMeasurement::new("APR", "GGUF").with_ttft(25.5);
        assert_eq!(m.ttft_ms, 25.5);
    }

    #[test]
    fn test_bench_measurement_with_tokens() {
        let duration = Duration::from_secs(2);
        let m = BenchMeasurement::new("APR", "GGUF").with_tokens(200, duration);
        assert_eq!(m.tokens_generated, 200);
        assert_eq!(m.duration, duration);
        assert!((m.tokens_per_sec - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_bench_measurement_with_tokens_zero_duration() {
        let m = BenchMeasurement::new("APR", "GGUF").with_tokens(100, Duration::ZERO);
        assert_eq!(m.tokens_generated, 100);
        // Zero duration means no TPS calculation
    }

    #[test]
    fn test_bench_measurement_with_gpu() {
        let m = BenchMeasurement::new("APR", "GGUF").with_gpu(95.0, 4096.0);
        assert_eq!(m.gpu_util, Some(95.0));
        assert_eq!(m.gpu_mem_mb, Some(4096.0));
    }

    #[test]
    fn test_bench_measurement_debug() {
        let m = BenchMeasurement::new("APR", "GGUF").with_throughput(100.0);
        let debug_str = format!("{:?}", m);
        assert!(debug_str.contains("BenchMeasurement"));
        assert!(debug_str.contains("APR"));
    }

    #[test]
    fn test_bench_measurement_clone() {
        let m = BenchMeasurement::new("APR", "GGUF").with_throughput(100.0).with_gpu(90.0, 1024.0);
        let cloned = m.clone();
        assert_eq!(cloned.engine, m.engine);
        assert_eq!(cloned.tokens_per_sec, m.tokens_per_sec);
        assert_eq!(cloned.gpu_util, m.gpu_util);
    }

    // =========================================================================
    // Coverage Tests: ProfilingHotspot
    // =========================================================================

    #[test]
    fn test_profiling_hotspot_debug() {
        let hotspot = ProfilingHotspot {
            component: "Attention".to_string(),
            time: Duration::from_millis(100),
            percentage: 50.0,
            call_count: 1000,
            avg_per_call: Duration::from_micros(100),
            explanation: "Expected".to_string(),
            is_expected: true,
        };
        let debug_str = format!("{:?}", hotspot);
        assert!(debug_str.contains("ProfilingHotspot"));
        assert!(debug_str.contains("Attention"));
    }

    #[test]
    fn test_profiling_hotspot_clone() {
        let hotspot = ProfilingHotspot {
            component: "GEMM".to_string(),
            time: Duration::from_millis(200),
            percentage: 75.0,
            call_count: 500,
            avg_per_call: Duration::from_micros(400),
            explanation: "Matrix multiplication".to_string(),
            is_expected: true,
        };
        let cloned = hotspot.clone();
        assert_eq!(cloned.component, hotspot.component);
        assert_eq!(cloned.percentage, hotspot.percentage);
    }

    // =========================================================================
    // Coverage Tests: BenchmarkGrid
    // =========================================================================

    #[test]
    fn test_benchmark_grid_new() {
        let grid = BenchmarkGrid::new();
        assert!(grid.gguf_apr.is_none());
        assert!(grid.gguf_ollama.is_none());
        assert!(grid.gguf_llamacpp.is_none());
        assert!(grid.hotspots.is_empty());
    }

    #[test]
    fn test_benchmark_grid_with_model() {
        let grid = BenchmarkGrid::new()
            .with_model("Llama-7B", "7B", "Q4_K_M");
        assert_eq!(grid.model_name, "Llama-7B");
        assert_eq!(grid.model_params, "7B");
        assert_eq!(grid.quantization, "Q4_K_M");
    }

    #[test]
    fn test_benchmark_grid_with_gpu() {
        let grid = BenchmarkGrid::new()
            .with_gpu("RTX 3090", 24.0);
        assert_eq!(grid.gpu_name, "RTX 3090");
        assert_eq!(grid.gpu_vram_gb, 24.0);
    }

    #[test]
    fn test_benchmark_grid_add_hotspot() {
        let mut grid = BenchmarkGrid::new();
        grid.add_hotspot(ProfilingHotspot {
            component: "Test".to_string(),
            time: Duration::from_millis(50),
            percentage: 25.0,
            call_count: 100,
            avg_per_call: Duration::from_micros(500),
            explanation: "Test hotspot".to_string(),
            is_expected: true,
        });
        assert_eq!(grid.hotspots.len(), 1);
        assert_eq!(grid.hotspots[0].component, "Test");
    }

    // =========================================================================
    // Coverage Tests: render_bar edge cases
    // =========================================================================

    #[test]
    fn test_render_bar_zero() {
        let bar = render_bar(0.0, 100.0, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 0);
        assert_eq!(bar.chars().filter(|c| *c == '░').count(), 10);
    }

    #[test]
    fn test_render_bar_full() {
        let bar = render_bar(100.0, 100.0, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 10);
        assert_eq!(bar.chars().filter(|c| *c == '░').count(), 0);
    }

    #[test]
    fn test_render_bar_over_max() {
        let bar = render_bar(150.0, 100.0, 10);
        // Should clamp to max
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 10);
    }

    // =========================================================================
    // Coverage Tests: truncate
    // =========================================================================

    #[test]
    fn test_truncate_short_string() {
        let result = truncate("short", 10);
        assert_eq!(result, "short");
    }

    #[test]
    fn test_truncate_exact_length() {
        let result = truncate("exactly10c", 10);
        assert_eq!(result, "exactly10c");
    }

    #[test]
    fn test_truncate_long_string() {
        let result = truncate("this is a very long string", 10);
        assert_eq!(result.len(), 10);
    }

    // =========================================================================
    // Coverage Tests: explain_inference_hotspot
    // =========================================================================

    #[test]
    fn test_explain_inference_hotspot_gemv() {
        let (explanation, is_expected) = explain_inference_hotspot("Q4K_GEMV", 50.0);
        assert!(is_expected);
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_explain_inference_hotspot_attention() {
        let (explanation, is_expected) = explain_inference_hotspot("Attention", 30.0);
        assert!(is_expected);
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_explain_inference_hotspot_unknown() {
        let (explanation, is_expected) = explain_inference_hotspot("UnknownComponent", 60.0);
        // High percentage for unknown component is unexpected
        assert!(!is_expected);
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_explain_inference_hotspot_low_percentage() {
        let (explanation, is_expected) = explain_inference_hotspot("SomeComponent", 5.0);
        // Low percentage unknown component returns empty string and is expected
        assert!(is_expected);
        // Note: The function returns empty string for low percentage unknown components
        // which means "nothing to report" - this is valid behavior
        let _ = explanation;
    }
}
