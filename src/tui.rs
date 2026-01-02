//! TUI Monitoring for LLM Inference
//!
//! Real-time terminal UI for monitoring inference performance.
//! Provides visual feedback on throughput, latency, and GPU utilization.
//!
//! # Usage
//!
//! ```rust,ignore
//! use realizar::tui::{InferenceTui, TuiConfig, InferenceMetrics};
//!
//! let config = TuiConfig::default();
//! let mut tui = InferenceTui::new(config);
//!
//! // Update with metrics during inference
//! tui.update(&metrics);
//!
//! // Render to string (for testing)
//! let output = tui.render_to_string();
//! ```
//!
//! # Visual Elements
//!
//! ```text
//! ╭─────────────────────────────────────────────────────────────╮
//! │             realizar Inference Monitor                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Throughput: 64.2 tok/s   Target: 192 tok/s (M4)           │
//! │  Latency:    15.6 ms/tok  P95: 23.4 ms                     │
//! │  GPU Memory: 4.2 GB / 24 GB                                │
//! │  Batch Size: 4            Queue: 12 pending                │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Throughput: ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█                       │
//! │  Latency:    ▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Status: ● Running   Tokens: 1,234   Requests: 42          │
//! ╰─────────────────────────────────────────────────────────────╯
//! ```

use std::collections::VecDeque;

/// TUI configuration
#[derive(Debug, Clone)]
pub struct TuiConfig {
    /// Refresh rate in milliseconds
    pub refresh_rate_ms: u64,
    /// Show throughput sparkline
    pub show_throughput_sparkline: bool,
    /// Show latency sparkline
    pub show_latency_sparkline: bool,
    /// Show GPU memory usage
    pub show_gpu_memory: bool,
    /// Title for the TUI window
    pub title: String,
    /// Target throughput for M4 parity
    pub m4_target_tok_per_sec: f64,
    /// Width of the TUI display
    pub width: usize,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            refresh_rate_ms: 100,
            show_throughput_sparkline: true,
            show_latency_sparkline: true,
            show_gpu_memory: true,
            title: "realizar Inference Monitor".to_string(),
            m4_target_tok_per_sec: 192.0,
            width: 65,
        }
    }
}

/// Real-time inference metrics
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    /// Current throughput (tokens/second)
    pub throughput_tok_per_sec: f64,
    /// Mean latency per token (milliseconds)
    pub latency_ms: f64,
    /// P95 latency (milliseconds)
    pub latency_p95_ms: f64,
    /// GPU memory used (bytes)
    pub gpu_memory_bytes: u64,
    /// GPU memory total (bytes)
    pub gpu_memory_total_bytes: u64,
    /// Current batch size
    pub batch_size: usize,
    /// Pending requests in queue
    pub queue_size: usize,
    /// Total tokens generated
    pub total_tokens: u64,
    /// Total requests processed
    pub total_requests: u64,
    /// Is currently running
    pub running: bool,
    /// Is using GPU
    pub using_gpu: bool,
}

impl InferenceMetrics {
    /// Create new metrics with defaults
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if throughput achieves M4 parity (192 tok/s)
    #[must_use]
    pub fn achieves_m4_parity(&self) -> bool {
        self.throughput_tok_per_sec >= 192.0
    }

    /// Calculate gap to M4 target
    #[must_use]
    pub fn gap_to_m4(&self) -> f64 {
        if self.throughput_tok_per_sec > 0.0 {
            192.0 / self.throughput_tok_per_sec
        } else {
            f64::INFINITY
        }
    }

    /// Format GPU memory as human-readable string
    #[must_use]
    pub fn format_gpu_memory(&self) -> String {
        let used_gb = self.gpu_memory_bytes as f64 / 1e9;
        let total_gb = self.gpu_memory_total_bytes as f64 / 1e9;
        format!("{:.1} GB / {:.1} GB", used_gb, total_gb)
    }
}

/// TUI state for rendering
#[derive(Debug, Clone)]
pub struct InferenceTui {
    /// Configuration
    config: TuiConfig,
    /// Current metrics
    metrics: InferenceMetrics,
    /// Throughput history (for sparkline)
    throughput_history: VecDeque<f64>,
    /// Latency history (for sparkline)
    latency_history: VecDeque<f64>,
    /// Maximum history size
    max_history: usize,
}

impl InferenceTui {
    /// Create new TUI with configuration
    #[must_use]
    pub fn new(config: TuiConfig) -> Self {
        Self {
            config,
            metrics: InferenceMetrics::default(),
            throughput_history: VecDeque::new(),
            latency_history: VecDeque::new(),
            max_history: 40,
        }
    }

    /// Update TUI with new metrics
    pub fn update(&mut self, metrics: &InferenceMetrics) {
        self.metrics = metrics.clone();

        // Add to history
        self.throughput_history
            .push_back(metrics.throughput_tok_per_sec);
        self.latency_history.push_back(metrics.latency_ms);

        // Trim history
        while self.throughput_history.len() > self.max_history {
            self.throughput_history.pop_front();
        }
        while self.latency_history.len() > self.max_history {
            self.latency_history.pop_front();
        }
    }

    /// Generate sparkline string from values
    fn sparkline(values: &VecDeque<f64>, width: usize) -> String {
        const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if values.is_empty() {
            return " ".repeat(width);
        }

        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = (max - min).max(0.001);

        let mut result: String = values
            .iter()
            .take(width)
            .map(|&v| {
                let normalized = (v - min) / range;
                let level = (normalized * 7.0).round().clamp(0.0, 7.0) as usize;
                BLOCKS[level]
            })
            .collect();

        // Pad to width
        while result.chars().count() < width {
            result.push(' ');
        }

        result
    }

    /// Render TUI to string (for testing and display)
    #[must_use]
    pub fn render_to_string(&self) -> String {
        let w = self.config.width;
        let inner_w = w - 2; // Account for │ borders on each side

        let mut lines = Vec::new();

        // Top border
        lines.push(format!("╭{}╮", "─".repeat(w - 2)));

        // Title
        let title = &self.config.title;
        let padding = (inner_w - title.len()) / 2;
        lines.push(format!(
            "│{}{}{}│",
            " ".repeat(padding),
            title,
            " ".repeat(inner_w - padding - title.len())
        ));

        // Separator
        lines.push(format!("├{}┤", "─".repeat(w - 2)));

        // Throughput line
        let status_icon = if self.metrics.achieves_m4_parity() {
            "✓"
        } else {
            "○"
        };
        let throughput_line = format!(
            "  Throughput: {:.1} tok/s {} Target: {:.0} tok/s (M4)",
            self.metrics.throughput_tok_per_sec, status_icon, self.config.m4_target_tok_per_sec
        );
        lines.push(Self::pad_line(&throughput_line, inner_w));

        // Latency line
        let latency_line = format!(
            "  Latency:    {:.1} ms/tok  P95: {:.1} ms",
            self.metrics.latency_ms, self.metrics.latency_p95_ms
        );
        lines.push(Self::pad_line(&latency_line, inner_w));

        // GPU memory line
        if self.config.show_gpu_memory {
            let gpu_line = format!("  GPU Memory: {}", self.metrics.format_gpu_memory());
            lines.push(Self::pad_line(&gpu_line, inner_w));
        }

        // Batch info line
        let batch_line = format!(
            "  Batch Size: {}            Queue: {} pending",
            self.metrics.batch_size, self.metrics.queue_size
        );
        lines.push(Self::pad_line(&batch_line, inner_w));

        // Separator
        lines.push(format!("├{}┤", "─".repeat(w - 2)));

        // Sparklines
        if self.config.show_throughput_sparkline {
            let sparkline = Self::sparkline(&self.throughput_history, 40);
            let spark_line = format!("  Throughput: {}", sparkline);
            lines.push(Self::pad_line(&spark_line, inner_w));
        }

        if self.config.show_latency_sparkline {
            let sparkline = Self::sparkline(&self.latency_history, 40);
            let spark_line = format!("  Latency:    {}", sparkline);
            lines.push(Self::pad_line(&spark_line, inner_w));
        }

        // Separator
        lines.push(format!("├{}┤", "─".repeat(w - 2)));

        // Status line
        let status = if self.metrics.running {
            "● Running"
        } else {
            "○ Stopped"
        };
        let gpu_status = if self.metrics.using_gpu { "GPU" } else { "CPU" };
        let status_line = format!(
            "  Status: {}  [{:>3}]  Tokens: {:>6}  Requests: {:>4}",
            status, gpu_status, self.metrics.total_tokens, self.metrics.total_requests
        );
        lines.push(Self::pad_line(&status_line, inner_w));

        // Bottom border
        lines.push(format!("╰{}╯", "─".repeat(w - 2)));

        lines.join("\n")
    }

    /// Pad line to fit within borders
    fn pad_line(content: &str, width: usize) -> String {
        let content_len = content.chars().count();
        if content_len >= width {
            format!("│{}│", &content[..width])
        } else {
            format!("│{}{}│", content, " ".repeat(width - content_len))
        }
    }

    /// Get current metrics
    #[must_use]
    pub fn metrics(&self) -> &InferenceMetrics {
        &self.metrics
    }

    /// Get throughput history for testing
    #[must_use]
    pub fn throughput_history(&self) -> &VecDeque<f64> {
        &self.throughput_history
    }

    /// Get latency history for testing
    #[must_use]
    pub fn latency_history(&self) -> &VecDeque<f64> {
        &self.latency_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PARITY-090: TUI Configuration Tests
    // =========================================================================

    /// PARITY-090a: Test TuiConfig defaults
    #[test]
    fn test_parity_090a_tui_config_defaults() {
        println!("PARITY-090a: TuiConfig Default Values");

        let config = TuiConfig::default();

        println!("  refresh_rate_ms: {}", config.refresh_rate_ms);
        println!("  m4_target_tok_per_sec: {}", config.m4_target_tok_per_sec);
        println!("  width: {}", config.width);

        assert_eq!(config.refresh_rate_ms, 100);
        assert_eq!(config.m4_target_tok_per_sec, 192.0);
        assert!(config.show_throughput_sparkline);
        assert!(config.show_latency_sparkline);
    }

    /// PARITY-090b: Test InferenceMetrics creation
    #[test]
    fn test_parity_090b_inference_metrics() {
        println!("PARITY-090b: InferenceMetrics");

        let metrics = InferenceMetrics {
            throughput_tok_per_sec: 64.0,
            latency_ms: 15.6,
            latency_p95_ms: 23.4,
            gpu_memory_bytes: 4_200_000_000,
            gpu_memory_total_bytes: 24_000_000_000,
            batch_size: 4,
            queue_size: 12,
            total_tokens: 1234,
            total_requests: 42,
            running: true,
            using_gpu: true,
        };

        println!("  throughput: {:.1} tok/s", metrics.throughput_tok_per_sec);
        println!("  achieves_m4: {}", metrics.achieves_m4_parity());
        println!("  gap_to_m4: {:.2}x", metrics.gap_to_m4());
        println!("  gpu_memory: {}", metrics.format_gpu_memory());

        assert!(!metrics.achieves_m4_parity());
        assert!((metrics.gap_to_m4() - 3.0).abs() < 0.1);
        assert!(metrics.format_gpu_memory().contains("4.2 GB"));
    }

    /// PARITY-090c: Test M4 parity detection
    #[test]
    fn test_parity_090c_m4_parity_detection() {
        println!("PARITY-090c: M4 Parity Detection");

        let test_cases = [
            (64.0, false, "Baseline - not M4"),
            (150.0, false, "Batch threshold - not M4"),
            (192.0, true, "Exactly M4"),
            (256.0, true, "Above M4"),
        ];

        for (throughput, expected, description) in test_cases {
            let metrics = InferenceMetrics {
                throughput_tok_per_sec: throughput,
                ..Default::default()
            };
            let achieves = metrics.achieves_m4_parity();
            println!("  {}: {} tok/s → M4={}", description, throughput, achieves);
            assert_eq!(achieves, expected, "{}", description);
        }
    }

    // =========================================================================
    // PARITY-091: TUI Rendering Tests
    // =========================================================================

    /// PARITY-091a: Test TUI creation and update
    #[test]
    fn test_parity_091a_tui_creation_update() {
        println!("PARITY-091a: TUI Creation and Update");

        let config = TuiConfig::default();
        let mut tui = InferenceTui::new(config);

        let metrics = InferenceMetrics {
            throughput_tok_per_sec: 64.0,
            latency_ms: 15.6,
            running: true,
            ..Default::default()
        };

        tui.update(&metrics);

        assert_eq!(tui.metrics().throughput_tok_per_sec, 64.0);
        assert_eq!(tui.throughput_history().len(), 1);
    }

    /// PARITY-091b: Test sparkline generation
    #[test]
    fn test_parity_091b_sparkline_generation() {
        println!("PARITY-091b: Sparkline Generation");

        let mut history = VecDeque::new();
        for i in 0..20 {
            history.push_back((i as f64) * 10.0);
        }

        let sparkline = InferenceTui::sparkline(&history, 20);
        println!("  Sparkline: {}", sparkline);

        assert_eq!(sparkline.chars().count(), 20);
        assert!(sparkline.contains('▁')); // Low values
        assert!(sparkline.contains('█')); // High values
    }

    /// PARITY-091c: Test TUI render output structure
    #[test]
    fn test_parity_091c_tui_render_structure() {
        println!("PARITY-091c: TUI Render Output Structure");

        let config = TuiConfig::default();
        let mut tui = InferenceTui::new(config);

        // Add some history
        for i in 0..10 {
            let metrics = InferenceMetrics {
                throughput_tok_per_sec: 50.0 + (i as f64) * 5.0,
                latency_ms: 20.0 - (i as f64),
                batch_size: 4,
                queue_size: 12,
                total_tokens: 1234,
                total_requests: 42,
                running: true,
                using_gpu: true,
                ..Default::default()
            };
            tui.update(&metrics);
        }

        let output = tui.render_to_string();
        println!("{}", output);

        // Verify structure
        assert!(output.contains("╭"), "Should have top border");
        assert!(output.contains("╰"), "Should have bottom border");
        assert!(
            output.contains("realizar Inference Monitor"),
            "Should have title"
        );
        assert!(output.contains("Throughput:"), "Should show throughput");
        assert!(output.contains("Latency:"), "Should show latency");
        assert!(output.contains("tok/s"), "Should show tok/s unit");
        assert!(output.contains("● Running"), "Should show running status");
    }

    /// PARITY-091d: Test TUI visual regression baseline
    #[test]
    fn test_parity_091d_visual_regression_baseline() {
        println!("PARITY-091d: Visual Regression Baseline");

        let config = TuiConfig {
            width: 65,
            ..Default::default()
        };
        let mut tui = InferenceTui::new(config);

        let metrics = InferenceMetrics {
            throughput_tok_per_sec: 64.2,
            latency_ms: 15.6,
            latency_p95_ms: 23.4,
            gpu_memory_bytes: 4_200_000_000,
            gpu_memory_total_bytes: 24_000_000_000,
            batch_size: 4,
            queue_size: 12,
            total_tokens: 1234,
            total_requests: 42,
            running: true,
            using_gpu: true,
        };

        tui.update(&metrics);
        let output = tui.render_to_string();

        println!("=== GOLDEN BASELINE ===");
        println!("{}", output);
        println!("=== END BASELINE ===");

        // Verify key visual elements
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines.len() >= 10, "Should have at least 10 lines");

        // Check border consistency
        assert!(lines[0].starts_with('╭'));
        assert!(lines[0].ends_with('╮'));
        assert!(lines.last().expect("test").starts_with('╰'));
        assert!(lines.last().expect("test").ends_with('╯'));

        // Check content
        assert!(
            output.contains("64.2 tok/s"),
            "Should show throughput value"
        );
        assert!(output.contains("15.6 ms/tok"), "Should show latency value");
        assert!(output.contains("1234"), "Should show token count");
    }

    /// PARITY-091e: Test history accumulation
    #[test]
    fn test_parity_091e_history_accumulation() {
        println!("PARITY-091e: History Accumulation");

        let config = TuiConfig::default();
        let mut tui = InferenceTui::new(config);

        // Add more than max_history items
        for i in 0..50 {
            let metrics = InferenceMetrics {
                throughput_tok_per_sec: (i as f64) * 2.0,
                latency_ms: 100.0 - (i as f64),
                ..Default::default()
            };
            tui.update(&metrics);
        }

        // Should be capped at max_history (40)
        assert_eq!(tui.throughput_history().len(), 40);
        assert_eq!(tui.latency_history().len(), 40);

        // Most recent should be last
        assert!((tui.throughput_history().back().expect("test") - 98.0).abs() < 0.1);
    }

    /// PARITY-091f: Test empty sparkline handling
    #[test]
    fn test_parity_091f_empty_sparkline() {
        println!("PARITY-091f: Empty Sparkline Handling");

        let empty: VecDeque<f64> = VecDeque::new();
        let sparkline = InferenceTui::sparkline(&empty, 20);

        println!("  Empty sparkline: '{}'", sparkline);
        assert_eq!(sparkline.len(), 20);
        assert!(sparkline.chars().all(|c| c == ' '));
    }
}
