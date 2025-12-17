//! realizar-monitor: Real-time btop-style TUI for inference server monitoring
//!
//! A rich terminal dashboard inspired by simular's TUI patterns.
//!
//! # Usage
//!
//! ```bash
//! # Start server in one terminal
//! realizar serve --model model.gguf --batch
//!
//! # Monitor in another terminal
//! realizar-monitor --url http://127.0.0.1:8080
//! ```
//!
//! # Layout
//!
//! ```text
//! ┌─────────────────────────────────────┬────────────────────────────┐
//! │ Throughput                     60%  │ Metrics                40% │
//! │ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁    │                            │
//! │                                     │ Throughput: 192.4 tok/s ↑ │
//! │ Current: 192.4 tok/s                │ Latency P50: 5.2ms        │
//! │ Peak:    256.1 tok/s                │ Latency P95: 7.8ms        │
//! │ Trend:   ↑ Rising                   │ Latency P99: 12.1ms       │
//! ├─────────────────────────────────────┤                            │
//! │ GPU Memory                          │ Requests: 1,234           │
//! │ ████████████████████░░░░░░░░ 67%    │ Tokens:   50,000          │
//! │ 16.1 GB / 24.0 GB                   │ Uptime:   1h 23m 45s      │
//! │                                     ├────────────────────────────┤
//! │ CUDA: ● Active                      │ System                     │
//! │ Batch Size: 32                      │ Model: phi-2-q4_k_m        │
//! │ Queue: 8 pending                    │ Batch: 32 optimal          │
//! └─────────────────────────────────────┴────────────────────────────┘
//! │ [q] Quit  [r] Reset  [p] Pause                               │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use clap::Parser;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline as RatatuiSparkline},
};
use serde::Deserialize;

/// Monitor state machine (PARITY-108a)
/// Explicit state enum for QA compliance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorState {
    /// Not connected to server
    Disconnected,
    /// Connected and receiving metrics
    Connected,
    /// Connected but updates paused
    Paused,
}

impl MonitorState {
    /// Check if the monitor is receiving updates
    #[allow(dead_code)]
    fn is_active(&self) -> bool {
        matches!(self, MonitorState::Connected)
    }
}

/// GPU utilization color coding (PARITY-108c)
/// Returns appropriate color based on GPU utilization percentage
///
/// - Green: ≤70% (healthy)
/// - Yellow: 71-90% (warning)
/// - Red: >90% (critical)
fn gpu_color(percent: u16) -> Color {
    if percent > 90 {
        Color::Red
    } else if percent > 70 {
        Color::Yellow
    } else {
        Color::Green
    }
}

/// Command line arguments
#[derive(Parser, Debug)]
#[command(name = "realizar-monitor")]
#[command(about = "Real-time btop-style monitoring TUI for realizar inference server")]
#[command(version)]
struct Args {
    /// Server URL to monitor
    #[arg(short, long, default_value = "http://127.0.0.1:8080")]
    url: String,

    /// Refresh rate in milliseconds
    #[arg(short, long, default_value = "100")]
    refresh_ms: u64,
}

/// Metrics from server /v1/metrics endpoint
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerMetrics {
    #[serde(default)]
    pub throughput_tok_per_sec: f64,
    #[serde(default)]
    pub latency_p50_ms: f64,
    #[serde(default)]
    pub latency_p95_ms: f64,
    #[serde(default)]
    pub latency_p99_ms: f64,
    #[serde(default)]
    pub gpu_memory_used_bytes: u64,
    #[serde(default)]
    pub gpu_memory_total_bytes: u64,
    #[serde(default)]
    pub gpu_utilization_percent: u32,
    #[serde(default)]
    pub cuda_path_active: bool,
    #[serde(default)]
    pub batch_size: usize,
    #[serde(default)]
    pub queue_depth: usize,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub total_requests: u64,
    #[serde(default)]
    pub uptime_secs: u64,
    #[serde(default)]
    pub model_name: String,
}

/// Time series data with circular buffer (inspired by simular)
#[derive(Debug, Clone)]
struct TimeSeries {
    data: VecDeque<f64>,
    capacity: usize,
}

impl TimeSeries {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, value: f64) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }

    fn as_u64_vec(&self) -> Vec<u64> {
        self.data.iter().map(|&v| v as u64).collect()
    }

    fn min(&self) -> Option<f64> {
        self.data.iter().cloned().reduce(f64::min)
    }

    fn max(&self) -> Option<f64> {
        self.data.iter().cloned().reduce(f64::max)
    }

    #[allow(dead_code)]
    fn last(&self) -> Option<f64> {
        self.data.back().copied()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    /// Compute trend direction (inspired by trueno-viz sparkline)
    fn trend(&self) -> &'static str {
        if self.data.len() < 5 {
            return "→"; // Not enough data
        }
        let recent: Vec<f64> = self.data.iter().rev().take(5).cloned().collect();
        let first_avg = (recent[3] + recent[4]) / 2.0;
        let last_avg = (recent[0] + recent[1]) / 2.0;

        let range = self.max().unwrap_or(1.0) - self.min().unwrap_or(0.0);
        let threshold = range * 0.05;

        if last_avg > first_avg + threshold {
            "↑"
        } else if last_avg < first_avg - threshold {
            "↓"
        } else {
            "→"
        }
    }

    /// Render sparkline as Unicode string (PARITY-109 QA-E08)
    /// Uses block characters to visualize time series data.
    #[allow(dead_code)]
    fn sparkline(&self, width: usize) -> String {
        const CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        let min = self.min().unwrap_or(0.0);
        let max = self.max().unwrap_or(1.0);
        let range = (max - min).max(0.001);

        self.data
            .iter()
            .rev()
            .take(width)
            .rev()
            .map(|&v| {
                let idx = ((v - min) / range * 7.0) as usize;
                CHARS[idx.min(7)]
            })
            .collect()
    }
}

/// Monitor application state (inspired by simular's DashboardState)
struct MonitorApp {
    /// Server URL
    url: String,
    /// Current metrics
    metrics: ServerMetrics,
    /// Throughput time series
    throughput_series: TimeSeries,
    /// Latency time series
    latency_series: TimeSeries,
    /// Peak throughput
    peak_throughput: f64,
    /// Monitor state machine (PARITY-108a)
    state: MonitorState,
    /// Last error message
    last_error: Option<String>,
    /// Should quit
    should_quit: bool,
    /// Start time
    #[allow(dead_code)]
    start_time: Instant,
}

impl MonitorApp {
    fn new(url: String) -> Self {
        Self {
            url,
            metrics: ServerMetrics::default(),
            throughput_series: TimeSeries::new(60),
            latency_series: TimeSeries::new(60),
            peak_throughput: 0.0,
            state: MonitorState::Disconnected,
            last_error: None,
            should_quit: false,
            start_time: Instant::now(),
        }
    }

    /// Fetch metrics from server
    fn fetch_metrics(&mut self) {
        // Only fetch if in Connected state (not Paused or Disconnected)
        if self.state == MonitorState::Paused {
            return;
        }

        let metrics_url = format!("{}/v1/metrics", self.url);

        match ureq::get(&metrics_url)
            .timeout(Duration::from_millis(500))
            .call()
        {
            Ok(response) => {
                match response.into_json::<ServerMetrics>() {
                    Ok(metrics) => {
                        // Update time series
                        self.throughput_series.push(metrics.throughput_tok_per_sec);
                        self.latency_series.push(metrics.latency_p50_ms);

                        // Track peak
                        if metrics.throughput_tok_per_sec > self.peak_throughput {
                            self.peak_throughput = metrics.throughput_tok_per_sec;
                        }

                        self.metrics = metrics;
                        self.state = MonitorState::Connected;
                        self.last_error = None;
                    },
                    Err(e) => {
                        self.state = MonitorState::Disconnected;
                        self.last_error = Some(format!("JSON parse error: {}", e));
                    },
                }
            },
            Err(e) => {
                self.state = MonitorState::Disconnected;
                self.last_error = Some(format!("Connection error: {}", e));
            },
        }
    }

    /// Format uptime as human-readable string
    fn format_uptime(&self) -> String {
        let secs = self.metrics.uptime_secs;
        if secs >= 3600 {
            format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
        } else if secs >= 60 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}s", secs)
        }
    }

    /// Format bytes as GB
    fn format_gb(bytes: u64) -> String {
        format!("{:.1} GB", bytes as f64 / 1e9)
    }

    /// Reset statistics
    fn reset(&mut self) {
        self.throughput_series = TimeSeries::new(60);
        self.latency_series = TimeSeries::new(60);
        self.peak_throughput = 0.0;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let mut app = MonitorApp::new(args.url);
    let refresh_duration = Duration::from_millis(args.refresh_ms);

    // Main loop
    let result = run_app(&mut terminal, &mut app, refresh_duration);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        eprintln!("Error: {err}");
    }

    Ok(())
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut MonitorApp,
    refresh: Duration,
) -> io::Result<()> {
    let mut last_fetch = Instant::now();

    loop {
        // Fetch metrics periodically
        if last_fetch.elapsed() >= refresh {
            app.fetch_metrics();
            last_fetch = Instant::now();
        }

        // Draw UI
        terminal.draw(|f| ui(f, app))?;

        // Handle input (non-blocking)
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            app.should_quit = true;
                        },
                        KeyCode::Char('r') => {
                            app.reset();
                        },
                        KeyCode::Char('p') => {
                            // Toggle between Connected and Paused states
                            app.state = match app.state {
                                MonitorState::Connected => MonitorState::Paused,
                                MonitorState::Paused => MonitorState::Connected,
                                MonitorState::Disconnected => MonitorState::Disconnected,
                            };
                        },
                        _ => {},
                    }
                }
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

fn ui(f: &mut Frame, app: &MonitorApp) {
    let area = f.area();

    // Main layout: left (60%) and right (40%) - inspired by simular
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Left side: throughput chart (60%) + GPU (25%) + controls (15%)
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(55),
            Constraint::Percentage(30),
            Constraint::Percentage(15),
        ])
        .split(main_chunks[0]);

    // Right side: metrics (60%) + system (40%)
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(main_chunks[1]);

    // ============================================================
    // LEFT PANEL: Throughput Chart
    // ============================================================
    let throughput_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Throughput ");

    let inner = throughput_block.inner(left_chunks[0]);
    f.render_widget(throughput_block, left_chunks[0]);

    let chart_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),    // Sparkline
            Constraint::Length(3), // Stats
        ])
        .split(inner);

    // Sparkline
    let data = app.throughput_series.as_u64_vec();
    let sparkline = RatatuiSparkline::default()
        .data(&data)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, chart_layout[0]);

    // Stats below sparkline
    let current = app.metrics.throughput_tok_per_sec;
    let peak = app.peak_throughput;
    let trend = app.throughput_series.trend();
    let trend_color = match trend {
        "↑" => Color::Green,
        "↓" => Color::Red,
        _ => Color::Yellow,
    };

    let stats_text = Paragraph::new(vec![
        Line::from(vec![
            Span::raw("Current: "),
            Span::styled(
                format!("{:.1} tok/s ", current),
                Style::default().fg(Color::Yellow).bold(),
            ),
            Span::styled(trend, Style::default().fg(trend_color).bold()),
            Span::raw("   Peak: "),
            Span::styled(
                format!("{:.1} tok/s", peak),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::raw("Samples: "),
            Span::raw(format!("{}", app.throughput_series.len())),
            Span::raw("   Min: "),
            Span::raw(format!("{:.1}", app.throughput_series.min().unwrap_or(0.0))),
            Span::raw("   Max: "),
            Span::raw(format!("{:.1}", app.throughput_series.max().unwrap_or(0.0))),
        ]),
    ]);
    f.render_widget(stats_text, chart_layout[1]);

    // ============================================================
    // LEFT PANEL: GPU Memory
    // ============================================================
    let gpu_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta))
        .title(" GPU Memory ");

    let gpu_inner = gpu_block.inner(left_chunks[1]);
    f.render_widget(gpu_block, left_chunks[1]);

    let gpu_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Memory bar
            Constraint::Length(1), // Memory text
            Constraint::Length(1), // Spacer
            Constraint::Min(2),    // Status lines
        ])
        .split(gpu_inner);

    // GPU Memory Gauge
    let gpu_percent = if app.metrics.gpu_memory_total_bytes > 0 {
        ((app.metrics.gpu_memory_used_bytes as f64 / app.metrics.gpu_memory_total_bytes as f64)
            * 100.0) as u16
    } else {
        0
    };

    // Use explicit gpu_color() function (PARITY-108c)
    let gauge_color = gpu_color(gpu_percent);

    let gpu_gauge = Gauge::default()
        .percent(gpu_percent)
        .gauge_style(Style::default().fg(gauge_color).bg(Color::DarkGray))
        .label(format!("{}%", gpu_percent));
    f.render_widget(gpu_gauge, gpu_layout[0]);

    // Memory text
    let used = MonitorApp::format_gb(app.metrics.gpu_memory_used_bytes);
    let total = MonitorApp::format_gb(app.metrics.gpu_memory_total_bytes);
    let mem_text =
        Paragraph::new(format!("{} / {}", used, total)).style(Style::default().fg(Color::White));
    f.render_widget(mem_text, gpu_layout[1]);

    // CUDA status and batch info
    let cuda_status = if app.metrics.cuda_path_active {
        Span::styled("● Active", Style::default().fg(Color::Green).bold())
    } else {
        Span::styled("○ Inactive", Style::default().fg(Color::Red))
    };

    let status_text = Paragraph::new(vec![
        Line::from(vec![Span::raw("CUDA: "), cuda_status]),
        Line::from(vec![
            Span::raw("Batch: "),
            Span::styled(
                format!("{}", app.metrics.batch_size),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw("   Queue: "),
            Span::styled(
                format!("{}", app.metrics.queue_depth),
                Style::default().fg(Color::Yellow),
            ),
        ]),
    ]);
    f.render_widget(status_text, gpu_layout[3]);

    // ============================================================
    // LEFT PANEL: Controls
    // ============================================================
    // Use MonitorState enum for status display (PARITY-108a)
    let (status_color, status_icon) = match app.state {
        MonitorState::Paused => (Color::Yellow, "⏸ Paused"),
        MonitorState::Connected => (Color::Green, "● Connected"),
        MonitorState::Disconnected => (Color::Red, "○ Disconnected"),
    };

    let controls = Paragraph::new(vec![Line::from(vec![
        Span::styled(status_icon, Style::default().fg(status_color)),
        Span::raw("   "),
        Span::styled("[q]", Style::default().fg(Color::Cyan)),
        Span::raw(" Quit  "),
        Span::styled("[r]", Style::default().fg(Color::Cyan)),
        Span::raw(" Reset  "),
        Span::styled("[p]", Style::default().fg(Color::Cyan)),
        Span::raw(" Pause"),
    ])])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" Controls "),
    );
    f.render_widget(controls, left_chunks[2]);

    // ============================================================
    // RIGHT PANEL: Metrics
    // ============================================================
    let latency_trend = app.latency_series.trend();
    let latency_color = match latency_trend {
        "↑" => Color::Red,   // Higher latency is bad
        "↓" => Color::Green, // Lower latency is good
        _ => Color::Yellow,
    };

    let metrics_text = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Throughput: "),
            Span::styled(
                format!("{:.1} tok/s", app.metrics.throughput_tok_per_sec),
                Style::default().fg(Color::Yellow).bold(),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Latency P50: "),
            Span::styled(
                format!("{:.1} ms ", app.metrics.latency_p50_ms),
                Style::default().fg(Color::Green),
            ),
            Span::styled(latency_trend, Style::default().fg(latency_color)),
        ]),
        Line::from(vec![
            Span::raw("  Latency P95: "),
            Span::styled(
                format!("{:.1} ms", app.metrics.latency_p95_ms),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Latency P99: "),
            Span::styled(
                format!("{:.1} ms", app.metrics.latency_p99_ms),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Requests: "),
            Span::styled(
                format!("{:>10}", format_number(app.metrics.total_requests)),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Tokens:   "),
            Span::styled(
                format!("{:>10}", format_number(app.metrics.total_tokens)),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Uptime:   "),
            Span::styled(
                format!("{:>10}", app.format_uptime()),
                Style::default().fg(Color::White),
            ),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue))
            .title(" Metrics "),
    );
    f.render_widget(metrics_text, right_chunks[0]);

    // ============================================================
    // RIGHT PANEL: System Info
    // ============================================================
    let model_name = if app.metrics.model_name.is_empty() {
        "N/A".to_string()
    } else {
        app.metrics.model_name.clone()
    };

    let error_line = if let Some(ref error) = app.last_error {
        Line::from(vec![
            Span::raw("  Error: "),
            Span::styled(error.clone(), Style::default().fg(Color::Red)),
        ])
    } else {
        Line::from(vec![
            Span::raw("  Status: "),
            Span::styled("OK", Style::default().fg(Color::Green)),
        ])
    };

    let system_text = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Model: "),
            Span::styled(model_name, Style::default().fg(Color::Cyan).bold()),
        ]),
        Line::from(vec![
            Span::raw("  Server: "),
            Span::styled(app.url.clone(), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
        error_line,
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" System "),
    );
    f.render_widget(system_text, right_chunks[1]);
}

/// Format number with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
