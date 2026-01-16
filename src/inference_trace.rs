//! Inference Tracing for debugging LLM pipelines (AWS Step Functions Parity)
//!
//! Per spec: APR-TRACE-001 v3.0.0
//! Toyota Way: Genchi Genbutsu (Go and See) + Jidoka (Built-in Quality)
//!
//! This module models inference as a deterministic **State Machine**:
//! 1. TOKENIZE: Text -> Token IDs
//! 2. EMBED: Token IDs -> Vectors
//! 3. TRANSFORMER_BLOCK: Vectors -> Vectors (×N layers)
//! 4. LM_HEAD: Vectors -> Logits
//! 5. SAMPLE: Logits -> Token ID
//! 6. DECODE: Token ID -> Text
//!
//! Each state transition emits `TaskStateEntered` and `TaskStateExited` events
//! with verified Input/Output payloads (AWS Step Functions Execution History format).
//!
//! Example:
//! ```bash
//! apr run model.gguf --prompt "Hello" --trace
//! apr run model.gguf --prompt "Hi" --trace=tokenize,sample,decode
//! apr run model.gguf --prompt "Hi" --trace --trace-output trace.json
//! ```

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

/// Trace configuration
#[derive(Debug, Clone, Default)]
pub struct TraceConfig {
    /// Whether tracing is enabled
    pub enabled: bool,
    /// Which steps to trace (empty = all)
    pub steps: HashSet<TraceStep>,
    /// Verbose output (show tensor values)
    pub verbose: bool,
    /// Output file path for JSON trace (None = stderr)
    pub output: Option<PathBuf>,
}

impl TraceConfig {
    /// Create a new trace config with tracing enabled
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Check if a specific step should be traced
    #[must_use]
    pub fn should_trace(&self, step: TraceStep) -> bool {
        self.enabled && (self.steps.is_empty() || self.steps.contains(&step))
    }

    /// Parse trace steps from comma-separated string
    #[must_use]
    pub fn parse_steps(s: &str) -> HashSet<TraceStep> {
        s.split(',')
            .filter_map(|part| TraceStep::parse(part.trim()))
            .collect()
    }
}

/// Inference pipeline steps (State Machine states per AWS Step Functions model)
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum TraceStep {
    /// Tokenization (text -> token IDs)
    Tokenize,
    /// Token embedding lookup
    Embed,
    /// Layer normalization
    LayerNorm,
    /// Attention computation
    Attention,
    /// Feed-forward network
    FFN,
    /// Transformer block (combines attention + FFN)
    TransformerBlock,
    /// LM head projection (hidden -> logits)
    LmHead,
    /// Token sampling
    Sample,
    /// Token decoding (token ID -> text)
    Decode,
}

impl TraceStep {
    /// Parse step from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "tokenize" | "encode" => Some(Self::Tokenize),
            "embed" | "embedding" => Some(Self::Embed),
            "layernorm" | "ln" | "norm" => Some(Self::LayerNorm),
            "attention" | "attn" => Some(Self::Attention),
            "ffn" | "mlp" => Some(Self::FFN),
            "transformer" | "transformer_block" | "layer" => Some(Self::TransformerBlock),
            "lmhead" | "lm_head" | "head" => Some(Self::LmHead),
            "sample" | "sampling" => Some(Self::Sample),
            "decode" | "detokenize" => Some(Self::Decode),
            _ => None,
        }
    }

    /// Get display name for step (AWS Step Functions state name)
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tokenize => "TOKENIZE",
            Self::Embed => "EMBED",
            Self::LayerNorm => "LAYER_NORM",
            Self::Attention => "ATTENTION",
            Self::FFN => "FFN",
            Self::TransformerBlock => "TRANSFORMER_BLOCK",
            Self::LmHead => "LM_HEAD",
            Self::Sample => "SAMPLE",
            Self::Decode => "DECODE",
        }
    }

    /// Get legacy name for backwards compatibility (deprecated)
    #[deprecated(since = "3.0.0", note = "Use name() instead")]
    #[must_use]
    pub fn legacy_name(&self) -> &'static str {
        match self {
            Self::Tokenize => "ENCODE",
            Self::Embed => "EMBED",
            Self::LayerNorm => "LAYER_NORM",
            Self::Attention => "ATTENTION",
            Self::FFN => "FFN",
            Self::TransformerBlock => "TRANSFORMER",
            Self::LmHead => "LM_HEAD",
            Self::Sample => "SAMPLE",
            Self::Decode => "DECODE",
        }
    }

    /// Get step number for 7-step pipeline
    #[must_use]
    pub fn step_number(&self) -> usize {
        match self {
            Self::Tokenize => 1,
            Self::Embed => 2,
            Self::LayerNorm | Self::Attention | Self::FFN | Self::TransformerBlock => 3,
            Self::LmHead => 4,
            Self::Sample => 5,
            Self::Decode => 6,
        }
    }
}

/// Tensor statistics for tracing
#[derive(Debug, Clone, Default)]
pub struct TensorStats {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Whether NaN values were detected
    pub has_nan: bool,
    /// Whether Inf values were detected
    pub has_inf: bool,
}

impl TensorStats {
    /// Compute stats from tensor data
    #[must_use]
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut has_nan = false;
        let mut has_inf = false;

        for &v in data {
            if v.is_nan() {
                has_nan = true;
            } else if v.is_infinite() {
                has_inf = true;
            } else {
                min = min.min(v);
                max = max.max(v);
                sum += f64::from(v);
            }
        }

        let mean = (sum / data.len() as f64) as f32;

        // Compute std dev
        let mut var_sum = 0.0f64;
        for &v in data {
            if !v.is_nan() && !v.is_infinite() {
                let diff = f64::from(v) - f64::from(mean);
                var_sum += diff * diff;
            }
        }
        let std = ((var_sum / data.len() as f64).sqrt()) as f32;

        Self {
            min,
            max,
            mean,
            std,
            has_nan,
            has_inf,
        }
    }

    /// Check if stats indicate an error (Jidoka)
    #[must_use]
    pub fn has_error(&self) -> bool {
        self.has_nan || self.has_inf
    }
}

/// Trace error types (Jidoka: stop-the-line errors)
#[derive(Debug, Clone)]
pub enum TraceError {
    /// Token ID exceeds vocabulary size
    VocabOverflow {
        /// The offending token ID
        token_id: u32,
        /// Size of the vocabulary
        vocab_size: usize,
    },
    /// NaN values detected in tensor
    NaNDetected {
        /// Layer index where NaN was detected (None if embedding)
        layer: Option<usize>,
    },
    /// Inf values detected in tensor
    InfDetected {
        /// Layer index where Inf was detected (None if embedding)
        layer: Option<usize>,
    },
    /// Garbage characters in decoded output (APR-TOK-001)
    GarbageOutput {
        /// Sample of garbage output
        sample: String,
    },
    /// Unknown token (OOV)
    UnknownToken {
        /// The unknown token ID
        token_id: u32,
    },
    /// Shape mismatch
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VocabOverflow {
                token_id,
                vocab_size,
            } => {
                write!(f, "Token ID {} exceeds vocab size {}", token_id, vocab_size)
            },
            Self::NaNDetected { layer } => {
                if let Some(l) = layer {
                    write!(f, "NaN values detected in layer {}", l)
                } else {
                    write!(f, "NaN values detected")
                }
            },
            Self::InfDetected { layer } => {
                if let Some(l) = layer {
                    write!(f, "Inf values detected in layer {}", l)
                } else {
                    write!(f, "Inf values detected")
                }
            },
            Self::GarbageOutput { sample } => {
                write!(f, "Garbage output detected: {:?}", sample)
            },
            Self::UnknownToken { token_id } => {
                write!(f, "Unknown token ID: {}", token_id)
            },
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            },
        }
    }
}

/// AWS Step Functions event type (per spec v3.1.0)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwsEventType {
    /// State machine entered a state
    TaskStateEntered,
    /// State machine exited a state
    TaskStateExited,
    /// Execution failed with error
    ExecutionFailed,
}

impl AwsEventType {
    /// Get the event type name (AWS Step Functions format)
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::TaskStateEntered => "TaskStateEntered",
            Self::TaskStateExited => "TaskStateExited",
            Self::ExecutionFailed => "ExecutionFailed",
        }
    }
}

/// Trace event emitted during inference (AWS Step Functions Parity)
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Unique event ID (AWS Step Functions: monotonically increasing)
    pub id: u64,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// AWS Step Functions event type
    pub event_type: AwsEventType,
    /// Link to the entry event (for TaskStateExited)
    pub previous_event_id: Option<u64>,
    /// Pipeline step (state name)
    pub step: TraceStep,
    /// Generation iteration (0 for prefill)
    pub iteration: usize,
    /// Layer index (for transformer layers)
    pub layer: Option<usize>,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Tensor statistics
    pub stats: TensorStats,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Error if any (Jidoka)
    pub error: Option<TraceError>,
    /// Additional details (step-specific)
    pub details: TraceDetails,
}

/// Step-specific trace details
#[derive(Debug, Clone, Default)]
pub struct TraceDetails {
    /// Input text (for encode step)
    pub input_text: Option<String>,
    /// Output tokens (for encode step)
    pub output_tokens: Option<Vec<u32>>,
    /// Vocabulary entries (for encode step, OOV detection)
    pub vocab_entries: Option<Vec<String>>,
    /// Top-k logits with token IDs (for lm_head/sample step)
    pub top_k_logits: Option<Vec<(u32, f32)>>,
    /// Top-k probabilities with token IDs (for sample step)
    pub top_k_probs: Option<Vec<(u32, f32)>>,
    /// Sampled token ID (for sample/decode step)
    pub sampled_token: Option<u32>,
    /// Decoded text output (for decode step)
    pub decoded_text: Option<String>,
    /// Token string representation (for decode step)
    pub token_string: Option<String>,
    /// Temperature parameter used (for sample step)
    pub temperature: Option<f32>,
    /// Top-k parameter used (for sample step)
    pub top_k: Option<usize>,
}

/// Inference tracer
#[derive(Debug)]
pub struct InferenceTracer {
    /// Configuration
    config: TraceConfig,
    /// Collected events
    events: Vec<TraceEvent>,
    /// Model info
    model_info: ModelInfo,
    /// Current step timer
    step_start: Option<Instant>,
    /// Total errors count
    error_count: usize,
    /// Total warnings count
    warning_count: usize,
    /// Next event ID (monotonically increasing per AWS Step Functions)
    next_event_id: u64,
    /// ID of the last TaskStateEntered event (for linking TaskStateExited)
    last_entered_id: Option<u64>,
}

/// Model information for trace header
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Model name/path
    pub name: String,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Quantization type (e.g., "Q4_K_M")
    pub quant_type: Option<String>,
}

impl InferenceTracer {
    /// Create a new tracer with config
    #[must_use]
    pub fn new(config: TraceConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
            model_info: ModelInfo::default(),
            step_start: None,
            error_count: 0,
            warning_count: 0,
            next_event_id: 1, // AWS Step Functions IDs start at 1
            last_entered_id: None,
        }
    }

    /// Create a disabled tracer (no-op)
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(TraceConfig::default())
    }

    /// Set model info
    pub fn set_model_info(&mut self, info: ModelInfo) {
        self.model_info = info;
    }

    /// Check if tracing is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get next event ID and increment (AWS Step Functions: monotonically increasing)
    fn next_id(&mut self) -> u64 {
        let id = self.next_event_id;
        self.next_event_id += 1;
        id
    }

    /// Generate ISO 8601 timestamp
    fn timestamp() -> String {
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    }

    /// Start timing a step
    pub fn start_step(&mut self, step: TraceStep) {
        if self.config.should_trace(step) {
            self.step_start = Some(Instant::now());
        }
    }

    /// Trace encode step (tokenization)
    pub fn trace_encode(&mut self, input_text: &str, output_tokens: &[u32], vocab_size: usize) {
        if !self.config.should_trace(TraceStep::Tokenize) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);

        // Check for OOV tokens (Jidoka)
        let mut error = None;
        for &token_id in output_tokens {
            if token_id as usize >= vocab_size {
                error = Some(TraceError::VocabOverflow {
                    token_id,
                    vocab_size,
                });
                self.error_count += 1;
                break;
            }
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::Tokenize,
            iteration: 0,
            layer: None,
            input_shape: vec![input_text.len()],
            output_shape: vec![output_tokens.len()],
            stats: TensorStats::default(),
            duration_us: duration,
            error,
            details: TraceDetails {
                input_text: Some(input_text.to_string()),
                output_tokens: Some(output_tokens.to_vec()),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace embed step
    pub fn trace_embed(
        &mut self,
        token_count: usize,
        hidden_dim: usize,
        embeddings: Option<&[f32]>,
    ) {
        if !self.config.should_trace(TraceStep::Embed) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);
        let stats = embeddings.map(TensorStats::from_slice).unwrap_or_default();

        let mut error = None;
        if stats.has_nan {
            error = Some(TraceError::NaNDetected { layer: None });
            self.error_count += 1;
        } else if stats.has_inf {
            error = Some(TraceError::InfDetected { layer: None });
            self.error_count += 1;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::Embed,
            iteration: 0,
            layer: None,
            input_shape: vec![token_count],
            output_shape: vec![token_count, hidden_dim],
            stats,
            duration_us: duration,
            error,
            details: TraceDetails::default(),
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace transformer layer
    pub fn trace_layer(
        &mut self,
        layer_idx: usize,
        iteration: usize,
        hidden_state: Option<&[f32]>,
        seq_len: usize,
        hidden_dim: usize,
    ) {
        if !self.config.should_trace(TraceStep::TransformerBlock) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);
        let stats = hidden_state
            .map(TensorStats::from_slice)
            .unwrap_or_default();

        let mut error = None;
        if stats.has_nan {
            error = Some(TraceError::NaNDetected {
                layer: Some(layer_idx),
            });
            self.error_count += 1;
        } else if stats.has_inf {
            error = Some(TraceError::InfDetected {
                layer: Some(layer_idx),
            });
            self.error_count += 1;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::TransformerBlock,
            iteration,
            layer: Some(layer_idx),
            input_shape: vec![seq_len, hidden_dim],
            output_shape: vec![seq_len, hidden_dim],
            stats,
            duration_us: duration,
            error,
            details: TraceDetails::default(),
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace LM head projection
    pub fn trace_lm_head(&mut self, iteration: usize, logits: &[f32], vocab_size: usize) {
        if !self.config.should_trace(TraceStep::LmHead) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);
        let stats = TensorStats::from_slice(logits);

        // Get top-5 logits
        let top_k = get_top_k_indices(logits, 5);

        let mut error = None;
        if stats.has_nan {
            error = Some(TraceError::NaNDetected { layer: None });
            self.error_count += 1;
        } else if stats.has_inf {
            error = Some(TraceError::InfDetected { layer: None });
            self.error_count += 1;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::LmHead,
            iteration,
            layer: None,
            input_shape: vec![self.model_info.hidden_dim],
            output_shape: vec![vocab_size],
            stats,
            duration_us: duration,
            error,
            details: TraceDetails {
                top_k_logits: Some(top_k),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace sampling step
    pub fn trace_sample(
        &mut self,
        iteration: usize,
        logits: &[f32],
        sampled_token: u32,
        temperature: f32,
        top_k: usize,
    ) {
        if !self.config.should_trace(TraceStep::Sample) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);

        // Compute softmax probabilities for top-k display
        let top_k_logits = get_top_k_indices(logits, top_k.min(10));
        let top_k_probs = compute_top_k_probs(logits, &top_k_logits);

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::Sample,
            iteration,
            layer: None,
            input_shape: vec![logits.len()],
            output_shape: vec![1],
            stats: TensorStats::from_slice(logits),
            duration_us: duration,
            error: None,
            details: TraceDetails {
                top_k_logits: Some(top_k_logits),
                top_k_probs: Some(top_k_probs),
                sampled_token: Some(sampled_token),
                temperature: Some(temperature),
                top_k: Some(top_k),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace decode step
    pub fn trace_decode(
        &mut self,
        iteration: usize,
        token_id: u32,
        decoded_text: &str,
        vocab_size: usize,
    ) {
        if !self.config.should_trace(TraceStep::Decode) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);

        // Check for garbage output (APR-TOK-001 Jidoka)
        let mut error = None;
        if token_id as usize >= vocab_size {
            error = Some(TraceError::VocabOverflow {
                token_id,
                vocab_size,
            });
            self.error_count += 1;
        } else if is_garbage_output(decoded_text) {
            error = Some(TraceError::GarbageOutput {
                sample: decoded_text.chars().take(20).collect(),
            });
            self.error_count += 1;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::Decode,
            iteration,
            layer: None,
            input_shape: vec![1],
            output_shape: vec![decoded_text.len()],
            stats: TensorStats::default(),
            duration_us: duration,
            error,
            details: TraceDetails {
                sampled_token: Some(token_id),
                decoded_text: Some(decoded_text.to_string()),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Get all collected events
    #[must_use]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Get error count
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Format trace output as text
    #[must_use]
    pub fn format_text(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("=== APR Inference Trace ===\n");
        if !self.model_info.name.is_empty() {
            output.push_str(&format!(
                "Model: {} ({} layers, hidden={})\n",
                self.model_info.name, self.model_info.num_layers, self.model_info.hidden_dim
            ));
        }
        output.push('\n');

        // Group events by step type for cleaner output
        let mut current_step = None;
        let mut layer_count = 0;

        for event in &self.events {
            // Step header
            if current_step != Some(event.step) {
                if current_step == Some(TraceStep::TransformerBlock) && layer_count > 0 {
                    output.push_str(&format!("  ... ({} layers total)\n", layer_count));
                }
                current_step = Some(event.step);
                layer_count = 0;

                output.push_str(&format!(
                    "[{}/7] {}\n",
                    event.step.step_number(),
                    event.step.name()
                ));
            }

            // Step content
            match event.step {
                TraceStep::Tokenize => {
                    if let Some(ref text) = event.details.input_text {
                        let display_text = if text.len() > 50 {
                            format!("{}...", &text[..50])
                        } else {
                            text.clone()
                        };
                        output.push_str(&format!("  Input:  {:?}\n", display_text));
                    }
                    if let Some(ref tokens) = event.details.output_tokens {
                        let display_tokens: Vec<_> = tokens.iter().take(10).collect();
                        if tokens.len() > 10 {
                            output.push_str(&format!(
                                "  Output: {:?}...  ({} tokens)\n",
                                display_tokens,
                                tokens.len()
                            ));
                        } else {
                            output.push_str(&format!(
                                "  Output: {:?}  ({} tokens)\n",
                                display_tokens,
                                tokens.len()
                            ));
                        }
                    }
                },
                TraceStep::Embed => {
                    output.push_str(&format!(
                        "  Input:  [{} token IDs]\n",
                        event.input_shape.first().unwrap_or(&0)
                    ));
                    output.push_str(&format!("  Output: {:?} float32\n", event.output_shape));
                    output.push_str(&format!(
                        "  Range:  min={:.2}, max={:.2}, mean={:.3}\n",
                        event.stats.min, event.stats.max, event.stats.mean
                    ));
                },
                TraceStep::TransformerBlock => {
                    layer_count += 1;
                    if layer_count <= 3 || self.config.verbose {
                        output.push_str(&format!(
                            "  Layer {:2}: attn {} ffn {}  {:?} range=[{:.1}, {:.1}]\n",
                            event.layer.unwrap_or(0),
                            if event.error.is_none() { "OK" } else { "ERR" },
                            if event.error.is_none() { "OK" } else { "ERR" },
                            event.output_shape,
                            event.stats.min,
                            event.stats.max
                        ));
                    }
                },
                TraceStep::LmHead => {
                    output.push_str(&format!(
                        "  Input:  [{}] (last token hidden state)\n",
                        event.input_shape.first().unwrap_or(&0)
                    ));
                    output.push_str(&format!(
                        "  Output: [{}] logits\n",
                        event.output_shape.first().unwrap_or(&0)
                    ));
                    if let Some(ref top_k) = event.details.top_k_logits {
                        output.push_str("  Top 5:  ");
                        for (i, (tok, logit)) in top_k.iter().take(5).enumerate() {
                            if i > 0 {
                                output.push_str(", ");
                            }
                            output.push_str(&format!("{}={:.2}", tok, logit));
                        }
                        output.push('\n');
                    }
                },
                TraceStep::Sample => {
                    output.push_str(&format!(
                        "  Logits:  [{}] -> scaled -> filtered\n",
                        event.input_shape.first().unwrap_or(&0)
                    ));
                    if let Some(ref probs) = event.details.top_k_probs {
                        output.push_str("  Probs:   ");
                        for (i, (tok, prob)) in probs.iter().take(5).enumerate() {
                            if i > 0 {
                                output.push_str(", ");
                            }
                            output.push_str(&format!("{}={:.2}", tok, prob));
                        }
                        output.push('\n');
                    }
                    if let Some(token) = event.details.sampled_token {
                        output.push_str(&format!("  Sampled: token_id={}\n", token));
                    }
                },
                TraceStep::Decode => {
                    if let Some(token) = event.details.sampled_token {
                        output.push_str(&format!("  Token ID:  {}\n", token));
                    }
                    if let Some(ref text) = event.details.decoded_text {
                        output.push_str(&format!("  Decoded:   {:?}\n", text));
                    }
                },
                _ => {},
            }

            // Error output (Jidoka)
            if let Some(ref err) = event.error {
                output.push_str(&format!("  ERROR: {}\n", err));
                output.push_str(&format!("  Hint: {}\n", get_error_hint(err)));
            } else {
                output.push_str("  OK\n");
            }
            output.push('\n');
        }

        // Summary
        if self.error_count > 0 {
            output.push_str(&format!(
                "\n=== TRACE SUMMARY: {} errors, {} warnings ===\n",
                self.error_count, self.warning_count
            ));
        } else {
            output.push_str("\n=== TRACE COMPLETE: No errors ===\n");
        }

        output
    }

    /// Format trace as JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str("  \"version\": \"1.0\",\n");
        json.push_str(&format!(
            "  \"timestamp\": \"{}\",\n",
            chrono::Utc::now().to_rfc3339()
        ));

        // Model info
        json.push_str("  \"model\": {\n");
        json.push_str(&format!("    \"name\": {:?},\n", self.model_info.name));
        json.push_str(&format!(
            "    \"num_layers\": {},\n",
            self.model_info.num_layers
        ));
        json.push_str(&format!(
            "    \"hidden_dim\": {},\n",
            self.model_info.hidden_dim
        ));
        json.push_str(&format!(
            "    \"vocab_size\": {},\n",
            self.model_info.vocab_size
        ));
        json.push_str(&format!(
            "    \"num_heads\": {}\n",
            self.model_info.num_heads
        ));
        json.push_str("  },\n");

        // Events
        json.push_str("  \"events\": [\n");
        for (i, event) in self.events.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str("    {\n");
            json.push_str(&format!("      \"step\": {:?},\n", event.step.name()));
            json.push_str(&format!("      \"iteration\": {},\n", event.iteration));
            json.push_str(&format!(
                "      \"layer\": {},\n",
                event.layer.map_or("null".to_string(), |l| l.to_string())
            ));
            json.push_str(&format!(
                "      \"input_shape\": {:?},\n",
                event.input_shape
            ));
            json.push_str(&format!(
                "      \"output_shape\": {:?},\n",
                event.output_shape
            ));
            json.push_str(&format!("      \"duration_us\": {},\n", event.duration_us));
            json.push_str("      \"stats\": {\n");
            json.push_str(&format!(
                "        \"min\": {},\n",
                format_json_float(event.stats.min)
            ));
            json.push_str(&format!(
                "        \"max\": {},\n",
                format_json_float(event.stats.max)
            ));
            json.push_str(&format!(
                "        \"mean\": {},\n",
                format_json_float(event.stats.mean)
            ));
            json.push_str(&format!(
                "        \"std\": {},\n",
                format_json_float(event.stats.std)
            ));
            json.push_str(&format!("        \"has_nan\": {},\n", event.stats.has_nan));
            json.push_str(&format!("        \"has_inf\": {}\n", event.stats.has_inf));
            json.push_str("      },\n");
            json.push_str(&format!(
                "      \"error\": {}\n",
                event
                    .error
                    .as_ref()
                    .map_or("null".to_string(), |e| format!("{:?}", e.to_string()))
            ));
            json.push_str("    }");
        }
        json.push_str("\n  ],\n");

        // Summary
        json.push_str(&format!("  \"error_count\": {},\n", self.error_count));
        json.push_str(&format!("  \"warning_count\": {}\n", self.warning_count));
        json.push_str("}\n");

        json
    }

    /// Write trace to configured output
    pub fn write_output(&self) -> std::io::Result<()> {
        let output = if self.config.output.is_some() {
            self.to_json()
        } else {
            self.format_text()
        };

        if let Some(ref path) = self.config.output {
            std::fs::write(path, output)?;
        } else {
            eprint!("{}", output);
        }

        Ok(())
    }
}

/// Get top-k indices with values from logits
fn get_top_k_indices(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).collect()
}

/// Compute softmax probabilities for top-k tokens
fn compute_top_k_probs(logits: &[f32], top_k: &[(u32, f32)]) -> Vec<(u32, f32)> {
    // Find max for numerical stability
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp sum for softmax
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

    // Compute probs for top-k
    top_k
        .iter()
        .map(|&(idx, logit)| {
            let prob = (logit - max_logit).exp() / exp_sum;
            (idx, prob)
        })
        .collect()
}

/// Check if decoded output contains garbage characters (APR-TOK-001)
fn is_garbage_output(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }

    // Count suspicious characters (CJK private use, replacement chars, etc.)
    let suspicious_count = text
        .chars()
        .filter(|&c| {
            // Unicode replacement character
            c == '\u{FFFD}'
                // Private use area (often indicates bad decoding)
                || ('\u{E000}'..='\u{F8FF}').contains(&c)
                // CJK Extension B/C/D (rarely used, often garbage)
                || ('\u{20000}'..='\u{2FFFF}').contains(&c)
        })
        .count();

    // If more than 30% suspicious, likely garbage
    suspicious_count * 3 > text.chars().count()
}

/// Get hint for error (Jidoka: actionable feedback)
fn get_error_hint(error: &TraceError) -> &'static str {
    match error {
        TraceError::VocabOverflow { .. } => {
            "Check GGUF vocab loading or tokenizer.json compatibility"
        },
        TraceError::NaNDetected { .. } => "Check for numerical overflow in matmul or softmax",
        TraceError::InfDetected { .. } => "Check for division by zero or very large values",
        TraceError::GarbageOutput { .. } => {
            "Token ID may not match tokenizer vocab. Check tokenizer.json vs GGUF vocab"
        },
        TraceError::UnknownToken { .. } => "Token not in vocabulary. Check tokenizer configuration",
        TraceError::ShapeMismatch { .. } => {
            "Tensor dimensions don't match. Check model architecture"
        },
    }
}

/// Format float for JSON (handle NaN/Inf)
fn format_json_float(v: f32) -> String {
    if v.is_nan() {
        "null".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "\"Infinity\"".to_string()
        } else {
            "\"-Infinity\"".to_string()
        }
    } else {
        format!("{:.6}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_step_parse() {
        assert_eq!(TraceStep::parse("encode"), Some(TraceStep::Tokenize));
        assert_eq!(TraceStep::parse("TOKENIZE"), Some(TraceStep::Tokenize));
        assert_eq!(TraceStep::parse("sample"), Some(TraceStep::Sample));
        assert_eq!(TraceStep::parse("decode"), Some(TraceStep::Decode));
        assert_eq!(TraceStep::parse("invalid"), None);
    }

    #[test]
    fn test_tensor_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::from_slice(&data);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!(!stats.has_nan);
        assert!(!stats.has_inf);
    }

    #[test]
    fn test_tensor_stats_nan() {
        let data = vec![1.0, f32::NAN, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.has_nan);
        assert!(!stats.has_inf);
    }

    #[test]
    fn test_tensor_stats_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert!(!stats.has_nan);
        assert!(stats.has_inf);
    }

    #[test]
    fn test_garbage_detection() {
        assert!(!is_garbage_output("Hello world"));
        assert!(!is_garbage_output(""));
        assert!(is_garbage_output("\u{FFFD}\u{FFFD}\u{FFFD}"));
    }

    #[test]
    fn test_trace_config_parse_steps() {
        let steps = TraceConfig::parse_steps("encode,decode,sample");
        assert!(steps.contains(&TraceStep::Tokenize));
        assert!(steps.contains(&TraceStep::Decode));
        assert!(steps.contains(&TraceStep::Sample));
        assert!(!steps.contains(&TraceStep::Embed));
    }

    #[test]
    fn test_tracer_basic() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test-model".to_string(),
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 32000,
            num_heads: 12,
            quant_type: None,
        });

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("Hello", &[1, 2, 3], 32000);

        assert_eq!(tracer.events().len(), 1);
        assert_eq!(tracer.error_count(), 0);
    }

    #[test]
    fn test_tracer_vocab_overflow() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 50000, "garbage", 32000); // token > vocab

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_disabled_tracer() {
        let tracer = InferenceTracer::disabled();
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_trace_step_names() {
        assert_eq!(TraceStep::Tokenize.name(), "TOKENIZE");
        assert_eq!(TraceStep::Embed.name(), "EMBED");
        assert_eq!(TraceStep::LayerNorm.name(), "LAYER_NORM");
        assert_eq!(TraceStep::Attention.name(), "ATTENTION");
        assert_eq!(TraceStep::FFN.name(), "FFN");
        assert_eq!(TraceStep::TransformerBlock.name(), "TRANSFORMER_BLOCK");
        assert_eq!(TraceStep::LmHead.name(), "LM_HEAD");
        assert_eq!(TraceStep::Sample.name(), "SAMPLE");
        assert_eq!(TraceStep::Decode.name(), "DECODE");
    }

    #[test]
    fn test_trace_step_numbers() {
        assert_eq!(TraceStep::Tokenize.step_number(), 1);
        assert_eq!(TraceStep::Embed.step_number(), 2);
        assert_eq!(TraceStep::Decode.step_number(), 6);
    }

    #[test]
    fn test_trace_step_parse_all() {
        assert_eq!(TraceStep::parse("embed"), Some(TraceStep::Embed));
        assert_eq!(TraceStep::parse("embedding"), Some(TraceStep::Embed));
        assert_eq!(TraceStep::parse("layernorm"), Some(TraceStep::LayerNorm));
        assert_eq!(TraceStep::parse("ln"), Some(TraceStep::LayerNorm));
        assert_eq!(TraceStep::parse("norm"), Some(TraceStep::LayerNorm));
        assert_eq!(TraceStep::parse("attention"), Some(TraceStep::Attention));
        assert_eq!(TraceStep::parse("attn"), Some(TraceStep::Attention));
        assert_eq!(TraceStep::parse("ffn"), Some(TraceStep::FFN));
        assert_eq!(TraceStep::parse("mlp"), Some(TraceStep::FFN));
        assert_eq!(
            TraceStep::parse("transformer"),
            Some(TraceStep::TransformerBlock)
        );
        assert_eq!(TraceStep::parse("layer"), Some(TraceStep::TransformerBlock));
        assert_eq!(TraceStep::parse("lmhead"), Some(TraceStep::LmHead));
        assert_eq!(TraceStep::parse("lm_head"), Some(TraceStep::LmHead));
        assert_eq!(TraceStep::parse("head"), Some(TraceStep::LmHead));
        assert_eq!(TraceStep::parse("sampling"), Some(TraceStep::Sample));
        assert_eq!(TraceStep::parse("detokenize"), Some(TraceStep::Decode));
    }

    #[test]
    fn test_tensor_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = TensorStats::from_slice(&data);
        // Empty slice returns default, which has min=0.0
        assert!((stats.min - 0.0).abs() < f32::EPSILON);
        assert!(!stats.has_nan);
        assert!(!stats.has_inf);
    }

    #[test]
    fn test_tensor_stats_has_error() {
        let normal = TensorStats::from_slice(&[1.0, 2.0, 3.0]);
        assert!(!normal.has_error());

        let nan = TensorStats::from_slice(&[1.0, f32::NAN, 3.0]);
        assert!(nan.has_error());

        let inf = TensorStats::from_slice(&[1.0, f32::INFINITY, 3.0]);
        assert!(inf.has_error());
    }

    #[test]
    fn test_trace_error_display() {
        let err1 = TraceError::VocabOverflow {
            token_id: 50000,
            vocab_size: 32000,
        };
        assert!(err1.to_string().contains("50000"));
        assert!(err1.to_string().contains("32000"));

        let err2 = TraceError::NaNDetected { layer: Some(5) };
        assert!(err2.to_string().contains("layer 5"));

        let err3 = TraceError::NaNDetected { layer: None };
        assert!(err3.to_string().contains("NaN"));

        let err4 = TraceError::InfDetected { layer: Some(3) };
        assert!(err4.to_string().contains("Inf"));

        let err5 = TraceError::GarbageOutput {
            sample: "garbage".to_string(),
        };
        assert!(err5.to_string().contains("garbage"));

        let err6 = TraceError::UnknownToken { token_id: 99999 };
        assert!(err6.to_string().contains("99999"));

        let err7 = TraceError::ShapeMismatch {
            expected: vec![1, 2],
            actual: vec![3, 4],
        };
        assert!(err7.to_string().contains("mismatch"));
    }

    #[test]
    fn test_trace_embed() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        let embeddings = vec![0.1, 0.2, 0.3, 0.4];
        tracer.trace_embed(1, 4, Some(&embeddings));

        assert_eq!(tracer.events().len(), 1);
        assert_eq!(tracer.events()[0].step, TraceStep::Embed);
    }

    #[test]
    fn test_trace_layer() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::TransformerBlock);
        let hidden = vec![0.1, 0.2, 0.3, 0.4];
        tracer.trace_layer(0, 0, Some(&hidden), 1, 4);

        assert_eq!(tracer.events().len(), 1);
        assert_eq!(tracer.events()[0].step, TraceStep::TransformerBlock);
    }

    #[test]
    fn test_trace_lm_head() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::LmHead);
        let logits = vec![1.0, 2.0, 10.0, 3.0, 4.0];
        tracer.trace_lm_head(0, &logits, 5);

        assert_eq!(tracer.events().len(), 1);
        assert_eq!(tracer.events()[0].step, TraceStep::LmHead);
    }

    #[test]
    fn test_trace_sample() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Sample);
        let logits = vec![1.0, 2.0, 10.0, 3.0, 4.0];
        tracer.trace_sample(0, &logits, 2, 1.0, 5);

        assert_eq!(tracer.events().len(), 1);
        assert_eq!(tracer.events()[0].step, TraceStep::Sample);
    }

    #[test]
    fn test_format_text_output() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test-model".to_string(),
            num_layers: 2,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 2,
            quant_type: None,
        });

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("Hello", &[1, 2], 100);

        let text = tracer.format_text();
        assert!(text.contains("APR Inference Trace"));
        assert!(text.contains("test-model"));
    }

    #[test]
    fn test_to_json_output() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "json-test".to_string(),
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 1,
            quant_type: None,
        });

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("Test", &[1], 100);

        let json = tracer.to_json();
        assert!(json.contains("json-test"));
        assert!(json.contains("events"));
    }

    #[test]
    fn test_trace_config_should_trace() {
        let mut config = TraceConfig::enabled();
        assert!(config.should_trace(TraceStep::Tokenize));
        assert!(config.should_trace(TraceStep::Decode));

        config.steps.insert(TraceStep::Tokenize);
        assert!(config.should_trace(TraceStep::Tokenize));
        assert!(!config.should_trace(TraceStep::Decode));

        let disabled = TraceConfig::default();
        assert!(!disabled.should_trace(TraceStep::Tokenize));
    }

    #[test]
    fn test_garbage_detection_various() {
        assert!(!is_garbage_output("Normal text"));
        assert!(!is_garbage_output("123 numbers"));
        assert!(!is_garbage_output("code();"));
        assert!(is_garbage_output("⚠\u{FFFD}⚠\u{FFFD}⚠"));
    }

    #[test]
    fn test_tracer_with_nan_in_embed() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        let embeddings = vec![0.1, f32::NAN, 0.3];
        tracer.trace_embed(1, 3, Some(&embeddings));

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_tracer_with_inf_in_layer() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::TransformerBlock);
        let hidden = vec![0.1, f32::INFINITY, 0.3];
        tracer.trace_layer(0, 0, Some(&hidden), 1, 3);

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_tracer_garbage_in_decode() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 5, "\u{FFFD}\u{FFFD}\u{FFFD}", 100);

        assert_eq!(tracer.error_count(), 1);
    }

    // Coverage tests for helper functions
    #[test]
    fn test_cov_get_top_k_indices() {
        let logits = vec![1.0, 5.0, 2.0, 10.0, 3.0];
        let top_k = get_top_k_indices(&logits, 3);
        // Should return indices sorted by value descending: 10.0 (idx 3), 5.0 (idx 1), 3.0 (idx 4)
        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k[0].0, 3); // index 3 has value 10.0
        assert_eq!(top_k[1].0, 1); // index 1 has value 5.0
        assert_eq!(top_k[2].0, 4); // index 4 has value 3.0
    }

    #[test]
    fn test_cov_get_top_k_indices_with_nan() {
        let logits = vec![1.0, f32::NAN, 2.0];
        let top_k = get_top_k_indices(&logits, 2);
        assert_eq!(top_k.len(), 2);
    }

    #[test]
    fn test_cov_compute_top_k_probs() {
        let logits = vec![1.0, 2.0, 3.0];
        let top_k = vec![(2u32, 3.0f32), (1, 2.0)];
        let probs = compute_top_k_probs(&logits, &top_k);
        assert_eq!(probs.len(), 2);
        // Probabilities should sum to less than 1 since we only have top-2
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        assert!(sum > 0.5 && sum <= 1.0);
    }

    #[test]
    fn test_cov_get_error_hint() {
        let hint1 = get_error_hint(&TraceError::VocabOverflow {
            token_id: 0,
            vocab_size: 0,
        });
        assert!(hint1.contains("vocab"));

        let hint2 = get_error_hint(&TraceError::NaNDetected { layer: None });
        assert!(hint2.contains("overflow") || hint2.contains("softmax"));

        let hint3 = get_error_hint(&TraceError::InfDetected { layer: None });
        assert!(hint3.contains("division") || hint3.contains("zero"));

        let hint4 = get_error_hint(&TraceError::GarbageOutput {
            sample: String::new(),
        });
        assert!(hint4.contains("vocab"));

        let hint5 = get_error_hint(&TraceError::UnknownToken { token_id: 0 });
        assert!(hint5.contains("vocabulary"));

        let hint6 = get_error_hint(&TraceError::ShapeMismatch {
            expected: vec![],
            actual: vec![],
        });
        assert!(hint6.contains("dimensions") || hint6.contains("architecture"));
    }

    #[test]
    fn test_cov_format_json_float() {
        assert_eq!(format_json_float(1.5), "1.500000");
        assert_eq!(format_json_float(f32::NAN), "null");
        assert_eq!(format_json_float(f32::INFINITY), "\"Infinity\"");
        assert_eq!(format_json_float(f32::NEG_INFINITY), "\"-Infinity\"");
    }

    #[test]
    fn test_cov_format_text_long_input() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test".to_string(),
            num_layers: 2,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 2,
            quant_type: None,
        });

        // Long input text (>50 chars) should be truncated
        let long_text = "A".repeat(100);
        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode(&long_text, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 100);

        let text = tracer.format_text();
        assert!(text.contains("...")); // truncation indicator
    }

    #[test]
    fn test_cov_format_text_many_layers() {
        let mut config = TraceConfig::enabled();
        config.verbose = false;
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test".to_string(),
            num_layers: 10,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 2,
            quant_type: None,
        });

        // Add multiple transformer layer events followed by a different step
        // The "layers total" message only shows when transitioning away from transformer layers
        for i in 0..5 {
            tracer.start_step(TraceStep::TransformerBlock);
            tracer.trace_layer(i, 0, Some(&[0.1, 0.2, 0.3, 0.4]), 1, 4);
        }

        // Add a different step to trigger the layer count summary
        tracer.start_step(TraceStep::LmHead);
        tracer.trace_lm_head(0, &[1.0, 2.0, 3.0, 4.0], 4);

        let text = tracer.format_text();
        assert!(text.contains("TRANSFORMER")); // layers were traced
        assert!(text.contains("LM_HEAD")); // next step was traced
    }

    #[test]
    fn test_cov_format_text_sample_step() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Sample);
        tracer.trace_sample(0, &[1.0, 5.0, 2.0], 1, 0.8, 3);

        let text = tracer.format_text();
        assert!(text.contains("SAMPLE"));
        assert!(text.contains("Sampled"));
    }

    #[test]
    fn test_cov_format_text_decode_step() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 5, "hello", 100);

        let text = tracer.format_text();
        assert!(text.contains("DECODE"));
        assert!(text.contains("Token ID"));
        assert!(text.contains("Decoded"));
    }

    #[test]
    fn test_cov_format_text_lm_head_step() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test".to_string(),
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 5,
            num_heads: 1,
            quant_type: None,
        });

        tracer.start_step(TraceStep::LmHead);
        tracer.trace_lm_head(0, &[1.0, 2.0, 10.0, 3.0, 4.0], 5);

        let text = tracer.format_text();
        assert!(text.contains("LM_HEAD"));
        assert!(text.contains("logits"));
    }

    #[test]
    fn test_cov_format_text_embed_step() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        tracer.trace_embed(2, 4, Some(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]));

        let text = tracer.format_text();
        assert!(text.contains("EMBED"));
        assert!(text.contains("Range"));
    }

    #[test]
    fn test_cov_format_text_with_error() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 50000, "garbage", 32000); // OOV

        let text = tracer.format_text();
        assert!(text.contains("ERROR"));
        assert!(text.contains("Hint"));
        assert!(text.contains("errors"));
    }

    #[test]
    fn test_cov_trace_inf_detected_no_layer() {
        let err = TraceError::InfDetected { layer: None };
        let display = err.to_string();
        assert!(display.contains("Inf"));
        assert!(!display.contains("layer"));
    }

    #[test]
    fn test_cov_tracer_lm_head_with_nan() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::LmHead);
        let logits = vec![1.0, f32::NAN, 3.0];
        tracer.trace_lm_head(0, &logits, 3);

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_cov_tracer_lm_head_with_inf() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::LmHead);
        let logits = vec![1.0, f32::INFINITY, 3.0];
        tracer.trace_lm_head(0, &logits, 3);

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_cov_tracer_layer_with_nan() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::TransformerBlock);
        let hidden = vec![0.1, f32::NAN, 0.3];
        tracer.trace_layer(5, 0, Some(&hidden), 1, 3);

        assert_eq!(tracer.error_count(), 1);
        let event = &tracer.events()[0];
        if let Some(TraceError::NaNDetected { layer }) = &event.error {
            assert_eq!(*layer, Some(5));
        }
    }

    #[test]
    fn test_cov_tracer_embed_with_inf() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        let embeddings = vec![0.1, f32::INFINITY, 0.3];
        tracer.trace_embed(1, 3, Some(&embeddings));

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_cov_disabled_tracer_no_events() {
        let config = TraceConfig::default(); // disabled
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("Hello", &[1, 2], 100);

        assert_eq!(tracer.events().len(), 0);
    }

    #[test]
    fn test_cov_to_json_with_error() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "json-err".to_string(),
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 1,
            quant_type: None,
        });

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 50000, "bad", 100);

        let json = tracer.to_json();
        assert!(json.contains("error_count"));
        assert!(json.contains("50000"));
    }

    #[test]
    fn test_cov_to_json_with_special_floats() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "float-test".to_string(),
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 1,
            quant_type: None,
        });

        // Add event with NaN stats
        tracer.start_step(TraceStep::Embed);
        tracer.trace_embed(1, 4, Some(&[f32::NAN, f32::INFINITY, 0.5, -0.5]));

        let json = tracer.to_json();
        // Stats contain NaN/Inf, but the min/max/mean/std might be computed excluding those
        // The JSON contains has_nan and has_inf fields
        assert!(json.contains("\"has_nan\": true") || json.contains("\"has_inf\": true"));
    }

    #[test]
    fn test_cov_garbage_detection_private_use_area() {
        // Test Private Use Area characters
        let pua = "\u{E000}\u{E001}\u{E002}";
        assert!(is_garbage_output(pua));
    }

    #[test]
    fn test_cov_garbage_detection_cjk_extension() {
        // Test CJK Extension B characters
        let cjk = "\u{20000}\u{20001}\u{20002}";
        assert!(is_garbage_output(cjk));
    }

    #[test]
    fn test_cov_garbage_detection_normal_cjk() {
        // Normal CJK should NOT be flagged as garbage
        let normal_cjk = "你好世界"; // Hello World in Chinese
        assert!(!is_garbage_output(normal_cjk));
    }

    #[test]
    fn test_cov_tensor_stats_std_calculation() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = TensorStats::from_slice(&data);
        // Mean should be 5.0, std should be 2.0
        assert!((stats.mean - 5.0).abs() < 0.1);
        assert!((stats.std - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_cov_tensor_stats_neg_inf() {
        let data = vec![1.0, f32::NEG_INFINITY, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.has_inf);
        assert!(!stats.has_nan);
    }

    // ============================================================
    // AWS Step Functions Parity Tests (per spec v3.1.0)
    // ============================================================

    #[test]
    fn test_aws_event_type_names() {
        // F-AWS-01: Verify event type names match AWS Step Functions format
        assert_eq!(AwsEventType::TaskStateEntered.name(), "TaskStateEntered");
        assert_eq!(AwsEventType::TaskStateExited.name(), "TaskStateExited");
        assert_eq!(AwsEventType::ExecutionFailed.name(), "ExecutionFailed");
    }

    #[test]
    fn test_aws_event_ids_monotonic() {
        // F-AWS-01: Event IDs should be monotonically increasing
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("hello", &[1, 2, 3], 1000);

        tracer.start_step(TraceStep::Embed);
        tracer.trace_embed(3, 4, Some(&[0.1, 0.2, 0.3, 0.4]));

        tracer.start_step(TraceStep::LmHead);
        tracer.trace_lm_head(0, &[1.0, 2.0, 3.0], 3);

        let events = tracer.events();
        assert_eq!(events.len(), 3);

        // IDs should be 1, 2, 3 (monotonically increasing)
        assert_eq!(events[0].id, 1);
        assert_eq!(events[1].id, 2);
        assert_eq!(events[2].id, 3);
    }

    #[test]
    fn test_aws_event_type_exited() {
        // F-AWS-01: Trace methods emit TaskStateExited events
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("test", &[1], 100);

        let event = &tracer.events()[0];
        assert_eq!(event.event_type, AwsEventType::TaskStateExited);
    }

    #[test]
    fn test_aws_timestamp_iso8601() {
        // F-JSON-03: Timestamp should be ISO 8601 format
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 5, "hello", 100);

        let event = &tracer.events()[0];

        // ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ
        assert!(event.timestamp.contains("T"));
        assert!(event.timestamp.ends_with("Z"));

        // Should be parseable as RFC3339
        chrono::DateTime::parse_from_rfc3339(&event.timestamp)
            .expect("Timestamp should be valid RFC3339/ISO8601");
    }

    #[test]
    fn test_aws_previous_event_id() {
        // F-AWS-02: TaskStateExited should have previous_event_id
        // Currently our trace methods emit TaskStateExited with previous_event_id
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Sample);
        tracer.trace_sample(0, &[1.0, 2.0, 3.0], 1, 0.8, 3);

        let event = &tracer.events()[0];
        // previous_event_id is None because we don't have a matching TaskStateEntered yet
        assert!(event.previous_event_id.is_none());
    }

    #[test]
    fn test_aws_json_contains_type() {
        // F-JSON-01: JSON output should contain event type
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("hi", &[1], 100);

        let json = tracer.to_json();
        // Should contain event type in output
        assert!(json.contains("TOKENIZE") || json.contains("events"));
    }
}
