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
    /// GPU kernel launch (PTX-level tracing, GH-219)
    KernelLaunch,
    /// Compute brick profiling breakdown (trueno BrickProfiler)
    BrickProfile,
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
            "kernel" | "kernel_launch" | "ptx" | "cuda" => Some(Self::KernelLaunch),
            "brick" | "brick_profile" | "profiler" | "bricks" => Some(Self::BrickProfile),
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
            Self::KernelLaunch => "KERNEL_LAUNCH",
            Self::BrickProfile => "BRICK_PROFILE",
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
            Self::KernelLaunch => "KERNEL_LAUNCH",
            Self::BrickProfile => "BRICK_PROFILE",
        }
    }

    /// Get step number for 8-step pipeline (7 = kernel-level)
    #[must_use]
    pub fn step_number(&self) -> usize {
        match self {
            Self::Tokenize => 1,
            Self::Embed => 2,
            Self::LayerNorm | Self::Attention | Self::FFN | Self::TransformerBlock => 3,
            Self::LmHead => 4,
            Self::Sample => 5,
            Self::Decode => 6,
            Self::KernelLaunch => 7,
            Self::BrickProfile => 8,
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
    /// Execution failed (F-JID-01: Jidoka)
    ExecutionFailed {
        /// Cause of failure
        cause: String,
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
            Self::ExecutionFailed { cause } => {
                write!(f, "Execution failed: {}", cause)
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
    /// Cause of failure (F-AWS-05: required for ExecutionFailed events)
    pub cause: Option<String>,
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
    /// Kernel name (for kernel_launch step, GH-219)
    pub kernel_name: Option<String>,
    /// Grid dimensions [x, y, z] (for kernel_launch step)
    pub grid_dims: Option<[u32; 3]>,
    /// Block dimensions [x, y, z] (for kernel_launch step)
    pub block_dims: Option<[u32; 3]>,
    /// Shared memory bytes (for kernel_launch step)
    pub shared_mem_bytes: Option<u32>,
    /// Layer index within the transformer (for kernel_launch context)
    pub kernel_layer: Option<usize>,
    /// Dispatch strategy: "grid_y" or "register_unroll" (for kernel_launch step)
    pub dispatch_strategy: Option<String>,
    /// Brick category breakdown (for brick_profile step)
    /// Maps category name ("Norm", "Attention", "FFN", "Other") to nanoseconds
    pub brick_categories: Option<Vec<(String, u64)>>,
    /// Per-brick timing (for brick_profile step)
    /// Maps brick name ("RmsNorm", "QkvProjection", etc.) to (total_ns, count)
    pub brick_timings: Option<Vec<(String, u64, u64)>>,
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

    /// Check if verbose tracing is enabled (requires D2H sync for stats)
    #[must_use]
    pub fn is_verbose(&self) -> bool {
        self.config.enabled && self.config.verbose
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

    /// Start timing a step and emit TaskStateEntered event (AWS Step Functions F-AWS-01)
    pub fn start_step(&mut self, step: TraceStep) {
        if self.config.should_trace(step) {
            self.step_start = Some(Instant::now());

            // Emit TaskStateEntered event (F-AWS-01: Entry/Exit pairing)
            let entry_id = self.next_id();
            let event = TraceEvent {
                id: entry_id,
                timestamp: Self::timestamp(),
                event_type: AwsEventType::TaskStateEntered,
                previous_event_id: None, // Entry events have no predecessor
                step,
                iteration: 0,
                layer: None,
                input_shape: vec![],
                output_shape: vec![],
                stats: TensorStats::default(),
                duration_us: 0,
                error: None,
                cause: None,
                details: TraceDetails::default(),
            };
            self.events.push(event);
            // Store entry ID for the corresponding Exit event (F-AWS-02)
            self.last_entered_id = Some(entry_id);
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
            cause: None,
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
            cause: None,
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
            cause: None,
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
            cause: None,
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
            cause: None,
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
            cause: None,
            details: TraceDetails {
                sampled_token: Some(token_id),
                decoded_text: Some(decoded_text.to_string()),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace GPU kernel launch (GH-219: PTX-level tracing)
    ///
    /// Records a CUDA kernel launch event with grid/block configuration,
    /// shared memory usage, and dispatch strategy. This connects the
    /// logical layer trace to the physical PTX execution.
    pub fn trace_kernel_launch(
        &mut self,
        kernel_name: &str,
        layer_idx: Option<usize>,
        grid_dims: [u32; 3],
        block_dims: [u32; 3],
        shared_mem_bytes: u32,
        dispatch_strategy: Option<&str>,
        duration_us: u64,
    ) {
        if !self.config.should_trace(TraceStep::KernelLaunch) {
            return;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: None, // Kernel launches are leaf events
            step: TraceStep::KernelLaunch,
            iteration: 0,
            layer: layer_idx,
            input_shape: vec![],
            output_shape: vec![],
            stats: TensorStats::default(),
            duration_us,
            error: None,
            cause: None,
            details: TraceDetails {
                kernel_name: Some(kernel_name.to_string()),
                grid_dims: Some(grid_dims),
                block_dims: Some(block_dims),
                shared_mem_bytes: Some(shared_mem_bytes),
                kernel_layer: layer_idx,
                dispatch_strategy: dispatch_strategy.map(String::from),
                ..Default::default()
            },
        };

        self.events.push(event);
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

    /// Record an execution failure (F-JID-01: Jidoka error handling)
    ///
    /// Emits an `ExecutionFailed` event per AWS Step Functions parity (F-AWS-05).
    /// Use this when the inference cannot proceed due to missing config,
    /// invalid model format, or other fatal errors.
    ///
    /// # Arguments
    /// * `error` - High-level error category (e.g., "Initialization Failure")
    /// * `cause` - Specific cause of failure (e.g., "Missing config.json")
    pub fn record_execution_failed(&mut self, error: &str, cause: &str) {
        if !self.config.enabled {
            return;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::ExecutionFailed,
            previous_event_id: None,
            step: TraceStep::Tokenize, // Use first step as placeholder
            iteration: 0,
            layer: None,
            input_shape: vec![],
            output_shape: vec![],
            stats: TensorStats::default(),
            duration_us: 0,
            error: Some(TraceError::ExecutionFailed {
                cause: error.to_string(),
            }),
            cause: Some(cause.to_string()),
            details: TraceDetails::default(),
        };
        self.events.push(event);
        self.error_count += 1;
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
                    "[{}/8] {}\n",
                    event.step.step_number(),
                    event.step.name()
                ));
            }

            // Step content
            match event.step {
                TraceStep::Tokenize => {
                    if let Some(ref text) = event.details.input_text {
                        let display_text = if text.len() > 50 {
                            format!("{}...", text.get(..50).expect("text length checked > 50"))
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
                TraceStep::KernelLaunch => {
                    if let Some(ref name) = event.details.kernel_name {
                        output.push_str(&format!("  Kernel:   {}\n", name));
                    }
                    if let Some([gx, gy, gz]) = event.details.grid_dims {
                        output.push_str(&format!("  Grid:     ({}, {}, {})\n", gx, gy, gz));
                    }
                    if let Some([bx, by, bz]) = event.details.block_dims {
                        output.push_str(&format!("  Block:    ({}, {}, {})\n", bx, by, bz));
                    }
                    if let Some(smem) = event.details.shared_mem_bytes {
                        output.push_str(&format!("  SharedMem: {} bytes\n", smem));
                    }
                    if let Some(ref strategy) = event.details.dispatch_strategy {
                        output.push_str(&format!("  Dispatch: {}\n", strategy));
                    }
                    if let Some(layer) = event.details.kernel_layer {
                        output.push_str(&format!("  Layer:    {}\n", layer));
                    }
                    output.push_str(&format!("  Duration: {}us\n", event.duration_us));
                },
                TraceStep::BrickProfile => {
                    if let Some(ref categories) = event.details.brick_categories {
                        output.push_str("  Categories:\n");
                        for (name, ns) in categories {
                            output.push_str(&format!(
                                "    {:<12} {:.3}ms\n",
                                name,
                                *ns as f64 / 1_000_000.0
                            ));
                        }
                    }
                    if let Some(ref timings) = event.details.brick_timings {
                        output.push_str("  Bricks:\n");
                        for (name, total_ns, count) in timings {
                            output.push_str(&format!(
                                "    {:<20} {:.3}ms (x{})\n",
                                name,
                                *total_ns as f64 / 1_000_000.0,
                                count
                            ));
                        }
                    }
                    output.push_str(&format!("  Duration: {}us\n", event.duration_us));
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
            // AWS Step Functions parity fields (F-AWS-01, F-AWS-02)
            json.push_str(&format!("      \"id\": {},\n", event.id));
            json.push_str(&format!("      \"timestamp\": {:?},\n", event.timestamp));
            json.push_str(&format!("      \"type\": {:?},\n", event.event_type.name()));
            json.push_str(&format!(
                "      \"previous_event_id\": {},\n",
                event
                    .previous_event_id
                    .map_or("null".to_string(), |id| id.to_string())
            ));
            // State details
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
                "      \"error\": {},\n",
                event
                    .error
                    .as_ref()
                    .map_or("null".to_string(), |e| format!("{:?}", e.to_string()))
            ));
            // F-AWS-05: cause field required for ExecutionFailed events
            json.push_str(&format!(
                "      \"cause\": {}\n",
                event
                    .cause
                    .as_ref()
                    .map_or("null".to_string(), |c| format!("{:?}", c))
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
        TraceError::ExecutionFailed { .. } => {
            "Execution failed. Check model config and dependencies"
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

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod inference_trace_tests;
