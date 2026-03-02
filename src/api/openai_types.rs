//! OpenAI-Compatible API Types (per spec §5.4)
//!
//! Types for OpenAI API compatibility including chat completions and streaming.
//! Extracted from api/mod.rs (PMAT-802)

use serde::{Deserialize, Serialize};

// ============================================================================
// OpenAI-Compatible API Types (per spec §5.4)
// ============================================================================

/// OpenAI-compatible chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID to use
    pub model: String,
    /// Chat messages
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Sampling temperature
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: usize,
    /// Stream responses
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// User identifier
    #[serde(default)]
    pub user: Option<String>,
}

pub(crate) fn default_n() -> usize {
    1
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant"
    pub role: String,
    /// Message content
    pub content: String,
    /// Optional name
    #[serde(default)]
    pub name: Option<String>,
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique request ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choices array
    pub choices: Vec<ChatChoice>,
    /// Token usage statistics
    pub usage: Usage,
    /// Brick-level trace data (tensor operations) - only present when X-Trace-Level: brick
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brick_trace: Option<TraceData>,
    /// Step-level trace data (forward pass steps) - only present when X-Trace-Level: step
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_trace: Option<TraceData>,
    /// Layer-level trace data (attention, MLP) - only present when X-Trace-Level: layer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_trace: Option<TraceData>,
}

/// Trace data for debugging inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    /// Trace level that was requested
    pub level: String,
    /// Number of operations traced
    pub operations: usize,
    /// Total time in microseconds
    pub total_time_us: u64,
    /// Per-operation timing breakdown
    pub breakdown: Vec<TraceOperation>,
}

/// Individual traced operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceOperation {
    /// Operation name
    pub name: String,
    /// Time in microseconds
    pub time_us: u64,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Build trace data based on X-Trace-Level header
///
/// Returns (brick_trace, step_trace, layer_trace) tuple based on requested level.
/// Only reports wall-clock totals — per-operation breakdown requires `apr profile`
/// with BrickProfiler instrumentation. We refuse to fabricate per-op estimates.
#[must_use]
pub fn build_trace_data(
    trace_level: Option<&str>,
    latency_us: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
    num_layers: usize,
) -> (Option<TraceData>, Option<TraceData>, Option<TraceData>) {
    match trace_level {
        Some("brick") => (
            Some(TraceData {
                level: "brick".to_string(),
                operations: completion_tokens,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} prompt + {} completion tokens, {} layers. \
                             Per-op breakdown not available — use `apr profile` for real brick-level telemetry",
                            prompt_tokens, completion_tokens, num_layers
                        )),
                    },
                ],
            }),
            None,
            None,
        ),
        Some("step") => (
            None,
            Some(TraceData {
                level: "step".to_string(),
                operations: completion_tokens,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} prompt + {} completion tokens, {} layers. \
                             Step-level breakdown not instrumented — use `apr profile` for real timing",
                            prompt_tokens, completion_tokens, num_layers
                        )),
                    },
                ],
            }),
            None,
        ),
        Some("layer") => (
            None,
            None,
            Some(TraceData {
                level: "layer".to_string(),
                operations: num_layers,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} layers, {} tokens. \
                             Per-layer breakdown not instrumented — use `apr profile --granular` for real per-layer timing",
                            num_layers, prompt_tokens + completion_tokens
                        )),
                    },
                ],
            }),
        ),
        _ => (None, None, None),
    }
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index
    pub index: usize,
    /// Generated message
    pub message: ChatMessage,
    /// Finish reason
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Completion tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// OpenAI-compatible models list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModelsResponse {
    /// Object type
    pub object: String,
    /// Model list
    pub data: Vec<OpenAIModel>,
}

/// OpenAI model info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModel {
    /// Model ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Created timestamp
    pub created: i64,
    /// Owner
    pub owned_by: String,
}

// ============================================================================
// OpenAI Streaming Types (SSE)
// ============================================================================

/// Streaming chat completion chunk (SSE format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique request ID
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choices array with deltas
    pub choices: Vec<ChatChunkChoice>,
}

/// Streaming choice with delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    /// Choice index
    pub index: usize,
    /// Delta content (partial message)
    pub delta: ChatDelta,
    /// Finish reason (None until done)
    pub finish_reason: Option<String>,
}

/// Delta content for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Role (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Content chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatCompletionChunk {
    /// Create a new chunk with content
    pub fn new(id: &str, model: &str, content: Option<String>, finish_reason: Option<String>) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            model: model.to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: if content.is_none() && finish_reason.is_none() {
                        Some("assistant".to_string())
                    } else {
                        None
                    },
                    content,
                },
                finish_reason,
            }],
        }
    }

    /// Create initial chunk with role only
    pub fn initial(id: &str, model: &str) -> Self {
        Self::new(id, model, None, None)
    }

    /// Create content chunk
    pub fn content(id: &str, model: &str, text: &str) -> Self {
        Self::new(id, model, Some(text.to_string()), None)
    }

    /// Create final chunk with finish reason
    pub fn done(id: &str, model: &str) -> Self {
        Self::new(id, model, None, Some("stop".to_string()))
    }
}
