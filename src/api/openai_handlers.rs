//! OpenAI-compatible API handlers
//!
//! Extracted from api/mod.rs (PMAT-802) to reduce module size.
//! Contains chat completion, streaming, and model list handlers.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::stream::Stream;

use super::{
    build_trace_data, clean_chat_output, format_chat_messages, AppState, ChatChoice,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ErrorResponse,
    OpenAIModel, OpenAIModelsResponse, Usage,
};
use crate::generate::{GenerationConfig, SamplingStrategy};
use crate::tokenizer::BPETokenizer;

// ============================================================================
// Shared helpers — eliminate duplication across backend paths
// ============================================================================

/// Record failure and return an error response.
fn fail_response(state: &AppState, status: StatusCode, msg: impl std::fmt::Display) -> Response {
    state.metrics.record_failure();
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

/// Current Unix timestamp.
fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Get tokenizer from state or return 500.
#[allow(clippy::result_large_err)]
fn require_tokenizer(state: &AppState) -> Result<Arc<BPETokenizer>, Response> {
    state.tokenizer.clone().ok_or_else(|| {
        fail_response(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "No tokenizer available",
        )
    })
}

/// Format chat messages, tokenize, validate non-empty.
#[allow(clippy::result_large_err)]
fn tokenize_chat_prompt(
    tokenizer: &BPETokenizer,
    messages: &[ChatMessage],
    model_hint: Option<&str>,
    state: &AppState,
) -> Result<Vec<u32>, Response> {
    let prompt_text = format_chat_messages(messages, model_hint);
    let ids = tokenizer.encode(&prompt_text);
    if ids.is_empty() {
        return Err(fail_response(
            state,
            StatusCode::BAD_REQUEST,
            "Messages cannot be empty",
        ));
    }
    Ok(ids)
}

/// Extract common generation parameters from the request.
fn chat_gen_params(
    request: &ChatCompletionRequest,
    tokenizer: &BPETokenizer,
) -> (usize, f32, u32) {
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);
    let eos_token_id = tokenizer
        .get_token_id("<|im_end|>")
        .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
        .unwrap_or(151645);
    (max_tokens, temperature, eos_token_id)
}

/// Build a non-streaming ChatCompletionResponse.
fn build_chat_response(
    request_id: String,
    model: String,
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    max_tokens: usize,
    trace_level: Option<&str>,
    latency: Duration,
) -> Response {
    let (brick_trace, step_trace, layer_trace) = build_trace_data(
        trace_level,
        latency.as_micros() as u64,
        prompt_tokens,
        completion_tokens,
        28,
    );
    Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
                name: None,
            },
            finish_reason: if completion_tokens >= max_tokens {
                "length".to_string()
            } else {
                "stop".to_string()
            },
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        brick_trace,
        step_trace,
        layer_trace,
    })
    .into_response()
}

/// Build a pre-generated SSE streaming response (all tokens already generated).
fn pregenerated_sse_response(
    token_ids: Vec<u32>,
    tokenizer: Arc<BPETokenizer>,
    request_id: String,
    model_name: String,
    clean: bool,
) -> Response {
    let stream = async_stream::stream! {
        let initial = ChatCompletionChunk::initial(&request_id, &model_name);
        if let Ok(data) = serde_json::to_string(&initial) {
            yield Ok::<_, Infallible>(Event::default().data(data));
        }

        for &token_id in &token_ids {
            if let Ok(text) = tokenizer.decode(&[token_id]) {
                let text = if clean { clean_chat_output(&text) } else { text };
                if !text.is_empty() {
                    let chunk = ChatCompletionChunk::content(&request_id, &model_name, &text);
                    if let Ok(data) = serde_json::to_string(&chunk) {
                        yield Ok(Event::default().data(data));
                    }
                }
            }
        }

        let done = ChatCompletionChunk::done(&request_id, &model_name);
        if let Ok(data) = serde_json::to_string(&done) {
            yield Ok(Event::default().data(data));
        }
        yield Ok(Event::default().data("[DONE]".to_string()));
    };
    Sse::new(stream).into_response()
}

/// Build a true-streaming SSE response with keep-alive (tokens arrive via channel).
fn true_streaming_sse_response(
    rx: tokio::sync::mpsc::Receiver<Result<u32, String>>,
    tokenizer: Arc<BPETokenizer>,
    request_id: String,
    model_name: String,
    metrics: Arc<crate::metrics::MetricsCollector>,
    start: Instant,
    clean: bool,
) -> Response {
    use tokio_stream::wrappers::ReceiverStream;
    use tokio_stream::StreamExt;

    let token_stream = ReceiverStream::new(rx);
    let mut completion_tokens = 0usize;

    let stream = async_stream::stream! {
        let initial = ChatCompletionChunk::initial(&request_id, &model_name);
        if let Ok(data) = serde_json::to_string(&initial) {
            yield Ok::<_, Infallible>(Event::default().data(data));
        }

        tokio::pin!(token_stream);
        while let Some(result) = token_stream.next().await {
            match result {
                Ok(token_id) => {
                    completion_tokens += 1;
                    if let Ok(text) = tokenizer.decode(&[token_id]) {
                        let text = if clean { clean_chat_output(&text) } else { text };
                        if !text.is_empty() {
                            let chunk = ChatCompletionChunk::content(&request_id, &model_name, &text);
                            if let Ok(data) = serde_json::to_string(&chunk) {
                                yield Ok(Event::default().data(data));
                            }
                        }
                    }
                }
                Err(e) => {
                    let error_chunk = serde_json::json!({ "error": e });
                    if let Ok(data) = serde_json::to_string(&error_chunk) {
                        yield Ok(Event::default().data(data));
                    }
                    break;
                }
            }
        }

        let done = ChatCompletionChunk::done(&request_id, &model_name);
        if let Ok(data) = serde_json::to_string(&done) {
            yield Ok(Event::default().data(data));
        }

        metrics.record_success(completion_tokens, start.elapsed());
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

// ============================================================================
// Handlers
// ============================================================================

/// OpenAI-compatible models listing handler
///
/// Returns available models in OpenAI API format (GET /v1/models).
pub async fn openai_models_handler(State(state): State<AppState>) -> Json<OpenAIModelsResponse> {
    let models = if let Some(registry) = &state.registry {
        registry
            .list()
            .into_iter()
            .map(|m| OpenAIModel {
                id: m.id,
                object: "model".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0),
                owned_by: "realizar".to_string(),
            })
            .collect()
    } else {
        // Single model mode
        vec![OpenAIModel {
            id: "default".to_string(),
            object: "model".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            owned_by: "realizar".to_string(),
        }]
    };

    Json(OpenAIModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

/// OpenAI-compatible /v1/chat/completions endpoint (supports streaming)
pub async fn openai_chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let start = Instant::now();

    // GH-152: Verbose request logging
    if state.is_verbose() {
        let msg_count = request.messages.len();
        let last_msg = request
            .messages
            .last()
            .map(|m| m.content.chars().take(50).collect::<String>())
            .unwrap_or_default();
        eprintln!(
            "[VERBOSE] POST /v1/chat/completions model={} messages={} last={:?}",
            request.model, msg_count, last_msg
        );
    }

    let trace_level = headers
        .get("X-Trace-Level")
        .and_then(|v| v.to_str().ok())
        .map(str::to_lowercase);

    let request_id = format!(
        "chatcmpl-q4k-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    // ── GPU model (non-batched --gpu mode) ──────────────────────────────
    #[cfg(feature = "gpu")]
    if let Some(gpu_model_lock) = state.gpu_model() {
        use crate::gpu::GpuGenerateConfig;

        let tokenizer = match require_tokenizer(&state) {
            Ok(t) => t,
            Err(r) => return r,
        };
        let prompt_ids = match tokenize_chat_prompt(&tokenizer, &request.messages, Some("qwen"), &state) {
            Ok(ids) => ids,
            Err(r) => return r,
        };
        let prompt_tokens = prompt_ids.len();
        let prompt_usize: Vec<usize> = prompt_ids.iter().map(|&x| x as usize).collect();
        let (max_tokens, temperature, eos_token_id) = chat_gen_params(&request, &tokenizer);

        let gpu_config = GpuGenerateConfig {
            max_tokens,
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos_token_id as usize],
            trace: false,
        };

        let mut model = match gpu_model_lock.write() {
            Ok(m) => m,
            Err(e) => {
                return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, format!("GPU model lock error: {e}"));
            }
        };
        let generated = match model.generate(&prompt_usize, &gpu_config) {
            Ok(g) => g,
            Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
        };

        let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).map(|&x| x as u32).collect();
        let completion_tokens = token_ids.len();

        if request.stream {
            state.metrics.record_success(completion_tokens, start.elapsed());
            return pregenerated_sse_response(token_ids, tokenizer, request_id, request.model.clone(), false);
        }

        let text = match tokenizer.decode(&token_ids) {
            Ok(t) => clean_chat_output(&t),
            Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
        };

        let latency = start.elapsed();
        state.metrics.record_success(completion_tokens, latency);
        return build_chat_response(
            request_id, request.model.clone(), text,
            prompt_tokens, completion_tokens, max_tokens,
            trace_level.as_deref(), latency,
        );
    }

    // ── Cached model (GPU batched --gpu --batch mode) ───────────────────
    #[cfg(feature = "gpu")]
    if let Some(cached_model) = state.cached_model() {
        use crate::gguf::QuantizedGenerateConfig;

        let tokenizer = match require_tokenizer(&state) {
            Ok(t) => t,
            Err(r) => return r,
        };
        let prompt_ids = match tokenize_chat_prompt(&tokenizer, &request.messages, Some("qwen"), &state) {
            Ok(ids) => ids,
            Err(r) => return r,
        };
        let prompt_tokens = prompt_ids.len();
        let (max_tokens, temperature, eos_token_id) = chat_gen_params(&request, &tokenizer);

        let q_config = QuantizedGenerateConfig {
            max_tokens,
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos_token_id],
            trace: false,
        };

        let generated = match cached_model.model().generate_with_cache(&prompt_ids, &q_config) {
            Ok(g) => g,
            Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
        };

        let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
        let completion_tokens = token_ids.len();

        if request.stream {
            state.metrics.record_success(completion_tokens, start.elapsed());
            return pregenerated_sse_response(token_ids, tokenizer, request_id, request.model.clone(), false);
        }

        let text = match tokenizer.decode(&token_ids) {
            Ok(t) => clean_chat_output(&t),
            Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
        };

        let latency = start.elapsed();
        state.metrics.record_success(completion_tokens, latency);
        return build_chat_response(
            request_id, request.model.clone(), text,
            prompt_tokens, completion_tokens, max_tokens,
            trace_level.as_deref(), latency,
        );
    }

    // ── CUDA-optimized model (755+ tok/s, 2.6x Ollama) ─────────────────
    #[cfg(feature = "cuda")]
    if let Some(cuda_model_lock) = state.cuda_model() {
        use crate::gguf::QuantizedGenerateConfig;

        let tokenizer = match require_tokenizer(&state) {
            Ok(t) => t,
            Err(r) => return r,
        };
        let prompt_ids = match tokenize_chat_prompt(&tokenizer, &request.messages, Some("qwen"), &state) {
            Ok(ids) => ids,
            Err(r) => return r,
        };
        let prompt_tokens = prompt_ids.len();
        let (max_tokens, temperature, eos_token_id) = chat_gen_params(&request, &tokenizer);

        let q_config = QuantizedGenerateConfig {
            max_tokens,
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos_token_id],
            trace: false,
        };

        if request.stream {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(16);
            let cuda_model_clone = cuda_model_lock.clone();
            let prompt_ids_clone = prompt_ids.clone();
            let q_config_clone = q_config.clone();

            tokio::task::spawn_blocking(move || {
                let mut cuda_model = cuda_model_clone.write().expect("operation failed");
                let result = cuda_model.generate_gpu_resident_streaming(
                    &prompt_ids_clone,
                    &q_config_clone,
                    |token_id| tx.blocking_send(Ok(token_id)).is_ok(),
                );
                if let Err(e) = result {
                    let _ = tx.blocking_send(Err(e.to_string()));
                }
            });

            return true_streaming_sse_response(
                rx, tokenizer, request_id, request.model.clone(),
                state.metrics.clone(), start, false,
            );
        }

        // Non-streaming CUDA
        let mut cuda_model = cuda_model_lock.write().expect("operation failed");
        let generated = match cuda_model.generate_gpu_resident(&prompt_ids, &q_config) {
            Ok(g) => g,
            Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
        };

        let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
        let completion_tokens = token_ids.len();
        let response_text = tokenizer
            .decode(&token_ids)
            .unwrap_or_else(|_| String::new());
        let response_text = clean_chat_output(&response_text);

        let latency = start.elapsed();
        state.metrics.record_success(completion_tokens, latency);
        return build_chat_response(
            request_id, request.model, response_text,
            prompt_tokens, completion_tokens, max_tokens,
            trace_level.as_deref(), latency,
        );
    }

    // ── Quantized model (GGUF serve mode) ───────────────────────────────
    if let Some(quantized_model) = state.quantized_model() {
        use crate::gguf::QuantizedGenerateConfig;

        let tokenizer = match require_tokenizer(&state) {
            Ok(t) => t,
            Err(r) => return r,
        };
        let prompt_ids = match tokenize_chat_prompt(&tokenizer, &request.messages, Some("qwen"), &state) {
            Ok(ids) => ids,
            Err(r) => return r,
        };
        let prompt_tokens = prompt_ids.len();
        let (max_tokens, temperature, eos_token_id) = chat_gen_params(&request, &tokenizer);

        let q_config = QuantizedGenerateConfig {
            max_tokens,
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos_token_id],
            trace: false,
        };

        if request.stream {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(16);
            let quantized_model_clone = quantized_model.clone();
            let prompt_ids_clone = prompt_ids.clone();
            let q_config_clone = q_config.clone();

            tokio::task::spawn_blocking(move || {
                let result = quantized_model_clone.generate_with_cache_streaming(
                    &prompt_ids_clone,
                    &q_config_clone,
                    |token_id| tx.blocking_send(Ok(token_id)).is_ok(),
                );
                if let Err(e) = result {
                    let _ = tx.blocking_send(Err(e.to_string()));
                }
            });

            return true_streaming_sse_response(
                rx, tokenizer, request_id, request.model.clone(),
                state.metrics.clone(), start, true,
            );
        }

        // Non-streaming quantized
        let generated = match quantized_model.generate_with_cache(&prompt_ids, &q_config) {
            Ok(g) => g,
            Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
        };

        let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
        let completion_tokens = token_ids.len();
        let text = match tokenizer.decode(&token_ids) {
            Ok(t) => clean_chat_output(&t),
            Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
        };

        let latency = start.elapsed();
        state.metrics.record_success(completion_tokens, latency);
        return build_chat_response(
            request_id, request.model.clone(), text,
            prompt_tokens, completion_tokens, max_tokens,
            trace_level.as_deref(), latency,
        );
    }

    // ── Registry-based model fallback ───────────────────────────────────
    let model_id = if request.model == "default" || request.model.is_empty() {
        None
    } else {
        Some(request.model.as_str())
    };

    let (model, tokenizer) = match state.get_model(model_id) {
        Ok((m, t)) => (m, t),
        Err(e) => return fail_response(&state, StatusCode::NOT_FOUND, e),
    };

    let prompt_text = format_chat_messages(&request.messages, Some(&request.model));
    let prompt_ids = tokenizer.encode(&prompt_text);
    if prompt_ids.is_empty() {
        return fail_response(&state, StatusCode::BAD_REQUEST, "Messages cannot be empty");
    }

    let prompt_tokens = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p };
    }

    let generated = match model.generate(&prompt, &config) {
        Ok(g) => g,
        Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let token_ids: Vec<u32> = match generated
        .iter()
        .map(|&id| u32::try_from(id).map_err(|_| format!("Token ID {id} exceeds u32 range")))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(ids) => ids,
        Err(e) => return fail_response(&state, StatusCode::BAD_REQUEST, e),
    };

    let generated_ids: Vec<u32> = token_ids[prompt.len()..].to_vec();
    let completion_tokens = generated_ids.len();

    if request.stream {
        state.metrics.record_success(completion_tokens, start.elapsed());
        return pregenerated_sse_response(generated_ids, tokenizer, request_id, request.model.clone(), false);
    }

    let response_text = match tokenizer.decode(&generated_ids) {
        Ok(t) => t,
        Err(e) => return fail_response(&state, StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let duration = start.elapsed();
    state.metrics.record_success(completion_tokens, duration);

    Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: request.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_text,
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        // Registry models don't support tracing
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    })
    .into_response()
}

/// OpenAI-compatible /v1/chat/completions streaming endpoint (SSE)
pub async fn openai_chat_completions_stream_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    let model_id = if request.model == "default" || request.model.is_empty() {
        None
    } else {
        Some(request.model.as_str())
    };

    let (model, tokenizer) = state.get_model(model_id).map_err(|e| {
        state.metrics.record_failure();
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let prompt_text = format_chat_messages(&request.messages, Some(&request.model));
    let prompt_ids = tokenizer.encode(&prompt_text);
    if prompt_ids.is_empty() {
        state.metrics.record_failure();
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Messages cannot be empty".to_string(),
            }),
        ));
    }

    let prompt_len = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p };
    }

    let request_id = format!(
        "chatcmpl-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let generated = model.generate(&prompt, &config).map_err(|e| {
        state.metrics.record_failure();
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let token_ids: Vec<u32> = generated
        .iter()
        .filter_map(|&id| u32::try_from(id).ok())
        .collect();

    let generated_ids = token_ids[prompt_len..].to_vec();
    let model_name = request.model.clone();
    let request_id_clone = request_id.clone();
    let tokenizer_clone = tokenizer;

    let stream = async_stream::stream! {
        let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
        let data = serde_json::to_string(&initial).unwrap_or_default();
        yield Ok(Event::default().data(format!("data: {}\n", data)));

        for &token_id in &generated_ids {
            let text = match tokenizer_clone.decode(&[token_id]) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, &text);
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            yield Ok(Event::default().data(format!("data: {}\n", data)));
        }

        let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
        let data = serde_json::to_string(&done).unwrap_or_default();
        yield Ok(Event::default().data(format!("data: {}\n", data)));

        yield Ok(Event::default().data("data: [DONE]\n".to_string()));
    };

    Ok(Sse::new(stream))
}
