//! OpenAI-compatible API handlers
//!
//! Extracted from api/mod.rs (PMAT-802) to reduce module size.
//! Contains chat completion, streaming, and model list handlers.

use std::convert::Infallible;

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
    use std::time::Instant;
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

    // Parse X-Trace-Level header for debugging
    let trace_level = headers
        .get("X-Trace-Level")
        .and_then(|v| v.to_str().ok())
        .map(str::to_lowercase);

    // Generate request ID
    let request_id = format!(
        "chatcmpl-q4k-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    // IMP-152: Try GPU model (non-batched --gpu mode)
    #[cfg(feature = "gpu")]
    if let Some(gpu_model_lock) = state.gpu_model() {
        use crate::gpu::GpuGenerateConfig;

        let tokenizer = match state.tokenizer.clone() {
            Some(t) => t,
            None => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "No tokenizer available".to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // Convert chat messages to prompt using ChatML
        let prompt_text = format_chat_messages(&request.messages, Some("qwen"));
        let prompt_ids: Vec<usize> = tokenizer
            .encode(&prompt_text)
            .iter()
            .map(|&x| x as usize)
            .collect();

        if prompt_ids.is_empty() {
            state.metrics.record_failure();
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Messages cannot be empty".to_string(),
                }),
            )
                .into_response();
        }

        let prompt_tokens = prompt_ids.len();
        let max_tokens = request.max_tokens.unwrap_or(256);
        let temperature = request.temperature.unwrap_or(0.7);

        // PMAT-088: Get EOS token ID for proper stop sequence (GPU path)
        let eos_token_id = tokenizer
            .get_token_id("<|im_end|>")
            .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
            .unwrap_or(151645) as usize;

        let gpu_config = GpuGenerateConfig {
            max_tokens,
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos_token_id],
        };

        // Generate using GPU model
        let generated = {
            let mut model = match gpu_model_lock.write() {
                Ok(m) => m,
                Err(e) => {
                    state.metrics.record_failure();
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("GPU model lock error: {e}"),
                        }),
                    )
                        .into_response();
                },
            };
            match model.generate(&prompt_ids, &gpu_config) {
                Ok(g) => g,
                Err(e) => {
                    state.metrics.record_failure();
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: e.to_string(),
                        }),
                    )
                        .into_response();
                },
            }
        };

        // Skip prompt tokens, convert to u32
        let token_ids: Vec<u32> = generated
            .iter()
            .skip(prompt_tokens)
            .map(|&x| x as u32)
            .collect();
        let completion_tokens = token_ids.len();

        // Handle streaming vs non-streaming
        if request.stream {
            let model_name = request.model.clone();
            let request_id_clone = request_id.clone();

            let stream = async_stream::stream! {
                // Send initial chunk with role
                let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
                if let Ok(data) = serde_json::to_string(&initial) {
                    yield Ok::<_, Infallible>(Event::default().data(data));
                }

                // Stream tokens one by one
                for &token_id in &token_ids {
                    if let Ok(text) = tokenizer.decode(&[token_id]) {
                        if !text.is_empty() {
                            let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, &text);
                            if let Ok(data) = serde_json::to_string(&chunk) {
                                yield Ok(Event::default().data(data));
                            }
                        }
                    }
                }

                // Send final chunk with finish reason
                let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
                if let Ok(data) = serde_json::to_string(&done) {
                    yield Ok(Event::default().data(data));
                }

                // Send [DONE] marker
                yield Ok(Event::default().data("[DONE]".to_string()));
            };

            state
                .metrics
                .record_success(completion_tokens, start.elapsed());
            return Sse::new(stream).into_response();
        }

        // Non-streaming response
        let text = match tokenizer.decode(&token_ids) {
            Ok(t) => t,
            Err(e) => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // PMAT-088: Clean output to prevent prompt injection
        let text = clean_chat_output(&text);

        let latency = start.elapsed();
        state.metrics.record_success(completion_tokens, latency);

        // Build trace data based on X-Trace-Level header (GPU path)
        let (brick_trace, step_trace, layer_trace) = build_trace_data(
            trace_level.as_deref(),
            latency.as_micros() as u64,
            prompt_tokens,
            completion_tokens,
            28, // Default layer count for Qwen2 models
        );

        return Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            model: request.model.clone(),
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
        .into_response();
    }

    // IMP-151: Try cached model (GPU batched --gpu --batch mode)
    #[cfg(feature = "gpu")]
    if let Some(cached_model) = state.cached_model() {
        use crate::gguf::QuantizedGenerateConfig;

        let tokenizer = match state.tokenizer.clone() {
            Some(t) => t,
            None => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "No tokenizer available".to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // Convert chat messages to prompt using ChatML (GGUF models are typically Qwen/ChatML)
        let prompt_text = format_chat_messages(&request.messages, Some("qwen"));

        // Tokenize prompt
        let prompt_ids = tokenizer.encode(&prompt_text);
        if prompt_ids.is_empty() {
            state.metrics.record_failure();
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Messages cannot be empty".to_string(),
                }),
            )
                .into_response();
        }

        let prompt_tokens = prompt_ids.len();
        let max_tokens = request.max_tokens.unwrap_or(256);
        let temperature = request.temperature.unwrap_or(0.7);

        // PMAT-088: Get EOS token ID for proper stop sequence
        let eos_token_id = tokenizer
            .get_token_id("<|im_end|>")
            .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
            .unwrap_or(151645);

        let q_config = QuantizedGenerateConfig {
            max_tokens,
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos_token_id],
            trace: false,
        };

        let generated = match cached_model
            .model()
            .generate_with_cache(&prompt_ids, &q_config)
        {
            Ok(g) => g,
            Err(e) => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // Skip prompt tokens
        let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
        let completion_tokens = token_ids.len();

        // Handle streaming vs non-streaming
        if request.stream {
            // Streaming response - return SSE
            let model_name = request.model.clone();
            let request_id_clone = request_id.clone();

            let stream = async_stream::stream! {
                // Send initial chunk with role
                let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
                if let Ok(data) = serde_json::to_string(&initial) {
                    yield Ok::<_, Infallible>(Event::default().data(data));
                }

                // Stream tokens one by one
                for &token_id in &token_ids {
                    // Decode single token
                    if let Ok(text) = tokenizer.decode(&[token_id]) {
                        if !text.is_empty() {
                            let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, &text);
                            if let Ok(data) = serde_json::to_string(&chunk) {
                                yield Ok(Event::default().data(data));
                            }
                        }
                    }
                }

                // Send final chunk with finish reason
                let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
                if let Ok(data) = serde_json::to_string(&done) {
                    yield Ok(Event::default().data(data));
                }

                // Send [DONE] marker
                yield Ok(Event::default().data("[DONE]".to_string()));
            };

            state
                .metrics
                .record_success(completion_tokens, start.elapsed());
            return Sse::new(stream).into_response();
        }

        // Non-streaming response
        let text = match tokenizer.decode(&token_ids) {
            Ok(t) => t,
            Err(e) => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // PMAT-088: Clean output to prevent prompt injection
        let text = clean_chat_output(&text);

        let latency = start.elapsed();
        state.metrics.record_success(completion_tokens, latency);

        // Build trace data based on X-Trace-Level header (cached GPU path)
        let (brick_trace, step_trace, layer_trace) = build_trace_data(
            trace_level.as_deref(),
            latency.as_micros() as u64,
            prompt_tokens,
            completion_tokens,
            28, // Default layer count for Qwen2 models
        );

        return Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            model: request.model.clone(),
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
        .into_response();
    }

    // PAR-111: CUDA-optimized model for high-performance GPU inference (755+ tok/s, 2.6x Ollama)
    #[cfg(feature = "cuda")]
    if let Some(cuda_model_lock) = state.cuda_model() {
        use crate::gguf::QuantizedGenerateConfig;

        let tokenizer = match state.tokenizer.clone() {
            Some(t) => t,
            None => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "No tokenizer available".to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // Convert chat messages to prompt using ChatML (GGUF models are typically Qwen/ChatML)
        let prompt_text = format_chat_messages(&request.messages, Some("qwen"));

        // Tokenize prompt
        let prompt_ids = tokenizer.encode(&prompt_text);
        if prompt_ids.is_empty() {
            state.metrics.record_failure();
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Messages cannot be empty".to_string(),
                }),
            )
                .into_response();
        }

        let prompt_tokens = prompt_ids.len();
        let max_tokens = request.max_tokens.unwrap_or(256);
        let temperature = request.temperature.unwrap_or(0.7);

        // PMAT-088: Get EOS token ID for proper stop sequence
        let eos_token_id = tokenizer
            .get_token_id("<|im_end|>")
            .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
            .unwrap_or(151645);

        let q_config = QuantizedGenerateConfig {
            max_tokens,
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos_token_id],
            trace: false,
        };

        // PAR-112: True streaming - handle streaming vs non-streaming with different paths
        if request.stream {
            // TRUE STREAMING: Generate tokens one-by-one and stream as they're produced
            use tokio::sync::mpsc;
            use tokio_stream::wrappers::ReceiverStream;
            use tokio_stream::StreamExt;

            let (tx, rx) = mpsc::channel::<Result<u32, String>>(16);
            let cuda_model_clone = cuda_model_lock.clone();
            let prompt_ids_clone = prompt_ids.clone();
            let q_config_clone = q_config.clone();

            // Spawn generation in a blocking task to avoid blocking the async runtime
            tokio::task::spawn_blocking(move || {
                let mut cuda_model = cuda_model_clone.write().expect("operation failed");

                // Use streaming generation - sends tokens via channel as they're generated
                let result = cuda_model.generate_gpu_resident_streaming(
                    &prompt_ids_clone,
                    &q_config_clone,
                    |token_id| {
                        // Send token through channel; return false to stop if channel closed
                        tx.blocking_send(Ok(token_id)).is_ok()
                    },
                );

                // Send error if generation failed
                if let Err(e) = result {
                    let _ = tx.blocking_send(Err(e.to_string()));
                }
            });

            // Convert channel receiver to SSE stream
            let model_name = request.model.clone();
            let request_id_clone = request_id.clone();
            let tokenizer_clone = tokenizer.clone();
            let metrics = state.metrics.clone();
            let start_time = start;

            let token_stream = ReceiverStream::new(rx);
            let mut completion_tokens = 0usize;

            let stream = async_stream::stream! {
                // Send initial chunk with role
                let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
                if let Ok(data) = serde_json::to_string(&initial) {
                    yield Ok::<_, Infallible>(Event::default().data(data));
                }

                // Stream tokens as they arrive from generation
                tokio::pin!(token_stream);
                while let Some(result) = token_stream.next().await {
                    match result {
                        Ok(token_id) => {
                            completion_tokens += 1;
                            // Decode and send immediately
                            if let Ok(text) = tokenizer_clone.decode(&[token_id]) {
                                if !text.is_empty() {
                                    let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, &text);
                                    if let Ok(data) = serde_json::to_string(&chunk) {
                                        yield Ok(Event::default().data(data));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // Send error chunk
                            let error_chunk = serde_json::json!({
                                "error": e
                            });
                            if let Ok(data) = serde_json::to_string(&error_chunk) {
                                yield Ok(Event::default().data(data));
                            }
                            break;
                        }
                    }
                }

                // Send final chunk with finish reason
                let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
                if let Ok(data) = serde_json::to_string(&done) {
                    yield Ok(Event::default().data(data));
                }

                // Record metrics
                metrics.record_success(completion_tokens, start_time.elapsed());

                // Send [DONE] marker
                yield Ok(Event::default().data("[DONE]"));
            };

            return Sse::new(stream)
                .keep_alive(
                    axum::response::sse::KeepAlive::new()
                        .interval(std::time::Duration::from_secs(15))
                        .text("keep-alive"),
                )
                .into_response();
        }

        // NON-STREAMING: Generate all tokens first, then return
        let mut cuda_model = cuda_model_lock.write().expect("operation failed");

        let generated = match cuda_model.generate_gpu_resident(&prompt_ids, &q_config) {
            Ok(g) => g,
            Err(e) => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // Skip prompt tokens
        let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
        let completion_tokens = token_ids.len();

        // Non-streaming: decode all tokens and return
        let response_text = tokenizer
            .decode(&token_ids)
            .unwrap_or_else(|_| String::new());

        // PMAT-088: Clean output to prevent prompt injection
        let response_text = clean_chat_output(&response_text);

        let elapsed = start.elapsed();
        state.metrics.record_success(completion_tokens, elapsed);

        // Build trace data based on X-Trace-Level header (CUDA optimized path)
        let (brick_trace, step_trace, layer_trace) = build_trace_data(
            trace_level.as_deref(),
            elapsed.as_micros() as u64,
            prompt_tokens,
            completion_tokens,
            28, // Default layer count for Qwen2 models
        );

        return Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            model: request.model,
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
            brick_trace,
            step_trace,
            layer_trace,
        })
        .into_response();
    }

    // IMP-150: Try quantized model (supports GGUF serve mode)
    if let Some(quantized_model) = state.quantized_model() {
        use crate::gguf::QuantizedGenerateConfig;

        let tokenizer = match state.tokenizer.clone() {
            Some(t) => t,
            None => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "No tokenizer available".to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // Convert chat messages to prompt using ChatML (GGUF models are typically Qwen/ChatML)
        let prompt_text = format_chat_messages(&request.messages, Some("qwen"));

        // Tokenize prompt
        let prompt_ids = tokenizer.encode(&prompt_text);
        if prompt_ids.is_empty() {
            state.metrics.record_failure();
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Messages cannot be empty".to_string(),
                }),
            )
                .into_response();
        }

        let prompt_tokens = prompt_ids.len();
        let max_tokens = request.max_tokens.unwrap_or(256);
        let temperature = request.temperature.unwrap_or(0.7);

        // PMAT-088: Get EOS token ID for proper stop sequence
        // ChatML uses <|im_end|> (token ID 151645 for Qwen models)
        let eos_token_id = tokenizer
            .get_token_id("<|im_end|>")
            .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
            .unwrap_or(151645); // Fallback to Qwen's <|im_end|> token ID

        let q_config = QuantizedGenerateConfig {
            max_tokens,
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos_token_id],
            trace: false,
        };

        // PMAT-087: True streaming - handle streaming vs non-streaming with different paths
        if request.stream {
            // TRUE STREAMING: Generate tokens one-by-one and stream as they're produced
            use tokio::sync::mpsc;
            use tokio_stream::wrappers::ReceiverStream;
            use tokio_stream::StreamExt;

            let (tx, rx) = mpsc::channel::<Result<u32, String>>(16);
            let quantized_model_clone = quantized_model.clone();
            let prompt_ids_clone = prompt_ids.clone();
            let q_config_clone = q_config.clone();

            // Spawn generation in a blocking task to avoid blocking the async runtime
            tokio::task::spawn_blocking(move || {
                // Use streaming generation - sends tokens via channel as they're generated
                let result = quantized_model_clone.generate_with_cache_streaming(
                    &prompt_ids_clone,
                    &q_config_clone,
                    |token_id| {
                        // Send token through channel; return false to stop if channel closed
                        tx.blocking_send(Ok(token_id)).is_ok()
                    },
                );

                // Send error if generation failed
                if let Err(e) = result {
                    let _ = tx.blocking_send(Err(e.to_string()));
                }
            });

            // Convert channel receiver to SSE stream
            let model_name = request.model.clone();
            let request_id_clone = request_id.clone();
            let tokenizer_clone = tokenizer.clone();
            let metrics = state.metrics.clone();
            let start_time = start;

            let token_stream = ReceiverStream::new(rx);
            let mut completion_tokens = 0usize;

            let stream = async_stream::stream! {
                // Send initial chunk with role
                let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
                if let Ok(data) = serde_json::to_string(&initial) {
                    yield Ok::<_, Infallible>(Event::default().data(data));
                }

                // Stream tokens as they arrive from generation
                tokio::pin!(token_stream);
                while let Some(result) = token_stream.next().await {
                    match result {
                        Ok(token_id) => {
                            completion_tokens += 1;
                            // Decode and send immediately
                            if let Ok(text) = tokenizer_clone.decode(&[token_id]) {
                                // PMAT-088: Clean individual tokens of stop sequences
                                let cleaned = clean_chat_output(&text);
                                if !cleaned.is_empty() {
                                    let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, &cleaned);
                                    if let Ok(data) = serde_json::to_string(&chunk) {
                                        yield Ok(Event::default().data(data));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // Send error chunk
                            let error_chunk = serde_json::json!({
                                "error": e
                            });
                            if let Ok(data) = serde_json::to_string(&error_chunk) {
                                yield Ok(Event::default().data(data));
                            }
                            break;
                        }
                    }
                }

                // Send final chunk with finish reason
                let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
                if let Ok(data) = serde_json::to_string(&done) {
                    yield Ok(Event::default().data(data));
                }

                // Record metrics
                metrics.record_success(completion_tokens, start_time.elapsed());

                // Send [DONE] marker
                yield Ok(Event::default().data("[DONE]"));
            };

            return Sse::new(stream)
                .keep_alive(
                    axum::response::sse::KeepAlive::new()
                        .interval(std::time::Duration::from_secs(15))
                        .text("keep-alive"),
                )
                .into_response();
        }

        // NON-STREAMING: Generate all tokens first, then return
        let generated = match quantized_model.generate_with_cache(&prompt_ids, &q_config) {
            Ok(g) => g,
            Err(e) => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // Skip prompt tokens
        let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
        let completion_tokens = token_ids.len();

        // Non-streaming response - return JSON
        let text = match tokenizer.decode(&token_ids) {
            Ok(t) => t,
            Err(e) => {
                state.metrics.record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
                    .into_response();
            },
        };

        // PMAT-088: Clean output - stop at first stop sequence to prevent prompt injection
        let text = clean_chat_output(&text);

        let latency = start.elapsed();
        state.metrics.record_success(completion_tokens, latency);

        // Build trace data based on X-Trace-Level header (quantized model path)
        let (brick_trace, step_trace, layer_trace) = build_trace_data(
            trace_level.as_deref(),
            latency.as_micros() as u64,
            prompt_tokens,
            completion_tokens,
            28, // Default layer count for Qwen2 models
        );

        return Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            model: request.model.clone(),
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
        .into_response();
    }

    // Fall back to registry-based model lookup
    let model_id = if request.model == "default" || request.model.is_empty() {
        None
    } else {
        Some(request.model.as_str())
    };

    let (model, tokenizer) = match state.get_model(model_id) {
        Ok((m, t)) => (m, t),
        Err(e) => {
            state.metrics.record_failure();
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response();
        },
    };

    // Convert chat messages to prompt using model-specific template
    let prompt_text = format_chat_messages(&request.messages, Some(&request.model));

    // Tokenize prompt
    let prompt_ids = tokenizer.encode(&prompt_text);
    if prompt_ids.is_empty() {
        state.metrics.record_failure();
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Messages cannot be empty".to_string(),
            }),
        )
            .into_response();
    }

    let prompt_tokens = prompt_ids.len();

    // Convert to usize for model
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    // Build generation config
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);

    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p };
    }

    // Generate
    let generated = match model.generate(&prompt, &config) {
        Ok(g) => g,
        Err(e) => {
            state.metrics.record_failure();
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response();
        },
    };

    // Convert back to u32 and decode
    let token_ids: Vec<u32> = match generated
        .iter()
        .map(|&id| u32::try_from(id).map_err(|_| format!("Token ID {id} exceeds u32 range")))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(ids) => ids,
        Err(e) => {
            state.metrics.record_failure();
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })).into_response();
        },
    };

    // Handle streaming for registry models
    if request.stream {
        let generated_ids: Vec<u32> = token_ids[prompt.len()..].to_vec();
        let model_name = request.model.clone();
        let request_id_clone = request_id.clone();
        let completion_tokens = generated_ids.len();

        let stream = async_stream::stream! {
            // Send initial chunk with role
            let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
            if let Ok(data) = serde_json::to_string(&initial) {
                yield Ok::<_, Infallible>(Event::default().data(data));
            }

            // Stream tokens one by one
            for &token_id in &generated_ids {
                if let Ok(text) = tokenizer.decode(&[token_id]) {
                    if !text.is_empty() {
                        let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, &text);
                        if let Ok(data) = serde_json::to_string(&chunk) {
                            yield Ok(Event::default().data(data));
                        }
                    }
                }
            }

            // Send final chunk with finish reason
            let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
            if let Ok(data) = serde_json::to_string(&done) {
                yield Ok(Event::default().data(data));
            }

            // Send [DONE] marker
            yield Ok(Event::default().data("[DONE]".to_string()));
        };

        state
            .metrics
            .record_success(completion_tokens, start.elapsed());
        return Sse::new(stream).into_response();
    }

    // Non-streaming response
    let generated_ids = &token_ids[prompt.len()..];
    let response_text = match tokenizer.decode(generated_ids) {
        Ok(t) => t,
        Err(e) => {
            state.metrics.record_failure();
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response();
        },
    };

    let completion_tokens = generated_ids.len();
    let duration = start.elapsed();

    // Record successful generation
    state.metrics.record_success(completion_tokens, duration);

    Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0),
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
    // Get model and tokenizer
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

    // Convert chat messages to prompt using model-specific template
    let prompt_text = format_chat_messages(&request.messages, Some(&request.model));

    // Tokenize prompt
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

    // Convert to usize for model
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    // Build generation config
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);

    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p };
    }

    // Generate request ID
    let request_id = format!(
        "chatcmpl-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    // Generate all tokens
    let generated = model.generate(&prompt, &config).map_err(|e| {
        state.metrics.record_failure();
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Convert to u32 for tokenizer
    let token_ids: Vec<u32> = generated
        .iter()
        .filter_map(|&id| u32::try_from(id).ok())
        .collect();

    // Get only the generated tokens (skip prompt)
    let generated_ids = token_ids[prompt_len..].to_vec();

    // Clone values for move into stream
    let model_name = request.model.clone();
    let request_id_clone = request_id.clone();
    let tokenizer_clone = tokenizer;

    // Create SSE stream
    let stream = async_stream::stream! {
        // Send initial chunk with role
        let initial = ChatCompletionChunk::initial(&request_id_clone, &model_name);
        let data = serde_json::to_string(&initial).unwrap_or_default();
        yield Ok(Event::default().data(format!("data: {}\n", data)));

        // Stream tokens one by one
        for &token_id in &generated_ids {
            // Decode single token
            let text = match tokenizer_clone.decode(&[token_id]) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let chunk = ChatCompletionChunk::content(&request_id_clone, &model_name, &text);
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            yield Ok(Event::default().data(format!("data: {}\n", data)));
        }

        // Send final chunk
        let done = ChatCompletionChunk::done(&request_id_clone, &model_name);
        let data = serde_json::to_string(&done).unwrap_or_default();
        yield Ok(Event::default().data(format!("data: {}\n", data)));

        // Send [DONE] marker
        yield Ok(Event::default().data("data: [DONE]\n".to_string()));
    };

    Ok(Sse::new(stream))
}
