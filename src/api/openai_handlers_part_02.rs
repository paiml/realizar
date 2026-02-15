
/// CUDA-optimized backend (true streaming support).
#[cfg(feature = "cuda")]
fn try_cuda_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
) -> Option<Response> {
    use crate::gguf::QuantizedGenerateConfig;

    let cuda_model_lock = state.cuda_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    let prompt_ids = match tokenize_chat_prompt(&tokenizer, &request.messages, Some("qwen"), state)
    {
        Ok(ids) => ids,
        Err(r) => return Some(r),
    };
    let prompt_tokens = prompt_ids.len();
    let (max_tokens, temperature, eos_token_id) = chat_gen_params(request, &tokenizer);

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

        return Some(true_streaming_sse_response(
            rx,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            state.metrics.clone(),
            start,
            false,
        ));
    }

    // Non-streaming CUDA
    let mut cuda_model = cuda_model_lock.write().expect("operation failed");
    let generated = match cuda_model.generate_gpu_resident(&prompt_ids, &q_config) {
        Ok(g) => g,
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();
    let response_text = tokenizer
        .decode(&token_ids)
        .unwrap_or_else(|_| String::new());
    let response_text = clean_chat_output(&response_text);

    let latency = start.elapsed();
    state.metrics.record_success(completion_tokens, latency);
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        response_text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        trace_level,
        latency,
    ))
}

/// Quantized model (GGUF serve mode) backend with true streaming.
fn try_quantized_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
) -> Option<Response> {
    use crate::gguf::QuantizedGenerateConfig;

    let quantized_model = state.quantized_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    let prompt_ids = match tokenize_chat_prompt(&tokenizer, &request.messages, Some("qwen"), state)
    {
        Ok(ids) => ids,
        Err(r) => return Some(r),
    };
    let prompt_tokens = prompt_ids.len();
    let (max_tokens, temperature, eos_token_id) = chat_gen_params(request, &tokenizer);

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

        return Some(true_streaming_sse_response(
            rx,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            state.metrics.clone(),
            start,
            true,
        ));
    }

    // Non-streaming quantized
    let generated = match quantized_model.generate_with_cache(&prompt_ids, &q_config) {
        Ok(g) => g,
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();
    let text = match tokenizer.decode(&token_ids) {
        Ok(t) => clean_chat_output(&t),
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let latency = start.elapsed();
    state.metrics.record_success(completion_tokens, latency);
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        trace_level,
        latency,
    ))
}

/// Convert usize token IDs to u32, returning error string on overflow
fn convert_token_ids(ids: &[usize]) -> Result<Vec<u32>, String> {
    ids.iter()
        .map(|&id| u32::try_from(id).map_err(|_| format!("Token ID {id} exceeds u32 range")))
        .collect()
}

/// Build generation config from request parameters
fn build_gen_config(request: &ChatCompletionRequest) -> GenerationConfig {
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);
    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p };
    }
    config
}

/// Registry-based model fallback (no specialized backend).
fn registry_fallback(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    start: Instant,
) -> Response {
    let model_id = if request.model == "default" || request.model.is_empty() {
        None
    } else {
        Some(request.model.as_str())
    };

    let (model, tokenizer) = match state.get_model(model_id) {
        Ok((m, t)) => (m, t),
        Err(e) => return fail_response(state, StatusCode::NOT_FOUND, e),
    };

    let prompt_text = format_chat_messages(&request.messages, Some(&request.model));
    let prompt_ids = tokenizer.encode(&prompt_text);
    if prompt_ids.is_empty() {
        return fail_response(state, StatusCode::BAD_REQUEST, "Messages cannot be empty");
    }

    let prompt_tokens = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();
    let config = build_gen_config(request);

    let generated = match model.generate(&prompt, &config) {
        Ok(g) => g,
        Err(e) => return fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let token_ids: Vec<u32> = match convert_token_ids(&generated) {
        Ok(ids) => ids,
        Err(e) => return fail_response(state, StatusCode::BAD_REQUEST, e),
    };

    let generated_ids: Vec<u32> = token_ids[prompt.len()..].to_vec();
    let completion_tokens = generated_ids.len();

    if request.stream {
        state
            .metrics
            .record_success(completion_tokens, start.elapsed());
        return pregenerated_sse_response(
            generated_ids,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            false,
        );
    }

    let response_text = match tokenizer.decode(&generated_ids) {
        Ok(t) => t,
        Err(e) => return fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let duration = start.elapsed();
    state.metrics.record_success(completion_tokens, duration);

    let max_tokens = request.max_tokens.unwrap_or(256);
    build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        response_text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        None,
        duration,
    )
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

    #[cfg(feature = "gpu")]
    if let Some(r) = try_gpu_backend(&state, &request, &request_id, trace_level.as_deref(), start) {
        return r;
    }

    #[cfg(feature = "gpu")]
    if let Some(r) =
        try_cached_backend(&state, &request, &request_id, trace_level.as_deref(), start)
    {
        return r;
    }

    #[cfg(feature = "cuda")]
    if let Some(r) = try_cuda_backend(&state, &request, &request_id, trace_level.as_deref(), start)
    {
        return r;
    }

    if let Some(r) =
        try_quantized_backend(&state, &request, &request_id, trace_level.as_deref(), start)
    {
        return r;
    }

    registry_fallback(&state, &request, &request_id, start)
}
