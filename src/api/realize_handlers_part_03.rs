
/// GPU model backend.
#[cfg(feature = "gpu")]
fn try_gpu_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<Option<CompletionResponse>, RErr> {
    use crate::gpu::GpuGenerateConfig;

    let gpu_model_lock = match state.gpu_model() {
        Some(l) => l,
        None => return Ok(None),
    };
    let tokenizer = state.tokenizer.clone().ok_or_else(|| {
        rerr(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "No tokenizer available",
        )
    })?;
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(
            state,
            StatusCode::BAD_REQUEST,
            "Prompt cannot be empty",
        ));
    }
    let prompt_tokens = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    let gpu_config = GpuGenerateConfig {
        max_tokens,
        temperature,
        top_k: 1,
        stop_tokens: Vec::new(),
        trace: false,
    };

    let mut gpu_model = gpu_model_lock.write().map_err(|e| {
        rerr(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("GPU lock: {e}"),
        )
    })?;
    let generated = gpu_model
        .generate(&prompt, &gpu_config)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let token_ids: Vec<u32> = generated
        .iter()
        .skip(prompt_tokens)
        .filter_map(|&id| u32::try_from(id).ok())
        .collect();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    let response_id = format!("cmpl-{}", &uuid::Uuid::new_v4().to_string()[..8]);
    Ok(Some(CompletionResponse {
        id: response_id,
        object: "text_completion".to_string(),
        created: epoch_secs(),
        model: request.model.clone(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// CPU model fallback.
fn registry_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<CompletionResponse, RErr> {
    let model_id = if request.model == "default" || request.model.is_empty() {
        None
    } else {
        Some(request.model.as_str())
    };

    let (model, tokenizer) = state
        .get_model(model_id)
        .map_err(|e| rerr(state, StatusCode::NOT_FOUND, e))?;
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(
            state,
            StatusCode::BAD_REQUEST,
            "Prompt cannot be empty",
        ));
    }
    let prompt_tokens = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p as f32 };
    }

    let generated = model
        .generate(&prompt, &config)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let token_ids: Vec<u32> = generated
        .iter()
        .skip(prompt_tokens)
        .filter_map(|&id| u32::try_from(id).ok())
        .collect();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    Ok(completion_resp(
        "cmpl",
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
    ))
}

pub async fn openai_completions_handler(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, RErr> {
    let start = std::time::Instant::now();
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7) as f32;

    #[cfg(feature = "gpu")]
    if let Some(r) =
        try_cached_completions(&state, &request, max_tokens, temperature, start).await?
    {
        return Ok(Json(r));
    }

    if let Some(r) = try_quantized_completions(&state, &request, max_tokens, temperature, start)? {
        return Ok(Json(r));
    }

    #[cfg(feature = "gpu")]
    if let Some(r) = try_gpu_completions(&state, &request, max_tokens, temperature, start)? {
        return Ok(Json(r));
    }

    Ok(Json(registry_completions(
        &state,
        &request,
        max_tokens,
        temperature,
        start,
    )?))
}

/// OpenAI-compatible embeddings handler (/v1/embeddings)
pub async fn openai_embeddings_handler(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Delegate to native handler
    realize_embed_handler(State(state), Json(request)).await
}
