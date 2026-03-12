
/// Try GGUF GPU generation. Takes model by value to avoid expensive clone (~1GB).
/// Returns `Ok(result)` on GPU success, `Err(model)` to return model for CPU fallback.
#[cfg(feature = "cuda")]
fn try_gguf_gpu_generate(
    model: crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    verbose: bool,
) -> std::result::Result<Result<(Vec<u32>, bool)>, crate::gguf::OwnedQuantizedModel> {
    use crate::gguf::OwnedQuantizedModelCuda;

    let mut cuda_model = match OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048) {
        Ok(m) => m,
        Err(e) => {
            if verbose {
                eprintln!("Backend: CPU (GPU unavailable: {})", e);
            }
            // Model is preserved inside CudaInitError for CPU fallback
            return Err(e.into_model());
        },
    };

    if verbose {
        eprintln!(
            "Backend: GPU ({}, {} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        );
    }

    if !validate_gpu_first_token(&mut cuda_model, gen_config) {
        // Validation failed — extract model back for CPU fallback
        return Err(cuda_model.into_model());
    }

    // Reuse existing CUDA model — generate_gpu_resident() creates fresh KV cache
    // and resets GPU KV positions internally, so validation doesn't "consume" it.
    let result = cuda_model
        .generate_gpu_resident(input_tokens, gen_config)
        .map(|tokens| (tokens, true))
        .map_err(|e| RealizarError::InferenceError(format!("GPU generation failed: {}", e)));
    Ok(result)
}

/// Run GGUF generation with GPU or CPU
#[allow(unused_variables)] // config used only in CUDA feature
fn run_gguf_generate(
    model: crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    config: &InferenceConfig,
) -> Result<(Vec<u32>, bool)> {
    let has_legacy_quant = model_has_legacy_quant(&model);

    // GPU path: pass model by value (zero-clone) — model is returned on failure for CPU fallback
    #[cfg(feature = "cuda")]
    let model = if !config.no_gpu && !has_legacy_quant {
        match try_gguf_gpu_generate(model, input_tokens, gen_config, config.verbose) {
            Ok(result) => return result,
            Err(returned_model) => returned_model, // GPU failed, use returned model for CPU
        }
    } else {
        model
    };

    log_cpu_backend(config.verbose, has_legacy_quant);
    let tokens = model
        .generate_with_cache(input_tokens, gen_config)
        .map_err(|e| RealizarError::InferenceError(format!("CPU generation failed: {}", e)))?;
    Ok((tokens, false))
}

/// Run APR model inference (PAR-302, PMAT-APR-CUDA-001)
///
/// Uses AprV2ModelCuda for GPU acceleration when available, falls back to
/// AprTransformer (CPU with proper RoPE and SwiGLU) otherwise.
/// PMAT-237: APR inference now uses PreparedTokens (compile-time enforced chat template).
/// Previously bypassed PreparedTokens entirely via prepare_apr_input_tokens().
fn run_apr_inference(
    config: &InferenceConfig,
    prepared: &PreparedTokens,
) -> Result<InferenceResult> {
    if config.verbose {
        eprintln!("Loading APR model: {}", config.model_path.display());
    }

    let load_start = Instant::now();
    let input_tokens = prepared.tokens();
    let input_token_count = prepared.input_count();

    // Try GPU path first
    #[cfg(feature = "cuda")]
    if !config.no_gpu {
        if let Some(result) =
            try_apr_cuda_inference(config, input_tokens, input_token_count, load_start)
        {
            return result;
        }
    }

    // CPU fallback: AprTransformer with RoPE and SwiGLU
    run_apr_cpu_inference(config, input_tokens, input_token_count, load_start)
}

/// GH-318: Map APR architecture string to chat template hint using contract.
///
/// Uses `normalize_architecture()` from tensor-names-v1.yaml — no fallback.
/// Unknown architectures default to "llama" (safest default per contract).
fn apr_arch_to_template_hint(apr_arch: &str, _model_name: &str) -> &'static str {
    crate::tensor_names::normalize_architecture(apr_arch)
}

/// Metadata captured from the model config before it is moved into CUDA.
#[cfg(feature = "cuda")]
struct AprCudaModelInfo {
    arch: String,
    num_layers: usize,
    vocab_size: usize,
    hidden_dim: usize,
}

/// Load an APR model and initialize it on CUDA, returning None on any failure.
#[cfg(feature = "cuda")]
fn load_apr_cuda_model(
    model_path: &std::path::Path,
    verbose: bool,
) -> Option<(crate::gguf::OwnedQuantizedModelCuda, AprCudaModelInfo)> {
    use crate::apr::MappedAprModel;
    use crate::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let mapped = MappedAprModel::from_path(model_path).map_err(|e| {
        if verbose { eprintln!("[APR-CUDA] MappedAprModel::from_path failed: {}", e); }
    }).ok()?;

    let model = OwnedQuantizedModel::from_apr(&mapped).map_err(|e| {
        if verbose { eprintln!("[APR-CUDA] OwnedQuantizedModel::from_apr failed: {}", e); }
    }).ok()?;

    if model_has_legacy_quant(&model) {
        return None;
    }

    let info = AprCudaModelInfo {
        arch: model.config.architecture.clone(),
        num_layers: model.config.num_layers,
        vocab_size: model.config.vocab_size,
        hidden_dim: model.config.hidden_dim,
    };

    let cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048).map_err(|e| {
        if verbose { eprintln!("Backend: CPU (GPU unavailable: {})", e); }
    }).ok()?;

    Some((cuda_model, info))
}

#[cfg(feature = "cuda")]
fn log_apr_cuda_info(
    info: &AprCudaModelInfo,
    cuda_model: &crate::gguf::OwnedQuantizedModelCuda,
    load_ms: f64,
) {
    eprintln!(
        "Architecture: {} ({} layers, vocab_size={})",
        info.arch, info.num_layers, info.vocab_size
    );
    eprintln!(
        "Config: hidden_size={}, quant=CUDA+KVCache, threads=1 (GPU)",
        info.hidden_dim
    );
    eprintln!("Model loaded in {:.1}ms", load_ms);
    eprintln!(
        "Backend: GPU ({}, {} MB VRAM)",
        cuda_model.device_name(),
        cuda_model.vram_mb()
    );
}

/// Try APR CUDA inference, returning None to fall through to CPU.
///
/// Converts APR Q4K model to `OwnedQuantizedModel` and uses the proven GGUF CUDA
/// pipeline (same path as `try_gguf_gpu_generate`). The previous wgpu path used
/// `AprF32ToGpuAdapter` which only reads F32 fields — empty for Q4K models → garbage.
#[cfg(feature = "cuda")]
fn try_apr_cuda_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Option<Result<InferenceResult>> {
    use crate::gguf::QuantizedGenerateConfig;

    let (mut cuda_model, info) = load_apr_cuda_model(&config.model_path, config.verbose)?;

    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        log_apr_cuda_info(&info, &cuda_model, load_ms);
    }

    // GH-373: EOS from model config + caller stop tokens + sibling tokenizer
    let stop_tokens = resolve_apr_stop_tokens(
        &cuda_model.model().config.eos_token_id,
        &config.stop_tokens,
        &config.model_path,
    );
    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens,
        trace: false,
    };

    if !validate_gpu_first_token(&mut cuda_model, &gen_config) {
        return None;
    }

    let infer_start = Instant::now();

    let tokens = match cuda_model.generate_gpu_resident(input_tokens, &gen_config) {
        Ok(t) => t,
        Err(e) => {
            let msg = e.to_string();
            // GH-278: Fall back to CPU for unsupported architectures (GPT-2 has no SwiGLU/RMSNorm)
            if msg.contains("not supported") || msg.contains("architecture") {
                if config.verbose {
                    eprintln!("[APR-CUDA] GPU-resident not supported, falling back to CPU: {msg}");
                }
                return None;
            }
            return Some(Err(RealizarError::InferenceError(format!(
                "GPU generation failed: {}",
                e
            ))));
        },
    };

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Some(Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "APR".to_string(),
        used_gpu: true,
    }))
}

/// Log model architecture and configuration details for APR CPU inference.
fn log_apr_cpu_model_info(
    verbose: bool,
    validated: &crate::safetensors::validation::ValidatedAprTransformer,
    load_ms: f64,
) {
    if !verbose {
        return;
    }
    let arch = &validated.config.architecture;
    let thread_count = rayon::current_num_threads();
    eprintln!(
        "Architecture: {} ({} layers, vocab_size={})",
        arch, validated.config.num_layers, validated.config.vocab_size
    );
    eprintln!(
        "Config: hidden_size={}, context_length={}, quant=F32 (dequantized), threads={}",
        validated.config.hidden_dim, validated.config.context_length, thread_count
    );
    eprintln!("Model loaded in {:.1}ms", load_ms);
    eprintln!("Backend: CPU (SIMD-accelerated)");
}

/// Try loading an APR model as LLaMA-style. Returns None if the model
/// needs the quantized path (GPT-2, load failure, large model OOM guard, etc.).
fn try_load_llama_style(
    model_path: &std::path::Path,
) -> Option<crate::safetensors::validation::ValidatedAprTransformer> {
    // GH-478: Skip F32 dequant path for large models to avoid OOM.
    // AprTransformer reads the entire file into Vec<u8> then dequantizes ALL
    // Q4K tensors to F32 eagerly. Peak memory ≈ file_size × 8.
    // For 32B Q4K (19 GB on disk), peak = 152 GB — exceeds most systems.
    // Route to OwnedQuantizedModel which keeps weights in Q4K form.
    if exceeds_f32_dequant_limit(model_path) {
        return None;
    }

    match crate::apr_transformer::AprTransformer::from_apr_file_validated(model_path) {
        Ok(t) => {
            let arch = t.config.architecture.to_lowercase();
            if arch.contains("gpt2") || arch.contains("gpt-2") {
                None
            } else {
                Some(t)
            }
        }
        Err(_) => None,
    }
}

/// GH-478: Check if F32 dequantization would exceed system memory.
///
/// Delegates to `contract_gate::exceeds_f32_dequant_estimate()` — the contract
/// system is the single source of truth for resource limit checks.
///
/// This pre-dispatch check prevents reading the file at all for large models,
/// routing directly to `OwnedQuantizedModel` (keeps weights in Q4K form).
/// A precise check also fires inside `from_apr_bytes()` via
/// `validate_f32_dequant_limits()` as a safety net.
fn exceeds_f32_dequant_limit(model_path: &std::path::Path) -> bool {
    let file_size = std::fs::metadata(model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let exceeds = crate::contract_gate::exceeds_f32_dequant_estimate(file_size);
    if exceeds {
        let mem_total = crate::contract_gate::system_memory_bytes().unwrap_or(0);
        eprintln!(
            "[GH-478] Model {} GB on disk → ~{} GB F32 dequant (system RAM: {} GB). \
             Using quantized CPU inference to avoid OOM.",
            file_size / (1 << 30),
            file_size.saturating_mul(8) / (1 << 30),
            mem_total / (1 << 30),
        );
    }
    exceeds
}

/// Run APR inference on CPU with KV-cache (PMAT-103)
fn run_apr_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Result<InferenceResult> {
    // GH-278: AprTransformer only supports LLaMA-style models (RoPE + SwiGLU).
    // For GPT-2 and other architectures, use OwnedQuantizedModel which supports
    // learned position embeddings, LayerNorm, GELU, etc.
    let validated = match try_load_llama_style(&config.model_path) {
        Some(t) => t,
        None => return run_apr_quantized_cpu_inference(config, input_tokens, input_token_count, load_start),
    };
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    log_apr_cpu_model_info(config.verbose, &validated, load_ms);

    // GH-373: Resolve stop tokens from model config, caller, and sibling tokenizer
    let stop_tokens = resolve_apr_stop_tokens(
        &validated.config.eos_token_id,
        &config.stop_tokens,
        &config.model_path,
    );
    if config.verbose && !stop_tokens.is_empty() {
        eprintln!("Stop tokens: {:?}", stop_tokens);
    }

    let infer_start = Instant::now();
    let mut all_tokens = input_tokens.to_vec();
    let mut cache = crate::apr_transformer::AprKVCache::new(&validated.config);

    // Prefill: populate KV cache
    for (pos, &token) in input_tokens.iter().enumerate() {
        let _ = validated.forward_with_cache(token, &mut cache, pos)?;
    }

    // Generate with KV cache (O(1) per token)
    let mut position = input_tokens.len();
    for _ in 0..config.max_tokens {
        let last_token = *all_tokens.last().unwrap_or(&1);
        let logits = validated.forward_with_cache(last_token, &mut cache, position)?;

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);

        // GH-373: EOS from model config + sibling tokenizer + caller
        if next_token == 0 || stop_tokens.contains(&next_token) {
            break;
        }

        all_tokens.push(next_token);
        position += 1;
    }

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &all_tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "APR".to_string(),
        used_gpu: false,
    })
}

/// GH-278: CPU inference for APR models using OwnedQuantizedModel
///
/// Used for architectures not supported by AprTransformer (GPT-2, etc.).
/// AprTransformer only supports LLaMA-style (RoPE + SwiGLU).
fn run_apr_quantized_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Result<InferenceResult> {
    use crate::apr::MappedAprModel;
    use crate::gguf::{OwnedQuantizedModel, QuantizedGenerateConfig};

    let mapped = MappedAprModel::from_path(&config.model_path)?;
    let model = OwnedQuantizedModel::from_apr(&mapped)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            model.config.architecture, model.config.num_layers, model.config.vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, quant=Q4_K (OwnedQuantizedModel CPU), threads={}",
            model.config.hidden_dim,
            rayon::current_num_threads()
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!("Backend: CPU (OwnedQuantizedModel fallback for non-LLaMA arch)");
    }

    // GH-373: Resolve stop tokens for quantized path
    let stop_tokens = resolve_apr_stop_tokens(
        &model.config.eos_token_id,
        &config.stop_tokens,
        &config.model_path,
    );

    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_k: config.top_k,
        stop_tokens,
        trace: config.trace,
    };

    let infer_start = Instant::now();
    let tokens = model.generate_with_cache(input_tokens, &gen_config)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "APR".to_string(),
        used_gpu: false,
    })
}

/// GH-373: Resolve stop tokens from model config, caller, and sibling tokenizer.
///
/// Merges EOS tokens from three sources:
/// 1. Model config (`eos_token_id` from APR/GGUF metadata)
/// 2. Caller-provided stop tokens (`InferenceConfig.stop_tokens`)
/// 3. Sibling tokenizer (ChatML markers like `<|im_end|>`, `<|endoftext|>`)
fn resolve_apr_stop_tokens(
    model_eos: &Option<u32>,
    caller_stop_tokens: &[u32],
    model_path: &std::path::Path,
) -> Vec<u32> {
    let mut tokens: Vec<u32> = model_eos.into_iter().copied().collect();

    // Caller-provided stop tokens
    for &t in caller_stop_tokens {
        if !tokens.contains(&t) {
            tokens.push(t);
        }
    }

    // Sibling tokenizer fallback (GH-373)
    if tokens.is_empty() {
        tokens = resolve_stop_tokens_from_tokenizer(model_path);
    }

    tokens
}

/// Load stop tokens from sibling tokenizer.json (GH-373 helper)
fn resolve_stop_tokens_from_tokenizer(model_path: &std::path::Path) -> Vec<u32> {
    let tokenizer = match crate::apr::AprV2Model::load_tokenizer(model_path) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let mut tokens: Vec<u32> = tokenizer.eos_id.into_iter().collect();

    // ChatML stop tokens for instruct models
    for marker in &["<|im_end|>", "<|endoftext|>"] {
        let id = tokenizer
            .special_tokens
            .get(*marker)
            .or_else(|| tokenizer.token_to_id.get(*marker));
        if let Some(&id) = id {
            if !tokens.contains(&id) {
                tokens.push(id);
            }
        }
    }

    tokens
}

/// Decode APR output tokens using available tokenizer (GH-156)
fn decode_apr_tokens(model_path: &std::path::Path, tokens: &[u32]) -> String {
    use crate::apr::AprV2Model;

    let text = if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        tokenizer.decode(tokens)
    } else if let Some(tokenizer) = find_fallback_tokenizer(model_path) {
        tokenizer.decode(tokens)
    } else {
        format!("[{} tokens generated, tokenizer not found]", tokens.len())
    };
    clean_model_output(&text)
}

/// Compute tokens per second from count and elapsed milliseconds
fn tok_per_sec(count: usize, ms: f64) -> f64 {
    if ms > 0.0 {
        count as f64 / (ms / 1000.0)
    } else {
        0.0
    }
}

/// Run SafeTensors model inference (PAR-301, PMAT-129)
///
/// PMAT-236: Accepts `PreparedTokens` (compile-time enforced chat template).
/// Previously, this function raw-encoded prompts WITHOUT chat template,
/// producing garbage output for instruct models.
fn run_safetensors_inference(
    config: &InferenceConfig,
    prepared: &PreparedTokens,
) -> Result<InferenceResult> {
    if config.verbose {
        eprintln!("Loading SafeTensors model: {}", config.model_path.display());
    }

    // PMAT-236: Use PreparedTokens (chat template already applied by prepare_tokens)
    let input_tokens = prepared.tokens().to_vec();
    let input_token_count = prepared.input_count();

    // PMAT-129: Try GPU path first
    #[cfg(feature = "cuda")]
    if !config.no_gpu {
        if let Some(result) =
            try_safetensors_cuda_inference(config, &input_tokens, input_token_count)
        {
            return result;
        }
    }

    // CPU fallback: SafeTensors → AprTransformer conversion
    run_safetensors_cpu_inference(config, &input_tokens, input_token_count)
}

/// Try SafeTensors CUDA inference, returning None to fall through to CPU
#[cfg(feature = "cuda")]
fn try_safetensors_cuda_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
) -> Option<Result<InferenceResult>> {
    use crate::safetensors_cuda::SafeTensorsCudaModel;

    let load_start = Instant::now();
    let mut cuda_model = match SafeTensorsCudaModel::load(&config.model_path, 0) {
        Ok(m) => m,
        Err(e) => {
            if config.verbose {
                eprintln!("Backend: CPU (GPU init failed: {})", e);
            }
            return None;
        },
    };

    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        eprintln!(
            "Architecture: SafeTensors ({} layers, vocab_size={})",
            cuda_model.config().num_layers,
            cuda_model.config().vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, context_length={}, quant=F16/BF16, threads=1 (GPU)",
            cuda_model.config().hidden_dim,
            cuda_model.config().context_length
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!(
            "Backend: GPU ({}, {} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        );
    }

    let infer_start = Instant::now();
    // GH-330: EOS from model config (Design by Contract)
    let eos_id = cuda_model.config().eos_token_id.unwrap_or(0);
    let tokens = match cuda_model.generate(input_tokens, config.max_tokens, eos_id) {
        Ok(t) => t,
        Err(e) => {
            return Some(Err(RealizarError::InferenceError(format!(
                "GPU generation failed: {}",
                e
            ))))
        },
    };

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Some(Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "SafeTensors".to_string(),
        used_gpu: true,
    }))
}
