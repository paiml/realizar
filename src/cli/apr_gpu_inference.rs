
/// Run APR inference with CUDA GPU acceleration (PMAT-106)
///
/// Uses the APR F32 GPU adapter to convert weights to GpuModel format
/// for high-performance inference.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn run_apr_inference_gpu(
    model_ref: &str,
    file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    verbose: bool,
    trace_config: Option<crate::inference_trace::TraceConfig>,
) -> Result<()> {
    use crate::apr::AprV2Model;
    use crate::apr_transformer::AprTransformer;
    use crate::gpu::adapters::AprF32ToGpuAdapter;
    use crate::gpu::GpuGenerateConfig;
    use crate::inference_trace::{InferenceTracer, ModelInfo, TraceStep};
    use std::path::Path;
    use std::time::Instant;

    // APR-TRACE-001: Create tracer
    let mut tracer = trace_config
        .map(InferenceTracer::new)
        .unwrap_or_else(InferenceTracer::disabled);

    let load_start = Instant::now();

    if verbose {
        println!("Backend: CUDA (GPU)");
        println!("Loading APR model for GPU inference...");
    }

    // Load APR as F32 transformer
    let transformer = AprTransformer::from_apr_bytes(file_data).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "parse_apr".to_string(),
            reason: format!("Failed to parse APR: {e}"),
        }
    })?;

    // APR-TRACE-001: Set model info
    tracer.set_model_info(ModelInfo {
        name: model_ref.to_string(),
        num_layers: transformer.config.num_layers,
        hidden_dim: transformer.config.hidden_dim,
        vocab_size: transformer.config.vocab_size,
        num_heads: transformer.config.num_heads,
        quant_type: Some("APR F32 (GPU)".to_string()),
    });

    // Convert to GpuModel using F32 adapter
    let mut gpu_model = AprF32ToGpuAdapter::to_gpu_model(&transformer).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "apr_to_gpu".to_string(),
            reason: format!("Failed to convert APR to GPU format: {e}"),
        }
    })?;

    // Debug: Compare CPU vs GPU weights and matmul outputs
    if verbose {
        print_gpu_debug_weights(&transformer, &gpu_model);
    }

    let load_time = load_start.elapsed();
    if verbose {
        println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);
    }

    // NOTE: Chat template is applied by the caller (mod.rs) before calling this function.
    // The `prompt` parameter already contains the formatted conversation with chat markers.

    // APR-TRACE-001: Trace tokenization
    tracer.start_step(TraceStep::Tokenize);

    // Use proper tokenizer from sibling tokenizer.json or embedded vocab
    let model_path = Path::new(model_ref);
    let prompt_tokens = AprV2Model::encode_text(model_path, prompt)
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());
    let prompt_len = prompt_tokens.len();

    tracer.trace_encode(prompt, &prompt_tokens, transformer.config.vocab_size);

    if verbose {
        println!("Prompt tokens: {}", prompt_len);
        // F-REGR-231 DEBUG: Show all token IDs
        eprintln!("[DEBUG-TOKENS] All tokens: {:?}", &prompt_tokens);
        println!("Temperature: {:.1}", temperature);
        println!();
    }

    // APR-TRACE-001: Trace embedding
    tracer.start_step(TraceStep::Embed);
    tracer.trace_embed(prompt_len, transformer.config.hidden_dim, None);

    // APR-TRACE-001: Trace transformer blocks (high-level, GPU generation is a black box)
    tracer.start_step(TraceStep::TransformerBlock);

    // Configure generation
    let gen_config = GpuGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature <= 0.01 { 1 } else { 40 },
        stop_tokens: vec![],
        trace: false,
    };

    // Convert prompt tokens to usize for GpuModel
    let prompt_tokens_usize: Vec<usize> = prompt_tokens.iter().map(|&t| t as usize).collect();

    // Run inference
    let gen_start = Instant::now();
    let generated = gpu_model
        .generate(&prompt_tokens_usize, &gen_config)
        .map_err(|e| crate::error::RealizarError::UnsupportedOperation {
            operation: "gpu_generate".to_string(),
            reason: format!("GPU generation failed: {e}"),
        })?;
    let gen_time = gen_start.elapsed();

    // Record transformer block completion (aggregate timing)
    tracer.trace_layer(
        transformer.config.num_layers - 1,
        0,
        None,
        1,
        transformer.config.hidden_dim,
    );

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output (convert usize back to u32)
    let output_tokens: Vec<u32> = generated[prompt_len..].iter().map(|&t| t as u32).collect();
    let output_text = decode_apr_output_tokens(model_path, &output_tokens);

    // APR-TRACE-001: Trace decode for each output token
    for (i, &token) in output_tokens.iter().enumerate() {
        tracer.start_step(TraceStep::Decode);
        let decoded = output_text
            .chars()
            .nth(i.min(output_text.len().saturating_sub(1)))
            .map_or_else(|| format!("<{token}>"), |c| c.to_string());
        tracer.trace_decode(i, token, &decoded, transformer.config.vocab_size);
    }

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "format": "APR",
                "backend": "CUDA",
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time.as_secs_f64() * 1000.0,
                "tokens_per_second": tokens_per_sec,
                "temperature": temperature,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        },
        _ => {
            if verbose {
                println!(
                    "Generated ({tokens_generated} tokens in {:.2}ms):",
                    gen_time.as_secs_f64() * 1000.0
                );
                println!("{output_text}");
                println!();
                println!("Performance: {:.1} tok/s (GPU)", tokens_per_sec);
            } else {
                println!("{output_text}");
            }
        },
    }

    // APR-TRACE-001: Write trace output if enabled
    if tracer.is_enabled() {
        if let Err(e) = tracer.write_output() {
            eprintln!("[TRACE] Warning: Failed to write trace output: {}", e);
        }
    }

    Ok(())
}
