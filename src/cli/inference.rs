//! Inference runner functions for CLI commands
//!
//! Contains run_gguf_inference, run_gguf_inference_gpu, run_safetensors_inference,
//! and run_apr_inference - extracted from main.rs (PMAT-802).

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]

use crate::error::Result;

/// Sample the next token from logits using temperature scaling
fn sample_next_token(logits: &[f32], temperature: f32) -> u32 {
    use crate::gguf::OwnedQuantizedModel;
    if temperature <= 0.01 {
        // Greedy decoding
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    } else {
        // Temperature sampling with top-k=40
        OwnedQuantizedModel::sample_topk(logits, temperature, 40)
    }
}

/// Print inference results in the requested format
fn print_inference_output(
    model_ref: &str,
    prompt: &str,
    output_text: &str,
    tokens_generated: usize,
    gen_time_ms: f64,
    tokens_per_sec: f64,
    temperature: f32,
    format: &str,
    verbose: bool,
) {
    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time_ms,
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
                    "Generated ({tokens_generated} tokens in {gen_time_ms:.2}ms):"
                );
                println!("{prompt}{output_text}");
                println!();
                println!("Performance: {tokens_per_sec:.1} tok/s");
            } else {
                // Ollama-style clean output: just the response
                println!("{output_text}");
            }
        },
    }
}

/// PAR-051: Print diagnostic info for CPU debug mode
#[allow(clippy::never_loop)]
fn print_cpu_debug_info(
    logits: &[f32],
    prompt_tokens: &[u32],
    model: &crate::gguf::OwnedQuantizedModel,
    mapped: &crate::gguf::MappedGGUFModel,
) {
    let mut top5: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top5.truncate(5);
    eprintln!("[PAR-051] Prompt tokens: {:?}", prompt_tokens);
    eprintln!("[PAR-051] Logits top5 after prefill: {:?}", top5);
    let greedy_token = top5[0].0 as u32;
    let decoded = mapped.model.decode(&[greedy_token]);
    eprintln!(
        "[PAR-051] Greedy next token: {} = {:?}",
        greedy_token, decoded
    );

    let last_token = prompt_tokens[prompt_tokens.len() - 1];
    let embed = model.embed(&[last_token]);
    eprintln!(
        "[PAR-051] Last token {} embed[0..5]: {:?}",
        last_token,
        &embed[..5.min(embed.len())]
    );
    eprintln!("[PAR-051] embed sum: {:.6}", embed.iter().sum::<f32>());

    let cfg = model.config();
    eprintln!(
        "[PAR-051] Config: hidden={}, heads={}/{}, layers={}, vocab={}, eps={:e}",
        cfg.hidden_dim,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.num_layers,
        cfg.vocab_size,
        cfg.eps
    );

    let logit_2 = logits.get(29906).copied().unwrap_or(f32::NAN);
    eprintln!("[PAR-051] Logit for token 29906 ('2'): {:.6}", logit_2);
}

pub fn run_gguf_inference(
    model_ref: &str,
    _file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    force_gpu: bool,
    verbose: bool,
    trace: bool,
) -> Result<()> {
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedKVCache};
    use std::time::Instant;

    // Handle --gpu flag warning when CUDA not available
    #[cfg(not(feature = "cuda"))]
    if force_gpu {
        eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
        eprintln!("Build with: cargo build --features cuda");
        eprintln!();
    }
    // Suppress unused warning when cuda feature not enabled
    #[cfg(not(feature = "cuda"))]
    let _ = force_gpu;

    let load_start = Instant::now();

    // Load model using memory-mapped file (same path as working examples)
    let mapped = MappedGGUFModel::from_path(model_ref).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "mmap_gguf".to_string(),
            reason: format!("Failed to mmap GGUF: {e}"),
        }
    })?;

    // GPU path: Use OwnedQuantizedModel with CUDA acceleration
    #[cfg(feature = "cuda")]
    if force_gpu {
        return run_gguf_inference_gpu(
            &mapped,
            prompt,
            max_tokens,
            temperature,
            format,
            load_start,
            verbose,
        );
    }

    // PAR-126: Five-Whys fix - use OwnedQuantizedModel for fast CPU inference
    // Root cause analysis:
    //   Why-1: CPU path was 14 tok/s vs Ollama's 200 tok/s
    //   Why-2: Old mmap-based transformers use per-matmul allocations
    //   Why-3: Each of 196 matmuls per token allocates/frees Vec
    //   Why-4: Vec allocation overhead + cache pollution from mmap page faults
    //   Why-5: OwnedQuantizedModel copies weights to RAM but uses _into methods
    // Solution: Use OwnedQuantizedModel - slower loading but faster inference
    let model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "load_model".to_string(),
            reason: format!("Failed to load model: {e}"),
        }
    })?;

    let load_time = load_start.elapsed();
    if verbose {
        println!("Backend: CPU (AVX2 + SIMD)");
        println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);
    }

    // Tokenize prompt using GGUF vocabulary
    let mut prompt_tokens: Vec<u32> = mapped
        .model
        .encode(prompt)
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());

    // Prepend BOS token if available
    if let Some(bos) = mapped.model.bos_token_id() {
        prompt_tokens.insert(0, bos);
    }
    let prompt_len = prompt_tokens.len();

    // Get EOS token for stopping
    let eos_token_id = mapped.model.eos_token_id();

    // Debug: show model info and encoded tokens
    let config = model.config();
    if verbose {
        println!(
            "Architecture: {:?}, Hidden: {}, Layers: {}, Heads: {}/{} (KV)",
            mapped.model.architecture(),
            config.hidden_dim,
            config.num_layers,
            config.num_heads,
            config.num_kv_heads
        );
        println!(
            "Prompt tokens: {} (BOS={:?}, EOS={:?})",
            prompt_len,
            mapped.model.bos_token_id(),
            eos_token_id
        );
        println!("Temperature: {:.1}", temperature);
        println!();
    }

    // Run inference with KV cache for O(n) per-token cost
    let gen_start = Instant::now();
    let max_seq_len = prompt_tokens.len() + max_tokens;
    let mut cache = OwnedQuantizedKVCache::from_config(config, max_seq_len);
    let mut all_tokens = prompt_tokens.clone();

    // PMAT-TRACE-GGUF-001: Trace config info
    if trace {
        eprintln!(
            "[TRACE-CACHE] GGUF model: {} layers, hidden_dim={}, vocab={}",
            config.num_layers, config.hidden_dim, config.vocab_size
        );
        eprintln!(
            "[TRACE-CACHE] Prefill: {} tokens, max_gen={}",
            prompt_tokens.len(),
            max_tokens
        );
    }

    // Prefill: process prompt tokens to populate KV cache
    // We process all tokens but keep the logits from the last one
    let prefill_start = Instant::now();
    let mut logits: Vec<f32> = vec![];
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        logits = model.forward_cached(token_id, &mut cache, pos)?;
    }
    if trace {
        eprintln!(
            "[TRACE-CACHE] Prefill complete: {} tokens in {:?}",
            prompt_tokens.len(),
            prefill_start.elapsed()
        );
    }

    // PAR-051: Diagnostic output (gated by CPU_DEBUG=1 environment variable)
    if std::env::var("CPU_DEBUG").is_ok_and(|v| v == "1") {
        print_cpu_debug_info(&logits, &prompt_tokens, &model, &mapped);
    }

    // Decode: generate new tokens one at a time
    // First iteration uses logits from prefill, subsequent use new logits
    for i in 0..max_tokens {
        let token_start = Instant::now();
        // For first iteration, use logits from prefill; otherwise compute new ones
        if i > 0 {
            let position = prompt_tokens.len() + i - 1;
            let last_token = *all_tokens
                .last()
                .expect("all_tokens should not be empty during generation");
            logits = model.forward_cached(last_token, &mut cache, position)?;
        }

        // Sample next token
        let next_token = sample_next_token(&logits, temperature);

        // PMAT-TRACE-GGUF-001: Per-token timing
        if trace && i > 0 {
            let position = prompt_tokens.len() + i - 1;
            eprintln!(
                "[TRACE-CACHE] pos={}: {} layers took {:?}",
                position,
                config.num_layers,
                token_start.elapsed()
            );
        }

        // PERF-002: Debug code removed (was PAR-058-DEBUG and PAR-060)

        // Stop on EOS
        if let Some(eos) = eos_token_id {
            if next_token == eos {
                // PERF-002: eprintln removed for performance
                break;
            }
        }

        all_tokens.push(next_token);
    }

    let generated = all_tokens;
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len() - prompt_len;
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using GGUF vocabulary, replacing SentencePiece markers with spaces
    let output_text = mapped
        .model
        .decode(&generated[prompt_len..])
        .replace('▁', " ");

    print_inference_output(
        model_ref,
        prompt,
        &output_text,
        tokens_generated,
        gen_time.as_secs_f64() * 1000.0,
        tokens_per_sec,
        temperature,
        format,
        verbose,
    );

    Ok(())
}

/// Run GGUF inference with CUDA GPU acceleration
///
/// Uses OwnedQuantizedModel with CUDA backend for high-performance inference.
/// Called when --gpu flag is specified and CUDA feature is enabled.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn run_gguf_inference_gpu(
    mapped: &crate::gguf::MappedGGUFModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    load_start: std::time::Instant,
    verbose: bool,
) -> Result<()> {
    use crate::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig};
    use std::time::Instant;

    if verbose {
        println!("Backend: CUDA (GPU)");
        println!("Creating quantized model with CUDA acceleration...");
    }

    // Create owned quantized model (required for CUDA - can't use borrowed mmap data)
    let quantized_model = OwnedQuantizedModel::from_mapped(mapped).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "create_quantized".to_string(),
            reason: format!("Failed to create quantized model: {e}"),
        }
    })?;

    // Get config info before wrapping
    let vocab_size = quantized_model.config.vocab_size;
    let hidden_dim = quantized_model.config.hidden_dim;
    let num_layers = quantized_model.layers.len();

    // PAR-046: Create OwnedQuantizedModelCuda wrapper for actual GPU acceleration
    // The previous implementation used OwnedQuantizedModel.enable_cuda() which only
    // initialized the executor but forward_cached still used CPU code paths.
    let max_seq_len = 256 + max_tokens; // Allow for prompt + generation
    let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(quantized_model, 0, max_seq_len)
        .map_err(|e| crate::error::RealizarError::UnsupportedOperation {
            operation: "OwnedQuantizedModelCuda::new".to_string(),
            reason: format!("CUDA initialization failed: {e}"),
        })?;
    if verbose {
        println!("  CUDA enabled on GPU: {}", cuda_model.device_name());
    }

    let load_time = load_start.elapsed();
    if verbose {
        println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);
    }

    // Tokenize prompt using GGUF vocabulary
    let mut prompt_tokens: Vec<u32> = mapped
        .model
        .encode(prompt)
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());

    // Prepend BOS token if available
    if let Some(bos) = mapped.model.bos_token_id() {
        prompt_tokens.insert(0, bos);
    }
    let prompt_len = prompt_tokens.len();

    // Get EOS token for stopping
    let eos_token_id = mapped.model.eos_token_id();

    if verbose {
        println!(
            "Vocab size: {}, Hidden dim: {}, Layers: {}",
            vocab_size, hidden_dim, num_layers
        );
        println!(
            "Prompt tokens: {} (BOS={:?}, EOS={:?})",
            prompt_len,
            mapped.model.bos_token_id(),
            eos_token_id
        );
        println!("Temperature: {:.1}", temperature);
        println!();
    }

    // PAR-046: Use CUDA-accelerated generation with GPU-resident KV cache
    // This calls generate_cuda_with_cache -> forward_single_cuda_with_cache -> GPU kernels
    let gen_start = Instant::now();

    // Build stop tokens list
    let mut stop_tokens = Vec::new();
    if let Some(eos) = eos_token_id {
        stop_tokens.push(eos);
    }

    // Configure CUDA generation
    let gen_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature <= 0.01 { 1 } else { 40 },
        stop_tokens,
        trace: false,
    };

    // PAR-047: Use generate_full_cuda_with_cache for maximum GPU acceleration
    // This path uses:
    // - GPU matmul for QKV, output projection, and FFN
    // - GPU incremental_attention_gpu with GQA support (PAR-021)
    // - Proper SwiGLU activation (PAR-015)
    // PAR-057: Use GPU-resident path for maximum performance (pre-uploads weights, minimal syncs)
    // Falls back to generate_full_cuda_with_cache if architecture not supported
    // PAR-058: Test GPU-resident vs standard CUDA path
    // PHASE-13: Skip GPU-resident if SKIP_GPU_RESIDENT=1 (for debugging)
    let skip_gpu_resident = std::env::var("SKIP_GPU_RESIDENT")
        .map(|v| v == "1")
        .unwrap_or(false);
    let generated = if cuda_model.supports_gpu_resident() && !skip_gpu_resident {
        if verbose {
            println!("Using GPU-resident path (pre-uploaded weights, ~2 syncs/token)");
        }
        cuda_model
            .generate_gpu_resident(&prompt_tokens, &gen_config)
            .map_err(|e| crate::error::RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident".to_string(),
                reason: format!("GPU-resident generation failed: {e}"),
            })?
    } else {
        if verbose {
            println!("Using standard CUDA path");
        }
        cuda_model
            .generate_full_cuda_with_cache(&prompt_tokens, &gen_config)
            .map_err(|e| crate::error::RealizarError::UnsupportedOperation {
                operation: "generate_full_cuda_with_cache".to_string(),
                reason: format!("CUDA generation failed: {e}"),
            })?
    };
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using GGUF vocabulary, replacing SentencePiece markers with spaces
    let output_text = mapped
        .model
        .decode(&generated[prompt_len..])
        .replace('▁', " ");

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": "GGUF (CUDA)",
                "backend": "GPU",
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time.as_secs_f64() * 1000.0,
                "tokens_per_second": tokens_per_sec,
                "temperature": temperature,
                "cuda_enabled": true,
                "cuda_device": cuda_model.device_name(),
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
                println!("{prompt}{output_text}");
                println!();
                println!("Performance: {:.1} tok/s (GPU)", tokens_per_sec);
            } else {
                // Ollama-style clean output: just the response
                println!("{output_text}");
            }
        },
    }

    Ok(())
}

/// Run SafeTensors inference with performance timing
pub fn run_safetensors_inference(
    model_ref: &str,
    prompt: &str,
    max_tokens: usize,
    _temperature: f32,
    format: &str,
) -> Result<()> {
    use crate::apr::AprV2Model;
    use crate::safetensors_infer::SafetensorsToAprConverter;
    use std::path::Path;
    use std::time::Instant;

    let load_start = Instant::now();
    let model_path = Path::new(model_ref);

    // Convert SafeTensors to AprTransformer (F32 weights)
    let transformer = SafetensorsToAprConverter::convert(model_path).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "convert_safetensors".to_string(),
            reason: format!("Failed to convert SafeTensors: {e}"),
        }
    })?;

    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);
    println!(
        "Architecture: {} ({} layers, vocab_size={})",
        transformer.config.architecture,
        transformer.config.num_layers,
        transformer.config.vocab_size
    );

    // Use proper tokenizer from sibling tokenizer.json
    let prompt_tokens = AprV2Model::encode_text(model_path, prompt).unwrap_or_else(|| {
        // Fallback: simple char tokenization
        prompt.chars().map(|c| c as u32).collect()
    });
    let prompt_len = prompt_tokens.len();

    println!("Prompt tokens: {}", prompt_len);
    println!();

    // Run inference
    let gen_start = Instant::now();
    let generated = transformer.generate(&prompt_tokens, max_tokens)?;
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using proper tokenizer
    let output_tokens = &generated[prompt_len..];
    let output_text = if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        tokenizer.decode(output_tokens)
    } else {
        output_tokens
            .iter()
            .map(|&t| char::from_u32(t.min(127)).unwrap_or('?'))
            .collect()
    };

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "format": "SafeTensors",
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time.as_secs_f64() * 1000.0,
                "tokens_per_second": tokens_per_sec,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        },
        _ => {
            println!(
                "Generated ({tokens_generated} tokens in {:.2}ms):",
                gen_time.as_secs_f64() * 1000.0
            );
            println!("{output_text}");
            println!();
            println!("Performance: {:.1} tok/s", tokens_per_sec);
        },
    }

    Ok(())
}

/// Run APR inference with performance timing
///
/// Supports both CPU and GPU backends (PMAT-106).
pub fn run_apr_inference(
    model_ref: &str,
    file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    force_gpu: bool,
    verbose: bool,
) -> Result<()> {
    use crate::apr::AprV2Model;
    use crate::apr_transformer::AprTransformer;
    use std::path::Path;
    use std::time::Instant;

    // Handle --gpu flag warning when CUDA not available
    #[cfg(not(feature = "cuda"))]
    if force_gpu {
        eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
        eprintln!("Build with: cargo build --features cuda");
        eprintln!();
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (force_gpu, verbose);

    // PMAT-106: GPU path for APR models
    #[cfg(feature = "cuda")]
    if force_gpu {
        return run_apr_inference_gpu(
            model_ref,
            file_data,
            prompt,
            max_tokens,
            temperature,
            format,
            verbose,
        );
    }

    let load_start = Instant::now();

    // Load APR transformer (CPU path)
    let transformer = AprTransformer::from_apr_bytes(file_data).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "parse_apr".to_string(),
            reason: format!("Failed to parse APR: {e}"),
        }
    })?;

    let load_time = load_start.elapsed();
    if verbose {
        println!("Backend: CPU (AVX2 + SIMD)");
        println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);
    }

    // NOTE: Chat template is applied by the caller (mod.rs) before calling this function.
    // The `prompt` parameter already contains the formatted conversation with chat markers.

    // Use proper tokenizer from sibling tokenizer.json or embedded vocab
    let model_path = Path::new(model_ref);
    let prompt_tokens = AprV2Model::encode_text(model_path, prompt).unwrap_or_else(|| {
        // Fallback: simple char tokenization
        prompt.chars().map(|c| c as u32).collect()
    });
    let prompt_len = prompt_tokens.len();

    if verbose {
        println!("Prompt tokens: {}", prompt_len);
        println!("Temperature: {:.1} (using greedy decoding)", temperature);
        println!();
    }

    // Run inference with timing
    // PMAT-103 FIX: Use generate_with_cache for O(n) instead of O(n²) complexity
    let gen_config = crate::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        ..Default::default()
    };
    let gen_start = Instant::now();
    let generated = transformer.generate_with_cache(&prompt_tokens, &gen_config)?;
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using proper tokenizer (PMAT-171)
    let output_tokens = &generated[prompt_len..];
    let output_text = decode_apr_output_tokens(model_path, output_tokens);

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "format": "APR",
                "backend": "CPU",
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
                println!("Performance: {:.1} tok/s", tokens_per_sec);
            } else {
                // Clean output: just the response
                println!("{output_text}");
            }
        },
    }

    Ok(())
}

/// Decode APR output tokens using the best available tokenizer (PMAT-171)
///
/// Tries embedded vocabulary first, then external tokenizer.json, then ASCII fallback.
fn decode_apr_output_tokens(model_path: &std::path::Path, output_tokens: &[u32]) -> String {
    use crate::apr::AprV2Model;

    let model = AprV2Model::load(model_path).ok();
    if let Some(ref m) = model {
        if let Some(simple_tok) = m.load_embedded_tokenizer() {
            return AprV2Model::decode_tokens(&simple_tok.id_to_token, output_tokens);
        }
        if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
            return tokenizer.decode(output_tokens);
        }
    } else if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        return tokenizer.decode(output_tokens);
    }
    // Ultimate fallback: simple ASCII
    output_tokens
        .iter()
        .map(|&t| char::from_u32(t.min(127)).unwrap_or('?'))
        .collect()
}

/// Print verbose debug weight comparison between CPU and GPU models
#[cfg(feature = "cuda")]
fn print_gpu_debug_weights(
    transformer: &crate::apr_transformer::AprTransformer,
    gpu_model: &crate::gpu::GpuModel,
) {
    let has_gate = transformer
        .layers
        .first()
        .is_some_and(|l| l.ffn_gate_weight.is_some());
    eprintln!(
        "[DEBUG-SwiGLU] APR transformer has gate weight: {}",
        has_gate
    );
    if has_gate {
        let gate_len = transformer.layers[0]
            .ffn_gate_weight
            .as_ref()
            .map_or(0, Vec::len);
        eprintln!(
            "[DEBUG-SwiGLU] Gate weight elements: {} (expected: {}x{}={})",
            gate_len,
            transformer.config.hidden_dim,
            transformer.config.intermediate_dim,
            transformer.config.hidden_dim * transformer.config.intermediate_dim
        );
    }

    let has_gpu_gate = gpu_model
        .block_weights
        .first()
        .is_some_and(|b| b.ffn_gate_weight.is_some());
    eprintln!("[DEBUG-SwiGLU] GpuModel has gate weight: {}", has_gpu_gate);

    if let Some(layer0) = transformer.layers.first() {
        eprintln!(
            "[DEBUG-WEIGHT] CPU qkv_weight first 5: {:?}",
            &layer0.qkv_weight[0..5.min(layer0.qkv_weight.len())]
        );
        eprintln!(
            "[DEBUG-WEIGHT] GPU qkv_weight first 5: {:?}",
            &gpu_model.block_weights[0].qkv_weight
                [0..5.min(gpu_model.block_weights[0].qkv_weight.len())]
        );
        eprintln!(
            "[DEBUG-WEIGHT] CPU fc1 (up) first 5: {:?}",
            &layer0.ffn_up_weight[0..5.min(layer0.ffn_up_weight.len())]
        );
        eprintln!(
            "[DEBUG-WEIGHT] GPU fc1 (up) first 5: {:?}",
            &gpu_model.block_weights[0].ffn_fc1_weight
                [0..5.min(gpu_model.block_weights[0].ffn_fc1_weight.len())]
        );

        let hidden_dim = transformer.config.hidden_dim;
        let test_embedding = &transformer.token_embedding[0..hidden_dim];

        // CPU matmul: y = x @ W where W is [out_dim, in_dim] (transposed internally)
        let cpu_qkv_dim = layer0.qkv_weight.len() / hidden_dim;
        let mut cpu_qkv = vec![0.0f32; cpu_qkv_dim];
        for o in 0..cpu_qkv_dim {
            let w_start = o * hidden_dim;
            let mut sum = 0.0f32;
            for i in 0..hidden_dim {
                sum += test_embedding[i] * layer0.qkv_weight[w_start + i];
            }
            cpu_qkv[o] = sum;
        }

        // GPU matmul: y = x @ W_t where W_t is [in_dim, out_dim] (already transposed)
        let gpu_qkv_weight = &gpu_model.block_weights[0].qkv_weight;
        let gpu_qkv_dim = gpu_qkv_weight.len() / hidden_dim;
        let mut gpu_qkv = vec![0.0f32; gpu_qkv_dim];
        for j in 0..gpu_qkv_dim {
            let mut sum = 0.0f32;
            for i in 0..hidden_dim {
                sum += test_embedding[i] * gpu_qkv_weight[i * gpu_qkv_dim + j];
            }
            gpu_qkv[j] = sum;
        }

        eprintln!(
            "[DEBUG-MATMUL] CPU QKV first 5: {:?}",
            &cpu_qkv[0..5.min(cpu_qkv.len())]
        );
        eprintln!(
            "[DEBUG-MATMUL] GPU QKV first 5: {:?}",
            &gpu_qkv[0..5.min(gpu_qkv.len())]
        );

        let max_diff: f32 = cpu_qkv
            .iter()
            .zip(gpu_qkv.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[DEBUG-MATMUL] Max diff: {}", max_diff);
        if max_diff > 0.01 {
            eprintln!("[DEBUG-MATMUL] WARNING: CPU vs GPU QKV mismatch!");
        }
    }
}

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
) -> Result<()> {
    use crate::apr::AprV2Model;
    use crate::apr_transformer::AprTransformer;
    use crate::gpu::adapters::AprF32ToGpuAdapter;
    use crate::gpu::GpuGenerateConfig;
    use std::path::Path;
    use std::time::Instant;

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

    // Use proper tokenizer from sibling tokenizer.json or embedded vocab
    let model_path = Path::new(model_ref);
    let prompt_tokens = AprV2Model::encode_text(model_path, prompt)
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());
    let prompt_len = prompt_tokens.len();

    if verbose {
        println!("Prompt tokens: {}", prompt_len);
        // F-REGR-231 DEBUG: Show all token IDs
        eprintln!("[DEBUG-TOKENS] All tokens: {:?}", &prompt_tokens);
        println!("Temperature: {:.1}", temperature);
        println!();
    }

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

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output (convert usize back to u32)
    let output_tokens: Vec<u32> = generated[prompt_len..].iter().map(|&t| t as u32).collect();
    let output_text = decode_apr_output_tokens(model_path, &output_tokens);

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

    Ok(())
}
