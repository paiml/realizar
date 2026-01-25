//! High-level inference API for CLI tools
//!
//! This module provides a simple, high-level API for running inference
//! that can be used by CLI tools like `apr run` and `apr chat`.
//!
//! # Architecture (APR-CLI-DELEGATE-001)
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │  apr-cli    │ --> │  realizar   │ --> │   trueno    │
//! │  (100 LOC)  │     │   infer.rs  │     │   SIMD/GPU  │
//! └─────────────┘     └─────────────┘     └─────────────┘
//! ```
//!
//! The `apr run` command delegates ALL inference to this module.
//! This eliminates ~1800 lines of duplicated code in apr-cli.
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::infer::{InferenceConfig, run_inference};
//!
//! let config = InferenceConfig::new("model.gguf")
//!     .with_prompt("Hello, world!")
//!     .with_max_tokens(32);
//!
//! let result = run_inference(config)?;
//! println!("{}", result.text);
//! ```

use crate::error::{RealizarError, Result};
use crate::format::{detect_format, ModelFormat};
use std::path::PathBuf;
use std::time::Instant;

/// Configuration for inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Path to model file (GGUF, APR, or SafeTensors)
    pub model_path: PathBuf,
    /// Text prompt for generation
    pub prompt: Option<String>,
    /// Token IDs for generation (alternative to prompt)
    pub input_tokens: Option<Vec<u32>>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Disable GPU acceleration
    pub no_gpu: bool,
    /// Enable inference tracing (APR-TRACE-001)
    pub trace: bool,
    /// Verbose tracing output
    pub trace_verbose: bool,
    /// Trace output file path
    pub trace_output: Option<PathBuf>,
    /// Specific trace steps to capture
    pub trace_steps: Option<Vec<String>>,
    /// Show verbose loading/progress output
    pub verbose: bool,
}

impl InferenceConfig {
    /// Create a new inference config for a model file
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            prompt: None,
            input_tokens: None,
            max_tokens: 32,
            temperature: 0.0, // Greedy by default
            top_k: 1,
            no_gpu: false,
            trace: false,
            trace_verbose: false,
            trace_output: None,
            trace_steps: None,
            verbose: false,
        }
    }

    /// Set the text prompt
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set input tokens directly
    #[must_use]
    pub fn with_input_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.input_tokens = Some(tokens);
        self
    }

    /// Set maximum tokens to generate
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature (0.0 = greedy)
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k sampling
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Disable GPU acceleration
    #[must_use]
    pub fn without_gpu(mut self) -> Self {
        self.no_gpu = true;
        self
    }

    /// Enable verbose output
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable inference tracing
    #[must_use]
    pub fn with_trace(mut self, trace: bool) -> Self {
        self.trace = trace;
        self
    }
}

/// Result from inference
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Generated text (decoded from tokens)
    pub text: String,
    /// All tokens (input + generated)
    pub tokens: Vec<u32>,
    /// Number of input tokens
    pub input_token_count: usize,
    /// Number of generated tokens
    pub generated_token_count: usize,
    /// Inference time in milliseconds
    pub inference_ms: f64,
    /// Tokens per second
    pub tok_per_sec: f64,
    /// Model load time in milliseconds
    pub load_ms: f64,
    /// Model format that was loaded
    pub format: String,
    /// Whether GPU was used
    pub used_gpu: bool,
}

/// Run inference on a model
///
/// This is the main entry point for inference. It handles:
/// - Model format detection (GGUF, APR, SafeTensors)
/// - Tokenization (using embedded tokenizer for GGUF)
/// - Generation with configurable sampling
/// - GPU acceleration when available
/// - Inference tracing (APR-TRACE-001)
///
/// # Errors
///
/// Returns error if:
/// - Model file cannot be read
/// - Model format is unsupported
/// - Generation fails
pub fn run_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    // Read model file header for format detection
    let data = std::fs::read(&config.model_path).map_err(|e| RealizarError::IoError {
        message: format!("Failed to read model: {}", e),
    })?;

    if data.len() < 8 {
        return Err(RealizarError::FormatError {
            reason: "File too small for format detection".to_string(),
        });
    }

    // Detect format
    let format = detect_format(&data[..8]).map_err(|e| RealizarError::FormatError {
        reason: format!("Format detection failed: {}", e),
    })?;

    match format {
        ModelFormat::Gguf => run_gguf_inference(config),
        ModelFormat::Apr => run_apr_inference(config),
        ModelFormat::SafeTensors => run_safetensors_inference(config),
    }
}

/// Run GGUF model inference
fn run_gguf_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    use crate::chat_template::{format_messages, ChatMessage};
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    // Verbose: Show loading message BEFORE loading (NOISY-GUARD F-UX-27)
    if config.verbose {
        eprintln!("Loading model: {}", config.model_path.display());
    }

    let load_start = Instant::now();

    // Load GGUF via mmap
    let mapped = MappedGGUFModel::from_path(&config.model_path)?;

    // Pre-fault mmap pages (PAR-200: avoid page faults during inference)
    prefault_mmap(mapped.data());

    // Create quantized model
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // Extract architecture from model name
    let arch = config
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map_or("Transformer", |s| {
            if s.to_lowercase().contains("qwen") {
                "Qwen2"
            } else if s.to_lowercase().contains("llama") {
                "LLaMA"
            } else if s.to_lowercase().contains("mistral") {
                "Mistral"
            } else if s.to_lowercase().contains("phi") {
                "Phi"
            } else {
                "Transformer"
            }
        });

    if config.verbose {
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            arch, model.config.num_layers, model.config.vocab_size
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
    }

    // Get input tokens
    let input_tokens = if let Some(ref tokens) = config.input_tokens {
        tokens.clone()
    } else if let Some(ref prompt) = config.prompt {
        // Detect instruct model and apply chat template
        let model_name = config
            .model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let is_instruct = model_name.to_lowercase().contains("instruct");

        let formatted_prompt = if is_instruct {
            let messages = vec![ChatMessage::user(prompt)];
            format_messages(&messages, Some(model_name)).unwrap_or_else(|_| prompt.clone())
        } else {
            prompt.clone()
        };

        mapped
            .model
            .encode(&formatted_prompt)
            .unwrap_or_else(|| vec![1u32])
    } else {
        vec![1u32] // BOS token
    };

    let input_token_count = input_tokens.len();

    // Configure generation
    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens.min(128),
        temperature: config.temperature,
        top_k: config.top_k,
        ..Default::default()
    };

    // Run inference (GPU or CPU)
    let infer_start = Instant::now();
    let (tokens, used_gpu) = run_gguf_generate(model, &input_tokens, &gen_config, config)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    // Decode output
    let generated_tokens = &tokens[input_token_count..];
    let text = mapped.model.decode(generated_tokens);

    // Clean output (strip ChatML markers)
    let text = clean_model_output(&text);

    let generated_token_count = generated_tokens.len();
    let tok_per_sec = if inference_ms > 0.0 {
        generated_token_count as f64 / (inference_ms / 1000.0)
    } else {
        0.0
    };

    Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec,
        load_ms,
        format: "GGUF".to_string(),
        used_gpu,
    })
}

/// Run GGUF generation with GPU or CPU
#[allow(unused_variables)] // config used only in CUDA feature
fn run_gguf_generate(
    model: crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    config: &InferenceConfig,
) -> Result<(Vec<u32>, bool)> {
    #[cfg(feature = "cuda")]
    if !config.no_gpu {
        use crate::gguf::OwnedQuantizedModelCuda;

        match OwnedQuantizedModelCuda::new(model.clone(), 0) {
            Ok(mut cuda_model) => {
                if config.verbose {
                    eprintln!(
                        "Backend: GPU ({}, {} MB VRAM)",
                        cuda_model.device_name(),
                        cuda_model.vram_mb()
                    );
                }
                let tokens = cuda_model
                    .generate_gpu_resident(input_tokens, gen_config)
                    .map_err(|e| {
                        RealizarError::InferenceError(format!("GPU generation failed: {}", e))
                    })?;
                return Ok((tokens, true));
            },
            Err(e) => {
                if config.verbose {
                    eprintln!("Backend: CPU (GPU unavailable: {})", e);
                }
            },
        }
    }

    // CPU fallback
    if config.verbose {
        eprintln!("Backend: CPU (SIMD-accelerated)");
    }
    let tokens = model
        .generate_with_cache(input_tokens, gen_config)
        .map_err(|e| RealizarError::InferenceError(format!("CPU generation failed: {}", e)))?;
    Ok((tokens, false))
}

/// Run APR model inference (PAR-302)
///
/// Uses AprTransformer with proper RoPE and SwiGLU for correct inference.
fn run_apr_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    use crate::apr::AprV2Model;
    use crate::apr_transformer::AprTransformer;

    // Verbose: Show loading message BEFORE loading
    if config.verbose {
        eprintln!("Loading APR model: {}", config.model_path.display());
    }

    let load_start = Instant::now();

    // Load APR into AprTransformer for proper inference with RoPE and SwiGLU
    let transformer = AprTransformer::from_apr_file(&config.model_path)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let arch = &transformer.config.architecture;

    if config.verbose {
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            arch, transformer.config.num_layers, transformer.config.vocab_size
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
    }

    // Get input tokens (use sibling tokenizer.json)
    let input_tokens = if let Some(ref tokens) = config.input_tokens {
        tokens.clone()
    } else if let Some(ref prompt) = config.prompt {
        // Load tokenizer from sibling tokenizer.json
        AprV2Model::encode_text(&config.model_path, prompt).unwrap_or_else(|| vec![1u32])
    } else {
        vec![1u32] // BOS token
    };

    let input_token_count = input_tokens.len();

    // PMAT-103 FIX: Use KV-cache for O(n) generation instead of O(n²)
    let infer_start = Instant::now();
    let mut all_tokens = input_tokens.clone();

    // Create KV cache
    let mut cache = crate::apr_transformer::AprKVCache::new(&transformer.config);

    // Prefill: process each input token to populate KV cache
    for (pos, &token) in input_tokens.iter().enumerate() {
        let _ = transformer.forward_with_cache(token, &mut cache, pos)?;
    }

    // Generate new tokens with KV cache (O(1) per token)
    let mut position = input_tokens.len();
    for _ in 0..config.max_tokens.min(128) {
        // Forward pass with KV cache
        let last_token = *all_tokens.last().unwrap_or(&1);
        let logits = transformer.forward_with_cache(last_token, &mut cache, position)?;

        // Greedy sampling (argmax)
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);

        // Check for EOS (Qwen2 EOS=151645, BOS=151643, standard=2)
        if next_token == 151645 || next_token == 151643 || next_token == 2 {
            break;
        }

        all_tokens.push(next_token);
        position += 1;
    }

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    // Decode output tokens
    // GH-156: Try multiple tokenizer sources for APR models
    let generated_tokens = &all_tokens[input_token_count..];
    let text = if let Some(tokenizer) = AprV2Model::load_tokenizer(&config.model_path) {
        tokenizer.decode(generated_tokens)
    } else if let Some(tokenizer) = find_fallback_tokenizer(&config.model_path) {
        tokenizer.decode(generated_tokens)
    } else {
        format!(
            "[{} tokens generated, tokenizer not found]",
            generated_tokens.len()
        )
    };

    // Clean output
    let text = clean_model_output(&text);

    let generated_token_count = generated_tokens.len();
    let tok_per_sec = if inference_ms > 0.0 {
        generated_token_count as f64 / (inference_ms / 1000.0)
    } else {
        0.0
    };

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec,
        load_ms,
        format: "APR".to_string(),
        used_gpu: false,
    })
}

/// Run SafeTensors model inference (PAR-301)
fn run_safetensors_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    use crate::apr::AprV2Model;
    use crate::apr_transformer::AprKVCache;
    use crate::safetensors_infer::SafetensorsToAprConverter;

    // Verbose: Show loading message BEFORE loading
    if config.verbose {
        eprintln!("Loading SafeTensors model: {}", config.model_path.display());
    }

    let load_start = Instant::now();

    // Convert SafeTensors to AprTransformer
    let transformer = SafetensorsToAprConverter::convert(&config.model_path)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let arch = &transformer.config.architecture;

    if config.verbose {
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            arch, transformer.config.num_layers, transformer.config.vocab_size
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
    }

    // Get input tokens (use sibling tokenizer.json)
    let input_tokens = if let Some(ref tokens) = config.input_tokens {
        tokens.clone()
    } else if let Some(ref prompt) = config.prompt {
        // Load tokenizer from sibling tokenizer.json
        AprV2Model::encode_text(&config.model_path, prompt).unwrap_or_else(|| vec![1u32])
    } else {
        vec![1u32] // BOS token
    };

    let input_token_count = input_tokens.len();

    // PMAT-103: Use KV-cached generation for O(n) instead of O(n²) complexity
    // Previous code used forward() in a loop which recomputed all tokens each time
    let infer_start = Instant::now();
    let mut cache = AprKVCache::new(&transformer.config);
    let mut all_tokens = input_tokens.clone();

    // Prefill phase: process all prompt tokens, get logits from last token
    let mut logits = Vec::new();
    for (pos, &token) in input_tokens.iter().enumerate() {
        logits = transformer.forward_with_cache(token, &mut cache, pos)?;
    }

    // Decode phase: sample and generate new tokens
    let max_gen = config.max_tokens.min(128);
    for _ in 0..max_gen {
        // Greedy sampling (argmax) from current logits
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);

        // Check for EOS (Qwen2 EOS=151645, BOS=151643, standard=2)
        if next_token == 151645 || next_token == 151643 || next_token == 2 {
            break;
        }

        all_tokens.push(next_token);

        // Process newly generated token to get next logits
        let pos = all_tokens.len() - 1; // Position of the just-added token
        logits = transformer.forward_with_cache(next_token, &mut cache, pos)?;
    }

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    // Decode output tokens
    let generated_tokens = &all_tokens[input_token_count..];
    let text = if let Some(tokenizer) = AprV2Model::load_tokenizer(&config.model_path) {
        tokenizer.decode(generated_tokens)
    } else {
        format!(
            "[{} tokens generated, tokenizer not found]",
            generated_tokens.len()
        )
    };

    // Clean output
    let text = clean_model_output(&text);

    let generated_token_count = generated_tokens.len();
    let tok_per_sec = if inference_ms > 0.0 {
        generated_token_count as f64 / (inference_ms / 1000.0)
    } else {
        0.0
    };

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec,
        load_ms,
        format: "SafeTensors".to_string(),
        used_gpu: false, // SafeTensors currently CPU-only
    })
}

/// Pre-fault mmap pages to avoid page faults during inference
fn prefault_mmap(data: &[u8]) {
    let page_size = 4096;
    let mut checksum: u8 = 0;
    for i in (0..data.len()).step_by(page_size) {
        checksum = checksum.wrapping_add(data[i]);
    }
    std::hint::black_box(checksum);
}

/// Find a fallback tokenizer for APR models (GH-156)
///
/// This function tries to load the embedded tokenizer from the APR model.
/// APR files can contain the vocabulary in metadata, so we don't need
/// a sibling tokenizer.json file.
///
/// # Arguments
/// * `model_path` - Path to the APR model file
///
/// # Returns
/// * `Some(BpeTokenizer)` - If embedded tokenizer found and converted
/// * `None` - If no embedded tokenizer available
fn find_fallback_tokenizer(model_path: &std::path::Path) -> Option<crate::apr::BpeTokenizer> {
    use crate::apr::AprV2Model;

    // Try to load the APR model and extract embedded tokenizer
    let model = AprV2Model::load(model_path).ok()?;
    let simple_tokenizer = model.load_embedded_tokenizer()?;

    // Convert SimpleTokenizer to BpeTokenizer for compatibility
    // SimpleTokenizer is decode-only, but BpeTokenizer has encode support
    // For fallback purposes, we only need decode, so this is fine
    Some(crate::apr::BpeTokenizer {
        token_to_id: simple_tokenizer
            .id_to_token
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect(),
        id_to_token: simple_tokenizer.id_to_token,
        merge_rules: Vec::new(), // No merge rules for embedded tokenizer
        bos_id: simple_tokenizer.bos_token_id,
        eos_id: simple_tokenizer.eos_token_id,
    })
}

/// Clean model output by stripping ChatML markers
fn clean_model_output(raw: &str) -> String {
    let mut cleaned = raw.to_string();
    let markers = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ];
    for marker in markers {
        cleaned = cleaned.replace(marker, "");
    }
    cleaned.trim().to_string()
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod infer_tests;
