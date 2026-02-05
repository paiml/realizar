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

/// PMAT-173: Convert GGML quantization type to human-readable string
/// Used for --verbose mode display (F-UX-038)
pub(crate) fn qtype_to_dtype_str(qtype: u32) -> &'static str {
    match qtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        30 => "BF16",
        _ => "Unknown",
    }
}

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
    /// INTERNAL: Use mock backend for testing (PMAT-COV-95)
    #[doc(hidden)]
    pub use_mock_backend: bool,
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
            use_mock_backend: false,
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

    /// Set trace output file path
    #[must_use]
    pub fn with_trace_output(mut self, path: impl Into<PathBuf>) -> Self {
        self.trace_output = Some(path.into());
        self
    }
}

// ============================================================================
// PreparedTokens - Compile-time chat template enforcement (PMAT-236)
// ============================================================================

/// Tokenized input that has been processed through chat template formatting.
///
/// # Compile-time enforcement (Poka-Yoke)
///
/// The inner `Vec<u32>` is **private** - the only way to construct `PreparedTokens`
/// is via `prepare_tokens()`, which ALWAYS applies chat template formatting for
/// instruct models. This makes it a **compile error** to pass raw tokens to
/// inference functions, preventing the bug where SafeTensors inference skipped
/// chat template application (producing "4" then garbage).
///
/// # Theoretical basis
///
/// Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*.
/// Brady, E. (2017). *Type-Driven Development with Idris*.
///
/// # References
///
/// - PMAT-236: Chat template enforcement for multi-format inference
/// - GH-205: SafeTensors inference garbage root cause
#[derive(Debug, Clone)]
pub struct PreparedTokens {
    /// Tokenized input (PRIVATE - enforces construction via prepare_tokens only)
    tokens: Vec<u32>,
    /// Number of input tokens (for separating prefill from generated tokens)
    input_count: usize,
}

impl PreparedTokens {
    /// Access the prepared token IDs (read-only).
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of input tokens.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.input_count
    }
}

/// Prepare tokens for inference, applying chat template for instruct models.
///
/// This is the ONLY way to create `PreparedTokens`. It handles:
/// 1. Format detection (GGUF vs SafeTensors vs APR)
/// 2. Architecture detection (Qwen2, LLaMA, Phi, etc.)
/// 3. Chat template application for instruct models
/// 4. Tokenization using the appropriate tokenizer
///
/// # Chat Template Rules
///
/// - If model name/architecture contains "instruct", chat template is applied
/// - GGUF: uses embedded tokenizer + architecture from metadata
/// - SafeTensors: uses sibling tokenizer.json + config.json architecture
/// - APR: uses sibling tokenizer.json + model metadata
///
/// # Errors
///
/// Returns error if the model cannot be read or tokenization fails.
pub fn prepare_tokens(config: &InferenceConfig, format: &ModelFormat) -> Result<PreparedTokens> {
    // If raw token IDs are provided, use them directly (user knows what they're doing)
    if let Some(ref tokens) = config.input_tokens {
        return Ok(PreparedTokens {
            input_count: tokens.len(),
            tokens: tokens.clone(),
        });
    }

    let prompt = match config.prompt {
        Some(ref p) => p.clone(),
        None => {
            return Ok(PreparedTokens {
                tokens: vec![1u32],
                input_count: 1,
            })
        }
    };

    match format {
        ModelFormat::Gguf => prepare_tokens_gguf(config, &prompt),
        ModelFormat::SafeTensors => prepare_tokens_safetensors(config, &prompt),
        ModelFormat::Apr => prepare_tokens_apr(config, &prompt),
    }
}

/// Prepare tokens for GGUF format (chat template from GGUF metadata)
fn prepare_tokens_gguf(config: &InferenceConfig, prompt: &str) -> Result<PreparedTokens> {
    use crate::chat_template::{format_messages, ChatMessage};
    use crate::gguf::MappedGGUFModel;

    let mapped = MappedGGUFModel::from_path(&config.model_path)?;
    let gguf_arch = mapped.model.architecture().unwrap_or("transformer");

    let is_instruct_arch = matches!(
        gguf_arch.to_lowercase().as_str(),
        "qwen2" | "qwen" | "llama" | "mistral" | "phi" | "phi3"
    );

    let model_name = config
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let filename_instruct = model_name.to_lowercase().contains("instruct");

    let formatted_prompt = if is_instruct_arch || filename_instruct {
        let template_hint = apr_arch_to_template_hint(gguf_arch, model_name);
        let messages = vec![ChatMessage::user(prompt)];
        format_messages(&messages, Some(template_hint)).unwrap_or_else(|_| prompt.to_string())
    } else {
        prompt.to_string()
    };

    let tokens = mapped
        .model
        .encode(&formatted_prompt)
        .unwrap_or_else(|| vec![1u32]);

    Ok(PreparedTokens {
        input_count: tokens.len(),
        tokens,
    })
}

/// Prepare tokens for SafeTensors format (chat template from config.json)
fn prepare_tokens_safetensors(config: &InferenceConfig, prompt: &str) -> Result<PreparedTokens> {
    use crate::apr::AprV2Model;
    use crate::chat_template::{format_messages, ChatMessage};
    use crate::safetensors::SafetensorsConfig;

    // Load config.json for architecture detection
    let st_config = SafetensorsConfig::load_from_sibling(&config.model_path);
    let architecture = st_config
        .as_ref()
        .map(SafetensorsConfig::architecture)
        .unwrap_or_default();

    let model_name = config
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // Detect instruct model from architecture or filename
    let arch_lower = architecture.to_lowercase();
    let is_instruct = arch_lower.contains("instruct")
        || model_name.to_lowercase().contains("instruct")
        || matches!(
            arch_lower.as_str(),
            "qwen2forcausallm" | "llamaforcausallm" | "mistralforcausallm" | "phiforcausallm"
        );

    let formatted_prompt = if is_instruct {
        let template_hint = safetensors_arch_to_template_hint(&architecture, model_name);
        let messages = vec![ChatMessage::user(prompt)];
        format_messages(&messages, Some(template_hint)).unwrap_or_else(|_| prompt.to_string())
    } else {
        prompt.to_string()
    };

    let tokens =
        AprV2Model::encode_text(&config.model_path, &formatted_prompt).unwrap_or_else(|| vec![1u32]);

    Ok(PreparedTokens {
        input_count: tokens.len(),
        tokens,
    })
}

/// Prepare tokens for APR format (chat template from model metadata)
fn prepare_tokens_apr(config: &InferenceConfig, prompt: &str) -> Result<PreparedTokens> {
    use crate::apr::AprV2Model;
    use crate::chat_template::{format_messages, ChatMessage};

    let model_name = config
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // PMAT-237: Detect instruct from MODEL DATA, not filename.
    // Filename heuristic silently skips chat template for hash-named APR files.
    // Three-tier detection: architecture metadata > vocab special tokens > filename fallback.
    let (apr_arch, has_chatml_tokens) = if config.model_path.extension().is_some_and(|e| e == "apr") {
        match AprV2Model::load(&config.model_path) {
            Ok(model) => {
                let arch = model.metadata().architecture.clone().unwrap_or_default();
                let has_chatml = model.metadata().get_embedded_vocabulary()
                    .is_some_and(|vocab: Vec<String>| vocab.iter().any(|t| t == "<|im_start|>"));
                (arch, has_chatml)
            }
            Err(_) => (String::new(), false),
        }
    } else {
        (String::new(), false)
    };

    let is_instruct_arch = matches!(
        apr_arch.to_lowercase().as_str(),
        "qwen2" | "qwen" | "llama" | "mistral" | "phi" | "phi3"
    );
    let filename_instruct = model_name.to_lowercase().contains("instruct");

    let is_instruct = is_instruct_arch || has_chatml_tokens || filename_instruct;

    let formatted_prompt = if is_instruct {
        let template_hint = apr_arch_to_template_hint(&apr_arch, model_name);
        let messages = vec![ChatMessage::user(prompt)];
        format_messages(&messages, Some(template_hint)).unwrap_or_else(|_| prompt.to_string())
    } else {
        prompt.to_string()
    };

    let tokens =
        AprV2Model::encode_text(&config.model_path, &formatted_prompt).unwrap_or_else(|| vec![1u32]);

    Ok(PreparedTokens {
        input_count: tokens.len(),
        tokens,
    })
}

/// Map SafeTensors architecture string to chat template hint
fn safetensors_arch_to_template_hint<'a>(architecture: &str, model_name: &'a str) -> &'a str {
    let arch_lower = architecture.to_lowercase();
    if arch_lower.contains("qwen") {
        "qwen2"
    } else if arch_lower.contains("llama") {
        "llama"
    } else if arch_lower.contains("mistral") {
        "mistral"
    } else if arch_lower.contains("phi") {
        "phi"
    } else {
        // Fall back to model name heuristic
        apr_arch_to_template_hint("unknown", model_name)
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

// ============================================================================
// Security - Path Validation (F-SEC-222)
// ============================================================================

/// Valid model file extensions
const VALID_MODEL_EXTENSIONS: &[&str] = &["gguf", "safetensors", "apr", "bin"];

/// Validate that a path is a valid model file path.
///
/// # Security (F-SEC-222)
///
/// This prevents path traversal attacks where an attacker could trick the
/// tool into reading arbitrary files (e.g., `/etc/passwd`, `~/.ssh/id_rsa`).
///
/// ## Validation Rules
///
/// 1. Path must have a valid model extension (.gguf, .safetensors, .apr, .bin)
/// 2. Path must not contain path traversal sequences (`../`)
/// 3. Path must be a regular file (not a directory, symlink to directory, etc.)
///
/// # Errors
///
/// Returns error if:
/// - Path has invalid or missing extension
/// - Path contains traversal sequences
/// - Path doesn't exist or isn't a file
pub(crate) fn validate_model_path(path: &std::path::Path) -> Result<()> {
    // Check for path traversal sequences
    let path_str = path.to_string_lossy();
    if path_str.contains("..") {
        return Err(RealizarError::SecurityError {
            reason: format!(
                "Path traversal detected: '{}'. Use absolute paths or paths without '..'",
                path_str
            ),
        });
    }

    // Check file extension
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();

    if !VALID_MODEL_EXTENSIONS.contains(&extension.as_str()) {
        return Err(RealizarError::SecurityError {
            reason: format!(
                "Invalid model file extension: '.{}'. Expected one of: {}",
                extension,
                VALID_MODEL_EXTENSIONS.join(", ")
            ),
        });
    }

    // Check that path exists and is a file
    if !path.exists() {
        return Err(RealizarError::IoError {
            message: format!("File not found: {}", path.display()),
        });
    }

    if !path.is_file() {
        return Err(RealizarError::SecurityError {
            reason: format!("Path is not a regular file: {}", path.display()),
        });
    }

    Ok(())
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
    // PMAT-COV-95: Mock backend for testing without disk I/O
    if config.use_mock_backend {
        return run_mock_inference(config);
    }

    // Validate path to prevent traversal attacks (F-SEC-222)
    validate_model_path(&config.model_path)?;

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

    // PMAT-236: Prepare tokens with chat template BEFORE format dispatch.
    // This is compile-time enforced - format-specific functions accept
    // PreparedTokens (private inner data) which can ONLY be created here.
    let prepared = prepare_tokens(config, &format)?;

    match format {
        ModelFormat::Gguf => run_gguf_inference(config, &prepared),
        ModelFormat::Apr => run_apr_inference(config, &prepared),
        ModelFormat::SafeTensors => run_safetensors_inference(config, &prepared),
    }
}

/// Run GGUF model inference
///
/// PMAT-236: Accepts `PreparedTokens` (compile-time enforced chat template).
fn run_gguf_inference(config: &InferenceConfig, prepared: &PreparedTokens) -> Result<InferenceResult> {
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    if config.verbose {
        eprintln!("Loading model: {}", config.model_path.display());
    }

    let load_start = Instant::now();
    let mapped = MappedGGUFModel::from_path(&config.model_path)?;
    prefault_mmap(mapped.data());
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // PMAT-109: Architecture from GGUF metadata (not filename)
    let gguf_arch = mapped.model.architecture().unwrap_or("transformer");

    if config.verbose {
        print_gguf_verbose_info(gguf_arch, &model, load_ms);
    }

    // PMAT-236: Use PreparedTokens (chat template already applied by prepare_tokens)
    let input_tokens = prepared.tokens().to_vec();
    let input_token_count = prepared.input_count();
    let model_config = model.config.clone();

    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens.min(128),
        temperature: config.temperature,
        top_k: config.top_k,
        trace: config.trace,
        ..Default::default()
    };

    let infer_start = Instant::now();
    let (tokens, used_gpu) = run_gguf_generate(model, &input_tokens, &gen_config, config)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let generated_tokens = &tokens[input_token_count..];
    let text = clean_model_output(&mapped.model.decode(generated_tokens));
    let generated_token_count = generated_tokens.len();
    let tps = tok_per_sec(generated_token_count, inference_ms);

    write_gguf_trace(
        config,
        &model_config,
        input_token_count,
        generated_token_count,
        load_ms,
        inference_ms,
        tps,
        used_gpu,
    );

    Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tps,
        load_ms,
        format: "GGUF".to_string(),
        used_gpu,
    })
}

/// Print verbose model info for GGUF inference
fn print_gguf_verbose_info(
    gguf_arch: &str,
    model: &crate::gguf::OwnedQuantizedModel,
    load_ms: f64,
) {
    let arch = match gguf_arch.to_lowercase().as_str() {
        "qwen2" | "qwen" => "Qwen2",
        "llama" => "LLaMA",
        "mistral" => "Mistral",
        "phi" | "phi3" => "Phi",
        _ => "Transformer",
    };
    let quant_type = qtype_to_dtype_str(model.lm_head_weight.qtype);
    let thread_count = rayon::current_num_threads();
    eprintln!(
        "Architecture: {} [GGUF: {}] ({} layers, vocab_size={})",
        arch, gguf_arch, model.config.num_layers, model.config.vocab_size
    );
    eprintln!(
        "Config: hidden_size={}, context_length={}, quant={}, threads={}",
        model.config.hidden_dim, model.config.context_length, quant_type, thread_count
    );
    eprintln!("Model loaded in {:.1}ms", load_ms);
}

/// Write GGUF trace output if requested (PMAT-SHOWCASE-METHODOLOGY-001)
fn write_gguf_trace(
    config: &InferenceConfig,
    model_config: &crate::gguf::GGUFConfig,
    input_token_count: usize,
    generated_token_count: usize,
    load_ms: f64,
    inference_ms: f64,
    tps: f64,
    used_gpu: bool,
) {
    let trace_path = match config.trace_output {
        Some(ref p) => p,
        None => return,
    };
    let trace_json = format!(
        r#"{{
  "version": "1.0",
  "timestamp": "{}",
  "model": {{
    "path": "{}",
    "format": "GGUF",
    "num_layers": {},
    "hidden_dim": {},
    "vocab_size": {},
    "num_heads": {}
  }},
  "inference": {{
    "input_tokens": {},
    "generated_tokens": {},
    "load_ms": {:.2},
    "inference_ms": {:.2},
    "tok_per_sec": {:.2},
    "used_gpu": {}
  }},
  "events": []
}}
"#,
        chrono::Utc::now().to_rfc3339(),
        config.model_path.display(),
        model_config.num_layers,
        model_config.hidden_dim,
        model_config.vocab_size,
        model_config.num_heads,
        input_token_count,
        generated_token_count,
        load_ms,
        inference_ms,
        tps,
        used_gpu
    );
    if let Err(e) = std::fs::write(trace_path, trace_json) {
        eprintln!(
            "Warning: Failed to write trace output to {}: {}",
            trace_path.display(),
            e
        );
    }
}

/// Check if a quantization type is legacy (Q4_0, Q4_1, Q5_0, Q5_1)
/// GPU only supports Q4_K/Q5_K/Q6_K; legacy types produce garbage on GPU.
#[inline]
fn is_legacy_gguf_quant(qtype: u32) -> bool {
    matches!(qtype, 2 | 3 | 6 | 7)
}

/// Check if model uses any legacy quantization types
fn model_has_legacy_quant(model: &crate::gguf::OwnedQuantizedModel) -> bool {
    is_legacy_gguf_quant(model.lm_head_weight.qtype)
        || model.layers.iter().any(|l| {
            is_legacy_gguf_quant(l.ffn_down_weight.qtype)
                || is_legacy_gguf_quant(l.ffn_up_weight.qtype)
                || is_legacy_gguf_quant(l.attn_output_weight.qtype)
        })
}

/// Log CPU backend selection reason
#[inline]
fn log_cpu_backend(verbose: bool, is_legacy: bool) {
    if !verbose {
        return;
    }
    if is_legacy {
        eprintln!("Backend: CPU (Q4_0 format - GPU Q4_K kernels incompatible)");
    } else {
        eprintln!("Backend: CPU (SIMD-accelerated)");
    }
}

/// F2-FIX: Validate GPU output by comparing first predicted token with CPU.
///
/// Runs a full CPU forward pass on the prompt, gets the argmax token,
/// then compares with GPU's first generated token. Returns `true` if they
/// agree (GPU is producing correct results for this model's dimensions).
///
/// This catches Q6K kernel bugs for certain hidden_dim values (e.g., 7B/3584)
/// where the GPU produces incorrect logits.
#[cfg(feature = "cuda")]
fn validate_gpu_first_token(
    model: &crate::gguf::OwnedQuantizedModel,
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
) -> bool {
    use crate::gguf::OwnedQuantizedKVCache;

    let kv_dim =
        model.config.num_kv_heads * (model.config.hidden_dim / model.config.num_heads);
    let mut cpu_cache = OwnedQuantizedKVCache::new(
        model.config.num_layers,
        kv_dim,
        input_tokens.len() + 1,
    );

    // Prefill all but last token
    for (pos, &tok) in input_tokens.iter().enumerate() {
        if pos < input_tokens.len() - 1 {
            let _ = model.forward_single_with_cache(tok, &mut cpu_cache, pos);
        }
    }

    // Get CPU prediction for last prompt token
    let cpu_logits = match model.forward_single_with_cache(
        input_tokens[input_tokens.len() - 1],
        &mut cpu_cache,
        input_tokens.len() - 1,
    ) {
        Ok(logits) => logits,
        Err(_) => return true, // CPU forward failed — can't validate, assume GPU is fine
    };

    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32);

    // Get GPU prediction for same token via generate (first token only)
    let gpu_first_config = crate::gguf::QuantizedGenerateConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        ..gen_config.clone()
    };
    match cuda_model.generate_gpu_resident(input_tokens, &gpu_first_config) {
        Ok(gpu_tokens) if gpu_tokens.len() > input_tokens.len() => {
            let gpu_first = gpu_tokens[input_tokens.len()];
            if gpu_first == cpu_argmax {
                true
            } else {
                eprintln!(
                    "[F2-VALIDATION] GPU first token {} != CPU first token {} — falling back to CPU",
                    gpu_first, cpu_argmax
                );
                false
            }
        }
        Ok(_) => true, // GPU produced EOS immediately, might be valid
        Err(_) => false,
    }
}

/// Try GPU generation with validation. Returns `Some((tokens, true))` on success,
/// `None` if GPU is unavailable or produces incorrect output.
#[cfg(feature = "cuda")]
fn try_gguf_gpu_generate(
    model: &crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    verbose: bool,
) -> Option<Result<(Vec<u32>, bool)>> {
    use crate::gguf::OwnedQuantizedModelCuda;

    let mut cuda_model = match OwnedQuantizedModelCuda::new(model.clone(), 0) {
        Ok(m) => m,
        Err(e) => {
            if verbose {
                eprintln!("Backend: CPU (GPU unavailable: {})", e);
            }
            return None;
        }
    };

    if verbose {
        eprintln!(
            "Backend: GPU ({}, {} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        );
    }

    if !validate_gpu_first_token(model, &mut cuda_model, input_tokens, gen_config) {
        return None;
    }

    // Re-create CUDA model for actual generation (validation consumed first)
    let mut cuda_model2 = match OwnedQuantizedModelCuda::new(model.clone(), 0) {
        Ok(m) => m,
        Err(_) => return None,
    };

    let result = cuda_model2
        .generate_gpu_resident(input_tokens, gen_config)
        .map(|tokens| (tokens, true))
        .map_err(|e| RealizarError::InferenceError(format!("GPU generation failed: {}", e)));
    Some(result)
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

    #[cfg(feature = "cuda")]
    if !config.no_gpu && !has_legacy_quant {
        if let Some(result) =
            try_gguf_gpu_generate(&model, input_tokens, gen_config, config.verbose)
        {
            return result;
        }
    }

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
fn run_apr_inference(config: &InferenceConfig, prepared: &PreparedTokens) -> Result<InferenceResult> {
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

/// Map APR architecture string to chat template hint
fn apr_arch_to_template_hint<'a>(apr_arch: &str, model_name: &'a str) -> &'a str {
    let arch_lower = apr_arch.to_lowercase();
    if arch_lower.contains("qwen") {
        "qwen2"
    } else if arch_lower.contains("llama") {
        "llama"
    } else if arch_lower.contains("mistral") {
        "mistral"
    } else if arch_lower.contains("phi") {
        "phi"
    } else {
        model_name
    }
}

/// Try APR CUDA inference, returning None to fall through to CPU
#[cfg(feature = "cuda")]
fn try_apr_cuda_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Option<Result<InferenceResult>> {
    use crate::apr_transformer::AprTransformer;
    use crate::gpu::adapters::AprF32ToGpuAdapter;
    use crate::gpu::GpuGenerateConfig;

    // PMAT-APR-PERF-001: Use GpuModel with incremental KV cache for O(n) decode
    // Previous path used AprV2ModelCuda which did full forward pass per token (O(n²))
    let validated = AprTransformer::from_apr_file_validated(&config.model_path).ok()?;

    // F1-FIX: Skip GPU path for F32 APR models. The GpuModel forward pass produces
    // incorrect results for F32 weights (garbage output). F32 models should use the
    // AprTransformer CPU path which has been validated via forward_traced().
    // Only use GPU for quantized APR models (Q4K/Q6K) where GPU kernels are correct.
    let has_quantized = validated.q4k_layers.is_some()
        || validated.lm_head_weight_q4k.is_some()
        || validated.lm_head_weight_q6k.is_some();
    if !has_quantized {
        return None;
    }

    let mut gpu_model = match AprF32ToGpuAdapter::to_gpu_model(&validated) {
        Ok(m) => m,
        Err(e) => {
            if config.verbose {
                eprintln!("Backend: CPU (GPU adapter failed: {})", e);
            }
            return None;
        },
    };

    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            validated.config.architecture,
            validated.config.num_layers,
            validated.config.vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, context_length={}, quant=GPU+KVCache, threads=1 (GPU)",
            validated.config.hidden_dim,
            validated.config.context_length
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!("Backend: GPU (with incremental KV cache)");
    }

    let infer_start = Instant::now();
    let eos_id = 151645u32; // Qwen2 EOS

    // Convert input tokens to usize for GpuModel API
    let prompt: Vec<usize> = input_tokens.iter().map(|&t| t as usize).collect();

    // generate_with_cache creates its own StreamingKVCache internally
    let gen_config = GpuGenerateConfig {
        max_tokens: config.max_tokens.min(128),
        temperature: 0.0, // Greedy
        top_k: 1,         // Greedy sampling
        stop_tokens: vec![eos_id as usize],
        trace: false,
    };

    let tokens = match gpu_model.generate_with_cache(&prompt, &gen_config) {
        Ok(t) => t.into_iter().map(|t| t as u32).collect::<Vec<_>>(),
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
        format: "APR".to_string(),
        used_gpu: true,
    }))
}

/// Run APR inference on CPU with KV-cache (PMAT-103)
fn run_apr_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Result<InferenceResult> {
    use crate::apr_transformer::AprTransformer;

    let validated = AprTransformer::from_apr_file_validated(&config.model_path)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
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

    let infer_start = Instant::now();
    let mut all_tokens = input_tokens.to_vec();
    let mut cache = crate::apr_transformer::AprKVCache::new(&validated.config);

    // Prefill: populate KV cache
    for (pos, &token) in input_tokens.iter().enumerate() {
        let _ = validated.forward_with_cache(token, &mut cache, pos)?;
    }

    // Generate with KV cache (O(1) per token)
    let mut position = input_tokens.len();
    for _ in 0..config.max_tokens.min(128) {
        let last_token = *all_tokens.last().unwrap_or(&1);
        let logits = validated.forward_with_cache(last_token, &mut cache, position)?;

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);

        // EOS check (Qwen2=151645, BOS=151643, standard=2)
        if next_token == 151645 || next_token == 151643 || next_token == 2 {
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
fn run_safetensors_inference(config: &InferenceConfig, prepared: &PreparedTokens) -> Result<InferenceResult> {
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
    let eos_id = 151645u32; // Qwen2 EOS
    let tokens = match cuda_model.generate(input_tokens, config.max_tokens.min(128), eos_id) {
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

/// Run SafeTensors inference on CPU via AprTransformer conversion (PMAT-103)
fn run_safetensors_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
) -> Result<InferenceResult> {
    use crate::apr::AprV2Model;
    use crate::apr_transformer::AprKVCache;
    use crate::safetensors_infer::SafetensorsToAprConverter;

    let load_start = Instant::now();
    let transformer = SafetensorsToAprConverter::convert(&config.model_path)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        let arch = &transformer.config.architecture;
        let thread_count = rayon::current_num_threads();
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            arch, transformer.config.num_layers, transformer.config.vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, context_length={}, quant=F32, threads={}",
            transformer.config.hidden_dim, transformer.config.context_length, thread_count
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!("Backend: CPU (SIMD-accelerated)");
    }

    let infer_start = Instant::now();
    let mut cache = AprKVCache::new(&transformer.config);
    let mut all_tokens = input_tokens.to_vec();

    // Prefill phase
    let mut logits = Vec::new();
    for (pos, &token) in input_tokens.iter().enumerate() {
        logits = transformer.forward_with_cache(token, &mut cache, pos)?;
    }

    // Decode phase: greedy sampling
    for _ in 0..config.max_tokens.min(128) {
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);

        if next_token == 151645 || next_token == 151643 || next_token == 2 {
            break;
        }

        all_tokens.push(next_token);
        let pos = all_tokens.len() - 1;
        logits = transformer.forward_with_cache(next_token, &mut cache, pos)?;
    }

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &all_tokens[input_token_count..];

    // DEBUG: Show generated token IDs
    eprintln!("[DEBUG] Input tokens: {:?}", &all_tokens[..input_token_count]);
    eprintln!("[DEBUG] Generated tokens: {:?}", generated_tokens);

    let text = if let Some(tokenizer) = AprV2Model::load_tokenizer(&config.model_path) {
        clean_model_output(&tokenizer.decode(generated_tokens))
    } else {
        format!(
            "[{} tokens generated, tokenizer not found]",
            generated_tokens.len()
        )
    };

    let generated_token_count = generated_tokens.len();

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "SafeTensors".to_string(),
        used_gpu: false,
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

    // F-REGR-232: Only search if the model can be loaded
    let model = AprV2Model::load(model_path).ok()?;

    // 1. Embedded BPE tokenizer (preferred — has merges)
    if let Some(bpe_tokenizer) = model.load_embedded_bpe_tokenizer() {
        return Some(bpe_tokenizer);
    }

    // 2. SimpleTokenizer converted to BPE (decode-only, no merges)
    if let Some(tok) = convert_simple_tokenizer_to_bpe(&model) {
        return Some(tok);
    }

    // 3. Search HuggingFace cache and APR tokenizer cache
    search_external_tokenizer_caches()
}

/// Convert embedded SimpleTokenizer to BpeTokenizer (GH-189)
fn convert_simple_tokenizer_to_bpe(
    model: &crate::apr::AprV2Model,
) -> Option<crate::apr::BpeTokenizer> {
    let simple_tokenizer = model.load_embedded_tokenizer()?;
    let token_to_id: std::collections::HashMap<String, u32> = simple_tokenizer
        .id_to_token
        .iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), id as u32))
        .collect();
    let special_tokens = crate::apr::extract_special_tokens_from_vocab(&token_to_id);
    Some(crate::apr::BpeTokenizer {
        token_to_id,
        id_to_token: simple_tokenizer.id_to_token,
        merge_rules: Vec::new(),
        bos_id: simple_tokenizer.bos_token_id,
        eos_id: simple_tokenizer.eos_token_id,
        special_tokens,
    })
}

/// Search HuggingFace and APR caches for Qwen tokenizer
fn search_external_tokenizer_caches() -> Option<crate::apr::BpeTokenizer> {
    use crate::apr::AprV2Model;

    let home = std::env::var("HOME").ok().map(std::path::PathBuf::from)?;

    // Search HuggingFace cache (PMAT-SHOWCASE-TOKENIZER-001)
    let hf_cache = home.join(".cache/huggingface/hub");
    if let Some(tok) = search_hf_cache_for_tokenizer(&hf_cache) {
        return Some(tok);
    }

    // Check APR tokenizer cache
    AprV2Model::load_tokenizer_from_path(&home.join(".apr/tokenizers/qwen2/tokenizer.json"))
}

/// Search HuggingFace model cache for Qwen tokenizer.json
fn search_hf_cache_for_tokenizer(hf_cache: &std::path::Path) -> Option<crate::apr::BpeTokenizer> {
    use crate::apr::AprV2Model;

    let entries = std::fs::read_dir(hf_cache).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("models--Qwen") {
            continue;
        }
        let snapshots_dir = entry.path().join("snapshots");
        let snapshots = std::fs::read_dir(&snapshots_dir).ok()?;
        for snapshot in snapshots.flatten() {
            let tokenizer_path = snapshot.path().join("tokenizer.json");
            if let Some(tok) = AprV2Model::load_tokenizer_from_path(&tokenizer_path) {
                return Some(tok);
            }
        }
    }
    None
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

// ============================================================================
// MOCK BACKEND (PMAT-COV-95: Testing without disk I/O)
// ============================================================================

/// Run mock inference for testing (PMAT-COV-95)
///
/// This function returns deterministic results without reading disk or
/// performing actual model inference. It exercises the full InferenceResult
/// construction, token counting, timing calculation, and formatting logic.
///
/// # Mock Behavior
///
/// - Input tokens: parsed from prompt or used directly from config
/// - Generated tokens: deterministic sequence [100, 101, 102, ...]
/// - Text output: "mock response for: <prompt>"
/// - Timing: simulated 10ms load, 50ms inference
/// - Format: "Mock"
pub fn run_mock_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    // Simulate model loading time
    let load_ms = 10.0;

    // Parse input tokens
    let input_tokens = if let Some(ref tokens) = config.input_tokens {
        tokens.clone()
    } else if let Some(ref prompt) = config.prompt {
        // Mock tokenization: each word becomes a token ID
        prompt
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| (i + 1) as u32)
            .collect()
    } else {
        vec![1u32] // BOS token
    };

    let input_token_count = input_tokens.len();

    // Generate deterministic output tokens
    let num_to_generate = config.max_tokens.min(32);
    let generated_tokens: Vec<u32> = (0..num_to_generate).map(|i| 100 + i as u32).collect();

    // Combine input and generated tokens
    let mut all_tokens = input_tokens;
    all_tokens.extend(&generated_tokens);

    // Mock text output
    let prompt_text = config.prompt.as_deref().unwrap_or("(no prompt)");
    let text = format!("mock response for: {}", prompt_text);

    // Simulate inference timing
    let inference_ms = 50.0 + (num_to_generate as f64 * 2.0);
    let generated_token_count = generated_tokens.len();
    let tok_per_sec = if inference_ms > 0.0 {
        generated_token_count as f64 / (inference_ms / 1000.0)
    } else {
        0.0
    };

    // Validate configuration constraints
    if config.temperature < 0.0 {
        return Err(RealizarError::InvalidConfiguration(
            "temperature cannot be negative".to_string(),
        ));
    }

    if config.max_tokens == 0 {
        return Err(RealizarError::InvalidConfiguration(
            "max_tokens must be > 0".to_string(),
        ));
    }

    // Write trace output if requested
    if let Some(ref trace_path) = config.trace_output {
        let trace_json = format!(
            r#"{{
  "version": "1.0",
  "mock": true,
  "input_tokens": {},
  "generated_tokens": {},
  "load_ms": {:.2},
  "inference_ms": {:.2}
}}
"#,
            input_token_count, generated_token_count, load_ms, inference_ms
        );
        std::fs::write(trace_path, trace_json).map_err(|e| RealizarError::IoError {
            message: format!("Failed to write trace: {}", e),
        })?;
    }

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec,
        load_ms,
        format: "Mock".to_string(),
        used_gpu: false,
    })
}

/// Create a mock inference config for testing
#[must_use]
pub fn mock_config(prompt: &str) -> InferenceConfig {
    InferenceConfig::new("/dev/null")
        .with_prompt(prompt)
        .with_max_tokens(16)
        .with_mock_backend()
}

impl InferenceConfig {
    /// Enable mock backend for testing (PMAT-COV-95)
    #[must_use]
    pub fn with_mock_backend(mut self) -> Self {
        self.use_mock_backend = true;
        self
    }
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod infer_tests;

// Additional coverage tests (tests_part_02.rs)
#[cfg(test)]
#[path = "tests_part_02.rs"]
mod infer_tests_part_02;

// Additional coverage tests (tests_part_03.rs)
#[cfg(test)]
#[path = "tests_part_03.rs"]
mod infer_tests_part_03;

// Helper functions coverage tests (tests_part_04.rs)
#[cfg(test)]
#[path = "tests_part_04.rs"]
mod infer_tests_part_04;

// T-COV-95 Coverage Bridge tests (Part 05 - B5)
#[cfg(test)]
#[path = "tests_part_05.rs"]
mod infer_tests_part_05;

// Mock backend tests (PMAT-COV-95)
#[cfg(test)]
#[path = "tests_mock.rs"]
mod infer_tests_mock;

// T-COV-95 Deep Coverage Bridge (Part 06 - validate_model_path, qtype_to_dtype_str, mock paths)
#[cfg(test)]
#[path = "tests_part_06.rs"]
mod infer_tests_part_06;

// T-COV-95 Synthetic Falsification (Part 07 - qtype all arms, InferenceConfig/Result fields)
#[cfg(test)]
#[path = "tests_part_07.rs"]
mod infer_tests_part_07;

// T-COV-95 Maimed Pygmy Campaign (Part 08 - Real inference paths with corrupted artifacts)
#[cfg(test)]
#[path = "tests_part_08.rs"]
mod infer_tests_part_08;
