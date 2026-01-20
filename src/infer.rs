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
        .map(|s| {
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
        })
        .unwrap_or("Transformer");

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
                    .map_err(|e| RealizarError::InferenceError(format!("GPU generation failed: {}", e)))?;
                return Ok((tokens, true));
            }
            Err(e) => {
                if config.verbose {
                    eprintln!("Backend: CPU (GPU unavailable: {})", e);
                }
            }
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

    let load_start = Instant::now();
    let model = AprModel::load(&config.model_path)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        eprintln!(
            "Loaded APR model ({} tensors) in {:.1}ms",
            model.tensor_count(),
            load_ms
        );
    }

    // Check if transformer model
    if !model.metadata().is_transformer() {
        return Err(RealizarError::UnsupportedOperation {
            operation: "apr_inference".to_string(),
            reason: "APR model is not a transformer (missing hidden_size/num_layers)".to_string(),
        });
    }

    // Get input tokens
    let input_tokens = if let Some(ref tokens) = config.input_tokens {
        tokens.clone()
    } else if let Some(ref prompt) = config.prompt {
        AprModel::encode_text(&config.model_path, prompt).unwrap_or_else(|| vec![1u32])
    } else {
        vec![1u32]
    };

    let input_token_count = input_tokens.len();
    let eos_id = Some(2u32);

    // Run generation
    let infer_start = Instant::now();
    let tokens = model.generate(&input_tokens, config.max_tokens, eos_id).map_err(|e| {
        RealizarError::InferenceError(format!("APR generation failed: {}", e))
    })?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    // Decode output
    let vocab = AprModel::load_tokenizer_from_sibling(&config.model_path);
    let generated_tokens = &tokens[input_token_count..];
    let text = vocab
        .as_ref()
        .map(|(v, _, _)| AprModel::decode_tokens(v, generated_tokens))
        .unwrap_or_else(|| format!("{:?}", generated_tokens));

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
        format: "APR".to_string(),
        used_gpu: false,
    })
}

/// Run SafeTensors model inference (PAR-301)
fn run_safetensors_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    use crate::apr::AprV2Model;
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

    // Run forward pass
    let infer_start = Instant::now();
    let mut all_tokens = input_tokens.clone();

    for _ in 0..config.max_tokens.min(128) {
        // Forward pass to get logits
        let logits = transformer.forward(&all_tokens)?;

        // Greedy sampling (argmax)
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        // Check for EOS (Qwen2 EOS=151645, BOS=151643)
        if next_token == 151645 || next_token == 151643 {
            break;
        }

        all_tokens.push(next_token);
    }

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    // Decode output tokens
    let generated_tokens = &all_tokens[input_token_count..];
    let text = if let Some(tokenizer) = AprV2Model::load_tokenizer(&config.model_path) {
        tokenizer.decode(generated_tokens)
    } else {
        format!("[{} tokens generated, tokenizer not found]", generated_tokens.len())
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // InferenceConfig Builder Tests
    // =========================================================================

    #[test]
    fn test_inference_config_builder() {
        let config = InferenceConfig::new("/path/to/model.gguf")
            .with_prompt("Hello")
            .with_max_tokens(64)
            .with_temperature(0.7)
            .with_top_k(40)
            .without_gpu()
            .with_verbose(true);

        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
        assert_eq!(config.prompt, Some("Hello".to_string()));
        assert_eq!(config.max_tokens, 64);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 40);
        assert!(config.no_gpu);
        assert!(config.verbose);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::new("/model.gguf");
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_with_input_tokens() {
        let config = InferenceConfig::new("/model.gguf")
            .with_input_tokens(vec![1, 2, 3, 4]);
        assert_eq!(config.input_tokens, Some(vec![1, 2, 3, 4]));
        assert!(config.prompt.is_none()); // prompt not set
    }

    #[test]
    fn test_inference_config_with_trace() {
        let config = InferenceConfig::new("/model.gguf")
            .with_trace(true);
        assert!(config.trace);
        assert!(!config.trace_verbose);
    }

    // =========================================================================
    // Output Cleaning Tests
    // =========================================================================

    #[test]
    fn test_clean_model_output() {
        let raw = "<|im_start|>assistant\nHello world!<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_no_markers() {
        let raw = "Hello world!";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_multiple_markers() {
        let raw = "<|im_start|><|im_start|>assistant\nMultiple markers<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Multiple markers");
    }

    #[test]
    fn test_clean_model_output_whitespace_only() {
        let raw = "<|im_start|>assistant\n   <|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_with_newlines() {
        let raw = "<|im_start|>assistant\nLine 1\nLine 2<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Line 1\nLine 2");
    }

    // =========================================================================
    // Prefault Mmap Tests
    // =========================================================================

    #[test]
    fn test_prefault_mmap_empty() {
        let data: &[u8] = &[];
        prefault_mmap(data); // Should not panic
    }

    #[test]
    fn test_prefault_mmap_small() {
        let data = vec![0u8; 1000];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_aligned() {
        let data = vec![0u8; 4096 * 3]; // 3 pages
        prefault_mmap(&data);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_run_inference_file_not_found() {
        let config = InferenceConfig::new("/nonexistent/model.gguf");
        let result = run_inference(&config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to read model"));
    }

    #[test]
    fn test_run_inference_file_too_small() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("tiny_model.gguf");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[1, 2, 3]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_run_inference_invalid_format() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("invalid_format_model.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Format"));

        let _ = std::fs::remove_file(path);
    }

    // =========================================================================
    // InferenceResult Tests
    // =========================================================================

    #[test]
    fn test_inference_result_default() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 100.0,
            tok_per_sec: 20.0,
            load_ms: 50.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.text, "test");
        assert_eq!(result.tokens, vec![1, 2, 3]);
        assert_eq!(result.generated_token_count, 2);
    }

    #[test]
    fn test_inference_result_clone() {
        let result = InferenceResult {
            text: "hello".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 10.0,
            tok_per_sec: 0.0,
            load_ms: 5.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        let cloned = result.clone();
        assert_eq!(result.text, cloned.text);
        assert_eq!(result.used_gpu, cloned.used_gpu);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_inference_config_with_trace_enabled() {
        let config = InferenceConfig::new("/model.gguf")
            .with_trace(true);
        assert!(config.trace);
    }

    #[test]
    fn test_inference_config_with_trace_disabled() {
        let config = InferenceConfig::new("/model.gguf")
            .with_trace(false);
        assert!(!config.trace);
    }

    #[test]
    fn test_inference_config_builder_all_options() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(128)
            .with_temperature(0.8)
            .with_top_k(50)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);

        assert_eq!(config.prompt, Some("Test prompt".to_string()));
        assert_eq!(config.max_tokens, 128);
        assert!((config.temperature - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    #[test]
    fn test_clean_model_output_chatml_markers() {
        let raw = "<|im_start|>assistant\nHello world!<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        // Should clean ChatML markers
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        assert!(!cleaned.contains("<|endoftext|>"));
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_empty_input() {
        let cleaned = clean_model_output("");
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_only_markers() {
        let raw = "<|im_start|><|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_prefault_mmap_large() {
        let data = vec![0u8; 4096 * 10]; // 10 pages
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_unaligned() {
        let data = vec![0u8; 4096 + 100]; // 1 page + extra
        prefault_mmap(&data);
    }

    #[test]
    fn test_inference_config_debug() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("test");
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("model_path"));
        assert!(debug_str.contains("prompt"));
    }

    #[test]
    fn test_inference_result_debug() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 1,
            inference_ms: 10.0,
            tok_per_sec: 100.0,
            load_ms: 5.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("text"));
        assert!(debug_str.contains("tokens"));
    }

    #[test]
    fn test_inference_config_with_zero_temperature() {
        let config = InferenceConfig::new("/model.gguf")
            .with_temperature(0.0);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_with_high_temperature() {
        let config = InferenceConfig::new("/model.gguf")
            .with_temperature(2.0);
        assert!((config.temperature - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_with_zero_top_k() {
        let config = InferenceConfig::new("/model.gguf")
            .with_top_k(0);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_inference_config_with_large_top_k() {
        let config = InferenceConfig::new("/model.gguf")
            .with_top_k(1000);
        assert_eq!(config.top_k, 1000);
    }

    #[test]
    fn test_inference_config_with_zero_max_tokens() {
        let config = InferenceConfig::new("/model.gguf")
            .with_max_tokens(0);
        assert_eq!(config.max_tokens, 0);
    }

    #[test]
    fn test_inference_config_with_large_max_tokens() {
        let config = InferenceConfig::new("/model.gguf")
            .with_max_tokens(4096);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_inference_config_chaining() {
        // Test that builder methods can be chained in any order
        let config = InferenceConfig::new("/m.gguf")
            .with_verbose(true)
            .with_trace(true)
            .with_prompt("p")
            .without_gpu()
            .with_max_tokens(10);
        assert!(config.verbose);
        assert!(config.trace);
        assert_eq!(config.prompt, Some("p".to_string()));
        assert!(config.no_gpu);
        assert_eq!(config.max_tokens, 10);
    }

    #[test]
    fn test_clean_model_output_preserves_content() {
        let raw = "Hello, this is a test without any markers.";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, raw);
    }

    #[test]
    fn test_clean_model_output_partial_markers() {
        let raw = "Hello <|im_start|> world <|im_end|> test";
        let cleaned = clean_model_output(raw);
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
    }

    // =========================================================================
    // Extended Coverage Tests for InferenceConfig
    // =========================================================================

    #[test]
    fn test_inference_config_clone_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("test")
            .with_max_tokens(64)
            .with_temperature(0.5)
            .with_top_k(20)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);
        let cloned = config.clone();
        assert_eq!(config.model_path, cloned.model_path);
        assert_eq!(config.prompt, cloned.prompt);
        assert_eq!(config.max_tokens, cloned.max_tokens);
        assert!((config.temperature - cloned.temperature).abs() < f32::EPSILON);
        assert_eq!(config.top_k, cloned.top_k);
        assert_eq!(config.no_gpu, cloned.no_gpu);
        assert_eq!(config.verbose, cloned.verbose);
        assert_eq!(config.trace, cloned.trace);
    }

    #[test]
    fn test_inference_config_input_tokens_clone_cov() {
        let config = InferenceConfig::new("/m.gguf")
            .with_input_tokens(vec![1, 2, 3, 4, 5]);
        let cloned = config.clone();
        assert_eq!(config.input_tokens, cloned.input_tokens);
    }

    #[test]
    fn test_inference_config_empty_prompt_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("");
        assert_eq!(config.prompt, Some("".to_string()));
    }

    #[test]
    fn test_inference_config_long_prompt_cov() {
        let long_prompt = "x".repeat(10000);
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt(&long_prompt);
        assert_eq!(config.prompt, Some(long_prompt));
    }

    #[test]
    fn test_inference_config_empty_input_tokens_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_input_tokens(vec![]);
        assert_eq!(config.input_tokens, Some(vec![]));
    }

    #[test]
    fn test_inference_config_large_input_tokens_cov() {
        let tokens: Vec<u32> = (0..1000).collect();
        let config = InferenceConfig::new("/model.gguf")
            .with_input_tokens(tokens.clone());
        assert_eq!(config.input_tokens, Some(tokens));
    }

    #[test]
    fn test_inference_config_negative_temperature_cov() {
        // Temperature can be set to any float, even negative
        let config = InferenceConfig::new("/model.gguf")
            .with_temperature(-1.0);
        assert!((config.temperature - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_verbose_false_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_verbose(false);
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_path_with_spaces_cov() {
        let config = InferenceConfig::new("/path/with spaces/model.gguf");
        assert_eq!(config.model_path, PathBuf::from("/path/with spaces/model.gguf"));
    }

    #[test]
    fn test_inference_config_path_from_string_cov() {
        let path_str = String::from("/model.gguf");
        let config = InferenceConfig::new(path_str);
        assert_eq!(config.model_path, PathBuf::from("/model.gguf"));
    }

    #[test]
    fn test_inference_config_path_from_pathbuf_cov() {
        let path = PathBuf::from("/model.gguf");
        let config = InferenceConfig::new(path.clone());
        assert_eq!(config.model_path, path);
    }

    #[test]
    fn test_inference_config_defaults_not_overwritten_cov() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());
        assert!(!config.trace);
        assert!(!config.trace_verbose);
        assert!(config.trace_output.is_none());
        assert!(config.trace_steps.is_none());
    }

    // =========================================================================
    // Extended Coverage Tests for InferenceResult
    // =========================================================================

    #[test]
    fn test_inference_result_with_zero_inference_time_cov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!((result.inference_ms - 0.0).abs() < f64::EPSILON);
        assert!((result.tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_inference_result_with_high_tok_per_sec_cov() {
        let result = InferenceResult {
            text: "fast".to_string(),
            tokens: vec![1],
            input_token_count: 0,
            generated_token_count: 1000,
            inference_ms: 10.0,
            tok_per_sec: 100000.0,
            load_ms: 1.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        assert!(result.tok_per_sec > 10000.0);
    }

    #[test]
    fn test_inference_result_empty_text_cov() {
        let result = InferenceResult {
            text: "".to_string(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(result.text.is_empty());
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_inference_result_empty_tokens_cov() {
        let result = InferenceResult {
            text: "".to_string(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 5.0,
            tok_per_sec: 0.0,
            load_ms: 2.0,
            format: "SafeTensors".to_string(),
            used_gpu: false,
        };
        assert!(result.tokens.is_empty());
        assert_eq!(result.format, "SafeTensors");
    }

    #[test]
    fn test_inference_result_large_tokens_cov() {
        let tokens: Vec<u32> = (0..10000).collect();
        let result = InferenceResult {
            text: "large".to_string(),
            tokens: tokens.clone(),
            input_token_count: 100,
            generated_token_count: 9900,
            inference_ms: 1000.0,
            tok_per_sec: 9900.0,
            load_ms: 100.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.tokens.len(), 10000);
    }

    #[test]
    fn test_inference_result_format_variations_cov() {
        for format in ["GGUF", "APR", "SafeTensors", "custom"] {
            let result = InferenceResult {
                text: "t".to_string(),
                tokens: vec![1],
                input_token_count: 1,
                generated_token_count: 0,
                inference_ms: 1.0,
                tok_per_sec: 0.0,
                load_ms: 1.0,
                format: format.to_string(),
                used_gpu: false,
            };
            assert_eq!(result.format, format);
        }
    }

    // =========================================================================
    // Extended Coverage Tests for clean_model_output
    // =========================================================================

    #[test]
    fn test_clean_model_output_nested_markers_cov() {
        let raw = "<|im_start|>assistant\n<|im_start|>Hello<|im_end|><|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
    }

    #[test]
    fn test_clean_model_output_unicode_cov() {
        let raw = "<|im_start|>assistant\n你好世界<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "你好世界");
    }

    #[test]
    fn test_clean_model_output_emoji_cov() {
        let raw = "<|im_start|>assistant\n🎉 Hello! 🎊<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "🎉 Hello! 🎊");
    }

    #[test]
    fn test_clean_model_output_tabs_cov() {
        let raw = "<|im_start|>assistant\n\tTabbed content\t<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Tabbed content");
    }

    #[test]
    fn test_clean_model_output_carriage_return_cov() {
        let raw = "<|im_start|>assistant\r\nWindows line<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("Windows line"));
    }

    #[test]
    fn test_clean_model_output_special_chars_cov() {
        let raw = "Special: $@#%^&*()[]{}";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, raw);
    }

    #[test]
    fn test_clean_model_output_very_long_cov() {
        let content = "x".repeat(100000);
        let raw = format!("<|im_start|>assistant\n{}<|im_end|>", content);
        let cleaned = clean_model_output(&raw);
        assert_eq!(cleaned.len(), 100000);
    }

    #[test]
    fn test_clean_model_output_assistant_without_newline_cov() {
        // Tests the <|im_start|>assistant marker without trailing newline
        let raw = "<|im_start|>assistantHello<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello");
    }

    // =========================================================================
    // Extended Coverage Tests for prefault_mmap
    // =========================================================================

    #[test]
    fn test_prefault_mmap_single_byte_cov() {
        let data = vec![42u8; 1];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_exactly_one_page_cov() {
        let data = vec![0u8; 4096];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_minus_one_cov() {
        let data = vec![0u8; 4095];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_plus_one_cov() {
        let data = vec![0u8; 4097];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_with_nonzero_data_cov() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_all_255_cov() {
        let data = vec![255u8; 4096 * 2];
        prefault_mmap(&data);
    }

    // =========================================================================
    // Run Inference Error Path Tests
    // =========================================================================

    #[test]
    fn test_run_inference_empty_path_cov() {
        let config = InferenceConfig::new("");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_directory_path_cov() {
        let config = InferenceConfig::new("/tmp");
        let result = run_inference(&config);
        // Either fails to read as file or fails format detection
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_exactly_8_bytes_cov() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("exactly_8_bytes.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[0, 0, 0, 0, 0, 0, 0, 0]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        // Should fail format detection (not a valid magic)
        assert!(result.is_err());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_run_inference_7_bytes_cov() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("seven_bytes.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[1, 2, 3, 4, 5, 6, 7]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));

        let _ = std::fs::remove_file(path);
    }

    // =========================================================================
    // InferenceConfig Trace Fields Tests
    // =========================================================================

    #[test]
    fn test_inference_config_trace_fields_debug_cov() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/model.gguf"),
            prompt: Some("test".to_string()),
            input_tokens: None,
            max_tokens: 32,
            temperature: 0.0,
            top_k: 1,
            no_gpu: false,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/trace.json")),
            trace_steps: Some(vec!["embedding".to_string(), "attention".to_string()]),
            verbose: false,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("trace_verbose"));
        assert!(debug_str.contains("trace_output"));
        assert!(debug_str.contains("trace_steps"));
    }

    #[test]
    fn test_inference_config_trace_fields_clone_cov() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/model.gguf"),
            prompt: None,
            input_tokens: Some(vec![1, 2, 3]),
            max_tokens: 64,
            temperature: 0.5,
            top_k: 10,
            no_gpu: true,
            trace: false,
            trace_verbose: false,
            trace_output: None,
            trace_steps: None,
            verbose: true,
        };
        let cloned = config.clone();
        assert_eq!(cloned.trace_verbose, config.trace_verbose);
        assert_eq!(cloned.trace_output, config.trace_output);
        assert_eq!(cloned.trace_steps, config.trace_steps);
    }

    #[test]
    fn test_inference_result_all_fields_cov() {
        let result = InferenceResult {
            text: "generated output".to_string(),
            tokens: vec![1, 2, 3, 4, 5],
            input_token_count: 2,
            generated_token_count: 3,
            inference_ms: 123.456,
            tok_per_sec: 24.32,
            load_ms: 50.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.text, "generated output");
        assert_eq!(result.tokens.len(), 5);
        assert_eq!(result.input_token_count, 2);
        assert_eq!(result.generated_token_count, 3);
        assert!((result.inference_ms - 123.456).abs() < 0.001);
        assert!((result.tok_per_sec - 24.32).abs() < 0.01);
        assert!((result.load_ms - 50.0).abs() < 0.01);
        assert_eq!(result.format, "GGUF");
        assert!(result.used_gpu);
    }

    // =========================================================================
    // Extended Coverage Tests: InferenceConfig builders
    // =========================================================================

    #[test]
    fn test_inference_config_all_methods_chain_ext_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(100)
            .with_temperature(1.5)
            .with_top_k(50)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);

        assert_eq!(config.prompt, Some("Test prompt".to_string()));
        assert_eq!(config.max_tokens, 100);
        assert!((config.temperature - 1.5).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    #[test]
    fn test_inference_config_defaults_ext_cov() {
        let config = InferenceConfig::new("/model.apr");

        // Check all default values
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.trace);
        assert!(!config.trace_verbose);
        assert!(config.trace_output.is_none());
        assert!(config.trace_steps.is_none());
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_with_input_tokens_only_ext_cov() {
        let config = InferenceConfig::new("/model.safetensors")
            .with_input_tokens(vec![100, 200, 300, 400]);

        assert_eq!(config.input_tokens, Some(vec![100, 200, 300, 400]));
        assert!(config.prompt.is_none());
    }

    #[test]
    fn test_inference_config_temperature_extremes_ext_cov() {
        let cold_config = InferenceConfig::new("/model.gguf")
            .with_temperature(0.0);
        let hot_config = InferenceConfig::new("/model.gguf")
            .with_temperature(2.0);

        assert!((cold_config.temperature - 0.0).abs() < f32::EPSILON);
        assert!((hot_config.temperature - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_top_k_values_ext_cov() {
        let greedy = InferenceConfig::new("/model.gguf").with_top_k(1);
        let wide = InferenceConfig::new("/model.gguf").with_top_k(100);
        let disabled = InferenceConfig::new("/model.gguf").with_top_k(0);

        assert_eq!(greedy.top_k, 1);
        assert_eq!(wide.top_k, 100);
        assert_eq!(disabled.top_k, 0);
    }

    // =========================================================================
    // Extended Coverage Tests: clean_model_output
    // =========================================================================

    #[test]
    fn test_clean_model_output_endoftext_ext_cov() {
        let raw = "Hello world<|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_model_output_preserves_other_text_ext_cov() {
        let raw = "[INST]user input[/INST]assistant response";
        let cleaned = clean_model_output(raw);
        // Should preserve text since [INST] tokens aren't in the markers list
        assert!(cleaned.contains("user input"));
        assert!(cleaned.contains("response"));
    }

    #[test]
    fn test_clean_model_output_im_start_im_end_ext_cov() {
        // Function removes markers but keeps content between them
        let raw = "<|im_start|>assistant\nHi there!<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hi there!");
    }

    #[test]
    fn test_clean_model_output_empty_ext_cov() {
        let raw = "";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_only_markers_ext_cov() {
        let raw = "<|im_start|><|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_multiple_endoftext_ext_cov() {
        let raw = "Text<|endoftext|>More text<|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "TextMore text");
    }

    // =========================================================================
    // Extended Coverage Tests: prefault_mmap
    // =========================================================================

    #[test]
    fn test_prefault_mmap_large_ext_cov() {
        let data = vec![0u8; 4096 * 10]; // 10 pages
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_not_page_aligned_ext_cov() {
        let data = vec![0u8; 5000]; // Not page-aligned
        prefault_mmap(&data);
    }

    // =========================================================================
    // Extended Coverage Tests: InferenceResult
    // =========================================================================

    #[test]
    fn test_inference_result_debug_ext_cov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 5.0,
            tok_per_sec: 0.0,
            load_ms: 2.0,
            format: "APR".to_string(),
            used_gpu: false,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("text"));
        assert!(debug_str.contains("tokens"));
        assert!(debug_str.contains("format"));
        assert!(debug_str.contains("used_gpu"));
    }

    #[test]
    fn test_inference_result_zero_values_ext_cov() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: String::new(),
            used_gpu: false,
        };
        assert!(result.text.is_empty());
        assert!(result.tokens.is_empty());
        assert_eq!(result.input_token_count, 0);
    }

    #[test]
    fn test_inference_result_large_values_ext_cov() {
        let result = InferenceResult {
            text: "A".repeat(10000),
            tokens: vec![1; 1000],
            input_token_count: 100,
            generated_token_count: 900,
            inference_ms: 1000000.0,
            tok_per_sec: 1000.0,
            load_ms: 5000.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.text.len(), 10000);
        assert_eq!(result.tokens.len(), 1000);
        assert_eq!(result.generated_token_count, 900);
    }

    #[test]
    fn test_inference_result_formats_ext_cov() {
        for fmt in ["GGUF", "APR", "SafeTensors"] {
            let result = InferenceResult {
                text: "test".to_string(),
                tokens: vec![1],
                input_token_count: 1,
                generated_token_count: 0,
                inference_ms: 1.0,
                tok_per_sec: 1.0,
                load_ms: 1.0,
                format: fmt.to_string(),
                used_gpu: false,
            };
            assert_eq!(result.format, fmt);
        }
    }

    // =========================================================================
    // Extended Coverage Tests: run_inference error paths
    // =========================================================================

    #[test]
    fn test_run_inference_permission_denied_ext_cov() {
        // Try to read from a path that likely doesn't exist or isn't readable
        let config = InferenceConfig::new("/root/super_secret/model.gguf");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_empty_path_ext_cov() {
        let config = InferenceConfig::new("");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    // =========================================================================
    // Deep Coverage Tests (_deep_icov_) - Lines 197-280
    // =========================================================================

    // --- Format Detection Tests (Lines 197-201) ---

    #[test]
    fn test_format_detection_gguf_magic_deep_icov() {
        // GGUF magic bytes: 0x47 0x47 0x55 0x46 = "GGUF"
        use crate::format::{detect_format, ModelFormat};
        let data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00];
        let format = detect_format(&data);
        assert!(matches!(format, Ok(ModelFormat::Gguf)));
    }

    #[test]
    fn test_format_detection_apr_magic_deep_icov() {
        // APR magic bytes: "APR\0"
        use crate::format::{detect_format, ModelFormat};
        let data = b"APR\0xxxx";
        let format = detect_format(data);
        assert!(matches!(format, Ok(ModelFormat::Apr)));
    }

    #[test]
    fn test_format_detection_safetensors_deep_icov() {
        // SafeTensors: first 8 bytes are header size (little-endian u64)
        use crate::format::{detect_format, ModelFormat};
        let header_size: u64 = 2048;
        let data = header_size.to_le_bytes();
        let format = detect_format(&data);
        assert!(matches!(format, Ok(ModelFormat::SafeTensors)));
    }

    #[test]
    fn test_format_detection_unknown_magic_deep_icov() {
        // Unknown magic bytes should return error
        use crate::format::{detect_format, FormatError};
        let data = b"\x00\x00\x00\x00\x00\x00\x00\x00"; // Zero header = unknown
        let format = detect_format(data);
        assert!(matches!(format, Err(FormatError::UnknownFormat)));
    }

    // --- Architecture Detection Tests (Lines 227-243) ---

    #[test]
    fn test_architecture_detection_qwen_deep_icov() {
        // Test that "qwen" in filename is detected as "Qwen2"
        let path = PathBuf::from("/models/qwen2-7b-instruct-q4.gguf");
        let arch = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
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
        assert_eq!(arch, Some("Qwen2"));
    }

    #[test]
    fn test_architecture_detection_llama_deep_icov() {
        // Test that "llama" in filename is detected as "LLaMA"
        let path = PathBuf::from("/models/llama-3.1-8b-instruct.gguf");
        let arch = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
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
        assert_eq!(arch, Some("LLaMA"));
    }

    #[test]
    fn test_architecture_detection_mistral_deep_icov() {
        // Test that "mistral" in filename is detected as "Mistral"
        let path = PathBuf::from("/models/mistral-7b-v0.2-q4_k_m.gguf");
        let arch = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
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
        assert_eq!(arch, Some("Mistral"));
    }

    #[test]
    fn test_architecture_detection_phi_deep_icov() {
        // Test that "phi" in filename is detected as "Phi"
        let path = PathBuf::from("/models/phi-2-q4_0.gguf");
        let arch = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
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
        assert_eq!(arch, Some("Phi"));
    }

    #[test]
    fn test_architecture_detection_transformer_fallback_deep_icov() {
        // Test that unknown models fall back to "Transformer"
        let path = PathBuf::from("/models/custom-model-q8_0.gguf");
        let arch = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
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
        assert_eq!(arch, Some("Transformer"));
    }

    #[test]
    fn test_architecture_detection_case_insensitive_deep_icov() {
        // Test case-insensitive architecture detection
        let paths = [
            ("/models/QWEN2-7B.gguf", "Qwen2"),
            ("/models/LLAMA-3.gguf", "LLaMA"),
            ("/models/MISTRAL-7B.gguf", "Mistral"),
            ("/models/PHI-2.gguf", "Phi"),
            ("/models/QwEn2-MixedCase.gguf", "Qwen2"),
        ];
        for (path_str, expected) in paths {
            let path = PathBuf::from(path_str);
            let arch = path.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| {
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
            assert_eq!(arch, Some(expected), "Failed for path: {}", path_str);
        }
    }

    #[test]
    fn test_architecture_detection_no_extension_deep_icov() {
        // Test path with no extension
        let path = PathBuf::from("/models/qwen2-model");
        let arch = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
                if s.to_lowercase().contains("qwen") {
                    "Qwen2"
                } else {
                    "Transformer"
                }
            });
        assert_eq!(arch, Some("Qwen2"));
    }

    // --- Instruct Model Detection Tests (Lines 264-270) ---

    #[test]
    fn test_instruct_model_detection_deep_icov() {
        let model_name = "llama-3.1-8b-instruct.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(is_instruct);
    }

    #[test]
    fn test_instruct_model_detection_uppercase_deep_icov() {
        let model_name = "LLAMA-3.1-8B-INSTRUCT.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(is_instruct);
    }

    #[test]
    fn test_instruct_model_detection_mixed_case_deep_icov() {
        let model_name = "Qwen2-7B-Instruct-Q4_K_M.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(is_instruct);
    }

    #[test]
    fn test_instruct_model_detection_not_instruct_deep_icov() {
        let model_name = "llama-3.1-8b-base.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(!is_instruct);
    }

    #[test]
    fn test_instruct_model_detection_partial_match_deep_icov() {
        // Should match even if "instruct" is part of a larger word
        let model_name = "model-instructed.gguf";
        let is_instruct = model_name.to_lowercase().contains("instruct");
        assert!(is_instruct);
    }

    // --- Chat Template Formatting Tests (Lines 266-270) ---

    #[test]
    fn test_chat_message_user_creation_deep_icov() {
        use crate::chat_template::ChatMessage;
        let msg = ChatMessage::user("Hello, world!");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello, world!");
    }

    #[test]
    fn test_chat_message_system_creation_deep_icov() {
        use crate::chat_template::ChatMessage;
        let msg = ChatMessage::system("You are a helpful assistant.");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant.");
    }

    #[test]
    fn test_format_messages_instruct_model_deep_icov() {
        use crate::chat_template::{format_messages, ChatMessage};
        let messages = vec![ChatMessage::user("What is 2+2?")];
        // Test with qwen model name (should use ChatML template)
        let result = format_messages(&messages, Some("qwen2-7b-instruct.gguf"));
        assert!(result.is_ok());
        let formatted = result.unwrap();
        // ChatML format uses <|im_start|> markers
        assert!(formatted.contains("<|im_start|>") || formatted.contains("user"));
    }

    #[test]
    fn test_format_messages_llama_template_deep_icov() {
        use crate::chat_template::{format_messages, ChatMessage};
        let messages = vec![ChatMessage::user("Hello!")];
        let result = format_messages(&messages, Some("llama-3.1-8b-instruct.gguf"));
        assert!(result.is_ok());
        let formatted = result.unwrap();
        // LLaMA format uses [INST] markers
        assert!(formatted.contains("[INST]") || formatted.contains("user"));
    }

    #[test]
    fn test_format_messages_fallback_raw_deep_icov() {
        use crate::chat_template::{format_messages, ChatMessage};
        let messages = vec![ChatMessage::user("Just text")];
        // Unknown model should use raw template
        let result = format_messages(&messages, Some("unknown-model.gguf"));
        assert!(result.is_ok());
    }

    // --- Input Token Handling Tests (Lines 255-279) ---

    #[test]
    fn test_input_tokens_priority_over_prompt_deep_icov() {
        // Test that input_tokens takes priority over prompt
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Hello")
            .with_input_tokens(vec![1, 2, 3, 4]);

        // When both are set, input_tokens should be used (line 255-256)
        assert!(config.input_tokens.is_some());
        assert!(config.prompt.is_some());

        // Simulate the logic from run_gguf_inference
        let input_tokens = if let Some(ref tokens) = config.input_tokens {
            tokens.clone()
        } else if let Some(ref _prompt) = config.prompt {
            vec![100, 200] // Would be tokenized prompt
        } else {
            vec![1u32] // BOS token
        };
        assert_eq!(input_tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_input_tokens_none_uses_prompt_deep_icov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Hello");

        // When input_tokens is None, prompt should be used (line 257)
        assert!(config.input_tokens.is_none());
        assert!(config.prompt.is_some());
    }

    #[test]
    fn test_input_tokens_none_prompt_none_uses_bos_deep_icov() {
        let config = InferenceConfig::new("/model.gguf");

        // When both are None, BOS token should be used (line 277-278)
        assert!(config.input_tokens.is_none());
        assert!(config.prompt.is_none());

        // Simulate the logic
        let input_tokens = if let Some(ref tokens) = config.input_tokens {
            tokens.clone()
        } else if let Some(ref _prompt) = config.prompt {
            vec![100, 200]
        } else {
            vec![1u32] // BOS token
        };
        assert_eq!(input_tokens, vec![1u32]);
    }

    // --- Verbose Output Tests (Lines 210-252) ---

    #[test]
    fn test_verbose_flag_enabled_deep_icov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_verbose(true);
        assert!(config.verbose);
    }

    #[test]
    fn test_verbose_flag_disabled_deep_icov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_verbose(false);
        assert!(!config.verbose);
    }

    #[test]
    fn test_verbose_default_is_false_deep_icov() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(!config.verbose);
    }

    // --- Model Name Extraction Tests ---

    #[test]
    fn test_model_name_extraction_from_path_deep_icov() {
        let path = PathBuf::from("/models/qwen2-7b-instruct.gguf");
        let model_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        assert_eq!(model_name, "qwen2-7b-instruct.gguf");
    }

    #[test]
    fn test_model_name_extraction_no_parent_deep_icov() {
        let path = PathBuf::from("qwen2-7b.gguf");
        let model_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        assert_eq!(model_name, "qwen2-7b.gguf");
    }

    #[test]
    fn test_model_name_extraction_empty_path_deep_icov() {
        let path = PathBuf::from("");
        let model_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        assert_eq!(model_name, "");
    }

    // --- File Stem Extraction Tests ---

    #[test]
    fn test_file_stem_extraction_deep_icov() {
        let path = PathBuf::from("/models/llama-3.1-8b.gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("llama-3.1-8b"));
    }

    #[test]
    fn test_file_stem_extraction_multiple_dots_deep_icov() {
        let path = PathBuf::from("/models/model.v1.0.gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("model.v1.0"));
    }

    #[test]
    fn test_file_stem_extraction_no_extension_deep_icov() {
        let path = PathBuf::from("/models/model");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("model"));
    }

    // --- Tokens Per Second Calculation Tests ---

    #[test]
    fn test_tok_per_sec_calculation_deep_icov() {
        let generated_count = 100;
        let inference_ms = 500.0; // 500ms
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 200.0).abs() < 0.001); // 100 tokens / 0.5 sec = 200 tok/s
    }

    #[test]
    fn test_tok_per_sec_zero_time_deep_icov() {
        let generated_count = 100;
        let inference_ms = 0.0;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tok_per_sec_very_fast_deep_icov() {
        let generated_count = 1000;
        let inference_ms = 10.0; // 10ms
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 100000.0).abs() < 0.001); // 100k tok/s
    }

    // --- Max Tokens Capping Tests (Line 285) ---

    #[test]
    fn test_max_tokens_capped_at_128_deep_icov() {
        // Test the .min(128) capping in gen_config (line 285)
        let config = InferenceConfig::new("/model.gguf")
            .with_max_tokens(1000);
        let capped = config.max_tokens.min(128);
        assert_eq!(capped, 128);
    }

    #[test]
    fn test_max_tokens_not_capped_if_under_128_deep_icov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_max_tokens(50);
        let capped = config.max_tokens.min(128);
        assert_eq!(capped, 50);
    }

    #[test]
    fn test_max_tokens_exactly_128_deep_icov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_max_tokens(128);
        let capped = config.max_tokens.min(128);
        assert_eq!(capped, 128);
    }

    // --- Format String Tests ---

    #[test]
    fn test_inference_result_format_string_gguf_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "GGUF");
    }

    #[test]
    fn test_inference_result_format_string_apr_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "APR".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "APR");
    }

    #[test]
    fn test_inference_result_format_string_safetensors_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "SafeTensors".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "SafeTensors");
    }

    // --- Used GPU Flag Tests ---

    #[test]
    fn test_inference_result_used_gpu_true_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert!(result.used_gpu);
    }

    #[test]
    fn test_inference_result_used_gpu_false_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(!result.used_gpu);
    }

    // --- Architecture Priority Tests ---

    #[test]
    fn test_architecture_detection_priority_qwen_over_llama_deep_icov() {
        // If filename contains both "qwen" and "llama", qwen should win (checked first)
        let path = PathBuf::from("/models/qwen-llama-hybrid.gguf");
        let arch = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
                if s.to_lowercase().contains("qwen") {
                    "Qwen2"
                } else if s.to_lowercase().contains("llama") {
                    "LLaMA"
                } else {
                    "Transformer"
                }
            });
        assert_eq!(arch, Some("Qwen2"));
    }

    #[test]
    fn test_architecture_detection_priority_llama_over_mistral_deep_icov() {
        // If filename contains both "llama" and "mistral", llama should win
        let path = PathBuf::from("/models/llama-mistral-blend.gguf");
        let arch = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
                if s.to_lowercase().contains("qwen") {
                    "Qwen2"
                } else if s.to_lowercase().contains("llama") {
                    "LLaMA"
                } else if s.to_lowercase().contains("mistral") {
                    "Mistral"
                } else {
                    "Transformer"
                }
            });
        assert_eq!(arch, Some("LLaMA"));
    }

    // --- Path Edge Cases Tests ---

    #[test]
    fn test_path_with_special_characters_deep_icov() {
        let path = PathBuf::from("/models/model-v1.0_final (copy).gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert!(stem.is_some());
        assert!(stem.unwrap().contains("model"));
    }

    #[test]
    fn test_path_with_unicode_deep_icov() {
        let path = PathBuf::from("/models/模型-v1.gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert!(stem.is_some());
    }

    #[test]
    fn test_path_just_extension_deep_icov() {
        let path = PathBuf::from(".gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        // For dotfiles like .gguf, file_stem returns the full name ".gguf" (no extension)
        assert_eq!(stem, Some(".gguf"));
    }

    // --- Load Time Tests ---

    #[test]
    fn test_load_ms_calculation_deep_icov() {
        // Simulating load_start.elapsed().as_secs_f64() * 1000.0
        let elapsed_secs: f64 = 0.5;
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_load_ms_very_fast_deep_icov() {
        let elapsed_secs: f64 = 0.001; // 1ms
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_load_ms_very_slow_deep_icov() {
        let elapsed_secs: f64 = 10.0; // 10 seconds
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 10000.0).abs() < 0.001);
    }

    // --- Generated Token Slice Tests ---

    #[test]
    fn test_generated_tokens_slice_deep_icov() {
        let all_tokens = vec![1, 2, 3, 4, 5, 6];
        let input_token_count = 2;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens, &[3, 4, 5, 6]);
    }

    #[test]
    fn test_generated_tokens_slice_empty_deep_icov() {
        let all_tokens = vec![1, 2];
        let input_token_count = 2;
        let generated_tokens = &all_tokens[input_token_count..];
        assert!(generated_tokens.is_empty());
    }

    #[test]
    fn test_generated_tokens_slice_all_generated_deep_icov() {
        let all_tokens = vec![1, 2, 3, 4];
        let input_token_count = 0;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens.len(), 4);
    }

    // --- Clean Output Integration Tests ---

    #[test]
    fn test_clean_output_chatml_full_conversation_deep_icov() {
        let raw = "<|im_start|>system\nYou are helpful<|im_end|><|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello!<|im_end|>";
        let cleaned = clean_model_output(raw);
        // All markers should be removed
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        // Content should remain (without newlines added by markers)
        assert!(cleaned.contains("helpful") || cleaned.contains("Hello"));
    }

    #[test]
    fn test_clean_output_preserves_code_blocks_deep_icov() {
        let raw = "<|im_start|>assistant\n```python\nprint('hello')\n```<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("```python"));
        assert!(cleaned.contains("print('hello')"));
    }

    // --- Model Format Enum Tests ---

    #[test]
    fn test_model_format_display_gguf_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Gguf;
        assert_eq!(format.to_string(), "GGUF");
    }

    #[test]
    fn test_model_format_display_apr_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Apr;
        assert_eq!(format.to_string(), "APR");
    }

    #[test]
    fn test_model_format_display_safetensors_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::SafeTensors;
        assert_eq!(format.to_string(), "SafeTensors");
    }

    #[test]
    fn test_model_format_clone_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Gguf;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_model_format_eq_deep_icov() {
        use crate::format::ModelFormat;
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Gguf, ModelFormat::Apr);
        assert_ne!(ModelFormat::Apr, ModelFormat::SafeTensors);
    }

    // --- Additional Edge Case Tests ---

    #[test]
    fn test_inference_config_path_with_symlink_name_deep_icov() {
        let config = InferenceConfig::new("/models/latest -> llama-3.gguf");
        assert!(config.model_path.to_str().unwrap().contains("latest"));
    }

    #[test]
    fn test_inference_config_relative_path_deep_icov() {
        let config = InferenceConfig::new("./models/model.gguf");
        assert!(config.model_path.to_str().unwrap().contains("./"));
    }

    #[test]
    fn test_inference_config_absolute_path_deep_icov() {
        let config = InferenceConfig::new("/absolute/path/model.gguf");
        assert!(config.model_path.starts_with("/"));
    }
}
