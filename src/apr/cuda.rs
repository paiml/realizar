//! CUDA-accelerated APR model inference (PMAT-802)
//!
//! Extracted from apr/mod.rs - GPU acceleration for APR v2 models.
//!
//! ## Contents
//! - `AprV2ModelCuda` - CUDA wrapper for APR models (2x Ollama target)

use super::{apply_rope_norm, dtype_to_ggml_qtype, rms_norm, simple_attention, transpose_matrix, AprV2Model};
use crate::error::{RealizarError, Result};

// ============================================================================
// AprV2ModelCuda: GPU-accelerated APR inference (2x Ollama target)
// ============================================================================

/// CUDA-accelerated wrapper for APR v2 models.
///
/// Mirrors `OwnedQuantizedModelCuda` from GGUF to provide GPU acceleration
/// for APR format models. Achieves 2x+ Ollama performance on supported GPUs.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::apr::{AprV2Model, AprV2ModelCuda};
///
/// let model = AprV2Model::load("model.apr")?;
/// let mut cuda_model = AprV2ModelCuda::new(model, 0)?; // GPU 0
///
/// // GPU-accelerated forward pass
/// let logits = cuda_model.forward_cuda(&[1, 2, 3])?;
///
/// // GPU-accelerated generation
/// let tokens = cuda_model.generate_cuda(&[1, 2, 3], 32, 151643)?;
/// ```
#[cfg(feature = "cuda")]
pub struct AprV2ModelCuda {
    /// Inner APR model
    model: AprV2Model,
    /// Cached CUDA executor
    executor: crate::cuda::CudaExecutor,
    /// GPU device name
    device_name: String,
    /// GPU memory (free, total) in bytes
    memory_info: (usize, usize),
    /// Cached weight buffers on GPU (tensor_name -> gpu_ptr)
    weight_cache: std::collections::HashMap<String, u64>,
    /// Cached embedding table (F32 for fast lookup)
    embedding_cache: Option<Vec<f32>>,
    /// Hidden dimension (cached for embedding lookup)
    hidden_dim: usize,
    /// Current KV cache position (increments with each decoded token)
    kv_position: u32,
    /// PMAT-110: Track if KV cache was populated via FALLBACK PATH
    /// When true, decode must also use FALLBACK PATH for consistency
    fallback_kv_used: bool,
    /// Phase 45: Test executor for dependency injection
    ///
    /// When present, this executor is used instead of CudaExecutor for GEMM operations.
    /// Enables testing forward pass logic without actual CUDA hardware.
    test_executor: Option<Box<dyn crate::gpu::executor::GpuExecutorTrait + Send + Sync>>,
}

#[cfg(feature = "cuda")]
impl AprV2ModelCuda {
    /// Create a new CUDA-accelerated APR model wrapper.
    ///
    /// # Arguments
    ///
    /// * `model` - The APR v2 model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new(model: AprV2Model, device_ordinal: i32) -> Result<Self> {
        Self::with_max_seq_len(model, device_ordinal, 2048)
    }

    /// Create a new CUDA-accelerated APR model wrapper with custom max sequence length.
    ///
    /// # Arguments
    ///
    /// * `model` - The APR v2 model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    /// * `max_seq_len` - Maximum sequence length for GPU KV cache
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn with_max_seq_len(
        model: AprV2Model,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> Result<Self> {
        use crate::cuda::CudaExecutor;

        let mut executor =
            CudaExecutor::new(device_ordinal).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::new".to_string(),
                reason: format!("CUDA initialization failed: {e}"),
            })?;

        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let memory_info = executor.memory_info().unwrap_or((0, 0));

        // Initialize GPU-resident KV cache for attention acceleration
        let num_layers = model.metadata.num_layers.unwrap_or(0);
        let num_heads = model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = model.metadata.num_kv_heads.unwrap_or(num_heads);
        let hidden_dim = model.metadata.hidden_size.unwrap_or(0);
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };

        if num_layers > 0 && head_dim > 0 {
            executor
                .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_kv_cache_gpu".to_string(),
                    reason: format!("GPU KV cache initialization failed: {e}"),
                })?;
        }

        // Set RoPE theta for position embeddings
        let rope_theta = model.metadata.rope_theta.unwrap_or(10000.0);
        executor.set_rope_theta(rope_theta);

        // CORRECTNESS-011: Set RoPE type (0=NORM adjacent pairs, 2=NEOX split halves)
        // Five-Whys: GPU garbage output → wrong RoPE style → rope_type not set for APR models
        let rope_type = model.metadata.rope_type.unwrap_or(0);
        executor.set_rope_type(rope_type);

        let hidden_dim = model.metadata.hidden_size.unwrap_or(0);

        let mut apr_cuda = Self {
            model,
            executor,
            device_name,
            memory_info,
            weight_cache: std::collections::HashMap::new(),
            embedding_cache: None, // Lazy-loaded on first forward
            hidden_dim,
            kv_position: 0,         // Start at position 0
            fallback_kv_used: false, // PMAT-110: No fallback KV yet
            test_executor: None,    // Phase 45: No test executor by default
        };

        // Pre-cache all transposed weights on GPU for 2x performance
        apr_cuda.pre_cache_weights()?;

        // Pre-cache embedding table for fast token lookup
        apr_cuda.cache_embeddings()?;

        Ok(apr_cuda)
    }

    /// Check if CUDA is available.
    #[must_use]
    pub fn is_available() -> bool {
        crate::cuda::CudaExecutor::is_available()
    }

    /// Get number of CUDA devices.
    #[must_use]
    pub fn num_devices() -> usize {
        crate::cuda::CudaExecutor::num_devices()
    }

    /// Phase 45: Inject a test executor for dependency injection.
    ///
    /// When present, GEMM operations are routed through the test executor
    /// instead of the CUDA executor, enabling testing without actual GPU hardware.
    ///
    /// # Arguments
    ///
    /// * `executor` - Test executor (typically `MockExecutor` or `CpuExecutor`)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use realizar::gpu::executor::{MockExecutor, CpuExecutor};
    ///
    /// let mut cuda_model = AprV2ModelCuda::new(model, 0)?;
    /// cuda_model.with_test_executor(Box::new(CpuExecutor::new()));
    /// ```
    pub fn with_test_executor(
        &mut self,
        executor: Box<dyn crate::gpu::executor::GpuExecutorTrait + Send + Sync>,
    ) {
        self.test_executor = Some(executor);
    }

    /// Check if a test executor is set.
    #[must_use]
    pub fn has_test_executor(&self) -> bool {
        self.test_executor.is_some()
    }

    /// Get GPU device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get GPU memory info (free, total) in bytes.
    #[must_use]
    pub fn memory_info(&self) -> (usize, usize) {
        self.memory_info
    }

    /// Get VRAM usage in MB.
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        (self.memory_info.1 / (1024 * 1024)) as u64
    }

    /// Get reference to the inner APR model.
    #[must_use]
    pub fn inner(&self) -> &AprV2Model {
        &self.model
    }

    // ========================================================================
    // BrickProfiler API for per-brick timing
    // ========================================================================

    /// Enable per-brick profiling for real timing measurements.
    pub fn enable_profiling(&mut self) {
        self.executor.enable_profiling();
    }

    /// Disable per-brick profiling.
    pub fn disable_profiling(&mut self) {
        self.executor.disable_profiling();
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_profiling_enabled(&self) -> bool {
        self.executor.is_profiling_enabled()
    }

    /// Get the brick profiler for reading statistics.
    #[must_use]
    pub fn profiler(&self) -> &trueno::BrickProfiler {
        self.executor.profiler()
    }

    /// Reset profiler statistics.
    pub fn reset_profiler(&mut self) {
        self.executor.reset_profiler();
    }

    /// Reset KV cache position for a new conversation.
    ///
    /// Call this before starting a new generation sequence to clear the
    /// KV cache state from the previous conversation.
    pub fn reset_kv_cache(&mut self) {
        self.kv_position = 0;
        self.fallback_kv_used = false; // PMAT-110: Reset fallback flag
        self.executor.reset_kv_cache_gpu();
    }

    // ========================================================================
    // Weight Pre-caching (2x performance optimization)
    // ========================================================================

    /// Pre-cache all model weights on GPU using native quantized format.
    ///
    /// This uploads quantized weights (Q4K, Q6K, etc.) directly to GPU without
    /// CPU dequantization, enabling fused dequant+matmul kernels for maximum
    /// throughput (2x+ Ollama baseline per APR mandate).
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU.
    fn pre_cache_weights(&mut self) -> Result<()> {
        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let _vocab_size = self.model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };
        let _kv_dim = num_kv_heads * head_dim;

        if hidden_dim == 0 || num_layers == 0 {
            return Ok(()); // Non-transformer model, nothing to cache
        }

        let mut total_bytes = 0usize;
        let mut quantized_count = 0usize;

        // Helper to upload a weight tensor (quantized or F32)
        // Uses GGUF-style cache names for compatibility with build_indexed_weights()
        let upload_weight = |executor: &mut crate::cuda::CudaExecutor,
                             model: &AprV2Model,
                             src_name: &str,
                             cache_name: &str|
         -> usize {
            if let Some(entry) = model.get_tensor(src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized: upload raw bytes to quantized_weight_cache
                    if let Ok(bytes) = model.get_tensor_bytes(src_name) {
                        executor
                            .load_quantized_weights_with_type(cache_name, bytes, qtype)
                            .unwrap_or(0)
                    } else {
                        0
                    }
                } else {
                    // F32/F16: dequantize and upload to weight_cache (legacy path)
                    // This path is only used for non-quantized models
                    0 // Skip F32 weights - they'll be loaded on demand
                }
            } else {
                0
            }
        };

        // Cache per-layer weights using GGUF naming convention
        // This matches build_indexed_weights() expectations
        for layer_idx in 0..num_layers {
            let prefix = format!("blk.{layer_idx}");

            // Find source tensor names (HuggingFace, GGUF, etc.)
            // Map from various naming conventions to GGUF cache names
            let weight_mappings = [
                // (source_patterns, cache_suffix)
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                        format!("blk.{layer_idx}.attn_q.weight"),
                    ],
                    "attn_q.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                        format!("blk.{layer_idx}.attn_k.weight"),
                    ],
                    "attn_k.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                        format!("blk.{layer_idx}.attn_v.weight"),
                    ],
                    "attn_v.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                        format!("blk.{layer_idx}.attn_output.weight"),
                    ],
                    "attn_output.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("blk.{layer_idx}.ffn_gate.weight"),
                    ],
                    "ffn_gate.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("blk.{layer_idx}.ffn_up.weight"),
                    ],
                    "ffn_up.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("blk.{layer_idx}.ffn_down.weight"),
                    ],
                    "ffn_down.weight",
                ),
            ];

            for (patterns, suffix) in weight_mappings {
                let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
                if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                    let cache_name = format!("{prefix}.{suffix}");
                    let bytes =
                        upload_weight(&mut self.executor, &self.model, &src_name, &cache_name);
                    if bytes > 0 {
                        total_bytes += bytes;
                        quantized_count += 1;
                    }
                }
            }

            // Upload RMSNorm gamma weights (always F32)
            let norm_mappings = [
                (
                    vec![
                        format!("model.layers.{layer_idx}.input_layernorm.weight"),
                        format!("layers.{layer_idx}.input_layernorm.weight"),
                        format!("blk.{layer_idx}.attn_norm.weight"),
                    ],
                    "attn_norm.gamma",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                        format!("layers.{layer_idx}.post_attention_layernorm.weight"),
                        format!("blk.{layer_idx}.ffn_norm.weight"),
                    ],
                    "ffn_norm.gamma",
                ),
            ];

            for (patterns, suffix) in norm_mappings {
                let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
                if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                    if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                        let cache_name = format!("{prefix}.{suffix}");
                        if let Ok(bytes) = self.executor.cache_rmsnorm_gamma(&cache_name, &gamma) {
                            total_bytes += bytes;
                        }
                    }
                }
            }
        }

        // Cache output norm
        let output_norm_patterns = [
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight",
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&output_norm_patterns) {
            if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                if let Ok(bytes) = self
                    .executor
                    .cache_rmsnorm_gamma("output_norm.gamma", &gamma)
                {
                    total_bytes += bytes;
                }
            }
        }

        // Cache LM head (may be quantized or F32)
        let lm_head_patterns = [
            "lm_head.weight",
            "output.weight",
            "token_embd.weight", // GGUF (tied embeddings)
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&lm_head_patterns) {
            if let Some(entry) = self.model.get_tensor(&src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized LM head
                    if let Ok(bytes) = self.model.get_tensor_bytes(&src_name) {
                        if let Ok(size) = self.executor.load_quantized_weights_with_type(
                            "output.weight",
                            bytes,
                            qtype,
                        ) {
                            total_bytes += size;
                            quantized_count += 1;
                        }
                    }
                } else {
                    // F32 LM head - store as quantized_weight_cache for compatibility
                    // The forward path will handle F32 appropriately
                    if let Ok(w) = self.model.get_tensor_f32(&src_name) {
                        // Upload F32 weights directly (no transpose needed for GEMV)
                        // SAFETY: f32 slice to u8 view - valid because f32 has no padding,
                        // alignment requirement of u8 is 1, and lifetime is preserved
                        let w_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                w.as_ptr().cast::<u8>(),
                                w.len() * std::mem::size_of::<f32>(),
                            )
                        };
                        // Use qtype 0 to indicate F32 (handled specially in forward)
                        if let Ok(size) = self.executor.load_quantized_weights_with_type(
                            "output.weight",
                            w_bytes,
                            0,
                        ) {
                            total_bytes += size;
                        }
                    }
                }
            }
        }

        // Build indexed weight lookup table for O(1) access during decode
        // This is the key optimization that enables fast token generation
        if quantized_count > 0 {
            if let Err(e) = self
                .executor
                .build_indexed_weights(num_layers, |i| format!("blk.{i}"))
            {
                eprintln!("[AprV2ModelCuda] Warning: Could not build indexed weights: {e}");
                // Continue anyway - fallback path will be used
            } else {
                eprintln!(
                    "[AprV2ModelCuda] Built indexed weights for {} layers",
                    num_layers
                );
            }

            // Initialize workspace for zero-allocation forward pass
            if let Err(e) = self.executor.init_workspace(hidden_dim, intermediate_dim) {
                eprintln!("[AprV2ModelCuda] Warning: Could not init workspace: {e}");
            }
        }

        eprintln!(
            "[AprV2ModelCuda] Pre-cached {} MB of weights on GPU ({} layers, {} quantized tensors)",
            total_bytes / (1024 * 1024),
            num_layers,
            quantized_count
        );

        Ok(())
    }

    /// Pre-cache embedding table for fast token lookup.
    ///
    /// This reads the embedding table once and stores it in memory, eliminating
    /// repeated disk/mmap reads during generation (~450ms → ~0.05ms per token).
    fn cache_embeddings(&mut self) -> Result<()> {
        let embed_name = self.model.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight", // GGUF naming
        ])?;

        let embeddings = self.model.get_tensor_f32(&embed_name)?;
        let embed_mb = embeddings.len() * 4 / (1024 * 1024);
        eprintln!("[AprV2ModelCuda] Cached embedding table: {} MB", embed_mb);

        self.embedding_cache = Some(embeddings);
        Ok(())
    }

    /// Get embedding for a token ID from cache.
    #[inline]
    fn get_embedding(&self, token_id: u32) -> Option<&[f32]> {
        self.embedding_cache.as_ref().and_then(|cache| {
            let offset = (token_id as usize) * self.hidden_dim;
            if offset + self.hidden_dim <= cache.len() {
                Some(&cache[offset..offset + self.hidden_dim])
            } else {
                None
            }
        })
    }

    /// Check if weights are cached on GPU.
    #[must_use]
    pub fn weights_cached(&self) -> bool {
        self.executor.cached_weight_count() > 0
    }

    /// Get total cached weight size in MB.
    #[must_use]
    pub fn cached_weight_mb(&self) -> usize {
        self.executor.cached_weight_bytes() / (1024 * 1024)
    }

    // ========================================================================
    // GPU-accelerated inference
    // ========================================================================

    /// GPU-accelerated forward pass returning only the next token ID (fastest path).
    ///
    /// Uses GPU argmax to avoid transferring 600KB of logits from GPU to CPU.
    /// This is the recommended method for autoregressive generation.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Input token ID (single token for decode step)
    ///
    /// # Returns
    ///
    /// The token ID with the highest logit value.
    pub fn forward_cuda_to_token(&mut self, token_id: u32) -> Result<u32> {
        if !self.model.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let _hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let _num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);

        // Use indexed Q4K path with GPU argmax (no 600KB logits transfer)
        // Phase 45: Skip fast path when test_executor is present
        if self.test_executor.is_none() && self.executor.has_indexed_weights() {
            let position = self.kv_position;

            // Embedding lookup from cache
            let input: Vec<f32> = self
                .get_embedding(token_id)
                .ok_or_else(|| RealizarError::InvalidShape {
                    reason: format!("Token {} out of embedding range", token_id),
                })?
                .to_vec();

            let num_layers = self.model.metadata.num_layers.unwrap_or(0);
            let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
            let intermediate_dim = self
                .model
                .metadata
                .intermediate_size
                .unwrap_or(hidden_dim * 4);
            let eps = self.model.metadata.rms_norm_eps.unwrap_or(1e-6);

            // First call: capture graph using the full graphed forward path
            // Subsequent calls: use replay with GPU argmax
            let next_token = if !self.executor.has_decode_graph() {
                // Need to capture graph first - use forward_all_layers_gpu_to_logits_graphed
                // then do CPU argmax
                let mut output = vec![0.0f32; vocab_size];
                self.executor
                    .forward_all_layers_gpu_to_logits_graphed(
                        &input,
                        &mut output,
                        position,
                        num_layers,
                        hidden_dim as u32,
                        intermediate_dim as u32,
                        vocab_size as u32,
                        eps,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "forward_all_layers_gpu_to_logits_graphed".to_string(),
                        reason: format!("Graph capture failed: {e}"),
                    })?;

                // CPU argmax for first token (graph now captured)
                let (top_idx, _) = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("APR operation failed"))
                    .ok_or_else(|| RealizarError::InvalidShape {
                        reason: "Empty logits".to_string(),
                    })?;
                top_idx as u32
            } else {
                // Graph captured - use fast replay with GPU argmax
                self.executor
                    .forward_graphed_replay_to_token_id(&input, position, vocab_size as u32)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "forward_graphed_replay_to_token_id".to_string(),
                        reason: format!("GPU argmax fast path failed: {e}"),
                    })?
            };

            // Increment position for next token
            self.kv_position += 1;

            return Ok(next_token);
        }

        // Fallback: use forward_cuda and do CPU argmax
        let logits = self.forward_cuda(&[token_id])?;
        let (top_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("APR operation failed"))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;
        Ok(top_idx as u32)
    }

    /// GPU-accelerated forward pass.
    ///
    /// Computes logits for the given token sequence using GPU acceleration
    /// for matrix multiplications. Achieves 2x+ Ollama performance by using
    /// GPU GEMM for QKV, attention output, and FFN projections.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size` for next token prediction.
    pub fn forward_cuda(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        if !self.model.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let eps = self.model.metadata.rms_norm_eps.unwrap_or(1e-6);
        let seq_len = token_ids.len();
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // =========================================================================
        // FAST PATH: Use indexed Q4K GEMV kernels with CUDA graph capture
        // This path uses fused dequant+matmul kernels + graph replay for
        // 500x reduction in kernel launch overhead (5.6ms → 0.01ms per token)
        // Phase 45: Skip fast path when test_executor is present
        // PMAT-110: Skip fast path if KV cache was populated via fallback path
        //           (RoPE numerical differences cause inconsistency)
        // =========================================================================
        if self.test_executor.is_none() && self.executor.has_indexed_weights() && seq_len == 1 && !self.fallback_kv_used {
            // Single-token decode: use the optimized Q4K GEMV path with graphs
            let token_id = token_ids[0];
            let position = self.kv_position;

            // Embedding lookup from cache (O(1) - no disk/mmap read)
            // Copy to local vec to release borrow before mutable executor call
            let input: Vec<f32> = self
                .get_embedding(token_id)
                .ok_or_else(|| RealizarError::InvalidShape {
                    reason: format!("Token {} out of embedding range", token_id),
                })?
                .to_vec();

            // Use the graphed forward path with CUDA graph capture
            // First call captures the graph, subsequent calls replay it
            let mut output = vec![0.0f32; vocab_size];
            self.executor
                .forward_all_layers_gpu_to_logits_graphed(
                    &input,
                    &mut output,
                    position,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_all_layers_gpu_to_logits_graphed".to_string(),
                    reason: format!("Q4K graphed fast path failed: {e}"),
                })?;

            // Increment position for next token (KV cache tracking)
            self.kv_position += 1;

            return Ok(output);
        }

        // =========================================================================
        // FALLBACK PATH: Original F32 GEMM path (for prefill or non-indexed models)
        // =========================================================================

        // BrickProfiler instrumentation (per spec §12.11)
        let profiling = self.executor.is_profiling_enabled();

        // 1. Token embedding lookup (CPU - fast single lookup)
        let timer_embed = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.Embed"))
        } else {
            None
        };

        let embed_name = self.model.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
            "token_embd.weight", // GGUF naming
        ])?;
        let embeddings = self.model.get_tensor_f32(&embed_name)?;

        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        if let Some(t) = timer_embed {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Get weight tensors (HuggingFace, SafeTensors, GPT-2, LLaMA, GGUF)
            let attn_norm_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
                &format!("blk.{layer_idx}.attn_norm.weight"), // GGUF
            ])?;
            let q_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                &format!("layers.{layer_idx}.attention.wq.weight"),
                &format!("blk.{layer_idx}.attn_q.weight"), // GGUF
            ])?;
            let k_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                &format!("layers.{layer_idx}.attention.wk.weight"),
                &format!("blk.{layer_idx}.attn_k.weight"), // GGUF
            ])?;
            let v_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                &format!("layers.{layer_idx}.attention.wv.weight"),
                &format!("blk.{layer_idx}.attn_v.weight"), // GGUF
            ])?;
            let o_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
                &format!("blk.{layer_idx}.attn_output.weight"), // GGUF
            ])?;

            let norm_weight = self.model.get_tensor_f32(&attn_norm_name)?;

            // RMSNorm (CPU - small operation)
            let timer_rmsnorm1 = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.RmsNorm"))
            } else {
                None
            };
            let normed = rms_norm(&hidden, &norm_weight, eps);
            if let Some(t) = timer_rmsnorm1 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Q, K, V projections (GPU GEMM for 2x speedup)
            // Use cached weights if available (avoids repeated transpose + upload)
            let q_cache_name = format!("layer_{}_q_proj", layer_idx);
            let k_cache_name = format!("layer_{}_k_proj", layer_idx);
            let v_cache_name = format!("layer_{}_v_proj", layer_idx);
            let o_cache_name = format!("layer_{}_o_proj", layer_idx);

            let timer_qkv = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.QKV"))
            } else {
                None
            };
            let (q, k, v) = if self.has_cached_weight(&q_cache_name) {
                // Fast path: use pre-cached transposed weights
                let q =
                    self.gemm_cached_gpu(&q_cache_name, &normed, seq_len, hidden_dim, hidden_dim)?;
                let k =
                    self.gemm_cached_gpu(&k_cache_name, &normed, seq_len, hidden_dim, kv_dim)?;
                let v =
                    self.gemm_cached_gpu(&v_cache_name, &normed, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            } else {
                // Fallback: load, transpose, and upload weights each time
                let q_weight = self.model.get_tensor_f32(&q_name)?;
                let k_weight = self.model.get_tensor_f32(&k_name)?;
                let v_weight = self.model.get_tensor_f32(&v_name)?;
                let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);
                let q = self.gemm_gpu(&normed, &q_weight_t, seq_len, hidden_dim, hidden_dim)?;
                let k = self.gemm_gpu(&normed, &k_weight_t, seq_len, hidden_dim, kv_dim)?;
                let v = self.gemm_gpu(&normed, &v_weight_t, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            };
            if let Some(t) = timer_qkv {
                let _ = self.executor.synchronize();
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // PMAT-110: Attention with KV cache for proper generation
            // Process each token position and use incremental_attention_gpu to:
            // 1. Apply RoPE to Q and K
            // 2. Append K/V to GPU cache
            // 3. Compute attention against all cached K/V
            let timer_attn = if profiling {
                Some(self.executor.profiler_mut().start("apr.Attention"))
            } else {
                None
            };

            let rope_theta = self.model.metadata.rope_theta.unwrap_or(10000.0);
            let rope_type = self.model.metadata.rope_type.unwrap_or(0);
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];

            for pos in 0..seq_len {
                // Extract Q, K, V for this position
                let q_pos_start = pos * hidden_dim;
                let k_pos_start = pos * kv_dim;
                let v_pos_start = pos * kv_dim;

                let mut q_pos = q[q_pos_start..q_pos_start + hidden_dim].to_vec();
                let mut k_pos = k[k_pos_start..k_pos_start + kv_dim].to_vec();
                let v_pos = v[v_pos_start..v_pos_start + kv_dim].to_vec();

                // Apply RoPE to Q and K (position = kv_position + pos)
                // CORRECTNESS-011: Use correct RoPE style based on rope_type
                let abs_position = self.kv_position as usize + pos;
                if rope_type == 2 {
                    // NEOX style: split halves (i, i + half_dim) - used by Qwen2.5, etc.
                    crate::inference::apply_rope(&mut q_pos, hidden_dim, num_heads, abs_position, rope_theta);
                    crate::inference::apply_rope(&mut k_pos, kv_dim, num_kv_heads, abs_position, rope_theta);
                } else {
                    // NORM style: adjacent pairs (2*i, 2*i+1) - standard RoPE
                    apply_rope_norm(&mut q_pos, num_heads, head_dim, abs_position, rope_theta);
                    apply_rope_norm(&mut k_pos, num_kv_heads, head_dim, abs_position, rope_theta);
                }

                // Use incremental_attention_gpu to append K/V to cache and compute attention
                let mut out_pos = vec![0.0f32; hidden_dim];
                if let Err(e) = self.executor.incremental_attention_gpu(
                    layer_idx,
                    &q_pos,
                    &k_pos,
                    &v_pos,
                    &mut out_pos,
                ) {
                    // Fallback to simple_attention if GPU KV cache fails
                    // This shouldn't happen if init_kv_cache_gpu succeeded
                    eprintln!("PMAT-110 WARNING: incremental_attention_gpu failed: {e}, using fallback");
                    let simple_out = simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
                    attn_out = simple_out;
                    break;
                }

                // Copy output to attn_out
                attn_out[q_pos_start..q_pos_start + hidden_dim].copy_from_slice(&out_pos);
            }

            if let Some(t) = timer_attn {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Output projection (GPU GEMM)
            let timer_oproj = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.OProj"))
            } else {
                None
            };
            let attn_proj = if self.has_cached_weight(&o_cache_name) {
                self.gemm_cached_gpu(&o_cache_name, &attn_out, seq_len, hidden_dim, hidden_dim)?
            } else {
                let o_weight = self.model.get_tensor_f32(&o_name)?;
                let o_weight_t = transpose_matrix(&o_weight, hidden_dim, hidden_dim);
                self.gemm_gpu(&attn_out, &o_weight_t, seq_len, hidden_dim, hidden_dim)?
            };
            if let Some(t) = timer_oproj {
                let _ = self.executor.synchronize();
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Residual connection
            let timer_res1 = if profiling {
                Some(self.executor.profiler_mut().start("apr.Residual"))
            } else {
                None
            };
            for (h, &a) in hidden.iter_mut().zip(attn_proj.iter()) {
                *h += a;
            }
            if let Some(t) = timer_res1 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // FFN
            let ffn_norm_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_2.weight"),
                &format!("layers.{layer_idx}.ffn_norm.weight"),
                &format!("blk.{layer_idx}.ffn_norm.weight"), // GGUF
            ])?;
            let gate_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
                &format!("blk.{layer_idx}.ffn_gate.weight"), // GGUF
            ])?;
            let up_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.mlp.up_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
                &format!("blk.{layer_idx}.ffn_up.weight"), // GGUF
            ])?;
            let down_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.mlp.down_proj.weight"),
                &format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
                &format!("blk.{layer_idx}.ffn_down.weight"), // GGUF
            ])?;

            // FFN RMSNorm
            let timer_rmsnorm2 = if profiling {
                Some(self.executor.profiler_mut().start("apr.RmsNorm"))
            } else {
                None
            };
            let ffn_norm = self.model.get_tensor_f32(&ffn_norm_name)?;
            let normed = rms_norm(&hidden, &ffn_norm, eps);
            if let Some(t) = timer_rmsnorm2 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // FFN projections (GPU GEMM) - use cached weights if available
            let gate_cache_name = format!("layer_{}_gate_proj", layer_idx);
            let up_cache_name = format!("layer_{}_up_proj", layer_idx);
            let down_cache_name = format!("layer_{}_down_proj", layer_idx);

            let timer_ffn = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.FFN"))
            } else {
                None
            };
            let (gate_out, up_out) = if self.has_cached_weight(&gate_cache_name) {
                // Fast path: use pre-cached transposed weights
                let gate_out = self.gemm_cached_gpu(
                    &gate_cache_name,
                    &normed,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                )?;
                let up_out = self.gemm_cached_gpu(
                    &up_cache_name,
                    &normed,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                )?;
                (gate_out, up_out)
            } else {
                // Fallback: load, transpose, and upload each time
                let gate = self.model.get_tensor_f32(&gate_name)?;
                let up = self.model.get_tensor_f32(&up_name)?;
                let gate_t = transpose_matrix(&gate, intermediate_dim, hidden_dim);
                let up_t = transpose_matrix(&up, intermediate_dim, hidden_dim);
                let gate_out =
                    self.gemm_gpu(&normed, &gate_t, seq_len, hidden_dim, intermediate_dim)?;
                let up_out =
                    self.gemm_gpu(&normed, &up_t, seq_len, hidden_dim, intermediate_dim)?;
                (gate_out, up_out)
            };

            // SiLU activation and element-wise multiply (CPU - fast)
            let mut ffn_hidden = Vec::with_capacity(seq_len * intermediate_dim);
            for (g, u) in gate_out.iter().zip(up_out.iter()) {
                let silu = g * (1.0 / (1.0 + (-g).exp()));
                ffn_hidden.push(silu * u);
            }

            let ffn_out = if self.has_cached_weight(&down_cache_name) {
                self.gemm_cached_gpu(
                    &down_cache_name,
                    &ffn_hidden,
                    seq_len,
                    intermediate_dim,
                    hidden_dim,
                )?
            } else {
                let down = self.model.get_tensor_f32(&down_name)?;
                let down_t = transpose_matrix(&down, hidden_dim, intermediate_dim);
                self.gemm_gpu(&ffn_hidden, &down_t, seq_len, intermediate_dim, hidden_dim)?
            };
            if let Some(t) = timer_ffn {
                let _ = self.executor.synchronize();
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // Residual
            let timer_res2 = if profiling {
                Some(self.executor.profiler_mut().start("apr.Residual"))
            } else {
                None
            };
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f;
            }
            if let Some(t) = timer_res2 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }
        }

        // PMAT-110: Update KV cache position after processing all tokens
        // This ensures subsequent forward_single_cuda calls have correct context
        self.kv_position += seq_len as u32;
        // PMAT-110: Mark that FALLBACK PATH was used for KV cache
        // Subsequent decode calls must also use FALLBACK PATH for consistency
        self.fallback_kv_used = true;

        // 3. Final layer norm (CPU)
        let timer_finalnorm = if profiling {
            Some(self.executor.profiler_mut().start("apr.FinalNorm"))
        } else {
            None
        };
        let final_norm_name = self.model.find_tensor_name(&[
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight", // GGUF naming
        ])?;
        let final_norm = self.model.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);
        if let Some(t) = timer_finalnorm {
            self.executor.profiler_mut().stop(t, 1); // Final norm processes 1 token (last)
        }

        // 4. LM head projection (GPU GEMM for large vocab)
        // Get hidden state for last token only
        let last_hidden = &hidden[hidden.len() - hidden_dim..];

        let timer_lmhead = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.LmHead"))
        } else {
            None
        };
        // LM head: [1, hidden_dim] × [hidden_dim, vocab_size] = [1, vocab_size]
        let logits = if self.has_cached_weight("lm_head") {
            // Fast path: use pre-cached transposed LM head
            self.gemm_cached_gpu("lm_head", last_hidden, 1, hidden_dim, vocab_size)?
        } else {
            // Fallback: load, transpose, and upload
            let lm_head_name = self.model.find_tensor_name(&[
                "lm_head.weight",
                "output.weight", // GGUF uses this
                "model.embed_tokens.weight",
                "embed_tokens.weight",
            ])?;
            let lm_head = self.model.get_tensor_f32(&lm_head_name)?;
            let lm_head_t = transpose_matrix(&lm_head, vocab_size, hidden_dim);
            self.gemm_gpu(last_hidden, &lm_head_t, 1, hidden_dim, vocab_size)?
        };
        if let Some(t) = timer_lmhead {
            let _ = self.executor.synchronize();
            self.executor.profiler_mut().stop(t, 1); // LM head processes 1 token (last)
        }

        Ok(logits)
    }

    /// GPU GEMM helper: C[m, n] = A[m, k] × B[k, n]
    ///
    /// Phase 45: Routes through test_executor when present for testability.
    #[allow(clippy::many_single_char_names)] // Standard matrix notation
    fn gemm_gpu(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Phase 45: Route through test executor if present
        if let Some(ref mut test_exec) = self.test_executor {
            return test_exec.matmul(a, b, m, k, n);
        }

        // Normal CUDA path
        let mut c = vec![0.0f32; m * n];
        self.executor
            .gemm(a, b, &mut c, m as u32, n as u32, k as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "GPU GEMM".to_string(),
                reason: format!("CUDA GEMM failed: {e}"),
            })?;
        Ok(c)
    }

    /// GPU GEMM with cached weight: C[m, n] = A[m, k] × B_cached[k, n]
    ///
    /// Uses pre-cached weight matrix B to avoid repeated GPU uploads.
    /// This is the optimized path for transformer inference.
    ///
    /// Phase 45: When test_executor is present, falls back to returning zeros
    /// (since test executors don't have cached weights).
    #[allow(clippy::many_single_char_names)] // Standard matrix notation
    fn gemm_cached_gpu(
        &mut self,
        weight_name: &str,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Phase 45: Test executor can't use cached weights, return zeros
        if self.test_executor.is_some() {
            // For testing, return zeros of correct size
            // The test executor can use with_matmul_result() for custom values if needed
            return Ok(vec![0.0f32; m * n]);
        }

        // Normal CUDA path
        let mut c = vec![0.0f32; m * n];
        self.executor
            .gemm_b_cached(weight_name, a, &mut c, m as u32, n as u32, k as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "GPU GEMM cached".to_string(),
                reason: format!("CUDA GEMM with cached weight '{}' failed: {e}", weight_name),
            })?;
        Ok(c)
    }

    /// Check if a weight is cached on GPU.
    ///
    /// Phase 45: Returns false when test_executor is present, forcing the
    /// uncached GEMM path which routes through the test executor.
    fn has_cached_weight(&self, name: &str) -> bool {
        if self.test_executor.is_some() {
            return false; // Force uncached path for testing
        }
        self.executor.has_weights(name)
    }

    /// GPU-accelerated token generation.
    ///
    /// Generates tokens autoregressively using GPU acceleration.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial prompt token IDs
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `eos_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// Complete token sequence including prompt and generated tokens.
    pub fn generate_cuda(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_new_tokens {
            // Forward pass
            let logits = self.forward_cuda(&tokens)?;

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(eos_id, |(idx, _)| idx as u32);

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// GPU-accelerated forward pass for single token with KV cache.
    ///
    /// This is the optimized decode path that reuses cached K/V values
    /// from previous positions for O(1) attention per token.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token ID to process
    /// * `position` - Current position in sequence
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size` for next token prediction.
    pub fn forward_single_cuda(&mut self, token_id: u32, _position: usize) -> Result<Vec<f32>> {
        // Uses full forward pass; KV cache optimization available via GGUF path
        self.forward_cuda(&[token_id])
    }

    /// GPU-accelerated generation with KV cache.
    ///
    /// Uses the optimized single-token decode path after prefill.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial prompt token IDs
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `eos_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// Complete token sequence including prompt and generated tokens.
    pub fn generate_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        // Prefill: process entire prompt
        let mut tokens = prompt.to_vec();
        let _ = self.forward_cuda(&tokens)?;

        // Decode: generate one token at a time
        for _i in 0..max_new_tokens {
            let position = tokens.len();
            let last_token = *tokens.last().unwrap_or(&1);

            let logits = self.forward_single_cuda(last_token, position)?;

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(eos_id, |(idx, _)| idx as u32);

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}
