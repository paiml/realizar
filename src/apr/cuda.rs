//! CUDA-accelerated APR model inference (PMAT-802)
//!
//! Extracted from apr/mod.rs - GPU acceleration for APR v2 models.
//!
//! ## Contents
//! - `AprV2ModelCuda` - CUDA wrapper for APR models (2x Ollama target)

use super::{
    apply_rope_norm, dtype_to_ggml_qtype, rms_norm, simple_attention, transpose_matrix, AprV2Model,
};
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
    /// GH-201: Streaming mode (true = layer-by-layer, false = full cache)
    ///
    /// In streaming mode, only one layer's weights are on GPU at a time.
    /// This reduces VRAM usage from ~6GB to ~1.5GB for 1.5B models.
    streaming_mode: bool,
    /// GH-201: Currently cached layer index in streaming mode
    ///
    /// When streaming, this tracks which layer's weights are currently on GPU.
    /// None means no layer weights are cached yet.
    cached_streaming_layer: Option<usize>,
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
        use crate::cuda::{check_vram_sufficient, CudaExecutor, StreamingConfig, StreamingMode};

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
        let vocab_size = model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };

        // GH-201: Check VRAM and select streaming mode
        let streaming_config = StreamingConfig {
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            max_seq_len,
        };

        let (free_vram, total_vram) = memory_info;
        let streaming_mode = match check_vram_sufficient(free_vram, total_vram, &streaming_config) {
            Ok(StreamingMode::FullCache) => {
                eprintln!(
                    "[AprV2ModelCuda] VRAM sufficient ({} MB free), using full cache mode",
                    free_vram / (1024 * 1024)
                );
                false
            }
            Ok(StreamingMode::LayerStreaming) => {
                eprintln!(
                    "[AprV2ModelCuda] GH-201: Limited VRAM ({} MB free), using layer streaming mode",
                    free_vram / (1024 * 1024)
                );
                true
            }
            Err(e) => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "VRAM check".to_string(),
                    reason: e,
                });
            }
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
        // BUG-2 FIX: Infer rope_type from architecture when not explicitly set
        let rope_type = model.metadata.rope_type.unwrap_or_else(|| {
            // Infer from architecture name (matches llama.cpp neox-style architectures)
            let arch = model.metadata.model_type.as_deref().unwrap_or("");
            let arch_lower = arch.to_lowercase();
            let is_neox = arch_lower.contains("qwen")
                || arch_lower.contains("phi")
                || arch_lower.contains("gemma")
                || arch_lower.contains("falcon")
                || arch_lower.contains("starcoder")
                || arch_lower.contains("gptneox")
                || arch_lower.contains("bert")
                || arch_lower.contains("stablelm");
            if is_neox { 2 } else { 0 }
        });
        let rms_norm_eps = model.metadata.rms_norm_eps.unwrap_or(1e-6);

        // PMAT-114: Trace model configuration for precision debugging
        if std::env::var("APR_TRACE_CONFIG").is_ok() {
            eprintln!(
                "[APR CONFIG] rope_theta={} (raw={:?})",
                rope_theta, model.metadata.rope_theta
            );
            eprintln!(
                "[APR CONFIG] rope_type={} (raw={:?})",
                rope_type, model.metadata.rope_type
            );
            eprintln!(
                "[APR CONFIG] rms_norm_eps={} (raw={:?})",
                rms_norm_eps, model.metadata.rms_norm_eps
            );
        }
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
            kv_position: 0,          // Start at position 0
            fallback_kv_used: false, // PMAT-110: No fallback KV yet
            test_executor: None,     // Phase 45: No test executor by default
            streaming_mode,          // GH-201: Set based on VRAM check
            cached_streaming_layer: None, // GH-201: No layer cached yet
        };

        // GH-201: Choose weight caching strategy based on streaming mode
        if streaming_mode {
            // Layer streaming: only cache LM head and norms, not per-layer weights
            apr_cuda.pre_cache_weights_streaming()?;
        } else {
            // Full cache: pre-cache all transposed weights on GPU for 2x performance
            apr_cuda.pre_cache_weights()?;
        }

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

    /// Get reference to the inner APR model (PMAT-APR-CUDA-001)
    #[must_use]
    pub fn model(&self) -> &AprV2Model {
        &self.model
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
        let kv_dim = num_kv_heads * head_dim;

        if hidden_dim == 0 || num_layers == 0 {
            return Ok(()); // Non-transformer model, nothing to cache
        }

        let mut total_bytes = 0usize;
        let mut quantized_count = 0usize;

        // Helper to upload a weight tensor (quantized or F32)
        // Uses GGUF-style cache names for compatibility with build_indexed_weights()
        // PMAT-113: Now caches F32 weights for GPU GEMM (was causing APR CUDA hang)
        let upload_weight = |executor: &mut crate::cuda::CudaExecutor,
                             model: &AprV2Model,
                             src_name: &str,
                             cache_name: &str|
         -> (usize, bool) {
            // Returns (bytes_uploaded, is_quantized)
            if let Some(entry) = model.get_tensor(src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized: upload raw bytes to quantized_weight_cache
                    if let Ok(bytes) = model.get_tensor_bytes(src_name) {
                        let size = executor
                            .load_quantized_weights_with_type(cache_name, bytes, qtype)
                            .unwrap_or(0);
                        (size, true)
                    } else {
                        (0, false)
                    }
                } else {
                    // PMAT-113: F32/F16 - cache on GPU for GEMM path
                    // PMAT-222: Transpose 2D F32 weights from [n, k] to [k, n] for gemm_b_cached
                    // HF convention stores weights as [out_dim, in_dim] but GEMM needs B[k, n]
                    if let Ok(weights) = model.get_tensor_f32(src_name) {
                        let final_weights = if entry.shape.len() == 2 {
                            let rows = entry.shape[0]; // out_dim (n)
                            let cols = entry.shape[1]; // in_dim (k)
                            let mut transposed = vec![0.0f32; weights.len()];
                            for i in 0..rows {
                                for j in 0..cols {
                                    transposed[j * rows + i] = weights[i * cols + j];
                                }
                            }
                            transposed
                        } else {
                            weights
                        };
                        let size = executor
                            .load_weights(cache_name, &final_weights)
                            .unwrap_or(0);
                        (size, false)
                    } else {
                        (0, false)
                    }
                }
            } else {
                (0, false)
            }
        };

        // Track F32 weight count for fallback path
        let mut f32_weight_count = 0usize;

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
                    let (bytes, is_quantized) =
                        upload_weight(&mut self.executor, &self.model, &src_name, &cache_name);
                    if bytes > 0 {
                        total_bytes += bytes;
                        if is_quantized {
                            quantized_count += 1;
                        } else {
                            f32_weight_count += 1;
                        }
                    }
                }
            }

            // PMAT-113: Cache fused QKV from APR import (PMAT-101)
            // APR models from HuggingFace have Q/K/V fused into qkv_proj.weight
            // Unfuse and cache as separate Q/K/V with names the forward path expects
            // NOTE: P1 quality issue exists (SATD-WARNING in generate_cuda_with_cache)
            // The APR import has corrupt tensor layouts - this caching doesn't fix that
            let fused_qkv_patterns = vec![format!(
                "model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            )];
            let fused_patterns_ref: Vec<&str> =
                fused_qkv_patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&fused_patterns_ref) {
                // Load and unfuse QKV for F32 models
                if let Ok(qkv_weight) = self.model.get_tensor_f32(&src_name) {
                    // Unfuse: Q is first hidden_dim rows, K is next kv_dim, V is last kv_dim
                    let q_size = hidden_dim * hidden_dim;
                    let k_size = kv_dim * hidden_dim;
                    let v_size = kv_dim * hidden_dim;

                    if qkv_weight.len() >= q_size + k_size + v_size {
                        // Cache unfused Q/K/V with forward path naming convention
                        let q_weight: Vec<f32> = qkv_weight[0..q_size].to_vec();
                        let k_weight: Vec<f32> = qkv_weight[q_size..q_size + k_size].to_vec();
                        let v_weight: Vec<f32> =
                            qkv_weight[q_size + k_size..q_size + k_size + v_size].to_vec();

                        // PMAT-114: Trace K weight for layer 0 to debug 100x difference
                        if layer_idx == 0 && std::env::var("APR_TRACE_LAYERS").is_ok() {
                            let k_sum: f32 = k_weight.iter().sum();
                            let k_mean = k_sum / k_weight.len() as f32;
                            let k_min = k_weight.iter().cloned().fold(f32::INFINITY, f32::min);
                            let k_max = k_weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            eprintln!("[PMAT-114] L0 K weight (pre-transpose): mean={:.6}, min={:.6}, max={:.6}, len={}",
                                k_mean, k_min, k_max, k_weight.len());
                            eprintln!(
                                "[PMAT-114] L0 K weight first10={:?}",
                                &k_weight[..10.min(k_weight.len())]
                            );
                        }

                        // Transpose for GPU GEMM (row-major to column-major)
                        let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                        let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                        let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);

                        // PMAT-114: Trace K weight after transpose
                        if layer_idx == 0 && std::env::var("APR_TRACE_LAYERS").is_ok() {
                            let k_sum: f32 = k_weight_t.iter().sum();
                            let k_mean = k_sum / k_weight_t.len() as f32;
                            eprintln!(
                                "[PMAT-114] L0 K weight (post-transpose): mean={:.6}, len={}",
                                k_mean,
                                k_weight_t.len()
                            );
                            eprintln!(
                                "[PMAT-114] L0 K weight_t first10={:?}",
                                &k_weight_t[..10.min(k_weight_t.len())]
                            );
                        }

                        // Cache with GGUF-style naming to match forward path (PMAT-805)
                        let q_cache_name = format!("blk.{layer_idx}.attn_q.weight");
                        let k_cache_name = format!("blk.{layer_idx}.attn_k.weight");
                        let v_cache_name = format!("blk.{layer_idx}.attn_v.weight");

                        if let Ok(bytes) = self.executor.load_weights(&q_cache_name, &q_weight_t) {
                            total_bytes += bytes;
                            f32_weight_count += 1;
                        }
                        if let Ok(bytes) = self.executor.load_weights(&k_cache_name, &k_weight_t) {
                            total_bytes += bytes;
                            f32_weight_count += 1;
                        }
                        if let Ok(bytes) = self.executor.load_weights(&v_cache_name, &v_weight_t) {
                            total_bytes += bytes;
                            f32_weight_count += 1;
                        }
                    }
                }
            }

            // PMAT-114: Cache fused QKV bias if present (Qwen2 has attention bias)
            let fused_qkv_bias_patterns =
                vec![format!("model.layers.{layer_idx}.self_attn.qkv_proj.bias")];
            let fused_bias_patterns_ref: Vec<&str> =
                fused_qkv_bias_patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&fused_bias_patterns_ref) {
                if let Ok(qkv_bias) = self.model.get_tensor_f32(&src_name) {
                    // Unfuse: Q_bias is first hidden_dim, K_bias is next kv_dim, V_bias is last kv_dim
                    let q_bias_size = hidden_dim;
                    let k_bias_size = kv_dim;
                    let v_bias_size = kv_dim;

                    if qkv_bias.len() >= q_bias_size + k_bias_size + v_bias_size {
                        let q_bias: Vec<f32> = qkv_bias[0..q_bias_size].to_vec();
                        let k_bias: Vec<f32> =
                            qkv_bias[q_bias_size..q_bias_size + k_bias_size].to_vec();
                        let v_bias: Vec<f32> = qkv_bias
                            [q_bias_size + k_bias_size..q_bias_size + k_bias_size + v_bias_size]
                            .to_vec();

                        // PMAT-114: Trace K bias for layer 0 to verify import
                        if layer_idx == 0 && std::env::var("APR_TRACE_LAYERS").is_ok() {
                            let k_bias_sum: f32 = k_bias.iter().sum();
                            let k_bias_mean = k_bias_sum / k_bias.len() as f32;
                            eprintln!(
                                "[PMAT-114] L0 K bias loaded: mean={:.6}, len={}",
                                k_bias_mean,
                                k_bias.len()
                            );
                        }

                        // Cache with forward path naming: layer_{}_q_bias, etc.
                        let q_bias_cache_name = format!("layer_{layer_idx}_q_bias");
                        let k_bias_cache_name = format!("layer_{layer_idx}_k_bias");
                        let v_bias_cache_name = format!("layer_{layer_idx}_v_bias");

                        if let Ok(bytes) = self.executor.load_weights(&q_bias_cache_name, &q_bias) {
                            total_bytes += bytes;
                        }
                        if let Ok(bytes) = self.executor.load_weights(&k_bias_cache_name, &k_bias) {
                            total_bytes += bytes;
                        }
                        if let Ok(bytes) = self.executor.load_weights(&v_bias_cache_name, &v_bias) {
                            total_bytes += bytes;
                        }
                    }
                }
            }

            // PMAT-113: Cache O projection with forward path naming
            let o_patterns = vec![
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                format!("blk.{layer_idx}.attn_output.weight"),
            ];
            let o_patterns_ref: Vec<&str> = o_patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&o_patterns_ref) {
                if let Ok(o_weight) = self.model.get_tensor_f32(&src_name) {
                    let o_weight_t = transpose_matrix(&o_weight, hidden_dim, hidden_dim);
                    let o_cache_name = format!("layer_{layer_idx}_o_proj");
                    if let Ok(bytes) = self.executor.load_weights(&o_cache_name, &o_weight_t) {
                        total_bytes += bytes;
                        f32_weight_count += 1;
                    }
                }
            }

            // PMAT-113: Cache FFN weights with forward path naming
            let ffn_patterns = [
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("blk.{layer_idx}.ffn_gate.weight"),
                    ],
                    format!("layer_{layer_idx}_gate_proj"),
                    intermediate_dim,
                    hidden_dim,
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("blk.{layer_idx}.ffn_up.weight"),
                    ],
                    format!("layer_{layer_idx}_up_proj"),
                    intermediate_dim,
                    hidden_dim,
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("blk.{layer_idx}.ffn_down.weight"),
                    ],
                    format!("layer_{layer_idx}_down_proj"),
                    hidden_dim,
                    intermediate_dim,
                ),
            ];

            for (patterns, cache_name, out_dim, in_dim) in ffn_patterns {
                let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
                if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                    if let Ok(weight) = self.model.get_tensor_f32(&src_name) {
                        let weight_t = transpose_matrix(&weight, out_dim, in_dim);
                        if let Ok(bytes) = self.executor.load_weights(&cache_name, &weight_t) {
                            total_bytes += bytes;
                            f32_weight_count += 1;
                        }
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

        // PMAT-113: Log both quantized and F32 weight counts
        eprintln!(
            "[AprV2ModelCuda] Pre-cached {} MB of weights on GPU ({} layers, {} quantized, {} F32 tensors)",
            total_bytes / (1024 * 1024),
            num_layers,
            quantized_count,
            f32_weight_count
        );

        Ok(())
    }

    /// GH-201: Pre-cache only essential weights in streaming mode.
    ///
    /// In streaming mode, we only cache:
    /// - LM head (required for every token)
    /// - Output norm gamma (required for every token)
    ///
    /// Per-layer weights are loaded on-demand via `ensure_layer_weights_loaded()`.
    /// This reduces VRAM usage from ~6GB to ~1.2GB for 1.5B models.
    fn pre_cache_weights_streaming(&mut self) -> Result<()> {
        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let _num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);

        if hidden_dim == 0 {
            return Ok(()); // Non-transformer model
        }

        let mut total_bytes = 0usize;

        // Cache output norm (always needed)
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

        // Cache LM head (always needed - may be quantized or F32)
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
                        }
                    }
                } else if let Ok(w) = self.model.get_tensor_f32(&src_name) {
                    // F32 LM head
                    // SAFETY: f32 slice to u8 view - valid because f32 has no padding
                    let w_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            w.as_ptr().cast::<u8>(),
                            w.len() * std::mem::size_of::<f32>(),
                        )
                    };
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

        let lm_head_mb = vocab_size * hidden_dim * 4 / (1024 * 1024);
        eprintln!(
            "[AprV2ModelCuda] GH-201: Streaming mode - cached {} MB (LM head ~{} MB, norms)",
            total_bytes / (1024 * 1024),
            lm_head_mb
        );
        eprintln!("[AprV2ModelCuda] GH-201: Layer weights will be streamed on-demand");

        Ok(())
    }

    /// GH-201: Ensure a specific layer's weights are loaded on GPU.
    ///
    /// In streaming mode, this uploads the layer's weights if not already cached.
    /// The previously cached layer's weights are replaced.
    ///
    /// In full cache mode, this is a no-op (all weights pre-cached).
    fn ensure_layer_weights_loaded(&mut self, layer_idx: usize) -> Result<()> {
        if !self.streaming_mode {
            return Ok(()); // Full cache mode - weights already on GPU
        }

        // Check if this layer is already cached
        if self.cached_streaming_layer == Some(layer_idx) {
            return Ok(());
        }

        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let _intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 0 };
        let kv_dim = num_kv_heads * head_dim;

        let prefix = format!("blk.{layer_idx}");
        let mut total_bytes = 0usize;

        // Clear previous layer's weights from GPU cache
        // (The executor will reuse the memory)

        // Helper to upload a weight tensor
        let upload_weight = |executor: &mut crate::cuda::CudaExecutor,
                             model: &AprV2Model,
                             src_name: &str,
                             cache_name: &str|
         -> usize {
            if let Some(entry) = model.get_tensor(src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized weight
                    if let Ok(bytes) = model.get_tensor_bytes(src_name) {
                        executor
                            .load_quantized_weights_with_type(cache_name, bytes, qtype)
                            .unwrap_or(0)
                    } else {
                        0
                    }
                } else if let Ok(weights) = model.get_tensor_f32(src_name) {
                    // F32 weight - transpose for GPU GEMM
                    let final_weights = if entry.shape.len() == 2 {
                        let rows = entry.shape[0];
                        let cols = entry.shape[1];
                        let mut transposed = vec![0.0f32; weights.len()];
                        for i in 0..rows {
                            for j in 0..cols {
                                transposed[j * rows + i] = weights[i * cols + j];
                            }
                        }
                        transposed
                    } else {
                        weights
                    };
                    executor.load_weights(cache_name, &final_weights).unwrap_or(0)
                } else {
                    0
                }
            } else {
                0
            }
        };

        // Upload attention weights
        let weight_mappings = [
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                    format!("blk.{layer_idx}.attn_q.weight"),
                ],
                "attn_q.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                    format!("blk.{layer_idx}.attn_k.weight"),
                ],
                "attn_k.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                    format!("blk.{layer_idx}.attn_v.weight"),
                ],
                "attn_v.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                    format!("blk.{layer_idx}.attn_output.weight"),
                ],
                "attn_output.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                    format!("blk.{layer_idx}.ffn_gate.weight"),
                ],
                "ffn_gate.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                    format!("blk.{layer_idx}.ffn_up.weight"),
                ],
                "ffn_up.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                    format!("blk.{layer_idx}.ffn_down.weight"),
                ],
                "ffn_down.weight",
            ),
        ];

        for (patterns, suffix) in weight_mappings {
            let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                let cache_name = format!("{prefix}.{suffix}");
                total_bytes += upload_weight(&mut self.executor, &self.model, &src_name, &cache_name);
            }
        }

        // Handle fused QKV if present
        let fused_qkv_patterns = vec![format!(
            "model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        )];
        let fused_patterns_ref: Vec<&str> = fused_qkv_patterns.iter().map(String::as_str).collect();
        if let Ok(src_name) = self.model.find_tensor_name(&fused_patterns_ref) {
            if let Ok(qkv_weight) = self.model.get_tensor_f32(&src_name) {
                let q_size = hidden_dim * hidden_dim;
                let k_size = kv_dim * hidden_dim;
                let v_size = kv_dim * hidden_dim;

                if qkv_weight.len() >= q_size + k_size + v_size {
                    let q_weight: Vec<f32> = qkv_weight[0..q_size].to_vec();
                    let k_weight: Vec<f32> = qkv_weight[q_size..q_size + k_size].to_vec();
                    let v_weight: Vec<f32> =
                        qkv_weight[q_size + k_size..q_size + k_size + v_size].to_vec();

                    // Transpose for GPU GEMM
                    let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                    let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                    let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);

                    let q_cache_name = format!("blk.{layer_idx}.attn_q.weight");
                    let k_cache_name = format!("blk.{layer_idx}.attn_k.weight");
                    let v_cache_name = format!("blk.{layer_idx}.attn_v.weight");

                    total_bytes += self.executor.load_weights(&q_cache_name, &q_weight_t).unwrap_or(0);
                    total_bytes += self.executor.load_weights(&k_cache_name, &k_weight_t).unwrap_or(0);
                    total_bytes += self.executor.load_weights(&v_cache_name, &v_weight_t).unwrap_or(0);
                }
            }
        }

        // Upload RMSNorm gamma weights
        let norm_mappings = [
            (
                vec![
                    format!("model.layers.{layer_idx}.input_layernorm.weight"),
                    format!("blk.{layer_idx}.attn_norm.weight"),
                ],
                "attn_norm.gamma",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
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
                    total_bytes += self.executor.cache_rmsnorm_gamma(&cache_name, &gamma).unwrap_or(0);
                }
            }
        }

        self.cached_streaming_layer = Some(layer_idx);

        // Only log for first few layers to avoid spam
        if layer_idx < 3 {
            eprintln!(
                "[AprV2ModelCuda] GH-201: Streamed layer {} weights ({} KB)",
                layer_idx,
                total_bytes / 1024
            );
        }

        Ok(())
    }

    /// GH-201: Check if model is in streaming mode.
    #[must_use]
    pub fn is_streaming_mode(&self) -> bool {
        self.streaming_mode
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
        // GH-201: Skip fast path in streaming mode (layer weights not pre-cached)
        if self.test_executor.is_none() && self.executor.has_indexed_weights() && !self.streaming_mode {
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
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
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
        // GH-201: Skip fast path in streaming mode (layer weights not pre-cached)
        // =========================================================================
        if self.test_executor.is_none()
            && self.executor.has_indexed_weights()
            && seq_len == 1
            && !self.fallback_kv_used
            && !self.streaming_mode
        {
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

        // PMAT-114: Layer tracing for Five-Whys analysis
        let trace_layers = std::env::var("APR_TRACE_LAYERS").is_ok();
        if trace_layers {
            // PMAT-114: Trace token IDs being processed
            eprintln!(
                "[PMAT-114] Input tokens ({} total): {:?}",
                token_ids.len(),
                &token_ids[..token_ids.len().min(20)]
            );
            if let Some(&last_token) = token_ids.last() {
                eprintln!("[PMAT-114] Last token ID: {}", last_token);
            }

            let last_hidden = &hidden[hidden.len() - hidden_dim..];
            let sum: f32 = last_hidden.iter().sum();
            let mean = sum / hidden_dim as f32;
            let min = last_hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = last_hidden
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[PMAT-114] After embed: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                mean,
                min,
                max,
                &last_hidden[..5.min(hidden_dim)]
            );
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

            // PMAT-APR-CUDA-002: Check for fused QKV first (from APR import)
            // APR import fuses Q/K/V into qkv_proj.weight for AprTransformer compatibility
            let fused_qkv_name = self.model.find_tensor_name(&[&format!(
                "model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            )]);
            let has_fused_qkv = fused_qkv_name.is_ok();

            // Only look for separate Q/K/V if fused is not available
            let (q_name, k_name, v_name) = if !has_fused_qkv {
                let q = self.model.find_tensor_name(&[
                    &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                    &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                    &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                    &format!("layers.{layer_idx}.attention.wq.weight"),
                    &format!("blk.{layer_idx}.attn_q.weight"), // GGUF
                ])?;
                let k = self.model.find_tensor_name(&[
                    &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                    &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                    &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                    &format!("layers.{layer_idx}.attention.wk.weight"),
                    &format!("blk.{layer_idx}.attn_k.weight"), // GGUF
                ])?;
                let v = self.model.find_tensor_name(&[
                    &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                    &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                    &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                    &format!("layers.{layer_idx}.attention.wv.weight"),
                    &format!("blk.{layer_idx}.attn_v.weight"), // GGUF
                ])?;
                (Some(q), Some(k), Some(v))
            } else {
                (None, None, None)
            };

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

            // PMAT-114: Detailed layer 0 tracing
            if trace_layers && layer_idx == 0 {
                let last = &normed[normed.len() - hidden_dim..];
                let sum: f32 = last.iter().sum();
                let mean = sum / hidden_dim as f32;
                eprintln!(
                    "[PMAT-114] L0 after RMSNorm: mean={:.6}, first5={:?}",
                    mean,
                    &last[..5]
                );
            }

            // Q, K, V projections (GPU GEMM for 2x speedup)
            // Use cached weights if available (avoids repeated transpose + upload)
            // PMAT-805 FIX: Use GGUF-style cache names to match pre_cache_weights()
            let q_cache_name = format!("blk.{}.attn_q.weight", layer_idx);
            let k_cache_name = format!("blk.{}.attn_k.weight", layer_idx);
            let v_cache_name = format!("blk.{}.attn_v.weight", layer_idx);
            let o_cache_name = format!("blk.{}.attn_output.weight", layer_idx);

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
            } else if has_fused_qkv {
                // PMAT-APR-CUDA-002: Handle fused QKV from APR import
                // qkv_proj.weight is [qkv_dim, hidden_dim] where qkv_dim = hidden_dim + 2*kv_dim
                let fused_name = fused_qkv_name.expect("checked above");
                let qkv_weight = self.model.get_tensor_f32(&fused_name)?;

                // Unfuse into Q, K, V: Q is first hidden_dim rows, K is next kv_dim, V is last kv_dim
                let q_size = hidden_dim * hidden_dim;
                let k_size = kv_dim * hidden_dim;
                let v_size = kv_dim * hidden_dim;

                if qkv_weight.len() < q_size + k_size + v_size {
                    return Err(RealizarError::InvalidShape {
                        reason: format!(
                            "Fused QKV weight too small: {} < {} (expected Q={}, K={}, V={})",
                            qkv_weight.len(),
                            q_size + k_size + v_size,
                            q_size,
                            k_size,
                            v_size
                        ),
                    });
                }

                let q_weight: Vec<f32> = qkv_weight[0..q_size].to_vec();
                let k_weight: Vec<f32> = qkv_weight[q_size..q_size + k_size].to_vec();
                let v_weight: Vec<f32> =
                    qkv_weight[q_size + k_size..q_size + k_size + v_size].to_vec();

                let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);
                let q = self.gemm_gpu(&normed, &q_weight_t, seq_len, hidden_dim, hidden_dim)?;
                let k = self.gemm_gpu(&normed, &k_weight_t, seq_len, hidden_dim, kv_dim)?;
                let v = self.gemm_gpu(&normed, &v_weight_t, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            } else {
                // Fallback: load separate Q/K/V, transpose, and upload weights each time
                let q_weight = self
                    .model
                    .get_tensor_f32(q_name.as_ref().expect("checked above"))?;
                let k_weight = self
                    .model
                    .get_tensor_f32(k_name.as_ref().expect("checked above"))?;
                let v_weight = self
                    .model
                    .get_tensor_f32(v_name.as_ref().expect("checked above"))?;
                let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);
                let q = self.gemm_gpu(&normed, &q_weight_t, seq_len, hidden_dim, hidden_dim)?;
                let k = self.gemm_gpu(&normed, &k_weight_t, seq_len, hidden_dim, kv_dim)?;
                let v = self.gemm_gpu(&normed, &v_weight_t, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            };

            // PMAT-114: Apply QKV bias if present (Qwen2 has attention bias)
            // Handle both fused (qkv_proj.bias) and separate (q_proj.bias, k_proj.bias, v_proj.bias) formats
            let fused_bias_name = format!("model.layers.{layer_idx}.self_attn.qkv_proj.bias");
            let (mut q, mut k, mut v) = (q, k, v);
            let mut bias_applied = false;

            // Try fused bias first (HuggingFace import with fused QKV)
            if let Ok(qkv_bias) = self.model.get_tensor_f32(&fused_bias_name) {
                // Unfuse: Q_bias is first hidden_dim, K_bias is next kv_dim, V_bias is last kv_dim
                let q_bias_size = hidden_dim;
                let k_bias_size = kv_dim;
                let v_bias_size = kv_dim;

                if qkv_bias.len() >= q_bias_size + k_bias_size + v_bias_size {
                    let q_bias = &qkv_bias[0..q_bias_size];
                    let k_bias = &qkv_bias[q_bias_size..q_bias_size + k_bias_size];
                    let v_bias = &qkv_bias
                        [q_bias_size + k_bias_size..q_bias_size + k_bias_size + v_bias_size];

                    // Add bias to each position in Q, K, V
                    for pos in 0..seq_len {
                        let q_start = pos * hidden_dim;
                        let k_start = pos * kv_dim;
                        let v_start = pos * kv_dim;

                        for (i, bias_val) in q_bias.iter().enumerate() {
                            q[q_start + i] += bias_val;
                        }
                        for (i, bias_val) in k_bias.iter().enumerate() {
                            k[k_start + i] += bias_val;
                        }
                        for (i, bias_val) in v_bias.iter().enumerate() {
                            v[v_start + i] += bias_val;
                        }
                    }

                    bias_applied = true;
                    // PMAT-114: Trace bias application for layer 0
                    if trace_layers && layer_idx == 0 {
                        let k_bias_mean: f32 = k_bias.iter().sum::<f32>() / k_bias.len() as f32;
                        eprintln!(
                            "[PMAT-114-APR] L0 has_qkv_bias=true (fused), K bias mean={:.6}",
                            k_bias_mean
                        );
                    }
                }
            }

            // PMAT-113 FIX: Try separate Q/K/V biases (APR converted from GGUF)
            // GGUF models have separate q_proj.bias, k_proj.bias, v_proj.bias
            if !bias_applied {
                let q_bias_name = format!("model.layers.{layer_idx}.self_attn.q_proj.bias");
                let k_bias_name = format!("model.layers.{layer_idx}.self_attn.k_proj.bias");
                let v_bias_name = format!("model.layers.{layer_idx}.self_attn.v_proj.bias");

                // Try to load separate biases
                let q_bias = self.model.get_tensor_f32(&q_bias_name).ok();
                let k_bias = self.model.get_tensor_f32(&k_bias_name).ok();
                let v_bias = self.model.get_tensor_f32(&v_bias_name).ok();

                // Apply Q bias
                if let Some(ref bias) = q_bias {
                    if bias.len() == hidden_dim {
                        for pos in 0..seq_len {
                            let start = pos * hidden_dim;
                            for (i, bias_val) in bias.iter().enumerate() {
                                q[start + i] += bias_val;
                            }
                        }
                    }
                }

                // Apply K bias
                if let Some(ref bias) = k_bias {
                    if bias.len() == kv_dim {
                        for pos in 0..seq_len {
                            let start = pos * kv_dim;
                            for (i, bias_val) in bias.iter().enumerate() {
                                k[start + i] += bias_val;
                            }
                        }
                        bias_applied = true;
                    }
                }

                // Apply V bias
                if let Some(ref bias) = v_bias {
                    if bias.len() == kv_dim {
                        for pos in 0..seq_len {
                            let start = pos * kv_dim;
                            for (i, bias_val) in bias.iter().enumerate() {
                                v[start + i] += bias_val;
                            }
                        }
                    }
                }

                // PMAT-114: Trace separate bias application for layer 0
                if trace_layers && layer_idx == 0 && bias_applied {
                    if let Some(ref kb) = k_bias {
                        let k_bias_mean: f32 = kb.iter().sum::<f32>() / kb.len() as f32;
                        eprintln!(
                            "[PMAT-114-APR] L0 has_qkv_bias=true (separate), K bias mean={:.6}",
                            k_bias_mean
                        );
                    }
                }
            }

            if let Some(t) = timer_qkv {
                let _ = self.executor.synchronize();
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // PMAT-114: Trace QKV for layer 0
            if trace_layers && layer_idx == 0 {
                let q_last = &q[q.len() - hidden_dim..];
                let k_last = &k[k.len() - kv_dim..];
                let v_last = &v[v.len() - kv_dim..];
                let q_mean: f32 = q_last.iter().sum::<f32>() / hidden_dim as f32;
                let k_mean: f32 = k_last.iter().sum::<f32>() / kv_dim as f32;
                let v_mean: f32 = v_last.iter().sum::<f32>() / kv_dim as f32;
                eprintln!(
                    "[PMAT-114] L0 after QKV: Q mean={:.6}, K mean={:.6}, V mean={:.6}",
                    q_mean, k_mean, v_mean
                );
                eprintln!("[PMAT-114] L0 Q first5={:?}", &q_last[..5]);
                eprintln!(
                    "[PMAT-114] L0 shapes: q={}, k={}, v={}, hidden_dim={}, kv_dim={}",
                    q.len(),
                    k.len(),
                    v.len(),
                    hidden_dim,
                    kv_dim
                );
                eprintln!(
                    "[PMAT-114] L0 has_fused_qkv={}, has_cached_q={}",
                    has_fused_qkv,
                    self.has_cached_weight(&q_cache_name)
                );
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
                    crate::inference::apply_rope(
                        &mut q_pos,
                        hidden_dim,
                        num_heads,
                        abs_position,
                        rope_theta,
                    );
                    crate::inference::apply_rope(
                        &mut k_pos,
                        kv_dim,
                        num_kv_heads,
                        abs_position,
                        rope_theta,
                    );
                } else {
                    // NORM style: adjacent pairs (2*i, 2*i+1) - standard RoPE
                    apply_rope_norm(&mut q_pos, num_heads, head_dim, abs_position, rope_theta, 0);
                    apply_rope_norm(&mut k_pos, num_kv_heads, head_dim, abs_position, rope_theta, 0);
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
                    eprintln!(
                        "PMAT-110 WARNING: incremental_attention_gpu failed: {e}, using fallback"
                    );
                    let simple_out =
                        simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
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
            // PMAT-805 FIX: Use GGUF-style cache names to match pre_cache_weights()
            let gate_cache_name = format!("blk.{}.ffn_gate.weight", layer_idx);
            let up_cache_name = format!("blk.{}.ffn_up.weight", layer_idx);
            let down_cache_name = format!("blk.{}.ffn_down.weight", layer_idx);

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

            // PMAT-114: Layer tracing for Five-Whys analysis
            if trace_layers && (layer_idx < 2 || layer_idx == num_layers - 1) {
                let last_hidden = &hidden[hidden.len() - hidden_dim..];
                let sum: f32 = last_hidden.iter().sum();
                let mean = sum / hidden_dim as f32;
                let min = last_hidden.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = last_hidden
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[PMAT-114] After layer {}: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                    layer_idx,
                    mean,
                    min,
                    max,
                    &last_hidden[..5.min(hidden_dim)]
                );
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
            // Fallback: load, transpose (if needed), and upload
            // BUG-APR-001: Add token_embd.weight for GGUF weight tying
            let lm_head_name = self.model.find_tensor_name(&[
                "lm_head.weight",
                "output.weight", // GGUF uses this
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "token_embd.weight", // GGUF tied embeddings
            ])?;
            let lm_head = self.model.get_tensor_f32(&lm_head_name)?;

            // BUG-APR-001-FIX: Detect weight tying and handle transposed layout
            // GGUF token_embd.weight is stored as [hidden_dim, vocab_size] - already correct for GEMM
            // Regular lm_head.weight is stored as [vocab_size, hidden_dim] - needs transpose
            let is_tied_embedding = lm_head_name == "token_embd.weight"
                || lm_head_name.ends_with("embed_tokens.weight");

            let lm_head_for_gemm = if is_tied_embedding && lm_head.len() == hidden_dim * vocab_size
            {
                // Tied embedding: already [hidden_dim, vocab_size], use as-is
                lm_head.clone()
            } else {
                // Regular lm_head: [vocab_size, hidden_dim], need transpose to [hidden_dim, vocab_size]
                transpose_matrix(&lm_head, vocab_size, hidden_dim)
            };
            self.gemm_gpu(last_hidden, &lm_head_for_gemm, 1, hidden_dim, vocab_size)?
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
    /// Dispatches to F32 GEMM or quantized GEMV based on weight cache location.
    ///
    /// PMAT-222: Added quantized dispatch for GGUF-sourced APR models.
    /// Phase 45: When test_executor is present, falls back to returning zeros.
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
            return Ok(vec![0.0f32; m * n]);
        }

        // PMAT-222: Check if weight is quantized (GGUF-sourced APR) or F32 (SafeTensors APR)
        if self.executor.has_quantized_weights(weight_name) {
            // Quantized path: dispatch to Q4K or Q6K GEMV kernels
            let qtype = self
                .executor
                .get_quantized_weight_type(weight_name)
                .unwrap_or(12);
            let mut c = vec![0.0f32; m * n];

            match qtype {
                12 => {
                    // Q4_K: use batched GEMV for m>1, single GEMV for m=1
                    if m == 1 {
                        self.executor
                            .q4k_gemv_cached(weight_name, a, &mut c, n as u32, k as u32)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "GPU Q4K GEMV cached".to_string(),
                                reason: format!("CUDA Q4K GEMV '{}' failed: {e}", weight_name),
                            })?;
                    } else {
                        self.executor
                            .batched_q4k_gemv_cached(
                                weight_name,
                                a,
                                &mut c,
                                m as u32,
                                k as u32,
                                n as u32,
                            )
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "GPU Q4K batched GEMV cached".to_string(),
                                reason: format!(
                                    "CUDA batched Q4K GEMV '{}' failed: {e}",
                                    weight_name
                                ),
                            })?;
                    }
                },
                14 => {
                    // Q6_K: use single GEMV, loop for batched
                    if m == 1 {
                        self.executor
                            .q6k_gemv_cached(weight_name, a, &mut c, n as u32, k as u32)
                            .map_err(|e| RealizarError::UnsupportedOperation {
                                operation: "GPU Q6K GEMV cached".to_string(),
                                reason: format!("CUDA Q6K GEMV '{}' failed: {e}", weight_name),
                            })?;
                    } else {
                        // Batched Q6K: process each row individually
                        for row in 0..m {
                            let row_input = &a[row * k..(row + 1) * k];
                            let row_output = &mut c[row * n..(row + 1) * n];
                            self.executor
                                .q6k_gemv_cached(
                                    weight_name,
                                    row_input,
                                    row_output,
                                    n as u32,
                                    k as u32,
                                )
                                .map_err(|e| RealizarError::UnsupportedOperation {
                                    operation: "GPU Q6K GEMV cached (batched)".to_string(),
                                    reason: format!(
                                        "CUDA Q6K GEMV '{}' row {row} failed: {e}",
                                        weight_name
                                    ),
                                })?;
                        }
                    }
                },
                _ => {
                    // Unsupported quantization type, fall back to F32 GEMM
                    self.executor
                        .gemm_b_cached(weight_name, a, &mut c, m as u32, n as u32, k as u32)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "GPU GEMM cached (qtype fallback)".to_string(),
                            reason: format!(
                                "CUDA GEMM '{}' qtype={qtype} failed: {e}",
                                weight_name
                            ),
                        })?;
                },
            }
            Ok(c)
        } else {
            // F32 path: standard GEMM with cached weights
            let mut c = vec![0.0f32; m * n];
            self.executor
                .gemm_b_cached(weight_name, a, &mut c, m as u32, n as u32, k as u32)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "GPU GEMM cached".to_string(),
                    reason: format!("CUDA GEMM with cached weight '{}' failed: {e}", weight_name),
                })?;
            Ok(c)
        }
    }

    /// Check if a weight is cached on GPU.
    ///
    /// Phase 45: Returns false when test_executor is present, forcing the
    /// uncached GEMM path which routes through the test executor.
    ///
    /// Issue #45 fix: Check BOTH weight_cache (f32) and quantized_weight_cache
    /// (Q4_K/Q5_K/Q6_K). APR models use quantized weights, so checking only
    /// weight_cache was causing cache misses and 278x slowdown.
    fn has_cached_weight(&self, name: &str) -> bool {
        if self.test_executor.is_some() {
            return false; // Force uncached path for testing
        }
        // Check both f32 cache and quantized cache
        self.executor.has_weights(name) || self.executor.has_quantized_weights(name)
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
        // PMAT-113-F: Diagnostic tracing for logit verification
        let trace_enabled = std::env::var("APR_TRACE_LOGITS").is_ok();

        // PMAT-114: Fixed prefill - KEEP logits from last token (like GGUF)
        // The logits from processing token[n-1] at position n-1 predict token[n]
        // This matches the GGUF pattern in generate_with_cache (lines 171-183)
        let mut tokens = prompt.to_vec();
        let mut logits = self.forward_cuda(&tokens)?;

        // Decode: generate one token at a time
        // First iteration uses logits from prefill (no extra forward needed)
        for i in 0..max_new_tokens {
            // For subsequent tokens, run forward pass on the newly generated token
            if i > 0 {
                let position = tokens.len();
                let last_token = *tokens.last().unwrap_or(&1);
                logits = self.forward_single_cuda(last_token, position)?;
            }

            // PMAT-113-F: Diagnostic tracing for Q1-Q3
            if trace_enabled && i < 3 {
                let nan_count = logits.iter().filter(|x| x.is_nan()).count();
                let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
                let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = logits.iter().sum();
                let mean = sum / logits.len() as f32;
                let variance: f32 =
                    logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32;

                eprintln!("[PMAT-113-F] Token {}: logits stats:", i);
                eprintln!(
                    "  NaN: {}, Inf: {}, len: {}",
                    nan_count,
                    inf_count,
                    logits.len()
                );
                eprintln!(
                    "  min: {:.4}, max: {:.4}, mean: {:.4}, var: {:.4}",
                    min, max, mean, variance
                );
                eprintln!(
                    "  kv_position: {}, kv_cache_len[0]: {:?}",
                    self.kv_position,
                    self.executor.kv_cache_len(0)
                );

                // Show top 5 token predictions
                let mut indexed: Vec<_> = logits.iter().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!(
                    "  Top 5 tokens: {:?}",
                    indexed
                        .iter()
                        .take(5)
                        .map(|(i, v)| (*i, **v))
                        .collect::<Vec<_>>()
                );
            }

            // Greedy sampling
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(eos_id, |(idx, _)| idx as u32);

            if trace_enabled && i < 3 {
                eprintln!(
                    "  Selected token: {} (logit: {:.4})",
                    next_token,
                    logits.get(next_token as usize).unwrap_or(&0.0)
                );
            }

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}
