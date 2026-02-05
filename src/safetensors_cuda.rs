//! SafeTensors CUDA Inference (PMAT-116)
//!
//! Direct GPU loading for HuggingFace SafeTensors models without intermediate
//! format conversion. Achieves GGUF GPU parity (200+ tok/s).
//!
//! ## Architecture
//!
//! ```text
//! SafeTensors file
//!     ↓ (mmap)
//! TensorView<'data>
//!     ↓ (F16/BF16 → F32 conversion)
//! &[f32] slice
//!     ↓ (executor.load_weights)
//! GPU memory (CudaSlice<f32>)
//!     ↓ (forward_single_cuda)
//! Logits → Token
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::safetensors_cuda::SafeTensorsCudaModel;
//!
//! let mut model = SafeTensorsCudaModel::load("model.safetensors", 0)?;
//! let tokens = model.generate(&[1, 2, 3], 32, 151645)?;
//! ```

use crate::cuda::CudaExecutor;
use crate::error::{RealizarError, Result};
use crate::safetensors::{MappedSafeTensorsModel, SafetensorsConfig};
use std::path::Path;

/// PMAT-120 FIX: Weight transposition for GEMM.
///
/// GEMM kernel computes C[m,n] = A[m,k] × B[k,n] with ROW-MAJOR storage:
/// - A[i,j] at offset `i * k + j`
/// - B[i,j] at offset `i * n + j`
/// - C[i,j] at offset `i * n + j`
///
/// HuggingFace stores Linear weights as [out_features, in_features] = [n, k].
/// GEMM needs B as [k, n]. Therefore: TRANSPOSE IS REQUIRED.
impl SafeTensorsCudaModel {
    /// Transpose weight from HuggingFace [n, k] to GEMM-required [k, n].
    ///
    /// HuggingFace: W[i, j] at offset `i * k + j` where i=0..n, j=0..k
    /// GEMM needs:  B[j, i] at offset `j * n + i` where j=0..k, i=0..n
    fn transpose_for_gemm(weight: &[f32], n: usize, k: usize) -> Vec<f32> {
        let expected_len = n * k;
        // Guard against index out of bounds (PMAT-805 fix)
        if weight.len() < expected_len {
            // Return zero-padded transposed array if weight is undersized
            // This handles edge cases with tied embeddings or partial weights
            let mut transposed = vec![0.0f32; expected_len];
            for i in 0..n {
                for j in 0..k {
                    let src_idx = i * k + j;
                    if src_idx < weight.len() {
                        let dst_idx = j * n + i;
                        transposed[dst_idx] = weight[src_idx];
                    }
                }
            }
            return transposed;
        }

        let mut transposed = vec![0.0f32; expected_len];
        for i in 0..n {
            for j in 0..k {
                // HuggingFace element at row i, col j
                let src_idx = i * k + j;
                // GEMM needs element at row j, col i
                let dst_idx = j * n + i;
                transposed[dst_idx] = weight[src_idx];
            }
        }
        transposed
    }

    /// Concatenate Q, K, V weights and transpose for GEMM.
    ///
    /// HuggingFace stores separately:
    /// - Q: [hidden_dim, hidden_dim] (n=hidden, k=hidden)
    /// - K: [kv_dim, hidden_dim] (n=kv_dim, k=hidden)
    /// - V: [kv_dim, hidden_dim] (n=kv_dim, k=hidden)
    ///
    /// GEMM needs combined QKV as [hidden_dim, hidden_dim + kv_dim + kv_dim].
    fn concat_qkv_transposed(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        // Transpose each weight matrix
        let q_t = Self::transpose_for_gemm(q, hidden_dim, hidden_dim);
        let k_t = Self::transpose_for_gemm(k, kv_dim, hidden_dim);
        let v_t = Self::transpose_for_gemm(v, kv_dim, hidden_dim);

        // After transpose:
        // q_t: [hidden_dim, hidden_dim] row-major
        // k_t: [hidden_dim, kv_dim] row-major
        // v_t: [hidden_dim, kv_dim] row-major

        // Concatenate along columns (output dimension):
        // Result: [hidden_dim, hidden_dim + kv_dim + kv_dim]
        let total_out = hidden_dim + kv_dim + kv_dim;
        let mut qkv = vec![0.0f32; hidden_dim * total_out];

        for row in 0..hidden_dim {
            let dst_start = row * total_out;

            // Copy Q row (hidden_dim elements)
            let q_src = row * hidden_dim;
            qkv[dst_start..dst_start + hidden_dim].copy_from_slice(&q_t[q_src..q_src + hidden_dim]);

            // Copy K row (kv_dim elements)
            let k_src = row * kv_dim;
            qkv[dst_start + hidden_dim..dst_start + hidden_dim + kv_dim]
                .copy_from_slice(&k_t[k_src..k_src + kv_dim]);

            // Copy V row (kv_dim elements)
            let v_src = row * kv_dim;
            qkv[dst_start + hidden_dim + kv_dim..dst_start + hidden_dim + 2 * kv_dim]
                .copy_from_slice(&v_t[v_src..v_src + kv_dim]);
        }

        qkv
    }
}

/// CUDA-accelerated SafeTensors model (PMAT-116)
///
/// Loads HuggingFace SafeTensors directly to GPU memory for high-performance
/// inference. Mirrors `AprV2ModelCuda` API for consistency.
///
/// ## GH-201: Streaming Mode
///
/// Supports two modes based on available VRAM:
/// - **Full Cache**: Pre-cache all weights (default when VRAM sufficient)
/// - **Layer Streaming**: Stream layer weights on-demand (when VRAM limited)
#[cfg(feature = "cuda")]
pub struct SafeTensorsCudaModel {
    /// CUDA executor with cached weights
    executor: CudaExecutor,
    /// Model configuration
    config: SafeTensorsCudaConfig,
    /// GPU device name
    device_name: String,
    /// GPU memory (free, total) in bytes
    memory_info: (usize, usize),
    /// Current KV cache position
    kv_position: u32,
    /// Cached embedding table (F32) - kept on CPU for token lookup
    embedding_cache: Vec<f32>,
    /// RMS norm epsilon
    epsilon: f32,
    /// RMS norm gamma weights (CPU copy for hybrid GPU/CPU path)
    /// Key format: "attn.{layer_idx}" or "ffn.{layer_idx}" or "output"
    gamma_cache: std::collections::HashMap<String, Vec<f32>>,
    /// PMAT-120 FIX: QKV bias cache (Qwen2 has attention bias terms)
    /// Key format: "qkv_bias.{layer_idx}" - concatenated Q+K+V biases
    qkv_bias_cache: std::collections::HashMap<String, Vec<f32>>,
    /// PMAT-120 FIX: Output projection bias cache
    /// Key format: "o_bias.{layer_idx}"
    o_bias_cache: std::collections::HashMap<String, Vec<f32>>,
    /// GH-201: Streaming mode (true = layer-by-layer, false = full cache)
    streaming_mode: bool,
    /// GH-201: Path to SafeTensors file (kept for streaming mode weight loading)
    model_path: Option<std::path::PathBuf>,
}

/// Configuration extracted from config.json
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct SafeTensorsCudaConfig {
    /// Model architecture (e.g., "Qwen2")
    pub architecture: String,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Maximum context length
    pub context_length: usize,
    /// RoPE theta
    pub rope_theta: f32,
    /// RMS norm epsilon
    pub eps: f32,
    /// F-GT-002: Whether to use tied embeddings (lm_head = embed_tokens)
    pub tie_word_embeddings: bool,
}

#[cfg(feature = "cuda")]
impl SafeTensorsCudaModel {
    /// Load SafeTensors model directly to GPU.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to .safetensors file
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if file not found, config.json missing, or CUDA unavailable.
    pub fn load(model_path: &Path, device_ordinal: i32) -> Result<Self> {
        Self::load_with_max_seq_len(model_path, device_ordinal, 2048)
    }

    /// Load SafeTensors model with custom max sequence length.
    pub fn load_with_max_seq_len(
        model_path: &Path,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> Result<Self> {
        // 1. Load SafeTensors via mmap (F-PARSE-036)
        let st_model = MappedSafeTensorsModel::load(model_path)?;

        // 2. Load config.json (F-LOAD-063)
        let json_config = SafetensorsConfig::load_from_sibling(model_path).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "safetensors_cuda_load".to_string(),
                reason: "config.json not found (required for SafeTensors GPU inference)"
                    .to_string(),
            }
        })?;

        // 3. Extract config (F-LOAD-064, F-LOAD-065)
        let config = Self::extract_config(&json_config)?;

        // 4. Initialize CUDA executor (F-CUDA-011)
        let mut executor =
            CudaExecutor::new(device_ordinal).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::new".to_string(),
                reason: format!("CUDA initialization failed: {e}"),
            })?;

        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let memory_info = executor.memory_info().unwrap_or((0, 0));

        // GH-201 FIX: Check VRAM and select streaming mode
        let (free_vram, total_vram) = memory_info;
        let streaming_config = crate::cuda::StreamingConfig {
            hidden_dim: config.hidden_dim,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            vocab_size: config.vocab_size,
            intermediate_dim: config.intermediate_dim,
            max_seq_len,
        };

        let streaming_mode = match crate::cuda::check_vram_sufficient(
            free_vram,
            total_vram,
            &streaming_config,
        ) {
            Ok(crate::cuda::StreamingMode::FullCache) => false,
            Ok(crate::cuda::StreamingMode::LayerStreaming) => {
                eprintln!(
                    "[GH-201] Using layer streaming mode (VRAM: {} MB free of {} MB)",
                    free_vram / (1024 * 1024),
                    total_vram / (1024 * 1024)
                );
                true
            }
            Err(msg) => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "safetensors_cuda_load".to_string(),
                    reason: msg,
                });
            }
        };

        // 5. Initialize GPU KV cache (F-PERF-085)
        let head_dim = config.hidden_dim / config.num_heads;
        executor
            .init_kv_cache_gpu(
                config.num_layers,
                config.num_heads,
                config.num_kv_heads,
                head_dim,
                max_seq_len,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_kv_cache_gpu".to_string(),
                reason: format!("GPU KV cache init failed: {e}"),
            })?;

        // 6. Set RoPE parameters
        executor.set_rope_theta(config.rope_theta);
        executor.set_rope_type(0); // NORM style for Qwen2

        // 7. Upload weights based on mode
        let (embedding_cache, gamma_cache, qkv_bias_cache, o_bias_cache) = if streaming_mode {
            // GH-201: Streaming mode - only upload LM head and norms
            Self::upload_weights_streaming(&mut executor, &st_model, &config)?
        } else {
            // Full cache mode - upload all weights
            Self::upload_weights(&mut executor, &st_model, &config)?
        };

        // Keep path for streaming mode (to reload weights on-demand)
        let model_path = if streaming_mode {
            Some(model_path.to_path_buf())
        } else {
            None
        };

        Ok(Self {
            executor,
            epsilon: config.eps,
            config,
            device_name,
            memory_info,
            kv_position: 0,
            embedding_cache,
            gamma_cache,
            qkv_bias_cache,
            o_bias_cache,
            streaming_mode,
            model_path,
        })
    }

    /// GH-201 FIX: Estimate VRAM required for model weights and KV cache.
    ///
    /// SafeTensors/APR GPU path pre-caches ALL weights upfront (unlike GGUF streaming),
    /// which can cause OOM on GPUs with limited VRAM. This function estimates the
    /// total memory footprint so we can fail early with an actionable error message.
    ///
    /// Memory components:
    /// - LM head: hidden_dim × vocab_size × 4 bytes
    /// - Per layer (×num_layers):
    ///   - QKV weights: hidden_dim × (hidden_dim + 2×kv_dim) × 4
    ///   - O projection: hidden_dim × hidden_dim × 4
    ///   - FFN gate: intermediate_dim × hidden_dim × 4
    ///   - FFN up: intermediate_dim × hidden_dim × 4
    ///   - FFN down: hidden_dim × intermediate_dim × 4
    ///   - Norms: 2 × hidden_dim × 4 (attn + ffn)
    /// - KV cache: 2 × num_layers × max_seq_len × kv_dim × 4
    fn estimate_vram_bytes(config: &SafeTensorsCudaConfig, max_seq_len: usize) -> usize {
        let hidden_dim = config.hidden_dim;
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let num_heads = config.num_heads;
        let intermediate_dim = config.intermediate_dim;
        let vocab_size = config.vocab_size;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // F32 = 4 bytes per element
        const F32_SIZE: usize = 4;

        // LM head (transposed: hidden_dim × vocab_size)
        let lm_head_bytes = hidden_dim * vocab_size * F32_SIZE;

        // Output norm gamma
        let output_norm_bytes = hidden_dim * F32_SIZE;

        // Per-layer weights
        let qkv_out_dim = hidden_dim + 2 * kv_dim;
        let per_layer_bytes = {
            // QKV (transposed: hidden_dim × qkv_out_dim)
            let qkv = hidden_dim * qkv_out_dim * F32_SIZE;
            // O projection (transposed: hidden_dim × hidden_dim)
            let o_proj = hidden_dim * hidden_dim * F32_SIZE;
            // FFN gate (transposed: hidden_dim × intermediate_dim)
            let ffn_gate = hidden_dim * intermediate_dim * F32_SIZE;
            // FFN up (transposed: hidden_dim × intermediate_dim)
            let ffn_up = hidden_dim * intermediate_dim * F32_SIZE;
            // FFN down (transposed: intermediate_dim × hidden_dim)
            let ffn_down = intermediate_dim * hidden_dim * F32_SIZE;
            // Attn + FFN norms (uploaded to rmsnorm_cache)
            let norms = 2 * hidden_dim * F32_SIZE;

            qkv + o_proj + ffn_gate + ffn_up + ffn_down + norms
        };

        let total_layer_bytes = num_layers * per_layer_bytes;

        // KV cache: 2 (K + V) × num_layers × max_seq_len × kv_dim
        let kv_cache_bytes = 2 * num_layers * max_seq_len * kv_dim * F32_SIZE;

        lm_head_bytes + output_norm_bytes + total_layer_bytes + kv_cache_bytes
    }

    /// Extract configuration from JSON config.
    fn extract_config(json: &SafetensorsConfig) -> Result<SafeTensorsCudaConfig> {
        let hidden_dim = json.hidden_size.ok_or_else(|| RealizarError::FormatError {
            reason: "config.json missing hidden_size".to_string(),
        })?;
        let num_layers = json
            .num_hidden_layers
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_hidden_layers".to_string(),
            })?;
        let num_heads = json
            .num_attention_heads
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_attention_heads".to_string(),
            })?;
        let vocab_size = json.vocab_size.ok_or_else(|| RealizarError::FormatError {
            reason: "config.json missing vocab_size".to_string(),
        })?;

        Ok(SafeTensorsCudaConfig {
            architecture: json.architecture(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads: json.num_kv_heads(),
            vocab_size,
            intermediate_dim: json.intermediate_size.unwrap_or(hidden_dim * 4),
            context_length: json.max_position_embeddings.unwrap_or(2048),
            rope_theta: json.rope_theta.unwrap_or(10000.0),
            eps: json.rms_norm_eps.unwrap_or(1e-6),
            tie_word_embeddings: json.tie_word_embeddings.unwrap_or(false),
        })
    }

    /// Upload all model weights to GPU.
    ///
    /// Returns (embedding_table, gamma_cache, qkv_bias_cache, o_bias_cache) - embedding kept on CPU
    /// for token lookup, gamma_cache kept on CPU for RMS norm operations, bias caches for attention.
    #[allow(clippy::type_complexity)]
    fn upload_weights(
        executor: &mut CudaExecutor,
        st_model: &MappedSafeTensorsModel,
        config: &SafeTensorsCudaConfig,
    ) -> Result<(
        Vec<f32>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
    )> {
        let hidden_dim = config.hidden_dim;
        let num_layers = config.num_layers;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let intermediate_dim = config.intermediate_dim;
        let vocab_size = config.vocab_size;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Gamma cache for CPU RMS norm
        let mut gamma_cache = std::collections::HashMap::new();
        // PMAT-120 FIX: Bias caches for attention projections
        let mut qkv_bias_cache = std::collections::HashMap::new();
        let mut o_bias_cache = std::collections::HashMap::new();

        // Embedding table (keep on CPU for token lookup)
        let embedding = st_model.get_tensor_auto("model.embed_tokens.weight")?;

        // Output norm - upload to rmsnorm_cache AND keep CPU copy
        let output_norm = st_model.get_tensor_auto("model.norm.weight")?;
        gamma_cache.insert("output".to_string(), output_norm.clone());
        executor.preload_output_norm(&output_norm).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "preload_output_norm".to_string(),
                reason: format!("Failed to upload output_norm: {e}"),
            }
        })?;

        // LM head (may be tied to embeddings) - use gemm_b_cached (B is weight)
        // F-GT-002 FIX: Check tie_word_embeddings config FIRST, not just tensor existence
        // When tie_word_embeddings=true, HuggingFace may store a placeholder lm_head.weight
        // that's all zeros - we MUST use the embedding matrix instead!
        let lm_head = if config.tie_word_embeddings {
            Self::transpose_for_gemm(&embedding, vocab_size, hidden_dim)
        } else if st_model.has_tensor("lm_head.weight") {
            let raw = st_model.get_tensor_auto("lm_head.weight")?;
            Self::transpose_for_gemm(&raw, vocab_size, hidden_dim)
        } else {
            // Fallback: assume tied if no lm_head tensor exists
            Self::transpose_for_gemm(&embedding, vocab_size, hidden_dim)
        };
        executor.load_weights("lm_head", &lm_head).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload lm_head: {e}"),
            }
        })?;

        // Per-layer weights (F-LOAD-057, F-LOAD-061, F-LOAD-062)
        for layer_idx in 0..num_layers {
            let prefix = format!("model.layers.{layer_idx}");

            // Attention norm - upload to rmsnorm_cache AND keep CPU copy
            let attn_norm =
                st_model.get_tensor_auto(&format!("{prefix}.input_layernorm.weight"))?;
            gamma_cache.insert(format!("attn.{layer_idx}"), attn_norm.clone());
            let attn_norm_key = format!("blk.{layer_idx}.attn_norm.gamma");
            executor
                .cache_rmsnorm_gamma(&attn_norm_key, &attn_norm)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cache_rmsnorm_gamma".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} attn_norm: {e}"),
                })?;

            // QKV weights (concatenate and transpose for gemm_b_cached)
            // PMAT-120 FIX: Use actual transpose for GEMM
            let q = st_model.get_tensor_auto(&format!("{prefix}.self_attn.q_proj.weight"))?;
            let k = st_model.get_tensor_auto(&format!("{prefix}.self_attn.k_proj.weight"))?;
            let v = st_model.get_tensor_auto(&format!("{prefix}.self_attn.v_proj.weight"))?;
            let qkv = Self::concat_qkv_transposed(&q, &k, &v, hidden_dim, kv_dim);
            executor
                .load_weights(&format!("blk.{layer_idx}.attn_qkv"), &qkv)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "load_weights".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} qkv: {e}"),
                })?;

            // PMAT-120 FIX: Load QKV bias terms (Qwen2 has attention biases!)
            // Concatenate Q+K+V biases into single vector
            let q_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.q_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; hidden_dim]);
            let k_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.k_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; kv_dim]);
            let v_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.v_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; kv_dim]);
            let mut qkv_bias = Vec::with_capacity(hidden_dim + 2 * kv_dim);
            qkv_bias.extend_from_slice(&q_bias);
            qkv_bias.extend_from_slice(&k_bias);
            qkv_bias.extend_from_slice(&v_bias);
            qkv_bias_cache.insert(format!("qkv_bias.{layer_idx}"), qkv_bias);

            // Output projection
            // PMAT-120 FIX: Use actual transpose for GEMM
            let o_raw = st_model.get_tensor_auto(&format!("{prefix}.self_attn.o_proj.weight"))?;
            let o = Self::transpose_for_gemm(&o_raw, hidden_dim, hidden_dim);
            executor
                .load_weights(&format!("blk.{layer_idx}.attn_output"), &o)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "load_weights".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} attn_output: {e}"),
                })?;

            // PMAT-120 FIX: Load o_proj bias (if present)
            if let Ok(o_bias) = st_model.get_tensor_auto(&format!("{prefix}.self_attn.o_proj.bias"))
            {
                o_bias_cache.insert(format!("o_bias.{layer_idx}"), o_bias);
            }

            // FFN norm - upload to rmsnorm_cache AND keep CPU copy
            let ffn_norm =
                st_model.get_tensor_auto(&format!("{prefix}.post_attention_layernorm.weight"))?;
            gamma_cache.insert(format!("ffn.{layer_idx}"), ffn_norm.clone());
            let ffn_norm_key = format!("blk.{layer_idx}.ffn_norm.gamma");
            executor
                .cache_rmsnorm_gamma(&ffn_norm_key, &ffn_norm)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cache_rmsnorm_gamma".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} ffn_norm: {e}"),
                })?;

            // FFN gate (SwiGLU)
            // PMAT-120 FIX: Use actual transpose for GEMM
            let gate_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.gate_proj.weight"))?;
            let gate = Self::transpose_for_gemm(&gate_raw, intermediate_dim, hidden_dim);
            executor
                .load_weights(&format!("blk.{layer_idx}.ffn_gate"), &gate)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "load_weights".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} ffn_gate: {e}"),
                })?;

            // FFN up
            // PMAT-120 FIX: Use actual transpose for GEMM
            let up_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.up_proj.weight"))?;
            let up = Self::transpose_for_gemm(&up_raw, intermediate_dim, hidden_dim);
            executor
                .load_weights(&format!("blk.{layer_idx}.ffn_up"), &up)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "load_weights".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} ffn_up: {e}"),
                })?;

            // FFN down
            // PMAT-120 FIX: Use actual transpose for GEMM
            let down_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.down_proj.weight"))?;
            let down = Self::transpose_for_gemm(&down_raw, hidden_dim, intermediate_dim);
            executor
                .load_weights(&format!("blk.{layer_idx}.ffn_down"), &down)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "load_weights".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} ffn_down: {e}"),
                })?;
        }

        Ok((embedding, gamma_cache, qkv_bias_cache, o_bias_cache))
    }

    /// GH-201: Upload only essential weights for streaming mode.
    ///
    /// Uploads:
    /// - LM head (always needed for logits)
    /// - Output norm (always needed)
    /// - Layer norms (small, needed for RMS norm)
    /// - Biases (small, kept on CPU)
    ///
    /// Does NOT upload:
    /// - Per-layer QKV, O, FFN weights (streamed on-demand)
    #[allow(clippy::type_complexity)]
    fn upload_weights_streaming(
        executor: &mut CudaExecutor,
        st_model: &MappedSafeTensorsModel,
        config: &SafeTensorsCudaConfig,
    ) -> Result<(
        Vec<f32>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
    )> {
        let hidden_dim = config.hidden_dim;
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let num_heads = config.num_heads;
        let vocab_size = config.vocab_size;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let mut gamma_cache = std::collections::HashMap::new();
        let mut qkv_bias_cache = std::collections::HashMap::new();
        let mut o_bias_cache = std::collections::HashMap::new();

        // Embedding table (keep on CPU for token lookup)
        let embedding = st_model.get_tensor_auto("model.embed_tokens.weight")?;

        // Output norm - upload to GPU AND keep CPU copy
        let output_norm = st_model.get_tensor_auto("model.norm.weight")?;
        gamma_cache.insert("output".to_string(), output_norm.clone());
        executor.preload_output_norm(&output_norm).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "preload_output_norm".to_string(),
                reason: format!("Failed to upload output_norm: {e}"),
            }
        })?;

        // LM head (always needed) - upload to GPU
        let lm_head = if config.tie_word_embeddings {
            Self::transpose_for_gemm(&embedding, vocab_size, hidden_dim)
        } else if st_model.has_tensor("lm_head.weight") {
            let raw = st_model.get_tensor_auto("lm_head.weight")?;
            Self::transpose_for_gemm(&raw, vocab_size, hidden_dim)
        } else {
            Self::transpose_for_gemm(&embedding, vocab_size, hidden_dim)
        };
        executor.load_weights("lm_head", &lm_head).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload lm_head: {e}"),
            }
        })?;

        // Per-layer: only upload norms and cache biases (small tensors)
        // Layer weights (QKV, O, FFN) will be streamed on-demand
        for layer_idx in 0..num_layers {
            let prefix = format!("model.layers.{layer_idx}");

            // Attention norm (small - upload to GPU)
            let attn_norm =
                st_model.get_tensor_auto(&format!("{prefix}.input_layernorm.weight"))?;
            gamma_cache.insert(format!("attn.{layer_idx}"), attn_norm.clone());
            let attn_norm_key = format!("blk.{layer_idx}.attn_norm.gamma");
            executor
                .cache_rmsnorm_gamma(&attn_norm_key, &attn_norm)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cache_rmsnorm_gamma".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} attn_norm: {e}"),
                })?;

            // FFN norm (small - upload to GPU)
            let ffn_norm =
                st_model.get_tensor_auto(&format!("{prefix}.post_attention_layernorm.weight"))?;
            gamma_cache.insert(format!("ffn.{layer_idx}"), ffn_norm.clone());
            let ffn_norm_key = format!("blk.{layer_idx}.ffn_norm.gamma");
            executor
                .cache_rmsnorm_gamma(&ffn_norm_key, &ffn_norm)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cache_rmsnorm_gamma".to_string(),
                    reason: format!("Failed to upload layer {layer_idx} ffn_norm: {e}"),
                })?;

            // Cache biases on CPU (small)
            let q_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.q_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; hidden_dim]);
            let k_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.k_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; kv_dim]);
            let v_bias = st_model
                .get_tensor_auto(&format!("{prefix}.self_attn.v_proj.bias"))
                .unwrap_or_else(|_| vec![0.0f32; kv_dim]);
            let mut qkv_bias = Vec::with_capacity(hidden_dim + 2 * kv_dim);
            qkv_bias.extend_from_slice(&q_bias);
            qkv_bias.extend_from_slice(&k_bias);
            qkv_bias.extend_from_slice(&v_bias);
            qkv_bias_cache.insert(format!("qkv_bias.{layer_idx}"), qkv_bias);

            if let Ok(o_bias) = st_model.get_tensor_auto(&format!("{prefix}.self_attn.o_proj.bias"))
            {
                o_bias_cache.insert(format!("o_bias.{layer_idx}"), o_bias);
            }
        }

        Ok((embedding, gamma_cache, qkv_bias_cache, o_bias_cache))
    }

    /// Get GPU device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get VRAM in MB.
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        (self.memory_info.1 / (1024 * 1024)) as u64
    }

    /// Get model configuration.
    #[must_use]
    pub fn config(&self) -> &SafeTensorsCudaConfig {
        &self.config
    }

    /// Reset KV cache for new conversation.
    pub fn reset_kv_cache(&mut self) {
        self.kv_position = 0;
        self.executor.reset_kv_cache_gpu();
    }

    /// Generate tokens with GPU acceleration (F-QUAL-066 to F-QUAL-080).
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `eos_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// All tokens (input + generated).
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();

        // PMAT-120 FIX: Prefill processes all input tokens, keeping logits from last one.
        // Previously, logits were discarded during prefill and the last input token was
        // processed AGAIN in the decode loop, causing duplicate KV entries and wrong RoPE.
        let mut last_logits = vec![];
        for &token in input_ids {
            last_logits = self.forward_single(token)?;
        }

        // Sample first new token from prefill logits (not from re-processing last input)
        let first_next = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);

        if first_next == eos_id {
            return Ok(tokens);
        }
        tokens.push(first_next);

        // Decode: generate remaining tokens
        for _ in 1..max_tokens {
            let last_token = *tokens.last().unwrap_or(&1);
            let logits = self.forward_single(last_token)?;

            // Greedy sampling (argmax)
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i as u32);

            if next_token == eos_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Forward pass for a single token.
    fn forward_single(&mut self, token: u32) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;

        // 1. Embedding lookup (CPU)
        let start = (token as usize) * hidden_dim;
        let end = start + hidden_dim;
        if end > self.embedding_cache.len() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "embedding_lookup".to_string(),
                reason: format!("Token {} out of range", token),
            });
        }
        let mut hidden = self.embedding_cache[start..end].to_vec();

        // 2. Transformer layers (GPU)
        // Position tracking is handled internally by incremental_attention_gpu
        for layer_idx in 0..self.config.num_layers {
            hidden = self.forward_layer(layer_idx, &hidden)?;
        }

        // 3. Output norm (CPU path)
        hidden = self.apply_rms_norm_cpu(&hidden)?;

        // 4. LM head projection (GPU) - C = A × B where B is cached lm_head
        let mut logits = vec![0.0f32; vocab_size];
        self.executor
            .gemm_b_cached(
                "lm_head",
                &hidden,           // A: [1, hidden_dim] row vector
                &mut logits,       // C: [1, vocab_size]
                1,                 // m = 1 (single token)
                vocab_size as u32, // n = vocab_size
                hidden_dim as u32, // k = hidden_dim
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "lm_head".to_string(),
                reason: format!("LM head GEMM failed: {e}"),
            })?;

        self.kv_position += 1;
        Ok(logits)
    }

    /// GH-201: Load layer weights on-demand for streaming mode.
    ///
    /// In streaming mode, we don't pre-cache all layer weights. Instead, we load
    /// them from the SafeTensors file on-demand for each layer.
    fn ensure_layer_weights_loaded(&mut self, layer_idx: usize) -> Result<()> {
        if !self.streaming_mode {
            return Ok(()); // Weights already pre-cached
        }

        let model_path = self.model_path.as_ref().ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "ensure_layer_weights_loaded".to_string(),
                reason: "Streaming mode enabled but model_path not set".to_string(),
            }
        })?;

        // Reload SafeTensors (mmap is cheap, it just maps the file)
        let st_model = MappedSafeTensorsModel::load(model_path)?;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let intermediate_dim = self.config.intermediate_dim;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let prefix = format!("model.layers.{layer_idx}");

        // Upload QKV weights (reuses buffer slot from previous layer)
        let q = st_model.get_tensor_auto(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let k = st_model.get_tensor_auto(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let v = st_model.get_tensor_auto(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let qkv = Self::concat_qkv_transposed(&q, &k, &v, hidden_dim, kv_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.attn_qkv"), &qkv)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} qkv: {e}"),
            })?;

        // Upload O projection
        let o_raw = st_model.get_tensor_auto(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let o = Self::transpose_for_gemm(&o_raw, hidden_dim, hidden_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.attn_output"), &o)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} attn_output: {e}"),
            })?;

        // Upload FFN gate
        let gate_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let gate = Self::transpose_for_gemm(&gate_raw, intermediate_dim, hidden_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.ffn_gate"), &gate)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} ffn_gate: {e}"),
            })?;

        // Upload FFN up
        let up_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.up_proj.weight"))?;
        let up = Self::transpose_for_gemm(&up_raw, intermediate_dim, hidden_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.ffn_up"), &up)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} ffn_up: {e}"),
            })?;

        // Upload FFN down
        let down_raw = st_model.get_tensor_auto(&format!("{prefix}.mlp.down_proj.weight"))?;
        let down = Self::transpose_for_gemm(&down_raw, hidden_dim, intermediate_dim);
        self.executor
            .load_weights(&format!("blk.{layer_idx}.ffn_down"), &down)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload layer {layer_idx} ffn_down: {e}"),
            })?;

        Ok(())
    }

    /// Forward pass for a single transformer layer.
    ///
    /// PMAT-120 FIX: RoPE is now applied explicitly here (not in incremental_attention_gpu).
    fn forward_layer(&mut self, layer_idx: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        // GH-201: Load layer weights on-demand if in streaming mode
        self.ensure_layer_weights_loaded(layer_idx)?;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        // 1. Pre-attention norm (CPU for now)
        let normed = self.apply_rms_norm_layer_cpu(hidden, layer_idx, "attn")?;

        // 2. QKV projection: [1, hidden_dim] × [hidden_dim, qkv_out_dim]^T
        let mut qkv = vec![0.0f32; qkv_out_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.attn_qkv"),
                &normed,
                &mut qkv,
                1,
                qkv_out_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "qkv_proj".to_string(),
                reason: format!("Layer {layer_idx} QKV GEMM failed: {e}"),
            })?;

        // PMAT-120 FIX: Add QKV bias (Qwen2 has attention biases!)
        if let Some(bias) = self.qkv_bias_cache.get(&format!("qkv_bias.{layer_idx}")) {
            for (q, b) in qkv.iter_mut().zip(bias.iter()) {
                *q += b;
            }
        }

        // 3. Split Q, K, V
        let mut q = qkv[..hidden_dim].to_vec();
        let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
        let v = qkv[hidden_dim + kv_dim..].to_vec();

        // 4. PMAT-120 FIX: Apply RoPE to Q and K before attention
        // Position is kv_position (number of tokens already processed)
        let position = self.kv_position as usize;
        let rope_theta = self.config.rope_theta;
        let half_dim = head_dim / 2;

        // Apply RoPE to Q (num_heads heads)
        for h in 0..num_heads {
            let head_start = h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx1 = head_start + i;
                let idx2 = head_start + i + half_dim;
                let x1 = q[idx1];
                let x2 = q[idx2];
                q[idx1] = x1 * cos_val - x2 * sin_val;
                q[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }

        // Apply RoPE to K (num_kv_heads heads for GQA)
        for h in 0..num_kv_heads {
            let head_start = h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx1 = head_start + i;
                let idx2 = head_start + i + half_dim;
                let x1 = k[idx1];
                let x2 = k[idx2];
                k[idx1] = x1 * cos_val - x2 * sin_val;
                k[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }

        // 5. Attention with KV cache (GPU)
        let mut attn_output = vec![0.0f32; hidden_dim];
        self.executor
            .incremental_attention_gpu(layer_idx, &q, &k, &v, &mut attn_output)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "attention".to_string(),
                reason: format!("Layer {layer_idx} attention failed: {e}"),
            })?;

        // 5. Output projection
        let mut attn_proj = vec![0.0f32; hidden_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.attn_output"),
                &attn_output,
                &mut attn_proj,
                1,
                hidden_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "attn_output".to_string(),
                reason: format!("Layer {layer_idx} attn output GEMM failed: {e}"),
            })?;

        // PMAT-120 FIX: Add o_proj bias (if present)
        if let Some(bias) = self.o_bias_cache.get(&format!("o_bias.{layer_idx}")) {
            for (p, b) in attn_proj.iter_mut().zip(bias.iter()) {
                *p += b;
            }
        }

        // 6. Residual connection
        let mut residual: Vec<f32> = hidden.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();

        // 7. Post-attention norm (CPU)
        let normed2 = self.apply_rms_norm_layer_cpu(&residual, layer_idx, "ffn")?;

        // 8. FFN gate projection
        let mut gate = vec![0.0f32; intermediate_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.ffn_gate"),
                &normed2,
                &mut gate,
                1,
                intermediate_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "ffn_gate".to_string(),
                reason: format!("Layer {layer_idx} FFN gate GEMM failed: {e}"),
            })?;

        // 9. FFN up projection
        let mut up = vec![0.0f32; intermediate_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.ffn_up"),
                &normed2,
                &mut up,
                1,
                intermediate_dim as u32,
                hidden_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "ffn_up".to_string(),
                reason: format!("Layer {layer_idx} FFN up GEMM failed: {e}"),
            })?;

        // 10. SwiGLU: silu(gate) * up
        let swiglu: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();

        // 11. FFN down projection
        let mut ffn_out = vec![0.0f32; hidden_dim];
        self.executor
            .gemm_b_cached(
                &format!("blk.{layer_idx}.ffn_down"),
                &swiglu,
                &mut ffn_out,
                1,
                hidden_dim as u32,
                intermediate_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "ffn_down".to_string(),
                reason: format!("Layer {layer_idx} FFN down GEMM failed: {e}"),
            })?;

        // 12. Residual connection
        for (r, f) in residual.iter_mut().zip(&ffn_out) {
            *r += f;
        }

        Ok(residual)
    }

    /// Apply RMS normalization with output gamma weights.
    ///
    /// RMS norm formula: (x / sqrt(mean(x^2) + eps)) * gamma
    fn apply_rms_norm_cpu(&self, x: &[f32]) -> Result<Vec<f32>> {
        // RMS norm: x / sqrt(mean(x^2) + eps) * gamma
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + self.epsilon).sqrt();

        // Get output gamma from cache
        let gamma =
            self.gamma_cache
                .get("output")
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "rms_norm".to_string(),
                    reason: "Output gamma not found in cache".to_string(),
                })?;

        // Apply normalization with gamma scaling
        Ok(x.iter()
            .zip(gamma.iter())
            .map(|(xi, gi)| (xi / rms) * gi)
            .collect())
    }

    /// Apply RMS normalization for a specific layer with gamma weights.
    ///
    /// RMS norm formula: (x / sqrt(mean(x^2) + eps)) * gamma
    fn apply_rms_norm_layer_cpu(
        &self,
        x: &[f32],
        layer_idx: usize,
        norm_type: &str,
    ) -> Result<Vec<f32>> {
        // RMS norm: x / sqrt(mean(x^2) + eps) * gamma
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + self.epsilon).sqrt();

        // Get layer gamma from cache
        let cache_key = format!("{norm_type}.{layer_idx}");
        let gamma = self.gamma_cache.get(&cache_key).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "rms_norm".to_string(),
                reason: format!("Layer {layer_idx} {norm_type} gamma not found in cache"),
            }
        })?;

        // Apply normalization with gamma scaling
        Ok(x.iter()
            .zip(gamma.iter())
            .map(|(xi, gi)| (xi / rms) * gi)
            .collect())
    }
}

// ============================================================================
// Tests (F-BUILD-007)
// ============================================================================

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_config_extraction() {
        let json = SafetensorsConfig {
            hidden_size: Some(1536),
            num_hidden_layers: Some(28),
            num_attention_heads: Some(12),
            num_key_value_heads: Some(2),
            vocab_size: Some(151936),
            intermediate_size: Some(8960),
            max_position_embeddings: Some(32768),
            rope_theta: Some(1000000.0),
            rms_norm_eps: Some(1e-6),
            architectures: Some(vec!["Qwen2ForCausalLM".to_string()]),
            model_type: Some("qwen2".to_string()),
            bos_token_id: Some(151643),
            eos_token_id: Some(151645),
            tie_word_embeddings: Some(true), // F-GT-002: Qwen2 uses tied embeddings
        };

        let config = SafeTensorsCudaModel::extract_config(&json).unwrap();
        assert_eq!(config.hidden_dim, 1536);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.intermediate_dim, 8960);
        assert_eq!(config.context_length, 32768);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((config.eps - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn test_config_extraction_defaults() {
        // Test with minimal config (uses defaults for optional fields)
        let json = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            num_key_value_heads: None, // Will default to num_attention_heads
            vocab_size: Some(50257),
            intermediate_size: None,       // Will default to 4 * hidden_dim
            max_position_embeddings: None, // Will default to 2048
            rope_theta: None,              // Will default to 10000.0
            rms_norm_eps: None,            // Will default to 1e-6
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None, // Will default to false
        };

        let config = SafeTensorsCudaModel::extract_config(&json).unwrap();
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.intermediate_dim, 768 * 4); // Default
        assert_eq!(config.context_length, 2048); // Default
        assert!((config.rope_theta - 10000.0).abs() < 0.1); // Default
        assert!((config.eps - 1e-6).abs() < 1e-9); // Default
    }

    #[test]
    fn test_config_extraction_missing_hidden_size() {
        let json = SafetensorsConfig {
            hidden_size: None, // Required, should fail
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            num_key_value_heads: None,
            vocab_size: Some(50257),
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None,
        };

        let result = SafeTensorsCudaModel::extract_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_extraction_missing_layers() {
        let json = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: None, // Required
            num_attention_heads: Some(12),
            num_key_value_heads: None,
            vocab_size: Some(50257),
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None,
        };

        let result = SafeTensorsCudaModel::extract_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_extraction_missing_attention_heads() {
        let json = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: None, // Required
            num_key_value_heads: None,
            vocab_size: Some(50257),
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None,
        };

        let result = SafeTensorsCudaModel::extract_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_extraction_missing_vocab_size() {
        let json = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            num_key_value_heads: None,
            vocab_size: None, // Required
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None,
        };

        let result = SafeTensorsCudaModel::extract_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_for_gemm_identity() {
        // 2x2 matrix transpose
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]] row-major
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 2, 2);
        // Expected: [[1,3],[2,4]] row-major = [1,3,2,4]
        assert_eq!(transposed, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_for_gemm_rectangular() {
        // 2x3 matrix (n=2 rows, k=3 cols) -> 3x2 (k=3 rows, n=2 cols)
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 2, 3);
        // Expected: [[1,4],[2,5],[3,6]] row-major = [1,4,2,5,3,6]
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_for_gemm_single_row() {
        // 1xk matrix (single row)
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2,3,4]]
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 1, 4);
        // Expected: [[1],[2],[3],[4]] row-major = [1,2,3,4] (same as input)
        assert_eq!(transposed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_for_gemm_single_col() {
        // nx1 matrix (single column)
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // [[1],[2],[3],[4]]
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 4, 1);
        // Expected: [[1,2,3,4]] row-major = [1,2,3,4] (same as input)
        assert_eq!(transposed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_for_gemm_4x4() {
        // 4x4 matrix
        let weight = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 4, 4);
        // Diagonal should be unchanged
        assert_eq!(transposed[0], 1.0);
        assert_eq!(transposed[5], 6.0);
        assert_eq!(transposed[10], 11.0);
        assert_eq!(transposed[15], 16.0);
        // Off-diagonal should swap
        assert_eq!(transposed[1], 5.0); // [0,1] gets [1,0]
        assert_eq!(transposed[4], 2.0); // [1,0] gets [0,1]
    }

    #[test]
    fn test_concat_qkv_transposed_simple() {
        // Simplest case: 2 heads, head_dim=2, kv_heads=1
        // hidden_dim = 4, kv_dim = 2
        let q = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]; // 4x4
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x4
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x4

        let qkv = SafeTensorsCudaModel::concat_qkv_transposed(&q, &k, &v, 4, 2);

        // Output should be [hidden_dim, hidden_dim + kv_dim + kv_dim] = [4, 8]
        assert_eq!(qkv.len(), 4 * 8);
    }

    #[test]
    fn test_concat_qkv_transposed_dimensions() {
        // Real-world-like dimensions
        let hidden_dim = 64;
        let kv_dim = 16;

        let q = vec![0.1f32; hidden_dim * hidden_dim];
        let k = vec![0.2f32; kv_dim * hidden_dim];
        let v = vec![0.3f32; kv_dim * hidden_dim];

        let qkv = SafeTensorsCudaModel::concat_qkv_transposed(&q, &k, &v, hidden_dim, kv_dim);

        let expected_len = hidden_dim * (hidden_dim + 2 * kv_dim);
        assert_eq!(qkv.len(), expected_len);
    }

    #[test]
    fn test_safetensors_cuda_config_debug() {
        let config = SafeTensorsCudaConfig {
            architecture: "Qwen2".to_string(),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 4,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
            tie_word_embeddings: true,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Qwen2"));
        assert!(debug_str.contains("768"));
        assert!(debug_str.contains("12"));
    }

    #[test]
    fn test_safetensors_cuda_config_clone() {
        let config = SafeTensorsCudaConfig {
            architecture: "LLaMA".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            tie_word_embeddings: false,
        };

        let cloned = config.clone();
        assert_eq!(cloned.architecture, config.architecture);
        assert_eq!(cloned.hidden_dim, config.hidden_dim);
        assert_eq!(cloned.num_layers, config.num_layers);
        assert_eq!(cloned.num_heads, config.num_heads);
        assert_eq!(cloned.num_kv_heads, config.num_kv_heads);
        assert_eq!(cloned.vocab_size, config.vocab_size);
        assert_eq!(cloned.intermediate_dim, config.intermediate_dim);
        assert_eq!(cloned.context_length, config.context_length);
        assert!((cloned.rope_theta - config.rope_theta).abs() < 0.001);
        assert!((cloned.eps - config.eps).abs() < 1e-10);
    }

    #[test]
    fn test_transpose_preserves_values() {
        // All values should be preserved, just reordered
        let weight: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 3, 4);

        let original_sum: f32 = weight.iter().sum();
        let transposed_sum: f32 = transposed.iter().sum();
        assert!((original_sum - transposed_sum).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_double_transpose_is_identity() {
        // Transpose twice should give original
        let weight: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        let n = 4;
        let k = 5;

        let transposed1 = SafeTensorsCudaModel::transpose_for_gemm(&weight, n, k);
        let transposed2 = SafeTensorsCudaModel::transpose_for_gemm(&transposed1, k, n);

        for (orig, back) in weight.iter().zip(transposed2.iter()) {
            assert!((orig - back).abs() < 1e-6);
        }
    }

    #[test]
    fn test_estimate_vram_bytes_qwen2_1_5b() {
        // GH-201: Test VRAM estimation for Qwen2.5-Coder-1.5B
        // This is the model that triggered the OOM issue
        let config = SafeTensorsCudaConfig {
            architecture: "Qwen2".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 8960,
            context_length: 32768,
            rope_theta: 1000000.0,
            eps: 1e-6,
            tie_word_embeddings: true,
        };

        let vram = SafeTensorsCudaModel::estimate_vram_bytes(&config, 2048);
        let vram_mb = vram / (1024 * 1024);

        // Qwen2.5-Coder-1.5B (1.5B params) requires ~6GB in F32:
        // - LM head: 1536 * 151936 * 4 = ~887 MB
        // - 28 layers × (QKV + O + gate + up + down + norms)
        //   - QKV: 1536 * (1536 + 2*256) * 4 = ~12.5 MB
        //   - O: 1536 * 1536 * 4 = ~9.4 MB
        //   - Gate: 1536 * 8960 * 4 = ~55 MB
        //   - Up: 1536 * 8960 * 4 = ~55 MB
        //   - Down: 8960 * 1536 * 4 = ~55 MB
        //   Total per layer: ~187 MB × 28 = ~5.2 GB
        // - KV cache: 2 * 28 * 2048 * 256 * 4 = ~57 MB
        // Total: ~6GB
        assert!(
            vram_mb > 5500 && vram_mb < 7000,
            "Expected 5.5-7 GB for Qwen2.5-Coder-1.5B F32, got {} MB",
            vram_mb
        );
    }

    #[test]
    fn test_estimate_vram_bytes_scales_with_layers() {
        // More layers should require more VRAM
        let config_12 = SafeTensorsCudaConfig {
            architecture: "Test".to_string(),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
            tie_word_embeddings: false,
        };

        let config_24 = SafeTensorsCudaConfig {
            num_layers: 24,
            ..config_12.clone()
        };

        let vram_12 = SafeTensorsCudaModel::estimate_vram_bytes(&config_12, 1024);
        let vram_24 = SafeTensorsCudaModel::estimate_vram_bytes(&config_24, 1024);

        // 24 layers should use roughly 2x the layer weight memory
        assert!(
            vram_24 > vram_12,
            "24 layers ({}) should use more VRAM than 12 layers ({})",
            vram_24,
            vram_12
        );
    }

    #[test]
    fn test_estimate_vram_bytes_scales_with_seq_len() {
        // Longer sequences need more KV cache
        let config = SafeTensorsCudaConfig {
            architecture: "Test".to_string(),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
            tie_word_embeddings: false,
        };

        let vram_1k = SafeTensorsCudaModel::estimate_vram_bytes(&config, 1024);
        let vram_4k = SafeTensorsCudaModel::estimate_vram_bytes(&config, 4096);

        // 4k context should use more VRAM due to KV cache
        assert!(
            vram_4k > vram_1k,
            "4k context ({}) should use more VRAM than 1k context ({})",
            vram_4k,
            vram_1k
        );
    }
}
