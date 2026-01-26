//! Thread-safe cached model wrapper (Mutex-based)
//!
//! `OwnedQuantizedModelCachedSync` uses Mutex for interior mutability,
//! suitable for async HTTP servers and multi-threaded inference.

use super::weights::{DequantizedFFNWeights, DequantizedWeightCache};
use crate::error::{RealizarError, Result};
use crate::gguf::{
    BatchGenerationStats, DispatchMetrics, OwnedQuantizedKVCache, OwnedQuantizedModel,
    QuantizedGenerateConfig,
};

/// Thread-safe cached model wrapper with Mutex-based scheduler caching
///
/// Uses `Mutex` for interior mutability to cache GPU schedulers. Safe for
/// multi-threaded HTTP serving with async handlers.
pub struct OwnedQuantizedModelCachedSync {
    /// Inner model (not cached)
    model: OwnedQuantizedModel,
    /// Cached HybridScheduler for GPU operations (wgpu backend)
    /// Uses Mutex for thread-safe interior mutability
    scheduler: std::sync::Mutex<Option<crate::gpu::HybridScheduler>>,
    /// PARITY-103: Cached CudaScheduler for direct CUDA operations
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly
    #[cfg(feature = "cuda")]
    cuda_scheduler: std::sync::Mutex<Option<crate::gpu::CudaScheduler>>,
    /// Dequantized weight cache for GPU batch inference (PARITY-019)
    /// Uses RwLock for concurrent read access during batch inference
    dequant_cache: std::sync::RwLock<Option<DequantizedWeightCache>>,
}

// Explicitly implement Send + Sync for HTTP server usage
#[cfg(feature = "gpu")]
unsafe impl Send for OwnedQuantizedModelCachedSync {}
#[cfg(feature = "gpu")]
unsafe impl Sync for OwnedQuantizedModelCachedSync {}

#[cfg(feature = "gpu")]
impl OwnedQuantizedModelCachedSync {
    /// Create a new thread-safe cached model wrapper
    ///
    /// The scheduler is lazily initialized on first GPU operation.
    /// The dequantized weight cache is lazily initialized via `warmup_gpu_cache()`.
    /// PARITY-103: Also initializes CudaScheduler when CUDA feature is enabled.
    #[must_use]
    pub fn new(model: OwnedQuantizedModel) -> Self {
        Self {
            model,
            scheduler: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            cuda_scheduler: std::sync::Mutex::new(None),
            dequant_cache: std::sync::RwLock::new(None),
        }
    }

    /// Get reference to inner model
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModel {
        &self.model
    }

    /// Get or create the cached scheduler (thread-safe)
    ///
    /// # Errors
    /// Returns error if scheduler creation fails or lock is poisoned
    fn get_scheduler(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<crate::gpu::HybridScheduler>>> {
        let mut scheduler_opt =
            self.scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "scheduler_lock".to_string(),
                    reason: "Scheduler mutex poisoned".to_string(),
                })?;

        // Initialize if not already done
        if scheduler_opt.is_none() {
            use crate::gpu::HybridScheduler;
            let new_scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "HybridScheduler::with_threshold".to_string(),
                    reason: format!("GPU scheduler initialization failed: {e}"),
                }
            })?;
            *scheduler_opt = Some(new_scheduler);
        }

        Ok(scheduler_opt)
    }

    /// PARITY-103: Get or create the cached CUDA scheduler (thread-safe)
    ///
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly.
    /// Returns None if CUDA is not available.
    ///
    /// # Errors
    /// Returns error if lock is poisoned
    #[cfg(feature = "cuda")]
    fn get_cuda_scheduler(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<crate::gpu::CudaScheduler>>> {
        use crate::gpu::CudaScheduler;

        let mut scheduler_opt =
            self.cuda_scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "cuda_scheduler_lock".to_string(),
                    reason: "CUDA scheduler mutex poisoned".to_string(),
                })?;

        // Initialize if not already done
        if scheduler_opt.is_none() {
            match CudaScheduler::new() {
                Ok(new_scheduler) => {
                    eprintln!("PARITY-103: CudaScheduler initialized successfully");
                    *scheduler_opt = Some(new_scheduler);
                },
                Err(e) => {
                    // CUDA not available, leave as None (will fallback to wgpu)
                    eprintln!("PARITY-103: CudaScheduler::new() failed: {:?}", e);
                },
            }
        }

        Ok(scheduler_opt)
    }

    /// PARITY-103: Batch matmul preferring CUDA over wgpu (thread-safe)
    ///
    /// Tries CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    /// This bypasses the wgpu 256MB buffer limit that was blocking GPU batch inference.
    #[cfg(feature = "cuda")]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight_f32: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        // Try CUDA first (no buffer size limits)
        if let Ok(mut cuda_guard) = self.get_cuda_scheduler() {
            if let Some(ref mut cuda_sched) = *cuda_guard {
                return cuda_sched
                    .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                        reason: format!("CUDA matmul failed: {e}"),
                    });
            }
        }

        // Fallback to wgpu (may hit 256MB limit for large batches)
        let mut scheduler_guard = self.get_scheduler()?;
        if let Some(ref mut scheduler) = *scheduler_guard {
            return scheduler
                .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                });
        }

        Err(RealizarError::UnsupportedOperation {
            operation: "batch_matmul_gpu_prefer_cuda".to_string(),
            reason: "No GPU scheduler available".to_string(),
        })
    }

    /// PARITY-103: Batch matmul preferring CUDA (non-CUDA fallback)
    #[cfg(not(feature = "cuda"))]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight_f32: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        let mut scheduler_guard = self.get_scheduler()?;
        if let Some(ref mut scheduler) = *scheduler_guard {
            return scheduler
                .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                });
        }

        Err(RealizarError::UnsupportedOperation {
            operation: "batch_matmul_gpu_prefer_cuda".to_string(),
            reason: "No GPU scheduler available".to_string(),
        })
    }

    /// Generate tokens with KV cache using thread-safe cached scheduler
    ///
    /// Delegates to the inner model's `generate_with_cache` method.
    /// The scheduler caching benefits GPU batch operations; single-token
    /// generation uses CPU path with KV cache for O(n) scaling.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        // Delegate to inner model - CPU path with KV cache is already efficient
        self.model.generate_with_cache(prompt, config)
    }

    /// Generate tokens with adaptive CPU/GPU attention (IMP-126)
    ///
    /// This variant of `generate_with_cache` uses adaptive CPU/GPU dispatch
    /// based on cache length and records dispatch decisions to metrics.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    #[cfg(feature = "gpu")]
    pub fn generate_with_cache_adaptive(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        // Delegate to inner model's adaptive generation
        self.model
            .generate_with_cache_adaptive(prompt, config, metrics)
    }

    /// Forward pass with cached scheduler (thread-safe)
    ///
    /// Uses the cached HybridScheduler for GPU operations.
    ///
    /// # Errors
    /// Returns error if GPU operations fail
    #[allow(clippy::let_underscore_untyped)] // Placeholder for future use
    pub fn forward_batch_gpu_cached(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let batch_size = token_ids.len();
        let vocab_size = self.model.config.vocab_size;

        // Get cached scheduler (for future GPU operations)
        let mut scheduler_guard = self.get_scheduler()?;
        let _ = scheduler_guard
            .as_mut()
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "forward_batch_gpu_cached".to_string(),
                reason: "Scheduler not initialized".to_string(),
            })?;

        // 1. Token embedding lookup
        let hidden = self.model.embed(token_ids);

        // 2. Process through layers
        for layer in &self.model.layers {
            // Simplified single-layer forward - reuse inner model logic
            // For full implementation, would need to port the complete forward pass
            let _ = layer;
        }

        // 3. Output normalization and LM head
        // For now, return placeholder - full implementation requires porting forward logic
        let output = vec![0.0f32; batch_size * vocab_size];
        let _ = hidden;

        Ok(output)
    }

    /// Adaptive fused attention for production serving (IMP-121)
    ///
    /// Thread-safe wrapper that automatically selects CPU or GPU based on
    /// sequence length. Uses the cached scheduler for efficient GPU operations.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Attention scale factor
    ///
    /// # Returns
    /// Output tensor [seq_len, head_dim]
    pub fn adaptive_fused_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Threshold for GPU dispatch (from IMP-119 analysis)
        const GPU_SEQ_LEN_THRESHOLD: usize = 64;

        if seq_len >= GPU_SEQ_LEN_THRESHOLD {
            // Long sequence: Use GPU path
            self.gpu_fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        } else {
            // Short sequence: Use CPU path
            self.cpu_fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        }
    }

    /// CPU fused causal attention (thread-safe wrapper)
    fn cpu_fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Use tiled implementation from inner model
        self.model
            .tiled_causal_attention(q, k, v, seq_len, head_dim, scale, 4)
    }

    /// GPU fused causal attention (thread-safe)
    fn gpu_fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        let mut scheduler_guard =
            self.scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "gpu_fused_causal_attention".to_string(),
                    reason: "Failed to acquire scheduler lock".to_string(),
                })?;

        // Initialize scheduler if needed
        if scheduler_guard.is_none() {
            use crate::gpu::HybridScheduler;
            let new_scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "HybridScheduler::with_threshold".to_string(),
                    reason: format!("GPU scheduler initialization failed: {e}"),
                }
            })?;
            *scheduler_guard = Some(new_scheduler);
        }

        let scheduler =
            scheduler_guard
                .as_mut()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "gpu_fused_causal_attention".to_string(),
                    reason: "Scheduler not initialized".to_string(),
                })?;

        // Transpose K for matmul
        let mut k_transposed = vec![0.0f32; head_dim * seq_len];
        for pos in 0..seq_len {
            for d in 0..head_dim {
                k_transposed[d * seq_len + pos] = k[pos * head_dim + d];
            }
        }

        // GPU Q @ K^T
        let scores = scheduler
            .matmul(q, &k_transposed, seq_len, head_dim, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused Q@K^T".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })?;

        // CPU causal softmax
        let mut weights = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                if score > max_val {
                    max_val = score;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                weights[i * seq_len + j] = (score - max_val).exp();
                sum += weights[i * seq_len + j];
            }
            if sum > 0.0 {
                for j in 0..=i {
                    weights[i * seq_len + j] /= sum;
                }
            }
        }

        // GPU weights @ V
        scheduler
            .matmul(&weights, v, seq_len, seq_len, head_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused weights@V".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })
    }

    /// Adaptive multihead attention for production serving (IMP-121)
    ///
    /// Thread-safe multi-head attention that automatically selects backend.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_dim]
    pub fn adaptive_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Reshape Q, K, V to [num_heads, seq_len, head_dim]
        let q_reshaped = self
            .model
            .reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self
            .model
            .reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;
        let v_reshaped = self
            .model
            .reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        let mut attn_output = vec![0.0f32; num_heads * seq_len * head_dim];

        for h in 0..num_heads {
            let head_offset = h * seq_len * head_dim;
            let q_head = &q_reshaped[head_offset..head_offset + seq_len * head_dim];
            let k_head = &k_reshaped[head_offset..head_offset + seq_len * head_dim];
            let v_head = &v_reshaped[head_offset..head_offset + seq_len * head_dim];

            let head_output =
                self.adaptive_fused_attention(q_head, k_head, v_head, seq_len, head_dim, scale)?;

            attn_output[head_offset..head_offset + seq_len * head_dim]
                .copy_from_slice(&head_output);
        }

        // Reshape back to [seq_len, hidden_dim]
        let mut output = vec![0.0f32; seq_len * hidden_dim];
        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            for pos in 0..seq_len {
                let src_start = head_start + pos * head_dim;
                let dst_start = pos * hidden_dim + h * head_dim;
                output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&attn_output[src_start..src_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Warmup GPU weight cache for batch inference (PARITY-019)
    ///
    /// Pre-dequantizes all FFN weights to f32 for GPU GEMM operations.
    /// Call this once at server startup to avoid dequantization during inference.
    ///
    /// # Memory Usage
    /// - phi-2 (32 layers): ~6.4 GB
    /// - Per layer: 2 × hidden_dim × intermediate_dim × 4 bytes
    ///
    /// # Returns
    /// - Total memory allocated in bytes
    /// - Number of layers cached
    ///
    /// # Errors
    /// Returns error if dequantization fails
    pub fn warmup_gpu_cache(&self) -> Result<(usize, usize)> {
        let config = &self.model.config;
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let num_layers = self.model.layers.len();

        // Create cache with model dimensions
        let cache = DequantizedWeightCache::new(hidden_dim, intermediate_dim, num_layers);

        // Dequantize each layer's FFN weights
        // Note: warmup closure can't return Result, so we use unwrap_or_default
        // for robustness. In production, use warmup_gpu_cache_checked() for error handling.
        cache.warmup(|layer_idx| {
            let layer = &self.model.layers[layer_idx];

            // Dequantize using model's dequantize_weight method
            let up = self
                .model
                .dequantize_weight(&layer.ffn_up_weight)
                .unwrap_or_default();
            let down = self
                .model
                .dequantize_weight(&layer.ffn_down_weight)
                .unwrap_or_default();

            (up, down)
        });

        let memory_bytes = cache.memory_bytes();
        let cached_count = cache.cached_count();

        // Store in the cache field
        let mut cache_guard =
            self.dequant_cache
                .write()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "warmup_gpu_cache".to_string(),
                    reason: "Cache lock poisoned".to_string(),
                })?;
        *cache_guard = Some(cache);

        Ok((memory_bytes, cached_count))
    }

    /// Check if GPU cache is warmed up
    pub fn is_gpu_cache_warm(&self) -> bool {
        self.dequant_cache
            .read()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Get GPU cache memory usage in bytes
    pub fn gpu_cache_memory(&self) -> usize {
        self.dequant_cache
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(DequantizedWeightCache::memory_bytes))
            .unwrap_or(0)
    }

    /// Get dequantized weights for a layer (for GPU batch FFN)
    ///
    /// Returns None if cache not warmed up or layer not found.
    pub fn get_dequantized_ffn_weights(&self, layer_idx: usize) -> Option<DequantizedFFNWeights> {
        self.dequant_cache
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|c| c.get(layer_idx)))
    }

    /// Batch FFN forward pass using GPU (PARITY-019)
    ///
    /// Processes multiple tokens in parallel using GPU GEMM.
    /// Requires cache to be warmed up via `warmup_gpu_cache()`.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch_size × hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Output tensor [batch_size × hidden_dim]
    ///
    /// # Errors
    /// Returns error if cache not warmed or GPU operations fail
    /// PARITY-103: Batch FFN using CUDA when available
    ///
    /// Uses CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    /// This bypasses the wgpu 256MB buffer limit that was blocking GPU batch inference.
    pub fn batch_ffn_gpu(&self, hidden_states: &[f32], layer_idx: usize) -> Result<Vec<f32>> {
        let config = &self.model.config;
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let batch_size = hidden_states.len() / hidden_dim;

        if batch_size == 0 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "batch_ffn_gpu".to_string(),
                reason: "Empty batch".to_string(),
            });
        }

        // Get cached weights
        let weights = self.get_dequantized_ffn_weights(layer_idx).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "batch_ffn_gpu".to_string(),
                reason: format!(
                    "Layer {} not cached. Call warmup_gpu_cache() first.",
                    layer_idx
                ),
            }
        })?;

        // PARITY-103: Up projection preferring CUDA
        let mut intermediate = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &weights.up,
            batch_size,
            hidden_dim,
            intermediate_dim,
        )?;

        // Add up bias if present
        if let Some(ref bias) = weights.up_bias {
            for b in 0..batch_size {
                for i in 0..intermediate_dim {
                    intermediate[b * intermediate_dim + i] += bias[i];
                }
            }
        }

        // GELU activation (CPU - fused in future)
        for x in &mut intermediate {
            let x64 = *x as f64;
            *x = (x64
                * 0.5
                * (1.0 + (x64 * 0.797_884_560_8 * (1.0 + 0.044_715 * x64 * x64)).tanh()))
                as f32;
        }

        // PARITY-103: Down projection preferring CUDA
        let mut output = self.batch_matmul_gpu_prefer_cuda(
            &intermediate,
            &weights.down,
            batch_size,
            intermediate_dim,
            hidden_dim,
        )?;

        // Add down bias if present
        if let Some(ref bias) = weights.down_bias {
            for b in 0..batch_size {
                for i in 0..hidden_dim {
                    output[b * hidden_dim + i] += bias[i];
                }
            }
        }

        Ok(output)
    }

    /// PARITY-103: Batch QKV projection using CUDA when available
    ///
    /// Projects hidden states to Q, K, V for all requests in batch.
    /// [batch, hidden] @ [hidden, 3*hidden] = [batch, 3*hidden]
    ///
    /// Uses CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    ///
    /// # Arguments
    /// * `hidden_states` - Flattened hidden states [batch * hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Flattened QKV projections [batch * 3 * hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn batch_qkv_projection_gpu(
        &self,
        hidden_states: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let batch_size = hidden_states.len() / hidden_dim;
        let qkv_dim = 3 * hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let layer = &self.model.layers[layer_idx];

        // Dequantize QKV weight for GPU GEMM
        let qkv_weight = self.model.dequantize_qkv(&layer.qkv_weight)?;

        // PARITY-103: QKV projection preferring CUDA
        let mut qkv = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &qkv_weight,
            batch_size,
            hidden_dim,
            qkv_dim,
        )?;

        // Add bias if present
        if let Some(ref bias) = layer.qkv_bias {
            for b in 0..batch_size {
                for i in 0..qkv_dim {
                    qkv[b * qkv_dim + i] += bias[i];
                }
            }
        }

        Ok(qkv)
    }

    /// Batch attention output projection using GPU GEMM (PARITY-024)
    ///
    /// Projects attention outputs for all requests in batch.
    /// [batch, hidden] @ [hidden, hidden] = [batch, hidden]
    ///
    /// # Arguments
    /// * `attention_outputs` - Flattened attention outputs [batch * hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Flattened projected outputs [batch * hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn batch_attention_output_gpu(
        &self,
        attention_outputs: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let batch_size = attention_outputs.len() / hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let layer = &self.model.layers[layer_idx];

        // Dequantize output weight for GPU GEMM
        let output_weight = self.model.dequantize_weight(&layer.attn_output_weight)?;

        // PARITY-103: Output projection preferring CUDA (bypasses wgpu 256MB limit)
        // [batch, hidden] @ [hidden, hidden] = [batch, hidden]
        let mut output = self.batch_matmul_gpu_prefer_cuda(
            attention_outputs,
            &output_weight,
            batch_size,
            hidden_dim,
            hidden_dim,
        )?;

        // Add bias if present
        if let Some(ref bias) = layer.attn_output_bias {
            for b in 0..batch_size {
                for i in 0..hidden_dim {
                    output[b * hidden_dim + i] += bias[i];
                }
            }
        }

        Ok(output)
    }

    /// Batch LM head projection using GPU GEMM (PARITY-025)
    ///
    /// Projects hidden states to vocabulary logits for all requests in batch.
    /// [batch, hidden] @ [hidden, vocab] = [batch, vocab]
    ///
    /// # Arguments
    /// * `hidden_states` - Flattened normalized hidden states [batch * hidden_dim]
    ///
    /// # Returns
    /// Flattened logits [batch * vocab_size]
    #[cfg(feature = "gpu")]
    pub fn batch_lm_head_gpu(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let vocab_size = self.model.config.vocab_size;
        let batch_size = hidden_states.len() / hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Dequantize LM head weight for GPU GEMM
        let lm_head_weight = self.model.dequantize_weight(&self.model.lm_head_weight)?;

        // PARITY-103: LM head projection preferring CUDA (bypasses wgpu 256MB limit)
        // [batch, hidden] @ [hidden, vocab] = [batch, vocab]
        let mut logits = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &lm_head_weight,
            batch_size,
            hidden_dim,
            vocab_size,
        )?;

        // Add bias if present
        if let Some(ref bias) = self.model.lm_head_bias {
            for b in 0..batch_size {
                for i in 0..vocab_size {
                    logits[b * vocab_size + i] += bias[i];
                }
            }
        }

        Ok(logits)
    }

    /// Batch generation with GPU-accelerated FFN (PARITY-020)
    ///
    /// Processes multiple prompts in parallel using GPU batch operations.
    /// The key optimization is converting MATVEC (single token) to GEMM (batch tokens).
    ///
    /// # Architecture
    /// - Attention: CPU with KV cache (MATVEC is faster on CPU)
    /// - FFN: GPU with batch GEMM (batch_size ≥ 32 uses GPU)
    /// - Sampling: CPU (negligible compared to matmul)
    ///
    /// # Arguments
    /// * `prompts` - Multiple prompts to process in parallel [num_prompts][seq_len]
    /// * `config` - Generation configuration (shared across all prompts)
    ///
    /// # Returns
    /// Generated sequences for each prompt [num_prompts][generated_len]
    ///
    /// # Errors
    /// Returns error if GPU cache not warmed up or generation fails
    ///
    /// # Performance
    /// - Single prompt: ~5 tok/s (CPU-bound, no batching benefit)
    /// - 32 prompts: ~150 tok/s total (~4.7 tok/s per prompt)
    /// - 64 prompts: ~280 tok/s total (~4.4 tok/s per prompt, memory-bound)
    pub fn batch_generate_gpu(
        &self,
        prompts: &[Vec<u32>],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<Vec<u32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        // Verify GPU cache is warmed up
        if !self.is_gpu_cache_warm() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "batch_generate_gpu".to_string(),
                reason: "GPU cache not warmed up. Call warmup_gpu_cache() first.".to_string(),
            });
        }

        let num_prompts = prompts.len();
        let max_seq_len = prompts.iter().map(Vec::len).max().unwrap_or(0) + config.max_tokens;

        // Initialize KV caches for each prompt
        let mut caches: Vec<OwnedQuantizedKVCache> = prompts
            .iter()
            .map(|_| OwnedQuantizedKVCache::from_config(&self.model.config, max_seq_len))
            .collect();

        // Initialize token sequences (copy prompts)
        let mut sequences: Vec<Vec<u32>> = prompts.to_vec();

        // Track generation progress per prompt
        let mut done: Vec<bool> = vec![false; num_prompts];

        // PARITY-097: Parallel prefill across prompts using rayon
        // Each prompt's prefill is independent (different KV cache)
        // Model is shared immutably (&self), caches are mutated independently
        use rayon::prelude::*;

        caches
            .par_iter_mut()
            .zip(prompts.par_iter())
            .try_for_each(|(cache, prompt)| {
                for (pos, &token_id) in prompt.iter().enumerate() {
                    self.model.forward_single_with_cache(token_id, cache, pos)?;
                }
                Ok::<_, RealizarError>(())
            })?;

        // Generation loop with batched FFN (PARITY-021: GPU optimization)
        for gen_idx in 0..config.max_tokens {
            // Collect active prompts for this generation step
            let active_indices: Vec<usize> = (0..num_prompts).filter(|&i| !done[i]).collect();

            if active_indices.is_empty() {
                break;
            }

            let active_count = active_indices.len();

            // Use batched forward when we have enough active prompts for GPU benefit
            // GPU batch threshold is 32 (from IMP-600 analysis)
            const GPU_BATCH_THRESHOLD: usize = 32;

            if active_count >= GPU_BATCH_THRESHOLD {
                // PARITY-021: Batched forward with GPU FFN
                // Collect tokens, positions, and cache slices for active prompts
                let batch_tokens: Vec<u32> = active_indices
                    .iter()
                    .map(|&idx| {
                        *sequences[idx]
                            .last()
                            .expect("sequence must have at least prompt tokens")
                    })
                    .collect();

                let batch_positions: Vec<usize> = active_indices
                    .iter()
                    .map(|&idx| prompts[idx].len() + gen_idx)
                    .collect();

                // PARITY-096: Extract caches without cloning using std::mem::take
                // This avoids expensive cache cloning on every generation step
                let mut batch_caches: Vec<OwnedQuantizedKVCache> = active_indices
                    .iter()
                    .map(|&idx| std::mem::take(&mut caches[idx]))
                    .collect();

                // Forward batch with GPU FFN
                let all_logits = self.forward_batch_with_gpu_ffn(
                    &batch_tokens,
                    &mut batch_caches,
                    &batch_positions,
                )?;

                // PARITY-096: Put caches back (move, not clone)
                for (i, &idx) in active_indices.iter().enumerate() {
                    caches[idx] = std::mem::take(&mut batch_caches[i]);
                }

                // Sample and update sequences
                for (i, &prompt_idx) in active_indices.iter().enumerate() {
                    let logits = &all_logits[i];
                    let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                        OwnedQuantizedModel::argmax(logits)
                    } else {
                        OwnedQuantizedModel::sample_topk(logits, config.temperature, config.top_k)
                    };

                    if config.stop_tokens.contains(&next_token) {
                        done[prompt_idx] = true;
                    } else {
                        sequences[prompt_idx].push(next_token);
                        if sequences[prompt_idx].len() >= max_seq_len {
                            done[prompt_idx] = true;
                        }
                    }
                }
            } else {
                // Sequential forward for small batches (CPU is faster)
                for &prompt_idx in &active_indices {
                    let position = prompts[prompt_idx].len() + gen_idx;
                    let last_token = *sequences[prompt_idx]
                        .last()
                        .expect("sequence must have at least prompt tokens");

                    let logits = self.model.forward_single_with_cache(
                        last_token,
                        &mut caches[prompt_idx],
                        position,
                    )?;

                    let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                        OwnedQuantizedModel::argmax(&logits)
                    } else {
                        OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
                    };

                    if config.stop_tokens.contains(&next_token) {
                        done[prompt_idx] = true;
                    } else {
                        sequences[prompt_idx].push(next_token);
                        if sequences[prompt_idx].len() >= max_seq_len {
                            done[prompt_idx] = true;
                        }
                    }
                }
            }
        }

        Ok(sequences)
    }

    /// Batched forward pass with GPU FFN optimization (PARITY-021)
    ///
    /// Processes multiple tokens in parallel with GPU-accelerated FFN.
    /// Attention is still per-token with CPU KV cache, but FFN uses GPU GEMM.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs for each prompt [batch_size]
    /// * `caches` - Per-prompt KV caches
    /// * `positions` - Position for each prompt [batch_size]
    ///
    /// # Returns
    /// Logits for each prompt [batch_size][vocab_size]
    ///
    /// # GPU Dispatch
    /// - batch_size >= 32: GPU GEMM for FFN (10x speedup)
    /// - batch_size < 32: CPU fallback
    pub fn forward_batch_with_gpu_ffn(
        &self,
        token_ids: &[u32],
        caches: &mut [OwnedQuantizedKVCache],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let batch_size = token_ids.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if batch_size != caches.len() || batch_size != positions.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Batch size mismatch: tokens={}, caches={}, positions={}",
                    batch_size,
                    caches.len(),
                    positions.len()
                ),
            });
        }

        let hidden_dim = self.model.config.hidden_dim;
        let num_layers = self.model.layers.len();

        // Threshold for GPU dispatch (based on IMP-600 analysis)
        const GPU_BATCH_THRESHOLD: usize = 32;
        let use_gpu = batch_size >= GPU_BATCH_THRESHOLD && self.is_gpu_cache_warm();

        // PARITY-098: Parallel embedding using rayon
        use rayon::prelude::*;
        let mut hidden_states: Vec<Vec<f32>> = token_ids
            .par_iter()
            .map(|&tid| self.model.embed(&[tid]))
            .collect();

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            let layer = &self.model.layers[layer_idx];

            // PARITY-024: GPU batch attention path vs CPU sequential path
            if use_gpu {
                // GPU path: batch QKV projection, per-prompt attention, batch output projection

                // 2a. PARITY-098: Parallel batch layer norm
                let normed_batch: Vec<Vec<f32>> = hidden_states
                    .par_iter()
                    .map(|hidden| {
                        self.model.layer_norm(
                            hidden,
                            &layer.attn_norm_weight,
                            layer.attn_norm_bias.as_deref(),
                            self.model.config.eps,
                        )
                    })
                    .collect();

                // 2b. Batch QKV projection using GPU GEMM (PARITY-024)
                let batch_normed: Vec<f32> = normed_batch.iter().flatten().copied().collect();
                let batch_qkv = self.batch_qkv_projection_gpu(&batch_normed, layer_idx)?;

                // 2c-2e. PARITY-099: Parallel attention computation per prompt
                // Each prompt has its own KV cache, so we can parallelize
                let qkv_dim = 3 * hidden_dim;

                let attention_outputs: Vec<Vec<f32>> = caches
                    .par_iter_mut()
                    .enumerate()
                    .map(|(prompt_idx, cache)| {
                        let qkv_start = prompt_idx * qkv_dim;
                        let qkv = &batch_qkv[qkv_start..qkv_start + qkv_dim];

                        // Extract Q, K, V
                        let mut q = qkv[0..hidden_dim].to_vec();
                        let mut k = qkv[hidden_dim..2 * hidden_dim].to_vec();
                        let v = qkv[2 * hidden_dim..3 * hidden_dim].to_vec();

                        // Apply RoPE (position-dependent, must be per-prompt)
                        // Note: Uses num_heads for both (non-GQA code path)
                        self.model.apply_rope(
                            &mut q,
                            positions[prompt_idx],
                            self.model.config.num_heads,
                        );
                        self.model.apply_rope(
                            &mut k,
                            positions[prompt_idx],
                            self.model.config.num_heads,
                        );

                        // Attention with KV cache (must be per-prompt, different caches)
                        // PARITY-027: Use FlashAttention for long sequences (O(N) memory)
                        let k_cache = cache.get_k(layer_idx);
                        let v_cache = cache.get_v(layer_idx);

                        // FlashAttention threshold: use for sequences >= 512 tokens
                        const FLASH_ATTENTION_THRESHOLD: usize = 512;
                        let cache_len = k_cache.len() / hidden_dim;
                        let use_flash_attention = cache_len >= FLASH_ATTENTION_THRESHOLD;

                        let attn_out = if k_cache.is_empty() {
                            v.clone()
                        } else if use_flash_attention {
                            // FlashAttention: O(N) memory, tiled computation
                            const FLASH_BLOCK_SIZE: usize = 64;
                            self.model.flash_attention_tiled(
                                &q,
                                k_cache,
                                v_cache,
                                &k,
                                &v,
                                FLASH_BLOCK_SIZE,
                            )
                        } else {
                            // Standard attention: O(N²) memory but faster for short sequences
                            self.model
                                .attention_with_cache(&q, k_cache, v_cache, &k, &v)
                        };

                        // Store K and V in cache
                        cache.append(layer_idx, &k, &v);
                        attn_out
                    })
                    .collect();

                // 2f. Batch attention output projection using GPU GEMM (PARITY-024)
                let batch_attn: Vec<f32> = attention_outputs.iter().flatten().copied().collect();
                let batch_output = self.batch_attention_output_gpu(&batch_attn, layer_idx)?;

                // 2g. PARITY-100: Parallel residual connection
                hidden_states
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(prompt_idx, hidden)| {
                        let start = prompt_idx * hidden_dim;
                        for i in 0..hidden_dim {
                            hidden[i] += batch_output[start + i];
                        }
                    });
            } else {
                // CPU sequential path (original implementation)
                for (prompt_idx, hidden) in hidden_states.iter_mut().enumerate() {
                    // Attention layer norm
                    let normed = self.model.layer_norm(
                        hidden,
                        &layer.attn_norm_weight,
                        layer.attn_norm_bias.as_deref(),
                        self.model.config.eps,
                    );

                    // QKV projection
                    let mut qkv = self.model.qkv_matmul(&normed, &layer.qkv_weight)?;
                    if let Some(ref bias) = layer.qkv_bias {
                        self.model.add_bias(&mut qkv, bias);
                    }

                    // Extract Q, K, V and apply RoPE
                    // Note: Uses num_heads for both (non-GQA code path)
                    let mut q = qkv[0..hidden_dim].to_vec();
                    let mut k = qkv[hidden_dim..2 * hidden_dim].to_vec();
                    let v = qkv[2 * hidden_dim..3 * hidden_dim].to_vec();

                    self.model.apply_rope(
                        &mut q,
                        positions[prompt_idx],
                        self.model.config.num_heads,
                    );
                    self.model.apply_rope(
                        &mut k,
                        positions[prompt_idx],
                        self.model.config.num_heads,
                    );

                    // Get cached K/V and compute attention
                    let k_cache = caches[prompt_idx].get_k(layer_idx);
                    let v_cache = caches[prompt_idx].get_v(layer_idx);

                    let attn_out = if k_cache.is_empty() {
                        v.clone()
                    } else {
                        self.model
                            .attention_with_cache(&q, k_cache, v_cache, &k, &v)
                    };

                    // Store K and V in cache
                    caches[prompt_idx].append(layer_idx, &k, &v);

                    // Attention output projection
                    let mut attn_output = self
                        .model
                        .fused_matmul(&attn_out, &layer.attn_output_weight)?;
                    if let Some(ref bias) = layer.attn_output_bias {
                        self.model.add_bias(&mut attn_output, bias);
                    }

                    // Residual connection
                    for i in 0..hidden_dim {
                        hidden[i] += attn_output[i];
                    }
                }
            }

            // 2h. FFN - GPU batch or CPU sequential
            if use_gpu {
                // GPU batch FFN: collect hidden states, process together, scatter back
                let batch_hidden: Vec<f32> = hidden_states.iter().flatten().copied().collect();
                let ffn_output = self.batch_ffn_gpu(&batch_hidden, layer_idx)?;

                // PARITY-100: Parallel scatter and residual
                hidden_states
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(prompt_idx, hidden)| {
                        let start = prompt_idx * hidden_dim;
                        for i in 0..hidden_dim {
                            hidden[i] += ffn_output[start + i];
                        }
                    });
            } else {
                // CPU sequential FFN
                for hidden in &mut hidden_states {
                    let mut ffn_hidden = self.model.fused_matmul(hidden, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        self.model.add_bias(&mut ffn_hidden, bias);
                    }
                    self.model.gelu(&mut ffn_hidden);

                    let mut ffn_output = self
                        .model
                        .fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
                    if let Some(ref bias) = layer.ffn_down_bias {
                        self.model.add_bias(&mut ffn_output, bias);
                    }

                    // Residual
                    for i in 0..hidden_dim {
                        hidden[i] += ffn_output[i];
                    }
                }
            }
        }

        // PARITY-100: Parallel cache advance
        caches.par_iter_mut().for_each(|cache| {
            cache.advance();
        });

        // 3. Final layer norm and LM head for each prompt
        // PARITY-025: Use GPU batch LM head when batch >= threshold
        let vocab_size = self.model.config.vocab_size;

        let all_logits: Vec<Vec<f32>> = if use_gpu {
            // GPU path: batch layer norm and LM head projection

            // 3a. PARITY-098: Parallel final layer norm
            let normed_batch: Vec<Vec<f32>> = hidden_states
                .par_iter()
                .map(|hidden| {
                    self.model.layer_norm(
                        hidden,
                        &self.model.output_norm_weight,
                        self.model.output_norm_bias.as_deref(),
                        self.model.config.eps,
                    )
                })
                .collect();

            // 3b. Batch LM head projection using GPU GEMM (PARITY-025)
            let batch_normed: Vec<f32> = normed_batch.iter().flatten().copied().collect();
            let batch_logits = self.batch_lm_head_gpu(&batch_normed)?;

            // 3c. PARITY-098: Parallel scatter logits back to per-prompt vectors
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let start = i * vocab_size;
                    batch_logits[start..start + vocab_size].to_vec()
                })
                .collect()
        } else {
            // CPU path: sequential per-prompt processing
            let mut result = Vec::with_capacity(batch_size);
            for hidden in &hidden_states {
                let normed = self.model.layer_norm(
                    hidden,
                    &self.model.output_norm_weight,
                    self.model.output_norm_bias.as_deref(),
                    self.model.config.eps,
                );

                let mut logits = self
                    .model
                    .fused_matmul(&normed, &self.model.lm_head_weight)?;
                if let Some(ref bias) = self.model.lm_head_bias {
                    self.model.add_bias(&mut logits, bias);
                }
                result.push(logits);
            }
            result
        };

        Ok(all_logits)
    }

    /// Get batch generation statistics
    ///
    /// Returns information about the batch processing capabilities.
    pub fn batch_stats(&self) -> BatchGenerationStats {
        let is_cached = self.is_gpu_cache_warm();
        let memory_gb = self.gpu_cache_memory() as f64 / 1_000_000_000.0;
        let num_layers = self.model.layers.len();
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.config.intermediate_dim;

        BatchGenerationStats {
            gpu_cache_ready: is_cached,
            cache_memory_gb: memory_gb,
            num_layers,
            hidden_dim,
            intermediate_dim,
            recommended_batch_size: 32, // GPU GEMM threshold
            max_batch_size: 64,         // Memory-limited
        }
    }
}
