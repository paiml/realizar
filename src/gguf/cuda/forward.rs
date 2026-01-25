//! Forward pass methods for CUDA-accelerated inference
//!
//! This module contains all forward pass implementations:
//! - `forward_cuda`: Basic forward pass
//! - `forward_single_cuda_with_cache`: Single token with KV cache
//! - `forward_single_full_cuda_with_cache`: Full GPU forward with cache
//! - `forward_gpu_resident`: GPU-resident forward (minimal CPU↔GPU transfers)
//! - Internal helpers: `fused_matmul_cuda`, `qkv_matmul_cuda`, `cuda_attention_with_cache`

use crate::error::{RealizarError, Result};
use crate::gguf::ops;
use super::{OwnedQuantizedModelCuda, OwnedQKVWeights, OwnedQuantizedTensor, OwnedQuantizedKVCache};

impl OwnedQuantizedModelCuda {
    /// Forward pass using CUDA GEMM acceleration (IMP-800a)
    ///
    /// Uses CudaExecutor for matrix multiplications in the FFN layers.
    /// Attention and embedding remain on CPU for now.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    ///
    /// Returns error if CUDA operations fail
    pub fn forward_cuda(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;

        // 1. Token embedding lookup (CPU - fast enough)
        let mut hidden = self.model.embed(token_ids);

        // 2. Process through transformer layers
        for layer in &self.model.layers {
            // 2a. Attention layer norm (CPU)
            let normed = ops::layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.model.config.eps,
            );

            // 2b. QKV projection (CPU - fused Q4_K for now)
            // GQA-aware dimensions: Q has num_heads, K/V have num_kv_heads
            let num_kv_heads = self.model.config.num_kv_heads;
            let head_dim = hidden_dim / self.model.config.num_heads;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = hidden_dim + 2 * kv_dim; // Q + K + V with GQA
            let mut qkv = self.model.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Attention (CPU - complex control flow)
            let seq_len = token_ids.len();
            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k = qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v = &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                // GQA-aware RoPE: Q uses num_heads, K uses num_kv_heads
                self.model
                    .apply_rope(&mut q, s, self.model.config.num_heads);
                self.model.apply_rope(&mut k, s, num_kv_heads);

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            let attn_out = self.model.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // 2d. Attention output projection (CPU - fused Q4_K)
            let mut attn_output = self
                .model
                .fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN up projection - try GPU GEMM if weights are dequantized
            // For now, use CPU fused ops (GPU overhead too high for m=1)
            let mut ffn_hidden = self.model.fused_matmul(&hidden, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(&mut ffn_hidden, bias);
            }

            // GELU activation (CPU)
            ops::gelu(&mut ffn_hidden);

            // 2g. FFN down projection (CPU fused)
            let mut ffn_output = self
                .model
                .fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm (CPU)
        let normed = ops::layer_norm(
            &hidden,
            &self.model.output_norm_weight,
            self.model.output_norm_bias.as_deref(),
            self.model.config.eps,
        );

        // 4. LM head projection (CPU fused)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self
            .model
            .fused_matmul(last_hidden, &self.model.lm_head_weight)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Generate tokens using CUDA acceleration (IMP-800a)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn forward_single_cuda_with_cache(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim; // GQA: K/V may have fewer heads than Q
        let num_layers = self.model.layers.len();
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast enough)
        let mut hidden = self.model.embed(&[token_id]);

        // 2. Process through transformer layers (index-based to avoid borrow issues)
        for layer_idx in 0..num_layers {
            // 2a. Attention layer norm (CPU)
            let normed = ops::layer_norm(
                &hidden,
                &self.model.layers[layer_idx].attn_norm_weight,
                self.model.layers[layer_idx].attn_norm_bias.as_deref(),
                eps,
            );

            // 2b. QKV projection (CPU - fused Q4_K)
            let mut qkv = self
                .model
                .qkv_matmul(&normed, &self.model.layers[layer_idx].qkv_weight)?;
            if let Some(ref bias) = self.model.layers[layer_idx].qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V and apply RoPE (GQA-aware dimensions)
            // Q has hidden_dim = num_heads * head_dim
            // K/V have kv_dim = num_kv_heads * head_dim (may be smaller for GQA)
            let mut q = qkv[0..hidden_dim].to_vec();
            let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
            let v = qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();

            self.model
                .apply_rope(&mut q, position, self.model.config.num_heads);
            self.model
                .apply_rope(&mut k, position, self.model.config.num_kv_heads);

            // 2d. Get cached K/V and compute attention
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            let attn_out = if k_cache.is_empty() {
                // First token - no cache yet, GQA expansion needed for output
                if num_kv_heads < num_heads {
                    // Expand V to match num_heads by repeating KV groups
                    let q_per_kv = num_heads / num_kv_heads;
                    let mut expanded = vec![0.0f32; hidden_dim];
                    for q_head in 0..num_heads {
                        let kv_head = q_head / q_per_kv;
                        let src_offset = kv_head * head_dim;
                        let dst_offset = q_head * head_dim;
                        expanded[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&v[src_offset..src_offset + head_dim]);
                    }
                    expanded
                } else {
                    v.clone()
                }
            } else {
                // Use GPU multi-head attention if cache is large enough (PARITY-044)
                let cache_len = if kv_dim > 0 {
                    k_cache.len() / kv_dim
                } else {
                    0
                };
                let total_len = cache_len + 1;

                // PAR-017: Lower GPU attention threshold for more consistent GPU usage
                // Previous: 32 tokens caused high variance with short sequences
                const GPU_ATTN_THRESHOLD: usize = 8;

                if total_len >= GPU_ATTN_THRESHOLD && num_kv_heads == num_heads {
                    // GPU path only works for non-GQA models currently
                    self.cuda_attention_with_cache(
                        &q, k_cache, v_cache, &k, &v, total_len, num_heads, head_dim,
                    )?
                } else {
                    // CPU path for short sequences or GQA models
                    // Use GQA-aware version that handles grouped KV heads correctly
                    self.model
                        .attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
                }
            };

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, &k, &v);

            // 2f. Attention output projection (CPU fused)
            let mut attn_output = self
                .model
                .fused_matmul(&attn_out, &self.model.layers[layer_idx].attn_output_weight)?;
            if let Some(ref bias) = self.model.layers[layer_idx].attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // PAR-047: FFN with proper SwiGLU/GELU detection
            // LLaMA-family models use SwiGLU (ffn_gate_weight present)
            // Phi-2 style models use GELU (no gate weight)
            let ffn_activated =
                if let Some(ref gate_weight) = self.model.layers[layer_idx].ffn_gate_weight {
                    // SwiGLU path (LLaMA, TinyLlama, Mistral, Qwen, etc.)
                    // Apply FFN norm if present (separate from attention norm in LLaMA-style)
                    let ffn_input =
                        if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                            ops::layer_norm(
                                &hidden,
                                ffn_norm,
                                self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                                eps,
                            )
                        } else {
                            hidden.clone()
                        };

                    let mut ffn_up = self
                        .model
                        .fused_matmul(&ffn_input, &self.model.layers[layer_idx].ffn_up_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }

                    let mut ffn_gate = self.model.fused_matmul(&ffn_input, gate_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    // SwiGLU: silu(gate) * up
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                } else {
                    // GELU path (phi-2 style, no gate weight)
                    let mut ffn_hidden = self
                        .model
                        .fused_matmul(&hidden, &self.model.layers[layer_idx].ffn_up_weight)?;
                    if let Some(ref bias) = self.model.layers[layer_idx].ffn_up_bias {
                        ops::add_bias(&mut ffn_hidden, bias);
                    }
                    ops::gelu(&mut ffn_hidden);
                    ffn_hidden
                };

            // 2i. FFN down projection (CPU fused)
            let mut ffn_output = self.model.fused_matmul(
                &ffn_activated,
                &self.model.layers[layer_idx].ffn_down_weight,
            )?;
            if let Some(ref bias) = self.model.layers[layer_idx].ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // 3. Final layer norm (CPU)
        let normed = ops::layer_norm(
            &hidden,
            &self.model.output_norm_weight,
            self.model.output_norm_bias.as_deref(),
            self.model.config.eps,
        );

        // 4. LM head projection (CPU fused)
        let mut logits = self
            .model
            .fused_matmul(&normed, &self.model.lm_head_weight)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// IMP-1010: GPU-accelerated fused Q4_K matmul
    ///
    /// Uses `CudaExecutor::q4k_matvec` to execute quantized matrix-vector
    /// multiplication directly on GPU, avoiding CPU SIMD overhead.
    ///
    /// # Performance Impact
    ///
    /// - CPU SIMD path: ~5 tok/s (limited by memory bandwidth)
    /// - GPU CUDA path: ~200 tok/s (theoretical, matching Ollama)
    /// - Key: Dequantize on GPU, not on CPU
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector (f32)
    /// * `weight` - Quantized weight tensor (Q4_K format)
    ///
    /// # Returns
    ///
    /// Output vector [out_dim]
    fn fused_matmul_cuda(
        &mut self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
    ) -> Result<Vec<f32>> {
        // Only Q4_K is supported for GPU acceleration (PARITY-041)
        const GGUF_TYPE_Q4_K: u32 = 12;

        if weight.qtype != GGUF_TYPE_Q4_K {
            // Fallback to CPU for non-Q4_K weights
            return self.model.fused_matmul(input, weight);
        }

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;

        // GPU kernel expects single input (seq_len=1 during token generation)
        if input.len() != in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "IMP-1010: Input length {} doesn't match weight in_dim {}",
                    input.len(),
                    in_dim
                ),
            });
        }

        // Allocate output buffer
        let mut output = vec![0.0f32; out_dim];

        // PAR-014: Use cached GEMV for weight reuse (avoids re-transfer each call)
        // Cache key is based on weight data pointer (stable since model owns data)
        let cache_key = format!("q4k_{:016x}", weight.data.as_ptr() as usize);

        // Lazy cache - upload weight on first use
        if !self.executor.has_quantized_weights(&cache_key) {
            self.executor
                .load_quantized_weights(&cache_key, &weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_q4k_cache".to_string(),
                    reason: format!("Failed to cache Q4_K weights: {e}"),
                })?;
        }

        // Execute Q4_K matmul on GPU using cached weights
        self.executor
            .q4k_gemv_cached(
                &cache_key,
                input,
                &mut output,
                out_dim as u32,
                in_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "q4k_gemv_cached".to_string(),
                reason: format!("CUDA Q4_K GEMV failed: {e}"),
            })?;

        Ok(output)
    }

    /// PAR-014: Fused matmul with explicit cache key
    ///
    /// Same as `fused_matmul_cuda` but accepts an explicit cache key, allowing
    /// the caller to use the original weight pointer for caching even when
    /// working with cloned weight data.
    fn fused_matmul_cuda_with_key(
        &mut self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        cache_key: &str,
    ) -> Result<Vec<f32>> {
        // Only Q4_K is supported for GPU acceleration
        const GGUF_TYPE_Q4_K: u32 = 12;

        if weight.qtype != GGUF_TYPE_Q4_K {
            // Fallback to CPU for non-Q4_K weights
            return self.model.fused_matmul(input, weight);
        }

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;

        if input.len() != in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "PAR-014: Input length {} doesn't match weight in_dim {}",
                    input.len(),
                    in_dim
                ),
            });
        }

        let mut output = vec![0.0f32; out_dim];

        // Lazy cache - upload weight on first use
        if !self.executor.has_quantized_weights(cache_key) {
            self.executor
                .load_quantized_weights(cache_key, &weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_q4k_cache".to_string(),
                    reason: format!("Failed to cache Q4_K weights: {e}"),
                })?;
        }

        // Execute Q4_K matmul on GPU using cached weights
        self.executor
            .q4k_gemv_cached(cache_key, input, &mut output, out_dim as u32, in_dim as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "q4k_gemv_cached".to_string(),
                reason: format!("CUDA Q4_K GEMV failed: {e}"),
            })?;

        Ok(output)
    }

    /// QKV matmul with CUDA - handles both fused and separate Q/K/V
    ///
    /// Five Whys Root Cause Fix: Supports TinyLlama and other LLaMA-style models
    fn qkv_matmul_cuda(&mut self, input: &[f32], qkv: &OwnedQKVWeights) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul_cuda(input, weight),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // Compute Q, K, V separately then concatenate
                let q_out = self.fused_matmul_cuda(input, q)?;
                let k_out = self.fused_matmul_cuda(input, k)?;
                let v_out = self.fused_matmul_cuda(input, v)?;

                // Concatenate Q, K, V
                let mut output = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// PAR-014: QKV matmul with explicit cache key for fused weights
    ///
    /// Same as `qkv_matmul_cuda` but accepts a cache key for the fused case.
    fn qkv_matmul_cuda_with_key(
        &mut self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        cache_key: &str,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => {
                self.fused_matmul_cuda_with_key(input, weight, cache_key)
            },
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // For separate Q/K/V, we still use the cloned pointers
                // (less critical since these are already separate tensors)
                let q_out = self.fused_matmul_cuda(input, q)?;
                let k_out = self.fused_matmul_cuda(input, k)?;
                let v_out = self.fused_matmul_cuda(input, v)?;

                let mut output = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// IMP-1010: Full GPU forward pass for single token with KV cache
    ///
    /// This method uses GPU acceleration for ALL matmul operations:
    /// - QKV projection (3x hidden_dim × hidden_dim)
    /// - Attention output projection (hidden_dim × hidden_dim)
    /// - FFN up projection (hidden_dim × 4*hidden_dim)
    /// - FFN down projection (4*hidden_dim × hidden_dim)
    /// - LM head projection (hidden_dim × vocab_size)
    ///
    /// # Performance Target
    ///
    /// - CPU SIMD path: ~5 tok/s
    /// - Full GPU path: ~200 tok/s (matching Ollama)
    pub fn forward_single_full_cuda_with_cache(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let num_layers = self.model.layers.len();
        let eps = self.model.config.eps;

        // PAR-021: GQA support
        // Q: [hidden_dim] = [num_heads * head_dim]
        // K: [kv_dim] = [num_kv_heads * head_dim] (smaller for GQA)
        // V: [kv_dim] = [num_kv_heads * head_dim] (smaller for GQA)
        let kv_dim = num_kv_heads * head_dim;

        // 1. Token embedding lookup (CPU - fast enough, single lookup)
        let mut hidden = self.model.embed(&[token_id]);

        // IMP-1010-DEBUG: Check embedding output (disabled for performance)
        #[allow(clippy::never_loop)]
        if false {
            let embed_sum: f32 = hidden.iter().sum();
            let embed_has_nan = hidden.iter().any(|x| x.is_nan());
            eprintln!(
                "[IMP-1010] pos{} embedding: sum={:.6e}, has_nan={}",
                position, embed_sum, embed_has_nan
            );
        }

        // PAR-016: Pre-capture LM head cache key for stable caching
        let lm_head_cache_key = format!(
            "q4k_{:016x}",
            self.model.lm_head_weight.data.as_ptr() as usize
        );

        // PAR-050: Detect RMSNorm architecture (LLaMA uses RMSNorm and SwiGLU)
        let use_rmsnorm = self
            .model
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // PAR-014: Capture original weight pointers BEFORE cloning for stable cache keys
            // This ensures weight caching works across forward passes
            let attn_output_cache_key = format!(
                "q4k_{:016x}",
                self.model.layers[layer_idx]
                    .attn_output_weight
                    .data
                    .as_ptr() as usize
            );
            let ffn_up_cache_key = format!(
                "q4k_{:016x}",
                self.model.layers[layer_idx].ffn_up_weight.data.as_ptr() as usize
            );
            let ffn_down_cache_key = format!(
                "q4k_{:016x}",
                self.model.layers[layer_idx].ffn_down_weight.data.as_ptr() as usize
            );
            // Capture QKV weight pointer for cache key (handles both Fused and Separate)
            let qkv_cache_key = match &self.model.layers[layer_idx].qkv_weight {
                OwnedQKVWeights::Fused(ref tensor) => {
                    format!("q4k_{:016x}", tensor.data.as_ptr() as usize)
                },
                OwnedQKVWeights::Separate { ref q, .. } => {
                    // Use Q tensor pointer as representative key for separate case
                    format!("q4k_{:016x}", q.data.as_ptr() as usize)
                },
            };

            // Clone weights to avoid borrow conflicts with &mut self
            // IMP-1010: This is necessary because fused_matmul_cuda needs &mut self
            let qkv_weight = self.model.layers[layer_idx].qkv_weight.clone();
            let qkv_bias = self.model.layers[layer_idx].qkv_bias.clone();
            let attn_norm_weight = self.model.layers[layer_idx].attn_norm_weight.clone();
            let attn_norm_bias = self.model.layers[layer_idx].attn_norm_bias.clone();
            let attn_output_weight_data =
                self.model.layers[layer_idx].attn_output_weight.data.clone();
            let attn_output_weight_in_dim = self.model.layers[layer_idx].attn_output_weight.in_dim;
            let attn_output_weight_out_dim =
                self.model.layers[layer_idx].attn_output_weight.out_dim;
            let attn_output_weight_qtype = self.model.layers[layer_idx].attn_output_weight.qtype;
            let attn_output_bias = self.model.layers[layer_idx].attn_output_bias.clone();
            let ffn_up_weight_data = self.model.layers[layer_idx].ffn_up_weight.data.clone();
            let ffn_up_weight_in_dim = self.model.layers[layer_idx].ffn_up_weight.in_dim;
            let ffn_up_weight_out_dim = self.model.layers[layer_idx].ffn_up_weight.out_dim;
            let ffn_up_weight_qtype = self.model.layers[layer_idx].ffn_up_weight.qtype;
            let ffn_up_bias = self.model.layers[layer_idx].ffn_up_bias.clone();
            let ffn_down_weight_data = self.model.layers[layer_idx].ffn_down_weight.data.clone();
            let ffn_down_weight_in_dim = self.model.layers[layer_idx].ffn_down_weight.in_dim;
            let ffn_down_weight_out_dim = self.model.layers[layer_idx].ffn_down_weight.out_dim;
            let ffn_down_weight_qtype = self.model.layers[layer_idx].ffn_down_weight.qtype;
            let ffn_down_bias = self.model.layers[layer_idx].ffn_down_bias.clone();
            // PAR-015: Extract FFN gate weight for SwiGLU (LLaMA models)
            let ffn_gate_weight = self.model.layers[layer_idx].ffn_gate_weight.clone();
            let ffn_gate_bias = self.model.layers[layer_idx].ffn_gate_bias.clone();
            let ffn_gate_cache_key = ffn_gate_weight
                .as_ref()
                .map(|w| format!("q4k_{:016x}", w.data.as_ptr() as usize));

            // Reconstruct weight tensors
            let attn_output_weight = OwnedQuantizedTensor {
                data: attn_output_weight_data,
                in_dim: attn_output_weight_in_dim,
                out_dim: attn_output_weight_out_dim,
                qtype: attn_output_weight_qtype,
            };
            let ffn_up_weight = OwnedQuantizedTensor {
                data: ffn_up_weight_data,
                in_dim: ffn_up_weight_in_dim,
                out_dim: ffn_up_weight_out_dim,
                qtype: ffn_up_weight_qtype,
            };
            let ffn_down_weight = OwnedQuantizedTensor {
                data: ffn_down_weight_data,
                in_dim: ffn_down_weight_in_dim,
                out_dim: ffn_down_weight_out_dim,
                qtype: ffn_down_weight_qtype,
            };

            // 2a. Attention layer norm (CPU - fast for single vector)
            // PAR-050: Use RMSNorm for LLaMA models (no bias), LayerNorm for others
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &attn_norm_weight, eps)
            } else {
                ops::layer_norm(&hidden, &attn_norm_weight, attn_norm_bias.as_deref(), eps)
            };

            // IMP-1010-DEBUG: Check normed output for NaN (disabled for performance)
            #[allow(clippy::never_loop)]
            if false {
                let normed_has_nan = normed.iter().any(|x| x.is_nan());
                let normed_sum: f32 = normed.iter().sum();
                let normed_max = normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} normed: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, normed_sum, normed_max, normed_has_nan
                );
            }

            // 2b. QKV projection (GPU - PAR-014: use pre-captured cache key)
            let mut qkv = self.qkv_matmul_cuda_with_key(&normed, &qkv_weight, &qkv_cache_key)?;
            if let Some(ref bias) = qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // IMP-1010-DEBUG: Check QKV output for NaN
            if false {
                let qkv_has_nan = qkv.iter().any(|x| x.is_nan());
                let qkv_sum: f32 = qkv.iter().sum();
                let qkv_max = qkv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} QKV: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, qkv_sum, qkv_max, qkv_has_nan
                );
            }

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            // PAR-021: For GQA, K and V have smaller kv_dim (num_kv_heads * head_dim)
            let mut q = qkv[0..hidden_dim].to_vec();
            let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
            let v = qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();

            self.model
                .apply_rope(&mut q, position, self.model.config.num_heads);
            self.model.apply_rope(&mut k, position, num_kv_heads);

            // 2d. Get cached K/V and compute attention
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            // CORRECTNESS-RESOLVED: Always use CPU attention (GPU attention precision issues)
            // GPU matmul is still used for QKV, output, and FFN projections
            let attn_out = if k_cache.is_empty() {
                // First token - no cache yet
                // PAR-021: Expand V for GQA (each KV head serves multiple Q heads)
                if num_kv_heads < num_heads {
                    let q_per_kv = num_heads / num_kv_heads;
                    let mut expanded_v = vec![0.0f32; hidden_dim];
                    for q_head in 0..num_heads {
                        let kv_head = q_head / q_per_kv;
                        let v_start = kv_head * head_dim;
                        let out_start = q_head * head_dim;
                        expanded_v[out_start..out_start + head_dim]
                            .copy_from_slice(&v[v_start..v_start + head_dim]);
                    }
                    expanded_v
                } else {
                    v.clone()
                }
            } else {
                // Use CPU GQA-aware attention (correct implementation)
                self.model
                    .attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
            };

            // 2e. Store K and V in cache (only CPU cache if no GPU cache)
            // IMP-1010-DEBUG: Always use CPU cache since GPU attention is disabled
            // if !self.executor.has_kv_cache_gpu() {
            cache.append(layer_idx, &k, &v);
            // }

            // 2f. Attention output projection (GPU - PAR-014: use pre-captured cache key)
            let mut attn_output = self.fused_matmul_cuda_with_key(
                &attn_out,
                &attn_output_weight,
                &attn_output_cache_key,
            )?;
            if let Some(ref bias) = attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // IMP-1010-DEBUG: Check attention output for NaN
            if false {
                let attn_out_has_nan = attn_out.iter().any(|x| x.is_nan());
                let attn_out_sum: f32 = attn_out.iter().sum();
                let attn_proj_has_nan = attn_output.iter().any(|x| x.is_nan());
                let attn_proj_sum: f32 = attn_output.iter().sum();
                let attn_proj_max = attn_output
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[IMP-1010] pos{} L{} attn_out: sum={:.6e}, has_nan={} | proj: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, attn_out_sum, attn_out_has_nan, attn_proj_sum, attn_proj_max, attn_proj_has_nan);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // IMP-1010-DEBUG: Check hidden after residual
            if false {
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                let hidden_sum: f32 = hidden.iter().sum();
                let hidden_max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} after attn: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, hidden_sum, hidden_max, hidden_has_nan
                );
            }

            // PAR-049-DEBUG: Compare attention output with CPU (disabled for performance)
            // Re-enable by changing `false` to `true` for debugging
            #[allow(clippy::never_loop, clippy::while_let_on_iterator)]
            if false {
                let cpu_attn = self.model.fused_matmul(&attn_out, &attn_output_weight)?;
                let max_diff = attn_output
                    .iter()
                    .zip(cpu_attn.iter())
                    .map(|(g, c)| (g - c).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "[PAR-049] L0 pos{} attn_output max_diff: {:.6e}",
                    position, max_diff
                );
                if position == 1 {
                    eprintln!(
                        "[PAR-049] L0 pos1 attn_out[0..5]: {:?}",
                        &attn_out[..5.min(attn_out.len())]
                    );
                    let k_len = cache.get_k(layer_idx).len();
                    let v_len = cache.get_v(layer_idx).len();
                    eprintln!(
                        "[PAR-049] L0 pos1 k_cache.len: {}, v_cache.len: {}",
                        k_len, v_len
                    );
                }
            }

            // 2h/2i. FFN
            // PAR-057: Re-enable fused FFN path now that kernels are fixed

            // IMP-1010-DEBUG: Check hidden state going into FFN for layers near NaN origin
            if false {
                let hidden_sum: f32 = hidden.iter().sum();
                let hidden_max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let hidden_min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                eprintln!("[IMP-1010] pos{} L{} before FFN: sum={:.6e}, min={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, hidden_sum, hidden_min, hidden_max, hidden_has_nan);
            }

            #[allow(clippy::overly_complex_bool_expr)]
            let ffn_output = if ffn_up_bias.is_none()
                && ffn_down_bias.is_none()
                && ffn_up_weight.qtype == 12
                && ffn_down_weight.qtype == 12
            {
                // Fused FFN path: up + GELU + down in single GPU round-trip
                let intermediate_dim = ffn_up_weight.out_dim;
                let mut output = vec![0.0f32; hidden_dim];

                // Ensure weights are cached
                if !self.executor.has_quantized_weights(&ffn_up_cache_key) {
                    self.executor
                        .load_quantized_weights(&ffn_up_cache_key, &ffn_up_weight.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_ffn_up_cache".to_string(),
                            reason: format!("Failed to cache FFN up weights: {e}"),
                        })?;
                }
                if !self.executor.has_quantized_weights(&ffn_down_cache_key) {
                    self.executor
                        .load_quantized_weights(&ffn_down_cache_key, &ffn_down_weight.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_ffn_down_cache".to_string(),
                            reason: format!("Failed to cache FFN down weights: {e}"),
                        })?;
                }

                self.executor
                    .fused_ffn_q4k(
                        &hidden,
                        &mut output,
                        &ffn_up_cache_key,
                        &ffn_down_cache_key,
                        hidden_dim as u32,
                        intermediate_dim as u32,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "cuda_fused_ffn".to_string(),
                        reason: format!("CUDA fused FFN failed: {e}"),
                    })?;

                // IMP-1010-DEBUG: Check fused FFN output for layers near NaN origin
                if false {
                    let out_sum: f32 = output.iter().sum();
                    let out_max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let out_min = output.iter().cloned().fold(f32::INFINITY, f32::min);
                    let out_has_nan = output.iter().any(|x| x.is_nan());
                    eprintln!("[IMP-1010] pos{} L{} fused_ffn out: sum={:.6e}, min={:.6e}, max={:.6e}, has_nan={}",
                        position, layer_idx, out_sum, out_min, out_max, out_has_nan);
                }

                output
            } else if let (Some(ref gate_weight), Some(ref gate_cache_key)) =
                (&ffn_gate_weight, &ffn_gate_cache_key)
            {
                // PAR-015/PAR-049: SwiGLU path for LLaMA models
                // Formula: down(silu(gate(norm(x))) * up(norm(x)))
                // PAR-049 FIX: Apply FFN layer norm before projections (was missing!)

                // Apply FFN layer norm if present (separate from attention norm in LLaMA-style)
                // PAR-050: Use RMSNorm for LLaMA models
                let ffn_input =
                    if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                        if use_rmsnorm {
                            ops::rms_norm(&hidden, ffn_norm, eps)
                        } else {
                            ops::layer_norm(
                                &hidden,
                                ffn_norm,
                                self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                                eps,
                            )
                        }
                    } else {
                        hidden.clone()
                    };

                // UP projection on normalized input
                let mut ffn_up =
                    self.fused_matmul_cuda_with_key(&ffn_input, &ffn_up_weight, &ffn_up_cache_key)?;
                if let Some(ref bias) = ffn_up_bias {
                    ops::add_bias(&mut ffn_up, bias);
                }

                // GATE projection on normalized input
                let mut ffn_gate =
                    self.fused_matmul_cuda_with_key(&ffn_input, gate_weight, gate_cache_key)?;
                if let Some(ref bias) = ffn_gate_bias {
                    ops::add_bias(&mut ffn_gate, bias);
                }

                // SiLU on gate, then multiply with up
                ops::silu(&mut ffn_gate);
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }

                // DOWN projection
                let mut ffn_output = self.fused_matmul_cuda_with_key(
                    &ffn_gate,
                    &ffn_down_weight,
                    &ffn_down_cache_key,
                )?;
                if let Some(ref bias) = ffn_down_bias {
                    ops::add_bias(&mut ffn_output, bias);
                }
                ffn_output
            } else {
                // GELU path for phi-2 style models (no gate projection)
                // IMP-1010 FIX: Apply FFN layer norm if present (parallel residual models like phi-2
                // use the same normalized input for both attention and FFN)
                let ffn_input =
                    if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                        if use_rmsnorm {
                            ops::rms_norm(&hidden, ffn_norm, eps)
                        } else {
                            ops::layer_norm(
                                &hidden,
                                ffn_norm,
                                self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                                eps,
                            )
                        }
                    } else {
                        // Parallel residual: use same normalized input as attention
                        normed.clone()
                    };
                let mut ffn_hidden =
                    self.fused_matmul_cuda_with_key(&ffn_input, &ffn_up_weight, &ffn_up_cache_key)?;
                if let Some(ref bias) = ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);

                let mut ffn_output = self.fused_matmul_cuda_with_key(
                    &ffn_hidden,
                    &ffn_down_weight,
                    &ffn_down_cache_key,
                )?;
                if let Some(ref bias) = ffn_down_bias {
                    ops::add_bias(&mut ffn_output, bias);
                }
                ffn_output
            };

            // PAR-049-DEBUG: Compare FFN output with CPU (disabled for performance)
            #[allow(clippy::never_loop)]
            if false {
                // Compute CPU FFN for comparison
                let _ffn_input_cpu =
                    if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
                        ops::layer_norm(
                            &hidden,
                            ffn_norm,
                            self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                            eps,
                        )
                    } else {
                        hidden.clone()
                    };
                // hidden before residual add of ffn
                let hidden_before_ffn: Vec<f32> = hidden
                    .iter()
                    .zip(&attn_output)
                    .map(|(h, a)| h - a)
                    .collect();
                eprintln!(
                    "[PAR-049] L0 hidden before attn residual[0..5]: {:?}",
                    &hidden_before_ffn[..5.min(hidden_before_ffn.len())]
                );
                eprintln!(
                    "[PAR-049] L0 ffn_output GPU[0..5]: {:?}",
                    &ffn_output[..5.min(ffn_output.len())]
                );
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }

            // IMP-1010-DEBUG: Check hidden after FFN residual
            if false {
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                let ffn_output_has_nan = ffn_output.iter().any(|x| x.is_nan());
                let ffn_output_sum: f32 = ffn_output.iter().sum();
                let ffn_output_max = ffn_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} ffn_out: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, ffn_output_sum, ffn_output_max, ffn_output_has_nan
                );
                let hidden_sum: f32 = hidden.iter().sum();
                let hidden_max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[IMP-1010] pos{} L{} final: sum={:.6e}, max={:.6e}, has_nan={}",
                    position, layer_idx, hidden_sum, hidden_max, hidden_has_nan
                );
            } else if position < 2 {
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                if hidden_has_nan {
                    let ffn_output_has_nan = ffn_output.iter().any(|x| x.is_nan());
                    let ffn_output_sum: f32 = ffn_output.iter().sum();
                    eprintln!(
                        "[IMP-1010] pos{} L{} ffn_out: sum={:.6e}, has_nan={}",
                        position, layer_idx, ffn_output_sum, ffn_output_has_nan
                    );
                    let hidden_sum: f32 = hidden.iter().sum();
                    eprintln!(
                        "[IMP-1010] pos{} L{} hidden after FFN: sum={:.6e}, has_nan={}",
                        position, layer_idx, hidden_sum, hidden_has_nan
                    );
                }
            }

            // PAR-049-DEBUG: Print hidden state after layer 0 and compute CPU reference (disabled for performance)
            // Re-enable by changing `false` to `true` for debugging
            #[allow(clippy::never_loop)]
            if false {
                eprintln!(
                    "[PAR-049] L0 GPU hidden[0..5]: {:?}",
                    &hidden[..5.min(hidden.len())]
                );

                // Compute CPU reference for layer 0
                // Start from embedding
                let cpu_hidden = self.model.embed(&[token_id]);
                let cpu_normed = ops::layer_norm(
                    &cpu_hidden,
                    &self.model.layers[0].attn_norm_weight,
                    self.model.layers[0].attn_norm_bias.as_deref(),
                    eps,
                );
                let cpu_qkv = self
                    .model
                    .qkv_matmul(&cpu_normed, &self.model.layers[0].qkv_weight)
                    .expect("CPU qkv");
                let mut cpu_q = cpu_qkv[0..hidden_dim].to_vec();
                let mut cpu_k = cpu_qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
                let cpu_v = cpu_qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();
                self.model.apply_rope(&mut cpu_q, 0, num_heads);
                self.model.apply_rope(&mut cpu_k, 0, num_kv_heads);

                // First token - expand V for GQA
                let cpu_attn_out = if num_kv_heads < num_heads {
                    let q_per_kv = num_heads / num_kv_heads;
                    let mut expanded = vec![0.0f32; hidden_dim];
                    for qh in 0..num_heads {
                        let kv_h = qh / q_per_kv;
                        expanded[qh * head_dim..(qh + 1) * head_dim]
                            .copy_from_slice(&cpu_v[kv_h * head_dim..(kv_h + 1) * head_dim]);
                    }
                    expanded
                } else {
                    cpu_v.clone()
                };

                let cpu_attn_proj = self
                    .model
                    .fused_matmul(&cpu_attn_out, &self.model.layers[0].attn_output_weight)
                    .expect("CPU attn proj");
                let mut cpu_h = cpu_hidden.clone();
                for i in 0..hidden_dim {
                    cpu_h[i] += cpu_attn_proj[i];
                }

                eprintln!(
                    "[PAR-049] L0 CPU hidden after attn[0..5]: {:?}",
                    &cpu_h[..5.min(cpu_h.len())]
                );

                // Compare attention residual state
                let hidden_after_attn: Vec<f32> =
                    hidden.iter().zip(&ffn_output).map(|(h, f)| h - f).collect();
                let max_diff_attn = hidden_after_attn
                    .iter()
                    .zip(cpu_h.iter())
                    .map(|(g, c)| (g - c).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("[PAR-049] L0 attn residual max_diff: {:.6e}", max_diff_attn);
            }
        }

        // Advance cache position
        cache.advance();

        // 3. Final layer norm (CPU - fast for single vector)
        // PAR-050: Use RMSNorm for LLaMA models
        let normed = if use_rmsnorm {
            ops::rms_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.config.eps,
            )
        } else {
            ops::layer_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.output_norm_bias.as_deref(),
                self.model.config.eps,
            )
        };

        // 4. LM head projection (GPU - IMP-1010, PAR-016: use pre-captured cache key)
        // Clone LM head weight to avoid borrow conflicts, but use stable cache key
        let lm_head_weight_data = self.model.lm_head_weight.data.clone();
        let lm_head_weight_in_dim = self.model.lm_head_weight.in_dim;
        let lm_head_weight_out_dim = self.model.lm_head_weight.out_dim;
        let lm_head_weight_qtype = self.model.lm_head_weight.qtype;
        let lm_head_weight = OwnedQuantizedTensor {
            data: lm_head_weight_data,
            in_dim: lm_head_weight_in_dim,
            out_dim: lm_head_weight_out_dim,
            qtype: lm_head_weight_qtype,
        };

        let mut logits =
            self.fused_matmul_cuda_with_key(&normed, &lm_head_weight, &lm_head_cache_key)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        // PAR-049-DEBUG: Compare final logits with CPU for position 1 (disabled for performance)
        // Re-enable by changing `false` to `true` for debugging
        // IMP-1010-DEBUG: Enable for all positions to debug garbage output
        #[allow(clippy::never_loop)]
        if false {
            // IMP-1010-DEBUG: Print hidden and normed stats
            let hidden_sum: f32 = hidden.iter().sum();
            let hidden_max = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let hidden_min = hidden.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!(
                "[IMP-1010] pos{} hidden: sum={:.6e}, min={:.6e}, max={:.6e}",
                position, hidden_sum, hidden_min, hidden_max
            );
            let normed_sum: f32 = normed.iter().sum();
            let normed_max = normed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let normed_min = normed.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!(
                "[IMP-1010] pos{} normed: sum={:.6e}, min={:.6e}, max={:.6e}",
                position, normed_sum, normed_min, normed_max
            );
            let cpu_logits = self.model.fused_matmul(&normed, &lm_head_weight)?;
            let max_diff = logits
                .iter()
                .zip(cpu_logits.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0f32, f32::max);
            let top5_gpu: Vec<_> = logits.iter().enumerate().map(|(i, &v)| (i, v)).fold(
                vec![(0usize, f32::MIN); 5],
                |mut acc, (i, v)| {
                    if v > acc[4].1 {
                        acc[4] = (i, v);
                        acc.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    acc
                },
            );
            let top5_cpu: Vec<_> = cpu_logits.iter().enumerate().map(|(i, &v)| (i, v)).fold(
                vec![(0usize, f32::MIN); 5],
                |mut acc, (i, v)| {
                    if v > acc[4].1 {
                        acc[4] = (i, v);
                        acc.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    acc
                },
            );
            eprintln!("[PAR-049] pos1 logits max_diff: {:.6e}", max_diff);
            eprintln!("[PAR-049] pos1 GPU top5: {:?}", top5_gpu);
            eprintln!("[PAR-049] pos1 CPU top5: {:?}", top5_cpu);
        }

        Ok(logits)
    }

    /// GPU-accelerated attention with KV cache using multi-head CUDA kernel (PARITY-044)
    ///
    /// Uses `CudaExecutor::flash_attention_multi_head` to process all heads in parallel.
    /// Memory layout: [n_heads, seq_len, head_dim]
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head (hidden_dim / num_heads)
    #[allow(clippy::too_many_arguments)]
    fn cuda_attention_with_cache(
        &mut self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        total_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = num_heads * head_dim;
        let cache_len = total_len - 1;

        // Build full K and V tensors for all heads: [n_heads, total_len, head_dim]
        let tensor_size = num_heads * total_len * head_dim;

        // For GPU multi-head attention, we need Q repeated across all positions
        // Q is [hidden_dim] = [n_heads * head_dim], expand to [n_heads, total_len, head_dim]
        let mut q_full = vec![0.0f32; tensor_size];
        let mut k_full = vec![0.0f32; tensor_size];
        let mut v_full = vec![0.0f32; tensor_size];

        // Reorganize from [seq_len, n_heads * head_dim] to [n_heads, seq_len, head_dim]
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let gpu_head_offset = head * total_len * head_dim;

            // Q: single query expanded to all positions (for proper broadcast)
            for pos in 0..total_len {
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                q_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&q[head_offset..head_offset + head_dim]);
            }

            // K: cached + current
            for pos in 0..cache_len {
                let cache_offset = pos * hidden_dim + head_offset;
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                k_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&k_cache[cache_offset..cache_offset + head_dim]);
            }
            // Current K
            let gpu_current_offset = gpu_head_offset + cache_len * head_dim;
            k_full[gpu_current_offset..gpu_current_offset + head_dim]
                .copy_from_slice(&current_k[head_offset..head_offset + head_dim]);

            // V: cached + current
            for pos in 0..cache_len {
                let cache_offset = pos * hidden_dim + head_offset;
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                v_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&v_cache[cache_offset..cache_offset + head_dim]);
            }
            // Current V
            v_full[gpu_current_offset..gpu_current_offset + head_dim]
                .copy_from_slice(&current_v[head_offset..head_offset + head_dim]);
        }

        // GPU multi-head attention using FlashAttention kernel
        let mut output_full = vec![0.0f32; tensor_size];
        self.executor
            .flash_attention_multi_head(
                &q_full,
                &k_full,
                &v_full,
                &mut output_full,
                total_len as u32,
                head_dim as u32,
                num_heads as u32,
                true, // causal masking for autoregressive decoding
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "flash_attention_multi_head".to_string(),
                reason: format!("CUDA attention failed: {e}"),
            })?;

        // Extract output for the last position and reorganize to [hidden_dim]
        let mut output = vec![0.0f32; hidden_dim];
        let last_pos = total_len - 1;
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let gpu_head_offset = head * total_len * head_dim;
            let gpu_pos_offset = gpu_head_offset + last_pos * head_dim;
            output[head_offset..head_offset + head_dim]
                .copy_from_slice(&output_full[gpu_pos_offset..gpu_pos_offset + head_dim]);
        }

        Ok(output)
    }

    /// Generate tokens using CUDA acceleration with KV cache (PARITY-044)
    ///
    /// Uses `forward_single_cuda_with_cache` for GPU-accelerated incremental decoding.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn forward_gpu_resident(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast, single lookup)
        let embedding = self.model.embed(&[token_id]);

        // PAR-060-DEBUG: Disabled for performance measurement
        // let embed_sum: f32 = embedding.iter().sum();
        // eprintln!("[PAR-060-DEBUG] Embedding sum: {:.6}", embed_sum);

        // 2. Fully GPU-resident forward: layers + output norm + LM head
        // PAR-054: Use CUDA graph-captured path for decode (reduces 280 launches to 1)
        // Only 2 syncs total: embedding upload + logits download
        let mut logits = vec![0.0f32; vocab_size];
        self.executor
            .forward_all_layers_gpu_to_logits_graphed(
                &embedding,
                &mut logits,
                position as u32,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size as u32,
                eps,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "forward_gpu_resident".to_string(),
                reason: format!("forward_all_layers_gpu_to_logits_graphed failed: {}", e),
            })?;

        // 3. Add LM head bias if present (CPU - fast)
        if let Some(ref bias) = self.model.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        // Advance cache position (for compatibility with cache-based generation)
        cache.advance();

        Ok(logits)
    }

    /// PAR-062: GPU-resident forward pass returning token ID directly
    ///
    /// Like `forward_gpu_resident` but uses GPU-side argmax for greedy sampling.
    /// Eliminates 600KB logits transfer per token, reducing to 4 bytes (token ID).
    ///
    /// # Performance Improvement
    ///
    /// - Before: Download 152064 x 4 = 600KB per token
    /// - After: Download 1 x 4 = 4 bytes per token
    /// - Expected speedup: ~1.2x overall throughput
    ///
    /// # Arguments
    ///
    /// * `token_id` - Input token
    /// * `cache` - KV cache (advanced but not used for logits)
    /// * `position` - Position in sequence
    ///
    /// # Returns
    ///
    /// Token ID with highest logit value (greedy sampling)
    ///
    /// # Errors
    ///
    /// Returns error if GPU operations fail or model has lm_head_bias (requires CPU path).
    pub fn forward_gpu_resident_to_token_id(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<u32> {
        // CORRECTNESS-013: Check if deterministic mode is requested
        // In this mode, download logits to CPU for argmax to ensure bit-exact
        // output matching between CPU and GPU inference paths.
        static CORRECTNESS_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_cpu_argmax = *CORRECTNESS_MODE.get_or_init(|| {
            std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // PAR-062: If model has LM head bias, fall back to CPU path
        // (bias addition requires CPU, so we'd download logits anyway)
        if self.model.lm_head_bias.is_some() || use_cpu_argmax {
            let logits = self.forward_gpu_resident(token_id, cache, position)?;
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32));
        }

        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast, single lookup)
        let embedding = self.model.embed(&[token_id]);

        // 2. Check if CUDA graph is captured; if not, use regular path first
        // The graphed path needs to be initialized via forward_all_layers_gpu_to_logits_graphed
        if !self.executor.has_decode_graph() {
            // First call - need to capture graph, use regular path
            let mut logits = vec![0.0f32; vocab_size];
            self.executor
                .forward_all_layers_gpu_to_logits_graphed(
                    &embedding,
                    &mut logits,
                    position as u32,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_gpu_resident_to_token_id".to_string(),
                    reason: format!("forward_all_layers_gpu_to_logits_graphed failed: {}", e),
                })?;

            cache.advance();
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32));
        }

        // 3. Use GPU argmax path - graph is captured, use optimized replay
        let next_token = self
            .executor
            .forward_graphed_replay_to_token_id(&embedding, position as u32, vocab_size as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "forward_gpu_resident_to_token_id".to_string(),
                reason: format!("forward_graphed_replay_to_token_id failed: {}", e),
            })?;

        cache.advance();
        Ok(next_token)
    }
}
