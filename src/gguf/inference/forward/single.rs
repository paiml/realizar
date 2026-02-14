//! Single-token forward pass with KV cache
//!
//! Contains forward_single_with_cache and forward_single_with_cache_adaptive.
//! These are the decode-phase entry points for autoregressive generation.

use crate::error::Result;
use crate::gguf::ops;
use crate::gguf::{
    DispatchMetrics, InferenceScratchBuffer, OwnedQuantizedKVCache, OwnedQuantizedModel,
    GGUF_TYPE_Q4_K,
};

impl OwnedQuantizedModel {
    /// Forward pass for a single token using KV cache (IMP-101c)
    ///
    /// This is O(n) per token instead of O(n²) due to KV cache reuse.
    ///
    /// # Arguments
    /// * `token_id` - Single input token ID
    /// * `cache` - Mutable reference to KV cache
    /// * `position` - Position in sequence for RoPE
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    pub fn forward_single_with_cache(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // DEBUG: Print hidden state after embedding
        let debug_forward = std::env::var("REALIZAR_DEBUG_FORWARD").is_ok();
        if debug_forward {
            let hidden_sum: f32 = hidden.iter().sum();
            eprintln!("[DEBUG-FORWARD] Token={}, Position={}", token_id, position);
            eprintln!(
                "[DEBUG-FORWARD] After embed: sum={:.6}, hidden[0..4]={:?}",
                hidden_sum,
                &hidden[..4.min(hidden.len())]
            );
        }

        // Detect if model uses RMSNorm (LLaMA-style) or LayerNorm (phi-2 style)
        // LLaMA models have ffn_gate_weight (SwiGLU) and no bias in norms
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // Pre-allocate attention output buffer - reused across all layers
        let mut attn_out_buffer = vec![0.0f32; hidden_dim];

        // GQA-DEBUG: Print embedding before layer 0
        if std::env::var("CPU_DEBUG").is_ok() {
            let embed_sum: f32 = hidden.iter().sum();
            let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden.len() as f32).sqrt();
            eprintln!(
                "[GQA-DEBUG-CPU-EMBED] Embedding before L0: first 5 = {:?}, sum={:.4}, rms={:.4}",
                &hidden[..5.min(hidden.len())],
                embed_sum,
                rms
            );
        }

        // PMAT-114: Embedding trace for Five-Whys comparison with APR
        if std::env::var("APR_TRACE_LAYERS").is_ok() {
            // PMAT-114: Trace token ID being processed
            eprintln!(
                "[PMAT-114-GGUF] Token ID: {}, position: {}",
                token_id, position
            );

            let sum: f32 = hidden.iter().sum();
            let mean = sum / hidden_dim as f32;
            let min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[PMAT-114-GGUF] After embed: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                mean,
                min,
                max,
                &hidden[..5.min(hidden.len())]
            );
        }

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a+2b. Fused attention layer norm + QKV projection
            // For RMSNorm models: fuse norm + matmul to eliminate intermediate allocation
            // For LayerNorm models: use separate operations (has bias)
            let mut qkv = if use_rmsnorm {
                self.fused_rmsnorm_qkv_matmul(
                    &hidden,
                    &layer.attn_norm_weight,
                    self.config.eps,
                    &layer.qkv_weight,
                )?
            } else {
                let normed = ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                );
                self.qkv_matmul(&normed, &layer.qkv_weight)?
            };

            // PMAT-114: Trace QKV BEFORE bias to isolate the issue
            if std::env::var("APR_TRACE_LAYERS").is_ok() && layer_idx == 0 {
                let num_kv_heads_trace = self.config.num_kv_heads;
                let head_dim_trace = hidden_dim / self.config.num_heads;
                let kv_dim_trace = num_kv_heads_trace * head_dim_trace;
                let k = &qkv[hidden_dim..hidden_dim + kv_dim_trace];
                let k_mean: f32 = k.iter().sum::<f32>() / kv_dim_trace as f32;
                eprintln!("[PMAT-114-GGUF] L0 K BEFORE bias: mean={:.6}", k_mean);
            }

            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            // Q: [hidden_dim] = [num_heads * head_dim]
            // K: [kv_dim] = [num_kv_heads * head_dim]
            // V: [kv_dim] = [num_kv_heads * head_dim]
            // Optimization: apply RoPE in-place to avoid Q/K copies
            let num_kv_heads = self.config.num_kv_heads;
            let head_dim = hidden_dim / self.config.num_heads;
            let kv_dim = num_kv_heads * head_dim;

            // PMAT-114: Trace QKV for layer 0 (before RoPE)
            if std::env::var("APR_TRACE_LAYERS").is_ok() && layer_idx == 0 {
                // Check if bias exists
                eprintln!(
                    "[PMAT-114-GGUF] L0 has_qkv_bias={}",
                    layer.qkv_bias.is_some()
                );
                if let Some(ref bias) = layer.qkv_bias {
                    let k_bias = &bias[hidden_dim..hidden_dim + kv_dim];
                    let k_bias_mean: f32 = k_bias.iter().sum::<f32>() / kv_dim as f32;
                    eprintln!(
                        "[PMAT-114-GGUF] L0 K bias mean={:.6}, first5={:?}",
                        k_bias_mean,
                        &k_bias[..5.min(kv_dim)]
                    );
                }

                let q = &qkv[0..hidden_dim];
                let k = &qkv[hidden_dim..hidden_dim + kv_dim];
                let v = &qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim];
                let q_mean: f32 = q.iter().sum::<f32>() / hidden_dim as f32;
                let k_mean: f32 = k.iter().sum::<f32>() / kv_dim as f32;
                let v_mean: f32 = v.iter().sum::<f32>() / kv_dim as f32;
                eprintln!("[PMAT-114-GGUF] L0 after QKV (pre-RoPE): Q mean={:.6}, K mean={:.6}, V mean={:.6}", q_mean, k_mean, v_mean);
                eprintln!("[PMAT-114-GGUF] L0 Q first5={:?}", q.get(..5).unwrap_or(&[]));
            }

            // Apply RoPE in-place to Q and K within QKV buffer
            self.apply_rope(&mut qkv[0..hidden_dim], position, self.config.num_heads);
            self.apply_rope(
                &mut qkv[hidden_dim..hidden_dim + kv_dim],
                position,
                num_kv_heads,
            );

            // Use slices to avoid copies (only copy K for cache storage)
            let q = &qkv[0..hidden_dim];
            let k = &qkv[hidden_dim..hidden_dim + kv_dim];
            let v = &qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim];

            // 2d. Get cached K/V and compute attention with GQA support
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            // Use pre-allocated attention output buffer (reused across layers)
            if k_cache.is_empty() {
                // First token - no cache yet, output is just weighted V
                // With single query and single K/V, need to expand V for all Q heads
                let q_per_kv = self.config.num_heads / num_kv_heads;
                for q_head in 0..self.config.num_heads {
                    let kv_head = q_head / q_per_kv;
                    let v_start = kv_head * head_dim;
                    let out_start = q_head * head_dim;
                    attn_out_buffer[out_start..out_start + head_dim]
                        .copy_from_slice(&v[v_start..v_start + head_dim]);
                }
            } else {
                // Use cached K/V for attention with GQA
                // Uses pre-allocated buffer to avoid 704 Vec allocations per token
                self.attention_with_cache_gqa_into(q, k_cache, v_cache, k, v, &mut attn_out_buffer);

                // CORRECTNESS-013: Debug CPU attention output for layer 0 at position 1+
                if layer_idx == 0 && position >= 1 && std::env::var("CPU_DEBUG").is_ok() {
                    eprintln!(
                        "[CORRECTNESS-013-CPU] Layer 0 attention output at pos={}, first 10: {:?}",
                        position,
                        &attn_out_buffer[..10.min(attn_out_buffer.len())]
                    );
                    for h in 0..3 {
                        let start = h * head_dim;
                        eprintln!(
                            "[CORRECTNESS-013-CPU] Head {} first 5: {:?}",
                            h,
                            &attn_out_buffer[start..start + 5]
                        );
                    }
                }
            }

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, k, v);

            // 2f. Attention output projection
            let mut attn_output = self.fused_matmul(&attn_out_buffer, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // 2h+2i. FFN with optional layer norm and SwiGLU/GELU activation
            // For RMSNorm + SwiGLU: fuse norm + up/gate matmuls to eliminate intermediate
            let ffn_activated = match (&layer.ffn_norm_weight, &layer.ffn_gate_weight) {
                // Fused path: RMSNorm + SwiGLU (LLaMA, TinyLlama, Mistral, etc.)
                (Some(ref ffn_norm), Some(ref gate_weight)) if use_rmsnorm => {
                    let (mut ffn_up, mut ffn_gate) = self.fused_rmsnorm_ffn_up_gate(
                        &hidden,
                        ffn_norm,
                        self.config.eps,
                        &layer.ffn_up_weight,
                        gate_weight,
                    )?;

                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    // SwiGLU: silu(gate) * up
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                },

                // Non-fused SwiGLU (LayerNorm models with gate)
                (ffn_norm_opt, Some(ref gate_weight)) => {
                    let ffn_input = if let Some(ref ffn_norm) = ffn_norm_opt {
                        ops::layer_norm(
                            &hidden,
                            ffn_norm,
                            layer.ffn_norm_bias.as_deref(),
                            self.config.eps,
                        )
                    } else {
                        hidden.clone()
                    };

                    let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }

                    let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    // SwiGLU: silu(gate) * up
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                },

                // GELU path (phi-2, GPT-2, etc.) - no gate weight
                (ffn_norm_opt, None) => {
                    let ffn_input = if let Some(ref ffn_norm) = ffn_norm_opt {
                        if use_rmsnorm {
                            ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                        } else {
                            ops::layer_norm(
                                &hidden,
                                ffn_norm,
                                layer.ffn_norm_bias.as_deref(),
                                self.config.eps,
                            )
                        }
                    } else {
                        hidden.clone()
                    };

                    let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_hidden, bias);
                    }
                    ops::gelu(&mut ffn_hidden);
                    ffn_hidden
                },
            };

            // 2j. FFN down projection
            let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }

            // DEBUG: Print hidden state after first layer
            if debug_forward && layer_idx == 0 {
                let hidden_sum: f32 = hidden.iter().sum();
                eprintln!(
                    "[DEBUG-FORWARD] After layer 0: sum={:.6}, hidden[0..4]={:?}",
                    hidden_sum,
                    &hidden[..4.min(hidden.len())]
                );
            }

            // GQA-DEBUG: Print hidden state after layer 0 for CPU/GPU comparison
            if std::env::var("CPU_DEBUG").is_ok() && layer_idx == 0 {
                let hidden_sum: f32 = hidden.iter().sum();
                let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden.len() as f32).sqrt();
                eprintln!(
                    "[GQA-DEBUG-CPU-L0] After layer 0: first 5 = {:?}, sum={:.4}, rms={:.4}",
                    &hidden[..5.min(hidden.len())],
                    hidden_sum,
                    rms
                );
            }

            // PMAT-114: Layer tracing for Five-Whys comparison with APR
            if std::env::var("APR_TRACE_LAYERS").is_ok()
                && (layer_idx < 2 || layer_idx == self.layers.len() - 1)
            {
                let sum: f32 = hidden.iter().sum();
                let mean = sum / hidden_dim as f32;
                let min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[PMAT-114-GGUF] After layer {}: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                    layer_idx,
                    mean,
                    min,
                    max,
                    &hidden[..5.min(hidden.len())]
                );
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // DEBUG: Print hidden state before LM head
        if debug_forward {
            let hidden_sum: f32 = hidden.iter().sum();
            let hidden_max = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let hidden_min = hidden.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!(
                "[DEBUG-FORWARD] Hidden after all layers: sum={:.4}, min={:.4}, max={:.4}",
                hidden_sum, hidden_min, hidden_max
            );
            eprintln!(
                "[DEBUG-FORWARD] Hidden[0..8]: {:?}",
                &hidden[..8.min(hidden.len())]
            );
            eprintln!(
                "[DEBUG-LM-HEAD] lm_head_weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
                self.lm_head_weight.in_dim,
                self.lm_head_weight.out_dim,
                self.lm_head_weight.qtype,
                self.lm_head_weight.data.len()
            );
            eprintln!(
                "[DEBUG-LM-HEAD] First 16 bytes of lm_head data: {:02x?}",
                &self.lm_head_weight.data[..16.min(self.lm_head_weight.data.len())]
            );
            eprintln!(
                "[DEBUG-LM-HEAD] output_norm_weight[0..4]: {:?}",
                &self.output_norm_weight[..4.min(self.output_norm_weight.len())]
            );
        }

        // 3+4. Fused final layer norm + LM head projection
        // For RMSNorm models: fuse norm + matmul to eliminate intermediate allocation
        let mut logits = if use_rmsnorm {
            self.fused_rmsnorm_lm_head(&hidden)?
        } else {
            let normed = ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            );
            self.fused_matmul(&normed, &self.lm_head_weight)?
        };

        // DEBUG: Verify Q8_0 matmul by manual computation
        if debug_forward {
            // Get the normalized hidden state
            let normed = ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps);
            eprintln!(
                "[DEBUG-VERIFY] Normed hidden[0..8]: {:?}",
                &normed[..8.min(normed.len())]
            );

            // Manual dequantize row 0 of LM head weight
            const Q8_0_BLOCK_BYTES: usize = 34;
            const Q8_0_BLOCK_SIZE: usize = 32;
            let blocks_per_row = self.lm_head_weight.in_dim.div_ceil(Q8_0_BLOCK_SIZE);
            let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

            // Dequantize row 0 (token 0's projection weights)
            let row0_data = &self.lm_head_weight.data[0..bytes_per_row];
            let mut row0_f32 = vec![0.0f32; self.lm_head_weight.in_dim];
            for block_idx in 0..blocks_per_row {
                let block_start = block_idx * Q8_0_BLOCK_BYTES;
                let block = &row0_data[block_start..block_start + Q8_0_BLOCK_BYTES];
                let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                for j in 0..32 {
                    let idx = block_idx * 32 + j;
                    if idx >= self.lm_head_weight.in_dim {
                        break;
                    }
                    row0_f32[idx] = (block[2 + j] as i8 as f32) * scale;
                }
            }
            eprintln!(
                "[DEBUG-VERIFY] LM head row 0 (dequantized) first 8: {:?}",
                &row0_f32[..8.min(row0_f32.len())]
            );

            // Compute dot product manually
            let manual_logit0: f32 = normed.iter().zip(row0_f32.iter()).map(|(a, b)| a * b).sum();
            eprintln!("[DEBUG-VERIFY] Manual logits[0] = {:.6}", manual_logit0);
            eprintln!("[DEBUG-VERIFY] Computed logits[0] = {:.6}", logits[0]);
            eprintln!(
                "[DEBUG-VERIFY] Difference = {:.6}",
                (manual_logit0 - logits[0]).abs()
            );

            // Check top tokens
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!(
                "[DEBUG-VERIFY] Top 5 tokens: {:?}",
                &indexed[..5.min(indexed.len())]
            );
        }

        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Single-token forward pass with pre-allocated scratch buffers
    ///
    /// Uses OwnedInferenceScratchBuffer to eliminate per-token allocations.
    /// For Qwen2.5-0.5B, this saves ~40KB of allocations per token.
    ///
    /// # Arguments
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for incremental decoding
    ///
    /// Forward pass with adaptive CPU/GPU attention selection (IMP-124)
    ///
    /// This variant of `forward_single_with_cache` uses `adaptive_attention_with_cache`
    /// to automatically select between CPU and GPU backends based on cache length.
    /// It also records dispatch decisions to the provided metrics tracker.
    ///
    /// # Arguments
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for incremental decoding
    /// * `position` - Position in sequence
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    #[cfg(feature = "gpu")]
    pub fn forward_single_with_cache_adaptive(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // Detect if model uses RMSNorm (LLaMA-style) or LayerNorm (phi-2 style)
        // LLaMA models have ffn_gate_weight (SwiGLU) and no bias in norms
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // GQA dimensions
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / self.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // PARITY-113: Track CUDA kernel count for GPU dispatch metrics
        #[cfg(feature = "cuda")]
        let cuda_enabled = self.cuda_enabled();

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm (RMSNorm for LLaMA, LayerNorm for others)
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };

            // 2b. QKV projection
            // PARITY-113: Record GPU dispatch when CUDA path is used for matmul
            #[cfg(feature = "cuda")]
            if cuda_enabled {
                let start = std::time::Instant::now();
                let qkv_result = self.qkv_matmul(&normed, &layer.qkv_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                let mut qkv = qkv_result;
                if let Some(ref bias) = layer.qkv_bias {
                    ops::add_bias(&mut qkv, bias);
                }

                // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
                let mut q = qkv[0..hidden_dim].to_vec();
                let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
                let v = qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();

                self.apply_rope(&mut q, position, self.config.num_heads);
                self.apply_rope(&mut k, position, num_kv_heads);

                // 2d. Get cached K/V and compute attention with GQA support
                let k_cache = cache.get_k(layer_idx);
                let v_cache = cache.get_v(layer_idx);

                let attn_out = if k_cache.is_empty() {
                    // First token - expand V for all Q heads (GQA)
                    let mut expanded_v = vec![0.0f32; hidden_dim];
                    let q_per_kv = self.config.num_heads / num_kv_heads;
                    for q_head in 0..self.config.num_heads {
                        let kv_head = q_head / q_per_kv;
                        let v_start = kv_head * head_dim;
                        let out_start = q_head * head_dim;
                        expanded_v[out_start..out_start + head_dim]
                            .copy_from_slice(&v[v_start..v_start + head_dim]);
                    }
                    expanded_v
                } else {
                    let start = std::time::Instant::now();
                    let result =
                        self.adaptive_attention_with_cache(&q, k_cache, v_cache, &k, &v)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    result
                };

                // 2e. Store K and V in cache
                cache.append(layer_idx, &k, &v);

                // 2f. Attention output projection
                let start = std::time::Instant::now();
                let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                if let Some(ref bias) = layer.attn_output_bias {
                    ops::add_bias(&mut attn_output, bias);
                }

                // 2g. Residual connection
                for i in 0..hidden_dim {
                    hidden[i] += attn_output[i];
                }

                // 2h. Pre-FFN layer norm (LLaMA uses separate ffn_norm with RMSNorm)
                let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                    if use_rmsnorm {
                        ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                    } else {
                        ops::layer_norm(
                            &hidden,
                            ffn_norm,
                            layer.ffn_norm_bias.as_deref(),
                            self.config.eps,
                        )
                    }
                } else {
                    hidden.clone()
                };

                // 2i. FFN with SwiGLU or GELU activation
                let ffn_activated = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                    // SwiGLU path
                    let start = std::time::Instant::now();
                    let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }

                    let start = std::time::Instant::now();
                    let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }

                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    ffn_gate
                } else {
                    // GELU path
                    let start = std::time::Instant::now();
                    let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_hidden, bias);
                    }
                    ops::gelu(&mut ffn_hidden);
                    ffn_hidden
                };

                // 2j. FFN down projection
                let start = std::time::Instant::now();
                let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut ffn_output, bias);
                }

                // Residual
                for i in 0..hidden_dim {
                    hidden[i] += ffn_output[i];
                }

                continue;
            }

            // CPU path (non-CUDA)
            let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            let mut q = qkv[0..hidden_dim].to_vec();
            let mut k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
            let v = qkv[hidden_dim + kv_dim..hidden_dim + 2 * kv_dim].to_vec();

            self.apply_rope(&mut q, position, self.config.num_heads);
            self.apply_rope(&mut k, position, num_kv_heads);

            // 2d. Get cached K/V and compute attention with adaptive dispatch
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            let attn_out = if k_cache.is_empty() {
                // First token - expand V for all Q heads (GQA)
                let mut expanded_v = vec![0.0f32; hidden_dim];
                let q_per_kv = self.config.num_heads / num_kv_heads;
                for q_head in 0..self.config.num_heads {
                    let kv_head = q_head / q_per_kv;
                    let v_start = kv_head * head_dim;
                    let out_start = q_head * head_dim;
                    expanded_v[out_start..out_start + head_dim]
                        .copy_from_slice(&v[v_start..v_start + head_dim]);
                }
                expanded_v
            } else {
                // Use adaptive attention with metrics tracking
                let cache_len = k_cache.len() / kv_dim;
                const GPU_CACHE_LEN_THRESHOLD: usize = 64;

                if cache_len >= GPU_CACHE_LEN_THRESHOLD {
                    let start = std::time::Instant::now();
                    let result =
                        self.adaptive_attention_with_cache(&q, k_cache, v_cache, &k, &v)?;
                    metrics.record_gpu_dispatch();
                    metrics.record_gpu_latency(start.elapsed());
                    result
                } else {
                    let start = std::time::Instant::now();
                    let result = self.attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v);
                    metrics.record_cpu_dispatch();
                    metrics.record_cpu_latency(start.elapsed());
                    result
                }
            };

            // 2e. Store K and V in cache for future tokens
            cache.append(layer_idx, &k, &v);

            // 2f. Attention output projection
            let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // 2h. Pre-FFN layer norm (LLaMA uses separate ffn_norm with RMSNorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                } else {
                    ops::layer_norm(
                        &hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    )
                }
            } else {
                hidden.clone()
            };

            // 2i. FFN with SwiGLU or GELU activation
            let ffn_activated = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU path
                let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_up, bias);
                }

                let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                if let Some(ref bias) = layer.ffn_gate_bias {
                    ops::add_bias(&mut ffn_gate, bias);
                }

                ops::silu(&mut ffn_gate);
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }
                ffn_gate
            } else {
                // GELU path
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);
                ffn_hidden
            };

            // 2j. FFN down projection
            let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }
        }

        // Advance cache position after processing all layers
        cache.advance();

        // 3. Final layer norm (RMSNorm for LLaMA, LayerNorm for others)
        let normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };

        // 4. LM head projection
        // PARITY-113: Record GPU dispatch for LM head when CUDA is enabled
        #[cfg(feature = "cuda")]
        if cuda_enabled {
            let start = std::time::Instant::now();
            let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
            metrics.record_gpu_dispatch();
            metrics.record_gpu_latency(start.elapsed());
            if let Some(ref bias) = self.lm_head_bias {
                ops::add_bias(&mut logits, bias);
            }
            return Ok(logits);
        }

        let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Zero-allocation forward pass using scratch buffers (IMP-131)
    ///
    /// All intermediate results are written to pre-allocated scratch buffers.
    /// Output logits are stored in `scratch.logits`.
    pub(crate) fn forward_single_with_scratch(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        scratch: &mut InferenceScratchBuffer,
    ) -> Result<()> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // Detect architecture
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // 1. Token embedding lookup → scratch.hidden
        self.embed_into(token_id, &mut scratch.hidden);

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm → scratch.normed
            if use_rmsnorm {
                ops::rms_norm_into(
                    &scratch.hidden,
                    &layer.attn_norm_weight,
                    self.config.eps,
                    &mut scratch.normed,
                );
            } else {
                ops::layer_norm_into(
                    &scratch.hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                    &mut scratch.normed,
                );
            }

            // 2b. QKV projection → scratch.qkv (zero-allocation via P1-REV)
            // PAR-126: Fix GQA dimension issue - use config instead of q_dim() which
            // incorrectly assumes Q=K=V for fused weights
            let num_kv_heads = self.config.num_kv_heads;
            let head_dim = hidden_dim / self.config.num_heads;
            let kv_dim = num_kv_heads * head_dim;
            // Q uses all heads, K/V use only kv_heads (GQA)
            let q_dim = hidden_dim;
            let k_dim = kv_dim;
            let v_dim = kv_dim;
            let qkv_dim = q_dim + k_dim + v_dim;

            // PAR-126: Pre-quantize normalized hidden to Q8K for VNNI-accelerated matmul
            // This allows reusing quantized activations for QKV projection
            // NOTE: Q8K requires hidden_dim to be multiple of 256. For smaller models
            // like 0.5B (hidden=896), fall back to f32 path.
            let use_q8k_path = hidden_dim.is_multiple_of(256);

            if use_q8k_path {
                use crate::quantize::quantize_activations_q8k_into;
                let hidden_sb = hidden_dim / 256;
                quantize_activations_q8k_into(
                    &scratch.normed[..hidden_dim],
                    &mut scratch.q8k_hidden_scales[..hidden_sb],
                    &mut scratch.q8k_hidden_quants[..hidden_dim],
                )?;

                // Write directly to scratch.qkv, using Q8K-accelerated path
                self.qkv_matmul_q8k_into(
                    &scratch.normed,
                    &layer.qkv_weight,
                    &mut scratch.qkv[..qkv_dim],
                    &scratch.q8k_hidden_scales[..hidden_sb],
                    &scratch.q8k_hidden_quants[..hidden_dim],
                )?;
            } else {
                // Fall back to f32 path for non-256-aligned hidden dims
                self.qkv_matmul_into(
                    &scratch.normed,
                    &layer.qkv_weight,
                    &mut scratch.qkv[..qkv_dim],
                )?;
            }

            // Copy from scratch.qkv to individual Q, K, V buffers
            scratch.q[..q_dim].copy_from_slice(&scratch.qkv[..q_dim]);
            scratch.k[..k_dim].copy_from_slice(&scratch.qkv[q_dim..q_dim + k_dim]);
            scratch.v[..v_dim].copy_from_slice(&scratch.qkv[q_dim + k_dim..qkv_dim]);

            // Add bias if present
            if let Some(ref bias) = layer.qkv_bias {
                for i in 0..q_dim {
                    scratch.q[i] += bias[i];
                }
                for i in 0..k_dim {
                    scratch.k[i] += bias[q_dim + i];
                }
                for i in 0..v_dim {
                    scratch.v[i] += bias[q_dim + k_dim + i];
                }
            }

            // Apply RoPE
            self.apply_rope(&mut scratch.q[..q_dim], position, self.config.num_heads);
            self.apply_rope(&mut scratch.k[..k_dim], position, self.config.num_kv_heads);

            // 2c. Compute attention
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            if k_cache.is_empty() {
                // First token - expand V if GQA
                if self.config.num_kv_heads < self.config.num_heads {
                    let head_dim = hidden_dim / self.config.num_heads;
                    let group_size = self.config.num_heads / self.config.num_kv_heads;
                    for h in 0..self.config.num_heads {
                        let kv_head = h / group_size;
                        let src_start = kv_head * head_dim;
                        let dst_start = h * head_dim;
                        scratch.attn_out[dst_start..dst_start + head_dim]
                            .copy_from_slice(&scratch.v[src_start..src_start + head_dim]);
                    }
                } else {
                    scratch.attn_out[..hidden_dim].copy_from_slice(&scratch.v[..hidden_dim]);
                }
            } else {
                self.attention_with_cache_gqa_into(
                    &scratch.q[..q_dim],
                    k_cache,
                    v_cache,
                    &scratch.k[..k_dim],
                    &scratch.v[..v_dim],
                    &mut scratch.attn_out,
                );
            }

            // Store K, V in cache
            cache.append(layer_idx, &scratch.k[..k_dim], &scratch.v[..v_dim]);

            // 2d. Attention output projection → scratch.attn_proj
            // PAR-128: Use Q8K-accelerated path for attention output projection
            // attn_out is hidden_dim sized, reuse hidden Q8K buffers
            let use_q8k_attn_out = use_q8k_path && layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K;

            if use_q8k_attn_out {
                use crate::quantize::{
                    fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
                };
                let hidden_sb = hidden_dim / 256;
                // Quantize attention output to Q8K (reuse hidden Q8K buffers)
                quantize_activations_q8k_into(
                    &scratch.attn_out[..hidden_dim],
                    &mut scratch.q8k_hidden_scales[..hidden_sb],
                    &mut scratch.q8k_hidden_quants[..hidden_dim],
                )?;
                fused_q4k_q8k_parallel_matvec_into(
                    &layer.attn_output_weight.data,
                    &scratch.q8k_hidden_scales[..hidden_sb],
                    &scratch.q8k_hidden_quants[..hidden_dim],
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                    &mut scratch.attn_proj,
                )?;
            } else {
                self.fused_matmul_into(
                    &scratch.attn_out[..hidden_dim],
                    &layer.attn_output_weight,
                    &mut scratch.attn_proj,
                )?;
            }
            if let Some(ref bias) = layer.attn_output_bias {
                for i in 0..hidden_dim {
                    scratch.attn_proj[i] += bias[i];
                }
            }

            // 2e. Residual connection
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.attn_proj[i];
            }

            // 2f. Pre-FFN layer norm → scratch.normed
            if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    ops::rms_norm_into(
                        &scratch.hidden,
                        ffn_norm,
                        self.config.eps,
                        &mut scratch.normed,
                    );
                } else {
                    ops::layer_norm_into(
                        &scratch.hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                        &mut scratch.normed,
                    );
                }
            } else {
                scratch.normed[..hidden_dim].copy_from_slice(&scratch.hidden[..hidden_dim]);
            }

            // 2g. FFN
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU path (LLaMA)
                // PAR-126: Use Q8K-accelerated path only if hidden_dim is 256-aligned
                if use_q8k_path {
                    // Pre-quantize normed hidden to Q8K for VNNI-accelerated FFN matmul
                    // Quantize once, reuse for both up and gate matmuls
                    use crate::quantize::quantize_activations_q8k_into;
                    let hidden_sb = hidden_dim / 256;
                    quantize_activations_q8k_into(
                        &scratch.normed[..hidden_dim],
                        &mut scratch.q8k_hidden_scales[..hidden_sb],
                        &mut scratch.q8k_hidden_quants[..hidden_dim],
                    )?;

                    // Use fused FFN up+gate kernel to eliminate rayon::join overhead
                    // This reduces parallel region spawns from 2 to 1 per layer
                    let up_weight = &layer.ffn_up_weight;
                    let q8k_scales = &scratch.q8k_hidden_scales[..hidden_sb];
                    let q8k_quants = &scratch.q8k_hidden_quants[..hidden_dim];

                    // Check if both weights are Q4K for fused path
                    if up_weight.qtype == GGUF_TYPE_Q4_K && gate_weight.qtype == GGUF_TYPE_Q4_K {
                        use crate::quantize::fused_q4k_q8k_ffn_up_gate_into;
                        fused_q4k_q8k_ffn_up_gate_into(
                            &up_weight.data,
                            &gate_weight.data,
                            q8k_scales,
                            q8k_quants,
                            up_weight.in_dim,
                            up_weight.out_dim,
                            &mut scratch.ffn_up,
                            &mut scratch.ffn_gate,
                        )?;
                    } else {
                        // Fallback to separate matmuls if not both Q4K
                        use crate::quantize::fused_q4k_q8k_parallel_matvec_into;
                        let (up_result, gate_result) = rayon::join(
                            || {
                                if up_weight.qtype == GGUF_TYPE_Q4_K {
                                    fused_q4k_q8k_parallel_matvec_into(
                                        &up_weight.data,
                                        q8k_scales,
                                        q8k_quants,
                                        up_weight.in_dim,
                                        up_weight.out_dim,
                                        &mut scratch.ffn_up,
                                    )
                                } else {
                                    self.fused_matmul_into(
                                        &scratch.normed[..hidden_dim],
                                        up_weight,
                                        &mut scratch.ffn_up,
                                    )
                                }
                            },
                            || {
                                if gate_weight.qtype == GGUF_TYPE_Q4_K {
                                    fused_q4k_q8k_parallel_matvec_into(
                                        &gate_weight.data,
                                        q8k_scales,
                                        q8k_quants,
                                        gate_weight.in_dim,
                                        gate_weight.out_dim,
                                        &mut scratch.ffn_gate,
                                    )
                                } else {
                                    self.fused_matmul_into(
                                        &scratch.normed[..hidden_dim],
                                        gate_weight,
                                        &mut scratch.ffn_gate,
                                    )
                                }
                            },
                        );
                        up_result?;
                        gate_result?;
                    }
                } else {
                    // Fall back to f32 path for non-256-aligned hidden dims
                    let up_weight = &layer.ffn_up_weight;
                    let (up_result, gate_result) = rayon::join(
                        || {
                            self.fused_matmul_into(
                                &scratch.normed[..hidden_dim],
                                up_weight,
                                &mut scratch.ffn_up,
                            )
                        },
                        || {
                            self.fused_matmul_into(
                                &scratch.normed[..hidden_dim],
                                gate_weight,
                                &mut scratch.ffn_gate,
                            )
                        },
                    );
                    up_result?;
                    gate_result?;
                }

                if let Some(ref bias) = layer.ffn_up_bias {
                    for i in 0..intermediate_dim {
                        scratch.ffn_up[i] += bias[i];
                    }
                }
                if let Some(ref bias) = layer.ffn_gate_bias {
                    for i in 0..intermediate_dim {
                        scratch.ffn_gate[i] += bias[i];
                    }
                }

                // SiLU on gate, multiply with up
                ops::silu(&mut scratch.ffn_gate[..intermediate_dim]);
                for i in 0..intermediate_dim {
                    scratch.ffn_gate[i] *= scratch.ffn_up[i];
                }

                // PAR-127: Use Q8K-accelerated FFN down projection for Q4K weights
                // Q6K uses f32 path since Q8K conversion overhead > bandwidth savings
                let use_q8k_down = intermediate_dim.is_multiple_of(256)
                    && layer.ffn_down_weight.qtype == GGUF_TYPE_Q4_K;

                if use_q8k_down {
                    use crate::quantize::{
                        fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
                    };
                    let inter_sb = intermediate_dim / 256;
                    quantize_activations_q8k_into(
                        &scratch.ffn_gate[..intermediate_dim],
                        &mut scratch.q8k_inter_scales[..inter_sb],
                        &mut scratch.q8k_inter_quants[..intermediate_dim],
                    )?;
                    fused_q4k_q8k_parallel_matvec_into(
                        &layer.ffn_down_weight.data,
                        &scratch.q8k_inter_scales[..inter_sb],
                        &scratch.q8k_inter_quants[..intermediate_dim],
                        layer.ffn_down_weight.in_dim,
                        layer.ffn_down_weight.out_dim,
                        &mut scratch.ffn_down,
                    )?;
                } else {
                    self.fused_matmul_into(
                        &scratch.ffn_gate[..intermediate_dim],
                        &layer.ffn_down_weight,
                        &mut scratch.ffn_down,
                    )?;
                }
                if let Some(ref bias) = layer.ffn_down_bias {
                    for i in 0..hidden_dim {
                        scratch.ffn_down[i] += bias[i];
                    }
                }
            } else {
                // GELU path (phi-2)
                // PAR-129: Use Q8K-accelerated FFN for GELU models (Q4K only)
                let use_q8k_gelu_up = use_q8k_path && layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K;
                let use_q8k_gelu_down = intermediate_dim.is_multiple_of(256)
                    && layer.ffn_down_weight.qtype == GGUF_TYPE_Q4_K;

                if use_q8k_gelu_up {
                    // Reuse already-quantized hidden from QKV (scratch.q8k_hidden_*)
                    use crate::quantize::fused_q4k_q8k_parallel_matvec_into;
                    let hidden_sb = hidden_dim / 256;
                    fused_q4k_q8k_parallel_matvec_into(
                        &layer.ffn_up_weight.data,
                        &scratch.q8k_hidden_scales[..hidden_sb],
                        &scratch.q8k_hidden_quants[..hidden_dim],
                        layer.ffn_up_weight.in_dim,
                        layer.ffn_up_weight.out_dim,
                        &mut scratch.ffn_up,
                    )?;
                } else {
                    self.fused_matmul_into(
                        &scratch.normed[..hidden_dim],
                        &layer.ffn_up_weight,
                        &mut scratch.ffn_up,
                    )?;
                }
                if let Some(ref bias) = layer.ffn_up_bias {
                    for i in 0..intermediate_dim {
                        scratch.ffn_up[i] += bias[i];
                    }
                }
                ops::gelu(&mut scratch.ffn_up[..intermediate_dim]);

                if use_q8k_gelu_down {
                    use crate::quantize::{
                        fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
                    };
                    let inter_sb = intermediate_dim / 256;
                    quantize_activations_q8k_into(
                        &scratch.ffn_up[..intermediate_dim],
                        &mut scratch.q8k_inter_scales[..inter_sb],
                        &mut scratch.q8k_inter_quants[..intermediate_dim],
                    )?;
                    fused_q4k_q8k_parallel_matvec_into(
                        &layer.ffn_down_weight.data,
                        &scratch.q8k_inter_scales[..inter_sb],
                        &scratch.q8k_inter_quants[..intermediate_dim],
                        layer.ffn_down_weight.in_dim,
                        layer.ffn_down_weight.out_dim,
                        &mut scratch.ffn_down,
                    )?;
                } else {
                    self.fused_matmul_into(
                        &scratch.ffn_up[..intermediate_dim],
                        &layer.ffn_down_weight,
                        &mut scratch.ffn_down,
                    )?;
                }
                if let Some(ref bias) = layer.ffn_down_bias {
                    for i in 0..hidden_dim {
                        scratch.ffn_down[i] += bias[i];
                    }
                }
            }

            // 2h. FFN residual
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_down[i];
            }
        }

        // 3. Final layer norm → scratch.normed
        if use_rmsnorm {
            ops::rms_norm_into(
                &scratch.hidden,
                &self.output_norm_weight,
                self.config.eps,
                &mut scratch.normed,
            );
        } else {
            ops::layer_norm_into(
                &scratch.hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
                &mut scratch.normed,
            );
        }

        // 4. LM head → scratch.logits
        self.fused_matmul_into(
            &scratch.normed[..hidden_dim],
            &self.lm_head_weight,
            &mut scratch.logits,
        )?;

        Ok(())
    }
}
