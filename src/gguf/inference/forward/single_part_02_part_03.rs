impl OwnedQuantizedModel {

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
}
