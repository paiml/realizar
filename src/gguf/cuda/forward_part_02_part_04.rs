impl OwnedQuantizedModelCuda {

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

            // 2b. QKV projection (GPU - PAR-014: use pre-captured cache key)
            let mut qkv = self.qkv_matmul_cuda_with_key(&normed, &qkv_weight, &qkv_cache_key)?;
            if let Some(ref bias) = qkv_bias {
                ops::add_bias(&mut qkv, bias);
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

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }


            // 2h/2i. FFN
            // PAR-057: Re-enable fused FFN path now that kernels are fixed

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

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }

            // NaN safety check for early positions
            if position < 2 {
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

        Ok(logits)
    }
}
