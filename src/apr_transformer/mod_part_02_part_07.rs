impl AprTransformer {

    /// Forward pass with KV cache for efficient autoregressive generation (Y4)
    ///
    /// Processes a single token using cached key-value pairs from previous positions.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token ID to process
    /// * `cache` - Mutable KV cache to read from and append to
    /// * `position` - Position in sequence (0-indexed)
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    pub fn forward_with_cache(
        &self,
        token_id: u32,
        cache: &mut AprKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        // DEBUG: Force F32 fallback to verify data layout issues
        let force_f32 = std::env::var("APR_FORCE_F32").is_ok();
        // NOISY-GUARD: Only print trace messages when REALIZE_TRACE is set
        let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // 2. Process through transformer layers
        let layers_start = std::time::Instant::now();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // PMAT-103: Get Q4K weights for this layer (if available) for fused kernels
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // F-REGR-231: Debug normed input
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: normed[0..5] = {:?}",
                    &normed[..5.min(normed.len())]
                );
            }

            // 2b. QKV projection (single token)
            // PMAT-103: Use fused Q4K kernels for separate Q, K, V weights when available
            let kv_size = num_kv_heads * head_dim;
            let (mut q, mut k, v) = if let Some(q4k) = q4k_layer {
                // Try Q4K fused kernels for Q, K
                let q = if let Some(ref q_bytes) = q4k.attn_q_weight {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: Q projection using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(q_bytes, &normed, hidden_dim, hidden_dim)?
                } else {
                    // Fallback to F32 for Q (should not happen for GGUF models)
                    let q_weight = &layer.qkv_weight[0..hidden_dim * hidden_dim];
                    self.matmul(&normed, q_weight, hidden_dim, hidden_dim)
                };

                let k = if let Some(ref k_bytes) = q4k.attn_k_weight {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: K projection using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(k_bytes, &normed, kv_size, hidden_dim)?
                } else {
                    let k_start = hidden_dim * hidden_dim;
                    let k_weight = &layer.qkv_weight[k_start..k_start + kv_size * hidden_dim];
                    self.matmul(&normed, k_weight, hidden_dim, kv_size)
                };

                // V can be Q4K or Q6K
                let v = if let Some(ref v_bytes) = q4k.attn_v_weight {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: V projection using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(v_bytes, &normed, kv_size, hidden_dim)?
                } else if let Some(ref v_bytes) = q4k.attn_v_weight_q6k {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: V projection using Q6K fused kernel");
                    }
                    matmul_q6k_rowmajor(v_bytes, &normed, kv_size, hidden_dim)?
                } else {
                    let v_start = hidden_dim * hidden_dim + kv_size * hidden_dim;
                    let v_weight = &layer.qkv_weight[v_start..v_start + kv_size * hidden_dim];
                    self.matmul(&normed, v_weight, hidden_dim, kv_size)
                };

                (q, k, v)
            } else {
                // Fallback: Combined QKV with F32 (legacy path)
                if trace_enabled && layer_idx == 0 && position == 0 {
                    eprintln!("[TRACE-CACHE] Layer 0: QKV projection using F32 (not fused)");
                }
                let qkv_out_dim = layer.qkv_weight.len() / hidden_dim;
                let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_out_dim);
                if let Some(ref bias) = layer.qkv_bias {
                    self.add_bias(&mut qkv, bias);
                }
                let q = qkv[0..hidden_dim].to_vec();
                let k = qkv[hidden_dim..hidden_dim + kv_size].to_vec();
                let v = qkv[hidden_dim + kv_size..hidden_dim + 2 * kv_size].to_vec();
                (q, k, v)
            };

            // Apply biases if present (for fused path)
            // The combined qkv_bias is [Q_bias | K_bias | V_bias]
            let mut v_mut = v;
            if q4k_layer.is_some() {
                if let Some(ref bias) = layer.qkv_bias {
                    // Split bias into Q, K, V portions
                    for (i, b) in bias[0..hidden_dim].iter().enumerate() {
                        q[i] += b;
                    }
                    for (i, b) in bias[hidden_dim..hidden_dim + kv_size].iter().enumerate() {
                        k[i] += b;
                    }
                    // V bias starts after Q and K biases
                    let v_bias_start = hidden_dim + kv_size;
                    for (i, b) in bias[v_bias_start..v_bias_start + kv_size]
                        .iter()
                        .enumerate()
                    {
                        v_mut[i] += b;
                    }
                }
            }
            let v = v_mut;

            // F-REGR-231: Debug K after bias
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: K after bias[0..5] = {:?}",
                    &k[..5.min(k.len())]
                );
                eprintln!(
                    "[TRACE-CACHE] Layer 0: V after bias[0..5] = {:?}",
                    &v[..5.min(v.len())]
                );
            }

            // PMAT-103: Apply RoPE to Q and K at current position
            // This was missing, causing garbage output
            self.apply_rope_f32(&mut q, position, num_heads, head_dim);
            self.apply_rope_f32(&mut k, position, num_kv_heads, head_dim);

            // F-REGR-231: Debug K after RoPE
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: K after RoPE[0..5] = {:?}",
                    &k[..5.min(k.len())]
                );
            }

            // 2c. Append K, V to cache (K now has RoPE applied)
            cache.append(layer_idx, &k, &v);

            // 2d. Compute attention with full cache
            let (k_cache, v_cache) = cache.get(layer_idx);
            let cache_len = cache.len();

            let attn_out = self.compute_attention_with_cache(
                &q, &k, &v, k_cache, v_cache,
                cache_len, position,
                num_heads, num_kv_heads, head_dim, hidden_dim,
            );

            // F-REGR-231: Debug attn_out before projection
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: attn_out[0..5] = {:?} (before output projection)",
                    &attn_out[..5.min(attn_out.len())]
                );
                let attn_out_sum: f32 = attn_out.iter().sum();
                eprintln!("[TRACE-CACHE] Layer 0: attn_out sum = {:.4}", attn_out_sum);
            }

            // 2e. Attention output projection
            // PMAT-103: Use Q4K fused kernel when available (single token path)
            let mut attn_output = if !force_f32 {
                if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.attn_output_weight.as_ref()) {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: attn_output using Q4K fused kernel");
                    }
                    matmul_q4k_rowmajor(q4k_bytes, &attn_out, hidden_dim, hidden_dim)?
                } else {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: attn_output using F32 fallback (slow!)");
                    }
                    self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
                }
            } else {
                if trace_enabled && layer_idx == 0 && position == 0 {
                    eprintln!("[TRACE-CACHE] Layer 0: attn_output using F32 (APR_FORCE_F32)");
                }
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
            };
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // F-REGR-231: Debug attn_output after projection
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: attn_output[0..5] = {:?} (after output projection)",
                    &attn_output[..5.min(attn_output.len())]
                );
            }

            // 2f. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // F-REGR-231: Debug hidden after attention
            if trace_enabled && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] Layer 0: hidden_after_attn[0..5] = {:?}",
                    &hidden[..5.min(hidden.len())]
                );
            }

            // 2g. Apply FFN norm if present (post_attention_layernorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                self.layer_norm(
                    &hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            // 2h. FFN projection (SwiGLU or standard GELU)
            // PMAT-103 FIX: Use Q4K/Q6K fused kernels when available (single token path)
            let intermediate_dim = self.config.intermediate_dim;
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: down(SiLU(gate(x)) * up(x))
                // GH-192/199: Compute gate and up in parallel (like GGUF path)
                let q4k_gate = if !force_f32 {
                    q4k_layer.and_then(|q| q.ffn_gate_weight.as_ref())
                } else {
                    None
                };
                let q4k_up = if !force_f32 {
                    q4k_layer.and_then(|q| q.ffn_up_weight.as_ref())
                } else {
                    None
                };
                let q6k_up = if !force_f32 && q4k_up.is_none() {
                    q4k_layer.and_then(|q| q.ffn_up_weight_q6k.as_ref())
                } else {
                    None
                };
                let (gate_result, up_result) = rayon::join(
                    || -> Result<Vec<f32>> {
                        if let Some(q4k_bytes) = q4k_gate {
                            if trace_enabled && layer_idx == 0 && position == 0 {
                                eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using Q4K fused kernel");
                            }
                            matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                        } else {
                            if trace_enabled && layer_idx == 0 && position == 0 {
                                if force_f32 {
                                    eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using F32 (APR_FORCE_F32)");
                                } else {
                                    eprintln!("[TRACE-CACHE] Layer 0: ffn_gate using F32 fallback (slow!)");
                                }
                            }
                            Ok(self.matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim))
                        }
                    },
                    || -> Result<Vec<f32>> {
                        if let Some(q4k_bytes) = q4k_up {
                            if trace_enabled && layer_idx == 0 && position == 0 {
                                eprintln!("[TRACE-CACHE] Layer 0: ffn_up using Q4K fused kernel");
                            }
                            matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                        } else if let Some(q6k_bytes) = q6k_up {
                            if trace_enabled && layer_idx == 0 && position == 0 {
                                eprintln!("[TRACE-CACHE] Layer 0: ffn_up using Q6K fused kernel");
                            }
                            matmul_q6k_rowmajor(q6k_bytes, &ffn_input, intermediate_dim, hidden_dim)
                        } else {
                            if trace_enabled && layer_idx == 0 && position == 0 {
                                if force_f32 {
                                    eprintln!("[TRACE-CACHE] Layer 0: ffn_up using F32 (APR_FORCE_F32)");
                                } else {
                                    eprintln!("[TRACE-CACHE] Layer 0: ffn_up using F32 fallback (slow!)");
                                }
                            }
                            Ok(self.matmul(
                                &ffn_input,
                                &layer.ffn_up_weight,
                                hidden_dim,
                                intermediate_dim,
                            ))
                        }
                    },
                );
                let gate = gate_result?;
                let up = up_result?;

                // SiLU(gate) * up, then down projection
                let mut ffn_hidden = Vec::with_capacity(gate.len());
                for (g, u) in gate.iter().zip(up.iter()) {
                    let silu_g = g / (1.0 + (-g).exp()); // SiLU = x * sigmoid(x)
                    ffn_hidden.push(silu_g * u);
                }

                // PMAT-103: Check for Q4K or Q6K down weight
                let mut out = if !force_f32 {
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_down_weight.as_ref()) {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using Q4K fused kernel");
                        }
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)?
                    } else if let Some(q6k_bytes) =
                        q4k_layer.and_then(|q| q.ffn_down_weight_q6k.as_ref())
                    {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using Q6K fused kernel");
                        }
                        matmul_q6k_rowmajor(q6k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)?
                    } else {
                        if trace_enabled && layer_idx == 0 && position == 0 {
                            eprintln!("[TRACE-CACHE] Layer 0: ffn_down using F32 fallback (slow!)");
                        }
                        self.matmul(
                            &ffn_hidden,
                            &layer.ffn_down_weight,
                            intermediate_dim,
                            hidden_dim,
                        )
                    }
                } else {
                    if trace_enabled && layer_idx == 0 && position == 0 {
                        eprintln!("[TRACE-CACHE] Layer 0: ffn_down using F32 (APR_FORCE_F32)");
                    }
                    self.matmul(
                        &ffn_hidden,
                        &layer.ffn_down_weight,
                        intermediate_dim,
                        hidden_dim,
                    )
                };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            } else {
                // Standard MLP: down(GELU(up(x)))
                // PMAT-103: Check for Q4K up weight
                let mut ffn_hidden =
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref()) {
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_input, intermediate_dim, hidden_dim)?
                    } else {
                        self.matmul(
                            &ffn_input,
                            &layer.ffn_up_weight,
                            hidden_dim,
                            intermediate_dim,
                        )
                    };
                if let Some(ref bias) = layer.ffn_up_bias {
                    self.add_bias(&mut ffn_hidden, bias);
                }
                self.gelu(&mut ffn_hidden);

                // PMAT-103: Check for Q4K down weight
                let mut out =
                    if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_down_weight.as_ref()) {
                        matmul_q4k_rowmajor(q4k_bytes, &ffn_hidden, hidden_dim, intermediate_dim)?
                    } else {
                        self.matmul(
                            &ffn_hidden,
                            &layer.ffn_down_weight,
                            intermediate_dim,
                            hidden_dim,
                        )
                    };
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            };

            // 2i. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }

            // F-REGR-231: Debug hidden state after each layer
            if trace_enabled && layer_idx < 2 && position == 0 {
                eprintln!(
                    "[TRACE-CACHE] After layer {}: hidden[0..5] = {:?}",
                    layer_idx,
                    &hidden[..5.min(hidden.len())]
                );
            }
        }
        if trace_enabled {
            eprintln!(
                "[TRACE-CACHE] pos={}: {} layers took {:?}",
                position,
                self.layers.len(),
                layers_start.elapsed()
            );
        }

        // NOTE: No advance() needed here - append() auto-advances on the last layer
        // (see F-REGR-231 fix in config.rs)

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection
        // PMAT-103: Use Q4K/Q6K fused kernel when available (single token path)
        let lm_start = std::time::Instant::now();
        let mut logits = if !force_f32 {
            if let Some(ref q4k_bytes) = self.lm_head_weight_q4k {
                if trace_enabled {
                    eprintln!("[TRACE-CACHE] lm_head using Q4K fused kernel");
                }
                matmul_q4k_rowmajor(q4k_bytes, &normed, self.config.vocab_size, hidden_dim)?
            } else if let Some(ref q6k_bytes) = self.lm_head_weight_q6k {
                let result =
                    matmul_q6k_rowmajor(q6k_bytes, &normed, self.config.vocab_size, hidden_dim)?;
                if trace_enabled {
                    eprintln!("[TRACE-CACHE] lm_head Q6K took {:?}", lm_start.elapsed());
                }
                result
            } else {
                self.matmul(
                    &normed,
                    &self.lm_head_weight,
                    hidden_dim,
                    self.config.vocab_size,
                )
            }
        } else {
            if trace_enabled {
                eprintln!("[TRACE-CACHE] lm_head using F32 (APR_FORCE_F32)");
            }
            self.matmul(
                &normed,
                &self.lm_head_weight,
                hidden_dim,
                self.config.vocab_size,
            )
        };
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }
}

include!("mod_part_02_part_07_attn.rs");
