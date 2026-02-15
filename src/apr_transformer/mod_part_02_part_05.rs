impl AprTransformer {

    /// Forward pass through the transformer
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        // NOISY-GUARD: Only print trace messages when REALIZE_TRACE is set
        let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // PMAT-103: Get Q4K weights for this layer (if available)
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection
            // Calculate qkv_dim from actual weight size (handles GQA models)
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // 2c. Proper attention with GQA support and RoPE
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let kv_dim = num_kv_heads * head_dim;
            let group_size = self.config.num_heads / num_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Split QKV and apply RoPE
            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V (layout: [Q..., K..., V...])
                let mut q_pos = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k_pos =
                    qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v_pos =
                    &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                // Apply RoPE to Q and K
                self.apply_rope_f32(&mut q_pos, s, self.config.num_heads, head_dim);
                self.apply_rope_f32(&mut k_pos, s, num_kv_heads, head_dim);

                q_all.extend_from_slice(&q_pos);
                k_all.extend_from_slice(&k_pos);
                v_all.extend_from_slice(v_pos);
            }

            // Compute scaled dot-product attention with causal mask
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
            for head in 0..self.config.num_heads {
                let kv_head = head / group_size;
                let q_head_offset = head * head_dim;
                let kv_head_offset = kv_head * head_dim;

                for i in 0..seq_len {
                    // Compute attention scores for this position
                    let mut scores = Vec::with_capacity(i + 1);
                    let q_start = i * hidden_dim + q_head_offset;

                    for j in 0..=i {
                        // Only attend to positions <= current (causal mask)
                        let k_start = j * kv_dim + kv_head_offset;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_all[q_start + d] * k_all[k_start + d];
                        }
                        scores.push(score * scale);
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    if exp_sum > 0.0 {
                        for s in &mut scores {
                            *s /= exp_sum;
                        }
                    }

                    // Weighted sum of V
                    let out_start = i * hidden_dim + q_head_offset;
                    for (j, &weight) in scores.iter().enumerate() {
                        let v_start = j * kv_dim + kv_head_offset;
                        for d in 0..head_dim {
                            attn_out[out_start + d] += weight * v_all[v_start + d];
                        }
                    }
                }
            }

            // 2d. Attention output projection
            // PMAT-103: Use Q4K fused kernel when available
            let mut attn_output = if let Some(q4k_bytes) =
                q4k_layer.and_then(|q| q.attn_output_weight.as_ref())
            {
                if trace_enabled && layer_idx == 0 {
                    eprintln!("[TRACE] Layer {layer_idx}: attn_output using Q4K fused kernel");
                }
                // Fused Q4K matmul: process each position separately
                // PMAT-103: Use column-major kernel for GGUF layout
                let seq_len = token_ids.len();
                let mut output = Vec::with_capacity(seq_len * hidden_dim);
                for s in 0..seq_len {
                    let input_slice = &attn_out[s * hidden_dim..(s + 1) * hidden_dim];
                    let pos_out =
                        matmul_q4k_rowmajor(q4k_bytes, input_slice, hidden_dim, hidden_dim)?;
                    output.extend(pos_out);
                }
                output
            } else {
                if trace_enabled && layer_idx == 0 {
                    eprintln!("[TRACE] Layer {layer_idx}: attn_output using F32 fallback (slow!)");
                }
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim)
            };
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Apply FFN norm if present (post_attention_layernorm)
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

            // 2g. FFN projection (SwiGLU or standard GELU)
            // PMAT-103: Use Q4K fused kernel when available for FFN
            let seq_len = token_ids.len();
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU: down(SiLU(gate(x)) * up(x))
                // GH-192/199: Compute gate and up in parallel (like GGUF path)
                let q4k_gate = q4k_layer.and_then(|q| q.ffn_gate_weight.as_ref());
                let q4k_up = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref());
                let (gate_result, up_result) = rayon::join(
                    || -> Result<Vec<f32>> {
                        if let Some(q4k_bytes) = q4k_gate {
                            if trace_enabled && layer_idx == 0 {
                                eprintln!("[TRACE] Layer {layer_idx}: ffn_gate using Q4K fused kernel");
                            }
                            let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                            for s in 0..seq_len {
                                let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                                let pos_out = matmul_q4k_rowmajor(
                                    q4k_bytes,
                                    input_slice,
                                    intermediate_dim,
                                    hidden_dim,
                                )?;
                                output.extend(pos_out);
                            }
                            Ok(output)
                        } else {
                            Ok(self.matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim))
                        }
                    },
                    || -> Result<Vec<f32>> {
                        if let Some(q4k_bytes) = q4k_up {
                            if trace_enabled && layer_idx == 0 {
                                eprintln!("[TRACE] Layer {layer_idx}: ffn_up using Q4K fused kernel");
                            }
                            let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                            for s in 0..seq_len {
                                let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                                let pos_out = matmul_q4k_rowmajor(
                                    q4k_bytes,
                                    input_slice,
                                    intermediate_dim,
                                    hidden_dim,
                                )?;
                                output.extend(pos_out);
                            }
                            Ok(output)
                        } else {
                            if trace_enabled && layer_idx == 0 {
                                eprintln!("[TRACE] Layer {layer_idx}: ffn_up using F32 fallback (slow!)");
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
                let mut out = if let Some(q4k_bytes) =
                    q4k_layer.and_then(|q| q.ffn_down_weight.as_ref())
                {
                    if trace_enabled && layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using Q4K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let input_slice =
                            &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                        let pos_out = matmul_q4k_rowmajor(
                            q4k_bytes,
                            input_slice,
                            hidden_dim,
                            intermediate_dim,
                        )?;
                        output.extend(pos_out);
                    }
                    output
                } else if let Some(q6k_bytes) =
                    q4k_layer.and_then(|q| q.ffn_down_weight_q6k.as_ref())
                {
                    if trace_enabled && layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using Q6K fused kernel");
                    }
                    let mut output = Vec::with_capacity(seq_len * hidden_dim);
                    for s in 0..seq_len {
                        let input_slice =
                            &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                        let pos_out = matmul_q6k_rowmajor(
                            q6k_bytes,
                            input_slice,
                            hidden_dim,
                            intermediate_dim,
                        )?;
                        output.extend(pos_out);
                    }
                    output
                } else {
                    if trace_enabled && layer_idx == 0 {
                        eprintln!("[TRACE] Layer {layer_idx}: ffn_down using F32 fallback (slow!)");
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
                        let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                        for s in 0..seq_len {
                            let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            // PMAT-103 FIX: Q4K kernel expects (ne0=output_dim, ne1=input_dim)
                            // ffn_up: [intermediate_dim, hidden_dim] maps hidden[1536] -> intermediate[8960]
                            let pos_out = matmul_q4k_rowmajor(
                                q4k_bytes,
                                input_slice,
                                intermediate_dim,
                                hidden_dim,
                            )?;
                            output.extend(pos_out);
                        }
                        output
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
                        let mut output = Vec::with_capacity(seq_len * hidden_dim);
                        for s in 0..seq_len {
                            let input_slice =
                                &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                            let pos_out = matmul_q4k_rowmajor(
                                q4k_bytes,
                                input_slice,
                                hidden_dim,
                                intermediate_dim,
                            )?;
                            output.extend(pos_out);
                        }
                        output
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

            // 2h. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self.matmul(
            last_hidden,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }
}
