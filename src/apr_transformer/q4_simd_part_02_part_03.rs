impl QuantizedAprTransformerQ4 {

    /// Forward pass for a single token using scratch buffer (zero allocation)
    ///
    /// This is the fastest path for autoregressive generation when combined
    /// with `forward_with_cache_and_scratch`. It reuses pre-allocated buffers
    /// to eliminate per-token allocations.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token to process
    /// * `scratch` - Pre-allocated scratch buffer (from `create_scratch()`)
    ///
    /// # Returns
    ///
    /// Logits over vocabulary
    pub fn forward_single_with_scratch(
        &self,
        token_id: u32,
        scratch: &mut AprInferenceScratch,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec_into;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // 1. Token embedding lookup (write directly to scratch.hidden)
        let offset = (token_id as usize) * hidden_dim;
        if offset + hidden_dim <= self.token_embedding.len() {
            scratch.hidden[..hidden_dim]
                .copy_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
        } else {
            scratch.hidden[..hidden_dim].fill(0.0);
        }

        // 2. Process through transformer layers
        for layer in &self.layers {
            // Pre-attention RMS norm (reuse scratch.normed)
            let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
            for i in 0..hidden_dim {
                scratch.normed[i] = scratch.hidden[i] / rms * layer.attn_norm_weight[i];
            }

            // QKV projection (zero-allocation - write directly to scratch.qkv_out)
            let qkv_dim = layer.qkv_weight.out_dim;
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.qkv_weight.data,
                &scratch.normed[..hidden_dim],
                hidden_dim,
                &mut scratch.qkv_out[..qkv_dim],
            )?;

            // Extract Q, K, V and apply RoPE (position=0 for single token)
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            scratch.q[..q_dim].copy_from_slice(&scratch.qkv_out[..q_dim]);
            scratch.k[..kv_dim].copy_from_slice(&scratch.qkv_out[q_dim..q_dim + kv_dim]);
            scratch.v[..kv_dim]
                .copy_from_slice(&scratch.qkv_out[q_dim + kv_dim..q_dim + 2 * kv_dim]);

            // Apply RoPE at position 0
            self.apply_rope(&mut scratch.q[..q_dim], 0, num_heads);
            self.apply_rope(&mut scratch.k[..kv_dim], 0, num_kv_heads);

            // For single token, attention is trivial: output = V (softmax of 1 element = 1.0)
            let group_size = num_heads / num_kv_heads;
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let v_offset = kv_head * head_dim;
                let out_offset = head * head_dim;
                scratch.attn_out[out_offset..out_offset + head_dim]
                    .copy_from_slice(&scratch.v[v_offset..v_offset + head_dim]);
            }

            // Output projection (uses scratch.ffn_out buffer)
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.attn_output_weight.data,
                &scratch.attn_out[..hidden_dim],
                layer.attn_output_weight.in_dim,
                &mut scratch.ffn_out[..layer.attn_output_weight.out_dim],
            )?;

            // Residual connection (attn)
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_out[i];
            }

            // Pre-FFN norm
            if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for i in 0..hidden_dim {
                    scratch.ffn_input[i] = scratch.hidden[i] / rms * ffn_norm[i];
                }
            } else {
                scratch.ffn_input[..hidden_dim].copy_from_slice(&scratch.normed[..hidden_dim]);
            }

            // FFN with SwiGLU
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            if let Some(gate) = &layer.ffn_gate_weight {
                // Up projection (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_up[..intermediate_dim],
                )?;

                // Gate projection (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &gate.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_gate[..intermediate_dim],
                )?;

                // SwiGLU: silu(gate) * up
                for i in 0..intermediate_dim {
                    let silu = scratch.ffn_gate[i] / (1.0 + (-scratch.ffn_gate[i]).exp());
                    scratch.ffn_up[i] *= silu;
                }
            } else {
                // GELU path (zero-allocation)
                fused_q4_0_q8_0_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &scratch.ffn_input[..hidden_dim],
                    hidden_dim,
                    &mut scratch.ffn_up[..intermediate_dim],
                )?;

                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for i in 0..intermediate_dim {
                    let x = scratch.ffn_up[i];
                    let t = (SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)).tanh();
                    scratch.ffn_up[i] = 0.5 * x * (1.0 + t);
                }
            }

            // Down projection (write to scratch.ffn_out)
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.ffn_down_weight.data,
                &scratch.ffn_up[..intermediate_dim],
                intermediate_dim,
                &mut scratch.ffn_out[..hidden_dim],
            )?;

            // Residual connection (FFN)
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_out[i];
            }
        }

        // 3. Final RMS norm
        let sq_sum: f32 = scratch.hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        for i in 0..hidden_dim {
            scratch.normed[i] = scratch.hidden[i] / rms * self.output_norm_weight[i];
        }

        // 4. LM head projection (still allocates - logits must be returned)
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        fused_q4_0_q8_0_parallel_matvec_into(
            &self.lm_head_weight.data,
            &scratch.normed[..hidden_dim],
            hidden_dim,
            &mut logits,
        )?;

        Ok(logits)
    }

    /// Forward pass with KV cache for efficient autoregressive generation
    ///
    /// This method only computes attention for the new token(s), reusing
    /// cached K/V from previous positions. Provides 1.5-2x speedup.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - New token IDs to process (typically 1 for generation)
    /// * `cache` - KV cache to use and update
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for the last token
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut AprKVCache,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;

        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // Position in the sequence (including cached positions)
        let cache_len = cache.len();
        let new_seq_len = token_ids.len();

        // 1. Token embedding lookup (F32)
        let mut hidden = Vec::with_capacity(new_seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-attention RMS norm
            let mut normed = Vec::with_capacity(hidden.len());
            for s in 0..new_seq_len {
                let start = s * hidden_dim;
                let slice = &hidden[start..start + hidden_dim];
                let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for (i, &x) in slice.iter().enumerate() {
                    normed.push(x / rms * layer.attn_norm_weight[i]);
                }
            }

            // QKV projection using SIMD matmul (only for new tokens)
            let qkv_dim = layer.qkv_weight.out_dim;
            let mut qkv_out = Vec::with_capacity(new_seq_len * qkv_dim);
            for s in 0..new_seq_len {
                let input = &normed[s * hidden_dim..(s + 1) * hidden_dim];
                let qkv = fused_q4_0_q8_0_parallel_matvec(
                    &layer.qkv_weight.data,
                    input,
                    hidden_dim,
                    qkv_dim,
                )?;
                qkv_out.extend(qkv);
            }

            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // Process new tokens: extract Q, K, V and apply RoPE
            let mut new_q = Vec::with_capacity(new_seq_len * q_dim);
            for s in 0..new_seq_len {
                let qkv_start = s * qkv_dim;
                let position = cache_len + s;

                // Extract Q, K, V for this position
                let mut q = qkv_out[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv_out[qkv_start + q_dim..qkv_start + q_dim + kv_dim].to_vec();
                let v =
                    qkv_out[qkv_start + q_dim + kv_dim..qkv_start + q_dim + 2 * kv_dim].to_vec();

                // Apply RoPE with correct position
                self.apply_rope(&mut q, position, num_heads);
                self.apply_rope(&mut k, position, num_kv_heads);

                new_q.extend_from_slice(&q);

                // Append to cache (K and V with RoPE applied to K)
                cache.append(layer_idx, &k, &v);
            }

            // Get full K and V from cache (includes new tokens)
            let (full_k, full_v) = cache.get(layer_idx);
            let total_seq_len = cache.len();

            // Compute attention: new Q attends to all cached K/V
            let attn_output = self.causal_attention_cached(
                &new_q,
                full_k,
                full_v,
                new_seq_len,
                total_seq_len,
                cache_len,
            );

            // Output projection using SIMD matmul
            let mut proj_out = Vec::with_capacity(new_seq_len * hidden_dim);
            for s in 0..new_seq_len {
                let input = &attn_output[s * hidden_dim..(s + 1) * hidden_dim];
                let proj = fused_q4_0_q8_0_parallel_matvec(
                    &layer.attn_output_weight.data,
                    input,
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                )?;
                proj_out.extend(proj);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += proj_out[i];
            }

            // Pre-FFN norm (if present)
            let ffn_input = if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let mut normed_ffn = Vec::with_capacity(hidden.len());
                for s in 0..new_seq_len {
                    let start = s * hidden_dim;
                    let slice = &hidden[start..start + hidden_dim];
                    let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                    let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                    for (i, &x) in slice.iter().enumerate() {
                        normed_ffn.push(x / rms * ffn_norm[i]);
                    }
                }
                normed_ffn
            } else {
                normed.clone()
            };

            // FFN with parallel up/gate for SwiGLU models
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            let ffn_up = if let Some(gate) = &layer.ffn_gate_weight {
                // SwiGLU: Parallel FFN up + gate
                let (ffn_up_result, ffn_gate_result) = rayon::join(
                    || {
                        let mut up = Vec::with_capacity(new_seq_len * intermediate_dim);
                        for s in 0..new_seq_len {
                            let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            if let Ok(u) = fused_q4_0_q8_0_parallel_matvec(
                                &layer.ffn_up_weight.data,
                                input,
                                hidden_dim,
                                intermediate_dim,
                            ) {
                                up.extend(u);
                            }
                        }
                        up
                    },
                    || {
                        let mut g = Vec::with_capacity(new_seq_len * intermediate_dim);
                        for s in 0..new_seq_len {
                            let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                            if let Ok(gv) = fused_q4_0_q8_0_parallel_matvec(
                                &gate.data,
                                input,
                                hidden_dim,
                                intermediate_dim,
                            ) {
                                g.extend(gv);
                            }
                        }
                        g
                    },
                );

                let mut up = ffn_up_result;
                for i in 0..up.len() {
                    let silu = ffn_gate_result[i] / (1.0 + (-ffn_gate_result[i]).exp());
                    up[i] *= silu;
                }
                up
            } else {
                // Non-SwiGLU: Sequential + GELU
                let mut up = Vec::with_capacity(new_seq_len * intermediate_dim);
                for s in 0..new_seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    up.extend(u);
                }
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for x in &mut up {
                    let t = (SQRT_2_OVER_PI * (*x + GELU_COEFF * *x * *x * *x)).tanh();
                    *x = 0.5 * *x * (1.0 + t);
                }
                up
            };

            // FFN: down projection
            let mut ffn_down = Vec::with_capacity(new_seq_len * hidden_dim);
            for s in 0..new_seq_len {
                let input = &ffn_up[s * intermediate_dim..(s + 1) * intermediate_dim];
                let down = fused_q4_0_q8_0_parallel_matvec(
                    &layer.ffn_down_weight.data,
                    input,
                    intermediate_dim,
                    hidden_dim,
                )?;
                ffn_down.extend(down);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_down[i];
            }
        }

        // 3. Final RMS norm (only for last token)
        let last_start = (new_seq_len - 1) * hidden_dim;
        let last_hidden = &hidden[last_start..last_start + hidden_dim];
        let sq_sum: f32 = last_hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed_final: Vec<f32> = last_hidden
            .iter()
            .enumerate()
            .map(|(i, &x)| x / rms * self.output_norm_weight[i])
            .collect();

        // 4. LM head projection using SIMD matmul
        let vocab_size = self.config.vocab_size;
        let logits = fused_q4_0_q8_0_parallel_matvec(
            &self.lm_head_weight.data,
            &normed_final,
            hidden_dim,
            vocab_size,
        )?;

        Ok(logits)
    }
}
