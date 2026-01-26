//! Forward pass implementations for OwnedQuantizedModel
//!
//! Contains forward, forward_cached methods.
//! These are the core inference entry points.

use crate::error::Result;
use crate::gguf::ops;
use crate::gguf::OwnedQKVWeights;
use crate::gguf::OwnedQuantizedModel;

impl OwnedQuantizedModel {
    /// Forward pass with fused Q4_K operations (IMP-100)
    ///
    /// This is 1.37x faster than dequantized f32 due to reduced memory bandwidth.
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
    /// Returns error if tensor operations fail
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        // Note: intermediate_dim is encoded in layer weight tensors (in_dim/out_dim)
        let _ = self.config.intermediate_dim;

        // 1. Token embedding lookup (f32, fast)
        let mut hidden = self.embed(token_ids);

        // Detect if model uses RMSNorm (LLaMA-style) or LayerNorm (phi-2 style)
        // LLaMA models have ffn_gate_weight (SwiGLU) and no bias in norms
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // 2. Process through transformer layers with FUSED Q4_K ops
        let cpu_debug_layers = std::env::var("CPU_DEBUG_LAYERS").is_ok();
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

            // CORRECTNESS-011: CPU intermediate debug at L0
            if cpu_debug_layers && layer_idx < 2 {
                eprintln!(
                    "[CPU-L{}] RMSNorm: first 3 = [{:.4}, {:.4}, {:.4}]",
                    layer_idx, normed[0], normed[1], normed[2]
                );
            }

            // 2b. QKV projection with FUSED dequant+dot (1.37x faster)
            // Note: qkv_dim may differ from 3*hidden_dim for GQA models
            let qkv_dim = layer.qkv_weight.out_dim();
            let q_dim = layer.qkv_weight.q_dim();
            // For GQA, k_dim and v_dim may be smaller than q_dim
            let k_dim = match &layer.qkv_weight {
                OwnedQKVWeights::Fused(_) => q_dim,
                OwnedQKVWeights::Separate { k, .. } => k.out_dim,
            };
            let v_dim = match &layer.qkv_weight {
                OwnedQKVWeights::Fused(_) => q_dim,
                OwnedQKVWeights::Separate { v, .. } => v.out_dim,
            };
            let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // CORRECTNESS-011: Q, K, V before RoPE (after bias)
            if cpu_debug_layers && (layer_idx < 2 || layer_idx == 4 || layer_idx == 5) {
                eprintln!(
                    "[CPU-L{}] Q (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx, qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]
                );
                // K starts at q_dim offset
                eprintln!(
                    "[CPU-L{}] K (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx,
                    qkv[q_dim],
                    qkv[q_dim + 1],
                    qkv[q_dim + 2],
                    qkv[q_dim + 3],
                    qkv[q_dim + 4]
                );
                // V starts at q_dim + k_dim offset
                let v_offset = q_dim + k_dim;
                eprintln!(
                    "[CPU-L{}] V (before RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    layer_idx,
                    qkv[v_offset],
                    qkv[v_offset + 1],
                    qkv[v_offset + 2],
                    qkv[v_offset + 3],
                    qkv[v_offset + 4]
                );
            }

            // 2c. Proper attention with RoPE and causal mask (IMP-101)
            let seq_len = token_ids.len();

            // Extract Q, K, V and apply RoPE to Q and K
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * k_dim);
            let mut v_all = Vec::with_capacity(seq_len * v_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V for this position (QKV layout: [Q..., K..., V...])
                let mut q = qkv[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv[qkv_start + q_dim..qkv_start + q_dim + k_dim].to_vec();
                let v = &qkv[qkv_start + q_dim + k_dim..qkv_start + q_dim + k_dim + v_dim];

                // Apply RoPE to Q and K (position-dependent rotation)
                // GQA: Q has num_heads, K has num_kv_heads
                self.apply_rope(&mut q, s, self.config.num_heads);
                self.apply_rope(&mut k, s, self.config.num_kv_heads);

                // CORRECTNESS-011: Q after RoPE at position 0
                if cpu_debug_layers && layer_idx < 2 && s == 0 {
                    eprintln!(
                        "[CPU-L{}] Q (after RoPE): first 3 = [{:.4}, {:.4}, {:.4}]",
                        layer_idx, q[0], q[1], q[2]
                    );
                    eprintln!(
                        "[CPU-L{}] K (after RoPE): first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                        layer_idx, k[0], k[1], k[2], k[3], k[4]
                    );
                    eprintln!(
                        "[CPU-L{}] V: first 5 = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                        layer_idx, v[0], v[1], v[2], v[3], v[4]
                    );
                }

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // Compute scaled dot-product attention with causal mask
            let attn_out = self.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // CORRECTNESS-011: Attention output
            if cpu_debug_layers && layer_idx < 2 {
                eprintln!(
                    "[CPU-L{}] Attn output: first 3 = [{:.4}, {:.4}, {:.4}]",
                    layer_idx, attn_out[0], attn_out[1], attn_out[2]
                );
            }

            // 2d. Attention output projection with FUSED ops
            // Input is q_dim (attention output), projects back to hidden_dim
            let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Pre-FFN layer norm (LLaMA uses separate ffn_norm with RMSNorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                // LLaMA-style: separate FFN layer norm (use RMSNorm for LLaMA)
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
                // phi-2 style: no separate FFN norm, use hidden directly
                // (some models apply attn_norm again, but we've already done residual)
                hidden.clone()
            };

            // 2g. FFN with SwiGLU or GELU activation
            let ffn_activated = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU path (LLaMA, TinyLlama, Mistral, etc.)
                // output = down(gate(x) * silu(up(x)))
                let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_up, bias);
                }

                let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                if let Some(ref bias) = layer.ffn_gate_bias {
                    ops::add_bias(&mut ffn_gate, bias);
                }

                // SwiGLU: down(silu(gate(x)) * up(x))
                // Apply SiLU to GATE projection, not up
                ops::silu(&mut ffn_gate);

                // Element-wise multiply: silu(gate) * up
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }

                ffn_gate
            } else {
                // GELU path (phi-2, GPT-2, etc.)
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);
                ffn_hidden
            };

            // 2g. FFN down projection with FUSED ops
            let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }

            // CORRECTNESS-011: Per-layer CPU debug output
            if cpu_debug_layers {
                let seq_len = token_ids.len();
                let last_hidden_start = (seq_len - 1) * hidden_dim;
                let last_h = &hidden[last_hidden_start..last_hidden_start + hidden_dim];
                let sum: f32 = last_h.iter().sum();
                let sq_sum: f32 = last_h.iter().map(|x| x * x).sum();
                let rms = (sq_sum / last_h.len() as f32).sqrt();
                eprintln!(
                    "[CPU-L{}] After layer: first 3 = [{:.4}, {:.4}, {:.4}], sum = {:.4}, rms = {:.4}",
                    layer_idx, last_h[0], last_h[1], last_h[2], sum, rms
                );
            }
        }

        // CORRECTNESS-011: CPU hidden state debug output (compare with GPU)
        if std::env::var("CPU_DEBUG").is_ok() {
            let seq_len = token_ids.len();
            let last_hidden_start = (seq_len - 1) * hidden_dim;
            let last_hidden_raw = &hidden[last_hidden_start..last_hidden_start + hidden_dim];

            let sum: f32 = last_hidden_raw.iter().sum();
            let sq_sum: f32 = last_hidden_raw.iter().map(|x| x * x).sum();
            let rms = (sq_sum / last_hidden_raw.len() as f32).sqrt();

            eprintln!("[CORRECTNESS-011] CPU Hidden before output_norm:");
            eprintln!(
                "  first 5 = {:?}",
                &last_hidden_raw[..5.min(last_hidden_raw.len())]
            );
            eprintln!("  sum = {:.4}, rms = {:.4}", sum, rms);
            eprintln!("  (GPU shows: sum=466.2486, rms=39.4793)");
        }

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

        // CORRECTNESS-011: CPU normed hidden state debug output
        if std::env::var("CPU_DEBUG").is_ok() {
            let seq_len = token_ids.len();
            let last_normed_start = (seq_len - 1) * hidden_dim;
            let last_normed = &normed[last_normed_start..last_normed_start + hidden_dim];

            let sum: f32 = last_normed.iter().sum();
            let sq_sum: f32 = last_normed.iter().map(|x| x * x).sum();
            let rms = (sq_sum / last_normed.len() as f32).sqrt();

            eprintln!("[CORRECTNESS-011] CPU Normed hidden:");
            eprintln!("  first 5 = {:?}", &last_normed[..5.min(last_normed.len())]);
            eprintln!("  sum = {:.4}, rms = {:.4}", sum, rms);
            eprintln!("  (GPU shows: sum=107.5945, rms=4.6616)");
        }

        // 4. LM head projection with FUSED ops (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        // Compute logits using fused op
        let mut logits = self.fused_matmul(last_hidden, &self.lm_head_weight)?;

        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// Forward pass with KV cache for efficient autoregressive decoding
    ///
    /// This method properly handles both architectures:
    /// - LLaMA-style: RMSNorm, SwiGLU FFN, GQA attention
    /// - phi-2 style: LayerNorm, GELU FFN, MHA attention
    ///
    /// Uses O(n) per-token cost instead of O(n²) by caching K/V.
    ///
    /// # Arguments
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for all layers
    /// * `position` - Position in sequence for RoPE
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    pub fn forward_cached(
        &self,
        token_id: u32,
        cache: &mut crate::gguf::OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // Detect architecture: LLaMA uses RMSNorm (no bias) and SwiGLU (has gate weight)
        let use_rmsnorm = self
            .layers
            .first()
            .is_some_and(|l| l.ffn_gate_weight.is_some() && l.attn_norm_bias.is_none());

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // PAR-052: Debug output for OwnedQuantizedModel forward path
        let debug_forward = std::env::var("REALIZAR_DEBUG_FORWARD").is_ok();
        if debug_forward && position == 0 {
            eprintln!("[PAR-052] OwnedQuantizedModel::forward_cached");
            eprintln!("[PAR-052] Token ID: {}, Position: {}", token_id, position);
            eprintln!("[PAR-052] use_rmsnorm: {}", use_rmsnorm);
            eprintln!(
                "[PAR-052] Embedding[0..8]: {:?}",
                &hidden[..8.min(hidden.len())]
            );
            eprintln!("[PAR-052] Embedding sum: {:.6}", hidden.iter().sum::<f32>());
        }

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm (RMSNorm for LLaMA, LayerNorm for phi-2)
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

            // PAR-052: Debug layer 0 normed and QKV values
            if debug_forward && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[PAR-052-L0] attn_norm[0..8]: {:?}",
                    &layer.attn_norm_weight[..8.min(layer.attn_norm_weight.len())]
                );
                eprintln!(
                    "[PAR-052-L0] normed[0..8]: {:?}",
                    &normed[..8.min(normed.len())]
                );
                eprintln!("[PAR-052-L0] normed sum: {:.6}", normed.iter().sum::<f32>());
            }

            // 2b. QKV projection
            let _qkv_dim = layer.qkv_weight.out_dim();
            let q_dim = layer.qkv_weight.q_dim();
            let k_dim = match &layer.qkv_weight {
                OwnedQKVWeights::Fused(_) => q_dim,
                OwnedQKVWeights::Separate { k, .. } => k.out_dim,
            };
            let v_dim = match &layer.qkv_weight {
                OwnedQKVWeights::Fused(_) => q_dim,
                OwnedQKVWeights::Separate { v, .. } => v.out_dim,
            };

            let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // PAR-052: Debug QKV after projection
            if debug_forward && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[PAR-052-L0] QKV dims: q={}, k={}, v={}, total={}",
                    q_dim,
                    k_dim,
                    v_dim,
                    qkv.len()
                );
                eprintln!("[PAR-052-L0] QKV sum: {:.6}", qkv.iter().sum::<f32>());
                eprintln!("[PAR-052-L0] Q[0..8]: {:?}", &qkv[..8.min(q_dim)]);
            }

            // 2c. Extract Q, K, V and apply RoPE
            let mut q = qkv[0..q_dim].to_vec();
            let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
            let v = qkv[q_dim + k_dim..q_dim + k_dim + v_dim].to_vec();

            // Apply RoPE with correct head counts for GQA
            self.apply_rope(&mut q, position, self.config.num_heads);
            self.apply_rope(&mut k, position, self.config.num_kv_heads);

            // PAR-052: Debug Q after RoPE
            if debug_forward && layer_idx == 0 && position == 0 {
                eprintln!(
                    "[PAR-052-L0] Q after RoPE[0..8]: {:?}",
                    &q[..8.min(q.len())]
                );
                eprintln!(
                    "[PAR-052-L0] K after RoPE[0..4]: {:?}",
                    &k[..4.min(k.len())]
                );
            }

            // 2d. Compute attention using cached K/V
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            let attn_out = if k_cache.is_empty() {
                // First token - just use V directly (self-attention with single token)
                // Expand V if GQA (num_kv_heads < num_heads)
                if self.config.num_kv_heads < self.config.num_heads {
                    let head_dim = hidden_dim / self.config.num_heads;
                    let group_size = self.config.num_heads / self.config.num_kv_heads;
                    (0..self.config.num_heads)
                        .flat_map(|h| {
                            let kv_head = h / group_size;
                            let start = kv_head * head_dim;
                            v[start..start + head_dim].iter().copied()
                        })
                        .collect()
                } else {
                    v.clone()
                }
            } else {
                // Use cached K/V for attention with GQA support
                self.attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
            };

            // 2e. Store K and V in cache (store original size, not expanded)
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

            // 2h. Pre-FFN layer norm (LLaMA has separate ffn_norm)
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

            // 2i. FFN with SwiGLU or GELU
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                // SwiGLU path (LLaMA)
                let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_up, bias);
                }

                let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                if let Some(ref bias) = layer.ffn_gate_bias {
                    ops::add_bias(&mut ffn_gate, bias);
                }

                // SiLU on gate, then multiply with up
                ops::silu(&mut ffn_gate);
                for i in 0..ffn_gate.len() {
                    ffn_gate[i] *= ffn_up[i];
                }

                let mut output = self.fused_matmul(&ffn_gate, &layer.ffn_down_weight)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut output, bias);
                }
                output
            } else {
                // GELU path (phi-2)
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);

                let mut output = self.fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut output, bias);
                }
                output
            };

            // 2j. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }
        }

        // Advance cache position
        cache.advance();

        // 3. Final layer norm
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

        // PAR-052: Debug final hidden state
        if debug_forward && position == 0 {
            eprintln!(
                "[PAR-052] Final hidden sum: {:.6}",
                hidden.iter().sum::<f32>()
            );
            eprintln!(
                "[PAR-052] Final hidden[0..8]: {:?}",
                &hidden[..8.min(hidden.len())]
            );
            eprintln!(
                "[PAR-052] After output_norm sum: {:.6}",
                normed.iter().sum::<f32>()
            );
            eprintln!(
                "[PAR-052] output_norm_weight[0..4]: {:?}",
                &self.output_norm_weight[..4.min(self.output_norm_weight.len())]
            );
            eprintln!(
                "[PAR-052] LM head weight dims: in={}, out={}",
                self.lm_head_weight.in_dim, self.lm_head_weight.out_dim
            );
        }

        // 4. LM head projection
        let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        // PAR-052: Debug final logits
        if debug_forward && position == 0 {
            // Find top-5 logits
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[PAR-052] Top-5 logits:");
            for (idx, val) in indexed.iter().take(5) {
                eprintln!("  Token {}: {:.6}", idx, val);
            }
            eprintln!("[PAR-052] Logits sum: {:.6}", logits.iter().sum::<f32>());
        }

        Ok(logits)
    }
}
