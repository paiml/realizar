
impl OwnedQuantizedModel {

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

include!("single_part_02_part_02.rs");
include!("single_part_02_part_03.rs");
