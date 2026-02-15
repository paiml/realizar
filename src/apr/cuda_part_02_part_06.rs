impl AprV2ModelCuda {

    /// GPU-accelerated forward pass.
    ///
    /// Computes logits for the given token sequence using GPU acceleration
    /// for matrix multiplications. Achieves 2x+ Ollama performance by using
    /// GPU GEMM for QKV, attention output, and FFN projections.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits vector of size `vocab_size` for next token prediction.
    pub fn forward_cuda(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        if !self.model.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let eps = self.model.metadata.rms_norm_eps.unwrap_or(1e-6);
        let seq_len = token_ids.len();
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // =========================================================================
        // FAST PATH: Use indexed Q4K GEMV kernels with CUDA graph capture
        // This path uses fused dequant+matmul kernels + graph replay for
        // 500x reduction in kernel launch overhead (5.6ms → 0.01ms per token)
        // Phase 45: Skip fast path when test_executor is present
        // PMAT-110: Skip fast path if KV cache was populated via fallback path
        //           (RoPE numerical differences cause inconsistency)
        // GH-201: Skip fast path in streaming mode (layer weights not pre-cached)
        // =========================================================================
        if self.test_executor.is_none()
            && self.executor.has_indexed_weights()
            && seq_len == 1
            && !self.fallback_kv_used
            && !self.streaming_mode
        {
            // Single-token decode: use the optimized Q4K GEMV path with graphs
            let token_id = token_ids[0];
            let position = self.kv_position;

            // Embedding lookup from cache (O(1) - no disk/mmap read)
            // Copy to local vec to release borrow before mutable executor call
            let input: Vec<f32> = self
                .get_embedding(token_id)
                .ok_or_else(|| RealizarError::InvalidShape {
                    reason: format!("Token {} out of embedding range", token_id),
                })?
                .to_vec();

            // Use the graphed forward path with CUDA graph capture
            // First call captures the graph, subsequent calls replay it
            let mut output = vec![0.0f32; vocab_size];
            self.executor
                .forward_all_layers_gpu_to_logits_graphed(
                    &input,
                    &mut output,
                    position,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_all_layers_gpu_to_logits_graphed".to_string(),
                    reason: format!("Q4K graphed fast path failed: {e}"),
                })?;

            // Increment position for next token (KV cache tracking)
            self.kv_position += 1;

            return Ok(output);
        }

        // =========================================================================
        // FALLBACK PATH: Original F32 GEMM path (for prefill or non-indexed models)
        // =========================================================================

        // BrickProfiler instrumentation (per spec §12.11)
        let profiling = self.executor.is_profiling_enabled();

        // 1. Token embedding lookup (CPU - fast single lookup)
        let timer_embed = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.Embed"))
        } else {
            None
        };

        let embed_name = self.model.find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
            "token_embd.weight", // GGUF naming
        ])?;
        let embeddings = self.model.get_tensor_f32(&embed_name)?;

        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= embeddings.len() {
                hidden.extend_from_slice(&embeddings[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        if let Some(t) = timer_embed {
            self.executor.profiler_mut().stop(t, seq_len as u64);
        }

        // PMAT-114: Layer tracing for Five-Whys analysis
        let trace_layers = std::env::var("APR_TRACE_LAYERS").is_ok();
        if trace_layers {
            // PMAT-114: Trace token IDs being processed
            eprintln!(
                "[PMAT-114] Input tokens ({} total): {:?}",
                token_ids.len(),
                &token_ids[..token_ids.len().min(20)]
            );
            if let Some(&last_token) = token_ids.last() {
                eprintln!("[PMAT-114] Last token ID: {}", last_token);
            }

            let last_hidden = &hidden[hidden.len() - hidden_dim..];
            let sum: f32 = last_hidden.iter().sum();
            let mean = sum / hidden_dim as f32;
            let min = last_hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = last_hidden
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[PMAT-114] After embed: mean={:.6}, min={:.6}, max={:.6}, first5={:?}",
                mean,
                min,
                max,
                &last_hidden[..5.min(hidden_dim)]
            );
        }

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Get weight tensors (HuggingFace, SafeTensors, GPT-2, LLaMA, GGUF)
            let attn_norm_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),
                &format!("transformer.h.{layer_idx}.ln_1.weight"),
                &format!("layers.{layer_idx}.attention_norm.weight"),
                &format!("blk.{layer_idx}.attn_norm.weight"), // GGUF
            ])?;

            // PMAT-APR-CUDA-002: Check for fused QKV first (from APR import)
            // APR import fuses Q/K/V into qkv_proj.weight for AprTransformer compatibility
            let fused_qkv_name = self.model.find_tensor_name(&[&format!(
                "model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            )]);
            let has_fused_qkv = fused_qkv_name.is_ok();

            // Only look for separate Q/K/V if fused is not available
            let (q_name, k_name, v_name) = if !has_fused_qkv {
                let q = self.model.find_tensor_name(&[
                    &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                    &format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                    &format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
                    &format!("layers.{layer_idx}.attention.wq.weight"),
                    &format!("blk.{layer_idx}.attn_q.weight"), // GGUF
                ])?;
                let k = self.model.find_tensor_name(&[
                    &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                    &format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                    &format!("transformer.h.{layer_idx}.attn.k_proj.weight"),
                    &format!("layers.{layer_idx}.attention.wk.weight"),
                    &format!("blk.{layer_idx}.attn_k.weight"), // GGUF
                ])?;
                let v = self.model.find_tensor_name(&[
                    &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                    &format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                    &format!("transformer.h.{layer_idx}.attn.v_proj.weight"),
                    &format!("layers.{layer_idx}.attention.wv.weight"),
                    &format!("blk.{layer_idx}.attn_v.weight"), // GGUF
                ])?;
                (Some(q), Some(k), Some(v))
            } else {
                (None, None, None)
            };

            let o_name = self.model.find_tensor_name(&[
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                &format!("transformer.h.{layer_idx}.attn.out_proj.weight"),
                &format!("layers.{layer_idx}.attention.wo.weight"),
                &format!("blk.{layer_idx}.attn_output.weight"), // GGUF
            ])?;

            let norm_weight = self.model.get_tensor_f32(&attn_norm_name)?;

            // RMSNorm (CPU - small operation)
            let timer_rmsnorm1 = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.RmsNorm"))
            } else {
                None
            };
            let normed = rms_norm(&hidden, &norm_weight, eps);
            if let Some(t) = timer_rmsnorm1 {
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // PMAT-114: Detailed layer 0 tracing
            if trace_layers && layer_idx == 0 {
                let last = &normed[normed.len() - hidden_dim..];
                let sum: f32 = last.iter().sum();
                let mean = sum / hidden_dim as f32;
                eprintln!(
                    "[PMAT-114] L0 after RMSNorm: mean={:.6}, first5={:?}",
                    mean,
                    last.get(..5).expect("hidden_dim is at least 5")
                );
            }

            // Q, K, V projections (GPU GEMM for 2x speedup)
            // Use cached weights if available (avoids repeated transpose + upload)
            // PMAT-805 FIX: Use GGUF-style cache names to match pre_cache_weights()
            let q_cache_name = format!("blk.{}.attn_q.weight", layer_idx);
            let k_cache_name = format!("blk.{}.attn_k.weight", layer_idx);
            let v_cache_name = format!("blk.{}.attn_v.weight", layer_idx);
            let o_cache_name = format!("blk.{}.attn_output.weight", layer_idx);

            let timer_qkv = if profiling {
                let _ = self.executor.synchronize();
                Some(self.executor.profiler_mut().start("apr.QKV"))
            } else {
                None
            };
            let (mut q, mut k, mut v) = if self.has_cached_weight(&q_cache_name) {
                // Fast path: use pre-cached transposed weights
                let q =
                    self.gemm_cached_gpu(&q_cache_name, &normed, seq_len, hidden_dim, hidden_dim)?;
                let k =
                    self.gemm_cached_gpu(&k_cache_name, &normed, seq_len, hidden_dim, kv_dim)?;
                let v =
                    self.gemm_cached_gpu(&v_cache_name, &normed, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            } else if has_fused_qkv {
                // PMAT-APR-CUDA-002: Handle fused QKV from APR import
                // qkv_proj.weight is [qkv_dim, hidden_dim] where qkv_dim = hidden_dim + 2*kv_dim
                let fused_name = fused_qkv_name.expect("checked above");
                let qkv_weight = self.model.get_tensor_f32(&fused_name)?;

                // Unfuse into Q, K, V: Q is first hidden_dim rows, K is next kv_dim, V is last kv_dim
                let q_size = hidden_dim * hidden_dim;
                let k_size = kv_dim * hidden_dim;
                let v_size = kv_dim * hidden_dim;

                if qkv_weight.len() < q_size + k_size + v_size {
                    return Err(RealizarError::InvalidShape {
                        reason: format!(
                            "Fused QKV weight too small: {} < {} (expected Q={}, K={}, V={})",
                            qkv_weight.len(),
                            q_size + k_size + v_size,
                            q_size,
                            k_size,
                            v_size
                        ),
                    });
                }

                let q_weight: Vec<f32> = qkv_weight[0..q_size].to_vec();
                let k_weight: Vec<f32> = qkv_weight[q_size..q_size + k_size].to_vec();
                let v_weight: Vec<f32> =
                    qkv_weight[q_size + k_size..q_size + k_size + v_size].to_vec();

                let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);
                let q = self.gemm_gpu(&normed, &q_weight_t, seq_len, hidden_dim, hidden_dim)?;
                let k = self.gemm_gpu(&normed, &k_weight_t, seq_len, hidden_dim, kv_dim)?;
                let v = self.gemm_gpu(&normed, &v_weight_t, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            } else {
                // Fallback: load separate Q/K/V, transpose, and upload weights each time
                let q_weight = self
                    .model
                    .get_tensor_f32(q_name.as_ref().expect("checked above"))?;
                let k_weight = self
                    .model
                    .get_tensor_f32(k_name.as_ref().expect("checked above"))?;
                let v_weight = self
                    .model
                    .get_tensor_f32(v_name.as_ref().expect("checked above"))?;
                let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);
                let q = self.gemm_gpu(&normed, &q_weight_t, seq_len, hidden_dim, hidden_dim)?;
                let k = self.gemm_gpu(&normed, &k_weight_t, seq_len, hidden_dim, kv_dim)?;
                let v = self.gemm_gpu(&normed, &v_weight_t, seq_len, hidden_dim, kv_dim)?;
                (q, k, v)
            };

            // PMAT-114: Apply QKV bias if present (Qwen2 has attention bias)
            self.apply_qkv_bias_for_layer(
                layer_idx, &mut q, &mut k, &mut v,
                hidden_dim, kv_dim, seq_len, trace_layers,
            )?;

            if let Some(t) = timer_qkv {
                let _ = self.executor.synchronize();
                self.executor.profiler_mut().stop(t, seq_len as u64);
            }

            // PMAT-114: Trace QKV for layer 0
            if trace_layers && layer_idx == 0 {
                let q_last = &q[q.len() - hidden_dim..];
                let k_last = &k[k.len() - kv_dim..];
                let v_last = &v[v.len() - kv_dim..];
                let q_mean: f32 = q_last.iter().sum::<f32>() / hidden_dim as f32;
                let k_mean: f32 = k_last.iter().sum::<f32>() / kv_dim as f32;
                let v_mean: f32 = v_last.iter().sum::<f32>() / kv_dim as f32;
                eprintln!(
                    "[PMAT-114] L0 after QKV: Q mean={:.6}, K mean={:.6}, V mean={:.6}",
                    q_mean, k_mean, v_mean
                );
                eprintln!("[PMAT-114] L0 Q first5={:?}", q_last.get(..5).expect("Q projection has at least 5 elements"));
                eprintln!(
                    "[PMAT-114] L0 shapes: q={}, k={}, v={}, hidden_dim={}, kv_dim={}",
                    q.len(),
                    k.len(),
                    v.len(),
                    hidden_dim,
                    kv_dim
                );
                eprintln!(
                    "[PMAT-114] L0 has_fused_qkv={}, has_cached_q={}",
                    has_fused_qkv,
                    self.has_cached_weight(&q_cache_name)
                );
            }

            // PMAT-110: Attention + output projection + residual
            self.forward_cuda_attention_layer(
                layer_idx, &q, &k, &v, &mut hidden,
                seq_len, hidden_dim, kv_dim,
                num_heads, num_kv_heads, head_dim,
                &o_name, &o_cache_name, profiling,
            )?;

            // FFN + residual + layer tracing
            self.forward_cuda_ffn_layer(
                layer_idx, &mut hidden,
                seq_len, hidden_dim, intermediate_dim,
                eps, num_layers, profiling, trace_layers,
            )?;
        }

        // PMAT-110: Update KV cache position after processing all tokens
        // This ensures subsequent forward_single_cuda calls have correct context
        self.kv_position += seq_len as u32;
        // PMAT-110: Mark that FALLBACK PATH was used for KV cache
        // Subsequent decode calls must also use FALLBACK PATH for consistency
        self.fallback_kv_used = true;

        // 3. Final layer norm (CPU)
        let timer_finalnorm = if profiling {
            Some(self.executor.profiler_mut().start("apr.FinalNorm"))
        } else {
            None
        };
        let final_norm_name = self.model.find_tensor_name(&[
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight", // GGUF naming
        ])?;
        let final_norm = self.model.get_tensor_f32(&final_norm_name)?;
        let hidden = rms_norm(&hidden, &final_norm, eps);
        if let Some(t) = timer_finalnorm {
            self.executor.profiler_mut().stop(t, 1); // Final norm processes 1 token (last)
        }

        // 4. LM head projection (GPU GEMM for large vocab)
        // Get hidden state for last token only
        let last_hidden = &hidden[hidden.len() - hidden_dim..];

        let timer_lmhead = if profiling {
            let _ = self.executor.synchronize();
            Some(self.executor.profiler_mut().start("apr.LmHead"))
        } else {
            None
        };
        // LM head: [1, hidden_dim] × [hidden_dim, vocab_size] = [1, vocab_size]
        let logits = if self.has_cached_weight("lm_head") {
            // Fast path: use pre-cached transposed LM head
            self.gemm_cached_gpu("lm_head", last_hidden, 1, hidden_dim, vocab_size)?
        } else {
            // Fallback: load, transpose (if needed), and upload
            // BUG-APR-001: Add token_embd.weight for GGUF weight tying
            let lm_head_name = self.model.find_tensor_name(&[
                "lm_head.weight",
                "output.weight", // GGUF uses this
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "token_embd.weight", // GGUF tied embeddings
            ])?;
            let lm_head = self.model.get_tensor_f32(&lm_head_name)?;

            // BUG-APR-001-FIX: Detect weight tying and handle transposed layout
            // GGUF token_embd.weight is stored as [hidden_dim, vocab_size] - already correct for GEMM
            // Regular lm_head.weight is stored as [vocab_size, hidden_dim] - needs transpose
            let is_tied_embedding = lm_head_name == "token_embd.weight"
                || lm_head_name.ends_with("embed_tokens.weight");

            let lm_head_for_gemm = if is_tied_embedding && lm_head.len() == hidden_dim * vocab_size
            {
                // Tied embedding: already [hidden_dim, vocab_size], use as-is
                lm_head.clone()
            } else {
                // Regular lm_head: [vocab_size, hidden_dim], need transpose to [hidden_dim, vocab_size]
                transpose_matrix(&lm_head, vocab_size, hidden_dim)
            };
            self.gemm_gpu(last_hidden, &lm_head_for_gemm, 1, hidden_dim, vocab_size)?
        };
        if let Some(t) = timer_lmhead {
            let _ = self.executor.synchronize();
            self.executor.profiler_mut().stop(t, 1); // LM head processes 1 token (last)
        }

        Ok(logits)
    }
}

include!("cuda_part_02_part_06_helpers.rs");
