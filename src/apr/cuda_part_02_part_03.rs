impl AprV2ModelCuda {

    // ========================================================================
    // Weight Pre-caching (2x performance optimization)
    // ========================================================================

    /// Pre-cache all model weights on GPU using native quantized format.
    ///
    /// This uploads quantized weights (Q4K, Q6K, etc.) directly to GPU without
    /// CPU dequantization, enabling fused dequant+matmul kernels for maximum
    /// throughput (2x+ Ollama baseline per APR mandate).
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU.
    fn pre_cache_weights(&mut self) -> Result<()> {
        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let _vocab_size = self.model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };
        let kv_dim = num_kv_heads * head_dim;

        if hidden_dim == 0 || num_layers == 0 {
            return Ok(()); // Non-transformer model, nothing to cache
        }

        let mut total_bytes = 0usize;
        let mut quantized_count = 0usize;

        // Helper to upload a weight tensor (quantized or F32)
        // Uses GGUF-style cache names for compatibility with build_indexed_weights()
        // PMAT-113: Now caches F32 weights for GPU GEMM (was causing APR CUDA hang)
        let upload_weight = |executor: &mut crate::cuda::CudaExecutor,
                             model: &AprV2Model,
                             src_name: &str,
                             cache_name: &str|
         -> (usize, bool) {
            // Returns (bytes_uploaded, is_quantized)
            if let Some(entry) = model.get_tensor(src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized: upload raw bytes to quantized_weight_cache
                    if let Ok(bytes) = model.get_tensor_bytes(src_name) {
                        let size = executor
                            .load_quantized_weights_with_type(cache_name, bytes, qtype)
                            .unwrap_or(0);
                        (size, true)
                    } else {
                        (0, false)
                    }
                } else {
                    // PMAT-113: F32/F16 - cache on GPU for GEMM path
                    // PMAT-222: Transpose 2D F32 weights from [n, k] to [k, n] for gemm_b_cached
                    // HF convention stores weights as [out_dim, in_dim] but GEMM needs B[k, n]
                    if let Ok(weights) = model.get_tensor_f32(src_name) {
                        let final_weights = if entry.shape.len() == 2 {
                            let rows = entry.shape[0]; // out_dim (n)
                            let cols = entry.shape[1]; // in_dim (k)
                            let mut transposed = vec![0.0f32; weights.len()];
                            for i in 0..rows {
                                for j in 0..cols {
                                    transposed[j * rows + i] = weights[i * cols + j];
                                }
                            }
                            transposed
                        } else {
                            weights
                        };
                        let size = executor
                            .load_weights(cache_name, &final_weights)
                            .unwrap_or(0);
                        (size, false)
                    } else {
                        (0, false)
                    }
                }
            } else {
                (0, false)
            }
        };

        // Track F32 weight count for fallback path
        let mut f32_weight_count = 0usize;

        // Cache per-layer weights using GGUF naming convention
        // This matches build_indexed_weights() expectations
        for layer_idx in 0..num_layers {
            let prefix = format!("blk.{layer_idx}");

            // Find source tensor names (HuggingFace, GGUF, etc.)
            // Map from various naming conventions to GGUF cache names
            let weight_mappings = [
                // (source_patterns, cache_suffix)
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                        format!("blk.{layer_idx}.attn_q.weight"),
                    ],
                    "attn_q.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                        format!("blk.{layer_idx}.attn_k.weight"),
                    ],
                    "attn_k.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                        format!("blk.{layer_idx}.attn_v.weight"),
                    ],
                    "attn_v.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                        format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                        format!("blk.{layer_idx}.attn_output.weight"),
                    ],
                    "attn_output.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("blk.{layer_idx}.ffn_gate.weight"),
                    ],
                    "ffn_gate.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("blk.{layer_idx}.ffn_up.weight"),
                    ],
                    "ffn_up.weight",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("blk.{layer_idx}.ffn_down.weight"),
                    ],
                    "ffn_down.weight",
                ),
            ];

            for (patterns, suffix) in weight_mappings {
                let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
                if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                    let cache_name = format!("{prefix}.{suffix}");
                    let (bytes, is_quantized) =
                        upload_weight(&mut self.executor, &self.model, &src_name, &cache_name);
                    if bytes > 0 {
                        total_bytes += bytes;
                        if is_quantized {
                            quantized_count += 1;
                        } else {
                            f32_weight_count += 1;
                        }
                    }
                }
            }

            // PMAT-113: Cache fused QKV from APR import (PMAT-101)
            // APR models from HuggingFace have Q/K/V fused into qkv_proj.weight
            // Unfuse and cache as separate Q/K/V with names the forward path expects
            // NOTE: P1 quality issue exists (SATD-WARNING in generate_cuda_with_cache)
            // The APR import has corrupt tensor layouts - this caching doesn't fix that
            let fused_qkv_patterns = vec![format!(
                "model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            )];
            let fused_patterns_ref: Vec<&str> =
                fused_qkv_patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&fused_patterns_ref) {
                // Load and unfuse QKV for F32 models
                if let Ok(qkv_weight) = self.model.get_tensor_f32(&src_name) {
                    // Unfuse: Q is first hidden_dim rows, K is next kv_dim, V is last kv_dim
                    let q_size = hidden_dim * hidden_dim;
                    let k_size = kv_dim * hidden_dim;
                    let v_size = kv_dim * hidden_dim;

                    if qkv_weight.len() >= q_size + k_size + v_size {
                        // Cache unfused Q/K/V with forward path naming convention
                        let q_weight: Vec<f32> = qkv_weight[0..q_size].to_vec();
                        let k_weight: Vec<f32> = qkv_weight[q_size..q_size + k_size].to_vec();
                        let v_weight: Vec<f32> =
                            qkv_weight[q_size + k_size..q_size + k_size + v_size].to_vec();

                        // PMAT-114: Trace K weight for layer 0 to debug 100x difference
                        if layer_idx == 0 && std::env::var("APR_TRACE_LAYERS").is_ok() {
                            let k_sum: f32 = k_weight.iter().sum();
                            let k_mean = k_sum / k_weight.len() as f32;
                            let k_min = k_weight.iter().cloned().fold(f32::INFINITY, f32::min);
                            let k_max = k_weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            eprintln!("[PMAT-114] L0 K weight (pre-transpose): mean={:.6}, min={:.6}, max={:.6}, len={}",
                                k_mean, k_min, k_max, k_weight.len());
                            eprintln!(
                                "[PMAT-114] L0 K weight first10={:?}",
                                &k_weight[..10.min(k_weight.len())]
                            );
                        }

                        // Transpose for GPU GEMM (row-major to column-major)
                        let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
                        let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
                        let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);

                        // PMAT-114: Trace K weight after transpose
                        if layer_idx == 0 && std::env::var("APR_TRACE_LAYERS").is_ok() {
                            let k_sum: f32 = k_weight_t.iter().sum();
                            let k_mean = k_sum / k_weight_t.len() as f32;
                            eprintln!(
                                "[PMAT-114] L0 K weight (post-transpose): mean={:.6}, len={}",
                                k_mean,
                                k_weight_t.len()
                            );
                            eprintln!(
                                "[PMAT-114] L0 K weight_t first10={:?}",
                                &k_weight_t[..10.min(k_weight_t.len())]
                            );
                        }

                        // Cache with GGUF-style naming to match forward path (PMAT-805)
                        let q_cache_name = format!("blk.{layer_idx}.attn_q.weight");
                        let k_cache_name = format!("blk.{layer_idx}.attn_k.weight");
                        let v_cache_name = format!("blk.{layer_idx}.attn_v.weight");

                        if let Ok(bytes) = self.executor.load_weights(&q_cache_name, &q_weight_t) {
                            total_bytes += bytes;
                            f32_weight_count += 1;
                        }
                        if let Ok(bytes) = self.executor.load_weights(&k_cache_name, &k_weight_t) {
                            total_bytes += bytes;
                            f32_weight_count += 1;
                        }
                        if let Ok(bytes) = self.executor.load_weights(&v_cache_name, &v_weight_t) {
                            total_bytes += bytes;
                            f32_weight_count += 1;
                        }
                    }
                }
            }

            // Weights are loaded via GGUF-style names (blk.{}.attn_output.weight etc.)
            // in the first pass above. Biases are read directly from the model at
            // inference time. This avoids duplicate GPU memory for 1.5B F32 models.

            // Upload RMSNorm gamma weights (always F32)
            let norm_mappings = [
                (
                    vec![
                        format!("model.layers.{layer_idx}.input_layernorm.weight"),
                        format!("layers.{layer_idx}.input_layernorm.weight"),
                        format!("blk.{layer_idx}.attn_norm.weight"),
                    ],
                    "attn_norm.gamma",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                        format!("layers.{layer_idx}.post_attention_layernorm.weight"),
                        format!("blk.{layer_idx}.ffn_norm.weight"),
                    ],
                    "ffn_norm.gamma",
                ),
                // GH-279: QK norm weights (Qwen3 per-head RMSNorm on Q and K)
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.q_norm.weight"),
                        format!("blk.{layer_idx}.attn_q_norm.weight"),
                    ],
                    "attn_q_norm.gamma",
                ),
                (
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.k_norm.weight"),
                        format!("blk.{layer_idx}.attn_k_norm.weight"),
                    ],
                    "attn_k_norm.gamma",
                ),
            ];

            for (patterns, suffix) in norm_mappings {
                let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
                if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                    if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                        let cache_name = format!("{prefix}.{suffix}");
                        if let Ok(bytes) = self.executor.cache_rmsnorm_gamma(&cache_name, &gamma) {
                            total_bytes += bytes;
                        }
                    }
                }
            }
        }

        // Cache output norm
        let output_norm_patterns = [
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight",
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&output_norm_patterns) {
            if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                if let Ok(bytes) = self
                    .executor
                    .cache_rmsnorm_gamma("output_norm.gamma", &gamma)
                {
                    total_bytes += bytes;
                }
            }
        }

        // Cache LM head (may be quantized or F32)
        let lm_head_patterns = [
            "lm_head.weight",
            "output.weight",
            "token_embd.weight", // GGUF (tied embeddings)
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&lm_head_patterns) {
            if let Some(entry) = self.model.get_tensor(&src_name) {
                if let Some(qtype) = dtype_to_ggml_qtype(&entry.dtype) {
                    // Quantized LM head
                    if let Ok(bytes) = self.model.get_tensor_bytes(&src_name) {
                        if let Ok(size) = self.executor.load_quantized_weights_with_type(
                            "output.weight",
                            bytes,
                            qtype,
                        ) {
                            total_bytes += size;
                            quantized_count += 1;
                        }
                    }
                } else {
                    // F32 LM head - store as quantized_weight_cache for compatibility
                    // The forward path will handle F32 appropriately
                    if let Ok(w) = self.model.get_tensor_f32(&src_name) {
                        // Upload F32 weights directly (no transpose needed for GEMV)
                        // SAFETY: f32 slice to u8 view - valid because f32 has no padding,
                        // alignment requirement of u8 is 1, and lifetime is preserved
                        let w_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                w.as_ptr().cast::<u8>(),
                                w.len() * std::mem::size_of::<f32>(),
                            )
                        };
                        // Use qtype 0 to indicate F32 (handled specially in forward)
                        if let Ok(size) = self.executor.load_quantized_weights_with_type(
                            "output.weight",
                            w_bytes,
                            0,
                        ) {
                            total_bytes += size;
                        }
                    }
                }
            }
        }

        // Build indexed weight lookup table for O(1) access during decode
        // This is the key optimization that enables fast token generation
        if quantized_count > 0 {
            // GH-279: Derive ArchConstraints from APR metadata for weight validation
            let arch_name = self.model.metadata.model_type.as_deref().unwrap_or("llama");
            let arch = crate::gguf::ArchConstraints::from_architecture(arch_name);
            if let Err(e) = self
                .executor
                .build_indexed_weights(num_layers, |i| format!("blk.{i}"), &arch)
            {
                eprintln!("[AprV2ModelCuda] Warning: Could not build indexed weights: {e}");
                // Continue anyway - fallback path will be used
            } else {
                eprintln!(
                    "[AprV2ModelCuda] Built indexed weights for {} layers",
                    num_layers
                );
            }

            // Initialize workspace for zero-allocation forward pass
            if let Err(e) = self.executor.init_workspace(hidden_dim, intermediate_dim) {
                eprintln!("[AprV2ModelCuda] Warning: Could not init workspace: {e}");
            }
        }

        // PMAT-113: Log both quantized and F32 weight counts
        eprintln!(
            "[AprV2ModelCuda] Pre-cached {} MB of weights on GPU ({} layers, {} quantized, {} F32 tensors)",
            total_bytes / (1024 * 1024),
            num_layers,
            quantized_count,
            f32_weight_count
        );

        Ok(())
    }
}
