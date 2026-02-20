impl OwnedQuantizedModelCuda {
    /// Generate tokens using CUDA acceleration
    ///
    /// Uses `forward_cuda` for each token generation step.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration (max_tokens, temperature, etc.)
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_cuda(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        let mut tokens = prompt.to_vec();

        for _ in 0..config.max_tokens {
            let logits = self.forward_cuda(&tokens)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);

                // Apply temperature and sample (simplified - take max after temperature)
                let max_logit = indexed[0].1;
                let _exp_sum: f32 = indexed
                    .iter()
                    .map(|(_, l)| ((l - max_logit) / config.temperature).exp())
                    .sum();

                // Take argmax (proper probabilistic sampling would use exp_sum for normalization)
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Generate tokens using CUDA with KV cache
    ///
    /// Uses `forward_single_cuda_with_cache` for incremental decoding with KV cache.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        let mut tokens = prompt.to_vec();

        // Process prompt tokens
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                // Just populate the cache
                let _ = self.forward_single_cuda_with_cache(token_id, &mut cache, pos)?;
            }
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _ in 0..config.max_tokens {
            let logits = self.forward_single_cuda_with_cache(last_token, &mut cache, position)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// IMP-1010: Full GPU-accelerated token generation
    ///
    /// Uses `forward_single_full_cuda_with_cache` for maximum GPU utilization.
    /// All matmul operations (5 per layer) run on GPU.
    ///
    /// # Performance Target
    ///
    /// - CPU path: ~5 tok/s (limited by memory bandwidth)
    /// - Full GPU path: ~200 tok/s (matching Ollama)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_full_cuda_with_cache(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill) - use full GPU path
        for (pos, &token_id) in prompt.iter().enumerate() {
            if pos < prompt.len() - 1 {
                // Just populate the cache
                let _ = self.forward_single_full_cuda_with_cache(token_id, &mut cache, pos)?;
            }
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _ in 0..config.max_tokens {
            let logits =
                self.forward_single_full_cuda_with_cache(last_token, &mut cache, position)?;

            // Greedy sampling (temperature=0)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx as u32)
            } else {
                // Top-k sampling
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(config.top_k);
                indexed[0].0 as u32
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            // PAR-050-DEBUG: Print sampled tokens
            if tokens.len() <= 15 {
                eprintln!(
                    "[PAR-050] Generated token {}: {} (position {})",
                    tokens.len() - prompt.len() + 1,
                    next_token,
                    position
                );
            }

            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// GPU-resident token generation with minimal CPU↔GPU transfers.
    ///
    /// # Reentrant
    ///
    /// This method creates fresh generation state on each call (new KV cache,
    /// reset GPU positions). It is safe and efficient to call multiple times
    /// on the same `OwnedQuantizedModelCuda` — weights are preloaded once
    /// during construction and reused across calls.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration (max_tokens, temperature, etc.)
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn generate_gpu_resident(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // GH-167: Check context length BEFORE GPU dispatch to return clean error
        // instead of cryptic CUDA_ERROR_UNKNOWN when attention matrix exceeds allocated size
        if prompt.len() > self.model.config.context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: prompt.len(),
                maximum: self.model.config.context_length,
            });
        }

        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        // (context may have been created on a different thread, e.g., main vs tokio worker)
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident".to_string(),
                reason: "Model architecture not supported for GPU-resident path (requires separate Q/K/V, SwiGLU, RMSNorm)".to_string(),
            });
        }

        // PAR-045: Create KV cache with GQA-aware dimensions
        // For GQA models, K/V have kv_dim = num_kv_heads * head_dim (smaller than hidden_dim)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim, // GQA: use kv_dim instead of hidden_dim
            prompt.len() + config.max_tokens,
        );

        // PAR-055 FIX: Reset GPU KV cache positions before new generation
        // Without this, cache positions accumulate across generate calls causing degradation
        self.executor.reset_kv_cache_gpu();

        // PMAT-PREFILL-FIX: Clear stale graph state (position_buf, seq_len_buf) from previous
        // generation. validate_gpu_first_token() captures a graph that sets position_buf=Some(0).
        // If batched prefill runs with stale position_buf, the KV scatter uses INDIRECT mode
        // which reads position from the device buffer (always 0), so ALL tokens scatter to
        // position 0 instead of their correct positions.
        self.executor.clear_decode_graph();

        let mut tokens = prompt.to_vec();

        if config.trace {
            eprintln!(
                "[TRACE-CACHE] GGUF model (GPU): {} layers, hidden_dim={}, vocab={}",
                self.model.config.num_layers,
                self.model.config.hidden_dim,
                self.model.config.vocab_size
            );
            eprintln!(
                "[TRACE-CACHE] Prefill: {} tokens, max_gen={}",
                prompt.len(),
                config.max_tokens
            );
        }

        // Serial prefill produces correct output. Batched prefill is opt-in via
        // BATCHED_PREFILL=1 for profiling the BatchedQ4KGemvKernel path.
        let use_serial_prefill = !std::env::var("BATCHED_PREFILL")
            .map(|v| v == "1")
            .unwrap_or(false);
        let prefill_start = std::time::Instant::now();
        let prefill_count = prompt.len() - 1; // All except last (last feeds into decode)
        if use_serial_prefill && prefill_count > 0 {
            for (pos, &token_id) in prompt.iter().enumerate() {
                if pos < prompt.len() - 1 {
                    let _ = self.forward_gpu_resident(token_id, &mut cache, pos)?;
                }
            }
            if config.trace {
                eprintln!(
                    "[TRACE-PREFILL] Serial prefill: {} tokens in {:?}",
                    prefill_count,
                    prefill_start.elapsed()
                );
            }
        } else if prefill_count > 0 {
            let prefill_tokens = &prompt[..prefill_count];
            let hidden_dim = self.model.config.hidden_dim;
            let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
            let num_layers = self.model.config.num_layers;
            let eps = self.model.config.eps;

            // Embed all prefill tokens at once
            let embeddings = self.model.embed(prefill_tokens);
            let positions: Vec<u32> = (0..prefill_count as u32).collect();

            // Initialize prefill workspace for S tokens
            self.executor
                .init_prefill_workspace(prefill_count, hidden_dim, intermediate_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_prefill_workspace".to_string(),
                    reason: format!("Prefill workspace init failed: {e}"),
                })?;

            // Run batched prefill through all layers
            self.executor
                .prefill_all_layers_gpu(
                    &embeddings,
                    &positions,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "prefill_all_layers_gpu".to_string(),
                    reason: format!("Batched prefill failed: {e}"),
                })?;

            // Restore decode workspace (M=1 for token-by-token decode)
            self.executor
                .init_workspace(hidden_dim, intermediate_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_workspace".to_string(),
                    reason: format!("Workspace restore failed: {e}"),
                })?;

            // Clear decode graph — the graph was captured for M=1 at an old position.
            // The next decode call will re-capture with position=prefill_count.
            self.executor.clear_decode_graph();

            if config.trace {
                eprintln!(
                    "[TRACE-PREFILL] Batched prefill: {} tokens in {:?} ({:.1} tok/s)",
                    prefill_count,
                    prefill_start.elapsed(),
                    prefill_count as f64 / prefill_start.elapsed().as_secs_f64()
                );
            }
        } else if config.trace {
            eprintln!("[TRACE-PREFILL] Single token prompt, no prefill needed");
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _token_num in 0..config.max_tokens {
            let token_start = std::time::Instant::now();
            // PAR-062: Use GPU argmax path for greedy sampling (150,000x data transfer reduction)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                // Greedy sampling - use GPU-side argmax (4 bytes transfer vs 600KB)
                self.forward_gpu_resident_to_token_id(last_token, &mut cache, position)?
            } else {
                // Non-greedy sampling - need full logits for proper temperature + top-k sampling
                // PAR-063: Resolved issue where GPU path always took top token instead of sampling
                let logits = self.forward_gpu_resident(last_token, &mut cache, position)?;
                OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };

            if config.trace {
                eprintln!(
                    "[TRACE-CACHE] pos={}: {} layers took {:?}",
                    position,
                    self.model.config.num_layers,
                    token_start.elapsed()
                );
            }

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }
}
