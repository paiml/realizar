
impl OwnedQuantizedModelCuda {

    /// PAR-112: True token-by-token streaming generation
    ///
    /// Generates tokens one at a time and calls the callback after each token.
    /// The callback receives the token ID and can return `false` to stop generation early.
    ///
    /// This enables true real-time streaming where each token is delivered
    /// as soon as it's generated, rather than pseudo-streaming where all tokens
    /// are generated first then iterated.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `on_token` - Callback called for each generated token, returns `false` to stop
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.generate_gpu_resident_streaming(&prompt, &config, |token_id| {
    ///     println!("Generated: {}", token_id);
    ///     true // continue generation
    /// })?;
    /// ```
    pub fn generate_gpu_resident_streaming<F>(
        &mut self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // GH-167: Check context length BEFORE GPU dispatch to return clean error
        if prompt.len() > self.model.config.context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: prompt.len(),
                maximum: self.model.config.context_length,
            });
        }

        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident_streaming".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }

        // Create KV cache with GQA-aware dimensions
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut cache = OwnedQuantizedKVCache::new(
            self.model.config.num_layers,
            kv_dim,
            prompt.len() + config.max_tokens,
        );

        // Reset GPU KV cache positions and clear stale graph state
        // PMAT-PREFILL-FIX: Must clear position_buf to avoid indirect scatter using stale position
        self.executor.reset_kv_cache_gpu();
        self.executor.clear_decode_graph();

        let mut tokens = prompt.to_vec();

        // Serial prefill is the default path. Set BATCHED_PREFILL=1 to opt-in
        // to batched prefill for profiling.
        let use_batched_prefill = std::env::var("BATCHED_PREFILL")
            .map(|v| v == "1")
            .unwrap_or(false);
        let prefill_count = prompt.len() - 1;
        if prefill_count > 0 && !use_batched_prefill {
            // Serial prefill: process tokens one at a time (correct output)
            for (pos, &token_id) in prompt.iter().enumerate() {
                if pos < prompt.len() - 1 {
                    let _ = self.forward_gpu_resident(token_id, &mut cache, pos)?;
                }
            }
        } else if prefill_count > 0 {
            let prefill_tokens = &prompt[..prefill_count];
            let hidden_dim = self.model.config.hidden_dim;
            let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
            let num_layers = self.model.config.num_layers;
            let eps = self.model.config.eps;

            let embeddings = self.model.embed(prefill_tokens);
            let positions: Vec<u32> = (0..prefill_count as u32).collect();

            self.executor
                .init_prefill_workspace(prefill_count, hidden_dim, intermediate_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_prefill_workspace".to_string(),
                    reason: format!("Prefill workspace init failed: {e}"),
                })?;

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

            self.executor
                .init_workspace(hidden_dim, intermediate_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_workspace".to_string(),
                    reason: format!("Workspace restore failed: {e}"),
                })?;

            self.executor.clear_decode_graph();
        }

        // Generate from last prompt token
        let mut position = prompt.len() - 1;
        let mut last_token = prompt[prompt.len() - 1];

        for _token_num in 0..config.max_tokens {
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                self.forward_gpu_resident_to_token_id(last_token, &mut cache, position)?
            } else {
                let logits = self.forward_gpu_resident(last_token, &mut cache, position)?;
                OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // PAR-112: Call the streaming callback IMMEDIATELY after generating each token
            // If callback returns false, stop generation early
            if !on_token(next_token) {
                break;
            }

            last_token = next_token;
            position += 1;
        }

        Ok(tokens)
    }

    /// PAR-106: Batched GPU-resident generation for continuous batching
    ///
    /// Processes multiple prompts concurrently with true weight sharing:
    /// - Single weight read produces N tokens (one per active request)
    /// - Target: 400 tok/s (2x Ollama) with 4+ concurrent requests
    ///
    /// Key optimization: Uses `forward_batch_with_cache_cuda_native` which
    /// amortizes memory bandwidth across the batch.
    pub fn generate_batch_gpu_resident(
        &mut self,
        prompts: &[Vec<u32>],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<Vec<u32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        // Check architecture support
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batch_gpu_resident".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }

        let num_prompts = prompts.len();
        let max_prompt_len = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let max_seq_len = max_prompt_len + config.max_tokens;

        // PAR-045: Create KV caches with GQA-aware dimensions
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let mut caches: Vec<OwnedQuantizedKVCache> = (0..num_prompts)
            .map(|_| OwnedQuantizedKVCache::new(self.model.config.num_layers, kv_dim, max_seq_len))
            .collect();

        // Reset GPU KV cache positions
        self.executor.reset_kv_cache_gpu();

        // Initialize token sequences
        let mut sequences: Vec<Vec<u32>> = prompts.to_vec();
        let mut done: Vec<bool> = vec![false; num_prompts];

        // Prefill: Process each prompt's tokens (can't batch different lengths easily)
        for (prompt_idx, prompt) in prompts.iter().enumerate() {
            for (pos, &token_id) in prompt.iter().enumerate() {
                if pos < prompt.len() - 1 {
                    // PAR-106: Use single-token forward for prefill
                    // (batched prefill would require padding/masking complexity)
                    let _ = self.forward_gpu_resident(token_id, &mut caches[prompt_idx], pos)?;
                }
            }
        }

        // Track positions per prompt
        let mut positions: Vec<usize> = prompts.iter().map(|p| p.len() - 1).collect();
        let mut last_tokens: Vec<u32> = prompts.iter().map(|p| p[p.len() - 1]).collect();

        // PAR-106: Batched decode loop with weight sharing
        for _gen_idx in 0..config.max_tokens {
            // Collect active prompts
            let active_indices: Vec<usize> = (0..num_prompts).filter(|&i| !done[i]).collect();

            if active_indices.is_empty() {
                break;
            }

            // PAR-106/PAR-108: Sequential CUDA graphs outperform batched CPU path.
            // The batched GEMV kernel is 15x faster, but CUDA graphs amortize
            // kernel launch overhead which is more impactful. Batched path achieves
            // ~225 tok/s vs ~360 tok/s for sequential graphs.
            //
            // To achieve 2x Ollama (400 tok/s), need multi-token CUDA graph capture
            // that batches M tokens into a single graph execution.
            for &prompt_idx in &active_indices {
                let next_token = self.forward_gpu_resident_to_token_id(
                    last_tokens[prompt_idx],
                    &mut caches[prompt_idx],
                    positions[prompt_idx],
                )?;

                if config.stop_tokens.contains(&next_token) {
                    done[prompt_idx] = true;
                } else {
                    sequences[prompt_idx].push(next_token);
                    last_tokens[prompt_idx] = next_token;
                    positions[prompt_idx] += 1;

                    if sequences[prompt_idx].len() >= max_seq_len {
                        done[prompt_idx] = true;
                    }
                }
            }
        }

        Ok(sequences)
    }
}

include!("generate_2.rs");
