/// PMAT-044: Streaming batched generation for continuous batching
///
/// Processes M requests concurrently using batched GEMV (weight sharing):
/// - Weights read ONCE per decode step for all M sequences
/// - Per-slot GPU KV caches (PAR-119)
/// - Per-slot streaming callbacks for SSE token delivery
///
/// # Performance Target
///
/// Jetson Orin Nano Super (c=4):
/// - Current (sequential): 34.3 tok/s aggregate
/// - Target (batched): 50-80 tok/s via weight amortization
impl OwnedQuantizedModelCuda {
    /// Generate tokens for M requests with per-request streaming callbacks.
    ///
    /// Each callback receives token IDs as they're generated and can return
    /// `false` to stop that request early (EOS or client disconnect).
    ///
    /// Uses PAR-111 batched GEMV + PAR-119 per-slot GPU KV caches for
    /// true weight sharing across concurrent requests.
    pub fn generate_batched_streaming(
        &mut self,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>>,
    ) -> Result<Vec<Vec<u32>>> {
        let m = prompts.len();
        eprintln!("[PMAT-044] generate_batched_streaming ENTRY m={m}");
        self.validate_batch_args(m, configs.len(), on_tokens.len())?;
        if m == 0 {
            return Ok(Vec::new());
        }

        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        // PMAT-044 FIX: Clear decode graph from prior M=1 requests. prefill_and_scatter
        // uses forward_gpu_resident which replays the M=1 decode graph. During graph
        // replay, CPU-side kv_cache_lengths is NOT updated (only GPU kernels run),
        // causing KV cache state mismatch. Force eager path for prefill by clearing graph.
        self.executor.clear_decode_graph();

        // GH-141: Clear batched decode graphs — workspace buffers may have been
        // reallocated between batches (init_workspace restores M=1 sizing).
        // A stale graph would reference freed buffer addresses → ILLEGAL_ADDRESS.
        self.executor.clear_batched_decode_graphs();

        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;
        let max_tokens_max = configs.iter().map(|c| c.max_tokens).max().unwrap_or(128);
        let max_prompt_len = prompts.iter().map(Vec::len).max().unwrap_or(0);

        // Create CPU KV caches for each slot (needed for prefill path)
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = self.model.config.hidden_dim / self.model.config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let max_seq_len = max_prompt_len + max_tokens_max;
        let mut caches: Vec<OwnedQuantizedKVCache> = (0..m)
            .map(|_| OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq_len))
            .collect();

        // GH-141: Free FP16 weight cache and prefill graphs before allocating batched KV.
        // On 8GB GPUs, keeping FP16 cache (2944 MB) alongside batched KV causes memory
        // pressure that slows HGEMM prefill from ~100ms to >2000ms per slot (VRAM thrash).
        // SGEMM prefill at ~100ms/slot is faster than thrashing HGEMM at ~2300ms/slot.
        self.executor.clear_fp16_weight_cache();
        self.executor.clear_prefill_graphs();

        // PMAT-045: Pre-allocate workspace for max_prompt_len BEFORE prefill.
        // Uses init_prefill_workspace (no 32-batch limit) to size buffers for prefill.
        // The buffer_capacity high-water mark ensures init_batched_workspace(m) skips
        // reallocation since buffer_capacity >= max_prompt_len >= m.
        self.executor
            .init_prefill_workspace(hidden_dim, intermediate_dim, max_prompt_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_prefill_workspace".to_string(),
                reason: format!("Failed to pre-allocate workspace for seq_len={max_prompt_len}: {e}"),
            })?;

        // Init batched KV caches for scatter targets
        self.executor
            .init_batched_kv_cache_gpu(num_layers, m)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_kv_cache_gpu".to_string(),
                reason: format!("Failed to init batched KV cache for M={m}: {e}"),
            })?;
        self.executor.reset_batched_kv_cache_gpu();

        // Prefill each slot sequentially (SGEMM — FP16 cache cleared above),
        // then scatter GPU KV to batched cache via D2D memcpy.
        let (mut sequences, mut positions, mut last_tokens) =
            self.prefill_and_scatter(prompts, &mut caches)?;

        // PMAT-045: Set workspace.batch_size = m for decode kernels (they check == m).
        // buffer_capacity stays at max_prompt_len (high-water mark), so this call
        // skips reallocation and just sets the logical batch_size.
        self.executor
            .init_batched_workspace(hidden_dim, intermediate_dim, m)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_workspace".to_string(),
                reason: format!("Failed to set decode batch_size={m}: {e}"),
            })?;

        // PMAT-044 FIX: Clear M=1 decode graph — prefill_and_scatter may have
        // captured an M=1 graph with stale addresses. init_batched_workspace skips
        // reallocation if batch_size matches (PMAT-045), preserving batched graphs.
        self.executor.clear_decode_graph();

        // Decode phase: batched
        let mut done = vec![false; m];
        self.decode_batched(
            m,
            max_tokens_max,
            hidden_dim,
            intermediate_dim,
            num_layers,
            vocab_size,
            eps,
            prompts,
            configs,
            on_tokens,
            &mut sequences,
            &mut positions,
            &mut last_tokens,
            &mut done,
        )?;

        // Restore single-token workspace for non-batched requests
        let _ = self
            .executor
            .init_workspace(hidden_dim, intermediate_dim);

        // PMAT-044 FIX: Reset batched KV state so subsequent M=1 prefill doesn't
        // take the batched attention path (batched_ffn.rs line 37). Without this,
        // M=1 prefill writes K/V to batched caches while M=1 decode reads from the
        // single KV cache (empty) → EOS on first token → 0 output tokens.
        self.executor.batched_kv_stride = 0;

        Ok(sequences)
    }

    fn validate_batch_args(&self, m: usize, configs_len: usize, callbacks_len: usize) -> Result<()> {
        if m > 32 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batched_streaming".to_string(),
                reason: format!("PMAT-044: batch size {m} exceeds max 32"),
            });
        }
        if configs_len != m || callbacks_len != m {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batched_streaming".to_string(),
                reason: format!(
                    "PMAT-044: prompts({m})/configs({configs_len})/callbacks({callbacks_len}) length mismatch"
                ),
            });
        }
        if !self.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_batched_streaming".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }
        Ok(())
    }

    fn prefill_and_scatter(
        &mut self,
        prompts: &[Vec<u32>],
        caches: &mut [OwnedQuantizedKVCache],
    ) -> Result<(Vec<Vec<u32>>, Vec<usize>, Vec<u32>)> {
        let m = prompts.len();
        let sequences: Vec<Vec<u32>> = prompts.to_vec();
        let mut positions: Vec<usize> = Vec::with_capacity(m);
        let mut last_tokens: Vec<u32> = Vec::with_capacity(m);
        let saved_stride = self.executor.batched_kv_stride;

        eprintln!("[PMAT-044] Batched streaming: {m} slots, prefilling...");

        for (slot_idx, prompt) in prompts.iter().enumerate() {
            self.executor.reset_kv_cache_gpu();
            let seq_len = prompt.len().saturating_sub(1);

            // PMAT-037: Use cuBLAS batched prefill (HGEMM) instead of serial token-by-token.
            // Five-Whys: serial prefill at c=4 causes 854ms TTFT (16x vs c=1).
            // cuBLAS prefill processes all prompt tokens at once via HGEMM, 4.9x faster.
            self.executor.batched_kv_stride = 0;
            self.run_prefill(prompt, &mut caches[slot_idx], seq_len, false)?;

            // Restore stride for D2D scatter: single GPU KV → batched KV slot
            self.executor.batched_kv_stride = saved_stride;
            self.executor
                .scatter_single_kv_to_batched(slot_idx, seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "scatter_single_kv_to_batched".to_string(),
                    reason: format!("Failed to scatter KV for slot {slot_idx}: {e}"),
                })?;

            positions.push(seq_len);
            last_tokens.push(prompt[prompt.len() - 1]);
        }

        // Leave stride restored for decode phase
        eprintln!("[PMAT-044] Prefill done, starting batched decode...");
        Ok((sequences, positions, last_tokens))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &mut self,
        m: usize,
        max_tokens_max: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        eps: f32,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        mut on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>>,
        sequences: &mut [Vec<u32>],
        positions: &mut [usize],
        last_tokens: &mut [u32],
        done: &mut [bool],
    ) -> Result<()> {
        let mut embed_buf = vec![0.0f32; m * hidden_dim];

        for gen_idx in 0..max_tokens_max {
            if done.iter().all(|&d| d) {
                break;
            }

            // Embed all tokens
            for slot_idx in 0..m {
                let offset = slot_idx * hidden_dim;
                if done[slot_idx] {
                    embed_buf[offset..offset + hidden_dim].fill(0.0);
                } else {
                    self.model
                        .embed_into(last_tokens[slot_idx], &mut embed_buf[offset..offset + hidden_dim]);
                }
            }

            let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();

            if gen_idx < 3 {
                eprintln!("[PMAT-044] decode step {gen_idx}: positions={pos_u32:?}, last_tokens={last_tokens:?}");
            }

            // PMAT-056: Multi-stream root cause fixed (scatter moved to self.stream),
            // but graph replay still 25% slower than eager due to capture overhead.
            // Keep eager by default until graph replay is optimized.
            // Enable with BATCHED_GRAPH=1 for testing.
            let use_graph = std::env::var("BATCHED_GRAPH").as_deref() == Ok("1");
            let token_ids = if use_graph {
                self.executor
                    .forward_batched_to_token_ids_graphed(
                        &embed_buf,
                        &pos_u32,
                        num_layers,
                        hidden_dim as u32,
                        intermediate_dim as u32,
                        vocab_size as u32,
                        eps,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "forward_batched_to_token_ids_graphed".to_string(),
                        reason: format!("Batched forward failed: {e}"),
                    })?
            } else {
                self.executor
                    .forward_batched_to_token_ids(
                        &embed_buf,
                        &pos_u32,
                        num_layers,
                        hidden_dim as u32,
                        intermediate_dim as u32,
                        vocab_size as u32,
                        eps,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "forward_batched_to_token_ids".to_string(),
                        reason: format!("Batched forward failed: {e}"),
                    })?
            };

            if gen_idx < 3 {
                eprintln!("[PMAT-044] decode step {gen_idx}: token_ids={token_ids:?}, done={done:?}");
            }

            self.distribute_tokens(
                m, prompts, configs, &mut on_tokens, &token_ids, sequences, positions,
                last_tokens, done,
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn distribute_tokens(
        &self,
        m: usize,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        on_tokens: &mut [Box<dyn FnMut(u32) -> bool + Send>],
        token_ids: &[u32],
        sequences: &mut [Vec<u32>],
        positions: &mut [usize],
        last_tokens: &mut [u32],
        done: &mut [bool],
    ) {
        for slot_idx in 0..m {
            if done[slot_idx] {
                continue;
            }

            let next_token = token_ids[slot_idx];

            if configs[slot_idx].stop_tokens.contains(&next_token) {
                done[slot_idx] = true;
                continue;
            }

            sequences[slot_idx].push(next_token);

            if !on_tokens[slot_idx](next_token) {
                done[slot_idx] = true;
                continue;
            }

            last_tokens[slot_idx] = next_token;
            positions[slot_idx] += 1;

            let generated = sequences[slot_idx].len() - prompts[slot_idx].len();
            if generated >= configs[slot_idx].max_tokens {
                done[slot_idx] = true;
            }
        }
    }
}
