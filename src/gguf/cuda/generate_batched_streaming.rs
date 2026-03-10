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

        // PMAT-045: Pre-allocate workspace for max_prompt_len BEFORE prefill.
        // buffer_capacity high-water mark ensures init_batched_workspace(m) skips
        // reallocation since buffer_capacity >= max_prompt_len >= m.
        self.executor
            .init_prefill_workspace(max_prompt_len, hidden_dim, intermediate_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_prefill_workspace".to_string(),
                reason: format!("Failed to pre-allocate workspace for seq_len={max_prompt_len}: {e}"),
            })?;

        // PMAT-060: Allocate batched KV alongside FP16 weight cache (both fit in VRAM).
        // Five-Whys: c=4 TTFT = 659ms (29.9x gap vs llama.cpp 22ms).
        // Why? SGEMM prefill at ~165ms/slot instead of HGEMM ~58ms/slot.
        // Why? FP16 weight cache (2944 MB) cleared before prefill (GH-141).
        // Why? Assumed FP16 + batched KV + Q4K > 8 GB VRAM.
        // Reality: FP16 (2944) + batched KV (56-896) + Q4K (850) = 3850-4700 MB.
        //          RTX 4060L has ~7.5 GB usable → fits with 2.8-3.6 GB headroom.
        // Fix: Keep FP16 during prefill, allocate batched KV, D2D scatter, THEN clear FP16.
        // GH-141 ILLEGAL_ADDRESS was from a different root cause (stale graph pointers).
        self.executor
            .init_batched_kv_cache_gpu(num_layers, m)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_kv_cache_gpu".to_string(),
                reason: format!("Failed to init batched KV cache for M={m}: {e}"),
            })?;
        self.executor.reset_batched_kv_cache_gpu();

        // Prefill each slot with HGEMM (FP16 present) + D2D scatter to batched KV.
        let (mut sequences, mut positions, mut last_tokens) =
            self.prefill_and_scatter(prompts, &mut caches)?;

        // PMAT-061: Keep FP16 weight cache during M>1 decode for cuBLAS HGEMM.
        // Five-Whys: c=4 decode 0.56x — Q4K GEMV compute-bound at M=4 (3.25x scaling).
        // cuBLAS HGEMM (tensor cores) is memory-bound — scales ~1x with M.
        // VRAM: FP16 (2944) + batched KV (896) + Q4K (850) = 4690 MB, fits in 7.5 GB.
        // FP16 cache stays for decode, cleared on cleanup after batch completes.
        let has_fp16 = self.executor.has_fp16_weight_cache();
        if !has_fp16 {
            // FP16 not available (e.g. HGEMM_PREFILL=0 or Jetson multi-service) —
            // fall through to DP4A GEMV decode path (existing behavior).
            self.executor.clear_prefill_graphs();
        } else {
            // Enable HGEMM batched decode — routes M>1 GEMV through cuBLAS tensor cores.
            self.executor.hgemm_batched_decode_active = true;
            self.executor.clear_prefill_graphs();
        }

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

        // PMAT-061: Disable HGEMM batched decode flag before cleanup.
        self.executor.hgemm_batched_decode_active = false;

        // PMAT-058: Clear workspace to force M=1-sized reallocation.
        // Batched workspace has buffer_capacity=M (4×), which spreads the working set
        // across 4× more L2 cache lines → 12% decode regression (139→123 tok/s).
        // Clearing forces init_workspace to reallocate tight M=1 buffers.
        self.executor.clear_workspace();
        let _ = self
            .executor
            .init_workspace(hidden_dim, intermediate_dim);

        // PMAT-044 FIX: Reset batched KV state so subsequent M=1 prefill doesn't
        // take the batched attention path (batched_ffn.rs line 37). Without this,
        // M=1 prefill writes K/V to batched caches while M=1 decode reads from the
        // single KV cache (empty) → EOS on first token → 0 output tokens.
        self.executor.batched_kv_stride = 0;

        // PMAT-058: Free batched KV caches to reclaim ~460MB VRAM.
        self.executor.free_batched_kv_caches();

        // PMAT-061: FP16 weight cache is now kept during decode (not cleared after prefill).
        // No need to rebuild — it's already in VRAM. Just ensure it's present for next c=1.
        if !self.executor.has_fp16_weight_cache() {
            // FP16 was never populated (HGEMM_PREFILL=0) — rebuild for c=1 HGEMM prefill.
            let _ = self.executor.warmup_hgemm_cache(
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size as u32,
            );
        }

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

        let prefill_start = std::time::Instant::now();

        // PMAT-051: Multi-prompt batched prefill — read weights once for all prompts.
        // Concatenate all prompts' tokens, run GEMM once per layer (M_total tokens),
        // only attention is per-prompt (prompt-local causal mask).
        let use_multi_prompt = m > 1
            && std::env::var("MULTI_PROMPT_PREFILL").as_deref() != Ok("0");

        if use_multi_prompt {
            let hidden_dim = self.model.config.hidden_dim;
            let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
            let num_layers = self.model.config.num_layers;
            let eps = self.model.config.eps;

            // Calculate per-prompt lengths (prefill_count = prompt.len() - 1)
            let prompt_lengths: Vec<usize> = prompts.iter()
                .map(|p| p.len().saturating_sub(1))
                .collect();
            let m_total: usize = prompt_lengths.iter().sum();

            // Build offsets into the packed buffer
            let prompt_offsets: Vec<usize> = prompt_lengths.iter()
                .scan(0, |acc, &len| { let off = *acc; *acc += len; Some(off) })
                .collect();

            // Embed all prompts, concatenate into packed [M_total × hidden_dim]
            let mut all_embeddings = Vec::with_capacity(m_total * hidden_dim);
            for (i, prompt) in prompts.iter().enumerate() {
                let embeddings = self.model.embed(&prompt[..prompt_lengths[i]]);
                all_embeddings.extend_from_slice(&embeddings);
            }

            // Build concatenated positions: [0..S0-1, 0..S1-1, ...]
            let mut all_positions = Vec::with_capacity(m_total);
            for &len in &prompt_lengths {
                all_positions.extend(0..len as u32);
            }

            // Ensure workspace sized for M_total
            self.executor
                .init_prefill_workspace(m_total, hidden_dim, intermediate_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_prefill_workspace".to_string(),
                    reason: format!("PMAT-051: workspace init failed for M_total={m_total}: {e}"),
                })?;

            eprintln!(
                "[PMAT-051] Multi-prompt prefill: {} prompts, M_total={}, reading weights once...",
                m, m_total
            );

            // Run multi-prompt prefill — weights read once, per-prompt attention
            self.executor
                .prefill_multi_prompt(
                    &all_embeddings,
                    &all_positions,
                    &prompt_offsets,
                    &prompt_lengths,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "prefill_multi_prompt".to_string(),
                    reason: format!("PMAT-051: multi-prompt prefill failed: {e}"),
                })?;

            // Build return values
            for (i, prompt) in prompts.iter().enumerate() {
                positions.push(prompt_lengths[i]);
                last_tokens.push(prompt[prompt.len() - 1]);
            }
        } else {
            // Sequential fallback (M=1 or MULTI_PROMPT_PREFILL=0)
            let saved_stride = self.executor.batched_kv_stride;
            eprintln!("[PMAT-044] Batched streaming: {m} slots, prefilling sequentially...");

            for (slot_idx, prompt) in prompts.iter().enumerate() {
                self.executor.reset_kv_cache_gpu();
                let seq_len = prompt.len().saturating_sub(1);

                self.executor.batched_kv_stride = 0;
                self.run_prefill(prompt, &mut caches[slot_idx], seq_len, false)?;

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
        }

        let prefill_elapsed = prefill_start.elapsed();
        eprintln!(
            "[PMAT-051] Prefill done in {:.1}ms ({:.0} tok/s), starting batched decode...",
            prefill_elapsed.as_secs_f64() * 1000.0,
            prompts.iter().map(|p| p.len() as f64).sum::<f64>() / prefill_elapsed.as_secs_f64(),
        );
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
