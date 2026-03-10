/// PMAT-072: Decode state for step-wise batched generation.
///
/// Extracted from `generate_batched_streaming` to allow lock release between
/// decode steps. The scheduler acquires the model lock for ~19ms per step
/// instead of ~660ms for the entire batch.
pub struct BatchedDecodeState {
    /// Number of concurrent slots (batch size)
    pub m: usize,
    /// Maximum KV cache slots pre-allocated (for PMAT-073 mid-batch joins)
    pub max_kv_slots: usize,
    /// Maximum tokens to generate across all slots
    pub max_tokens_max: usize,
    /// Current decode step index (incremented by `distribute_tokens`)
    pub gen_idx: usize,
    /// Model hidden dimension (for embedding buffer sizing)
    pub hidden_dim: usize,
    /// Model intermediate dimension (for workspace cleanup)
    pub intermediate_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Layer norm epsilon
    pub eps: f32,
    /// Original prompts per slot (for generated-length tracking)
    pub prompts: Vec<Vec<u32>>,
    /// Generation config per slot (stop tokens, max tokens)
    pub configs: Vec<QuantizedGenerateConfig>,
    /// Per-slot streaming callbacks (SSE token delivery)
    pub on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>>,
    /// Accumulated sequences per slot (prompt + generated tokens)
    pub sequences: Vec<Vec<u32>>,
    /// Current position per slot (for RoPE)
    pub positions: Vec<usize>,
    /// Last generated token per slot (input for next decode step)
    pub last_tokens: Vec<u32>,
    /// Whether each slot has finished generating
    pub done: Vec<bool>,
    /// Pre-allocated embedding buffer `[m × hidden_dim]`
    pub embed_buf: Vec<f32>,
}

impl BatchedDecodeState {
    /// Distribute generated tokens to SSE callbacks. No model lock needed.
    ///
    /// Returns true if all slots are done.
    pub fn distribute_tokens(&mut self, token_ids: &[u32]) -> bool {
        for slot_idx in 0..self.m {
            if self.done[slot_idx] {
                continue;
            }

            let next_token = token_ids[slot_idx];

            if self.configs[slot_idx].stop_tokens.contains(&next_token) {
                self.done[slot_idx] = true;
                continue;
            }

            self.sequences[slot_idx].push(next_token);

            if !self.on_tokens[slot_idx](next_token) {
                self.done[slot_idx] = true;
                continue;
            }

            self.last_tokens[slot_idx] = next_token;
            self.positions[slot_idx] += 1;

            let generated = self.sequences[slot_idx].len() - self.prompts[slot_idx].len();
            if generated >= self.configs[slot_idx].max_tokens {
                self.done[slot_idx] = true;
            }
        }
        self.gen_idx += 1;
        self.done.iter().all(|&d| d)
    }

    /// Check if all slots have finished generating.
    pub fn all_done(&self) -> bool {
        self.done.iter().all(|&d| d)
    }
}

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
    /// Backward-compatible wrapper around the PMAT-072 step-wise API.
    /// For lock-releasing batched decode, use `batched_setup_and_prefill`,
    /// `batched_decode_step`, and `batched_cleanup` directly.
    pub fn generate_batched_streaming(
        &mut self,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>>,
    ) -> Result<Vec<Vec<u32>>> {
        let m = prompts.len();
        eprintln!("[PMAT-044] generate_batched_streaming ENTRY m={m}");
        if m == 0 {
            return Ok(Vec::new());
        }

        let mut state = self.batched_setup_and_prefill(prompts, configs, on_tokens, m)?;

        while !state.all_done() && state.gen_idx < state.max_tokens_max {
            let token_ids = self.batched_decode_step(&mut state)?;
            state.distribute_tokens(&token_ids);
        }

        self.batched_cleanup(&state);

        Ok(state.sequences)
    }

    /// PMAT-072: Setup and prefill phase for step-wise batched generation.
    ///
    /// Performs validation, graph clearing, workspace init, KV cache allocation,
    /// and prefill. Returns a `BatchedDecodeState` for use with `batched_decode_step`.
    ///
    /// Caller must hold model write lock for the duration of this call.
    pub fn batched_setup_and_prefill(
        &mut self,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>>,
        max_kv_slots: usize,
    ) -> Result<BatchedDecodeState> {
        let m = prompts.len();
        eprintln!("[PMAT-072] batched_setup_and_prefill ENTRY m={m}");
        self.validate_batch_args(m, configs.len(), on_tokens.len())?;

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
        // PMAT-073: Pre-allocate batched KV for max_kv_slots to enable mid-batch joins.
        // Extra slots cost ~224 MB/slot but avoid reallocation when new requests join.
        let kv_alloc = max_kv_slots.max(m);
        self.executor
            .init_batched_kv_cache_gpu(num_layers, kv_alloc)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_kv_cache_gpu".to_string(),
                reason: format!("Failed to init batched KV cache for M={kv_alloc}: {e}"),
            })?;
        self.executor.reset_batched_kv_cache_gpu();

        // Prefill each slot with HGEMM (FP16 present) + D2D scatter to batched KV.
        let (sequences, positions, last_tokens) =
            self.prefill_and_scatter(prompts, &mut caches)?;

        // PMAT-062: DP4A GEMV for batched decode (default), not cuBLAS HGEMM.
        // Five-Whys: HGEMM batched decode is 13% slower (22.1ms vs 19.3ms ITL at M=4).
        // Why? FP16 reads 3.5x more data (2 B/elem vs Q4K 0.5625 B/elem).
        // Why? Tensor cores help but don't compensate for 3.5x BW penalty.
        // Why? RTX 4060L has 272 GB/s BW — FP16 M=4 GEMM is BW-limited.
        // Also: HGEMM flag blocks fused gate+up DP4A (batched_ffn.rs:107).
        // Override: HGEMM_BATCHED_DECODE=1 to force cuBLAS tensor core path.
        let use_hgemm_decode = std::env::var("HGEMM_BATCHED_DECODE").as_deref() == Ok("1")
            && self.executor.has_fp16_weight_cache();
        if use_hgemm_decode {
            self.executor.hgemm_batched_decode_active = true;
        }
        self.executor.clear_prefill_graphs();

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

        let embed_buf = vec![0.0f32; m * hidden_dim];
        let done = vec![false; m];

        Ok(BatchedDecodeState {
            m,
            max_kv_slots: kv_alloc,
            max_tokens_max,
            gen_idx: 0,
            hidden_dim,
            intermediate_dim,
            num_layers,
            vocab_size,
            eps,
            prompts: prompts.to_vec(),
            configs: configs.to_vec(),
            on_tokens,
            sequences,
            positions,
            last_tokens,
            done,
            embed_buf,
        })
    }

    /// PMAT-072: Single decode step for step-wise batched generation.
    ///
    /// Embeds current tokens, runs one forward pass, returns raw token IDs.
    /// Call `state.distribute_tokens(&token_ids)` after releasing the model lock
    /// to deliver tokens to SSE callbacks without holding the lock.
    ///
    /// Caller must hold model write lock for the duration of this call.
    pub fn batched_decode_step(
        &mut self,
        state: &mut BatchedDecodeState,
    ) -> Result<Vec<u32>> {
        // Embed all tokens into state's pre-allocated buffer
        for slot_idx in 0..state.m {
            let offset = slot_idx * state.hidden_dim;
            if state.done[slot_idx] {
                state.embed_buf[offset..offset + state.hidden_dim].fill(0.0);
            } else {
                self.model.embed_into(
                    state.last_tokens[slot_idx],
                    &mut state.embed_buf[offset..offset + state.hidden_dim],
                );
            }
        }

        let pos_u32: Vec<u32> = state.positions.iter().map(|&p| p as u32).collect();

        if state.gen_idx < 3 {
            eprintln!(
                "[PMAT-072] decode step {}: positions={:?}, last_tokens={:?}",
                state.gen_idx, &pos_u32[..state.m], state.last_tokens
            );
        }

        // PMAT-056: Multi-stream root cause fixed (scatter moved to self.stream),
        // but graph replay still 25% slower than eager due to capture overhead.
        // Keep eager by default until graph replay is optimized.
        // Enable with BATCHED_GRAPH=1 for testing.
        let use_graph = std::env::var("BATCHED_GRAPH").as_deref() == Ok("1");
        let token_ids = if use_graph {
            self.executor
                .forward_batched_to_token_ids_graphed(
                    &state.embed_buf,
                    &pos_u32,
                    state.num_layers,
                    state.hidden_dim as u32,
                    state.intermediate_dim as u32,
                    state.vocab_size as u32,
                    state.eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_batched_to_token_ids_graphed".to_string(),
                    reason: format!("Batched forward failed: {e}"),
                })?
        } else {
            self.executor
                .forward_batched_to_token_ids(
                    &state.embed_buf,
                    &pos_u32,
                    state.num_layers,
                    state.hidden_dim as u32,
                    state.intermediate_dim as u32,
                    state.vocab_size as u32,
                    state.eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_batched_to_token_ids".to_string(),
                    reason: format!("Batched forward failed: {e}"),
                })?
        };

        if state.gen_idx < 3 {
            eprintln!(
                "[PMAT-072] decode step {}: token_ids={token_ids:?}, done={:?}",
                state.gen_idx, state.done
            );
        }

        Ok(token_ids)
    }

    /// PMAT-072: Cleanup after step-wise batched generation.
    ///
    /// Resets workspace to M=1 sizing, frees batched KV caches, restores
    /// FP16 weight cache for subsequent single-request inference.
    ///
    /// Caller must hold model write lock for the duration of this call.
    pub fn batched_cleanup(&mut self, state: &BatchedDecodeState) {
        // PMAT-061: Disable HGEMM batched decode flag before cleanup.
        self.executor.hgemm_batched_decode_active = false;

        // PMAT-058: Clear workspace to force M=1-sized reallocation.
        // Batched workspace has buffer_capacity=M (4×), which spreads the working set
        // across 4× more L2 cache lines → 12% decode regression (139→123 tok/s).
        // Clearing forces init_workspace to reallocate tight M=1 buffers.
        self.executor.clear_workspace();
        let _ = self
            .executor
            .init_workspace(state.hidden_dim, state.intermediate_dim);

        // PMAT-044 FIX: Reset batched KV state so subsequent M=1 prefill doesn't
        // take the batched attention path (batched_ffn.rs line 37). Without this,
        // M=1 prefill writes K/V to batched caches while M=1 decode reads from the
        // single KV cache (empty) → EOS on first token → 0 output tokens.
        self.executor.batched_kv_stride = 0;

        // PMAT-058: Free batched KV caches to reclaim ~460MB VRAM.
        self.executor.free_batched_kv_caches();

        // PMAT-061: Weight cache is now kept during decode (not cleared after prefill).
        // PMAT-067: Prefer FP8 cache on sm_89+ — skip FP16 cache rebuild.
        if self.executor.gpu_profile.fp8_prefill {
            // FP8 cache handles its own lazy population — no rebuild needed.
        } else if !self.executor.has_fp16_weight_cache() {
            // FP16 was never populated (HGEMM_PREFILL=0) — rebuild for c=1 HGEMM prefill.
            let _ = self.executor.warmup_hgemm_cache(
                state.num_layers,
                state.hidden_dim as u32,
                state.intermediate_dim as u32,
                state.vocab_size as u32,
            );
        }
    }

    /// PMAT-073: Add a new request to a running batch mid-generation.
    ///
    /// Prefills the new slot using M=1 forward, scatters KV to the batched cache
    /// at the next available slot index, and extends the decode state.
    ///
    /// Caller must hold model write lock for the duration of this call.
    /// The prefill takes ~40-130ms depending on prompt length (one-time cost
    /// that briefly increases ITL for existing slots).
    pub fn add_slot_to_batch(
        &mut self,
        state: &mut BatchedDecodeState,
        prompt: Vec<u32>,
        config: QuantizedGenerateConfig,
        on_token: Box<dyn FnMut(u32) -> bool + Send>,
    ) -> Result<()> {
        let new_slot = state.m;
        if new_slot >= state.max_kv_slots {
            return Err(RealizarError::UnsupportedOperation {
                operation: "add_slot_to_batch".to_string(),
                reason: format!(
                    "PMAT-073: batch full (m={}, max_kv_slots={})",
                    state.m, state.max_kv_slots
                ),
            });
        }

        let seq_len = prompt.len().saturating_sub(1);
        let prefill_start = std::time::Instant::now();

        // Clear decode graphs — M=1 prefill uses different kernel config
        self.executor.clear_decode_graph();
        self.executor.clear_batched_decode_graphs();

        // Init workspace for prefill (high-water mark, usually no realloc)
        self.executor
            .init_prefill_workspace(seq_len, state.hidden_dim, state.intermediate_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "add_slot_prefill_workspace".to_string(),
                reason: format!("PMAT-073: workspace init for slot {new_slot}: {e}"),
            })?;

        // Prefill using M=1 path (writes to single-slot KV cache)
        let kv_dim = self.model.config.num_kv_heads
            * (self.model.config.hidden_dim / self.model.config.num_heads);
        let max_seq_len = seq_len + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::new(state.num_layers, kv_dim, max_seq_len);

        let saved_stride = self.executor.batched_kv_stride;
        self.executor.batched_kv_stride = 0;
        self.executor.reset_kv_cache_gpu();
        self.run_prefill(&prompt, &mut cache, seq_len, false)?;
        self.executor.batched_kv_stride = saved_stride;

        // Scatter KV from single-slot cache to batched cache at new_slot
        self.executor
            .scatter_single_kv_to_batched(new_slot, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "scatter_single_kv_to_batched".to_string(),
                reason: format!("PMAT-073: scatter for slot {new_slot}: {e}"),
            })?;

        // Switch back to batched decode mode with M+1
        let new_m = new_slot + 1;
        self.executor.clear_prefill_graphs();
        self.executor
            .init_batched_workspace(state.hidden_dim, state.intermediate_dim, new_m)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_batched_workspace".to_string(),
                reason: format!("PMAT-073: workspace resize for M={new_m}: {e}"),
            })?;
        self.executor.clear_decode_graph();

        // Update state
        let last_token = prompt[prompt.len() - 1];
        state.sequences.push(prompt.clone());
        state.prompts.push(prompt);
        state.configs.push(config);
        state.on_tokens.push(on_token);
        state.positions.push(seq_len);
        state.last_tokens.push(last_token);
        state.done.push(false);
        state.m = new_m;
        state.embed_buf.resize(new_m * state.hidden_dim, 0.0);
        state.max_tokens_max = state
            .max_tokens_max
            .max(state.configs.last().unwrap().max_tokens);

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[PMAT-073] Mid-batch join: slot {} added (now M={}), prefill {:.1}ms ({} tokens)",
            new_slot, new_m, prefill_ms, seq_len,
        );

        Ok(())
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
}
