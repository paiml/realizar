impl CudaExecutor {
    /// PAR-054: Graph-captured forward pass for decode (M=1)
    ///
    /// Uses CUDA graph capture to reduce kernel launch overhead from ~280 launches
    /// to 1 graph launch (~10µs vs ~5.6ms overhead).
    ///
    /// First decode token: captures the kernel sequence into a graph
    /// Subsequent tokens: replays the captured graph with updated position
    ///
    /// # Performance
    ///
    /// - Without graphs: ~280 kernel launches × ~20µs = ~5.6ms overhead/token
    /// - With graphs: 1 graph launch × ~10µs = ~0.01ms overhead/token
    /// - Expected speedup: ~500x reduction in launch overhead
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu_to_logits_graphed(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // PAR-054: Environment variable to disable CUDA graphs for debugging
        // Set CUDA_GRAPH_DISABLE=1 to use non-graphed path
        static GRAPH_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let graph_disabled = *GRAPH_DISABLED.get_or_init(|| {
            std::env::var("CUDA_GRAPH_DISABLE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        if graph_disabled {
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-064-DEBUG: Allow disabling graph mode for debugging
        let skip_graph = std::env::var("SKIP_CUDA_GRAPH")
            .map(|v| v == "1")
            .unwrap_or(false);

        // PAR-054: Check if we should capture or replay
        if !skip_graph && self.decode_graph.is_some() && self.decode_token_count > 0 {
            // Replay path: update position and launch graph
            if self.decode_token_count <= 3 && verbose() {
                eprintln!(
                    "[PAR-054] Graph replay #{} (pos={})",
                    self.decode_token_count, position
                );
            }
            return self.forward_graphed_replay(input, logits, position);
        }

        // PAR-118: Skip graph capture if a previous attempt failed.
        // Flash Decoding's stream.synchronize() is forbidden during capture (error 901).
        // Retrying on every token corrupts CUDA state → CUDA_ERROR_INVALID_VALUE crash.
        if self.graph_capture_failed {
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // First token or no graph yet: try to capture
        // We need workspace path for stable addresses
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        if !use_workspace {
            // Fall back to non-graphed path if workspace not available
            eprintln!("[PAR-054] Workspace not ready, using non-graphed path (has_workspace={}, has_indexed={}, layers={})",
                self.has_workspace(), self.has_indexed_weights(), self.indexed_layer_weights.len());
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Verify lm_head_ptr is set (needed for graph-captured LM head projection)
        if self.lm_head_ptr == 0 {
            eprintln!("[PAR-054] lm_head_ptr not set, using non-graphed path");
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Initialize position buffer if needed
        if self.position_buf.is_none() {
            let pos_buf = GpuBuffer::from_host(&self.context, &[position])?;
            self.position_buf = Some(pos_buf);
        } else {
            // Update position
            self.position_buf
                .as_mut()
                .expect("position_buf must be initialized")
                .copy_from_host(&[position])?;
        }

        // PAR-061: Initialize seq_len buffer for indirect attention kernel
        // seq_len = position + 1 (new sequence length after adding this token)
        let seq_len = position + 1;
        if self.seq_len_buf.is_none() {
            let len_buf = GpuBuffer::from_host(&self.context, &[seq_len])?;
            self.seq_len_buf = Some(len_buf);
        } else {
            self.seq_len_buf
                .as_mut()
                .expect("seq_len_buf must be initialized")
                .copy_from_host(&[seq_len])?;
        }

        // PAR-054: Initialize stable input buffer if needed
        let hidden_size = hidden_dim as usize;
        if self.graph_input_buf.is_none()
            || self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .len()
                != hidden_size
        {
            let input_buf = GpuBuffer::from_host(&self.context, input)?;
            self.graph_input_buf = Some(input_buf);
        } else {
            self.graph_input_buf
                .as_mut()
                .expect("graph_input_buf must be initialized")
                .copy_from_host(input)?;
        }

        // PAR-054: Pre-allocate normed_hidden_buf before capture
        if self.workspace.normed_hidden_buf.is_none() {
            let normed_buf = GpuBuffer::new(&self.context, hidden_size)?;
            self.workspace.normed_hidden_buf = Some(normed_buf);
        }

        // PAR-054: Pre-allocate logits_buf before capture
        if self.workspace.logits_buf.is_none() {
            let logits_buf = GpuBuffer::new(&self.context, vocab_size as usize)?;
            self.workspace.logits_buf = Some(logits_buf);
        }

        // PAR-054-FIX: Pre-load all kernel modules BEFORE graph capture
        // Root cause: CudaModule::from_ptx allocates memory which breaks capture
        self.preload_modules_for_capture(num_layers, hidden_dim, intermediate_dim, vocab_size)?;

        // PAR-064-DEBUG: Skip graph capture if SKIP_CUDA_GRAPH=1
        if skip_graph {
            eprintln!("[PAR-064-DEBUG] SKIP_CUDA_GRAPH=1, using non-graphed path");
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Try CUDA graph capture, fall back to non-graphed path if fails
        // Some operations (memory allocation, synchronization) aren't graph-compatible
        // PAR-118: Set is_capturing flag so incremental_attention_into_inner skips
        // Flash Decoding (which calls stream.synchronize(), forbidden during capture)
        self.is_capturing = true;
        let capture_result = self.try_graph_capture(
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
        );
        self.is_capturing = false;

        match capture_result {
            Ok(()) => {
                // CORRECTNESS-010: Graph capture defines the work but doesn't execute it.
                // Must launch the graph once to produce actual output for first token.
                if let Some(ref graph_exec) = self.decode_graph {
                    self.stream.launch_graph(graph_exec)?;
                }
                // Graph captured and launched, sync and download logits
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
            Err(e) => {
                // Graph capture failed, fall back to non-graphed path.
                // PAR-118: Set flag to prevent futile retries that corrupt CUDA state.
                self.graph_capture_failed = true;
                eprintln!(
                    "[PAR-054] Graph capture failed: {:?}, using non-graphed path (no retry)",
                    e
                );
                // PAR-070: Pass position for correct RoPE and KV cache behavior
                self.forward_all_layers_gpu_to_logits(
                    input,
                    logits,
                    position,
                    num_layers,
                    hidden_dim,
                    intermediate_dim,
                    vocab_size,
                    epsilon,
                )
            },
        }
    }
}
