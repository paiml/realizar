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
        // Fast path: replay existing graph or fall back to non-graphed
        if let Some(result) = self.graphed_fast_path(
            input, logits, position, num_layers, hidden_dim,
            intermediate_dim, vocab_size, epsilon,
        )? {
            return result;
        }

        // Prepare device buffers for graph capture
        self.prepare_graph_buffers(input, position, hidden_dim, vocab_size)?;

        // PAR-054-FIX: Pre-load all kernel modules BEFORE graph capture
        self.preload_modules_for_capture(num_layers, hidden_dim, intermediate_dim, vocab_size)?;

        // PAR-064-DEBUG: Skip graph capture if SKIP_CUDA_GRAPH=1
        let skip_graph = std::env::var("SKIP_CUDA_GRAPH")
            .map(|v| v == "1")
            .unwrap_or(false);
        if skip_graph {
            eprintln!("[PAR-064-DEBUG] SKIP_CUDA_GRAPH=1, using non-graphed path");
            return self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            );
        }

        // PAR-054: Try CUDA graph capture, fall back on failure
        self.is_capturing = true;
        let capture_result = self.try_graph_capture(
            num_layers, hidden_dim, intermediate_dim, vocab_size, epsilon,
        );
        self.is_capturing = false;

        match capture_result {
            Ok(()) => {
                // CORRECTNESS-010: Graph capture defines the work but doesn't execute it.
                if let Some(ref graph_exec) = self.decode_graph {
                    self.stream.launch_graph(graph_exec)?;
                }
                // GH-93: Increment token count so subsequent calls replay instead of recapture.
                self.decode_token_count += 1;
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
            Err(e) => {
                self.graph_capture_failed = true;
                eprintln!(
                    "[PAR-054] Graph capture failed: {:?}, using non-graphed path (no retry)", e
                );
                // PMAT-374: Sync stream + re-upload input (capture modified workspace state)
                let _ = self.stream.synchronize();
                self.prepare_graph_buffers(input, position, hidden_dim, vocab_size)?;
                self.forward_all_layers_gpu_to_logits(
                    input, logits, position, num_layers, hidden_dim,
                    intermediate_dim, vocab_size, epsilon,
                )
            },
        }
    }

    /// Check early-exit conditions: replay, disabled, failed, prerequisites.
    /// Returns Some(Ok(())) to replay/fallback, None to continue to capture.
    #[allow(clippy::too_many_arguments)]
    fn graphed_fast_path(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Option<Result<(), GpuError>>, GpuError> {
        // PMAT-374: CUDA graph capture disabled by default — poisons context on driver
        // 590.48.01 (sm_89). Opt-in with CUDA_GRAPH_ENABLE=1.
        static GRAPH_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let graph_disabled = !*GRAPH_ENABLED.get_or_init(|| {
            std::env::var("CUDA_GRAPH_ENABLE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        if graph_disabled {
            return Ok(Some(self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            )));
        }

        let skip_graph = std::env::var("SKIP_CUDA_GRAPH")
            .map(|v| v == "1")
            .unwrap_or(false);

        // PAR-054: Replay existing graph
        if !skip_graph && self.decode_graph.is_some() && self.decode_token_count > 0 {
            if self.decode_token_count <= 3 && verbose() {
                eprintln!(
                    "[PAR-054] Graph replay #{} (pos={})",
                    self.decode_token_count, position
                );
            }
            return Ok(Some(self.forward_graphed_replay(input, logits, position)));
        }

        // PAR-118: Skip if previous capture failed
        if self.graph_capture_failed {
            return Ok(Some(self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            )));
        }

        // Check prerequisites for graph capture
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        if !use_workspace {
            eprintln!("[PAR-054] Workspace not ready, using non-graphed path");
            return Ok(Some(self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            )));
        }

        if self.lm_head_ptr == 0 {
            eprintln!("[PAR-054] lm_head_ptr not set, using non-graphed path");
            return Ok(Some(self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            )));
        }

        Ok(None) // Proceed to capture
    }

    /// Initialize or update device buffers needed for graph capture/replay.
    fn prepare_graph_buffers(
        &mut self,
        input: &[f32],
        position: u32,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        // Position buffer
        if self.position_buf.is_none() {
            self.position_buf = Some(GpuBuffer::from_host(&self.context, &[position])?);
        } else {
            self.position_buf.as_mut().expect("just checked").copy_from_host(&[position])?;
        }

        // Seq_len buffer (position + 1)
        let seq_len = position + 1;
        if self.seq_len_buf.is_none() {
            self.seq_len_buf = Some(GpuBuffer::from_host(&self.context, &[seq_len])?);
        } else {
            self.seq_len_buf.as_mut().expect("just checked").copy_from_host(&[seq_len])?;
        }

        // Input buffer
        let hidden_size = hidden_dim as usize;
        let needs_realloc = self.graph_input_buf.is_none()
            || self.graph_input_buf.as_ref().expect("just checked").len() != hidden_size;
        if needs_realloc {
            self.graph_input_buf = Some(GpuBuffer::from_host(&self.context, input)?);
        } else {
            self.graph_input_buf.as_mut().expect("just checked").copy_from_host(input)?;
        }

        // Pre-allocate output buffers if needed
        if self.workspace.normed_hidden_buf.is_none() {
            self.workspace.normed_hidden_buf = Some(GpuBuffer::new(&self.context, hidden_size)?);
        }
        // PMAT-088: Check size, not just existence — batched decode may have resized
        // logits_buf to M*vocab_size, causing D2H copy length mismatch.
        let needs_logits = self.workspace.logits_buf.as_ref().map_or(true, |b| {
            b.len() != vocab_size as usize
        });
        if needs_logits {
            self.workspace.logits_buf = Some(GpuBuffer::new(&self.context, vocab_size as usize)?);
        }

        Ok(())
    }
}
