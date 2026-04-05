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

        // trueno#243: Skip stream capture (code 901 poisons context on driver 570.207).
        // Use manual graph construction via cuGraphAddKernelNode instead.
        eprintln!(
            "[trueno#243] Manual graph construction: pos={}, has_graph={}, capture_failed={}, token_count={}",
            position, self.decode_graph.is_some(), self.graph_capture_failed, self.decode_token_count
        );
        self.begin_graph_recording();
        self.is_capturing = true;
        let eager_result = self.forward_workspace_captured(
            num_layers, hidden_dim, intermediate_dim, vocab_size, epsilon,
        );
        self.is_capturing = false;

        if let Err(eager_err) = eager_result {
            self.graph_recording = false;
            self.graph_capture_failed = true;
            eprintln!("[trueno#243] Eager forward during recording failed: {:?}", eager_err);
            // Re-prepare buffers and fall back to non-recorded eager
            let _ = self.stream.synchronize();
            self.prepare_graph_buffers(input, position, hidden_dim, vocab_size)?;
            return self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            );
        }

        // Eager pass succeeded — build graph from recorded kernels
        match self.end_graph_recording() {
            Ok(n) if n > 0 => {
                // Manual graph built! First token computed by eager pass.
                self.decode_token_count += 1;
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }

                // realizr#198 A/B TEST: Replay graph at SAME position.
                // If graph == eager, logits should be identical.
                // The eager pass already wrote the correct KV + logits.
                // Graph replay re-runs all kernels with the same input/position.
                static AB_TEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                if position > 0 && *AB_TEST.get_or_init(|| std::env::var("GRAPH_AB_TEST").as_deref() == Ok("1")) {
                    // Save eager intermediate buffers BEFORE graph replay overwrites them
                    let eager_hb1 = if let Some(ref hb) = self.workspace.hidden_buf1 {
                        let mut v = vec![0.0f32; hidden_dim as usize];
                        hb.copy_to_host(&mut v)?;
                        Some(v)
                    } else { None };
                    let eager_hb2 = if let Some(ref hb) = self.workspace.hidden_buf2 {
                        let mut v = vec![0.0f32; hidden_dim as usize];
                        hb.copy_to_host(&mut v)?;
                        Some(v)
                    } else { None };
                    let eager_normed = if let Some(ref nb) = self.workspace.normed_hidden_buf {
                        let mut v = vec![0.0f32; hidden_dim as usize];
                        nb.copy_to_host(&mut v)?;
                        Some(v)
                    } else { None };

                    // Replay with same buffers (input/position already set)
                    if let Some(ref graph_exec) = self.decode_graph {
                        self.stream.launch_graph(graph_exec)?;
                        self.stream.synchronize()?;
                    }

                    // Compare hidden_buf1 (RMSNorm layer 0 output → overwritten by layer 0 output proj)
                    if let (Some(eager_h), Some(ref hb)) = (&eager_hb1, &self.workspace.hidden_buf1) {
                        let mut graph_h = vec![0.0f32; hidden_dim as usize];
                        hb.copy_to_host(&mut graph_h)?;
                        let h_diff: f32 = eager_h.iter().zip(graph_h.iter())
                            .map(|(e, g)| (e - g).abs()).fold(0.0f32, f32::max);
                        eprintln!(
                            "[realizr#198-AB] hidden_buf1 max_diff={:.8}, eager[0..3]={:?}, graph[0..3]={:?}",
                            h_diff, &eager_h[..3], &graph_h[..3]
                        );
                    }
                    // Compare hidden_buf2 (output after ALL 28 layers)
                    if let (Some(eager_h2), Some(ref hb2)) = (&eager_hb2, &self.workspace.hidden_buf2) {
                        let mut graph_h2 = vec![0.0f32; hidden_dim as usize];
                        hb2.copy_to_host(&mut graph_h2)?;
                        let h2_diff: f32 = eager_h2.iter().zip(graph_h2.iter())
                            .map(|(e, g)| (e - g).abs()).fold(0.0f32, f32::max);
                        eprintln!(
                            "[realizr#198-AB] hidden_buf2 max_diff={:.6}, eager[0..3]={:?}, graph[0..3]={:?}",
                            h2_diff, &eager_h2[..3], &graph_h2[..3]
                        );
                    }
                    // Compare normed_hidden_buf (output RMSNorm applied to final hidden)
                    if let (Some(eager_n), Some(ref nb)) = (&eager_normed, &self.workspace.normed_hidden_buf) {
                        let mut graph_n = vec![0.0f32; hidden_dim as usize];
                        nb.copy_to_host(&mut graph_n)?;
                        let n_diff: f32 = eager_n.iter().zip(graph_n.iter())
                            .map(|(e, g)| (e - g).abs()).fold(0.0f32, f32::max);
                        eprintln!(
                            "[realizr#198-AB] normed_hidden max_diff={:.6}, eager[0..3]={:?}, graph[0..3]={:?}",
                            n_diff, &eager_n[..3], &graph_n[..3]
                        );
                    }

                    // Compare logits
                    if let Some(ref lb) = self.workspace.logits_buf {
                        let mut graph_logits = vec![0.0f32; vocab_size as usize];
                        lb.copy_to_host(&mut graph_logits)?;
                        let max_diff: f32 = logits.iter().zip(graph_logits.iter())
                            .map(|(e, g)| (e - g).abs()).fold(0.0f32, f32::max);
                        let eager_argmax = logits.iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
                        let graph_argmax = graph_logits.iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
                        eprintln!(
                            "[realizr#198-AB] logits: eager_argmax={}, graph_argmax={}, max_diff={:.6}, match={}",
                            eager_argmax, graph_argmax, max_diff, eager_argmax == graph_argmax
                        );
                        eprintln!(
                            "[realizr#198-AB] eager[0..5]={:?}, graph[0..5]={:?}",
                            &logits[..5], &graph_logits[..5]
                        );
                    }
                }

                Ok(())
            },
            Ok(_) => {
                // No kernels recorded — recording not wired to all ops yet.
                // First token was computed by eager pass.
                eprintln!("[trueno#243] 0 kernels recorded, using eager path for subsequent tokens");
                self.graph_capture_failed = true;
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
            Err(graph_err) => {
                self.graph_capture_failed = true;
                eprintln!("[trueno#243] Manual graph build failed: {:?}", graph_err);
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
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
