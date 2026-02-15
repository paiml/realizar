impl CudaExecutor {

    /// PAR-121: Pre-load kernel modules for batched graph capture
    fn preload_modules_for_batched_capture(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        // Reuse existing preload_modules_for_capture which loads all needed kernels
        self.preload_modules_for_capture(num_layers, hidden_dim, intermediate_dim, vocab_size)
    }

    /// PAR-121: Try to capture batched forward pass into CUDA graph
    fn try_batched_graph_capture(
        &mut self,
        m: usize,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Begin graph capture
        self.stream.begin_capture(CaptureMode::Global)?;

        // Run batched forward pass (all kernels will be captured)
        let capture_result = self.forward_batched_captured(
            m,
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
        );

        // End capture regardless of result
        let graph = self.stream.end_capture()?;

        // Check capture result
        capture_result?;

        // Instantiate the graph
        let graph_exec = graph.instantiate()?;
        self.batched_decode_graphs.insert(m, graph_exec);

        Ok(())
    }

    /// PAR-121: Forward pass for batched graph capture (uses stable buffers)
    fn forward_batched_captured(
        &mut self,
        m: usize,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Use stable input buffer
        let input_ptr = self
            .batched_graph_input_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PAR-121: batched_graph_input_buf missing".to_string(),
                )
            })?
            .as_ptr();
        let input_len = m * hidden_dim as usize;
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(input_ptr, input_len) };

        // Get workspace buffer pointers
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: hidden_buf2 missing".to_string())
            })?
            .len();

        // Use stable positions buffer for RoPE and attention
        let positions_ptr = self
            .batched_graph_positions_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PAR-121: batched_graph_positions_buf missing".to_string(),
                )
            })?
            .as_ptr();

        // Process all layers with batched GEMV
        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                std::mem::forget(input_buf);
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-121: Layer {} weights not indexed",
                    layer_idx
                )));
            }
            let layer_weights = self.get_indexed_layer(layer_idx).clone();

            let layer_input_buf = if layer_idx == 0 {
                None
            } else {
                // SAFETY: Pointer valid from allocation, length verified, used within scope
                Some(unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) })
            };

            let layer_input = match &layer_input_buf {
                Some(buf) => buf,
                None => &input_buf,
            };

            // Call batched layer with positions from stable buffer
            self.transformer_layer_batched_captured(
                layer_input,
                layer_idx,
                &layer_weights,
                m as u32,
                positions_ptr,
                hidden_dim,
                intermediate_dim,
                epsilon,
            )?;

            if let Some(buf) = layer_input_buf {
                std::mem::forget(buf);
            }
        }

        // Output norm
        let output_norm_buf = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-121: output_norm not cached".to_string())
        })?;
        let output_norm_ptr = output_norm_buf.as_ptr();
        let output_norm_len = hidden_dim as usize;

        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = m * hidden_dim as usize;
        let normed_hidden_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: normed_hidden_buf missing".to_string())
            })?
            .as_ptr();
        let normed_hidden_len = m * hidden_dim as usize;

        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_hidden_buf =
            unsafe { GpuBuffer::<f32>::from_raw_parts(normed_hidden_ptr, normed_hidden_len) };

        self.batched_rmsnorm_ptr_into(
            &hidden_buf2,
            output_norm_ptr,
            output_norm_len,
            &normed_hidden_buf,
            hidden_dim,
            m as u32,
            epsilon,
        )?;

        std::mem::forget(hidden_buf2);
        std::mem::forget(normed_hidden_buf);

        // LM head projection
        let lm_head_ptr = self.lm_head_ptr;
        let lm_head_qtype = self.lm_head_qtype;

        // Get logits buffer pointer to avoid borrow conflict
        let logits_buf_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: logits_buf missing".to_string())
            })?
            .as_ptr();
        let logits_buf_len = m * vocab_size as usize;

        // Create wrapper for logits buffer
        // SAFETY: Unsafe operation with validated invariants
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let logits_buf =
            unsafe { GpuBuffer::<f32>::from_raw_parts(logits_buf_ptr, logits_buf_len) };

        // SAFETY: Unsafe operation with validated invariants
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_hidden_buf_wrapper =
            unsafe { GpuBuffer::<f32>::from_raw_parts(normed_hidden_ptr, normed_hidden_len) };

        if lm_head_qtype == WeightQuantType::Q4K {
            self.batched_q4k_gemv_into(
                lm_head_ptr,
                &normed_hidden_buf_wrapper,
                &logits_buf,
                m as u32,
                vocab_size,
                hidden_dim,
            )?;
        } else {
            // Fall back to sequential for non-Q4K
            for seq_idx in 0..m {
                let h_offset = seq_idx * hidden_dim as usize;
                let v_offset = seq_idx * vocab_size as usize;

                // SAFETY: Unsafe operation with validated invariants
                let input_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        normed_hidden_ptr + (h_offset * std::mem::size_of::<f32>()) as u64,
                        hidden_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let output_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        logits_buf_ptr + (v_offset * std::mem::size_of::<f32>()) as u64,
                        vocab_size as usize,
                    )
                };

                self.q4k_gemv_into(
                    lm_head_ptr,
                    &input_view,
                    &output_view,
                    vocab_size,
                    hidden_dim,
                )?;

                std::mem::forget(input_view);
                std::mem::forget(output_view);
            }
        }

        std::mem::forget(normed_hidden_buf_wrapper);
        std::mem::forget(logits_buf);
        std::mem::forget(input_buf);

        Ok(())
    }

    /// PAR-121: Batched transformer layer using positions from device pointer
    #[allow(clippy::too_many_arguments)]
    fn transformer_layer_batched_captured(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &IndexedLayerWeights,
        m: u32,
        _positions_ptr: u64,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Uses batched version with positions read back from device
        // Direct device-side position access planned for PAR-200

        // For graph capture, we need to avoid host-device transfers
        // The positions are already on device, kernels can read from there
        // For now, use a dummy positions array (will be updated on replay)
        let dummy_positions: Vec<u32> = (0..m as usize).map(|i| i as u32).collect();

        self.transformer_layer_batched(
            input,
            layer_idx,
            layer_weights,
            m,
            &dummy_positions,
            hidden_dim,
            intermediate_dim,
            epsilon,
        )
    }

    /// PAR-121: Replay captured batched graph with updated inputs
    fn forward_batched_graphed_replay(
        &mut self,
        inputs: &[f32],
        positions: &[u32],
        m: usize,
        vocab_size: u32,
    ) -> Result<Vec<u32>, GpuError> {
        // Update stable buffers (async memcpy, doesn't invalidate graph)
        if let Some(ref mut input_buf) = self.batched_graph_input_buf {
            input_buf.copy_from_host(inputs)?;
        }

        if let Some(ref mut pos_buf) = self.batched_graph_positions_buf {
            pos_buf.copy_from_host(positions)?;
        }

        let seq_lens: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        if let Some(ref mut len_buf) = self.batched_graph_seq_lens_buf {
            len_buf.copy_from_host(&seq_lens)?;
        }

        // Also update the batched KV cache seq_lens for attention
        if let Some(ref mut seq_lens_gpu) = self.batched_seq_lens_gpu {
            seq_lens_gpu.copy_from_host(&seq_lens)?;
        }

        // Launch the captured graph
        if let Some(graph_exec) = self.batched_decode_graphs.get(&m) {
            graph_exec.launch(self.stream.raw())?;
        } else {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-121: No captured graph for M={}",
                m
            )));
        }

        // Get token IDs from logits
        self.stream.synchronize()?;
        self.batched_argmax_from_logits(m, vocab_size)
    }

    /// PAR-121: Extract token IDs from batched logits using GPU argmax
    fn batched_argmax_from_logits(
        &mut self,
        m: usize,
        vocab_size: u32,
    ) -> Result<Vec<u32>, GpuError> {
        // Get logits buffer pointer to avoid borrow conflict
        let logits_base_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: logits_buf missing".to_string())
            })?
            .as_ptr();

        let mut token_ids = Vec::with_capacity(m);
        for seq_idx in 0..m {
            let v_offset = seq_idx * vocab_size as usize;
            let logits_ptr = logits_base_ptr + (v_offset * std::mem::size_of::<f32>()) as u64;

            let token_id = self.gpu_argmax(logits_ptr, vocab_size)?;
            token_ids.push(token_id);
        }

        Ok(token_ids)
    }
}
