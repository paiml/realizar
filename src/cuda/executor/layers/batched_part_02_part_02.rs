impl CudaExecutor {
    /// PAR-111: Batched forward pass for M sequences returning M token IDs
    ///
    /// Processes M sequences in parallel through all transformer layers using
    /// batched GEMV kernels that read/dequantize weights ONCE for all M inputs.
    ///
    /// # Performance
    ///
    /// - M=1: Baseline (~360 tok/s)
    /// - M=4: 16x GEMV speedup → 857+ tok/s aggregate throughput
    ///
    /// # Arguments
    ///
    /// * `inputs` - M embeddings packed [M × hidden_dim]
    /// * `positions` - M sequence positions for RoPE
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `vocab_size` - Vocabulary size
    /// * `epsilon` - RMSNorm epsilon
    ///
    /// # Returns
    ///
    /// M token IDs (greedy argmax)
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batched_to_token_ids(
        &mut self,
        inputs: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Vec<u32>, GpuError> {
        let m = positions.len();
        // PAR-129: Extended to M=32 via 4-warp kernel
        if m == 0 || m > 32 {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-111: batch size must be 1-32, got {}",
                m
            )));
        }
        let expected_input_len = m * hidden_dim as usize;
        if inputs.len() != expected_input_len {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-111: inputs.len() {} != M*hidden_dim = {}",
                inputs.len(),
                expected_input_len
            )));
        }

        // Verify batched workspace initialized
        if !self.workspace.initialized || self.workspace.batch_size != m {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-111: Batched workspace not initialized for M={}",
                m
            )));
        }

        // 1. Upload M embeddings to GPU
        let input_buf = GpuBuffer::from_host(&self.context, inputs)?;

        // Get workspace buffer pointers to avoid borrow conflicts
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 missing".to_string())
            })?
            .len();

        // 2. Process all layers with batched GEMV
        for layer_idx in 0..num_layers {
            // Get indexed layer weights (must be pre-built via build_indexed_weights)
            if layer_idx >= self.indexed_layer_weights.len() {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-111: Layer {} weights not indexed (have {})",
                    layer_idx,
                    self.indexed_layer_weights.len()
                )));
            }
            let layer_weights = self.get_indexed_layer(layer_idx).clone();

            // Use workspace output from previous layer (or input_buf for first layer)
            // SAFETY: hidden_buf2 is valid for the lifetime of this function
            let layer_input_buf = if layer_idx == 0 {
                None // Use input_buf directly
            } else {
                // SAFETY: Pointer valid from allocation, length verified, used within scope
                Some(unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) })
            };

            let layer_input = match &layer_input_buf {
                Some(buf) => buf,
                None => &input_buf,
            };

            self.transformer_layer_batched(
                layer_input,
                layer_idx,
                &layer_weights,
                m as u32,
                positions,
                hidden_dim,
                intermediate_dim,
                epsilon,
            )?;

            // Prevent drop of borrowed buffer
            if let Some(buf) = layer_input_buf {
                std::mem::forget(buf);
            }
        }

        // 3. Output norm (PAR-115: Batched - single launch for M sequences)
        let output_norm_buf = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-111: output_norm not cached".to_string())
        })?;
        let output_norm_ptr = output_norm_buf.as_ptr();
        let output_norm_len = hidden_dim as usize;

        // Get buffer pointers to avoid borrow conflicts
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = m * hidden_dim as usize;
        let normed_hidden_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: normed_hidden_buf missing".to_string())
            })?
            .as_ptr();
        let normed_hidden_len = m * hidden_dim as usize;

        // PAR-115: Use batched RMSNorm (M sequences in single kernel launch)
        // SAFETY: Buffers are valid for the lifetime of this function
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

        // 4. LM head projection (BATCHED GEMV)
        if self.lm_head_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-111: LM head not indexed".to_string(),
            ));
        }
        let lm_head_ptr = self.lm_head_ptr;
        let lm_head_qtype = self.lm_head_qtype;

        // Allocate logits buffer (M × vocab_size)
        let logits_buf = GpuBuffer::new(&self.context, m * vocab_size as usize)?;

        // Get normed_hidden buffer pointer to avoid borrow conflict
        let normed_hidden_buf_len = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: normed_hidden_buf missing".to_string())
            })?
            .len();
        // SAFETY: normed_hidden_buf is valid for the lifetime of this function
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_hidden_buf_wrapper =
            unsafe { GpuBuffer::<f32>::from_raw_parts(normed_hidden_ptr, normed_hidden_buf_len) };

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
                        logits_buf.as_ptr() + (v_offset * std::mem::size_of::<f32>()) as u64,
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

        // Prevent drop of borrowed buffer
        std::mem::forget(normed_hidden_buf_wrapper);

        // 5. Batched argmax (M sequential GPU argmax calls)
        self.stream.synchronize()?;

        let mut token_ids = Vec::with_capacity(m);
        for seq_idx in 0..m {
            let v_offset = seq_idx * vocab_size as usize;
            let logits_ptr = logits_buf.as_ptr() + (v_offset * std::mem::size_of::<f32>()) as u64;

            let token_id = self.gpu_argmax(logits_ptr, vocab_size)?;
            token_ids.push(token_id);
        }

        Ok(token_ids)
    }

    /// PAR-121: Graph-captured batched forward pass for M sequences
    ///
    /// Uses CUDA graph capture to reduce kernel launch overhead for batched decode.
    /// First call with batch size M: captures the kernel sequence into a graph.
    /// Subsequent calls with same M: replays captured graph with updated inputs.
    ///
    /// # Performance
    ///
    /// - Without graphs (M=2): 404.6 tok/s
    /// - With graphs (M=2): Target ~550+ tok/s (2x Ollama)
    /// - Key: Combines batched GEMV efficiency + CUDA graph launch reduction
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batched_to_token_ids_graphed(
        &mut self,
        inputs: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Vec<u32>, GpuError> {
        let m = positions.len();
        // PAR-129: Extended to M=32 via 4-warp kernel
        if m == 0 || m > 32 {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-121: batch size must be 1-32, got {}",
                m
            )));
        }
        let expected_input_len = m * hidden_dim as usize;
        if inputs.len() != expected_input_len {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-121: inputs.len() {} != M*hidden_dim = {}",
                inputs.len(),
                expected_input_len
            )));
        }

        // Verify batched workspace initialized
        if !self.workspace.initialized || self.workspace.batch_size != m {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-121: Batched workspace not initialized for M={}",
                m
            )));
        }

        // Check if we have a captured graph for this batch size
        if self.batched_decode_graphs.contains_key(&m) && self.batched_graph_batch_size == m {
            // Replay path: update inputs and replay graph
            return self.forward_batched_graphed_replay(inputs, positions, m, vocab_size);
        }

        // First call or batch size changed: need to capture graph
        // Initialize stable buffers for graph capture
        self.init_batched_graph_buffers(m, hidden_dim, vocab_size)?;

        // Pre-load all kernel modules before capture
        self.preload_modules_for_batched_capture(
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
        )?;

        // Copy inputs to stable buffer
        if let Some(ref mut input_buf) = self.batched_graph_input_buf {
            input_buf.copy_from_host(inputs)?;
        }

        // Copy positions to stable buffer
        if let Some(ref mut pos_buf) = self.batched_graph_positions_buf {
            pos_buf.copy_from_host(positions)?;
        }

        // Copy seq_lens (position + 1 for each) to stable buffer
        let seq_lens: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        if let Some(ref mut len_buf) = self.batched_graph_seq_lens_buf {
            len_buf.copy_from_host(&seq_lens)?;
        }

        // Try to capture graph
        let capture_result = self.try_batched_graph_capture(
            m,
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
        );

        match capture_result {
            Ok(()) => {
                // Graph captured successfully
                self.batched_graph_batch_size = m;
                eprintln!("[PAR-121] ✓ Batched CUDA graph captured for M={}", m);

                // Get token IDs from logits
                self.stream.synchronize()?;
                self.batched_argmax_from_logits(m, vocab_size)
            },
            Err(e) => {
                // Graph capture failed, fall back to non-graphed path
                eprintln!(
                    "[PAR-121] Graph capture failed for M={}: {:?}, using non-graphed path",
                    m, e
                );
                self.forward_batched_to_token_ids(
                    inputs,
                    positions,
                    num_layers,
                    hidden_dim,
                    intermediate_dim,
                    vocab_size,
                    epsilon,
                )
            },
        }
    }

    /// PAR-121: Initialize stable buffers for batched graph capture
    fn init_batched_graph_buffers(
        &mut self,
        m: usize,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        let input_size = m * hidden_dim as usize;

        // Allocate or reallocate input buffer
        if self.batched_graph_input_buf.is_none()
            || self
                .batched_graph_input_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                != input_size
        {
            self.batched_graph_input_buf = Some(GpuBuffer::new(&self.context, input_size)?);
        }

        // Allocate or reallocate positions buffer
        if self.batched_graph_positions_buf.is_none()
            || self
                .batched_graph_positions_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                != m
        {
            self.batched_graph_positions_buf = Some(GpuBuffer::new(&self.context, m)?);
        }

        // Allocate or reallocate seq_lens buffer
        if self.batched_graph_seq_lens_buf.is_none()
            || self
                .batched_graph_seq_lens_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                != m
        {
            self.batched_graph_seq_lens_buf = Some(GpuBuffer::new(&self.context, m)?);
        }

        // Ensure workspace logits buffer is allocated for graph capture
        let logits_size = m * vocab_size as usize;
        if self.workspace.logits_buf.is_none()
            || self
                .workspace
                .logits_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                != logits_size
        {
            self.workspace.logits_buf = Some(GpuBuffer::new(&self.context, logits_size)?);
        }

        Ok(())
    }
}
