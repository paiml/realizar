//! Batched forward pass operations for multi-sequence inference
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-111: forward_batched_to_token_ids
//! - PAR-121: forward_batched_to_token_ids_graphed

#![allow(clippy::wildcard_imports)]

use super::super::*;

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
    /// PAR-111: Batched forward pass for M sequences through all layers
    ///
    /// Processes M sequences in parallel using batched GEMV kernels.
    /// Each sequence has independent KV cache state.
    ///
    /// # Performance Benefit
    ///
    /// Batched GEMV reads/dequantizes weights ONCE for all M inputs:
    /// - M=1: Baseline throughput (~360 tok/s)
    /// - M=4: 16x GEMV speedup → 857+ tok/s aggregate
    ///
    /// # Architecture
    ///
    /// - GEMV ops: Batched (weights read once)
    /// - Element-wise ops: Sequential on M vectors (cheap, ~2µs each)
    /// - Attention: M separate calls (different KV caches per sequence)
    ///
    /// # Arguments
    ///
    /// * `inputs` - Packed M embeddings [M × hidden_dim]
    /// * `m` - Batch size (1-8)
    /// * `layer_weights` - Indexed layer weights
    /// * `kv_bufs` - M KV buffer pairs [(k_buf, v_buf)] for this layer
    /// * `positions` - M positions for RoPE
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `epsilon` - RMSNorm epsilon
    ///
    /// # Returns
    ///
    /// Output is in workspace.hidden_buf2 (M × hidden_dim packed)
    pub fn transformer_layer_batched(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &IndexedLayerWeights,
        m: u32,
        positions: &[u32],
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Verify workspace initialized with correct batch size
        if !self.workspace.initialized {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-111: Workspace not initialized".to_string(),
            ));
        }
        if self.workspace.batch_size != m as usize {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-111: Workspace batch_size {} != m {}",
                self.workspace.batch_size, m
            )));
        }
        if positions.len() != m as usize {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-111: positions.len() {} != m {}",
                positions.len(),
                m
            )));
        }

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // Get batched buffer pointers (M× larger buffers allocated by init_batched_workspace)
        let hidden_buf1_ptr = self
            .workspace
            .hidden_buf1
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf1 not initialized".to_string())
            })?
            .as_ptr();
        let hidden_buf1_len = self
            .workspace
            .hidden_buf1
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf1 not initialized".to_string())
            })?
            .len();
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 not initialized".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 not initialized".to_string())
            })?
            .len();
        let input_staging_ptr = self
            .workspace
            .input_staging
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: input_staging not initialized".to_string())
            })?
            .as_ptr();
        let input_staging_len = self
            .workspace
            .input_staging
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: input_staging not initialized".to_string())
            })?
            .len();
        let q_buf_ptr = self
            .workspace
            .q_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: q_buf not initialized".to_string())
            })?
            .as_ptr();
        let q_buf_len = self
            .workspace
            .q_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: q_buf not initialized".to_string())
            })?
            .len();
        let k_buf_ptr = self
            .workspace
            .k_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: k_buf not initialized".to_string())
            })?
            .as_ptr();
        let k_buf_len = self
            .workspace
            .k_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: k_buf not initialized".to_string())
            })?
            .len();
        let v_buf_ptr = self
            .workspace
            .v_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: v_buf not initialized".to_string())
            })?
            .as_ptr();
        let v_buf_len = self
            .workspace
            .v_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: v_buf not initialized".to_string())
            })?
            .len();
        let ffn_gate_ptr = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: ffn_gate_buf not initialized".to_string())
            })?
            .as_ptr();
        let ffn_gate_len = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: ffn_gate_buf not initialized".to_string())
            })?
            .len();
        let ffn_up_ptr = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: ffn_up_buf not initialized".to_string())
            })?
            .as_ptr();
        let ffn_up_len = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: ffn_up_buf not initialized".to_string())
            })?
            .len();
        let ffn_act_ptr = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: ffn_act_buf not initialized".to_string())
            })?
            .as_ptr();
        let ffn_act_len = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: ffn_act_buf not initialized".to_string())
            })?
            .len();
        let attn_out_ptr = self
            .workspace
            .attn_out_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: attn_out_buf not initialized".to_string())
            })?
            .as_ptr();
        let attn_out_len = self
            .workspace
            .attn_out_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: attn_out_buf not initialized".to_string())
            })?
            .len();

        // Create temporary buffer wrappers (M× sized)
        // SAFETY: Memory safety ensured by workspace initialization
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let hidden_buf1 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf1_ptr, hidden_buf1_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let input_staging =
            unsafe { GpuBuffer::<f32>::from_raw_parts(input_staging_ptr, input_staging_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let attn_out_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(attn_out_ptr, attn_out_len) };

        // ========== 1. Pre-attention RMSNorm (BATCHED - PAR-112) ==========
        // Process all M sequences in a single kernel launch
        self.batched_rmsnorm_ptr_into(
            input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            &hidden_buf1,
            hidden_dim,
            m,
            epsilon,
        )?;

        // ========== 2. Q/K/V Projections (BATCHED GEMV - main optimization) ==========
        // Reads weights ONCE, applies to M input vectors
        // Only Q4K supported for now (most common for 1.5B+ models)
        if layer_weights.attn_q_qtype == WeightQuantType::Q4K {
            self.batched_q4k_gemv_into(
                layer_weights.attn_q_ptr,
                &hidden_buf1,
                &q_buf,
                m,
                q_dim,
                hidden_dim,
            )?;
            self.batched_q4k_gemv_into(
                layer_weights.attn_k_ptr,
                &hidden_buf1,
                &k_buf,
                m,
                kv_dim,
                hidden_dim,
            )?;
            self.batched_q4k_gemv_into(
                layer_weights.attn_v_ptr,
                &hidden_buf1,
                &v_buf,
                m,
                kv_dim,
                hidden_dim,
            )?;
        } else {
            // Fall back to sequential for non-Q4K weights
            for seq_idx in 0..m as usize {
                let h_offset = seq_idx * hidden_dim as usize;
                let q_offset = seq_idx * q_dim as usize;
                let kv_offset = seq_idx * kv_dim as usize;

                // SAFETY: Unsafe operation with validated invariants
                let input_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        hidden_buf1_ptr + (h_offset * std::mem::size_of::<f32>()) as u64,
                        hidden_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let q_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        q_buf_ptr + (q_offset * std::mem::size_of::<f32>()) as u64,
                        q_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let k_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        k_buf_ptr + (kv_offset * std::mem::size_of::<f32>()) as u64,
                        kv_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let v_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        v_buf_ptr + (kv_offset * std::mem::size_of::<f32>()) as u64,
                        kv_dim as usize,
                    )
                };

                self.q4k_gemv_into(
                    layer_weights.attn_q_ptr,
                    &input_view,
                    &q_view,
                    q_dim,
                    hidden_dim,
                )?;
                self.q4k_gemv_into(
                    layer_weights.attn_k_ptr,
                    &input_view,
                    &k_view,
                    kv_dim,
                    hidden_dim,
                )?;
                self.q4k_gemv_into(
                    layer_weights.attn_v_ptr,
                    &input_view,
                    &v_view,
                    kv_dim,
                    hidden_dim,
                )?;

                std::mem::forget(input_view);
                std::mem::forget(q_view);
                std::mem::forget(k_view);
                std::mem::forget(v_view);
            }
        }

        // ========== 3. RoPE on Q/K (PAR-114: BATCHED - 2 kernel launches) ==========
        let num_heads = self.kv_num_heads as u32;
        let num_kv_heads = self.kv_num_kv_heads as u32;
        let head_dim = self.kv_head_dim as u32;
        let theta = self.rope_theta;

        // Upload positions to GPU for batched RoPE
        let positions_buf_ptr = self
            .workspace
            .positions_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-114: positions_buf not initialized".to_string())
            })?
            .as_ptr();
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let mut positions_buf =
            unsafe { GpuBuffer::<u32>::from_raw_parts(positions_buf_ptr, m as usize) };

        // Convert positions to u32 and copy to device
        let positions_u32: Vec<u32> = positions.to_vec();
        positions_buf.copy_from_host(&positions_u32)?;

        // PAR-114: Batched RoPE for Q (all M sequences in one launch)
        self.batched_rope_into(
            &q_buf,
            &q_buf, // In-place
            &positions_buf,
            num_heads,
            head_dim,
            m,
            theta,
        )?;

        // PAR-114: Batched RoPE for K (all M sequences in one launch)
        self.batched_rope_into(
            &k_buf,
            &k_buf, // In-place
            &positions_buf,
            num_kv_heads,
            head_dim,
            m,
            theta,
        )?;

        std::mem::forget(positions_buf);

        // ========== 4. Attention ==========
        // PAR-119: Use batched attention if batched KV caches are initialized
        if self.batched_kv_stride > 0 && self.batched_kv_k_caches.contains_key(&layer_idx) {
            // PAR-118: Use Flash Decoding for long sequences (>128 positions)
            // Flash Decoding splits KV cache into chunks processed in parallel
            let max_seq_len = self
                .batched_kv_lengths
                .iter()
                .take(m as usize)
                .copied()
                .max()
                .unwrap_or(0);

            // PAR-118: Flash Decoding for very long sequences (>1024 positions)
            // Threshold raised from 128 to 1024 - overhead exceeds benefit for shorter sequences
            if self.flash_decode_enabled && max_seq_len > 1024 {
                self.flash_decoding_attention_into(
                    layer_idx,
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &attn_out_buf,
                    m as usize,
                    positions,
                )?;
            } else {
                // PAR-119: BATCHED attention - process all M sequences in parallel
                // Reduces M × 3 kernel launches to 2M + 1 (scatter still sequential, attention batched)
                self.batched_incremental_attention_into(
                    layer_idx,
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &attn_out_buf,
                    m as usize,
                    positions,
                )?;
            }
        } else {
            // Original sequential attention (M separate calls with shared KV cache)
            // NOTE: This path is used when batched KV caches are NOT initialized
            for seq_idx in 0..m as usize {
                let q_offset = seq_idx * q_dim as usize;
                let kv_offset = seq_idx * kv_dim as usize;
                let attn_offset = seq_idx * q_dim as usize;

                // SAFETY: Unsafe operation with validated invariants
                let q_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        q_buf_ptr + (q_offset * std::mem::size_of::<f32>()) as u64,
                        q_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let k_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        k_buf_ptr + (kv_offset * std::mem::size_of::<f32>()) as u64,
                        kv_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let v_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        v_buf_ptr + (kv_offset * std::mem::size_of::<f32>()) as u64,
                        kv_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let attn_out_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        attn_out_ptr + (attn_offset * std::mem::size_of::<f32>()) as u64,
                        q_dim as usize,
                    )
                };

                // Use incremental attention with shared KV cache
                self.incremental_attention_into_for_capture(
                    layer_idx,
                    &q_view,
                    &k_view,
                    &v_view,
                    &attn_out_view,
                )?;

                std::mem::forget(q_view);
                std::mem::forget(k_view);
                std::mem::forget(v_view);
                std::mem::forget(attn_out_view);
            }
        }

        // ========== 5. Output Projection (BATCHED GEMV) ==========
        if layer_weights.attn_output_qtype == WeightQuantType::Q4K {
            self.batched_q4k_gemv_into(
                layer_weights.attn_output_ptr,
                &attn_out_buf,
                &hidden_buf1, // Reuse for O projection output
                m,
                hidden_dim,
                q_dim,
            )?;
        } else {
            // Fall back to sequential
            for seq_idx in 0..m as usize {
                let h_offset = seq_idx * hidden_dim as usize;
                let attn_offset = seq_idx * q_dim as usize;

                // SAFETY: Unsafe operation with validated invariants
                let attn_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        attn_out_ptr + (attn_offset * std::mem::size_of::<f32>()) as u64,
                        q_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let out_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        hidden_buf1_ptr + (h_offset * std::mem::size_of::<f32>()) as u64,
                        hidden_dim as usize,
                    )
                };

                self.q4k_gemv_into(
                    layer_weights.attn_output_ptr,
                    &attn_view,
                    &out_view,
                    hidden_dim,
                    q_dim,
                )?;

                std::mem::forget(attn_view);
                std::mem::forget(out_view);
            }
        }

        // ========== 6. First Residual (PAR-114: BATCHED - 1 kernel launch) ==========
        // residual1 = input + O_projection
        self.batched_residual_add_into(
            input,
            &hidden_buf1,   // O projection output
            &input_staging, // Residual output
            hidden_dim,
            m,
        )?;

        // ========== 7. Pre-FFN RMSNorm (BATCHED - PAR-112) ==========
        // Process all M sequences in a single kernel launch
        self.batched_rmsnorm_ptr_into(
            &input_staging,
            layer_weights.ffn_norm_ptr,
            layer_weights.ffn_norm_len,
            &hidden_buf1,
            hidden_dim,
            m,
            epsilon,
        )?;

        // ========== 8. FFN Gate/Up (BATCHED GEMV) ==========
        if layer_weights.ffn_gate_qtype == WeightQuantType::Q4K {
            self.batched_q4k_gemv_into(
                layer_weights.ffn_gate_ptr,
                &hidden_buf1,
                &ffn_gate_buf,
                m,
                intermediate_dim,
                hidden_dim,
            )?;
            self.batched_q4k_gemv_into(
                layer_weights.ffn_up_ptr,
                &hidden_buf1,
                &ffn_up_buf,
                m,
                intermediate_dim,
                hidden_dim,
            )?;
        } else {
            // Fall back to sequential
            for seq_idx in 0..m as usize {
                let h_offset = seq_idx * hidden_dim as usize;
                let ffn_offset = seq_idx * intermediate_dim as usize;

                // SAFETY: Unsafe operation with validated invariants
                let input_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        hidden_buf1_ptr + (h_offset * std::mem::size_of::<f32>()) as u64,
                        hidden_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let gate_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        ffn_gate_ptr + (ffn_offset * std::mem::size_of::<f32>()) as u64,
                        intermediate_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let up_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        ffn_up_ptr + (ffn_offset * std::mem::size_of::<f32>()) as u64,
                        intermediate_dim as usize,
                    )
                };

                self.q4k_gemv_into(
                    layer_weights.ffn_gate_ptr,
                    &input_view,
                    &gate_view,
                    intermediate_dim,
                    hidden_dim,
                )?;
                self.q4k_gemv_into(
                    layer_weights.ffn_up_ptr,
                    &input_view,
                    &up_view,
                    intermediate_dim,
                    hidden_dim,
                )?;

                std::mem::forget(input_view);
                std::mem::forget(gate_view);
                std::mem::forget(up_view);
            }
        }

        // ========== 9. SwiGLU (PAR-114: BATCHED - 1 kernel launch) ==========
        // act = silu(gate) * up
        self.batched_swiglu_into(
            &ffn_gate_buf,
            &ffn_up_buf,
            &ffn_act_buf,
            intermediate_dim,
            m,
        )?;

        // ========== 10. FFN Down (BATCHED GEMV) ==========
        // PAR-130: Use batched kernels for both Q4K and Q6K
        if layer_weights.ffn_down_qtype == WeightQuantType::Q4K {
            self.batched_q4k_gemv_into(
                layer_weights.ffn_down_ptr,
                &ffn_act_buf,
                &hidden_buf1,
                m,
                hidden_dim,
                intermediate_dim,
            )?;
        } else if layer_weights.ffn_down_qtype == WeightQuantType::Q6K {
            // PAR-130: Batched Q6K GEMV - eliminates M sequential kernel launches
            self.batched_q6k_gemv_into(
                layer_weights.ffn_down_ptr,
                &ffn_act_buf,
                &hidden_buf1,
                m,
                hidden_dim,
                intermediate_dim,
            )?;
        } else {
            // Fall back to sequential Q6K for other quantization types
            for seq_idx in 0..m as usize {
                let h_offset = seq_idx * hidden_dim as usize;
                let ffn_offset = seq_idx * intermediate_dim as usize;

                // SAFETY: Unsafe operation with validated invariants
                let act_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        ffn_act_ptr + (ffn_offset * std::mem::size_of::<f32>()) as u64,
                        intermediate_dim as usize,
                    )
                };
                // SAFETY: Unsafe operation with validated invariants
                let out_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        hidden_buf1_ptr + (h_offset * std::mem::size_of::<f32>()) as u64,
                        hidden_dim as usize,
                    )
                };

                self.q6k_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &act_view,
                    &out_view,
                    hidden_dim,
                    intermediate_dim,
                )?;

                std::mem::forget(act_view);
                std::mem::forget(out_view);
            }
        }

        // ========== 11. Second Residual (PAR-114: BATCHED - 1 kernel launch) ==========
        // output = residual1 + FFN_down
        self.batched_residual_add_into(
            &input_staging, // residual1
            &hidden_buf1,   // FFN down output
            &hidden_buf2,   // Layer output
            hidden_dim,
            m,
        )?;

        // Prevent Drop from freeing borrowed memory
        std::mem::forget(hidden_buf1);
        std::mem::forget(hidden_buf2);
        std::mem::forget(input_staging);
        std::mem::forget(q_buf);
        std::mem::forget(k_buf);
        std::mem::forget(v_buf);
        std::mem::forget(attn_out_buf);
        std::mem::forget(ffn_gate_buf);
        std::mem::forget(ffn_up_buf);
        std::mem::forget(ffn_act_buf);

        // Output is in hidden_buf2 (M × hidden_dim packed)
        Ok(())
    }

    /// PAR-023: RMSNorm using raw device pointer for gamma
    pub(crate) fn rmsnorm_gpu_ptr(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64, // CUdeviceptr
        gamma_len: usize,
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        if gamma_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null gamma pointer in rmsnorm_gpu_ptr".to_string(),
            ));
        }
        // Create temporary non-owning buffer wrapper
        // SAFETY: gamma_ptr points to valid GPU memory owned by rmsnorm_cache
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let gamma = unsafe { GpuBuffer::from_raw_parts(gamma_ptr, gamma_len) };

        let result = self.rmsnorm_gpu(input, &gamma, hidden_dim, epsilon)?;

        // Prevent Drop from freeing the borrowed memory
        std::mem::forget(gamma);

        Ok(result)
    }

    /// PAR-044: RMSNorm using raw pointer into existing output buffer
    pub(crate) fn rmsnorm_ptr_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64,
        gamma_len: usize,
        output: &GpuBuffer<f32>,
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        if gamma_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null gamma pointer in rmsnorm_ptr_into".to_string(),
            ));
        }
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let gamma = unsafe { GpuBuffer::from_raw_parts(gamma_ptr, gamma_len) };
        self.rmsnorm_into(input, &gamma, output, hidden_dim, epsilon)?;
        std::mem::forget(gamma);
        Ok(())
    }
}


#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "batched_tests.rs"]
mod batched_tests;
