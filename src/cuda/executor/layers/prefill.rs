//! Batched prefill for prompt processing
//!
//! PMAT-PREFILL: Process all S prompt tokens through all transformer layers
//! in a single pass, replacing the serial token-by-token prefill loop.
//!
//! Expected improvement: Prefill 510ms → ~50ms (10x) for 20-token prompts.
//!
//! Uses the existing batched GEMV infrastructure (`transformer_layer_batched`)
//! which already handles M tokens at different positions. For prefill, M=S.

#![allow(clippy::wildcard_imports)]

use super::super::*;

impl CudaExecutor {
    /// PMAT-PREFILL: Initialize workspace for batched prefill
    ///
    /// Allocates workspace buffers sized for `max_seq_len` tokens.
    /// This is separate from the decode workspace because prefill
    /// processes many more tokens simultaneously.
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum prompt length to support
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    ///
    /// # Errors
    ///
    /// Returns error if GPU allocation fails.
    pub fn init_prefill_workspace(
        &mut self,
        max_seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<(), GpuError> {
        if max_seq_len == 0 {
            return Err(GpuError::InvalidParameter(
                "PMAT-PREFILL: max_seq_len must be > 0".to_string(),
            ));
        }

        let q_dim = self.kv_num_heads * self.kv_head_dim;
        let kv_dim = self.kv_num_kv_heads * self.kv_head_dim;
        let m = max_seq_len;

        // Allocate M× sized buffers for prefill
        self.workspace.hidden_buf1 = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        self.workspace.hidden_buf2 = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        self.workspace.input_staging = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        self.workspace.q_buf = Some(GpuBuffer::new(&self.context, q_dim * m)?);
        self.workspace.k_buf = Some(GpuBuffer::new(&self.context, kv_dim * m)?);
        self.workspace.v_buf = Some(GpuBuffer::new(&self.context, kv_dim * m)?);
        self.workspace.attn_out_buf = Some(GpuBuffer::new(&self.context, q_dim * m)?);
        self.workspace.ffn_gate_buf = Some(GpuBuffer::new(&self.context, intermediate_dim * m)?);
        self.workspace.ffn_up_buf = Some(GpuBuffer::new(&self.context, intermediate_dim * m)?);
        self.workspace.ffn_act_buf = Some(GpuBuffer::new(&self.context, intermediate_dim * m)?);
        self.workspace.normed_hidden_buf = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        self.workspace.positions_buf = Some(GpuBuffer::new(&self.context, m)?);
        // Logits only for last token (vocab_size allocated in decode workspace)
        // We don't need logits_buf for prefill since we only cache KV

        self.workspace.hidden_dim = hidden_dim;
        self.workspace.q_dim = q_dim;
        self.workspace.kv_dim = kv_dim;
        self.workspace.intermediate_dim = intermediate_dim;
        self.workspace.batch_size = m;
        self.workspace.initialized = true;

        Ok(())
    }

    /// PMAT-PREFILL: Process all prompt tokens through all layers in one pass
    ///
    /// Replaces the serial prefill loop that processes tokens one at a time.
    /// Uses batched GEMV kernels to process S tokens simultaneously through
    /// each transformer layer.
    ///
    /// After prefill completes:
    /// - KV cache is populated with entries at positions 0..S-1
    /// - No logits are returned (prefill only caches KV, doesn't predict)
    ///
    /// # Arguments
    ///
    /// * `embeddings` - All prompt token embeddings packed [S × hidden_dim]
    /// * `positions` - Position indices [0, 1, ..., S-1]
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `epsilon` - RMSNorm epsilon
    ///
    /// # Returns
    ///
    /// Ok(()) on success. KV cache is populated as side effect.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_all_layers_gpu(
        &mut self,
        embeddings: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let s = positions.len();
        if s == 0 {
            return Ok(());
        }

        let expected_input_len = s * hidden_dim as usize;
        if embeddings.len() != expected_input_len {
            return Err(GpuError::InvalidParameter(format!(
                "PMAT-PREFILL: embeddings.len() {} != S*hidden_dim = {}",
                embeddings.len(),
                expected_input_len
            )));
        }

        // Verify workspace initialized for this batch size
        if !self.workspace.initialized || self.workspace.batch_size < s {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PMAT-PREFILL: Workspace not initialized for S={} (have batch_size={})",
                s, self.workspace.batch_size
            )));
        }

        // 1. Upload all S embeddings to GPU
        let input_buf = GpuBuffer::from_host(&self.context, embeddings)?;

        // Get workspace buffer pointers
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PMAT-PREFILL: hidden_buf2 missing".to_string(),
                )
            })?
            .as_ptr();
        let hidden_buf2_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PMAT-PREFILL: hidden_buf2 missing".to_string(),
                )
            })?
            .len();

        // 2. Process all layers with batched GEMV
        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PMAT-PREFILL: Layer {} weights not indexed (have {})",
                    layer_idx,
                    self.indexed_layer_weights.len()
                )));
            }
            let layer_weights = self.get_indexed_layer(layer_idx).clone();

            // Use workspace output from previous layer (or input_buf for first layer)
            let layer_input_buf = if layer_idx == 0 {
                None // Use input_buf directly
            } else {
                // SAFETY: Pointer valid from allocation, length verified, used within scope
                Some(unsafe {
                    GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len)
                })
            };

            let layer_input = match &layer_input_buf {
                Some(buf) => buf,
                None => &input_buf,
            };

            self.transformer_layer_batched(
                layer_input,
                layer_idx,
                &layer_weights,
                s as u32,
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

        // After all layers, output is in hidden_buf2 [S × hidden_dim]
        // KV cache has been populated by transformer_layer_batched for each layer.
        // We don't need to compute logits — prefill only caches KV state.

        // Sync to ensure all GPU operations complete
        self.stream.synchronize()?;

        Ok(())
    }
}
