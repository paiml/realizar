//! Transformer workspace and GEMV buffer pool management
//!
//! This module implements:
//! - PAR-044: Transformer Workspace (zero-allocation forward pass)
//! - PAR-111: Batched workspace for multi-sequence processing
//! - PAR-007: GEMV Buffer Pool (avoid per-call allocation)

use super::*;

impl CudaExecutor {
    // ========================================================================
    // PAR-044: Transformer Workspace (zero-allocation forward pass)
    // ========================================================================

    /// Initialize workspace buffers for zero-allocation forward pass
    ///
    /// MUST be called after `build_indexed_weights()` and before first forward pass.
    /// Allocates all intermediate buffers once; they are reused for every token.
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    ///
    /// # Errors
    ///
    /// Returns error if GPU allocation fails.
    pub fn init_workspace(
        &mut self,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<(), GpuError> {
        let q_dim = self.kv_num_heads * self.kv_head_dim;
        let kv_dim = self.kv_num_kv_heads * self.kv_head_dim;

        // Allocate all workspace buffers (10 buffers total for zero-allocation forward)
        // PAR-051: Added attn_out_buf to eliminate 28 allocations per token
        self.workspace.hidden_buf1 = Some(GpuBuffer::new(&self.context, hidden_dim)?);
        self.workspace.hidden_buf2 = Some(GpuBuffer::new(&self.context, hidden_dim)?);
        self.workspace.input_staging = Some(GpuBuffer::new(&self.context, hidden_dim)?);
        self.workspace.q_buf = Some(GpuBuffer::new(&self.context, q_dim)?);
        self.workspace.k_buf = Some(GpuBuffer::new(&self.context, kv_dim)?);
        self.workspace.v_buf = Some(GpuBuffer::new(&self.context, kv_dim)?);
        self.workspace.attn_out_buf = Some(GpuBuffer::new(&self.context, q_dim)?); // PAR-051
        self.workspace.ffn_gate_buf = Some(GpuBuffer::new(&self.context, intermediate_dim)?);
        self.workspace.ffn_up_buf = Some(GpuBuffer::new(&self.context, intermediate_dim)?);
        self.workspace.ffn_act_buf = Some(GpuBuffer::new(&self.context, intermediate_dim)?);

        self.workspace.hidden_dim = hidden_dim;
        self.workspace.q_dim = q_dim;
        self.workspace.kv_dim = kv_dim;
        self.workspace.intermediate_dim = intermediate_dim;
        self.workspace.batch_size = 1;
        self.workspace.initialized = true;

        Ok(())
    }

    /// PAR-111: Initialize batched workspace for multi-sequence processing
    ///
    /// Allocates M× larger buffers for processing batch_size sequences in parallel.
    /// Used with batched GEMV kernels to achieve 16x speedup over sequential.
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `batch_size` - Number of sequences to process in parallel (typically 4)
    ///
    /// # Errors
    ///
    /// Returns error if GPU allocation fails.
    pub fn init_batched_workspace(
        &mut self,
        hidden_dim: usize,
        intermediate_dim: usize,
        batch_size: usize,
    ) -> Result<(), GpuError> {
        // PAR-129: Extended to M=32 via 4-warp kernel
        if batch_size == 0 || batch_size > 32 {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-111: batch_size must be 1-32, got {}",
                batch_size
            )));
        }

        let q_dim = self.kv_num_heads * self.kv_head_dim;
        let kv_dim = self.kv_num_kv_heads * self.kv_head_dim;

        // PAR-111: Allocate M× larger buffers for batched processing
        let m = batch_size;
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
        // PAR-111: normed_hidden_buf for output norm before LM head
        self.workspace.normed_hidden_buf = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        // PAR-114: positions buffer for batched RoPE
        self.workspace.positions_buf = Some(GpuBuffer::new(&self.context, m)?);

        self.workspace.hidden_dim = hidden_dim;
        self.workspace.q_dim = q_dim;
        self.workspace.kv_dim = kv_dim;
        self.workspace.intermediate_dim = intermediate_dim;
        self.workspace.batch_size = batch_size;
        self.workspace.initialized = true;

        eprintln!(
            "[PAR-111] Initialized batched workspace: batch_size={}, hidden={}×{}, q={}×{}, kv={}×{}, ffn={}×{}",
            batch_size,
            hidden_dim, m,
            q_dim, m,
            kv_dim, m,
            intermediate_dim, m
        );

        Ok(())
    }

    /// Check if workspace is initialized
    #[must_use]
    pub fn has_workspace(&self) -> bool {
        self.workspace.initialized
    }

    /// PAR-111: Get the batch size of the current workspace
    #[must_use]
    pub fn workspace_batch_size(&self) -> usize {
        self.workspace.batch_size
    }

    /// PAR-062: Check if CUDA decode graph has been captured
    ///
    /// Returns true if the decode graph is ready for replay.
    /// The graph is captured on first forward pass with `forward_all_layers_gpu_to_logits_graphed`.
    #[must_use]
    pub fn has_decode_graph(&self) -> bool {
        self.decode_graph.is_some()
    }

    /// Clear workspace buffers (releases GPU memory)
    pub fn clear_workspace(&mut self) {
        self.workspace = TransformerWorkspace::default();
    }

    /// Clear decode graph and related state
    ///
    /// Call this before starting a new generation session to ensure
    /// the graph is recaptured with fresh state.
    pub fn clear_decode_graph(&mut self) {
        self.decode_graph = None;
        self.decode_token_count = 0;
        self.graph_input_buf = None;
        self.position_buf = None;
        self.seq_len_buf = None;
    }

    // ========================================================================
    // PAR-007: GEMV Buffer Pool (avoid per-call allocation)
    // ========================================================================

    /// Ensure GEMV input buffer has exact required size
    ///
    /// Returns a reference to the GPU buffer pointer. The buffer is
    /// reallocated only when the size changes (common case: same size reused).
    pub(crate) fn ensure_gemv_input_buffer(
        &mut self,
        required_size: usize,
    ) -> Result<u64, GpuError> {
        // Reallocate only if size changed (common case: reuse existing buffer)
        if self.gemv_input_size != required_size {
            self.gemv_input_buffer = Some(GpuBuffer::new(&self.context, required_size)?);
            self.gemv_input_size = required_size;
        }
        Ok(self
            .gemv_input_buffer
            .as_ref()
            .expect("buffer just created")
            .as_ptr())
    }

    /// Ensure GEMV output buffer has exact required size
    pub(crate) fn ensure_gemv_output_buffer(
        &mut self,
        required_size: usize,
    ) -> Result<u64, GpuError> {
        if self.gemv_output_size != required_size {
            self.gemv_output_buffer = Some(GpuBuffer::new(&self.context, required_size)?);
            self.gemv_output_size = required_size;
        }
        Ok(self
            .gemv_output_buffer
            .as_ref()
            .expect("buffer just created")
            .as_ptr())
    }

    /// Copy input data to cached GEMV input buffer
    pub(crate) fn copy_to_gemv_input(&mut self, input: &[f32]) -> Result<(), GpuError> {
        let buf = self
            .gemv_input_buffer
            .as_mut()
            .expect("buffer should exist");
        buf.copy_from_host(input)
    }

    /// Copy output data from cached GEMV output buffer
    pub(crate) fn copy_from_gemv_output(&self, output: &mut [f32]) -> Result<(), GpuError> {
        let buf = self
            .gemv_output_buffer
            .as_ref()
            .expect("buffer should exist");
        buf.copy_to_host(output)
    }

    /// Get GEMV buffer pool statistics
    #[must_use]
    pub fn gemv_buffer_stats(&self) -> (usize, usize) {
        (self.gemv_input_size * 4, self.gemv_output_size * 4) // bytes
    }

    /// Clear GEMV buffers (releases GPU memory)
    pub fn clear_gemv_buffers(&mut self) {
        self.gemv_input_buffer = None;
        self.gemv_output_buffer = None;
        self.gemv_input_size = 0;
        self.gemv_output_size = 0;
    }
}
