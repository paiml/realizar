impl CudaExecutor {
    /// Batched QKV projection dispatch: Q4K → batched kernel, others → sequential fallback.
    #[allow(clippy::too_many_arguments)]
    fn batched_gemv_with_fallback(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        packed_input: &GpuBuffer<f32>,
        packed_output: &GpuBuffer<f32>,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n_per_seq: u32,
        k_per_seq: u32,
    ) -> Result<(), GpuError> {
        if qtype == WeightQuantType::Q4K {
            self.batched_q4k_gemv_into(
                weight_ptr, packed_input, packed_output, m, n_per_seq, k_per_seq,
            )
        } else {
            for seq_idx in 0..m as usize {
                let in_offset = seq_idx * k_per_seq as usize;
                let out_offset = seq_idx * n_per_seq as usize;
                let input_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        packed_input_ptr + (in_offset * std::mem::size_of::<f32>()) as u64,
                        k_per_seq as usize,
                    )
                };
                let output_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        packed_output_ptr + (out_offset * std::mem::size_of::<f32>()) as u64,
                        n_per_seq as usize,
                    )
                };
                self.gemv_dispatch(qtype, weight_ptr, &input_view, &output_view, n_per_seq, k_per_seq)?;
                std::mem::forget(input_view);
                std::mem::forget(output_view);
            }
            Ok(())
        }
    }

    /// Phase 1: RMSNorm + QKV projections + bias + RoPE (batched)
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn batched_qkv_rope_phase(
        &mut self,
        input: &GpuBuffer<f32>,
        hidden_buf1: &GpuBuffer<f32>,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        v_buf: &GpuBuffer<f32>,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        hidden_buf1_ptr: u64,
        _layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        m: u32,
        positions: &[u32],
        hidden_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // ========== 1. Pre-attention RMSNorm (BATCHED - PAR-112) ==========
        self.batched_rmsnorm_ptr_into(
            input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            hidden_buf1,
            hidden_dim,
            m,
            epsilon,
        )?;

        // ========== 2. Q/K/V Projections (BATCHED GEMV) ==========
        self.batched_gemv_with_fallback(
            layer_weights.attn_q_qtype, layer_weights.attn_q_ptr,
            hidden_buf1, q_buf, hidden_buf1_ptr, q_buf_ptr,
            m, q_dim, hidden_dim,
        )?;
        self.batched_gemv_with_fallback(
            layer_weights.attn_k_qtype, layer_weights.attn_k_ptr,
            hidden_buf1, k_buf, hidden_buf1_ptr, k_buf_ptr,
            m, kv_dim, hidden_dim,
        )?;
        self.batched_gemv_with_fallback(
            layer_weights.attn_v_qtype, layer_weights.attn_v_ptr,
            hidden_buf1, v_buf, hidden_buf1_ptr, v_buf_ptr,
            m, kv_dim, hidden_dim,
        )?;

        // ========== 2b. QKV Bias ==========
        if layer_weights.attn_q_bias_len > 0 {
            let q_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_q_bias_ptr,
                    layer_weights.attn_q_bias_len,
                )
            };
            for seq_idx in 0..m as usize {
                let offset = seq_idx * q_dim as usize;
                let q_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        q_buf_ptr + (offset * std::mem::size_of::<f32>()) as u64,
                        q_dim as usize,
                    )
                };
                self.residual_add_into(&q_view, &q_bias_buf, &q_view, q_dim)?;
                std::mem::forget(q_view);
            }
            std::mem::forget(q_bias_buf);
        }
        if layer_weights.attn_k_bias_len > 0 {
            let k_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_k_bias_ptr,
                    layer_weights.attn_k_bias_len,
                )
            };
            for seq_idx in 0..m as usize {
                let offset = seq_idx * kv_dim as usize;
                let k_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        k_buf_ptr + (offset * std::mem::size_of::<f32>()) as u64,
                        kv_dim as usize,
                    )
                };
                self.residual_add_into(&k_view, &k_bias_buf, &k_view, kv_dim)?;
                std::mem::forget(k_view);
            }
            std::mem::forget(k_bias_buf);
        }
        if layer_weights.attn_v_bias_len > 0 {
            let v_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_v_bias_ptr,
                    layer_weights.attn_v_bias_len,
                )
            };
            for seq_idx in 0..m as usize {
                let offset = seq_idx * kv_dim as usize;
                let v_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        v_buf_ptr + (offset * std::mem::size_of::<f32>()) as u64,
                        kv_dim as usize,
                    )
                };
                self.residual_add_into(&v_view, &v_bias_buf, &v_view, kv_dim)?;
                std::mem::forget(v_view);
            }
            std::mem::forget(v_bias_buf);
        }

        // ========== 3. RoPE on Q/K (PAR-114) ==========
        let num_heads = self.kv_num_heads as u32;
        let num_kv_heads = self.kv_num_kv_heads as u32;
        let head_dim = self.kv_head_dim as u32;
        let theta = self.rope_theta;

        let positions_buf_ptr = self
            .workspace
            .positions_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-114: positions_buf not initialized".to_string())
            })?
            .as_ptr();
        // SAFETY: positions_buf_ptr was obtained from a valid GpuBuffer via as_ptr(),
        // and m is the batch size matching the original allocation.
        let mut positions_buf =
            unsafe { GpuBuffer::<u32>::from_raw_parts(positions_buf_ptr, m as usize) };
        let positions_u32: Vec<u32> = positions.to_vec();
        positions_buf.copy_from_host(&positions_u32)?;

        if self.rope_type == 2 {
            // Sequential NEOX RoPE (no batched NEOX kernel yet)
            for seq_idx in 0..m as usize {
                let q_offset = seq_idx * q_dim as usize;
                let kv_offset = seq_idx * kv_dim as usize;
                let position = positions[seq_idx];
                let q_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        q_buf_ptr + (q_offset * std::mem::size_of::<f32>()) as u64,
                        q_dim as usize,
                    )
                };
                let k_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        k_buf_ptr + (kv_offset * std::mem::size_of::<f32>()) as u64,
                        kv_dim as usize,
                    )
                };
                self.rope_neox_into(&q_view, &q_view, position, num_heads, head_dim, theta)?;
                self.rope_neox_into(&k_view, &k_view, position, num_kv_heads, head_dim, theta)?;
                std::mem::forget(q_view);
                std::mem::forget(k_view);
            }
        } else {
            // PAR-114: Batched RoPE (all M sequences in one launch)
            self.batched_rope_into(q_buf, q_buf, &positions_buf, num_heads, head_dim, m, theta)?;
            self.batched_rope_into(k_buf, k_buf, &positions_buf, num_kv_heads, head_dim, m, theta)?;
        }

        std::mem::forget(positions_buf);
        Ok(())
    }
}
