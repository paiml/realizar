impl CudaExecutor {
    /// Phase 2: Attention + output projection + residuals + FFN (batched)
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn batched_attn_ffn_phase(
        &mut self,
        input: &GpuBuffer<f32>,
        hidden_buf1: &GpuBuffer<f32>,
        hidden_buf2: &GpuBuffer<f32>,
        input_staging: &GpuBuffer<f32>,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        v_buf: &GpuBuffer<f32>,
        attn_out_buf: &GpuBuffer<f32>,
        ffn_gate_buf: &GpuBuffer<f32>,
        ffn_up_buf: &GpuBuffer<f32>,
        ffn_act_buf: &GpuBuffer<f32>,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        hidden_buf1_ptr: u64,
        ffn_gate_ptr: u64,
        ffn_up_ptr: u64,
        ffn_act_ptr: u64,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        m: u32,
        positions: &[u32],
        hidden_dim: u32,
        intermediate_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // ========== 4. Attention ==========
        // PAR-119: Use batched attention if batched KV caches are initialized
        if self.batched_kv_stride > 0 && self.batched_kv_k_caches.contains_key(&layer_idx) {
            let max_seq_len = self
                .batched_kv_lengths
                .iter()
                .take(m as usize)
                .copied()
                .max()
                .unwrap_or(0);

            if self.flash_decode_enabled && max_seq_len > 1024 {
                self.flash_decoding_attention_into(
                    layer_idx, q_buf, k_buf, v_buf, attn_out_buf,
                    m as usize, positions,
                )?;
            } else {
                self.batched_incremental_attention_into(
                    layer_idx, q_buf, k_buf, v_buf, attn_out_buf,
                    m as usize, positions,
                )?;
            }
        } else {
            // Sequential attention fallback (shared KV cache)
            for seq_idx in 0..m as usize {
                let q_offset = seq_idx * q_dim as usize;
                let kv_offset = seq_idx * kv_dim as usize;
                let attn_offset = seq_idx * q_dim as usize;

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
                let v_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        v_buf_ptr + (kv_offset * std::mem::size_of::<f32>()) as u64,
                        kv_dim as usize,
                    )
                };
                let attn_out_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        attn_out_ptr + (attn_offset * std::mem::size_of::<f32>()) as u64,
                        q_dim as usize,
                    )
                };

                self.incremental_attention_into_for_capture(
                    layer_idx, &q_view, &k_view, &v_view, &attn_out_view,
                )?;

                std::mem::forget(q_view);
                std::mem::forget(k_view);
                std::mem::forget(v_view);
                std::mem::forget(attn_out_view);
            }
        }

        // ========== 5. Output Projection (BATCHED GEMV) ==========
        self.batched_gemv_with_fallback(
            layer_weights.attn_output_qtype, layer_weights.attn_output_ptr,
            attn_out_buf, hidden_buf1, attn_out_ptr, hidden_buf1_ptr,
            m, hidden_dim, q_dim,
        )?;

        // ========== 6. First Residual (PAR-114: BATCHED) ==========
        self.batched_residual_add_into(input, hidden_buf1, input_staging, hidden_dim, m)?;

        // ========== 7. Pre-FFN RMSNorm (BATCHED - PAR-112) ==========
        self.batched_rmsnorm_ptr_into(
            input_staging,
            layer_weights.ffn_norm_ptr,
            layer_weights.ffn_norm_len,
            hidden_buf1,
            hidden_dim,
            m,
            epsilon,
        )?;

        // ========== 8. FFN Gate/Up (BATCHED GEMV) ==========
        self.batched_gemv_with_fallback(
            layer_weights.ffn_gate_qtype, layer_weights.ffn_gate_ptr,
            hidden_buf1, ffn_gate_buf, hidden_buf1_ptr, ffn_gate_ptr,
            m, intermediate_dim, hidden_dim,
        )?;
        self.batched_gemv_with_fallback(
            layer_weights.ffn_up_qtype, layer_weights.ffn_up_ptr,
            hidden_buf1, ffn_up_buf, hidden_buf1_ptr, ffn_up_ptr,
            m, intermediate_dim, hidden_dim,
        )?;

        // ========== 9. SwiGLU (PAR-114: BATCHED) ==========
        self.batched_swiglu_into(ffn_gate_buf, ffn_up_buf, ffn_act_buf, intermediate_dim, m)?;

        // ========== 10. FFN Down (BATCHED GEMV) ==========
        // PAR-130: Use batched kernels for both Q4K and Q6K
        if layer_weights.ffn_down_qtype == WeightQuantType::Q4K {
            self.batched_q4k_gemv_into(
                layer_weights.ffn_down_ptr,
                ffn_act_buf, hidden_buf1, m, hidden_dim, intermediate_dim,
            )?;
        } else if layer_weights.ffn_down_qtype == WeightQuantType::Q6K {
            self.batched_q6k_gemv_into(
                layer_weights.ffn_down_ptr,
                ffn_act_buf, hidden_buf1, m, hidden_dim, intermediate_dim,
            )?;
        } else {
            // Fall back to sequential for other quantization types
            for seq_idx in 0..m as usize {
                let h_offset = seq_idx * hidden_dim as usize;
                let ffn_offset = seq_idx * intermediate_dim as usize;
                let act_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        ffn_act_ptr + (ffn_offset * std::mem::size_of::<f32>()) as u64,
                        intermediate_dim as usize,
                    )
                };
                let out_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        hidden_buf1_ptr + (h_offset * std::mem::size_of::<f32>()) as u64,
                        hidden_dim as usize,
                    )
                };
                self.q6k_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &act_view, &out_view, hidden_dim, intermediate_dim,
                )?;
                std::mem::forget(act_view);
                std::mem::forget(out_view);
            }
        }

        // ========== 11. Second Residual (PAR-114: BATCHED) ==========
        self.batched_residual_add_into(input_staging, hidden_buf1, hidden_buf2, hidden_dim, m)?;

        Ok(())
    }
}
