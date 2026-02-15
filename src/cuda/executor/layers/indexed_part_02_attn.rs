impl CudaExecutor {
    /// Phase 3-5: Attention + output projection + residual1
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn workspace_attention_residual_phase(
        &mut self,
        input: &GpuBuffer<f32>,
        hidden_buf1: &GpuBuffer<f32>,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        v_buf: &GpuBuffer<f32>,
        attn_out_buf: &GpuBuffer<f32>,
        input_staging: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &IndexedLayerWeights,
        hidden_dim: u32,
        q_dim: u32,
        skip_debug: bool,
        profiling: bool,
    ) -> Result<(), GpuError> {
        // 3. PAR-051: Incremental attention into pre-allocated workspace buffer
        // Eliminates 28 GPU allocations per token
        // PAR-054-FIX: Use capture-safe version during graph capture to skip debug sync
        let timer_attn = if profiling {
            self.start_brick_id(trueno::BrickId::AttentionScore)
        } else {
            None
        };
        let _seq_len = if skip_debug {
            self.incremental_attention_into_for_capture(
                layer_idx,
                q_buf,
                k_buf,
                v_buf,
                attn_out_buf,
            )?
        } else {
            self.incremental_attention_into(layer_idx, q_buf, k_buf, v_buf, attn_out_buf)?
        };
        if profiling {
            self.stop_brick_id(timer_attn, 1);
        }

        // PAR-058-DEBUG: Check attention output (skip during graph capture)
        if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3) {
            // PAR-058: Must sync on compute_stream since attention kernel runs there
            self.compute_stream.synchronize()?;
            let mut attn_out = vec![0.0f32; attn_out_buf.len()];
            attn_out_buf.copy_to_host(&mut attn_out)?;
            let nan_indices: Vec<usize> = attn_out
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_nan())
                .map(|(i, _)| i)
                .collect();
            if !nan_indices.is_empty() {
                // Analyze pattern by head (each head has 128 elements)
                let head_dim = 128;
                let mut heads_with_nan: Vec<usize> = Vec::new();
                for head in 0..12 {
                    let start = head * head_dim;
                    let end = start + head_dim;
                    let nan_in_head = nan_indices
                        .iter()
                        .filter(|&&i| i >= start && i < end)
                        .count();
                    if nan_in_head > 0 {
                        heads_with_nan.push(head);
                    }
                }
                eprintln!(
                    "[PAR-058-L{}] Attn output has {} NaN, heads with NaN: {:?}",
                    layer_idx,
                    nan_indices.len(),
                    heads_with_nan
                );
                eprintln!(
                    "[PAR-058-L{}] First 10 NaN indices: {:?}",
                    layer_idx,
                    &nan_indices[..10.min(nan_indices.len())]
                );
                if let Some((idx, val)) = attn_out.iter().enumerate().find(|(_, v)| !v.is_nan()) {
                    eprintln!(
                        "[PAR-058-L{}] First OK value at idx {}: {}",
                        layer_idx, idx, val
                    );
                }
            } else {
                eprintln!(
                    "[PAR-058-L{}] Attn OK, first 3: {:?}",
                    layer_idx,
                    &attn_out[..3.min(attn_out.len())]
                );
            }
        }

        // 4. Output projection: attn_out_buf -> hidden_buf1 (reuse, normed no longer needed)
        let timer_oproj = if profiling {
            self.start_brick_id(trueno::BrickId::OutputProjection)
        } else {
            None
        };
        self.gemv_dispatch(
            layer_weights.attn_output_qtype,
            layer_weights.attn_output_ptr,
            attn_out_buf, hidden_buf1, hidden_dim, q_dim,
        )?;
        if profiling {
            self.stop_brick_id(timer_oproj, 1);
        }

        // PAR-058-DEBUG: Check output projection (skip during graph capture)
        if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3) {
            self.stream.synchronize()?;
            let mut out_proj = vec![0.0f32; hidden_buf1.len()];
            hidden_buf1.copy_to_host(&mut out_proj)?;
            let nan_count = out_proj.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                eprintln!(
                    "[PAR-058-L{}] Output projection has {} NaN",
                    layer_idx, nan_count
                );
            } else {
                eprintln!(
                    "[PAR-058-L{}] Output proj OK, first 3: {:?}",
                    layer_idx,
                    &out_proj[..3.min(out_proj.len())]
                );
            }
        }

        // 5. First residual: input + projected -> input_staging (PAR-044 FIX)
        // NOTE: Using input_staging instead of hidden_buf2 to avoid read/write conflict
        // when input IS hidden_buf2 (layers 1+)
        // PAR-075: Cannot fuse with RmsNorm2 because we need input_staging for second residual
        let timer_res1 = if profiling {
            self.start_brick_timer("Residual1")
        } else {
            None
        };
        self.residual_add_into(input, hidden_buf1, input_staging, hidden_dim)?;
        if profiling {
            self.stop_brick_timer(timer_res1, 1);
        }

        // PAR-058-DEBUG: Check residual1 output (skip during graph capture)
        if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3) {
            self.stream.synchronize()?;
            let mut resid1 = vec![0.0f32; input_staging.len()];
            input_staging.copy_to_host(&mut resid1)?;
            let nan_count = resid1.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[PAR-058-L{}] Residual1 has {} NaN", layer_idx, nan_count);
            } else {
                eprintln!(
                    "[PAR-058-L{}] Residual1 OK, first 3: {:?}",
                    layer_idx,
                    &resid1[..3.min(resid1.len())]
                );
            }
        }

        Ok(())
    }
}
