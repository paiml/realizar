impl CudaExecutor {
    /// Phase 6-10: FFN RMSNorm + gate/up projections + SwiGLU + down projection + residual2
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn workspace_ffn_phase(
        &mut self,
        hidden_buf1: &GpuBuffer<f32>,
        hidden_buf2: &GpuBuffer<f32>,
        input_staging: &GpuBuffer<f32>,
        ffn_gate_buf: &GpuBuffer<f32>,
        ffn_up_buf: &GpuBuffer<f32>,
        ffn_act_buf: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        skip_debug: bool,
        profiling: bool,
    ) -> Result<(), GpuError> {
        // 6. Pre-FFN RMSNorm: residual1 (input_staging) -> hidden_buf1 (ffn_normed)
        let timer_rmsnorm2 = if profiling {
            self.start_brick_id(trueno::BrickId::RmsNorm)
        } else {
            None
        };
        self.rmsnorm_ptr_into(
            input_staging,
            layer_weights.ffn_norm_ptr,
            layer_weights.ffn_norm_len,
            hidden_buf1,
            hidden_dim,
            epsilon,
        )?;
        if profiling {
            self.stop_brick_id(timer_rmsnorm2, 1);
        }

        // 7. FFN gate/up projections -> workspace buffers
        // PAR-077: Fused kernel BLOCKED - 3x slower due to shared memory + barrier overhead
        let timer_ffn_gate_up = if profiling {
            self.start_brick_id(trueno::BrickId::GateProjection)
        } else {
            None
        };

        // Gate projection
        self.gemv_dispatch(
            layer_weights.ffn_gate_qtype,
            layer_weights.ffn_gate_ptr,
            hidden_buf1, ffn_gate_buf, intermediate_dim, hidden_dim,
        )?;

        // Up projection
        self.gemv_dispatch(
            layer_weights.ffn_up_qtype,
            layer_weights.ffn_up_ptr,
            hidden_buf1, ffn_up_buf, intermediate_dim, hidden_dim,
        )?;

        if profiling {
            self.stop_brick_id(timer_ffn_gate_up, 1);
        }

        // PAR-058-DEBUG: Check FFN gate/up outputs (skip during graph capture)
        if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3) {
            self.stream.synchronize()?;
            let mut gate_out = vec![0.0f32; ffn_gate_buf.len()];
            ffn_gate_buf.copy_to_host(&mut gate_out)?;
            let gate_nan = gate_out.iter().filter(|x| x.is_nan()).count();
            if gate_nan > 0 {
                eprintln!("[PAR-058-L{}] FFN gate has {} NaN", layer_idx, gate_nan);
            } else {
                eprintln!(
                    "[PAR-058-L{}] FFN gate OK, first 3: {:?}",
                    layer_idx,
                    &gate_out[..3.min(gate_out.len())]
                );
            }
            let mut up_out = vec![0.0f32; ffn_up_buf.len()];
            ffn_up_buf.copy_to_host(&mut up_out)?;
            let up_nan = up_out.iter().filter(|x| x.is_nan()).count();
            if up_nan > 0 {
                eprintln!("[PAR-058-L{}] FFN up has {} NaN", layer_idx, up_nan);
            } else {
                eprintln!(
                    "[PAR-058-L{}] FFN up OK, first 3: {:?}",
                    layer_idx,
                    &up_out[..3.min(up_out.len())]
                );
            }
        }

        // 8. SwiGLU activation: gate * silu(up) -> ffn_act_buf
        let timer_swiglu = if profiling {
            self.start_brick_id(trueno::BrickId::Activation)
        } else {
            None
        };
        self.fused_swiglu_into(ffn_gate_buf, ffn_up_buf, ffn_act_buf, intermediate_dim)?;
        if profiling {
            self.stop_brick_id(timer_swiglu, 1);
        }

        // PAR-058-DEBUG: Check SwiGLU output (skip during graph capture)
        if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3) {
            self.stream.synchronize()?;
            let mut swiglu_out = vec![0.0f32; ffn_act_buf.len()];
            ffn_act_buf.copy_to_host(&mut swiglu_out)?;
            let swiglu_nan = swiglu_out.iter().filter(|x| x.is_nan()).count();
            if swiglu_nan > 0 {
                eprintln!("[PAR-058-L{}] SwiGLU has {} NaN", layer_idx, swiglu_nan);
            } else {
                eprintln!(
                    "[PAR-058-L{}] SwiGLU OK, first 3: {:?}",
                    layer_idx,
                    &swiglu_out[..3.min(swiglu_out.len())]
                );
            }
        }

        // PAR-058-DEBUG: Check FFN down weight info (skip during graph capture)
        if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3) {
            eprintln!(
                "[PAR-058-L{}] FFN down weight ptr={:#x}, len={}, qtype={:?}",
                layer_idx,
                layer_weights.ffn_down_ptr,
                layer_weights.ffn_down_len,
                layer_weights.ffn_down_qtype
            );
            eprintln!(
                "[PAR-058-L{}] FFN down call: n={}, k={}",
                layer_idx, hidden_dim, intermediate_dim
            );
            let n_super_blocks = (intermediate_dim as usize + 255) / 256;
            let expected_q4k = hidden_dim as usize * n_super_blocks * 144;
            let expected_q5k = hidden_dim as usize * n_super_blocks * 176;
            eprintln!(
                "[PAR-058-L{}] Expected sizes: Q4K={}, Q5K={} (n_sb={})",
                layer_idx, expected_q4k, expected_q5k, n_super_blocks
            );
        }

        // 9. FFN down projection: ffn_act -> hidden_buf1 (reuse, ffn_normed no longer needed)
        // PAR-058: Use correct kernel based on FFN down quantization type
        // PAR-105-FIX: Only override qtype if metadata qtype doesn't match expected size
        let metadata_qtype = layer_weights.ffn_down_qtype;
        let metadata_matches = metadata_qtype.matches_size(
            layer_weights.ffn_down_len,
            hidden_dim as usize,
            intermediate_dim as usize,
        );
        let ffn_down_qtype = if metadata_matches {
            metadata_qtype
        } else {
            WeightQuantType::from_size(
                layer_weights.ffn_down_len,
                hidden_dim as usize,
                intermediate_dim as usize,
            )
            .unwrap_or(metadata_qtype)
        };

        // Log if we overrode the type
        if !skip_debug && ffn_down_qtype != layer_weights.ffn_down_qtype && layer_idx == 0 {
            eprintln!(
                "[PAR-058] FFN down qtype override: {:?} -> {:?} (size-based detection)",
                layer_weights.ffn_down_qtype, ffn_down_qtype
            );
        }

        // CORRECTNESS-002: Debug actual qtype being used
        if !skip_debug && layer_idx == 2 {
            eprintln!(
                "[CORRECTNESS-002] L2 FFN down: metadata_qtype={:?}, detected_qtype={:?}",
                layer_weights.ffn_down_qtype, ffn_down_qtype
            );
        }

        // CORRECTNESS-002: Debug first super-block of Layer 2 FFN down weights (Q4K only)
        if !skip_debug && layer_idx == 2 && ffn_down_qtype == WeightQuantType::Q4K {
            self.stream.synchronize()?;
            eprintln!(
                "[CORRECTNESS-002] L2 FFN down: ptr={:#x}, n={}, k={}",
                layer_weights.ffn_down_ptr, hidden_dim, intermediate_dim
            );
            let mut host_data = vec![0u8; 144];
            let debug_buf =
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe { GpuBuffer::<u8>::from_raw_parts(layer_weights.ffn_down_ptr, 144) };
            debug_buf.copy_to_host(&mut host_data)?;
            std::mem::forget(debug_buf);
            let d_bytes = [host_data[0], host_data[1]];
            let dmin_bytes = [host_data[2], host_data[3]];
            let d_f16 = half::f16::from_le_bytes(d_bytes);
            let dmin_f16 = half::f16::from_le_bytes(dmin_bytes);
            eprintln!(
                "[CORRECTNESS-002] L2 FFN down sb0: d_f16={:?} ({:.6}), dmin_f16={:?} ({:.6})",
                d_f16, d_f16.to_f32(), dmin_f16, dmin_f16.to_f32()
            );
            eprintln!(
                "[CORRECTNESS-002] L2 FFN down sb0 first 20 bytes: {:?}",
                host_data.get(..20).unwrap_or(&[])
            );
        }

        let timer_ffn_down = if profiling {
            self.start_brick_id(trueno::BrickId::DownProjection)
        } else {
            None
        };
        self.gemv_dispatch(
            ffn_down_qtype,
            layer_weights.ffn_down_ptr,
            ffn_act_buf, hidden_buf1, hidden_dim, intermediate_dim,
        )?;
        if profiling {
            self.stop_brick_id(timer_ffn_down, 1);
        }

        // PAR-058-DEBUG: Check FFN down output (skip during graph capture)
        if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3) {
            self.stream.synchronize()?;
            let mut ffn_down = vec![0.0f32; hidden_buf1.len()];
            hidden_buf1.copy_to_host(&mut ffn_down)?;
            let nan_count = ffn_down.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                eprintln!(
                    "[PAR-058-L{}] FFN down has {} NaN, first 10: {:?}",
                    layer_idx,
                    nan_count,
                    &ffn_down[..10.min(ffn_down.len())]
                );
            } else {
                eprintln!(
                    "[PAR-058-L{}] FFN down OK, first 3: {:?}",
                    layer_idx,
                    &ffn_down[..3.min(ffn_down.len())]
                );
            }
        }

        // 10. Second residual: residual1 (input_staging) + ffn_out (hidden_buf1) -> hidden_buf2
        // PAR-044 FIX: Now safe because residual1 is in input_staging, not hidden_buf2
        let timer_res2 = if profiling {
            self.start_brick_timer("Residual2")
        } else {
            None
        };
        self.residual_add_into(input_staging, hidden_buf1, hidden_buf2, hidden_dim)?;
        if profiling {
            self.stop_brick_timer(timer_res2, 1);
        }

        // PAR-058-DEBUG: Check layer output (skip during graph capture)
        if !skip_debug && layer_idx < 10 {
            self.stream.synchronize()?;
            let mut layer_out = vec![0.0f32; hidden_buf2.len()];
            hidden_buf2.copy_to_host(&mut layer_out)?;
            let nan_count = layer_out.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                eprintln!(
                    "[PAR-058-L{}] Layer output has {} NaN (qtype: {:?})",
                    layer_idx, nan_count, layer_weights.ffn_down_qtype
                );
            } else {
                eprintln!(
                    "[PAR-058-L{}] Layer output OK, first 3: {:?}",
                    layer_idx,
                    &layer_out[..3.min(layer_out.len())]
                );
            }
        }

        Ok(())
    }
}
