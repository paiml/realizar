//! Indexed layer operations for optimized decode path
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-043: transformer_layer_indexed (hot path for decode)
//! - Private helpers for indexed operations

#![allow(clippy::wildcard_imports)]

use super::super::*;

impl CudaExecutor {
    /// PAR-043: Transformer layer using pre-indexed device pointers (async, no sync)
    ///
    /// This is the **hot path** for decode. Eliminates ALL string formatting and HashMap
    /// lookups (7 per layer = ~224 allocations/lookups per forward pass for 32 layers).
    ///
    /// Measured improvement: ~10ms per token overhead → ~0.1ms per token overhead
    ///
    /// # Arguments
    ///
    /// * `input` - GPU buffer with hidden states [hidden_dim]
    /// * `layer_idx` - Layer index (for incremental attention KV cache)
    /// * `layer_weights` - Pre-indexed pointers from `indexed_layer_weights`
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_indexed(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &IndexedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // 1. Pre-attention RMSNorm using indexed gamma pointer
        let normed = self.rmsnorm_gpu_ptr(
            input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            hidden_dim,
            epsilon,
        )?;

        // 2. Q/K/V projections using indexed pointers (no sync)
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        let q =
            self.q4k_gemv_indexed_async(layer_weights.attn_q_ptr, &normed, q_dim, hidden_dim)?;
        let k =
            self.q4k_gemv_indexed_async(layer_weights.attn_k_ptr, &normed, kv_dim, hidden_dim)?;
        // PAR-058: Use correct kernel based on V weight quantization type
        let v = match layer_weights.attn_v_qtype {
            WeightQuantType::Q6K => {
                self.q6k_gemv_indexed_async(layer_weights.attn_v_ptr, &normed, kv_dim, hidden_dim)?
            },
            _ => {
                self.q4k_gemv_indexed_async(layer_weights.attn_v_ptr, &normed, kv_dim, hidden_dim)?
            },
        };

        // 3. Incremental attention (has internal sync for KV cache update)
        let (attn_out, _seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;

        // 4. Output projection using indexed pointer (no sync)
        let projected = self.q4k_gemv_indexed_async(
            layer_weights.attn_output_ptr,
            &attn_out,
            hidden_dim,
            q_dim,
        )?;

        // 5. First residual add (no sync)
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;

        // 6. Pre-FFN RMSNorm using indexed gamma pointer
        let ffn_normed = self.rmsnorm_gpu_ptr(
            &residual1,
            layer_weights.ffn_norm_ptr,
            layer_weights.ffn_norm_len,
            hidden_dim,
            epsilon,
        )?;

        // 7. FFN SwiGLU using indexed pointers (no sync)
        let ffn_out = self.fused_ffn_swiglu_indexed_gpu(
            &ffn_normed,
            layer_weights.ffn_gate_ptr,
            layer_weights.ffn_up_ptr,
            layer_weights.ffn_down_ptr,
            hidden_dim,
            intermediate_dim,
        )?;

        // 8. Second residual add (no sync)
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;

        Ok(output)
    }

    /// PAR-044: Transformer layer with zero allocations using workspace buffers
    ///
    /// Uses pre-allocated workspace buffers for all intermediate tensors.
    /// Eliminates ~288 buffer allocations per token.
    ///
    /// # Buffer Usage
    ///
    /// Workspace buffers used:
    /// - hidden_buf1: normed, projected, ffn_normed, ffn_out (reused)
    /// - hidden_buf2: residual1, final output
    /// - q_buf: Q projection, then attention output
    /// - k_buf: K projection
    /// - v_buf: V projection
    /// - ffn_gate_buf: FFN gate projection
    /// - ffn_up_buf: FFN up projection
    /// - ffn_act_buf: SwiGLU activation result
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success. Output is written to workspace.hidden_buf2.
    /// PAR-054: Transformer layer for graph capture (no debug output)
    /// PAR-070: Takes position but uses indirect mode (reads from position_buf) during graph capture
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transformer_layer_workspace_for_capture(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &IndexedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        position: u32,
    ) -> Result<(), GpuError> {
        self.transformer_layer_workspace_inner(
            input,
            layer_idx,
            layer_weights,
            hidden_dim,
            intermediate_dim,
            epsilon,
            position,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transformer_layer_workspace(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &IndexedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        position: u32, // PAR-070: Explicit position for RoPE and KV cache
    ) -> Result<(), GpuError> {
        // PERF-001: skip_debug=true disables stream.synchronize() calls and debug prints
        // that were causing ~4x slowdown (70 tok/s -> target 280+ tok/s)
        // CORRECTNESS-001: Set GPU_DEBUG=1 to enable layer-by-layer debug output
        static GPU_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let skip_debug = !*GPU_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        self.transformer_layer_workspace_inner(
            input,
            layer_idx,
            layer_weights,
            hidden_dim,
            intermediate_dim,
            epsilon,
            position,
            skip_debug,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn transformer_layer_workspace_inner(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &IndexedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        position: u32, // PAR-070: Explicit position for RoPE and KV cache
        skip_debug: bool,
    ) -> Result<(), GpuError> {
        // Verify workspace is initialized
        if !self.workspace.initialized {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-044: Workspace not initialized. Call init_workspace() first.".to_string(),
            ));
        }

        // Get dimension info
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // PAR-044: Get buffer pointers/lengths to avoid borrow conflicts
        // SAFETY: All workspace buffers are initialized (verified above) and remain valid
        let hidden_buf1_ptr = self
            .workspace
            .hidden_buf1
            .as_ref()
            .expect("hidden_buf1 must be initialized")
            .as_ptr();
        let hidden_buf1_len = self
            .workspace
            .hidden_buf1
            .as_ref()
            .expect("hidden_buf1 must be initialized")
            .len();
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .as_ptr();
        let hidden_buf2_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .len();
        // PAR-044 FIX: Use input_staging as scratch for residual1 to avoid read/write conflict
        // when input aliases hidden_buf2 (layers 1+)
        let input_staging_ptr = self
            .workspace
            .input_staging
            .as_ref()
            .expect("input_staging must be initialized")
            .as_ptr();
        let input_staging_len = self
            .workspace
            .input_staging
            .as_ref()
            .expect("input_staging must be initialized")
            .len();
        let q_buf_ptr = self
            .workspace
            .q_buf
            .as_ref()
            .expect("q_buf must be initialized")
            .as_ptr();
        let q_buf_len = self
            .workspace
            .q_buf
            .as_ref()
            .expect("q_buf must be initialized")
            .len();
        let k_buf_ptr = self
            .workspace
            .k_buf
            .as_ref()
            .expect("k_buf must be initialized")
            .as_ptr();
        let k_buf_len = self
            .workspace
            .k_buf
            .as_ref()
            .expect("k_buf must be initialized")
            .len();
        let v_buf_ptr = self
            .workspace
            .v_buf
            .as_ref()
            .expect("v_buf must be initialized")
            .as_ptr();
        let v_buf_len = self
            .workspace
            .v_buf
            .as_ref()
            .expect("v_buf must be initialized")
            .len();
        let ffn_gate_ptr = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .expect("ffn_gate_buf must be initialized")
            .as_ptr();
        let ffn_gate_len = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .expect("ffn_gate_buf must be initialized")
            .len();
        let ffn_up_ptr = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .expect("ffn_up_buf must be initialized")
            .as_ptr();
        let ffn_up_len = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .expect("ffn_up_buf must be initialized")
            .len();
        let ffn_act_ptr = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .expect("ffn_act_buf must be initialized")
            .as_ptr();
        let ffn_act_len = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .expect("ffn_act_buf must be initialized")
            .len();
        // PAR-051: Attention output workspace buffer
        let attn_out_ptr = self
            .workspace
            .attn_out_buf
            .as_ref()
            .expect("attn_out_buf must be initialized")
            .as_ptr();
        let attn_out_len = self
            .workspace
            .attn_out_buf
            .as_ref()
            .expect("attn_out_buf must be initialized")
            .len();

        // Create temporary non-owning buffer wrappers
        // These will be forgotten at the end to avoid freeing borrowed memory
        let hidden_buf1 =
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf1_ptr, hidden_buf1_len) };
        let hidden_buf2 =
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        let input_staging =
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { GpuBuffer::<f32>::from_raw_parts(input_staging_ptr, input_staging_len) };
        // PAR-060: Q/K buffers for RoPE application
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        // PAR-051: Attention output buffer (eliminates 28 allocations per token)
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let attn_out_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(attn_out_ptr, attn_out_len) };

        // PAR-073: Check if profiling is enabled (avoid overhead when disabled)
        let profiling = self.profiler.is_enabled();

        // 1. Pre-attention RMSNorm: input -> hidden_buf1 (normed)
        let timer_rmsnorm1 = if profiling {
            self.start_brick_timer("RmsNorm1")
        } else {
            None
        };
        self.rmsnorm_ptr_into(
            input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            &hidden_buf1,
            hidden_dim,
            epsilon,
        )?;
        if profiling {
            self.stop_brick_timer(timer_rmsnorm1, 1);
        }

        // PAR-058-DEBUG: Check after RMSNorm (skip during graph capture)
        if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3) {
            self.stream.synchronize()?;
            let mut rmsnorm_out = vec![0.0f32; hidden_buf1.len()];
            hidden_buf1.copy_to_host(&mut rmsnorm_out)?;
            let nan_count = rmsnorm_out.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                eprintln!(
                    "[PAR-058-L{}] RMSNorm output has {} NaN",
                    layer_idx, nan_count
                );
            } else {
                eprintln!(
                    "[PAR-058-L{}] RMSNorm OK, first 3: {:?}",
                    layer_idx,
                    &rmsnorm_out[..3.min(rmsnorm_out.len())]
                );
            }
        }

        // 2. Q/K/V projections using indexed pointers -> workspace buffers
        // PAR-058: Use correct kernel based on weight quantization type
        // Qwen 0.5B uses Q5_0 for Q/K weights, not Q4K
        let timer_qkv = if profiling {
            self.start_brick_timer("QKV")
        } else {
            None
        };
        match layer_weights.attn_q_qtype {
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    layer_weights.attn_q_ptr,
                    &hidden_buf1,
                    &q_buf,
                    q_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    layer_weights.attn_q_ptr,
                    &hidden_buf1,
                    &q_buf,
                    q_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_1 => {
                self.q4_1_gemv_into(
                    layer_weights.attn_q_ptr,
                    &hidden_buf1,
                    &q_buf,
                    q_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    layer_weights.attn_q_ptr,
                    &hidden_buf1,
                    &q_buf,
                    q_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(
                    layer_weights.attn_q_ptr,
                    &hidden_buf1,
                    &q_buf,
                    q_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(
                    layer_weights.attn_q_ptr,
                    &hidden_buf1,
                    &q_buf,
                    q_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4K => {
                // CORRECTNESS-011: Debug Q4K GEMV parameters
                if !skip_debug && layer_idx == 0 {
                    self.stream.synchronize()?;
                    let mut input_check = vec![0.0f32; hidden_buf1.len()];
                    hidden_buf1.copy_to_host(&mut input_check)?;
                    eprintln!(
                        "[CORRECTNESS-011-L0] Q4K GEMV params: n={}, k={}",
                        q_dim, hidden_dim
                    );
                    eprintln!(
                        "[CORRECTNESS-011-L0] Input (hidden_buf1): first 5 = {:?}",
                        &input_check[..5.min(input_check.len())]
                    );
                    eprintln!(
                        "[CORRECTNESS-011-L0] Weight ptr = {:#x}, len = {}",
                        layer_weights.attn_q_ptr, layer_weights.attn_q_len
                    );
                }
                self.q4k_gemv_into(
                    layer_weights.attn_q_ptr,
                    &hidden_buf1,
                    &q_buf,
                    q_dim,
                    hidden_dim,
                )?;
                // CORRECTNESS-011: Debug Q output
                if !skip_debug && layer_idx == 0 {
                    self.stream.synchronize()?;
                    let mut q_check = vec![0.0f32; q_buf.len()];
                    q_buf.copy_to_host(&mut q_check)?;
                    eprintln!(
                        "[CORRECTNESS-011-L0] Q output: first 5 = {:?}",
                        &q_check[..5.min(q_check.len())]
                    );
                }
            },
        }
        // GQA-DEBUG: Print K qtype for debugging
        if !skip_debug && layer_idx == 0 {
            eprintln!("[GQA-DEBUG-GPU-L0] K qtype = {:?}, ptr = {:#x}, len = {}",
                layer_weights.attn_k_qtype, layer_weights.attn_k_ptr, layer_weights.attn_k_len);
        }
        match layer_weights.attn_k_qtype {
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    layer_weights.attn_k_ptr,
                    &hidden_buf1,
                    &k_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    layer_weights.attn_k_ptr,
                    &hidden_buf1,
                    &k_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_1 => {
                self.q4_1_gemv_into(
                    layer_weights.attn_k_ptr,
                    &hidden_buf1,
                    &k_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    layer_weights.attn_k_ptr,
                    &hidden_buf1,
                    &k_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(
                    layer_weights.attn_k_ptr,
                    &hidden_buf1,
                    &k_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(
                    layer_weights.attn_k_ptr,
                    &hidden_buf1,
                    &k_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4K => {
                self.q4k_gemv_into(
                    layer_weights.attn_k_ptr,
                    &hidden_buf1,
                    &k_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
        }
        // PAR-058: Use correct kernel based on V weight quantization type
        match layer_weights.attn_v_qtype {
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    layer_weights.attn_v_ptr,
                    &hidden_buf1,
                    &v_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    layer_weights.attn_v_ptr,
                    &hidden_buf1,
                    &v_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_1 => {
                self.q4_1_gemv_into(
                    layer_weights.attn_v_ptr,
                    &hidden_buf1,
                    &v_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    layer_weights.attn_v_ptr,
                    &hidden_buf1,
                    &v_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(
                    layer_weights.attn_v_ptr,
                    &hidden_buf1,
                    &v_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(
                    layer_weights.attn_v_ptr,
                    &hidden_buf1,
                    &v_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4K => {
                self.q4k_gemv_into(
                    layer_weights.attn_v_ptr,
                    &hidden_buf1,
                    &v_buf,
                    kv_dim,
                    hidden_dim,
                )?;
            },
        }
        if profiling {
            self.stop_brick_timer(timer_qkv, 1);
        }

        // BIAS-FIX: Add QKV bias after GEMV (Qwen2.5 models have QKV bias)
        // Only add if bias exists (len > 0)
        if layer_weights.attn_q_bias_len > 0 {
            // Create non-owning buffer wrapper from device pointer
            // SAFETY: bias_ptr is valid device memory owned by bias_cache
            let q_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_q_bias_ptr,
                    layer_weights.attn_q_bias_len,
                )
            };

            // Add bias in-place: q_buf = q_buf + q_bias
            self.residual_add_into(&q_buf, &q_bias_buf, &q_buf, q_dim)?;

            // Prevent Drop from freeing borrowed memory
            std::mem::forget(q_bias_buf);

            // Debug log for layer 0, 4, 5
            if !skip_debug && (layer_idx == 0 || layer_idx == 4 || layer_idx == 5) {
                self.stream.synchronize()?;
                let mut q_check = vec![0.0f32; q_buf.len()];
                q_buf.copy_to_host(&mut q_check)?;
                eprintln!(
                    "[BIAS-FIX-L{}] Q after bias: first 5 = {:?}",
                    layer_idx,
                    &q_check[..5.min(q_check.len())]
                );
            }
        }
        if layer_weights.attn_k_bias_len > 0 {
            // GQA-DEBUG: Print K values BEFORE bias to isolate issue
            if !skip_debug && layer_idx == 0 {
                self.stream.synchronize()?;
                let mut k_pre = vec![0.0f32; k_buf.len()];
                k_buf.copy_to_host(&mut k_pre)?;
                eprintln!(
                    "[GQA-DEBUG-L0] K BEFORE bias: first 5 = {:?}",
                    &k_pre[..5.min(k_pre.len())]
                );
                // Also print the bias values
                let k_bias_buf_check = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        layer_weights.attn_k_bias_ptr,
                        layer_weights.attn_k_bias_len,
                    )
                };
                let mut k_bias_vals = vec![0.0f32; k_bias_buf_check.len()];
                k_bias_buf_check.copy_to_host(&mut k_bias_vals)?;
                eprintln!(
                    "[GQA-DEBUG-L0] K bias values: first 5 = {:?}",
                    &k_bias_vals[..5.min(k_bias_vals.len())]
                );
                std::mem::forget(k_bias_buf_check);
            }

            // SAFETY: Pointer and length from layer_weights validated at model load time
            let k_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_k_bias_ptr,
                    layer_weights.attn_k_bias_len,
                )
            };
            self.residual_add_into(&k_buf, &k_bias_buf, &k_buf, kv_dim)?;
            std::mem::forget(k_bias_buf);

            // Debug log for layer 0, 4, 5
            if !skip_debug && (layer_idx == 0 || layer_idx == 4 || layer_idx == 5) {
                self.stream.synchronize()?;
                let mut k_check = vec![0.0f32; k_buf.len()];
                k_buf.copy_to_host(&mut k_check)?;
                eprintln!(
                    "[BIAS-FIX-L{}] K after bias: first 5 = {:?}",
                    layer_idx,
                    &k_check[..5.min(k_check.len())]
                );
            }
        }
        if layer_weights.attn_v_bias_len > 0 {
            // SAFETY: Pointer and length from layer_weights validated at model load time
            let v_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_v_bias_ptr,
                    layer_weights.attn_v_bias_len,
                )
            };
            self.residual_add_into(&v_buf, &v_bias_buf, &v_buf, kv_dim)?;
            std::mem::forget(v_bias_buf);

            // Debug log for layer 0, 4, 5
            if !skip_debug && (layer_idx == 0 || layer_idx == 4 || layer_idx == 5) {
                self.stream.synchronize()?;
                let mut v_check = vec![0.0f32; v_buf.len()];
                v_buf.copy_to_host(&mut v_check)?;
                eprintln!(
                    "[BIAS-FIX-L{}] V after bias: first 5 = {:?}",
                    layer_idx,
                    &v_check[..5.min(v_check.len())]
                );
            }
        }

        // PAR-058-DEBUG: Check Q/K/V after projections (skip during graph capture)
        if !skip_debug
            && (layer_idx == 0
                || layer_idx == 1
                || layer_idx == 2
                || layer_idx == 3
                || layer_idx == 5)
        {
            self.stream.synchronize()?;
            // Print weight pointers
            eprintln!(
                "[PAR-058-L{}] Weight ptrs: Q={:#x}, K={:#x}, V={:#x}",
                layer_idx,
                layer_weights.attn_q_ptr,
                layer_weights.attn_k_ptr,
                layer_weights.attn_v_ptr
            );
            eprintln!(
                "[PAR-058-L{}] Weight lens: Q={}, K={}, V={}",
                layer_idx,
                layer_weights.attn_q_len,
                layer_weights.attn_k_len,
                layer_weights.attn_v_len
            );

            let mut q_out = vec![0.0f32; q_buf.len()];
            q_buf.copy_to_host(&mut q_out)?;
            let q_nan = q_out.iter().filter(|x| x.is_nan()).count();
            if q_nan > 0 {
                eprintln!("[PAR-058-L{}] Q has {} NaN", layer_idx, q_nan);
            } else {
                eprintln!(
                    "[PAR-058-L{}] Q OK, first 3: {:?}",
                    layer_idx,
                    &q_out[..3.min(q_out.len())]
                );
            }
            // Also check K values
            let mut k_out = vec![0.0f32; k_buf.len()];
            k_buf.copy_to_host(&mut k_out)?;
            let k_nan = k_out.iter().filter(|x| x.is_nan()).count();
            let k_max = k_out.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let k_min = k_out.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            eprintln!(
                "[PAR-058-L{}] K stats: nan={}, min={:.4}, max={:.4}, first 5: {:?}",
                layer_idx,
                k_nan,
                k_min,
                k_max,
                &k_out[..5.min(k_out.len())]
            );
            // Also check V values
            let mut v_out = vec![0.0f32; v_buf.len()];
            v_buf.copy_to_host(&mut v_out)?;
            let v_nan = v_out.iter().filter(|x| x.is_nan()).count();
            let v_max = v_out.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let v_min = v_out.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            eprintln!(
                "[PAR-058-L{}] V stats: nan={}, min={:.4}, max={:.4}, first 5: {:?}",
                layer_idx,
                v_nan,
                v_min,
                v_max,
                &v_out[..5.min(v_out.len())]
            );
        }

        // PAR-060: Apply RoPE to Q and K before attention using GPU kernel
        // This eliminates 28 GPU syncs + D2H/H2D copies per token
        // PAR-070: Use explicit position parameter instead of deriving from cache length
        let timer_rope = if profiling {
            self.start_brick_timer("RoPE")
        } else {
            None
        };
        {
            // Apply RoPE on GPU - Q has num_heads, K has num_kv_heads (GQA)
            let num_heads = self.kv_num_heads as u32;
            let num_kv_heads = self.kv_num_kv_heads as u32;
            let head_dim = self.kv_head_dim as u32;
            let theta = self.rope_theta;

            // Apply RoPE to Q and K (in-place)
            // PAR-061: Use indirect position for CUDA graph capture to avoid baking position
            if layer_idx == 0 && verbose() {
                eprintln!(
                    "[CORRECTNESS-010] RoPE: skip_debug={}, position_buf={}, using {}",
                    skip_debug,
                    self.position_buf.is_some(),
                    if skip_debug && self.position_buf.is_some() {
                        "indirect"
                    } else {
                        "direct"
                    }
                );
            }
            // CORRECTNESS-011: Use NEOX RoPE style for rope_type == 2 (Qwen2.5, etc.)
            if skip_debug && self.position_buf.is_some() {
                // Graph capture mode: read position from device memory (updated before replay)
                // Clone the buffer pointer to avoid borrow conflict with &mut self
                let pos_buf_ptr = self
                    .position_buf
                    .as_ref()
                    .expect("position_buf must be initialized")
                    .as_ptr();
                let pos_buf_len = self
                    .position_buf
                    .as_ref()
                    .expect("position_buf must be initialized")
                    .len();
                // SAFETY: Memory safety ensured by bounds checking and alignment
                // SAFETY: Pointer valid from allocation, length verified, used within scope
                let pos_buf = unsafe { GpuBuffer::<u32>::from_raw_parts(pos_buf_ptr, pos_buf_len) };
                if self.rope_type == 2 {
                    // NEOX style: split halves (i, i + half_dim)
                    self.rope_neox_indirect_into(
                        &q_buf, &q_buf, &pos_buf, num_heads, head_dim, theta,
                    )?;
                    self.rope_neox_indirect_into(
                        &k_buf,
                        &k_buf,
                        &pos_buf,
                        num_kv_heads,
                        head_dim,
                        theta,
                    )?;
                } else {
                    // NORM style: adjacent pairs (2*i, 2*i+1)
                    self.rope_indirect_into(&q_buf, &q_buf, &pos_buf, num_heads, head_dim, theta)?;
                    self.rope_indirect_into(
                        &k_buf,
                        &k_buf,
                        &pos_buf,
                        num_kv_heads,
                        head_dim,
                        theta,
                    )?;
                }
                std::mem::forget(pos_buf); // Don't drop - it's a view into self.position_buf
            } else {
                // Normal mode: use direct position value
                if self.rope_type == 2 {
                    // NEOX style: split halves (i, i + half_dim)
                    self.rope_neox_into(&q_buf, &q_buf, position, num_heads, head_dim, theta)?;
                    self.rope_neox_into(&k_buf, &k_buf, position, num_kv_heads, head_dim, theta)?;
                } else {
                    // NORM style: adjacent pairs (2*i, 2*i+1)
                    self.rope_into(&q_buf, &q_buf, position, num_heads, head_dim, theta)?;
                    self.rope_into(&k_buf, &k_buf, position, num_kv_heads, head_dim, theta)?;
                }
            }

            if !skip_debug && (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 3)
            {
                // Debug: download and print (only for layer 0/2, skip during graph capture)
                self.stream.synchronize()?;
                let mut q_host = vec![0.0f32; q_buf.len()];
                let mut k_host = vec![0.0f32; k_buf.len()];
                q_buf.copy_to_host(&mut q_host)?;
                k_buf.copy_to_host(&mut k_host)?;
                eprintln!("[PAR-060-L{}] Applied GPU RoPE at position {}, theta={}, Q first 3: {:?}, K first 3: {:?}",
                    layer_idx, position, theta, &q_host[..3.min(q_host.len())], &k_host[..3.min(k_host.len())]);
            }
        }
        if profiling {
            self.stop_brick_timer(timer_rope, 1);
        }

        // 3. PAR-051: Incremental attention into pre-allocated workspace buffer
        // Eliminates 28 GPU allocations per token
        // PAR-054-FIX: Use capture-safe version during graph capture to skip debug sync
        let timer_attn = if profiling {
            self.start_brick_timer("Attention")
        } else {
            None
        };
        let _seq_len = if skip_debug {
            self.incremental_attention_into_for_capture(
                layer_idx,
                &q_buf,
                &k_buf,
                &v_buf,
                &attn_out_buf,
            )?
        } else {
            self.incremental_attention_into(layer_idx, &q_buf, &k_buf, &v_buf, &attn_out_buf)?
        };
        if profiling {
            self.stop_brick_timer(timer_attn, 1);
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
                // Show first few NaN indices
                eprintln!(
                    "[PAR-058-L{}] First 10 NaN indices: {:?}",
                    layer_idx,
                    &nan_indices[..10.min(nan_indices.len())]
                );
                // Show first OK value
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
        // PAR-058: Use correct kernel based on output projection quantization type
        let timer_oproj = if profiling {
            self.start_brick_timer("OProj")
        } else {
            None
        };
        match layer_weights.attn_output_qtype {
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    layer_weights.attn_output_ptr,
                    &attn_out_buf,
                    &hidden_buf1,
                    hidden_dim,
                    q_dim,
                )?;
            },
            // PAR-058: Add Q5_0 support for output projection (Qwen 0.5B)
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    layer_weights.attn_output_ptr,
                    &attn_out_buf,
                    &hidden_buf1,
                    hidden_dim,
                    q_dim,
                )?;
            },
            _ => {
                self.q4k_gemv_into(
                    layer_weights.attn_output_ptr,
                    &attn_out_buf,
                    &hidden_buf1,
                    hidden_dim,
                    q_dim,
                )?;
            },
        }
        if profiling {
            self.stop_brick_timer(timer_oproj, 1);
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
        self.residual_add_into(input, &hidden_buf1, &input_staging, hidden_dim)?;
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

        // 6. Pre-FFN RMSNorm: residual1 (input_staging) -> hidden_buf1 (ffn_normed)
        let timer_rmsnorm2 = if profiling {
            self.start_brick_timer("RmsNorm2")
        } else {
            None
        };
        self.rmsnorm_ptr_into(
            &input_staging,
            layer_weights.ffn_norm_ptr,
            layer_weights.ffn_norm_len,
            &hidden_buf1,
            hidden_dim,
            epsilon,
        )?;
        if profiling {
            self.stop_brick_timer(timer_rmsnorm2, 1);
        }

        // 7. FFN gate/up projections -> workspace buffers
        // PAR-077: Fused kernel BLOCKED - 3x slower due to shared memory + barrier overhead
        // Root cause: Input is 6KB, weights are 15MB - weights dominate by 2500x
        // L2 cache naturally serves input reuse between gate/up kernels
        let timer_ffn_gate_up = if profiling {
            self.start_brick_timer("FFNGateUp")
        } else {
            None
        };
        match layer_weights.ffn_gate_qtype {
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    layer_weights.ffn_gate_ptr,
                    &hidden_buf1,
                    &ffn_gate_buf,
                    intermediate_dim,
                    hidden_dim,
                )?;
            },
            // PAR-058: Add Q5_0 support for FFN gate (Qwen 0.5B)
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    layer_weights.ffn_gate_ptr,
                    &hidden_buf1,
                    &ffn_gate_buf,
                    intermediate_dim,
                    hidden_dim,
                )?;
            },
            _ => {
                self.q4k_gemv_into(
                    layer_weights.ffn_gate_ptr,
                    &hidden_buf1,
                    &ffn_gate_buf,
                    intermediate_dim,
                    hidden_dim,
                )?;
            },
        }
        match layer_weights.ffn_up_qtype {
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    layer_weights.ffn_up_ptr,
                    &hidden_buf1,
                    &ffn_up_buf,
                    intermediate_dim,
                    hidden_dim,
                )?;
            },
            // PAR-058: Add Q5_0 support for FFN up (Qwen 0.5B)
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    layer_weights.ffn_up_ptr,
                    &hidden_buf1,
                    &ffn_up_buf,
                    intermediate_dim,
                    hidden_dim,
                )?;
            },
            _ => {
                self.q4k_gemv_into(
                    layer_weights.ffn_up_ptr,
                    &hidden_buf1,
                    &ffn_up_buf,
                    intermediate_dim,
                    hidden_dim,
                )?;
            },
        }
        if profiling {
            self.stop_brick_timer(timer_ffn_gate_up, 1);
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
            self.start_brick_timer("SwiGLU")
        } else {
            None
        };
        self.fused_swiglu_into(&ffn_gate_buf, &ffn_up_buf, &ffn_act_buf, intermediate_dim)?;
        if profiling {
            self.stop_brick_timer(timer_swiglu, 1);
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
            // Expected sizes: Q4K=144/sb, Q5K=176/sb, Q6K=210/sb, Q8_0=34/32elem
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
        // For some dimensions, Q4_0 and Q4K have IDENTICAL byte sizes (e.g., 896×4864)
        // In such cases, TRUST the metadata qtype rather than guessing wrong
        let metadata_qtype = layer_weights.ffn_down_qtype;
        let metadata_matches = metadata_qtype.matches_size(
            layer_weights.ffn_down_len,
            hidden_dim as usize,
            intermediate_dim as usize,
        );
        let ffn_down_qtype = if metadata_matches {
            // Metadata qtype produces correct size - trust it
            metadata_qtype
        } else {
            // Metadata qtype wrong, try size-based detection
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

        let timer_ffn_down = if profiling {
            self.start_brick_timer("FFNDown")
        } else {
            None
        };
        match ffn_down_qtype {
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &ffn_act_buf,
                    &hidden_buf1,
                    hidden_dim,
                    intermediate_dim,
                )?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &ffn_act_buf,
                    &hidden_buf1,
                    hidden_dim,
                    intermediate_dim,
                )?;
            },
            WeightQuantType::Q4_1 => {
                // PAR-058: Q4_1 for Qwen2.5-0.5B FFN down (size-based detection)
                self.q4_1_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &ffn_act_buf,
                    &hidden_buf1,
                    hidden_dim,
                    intermediate_dim,
                )?;
            },
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &ffn_act_buf,
                    &hidden_buf1,
                    hidden_dim,
                    intermediate_dim,
                )?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &ffn_act_buf,
                    &hidden_buf1,
                    hidden_dim,
                    intermediate_dim,
                )?;
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &ffn_act_buf,
                    &hidden_buf1,
                    hidden_dim,
                    intermediate_dim,
                )?;
            },
            WeightQuantType::Q4K => {
                // CORRECTNESS-002: Debug first super-block of Layer 2 FFN down weights
                if !skip_debug && layer_idx == 2 {
                    self.stream.synchronize()?;
                    eprintln!(
                        "[CORRECTNESS-002] L2 FFN down: ptr={:#x}, n={}, k={}",
                        layer_weights.ffn_down_ptr, hidden_dim, intermediate_dim
                    );
                    // Read d and dmin via GpuBuffer
                    let mut host_data = vec![0u8; 144];
                    let debug_buf =
                        // SAFETY: Memory safety ensured by bounds checking and alignment
                        unsafe { GpuBuffer::<u8>::from_raw_parts(layer_weights.ffn_down_ptr, 144) };
                    debug_buf.copy_to_host(&mut host_data)?;
                    std::mem::forget(debug_buf); // Don't free the borrowed memory
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
                        &host_data[..20]
                    );
                }
                self.q4k_gemv_into(
                    layer_weights.ffn_down_ptr,
                    &ffn_act_buf,
                    &hidden_buf1,
                    hidden_dim,
                    intermediate_dim,
                )?;
            },
        }
        if profiling {
            self.stop_brick_timer(timer_ffn_down, 1);
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
        self.residual_add_into(&input_staging, &hidden_buf1, &hidden_buf2, hidden_dim)?;
        if profiling {
            self.stop_brick_timer(timer_res2, 1);
        }

        // PAR-058-DEBUG: Check layer output - check first 10 layers to find where NaN starts (skip during graph capture)
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

        // Prevent Drop from freeing the borrowed memory
        std::mem::forget(hidden_buf1);
        std::mem::forget(hidden_buf2);
        std::mem::forget(input_staging);
        std::mem::forget(q_buf);
        std::mem::forget(k_buf);
        std::mem::forget(v_buf);
        std::mem::forget(attn_out_buf); // PAR-051
        std::mem::forget(ffn_gate_buf);
        std::mem::forget(ffn_up_buf);
        std::mem::forget(ffn_act_buf);

        // Output is now in hidden_buf2
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Tests for transformer_layer_indexed
    // ========================================================================

    #[test]
    fn test_transformer_layer_indexed_missing_kv_cache() {
        let Some(mut exec) = create_executor() else { return; };

        // Create dummy IndexedLayerWeights using Default
        let layer_weights = IndexedLayerWeights::default();

        let input: Vec<f32> = vec![0.1; 256];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        // Will fail due to missing KV cache setup or zero pointers
        let result = exec.transformer_layer_indexed(
            &input_buf,
            0,              // layer_idx
            &layer_weights,
            256,            // hidden_dim
            1024,           // intermediate_dim
            1e-5,           // epsilon
        );

        // Expected to fail - KV cache not initialized or zero pointers
        assert!(result.is_err());
    }

    #[test]
    fn test_indexed_layer_weights_default() {
        // Test that default creates valid zeroed structure
        let weights = IndexedLayerWeights::default();
        assert_eq!(weights.attn_norm_ptr, 0);
        assert_eq!(weights.attn_norm_len, 0);
        assert_eq!(weights.attn_q_ptr, 0);
        assert!(matches!(weights.attn_v_qtype, WeightQuantType::Q4K));
    }

    #[test]
    fn test_weight_quant_type_variants() {
        // Test WeightQuantType::Q6K path exists in transformer_layer_indexed
        // The match arm for Q6K uses q6k_gemv_indexed_async
        assert!(matches!(WeightQuantType::Q6K, WeightQuantType::Q6K));
        assert!(matches!(WeightQuantType::Q4K, WeightQuantType::Q4K));
        assert!(matches!(WeightQuantType::Q5K, WeightQuantType::Q5K));
    }
}
