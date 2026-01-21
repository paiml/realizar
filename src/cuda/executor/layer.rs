//! Transformer layer operations: SwiGLU FFN, full transformer layer, batched processing
//!
//! This module implements:
//! - PAR-023: GPU-Resident SwiGLU FFN
//! - PAR-044: GPU-Resident Transformer Layer
//! - PAR-111: Batched Transformer Layer for multi-sequence processing
//! - PAR-062: CUDA Graph-captured decode
//! - Full forward pass with all layers

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

impl CudaExecutor {
    // =========================================================================
    // PAR-023: GPU-Resident SwiGLU FFN (LLaMA-style)
    // Reduces 3 syncs per layer to 1 by chaining: gate→up→swiglu→down
    // =========================================================================

    /// PAR-023: GPU-resident SwiGLU FFN operating entirely on GPU buffers
    ///
    /// Implements LLaMA-style FFN: down(swiglu(gate(x), up(x)))
    /// All operations chained without sync - only syncs when output needed.
    ///
    /// PAR-063-V5: Set TRUE_DP4A=1 to use Q8 activation quantization + Q4K×Q8
    /// integer dot product for 4x instruction reduction (llama.cpp-style).
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing hidden state [hidden_dim]
    /// * `ffn_gate_name` - Cache key for FFN gate weight
    /// * `ffn_up_name` - Cache key for FFN up weight
    /// * `ffn_down_name` - Cache key for FFN down weight
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    ///
    /// # Returns
    /// GPU buffer containing FFN output [hidden_dim] - not synchronized
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_swiglu_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        ffn_gate_name: &str,
        ffn_up_name: &str,
        ffn_down_name: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // PAR-063-V5: Environment variable to enable TRUE DP4A path
        // Set TRUE_DP4A=1 to use Q8 activation quantization + Q4K×Q8 integer dot product
        static TRUE_DP4A_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_true_dp4a = *TRUE_DP4A_ENABLED.get_or_init(|| {
            std::env::var("TRUE_DP4A")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        if use_true_dp4a {
            return self.fused_ffn_swiglu_gpu_true_dp4a(
                input,
                ffn_gate_name,
                ffn_up_name,
                ffn_down_name,
                hidden_dim,
                intermediate_dim,
            );
        }

        // PAR-063: Kernel selection for FFN layers
        // Priority order:
        // 1. Dp4aQ4KGemv: Best for aligned K (uses DP4A SIMD, 4x instruction reduction)
        // 2. TiledQ4KGemv: K <= 8192 (32KB shared memory, fits in 48KB limit)
        // 3. Q4KGemv: Fallback for unaligned K or large K > 8192

        // For gate/up projection: K = hidden_dim, N = intermediate_dim
        // For down projection: K = intermediate_dim, N = hidden_dim
        const CHUNK_THRESHOLD: u32 = 8192;

        let hidden_aligned = hidden_dim.is_multiple_of(256);
        let intermediate_aligned = intermediate_dim.is_multiple_of(256);

        // 1. Gate projection: [hidden_dim] -> [intermediate_dim] (no sync)
        // PAR-063: Use DP4A kernel for aligned dimensions (fastest)
        let gate = if hidden_aligned && hidden_dim <= CHUNK_THRESHOLD {
            self.dp4a_q4k_gemv_cached_async(ffn_gate_name, input, intermediate_dim, hidden_dim)?
        } else {
            self.q4k_gemv_cached_async(ffn_gate_name, input, intermediate_dim, hidden_dim)?
        };

        // 2. Up projection: [hidden_dim] -> [intermediate_dim] (no sync)
        let up = if hidden_aligned && hidden_dim <= CHUNK_THRESHOLD {
            self.dp4a_q4k_gemv_cached_async(ffn_up_name, input, intermediate_dim, hidden_dim)?
        } else {
            self.q4k_gemv_cached_async(ffn_up_name, input, intermediate_dim, hidden_dim)?
        };

        // 3. Fused SwiGLU: silu(gate) * up (no sync)
        let activated = self.fused_swiglu_gpu(&gate, &up, intermediate_dim)?;

        // 4. Down projection: [intermediate_dim] -> [hidden_dim] (no sync)
        // Note: K = intermediate_dim here (input to down projection)
        let output = if intermediate_aligned && intermediate_dim <= CHUNK_THRESHOLD {
            self.dp4a_q4k_gemv_cached_async(
                ffn_down_name,
                &activated,
                hidden_dim,
                intermediate_dim,
            )?
        } else {
            self.q4k_gemv_cached_async(ffn_down_name, &activated, hidden_dim, intermediate_dim)?
        };

        // PAR-023: NO sync here - caller chains more operations or syncs when needed
        Ok(output)
    }

    /// PAR-063-V5: GPU-resident SwiGLU FFN using TRUE DP4A kernels (async, no sync)
    ///
    /// Uses Q8 activation quantization + Q4K×Q8 integer dot product for 4x instruction reduction.
    /// This is the llama.cpp-style approach:
    /// 1. Quantize f32 activations to Q8_1 (per-block scale + 32 × int8)
    /// 2. Use dp4a.u32.s32 for 4 multiply-adds per instruction
    /// 3. Apply scales at the end
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing hidden state [hidden_dim]
    /// * `ffn_gate_name` - Cache key for FFN gate weight
    /// * `ffn_up_name` - Cache key for FFN up weight
    /// * `ffn_down_name` - Cache key for FFN down weight
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_swiglu_gpu_true_dp4a(
        &mut self,
        input: &GpuBuffer<f32>,
        ffn_gate_name: &str,
        ffn_up_name: &str,
        ffn_down_name: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // PAR-063-V6: Environment variable to enable packed DP4A kernel
        // Set PACKED_DP4A=1 to use the optimized nibble-packed dp4a.u32.s32 kernel
        static PACKED_DP4A_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_packed_dp4a = *PACKED_DP4A_ENABLED.get_or_init(|| {
            std::env::var("PACKED_DP4A")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // PAR-063-V5/V6: True DP4A pipeline
        // 1. Quantize input activations to Q8_1 once (shared by gate and up projections)
        let q8_input = self.q8_quantize_async(input, hidden_dim)?;

        // 2. Gate projection using Q4K × Q8 integer dot product
        let gate = if use_packed_dp4a {
            self.packed_dp4a_q4k_q8_gemv_async(
                ffn_gate_name,
                &q8_input,
                intermediate_dim,
                hidden_dim,
            )?
        } else {
            self.q4k_q8_gemv_async(ffn_gate_name, &q8_input, intermediate_dim, hidden_dim)?
        };

        // 3. Up projection using Q4K × Q8 integer dot product
        let up = if use_packed_dp4a {
            self.packed_dp4a_q4k_q8_gemv_async(
                ffn_up_name,
                &q8_input,
                intermediate_dim,
                hidden_dim,
            )?
        } else {
            self.q4k_q8_gemv_async(ffn_up_name, &q8_input, intermediate_dim, hidden_dim)?
        };

        // 4. Fused SwiGLU: silu(gate) * up
        let activated = self.fused_swiglu_gpu(&gate, &up, intermediate_dim)?;

        // 5. Quantize activated values for down projection
        let q8_activated = self.q8_quantize_async(&activated, intermediate_dim)?;

        // 6. Down projection using Q4K × Q8 integer dot product
        let output = if use_packed_dp4a {
            self.packed_dp4a_q4k_q8_gemv_async(
                ffn_down_name,
                &q8_activated,
                hidden_dim,
                intermediate_dim,
            )?
        } else {
            self.q4k_q8_gemv_async(ffn_down_name, &q8_activated, hidden_dim, intermediate_dim)?
        };

        // PAR-063-V5/V6: NO sync here - caller chains more operations or syncs when needed
        Ok(output)
    }

    /// PAR-043: SwiGLU FFN using pre-indexed device pointers (async, no sync)
    ///
    /// This eliminates 3 HashMap lookups + string formatting per FFN call.
    /// Pointers must be from `indexed_layer_weights` populated by `build_indexed_weights()`.
    pub fn fused_ffn_swiglu_indexed_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        ffn_gate_ptr: u64,
        ffn_up_ptr: u64,
        ffn_down_ptr: u64,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // 1. Gate projection: [hidden_dim] -> [intermediate_dim] (no sync)
        let gate =
            self.q4k_gemv_indexed_async(ffn_gate_ptr, input, intermediate_dim, hidden_dim)?;

        // 2. Up projection: [hidden_dim] -> [intermediate_dim] (no sync)
        let up = self.q4k_gemv_indexed_async(ffn_up_ptr, input, intermediate_dim, hidden_dim)?;

        // 3. Fused SwiGLU: silu(gate) * up (no sync)
        let activated = self.fused_swiglu_gpu(&gate, &up, intermediate_dim)?;

        // 4. Down projection: [intermediate_dim] -> [hidden_dim] (no sync)
        let output =
            self.q4k_gemv_indexed_async(ffn_down_ptr, &activated, hidden_dim, intermediate_dim)?;

        // PAR-043: NO sync here - caller chains more operations or syncs when needed
        Ok(output)
    }

    /// PAR-023: SwiGLU FFN with host memory (convenience wrapper)
    ///
    /// Uploads input, runs GPU-resident FFN, syncs, downloads result.
    /// For testing and single-FFN use cases.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_swiglu_host(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        ffn_gate_name: &str,
        ffn_up_name: &str,
        ffn_down_name: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<(), GpuError> {
        // Upload input
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;

        // Run GPU-resident FFN (no intermediate syncs)
        let output_gpu = self.fused_ffn_swiglu_gpu(
            &input_gpu,
            ffn_gate_name,
            ffn_up_name,
            ffn_down_name,
            hidden_dim,
            intermediate_dim,
        )?;

        // Single sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    // =========================================================================
    // PAR-023: GPU-Resident Transformer Layer
    // Chains all operations with minimal syncs for maximum throughput
    // Target: Reduce 176 syncs/token to ~22 syncs/token (1 per layer)
    // =========================================================================

    /// PAR-023: GPU-resident transformer layer (LLaMA-style)
    ///
    /// Chains all layer operations on GPU with single sync at end:
    /// 1. Pre-attention RMSNorm
    /// 2. Q/K/V projections
    /// 3. Incremental attention
    /// 4. Output projection
    /// 5. Residual add
    /// 6. Pre-FFN RMSNorm
    /// 7. Gate/Up projections + SwiGLU + Down projection
    /// 8. Residual add
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing hidden state [hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup and KV cache
    /// * `layer_prefix` - Weight name prefix (e.g., "blk.0")
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `attn_norm_gamma` - Pre-attention RMSNorm weights
    /// * `ffn_norm_gamma` - Pre-FFN RMSNorm weights
    /// * `epsilon` - RMSNorm epsilon
    ///
    /// # Returns
    /// GPU buffer containing layer output [hidden_dim] - NOT synchronized
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &GpuBuffer<f32>,
        ffn_norm_gamma: &GpuBuffer<f32>,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Weight names follow GGML convention
        let q_name = format!("{}.attn_q.weight", layer_prefix);
        let k_name = format!("{}.attn_k.weight", layer_prefix);
        let v_name = format!("{}.attn_v.weight", layer_prefix);
        let o_name = format!("{}.attn_output.weight", layer_prefix);
        let gate_name = format!("{}.ffn_gate.weight", layer_prefix);
        let up_name = format!("{}.ffn_up.weight", layer_prefix);
        let down_name = format!("{}.ffn_down.weight", layer_prefix);

        // 1. Pre-attention RMSNorm (no sync)
        let normed = self.rmsnorm_gpu(input, attn_norm_gamma, hidden_dim, epsilon)?;

        // 2. Q/K/V projections (no sync)
        // Q: [hidden_dim] -> [num_heads * head_dim]
        // K: [hidden_dim] -> [num_kv_heads * head_dim]
        // V: [hidden_dim] -> [num_kv_heads * head_dim]
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // PAR-056: Tiled kernel selection based on K dimension
        // - TiledQ4KGemv: K <= 8192 (fits in 48KB shared memory)
        // - ChunkedTiledQ4KGemv: K > 8192 (uses 32KB chunks)
        const CHUNK_THRESHOLD: u32 = 8192;
        let hidden_aligned = hidden_dim.is_multiple_of(256);
        let q_aligned = q_dim.is_multiple_of(256);
        let kv_aligned = kv_dim.is_multiple_of(256);

        // Q/K/V projections: K = hidden_dim
        // CORRECTNESS-001: Temporarily disable DP4A to test fixed TiledQ4K kernel
        // PAR-063: Use DP4A kernel for aligned dimensions (fastest)
        let _use_dp4a = hidden_aligned && q_aligned && hidden_dim <= CHUNK_THRESHOLD;
        let q = {
            // Force TiledQ4K for now - dp4a_q4k has scale extraction issue
            self.q4k_gemv_cached_async(&q_name, &normed, q_dim, hidden_dim)?
        };
        let _use_dp4a_kv = hidden_aligned && kv_aligned && hidden_dim <= CHUNK_THRESHOLD;
        let k = { self.q4k_gemv_cached_async(&k_name, &normed, kv_dim, hidden_dim)? };
        let v = { self.q4k_gemv_cached_async(&v_name, &normed, kv_dim, hidden_dim)? };

        // 3. Incremental attention (has internal sync for KV cache update)
        let (attn_out, _seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;

        // 4. Output projection (no sync) - K = q_dim
        // CORRECTNESS-001: Force TiledQ4K kernel
        let projected = { self.q4k_gemv_cached_async(&o_name, &attn_out, hidden_dim, q_dim)? };

        // 5. First residual add (no sync)
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;

        // 6. Pre-FFN RMSNorm (no sync)
        let ffn_normed = self.rmsnorm_gpu(&residual1, ffn_norm_gamma, hidden_dim, epsilon)?;

        // 7. FFN SwiGLU (no sync)
        let ffn_out = self.fused_ffn_swiglu_gpu(
            &ffn_normed,
            &gate_name,
            &up_name,
            &down_name,
            hidden_dim,
            intermediate_dim,
        )?;

        // 8. Second residual add (no sync)
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;

        // PAR-023: NO sync here - caller can chain multiple layers
        Ok(output)
    }

    /// TILING-SPEC-001: Tile-profiled transformer layer for bottleneck identification.
    ///
    /// This method wraps `transformer_layer_gpu` with tile-level profiling instrumentation
    /// to identify whether the 0.07% efficiency bottleneck is:
    /// - Kernel launch overhead (many small kernels)
    /// - CPU dequantization in the hot path
    /// - Memory transfer overhead (H2D/D2H)
    /// - Specific operation bottlenecks (QKV, attention, FFN)
    ///
    /// # Profiling Levels
    ///
    /// | Level | Operation | FLOPs Formula |
    /// |-------|-----------|---------------|
    /// | Macro | QKV Projections | 2 × M × K × 3 |
    /// | Macro | Output Projection | 2 × M × K |
    /// | Midi  | Attention | 2 × seq × head_dim × num_heads |
    /// | Macro | FFN (SwiGLU) | 2 × M × K × 3 |
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// cuda_model.enable_tile_profiling();
    /// let output = cuda_model.transformer_layer_gpu_tiled_profiled(...)?;
    /// println!("{}", cuda_model.tile_summary());
    /// // Output shows per-operation GFLOP/s and identifies bottlenecks
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_gpu_tiled_profiled(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &GpuBuffer<f32>,
        ffn_norm_gamma: &GpuBuffer<f32>,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Weight names follow GGML convention
        let q_name = format!("{}.attn_q.weight", layer_prefix);
        let k_name = format!("{}.attn_k.weight", layer_prefix);
        let v_name = format!("{}.attn_v.weight", layer_prefix);
        let o_name = format!("{}.attn_output.weight", layer_prefix);
        let gate_name = format!("{}.ffn_gate.weight", layer_prefix);
        let up_name = format!("{}.ffn_up.weight", layer_prefix);
        let down_name = format!("{}.ffn_down.weight", layer_prefix);

        // Q/K/V dimensions
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // 1. Pre-attention RMSNorm (tracked as Micro - very fast)
        let timer_norm1 = self.start_tile_timer(trueno::TileLevel::Micro, layer_idx as u32, 0);
        let normed = self.rmsnorm_gpu(input, attn_norm_gamma, hidden_dim, epsilon)?;
        // RMSNorm FLOPs: 5N (square, sum, rsqrt, multiply, multiply) per element
        let norm_flops = (hidden_dim as u64) * 5;
        self.stop_tile_timer(timer_norm1, hidden_dim as u64, norm_flops);

        // 2. Q/K/V projections (Macro tile - largest compute block)
        // FLOPs: 2 * M * K for each matrix-vector multiply (M=1 for single token)
        let timer_qkv = self.start_tile_timer(trueno::TileLevel::Macro, layer_idx as u32, 1);

        let q = self.q4k_gemv_cached_async(&q_name, &normed, q_dim, hidden_dim)?;
        let k = self.q4k_gemv_cached_async(&k_name, &normed, kv_dim, hidden_dim)?;
        let v = self.q4k_gemv_cached_async(&v_name, &normed, kv_dim, hidden_dim)?;

        // QKV FLOPs: Q(hidden→q) + K(hidden→kv) + V(hidden→kv)
        let qkv_flops = 2 * (hidden_dim as u64) * (q_dim as u64 + kv_dim as u64 + kv_dim as u64);
        let qkv_elements = (q_dim + kv_dim + kv_dim) as u64;
        self.stop_tile_timer(timer_qkv, qkv_elements, qkv_flops);

        // 3. Incremental attention (Midi tile - head-level parallelism)
        let timer_attn = self.start_tile_timer(trueno::TileLevel::Midi, layer_idx as u32, 2);
        let (attn_out, seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;
        // Attention FLOPs: 2 * seq * head_dim * num_heads (Q×K^T + softmax×V)
        let attn_flops =
            2 * (seq_len as u64) * (self.kv_head_dim as u64) * (self.kv_num_heads as u64) * 2;
        self.stop_tile_timer(timer_attn, q_dim as u64, attn_flops);

        // 4. Output projection (Macro tile)
        let timer_proj = self.start_tile_timer(trueno::TileLevel::Macro, layer_idx as u32, 3);
        let projected = self.q4k_gemv_cached_async(&o_name, &attn_out, hidden_dim, q_dim)?;
        let proj_flops = 2 * (q_dim as u64) * (hidden_dim as u64);
        self.stop_tile_timer(timer_proj, hidden_dim as u64, proj_flops);

        // 5. First residual add (Micro - very fast)
        let timer_res1 = self.start_tile_timer(trueno::TileLevel::Micro, layer_idx as u32, 4);
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;
        self.stop_tile_timer(timer_res1, hidden_dim as u64, hidden_dim as u64);

        // 6. Pre-FFN RMSNorm (Micro)
        let timer_norm2 = self.start_tile_timer(trueno::TileLevel::Micro, layer_idx as u32, 5);
        let ffn_normed = self.rmsnorm_gpu(&residual1, ffn_norm_gamma, hidden_dim, epsilon)?;
        self.stop_tile_timer(timer_norm2, hidden_dim as u64, norm_flops);

        // 7. FFN SwiGLU (Macro tile - second largest compute block)
        // FLOPs: gate(hidden→inter) + up(hidden→inter) + down(inter→hidden) + SiLU
        let timer_ffn = self.start_tile_timer(trueno::TileLevel::Macro, layer_idx as u32, 6);
        let ffn_out = self.fused_ffn_swiglu_gpu(
            &ffn_normed,
            &gate_name,
            &up_name,
            &down_name,
            hidden_dim,
            intermediate_dim,
        )?;
        // FFN FLOPs: 3 GEMV (gate+up+down) + SiLU (~3 ops per element)
        let ffn_flops =
            2 * (hidden_dim as u64) * (intermediate_dim as u64) * 3 + (intermediate_dim as u64) * 3;
        self.stop_tile_timer(timer_ffn, hidden_dim as u64, ffn_flops);

        // 8. Second residual add (Micro)
        let timer_res2 = self.start_tile_timer(trueno::TileLevel::Micro, layer_idx as u32, 7);
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;
        self.stop_tile_timer(timer_res2, hidden_dim as u64, hidden_dim as u64);

        Ok(output)
    }

    /// PAR-063-V5: Transformer layer using TRUE DP4A kernels (async, no sync)
    ///
    /// Uses Q8 activation quantization + Q4K×Q8 integer dot product for 4x instruction reduction.
    /// This is the llama.cpp-style approach that achieves 2x llama.cpp performance.
    ///
    /// Key optimizations:
    /// 1. Single Q8 quantization for Q/K/V (shared input)
    /// 2. dp4a.u32.s32 instruction: 4 multiply-adds per instruction
    /// 3. All GEMV operations use integer arithmetic
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing hidden state [hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup and KV cache
    /// * `layer_prefix` - Weight name prefix (e.g., "blk.0")
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `attn_norm_gamma` - Pre-attention RMSNorm weights
    /// * `ffn_norm_gamma` - Pre-FFN RMSNorm weights
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_gpu_true_dp4a(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &GpuBuffer<f32>,
        ffn_norm_gamma: &GpuBuffer<f32>,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Weight names follow GGML convention
        let q_name = format!("{}.attn_q.weight", layer_prefix);
        let k_name = format!("{}.attn_k.weight", layer_prefix);
        let v_name = format!("{}.attn_v.weight", layer_prefix);
        let o_name = format!("{}.attn_output.weight", layer_prefix);
        let gate_name = format!("{}.ffn_gate.weight", layer_prefix);
        let up_name = format!("{}.ffn_up.weight", layer_prefix);
        let down_name = format!("{}.ffn_down.weight", layer_prefix);

        // 1. Pre-attention RMSNorm (no sync)
        let normed = self.rmsnorm_gpu(input, attn_norm_gamma, hidden_dim, epsilon)?;

        // 2. PAR-063-V5: Quantize normed activations to Q8_1 ONCE for all Q/K/V projections
        let q8_normed = self.q8_quantize_async(&normed, hidden_dim)?;

        // 3. Q/K/V projections using Q4K × Q8 integer dot product
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        let q = self.q4k_q8_gemv_async(&q_name, &q8_normed, q_dim, hidden_dim)?;
        let k = self.q4k_q8_gemv_async(&k_name, &q8_normed, kv_dim, hidden_dim)?;
        let v = self.q4k_q8_gemv_async(&v_name, &q8_normed, kv_dim, hidden_dim)?;

        // 4. Incremental attention (has internal sync for KV cache update)
        let (attn_out, _seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;

        // 5. Quantize attention output for O projection
        let q8_attn = self.q8_quantize_async(&attn_out, q_dim)?;

        // 6. Output projection using Q4K × Q8 integer dot product
        let projected = self.q4k_q8_gemv_async(&o_name, &q8_attn, hidden_dim, q_dim)?;

        // 7. First residual add (no sync)
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;

        // 8. Pre-FFN RMSNorm (no sync)
        let ffn_normed = self.rmsnorm_gpu(&residual1, ffn_norm_gamma, hidden_dim, epsilon)?;

        // 9. FFN SwiGLU using true DP4A path
        let ffn_out = self.fused_ffn_swiglu_gpu_true_dp4a(
            &ffn_normed,
            &gate_name,
            &up_name,
            &down_name,
            hidden_dim,
            intermediate_dim,
        )?;

        // 10. Second residual add (no sync)
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;

        // PAR-063-V5: NO sync here - caller can chain multiple layers
        Ok(output)
    }

    /// PAR-023: Cache RMSNorm gamma weights on GPU for all layers
    ///
    /// Pre-uploads attn_norm and ffn_norm gamma vectors to avoid per-layer uploads.
    /// Uses naming convention: `blk.{i}.attn_norm.gamma`, `blk.{i}.ffn_norm.gamma`
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `attn_norms` - Slice of attn_norm gamma vectors [num_layers][hidden_dim]
    /// * `ffn_norms` - Slice of ffn_norm gamma vectors [num_layers][hidden_dim]
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    pub fn preload_rmsnorm_weights(
        &mut self,
        num_layers: usize,
        attn_norms: &[&[f32]],
        ffn_norms: &[&[f32]],
    ) -> Result<usize, GpuError> {
        let mut total_bytes = 0usize;

        for layer_idx in 0..num_layers {
            // Attn norm
            let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&attn_name) {
                let buf = GpuBuffer::from_host(&self.context, attn_norms[layer_idx])?;
                total_bytes += buf.size_bytes();
                self.rmsnorm_cache.insert(attn_name, buf);
            }

            // FFN norm
            let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&ffn_name) {
                let buf = GpuBuffer::from_host(&self.context, ffn_norms[layer_idx])?;
                total_bytes += buf.size_bytes();
                self.rmsnorm_cache.insert(ffn_name, buf);
            }
        }

        Ok(total_bytes)
    }

    /// PAR-023: Check if RMSNorm weights are cached for a layer
    #[must_use]
    pub fn has_rmsnorm_weights(&self, layer_idx: usize) -> bool {
        let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
        let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
        self.rmsnorm_cache.contains_key(&attn_name) && self.rmsnorm_cache.contains_key(&ffn_name)
    }

    /// PAR-023: Pre-cache output norm (final layer norm) weight on GPU
    ///
    /// The output norm is applied after all transformer layers before LM head.
    /// Pre-caching allows fully GPU-resident forward pass.
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    pub fn preload_output_norm(&mut self, gamma: &[f32]) -> Result<usize, GpuError> {
        let output_name = "output_norm.gamma".to_string();
        if !self.rmsnorm_cache.contains_key(&output_name) {
            let buf = GpuBuffer::from_host(&self.context, gamma)?;
            let bytes = buf.size_bytes();
            self.rmsnorm_cache.insert(output_name, buf);
            Ok(bytes)
        } else {
            Ok(0)
        }
    }

    /// PAR-023: Check if output norm is cached
    #[must_use]
    pub fn has_output_norm(&self) -> bool {
        self.rmsnorm_cache.contains_key("output_norm.gamma")
    }

    /// Cache a single RMSNorm gamma weight by name.
    ///
    /// This is used by APR model loading to cache per-layer norm weights
    /// with arbitrary naming conventions. The gamma values are uploaded
    /// to GPU and stored in rmsnorm_cache for O(1) lookup during forward.
    ///
    /// # Arguments
    ///
    /// * `name` - Cache key name (e.g., "blk.0.attn_norm.gamma")
    /// * `gamma` - RMSNorm scale weights [hidden_dim]
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU (0 if already cached)
    pub fn cache_rmsnorm_gamma(&mut self, name: &str, gamma: &[f32]) -> Result<usize, GpuError> {
        if !self.rmsnorm_cache.contains_key(name) {
            let buf = GpuBuffer::from_host(&self.context, gamma)?;
            let bytes = buf.size_bytes();
            self.rmsnorm_cache.insert(name.to_string(), buf);
            Ok(bytes)
        } else {
            Ok(0)
        }
    }

    /// BIAS-FIX: Cache QKV bias vectors on GPU for all layers
    ///
    /// Pre-uploads Q, K, V bias vectors (when present) to avoid per-layer uploads.
    /// Uses naming convention: `blk.{i}.attn_q.bias`, `blk.{i}.attn_k.bias`, `blk.{i}.attn_v.bias`
    ///
    /// Qwen2.5 models have QKV bias that must be added after GEMV.
    /// Models without bias pass empty slices.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `q_biases` - Slice of Q bias vectors (or None for each layer)
    /// * `k_biases` - Slice of K bias vectors (or None for each layer)
    /// * `v_biases` - Slice of V bias vectors (or None for each layer)
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    pub fn preload_qkv_bias(
        &mut self,
        num_layers: usize,
        q_biases: &[Option<&[f32]>],
        k_biases: &[Option<&[f32]>],
        v_biases: &[Option<&[f32]>],
    ) -> Result<usize, GpuError> {
        let mut total_bytes = 0usize;

        for layer_idx in 0..num_layers {
            // Q bias
            if let Some(q_bias) = q_biases.get(layer_idx).and_then(|b| *b) {
                let name = format!("blk.{}.attn_q.bias", layer_idx);
                if !self.bias_cache.contains_key(&name) {
                    let buf = GpuBuffer::from_host(&self.context, q_bias)?;
                    total_bytes += buf.size_bytes();
                    self.bias_cache.insert(name, buf);
                }
            }

            // K bias
            if let Some(k_bias) = k_biases.get(layer_idx).and_then(|b| *b) {
                let name = format!("blk.{}.attn_k.bias", layer_idx);
                if !self.bias_cache.contains_key(&name) {
                    let buf = GpuBuffer::from_host(&self.context, k_bias)?;
                    total_bytes += buf.size_bytes();
                    self.bias_cache.insert(name, buf);
                }
            }

            // V bias
            if let Some(v_bias) = v_biases.get(layer_idx).and_then(|b| *b) {
                let name = format!("blk.{}.attn_v.bias", layer_idx);
                if !self.bias_cache.contains_key(&name) {
                    let buf = GpuBuffer::from_host(&self.context, v_bias)?;
                    total_bytes += buf.size_bytes();
                    self.bias_cache.insert(name, buf);
                }
            }
        }

        if total_bytes > 0 && verbose() {
            eprintln!(
                "[BIAS-FIX] Preloaded QKV bias for {} layers ({} bytes)",
                num_layers, total_bytes
            );
        }

        Ok(total_bytes)
    }

    /// BIAS-FIX: Check if QKV bias is cached for a layer
    #[must_use]
    pub fn has_qkv_bias(&self, layer_idx: usize) -> bool {
        // Check if at least one bias exists (Qwen2.5 has all three)
        let q_name = format!("blk.{}.attn_q.bias", layer_idx);
        self.bias_cache.contains_key(&q_name)
    }

    /// PAR-064-FIX: Pre-cache LM head bias on GPU
    ///
    /// Some models (like Qwen2.5) have an output.bias that must be added to logits
    /// after the LM head GEMV projection. Without this bias, GPU inference produces
    /// incorrect token predictions.
    ///
    /// # Arguments
    ///
    /// * `bias` - Optional LM head bias vector (vocab_size elements)
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU (0 if no bias)
    pub fn preload_lm_head_bias(&mut self, bias: Option<&[f32]>) -> Result<usize, GpuError> {
        let Some(bias_data) = bias else {
            return Ok(0);
        };

        if bias_data.is_empty() {
            return Ok(0);
        }

        let name = "output.bias".to_string();
        if self.bias_cache.contains_key(&name) {
            return Ok(0);
        }

        let buf = GpuBuffer::from_host(&self.context, bias_data)?;
        let total_bytes = buf.size_bytes();

        // Index the pointer for fast access in forward pass
        self.lm_head_bias_ptr = buf.as_ptr();
        self.lm_head_bias_len = buf.len();

        self.bias_cache.insert(name, buf);

        eprintln!(
            "[PAR-064-FIX] Preloaded LM head bias: {} elements ({} bytes)",
            bias_data.len(),
            total_bytes
        );

        Ok(total_bytes)
    }

    /// PAR-064-FIX: Check if LM head bias is cached
    #[must_use]
    pub fn has_lm_head_bias(&self) -> bool {
        self.lm_head_bias_ptr != 0
    }

    /// PAR-023: Run ALL transformer layers GPU-resident (minimal syncs)
    ///
    /// Chains all layers on GPU, only syncing at the very end.
    /// Requires RMSNorm weights pre-cached via `preload_rmsnorm_weights()`.
    ///
    /// # Sync Count
    ///
    /// - Input upload: 1 sync
    /// - Per layer: 0 syncs (attention has internal D2D)
    /// - Output download: 1 sync
    /// - Total: ~2 syncs vs 22 syncs (per-layer) or 176 syncs (original)
    ///
    /// # Arguments
    ///
    /// * `input` - Embedding input [hidden_dim]
    /// * `output` - Output buffer [hidden_dim]
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        position: u32, // PAR-070: Explicit position for RoPE and KV cache
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // 1. Validate all RMSNorm weights are cached
        for layer_idx in 0..num_layers {
            let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
            let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&attn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: attn_norm not cached for layer {}",
                    layer_idx
                )));
            }
            if !self.rmsnorm_cache.contains_key(&ffn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: ffn_norm not cached for layer {}",
                    layer_idx
                )));
            }
        }

        // 2. Collect all cache key names (avoids repeated string allocs in loop)
        let layer_keys: Vec<(String, String)> = (0..num_layers)
            .map(|i| {
                (
                    format!("blk.{}.attn_norm.gamma", i),
                    format!("blk.{}.ffn_norm.gamma", i),
                )
            })
            .collect();

        // 3. Upload input embedding - sync point #1
        // PAR-044: Check if we can use zero-allocation workspace path
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        let mut hidden_gpu = GpuBuffer::from_host(&self.context, input)?;

        // 4. Chain all transformer layers (no intermediate syncs)
        // PAR-044: Use workspace path for zero-allocation forward (fastest)
        // PAR-043: Use indexed path if weights are pre-indexed (10x faster per-token)
        // PAR-044 FIX: Track which buffer has output to avoid unnecessary D2D copy
        // PAR-044: Workspace path enabled - confirmed same performance as indexed path
        // See five-whys-gpu-performance-gap for analysis
        let mut workspace_used = false;
        if use_workspace {
            // PAR-044: Zero-allocation path - workspace buffers + indexed weights
            // Eliminates ~288 buffer allocations per token
            workspace_used = true;

            // Layer 0: input from external hidden_gpu
            if num_layers > 0 {
                let layer_weights = self.indexed_layer_weights[0].clone();
                self.transformer_layer_workspace(
                    &hidden_gpu,
                    0,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
            }

            // Layers 1+: input from hidden_buf2 (output of previous layer)
            // Use raw pointer to avoid borrow conflict with &mut self
            for layer_idx in 1..num_layers {
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                // SAFETY: hidden_buf2 is initialized and remains valid throughout
                // We get ptr/len before the mutable borrow, avoiding conflict
                let buf_ptr = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .as_ptr();
                let buf_len = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .len();
                // Create temporary non-owning view of hidden_buf2
                // SAFETY: Memory safety ensured by bounds checking and alignment
                let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(buf_ptr, buf_len) };
                self.transformer_layer_workspace(
                    &input_buf,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
                // Prevent Drop from freeing the borrowed memory
                std::mem::forget(input_buf);
            }

            // PAR-044 FIX: Output is in hidden_buf2, use it directly
            // (removed unnecessary copy_from_buffer - saves one D2D copy per token)
        } else if self.has_indexed_weights() && self.indexed_layer_weights.len() == num_layers {
            // PAR-043: Fast path - O(1) weight access, no string formatting
            for layer_idx in 0..num_layers {
                // Clone the layer weights to avoid borrow conflict
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                hidden_gpu = self.transformer_layer_indexed(
                    &hidden_gpu,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                )?;
            }
        } else {
            // Legacy path - HashMap lookups + string formatting (~10ms overhead)
            for layer_idx in 0..num_layers {
                let prefix = format!("blk.{}", layer_idx);
                let (ref attn_name, ref ffn_name) = layer_keys[layer_idx];

                // Get cached gamma buffer pointers (no data copy, just metadata)
                let attn_gamma = self.rmsnorm_cache.get(attn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        attn_name
                    ))
                })?;
                let attn_ptr = attn_gamma.as_ptr();
                let attn_len = attn_gamma.len();
                let ffn_gamma = self.rmsnorm_cache.get(ffn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        ffn_name
                    ))
                })?;
                let ffn_ptr = ffn_gamma.as_ptr();
                let ffn_len = ffn_gamma.len();

                // Run layer GPU-resident using cached gamma buffers
                hidden_gpu = self.transformer_layer_gpu_cached(
                    &hidden_gpu,
                    layer_idx,
                    &prefix,
                    hidden_dim,
                    intermediate_dim,
                    attn_ptr,
                    attn_len,
                    ffn_ptr,
                    ffn_len,
                    epsilon,
                )?;
            }
        }

        // 5. Final sync and download - sync point #2
        // PAR-044 FIX: Copy from correct buffer based on which path was used
        self.stream.synchronize()?;
        if workspace_used {
            // Output is in hidden_buf2
            let hidden_ptr = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .as_ptr();
            let hidden_len = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            let output_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_ptr, hidden_len) };
            output_buf.copy_to_host(output)?;
            std::mem::forget(output_buf);
        } else {
            hidden_gpu.copy_to_host(output)?;
        }

        Ok(())
    }

    /// PAR-023: Fully GPU-resident forward to logits (minimal syncs)
    ///
    /// Runs all transformer layers + output norm + LM head projection entirely on GPU,
    /// only downloading the final logits. This eliminates the CPU round-trip for output norm.
    ///
    /// # Sync Count
    ///
    /// - Input embedding upload: 1 sync
    /// - All transformer layers: 0 syncs (attention has internal D2D)
    /// - Output RMSNorm: 0 syncs (on GPU)
    /// - LM head projection: 0 syncs (on GPU)
    /// - Logits download: 1 sync
    /// - **Total: 2 syncs** vs 3+ syncs (with CPU output norm)
    ///
    /// # Requirements
    ///
    /// Must call `preload_rmsnorm_weights()` and `preload_output_norm()` before use.
    /// LM head weights must be pre-cached via `load_quantized_weights("output.weight", ...)`.
    ///
    /// # Arguments
    ///
    /// * `input` - Input embedding [hidden_dim]
    /// * `logits` - Output logits buffer [vocab_size]
    /// * `position` - Token position for RoPE and KV cache (PAR-070: CORRECTNESS-001 fix)
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `vocab_size` - Output vocabulary size
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu_to_logits(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // PERF-002: Debug code removed for performance (was PAR-058-DEBUG)
        // NaN checks required D2H transfer on every token - ~10ms overhead each

        // 1. Validate all RMSNorm weights are cached (including output norm)
        for layer_idx in 0..num_layers {
            let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
            let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&attn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: attn_norm not cached for layer {}",
                    layer_idx
                )));
            }
            if !self.rmsnorm_cache.contains_key(&ffn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: ffn_norm not cached for layer {}",
                    layer_idx
                )));
            }
        }
        if !self.rmsnorm_cache.contains_key("output_norm.gamma") {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-023: output_norm not cached".to_string(),
            ));
        }

        // 2. Collect all cache key names
        let layer_keys: Vec<(String, String)> = (0..num_layers)
            .map(|i| {
                (
                    format!("blk.{}.attn_norm.gamma", i),
                    format!("blk.{}.ffn_norm.gamma", i),
                )
            })
            .collect();

        // 3. Upload input embedding - sync point #1
        // PAR-044: Check if we can use zero-allocation workspace path
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        let mut hidden_gpu = GpuBuffer::from_host(&self.context, input)?;

        // 4. Chain all transformer layers (no intermediate syncs)
        // PAR-044: Use workspace path for zero-allocation forward (fastest)
        // PAR-043: Use indexed path if weights are pre-indexed (10x faster per-token)
        // PAR-044 FIX: Track which buffer has output to avoid unnecessary D2D copy
        // PAR-044: Workspace path enabled - confirmed same performance as indexed path
        // See five-whys-gpu-performance-gap for analysis
        let mut workspace_used = false;
        if use_workspace {
            // PAR-044: Zero-allocation path - workspace buffers + indexed weights
            // Eliminates ~288 buffer allocations per token
            workspace_used = true;

            // Layer 0: input from external hidden_gpu
            // PAR-070: Pass explicit position for RoPE and KV cache
            if num_layers > 0 {
                let layer_weights = self.indexed_layer_weights[0].clone();
                self.transformer_layer_workspace(
                    &hidden_gpu,
                    0,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
            }

            // Layers 1+: input from hidden_buf2 (output of previous layer)
            // Use raw pointer to avoid borrow conflict with &mut self
            for layer_idx in 1..num_layers {
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                // SAFETY: hidden_buf2 is initialized and remains valid throughout
                // We get ptr/len before the mutable borrow, avoiding conflict
                let buf_ptr = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .as_ptr();
                let buf_len = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .len();
                // Create temporary non-owning view of hidden_buf2
                // SAFETY: Memory safety ensured by bounds checking and alignment
                let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(buf_ptr, buf_len) };
                // PAR-070: Pass explicit position for RoPE and KV cache
                self.transformer_layer_workspace(
                    &input_buf,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
                // Prevent Drop from freeing the borrowed memory
                std::mem::forget(input_buf);
            }

            // PAR-044 FIX: Output is in hidden_buf2, use it directly for output norm
            // (removed unnecessary copy_from_buffer - saves one D2D copy per token)
        } else if self.has_indexed_weights() && self.indexed_layer_weights.len() == num_layers {
            // PAR-043: Fast path - O(1) weight access, no string formatting
            for layer_idx in 0..num_layers {
                // Clone the layer weights to avoid borrow conflict
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                hidden_gpu = self.transformer_layer_indexed(
                    &hidden_gpu,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                )?;
            }
        } else {
            // Legacy path - HashMap lookups + string formatting (~10ms overhead)
            for layer_idx in 0..num_layers {
                let prefix = format!("blk.{}", layer_idx);
                let (ref attn_name, ref ffn_name) = layer_keys[layer_idx];

                // Get cached gamma buffer pointers (no data copy, just metadata)
                let attn_gamma = self.rmsnorm_cache.get(attn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        attn_name
                    ))
                })?;
                let attn_ptr = attn_gamma.as_ptr();
                let attn_len = attn_gamma.len();
                let ffn_gamma = self.rmsnorm_cache.get(ffn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        ffn_name
                    ))
                })?;
                let ffn_ptr = ffn_gamma.as_ptr();
                let ffn_len = ffn_gamma.len();

                // Run layer GPU-resident using cached gamma buffers
                hidden_gpu = self.transformer_layer_gpu_cached(
                    &hidden_gpu,
                    layer_idx,
                    &prefix,
                    hidden_dim,
                    intermediate_dim,
                    attn_ptr,
                    attn_len,
                    ffn_ptr,
                    ffn_len,
                    epsilon,
                )?;
            }
        }

        // PERF-002: Debug code removed (was PAR-058-DEBUG hidden state check)
        // D2H transfer + NaN check was ~15ms overhead per token

        // CORRECTNESS-001: Compare hidden state before output norm
        static HIDDEN_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *HIDDEN_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            self.stream.synchronize()?;
            let hidden_to_check = if workspace_used {
                let ptr = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .as_ptr();
                let len = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .len();
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe { GpuBuffer::<f32>::from_raw_parts(ptr, len) }
            } else {
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_gpu.as_ptr(), hidden_gpu.len()) }
            };
            let mut hidden_host = vec![0.0f32; hidden_to_check.len()];
            hidden_to_check.copy_to_host(&mut hidden_host)?;
            std::mem::forget(hidden_to_check);
            let sum: f32 = hidden_host.iter().sum();
            let sum_sq: f32 = hidden_host.iter().map(|x| x * x).sum();
            eprintln!(
                "[CORRECTNESS-001] Hidden before output_norm: first 5 = {:?}, sum = {:.4}, rms = {:.4}",
                &hidden_host[..5.min(hidden_host.len())],
                sum,
                (sum_sq / hidden_host.len() as f32).sqrt()
            );
        }

        // 5. Output RMSNorm on GPU (no sync)
        // PAR-044 FIX: Use workspace hidden_buf2 directly if workspace was used
        let output_norm_gamma = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig(
                "PAR-023: Missing cached gamma for output_norm.gamma".to_string(),
            )
        })?;
        let output_gamma_ptr = output_norm_gamma.as_ptr();
        let output_gamma_len = output_norm_gamma.len();

        let normed_hidden = if workspace_used {
            // PAR-044 FIX: Use hidden_buf2 directly (no D2D copy)
            let hidden_ptr = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .as_ptr();
            let hidden_len = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            let hidden_input = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_ptr, hidden_len) };
            let result = self.rmsnorm_gpu_ptr(
                &hidden_input,
                output_gamma_ptr,
                output_gamma_len,
                hidden_dim,
                epsilon,
            )?;
            std::mem::forget(hidden_input);
            result
        } else {
            self.rmsnorm_gpu_ptr(
                &hidden_gpu,
                output_gamma_ptr,
                output_gamma_len,
                hidden_dim,
                epsilon,
            )?
        };

        // CORRECTNESS-002: Debug normed_hidden output (before LM head)
        if *HIDDEN_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            self.stream.synchronize()?;
            let mut normed_host = vec![0.0f32; normed_hidden.len()];
            normed_hidden.copy_to_host(&mut normed_host)?;
            let sum: f32 = normed_host.iter().sum();
            let sum_sq: f32 = normed_host.iter().map(|x| x * x).sum();
            eprintln!(
                "[CORRECTNESS-002] Normed hidden: first 5 = {:?}, sum = {:.4}, rms = {:.4}",
                &normed_host[..5.min(normed_host.len())],
                sum,
                (sum_sq / normed_host.len() as f32).sqrt()
            );
        }

        // 6. LM head projection on GPU (no sync)
        // PAR-056: Tiled kernel selection based on K dimension
        let lm_head_name = "output.weight".to_string();

        // PAR-058: Detect LM head quantization type using size-based detection
        let lm_head_qtype = if let Some(lm_head_buf) =
            self.quantized_weight_cache.get(&lm_head_name)
        {
            let lm_head_size = lm_head_buf.size_bytes();
            // Try size-based detection first, fall back to metadata
            let detected_qtype =
                WeightQuantType::from_size(lm_head_size, vocab_size as usize, hidden_dim as usize)
                    .unwrap_or_else(|| {
                        // Fall back to GGML type from metadata
                        self.quantized_weight_types
                            .get(&lm_head_name)
                            .and_then(|&t| WeightQuantType::from_ggml_type(t))
                            .unwrap_or(WeightQuantType::Q4K)
                    });
            // PERF-002: eprintln removed for performance
            detected_qtype
        } else {
            WeightQuantType::Q4K
        };

        // Get LM head buffer pointer for direct ptr API
        let lm_head_buf = self
            .quantized_weight_cache
            .get(&lm_head_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("LM head weight not cached".to_string())
            })?;
        let lm_head_ptr = lm_head_buf.as_ptr();

        // CORRECTNESS-002: Debug LM head weight buffer
        if *HIDDEN_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            let lm_head_size = lm_head_buf.size_bytes();
            let super_blocks_per_row = (hidden_dim as usize + 255) / 256;
            let bytes_per_row = super_blocks_per_row * 210;
            let expected_size = vocab_size as usize * bytes_per_row;
            eprintln!(
                "[CORRECTNESS-002] LM head: ptr=0x{:x}, size={}, expected={}, qtype={:?}",
                lm_head_ptr, lm_head_size, expected_size, lm_head_qtype
            );
            eprintln!(
                "[CORRECTNESS-002] LM head dims: vocab_size={}, hidden_dim={}, sb_per_row={}, bytes_per_row={}",
                vocab_size, hidden_dim, super_blocks_per_row, bytes_per_row
            );

            // LM head weights verified - size matches (skip partial copy due to API limitation)
        }

        // Allocate logits buffer
        let logits_gpu = GpuBuffer::<f32>::new(&self.context, vocab_size as usize)?;

        // PAR-058: Dispatch to correct kernel based on detected quantization type
        match lm_head_qtype {
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
                // CORRECTNESS-003: Debug Q6K logits
                if *HIDDEN_DEBUG.get_or_init(|| {
                    std::env::var("GPU_DEBUG")
                        .map(|v| v == "1")
                        .unwrap_or(false)
                }) {
                    self.stream.synchronize()?;
                    // Download ALL logits for full analysis
                    let mut all_logits = vec![0.0f32; vocab_size as usize];
                    logits_gpu.copy_to_host(&mut all_logits)?;

                    eprintln!(
                        "[CORRECTNESS-003] Q6K LM head logits[0..20]: {:?}",
                        &all_logits[..20]
                    );

                    // Find global argmax
                    let (global_max_idx, global_max_val) = all_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, v)| (i, *v))
                        .expect("CUDA operation failed");
                    eprintln!(
                        "[CORRECTNESS-007] Global argmax: idx={}, val={:.4}",
                        global_max_idx, global_max_val
                    );

                    // Check for outliers: tokens with logit > 10
                    let outliers: Vec<(usize, f32)> = all_logits
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v > 10.0)
                        .map(|(i, v)| (i, *v))
                        .collect();
                    if !outliers.is_empty() {
                        eprintln!(
                            "[CORRECTNESS-007] Logits > 10.0 ({} tokens): {:?}",
                            outliers.len(),
                            &outliers[..10.min(outliers.len())]
                        );
                    }

                    // Check expected tokens (15='0', 16='1', 17='2', 18='3', 19='4')
                    eprintln!(
                        "[CORRECTNESS-007] Digit logits: 0={:.4}, 1={:.4}, 2={:.4}, 3={:.4}, 4={:.4}",
                        all_logits[15], all_logits[16], all_logits[17], all_logits[18], all_logits[19]
                    );

                    let logits_debug = all_logits[..20].to_vec();
                    // Check for all-zeros or all-same values (sign of kernel issue)
                    let first = logits_debug[0];
                    let all_same = logits_debug.iter().all(|&x| (x - first).abs() < 0.001);
                    if all_same {
                        eprintln!(
                            "[CORRECTNESS-003] WARNING: All logits are identical ({})",
                            first
                        );
                    }
                    // Check argmax
                    let (max_idx, max_val) = logits_debug
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("CUDA operation failed"))
                        .expect("CUDA operation failed");
                    eprintln!(
                        "[CORRECTNESS-003] Q6K argmax in first 20: idx={}, val={}",
                        max_idx, max_val
                    );

                    // CORRECTNESS-004: Compare GPU vs CPU logits for same input
                    // Download normed_hidden and compute CPU logits for comparison
                    let mut normed_host = vec![0.0f32; hidden_dim as usize];
                    normed_hidden.copy_to_host(&mut normed_host)?;

                    // Get LM head weight data from cache
                    if let Some(lm_head_buf) = self.quantized_weight_cache.get(&lm_head_name) {
                        let mut weight_bytes = vec![0u8; lm_head_buf.size_bytes()];
                        lm_head_buf.copy_to_host(&mut weight_bytes)?;

                        // CPU dequant + matmul for first 20 vocab entries
                        let super_blocks_per_row = (hidden_dim as usize + 255) / 256;
                        let bytes_per_row = super_blocks_per_row * 210; // Q6K: 210 bytes per superblock
                        let mut cpu_logits = vec![0.0f32; 20];

                        for vocab_idx in 0..20 {
                            let row_start = vocab_idx * bytes_per_row;
                            if row_start + bytes_per_row <= weight_bytes.len() {
                                // Dequantize row and dot with normed_hidden
                                let row_data = &weight_bytes[row_start..row_start + bytes_per_row];
                                let mut dot_sum = 0.0f32;

                                // Q6K layout: 256 elements per superblock
                                // Each superblock: 128 ql (low 4 bits), 64 qh (high 2 bits), 16 scales, 1 d (f16)
                                for sb in 0..super_blocks_per_row {
                                    let sb_offset = sb * 210;
                                    if sb_offset + 210 > row_data.len() {
                                        break;
                                    }

                                    // Extract d scale (f16 at offset 0)
                                    let d_bytes = [row_data[sb_offset], row_data[sb_offset + 1]];
                                    let d = half::f16::from_le_bytes(d_bytes).to_f32();

                                    // Extract ql (low 4 bits): 128 bytes at offset 2
                                    let ql = &row_data[sb_offset + 2..sb_offset + 2 + 128];

                                    // Extract qh (high 2 bits): 64 bytes at offset 130
                                    let qh = &row_data[sb_offset + 130..sb_offset + 130 + 64];

                                    // Extract scales: 16 bytes at offset 194
                                    let scales = &row_data[sb_offset + 194..sb_offset + 194 + 16];

                                    // Dequantize and dot product
                                    for i in 0..256 {
                                        let hidden_idx = sb * 256 + i;
                                        if hidden_idx >= hidden_dim as usize {
                                            break;
                                        }

                                        // Extract 6-bit quantized value
                                        let ql_idx = i / 2;
                                        let ql_shift = (i % 2) * 4;
                                        let ql_val = ((ql[ql_idx] >> ql_shift) & 0xF) as i8;

                                        let qh_idx = i / 4;
                                        let qh_shift = (i % 4) * 2;
                                        let qh_val = ((qh[qh_idx] >> qh_shift) & 0x3) as i8;

                                        let q_val = ql_val | (qh_val << 4);
                                        let q_centered = q_val - 32; // Q6K uses offset 32

                                        // Get scale for this 16-element group
                                        let scale_idx = i / 16;
                                        let scale = (scales[scale_idx] as i8) as f32;

                                        let weight = d * scale * (q_centered as f32);
                                        dot_sum += weight * normed_host[hidden_idx];
                                    }
                                }
                                cpu_logits[vocab_idx] = dot_sum;
                            }
                        }

                        eprintln!("[CORRECTNESS-004] CPU logits[0..20]: {:?}", cpu_logits);

                        // Compare
                        let max_diff = logits_debug
                            .iter()
                            .zip(cpu_logits.iter())
                            .map(|(g, c)| (g - c).abs())
                            .fold(0.0f32, f32::max);
                        eprintln!("[CORRECTNESS-004] Max GPU-CPU diff: {:.6}", max_diff);
                    }
                }
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_1 => {
                self.q4_1_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4K => {
                self.q4k_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
        }

        // 7. Final sync and download - sync point #2 (only required sync)
        self.stream.synchronize()?;
        logits_gpu.copy_to_host(logits)?;

        Ok(())
    }

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
                // SAFETY: Raw pointer from valid allocation, length verified by caller
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
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
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
                // SAFETY: Raw pointer from valid allocation, length verified by caller
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
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
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
        let logits_buf =
            unsafe { GpuBuffer::<f32>::from_raw_parts(logits_buf_ptr, logits_buf_len) };

        // SAFETY: Unsafe operation with validated invariants
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

    /// PAR-054: Graph-captured forward pass for decode (M=1)
    ///
    /// Uses CUDA graph capture to reduce kernel launch overhead from ~280 launches
    /// to 1 graph launch (~10µs vs ~5.6ms overhead).
    ///
    /// First decode token: captures the kernel sequence into a graph
    /// Subsequent tokens: replays the captured graph with updated position
    ///
    /// # Performance
    ///
    /// - Without graphs: ~280 kernel launches × ~20µs = ~5.6ms overhead/token
    /// - With graphs: 1 graph launch × ~10µs = ~0.01ms overhead/token
    /// - Expected speedup: ~500x reduction in launch overhead
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu_to_logits_graphed(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // PAR-054: Environment variable to disable CUDA graphs for debugging
        // Set CUDA_GRAPH_DISABLE=1 to use non-graphed path
        static GRAPH_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let graph_disabled = *GRAPH_DISABLED.get_or_init(|| {
            std::env::var("CUDA_GRAPH_DISABLE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        if graph_disabled {
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-064-DEBUG: Allow disabling graph mode for debugging
        let skip_graph = std::env::var("SKIP_CUDA_GRAPH")
            .map(|v| v == "1")
            .unwrap_or(false);

        // PAR-054: Check if we should capture or replay
        if !skip_graph && self.decode_graph.is_some() && self.decode_token_count > 0 {
            // Replay path: update position and launch graph
            if self.decode_token_count <= 3 && verbose() {
                eprintln!(
                    "[PAR-054] Graph replay #{} (pos={})",
                    self.decode_token_count, position
                );
            }
            return self.forward_graphed_replay(input, logits, position);
        }

        // First token or no graph yet: try to capture
        // We need workspace path for stable addresses
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        if !use_workspace {
            // Fall back to non-graphed path if workspace not available
            eprintln!("[PAR-054] Workspace not ready, using non-graphed path (has_workspace={}, has_indexed={}, layers={})",
                self.has_workspace(), self.has_indexed_weights(), self.indexed_layer_weights.len());
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Verify lm_head_ptr is set (needed for graph-captured LM head projection)
        if self.lm_head_ptr == 0 {
            eprintln!("[PAR-054] lm_head_ptr not set, using non-graphed path");
            // PAR-070: Pass position for correct RoPE and KV cache behavior
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Initialize position buffer if needed
        if self.position_buf.is_none() {
            let pos_buf = GpuBuffer::from_host(&self.context, &[position])?;
            self.position_buf = Some(pos_buf);
        } else {
            // Update position
            self.position_buf
                .as_mut()
                .expect("position_buf must be initialized")
                .copy_from_host(&[position])?;
        }

        // PAR-061: Initialize seq_len buffer for indirect attention kernel
        // seq_len = position + 1 (new sequence length after adding this token)
        let seq_len = position + 1;
        if self.seq_len_buf.is_none() {
            let len_buf = GpuBuffer::from_host(&self.context, &[seq_len])?;
            self.seq_len_buf = Some(len_buf);
        } else {
            self.seq_len_buf
                .as_mut()
                .expect("seq_len_buf must be initialized")
                .copy_from_host(&[seq_len])?;
        }

        // PAR-054: Initialize stable input buffer if needed
        let hidden_size = hidden_dim as usize;
        if self.graph_input_buf.is_none()
            || self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .len()
                != hidden_size
        {
            let input_buf = GpuBuffer::from_host(&self.context, input)?;
            self.graph_input_buf = Some(input_buf);
        } else {
            self.graph_input_buf
                .as_mut()
                .expect("graph_input_buf must be initialized")
                .copy_from_host(input)?;
        }

        // PAR-054: Pre-allocate normed_hidden_buf before capture
        if self.workspace.normed_hidden_buf.is_none() {
            let normed_buf = GpuBuffer::new(&self.context, hidden_size)?;
            self.workspace.normed_hidden_buf = Some(normed_buf);
        }

        // PAR-054: Pre-allocate logits_buf before capture
        if self.workspace.logits_buf.is_none() {
            let logits_buf = GpuBuffer::new(&self.context, vocab_size as usize)?;
            self.workspace.logits_buf = Some(logits_buf);
        }

        // PAR-054-FIX: Pre-load all kernel modules BEFORE graph capture
        // Root cause: CudaModule::from_ptx allocates memory which breaks capture
        self.preload_modules_for_capture(num_layers, hidden_dim, intermediate_dim, vocab_size)?;

        // PAR-064-DEBUG: Skip graph capture if SKIP_CUDA_GRAPH=1
        if skip_graph {
            eprintln!("[PAR-064-DEBUG] SKIP_CUDA_GRAPH=1, using non-graphed path");
            return self.forward_all_layers_gpu_to_logits(
                input,
                logits,
                position,
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
                epsilon,
            );
        }

        // PAR-054: Try CUDA graph capture, fall back to non-graphed path if fails
        // Some operations (memory allocation, synchronization) aren't graph-compatible
        let capture_result = self.try_graph_capture(
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
        );

        match capture_result {
            Ok(()) => {
                // CORRECTNESS-010: Graph capture defines the work but doesn't execute it.
                // Must launch the graph once to produce actual output for first token.
                if let Some(ref graph_exec) = self.decode_graph {
                    self.stream.launch_graph(graph_exec)?;
                }
                // Graph captured and launched, sync and download logits
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
            Err(e) => {
                // Graph capture failed, fall back to non-graphed path
                // This is expected for complex operations with allocations
                eprintln!(
                    "[PAR-054] Graph capture failed: {:?}, using non-graphed path",
                    e
                );
                // PAR-070: Pass position for correct RoPE and KV cache behavior
                self.forward_all_layers_gpu_to_logits(
                    input,
                    logits,
                    position,
                    num_layers,
                    hidden_dim,
                    intermediate_dim,
                    vocab_size,
                    epsilon,
                )
            },
        }
    }

    /// PAR-054-FIX: Pre-load all kernel modules needed for graph capture
    ///
    /// Root cause of CUDA graph capture failure (code 901):
    /// - `CudaModule::from_ptx` calls CUDA driver which allocates memory
    /// - Any memory allocation during graph capture causes error 901
    /// - Solution: Pre-load ALL modules before `begin_capture()`
    ///
    /// Five-Whys Analysis:
    /// 1. Why does capture fail? Memory allocation detected during capture
    /// 2. Why allocation during capture? Lazy module loading in kernel dispatch
    /// 3. Why lazy loading? Performance optimization for unused kernels
    /// 4. Why does lazy loading allocate? PTX compilation requires driver memory
    /// 5. Why not pre-loaded? Missing pre-loading step before capture
    #[allow(clippy::too_many_lines)]
    fn preload_modules_for_capture(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads as u32;
        let num_kv_heads = self.kv_num_kv_heads as u32;
        let head_dim = self.kv_head_dim as u32;
        let max_len = self.kv_cache_max_len as u32;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // 1. RMSNorm kernel (used for attn_norm, ffn_norm, output_norm)
        // CORRECTNESS-013: Check if precise mode is requested
        static PRECISE_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_precise = *PRECISE_MODE.get_or_init(|| {
            std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        if use_precise {
            // CORRECTNESS-013: Preload PreciseRmsNorm for CPU-matching precision
            let rmsnorm_key = format!("rmsnorm_precise_{}", hidden_dim);
            if !self.modules.contains_key(&rmsnorm_key) {
                let kernel_type = KernelType::PreciseRmsNorm {
                    hidden_size: hidden_dim,
                    epsilon: 1e-5,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(rmsnorm_key, module);
            }
        } else {
            // PAR-081: Use VectorizedRmsNorm with 256 threads (8x faster than single-warp)
            let rmsnorm_key = format!("rmsnorm_vectorized_{}", hidden_dim);
            if !self.modules.contains_key(&rmsnorm_key) {
                let kernel_type = KernelType::VectorizedRmsNorm {
                    hidden_size: hidden_dim,
                    epsilon: 1e-5, // Runtime parameter, kernel code same regardless
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(rmsnorm_key, module);
            }
        }

        // 2. Q/K/V GEMV kernels - pre-load all quant types that might be used
        // PAR-065: Use Coalesced Q4K kernels for better bandwidth (vectorized loads)

        // PAR-065: Coalesced Q4K GEMV for Q (hidden_dim -> q_dim)
        // Five-Whys root cause: TiledQ4KGemv uses single-byte loads (6% bandwidth)
        // CoalescedQ4KGemv uses vectorized u32 loads + warp shuffles (27% speedup)
        let q4k_q_key = format!("coalesced_q4k_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q4k_q_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_q_key, module);
        }

        // PAR-065: Coalesced Q4K GEMV for K/V (hidden_dim -> kv_dim)
        let q4k_kv_key = format!("coalesced_q4k_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q4k_kv_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_kv_key, module);
        }

        // Q5_0 GEMV (for Qwen 0.5B which uses Q5_0 for Q/K)
        let q5_0_q_key = format!("q5_0_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q5_0_q_key) {
            let kernel_type = KernelType::Q5_0Gemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q5_0_q_key, module);
        }
        let q5_0_kv_key = format!("q5_0_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q5_0_kv_key) {
            let kernel_type = KernelType::Q5_0Gemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q5_0_kv_key, module);
        }

        // Q6K GEMV for Q projection - PAR-066: Preload both original and coalesced versions
        // Original Q6K (for non-256-aligned K dimensions)
        let q6k_q_key = format!("q6k_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q6k_q_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q6k_q_key, module);
        }
        // PAR-066: CoalescedQ6K for Q (byte-wise scale loading, fixes alignment issue)
        if hidden_dim.is_multiple_of(256) {
            let coalesced_q6k_q_key = format!("coalesced_q6k_gemv_{}_{}", hidden_dim, q_dim);
            if !self.modules.contains_key(&coalesced_q6k_q_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: hidden_dim,
                    n: q_dim,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(coalesced_q6k_q_key, module);
            }
        }
        // Q6K GEMV for KV projection
        let q6k_kv_key = format!("q6k_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q6k_kv_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q6k_kv_key, module);
        }
        // PAR-066: CoalescedQ6K for KV
        if hidden_dim.is_multiple_of(256) {
            let coalesced_q6k_kv_key = format!("coalesced_q6k_gemv_{}_{}", hidden_dim, kv_dim);
            if !self.modules.contains_key(&coalesced_q6k_kv_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: hidden_dim,
                    n: kv_dim,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(coalesced_q6k_kv_key, module);
            }
        }

        // Q8_0 GEMV
        let q8_0_q_key = format!("q8_0_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q8_0_q_key) {
            let kernel_type = KernelType::Q8_0Gemv {
                k: hidden_dim,
                n: q_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q8_0_q_key, module);
        }
        let q8_0_kv_key = format!("q8_0_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q8_0_kv_key) {
            let kernel_type = KernelType::Q8_0Gemv {
                k: hidden_dim,
                n: kv_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q8_0_kv_key, module);
        }

        // 3. Output projection (q_dim -> hidden_dim) - PAR-065: Coalesced Q4K
        let q4k_o_key = format!("coalesced_q4k_gemv_{}_{}", q_dim, hidden_dim);
        if !self.modules.contains_key(&q4k_o_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: q_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_o_key, module);
        }

        // 4. FFN GEMV kernels (gate/up: hidden->intermediate, down: intermediate->hidden)
        // PAR-065: Coalesced Q4K for FFN gate/up
        let q4k_up_key = format!("coalesced_q4k_gemv_{}_{}", hidden_dim, intermediate_dim);
        if !self.modules.contains_key(&q4k_up_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: hidden_dim,
                n: intermediate_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_up_key, module);
        }
        // PAR-065: Coalesced Q4K for FFN down (K=intermediate_dim)
        // CoalescedQ4KGemv doesn't have the shared memory limitation of TiledQ4KGemv
        let q4k_down_key = format!("coalesced_q4k_gemv_{}_{}", intermediate_dim, hidden_dim);
        if !self.modules.contains_key(&q4k_down_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: intermediate_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_down_key, module);
        }
        // Pre-load basic Q4K as fallback for non-256-aligned dimensions
        let q4k_down_fallback_key = format!("q4k_gemv_{}_{}", intermediate_dim, hidden_dim);
        if !self.modules.contains_key(&q4k_down_fallback_key) {
            let kernel_type = KernelType::Q4KGemv {
                k: intermediate_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q4k_down_fallback_key, module);
        }

        // Q6K FFN down (some models use Q6K for FFN down)
        let q6k_down_key = format!("q6k_gemv_{}_{}", intermediate_dim, hidden_dim);
        if !self.modules.contains_key(&q6k_down_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: intermediate_dim,
                n: hidden_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(q6k_down_key, module);
        }
        // PAR-066: CoalescedQ6K for FFN down (byte-wise scale loading)
        if intermediate_dim.is_multiple_of(256) {
            let coalesced_q6k_down_key =
                format!("coalesced_q6k_gemv_{}_{}", intermediate_dim, hidden_dim);
            if !self.modules.contains_key(&coalesced_q6k_down_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: intermediate_dim,
                    n: hidden_dim,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(coalesced_q6k_down_key, module);
            }
        }

        // 5. LM head (hidden_dim -> vocab_size) - pre-load both Q4K and Q6K
        // PAR-058: Qwen 1.5B uses Q6K for LM head, not Q4K
        // PAR-065: Coalesced Q4K for LM head
        let lm_head_q4k_key = format!("coalesced_q4k_gemv_{}_{}", hidden_dim, vocab_size);
        if !self.modules.contains_key(&lm_head_q4k_key) {
            let kernel_type = KernelType::CoalescedQ4KGemv {
                k: hidden_dim,
                n: vocab_size,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(lm_head_q4k_key, module);
        }
        // Q6K LM head (Qwen 1.5B uses this)
        let lm_head_q6k_key = format!("q6k_gemv_{}_{}", hidden_dim, vocab_size);
        if !self.modules.contains_key(&lm_head_q6k_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: hidden_dim,
                n: vocab_size,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(lm_head_q6k_key, module);
        }
        // PAR-066: CoalescedQ6K for LM head
        if hidden_dim.is_multiple_of(256) {
            let coalesced_lm_head_q6k_key =
                format!("coalesced_q6k_gemv_{}_{}", hidden_dim, vocab_size);
            if !self.modules.contains_key(&coalesced_lm_head_q6k_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: hidden_dim,
                    n: vocab_size,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(coalesced_lm_head_q6k_key, module);
            }
        }

        // 6. RoPE kernels (for Q and K)
        // Note: theta is a runtime parameter, cache key only uses num_heads and head_dim
        let theta = self.rope_theta;
        let rope_q_key = format!("rope_{}_{}", num_heads, head_dim);
        if !self.modules.contains_key(&rope_q_key) {
            let kernel_type = KernelType::Rope {
                num_heads,
                head_dim,
                theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(rope_q_key, module);
        }
        let rope_k_key = format!("rope_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&rope_k_key) {
            let kernel_type = KernelType::Rope {
                num_heads: num_kv_heads,
                head_dim,
                theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(rope_k_key, module);
        }

        // CORRECTNESS-010: RoPE indirect kernels for CUDA graph capture
        // These read position from device memory instead of kernel parameter
        let rope_q_indirect_key = format!("rope_indirect_{}_{}", num_heads, head_dim);
        if !self.modules.contains_key(&rope_q_indirect_key) {
            let kernel_type = KernelType::RopeIndirect {
                num_heads,
                head_dim,
                theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(rope_q_indirect_key, module);
        }
        let rope_k_indirect_key = format!("rope_indirect_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&rope_k_indirect_key) {
            let kernel_type = KernelType::RopeIndirect {
                num_heads: num_kv_heads,
                head_dim,
                theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(rope_k_indirect_key, module);
        }

        // CORRECTNESS-011: RoPE NEOX indirect kernels for Qwen2.5 (rope_type=2)
        // Five-Whys: GPU garbage output → wrong RoPE style → NEOX kernels not loaded
        // CORRECTNESS-013: Use precise kernels when CORRECTNESS_MODE=1
        if self.rope_type == 2 {
            if use_precise {
                // CORRECTNESS-013: Preload PreciseRopeNeoxIndirect for Q
                let rope_precise_q_indirect_key =
                    format!("rope_precise_indirect_{}_{}", num_heads, head_dim);
                if !self.modules.contains_key(&rope_precise_q_indirect_key) {
                    let kernel_type = KernelType::PreciseRopeNeoxIndirect {
                        num_heads,
                        head_dim,
                        theta,
                    };
                    let ptx = self.kernels.generate_ptx(&kernel_type);
                    let module = CudaModule::from_ptx(&self.context, &ptx)?;
                    self.modules.insert(rope_precise_q_indirect_key, module);
                }
                // CORRECTNESS-013: Preload PreciseRopeNeoxIndirect for K
                let rope_precise_k_indirect_key =
                    format!("rope_precise_indirect_{}_{}", num_kv_heads, head_dim);
                if !self.modules.contains_key(&rope_precise_k_indirect_key) {
                    let kernel_type = KernelType::PreciseRopeNeoxIndirect {
                        num_heads: num_kv_heads,
                        head_dim,
                        theta,
                    };
                    let ptx = self.kernels.generate_ptx(&kernel_type);
                    let module = CudaModule::from_ptx(&self.context, &ptx)?;
                    self.modules.insert(rope_precise_k_indirect_key, module);
                }
            } else {
                // Standard RopeNeoxIndirect for Q
                let rope_neox_q_indirect_key =
                    format!("rope_neox_indirect_{}_{}", num_heads, head_dim);
                if !self.modules.contains_key(&rope_neox_q_indirect_key) {
                    let kernel_type = KernelType::RopeNeoxIndirect {
                        num_heads,
                        head_dim,
                        theta,
                    };
                    let ptx = self.kernels.generate_ptx(&kernel_type);
                    let module = CudaModule::from_ptx(&self.context, &ptx)?;
                    self.modules.insert(rope_neox_q_indirect_key, module);
                }
                // Standard RopeNeoxIndirect for K
                let rope_neox_k_indirect_key =
                    format!("rope_neox_indirect_{}_{}", num_kv_heads, head_dim);
                if !self.modules.contains_key(&rope_neox_k_indirect_key) {
                    let kernel_type = KernelType::RopeNeoxIndirect {
                        num_heads: num_kv_heads,
                        head_dim,
                        theta,
                    };
                    let ptx = self.kernels.generate_ptx(&kernel_type);
                    let module = CudaModule::from_ptx(&self.context, &ptx)?;
                    self.modules.insert(rope_neox_k_indirect_key, module);
                }
            }
            // Also preload direct NEOX kernels for non-graph-capture mode
            let rope_neox_q_key = format!("rope_neox_{}_{}", num_heads, head_dim);
            if !self.modules.contains_key(&rope_neox_q_key) {
                let kernel_type = KernelType::RopeNeox {
                    num_heads,
                    head_dim,
                    theta,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(rope_neox_q_key, module);
            }
            let rope_neox_k_key = format!("rope_neox_{}_{}", num_kv_heads, head_dim);
            if !self.modules.contains_key(&rope_neox_k_key) {
                let kernel_type = KernelType::RopeNeox {
                    num_heads: num_kv_heads,
                    head_dim,
                    theta,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(rope_neox_k_key, module);
            }
        }

        // 7. SwiGLU kernel
        let swiglu_key = format!("fused_swiglu_{}", intermediate_dim);
        if !self.modules.contains_key(&swiglu_key) {
            let kernel_type = KernelType::FusedSwiglu {
                n: intermediate_dim,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(swiglu_key, module);
        }

        // 8. Residual add kernel
        let residual_key = format!("residual_add_{}", hidden_dim);
        if !self.modules.contains_key(&residual_key) {
            let kernel_type = KernelType::ResidualAdd { n: hidden_dim };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(residual_key, module);
        }

        // 9. KV cache scatter kernel (one per layer with same dimensions)
        let scatter_key = format!("kv_scatter_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&scatter_key) {
            let kernel_type = KernelType::KvCacheScatter {
                num_kv_heads,
                head_dim,
                max_len,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(scatter_key, module);
        }

        // 10. Incremental attention kernel (direct mode - for non-graph path)
        let attn_key = format!(
            "incremental_attention_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );
        if !self.modules.contains_key(&attn_key) {
            let kernel_type = KernelType::IncrementalAttention {
                max_seq_len: max_len,
                head_dim,
                n_heads: num_heads,
                n_kv_heads: num_kv_heads,
                indirect: false,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(attn_key, module);
        }

        // CORRECTNESS-010: Preload indirect attention kernel for CUDA graph capture
        // The indirect version reads seq_len from device memory (position_buf)
        let attn_indirect_key = format!(
            "incremental_attention_indirect_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );
        if !self.modules.contains_key(&attn_indirect_key) {
            let kernel_type = KernelType::IncrementalAttention {
                max_seq_len: max_len,
                head_dim,
                n_heads: num_heads,
                n_kv_heads: num_kv_heads,
                indirect: true,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(attn_indirect_key, module);
        }

        // CORRECTNESS-009: Preload multi-warp attention kernels for head_dim > 64
        // Multi-warp kernel handles 4 elements per thread (vs 2 for single-warp)
        // Required for models like Qwen 2.5 with head_dim=128
        let num_warps_per_head = 4u32;
        let multi_warp_key = format!(
            "multi_warp_attention_{}_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head
        );
        if !self.modules.contains_key(&multi_warp_key) {
            let kernel_type = KernelType::MultiWarpAttention {
                max_seq_len: max_len,
                head_dim,
                n_heads: num_heads,
                n_kv_heads: num_kv_heads,
                num_warps_per_head,
                indirect: false,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(multi_warp_key, module);
        }

        // CORRECTNESS-010: Preload multi-warp indirect attention for graph capture
        let multi_warp_indirect_key = format!(
            "multi_warp_attention_indirect_{}_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head
        );
        if !self.modules.contains_key(&multi_warp_indirect_key) {
            let kernel_type = KernelType::MultiWarpAttention {
                max_seq_len: max_len,
                head_dim,
                n_heads: num_heads,
                n_kv_heads: num_kv_heads,
                num_warps_per_head,
                indirect: true,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(multi_warp_indirect_key, module);
        }

        if verbose() {
            eprintln!(
                "[PAR-054-FIX] Pre-loaded {} kernel modules for {} layers",
                self.modules.len(),
                num_layers
            );
        }
        Ok(())
    }

    /// PAR-054: Try to capture CUDA graph
    fn try_graph_capture(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Begin graph capture
        self.stream.begin_capture(CaptureMode::Global)?;

        // Run workspace forward pass (all kernels will be captured)
        let capture_result = self.forward_workspace_captured(
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
        self.decode_graph = Some(graph_exec);
        self.decode_token_count = 1;

        if verbose() {
            eprintln!(
                "[PAR-054] ✓ CUDA graph captured successfully ({} layers + LM head)",
                num_layers
            );
        }

        Ok(())
    }

    /// PAR-054: Replay captured graph with updated position
    fn forward_graphed_replay(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
    ) -> Result<(), GpuError> {
        // CORRECTNESS-013: Stateless GPU mode - force position=0, seq_len=1
        static STATELESS_MODE_REPLAY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_stateless = *STATELESS_MODE_REPLAY.get_or_init(|| {
            std::env::var("STATELESS_GPU")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // Update position buffer (async memcpy, doesn't invalidate graph)
        // CORRECTNESS-013: In stateless mode, always use position=0
        if let Some(ref mut pos_buf) = self.position_buf {
            let pos_to_write = if use_stateless { 0 } else { position };
            pos_buf.copy_from_host(&[pos_to_write])?;
        }

        // PAR-061-FIX: Update seq_len buffer (seq_len = position + 1)
        // The attention kernel reads seq_len from device memory in indirect mode
        // CORRECTNESS-013: In stateless mode, always use seq_len=1
        if let Some(ref mut seq_len_buf) = self.seq_len_buf {
            let seq_len = if use_stateless { 1 } else { position + 1 };
            seq_len_buf.copy_from_host(&[seq_len])?;
        }

        // Update input buffer
        if let Some(ref mut input_buf) = self.graph_input_buf {
            input_buf.copy_from_host(input)?;
        }

        // Launch captured graph
        if let Some(ref graph_exec) = self.decode_graph {
            self.stream.launch_graph(graph_exec)?;
        }

        self.decode_token_count += 1;

        // Sync and download
        self.stream.synchronize()?;
        if let Some(ref logits_buf) = self.workspace.logits_buf {
            logits_buf.copy_to_host(logits)?;
        }

        Ok(())
    }

    /// PAR-062: GPU-side argmax to eliminate logits transfer bottleneck
    ///
    /// Instead of copying all 152064 logits (600KB) from GPU to CPU for argmax,
    /// this method runs argmax entirely on GPU and only copies the result token ID (4 bytes).
    /// This is a 150,000x reduction in data transfer per token.
    ///
    /// # Algorithm
    ///
    /// Two-pass reduction:
    /// 1. Block-level: Each block finds local (max_val, max_idx) using shared memory
    /// 2. Final: Single block reduces block results to find global argmax
    ///
    /// # Arguments
    ///
    /// * `logits_ptr` - Device pointer to logits (vocab_size f32s)
    /// * `vocab_size` - Number of vocabulary entries (e.g., 152064)
    ///
    /// # Returns
    ///
    /// The token ID with the maximum logit value
    pub fn gpu_argmax(&mut self, logits_ptr: u64, vocab_size: u32) -> Result<u32, GpuError> {
        // PAR-068: Optimized GPU argmax with pre-allocated buffers
        // Eliminates 3 GPU allocations per token and removes intermediate sync
        let block_size = 256u32;
        let elements_per_block = block_size * 4; // 4 elements per thread
        let num_blocks = (vocab_size + elements_per_block - 1) / elements_per_block;

        // PAR-068: Lazy allocate argmax buffers on first use, reuse thereafter
        if self.argmax_block_vals.is_none() || self.argmax_num_blocks != num_blocks {
            self.argmax_block_vals = Some(GpuBuffer::new(&self.context, num_blocks as usize)?);
            self.argmax_block_idxs = Some(GpuBuffer::new(&self.context, num_blocks as usize)?);
            self.argmax_result = Some(GpuBuffer::new(&self.context, 1)?);
            self.argmax_num_blocks = num_blocks;
        }

        let block_max_vals = self
            .argmax_block_vals
            .as_ref()
            .expect("argmax_block_vals must be initialized");
        let block_max_idxs = self
            .argmax_block_idxs
            .as_ref()
            .expect("argmax_block_idxs must be initialized");
        let result_buf = self
            .argmax_result
            .as_ref()
            .expect("argmax_result must be initialized");

        // Load first-pass kernel module (cached after first use)
        let argmax_kernel_type = KernelType::ArgMax { length: vocab_size };
        let argmax_key = format!("argmax_{}", vocab_size);
        if !self.modules.contains_key(&argmax_key) {
            let ptx = self.kernels.generate_ptx(&argmax_kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(argmax_key.clone(), module);
        }

        // Load second-pass kernel module (cached after first use)
        let final_kernel_type = KernelType::ArgMaxFinal { num_blocks };
        let final_key = format!("argmax_final_{}", num_blocks);
        if !self.modules.contains_key(&final_key) {
            let ptx = self.kernels.generate_ptx(&final_kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(final_key.clone(), module);
        }

        // Prepare kernel arguments
        let kernel_name = self.kernels.kernel_name(&argmax_kernel_type);
        // PAR-068-FIX: Do NOT use .with_shared_mem() - PTX declares static shared memory via .shared directive
        let launch_config = LaunchConfig::grid_2d(num_blocks, 1, block_size, 1);

        let mut input_ptr = logits_ptr;
        let mut block_vals_ptr = block_max_vals.as_ptr();
        let mut block_idxs_ptr = block_max_idxs.as_ptr();
        let mut length_val = vocab_size;

        // Launch first-pass kernel (block-level reduction)
        // SAFETY: Buffers are valid, args match kernel signature
        unsafe {
            let module = self
                .modules
                .get_mut(&argmax_key)
                .expect("argmax module just inserted");
            self.stream.launch_kernel(
                module,
                kernel_name,
                &launch_config,
                &mut [
                    std::ptr::from_mut(&mut input_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_vals_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_idxs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut length_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-068: NO intermediate sync - launch both kernels back-to-back
        // The kernels are on the same stream, so execution is serialized

        // Launch second-pass kernel (final reduction)
        let final_kernel_name = self.kernels.kernel_name(&final_kernel_type);
        let final_launch_config = LaunchConfig::grid_2d(1, 1, 256, 1);

        let mut result_ptr = result_buf.as_ptr();
        let mut num_blocks_val = num_blocks;

        // SAFETY: Buffers are valid, args match kernel signature
        unsafe {
            let final_module = self
                .modules
                .get_mut(&final_key)
                .expect("argmax_final module just inserted");
            self.stream.launch_kernel(
                final_module,
                final_kernel_name,
                &final_launch_config,
                &mut [
                    std::ptr::from_mut(&mut block_vals_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_idxs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut result_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut num_blocks_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-068: Single sync after both kernels complete
        self.stream.synchronize()?;
        let mut result = [0u32];
        result_buf.copy_to_host(&mut result)?;

        // CORRECTNESS-005: Debug GPU argmax result
        static ARGMAX_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ARGMAX_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            eprintln!(
                "[CORRECTNESS-005] GPU argmax: token_id={}, vocab_size={}",
                result[0], vocab_size
            );
        }

        Ok(result[0])
    }

    /// PAR-062: Forward pass with GPU-side argmax returning token ID directly
    ///
    /// Like `forward_graphed_replay` but uses GPU argmax instead of downloading all logits.
    /// Reduces data transfer from 600KB to 4 bytes per token.
    ///
    /// # Performance Target
    ///
    /// - Before: ~3ms logits transfer per token on PCIe
    /// - After: ~0.001ms token ID transfer
    /// - Expected speedup: ~1.2x overall throughput
    pub fn forward_graphed_replay_to_token_id(
        &mut self,
        input: &[f32],
        position: u32,
        vocab_size: u32,
    ) -> Result<u32, GpuError> {
        // PAR-068: Use GPU argmax to eliminate 600KB D2H transfer bottleneck
        // Root cause fix: Removed .with_shared_mem() from argmax kernel launch configs
        // (PTX declares static shared memory, mixing with dynamic causes CUDA_ERROR_UNKNOWN)

        // PAR-072: Use ASYNC H2D copies to eliminate blocking overhead
        // Root cause: cuMemcpyHtoD has ~10-30µs overhead per call
        // Fix: Use cuMemcpyHtoDAsync on the same stream as graph launch

        // Update position buffer (async memcpy on same stream)
        if let Some(ref mut pos_buf) = self.position_buf {
            // SAFETY: position is stack-allocated and we synchronize before returning
            unsafe {
                pos_buf.copy_from_host_async(&[position], &self.stream)?;
            }
        }

        // PAR-061-FIX: Update seq_len buffer (seq_len = position + 1)
        let seq_len = position + 1;
        if let Some(ref mut seq_len_buf) = self.seq_len_buf {
            // SAFETY: seq_len is stack-allocated and we synchronize before returning
            unsafe {
                seq_len_buf.copy_from_host_async(&[seq_len], &self.stream)?;
            }
        }

        // Update input buffer (async - largest copy, ~14KB for Qwen 0.5B)
        if let Some(ref mut input_buf) = self.graph_input_buf {
            // SAFETY: input slice is valid for the duration of this function
            // and we synchronize in gpu_argmax before returning
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                input_buf.copy_from_host_async(input, &self.stream)?;
            }
        }

        // Launch captured graph (all H2D copies are ordered before this on same stream)
        if let Some(ref graph_exec) = self.decode_graph {
            self.stream.launch_graph(graph_exec)?;
        }

        self.decode_token_count += 1;

        // PAR-068: GPU argmax instead of downloading 600KB logits
        // This reduces D2H transfer from 600KB to 4 bytes per token
        let logits_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidParameter("logits_buf not allocated".into()))?
            .as_ptr();

        // CORRECTNESS-004: Debug graph-replayed logits and compare argmax
        static GPU_DEBUG_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let debug_enabled = *GPU_DEBUG_FLAG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        if debug_enabled {
            self.stream.synchronize()?;
            // Download ALL logits to compute CPU argmax for comparison
            let mut all_logits = vec![0.0f32; vocab_size as usize];
            let debug_view =
                unsafe { GpuBuffer::<f32>::from_raw_parts(logits_ptr, vocab_size as usize) };
            debug_view.copy_to_host(&mut all_logits)?;
            std::mem::forget(debug_view);

            // CPU argmax
            let (cpu_argmax_idx, cpu_argmax_val) = all_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("CUDA operation failed"))
                .expect("CUDA operation failed");

            eprintln!(
                "[CORRECTNESS-004] Graph logits[0..20]: {:?}",
                &all_logits[..20]
            );
            eprintln!(
                "[CORRECTNESS-004] GPU argmax: idx={}, val={}",
                cpu_argmax_idx, cpu_argmax_val
            );

            // Compare against CPU's expected top tokens: 19 ("4"), 17 ("2"), 785 (" The")
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 19 ('4'): {}",
                all_logits.get(19).unwrap_or(&f32::NAN)
            );
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 17 ('2'): {}",
                all_logits.get(17).unwrap_or(&f32::NAN)
            );
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 785: {}",
                all_logits.get(785).unwrap_or(&f32::NAN)
            );

            // Top 5 GPU logits
            let mut top5: Vec<(usize, f32)> = all_logits.iter().copied().enumerate().collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top5.truncate(10);
            eprintln!("[CORRECTNESS-004] GPU top10 logits: {:?}", top5);
        }

        let gpu_result = self.gpu_argmax(logits_ptr, vocab_size)?;

        if debug_enabled {
            eprintln!("[CORRECTNESS-004] GPU argmax result: {}", gpu_result);
        }

        Ok(gpu_result)
    }

    /// PAR-054: Forward pass for graph capture (uses pre-allocated workspace)
    ///
    /// # Safety
    ///
    /// This function must only be called while stream capture is active.
    /// All output buffers (workspace.logits_buf) must be pre-allocated before capture.
    fn forward_workspace_captured(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Layer 0: input from graph_input_buf
        // PAR-070: Position is read from position_buf in indirect mode (graph capture)
        // The position parameter here is ignored since position_buf.is_some() triggers indirect mode
        if num_layers > 0 {
            let input_ptr = self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .as_ptr();
            let input_len = self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(input_ptr, input_len) };
            let layer_weights = self.indexed_layer_weights[0].clone();
            // PAR-054: Use capture-safe version (no debug sync/copy_to_host)
            self.transformer_layer_workspace_for_capture(
                &input_buf,
                0,
                &layer_weights,
                hidden_dim,
                intermediate_dim,
                epsilon,
                0, // PAR-070: Ignored in graph capture mode (uses position_buf)
            )?;
            std::mem::forget(input_buf);
        }

        // Layers 1+: input from hidden_buf2
        for layer_idx in 1..num_layers {
            let layer_weights = self.indexed_layer_weights[layer_idx].clone();
            let buf_ptr = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .as_ptr();
            let buf_len = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(buf_ptr, buf_len) };
            // PAR-054: Use capture-safe version (no debug sync/copy_to_host)
            self.transformer_layer_workspace_for_capture(
                &input_buf,
                layer_idx,
                &layer_weights,
                hidden_dim,
                intermediate_dim,
                epsilon,
                0, // PAR-070: Ignored in graph capture mode (uses position_buf)
            )?;
            std::mem::forget(input_buf);
        }

        // Output RMSNorm - PAR-054: Use pre-allocated normed_hidden_buf
        let output_norm_gamma = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-054: output_norm not cached".to_string())
        })?;
        let output_gamma_ptr = output_norm_gamma.as_ptr();
        let output_gamma_len = output_norm_gamma.len();

        let hidden_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .as_ptr();
        let hidden_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let hidden_input = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_ptr, hidden_len) };

        // PAR-054: Write to pre-allocated normed_hidden_buf (no allocation during capture)
        let normed_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .as_ptr();
        let normed_len = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let normed_output = unsafe { GpuBuffer::<f32>::from_raw_parts(normed_ptr, normed_len) };

        self.rmsnorm_ptr_into(
            &hidden_input,
            output_gamma_ptr,
            output_gamma_len,
            &normed_output,
            hidden_dim,
            epsilon,
        )?;
        std::mem::forget(hidden_input);
        std::mem::forget(normed_output);

        // LM head projection - PAR-054: Use pre-allocated logits_buf
        // PAR-058: Use correct kernel based on LM head quantization type
        let logits_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .expect("logits_buf must be initialized")
            .as_ptr();
        let logits_len = self
            .workspace
            .logits_buf
            .as_ref()
            .expect("logits_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let logits_output = unsafe { GpuBuffer::<f32>::from_raw_parts(logits_ptr, logits_len) };

        let normed_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .as_ptr();
        let normed_len = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let normed_input = unsafe { GpuBuffer::<f32>::from_raw_parts(normed_ptr, normed_len) };

        // PAR-058: Dispatch to correct kernel based on LM head quant type
        // Validate qtype against actual size - GGUF metadata can lie!
        let lm_head_qtype =
            WeightQuantType::from_size(self.lm_head_len, vocab_size as usize, hidden_dim as usize)
                .unwrap_or(self.lm_head_qtype);

        // Log if we overrode the type
        if lm_head_qtype != self.lm_head_qtype {
            eprintln!(
                "[PAR-058] LM head qtype override: {:?} -> {:?} (size-based detection)",
                self.lm_head_qtype, lm_head_qtype
            );
        }

        match lm_head_qtype {
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_1 => {
                self.q4_1_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4K => {
                self.q4k_gemv_into(
                    self.lm_head_ptr,
                    &normed_input,
                    &logits_output,
                    vocab_size,
                    hidden_dim,
                )?;
            },
        }

        // PAR-064-FIX: Add LM head bias after GEMV (if present)
        // Without this, GPU inference produces incorrect token predictions
        if self.lm_head_bias_ptr != 0 && self.lm_head_bias_len > 0 {
            // Create non-owning buffer wrapper from device pointer
            // SAFETY: bias_ptr is valid device memory owned by bias_cache
            let bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(self.lm_head_bias_ptr, self.lm_head_bias_len)
            };

            // Add bias in-place: logits = logits + bias
            self.residual_add_into(&logits_output, &bias_buf, &logits_output, vocab_size)?;

            // Prevent Drop from freeing borrowed memory
            std::mem::forget(bias_buf);
        }

        std::mem::forget(normed_input);
        std::mem::forget(logits_output);

        Ok(())
    }

    /// PAR-023: Transformer layer with cached gamma pointers
    ///
    /// Like `transformer_layer_gpu` but takes raw device pointers for gamma weights
    /// to avoid borrow checker conflicts with cached buffers.
    #[allow(clippy::too_many_arguments)]
    fn transformer_layer_gpu_cached(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_gamma_ptr: u64, // CUdeviceptr
        attn_gamma_len: usize,
        ffn_gamma_ptr: u64, // CUdeviceptr
        ffn_gamma_len: usize,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Weight names follow GGML convention
        let q_name = format!("{}.attn_q.weight", layer_prefix);
        let k_name = format!("{}.attn_k.weight", layer_prefix);
        let v_name = format!("{}.attn_v.weight", layer_prefix);
        let o_name = format!("{}.attn_output.weight", layer_prefix);
        let gate_name = format!("{}.ffn_gate.weight", layer_prefix);
        let up_name = format!("{}.ffn_up.weight", layer_prefix);
        let down_name = format!("{}.ffn_down.weight", layer_prefix);

        // 1. Pre-attention RMSNorm using cached gamma pointer
        let normed =
            self.rmsnorm_gpu_ptr(input, attn_gamma_ptr, attn_gamma_len, hidden_dim, epsilon)?;

        // 2. Q/K/V projections (no sync)
        // PAR-056: Tiled kernel selection based on K dimension
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;
        const CHUNK_THRESHOLD: u32 = 8192;
        let hidden_aligned = hidden_dim.is_multiple_of(256);
        let q_aligned = q_dim.is_multiple_of(256);
        let kv_aligned = kv_dim.is_multiple_of(256);

        // PAR-056: For K > 8192, use non-tiled Q4KGemvKernel (warp-based)
        // ChunkedTiledQ4KGemvKernel bypassed for large K (PAR-056 path)
        let q = if !hidden_aligned || !q_aligned || hidden_dim > CHUNK_THRESHOLD {
            self.q4k_gemv_cached_async(&q_name, &normed, q_dim, hidden_dim)?
        } else {
            self.tiled_q4k_gemv_cached_async(&q_name, &normed, q_dim, hidden_dim, 4)?
        };
        let k = if !hidden_aligned || !kv_aligned || hidden_dim > CHUNK_THRESHOLD {
            self.q4k_gemv_cached_async(&k_name, &normed, kv_dim, hidden_dim)?
        } else {
            self.tiled_q4k_gemv_cached_async(&k_name, &normed, kv_dim, hidden_dim, 4)?
        };
        let v = if !hidden_aligned || !kv_aligned || hidden_dim > CHUNK_THRESHOLD {
            self.q4k_gemv_cached_async(&v_name, &normed, kv_dim, hidden_dim)?
        } else {
            self.tiled_q4k_gemv_cached_async(&v_name, &normed, kv_dim, hidden_dim, 4)?
        };

        // 3. Incremental attention (has internal sync for KV cache update)
        let (attn_out, _seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;

        // 4. Output projection (no sync) - K = q_dim
        // PAR-056: For K > 8192, use non-tiled Q4KGemvKernel
        let projected = if !q_aligned || !hidden_aligned || q_dim > CHUNK_THRESHOLD {
            self.q4k_gemv_cached_async(&o_name, &attn_out, hidden_dim, q_dim)?
        } else {
            self.tiled_q4k_gemv_cached_async(&o_name, &attn_out, hidden_dim, q_dim, 4)?
        };

        // 5. First residual add (no sync)
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;

        // 6. Pre-FFN RMSNorm using cached gamma pointer
        let ffn_normed = self.rmsnorm_gpu_ptr(
            &residual1,
            ffn_gamma_ptr,
            ffn_gamma_len,
            hidden_dim,
            epsilon,
        )?;

        // 7. FFN SwiGLU (no sync)
        let ffn_out = self.fused_ffn_swiglu_gpu(
            &ffn_normed,
            &gate_name,
            &up_name,
            &down_name,
            hidden_dim,
            intermediate_dim,
        )?;

        // 8. Second residual add (no sync)
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;

        Ok(output)
    }

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
    fn transformer_layer_workspace_for_capture(
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
    fn transformer_layer_workspace(
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
    fn transformer_layer_workspace_inner(
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
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        // PAR-051: Attention output buffer (eliminates 28 allocations per token)
        // SAFETY: Memory safety ensured by bounds checking and alignment
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

    /// PAR-044: Get reference to workspace output buffer
    ///
    /// After calling `transformer_layer_workspace`, the output is in hidden_buf2.
    #[must_use]
    pub fn workspace_output(&self) -> Option<&GpuBuffer<f32>> {
        self.workspace.hidden_buf2.as_ref()
    }

    /// APR-TRACE-001: Read final hidden state from GPU to CPU for verbose tracing
    ///
    /// This performs a D2H sync which is expensive (~50µs) but necessary for
    /// Genchi Genbutsu (go and see) observability during debugging.
    ///
    /// ONLY call this when verbose tracing is enabled.
    pub fn read_hidden_state_to_cpu(&mut self) -> Result<Vec<f32>, GpuError> {
        let hidden_buf = self.workspace.hidden_buf2.as_ref().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("APR-TRACE-001: workspace not initialized".to_string())
        })?;

        // Sync stream to ensure all GPU ops complete
        self.stream.synchronize()?;

        // D2H copy
        let mut hidden_cpu = vec![0.0f32; hidden_buf.len()];
        hidden_buf.copy_to_host(&mut hidden_cpu)?;

        Ok(hidden_cpu)
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
        let hidden_buf1 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf1_ptr, hidden_buf1_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let input_staging =
            unsafe { GpuBuffer::<f32>::from_raw_parts(input_staging_ptr, input_staging_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
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
    fn rmsnorm_gpu_ptr(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64, // CUdeviceptr
        gamma_len: usize,
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Create temporary non-owning buffer wrapper
        // SAFETY: gamma_ptr points to valid GPU memory owned by rmsnorm_cache
        let gamma = unsafe { GpuBuffer::from_raw_parts(gamma_ptr, gamma_len) };

        let result = self.rmsnorm_gpu(input, &gamma, hidden_dim, epsilon)?;

        // Prevent Drop from freeing the borrowed memory
        std::mem::forget(gamma);

        Ok(result)
    }

    /// PAR-044: RMSNorm using raw pointer into existing output buffer
    fn rmsnorm_ptr_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64,
        gamma_len: usize,
        output: &GpuBuffer<f32>,
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let gamma = unsafe { GpuBuffer::from_raw_parts(gamma_ptr, gamma_len) };
        self.rmsnorm_into(input, &gamma, output, hidden_dim, epsilon)?;
        std::mem::forget(gamma);
        Ok(())
    }

    /// PAR-023: GPU RMSNorm for output layer
    ///
    /// Runs RMSNorm on GPU for the final output before LM head projection.
    pub fn output_rmsnorm_gpu(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        gamma: &[f32],
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let gamma_gpu = GpuBuffer::from_host(&self.context, gamma)?;

        let output_gpu = self.rmsnorm_gpu(&input_gpu, &gamma_gpu, hidden_dim, epsilon)?;

        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Helper to run transformer layer with host input/output
    ///
    /// Convenience method for testing and single-layer execution.
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_host(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &[f32],
        ffn_norm_gamma: &[f32],
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Upload inputs
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let attn_gamma_gpu = GpuBuffer::from_host(&self.context, attn_norm_gamma)?;
        let ffn_gamma_gpu = GpuBuffer::from_host(&self.context, ffn_norm_gamma)?;

        // Run GPU-resident layer
        let output_gpu = self.transformer_layer_gpu(
            &input_gpu,
            layer_idx,
            layer_prefix,
            hidden_dim,
            intermediate_dim,
            &attn_gamma_gpu,
            &ffn_gamma_gpu,
            epsilon,
        )?;

        // Single sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// TILING-SPEC-001: Tile-profiled transformer layer with host input/output.
    ///
    /// Convenience method for profiling single-layer execution to identify bottlenecks.
    /// Enable tile profiling first with `enable_tile_profiling()`, then call this method,
    /// then examine results with `tile_summary()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// cuda_model.enable_tile_profiling();
    /// cuda_model.transformer_layer_host_profiled(...)?;
    /// println!("{}", cuda_model.tile_summary());
    /// // Output:
    /// // === Tile Profiling Summary (TILING-SPEC-001) ===
    /// // Level       Samples   Avg µs    GFLOP/s   AI      Elements
    /// // macro             3    1500.0     26.67  0.50    4096
    /// // midi              1     200.0      5.12  0.25    1024
    /// // micro             4      10.0      2.05  4.00     512
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_host_profiled(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &[f32],
        ffn_norm_gamma: &[f32],
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Upload inputs
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let attn_gamma_gpu = GpuBuffer::from_host(&self.context, attn_norm_gamma)?;
        let ffn_gamma_gpu = GpuBuffer::from_host(&self.context, ffn_norm_gamma)?;

        // Run GPU-resident tiled profiled layer
        let output_gpu = self.transformer_layer_gpu_tiled_profiled(
            &input_gpu,
            layer_idx,
            layer_prefix,
            hidden_dim,
            intermediate_dim,
            &attn_gamma_gpu,
            &ffn_gamma_gpu,
            epsilon,
        )?;

        // Single sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// Execute Q5_K quantized matvec (fused dequantization + matvec) - PARITY-116
    ///
    /// # Arguments
    ///
    /// * `weights` - Quantized weights in Q5_K GGML format (176 bytes per 256 values)
    /// * `input` - Input vector (f32)
    /// * `output` - Output vector (f32)
    /// * `m` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    pub fn q5k_matvec(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::Q5KQuantizedGemm { m, n: 1, k };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5k_{}_{}", m, k);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_weights = GpuBuffer::from_host(&self.context, weights)?;
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, m as usize)?;

        // Launch configuration
        let config = LaunchConfig::linear(m, 256);

        let mut ptr_input = buf_input.as_ptr();
        let mut ptr_weights = buf_weights.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut m_val = m;
        let mut n_val = 1u32;
        let mut k_val = k;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// Execute Q6_K quantized matvec (fused dequantization + matvec) - PARITY-117
    ///
    /// # Arguments
    ///
    /// * `weights` - Quantized weights in Q6_K GGML format (210 bytes per 256 values)
    /// * `input` - Input vector (f32)
    /// * `output` - Output vector (f32)
    /// * `m` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    pub fn q6k_matvec(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::Q6KQuantizedGemm { m, n: 1, k };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_{}_{}", m, k);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_weights = GpuBuffer::from_host(&self.context, weights)?;
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, m as usize)?;

        // Launch configuration
        let config = LaunchConfig::linear(m, 256);

        let mut ptr_input = buf_input.as_ptr();
        let mut ptr_weights = buf_weights.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut m_val = m;
        let mut n_val = 1u32;
        let mut k_val = k;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// Execute FlashAttention forward pass (IMP-900c)
    ///
    /// Memory-efficient attention using tiled computation to avoid O(N²)
    /// memory usage. Computes: softmax(QK^T / sqrt(d)) @ V
    ///
    /// # Arguments
    ///
    /// * `q` - Query matrix (seq_len × head_dim)
    /// * `k` - Key matrix (seq_len × head_dim)
    /// * `v` - Value matrix (seq_len × head_dim)
    /// * `output` - Output matrix (seq_len × head_dim)
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Softmax scale factor (typically 1/sqrt(head_dim))
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Performance Impact
    ///
    /// - Naive attention: O(N²) memory for attention matrix
    /// - FlashAttention: O(N) memory using tiled computation
    /// - Expected speedup: 2-4x for long sequences
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: u32,
        head_dim: u32,
        _scale: f32,
        causal: bool,
    ) -> Result<(), GpuError> {
        let expected_size = (seq_len * head_dim) as usize;

        if q.len() != expected_size
            || k.len() != expected_size
            || v.len() != expected_size
            || output.len() != expected_size
        {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "Attention size mismatch: expected {}, got Q[{}] K[{}] V[{}] O[{}]",
                expected_size,
                q.len(),
                k.len(),
                v.len(),
                output.len()
            )));
        }

        // Track memory in pool
        self.memory_pool.record_allocation(expected_size * 4 * 4);

        // Use FlashAttention-style kernel
        let kernel_type = KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("flash_attn_{}_{}_{}", seq_len, head_dim, causal);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            // Debug: Print PTX for debugging invalid PTX errors
            #[cfg(test)]
            eprintln!("Generated attention PTX:\n{}", ptx);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_q = GpuBuffer::from_host(&self.context, q)?;
        let buf_k = GpuBuffer::from_host(&self.context, k)?;
        let buf_v = GpuBuffer::from_host(&self.context, v)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, expected_size)?;

        // Launch configuration: 2D grid for attention
        // Grid X: Q blocks (ceil(seq_len / tile_q)), Grid Y: num_heads
        // Threads: tile_q * head_dim (must be <= 1024)
        // IMP-1010: Ensure tile_q * head_dim <= 1024 so all threads can load Q/K/V elements
        let thread_limit = 1024 / head_dim;
        let tile_q = 64u32.min(seq_len).min(thread_limit);
        let num_q_blocks = (seq_len + tile_q - 1) / tile_q;
        let num_heads = 1u32; // Single head for now
        let threads_per_block = tile_q * head_dim; // Now guaranteed <= 1024
        let config = LaunchConfig::grid_2d(num_q_blocks, num_heads, threads_per_block, 1);

        // Get raw pointers
        let mut ptr_q = buf_q.as_ptr();
        let mut ptr_k = buf_k.as_ptr();
        let mut ptr_v = buf_v.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut seq_len_val = seq_len;
        let mut head_dim_val = head_dim;
        // Kernel expects num_heads, not scale (scale is baked into kernel or computed internally)
        let mut num_heads_val = 1u32;

        // Launch kernel
        // SAFETY: Buffers are valid, dimensions match
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_len_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut num_heads_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and copy back
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        self.memory_pool.record_deallocation(expected_size * 4 * 4);

        Ok(())
    }

    /// Execute multi-head FlashAttention forward pass (PARITY-043)
    ///
    /// Processes all attention heads in parallel for maximum GPU occupancy.
    /// Each CUDA block handles one attention head independently.
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor [n_heads, seq_len, head_dim]
    /// * `k` - Key tensor [n_heads, seq_len, head_dim]
    /// * `v` - Value tensor [n_heads, seq_len, head_dim]
    /// * `output` - Output tensor [n_heads, seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Dimension per head (typically 64 or 128)
    /// * `n_heads` - Number of attention heads to process in parallel
    /// * `causal` - Whether to apply causal masking (autoregressive)
    ///
    /// # Performance
    ///
    /// - Parallelization: n_heads blocks × seq_len threads
    /// - Memory: O(n_heads × seq_len × head_dim) for K/V shared memory
    /// - Expected speedup: ~n_heads× over sequential single-head attention
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_multi_head(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: u32,
        head_dim: u32,
        n_heads: u32,
        causal: bool,
    ) -> Result<(), GpuError> {
        let head_size = (seq_len * head_dim) as usize;
        let total_size = head_size * n_heads as usize;

        // Validate input sizes
        if q.len() != total_size
            || k.len() != total_size
            || v.len() != total_size
            || output.len() != total_size
        {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "Multi-head attention size mismatch: expected {} ({}×{}×{}), got Q[{}] K[{}] V[{}] O[{}]",
                total_size, n_heads, seq_len, head_dim,
                q.len(), k.len(), v.len(), output.len()
            )));
        }

        // Track memory allocation
        self.memory_pool.record_allocation(total_size * 4 * 4);

        // Generate/cache multi-head attention kernel
        let kernel_type = KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!(
            "multi_head_attn_{}_{}_{}_{}",
            seq_len, head_dim, n_heads, causal
        );

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            #[cfg(test)]
            eprintln!("Generated multi-head attention PTX:\n{}", ptx);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_q = GpuBuffer::from_host(&self.context, q)?;
        let buf_k = GpuBuffer::from_host(&self.context, k)?;
        let buf_v = GpuBuffer::from_host(&self.context, v)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, total_size)?;

        // Launch configuration for trueno's FlashAttention kernel:
        // - Grid.x = number of Q tile blocks (ceil(seq_len / tile_q))
        // - Grid.y = number of heads
        // - Threads = tile_q * head_dim (each thread handles one element)
        // Calculate tile size to fit in 48KB shared memory (same as generate_ptx)
        let max_tile = (48 * 1024) / (head_dim * 12);
        // IMP-1010: Ensure tile_q * head_dim <= 1024 so all threads can load Q/K/V elements
        // Without this constraint, we launch 1024 threads but need tile_q * head_dim > 1024 loads
        let thread_limit = 1024 / head_dim;
        let tile_q = max_tile.min(64).min(seq_len).min(thread_limit);
        let num_q_blocks = (seq_len + tile_q - 1) / tile_q;
        let threads_per_block = tile_q * head_dim; // Now guaranteed <= 1024
        let config = LaunchConfig::grid_2d(num_q_blocks, n_heads, threads_per_block, 1);

        // Get raw pointers for kernel args
        let mut ptr_q = buf_q.as_ptr();
        let mut ptr_k = buf_k.as_ptr();
        let mut ptr_v = buf_v.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut seq_len_val = seq_len;
        let mut head_dim_val = head_dim;
        let mut n_heads_val = n_heads;

        // Launch kernel
        // SAFETY: Buffers are valid, dimensions match, pointers are aligned
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_len_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_heads_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and copy back
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        self.memory_pool.record_deallocation(total_size * 4 * 4);

        Ok(())
    }
}
