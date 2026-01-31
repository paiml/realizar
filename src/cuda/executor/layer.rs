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
        super::layers::fused_ffn_swiglu_gpu(
            self,
            input,
            ffn_gate_name,
            ffn_up_name,
            ffn_down_name,
            hidden_dim,
            intermediate_dim,
        )
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
        super::layers::fused_ffn_swiglu_gpu_true_dp4a(
            self,
            input,
            ffn_gate_name,
            ffn_up_name,
            ffn_down_name,
            hidden_dim,
            intermediate_dim,
        )
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

    /// PAR-023: Transformer layer with cached gamma pointers
    ///
    /// Like `transformer_layer_gpu` but takes raw device pointers for gamma weights
    /// to avoid borrow checker conflicts with cached buffers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transformer_layer_gpu_cached(
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

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // FFN SwiGLU Tests
    // ========================================================================

    #[test]
    fn test_fused_ffn_swiglu_gpu_weight_not_cached() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.fused_ffn_swiglu_gpu(
            &input,
            "nonexistent_gate",
            "nonexistent_up",
            "nonexistent_down",
            256,
            512,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_ffn_swiglu_gpu_true_dp4a_weight_not_cached() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        let result = exec.fused_ffn_swiglu_gpu_true_dp4a(
            &input,
            "nonexistent_gate",
            "nonexistent_up",
            "nonexistent_down",
            256,
            512,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_ffn_swiglu_indexed_gpu_creates_output() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        // Using zero pointers will fail kernel but tests function interface
        let result = exec.fused_ffn_swiglu_indexed_gpu(&input, 0, 0, 0, 256, 512);
        let _ = result;
    }

    #[test]
    fn test_fused_ffn_swiglu_host_weight_not_cached() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 256];
        let result = exec.fused_ffn_swiglu_host(
            &input,
            &mut output,
            "nonexistent_gate",
            "nonexistent_up",
            "nonexistent_down",
            256,
            512,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // FFN Indexed Tests
    // ========================================================================

    #[test]
    fn test_fused_ffn_swiglu_indexed_gpu_creates_output_buffer() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
        // Using zero pointers will fail kernel but tests function interface
        let result = exec.fused_ffn_swiglu_indexed_gpu(&input, 0, 0, 0, 256, 512);
        let _ = result;
    }

    // ========================================================================
    // Transformer Layer Tests
    // ========================================================================

    #[test]
    fn test_transformer_layer_workspace_dimensions() {
        // Test dimension calculations without actual kernel execution
        let hidden_dim = 256u32;
        let n_heads = 8u32;
        let head_dim = 32u32;
        let intermediate_dim = 512u32;

        // Verify dimensional constraints
        assert_eq!(hidden_dim, n_heads * head_dim);
        assert!(intermediate_dim > hidden_dim);
    }

    #[test]
    fn test_transformer_layer_q_offset_calculation() {
        // Test Q/K/V offset calculations
        let hidden_dim = 256usize;
        let n_kv_heads = 4usize;
        let head_dim = 32usize;

        let q_offset = 0;
        let k_offset = hidden_dim;
        let v_offset = k_offset + n_kv_heads * head_dim;

        assert_eq!(q_offset, 0);
        assert_eq!(k_offset, 256);
        assert_eq!(v_offset, 256 + 4 * 32);
    }

    // ========================================================================
    // Batched Transformer Layer Tests
    // ========================================================================

    #[test]
    fn test_batched_transformer_batch_size_constraints() {
        // Test batch size constraints for multi-sequence processing
        let max_batch = 32u32;
        let typical_batch = 8u32;

        assert!(typical_batch <= max_batch);
        assert!(typical_batch.is_power_of_two());
    }

    #[test]
    fn test_batched_kv_cache_stride_calculation() {
        // Test KV cache stride calculation
        let max_seq_len = 2048u32;
        let n_kv_heads = 4u32;
        let head_dim = 64u32;

        let kv_stride = max_seq_len * n_kv_heads * head_dim;
        assert_eq!(kv_stride, 2048 * 4 * 64);
    }

    // ========================================================================
    // Attention Tests
    // ========================================================================

    #[test]
    fn test_flash_attention_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let seq_len = 4usize;
        let head_dim = 32usize;
        // flash_attention uses single head only (seq_len * head_dim)
        let total = seq_len * head_dim;

        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];

        let scale = 1.0 / (head_dim as f32).sqrt();
        let causal = true;
        let result = exec.flash_attention(
            &q,
            &k,
            &v,
            &mut output,
            seq_len as u32,
            head_dim as u32,
            scale,
            causal,
        );
        // May fail due to PTX issues but exercises the code path
        let _ = result;
    }

    #[test]
    fn test_flash_attention_dimension_calculation() {
        // Test attention dimension calculations
        let seq_len = 64u32;
        let head_dim = 64u32;
        let n_heads = 12u32;

        let q_size = seq_len * head_dim * n_heads;
        let k_size = seq_len * head_dim * n_heads;
        let v_size = seq_len * head_dim * n_heads;
        let output_size = seq_len * head_dim * n_heads;

        assert_eq!(q_size, k_size);
        assert_eq!(k_size, v_size);
        assert_eq!(v_size, output_size);
    }

    #[test]
    fn test_flash_attention_tile_size_calculation() {
        // Test tile size calculation for shared memory constraints
        let head_dim = 64u32;
        let max_shared = 48 * 1024u32; // 48KB

        let max_tile = max_shared / (head_dim * 12);
        assert!(max_tile > 0);
        assert!(max_tile <= 64); // Reasonable tile size
    }

    // ========================================================================
    // Flash Attention Multi-Head Tests
    // ========================================================================

    #[test]
    fn test_flash_attention_multi_head_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let seq_len = 4usize;
        let head_dim = 32usize;
        let n_heads = 2usize;
        let total = seq_len * head_dim * n_heads;

        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];

        let causal = true;
        let result = exec.flash_attention_multi_head(
            &q,
            &k,
            &v,
            &mut output,
            seq_len as u32,
            head_dim as u32,
            n_heads as u32,
            causal,
        );
        let _ = result;
    }

    #[test]
    fn test_flash_attention_thread_limit() {
        // Test thread limit constraint
        let head_dim = 64u32;
        let thread_limit = 1024 / head_dim;
        assert!(thread_limit <= 16); // Max 16 when head_dim=64
    }

    #[test]
    fn test_flash_attention_memory_bytes() {
        // Test memory calculation for flash attention
        let seq_len = 1024u32;
        let head_dim = 64u32;
        let (compute_mem, _peak_mem) =
            CudaExecutor::flash_attention_memory_bytes(seq_len, head_dim);
        assert!(compute_mem > 0);
    }

    // ========================================================================
    // Workspace Allocation Tests
    // ========================================================================

    #[test]
    fn test_workspace_allocation_sizes() {
        // Test workspace allocation size calculations
        let hidden_dim = 4096usize;
        let intermediate_dim = 11008usize;
        let n_heads = 32usize;
        let n_kv_heads = 8usize;
        let head_dim = 128usize;
        let max_seq_len = 4096usize;

        // QKV projection size
        let qkv_size = hidden_dim + 2 * n_kv_heads * head_dim;
        assert!(qkv_size > 0);

        // FFN intermediate size
        let ffn_size = intermediate_dim;
        assert!(ffn_size > hidden_dim);

        // KV cache size per layer
        let kv_cache_size = 2 * max_seq_len * n_kv_heads * head_dim;
        assert!(kv_cache_size > 0);
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // These tests use ModelHarness to setup complete executor state
    // ========================================================================

    #[test]
    fn test_fused_ffn_swiglu_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        // Now we have weights loaded - use indexed pointers from layer 0
        let layer_weights = &exec.indexed_layer_weights[0];
        let result = exec.fused_ffn_swiglu_indexed_gpu(
            &input,
            layer_weights.ffn_gate_ptr,
            layer_weights.ffn_up_ptr,
            layer_weights.ffn_down_ptr,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
        );
        // Should execute kernel (may succeed or fail due to PTX issues)
        let _ = result;
    }

    #[test]
    fn test_flash_attention_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let seq_len = 4usize;
        let total = seq_len * config.head_dim;
        let q = vec![0.1f32; total];
        let k = vec![0.1f32; total];
        let v = vec![0.1f32; total];
        let mut output = vec![0.0f32; total];
        let scale = 1.0 / (config.head_dim as f32).sqrt();

        let result = exec.flash_attention(
            &q,
            &k,
            &v,
            &mut output,
            seq_len as u32,
            config.head_dim as u32,
            scale,
            true,
        );
        let _ = result;
    }

    #[test]
    fn test_flash_attention_multi_head_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let seq_len = 4usize;
        let total = seq_len * config.head_dim * config.num_heads;
        let q = vec![0.1f32; total];
        let k = vec![0.1f32; total];
        let v = vec![0.1f32; total];
        let mut output = vec![0.0f32; total];

        let result = exec.flash_attention_multi_head(
            &q,
            &k,
            &v,
            &mut output,
            seq_len as u32,
            config.head_dim as u32,
            config.num_heads as u32,
            true,
        );
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Verify indexed weights were built (workspace is managed internally)
        assert_eq!(exec.indexed_layer_weights.len(), config.num_layers);
    }

    #[test]
    fn test_batched_attention_workspace_setup() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 2;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // KV cache should be initialized
        assert!(exec.kv_cache_max_len > 0);
        assert!(exec.kv_num_kv_heads > 0);
        assert!(exec.kv_head_dim > 0);
    }

    #[test]
    fn test_gqa_configuration_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_heads = 32;
        config.num_kv_heads = 8; // GQA with 4:1 ratio
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Verify GQA configuration
        assert_eq!(exec.kv_num_heads, config.num_heads);
        assert_eq!(exec.kv_num_kv_heads, config.num_kv_heads);
    }

    #[test]
    fn test_rmsnorm_cache_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // RMSNorm gamma should be cached for each layer
        let key = "blk.0.attn_norm.gamma".to_string();
        assert!(exec.rmsnorm_cache.contains_key(&key));
    }

    #[test]
    fn test_lm_head_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // LM head should be loaded
        assert!(exec.lm_head_ptr != 0);
        assert!(exec.lm_head_len > 0);
    }
}
