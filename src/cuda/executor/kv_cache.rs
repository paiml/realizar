//! KV Cache management for GPU-resident inference
//!
//! This module implements:
//! - PAR-018: GPU-Resident KV Cache initialization
//! - PAR-021: GQA (Grouped Query Attention) support
//! - PAR-119: Batched KV cache for multi-sequence processing
//! - QWEN-007: Q8 quantized KV cache for 4x memory reduction
//! - Cache reset, rollback, and continuation

use super::*;
use crate::quantize::Q8_0Block;

impl CudaExecutor {
    // ========================================================================
    // PAR-018: GPU-Resident KV Cache for Incremental Attention
    // ========================================================================

    /// Initialize GPU KV cache for a given number of layers and max sequence length
    ///
    /// Pre-allocates GPU memory for all layers to avoid allocation during inference.
    /// Call this once at model load time with the expected max sequence length.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA, <= num_heads)
    /// * `head_dim` - Dimension per head
    /// * `max_len` - Maximum sequence length to support
    #[allow(clippy::too_many_arguments)]
    pub fn init_kv_cache_gpu(
        &mut self,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_len: usize,
    ) -> Result<(), GpuError> {
        // Store dimensions (PAR-021: track both Q heads and KV heads for GQA)
        self.kv_num_heads = num_heads;
        self.kv_num_kv_heads = num_kv_heads;
        self.kv_head_dim = head_dim;
        self.kv_cache_max_len = max_len;

        // Pre-allocate K and V buffers for each layer
        // PAR-021 GQA: Layout is [num_kv_heads, max_len, head_dim]
        let buffer_size = num_kv_heads * max_len * head_dim;

        for layer_idx in 0..num_layers {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);

            // Allocate if not already present
            if !self.kv_cache_gpu.contains_key(&k_key) {
                let k_buf = GpuBuffer::<f32>::new(&self.context, buffer_size)?;
                let v_buf = GpuBuffer::<f32>::new(&self.context, buffer_size)?;
                self.kv_cache_gpu.insert(k_key, k_buf);
                self.kv_cache_gpu.insert(v_key, v_buf);
                self.kv_cache_lengths.insert(layer_idx, 0);
            }
        }

        let total_bytes = num_layers * 2 * buffer_size * 4;
        self.memory_pool.record_allocation(total_bytes);

        Ok(())
    }

    /// PAR-119: Initialize batched KV caches for true multi-sequence batching
    ///
    /// Allocates M separate KV caches per layer, enabling parallel attention
    /// across M sequences. This eliminates the sequential attention bottleneck
    /// identified in Five-Whys analysis.
    ///
    /// Memory layout per layer:
    /// - K cache: [M, num_kv_heads, max_len, head_dim]
    /// - V cache: same
    /// - Stride: num_kv_heads × max_len × head_dim (per sequence)
    pub fn init_batched_kv_cache_gpu(
        &mut self,
        num_layers: usize,
        batch_size: usize,
    ) -> Result<(), GpuError> {
        // PAR-129: Extended to M=32 via 4-warp kernel
        if batch_size == 0 || batch_size > 32 {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-119: batch_size must be 1-32, got {}",
                batch_size
            )));
        }

        // Must have regular KV cache initialized first (to get dimensions)
        if self.kv_cache_max_len == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-119: Must call init_kv_cache_gpu before init_batched_kv_cache_gpu".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        // Per-sequence stride
        let stride = num_kv_heads * max_len * head_dim;
        self.batched_kv_stride = stride;

        // M× larger buffer per layer
        let buffer_size = batch_size * stride;

        // PAR-119: Check if we need to reallocate (batch_size changed)
        let need_realloc = batch_size > self.batched_kv_allocated_batch;
        if need_realloc {
            // Clear existing caches - they're too small
            self.batched_kv_k_caches.clear();
            self.batched_kv_v_caches.clear();
        }

        for layer_idx in 0..num_layers {
            // Allocate if not already present or after realloc
            if !self.batched_kv_k_caches.contains_key(&layer_idx) {
                let k_buf = GpuBuffer::<f32>::new(&self.context, buffer_size)?;
                let v_buf = GpuBuffer::<f32>::new(&self.context, buffer_size)?;
                self.batched_kv_k_caches.insert(layer_idx, k_buf);
                self.batched_kv_v_caches.insert(layer_idx, v_buf);
            }
        }

        // Track allocated batch size
        self.batched_kv_allocated_batch = batch_size;

        // Initialize per-sequence lengths (all start at 0)
        self.batched_kv_lengths = vec![0; batch_size];

        // Allocate GPU pointer arrays for batched attention
        self.batched_k_ptrs = Some(GpuBuffer::new(&self.context, batch_size)?);
        self.batched_v_ptrs = Some(GpuBuffer::new(&self.context, batch_size)?);
        self.batched_seq_lens_gpu = Some(GpuBuffer::new(&self.context, batch_size)?);

        let total_bytes = num_layers * 2 * buffer_size * 4 + batch_size * 24; // caches + ptr arrays
        self.memory_pool.record_allocation(total_bytes);

        eprintln!(
            "[PAR-119] Initialized batched KV cache: {} layers × {} sequences, stride={}, total={}MB",
            num_layers,
            batch_size,
            stride,
            total_bytes / (1024 * 1024)
        );

        Ok(())
    }

    /// PAR-119: Reset batched KV caches for new generation
    pub fn reset_batched_kv_cache_gpu(&mut self) {
        for len in &mut self.batched_kv_lengths {
            *len = 0;
        }
    }

    /// Clear KV cache for a new generation (reset sequence position to 0)
    pub fn reset_kv_cache_gpu(&mut self) {
        for len in self.kv_cache_lengths.values_mut() {
            *len = 0;
        }
    }

    /// PAR-105: Rollback KV cache to a specific position (for speculative decode)
    ///
    /// This allows undoing speculative tokens without losing the prefill history.
    /// Unlike reset_kv_cache_gpu, this preserves KV values up to `position`.
    pub fn rollback_kv_cache_gpu(&mut self, position: usize) {
        for len in self.kv_cache_lengths.values_mut() {
            if *len > position {
                *len = position;
            }
        }
    }

    /// PAR-060: Set RoPE theta (rotary position embedding base frequency)
    ///
    /// This must be called after init_kv_cache_gpu with the model's rope_theta value.
    /// Common values: 10000.0 (LLaMA), 1000000.0 (Qwen2, long context models)
    pub fn set_rope_theta(&mut self, theta: f32) {
        self.rope_theta = theta;
    }

    /// CORRECTNESS-011: Set RoPE type (0=NORM adjacent pairs, 2=NEOX split halves)
    ///
    /// Qwen2.5 models use rope_type=2 (NEOX style).
    pub fn set_rope_type(&mut self, rope_type: u32) {
        self.rope_type = rope_type;
    }

    /// PAR-060: Apply RoPE to Q and K vectors (CPU fallback, will be GPU-accelerated later)
    ///
    /// Rotates Q and K by position-dependent angles to inject positional information.
    /// This is called before attention to enable position-aware attention.
    fn apply_rope_to_buffer(&self, buffer: &mut [f32], num_heads: usize, position: usize) {
        let head_dim = self.kv_head_dim;
        let half_dim = head_dim / 2;

        for h in 0..num_heads {
            let head_start = h * head_dim;

            for i in 0..half_dim {
                let freq = 1.0 / self.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx1 = head_start + i;
                let idx2 = head_start + i + half_dim;

                if idx2 < buffer.len() {
                    let x1 = buffer[idx1];
                    let x2 = buffer[idx2];
                    buffer[idx1] = x1 * cos_val - x2 * sin_val;
                    buffer[idx2] = x1 * sin_val + x2 * cos_val;
                }
            }
        }
    }

    /// Get current KV cache length for a layer
    #[must_use]
    pub fn kv_cache_len(&self, layer_idx: usize) -> usize {
        self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0)
    }

    /// Check if GPU KV cache is initialized (PAR-020)
    #[must_use]
    pub fn has_kv_cache_gpu(&self) -> bool {
        self.kv_cache_max_len > 0
    }

    /// Append new K/V to GPU cache and run flash attention
    ///
    /// This is the main incremental attention method for autoregressive decoding.
    /// Only the new K/V vectors are transferred to GPU (hidden_dim floats each),
    /// avoiding the O(seq_len × hidden_dim) transfer that was the main bottleneck.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Transformer layer index
    /// * `q` - Query vector for current position [hidden_dim]
    /// * `current_k` - Key vector for current position [hidden_dim]
    /// * `current_v` - Value vector for current position [hidden_dim]
    /// * `output` - Output buffer [hidden_dim]
    ///
    /// # Returns
    ///
    /// New total sequence length after appending
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_cached(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        output: &mut [f32],
    ) -> Result<usize, GpuError> {
        let num_heads = self.kv_num_heads;
        let head_dim = self.kv_head_dim;
        let hidden_dim = num_heads * head_dim;
        let max_len = self.kv_cache_max_len;

        // Validate dimensions
        if q.len() != hidden_dim || current_k.len() != hidden_dim || current_v.len() != hidden_dim {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-018: dimension mismatch - expected {}, got Q[{}] K[{}] V[{}]",
                hidden_dim,
                q.len(),
                current_k.len(),
                current_v.len()
            )));
        }

        // Get current cache length and check bounds
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let new_len = cache_len + 1;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-018: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // Get cache buffer keys
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // Reorganize current_k/v from [hidden_dim] to [num_heads, 1, head_dim]
        // and upload to correct position in GPU cache
        // GPU layout: [num_heads, max_len, head_dim]
        // Position for new data: head * (max_len * head_dim) + cache_len * head_dim
        {
            let k_buf = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-018: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;

            // Copy each head's K portion to correct position
            for head in 0..num_heads {
                let src_offset = head * head_dim;
                let dst_offset = head * (max_len * head_dim) + cache_len * head_dim;
                k_buf
                    .copy_from_host_at(&current_k[src_offset..src_offset + head_dim], dst_offset)?;
            }
        }

        {
            let v_buf = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-018: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;

            // Copy each head's V portion to correct position
            for head in 0..num_heads {
                let src_offset = head * head_dim;
                let dst_offset = head * (max_len * head_dim) + cache_len * head_dim;
                v_buf
                    .copy_from_host_at(&current_v[src_offset..src_offset + head_dim], dst_offset)?;
            }
        }

        // Update cache length
        self.kv_cache_lengths.insert(layer_idx, new_len);

        // For GPU-only attention, we need to compact K/V from max_len layout to new_len layout
        // This is necessary because the flash attention kernel expects contiguous seq_len data
        //
        // Current GPU layout: [num_heads, max_len, head_dim] with only new_len positions filled
        // Required layout:    [num_heads, new_len, head_dim] contiguous
        //
        // Options:
        // A) D2D copy to compact buffers (faster than D2H+H2D for long sequences)
        // B) Use padded kernel that handles max_len with actual_len mask (requires kernel change)
        // C) For now: read back and use existing flash_attention_multi_head (baseline)
        //
        // PAR-018 Phase 1: Use compacted read approach for correctness
        // PAR-019 (future): Implement D2D compaction or padded kernel for full GPU residency

        let tensor_size = num_heads * new_len * head_dim;

        // Build Q tensor on CPU: [num_heads, new_len, head_dim]
        // Q is the same for all positions (broadcasting optimization possible in future)
        let mut q_full = vec![0.0f32; tensor_size];
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let gpu_head_offset = head * new_len * head_dim;
            for pos in 0..new_len {
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                q_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&q[head_offset..head_offset + head_dim]);
            }
        }

        // Read compacted K/V from GPU cache
        // Uses new copy_to_host_at for partial reads
        let mut k_data = vec![0.0f32; tensor_size];
        let mut v_data = vec![0.0f32; tensor_size];

        {
            let k_buf = self
                .kv_cache_gpu
                .get(&k_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("KV cache K not found".to_string()))?;
            let v_buf = self
                .kv_cache_gpu
                .get(&v_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("KV cache V not found".to_string()))?;

            for head in 0..num_heads {
                let gpu_head_offset = head * max_len * head_dim;
                let out_head_offset = head * new_len * head_dim;

                // Batch read: read new_len contiguous positions per head
                // This is more efficient than per-position reads
                k_buf.copy_to_host_at(
                    &mut k_data[out_head_offset..out_head_offset + new_len * head_dim],
                    gpu_head_offset,
                )?;
                v_buf.copy_to_host_at(
                    &mut v_data[out_head_offset..out_head_offset + new_len * head_dim],
                    gpu_head_offset,
                )?;
            }
        }

        // Run attention.
        // Flash attention requires seq_len >= head_dim (trueno-gpu AttentionKernel
        // shared memory layout limitation).  Fall back to CPU softmax attention
        // for small seq_len (common during early autoregressive generation).
        let mut output_full = vec![0.0f32; tensor_size];
        if new_len >= head_dim {
            self.flash_attention_multi_head(
                &q_full,
                &k_data,
                &v_data,
                &mut output_full,
                new_len as u32,
                head_dim as u32,
                num_heads as u32,
                true, // causal
            )?;
        } else {
            // CPU fallback: standard scaled dot-product attention per head
            let scale = 1.0 / (head_dim as f32).sqrt();
            for head in 0..num_heads {
                let ho = head * new_len * head_dim;
                for row in 0..new_len {
                    // Compute attention scores: Q[row] · K[col] for col <= row (causal)
                    let mut scores = vec![f32::NEG_INFINITY; new_len];
                    for col in 0..=row {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot +=
                                q_full[ho + row * head_dim + d] * k_data[ho + col * head_dim + d];
                        }
                        scores[col] = dot * scale;
                    }
                    // Softmax
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
                    // Weighted sum of V
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for col in 0..=row {
                            let w = ((scores[col] - max_s).exp()) / exp_sum;
                            acc += w * v_data[ho + col * head_dim + d];
                        }
                        output_full[ho + row * head_dim + d] = acc;
                    }
                }
            }
        }

        // Extract output for last position, reorganize to [hidden_dim]
        let last_pos = new_len - 1;
        for head in 0..num_heads {
            let gpu_offset = head * new_len * head_dim + last_pos * head_dim;
            let out_offset = head * head_dim;
            output[out_offset..out_offset + head_dim]
                .copy_from_slice(&output_full[gpu_offset..gpu_offset + head_dim]);
        }

        Ok(new_len)
    }

    /// PAR-020: True GPU-resident incremental attention for M=1 autoregressive decoding
    ///
    /// Unlike `flash_attention_cached` which does D2H+H2D roundtrips, this method:
    /// 1. Appends new K/V to GPU-resident cache (H2D, small transfer)
    /// 2. Launches IncrementalAttentionKernel directly on GPU buffers
    /// 3. Downloads only the output (D2H, small transfer)
    ///
    /// Target performance: Eliminate ~66 MB/token transfer overhead for TinyLlama
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Transformer layer index
    /// * `q` - Query vector for current position [num_heads, head_dim]
    /// * `current_k` - Key vector for current position [num_heads, head_dim]
    /// * `current_v` - Value vector for current position [num_heads, head_dim]
    /// * `output` - Output buffer [num_heads, head_dim]
    ///
    /// # Returns
    ///
    /// New total sequence length after appending
    #[allow(clippy::too_many_arguments)]
    pub fn incremental_attention_gpu(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        output: &mut [f32],
    ) -> Result<usize, GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let q_dim = num_heads * head_dim; // Q/output dimension
        let kv_dim = num_kv_heads * head_dim; // K/V dimension (smaller for GQA)
        let max_len = self.kv_cache_max_len;

        // PAR-021 GQA: Q has num_heads dimensions, K/V have num_kv_heads dimensions
        if q.len() != q_dim {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-021: Q dimension mismatch - expected {}, got {}",
                q_dim,
                q.len()
            )));
        }
        if current_k.len() != kv_dim || current_v.len() != kv_dim {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-021: K/V dimension mismatch - expected {}, got K[{}] V[{}]",
                kv_dim,
                current_k.len(),
                current_v.len()
            )));
        }

        // Get current cache length and check bounds
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let new_len = cache_len + 1;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-020: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // Get cache buffer keys
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // Append new K/V to GPU cache
        // PAR-021 GQA: Layout is [num_kv_heads, max_len, head_dim]
        {
            let k_buf = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-020: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            for kv_head in 0..num_kv_heads {
                let src_offset = kv_head * head_dim;
                let dst_offset = kv_head * (max_len * head_dim) + cache_len * head_dim;
                k_buf
                    .copy_from_host_at(&current_k[src_offset..src_offset + head_dim], dst_offset)?;
            }
        }

        {
            let v_buf = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-020: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            for kv_head in 0..num_kv_heads {
                let src_offset = kv_head * head_dim;
                let dst_offset = kv_head * (max_len * head_dim) + cache_len * head_dim;
                v_buf
                    .copy_from_host_at(&current_v[src_offset..src_offset + head_dim], dst_offset)?;
            }
        }

        // Update cache length
        self.kv_cache_lengths.insert(layer_idx, new_len);

        // Upload Q to GPU (small transfer: num_heads * head_dim floats)
        let mut q_buf = GpuBuffer::<f32>::new(&self.context, q_dim)?;
        q_buf.copy_from_host(q)?;

        // Allocate output buffer (same size as Q)
        let out_buf = GpuBuffer::<f32>::new(&self.context, q_dim)?;

        // Get kernel module (PAR-021: includes n_kv_heads for GQA)
        let kernel_type = KernelType::IncrementalAttention {
            max_seq_len: max_len as u32,
            head_dim: head_dim as u32,
            n_heads: num_heads as u32,
            n_kv_heads: num_kv_heads as u32,
            indirect: false,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let ptx = self.kernels.generate_ptx(&kernel_type);
        let module_key = format!(
            "incremental_attention_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );

        if !self.modules.contains_key(&module_key) {
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(module_key.clone(), module);
        }
        let module = self
            .modules
            .get_mut(&module_key)
            .expect("module just inserted");

        // Get K and V buffer pointers
        let k_buf = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?;
        let v_buf = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?;

        // Launch kernel
        // Grid: (num_heads, 1, 1) - one block per head
        // Block: (32, 1, 1) - one warp per block
        let config = LaunchConfig::grid_2d(num_heads as u32, 1, 32, 1);

        let mut ptr_q = q_buf.as_ptr();
        let mut ptr_k = k_buf.as_ptr();
        let mut ptr_v = v_buf.as_ptr();
        let mut ptr_out = out_buf.as_ptr();
        let mut seq_len_val = new_len as u32;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.compute_stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_len_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and download output
        self.compute_stream.synchronize()?;
        out_buf.copy_to_host(output)?;

        Ok(new_len)
    }

    // ========================================================================
    // QWEN-007: Q8 Quantized KV Cache for 4x Memory Reduction
    // ========================================================================

    /// Initialize Q8 quantized KV cache for memory-constrained inference
    ///
    /// Q8 format stores KV vectors as INT8 with per-block FP32 scales,
    /// reducing memory by 4x compared to FP32 while maintaining quality.
    ///
    /// Memory comparison for 32-layer, 8 KV-head, 128-dim, 4096 seq model:
    /// - FP32: 32 × 2 × 4096 × 8 × 128 × 4 = 8.59 GB
    /// - Q8:   32 × 2 × 4096 × 8 × 128 × 1 + scales = 2.15 GB (4x reduction)
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `head_dim` - Dimension per head (should be divisible by 32)
    /// * `max_len` - Maximum sequence length to support
    #[allow(clippy::too_many_arguments)]
    pub fn init_kv_cache_q8_gpu(
        &mut self,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_len: usize,
    ) -> Result<(), GpuError> {
        // Validate head_dim is divisible by 32 (Q8 block size)
        if !head_dim.is_multiple_of(32) {
            return Err(GpuError::InvalidParameter(format!(
                "QWEN-007: head_dim must be divisible by 32 for Q8 quantization, got {}",
                head_dim
            )));
        }

        // Store dimensions
        self.kv_num_heads = num_heads;
        self.kv_num_kv_heads = num_kv_heads;
        self.kv_head_dim = head_dim;
        self.kv_cache_max_len = max_len;
        self.kv_cache_q8_enabled = true;

        // Q8 buffer sizes
        // Values: [num_kv_heads × max_len × head_dim] as i8 (1 byte each)
        let values_size = num_kv_heads * max_len * head_dim;
        // Scales: [num_kv_heads × max_len × (head_dim / 32)] as f32 (4 bytes each)
        let scales_size = num_kv_heads * max_len * (head_dim / 32);

        for layer_idx in 0..num_layers {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);

            // Allocate Q8 buffers if not already present
            if !self.kv_cache_q8_k.contains_key(&k_key) {
                // Quantized values (i8)
                let k_buf = GpuBuffer::<i8>::new(&self.context, values_size)?;
                let v_buf = GpuBuffer::<i8>::new(&self.context, values_size)?;
                self.kv_cache_q8_k.insert(k_key.clone(), k_buf);
                self.kv_cache_q8_v.insert(v_key.clone(), v_buf);

                // Scales (f32)
                let k_scales_key = format!("kv_{}_k_scales", layer_idx);
                let v_scales_key = format!("kv_{}_v_scales", layer_idx);
                let k_scales = GpuBuffer::<f32>::new(&self.context, scales_size)?;
                let v_scales = GpuBuffer::<f32>::new(&self.context, scales_size)?;
                self.kv_cache_q8_k_scales.insert(k_scales_key, k_scales);
                self.kv_cache_q8_v_scales.insert(v_scales_key, v_scales);

                self.kv_cache_lengths.insert(layer_idx, 0);
            }
        }

        // Total memory: values (1 byte) + scales (4 bytes per 32 values)
        let total_bytes = num_layers * 2 * (values_size + scales_size * 4);
        self.memory_pool.record_allocation(total_bytes);

        Ok(())
    }

    /// Check if Q8 KV cache mode is enabled
    #[must_use]
    pub fn is_kv_cache_q8_enabled(&self) -> bool {
        self.kv_cache_q8_enabled
    }

    /// Get Q8 KV cache memory usage in bytes
    #[must_use]
    pub fn kv_cache_q8_memory_bytes(&self) -> usize {
        if !self.kv_cache_q8_enabled {
            return 0;
        }
        let num_layers = self.kv_cache_lengths.len();
        let values_size = self.kv_num_kv_heads * self.kv_cache_max_len * self.kv_head_dim;
        let scales_size = self.kv_num_kv_heads * self.kv_cache_max_len * (self.kv_head_dim / 32);
        num_layers * 2 * (values_size + scales_size * 4)
    }

    /// Get FP32 equivalent memory for comparison (what would be used without Q8)
    #[must_use]
    pub fn kv_cache_fp32_equivalent_bytes(&self) -> usize {
        let num_layers = self.kv_cache_lengths.len();
        let fp32_size = self.kv_num_kv_heads * self.kv_cache_max_len * self.kv_head_dim * 4;
        num_layers * 2 * fp32_size
    }

    /// Write K/V vectors to Q8 cache at a specific position
    ///
    /// Quantizes FP32 K/V vectors to Q8 format and uploads to GPU.
    /// This is the Phase 2 implementation using CPU quantization.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index
    /// * `position` - Position in the sequence (0-indexed)
    /// * `k` - Key vector [num_kv_heads × head_dim]
    /// * `v` - Value vector [num_kv_heads × head_dim]
    pub fn write_kv_q8(
        &mut self,
        layer_idx: usize,
        position: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), GpuError> {
        if !self.kv_cache_q8_enabled {
            return Err(GpuError::InvalidParameter(
                "Q8 KV cache not enabled. Call init_kv_cache_q8_gpu first.".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        // Validate input sizes
        let expected_size = num_kv_heads * head_dim;
        if k.len() != expected_size || v.len() != expected_size {
            return Err(GpuError::InvalidParameter(format!(
                "K/V size mismatch: expected {}, got K={}, V={}",
                expected_size,
                k.len(),
                v.len()
            )));
        }

        if position >= max_len {
            return Err(GpuError::InvalidParameter(format!(
                "Position {} exceeds max_len {}",
                position, max_len
            )));
        }

        // Quantize K and V to Q8 (CPU-side for Phase 2)
        let blocks_per_head = head_dim / 32;
        let mut k_quants = Vec::with_capacity(expected_size);
        let mut k_scales = Vec::with_capacity(num_kv_heads * blocks_per_head);
        let mut v_quants = Vec::with_capacity(expected_size);
        let mut v_scales = Vec::with_capacity(num_kv_heads * blocks_per_head);

        for head in 0..num_kv_heads {
            for block_idx in 0..blocks_per_head {
                let start = head * head_dim + block_idx * 32;
                let k_block_vals: [f32; 32] = k[start..start + 32].try_into().map_err(|_| {
                    GpuError::InvalidParameter("K block extraction failed".to_string())
                })?;
                let v_block_vals: [f32; 32] = v[start..start + 32].try_into().map_err(|_| {
                    GpuError::InvalidParameter("V block extraction failed".to_string())
                })?;

                let k_block = Q8_0Block::quantize(&k_block_vals);
                let v_block = Q8_0Block::quantize(&v_block_vals);

                k_quants.extend_from_slice(&k_block.quants);
                k_scales.push(k_block.scale);
                v_quants.extend_from_slice(&v_block.quants);
                v_scales.push(v_block.scale);
            }
        }

        // Upload to GPU at the correct position offset
        // Layout: [num_kv_heads, max_len, head_dim]
        // Position offset = position * head_dim for each head
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);
        let k_scales_key = format!("kv_{}_k_scales", layer_idx);
        let v_scales_key = format!("kv_{}_v_scales", layer_idx);

        // Get mutable references to buffers
        let k_buf = self.kv_cache_q8_k.get_mut(&k_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 K cache for layer {} not found", layer_idx))
        })?;
        let k_scales_buf = self.kv_cache_q8_k_scales.get_mut(&k_scales_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "Q8 K scales for layer {} not found",
                layer_idx
            ))
        })?;

        // Upload K quants and scales for each head at the correct position
        for head in 0..num_kv_heads {
            let src_q_offset = head * head_dim;
            let dst_q_offset = head * max_len * head_dim + position * head_dim;
            k_buf.copy_from_host_at(
                &k_quants[src_q_offset..src_q_offset + head_dim],
                dst_q_offset,
            )?;

            let src_s_offset = head * blocks_per_head;
            let dst_s_offset = head * max_len * blocks_per_head + position * blocks_per_head;
            k_scales_buf.copy_from_host_at(
                &k_scales[src_s_offset..src_s_offset + blocks_per_head],
                dst_s_offset,
            )?;
        }

        // Same for V
        let v_buf = self.kv_cache_q8_v.get_mut(&v_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 V cache for layer {} not found", layer_idx))
        })?;
        let v_scales_buf = self.kv_cache_q8_v_scales.get_mut(&v_scales_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "Q8 V scales for layer {} not found",
                layer_idx
            ))
        })?;

        for head in 0..num_kv_heads {
            let src_q_offset = head * head_dim;
            let dst_q_offset = head * max_len * head_dim + position * head_dim;
            v_buf.copy_from_host_at(
                &v_quants[src_q_offset..src_q_offset + head_dim],
                dst_q_offset,
            )?;

            let src_s_offset = head * blocks_per_head;
            let dst_s_offset = head * max_len * blocks_per_head + position * blocks_per_head;
            v_scales_buf.copy_from_host_at(
                &v_scales[src_s_offset..src_s_offset + blocks_per_head],
                dst_s_offset,
            )?;
        }

        // Update cache length
        let current_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        if position >= current_len {
            self.kv_cache_lengths.insert(layer_idx, position + 1);
        }

        Ok(())
    }

    /// Read K/V vectors from Q8 cache at a range of positions
    ///
    /// Downloads Q8 data from GPU and dequantizes to FP32.
    /// This is the Phase 2 implementation using CPU dequantization.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index
    /// * `start_pos` - Start position (inclusive)
    /// * `end_pos` - End position (exclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (K, V) vectors, each [seq_len × num_kv_heads × head_dim]
    pub fn read_kv_q8(
        &self,
        layer_idx: usize,
        start_pos: usize,
        end_pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        if !self.kv_cache_q8_enabled {
            return Err(GpuError::InvalidParameter(
                "Q8 KV cache not enabled. Call init_kv_cache_q8_gpu first.".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let blocks_per_head = head_dim / 32;
        let seq_len = end_pos.saturating_sub(start_pos);

        if end_pos > max_len {
            return Err(GpuError::InvalidParameter(format!(
                "End position {} exceeds max_len {}",
                end_pos, max_len
            )));
        }

        // Get buffer references
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);
        let k_scales_key = format!("kv_{}_k_scales", layer_idx);
        let v_scales_key = format!("kv_{}_v_scales", layer_idx);

        let k_buf = self.kv_cache_q8_k.get(&k_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 K cache for layer {} not found", layer_idx))
        })?;
        let k_scales_buf = self.kv_cache_q8_k_scales.get(&k_scales_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "Q8 K scales for layer {} not found",
                layer_idx
            ))
        })?;
        let v_buf = self.kv_cache_q8_v.get(&v_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 V cache for layer {} not found", layer_idx))
        })?;
        let v_scales_buf = self.kv_cache_q8_v_scales.get(&v_scales_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "Q8 V scales for layer {} not found",
                layer_idx
            ))
        })?;

        // Download quantized data for each position
        let mut k_out = Vec::with_capacity(seq_len * num_kv_heads * head_dim);
        let mut v_out = Vec::with_capacity(seq_len * num_kv_heads * head_dim);

        for pos in start_pos..end_pos {
            for head in 0..num_kv_heads {
                // Download K quants and scales
                let q_offset = head * max_len * head_dim + pos * head_dim;
                let s_offset = head * max_len * blocks_per_head + pos * blocks_per_head;

                let mut k_quants = vec![0i8; head_dim];
                let mut k_scales = vec![0.0f32; blocks_per_head];
                k_buf.copy_to_host_at(&mut k_quants, q_offset)?;
                k_scales_buf.copy_to_host_at(&mut k_scales, s_offset)?;

                let mut v_quants = vec![0i8; head_dim];
                let mut v_scales = vec![0.0f32; blocks_per_head];
                v_buf.copy_to_host_at(&mut v_quants, q_offset)?;
                v_scales_buf.copy_to_host_at(&mut v_scales, s_offset)?;

                // Dequantize
                for block_idx in 0..blocks_per_head {
                    let block_start = block_idx * 32;
                    let block = Q8_0Block {
                        scale: k_scales[block_idx],
                        quants: k_quants[block_start..block_start + 32]
                            .try_into()
                            .expect("32 values"),
                    };
                    k_out.extend_from_slice(&block.dequantize());

                    let block = Q8_0Block {
                        scale: v_scales[block_idx],
                        quants: v_quants[block_start..block_start + 32]
                            .try_into()
                            .expect("32 values"),
                    };
                    v_out.extend_from_slice(&block.dequantize());
                }
            }
        }

        Ok((k_out, v_out))
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
    // KV Cache Initialization Tests
    // ========================================================================

    #[test]
    fn test_init_kv_cache_gpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let n_layers = 4usize;
        let n_kv_heads = 4usize;
        let head_dim = 64usize;
        let max_seq_len = 1024usize;

        let result =
            exec.init_kv_cache_gpu(n_layers, n_kv_heads, head_dim, max_seq_len, n_kv_heads * 4);
        assert!(result.is_ok());

        // Verify cache is initialized
        assert!(exec.has_kv_cache_gpu());
        assert!(exec.kv_cache_max_len > 0);
    }

    #[test]
    fn test_init_batched_kv_cache_gpu_requires_kv_cache() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Without init_kv_cache_gpu first, should fail
        let result = exec.init_batched_kv_cache_gpu(4, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_init_batched_kv_cache_gpu_after_kv_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // First init regular KV cache
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Then init batched cache
        let result = exec.init_batched_kv_cache_gpu(4, 8);
        assert!(result.is_ok());

        // Verify batched cache is initialized
        assert_eq!(exec.batched_kv_allocated_batch, 8);
    }

    // ========================================================================
    // KV Cache State Tests
    // ========================================================================

    #[test]
    fn test_has_kv_cache_gpu_initial_false() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(!exec.has_kv_cache_gpu());
    }

    #[test]
    fn test_kv_cache_len_uninitialized() {
        let Some(exec) = create_executor() else {
            return;
        };
        // Uninitialized layer should return 0
        assert_eq!(exec.kv_cache_len(0), 0);
        assert_eq!(exec.kv_cache_len(99), 0);
    }

    #[test]
    fn test_kv_cache_len_after_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Initially should be 0 for each layer
        assert_eq!(exec.kv_cache_len(0), 0);
        assert_eq!(exec.kv_cache_len(1), 0);
    }

    // ========================================================================
    // KV Cache Reset Tests
    // ========================================================================

    #[test]
    fn test_reset_kv_cache_gpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Reset should succeed
        exec.reset_kv_cache_gpu();

        // All lengths should be 0
        assert_eq!(exec.kv_cache_len(0), 0);
    }

    #[test]
    fn test_reset_batched_kv_cache_gpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Init regular cache first
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();
        exec.init_batched_kv_cache_gpu(4, 8).unwrap();

        exec.reset_batched_kv_cache_gpu();

        // Batched lengths should all be 0
        assert!(exec.batched_kv_lengths.iter().all(|&len| len == 0));
    }

    // ========================================================================
    // RoPE Configuration Tests
    // ========================================================================

    #[test]
    fn test_set_rope_theta() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.set_rope_theta(10000.0);
        assert_eq!(exec.rope_theta, 10000.0);

        exec.set_rope_theta(500000.0); // Longer context
        assert_eq!(exec.rope_theta, 500000.0);
    }

    #[test]
    fn test_set_rope_type() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.set_rope_type(0); // NORM
        assert_eq!(exec.rope_type, 0);

        exec.set_rope_type(2); // NEOX (GPT-NeoX style)
        assert_eq!(exec.rope_type, 2);
    }

    // ========================================================================
    // KV Cache Rollback Tests
    // ========================================================================

    #[test]
    fn test_rollback_kv_cache_gpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Rollback to position 5
        exec.rollback_kv_cache_gpu(5);

        // All layers should be rolled back to 5
        for layer in 0..4 {
            assert!(exec.kv_cache_len(layer) <= 5);
        }
    }

    #[test]
    fn test_rollback_to_zero() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Rollback to 0 should be equivalent to reset
        exec.rollback_kv_cache_gpu(0);

        assert_eq!(exec.kv_cache_len(0), 0);
    }

    // ========================================================================
    // Flash Attention Cached Tests
    // ========================================================================

    #[test]
    fn test_flash_attention_cached_requires_kv_cache() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Without KV cache initialization
        let q = vec![1.0f32; 256];
        let k = vec![1.0f32; 256];
        let v = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 256];

        // flash_attention_cached takes (layer_idx, q, current_k, current_v, output)
        let result = exec.flash_attention_cached(0, &q, &k, &v, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_incremental_attention_gpu_requires_kv_cache() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let q = vec![1.0f32; 256];
        let k = vec![1.0f32; 256];
        let v = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 256];

        // incremental_attention_gpu takes (layer_idx, q, current_k, current_v, output)
        let result = exec.incremental_attention_gpu(0, &q, &k, &v, &mut output);
        assert!(result.is_err());
    }

    // ========================================================================
    // KV Cache Memory Calculation Tests
    // ========================================================================

    #[test]
    fn test_kv_cache_memory_calculation() {
        // Test memory calculation for KV cache
        let n_layers = 32usize;
        let n_kv_heads = 8usize;
        let head_dim = 128usize;
        let max_seq_len = 4096usize;

        let per_layer_bytes = 2 * max_seq_len * n_kv_heads * head_dim * 4; // K + V, f32
        let total_bytes = n_layers * per_layer_bytes;

        // Verify it's a reasonable size (1-10 GB range for large models)
        assert!(total_bytes > 1_000_000_000); // > 1GB
        assert!(total_bytes < 20_000_000_000); // < 20GB
    }

    #[test]
    fn test_gqa_kv_cache_savings() {
        // Test memory savings from GQA (fewer KV heads)
        let n_layers = 32usize;
        let head_dim = 128usize;
        let max_seq_len = 4096usize;

        // MHA: 32 KV heads
        let mha_per_layer = 2 * max_seq_len * 32 * head_dim * 4;
        let mha_total = n_layers * mha_per_layer;

        // GQA: 8 KV heads (4x savings)
        let gqa_per_layer = 2 * max_seq_len * 8 * head_dim * 4;
        let gqa_total = n_layers * gqa_per_layer;

        assert_eq!(mha_total / gqa_total, 4);
    }

    // ========================================================================
    // QWEN-007: Q8 KV Cache Tests
    // ========================================================================

    #[test]
    fn test_q8_kv_cache_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Initialize Q8 KV cache
        let result = exec.init_kv_cache_q8_gpu(
            4,   // num_layers
            8,   // num_heads
            4,   // num_kv_heads (GQA)
            128, // head_dim (divisible by 32)
            512, // max_len
        );
        assert!(result.is_ok(), "Q8 KV cache init failed: {:?}", result);
        assert!(exec.is_kv_cache_q8_enabled());
    }

    #[test]
    fn test_q8_kv_cache_invalid_head_dim() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // head_dim not divisible by 32 should fail
        let result = exec.init_kv_cache_q8_gpu(4, 8, 4, 100, 512);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("divisible by 32"));
        }
    }

    #[test]
    fn test_q8_kv_cache_memory_calculation() {
        // Test Q8 memory calculation vs FP32
        let n_layers = 32usize;
        let n_kv_heads = 8usize;
        let head_dim = 128usize;
        let max_seq_len = 4096usize;

        // FP32: 4 bytes per value
        let fp32_bytes = n_layers * 2 * n_kv_heads * max_seq_len * head_dim * 4;

        // Q8: 1 byte per value + 4 bytes per 32 values (scale)
        let q8_values = n_layers * 2 * n_kv_heads * max_seq_len * head_dim * 1;
        let q8_scales = n_layers * 2 * n_kv_heads * max_seq_len * (head_dim / 32) * 4;
        let q8_bytes = q8_values + q8_scales;

        // Q8 should be ~4x smaller (actually 4x / (1 + 1/8) ≈ 3.56x due to scales)
        let reduction = fp32_bytes as f64 / q8_bytes as f64;
        assert!(reduction > 3.5, "Expected >3.5x reduction, got {:.2}x", reduction);
        assert!(reduction < 4.0, "Expected <4x reduction, got {:.2}x", reduction);
    }

    #[test]
    fn test_q8_kv_cache_memory_methods() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Initialize Q8 KV cache
        exec.init_kv_cache_q8_gpu(4, 8, 4, 128, 512).unwrap();

        let q8_mem = exec.kv_cache_q8_memory_bytes();
        let fp32_equiv = exec.kv_cache_fp32_equivalent_bytes();

        assert!(q8_mem > 0, "Q8 memory should be > 0");
        assert!(fp32_equiv > q8_mem, "FP32 equivalent should be > Q8 memory");

        let reduction = fp32_equiv as f64 / q8_mem as f64;
        assert!(reduction > 3.5, "Expected >3.5x reduction, got {:.2}x", reduction);
    }

    #[test]
    fn test_q8_kv_cache_write_read_roundtrip() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_kv_heads = 4;
        let head_dim = 64; // Divisible by 32
        let max_len = 16;

        // Initialize Q8 KV cache
        exec.init_kv_cache_q8_gpu(2, 8, num_kv_heads, head_dim, max_len)
            .unwrap();

        // Create test K/V vectors with known values
        let size = num_kv_heads * head_dim;
        let k: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..size).map(|i| (i as f32) * -0.01).collect();

        // Write to position 0
        exec.write_kv_q8(0, 0, &k, &v).unwrap();

        // Read back
        let (k_out, v_out) = exec.read_kv_q8(0, 0, 1).unwrap();

        // Verify dimensions
        assert_eq!(k_out.len(), size, "K output size mismatch");
        assert_eq!(v_out.len(), size, "V output size mismatch");

        // Verify values are close (Q8 has ~1% quantization error max)
        for i in 0..size {
            let k_err = (k[i] - k_out[i]).abs();
            let v_err = (v[i] - v_out[i]).abs();
            // Allow 1% relative error or 0.01 absolute error
            let k_tol = (k[i].abs() * 0.02).max(0.02);
            let v_tol = (v[i].abs() * 0.02).max(0.02);
            assert!(
                k_err < k_tol,
                "K[{}]: expected {}, got {}, err {} > tol {}",
                i,
                k[i],
                k_out[i],
                k_err,
                k_tol
            );
            assert!(
                v_err < v_tol,
                "V[{}]: expected {}, got {}, err {} > tol {}",
                i,
                v[i],
                v_out[i],
                v_err,
                v_tol
            );
        }
    }

    #[test]
    fn test_q8_kv_cache_multiple_positions() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_kv_heads = 2;
        let head_dim = 32; // Minimal divisible by 32
        let max_len = 8;

        exec.init_kv_cache_q8_gpu(1, 4, num_kv_heads, head_dim, max_len)
            .unwrap();

        let size = num_kv_heads * head_dim;

        // Write to multiple positions
        for pos in 0..4 {
            let k: Vec<f32> = (0..size).map(|i| (pos as f32 + i as f32) * 0.1).collect();
            let v: Vec<f32> = (0..size).map(|i| -(pos as f32 + i as f32) * 0.1).collect();
            exec.write_kv_q8(0, pos, &k, &v).unwrap();
        }

        // Read all positions at once
        let (k_all, v_all) = exec.read_kv_q8(0, 0, 4).unwrap();

        assert_eq!(k_all.len(), 4 * size, "K all size mismatch");
        assert_eq!(v_all.len(), 4 * size, "V all size mismatch");
    }

    #[test]
    fn test_q8_kv_cache_not_enabled_error() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Don't initialize Q8 cache
        let k = vec![1.0f32; 128];
        let v = vec![1.0f32; 128];

        let result = exec.write_kv_q8(0, 0, &k, &v);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }
}
