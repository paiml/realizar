//! KV Cache management for GPU-resident inference
//!
//! This module implements:
//! - PAR-018: GPU-Resident KV Cache initialization
//! - PAR-021: GQA (Grouped Query Attention) support
//! - PAR-119: Batched KV cache for multi-sequence processing
//! - Cache reset, rollback, and continuation

use super::*;

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

        // Run flash attention
        let mut output_full = vec![0.0f32; tensor_size];
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
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
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

}
