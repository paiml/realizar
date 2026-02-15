impl CudaExecutor {

    /// PAR-119: Batched incremental attention for M sequences in parallel
    ///
    /// This eliminates the sequential attention bottleneck by processing all M sequences
    /// in a single kernel launch. Uses BatchedIncrementalAttentionKernel with pointer arrays
    /// to access per-sequence KV caches.
    ///
    /// # Arguments
    /// * `layer_idx` - Transformer layer index
    /// * `q_batched` - Q projections [M, num_heads, head_dim]
    /// * `k_batched` - K projections [M, num_kv_heads, head_dim]
    /// * `v_batched` - V projections [M, num_kv_heads, head_dim]
    /// * `out_batched` - Output buffer [M, num_heads, head_dim]
    /// * `m` - Batch size (number of sequences)
    /// * `positions` - Position for each sequence [M]
    #[allow(clippy::too_many_arguments)]
    pub fn batched_incremental_attention_into(
        &mut self,
        layer_idx: usize,
        q_batched: &GpuBuffer<f32>,
        k_batched: &GpuBuffer<f32>,
        v_batched: &GpuBuffer<f32>,
        out_batched: &GpuBuffer<f32>,
        m: usize,
        positions: &[u32],
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let stride = self.batched_kv_stride;

        if stride == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-119: Batched KV cache not initialized (call init_batched_kv_cache_gpu first)"
                    .to_string(),
            ));
        }

        // Get batched KV cache buffers for this layer
        let k_cache = self.batched_kv_k_caches.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-119: Batched K cache not found for layer {}",
                layer_idx
            ))
        })?;
        let v_cache = self.batched_kv_v_caches.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-119: Batched V cache not found for layer {}",
                layer_idx
            ))
        })?;

        // Step 1: Scatter K/V to per-sequence caches
        // For each sequence seq_idx, scatter to cache[seq_idx * stride + pos * head_dim * num_kv_heads]
        let kv_dim = num_kv_heads * head_dim;
        let scatter_config = LaunchConfig {
            grid: (num_kv_heads as u32, 1, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem: 0,
        };

        // Get or compile scatter kernel
        let scatter_type = KernelType::KvCacheScatter {
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            max_len: max_len as u32,
        };
        let scatter_name = self.kernels.kernel_name(&scatter_type);
        let scatter_key = format!("kv_scatter_{}_{}", num_kv_heads, head_dim);

        if !self.modules.contains_key(&scatter_key) {
            let scatter_ptx = self.kernels.generate_ptx(&scatter_type);
            let module = self.compile_ptx(&scatter_ptx)?;
            self.modules.insert(scatter_key.clone(), module);
        }

        // Scatter K and V for each sequence (still sequential, but now to separate caches)
        for seq_idx in 0..m {
            let pos = positions[seq_idx] as usize;

            // Calculate source and destination pointers for this sequence
            let k_src_offset = seq_idx * kv_dim;
            let k_dst_offset = seq_idx * stride;
            let k_src_ptr = k_batched.as_ptr() + (k_src_offset * std::mem::size_of::<f32>()) as u64;
            let k_dst_ptr = k_cache.as_ptr() + (k_dst_offset * std::mem::size_of::<f32>()) as u64;

            let mut k_src = k_src_ptr;
            let mut k_dst = k_dst_ptr;
            let mut pos_val = pos as u32;
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;

            let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
            // SAFETY: Unsafe operation with validated invariants
            unsafe {
                self.compute_stream.launch_kernel(
                    scatter_module,
                    scatter_name,
                    &scatter_config,
                    &mut [
                        std::ptr::from_mut(&mut k_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut pos_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }

            // Scatter V
            let v_src_offset = seq_idx * kv_dim;
            let v_dst_offset = seq_idx * stride;
            let v_src_ptr = v_batched.as_ptr() + (v_src_offset * std::mem::size_of::<f32>()) as u64;
            let v_dst_ptr = v_cache.as_ptr() + (v_dst_offset * std::mem::size_of::<f32>()) as u64;

            let mut v_src = v_src_ptr;
            let mut v_dst = v_dst_ptr;

            let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
            // SAFETY: Unsafe operation with validated invariants
            unsafe {
                self.compute_stream.launch_kernel(
                    scatter_module,
                    scatter_name,
                    &scatter_config,
                    &mut [
                        std::ptr::from_mut(&mut v_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut pos_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Update per-sequence cache lengths
        for seq_idx in 0..m {
            let pos = positions[seq_idx] as usize;
            if seq_idx < self.batched_kv_lengths.len() {
                self.batched_kv_lengths[seq_idx] = pos + 1;
            }
        }

        // Step 2: Build pointer arrays for batched attention
        // Each pointer points to the start of that sequence's KV cache
        let k_cache_base = k_cache.as_ptr();
        let v_cache_base = v_cache.as_ptr();
        let stride_bytes = (stride * std::mem::size_of::<f32>()) as u64;

        let k_ptrs: Vec<u64> = (0..m)
            .map(|seq_idx| k_cache_base + seq_idx as u64 * stride_bytes)
            .collect();
        let v_ptrs: Vec<u64> = (0..m)
            .map(|seq_idx| v_cache_base + seq_idx as u64 * stride_bytes)
            .collect();
        let seq_lens: Vec<u32> = (0..m)
            .map(|seq_idx| self.batched_kv_lengths.get(seq_idx).copied().unwrap_or(1) as u32)
            .collect();

        // Step 3: Launch batched attention kernel
        // Compile module BEFORE taking mutable borrows on ptr buffers to
        // avoid borrow-checker conflict (compile_ptx borrows &self).
        let kernel = BatchedIncrementalAttentionKernel::new(
            max_len as u32,
            head_dim as u32,
            num_heads as u32,
            num_kv_heads as u32,
            m as u32,
        );
        let kernel_name = kernel.name();
        let module_key = format!(
            "batched_incr_attn_{}_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads, m
        );

        if !self.modules.contains_key(&module_key) {
            // PAR-119: Use emit_ptx() to get full module with version/target headers
            let ptx_source = kernel.emit_ptx();
            let module = self.compile_ptx(&ptx_source)?;
            self.modules.insert(module_key.clone(), module);
        }

        // Upload pointer arrays and sequence lengths to GPU
        let k_ptrs_buf = self.batched_k_ptrs.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-119: batched_k_ptrs not allocated".to_string())
        })?;
        let v_ptrs_buf = self.batched_v_ptrs.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-119: batched_v_ptrs not allocated".to_string())
        })?;
        let seq_lens_buf = self.batched_seq_lens_gpu.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-119: batched_seq_lens_gpu not allocated".to_string())
        })?;

        k_ptrs_buf.copy_from_host(&k_ptrs)?;
        v_ptrs_buf.copy_from_host(&v_ptrs)?;
        seq_lens_buf.copy_from_host(&seq_lens)?;
        let module = self
            .modules
            .get_mut(&module_key)
            .expect("module just inserted");

        // Grid: (num_heads, batch_size, 1), Block: (32, 1, 1)
        let config = LaunchConfig {
            grid: (num_heads as u32, m as u32, 1),
            block: (32, 1, 1),
            shared_mem: 0,
        };

        let mut q_ptr = q_batched.as_ptr();
        let mut k_ptrs_ptr = k_ptrs_buf.as_ptr();
        let mut v_ptrs_ptr = v_ptrs_buf.as_ptr();
        let mut out_ptr = out_batched.as_ptr();
        let mut seq_lens_ptr = seq_lens_buf.as_ptr();

        // SAFETY: Unsafe operation with validated invariants
        unsafe {
            self.compute_stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut q_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_ptrs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_ptrs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut out_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_lens_ptr) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    // =========================================================================
    // PAR-118: Flash Decoding - Split-K Attention for 2X Ollama Performance
    // =========================================================================

    /// Initialize Flash Decoding buffers for split-K attention.
    ///
    /// Flash Decoding splits the KV cache into chunks processed in parallel,
    /// then reduces partial results. This amortizes memory bandwidth across
    /// multiple thread blocks, achieving higher throughput for long sequences.
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Head dimension
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `batch_size` - Batch size (M)
    ///
    /// # Performance
    /// - Expected 1.5-2x speedup over sequential attention for seq_len > 128
    /// - Minimal overhead for short sequences (< 128 positions)
    pub fn init_flash_decoding(
        &mut self,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        batch_size: usize,
    ) -> Result<(), GpuError> {
        use trueno_gpu::kernels::FLASH_DECODE_CHUNK_SIZE;

        // Calculate partials buffer size
        // Layout: [M, num_heads, max_chunks, head_dim + 2]
        let max_chunks =
            (max_seq_len + FLASH_DECODE_CHUNK_SIZE as usize - 1) / FLASH_DECODE_CHUNK_SIZE as usize;
        let partials_per_head = max_chunks * (head_dim + 2);
        let total_partials = batch_size * num_heads * partials_per_head;

        // Allocate partials buffer
        self.flash_decode_partials = Some(GpuBuffer::new(&self.context, total_partials)?);
        self.flash_decode_max_seq_len = max_seq_len;
        self.flash_decode_max_chunks = max_chunks;
        self.flash_decode_enabled = true;

        // PAR-118: Persistent seq_lens buffer — remains valid for the lifetime of async kernels,
        // ensuring the device pointer is not invalidated before kernel completion.
        self.flash_decode_seq_lens_buf = Some(GpuBuffer::from_host(&self.context, &[0u32])?);

        // PAR-118: Populate per-layer K/V pointer buffers for graph-compatible Flash Decoding.
        // These contain the GPU base address of each layer's KV cache, enabling the
        // FlashDecodingChunkKernel to read k_ptrs[0] / v_ptrs[0] indirectly.
        // Addresses are stable because KV cache buffers are allocated once in init_kv_cache_gpu.
        let num_layers = self.kv_cache_lengths.len().max(
            self.kv_cache_gpu
                .keys()
                .filter_map(|k| {
                    k.strip_prefix("kv_")
                        .and_then(|s| s.strip_suffix("_k"))
                        .and_then(|s| s.parse::<usize>().ok())
                })
                .max()
                .map_or(0, |m| m + 1),
        );

        for layer_idx in 0..num_layers {
            let k_key = format!("kv_{layer_idx}_k");
            let v_key = format!("kv_{layer_idx}_v");

            if let (Some(k_cache), Some(v_cache)) =
                (self.kv_cache_gpu.get(&k_key), self.kv_cache_gpu.get(&v_key))
            {
                let k_ptr_buf = GpuBuffer::from_host(&self.context, &[k_cache.as_ptr()])?;
                let v_ptr_buf = GpuBuffer::from_host(&self.context, &[v_cache.as_ptr()])?;
                self.flash_decode_k_ptrs.insert(layer_idx, k_ptr_buf);
                self.flash_decode_v_ptrs.insert(layer_idx, v_ptr_buf);
            }
        }

        // PAR-118: Pre-compile Flash Decoding PTX modules to avoid compilation during
        // CUDA graph capture (which causes CUDA_ERROR_UNKNOWN/901).
        // Must happen BEFORE any graph capture attempt.
        {
            use trueno_gpu::kernels::{
                FlashDecodingChunkKernel, FlashDecodingReduceKernel, Kernel,
            };

            let num_kv_heads = self.kv_num_kv_heads;
            let max_len = self.kv_cache_max_len;

            let chunk_kernel = FlashDecodingChunkKernel::new(
                max_len as u32,
                head_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                1, // M=1 for single-sequence decode
            );
            let chunk_module_key = format!(
                "flash_decode_chunk_{}_{}_{}_{}",
                max_len, head_dim, num_heads, num_kv_heads
            );
            if !self.modules.contains_key(&chunk_module_key) {
                let chunk_ptx = chunk_kernel.emit_ptx();
                let module = self.compile_ptx(&chunk_ptx)?;
                self.modules.insert(chunk_module_key, module);
            }

            let reduce_kernel = FlashDecodingReduceKernel::new(
                head_dim as u32,
                num_heads as u32,
                1, // M=1
            );
            let reduce_module_key = format!("flash_decode_reduce_{}_{}", head_dim, num_heads);
            if !self.modules.contains_key(&reduce_module_key) {
                let reduce_ptx = reduce_kernel.emit_ptx();
                let module = self.compile_ptx(&reduce_ptx)?;
                self.modules.insert(reduce_module_key, module);
            }
        }

        Ok(())
    }
}
