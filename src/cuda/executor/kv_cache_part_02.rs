
impl CudaExecutor {

    // ========================================================================
    // QWEN-007 Phase 3: GPU-side Q8 Dequantization
    // ========================================================================

    /// Dequantize Q8 KV cache to FP32 on GPU
    ///
    /// Uses the Q8Dequant kernel to dequantize K/V from Q8 format to FP32
    /// directly on the GPU, returning FP32 buffers that can be used with
    /// existing attention kernels.
    ///
    /// Memory layout:
    /// - Input (Q8): [num_kv_heads, max_len, head_dim] with positions 0..seq_len filled
    /// - Output (FP32): [num_kv_heads, seq_len, head_dim] contiguous
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index
    /// * `seq_len` - Number of positions to dequantize (from 0 to seq_len-1)
    ///
    /// # Returns
    ///
    /// Tuple of (K_fp32, V_fp32) GPU buffers, each [num_kv_heads × seq_len × head_dim]
    pub fn dequantize_kv_q8_gpu(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<(GpuBuffer<f32>, GpuBuffer<f32>), GpuError> {
        if !self.kv_cache_q8_enabled {
            return Err(GpuError::InvalidParameter(
                "Q8 KV cache not enabled. Call init_kv_cache_q8_gpu first.".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        if seq_len > max_len {
            return Err(GpuError::InvalidParameter(format!(
                "seq_len {} exceeds max_len {}",
                seq_len, max_len
            )));
        }

        // Total elements to dequantize per K/V (output is contiguous)
        let total_elements = seq_len * num_kv_heads * head_dim;
        let blocks_per_head = head_dim / 32;

        // Get Q8 buffer keys
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);
        let k_scales_key = format!("kv_{}_k_scales", layer_idx);
        let v_scales_key = format!("kv_{}_v_scales", layer_idx);

        // Get Q8 buffer pointers
        let k_q8_buf = self.kv_cache_q8_k.get(&k_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 K cache for layer {} not found", layer_idx))
        })?;
        let k_scales_buf = self
            .kv_cache_q8_k_scales
            .get(&k_scales_key)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "Q8 K scales for layer {} not found",
                    layer_idx
                ))
            })?;
        let v_q8_buf = self.kv_cache_q8_v.get(&v_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 V cache for layer {} not found", layer_idx))
        })?;
        let v_scales_buf = self
            .kv_cache_q8_v_scales
            .get(&v_scales_key)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "Q8 V scales for layer {} not found",
                    layer_idx
                ))
            })?;

        // Allocate output FP32 buffers
        let k_fp32_buf = GpuBuffer::<f32>::new(&self.context, total_elements)?;
        let v_fp32_buf = GpuBuffer::<f32>::new(&self.context, total_elements)?;

        // Elements to process per head (seq_len positions × head_dim values)
        let elements_per_head = seq_len * head_dim;
        let _scales_per_head = seq_len * blocks_per_head;

        // Generate and compile Q8 dequant kernel for per-head processing
        let kernel_type = crate::cuda::KernelType::Q8Dequant {
            n: elements_per_head as u32,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let ptx = self.kernels.generate_ptx(&kernel_type);
        let module_key = format!("q8_dequant_{}", elements_per_head);

        if !self.modules.contains_key(&module_key) {
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(module_key.clone(), module);
        }
        let module = self
            .modules
            .get_mut(&module_key)
            .expect("module just inserted");

        // Launch config: 256 threads per block
        let threads_per_block = 256u32;
        let config = LaunchConfig::linear(elements_per_head as u32, threads_per_block);

        // Get base pointers
        let k_q8_base = k_q8_buf.as_ptr();
        let k_scales_base = k_scales_buf.as_ptr();
        let k_out_base = k_fp32_buf.as_ptr();
        let v_q8_base = v_q8_buf.as_ptr();
        let v_scales_base = v_scales_buf.as_ptr();
        let v_out_base = v_fp32_buf.as_ptr();

        // Process each head separately with correct offsets
        // Input layout: [num_kv_heads, max_len, head_dim] - stride = max_len * head_dim per head
        // Output layout: [num_kv_heads, seq_len, head_dim] - stride = seq_len * head_dim per head
        let src_quant_stride = max_len * head_dim; // bytes for i8
        let src_scale_stride = max_len * blocks_per_head * 4; // bytes for f32
        let dst_stride = elements_per_head * 4; // bytes for f32

        for head in 0..num_kv_heads {
            // Calculate offsets for this head
            let src_quant_offset = head * src_quant_stride;
            let src_scale_offset = head * src_scale_stride;
            let dst_offset = head * dst_stride;

            // SAFETY: Pointer arithmetic to access head-specific regions
            // The offsets are within allocated buffer bounds
            unsafe {
                // Dequantize K for this head
                let mut k_q8_ptr = k_q8_base + src_quant_offset as u64;
                let mut k_scales_ptr = k_scales_base + src_scale_offset as u64;
                let mut k_out_ptr = k_out_base + dst_offset as u64;
                let mut n_val = elements_per_head as u32;

                self.compute_stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut k_q8_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_scales_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_out_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    ],
                )?;

                // Dequantize V for this head
                let mut v_q8_ptr = v_q8_base + src_quant_offset as u64;
                let mut v_scales_ptr = v_scales_base + src_scale_offset as u64;
                let mut v_out_ptr = v_out_base + dst_offset as u64;

                self.compute_stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut v_q8_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_scales_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_out_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Synchronize to ensure all head dequantizations are complete
        self.compute_stream.synchronize()?;

        Ok((k_fp32_buf, v_fp32_buf))
    }

    // ========================================================================
    // QWEN-007 Phase 4: Q8 Incremental Attention
    // ========================================================================

    /// Incremental attention using Q8 quantized KV cache
    ///
    /// This is the Q8 variant of `incremental_attention_gpu`. It:
    /// 1. Quantizes incoming K/V to Q8 format
    /// 2. Appends to Q8 GPU cache
    /// 3. Dequantizes full cache to FP32 on GPU
    /// 4. Runs attention kernel against dequantized K/V
    ///
    /// Memory savings: ~3.56x for KV cache storage
    /// Tradeoff: Additional dequantization kernel launch per attention call
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Transformer layer index
    /// * `q` - Query vector for current position [num_heads × head_dim]
    /// * `current_k` - Key vector for current position [num_kv_heads × head_dim]
    /// * `current_v` - Value vector for current position [num_kv_heads × head_dim]
    /// * `output` - Output buffer [num_heads × head_dim]
    ///
    /// # Returns
    ///
    /// New total sequence length after appending
    #[allow(clippy::too_many_arguments)]
    pub fn incremental_attention_q8_gpu(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        output: &mut [f32],
    ) -> Result<usize, GpuError> {
        if !self.kv_cache_q8_enabled {
            return Err(GpuError::InvalidParameter(
                "Q8 KV cache not enabled. Call init_kv_cache_q8_gpu first.".to_string(),
            ));
        }

        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let max_len = self.kv_cache_max_len;

        // Validate dimensions
        if q.len() != q_dim {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "QWEN-007: Q dimension mismatch - expected {}, got {}",
                q_dim,
                q.len()
            )));
        }
        if current_k.len() != kv_dim || current_v.len() != kv_dim {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "QWEN-007: K/V dimension mismatch - expected {}, got K[{}] V[{}]",
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
                "QWEN-007: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // Step 1: Quantize and write K/V to Q8 cache
        self.write_kv_q8(layer_idx, cache_len, current_k, current_v)?;

        // Step 2: Dequantize full cache to FP32 on GPU
        let (k_fp32_buf, v_fp32_buf) = self.dequantize_kv_q8_gpu(layer_idx, new_len)?;

        // Step 3: Upload Q to GPU
        let mut q_buf = GpuBuffer::<f32>::new(&self.context, q_dim)?;
        q_buf.copy_from_host(q)?;

        // Step 4: Allocate output buffer
        let out_buf = GpuBuffer::<f32>::new(&self.context, q_dim)?;

        // Step 5: Get kernel module (same as FP32 incremental attention)
        let kernel_type = KernelType::IncrementalAttention {
            max_seq_len: new_len as u32, // Use actual seq_len, not max_len
            head_dim: head_dim as u32,
            n_heads: num_heads as u32,
            n_kv_heads: num_kv_heads as u32,
            indirect: false,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let ptx = self.kernels.generate_ptx(&kernel_type);
        let module_key = format!(
            "incremental_attention_q8_{}_{}_{}_{}",
            new_len, head_dim, num_heads, num_kv_heads
        );

        if !self.modules.contains_key(&module_key) {
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(module_key.clone(), module);
        }
        let module = self
            .modules
            .get_mut(&module_key)
            .expect("module just inserted");

        // Step 6: Launch attention kernel
        // Grid: (num_heads, 1, 1) - one block per head
        // Block: (32, 1, 1) - one warp per block
        let config = LaunchConfig::grid_2d(num_heads as u32, 1, 32, 1);

        let mut ptr_q = q_buf.as_ptr();
        let mut ptr_k = k_fp32_buf.as_ptr();
        let mut ptr_v = v_fp32_buf.as_ptr();
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

include!("kv_cache_part_02_part_02.rs");
include!("kv_cache_part_02_part_03.rs");
include!("kv_cache_part_02_part_04.rs");
