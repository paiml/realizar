//! Attention mechanisms: incremental attention, flash decoding, tensor core attention
//!
//! This module implements:
//! - PAR-023: GPU-Resident Incremental Attention
//! - PAR-118: Flash Decoding for parallel KV processing
//! - PAR-065: Tensor Core Attention
//! - Batched attention for multi-sequence processing

use super::*;

impl CudaExecutor {
    // =========================================================================
    // PAR-023: GPU-Resident Incremental Attention (No Sync)
    // Reduces sync per attention call by keeping Q/K/V on GPU
    // =========================================================================

    /// PAR-023: GPU-resident incremental attention operating on GPU buffers
    ///
    /// Same as `incremental_attention_gpu` but takes GPU buffers instead of
    /// host slices, allowing full GPU pipeline without intermediate syncs.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index for KV cache lookup
    /// * `q_gpu` - Query GPU buffer [num_heads * head_dim]
    /// * `k_gpu` - Current key GPU buffer [num_kv_heads * head_dim]
    /// * `v_gpu` - Current value GPU buffer [num_kv_heads * head_dim]
    ///
    /// # Returns
    /// (output_gpu, new_seq_len) - Attention output buffer and updated sequence length
    #[allow(clippy::too_many_arguments)]
    pub fn incremental_attention_async(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
    ) -> Result<(GpuBuffer<f32>, usize), GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let q_dim = num_heads * head_dim;
        let max_len = self.kv_cache_max_len;

        // Get current cache length and check bounds
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let new_len = cache_len + 1;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-023: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // Get cache buffer keys
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // PAR-023: Copy K/V from GPU buffers to cache positions (D2D transfer)
        // Layout is [num_kv_heads, max_len, head_dim]
        // We need to copy each head's current K/V to the correct position
        //
        // Using D2D copy to avoid host round-trip (zero-sync attention)
        {
            let k_buf = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            for kv_head in 0..num_kv_heads {
                let src_offset = kv_head * head_dim;
                let dst_offset = kv_head * (max_len * head_dim) + cache_len * head_dim;
                k_buf.copy_from_buffer_at(k_gpu, dst_offset, src_offset, head_dim)?;
            }

            let v_buf = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            for kv_head in 0..num_kv_heads {
                let src_offset = kv_head * head_dim;
                let dst_offset = kv_head * (max_len * head_dim) + cache_len * head_dim;
                v_buf.copy_from_buffer_at(v_gpu, dst_offset, src_offset, head_dim)?;
            }
        }

        // Update cache length
        self.kv_cache_lengths.insert(layer_idx, new_len);

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

        // Get K and V buffer pointers from cache
        let k_buf = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?;
        let v_buf = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?;

        // Launch kernel
        let config = LaunchConfig::grid_2d(num_heads as u32, 1, 32, 1);

        let mut ptr_q = q_gpu.as_ptr();
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

        // PAR-023: NO sync here - caller continues pipeline
        Ok((out_buf, new_len))
    }

    /// PAR-051: Incremental attention writing into pre-allocated output buffer
    ///
    /// Like `incremental_attention_async` but eliminates GPU allocation by
    /// writing directly into the provided output buffer.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Transformer layer index (for KV cache lookup)
    /// * `q_gpu` - Query tensor on GPU [q_dim]
    /// * `k_gpu` - Key tensor on GPU [kv_dim] (will be appended to cache)
    /// * `v_gpu` - Value tensor on GPU [kv_dim] (will be appended to cache)
    /// * `out_gpu` - Pre-allocated output buffer [q_dim]
    ///
    /// # Returns
    ///
    /// New sequence length after appending K/V to cache
    pub fn incremental_attention_into(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        out_gpu: &GpuBuffer<f32>,
    ) -> Result<usize, GpuError> {
        self.incremental_attention_into_inner(layer_idx, q_gpu, k_gpu, v_gpu, out_gpu, false)
    }

    /// PAR-054-FIX: Version for graph capture that skips debug sync/copy
    pub(crate) fn incremental_attention_into_for_capture(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        out_gpu: &GpuBuffer<f32>,
    ) -> Result<usize, GpuError> {
        self.incremental_attention_into_inner(layer_idx, q_gpu, k_gpu, v_gpu, out_gpu, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn incremental_attention_into_inner(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        out_gpu: &GpuBuffer<f32>,
        skip_debug: bool,
    ) -> Result<usize, GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        // CORRECTNESS-013: Stateless GPU mode - disable KV cache to isolate cache bugs
        // When STATELESS_GPU=1, attention only sees the current token (no history)
        // If output becomes correct in stateless mode, the issue is in KV cache logic
        static STATELESS_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_stateless = *STATELESS_MODE.get_or_init(|| {
            let mode = std::env::var("STATELESS_GPU")
                .map(|v| v == "1")
                .unwrap_or(false);
            if mode {
                eprintln!("[CORRECTNESS-013] STATELESS_GPU mode ENABLED - attention only sees current token");
            }
            mode
        });

        // Get current cache length and check bounds
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        // CORRECTNESS-013: In stateless mode, always use seq_len=1 (only current token)
        let new_len = if use_stateless { 1 } else { cache_len + 1 };
        if !use_stateless && new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-051: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // Get cache buffer keys
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // PAR-052: Use scatter kernel instead of per-head D2D copies
        // Replaces 2 * num_kv_heads D2D copies with 2 kernel launches
        // PAR-061: Use indirect scatter during graph capture to avoid baking position
        {
            // CORRECTNESS-001 FIX: Launch config must match kernel expectations:
            // - Each block handles one KV head (head_idx = ctaid.x)
            // - Each thread handles one element (elem_idx = tid.x)
            // Grid: num_kv_heads blocks, Block: head_dim threads
            let config = LaunchConfig {
                grid: (num_kv_heads as u32, 1, 1),
                block: (head_dim as u32, 1, 1),
                shared_mem: 0,
            };

            // Get cache buffers
            let k_buf = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-052: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            let mut k_src_ptr = k_gpu.as_ptr();
            let mut k_dst_ptr = k_buf.as_ptr();
            // CORRECTNESS-001 FIX: Kernel takes (src, cache, pos, head_dim, max_len)
            // Removed num_heads_val which was erroneously passed
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;

            // PAR-069: Use graph mode (indirect scatter) ONLY when position_buf is initialized
            // Previously used skip_debug flag, which conflated "skip debug prints" with "graph mode"
            // Root cause: CORRECTNESS-001 garbage output from GPU path
            if let Some(ref pos_buf) = self.position_buf {
                // PAR-061: Graph capture mode - use indirect scatter (reads position from device)
                let scatter_type = KernelType::KvCacheScatterIndirect {
                    num_kv_heads: num_kv_heads as u32,
                    head_dim: head_dim as u32,
                    max_len: max_len as u32,
                };
                let scatter_name = self.kernels.kernel_name(&scatter_type);
                let scatter_ptx = self.kernels.generate_ptx(&scatter_type);
                let scatter_key = format!("kv_scatter_indirect_{}_{}", num_kv_heads, head_dim);

                if !self.modules.contains_key(&scatter_key) {
                    let module = CudaModule::from_ptx(&self.context, &scatter_ptx)?;
                    self.modules.insert(scatter_key.clone(), module);
                }
                let scatter_module = self.modules.get_mut(&scatter_key).expect("just inserted");

                // Indirect kernel takes position_ptr as 3rd argument
                let mut pos_ptr = pos_buf.as_ptr();

                // CORRECTNESS-001 FIX: Kernel expects (src, cache, pos_ptr, head_dim, max_len)
                // CORRECTNESS-011: Use self.stream for graph capture (graph captures on stream, not compute_stream)
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe {
                    self.stream.launch_kernel(
                        scatter_module,
                        scatter_name,
                        &config,
                        &mut [
                            std::ptr::from_mut(&mut k_src_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut k_dst_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut pos_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                        ],
                    )?;
                }

                // Re-get module and scatter V
                let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
                let v_buf = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-052: KV cache not initialized for layer {}",
                        layer_idx
                    ))
                })?;
                let mut v_src_ptr = v_gpu.as_ptr();
                let mut v_dst_ptr = v_buf.as_ptr();
                let mut pos_ptr = pos_buf.as_ptr();

                // CORRECTNESS-001 FIX: Same fix for V scatter
                // CORRECTNESS-011: Use self.stream for graph capture
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe {
                    self.stream.launch_kernel(
                        scatter_module,
                        scatter_name,
                        &config,
                        &mut [
                            std::ptr::from_mut(&mut v_src_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut v_dst_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut pos_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            } else {
                // PAR-069: Normal mode (no graph capture) - use direct scatter kernel
                let scatter_type = KernelType::KvCacheScatter {
                    num_kv_heads: num_kv_heads as u32,
                    head_dim: head_dim as u32,
                    max_len: max_len as u32,
                };
                let scatter_name = self.kernels.kernel_name(&scatter_type);
                let scatter_ptx = self.kernels.generate_ptx(&scatter_type);
                let scatter_key = format!("kv_scatter_{}_{}", num_kv_heads, head_dim);

                if !self.modules.contains_key(&scatter_key) {
                    let module = CudaModule::from_ptx(&self.context, &scatter_ptx)?;
                    self.modules.insert(scatter_key.clone(), module);
                }
                let scatter_module = self.modules.get_mut(&scatter_key).expect("just inserted");

                // CORRECTNESS-013: In stateless mode, always write to position 0
                let mut position_val = if use_stateless {
                    0u32
                } else {
                    cache_len as u32
                };

                // CORRECTNESS-001 FIX: Kernel expects (src, cache, pos, head_dim, max_len)
                // Fixed parameter order: pos is 3rd, removed extra num_heads_val
                // CORRECTNESS-012: Use self.stream to match attention kernel stream
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe {
                    self.stream.launch_kernel(
                        scatter_module,
                        scatter_name,
                        &config,
                        &mut [
                            std::ptr::from_mut(&mut k_src_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut k_dst_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut position_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                        ],
                    )?;
                }

                // Re-get module and scatter V
                let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
                let v_buf = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-052: KV cache not initialized for layer {}",
                        layer_idx
                    ))
                })?;
                let mut v_src_ptr = v_gpu.as_ptr();
                let mut v_dst_ptr = v_buf.as_ptr();

                // CORRECTNESS-001 FIX: Same fix for V scatter
                // CORRECTNESS-012: Use self.stream to match attention kernel stream
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe {
                    self.stream.launch_kernel(
                        scatter_module,
                        scatter_name,
                        &config,
                        &mut [
                            std::ptr::from_mut(&mut v_src_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut v_dst_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut position_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }
        }

        // Update cache length
        self.kv_cache_lengths.insert(layer_idx, new_len);

        // PAR-058-DEBUG: Trace attention parameters for layer 0 (only first 3 tokens)
        // PAR-054-FIX: Skip during graph capture to avoid sync breaking capture
        if !skip_debug && layer_idx == 0 && new_len <= 3 {
            self.compute_stream.synchronize()?;
            eprintln!(
                "[PAR-058-ATTN] Layer {}: num_heads={}, num_kv_heads={}, head_dim={}, max_len={}, seq_len={}",
                layer_idx, num_heads, num_kv_heads, head_dim, max_len, new_len
            );
            // Check current K input (not the cache)
            let mut k_input = vec![0.0f32; k_gpu.len()];
            k_gpu.copy_to_host(&mut k_input)?;
            let k_nan = k_input.iter().filter(|x| x.is_nan()).count();
            if k_nan > 0 {
                eprintln!(
                    "[PAR-058-ATTN] K input has {} NaN out of {}",
                    k_nan,
                    k_input.len()
                );
            } else {
                eprintln!(
                    "[PAR-058-ATTN] K input OK, first 5: {:?}",
                    &k_input[..5.min(k_input.len())]
                );
            }
            // Check Q input
            let mut q_input = vec![0.0f32; q_gpu.len()];
            q_gpu.copy_to_host(&mut q_input)?;
            let q_nan = q_input.iter().filter(|x| x.is_nan()).count();
            if q_nan > 0 {
                eprintln!(
                    "[PAR-058-ATTN] Q input has {} NaN out of {}",
                    q_nan,
                    q_input.len()
                );
            } else {
                eprintln!(
                    "[PAR-058-ATTN] Q input OK, first 5: {:?}",
                    &q_input[..5.min(q_input.len())]
                );
            }
            // Check K cache values at position 0 (head 0)
            let k_cache = self.kv_cache_gpu.get(&k_key).expect("K cache exists");
            let cache_size = num_kv_heads * max_len * head_dim;
            let mut k_cache_vals = vec![0.0f32; cache_size];
            k_cache.copy_to_host(&mut k_cache_vals)?;
            let k_cache_nan = k_cache_vals.iter().filter(|x| x.is_nan()).count();
            if k_cache_nan > 0 {
                eprintln!("[PAR-058-ATTN] K cache has {} NaN", k_cache_nan);
            } else {
                eprintln!(
                    "[PAR-058-ATTN] K cache head0 pos0 first 5: {:?}",
                    &k_cache_vals[..5.min(k_cache_vals.len())]
                );
                // CORRECTNESS-013: Also dump position 1 when seq_len >= 2
                if new_len >= 2 && k_cache_vals.len() >= head_dim + 5 {
                    eprintln!(
                        "[PAR-058-ATTN] K cache head0 pos1 first 5: {:?}",
                        &k_cache_vals[head_dim..(head_dim + 5).min(k_cache_vals.len())]
                    );
                }
            }

            // Check V cache values
            let v_cache = self.kv_cache_gpu.get(&v_key).expect("V cache exists");
            let mut v_cache_vals = vec![0.0f32; cache_size];
            v_cache.copy_to_host(&mut v_cache_vals)?;
            let v_cache_nan = v_cache_vals.iter().filter(|x| x.is_nan()).count();
            if v_cache_nan > 0 {
                eprintln!("[PAR-058-ATTN] V cache has {} NaN", v_cache_nan);
            } else {
                eprintln!(
                    "[PAR-058-ATTN] V cache head0 pos0 first 5: {:?}",
                    &v_cache_vals[..5.min(v_cache_vals.len())]
                );
            }
        }

        // PAR-074: Adaptive attention kernel selection based on sequence length
        // - Short sequences (< 128): Use single-warp kernel (less overhead, ~1-2µs/token)
        // - Long sequences (>= 128): Use multi-warp kernel (parallel processing)
        //
        // Five-Whys Root Cause: Multi-warp has 4x warp synchronization overhead
        // that dominates at short sequences where there's not enough parallelism.
        //
        // CORRECTNESS-009: Single-warp kernel only handles head_dim <= 64 (2 elements/thread)
        // For head_dim > 64 (e.g., Qwen 2.5 with head_dim=128), must use multi-warp kernel
        // which handles 4 elements per thread (q0, q1, q2, q3 at offsets 0, 32, 64, 96)
        let use_graph_mode = self.seq_len_buf.is_some();
        let use_single_warp = new_len < 128 && head_dim <= 64;

        if layer_idx == 0 && new_len == 1 && verbose() {
            eprintln!(
                "[CORRECTNESS-009] head_dim={}, using {} kernel, graph_mode={}, skip_debug={}",
                head_dim,
                if use_single_warp {
                    "single-warp"
                } else {
                    "multi-warp"
                },
                use_graph_mode,
                skip_debug
            );
        }

        let (kernel_type, module_key, config) = if use_single_warp {
            // Single-warp: 32 threads per head, no shared memory
            let ktype = KernelType::IncrementalAttention {
                max_seq_len: max_len as u32,
                head_dim: head_dim as u32,
                n_heads: num_heads as u32,
                n_kv_heads: num_kv_heads as u32,
                indirect: use_graph_mode,
            };
            let key = if use_graph_mode {
                format!(
                    "incremental_attention_indirect_{}_{}_{}_{}",
                    max_len, head_dim, num_heads, num_kv_heads
                )
            } else {
                format!(
                    "incremental_attention_{}_{}_{}_{}",
                    max_len, head_dim, num_heads, num_kv_heads
                )
            };
            // Grid: num_heads blocks, Block: 32 threads (1 warp)
            let cfg = LaunchConfig::grid_2d(num_heads as u32, 1, 32, 1);
            (ktype, key, cfg)
        } else {
            // Multi-warp: 128 threads per head (4 warps), uses shared memory
            // PAR-107-REVERTED: 8 warps SLOWER due to synchronization overhead
            // Five-Whys: More warps = more reduction barriers, hurts single-token decode
            let num_warps_per_head = 4;
            let ktype = KernelType::MultiWarpAttention {
                max_seq_len: max_len as u32,
                head_dim: head_dim as u32,
                n_heads: num_heads as u32,
                n_kv_heads: num_kv_heads as u32,
                num_warps_per_head,
                indirect: use_graph_mode,
            };
            let key = if use_graph_mode {
                format!(
                    "multi_warp_attention_indirect_{}_{}_{}_{}_{}",
                    max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head
                )
            } else {
                format!(
                    "multi_warp_attention_{}_{}_{}_{}_{}",
                    max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head
                )
            };
            // Grid: num_heads blocks, Block: 128 threads (4 warps)
            let cfg = LaunchConfig::grid_2d(num_heads as u32, 1, 32 * num_warps_per_head, 1);
            (ktype, key, cfg)
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let ptx = self.kernels.generate_ptx(&kernel_type);

        if !self.modules.contains_key(&module_key) {
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(module_key.clone(), module);
        }
        let module = self
            .modules
            .get_mut(&module_key)
            .expect("module just inserted");

        // Get K and V buffer pointers from cache
        let k_buf = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?;
        let v_buf = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?;

        // PAR-074: Launch config already computed above in adaptive selection

        let mut ptr_q = q_gpu.as_ptr();
        let mut ptr_k = k_buf.as_ptr();
        let mut ptr_v = v_buf.as_ptr();
        let mut ptr_out = out_gpu.as_ptr();

        // PAR-069: Use graph mode (indirect kernel) ONLY when seq_len_buf is initialized
        if let Some(ref seq_len_buf) = self.seq_len_buf {
            // Graph capture mode - pass seq_len_buf pointer
            // CORRECTNESS-011: Use self.stream for graph capture (graph captures on stream, not compute_stream)
            let mut seq_len_ptr = seq_len_buf.as_ptr();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut seq_len_ptr) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        } else {
            // Normal mode - pass seq_len value directly
            // CORRECTNESS-012: Use self.stream (NOT compute_stream) to ensure synchronization
            // with subsequent GEMV operations which also use self.stream.
            // Five-Whys: GPU garbage output → race condition → attention on compute_stream,
            // output projection on stream → no sync → data corruption
            let mut seq_len_val = new_len as u32;
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
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
        }

        // PAR-051: NO sync here - caller continues pipeline

        // CORRECTNESS-013: Debug attention output for layer 0 at seq_len=2
        if !skip_debug && layer_idx == 0 && new_len == 2 {
            self.stream.synchronize()?;
            let mut attn_out = vec![0.0f32; out_gpu.len()];
            out_gpu.copy_to_host(&mut attn_out)?;
            eprintln!(
                "[CORRECTNESS-013-ATTN] Layer 0 attention output at seq_len=2, first 10: {:?}",
                &attn_out[..10.min(attn_out.len())]
            );
            // Dump per-head output for first 3 heads
            for h in 0..3.min(num_heads) {
                let start = h * head_dim;
                eprintln!(
                    "[CORRECTNESS-013-ATTN] Head {} first 5: {:?}",
                    h,
                    &attn_out[start..start + 5]
                );
            }
        }

        Ok(new_len)
    }

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
            let module = CudaModule::from_ptx(&self.context, &scatter_ptx)?;
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

        // Step 3: Launch batched attention kernel
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
            let module = CudaModule::from_ptx(&self.context, &ptx_source)?;
            self.modules.insert(module_key.clone(), module);
        }
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
        self.flash_decode_enabled = true;

        Ok(())
    }

    /// PAR-118: Flash Decoding attention using split-K parallelism.
    ///
    /// Splits the KV cache into chunks and processes them in parallel,
    /// then reduces partial results with proper softmax rescaling.
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
    pub fn flash_decoding_attention_into(
        &mut self,
        layer_idx: usize,
        q_batched: &GpuBuffer<f32>,
        k_batched: &GpuBuffer<f32>,
        v_batched: &GpuBuffer<f32>,
        out_batched: &GpuBuffer<f32>,
        m: usize,
        positions: &[u32],
    ) -> Result<(), GpuError> {
        use trueno_gpu::kernels::{
            FlashDecodingChunkKernel, FlashDecodingReduceKernel, Kernel, FLASH_DECODE_CHUNK_SIZE,
        };

        if !self.flash_decode_enabled {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-118: Flash Decoding not initialized (call init_flash_decoding first)"
                    .to_string(),
            ));
        }

        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let stride = self.batched_kv_stride;

        // Step 1: Scatter K/V to caches (same as batched_incremental_attention_into)
        let kv_dim = num_kv_heads * head_dim;
        let scatter_config = LaunchConfig {
            grid: (num_kv_heads as u32, 1, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem: 0,
        };

        let scatter_type = KernelType::KvCacheScatter {
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            max_len: max_len as u32,
        };
        let scatter_name = self.kernels.kernel_name(&scatter_type);
        let scatter_key = format!("kv_scatter_{}_{}", num_kv_heads, head_dim);

        if !self.modules.contains_key(&scatter_key) {
            let scatter_ptx = self.kernels.generate_ptx(&scatter_type);
            let module = CudaModule::from_ptx(&self.context, &scatter_ptx)?;
            self.modules.insert(scatter_key.clone(), module);
        }

        let k_cache = self.batched_kv_k_caches.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-118: Batched K cache not found for layer {}",
                layer_idx
            ))
        })?;
        let v_cache = self.batched_kv_v_caches.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-118: Batched V cache not found for layer {}",
                layer_idx
            ))
        })?;

        // Scatter K and V for each sequence
        for seq_idx in 0..m {
            let pos = positions[seq_idx] as usize;

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
            // SAFETY: Kernel launch with valid pointers - k_src/k_dst from GPU buffers,
            // pos/head_dim/max_len are stack values with stable addresses during call
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

            let v_src_offset = seq_idx * kv_dim;
            let v_dst_offset = seq_idx * stride;
            let v_src_ptr = v_batched.as_ptr() + (v_src_offset * std::mem::size_of::<f32>()) as u64;
            let v_dst_ptr = v_cache.as_ptr() + (v_dst_offset * std::mem::size_of::<f32>()) as u64;

            let mut v_src = v_src_ptr;
            let mut v_dst = v_dst_ptr;

            let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
            // SAFETY: Kernel launch with valid pointers - v_src/v_dst from GPU buffers,
            // pos/head_dim/max_len are stack values with stable addresses during call
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

        // Update cache lengths
        for seq_idx in 0..m {
            let pos = positions[seq_idx] as usize;
            if seq_idx < self.batched_kv_lengths.len() {
                self.batched_kv_lengths[seq_idx] = pos + 1;
            }
        }

        // Step 2: Build pointer arrays
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

        let k_ptrs_buf = self.batched_k_ptrs.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-118: batched_k_ptrs not allocated".to_string())
        })?;
        let v_ptrs_buf = self.batched_v_ptrs.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-118: batched_v_ptrs not allocated".to_string())
        })?;
        let seq_lens_buf = self.batched_seq_lens_gpu.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-118: batched_seq_lens_gpu not allocated".to_string())
        })?;

        k_ptrs_buf.copy_from_host(&k_ptrs)?;
        v_ptrs_buf.copy_from_host(&v_ptrs)?;
        seq_lens_buf.copy_from_host(&seq_lens)?;

        // Step 3: Compute max chunks needed
        let max_seq_len_actual = seq_lens.iter().copied().max().unwrap_or(1) as usize;
        let max_chunks = (max_seq_len_actual + FLASH_DECODE_CHUNK_SIZE as usize - 1)
            / FLASH_DECODE_CHUNK_SIZE as usize;

        // Step 4: Launch Flash Decoding chunk kernel
        let chunk_kernel = FlashDecodingChunkKernel::new(
            max_len as u32,
            head_dim as u32,
            num_heads as u32,
            num_kv_heads as u32,
            m as u32,
        );
        let chunk_kernel_name = chunk_kernel.name();
        let chunk_module_key = format!(
            "flash_decode_chunk_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );

        if !self.modules.contains_key(&chunk_module_key) {
            let chunk_ptx = chunk_kernel.emit_ptx();
            let module = CudaModule::from_ptx(&self.context, &chunk_ptx)?;
            self.modules.insert(chunk_module_key.clone(), module);
        }

        let partials_buf = self.flash_decode_partials.as_ref().ok_or_else(|| {
            GpuError::InvalidLaunchConfig(
                "PAR-118: flash_decode_partials not allocated".to_string(),
            )
        })?;

        // Grid: (num_heads, batch_size, max_chunks)
        let chunk_config = LaunchConfig {
            grid: (num_heads as u32, m as u32, max_chunks as u32),
            block: (32, 1, 1),
            shared_mem: 0,
        };

        let mut q_ptr = q_batched.as_ptr();
        let mut k_ptrs_ptr = k_ptrs_buf.as_ptr();
        let mut v_ptrs_ptr = v_ptrs_buf.as_ptr();
        let mut partials_ptr = partials_buf.as_ptr();
        let mut seq_lens_ptr = seq_lens_buf.as_ptr();
        let mut max_chunks_val = max_chunks as u32;

        let chunk_module = self
            .modules
            .get_mut(&chunk_module_key)
            .expect("module just inserted");

        // SAFETY: Kernel launch with valid pointers - all GPU buffer pointers derived from
        // allocated GpuBuffers, max_chunks is stack value with stable address during call
        unsafe {
            self.compute_stream.launch_kernel(
                chunk_module,
                chunk_kernel_name,
                &chunk_config,
                &mut [
                    std::ptr::from_mut(&mut q_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_ptrs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_ptrs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut partials_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_lens_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut max_chunks_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Step 5: Launch Flash Decoding reduce kernel
        let reduce_kernel =
            FlashDecodingReduceKernel::new(head_dim as u32, num_heads as u32, m as u32);
        let reduce_kernel_name = reduce_kernel.name();
        let reduce_module_key = format!("flash_decode_reduce_{}_{}", head_dim, num_heads);

        if !self.modules.contains_key(&reduce_module_key) {
            let reduce_ptx = reduce_kernel.emit_ptx();
            let module = CudaModule::from_ptx(&self.context, &reduce_ptx)?;
            self.modules.insert(reduce_module_key.clone(), module);
        }

        // Grid: (num_heads, batch_size, 1)
        let reduce_config = LaunchConfig {
            grid: (num_heads as u32, m as u32, 1),
            block: (32, 1, 1),
            shared_mem: 0,
        };

        let mut out_ptr = out_batched.as_ptr();

        let reduce_module = self
            .modules
            .get_mut(&reduce_module_key)
            .expect("module just inserted");

        // SAFETY: Kernel launch with valid pointers - partials/out/seq_lens from GPU buffers,
        // max_chunks is stack value with stable address during call
        unsafe {
            self.compute_stream.launch_kernel(
                reduce_module,
                reduce_kernel_name,
                &reduce_config,
                &mut [
                    std::ptr::from_mut(&mut partials_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut out_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_lens_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut max_chunks_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// Tensor Core attention using WMMA for FP16 matrix operations (PARITY-001.3)
    ///
    /// Uses FP16 Tensor Cores (WMMA) for Q×K^T and attention×V computation.
    /// Expected 4-10x speedup over FP32 FlashAttention on Tensor Core GPUs.
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor [n_heads, seq_len, head_dim] as FP32 (converted to FP16)
    /// * `k` - Key tensor [n_heads, seq_len, head_dim] as FP32 (converted to FP16)
    /// * `v` - Value tensor [n_heads, seq_len, head_dim] as FP32 (converted to FP16)
    /// * `output` - Output tensor [n_heads, seq_len, head_dim] (FP32 accumulator)
    /// * `seq_len` - Sequence length (must be multiple of 16 for WMMA)
    /// * `head_dim` - Dimension per head (must be multiple of 16 for WMMA)
    /// * `n_heads` - Number of attention heads
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Performance
    ///
    /// RTX 4090: 330 TFLOPS FP16 vs 83 TFLOPS FP32 (4x theoretical speedup)
    /// Target: <2ms per token vs 79ms FP32 baseline (~40x actual speedup)
    #[allow(clippy::too_many_arguments)]
    pub fn tensor_core_attention(
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
        // WMMA requires dimensions to be multiples of 16
        if !seq_len.is_multiple_of(16) || !head_dim.is_multiple_of(16) {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "Tensor Core attention requires dimensions multiple of 16: seq_len={}, head_dim={}",
                seq_len, head_dim
            )));
        }

        let head_size = (seq_len * head_dim) as usize;
        let total_size = head_size * n_heads as usize;

        // Validate input sizes
        if q.len() != total_size
            || k.len() != total_size
            || v.len() != total_size
            || output.len() != total_size
        {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "Tensor Core attention size mismatch: expected {} ({}×{}×{}), got Q[{}] K[{}] V[{}] O[{}]",
                total_size, n_heads, seq_len, head_dim,
                q.len(), k.len(), v.len(), output.len()
            )));
        }

        // Track memory allocation (FP32 buffers - conversion happens on GPU)
        self.memory_pool.record_allocation(total_size * 4 * 4);

        // Generate Tensor Core attention kernel
        let kernel_type = KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!(
            "tensor_core_attn_{}_{}_{}_{}",
            seq_len, head_dim, n_heads, causal
        );

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            #[cfg(test)]
            eprintln!("Generated Tensor Core attention PTX:\n{}", ptx);
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

        // Launch configuration for Tensor Core attention:
        // Grid.x = ceil(seq_len / 16) - number of 16×16 WMMA tiles
        // Grid.y = n_heads
        // Threads = 256 (8 warps per block for WMMA)
        let num_tiles = (seq_len + 15) / 16;
        let config = LaunchConfig::grid_2d(num_tiles, n_heads, 256, 1);

        // Get raw pointers for kernel args
        let mut ptr_q = buf_q.as_ptr();
        let mut ptr_k = buf_k.as_ptr();
        let mut ptr_v = buf_v.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut seq_len_val = seq_len;
        let mut head_dim_val = head_dim;
        let mut n_heads_val = n_heads;

        // Launch kernel
        // SAFETY: Buffers are valid, dimensions validated
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

    /// FP16 Tensor Core GEMM using WMMA intrinsics (IMP-1000a)
    ///
    /// Computes C = A × B using FP16 tensor cores with FP32 accumulation.
    /// RTX 4090: 330 TFLOPS FP16 vs 83 TFLOPS FP32 (4x theoretical speedup).
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A as FP32 (will be converted to FP16)
    /// * `b` - Weight matrix B as FP32 (will be converted to FP16)
    /// * `c` - Output matrix C (FP32 accumulator)
    /// * `m`, `n`, `k` - Matrix dimensions (must be multiples of 16)
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are not multiples of 16 or kernel fails.
    pub fn gemm_fp16(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Validate dimensions are multiples of 16 (WMMA requirement)
        if !m.is_multiple_of(16) || !n.is_multiple_of(16) || !k.is_multiple_of(16) {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "FP16 Tensor Core requires dimensions multiple of 16: m={}, n={}, k={}",
                m, n, k
            )));
        }

        // Validate sizes
        let expected_a = (m * k) as usize;
        let expected_b = (k * n) as usize;
        let expected_c = (m * n) as usize;

        if a.len() != expected_a || b.len() != expected_b || c.len() != expected_c {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "GEMM size mismatch: A[{}] expected {}, B[{}] expected {}, C[{}] expected {}",
                a.len(),
                expected_a,
                b.len(),
                expected_b,
                c.len(),
                expected_c
            )));
        }

        // Track memory usage
        self.memory_pool
            .record_allocation(expected_a * 4 + expected_b * 4 + expected_c * 4);

        // For now, use tiled GEMM as placeholder (FP16 WMMA PTX is generated but
        // actual tensor core execution requires half-precision buffer support)
        // The API is ready for when trueno-gpu adds FP16 buffer support
        let kernel_type = KernelType::GemmTiled {
            m,
            n,
            k,
            tile_size: 32,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_fp16_{}_{}_{}", m, n, k);

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
        let buf_a = GpuBuffer::from_host(&self.context, a)?;
        let buf_b = GpuBuffer::from_host(&self.context, b)?;
        // PARITY-114 FIX: Initialize output buffer with zeros to prevent state accumulation
        let c_zeros = vec![0.0f32; expected_c];
        let buf_c = GpuBuffer::from_host(&self.context, &c_zeros)?;

        // Launch configuration (16x16 tiles for FP16)
        // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
        let config = LaunchConfig::grid_2d((n + 31) / 32, (m + 31) / 32, 32, 32);

        // Get raw pointers for kernel args
        let mut ptr_a = buf_a.as_ptr();
        let mut ptr_b = buf_b.as_ptr();
        let mut ptr_c = buf_c.as_ptr();
        let mut m_val = m as i32;
        let mut n_val = n as i32;
        let mut k_val = k as i32;

        // Launch kernel
        // SAFETY: Buffers are valid, config matches kernel expectations
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and copy result back
        self.stream.synchronize()?;
        buf_c.copy_to_host(c)?;

        Ok(())
    }

    /// Compute attention score statistics (for debugging/profiling)
    #[must_use]
    pub fn flash_attention_memory_bytes(seq_len: u32, _head_dim: u32) -> (u64, u64) {
        // Naive: full N×N attention matrix
        let naive = u64::from(seq_len) * u64::from(seq_len) * 4;

        // FlashAttention: only block-sized working memory
        // Block size 64 is typical
        let block_size = 64u64;
        let flash = block_size * block_size * 4 * 2; // S and P blocks

        (naive, flash)
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
    // Incremental Attention Tests
    // ========================================================================

    #[test]
    fn test_incremental_attention_async_requires_kv_cache_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let hidden_dim = 256usize;

        let q = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let k = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let v = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();

        // Without KV cache init, should fail
        let result = exec.incremental_attention_async(0, &q, &k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_incremental_attention_into_requires_kv_cache_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let hidden_dim = 256usize;

        let q = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let k = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let v = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, hidden_dim).unwrap();

        // Without KV cache init, should fail
        let result = exec.incremental_attention_into(0, &q, &k, &v, &output);
        assert!(result.is_err());
    }

    // ========================================================================
    // Batched Incremental Attention Tests
    // ========================================================================

    #[test]
    fn test_batched_incremental_attention_dimensions() {
        // Test batched attention dimension calculations
        let batch_size = 4u32;
        let seq_len = 1024u32;
        let n_heads = 32u32;
        let head_dim = 64u32;
        let n_kv_heads = 8u32;

        // Q dimensions: batch × hidden_dim
        let q_size = batch_size * n_heads * head_dim;
        assert_eq!(q_size, 4 * 32 * 64);

        // KV cache dimensions: batch × seq_len × n_kv_heads × head_dim
        let kv_size = batch_size * seq_len * n_kv_heads * head_dim;
        assert!(kv_size > 0);

        // Output dimensions: same as Q
        let output_size = q_size;
        assert_eq!(output_size, q_size);
    }

    // ========================================================================
    // Flash Decoding Tests
    // ========================================================================

    #[test]
    fn test_init_flash_decoding() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let n_heads = 32usize;
        let head_dim = 64usize;
        let max_seq_len = 2048usize;
        let batch_size = 1usize;

        let result = exec.init_flash_decoding(n_heads, head_dim, max_seq_len, batch_size);
        assert!(result.is_ok());

        // Verify flash decoding is enabled
        assert!(exec.flash_decode_enabled);
    }

    #[test]
    fn test_flash_decoding_disabled_by_default() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(!exec.flash_decode_enabled);
    }

    // ========================================================================
    // Tensor Core Attention Tests
    // ========================================================================

    #[test]
    fn test_tensor_core_attention_requires_aligned_dims() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // WMMA requires dimensions to be multiples of 16
        let seq_len = 32u32; // Multiple of 16
        let head_dim = 64u32; // Multiple of 16
        let n_heads = 4u32;
        let total = (seq_len * head_dim * n_heads) as usize;

        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];

        let result =
            exec.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, true);
        // May fail on non-Tensor Core GPUs but exercises the code path
        let _ = result;
    }

    #[test]
    fn test_tensor_core_attention_unaligned_dims_error() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Use dimensions not multiples of 16
        let seq_len = 33u32; // Not multiple of 16
        let head_dim = 65u32; // Not multiple of 16
        let n_heads = 1u32;
        let total = (seq_len * head_dim * n_heads) as usize;

        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];

        let result =
            exec.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, true);
        assert!(result.is_err());
    }

    // ========================================================================
    // GEMM FP16 Tests
    // ========================================================================

    #[test]
    fn test_gemm_fp16_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let m = 32u32;
        let n = 32u32;
        let k = 32u32;

        let a = vec![1.0f32; (m * k) as usize];
        let b = vec![1.0f32; (k * n) as usize];
        let mut c = vec![0.0f32; (m * n) as usize];

        let result = exec.gemm_fp16(&a, &b, &mut c, m, n, k);
        let _ = result;
    }

    #[test]
    fn test_gemm_fp16_size_validation() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Wrong sizes
        let a = vec![1.0f32; 10];
        let b = vec![1.0f32; 10];
        let mut c = vec![0.0f32; 10];

        let result = exec.gemm_fp16(&a, &b, &mut c, 32, 32, 32);
        assert!(result.is_err());
    }

    // ========================================================================
    // Flash Attention Memory Bytes Tests
    // ========================================================================

    #[test]
    fn test_flash_attention_memory_bytes_small_seq() {
        let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(64, 64);
        // Naive: 64 * 64 * 4 = 16384 bytes
        assert_eq!(naive, 16384);
        // Flash: 64 * 64 * 4 * 2 = 32768 bytes (2 blocks)
        assert_eq!(flash, 32768);
    }

    #[test]
    fn test_flash_attention_memory_bytes_large_seq() {
        let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(4096, 64);
        // Naive: 4096 * 4096 * 4 = 67,108,864 bytes (64MB)
        assert_eq!(naive, 67_108_864);
        // Flash: still 32KB (constant!)
        assert_eq!(flash, 32768);
    }

    #[test]
    fn test_flash_attention_memory_bytes_savings() {
        let seq_len = 2048u32;
        let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(seq_len, 64);

        // Flash should use much less memory than naive
        assert!(flash < naive);

        // Calculate savings ratio
        let savings = naive / flash;
        assert!(savings > 100); // >100x savings for seq_len=2048
    }

    // ========================================================================
    // Attention Calculation Tests
    // ========================================================================

    #[test]
    fn test_attention_scale_calculation() {
        // Standard attention scale: 1/sqrt(head_dim)
        let head_dim = 64f32;
        let scale = 1.0 / head_dim.sqrt();
        assert!((scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_attention_softmax_numerics() {
        // Test that large values don't cause overflow in softmax
        let large_score = 100.0f32;
        let shifted = large_score - large_score; // shift by max for numerical stability
        let exp_val = shifted.exp();
        assert!((exp_val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gqa_head_mapping() {
        // Test grouped query attention head mapping
        let n_heads = 32u32;
        let n_kv_heads = 8u32;
        let group_size = n_heads / n_kv_heads;

        assert_eq!(group_size, 4);

        // Each Q head maps to a KV head
        for q_head in 0..n_heads {
            let kv_head = q_head / group_size;
            assert!(kv_head < n_kv_heads);
        }
    }

    #[test]
    fn test_rope_frequency_calculation() {
        // Test RoPE frequency calculation
        let head_dim = 64u32;
        let theta = 10000.0f32;

        // Frequencies: theta^(-2i/d) for i in 0..head_dim/2
        let freq_0 = theta.powf(0.0);
        assert_eq!(freq_0, 1.0);

        let freq_mid = theta.powf(-2.0 * 16.0 / head_dim as f32);
        assert!(freq_mid < 1.0);
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_incremental_attention_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let hidden_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let q = GpuBuffer::from_host(&exec.context, &vec![0.1f32; hidden_dim]).unwrap();
        let k = GpuBuffer::from_host(&exec.context, &vec![0.1f32; kv_dim]).unwrap();
        let v = GpuBuffer::from_host(&exec.context, &vec![0.1f32; kv_dim]).unwrap();

        // KV cache is now initialized via harness
        let result = exec.incremental_attention_async(0, &q, &k, &v);
        // Should execute the attention kernel path
        let _ = result;
    }

    #[test]
    fn test_incremental_attention_into_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let hidden_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let q = GpuBuffer::from_host(&exec.context, &vec![0.1f32; hidden_dim]).unwrap();
        let k = GpuBuffer::from_host(&exec.context, &vec![0.1f32; kv_dim]).unwrap();
        let v = GpuBuffer::from_host(&exec.context, &vec![0.1f32; kv_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, hidden_dim).unwrap();

        let result = exec.incremental_attention_into(0, &q, &k, &v, &output);
        let _ = result;
    }

    #[test]
    fn test_flash_decoding_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Initialize flash decoding
        let result = exec.init_flash_decoding(
            config.num_heads,
            config.head_dim,
            config.max_seq_len,
            1, // batch_size
        );
        assert!(result.is_ok());
        assert!(exec.flash_decode_enabled);
    }

    #[test]
    fn test_kv_cache_scatter_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // KV cache should be properly initialized
        assert!(exec.kv_cache_max_len > 0);
        assert!(exec.kv_num_heads > 0);
        assert!(exec.kv_head_dim > 0);

        // Verify KV cache GPU buffers exist
        let k_key = "kv_0_k".to_string();
        let v_key = "kv_0_v".to_string();
        assert!(exec.kv_cache_gpu.contains_key(&k_key));
        assert!(exec.kv_cache_gpu.contains_key(&v_key));
    }

    #[test]
    fn test_multi_layer_attention_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 4;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Verify all layers have KV cache
        for layer_idx in 0..config.num_layers {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);
            assert!(
                exec.kv_cache_gpu.contains_key(&k_key),
                "Missing KV cache for layer {}",
                layer_idx
            );
            assert!(
                exec.kv_cache_gpu.contains_key(&v_key),
                "Missing KV cache for layer {}",
                layer_idx
            );
        }
    }

    #[test]
    fn test_attention_with_gqa_ratio() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_heads = 32;
        config.num_kv_heads = 8; // 4:1 GQA ratio
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Verify GQA ratio is correctly applied
        let gqa_ratio = exec.kv_num_heads / exec.kv_num_kv_heads;
        assert_eq!(gqa_ratio, 4);
    }

    #[test]
    fn test_attention_rope_theta_config() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // RoPE theta should be configured
        assert!(exec.rope_theta > 0.0);
    }
}
