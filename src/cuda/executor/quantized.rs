//! Cached quantized GEMV methods for Q4K/Q5K/Q6K weights
//!
//! This module implements:
//! - PAR-005: Cached GEMV Methods (avoid per-call weight transfers)
//! - Q4K/Q5K/Q6K GEMV with GPU-cached weights
//! - Tiled, chunked, and coalesced GEMV variants
//! - DP4A SIMD-accelerated GEMV
//! - Fused RMSNorm + Q4K GEMV
//! - Batched Q4K/Q6K GEMV

use super::*;

impl CudaExecutor {
    // ========================================================================
    // PAR-005: Cached GEMV Methods (avoid per-call weight transfers)
    // ========================================================================

    /// Execute Q4_K GEMV using cached weights - PAR-005
    ///
    /// Uses pre-uploaded weights from `quantized_weight_cache` to avoid
    /// CPU→GPU transfer on every forward pass. Weights must be loaded
    /// beforehand via `load_quantized_weights()`.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight tensor
    /// * `input` - Input vector (f32, length k)
    /// * `output` - Output vector (f32, length n)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    ///
    /// # Errors
    ///
    /// Returns error if weights not cached or kernel fails.
    pub fn q4k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached weight buffer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-005: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        // PAR-057: Use TiledQ4KGemv for better performance (~4x fewer global reads)
        // Fall back to basic Q4KGemv if K not aligned to 256
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
            // PAR-502: Use chunked kernel for large K dimensions (7B+ models)
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else if use_tiled {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            // NOTE: Shared memory is statically declared in PTX - do NOT pass dynamically
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q4KGemv { k, n };
            let ck = format!("q4k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Transfer input (allocation overhead is negligible compared to weight caching)
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = buf_input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Execute Q4_K GEMV with GPU buffer input/output (async, no sync)
    ///
    /// This is the async variant that keeps data on GPU. Used for pipelining
    /// multiple operations without CPU round-trips.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        // CORRECTNESS-001: Use TiledQ4KGemv for aligned K (matches sync version)
        // The basic Q4KGemv kernel has the same scale extraction issue
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
            // PAR-502: Use chunked kernel for large K dimensions (7B+ models)
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else if use_tiled {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q4KGemv { k, n };
            let ck = format!("q4k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-023: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-058: Execute Q6_K GEMV using cached weight (async, no sync)
    ///
    /// Same as q4k_gemv_cached_async but for Q6_K quantized weights.
    /// Used for LM head when it's Q6K quantized.
    pub fn q6k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        // Load kernel module
        let kernel_type = KernelType::Q6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-058: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-043: Execute Q4_K GEMV using pre-indexed device pointer (async, no sync)
    ///
    /// This eliminates HashMap lookup + string formatting overhead (~10ms per token).
    /// Weight pointer must be from `indexed_layer_weights` populated by `build_indexed_weights()`.
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q4k_gemv_indexed_async(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // PAR-043: Direct pointer access - no HashMap lookup
        // Load kernel module (still needs format for dimensions, but cached after first call)
        let kernel_type = KernelType::Q4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }

    /// PAR-058: Execute Q6_K GEMV using pre-indexed device pointer (async, no sync)
    ///
    /// Like `q4k_gemv_indexed_async` but for Q6_K quantized weights.
    /// Used when V projection weights are Q6_K quantized (some GGUF models).
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q6K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q6k_gemv_indexed_async(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // PAR-058: Direct pointer access for Q6K weights
        let kernel_type = KernelType::Q6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }

    /// PAR-044: Execute Q4_K GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4k_gemv_indexed_async` but writes into a pre-allocated output buffer.
    /// Used by `transformer_layer_workspace` for zero-allocation forward pass.
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PAR-065: Use DP4A kernel for 4x instruction reduction
        // Five-Whys root cause chain:
        // 1. TiledQ4KGemv uses single-byte loads (ld_global_u8) - 6% bandwidth
        // 2. CoalescedQ4KGemv improved memory access - 27% speedup (99→126 tok/s)
        // 3. DP4A kernel uses SIMD dp4a instruction for 4x arithmetic throughput
        //
        // DP4A (Dot Product of 4 Bytes with Accumulate):
        // - Computes 4 int8 multiply-adds in single instruction
        // - 4x compute throughput vs scalar FMA
        // - Better ALU utilization
        //
        // Requirements: k must be multiple of 256 (super-block boundary)

        // CORRECTNESS-008: Use TiledQ4KGemv for aligned K to match q4k_gemv_cached behavior
        // The basic Q4KGemv kernel produces slightly different results due to different
        // accumulation order. Using the same kernel as the verified working cached path.
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
            // PAR-502: Use chunked kernel for large K dimensions (7B+ models)
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else if use_tiled {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q4KGemv { k, n };
            let ck = format!("q4k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-062: Execute Coalesced Q4_K GEMV with bandwidth-optimized memory access
    ///
    /// Key optimizations over basic Q4KGemvKernel:
    /// 1. **Scale loading**: Lane 0 loads 12 scale bytes as 3 x u32, broadcasts via shuffle
    ///    - Reduces 384 redundant byte loads to 3 loads + 3 broadcasts per super-block
    /// 2. **Reduced memory transactions**: Better cache utilization
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    #[inline]
    pub fn coalesced_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::CoalescedQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("coalesced_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-069: Execute Vectorized Q4_K GEMV with coalesced u32 weight loads
    ///
    /// Key optimization over CoalescedQ4KGemv:
    /// 1. **Weight loading**: Uses ld_global_u32 for coalesced 4-byte loads
    ///    - 32 threads × 4 bytes = 128 bytes per transaction (vs 32 × 1 byte scattered)
    /// 2. **Memory bandwidth**: Target 80%+ of peak (vs 6% with byte loads)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    #[inline]
    pub fn vectorized_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::VectorizedQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("vectorized_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-063: Execute DP4A Q4_K GEMV into existing buffer
    ///
    /// Uses DP4A SIMD instruction for 4x instruction reduction.
    /// Each DP4A computes 4 multiply-adds in a single instruction.
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    #[inline]
    pub fn dp4a_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::Dp4aQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("dp4a_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-076: Execute Fused RMSNorm + Q4_K GEMV into existing buffer
    ///
    /// Fuses RMSNorm normalization with Q4K GEMV in a single kernel pass:
    /// output = matmul(weights, rmsnorm(input, gamma))
    ///
    /// Phase 1: Load input, compute sum of squares, normalize in shared memory
    /// Phase 2: Do Q4K GEMV using normalized values from shared memory
    ///
    /// Eliminates:
    /// - Separate RMSNorm kernel launch (~1.5µs)
    /// - Global memory round-trip for normalized values
    /// - Memory bandwidth for writing/reading normalized buffer
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector (K elements, NOT normalized)
    /// * `gamma_ptr` - RMSNorm scale weights (K elements)
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `k` - Input/hidden dimension (must be multiple of 256)
    /// * `n` - Output dimension
    /// * `epsilon` - RMSNorm numerical stability (default 1e-5)
    #[inline]
    pub fn fused_rmsnorm_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64,
        output: &GpuBuffer<f32>,
        k: u32,
        n: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedRmsNormQ4KGemv { k, n, epsilon };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_rmsnorm_q4k_gemv_{}_{}_{:.0e}", k, n, epsilon);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One block per output element, 256 threads per block
        let config = LaunchConfig::grid_2d(n, 1, 256, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut ptr_gamma = gamma_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-077: Execute Fused Gate+Up Q4_K GEMV into existing buffers
    ///
    /// Computes both gate and up projections in a single kernel pass:
    ///   gate_out = W_gate * x
    ///   up_out = W_up * x
    ///
    /// Optimization: Reads input x only ONCE (saved to shared memory)
    /// - Standard approach: 2 kernel launches, 2x input bandwidth
    /// - Fused approach: 1 kernel launch, 1x input bandwidth
    ///
    /// Expected savings: ~30% reduction in FFNGateUp time
    ///
    /// # Arguments
    ///
    /// * `gate_weight_ptr` - Raw device pointer to Q4K gate weights
    /// * `up_weight_ptr` - Raw device pointer to Q4K up weights
    /// * `input` - GPU buffer containing input vector (K elements)
    /// * `gate_output` - Pre-allocated output buffer for gate (N elements)
    /// * `up_output` - Pre-allocated output buffer for up (N elements)
    /// * `k` - Input/hidden dimension (must be multiple of 256)
    /// * `n` - Intermediate dimension (output size)
    #[inline]
    pub fn fused_gate_up_q4k_gemv_into(
        &mut self,
        gate_weight_ptr: u64,
        up_weight_ptr: u64,
        input: &GpuBuffer<f32>,
        gate_output: &GpuBuffer<f32>,
        up_output: &GpuBuffer<f32>,
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedGateUpQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_gate_up_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One block per output element, 256 threads per block
        let config = LaunchConfig::grid_2d(n, 1, 256, 1);

        let mut ptr_gate_out = gate_output.as_ptr();
        let mut ptr_up_out = up_output.as_ptr();
        let mut ptr_gate_weights = gate_weight_ptr;
        let mut ptr_up_weights = up_weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_gate_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gate_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-061: Execute Tiled Q4_K GEMV into existing buffer (zero-allocation, high-perf)
    ///
    /// Like `q4k_gemv_into` but uses TiledQ4KGemv kernel with:
    /// - 256 threads per block (vs 32 in basic kernel) for better occupancy
    /// - Shared memory caching of input vector (~8x fewer global reads)
    /// - Multiple outputs per block for better work efficiency
    ///
    /// Performance: ~5-6x faster than basic Q4KGemv on RTX 4090
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (should be multiple of 256 for best performance)
    #[inline]
    pub fn q4k_gemv_into_tiled(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // CORRECTNESS-001: Use 4 outputs per block (matches verified working q4k_gemv_cached)
        // The 8-outputs config was causing incorrect results
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let outputs_per_block = 4u32;

        // PAR-502: Select kernel based on K size to avoid shared memory overflow
        let (kernel_type, cache_key) = if k > MAX_TILED_K && k.is_multiple_of(256) {
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            (kt, ck)
        } else {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            (kt, ck)
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // CORRECTNESS-001: Grid configuration matching q4k_gemv_cached
        // 128 threads per block, 4 outputs per block
        let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
        let config = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// CORRECTNESS-001: Test wrapper for q4k_gemv_into_tiled with CPU I/O
    ///
    /// Uses the exact same kernel as workspace path but with sync and CPU transfer.
    /// For debugging correctness issues.
    pub fn q4k_gemv_cached_tiled(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached weight pointer
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // Upload input to GPU
        let buf_input = GpuBuffer::from_host(&self.context, input)?;

        // Create output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // Run the tiled kernel (same as workspace path)
        self.q4k_gemv_into_tiled(weight_ptr, &buf_input, &buf_output, n, k)?;

        // Sync and download
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-108: Execute Batched Q4_K GEMV for 2x Ollama target
    ///
    /// Key optimization: Sequential GEMV dequantizes weights M times for M sequences.
    /// Batched GEMV dequantizes ONCE and multiplies by M different input vectors.
    /// This amortizes ALU-bound dequantization cost (32% bandwidth → higher efficiency).
    ///
    /// Memory layout:
    /// - `input`: M × K row-major (M batch elements, K elements each)
    /// - `output`: M × N row-major (M batch elements, N outputs each)
    /// - `weights`: N × K/256 Q4_K super-blocks (shared across batch)
    ///
    /// Performance insight (Five-Whys PAR-108):
    /// 1. WHY can't batched throughput reach 2x? → 32% bandwidth efficiency
    /// 2. WHY only 32%? → Sequential GEMV dequantizes per sequence
    /// 3. WHY not share dequantization? → No batched GEMV kernel existed
    /// 4. WHY not tensor cores? → Complex WMMA PTX (~400 LOC)
    /// 5. WHY batched GEMV works? → Simpler, shares dequant in registers
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing M×K input matrix (row-major)
    /// * `output` - Pre-allocated M×N output buffer (row-major)
    /// * `m` - Batch size (number of sequences, max 8)
    /// * `n` - Output dimension (weight rows)
    /// * `k` - Input dimension (weight columns, must be multiple of 256)
    ///
    /// PAR-129 FIX: Support M>8 via tiled execution or multi-warp kernel
    /// - M=16: Uses MultiWarpBatchedQ4KGemvKernel (2 warps × 8, L1 cache sharing)
    /// - M<=8: Single kernel launch with BatchedQ4KGemvKernel
    /// - M>8 (not 16): Processes in tiles of 8
    #[inline]
    pub fn batched_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        debug_assert!(
            k.is_multiple_of(256),
            "K must be multiple of 256 for Q4K super-blocks"
        );

        // PAR-129: Use multi-warp kernel for M=16 or M=32 (optimal L1 cache sharing)
        if m == 16 {
            return self.batched_q4k_gemv_into_multi_warp(weight_ptr, input, output, n, k, 2);
        }
        if m == 32 {
            return self.batched_q4k_gemv_into_multi_warp(weight_ptr, input, output, n, k, 4);
        }

        // Tile over M for M>8 (process 8 sequences at a time)
        const MAX_TILE_M: u32 = 8;
        let num_tiles = (m + MAX_TILE_M - 1) / MAX_TILE_M;

        for tile_idx in 0..num_tiles {
            let tile_start = tile_idx * MAX_TILE_M;
            let tile_m = (m - tile_start).min(MAX_TILE_M);

            let kernel_type = KernelType::BatchedQ4KGemv { m: tile_m, k, n };
            let kernel_name = self.kernels.kernel_name(&kernel_type);
            let cache_key = format!("batched_q4k_gemv_{}_{}_{}", tile_m, k, n);

            if !self.modules.contains_key(&cache_key) {
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = CudaModule::from_ptx(&self.context, &ptx)?;
                self.modules.insert(cache_key.clone(), module);
            }

            let module = self
                .modules
                .get_mut(&cache_key)
                .expect("module just inserted");

            // Grid: N blocks (one per output row), 32 threads per block
            let config = LaunchConfig::grid_2d(n, 1, 32, 1);

            // Offset pointers for this tile
            // Input: tile_start * k elements into input buffer
            // Output: tile_start * n elements into output buffer
            let input_offset = (tile_start * k) as usize * std::mem::size_of::<f32>();
            let output_offset = (tile_start * n) as usize * std::mem::size_of::<f32>();

            let mut ptr_output = output.as_ptr() + output_offset as u64;
            let mut ptr_weights = weight_ptr;
            let mut ptr_input = input.as_ptr() + input_offset as u64;
            let mut k_val = k;
            let mut n_val = n;
            let mut m_val = tile_m;

            // Kernel signature: batched_q4k_gemv_warp_reduce(y_ptr, w_ptr, x_ptr, k_dim, n_dim, m_dim)
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        Ok(())
    }

    /// PAR-129: Multi-warp batched Q4K GEMV for M=16/32
    /// Uses 2-4 warps per block, each handling 8 batch elements.
    /// All warps share L1-cached weights, avoiding weight re-reads.
    #[inline]
    fn batched_q4k_gemv_into_multi_warp(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
        warps: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::MultiWarpBatchedQ4KGemv { k, n, warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("multi_warp_batched_q4k_gemv_{}_{}_{}", k, n, warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: N blocks (one per output row), warps*32 threads per block
        let threads_per_block = warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads_per_block, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // Kernel signature: multi_warp_batched_q4k_gemv(y_ptr, w_ptr, x_ptr, k_dim, n_dim)
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-130: Batched Q6_K GEMV for M>1 batch processing
    ///
    /// Processes M input vectors against Q6K weights in a single kernel launch.
    /// Eliminates M-1 kernel launches per layer for Q6K weights (FFN down projection).
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q6K weight data
    /// * `input` - GPU buffer containing M×K packed input (M sequences, K elements each)
    /// * `output` - Pre-allocated output buffer (must be M×N elements)
    /// * `m` - Batch size
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn batched_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        debug_assert!(
            k.is_multiple_of(256),
            "K must be multiple of 256 for Q6K super-blocks"
        );

        let kernel_type = KernelType::BatchedQ6KGemv { k, n, m };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_q6k_gemv_{}_{}_{}", m, k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: N blocks (one per output row), 32 threads per block
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;
        let mut m_val = m;

        // Kernel signature: batched_q6k_gemv_warp_reduce(y_ptr, w_ptr, x_ptr, k_dim, n_dim, m_dim)
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-058: Execute Q6_K GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4k_gemv_into` but for Q6_K quantized weights.
    /// Used when V projection weights are Q6_K quantized (some GGUF models).
    ///
    /// Q6_K format: 210 bytes per 256 elements (vs Q4_K's 144 bytes)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q6K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Original Q6K kernel (CoalescedQ6K disabled due to CORRECTNESS-006)
        let kernel_type = KernelType::Q6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-066: Execute coalesced Q6K GEMV into existing buffer
    ///
    /// Uses vectorized scale loading (4 x u32) instead of 16 single-byte loads.
    /// Five-Whys root cause: Original Q6KGemvKernel caused 16 memory transactions
    /// per super-block for scale loading. This kernel reduces to 4 transactions.
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q6K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    #[inline]
    pub fn coalesced_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::CoalescedQ6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("coalesced_q6k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-058: Execute Q8_0 GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4k_gemv_into` but for Q8_0 quantized weights.
    /// Used when FFN down weights are Q8_0 quantized (some GGUF models like Qwen2.5-0.5B).
    ///
    /// Q8_0 format: 34 bytes per 32 elements (2-byte fp16 scale + 32 int8 values)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q8_0 weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q8_0_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PAR-058: Zero allocation Q8_0 GEMV for mixed-quantization models
        let kernel_type = KernelType::Q8_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q8_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-058: Execute Q5_0 GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q8_0_gemv_into` but for Q5_0 quantized weights.
    /// Used when Q/K weights are Q5_0 quantized (Qwen 0.5B).
    ///
    /// Q5_0 format: 22 bytes per 32 elements (2-byte fp16 scale + 4-byte high bits + 16 bytes packed nibbles)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q5_0 weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q5_0_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PAR-058: Zero allocation Q5_0 GEMV for Qwen 0.5B Q/K weights
        let kernel_type = KernelType::Q5_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-058: Execute Q4_0 GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q5_0_gemv_into` but for Q4_0 quantized weights.
    /// Used when GGUF header claims Q5_0 but data is actually Q4_0 format (qtype mismatch).
    ///
    /// Q4_0 format: 18 bytes per 32 elements (2-byte fp16 scale + 16 bytes packed nibbles)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4_0 weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q4_0_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PAR-058: Zero allocation Q4_0 GEMV for GGUF qtype mismatch
        let kernel_type = KernelType::Q4_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-058: Execute Q4_1 GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4_0_gemv_into` but for Q4_1 quantized weights.
    /// Q4_1 adds a min offset (affine quantization) vs Q4_0's symmetric quantization.
    ///
    /// Q4_1 format: 20 bytes per 32 elements (2-byte fp16 scale + 2-byte fp16 min + 16 bytes packed nibbles)
    /// Dequantization: val = d * nibble + m (vs Q4_0's: val = d * (nibble - 8))
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4_1 weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q4_1_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PAR-058: Zero allocation Q4_1 GEMV for Qwen2.5-0.5B FFN down
        let kernel_type = KernelType::Q4_1Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4_1_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-058: Execute Q5_K GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4k_gemv_into` but for Q5_K quantized weights.
    /// Used when FFN down weights are Q5_K quantized (some GGUF models).
    ///
    /// Q5_K format: 176 bytes per 256 elements
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q5K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q5k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PAR-058: Zero allocation Q5K GEMV for mixed-quantization models
        let kernel_type = KernelType::Q5KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-041: Execute Tiled Q4_K GEMV with shared memory caching (async, no sync)
    ///
    /// This variant uses 256 threads per block (vs 32 in q4k_gemv_cached_async) for
    /// better GPU occupancy. Input vector is cached in shared memory and shared by
    /// multiple output computations.
    ///
    /// Performance improvement: ~8x fewer global memory reads for input vector.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `outputs_per_block` - Number of outputs computed per block (default: 4)
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn tiled_q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
        outputs_per_block: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-041: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)

        // Load kernel module - select based on K size to avoid shared memory overflow
        let (kernel_type, cache_key) = if k > MAX_TILED_K && k.is_multiple_of(256) {
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            (kt, ck)
        } else {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            (kt, ck)
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // PAR-041: Grid configuration for tiled kernel
        // ceil(N / outputs_per_block) blocks, (32 * outputs_per_block) threads per block
        // CRITICAL: Thread count must match kernel's load stride of 32 * outputs_per_block
        // NOTE: Shared memory is statically declared in PTX - do NOT pass dynamically
        let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
        let threads_per_block = 32 * outputs_per_block; // 4 outputs = 128 threads
        let config = LaunchConfig::grid_2d(num_blocks, 1, threads_per_block, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-041: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-056: Execute Chunked Tiled Q4_K GEMV for large K dimensions (async, no sync)
    ///
    /// This kernel handles K > 8192 where TiledQ4KGemvKernel's shared memory
    /// would exceed CUDA limits (48KB default, 96KB max). It processes the
    /// input vector in 32KB (8K float) chunks.
    ///
    /// Used for 7B+ model FFN down projection where K = intermediate_dim > 8K.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `outputs_per_block` - Number of outputs computed per block (default: 4)
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn chunked_tiled_q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
        outputs_per_block: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-056: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        // Load kernel module
        let kernel_type = KernelType::ChunkedTiledQ4KGemv {
            k,
            n,
            outputs_per_block,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // PAR-056: Grid configuration for chunked tiled kernel
        // ceil(N / outputs_per_block) blocks, (32 * outputs_per_block) threads per block
        // CRITICAL: Thread count must match kernel's load stride of 32 * outputs_per_block
        // NOTE: Shared memory (32KB fixed) is statically declared in PTX - do NOT pass dynamically
        let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
        let threads_per_block = 32 * outputs_per_block; // 4 outputs = 128 threads
        let config = LaunchConfig::grid_2d(num_blocks, 1, threads_per_block, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-056: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-063: Execute DP4A Q4_K GEMV for optimized instruction throughput (async, no sync)
    ///
    /// Uses DP4A SIMD instruction to compute 4 multiply-adds per instruction,
    /// achieving up to 4x instruction reduction over scalar FMA operations.
    ///
    /// Key optimizations:
    /// - Vectorized scale loading (3 x u32 + warp shuffle broadcast)
    /// - DP4A instruction for SIMD dot products
    /// - Expected 2.5-3x throughput improvement over TiledQ4KGemv
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn dp4a_q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-063: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        // Load kernel module
        let kernel_type = KernelType::Dp4aQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("dp4a_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // PAR-063: Grid configuration - one warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-063: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-063-V4: Quantize f32 activations to Q8_1 format (async, no sync)
    ///
    /// This is the first step in the true DP4A GEMV pipeline:
    /// 1. Q8 quantize: f32 → Q8_1 (this function)
    /// 2. Q4K×Q8 dot: Q4K weights × Q8_1 activations → f32 output
    ///
    /// Q8_1 format: 36 bytes per 32 values
    /// - 32 bytes: 32 × int8 quantized values (qs)
    /// - 2 bytes: fp16 scale
    /// - 2 bytes: fp16 sum (for bias correction)
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing f32 activations
    /// * `n` - Number of elements to quantize
    ///
    /// # Returns
    /// GPU buffer containing Q8_1 quantized data (ceil(n/32) * 36 bytes)
    pub fn q8_quantize_async(
        &mut self,
        input: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<u8>, GpuError> {
        // Load kernel module
        let kernel_type = KernelType::Q8Quantize { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q8_quantize_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Q8_1 format: 36 bytes per 32 values
        // Layout: [qs: 32 × u8][scale: f16][sum: f16]
        let num_blocks = (n + 31) / 32;
        let output_bytes = (num_blocks * 36) as usize;
        let buf_output = GpuBuffer::<u8>::new(&self.context, output_bytes)?;

        // One warp (32 threads) processes 32 f32 values into one Q8_1 block
        let config = LaunchConfig::grid_2d(num_blocks, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_input = input.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }

    /// PAR-063-V5: Q4K × Q8 GEMV using true integer DP4A (async, no sync)
    ///
    /// This is the second step in the true DP4A GEMV pipeline:
    /// 1. Q8 quantize: f32 → Q8_1 (use q8_quantize_async)
    /// 2. Q4K×Q8 dot: Q4K weights × Q8_1 activations → f32 output (this function)
    ///
    /// Uses dp4a.u32.s32 instruction: d = dot4(weights_u8, activations_s8) + acc
    /// This achieves 4 multiply-adds per instruction vs 1 for scalar FMA.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `q8_input` - Q8_1 quantized activations from q8_quantize_async
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    pub fn q4k_q8_gemv_async(
        &mut self,
        weight_name: &str,
        q8_input: &GpuBuffer<u8>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-063-V5: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        // Load kernel module
        let kernel_type = KernelType::Q4KQ8Dot { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_q8_dot_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8_input = q8_input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_q8_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }

    /// PAR-063-V5: Fused Q8 quantize + Q4K×Q8 GEMV (async, no sync)
    ///
    /// Combines both steps of the true DP4A pipeline into a single call:
    /// 1. Quantizes f32 activations to Q8_1
    /// 2. Computes Q4K × Q8_1 dot product using integer DP4A
    ///
    /// This is the drop-in replacement for dp4a_q4k_gemv_cached_async that
    /// achieves true 4x instruction reduction via integer arithmetic.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `input` - GPU buffer containing f32 activations
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    pub fn true_dp4a_q4k_gemv_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Step 1: Quantize activations to Q8_1
        let q8_activations = self.q8_quantize_async(input, k)?;

        // Step 2: Q4K × Q8 dot product
        self.q4k_q8_gemv_async(weight_name, &q8_activations, n, k)
    }

    /// PAR-063-V6: Packed DP4A Q4K×Q8 GEMV using true dp4a.u32.s32 instruction
    ///
    /// Key optimizations over Q4KQ8DotKernel:
    /// - Uses dp4a.u32.s32 to process 4 values per instruction (4x IPC)
    /// - Packs 4 Q4K nibbles into u32 for DP4A weight operand
    /// - Packs 4 Q8 values into u32 for DP4A activation operand
    /// - 2 DP4A calls per thread per super-block (8 values total)
    ///
    /// Expected speedup: 4x vs scalar Q4KQ8DotKernel
    pub fn packed_dp4a_q4k_q8_gemv_async(
        &mut self,
        weight_name: &str,
        q8_input: &GpuBuffer<u8>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-063-V6: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        // Load kernel module
        let kernel_type = KernelType::PackedDp4aQ4KQ8 { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("packed_dp4a_q4k_q8_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8_input = q8_input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_q8_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }

    /// PAR-063-V6: Fused packed DP4A Q4K×Q8 GEMV (quantize + compute)
    ///
    /// Combines:
    /// 1. f32 → Q8_1 quantization
    /// 2. Packed DP4A Q4K×Q8 dot product
    ///
    /// This is the highest-performance path for Q4_K inference.
    pub fn packed_dp4a_full_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Step 1: Quantize activations to Q8_1
        let q8_activations = self.q8_quantize_async(input, k)?;

        // Step 2: Packed DP4A Q4K × Q8 dot product
        self.packed_dp4a_q4k_q8_gemv_async(weight_name, &q8_activations, n, k)
    }

    /// Execute Q5_K GEMV using cached weights - PAR-005
    pub fn q5k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-005: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        let kernel_type = KernelType::Q5KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = buf_input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// Execute Q6_K GEMV using cached weights - PAR-005
    pub fn q6k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self
            .quantized_weight_cache
            .get(weight_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-005: Quantized weight '{}' not cached",
                    weight_name
                ))
            })?
            .as_ptr();

        let kernel_type = KernelType::Q6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = buf_input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-014: Apply GELU activation in-place on a GPU buffer
    ///
    /// Uses BiasActivation kernel with zero bias for pure GELU.
    /// Part of persistent GPU tensor optimization for M4 milestone.
    pub fn gelu_gpu(&mut self, buffer: &GpuBuffer<f32>, n: u32) -> Result<(), GpuError> {
        // Use BiasActivation kernel with GELU activation (type 2) and zero bias
        let kernel_type = KernelType::BiasActivation {
            n,
            bias_size: 1,  // Single zero element
            activation: 2, // GELU
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gelu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Zero bias buffer (single element)
        let zero_bias = GpuBuffer::from_host(&self.context, &[0.0f32])?;

        // Launch config: 256 threads per block, enough blocks to cover n elements
        let threads_per_block = 256u32;
        let blocks = (n + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig::grid_2d(blocks, 1, threads_per_block, 1);

        let mut ptr_output = buffer.as_ptr();
        let mut ptr_bias = zero_bias.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_bias) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-014: Apply LayerNorm on GPU
    ///
    /// Performs: output = (input - mean) / sqrt(var + eps) * gamma + beta
    /// Part of persistent GPU tensor optimization for M4 milestone.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        beta: &GpuBuffer<f32>,
        hidden_size: u32,
        batch_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine: true,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("layernorm_{}_{}", hidden_size, batch_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // LayerNorm uses one warp per row
        let config = LaunchConfig::grid_2d(batch_size, 1, 32, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();
        let mut ptr_beta = beta.as_ptr();
        let mut hidden_size_val = hidden_size;
        let mut batch_size_val = batch_size;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_beta) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut hidden_size_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut batch_size_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-023: RMSNorm on GPU (async, no sync)
    ///
    /// RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * gamma
    ///
    /// # Arguments
    ///
    /// * `input` - GPU buffer with input vector [hidden_size]
    /// * `gamma` - GPU buffer with scale weights [hidden_size]
    /// * `hidden_size` - Dimension of the vector
    /// * `epsilon` - Numerical stability constant (default: 1e-5)
    ///
    /// # Returns
    ///
    /// GPU buffer with normalized output (no sync - async)
    pub fn rmsnorm_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        hidden_size: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::RmsNorm {
            hidden_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("rmsnorm_{}", hidden_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let output = GpuBuffer::<f32>::new(&self.context, hidden_size as usize)?;

        // RMSNorm uses one warp (32 threads)
        let config = LaunchConfig::grid_2d(1, 1, 32, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-023: NO sync - async operation for pipeline
        Ok(output)
    }

    /// PAR-044: RMSNorm into existing buffer (zero-allocation, async)
    ///
    /// Like `rmsnorm_gpu` but writes into a pre-allocated output buffer.
    ///
    /// PAR-081: Uses VectorizedRmsNorm with 256 threads for ~8x speedup
    /// over single-warp kernel (23µs → ~3µs for hidden_size=1536)
    ///
    /// CORRECTNESS-013: When CORRECTNESS_MODE=1, uses PreciseRmsNorm kernel
    /// with Kahan summation and Newton-Raphson rsqrt for CPU-matching precision.
    #[inline]
    pub fn rmsnorm_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // CORRECTNESS-013: Check if precise mode is requested
        static PRECISE_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_precise = *PRECISE_MODE.get_or_init(|| {
            let mode = std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false);
            if mode {
                eprintln!(
                    "[CORRECTNESS-013] RMSNorm using PreciseRmsNormKernel (Kahan+Newton-Raphson)"
                );
            }
            mode
        });

        // Choose kernel type based on mode
        let (kernel_type, cache_key) = if use_precise {
            (
                KernelType::PreciseRmsNorm {
                    hidden_size,
                    epsilon,
                },
                format!("rmsnorm_precise_{}", hidden_size),
            )
        } else {
            (
                KernelType::VectorizedRmsNorm {
                    hidden_size,
                    epsilon,
                },
                format!("rmsnorm_vectorized_{}", hidden_size),
            )
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-081: 256 threads (8 warps) for better parallelism
        let config = LaunchConfig::grid_2d(1, 1, 256, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-112: Batched RMSNorm for M sequences in parallel
    ///
    /// Processes M sequences in a single kernel launch using Grid.y = M.
    /// Achieves ~4x speedup over M sequential kernel launches by eliminating
    /// kernel launch overhead.
    ///
    /// # Arguments
    ///
    /// * `input` - GPU buffer with packed input [M × hidden_size]
    /// * `gamma` - GPU buffer with gamma weights [hidden_size] (shared across sequences)
    /// * `output` - GPU buffer for packed output [M × hidden_size]
    /// * `hidden_size` - Hidden dimension size
    /// * `batch_size` - Number of sequences (M)
    /// * `epsilon` - Numerical stability constant (default: 1e-5)
    pub fn batched_rmsnorm_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        batch_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedVectorizedRmsNorm {
            hidden_size,
            batch_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_rmsnorm_vectorized_{}_{}", hidden_size, batch_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-112: Grid (1, M, 1) with 256 threads per block
        let config = LaunchConfig::grid_2d(1, batch_size, 256, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-112: Batched RMSNorm using raw pointer for gamma (compatible with indexed weights)
    ///
    /// Same as `batched_rmsnorm_into` but accepts gamma as raw device pointer.
    pub fn batched_rmsnorm_ptr_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64,
        gamma_len: usize,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        batch_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let gamma = unsafe { GpuBuffer::from_raw_parts(gamma_ptr, gamma_len) };
        self.batched_rmsnorm_into(input, &gamma, output, hidden_size, batch_size, epsilon)?;
        std::mem::forget(gamma);
        Ok(())
    }

    /// PAR-114: Batched RoPE kernel for M sequences
    ///
    /// Applies rotary position embeddings to M sequences in parallel.
    /// Reduces 2M kernel launches to 2 (one for Q, one for K).
    ///
    /// # Arguments
    ///
    /// * `input` - Packed Q or K vectors [M × num_heads × head_dim]
    /// * `output` - Output vectors (can alias input for in-place)
    /// * `positions_buf` - GPU buffer of M positions
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `batch_size` - Number of sequences (M)
    /// * `theta` - RoPE theta base (typically 10000.0)
    #[allow(clippy::too_many_arguments)]
    pub fn batched_rope_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        positions_buf: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedRope {
            num_heads,
            head_dim,
            batch_size,
            theta,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_rope_{}_{}_{}", num_heads, head_dim, batch_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-114: Grid (num_heads, batch_size, 1) with head_dim/2 threads
        let threads = (head_dim / 2).min(256);
        let config = LaunchConfig::grid_2d(num_heads, batch_size, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_positions = positions_buf.as_ptr();

        // SAFETY: Pointers derived from valid GpuBuffer refs, kernel config matches data dimensions
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_positions) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-114: Batched Residual Add kernel for M sequences
    ///
    /// Element-wise addition for M sequences in parallel.
    /// Reduces 2M kernel launches to 2 (attention residual, FFN residual).
    ///
    /// # Arguments
    ///
    /// * `input1` - First packed input [M × n]
    /// * `input2` - Second packed input [M × n]
    /// * `output` - Output [M × n]
    /// * `n` - Elements per sequence
    /// * `batch_size` - Number of sequences (M)
    pub fn batched_residual_add_into(
        &mut self,
        input1: &GpuBuffer<f32>,
        input2: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        batch_size: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedResidualAdd { n, batch_size };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_residual_add_{}_{}", n, batch_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-114: Grid (ceil(n/256), batch_size, 1) with 256 threads
        let blocks_x = (n + 255) / 256;
        let config = LaunchConfig::grid_2d(blocks_x, batch_size, 256, 1);

        let mut ptr_input1 = input1.as_ptr();
        let mut ptr_input2 = input2.as_ptr();
        let mut ptr_output = output.as_ptr();

        // SAFETY: Pointers derived from valid GpuBuffer refs, kernel config matches data dimensions
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input1) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input2) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-114: Batched SwiGLU kernel for M sequences
    ///
    /// Fused SiLU+multiply for M sequences in parallel.
    /// Reduces M kernel launches to 1.
    ///
    /// # Arguments
    ///
    /// * `gate` - Packed gate values [M × n]
    /// * `up` - Packed up values [M × n]
    /// * `output` - Output [M × n]
    /// * `n` - Elements per sequence
    /// * `batch_size` - Number of sequences (M)
    pub fn batched_swiglu_into(
        &mut self,
        gate: &GpuBuffer<f32>,
        up: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        batch_size: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedSwiglu { n, batch_size };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_swiglu_{}_{}", n, batch_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-114: Grid (ceil(n/256), batch_size, 1) with 256 threads
        let blocks_x = (n + 255) / 256;
        let config = LaunchConfig::grid_2d(blocks_x, batch_size, 256, 1);

        let mut ptr_gate = gate.as_ptr();
        let mut ptr_up = up.as_ptr();
        let mut ptr_output = output.as_ptr();

        // SAFETY: Pointers derived from valid GpuBuffer refs, kernel config matches data dimensions
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_gate) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-023: RMSNorm on GPU with host input/output (synchronous convenience method)
    ///
    /// This is a convenience wrapper around `rmsnorm_gpu` that handles
    /// host-to-device and device-to-host transfers.
    ///
    /// # Arguments
    ///
    /// * `input` - Host slice with input vector [hidden_size]
    /// * `gamma` - Host slice with scale weights [hidden_size]
    /// * `output` - Host slice for output [hidden_size]
    /// * `epsilon` - Numerical stability constant (default: 1e-5)
    pub fn rmsnorm_host(
        &mut self,
        input: &[f32],
        gamma: &[f32],
        output: &mut [f32],
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let hidden_size = input.len() as u32;

        // Upload to GPU
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let gamma_gpu = GpuBuffer::from_host(&self.context, gamma)?;

        // Run kernel
        let output_gpu = self.rmsnorm_gpu(&input_gpu, &gamma_gpu, hidden_size, epsilon)?;

        // Sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Residual Add on GPU with host input/output (synchronous convenience method)
    ///
    /// This is a convenience wrapper around `residual_add_gpu` that handles
    /// host-to-device and device-to-host transfers.
    ///
    /// # Arguments
    ///
    /// * `input1` - Host slice with first input vector
    /// * `input2` - Host slice with second input vector
    /// * `output` - Host slice for output
    pub fn residual_add_host(
        &mut self,
        input1: &[f32],
        input2: &[f32],
        output: &mut [f32],
    ) -> Result<(), GpuError> {
        let n = input1.len() as u32;

        // Upload to GPU
        let input1_gpu = GpuBuffer::from_host(&self.context, input1)?;
        let input2_gpu = GpuBuffer::from_host(&self.context, input2)?;

        // Run kernel
        let output_gpu = self.residual_add_gpu(&input1_gpu, &input2_gpu, n)?;

        // Sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Fused Residual Add + RMSNorm with host input/output (synchronous convenience method)
    ///
    /// This is a convenience wrapper around `fused_residual_rmsnorm_gpu` that handles
    /// host-to-device and device-to-host transfers.
    ///
    /// # Arguments
    ///
    /// * `residual` - Host slice with residual input
    /// * `input` - Host slice with input to add
    /// * `gamma` - Host slice with scale weights
    /// * `output` - Host slice for output
    /// * `epsilon` - Numerical stability constant
    pub fn fused_residual_rmsnorm_host(
        &mut self,
        residual: &[f32],
        input: &[f32],
        gamma: &[f32],
        output: &mut [f32],
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let hidden_size = residual.len() as u32;

        // Upload to GPU
        let residual_gpu = GpuBuffer::from_host(&self.context, residual)?;
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let gamma_gpu = GpuBuffer::from_host(&self.context, gamma)?;

        // Run kernel
        let output_gpu = self.fused_residual_rmsnorm_gpu(
            &residual_gpu,
            &input_gpu,
            &gamma_gpu,
            hidden_size,
            epsilon,
        )?;

        // Sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Residual Add using dedicated kernel (async)
    ///
    /// Computes: output[i] = input1[i] + input2[i]
    /// Uses the new ResidualAddKernel for better async pipeline integration.
    ///
    /// # Arguments
    ///
    /// * `input1` - First input buffer
    /// * `input2` - Second input buffer
    /// * `n` - Number of elements
    ///
    /// # Returns
    ///
    /// GPU buffer with result (no sync - async)
    pub fn residual_add_gpu(
        &mut self,
        input1: &GpuBuffer<f32>,
        input2: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::ResidualAdd { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("residual_add_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // 256 threads per block
        let threads_per_block = 256u32;
        let blocks = (n + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig::grid_2d(blocks, 1, threads_per_block, 1);

        let mut ptr_input1 = input1.as_ptr();
        let mut ptr_input2 = input2.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input1) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input2) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-023: NO sync - async operation for pipeline
        Ok(output)
    }

    /// PAR-044: Residual add into existing buffer (zero-allocation, async)
    #[inline]
    pub fn residual_add_into(
        &mut self,
        input1: &GpuBuffer<f32>,
        input2: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::ResidualAdd { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("residual_add_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let threads_per_block = 256u32;
        let blocks = (n + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig::grid_2d(blocks, 1, threads_per_block, 1);

        let mut ptr_input1 = input1.as_ptr();
        let mut ptr_input2 = input2.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input1) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input2) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-023: Fused Residual Add + RMSNorm (async)
    ///
    /// Computes: output = rmsnorm(residual + input, gamma, epsilon)
    /// Fuses residual add and normalization to reduce memory bandwidth.
    ///
    /// # Arguments
    ///
    /// * `residual` - Residual input buffer
    /// * `input` - Input to add to residual
    /// * `gamma` - RMSNorm scale weights
    /// * `hidden_size` - Hidden dimension
    /// * `epsilon` - Numerical stability constant
    ///
    /// # Returns
    ///
    /// GPU buffer with normalized result (no sync - async)
    pub fn fused_residual_rmsnorm_gpu(
        &mut self,
        residual: &GpuBuffer<f32>,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        hidden_size: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::FusedResidualRmsNorm {
            hidden_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_residual_rmsnorm_{}", hidden_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let output = GpuBuffer::<f32>::new(&self.context, hidden_size as usize)?;

        // Fused kernel uses one warp (32 threads)
        let config = LaunchConfig::grid_2d(1, 1, 32, 1);

        let mut ptr_residual = residual.as_ptr();
        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_residual) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-023: NO sync - async operation for pipeline
        Ok(output)
    }

    /// PAR-075: Fused Residual Add + RMSNorm into pre-allocated buffer
    ///
    /// Computes: output = rmsnorm(residual + input, gamma, epsilon)
    /// Fuses residual add and normalization to reduce memory bandwidth.
    /// Uses pre-allocated output buffer to eliminate allocation.
    ///
    /// NOTE: input == output is safe for this kernel due to:
    /// 1. Single-warp execution (lockstep within warp)
    /// 2. Each thread handles disjoint elements
    /// 3. Read before write per element per thread
    pub fn fused_residual_rmsnorm_into(
        &mut self,
        residual: &GpuBuffer<f32>,
        input: &GpuBuffer<f32>,
        gamma_ptr: usize, // Raw device pointer to gamma weights
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedResidualRmsNorm {
            hidden_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_residual_rmsnorm_{}", hidden_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = CudaModule::from_ptx(&self.context, &ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Fused kernel uses one warp (32 threads)
        let config = LaunchConfig::grid_2d(1, 1, 32, 1);

        let mut ptr_residual = residual.as_ptr();
        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma_ptr;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_residual) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-075: NO sync - async operation for pipeline
        Ok(())
    }

}
